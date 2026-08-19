// CPU unit tests for the pushpull *backward* ops.
//
// Exercises ff::cpu::{pull,push,count,grad}_backward through the full stack
// (dispatch -> impl -> kernels).
//
// The oracle is a central finite difference of the scalar loss
//
//     L(inp, grid) = < forward_op(inp, grid), ginp >
//
// whose analytic gradients are exactly what the backward ops return:
// `out` == dL/d(inp) and `gout` == dL/d(grid). Every element of `inp` and
// `grid` is differenced independently, so this is a full Jacobian check
// against the (independently tested) forward ops rather than a smoke test.
//
// Deliberate choices:
//   * float64 only -- a finite-difference oracle needs the headroom.
//   * `extrapolate = 1` so no sample is gated by the FOV test (which is a
//     step function, and therefore not finite-differentiable).
//   * grid coordinates are kept away from integer *and* half-integer
//     positions: those are the spline knots, where the interpolant's
//     higher derivatives jump and a central difference straddling one
//     would be meaningless. See `SAFE_1D` / `safe_coord`.
//   * the input volume and the sampled grid have *different* spatial
//     shapes wherever possible. Equal shapes make several stride arrays
//     coincide and hide indexing bugs (that is exactly how the
//     `push_backward` `stride_inp`/`stride_ginp` mix-up survived in
//     jitfields).
//
// Build (from fastfields-cpu-lib):
//   clang++ -std=c++11 -O2 -I. tests/test_pushpull_backward.cpp pushpull.cpp \
//           -o build/test_pushpull_backward
//   ./build/test_pushpull_backward

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <stdexcept>
#include "dlpack.h"
#include "pushpull.h"

namespace {

// Spline orders / bounds (mirror spline_t / bound_t enum values).
constexpr int8_t NEAREST = 0, LINEAR = 1, QUADRATIC = 2, CUBIC = 3;
constexpr int8_t DCT2 = 3, DST2 = 5;   // bound values

int g_failures = 0, g_checks = 0;

void check_close(double a, double b, const std::string & what,
                 double tol = 2e-5)
{
    ++g_checks;
    double diff  = std::fabs(a - b);
    double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    if (!(diff / scale <= tol)) {   // `!(<=)` also catches NaN
        ++g_failures;
        if (g_failures < 40)
            std::printf("  MISMATCH [%s]: got %.10g expected %.10g\n",
                        what.c_str(), a, b);
    }
}

int64_t numel(const std::vector<int64_t> & shape)
{
    int64_t n = 1;
    for (size_t i = 0; i < shape.size(); ++i) n *= shape[i];
    return n;
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t> & shape)
{
    std::vector<int64_t> s(shape.size());
    int64_t acc = 1;
    for (int64_t d = (int64_t)shape.size() - 1; d >= 0; --d) {
        s[d] = acc; acc *= shape[d];
    }
    return s;
}

// A float64 CPU tensor that owns its shape/stride/data storage. `dl()` hands
// out a DLTensor viewing them; the vectors are never resized afterwards, so
// the pointers stay valid.
struct Tensor {
    std::vector<int64_t> shape, stride;
    std::vector<double>  data;

    explicit Tensor(const std::vector<int64_t> & s)
        : shape(s), stride(contiguous_strides(s)), data(numel(s), 0.0) {}

    DLTensor dl()
    {
        DLTensor t;
        t.data               = static_cast<void*>(data.data());
        t.device.device_type = kDLCPU;
        t.device.device_id   = 0;
        t.ndim               = static_cast<int32_t>(shape.size());
        t.dtype.code         = static_cast<uint8_t>(kDLFloat);
        t.dtype.bits         = 64;
        t.dtype.lanes        = 1;
        t.shape              = shape.data();
        t.strides            = stride.data();
        t.byte_offset        = 0;
        return t;
    }

    void zero() { for (size_t i = 0; i < data.size(); ++i) data[i] = 0.0; }
};

double dot(const std::vector<double> & a, const std::vector<double> & b)
{
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) acc += a[i] * b[i];
    return acc;
}

void fill_random(std::vector<double> & v, std::mt19937 & rng)
{
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (size_t i = 0; i < v.size(); ++i) v[i] = d(rng);
}

// Coordinates that stay clear of the spline knots (integers, and
// half-integers for the even orders) by a wide margin compared to the
// finite-difference step.
const double SAFE_1D[8] = {0.27, 1.13, 1.81, 2.36, 2.69, 3.14, 3.88, 4.31};

double safe_coord(int64_t k, int64_t extent)
{
    // wrap through the table, then squeeze into (0.2, extent-1.2)
    double v = SAFE_1D[k % 8];
    double hi = (double)extent - 1.2;
    if (hi < 0.25) hi = 0.25;
    while (v > hi) v -= 1.0;
    if (v < 0.22) v = 0.22 + 0.13 * (double)(k % 3);
    return v;
}

/***********************************************************************
 *                    forward ops -> scalar loss                       *
 ***********************************************************************/

struct Opts { int8_t spline, bound, extrapolate; bool abs; };

// L = <pull(inp, grid), ginp>
double loss_pull(Tensor & inp, Tensor & grid, const Tensor & ginp,
                 const Opts & o)
{
    Tensor out(ginp.shape);
    DLTensor ot = out.dl(), it = inp.dl(), gt = grid.dl();
    ff::cpu::pull(ot, it, gt, o.spline, o.bound, o.extrapolate, 0);
    return dot(out.data, ginp.data);
}

// L = <push(inp, grid), ginp>   (ginp has the pushed volume's shape)
double loss_push(Tensor & inp, Tensor & grid, const Tensor & ginp,
                 const Opts & o)
{
    Tensor out(ginp.shape);   // zero-initialised: push accumulates
    DLTensor ot = out.dl(), it = inp.dl(), gt = grid.dl();
    ff::cpu::push(ot, it, gt, o.spline, o.bound, o.extrapolate, 0);
    return dot(out.data, ginp.data);
}

// L = <count(grid), ginp>
double loss_count(Tensor & grid, const Tensor & ginp, const Opts & o)
{
    Tensor out(ginp.shape);   // zero-initialised: count accumulates
    DLTensor ot = out.dl(), gt = grid.dl();
    ff::cpu::count(ot, gt, o.spline, o.bound, o.extrapolate, 0);
    return dot(out.data, ginp.data);
}

// L = <grad(inp, grid), ginp>
double loss_grad(Tensor & inp, Tensor & grid, const Tensor & ginp,
                 const Opts & o)
{
    Tensor out(ginp.shape);
    DLTensor ot = out.dl(), it = inp.dl(), gt = grid.dl();
    ff::cpu::grad(ot, it, gt, o.spline, o.bound, o.extrapolate, o.abs, 0);
    return dot(out.data, ginp.data);
}

/***********************************************************************
 *                       finite-difference oracle                      *
 ***********************************************************************/

// Central difference of `loss` wrt every element of `x`, compared against
// `analytic`. `h` is deliberately large-ish: the interpolants are smooth
// polynomials between knots, so truncation error is ~h^2 while the
// cancellation error is ~eps/h.
template <typename Loss>
void fd_check(std::vector<double> & x, const std::vector<double> & analytic,
              Loss loss, const std::string & what,
              double h = 1e-4, double tol = 2e-5)
{
    if (x.size() != analytic.size())
        throw std::runtime_error("fd_check: size mismatch for " + what);
    for (size_t i = 0; i < x.size(); ++i) {
        double x0 = x[i];
        x[i] = x0 + h; double lp = loss();
        x[i] = x0 - h; double lm = loss();
        x[i] = x0;
        check_close(analytic[i], (lp - lm) / (2.0 * h), what, tol);
    }
}

/***********************************************************************
 *                             test bodies                             *
 ***********************************************************************/

// One (ndim, inshape, outshape, C) x (order, bound) configuration.
struct Case {
    std::vector<int64_t> batch;     // leading batch dims
    std::vector<int64_t> inshape;   // field spatial shape
    std::vector<int64_t> outshape;  // sampled spatial shape
    int64_t              channels;
};

std::vector<int64_t> cat(const std::vector<int64_t> & a,
                         const std::vector<int64_t> & b, int64_t last)
{
    std::vector<int64_t> v(a);
    v.insert(v.end(), b.begin(), b.end());
    v.push_back(last);
    return v;
}

std::string tag(const char * op, const Case & c, const Opts & o)
{
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s d=%d order=%d bound=%d",
                  op, (int)c.inshape.size(), (int)o.spline, (int)o.bound);
    return std::string(buf);
}

Tensor make_grid(const Case & c, std::mt19937 & rng)
{
    const int64_t d = (int64_t)c.inshape.size();
    Tensor grid(cat(c.batch, c.outshape, d));
    std::uniform_int_distribution<int> jitter(0, 7);
    for (size_t i = 0; i < grid.data.size(); ++i) {
        int64_t axis = (int64_t)(i % (size_t)d);
        grid.data[i] = safe_coord((int64_t)i + jitter(rng), c.inshape[axis]);
    }
    return grid;
}

void test_pull_backward(const Case & c, const Opts & o, std::mt19937 & rng)
{
    const int64_t d = (int64_t)c.inshape.size();
    Tensor inp (cat(c.batch, c.inshape,  c.channels));
    Tensor ginp(cat(c.batch, c.outshape, c.channels));
    Tensor grid = make_grid(c, rng);
    fill_random(inp.data,  rng);
    fill_random(ginp.data, rng);

    Tensor out (inp.shape);              // dL/d(inp);  accumulated -> pre-zeroed
    Tensor gout(cat(c.batch, c.outshape, d));   // dL/d(grid)
    {
        DLTensor ot = out.dl(), got = gout.dl(), it = inp.dl(),
                 git = ginp.dl(), gt = grid.dl();
        ff::cpu::pull_backward(ot, got, it, git, gt,
                               o.spline, o.bound, o.extrapolate, 0);
    }

    const std::string t = tag("pull_backward", c, o);
    fd_check(inp.data,  out.data,
             [&]{ return loss_pull(inp, grid, ginp, o); }, t + " d/dinp");
    fd_check(grid.data, gout.data,
             [&]{ return loss_pull(inp, grid, ginp, o); }, t + " d/dgrid");
}

void test_push_backward(const Case & c, const Opts & o, std::mt19937 & rng)
{
    const int64_t d = (int64_t)c.inshape.size();
    // push: inp is grid-shaped, the pushed volume (and hence ginp) is
    // inshape-shaped.
    Tensor inp (cat(c.batch, c.outshape, c.channels));
    Tensor ginp(cat(c.batch, c.inshape,  c.channels));
    Tensor grid = make_grid(c, rng);
    fill_random(inp.data,  rng);
    fill_random(ginp.data, rng);

    Tensor out (inp.shape);                     // dL/d(inp), grid-shaped
    Tensor gout(cat(c.batch, c.outshape, d));   // dL/d(grid)
    {
        DLTensor ot = out.dl(), got = gout.dl(), it = inp.dl(),
                 git = ginp.dl(), gt = grid.dl();
        ff::cpu::push_backward(ot, got, it, git, gt,
                               o.spline, o.bound, o.extrapolate, 0);
    }

    const std::string t = tag("push_backward", c, o);
    fd_check(inp.data,  out.data,
             [&]{ return loss_push(inp, grid, ginp, o); }, t + " d/dinp");
    fd_check(grid.data, gout.data,
             [&]{ return loss_push(inp, grid, ginp, o); }, t + " d/dgrid");
}

void test_count_backward(const Case & c, const Opts & o, std::mt19937 & rng)
{
    const int64_t d = (int64_t)c.inshape.size();
    Tensor ginp(cat(c.batch, c.inshape, 1));
    Tensor grid = make_grid(c, rng);
    fill_random(ginp.data, rng);

    Tensor gout(cat(c.batch, c.outshape, d));
    {
        DLTensor got = gout.dl(), git = ginp.dl(), gt = grid.dl();
        ff::cpu::count_backward(got, git, gt,
                                o.spline, o.bound, o.extrapolate, 0);
    }

    const std::string t = tag("count_backward", c, o);
    fd_check(grid.data, gout.data,
             [&]{ return loss_count(grid, ginp, o); }, t + " d/dgrid");
}

void test_grad_backward(const Case & c, const Opts & o, std::mt19937 & rng)
{
    const int64_t d = (int64_t)c.inshape.size();
    Tensor inp(cat(c.batch, c.inshape, c.channels));
    // grad's output carries an extra trailing (D) axis.
    std::vector<int64_t> gshape = cat(c.batch, c.outshape, c.channels);
    gshape.push_back(d);
    Tensor ginp(gshape);
    Tensor grid = make_grid(c, rng);
    fill_random(inp.data,  rng);
    fill_random(ginp.data, rng);

    Tensor out (inp.shape);
    Tensor gout(cat(c.batch, c.outshape, d));
    {
        DLTensor ot = out.dl(), got = gout.dl(), it = inp.dl(),
                 git = ginp.dl(), gt = grid.dl();
        ff::cpu::grad_backward(ot, got, it, git, gt,
                               o.spline, o.bound, o.extrapolate, o.abs, 0);
    }

    const std::string t = tag("grad_backward", c, o);
    fd_check(inp.data,  out.data,
             [&]{ return loss_grad(inp, grid, ginp, o); }, t + " d/dinp");
    fd_check(grid.data, gout.data,
             [&]{ return loss_grad(inp, grid, ginp, o); }, t + " d/dgrid");
}

/***********************************************************************
 *                    non-differentiable-path checks                   *
 ***********************************************************************/

// Samples outside the field of view must get an exactly-zero grid gradient
// (the FOV test is a gate, not a smooth factor -- finite differences cannot
// see this, so it is asserted directly).
void test_out_of_fov_zero()
{
    const Opts o = {LINEAR, DCT2, /*extrapolate=*/0, false};
    Case c; c.inshape = {5}; c.outshape = {3}; c.channels = 1;

    Tensor inp (cat(c.batch, c.inshape,  c.channels));
    Tensor ginp(cat(c.batch, c.outshape, c.channels));
    Tensor grid(cat(c.batch, c.outshape, 1));
    for (size_t i = 0; i < inp.data.size(); ++i)  inp.data[i]  = 1.0 + (double)i;
    for (size_t i = 0; i < ginp.data.size(); ++i) ginp.data[i] = 1.0;
    grid.data[0] = -3.5;   // well outside
    grid.data[1] =  1.3;   // inside
    grid.data[2] = 11.0;   // well outside

    Tensor out(inp.shape), gout(grid.shape);
    DLTensor ot = out.dl(), got = gout.dl(), it = inp.dl(),
             git = ginp.dl(), gt = grid.dl();
    ff::cpu::pull_backward(ot, got, it, git, gt,
                           o.spline, o.bound, o.extrapolate, 0);
    check_close(gout.data[0], 0.0, "fov zero lo");
    check_close(gout.data[2], 0.0, "fov zero hi");
    ++g_checks;
    if (std::fabs(gout.data[1]) < 1e-12) {
        ++g_failures;
        std::printf("  MISMATCH [fov inside]: in-FOV gradient is zero\n");
    }
}

// `pull_backward`'s dL/dinp is a scatter: it accumulates into `out`. Calling
// it twice on a non-zeroed buffer must double the field gradient -- and must
// *not* double the grid gradient, which is overwritten.
void test_accumulation_semantics()
{
    const Opts o = {CUBIC, DCT2, 1, false};
    Case c; c.inshape = {6}; c.outshape = {4}; c.channels = 2;
    std::mt19937 rng(7);

    Tensor inp (cat(c.batch, c.inshape,  c.channels));
    Tensor ginp(cat(c.batch, c.outshape, c.channels));
    Tensor grid = make_grid(c, rng);
    fill_random(inp.data,  rng);
    fill_random(ginp.data, rng);

    Tensor out(inp.shape), gout(grid.shape);
    DLTensor ot = out.dl(), got = gout.dl(), it = inp.dl(),
             git = ginp.dl(), gt = grid.dl();
    ff::cpu::pull_backward(ot, got, it, git, gt, o.spline, o.bound, 1, 0);
    std::vector<double> once  = out.data;
    std::vector<double> gonce = gout.data;
    ff::cpu::pull_backward(ot, got, it, git, gt, o.spline, o.bound, 1, 0);
    for (size_t i = 0; i < once.size(); ++i)
        check_close(out.data[i], 2.0 * once[i], "pull_backward accumulates");
    for (size_t i = 0; i < gonce.size(); ++i)
        check_close(gout.data[i], gonce[i], "pull_backward gout overwrites");
}

// Non-contiguous inputs: a transposed view of the field must give the same
// answer as its contiguous copy. This is what makes the stride arrays load-
// bearing (with everything contiguous and equally shaped, several of them
// coincide and indexing bugs stay invisible).
void test_strided_field()
{
    const Opts o = {CUBIC, DCT2, 1, false};
    std::mt19937 rng(11);
    const int64_t H = 4, W = 5, M = 3, C = 2;

    // contiguous (H, W, C)
    Tensor inp({H, W, C});
    fill_random(inp.data, rng);
    Tensor ginp({M, M, C});
    fill_random(ginp.data, rng);
    Case c; c.inshape = {H, W}; c.outshape = {M, M}; c.channels = C;
    Tensor grid = make_grid(c, rng);

    Tensor out({H, W, C}), gout({M, M, 2});
    {
        DLTensor ot = out.dl(), got = gout.dl(), it = inp.dl(),
                 git = ginp.dl(), gt = grid.dl();
        ff::cpu::pull_backward(ot, got, it, git, gt, o.spline, o.bound, 1, 0);
    }

    // same field, stored (W, H, C) and viewed with permuted strides
    Tensor inpT({W, H, C});
    for (int64_t i = 0; i < H; ++i)
        for (int64_t j = 0; j < W; ++j)
            for (int64_t k = 0; k < C; ++k)
                inpT.data[(j * H + i) * C + k] = inp.data[(i * W + j) * C + k];
    Tensor outT({W, H, C});

    Tensor gout2({M, M, 2});
    {
        DLTensor it = inpT.dl();
        it.shape = nullptr; it.strides = nullptr;   // replaced below
        std::vector<int64_t> vshape  = {H, W, C};
        std::vector<int64_t> vstride = {C, H * C, 1};
        it.shape = vshape.data(); it.strides = vstride.data();
        DLTensor ot = outT.dl();
        ot.shape = vshape.data(); ot.strides = vstride.data();
        DLTensor got = gout2.dl(), git = ginp.dl(), gt = grid.dl();
        ff::cpu::pull_backward(ot, got, it, git, gt, o.spline, o.bound, 1, 0);
    }

    for (int64_t i = 0; i < H; ++i)
        for (int64_t j = 0; j < W; ++j)
            for (int64_t k = 0; k < C; ++k)
                check_close(outT.data[(j * H + i) * C + k],
                            out.data[(i * W + j) * C + k], "strided dL/dinp");
    for (size_t i = 0; i < gout.data.size(); ++i)
        check_close(gout2.data[i], gout.data[i], "strided dL/dgrid");
}

} // namespace

int main()
{
    std::printf("pushpull backward CPU tests\n");
    std::mt19937 rng(20260731);

    // Orders and bounds that the `test` target instantiates statically; the
    // remaining ones share the Dynamic path and are covered by the sweep in
    // test_pushpull.cpp.
    const int8_t orders[4] = {NEAREST, LINEAR, QUADRATIC, CUBIC};
    const int8_t bounds[2] = {DCT2, DST2};

    // Deliberately different in/out spatial shapes (see the header comment).
    std::vector<Case> cases;
    { Case c; c.inshape = {7};       c.outshape = {4};       c.channels = 2; cases.push_back(c); }
    { Case c; c.inshape = {4, 5};    c.outshape = {3, 2};    c.channels = 2; cases.push_back(c); }
    { Case c; c.inshape = {4, 4, 5}; c.outshape = {2, 3, 2}; c.channels = 1; cases.push_back(c); }
    // ... and one with a leading batch dim, to exercise the batch offsets.
    { Case c; c.batch = {2}; c.inshape = {5, 4}; c.outshape = {2, 3}; c.channels = 2;
      cases.push_back(c); }

    try {
        for (size_t ci = 0; ci < cases.size(); ++ci)
        for (int bi = 0; bi < 2; ++bi)
        for (int oi = 0; oi < 4; ++oi)
        {
            Opts o = {orders[oi], bounds[bi], 1, false};
            test_pull_backward (cases[ci], o, rng);
            test_push_backward (cases[ci], o, rng);
            test_count_backward(cases[ci], o, rng);
            test_grad_backward (cases[ci], o, rng);
        }

        test_out_of_fov_zero();
        test_accumulation_semantics();
        test_strided_field();
    } catch (const std::exception & e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 1;
    }

    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
