// CPU unit tests for the reg_field module.
//
// Exercises ff::cpu::field_matvec / field_diag against hand-written references:
//   * absolute-only matvec == per-channel elementwise scaling (exact)
//   * membrane-only matvec == membrane[c] * discrete negative Laplacian (exact,
//     with explicit Zero and DCT2 boundary references)
//   * diag == matvec on unit vectors in the interior
//
// Build (from fastfields-cpu-lib):
//   clang++ -std=c++11 -O2 -ferror-limit=5 -I. tests/test_reg_field.cpp reg_field.cpp -o build/test_reg_field
//   ./build/test_reg_field

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "dlpack.h"
#include "reg_field.h"
#include "posdef.h"

namespace {

enum { B_ZERO = 0, B_DCT2 = 3, B_DST2 = 5, B_DFT = 6 };

template <typename T>
DLTensor make_cpu_tensor(T* data, std::vector<int64_t>& shape,
                         std::vector<int64_t>& strides, uint8_t bits)
{
    DLTensor t;
    t.data               = static_cast<void*>(data);
    t.device.device_type = kDLCPU;
    t.device.device_id   = 0;
    t.ndim               = static_cast<int32_t>(shape.size());
    t.dtype.code         = static_cast<uint8_t>(kDLFloat);
    t.dtype.bits         = bits;
    t.dtype.lanes        = 1;
    t.shape              = shape.data();
    t.strides            = strides.data();
    t.byte_offset        = 0;
    return t;
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> s(shape.size());
    int64_t acc = 1;
    for (int64_t d = (int64_t)shape.size() - 1; d >= 0; --d) { s[d] = acc; acc *= shape[d]; }
    return s;
}

int g_failures = 0;
int g_checks   = 0;

void check_close(double a, double b, const char* what, double tol = 1e-5)
{
    ++g_checks;
    double diff = std::fabs(a - b);
    double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    if (diff / scale > tol) {
        ++g_failures;
        std::printf("  MISMATCH [%s]: got %.8g expected %.8g\n", what, a, b);
    }
}

double neigh_1d(const std::vector<double>& f, int64_t idx, int bound)
{
    int64_t n = (int64_t)f.size();
    if (idx >= 0 && idx < n) return f[idx];
    if (bound == B_ZERO) return 0.0;
    int64_t r = idx;
    if (r < 0)  r = -r - 1;
    if (r >= n) r = 2*n - r - 1;
    if (r < 0)  r = 0;
    if (r >= n) r = n - 1;
    return f[r];
}

// 1D field with C channels, contiguous (N, C).
template <typename scalar_t>
void run_1d(int64_t N, int64_t C, const std::vector<double>& absolute,
            const std::vector<double>& membrane, bool use_membrane,
            int bound, uint8_t bits)
{
    std::vector<int64_t> shape = {N, C};
    std::vector<int64_t> str   = contiguous_strides(shape);
    int64_t numel = N * C;
    std::vector<scalar_t> inp(numel), out(numel, scalar_t(-777));
    // per-channel field
    std::vector<std::vector<double>> f(C, std::vector<double>(N));
    for (int64_t x = 0; x < N; ++x)
        for (int64_t c = 0; c < C; ++c) {
            double v = std::sin(0.4 * x + 0.7 * c + 1.0) + 0.3 * x - 0.2 * c;
            inp[x * C + c] = (scalar_t)v;
            f[c][x] = v;
        }

    DLTensor tin  = make_cpu_tensor(inp.data(), shape, str, bits);
    DLTensor tout = make_cpu_tensor(out.data(), shape, str, bits);

    const double* mem = use_membrane ? membrane.data() : nullptr;
    ff::cpu::field_matvec(tout, tin, nullptr, absolute.data(), mem, nullptr,
                          (int8_t)bound, 1, 0);

    for (int64_t x = 0; x < N; ++x)
        for (int64_t c = 0; c < C; ++c) {
            double center = f[c][x];
            double expect = absolute[c] * center;
            if (use_membrane) {
                double L = neigh_1d(f[c], x-1, bound);
                double R = neigh_1d(f[c], x+1, bound);
                expect += membrane[c] * (2*center - L - R);
            }
            check_close((double)out[x * C + c], expect, "field1d_matvec");
        }
}

template <typename scalar_t>
void run_1d_diag(int64_t N, int64_t C, const std::vector<double>& absolute,
                 const std::vector<double>& membrane, int bound, uint8_t bits)
{
    std::vector<int64_t> shape = {N, C};
    std::vector<int64_t> str   = contiguous_strides(shape);
    int64_t numel = N * C;

    std::vector<scalar_t> diag(numel, 0);
    DLTensor td = make_cpu_tensor(diag.data(), shape, str, bits);
    ff::cpu::field_diag(td, nullptr, absolute.data(), membrane.data(), nullptr,
                        (int8_t)bound, 1, 0);

    // interior only (see note in test_reg_flow.cpp about the boundary convention)
    for (int64_t x = 1; x < N - 1; ++x)
        for (int64_t c = 0; c < C; ++c) {
            std::vector<scalar_t> e(numel, 0), o(numel, 0);
            e[x * C + c] = 1;
            DLTensor te = make_cpu_tensor(e.data(), shape, str, bits);
            DLTensor to = make_cpu_tensor(o.data(), shape, str, bits);
            ff::cpu::field_matvec(to, te, nullptr, absolute.data(), membrane.data(),
                                  nullptr, (int8_t)bound, 1, 0);
            check_close((double)diag[x * C + c], (double)o[x * C + c], "field1d_diag_interior");
        }
}

// 2D field (H, W, C), absolute-only exact scaling.
template <typename scalar_t>
void run_2d_absolute(int64_t H, int64_t W, int64_t C,
                     const std::vector<double>& absolute, uint8_t bits)
{
    std::vector<int64_t> shape = {H, W, C};
    std::vector<int64_t> str   = contiguous_strides(shape);
    int64_t numel = H * W * C;
    std::vector<scalar_t> inp(numel), out(numel, scalar_t(-1));
    for (int64_t i = 0; i < numel; ++i) inp[i] = (scalar_t)(0.05 * i - 2.0);

    DLTensor tin  = make_cpu_tensor(inp.data(), shape, str, bits);
    DLTensor tout = make_cpu_tensor(out.data(), shape, str, bits);
    ff::cpu::field_matvec(tout, tin, nullptr, absolute.data(), nullptr, nullptr,
                          (int8_t)B_DCT2, 2, 0);

    for (int64_t i = 0; i < numel; ++i) {
        int64_t c = i % C;
        check_close((double)out[i], absolute[c] * (double)inp[i], "field2d_abs");
    }
}

// 1D field bending interior == bending[c] * biharmonic stencil [1,-4,6,-4,1].
template <typename scalar_t>
void run_1d_bending(int64_t N, int64_t C, const std::vector<double>& bending, uint8_t bits)
{
    std::vector<int64_t> shape = {N, C};
    std::vector<int64_t> str   = contiguous_strides(shape);
    int64_t numel = N * C;
    std::vector<scalar_t> inp(numel), out(numel, scalar_t(-1));
    std::vector<double> zero(C, 0.0);
    std::vector<std::vector<double>> f(C, std::vector<double>(N));
    for (int64_t x = 0; x < N; ++x)
        for (int64_t c = 0; c < C; ++c) {
            double v = std::sin(0.35*x + 0.6*c + 0.2);
            inp[x*C+c] = (scalar_t)v; f[c][x] = v;
        }
    DLTensor ti = make_cpu_tensor(inp.data(), shape, str, bits);
    DLTensor to = make_cpu_tensor(out.data(), shape, str, bits);
    ff::cpu::field_matvec(to, ti, nullptr, zero.data(), zero.data(), bending.data(),
                          (int8_t)B_DCT2, 1, 0);
    for (int64_t x = 2; x < N-2; ++x)
        for (int64_t c = 0; c < C; ++c) {
            double e = bending[c] * (f[c][x-2] - 4*f[c][x-1] + 6*f[c][x]
                                     - 4*f[c][x+1] + f[c][x+2]);
            check_close((double)out[x*C+c], e, "field1d_bending");
        }
}

// 3D field membrane interior == membrane[c] * (6*center - 6 face-neighbours).
template <typename scalar_t>
void run_3d_membrane(int64_t N, int64_t C, const std::vector<double>& membrane, uint8_t bits)
{
    std::vector<int64_t> shape = {N, N, N, C};
    std::vector<int64_t> str   = contiguous_strides(shape);
    int64_t numel = N * N * N * C;
    std::vector<scalar_t> inp(numel), out(numel, scalar_t(-1));
    std::vector<double> absolute(C, 0.0);
    auto idx = [&](int64_t x,int64_t y,int64_t z,int64_t c){ return ((x*N+y)*N+z)*C+c; };
    for (int64_t i = 0; i < numel; ++i) inp[i] = (scalar_t)std::sin(0.17*i + 0.3);

    DLTensor ti = make_cpu_tensor(inp.data(), shape, str, bits);
    DLTensor to = make_cpu_tensor(out.data(), shape, str, bits);
    ff::cpu::field_matvec(to, ti, nullptr, absolute.data(), membrane.data(), nullptr,
                          (int8_t)B_DCT2, 3, 0);

    for (int64_t x = 1; x < N-1; ++x)
    for (int64_t y = 1; y < N-1; ++y)
    for (int64_t z = 1; z < N-1; ++z)
    for (int64_t c = 0; c < C; ++c) {
        double ctr = (double)inp[idx(x,y,z,c)];
        double s = (double)inp[idx(x-1,y,z,c)] + (double)inp[idx(x+1,y,z,c)]
                 + (double)inp[idx(x,y-1,z,c)] + (double)inp[idx(x,y+1,z,c)]
                 + (double)inp[idx(x,y,z-1,c)] + (double)inp[idx(x,y,z+1,c)];
        check_close((double)out[idx(x,y,z,c)], membrane[c]*(6*ctr - s), "field3d_membrane");
    }
}

// --- B4: negative / validation tests --------------------------------------
// Bad dtype: float16 must throw at the dtype dispatch, not silently no-op.
void test_bad_dtype_throws()
{
    const int64_t N = 6, C = 2;
    std::vector<uint16_t> inp(N*C,0), out(N*C,0);          // float16 payload
    std::vector<double> absolute(C, 1.0);
    std::vector<int64_t> sh={N,C}, st=contiguous_strides(sh);
    DLTensor tin =make_cpu_tensor(inp.data(),sh,st,16);
    DLTensor tout=make_cpu_tensor(out.data(),sh,st,16);
    bool threw = false;
    try { ff::cpu::field_matvec(tout, tin, nullptr, absolute.data(), nullptr, nullptr,
                                (int8_t)B_DCT2, 1, 0); }
    catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [reg_field.bad_dtype_throws]\n"); }
}

// Shape mismatch: out and inp with different spatial extents must throw.
void test_shape_mismatch_throws()
{
    const int64_t N = 6, C = 2;
    std::vector<double> inp((N+1)*C,0), out(N*C,0);        // differing spatial dim
    std::vector<double> absolute(C, 1.0);
    std::vector<int64_t> shi={N+1,C}, sti=contiguous_strides(shi);
    std::vector<int64_t> sho={N,C},   sto=contiguous_strides(sho);
    DLTensor tin =make_cpu_tensor(inp.data(),shi,sti,64);
    DLTensor tout=make_cpu_tensor(out.data(),sho,sto,64);
    bool threw = false;
    try { ff::cpu::field_matvec(tout, tin, nullptr, absolute.data(), nullptr, nullptr,
                                (int8_t)B_DCT2, 1, 0); }
    catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [reg_field.shape_mismatch_throws]\n"); }
}

// field_kernel: the materialised per-channel stencil must equal the operator's
// impulse response in the interior. Apply field_matvec to a unit impulse at the
// centre of a domain large enough that the response never touches a boundary,
// then compare the centred window to the kernel returned by field_kernel. Field
// channels are independent, so an impulse in channel c0 only affects output
// channel c0. `order` selects which penalty pointers are passed (and the
// stencil width): 1 absolute, 3 membrane, 5 bending.
template <typename scalar_t>
void run_2d_kernel_impulse(int64_t kd, int64_t C, int order,
                           const std::vector<double>& absolute,
                           const std::vector<double>& membrane,
                           const std::vector<double>& bending,
                           uint8_t bits, int bound = B_DCT2)
{
    const double* ap = absolute.data();
    const double* mp = (order >= 2) ? membrane.data() : nullptr;
    const double* bp = (order >= 3) ? bending.data()  : nullptr;

    std::vector<int64_t> kshape = {kd, kd, C};
    std::vector<int64_t> kstr   = contiguous_strides(kshape);
    std::vector<scalar_t> K(kd * kd * C, scalar_t(0));
    DLTensor tK = make_cpu_tensor(K.data(), kshape, kstr, bits);
    ff::cpu::field_kernel(tK, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);

    const int64_t N = 2 * kd + 1, cc = N / 2, half = kd / 2;
    std::vector<int64_t> fshape = {N, N, C};
    std::vector<int64_t> fstr   = contiguous_strides(fshape);
    int64_t fnumel = N * N * C;
    auto fidx = [&](int64_t i, int64_t j, int64_t c){ return (i * N + j) * C + c; };

    for (int64_t c0 = 0; c0 < C; ++c0) {
        std::vector<scalar_t> x(fnumel, scalar_t(0)), o(fnumel, scalar_t(0));
        x[fidx(cc, cc, c0)] = 1;
        DLTensor tx = make_cpu_tensor(x.data(), fshape, fstr, bits);
        DLTensor to = make_cpu_tensor(o.data(), fshape, fstr, bits);
        ff::cpu::field_matvec(to, tx, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);
        for (int64_t a = 0; a < kd; ++a)
        for (int64_t b = 0; b < kd; ++b)
        for (int64_t c = 0; c < C; ++c) {
            double got  = (double)o[fidx(cc + (a - half), cc + (b - half), c)];
            double kern = (c == c0) ? (double)K[(a * kd + b) * C + c] : 0.0;
            char buf[96];
            std::snprintf(buf, sizeof(buf),
                "field2d_kernel_impulse[kd=%lld C=%lld order=%d bnd=%d]",
                (long long)kd, (long long)C, order, bound);
            check_close(got, kern, buf);
        }
    }
}

// field_relax: relaxation drives (H + L) x -> g. With a diagonal Hessian
// (H x = hdiag * x), the residual hdiag*sol + L*sol - grd must go to ~0.
template <typename scalar_t>
void run_2d_relax(int64_t Hgt, int64_t W, int64_t C, double hdiag, int order,
                  const std::vector<double>& absolute,
                  const std::vector<double>& membrane,
                  const std::vector<double>& bending,
                  uint8_t bits, int bound = B_DCT2, int niter = 250)
{
    const int64_t K = C * (C + 1) / 2;
    std::vector<int64_t> fshape = {Hgt, W, C};
    std::vector<int64_t> hshape = {Hgt, W, K};
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    std::vector<int64_t> hstr = contiguous_strides(hshape);
    int64_t fnum = Hgt * W * C, hnum = Hgt * W * K;

    std::vector<scalar_t> sol(fnum, 0), grd(fnum), hes(hnum, 0);
    for (int64_t i = 0; i < fnum; ++i)
        grd[i] = (scalar_t)std::sin(0.4 * i + 0.2);
    // diagonal-only Hessian: first C packed entries are the diagonal.
    for (int64_t p = 0; p < Hgt * W; ++p)
        for (int64_t c = 0; c < C; ++c) hes[p * K + c] = (scalar_t)hdiag;

    const double* ap = absolute.data();
    const double* mp = (order >= 2) ? membrane.data() : nullptr;
    const double* bp = (order >= 3) ? bending.data()  : nullptr;

    DLTensor tsol = make_cpu_tensor(sol.data(), fshape, fstr, bits);
    DLTensor thes = make_cpu_tensor(hes.data(), hshape, hstr, bits);
    DLTensor tgrd = make_cpu_tensor(grd.data(), fshape, fstr, bits);
    ff::cpu::field_relax(tsol, thes, tgrd, nullptr, ap, mp, bp,
                         (int8_t)bound, 2, niter, 0);

    std::vector<scalar_t> Lx(fnum, 0);
    DLTensor tLx = make_cpu_tensor(Lx.data(), fshape, fstr, bits);
    ff::cpu::field_matvec(tLx, tsol, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);
    double res = 0, nrm = 0;
    for (int64_t i = 0; i < fnum; ++i) {
        double r = hdiag * (double)sol[i] + (double)Lx[i] - (double)grd[i];
        res += r * r;
        nrm += (double)grd[i] * (double)grd[i];
    }
    double rel = std::sqrt(res / nrm);
    char buf[96];
    std::snprintf(buf, sizeof(buf),
        "field2d_relax_residual[C=%lld order=%d bound=%d] rel=%.2e",
        (long long)C, order, bound, rel);
    check_close(rel, 0.0, buf, 3e-3);
}

// field_matvec_rls: the RLS/JRLS operator L(w) must stay self-adjoint under
// a spatially varying (positive) weight map, i.e. <L(w)x, y> == <x, L(w)y>
// for any x, y -- same oracle as the plain-operator symmetry check, but
// exercising the weighted matvec path (and, via wc, both the shared-weight
// RLS and per-channel JRLS dispatch).
template <typename scalar_t>
void run_2d_matvec_rls_symmetry(int64_t Hgt, int64_t W, int64_t C, int64_t wc,
                                int order, const std::vector<double>& absolute,
                                const std::vector<double>& membrane,
                                const std::vector<double>& bending,
                                uint8_t bits, int bound = B_DCT2)
{
    std::vector<int64_t> fshape = {Hgt, W, C}, wshape = {Hgt, W, wc};
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    std::vector<int64_t> wstr = contiguous_strides(wshape);
    int64_t fnum = Hgt * W * C, wnum = Hgt * W * wc;

    std::vector<scalar_t> x(fnum), y(fnum), w(wnum), Lx(fnum, 0), Ly(fnum, 0);
    for (int64_t i = 0; i < fnum; ++i) {
        x[i] = (scalar_t)std::sin(0.31 * i + 0.11);
        y[i] = (scalar_t)std::cos(0.23 * i + 0.71);
    }
    for (int64_t i = 0; i < wnum; ++i)
        w[i] = (scalar_t)(0.5 + std::fabs(std::sin(0.17 * i + 1.3)));

    const double* ap = absolute.data();
    const double* mp = (order >= 2) ? membrane.data() : nullptr;
    const double* bp = (order >= 3) ? bending.data()  : nullptr;

    DLTensor tx  = make_cpu_tensor(x.data(),  fshape, fstr, bits);
    DLTensor ty  = make_cpu_tensor(y.data(),  fshape, fstr, bits);
    DLTensor tw  = make_cpu_tensor(w.data(),  wshape, wstr, bits);
    DLTensor tLx = make_cpu_tensor(Lx.data(), fshape, fstr, bits);
    DLTensor tLy = make_cpu_tensor(Ly.data(), fshape, fstr, bits);
    ff::cpu::field_matvec_rls(tLx, tx, tw, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);
    ff::cpu::field_matvec_rls(tLy, ty, tw, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);

    double lhs = 0, rhs = 0;
    for (int64_t i = 0; i < fnum; ++i) { lhs += (double)Lx[i]*(double)y[i]; rhs += (double)x[i]*(double)Ly[i]; }
    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "field2d_matvec_rls_symmetry[C=%lld wc=%lld order=%d bound=%d]",
        (long long)C, (long long)wc, order, bound);
    check_close(lhs, rhs, buf);
}

// field_matvec_rls under a *uniform* weight map (w == 1) must reduce exactly to
// the plain, unweighted field_matvec at the same order: the weighted operator
// is a strict generalisation of the unweighted one, so w == 1 is the identity
// case. This pins the *magnitude* of every penalty term, which the
// self-adjointness check above cannot see -- a term computed at the wrong
// strength stays perfectly symmetric.
//
// Regression test for fastfields-kernels#40: make_kernel_bending_rls used to
// rescale its whole kernel table by 0.25, which is right for the second-order
// (bending) coefficients but wrong for the first-order entries, since those
// also carry a membrane contribution (wanting 0.5). Membrane therefore ran at
// half strength whenever the bending-order kernel was the one constructed.
template <typename scalar_t>
void run_2d_matvec_rls_unit_weight(int64_t Hgt, int64_t W, int64_t C, int64_t wc,
                                   int order, const std::vector<double>& absolute,
                                   const std::vector<double>& membrane,
                                   const std::vector<double>& bending,
                                   uint8_t bits, int bound = B_DCT2)
{
    std::vector<int64_t> fshape = {Hgt, W, C}, wshape = {Hgt, W, wc};
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    std::vector<int64_t> wstr = contiguous_strides(wshape);
    int64_t fnum = Hgt * W * C, wnum = Hgt * W * wc;

    std::vector<scalar_t> x(fnum), w(wnum, (scalar_t)1), Lw(fnum, 0), L(fnum, 0);
    for (int64_t i = 0; i < fnum; ++i)
        x[i] = (scalar_t)std::sin(0.31 * i + 0.11);

    const double* ap = absolute.data();
    const double* mp = (order >= 2) ? membrane.data() : nullptr;
    const double* bp = (order >= 3) ? bending.data()  : nullptr;

    DLTensor tx  = make_cpu_tensor(x.data(),  fshape, fstr, bits);
    DLTensor tw  = make_cpu_tensor(w.data(),  wshape, wstr, bits);
    DLTensor tLw = make_cpu_tensor(Lw.data(), fshape, fstr, bits);
    DLTensor tL  = make_cpu_tensor(L.data(),  fshape, fstr, bits);
    ff::cpu::field_matvec_rls(tLw, tx, tw, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);
    ff::cpu::field_matvec(tL, tx, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);

    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "field2d_matvec_rls_unit_weight[C=%lld wc=%lld order=%d bound=%d]",
        (long long)C, (long long)wc, order, bound);
    for (int64_t i = 0; i < fnum; ++i)
        check_close((double)Lw[i], (double)L[i], buf);
}

// field_diag_rls must reproduce the (weighted) operator's diagonal, same as
// the plain diag/matvec-on-unit-vector check but through the RLS path.
template <typename scalar_t>
void run_2d_diag_rls(int64_t Hgt, int64_t W, int64_t C, int64_t wc, int order,
                     const std::vector<double>& absolute,
                     const std::vector<double>& membrane,
                     const std::vector<double>& bending,
                     uint8_t bits, int bound = B_DCT2)
{
    std::vector<int64_t> fshape = {Hgt, W, C}, wshape = {Hgt, W, wc};
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    std::vector<int64_t> wstr = contiguous_strides(wshape);
    int64_t fnum = Hgt * W * C, wnum = Hgt * W * wc;

    std::vector<scalar_t> w(wnum), d(fnum, 0);
    for (int64_t i = 0; i < wnum; ++i)
        w[i] = (scalar_t)(0.5 + std::fabs(std::sin(0.17 * i + 1.3)));

    const double* ap = absolute.data();
    const double* mp = (order >= 2) ? membrane.data() : nullptr;
    const double* bp = (order >= 3) ? bending.data()  : nullptr;

    DLTensor tw = make_cpu_tensor(w.data(), wshape, wstr, bits);
    DLTensor td = make_cpu_tensor(d.data(), fshape, fstr, bits);
    ff::cpu::field_diag_rls(td, tw, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);

    auto idx = [&](int64_t i, int64_t j, int64_t c) { return (i * W + j) * C + c; };
    for (int64_t i = 1; i < Hgt - 1; ++i)
    for (int64_t j = 1; j < W - 1; ++j)
    for (int64_t c = 0; c < C; ++c) {
        std::vector<scalar_t> e(fnum, 0), o(fnum, 0);
        e[idx(i, j, c)] = 1;
        DLTensor te = make_cpu_tensor(e.data(), fshape, fstr, bits);
        DLTensor to = make_cpu_tensor(o.data(), fshape, fstr, bits);
        ff::cpu::field_matvec_rls(to, te, tw, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);
        check_close((double)d[idx(i, j, c)], (double)o[idx(i, j, c)],
                    "field2d_diag_rls_interior");
    }
}

// field_matvec_rls / field_diag_rls: a genuine per-channel weight map
// (wc == nc) -- ground-truth regression test for fastfields-cpu-lib#65.
// Before the fix, `field_rls_is_jrls()` had RLS/JRLS backwards: a genuine
// per-channel weight (wc == nc) was mislabelled as JRLS and dispatched to
// the broadcast-only `_jrls` kernels, which read the weight value once per
// voxel (outside any per-channel loop) and silently broadcast channel 0's
// weight to every channel; every prior RLS/JRLS test here uses either an
// all-ones weight (invisible to a broadcast bug: every channel's "true"
// weight is the same 1.0 anyway) or only checks self-adjointness (symmetric
// regardless of which value gets broadcast, as long as it is applied
// consistently forward and adjoint) -- neither catches this.
//
// The oracle: the field regulariser documentedly never couples channels
// (each channel's penalty is applied independently), so applying the
// operator to a C-channel field with a per-channel weight map must exactly
// decompose into C independent single-channel problems, each run through
// the wc == 1 (broadcast/JRLS) path instead -- a *different* dispatch
// branch that is already extensively verified above. This is therefore not
// a self-consistency check against the (potentially buggy) per-channel
// code: it is an independent ground truth from a separately dispatched,
// separately tested code path. Channel weights are given distinct
// per-channel frequency and phase so that w(...,c) != w(...,c') at every
// voxel, which is exactly what makes a channel-0 broadcast visible for
// every c > 0.
template <typename scalar_t>
void run_2d_rls_jrls_per_channel(int64_t Hgt, int64_t W, int64_t C, int order,
                                 const std::vector<double> & absolute,
                                 const std::vector<double> & membrane,
                                 const std::vector<double> & bending,
                                 uint8_t bits, int bound = B_DCT2)
{
    std::vector<int64_t> fshape = {Hgt, W, C};
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    std::vector<int64_t> sshape = {Hgt, W, 1}; // single-channel: C == wc == 1
    std::vector<int64_t> sstr = contiguous_strides(sshape);
    int64_t fnum = Hgt * W * C, snum = Hgt * W;

    std::vector<scalar_t> x(fnum), w(fnum), Lx(fnum, 0), diagm(fnum, 0);
    for (int64_t i = 0; i < snum; ++i)
        for (int64_t c = 0; c < C; ++c) {
            x[i * C + c] = (scalar_t)std::sin(0.31 * i + 0.6 * c + 0.11);
            w[i * C + c] =
                (scalar_t)(0.5 + std::fabs(std::sin(0.19 * i * (c + 1) +
                                                    0.9 * (c + 1) + 0.4)));
        }

    const double * ap = absolute.data();
    const double * mp = (order >= 2) ? membrane.data() : nullptr;
    const double * bp = (order >= 3) ? bending.data() : nullptr;

    DLTensor tx = make_cpu_tensor(x.data(), fshape, fstr, bits);
    DLTensor tw = make_cpu_tensor(w.data(), fshape, fstr,
                                  bits); // wc==nc -> RLS (genuine per-channel)
    DLTensor tLx = make_cpu_tensor(Lx.data(), fshape, fstr, bits);
    DLTensor tdg = make_cpu_tensor(diagm.data(), fshape, fstr, bits);
    ff::cpu::field_matvec_rls(tLx, tx, tw, nullptr, ap, mp, bp, (int8_t)bound,
                              2, 0);
    ff::cpu::field_diag_rls(tdg, tw, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);

    for (int64_t c = 0; c < C; ++c) {
        std::vector<scalar_t> xc(snum), wc_(snum), Lc(snum, 0), dc(snum, 0);
        for (int64_t i = 0; i < snum; ++i) {
            xc[i] = x[i * C + c];
            wc_[i] = w[i * C + c];
        }
        std::vector<double> ac{absolute[c]};
        std::vector<double> mc{(order >= 2) ? membrane[c] : 0.0};
        std::vector<double> bc{(order >= 3) ? bending[c] : 0.0};
        const double * mcp = (order >= 2) ? mc.data() : nullptr;
        const double * bcp = (order >= 3) ? bc.data() : nullptr;

        DLTensor txc = make_cpu_tensor(xc.data(), sshape, sstr, bits);
        DLTensor twc = make_cpu_tensor(wc_.data(), sshape, sstr,
                                       bits); // wc==1 -> JRLS (broadcast)
        DLTensor tLc = make_cpu_tensor(Lc.data(), sshape, sstr, bits);
        DLTensor tdc = make_cpu_tensor(dc.data(), sshape, sstr, bits);
        ff::cpu::field_matvec_rls(tLc, txc, twc, nullptr, ac.data(), mcp, bcp,
                                  (int8_t)bound, 2, 0);
        ff::cpu::field_diag_rls(tdc, twc, nullptr, ac.data(), mcp, bcp,
                                (int8_t)bound, 2, 0);

        char bufm[160], bufd[160];
        std::snprintf(
            bufm, sizeof(bufm),
            "field2d_matvec_jrls_per_channel[C=%lld c=%lld order=%d bnd=%d]",
            (long long)C, (long long)c, order, bound);
        std::snprintf(
            bufd, sizeof(bufd),
            "field2d_diag_jrls_per_channel[C=%lld c=%lld order=%d bnd=%d]",
            (long long)C, (long long)c, order, bound);
        for (int64_t i = 0; i < snum; ++i) {
            check_close((double)Lx[i * C + c], (double)Lc[i], bufm);
            check_close((double)diagm[i * C + c], (double)dc[i], bufd);
        }
    }
}

// field_relax_rls: relaxation drives (H + L(w)) x -> g, same oracle as
// run_2d_relax but through the weighted (RLS/JRLS) operator.
template <typename scalar_t>
void run_2d_relax_rls(int64_t Hgt, int64_t W, int64_t C, int64_t wc,
                      double hdiag, int order,
                      const std::vector<double>& absolute,
                      const std::vector<double>& membrane,
                      const std::vector<double>& bending,
                      uint8_t bits, int bound = B_DCT2, int niter = 250)
{
    const int64_t K = C * (C + 1) / 2;
    std::vector<int64_t> fshape = {Hgt, W, C}, hshape = {Hgt, W, K}, wshape = {Hgt, W, wc};
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    std::vector<int64_t> hstr = contiguous_strides(hshape);
    std::vector<int64_t> wstr = contiguous_strides(wshape);
    int64_t fnum = Hgt * W * C, hnum = Hgt * W * K, wnum = Hgt * W * wc;

    std::vector<scalar_t> sol(fnum, 0), grd(fnum), hes(hnum, 0), w(wnum);
    for (int64_t i = 0; i < fnum; ++i)
        grd[i] = (scalar_t)std::sin(0.4 * i + 0.2);
    for (int64_t p = 0; p < Hgt * W; ++p)
        for (int64_t c = 0; c < C; ++c) hes[p * K + c] = (scalar_t)hdiag;
    for (int64_t i = 0; i < wnum; ++i)
        w[i] = (scalar_t)(0.5 + std::fabs(std::sin(0.17 * i + 1.3)));

    const double* ap = absolute.data();
    const double* mp = (order >= 2) ? membrane.data() : nullptr;
    const double* bp = (order >= 3) ? bending.data()  : nullptr;

    DLTensor tsol = make_cpu_tensor(sol.data(), fshape, fstr, bits);
    DLTensor thes = make_cpu_tensor(hes.data(), hshape, hstr, bits);
    DLTensor tgrd = make_cpu_tensor(grd.data(), fshape, fstr, bits);
    DLTensor tw   = make_cpu_tensor(w.data(),   wshape, wstr, bits);
    ff::cpu::field_relax_rls(tsol, thes, tgrd, tw, nullptr, ap, mp, bp,
                             (int8_t)bound, 2, niter, 0);

    std::vector<scalar_t> Lx(fnum, 0);
    DLTensor tLx = make_cpu_tensor(Lx.data(), fshape, fstr, bits);
    ff::cpu::field_matvec_rls(tLx, tsol, tw, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);
    double res = 0, nrm = 0;
    for (int64_t i = 0; i < fnum; ++i) {
        double r = hdiag * (double)sol[i] + (double)Lx[i] - (double)grd[i];
        res += r * r;
        nrm += (double)grd[i] * (double)grd[i];
    }
    double rel = std::sqrt(res / nrm);
    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "field2d_relax_rls_residual[C=%lld wc=%lld order=%d bound=%d] rel=%.2e",
        (long long)C, (long long)wc, order, bound, rel);
    check_close(rel, 0.0, buf, 3e-3);
}

// Regression test for fastfields-cpu-impl#51: relax_bending_rls_ and
// relax_bending_jrls_ used a `2*niter` colour-sweep loop bound (copied from
// the membrane/patch1 sites) instead of `pow<ndim>(3)*niter` (patch3, which
// bending's reach-2 stencil needs). A residual-convergence check (like
// run_2d_relax_rls above) does not reliably catch this on a small grid --
// with `niter` large enough, `2*niter` iterations still end up completing
// many partial passes through the 3^ndim==9 (2D) colour cycle and can
// converge anyway. This test uses a sharper, deterministic oracle instead:
// with a **zero warm start** and **niter=1** (one intended full sweep),
// every interior voxel must move off exactly 0.0 -- any left bit-identical
// to their initial value were never visited by the (buggy) colour loop.
template <typename scalar_t>
void run_2d_relax_rls_bending_sweep_visits_every_colour(int64_t wc,
                                                        uint8_t bits)
{
    const int64_t Hgt = 9, W = 9, C = 2, K = C * (C + 1) / 2;
    std::vector<int64_t> fshape = {Hgt, W, C}, hshape = {Hgt, W, K},
                          wshape = {Hgt, W, wc};
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    std::vector<int64_t> hstr = contiguous_strides(hshape);
    std::vector<int64_t> wstr = contiguous_strides(wshape);
    int64_t fnum = Hgt * W * C, hnum = Hgt * W * K, wnum = Hgt * W * wc;

    std::vector<scalar_t> sol(fnum, 0), grd(fnum), hes(hnum, 0), w(wnum);
    for (int64_t i = 0; i < fnum; ++i)
        grd[i] = (scalar_t)(1.0 + std::fabs(std::sin(0.4 * i + 0.2)));
    for (int64_t p = 0; p < Hgt * W; ++p)
        for (int64_t c = 0; c < C; ++c) hes[p * K + c] = (scalar_t)4.0;
    for (int64_t i = 0; i < wnum; ++i)
        w[i] = (scalar_t)(0.5 + std::fabs(std::sin(0.17 * i + 1.3)));

    std::vector<double> ap{0.3, 0.4}, mp{0.5, 0.6}, bp{1.0, 0.8};

    DLTensor tsol = make_cpu_tensor(sol.data(), fshape, fstr, bits);
    DLTensor thes = make_cpu_tensor(hes.data(), hshape, hstr, bits);
    DLTensor tgrd = make_cpu_tensor(grd.data(), fshape, fstr, bits);
    DLTensor tw   = make_cpu_tensor(w.data(),   wshape, wstr, bits);
    ff::cpu::field_relax_rls(tsol, thes, tgrd, tw, nullptr, ap.data(),
                             mp.data(), bp.data(), (int8_t)B_DCT2, 2,
                             /*nb_iter=*/1, 0);

    // Bending's stencil reaches 2, so restrict to the interior (margin 2)
    // to sidestep boundary-fold subtleties and isolate the colour-sweep
    // coverage question itself.
    int64_t untouched = 0, total = 0;
    for (int64_t i = 2; i < Hgt - 2; ++i)
    for (int64_t j = 2; j < W - 2; ++j)
    for (int64_t c = 0; c < C; ++c) {
        ++total;
        if (sol[(i * W + j) * C + c] == (scalar_t)0)
            ++untouched;
    }
    ++g_checks;
    if (untouched != 0) {
        ++g_failures;
        std::printf(
            "  MISMATCH [field2d_relax_rls_bending_sweep_visits_every_colour"
            "[wc=%lld]]: %lld/%lld interior voxels left bit-identical to "
            "their zero warm start after one intended full sweep (niter=1) "
            "-- the colour loop did not visit every 3^ndim colour\n",
            (long long)wc, (long long)untouched, (long long)total);
    }
}

// field_addmatvec_/field_submatvec_ must reproduce out (+/-)= field_matvec(inp)
// against a nonzero pre-existing out buffer.
template <typename scalar_t>
void run_2d_matvec_addsub(int64_t Hgt, int64_t W, int64_t C, int order,
                          const std::vector<double>& absolute,
                          const std::vector<double>& membrane,
                          const std::vector<double>& bending,
                          uint8_t bits, int bound = B_DCT2)
{
    std::vector<int64_t> fshape = {Hgt, W, C};
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    int64_t fnum = Hgt * W * C;

    std::vector<scalar_t> x(fnum), base(fnum), Lx(fnum, 0), acc_add(fnum), acc_sub(fnum);
    for (int64_t i = 0; i < fnum; ++i) {
        x[i]    = (scalar_t)std::sin(0.31 * i + 0.11);
        base[i] = (scalar_t)std::cos(0.17 * i + 0.4);
    }
    acc_add = base;
    acc_sub = base;

    const double* ap = absolute.data();
    const double* mp = (order >= 2) ? membrane.data() : nullptr;
    const double* bp = (order >= 3) ? bending.data()  : nullptr;

    DLTensor tx       = make_cpu_tensor(x.data(),       fshape, fstr, bits);
    DLTensor tLx      = make_cpu_tensor(Lx.data(),      fshape, fstr, bits);
    DLTensor tacc_add = make_cpu_tensor(acc_add.data(), fshape, fstr, bits);
    DLTensor tacc_sub = make_cpu_tensor(acc_sub.data(), fshape, fstr, bits);

    ff::cpu::field_matvec(tLx, tx, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);
    ff::cpu::field_addmatvec_(tacc_add, tx, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);
    ff::cpu::field_submatvec_(tacc_sub, tx, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);

    for (int64_t i = 0; i < fnum; ++i) {
        check_close((double)acc_add[i], (double)base[i] + (double)Lx[i], "field2d_addmatvec");
        check_close((double)acc_sub[i], (double)base[i] - (double)Lx[i], "field2d_submatvec");
    }
}

// field_adddiag_/_sub and field_addkernel_/_sub must reproduce
// out (+/-)= field_diag(...) / field_kernel(...) against a nonzero pre-existing
// out buffer. These are the in-place-only accumulate entry points restored from
// jitfields (op '+' / '-'); the '=' path is field_diag / field_kernel.
template <typename scalar_t>
void run_2d_diag_kernel_addsub(int64_t Hgt, int64_t W, int64_t C, int order,
                               const std::vector<double>& absolute,
                               const std::vector<double>& membrane,
                               const std::vector<double>& bending,
                               uint8_t bits, int bound = B_DCT2)
{
    std::vector<int64_t> fshape = {Hgt, W, C};
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    int64_t fnum = Hgt * W * C;

    const double* ap = absolute.data();
    const double* mp = (order >= 2) ? membrane.data() : nullptr;
    const double* bp = (order >= 3) ? bending.data()  : nullptr;

    // ---- diag ----
    {
        std::vector<scalar_t> base(fnum), D(fnum, 0), acc_add(fnum), acc_sub(fnum);
        for (int64_t i = 0; i < fnum; ++i)
            base[i] = (scalar_t)std::cos(0.17 * i + 0.4);
        acc_add = base;
        acc_sub = base;

        DLTensor tD  = make_cpu_tensor(D.data(),       fshape, fstr, bits);
        DLTensor ta  = make_cpu_tensor(acc_add.data(), fshape, fstr, bits);
        DLTensor ts  = make_cpu_tensor(acc_sub.data(), fshape, fstr, bits);

        ff::cpu::field_diag    (tD, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);
        ff::cpu::field_adddiag_(ta, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);
        ff::cpu::field_subdiag_(ts, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);

        for (int64_t i = 0; i < fnum; ++i) {
            check_close((double)acc_add[i], (double)base[i] + (double)D[i], "field2d_adddiag");
            check_close((double)acc_sub[i], (double)base[i] - (double)D[i], "field2d_subdiag");
        }
    }

    // ---- kernel (stencil) ----
    {
        std::vector<scalar_t> base(fnum), K(fnum, 0), acc_add(fnum), acc_sub(fnum);
        for (int64_t i = 0; i < fnum; ++i)
            base[i] = (scalar_t)std::sin(0.23 * i + 0.9);
        acc_add = base;
        acc_sub = base;

        DLTensor tK = make_cpu_tensor(K.data(),       fshape, fstr, bits);
        DLTensor ta = make_cpu_tensor(acc_add.data(), fshape, fstr, bits);
        DLTensor ts = make_cpu_tensor(acc_sub.data(), fshape, fstr, bits);

        ff::cpu::field_kernel    (tK, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);
        ff::cpu::field_addkernel_(ta, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);
        ff::cpu::field_subkernel_(ts, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);

        for (int64_t i = 0; i < fnum; ++i) {
            check_close((double)acc_add[i], (double)base[i] + (double)K[i], "field2d_addkernel");
            check_close((double)acc_sub[i], (double)base[i] - (double)K[i], "field2d_subkernel");
        }
    }
}

// field_precond solves (H + diag(L)) x = grd exactly, per voxel (a Jacobi-type
// direct solve -- unlike field_relax, which iterates on the full (H + L)
// system). Verify the defining residual H*x + diag(L).*x == grd using the
// independently-tested posdef::sym_matvec and field_diag, and that the
// in-place field_precond_ agrees with the out-of-place field_precond.
template <typename scalar_t>
void run_2d_precond(int64_t Hgt, int64_t W, int64_t C, double hdiag, int order,
                    const std::vector<double>& absolute,
                    const std::vector<double>& membrane,
                    const std::vector<double>& bending,
                    uint8_t bits, int bound = B_DCT2)
{
    const int64_t K = C * (C + 1) / 2;
    std::vector<int64_t> fshape = {Hgt, W, C};
    std::vector<int64_t> hshape = {Hgt, W, K};
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    std::vector<int64_t> hstr = contiguous_strides(hshape);
    int64_t fnum = Hgt * W * C, hnum = Hgt * W * K;

    std::vector<scalar_t> grd(fnum), hes(hnum, 0), out(fnum, 0), sol(fnum);
    for (int64_t i = 0; i < fnum; ++i)
        grd[i] = (scalar_t)std::sin(0.4 * i + 0.2);
    for (int64_t p = 0; p < Hgt * W; ++p)
        for (int64_t c = 0; c < C; ++c) hes[p * K + c] = (scalar_t)hdiag;

    const double* ap = absolute.data();
    const double* mp = (order >= 2) ? membrane.data() : nullptr;
    const double* bp = (order >= 3) ? bending.data()  : nullptr;

    DLTensor thes = make_cpu_tensor(hes.data(), hshape, hstr, bits);
    DLTensor tgrd = make_cpu_tensor(grd.data(), fshape, fstr, bits);
    DLTensor tout = make_cpu_tensor(out.data(), fshape, fstr, bits);
    ff::cpu::field_precond(tout, thes, tgrd, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);

    // in-place variant must agree with the out-of-place one
    sol = grd;
    DLTensor tsol = make_cpu_tensor(sol.data(), fshape, fstr, bits);
    ff::cpu::field_precond_(tsol, thes, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);
    for (int64_t i = 0; i < fnum; ++i)
        check_close((double)sol[i], (double)out[i], "field2d_precond_inplace_matches");

    // residual: H*x + diag(L).*x == grd (exact per-voxel Jacobi solve)
    std::vector<scalar_t> diagL(fnum, 0), Hx(fnum, 0);
    DLTensor tdiagL = make_cpu_tensor(diagL.data(), fshape, fstr, bits);
    ff::cpu::field_diag(tdiagL, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);
    DLTensor tHx = make_cpu_tensor(Hx.data(), fshape, fstr, bits);
    ff::cpu::sym_matvec(tHx, thes, tout, 0);
    for (int64_t i = 0; i < fnum; ++i) {
        double lhs = (double)Hx[i] + (double)diagL[i] * (double)out[i];
        check_close(lhs, (double)grd[i], "field2d_precond_residual");
    }
}

// field_forward computes (H + L) @ x; verify it equals sym_matvec(H, x) +
// field_matvec(x) elementwise, using the independently-tested primitives
// (and a full compact-symmetric Hessian, not just its diagonal).
template <typename scalar_t>
void run_2d_forward(int64_t Hgt, int64_t W, int64_t C, int order,
                    const std::vector<double>& absolute,
                    const std::vector<double>& membrane,
                    const std::vector<double>& bending,
                    uint8_t bits, int bound = B_DCT2)
{
    const int64_t K = C * (C + 1) / 2;
    std::vector<int64_t> fshape = {Hgt, W, C};
    std::vector<int64_t> hshape = {Hgt, W, K};
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    std::vector<int64_t> hstr = contiguous_strides(hshape);
    int64_t fnum = Hgt * W * C, hnum = Hgt * W * K;

    std::vector<scalar_t> x(fnum), hes(hnum), out(fnum, 0), Hx(fnum, 0), Lx(fnum, 0);
    for (int64_t i = 0; i < fnum; ++i)
        x[i] = (scalar_t)std::sin(0.31 * i + 0.11);
    for (int64_t i = 0; i < hnum; ++i)
        hes[i] = (scalar_t)(0.5 + std::fabs(std::sin(0.19 * i + 0.7)));

    const double* ap = absolute.data();
    const double* mp = (order >= 2) ? membrane.data() : nullptr;
    const double* bp = (order >= 3) ? bending.data()  : nullptr;

    DLTensor tx   = make_cpu_tensor(x.data(),   fshape, fstr, bits);
    DLTensor thes = make_cpu_tensor(hes.data(), hshape, hstr, bits);
    DLTensor tout = make_cpu_tensor(out.data(), fshape, fstr, bits);
    ff::cpu::field_forward(tout, thes, tx, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);

    DLTensor tHx = make_cpu_tensor(Hx.data(), fshape, fstr, bits);
    DLTensor tLx = make_cpu_tensor(Lx.data(), fshape, fstr, bits);
    ff::cpu::sym_matvec(tHx, thes, tx, 0);
    ff::cpu::field_matvec(tLx, tx, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);

    for (int64_t i = 0; i < fnum; ++i)
        check_close((double)out[i], (double)Hx[i] + (double)Lx[i], "field2d_forward");
}

} // namespace

int main()
{
    std::printf("reg_field module CPU tests\n");
    test_bad_dtype_throws();
    test_shape_mismatch_throws();

    // absolute-only, per-channel scaling, 1 and 2 channels
    run_1d<double>(9, 1, {2.5}, {0.0}, false, B_ZERO, 64);
    run_1d<float >(9, 2, {2.5, -1.5}, {0,0}, false, B_DCT2, 32);

    // membrane-only, negative Laplacian per channel, Zero + DCT2
    run_1d<double>(11, 1, {0.0}, {1.0}, true, B_ZERO, 64);
    run_1d<double>(11, 2, {0.0, 0.0}, {1.0, 2.0}, true, B_DCT2, 64);
    run_1d<float >(13, 2, {0.0, 0.0}, {1.3, 0.7}, true, B_ZERO, 32);

    // combined absolute + membrane
    run_1d<double>(13, 2, {0.7, 0.4}, {1.3, 0.9}, true, B_DCT2, 64);

    // diagonal consistency (interior)
    run_1d_diag<double>(11, 2, {0.7, 0.4}, {1.3, 0.9}, B_DCT2, 64);
    run_1d_diag<double>(11, 1, {0.0}, {1.0}, B_ZERO, 64);

    // 2D absolute
    run_2d_absolute<double>(4, 5, 2, {1.75, -0.5}, 64);
    run_2d_absolute<float >(3, 4, 1, {0.5}, 32);

    // 1D bending (interior biharmonic stencil, per channel)
    run_1d_bending<double>(11, 2, {1.0, 0.5}, 64);
    run_1d_bending<double>(9,  1, {2.0}, 64);

    // 3D membrane (interior negative Laplacian, per channel)
    run_3d_membrane<double>(5, 2, {1.0, 2.5}, 64);
    run_3d_membrane<double>(4, 1, {1.3}, 64);

    // field_kernel: the per-channel stencil == the operator's impulse response.
    run_2d_kernel_impulse<double>(1, 2, 1, {2.5, 1.5}, {0, 0}, {0, 0}, 64);
    run_2d_kernel_impulse<double>(3, 2, 2, {0.3, 0.4}, {1.0, 0.7}, {0, 0}, 64);
    run_2d_kernel_impulse<double>(5, 2, 3, {0.3, 0.4}, {0.5, 0.6},
                                  {1.0, 0.8}, 64);
    run_2d_kernel_impulse<float >(3, 1, 2, {0.0}, {1.0}, {0.0}, 32);
    run_2d_kernel_impulse<double>(3, 2, 2, {0.3, 0.4}, {1.0, 0.7}, {0, 0}, 64,
                                  B_ZERO);

    // field_relax: relaxation drives (H + L) x -> g (residual check).
    run_2d_relax<double>(6, 7, 2, 4.0, 2, {0.5, 0.3}, {1.0, 0.7}, {0, 0}, 64);
    run_2d_relax<double>(6, 7, 2, 4.0, 1, {2.0, 1.5}, {0, 0}, {0, 0}, 64);
    run_2d_relax<double>(6, 7, 1, 6.0, 3, {0.3}, {0.5}, {1.0}, 64);
    run_2d_relax<double>(6, 7, 2, 8.0, 3, {0.3, 0.4}, {0.5, 0.6},
                         {1.0, 0.8}, 64);

    // field_addmatvec_ / field_submatvec_: accumulate/subtract into a
    // pre-existing out buffer instead of overwriting it.
    for (int bnd : {B_ZERO, B_DCT2, B_DFT}) {
        run_2d_matvec_addsub<double>(5, 6, 2, 1, {1.75, 0.9}, {0, 0}, {0, 0}, 64, bnd);
        run_2d_matvec_addsub<double>(5, 6, 2, 2, {0.3, 0.4}, {1.0, 0.7}, {0, 0}, 64, bnd);
        run_2d_matvec_addsub<double>(5, 6, 2, 3, {0.3, 0.4}, {1.0, 0.7}, {1.1, 0.8}, 64, bnd);
    }
    run_2d_matvec_addsub<float>(5, 5, 2, 2, {0.3, 0.4}, {1.0, 0.7}, {0, 0}, 32);

    // field_adddiag_/_sub, field_addkernel_/_sub: accumulate/subtract into a
    // pre-existing out buffer instead of overwriting it (jitfields op '+'/'-').
    for (int bnd : {B_ZERO, B_DCT2, B_DFT}) {
        run_2d_diag_kernel_addsub<double>(5, 6, 2, 1, {1.75, 0.9}, {0, 0}, {0, 0}, 64, bnd);
        run_2d_diag_kernel_addsub<double>(5, 6, 2, 2, {0.3, 0.4}, {1.0, 0.7}, {0, 0}, 64, bnd);
        run_2d_diag_kernel_addsub<double>(7, 7, 2, 3, {0.3, 0.4}, {1.0, 0.7}, {1.1, 0.8}, 64, bnd);
    }
    run_2d_diag_kernel_addsub<float>(5, 5, 2, 2, {0.3, 0.4}, {1.0, 0.7}, {0, 0}, 32);

    // field_matvec_rls / field_diag_rls / field_relax_rls: RLS (wc=1, shared
    // weight) and JRLS (wc=C, per-channel weight), for the absolute, membrane
    // and bending penalties. The bending-order RLS/JRLS path
    // (matvec_bending_rls/_jrls) previously had a self-adjointness bug in its
    // varying-weight coefficient math (a mis-indexed cross-term neighbour in
    // the kernels layer); fixed upstream, see fastfields-kernels#34.
    for (int bnd : {B_ZERO, B_DCT2}) {
        // absolute-only
        run_2d_matvec_rls_symmetry<double>(5, 6, 2, 1, 1, {1.75, 0.9}, {0, 0}, {0, 0}, 64, bnd);
        run_2d_matvec_rls_symmetry<double>(5, 6, 2, 2, 1, {1.75, 0.9}, {0, 0}, {0, 0}, 64, bnd);
        // membrane
        run_2d_matvec_rls_symmetry<double>(5, 6, 2, 1, 2, {0.3, 0.4}, {1.0, 0.7}, {0, 0}, 64, bnd);
        run_2d_matvec_rls_symmetry<double>(5, 6, 2, 2, 2, {0.3, 0.4}, {1.0, 0.7}, {0, 0}, 64, bnd);
        run_2d_matvec_rls_symmetry<double>(5, 6, 1, 1, 2, {0.0}, {1.0}, {0.0}, 64, bnd);
    }
    run_2d_matvec_rls_symmetry<float>(5, 5, 2, 1, 2, {0.3, 0.4}, {1.0, 0.7}, {0, 0}, 32);

    // bending order (Zero excluded -- separately-tracked OOB weight-map read
    // under that boundary, fastfields-kernels#34 finding S1; DCT2/DST2/DFT
    // are the boundaries the fix was verified against).
    for (int bnd : {B_DCT2, B_DST2, B_DFT}) {
        run_2d_matvec_rls_symmetry<double>(5, 6, 2, 1, 3, {0.3, 0.4}, {1.0, 0.7}, {1.1, 0.9}, 64, bnd);
        run_2d_matvec_rls_symmetry<double>(5, 6, 2, 2, 3, {0.3, 0.4}, {1.0, 0.7}, {1.1, 0.9}, 64, bnd);
        run_2d_matvec_rls_symmetry<double>(7, 7, 1, 1, 3, {0.0}, {0.0}, {1.0}, 64, bnd);
        run_2d_matvec_rls_symmetry<double>(4, 9, 1, 1, 3, {0.0}, {0.0}, {1.0}, 64, bnd);
    }

    // field_matvec_rls with w == 1 must collapse onto the unweighted
    // field_matvec at every order -- a magnitude check the symmetry oracle
    // above is blind to. Restricted to the boundaries whose implicit extension
    // maps a constant weight map to itself (DCT2 reflects, DFT wraps); Zero
    // would extend w by 0 and DST2 by -w, so neither is an identity case for
    // the weighted operator.
    //
    // The bending-order rows are the regression test for
    // fastfields-kernels#40 (membrane at half strength inside
    // make_kernel_bending_rls): before the fix the membrane-carrying rows below
    // were off by ~1 in absolute terms, while absolute-only and bending-only
    // already reduced exactly.
    for (int bnd : {B_DCT2, B_DFT}) {
        // absolute-only and membrane orders (already correct; guard rails)
        run_2d_matvec_rls_unit_weight<double>(5, 6, 2, 1, 1, {1.75, 0.9}, {0, 0}, {0, 0}, 64, bnd);
        run_2d_matvec_rls_unit_weight<double>(5, 6, 2, 2, 1, {1.75, 0.9}, {0, 0}, {0, 0}, 64, bnd);
        run_2d_matvec_rls_unit_weight<double>(5, 6, 2, 1, 2, {0.3, 0.4}, {1.0, 0.7}, {0, 0}, 64, bnd);
        run_2d_matvec_rls_unit_weight<double>(5, 6, 2, 2, 2, {0.3, 0.4}, {1.0, 0.7}, {0, 0}, 64, bnd);

        // bending order: absolute-only / bending-only reduced exactly even
        // before #40; the membrane-carrying rows are the ones that did not.
        run_2d_matvec_rls_unit_weight<double>(7, 7, 1, 1, 3, {1.7}, {0.0}, {0.0}, 64, bnd);
        run_2d_matvec_rls_unit_weight<double>(7, 7, 1, 1, 3, {0.0}, {0.0}, {1.1}, 64, bnd);
        run_2d_matvec_rls_unit_weight<double>(7, 7, 1, 1, 3, {0.0}, {1.0}, {0.0}, 64, bnd);
        run_2d_matvec_rls_unit_weight<double>(5, 6, 2, 1, 3, {0.3, 0.4}, {1.0, 0.7}, {1.1, 0.9}, 64, bnd);
        run_2d_matvec_rls_unit_weight<double>(5, 6, 2, 2, 3, {0.3, 0.4}, {1.0, 0.7}, {1.1, 0.9}, 64, bnd);
    }

    run_2d_diag_rls<double>(6, 7, 2, 1, 1, {1.75, 0.9}, {0, 0}, {0, 0}, 64);
    run_2d_diag_rls<double>(6, 7, 2, 2, 2, {0.3, 0.4}, {1.0, 0.7}, {0, 0}, 64);

    // A genuine per-channel weight map (wc == nc): ground-truth regression
    // test for fastfields-cpu-lib#65 (the field_rls_is_jrls predicate used
    // to have RLS/JRLS backwards, mis-routing this case to the
    // broadcast-only _jrls kernels, which silently broadcast channel 0's
    // weight to every channel). See the oracle's own doc comment for why
    // decomposing into single-channel wc==1 problems is an independent
    // ground truth rather than a self-consistency check.
    for (int bnd : {B_ZERO, B_DCT2, B_DST2, B_DFT}) {
        run_2d_rls_jrls_per_channel<double>(6, 7, 3, 1, {1.75, 0.9, 1.3},
                                            {0, 0, 0}, {0, 0, 0}, 64, bnd);
        run_2d_rls_jrls_per_channel<double>(6, 7, 2, 2, {0.3, 0.4}, {1.0, 0.7},
                                            {0, 0}, 64, bnd);
        run_2d_rls_jrls_per_channel<double>(
            7, 8, 3, 2, {0.3, 0.4, 0.2}, {1.0, 0.7, 1.2}, {0, 0, 0}, 64, bnd);
    }
    // bending order (Zero excluded, as for the symmetry/unit-weight suites
    // above -- separately-tracked OOB weight-map read at that boundary,
    // fastfields-kernels#34 finding S1).
    for (int bnd : {B_DCT2, B_DST2, B_DFT}) {
        run_2d_rls_jrls_per_channel<double>(7, 8, 2, 3, {0.3, 0.4}, {1.0, 0.7},
                                            {1.1, 0.9}, 64, bnd);
        run_2d_rls_jrls_per_channel<double>(7, 8, 3, 3, {0.3, 0.4, 0.5},
                                            {1.0, 0.7, 0.6}, {1.1, 0.9, 0.8},
                                            64, bnd);
    }
    run_2d_rls_jrls_per_channel<float>(6, 6, 3, 2, {0.3, 0.4, 0.2},
                                       {1.0, 0.7, 1.2}, {0, 0, 0}, 32);

    run_2d_relax_rls<double>(6, 7, 2, 1, 4.0, 2, {0.5, 0.3}, {1.0, 0.7}, {0, 0}, 64);
    run_2d_relax_rls<double>(6, 7, 2, 2, 4.0, 2, {0.5, 0.3}, {1.0, 0.7}, {0, 0}, 64);
    // Regression test for fastfields-cpu-impl#51: relax_bending_rls_ and
    // relax_bending_jrls_ used a `2*niter` colour-sweep loop bound (copied
    // from the membrane/patch1 sites) instead of `pow<ndim>(3)*niter`
    // (patch3, as bending's reach-2 stencil needs), which visited only 2 of
    // 3^ndim colours and left most of the field unrelaxed. Exercises both
    // the RLS (wc=1) and genuine JRLS (wc=C) weight-map paths at order=3
    // (bending); a failure here means the fix (or the underlying bug) has
    // regressed.
    run_2d_relax_rls<double>(6, 7, 2, 1, 8.0, 3, {0.3, 0.4}, {0.5, 0.6},
                             {1.0, 0.8}, 64);
    run_2d_relax_rls<double>(6, 7, 2, 2, 8.0, 3, {0.3, 0.4}, {0.5, 0.6},
                             {1.0, 0.8}, 64);
    run_2d_relax_rls_bending_sweep_visits_every_colour<double>(1, 64);
    run_2d_relax_rls_bending_sweep_visits_every_colour<double>(2, 64);

    // field_precond / field_precond_: Jacobi-type direct solve of
    // (H + diag(L)) x = grd, and in-place/out-of-place agreement.
    for (int bnd : {B_ZERO, B_DCT2, B_DFT}) {
        run_2d_precond<double>(6, 7, 2, 4.0, 1, {2.0, 1.5}, {0, 0}, {0, 0}, 64, bnd);
        run_2d_precond<double>(6, 7, 2, 4.0, 2, {0.5, 0.3}, {1.0, 0.7}, {0, 0}, 64, bnd);
        run_2d_precond<double>(6, 7, 1, 6.0, 3, {0.3}, {0.5}, {1.0}, 64, bnd);
        run_2d_precond<double>(6, 7, 2, 8.0, 3, {0.3, 0.4}, {0.5, 0.6}, {1.0, 0.8}, 64, bnd);
    }
    run_2d_precond<float>(6, 6, 2, 4.0, 2, {0.5, 0.3}, {1.0, 0.7}, {0, 0}, 32);

    // field_forward: (H + L) @ x == sym_matvec(H, x) + field_matvec(x).
    for (int bnd : {B_ZERO, B_DCT2, B_DFT}) {
        run_2d_forward<double>(6, 7, 2, 1, {2.0, 1.5}, {0, 0}, {0, 0}, 64, bnd);
        run_2d_forward<double>(6, 7, 2, 2, {0.5, 0.3}, {1.0, 0.7}, {0, 0}, 64, bnd);
        run_2d_forward<double>(6, 7, 1, 3, {0.3}, {0.5}, {1.0}, 64, bnd);
        run_2d_forward<double>(6, 7, 2, 3, {0.3, 0.4}, {0.5, 0.6}, {1.0, 0.8}, 64, bnd);
    }
    run_2d_forward<float>(6, 6, 2, 2, {0.5, 0.3}, {1.0, 0.7}, {0, 0}, 32);

    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
