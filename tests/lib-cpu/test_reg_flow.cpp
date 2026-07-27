// CPU unit tests for the reg_flow module.
//
// Exercises ff::cpu::flow_matvec / flow_diag against hand-written references:
//   * absolute-only matvec == elementwise scaling by `absolute` (exact)
//   * membrane-only matvec == membrane * discrete negative Laplacian (exact,
//     with an explicit boundary-condition reference for Zero and DCT2)
//   * diag == matvec applied to unit vectors (per position/channel)
//
// Build (from fastfields-cpu-lib):
//   clang++ -std=c++11 -O2 -ferror-limit=5 -I. tests/test_reg_flow.cpp reg_flow.cpp -o build/test_reg_flow
//   ./build/test_reg_flow

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "dlpack.h"
#include "reg_flow.h"

namespace {

// bound enum values (see kernels/bounds.h)
enum { B_ZERO = 0, B_DCT1 = 2, B_DCT2 = 3, B_DST1 = 4, B_DST2 = 5, B_DFT = 6 };

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

// 1D neighbour value under a given boundary condition.
double neigh_1d(const std::vector<double>& f, int64_t idx, int bound)
{
    int64_t n = (int64_t)f.size();
    if (idx >= 0 && idx < n) return f[idx];
    if (bound == B_ZERO) return 0.0;            // sign 0 -> contributes 0
    // DCT2: reflect w.r.t. edge, sign +1
    int64_t r = idx;
    if (r < 0)  r = -r - 1;
    if (r >= n) r = 2*n - r - 1;
    if (r < 0)  r = 0;
    if (r >= n) r = n - 1;
    return f[r];
}

// ---------------- 1D flow (C == 1) ----------------

template <typename scalar_t>
void run_1d(int64_t N, double absolute, double membrane, int bound, uint8_t bits)
{
    std::vector<double> ref(N);
    std::vector<scalar_t> inp(N), out(N, scalar_t(-12345));
    for (int64_t i = 0; i < N; ++i) {
        double v = std::sin(0.3 * i + 1.0) + 0.5 * i;
        inp[i] = (scalar_t)v;
        ref[i] = v;
    }

    std::vector<int64_t> shape = {N, 1};   // (spatial, C=1)
    std::vector<int64_t> str   = contiguous_strides(shape);
    DLTensor tin  = make_cpu_tensor(inp.data(),  shape, str, bits);
    DLTensor tout = make_cpu_tensor(out.data(), shape, str, bits);

    ff::cpu::flow_matvec(tout, tin, nullptr, absolute, membrane, 0.0,
                         0.0, 0.0, (int8_t)bound, /*ndim*/1, 0);

    for (int64_t x = 0; x < N; ++x) {
        double c  = ref[x];
        double L  = neigh_1d(ref, x-1, bound);
        double R  = neigh_1d(ref, x+1, bound);
        double expect = absolute * c + membrane * (2*c - L - R);
        check_close((double)out[x], expect, "flow1d_matvec");
    }
}

// diag vs matvec on unit vectors (1D)
template <typename scalar_t>
void run_1d_diag(int64_t N, double absolute, double membrane, int bound, uint8_t bits)
{
    std::vector<int64_t> shape = {N, 1};
    std::vector<int64_t> str   = contiguous_strides(shape);

    std::vector<scalar_t> diag(N, 0);
    DLTensor td = make_cpu_tensor(diag.data(), shape, str, bits);
    ff::cpu::flow_diag(td, nullptr, absolute, membrane, 0.0, 0.0, 0.0, (int8_t)bound, 1, 0);

    // NOTE: jitfields' diag_membrane is a preconditioner diagonal that uses a
    // sign-based boundary term (kernel[0] - kernel[1]*(sign(x-1)+sign(x+1))).
    // It equals the exact operator diagonal e_x^T A e_x only in the interior;
    // at the boundary it intentionally differs. We therefore only assert the
    // interior entries against matvec-on-unit-vectors.
    for (int64_t x = 1; x < N - 1; ++x) {
        std::vector<scalar_t> e(N, 0), o(N, 0);
        e[x] = 1;
        DLTensor te = make_cpu_tensor(e.data(), shape, str, bits);
        DLTensor to = make_cpu_tensor(o.data(), shape, str, bits);
        ff::cpu::flow_matvec(to, te, nullptr, absolute, membrane, 0.0, 0.0, 0.0, (int8_t)bound, 1, 0);
        check_close((double)diag[x], (double)o[x], "flow1d_diag_interior");
    }
}

// ---------------- 2D flow (C == 2) ----------------
// absolute-only: exact elementwise scaling on both channels.
template <typename scalar_t>
void run_2d_absolute(int64_t H, int64_t W, double absolute, uint8_t bits)
{
    std::vector<int64_t> shape = {H, W, 2};
    std::vector<int64_t> str   = contiguous_strides(shape);
    int64_t numel = H * W * 2;
    std::vector<scalar_t> inp(numel), out(numel, scalar_t(-999));
    for (int64_t i = 0; i < numel; ++i) inp[i] = (scalar_t)(0.1 * i - 3.0);

    DLTensor tin  = make_cpu_tensor(inp.data(), shape, str, bits);
    DLTensor tout = make_cpu_tensor(out.data(), shape, str, bits);
    ff::cpu::flow_matvec(tout, tin, nullptr, absolute, 0.0, 0.0, 0.0, 0.0, (int8_t)B_DCT2, 2, 0);

    for (int64_t i = 0; i < numel; ++i)
        check_close((double)out[i], absolute * (double)inp[i], "flow2d_abs");
}

// 1D flow bending interior == bending * biharmonic stencil [1,-4,6,-4,1].
template <typename scalar_t>
void run_1d_bending(int64_t N, double bending, uint8_t bits)
{
    std::vector<int64_t> shape = {N, 1};
    std::vector<int64_t> str   = contiguous_strides(shape);
    std::vector<double> f(N);
    std::vector<scalar_t> inp(N), out(N, scalar_t(-1));
    for (int64_t x = 0; x < N; ++x) { double v = std::cos(0.27*x + 0.4); inp[x]=(scalar_t)v; f[x]=v; }
    DLTensor ti = make_cpu_tensor(inp.data(), shape, str, bits);
    DLTensor to = make_cpu_tensor(out.data(), shape, str, bits);
    ff::cpu::flow_matvec(to, ti, nullptr, 0.0, 0.0, bending, 0.0, 0.0, (int8_t)B_DCT2, 1, 0);
    for (int64_t x = 2; x < N-2; ++x) {
        double e = bending * (f[x-2] - 4*f[x-1] + 6*f[x] - 4*f[x+1] + f[x+2]);
        check_close((double)out[x], e, "flow1d_bending");
    }
}

// ---------------- 3D flow (C == 3) ----------------
// membrane interior == m * (6*center - 6 face-neighbours), per channel.
template <typename scalar_t>
void run_3d_membrane(int64_t N, double membrane, uint8_t bits)
{
    const int64_t C = 3;
    std::vector<int64_t> shape = {N, N, N, C};
    std::vector<int64_t> str   = contiguous_strides(shape);
    int64_t numel = N * N * N * C;
    std::vector<scalar_t> inp(numel), out(numel, scalar_t(-1));
    auto idx = [&](int64_t x,int64_t y,int64_t z,int64_t c){ return ((x*N+y)*N+z)*C+c; };
    for (int64_t i = 0; i < numel; ++i) inp[i] = (scalar_t)std::sin(0.21*i + 0.5);

    DLTensor ti = make_cpu_tensor(inp.data(), shape, str, bits);
    DLTensor to = make_cpu_tensor(out.data(), shape, str, bits);
    ff::cpu::flow_matvec(to, ti, nullptr, 0.0, membrane, 0.0, 0.0, 0.0, (int8_t)B_DCT2, 3, 0);

    for (int64_t x = 1; x < N-1; ++x)
    for (int64_t y = 1; y < N-1; ++y)
    for (int64_t z = 1; z < N-1; ++z)
    for (int64_t c = 0; c < C; ++c) {
        double ctr = (double)inp[idx(x,y,z,c)];
        double s = (double)inp[idx(x-1,y,z,c)] + (double)inp[idx(x+1,y,z,c)]
                 + (double)inp[idx(x,y-1,z,c)] + (double)inp[idx(x,y+1,z,c)]
                 + (double)inp[idx(x,y,z-1,c)] + (double)inp[idx(x,y,z+1,c)];
        check_close((double)out[idx(x,y,z,c)], membrane*(6*ctr - s), "flow3d_membrane");
    }
}

// --- B4: negative / validation tests --------------------------------------
// Bad dtype: float16 must throw at the dtype dispatch, not silently no-op.
// (ndim=1 -> C must equal ndim == 1.)
void test_bad_dtype_throws()
{
    const int64_t N = 6;
    std::vector<uint16_t> inp(N,0), out(N,0);              // float16 payload, C=1
    std::vector<int64_t> sh={N,1}, st=contiguous_strides(sh);
    DLTensor tin =make_cpu_tensor(inp.data(),sh,st,16);
    DLTensor tout=make_cpu_tensor(out.data(),sh,st,16);
    bool threw = false;
    try { ff::cpu::flow_matvec(tout, tin, nullptr, 1.0, 0.0, 0.0, 0.0, 0.0, (int8_t)B_DCT2, 1, 0); }
    catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [reg_flow.bad_dtype_throws]\n"); }
}

// Shape mismatch: out and inp with different spatial extents must throw.
void test_shape_mismatch_throws()
{
    const int64_t N = 6;
    std::vector<double> inp(N+1,0), out(N,0);              // differing spatial dim, C=1
    std::vector<int64_t> shi={N+1,1}, sti=contiguous_strides(shi);
    std::vector<int64_t> sho={N,1},   sto=contiguous_strides(sho);
    DLTensor tin =make_cpu_tensor(inp.data(),shi,sti,64);
    DLTensor tout=make_cpu_tensor(out.data(),sho,sto,64);
    bool threw = false;
    try { ff::cpu::flow_matvec(tout, tin, nullptr, 1.0, 0.0, 0.0, 0.0, 0.0, (int8_t)B_DCT2, 1, 0); }
    catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [reg_flow.shape_mismatch_throws]\n"); }
}

// ---------------- 2D flow linear-elastic (Lamé: shears/div) ----------------
// The regulariser operator L is symmetric, so <L x, y> == <x, L y> for any
// x, y. This is a strong check of the coupled shears/div stencil (matvec_all)
// without re-deriving the exact kernel. Summed over ALL elements, symmetry is
// boundary-condition-independent.
template <typename scalar_t>
void run_2d_lame_symmetry(int64_t H, int64_t W, double absolute,
                          double membrane, double bending, double shears,
                          double div, uint8_t bits, int bound = B_DCT2)
{
    std::vector<int64_t> shape = {H, W, 2};
    std::vector<int64_t> str   = contiguous_strides(shape);
    int64_t numel = H * W * 2;
    std::vector<scalar_t> x(numel), y(numel), Lx(numel, 0), Ly(numel, 0);
    for (int64_t i = 0; i < numel; ++i) {
        x[i] = (scalar_t)std::sin(0.3 * i + 0.1);
        y[i] = (scalar_t)std::cos(0.2 * i + 0.7);
    }
    DLTensor tx  = make_cpu_tensor(x.data(),  shape, str, bits);
    DLTensor ty  = make_cpu_tensor(y.data(),  shape, str, bits);
    DLTensor tLx = make_cpu_tensor(Lx.data(), shape, str, bits);
    DLTensor tLy = make_cpu_tensor(Ly.data(), shape, str, bits);
    ff::cpu::flow_matvec(tLx, tx, nullptr, absolute, membrane, bending,
                         shears, div, (int8_t)bound, 2, 0);
    ff::cpu::flow_matvec(tLy, ty, nullptr, absolute, membrane, bending,
                         shears, div, (int8_t)bound, 2, 0);
    double lhs = 0, rhs = 0;
    for (int64_t i = 0; i < numel; ++i) {
        lhs += (double)Lx[i] * (double)y[i];
        rhs += (double)x[i]  * (double)Ly[i];
    }
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "flow2d_lame_symmetry[H=%lld W=%lld a=%g m=%g b=%g s=%g d=%g bound=%d]",
            (long long)H, (long long)W, absolute, membrane, bending, shears, div, bound);
        check_close(lhs, rhs, buf);
    }
}

// The lame path must reproduce the diagonal: diag_all[x,c] equals the (x,c)
// entry of L, i.e. matvec_all applied to the unit vector e_{x,c}, in the
// interior (the preconditioner diagonal differs only at the boundary).
template <typename scalar_t>
void run_2d_lame_diag(int64_t H, int64_t W, double absolute, double membrane,
                      double shears, double div, uint8_t bits)
{
    std::vector<int64_t> shape = {H, W, 2};
    std::vector<int64_t> str   = contiguous_strides(shape);
    int64_t numel = H * W * 2;
    std::vector<scalar_t> d(numel, 0);
    DLTensor td = make_cpu_tensor(d.data(), shape, str, bits);
    ff::cpu::flow_diag(td, nullptr, absolute, membrane, 0.0, shears, div,
                       (int8_t)B_DCT2, 2, 0);
    auto idx = [&](int64_t i, int64_t j, int64_t c) {
        return (i * W + j) * 2 + c;
    };
    for (int64_t i = 1; i < H - 1; ++i)
    for (int64_t j = 1; j < W - 1; ++j)
    for (int64_t c = 0; c < 2; ++c) {
        std::vector<scalar_t> e(numel, 0), o(numel, 0);
        e[idx(i, j, c)] = 1;
        DLTensor te = make_cpu_tensor(e.data(), shape, str, bits);
        DLTensor to = make_cpu_tensor(o.data(), shape, str, bits);
        ff::cpu::flow_matvec(to, te, nullptr, absolute, membrane, 0.0, shears,
                             div, (int8_t)B_DCT2, 2, 0);
        check_close((double)d[idx(i, j, c)], (double)o[idx(i, j, c)],
                    "flow2d_lame_diag_interior");
    }
}

// 3D analogue of run_2d_lame_symmetry: the full linear-elastic flow operator
// must be self-adjoint in 3D too (all three cross-channel coupling blocks are
// transposes of each other).
template <typename scalar_t>
void run_3d_lame_symmetry(int64_t D, int64_t H, int64_t W, double absolute,
                          double membrane, double bending, double shears,
                          double div, uint8_t bits, int bound = B_DCT2)
{
    std::vector<int64_t> shape = {D, H, W, 3};
    std::vector<int64_t> str   = contiguous_strides(shape);
    int64_t numel = D * H * W * 3;
    std::vector<scalar_t> x(numel), y(numel), Lx(numel, 0), Ly(numel, 0);
    for (int64_t i = 0; i < numel; ++i) {
        x[i] = (scalar_t)std::sin(0.3 * i + 0.1);
        y[i] = (scalar_t)std::cos(0.2 * i + 0.7);
    }
    DLTensor tx  = make_cpu_tensor(x.data(),  shape, str, bits);
    DLTensor ty  = make_cpu_tensor(y.data(),  shape, str, bits);
    DLTensor tLx = make_cpu_tensor(Lx.data(), shape, str, bits);
    DLTensor tLy = make_cpu_tensor(Ly.data(), shape, str, bits);
    ff::cpu::flow_matvec(tLx, tx, nullptr, absolute, membrane, bending,
                         shears, div, (int8_t)bound, 3, 0);
    ff::cpu::flow_matvec(tLy, ty, nullptr, absolute, membrane, bending,
                         shears, div, (int8_t)bound, 3, 0);
    double lhs = 0, rhs = 0;
    for (int64_t i = 0; i < numel; ++i) {
        lhs += (double)Lx[i] * (double)y[i];
        rhs += (double)x[i]  * (double)Ly[i];
    }
    char buf[144];
    std::snprintf(buf, sizeof(buf),
        "flow3d_lame_symmetry[D=%lld H=%lld W=%lld a=%g m=%g b=%g s=%g d=%g bound=%d]",
        (long long)D, (long long)H, (long long)W, absolute, membrane, bending,
        shears, div, bound);
    check_close(lhs, rhs, buf);
}

// flow_relax must drive the warm-started solution towards solving
// (H + L) x = g. With a strongly diagonal Hessian H = hdiag*I the red-black
// Gauss-Seidel sweeps converge; we check the residual r = H x + L x - g is
// small after enough iterations. The check independently recomputes L x via
// flow_matvec, so it does not assume anything about the relaxer internals.
template <typename scalar_t>
void run_2d_relax(int64_t Hgt, int64_t W, double hdiag, double absolute,
                  double membrane, double bending, double shears, double div,
                  uint8_t bits, int bound = B_DCT2, int niter = 150)
{
    std::vector<int64_t> fshape = {Hgt, W, 2};      // flow / grad: ndim=2
    std::vector<int64_t> hshape = {Hgt, W, 3};      // sym Hessian: 2*(2+1)/2
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    std::vector<int64_t> hstr = contiguous_strides(hshape);
    int64_t fnum = Hgt * W * 2, hnum = Hgt * W * 3;

    std::vector<scalar_t> sol(fnum, 0), grd(fnum), hes(hnum, 0);
    for (int64_t i = 0; i < fnum; ++i)
        grd[i] = (scalar_t)std::sin(0.4 * i + 0.2);
    // diagonal-only Hessian: entries 0,1 are the diagonal, entry 2 the offdiag
    for (int64_t p = 0; p < Hgt * W; ++p) {
        hes[p * 3 + 0] = (scalar_t)hdiag;
        hes[p * 3 + 1] = (scalar_t)hdiag;
        hes[p * 3 + 2] = 0;
    }

    DLTensor tsol = make_cpu_tensor(sol.data(), fshape, fstr, bits);
    DLTensor thes = make_cpu_tensor(hes.data(), hshape, hstr, bits);
    DLTensor tgrd = make_cpu_tensor(grd.data(), fshape, fstr, bits);
    ff::cpu::flow_relax(tsol, thes, tgrd, nullptr, absolute, membrane, bending,
                        shears, div, (int8_t)bound, 2, niter, 0);

    // residual: H x + L x - g
    std::vector<scalar_t> Lx(fnum, 0);
    DLTensor tLx = make_cpu_tensor(Lx.data(), fshape, fstr, bits);
    ff::cpu::flow_matvec(tLx, tsol, nullptr, absolute, membrane, bending,
                         shears, div, (int8_t)bound, 2, 0);
    double res = 0, nrm = 0;
    for (int64_t i = 0; i < fnum; ++i) {
        double r = hdiag * (double)sol[i] + (double)Lx[i] - (double)grd[i];
        res += r * r;
        nrm += (double)grd[i] * (double)grd[i];
    }
    double rel = std::sqrt(res / nrm);
    char buf[128];
    std::snprintf(buf, sizeof(buf),
        "flow2d_relax_residual[m=%g b=%g s=%g d=%g bound=%d] rel=%.2e",
        membrane, bending, shears, div, bound, rel);
    check_close(rel, 0.0, buf, 2e-3);
}

} // namespace

int main()
{
    std::printf("reg_flow module CPU tests\n");
    test_bad_dtype_throws();
    test_shape_mismatch_throws();

    // absolute-only (exact scaling), 1D, both dtypes
    run_1d<float >(9, 2.5, 0.0, B_ZERO, 32);
    run_1d<double>(9, 2.5, 0.0, B_DCT2, 64);

    // membrane-only (negative Laplacian), 1D, Zero + DCT2 boundaries
    run_1d<float >(11, 0.0, 1.0, B_ZERO, 32);
    run_1d<double>(11, 0.0, 1.0, B_DCT2, 64);
    run_1d<double>(11, 0.0, 3.0, B_ZERO, 64);

    // combined absolute + membrane
    run_1d<double>(13, 0.7, 1.3, B_DCT2, 64);

    // diagonal consistency
    run_1d_diag<double>(11, 0.7, 1.3, B_DCT2, 64);
    run_1d_diag<double>(11, 0.0, 1.0, B_ZERO, 64);
    run_1d_diag<float >(9,  2.5, 0.0, B_ZERO, 32);

    // 2D absolute
    run_2d_absolute<double>(4, 5, 1.75, 64);
    run_2d_absolute<float >(3, 6, 0.5,  32);

    // 1D bending (interior biharmonic stencil)
    run_1d_bending<double>(11, 1.0, 64);
    run_1d_bending<double>(9,  2.5, 64);

    // 3D membrane (interior negative Laplacian)
    run_3d_membrane<double>(5, 1.0, 64);
    run_3d_membrane<double>(4, 2.5, 64);

    // 2D linear-elastic (shears/div) — operator symmetry across parameter
    // combinations, exercising the matvec_all / diag_all path.
    // control: membrane operator IS symmetric under DCT2
    run_2d_lame_symmetry<double>(5, 6, 0.0, 1.0, 0.0, 0.0, 0.0, 64, B_DCT2);
    // lame under periodic (DFT) — circulant, so exactly symmetric if the
    // interior stencil is right; isolates any DCT2 boundary asymmetry.
    run_2d_lame_symmetry<double>(6, 6, 0.0, 0.0, 0.0, 1.0, 0.0, 64, B_DFT);
    run_2d_lame_symmetry<double>(6, 6, 0.0, 0.0, 0.0, 0.0, 1.0, 64, B_DFT);
    run_2d_lame_symmetry<double>(6, 6, 0.0, 0.0, 0.0, 1.3, 0.7, 64, B_DFT);
    // lame under DCT2 (the previously-untested combos)
    run_2d_lame_symmetry<double>(5, 6, 0.0, 0.0, 0.0, 1.0, 0.0, 64);  // shears
    run_2d_lame_symmetry<double>(5, 6, 0.0, 0.0, 0.0, 0.0, 1.0, 64);  // div
    run_2d_lame_symmetry<double>(5, 6, 0.0, 0.0, 0.0, 1.3, 0.7, 64);  // both
    run_2d_lame_symmetry<double>(6, 5, 0.5, 0.9, 0.4, 1.3, 0.7, 64);  // all 5
    run_2d_lame_symmetry<float >(5, 5, 0.0, 0.0, 0.0, 1.0, 1.0, 32);
    // Other boundaries whose adjoint the transpose map handles: DST2
    // (Dirichlet, transpose<->DCT2, same reflect_N index) and Zero.
    run_2d_lame_symmetry<double>(5, 6, 0.0, 0.0, 0.0, 1.3, 0.7, 64, B_DST2);
    run_2d_lame_symmetry<double>(6, 5, 0.5, 0.9, 0.4, 1.3, 0.7, 64, B_DST2);
    run_2d_lame_symmetry<double>(5, 6, 0.0, 0.0, 0.0, 1.3, 0.7, 64, B_ZERO);
    // NOTE: DCT1/DST1 (whole-sample reflect, reflect_N-/+1) are intentionally
    // NOT asserted self-adjoint. Their forward and adjoint use different index
    // centres, so a single companion-boundary read cannot reproduce D^T; the
    // lame operator is not SPD under DCT1/DST1. This matches upstream and is a
    // documented limitation (fastfields-lib#26): flow regularisation uses
    // DCT2/Neumann or DFT in practice.
    // diagonal consistency for the lame path
    run_2d_lame_diag<double>(6, 6, 0.3, 0.5, 1.2, 0.8, 64);
    run_2d_lame_diag<double>(6, 6, 0.0, 0.0, 1.0, 0.0, 64);

    // 3D linear-elastic operator symmetry (exercises the three cross-channel
    // coupling blocks of matvec_all / matvec_lame).
    // periodic (DFT) — isolates the interior stencil from any boundary effect
    run_3d_lame_symmetry<double>(4, 4, 4, 0.0, 0.0, 0.0, 1.0, 0.0, 64, B_DFT);
    run_3d_lame_symmetry<double>(4, 4, 4, 0.0, 0.0, 0.0, 0.0, 1.0, 64, B_DFT);
    run_3d_lame_symmetry<double>(4, 4, 4, 0.0, 0.0, 0.0, 1.3, 0.7, 64, B_DFT);
    // DCT2 (Neumann) — the boundary asymmetry the transpose fix addresses
    run_3d_lame_symmetry<double>(4, 5, 4, 0.0, 0.0, 0.0, 1.0, 0.0, 64);  // shears
    run_3d_lame_symmetry<double>(4, 5, 4, 0.0, 0.0, 0.0, 0.0, 1.0, 64);  // div
    run_3d_lame_symmetry<double>(5, 4, 4, 0.0, 0.0, 0.0, 1.3, 0.7, 64);  // both
    run_3d_lame_symmetry<double>(4, 4, 5, 0.5, 0.9, 0.4, 1.3, 0.7, 64);  // all 5
    run_3d_lame_symmetry<float >(4, 4, 4, 0.0, 0.0, 0.0, 1.0, 1.0, 32);
    // DST2 (3D three-way coupling). DCT1/DST1 excluded — see 2D note above.
    run_3d_lame_symmetry<double>(4, 5, 4, 0.5, 0.9, 0.4, 1.3, 0.7, 64, B_DST2);

    // flow_relax: relaxation drives (H + L) x -> g (residual check).
    run_2d_relax<double>(6, 7, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0, 64);  // membrane
    run_2d_relax<double>(6, 7, 6.0, 0.5, 1.0, 0.0, 0.0, 0.0, 64);  // abs+mem
    run_2d_relax<double>(6, 7, 6.0, 0.0, 0.0, 1.0, 0.0, 0.0, 64);  // bending
    run_2d_relax<double>(6, 7, 6.0, 0.0, 0.0, 0.0, 1.0, 0.5, 64);  // lame
    run_2d_relax<double>(6, 7, 8.0, 0.3, 0.7, 0.4, 1.0, 0.5, 64);  // all

    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
