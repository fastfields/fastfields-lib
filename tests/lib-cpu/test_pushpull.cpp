// CPU unit tests for the pushpull module.
//
// Exercises ff::cpu::{pull,push,count,grad} through the full stack
// (dispatch -> impl -> kernels). Checks:
//   * pull at integer nodes with linear spline == gather (identity)
//   * pull at a midpoint == linear interpolation of neighbours
//   * push is the numerical adjoint of pull: <pull(x),g> == <x,push(g)>
//   * count == push of ones
//   * grad of a linear ramp == constant slope
//
// Build (from fastfields-cpu-lib):
//   clang++ -std=c++11 -O2 -I. tests/test_pushpull.cpp pushpull.cpp -o build/test_pushpull
//   ./build/test_pushpull

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
#include "dlpack.h"
#include "pushpull.h"

namespace {

// Spline orders / bounds (mirror spline_t / bound_t enum values).
constexpr int8_t NEAREST = 0, LINEAR = 1, QUADRATIC = 2, CUBIC = 3;
constexpr int8_t DCT2 = 3;   // bound value (also == CUBIC spline value)
// All eight bound conditions, for the per-bound sweep at the reference orders.
constexpr int8_t ALL_BOUNDS[8] = {0, 1, 2, 3, 4, 5, 6, 7};

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

int g_failures = 0, g_checks = 0;

void check_close(double a, double b, const char* what, double tol = 1e-4)
{
    ++g_checks;
    double diff  = std::fabs(a - b);
    double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    if (diff / scale > tol) {
        ++g_failures;
        std::printf("  MISMATCH [%s]: got %.6g expected %.6g\n", what, a, b);
    }
}

// The test bodies are templated on scalar_t/bits/tol so both double (bits=64)
// and float (bits=32, B1: the dominant warp/resample ML dtype, previously never
// instantiated) run through the full dispatch -> impl -> kernels stack.

// ---- 1D linear pull == identity at integer nodes -------------------
template <typename T>
void test_pull_identity_1d(uint8_t bits, double tol)
{
    const int64_t N = 6, C = 2;
    std::vector<T> inp(N * C);
    for (int64_t i = 0; i < N * C; ++i) inp[i] = (T)(0.5 + i);

    // grid: sample every integer node 0..N-1
    std::vector<T> grid(N * 1);
    for (int64_t i = 0; i < N; ++i) grid[i] = (T)i;
    std::vector<T> out(N * C, (T)-999.0);

    std::vector<int64_t> is = {N, C},  iss = contiguous_strides(is);
    std::vector<int64_t> gs = {N, 1},  gss = contiguous_strides(gs);
    std::vector<int64_t> os = {N, C},  oss = contiguous_strides(os);
    DLTensor it = make_cpu_tensor(inp.data(),  is, iss, bits);
    DLTensor gt = make_cpu_tensor(grid.data(), gs, gss, bits);
    DLTensor ot = make_cpu_tensor(out.data(),  os, oss, bits);

    ff::cpu::pull(ot, it, gt, LINEAR, DCT2, 1, 0);
    for (int64_t i = 0; i < N * C; ++i) check_close((double)out[i], (double)inp[i], "pull_id_1d", tol);
}

// ---- 1D linear pull midpoint == average ----------------------------
template <typename T>
void test_pull_midpoint_1d(uint8_t bits, double tol)
{
    const int64_t N = 4, C = 1;
    std::vector<T> inp = {(T)1.0, (T)3.0, (T)7.0, (T)10.0};
    std::vector<T> grid = {(T)0.5, (T)1.5, (T)2.25};
    const int64_t M = grid.size();
    std::vector<T> out(M * C, (T)0.0);

    std::vector<int64_t> is = {N, C}, iss = contiguous_strides(is);
    std::vector<int64_t> gs = {M, 1}, gss = contiguous_strides(gs);
    std::vector<int64_t> os = {M, C}, oss = contiguous_strides(os);
    DLTensor it = make_cpu_tensor(inp.data(),  is, iss, bits);
    DLTensor gt = make_cpu_tensor(grid.data(), gs, gss, bits);
    DLTensor ot = make_cpu_tensor(out.data(),  os, oss, bits);

    ff::cpu::pull(ot, it, gt, LINEAR, DCT2, 1, 0);
    check_close((double)out[0], 0.5 * 1.0 + 0.5 * 3.0,  "pull_mid_1d", tol);
    check_close((double)out[1], 0.5 * 3.0 + 0.5 * 7.0,  "pull_mid_1d", tol);
    check_close((double)out[2], 0.75 * 7.0 + 0.25 * 10.0, "pull_mid_1d", tol);
}

// ---- push is the adjoint of pull (1D and 2D) -----------------------
// Returns |<pull(inp),g> - <inp,push(g)>|.
template <typename T>
double adjoint_residual_1d(int8_t order, int8_t bound, unsigned seed, uint8_t bits)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> uc(0.0, 5.0);   // in-range coords into N=6

    const int64_t N = 6, C = 3, M = 9;
    std::vector<T> inp(N * C), g(M * C);
    for (auto& v : inp) v = (T)u(rng);
    for (auto& v : g)   v = (T)u(rng);
    std::vector<T> grid(M);
    for (auto& v : grid) v = (T)uc(rng);

    std::vector<T> pulled(M * C, (T)0.0), pushed(N * C, (T)0.0);

    std::vector<int64_t> is = {N, C}, iss = contiguous_strides(is);
    std::vector<int64_t> gs = {M, 1}, gss = contiguous_strides(gs);
    std::vector<int64_t> ms = {M, C}, mss = contiguous_strides(ms);
    DLTensor it = make_cpu_tensor(inp.data(),    is, iss, bits);
    DLTensor gt = make_cpu_tensor(grid.data(),   gs, gss, bits);
    DLTensor po = make_cpu_tensor(pulled.data(), ms, mss, bits);
    DLTensor go = make_cpu_tensor(g.data(),      ms, mss, bits);
    DLTensor so = make_cpu_tensor(pushed.data(), is, iss, bits);

    ff::cpu::pull(po, it, gt, order, bound, 1, 0);
    ff::cpu::push(so, go, gt, order, bound, 1, 0);   // pushed pre-zeroed

    double lhs = 0.0, rhs = 0.0;
    for (int64_t i = 0; i < M * C; ++i) lhs += (double)pulled[i] * (double)g[i];
    for (int64_t i = 0; i < N * C; ++i) rhs += (double)inp[i] * (double)pushed[i];
    return std::fabs(lhs - rhs);
}

template <typename T>
double adjoint_residual_2d(int8_t order, int8_t bound, unsigned seed, uint8_t bits)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> ux(0.0, 3.0);   // into Nx=4
    std::uniform_real_distribution<double> uy(0.0, 4.0);   // into Ny=5

    const int64_t Nx = 4, Ny = 5, C = 2, Mx = 3, My = 3;
    std::vector<T> inp(Nx * Ny * C), g(Mx * My * C);
    for (auto& v : inp) v = (T)u(rng);
    for (auto& v : g)   v = (T)u(rng);
    std::vector<T> grid(Mx * My * 2);
    for (int64_t k = 0; k < Mx * My; ++k) { grid[2*k] = (T)ux(rng); grid[2*k+1] = (T)uy(rng); }

    std::vector<T> pulled(Mx * My * C, (T)0.0), pushed(Nx * Ny * C, (T)0.0);

    std::vector<int64_t> is = {Nx, Ny, C}, iss = contiguous_strides(is);
    std::vector<int64_t> gs = {Mx, My, 2}, gss = contiguous_strides(gs);
    std::vector<int64_t> ms = {Mx, My, C}, mss = contiguous_strides(ms);
    DLTensor it = make_cpu_tensor(inp.data(),    is, iss, bits);
    DLTensor gt = make_cpu_tensor(grid.data(),   gs, gss, bits);
    DLTensor po = make_cpu_tensor(pulled.data(), ms, mss, bits);
    DLTensor go = make_cpu_tensor(g.data(),      ms, mss, bits);
    DLTensor so = make_cpu_tensor(pushed.data(), is, iss, bits);

    ff::cpu::pull(po, it, gt, order, bound, 1, 0);
    ff::cpu::push(so, go, gt, order, bound, 1, 0);

    double lhs = 0.0, rhs = 0.0;
    for (size_t i = 0; i < pulled.size(); ++i) lhs += (double)pulled[i] * (double)g[i];
    for (size_t i = 0; i < pushed.size(); ++i) rhs += (double)inp[i] * (double)pushed[i];
    return std::fabs(lhs - rhs);
}

// ---- count == push of ones -----------------------------------------
template <typename T>
void test_count_equals_push_ones_1d(int8_t order, unsigned seed, uint8_t bits, double tol)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uc(0.0, 5.0);

    const int64_t N = 6, M = 10;
    std::vector<T> grid(M);
    for (auto& v : grid) v = (T)uc(rng);
    std::vector<T> ones(M * 1, (T)1.0);
    std::vector<T> pushed(N * 1, (T)0.0), counted(N * 1, (T)0.0);

    std::vector<int64_t> gs = {M, 1}, gss = contiguous_strides(gs);
    std::vector<int64_t> ms = {M, 1}, mss = contiguous_strides(ms);
    std::vector<int64_t> ns = {N, 1}, nss = contiguous_strides(ns);
    DLTensor gt = make_cpu_tensor(grid.data(),    gs, gss, bits);
    DLTensor od = make_cpu_tensor(ones.data(),    ms, mss, bits);
    DLTensor so = make_cpu_tensor(pushed.data(),  ns, nss, bits);
    DLTensor co = make_cpu_tensor(counted.data(), ns, nss, bits);

    ff::cpu::push (so, od, gt, order, DCT2, 1, 0);
    ff::cpu::count(co,     gt, order, DCT2, 1, 0);
    for (int64_t i = 0; i < N; ++i) check_close((double)counted[i], (double)pushed[i], "count_vs_push", tol);
}

// ---- grad of a linear ramp == constant slope -----------------------
template <typename T>
void test_grad_ramp_1d(uint8_t bits, double tol)
{
    // inp[i] = a*i + b -> spatial gradient == a everywhere (linear spline)
    const int64_t N = 6, C = 1;
    const double a = 2.5, b = -1.0;
    std::vector<T> inp(N);
    for (int64_t i = 0; i < N; ++i) inp[i] = (T)(a * i + b);

    std::vector<T> grid = {(T)0.5, (T)1.5, (T)2.5, (T)3.5};
    const int64_t M = grid.size();
    std::vector<T> out(M * C * 1, (T)0.0);   // (M, C, D=1)

    std::vector<int64_t> is = {N, C},    iss = contiguous_strides(is);
    std::vector<int64_t> gs = {M, 1},    gss = contiguous_strides(gs);
    std::vector<int64_t> os = {M, C, 1}, oss = contiguous_strides(os);
    DLTensor it = make_cpu_tensor(inp.data(),  is, iss, bits);
    DLTensor gt = make_cpu_tensor(grid.data(), gs, gss, bits);
    DLTensor ot = make_cpu_tensor(out.data(),  os, oss, bits);

    ff::cpu::grad(ot, it, gt, LINEAR, DCT2, 1, false, 0);
    for (int64_t i = 0; i < M; ++i) check_close((double)out[i], a, "grad_ramp_1d", tol);
}

// B2: 64-bit index + non-contiguous stride path. A leading batch dim of size 2
// with an inp stride >= INT32_MAX makes canUse32BitIndexMath return false (it
// keys on the max element offset), forcing the int64_t offset_t instantiation
// and a strided (non-contiguous) inp read -- exactly the class of the
// already-fixed "2D pull used size[1] for the x-axis" bug. The reference is the
// same pull on a small contiguous tensor. The big buffer is lazily allocated
// (only the 2 touched planes commit pages); if the virtual allocation is
// refused we skip. float keeps it to ~8.6 GB virtual.
void test_inflated_stride()
{
    const int64_t B = 2, N = 6, C = 2, M = 5;
    const int64_t P = (int64_t)1 << 31;                 // 2147483648 > INT32_MAX

    std::mt19937 rng(777);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> uc(0.0, 5.0);
    std::vector<float> inp(B * N * C), grid(B * M);
    for (auto& v : inp)  v = (float)u(rng);
    for (auto& v : grid) v = (float)uc(rng);

    std::vector<int64_t> is = {B, N, C}, iss = contiguous_strides(is);
    std::vector<int64_t> gs = {B, M, 1}, gss = contiguous_strides(gs);
    std::vector<int64_t> os = {B, M, C}, oss = contiguous_strides(os);

    // Reference: fully contiguous.
    std::vector<float> out_ref(B * M * C, 0.0f);
    {
        DLTensor it = make_cpu_tensor(inp.data(),     is, iss, 32);
        DLTensor gt = make_cpu_tensor(grid.data(),    gs, gss, 32);
        DLTensor ot = make_cpu_tensor(out_ref.data(), os, oss, 32);
        ff::cpu::pull(ot, it, gt, LINEAR, DCT2, 1, 0);
    }

    // Under test: inp with an inflated batch stride P (planes at offsets 0, P).
    const size_t nelem = (size_t)P * (B - 1) + (size_t)(N * C);
    float * inbig = static_cast<float*>(std::calloc(nelem, sizeof(float)));
    if (!inbig) {
        std::printf("  [pushpull inflated-stride] skipped (lazy alloc of %.1f GB refused)\n",
                    nelem * sizeof(float) / 1e9);
        return;
    }
    for (int64_t bt = 0; bt < B; ++bt)
        for (int64_t i = 0; i < N * C; ++i) inbig[bt*P + i] = inp[bt*N*C + i];

    std::vector<float> out_str(B * M * C, 0.0f);
    std::vector<int64_t> isi = {B, N, C}, issi = {P, C, 1};   // inflated, non-contiguous
    DLTensor it = make_cpu_tensor(inbig,         isi, issi, 32);
    DLTensor gt = make_cpu_tensor(grid.data(),    gs,  gss,  32);
    DLTensor ot = make_cpu_tensor(out_str.data(), os,  oss,  32);
    ff::cpu::pull(ot, it, gt, LINEAR, DCT2, 1, 0);

    for (int64_t i = 0; i < B * M * C; ++i)
        check_close((double)out_str[i], (double)out_ref[i], "inflated_stride", 1e-4);

    std::free(inbig);
}

// ---- adjoint residual for arbitrary (batch..., spatial..., C) --------------
// Closes the gaps that the covering matrix ran only nbatch=0 and square 2D:
// exercises >=2 batch dims and an anisotropic 3D grid (distinct extents catch an
// axis swap that the adjoint identity alone is blind to).
template <typename T>
double adjoint_nd(const std::vector<int64_t>& batch,
                  const std::vector<int64_t>& in_sp, const std::vector<int64_t>& gr_sp,
                  int64_t C, int8_t order, int8_t bound, int8_t extrap, unsigned seed, uint8_t bits)
{
    const int D = (int)in_sp.size();
    std::vector<int64_t> ish = batch, gsh = batch, osh = batch;
    for (auto e : in_sp) ish.push_back(e);  ish.push_back(C);
    for (auto e : gr_sp) gsh.push_back(e);  gsh.push_back(D);
    for (auto e : gr_sp) osh.push_back(e);  osh.push_back(C);
    auto iss = contiguous_strides(ish), gss = contiguous_strides(gsh), oss = contiguous_strides(osh);
    auto prod = [](const std::vector<int64_t>& s){ int64_t p = 1; for (auto e : s) p *= e; return p; };
    const int64_t inN = prod(ish), grN = prod(gsh), otN = prod(osh);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0), uc(0.0, 1.0);
    std::vector<T> inp(inN), grid(grN), out(otN, (T)0), yv(otN), pushed(inN, (T)0);
    for (auto& v : inp) v = (T)u(rng);
    for (auto& v : yv)  v = (T)u(rng);
    // per-voxel D coords in [-1, in_sp[d]+1] (spill out of range to hit the boundary)
    for (int64_t v = 0; v < grN / D; ++v)
        for (int d = 0; d < D; ++d) grid[v*D + d] = (T)(uc(rng) * (in_sp[d] + 1) - 1);

    DLTensor it = make_cpu_tensor(inp.data(),  ish, iss, bits);
    DLTensor gt = make_cpu_tensor(grid.data(), gsh, gss, bits);
    DLTensor ot = make_cpu_tensor(out.data(),  osh, oss, bits);
    ff::cpu::pull(ot, it, gt, order, bound, extrap, 0);
    double lhs = 0; for (int64_t i = 0; i < otN; ++i) lhs += (double)out[i] * (double)yv[i];

    DLTensor pt = make_cpu_tensor(pushed.data(), ish, iss, bits);
    DLTensor yt = make_cpu_tensor(yv.data(),     osh, oss, bits);
    ff::cpu::push(pt, yt, gt, order, bound, extrap, 0);
    double rhs = 0; for (int64_t i = 0; i < inN; ++i) rhs += (double)inp[i] * (double)pushed[i];

    double scale = std::max(1.0, std::max(std::fabs(lhs), std::fabs(rhs)));
    return std::fabs(lhs - rhs) / scale;
}

// ---- extrapolate 0 / -1 field-of-view gating (never exercised before) ------
template <typename T>
void test_extrapolate(uint8_t bits, double tol)
{
    const int64_t N = 6, C = 1;
    std::vector<T> inp(N); for (int64_t i = 0; i < N; ++i) inp[i] = (T)(i + 1);
    std::vector<int64_t> is = {N, C}, iss = contiguous_strides(is);
    auto pull1 = [&](double x, int8_t ex) {
        std::vector<T> g = {(T)x};        std::vector<int64_t> gs = {1, 1}, gss = contiguous_strides(gs);
        std::vector<T> o(C, (T)0);        std::vector<int64_t> os = {1, C}, oss = contiguous_strides(os);
        DLTensor it = make_cpu_tensor(inp.data(), is, iss, bits);
        DLTensor gt = make_cpu_tensor(g.data(),   gs, gss, bits);
        DLTensor ot = make_cpu_tensor(o.data(),   os, oss, bits);
        ff::cpu::pull(ot, it, gt, LINEAR, DCT2, ex, 0);
        return (double)o[0];
    };
    // well inside: all three modes agree
    check_close(pull1(2.5,  0), pull1(2.5, 1), "extrap0_inside",  tol);
    check_close(pull1(2.5, -1), pull1(2.5, 1), "extrapm1_inside", tol);
    // x=5.2: past the last voxel CENTRE (5) -> ex=0 rejects (->0); within the last
    // voxel EDGE (5.5) -> ex=-1 keeps it (== ex=1).
    check_close(pull1(5.2,  0), 0.0,           "extrap0_oof",   tol);
    check_close(pull1(5.2, -1), pull1(5.2, 1), "extrapm1_edge", tol);
    // far outside: both reject
    check_close(pull1(-3.0,  0), 0.0, "extrap0_far",   tol);
    check_close(pull1(-3.0, -1), 0.0, "extrapm1_far",  tol);
}

// --- B4: negative / validation tests --------------------------------------
// Bad dtype: float16 must throw at the dtype dispatch, not silently no-op.
void test_bad_dtype_throws()
{
    const int64_t N = 6, C = 2, M = 4;
    std::vector<uint16_t> inp(N*C,0), grid(M,0), out(M*C,0);   // float16 payload
    std::vector<int64_t> is={N,C},iss=contiguous_strides(is);
    std::vector<int64_t> gs={M,1},gss=contiguous_strides(gs);
    std::vector<int64_t> os={M,C},oss=contiguous_strides(os);
    DLTensor it=make_cpu_tensor(inp.data(), is,iss,16);
    DLTensor gt=make_cpu_tensor(grid.data(),gs,gss,16);
    DLTensor ot=make_cpu_tensor(out.data(), os,oss,16);
    bool threw = false;
    try { ff::cpu::pull(ot, it, gt, LINEAR, DCT2, 1, 0); } catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [pushpull.bad_dtype_throws]\n"); }
}

// Shape mismatch: inp and grid must have the same rank; a rank-3 inp against a
// rank-2 grid must throw.
void test_shape_mismatch_throws()
{
    const int64_t N = 6, C = 2, M = 4;
    std::vector<double> inp(1*N*C,0), grid(M,0), out(M*C,0);
    std::vector<int64_t> is={1,N,C},iss=contiguous_strides(is);   // rank 3 (bad)
    std::vector<int64_t> gs={M,1},  gss=contiguous_strides(gs);   // rank 2
    std::vector<int64_t> os={M,C},  oss=contiguous_strides(os);   // rank 2
    DLTensor it=make_cpu_tensor(inp.data(), is,iss,64);
    DLTensor gt=make_cpu_tensor(grid.data(),gs,gss,64);
    DLTensor ot=make_cpu_tensor(out.data(), os,oss,64);
    bool threw = false;
    try { ff::cpu::pull(ot, it, gt, LINEAR, DCT2, 1, 0); } catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [pushpull.shape_mismatch_throws]\n"); }
}

} // namespace

int main()
{
    std::printf("pushpull module CPU tests\n");
    test_bad_dtype_throws();
    test_shape_mismatch_throws();

    test_pull_identity_1d<double>(64, 1e-4);
    test_pull_midpoint_1d<double>(64, 1e-4);
    test_grad_ramp_1d<double>(64, 1e-4);
    // B1: float (bits=32) value checks with looser tolerances.
    test_pull_identity_1d<float>(32, 2e-3);
    test_pull_midpoint_1d<float>(32, 2e-3);
    test_grad_ramp_1d<float>(32, 2e-3);

    // Covering matrix (see the FF_TEST_SPARSE note in pushpull.cpp): rather than
    // the full order x bound x ndim cross-product, exercise (a) every spline
    // order once, (b) every bound once at the two reference orders, and (c) the
    // order x ndim interaction on a non-square 2D grid -- the classes where the
    // bugs actually live. This matches the sparse instantiation set exactly.
    // Each is run for double (exact, tol 1e-9) and, for B1, float (tol 1e-3).
    for (unsigned s = 1; s <= 4; ++s) {
        // (a) per-order: each order 0..7 at DCT2 (weights/indices are per-order).
        for (int8_t ord = 0; ord <= 7; ++ord) {
            check_close(adjoint_residual_1d<double>(ord, DCT2, s + ord * 11, 64), 0.0,
                        "adjoint_1d_order", 1e-9);
            check_close(adjoint_residual_1d<float >(ord, DCT2, s + ord * 11, 32), 0.0,
                        "adjoint_1d_order_f32", 1e-3);
        }
        // (b) per-bound: every bound at Linear (specialised path) and Cubic
        //     (generic "Any" path) -- boundary wrapping is per-bound.
        for (int b = 0; b < 8; ++b) {
            check_close(adjoint_residual_1d<double>(LINEAR, ALL_BOUNDS[b], s + 100 + b, 64),
                        0.0, "adjoint_1d_linear_bound", 1e-9);
            check_close(adjoint_residual_1d<double>(CUBIC,  ALL_BOUNDS[b], s + 200 + b, 64),
                        0.0, "adjoint_1d_cubic_bound", 1e-9);
            check_close(adjoint_residual_1d<float >(LINEAR, ALL_BOUNDS[b], s + 100 + b, 32),
                        0.0, "adjoint_1d_linear_bound_f32", 1e-3);
        }
        // (c) order x ndim on a NON-SQUARE (Nx=4, Ny=5) grid: regression for the
        //     2D any-spline pull that used size[1] for the x axis (adjointness
        //     breaks + x-taps read OOB unless pull/push agree on per-axis sizes).
        check_close(adjoint_residual_2d<double>(LINEAR,    DCT2, s + 300, 64), 0.0, "adjoint_2d_linear", 1e-9);
        check_close(adjoint_residual_2d<double>(QUADRATIC, DCT2, s + 310, 64), 0.0, "adjoint_2d_quad",   1e-9);
        check_close(adjoint_residual_2d<double>(CUBIC,     DCT2, s + 320, 64), 0.0, "adjoint_2d_cubic",  1e-9);
        check_close(adjoint_residual_2d<float >(LINEAR,    DCT2, s + 300, 32), 0.0, "adjoint_2d_linear_f32", 1e-3);
        // (d) count == push(ones), at both reference orders (double + float).
        test_count_equals_push_ones_1d<double>(LINEAR, s + 400, 64, 1e-4);
        test_count_equals_push_ones_1d<double>(CUBIC,  s + 410, 64, 1e-4);
        test_count_equals_push_ones_1d<float >(LINEAR, s + 400, 32, 2e-3);
    }
    // Gap closers (fable review): >=2 batch dims, anisotropic 3D (distinct
    // extents catch axis swaps the adjoint is blind to), and the extrapolate
    // 0/-1 FOV modes. Runs at Linear + Cubic (both instantiated under sparse).
    for (int8_t ord : {LINEAR, CUBIC}) {
        check_close(adjoint_nd<double>({2,3}, {4,5},   {3,4},   2, ord, DCT2, 1, 900+ord, 64),
                    0.0, "adjoint_nbatch2_2d",     1e-9);
        check_close(adjoint_nd<float >({2,3}, {4,5},   {3,4},   2, ord, DCT2, 1, 900+ord, 32),
                    0.0, "adjoint_nbatch2_2d_f32", 1e-3);
        check_close(adjoint_nd<double>({},    {4,5,6}, {3,4,5}, 2, ord, DCT2, 1, 950+ord, 64),
                    0.0, "adjoint_aniso_3d",       1e-9);
    }
    test_extrapolate<double>(64, 1e-4);
    test_extrapolate<float >(32, 2e-3);

    // B2: 64-bit index + non-contiguous stride path.
    test_inflated_stride();

    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
