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
#include <cmath>
#include <vector>
#include <random>
#include "dlpack.h"
#include "pushpull.h"

namespace {

// Spline orders / bounds (mirror spline_t / bound_t enum values).
constexpr int8_t NEAREST = 0, LINEAR = 1;
constexpr int8_t DCT2 = 3;

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

// ---- 1D linear pull == identity at integer nodes -------------------
void test_pull_identity_1d()
{
    const int64_t N = 6, C = 2;
    std::vector<double> inp(N * C);
    for (int64_t i = 0; i < N * C; ++i) inp[i] = 0.5 + i;

    // grid: sample every integer node 0..N-1
    std::vector<double> grid(N * 1);
    for (int64_t i = 0; i < N; ++i) grid[i] = (double)i;
    std::vector<double> out(N * C, -999.0);

    std::vector<int64_t> is = {N, C},  iss = contiguous_strides(is);
    std::vector<int64_t> gs = {N, 1},  gss = contiguous_strides(gs);
    std::vector<int64_t> os = {N, C},  oss = contiguous_strides(os);
    DLTensor it = make_cpu_tensor(inp.data(),  is, iss, 64);
    DLTensor gt = make_cpu_tensor(grid.data(), gs, gss, 64);
    DLTensor ot = make_cpu_tensor(out.data(),  os, oss, 64);

    ff::cpu::pull(ot, it, gt, LINEAR, DCT2, 1, 0);
    for (int64_t i = 0; i < N * C; ++i) check_close(out[i], inp[i], "pull_id_1d");
}

// ---- 1D linear pull midpoint == average ----------------------------
void test_pull_midpoint_1d()
{
    const int64_t N = 4, C = 1;
    std::vector<double> inp = {1.0, 3.0, 7.0, 10.0};
    std::vector<double> grid = {0.5, 1.5, 2.25};
    const int64_t M = grid.size();
    std::vector<double> out(M * C, 0.0);

    std::vector<int64_t> is = {N, C}, iss = contiguous_strides(is);
    std::vector<int64_t> gs = {M, 1}, gss = contiguous_strides(gs);
    std::vector<int64_t> os = {M, C}, oss = contiguous_strides(os);
    DLTensor it = make_cpu_tensor(inp.data(),  is, iss, 64);
    DLTensor gt = make_cpu_tensor(grid.data(), gs, gss, 64);
    DLTensor ot = make_cpu_tensor(out.data(),  os, oss, 64);

    ff::cpu::pull(ot, it, gt, LINEAR, DCT2, 1, 0);
    check_close(out[0], 0.5 * 1.0 + 0.5 * 3.0,  "pull_mid_1d");
    check_close(out[1], 0.5 * 3.0 + 0.5 * 7.0,  "pull_mid_1d");
    check_close(out[2], 0.75 * 7.0 + 0.25 * 10.0, "pull_mid_1d");
}

// ---- push is the adjoint of pull (1D and 2D) -----------------------
// Returns |<pull(inp),g> - <inp,push(g)>|.
double adjoint_residual_1d(int8_t order, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> uc(0.0, 5.0);   // in-range coords into N=6

    const int64_t N = 6, C = 3, M = 9;
    std::vector<double> inp(N * C), g(M * C);
    for (auto& v : inp) v = u(rng);
    for (auto& v : g)   v = u(rng);
    std::vector<double> grid(M);
    for (auto& v : grid) v = uc(rng);

    std::vector<double> pulled(M * C, 0.0), pushed(N * C, 0.0);

    std::vector<int64_t> is = {N, C}, iss = contiguous_strides(is);
    std::vector<int64_t> gs = {M, 1}, gss = contiguous_strides(gs);
    std::vector<int64_t> ms = {M, C}, mss = contiguous_strides(ms);
    DLTensor it = make_cpu_tensor(inp.data(),    is, iss, 64);
    DLTensor gt = make_cpu_tensor(grid.data(),   gs, gss, 64);
    DLTensor po = make_cpu_tensor(pulled.data(), ms, mss, 64);
    DLTensor go = make_cpu_tensor(g.data(),      ms, mss, 64);
    DLTensor so = make_cpu_tensor(pushed.data(), is, iss, 64);

    ff::cpu::pull(po, it, gt, order, DCT2, 1, 0);
    ff::cpu::push(so, go, gt, order, DCT2, 1, 0);   // pushed pre-zeroed

    double lhs = 0.0, rhs = 0.0;
    for (int64_t i = 0; i < M * C; ++i) lhs += pulled[i] * g[i];
    for (int64_t i = 0; i < N * C; ++i) rhs += inp[i] * pushed[i];
    return std::fabs(lhs - rhs);
}

double adjoint_residual_2d(int8_t order, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> ux(0.0, 3.0);   // into Nx=4
    std::uniform_real_distribution<double> uy(0.0, 4.0);   // into Ny=5

    const int64_t Nx = 4, Ny = 5, C = 2, Mx = 3, My = 3;
    std::vector<double> inp(Nx * Ny * C), g(Mx * My * C);
    for (auto& v : inp) v = u(rng);
    for (auto& v : g)   v = u(rng);
    std::vector<double> grid(Mx * My * 2);
    for (int64_t k = 0; k < Mx * My; ++k) { grid[2*k] = ux(rng); grid[2*k+1] = uy(rng); }

    std::vector<double> pulled(Mx * My * C, 0.0), pushed(Nx * Ny * C, 0.0);

    std::vector<int64_t> is = {Nx, Ny, C}, iss = contiguous_strides(is);
    std::vector<int64_t> gs = {Mx, My, 2}, gss = contiguous_strides(gs);
    std::vector<int64_t> ms = {Mx, My, C}, mss = contiguous_strides(ms);
    DLTensor it = make_cpu_tensor(inp.data(),    is, iss, 64);
    DLTensor gt = make_cpu_tensor(grid.data(),   gs, gss, 64);
    DLTensor po = make_cpu_tensor(pulled.data(), ms, mss, 64);
    DLTensor go = make_cpu_tensor(g.data(),      ms, mss, 64);
    DLTensor so = make_cpu_tensor(pushed.data(), is, iss, 64);

    ff::cpu::pull(po, it, gt, order, DCT2, 1, 0);
    ff::cpu::push(so, go, gt, order, DCT2, 1, 0);

    double lhs = 0.0, rhs = 0.0;
    for (size_t i = 0; i < pulled.size(); ++i) lhs += pulled[i] * g[i];
    for (size_t i = 0; i < pushed.size(); ++i) rhs += inp[i] * pushed[i];
    return std::fabs(lhs - rhs);
}

// ---- count == push of ones -----------------------------------------
void test_count_equals_push_ones_1d(int8_t order, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uc(0.0, 5.0);

    const int64_t N = 6, M = 10;
    std::vector<double> grid(M);
    for (auto& v : grid) v = uc(rng);
    std::vector<double> ones(M * 1, 1.0);
    std::vector<double> pushed(N * 1, 0.0), counted(N * 1, 0.0);

    std::vector<int64_t> gs = {M, 1}, gss = contiguous_strides(gs);
    std::vector<int64_t> ms = {M, 1}, mss = contiguous_strides(ms);
    std::vector<int64_t> ns = {N, 1}, nss = contiguous_strides(ns);
    DLTensor gt = make_cpu_tensor(grid.data(),    gs, gss, 64);
    DLTensor od = make_cpu_tensor(ones.data(),    ms, mss, 64);
    DLTensor so = make_cpu_tensor(pushed.data(),  ns, nss, 64);
    DLTensor co = make_cpu_tensor(counted.data(), ns, nss, 64);

    ff::cpu::push (so, od, gt, order, DCT2, 1, 0);
    ff::cpu::count(co,     gt, order, DCT2, 1, 0);
    for (int64_t i = 0; i < N; ++i) check_close(counted[i], pushed[i], "count_vs_push");
}

// ---- grad of a linear ramp == constant slope -----------------------
void test_grad_ramp_1d()
{
    // inp[i] = a*i + b -> spatial gradient == a everywhere (linear spline)
    const int64_t N = 6, C = 1;
    const double a = 2.5, b = -1.0;
    std::vector<double> inp(N);
    for (int64_t i = 0; i < N; ++i) inp[i] = a * i + b;

    std::vector<double> grid = {0.5, 1.5, 2.5, 3.5};
    const int64_t M = grid.size();
    std::vector<double> out(M * C * 1, 0.0);   // (M, C, D=1)

    std::vector<int64_t> is = {N, C},    iss = contiguous_strides(is);
    std::vector<int64_t> gs = {M, 1},    gss = contiguous_strides(gs);
    std::vector<int64_t> os = {M, C, 1}, oss = contiguous_strides(os);
    DLTensor it = make_cpu_tensor(inp.data(),  is, iss, 64);
    DLTensor gt = make_cpu_tensor(grid.data(), gs, gss, 64);
    DLTensor ot = make_cpu_tensor(out.data(),  os, oss, 64);

    ff::cpu::grad(ot, it, gt, LINEAR, DCT2, 1, false, 0);
    for (int64_t i = 0; i < M; ++i) check_close(out[i], a, "grad_ramp_1d");
}

} // namespace

int main()
{
    std::printf("pushpull module CPU tests\n");

    test_pull_identity_1d();
    test_pull_midpoint_1d();
    test_grad_ramp_1d();

    for (unsigned s = 1; s <= 5; ++s) {
        check_close(adjoint_residual_1d(NEAREST, s),      0.0, "adjoint_1d_nearest", 1e-9);
        check_close(adjoint_residual_1d(LINEAR,  s + 10), 0.0, "adjoint_1d_linear",  1e-9);
        check_close(adjoint_residual_1d(2,       s + 20), 0.0, "adjoint_1d_quad",    1e-9);
        check_close(adjoint_residual_1d(3,       s + 30), 0.0, "adjoint_1d_cubic",   1e-9);
        check_close(adjoint_residual_2d(LINEAR,  s + 40), 0.0, "adjoint_2d_linear",  1e-9);
        test_count_equals_push_ones_1d(LINEAR, s + 50);
        test_count_equals_push_ones_1d(3,      s + 60);
    }

    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
