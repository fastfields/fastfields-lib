// CPU unit tests for the splinc module (spline coefficient prefilter).
//
// Strategy: prefilter a signal with ff::cpu::spline_coeff, then evaluate the
// spline interpolation of the coefficients at the original integer sample
// points (reusing the library's own spline weights / boundary handling).
// A correct prefilter makes interpolation reproduce the input samples.
//
// Build (from fastfields-cpu-lib):
//   clang++ -std=c++11 -O2 -ferror-limit=5 -I. tests/test_splinc.cpp splinc.cpp -o build/test_splinc && ./build/test_splinc

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
#include <fastfields/core/dlpack.h>
#include <fastfields/api/cpu/splinc.h>
#include <fastfields/impl/kernels/spline.h>
#include <fastfields/impl/kernels/bounds.h>

using ff::cpu::spline::type;
using btype = ff::cpu::bound::type;

namespace {

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

void check_close(double a, double b, const char* what, double tol = 2e-3)
{
    ++g_checks;
    double diff = std::fabs(a - b);
    double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    if (diff / scale > tol) {
        ++g_failures;
        std::printf("  MISMATCH [%s]: got %.6g expected %.6g\n", what, a, b);
    }
}

// Evaluate the spline interpolation of coefficients `c` (length n, unit stride)
// at integer node p, for given order + boundary.
double eval_spline(const std::vector<double>& c, int64_t p, int8_t order, btype B)
{
    int64_t n = (int64_t)c.size();
    type I = static_cast<type>(order);
    int64_t low, upp;
    ff::cpu::spline::bounds(I, (double)p, low, upp);
    double val = 0.0;
    for (int64_t k = low; k <= upp; ++k) {
        int8_t   sgn = ff::cpu::bound::sign(B, k, n);
        if (sgn == 0) continue;
        int64_t  idx = ff::cpu::bound::index(B, k, n);
        double   w   = ff::cpu::spline::weight<double>(I, (double)(p - k));
        val += (double)sgn * c[idx] * w;
    }
    return val;
}

template <typename scalar_t>
void run_case(int64_t nbatch, int64_t n, int8_t order, btype B, uint8_t bits, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    std::vector<double> orig(nbatch * n);
    std::vector<scalar_t> data(nbatch * n);
    for (int64_t i = 0; i < nbatch * n; ++i) { double v = u(rng); orig[i] = v; data[i] = (scalar_t)v; }

    std::vector<int64_t> shape   = {nbatch, n};
    std::vector<int64_t> strides = contiguous_strides(shape);
    DLTensor t = make_cpu_tensor(data.data(), shape, strides, bits);

    ff::cpu::spline_coeff(t, order, (int8_t)B, 0);

    for (int64_t b = 0; b < nbatch; ++b) {
        std::vector<double> c(n);
        for (int64_t i = 0; i < n; ++i) c[i] = (double)data[b * n + i];
        for (int64_t p = 0; p < n; ++p) {
            double got = eval_spline(c, p, order, B);
            check_close(got, orig[b * n + p], "reproduce");
        }
    }
}

// B4: an unsupported dtype (float16) must throw, not silently no-op.
// spline_coeff takes a single tensor (no cross-tensor shape-mismatch case), and
// orders 0/1 short-circuit before the dtype dispatch, so a cubic order is used
// to actually reach the guard.
void test_bad_dtype_throws()
{
    std::vector<uint16_t> data(2 * 8, 0);                 // float16 payload
    std::vector<int64_t> shape = {2, 8}, strides = contiguous_strides(shape);
    DLTensor t = make_cpu_tensor(data.data(), shape, strides, 16);
    bool threw = false;
    try { ff::cpu::spline_coeff(t, /*order=*/3, (int8_t)btype::DCT2, 0); }
    catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [splinc.bad_dtype_throws]\n"); }
}

// ---------------------------------------------------------------------------
// Interpolation property, for every boundary condition x order x short axes.
//
// For each (bound, order, n) a few lines are prefiltered and sampled back at
// the integer nodes with the *same* boundary condition; the largest error over
// the lines is one check per dtype. Tolerances are absolute on inputs in
// [-1, 1]; measured worst cases are ~2e-14 (float64) and ~1.3e-6 (float32),
// at order 6. Every bound the library knows except
// NoCheck is covered -- including zero/dst1/dst2 and zero/replicate at orders
// >= 4, whose edges are corrected exactly. Short axes (n = 1..8) reach the
// closed-form initial conditions and, for zero/replicate, the small dense
// solve; n = 130 is past every pole's horizon (63 samples at order 7) on both
// sides at once.
//
// The axis is laid out with a non-unit stride (memory is [n, nbatch] and the
// DLTensor views it as [nbatch, n] with strides (1, nbatch)), so the strided
// reads of the initial conditions and of the edge correction are exercised.
// ---------------------------------------------------------------------------

const btype k_all_bounds[7] = {btype::Zero, btype::Replicate, btype::DCT1,
                               btype::DCT2, btype::DST1, btype::DST2,
                               btype::DFT};
const char * k_bound_names[7] = {"zero", "replicate", "dct1", "dct2",
                                 "dst1", "dst2", "dft"};
const int64_t k_lengths[] = {1, 2, 3, 4, 5, 6, 7, 8, 11, 16, 33, 70, 130};
const int     k_nlengths  = sizeof(k_lengths) / sizeof(k_lengths[0]);

// g_failed[bound][order] collects the axis lengths that failed, for the summary.
std::vector<int64_t> g_failed[7][8];

template <typename scalar_t>
bool interpolates(int64_t n, int8_t order, btype B, uint8_t bits,
                  unsigned seed, double tol)
{
    const int64_t nbatch = 3;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    std::vector<double>   orig(nbatch * n);
    std::vector<scalar_t> data(nbatch * n);
    for (int64_t i = 0; i < nbatch * n; ++i) {
        double v = u(rng); orig[i] = v; data[i] = (scalar_t)v;
    }
    std::vector<int64_t> shape   = {nbatch, n};
    std::vector<int64_t> strides = {1, nbatch};
    DLTensor t = make_cpu_tensor(data.data(), shape, strides, bits);

    try { ff::cpu::spline_coeff(t, order, (int8_t)B, 0); }
    catch (const std::exception &) { return false; }

    double maxerr = 0.0;
    for (int64_t b = 0; b < nbatch; ++b) {
        std::vector<double> c(n);
        for (int64_t i = 0; i < n; ++i) c[i] = (double)data[i * nbatch + b];
        for (int64_t p = 0; p < n; ++p) {
            double err = std::fabs(eval_spline(c, p, order, B)
                                   - orig[p * nbatch + b]);
            if (!(err <= maxerr)) maxerr = err;   // NaN-propagating
        }
    }
    return maxerr <= tol;
}

void test_interpolation_matrix()
{
    for (int b = 0; b < 7; ++b)
    for (int8_t order = 0; order <= 7; ++order)
    for (int l = 0; l < k_nlengths; ++l) {
        const int64_t n = k_lengths[l];
        unsigned seed = 1000u + 97u * b + 13u * order + (unsigned)n;
        bool ok64 = interpolates<double>(n, order, k_all_bounds[b], 64,
                                         seed, 1e-11);
        bool ok32 = interpolates<float >(n, order, k_all_bounds[b], 32,
                                         seed + 7u, 1e-5);
        g_checks += 2;
        if (!ok64 || !ok32) {
            g_failures += !ok64 + !ok32;
            g_failed[b][order].push_back(n);
        }
    }
    bool any = false;
    for (int b = 0; b < 7; ++b)
    for (int o = 0; o < 8; ++o) {
        if (g_failed[b][o].empty()) continue;
        if (!any) std::printf("  interpolation failures (bound, order: n):\n");
        any = true;
        std::printf("    %-9s order %d: n =", k_bound_names[b], o);
        for (size_t i = 0; i < g_failed[b][o].size(); ++i)
            std::printf(" %lld", (long long)g_failed[b][o][i]);
        std::printf("\n");
    }
}

// NoCheck has no extension at all, so there is nothing to prefilter against:
// the backend must refuse it rather than alias it onto another recursion.
void test_nocheck_throws()
{
    std::vector<double> data(2 * 8, 0.0);
    std::vector<int64_t> shape = {2, 8}, strides = contiguous_strides(shape);
    DLTensor t = make_cpu_tensor(data.data(), shape, strides, 64);
    bool threw = false;
    try { ff::cpu::spline_coeff(t, 3, (int8_t)btype::NoCheck, 0); }
    catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [splinc.nocheck_throws]\n"); }
}

} // namespace

int main()
{
    std::printf("splinc module CPU tests\n");
    test_bad_dtype_throws();
    test_nocheck_throws();
    test_interpolation_matrix();
    // DCT2 is the reliable boundary condition for the prefilter/interpolation
    // round-trip (scipy-derived initial conditions).
    for (unsigned s = 1; s <= 4; ++s) {
        for (int8_t order = 0; order <= 7; ++order) {
            run_case<float >(3, 24, order, btype::DCT2, 32, s);
            run_case<double>(2, 31, order, btype::DCT2, 64, s + 50);
        }
        // A couple of other boundary conditions with cubic order.
        run_case<double>(1, 20, 3, btype::DFT,       64, s + 100);
        run_case<double>(1, 20, 3, btype::Replicate, 64, s + 200);
        // order 0/1 must leave the data untouched (identity prefilter).
        run_case<double>(2, 16, 1, btype::DCT2, 64, s + 300);
    }
    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
