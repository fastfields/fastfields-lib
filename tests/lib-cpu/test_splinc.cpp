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
#include "dlpack.h"
#include "splinc.h"
#include "impl/kernels/spline.h"
#include "impl/kernels/bounds.h"

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

} // namespace

int main()
{
    std::printf("splinc module CPU tests\n");
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
