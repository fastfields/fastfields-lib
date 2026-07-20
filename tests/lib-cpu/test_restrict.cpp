// CPU unit tests for the restrict module.
//
// `restriction` is the adjoint (transpose) of the resize `resample`
// prolongation. We verify the defining adjoint identity numerically:
//     <resample(c), g>_fine  ==  <c, restriction(g)>_coarse
// for random coarse c and fine g, matching order / bound / shift and using
// reciprocal scales (resize: coarse/fine, restrict: fine/coarse).
//
// Build (from fastfields-cpu-lib):
//   clang++ -std=c++11 -O2 -ferror-limit=5 -I. tests/test_restrict.cpp resize.cpp restrict.cpp -o build/test_restrict && ./build/test_restrict

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include "dlpack.h"
#include "resize.h"
#include "restrict.h"

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

std::vector<int64_t> cstrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> s(shape.size());
    int64_t acc = 1;
    for (int64_t d = (int64_t)shape.size() - 1; d >= 0; --d) { s[d] = acc; acc *= shape[d]; }
    return s;
}

int g_failures = 0, g_checks = 0;

void check_close(double a, double b, const char* what, double tol = 1e-9)
{
    ++g_checks;
    double diff = std::fabs(a - b);
    double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    if (diff / scale > tol) {
        ++g_failures;
        std::printf("  MISMATCH [%s]: <Pc,g>=%.10g  <c,Rg>=%.10g\n", what, a, b);
    }
}

double dot(const std::vector<double>& a, const std::vector<double>& b)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// 1D adjoint check: coarse nc -> fine nf (factor nf/nc), edge anchor.
void test_adjoint_1d(int64_t nc, int64_t nf, int8_t order, int8_t bound, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    std::vector<double> c(nc), g(nf);
    for (auto& v : c) v = u(rng);
    for (auto& v : g) v = u(rng);

    std::vector<int64_t> shc = {nc}, stc = cstrides(shc);
    std::vector<int64_t> shf = {nf}, stf = cstrides(shf);

    // P: coarse -> fine (resample). edge anchor: shift=0.5, scale = coarse/fine.
    std::vector<double> Pc(nf, 0.0);
    {
        DLTensor ti = make_cpu_tensor(c.data(),  shc, stc, 64);
        DLTensor to = make_cpu_tensor(Pc.data(), shf, stf, 64);
        double scale[1] = {(double)nc / (double)nf};
        ff::cpu::resample(to, ti, order, bound, 0.5, scale, 1, 0);
    }

    // R = P^T: fine -> coarse (restriction). reciprocal scale = fine/coarse.
    std::vector<double> Rg(nc, 0.0); // pre-zeroed (restriction accumulates)
    {
        DLTensor ti = make_cpu_tensor(g.data(),  shf, stf, 64);
        DLTensor to = make_cpu_tensor(Rg.data(), shc, stc, 64);
        double scale[1] = {(double)nf / (double)nc};
        ff::cpu::restriction(to, ti, order, bound, 0.5, scale, 1, 0);
    }

    check_close(dot(Pc, g), dot(c, Rg), "adj1d");
}

// 2D adjoint check with a batch dim.
void test_adjoint_2d(int64_t B, int64_t hc, int64_t wc, int64_t hf, int64_t wf,
                     int8_t order, int8_t bound, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    std::vector<double> c(B * hc * wc), g(B * hf * wf);
    for (auto& v : c) v = u(rng);
    for (auto& v : g) v = u(rng);

    std::vector<int64_t> shc = {B, hc, wc}, stc = cstrides(shc);
    std::vector<int64_t> shf = {B, hf, wf}, stf = cstrides(shf);

    std::vector<double> Pc(B * hf * wf, 0.0);
    {
        DLTensor ti = make_cpu_tensor(c.data(),  shc, stc, 64);
        DLTensor to = make_cpu_tensor(Pc.data(), shf, stf, 64);
        double scale[2] = {(double)hc / (double)hf, (double)wc / (double)wf};
        ff::cpu::resample(to, ti, order, bound, 0.5, scale, 2, 0);
    }

    std::vector<double> Rg(B * hc * wc, 0.0);
    {
        DLTensor ti = make_cpu_tensor(g.data(),  shf, stf, 64);
        DLTensor to = make_cpu_tensor(Rg.data(), shc, stc, 64);
        double scale[2] = {(double)hf / (double)hc, (double)wf / (double)wc};
        ff::cpu::restriction(to, ti, order, bound, 0.5, scale, 2, 0);
    }

    check_close(dot(Pc, g), dot(c, Rg), "adj2d");
}

} // namespace

int main()
{
    std::printf("restrict module CPU tests\n");
    for (unsigned s = 1; s <= 4; ++s) {
        // 1D, several orders / bounds / factors
        test_adjoint_1d(4, 8,  1, 3 /*DCT2*/, s);
        test_adjoint_1d(4, 8,  3, 3, s + 10);
        test_adjoint_1d(5, 10, 2, 3, s + 20);
        test_adjoint_1d(4, 12, 1, 1 /*Replicate*/, s + 30);
        test_adjoint_1d(6, 9,  3, 3, s + 40); // non-integer factor
        // 2D with batch
        test_adjoint_2d(2, 3, 4, 6, 8, 1, 3, s + 50);
        test_adjoint_2d(2, 3, 3, 6, 6, 3, 3, s + 60);
    }
    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
