// CPU unit tests for the resize (prolongation) module.
//
// Invariants checked:
//   * resize by factor 1 (scale=1, shift=0) is the identity for interpolating
//     orders (nearest/linear).
//   * upsampling a linear ramp by an integer factor with linear interpolation
//     stays linear (matches the analytic ramp at interior nodes).
//   * 2D linear identity reproduces the input.
//
// Build (from fastfields-cpu-lib):
//   clang++ -std=c++11 -O2 -ferror-limit=5 -I. tests/test_resize.cpp resize.cpp -o build/test_resize && ./build/test_resize

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include "dlpack.h"
#include "resize.h"

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

void check_close(double a, double b, const char* what, double tol = 1e-4)
{
    ++g_checks;
    double diff = std::fabs(a - b);
    double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    if (diff / scale > tol) {
        ++g_failures;
        std::printf("  MISMATCH [%s]: got %.6g expected %.6g\n", what, a, b);
    }
}

// 1D identity: scale=1, shift=0, out shape == in shape.
void test_identity_1d(int8_t order)
{
    const int64_t n = 12;
    std::vector<double> in(n), out(n, -999.0);
    for (int64_t i = 0; i < n; ++i) in[i] = std::sin(0.3 * i) + 0.5 * i;

    std::vector<int64_t> sh = {n}, st = cstrides(sh);
    DLTensor ti = make_cpu_tensor(in.data(),  sh, st, 64);
    DLTensor to = make_cpu_tensor(out.data(), sh, st, 64);

    double scale[1] = {1.0};
    ff::cpu::resample(to, ti, order, /*bound=*/3 /*DCT2*/, /*shift=*/0.0, scale, /*ndim=*/1, 0);
    for (int64_t i = 0; i < n; ++i) check_close(out[i], in[i], "identity1d");
}

// 1D linear-ramp upsample stays linear.
void test_ramp_upsample(int factor)
{
    const int64_t ni = 8;
    const int64_t no = ni * factor;
    const double a = 2.0, b = 0.75;
    std::vector<double> in(ni), out(no, 0.0);
    for (int64_t i = 0; i < ni; ++i) in[i] = a + b * i;

    std::vector<int64_t> shi = {ni}, sti = cstrides(shi);
    std::vector<int64_t> sho = {no}, sto = cstrides(sho);
    DLTensor ti = make_cpu_tensor(in.data(),  shi, sti, 64);
    DLTensor to = make_cpu_tensor(out.data(), sho, sto, 64);

    // anchor "first"/None: shift=0, scale=1/factor  ->  x = w/factor
    double scale[1] = {1.0 / factor};
    ff::cpu::resample(to, ti, /*order=*/1 /*Linear*/, /*bound=*/1 /*Replicate*/, 0.0, scale, 1, 0);

    for (int64_t w = 0; w < no; ++w) {
        double x = (double)w / factor;
        if (x <= (double)(ni - 1)) // interior: exact linear interpolation
            check_close(out[w], a + b * x, "ramp");
    }
}

// 2D linear identity (scale=1, shift=0) reproduces the input, incl. a batch dim.
void test_identity_2d()
{
    const int64_t B = 2, H = 5, W = 6;
    std::vector<double> in(B * H * W), out(B * H * W, -1.0);
    for (int64_t i = 0; i < B * H * W; ++i) in[i] = 0.1 * i - 3.0;

    std::vector<int64_t> sh = {B, H, W}, st = cstrides(sh);
    DLTensor ti = make_cpu_tensor(in.data(),  sh, st, 64);
    DLTensor to = make_cpu_tensor(out.data(), sh, st, 64);

    double scale[2] = {1.0, 1.0};
    ff::cpu::resample(to, ti, /*order=*/1, /*bound=*/3, 0.0, scale, /*ndim=*/2, 0);
    for (int64_t i = 0; i < B * H * W; ++i) check_close(out[i], in[i], "identity2d");
}

} // namespace

// Regression (A4): an unsupported dtype must throw, not silently return with
// the output buffer untouched.
void test_bad_dtype_throws()
{
    std::vector<uint16_t> in(4, 0), out(4, 0);            // float16 payload
    std::vector<int64_t> sh = {4}, st = cstrides(sh);
    DLTensor ti = make_cpu_tensor(in.data(),  sh, st, 16); // kDLFloat, 16 bits
    DLTensor to = make_cpu_tensor(out.data(), sh, st, 16);
    bool threw = false;
    try {
        ff::cpu::resample(to, ti, /*order=*/1, /*bound=*/3, 0.0, nullptr, 1, 0);
    } catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [resize.bad_dtype_throws]\n"); }
}

int main()
{
    std::printf("resize module CPU tests\n");
    test_identity_1d(0); // nearest
    test_identity_1d(1); // linear
    test_ramp_upsample(2);
    test_ramp_upsample(3);
    test_ramp_upsample(4);
    test_identity_2d();
    test_bad_dtype_throws();
    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
