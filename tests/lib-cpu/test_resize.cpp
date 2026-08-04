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
//   clang++ -std=c++17 -O2 -ferror-limit=5 -DTNY_MAX_RANK=64 -I. -I<teeny>/include -I<teeny>/external/cccl/libcudacxx/include tests/test_resize.cpp resize.cpp -o build/test_resize && ./build/test_resize

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "dlpack.h"
#include "resize.h"
#include "impl/kernels/resize.h"   // legacy Multiscale::resize -- independent oracle

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
// Templated on scalar_t/bits/tol so both double (bits=64) and float (bits=32,
// B1: the dominant ML dtype, previously never instantiated) are exercised.
template <typename T>
void test_identity_1d(int8_t order, uint8_t bits, double tol)
{
    const int64_t n = 12;
    std::vector<T> in(n), out(n, (T)-999.0);
    for (int64_t i = 0; i < n; ++i) in[i] = (T)(std::sin(0.3 * i) + 0.5 * i);

    std::vector<int64_t> sh = {n}, st = cstrides(sh);
    DLTensor ti = make_cpu_tensor(in.data(),  sh, st, bits);
    DLTensor to = make_cpu_tensor(out.data(), sh, st, bits);

    double scale[1] = {1.0};
    ff::cpu::resample(to, ti, order, /*bound=*/3 /*DCT2*/, /*shift=*/0.0, scale, /*ndim=*/1, 0);
    for (int64_t i = 0; i < n; ++i) check_close((double)out[i], (double)in[i], "identity1d", tol);
}

// 1D linear-ramp upsample stays linear.
template <typename T>
void test_ramp_upsample(int factor, uint8_t bits, double tol)
{
    const int64_t ni = 8;
    const int64_t no = ni * factor;
    const double a = 2.0, b = 0.75;
    std::vector<T> in(ni), out(no, (T)0.0);
    for (int64_t i = 0; i < ni; ++i) in[i] = (T)(a + b * i);

    std::vector<int64_t> shi = {ni}, sti = cstrides(shi);
    std::vector<int64_t> sho = {no}, sto = cstrides(sho);
    DLTensor ti = make_cpu_tensor(in.data(),  shi, sti, bits);
    DLTensor to = make_cpu_tensor(out.data(), sho, sto, bits);

    // anchor "first"/None: shift=0, scale=1/factor  ->  x = w/factor
    double scale[1] = {1.0 / factor};
    ff::cpu::resample(to, ti, /*order=*/1 /*Linear*/, /*bound=*/1 /*Replicate*/, 0.0, scale, 1, 0);

    for (int64_t w = 0; w < no; ++w) {
        double x = (double)w / factor;
        if (x <= (double)(ni - 1)) // interior: exact linear interpolation
            check_close((double)out[w], a + b * x, "ramp", tol);
    }
}

// 2D linear identity (scale=1, shift=0) reproduces the input, incl. a batch dim.
template <typename T>
void test_identity_2d(uint8_t bits, double tol)
{
    const int64_t B = 2, H = 5, W = 6;
    std::vector<T> in(B * H * W), out(B * H * W, (T)-1.0);
    for (int64_t i = 0; i < B * H * W; ++i) in[i] = (T)(0.1 * i - 3.0);

    std::vector<int64_t> sh = {B, H, W}, st = cstrides(sh);
    DLTensor ti = make_cpu_tensor(in.data(),  sh, st, bits);
    DLTensor to = make_cpu_tensor(out.data(), sh, st, bits);

    double scale[2] = {1.0, 1.0};
    ff::cpu::resample(to, ti, /*order=*/1, /*bound=*/3, 0.0, scale, /*ndim=*/2, 0);
    for (int64_t i = 0; i < B * H * W; ++i) check_close((double)out[i], (double)in[i], "identity2d", tol);
}

// B2: 64-bit index + non-contiguous stride path. A leading batch dim of size 2
// with a stride >= INT32_MAX makes canUse32BitIndexMath return false (it keys
// on the max element offset), forcing the int64_t offset_t instantiation and a
// strided (non-contiguous) inp read. The reference is the same call on a small
// contiguous tensor. The big buffer is lazily allocated (only the 2 touched
// planes commit pages); if the virtual allocation is refused we skip. float
// keeps it to ~8.6 GB virtual.
void test_inflated_stride()
{
    const int64_t B = 2, n = 12;
    const int64_t P = (int64_t)1 << 31;                 // 2147483648 > INT32_MAX

    // Reference: contiguous [B, n] float, identity resample (scale=1).
    std::vector<float> in(B * n), out_ref(B * n, -1.0f);
    for (int64_t i = 0; i < B * n; ++i) in[i] = std::sin(0.2f * i) + 0.3f * i;
    std::vector<int64_t> sh = {B, n}, st = cstrides(sh);
    double scale[1] = {1.0};
    {
        DLTensor ti = make_cpu_tensor(in.data(),      sh, st, 32);
        DLTensor to = make_cpu_tensor(out_ref.data(), sh, st, 32);
        ff::cpu::resample(to, ti, /*order=*/1, /*bound=*/3, 0.0, scale, /*ndim=*/1, 0);
    }

    // Under test: inp with an inflated batch stride P (rows at offsets 0 and P).
    const size_t nelem = (size_t)P * (B - 1) + (size_t)n;
    float * inbig = static_cast<float*>(std::calloc(nelem, sizeof(float)));
    if (!inbig) {
        std::printf("  [resize inflated-stride] skipped (lazy alloc of %.1f GB refused)\n",
                    nelem * sizeof(float) / 1e9);
        return;
    }
    for (int64_t bt = 0; bt < B; ++bt)
        for (int64_t i = 0; i < n; ++i) inbig[bt*P + i] = in[bt*n + i];

    std::vector<float> out_str(B * n, -1.0f);
    std::vector<int64_t> shi = {B, n}, sti = {P, 1};   // inflated, non-contiguous
    std::vector<int64_t> sho = {B, n}, sto = cstrides(sho);
    DLTensor ti = make_cpu_tensor(inbig,        shi, sti, 32);
    DLTensor to = make_cpu_tensor(out_str.data(), sho, sto, 32);
    ff::cpu::resample(to, ti, /*order=*/1, /*bound=*/3, 0.0, scale, /*ndim=*/1, 0);

    for (int64_t i = 0; i < B * n; ++i)
        check_close((double)out_str[i], (double)out_ref[i], "inflated_stride", 1e-4);

    std::free(inbig);
}

// Regression: DLPack allows DLTensor.strides == NULL for a compact row-major
// tensor. The dispatch/autocast layer must treat a null strides field as
// contiguous strides rather than dereferencing it, so a call with strides=NULL
// must match the same call with explicit contiguous strides.
void test_null_strides_contiguous()
{
    const int64_t B = 2, H = 5, W = 6;
    std::vector<double> in(B * H * W);
    for (int64_t i = 0; i < B * H * W; ++i) in[i] = 0.1 * i - 3.0;

    std::vector<int64_t> sh = {B, H, W}, st = cstrides(sh);
    double scale[2] = {1.0, 1.0};

    // Reference: explicit contiguous strides.
    std::vector<double> out_ref(B * H * W, -1.0);
    DLTensor ti_ref = make_cpu_tensor(in.data(),      sh, st, 64);
    DLTensor to_ref = make_cpu_tensor(out_ref.data(), sh, st, 64);
    ff::cpu::resample(to_ref, ti_ref, /*order=*/1, /*bound=*/3, 0.0, scale, 2, 0);

    // Under test: strides == NULL on both input and output.
    std::vector<double> out_null(B * H * W, -1.0);
    DLTensor ti = make_cpu_tensor(in.data(),       sh, st, 64);
    DLTensor to = make_cpu_tensor(out_null.data(),  sh, st, 64);
    ti.strides = nullptr;
    to.strides = nullptr;
    ff::cpu::resample(to, ti, /*order=*/1, /*bound=*/3, 0.0, scale, 2, 0);

    for (int64_t i = 0; i < B * H * W; ++i)
        check_close(out_null[i], out_ref[i], "null_strides");
}

// Companion to the above for the OTHER DLPack descriptor feature this entry
// point used to normalise by hand: a non-zero byte_offset (was folded into the
// data pointer by VOIDPTR, now folded by the importer). Both operands carry an
// offset, and the padding IN FRONT of each is asserted untouched -- so a
// mis-folded offset fails loudly instead of reading right and writing left.
//
// Pins PRE-EXISTING behaviour (passes against the old implementation too);
// byte_offset was not covered for resample before.
void test_byte_offset()
{
    const int64_t B = 2, H = 5, W = 6, N = B * H * W;
    const int64_t PAD = 5;
    const double SENTINEL = -12345.0;

    std::vector<double> in(N);
    for (int64_t i = 0; i < N; ++i) in[i] = 0.1 * i - 3.0;

    std::vector<int64_t> sh = {B, H, W}, st = cstrides(sh);
    double scale[2] = {1.0, 1.0};

    // Reference: zero byte_offset.
    std::vector<double> out_ref(N, -1.0);
    DLTensor ti_ref = make_cpu_tensor(in.data(),      sh, st, 64);
    DLTensor to_ref = make_cpu_tensor(out_ref.data(), sh, st, 64);
    ff::cpu::resample(to_ref, ti_ref, /*order=*/1, /*bound=*/3, 0.0, scale, 2, 0);

    // Under test: byte_offset != 0 on both input and output, padded in front.
    std::vector<double> in_off(PAD + N, SENTINEL);
    for (int64_t i = 0; i < N; ++i) in_off[PAD + i] = in[i];
    std::vector<double> out_off(PAD + N, SENTINEL);

    DLTensor ti = make_cpu_tensor(in_off.data(),  sh, st, 64);
    DLTensor to = make_cpu_tensor(out_off.data(), sh, st, 64);
    ti.byte_offset = PAD * (int64_t)sizeof(double);
    to.byte_offset = PAD * (int64_t)sizeof(double);
    ff::cpu::resample(to, ti, /*order=*/1, /*bound=*/3, 0.0, scale, 2, 0);

    for (int64_t i = 0; i < N; ++i)
        check_close(out_off[PAD + i], out_ref[i], "byte_offset");
    for (int64_t p = 0; p < PAD; ++p) {
        ++g_checks;
        if (out_off[p] != SENTINEL) {
            ++g_failures;
            std::printf("  FAIL [resize.byte_offset_pad]: pad %lld written\n",
                        (long long)p);
        }
    }
}

// ---------------------------------------------------------------------------
// Independent oracle: the pre-teeny hand-unrolled Multiscale::resize (unchanged
// in kernels/resize.h). The teeny path (resample -> impl/resize.h -> vox::pull_at)
// must match it voxel-for-voxel across orders and bounds -- including the runtime
// Dynamic-route bounds (Replicate/DCT1/DST1), whose reference here is the static
// Multiscale instantiation. Tensors are contiguous with a single batch dim.
// ---------------------------------------------------------------------------
namespace rz = ff::cpu::resize;
using spl_t = ff::cpu::spline::type;
using bnd_t = ff::cpu::bound::type;

template <int D>
int64_t prod_spatial(const int64_t sp[D]) { int64_t p = 1; for (int d=0;d<D;++d) p*=sp[d]; return p; }

template <int D, spl_t I, bnd_t Bd, typename T>
void oracle_check(uint8_t bits, double tol)
{
    // batch(2) x spatial input, resampled to a different spatial shape
    const int64_t Bt = 2;
    int64_t insp[D], outsp[D];
    for (int d = 0; d < D; ++d) { insp[d] = 5 + d; outsp[d] = 7 - d; }  // 1D 5->7; 2D {5,6}->{7,6}
    const int64_t innum = prod_spatial<D>(insp), outnum = prod_spatial<D>(outsp);

    std::vector<T> in(Bt*innum), out(Bt*outnum, (T)-123.0);
    for (size_t i = 0; i < in.size(); ++i) in[i] = (T)(std::sin(0.37*i) + 0.11*i);

    std::vector<int64_t> shi(D+1), sho(D+1);
    shi[0] = Bt; sho[0] = Bt;
    for (int d = 0; d < D; ++d) { shi[d+1] = insp[d]; sho[d+1] = outsp[d]; }
    std::vector<int64_t> sti = cstrides(shi), sto = cstrides(sho);

    DLTensor ti = make_cpu_tensor(in.data(),  shi, sti, bits);
    DLTensor to = make_cpu_tensor(out.data(), sho, sto, bits);

    double scale[D]; for (int d=0;d<D;++d) scale[d] = (double)insp[d] / (double)outsp[d];
    const double shift = 0.5;
    ff::cpu::resample(to, ti, (int8_t)(int)I, (int8_t)(int)Bd, shift, scale, D, 0);

    // oracle: per (batch, out-voxel), the untouched Multiscale gather
    int64_t inspat_stride[D];
    { int64_t acc = 1; for (int d = D-1; d >= 0; --d) { inspat_stride[d] = acc; acc *= insp[d]; } }
    for (int64_t b = 0; b < Bt; ++b) {
        const T* inb = in.data() + b*innum;
        for (int64_t j = 0; j < outnum; ++j) {
            int64_t idx[D], rem = j;
            for (int d = D-1; d >= 0; --d) { idx[d] = rem % outsp[d]; rem /= outsp[d]; }
            T expv;
            rz::Multiscale<D, I, Bd>::resize(&expv, inb, idx, insp, inspat_stride, scale, shift);
            check_close((double)out[b*outnum + j], (double)expv, "oracle", tol);
        }
    }
}

// Run one (D, order, bound) combo for both dtypes. Callers pass only combos that
// resample() instantiates under FF_TEST_SPARSE (Linear/Cubic -> any bound;
// Nearest/Quadratic -> DCT2).
template <int D, spl_t I, bnd_t Bd>
void oracle_both()
{
    oracle_check<D, I, Bd, double>(64, 1e-9);
    oracle_check<D, I, Bd, float >(32, 3e-3);
}

// B5: higher-order spline value check. `resample` treats the input as spline
// COEFFICIENTS, and B-splines of any order reproduce degree-1 polynomials, so a
// linear ramp of coefficients reconstructs the analytic ramp exactly -- in the
// DEEP INTERIOR, where the spline support does not reach the (non-linear-
// extending) boundary. This exercises orders 2..7 (instantiated but never
// value-checked under FF_TEST_SPARSE); DCT2 is used because the sparse build
// only instantiates DCT2 for the non-{Linear,Cubic} orders.
template <typename T>
void test_ramp_reproduce_order(int8_t order, int factor, uint8_t bits, double tol)
{
    const int64_t ni = 20;
    const int64_t no = ni * factor;
    const double a = 2.0, b = 0.75;
    std::vector<T> in(ni), out(no, (T)0.0);
    for (int64_t i = 0; i < ni; ++i) in[i] = (T)(a + b * i);

    std::vector<int64_t> shi = {ni}, sti = cstrides(shi);
    std::vector<int64_t> sho = {no}, sto = cstrides(sho);
    DLTensor ti = make_cpu_tensor(in.data(),  shi, sti, bits);
    DLTensor to = make_cpu_tensor(out.data(), sho, sto, bits);

    double scale[1] = {1.0 / factor};
    ff::cpu::resample(to, ti, order, /*bound=*/3 /*DCT2*/, 0.0, scale, 1, 0);

    // spline half-support (radius) in input-node units; only check output
    // coords whose stencil stays clear of both boundaries.
    const double radius = 0.5 * (double)(order + 1);
    for (int64_t w = 0; w < no; ++w) {
        double x = (double)w / factor;
        if (x >= radius && x <= (double)(ni - 1) - radius)
            check_close((double)out[w], a + b * x, "ramp_order", tol);
    }
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
    // double (bits=64) -- the original coverage.
    test_identity_1d<double>(0, 64, 1e-4); // nearest
    test_identity_1d<double>(1, 64, 1e-4); // linear
    test_ramp_upsample<double>(2, 64, 1e-4);
    test_ramp_upsample<double>(3, 64, 1e-4);
    test_ramp_upsample<double>(4, 64, 1e-4);
    test_identity_2d<double>(64, 1e-4);
    // B1: float (bits=32) with looser tolerances -- previously uninstantiated.
    test_identity_1d<float>(0, 32, 2e-3); // nearest
    test_identity_1d<float>(1, 32, 2e-3); // linear
    test_ramp_upsample<float>(2, 32, 2e-3);
    test_ramp_upsample<float>(3, 32, 2e-3);
    test_ramp_upsample<float>(4, 32, 2e-3);
    test_identity_2d<float>(32, 2e-3);
    // B5: higher-order spline value checks (Quadratic..SeventhOrder), which the
    // order-0/1 checks above never reached. double (tol 1e-6) + float Cubic.
    for (int8_t order = 2; order <= 7; ++order)
        test_ramp_reproduce_order<double>(order, 2, 64, 1e-6);
    test_ramp_reproduce_order<double>(3, 3, 64, 1e-6);   // Cubic, odd factor
    test_ramp_reproduce_order<float >(3, 2, 32, 2e-3);   // Cubic, float
    // B2: 64-bit index + non-contiguous stride path.
    test_inflated_stride();
    test_null_strides_contiguous();
    test_byte_offset();
    test_bad_dtype_throws();

    // Brute-force oracle vs the untouched Multiscale gather. Static bounds
    // (DCT2/DFT/Zero/DST2) AND the runtime Dynamic-route bounds
    // (Replicate/DCT1/DST1) must both match. Combos honour FF_TEST_SPARSE:
    // Linear/Cubic take every bound, Nearest/Quadratic only DCT2.
    // -- 1D, order 1 (Linear): full bound sweep, incl. the Dynamic route.
    oracle_both<1, spl_t::Linear, bnd_t::DCT2>();
    oracle_both<1, spl_t::Linear, bnd_t::DFT>();
    oracle_both<1, spl_t::Linear, bnd_t::Zero>();
    oracle_both<1, spl_t::Linear, bnd_t::DST2>();
    oracle_both<1, spl_t::Linear, bnd_t::Replicate>();   // Dynamic route
    oracle_both<1, spl_t::Linear, bnd_t::DCT1>();         // Dynamic route
    oracle_both<1, spl_t::Linear, bnd_t::DST1>();         // Dynamic route
    // -- 1D, order 3 (Cubic): a representative static + Dynamic pair.
    oracle_both<1, spl_t::Cubic, bnd_t::DCT2>();
    oracle_both<1, spl_t::Cubic, bnd_t::Replicate>();     // Dynamic route
    // -- 1D, orders 0/2 (Nearest/Quadratic): DCT2 only (sparse-legal).
    oracle_both<1, spl_t::Nearest,   bnd_t::DCT2>();
    oracle_both<1, spl_t::Quadratic, bnd_t::DCT2>();
    // -- 2D: static + Dynamic at Linear and Cubic.
    oracle_both<2, spl_t::Linear, bnd_t::DCT2>();
    oracle_both<2, spl_t::Linear, bnd_t::Replicate>();    // Dynamic route
    oracle_both<2, spl_t::Cubic,  bnd_t::DCT2>();
    oracle_both<2, spl_t::Cubic,  bnd_t::DST1>();          // Dynamic route
    // -- 3D: Linear static + Dynamic, Cubic static.
    oracle_both<3, spl_t::Linear, bnd_t::DCT2>();
    oracle_both<3, spl_t::Linear, bnd_t::Replicate>();    // Dynamic route
    oracle_both<3, spl_t::Cubic,  bnd_t::DCT2>();
    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
