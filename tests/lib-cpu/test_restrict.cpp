// CPU unit tests for the restrict module.
//
// `restriction` is the adjoint (transpose) of the resize `resample`
// prolongation. We verify the defining adjoint identity numerically:
//     <resample(c), g>_fine  ==  <c, restriction(g)>_coarse
// for random coarse c and fine g, matching order / bound / shift and using
// reciprocal scales (resize: coarse/fine, restrict: fine/coarse).
//
// Build (from fastfields-cpu-lib):
//   clang++ -std=c++17 -O2 -ferror-limit=5 -DTNY_MAX_RANK=64 -I. -I<teeny>/include -I<teeny>/external/cccl/libcudacxx/include tests/test_restrict.cpp resize.cpp restrict.cpp -o build/test_restrict && ./build/test_restrict

#include <cstdio>
#include <cstdint>
#include <cstdlib>
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

template <typename T>
double dotT(const std::vector<T>& a, const std::vector<T>& b)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += (double)a[i] * (double)b[i];
    return s;
}

// 1D adjoint check: coarse nc -> fine nf (factor nf/nc), edge anchor.
// Templated on scalar_t/bits/tol so both double (bits=64) and float (bits=32,
// B1: the dominant ML dtype, previously never instantiated) are exercised.
template <typename T>
void test_adjoint_1d(int64_t nc, int64_t nf, int8_t order, int8_t bound,
                     unsigned seed, uint8_t bits, double tol, double shift = 0.5)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    std::vector<T> c(nc), g(nf);
    for (auto& v : c) v = (T)u(rng);
    for (auto& v : g) v = (T)u(rng);

    std::vector<int64_t> shc = {nc}, stc = cstrides(shc);
    std::vector<int64_t> shf = {nf}, stf = cstrides(shf);

    // P: coarse -> fine (resample). scale = coarse/fine.
    std::vector<T> Pc(nf, (T)0.0);
    {
        DLTensor ti = make_cpu_tensor(c.data(),  shc, stc, bits);
        DLTensor to = make_cpu_tensor(Pc.data(), shf, stf, bits);
        double scale[1] = {(double)nc / (double)nf};
        ff::cpu::resample(to, ti, order, bound, shift, scale, 1, 0);
    }

    // R = P^T: fine -> coarse (restriction). reciprocal scale = fine/coarse.
    std::vector<T> Rg(nc, (T)0.0); // pre-zeroed (restriction accumulates)
    {
        DLTensor ti = make_cpu_tensor(g.data(),  shf, stf, bits);
        DLTensor to = make_cpu_tensor(Rg.data(), shc, stc, bits);
        double scale[1] = {(double)nf / (double)nc};
        ff::cpu::restriction(to, ti, order, bound, shift, scale, 1, 0);
    }

    check_close(dotT(Pc, g), dotT(c, Rg), "adj1d", tol);
}

// Adjoint identity sweep across shift x order x factor x bound. The adjoint
// P^T == R must hold for EVERY anchor, not just shift=0.5. shift=0.0 with even
// orders / integer factors is the multigrid default (bindings default
// spline=2, shift=0.0) and exercises the coarse-side support edges + tie taps
// that a fixed-pad or open-window restrict mis-handles. Respects FF_TEST_SPARSE
// (Linear/Cubic get every bound; Nearest/Quadratic only DCT2).
void test_adjoint_sweep()
{
    const int8_t DCT2 = 3;
    // (order, is_sparse_order) -> bounds to test
    struct Cfg { int8_t order; bool full_bounds; };
    const Cfg cfgs[] = { {0,false}, {1,true}, {2,false}, {3,true} };
    // full bound set (DCT2, DFT, Zero, Replicate, DCT1, DST1); sparse = DCT2 only
    const int8_t full[] = {3 /*DCT2*/, 6 /*DFT*/, 0 /*Zero*/, 1 /*Replicate*/, 2 /*DCT1*/, 4 /*DST1*/};
    const double shifts[] = {0.0, 0.5, 0.25};
    const int64_t factors[] = {2, 3, 4};
    unsigned seed = 1000;
    for (const auto& cf : cfgs)
    for (double sh : shifts)
    for (int64_t f : factors) {
        const int64_t nc = 5, nf = nc * f;
        if (cf.full_bounds) {
            for (int8_t b : full)
                test_adjoint_1d<double>(nc, nf, cf.order, b, ++seed, 64, 1e-9, sh);
        } else {
            test_adjoint_1d<double>(nc, nf, cf.order, DCT2, ++seed, 64, 1e-9, sh);
        }
    }
}

// 2D adjoint check with a batch dim.
template <typename T>
void test_adjoint_2d(int64_t B, int64_t hc, int64_t wc, int64_t hf, int64_t wf,
                     int8_t order, int8_t bound, unsigned seed, uint8_t bits, double tol)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    std::vector<T> c(B * hc * wc), g(B * hf * wf);
    for (auto& v : c) v = (T)u(rng);
    for (auto& v : g) v = (T)u(rng);

    std::vector<int64_t> shc = {B, hc, wc}, stc = cstrides(shc);
    std::vector<int64_t> shf = {B, hf, wf}, stf = cstrides(shf);

    std::vector<T> Pc(B * hf * wf, (T)0.0);
    {
        DLTensor ti = make_cpu_tensor(c.data(),  shc, stc, bits);
        DLTensor to = make_cpu_tensor(Pc.data(), shf, stf, bits);
        double scale[2] = {(double)hc / (double)hf, (double)wc / (double)wf};
        ff::cpu::resample(to, ti, order, bound, 0.5, scale, 2, 0);
    }

    std::vector<T> Rg(B * hc * wc, (T)0.0);
    {
        DLTensor ti = make_cpu_tensor(g.data(),  shf, stf, bits);
        DLTensor to = make_cpu_tensor(Rg.data(), shc, stc, bits);
        double scale[2] = {(double)hf / (double)hc, (double)wf / (double)wc};
        ff::cpu::restriction(to, ti, order, bound, 0.5, scale, 2, 0);
    }

    check_close(dotT(Pc, g), dotT(c, Rg), "adj2d", tol);
}

// B2: 64-bit index + non-contiguous stride path for restriction. A leading
// batch dim of size 2 with a stride >= INT32_MAX makes canUse32BitIndexMath
// return false (it keys on the max element offset), forcing the int64_t
// offset_t instantiation and a strided (non-contiguous) fine-input read. The
// reference is the same restriction on a small contiguous tensor. The big
// buffer is lazily allocated (only the 2 touched planes commit pages); if the
// virtual allocation is refused we skip. float keeps it to ~8.6 GB virtual.
void test_inflated_stride()
{
    const int64_t B = 2, nc = 4, nf = 8;
    const int64_t P = (int64_t)1 << 31;                 // 2147483648 > INT32_MAX

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<float> g(B * nf);
    for (auto& v : g) v = (float)u(rng);

    std::vector<int64_t> shc = {B, nc}, stc = cstrides(shc);
    std::vector<int64_t> shf = {B, nf}, stf = cstrides(shf);
    double scale[1] = {(double)nf / (double)nc};

    // Reference: contiguous fine input.
    std::vector<float> Rg_ref(B * nc, 0.0f);
    {
        DLTensor ti = make_cpu_tensor(g.data(),      shf, stf, 32);
        DLTensor to = make_cpu_tensor(Rg_ref.data(), shc, stc, 32);
        ff::cpu::restriction(to, ti, /*order=*/1, /*bound=*/3, 0.5, scale, /*ndim=*/1, 0);
    }

    // Under test: fine input with an inflated batch stride P.
    const size_t nelem = (size_t)P * (B - 1) + (size_t)nf;
    float * gbig = static_cast<float*>(std::calloc(nelem, sizeof(float)));
    if (!gbig) {
        std::printf("  [restrict inflated-stride] skipped (lazy alloc of %.1f GB refused)\n",
                    nelem * sizeof(float) / 1e9);
        return;
    }
    for (int64_t bt = 0; bt < B; ++bt)
        for (int64_t i = 0; i < nf; ++i) gbig[bt*P + i] = g[bt*nf + i];

    std::vector<float> Rg_str(B * nc, 0.0f);
    std::vector<int64_t> shfi = {B, nf}, stfi = {P, 1};   // inflated, non-contiguous
    DLTensor ti = make_cpu_tensor(gbig,          shfi, stfi, 32);
    DLTensor to = make_cpu_tensor(Rg_str.data(), shc,  stc,  32);
    ff::cpu::restriction(to, ti, /*order=*/1, /*bound=*/3, 0.5, scale, /*ndim=*/1, 0);

    for (int64_t i = 0; i < B * nc; ++i)
        check_close((double)Rg_str[i], (double)Rg_ref[i], "inflated_stride", 1e-4);

    std::free(gbig);
}

// Descriptor variants. DLPack allows DLTensor.strides == NULL for a compact
// row-major tensor, and allows a non-zero byte_offset. This entry point used to
// normalise both by hand (ContiguousStrides / VOIDPTR); they are now folded by
// the importer instead. Either spelling must reproduce the explicit-strides,
// zero-offset result exactly -- on BOTH operands, since restriction takes two.
// The byte_offset case also asserts the padding IN FRONT of each offset is
// untouched, so a mis-folded offset fails loudly rather than reading right and
// writing left.
//
// NB restriction ACCUMULATES into out, so every output buffer here is
// pre-zeroed (and the pad sentinel is therefore a genuine "was not written"
// check, not merely "was not overwritten").
//
// These pin PRE-EXISTING behaviour (they pass against the old implementation
// too); neither variant was covered for restriction before.
template <typename T>
void test_descriptor_variants_dtype(uint8_t bits, unsigned seed, double tol)
{
    const int64_t B = 2, nc = 5, nf = 10;
    const int64_t NF = B * nf, NC = B * nc;
    const int8_t order = 1, bound = 3 /*DCT2*/;
    const int64_t PAD = 5;
    const T SENTINEL = (T)-12345.0;
    double scale[1] = {(double)nf / (double)nc};

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<T> g(NF);
    for (auto & v : g) v = (T)u(rng);

    std::vector<int64_t> shf = {B, nf}, stf = cstrides(shf);
    std::vector<int64_t> shc = {B, nc}, stc = cstrides(shc);

    // Reference: explicit contiguous strides, byte_offset == 0.
    std::vector<T> ref(NC, (T)0);
    {
        DLTensor ti = make_cpu_tensor(g.data(),   shf, stf, bits);
        DLTensor to = make_cpu_tensor(ref.data(), shc, stc, bits);
        ff::cpu::restriction(to, ti, order, bound, 0.5, scale, 1, 0);
    }

    // (1) strides == NULL on BOTH operands
    {
        std::vector<T> out(NC, (T)0);
        DLTensor ti = make_cpu_tensor(g.data(),   shf, stf, bits);
        DLTensor to = make_cpu_tensor(out.data(), shc, stc, bits);
        ti.strides = nullptr; to.strides = nullptr;
        ff::cpu::restriction(to, ti, order, bound, 0.5, scale, 1, 0);
        for (int64_t i = 0; i < NC; ++i)
            check_close((double)out[i], (double)ref[i], "restrict.null_strides", tol);
    }

    // (2) byte_offset != 0 on BOTH operands, each with a sentinel pad in front
    {
        std::vector<T> gin(PAD + NF, SENTINEL);
        for (int64_t i = 0; i < NF; ++i) gin[PAD + i] = g[i];
        std::vector<T> out(PAD + NC, SENTINEL);
        for (int64_t i = 0; i < NC; ++i) out[PAD + i] = (T)0;   // pre-zero the live region
        DLTensor ti = make_cpu_tensor(gin.data(), shf, stf, bits);
        DLTensor to = make_cpu_tensor(out.data(), shc, stc, bits);
        ti.byte_offset = PAD * (int64_t)sizeof(T);
        to.byte_offset = PAD * (int64_t)sizeof(T);
        ff::cpu::restriction(to, ti, order, bound, 0.5, scale, 1, 0);
        for (int64_t i = 0; i < NC; ++i)
            check_close((double)out[PAD + i], (double)ref[i], "restrict.byte_offset", tol);
        for (int64_t p = 0; p < PAD; ++p) {
            ++g_checks;
            if (out[p] != SENTINEL) {
                ++g_failures;
                std::printf("  FAIL [restrict.byte_offset_pad]: pad %lld written\n",
                            (long long)p);
            }
        }
    }

    // (3) both at once
    {
        std::vector<T> gin(PAD + NF, SENTINEL);
        for (int64_t i = 0; i < NF; ++i) gin[PAD + i] = g[i];
        std::vector<T> out(PAD + NC, SENTINEL);
        for (int64_t i = 0; i < NC; ++i) out[PAD + i] = (T)0;
        DLTensor ti = make_cpu_tensor(gin.data(), shf, stf, bits);
        DLTensor to = make_cpu_tensor(out.data(), shc, stc, bits);
        ti.strides = nullptr; to.strides = nullptr;
        ti.byte_offset = PAD * (int64_t)sizeof(T);
        to.byte_offset = PAD * (int64_t)sizeof(T);
        ff::cpu::restriction(to, ti, order, bound, 0.5, scale, 1, 0);
        for (int64_t i = 0; i < NC; ++i)
            check_close((double)out[PAD + i], (double)ref[i],
                        "restrict.null_strides_and_offset", tol);
        for (int64_t p = 0; p < PAD; ++p) {
            ++g_checks;
            if (out[p] != SENTINEL) {
                ++g_failures;
                std::printf("  FAIL [restrict.both_pad]: pad %lld written\n",
                            (long long)p);
            }
        }
    }
}

void test_descriptor_variants()
{
    test_descriptor_variants_dtype<double>(64, 8001, 1e-9);
    test_descriptor_variants_dtype<float >(32, 8002, 1e-5);
}

} // namespace

// Regression (A4): an unsupported dtype must throw, not silently no-op.
void test_bad_dtype_throws()
{
    std::vector<uint16_t> c(8, 0), out(4, 0);            // float16 payload
    std::vector<int64_t> shf = {8}, stf = cstrides(shf);
    std::vector<int64_t> shc = {4}, stc = cstrides(shc);
    DLTensor ti = make_cpu_tensor(c.data(),   shf, stf, 16); // kDLFloat, 16 bits
    DLTensor to = make_cpu_tensor(out.data(), shc, stc, 16);
    bool threw = false;
    try {
        ff::cpu::restriction(to, ti, /*order=*/1, /*bound=*/3, 0.0, nullptr, 1, 0);
    } catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [restrict.bad_dtype_throws]\n"); }
}

int main()
{
    std::printf("restrict module CPU tests\n");
    test_bad_dtype_throws();
    test_descriptor_variants();
    for (unsigned s = 1; s <= 4; ++s) {
        // double (bits=64) -- the original coverage.
        // 1D, several orders / bounds / factors
        test_adjoint_1d<double>(4, 8,  1, 3 /*DCT2*/, s,      64, 1e-9);
        test_adjoint_1d<double>(4, 8,  3, 3, s + 10,          64, 1e-9);
        test_adjoint_1d<double>(5, 10, 2, 3, s + 20,          64, 1e-9);
        test_adjoint_1d<double>(4, 12, 1, 1 /*Replicate*/, s + 30, 64, 1e-9);
        test_adjoint_1d<double>(6, 9,  3, 3, s + 40,          64, 1e-9); // non-integer factor
        // 2D with batch
        test_adjoint_2d<double>(2, 3, 4, 6, 8, 1, 3, s + 50,  64, 1e-9);
        test_adjoint_2d<double>(2, 3, 3, 6, 6, 3, 3, s + 60,  64, 1e-9);
        // B1: float (bits=32) with a looser adjoint tolerance -- the same
        // identity must hold; float32 was previously never instantiated.
        test_adjoint_1d<float>(4, 8,  1, 3, s,               32, 1e-4);
        test_adjoint_1d<float>(4, 8,  3, 3, s + 10,          32, 1e-4);
        test_adjoint_1d<float>(5, 10, 2, 3, s + 20,          32, 1e-4);
        test_adjoint_1d<float>(4, 12, 1, 1, s + 30,          32, 1e-4);
        test_adjoint_1d<float>(6, 9,  3, 3, s + 40,          32, 1e-4);
        test_adjoint_2d<float>(2, 3, 4, 6, 8, 1, 3, s + 50,  32, 1e-4);
        test_adjoint_2d<float>(2, 3, 3, 6, 6, 3, 3, s + 60,  32, 1e-4);
    }
    // B2: 64-bit index + non-contiguous stride path.
    test_inflated_stride();
    // Adjoint identity across shift x order x factor x bound (incl. shift=0).
    test_adjoint_sweep();
    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
