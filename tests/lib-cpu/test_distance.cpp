// CPU unit tests for the distance module.
//
// Exercises the full stack (dispatch -> impl -> kernels) for
// ff::cpu::dt_euclidean and ff::cpu::dt_l1 by comparing against a
// brute-force O(n^2) reference. Both transforms operate along the last
// axis; all leading axes are treated as batch.
//
// Build (from fastfields-cpu-lib):
//   clang++ -std=c++11 -O2 -I. tests/test_distance.cpp distance.cpp -o build/test_distance
//   ./build/test_distance

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <limits>
#include <random>
#include <stdexcept>
#include "dlpack.h"
#include "distance.h"

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

template <typename T>
DLTensor make_cpu_tensor(T* data, std::vector<int64_t>& shape,
                         std::vector<int64_t>& strides, uint8_t bits,
                         bool null_strides = false, uint64_t byte_offset = 0)
{
    DLTensor t;
    t.data                 = static_cast<void*>(data);
    t.device.device_type   = kDLCPU;
    t.device.device_id     = 0;
    t.ndim                 = static_cast<int32_t>(shape.size());
    t.dtype.code           = static_cast<uint8_t>(kDLFloat);
    t.dtype.bits           = bits;
    t.dtype.lanes          = 1;
    t.shape                = shape.data();
    // DLPack lets `strides` be NULL, meaning "compact row-major".
    t.strides              = null_strides ? nullptr : strides.data();
    t.byte_offset          = byte_offset;
    return t;
}

// Contiguous row-major strides (in elements, as DLPack expects).
std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> s(shape.size());
    int64_t acc = 1;
    for (int64_t d = (int64_t)shape.size() - 1; d >= 0; --d) {
        s[d] = acc;
        acc *= shape[d];
    }
    return s;
}

// Brute-force 1D transform along the last axis for one batch row.
// euclidean: out[i] = min_j ( in[j] + (w*(i-j))^2 )
// l1:        out[i] = min_j ( in[j] + w*|i-j|      )
void brute_row(const std::vector<double>& in, std::vector<double>& out,
              double w, bool euclidean)
{
    int64_t n = (int64_t)in.size();
    out.assign(n, INF);
    for (int64_t i = 0; i < n; ++i) {
        double best = INF;
        for (int64_t j = 0; j < n; ++j) {
            double d = euclidean ? (w * (i - j)) * (w * (i - j))
                                 : w * std::fabs((double)(i - j));
            double cand = in[j] + d;
            if (cand < best) best = cand;
        }
        out[i] = best;
    }
}

int g_failures = 0;
int g_checks   = 0;

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

// `null_strides` / `pad` exercise the two DLPack descriptor features the
// l1/euclidean path used to normalise by hand (see test_descriptor_variants):
// a NULL `strides` field, and a non-zero `byte_offset` (expressed here as
// `pad` LEADING elements the transform must not touch). Both default off, so
// every pre-existing call site is bit-for-bit the case it always was.
template <typename scalar_t>
void run_case(int64_t nbatch, int64_t n, double w, bool euclidean, uint8_t bits,
             unsigned seed, const char* tag = nullptr,
             bool null_strides = false, int64_t pad = 0)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    // A sentinel in front of byte_offset: it must survive the call untouched.
    const scalar_t SENTINEL = static_cast<scalar_t>(-12345.0);
    std::vector<scalar_t> data(pad + nbatch * n, SENTINEL);
    std::vector<std::vector<double>> rows_in(nbatch);
    for (int64_t b = 0; b < nbatch; ++b) {
        rows_in[b].resize(n);
        // Ensure at least one feature (0) per row so distances are finite.
        int64_t forced = (int64_t)(u(rng) * n) % n;
        for (int64_t i = 0; i < n; ++i) {
            bool feature = (i == forced) || (u(rng) < 0.3);
            double v = feature ? 0.0 : INF;
            rows_in[b][i] = v;
            data[pad + b * n + i] = static_cast<scalar_t>(v);
        }
    }

    std::vector<int64_t> shape   = {nbatch, n};
    std::vector<int64_t> strides = contiguous_strides(shape);
    DLTensor t = make_cpu_tensor(data.data(), shape, strides, bits, null_strides,
                                 static_cast<uint64_t>(pad) * sizeof(scalar_t));

    if (euclidean) ff::cpu::dt_euclidean(t, w, 0);
    else           ff::cpu::dt_l1(t, w, 0);

    const char* what = tag ? tag : (euclidean ? "eucl" : "l1");
    for (int64_t b = 0; b < nbatch; ++b) {
        std::vector<double> ref;
        brute_row(rows_in[b], ref, w, euclidean);
        for (int64_t i = 0; i < n; ++i)
            check_close((double)data[pad + b * n + i], ref[i], what);
    }
    // Nothing before byte_offset may have been written.
    for (int64_t i = 0; i < pad; ++i)
        check_close((double)data[i], (double)SENTINEL, "pad-untouched");
}

// B4: an unsupported dtype (float16) must throw, not silently no-op. dt_euclidean
// and dt_l1 operate in place on a single tensor, so there is no cross-tensor
// shape-mismatch case to exercise -- only the dtype-dispatch guard.
void test_bad_dtype_throws()
{
    std::vector<uint16_t> data(2 * 4, 0);                 // float16 payload
    std::vector<int64_t> shape = {2, 4}, strides = contiguous_strides(shape);
    DLTensor t = make_cpu_tensor(data.data(), shape, strides, 16);
    bool threw = false;
    try { ff::cpu::dt_euclidean(t, 1.0, 0); } catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [distance.euclidean.bad_dtype_throws]\n"); }
    threw = false;
    try { ff::cpu::dt_l1(t, 1.0, 0); } catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [distance.l1.bad_dtype_throws]\n"); }
}

// cpu-lib#72: the two DLPack descriptor features dt_l1/dt_euclidean must keep
// honouring now that the import goes through teeny's from_dlpack instead of
// this repo's hand-rolled ContiguousStrides (NULL strides -> compact
// row-major) and VOIDPTR (fold byte_offset into the data pointer).
// `strides == NULL` in particular is listed in MIGRATION.md as a historical
// soft spot and was previously unexercised by any test, so pin both here:
// each variant must reproduce the fully-explicit descriptor's answer, and the
// byte_offset cases must leave the padding in front of the tensor untouched.
void test_descriptor_variants()
{
    // NULL strides (compact row-major), both transforms, both dtypes.
    run_case<float >(4, 17, 1.0, true , 32, 11, "eucl.null_strides", true, 0);
    run_case<double>(3, 23, 1.3, true , 64, 12, "eucl.null_strides", true, 0);
    run_case<float >(4, 17, 1.0, false, 32, 13, "l1.null_strides"  , true, 0);
    run_case<double>(3, 23, 1.3, false, 64, 14, "l1.null_strides"  , true, 0);

    // byte_offset != 0 (explicit strides), both transforms, both dtypes.
    run_case<float >(4, 17, 1.0, true , 32, 21, "eucl.byte_offset" , false, 5);
    run_case<double>(3, 23, 1.3, true , 64, 22, "eucl.byte_offset" , false, 5);
    run_case<float >(4, 17, 1.0, false, 32, 23, "l1.byte_offset"   , false, 5);
    run_case<double>(3, 23, 1.3, false, 64, 24, "l1.byte_offset"   , false, 5);

    // Both at once: NULL strides AND a non-zero byte_offset.
    run_case<float >(4, 17, 1.0, true , 32, 31, "eucl.null+offset" , true , 5);
    run_case<double>(3, 23, 1.3, false, 64, 32, "l1.null+offset"   , true , 5);
}

} // namespace

int main()
{
    std::printf("distance module CPU tests\n");
    test_bad_dtype_throws();
    test_descriptor_variants();
    // float32 and float64, various sizes / spacings, euclidean + l1.
    for (unsigned seed = 1; seed <= 5; ++seed) {
        run_case<float >(4, 17, 1.0, true , 32, seed);
        run_case<float >(3, 33, 0.7, true , 32, seed + 100);
        run_case<double>(4, 17, 1.0, true , 64, seed + 200);
        run_case<double>(2, 40, 2.5, true , 64, seed + 300);
        run_case<float >(4, 17, 1.0, false, 32, seed + 400);
        run_case<double>(3, 29, 1.3, false, 64, seed + 500);
    }
    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
