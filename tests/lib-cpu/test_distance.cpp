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
#include "dlpack.h"
#include "distance.h"

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();

template <typename T>
DLTensor make_cpu_tensor(T* data, std::vector<int64_t>& shape,
                         std::vector<int64_t>& strides, uint8_t bits)
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
    t.strides              = strides.data();
    t.byte_offset          = 0;
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

template <typename scalar_t>
void run_case(int64_t nbatch, int64_t n, double w, bool euclidean, uint8_t bits,
             unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    std::vector<scalar_t> data(nbatch * n);
    std::vector<std::vector<double>> rows_in(nbatch);
    for (int64_t b = 0; b < nbatch; ++b) {
        rows_in[b].resize(n);
        // Ensure at least one feature (0) per row so distances are finite.
        int64_t forced = (int64_t)(u(rng) * n) % n;
        for (int64_t i = 0; i < n; ++i) {
            bool feature = (i == forced) || (u(rng) < 0.3);
            double v = feature ? 0.0 : INF;
            rows_in[b][i] = v;
            data[b * n + i] = static_cast<scalar_t>(v);
        }
    }

    std::vector<int64_t> shape   = {nbatch, n};
    std::vector<int64_t> strides = contiguous_strides(shape);
    DLTensor t = make_cpu_tensor(data.data(), shape, strides, bits);

    if (euclidean) ff::cpu::dt_euclidean(t, w, 0);
    else           ff::cpu::dt_l1(t, w, 0);

    for (int64_t b = 0; b < nbatch; ++b) {
        std::vector<double> ref;
        brute_row(rows_in[b], ref, w, euclidean);
        for (int64_t i = 0; i < n; ++i)
            check_close((double)data[b * n + i], ref[i], euclidean ? "eucl" : "l1");
    }
}

} // namespace

int main()
{
    std::printf("distance module CPU tests\n");
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
