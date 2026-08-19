// CPU unit tests for the spline-distance dispatch (ff::cpu::dt_spline_*).
//
// Exercises the full stack (dispatch -> impl -> kernels) for
//   dt_spline_table / dt_spline_brent / dt_spline_gaussnewton
// against an independent analytic reference.
//
// The reference uses a *linear* spline (order 1). For a linear spline the
// coefficients ARE the node values, and the spline evaluated at a continuous
// node coordinate t in [0, N-1] is the linear interpolation of the two
// surrounding control points. This lets us compute the exact squared distance
// from a query point to the spline sampled at a dictionary of times, with no
// dependency on the kernel under test.
//
// Contract (see fastfields-lib/distance.h), fully-batched form used by the
// low-level dispatch (jitfields semantics):
//   loc    (*B, D)          query points
//   coeff  (*B, N, D)       spline control points (N = npoints)
//   times  (*B, K)          candidate node coordinates in [0, N-1]
//   time   (*B,)            output best time
//   dist   (*B,)            output best squared distance
//
// Build (from fastfields-cpu-lib):
//   clang++ -std=c++11 -O2 -I. tests/test_distance_spline.cpp distance.cpp \
//       -o build/test_distance_spline && ./build/test_distance_spline

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <limits>
#include <random>
#include <stdexcept>
#include "dlpack.h"
#include "distance.h"
#include "impl/kernels/spline.h"
#include "impl/kernels/bounds.h"

// Spline weight / boundary helpers reused to build an independent reference for
// the higher-order (Cubic) spline-distance path (B5).
using ff::cpu::spline::type;
using btype = ff::cpu::bound::type;

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

int g_failures = 0;
int g_checks   = 0;

void check_close(double a, double b, const char* what, double tol = 1e-3)
{
    ++g_checks;
    double diff  = std::fabs(a - b);
    double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    if (diff / scale > tol) {
        ++g_failures;
        std::printf("  MISMATCH [%s]: got %.6g expected %.6g\n", what, a, b);
    }
}

void check_true(bool cond, const char* what)
{
    ++g_checks;
    if (!cond) {
        ++g_failures;
        std::printf("  FAIL [%s]\n", what);
    }
}

// Linear-spline evaluation at node coordinate t (clamped to [0, N-1]).
// control: N points of dimension D, row-major.
void linear_eval(const std::vector<double>& control, int64_t N, int64_t D,
                 double t, std::vector<double>& out)
{
    if (t < 0)       t = 0;
    if (t > N - 1)   t = N - 1;
    int64_t k0 = (int64_t)std::floor(t);
    if (k0 > N - 2)  k0 = (N >= 2 ? N - 2 : 0);
    int64_t k1 = (N >= 2 ? k0 + 1 : 0);
    double frac = t - (double)k0;
    out.resize(D);
    for (int64_t d = 0; d < D; ++d)
        out[d] = (1.0 - frac) * control[k0 * D + d] + frac * control[k1 * D + d];
}

// Reference best (time, sqdist) over a dictionary of candidate times.
void linear_ref(const std::vector<double>& control, int64_t N, int64_t D,
                const std::vector<double>& loc,
                const std::vector<double>& times,
                double& best_time, double& best_dist)
{
    best_dist = INF;
    best_time = 0;
    std::vector<double> val;
    for (double t : times) {
        linear_eval(control, N, D, t, val);
        double d = 0;
        for (int64_t k = 0; k < D; ++k) {
            double diff = val[k] - loc[k];
            d += diff * diff;
        }
        if (d < best_dist) { best_dist = d; best_time = t; }
    }
}

// int8_t spline/bound constants (mirror fastfields-lib/distance.h).
constexpr int8_t SPLINE_LINEAR = 1;
constexpr int8_t SPLINE_CUBIC  = 3;
constexpr int8_t BOUND_DCT2    = 3;

// ---------------------------------------------------------------------------
// Fully-batched table test: loc (B,D), coeff (B,N,D), times (B,K).
// The same spline is replicated across the batch (physical tiling, non-zero
// batch strides) so the per-batch offset machinery is genuinely exercised.
// ---------------------------------------------------------------------------
template <typename scalar_t>
void run_table_batched(int64_t B, int64_t N, int64_t D, int64_t K, uint8_t bits,
                       unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-2.0, 2.0);

    // One shared spline (N control points, D dims).
    std::vector<double> control(N * D);
    for (auto& c : control) c = u(rng);

    // Candidate node coordinates spanning [0, N-1] (plus a couple of exact nodes).
    std::vector<double> cand(K);
    for (int64_t j = 0; j < K; ++j)
        cand[j] = (double)(N - 1) * (double)j / (double)(K - 1);

    // Batched device tensors (spline + times tiled across B).
    std::vector<scalar_t> loc(B * D), coeff(B * N * D), times(B * K);
    std::vector<double>   loc_d(B * D);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t d = 0; d < D; ++d) {
            double v = u(rng);
            loc_d[b * D + d]        = v;
            loc[b * D + d]          = (scalar_t)v;
        }
        for (int64_t i = 0; i < N * D; ++i)
            coeff[b * N * D + i] = (scalar_t)control[i];
        for (int64_t j = 0; j < K; ++j)
            times[b * K + j] = (scalar_t)cand[j];
    }

    std::vector<scalar_t> time_out(B, 0), dist_out(B, (scalar_t)INF);

    std::vector<int64_t> sh_loc   = {B, D},    st_loc   = contiguous_strides(sh_loc);
    std::vector<int64_t> sh_coeff = {B, N, D}, st_coeff = contiguous_strides(sh_coeff);
    std::vector<int64_t> sh_times = {B, K},    st_times = contiguous_strides(sh_times);
    std::vector<int64_t> sh_out   = {B},       st_out   = contiguous_strides(sh_out);

    DLTensor t_loc   = make_cpu_tensor(loc.data(),   sh_loc,   st_loc,   bits);
    DLTensor t_coeff = make_cpu_tensor(coeff.data(), sh_coeff, st_coeff, bits);
    DLTensor t_times = make_cpu_tensor(times.data(), sh_times, st_times, bits);
    DLTensor t_time  = make_cpu_tensor(time_out.data(), sh_out, st_out,  bits);
    DLTensor t_dist  = make_cpu_tensor(dist_out.data(), sh_out, st_out,  bits);

    ff::cpu::dt_spline_table(t_time, t_dist, t_loc, t_coeff, t_times,
                             SPLINE_LINEAR, BOUND_DCT2, 0);

    for (int64_t b = 0; b < B; ++b) {
        std::vector<double> loc_row(loc_d.begin() + b * D, loc_d.begin() + (b + 1) * D);
        double ref_t, ref_d;
        linear_ref(control, N, D, loc_row, cand, ref_t, ref_d);
        check_close((double)dist_out[b], ref_d, "table.dist");
        check_close((double)time_out[b], ref_t, "table.time");
    }
}

// ---------------------------------------------------------------------------
// nbatch == 0 case: loc (D,), coeff (N,D), times (K,), scalar outputs.
// This is the literal documented shape (coeff (N,D), times (K,)).
// ---------------------------------------------------------------------------
template <typename scalar_t>
void run_table_scalar(int64_t N, int64_t D, int64_t K, uint8_t bits, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-2.0, 2.0);

    std::vector<double> control(N * D);
    for (auto& c : control) c = u(rng);
    std::vector<double> cand(K);
    for (int64_t j = 0; j < K; ++j)
        cand[j] = (double)(N - 1) * (double)j / (double)(K - 1);

    std::vector<double>   loc_d(D);
    std::vector<scalar_t> loc(D), coeff(N * D), times(K);
    for (int64_t d = 0; d < D; ++d) { loc_d[d] = u(rng); loc[d] = (scalar_t)loc_d[d]; }
    for (int64_t i = 0; i < N * D; ++i) coeff[i] = (scalar_t)control[i];
    for (int64_t j = 0; j < K; ++j)     times[j] = (scalar_t)cand[j];

    scalar_t time_out = 0, dist_out = (scalar_t)INF;

    std::vector<int64_t> sh_loc   = {D},    st_loc   = contiguous_strides(sh_loc);
    std::vector<int64_t> sh_coeff = {N, D}, st_coeff = contiguous_strides(sh_coeff);
    std::vector<int64_t> sh_times = {K},    st_times = contiguous_strides(sh_times);
    std::vector<int64_t> sh_out; // 0-d
    std::vector<int64_t> st_out; // 0-d

    DLTensor t_loc   = make_cpu_tensor(loc.data(),   sh_loc,   st_loc,   bits);
    DLTensor t_coeff = make_cpu_tensor(coeff.data(), sh_coeff, st_coeff, bits);
    DLTensor t_times = make_cpu_tensor(times.data(), sh_times, st_times, bits);
    DLTensor t_time  = make_cpu_tensor(&time_out,    sh_out,   st_out,   bits);
    DLTensor t_dist  = make_cpu_tensor(&dist_out,    sh_out,   st_out,   bits);

    ff::cpu::dt_spline_table(t_time, t_dist, t_loc, t_coeff, t_times,
                             SPLINE_LINEAR, BOUND_DCT2, 0);

    double ref_t, ref_d;
    linear_ref(control, N, D, loc_d, cand, ref_t, ref_d);
    check_close((double)dist_out, ref_d, "table0.dist");
    check_close((double)time_out, ref_t, "table0.time");
}

// ---------------------------------------------------------------------------
// Brent / Gauss-Newton smoke test (batched). Seed from the table result, then
// refine in-place; refined squared distance must be finite, non-negative, and
// no worse than the table result (within tolerance).
// ---------------------------------------------------------------------------
template <typename scalar_t>
void run_refine_batched(int64_t B, int64_t N, int64_t D, int64_t K, uint8_t bits,
                        unsigned seed, bool use_brent)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-2.0, 2.0);

    std::vector<double> control(N * D);
    for (auto& c : control) c = u(rng);
    std::vector<double> cand(K);
    for (int64_t j = 0; j < K; ++j)
        cand[j] = (double)(N - 1) * (double)j / (double)(K - 1);

    std::vector<scalar_t> loc(B * D), coeff(B * N * D), times(B * K);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t d = 0; d < D; ++d) loc[b * D + d] = (scalar_t)u(rng);
        for (int64_t i = 0; i < N * D; ++i) coeff[b * N * D + i] = (scalar_t)control[i];
        for (int64_t j = 0; j < K; ++j) times[b * K + j] = (scalar_t)cand[j];
    }

    std::vector<scalar_t> time_out(B, 0), dist_out(B, (scalar_t)INF);

    std::vector<int64_t> sh_loc   = {B, D},    st_loc   = contiguous_strides(sh_loc);
    std::vector<int64_t> sh_coeff = {B, N, D}, st_coeff = contiguous_strides(sh_coeff);
    std::vector<int64_t> sh_times = {B, K},    st_times = contiguous_strides(sh_times);
    std::vector<int64_t> sh_out   = {B},       st_out   = contiguous_strides(sh_out);

    DLTensor t_loc   = make_cpu_tensor(loc.data(),   sh_loc,   st_loc,   bits);
    DLTensor t_coeff = make_cpu_tensor(coeff.data(), sh_coeff, st_coeff, bits);
    DLTensor t_times = make_cpu_tensor(times.data(), sh_times, st_times, bits);
    DLTensor t_time  = make_cpu_tensor(time_out.data(), sh_out, st_out,  bits);
    DLTensor t_dist  = make_cpu_tensor(dist_out.data(), sh_out, st_out,  bits);

    // Seed with the table result.
    ff::cpu::dt_spline_table(t_time, t_dist, t_loc, t_coeff, t_times,
                             SPLINE_LINEAR, BOUND_DCT2, 0);
    std::vector<scalar_t> table_dist = dist_out;

    if (use_brent)
        ff::cpu::dt_spline_brent(t_time, t_dist, t_loc, t_coeff,
                                 64, 1e-6, 0.1, SPLINE_LINEAR, BOUND_DCT2, 0);
    else
        ff::cpu::dt_spline_gaussnewton(t_time, t_dist, t_loc, t_coeff,
                                       64, 1e-6, SPLINE_LINEAR, BOUND_DCT2, 0);

    for (int64_t b = 0; b < B; ++b) {
        double d = (double)dist_out[b];
        check_true(std::isfinite(d),                   use_brent ? "brent.finite"  : "gn.finite");
        check_true(d >= -1e-6,                         use_brent ? "brent.nonneg"  : "gn.nonneg");
        check_true(d <= (double)table_dist[b] + 1e-3,  use_brent ? "brent.improve" : "gn.improve");
    }
}

// ---------------------------------------------------------------------------
// B5: independent reference for an arbitrary spline order. The table kernel
// evaluates the spline at each candidate time via a 1D pull, i.e.
//   s(t)[d] = sum_k sign(B,k,N) * coeff[idx(B,k,N), d] * weight(order, t-k)
// We reproduce that reconstruction here using the library spline weights /
// boundary handling (same pattern as test_splinc's eval_spline), generalised
// to D-dimensional control points, then brute-force the argmin over the same
// candidate dictionary. This exercises the DEFAULT cubic path (header default),
// which the linear-only tests above never touched.
// ---------------------------------------------------------------------------
void spline_eval_D(const std::vector<double>& coeff, int64_t N, int64_t D,
                   double t, int8_t order, btype B, std::vector<double>& out)
{
    type I = static_cast<type>(order);
    int64_t low, upp;
    ff::cpu::spline::bounds(I, t, low, upp);
    out.assign(D, 0.0);
    for (int64_t k = low; k <= upp; ++k) {
        int8_t sgn = ff::cpu::bound::sign(B, k, N);
        if (sgn == 0) continue;
        int64_t idx = ff::cpu::bound::index(B, k, N);
        double  w   = ff::cpu::spline::weight<double>(I, t - (double)k);
        for (int64_t d = 0; d < D; ++d)
            out[d] += (double)sgn * coeff[idx * D + d] * w;
    }
}

void spline_ref(const std::vector<double>& coeff, int64_t N, int64_t D,
                const std::vector<double>& loc, const std::vector<double>& times,
                int8_t order, btype B, double& best_time, double& best_dist)
{
    best_dist = INF;
    best_time = 0;
    std::vector<double> val;
    for (double t : times) {
        spline_eval_D(coeff, N, D, t, order, B, val);
        double d = 0;
        for (int64_t k = 0; k < D; ++k) { double diff = val[k] - loc[k]; d += diff * diff; }
        if (d < best_dist) { best_dist = d; best_time = t; }
    }
}

// Fully-batched table test for a given spline order (default = Cubic). Same
// tiled-batch layout as run_table_batched, but validated against spline_ref.
template <typename scalar_t>
void run_table_order(int64_t B, int64_t N, int64_t D, int64_t K, int8_t order,
                     uint8_t bits, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-2.0, 2.0);

    std::vector<double> control(N * D);
    for (auto& c : control) c = u(rng);

    std::vector<double> cand(K);
    for (int64_t j = 0; j < K; ++j)
        cand[j] = (double)(N - 1) * (double)j / (double)(K - 1);

    std::vector<scalar_t> loc(B * D), coeff(B * N * D), times(B * K);
    std::vector<double>   loc_d(B * D);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t d = 0; d < D; ++d) { double v = u(rng); loc_d[b*D+d] = v; loc[b*D+d] = (scalar_t)v; }
        for (int64_t i = 0; i < N * D; ++i) coeff[b*N*D + i] = (scalar_t)control[i];
        for (int64_t j = 0; j < K; ++j)     times[b*K + j]   = (scalar_t)cand[j];
    }

    std::vector<scalar_t> time_out(B, 0), dist_out(B, (scalar_t)INF);

    std::vector<int64_t> sh_loc   = {B, D},    st_loc   = contiguous_strides(sh_loc);
    std::vector<int64_t> sh_coeff = {B, N, D}, st_coeff = contiguous_strides(sh_coeff);
    std::vector<int64_t> sh_times = {B, K},    st_times = contiguous_strides(sh_times);
    std::vector<int64_t> sh_out   = {B},       st_out   = contiguous_strides(sh_out);

    DLTensor t_loc   = make_cpu_tensor(loc.data(),   sh_loc,   st_loc,   bits);
    DLTensor t_coeff = make_cpu_tensor(coeff.data(), sh_coeff, st_coeff, bits);
    DLTensor t_times = make_cpu_tensor(times.data(), sh_times, st_times, bits);
    DLTensor t_time  = make_cpu_tensor(time_out.data(), sh_out, st_out,  bits);
    DLTensor t_dist  = make_cpu_tensor(dist_out.data(), sh_out, st_out,  bits);

    ff::cpu::dt_spline_table(t_time, t_dist, t_loc, t_coeff, t_times,
                             order, BOUND_DCT2, 0);

    for (int64_t b = 0; b < B; ++b) {
        std::vector<double> loc_row(loc_d.begin() + b*D, loc_d.begin() + (b+1)*D);
        double ref_t, ref_d;
        spline_ref(control, N, D, loc_row, cand, order, btype::DCT2, ref_t, ref_d);
        check_close((double)dist_out[b], ref_d, "order.dist");
        check_close((double)time_out[b], ref_t, "order.time");
    }
}

// --- B4: negative / validation tests ---------------------------------------
// An unsupported dtype (float16) must throw, not silently no-op.
void test_bad_dtype_throws()
{
    const int64_t B = 2, N = 4, D = 2, K = 5;
    std::vector<uint16_t> loc(B*D,0), coeff(B*N*D,0), times(B*K,0), tout(B,0), dout(B,0);
    std::vector<int64_t> sh_loc={B,D},st_loc=contiguous_strides(sh_loc);
    std::vector<int64_t> sh_c={B,N,D},st_c=contiguous_strides(sh_c);
    std::vector<int64_t> sh_t={B,K},st_t=contiguous_strides(sh_t);
    std::vector<int64_t> sh_o={B},st_o=contiguous_strides(sh_o);
    DLTensor t_loc=make_cpu_tensor(loc.data(),sh_loc,st_loc,16);
    DLTensor t_coeff=make_cpu_tensor(coeff.data(),sh_c,st_c,16);
    DLTensor t_times=make_cpu_tensor(times.data(),sh_t,st_t,16);
    DLTensor t_time=make_cpu_tensor(tout.data(),sh_o,st_o,16);
    DLTensor t_dist=make_cpu_tensor(dout.data(),sh_o,st_o,16);
    bool threw = false;
    try { ff::cpu::dt_spline_table(t_time,t_dist,t_loc,t_coeff,t_times,SPLINE_CUBIC,BOUND_DCT2,0); }
    catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [distance_spline.bad_dtype_throws]\n"); }
}

// A shape mismatch (here: a wrong-rank output) must throw.
void test_shape_mismatch_throws()
{
    const int64_t B = 2, N = 4, D = 2, K = 5;
    std::vector<double> loc(B*D,0), coeff(B*N*D,0), times(B*K,0), tout(B,0), dout(B,0);
    std::vector<int64_t> sh_loc={B,D},st_loc=contiguous_strides(sh_loc);
    std::vector<int64_t> sh_c={B,N,D},st_c=contiguous_strides(sh_c);
    std::vector<int64_t> sh_t={B,K},st_t=contiguous_strides(sh_t);
    // nbatch = loc.ndim-1 = 1; a rank-2 `time` violates time.ndim == nbatch.
    std::vector<int64_t> sh_bad={B,1},st_bad=contiguous_strides(sh_bad);
    std::vector<int64_t> sh_o={B},st_o=contiguous_strides(sh_o);
    DLTensor t_loc=make_cpu_tensor(loc.data(),sh_loc,st_loc,64);
    DLTensor t_coeff=make_cpu_tensor(coeff.data(),sh_c,st_c,64);
    DLTensor t_times=make_cpu_tensor(times.data(),sh_t,st_t,64);
    DLTensor t_time=make_cpu_tensor(tout.data(),sh_bad,st_bad,64);  // wrong rank
    DLTensor t_dist=make_cpu_tensor(dout.data(),sh_o,st_o,64);
    bool threw = false;
    try { ff::cpu::dt_spline_table(t_time,t_dist,t_loc,t_coeff,t_times,SPLINE_CUBIC,BOUND_DCT2,0); }
    catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [distance_spline.shape_mismatch_throws]\n"); }
}

} // namespace

int main()
{
    std::printf("spline distance CPU tests\n");
    test_bad_dtype_throws();
    test_shape_mismatch_throws();
    for (unsigned seed = 1; seed <= 6; ++seed) {
        // Table, batched, float32 / float64, dims 1..3.
        run_table_batched<float >(5, 6, 1, 40, 32, seed);
        run_table_batched<float >(4, 7, 2, 50, 32, seed + 10);
        run_table_batched<float >(3, 8, 3, 60, 32, seed + 20);
        run_table_batched<double>(5, 6, 1, 40, 64, seed + 30);
        run_table_batched<double>(4, 7, 2, 50, 64, seed + 40);
        run_table_batched<double>(3, 8, 3, 60, 64, seed + 50);

        // Table, nbatch == 0 (literal documented shapes).
        run_table_scalar<float >(6, 2, 45, 32, seed + 60);
        run_table_scalar<double>(7, 3, 55, 64, seed + 70);

        // B5: DEFAULT cubic path (and a couple of higher orders) validated
        // against the independent spline reconstruction reference, dims 1..3.
        run_table_order<double>(4, 7, 1, 60, SPLINE_CUBIC, 64, seed + 130);
        run_table_order<double>(3, 8, 2, 70, SPLINE_CUBIC, 64, seed + 140);
        run_table_order<double>(3, 8, 3, 80, SPLINE_CUBIC, 64, seed + 150);
        run_table_order<double>(3, 8, 2, 70, /*Quadratic*/2, 64, seed + 160);
        run_table_order<double>(3, 9, 2, 90, /*FifthOrder*/5, 64, seed + 170);

        // Brent / Gauss-Newton smoke.
        run_refine_batched<float >(4, 7, 2, 40, 32, seed + 80, true);
        run_refine_batched<double>(3, 8, 3, 40, 64, seed + 90, false);
        run_refine_batched<double>(4, 6, 1, 40, 64, seed + 95, true);
    }
    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
