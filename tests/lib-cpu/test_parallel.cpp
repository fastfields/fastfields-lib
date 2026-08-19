// CPU tests for the *parallel* execution path and the accumulation
// (`anyAtomicAdd`) contract it relies on.
//
// Why this file exists
// --------------------
// Every other tests/lib-cpu/test_*.cpp runs on workloads of a few hundred to a
// few thousand elements. `parallel_for` (impl/kernels/parallel.h) only hands
// work to the thread pool when
//
//     numiter > grain_size   AND   get_parallel_threads() > 1
//
// with a default grain of GRAIN_SIZE == 32768. Measured on the suite as it
// stands: 38,657 parallel_for calls, none of which dispatched to the pool, and
// the largest range any of them ever saw was 512 elements -- 64x below the
// grain. On top of that `default_num_threads_from_hardware()` halves
// `std::thread::hardware_concurrency()` on x86-64, so a 2-vCPU runner reports
// one thread and disables the pool outright regardless of workload size.
//
// The consequence is that the whole multi-threaded half of the library -- the
// thread pool, `invoke_parallel`, and the way `push`/`restriction` split their
// loops to stay race-free -- is compiled by the gate but never executed by it.
//
// This file forces both conditions and then checks the things that can only go
// wrong once more than one thread is running:
//
//   A. `parallel_for` really does dispatch to the pool, and the chunks it hands
//      out tile [begin, end) exactly once with no gap and no overlap.
//   B. `push` and `restriction` -- the two ops that accumulate into shared
//      output voxels -- produce the same answer multi-threaded as
//      single-threaded, on a workload large enough to take the parallel branch.
//   C. The accumulation contract itself. `bound::add` calls `ff::anyAtomicAdd`,
//      which is only genuinely atomic when `has_atomic_add<scalar_t>::value` is
//      true; otherwise it is a plain read-modify-write. That same trait is what
//      impl/cpu/{pushpull,restrict}.h switch on to decide whether to
//      parallelise over *all* elements (needs real atomics) or only over the
//      batch dimension (disjoint outputs, no atomics needed). The two must
//      agree, so the trait is probed under contention here rather than assumed.
//
// Build: picked up automatically by `make test-lib-cpu` (the Makefile globs
// tests/lib-cpu/test_*.cpp).

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <thread>
#include <set>
#include <mutex>

#include "fastfields/core/dlpack.h"
#include "fastfields/impl/kernels/parallel.h"
#include "fastfields/impl/kernels/atomic.h"
#include "fastfields/api/cpu/pushpull.h"
#include "fastfields/api/cpu/restrict.h"

namespace {

int g_failures = 0, g_checks = 0;

void check(bool ok, const char * what)
{
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL [%s]\n", what);
    }
}

void check_close(double a, double b, const char * what, double tol = 1e-12)
{
    ++g_checks;
    double diff = std::fabs(a - b);
    double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    if (diff / scale > tol) {
        ++g_failures;
        std::printf("  MISMATCH [%s]: %.17g vs %.17g\n", what, a, b);
    }
}

template <typename T>
DLTensor make_cpu_tensor(T * data, std::vector<int64_t> & shape,
                         std::vector<int64_t> & strides, uint8_t bits)
{
    DLTensor t;
    t.data = static_cast<void *>(data);
    t.device.device_type = kDLCPU;
    t.device.device_id = 0;
    t.ndim = static_cast<int32_t>(shape.size());
    t.dtype.code = static_cast<uint8_t>(kDLFloat);
    t.dtype.bits = bits;
    t.dtype.lanes = 1;
    t.shape = shape.data();
    t.strides = strides.data();
    t.byte_offset = 0;
    return t;
}

std::vector<int64_t> cstrides(const std::vector<int64_t> & shape)
{
    std::vector<int64_t> s(shape.size());
    int64_t acc = 1;
    for (int64_t d = (int64_t)shape.size() - 1; d >= 0; --d) {
        s[d] = acc;
        acc *= shape[d];
    }
    return s;
}

// ===================================================================
// A. parallel_for actually parallelises, and tiles its range exactly
// ===================================================================
//
// grain_size is a *parameter* of parallel_for, so this needs no build-time
// override of GRAIN_SIZE: a small grain plus >1 thread is enough to reach
// internal::invoke_parallel and therefore the thread pool.
void test_parallel_for_partition()
{
    const long N = 100000;
    const long grain = 1000;

    std::vector<int> visits(N, 0);
    std::mutex mtx;
    std::set<std::thread::id> ids;

    ff::parallel_for(0, N, grain, [&](long start, long end) {
        for (long i = start; i < end; ++i) visits[i] += 1;
        std::lock_guard<std::mutex> lk(mtx);
        ids.insert(std::this_thread::get_id());
    });

    long bad = 0;
    for (long i = 0; i < N; ++i)
        if (visits[i] != 1) ++bad;
    check(bad == 0, "parallel_for.each_index_visited_exactly_once");

    // The point of the file: prove the pool was actually entered. The
    // invariant that holds is "the work did not run on the calling thread" --
    // invoke_parallel hands every chunk to the pool and blocks on the futures,
    // so the caller never executes one itself.
    //
    // Note what is deliberately NOT asserted: that more than one worker ran.
    // ThreadPool::pushWork enqueues every task of a batch onto mWorkers.front()
    // when called from a non-worker thread, and the other workers only acquire
    // work by stealing -- so on a loaded machine worker 0 can drain the whole
    // queue first and the observed worker count legitimately drops to 1.
    // Asserting >1 here is flaky; it was, before this comment existed.
    check(ids.find(std::this_thread::get_id()) == ids.end(),
          "parallel_for.work_ran_off_the_calling_thread");
    check(!ids.empty(), "parallel_for.work_ran_somewhere");
    std::printf("  parallel_for: threads=%zu distinct_workers_used=%zu\n",
                ff::get_parallel_threads(), ids.size());

    // Degenerate ranges must not dispatch or drop work.
    long calls = 0;
    ff::parallel_for(5, 5, 1, [&](long, long) { ++calls; });
    check(calls == 0, "parallel_for.empty_range_no_call");

    std::vector<int> one(1, 0);
    ff::parallel_for(0, 1, 1, [&](long s, long e) {
        for (long i = s; i < e; ++i) one[i] += 1;
    });
    check(one[0] == 1, "parallel_for.single_element");
}

// ===================================================================
// B. push / restriction: parallel result == serial result
// ===================================================================
//
// `push` accumulates into overlapping output voxels, so it is the op most
// exposed to a bad parallel split. impl/cpu/pushpull.h picks its split from
// has_atomic_add<scalar_t>::value:
//   true  -> parallel over every grid point,      relies on real atomics;
//   false -> parallel over the batch dim only,    outputs are disjoint.
// Either way the answer must not depend on the thread count. The workload
// below is sized so that *both* branches dispatch to the pool:
//   all-elements branch: numel   = B*H*W = 131072 > GRAIN_SIZE (32768)
//   batch branch:        numel_b = B     = 32     > GRAIN_SIZE/(H*W) = 8
template <typename T>
void push_once(std::vector<T> & out, const std::vector<T> & inp,
               const std::vector<T> & grid, int64_t B, int64_t H, int64_t W,
               int64_t C, uint8_t bits, int8_t order, int8_t bound)
{
    std::fill(out.begin(), out.end(), (T)0); // push accumulates into `out`
    std::vector<T> inp_copy(inp), grid_copy(grid);

    std::vector<int64_t> os = {B, H, W, C}, oss = cstrides(os);
    std::vector<int64_t> is = {B, H, W, C}, iss = cstrides(is);
    std::vector<int64_t> gs = {B, H, W, 2}, gss = cstrides(gs);

    DLTensor ot = make_cpu_tensor(out.data(), os, oss, bits);
    DLTensor it = make_cpu_tensor(inp_copy.data(), is, iss, bits);
    DLTensor gt = make_cpu_tensor(grid_copy.data(), gs, gss, bits);

    ff::cpu::push(ot, it, gt, order, bound, 1, 0);
}

template <typename T>
void test_push_thread_invariant(uint8_t bits, int8_t order, int8_t bound,
                                const char * what)
{
    const int64_t B = 32, H = 64, W = 64, C = 1;
    std::mt19937 rng(20260819u);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> ux(0.0, (double)(W - 1));
    std::uniform_real_distribution<double> uy(0.0, (double)(H - 1));

    std::vector<T> inp((size_t)(B * H * W * C));
    for (auto & v : inp) v = (T)u(rng);
    std::vector<T> grid((size_t)(B * H * W * 2));
    for (size_t k = 0; k < (size_t)(B * H * W); ++k) {
        grid[2 * k] = (T)ux(rng);
        grid[2 * k + 1] = (T)uy(rng);
    }

    std::vector<T> par((size_t)(B * H * W * C)), ser((size_t)(B * H * W * C));

    const size_t nthreads = ff::get_parallel_threads();
    push_once(par, inp, grid, B, H, W, C, bits, order, bound); // multi-threaded
    ff::set_num_threads(1);
    push_once(ser, inp, grid, B, H, W, C, bits, order, bound); // serial
    ff::set_num_threads(nthreads);

    // Both runs sum the same contributions in the same per-output order (the
    // split is over disjoint outputs, or guarded by atomics), so this is an
    // exact-equality check, not an approximate one.
    size_t bad = 0;
    for (size_t i = 0; i < par.size(); ++i)
        if (!(par[i] == ser[i])) ++bad;
    ++g_checks;
    if (bad) {
        ++g_failures;
        std::printf("  MISMATCH [%s]: %zu/%zu voxels differ between the "
                    "parallel and serial runs\n",
                    what, bad, par.size());
    }

    // Mass conservation: with extrapolate=1 every input value is fully
    // splatted, so the total pushed mass equals the total input mass regardless
    // of the number of threads. A lost atomic update shows up here as a
    // deficit.
    double sum_in = 0.0, sum_par = 0.0;
    for (size_t i = 0; i < inp.size(); ++i) sum_in += (double)inp[i];
    for (size_t i = 0; i < par.size(); ++i) sum_par += (double)par[i];
    check_close(sum_par, sum_in, "push.mass_conserved_parallel",
                bits == 32 ? 1e-4 : 1e-9);
}

// restriction() accumulates into `out` too, and its parallel split lives in
// impl/cpu/restrict.h behind the same trait.
template <typename T>
void restrict_once(std::vector<T> & out, const std::vector<T> & inp, int64_t B,
                   int64_t Hi, int64_t Wi, int64_t Ho, int64_t Wo, uint8_t bits,
                   int8_t order, int8_t bound)
{
    std::fill(out.begin(), out.end(), (T)0); // restriction accumulates
    std::vector<T> inp_copy(inp);
    std::vector<int64_t> is = {B, Hi, Wi}, iss = cstrides(is);
    std::vector<int64_t> os = {B, Ho, Wo}, oss = cstrides(os);
    DLTensor it = make_cpu_tensor(inp_copy.data(), is, iss, bits);
    DLTensor ot = make_cpu_tensor(out.data(), os, oss, bits);
    const double scale[2] = {(double)Hi / (double)Ho, (double)Wi / (double)Wo};
    ff::cpu::restriction(ot, it, order, bound, /*shift*/ 0.0, scale, /*ndim*/ 2,
                         0);
}

template <typename T>
void test_restriction_thread_invariant(uint8_t bits, int8_t order, int8_t bound,
                                       const char * what)
{
    // out spatial 64x64 -> padded 66x66 = 4356 elements per batch element, so
    // the batch-split grain is GRAIN_SIZE/4356 = 7 and B = 16 clears it.
    const int64_t B = 16, Hi = 128, Wi = 128, Ho = 64, Wo = 64;
    std::mt19937 rng(4242u);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<T> inp((size_t)(B * Hi * Wi));
    for (auto & v : inp) v = (T)u(rng);

    std::vector<T> par((size_t)(B * Ho * Wo)), ser((size_t)(B * Ho * Wo));
    const size_t nthreads = ff::get_parallel_threads();
    restrict_once(par, inp, B, Hi, Wi, Ho, Wo, bits, order, bound);
    ff::set_num_threads(1);
    restrict_once(ser, inp, B, Hi, Wi, Ho, Wo, bits, order, bound);
    ff::set_num_threads(nthreads);

    size_t bad = 0;
    for (size_t i = 0; i < par.size(); ++i)
        if (!(par[i] == ser[i])) ++bad;
    ++g_checks;
    if (bad) {
        ++g_failures;
        std::printf("  MISMATCH [%s]: %zu/%zu coefficients differ between the "
                    "parallel and serial runs\n",
                    what, bad, par.size());
    }
}

// ===================================================================
// C. The anyAtomicAdd contract
// ===================================================================
//
// `has_atomic_add<T>::value` decides two things at once: whether
// `ff::anyAtomicAdd` is a real atomic, and (in impl/cpu/{pushpull,restrict}.h)
// whether the loops may be split in a way that races. This checks they agree,
// by hammering one address from several threads at once.
//
// If the trait says "atomic", a lost update is a hard failure. If it says
// "not atomic", the observed total is reported and nothing is asserted -- the
// callers are then required never to race on it, which is what part B tests.
// Either way the assertion below becomes live the day the trait changes.
template <typename T> void test_atomic_contract(const char * tname)
{
    const int nthreads = 8;
    const int per_thread = 20000;

    T value = (T)0;
    std::vector<std::thread> ths;
    for (int t = 0; t < nthreads; ++t)
        ths.push_back(std::thread([&value, per_thread] {
            for (int i = 0; i < per_thread; ++i) ff::anyAtomicAdd(&value, (T)1);
        }));
    for (size_t i = 0; i < ths.size(); ++i) ths[i].join();

    const double expect = (double)nthreads * (double)per_thread;
    if (ff::has_atomic_add<T>::value) {
        check_close((double)value, expect, "anyAtomicAdd.no_lost_updates");
    }
    else {
        ++g_checks; // recorded, not asserted: the fallback is a plain RMW
        std::printf("  anyAtomicAdd<%s>: has_atomic_add=false (non-atomic "
                    "fallback); contended total %.0f of %.0f -- callers must "
                    "not race on it\n",
                    tname, (double)value, expect);
    }
}

} // namespace

int main()
{
    std::printf("parallel / atomics CPU tests\n");

    // Force the pool on. default_num_threads_from_hardware() halves
    // hardware_concurrency() on x86-64, so a 2-vCPU CI runner would otherwise
    // report one thread and every parallel_for below would run serially --
    // exactly the blind spot this file exists to remove.
    const size_t nthreads = ff::set_num_threads(4);
    std::printf("  backend=%s threads=%zu grain=%lld hw=%u\n",
                ff::get_parallel_backend().c_str(), nthreads,
                (long long)ff::GRAIN_SIZE, std::thread::hardware_concurrency());
    check(nthreads > 1, "set_num_threads.pool_enabled");

    test_parallel_for_partition();

    // 1 = Linear, 2 = Quadratic; 3 = DCT2 (the library default bound).
    test_push_thread_invariant<double>(64, 1, 3, "push.f64.linear.dct2");
    test_push_thread_invariant<double>(64, 2, 3, "push.f64.quadratic.dct2");
    test_push_thread_invariant<float>(32, 1, 3, "push.f32.linear.dct2");

    test_restriction_thread_invariant<double>(64, 1, 3,
                                              "restriction.f64.linear.dct2");
    test_restriction_thread_invariant<float>(32, 1, 3,
                                             "restriction.f32.linear.dct2");

    test_atomic_contract<float>("float");
    test_atomic_contract<double>("double");

    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
