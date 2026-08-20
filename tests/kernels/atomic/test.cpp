/* What `ff::anyAtomicAdd` actually is on the CPU, pinned.
 *
 * `impl/kernels/atomic.h` offers two CPU implementations and picks between
 * them with `has_atomic_add<T>`:
 *
 *   AtomicAdd<false>  -- a plain, non-atomic `*address += val`
 *   AtomicAdd<true>   -- a std::atomic compare-exchange loop
 *
 * and `impl/cpu/pushpull.h` / `impl/cpu/restrict.h` branch on the *same*
 * predicate to choose a parallelisation strategy: `if (has_atomic_add<T>)`
 * parallelise over every element and let the atomics resolve the collisions,
 * else parallelise over the batch dimension only, so that concurrent threads
 * write to disjoint output slices and no atomicity is required.
 *
 * The predicate answers **false for every type, in every language standard**
 * -- `has_fetch_add` probes `&C::fetch_add`, which is an overload set on
 * `std::atomic` (with and without a memory_order argument), so taking its
 * address is ambiguous and SFINAE rejects it before the C++ version is ever
 * relevant. The CAS specialisation is therefore unreachable, and the scatter
 * paths always take the disjoint-slice branch. That is what makes them
 * correct, so it is worth a test rather than a comment: someone "fixing" the
 * detector would silently switch every scatter op onto a strategy whose
 * atomics are a plain `+=`.
 *
 * Not part of `make test` (`make test-atomics` runs it). It links nothing and
 * takes milliseconds; it is built by the tsan CI leg, where the threaded case
 * below is the interesting one.
 */
#include "fastfields/core/atomic.h"
#include "fastfields/core/parallel.h"
#include <cstdio>
#include <cstdint>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const char * what)
{
    if (!ok) { ++g_failures; std::printf("  FAIL [%s]\n", what); }
    else                     std::printf("  ok:   %s\n", what);
}

int main()
{
    /* 1. The predicate. If any of these ever flips, read the header comment
     *    above before changing the expectation -- `anyAtomicAdd` does not even
     *    compile for a type the predicate accepts, because
     *    `AtomicAdd<true>::atomicAdd` returns void while `anyAtomicAdd`
     *    returns T. */
    check(!ff::has_atomic_add<float>::value,   "has_atomic_add<float>   is false");
    check(!ff::has_atomic_add<double>::value,  "has_atomic_add<double>  is false");
    check(!ff::has_atomic_add<int32_t>::value, "has_atomic_add<int32_t> is false");
    check(!ff::has_atomic_add<int64_t>::value, "has_atomic_add<int64_t> is false");

    /* 2. Single-threaded semantics of the implementation that is actually
     *    selected: accumulate, and return the *new* value. (Note the CUDA
     *    `gpuAtomicAdd` this shadows returns the *old* value; nothing in the
     *    tree uses either return value, which is why the divergence is
     *    harmless today.) */
    {
        double x = 1.0;
        double r = ff::anyAtomicAdd(&x, 2.0);
        check(x == 3.0, "anyAtomicAdd accumulates into the target");
        check(r == 3.0, "anyAtomicAdd returns the new value (CUDA returns the old one)");
        ff::anyAtomicAddNoReturn(&x, -3.0);
        check(x == 0.0, "anyAtomicAddNoReturn accumulates into the target");
    }

    /* 3. The strategy the scatter ops rely on: many threads accumulating into
     *    *disjoint* slots is race-free even though the add is not atomic.
     *    Under TSan this is the case that must stay clean; it fails there the
     *    moment two workers are given overlapping output. */
    {
        const long n = 4096;
        std::vector<double> out((size_t)n, 0.0);
        double * p = &out[0];
        ff::parallel_for(0, n, /*grain_size=*/1, [p](long begin, long end) {
            for (long i = begin; i < end; ++i)
                ff::anyAtomicAdd(p + i, 1.0);
        });
        bool all_one = true;
        for (long i = 0; i < n; ++i) if (out[(size_t)i] != 1.0) all_one = false;
        check(all_one, "disjoint concurrent accumulation is exact");
        std::printf("  (%zu worker threads available)\n", ff::get_parallel_threads());
    }

    std::printf("%s\n", g_failures ? "FAILURES" : "All checks passed.");
    return g_failures ? 1 : 0;
}
