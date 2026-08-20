#pragma once
/* LICENSE:
 * Most of the functions are adapted from PyTorch/ATen's ParallelNative
 * https://github.com/pytorch/pytorch/blob/master/LICENSE
 */
#include <cstdint>
#include <fastfields/core/defines.h>
#include "parallel_impl.h"

/* The minimum number of elements a `parallel_for` must cover before it is
 * worth handing to the thread pool; below it, `parallel_for` calls `f` inline
 * on the calling thread and no thread is ever created.
 *
 * Overridable at build time (`-DFF_GRAIN_SIZE=<n>`) for one specific reason:
 * *every* workload in `tests/lib-cpu/` sits below the shipping value, so with
 * the default the whole suite -- 59,886 checks -- runs single-threaded and
 * `internal::invoke_parallel`, the thread pool, and every accumulate-into-a-
 * shared-output path are never executed concurrently (measured: zero `clone`
 * syscalls across all 13 test binaries). Building the suite with a small
 * FF_GRAIN_SIZE makes the same checks run multi-threaded without inventing new,
 * slower, large-volume test cases, which is what the `tsan` CI leg does.
 *
 * This is a threshold, never a correctness switch: results must be identical at
 * any value. Do not use it to tune performance -- 32768 is the shipping value.
 */
#ifndef FF_GRAIN_SIZE
#   define FF_GRAIN_SIZE 32768
#endif

FF_NAMESPACE_BEGIN(FF_NS)

constexpr int64_t GRAIN_SIZE = FF_GRAIN_SIZE;

template <class F>
inline void parallel_for(int64_t begin, int64_t end, int64_t grain_size, const F& f)
{
    if (begin >= end) return;

    const auto numiter = end - begin;
    const bool use_parallel =  (numiter > grain_size && numiter > 1 &&
                                // !internal::in_parallel_region() &&
                                get_parallel_threads() > 1);
    if (!use_parallel) {
        // internal::ThreadIdGuard tid_guard(0);
        f(begin, end);
        return;
    }

    internal::invoke_parallel(begin, end, grain_size, f);
}

FF_NAMESPACE_END(FF_NS)
