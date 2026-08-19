#ifndef FF_CPU_DISTANCE_L1
#define FF_CPU_DISTANCE_L1
#include <cstdint>
#include <type_traits>
#include <teeny/teeny.h>
#include "kernels/cuda_switch.h"
#include "kernels/distance.h"
#include "kernels/parallel.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_l1)

// 1-D L1 distance sweep along the LAST axis, batched over every leading axis.
//
// The TENSOR is the argument: `at` is a teeny `anyrank` over (*batch, n) that
// the caller (*-lib) built once from its own DLPack tensor. Every geometric
// quantity -- rank, per-line length, per-line stride, the batch offsets -- is
// derived from the carrier, so there is no (ndim, size[], stride[]) tuple to
// pass, to keep in sync, or to get wrong (TEENY-MIGRATION.md sec. 9, R2/R3).
//
// teeny's peel replaces the hand-written index2offset batch plumbing: the
// carrier hands out each rank-1 line as a view whose data pointer already has
// the (arbitrarily strided) batch offset folded in, so the sweep kernel is
// called unchanged on the raw pointer + stride. Add or drop batch axes by
// changing the data, not the loop.
//
// TEMPLATE SHAPE (the one the rest of the umbrella copies): one parameter per
// TENSOR -- here `A`, the carrier -- plus the ordinary deduced parameter for
// each scalar. The element type is deliberately NOT re-derived from the
// carrier: dtype dispatch belongs to *-lib (R1), so `scalar_t` still arrives
// from there, exactly as it did through the old signature. The static_assert
// ties the two together so a carrier/spacing mismatch reports itself in one
// line instead of surfacing as a pointer-conversion error inside kernel().
//
// `dt` writes through the carrier, so its element type is non-const here; a
// read-only operand would be a carrier of `const scalar_t` (R4).
template <class A, typename scalar_t>
inline void
dt(
    A          at ,   // (*batch, n) carrier -- transformed in place
    scalar_t   w  )   // pixel spacing
{
    using offset_t = decltype(at.size(0));   // the carrier's own offset type
    static_assert(
        std::is_same<scalar_t,
                     typename std::remove_pointer<decltype(A::data)>::type>::value,
        "distance_l1::dt: spacing type must match the carrier's element type");

    const offset_t nlines = at.template size_front<-1>();
    parallel_for(0, nlines, GRAIN_SIZE, [&](int64_t start, int64_t end) {
    for (offset_t i = start; i < end; ++i) {
        auto line = at.template peel_front_at<-1>(i);
        kernel<offset_t, scalar_t>(line.data(), line.extent(0), line.stride(0), w);
    }});
}

FF_NAMESPACE_END(distance_l1)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_CPU_DISTANCE_L1
