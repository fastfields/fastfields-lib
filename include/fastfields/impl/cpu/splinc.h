#ifndef FF_SPLINC_LOOP
#define FF_SPLINC_LOOP
#include <teeny/teeny.h>
#include "kernels/cuda_switch.h"
#include "kernels/splinc.h"
#include "kernels/parallel.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(splinc)

// 1-D spline prefilter (causal/anticausal IIR pole recursion) along the LAST
// axis, batched over every leading axis.
//
// The TENSOR is the argument: `at` is a teeny `anyrank` over (*batch, n) that
// the caller (*-lib) built once from its own DLPack tensor. Rank, per-line
// length, per-line stride and the batch offsets are all derived from the
// carrier, so there is no (nbatch, size[], stride[]) tuple to pass, to keep in
// sync, or to get wrong (TEENY-MIGRATION.md sec. 9, R2/R3).
//
// teeny's peel replaces the hand-written index2offset batch plumbing: the
// carrier hands out each rank-1 line as a view whose data pointer already has
// the (arbitrarily strided) batch offset folded in, so the sweep kernel is
// called unchanged on the raw pointer + size + stride.
//
// TEMPLATE SHAPE (Phase A's, fastfields-cpu-impl#60): one parameter per TENSOR
// -- here `A`, the carrier -- plus the ordinary deduced parameter per scalar
// operand. splinc has no scalar operand of the element type (the poles are
// `reduce_t`, deduced from the pointer as before), so unlike distance's `dt`
// there is no `scalar_t` parameter to tie back: the element type arrives inside
// the carrier, and `filter` deduces it from `line.data()` exactly as it did.
template <
    int npoles,
    bound::type B,
    class A,
    typename reduce_t
>
void loop(
    A                at    ,   // (*batch, n) carrier -- prefiltered in place
    const reduce_t * _poles)   // [npoles] filter poles
{
    using offset_t = decltype(at.size(0));   // the carrier's own offset type

    reduce_t poles[npoles];
    for (int k = 0; k < npoles; ++k) poles[k] = _poles[k];

    const offset_t nlines = at.template size_front<-1>();

    parallel_for(0, nlines, GRAIN_SIZE, [&](int64_t start, int64_t end) {
    for (offset_t i = start; i < end; ++i) {
        auto line = at.template peel_front_at<-1>(i);
        splinc::filter<B, npoles>(line.data(), line.extent(0), line.stride(0), poles);
    }});
}

FF_NAMESPACE_END(splinc)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_SPLINC_LOOP
