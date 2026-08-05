#ifndef FF_PUSHPULL_CPU
#define FF_PUSHPULL_CPU
// Teeny-based pushpull impl: the batch/spatial loops that drive the single-voxel
// kernels in kernels/pushpull/teeny.h (ff::cpu::pushpull::vox::*).
//
// THE TENSORS ARE THE ARGUMENTS. Each entry point takes one teeny `anyrank`
// carrier per tensor, built once by the caller (*-lib) straight from its own
// DLPack descriptor. Every geometric quantity -- rank, batch count, spatial
// extents, per-voxel offsets -- is derived from the carriers, so there is no
// (nbatch, size[], stride[]) tuple to pass, to keep in sync, or to get wrong
// (TEENY-MIGRATION.md sec. 9, R2/R3). Read-only operands are carriers of
// `const scalar_t` (R4); push/count's scatter target stays writable.
//
// What is NOT derivable from a tensor stays a template parameter supplied by
// the *-lib dispatch (R1): the spatial rank D, the spline order O, the boundary
// B, grad's ABS, and the accumulation type reduce_t. The offset type is the
// carrier's own (`decltype(ao.size(0))`), and `extrapolate` / `bound` remain
// runtime kernel parameters -- they are behaviour, not geometry.
//
// The (*batch, *spatial, C) decomposition is done with teeny's anyrank peel
// instead of the hand-written index2offset plumbing. ALL four ops parallelise
// flat over the grid voxels (batch x spatial_grid): out/grid peel the last 1
// (grad: last 2) dims -> the voxel cell; inp peels only the batch (last D+1
// kept) -> that batch's spatial volume. The channel loop lives in the kernel.
//   * READS (pull, grad) write per-voxel-disjoint outputs -> no contention.
//   * SCATTERS (push, count) accumulate into a shared output via anyAtomicAdd,
//     a lock-free CAS on the host (atomic.h) that is atomic for float/double too,
//     so a flat voxel-parallel scatter is race-free -- no batch-serial fallback,
//     full parallelism even at nbatch<=1.
//
// PRECONDITION (unchanged, and deliberately unchecked here). The voxel-shaped
// pair -- out/grid for pull and grad, inp/grid for push -- is addressed by ONE
// flat (*batch, *spatial) voxel number, so those two tensors must agree in rank
// and in their batch and spatial extents. *-lib enforces exactly that at the
// DLPack boundary (CHECK_SAME / CHECK_SAME_BATCH / CHECK_SAME_SPATIAL), where a
// violation can still be reported as a named std::invalid_argument; this layer
// trusts it, as it did when the same equality was implied by the single `rank`
// argument it used to be handed.
#include "kernels/pushpull/teeny.h"
#include "kernels/parallel.h"
#include "kernels/utils.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(pushpull)

// Product of the D spatial-grid extents (voxels per batch element), read off the
// grid carrier itself -- this used to scan the caller's size_grid[] array (R2).
template <int D, class AG>
static inline auto _grid_spatial(int nbatch, const AG & ag) {
    using offset_t = decltype(ag.size(0));
    offset_t nsp = 1;
    for (int d = 0; d < D; ++d) nsp *= ag.size(nbatch + d);
    return nsp;
}

// ---- pull: out(*b,*grid,C) <- gather inp(*b,*spln,C) at grid(*b,*grid,D) ----
template <int D, int O, bound_t B, typename reduce_t, class AO, class AI, class AG>
void pull(AO ao, const AI ai, const AG ag, int extrapolate, bound_t bound)
{
    using offset_t = decltype(ao.size(0));          // the carrier's own offset type
    const int nbatch = ao.ndim - D - 1;             // out is (*b, *grid, C)

    const offset_t nsp  = _grid_spatial<D>(nbatch, ag);
    const offset_t nvox = ao.template size_front<-1>();     // batch x spatial_grid voxels

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        // the sampled inp volume changes only every nsp voxels -> peel once per
        // batch cell, not per voxel (peel_front_at is a mixed-radix batch decode).
        offset_t cur_b = (nsp > 0) ? static_cast<offset_t>(start) / nsp : 0;
        auto ic = ai.template peel_front_at<-(D + 1)>(cur_b);    // (*spln, C), this batch
        for (offset_t i = start; i < end; ++i) {
            const offset_t b = (nsp > 0) ? i / nsp : 0;
            if (b != cur_b) { ic = ai.template peel_front_at<-(D + 1)>(b); cur_b = b; }
            auto oc = ao.template peel_front_at<-1>(i);          // (C,)
            auto gc = ag.template peel_front_at<-1>(i);          // (D,)
            reduce_t loc[D];
            for (int d = 0; d < D; ++d) loc[d] = static_cast<reduce_t>(gc(d));
            vox::pull<D, O, B, reduce_t, offset_t>(oc, ic, loc, extrapolate, bound);
        }
    });
}

// ---- push: out(*b,*spln,C) <- scatter inp(*b,*grid,C) at grid(*b,*grid,D) ----
//      (out pre-zeroed by the caller; batch-parallel / spatial-sequential)
template <int D, int O, bound_t B, typename reduce_t, class AO, class AI, class AG>
void push(AO ao, const AI ai, const AG ag, int extrapolate, bound_t bound)
{
    using offset_t = decltype(ao.size(0));
    const int nbatch = ao.ndim - D - 1;             // out is (*b, *spln, C)

    const offset_t nsp  = _grid_spatial<D>(nbatch, ag);
    const offset_t nvox = ai.template size_front<-1>();     // batch x spatial_grid voxels (inp is grid-shaped)

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        // the splatted out volume changes only every nsp voxels -> peel once per
        // batch cell (the atomic scatter target; view is per-thread within a grain).
        offset_t cur_b = (nsp > 0) ? static_cast<offset_t>(start) / nsp : 0;
        auto oc = ao.template peel_front_at<-(D + 1)>(cur_b);   // (*spln, C), this batch (atomic scatter)
        for (offset_t i = start; i < end; ++i) {
            const offset_t b = (nsp > 0) ? i / nsp : 0;
            if (b != cur_b) { oc = ao.template peel_front_at<-(D + 1)>(b); cur_b = b; }
            auto ic = ai.template peel_front_at<-1>(i);         // (C,)
            auto gc = ag.template peel_front_at<-1>(i);         // (D,)
            reduce_t loc[D];
            for (int d = 0; d < D; ++d) loc[d] = static_cast<reduce_t>(gc(d));
            vox::push<D, O, B, reduce_t, offset_t>(oc, ic, loc, extrapolate, bound);
        }
    });
}

// ---- count: out(*b,*spln,1) <- scatter the interpolation weights (no inp) ----
template <int D, int O, bound_t B, typename reduce_t, class AO, class AG>
void count(AO ao, const AG ag, int extrapolate, bound_t bound)
{
    using offset_t = decltype(ao.size(0));
    const int nbatch = ao.ndim - D - 1;             // out is (*b, *spln, 1)

    const offset_t nsp  = _grid_spatial<D>(nbatch, ag);
    const offset_t nvox = ag.template size_front<-1>();     // batch x spatial_grid voxels

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        offset_t cur_b = (nsp > 0) ? static_cast<offset_t>(start) / nsp : 0;
        auto oc = ao.template peel_front_at<-(D + 1)>(cur_b);   // (*spln, 1), this batch (atomic scatter)
        for (offset_t i = start; i < end; ++i) {
            const offset_t b = (nsp > 0) ? i / nsp : 0;
            if (b != cur_b) { oc = ao.template peel_front_at<-(D + 1)>(b); cur_b = b; }
            auto gc = ag.template peel_front_at<-1>(i);
            reduce_t loc[D];
            for (int d = 0; d < D; ++d) loc[d] = static_cast<reduce_t>(gc(d));
            vox::count<D, O, B, reduce_t, offset_t>(oc, loc, extrapolate, bound);
        }
    });
}

// ---- grad: out(*b,*grid,C,D) <- spatial gradient of the pull ----------------
template <int D, int O, bound_t B, bool ABS, typename reduce_t, class AO, class AI, class AG>
void grad(AO ao, const AI ai, const AG ag, int extrapolate, bound_t bound)
{
    using offset_t = decltype(ao.size(0));
    // The one entry whose two carriers DISAGREE in rank: grad's output carries
    // the extra trailing (C, D) pair, so it arrives at rank nbatch + D + 2 while
    // inp and grid are nbatch + D + 1. Hence -2 here, -1 everywhere else.
    const int nbatch = ao.ndim - D - 2;             // out is (*b, *grid, C, D)

    const offset_t nsp  = _grid_spatial<D>(nbatch, ag);
    const offset_t nvox = ag.template size_front<-1>();     // batch x spatial_grid voxels

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        offset_t cur_b = (nsp > 0) ? static_cast<offset_t>(start) / nsp : 0;
        auto ic = ai.template peel_front_at<-(D + 1)>(cur_b);    // (*spln, C), this batch
        for (offset_t i = start; i < end; ++i) {
            const offset_t b = (nsp > 0) ? i / nsp : 0;
            if (b != cur_b) { ic = ai.template peel_front_at<-(D + 1)>(b); cur_b = b; }
            auto oc = ao.template peel_front_at<-2>(i);          // (C, D)
            auto gc = ag.template peel_front_at<-1>(i);          // (D,)
            reduce_t loc[D];
            for (int d = 0; d < D; ++d) loc[d] = static_cast<reduce_t>(gc(d));
            vox::grad<D, O, B, ABS, reduce_t, offset_t>(oc, ic, loc, extrapolate, bound);
        }
    });
}

FF_NAMESPACE_END(pushpull)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_PUSHPULL_CPU
