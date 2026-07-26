#ifndef FF_PUSHPULL_CPU
#define FF_PUSHPULL_CPU
// Teeny-based pushpull impl: the batch/spatial loops that drive the single-voxel
// kernels in kernels/pushpull/teeny.h (ff::cpu::pushpull::vox::*).
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
// Each tensor is wrapped from its OWN shape/stride arrays (the lib passes all
// three), so no trailing-dim reconstruction is needed. Order O and boundary B
// are compile-time (B == bound_t::Dynamic routes the runtime `bound` through the
// kernel's `rt` arg); reduce_t is the accumulation type (double from the lib).
#include "kernels/pushpull/teeny.h"
#include "kernels/parallel.h"
#include "kernels/utils.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(pushpull)

// product of the D spatial-grid extents (voxels per batch element)
template <int D, typename offset_t>
static inline offset_t _grid_spatial(offset_t nbatch, const offset_t * size_grid) {
    offset_t nsp = 1;
    for (int d = 0; d < D; ++d) nsp *= size_grid[nbatch + d];
    return nsp;
}

// ---- pull: out(*b,*grid,C) <- gather inp(*b,*spln,C) at grid(*b,*grid,D) ----
template <int D, int O, bound_t B, typename reduce_t, typename scalar_t, typename offset_t>
void pull(offset_t nbatch, int extrapolate, bound_t bound,
          scalar_t * out, const scalar_t * inp, const scalar_t * grid,
          const offset_t * size_out, const offset_t * size_inp, const offset_t * size_grid,
          const offset_t * stride_out, const offset_t * stride_inp, const offset_t * stride_grid)
{
    const int rank = static_cast<int>(nbatch) + D + 1;
    auto ao = tny::as_anyrank(out,  size_out,  stride_out,  rank, tny::copy_meta);
    auto ai = tny::as_anyrank(inp,  size_inp,  stride_inp,  rank, tny::copy_meta);
    auto ag = tny::as_anyrank(grid, size_grid, stride_grid, rank, tny::copy_meta);

    const offset_t nsp  = _grid_spatial<D>(nbatch, size_grid);
    const offset_t nvox = ao.template size_front<-1>();     // batch x spatial_grid voxels

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = start; i < end; ++i) {
            const offset_t b = (nsp > 0) ? i / nsp : 0;
            auto oc = ao.template peel_front_at<-1>(i);          // (C,)
            auto gc = ag.template peel_front_at<-1>(i);          // (D,)
            auto ic = ai.template peel_front_at<-(D + 1)>(b);    // (*spln, C)
            reduce_t loc[D];
            for (int d = 0; d < D; ++d) loc[d] = static_cast<reduce_t>(gc(d));
            vox::pull<D, O, B, reduce_t, offset_t>(oc, ic, loc, extrapolate, bound);
        }
    });
}

// ---- push: out(*b,*spln,C) <- scatter inp(*b,*grid,C) at grid(*b,*grid,D) ----
//      (out pre-zeroed by the caller; batch-parallel / spatial-sequential)
template <int D, int O, bound_t B, typename reduce_t, typename scalar_t, typename offset_t>
void push(offset_t nbatch, int extrapolate, bound_t bound,
          scalar_t * out, const scalar_t * inp, const scalar_t * grid,
          const offset_t * size_out, const offset_t * size_inp, const offset_t * size_grid,
          const offset_t * stride_out, const offset_t * stride_inp, const offset_t * stride_grid)
{
    const int rank = static_cast<int>(nbatch) + D + 1;
    auto ao = tny::as_anyrank(out,  size_out,  stride_out,  rank, tny::copy_meta);
    auto ai = tny::as_anyrank(inp,  size_inp,  stride_inp,  rank, tny::copy_meta);
    auto ag = tny::as_anyrank(grid, size_grid, stride_grid, rank, tny::copy_meta);

    const offset_t nsp  = _grid_spatial<D>(nbatch, size_grid);
    const offset_t nvox = ai.template size_front<-1>();     // batch x spatial_grid voxels (inp is grid-shaped)

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = start; i < end; ++i) {
            const offset_t b = (nsp > 0) ? i / nsp : 0;
            auto oc = ao.template peel_front_at<-(D + 1)>(b);   // (*spln, C), this batch (atomic scatter)
            auto ic = ai.template peel_front_at<-1>(i);         // (C,)
            auto gc = ag.template peel_front_at<-1>(i);         // (D,)
            reduce_t loc[D];
            for (int d = 0; d < D; ++d) loc[d] = static_cast<reduce_t>(gc(d));
            vox::push<D, O, B, reduce_t, offset_t>(oc, ic, loc, extrapolate, bound);
        }
    });
}

// ---- count: out(*b,*spln,1) <- scatter the interpolation weights (no inp) ----
template <int D, int O, bound_t B, typename reduce_t, typename scalar_t, typename offset_t>
void count(offset_t nbatch, int extrapolate, bound_t bound,
           scalar_t * out, const scalar_t * grid,
           const offset_t * size_out, const offset_t * size_grid,
           const offset_t * stride_out, const offset_t * stride_grid)
{
    const int rank = static_cast<int>(nbatch) + D + 1;
    auto ao = tny::as_anyrank(out,  size_out,  stride_out,  rank, tny::copy_meta);
    auto ag = tny::as_anyrank(grid, size_grid, stride_grid, rank, tny::copy_meta);

    const offset_t nsp  = _grid_spatial<D>(nbatch, size_grid);
    const offset_t nvox = ag.template size_front<-1>();     // batch x spatial_grid voxels

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = start; i < end; ++i) {
            const offset_t b = (nsp > 0) ? i / nsp : 0;
            auto oc = ao.template peel_front_at<-(D + 1)>(b);   // (*spln, 1), this batch (atomic scatter)
            auto gc = ag.template peel_front_at<-1>(i);
            reduce_t loc[D];
            for (int d = 0; d < D; ++d) loc[d] = static_cast<reduce_t>(gc(d));
            vox::count<D, O, B, reduce_t, offset_t>(oc, loc, extrapolate, bound);
        }
    });
}

// ---- grad: out(*b,*grid,C,D) <- spatial gradient of the pull ----------------
template <int D, int O, bound_t B, bool ABS, typename reduce_t, typename scalar_t, typename offset_t>
void grad(offset_t nbatch, int extrapolate, bound_t bound,
          scalar_t * out, const scalar_t * inp, const scalar_t * grid,
          const offset_t * size_out, const offset_t * size_inp, const offset_t * size_grid,
          const offset_t * stride_out, const offset_t * stride_inp, const offset_t * stride_grid)
{
    const int orank = static_cast<int>(nbatch) + D + 2;   // out has the extra D axis
    const int rank  = static_cast<int>(nbatch) + D + 1;
    auto ao = tny::as_anyrank(out,  size_out,  stride_out,  orank, tny::copy_meta);
    auto ai = tny::as_anyrank(inp,  size_inp,  stride_inp,  rank,  tny::copy_meta);
    auto ag = tny::as_anyrank(grid, size_grid, stride_grid, rank,  tny::copy_meta);

    const offset_t nsp  = _grid_spatial<D>(nbatch, size_grid);
    const offset_t nvox = ag.template size_front<-1>();     // batch x spatial_grid voxels

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = start; i < end; ++i) {
            const offset_t b = (nsp > 0) ? i / nsp : 0;
            auto oc = ao.template peel_front_at<-2>(i);          // (C, D)
            auto gc = ag.template peel_front_at<-1>(i);          // (D,)
            auto ic = ai.template peel_front_at<-(D + 1)>(b);    // (*spln, C)
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
