#ifndef FF_PUSHPULL_CUDA
#define FF_PUSHPULL_CUDA
// Teeny-based CUDA pushpull impl -- the device mirror of the CPU launcher
// (fastfields-cpu-impl/pushpull.h). Same math, same representation, same
// (*batch, *spatial, C) decomposition via teeny's anyrank peel; only the
// scheduling differs (a `__global__` grid-stride loop replaces parallel_for).
//
//   ALL four ops parallelise FLAT over the grid voxels (batch x spatial_grid):
//   out/grid peel the last 1 (grad: last 2) dims -> the voxel cell; inp peels
//   only the batch (last D+1 kept) -> that batch's spatial volume. The channel
//   loop lives in the shared single-voxel kernel (kernels/pushpull/teeny.h,
//   ff::cuda::pushpull::vox::*), byte-for-byte the same call the CPU body uses.
//     * READS (pull, grad) write per-voxel-disjoint outputs -> no contention.
//     * SCATTERS (push, count) accumulate into a shared output via the DEVICE
//       atomics baked into vox::push / vox::count (anyAtomicAddNoReturn, atomic.h),
//       so a flat voxel-parallel scatter is race-free on device -- no batch-serial
//       fallback needed. Each thread peels its own batch cell (b = i / nsp) and
//       scatters into it; overlapping taps from different threads add atomically.
//
// Device port vs. the CPU version:
//   * the parallel_for becomes a `__global__` grid-stride loop over the voxels;
//     each tensor is wrapped as a DEVICE-PASSABLE teeny anyrank carrier
//     (`as_anyrank<TNY_MAX_RANK, storage::gpu_view>(..., copy_meta)` -- the
//     shape/stride travel INLINE with the carrier, so it is trivially copyable and
//     passes into the kernel BY VALUE; no separate device copy of shape/stride);
//   * `nbatch` and `extrapolate` stay RUNTIME (folded into the peel / the FOV
//     test), so a single device instantiation per (D, O, B, dtype, offset) covers
//     every batch rank -- no per-nbatch specialisation explosion;
//   * the CUDA `stream` is forwarded to the launch and synchronised after.
//
// Each tensor is wrapped from its OWN shape/stride arrays (the lib passes all
// three, on the host), so no trailing-dim reconstruction is needed. Order O and
// boundary B are compile-time (B == bound_t::Dynamic routes the runtime `bound`
// through the kernel's `rt` arg); reduce_t is the accumulation type (double).
#include "fastfields/impl/kernels/pushpull/teeny.h"   // vox::pull/push/count/grad (+ <teeny/teeny.h>)
#include "utils.h"                    // GET_BLOCKS / CUDA_NUM_THREADS
#include <cstdint>

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(pushpull)

// Device-passable anyrank carrier over (*batch, *spatial, trailing). Shape/stride
// are COPIED inline (copy_meta) so the carrier passes into the kernel by value;
// the DATA pointer lives in device memory (storage::gpu_view). `rank` is the FULL
// tensor rank (grad's output carries one extra axis -> pass rank+1 for it).
template <typename T, typename offset_t>
static inline auto _any(T* p, const offset_t* size, const offset_t* stride, int rank)
{
    return tny::as_anyrank<TNY_MAX_RANK, tny::storage::gpu_view>(
        p, size, stride, rank, tny::copy_meta);
}

// product of the D spatial-grid extents (voxels per batch element). Host-side:
// `size_grid` is a host array here (the lib passes the narrowed shape vectors).
template <int D, typename offset_t>
static inline offset_t _grid_spatial(offset_t nbatch, const offset_t * size_grid) {
    offset_t nsp = 1;
    for (int d = 0; d < D; ++d) nsp *= size_grid[nbatch + d];
    return nsp;
}

// ============================================================================
//                                  PULL
//   out(*b,*grid,C) <- gather inp(*b,*spln,C) at grid(*b,*grid,D)
// ============================================================================
template <int D, int O, bound_t B, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, class AG>
CUGLOB void _pull_k(AO ao, AI ai, AG ag, int extrapolate, bound_t bound,
                    offset_t nvox, offset_t nsp)
{
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b  = (nsp > 0) ? i / nsp : offset_t(0);
        auto ic = ai.template peel_front_at<-(D + 1)>(b);   // (*spln, C), this batch
        auto oc = ao.template peel_front_at<-1>(i);         // (C,)
        auto gc = ag.template peel_front_at<-1>(i);         // (D,)
        reduce_t loc[D];
        for (int d = 0; d < D; ++d) loc[d] = static_cast<reduce_t>(gc(d));
        vox::pull<D, O, B, reduce_t, offset_t>(oc, ic, loc, extrapolate, bound);
    }
}

template <int D, int O, bound_t B, typename reduce_t, typename scalar_t, typename offset_t>
void pull(offset_t nbatch, int extrapolate, bound_t bound,
          scalar_t * out, const scalar_t * inp, const scalar_t * grid,
          const offset_t * size_out, const offset_t * size_inp, const offset_t * size_grid,
          const offset_t * stride_out, const offset_t * stride_inp, const offset_t * stride_grid,
          cudaStream_t stream = 0)
{
    const int rank = static_cast<int>(nbatch) + D + 1;
    auto ao = _any(out,  size_out,  stride_out,  rank);
    auto ai = _any(inp,  size_inp,  stride_inp,  rank);
    auto ag = _any(grid, size_grid, stride_grid, rank);

    const offset_t nsp  = _grid_spatial<D>(nbatch, size_grid);
    const offset_t nvox = ao.template size_front<-1>();     // batch x spatial_grid voxels (out is grid-shaped)

    _pull_k<D, O, B, reduce_t, scalar_t, offset_t, decltype(ao), decltype(ai), decltype(ag)>
        <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, ai, ag, extrapolate, bound, nvox, nsp);
    cudaStreamSynchronize(stream);
}

// ============================================================================
//                                  PUSH
//   out(*b,*spln,C) <- scatter inp(*b,*grid,C) at grid(*b,*grid,D)
//   (out pre-zeroed by the caller; scatter is atomic on device)
// ============================================================================
template <int D, int O, bound_t B, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, class AG>
CUGLOB void _push_k(AO ao, AI ai, AG ag, int extrapolate, bound_t bound,
                    offset_t nvox, offset_t nsp)
{
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b  = (nsp > 0) ? i / nsp : offset_t(0);
        auto oc = ao.template peel_front_at<-(D + 1)>(b);   // (*spln, C), this batch (atomic scatter target)
        auto ic = ai.template peel_front_at<-1>(i);         // (C,)
        auto gc = ag.template peel_front_at<-1>(i);         // (D,)
        reduce_t loc[D];
        for (int d = 0; d < D; ++d) loc[d] = static_cast<reduce_t>(gc(d));
        vox::push<D, O, B, reduce_t, offset_t>(oc, ic, loc, extrapolate, bound);
    }
}

template <int D, int O, bound_t B, typename reduce_t, typename scalar_t, typename offset_t>
void push(offset_t nbatch, int extrapolate, bound_t bound,
          scalar_t * out, const scalar_t * inp, const scalar_t * grid,
          const offset_t * size_out, const offset_t * size_inp, const offset_t * size_grid,
          const offset_t * stride_out, const offset_t * stride_inp, const offset_t * stride_grid,
          cudaStream_t stream = 0)
{
    const int rank = static_cast<int>(nbatch) + D + 1;
    auto ao = _any(out,  size_out,  stride_out,  rank);
    auto ai = _any(inp,  size_inp,  stride_inp,  rank);
    auto ag = _any(grid, size_grid, stride_grid, rank);

    const offset_t nsp  = _grid_spatial<D>(nbatch, size_grid);
    const offset_t nvox = ai.template size_front<-1>();     // batch x spatial_grid voxels (inp is grid-shaped)

    _push_k<D, O, B, reduce_t, scalar_t, offset_t, decltype(ao), decltype(ai), decltype(ag)>
        <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, ai, ag, extrapolate, bound, nvox, nsp);
    cudaStreamSynchronize(stream);
}

// ============================================================================
//                                  COUNT
//   out(*b,*spln,1) <- scatter the interpolation weights (no inp)
//   (out pre-zeroed by the caller; scatter is atomic on device)
// ============================================================================
template <int D, int O, bound_t B, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AG>
CUGLOB void _count_k(AO ao, AG ag, int extrapolate, bound_t bound,
                     offset_t nvox, offset_t nsp)
{
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b  = (nsp > 0) ? i / nsp : offset_t(0);
        auto oc = ao.template peel_front_at<-(D + 1)>(b);   // (*spln, 1), this batch (atomic scatter target)
        auto gc = ag.template peel_front_at<-1>(i);         // (D,)
        reduce_t loc[D];
        for (int d = 0; d < D; ++d) loc[d] = static_cast<reduce_t>(gc(d));
        vox::count<D, O, B, reduce_t, offset_t>(oc, loc, extrapolate, bound);
    }
}

template <int D, int O, bound_t B, typename reduce_t, typename scalar_t, typename offset_t>
void count(offset_t nbatch, int extrapolate, bound_t bound,
           scalar_t * out, const scalar_t * grid,
           const offset_t * size_out, const offset_t * size_grid,
           const offset_t * stride_out, const offset_t * stride_grid,
           cudaStream_t stream = 0)
{
    const int rank = static_cast<int>(nbatch) + D + 1;
    auto ao = _any(out,  size_out,  stride_out,  rank);
    auto ag = _any(grid, size_grid, stride_grid, rank);

    const offset_t nsp  = _grid_spatial<D>(nbatch, size_grid);
    const offset_t nvox = ag.template size_front<-1>();     // batch x spatial_grid voxels

    _count_k<D, O, B, reduce_t, scalar_t, offset_t, decltype(ao), decltype(ag)>
        <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, ag, extrapolate, bound, nvox, nsp);
    cudaStreamSynchronize(stream);
}

// ============================================================================
//                                  GRAD
//   out(*b,*grid,C,D) <- spatial gradient of the pull
// ============================================================================
template <int D, int O, bound_t B, bool ABS, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, class AG>
CUGLOB void _grad_k(AO ao, AI ai, AG ag, int extrapolate, bound_t bound,
                    offset_t nvox, offset_t nsp)
{
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b  = (nsp > 0) ? i / nsp : offset_t(0);
        auto ic = ai.template peel_front_at<-(D + 1)>(b);   // (*spln, C), this batch
        auto oc = ao.template peel_front_at<-2>(i);         // (C, D)
        auto gc = ag.template peel_front_at<-1>(i);         // (D,)
        reduce_t loc[D];
        for (int d = 0; d < D; ++d) loc[d] = static_cast<reduce_t>(gc(d));
        vox::grad<D, O, B, ABS, reduce_t, offset_t>(oc, ic, loc, extrapolate, bound);
    }
}

template <int D, int O, bound_t B, bool ABS, typename reduce_t, typename scalar_t, typename offset_t>
void grad(offset_t nbatch, int extrapolate, bound_t bound,
          scalar_t * out, const scalar_t * inp, const scalar_t * grid,
          const offset_t * size_out, const offset_t * size_inp, const offset_t * size_grid,
          const offset_t * stride_out, const offset_t * stride_inp, const offset_t * stride_grid,
          cudaStream_t stream = 0)
{
    const int orank = static_cast<int>(nbatch) + D + 2;     // out has the extra D axis
    const int rank  = static_cast<int>(nbatch) + D + 1;
    auto ao = _any(out,  size_out,  stride_out,  orank);
    auto ai = _any(inp,  size_inp,  stride_inp,  rank);
    auto ag = _any(grid, size_grid, stride_grid, rank);

    const offset_t nsp  = _grid_spatial<D>(nbatch, size_grid);
    const offset_t nvox = ag.template size_front<-1>();     // batch x spatial_grid voxels

    _grad_k<D, O, B, ABS, reduce_t, scalar_t, offset_t, decltype(ao), decltype(ai), decltype(ag)>
        <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, ai, ag, extrapolate, bound, nvox, nsp);
    cudaStreamSynchronize(stream);
}

FF_NAMESPACE_END(pushpull)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_PUSHPULL_CUDA
