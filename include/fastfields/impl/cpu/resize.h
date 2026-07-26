#ifndef FF_RESIZE_CPU
#define FF_RESIZE_CPU
// Teeny-based resize (spline resampling) impl. One separable pull over the
// static spatial rank D (kernels/pushpull/teeny.h vox::pull_at) replaces the
// per-rank Multiscale gather trees: resize IS a pull whose sampling coordinate
// is an affine map of the output voxel index, so it reuses the pushpull kernel
// rather than carrying its own weight/index machinery.
//
// The (*batch, *spatial) decomposition is done with teeny's anyrank peel. resize
// has NO channel axis -- every leading dim is batch, only the last D are spatial
// -- so each (batch, out-voxel) reads a single interpolated scalar from the
// matching input batch volume. The output voxel index idx maps to the input
// coordinate  loc[d] = scale[d] * (idx[d] + shift) - shift  (jitfields' anchor
// convention). Boundary handling is the kernel's job (no FOV test -> extrapolate
// is always in-bounds), so out-of-range taps fold through the boundary condition
// exactly like the legacy Multiscale::resize.
//
// Order O and boundary B are compile-time (B == bound_t::Dynamic routes the
// runtime `bound` through the kernel); reduce_t is the accumulation type
// (double from the lib).
#include "kernels/pushpull/teeny.h"
#include "kernels/parallel.h"
#include "kernels/utils.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(resize)

template <
    int D, int O, bound_t B,
    typename reduce_t, typename scalar_t, typename offset_t
>
void loop(
          offset_t   nbatch,
          scalar_t * out,             // (*batch, *out_spatial) tensor
    const scalar_t * inp,             // (*batch, *inp_spatial) tensor
          reduce_t   shift,
    const reduce_t * _scale,          // [D] per-axis scaling
    const offset_t * size_out,        // [nbatch + D] output shape
    const offset_t * size_inp,        // [nbatch + D] input shape
    const offset_t * stride_out,      // [nbatch + D] output strides
    const offset_t * stride_inp,      // [nbatch + D] input strides
          bound_t    bound = bound_t::Dynamic   // runtime bound (B == Dynamic route)
)
{
    reduce_t scale[D];
    for (int d = 0; d < D; ++d) scale[d] = _scale[d];

    const int rank = static_cast<int>(nbatch) + D;
    auto ao = tny::as_anyrank(out, size_out, stride_out, rank, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size_inp, stride_inp, rank, tny::copy_meta);

    // output spatial grid extents (the last D dims); nsp = voxels per batch cell
    offset_t osp[D]; offset_t nsp = 1;
    for (int d = 0; d < D; ++d) { osp[d] = size_out[nbatch + d]; nsp *= osp[d]; }

    const offset_t ncell = ao.template size_front<-D>();   // #batch cells
    const offset_t nvox  = ncell * nsp;                    // total output voxels

    // Flat over every output voxel (batch x out-grid) -> full parallelism even at
    // nbatch <= 1. Reads are voxel-disjoint (no scatter), so no contention.
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = start; i < end; ++i)
    {
        const offset_t b  = (nsp > 0) ? i / nsp : 0;
        offset_t       sp = i - b * nsp;
        offset_t idx[D];                                   // out spatial multi-index (row-major)
        for (int d = D - 1; d >= 0; --d) { idx[d] = sp % osp[d]; sp /= osp[d]; }

        auto oc = ao.template peel_front_at<-D>(b);        // out spatial volume (this batch)
        auto ic = ai.template peel_front_at<-D>(b);        // inp spatial volume (this batch)

        reduce_t loc[D];
        for (int d = 0; d < D; ++d)
            loc[d] = scale[d] * (static_cast<reduce_t>(idx[d]) + shift) - shift;

        reduce_t val = pushpull::vox::pull_at<D, O, B, reduce_t, offset_t>(
            ic, loc, /*extrapolate=*/1, bound);

        if      constexpr (D == 1) oc(idx[0])                 = static_cast<scalar_t>(val);
        else if constexpr (D == 2) oc(idx[0], idx[1])         = static_cast<scalar_t>(val);
        else                       oc(idx[0], idx[1], idx[2]) = static_cast<scalar_t>(val);
    }});
}

FF_NAMESPACE_END(resize)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_RESIZE_CPU
