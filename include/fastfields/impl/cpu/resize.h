#ifndef FF_RESIZE_CPU
#define FF_RESIZE_CPU
// Teeny-based resize (spline resampling) impl. resize is a pull whose sampling
// coordinate is an AFFINE map of the output voxel index:
//   loc[d] = scale[d] * (idx[d] + shift) - shift.
//
// SEPARABLE per-axis weight tables (grid regularity). Because loc[d] depends only
// on the d-th output coordinate, the per-axis neighbourhood (the O+1 sign-folded
// weights + strided offsets, pushpull::_make_axis) has only size_out[d] distinct
// values along axis d -- not `numel`. Precompute those tables ONCE (batch-
// invariant: the folded offsets are relative to each cell's base pointer, and the
// input spatial extents/strides don't vary per batch), then the per-voxel loop
// just assembles the D rows and runs the shared gather recursion. All spline-
// weight + boundary math leaves the hot loop; K = O+1 stays a compile-time count
// so pushpull::_pull_rec still fully unrolls and folds to immediates.
//
// resize has NO channel axis (every leading dim is batch, only the last D are
// spatial) and no FOV test (boundary is the kernel's job -> always in-bounds),
// matching the legacy Multiscale::resize. OUTPUT-DRIVEN gather -> disjoint writes,
// no atomics. Order O and boundary B are compile-time (B == bound_t::Dynamic
// routes the runtime bound); reduce_t is the accumulation type (double).
#include <teeny/teeny.h>
#include "kernels/pushpull/teeny.h"   // _axis / _make_axis / _pull_rec
#include "kernels/parallel.h"
#include "kernels/utils.h"
#include <vector>

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
    offset_t osize[D], iext[D], istr[D];
    for (int d = 0; d < D; ++d) {
        scale[d] = _scale[d];
        osize[d] = size_out[nbatch + d];
        iext[d]  = size_inp[nbatch + d];    // input spatial extent (batch-invariant)
        istr[d]  = stride_inp[nbatch + d];
    }

    // Per-axis neighbourhood tables, built once (grid regularity).
    using axis_t = pushpull::_axis<reduce_t, offset_t>;
    std::vector<std::vector<axis_t>> axtab(D);
    for (int d = 0; d < D; ++d) {
        axtab[d].resize(static_cast<size_t>(osize[d]));
        for (offset_t idx = 0; idx < osize[d]; ++idx) {
            const reduce_t coord = scale[d] * (static_cast<reduce_t>(idx) + shift) - shift;
            axtab[d][static_cast<size_t>(idx)] =
                pushpull::_make_axis<O, B, reduce_t, offset_t>(coord, iext[d], istr[d], bound);
        }
    }

    offset_t nsp = 1;
    for (int d = 0; d < D; ++d) nsp *= osize[d];

    const int rank = static_cast<int>(nbatch) + D;
    auto ao = tny::as_anyrank(out, size_out, stride_out, rank, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size_inp, stride_inp, rank, tny::copy_meta);

    const offset_t ncell = ao.template size_front<-D>();   // #batch cells
    const offset_t nvox  = ncell * nsp;                    // total output voxels

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = start; i < end; ++i)
    {
        const offset_t b  = (nsp > 0) ? i / nsp : 0;
        offset_t       sp = i - b * nsp;
        offset_t m[D];                                     // out spatial multi-index (row-major)
        for (int d = D - 1; d >= 0; --d) { m[d] = sp % osize[d]; sp /= osize[d]; }

        auto oc = ao.template peel_front_at<-D>(b);        // out spatial volume
        auto ic = ai.template peel_front_at<-D>(b);        // inp spatial volume

        // view the D precomputed neighbourhoods as compile-time-count rows and
        // run the shared separable gather (gather.h) -- O+1 taps per axis unroll.
        row_k<reduce_t, offset_t, O + 1> rows[D];
        for (int d = 0; d < D; ++d) {
            const axis_t & a = axtab[d][static_cast<size_t>(m[d])];
            rows[d].w = a.w; rows[d].o = a.off;
        }
        const reduce_t val = gather_sep<D, row_k<reduce_t, offset_t, O + 1>,
                                        scalar_t, offset_t, reduce_t>(ic.data(), rows);

        if      constexpr (D == 1) oc(m[0])              = static_cast<scalar_t>(val);
        else if constexpr (D == 2) oc(m[0], m[1])        = static_cast<scalar_t>(val);
        else                       oc(m[0], m[1], m[2])  = static_cast<scalar_t>(val);
    }});
}

FF_NAMESPACE_END(resize)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_RESIZE_CPU
