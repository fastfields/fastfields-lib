#ifndef FF_RESIZE_CPU
#define FF_RESIZE_CPU
// Teeny-based resize (spline resampling) impl. resize is a pull whose sampling
// coordinate is an AFFINE map of the output voxel index:
//   loc[d] = scale[d] * (idx[d] + shift) - shift.
//
// SEPARABLE per-axis weight tables (grid regularity). Because loc[d] depends only
// on the d-th output coordinate, the per-axis neighbourhood (the O+1 sign-folded
// weights + strided offsets, pushpull::_make_axis) has only osize[d] distinct
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
#include <type_traits>
#include <vector>

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(resize)

// The TENSORS are the arguments: `ao`/`ai` are teeny `anyrank` carriers that the
// caller (*-lib) built once from its own DLPack tensors. Every geometric
// quantity -- the batch rank, the output spatial extents, the input spatial
// extents and strides, the batch offsets -- is derived from the carrier that
// OWNS it, so the four shared size[]/stride[] arrays and the `nbatch` count are
// gone (TEENY-MIGRATION.md sec. 9, R2/R3: each tensor carries its own metadata,
// and no argument restates something a tensor already knows). Note the two
// tensors' spatial extents genuinely differ here -- that is the whole operation
// -- which is exactly why one shared size[] pair for two operands was the wrong
// shape.
//
// TEMPLATE SHAPE (Phase A's, fastfields-cpu-impl#60): one parameter per TENSOR
// (`AO`, `AI`), plus the ordinary deduced parameter per scalar. D/O/B stay
// compile-time template parameters supplied by the *-lib dispatch (R1) -- they
// are not geometry and are not derivable from a carrier.
//
// The read-only operand is a carrier of `const scalar_t` (R4), so writing
// through `ai` is a compile error rather than a convention.
template <
    int D, int O, bound_t B,
    class AO, class AI, typename reduce_t
>
void loop(
          AO         ao   ,                     // (*batch, *out_spatial) carrier
          AI         ai   ,                     // (*batch, *inp_spatial) carrier (const element)
          reduce_t   shift,
    const reduce_t * _scale,                    // [D] per-axis scaling
          bound_t    bound = bound_t::Dynamic   // runtime bound (B == Dynamic route)
)
{
    using offset_t = decltype(ao.size(0));   // the carrier's own offset type
    using scalar_t = typename std::remove_pointer<decltype(AO::data)>::type;

    // The one precondition the carriers do not already enforce between them.
    // `peel_front_at<-D>` asserts ndim >= D on each carrier by itself, but
    // nothing ties the two ranks together -- and the batch cell index is SHARED
    // between them, so a rank mismatch would peel `ai` at an index its own batch
    // does not have. Entry-only, outside every loop, compiled out under NDEBUG.
    // Batch EXTENT equality stays the *-lib's CHECK_SAME_BATCH (behavioural ABI,
    // unchanged).
    _TNY_CHECK(ao.ndim == ai.ndim,
               "resize::loop: out and inp carriers must have the same rank");

    const int nbatch = ao.ndim - D;

    reduce_t scale[D];
    offset_t osize[D], iext[D], istr[D];
    for (int d = 0; d < D; ++d) {
        scale[d] = _scale[d];
        osize[d] = ao.size(nbatch + d);
        iext[d]  = ai.size(nbatch + d);     // input spatial extent (batch-invariant)
        istr[d]  = ai.stride(nbatch + d);
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

    const offset_t ncell = ao.template size_front<-D>();   // #batch cells
    const offset_t nvox  = ncell * nsp;                    // total output voxels

    parallel_for(0, nvox, GRAIN_SIZE, [&](int64_t start, int64_t end) {
    // Peel the batch cell ONCE per cell (it changes only every nsp voxels), not
    // per voxel: peel_front_at does a mixed-radix decode + mapping build, wasted
    // if repeated across a cell's spatial sweep.
    offset_t cur_b = (nsp > 0) ? static_cast<offset_t>(start) / nsp : 0;
    auto oc = ao.template peel_front_at<-D>(cur_b);        // out spatial volume
    auto ic = ai.template peel_front_at<-D>(cur_b);        // inp spatial volume
    for (offset_t i = start; i < end; ++i)
    {
        const offset_t b  = (nsp > 0) ? i / nsp : 0;
        if (b != cur_b) {
            oc = ao.template peel_front_at<-D>(b);
            ic = ai.template peel_front_at<-D>(b);
            cur_b = b;
        }
        offset_t sp = i - b * nsp;
        offset_t m[D];                                     // out spatial multi-index (row-major)
        for (int d = D - 1; d >= 0; --d) { m[d] = sp % osize[d]; sp /= osize[d]; }

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
