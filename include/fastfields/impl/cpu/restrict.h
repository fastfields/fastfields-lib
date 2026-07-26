#ifndef FF_RESTRICT_CPU
#define FF_RESTRICT_CPU
// Teeny-based restrict (spline restriction / prolongation-transpose) impl.
// restriction is the exact ADJOINT of resize's prolongation, so it is a scatter
// (kernels/pushpull/teeny.h vox::push_at) at the RECIPROCAL sampling scale --
// no separate dilated-kernel machinery. This drops the legacy padded-grid +
// boundary-wrap driver: the boundary folding now lives in push_at's neighbour
// builder (bound::sign/index on the coarse output extents), exactly where the
// old driver applied it.
//
// Derivation (spline evenness): the legacy restrict gathers each coarse voxel k
// from a dilated fine window with weight  spline((x_k - i)/scl),  x_k =
// (k+shift)*scl - shift. With c_i = (i+shift)/scl - shift (the fine voxel's
// coordinate in coarse space), (x_k - i)/scl = k - c_i, and spline is even, so
// the weight is spline(c_i - k) -- precisely the prolongation weight from coarse
// k to fine i. Summing over fine i is therefore P^T applied to the fine field:
// for each fine voxel, scatter its value into the coarse grid at loc = c_i.
//
// out (coarse) is pre-zeroed by the caller and accumulated into; the flat
// voxel-parallel scatter is race-free via the lock-free host atomic add (adjoint
// of resize's disjoint gather). Order O and boundary B are compile-time (B ==
// bound_t::Dynamic routes the runtime `bound`); reduce_t is the accumulation
// type (double from the lib).
#include "kernels/pushpull/teeny.h"
#include "kernels/parallel.h"
#include "kernels/utils.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(restrict)

template <
    int D, int O, bound_t B,
    typename reduce_t, typename scalar_t, typename offset_t
>
void loop(
          offset_t   nbatch,
          scalar_t * out,             // (*batch, *out_spatial) coarse tensor (pre-zeroed)
    const scalar_t * inp,             // (*batch, *inp_spatial) fine tensor
          reduce_t   shift,
    const reduce_t * _scale,          // [D] per-axis scaling (fine / coarse)
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

    // input (fine) spatial grid extents (the last D dims); nsp = voxels per cell
    offset_t isp[D]; offset_t nsp = 1;
    for (int d = 0; d < D; ++d) { isp[d] = size_inp[nbatch + d]; nsp *= isp[d]; }

    const offset_t ncell = ai.template size_front<-D>();   // #batch cells
    const offset_t nvox  = ncell * nsp;                    // total fine voxels

    // Flat over every fine voxel (batch x fine-grid). Each scatters into the
    // shared coarse output via the lock-free atomic add -> race-free, full
    // parallelism even at nbatch <= 1 (adjoint of resize's disjoint gather).
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = start; i < end; ++i)
    {
        const offset_t b  = (nsp > 0) ? i / nsp : 0;
        offset_t       sp = i - b * nsp;
        offset_t idx[D];                                   // fine spatial multi-index (row-major)
        for (int d = D - 1; d >= 0; --d) { idx[d] = sp % isp[d]; sp /= isp[d]; }

        auto oc = ao.template peel_front_at<-D>(b);        // coarse out volume (scatter target)
        auto ic = ai.template peel_front_at<-D>(b);        // fine inp volume

        reduce_t val;
        if      constexpr (D == 1) val = static_cast<reduce_t>(ic(idx[0]));
        else if constexpr (D == 2) val = static_cast<reduce_t>(ic(idx[0], idx[1]));
        else                       val = static_cast<reduce_t>(ic(idx[0], idx[1], idx[2]));

        reduce_t loc[D];                                   // fine voxel's coordinate in coarse space
        for (int d = 0; d < D; ++d)
            loc[d] = (static_cast<reduce_t>(idx[d]) + shift) / scale[d] - shift;

        pushpull::vox::push_at<D, O, B, reduce_t, offset_t>(
            oc, val, loc, /*extrapolate=*/1, bound);
    }});
}

FF_NAMESPACE_END(restrict)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_RESTRICT_CPU
