#ifndef FF_RESTRICT_CPU
#define FF_RESTRICT_CPU
// Teeny-based restrict (spline restriction) impl. OUTPUT-DRIVEN dilated gather:
// each coarse output voxel gathers from a scale-widened window of the fine input
// (the shared kernels/restrict.h Multiscale::gather), so writes are output-
// disjoint and NO atomics are needed -- this is the grid-regular path resize/
// restrict have always used, and the reason restrict iterates OUTPUT voxels, not
// input ones (a scatter would contend). teeny replaces only the batch/boundary
// plumbing.
//
// Boundary is applied on the COARSE side (the adjoint of resize's coarse-side
// boundary): a coarse tap outside [0,n) folds back in via bound::index/sign.
// Rather than the legacy padded-grid forward scatter (which collides at the
// boundary and needs an atomic add), we invert the fold ONCE into per-axis
// source lists: for each real output index m, the (few) extended coarse
// positions k with index(k)=m and their signs. Interior m has the single entry
// {m,+1}, so the bulk cost is one gather per output voxel; only the boundary
// shell sums a handful. Fully output-parallel, atomic-free, exact same result as
// the padded scatter (a reorganisation of the same sum: sum over source k of
// sign(k)*gather(k), grouped by target index(k)).
//
// Order O and boundary B are compile-time (B == bound_t::Dynamic routes the
// runtime bound); reduce_t is the accumulation type (double from the lib).
#include <teeny/teeny.h>
#include "kernels/restrict.h"
#include "kernels/bounds.h"
#include "kernels/parallel.h"
#include "kernels/utils.h"
#include <vector>
#include <utility>

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(restrict)

// Coarse-side boundary sign/index for an extended tap k (compile-time B, or the
// runtime `rt` when B == bound_t::Dynamic) -- mirrors the pushpull kernel's
// _bound_at, on the OUTPUT (coarse) extents.
template <bound_t B, typename offset_t>
static inline void _rbound(bound_t rt, offset_t k, offset_t n, int8_t & s, offset_t & ix) {
    if (B == bound_t::Dynamic) {
        s  = bound::sign (rt, k, n);
        ix = bound::index(rt, k, n);
    } else {
        s  = bound::utils<B>::template sign <offset_t>(k, n);
        ix = bound::utils<B>::template index<offset_t>(k, n);
    }
}

template <
    int D, int O, bound_t B,
    typename reduce_t, typename scalar_t, typename offset_t
>
void loop(
          offset_t   nbatch,
          scalar_t * out,             // (*batch, *out_spatial) coarse tensor (overwritten)
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
    constexpr spline::type I = static_cast<spline::type>(O);
    constexpr int pad = (O + 1) / 2;           // coarse-side fold radius

    reduce_t scale[D];
    for (int d = 0; d < D; ++d) scale[d] = _scale[d];

    // fine-input spatial size/stride (the gather reads these)
    offset_t fsize[D], fstride[D];
    for (int d = 0; d < D; ++d) { fsize[d]  = size_inp[nbatch + d]; fstride[d] = stride_inp[nbatch + d]; }

    // coarse-output spatial extents
    offset_t osize[D];
    for (int d = 0; d < D; ++d) osize[d] = size_out[nbatch + d];

    // per-axis fold lists: src[d][m] = the (extended coarse position k, sign)
    // pairs with index(k) == m. Built once; interior m holds just {m,+1}.
    std::vector<std::vector<std::vector<std::pair<offset_t, int8_t>>>> src(D);
    for (int d = 0; d < D; ++d) {
        const offset_t n = osize[d];
        src[d].assign(static_cast<size_t>(n), {});
        for (offset_t k = -static_cast<offset_t>(pad); k < n + pad; ++k) {
            int8_t s; offset_t m;
            _rbound<B>(bound, k, n, s, m);
            if (s == 0) continue;                          // zero-boundary tap: dropped
            if (m >= 0 && m < n)
                src[d][static_cast<size_t>(m)].push_back({k, s});
        }
    }

    offset_t nsp = 1;
    for (int d = 0; d < D; ++d) nsp *= osize[d];

    const int rank = static_cast<int>(nbatch) + D;
    auto ao = tny::as_anyrank(out, size_out, stride_out, rank, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size_inp, stride_inp, rank, tny::copy_meta);

    const offset_t ncell = ao.template size_front<-D>();   // #batch cells
    const offset_t nvox  = ncell * nsp;                    // total coarse output voxels

    // Flat over every output (coarse) voxel -> disjoint writes, NO atomics.
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = start; i < end; ++i)
    {
        const offset_t b  = (nsp > 0) ? i / nsp : 0;
        offset_t       sp = i - b * nsp;
        offset_t m[D];                                     // coarse spatial multi-index (row-major)
        for (int d = D - 1; d >= 0; --d) { m[d] = sp % osize[d]; sp /= osize[d]; }

        auto oc = ao.template peel_front_at<-D>(b);        // coarse out volume
        auto ic = ai.template peel_front_at<-D>(b);        // fine inp volume
        const scalar_t * ip = ic.data();

        // Sum sign(k)*gather(k) over every folded coarse source of this output
        // voxel (the per-axis product), accumulating locally -> single write.
        reduce_t acc = static_cast<reduce_t>(0);
        if constexpr (D == 1) {
            for (auto & e0 : src[0][m[0]]) {
                const offset_t loc[1] = { e0.first };
                acc += static_cast<reduce_t>(e0.second)
                     * Multiscale<1, zero, I>::gather(ip, loc, fsize, fstride, scale, shift);
            }
        } else if constexpr (D == 2) {
            for (auto & e0 : src[0][m[0]])
            for (auto & e1 : src[1][m[1]]) {
                const offset_t loc[2] = { e0.first, e1.first };
                acc += static_cast<reduce_t>(e0.second * e1.second)
                     * Multiscale<2, zero, I>::gather(ip, loc, fsize, fstride, scale, shift);
            }
        } else {
            for (auto & e0 : src[0][m[0]])
            for (auto & e1 : src[1][m[1]])
            for (auto & e2 : src[2][m[2]]) {
                const offset_t loc[3] = { e0.first, e1.first, e2.first };
                acc += static_cast<reduce_t>(e0.second * e1.second * e2.second)
                     * Multiscale<3, zero, I>::gather(ip, loc, fsize, fstride, scale, shift);
            }
        }

        if      constexpr (D == 1) oc(m[0])              = static_cast<scalar_t>(acc);
        else if constexpr (D == 2) oc(m[0], m[1])        = static_cast<scalar_t>(acc);
        else                       oc(m[0], m[1], m[2])  = static_cast<scalar_t>(acc);
    }});
}

FF_NAMESPACE_END(restrict)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_RESTRICT_CPU
