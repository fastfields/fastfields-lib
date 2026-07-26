#ifndef FF_RESTRICT_CPU
#define FF_RESTRICT_CPU
// Teeny-based restrict (spline restriction) impl. restriction is the exact
// ADJOINT of resize's prolongation, so it is built by TRANSPOSING resize's pull
// tap enumeration -- guaranteeing the adjoint identity for every shift/order/
// scale/boundary (a fixed-pad coarse fold truncates even-order support at
// shift=0, and an open nearest window drops tie taps; transposing the actual
// pull avoids both by construction).
//
// SEPARABLE per-axis weight tables (grid regularity). resize sends fine voxel i
// to coarse tap nb=low(c_i)+k with weight s*w(|c_i-nb|) (c_i = the fine voxel's
// coordinate in coarse space, s/index = the coarse boundary fold). Its transpose
// gathers, for coarse output m, every fine voxel whose pull reaches m. Because
// the pull is separable, we tabulate per axis d: W[d][m] = the (fine offset,
// signed weight) taps landing on output m along d. Interior m holds ~scale*(O+1)
// taps; the D-dim gather is one product over the per-axis tables (NO corner
// blowup, NO per-voxel weight evals). Tables are built ONCE (the pull weights
// depend only on the per-axis coordinate) and reused across every orthogonal
// line and every batch cell.
//
// OUTPUT-DRIVEN: iterate coarse output voxels -> disjoint accumulates, NO atomics
// (a scatter would contend). Order O and boundary B are compile-time (B ==
// bound_t::Dynamic routes the runtime bound through _bound_at); reduce_t is the
// accumulation type (double from the lib). restriction ACCUMULATES into the
// pre-zeroed out (the documented contract; matches the CUDA path).
#include <teeny/teeny.h>
#include "kernels/pushpull/teeny.h"   // _low / _fastweight / _bound_at: resize's pull taps
#include "kernels/bounds.h"
#include "kernels/parallel.h"
#include "kernels/utils.h"
#include <cmath>
#include <vector>
#include <utility>

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(restrict)

template <
    int D, int O, bound_t B,
    typename reduce_t, typename scalar_t, typename offset_t
>
void loop(
          offset_t   nbatch,
          scalar_t * out,             // (*batch, *out_spatial) coarse tensor (pre-zeroed; accumulated)
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
    offset_t osize[D], isize[D], fstride[D];
    for (int d = 0; d < D; ++d) {
        scale[d]   = _scale[d];
        osize[d]   = size_out[nbatch + d];   // coarse
        isize[d]   = size_inp[nbatch + d];   // fine
        fstride[d] = stride_inp[nbatch + d];
    }

    // Per-axis transpose tables: W[d][m] = (fine offset, signed weight) taps that
    // resize's pull sends onto coarse output m along axis d. Exact transpose of
    // pushpull::_make_axis (same _low, same tap nb, same s/index, same weight).
    using entry_t = std::pair<offset_t, reduce_t>;
    std::vector<std::vector<std::vector<entry_t>>> W(D);
    for (int d = 0; d < D; ++d) {
        const offset_t nc = osize[d], nf = isize[d];
        W[d].assign(static_cast<size_t>(nc), {});
        for (offset_t i = 0; i < nf; ++i) {
            const reduce_t c   = (static_cast<reduce_t>(i) + shift) / scale[d] - shift;
            const offset_t low = pushpull::_low<O, reduce_t, offset_t>(c);
            for (int k = 0; k <= O; ++k) {
                const offset_t nb = low + static_cast<offset_t>(k);
                int8_t s; offset_t ix;
                pushpull::_bound_at<B>(bound, nb, nc, s, ix);
                if (s == 0) continue;                          // zero-boundary tap: dropped
                const reduce_t w = static_cast<reduce_t>(s)
                    * pushpull::_fastweight<O>(
                        static_cast<reduce_t>(std::fabs(c - static_cast<reduce_t>(nb))));
                W[d][static_cast<size_t>(ix)].push_back({ i * fstride[d], w });
            }
        }
    }

    offset_t nsp = 1;
    for (int d = 0; d < D; ++d) nsp *= osize[d];

    const int rank = static_cast<int>(nbatch) + D;
    auto ao = tny::as_anyrank(out, size_out, stride_out, rank, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size_inp, stride_inp, rank, tny::copy_meta);

    const offset_t ncell = ao.template size_front<-D>();   // #batch cells
    const offset_t nvox  = ncell * nsp;                    // total coarse output voxels

    // Flat over every output (coarse) voxel -> disjoint accumulates, NO atomics.
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

        // one separable gather over the per-axis tap tables (product of axes)
        reduce_t acc = static_cast<reduce_t>(0);
        if constexpr (D == 1) {
            for (const auto & e0 : W[0][m[0]])
                acc += e0.second * static_cast<reduce_t>(ip[e0.first]);
        } else if constexpr (D == 2) {
            for (const auto & e0 : W[0][m[0]])
            for (const auto & e1 : W[1][m[1]])
                acc += (e0.second * e1.second) * static_cast<reduce_t>(ip[e0.first + e1.first]);
        } else {
            for (const auto & e0 : W[0][m[0]])
            for (const auto & e1 : W[1][m[1]])
            for (const auto & e2 : W[2][m[2]])
                acc += (e0.second * e1.second * e2.second)
                     * static_cast<reduce_t>(ip[e0.first + e1.first + e2.first]);
        }

        if      constexpr (D == 1) oc(m[0])              += static_cast<scalar_t>(acc);
        else if constexpr (D == 2) oc(m[0], m[1])        += static_cast<scalar_t>(acc);
        else                       oc(m[0], m[1], m[2])  += static_cast<scalar_t>(acc);
    }});
}

FF_NAMESPACE_END(restrict)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_RESTRICT_CPU
