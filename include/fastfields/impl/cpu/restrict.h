#ifndef FF_RESTRICT_CPU
#define FF_RESTRICT_CPU
// Teeny-based restrict (spline restriction) impl. restriction is the exact
// ADJOINT of resize's prolongation, so it is built by TRANSPOSING resize's pull
// tap enumeration -- guaranteeing the adjoint identity for every shift/order/
// scale/boundary (a fixed-pad coarse fold truncates even-order support at
// shift=0, and an open nearest window drops tie taps; transposing the actual
// pull avoids both by construction).
//
// SEPARABLE, FLAT CSR weight tables (grid regularity + device portability).
// resize sends fine voxel i to coarse tap nb=low(c_i)+k with weight s*w(|c_i-nb|)
// (c_i = the fine voxel's coordinate in coarse space, s/index = the coarse
// boundary fold). Its transpose gathers, for coarse output m, every fine voxel
// whose pull reaches m. Because the pull is separable we tabulate PER AXIS d a
// flat CSR: row[d][0..nc] + (foff[d], fwt[d]) = the (fine-offset, signed-weight)
// taps landing on output m along d. Interior m holds ~scale*(O+1) taps; the
// D-dim gather is one product over the per-axis tables (kernels/restrict.h
// csr_gather) -- no corner blow-up, no per-voxel weight evals. Tables are built
// ONCE (the pull weights depend only on the per-axis coordinate) and reused
// across every orthogonal line and every batch cell.
//
// FLAT arrays (not nested std containers): the six per-axis buffers are exactly
// what the CUDA port cudaMemcpy's to the device, and csr_gather is device-capable
// (CUDEV) -- CPU and CUDA share the representation and the gather verbatim.
//
// OUTPUT-DRIVEN: iterate coarse output voxels -> disjoint accumulates, NO atomics
// (a scatter would contend). Order O and boundary B are compile-time (B ==
// bound_t::Dynamic routes the runtime bound through _bound_at); reduce_t is the
// accumulation type (double). restriction ACCUMULATES into the pre-zeroed out
// (the documented contract; matches the CUDA path).
#include <teeny/teeny.h>
#include "kernels/pushpull/teeny.h"   // _low / _fastweight / _bound_at + gather_sep / row_n
#include "kernels/parallel.h"
#include "kernels/utils.h"
#include <cmath>
#include <vector>

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

    // Per-axis FLAT CSR transpose tables. A tap on output m along d is the exact
    // transpose of pushpull::_make_axis (same _low, tap nb, s/index, weight), so
    // restrict is adjoint-exact. Two passes: count taps per output index, prefix-
    // sum into row offsets, then scatter (fine offset, signed weight) into place.
    std::vector<offset_t> row[D], foff[D];
    std::vector<reduce_t> fwt[D];
    for (int d = 0; d < D; ++d) {
        const offset_t nc = osize[d], nf = isize[d];
        row[d].assign(static_cast<size_t>(nc) + 1, 0);
        for (offset_t i = 0; i < nf; ++i) {
            const reduce_t c   = (static_cast<reduce_t>(i) + shift) / scale[d] - shift;
            const offset_t low = pushpull::_low<O, reduce_t, offset_t>(c);
            for (int k = 0; k <= O; ++k) {
                int8_t s; offset_t ix;
                pushpull::_bound_at<B>(bound, low + static_cast<offset_t>(k), nc, s, ix);
                if (s != 0) row[d][static_cast<size_t>(ix) + 1] += 1;
            }
        }
        for (offset_t m = 0; m < nc; ++m) row[d][m + 1] += row[d][m];   // -> CSR offsets
        const offset_t tot = row[d][nc];
        foff[d].resize(static_cast<size_t>(tot));
        fwt[d].resize(static_cast<size_t>(tot));
        std::vector<offset_t> cur(row[d].begin(), row[d].begin() + nc);  // write cursor per row
        for (offset_t i = 0; i < nf; ++i) {
            const reduce_t c   = (static_cast<reduce_t>(i) + shift) / scale[d] - shift;
            const offset_t low = pushpull::_low<O, reduce_t, offset_t>(c);
            for (int k = 0; k <= O; ++k) {
                const offset_t nb = low + static_cast<offset_t>(k);
                int8_t s; offset_t ix;
                pushpull::_bound_at<B>(bound, nb, nc, s, ix);
                if (s == 0) continue;
                const reduce_t w = static_cast<reduce_t>(s)
                    * pushpull::_fastweight<O>(
                        static_cast<reduce_t>(std::fabs(c - static_cast<reduce_t>(nb))));
                const offset_t e = cur[static_cast<size_t>(ix)]++;
                foff[d][static_cast<size_t>(e)] = i * fstride[d];
                fwt[d][static_cast<size_t>(e)]  = w;
            }
        }
    }

    // raw pointer views of the flat CSR (device port passes device pointers here)
    const offset_t * rowp[D]; const offset_t * foffp[D]; const reduce_t * fwtp[D];
    for (int d = 0; d < D; ++d) { rowp[d] = row[d].data(); foffp[d] = foff[d].data(); fwtp[d] = fwt[d].data(); }

    offset_t nsp = 1;
    for (int d = 0; d < D; ++d) nsp *= osize[d];

    const int rank = static_cast<int>(nbatch) + D;
    auto ao = tny::as_anyrank(out, size_out, stride_out, rank, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size_inp, stride_inp, rank, tny::copy_meta);

    const offset_t ncell = ao.template size_front<-D>();   // #batch cells
    const offset_t nvox  = ncell * nsp;                    // total coarse output voxels

    // Flat over every output (coarse) voxel -> disjoint accumulates, NO atomics.
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    // Peel the batch cell ONCE per cell (changes only every nsp voxels), not per
    // voxel -- see the resize driver.
    offset_t cur_b = (nsp > 0) ? static_cast<offset_t>(start) / nsp : 0;
    auto oc = ao.template peel_front_at<-D>(cur_b);        // coarse out volume
    auto ic = ai.template peel_front_at<-D>(cur_b);        // fine inp volume
    for (offset_t i = start; i < end; ++i)
    {
        const offset_t b  = (nsp > 0) ? i / nsp : 0;
        if (b != cur_b) {
            oc = ao.template peel_front_at<-D>(b);
            ic = ai.template peel_front_at<-D>(b);
            cur_b = b;
        }
        offset_t sp = i - b * nsp;
        offset_t m[D];                                     // coarse spatial multi-index (row-major)
        for (int d = D - 1; d >= 0; --d) { m[d] = sp % osize[d]; sp /= osize[d]; }

        // view each axis's CSR slice as a runtime-count row and run the shared
        // separable gather (gather.h) -- the same recursion resize/pull use.
        row_n<reduce_t, offset_t> rows[D];
        for (int d = 0; d < D; ++d) {
            const offset_t lo = rowp[d][m[d]], hi = rowp[d][m[d] + 1];
            rows[d].w = fwtp[d] + lo; rows[d].o = foffp[d] + lo; rows[d].n = hi - lo;
        }
        const reduce_t acc = gather_sep<D, row_n<reduce_t, offset_t>,
                                        scalar_t, offset_t, reduce_t>(ic.data(), rows);

        if      constexpr (D == 1) oc(m[0])              += static_cast<scalar_t>(acc);
        else if constexpr (D == 2) oc(m[0], m[1])        += static_cast<scalar_t>(acc);
        else                       oc(m[0], m[1], m[2])  += static_cast<scalar_t>(acc);
    }});
}

FF_NAMESPACE_END(restrict)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_RESTRICT_CPU
