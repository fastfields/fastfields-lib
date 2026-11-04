#pragma once
/* CPU loops for displacement-field inversion ("inv_flow").
 *
 * The math is one layer down and in two headers:
 * impl/kernels/simplex.h holds the shared simplex walk, the barycentric
 * coordinates and the owner-election protocol (and explains why the op runs in
 * two passes), while impl/kernels/inv_flow.h holds what makes this mesh the
 * implicit one -- the cell stencils and the simplex-index decode.
 *
 * This header owns only the iteration: three `parallel_for`s over, in order,
 *
 *   1. every voxel      -- set the owner buffer to "unclaimed";
 *   2. every simplex    -- rasterise it and claim the voxels it covers;
 *   3. every voxel      -- read the winning simplex back and interpolate.
 *
 * The passes are separated by the `parallel_for` joins, so pass 2 sees every
 * claim pass 1 made and pass 3 sees every claim pass 2 made. Within pass 2 the
 * only shared write is the atomic minimum on the owner buffer, and within pass
 * 3 each voxel has exactly one writer -- so the whole op is free of data races
 * and its result is independent of FF_GRAIN_SIZE.
 */
#include <fastfields/core/cuda_switch.h>
#include <fastfields/impl/kernels/inv_flow.h>
#include <fastfields/impl/kernels/simplex.h>
#include <fastfields/impl/kernels/batch.h>
#include <fastfields/impl/kernels/parallel.h>

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(inv_flow)

/* Invert a displacement field.
 *
 * `out` and `disp` are (*batch, *spatial, ndim) and `owner` is
 * (*batch, *spatial); `size` is the shared (*batch, *spatial) shape, of length
 * nbatch + ndim, and the two `stride_*` arrays for the fields have one more
 * entry than that for the trailing component axis.
 *
 * On return every voxel of `out` holds either the inverse displacement or a
 * quiet NaN, and the matching `owner` entry either the index of the simplex
 * the value came from or -1. Neither output needs to be initialised by the
 * caller. */
template <int ndim, typename scalar_t, typename offset_t, typename reduce_t>
void loop(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, ndim)
          int64_t  * owner,         // (*batch, *spatial)
    const scalar_t * disp,          // (*batch, *spatial, ndim)
    const offset_t * size,          // [nbatch + ndim]
    const offset_t * stride_out,    // [nbatch + ndim + 1]
    const offset_t * stride_own,    // [nbatch + ndim]
    const offset_t * stride_disp)   // [nbatch + ndim + 1]
{
    const offset_t * shape       = size + nbatch;
    const offset_t * sout        = stride_out  + nbatch;
    const offset_t * sown        = stride_own  + nbatch;
    const offset_t * sdisp       = stride_disp + nbatch;
    const offset_t   sout_comp   = stride_out [nbatch + ndim];
    const offset_t   sdisp_comp  = stride_disp[nbatch + ndim];

    const int64_t nvox   = static_cast<int64_t>(prod<ndim>(shape));
    const int64_t nelem  = static_cast<int64_t>(prod(size, nbatch));

    // One cell per interior corner of the grid, each cut into
    // Cell<ndim>::simplices simplices.
    int64_t ncell = 1;
    for (int d = 0; d < ndim; ++d)
        ncell *= static_cast<int64_t>(shape[d]) - 1;
    const int64_t nsimplex = ncell * Cell<ndim>::simplices;

    if (nvox <= 0 || nelem <= 0) return;

    // Each pass walks the batch element in its outer index, so `b` is constant
    // over runs of nvox (or nsimplex) consecutive items. The three base
    // offsets are therefore recomputed only when it changes, rather than once
    // per voxel; with nbatch == 0 -- a single field, the common case --
    // `index2offset` returns 0 without looping at all.
    const int nb = static_cast<int>(nbatch);

    // ---- pass 1: no simplex owns anything yet ---------------------------
    parallel_for(0, nelem * nvox, GRAIN_SIZE, [&](long start, long end) {
        int64_t   cached  = -1;
        int64_t * o_base  = owner;
        for (int64_t i = start; i < end; ++i) {
            const int64_t  b = i / nvox;
            const offset_t v = static_cast<offset_t>(i % nvox);
            if (b != cached) {
                const offset_t bo = static_cast<offset_t>(b);
                o_base = owner + index2offset(bo, nb, size, stride_own);
                cached = b;
            }
            offset_t sub[ndim];
            simplex::unravel<ndim>(v, shape, sub);
            o_base[sub2offset<ndim>(sub, sown)] = simplex::unclaimed();
        }
    });

    // ---- pass 2: rasterise each deformed simplex ------------------------
    if (nsimplex > 0) {
        parallel_for(0, nelem * nsimplex, GRAIN_SIZE,
                     [&](long start, long end) {
            int64_t          cached = -1;
            const scalar_t * d_base = disp;
            int64_t        * o_base = owner;
            for (int64_t i = start; i < end; ++i) {
                const int64_t b   = i / nsimplex;
                const int64_t sid = i % nsimplex;
                if (b != cached) {
                    const offset_t bo = static_cast<offset_t>(b);
                    d_base = disp  + index2offset(bo, nb, size, stride_disp);
                    o_base = owner + index2offset(bo, nb, size, stride_own);
                    cached = b;
                }

                // `vsrc` is not read here: `corners` computes both domains
                // because the resolve pass needs both, and the dead stores
                // fold away once it is inlined.
                reduce_t vsrc[ndim + 1][ndim], vdst[ndim + 1][ndim];
                corners<ndim>(vsrc, vdst, sid, d_base, shape, sdisp,
                              sdisp_comp);
                simplex::Rasteriser<ndim>::claim(o_base, sid, vdst, shape,
                                                 sown);
            }
        });
    }

    // ---- pass 3: one writer per voxel ------------------------------------
    parallel_for(0, nelem * nvox, GRAIN_SIZE, [&](long start, long end) {
        int64_t          cached   = -1;
        const scalar_t * d_base   = disp;
        int64_t        * o_base   = owner;
        scalar_t       * out_base = out;
        for (int64_t i = start; i < end; ++i) {
            const int64_t  b = i / nvox;
            const offset_t v = static_cast<offset_t>(i % nvox);
            if (b != cached) {
                const offset_t bo = static_cast<offset_t>(b);
                d_base   = disp  + index2offset(bo, nb, size, stride_disp);
                o_base   = owner + index2offset(bo, nb, size, stride_own);
                out_base = out   + index2offset(bo, nb, size, stride_out);
                cached   = b;
            }
            offset_t sub[ndim];
            simplex::unravel<ndim>(v, shape, sub);

            int64_t  * ow  = o_base   + sub2offset<ndim>(sub, sown);
            scalar_t * val = out_base + sub2offset<ndim>(sub, sout);

            reduce_t srcp[ndim];
            const int64_t sid = *ow;
            const bool got = (sid != simplex::unclaimed())
                && source<ndim>(srcp, sid, sub, d_base, shape,
                                sdisp, sdisp_comp);
            if (got) {
                for (int d = 0; d < ndim; ++d)
                    val[d * sout_comp] = static_cast<scalar_t>(
                        srcp[d] - static_cast<reduce_t>(sub[d]));
            } else {
                *ow = simplex::uncovered();
                const scalar_t missing =
                    simplex::not_a_number(static_cast<scalar_t>(0));
                for (int d = 0; d < ndim; ++d)
                    val[d * sout_comp] = missing;
            }
        }
    });
}

FF_NAMESPACE_END(inv_flow)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
