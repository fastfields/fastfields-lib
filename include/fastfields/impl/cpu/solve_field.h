#ifndef FF_SOLVE_FIELD_CPU
#define FF_SOLVE_FIELD_CPU
#include <vector>
#include "fastfields/core/cuda_switch.h"
#include "fastfields/core/utils.h"
#include "fastfields/core/batch.h"
#include "fastfields/core/parallel.h"

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(solve_field)

/**********************************************************************
 *
 *  Vector (BLAS-1 style) primitives for the iterative field solvers.
 *
 *  These are the element loops that the conjugate-gradient driver needs
 *  on top of the operator applications it already gets from `posdef`
 *  (`sym_matvec` / `sym_solve`) and `reg_field` (`matvec` / `diag`):
 *  an inner product and a two-vector linear combination.
 *
 *  Tensors follow the field layout `(*batch, *spatial, C)`: `nall` is the
 *  number of leading (batch + spatial) dimensions, and `size` / `stride`
 *  are `nall + 1` long, the trailing entry describing the channel axis.
 *  Strides are arbitrary, so the same routines serve both the caller's
 *  (possibly strided) buffers and the solver's contiguous scratch.
 *
 **********************************************************************/

/**
 * @brief Inner product `<x, y>`, accumulated in `reduce_t`.
 *
 * The reduction is *deterministic*: the element range is cut into
 * fixed blocks of `GRAIN_SIZE` voxels whose partial sums are summed
 * back in index order, so the result does not depend on how the thread
 * pool happens to schedule the blocks. That matters here because the
 * CG coefficients (`alpha`, `beta`) are computed from these dot
 * products, and a run-to-run wobble in them would make the whole solve
 * non-reproducible.
 */
template <typename reduce_t, typename scalar_t, typename offset_t>
reduce_t dot(
          offset_t   nall,          // number of leading (batch+spatial) dims
    const scalar_t * x,             // (*batch, *spatial, C) tensor
    const scalar_t * y,             // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_x,      // [*batch, *spatial, C] vector
    const offset_t * stride_y       // [*batch, *spatial, C] vector
)
{
    const offset_t numel = prod(size, static_cast<size_t>(nall));
    if (numel <= 0) return static_cast<reduce_t>(0);

    const offset_t nc  = size[nall];
    const offset_t xsc = stride_x[nall];
    const offset_t ysc = stride_y[nall];

    const int64_t block   = GRAIN_SIZE;
    const int64_t nblocks = (static_cast<int64_t>(numel) + block - 1) / block;

    std::vector<reduce_t> partial(static_cast<size_t>(nblocks),
                                  static_cast<reduce_t>(0));

    parallel_for(0, nblocks, 1, [&](long bstart, long bend) {
    for (int64_t b = bstart; b < bend; ++b)
    {
        const offset_t i0 = static_cast<offset_t>(b * block);
        const offset_t i1 = static_cast<offset_t>(
            (b + 1) * block < static_cast<int64_t>(numel)
                ? (b + 1) * block : static_cast<int64_t>(numel));

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t i = i0; i < i1; ++i)
        {
            const offset_t xo = index2offset(i, nall, size, stride_x);
            const offset_t yo = index2offset(i, nall, size, stride_y);
            for (offset_t c = 0; c < nc; ++c)
                acc += static_cast<reduce_t>(x[xo + c * xsc])
                     * static_cast<reduce_t>(y[yo + c * ysc]);
        }
        partial[static_cast<size_t>(b)] = acc;
    }});

    reduce_t total = static_cast<reduce_t>(0);
    for (int64_t b = 0; b < nblocks; ++b)
        total += partial[static_cast<size_t>(b)];
    return total;
}

/**
 * @brief In-place linear combination `y = a * x + b * y`.
 *
 * The single primitive covers every vector update CG performs:
 *   - `b == 0`          : scaled copy      (`p  = z`,        a=1, b=0)
 *   - `b == 1`          : axpy             (`x += alpha*p`,  a=alpha, b=1)
 *   - `a == 1`          : x-plus-scaled-y  (`p  = z + beta*p`, b=beta)
 *   - `a == 1, b == -1` : residual         (`r  = g - A x`)
 * Writing them all through one routine keeps a single traversal of the
 * strided index arithmetic instead of one per update flavour.
 */
template <typename reduce_t, typename scalar_t, typename offset_t>
void axpby_(
          offset_t   nall,          // number of leading (batch+spatial) dims
          scalar_t * y,             // (*batch, *spatial, C) tensor, in/out
    const scalar_t * x,             // (*batch, *spatial, C) tensor
          reduce_t   a,             // weight of `x`
          reduce_t   b,             // weight of `y`
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_y,      // [*batch, *spatial, C] vector
    const offset_t * stride_x       // [*batch, *spatial, C] vector
)
{
    const offset_t numel = prod(size, static_cast<size_t>(nall));
    if (numel <= 0) return;

    const offset_t nc  = size[nall];
    const offset_t ysc = stride_y[nall];
    const offset_t xsc = stride_x[nall];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = static_cast<offset_t>(start);
         i < static_cast<offset_t>(end); ++i)
    {
        const offset_t yo = index2offset(i, nall, size, stride_y);
        const offset_t xo = index2offset(i, nall, size, stride_x);
        // `b == 0` is a plain (scaled) copy: skip reading `y` altogether so
        // that an uninitialised destination cannot poison the result with a
        // `0 * NaN`.
        if (b == static_cast<reduce_t>(0))
            for (offset_t c = 0; c < nc; ++c)
                y[yo + c * ysc] = static_cast<scalar_t>(
                    a * static_cast<reduce_t>(x[xo + c * xsc]));
        else
            for (offset_t c = 0; c < nc; ++c)
            {
                scalar_t & dst = y[yo + c * ysc];
                dst = static_cast<scalar_t>(
                    a * static_cast<reduce_t>(x[xo + c * xsc])
                  + b * static_cast<reduce_t>(dst));
            }
    }});
}

FF_NAMESPACE_END(solve_field)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)

#endif // FF_SOLVE_FIELD_CPU
