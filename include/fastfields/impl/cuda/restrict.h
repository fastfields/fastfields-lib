#pragma once
/* TODO
 * - implement special case (order=1 + scale=2) for dim 2 and 3
 * - check if using an inner loop across batch elements is more efficient
 *   (we currently use an outer loop, so we recompute indices many times)
 */

#include <fastfields/core/cuda_switch.h>
#include <fastfields/core/spline.h>
#include <fastfields/core/bounds.h>
#include <fastfields/core/batch.h>
#include <fastfields/impl/kernels/restrict.h>
#include "utils.h"
#include <cstdint>
#include <stdexcept>

using namespace std;
FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(restrict)

// Largest number of batch dimensions the CUDA launcher instantiates. The
// device kernel is templated on a compile-time `nbatch`, so the host launcher
// dispatches the runtime value to a static instantiation.
#ifndef FF_RESTRICT_MAX_NBATCH
#define FF_RESTRICT_MAX_NBATCH 1
#endif

template <int nbatch, int ndim,
          typename scalar_t, typename offset_t, typename reduce_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY,
          int U=zero>
FF_CUGLOB
void kernel(
    scalar_t * out,                 // (*batch, *shape) tensor
    const scalar_t * inp,           // (*batch, *shape) tensor
    reduce_t shift,
    const reduce_t * _scale,        // [*shape] vector
    const offset_t * _size_out,     // [*batch, *shape] vector
    const offset_t * _size_inp,     // [*batch, *shape] vector
    const offset_t * _stride_out,   // [*batch, *shape] vector
    const offset_t * _stride_inp)   // [*batch, *shape] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;

    using bound_utils_x  = bound::utils<BX>;
    using bound_utils_y  = bound::utils<BY>;
    using bound_utils_z  = bound::utils<BZ>;
    constexpr int nall = ndim + nbatch;
    constexpr int spline_order_x = static_cast<int>(IX);
    constexpr int spline_order_y = static_cast<int>(IY);
    constexpr int spline_order_z = static_cast<int>(IZ);
    constexpr int padding_x = (spline_order_x + 1)/2;
    constexpr int padding_y = (spline_order_y + 1)/2;
    constexpr int padding_z = (spline_order_z + 1)/2;

    // copy vectors to the stack
    reduce_t scale      [ndim]; fillfrom<ndim>(scale,      _scale);
    offset_t size_out   [nall]; fillfrom<nall>(size_out,   _size_out);
    offset_t size_inp   [nall]; fillfrom<nall>(size_inp,   _size_inp);
    offset_t stride_out [nall]; fillfrom<nall>(stride_out, _stride_out);
    offset_t stride_inp [nall]; fillfrom<nall>(stride_inp, _stride_inp);

    offset_t fullsize[nall]; fillfrom<nall>(fullsize, size_out);
    if (ndim > 0) fullsize[nbatch]   += 2 * padding_x;
    if (ndim > 1) fullsize[nbatch+1] += 2 * padding_y;
    if (ndim > 2) fullsize[nbatch+2] += 2 * padding_z;

    offset_t numel = prod<nall>(fullsize);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t inp_offset = index2offset<nbatch>(i, size_out, stride_inp);

        signed char sgn = 1;
        offset_t loc[nall]; index2sub<nall>(i, fullsize, loc);
        offset_t sub[nall]; fillfrom<nall>(sub, loc);
        if (ndim > 0) {
            loc[nbatch]   -= padding_x;
            sgn           *= bound_utils_x::sign(loc[nbatch],  size_out[nbatch]);
            sub[nbatch]    = bound_utils_x::index(loc[nbatch], size_out[nbatch]);
        }
        if (ndim > 1) {
            loc[nbatch+1] -= padding_y;
            sgn           *= bound_utils_y::sign(loc[nbatch+1],  size_out[nbatch+1]);
            sub[nbatch+1]  = bound_utils_y::index(loc[nbatch+1], size_out[nbatch+1]);
        }
        if (ndim > 2) {
            loc[nbatch+2] -= padding_z;
            sgn           *= bound_utils_z::sign(loc[nbatch+2],  size_out[nbatch+2]);
            sub[nbatch+2]  = bound_utils_z::index(loc[nbatch+2], size_out[nbatch+2]);
        }
        if (!sgn) continue;

        offset_t out_offset = sub2offset<nall>(sub, stride_out);

        Multiscale<ndim, U, IX, IY, IZ>::restrict(
            out + out_offset, inp + inp_offset,
            loc + nbatch, size_inp + nbatch, stride_inp + nbatch,
            scale, shift, sgn);
    }
}


template <int nbatch, int ndim,
          typename scalar_t, typename offset_t, typename reduce_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY,
          int U=zero>
FF_CUGLOB
void kernel2(
    scalar_t * out,                // (*batch, *shape) tensor
    const scalar_t * inp,          // (*batch, *shape) tensor
    reduce_t shift,
    const reduce_t * scale,        // [*shape] vector
    const offset_t * size_out,     // [*batch, *shape] vector
    const offset_t * size_inp,     // [*batch, *shape] vector
    const offset_t * stride_out,   // [*batch, *shape] vector
    const offset_t * stride_inp)   // [*batch, *shape] vector
{
    return kernel<nbatch, ndim, scalar_t, offset_t, reduce_t,
                  IX, BX, IY, BY, IZ, BZ, two>
        (out, inp, shift, scale, size_out, size_inp, stride_out, stride_inp);
}

template <int nbatch, int ndim,
          typename scalar_t, typename offset_t, typename reduce_t>
FF_CUGLOB
void kernelnd(
    scalar_t * out,                 // (*batch, *shape) tensor
    const scalar_t * inp,           // (*batch, *shape) tensor
    reduce_t shift,
    const reduce_t * _scale,        // [*shape] vector
    const unsigned char * _order,   // [*shape] vector
    const unsigned char * _bnd,     // [*shape] vector
    const offset_t * _size_out,     // [*batch, *shape] vector
    const offset_t * _size_inp,     // [*batch, *shape] vector
    const offset_t * _stride_out,   // [*batch, *shape] vector
    const offset_t * _stride_inp)   // [*batch, *shape] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr int nall = ndim + nbatch;

    const spline::type * corder = reinterpret_cast<const spline::type *>(_order);
    const bound::type  * cbnd   = reinterpret_cast<const bound::type *>(_bnd);

    // copy vectors to the stack
    reduce_t scale      [ndim]; fillfrom<ndim>(scale,      _scale);
    spline::type order  [ndim]; fillfrom<ndim>(order,      corder);
    bound::type  bnd    [ndim]; fillfrom<ndim>(bnd,        cbnd);
    offset_t size_out   [nall]; fillfrom<nall>(size_out,   _size_out);
    offset_t size_inp   [nall]; fillfrom<nall>(size_inp,   _size_inp);
    offset_t stride_out [nall]; fillfrom<nall>(stride_out, _stride_out);
    offset_t stride_inp [nall]; fillfrom<nall>(stride_inp, _stride_inp);

    offset_t fullsize[nall]; fillfrom<nall>(fullsize, size_out);
    for (int d=0; d < ndim; ++d)
        fullsize[nbatch+d] += 2 * ((static_cast<int>(order[d]) + 1) / 2);

    offset_t numel = prod<nall>(fullsize);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t inp_offset = index2offset<nbatch>(i, size_out, stride_inp);

        signed char sgn = 1;
        offset_t loc[nall]; index2sub<nall>(i, fullsize, loc);
        offset_t sub[nall]; fillfrom<nall>(sub, loc);
        for (int d=0; d < ndim; ++d) {
            loc[nbatch+d]   -= (static_cast<int>(order[d]) + 1) / 2;
            sgn             *= bound::sign(bnd[d], loc[nbatch+d],  size_out[nbatch+d]);
            sub[nbatch+d]    = bound::index(bnd[d], loc[nbatch+d], size_out[nbatch+d]);
        }
        if (!sgn) continue;

        offset_t out_offset = sub2offset<nall>(sub, stride_out);

        Multiscale<ndim>::restrict(
            out + out_offset, inp + inp_offset,
            loc + nbatch, size_inp + nbatch, stride_inp + nbatch,
            order, scale, shift, sgn);
    }
}

// Host launcher: allocate device copies of the scale/shape/stride vectors,
// launch the device `kernel` over the (padded) output elements on `stream`,
// then free. `scale` has length ndim; the shape/stride vectors have length
// nall = ndim + nbatch (host arrays). `out`/`inp` are device memory. The grid
// is sized from the coarse output element count; the kernel's grid-stride loop
// covers the padded range regardless.
template <
    int ndim,
    typename scalar_t,
    typename offset_t,
    typename reduce_t,
    spline::type IX,    bound::type BX,
    spline::type IY=IX, bound::type BY=BX,
    spline::type IZ=IY, bound::type BZ=BY
>
FF_CUHOST
void loop(
          offset_t   nbatch,
          scalar_t * out,
    const scalar_t * inp,
          reduce_t   shift,
    const reduce_t * scale,
    const offset_t * size_out,
    const offset_t * size_inp,
    const offset_t * stride_out,
    const offset_t * stride_inp,
          intptr_t   stream = 0)
{
    const offset_t nall = ndim + nbatch;
    reduce_t * d_scale = nullptr;
    offset_t * d_so = nullptr, * d_si = nullptr, * d_to = nullptr, * d_ti = nullptr;

    try
    {
        d_scale = copyToDevice(scale,      static_cast<offset_t>(ndim));
        d_so    = copyToDevice(size_out,   nall);
        d_si    = copyToDevice(size_inp,   nall);
        d_to    = copyToDevice(stride_out, nall);
        d_ti    = copyToDevice(stride_inp, nall);

        offset_t numel = 1;
        for (offset_t d = 0; d < nall; ++d) numel *= size_out[d];

        cudaStream_t s = (cudaStream_t)(std::intptr_t)stream;
        const int blocks  = GET_BLOCKS(numel);
        const int threads = CUDA_NUM_THREADS;

#       define FF_RESTRICT_LAUNCH(NB)                                        \
            kernel<NB, ndim, scalar_t, offset_t, reduce_t,                  \
                   IX, BX, IY, BY, IZ, BZ>                                  \
                <<<blocks, threads, 0, s>>>(                               \
                    out, inp, shift, d_scale, d_so, d_si, d_to, d_ti)

        switch (nbatch)
        {
            // Cases run 0..FF_RESTRICT_MAX_NBATCH; each `case` is what
            // actually instantiates a static-nbatch kernel, so the range is
            // kept small to bound nvcc compile time.
            case 0: FF_RESTRICT_LAUNCH(0); break;
            case 1: FF_RESTRICT_LAUNCH(1); break;
            default:
                throw std::logic_error(
                    "restrict: nbatch too large for CUDA launcher");
        }
#       undef FF_RESTRICT_LAUNCH
    }
    catch (...)
    {
        freeDevice(d_scale, d_so, d_si, d_to, d_ti);
        throw;
    }
    freeDevice(d_scale, d_so, d_si, d_to, d_ti);
}

FF_NAMESPACE_END(restrict)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
