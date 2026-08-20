#pragma once
/* TODO
 * - check if using an inner loop across batch elements is more efficient
 *   (we currently use an outer loop, so we recompute indices many times)
 */

#include <fastfields/core/cuda_switch.h>
#include <fastfields/core/spline.h>
#include <fastfields/core/bounds.h>
#include <fastfields/core/batch.h>
#include <fastfields/impl/kernels/resize.h>
#include "utils.h"
#include <cstdint>
#include <stdexcept>

using namespace std;
FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(resize)

// Largest number of batch dimensions the CUDA launcher instantiates. The
// device kernel is templated on a compile-time `nbatch`, so the host launcher
// dispatches the runtime value to a static instantiation.
#ifndef FF_RESIZE_MAX_NBATCH
#define FF_RESIZE_MAX_NBATCH 1
#endif

template <int nbatch, int ndim,
          typename scalar_t, typename offset_t, typename reduce_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
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
    constexpr int nall = ndim + nbatch;

    // copy vectors to the stack
    reduce_t scale      [ndim]; fillfrom<ndim>(scale,      _scale);
    offset_t size_out   [nall]; fillfrom<nall>(size_out,   _size_out);
    offset_t size_inp   [nall]; fillfrom<nall>(size_inp,   _size_inp);
    offset_t stride_out [nall]; fillfrom<nall>(stride_out, _stride_out);
    offset_t stride_inp [nall]; fillfrom<nall>(stride_inp, _stride_inp);

    offset_t numel = prod<nall>(size_out);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_nd<ndim,nall>(i, size_out, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size_out, stride_out);

        Multiscale<ndim, IX, BX, IY, BY, IZ, BZ>::resize(
            out + out_offset, inp + inp_offset,
            loc, size_inp + nbatch, stride_inp + nbatch,
            scale, shift);
    }
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

    offset_t numel = prod<nall>(size_out);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_nd<ndim,nall>(i, size_out, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size_out, stride_out);

        Multiscale<ndim>::resize(
            out + out_offset, inp + inp_offset,
            loc, size_inp + nbatch, stride_inp + nbatch,
            order, bnd, scale, shift);
    }
}

// Host launcher: allocate device copies of the scale/shape/stride vectors,
// launch the device `kernel` over the output elements on `stream`, then free.
// `scale` has length ndim; the shape/stride vectors have length nall =
// ndim + nbatch (host arrays). `out`/`inp` are device memory.
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

#       define FF_RESIZE_LAUNCH(NB)                                          \
            kernel<NB, ndim, scalar_t, offset_t, reduce_t,                  \
                   IX, BX, IY, BY, IZ, BZ>                                  \
                <<<blocks, threads, 0, s>>>(                               \
                    out, inp, shift, d_scale, d_so, d_si, d_to, d_ti)

        switch (nbatch)
        {
            // Cases run 0..FF_RESIZE_MAX_NBATCH; each `case` is what actually
            // instantiates a static-nbatch kernel, so the range is kept small
            // to bound nvcc compile time (spline x bound x ndim x dtype x
            // offset is already a large instantiation matrix).
            case 0: FF_RESIZE_LAUNCH(0); break;
            case 1: FF_RESIZE_LAUNCH(1); break;
            default:
                throw std::logic_error(
                    "resize: nbatch too large for CUDA launcher");
        }
#       undef FF_RESIZE_LAUNCH
    }
    catch (...)
    {
        freeDevice(d_scale, d_so, d_si, d_to, d_ti);
        throw;
    }
    freeDevice(d_scale, d_so, d_si, d_to, d_ti);
}

FF_NAMESPACE_END(resize)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
