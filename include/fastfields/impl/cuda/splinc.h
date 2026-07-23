#include "kernels/cuda_switch.h"
#include "kernels/splinc.h"
#include "kernels/bounds.h"
#include "kernels/batch.h"
#include "utils.h"
#include <cstdint>
#include <stdexcept>

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(splinc)

// Largest number of batch dimensions the CUDA launcher instantiates.
// The device kernel is templated on a compile-time `nbatch`, so the host
// launcher dispatches the runtime value to a static instantiation.
#ifndef FF_SPLINC_MAX_NBATCH
#define FF_SPLINC_MAX_NBATCH 5
#endif

template <int nbatch, int npoles, bound::type B,
          typename scalar_t, typename offset_t, typename reduce_t>
CUGLOB
void kernel(
    scalar_t * inp,
    const offset_t * _size,
    const offset_t * _stride,
    const reduce_t * _poles)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;

    constexpr int ndim = nbatch + 1;
    reduce_t poles  [npoles];  fillfrom<npoles>(poles, _poles);
    offset_t size   [ndim];    fillfrom<ndim>(size,    _size);
    offset_t stride [ndim];    fillfrom<ndim>(stride,  _stride);

    offset_t numel = prod<nbatch>(size);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t offset = index2offset<nbatch>(i, size, stride);
        splinc::filter<B,npoles>(
            inp + offset, size[nbatch], stride[nbatch], poles);
    }
}

// Host launcher: allocate device copies of the shape/stride/poles vectors,
// launch the device `kernel` over the batch elements on `stream`, then free.
// `size`/`stride` are host arrays of length ndim = nbatch + 1; `poles` is a
// host array of length npoles; `inp` is device memory.
template <int npoles, bound::type B,
          typename scalar_t, typename offset_t, typename reduce_t>
CUHOST
void loop(
          offset_t   nbatch,
          scalar_t * inp,
    const offset_t * size,
    const offset_t * stride,
    const reduce_t * poles,
          int        stream = 0)
{
    const offset_t ndim = nbatch + 1;
    offset_t * d_size   = nullptr;
    offset_t * d_stride = nullptr;
    reduce_t * d_poles  = nullptr;

    try
    {
        d_size   = copyToDevice(size,   ndim);
        d_stride = copyToDevice(stride, ndim);
        d_poles  = copyToDevice(poles,  static_cast<offset_t>(npoles));

        offset_t numel = 1;
        for (offset_t d = 0; d < nbatch; ++d) numel *= size[d];

        cudaStream_t s = (cudaStream_t)(std::intptr_t)stream;
        const int blocks  = GET_BLOCKS(numel);
        const int threads = CUDA_NUM_THREADS;

#       define FF_SPLINC_LAUNCH(NB)                                          \
            kernel<NB, npoles, B, scalar_t, offset_t, reduce_t>             \
                <<<blocks, threads, 0, s>>>(inp, d_size, d_stride, d_poles)

        switch (nbatch)
        {
            case 0: FF_SPLINC_LAUNCH(0); break;
            case 1: FF_SPLINC_LAUNCH(1); break;
            case 2: FF_SPLINC_LAUNCH(2); break;
            case 3: FF_SPLINC_LAUNCH(3); break;
            case 4: FF_SPLINC_LAUNCH(4); break;
            case 5: FF_SPLINC_LAUNCH(5); break;
            default:
                throw std::logic_error(
                    "splinc: nbatch too large for CUDA launcher");
        }
#       undef FF_SPLINC_LAUNCH
    }
    catch (...)
    {
        freeDevice(d_size, d_stride, d_poles);
        throw;
    }
    freeDevice(d_size, d_stride, d_poles);
}

FF_NAMESPACE_END(splinc)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
