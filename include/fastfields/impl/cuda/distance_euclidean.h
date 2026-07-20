#pragma once
#include "kernels/cuda_switch.h"
#include "kernels/distance.h"
#include "kernels/batch.h"
#include "utils.h"
#include <exception>

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_e)

// CUDA kernel
template <typename scalar_t, typename offset_t>
CUGLOB void dt_kernel(
          offset_t   ndim   ,   // number of dimensions
          scalar_t * f      ,   // pointer to data [*batch, n]
          char     * buf    ,   // buffer (n*(offset_t + 2 * scalar_t))
          scalar_t   w      ,   // pixel spacing
    const offset_t * size   ,   // [ndim] data shape   == (*batch, n)
    const offset_t * stride )   // [ndim] data strides
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t nbatch = ndim - 1;

    offset_t n = size[nbatch];
    offset_t s = stride[nbatch];
    w *= w; // square spacing

    offset_t stride_buf = blockDim.x * gridDim.x;
    offset_t * v = reinterpret_cast<offset_t *>(buf);
    scalar_t * z = reinterpret_cast<scalar_t *>(buf
                 + stride_buf * n * sizeof(offset_t));
    scalar_t * d = reinterpret_cast<scalar_t *>(buf
                 + stride_buf * n * (sizeof(offset_t) + sizeof(scalar_t)));
    v += index;
    z += index;
    d += index;

    offset_t numel = prod(size, nbatch);
    for (offset_t i=index; index < numel; index += stride_buf, i=index)
    {
        offset_t offset = index2offset(i, nbatch, size, stride);
        kernel(f + offset, v, z, d, w, n, s, stride_buf);
    }
}

// Templated entrypoint that launches the CUDA kernel
template <typename scalar_t, typename offset_t>
CUHOST void dt(
          offset_t   ndim     ,     // number of dimensions
          scalar_t * f        ,     // pointer to data [*batch, n]
          scalar_t   w        ,     // pixel spacing
    const offset_t * size     ,     // [ndim] data shape   == (*batch, n)
    const offset_t * stride   )     // [ndim] data strides
{
    char     * buffer        = nullptr;
    offset_t * size_device   = nullptr;
    offset_t * stride_device = nullptr;

    try
    {
        offset_t nbatch      = ndim - 1;
        offset_t vector_size = size[nbatch];
        offset_t batch_size  = prod(size, nbatch);
        offset_t buffer_size = vector_size * (sizeof(offset_t) + 2*sizeof(scalar_t));
        buffer        = allocDevice<char>(buffer_size);
        size_device   = copyToDevice(size,   ndim);
        stride_device = copyToDevice(stride, ndim);
        dt_kernel<scalar_t, offset_t>
            <<<GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0>>>
            (ndim, f, buffer, w, size_device, stride_device);
    }
    catch (const std::exception &exc)
    {
        freeDevice(buffer, size_device, stride_device);
        throw exc;
    }
    freeDevice(buffer, size_device, stride_device);
}

FF_NAMESPACE_END(distance_e)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
