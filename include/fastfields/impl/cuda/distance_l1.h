#pragma once
#include "fastfields/core/cuda_switch.h"
#include "fastfields/impl/kernels/distance.h"
#include "fastfields/impl/kernels/batch.h"
#include "utils.h"
#include <exception>
#include <cstdint>

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_l1)

// CUDA kernel
template <typename scalar_t, typename offset_t>
FF_CUGLOB void dt_kernel(
          offset_t   ndim   ,   // number of dimensions
          scalar_t * f      ,   // pointer to data [*batch, n]
          scalar_t   w      ,   // pixel spacing
    const offset_t * size   ,   // [ndim] data shape   == (*batch, n)
    const offset_t * stride )   // [ndim] data strides
{
    offset_t cuindex  = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t custride = blockDim.x * gridDim.x;
    offset_t nbatch   = ndim - 1;

    offset_t n = size[nbatch];
    offset_t s = stride[nbatch];

    offset_t numel = prod(size, nbatch);
    for (offset_t i=cuindex; cuindex < numel; cuindex += custride, i=cuindex)
    {
        offset_t offset = index2offset(i, nbatch, size, stride);
        kernel(f + offset, n, s, w);
    }
}

// Templated entrypoint that launches the CUDA kernel
template <typename scalar_t, typename offset_t>
FF_CUHOST void dt(
          offset_t   ndim     ,     // number of dimensions
          scalar_t * f        ,     // pointer to data [*batch, n]
          scalar_t   w        ,     // pixel spacing
    const offset_t * size     ,     // [ndim] data shape   == (*batch, n)
    const offset_t * stride   ,     // [ndim] data strides
          intptr_t   stream = 0)    // CUDA stream (0 == default stream)
{
    offset_t * size_device   = nullptr;
    offset_t * stride_device = nullptr;

    try
    {
        offset_t nbatch      = ndim - 1;
        offset_t vector_size = size[nbatch];
        offset_t batch_size  = prod(size, nbatch);
        cudaStream_t s = (cudaStream_t)(std::intptr_t)stream;
        size_device   = copyToDeviceAsync(size,   ndim, s);
        stride_device = copyToDeviceAsync(stride, ndim, s);
        dt_kernel<scalar_t, offset_t>
            <<<GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, s>>>
            (ndim, f, w, size_device, stride_device);
    }
    catch (const std::exception &exc)
    {
        freeDevice(size_device, stride_device);
        throw exc;
    }
    freeDevice(size_device, stride_device);
}

FF_NAMESPACE_END(distance_l1)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
