#pragma once
#include <teeny/teeny.h>
#include "fastfields/core/cuda_switch.h"
#include "fastfields/impl/kernels/distance.h"
#include "utils.h"
#include <exception>

// Teeny-based CUDA squared-Euclidean distance transform (lower-envelope of
// parabolas) along the LAST axis, batched over the leading axes. The batch/line
// plumbing is a teeny anyrank peel (mirror of the CPU `distance_e` impl and the
// other teeny CUDA launchers) instead of index2offset + a device copy of the
// shape/stride arrays -- the device-passable carrier inlines them. The sweep
// kernel and its per-lane scratch (v/z/d, one length-n set per launched lane)
// are unchanged.

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_e)

// One line per (grid-stride) iteration. Each of the stride_buf = (blocks*threads)
// lanes owns a length-n scratch triple (v: offset_t, z & d: scalar_t) at its lane
// offset `index`; the peel gives the line pointer/length/stride on the device.
template <typename scalar_t, typename offset_t, class AF>
CUGLOB void dt_kernel(AF af, char * buf, scalar_t w, offset_t n, offset_t nlines)
{
    const offset_t index      = threadIdx.x + blockIdx.x * blockDim.x;
    const offset_t stride_buf  = static_cast<offset_t>(gridDim.x) * blockDim.x;
    w *= w;   // square spacing

    offset_t * v = reinterpret_cast<offset_t *>(buf) + index;
    scalar_t * z = reinterpret_cast<scalar_t *>(buf
                 + stride_buf * n * sizeof(offset_t)) + index;
    scalar_t * d = reinterpret_cast<scalar_t *>(buf
                 + stride_buf * n * (sizeof(offset_t) + sizeof(scalar_t))) + index;

    for (offset_t i = index; i < nlines; i += stride_buf)
    {
        auto line = af.template peel_front_at<-1>(i);
        kernel(line.data(), v, z, d, w, line.extent(0), line.stride(0), stride_buf);
    }
}

// Host launcher: wrap `f` as a device-passable anyrank carrier, allocate the
// per-lane scratch, and launch the grid-stride kernel over the lines. `size`/
// `stride` are host arrays of length ndim = nbatch + 1; `f` is device memory.
template <typename scalar_t, typename offset_t>
CUHOST void dt(
          offset_t   ndim     ,     // number of dimensions (== nbatch + 1)
          scalar_t * f        ,     // pointer to data [*batch, n]
          scalar_t   w        ,     // pixel spacing
    const offset_t * size     ,     // [ndim] data shape   == (*batch, n)
    const offset_t * stride   ,     // [ndim] data strides
          int        stream = 0)
{
    char * buffer = nullptr;
    try
    {
        const offset_t nbatch      = ndim - 1;
        const offset_t vector_size = size[nbatch];       // n, the transform axis
        auto af = tny::as_anyrank<TNY_MAX_RANK, tny::storage::gpu_view>(
                      f, size, stride, static_cast<int>(ndim), tny::copy_meta);
        const offset_t nlines = af.template size_front<-1>();

        const int num_blocks = GET_BLOCKS(nlines);
        // Each of the stride_buf = (blocks*threads) lanes gets its own length-n
        // scratch, so the buffer scales with the launched thread count -- not just
        // the vector length (missing this factor was a device-side OOB write).
        const offset_t stride_buf  = static_cast<offset_t>(num_blocks) * CUDA_NUM_THREADS;
        const offset_t buffer_size = stride_buf * vector_size
                                   * (sizeof(offset_t) + 2 * sizeof(scalar_t));
        buffer = allocDevice<char>(buffer_size);

        cudaStream_t s = reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream));
        dt_kernel<scalar_t, offset_t, decltype(af)>
            <<<num_blocks, CUDA_NUM_THREADS, 0, s>>>
            (af, buffer, w, vector_size, nlines);
    }
    catch (const std::exception &exc)
    {
        freeDevice(buffer);
        throw exc;
    }
    freeDevice(buffer);
}

FF_NAMESPACE_END(distance_e)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
