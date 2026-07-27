#pragma once
#include <teeny/teeny.h>
#include "kernels/cuda_switch.h"
#include "kernels/distance.h"
#include "utils.h"
#include <exception>
#include <cstdint>

// Teeny-based CUDA L1 distance transform: the batch/line plumbing is a teeny
// anyrank peel (mirror of the CPU `distance_l1` impl and the other teeny CUDA
// launchers) instead of the hand-written index2offset + copyToDevice of the
// shape/stride arrays -- the device-passable carrier inlines them, so the host
// no longer copies/frees them. The single-line L1 sweep kernel is unchanged.

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_l1)

// One line per (grid-stride) iteration: peel the rank-1 line with
// peel_front_at<-1> (device-safe, _TNY_API) and run the shared per-line sweep.
template <typename scalar_t, typename offset_t, class AF>
CUGLOB void dt_kernel(AF af, scalar_t w, offset_t nlines)
{
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nlines;
         i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto line = af.template peel_front_at<-1>(i);
        kernel(line.data(), line.extent(0), line.stride(0), w);
    }
}

// Host launcher: wrap `f` as a device-passable anyrank carrier and launch the
// grid-stride kernel over the lines (all leading axes batched). `size`/`stride`
// are host arrays of length ndim = nbatch + 1; `f` is device memory.
template <typename scalar_t, typename offset_t>
CUHOST void dt(
          offset_t   ndim     ,     // number of dimensions (== nbatch + 1)
          scalar_t * f        ,     // pointer to data [*batch, n]
          scalar_t   w        ,     // pixel spacing
    const offset_t * size     ,     // [ndim] data shape   == (*batch, n)
    const offset_t * stride   ,     // [ndim] data strides
          int        stream = 0)
{
    auto af = tny::as_anyrank<TNY_MAX_RANK, tny::storage::gpu_view>(
                  f, size, stride, static_cast<int>(ndim), tny::copy_meta);
    const offset_t nlines = af.template size_front<-1>();

    cudaStream_t s = reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream));
    dt_kernel<scalar_t, offset_t, decltype(af)>
        <<<GET_BLOCKS(nlines), CUDA_NUM_THREADS, 0, s>>>(af, w, nlines);
}

FF_NAMESPACE_END(distance_l1)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
