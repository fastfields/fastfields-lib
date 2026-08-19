#ifndef FF_SPLINC_CUDA
#define FF_SPLINC_CUDA
// Teeny-based CUDA splinc impl -- the device mirror of the CPU launcher
// (fastfields-cpu-impl/splinc.h). Same math, same representation:
//
//   * PER-LINE 1-D spline prefilter (causal/anticausal IIR pole recursion) along
//     the LAST axis, batched over every leading axis. teeny's peel replaces the
//     hand-written index2offset batch plumbing: an `anyrank` over (*batch, n)
//     hands out each rank-1 line as a view whose data pointer already has the
//     (arbitrarily strided) batch offset folded in, so the sweep kernel is
//     called unchanged on the raw pointer + size + stride -- the SAME shared
//     `splinc::filter` the CPU body uses. NO host precompute, NO atomics (each
//     line is independent).
//
// Device port vs. the CPU version:
//   * the batch loop is a `__global__` grid-stride loop over the lines instead
//     of parallel_for; the tensor is wrapped as a DEVICE-PASSABLE teeny anyrank
//     carrier (`as_anyrank<TNY_MAX_RANK, storage::gpu_view>(..., copy_meta)` --
//     the shape/stride travel INLINE with the carrier, so it is trivially
//     copyable and passes into the kernel BY VALUE; no separate device copy of
//     shape/stride);
//   * the (<= 3) filter poles travel by value in a tiny POD (poles_dev).
#include <teeny/teeny.h>
#include <cstdint>
#include "fastfields/core/cuda_switch.h"
#include "fastfields/impl/kernels/splinc.h"
#include "utils.h"                    // GET_BLOCKS / CUDA_NUM_THREADS

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(splinc)

// Trivially-copyable carrier for the (<= 3) filter poles, passed into the kernel
// BY VALUE (no device copy needed for such a tiny array).
template <typename reduce_t, int npoles>
struct poles_dev { reduce_t p[npoles]; };

// One spline line per (grid-stride) iteration: peel the rank-1 line with
// peel_front_at<-1> (device-safe, _TNY_API) and run the shared per-line filter.
template <int npoles, bound::type B, typename scalar_t, typename offset_t,
          typename reduce_t, class AT>
CUGLOB void
_splinc_kernel(AT at, poles_dev<reduce_t, npoles> poles, offset_t nlines)
{
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nlines;
         i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto line = at.template peel_front_at<-1>(i);
        splinc::filter<B, npoles>(line.data(), line.extent(0), line.stride(0), poles.p);
    }
}

// Host launcher: wrap `inp` as a device-passable anyrank carrier, launch the
// grid-stride kernel over the lines on `stream`. `size`/`stride` are host arrays
// of length ndim = nbatch + 1; `_poles` is a host array of length npoles; `inp`
// is device memory.
template <int npoles, bound::type B,
          typename scalar_t, typename offset_t, typename reduce_t>
CUHOST void loop(
          offset_t   nbatch,
          scalar_t * inp,
    const offset_t * size,
    const offset_t * stride,
    const reduce_t * _poles,
          int        stream = 0
)
{
    poles_dev<reduce_t, npoles> poles;
    for (int k = 0; k < npoles; ++k) poles.p[k] = _poles[k];

    const int ndim = static_cast<int>(nbatch) + 1;
    auto at = tny::as_anyrank<TNY_MAX_RANK, tny::storage::gpu_view>(
                  inp, size, stride, ndim, tny::copy_meta);
    const offset_t nlines = at.template size_front<-1>();

    cudaStream_t s = reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream));
    const int blocks  = GET_BLOCKS(nlines);
    const int threads = CUDA_NUM_THREADS;
    _splinc_kernel<npoles, B, scalar_t, offset_t, reduce_t>
        <<<blocks, threads, 0, s>>>(at, poles, nlines);
}

FF_NAMESPACE_END(splinc)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_SPLINC_CUDA
