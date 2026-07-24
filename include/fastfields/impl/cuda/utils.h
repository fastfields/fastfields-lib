#pragma once
#include "kernels/cuda_switch.h"
#include <cstdint>      // int64_t
#include <new>          // std::bad_alloc
#include <stdexcept>    // std::range_error
#include <limits>       // std::numeric_limits

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

/***********************************************************************
 *                       KERNEL LAUNCH HELPERS                         *
 ***********************************************************************/

// Number of threads per block for CUDA kernels. (Copied from PyTorch)
static constexpr int CUDA_NUM_THREADS = 1024;

// Set the number of blocks for CUDA kernel launches. (Copied from PyTorch)
CUHOST inline int
GET_BLOCKS(
    const int64_t N,
    const int64_t max_threads_per_block = CUDA_NUM_THREADS
)
{
  constexpr int64_t max_int = std::numeric_limits<int>::max();

  // Round up division for positive number that cannot cause integer overflow
  auto block_num = (N - 1) / max_threads_per_block + 1;

  // Guard against overflowing the grid's 32-bit block count. The check must
  // fire when the count is TOO LARGE, not when it is in range.
  if (block_num > max_int)
    throw std::range_error("Can't schedule too many blocks on CUDA device");

  return static_cast<int>(block_num);
}

/***********************************************************************
 *                      TEMPORARY BUFFER CLEANUP                       *
 ***********************************************************************/

template <class... U>
struct _CudaBuffers
{
    CUHOST static inline void freeDevice(U...) {}
    CUHOST static inline void freeHost  (U...) {}
};

template <class U0, class... U>
struct _CudaBuffers<U0, U...>
{
    CUHOST static inline void freeDevice(U0 buffer0, U... buffers)
    {

        _CudaBuffers<U0>::freeDevice(buffer0);
        _CudaBuffers<U...>::freeDevice(buffers...);
    }

    CUHOST static inline void freeHost(U0 buffer0, U... buffers)
    {

        _CudaBuffers<U0>::freeHost(buffer0);
        _CudaBuffers<U...>::freeHost(buffers...);
    }
};

template <class U0>
struct _CudaBuffers<U0>
{
    CUHOST static inline void freeDevice(U0 buffer0)
    {
        if (buffer0)
            cudaFree(static_cast<void*>(buffer0));
    }

    CUHOST static inline void freeHost(U0 buffer0)
    {
        if (buffer0)
            cudaFreeHost(static_cast<void*>(buffer0));
    }
};

template <class... U>
CUHOST inline void freeDevice(U... buffers)
{
    return _CudaBuffers<U...>::freeDevice(buffers...);
}

template <class... U>
CUHOST inline void freeHost(U... buffers)
{
    return _CudaBuffers<U...>::freeHost(buffers...);
}

template <class F>
CUHOST inline void error(F exc, const char * msg)
{
    throw exc(msg);
}

/***********************************************************************
 *                          ALLOCATE                                   *
 ***********************************************************************/

template <class A, class B>
struct ff_is_same { static constexpr bool value = false; };

template <class A>
struct ff_is_same<A,A> { static constexpr bool value = true; };

template <class O, class S>
CUHOST inline O * allocDevice(S size)
{
    O * out = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&out), size * sizeof(O));
    if (err) throw std::bad_alloc();
    return out;
}

template <class O, class S>
CUHOST inline O * allocHost(S size)
{
    O * out = nullptr;
    cudaError_t err = cudaMallocHost(reinterpret_cast<void**>(&out), size * sizeof(O));
    if (err) throw std::bad_alloc();
    return out;
}

/***********************************************************************
 *                       COPY CONTIGUOUS VECTORS                       *
 ***********************************************************************/

template <class O, class I, class S>
CUHOST inline O * copyToDevice(const I * inp, S size, O * out = nullptr)
{
    cudaError_t err;
    constexpr bool needs_tmp = ff_is_same<I,O>::value;
    const S bytesize = size * sizeof(O);

    O * tmp = nullptr;
    O * owntmp = nullptr;
    O * ownout = nullptr;

    try
    {
        // Allocate on device
        if (!out) out = ownout = allocDevice<O>(size);

        // Create converted copy on host if needed
        if (needs_tmp)
        {
            tmp = const_cast<O*>(reinterpret_cast<const O*>(inp));
        }
        else
        {
            tmp = owntmp = allocHost<O>(size);
            const I *i = inp;
            for (O *o = tmp; i != inp + size;)
                *o++ = static_cast<O>(*i++);
        }

        // Copy to device
        err = cudaMemcpy(
            static_cast<void*>(out),
            static_cast<const void *>(tmp),
            bytesize,
            cudaMemcpyHostToDevice
        );
        if (err) throw std::bad_alloc();
    }
    catch (const std::exception &exc)
    {
        freeDevice(ownout);
        freeHost(owntmp);
        throw exc;
    }
    freeHost(owntmp);
    return out;
}

template <class I, class S>
CUHOST inline I * copyToDevice(const I * inp, S size, I * out = nullptr)
{
    return copyToDevice<I,I,S>(inp, size, out);
}


template <class O, class I, class S>
CUHOST inline O * copyToHost(const I * inp, S size, O * out = nullptr)
{
    O * ownout = nullptr;
    try
    {
        if (!out) out = ownout = allocHost<O>(size);

        if (ff_is_same<I,O>::value)
        {
            // Same type: direct device -> host copy.
            cudaError_t err = cudaMemcpy(
                static_cast<void*>(out), static_cast<const void*>(inp),
                size * sizeof(O), cudaMemcpyDeviceToHost);
            if (err) throw std::bad_alloc();
        }
        else
        {
            // Different type: stage the raw device data on the host (as I) and
            // convert on the host. Converting straight from `inp` would
            // dereference device pointers on the host (crash/UB), and doing it
            // before the D2H copy converted the wrong direction.
            I * stage = allocHost<I>(size);
            cudaError_t err = cudaMemcpy(
                static_cast<void*>(stage), static_cast<const void*>(inp),
                size * sizeof(I), cudaMemcpyDeviceToHost);
            if (err) { freeHost(stage); throw std::bad_alloc(); }
            for (S k = 0; k < size; ++k)
                out[k] = static_cast<O>(stage[k]);
            freeHost(stage);
        }
    }
    catch (const std::exception &exc)
    {
        freeHost(ownout);
        throw exc;
    }
    return out;
}

template <class I, class S>
CUHOST inline I * copyToHost(const I * inp, S size, I * out = nullptr)
{
    return copyToHost<I,I,S>(inp, size, out);
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
