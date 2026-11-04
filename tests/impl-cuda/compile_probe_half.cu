/**
 * nvcc compile probe for core/half.h -- the NATIVE path.
 *
 * There is no GPU in CI, so like every other probe under tests/impl-cuda/ this
 * is validated by nvcc accepting it, not by running it. What it proves:
 *
 *   - `ff::half` / `ff::bfloat16` alias the native CUDA types under nvcc, so
 *     device code gets hardware half math and the DLPack layout is unchanged;
 *   - <cuda_bf16.h> is actually reachable (cuda_switch.h already pulls in
 *     <cuda_fp16.h>; bf16 is the one core/half.h has to add itself);
 *   - `compute_type<T>` resolves in device code;
 *   - a __global__ kernel can take these types by value and do arithmetic.
 *
 * Compiled by `make test-impl-cuda` (nvcc, -std=c++14 -O1 in CI).
 */
#include <fastfields/core/half.h>

#include <type_traits>

#ifndef FF_CUDA_HALF
#error "core/half.h must select the native CUDA types under nvcc"
#endif

static_assert(std::is_same<ff::half, __half>::value,
              "ff::half must BE __half under nvcc");
static_assert(std::is_same<ff::bfloat16, __nv_bfloat16>::value,
              "ff::bfloat16 must BE __nv_bfloat16 under nvcc");
static_assert(sizeof(ff::half) == 2 && sizeof(ff::bfloat16) == 2, "16-bit");
static_assert(std::is_trivially_copyable<ff::half>::value,
              "must be passable by value into a kernel");
static_assert(std::is_same<ff::compute_type<ff::half>::type, float>::value,
              "half accumulates in float");
static_assert(std::is_same<ff::compute_type<ff::bfloat16>::type, float>::value,
              "bfloat16 accumulates in float");
static_assert(std::is_same<ff::compute_type<double>::type, double>::value,
              "double accumulates in double");

// The shape a real kernel would take: 16-bit storage, float accumulator.
template <typename scalar_t>
__global__ void probe_axpy(scalar_t * out, const scalar_t * inp, int n)
{
    typedef typename ff::compute_type<scalar_t>::type acc_t;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc_t acc = static_cast<acc_t>(0);
    for (int k = 0; k < 4; ++k)
        acc += static_cast<acc_t>(inp[i]) * static_cast<acc_t>(k);
    out[i] = static_cast<scalar_t>(acc);
}

template __global__ void probe_axpy<ff::half>(ff::half *, const ff::half *,
                                              int);
template __global__ void probe_axpy<ff::bfloat16>(ff::bfloat16 *,
                                                  const ff::bfloat16 *, int);
template __global__ void probe_axpy<float>(float *, const float *, int);
