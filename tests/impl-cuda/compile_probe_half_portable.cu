/**
 * nvcc compile probe for core/half.h -- the PORTABLE path, forced under nvcc
 * with FF_PORTABLE_HALF.
 *
 * The native probe only proves the `typedef __half half` aliases resolve. This
 * one compiles the software structs themselves with nvcc, in DEVICE code,
 * which is the part that is easy to get wrong and impossible to catch on a
 * host-only build: every conversion helper has to be FF_CUHOSTDEV, and `memcpy`
 * (which replaced teeny's UB union pun) has to be callable from __device__.
 *
 * That matters beyond the debug switch: it is what lets a host-side reference
 * be diffed against the hardware result on the same GPU, which is the only way
 * half correctness gets tested at all once there IS a GPU in CI.
 *
 * Compiled by `make test-impl-cuda`.
 */
#define FF_PORTABLE_HALF 1
#include <fastfields/core/half.h>

#include <type_traits>

#ifdef FF_CUDA_HALF
#error "FF_PORTABLE_HALF must force the software structs even under nvcc"
#endif

static_assert(!std::is_same<ff::half, __half>::value,
              "the portable half must be its own type, not __half");
static_assert(sizeof(ff::half) == 2 && sizeof(ff::bfloat16) == 2, "16-bit");
static_assert(std::is_trivially_copyable<ff::half>::value,
              "must be passable by value into a kernel");
static_assert(std::is_trivially_default_constructible<ff::half>::value,
              "no NSDMI: __shared__ arrays need a trivial default constructor");

// The software conversions must run in device code.
__global__ void probe_convert(ff::half * oh, ff::bfloat16 * ob,
                              const float * inp, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ff::half h(inp[i]);
    ff::bfloat16 b(inp[i]);
    // the mixed-type overloads that kill the reduce_t/scalar_t ambiguity must
    // resolve on device too
    float acc = static_cast<float>(h) * 2.0f + h * 3.0f;
    oh[i] = ff::half(acc);
    ob[i] = ff::bfloat16(static_cast<float>(b) + acc);
}

// __shared__ needs a trivial default constructor -- the reason this port drops
// teeny's NSDMI. If that regresses, this stops compiling.
__global__ void probe_shared(ff::half * out)
{
    __shared__ ff::half tile[32];
    tile[threadIdx.x] = ff::half(static_cast<float>(threadIdx.x));
    __syncthreads();
    out[threadIdx.x] = tile[31 - threadIdx.x];
}
