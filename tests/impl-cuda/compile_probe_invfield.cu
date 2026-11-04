// Compile-only probe for the invfield launcher and its three device kernels.
//
// The real build (`src/lib-cuda/invfield.cpp`) instantiates `loop` for every
// (ndim, scalar_t, offset_t) combination its dispatcher can select, so this TU
// is not what makes the launcher compile. It earns its place for two other
// reasons.
//
//   * It takes the address of each `__global__` kernel, which odr-uses and so
//     type-checks the kernel *body* in a translation unit that does not go
//     through `FF_CUDA_LAUNCH`. A launch site and a kernel signature can drift
//     apart in a way that only one of the two notices; checking both
//     independently is the point.
//
//   * The workflow compiles this directory with `-Werror=shadow=local`, which
//     the library build does not.
//
// Compiled, never linked and never run -- there is no GPU in CI.
//
// What it does NOT buy: anything semantic. The owner election, the atomic
// minimum and the barycentric interval solve are all checked on the CPU
// backend (tests/lib-cpu/test_invfield.cpp), which shares every line of the
// math; what cannot be checked anywhere is that the three launches really do
// run in order and that `anyAtomicMin`'s `atomicCAS` loop converges on real
// hardware.

#include <fastfields/impl/cuda/invfield.h>

namespace I = ff::cuda::invfield;

template <int D, typename scalar_t, typename offset_t>
static void probe_loop(offset_t nbatch, scalar_t * out, long long * owner,
                       const scalar_t * disp, const offset_t * size,
                       const offset_t * stride)
{
    I::loop<D, scalar_t, offset_t, double>(
        nbatch, out, reinterpret_cast<int64_t *>(owner), disp,
        size, stride, stride, stride, /*stream=*/0);
}

// Taking the address of a `__global__` specialization instantiates its body.
// The pointers are discarded; nothing here is ever launched.
template <int D, typename scalar_t, typename offset_t>
static const void * probe_kernels(int which)
{
    const void * const addrs[3] = {
        reinterpret_cast<const void *>(&I::kernel_reset<D, offset_t>),
        reinterpret_cast<const void *>(
            &I::kernel_claim<D, scalar_t, offset_t, double>),
        reinterpret_cast<const void *>(
            &I::kernel_resolve<D, scalar_t, offset_t, double>),
    };
    return addrs[which % 3];
}

const void * ff_compile_probe_invfield_kernels(int which)
{
    switch ((which / 3) % 4) {
        case 0:  return probe_kernels<2, float,  int >(which);
        case 1:  return probe_kernels<3, float,  int >(which);
        case 2:  return probe_kernels<2, double, long>(which);
        default: return probe_kernels<3, double, long>(which);
    }
}

// One host entry point per (ndim, scalar_t, offset_t) combination the
// dispatcher can select. Never called.
void ff_compile_probe_invfield_loop(
    // 2D / 3D, float, 32-bit offsets
    int nb32, float * o32, long long * w32, const float * d32,
    const int * sz32,
    // 2D / 3D, double, 64-bit offsets
    long nb64, double * o64, long long * w64, const double * d64,
    const long * sz64)
{
    probe_loop<2, float,  int >(nb32, o32, w32, d32, sz32, sz32);
    probe_loop<3, float,  int >(nb32, o32, w32, d32, sz32, sz32);
    probe_loop<2, double, int >(nb32, o64, w64, d64, sz32, sz32);
    probe_loop<3, double, int >(nb32, o64, w64, d64, sz32, sz32);
    probe_loop<2, float,  long>(nb64, o32, w32, d32, sz64, sz64);
    probe_loop<3, float,  long>(nb64, o32, w32, d32, sz64, sz64);
    probe_loop<2, double, long>(nb64, o64, w64, d64, sz64, sz64);
    probe_loop<3, double, long>(nb64, o64, w64, d64, sz64, sz64);
}
