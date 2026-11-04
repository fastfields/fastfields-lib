// Compile-only probe for the inv_flow launcher and its three device kernels.
//
// The real build (`src/lib-cuda/inv_flow.cpp`) instantiates `loop` for every
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
//   * It instantiates the SHARED walker, `simplex::Rasteriser<ndim>::claim`,
//     on its own. That header is what the general explicit-mesh rasteriser
//     (fastfields-lib#111's follow-up) will reuse, and nothing in the library
//     build reaches it except through `inv_flow` -- so without this, a change
//     that kept `inv_flow` compiling while breaking the walker's own
//     signature would go unnoticed until the second op existed.
//
//     (The sibling `compile_probe_mesh.cu` also claims this directory is
//     compiled with `-Werror=shadow=local`. It is not, as of this workflow:
//     `compile-probe-cuda` passes `CXXFLAGS="-std=c++14 -O1"` and nothing
//     else. The code here is shadow-clean regardless.)
//
// Compiled, never linked and never run -- there is no GPU in CI.
//
// What it does NOT buy: anything semantic. The owner election, the atomic
// minimum and the barycentric interval solve are all checked on the CPU
// backend (tests/lib-cpu/test_inv_flow.cpp), which shares every line of the
// math; what cannot be checked anywhere is that the three launches really do
// run in order and that `anyAtomicMin`'s `atomicCAS` loop converges on real
// hardware.

#include <fastfields/impl/cuda/inv_flow.h>

namespace I = ff::cuda::inv_flow;
namespace S = ff::cuda::simplex;

// The SHARED walker, instantiated on its own rather than only through
// `kernel_claim`. impl/kernels/simplex.h knows nothing about grids and is what
// the general explicit-mesh rasteriser (#111's follow-up) will reuse, so it is
// worth type-checking without an op wrapped around it -- a change that keeps
// `inv_flow` compiling but breaks the walker's own signature should fail here.
//
// `claim` is FF_CUDEV (it writes the owner buffer atomically), so this probe
// is itself FF_CUDEV and is reached from an FF_CUGLOB function below, not
// from the host: calling it from host code would be the fastfields-lib#150
// trap in reverse, and nvcc *would* diagnose that one (a non-template caller).
template <int D, typename reduce_t, typename offset_t>
FF_CUDEV static void probe_walker(long long * owner, long long sid,
                                  const reduce_t vert[D+1][D],
                                  const offset_t * size,
                                  const offset_t * stride)
{
    S::Rasteriser<D>::claim(reinterpret_cast<int64_t *>(owner),
                            static_cast<int64_t>(sid), vert, size, stride);
}

template <int D, typename reduce_t, typename offset_t>
FF_CUGLOB void probe_walker_kernel(long long * owner, long long sid,
                                   const offset_t * size,
                                   const offset_t * stride)
{
    reduce_t vert[D+1][D];
    for (int i = 0; i <= D; ++i)
        for (int d = 0; d < D; ++d)
            vert[i][d] = static_cast<reduce_t>(i == d ? 1 : 0);
    probe_walker<D, reduce_t, offset_t>(owner, sid, vert, size, stride);
}

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

const void * ff_compile_probe_inv_flow_kernels(int which)
{
    switch ((which / 3) % 4) {
        case 0:  return probe_kernels<2, float,  int >(which);
        case 1:  return probe_kernels<3, float,  int >(which);
        case 2:  return probe_kernels<2, double, long>(which);
        default: return probe_kernels<3, double, long>(which);
    }
}

const void * ff_compile_probe_simplex_walker(int which)
{
    switch (which % 4) {
        case 0:  return reinterpret_cast<const void *>(
                     &probe_walker_kernel<2, double, int >);
        case 1:  return reinterpret_cast<const void *>(
                     &probe_walker_kernel<3, double, int >);
        case 2:  return reinterpret_cast<const void *>(
                     &probe_walker_kernel<2, double, long>);
        default: return reinterpret_cast<const void *>(
                     &probe_walker_kernel<3, double, long>);
    }
}

// One host entry point per (ndim, scalar_t, offset_t) combination the
// dispatcher can select. Never called.
void ff_compile_probe_inv_flow_loop(
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
