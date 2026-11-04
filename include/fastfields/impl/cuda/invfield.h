#pragma once
/* CUDA loops for displacement-field inversion ("invfield").
 *
 * The per-simplex and per-voxel math is in impl/kernels/invfield.h, which also
 * explains the algorithm and why it runs in two passes. This header owns only
 * the device kernels and the host launcher, and is the structural mirror of
 * impl/cpu/invfield.h: three launches over, in order,
 *
 *   1. every voxel      -- set the owner buffer to "unclaimed";
 *   2. every simplex    -- rasterise it and claim the voxels it covers;
 *   3. every voxel      -- read the winning simplex back and interpolate.
 *
 * Launch 2 must see every write launch 1 made and launch 3 every write launch
 * 2 made, which is exactly what the stream gives: the three are enqueued on
 * the caller's stream in order, and kernels on one stream do not overlap. The
 * only shared write in launch 2 is the atomic minimum on the owner buffer, and
 * in launch 3 each voxel has one writer -- so the op has no race within a
 * launch either, and its result is identical to the CPU backend's.
 *
 * Unlike every other CUDA launcher here, the kernels are NOT templated on a
 * compile-time `nbatch`. They do not need to be: the only per-voxel index
 * arrays they build are over the `ndim` spatial axes, which is a compile-time
 * extent already, and the batch element is just a linear index turned into
 * three offsets. So there is no FF_INVFIELD_MAX_NBATCH, no switch over the
 * runtime value, and no instantiation count that grows with it.
 */
#include <fastfields/core/cuda_switch.h>
#include <fastfields/impl/kernels/invfield.h>
#include <fastfields/impl/kernels/batch.h>
#include "utils.h"
#include "launch.h"   // FF_CUDA_LAUNCH -- the only checked kernel launch
#include <cstdint>
#include <stdexcept>

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(invfield)

/* Pass 1: no simplex owns anything yet. */
template <int ndim, typename offset_t>
FF_CUGLOB
void kernel_reset(
          int64_t  * owner,         // (*batch, *spatial)
          int        nbatch,
    const offset_t * size,          // [nbatch + ndim]
    const offset_t * stride_own,    // [nbatch + ndim]
          int64_t    numel,         // nelem * nvox
          int64_t    nvox)
{
    int64_t index = threadIdx.x + blockIdx.x * (int64_t)blockDim.x;
    const int64_t step = (int64_t)blockDim.x * gridDim.x;
    for (; index < numel; index += step) {
        const offset_t b = static_cast<offset_t>(index / nvox);
        const offset_t v = static_cast<offset_t>(index % nvox);
        offset_t sub[ndim];
        unravel<ndim>(v, size + nbatch, sub);
        int64_t * base = owner + index2offset(b, nbatch, size, stride_own);
        base[sub2offset<ndim>(sub, stride_own + nbatch)] = unclaimed();
    }
}

/* Pass 2: rasterise each deformed simplex and claim the voxels it covers. */
template <int ndim, typename scalar_t, typename offset_t, typename reduce_t>
FF_CUGLOB
void kernel_claim(
          int64_t  * owner,         // (*batch, *spatial)
    const scalar_t * disp,          // (*batch, *spatial, ndim)
          int        nbatch,
    const offset_t * size,          // [nbatch + ndim]
    const offset_t * stride_own,    // [nbatch + ndim]
    const offset_t * stride_disp,   // [nbatch + ndim + 1]
          int64_t    numel,         // nelem * nsimplex
          int64_t    nsimplex)
{
    int64_t index = threadIdx.x + blockIdx.x * (int64_t)blockDim.x;
    const int64_t step = (int64_t)blockDim.x * gridDim.x;
    const offset_t sdisp_comp = stride_disp[nbatch + ndim];
    for (; index < numel; index += step) {
        const offset_t b   = static_cast<offset_t>(index / nsimplex);
        const int64_t  sid = index % nsimplex;
        const scalar_t * d_base =
            disp + index2offset(b, nbatch, size, stride_disp);
        int64_t * o_base =
            owner + index2offset(b, nbatch, size, stride_own);

        reduce_t vsrc[ndim + 1][ndim], vdst[ndim + 1][ndim];
        corners<ndim>(vsrc, vdst, sid, d_base, size + nbatch,
                      stride_disp + nbatch, sdisp_comp);
        Rasteriser<ndim>::claim(o_base, sid, vdst, size + nbatch,
                                stride_own + nbatch);
    }
}

/* Pass 3: one writer per voxel. */
template <int ndim, typename scalar_t, typename offset_t, typename reduce_t>
FF_CUGLOB
void kernel_resolve(
          scalar_t * out,           // (*batch, *spatial, ndim)
          int64_t  * owner,         // (*batch, *spatial)
    const scalar_t * disp,          // (*batch, *spatial, ndim)
          int        nbatch,
    const offset_t * size,          // [nbatch + ndim]
    const offset_t * stride_out,    // [nbatch + ndim + 1]
    const offset_t * stride_own,    // [nbatch + ndim]
    const offset_t * stride_disp,   // [nbatch + ndim + 1]
          int64_t    numel,         // nelem * nvox
          int64_t    nvox)
{
    int64_t index = threadIdx.x + blockIdx.x * (int64_t)blockDim.x;
    const int64_t step = (int64_t)blockDim.x * gridDim.x;
    const offset_t sout_comp  = stride_out [nbatch + ndim];
    const offset_t sdisp_comp = stride_disp[nbatch + ndim];
    for (; index < numel; index += step) {
        const offset_t b = static_cast<offset_t>(index / nvox);
        const offset_t v = static_cast<offset_t>(index % nvox);
        offset_t sub[ndim];
        unravel<ndim>(v, size + nbatch, sub);

        const scalar_t * d_base =
            disp + index2offset(b, nbatch, size, stride_disp);
        int64_t * o_base =
            owner + index2offset(b, nbatch, size, stride_own);
        scalar_t * out_base =
            out + index2offset(b, nbatch, size, stride_out);

        int64_t  * ow  = o_base   + sub2offset<ndim>(sub, stride_own + nbatch);
        scalar_t * val = out_base + sub2offset<ndim>(sub, stride_out + nbatch);

        reduce_t srcp[ndim];
        const int64_t sid = *ow;
        const bool got = (sid != unclaimed())
            && source<ndim>(srcp, sid, sub, d_base, size + nbatch,
                            stride_disp + nbatch, sdisp_comp);
        if (got) {
            for (int d = 0; d < ndim; ++d)
                val[d * sout_comp] = static_cast<scalar_t>(
                    srcp[d] - static_cast<reduce_t>(sub[d]));
        } else {
            *ow = uncovered();
            for (int d = 0; d < ndim; ++d)
                val[d * sout_comp] = not_a_number(static_cast<scalar_t>(0));
        }
    }
}

/* Host launcher: upload the shape and stride vectors, enqueue the three passes
 * on `stream`, then free.
 *
 * `size` and the three `stride_*` arrays are host arrays; `out`, `owner` and
 * `disp` are device memory. The frees run while the kernels may still be in
 * flight, which is safe only because `cudaFree` synchronises the device --
 * the same bargain every launcher in this directory makes, stated here so the
 * next person does not have to rediscover it. */
template <int ndim, typename scalar_t, typename offset_t, typename reduce_t>
FF_CUHOST
void loop(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, ndim)
          int64_t  * owner,         // (*batch, *spatial)
    const scalar_t * disp,          // (*batch, *spatial, ndim)
    const offset_t * size,          // [nbatch + ndim]
    const offset_t * stride_out,    // [nbatch + ndim + 1]
    const offset_t * stride_own,    // [nbatch + ndim]
    const offset_t * stride_disp,   // [nbatch + ndim + 1]
          intptr_t   stream = 0)
{
    const offset_t nall = nbatch + ndim;
    offset_t * d_size = nullptr, * d_so = nullptr;
    offset_t * d_sw   = nullptr, * d_sd = nullptr;

    // prod() is in impl/kernels/utils.h, every function of which is
    // FF_CUHOSTDEV -- see the contract note at the top of that header; a
    // __device__-only helper called from this template would compile, link
    // and then exit(1) at run time (fastfields-lib#150).
    int64_t nvox  = 1;
    int64_t ncell = 1;
    for (int d = 0; d < ndim; ++d) {
        nvox  *= static_cast<int64_t>(size[nbatch + d]);
        ncell *= static_cast<int64_t>(size[nbatch + d]) - 1;
    }
    const int64_t nelem    = static_cast<int64_t>(prod(size, nbatch));
    const int64_t nsimplex = ncell * Cell<ndim>::simplices;
    if (nvox <= 0 || nelem <= 0) return;

    try
    {
        d_size = copyToDevice(size,        nall);
        d_so   = copyToDevice(stride_out,  static_cast<offset_t>(nall + 1));
        d_sw   = copyToDevice(stride_own,  nall);
        d_sd   = copyToDevice(stride_disp, static_cast<offset_t>(nall + 1));

        cudaStream_t s = (cudaStream_t)(std::intptr_t)stream;
        const int threads = CUDA_NUM_THREADS;
        const int nb      = static_cast<int>(nbatch);

        FF_CUDA_LAUNCH(
            (kernel_reset<ndim, offset_t>),
            GET_BLOCKS(nelem * nvox), threads, 0, s,
            owner, nb, d_size, d_sw, nelem * nvox, nvox);

        if (nsimplex > 0)
            FF_CUDA_LAUNCH(
                (kernel_claim<ndim, scalar_t, offset_t, reduce_t>),
                GET_BLOCKS(nelem * nsimplex), threads, 0, s,
                owner, disp, nb, d_size, d_sw, d_sd,
                nelem * nsimplex, nsimplex);

        FF_CUDA_LAUNCH(
            (kernel_resolve<ndim, scalar_t, offset_t, reduce_t>),
            GET_BLOCKS(nelem * nvox), threads, 0, s,
            out, owner, disp, nb, d_size, d_so, d_sw, d_sd,
            nelem * nvox, nvox);
    }
    catch (...)
    {
        freeDevice(d_size, d_so, d_sw, d_sd);
        throw;
    }
    freeDevice(d_size, d_so, d_sw, d_sd);
}

FF_NAMESPACE_END(invfield)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
