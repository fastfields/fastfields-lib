#pragma once
/* CUDA loops for displacement-field inversion ("inv_flow").
 *
 * The math is one layer down and in two headers:
 * impl/kernels/simplex.h holds the shared simplex walk, the barycentric
 * coordinates and the owner-election protocol (and explains why the op runs in
 * two passes), while impl/kernels/inv_flow.h holds what makes this mesh the
 * implicit one -- the cell stencils and the simplex-index decode.
 *
 * This header owns only the device kernels and the host launcher, and is the
 * structural mirror of impl/cpu/inv_flow.h: three launches over, in order,
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
 * three offsets. So there is no FF_INV_FLOW_MAX_NBATCH, no switch over the
 * runtime value, and no instantiation count that grows with it.
 */
#include <fastfields/core/cuda_switch.h>
#include <fastfields/impl/kernels/inv_flow.h>
#include <fastfields/impl/kernels/simplex.h>
#include <fastfields/impl/kernels/batch.h>
#include "utils.h"
#include "launch.h"   // FF_CUDA_LAUNCH -- the only checked kernel launch
#include <cstdint>
#include <stdexcept>

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(inv_flow)

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
        simplex::unravel<ndim>(v, size + nbatch, sub);
        int64_t * base = owner + index2offset(b, nbatch, size, stride_own);
        base[sub2offset<ndim>(sub, stride_own + nbatch)] =
            simplex::unclaimed();
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

        // `vsrc` is not read here: `corners` computes both domains because
        // the resolve pass needs both, and the dead stores fold away once it
        // is inlined.
        reduce_t vsrc[ndim + 1][ndim], vdst[ndim + 1][ndim];
        corners<ndim>(vsrc, vdst, sid, d_base, size + nbatch,
                      stride_disp + nbatch, sdisp_comp);
        simplex::Rasteriser<ndim>::claim(o_base, sid, vdst, size + nbatch,
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
        simplex::unravel<ndim>(v, size + nbatch, sub);

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
        const bool got = (sid != simplex::unclaimed())
            && source<ndim>(srcp, sid, sub, d_base, size + nbatch,
                            stride_disp + nbatch, sdisp_comp);
        if (got) {
            for (int d = 0; d < ndim; ++d)
                val[d * sout_comp] = static_cast<scalar_t>(
                    srcp[d] - static_cast<reduce_t>(sub[d]));
        } else {
            *ow = simplex::uncovered();
            const scalar_t missing =
                simplex::not_a_number(static_cast<scalar_t>(0));
            for (int d = 0; d < ndim; ++d)
                val[d * sout_comp] = missing;
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
        const int nb = static_cast<int>(nbatch);

        // A smaller block than CUDA_NUM_THREADS, which every other launcher
        // here uses. That constant is 1024 -- the *architectural maximum*,
        // not a tuned value (see the note above it in utils.h) -- and these
        // kernels are the register-hungriest in the tree: `kernel_resolve`
        // holds two (ndim+1) x ndim corner arrays, the ndim+1 barycentric
        // gradients, the coordinates and the result at once, which for the
        // 3D float64 case is a few hundred bytes of live per-thread state.
        // 1024 threads of that does not fit a register file, and the way
        // that shows up is `cudaErrorLaunchOutOfResources` -- a LAUNCH
        // FAILURE, not a slowdown.
        //
        // Shrinking the block is safe here in a way launch.h warns it is not
        // for `distance_euclidean::dt` and `distance_mesh::sdt`: those size a
        // host-side scratch buffer from the launch geometry, whereas nothing
        // in this file does. All three kernels are plain grid-stride loops,
        // so they cover every element at any block size and the grid below is
        // sized from the same number.
        //
        // 256 is chosen for headroom, not measured: there is no GPU in CI to
        // measure on. If this op is ever benchmarked on real hardware, this
        // is the first number to revisit.
        const int threads = 256;

        FF_CUDA_LAUNCH(
            (kernel_reset<ndim, offset_t>),
            GET_BLOCKS(nelem * nvox, threads), threads, 0, s,
            owner, nb, d_size, d_sw, nelem * nvox, nvox);

        if (nsimplex > 0)
            FF_CUDA_LAUNCH(
                (kernel_claim<ndim, scalar_t, offset_t, reduce_t>),
                GET_BLOCKS(nelem * nsimplex, threads), threads, 0, s,
                owner, disp, nb, d_size, d_sw, d_sd,
                nelem * nsimplex, nsimplex);

        FF_CUDA_LAUNCH(
            (kernel_resolve<ndim, scalar_t, offset_t, reduce_t>),
            GET_BLOCKS(nelem * nvox, threads), threads, 0, s,
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

FF_NAMESPACE_END(inv_flow)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
