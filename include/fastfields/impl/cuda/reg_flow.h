#ifndef FF_REGULARISERS_FLOW_CUDA
#define FF_REGULARISERS_FLOW_CUDA
// Teeny-based CUDA reg_flow impl -- the device mirror of the CPU launcher
// (fastfields-cpu-impl/reg_flow.h). Same math, same representation:
//
//   * ABSOLUTE is POINTWISE: teeny's peel hands each (*batch,*spatial) voxel's
//     rank-1 channel cell to the shared single-voxel kernel (kernels/
//     regularisers/flow). `peel_front_at<-1>` folds the (arbitrarily strided)
//     batch/spatial offset into each cell's pointer -- the same call the CPU body
//     uses, so "CPU works + CUDA compiles" gives real confidence they compute the
//     same thing. NO host precompute of offsets, NO atomics (disjoint writes).
//   * MEMBRANE / BENDING are STENCIL ops: they gather spatial NEIGHBOURS with
//     boundary conditions, so each voxel needs its spatial multi-index `loc` and
//     the spatial size/stride. We peel the (*spatial,C) volume of a batch cell
//     with `peel_front_at<-(ndim+1)>(b)`, decode `loc` within it, offset each base
//     pointer, and call the single-voxel kernel -- exactly the CPU loop body.
//
// Device port vs. the CPU version:
//   * the parallel_for becomes a `__global__` grid-stride loop over the voxels;
//     each tensor is wrapped as a DEVICE-PASSABLE teeny anyrank carrier
//     (`as_anyrank<TNY_MAX_RANK, storage::gpu_view>(..., copy_meta)` -- shape/
//     stride travel INLINE, so the carrier is trivially copyable and passes into
//     the kernel BY VALUE; no separate device copy of shape/stride);
//   * the (small, compile-time-sized) convolution kernel table is built ON THE
//     HOST (identical to the CPU impl) and cudaMemcpy'd to the device;
//   * the spatial size/stride the stencil single-voxel kernels index are copied
//     into a tiny by-value POD (`flow_sp`) passed in the launch -- ndim <= 3.
//   * this teeny launcher handles an ARBITRARY batch rank (the batch offset folds
//     into the peel), unlike the legacy launcher which capped nbatch at 3.
//
// The op ('=','+','-') is threaded through exactly as the CPU `op_apply` does:
// `Op<op,scalar_t,reduce_t>::f` is the function-pointer non-type template arg the
// single-voxel kernels take (the C++17 device path, same as the legacy launcher).
#include <teeny/teeny.h>
#include <cstdint>
#include "kernels/cuda_switch.h"
#include "kernels/bounds.h"
#include "kernels/utils.h"
#include "kernels/regularisers/flow.h"
#include "utils.h"                    // GET_BLOCKS / CUDA_NUM_THREADS / copyToDevice / freeDevice

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_flow)

// Device-passable anyrank carrier over (*batch, *spatial, C). Shape/stride are
// COPIED inline (copy_meta) so the carrier passes into the kernel by value; the
// DATA pointer lives in device memory (storage::gpu_view).
template <typename T, typename offset_t>
static inline auto _any(T* p, const offset_t* size, const offset_t* stride, offset_t nall)
{
    return tny::as_anyrank<TNY_MAX_RANK, tny::storage::gpu_view>(
        p, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);
}

// Tiny by-value spatial metadata (spatial extents + spatial out/inp strides) the
// stencil single-voxel kernels index. ndim <= 3, so passing it by value into the
// kernel is trivial (no device copy of the shape/stride arrays).
template <int ndim, typename offset_t>
struct flow_sp {
    offset_t size[ndim];   // spatial extents  (size[nbatch + d])
    offset_t sout[ndim];   // spatial strides of out
    offset_t sinp[ndim];   // spatial strides of inp
};

//======================================================================
//                              ABSOLUTE  (pointwise)
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, bound::type... BOUND>
CUGLOB void _matvec_absolute_k(AO ao, AI ai, const reduce_t* kernel,
                               offset_t osc, offset_t isc, offset_t nvox)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        auto ic = ai.template peel_front_at<-1>(i);
        Impl::template matvec_absolute<opfunc>(oc.data(), ic.data(), osc, isc, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void matvec_absolute(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_out[nall], isc = stride_inp[nall];

    reduce_t kbuf[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kbuf, absolute, voxel_size);

    auto ao = _any(out, size, stride_out, nall);
    auto ai = _any(inp, size, stride_inp, nall);
    const offset_t nvox = ao.template size_front<-1>();

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_absolute));
    try {
        _matvec_absolute_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), decltype(ai), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, ai, d_kernel, osc, isc, nvox);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _diag_absolute_k(AO ao, const reduce_t* kernel, offset_t sc, offset_t nvox)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        Impl::template diag_absolute<opfunc>(oc.data(), sc, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_absolute(
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t sc = stride[nall];

    reduce_t kbuf[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kbuf, absolute, voxel_size);

    auto ao = _any(out, size, stride, nall);
    const offset_t nvox = ao.template size_front<-1>();

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_absolute));
    try {
        _diag_absolute_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, sc, nvox);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

//======================================================================
//                              MEMBRANE  (stencil)
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, bound::type... BOUND>
CUGLOB void _matvec_membrane_k(AO ao, AI ai, const reduce_t* kernel,
                               flow_sp<ndim, offset_t> sp, offset_t osc, offset_t isc,
                               offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0, io = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d]; io += c * sp.sinp[d];
        }
        Impl::template matvec_membrane<opfunc>(
            vo.data() + oo, vi.data() + io, loc, sp.size, sp.sinp, osc, isc, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void matvec_membrane(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_out[nall], isc = stride_inp[nall];

    reduce_t kbuf[Impl::kernelsize_membrane];
    Impl::make_kernel_membrane(kbuf, absolute, membrane, voxel_size);

    flow_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_inp[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto ai = _any(inp, size, stride_inp, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_membrane));
    try {
        _matvec_membrane_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), decltype(ai), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, ai, d_kernel, sp, osc, isc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _diag_membrane_k(AO ao, const reduce_t* kernel,
                             flow_sp<ndim, offset_t> sp, offset_t sc,
                             offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d];
        }
        Impl::template diag_membrane<opfunc>(vo.data() + oo, sc, loc, sp.size, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_membrane(
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t sc = stride[nall];

    reduce_t kbuf[Impl::kernelsize_membrane];
    Impl::make_kernel_membrane(kbuf, absolute, membrane, voxel_size);

    flow_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride[nbatch + d];
        sp.sinp[d] = stride[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_membrane));
    try {
        _diag_membrane_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, sp, sc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

//======================================================================
//                              BENDING  (stencil)
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, bound::type... BOUND>
CUGLOB void _matvec_bending_k(AO ao, AI ai, const reduce_t* kernel,
                              flow_sp<ndim, offset_t> sp, offset_t osc, offset_t isc,
                              offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0, io = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d]; io += c * sp.sinp[d];
        }
        Impl::template matvec_bending<opfunc>(
            vo.data() + oo, vi.data() + io, loc, sp.size, sp.sinp, osc, isc, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void matvec_bending(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     bending,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_out[nall], isc = stride_inp[nall];

    reduce_t kbuf[Impl::kernelsize_bending];
    Impl::make_kernel_bending(kbuf, absolute, membrane, bending, voxel_size);

    flow_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_inp[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto ai = _any(inp, size, stride_inp, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_bending));
    try {
        _matvec_bending_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), decltype(ai), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, ai, d_kernel, sp, osc, isc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _diag_bending_k(AO ao, const reduce_t* kernel,
                            flow_sp<ndim, offset_t> sp, offset_t sc,
                            offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d];
        }
        Impl::template diag_bending<opfunc>(vo.data() + oo, sc, loc, sp.size, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_bending(
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     bending,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t sc = stride[nall];

    reduce_t kbuf[Impl::kernelsize_bending];
    Impl::make_kernel_bending(kbuf, absolute, membrane, bending, voxel_size);

    flow_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride[nbatch + d];
        sp.sinp[d] = stride[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_bending));
    try {
        _diag_bending_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, sp, sc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

FF_NAMESPACE_END(reg_flow)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_REGULARISERS_FLOW_CUDA
