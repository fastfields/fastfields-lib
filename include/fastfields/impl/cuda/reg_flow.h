#pragma once
#include <fastfields/core/cuda_switch.h>
#include <fastfields/core/bounds.h>
#include <fastfields/core/utils.h>
#include <fastfields/core/batch.h>
#include <fastfields/impl/kernels/regularisers/flow.h>
#include <fastfields/impl/kernels/posdef.h>
#include "utils.h"       // allocDevice / copyToDevice / freeDevice / GET_BLOCKS
#include <stdexcept>     // std::logic_error

using namespace std;
FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_flow)

//======================================================================
//                              ABSOLUTE
//======================================================================

// --- ABSOLUTE: matvec -----------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void matvec_absolute(
    bound::BoundVec bnd,
    scalar_t * out,                  // (*batch, *spatial, channels) tensor
    const scalar_t * inp,            // (*batch, *spatial, channels) tensor
    const offset_t * _size,          // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * _stride_inp,    // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,    // [*spatial] vector
    reduce_t absolute)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size, _size);
    offset_t stride_out [nall+1]; fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp [nall+1]; fillfrom<nall+1>(stride_inp, _stride_inp);
    reduce_t voxel_size [ndim];   fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    // compute kernel
    reduce_t kernel[Impl::kernelsize_absolute];
    impl.make_kernel_absolute(kernel, absolute, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t inp_offset = index2offset<nall>(i, size, stride_inp);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);

        impl.template matvec_absolute<opfunc>(
            out + out_offset, inp + inp_offset, osc, isc, kernel);
    }
}

// --- ABSOLUTE: kernel ------------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void kernel_absolute(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size, _size);
    offset_t stride     [nall+1]; fillfrom<nall+1>(stride, _stride);
    reduce_t voxel_size [ndim];   fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t sc = stride[nall];
    offset_t numel = prod<nbatch>(size);  // loop across batch only

    reduce_t kernel[Impl::kernelsize_absolute];
    impl.make_kernel_absolute(kernel, absolute, voxel_size);

    offset_t offset = center_offset<ndim>(size+nbatch, stride+nbatch);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride);
        out_offset += offset;

        impl.template kernel_absolute<opfunc>(out + out_offset, sc, kernel);
    }
}

// --- ABSOLUTE: diagonal ----------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void diag_absolute(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size, _size);
    offset_t stride     [nall+1]; fillfrom<nall+1>(stride, _stride);
    reduce_t voxel_size [ndim];   fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t sc = stride[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_absolute];
    impl.make_kernel_absolute(kernel, absolute, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride, loc);

        impl.template diag_absolute<opfunc>(out + out_offset, sc, kernel);
    }
}

//======================================================================
//                              MEMBRANE
//======================================================================

// --- MEMBRANE: matvec -----------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void matvec_membrane(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const scalar_t * inp,           // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,   // [*batch, *spatial, channels] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size,       _size);
    offset_t stride_out [nall+1]; fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp [nall+1]; fillfrom<nall+1>(stride_inp, _stride_inp);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane];
    impl.make_kernel_membrane(kernel, absolute, membrane, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim,nall>(i, size, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);

        impl.template matvec_membrane<opfunc>(
            out + out_offset, inp + inp_offset,
            loc, size + nbatch, stride_inp + nbatch, osc, isc, kernel);
    }
}

// --- MEMBRANE: kernel ------------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void kernel_membrane(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size,       _size);
    offset_t stride     [nall+1]; fillfrom<nall+1>(stride,     _stride);
    reduce_t voxel_size [ndim];   fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t sc = stride[nall];
    offset_t numel = prod<nbatch>(size);

    reduce_t kernel[Impl::kernelsize_membrane];
    impl.make_fullkernel_membrane(kernel, absolute, membrane, voxel_size);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride);
        out_offset += offset;

        impl.template kernel_membrane<opfunc>(
            out + out_offset, sc, stride + nbatch, kernel);
    }
}

// --- MEMBRANE: diagonal ----------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void diag_membrane(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size,       _size);
    offset_t stride     [nall+1]; fillfrom<nall+1>(stride,     _stride);
    reduce_t voxel_size [ndim];   fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t sc = stride[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane];
    impl.make_kernel_membrane(kernel, absolute, membrane, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride, loc);

        impl.template diag_membrane<opfunc>(
            out + out_offset, sc, loc, size + nbatch, kernel);
    }
}

// --- MEMBRANE: relax -------------------------------------------------

template <int nbatch, int ndim,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void relax_membrane_(
    bound::BoundVec bnd,
    scalar_t * sol,                 // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * _stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * _stride_grd,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane,
    int n)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size,       _size);
    offset_t stride_sol [nall+1]; fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes [nall+1]; fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd [nall+1]; fillfrom<nall+1>(stride_grd, _stride_grd);
    reduce_t voxel_size [ndim];   fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane];
    impl.make_kernel_membrane(kernel, absolute, membrane, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    offset_t loc[ndim];
    scalar_t val[ndim], diag[ndim];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch1<ndim, offset_t>(loc, n))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);

        // gradient
#       pragma unroll
        for (int d=0; d<ndim; ++d)
            val[d] = grd[grd_offset + gsc*d];

        // minus convolution
        impl.template matvec_membrane<isub>(
            val, sol + sol_offset,
            loc, size + nbatch, stride_sol + nbatch,
            static_cast<offset_t>(1), osc, kernel);

        // diagonal
        impl.template diag_membrane<set>(
            diag, static_cast<offset_t>(1), loc, size + nbatch, kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}

//======================================================================
//                              BENDING
//======================================================================

// --- BENDING: matvec ------------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void matvec_bending(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane, reduce_t bending)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size,       _size);
    offset_t stride_out [nall+1]; fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp [nall+1]; fillfrom<nall+1>(stride_inp, _stride_inp);
    reduce_t voxel_size [ndim];   fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending];
    impl.make_kernel_bending(kernel, absolute, membrane, bending, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim,nall>(i, size, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);

        impl.template matvec_bending<opfunc>(
            out + out_offset, inp + inp_offset,
            loc, size + nbatch, stride_inp + nbatch, osc, isc, kernel);
    }
}

// --- BENDING: kernel -------------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void kernel_bending(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane, reduce_t bending)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride[nall+1];      fillfrom<nall+1>(stride,     _stride);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t sc = stride[nall];
    offset_t numel = prod<nbatch>(size);

    reduce_t kernel[Impl::kernelsize_bending];
    impl.make_fullkernel_bending(kernel, absolute, membrane, bending, voxel_size);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride);
        out_offset += offset;

        impl.template kernel_bending<opfunc>(
            out + out_offset, sc, stride + nbatch, kernel);
    }
}

// --- BENDING: diagonal -----------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void diag_bending(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane, reduce_t bending)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride[nall+1];      fillfrom<nall+1>(stride,     _stride);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t sc = stride[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending];
    impl.make_kernel_bending(kernel, absolute, membrane, bending, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride, loc);

        impl.template diag_bending<opfunc>(
            out + out_offset, sc, loc, size + nbatch, kernel);
    }
}

// --- BENDING: relax --------------------------------------------------

template <int nbatch, int ndim,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void relax_bending_(
    bound::BoundVec bnd,
    scalar_t * sol,                 // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * _stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * _stride_grd,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane, reduce_t bending,
    int n=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending];
    impl.make_kernel_bending(kernel, absolute, membrane, bending, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    offset_t loc[ndim];
    scalar_t val[ndim], diag[ndim];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch3<ndim, offset_t>(loc, n))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);

        // gradient
#       pragma unroll
        for (int d=0; d<ndim; ++d)
            val[d] = grd[grd_offset + gsc*d];

        // minus convolution
        impl.template matvec_bending<isub>(
            val, sol + sol_offset,
            loc, size + nbatch, stride_sol + nbatch,
            static_cast<offset_t>(1), osc, kernel);

        // diagonal
        impl.template diag_bending<set>(
            diag, static_cast<offset_t>(1), loc, size + nbatch, kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}

//======================================================================
//                              LAME
//======================================================================

// --- LAME: matvec ---------------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void matvec_lame(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane, reduce_t shears, reduce_t div)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp[nall+1];  fillfrom<nall+1>(stride_inp, _stride_inp);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_lame];
    impl.make_kernel_lame(kernel, absolute, membrane, shears, div, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim,nall>(i, size, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);

        impl.template matvec_lame<opfunc>(
            out + out_offset, inp + inp_offset,
            loc, size + nbatch, stride_inp + nbatch, osc, isc, kernel);
    }
}

// --- LAME: kernel ----------------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void kernel_lame(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, C, C) tensor
    const offset_t * _size,         // [*batch, *spatial, C, C] vector
    const offset_t * _stride,       // [*batch, *spatial, C, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane, reduce_t shears, reduce_t div)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+2];        fillfrom<nall+2>(size,     _size);
    offset_t stride[nall+2];      fillfrom<nall+2>(stride,   _stride);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t numel = prod<nbatch>(size);

    reduce_t kernel[Impl::kernelsize_lame];
    impl.make_fullkernel_lame(kernel, absolute, membrane, shears, div, voxel_size);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride);
        out_offset += offset;

        impl.template kernel_lame<opfunc>(
            out + out_offset, stride + nall, stride + nbatch, kernel);
    }
}

// --- LAME: diagonal --------------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void diag_lame(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride,       // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane, reduce_t shears, reduce_t div)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride[nall+1];      fillfrom<nall+1>(stride,     _stride);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t sc = stride[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_lame];
    impl.make_kernel_lame(kernel, absolute, membrane, shears, div, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride, loc);

        impl.template diag_lame<opfunc>(
            out + out_offset, sc, loc, size + nbatch, kernel);
    }
}
// --- LAME: relax -----------------------------------------------------

template <int nbatch, int ndim,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void relax_lame_(
    bound::BoundVec bnd,
    scalar_t * sol,                 // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * _stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * _stride_grd,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane, reduce_t shears, reduce_t div,
    int n=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_lame];
    impl.make_kernel_lame(kernel, absolute, membrane, shears, div, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    offset_t loc[ndim];
    scalar_t val[ndim], diag[ndim];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch2<ndim, offset_t>(loc, n))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);

        // gradient
#       pragma unroll
        for (int d=0; d<ndim; ++d)
            val[d] = grd[grd_offset + gsc*d];

        // minus convolution
        impl.template matvec_lame<isub>(
            val, sol + sol_offset,
            loc, size + nbatch, stride_sol + nbatch,
            static_cast<offset_t>(1), osc, kernel);

        // diagonal
        impl.template diag_lame<set>(
            diag, static_cast<offset_t>(1), loc, size + nbatch, kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}

//======================================================================
//                          LAME + BENDING
//======================================================================

// --- BENDING+LAME: matvec -------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void matvec_all(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane, reduce_t bending,
    reduce_t shears, reduce_t div)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp[nall+1];  fillfrom<nall+1>(stride_inp, _stride_inp);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_all];
    impl.make_kernel_all(kernel, absolute, membrane, bending, shears, div, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim,nall>(i, size, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);

        impl.template matvec_all<opfunc>(
            out + out_offset, inp + inp_offset,
            loc, size + nbatch, stride_inp + nbatch, osc, isc, kernel);
    }
}

// --- BENDING+LAME: kernel --------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void kernel_all(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, C, C) tensor
    const offset_t * _size,         // [*batch, *spatial, C, C] vector
    const offset_t * _stride,       // [*batch, *spatial, C, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane, reduce_t bending,
    reduce_t shears, reduce_t div)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+2];        fillfrom<nall+2>(size,     _size);
    offset_t stride[nall+2];      fillfrom<nall+2>(stride,   _stride);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t numel = prod<nbatch>(size);

    reduce_t kernel[Impl::kernelsize_all];
    impl.make_fullkernel_all(kernel, absolute, membrane, bending, shears, div, voxel_size);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride);
        out_offset += offset;

        impl.template kernel_all<opfunc>(
            out + out_offset, stride + nall, stride + nbatch, kernel);
    }
}

// --- BENDING+LAME: diagonal ------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void diag_all(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride,       // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane, reduce_t bending,
    reduce_t shears, reduce_t div)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride[nall+1];      fillfrom<nall+1>(stride,     _stride);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t sc = stride[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_all];
    impl.make_kernel_all(kernel, absolute, membrane, bending, shears, div, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride, loc);

        impl.template diag_all<opfunc>(
            out + out_offset, sc, loc, size + nbatch, kernel);
    }
}

// --- BENDING+LAME: relax ---------------------------------------------

template <int nbatch, int ndim,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void relax_all_(
    bound::BoundVec bnd,
    scalar_t * sol,                 // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * _stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * _stride_grd,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane, reduce_t bending,
    reduce_t shears, reduce_t div,
    int n=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_all];
    impl.make_kernel_all(kernel, absolute, membrane, bending, shears, div, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

        offset_t loc[ndim];
        scalar_t val[ndim], diag[ndim];
        reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch3<ndim, offset_t>(loc, n))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);

        // gradient
#       pragma unroll
        for (int d=0; d<ndim; ++d)
            val[d] = grd[grd_offset + gsc*d];

        // minus convolution
        impl.template matvec_all<isub>(
            val, sol + sol_offset,
            loc, size + nbatch, stride_sol + nbatch,
            static_cast<offset_t>(1), osc, kernel);

        // diagonal
        impl.template diag_all<set>(
            diag, static_cast<offset_t>(1), loc, size + nbatch, kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}

//======================================================================
//                           MEMBRANE JRLS
//======================================================================

// --- MEMBRANE+JRLS: matvec ------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void matvec_membrane_jrls(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp[nall+1];  fillfrom<nall+1>(stride_inp, _stride_inp);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane_jrls];
    impl.make_kernel_membrane_jrls(kernel, absolute, membrane, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim,nall>(i, size, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        impl.template matvec_membrane_jrls<opfunc>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
            osc, isc, kernel);
    }
}

// --- MEMBRANE+JRLS: diagonal -----------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void diag_membrane_jrls(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,   // [*batch, *spatial, channels] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane_jrls];
    impl.make_kernel_membrane_jrls(kernel, absolute, membrane, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride_out, loc);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        impl.template diag_membrane_jrls<opfunc>(
            out + out_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_wgt + nbatch, osc, kernel);
    }
}

// --- MEMBRANE+JRLS: relax --------------------------------------------

template <int nbatch, int ndim,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void relax_membrane_jrls_(
    bound::BoundVec bnd,
    scalar_t * sol,                 // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * _stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * _stride_grd,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane,
    int n=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane_jrls];
    impl.make_kernel_membrane_jrls(kernel, absolute, membrane, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    offset_t loc[ndim];
    scalar_t val[ndim], diag[ndim];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch1<ndim, offset_t>(loc, n))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        // gradient
#       pragma unroll
        for (int d=0; d<ndim; ++d)
            val[d] = grd[grd_offset + gsc*d];

        // minus convolution
        impl.template matvec_membrane_jrls<isub>(
            val, sol + sol_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), osc, kernel);

        // diagonal
        impl.template diag_membrane_jrls<set>(
            diag, wgt + wgt_offset, loc,
            size + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}

//======================================================================
//                           LAME JRLS
//======================================================================

// --- LAME+JRLS: matvec ----------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void matvec_lame_jrls(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane, reduce_t shears, reduce_t div)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp[nall+1];  fillfrom<nall+1>(stride_inp, _stride_inp);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_lame_jrls];
    impl.make_kernel_lame_jrls(kernel, absolute, membrane, shears, div, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim,nall>(i, size, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        impl.template matvec_lame_jrls<opfunc>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
            osc, isc, kernel);
    }
}

// --- LAME+JRLS: diagonal ---------------------------------------------

template <int nbatch, int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void diag_lame_jrls(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,   // [*batch, *spatial, channels] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane, reduce_t shears, reduce_t div)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_lame_jrls];
    impl.make_kernel_lame_jrls(kernel, absolute, membrane, shears, div, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride_out, loc);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        impl.template diag_lame_jrls<opfunc>(
            out + out_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_wgt + nbatch, osc, kernel);
    }
}

// --- LAME+JRLS: relax ------------------------------------------------

template <int nbatch, int ndim,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void relax_lame_jrls_(
    bound::BoundVec bnd,
    scalar_t * sol,                 // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * _stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * _stride_grd,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    reduce_t absolute, reduce_t membrane, reduce_t shears, reduce_t div,
    int n=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_lame_jrls];
    impl.make_kernel_lame_jrls(kernel, absolute, membrane, shears, div, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    offset_t loc[ndim];
    scalar_t val[ndim], diag[ndim];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch2<ndim, offset_t>(loc, n))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        // gradient
#       pragma unroll
        for (int d=0; d<ndim; ++d)
            val[d] = grd[grd_offset + gsc*d];

        // minus convolution
        impl.template matvec_lame_jrls<isub>(
            val, sol + sol_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), osc, kernel);

        // diagonal
        impl.template diag_lame_jrls<set>(
            diag, wgt + wgt_offset, loc,
            size + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}

//======================================================================
//                          HOST LAUNCHERS
//======================================================================
//
// The device kernels above are templated on a *compile-time* number of batch
// dimensions (`nbatch`).  The cuda-lib dispatch layer only knows `nbatch` at
// runtime, so these FF_CUHOST launchers:
//   1. copy the (host) shape / stride / voxel-size vectors to the device,
//   2. dispatch the runtime `nbatch` to a bounded set of compile-time
//      instantiations of the matching device kernel,
//   3. launch it on the supplied CUDA `stream`,
//   4. free the temporary device vectors.
//
// The flow kernels carry the channel count implicitly (== ndim), so `nbatch`
// is the *only* runtime->compile-time bridge required here.  A `nbatch` beyond
// the supported range throws std::logic_error (correctly typed, so the module
// still compiles + links).
//
// NOTE: launcher and device kernel deliberately share a name (the cuda-lib
// dispatcher calls e.g. `reg_flow::matvec_absolute<ndim, op, ...>`).  They are
// distinct overloads: the device kernel leads with two `int` params
// (nbatch, ndim) whereas the launcher leads with `int ndim, char op`, so the
// explicit template-argument lists select unambiguously in both directions.

// Dispatch the runtime `nbatch` to a compile-time device-kernel launch.
// `KERN` is the (device) kernel name; the trailing args are the kernel args.
#define FF_REGFLOW_LAUNCH_NBATCH(KERN, ...)                                    \
    switch (nbatch) {                                                          \
        case 0: KERN<0, ndim, op, reduce_t, scalar_t, offset_t, BOUND...>      \
                    <<<blocks, CUDA_NUM_THREADS, 0, stream>>>(bnd, __VA_ARGS__); break; \
        case 1: KERN<1, ndim, op, reduce_t, scalar_t, offset_t, BOUND...>      \
                    <<<blocks, CUDA_NUM_THREADS, 0, stream>>>(bnd, __VA_ARGS__); break; \
        case 2: KERN<2, ndim, op, reduce_t, scalar_t, offset_t, BOUND...>      \
                    <<<blocks, CUDA_NUM_THREADS, 0, stream>>>(bnd, __VA_ARGS__); break; \
        case 3: KERN<3, ndim, op, reduce_t, scalar_t, offset_t, BOUND...>      \
                    <<<blocks, CUDA_NUM_THREADS, 0, stream>>>(bnd, __VA_ARGS__); break; \
        default: throw std::logic_error(                                       \
            "ff::cuda::reg_flow: nbatch > 3 is not supported by the CUDA launcher"); \
    }

// Same as FF_REGFLOW_LAUNCH_NBATCH but for the relaxers, whose device kernels
// take no `op` template parameter (they always accumulate in place). The last
// kernel argument is the red-black colour index `col`.
#define FF_REGFLOW_LAUNCH_RELAX(KERN, ...)                                     \
    switch (nbatch) {                                                          \
        case 0: KERN<0, ndim, reduce_t, scalar_t, offset_t, BOUND...>          \
                    <<<blocks, CUDA_NUM_THREADS, 0, stream>>>(bnd, __VA_ARGS__); break; \
        case 1: KERN<1, ndim, reduce_t, scalar_t, offset_t, BOUND...>          \
                    <<<blocks, CUDA_NUM_THREADS, 0, stream>>>(bnd, __VA_ARGS__); break; \
        case 2: KERN<2, ndim, reduce_t, scalar_t, offset_t, BOUND...>          \
                    <<<blocks, CUDA_NUM_THREADS, 0, stream>>>(bnd, __VA_ARGS__); break; \
        case 3: KERN<3, ndim, reduce_t, scalar_t, offset_t, BOUND...>          \
                    <<<blocks, CUDA_NUM_THREADS, 0, stream>>>(bnd, __VA_ARGS__); break; \
        default: throw std::logic_error(                                       \
            "ff::cuda::reg_flow: nbatch > 3 is not supported by the CUDA launcher"); \
    }

// --- ABSOLUTE ---------------------------------------------------------

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void matvec_absolute(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_inp = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_inp = copyToDevice(stride_inp, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(matvec_absolute,
            out, inp, d_size, d_stride_out, d_stride_inp, d_vx, absolute)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_inp, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_inp, d_vx);
}

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void diag_absolute(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride_out,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(diag_absolute,
            out, d_size, d_stride_out, d_vx, absolute)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_vx);
}

// --- MEMBRANE ---------------------------------------------------------

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void matvec_membrane(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_inp = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_inp = copyToDevice(stride_inp, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(matvec_membrane,
            out, inp, d_size, d_stride_out, d_stride_inp, d_vx, absolute, membrane)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_inp, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_inp, d_vx);
}

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void diag_membrane(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride_out,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(diag_membrane,
            out, d_size, d_stride_out, d_vx, absolute, membrane)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_vx);
}

// --- BENDING ----------------------------------------------------------

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void matvec_bending(
    const bound::BoundVec & bnd,
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
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_inp = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_inp = copyToDevice(stride_inp, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(matvec_bending,
            out, inp, d_size, d_stride_out, d_stride_inp, d_vx, absolute, membrane, bending)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_inp, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_inp, d_vx);
}

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void diag_bending(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride_out,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     bending,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(diag_bending,
            out, d_size, d_stride_out, d_vx, absolute, membrane, bending)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_vx);
}

// --- BENDING+LAME (absolute/membrane/bending + shears/div) ------------

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void matvec_all(
    const bound::BoundVec & bnd,
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
          reduce_t     shears,
          reduce_t     div,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_inp = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_inp = copyToDevice(stride_inp, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(matvec_all,
            out, inp, d_size, d_stride_out, d_stride_inp, d_vx,
            absolute, membrane, bending, shears, div)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_inp, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_inp, d_vx);
}

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void diag_all(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride_out,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     bending,
          reduce_t     shears,
          reduce_t     div,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(diag_all,
            out, d_size, d_stride_out, d_vx,
            absolute, membrane, bending, shears, div)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_vx);
}

// --- MEMBRANE + JRLS ---------------------------------------------------
//
// The JRLS device kernels add a per-voxel, per-channel weight `wgt` on top
// of the plain membrane/lame operators (joint reweighted least squares).
// Their (nbatch, ndim[, op]) template shapes match the plain kernels
// exactly, so the FF_REGFLOW_LAUNCH_NBATCH / FF_REGFLOW_LAUNCH_RELAX macros
// above are reused as-is -- no new dispatch macro is needed. `wgt` varies
// per channel like `out`/`inp`, so its stride vector is only nall entries
// (no explicit channel-stride slot) -- the device kernels fillfrom<nall>(...)
// it, one fewer element than stride_out/stride_inp/stride_sol.

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void matvec_membrane_jrls(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_inp = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_inp = copyToDevice(stride_inp, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(matvec_membrane_jrls,
            out, inp, wgt, d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx, absolute, membrane)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx);
}

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void diag_membrane_jrls(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(diag_membrane_jrls,
            out, wgt, d_size, d_stride_out, d_stride_wgt, d_vx, absolute, membrane)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_wgt, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_wgt, d_vx);
}

// --- LAME + JRLS ---------------------------------------------------------

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void matvec_lame_jrls(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     shears,
          reduce_t     div,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_inp = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_inp = copyToDevice(stride_inp, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(matvec_lame_jrls,
            out, inp, wgt, d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx,
            absolute, membrane, shears, div)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx);
}

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void diag_lame_jrls(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     shears,
          reduce_t     div,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(diag_lame_jrls,
            out, wgt, d_size, d_stride_out, d_stride_wgt, d_vx,
            absolute, membrane, shears, div)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_wgt, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_wgt, d_vx);
}

// --- KERNEL launchers ------------------------------------------------
//
// Materialise the Toeplitz stencil of the operator. The device kernels loop
// over the *batch* dims only (one centred stencil per batch element), so the
// launch grid is sized on `prod(size, nbatch)`. The per-channel (vector)
// stencils carry a (*batch, *spatial, C) layout (arrays length nall+1); the
// Lamé (matrix) stencils carry (*batch, *spatial, C, C) (arrays length nall+2).

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void kernel_absolute(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride_out,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nbatch);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(kernel_absolute,
            out, d_size, d_stride_out, d_vx, absolute)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_vx);
}

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void kernel_membrane(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride_out,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nbatch);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(kernel_membrane,
            out, d_size, d_stride_out, d_vx, absolute, membrane)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_vx);
}

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void kernel_bending(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride_out,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     bending,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nbatch);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(kernel_bending,
            out, d_size, d_stride_out, d_vx, absolute, membrane, bending)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_vx);
}

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void kernel_lame(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride_out,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     shears,
          reduce_t     div,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nbatch);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 2);
        d_stride_out = copyToDevice(stride_out, nall + 2);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(kernel_lame,
            out, d_size, d_stride_out, d_vx, absolute, membrane, shears, div)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_vx);
}

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void kernel_all(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride_out,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     bending,
          reduce_t     shears,
          reduce_t     div,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nbatch);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr;
    reduce_t * d_vx   = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 2);
        d_stride_out = copyToDevice(stride_out, nall + 2);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        FF_REGFLOW_LAUNCH_NBATCH(kernel_all,
            out, d_size, d_stride_out, d_vx,
            absolute, membrane, bending, shears, div)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_vx);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_vx);
}

// --- RELAX launchers -------------------------------------------------
//
// Gauss-Seidel relaxation over the red-black (patch3: 3^ndim) colouring. Each
// device-kernel launch updates one colour; the host loops colours x nb_iter
// (a single counter `col`, which the kernel folds mod 3^ndim). Mirrors the CPU
// relaxers' loop structure. Compile-validated only (no GPU in CI).

// number of red-black colours for the patch3 scheme
#define FF_REGFLOW_NCOLOURS(P) (pow<ndim>(static_cast<offset_t>(P)))

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void relax_membrane_(
    const bound::BoundVec & bnd,
          offset_t nbatch, scalar_t * sol, const scalar_t * hes,
    const scalar_t * grd, const offset_t * size, const offset_t * stride_sol,
    const offset_t * stride_hes, const offset_t * stride_grd,
    const reduce_t * voxel_size, reduce_t absolute, reduce_t membrane,
          int nb_iter, cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    const offset_t ncol   = static_cast<offset_t>(2);   /* patch1 */
    offset_t * d_size=nullptr,* d_ss=nullptr,* d_sh=nullptr,* d_sg=nullptr;
    reduce_t * d_vx=nullptr;
    try {
        d_size = copyToDevice(size,       nall + 1);
        d_ss   = copyToDevice(stride_sol, nall + 1);
        d_sh   = copyToDevice(stride_hes, nall + 1);
        d_sg   = copyToDevice(stride_grd, nall + 1);
        d_vx   = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        for (offset_t col = 0; col < ncol * nb_iter; ++col)
            FF_REGFLOW_LAUNCH_RELAX(relax_membrane_,
                sol, hes, grd, d_size, d_ss, d_sh, d_sg, d_vx,
                absolute, membrane, static_cast<int>(col))
    } catch (const std::exception &) {
        freeDevice(d_size, d_ss, d_sh, d_sg, d_vx); throw;
    }
    freeDevice(d_size, d_ss, d_sh, d_sg, d_vx);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void relax_bending_(
    const bound::BoundVec & bnd,
          offset_t nbatch, scalar_t * sol, const scalar_t * hes,
    const scalar_t * grd, const offset_t * size, const offset_t * stride_sol,
    const offset_t * stride_hes, const offset_t * stride_grd,
    const reduce_t * voxel_size, reduce_t absolute, reduce_t membrane,
          reduce_t bending, int nb_iter, cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    const offset_t ncol   = FF_REGFLOW_NCOLOURS(3); /* patch3 */
    offset_t * d_size=nullptr,* d_ss=nullptr,* d_sh=nullptr,* d_sg=nullptr;
    reduce_t * d_vx=nullptr;
    try {
        d_size = copyToDevice(size,       nall + 1);
        d_ss   = copyToDevice(stride_sol, nall + 1);
        d_sh   = copyToDevice(stride_hes, nall + 1);
        d_sg   = copyToDevice(stride_grd, nall + 1);
        d_vx   = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        for (offset_t col = 0; col < ncol * nb_iter; ++col)
            FF_REGFLOW_LAUNCH_RELAX(relax_bending_,
                sol, hes, grd, d_size, d_ss, d_sh, d_sg, d_vx,
                absolute, membrane, bending, static_cast<int>(col))
    } catch (const std::exception &) {
        freeDevice(d_size, d_ss, d_sh, d_sg, d_vx); throw;
    }
    freeDevice(d_size, d_ss, d_sh, d_sg, d_vx);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void relax_lame_(
    const bound::BoundVec & bnd,
          offset_t nbatch, scalar_t * sol, const scalar_t * hes,
    const scalar_t * grd, const offset_t * size, const offset_t * stride_sol,
    const offset_t * stride_hes, const offset_t * stride_grd,
    const reduce_t * voxel_size, reduce_t absolute, reduce_t membrane,
          reduce_t shears, reduce_t div, int nb_iter, cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    const offset_t ncol   = FF_REGFLOW_NCOLOURS(2); /* patch2 */
    offset_t * d_size=nullptr,* d_ss=nullptr,* d_sh=nullptr,* d_sg=nullptr;
    reduce_t * d_vx=nullptr;
    try {
        d_size = copyToDevice(size,       nall + 1);
        d_ss   = copyToDevice(stride_sol, nall + 1);
        d_sh   = copyToDevice(stride_hes, nall + 1);
        d_sg   = copyToDevice(stride_grd, nall + 1);
        d_vx   = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        for (offset_t col = 0; col < ncol * nb_iter; ++col)
            FF_REGFLOW_LAUNCH_RELAX(relax_lame_,
                sol, hes, grd, d_size, d_ss, d_sh, d_sg, d_vx,
                absolute, membrane, shears, div, static_cast<int>(col))
    } catch (const std::exception &) {
        freeDevice(d_size, d_ss, d_sh, d_sg, d_vx); throw;
    }
    freeDevice(d_size, d_ss, d_sh, d_sg, d_vx);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void relax_all_(
    const bound::BoundVec & bnd,
          offset_t nbatch, scalar_t * sol, const scalar_t * hes,
    const scalar_t * grd, const offset_t * size, const offset_t * stride_sol,
    const offset_t * stride_hes, const offset_t * stride_grd,
    const reduce_t * voxel_size, reduce_t absolute, reduce_t membrane,
          reduce_t bending, reduce_t shears, reduce_t div, int nb_iter,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    const offset_t ncol   = FF_REGFLOW_NCOLOURS(3); /* patch3 */
    offset_t * d_size=nullptr,* d_ss=nullptr,* d_sh=nullptr,* d_sg=nullptr;
    reduce_t * d_vx=nullptr;
    try {
        d_size = copyToDevice(size,       nall + 1);
        d_ss   = copyToDevice(stride_sol, nall + 1);
        d_sh   = copyToDevice(stride_hes, nall + 1);
        d_sg   = copyToDevice(stride_grd, nall + 1);
        d_vx   = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        for (offset_t col = 0; col < ncol * nb_iter; ++col)
            FF_REGFLOW_LAUNCH_RELAX(relax_all_,
                sol, hes, grd, d_size, d_ss, d_sh, d_sg, d_vx,
                absolute, membrane, bending, shears, div,
                static_cast<int>(col))
    } catch (const std::exception &) {
        freeDevice(d_size, d_ss, d_sh, d_sg, d_vx); throw;
    }
    freeDevice(d_size, d_ss, d_sh, d_sg, d_vx);
}

// --- RELAX + JRLS ------------------------------------------------------
//
// Same red-black colouring as the plain relaxers (relax_membrane_jrls_ uses
// patch1: 2 colours; relax_lame_jrls_ uses patch2: 2^ndim colours), plus the
// per-voxel, per-channel `wgt` weight. `wgt`'s stride vector is only nall
// entries (mirrors the JRLS matvec/diag launchers above).

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void relax_membrane_jrls_(
    const bound::BoundVec & bnd,
          offset_t nbatch, scalar_t * sol, const scalar_t * hes,
    const scalar_t * grd, const scalar_t * wgt,
    const offset_t * size, const offset_t * stride_sol,
    const offset_t * stride_hes, const offset_t * stride_grd,
    const offset_t * stride_wgt,
    const reduce_t * voxel_size, reduce_t absolute, reduce_t membrane,
          int nb_iter, cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    const offset_t ncol   = static_cast<offset_t>(2);   /* patch1 */
    offset_t * d_size=nullptr,* d_ss=nullptr,* d_sh=nullptr,* d_sg=nullptr,* d_sw=nullptr;
    reduce_t * d_vx=nullptr;
    try {
        d_size = copyToDevice(size,       nall + 1);
        d_ss   = copyToDevice(stride_sol, nall + 1);
        d_sh   = copyToDevice(stride_hes, nall + 1);
        d_sg   = copyToDevice(stride_grd, nall + 1);
        d_sw   = copyToDevice(stride_wgt, nall);
        d_vx   = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        for (offset_t col = 0; col < ncol * nb_iter; ++col)
            FF_REGFLOW_LAUNCH_RELAX(relax_membrane_jrls_,
                sol, hes, grd, wgt, d_size, d_ss, d_sh, d_sg, d_sw, d_vx,
                absolute, membrane, static_cast<int>(col))
    } catch (const std::exception &) {
        freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_vx); throw;
    }
    freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_vx);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void relax_lame_jrls_(
    const bound::BoundVec & bnd,
          offset_t nbatch, scalar_t * sol, const scalar_t * hes,
    const scalar_t * grd, const scalar_t * wgt,
    const offset_t * size, const offset_t * stride_sol,
    const offset_t * stride_hes, const offset_t * stride_grd,
    const offset_t * stride_wgt,
    const reduce_t * voxel_size, reduce_t absolute, reduce_t membrane,
          reduce_t shears, reduce_t div, int nb_iter, cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    const offset_t ncol   = FF_REGFLOW_NCOLOURS(2); /* patch2 */
    offset_t * d_size=nullptr,* d_ss=nullptr,* d_sh=nullptr,* d_sg=nullptr,* d_sw=nullptr;
    reduce_t * d_vx=nullptr;
    try {
        d_size = copyToDevice(size,       nall + 1);
        d_ss   = copyToDevice(stride_sol, nall + 1);
        d_sh   = copyToDevice(stride_hes, nall + 1);
        d_sg   = copyToDevice(stride_grd, nall + 1);
        d_sw   = copyToDevice(stride_wgt, nall);
        d_vx   = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        for (offset_t col = 0; col < ncol * nb_iter; ++col)
            FF_REGFLOW_LAUNCH_RELAX(relax_lame_jrls_,
                sol, hes, grd, wgt, d_size, d_ss, d_sh, d_sg, d_sw, d_vx,
                absolute, membrane, shears, div, static_cast<int>(col))
    } catch (const std::exception &) {
        freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_vx); throw;
    }
    freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_vx);
}

#undef FF_REGFLOW_NCOLOURS
#undef FF_REGFLOW_LAUNCH_RELAX
#undef FF_REGFLOW_LAUNCH_NBATCH

FF_NAMESPACE_END(reg_flow)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
