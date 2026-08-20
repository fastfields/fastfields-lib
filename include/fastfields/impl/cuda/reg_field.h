#pragma once
#include "fastfields/core/cuda_switch.h"
#include "fastfields/impl/kernels/bounds.h"
#include "fastfields/impl/kernels/utils.h"
#include "fastfields/impl/kernels/batch.h"
#include "fastfields/impl/kernels/regularisers/field.h"
#include "fastfields/impl/kernels/posdef.h"
#include "utils.h"       // allocDevice / copyToDevice / freeDevice / GET_BLOCKS
#include <stdexcept>     // std::logic_error

using namespace std;
FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_field)

//======================================================================
//                              ABSOLUTE
//======================================================================

// --- ABSOLUTE: matvec -----------------------------------------------

template <int nbatch, int ndim, int C, char op,
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
    const reduce_t absolute[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size, _size);
    offset_t stride_out [nall+1]; fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp [nall+1]; fillfrom<nall+1>(stride_inp, _stride_inp);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    // compute kernel
    reduce_t kernel[Impl::kernelsize_absolute];
    impl.make_kernel_absolute(kernel, absolute);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t inp_offset = index2offset<nall>(i, size, stride_inp);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);

        impl.template matvec_absolute<opfunc>(
            out + out_offset, inp + inp_offset, osc, isc, kernel);
    }
}

// --- ABSOLUTE: kernel ------------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void kernel_absolute(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t absolute[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size, _size);
    offset_t stride     [nall+1]; fillfrom<nall+1>(stride, _stride);
    offset_t sc = stride[nall];
    offset_t numel = prod<nbatch>(size);  // loop across batch only

    reduce_t kernel[Impl::kernelsize_absolute];
    impl.make_kernel_absolute(kernel, absolute);

    offset_t offset = center_offset<ndim>(size+nbatch, stride+nbatch);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride);
        out_offset += offset;

        impl.template kernel_absolute<opfunc>(out + out_offset, sc, kernel);
    }
}

// --- ABSOLUTE: diagonal ----------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void diag_absolute(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t absolute[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size       [nall+1]; fillfrom<nall+1>(size, _size);
    offset_t stride     [nall+1]; fillfrom<nall+1>(stride, _stride);
    offset_t sc = stride[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_absolute];
    impl.make_kernel_absolute(kernel, absolute);

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

template <int nbatch, int ndim, int C, char op,
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
    const reduce_t absolute[C],
    const reduce_t membrane[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
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

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void kernel_membrane(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
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

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void diag_membrane(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
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

template <int nbatch, int ndim, int C,
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
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    int niter)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
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
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    offset_t loc[ndim];
    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch1<ndim, offset_t>(loc, niter))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

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

template <int nbatch, int ndim, int C, char op,
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
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    const reduce_t bending[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
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

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void kernel_bending(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    const reduce_t bending[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
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

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void diag_bending(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride,       // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    const reduce_t bending[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
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

template <int nbatch, int ndim, int C,
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
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    const reduce_t bending[C],
    int niter=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
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
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    offset_t loc[ndim];
    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch3<ndim, offset_t>(loc, niter))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

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
//                           ABSOLUTE RLS
//======================================================================

// --- ABSOLUTE+RLS: matvec --------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void matvec_absolute_rls(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * absolute)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp[nall+1];  fillfrom<nall+1>(stride_inp, _stride_inp);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_absolute];
    impl.make_kernel_absolute(kernel, absolute);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t inp_offset = index2offset<nall>(i, size, stride_inp);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        impl.template matvec_absolute_rls<opfunc>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            osc, isc, wsc, kernel);
    }
}

// --- ABSOLUTE+RLS: diagonal ------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void diag_absolute_rls(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,   // [*batch, *spatial, channels] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, channels] vector
    const reduce_t * absolute)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    offset_t osc = stride_out[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_absolute];
    impl.make_kernel_absolute(kernel, absolute);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        impl.template diag_absolute_rls<opfunc>(
            out + out_offset, wgt + wgt_offset, osc, wsc, kernel);
    }
}

// --- ABSOLUTE+RLS: relax ---------------------------------------------

template <int nbatch, int ndim, int C,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void relax_absolute_rls_(
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
    const reduce_t * absolute,
    int niter=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_absolute];
    impl.make_kernel_absolute(kernel, absolute);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset<nall>(i, size, stride_sol);
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

        // minus convolution
        impl.template matvec_absolute_rls<isub>(
            val, sol + sol_offset, wgt + wgt_offset,
            static_cast<offset_t>(1), osc, wsc, kernel);

        // diagonal
        impl.template diag_absolute_rls<set>(
            diag, wgt + wgt_offset,
            static_cast<offset_t>(1), wsc, kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}

//======================================================================
//                           ABSOLUTE JRLS
//======================================================================

// --- ABSOLUTE+JRLS: matvec -------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void matvec_absolute_jrls(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * absolute)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp[nall+1];  fillfrom<nall+1>(stride_inp, _stride_inp);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_absolute];
    impl.make_kernel_absolute(kernel, absolute);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t inp_offset = index2offset<nall>(i, size, stride_inp);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        impl.template matvec_absolute_jrls<opfunc>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            osc, isc, kernel);
    }
}

// --- ABSOLUTE+JRLS: diagonal -----------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void diag_absolute_jrls(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,   // [*batch, *spatial, channels] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, channels] vector
    const reduce_t * absolute)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    offset_t osc = stride_out[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_absolute];
    impl.make_kernel_absolute(kernel, absolute);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        impl.template diag_absolute_jrls<opfunc>(
            out + out_offset, wgt + wgt_offset, osc, kernel);
    }
}

// --- ABSOLUTE+JRLS: relax --------------------------------------------

template <int nbatch, int ndim, int C,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void relax_absolute_jrls_(
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
    const reduce_t * absolute,
    int niter=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_absolute];
    impl.make_kernel_absolute(kernel, absolute);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset<nall>(i, size, stride_sol);
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

        // minus convolution
        impl.template matvec_absolute_jrls<isub>(
            val, sol + sol_offset, wgt + wgt_offset,
            static_cast<offset_t>(1), osc, kernel);

        // diagonal
        impl.template diag_absolute_jrls<set>(
            diag, wgt + wgt_offset,
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
//                           MEMBRANE RLS
//======================================================================

// --- MEMBRANE+RLS: matvec --------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void matvec_membrane_rls(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    const reduce_t bending[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp[nall+1];  fillfrom<nall+1>(stride_inp, _stride_inp);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane_rls];
    impl.make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim,nall>(i, size, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        impl.template matvec_membrane_rls<opfunc>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
            osc, isc, wsc, kernel);
    }
}

// --- MEMBRANE+RLS: diagonal ------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void diag_membrane_rls(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,   // [*batch, *spatial, channels] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t absolute[C],
    const reduce_t membrane[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane_rls];
    impl.make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride_out, loc);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        impl.template diag_membrane_rls<opfunc>(
            out + out_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_wgt + nbatch, osc, wsc, kernel);
    }
}

// --- MEMBRANE+RLS: relax ---------------------------------------------

template <int nbatch, int ndim, int C,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void relax_membrane_rls_(
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
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    int niter=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane_rls];
    impl.make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    offset_t loc[ndim];
    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch1<ndim, offset_t>(loc, niter))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

        // minus convolution
        impl.template matvec_membrane_rls<isub>(
            val, sol + sol_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), osc, wsc, kernel);

        // diagonal
        impl.template diag_membrane_rls<set>(
            diag, wgt + wgt_offset, loc,
            size + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), wsc, kernel);

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

template <int nbatch, int ndim, int C, char op,
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
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    const reduce_t bending[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
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

    reduce_t kernel[Impl::kernelsize_membrane_rls];
    impl.make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size);

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

template <int nbatch, int ndim, int C, char op,
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
    const reduce_t absolute[C],
    const reduce_t membrane[C])
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane_rls];
    impl.make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size);

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

template <int nbatch, int ndim, int C,
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
    const reduce_t absolute[C],
    const reduce_t membrane[C],
    int niter=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
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

    reduce_t kernel[Impl::kernelsize_membrane_rls];
    impl.make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    offset_t loc[ndim];
    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch1<ndim, offset_t>(loc, niter))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

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
//                           BENDING RLS
//======================================================================

// --- BENDING+RLS: matvec ---------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void matvec_bending_rls(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_inp[nall+1];  fillfrom<nall+1>(stride_inp, _stride_inp);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);  // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending_rls];
    impl.make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim,nall>(i, size, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        impl.template matvec_bending_rls<opfunc>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
            osc, isc, wsc, kernel);
    }
}

// --- BENDING+RLS: diagonal -------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void diag_bending_rls(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,   // [*batch, *spatial, channels] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending_rls];
    impl.make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride_out, loc);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        impl.template diag_bending_rls<opfunc>(
            out + out_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_wgt + nbatch, osc, wsc, kernel);
    }
}

// --- BENDING+RLS: relax ----------------------------------------------

template <int nbatch, int ndim, int C,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void relax_bending_rls_(
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
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending,
    int niter=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_sol[nall+1];  fillfrom<nall+1>(stride_sol, _stride_sol);
    offset_t stride_hes[nall+1];  fillfrom<nall+1>(stride_hes, _stride_hes);
    offset_t stride_grd[nall+1];  fillfrom<nall+1>(stride_grd, _stride_grd);
    offset_t stride_wgt[nall+1];  fillfrom<nall+1>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];
    offset_t wsc = stride_wgt[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending_rls];
    impl.make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    offset_t loc[ndim];
    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch3<ndim, offset_t>(loc, niter))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

        // minus convolution
        impl.template matvec_bending_rls<isub>(
            val, sol + sol_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), osc, wsc, kernel);

        // diagonal
        impl.template diag_bending_rls<set>(
            diag, wgt + wgt_offset, loc,
            size + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), wsc, kernel);

        // sol += (hes + diag) \ (grad - conv(sol))
        PosDef::relax_(
            Strided(sol + sol_offset, osc),
            StridedConst(hes + hes_offset, hsc),
            val, diag, buf, static_cast<reduce_t>(0)
        );
    }
}

//======================================================================
//                           BENDING JRLS
//======================================================================

// --- BENDING+JRLS: matvec --------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void matvec_bending_jrls(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * _size,         // [*batch, *spatial, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial, C] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
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

    reduce_t kernel[Impl::kernelsize_bending_rls];
    impl.make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim,nall>(i, size, stride_inp, loc);
        offset_t out_offset = index2offset<nall>(i, size, stride_out);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        impl.template matvec_bending_jrls<opfunc>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
            osc, isc, kernel);
    }
}

// --- BENDING+JRLS: diagonal ------------------------------------------

template <int nbatch, int ndim, int C, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void diag_bending_jrls(
    bound::BoundVec bnd,
    scalar_t * out,                 // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * _size,         // [*batch, *spatial, channels] vector
    const offset_t * _stride_out,   // [*batch, *spatial, channels] vector
    const offset_t * _stride_wgt,   // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);

    // copy vectors to the stack
    offset_t size[nall+1];        fillfrom<nall+1>(size,       _size);
    offset_t stride_out[nall+1];  fillfrom<nall+1>(stride_out, _stride_out);
    offset_t stride_wgt[nall];    fillfrom<nall>(stride_wgt, _stride_wgt);
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t osc = stride_out[nall];
    offset_t numel = prod<nall>(size);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_bending_rls];
    impl.make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size);

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim,nall>(i, size, stride_out, loc);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        impl.template diag_bending_jrls<opfunc>(
            out + out_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_wgt + nbatch, osc, kernel);
    }
}

// --- BENDING+JRLS: relax ---------------------------------------------

template <int nbatch, int ndim, int C,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUGLOB
void relax_bending_jrls_(
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
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending,
    int niter=1)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    static constexpr int nall = nbatch + ndim;
    using Impl = RegField<C, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    Impl impl(bnd);
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, C>;
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

    reduce_t kernel[Impl::kernelsize_bending_rls];
    impl.make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;

    offset_t loc[ndim];
    scalar_t val[C], diag[C];
    reduce_t buf[CC ? CC : 1];

    for (offset_t i=index; index < numel; index += index_stride, i=index)
    {
        offset_t sol_offset = index2offset_v2<ndim,nall>(i, size, stride_sol, loc);
        if (!patch3<ndim, offset_t>(loc, niter))
            continue;
        offset_t grd_offset = index2offset<nall>(i, size, stride_grd);
        offset_t hes_offset = index2offset<nall>(i, size, stride_hes);
        offset_t wgt_offset = index2offset<nall>(i, size, stride_wgt);

        // gradient
        for (int c=0; c<C; ++c)
            val[c] = grd[grd_offset + gsc*c];

        // minus convolution
        impl.template matvec_bending_jrls<isub>(
            val, sol + sol_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
            static_cast<offset_t>(1), osc, kernel);

        // diagonal
        impl.template diag_bending_jrls<set>(
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
// dimensions (`nbatch`) and a *compile-time* channel count (`C`).  The cuda-lib
// dispatch layer only knows both at runtime, so these FF_CUHOST launchers:
//   1. copy the (host) shape / stride / voxel-size / weight vectors to the
//      device,
//   2. dispatch the runtime (`nbatch`, `nc`) pair to a bounded set of
//      compile-time instantiations of the matching device kernel,
//   3. launch it on the supplied CUDA `stream`,
//   4. free the temporary device vectors.
//
// Supported ranges are nbatch in [0, 1] and nc (channels) in [1, 3]; anything
// outside throws std::logic_error (correctly typed, so the module still
// compiles + links).  The ranges are deliberately small to keep the
// (nbatch x nc x ndim x bound x dtype) template fan-out — and thus nvcc
// compile time — bounded; widen them here if larger fields are needed.
//
// NOTE: launcher and device kernel deliberately share a name (the cuda-lib
// dispatcher calls e.g. `reg_field::matvec_absolute<ndim, op, ...>`).  They are
// distinct overloads: the device kernel leads with three `int` params
// (nbatch, ndim, C) whereas the launcher leads with `int ndim, char op`, so the
// explicit template-argument lists select unambiguously in both directions.

// Inner dispatch: given a compile-time NB (nbatch), pick the channel count.
#define FF_REGFIELD_LAUNCH_C(KERN, NB, ...)                                    \
    switch (nc) {                                                              \
        case 1: KERN<NB, ndim, 1, op, reduce_t, scalar_t, offset_t, BOUND...>  \
                    <<<blocks, CUDA_NUM_THREADS, 0, stream>>>(bnd, __VA_ARGS__); break; \
        case 2: KERN<NB, ndim, 2, op, reduce_t, scalar_t, offset_t, BOUND...>  \
                    <<<blocks, CUDA_NUM_THREADS, 0, stream>>>(bnd, __VA_ARGS__); break; \
        case 3: KERN<NB, ndim, 3, op, reduce_t, scalar_t, offset_t, BOUND...>  \
                    <<<blocks, CUDA_NUM_THREADS, 0, stream>>>(bnd, __VA_ARGS__); break; \
        default: throw std::logic_error(                                       \
            "ff::cuda::reg_field: channel count outside [1, 3] is not "        \
            "supported by the CUDA launcher");                                 \
    }

// Outer dispatch: pick the (compile-time) number of batch dimensions.
#define FF_REGFIELD_LAUNCH(KERN, ...)                                          \
    switch (nbatch) {                                                          \
        case 0: FF_REGFIELD_LAUNCH_C(KERN, 0, __VA_ARGS__); break;             \
        case 1: FF_REGFIELD_LAUNCH_C(KERN, 1, __VA_ARGS__); break;             \
        default: throw std::logic_error(                                       \
            "ff::cuda::reg_field: nbatch > 1 is not supported by the CUDA "    \
            "launcher");                                                        \
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
    const reduce_t   * absolute,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_inp = nullptr;
    reduce_t * d_abs  = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_inp = copyToDevice(stride_inp, nall + 1);
        d_abs        = copyToDevice(absolute,   nc);
        FF_REGFIELD_LAUNCH(matvec_absolute,
            out, inp, d_size, d_stride_out, d_stride_inp, d_abs)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_inp, d_abs);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_inp, d_abs);
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
    const reduce_t   * absolute,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr;
    reduce_t * d_abs  = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_abs        = copyToDevice(absolute,   nc);
        FF_REGFIELD_LAUNCH(diag_absolute,
            out, d_size, d_stride_out, d_abs)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_abs);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_abs);
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_inp = nullptr;
    reduce_t * d_vx = nullptr, * d_abs = nullptr, * d_mem = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_inp = copyToDevice(stride_inp, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs        = copyToDevice(absolute,   nc);
        d_mem        = copyToDevice(membrane,   nc);
        FF_REGFIELD_LAUNCH(matvec_membrane,
            out, inp, d_size, d_stride_out, d_stride_inp, d_vx, d_abs, d_mem)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_inp, d_vx, d_abs, d_mem);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_inp, d_vx, d_abs, d_mem);
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr;
    reduce_t * d_vx = nullptr, * d_abs = nullptr, * d_mem = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs        = copyToDevice(absolute,   nc);
        d_mem        = copyToDevice(membrane,   nc);
        FF_REGFIELD_LAUNCH(diag_membrane,
            out, d_size, d_stride_out, d_vx, d_abs, d_mem)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_vx, d_abs, d_mem);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_vx, d_abs, d_mem);
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_inp = nullptr;
    reduce_t * d_vx = nullptr, * d_abs = nullptr, * d_mem = nullptr, * d_ben = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_inp = copyToDevice(stride_inp, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs        = copyToDevice(absolute,   nc);
        d_mem        = copyToDevice(membrane,   nc);
        d_ben        = copyToDevice(bending,    nc);
        FF_REGFIELD_LAUNCH(matvec_bending,
            out, inp, d_size, d_stride_out, d_stride_inp, d_vx, d_abs, d_mem, d_ben)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_inp, d_vx, d_abs, d_mem, d_ben);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_inp, d_vx, d_abs, d_mem, d_ben);
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr;
    reduce_t * d_vx = nullptr, * d_abs = nullptr, * d_mem = nullptr, * d_ben = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs        = copyToDevice(absolute,   nc);
        d_mem        = copyToDevice(membrane,   nc);
        d_ben        = copyToDevice(bending,    nc);
        FF_REGFIELD_LAUNCH(diag_bending,
            out, d_size, d_stride_out, d_vx, d_abs, d_mem, d_ben)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_vx, d_abs, d_mem, d_ben);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_vx, d_abs, d_mem, d_ben);
}

// --- KERNEL launchers ------------------------------------------------
//
// Materialise the per-channel Toeplitz stencil. The device kernels loop over
// the *batch* dims only (one centred stencil per batch element), so the launch
// grid is sized on prod(size, nbatch). kernel_absolute takes no voxel_size
// (the L2 stencil is scale-free).

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void kernel_absolute(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride_out,
    const reduce_t   * absolute,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nbatch);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr;
    reduce_t * d_abs = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_abs        = copyToDevice(absolute,   nc);
        FF_REGFIELD_LAUNCH(kernel_absolute,
            out, d_size, d_stride_out, d_abs)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_abs);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_abs);
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nbatch);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr;
    reduce_t * d_vx = nullptr, * d_abs = nullptr, * d_mem = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs        = copyToDevice(absolute,   nc);
        d_mem        = copyToDevice(membrane,   nc);
        FF_REGFIELD_LAUNCH(kernel_membrane,
            out, d_size, d_stride_out, d_vx, d_abs, d_mem)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_vx, d_abs, d_mem);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_vx, d_abs, d_mem);
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nbatch);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr;
    reduce_t * d_vx = nullptr, * d_abs = nullptr, * d_mem = nullptr, * d_ben = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs        = copyToDevice(absolute,   nc);
        d_mem        = copyToDevice(membrane,   nc);
        d_ben        = copyToDevice(bending,    nc);
        FF_REGFIELD_LAUNCH(kernel_bending,
            out, d_size, d_stride_out, d_vx, d_abs, d_mem, d_ben)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_vx, d_abs, d_mem, d_ben);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_vx, d_abs, d_mem, d_ben);
}

// --- RELAX launchers -------------------------------------------------
//
// Gauss-Seidel relaxation over a red-black colouring (patch1: 2 colours for
// membrane, patch3: 3^ndim colours for bending). Each device-kernel launch
// updates one colour; the host loops colours x nb_iter with a single counter
// `col`, which the kernel folds via patch1/patch3. The relax device kernels
// overwrite `sol` in place, so — unlike matvec/diag/kernel — they carry no
// `op` template param; hence a dedicated dispatch macro.

#define FF_REGFIELD_LAUNCH_RELAX_C(KERN, NB, ...)                              \
    switch (nc) {                                                             \
        case 1: KERN<NB, ndim, 1, reduce_t, scalar_t, offset_t, BOUND...>     \
                    <<<blocks, CUDA_NUM_THREADS, 0, stream>>>(bnd, __VA_ARGS__); break; \
        case 2: KERN<NB, ndim, 2, reduce_t, scalar_t, offset_t, BOUND...>     \
                    <<<blocks, CUDA_NUM_THREADS, 0, stream>>>(bnd, __VA_ARGS__); break; \
        case 3: KERN<NB, ndim, 3, reduce_t, scalar_t, offset_t, BOUND...>     \
                    <<<blocks, CUDA_NUM_THREADS, 0, stream>>>(bnd, __VA_ARGS__); break; \
        default: throw std::logic_error(                                      \
            "ff::cuda::reg_field: channel count outside [1, 3] is not "       \
            "supported by the CUDA relax launcher");                          \
    }

#define FF_REGFIELD_LAUNCH_RELAX(KERN, ...)                                    \
    switch (nbatch) {                                                         \
        case 0: FF_REGFIELD_LAUNCH_RELAX_C(KERN, 0, __VA_ARGS__); break;      \
        case 1: FF_REGFIELD_LAUNCH_RELAX_C(KERN, 1, __VA_ARGS__); break;      \
        default: throw std::logic_error(                                      \
            "ff::cuda::reg_field: nbatch > 1 is not supported by the CUDA "   \
            "relax launcher");                                                \
    }

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void relax_membrane_(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * sol,
    const scalar_t   * hes,
    const scalar_t   * grd,
    const offset_t   * size,
    const offset_t   * stride_sol,
    const offset_t   * stride_hes,
    const offset_t   * stride_grd,
    const reduce_t   * voxel_size,
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          int          nb_iter,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    const offset_t ncol   = static_cast<offset_t>(2);   /* patch1 */
    offset_t * d_size=nullptr,* d_ss=nullptr,* d_sh=nullptr,* d_sg=nullptr;
    reduce_t * d_vx=nullptr,* d_abs=nullptr,* d_mem=nullptr;
    try {
        d_size = copyToDevice(size,       nall + 1);
        d_ss   = copyToDevice(stride_sol, nall + 1);
        d_sh   = copyToDevice(stride_hes, nall + 1);
        d_sg   = copyToDevice(stride_grd, nall + 1);
        d_vx   = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs  = copyToDevice(absolute,   nc);
        d_mem  = copyToDevice(membrane,   nc);
        for (offset_t col = 0; col < ncol * nb_iter; ++col)
            FF_REGFIELD_LAUNCH_RELAX(relax_membrane_,
                sol, hes, grd, d_size, d_ss, d_sh, d_sg, d_vx,
                d_abs, d_mem, static_cast<int>(col))
    } catch (const std::exception &) {
        freeDevice(d_size, d_ss, d_sh, d_sg, d_vx, d_abs, d_mem); throw;
    }
    freeDevice(d_size, d_ss, d_sh, d_sg, d_vx, d_abs, d_mem);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void relax_bending_(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * sol,
    const scalar_t   * hes,
    const scalar_t   * grd,
    const offset_t   * size,
    const offset_t   * stride_sol,
    const offset_t   * stride_hes,
    const offset_t   * stride_grd,
    const reduce_t   * voxel_size,
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          int          nb_iter,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    const offset_t ncol   = pow<ndim>(static_cast<offset_t>(3)); /* patch3 */
    offset_t * d_size=nullptr,* d_ss=nullptr,* d_sh=nullptr,* d_sg=nullptr;
    reduce_t * d_vx=nullptr,* d_abs=nullptr,* d_mem=nullptr,* d_ben=nullptr;
    try {
        d_size = copyToDevice(size,       nall + 1);
        d_ss   = copyToDevice(stride_sol, nall + 1);
        d_sh   = copyToDevice(stride_hes, nall + 1);
        d_sg   = copyToDevice(stride_grd, nall + 1);
        d_vx   = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs  = copyToDevice(absolute,   nc);
        d_mem  = copyToDevice(membrane,   nc);
        d_ben  = copyToDevice(bending,    nc);
        for (offset_t col = 0; col < ncol * nb_iter; ++col)
            FF_REGFIELD_LAUNCH_RELAX(relax_bending_,
                sol, hes, grd, d_size, d_ss, d_sh, d_sg, d_vx,
                d_abs, d_mem, d_ben, static_cast<int>(col))
    } catch (const std::exception &) {
        freeDevice(d_size, d_ss, d_sh, d_sg, d_vx, d_abs, d_mem, d_ben); throw;
    }
    freeDevice(d_size, d_ss, d_sh, d_sg, d_vx, d_abs, d_mem, d_ben);
}

// --- RLS / JRLS launchers ---------------------------------------------
//
// The RLS/JRLS device kernels add a per-voxel weight `wgt` on top of the
// plain absolute/membrane/bending operators (reweighted least squares).
// Their (nbatch, ndim, C[, op]) template shapes match the plain kernels
// exactly, so the FF_REGFIELD_LAUNCH(_C) / FF_REGFIELD_LAUNCH_RELAX(_C)
// macros above are reused as-is -- no new dispatch macro is needed.
//
// RLS: `wgt` is a single weight shared across all C channels, but its
// stride vector still spans the full [*batch, *spatial, C] range (nall+1
// entries, mirroring stride_out/stride_inp) -- the device kernels
// fillfrom<nall+1>(...) it.
//
// JRLS: `wgt` varies per channel like `out`/`inp`, and its stride vector is
// only nall entries (no explicit channel-stride slot) -- the device kernels
// fillfrom<nall>(...) it. Copy one fewer element to device accordingly.
//
// Absolute has no neighbour coupling (it is a pointwise/diagonal operator),
// so relax_absolute_{rls,jrls}_ never colour-select via patch1/patch3 (the
// `niter` kernel parameter is accepted but unused by those two kernels);
// the launchers below still loop `nb_iter` times with a single colour
// (ncol=1) to keep the same nb_iter-driven public API as their membrane/
// bending siblings.
//
// NOTE: matvec_membrane_{rls,jrls} carry an extra `bending` coefficient
// parameter in the device-kernel signature that the kernel body never
// reads (make_kernel_membrane_rls only consumes absolute/membrane) -- this
// is a pre-existing property of those two kernels (not modified here); the
// launchers below forward it unchanged to match the real signature.

// --- ABSOLUTE + RLS -----------------------------------------------------

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void matvec_absolute_rls(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const offset_t   * stride_wgt,
    const reduce_t   * absolute,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_inp = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_abs  = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_inp = copyToDevice(stride_inp, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall + 1);
        d_abs        = copyToDevice(absolute,   nc);
        FF_REGFIELD_LAUNCH(matvec_absolute_rls,
            out, inp, wgt, d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_abs)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_abs);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_abs);
}

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void diag_absolute_rls(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_wgt,
    const reduce_t   * absolute,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_abs  = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall + 1);
        d_abs        = copyToDevice(absolute,   nc);
        FF_REGFIELD_LAUNCH(diag_absolute_rls,
            out, wgt, d_size, d_stride_out, d_stride_wgt, d_abs)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_wgt, d_abs);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_wgt, d_abs);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void relax_absolute_rls_(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * sol,
    const scalar_t   * hes,
    const scalar_t   * grd,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_sol,
    const offset_t   * stride_hes,
    const offset_t   * stride_grd,
    const offset_t   * stride_wgt,
    const reduce_t   * absolute,
          int          nb_iter,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    const offset_t ncol   = static_cast<offset_t>(1);  /* no neighbour coupling: single colour */
    offset_t * d_size=nullptr,* d_ss=nullptr,* d_sh=nullptr,* d_sg=nullptr,* d_sw=nullptr;
    reduce_t * d_abs=nullptr;
    try {
        d_size = copyToDevice(size,       nall + 1);
        d_ss   = copyToDevice(stride_sol, nall + 1);
        d_sh   = copyToDevice(stride_hes, nall + 1);
        d_sg   = copyToDevice(stride_grd, nall + 1);
        d_sw   = copyToDevice(stride_wgt, nall + 1);
        d_abs  = copyToDevice(absolute,   nc);
        for (offset_t col = 0; col < ncol * nb_iter; ++col)
            FF_REGFIELD_LAUNCH_RELAX(relax_absolute_rls_,
                sol, hes, grd, wgt, d_size, d_ss, d_sh, d_sg, d_sw, d_abs, static_cast<int>(col))
    } catch (const std::exception &) {
        freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_abs); throw;
    }
    freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_abs);
}

// --- ABSOLUTE + JRLS ----------------------------------------------------

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void matvec_absolute_jrls(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const offset_t   * stride_wgt,
    const reduce_t   * absolute,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_inp = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_abs  = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_inp = copyToDevice(stride_inp, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall);
        d_abs        = copyToDevice(absolute,   nc);
        FF_REGFIELD_LAUNCH(matvec_absolute_jrls,
            out, inp, wgt, d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_abs)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_abs);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_abs);
}

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void diag_absolute_jrls(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_wgt,
    const reduce_t   * absolute,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_abs  = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall);
        d_abs        = copyToDevice(absolute,   nc);
        FF_REGFIELD_LAUNCH(diag_absolute_jrls,
            out, wgt, d_size, d_stride_out, d_stride_wgt, d_abs)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_wgt, d_abs);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_wgt, d_abs);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void relax_absolute_jrls_(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * sol,
    const scalar_t   * hes,
    const scalar_t   * grd,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_sol,
    const offset_t   * stride_hes,
    const offset_t   * stride_grd,
    const offset_t   * stride_wgt,
    const reduce_t   * absolute,
          int          nb_iter,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    const offset_t ncol   = static_cast<offset_t>(1);  /* no neighbour coupling: single colour */
    offset_t * d_size=nullptr,* d_ss=nullptr,* d_sh=nullptr,* d_sg=nullptr,* d_sw=nullptr;
    reduce_t * d_abs=nullptr;
    try {
        d_size = copyToDevice(size,       nall + 1);
        d_ss   = copyToDevice(stride_sol, nall + 1);
        d_sh   = copyToDevice(stride_hes, nall + 1);
        d_sg   = copyToDevice(stride_grd, nall + 1);
        d_sw   = copyToDevice(stride_wgt, nall);
        d_abs  = copyToDevice(absolute,   nc);
        for (offset_t col = 0; col < ncol * nb_iter; ++col)
            FF_REGFIELD_LAUNCH_RELAX(relax_absolute_jrls_,
                sol, hes, grd, wgt, d_size, d_ss, d_sh, d_sg, d_sw, d_abs, static_cast<int>(col))
    } catch (const std::exception &) {
        freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_abs); throw;
    }
    freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_abs);
}

// --- MEMBRANE + RLS ------------------------------------------------------

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void matvec_membrane_rls(
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_inp = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_vx = nullptr, * d_abs = nullptr, * d_mem = nullptr, * d_ben = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_inp = copyToDevice(stride_inp, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs        = copyToDevice(absolute,   nc);
        d_mem        = copyToDevice(membrane,   nc);
        d_ben        = copyToDevice(bending,    nc);
        FF_REGFIELD_LAUNCH(matvec_membrane_rls,
            out, inp, wgt, d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx, d_abs, d_mem, d_ben)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx, d_abs, d_mem, d_ben);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx, d_abs, d_mem, d_ben);
}

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void diag_membrane_rls(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_vx = nullptr, * d_abs = nullptr, * d_mem = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs        = copyToDevice(absolute,   nc);
        d_mem        = copyToDevice(membrane,   nc);
        FF_REGFIELD_LAUNCH(diag_membrane_rls,
            out, wgt, d_size, d_stride_out, d_stride_wgt, d_vx, d_abs, d_mem)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_wgt, d_vx, d_abs, d_mem);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_wgt, d_vx, d_abs, d_mem);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void relax_membrane_rls_(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * sol,
    const scalar_t   * hes,
    const scalar_t   * grd,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_sol,
    const offset_t   * stride_hes,
    const offset_t   * stride_grd,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          int          nb_iter,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    const offset_t ncol   = static_cast<offset_t>(2);   /* patch1 */
    offset_t * d_size=nullptr,* d_ss=nullptr,* d_sh=nullptr,* d_sg=nullptr,* d_sw=nullptr;
    reduce_t * d_vx=nullptr,* d_abs=nullptr,* d_mem=nullptr;
    try {
        d_size = copyToDevice(size,       nall + 1);
        d_ss   = copyToDevice(stride_sol, nall + 1);
        d_sh   = copyToDevice(stride_hes, nall + 1);
        d_sg   = copyToDevice(stride_grd, nall + 1);
        d_sw   = copyToDevice(stride_wgt, nall + 1);
        d_vx   = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs  = copyToDevice(absolute,   nc);
        d_mem  = copyToDevice(membrane,   nc);
        for (offset_t col = 0; col < ncol * nb_iter; ++col)
            FF_REGFIELD_LAUNCH_RELAX(relax_membrane_rls_,
                sol, hes, grd, wgt, d_size, d_ss, d_sh, d_sg, d_sw, d_vx,
                d_abs, d_mem, static_cast<int>(col))
    } catch (const std::exception &) {
        freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_vx, d_abs, d_mem); throw;
    }
    freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_vx, d_abs, d_mem);
}

// --- MEMBRANE + JRLS -----------------------------------------------------

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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_inp = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_vx = nullptr, * d_abs = nullptr, * d_mem = nullptr, * d_ben = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_inp = copyToDevice(stride_inp, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs        = copyToDevice(absolute,   nc);
        d_mem        = copyToDevice(membrane,   nc);
        d_ben        = copyToDevice(bending,    nc);
        FF_REGFIELD_LAUNCH(matvec_membrane_jrls,
            out, inp, wgt, d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx, d_abs, d_mem, d_ben)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx, d_abs, d_mem, d_ben);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx, d_abs, d_mem, d_ben);
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_vx = nullptr, * d_abs = nullptr, * d_mem = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs        = copyToDevice(absolute,   nc);
        d_mem        = copyToDevice(membrane,   nc);
        FF_REGFIELD_LAUNCH(diag_membrane_jrls,
            out, wgt, d_size, d_stride_out, d_stride_wgt, d_vx, d_abs, d_mem)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_wgt, d_vx, d_abs, d_mem);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_wgt, d_vx, d_abs, d_mem);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void relax_membrane_jrls_(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * sol,
    const scalar_t   * hes,
    const scalar_t   * grd,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_sol,
    const offset_t   * stride_hes,
    const offset_t   * stride_grd,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          int          nb_iter,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    const offset_t ncol   = static_cast<offset_t>(2);   /* patch1 */
    offset_t * d_size=nullptr,* d_ss=nullptr,* d_sh=nullptr,* d_sg=nullptr,* d_sw=nullptr;
    reduce_t * d_vx=nullptr,* d_abs=nullptr,* d_mem=nullptr;
    try {
        d_size = copyToDevice(size,       nall + 1);
        d_ss   = copyToDevice(stride_sol, nall + 1);
        d_sh   = copyToDevice(stride_hes, nall + 1);
        d_sg   = copyToDevice(stride_grd, nall + 1);
        d_sw   = copyToDevice(stride_wgt, nall);
        d_vx   = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs  = copyToDevice(absolute,   nc);
        d_mem  = copyToDevice(membrane,   nc);
        for (offset_t col = 0; col < ncol * nb_iter; ++col)
            FF_REGFIELD_LAUNCH_RELAX(relax_membrane_jrls_,
                sol, hes, grd, wgt, d_size, d_ss, d_sh, d_sg, d_sw, d_vx,
                d_abs, d_mem, static_cast<int>(col))
    } catch (const std::exception &) {
        freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_vx, d_abs, d_mem); throw;
    }
    freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_vx, d_abs, d_mem);
}

// --- BENDING + RLS -------------------------------------------------------

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void matvec_bending_rls(
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_inp = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_vx = nullptr, * d_abs = nullptr, * d_mem = nullptr, * d_ben = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_inp = copyToDevice(stride_inp, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs        = copyToDevice(absolute,   nc);
        d_mem        = copyToDevice(membrane,   nc);
        d_ben        = copyToDevice(bending,    nc);
        FF_REGFIELD_LAUNCH(matvec_bending_rls,
            out, inp, wgt, d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx, d_abs, d_mem, d_ben)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx, d_abs, d_mem, d_ben);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx, d_abs, d_mem, d_ben);
}

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void diag_bending_rls(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_vx = nullptr, * d_abs = nullptr, * d_mem = nullptr, * d_ben = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall + 1);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs        = copyToDevice(absolute,   nc);
        d_mem        = copyToDevice(membrane,   nc);
        d_ben        = copyToDevice(bending,    nc);
        FF_REGFIELD_LAUNCH(diag_bending_rls,
            out, wgt, d_size, d_stride_out, d_stride_wgt, d_vx, d_abs, d_mem, d_ben)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_wgt, d_vx, d_abs, d_mem, d_ben);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_wgt, d_vx, d_abs, d_mem, d_ben);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void relax_bending_rls_(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * sol,
    const scalar_t   * hes,
    const scalar_t   * grd,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_sol,
    const offset_t   * stride_hes,
    const offset_t   * stride_grd,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          int          nb_iter,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    const offset_t ncol   = pow<ndim>(static_cast<offset_t>(3)); /* patch3 */
    offset_t * d_size=nullptr,* d_ss=nullptr,* d_sh=nullptr,* d_sg=nullptr,* d_sw=nullptr;
    reduce_t * d_vx=nullptr,* d_abs=nullptr,* d_mem=nullptr,* d_ben=nullptr;
    try {
        d_size = copyToDevice(size,       nall + 1);
        d_ss   = copyToDevice(stride_sol, nall + 1);
        d_sh   = copyToDevice(stride_hes, nall + 1);
        d_sg   = copyToDevice(stride_grd, nall + 1);
        d_sw   = copyToDevice(stride_wgt, nall + 1);
        d_vx   = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs  = copyToDevice(absolute,   nc);
        d_mem  = copyToDevice(membrane,   nc);
        d_ben  = copyToDevice(bending,    nc);
        for (offset_t col = 0; col < ncol * nb_iter; ++col)
            FF_REGFIELD_LAUNCH_RELAX(relax_bending_rls_,
                sol, hes, grd, wgt, d_size, d_ss, d_sh, d_sg, d_sw, d_vx,
                d_abs, d_mem, d_ben, static_cast<int>(col))
    } catch (const std::exception &) {
        freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_vx, d_abs, d_mem, d_ben); throw;
    }
    freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_vx, d_abs, d_mem, d_ben);
}

// --- BENDING + JRLS ------------------------------------------------------

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void matvec_bending_jrls(
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_inp = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_vx = nullptr, * d_abs = nullptr, * d_mem = nullptr, * d_ben = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_inp = copyToDevice(stride_inp, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs        = copyToDevice(absolute,   nc);
        d_mem        = copyToDevice(membrane,   nc);
        d_ben        = copyToDevice(bending,    nc);
        FF_REGFIELD_LAUNCH(matvec_bending_jrls,
            out, inp, wgt, d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx, d_abs, d_mem, d_ben)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx, d_abs, d_mem, d_ben);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_inp, d_stride_wgt, d_vx, d_abs, d_mem, d_ben);
}

template <int ndim, char op,
          typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void diag_bending_jrls(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    offset_t * d_size = nullptr, * d_stride_out = nullptr, * d_stride_wgt = nullptr;
    reduce_t * d_vx = nullptr, * d_abs = nullptr, * d_mem = nullptr, * d_ben = nullptr;
    try {
        d_size       = copyToDevice(size,       nall + 1);
        d_stride_out = copyToDevice(stride_out, nall + 1);
        d_stride_wgt = copyToDevice(stride_wgt, nall);
        d_vx         = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs        = copyToDevice(absolute,   nc);
        d_mem        = copyToDevice(membrane,   nc);
        d_ben        = copyToDevice(bending,    nc);
        FF_REGFIELD_LAUNCH(diag_bending_jrls,
            out, wgt, d_size, d_stride_out, d_stride_wgt, d_vx, d_abs, d_mem, d_ben)
    } catch (const std::exception &) {
        freeDevice(d_size, d_stride_out, d_stride_wgt, d_vx, d_abs, d_mem, d_ben);
        throw;
    }
    freeDevice(d_size, d_stride_out, d_stride_wgt, d_vx, d_abs, d_mem, d_ben);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
FF_CUHOST void relax_bending_jrls_(
    const bound::BoundVec & bnd,
          offset_t     nbatch,
          scalar_t   * sol,
    const scalar_t   * hes,
    const scalar_t   * grd,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_sol,
    const offset_t   * stride_hes,
    const offset_t   * stride_grd,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          int          nb_iter,
          cudaStream_t stream)
{
    const offset_t nall   = nbatch + ndim;
    const offset_t nc     = size[nall];
    const offset_t numel  = prod(size, nall);
    const int      blocks = GET_BLOCKS(numel);
    const offset_t ncol   = pow<ndim>(static_cast<offset_t>(3)); /* patch3 */
    offset_t * d_size=nullptr,* d_ss=nullptr,* d_sh=nullptr,* d_sg=nullptr,* d_sw=nullptr;
    reduce_t * d_vx=nullptr,* d_abs=nullptr,* d_mem=nullptr,* d_ben=nullptr;
    try {
        d_size = copyToDevice(size,       nall + 1);
        d_ss   = copyToDevice(stride_sol, nall + 1);
        d_sh   = copyToDevice(stride_hes, nall + 1);
        d_sg   = copyToDevice(stride_grd, nall + 1);
        d_sw   = copyToDevice(stride_wgt, nall);
        d_vx   = copyToDevice(voxel_size, static_cast<offset_t>(ndim));
        d_abs  = copyToDevice(absolute,   nc);
        d_mem  = copyToDevice(membrane,   nc);
        d_ben  = copyToDevice(bending,    nc);
        for (offset_t col = 0; col < ncol * nb_iter; ++col)
            FF_REGFIELD_LAUNCH_RELAX(relax_bending_jrls_,
                sol, hes, grd, wgt, d_size, d_ss, d_sh, d_sg, d_sw, d_vx,
                d_abs, d_mem, d_ben, static_cast<int>(col))
    } catch (const std::exception &) {
        freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_vx, d_abs, d_mem, d_ben); throw;
    }
    freeDevice(d_size, d_ss, d_sh, d_sg, d_sw, d_vx, d_abs, d_mem, d_ben);
}

#undef FF_REGFIELD_LAUNCH_RELAX
#undef FF_REGFIELD_LAUNCH_RELAX_C
#undef FF_REGFIELD_LAUNCH
#undef FF_REGFIELD_LAUNCH_C

FF_NAMESPACE_END(reg_field)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
