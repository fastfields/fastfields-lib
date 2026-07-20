#ifndef FF_REGULARISERS_FIELD_CPU
#define FF_REGULARISERS_FIELD_CPU
#include <stdexcept>
#include "kernels/cuda_switch.h"
#include "kernels/bounds.h"
#include "kernels/utils.h"
#include "kernels/batch.h"
#include "kernels/parallel.h"
#include "kernels/regularisers/field.h"
#include "kernels/posdef.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_field)

//======================================================================
//                              ABSOLUTE
//======================================================================

// --- ABSOLUTE: matvec -----------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_absolute(
          offset_t   nbatch,
          scalar_t * out,            // (*batch, *spatial, channels) tensor
    const scalar_t * inp,            // (*batch, *spatial, channels) tensor
    const offset_t * size,           // [*batch, *spatial, channels] vector
    const offset_t * stride_out,     // [*batch, *spatial, channels] vector
    const offset_t * stride_inp,     // [*batch, *spatial, channels] vector
    const reduce_t * absolute
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t isc    = stride_inp[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);  // no outer loop across channels

    // compute kernel
    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t inp_offset = index2offset(i, nall, size, stride_inp);
        offset_t out_offset = index2offset(i, nall, size, stride_out);

        Impl::template matvec_absolute<set>(
            out + out_offset, inp + inp_offset, osc, isc, kernel, nc);
    }});
    delete[] kernel;
}

// --- ABSOLUTE: kernel ------------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void kernel_absolute(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride,        // [*batch, *spatial, channels] vector
    const reduce_t * absolute
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t nall   = nbatch + ndim;
    offset_t sc     = stride[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nbatch);  // loop across batch only

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);

    offset_t offset = center_offset<ndim>(size+nbatch, stride+nbatch);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset(i, nbatch, size, stride);
        out_offset += offset;

        Impl::template kernel_absolute<set>(out + out_offset, sc, kernel, nc);
    }});
    delete[] kernel;
}

// --- ABSOLUTE: diagonal ----------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_absolute(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride,        // [*batch, *spatial, channels] vector
    const reduce_t * absolute
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    offset_t nall   = nbatch + ndim;
    offset_t sc     = stride[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);  // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim>(i, nall, size, stride, loc);

        Impl::template diag_absolute<set>(out + out_offset, sc, kernel, nc);
    }});
    delete[] kernel;
}

//======================================================================
//                              MEMBRANE
//======================================================================

// --- MEMBRANE: matvec -----------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_membrane(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * inp,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_inp,    // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t isc    = stride_inp[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);  // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane(nc)];
    Impl::make_kernel_membrane(kernel, absolute, membrane, voxel_size, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim>(i, nall, size, stride_inp, loc);
        offset_t out_offset = index2offset(i, nall, size, stride_out);

        Impl::template matvec_membrane<set>(
            out + out_offset, inp + inp_offset,
            loc, size + nbatch, stride_inp + nbatch, osc, isc, kernel, nc);
    }});
    delete[] kernel;
}

// --- MEMBRANE: kernel ------------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void kernel_membrane(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride,        // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t sc     = stride[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nbatch);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane(nc)];
    Impl::make_fullkernel_membrane(kernel, absolute, membrane, voxel_size, nc);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset(i, nbatch, size, stride);
        out_offset += offset;

        Impl::template kernel_membrane<set>(
            out + out_offset, sc, stride + nbatch, kernel, nc);
    }});
    delete[] kernel;
}

// --- MEMBRANE: diagonal ----------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_membrane(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride,        // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall  = nbatch + ndim;
    offset_t sc    = stride[nall];
    offset_t nc     = size[nall];
    offset_t numel = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane(nc)];
    Impl::make_kernel_membrane(kernel, absolute, membrane, voxel_size, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim>(i, nall, size, stride, loc);

        Impl::template diag_membrane<set>(
            out + out_offset, sc, loc, size + nbatch, kernel, nc);
    }});
    delete[] kernel;
}

// --- MEMBRANE: relax -------------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_membrane_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
          int        niter=1
)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall  = nbatch + ndim;
    offset_t osc   = stride_sol[nall];
    offset_t hsc   = stride_hes[nall];
    offset_t gsc   = stride_grd[nall];
    offset_t nc    = size[nall];
    offset_t numel = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane(nc)];
    Impl::make_kernel_membrane(kernel, absolute, membrane, voxel_size, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size(nc);

    for (offset_t n=0; n<2*niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t * val  = new reduce_t[nc];
        scalar_t * diag = new reduce_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t sol_offset = index2offset_v2<ndim>(i, nall, size, stride_sol, loc);
            if (!patch1<ndim>(loc, n))
                continue;
            offset_t grd_offset = index2offset(i, nall, size, stride_grd);
            offset_t hes_offset = index2offset(i, nall, size, stride_hes);

            // gradient
            for (offset_t c=0; c<nc; ++c)
                val[c] = grd[grd_offset + gsc*c];

            // minus convolution
            Impl::template matvec_membrane<isub>(
                val, sol + sol_offset,
                loc, size + nbatch, stride_sol + nbatch,
                static_cast<offset_t>(1), osc, kernel, nc);

            // diagonal
            Impl::template diag_membrane<set>(
                diag, static_cast<offset_t>(1), loc, size + nbatch, kernel, nc);

            // sol += (hes + diag) \ (grad - conv(sol))
            PosDef::relax_(
                nc,
                Strided(sol + sol_offset, osc),
                StridedConst(hes + hes_offset, hsc),
                val, diag, buf, static_cast<reduce_t>(0)
            );
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    delete[] kernel;
}

//======================================================================
//                              BENDING
//======================================================================

// --- BENDING: matvec ------------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_bending(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, 0) tensor
    const scalar_t * inp,           // (*batch, *spatial, 0) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_out,    // [*batch, *spatial, 0] vector
    const offset_t * stride_inp,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall  = nbatch + ndim;
    offset_t osc   = stride_out[nall];
    offset_t isc   = stride_inp[nall];
    offset_t nc    = size[nall];
    offset_t numel = prod(size, nall);  // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending(nc)];
    Impl::make_kernel_bending(kernel, absolute, membrane, bending, voxel_size, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            offset_t loc[ndim];
            offset_t inp_offset = index2offset_v2<ndim>(i, nall, size, stride_inp, loc);
            offset_t out_offset = index2offset(i, nall, size, stride_out);

            Impl::template matvec_bending<set>(
                out + out_offset, inp + inp_offset,
                loc, size + nbatch, stride_inp + nbatch, osc, isc, kernel, nc);
        }
    });
    delete[] kernel;
}

// --- BENDING: kernel -------------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void kernel_bending(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride,        // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall  = nbatch + ndim;
    offset_t sc    = stride[nall];
    offset_t nc    = size[nall];
    offset_t numel = prod(size, nbatch);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending(nc)];
    Impl::make_fullkernel_bending(kernel, absolute, membrane, bending, voxel_size, nc);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset(i, nbatch, size, stride);
        out_offset += offset;

        Impl::template kernel_bending<set>(
            out + out_offset, sc, stride + nbatch, kernel, nc);
    }});
    delete[] kernel;
}

// --- BENDING: diagonal -----------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_bending(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride,        // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t sc     = stride[nall];
    offset_t nc    = size[nall];
    offset_t numel  = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending(nc)];
    Impl::make_kernel_bending(kernel, absolute, membrane, bending, voxel_size, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim>(i, nall, size, stride, loc);

        Impl::template diag_bending<set>(
            out + out_offset, sc, loc, size + nbatch, kernel, nc);
    }});
    delete[] kernel;
}

// --- BENDING: relax --------------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_bending_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending,
          int        niter=1)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t, 0>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall  = nbatch + ndim;
    offset_t osc   = stride_sol[nall];
    offset_t hsc   = stride_hes[nall];
    offset_t gsc   = stride_grd[nall];
    offset_t nc    = size[nall];
    offset_t numel = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending(nc)];
    Impl::make_kernel_bending(kernel, absolute, membrane, bending, voxel_size, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size(nc);

    for (offset_t n = 0; n < pow<ndim>(3)*niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t * val  = new scalar_t[nc];
        scalar_t * diag = new scalar_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t sol_offset = index2offset_v2<ndim>(i, nall, size, stride_sol, loc);
            if (!patch3<ndim>(loc, n))
                continue;
            offset_t grd_offset = index2offset(i, nall, size, stride_grd);
            offset_t hes_offset = index2offset(i, nall, size, stride_hes);

            // gradient
            for (offset_t c=0; c<nc; ++c)
                val[c] = grd[grd_offset + gsc*c];

            // minus convolution
            Impl::template matvec_bending<isub>(
                val, sol + sol_offset,
                loc, size + nbatch, stride_sol + nbatch,
                static_cast<offset_t>(1), osc, kernel, nc);

            // diagonal
            Impl::template diag_bending<set>(
                diag, static_cast<offset_t>(1), loc, size + nbatch, kernel, nc);

            // sol += (hes + diag) \ (grad - conv(sol))
            PosDef::relax_(
                nc,
                Strided(sol + sol_offset, osc),
                StridedConst(hes + hes_offset, hsc),
                val, diag, buf, static_cast<reduce_t>(0)
            );
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    delete[] kernel;
}

//======================================================================
//                           ABSOLUTE RLS
//======================================================================

// --- ABSOLUTE+RLS: matvec --------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_absolute_rls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, 0) tensor
    const scalar_t * inp,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_out,    // [*batch, *spatial, 0] vector
    const offset_t * stride_inp,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * absolute
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t isc    = stride_inp[nall];
    offset_t wsc    = stride_wgt[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);  // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t inp_offset = index2offset(i, nall, size, stride_inp);
        offset_t out_offset = index2offset(i, nall, size, stride_out);
        offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

        Impl::template matvec_absolute_rls<set>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            osc, isc, wsc, kernel, nc);
    }});
    delete[] kernel;
}

// --- ABSOLUTE+RLS: diagonal ------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_absolute_rls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, channels] vector
    const reduce_t * absolute
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t wsc    = stride_wgt[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset(i, nall, size, stride_out);
        offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

        Impl::template diag_absolute_rls<set>(
            out + out_offset, wgt + wgt_offset, osc, wsc, kernel, nc);
    }});
    delete[] kernel;
}

// --- ABSOLUTE+RLS: relax ---------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_absolute_rls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * absolute,
          int        niter=1
)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];
    offset_t wsc    = stride_wgt[nall];
    offset_t nc    = size[nall];
    offset_t numel  = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size(nc);

    for (offset_t n=0; n<niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        scalar_t * val  = new scalar_t[nc];
        scalar_t * diag = new scalar_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t sol_offset = index2offset(i, nall, size, stride_sol);
            offset_t grd_offset = index2offset(i, nall, size, stride_grd);
            offset_t hes_offset = index2offset(i, nall, size, stride_hes);
            offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

            // gradient
            for (offset_t c=0; c<nc; ++c)
                val[c] = grd[grd_offset + gsc*c];

            // minus convolution
            Impl::template matvec_absolute_rls<isub>(
                val, sol + sol_offset, wgt + wgt_offset,
                static_cast<offset_t>(1), osc, wsc, kernel, nc);

            // diagonal
            Impl::template diag_absolute_rls<set>(
                diag, wgt + wgt_offset,
                static_cast<offset_t>(1), wsc, kernel, nc);

            // sol += (hes + diag) \ (grad - conv(sol))
            PosDef::relax_(
                nc,
                Strided(sol + sol_offset, osc),
                StridedConst(hes + hes_offset, hsc),
                val, diag, buf, static_cast<reduce_t>(0)
            );
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    delete[] kernel;
}

//======================================================================
//                           ABSOLUTE JRLS
//======================================================================

// --- ABSOLUTE+JRLS: matvec -------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_absolute_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, 0) tensor
    const scalar_t * inp,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_out,    // [*batch, *spatial, 0] vector
    const offset_t * stride_inp,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * absolute
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t isc    = stride_inp[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);  // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t inp_offset = index2offset(i, nall, size, stride_inp);
        offset_t out_offset = index2offset(i, nall, size, stride_out);
        offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

        Impl::template matvec_absolute_jrls<set>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            osc, isc, kernel, nc);
    }});
    delete[] kernel;
}

// --- ABSOLUTE+JRLS: diagonal -----------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_absolute_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, channels] vector
    const reduce_t * absolute
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset(i, nall, size, stride_out);
        offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

        Impl::template diag_absolute_jrls<set>(
            out + out_offset, wgt + wgt_offset, osc, kernel, nc);
    }});
    delete[] kernel;
}

// --- ABSOLUTE+JRLS: relax --------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_absolute_jrls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * absolute,
          int        niter=1
)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t, 0>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size(nc);

    for (offset_t n=0; n<niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        scalar_t * val  = new scalar_t[nc];
        scalar_t * diag = new scalar_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t sol_offset = index2offset(i, nall, size, stride_sol);
            offset_t grd_offset = index2offset(i, nall, size, stride_grd);
            offset_t hes_offset = index2offset(i, nall, size, stride_hes);
            offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

            // gradient
            for (offset_t c=0; c<nc; ++c)
                val[c] = grd[grd_offset + gsc*c];

            // minus convolution
            Impl::template matvec_absolute_jrls<isub>(
                val, sol + sol_offset, wgt + wgt_offset,
                static_cast<offset_t>(1), osc, kernel, nc);

            // diagonal
            Impl::template diag_absolute_jrls<set>(
                diag, wgt + wgt_offset,
                static_cast<offset_t>(1), kernel, nc);

            // sol += (hes + diag) \ (grad - conv(sol))
            PosDef::relax_(
                nc,
                Strided(sol + sol_offset, osc),
                StridedConst(hes + hes_offset, hsc),
                val, diag, buf, static_cast<reduce_t>(0)
            );
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    delete[] kernel;
}

//======================================================================
//                           MEMBRANE RLS
//======================================================================

// --- MEMBRANE+RLS: matvec --------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_membrane_rls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, 0) tensor
    const scalar_t * inp,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_out,    // [*batch, *spatial, 0] vector
    const offset_t * stride_inp,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t isc    = stride_inp[nall];
    offset_t wsc    = stride_wgt[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);  // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane_rls(nc)];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim>(i, nall, size, stride_inp, loc);
        offset_t out_offset = index2offset(i, nall, size, stride_out);
        offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

        Impl::template matvec_membrane_rls<set>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
            osc, isc, wsc, kernel, nc);
    }});
    delete[] kernel;
}

// --- MEMBRANE+RLS: diagonal ------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_membrane_rls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t wsc    = stride_wgt[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane_rls(nc)];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim>(i, nall, size, stride_out, loc);
        offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

        Impl::template diag_membrane_rls<set>(
            out + out_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_wgt + nbatch, osc, wsc, kernel, nc);
    }});
    delete[] kernel;
}

// --- MEMBRANE+RLS: relax ---------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_membrane_rls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
          int        niter=1
)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t, 0>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];
    offset_t wsc    = stride_wgt[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane_rls(nc)];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size;

    for (offset_t n=0; n<2*niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t * val  = new scalar_t[nc];
        scalar_t * diag = new scalar_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t sol_offset = index2offset_v2<ndim>(i, nall, size, stride_sol, loc);
            if (!patch1<ndim>(loc, n))
                continue;
            offset_t grd_offset = index2offset(i, nall, size, stride_grd);
            offset_t hes_offset = index2offset(i, nall, size, stride_hes);
            offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

            // gradient
            for (offset_t c=0; c<nc; ++c)
                val[c] = grd[grd_offset + gsc*c];

            // minus convolution
            Impl::template matvec_membrane_rls<isub>(
                val, sol + sol_offset, wgt + wgt_offset,
                loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
                static_cast<offset_t>(1), osc, wsc, kernel, nc);

            // diagonal
            Impl::template diag_membrane_rls<set>(
                diag, wgt + wgt_offset, loc,
                size + nbatch, stride_wgt + nbatch,
                static_cast<offset_t>(1), wsc, kernel, nc);

            // sol += (hes + diag) \ (grad - conv(sol))
            PosDef::relax_(
                nc,
                Strided(sol + sol_offset, osc),
                StridedConst(hes + hes_offset, hsc),
                val, diag, buf, static_cast<reduce_t>(0)
            );
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    delete[] kernel;
}

//======================================================================
//                           MEMBRANE JRLS
//======================================================================

// --- MEMBRANE+JRLS: matvec -------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_membrane_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, 0) tensor
    const scalar_t * inp,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_out,    // [*batch, *spatial, 0] vector
    const offset_t * stride_inp,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t isc    = stride_inp[nall];
    offset_t wsc    = stride_wgt[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);  // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane_rls(nc)];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim>(i, nall, size, stride_inp, loc);
        offset_t out_offset = index2offset(i, nall, size, stride_out);
        offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

        Impl::template matvec_membrane_jrls<set>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
            osc, isc, kernel, nc);
    }});
    delete[] kernel;
}

// --- MEMBRANE+JRLS: diagonal -----------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_membrane_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane_rls(nc)];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim>(i, nall, size, stride_out, loc);
        offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

        Impl::template diag_membrane_jrls<set>(
            out + out_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_wgt + nbatch, osc, kernel, nc);
    }});
    delete[] kernel;
}

// --- MEMBRANE+JRLS: relax ---------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_membrane_jrls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
          int        niter=1
)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t, 0>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall  = nbatch + ndim;
    offset_t osc   = stride_sol[nall];
    offset_t hsc   = stride_hes[nall];
    offset_t gsc   = stride_grd[nall];
    offset_t nc    = size[nall];
    offset_t numel = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane_rls(nc)];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size(nc);

    for (offset_t n=0; n<2*niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t   loc[ndim];
        scalar_t * val  = new scalar_t[nc];
        scalar_t * diag = new scalar_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t sol_offset = index2offset_v2<ndim>(i, nall, size, stride_sol, loc);
            if (!patch1<ndim>(loc, n))
                continue;
            offset_t grd_offset = index2offset(i, nall, size, stride_grd);
            offset_t hes_offset = index2offset(i, nall, size, stride_hes);
            offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

            // gradient
            for (offset_t c=0; c<nc; ++c)
                val[c] = grd[grd_offset + gsc*c];

            // minus convolution
            Impl::template matvec_membrane_jrls<isub>(
                val, sol + sol_offset, wgt + wgt_offset,
                loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
                static_cast<offset_t>(1), osc, kernel, nc);

            // diagonal
            Impl::template diag_membrane_jrls<set>(
                diag, wgt + wgt_offset, loc,
                size + nbatch, stride_wgt + nbatch,
                static_cast<offset_t>(1), kernel, nc);

            // sol += (hes + diag) \ (grad - conv(sol))
            PosDef::relax_(
                nc,
                Strided(sol + sol_offset, osc),
                StridedConst(hes + hes_offset, hsc),
                val, diag, buf, static_cast<reduce_t>(0)
            );
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    delete[] kernel;
}

//======================================================================
//                           BENDING RLS
//======================================================================

// --- BENDING+RLS: matvec ---------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_bending_rls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, 0) tensor
    const scalar_t * inp,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_out,    // [*batch, *spatial, 0] vector
    const offset_t * stride_inp,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t isc    = stride_inp[nall];
    offset_t wsc    = stride_wgt[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);  // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending_rls(nc)];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim>(i, nall, size, stride_inp, loc);
        offset_t out_offset = index2offset(i, nall, size, stride_out);
        offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

        Impl::template matvec_bending_rls<set>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
            osc, isc, wsc, kernel, nc);
    }});
    delete[] kernel;
}

// --- BENDING+RLS: diagonal -------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_bending_rls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t wsc    = stride_wgt[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending_rls(nc)];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim>(i, nall, size, stride_out, loc);
        offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

        Impl::template diag_bending_rls<set>(
            out + out_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_wgt + nbatch, osc, wsc, kernel, nc);
    }});
    delete[] kernel;
}

// --- BENDING+RLS: relax ----------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_bending_rls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending,
          int        niter=1
)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall    = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];
    offset_t wsc    = stride_wgt[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending_rls(nc)];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size(nc);

    for (offset_t n=0; n<2*niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t * val  = new scalar_t[nc];
        scalar_t * diag = new scalar_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t sol_offset = index2offset_v2<ndim>(i, nall, size, stride_sol, loc);
            if (!patch3<ndim>(loc, n))
                continue;
            offset_t grd_offset = index2offset(i, nall, size, stride_grd);
            offset_t hes_offset = index2offset(i, nall, size, stride_hes);
            offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

            // gradient
            for (offset_t c=0; c<nc; ++c)
                val[c] = grd[grd_offset + gsc*c];

            // minus convolution
            Impl::template matvec_bending_rls<isub>(
                val, sol + sol_offset, wgt + wgt_offset,
                loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
                static_cast<offset_t>(1), osc, wsc, kernel, nc);

            // diagonal
            Impl::template diag_bending_rls<set>(
                diag, wgt + wgt_offset, loc,
                size + nbatch, stride_wgt + nbatch,
                static_cast<offset_t>(1), wsc, kernel, nc);

            // sol += (hes + diag) \ (grad - conv(sol))
            PosDef::relax_(
                nc,
                Strided(sol + sol_offset, osc),
                StridedConst(hes + hes_offset, hsc),
                val, diag, buf, static_cast<reduce_t>(0)
            );
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    delete[] kernel;
}

//======================================================================
//                           BENDING JRLS
//======================================================================

// --- BENDING+JRLS: matvec --------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_bending_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, 0) tensor
    const scalar_t * inp,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_out,    // [*batch, *spatial, 0] vector
    const offset_t * stride_inp,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t isc    = stride_inp[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);  // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending_rls(nc)];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t loc[ndim];
        offset_t inp_offset = index2offset_v2<ndim>(i, nall, size, stride_inp, loc);
        offset_t out_offset = index2offset(i, nall, size, stride_out);
        offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

        Impl::template matvec_bending_jrls<set>(
            out + out_offset, inp + inp_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
            osc, isc, kernel, nc);
    }});
    delete[] kernel;
}

// --- BENDING+JRLS: diagonal ------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_bending_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending_rls(nc)];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size, nc);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t loc[ndim];
        offset_t out_offset = index2offset_v2<ndim>(i, nall, size, stride_out, loc);
        offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

        Impl::template diag_bending_jrls<set>(
            out + out_offset, wgt + wgt_offset,
            loc, size + nbatch, stride_wgt + nbatch, osc, kernel, nc);
    }});
    delete[] kernel;
}

// --- BENDING+JRLS: relax ---------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_bending_jrls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending,
          int        niter=1
)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall  = nbatch + ndim;
    offset_t osc   = stride_sol[nall];
    offset_t hsc   = stride_hes[nall];
    offset_t gsc   = stride_grd[nall];
    offset_t nc    = size[nall];
    offset_t numel = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending_rls(nc)];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size(nc);

    for (offset_t n=0; n<2*niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t   loc[ndim];
        scalar_t * val  = new scalar_t[nc];
        scalar_t * diag = new scalar_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t sol_offset = index2offset_v2<ndim>(i, nall, size, stride_sol, loc);
            if (!patch3<ndim>(loc, n))
                continue;
            offset_t grd_offset = index2offset(i, nall, size, stride_grd);
            offset_t hes_offset = index2offset(i, nall, size, stride_hes);
            offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

            // gradient
            for (offset_t c=0; c<nc; ++c)
                val[c] = grd[grd_offset + gsc*c];

            // minus convolution
            Impl::template matvec_bending_jrls<isub>(
                val, sol + sol_offset, wgt + wgt_offset,
                loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
                static_cast<offset_t>(1), osc, kernel, nc);

            // diagonal
            Impl::template diag_bending_jrls<set>(
                diag, wgt + wgt_offset, loc,
                size + nbatch, stride_wgt + nbatch,
                static_cast<offset_t>(1), kernel, nc);

            // sol += (hes + diag) \ (grad - conv(sol))
            PosDef::relax_(
                nc,
                Strided(sol + sol_offset, osc),
                StridedConst(hes + hes_offset, hsc),
                val, diag, buf, static_cast<reduce_t>(0)
            );
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    delete[] kernel;
}

FF_NAMESPACE_END(reg_field)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_REGULARISERS_FIELD_CPU
