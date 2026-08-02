#ifndef FF_REGULARISERS_FLOW_CPU
#define FF_REGULARISERS_FLOW_CPU
#include <stdexcept>
#include <teeny/teeny.h>
#include "kernels/cuda_switch.h"
#include "kernels/bounds.h"
#include "kernels/utils.h"
#include "kernels/batch.h"
#include "kernels/parallel.h"
#include "kernels/regularisers/flow.h"
#include "kernels/posdef.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_flow)

//----------------------------------------------------------------------
//  op dispatch helper (issue #6a)
//----------------------------------------------------------------------
// Thread the compile-time op ('=', '+', '-') through a function-template
// wrapper. Mirrors the CUDA impl's `Op<op,scalar_t,reduce_t>::f` dispatch, but
// as a named function template because C++11 (unlike the CUDA/C++17 path) only
// accepts a *function id-expression* -- not a constexpr pointer variable -- as
// a non-type template argument of function-pointer type.
template <char op, typename scalar_t, typename reduce_t>
inline scalar_t & op_apply(scalar_t & out, const reduce_t & in)
{
    return Op<op, scalar_t, reduce_t>::f(out, in);
}

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
    const reduce_t * _voxel_size,    // [*spatial] vector
          reduce_t   absolute
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t isc  = stride_inp[nall];

    // compute kernel
    reduce_t kernel[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kernel, absolute, voxel_size);

    // pointwise: peel each (*batch,*spatial,C) tensor down to its per-voxel C cell.
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nvox = ao.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        auto ic = ai.template peel_front_at<-1>(i);
        Impl::template matvec_absolute<op_apply<op, scalar_t, reduce_t> >(
            oc.data(), ic.data(), osc, isc, kernel);
    }});
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
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size [ndim];   fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t sc     = stride[nall];

    reduce_t kernel[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kernel, absolute, voxel_size);

    offset_t offset = center_offset<ndim>(size+nbatch, stride+nbatch);

    // batch-only: peel each (*batch,*spatial) cell to its batch origin, write
    // the kernel at the spatial center (no per-voxel spatial decode needed).
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall), tny::copy_meta);
    const offset_t nbcell = ao.template size_front<-ndim>();

    parallel_for(0, nbcell, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        auto vo = ao.template peel_front_at<-ndim>(i);
        Impl::template kernel_absolute<op_apply<op, scalar_t, reduce_t> >(vo.data() + offset, sc, kernel);
    }});
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
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t sc   = stride[nall];

    reduce_t kernel[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kernel, absolute, voxel_size);

    // pointwise: peel to the per-voxel C cell (the diagonal is position-independent).
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nvox = ao.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            auto oc = ao.template peel_front_at<-1>(i);
            Impl::template diag_absolute<op_apply<op, scalar_t, reduce_t> >(oc.data(), sc, kernel);
        }
    });
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
          reduce_t   absolute,
          reduce_t   membrane
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t isc  = stride_inp[nall];

    reduce_t kernel[Impl::kernelsize_membrane];
    Impl::make_kernel_membrane(kernel, absolute, membrane, voxel_size);

    // stencil: peel out/inp volumes per batch cell, decode loc + spatial offset per voxel.
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
            auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
            auto ci = tny::peel_front<-1>(vi).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
            {
                Impl::template matvec_membrane<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), (*ci).data(),
                    v.index.data(), size + nbatch, stride_inp + nbatch, osc, isc, kernel);
                ++ci;
            }
        }
    });
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
          reduce_t   absolute,
          reduce_t   membrane
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t sc     = stride[nall];

    reduce_t kernel[Impl::kernelsize_membrane];
    Impl::make_fullkernel_membrane(kernel, absolute, membrane, voxel_size);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    // batch-only: peel each (*batch,*spatial) cell to its batch origin, write
    // the full kernel at the spatial center.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall), tny::copy_meta);
    const offset_t nbcell = ao.template size_front<-ndim>();

    parallel_for(0, nbcell, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            auto vo = ao.template peel_front_at<-ndim>(i);

            Impl::template kernel_membrane<op_apply<op, scalar_t, reduce_t> >(
                vo.data() + offset, sc, stride + nbatch, kernel);
        }
    });
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
          reduce_t   absolute,
          reduce_t   membrane
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t,  offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t sc   = stride[nall];

    reduce_t kernel[Impl::kernelsize_membrane];
    Impl::make_kernel_membrane(kernel, absolute, membrane, voxel_size);

    // stencil (boundary-aware diagonal): peel per batch cell, decode loc + offset per voxel.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
            for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
                Impl::template diag_membrane<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), sc, v.index.data(), size + nbatch, kernel);
        }
    });
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
          scalar_t * sol,           // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_sol,    // [*batch, *spatial, C] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          int        niter=1
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];

    reduce_t kernel[Impl::kernelsize_membrane];
    Impl::make_kernel_membrane(kernel, absolute, membrane, voxel_size);
    constexpr int DD = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    // Gauss-Seidel / patch-coloring relax, same math as before, but the
    // per-voxel offset now comes from peeling (*batch,*spatial) cells
    // instead of index2offset_v2: peel sol/hes/grd per batch cell (rank
    // nbatch+ndim carriers, no trailing channel needed -- only the batch
    // origin pointer matters here), caching across the inner loop like the
    // matvec/diag stencils above, then fold the spatial offset by hand.
    auto as = tny::as_anyrank(sol, size, stride_sol, static_cast<int>(nall), tny::copy_meta);
    auto ah = tny::as_anyrank(hes, size, stride_hes, static_cast<int>(nall), tny::copy_meta);
    auto ag = tny::as_anyrank(grd, size, stride_grd, static_cast<int>(nall), tny::copy_meta);

    offset_t osp[ndim]; offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) { osp[d] = size[nbatch + d]; nsp *= osp[d]; }
    const offset_t numel = as.template size_front<-ndim>() * nsp;    // no outer loop across channels

    for (offset_t n=0; n<2*niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t val[ndim], diag[ndim];
        reduce_t buf[DD ? DD : 1];
        offset_t cur_b = (nsp > 0) ? static_cast<offset_t>(start) / nsp : 0;
        auto vs = as.template peel_front_at<-ndim>(cur_b);
        auto vh = ah.template peel_front_at<-ndim>(cur_b);
        auto vg = ag.template peel_front_at<-ndim>(cur_b);
        for (offset_t i=start; i < end; ++i)
        {
            const offset_t b = (nsp > 0) ? i / nsp : 0;
            if (b != cur_b) {
                vs = as.template peel_front_at<-ndim>(b);
                vh = ah.template peel_front_at<-ndim>(b);
                vg = ag.template peel_front_at<-ndim>(b);
                cur_b = b;
            }
            offset_t sp = i - b * nsp, sol_offset = 0, hes_offset = 0, grd_offset = 0;
            for (int d = ndim - 1; d >= 0; --d) {
                const offset_t c = sp % osp[d]; sp /= osp[d]; loc[d] = c;
                sol_offset += c * stride_sol[nbatch + d];
                hes_offset += c * stride_hes[nbatch + d];
                grd_offset += c * stride_grd[nbatch + d];
            }
            if (!patch1<ndim>(loc, n))
                continue;

            scalar_t       * sol_ptr = vs.data() + sol_offset;
            const scalar_t * hes_ptr = vh.data() + hes_offset;
            const scalar_t * grd_ptr = vg.data() + grd_offset;

            // gradient
#           pragma unroll
            for (int d=0; d<ndim; ++d)
                val[d] = grd_ptr[gsc*d];

            // minus convolution
            Impl::template matvec_membrane<isub>(
                val, sol_ptr,
                loc, size + nbatch, stride_sol + nbatch,
                static_cast<offset_t>(1), osc, kernel);

            // diagonal
            Impl::template diag_membrane<set>(
                diag, static_cast<offset_t>(1), loc, size + nbatch, kernel);

            // sol += (hes + diag) \ (grad - conv(sol))
            PosDef::relax_(
                Strided(sol_ptr, osc),
                StridedConst(hes_ptr, hsc),
                val, diag, buf, static_cast<reduce_t>(0)
            );
        }
    });
    }
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
          scalar_t * out,           // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_out,    // [*batch, *spatial, C] vector
    const offset_t * stride_inp,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t absolute,
          reduce_t membrane,
          reduce_t bending
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t isc  = stride_inp[nall];

    reduce_t kernel[Impl::kernelsize_bending];
    Impl::make_kernel_bending(kernel, absolute, membrane, bending, voxel_size);

    // stencil: peel out/inp volumes per batch cell, decode loc + spatial offset per voxel.
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
            auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
            auto ci = tny::peel_front<-1>(vi).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
            {
                Impl::template matvec_bending<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), (*ci).data(),
                    v.index.data(), size + nbatch, stride_inp + nbatch, osc, isc, kernel);
                ++ci;
            }
        }
    });
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
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   bending
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall = nbatch + ndim;
    offset_t sc = stride[nall];

    reduce_t kernel[Impl::kernelsize_bending];
    Impl::make_fullkernel_bending(kernel, absolute, membrane, bending, voxel_size);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    // batch-only: peel each (*batch,*spatial) cell to its batch origin, write
    // the full kernel at the spatial center.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall), tny::copy_meta);
    const offset_t nbcell = ao.template size_front<-ndim>();

    parallel_for(0, nbcell, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            auto vo = ao.template peel_front_at<-ndim>(i);

            Impl::template kernel_bending<op_apply<op, scalar_t, reduce_t> >(
                vo.data() + offset, sc, stride + nbatch, kernel);
        }
    });
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
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   bending
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t sc   = stride[nall];

    reduce_t kernel[Impl::kernelsize_bending];
    Impl::make_kernel_bending(kernel, absolute, membrane, bending, voxel_size);

    // stencil (boundary-aware diagonal): peel per batch cell, decode loc + offset per voxel.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
            for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
                Impl::template diag_bending<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), sc, v.index.data(), size + nbatch, kernel);
        }
    });
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
          scalar_t * sol,           // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_sol,    // [*batch, *spatial, C] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   bending,
          int        niter=1
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall = nbatch + ndim;
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];

    reduce_t kernel[Impl::kernelsize_bending];
    Impl::make_kernel_bending(kernel, absolute, membrane, bending, voxel_size);
    constexpr int DD = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    // Gauss-Seidel / patch-coloring relax (see relax_membrane_ for the peel
    // rationale): peel sol/hes/grd per batch cell, cache across the inner
    // loop, fold the spatial offset by hand.
    auto as = tny::as_anyrank(sol, size, stride_sol, static_cast<int>(nall), tny::copy_meta);
    auto ah = tny::as_anyrank(hes, size, stride_hes, static_cast<int>(nall), tny::copy_meta);
    auto ag = tny::as_anyrank(grd, size, stride_grd, static_cast<int>(nall), tny::copy_meta);

    offset_t osp[ndim]; offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) { osp[d] = size[nbatch + d]; nsp *= osp[d]; }
    const offset_t numel = as.template size_front<-ndim>() * nsp;    // no outer loop across channels

    for (offset_t n = 0; n < pow<ndim>(3)*niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t val[ndim], diag[ndim];
        reduce_t buf[DD ? DD : 1];
        offset_t cur_b = (nsp > 0) ? static_cast<offset_t>(start) / nsp : 0;
        auto vs = as.template peel_front_at<-ndim>(cur_b);
        auto vh = ah.template peel_front_at<-ndim>(cur_b);
        auto vg = ag.template peel_front_at<-ndim>(cur_b);
        for (offset_t i=start; i < end; ++i)
        {
            const offset_t b = (nsp > 0) ? i / nsp : 0;
            if (b != cur_b) {
                vs = as.template peel_front_at<-ndim>(b);
                vh = ah.template peel_front_at<-ndim>(b);
                vg = ag.template peel_front_at<-ndim>(b);
                cur_b = b;
            }
            offset_t sp = i - b * nsp, sol_offset = 0, hes_offset = 0, grd_offset = 0;
            for (int d = ndim - 1; d >= 0; --d) {
                const offset_t c = sp % osp[d]; sp /= osp[d]; loc[d] = c;
                sol_offset += c * stride_sol[nbatch + d];
                hes_offset += c * stride_hes[nbatch + d];
                grd_offset += c * stride_grd[nbatch + d];
            }
            if (!patch3<ndim>(loc, n))
                continue;

            scalar_t       * sol_ptr = vs.data() + sol_offset;
            const scalar_t * hes_ptr = vh.data() + hes_offset;
            const scalar_t * grd_ptr = vg.data() + grd_offset;

            // gradient
#           pragma unroll
            for (int d=0; d<ndim; ++d)
                val[d] = grd_ptr[gsc*d];

            // minus convolution
            Impl::template matvec_bending<isub>(
                val, sol_ptr,
                loc, size + nbatch, stride_sol + nbatch,
                static_cast<offset_t>(1), osc, kernel);

            // diagonal
            Impl::template diag_bending<set>(
                diag, static_cast<offset_t>(1), loc, size + nbatch, kernel);

            // sol += (hes + diag) \ (grad - conv(sol))
            PosDef::relax_(
                Strided(sol_ptr, osc),
                StridedConst(hes_ptr, hsc),
                val, diag, buf, static_cast<reduce_t>(0)
            );
        }
    });
    }
}

//======================================================================
//                              LAME
//======================================================================

// --- LAME: matvec ---------------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_lame(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_out,    // [*batch, *spatial, C] vector
    const offset_t * stride_inp,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   shears,
          reduce_t   div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t isc    = stride_inp[nall];
    reduce_t kernel[Impl::kernelsize_lame];
    Impl::make_kernel_lame(kernel, absolute, membrane, shears, div, voxel_size);

    // stencil: peel out/inp volumes per batch cell, then let teeny's spatial
    // peel supply `loc` and the per-voxel base pointers (same shape as
    // matvec_membrane; replaces index2offset_v2).
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
            auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
            auto ci = tny::peel_front<-1>(vi).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
            {
                Impl::template matvec_lame<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), (*ci).data(),
                    v.index.data(), size + nbatch, stride_inp + nbatch, osc, isc, kernel);
                ++ci;
            }
        }
    });
}

// --- LAME: kernel ----------------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void kernel_lame(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, C, C) tensor
    const offset_t * size,          // [*batch, *spatial, C, C] vector
    const offset_t * stride,        // [*batch, *spatial, C, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   shears,
          reduce_t   div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall = nbatch + ndim;

    reduce_t kernel[Impl::kernelsize_lame];
    Impl::make_fullkernel_lame(kernel, absolute, membrane, shears, div, voxel_size);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    // batch-only: peel each (*batch,*spatial) cell to its batch origin, write
    // the full kernel at the spatial center.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall), tny::copy_meta);
    const offset_t nbcell = ao.template size_front<-ndim>();

    parallel_for(0, nbcell, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            auto vo = ao.template peel_front_at<-ndim>(i);

            Impl::template kernel_lame<op_apply<op, scalar_t, reduce_t> >(
                vo.data() + offset, stride + nall, stride + nbatch, kernel);
        }
    });
}

// --- LAME: diagonal --------------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_lame(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride,        // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   shears,
          reduce_t   div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t sc     = stride[nall];
    reduce_t kernel[Impl::kernelsize_lame];
    Impl::make_kernel_lame(kernel, absolute, membrane, shears, div, voxel_size);

    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
            for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
                Impl::template diag_lame<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), sc, v.index.data(), size + nbatch, kernel);
        }
    });
}
// --- LAME: relax -----------------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_lame_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_sol,    // [*batch, *spatial, C] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   shears,
          reduce_t   div,
          int        niter=1
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];

    reduce_t kernel[Impl::kernelsize_lame];
    Impl::make_kernel_lame(kernel, absolute, membrane, shears, div, voxel_size);
    constexpr int DD = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    // Gauss-Seidel / patch-coloring relax (see relax_membrane_ for the peel
    // rationale): peel sol/hes/grd per batch cell, cache across the inner
    // loop, fold the spatial offset by hand.
    auto as = tny::as_anyrank(sol, size, stride_sol, static_cast<int>(nall), tny::copy_meta);
    auto ah = tny::as_anyrank(hes, size, stride_hes, static_cast<int>(nall), tny::copy_meta);
    auto ag = tny::as_anyrank(grd, size, stride_grd, static_cast<int>(nall), tny::copy_meta);

    offset_t osp[ndim]; offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) { osp[d] = size[nbatch + d]; nsp *= osp[d]; }
    const offset_t numel = as.template size_front<-ndim>() * nsp;    // no outer loop across channels

    for (offset_t n = 0; n < pow<ndim>(2)*niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t val[ndim], diag[ndim];
        reduce_t buf[DD ? DD : 1];
        offset_t cur_b = (nsp > 0) ? static_cast<offset_t>(start) / nsp : 0;
        auto vs = as.template peel_front_at<-ndim>(cur_b);
        auto vh = ah.template peel_front_at<-ndim>(cur_b);
        auto vg = ag.template peel_front_at<-ndim>(cur_b);
        for (offset_t i=start; i < end; ++i)
        {
            const offset_t b = (nsp > 0) ? i / nsp : 0;
            if (b != cur_b) {
                vs = as.template peel_front_at<-ndim>(b);
                vh = ah.template peel_front_at<-ndim>(b);
                vg = ag.template peel_front_at<-ndim>(b);
                cur_b = b;
            }
            offset_t sp = i - b * nsp, sol_offset = 0, hes_offset = 0, grd_offset = 0;
            for (int d = ndim - 1; d >= 0; --d) {
                const offset_t c = sp % osp[d]; sp /= osp[d]; loc[d] = c;
                sol_offset += c * stride_sol[nbatch + d];
                hes_offset += c * stride_hes[nbatch + d];
                grd_offset += c * stride_grd[nbatch + d];
            }
            if (!patch2<ndim>(loc, n))
                continue;

            scalar_t       * sol_ptr = vs.data() + sol_offset;
            const scalar_t * hes_ptr = vh.data() + hes_offset;
            const scalar_t * grd_ptr = vg.data() + grd_offset;

            // gradient
#           pragma unroll
            for (int d=0; d<ndim; ++d)
                val[d] = grd_ptr[gsc*d];

            // minus convolution
            Impl::template matvec_lame<isub>(
                val, sol_ptr,
                loc, size + nbatch, stride_sol + nbatch,
                static_cast<offset_t>(1), osc, kernel);

            // diagonal
            Impl::template diag_lame<set>(
                diag, static_cast<offset_t>(1), loc, size + nbatch, kernel);

            // sol += (hes + diag) \ (grad - conv(sol))
            PosDef::relax_(
                Strided(sol_ptr, osc),
                StridedConst(hes_ptr, hsc),
                val, diag, buf, static_cast<reduce_t>(0)
            );
        }
    });
    }
}

//======================================================================
//                          LAME + BENDING
//======================================================================

// --- BENDING+LAME: matvec -------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_all(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_out,    // [*batch, *spatial, C] vector
    const offset_t * stride_inp,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   bending,
          reduce_t   shears,
          reduce_t   div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t isc  = stride_inp[nall];

    reduce_t kernel[Impl::kernelsize_all];
    Impl::make_kernel_all(kernel, absolute, membrane, bending, shears, div, voxel_size);

    // stencil: peel out/inp volumes per batch cell, decode loc + spatial offset per voxel.
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
            auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
            auto ci = tny::peel_front<-1>(vi).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
            {
                Impl::template matvec_all<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), (*ci).data(),
                    v.index.data(), size + nbatch, stride_inp + nbatch, osc, isc, kernel);
                ++ci;
            }
        }
    });
}

// --- BENDING+LAME: kernel --------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void kernel_all(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, C, C) tensor
    const offset_t * size,          // [*batch, *spatial, C, C] vector
    const offset_t * stride,        // [*batch, *spatial, C, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   bending,
          reduce_t   shears,
          reduce_t   div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall = nbatch + ndim;

    reduce_t kernel[Impl::kernelsize_all];
    Impl::make_fullkernel_all(kernel, absolute, membrane, bending, shears, div, voxel_size);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    // batch-only: peel each (*batch,*spatial) cell to its batch origin, write
    // the full kernel at the spatial center.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall), tny::copy_meta);
    const offset_t nbcell = ao.template size_front<-ndim>();

    parallel_for(0, nbcell, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            auto vo = ao.template peel_front_at<-ndim>(i);

            Impl::template kernel_all<op_apply<op, scalar_t, reduce_t> >(
                vo.data() + offset, stride + nall, stride + nbatch, kernel);
        }
    });
}

// --- BENDING+LAME: diagonal ------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_all(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride,        // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   bending,
          reduce_t   shears,
          reduce_t   div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t sc   = stride[nall];

    reduce_t kernel[Impl::kernelsize_all];
    Impl::make_kernel_all(kernel, absolute, membrane, bending, shears, div, voxel_size);

    // stencil (boundary-aware diagonal): peel per batch cell, decode loc + offset per voxel.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
            for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
                Impl::template diag_all<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), sc, v.index.data(), size + nbatch, kernel);
        }
    });
}

// --- BENDING+LAME: relax ---------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_all_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_sol,    // [*batch, *spatial, C] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   bending,
          reduce_t   shears,
          reduce_t   div,
          int        niter=1
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];

    reduce_t kernel[Impl::kernelsize_all];
    Impl::make_kernel_all(kernel, absolute, membrane, bending, shears, div, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    // Gauss-Seidel / patch-coloring relax (see relax_membrane_ for the peel
    // rationale): peel sol/hes/grd per batch cell, cache across the inner
    // loop, fold the spatial offset by hand.
    auto as = tny::as_anyrank(sol, size, stride_sol, static_cast<int>(nall), tny::copy_meta);
    auto ah = tny::as_anyrank(hes, size, stride_hes, static_cast<int>(nall), tny::copy_meta);
    auto ag = tny::as_anyrank(grd, size, stride_grd, static_cast<int>(nall), tny::copy_meta);

    offset_t osp[ndim]; offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) { osp[d] = size[nbatch + d]; nsp *= osp[d]; }
    const offset_t numel = as.template size_front<-ndim>() * nsp;    // no outer loop across channels

    for (offset_t n = 0; n < pow<ndim>(3)*niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t val[ndim], diag[ndim];
        reduce_t buf[CC ? CC : 1];
        offset_t cur_b = (nsp > 0) ? static_cast<offset_t>(start) / nsp : 0;
        auto vs = as.template peel_front_at<-ndim>(cur_b);
        auto vh = ah.template peel_front_at<-ndim>(cur_b);
        auto vg = ag.template peel_front_at<-ndim>(cur_b);
        for (offset_t i=start; i < end; ++i)
        {
            const offset_t b = (nsp > 0) ? i / nsp : 0;
            if (b != cur_b) {
                vs = as.template peel_front_at<-ndim>(b);
                vh = ah.template peel_front_at<-ndim>(b);
                vg = ag.template peel_front_at<-ndim>(b);
                cur_b = b;
            }
            offset_t sp = i - b * nsp, sol_offset = 0, hes_offset = 0, grd_offset = 0;
            for (int d = ndim - 1; d >= 0; --d) {
                const offset_t c = sp % osp[d]; sp /= osp[d]; loc[d] = c;
                sol_offset += c * stride_sol[nbatch + d];
                hes_offset += c * stride_hes[nbatch + d];
                grd_offset += c * stride_grd[nbatch + d];
            }
            if (!patch3<ndim>(loc, n))
                continue;

            scalar_t       * sol_ptr = vs.data() + sol_offset;
            const scalar_t * hes_ptr = vh.data() + hes_offset;
            const scalar_t * grd_ptr = vg.data() + grd_offset;

            // gradient
#           pragma unroll
            for (int d=0; d<ndim; ++d)
                val[d] = grd_ptr[gsc*d];

            // minus convolution
            Impl::template matvec_all<isub>(
                val, sol_ptr,
                loc, size + nbatch, stride_sol + nbatch,
                static_cast<offset_t>(1), osc, kernel);

            // diagonal
            Impl::template diag_all<set>(
                diag, static_cast<offset_t>(1), loc, size + nbatch, kernel);

            // sol += (hes + diag) \ (grad - conv(sol))
            PosDef::relax_(
                Strided(sol_ptr, osc),
                StridedConst(hes_ptr, hsc),
                val, diag, buf, static_cast<reduce_t>(0)
            );
        }
    });
    }
}

//======================================================================
//                           MEMBRANE JRLS
//======================================================================

// --- MEMBRANE+JRLS: matvec ------------------------------------------

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
          scalar_t * out,           // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_out,    // [*batch, *spatial, C] vector
    const offset_t * stride_inp,    // [*batch, *spatial, C] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t isc    = stride_inp[nall];
    reduce_t kernel[Impl::kernelsize_membrane_jrls];
    Impl::make_kernel_membrane_jrls(kernel, absolute, membrane, voxel_size);

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
            auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
            auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
            auto ci = tny::peel_front<-1>(vi).subrange(lo, hi).begin();
            auto cw = tny::peel_front<-1>(vw).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
            {
                Impl::template matvec_membrane_jrls<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), (*ci).data(), (*cw).data(),
                    v.index.data(), size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
                    osc, isc, kernel);
                ++ci;
                ++cw;
            }
        }
    });
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
          reduce_t   absolute,
          reduce_t   membrane
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    reduce_t kernel[Impl::kernelsize_membrane_jrls];
    Impl::make_kernel_membrane_jrls(kernel, absolute, membrane, voxel_size);

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
            auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
            auto cw = tny::peel_front<-1>(vw).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
            {
                Impl::template diag_membrane_jrls<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), (*cw).data(),
                    v.index.data(), size + nbatch, stride_wgt + nbatch, osc, kernel);
                ++cw;
            }
        }
    });
}

// --- MEMBRANE+JRLS: relax --------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_membrane_jrls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_sol,    // [*batch, *spatial, C] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, C] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          int        niter=1
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];
    offset_t numel  = prod(size, nall);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_membrane_jrls];
    Impl::make_kernel_membrane_jrls(kernel, absolute, membrane, voxel_size);
    constexpr int DD = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    for (offset_t n=0; n<2*niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t val[ndim], diag[ndim];
        reduce_t buf[DD ? DD : 1];
        for (offset_t i=start; i < end; ++i)
        {
            offset_t sol_offset = index2offset_v2<ndim>(i, nall, size, stride_sol, loc);
            if (!patch1<ndim>(loc, n))
                continue;
            offset_t grd_offset = index2offset(i, nall, size, stride_grd);
            offset_t hes_offset = index2offset(i, nall, size, stride_hes);
            offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

            // gradient
#           pragma unroll
            for (int d=0; d<ndim; ++d)
                val[d] = grd[grd_offset + gsc*d];

            // minus convolution
            Impl::template matvec_membrane_jrls<isub>(
                val, sol + sol_offset, wgt + wgt_offset,
                loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
                static_cast<offset_t>(1), osc, kernel);

            // diagonal
            Impl::template diag_membrane_jrls<set>(
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
    });
    }
}

//======================================================================
//                           LAME JRLS
//======================================================================

// --- LAME+JRLS: matvec ----------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_lame_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_out,    // [*batch, *spatial, C] vector
    const offset_t * stride_inp,    // [*batch, *spatial, C] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t absolute,
          reduce_t membrane,
          reduce_t shears,
          reduce_t div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall = nbatch + ndim;
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    reduce_t kernel[Impl::kernelsize_lame_jrls];
    Impl::make_kernel_lame_jrls(kernel, absolute, membrane, shears, div, voxel_size);

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
            auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
            auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
            auto ci = tny::peel_front<-1>(vi).subrange(lo, hi).begin();
            auto cw = tny::peel_front<-1>(vw).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
            {
                Impl::template matvec_lame_jrls<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), (*ci).data(), (*cw).data(),
                    v.index.data(), size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
                    osc, isc, kernel);
                ++ci;
                ++cw;
            }
        }
    });
}

// --- LAME+JRLS: diagonal ---------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_lame_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   shears,
          reduce_t   div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    reduce_t kernel[Impl::kernelsize_lame_jrls];
    Impl::make_kernel_lame_jrls(kernel, absolute, membrane, shears, div, voxel_size);

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
            auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
            auto cw = tny::peel_front<-1>(vw).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
            {
                Impl::template diag_lame_jrls<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), (*cw).data(),
                    v.index.data(), size + nbatch, stride_wgt + nbatch, osc, kernel);
                ++cw;
            }
        }
    });
}

// --- LAME+JRLS: relax ------------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_lame_jrls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,         // [*batch, *spatial, C] vector
    const offset_t * stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * stride_grd,   // [*batch, *spatial, C] vector
    const offset_t * stride_wgt,   // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   shears,
          reduce_t   div,
          int        niter=1
)
{
    using Impl          = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];
    offset_t numel  = prod(size, nall);    // no outer loop across channels

    reduce_t kernel[Impl::kernelsize_lame_jrls];
    Impl::make_kernel_lame_jrls(kernel, absolute, membrane, shears, div, voxel_size);
    constexpr int DD = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    for (offset_t n = 0; n < pow<ndim>(2)*niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t val[ndim], diag[ndim];
        reduce_t buf[DD ? DD : 1];
        for (offset_t i=start; i < end; ++i)
        {
            offset_t sol_offset = index2offset_v2<ndim>(i, nall, size, stride_sol, loc);
            if (!patch2<ndim>(loc, n))
                continue;
            offset_t grd_offset = index2offset(i, nall, size, stride_grd);
            offset_t hes_offset = index2offset(i, nall, size, stride_hes);
            offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

            // gradient
#           pragma unroll
            for (int d=0; d<ndim; ++d)
                val[d] = grd[grd_offset + gsc*d];

            // minus convolution
            Impl::template matvec_lame_jrls<isub>(
                val, sol + sol_offset, wgt + wgt_offset,
                loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
                static_cast<offset_t>(1), osc, kernel);

            // diagonal
            Impl::template diag_lame_jrls<set>(
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
    });
    }
}

FF_NAMESPACE_END(reg_flow)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)


#endif // FF_REGULARISERS_FLOW_CPU
