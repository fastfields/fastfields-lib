#ifndef FF_POSDEF_CPU
#define FF_POSDEF_CPU
#include "kernels/cuda_switch.h"
#include "kernels/posdef.h"
#include "kernels/batch.h"
#include "kernels/parallel.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(posdef)

template <int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_matvec(
          offset_t   nbatch,
          offset_t   nchannel,
          scalar_t * out,
    const scalar_t * hes,
    const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_hes,
    const offset_t * stride_inp
)
{
    offset_t numel = prod(size, nbatch);

    offset_t isc = stride_inp[nbatch];
    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset(i, nbatch, size, stride_out);
        offset_t hes_offset = index2offset(i, nbatch, size, stride_hes);
        offset_t inp_offset = index2offset(i, nbatch, size, stride_inp);

        if (C == -1)
            utils<type::Sym, offset_t>::matvec(
                nchannel,
                internal::pointer(out + out_offset, osc),
                internal::pointer(hes + hes_offset, hsc),
                internal::pointer(inp + inp_offset, isc),
                static_cast<reduce_t>(0));
        else
            utils<type::Sym, offset_t, (C < 0 ? 1 : C)>::matvec(
                internal::pointer(out + out_offset, osc),
                internal::pointer(hes + hes_offset, hsc),
                internal::pointer(inp + inp_offset, isc),
                static_cast<reduce_t>(0));
    }});
}

template <int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_matvec_backward(
          offset_t   nbatch,
          offset_t   nchannel,
          scalar_t * out,
    const scalar_t * grd,
    const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_grd,
    const offset_t * stride_inp)
{
    offset_t numel = prod(size, nbatch);

    // offset_t nc  = size[nbatch];
    offset_t isc = stride_inp[nbatch];
    offset_t gsc = stride_grd[nbatch];
    offset_t osc = stride_out[nbatch];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset(i, nbatch, size, stride_out);
        offset_t grd_offset = index2offset(i, nbatch, size, stride_grd);
        offset_t inp_offset = index2offset(i, nbatch, size, stride_inp);

        if (C == -1)
            utils<type::Sym, offset_t>::matvec_backward(
                nchannel,
                internal::pointer(out + out_offset, osc),
                internal::pointer(grd + grd_offset, gsc),
                internal::pointer(inp + inp_offset, isc),
                static_cast<reduce_t>(0));
        else
            utils<type::Sym, offset_t, (C < 0 ? 1 : C)>::matvec_backward(
                internal::pointer(out + out_offset, osc),
                internal::pointer(grd + grd_offset, gsc),
                internal::pointer(inp + inp_offset, isc),
                static_cast<reduce_t>(0));
    }});
}

template <int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_addmatvec_(
          offset_t   nbatch,
          offset_t   nchannel,
          scalar_t * out,
    const scalar_t * hes,
    const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_hes,
    const offset_t * stride_inp
)
{
    offset_t numel = prod(size, nbatch);

    offset_t nc  = size[nbatch];
    offset_t isc = stride_inp[nbatch];
    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset(i, nbatch, size, stride_out);
        offset_t hes_offset = index2offset(i, nbatch, size, stride_hes);
        offset_t inp_offset = index2offset(i, nbatch, size, stride_inp);

        if (C == -1)
            utils<type::Sym, offset_t>::addmatvec_(
                nchannel,
                internal::pointer(out + out_offset, osc),
                internal::pointer(hes + hes_offset, hsc),
                internal::pointer(inp + inp_offset, isc),
                static_cast<reduce_t>(0));
        else
            utils<type::Sym, offset_t, (C < 0 ? 1 : C)>::addmatvec_(
                internal::pointer(out + out_offset, osc),
                internal::pointer(hes + hes_offset, hsc),
                internal::pointer(inp + inp_offset, isc),
                static_cast<reduce_t>(0));
    }});
}

template <int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_submatvec_(
          offset_t   nbatch,
          offset_t   nchannel,
          scalar_t * out,
    const scalar_t * hes,
    const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_hes,
    const offset_t * stride_inp
)
{
    offset_t numel = prod(size, nbatch);

    offset_t isc = stride_inp[nbatch];
    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset(i, nbatch, size, stride_out);
        offset_t hes_offset = index2offset(i, nbatch, size, stride_hes);
        offset_t inp_offset = index2offset(i, nbatch, size, stride_inp);

        if (C == -1)
            utils<type::Sym, offset_t>::submatvec_(
                nchannel,
                internal::pointer(out + out_offset, osc),
                internal::pointer(hes + hes_offset, hsc),
                internal::pointer(inp + inp_offset, isc),
                static_cast<reduce_t>(0));
        else
            utils<type::Sym, offset_t, (C < 0 ? 1 : C)>::submatvec_(
                internal::pointer(out + out_offset, osc),
                internal::pointer(hes + hes_offset, hsc),
                internal::pointer(inp + inp_offset, isc),
                static_cast<reduce_t>(0));
    }});
}


template <int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_solve_tpl(
          offset_t   nbatch,
          scalar_t * out,
    const scalar_t * inp,
    const scalar_t * hes,
    const scalar_t * wgt,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_inp,
    const offset_t * stride_hes,
    const offset_t * stride_wgt
)
{
    offset_t numel = prod(size, nbatch);

    offset_t isc = stride_inp[nbatch];
    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];
    offset_t wsc = stride_wgt ? stride_wgt[nbatch] : 0;
    constexpr int CC = utils<type::Sym, offset_t, C>::work_size;

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    reduce_t buffer[CC];
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset(i, nbatch, size, stride_out);
        offset_t inp_offset = index2offset(i, nbatch, size, stride_inp);
        offset_t hes_offset = index2offset(i, nbatch, size, stride_hes);
        offset_t wgt_offset = stride_wgt ? index2offset(i, nbatch, size, stride_wgt) : 0;

        utils<type::Sym, offset_t, C>::solve(
            internal::pointer(out + out_offset, osc),
            internal::pointer(inp + inp_offset, isc),
            internal::pointer(hes + hes_offset, hsc),
            wgt ? internal::pointer(wgt + wgt_offset, wsc) : nullptr,
            buffer, static_cast<reduce_t>(0));
    }});
}


template <typename reduce_t, typename scalar_t, typename offset_t>
void sym_solve(
          offset_t   nbatch,
          offset_t   nchannel,
          scalar_t * out,
    const scalar_t * inp,
    const scalar_t * hes,
    const scalar_t * wgt,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_inp,
    const offset_t * stride_hes,
    const offset_t * stride_wgt
)
{
    offset_t numel = prod(size, nbatch);

    offset_t isc = stride_inp[nbatch];
    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];
    offset_t wsc = stride_wgt ? stride_wgt[nbatch] : 0;
    offset_t CC = utils<type::Sym, offset_t>::work_size(nchannel);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    reduce_t * buffer = new reduce_t[CC];
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset(i, nbatch, size, stride_out);
        offset_t inp_offset = index2offset(i, nbatch, size, stride_inp);
        offset_t hes_offset = index2offset(i, nbatch, size, stride_hes);
        offset_t wgt_offset = stride_wgt ? index2offset(i, nbatch, size, stride_wgt) : 0;

        utils<type::Sym, offset_t>::solve(
            nchannel,
            internal::pointer(out + out_offset, osc),
            internal::pointer(inp + inp_offset, isc),
            internal::pointer(hes + hes_offset, hsc),
            wgt ? internal::pointer(wgt + wgt_offset, wsc) : nullptr,
            buffer, static_cast<reduce_t>(0));
    }
    delete[] buffer;
    });
}


template <int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_solve_tpl_(
          offset_t   nbatch,
          scalar_t * out,
    const scalar_t * hes,
    const scalar_t * wgt,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_hes,
    const offset_t * stride_wgt
)
{
    offset_t numel = prod(size, nbatch);

    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];
    offset_t wsc = stride_wgt ? stride_wgt[nbatch] : 0;
    constexpr int CC = utils<type::Sym, offset_t, C>::work_size;

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    reduce_t buffer[CC];
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset(i, nbatch, size, stride_out);
        offset_t hes_offset = index2offset(i, nbatch, size, stride_hes);
        offset_t wgt_offset = stride_wgt ? index2offset(i, nbatch, size, stride_wgt) : 0;

        utils<type::Sym, offset_t, C>::solve_(
            internal::pointer(out + out_offset, osc),
            internal::pointer(hes + hes_offset, hsc),
            wgt ? internal::pointer(wgt + wgt_offset, wsc) : nullptr,
            buffer, static_cast<reduce_t>(0));
    }});
}


template <typename reduce_t, typename scalar_t, typename offset_t>
void sym_solve_(
          offset_t   nbatch,
          offset_t   nchannel,
          scalar_t * out,
    const scalar_t * hes,
    const scalar_t * wgt,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_hes,
    const offset_t * stride_wgt
)
{
    offset_t numel = prod(size, nbatch);

    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];
    offset_t wsc = stride_wgt ? stride_wgt[nbatch] : 0;
    offset_t CC = utils<type::Sym, offset_t>::work_size(nchannel);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    reduce_t * buffer = new reduce_t[CC];
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset(i, nbatch, size, stride_out);
        offset_t hes_offset = index2offset(i, nbatch, size, stride_hes);
        offset_t wgt_offset = stride_wgt ? index2offset(i, nbatch, size, stride_wgt) : 0;

        utils<type::Sym, offset_t>::solve_(
            nchannel,
            internal::pointer(out + out_offset, osc),
            internal::pointer(hes + hes_offset, hsc),
            wgt ? internal::pointer(wgt + wgt_offset, wsc) : nullptr,
            buffer, static_cast<reduce_t>(0));
    }
    delete[] buffer;
    });
}


template <int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_invert_tpl(
          offset_t   nbatch,
          scalar_t * out,
    const scalar_t * hes,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_hes
)
{
    offset_t numel = prod(size, nbatch);

    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];
    constexpr int CC = utils<type::Sym, offset_t, C>::work_size;

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    reduce_t buffer[CC];
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset(i, nbatch, size, stride_out);
        offset_t hes_offset = index2offset(i, nbatch, size, stride_hes);

        utils<type::Sym, offset_t, C>::invert(
            internal::pointer(out + out_offset, osc),
            internal::pointer(hes + hes_offset, hsc),
            buffer, static_cast<reduce_t>(0));
    }});
}


template <typename reduce_t, typename scalar_t, typename offset_t>
void sym_invert(
          offset_t   nbatch,
          offset_t   nchannel,
          scalar_t * out,
    const scalar_t * hes,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_hes
)
{
    offset_t numel = prod(size, nbatch);

    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];
    offset_t CC = utils<type::Sym, offset_t>::work_size(nchannel);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    reduce_t * buffer = new reduce_t[CC];
    for (offset_t i=start; i < end; ++i)
    {
        offset_t out_offset = index2offset(i, nbatch, size, stride_out);
        offset_t hes_offset = index2offset(i, nbatch, size, stride_hes);

        utils<type::Sym, offset_t>::invert(
            nchannel,
            internal::pointer(out + out_offset, osc),
            internal::pointer(hes + hes_offset, hsc),
            buffer, static_cast<reduce_t>(0));
    }
    delete[] buffer;
    });
}


template <int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_invert_tpl_(
          offset_t   nbatch,
          scalar_t * hes,
    const offset_t * size,
    const offset_t * stride
)
{
    offset_t numel = prod(size, nbatch);

    offset_t sc = stride[nbatch];
    constexpr int CC = utils<type::Sym, offset_t, C>::work_size;

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    reduce_t buffer[CC];
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset = index2offset(i, nbatch, size, stride);

        utils<type::Sym, offset_t, C>::invert_(
            internal::pointer(hes + offset, sc),
            buffer, static_cast<reduce_t>(0));
    }});
}


template <typename reduce_t, typename scalar_t, typename offset_t>
void sym_invert_(
          offset_t   nbatch,
          offset_t   nchannel,
          scalar_t * hes,
    const offset_t * size,
    const offset_t * stride
)
{
    offset_t numel = prod(size, nbatch);

    offset_t sc = stride[nbatch];
    offset_t CC = utils<type::Sym, offset_t>::work_size(nchannel);

    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    reduce_t * buffer = new reduce_t[CC];
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset = index2offset(i, nbatch, size, stride);

        utils<type::Sym, offset_t>::invert_(
            nchannel,
            internal::pointer(hes + offset, sc),
            buffer, static_cast<reduce_t>(0));
    }
    delete[] buffer;
    });
}


FF_NAMESPACE_END(posdef)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_POSDEF_CPU
