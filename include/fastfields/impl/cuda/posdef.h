#pragma once
#include "fastfields/core/cuda_switch.h"
#include "fastfields/impl/kernels/posdef.h"
#include "fastfields/impl/kernels/batch.h"
#include "utils.h"
#include <cstdint>
#include <stdexcept>

using namespace std;
FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(posdef)

// Largest number of batch dimensions the CUDA launchers instantiate. The
// device kernels are templated on a compile-time `nbatch`, so the host
// launcher dispatches the runtime value to a static instantiation.
#ifndef FF_POSDEF_MAX_NBATCH
#define FF_POSDEF_MAX_NBATCH 7
#endif

template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUGLOB
void sym_matvec_k(scalar_t * out, const scalar_t * hes, const scalar_t * inp,
                const offset_t * size,
                const offset_t * stride_out,
                const offset_t * stride_hes,
                const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t numel = prod<nbatch>(size);

    offset_t isc = stride_inp[nbatch];
    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];

    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride_out);
        offset_t hes_offset = index2offset<nbatch>(i, size, stride_hes);
        offset_t inp_offset = index2offset<nbatch>(i, size, stride_inp);

        utils<type::Sym, offset_t, C>::matvec(
            internal::pointer(out + out_offset, osc),
            internal::pointer(hes + hes_offset, hsc),
            internal::pointer(inp + inp_offset, isc),
            static_cast<reduce_t>(0));
    }
}

template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUGLOB
void sym_matvec_backward_k(
    scalar_t * out, const scalar_t * grd, const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_grd,
    const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t numel = prod<nbatch>(size);

    offset_t isc = stride_inp[nbatch];
    offset_t gsc = stride_grd[nbatch];
    offset_t osc = stride_out[nbatch];

    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride_out);
        offset_t grd_offset = index2offset<nbatch>(i, size, stride_grd);
        offset_t inp_offset = index2offset<nbatch>(i, size, stride_inp);

        utils<type::Sym, offset_t, C>::matvec_backward(
            internal::pointer(out + out_offset, osc),
            internal::pointer(grd + grd_offset, gsc),
            internal::pointer(inp + inp_offset, isc),
            static_cast<reduce_t>(0));
    }
}

template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUGLOB
void sym_addmatvec__k(scalar_t * out, const scalar_t * hes, const scalar_t * inp,
                    const offset_t * size,
                    const offset_t * stride_out,
                    const offset_t * stride_hes,
                    const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t numel = prod<nbatch>(size);

    offset_t isc = stride_inp[nbatch];
    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];

    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride_out);
        offset_t hes_offset = index2offset<nbatch>(i, size, stride_hes);
        offset_t inp_offset = index2offset<nbatch>(i, size, stride_inp);

        utils<type::Sym, offset_t, C>::addmatvec_(
            internal::pointer(out + out_offset, osc),
            internal::pointer(hes + hes_offset, hsc),
            internal::pointer(inp + inp_offset, isc),
            static_cast<reduce_t>(0));
    }
}

template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUGLOB
void sym_submatvec__k(scalar_t * out, const scalar_t * hes, const scalar_t * inp,
                    const offset_t * size,
                    const offset_t * stride_out,
                    const offset_t * stride_hes,
                    const offset_t * stride_inp)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t numel = prod<nbatch>(size);

    offset_t isc = stride_inp[nbatch];
    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];

    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride_out);
        offset_t hes_offset = index2offset<nbatch>(i, size, stride_hes);
        offset_t inp_offset = index2offset<nbatch>(i, size, stride_inp);

        utils<type::Sym, offset_t, C>::submatvec_(
            internal::pointer(out + out_offset, osc),
            internal::pointer(hes + hes_offset, hsc),
            internal::pointer(inp + inp_offset, isc),
            static_cast<reduce_t>(0));
    }
}


template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUGLOB
void sym_solve_k(scalar_t * out, const scalar_t * inp,
               const scalar_t * hes, const scalar_t * wgt,
               const offset_t * size,
               const offset_t * stride_out,
               const offset_t * stride_inp,
               const offset_t * stride_hes,
               const offset_t * stride_wgt)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t numel = prod<nbatch>(size);

    offset_t isc = stride_inp[nbatch];
    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];
    offset_t wsc = stride_wgt ? stride_wgt[nbatch] : 0;
    constexpr int CC = utils<type::Sym, offset_t, C>::work_size;
    reduce_t buffer[CC > 0 ? CC : 1];  // avoid zero-sized array (work_size==0 for small static C)

    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride_out);
        offset_t inp_offset = index2offset<nbatch>(i, size, stride_inp);
        offset_t hes_offset = index2offset<nbatch>(i, size, stride_hes);
        offset_t wgt_offset = stride_wgt ? index2offset<nbatch>(i, size, stride_wgt) : 0;

        utils<type::Sym, offset_t, C>::solve(
            internal::pointer(out + out_offset, osc),
            internal::pointer(inp + inp_offset, isc),
            internal::pointer(hes + hes_offset, hsc),
            internal::pointer(wgt + wgt_offset, wsc),  // null-data Pointer when wgt==nullptr (device-safe; solve checks if(w))
            buffer, static_cast<reduce_t>(0));
    }
}


template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUGLOB
void sym_solve__k(scalar_t * out,
                const scalar_t * hes, const scalar_t * wgt,
                const offset_t * size,
                const offset_t * stride_out,
                const offset_t * stride_hes,
                const offset_t * stride_wgt)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t numel = prod<nbatch>(size);

    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];
    offset_t wsc = stride_wgt ? stride_wgt[nbatch] : 0;
    constexpr int CC = utils<type::Sym, offset_t, C>::work_size;

    reduce_t buffer[CC > 0 ? CC : 1];  // avoid zero-sized array (work_size==0 for small static C)
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride_out);
        offset_t hes_offset = index2offset<nbatch>(i, size, stride_hes);
        offset_t wgt_offset = stride_wgt ? index2offset<nbatch>(i, size, stride_wgt) : 0;

        utils<type::Sym, offset_t, C>::solve_(
            internal::pointer(out + out_offset, osc),
            internal::pointer(hes + hes_offset, hsc),
            internal::pointer(wgt + wgt_offset, wsc),  // null-data Pointer when wgt==nullptr (device-safe; solve checks if(w))
            buffer, static_cast<reduce_t>(0));
    };
}

template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUGLOB
void sym_invert_k(scalar_t * out, const scalar_t * hes,
                const offset_t * size,
                const offset_t * stride_out,
                const offset_t * stride_hes)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t numel = prod<nbatch>(size);

    offset_t hsc = stride_hes[nbatch];
    offset_t osc = stride_out[nbatch];
    constexpr int CC = utils<type::Sym, offset_t, C>::work_size;

    reduce_t buffer[CC > 0 ? CC : 1];  // avoid zero-sized array (work_size==0 for small static C)
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t out_offset = index2offset<nbatch>(i, size, stride_out);
        offset_t hes_offset = index2offset<nbatch>(i, size, stride_hes);

        utils<type::Sym, offset_t, C>::invert(
            internal::pointer(out + out_offset, osc),
            internal::pointer(hes + hes_offset, hsc),
            buffer, static_cast<reduce_t>(0));
    }
}

template <int nbatch, int C, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUGLOB
void sym_invert__k(scalar_t * hes,
                 const offset_t * size,
                 const offset_t * stride)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t numel = prod<nbatch>(size);

    offset_t sc = stride[nbatch];
    constexpr int CC = utils<type::Sym, offset_t, C>::work_size;

    reduce_t buffer[CC > 0 ? CC : 1];  // avoid zero-sized array (work_size==0 for small static C)
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t offset = index2offset<nbatch>(i, size, stride);

        utils<type::Sym, offset_t, C>::invert_(
            internal::pointer(hes + offset, sc),
            buffer, static_cast<reduce_t>(0));
    }
}

/***********************************************************************
 *                          HOST LAUNCHERS                             *
 *                                                                     *
 * The device kernels above are templated on a compile-time `nbatch`   *
 * and a compile-time channel count `C`. The host launchers mirror the *
 * CPU-impl signatures (runtime `nbatch`/`nchannel`) and dispatch those *
 * to a static instantiation before launching the kernel on `stream`.  *
 * The CUDA compact-symmetric kernels only implement the static-C path, *
 * so channel counts are supported for C in {1,2,3}; larger counts      *
 * (and the dynamic C=-1 path) throw std::logic_error.                  *
 ***********************************************************************/

// Dispatch the runtime `nbatch` to a compile-time NB and launch (per op,
// FF_LAUNCH(NB, CC) is #defined to the matching kernel launch).
#define FF_POSDEF_NB_SWITCH(CC)                                             \
    switch (nbatch) {                                                       \
        case 0: FF_LAUNCH(0, CC); break;                                    \
        case 1: FF_LAUNCH(1, CC); break;                                    \
        case 2: FF_LAUNCH(2, CC); break;                                    \
        case 3: FF_LAUNCH(3, CC); break;                                    \
        case 4: FF_LAUNCH(4, CC); break;                                    \
        case 5: FF_LAUNCH(5, CC); break;                                    \
        case 6: FF_LAUNCH(6, CC); break;                                    \
        case 7: FF_LAUNCH(7, CC); break;                                    \
        default: throw std::logic_error(                                   \
            "posdef: nbatch too large for CUDA launcher");                 \
    }

// Dispatch the runtime `nchannel` to a compile-time CC in {1,2,3}.
#define FF_POSDEF_NC_SWITCH                                                 \
    switch (nchannel) {                                                     \
        case 1: FF_POSDEF_NB_SWITCH(1); break;                             \
        case 2: FF_POSDEF_NB_SWITCH(2); break;                             \
        case 3: FF_POSDEF_NB_SWITCH(3); break;                             \
        default: throw std::logic_error(                                   \
            "posdef: nchannel > 3 not supported by CUDA launcher");        \
    }

#define FF_POSDEF_PROLOGUE                                                  \
    offset_t numel = 1;                                                     \
    for (offset_t _d = 0; _d < nbatch; ++_d) numel *= size[_d];            \
    cudaStream_t s   = (cudaStream_t)(std::intptr_t)stream;                 \
    const int blocks  = GET_BLOCKS(numel);                                 \
    const int threads = CUDA_NUM_THREADS

template <int C, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUHOST
void sym_matvec(
          offset_t   nbatch,
          offset_t   nchannel,
          scalar_t * out,
    const scalar_t * hes,
    const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_hes,
    const offset_t * stride_inp,
          intptr_t   stream = 0)
{
    const offset_t ndim = nbatch + 1;
    offset_t * d_size = nullptr, * d_so = nullptr, * d_sh = nullptr, * d_si = nullptr;
    try {
        d_size = copyToDevice(size,       ndim);
        d_so   = copyToDevice(stride_out, ndim);
        d_sh   = copyToDevice(stride_hes, ndim);
        d_si   = copyToDevice(stride_inp, ndim);
        FF_POSDEF_PROLOGUE;
#       define FF_LAUNCH(NB, CC)                                            \
            sym_matvec_k<NB, CC, reduce_t, scalar_t, offset_t>            \
                <<<blocks, threads, 0, s>>>(                              \
                    out, hes, inp, d_size, d_so, d_sh, d_si)
        FF_POSDEF_NC_SWITCH;
#       undef FF_LAUNCH
    } catch (...) { freeDevice(d_size, d_so, d_sh, d_si); throw; }
    freeDevice(d_size, d_so, d_sh, d_si);
}

template <int C, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUHOST
void sym_matvec_backward(
          offset_t   nbatch,
          offset_t   nchannel,
          scalar_t * out,
    const scalar_t * grd,
    const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_grd,
    const offset_t * stride_inp,
          intptr_t   stream = 0)
{
    const offset_t ndim = nbatch + 1;
    offset_t * d_size = nullptr, * d_so = nullptr, * d_sg = nullptr, * d_si = nullptr;
    try {
        d_size = copyToDevice(size,       ndim);
        d_so   = copyToDevice(stride_out, ndim);
        d_sg   = copyToDevice(stride_grd, ndim);
        d_si   = copyToDevice(stride_inp, ndim);
        FF_POSDEF_PROLOGUE;
#       define FF_LAUNCH(NB, CC)                                            \
            sym_matvec_backward_k<NB, CC, reduce_t, scalar_t, offset_t>   \
                <<<blocks, threads, 0, s>>>(                              \
                    out, grd, inp, d_size, d_so, d_sg, d_si)
        FF_POSDEF_NC_SWITCH;
#       undef FF_LAUNCH
    } catch (...) { freeDevice(d_size, d_so, d_sg, d_si); throw; }
    freeDevice(d_size, d_so, d_sg, d_si);
}

template <int C, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUHOST
void sym_addmatvec_(
          offset_t   nbatch,
          offset_t   nchannel,
          scalar_t * out,
    const scalar_t * hes,
    const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_hes,
    const offset_t * stride_inp,
          intptr_t   stream = 0)
{
    const offset_t ndim = nbatch + 1;
    offset_t * d_size = nullptr, * d_so = nullptr, * d_sh = nullptr, * d_si = nullptr;
    try {
        d_size = copyToDevice(size,       ndim);
        d_so   = copyToDevice(stride_out, ndim);
        d_sh   = copyToDevice(stride_hes, ndim);
        d_si   = copyToDevice(stride_inp, ndim);
        FF_POSDEF_PROLOGUE;
#       define FF_LAUNCH(NB, CC)                                            \
            sym_addmatvec__k<NB, CC, reduce_t, scalar_t, offset_t>        \
                <<<blocks, threads, 0, s>>>(                              \
                    out, hes, inp, d_size, d_so, d_sh, d_si)
        FF_POSDEF_NC_SWITCH;
#       undef FF_LAUNCH
    } catch (...) { freeDevice(d_size, d_so, d_sh, d_si); throw; }
    freeDevice(d_size, d_so, d_sh, d_si);
}

template <int C, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUHOST
void sym_submatvec_(
          offset_t   nbatch,
          offset_t   nchannel,
          scalar_t * out,
    const scalar_t * hes,
    const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_hes,
    const offset_t * stride_inp,
          intptr_t   stream = 0)
{
    const offset_t ndim = nbatch + 1;
    offset_t * d_size = nullptr, * d_so = nullptr, * d_sh = nullptr, * d_si = nullptr;
    try {
        d_size = copyToDevice(size,       ndim);
        d_so   = copyToDevice(stride_out, ndim);
        d_sh   = copyToDevice(stride_hes, ndim);
        d_si   = copyToDevice(stride_inp, ndim);
        FF_POSDEF_PROLOGUE;
#       define FF_LAUNCH(NB, CC)                                            \
            sym_submatvec__k<NB, CC, reduce_t, scalar_t, offset_t>        \
                <<<blocks, threads, 0, s>>>(                              \
                    out, hes, inp, d_size, d_so, d_sh, d_si)
        FF_POSDEF_NC_SWITCH;
#       undef FF_LAUNCH
    } catch (...) { freeDevice(d_size, d_so, d_sh, d_si); throw; }
    freeDevice(d_size, d_so, d_sh, d_si);
}

template <typename reduce_t, typename scalar_t, typename offset_t>
FF_CUHOST
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
    const offset_t * stride_wgt,
          intptr_t   stream = 0)
{
    const offset_t ndim = nbatch + 1;
    offset_t * d_size = nullptr, * d_so = nullptr, * d_si = nullptr,
             * d_sh = nullptr, * d_sw = nullptr;
    try {
        d_size = copyToDevice(size,       ndim);
        d_so   = copyToDevice(stride_out, ndim);
        d_si   = copyToDevice(stride_inp, ndim);
        d_sh   = copyToDevice(stride_hes, ndim);
        d_sw   = stride_wgt ? copyToDevice(stride_wgt, ndim) : nullptr;
        FF_POSDEF_PROLOGUE;
#       define FF_LAUNCH(NB, CC)                                            \
            sym_solve_k<NB, CC, reduce_t, scalar_t, offset_t>            \
                <<<blocks, threads, 0, s>>>(                              \
                    out, inp, hes, wgt, d_size, d_so, d_si, d_sh, d_sw)
        FF_POSDEF_NC_SWITCH;
#       undef FF_LAUNCH
    } catch (...) { freeDevice(d_size, d_so, d_si, d_sh, d_sw); throw; }
    freeDevice(d_size, d_so, d_si, d_sh, d_sw);
}

template <typename reduce_t, typename scalar_t, typename offset_t>
FF_CUHOST
void sym_solve_(
          offset_t   nbatch,
          offset_t   nchannel,
          scalar_t * out,
    const scalar_t * hes,
    const scalar_t * wgt,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_hes,
    const offset_t * stride_wgt,
          intptr_t   stream = 0)
{
    const offset_t ndim = nbatch + 1;
    offset_t * d_size = nullptr, * d_so = nullptr, * d_sh = nullptr, * d_sw = nullptr;
    try {
        d_size = copyToDevice(size,       ndim);
        d_so   = copyToDevice(stride_out, ndim);
        d_sh   = copyToDevice(stride_hes, ndim);
        d_sw   = stride_wgt ? copyToDevice(stride_wgt, ndim) : nullptr;
        FF_POSDEF_PROLOGUE;
#       define FF_LAUNCH(NB, CC)                                            \
            sym_solve__k<NB, CC, reduce_t, scalar_t, offset_t>           \
                <<<blocks, threads, 0, s>>>(                              \
                    out, hes, wgt, d_size, d_so, d_sh, d_sw)
        FF_POSDEF_NC_SWITCH;
#       undef FF_LAUNCH
    } catch (...) { freeDevice(d_size, d_so, d_sh, d_sw); throw; }
    freeDevice(d_size, d_so, d_sh, d_sw);
}

template <typename reduce_t, typename scalar_t, typename offset_t>
FF_CUHOST
void sym_invert(
          offset_t   nbatch,
          offset_t   nchannel,
          scalar_t * out,
    const scalar_t * hes,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_hes,
          intptr_t   stream = 0)
{
    const offset_t ndim = nbatch + 1;
    offset_t * d_size = nullptr, * d_so = nullptr, * d_sh = nullptr;
    try {
        d_size = copyToDevice(size,       ndim);
        d_so   = copyToDevice(stride_out, ndim);
        d_sh   = copyToDevice(stride_hes, ndim);
        FF_POSDEF_PROLOGUE;
#       define FF_LAUNCH(NB, CC)                                            \
            sym_invert_k<NB, CC, reduce_t, scalar_t, offset_t>           \
                <<<blocks, threads, 0, s>>>(                              \
                    out, hes, d_size, d_so, d_sh)
        FF_POSDEF_NC_SWITCH;
#       undef FF_LAUNCH
    } catch (...) { freeDevice(d_size, d_so, d_sh); throw; }
    freeDevice(d_size, d_so, d_sh);
}

template <typename reduce_t, typename scalar_t, typename offset_t>
FF_CUHOST
void sym_invert_(
          offset_t   nbatch,
          offset_t   nchannel,
          scalar_t * hes,
    const offset_t * size,
    const offset_t * stride,
          intptr_t   stream = 0)
{
    const offset_t ndim = nbatch + 1;
    offset_t * d_size = nullptr, * d_st = nullptr;
    try {
        d_size = copyToDevice(size,   ndim);
        d_st   = copyToDevice(stride, ndim);
        FF_POSDEF_PROLOGUE;
#       define FF_LAUNCH(NB, CC)                                            \
            sym_invert__k<NB, CC, reduce_t, scalar_t, offset_t>          \
                <<<blocks, threads, 0, s>>>(hes, d_size, d_st)
        FF_POSDEF_NC_SWITCH;
#       undef FF_LAUNCH
    } catch (...) { freeDevice(d_size, d_st); throw; }
    freeDevice(d_size, d_st);
}

#undef FF_POSDEF_PROLOGUE
#undef FF_POSDEF_NC_SWITCH
#undef FF_POSDEF_NB_SWITCH

FF_NAMESPACE_END(posdef)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
