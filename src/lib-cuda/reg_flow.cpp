/**
 * `reg_flow`'s exported entry points -- and nothing else.
 *
 * This translation unit instantiates no kernel template. It normalises
 * strides, validates arguments, decides the offset width, and hands the call
 * to one of the slice functions declared in `reg_flow_slice.h`; the slice TUs
 * (`reg_flow_<family>_<n>d.cpp`) hold the instantiations. See that header for
 * why the seam is a forwarding call rather than `extern template` or a
 * per-slice `-D` on one source.
 *
 * Every argument check, every error message and the order they fire in are
 * unchanged from the single-TU form. That matters: they are the observable
 * behaviour of these functions and the hub's tests pin them.
 */

#include <stdexcept>
#include <string>
#include <cstdint>
#include "fastfields/api/cuda/reg_flow.h"
#include "fastfields/api/cuda/posdef.h"
#include "fastfields/core/dispatch.h"
#include "fastfields/core/dlpack.h"
#include "fastfields/core/cuda_switch.h"
// `FF_CANUSE32BITS` expands to an unqualified `canUse32BitIndexMath`, which is
// declared in impl/kernels/utils.h -- NOT in core/autocast.h, whose `//
// canUse32BitIndexMath` include comment in core/dispatch.h suggests otherwise.
// Every other dispatch source pulls the kernels in wholesale and never noticed;
// this TU is the first that does not, so it has to name the real home.
#include "fastfields/impl/kernels/utils.h"  // canUse32BitIndexMath
#include "fastfields/impl/cuda/utils.h"     // allocDevice / freeDevice
#include "reg_flow_slice.h"

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)

/***********************************************************************
 *                            DISPATCH                                 *
 ***********************************************************************/

// What is left of the old `NDIM_SWITCH`: it still picks the spatial rank, but
// the arm is now a call into another translation unit instead of a template
// argument. The dtype x offset x boundary pyramid that used to sit under each
// arm moved with the instantiations, into the slice TUs.
#define FF_FLOW_ND_SWITCH(FN, ARGS)                                     \
    switch (ndim) {                                                     \
        case 1: return flow_slice::FN##_1d ARGS;                        \
        case 2: return flow_slice::FN##_2d ARGS;                        \
        case 3: return flow_slice::FN##_3d ARGS;                        \
        default: throw std::invalid_argument("Only 1D, 2D and 3D flow are supported"); \
    }

#define FF_FLOW_MV_CALL                                                 \
    (out, inp, voxel_size, absolute, membrane, bending, shears, div,    \
     bound, nbatch, use_32bits, stream)

#define FF_FLOW_DG_CALL                                                 \
    (out, voxel_size, absolute, membrane, bending, shears, div,         \
     bound, nbatch, use_32bits, stream)

#define FF_FLOW_RX_CALL                                                 \
    (sol, hes, grd, voxel_size, absolute, membrane, bending, shears,    \
     div, bound, nb_iter, nbatch, use_32bits, stream)

void flow_matvec(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _inp(inp_);
    DLTensor       & out = _out.t;
    const DLTensor & inp = _inp.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    FF_CHECK_NO_LANES  (out)
    FF_CHECK_SAME_DTYPE(out, inp)
    FF_CHECK_SAME      (out.ndim, inp.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    FF_CHECK_SAME      (out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    FF_CHECK_SAME_SHAPE_N(out, inp, out.ndim)

    const bool use_32bits = FF_CANUSE32BITS(out) && FF_CANUSE32BITS(inp);

    FF_FLOW_ND_SWITCH(matvec, FF_FLOW_MV_CALL)
}

/**
 * @brief `flow_matvec` variant that accumulates into `out`: `out += L(inp)`,
 *        instead of overwriting it. Same conventions otherwise.
 */
void flow_addmatvec_(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _inp(inp_);
    DLTensor       & out = _out.t;
    const DLTensor & inp = _inp.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    FF_CHECK_NO_LANES  (out)
    FF_CHECK_SAME_DTYPE(out, inp)
    FF_CHECK_SAME      (out.ndim, inp.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    FF_CHECK_SAME      (out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    FF_CHECK_SAME_SHAPE_N(out, inp, out.ndim)

    const bool use_32bits = FF_CANUSE32BITS(out) && FF_CANUSE32BITS(inp);

    FF_FLOW_ND_SWITCH(addmatvec, FF_FLOW_MV_CALL)
}

/**
 * @brief `flow_matvec` variant that subtracts from `out`: `out -= L(inp)`,
 *        instead of overwriting it. Same conventions otherwise.
 */
void flow_submatvec_(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _inp(inp_);
    DLTensor       & out = _out.t;
    const DLTensor & inp = _inp.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    FF_CHECK_NO_LANES  (out)
    FF_CHECK_SAME_DTYPE(out, inp)
    FF_CHECK_SAME      (out.ndim, inp.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    FF_CHECK_SAME      (out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    FF_CHECK_SAME_SHAPE_N(out, inp, out.ndim)

    const bool use_32bits = FF_CANUSE32BITS(out) && FF_CANUSE32BITS(inp);

    FF_FLOW_ND_SWITCH(submatvec, FF_FLOW_MV_CALL)
}

void flow_diag(
          DLTensor & out_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_);
    DLTensor & out = _out.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    FF_CHECK_NO_LANES(out)
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    FF_CHECK_SAME    (out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")

    const bool use_32bits = FF_CANUSE32BITS(out);

    FF_FLOW_ND_SWITCH(diag, FF_FLOW_DG_CALL)
}

/**
 * @brief `flow_diag` variant that accumulates: `out += diag(L)`. In-place only,
 *        matching the jitfields C-level `op='+'` entry point.
 */
void flow_adddiag_(
          DLTensor & out_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_);
    DLTensor & out = _out.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    FF_CHECK_NO_LANES(out)
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    FF_CHECK_SAME    (out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")

    const bool use_32bits = FF_CANUSE32BITS(out);

    FF_FLOW_ND_SWITCH(adddiag, FF_FLOW_DG_CALL)
}

/**
 * @brief `flow_diag` variant that accumulates: `out -= diag(L)`. In-place only,
 *        matching the jitfields C-level `op='-'` entry point.
 */
void flow_subdiag_(
          DLTensor & out_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_);
    DLTensor & out = _out.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    FF_CHECK_NO_LANES(out)
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    FF_CHECK_SAME    (out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")

    const bool use_32bits = FF_CANUSE32BITS(out);

    FF_FLOW_ND_SWITCH(subdiag, FF_FLOW_DG_CALL)
}

void flow_kernel(
          DLTensor & out_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_);
    DLTensor & out = _out.t;

    // The Lamé (shears/div) stencil is a C x C matrix of kernels (one extra
    // trailing axis); every other penalty gives a per-channel vector of
    // kernels. The output rank tells us which, and fixes nbatch.
    const bool is_matrix = (shears != 0.0 || div != 0.0);
    const int  ntrail    = is_matrix ? 2 : 1;
    const int32_t nbatch = out.ndim - ndim - ntrail;

    FF_CHECK_NO_LANES(out)
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    FF_CHECK_SAME(out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    if (is_matrix)
        FF_CHECK_SAME(out.shape[out.ndim-2], (int64_t)ndim,
                   "Lamé kernel needs a trailing (ndim, ndim) matrix axis")

    const bool use_32bits = FF_CANUSE32BITS(out);

    FF_FLOW_ND_SWITCH(kernel, FF_FLOW_DG_CALL)
}

/**
 * @brief `flow_kernel` variant that accumulates: `out += K (the stencil)`. In-place only,
 *        matching the jitfields C-level `op='+'` entry point.
 */
void flow_addkernel_(
          DLTensor & out_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_);
    DLTensor & out = _out.t;

    // The Lamé (shears/div) stencil is a C x C matrix of kernels (one extra
    // trailing axis); every other penalty gives a per-channel vector of
    // kernels. The output rank tells us which, and fixes nbatch.
    const bool is_matrix = (shears != 0.0 || div != 0.0);
    const int  ntrail    = is_matrix ? 2 : 1;
    const int32_t nbatch = out.ndim - ndim - ntrail;

    FF_CHECK_NO_LANES(out)
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    FF_CHECK_SAME(out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    if (is_matrix)
        FF_CHECK_SAME(out.shape[out.ndim-2], (int64_t)ndim,
                   "Lamé kernel needs a trailing (ndim, ndim) matrix axis")

    const bool use_32bits = FF_CANUSE32BITS(out);

    FF_FLOW_ND_SWITCH(addkernel, FF_FLOW_DG_CALL)
}

/**
 * @brief `flow_kernel` variant that accumulates: `out -= K (the stencil)`. In-place only,
 *        matching the jitfields C-level `op='-'` entry point.
 */
void flow_subkernel_(
          DLTensor & out_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_);
    DLTensor & out = _out.t;

    // The Lamé (shears/div) stencil is a C x C matrix of kernels (one extra
    // trailing axis); every other penalty gives a per-channel vector of
    // kernels. The output rank tells us which, and fixes nbatch.
    const bool is_matrix = (shears != 0.0 || div != 0.0);
    const int  ntrail    = is_matrix ? 2 : 1;
    const int32_t nbatch = out.ndim - ndim - ntrail;

    FF_CHECK_NO_LANES(out)
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    FF_CHECK_SAME(out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    if (is_matrix)
        FF_CHECK_SAME(out.shape[out.ndim-2], (int64_t)ndim,
                   "Lamé kernel needs a trailing (ndim, ndim) matrix axis")

    const bool use_32bits = FF_CANUSE32BITS(out);

    FF_FLOW_ND_SWITCH(subkernel, FF_FLOW_DG_CALL)
}

void flow_relax(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        nb_iter   ,
          intptr_t   stream
)
{
    const int32_t nbatch = sol.ndim - ndim - 1;
    FF_CHECK_NO_LANES  (sol)
    FF_CHECK_SAME_DTYPE(sol, hes)
    FF_CHECK_SAME_DTYPE(sol, grd)
    FF_CHECK_SAME      (sol.ndim, grd.ndim, "Tensors do not have the same number of dimensions")
    FF_CHECK_SAME      (sol.ndim, hes.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    FF_CHECK_SAME      (sol.shape[sol.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    FF_CHECK_SAME      (grd.shape[grd.ndim-1], (int64_t)ndim, "Gradient channel dimension must equal ndim")
    FF_CHECK_SAME_SHAPE_N(sol, grd, sol.ndim)

    const bool use_32bits = FF_CANUSE32BITS(sol) && FF_CANUSE32BITS(hes) &&
                            FF_CANUSE32BITS(grd);

    FF_FLOW_ND_SWITCH(relax, FF_FLOW_RX_CALL)
}

void flow_forward(
          DLTensor & out       ,
    const DLTensor & hes       ,
    const DLTensor & inp       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream    )
{
    sym_matvec(out, hes, inp, stream);
    flow_addmatvec_(out, inp, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
}

// `flow_diag`'s regulariser diagonal doesn't depend on the operand being
// solved for, so `flow_precond[_]` materialise it into a fresh contiguous
// device scratch buffer shaped like `grd`/`sol` and hand it to
// posdef::sym_solve[_] as the per-channel weight map. Caller owns the
// returned device pointer and must freeDevice() it.
static inline uint8_t * flow_precond_diag(
    const DLTensor & like      ,
    const double    * voxel_size,
          double      absolute  ,
          double      membrane  ,
          double      bending   ,
          double      shears    ,
          double      div       ,
          int8_t      bound     ,
          int         ndim      ,
          intptr_t    stream    ,
          DLTensor  & diag_t    )
{
    size_t numel = 1;
    for (int32_t d = 0; d < like.ndim; ++d)
        numel *= static_cast<size_t>(like.shape[d]);
    uint8_t * diag_buf = allocDevice<uint8_t>(numel * static_cast<size_t>(like.dtype.bits) / 8);

    diag_t.data        = diag_buf;
    diag_t.device       = like.device;
    diag_t.ndim         = like.ndim;
    diag_t.dtype        = like.dtype;
    diag_t.shape        = like.shape;
    diag_t.strides      = nullptr;
    diag_t.byte_offset  = 0;

    flow_diag(diag_t, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
    return diag_buf;
}

void flow_precond(
          DLTensor & out       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream    )
{
    FF_CHECK_NO_LANES(grd)
    if (grd.ndim - ndim - 1 < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");

    DLTensor diag_t;
    uint8_t * diag_buf = flow_precond_diag(
        grd, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream, diag_t);
    try {
        sym_solve(out, hes, grd, diag_t, stream);
    } catch (...) {
        freeDevice(diag_buf);
        throw;
    }
    freeDevice(diag_buf);
}

void flow_precond_(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream    )
{
    FF_CHECK_NO_LANES(sol)
    if (sol.ndim - ndim - 1 < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");

    DLTensor diag_t;
    uint8_t * diag_buf = flow_precond_diag(
        sol, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream, diag_t);
    try {
        sym_solve_(sol, hes, diag_t, stream);
    } catch (...) {
        freeDevice(diag_buf);
        throw;
    }
    freeDevice(diag_buf);
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
