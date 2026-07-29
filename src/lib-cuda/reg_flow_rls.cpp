#include <stdexcept>
#include <string>
#include <cstdint>
#include "reg_flow.h"
#include "posdef.h"
#include "autocast.h"
#include "dlpack.h"
#include "impl/kernels/cuda_switch.h"
#include "impl/kernels/bounds.h"
#include "impl/kernels/utils.h"
#include "impl/reg_flow.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

#define VOIDPTR(x)      (static_cast<void*>(static_cast<char*>(x.data) + x.byte_offset))
#define CVOIDPTR(x)     (static_cast<const void*>(static_cast<const char*>(x.data) + x.byte_offset))
#define CANUSE32BITS(x) (canUse32BitIndexMath(x.ndim, x.shape, x.strides))

// reduction / accumulation type (matches jitfields' float64 default)
typedef double reduce_t;

/***********************************************************************
 *                              CHECKS                                 *
 ***********************************************************************/

#define CHECK_NO_LANES(tensor)                                          \
    if (tensor.dtype.lanes > 1)                                         \
        throw std::invalid_argument("Only scalar data types are supported");

#define CHECK_SAME(X, Y, msg)                                           \
    if (X != Y) throw std::invalid_argument(msg);

#define CHECK_SAME_DTYPE(X, Y)                                          \
    if ((X.dtype.code  != Y.dtype.code) ||                              \
        (X.dtype.bits  != Y.dtype.bits) ||                             \
        (X.dtype.lanes != Y.dtype.lanes))                              \
        throw std::invalid_argument("Tensors do not have the same data type");

#define CHECK_SAME_SHAPE(X, Y, D)                                       \
    for (int32_t d=0; d < D; ++d)                                       \
        if (X.shape[d] != Y.shape[d])                                   \
            throw std::invalid_argument("Tensors do not have the same shape");

/***********************************************************************
 *                             WRAPPERS                                *
 ***********************************************************************/

namespace {

// int -> cudaStream_t (0 == default stream). The public ABI carries the stream
// as an int; the cuda-impl launchers take a cudaStream_t. Mirrors
// pushpull::_pp_stream in the cuda-impl layer.
static inline cudaStream_t _reg_stream(int stream)
{
    return reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream));
}

// Reweighted-least-squares (JRLS) variant of `_flow_matvec`: an extra
// per-voxel weight map `wgt` modulates the penalty strength. A non-zero
// shears/div selects the weighted Lamé stencil (which also folds in
// absolute/membrane); otherwise the weighted membrane stencil is used
// (covers absolute-only too). Mirrors the CPU `_flow_matvec_rls`; bending is
// rejected by the public wrapper before this is ever called.
template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _flow_matvec_rls(
    const bound::BoundVec & bvec,
          int64_t   nbatch     ,
          void    * out        ,
    const void    * inp        ,
    const void    * wgt        ,
    const double  * voxel_size ,
          double    absolute   ,
          double    membrane   ,
          double    shears     ,
          double    div        ,
    const int64_t * size       ,
    const int64_t * stride_out ,
    const int64_t * stride_inp ,
    const int64_t * stride_wgt ,
          cudaStream_t stream  )
{
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nall1);
    const offset_t * _stride_inp = copy_if_needed<offset_t *>(stride_inp, nall1);
    const offset_t * _stride_wgt = copy_if_needed<offset_t *>(stride_wgt, nall1);
          scalar_t * _out = static_cast<      scalar_t *>(out);
    const scalar_t * _inp = static_cast<const scalar_t *>(inp);
    const scalar_t * _wgt = static_cast<const scalar_t *>(wgt);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    if (shears != 0.0 || div != 0.0)
        reg_flow::matvec_lame_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp, _wgt,
            _size, _stride_out, _stride_inp, _stride_wgt, vx,
            absolute, membrane, shears, div, stream);
    else
        reg_flow::matvec_membrane_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp, _wgt,
            _size, _stride_out, _stride_inp, _stride_wgt, vx, absolute, membrane, stream);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_inp);
    free_if_needed<int64_t *>(_stride_wgt);
}

template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _flow_diag_rls(
    const bound::BoundVec & bvec,
          int64_t   nbatch     ,
          void    * out        ,
    const void    * wgt        ,
    const double  * voxel_size ,
          double    absolute   ,
          double    membrane   ,
          double    shears     ,
          double    div        ,
    const int64_t * size       ,
    const int64_t * stride_out ,
    const int64_t * stride_wgt ,
          cudaStream_t stream  )
{
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nall1);
    const offset_t * _stride_wgt = copy_if_needed<offset_t *>(stride_wgt, nall1);
          scalar_t * _out = static_cast<scalar_t *>(out);
    const scalar_t * _wgt = static_cast<const scalar_t *>(wgt);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    if (shears != 0.0 || div != 0.0)
        reg_flow::diag_lame_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _wgt,
            _size, _stride_out, _stride_wgt, vx, absolute, membrane, shears, div, stream);
    else
        reg_flow::diag_membrane_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _wgt,
            _size, _stride_out, _stride_wgt, vx, absolute, membrane, stream);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_wgt);
}

template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _flow_relax_rls(
    const bound::BoundVec & bvec,
          int64_t   nbatch     ,
          void    * sol        ,
    const void    * hes        ,
    const void    * grd        ,
    const void    * wgt        ,
    const double  * voxel_size ,
          double    absolute   ,
          double    membrane   ,
          double    shears     ,
          double    div        ,
          int       niter      ,
    const int64_t * size       ,
    const int64_t * stride_sol ,
    const int64_t * stride_hes ,
    const int64_t * stride_grd ,
    const int64_t * stride_wgt ,
          cudaStream_t stream  )
{
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_sol = copy_if_needed<offset_t *>(stride_sol, nall1);
    const offset_t * _stride_hes = copy_if_needed<offset_t *>(stride_hes, nall1);
    const offset_t * _stride_grd = copy_if_needed<offset_t *>(stride_grd, nall1);
    const offset_t * _stride_wgt = copy_if_needed<offset_t *>(stride_wgt, nall1);
          scalar_t * _sol = static_cast<      scalar_t *>(sol);
    const scalar_t * _hes = static_cast<const scalar_t *>(hes);
    const scalar_t * _grd = static_cast<const scalar_t *>(grd);
    const scalar_t * _wgt = static_cast<const scalar_t *>(wgt);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    if (shears != 0.0 || div != 0.0)
        reg_flow::relax_lame_jrls_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd, _wgt,
            _size, _stride_sol, _stride_hes, _stride_grd, _stride_wgt, vx,
            absolute, membrane, shears, div, niter, stream);
    else
        reg_flow::relax_membrane_jrls_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd, _wgt,
            _size, _stride_sol, _stride_hes, _stride_grd, _stride_wgt, vx,
            absolute, membrane, niter, stream);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_sol);
    free_if_needed<int64_t *>(_stride_hes);
    free_if_needed<int64_t *>(_stride_grd);
    free_if_needed<int64_t *>(_stride_wgt);
}

} // anonymous namespace

/***********************************************************************
 *                            DISPATCH                                 *
 ***********************************************************************/

#define BND1(B) B
#define BND2(B) B, B
#define BND3(B) B, B, B
// Which of these boundary conditions gets a dedicated (static) instantiation
// and which shares the single Dynamic (runtime) one is a build-time choice --
// see FF_STATIC_BOUND_* in kernels/bounds.h. `bvec` carries the runtime
// condition for whichever ones fall back to Dynamic.
#define BOUND_SWITCH(DT, NDIM, BND)                                     \
    switch (bnd) {                                                      \
        case bound::type::Zero:      DT(NDIM, BND(FF_BOUND_ZERO));      break; \
        case bound::type::Replicate: DT(NDIM, BND(FF_BOUND_REPLICATE)); break; \
        case bound::type::DCT1:      DT(NDIM, BND(FF_BOUND_DCT1));      break; \
        case bound::type::DCT2:      DT(NDIM, BND(FF_BOUND_DCT2));      break; \
        case bound::type::DST1:      DT(NDIM, BND(FF_BOUND_DST1));      break; \
        case bound::type::DST2:      DT(NDIM, BND(FF_BOUND_DST2));      break; \
        case bound::type::DFT:       DT(NDIM, BND(FF_BOUND_DFT));       break; \
        case bound::type::NoCheck:   DT(NDIM, BND(FF_BOUND_NOCHECK));   break; \
        default: throw std::invalid_argument("Unsupported boundary condition"); \
    }

#define NDIM_SWITCH(DT)                                                 \
    switch (ndim) {                                                     \
        case 1: BOUND_SWITCH(DT, 1, BND1); break;                       \
        case 2: BOUND_SWITCH(DT, 2, BND2); break;                       \
        case 3: BOUND_SWITCH(DT, 3, BND3); break;                       \
        default: throw std::invalid_argument("Only 1D, 2D and 3D flow are supported"); \
    }

#define RLS_MV_DT(NDIM, BNDS...)                                        \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_matvec_rls<NDIM, float,  int32_t, BNDS>(RLS_MV_ARGS) \
                : _flow_matvec_rls<NDIM, float,  int64_t, BNDS>(RLS_MV_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_matvec_rls<NDIM, double, int32_t, BNDS>(RLS_MV_ARGS) \
                : _flow_matvec_rls<NDIM, double, int64_t, BNDS>(RLS_MV_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define RLS_DG_DT(NDIM, BNDS...)                                        \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_diag_rls<NDIM, float,  int32_t, BNDS>(RLS_DG_ARGS) \
                : _flow_diag_rls<NDIM, float,  int64_t, BNDS>(RLS_DG_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_diag_rls<NDIM, double, int32_t, BNDS>(RLS_DG_ARGS) \
                : _flow_diag_rls<NDIM, double, int64_t, BNDS>(RLS_DG_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define RLS_RX_DT(NDIM, BNDS...)                                        \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_relax_rls<NDIM, float,  int32_t, BNDS>(RLS_RX_ARGS) \
                : _flow_relax_rls<NDIM, float,  int64_t, BNDS>(RLS_RX_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_relax_rls<NDIM, double, int32_t, BNDS>(RLS_RX_ARGS) \
                : _flow_relax_rls<NDIM, double, int64_t, BNDS>(RLS_RX_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

// bending has no _jrls kernel at the impl layer (matches jitfields, which
// never wired that combination either) -- reject it explicitly rather than
// silently falling back to the unweighted stencil.
static inline void flow_rls_check_bending(double bending, const char * who)
{
    if (bending != 0.0)
        throw std::invalid_argument(
            std::string(who) + ": bending penalty is not supported with "
            "RLS/JRLS weighting");
}

void flow_matvec_rls(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const DLTensor & wgt_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream
)
{
    flow_rls_check_bending(bending, "flow_matvec_rls");

    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _inp(inp_), _wgt(wgt_);
    DLTensor       & out = _out.t;
    const DLTensor & inp = _inp.t;
    const DLTensor & wgt = _wgt.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME_DTYPE(out, wgt)
    CHECK_SAME      (out.ndim, inp.ndim, "Tensors do not have the same number of dimensions")
    CHECK_SAME      (out.ndim, wgt.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME      (out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    CHECK_SAME_SHAPE(out, inp, out.ndim)
    CHECK_SAME_SHAPE(out, wgt, out.ndim - 1)
    CHECK_SAME      (wgt.shape[wgt.ndim-1], (int64_t)1,
                     "flow_matvec_rls: weight tensor's trailing dimension must be 1")

    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp) && CANUSE32BITS(wgt);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    const bound::BoundVec bvec(bnd);
    const cudaStream_t cstream = _reg_stream(stream);

#define RLS_MV_ARGS bvec, static_cast<int64_t>(nbatch), VOIDPTR(out), CVOIDPTR(inp), \
                CVOIDPTR(wgt), voxel_size, absolute, membrane, shears, div,     \
                out.shape, out.strides, inp.strides, wgt.strides, cstream
    NDIM_SWITCH(RLS_MV_DT)
#undef RLS_MV_ARGS
}

void flow_diag_rls(
          DLTensor & out_      ,
    const DLTensor & wgt_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream
)
{
    flow_rls_check_bending(bending, "flow_diag_rls");

    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _wgt(wgt_);
    DLTensor       & out = _out.t;
    const DLTensor & wgt = _wgt.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, wgt)
    CHECK_SAME      (out.ndim, wgt.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME      (out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    CHECK_SAME_SHAPE(out, wgt, out.ndim - 1)
    CHECK_SAME      (wgt.shape[wgt.ndim-1], (int64_t)1,
                     "flow_diag_rls: weight tensor's trailing dimension must be 1")

    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(wgt);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    const bound::BoundVec bvec(bnd);
    const cudaStream_t cstream = _reg_stream(stream);

#define RLS_DG_ARGS bvec, static_cast<int64_t>(nbatch), VOIDPTR(out), CVOIDPTR(wgt), \
                voxel_size, absolute, membrane, shears, div,                   \
                out.shape, out.strides, wgt.strides, cstream
    NDIM_SWITCH(RLS_DG_DT)
#undef RLS_DG_ARGS
}

void flow_relax_rls(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const DLTensor & wgt_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        nb_iter   ,
          int        stream
)
{
    flow_rls_check_bending(bending, "flow_relax_rls");

    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _sol(sol), _hes(hes), _grd(grd), _wgt(wgt_);
    DLTensor       & s = _sol.t;
    const DLTensor & h = _hes.t;
    const DLTensor & g = _grd.t;
    const DLTensor & wgt = _wgt.t;

    const int32_t nbatch = s.ndim - ndim - 1;
    CHECK_NO_LANES  (s)
    CHECK_SAME_DTYPE(s, h)
    CHECK_SAME_DTYPE(s, g)
    CHECK_SAME_DTYPE(s, wgt)
    CHECK_SAME      (s.ndim, g.ndim, "Tensors do not have the same number of dimensions")
    CHECK_SAME      (s.ndim, h.ndim, "Tensors do not have the same number of dimensions")
    CHECK_SAME      (s.ndim, wgt.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME      (s.shape[s.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    CHECK_SAME      (g.shape[g.ndim-1], (int64_t)ndim, "Gradient channel dimension must equal ndim")
    CHECK_SAME_SHAPE(s, g, s.ndim)
    CHECK_SAME_SHAPE(s, wgt, s.ndim - 1)
    CHECK_SAME      (wgt.shape[wgt.ndim-1], (int64_t)1,
                     "flow_relax_rls: weight tensor's trailing dimension must be 1")

    const bool     use_32bits = CANUSE32BITS(s) && CANUSE32BITS(h) &&
                                CANUSE32BITS(g) && CANUSE32BITS(wgt);
    const auto     code = static_cast<DLDataTypeCode>(s.dtype.code);
    const auto     bits = s.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    const bound::BoundVec bvec(bnd);
    const cudaStream_t cstream = _reg_stream(stream);

#define RLS_RX_ARGS bvec, static_cast<int64_t>(nbatch), VOIDPTR(s), CVOIDPTR(h),   \
                CVOIDPTR(g), CVOIDPTR(wgt), voxel_size, absolute, membrane,  \
                shears, div, nb_iter, s.shape, s.strides, h.strides,         \
                g.strides, wgt.strides, cstream
    NDIM_SWITCH(RLS_RX_DT)
#undef RLS_RX_ARGS
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
