#include <stdexcept>
#include <cmath>
#include "posdef.h"
#include "autocast.h"
#include "dlpack.h"
#include "impl/kernels/cuda_switch.h"
#include "impl/kernels/utils.h"
#include "impl/posdef.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

#define VOIDPTR(x)      (static_cast<void*>(static_cast<char*>(x.data) + x.byte_offset))
#define CVOIDPTR(x)     (x.data ? static_cast<const void*>(static_cast<const char*>(x.data) + x.byte_offset) : nullptr)
#define CANUSE32BITS(x) (canUse32BitIndexMath(x.ndim, x.shape, x.strides))

// reduce/accumulation type used by the compact-symmetric kernels.
// jitfields defaults to float64; we do the same for CPU accuracy.
typedef double reduce_t;

/***********************************************************************
 *                              CHECKS                                 *
 ***********************************************************************/

#define CHECK_NO_LANES(tensor)                                          \
    if (tensor.dtype.lanes > 1)                                         \
        throw std::invalid_argument(                                    \
            "Only scalar data types are supported"                      \
        );

#define CHECK_SAME(X, Y, msg)                                           \
    if (X != Y) throw std::invalid_argument(msg);

#define CHECK_SAME_BATCH(X, Y, D)                                       \
    if (X.ndim < D || Y.ndim < D)                                       \
        throw std::invalid_argument(                                    \
            "Number of dimensions does not match"                       \
        );                                                              \
    for (int32_t d=0; d < D; ++d)                                       \
        if (X.shape[d] != Y.shape[d])                                   \
            throw std::invalid_argument(                                \
                "Tensors do not have the same batch shape"              \
            );                                                          \

#define CHECK_SAME_DTYPE(X, Y)                                          \
    if (                                                                \
        (X.dtype.code  != Y.dtype.code) ||                             \
        (X.dtype.bits  != Y.dtype.bits) ||                             \
        (X.dtype.lanes != Y.dtype.lanes)                               \
    )                                                                   \
        throw std::invalid_argument(                                    \
            "Tensors do not have the same data type"                    \
        );

// C such that C*(C+1)/2 == CC (the compact-symmetric length).
static inline int64_t channels_from_packed(int64_t CC)
{
    int64_t C = static_cast<int64_t>((std::sqrt(1.0 + 8.0 * (double)CC) - 1.0) / 2.0);
    // guard against floating-point rounding at the boundary
    while ((C * (C + 1)) / 2 < CC) ++C;
    while (C > 0 && (C * (C + 1)) / 2 > CC) --C;
    if ((C * (C + 1)) / 2 != CC)
        throw std::invalid_argument(
            "Matrix last dimension is not a valid compact-symmetric size"
        );
    return C;
}

/***********************************************************************
 *                             DISPATCH                                *
 ***********************************************************************/

// dtype (float32/float64) x offset (int32/int64), with a static channel
// count C threaded through as a template parameter.
#define DISPATCH_SYM_C_DT(C, func, args...)                             \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return (                                           \
                use_32bits ? func<C,float, int32_t>(args)               \
                           : func<C,float, int64_t>(args));             \
            case 64: return (                                           \
                use_32bits ? func<C,double,int32_t>(args)               \
                           : func<C,double,int64_t>(args));             \
            default: break;                                             \
        };                                                              \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

// select a static C in {1,2,3} for the small cases, fall back to the
// dynamic implementation (C=-1, `nchannel` passed at runtime) otherwise.
#define DISPATCH_SYM_C(func, args...)                                   \
{                                                                       \
    switch (nchannel) {                                                 \
        case 1: DISPATCH_SYM_C_DT( 1, func, args);                      \
        case 2: DISPATCH_SYM_C_DT( 2, func, args);                      \
        case 3: DISPATCH_SYM_C_DT( 3, func, args);                      \
        default: DISPATCH_SYM_C_DT(-1, func, args);                     \
    };                                                                  \
}

// dtype x offset only (dynamic channel count, `nchannel` passed at runtime).
#define DISPATCH_SYM(func, args...)                                     \
{                                                                       \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return (                                           \
                use_32bits ? func<float, int32_t>(args)                 \
                           : func<float, int64_t>(args));               \
            case 64: return (                                           \
                use_32bits ? func<double,int32_t>(args)                 \
                           : func<double,int64_t>(args));               \
            default: break;                                             \
        };                                                              \
        default: break;                                                 \
    };                                                                  \
    throw std::invalid_argument("only floating point data types are supported"); \
}

// layout (Eye/Diag/ESTATICS/Sym/Full, from guess_type) x static C x dtype x
// offset. For ops that carry a vector, so the channel count C is known. On the
// ambiguous small-C packed lengths guess_type resolves to the cheapest layout
// (its Eye>Diag>ESTATICS>Sym>Full order is the efficiency order), and those
// collisions are exact (C=1: all == a scalar; C=2: ESTATICS == Sym matrix).
#define DISPATCH_C_DT(TY, C, func, args...)                             \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return (                                           \
                use_32bits ? func<TY,C,float, int32_t>(args)            \
                           : func<TY,C,float, int64_t>(args));          \
            case 64: return (                                           \
                use_32bits ? func<TY,C,double,int32_t>(args)            \
                           : func<TY,C,double,int64_t>(args));          \
            default: break;                                             \
        };                                                              \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define DISPATCH_C(TY, func, args...)                                   \
    switch (nchannel) {                                                 \
        case 1: DISPATCH_C_DT(TY, 1, func, args);                       \
        case 2: DISPATCH_C_DT(TY, 2, func, args);                       \
        case 3: DISPATCH_C_DT(TY, 3, func, args);                       \
        default: DISPATCH_C_DT(TY, -1, func, args);                     \
    }

#define DISPATCH_TYPE(func, args...)                                    \
{                                                                       \
    switch (mtype) {                                                    \
        case posdef::type::Eye:      DISPATCH_C(posdef::type::Eye,      func, args);\
        case posdef::type::Diag:     DISPATCH_C(posdef::type::Diag,     func, args);\
        case posdef::type::ESTATICS: DISPATCH_C(posdef::type::ESTATICS, func, args);\
        case posdef::type::Sym:      DISPATCH_C(posdef::type::Sym,      func, args);\
        case posdef::type::Full:     DISPATCH_C(posdef::type::Full,     func, args);\
        default: throw std::invalid_argument("unsupported matrix layout");         \
    };                                                                  \
}

/***********************************************************************
 *                              MATVEC                                 *
 ***********************************************************************/

namespace {
template <posdef::type Ty, int C, typename scalar_t, typename offset_t>
inline void _matvec(
          int64_t   nbatch    ,
          int64_t   nchannel  ,
          void    * out       ,   // (*batch, C)
    const void    * hes       ,   // (*batch, CC)
    const void    * inp       ,   // (*batch, C)
    const int64_t * size      ,   // [ndim] out shape
    const int64_t * stride_out ,
    const int64_t * stride_hes ,
    const int64_t * stride_inp )
{
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nbatch+1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nbatch+1);
    const offset_t * _stride_hes = copy_if_needed<offset_t *>(stride_hes, nbatch+1);
    const offset_t * _stride_inp = copy_if_needed<offset_t *>(stride_inp, nbatch+1);
          scalar_t * _out = static_cast<      scalar_t *>(out);
    const scalar_t * _hes = static_cast<const scalar_t *>(hes);
    const scalar_t * _inp = static_cast<const scalar_t *>(inp);
    posdef::matvec<Ty, C, reduce_t, scalar_t, offset_t>(
        static_cast<offset_t>(nbatch), static_cast<offset_t>(nchannel),
        _out, _hes, _inp, _size, _stride_out, _stride_hes, _stride_inp);
    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_hes);
    free_if_needed<int64_t *>(_stride_inp);
}
}

void sym_matvec(
          DLTensor & out_,
    const DLTensor & hessian_,
    const DLTensor & inp_,
          int        /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _hes(hessian_), _inp(inp_);
    DLTensor       & out     = _out.t;
    const DLTensor & hessian = _hes.t;
    const DLTensor & inp     = _inp.t;

    const bool use_32bits = CANUSE32BITS(out) && CANUSE32BITS(hessian) && CANUSE32BITS(inp);
    const int32_t nbatch   = out.ndim - 1;
    const int64_t nchannel = out.shape[out.ndim-1];
    const int64_t CC       = hessian.shape[hessian.ndim-1];
    const auto    code     = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto    bits     = out.dtype.bits;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, hessian)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME      (inp.shape[inp.ndim-1], nchannel, "Input and output channel counts differ")
    CHECK_SAME_BATCH(out, inp,     nbatch)
    CHECK_SAME_BATCH(out, hessian, nbatch)
    const auto mtype = posdef::guess_type<int64_t>(nchannel, CC);  // validates CC

    DISPATCH_TYPE(
        _matvec,
        nbatch, nchannel,
        VOIDPTR(out), CVOIDPTR(hessian), CVOIDPTR(inp),
        out.shape, out.strides, hessian.strides, inp.strides
    )
}

namespace {
template <posdef::type Ty, int C, typename scalar_t, typename offset_t>
inline void _addmatvec_(
          int64_t nbatch, int64_t nchannel,
          void * out, const void * hes, const void * inp,
    const int64_t * size, const int64_t * stride_out,
    const int64_t * stride_hes, const int64_t * stride_inp)
{
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nbatch+1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nbatch+1);
    const offset_t * _stride_hes = copy_if_needed<offset_t *>(stride_hes, nbatch+1);
    const offset_t * _stride_inp = copy_if_needed<offset_t *>(stride_inp, nbatch+1);
          scalar_t * _out = static_cast<      scalar_t *>(out);
    const scalar_t * _hes = static_cast<const scalar_t *>(hes);
    const scalar_t * _inp = static_cast<const scalar_t *>(inp);
    posdef::addmatvec_<Ty, C, reduce_t, scalar_t, offset_t>(
        static_cast<offset_t>(nbatch), static_cast<offset_t>(nchannel),
        _out, _hes, _inp, _size, _stride_out, _stride_hes, _stride_inp);
    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_hes);
    free_if_needed<int64_t *>(_stride_inp);
}

template <posdef::type Ty, int C, typename scalar_t, typename offset_t>
inline void _submatvec_(
          int64_t nbatch, int64_t nchannel,
          void * out, const void * hes, const void * inp,
    const int64_t * size, const int64_t * stride_out,
    const int64_t * stride_hes, const int64_t * stride_inp)
{
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nbatch+1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nbatch+1);
    const offset_t * _stride_hes = copy_if_needed<offset_t *>(stride_hes, nbatch+1);
    const offset_t * _stride_inp = copy_if_needed<offset_t *>(stride_inp, nbatch+1);
          scalar_t * _out = static_cast<      scalar_t *>(out);
    const scalar_t * _hes = static_cast<const scalar_t *>(hes);
    const scalar_t * _inp = static_cast<const scalar_t *>(inp);
    posdef::submatvec_<Ty, C, reduce_t, scalar_t, offset_t>(
        static_cast<offset_t>(nbatch), static_cast<offset_t>(nchannel),
        _out, _hes, _inp, _size, _stride_out, _stride_hes, _stride_inp);
    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_hes);
    free_if_needed<int64_t *>(_stride_inp);
}
}

void sym_addmatvec_(
          DLTensor & out_,
    const DLTensor & hessian_,
    const DLTensor & inp_,
          int        /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _hes(hessian_), _inp(inp_);
    DLTensor       & out     = _out.t;
    const DLTensor & hessian = _hes.t;
    const DLTensor & inp     = _inp.t;

    const bool use_32bits = CANUSE32BITS(out) && CANUSE32BITS(hessian) && CANUSE32BITS(inp);
    const int32_t nbatch   = out.ndim - 1;
    const int64_t nchannel = out.shape[out.ndim-1];
    const int64_t CC       = hessian.shape[hessian.ndim-1];
    const auto    code     = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto    bits     = out.dtype.bits;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, hessian)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME      (inp.shape[inp.ndim-1], nchannel, "Input and output channel counts differ")
    CHECK_SAME_BATCH(out, inp,     nbatch)
    CHECK_SAME_BATCH(out, hessian, nbatch)
    const auto mtype = posdef::guess_type<int64_t>(nchannel, CC);

    DISPATCH_TYPE(
        _addmatvec_,
        nbatch, nchannel,
        VOIDPTR(out), CVOIDPTR(hessian), CVOIDPTR(inp),
        out.shape, out.strides, hessian.strides, inp.strides
    )
}

void sym_submatvec_(
          DLTensor & out_,
    const DLTensor & hessian_,
    const DLTensor & inp_,
          int        /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _hes(hessian_), _inp(inp_);
    DLTensor       & out     = _out.t;
    const DLTensor & hessian = _hes.t;
    const DLTensor & inp     = _inp.t;

    const bool use_32bits = CANUSE32BITS(out) && CANUSE32BITS(hessian) && CANUSE32BITS(inp);
    const int32_t nbatch   = out.ndim - 1;
    const int64_t nchannel = out.shape[out.ndim-1];
    const int64_t CC       = hessian.shape[hessian.ndim-1];
    const auto    code     = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto    bits     = out.dtype.bits;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, hessian)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME      (inp.shape[inp.ndim-1], nchannel, "Input and output channel counts differ")
    CHECK_SAME_BATCH(out, inp,     nbatch)
    CHECK_SAME_BATCH(out, hessian, nbatch)
    const auto mtype = posdef::guess_type<int64_t>(nchannel, CC);

    DISPATCH_TYPE(
        _submatvec_,
        nbatch, nchannel,
        VOIDPTR(out), CVOIDPTR(hessian), CVOIDPTR(inp),
        out.shape, out.strides, hessian.strides, inp.strides
    )
}

/***********************************************************************
 *                          MATVEC BACKWARD                           *
 ***********************************************************************/

namespace {
template <int C, typename scalar_t, typename offset_t>
inline void _sym_matvec_backward(
          int64_t nbatch, int64_t nchannel,
          void * out, const void * grd, const void * inp,
    const int64_t * size, const int64_t * stride_out,
    const int64_t * stride_grd, const int64_t * stride_inp)
{
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nbatch+1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nbatch+1);
    const offset_t * _stride_grd = copy_if_needed<offset_t *>(stride_grd, nbatch+1);
    const offset_t * _stride_inp = copy_if_needed<offset_t *>(stride_inp, nbatch+1);
          scalar_t * _out = static_cast<      scalar_t *>(out);
    const scalar_t * _grd = static_cast<const scalar_t *>(grd);
    const scalar_t * _inp = static_cast<const scalar_t *>(inp);
    posdef::sym_matvec_backward<C, reduce_t, scalar_t, offset_t>(
        static_cast<offset_t>(nbatch), static_cast<offset_t>(nchannel),
        _out, _grd, _inp, _size, _stride_out, _stride_grd, _stride_inp);
    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_grd);
    free_if_needed<int64_t *>(_stride_inp);
}
}

void sym_matvec_backward(
          DLTensor & out_,      // (*batch, C*(C+1)/2)
    const DLTensor & grd_,      // (*batch, C)
    const DLTensor & inp_,      // (*batch, C)
          int        /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _grd(grd_), _inp(inp_);
    DLTensor       & out = _out.t;
    const DLTensor & grd = _grd.t;
    const DLTensor & inp = _inp.t;

    const bool use_32bits = CANUSE32BITS(out) && CANUSE32BITS(grd) && CANUSE32BITS(inp);
    const int32_t nbatch   = grd.ndim - 1;
    const int64_t nchannel = grd.shape[grd.ndim-1];
    const auto    code     = static_cast<DLDataTypeCode>(grd.dtype.code);
    const auto    bits     = grd.dtype.bits;
    CHECK_NO_LANES  (grd)
    CHECK_SAME_DTYPE(grd, out)
    CHECK_SAME_DTYPE(grd, inp)
    CHECK_SAME      (inp.shape[inp.ndim-1], nchannel, "Input and grad channel counts differ")
    CHECK_SAME      (out.shape[out.ndim-1]*2, nchannel*(nchannel+1), "Matrix is not compatible with the channel count")
    CHECK_SAME_BATCH(grd, out, nbatch)
    CHECK_SAME_BATCH(grd, inp, nbatch)

    DISPATCH_SYM_C(
        _sym_matvec_backward,
        nbatch, nchannel,
        VOIDPTR(out), CVOIDPTR(grd), CVOIDPTR(inp),
        grd.shape, out.strides, grd.strides, inp.strides
    )
}

/***********************************************************************
 *                               SOLVE                                *
 ***********************************************************************/

namespace {
template <typename scalar_t, typename offset_t>
inline void _sym_solve(
          int64_t nbatch, int64_t nchannel,
          void * out, const void * inp, const void * hes, const void * wgt,
    const int64_t * size, const int64_t * stride_out,
    const int64_t * stride_inp, const int64_t * stride_hes,
    const int64_t * stride_wgt)
{
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nbatch+1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nbatch+1);
    const offset_t * _stride_inp = copy_if_needed<offset_t *>(stride_inp, nbatch+1);
    const offset_t * _stride_hes = copy_if_needed<offset_t *>(stride_hes, nbatch+1);
    const offset_t * _stride_wgt = wgt ? copy_if_needed<offset_t *>(stride_wgt, nbatch+1) : nullptr;
          scalar_t * _out = static_cast<      scalar_t *>(out);
    const scalar_t * _inp = static_cast<const scalar_t *>(inp);
    const scalar_t * _hes = static_cast<const scalar_t *>(hes);
    const scalar_t * _wgt = static_cast<const scalar_t *>(wgt);
    posdef::sym_solve<reduce_t, scalar_t, offset_t>(
        static_cast<offset_t>(nbatch), static_cast<offset_t>(nchannel),
        _out, _inp, _hes, _wgt, _size, _stride_out, _stride_inp, _stride_hes, _stride_wgt);
    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_inp);
    free_if_needed<int64_t *>(_stride_hes);
    if (_stride_wgt) free_if_needed<int64_t *>(_stride_wgt);
}
}

void sym_solve(
          DLTensor & out_,
    const DLTensor & hessian_,
    const DLTensor & inp_,
    const DLTensor & weight_,
          int        /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch. The weight is
    // optional (null-data placeholder) and only normalised when present.
    ContiguousStrides _out(out_), _hes(hessian_), _inp(inp_),
                      _wgt(weight_, weight_.data != nullptr);
    DLTensor       & out     = _out.t;
    const DLTensor & hessian = _hes.t;
    const DLTensor & inp     = _inp.t;
    const DLTensor & weight  = _wgt.t;

    const bool has_wgt = (weight.data != nullptr);
    bool use_32bits = CANUSE32BITS(out) && CANUSE32BITS(hessian) && CANUSE32BITS(inp);
    if (has_wgt) use_32bits = use_32bits && CANUSE32BITS(weight);
    const int32_t nbatch   = out.ndim - 1;
    const int64_t nchannel = out.shape[out.ndim-1];
    const int64_t CC       = hessian.shape[hessian.ndim-1];
    const auto    code     = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto    bits     = out.dtype.bits;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, hessian)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME      (inp.shape[inp.ndim-1], nchannel, "Input and output channel counts differ")
    CHECK_SAME      (hessian.shape[hessian.ndim-1]*2, nchannel*(nchannel+1), "Matrix is not compatible with the channel count")
    CHECK_SAME_BATCH(out, inp,     nbatch)
    CHECK_SAME_BATCH(out, hessian, nbatch)
    if (has_wgt) { CHECK_SAME_DTYPE(out, weight) CHECK_SAME_BATCH(out, weight, nbatch) }

    DISPATCH_SYM(
        _sym_solve,
        nbatch, nchannel,
        VOIDPTR(out), CVOIDPTR(inp), CVOIDPTR(hessian),
        has_wgt ? VOIDPTR(weight) : nullptr,
        out.shape, out.strides, inp.strides, hessian.strides,
        has_wgt ? weight.strides : nullptr
    )
}

namespace {
template <typename scalar_t, typename offset_t>
inline void _sym_solve_(
          int64_t nbatch, int64_t nchannel,
          void * out, const void * hes, const void * wgt,
    const int64_t * size, const int64_t * stride_out,
    const int64_t * stride_hes, const int64_t * stride_wgt)
{
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nbatch+1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nbatch+1);
    const offset_t * _stride_hes = copy_if_needed<offset_t *>(stride_hes, nbatch+1);
    const offset_t * _stride_wgt = wgt ? copy_if_needed<offset_t *>(stride_wgt, nbatch+1) : nullptr;
          scalar_t * _out = static_cast<      scalar_t *>(out);
    const scalar_t * _hes = static_cast<const scalar_t *>(hes);
    const scalar_t * _wgt = static_cast<const scalar_t *>(wgt);
    posdef::sym_solve_<reduce_t, scalar_t, offset_t>(
        static_cast<offset_t>(nbatch), static_cast<offset_t>(nchannel),
        _out, _hes, _wgt, _size, _stride_out, _stride_hes, _stride_wgt);
    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_hes);
    if (_stride_wgt) free_if_needed<int64_t *>(_stride_wgt);
}
}

void sym_solve_(
          DLTensor & inp_out_,
    const DLTensor & hessian_,
    const DLTensor & weight_,
          int        /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch. The weight is
    // optional (null-data placeholder) and only normalised when present.
    ContiguousStrides _io(inp_out_), _hes(hessian_),
                      _wgt(weight_, weight_.data != nullptr);
    DLTensor       & inp_out = _io.t;
    const DLTensor & hessian = _hes.t;
    const DLTensor & weight  = _wgt.t;

    const bool has_wgt = (weight.data != nullptr);
    bool use_32bits = CANUSE32BITS(inp_out) && CANUSE32BITS(hessian);
    if (has_wgt) use_32bits = use_32bits && CANUSE32BITS(weight);
    const int32_t nbatch   = inp_out.ndim - 1;
    const int64_t nchannel = inp_out.shape[inp_out.ndim-1];
    const auto    code     = static_cast<DLDataTypeCode>(inp_out.dtype.code);
    const auto    bits     = inp_out.dtype.bits;
    CHECK_NO_LANES  (inp_out)
    CHECK_SAME_DTYPE(inp_out, hessian)
    CHECK_SAME      (hessian.shape[hessian.ndim-1]*2, nchannel*(nchannel+1), "Matrix is not compatible with the channel count")
    CHECK_SAME_BATCH(inp_out, hessian, nbatch)
    if (has_wgt) { CHECK_SAME_DTYPE(inp_out, weight) CHECK_SAME_BATCH(inp_out, weight, nbatch) }

    DISPATCH_SYM(
        _sym_solve_,
        nbatch, nchannel,
        VOIDPTR(inp_out), CVOIDPTR(hessian),
        has_wgt ? VOIDPTR(weight) : nullptr,
        inp_out.shape, inp_out.strides, hessian.strides,
        has_wgt ? weight.strides : nullptr
    )
}

/***********************************************************************
 *                              INVERT                                *
 ***********************************************************************/

namespace {
template <typename scalar_t, typename offset_t>
inline void _sym_invert(
          int64_t nbatch, int64_t nchannel,
          void * out, const void * hes,
    const int64_t * size, const int64_t * stride_out, const int64_t * stride_hes)
{
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nbatch+1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nbatch+1);
    const offset_t * _stride_hes = copy_if_needed<offset_t *>(stride_hes, nbatch+1);
          scalar_t * _out = static_cast<      scalar_t *>(out);
    const scalar_t * _hes = static_cast<const scalar_t *>(hes);
    posdef::sym_invert<reduce_t, scalar_t, offset_t>(
        static_cast<offset_t>(nbatch), static_cast<offset_t>(nchannel),
        _out, _hes, _size, _stride_out, _stride_hes);
    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_hes);
}
}

void sym_invert(
          DLTensor & out_,      // (*batch, C*(C+1)/2)
    const DLTensor & hessian_,  // (*batch, C*(C+1)/2)
          int        /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _hes(hessian_);
    DLTensor       & out     = _out.t;
    const DLTensor & hessian = _hes.t;

    const bool use_32bits = CANUSE32BITS(out) && CANUSE32BITS(hessian);
    const int32_t nbatch   = out.ndim - 1;
    const int64_t CC       = hessian.shape[hessian.ndim-1];
    const int64_t nchannel = channels_from_packed(CC);
    const auto    code     = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto    bits     = out.dtype.bits;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, hessian)
    CHECK_SAME      (out.shape[out.ndim-1], CC, "Output and matrix must share the compact layout")
    CHECK_SAME_BATCH(out, hessian, nbatch)

    DISPATCH_SYM(
        _sym_invert,
        nbatch, nchannel,
        VOIDPTR(out), CVOIDPTR(hessian),
        out.shape, out.strides, hessian.strides
    )
}

namespace {
template <typename scalar_t, typename offset_t>
inline void _sym_invert_(
          int64_t nbatch, int64_t nchannel,
          void * hes,
    const int64_t * size, const int64_t * stride)
{
    const offset_t * _size   = copy_if_needed<offset_t *>(size,   nbatch+1);
    const offset_t * _stride = copy_if_needed<offset_t *>(stride, nbatch+1);
          scalar_t * _hes = static_cast<scalar_t *>(hes);
    posdef::sym_invert_<reduce_t, scalar_t, offset_t>(
        static_cast<offset_t>(nbatch), static_cast<offset_t>(nchannel),
        _hes, _size, _stride);
    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride);
}
}

void sym_invert_(
          DLTensor & hessian_,
          int        /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _hes(hessian_);
    DLTensor & hessian = _hes.t;

    const bool use_32bits = CANUSE32BITS(hessian);
    const int32_t nbatch   = hessian.ndim - 1;
    const int64_t CC       = hessian.shape[hessian.ndim-1];
    const int64_t nchannel = channels_from_packed(CC);
    const auto    code     = static_cast<DLDataTypeCode>(hessian.dtype.code);
    const auto    bits     = hessian.dtype.bits;
    CHECK_NO_LANES(hessian)

    DISPATCH_SYM(
        _sym_invert_,
        nbatch, nchannel,
        VOIDPTR(hessian),
        hessian.shape, hessian.strides
    )
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
