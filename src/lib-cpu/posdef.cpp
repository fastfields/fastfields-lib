#include <stdexcept>
#include <cmath>
#include "posdef.h"
#include "dlpack.h"
// R7 (TEENY-MIGRATION.md sec. 9): fastfields vendors DLPack v1.2, teeny v1.1,
// and both use the guard DLPACK_DLPACK_H_ -- so whichever is seen first wins
// for the whole TU. Our "dlpack.h" is included ABOVE on purpose; keep it there.
#include <teeny/dlpack.h>
#include "impl/kernels/cuda_switch.h"
#include "impl/kernels/utils.h"
#include "impl/posdef.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

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

// The impl reads exactly nbatch+1 strides per tensor (the batch axes plus one
// trailing channel/packed axis). A tensor with extra trailing dims would pass
// CHECK_SAME_BATCH yet be decoded against the wrong last axis, so pin the rank.
#define CHECK_RANK(X, R)                                                \
    if (X.ndim != (R))                                                  \
        throw std::invalid_argument(                                    \
            "Tensor must have exactly one channel axis after the batch axes" \
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

// dtype (float32/float64), with a static channel count C threaded through as a
// template parameter.
//
// D1/R5 (TEENY-MIGRATION.md sec. 9): there is no int32 offset arm any more. The
// CPU narrowing was measured a wash on a 64-bit ALU, and the carriers teeny
// imports below always decode in int64, so every `use_32bits ? f<...,int32_t>
// : f<...,int64_t>` ternary in this file collapses to the int64 instantiation.
// Small tensors now run that one; the instantiation count halves.
#define DISPATCH_SYM_C_DT(C, func, args...)                             \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return func<C,float >(args);                       \
            case 64: return func<C,double>(args);                       \
            default: break;                                             \
        };                                                              \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

// select a static C in {1,2,3} for the small cases, fall back to the
// dynamic implementation (C=-1, the impl reads the channel count off the
// carrier) otherwise.
#define DISPATCH_SYM_C(func, args...)                                   \
{                                                                       \
    switch (nchannel) {                                                 \
        case 1: DISPATCH_SYM_C_DT( 1, func, args);                      \
        case 2: DISPATCH_SYM_C_DT( 2, func, args);                      \
        case 3: DISPATCH_SYM_C_DT( 3, func, args);                      \
        default: DISPATCH_SYM_C_DT(-1, func, args);                     \
    };                                                                  \
}

// dtype only (dynamic channel count, read off the carrier by the impl).
#define DISPATCH_SYM(func, args...)                                     \
{                                                                       \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return func<float >(args);                         \
            case 64: return func<double>(args);                         \
            default: break;                                             \
        };                                                              \
        default: break;                                                 \
    };                                                                  \
    throw std::invalid_argument("only floating point data types are supported"); \
}

// layout (Eye/Diag/ESTATICS/Sym/Full, from guess_type) x static C x dtype.
// For ops that carry a vector, so the channel count C is known. On the
// ambiguous small-C packed lengths guess_type resolves to the cheapest layout
// (its Eye>Diag>ESTATICS>Sym>Full order is the efficiency order), and those
// collisions are exact (C=1: all == a scalar; C=2: ESTATICS == Sym matrix).
#define DISPATCH_C_DT(TY, C, func, args...)                             \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return func<TY,C,float >(args);                    \
            case 64: return func<TY,C,double>(args);                    \
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

// layout x dtype, DYNAMIC channel count (for solve, which does not specialise
// on a static C).
#define DISPATCH_TY_DT(TY, func, args...)                               \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return func<TY,float >(args);                      \
            case 64: return func<TY,double>(args);                      \
            default: break;                                             \
        };                                                              \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define DISPATCH_TYPE_DYN(func, args...)                                \
{                                                                       \
    switch (mtype) {                                                    \
        case posdef::type::Eye:      DISPATCH_TY_DT(posdef::type::Eye,      func, args);\
        case posdef::type::Diag:     DISPATCH_TY_DT(posdef::type::Diag,     func, args);\
        case posdef::type::ESTATICS: DISPATCH_TY_DT(posdef::type::ESTATICS, func, args);\
        case posdef::type::Sym:      DISPATCH_TY_DT(posdef::type::Sym,      func, args);\
        case posdef::type::Full:     DISPATCH_TY_DT(posdef::type::Full,     func, args);\
        default: throw std::invalid_argument("unsupported matrix layout");             \
    };                                                                  \
}

/***********************************************************************
 *                              MATVEC                                 *
 ***********************************************************************/

// The dtype arm's IMPORT POINT. `tny::from_dlpack` does by construction the
// three things this file used to do by hand: it folds `byte_offset` into the
// data pointer (was VOIDPTR/CVOIDPTR), expands a NULL `strides` field to compact
// row-major (was ContiguousStrides), and copies the shape/stride metadata into a
// self-contained carrier -- built ONCE per tensor, from that tensor's OWN
// descriptor, so no operand is ever decoded against another's geometry (R3).
// Read-only operands import as carriers of `const scalar_t` (R4).
namespace {
template <posdef::type Ty, int C, typename scalar_t>
inline void _matvec(
          DLTensor & out ,   // (*batch, C)
    const DLTensor & hes ,   // (*batch, CC)
    const DLTensor & inp )   // (*batch, C)
{
    auto ao = tny::from_dlpack<      scalar_t>(&out);
    auto ah = tny::from_dlpack<const scalar_t>(&hes);
    auto ai = tny::from_dlpack<const scalar_t>(&inp);
    posdef::matvec<Ty, C, reduce_t>(ao, ah, ai);
}
}

void sym_matvec(
          DLTensor & out,
    const DLTensor & hessian,
    const DLTensor & inp,
          int        /* stream <unused> */
)
{
    const int32_t nbatch   = out.ndim - 1;
    const int64_t nchannel = out.shape[out.ndim-1];
    const int64_t CC       = hessian.shape[hessian.ndim-1];
    const auto    code     = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto    bits     = out.dtype.bits;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, hessian)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME      (inp.shape[inp.ndim-1], nchannel, "Input and output channel counts differ")
    CHECK_RANK      (out,     nbatch+1)
    CHECK_RANK      (hessian, nbatch+1)
    CHECK_RANK      (inp,     nbatch+1)
    CHECK_SAME_BATCH(out, inp,     nbatch)
    CHECK_SAME_BATCH(out, hessian, nbatch)
    const auto mtype = posdef::guess_type<int64_t>(nchannel, CC);  // validates CC

    DISPATCH_TYPE(_matvec, out, hessian, inp)
}

namespace {
template <posdef::type Ty, int C, typename scalar_t>
inline void _addmatvec_(DLTensor & out, const DLTensor & hes, const DLTensor & inp)
{
    auto ao = tny::from_dlpack<      scalar_t>(&out);
    auto ah = tny::from_dlpack<const scalar_t>(&hes);
    auto ai = tny::from_dlpack<const scalar_t>(&inp);
    posdef::addmatvec_<Ty, C, reduce_t>(ao, ah, ai);
}

template <posdef::type Ty, int C, typename scalar_t>
inline void _submatvec_(DLTensor & out, const DLTensor & hes, const DLTensor & inp)
{
    auto ao = tny::from_dlpack<      scalar_t>(&out);
    auto ah = tny::from_dlpack<const scalar_t>(&hes);
    auto ai = tny::from_dlpack<const scalar_t>(&inp);
    posdef::submatvec_<Ty, C, reduce_t>(ao, ah, ai);
}
}

void sym_addmatvec_(
          DLTensor & out,
    const DLTensor & hessian,
    const DLTensor & inp,
          int        /* stream <unused> */
)
{
    const int32_t nbatch   = out.ndim - 1;
    const int64_t nchannel = out.shape[out.ndim-1];
    const int64_t CC       = hessian.shape[hessian.ndim-1];
    const auto    code     = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto    bits     = out.dtype.bits;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, hessian)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME      (inp.shape[inp.ndim-1], nchannel, "Input and output channel counts differ")
    CHECK_RANK      (out,     nbatch+1)
    CHECK_RANK      (hessian, nbatch+1)
    CHECK_RANK      (inp,     nbatch+1)
    CHECK_SAME_BATCH(out, inp,     nbatch)
    CHECK_SAME_BATCH(out, hessian, nbatch)
    const auto mtype = posdef::guess_type<int64_t>(nchannel, CC);

    DISPATCH_TYPE(_addmatvec_, out, hessian, inp)
}

void sym_submatvec_(
          DLTensor & out,
    const DLTensor & hessian,
    const DLTensor & inp,
          int        /* stream <unused> */
)
{
    const int32_t nbatch   = out.ndim - 1;
    const int64_t nchannel = out.shape[out.ndim-1];
    const int64_t CC       = hessian.shape[hessian.ndim-1];
    const auto    code     = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto    bits     = out.dtype.bits;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, hessian)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME      (inp.shape[inp.ndim-1], nchannel, "Input and output channel counts differ")
    CHECK_RANK      (out,     nbatch+1)
    CHECK_RANK      (hessian, nbatch+1)
    CHECK_RANK      (inp,     nbatch+1)
    CHECK_SAME_BATCH(out, inp,     nbatch)
    CHECK_SAME_BATCH(out, hessian, nbatch)
    const auto mtype = posdef::guess_type<int64_t>(nchannel, CC);

    DISPATCH_TYPE(_submatvec_, out, hessian, inp)
}

/***********************************************************************
 *                          MATVEC BACKWARD                           *
 ***********************************************************************/

namespace {
template <int C, typename scalar_t>
inline void _sym_matvec_backward(DLTensor & out, const DLTensor & grd, const DLTensor & inp)
{
    auto ao = tny::from_dlpack<      scalar_t>(&out);
    auto ag = tny::from_dlpack<const scalar_t>(&grd);
    auto ai = tny::from_dlpack<const scalar_t>(&inp);
    posdef::sym_matvec_backward<C, reduce_t>(ao, ag, ai);
}
}

void sym_matvec_backward(
          DLTensor & out,      // (*batch, C*(C+1)/2)
    const DLTensor & grd,      // (*batch, C)
    const DLTensor & inp,      // (*batch, C)
          int        /* stream <unused> */
)
{
    const int32_t nbatch   = grd.ndim - 1;
    const int64_t nchannel = grd.shape[grd.ndim-1];
    const auto    code     = static_cast<DLDataTypeCode>(grd.dtype.code);
    const auto    bits     = grd.dtype.bits;
    CHECK_NO_LANES  (grd)
    CHECK_SAME_DTYPE(grd, out)
    CHECK_SAME_DTYPE(grd, inp)
    CHECK_SAME      (inp.shape[inp.ndim-1], nchannel, "Input and grad channel counts differ")
    CHECK_SAME      (out.shape[out.ndim-1]*2, nchannel*(nchannel+1), "Matrix is not compatible with the channel count")
    CHECK_RANK      (grd, nbatch+1)
    CHECK_RANK      (out, nbatch+1)
    CHECK_RANK      (inp, nbatch+1)
    CHECK_SAME_BATCH(grd, out, nbatch)
    CHECK_SAME_BATCH(grd, inp, nbatch)

    DISPATCH_SYM_C(_sym_matvec_backward, out, grd, inp)
}

/***********************************************************************
 *                               SOLVE                                *
 ***********************************************************************/

// The optional weight stops HERE, at the DLPack boundary: the `has_wgt` test
// picks between the impl's two `solve` overloads instead of handing it a carrier
// over a null pointer, which would smuggle the pointer-era sentinel straight
// through the tensor boundary (and cost a runtime branch per voxel).
namespace {
template <posdef::type Ty, typename scalar_t>
inline void _solve(DLTensor & out, const DLTensor & hes,
                   const DLTensor & inp, const DLTensor & wgt)
{
    auto ao = tny::from_dlpack<      scalar_t>(&out);
    auto ai = tny::from_dlpack<const scalar_t>(&inp);
    auto ah = tny::from_dlpack<const scalar_t>(&hes);
    if (wgt.data != nullptr)
        posdef::solve<Ty, reduce_t>(ao, ai, ah, tny::from_dlpack<const scalar_t>(&wgt));
    else
        posdef::solve<Ty, reduce_t>(ao, ai, ah);
}
}

void sym_solve(
          DLTensor & out,
    const DLTensor & hessian,
    const DLTensor & inp,
    const DLTensor & weight,
          int        /* stream <unused> */
)
{
    const bool has_wgt = (weight.data != nullptr);
    const int32_t nbatch   = out.ndim - 1;
    const int64_t nchannel = out.shape[out.ndim-1];
    const int64_t CC       = hessian.shape[hessian.ndim-1];
    const auto    code     = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto    bits     = out.dtype.bits;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, hessian)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME      (inp.shape[inp.ndim-1], nchannel, "Input and output channel counts differ")
    CHECK_RANK      (out,     nbatch+1)
    CHECK_RANK      (hessian, nbatch+1)
    CHECK_RANK      (inp,     nbatch+1)
    CHECK_SAME_BATCH(out, inp,     nbatch)
    CHECK_SAME_BATCH(out, hessian, nbatch)
    if (has_wgt) { CHECK_SAME_DTYPE(out, weight) CHECK_RANK(weight, nbatch+1) CHECK_SAME_BATCH(out, weight, nbatch) }
    const auto mtype = posdef::guess_type<int64_t>(nchannel, CC);

    DISPATCH_TYPE_DYN(_solve, out, hessian, inp, weight)
}

namespace {
template <posdef::type Ty, typename scalar_t>
inline void _solve_(DLTensor & inp_out, const DLTensor & hes, const DLTensor & wgt)
{
    auto ao = tny::from_dlpack<      scalar_t>(&inp_out);
    auto ah = tny::from_dlpack<const scalar_t>(&hes);
    if (wgt.data != nullptr)
        posdef::solve_<Ty, reduce_t>(ao, ah, tny::from_dlpack<const scalar_t>(&wgt));
    else
        posdef::solve_<Ty, reduce_t>(ao, ah);
}
}

void sym_solve_(
          DLTensor & inp_out,
    const DLTensor & hessian,
    const DLTensor & weight,
          int        /* stream <unused> */
)
{
    const bool has_wgt = (weight.data != nullptr);
    const int32_t nbatch   = inp_out.ndim - 1;
    const int64_t nchannel = inp_out.shape[inp_out.ndim-1];
    const int64_t CC       = hessian.shape[hessian.ndim-1];
    const auto    code     = static_cast<DLDataTypeCode>(inp_out.dtype.code);
    const auto    bits     = inp_out.dtype.bits;
    CHECK_NO_LANES  (inp_out)
    CHECK_SAME_DTYPE(inp_out, hessian)
    CHECK_RANK      (inp_out, nbatch+1)
    CHECK_RANK      (hessian, nbatch+1)
    CHECK_SAME_BATCH(inp_out, hessian, nbatch)
    if (has_wgt) { CHECK_SAME_DTYPE(inp_out, weight) CHECK_RANK(weight, nbatch+1) CHECK_SAME_BATCH(inp_out, weight, nbatch) }
    const auto mtype = posdef::guess_type<int64_t>(nchannel, CC);

    DISPATCH_TYPE_DYN(_solve_, inp_out, hessian, weight)
}

/***********************************************************************
 *                              INVERT                                *
 ***********************************************************************/

namespace {
template <typename scalar_t>
inline void _sym_invert(DLTensor & out, const DLTensor & hes)
{
    auto ao = tny::from_dlpack<      scalar_t>(&out);
    auto ah = tny::from_dlpack<const scalar_t>(&hes);
    posdef::sym_invert<reduce_t>(ao, ah);
}
}

void sym_invert(
          DLTensor & out,      // (*batch, C*(C+1)/2)
    const DLTensor & hessian,  // (*batch, C*(C+1)/2)
          int        /* stream <unused> */
)
{
    const int32_t nbatch   = out.ndim - 1;
    const int64_t CC       = hessian.shape[hessian.ndim-1];
    // channels_from_packed VALIDATES CC (it throws when the packed length is not
    // a triangular number). The impl now derives its own channel count from the
    // carrier, so the result is unused here -- but the call, and its position in
    // the evaluation order, are part of this entry point's behaviour.
    const int64_t nchannel = channels_from_packed(CC);
    (void)nchannel;
    const auto    code     = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto    bits     = out.dtype.bits;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, hessian)
    CHECK_SAME      (out.shape[out.ndim-1], CC, "Output and matrix must share the compact layout")
    CHECK_RANK      (out,     nbatch+1)
    CHECK_RANK      (hessian, nbatch+1)
    CHECK_SAME_BATCH(out, hessian, nbatch)

    DISPATCH_SYM(_sym_invert, out, hessian)
}

namespace {
template <typename scalar_t>
inline void _sym_invert_(DLTensor & hes)
{
    auto ah = tny::from_dlpack<scalar_t>(&hes);
    posdef::sym_invert_<reduce_t>(ah);
}
}

void sym_invert_(
          DLTensor & hessian,
          int        /* stream <unused> */
)
{
    const int64_t CC       = hessian.shape[hessian.ndim-1];
    // Validates CC and throws on a bad packed length -- see sym_invert above.
    const int64_t nchannel = channels_from_packed(CC);
    (void)nchannel;
    const auto    code     = static_cast<DLDataTypeCode>(hessian.dtype.code);
    const auto    bits     = hessian.dtype.bits;
    CHECK_NO_LANES(hessian)

    DISPATCH_SYM(_sym_invert_, hessian)
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
