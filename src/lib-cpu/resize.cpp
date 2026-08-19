#include <stdexcept>
#include "resize.h"
#include "dlpack.h"
// R7 (TEENY-MIGRATION.md sec. 9): fastfields vendors DLPack v1.2, teeny v1.1,
// and both use the guard DLPACK_DLPACK_H_ -- so whichever is seen first wins
// for the whole TU. Our "dlpack.h" is included ABOVE on purpose; keep it there.
#include <teeny/dlpack.h>
#include "impl/kernels/cuda_switch.h"
#include "impl/kernels/utils.h"
#include "impl/resize.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

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

#define CHECK_SAME_DTYPE(X, Y)                                          \
    if (                                                                \
        (X.dtype.code  != Y.dtype.code) ||                              \
        (X.dtype.bits  != Y.dtype.bits) ||                              \
        (X.dtype.lanes != Y.dtype.lanes)                                \
    )                                                                   \
        throw std::invalid_argument(                                    \
            "Tensors do not have the same data type"                    \
        );

#define CHECK_SAME_BATCH(X, Y, D)                                       \
    for (int32_t d=0; d < D; ++d)                                       \
        if (X.shape[d] != Y.shape[d])                                   \
            throw std::invalid_argument(                                \
                "Tensors do not have the same batch shape"              \
            );

/***********************************************************************
 *                            DISPATCH                                 *
 ***********************************************************************/

// The dtype arm's IMPORT POINT. `tny::from_dlpack` builds each teeny carrier
// once, straight off the bare DLTensor, and does by construction all three
// things this path used to do by hand: it folds `byte_offset` into the data
// pointer (was VOIDPTR), expands a NULL `strides` field to row-major (was
// ContiguousStrides), and copies that tensor's own shape/stride metadata into
// the carrier -- so the impl below needs no pointers, no nbatch, and none of
// the four shared size[]/stride[] arrays (R3: one carrier per tensor).
//
// The input is imported as a `const scalar_t` carrier (R4), so the read-only
// operand is read-only in the type system, with no const_cast.
//
// D1/R5: the int32 offset arm is gone. The CPU narrowing was measured a wash on
// a 64-bit ALU (distance-slice review), so this is now the single int64
// instantiation; `offset_t` is whatever the carrier carries (int64 off DLPack).
namespace {
template <int ndim, int O, bound_t B, typename scalar_t>
inline void _resample(
          bound_t    bnd  ,   // runtime bound (consumed only on the B == Dynamic route)
          DLTensor & out  ,   // (*batch, *outshape) tensor
    const DLTensor & inp  ,   // (*batch, *inshape)  tensor
          double     shift,   // anchor shift
    const double   * scale)   // [ndim] scaling
{
    auto ao = tny::from_dlpack<      scalar_t>(&out);
    auto ai = tny::from_dlpack<const scalar_t>(&inp);

    // When no scaling is provided, default to the input/output size ratio
    // (identity when the shapes match). The kernel dereferences scale[d]
    // unconditionally, so a null pointer would otherwise segfault.
    const double * _scale = scale;
    double default_scale[ndim];
    if (!_scale) {
        const int64_t nbatch = out.ndim - ndim;
        for (int d = 0; d < ndim; ++d)
            default_scale[d] = static_cast<double>(inp.shape[nbatch + d])
                             / static_cast<double>(out.shape[nbatch + d]);
        _scale = default_scale;
    }

    resize::loop<ndim, O, B>(ao, ai, shift, _scale, bnd);
}
} // anonymous namespace

// dtype, given static dim D, order O and compile-time bound B. (D1/R5: the
// offset-width leg of this dispatch is gone -- int64 only on the CPU.)
#define RS_DTYPE(D, O, B, args...)                                      \
    switch (bits) {                                                     \
        case 32: return _resample<D,O,B,float >(args);                  \
        case 64: return _resample<D,O,B,double>(args);                  \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only float32 / float64 are supported");

// bound -> compile-time B, applying the static/Dynamic split (cpu-lib#22):
// DFT/DCT2/DST2/Zero/NoCheck compile statically; DCT1/DST1/Replicate route to
// the single Dynamic instantiation (the runtime `bnd` selects inside the kernel).
#define RS_BOUND(D, O, args...)                                         \
    switch (bnd) {                                                      \
        case bound_t::DFT:       RS_DTYPE(D,O,bound_t::DFT,     args); break; \
        case bound_t::DCT2:      RS_DTYPE(D,O,bound_t::DCT2,    args); break; \
        case bound_t::DST2:      RS_DTYPE(D,O,bound_t::DST2,    args); break; \
        case bound_t::Zero:      RS_DTYPE(D,O,bound_t::Zero,    args); break; \
        case bound_t::NoCheck:   RS_DTYPE(D,O,bound_t::NoCheck, args); break; \
        case bound_t::DCT1:                                             \
        case bound_t::DST1:                                             \
        case bound_t::Replicate: RS_DTYPE(D,O,bound_t::Dynamic, args); break; \
        default: throw std::invalid_argument("Unsupported boundary condition"); \
    }

// Fast-test builds (`-DFF_TEST_SPARSE`) instantiate a covering subset of the
// order x bound matrix: all bounds for Linear + Cubic, DCT2 only for the rest.
// The library / CI build keeps the full matrix (also the compile gate).
#ifdef FF_TEST_SPARSE
#define RS_BOUND_SPARSE(D, O, args...)                                  \
    switch (bnd) {                                                      \
        case bound_t::DCT2: RS_DTYPE(D, O, bound_t::DCT2, args); break; \
        default: throw std::invalid_argument(                          \
            "bound not instantiated in FF_TEST_SPARSE build");         \
    }
#define RS_ORDER(D, args...)                                            \
    switch (spl) {                                                      \
        case spline_t::Nearest:      RS_BOUND_SPARSE(D, 0, args); break; \
        case spline_t::Linear:       RS_BOUND       (D, 1, args); break; \
        case spline_t::Quadratic:    RS_BOUND_SPARSE(D, 2, args); break; \
        case spline_t::Cubic:        RS_BOUND       (D, 3, args); break; \
        case spline_t::FourthOrder:  RS_BOUND_SPARSE(D, 4, args); break; \
        case spline_t::FifthOrder:   RS_BOUND_SPARSE(D, 5, args); break; \
        case spline_t::SixthOrder:   RS_BOUND_SPARSE(D, 6, args); break; \
        case spline_t::SeventhOrder: RS_BOUND_SPARSE(D, 7, args); break; \
        default: throw std::invalid_argument("Unsupported spline order");   \
    }
#else
#define RS_ORDER(D, args...)                                            \
    switch (spl) {                                                      \
        case spline_t::Nearest:      RS_BOUND(D, 0, args); break;       \
        case spline_t::Linear:       RS_BOUND(D, 1, args); break;       \
        case spline_t::Quadratic:    RS_BOUND(D, 2, args); break;       \
        case spline_t::Cubic:        RS_BOUND(D, 3, args); break;       \
        case spline_t::FourthOrder:  RS_BOUND(D, 4, args); break;       \
        case spline_t::FifthOrder:   RS_BOUND(D, 5, args); break;       \
        case spline_t::SixthOrder:   RS_BOUND(D, 6, args); break;       \
        case spline_t::SeventhOrder: RS_BOUND(D, 7, args); break;       \
        default: throw std::invalid_argument("Unsupported spline order");   \
    }
#endif

#define DISPATCH_RESIZE(args...)                                        \
{                                                                       \
    const auto     code = static_cast<DLDataTypeCode>(inp.dtype.code);  \
    const auto     bits = inp.dtype.bits;                               \
    const spline_t spl  = static_cast<spline_t>(spline);               \
    const bound_t  bnd  = static_cast<bound_t >(bound);                \
    if (code != kDLFloat)                                              \
        throw std::invalid_argument(                                  \
            "Unsupported data type for resample (only float32/float64)"); \
    switch (ndim) {                                                    \
        case 1: RS_ORDER(1, args); break;                             \
        case 2: RS_ORDER(2, args); break;                             \
        case 3: RS_ORDER(3, args); break;                             \
        default: throw std::invalid_argument(                         \
            "Only 1D, 2D and 3D resize are supported");                \
    };                                                                 \
}

void resample(
          DLTensor & out    ,
    const DLTensor & inp    ,
          int8_t     spline ,
          int8_t     bound  ,
          double     shift  ,
    const double   * scale  ,
          int        ndim   ,
          int        /* stream <unused> */
)
{
    // A NULL strides field (DLPack's compact row-major shorthand) and a non-zero
    // byte_offset are now normalised by `tny::from_dlpack` inside the dtype arm,
    // so there is nothing to pre-normalise here. The validation below is
    // unchanged, in both content and ORDER (behavioural ABI) -- and it reads
    // only `.ndim`/`.shape`/`.dtype`, which normalisation never affected.
    const int32_t nbatch = out.ndim - ndim;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME      (out.ndim, inp.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME_BATCH(out, inp, nbatch)

    DISPATCH_RESIZE(
        static_cast<bound_t>(bound),
        out,
        inp,
        shift,
        scale
    )
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
