#include <stdexcept>
#include "resize.h"
#include "autocast.h"
#include "dlpack.h"
#include "impl/kernels/cuda_switch.h"
#include "impl/kernels/utils.h"
#include "impl/resize.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

#define VOIDPTR(x)      (static_cast<void*>(static_cast<char*>(x.data) + x.byte_offset))
#define CANUSE32BITS(x) (canUse32BitIndexMath(x.ndim, x.shape, x.strides))

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

namespace {
template <int ndim, int O, bound_t B, typename scalar_t, typename offset_t>
inline void _resample(
          int64_t   nbatch     ,   // number of batch dimensions
          bound_t   bnd        ,   // runtime bound (consumed only on the B == Dynamic route)
          void    * out        ,   // (*batch, *outshape) tensor
    const void    * inp        ,   // (*batch, *inshape)  tensor
          double    shift      ,   // anchor shift
    const double  * scale      ,   // [ndim] scaling
    const int64_t * size_out   ,   // [nall] output shape
    const int64_t * size_inp   ,   // [nall] input shape
    const int64_t * stride_out ,   // [nall] output strides
    const int64_t * stride_inp )   // [nall] input strides
{
    const int64_t nall = nbatch + ndim;   // == out.ndim == inp.ndim
    const offset_t * _size_out   = copy_if_needed<offset_t *>(size_out,   nall);
    const offset_t * _size_inp   = copy_if_needed<offset_t *>(size_inp,   nall);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nall);
    const offset_t * _stride_inp = copy_if_needed<offset_t *>(stride_inp, nall);
          scalar_t * _out        = static_cast<      scalar_t *>(out);
    const scalar_t * _inp        = static_cast<const scalar_t *>(inp);

    // When no scaling is provided, default to the input/output size ratio
    // (identity when the shapes match). The kernel dereferences scale[d]
    // unconditionally, so a null pointer would otherwise segfault.
    const double * _scale = scale;
    double default_scale[ndim];
    if (!_scale) {
        for (int d = 0; d < ndim; ++d)
            default_scale[d] = static_cast<double>(size_inp[nbatch + d])
                             / static_cast<double>(size_out[nbatch + d]);
        _scale = default_scale;
    }

    resize::loop<ndim, O, B, double, scalar_t, offset_t>(
        static_cast<offset_t>(nbatch), _out, _inp, shift, _scale,
        _size_out, _size_inp, _stride_out, _stride_inp, bnd);

    free_if_needed<int64_t *>(_size_out);
    free_if_needed<int64_t *>(_size_inp);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_inp);
}
} // anonymous namespace

// dtype x offset, given static dim D, order O and compile-time bound B.
#define RS_DTYPE(D, O, B, args...)                                      \
    switch (bits) {                                                     \
        case 32: return (use_32bits ? _resample<D,O,B,float, int32_t>(args) \
                                    : _resample<D,O,B,float, int64_t>(args)); \
        case 64: return (use_32bits ? _resample<D,O,B,double,int32_t>(args) \
                                    : _resample<D,O,B,double,int64_t>(args)); \
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
    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp); \
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
          DLTensor & out_   ,
    const DLTensor & inp_   ,
          int8_t     spline ,
          int8_t     bound  ,
          double     shift  ,
    const double   * scale  ,
          int        ndim   ,
          int        /* stream <unused> */
)
{
    // Normalise a NULL strides field (compact row-major) so the dispatch and
    // impl loops below can dereference strides unconditionally.
    ContiguousStrides _out(out_), _inp(inp_);
    DLTensor & out = _out.t;
    DLTensor & inp = _inp.t;

    const int32_t nbatch = out.ndim - ndim;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME      (out.ndim, inp.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME_BATCH(out, inp, nbatch)

    DISPATCH_RESIZE(
        static_cast<int64_t>(nbatch),
        static_cast<bound_t>(bound),
        VOIDPTR(out),
        VOIDPTR(inp),
        shift,
        scale,
        out.shape,
        inp.shape,
        out.strides,
        inp.strides
    )
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
