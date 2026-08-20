#include <stdexcept>
#include <cstdint>
#include <fastfields/api/cpu/restrict.h>
#include <fastfields/core/autocast.h>
#include <fastfields/core/dispatch.h>
#include <fastfields/core/dlpack.h>
#include <fastfields/core/cuda_switch.h>
#include <fastfields/impl/kernels/utils.h>
#include <fastfields/impl/cpu/restrict.h>

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)

/***********************************************************************
 *                            DISPATCH                                 *
 ***********************************************************************/

namespace {
template <int ndim, spline::type I, bound::type B, typename scalar_t, typename offset_t>
inline void _restriction(
          int64_t   nbatch     ,   // number of batch dimensions
          void    * out        ,   // (*batch, *outshape) coarse tensor (pre-zeroed)
    const void    * inp        ,   // (*batch, *inshape)  fine tensor
          double    shift      ,   // anchor shift
    const double  * scale      ,   // [ndim] scaling
    const int64_t * size_out   ,   // [nall] output shape
    const int64_t * size_inp   ,   // [nall] input shape
    const int64_t * stride_out ,   // [nall] output strides
    const int64_t * stride_inp )   // [nall] input strides
{
    const int64_t nall = nbatch + ndim;   // == out.ndim == inp.ndim
    const IndexArray<offset_t> _size_out   (size_out, nall);
    const IndexArray<offset_t> _size_inp   (size_inp, nall);
    const IndexArray<offset_t> _stride_out (stride_out, nall);
    const IndexArray<offset_t> _stride_inp (stride_inp, nall);
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

    restrict::loop<ndim, scalar_t, offset_t, double, I, B>(
        static_cast<offset_t>(nbatch), _out, _inp, shift, _scale,
        _size_out, _size_inp, _stride_out, _stride_inp);

}
} // anonymous namespace

#define RT_DTYPE(D, I, B, args...)                                      \
    switch (code) {                                                     \
        case kDLFloat: switch (inp.dtype.bits) {                        \
            case 32: return (                                           \
                use_32bits ? _restriction<D,I,B,float, off32_t>(args)   \
                           : _restriction<D,I,B,float, int64_t>(args)); \
            case 64: return (                                           \
                use_32bits ? _restriction<D,I,B,double,off32_t>(args)   \
                           : _restriction<D,I,B,double,int64_t>(args)); \
            default: break;                                             \
        };                                                              \
        default: break;                                                 \
    }

#define RT_BOUND(D, I, args...)                                         \
    switch (bnd) {                                                      \
        case bound_t::Zero:      RT_DTYPE(D, I, bound_t::Zero,      args); break; \
        case bound_t::Replicate: RT_DTYPE(D, I, bound_t::Replicate, args); break; \
        case bound_t::DCT1:      RT_DTYPE(D, I, bound_t::DCT1,      args); break; \
        case bound_t::DCT2:      RT_DTYPE(D, I, bound_t::DCT2,      args); break; \
        case bound_t::DST1:      RT_DTYPE(D, I, bound_t::DST1,      args); break; \
        case bound_t::DST2:      RT_DTYPE(D, I, bound_t::DST2,      args); break; \
        case bound_t::DFT:       RT_DTYPE(D, I, bound_t::DFT,       args); break; \
        case bound_t::NoCheck:   RT_DTYPE(D, I, bound_t::NoCheck,   args); break; \
        default: throw std::invalid_argument("Unsupported boundary condition"); \
    }

// Fast-test builds (`-DFF_TEST_SPARSE`) instantiate a covering subset of the
// order x bound matrix: all bounds for Linear + Cubic, DCT2 only for the rest.
// The library / CI build keeps the full matrix (also the compile gate).
#ifdef FF_TEST_SPARSE
#define RT_BOUND_SPARSE(D, I, args...)                                  \
    switch (bnd) {                                                      \
        case bound_t::DCT2: RT_DTYPE(D, I, bound_t::DCT2, args); break; \
        default: throw std::invalid_argument(                          \
            "bound not instantiated in FF_TEST_SPARSE build");         \
    }
#define RT_ORDER(D, args...)                                            \
    switch (spl) {                                                      \
        case spline_t::Nearest:      RT_BOUND_SPARSE(D, spline_t::Nearest,      args); break; \
        case spline_t::Linear:       RT_BOUND(D, spline_t::Linear,       args); break; \
        case spline_t::Quadratic:    RT_BOUND_SPARSE(D, spline_t::Quadratic,    args); break; \
        case spline_t::Cubic:        RT_BOUND(D, spline_t::Cubic,        args); break; \
        case spline_t::FourthOrder:  RT_BOUND_SPARSE(D, spline_t::FourthOrder,  args); break; \
        case spline_t::FifthOrder:   RT_BOUND_SPARSE(D, spline_t::FifthOrder,   args); break; \
        case spline_t::SixthOrder:   RT_BOUND_SPARSE(D, spline_t::SixthOrder,   args); break; \
        case spline_t::SeventhOrder: RT_BOUND_SPARSE(D, spline_t::SeventhOrder, args); break; \
        default: throw std::invalid_argument("Unsupported spline order");   \
    }
#else
#define RT_ORDER(D, args...)                                            \
    switch (spl) {                                                      \
        case spline_t::Nearest:      RT_BOUND(D, spline_t::Nearest,      args); break; \
        case spline_t::Linear:       RT_BOUND(D, spline_t::Linear,       args); break; \
        case spline_t::Quadratic:    RT_BOUND(D, spline_t::Quadratic,    args); break; \
        case spline_t::Cubic:        RT_BOUND(D, spline_t::Cubic,        args); break; \
        case spline_t::FourthOrder:  RT_BOUND(D, spline_t::FourthOrder,  args); break; \
        case spline_t::FifthOrder:   RT_BOUND(D, spline_t::FifthOrder,   args); break; \
        case spline_t::SixthOrder:   RT_BOUND(D, spline_t::SixthOrder,   args); break; \
        case spline_t::SeventhOrder: RT_BOUND(D, spline_t::SeventhOrder, args); break; \
        default: throw std::invalid_argument("Unsupported spline order");   \
    }
#endif

#define DISPATCH_RESTRICT(args...)                                        \
{                                                                         \
    const bool use_32bits = FF_CANUSE32BITS(out) && FF_CANUSE32BITS(inp); \
    const auto code = static_cast<DLDataTypeCode>(inp.dtype.code);        \
    const spline_t spl = static_cast<spline_t>(spline);                   \
    const bound_t  bnd = static_cast<bound_t >(bound);                    \
    switch (ndim) {                                                       \
        case 1: RT_ORDER(1, args); break;                                 \
        case 2: RT_ORDER(2, args); break;                                 \
        case 3: RT_ORDER(3, args); break;                                 \
        default: throw std::invalid_argument(                             \
            "Only 1D, 2D and 3D restrict are supported");                 \
    };                                                                    \
    /* Reached only when a valid dim/spline/bound had an unsupported      \
       dtype: the RT_DTYPE switch fell through without returning. */      \
    throw std::invalid_argument(                                          \
        "Unsupported data type for restriction (only float32/float64)");  \
}

void restriction(
          DLTensor & out_   ,
    const DLTensor & inp_   ,
          int8_t     spline ,
          int8_t     bound  ,
          double     shift  ,
    const double   * scale  ,
          int        ndim   ,
          intptr_t   /* stream <unused> */
)
{
    // Normalise a NULL strides field (compact row-major) so the dispatch and
    // impl loops below can dereference strides unconditionally.
    ContiguousStrides _out(out_), _inp(inp_);
    DLTensor & out = _out.t;
    DLTensor & inp = _inp.t;

    const int32_t nbatch = out.ndim - ndim;
    FF_CHECK_NO_LANES  (out)
    FF_CHECK_SAME_DTYPE(out, inp)
    FF_CHECK_SAME      (out.ndim, inp.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    FF_CHECK_SAME_BATCH(out, inp, nbatch)

    DISPATCH_RESTRICT(
        static_cast<int64_t>(nbatch),
        FF_VOIDPTR(out),
        FF_VOIDPTR(inp),
        shift,
        scale,
        out.shape,
        inp.shape,
        out.strides,
        inp.strides
    )
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
