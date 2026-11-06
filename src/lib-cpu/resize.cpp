#include <stdexcept>
#include <cstdint>
#include <fastfields/api/cpu/resize.h>
#include <fastfields/core/autocast.h>
#include <fastfields/core/dispatch.h>
#include <fastfields/core/dlpack.h>
#include <fastfields/core/cuda_switch.h>
#include <fastfields/impl/kernels/utils.h>
#include <fastfields/impl/cpu/resize.h>

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)

/***********************************************************************
 *                            DISPATCH                                 *
 ***********************************************************************/

namespace {
template <int ndim, spline::type I, bound::type B, typename scalar_t, typename offset_t>
inline void _resample(
    const bound::BoundVec  & bvec,       // runtime bound (Dynamic axes)
    const spline::SplineVec & svec,       // runtime order (Dynamic axes)
          int64_t   nbatch     ,   // number of batch dimensions
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

    resize::loop<ndim, scalar_t, offset_t, double, I, B, I, B, I, B,
                 FF_SPLINE_DYNAMIC_MASK & resize::fast_all>(
        static_cast<offset_t>(nbatch), _out, _inp, shift, _scale,
        _size_out, _size_inp, _stride_out, _stride_inp, bvec, svec);
}
} // anonymous namespace

#define RS_DTYPE(D, I, B, args...)                                      \
    switch (code) {                                                     \
        case kDLFloat: switch (inp.dtype.bits) {                        \
            case 32: return (                                           \
                use_32bits ? _resample<D,I,B,float, off32_t>(bvec, svec, args)      \
                           : _resample<D,I,B,float, int64_t>(bvec, svec, args));    \
            case 64: return (                                           \
                use_32bits ? _resample<D,I,B,double,off32_t>(bvec, svec, args)      \
                           : _resample<D,I,B,double,int64_t>(bvec, svec, args));    \
            default: break;                                             \
        };                                                              \
        default: break;                                                 \
    }

// The template arguments are FF_BOUND_<NAME> / FF_SPLINE_<NAME> (kernels/
// bounds.h, kernels/spline.h): the condition/order itself when it is
// statically compiled, `Dynamic` otherwise, per the interpolation
// family's PP_BOUNDFLAGS / PP_SPLINEFLAGS (see this directory's Makefile).
// The switch labels stay exhaustive on the *runtime* value either way;
// `bvec` / `svec` carry that value to the Dynamic instantiations. Every
// combination is therefore always functional -- there is no sparse
// variant that throws (the old `-DFF_TEST_SPARSE` branch is gone, as
// pushpull's went before it) -- and results are bit-identical at every
// policy: a Dynamic axis runs the same arithmetic its static
// instantiation would (see MultiscaleDynamic in kernels/resize.h).
#define RS_BOUND(D, I, args...)                                         \
    switch (bnd) {                                                      \
        case bound_t::Zero:      RS_DTYPE(D, I, FF_BOUND_ZERO,      args); break; \
        case bound_t::Replicate: RS_DTYPE(D, I, FF_BOUND_REPLICATE, args); break; \
        case bound_t::DCT1:      RS_DTYPE(D, I, FF_BOUND_DCT1,      args); break; \
        case bound_t::DCT2:      RS_DTYPE(D, I, FF_BOUND_DCT2,      args); break; \
        case bound_t::DST1:      RS_DTYPE(D, I, FF_BOUND_DST1,      args); break; \
        case bound_t::DST2:      RS_DTYPE(D, I, FF_BOUND_DST2,      args); break; \
        case bound_t::DFT:       RS_DTYPE(D, I, FF_BOUND_DFT,       args); break; \
        case bound_t::NoCheck:   RS_DTYPE(D, I, FF_BOUND_NOCHECK,   args); break; \
        default: throw std::invalid_argument("Unsupported boundary condition"); \
    }

#define RS_ORDER(D, args...)                                            \
    switch (spl) {                                                      \
        case spline_t::Nearest:      RS_BOUND(D, FF_SPLINE_NEAREST,      args); break; \
        case spline_t::Linear:       RS_BOUND(D, FF_SPLINE_LINEAR,       args); break; \
        case spline_t::Quadratic:    RS_BOUND(D, FF_SPLINE_QUADRATIC,    args); break; \
        case spline_t::Cubic:        RS_BOUND(D, FF_SPLINE_CUBIC,        args); break; \
        case spline_t::FourthOrder:  RS_BOUND(D, FF_SPLINE_FOURTHORDER,  args); break; \
        case spline_t::FifthOrder:   RS_BOUND(D, FF_SPLINE_FIFTHORDER,   args); break; \
        case spline_t::SixthOrder:   RS_BOUND(D, FF_SPLINE_SIXTHORDER,   args); break; \
        case spline_t::SeventhOrder: RS_BOUND(D, FF_SPLINE_SEVENTHORDER, args); break; \
        default: throw std::invalid_argument("Unsupported spline order");   \
    }

#define DISPATCH_RESIZE(args...)                                          \
{                                                                         \
    const bool use_32bits = FF_CANUSE32BITS(out) && FF_CANUSE32BITS(inp); \
    const auto code = static_cast<DLDataTypeCode>(inp.dtype.code);        \
    const spline_t spl = static_cast<spline_t>(spline);                   \
    const bound_t  bnd = static_cast<bound_t >(bound);                    \
    const bound::BoundVec   bvec(bnd);                                    \
    const spline::SplineVec svec(spl);                                    \
    switch (ndim) {                                                       \
        case 1: RS_ORDER(1, args); break;                                 \
        case 2: RS_ORDER(2, args); break;                                 \
        case 3: RS_ORDER(3, args); break;                                 \
        default: throw std::invalid_argument(                             \
            "Only 1D, 2D and 3D resize are supported");                   \
    };                                                                    \
    /* Reached only when a valid dim/spline/bound had an unsupported      \
       dtype: the RS_DTYPE switch fell through without returning. */      \
    throw std::invalid_argument(                                          \
        "Unsupported data type for resample (only float32/float64)");     \
}

void resample(
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

    DISPATCH_RESIZE(
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
