#include <stdexcept>
#include <cstdint>
#include <fastfields/api/cpu/invfield.h>
#include <fastfields/core/autocast.h>
#include <fastfields/core/dispatch.h>
#include <fastfields/core/dlpack.h>
#include <fastfields/core/cuda_switch.h>
#include <fastfields/impl/kernels/utils.h>
#include <fastfields/impl/cpu/invfield.h>

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)

/***********************************************************************
 *                            DISPATCH                                 *
 ***********************************************************************/

namespace {
template <int ndim, typename scalar_t, typename offset_t>
inline void _invert_field(
          int64_t   nbatch       ,   // number of batch dimensions
          void    * out          ,   // (*batch, *spatial, ndim)
          void    * owner        ,   // (*batch, *spatial) int64
    const void    * disp         ,   // (*batch, *spatial, ndim)
    const int64_t * size         ,   // [nbatch + ndim] shared shape
    const int64_t * stride_out   ,   // [nbatch + ndim + 1]
    const int64_t * stride_own   ,   // [nbatch + ndim]
    const int64_t * stride_disp )    // [nbatch + ndim + 1]
{
    const int64_t nall = nbatch + ndim;
    const IndexArray<offset_t> _size        (size,        nall);
    const IndexArray<offset_t> _stride_out  (stride_out,  nall + 1);
    const IndexArray<offset_t> _stride_own  (stride_own,  nall);
    const IndexArray<offset_t> _stride_disp (stride_disp, nall + 1);

    invfield::loop<ndim, scalar_t, offset_t, double>(
        static_cast<offset_t>(nbatch),
        static_cast<      scalar_t *>(out),
        static_cast<      int64_t  *>(owner),
        static_cast<const scalar_t *>(disp),
        _size, _stride_out, _stride_own, _stride_disp);
}
} // anonymous namespace

// The op has no bound/spline/order axis at all -- a simplex has no boundary
// condition and no interpolation order -- so this is the smallest dispatch in
// the tree: dim x dtype x offset, and nothing else.
#define IF_DTYPE(D, args...)                                             \
    switch (code) {                                                      \
        case kDLFloat: switch (disp.dtype.bits) {                        \
            case 32: return (                                            \
                use_32bits ? _invert_field<D,float, off32_t>(args)       \
                           : _invert_field<D,float, int64_t>(args));     \
            case 64: return (                                            \
                use_32bits ? _invert_field<D,double,off32_t>(args)       \
                           : _invert_field<D,double,int64_t>(args));     \
            default: break;                                              \
        };                                                               \
        default: break;                                                  \
    }

#define DISPATCH_INVFIELD(args...)                                        \
{                                                                         \
    const bool use_32bits = FF_CANUSE32BITS(out)                          \
                         && FF_CANUSE32BITS(own)                          \
                         && FF_CANUSE32BITS(disp);                        \
    const auto code = static_cast<DLDataTypeCode>(disp.dtype.code);       \
    switch (ndim) {                                                       \
        case 2: IF_DTYPE(2, args); break;                                 \
        case 3: IF_DTYPE(3, args); break;                                 \
        default: throw std::invalid_argument(                             \
            "Only 2D and 3D field inversion is supported");               \
    };                                                                    \
    /* Reached only when a valid dim had an unsupported dtype: the        \
       IF_DTYPE switch fell through without returning. */                 \
    throw std::invalid_argument(                                          \
        "Unsupported data type for invert_field (only float32/float64)"); \
}

void invert_field(
          DLTensor & out_   ,
          DLTensor & own_   ,
    const DLTensor & disp_  ,
          int        ndim   ,
          intptr_t   /* stream <unused> */
)
{
    // Normalise a NULL strides field (compact row-major) so the dispatch and
    // impl loops below can dereference strides unconditionally.
    ContiguousStrides _out(out_), _own(own_), _disp(disp_);
    DLTensor & out  = _out.t;
    DLTensor & own  = _own.t;
    DLTensor & disp = _disp.t;

    FF_CHECK_NO_LANES  (out)
    FF_CHECK_NO_LANES  (own)
    FF_CHECK_SAME_DTYPE(out, disp)
    FF_CHECK_SAME_SHAPE(out, disp)

    if (ndim != 2 && ndim != 3)
        throw std::invalid_argument(
            "Only 2D and 3D field inversion is supported");

    // (*batch, *spatial, ndim) for the two fields, (*batch, *spatial) for the
    // owner map -- so the owner map has exactly one dimension fewer.
    const int32_t nbatch = own.ndim - ndim;
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    FF_CHECK_SAME(out.ndim, own.ndim + 1,
        "The owner map must have one dimension fewer than the field")
    FF_CHECK_SAME_SHAPE_N(out, own, own.ndim)
    FF_CHECK_SAME(out.shape[out.ndim - 1], static_cast<int64_t>(ndim),
        "The last dimension of the field must hold its ndim components")

    if (own.dtype.code != kDLInt || own.dtype.bits != 64)
        throw std::invalid_argument("The owner map must be int64");

    // A cell needs two nodes along every spatial axis; with fewer there is no
    // mesh at all, and the cell-grid shape `spatial[d] - 1` would be empty or
    // negative.
    for (int d = 0; d < ndim; ++d)
        if (own.shape[nbatch + d] < 2)
            throw std::invalid_argument(
                "Every spatial dimension must have at least 2 elements");

    DISPATCH_INVFIELD(
        static_cast<int64_t>(nbatch),
        FF_VOIDPTR(out),
        FF_VOIDPTR(own),
        FF_CVOIDPTR(disp),
        own.shape,
        out.strides,
        own.strides,
        disp.strides
    )
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
