#include <stdexcept>
#include <cstdint>
#include <utility>
#include "reg_flow.h"
#include "reg_dispatch.h"
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

// length of the shape/stride arrays: (*batch, *spatial, C) == out.ndim
template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _flow_matvec(
          int64_t   nbatch     ,
          void    * out        ,
    const void    * inp        ,
    const double  * voxel_size ,
          double    absolute   ,
          double    membrane   ,
          double    bending    ,
    const int64_t * size       ,
    const int64_t * stride_out ,
    const int64_t * stride_inp ,
          cudaStream_t stream  )
{
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nall1);
    const offset_t * _stride_inp = copy_if_needed<offset_t *>(stride_inp, nall1);
          scalar_t * _out = static_cast<      scalar_t *>(out);
    const scalar_t * _inp = static_cast<const scalar_t *>(inp);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    if (bending != 0.0)
        reg_flow::matvec_bending<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute, membrane, bending, stream);
    else if (membrane != 0.0)
        reg_flow::matvec_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute, membrane, stream);
    else
        reg_flow::matvec_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute, stream);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_inp);
}

template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _flow_diag(
          int64_t   nbatch     ,
          void    * out        ,
    const double  * voxel_size ,
          double    absolute   ,
          double    membrane   ,
          double    bending    ,
    const int64_t * size       ,
    const int64_t * stride_out ,
          cudaStream_t stream  )
{
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nall1);
          scalar_t * _out = static_cast<scalar_t *>(out);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    if (bending != 0.0)
        reg_flow::diag_bending<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane, bending, stream);
    else if (membrane != 0.0)
        reg_flow::diag_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane, stream);
    else
        reg_flow::diag_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, stream);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
}

// Dispatch adapters (one per exported entry point). Each names the templated
// worker the shared dispatch chain should land on; `reg_dispatch.h` owns
// everything else -- the axis-rank and boundary match, repeating that boundary
// across the axes, the dtype/offset-width leaf, and every rejection message.
struct matvec_op {
    template <int D, typename scalar_t, typename offset_t, bound::type... BOUND, typename... Args>
    static void run(Args &&... args)
    { _flow_matvec<D, scalar_t, offset_t, BOUND...>(std::forward<Args>(args)...); }
};

struct diag_op {
    template <int D, typename scalar_t, typename offset_t, bound::type... BOUND, typename... Args>
    static void run(Args &&... args)
    { _flow_diag<D, scalar_t, offset_t, BOUND...>(std::forward<Args>(args)...); }
};

} // anonymous namespace

/***********************************************************************
 *                            DISPATCH                                 *
 ***********************************************************************/

// The runtime (ndim, bound) -> compile-time step, the per-axis boundary pack,
// and the dtype x offset-width leaf all live in `reg_dispatch.h` (built on
// teeny's `dispatch_values`). The message an out-of-range `ndim` gets is the
// one thing that differs between the field and flow entry points, so it is
// passed in.
static constexpr const char * NDIM_MSG = "Only 1D, 2D and 3D flow are supported";

void flow_matvec(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _inp(inp_);
    DLTensor       & out = _out.t;
    const DLTensor & inp = _inp.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME      (out.ndim, inp.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME      (out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    CHECK_SAME_SHAPE(out, inp, out.ndim)

    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    const cudaStream_t cstream = _reg_stream(stream);

    reg_dispatch::dispatch_nd_bound<matvec_op>(
        ndim, bnd, NDIM_MSG,
        code, bits, use_32bits,
        static_cast<int64_t>(nbatch), VOIDPTR(out), CVOIDPTR(inp),
        voxel_size, absolute, membrane, bending,
        out.shape, out.strides, inp.strides, cstream);
}

void flow_diag(
          DLTensor & out_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_);
    DLTensor & out = _out.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES(out)
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME    (out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")

    const bool     use_32bits = CANUSE32BITS(out);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    const cudaStream_t cstream = _reg_stream(stream);

    reg_dispatch::dispatch_nd_bound<diag_op>(
        ndim, bnd, NDIM_MSG,
        code, bits, use_32bits,
        static_cast<int64_t>(nbatch), VOIDPTR(out),
        voxel_size, absolute, membrane, bending,
        out.shape, out.strides, cstream);
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
