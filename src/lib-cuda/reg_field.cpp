#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include "reg_field.h"
#include "posdef.h"
#include "autocast.h"
#include "dlpack.h"
#include "impl/kernels/cuda_switch.h"
#include "impl/kernels/bounds.h"
#include "impl/kernels/utils.h"
#include "impl/reg_field.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

#define VOIDPTR(x)      (static_cast<void*>(static_cast<char*>(x.data) + x.byte_offset))
#define CVOIDPTR(x)     (static_cast<const void*>(static_cast<const char*>(x.data) + x.byte_offset))
#define CANUSE32BITS(x) (canUse32BitIndexMath(x.ndim, x.shape, x.strides))

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

// build a length-nc reduce_t vector from a (possibly null) double array
static inline std::vector<reduce_t> as_weights(const double * w, int64_t nc)
{
    std::vector<reduce_t> v(static_cast<size_t>(nc), reduce_t(0));
    if (w) for (int64_t c = 0; c < nc; ++c) v[static_cast<size_t>(c)] = w[c];
    return v;
}

template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_matvec(
    const bound::BoundVec & bvec,
          int64_t   nbatch     ,
          int64_t   nc         ,
          void    * out        ,
    const void    * inp        ,
    const double  * voxel_size ,
    const double  * absolute   ,
    const double  * membrane   ,
    const double  * bending    ,
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

    std::vector<reduce_t> a = as_weights(absolute, nc);
    std::vector<reduce_t> m = as_weights(membrane, nc);
    std::vector<reduce_t> b = as_weights(bending,  nc);

    if (bending)
        reg_field::matvec_bending<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, a.data(), m.data(), b.data(), stream);
    else if (membrane)
        reg_field::matvec_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, a.data(), m.data(), stream);
    else
        reg_field::matvec_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, a.data(), stream);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_inp);
}

// Accumulate variant of _field_matvec: out += L(inp) (op='+') or out -= L(inp)
// (op='-'), instead of overwriting out. Mirrors the CPU `_field_matvec_acc`.
template <int ndim, char op, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_matvec_acc(
    const bound::BoundVec & bvec,
          int64_t   nbatch     ,
          int64_t   nc         ,
          void    * out        ,
    const void    * inp        ,
    const double  * voxel_size ,
    const double  * absolute   ,
    const double  * membrane   ,
    const double  * bending    ,
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

    std::vector<reduce_t> a = as_weights(absolute, nc);
    std::vector<reduce_t> m = as_weights(membrane, nc);
    std::vector<reduce_t> b = as_weights(bending,  nc);

    if (bending)
        reg_field::matvec_bending<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, a.data(), m.data(), b.data(), stream);
    else if (membrane)
        reg_field::matvec_membrane<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, a.data(), m.data(), stream);
    else
        reg_field::matvec_absolute<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, a.data(), stream);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_inp);
}

template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_diag(
    const bound::BoundVec & bvec,
          int64_t   nbatch     ,
          int64_t   nc         ,
          void    * out        ,
    const double  * voxel_size ,
    const double  * absolute   ,
    const double  * membrane   ,
    const double  * bending    ,
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

    std::vector<reduce_t> a = as_weights(absolute, nc);
    std::vector<reduce_t> m = as_weights(membrane, nc);
    std::vector<reduce_t> b = as_weights(bending,  nc);

    if (bending)
        reg_field::diag_bending<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, a.data(), m.data(), b.data(), stream);
    else if (membrane)
        reg_field::diag_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, a.data(), m.data(), stream);
    else
        reg_field::diag_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, a.data(), stream);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
}

// Materialise the per-channel Toeplitz stencil (*batch, *spatial, C); the field
// regulariser never couples channels, so there is no matrix case. Mirrors
// _field_diag; forwards the CUDA stream. kernel_absolute takes no voxel_size.
template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_kernel(
    const bound::BoundVec & bvec,
          int64_t   nbatch     ,
          int64_t   nc         ,
          void    * out        ,
    const double  * voxel_size ,
    const double  * absolute   ,
    const double  * membrane   ,
    const double  * bending    ,
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

    std::vector<reduce_t> a = as_weights(absolute, nc);
    std::vector<reduce_t> m = as_weights(membrane, nc);
    std::vector<reduce_t> b = as_weights(bending,  nc);

    if (bending)
        reg_field::kernel_bending<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, a.data(), m.data(), b.data(), stream);
    else if (membrane)
        reg_field::kernel_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, a.data(), m.data(), stream);
    else
        reg_field::kernel_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, a.data(), stream);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
}

// Gauss-Seidel relaxation: refine `sol` towards the solution of (H + L) x = g,
// dispatching bending vs membrane and forwarding the CUDA stream. Mirrors the
// CPU `_field_relax`; the cuda-impl launchers loop the red-black colours x
// nb_iter internally.
template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_relax(
    const bound::BoundVec & bvec,
          int64_t   nbatch     ,
          int64_t   nc         ,
          void    * sol        ,
    const void    * hes        ,
    const void    * grd        ,
    const double  * voxel_size ,
    const double  * absolute   ,
    const double  * membrane   ,
    const double  * bending    ,
          int       niter      ,
    const int64_t * size       ,
    const int64_t * stride_sol ,
    const int64_t * stride_hes ,
    const int64_t * stride_grd ,
          cudaStream_t stream  )
{
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_sol = copy_if_needed<offset_t *>(stride_sol, nall1);
    const offset_t * _stride_hes = copy_if_needed<offset_t *>(stride_hes, nall1);
    const offset_t * _stride_grd = copy_if_needed<offset_t *>(stride_grd, nall1);
          scalar_t * _sol = static_cast<      scalar_t *>(sol);
    const scalar_t * _hes = static_cast<const scalar_t *>(hes);
    const scalar_t * _grd = static_cast<const scalar_t *>(grd);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    std::vector<reduce_t> a = as_weights(absolute, nc);
    std::vector<reduce_t> m = as_weights(membrane, nc);
    std::vector<reduce_t> b = as_weights(bending,  nc);

    if (bending)
        reg_field::relax_bending_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd,
            _size, _stride_sol, _stride_hes, _stride_grd, vx,
            a.data(), m.data(), b.data(), niter, stream);
    else
        reg_field::relax_membrane_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd,
            _size, _stride_sol, _stride_hes, _stride_grd, vx,
            a.data(), m.data(), niter, stream);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_sol);
    free_if_needed<int64_t *>(_stride_hes);
    free_if_needed<int64_t *>(_stride_grd);
}


} // anonymous namespace

/***********************************************************************
 *                            DISPATCH                                 *
 ***********************************************************************/

#define BND1(B) B
#define BND2(B) B, B
#define BND3(B) B, B, B

#define MV_DT(NDIM, BNDS...)                                            \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _field_matvec<NDIM, float,  int32_t, BNDS>(MV_ARGS)   \
                : _field_matvec<NDIM, float,  int64_t, BNDS>(MV_ARGS);  \
            case 64: return use_32bits                                  \
                ? _field_matvec<NDIM, double, int32_t, BNDS>(MV_ARGS)   \
                : _field_matvec<NDIM, double, int64_t, BNDS>(MV_ARGS);  \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define ADD_MV_DT(NDIM, BNDS...)                                        \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _field_matvec_acc<NDIM, '+', float,  int32_t, BNDS>(MV_ARGS) \
                : _field_matvec_acc<NDIM, '+', float,  int64_t, BNDS>(MV_ARGS); \
            case 64: return use_32bits                                  \
                ? _field_matvec_acc<NDIM, '+', double, int32_t, BNDS>(MV_ARGS) \
                : _field_matvec_acc<NDIM, '+', double, int64_t, BNDS>(MV_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define SUB_MV_DT(NDIM, BNDS...)                                        \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _field_matvec_acc<NDIM, '-', float,  int32_t, BNDS>(MV_ARGS) \
                : _field_matvec_acc<NDIM, '-', float,  int64_t, BNDS>(MV_ARGS); \
            case 64: return use_32bits                                  \
                ? _field_matvec_acc<NDIM, '-', double, int32_t, BNDS>(MV_ARGS) \
                : _field_matvec_acc<NDIM, '-', double, int64_t, BNDS>(MV_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define KN_DT(NDIM, BNDS...)                                            \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _field_kernel<NDIM, float,  int32_t, BNDS>(KN_ARGS)   \
                : _field_kernel<NDIM, float,  int64_t, BNDS>(KN_ARGS);  \
            case 64: return use_32bits                                  \
                ? _field_kernel<NDIM, double, int32_t, BNDS>(KN_ARGS)   \
                : _field_kernel<NDIM, double, int64_t, BNDS>(KN_ARGS);  \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define DG_DT(NDIM, BNDS...)                                            \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _field_diag<NDIM, float,  int32_t, BNDS>(DG_ARGS)     \
                : _field_diag<NDIM, float,  int64_t, BNDS>(DG_ARGS);    \
            case 64: return use_32bits                                  \
                ? _field_diag<NDIM, double, int32_t, BNDS>(DG_ARGS)     \
                : _field_diag<NDIM, double, int64_t, BNDS>(DG_ARGS);    \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define RX_DT(NDIM, BNDS...)                                            \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _field_relax<NDIM, float,  int32_t, BNDS>(RX_ARGS)    \
                : _field_relax<NDIM, float,  int64_t, BNDS>(RX_ARGS);   \
            case 64: return use_32bits                                  \
                ? _field_relax<NDIM, double, int32_t, BNDS>(RX_ARGS)    \
                : _field_relax<NDIM, double, int64_t, BNDS>(RX_ARGS);   \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

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
        default: throw std::invalid_argument("Only 1D, 2D and 3D field are supported"); \
    }

void field_matvec(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
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
    CHECK_SAME_SHAPE(out, inp, out.ndim)

    const int64_t    nc = out.shape[out.ndim - 1];
    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    const bound::BoundVec bvec(bnd);
    const cudaStream_t cstream = _reg_stream(stream);

#define MV_ARGS bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(out), CVOIDPTR(inp), \
                voxel_size, absolute, membrane, bending,                       \
                out.shape, out.strides, inp.strides, cstream
    NDIM_SWITCH(MV_DT)
#undef MV_ARGS
}

/**
 * @brief `field_matvec` variant that accumulates into `out`: `out += L(inp)`,
 *        instead of overwriting it. Same conventions otherwise.
 */
void field_matvec_add(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
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
    CHECK_SAME_SHAPE(out, inp, out.ndim)

    const int64_t    nc = out.shape[out.ndim - 1];
    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    const bound::BoundVec bvec(bnd);
    const cudaStream_t cstream = _reg_stream(stream);

#define MV_ARGS bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(out), CVOIDPTR(inp), \
                voxel_size, absolute, membrane, bending,                       \
                out.shape, out.strides, inp.strides, cstream
    NDIM_SWITCH(ADD_MV_DT)
#undef MV_ARGS
}

/**
 * @brief `field_matvec` variant that subtracts from `out`: `out -= L(inp)`,
 *        instead of overwriting it. Same conventions otherwise.
 */
void field_matvec_sub(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
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
    CHECK_SAME_SHAPE(out, inp, out.ndim)

    const int64_t    nc = out.shape[out.ndim - 1];
    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    const bound::BoundVec bvec(bnd);
    const cudaStream_t cstream = _reg_stream(stream);

#define MV_ARGS bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(out), CVOIDPTR(inp), \
                voxel_size, absolute, membrane, bending,                       \
                out.shape, out.strides, inp.strides, cstream
    NDIM_SWITCH(SUB_MV_DT)
#undef MV_ARGS
}

void field_diag(
          DLTensor & out_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
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

    const int64_t    nc = out.shape[out.ndim - 1];
    const bool     use_32bits = CANUSE32BITS(out);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    const bound::BoundVec bvec(bnd);
    const cudaStream_t cstream = _reg_stream(stream);

#define DG_ARGS bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(out), \
                voxel_size, absolute, membrane, bending,         \
                out.shape, out.strides, cstream
    NDIM_SWITCH(DG_DT)
#undef DG_ARGS
}

void field_kernel(
          DLTensor & out_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
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

    const int64_t    nc = out.shape[out.ndim - 1];
    const bool     use_32bits = CANUSE32BITS(out);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    const bound::BoundVec bvec(bnd);
    const cudaStream_t cstream = _reg_stream(stream);

#define KN_ARGS bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(out), \
                voxel_size, absolute, membrane, bending,         \
                out.shape, out.strides, cstream
    NDIM_SWITCH(KN_DT)
#undef KN_ARGS
}

void field_relax(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        nb_iter   ,
          int        stream
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _sol(sol), _hes(hes), _grd(grd);
    DLTensor       & s = _sol.t;
    const DLTensor & h = _hes.t;
    const DLTensor & g = _grd.t;

    const int32_t nbatch = s.ndim - ndim - 1;
    CHECK_NO_LANES  (s)
    CHECK_SAME_DTYPE(s, h)
    CHECK_SAME_DTYPE(s, g)
    CHECK_SAME      (s.ndim, g.ndim, "Tensors do not have the same number of dimensions")
    CHECK_SAME      (s.ndim, h.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME_SHAPE(s, g, s.ndim)

    const int64_t    nc = s.shape[s.ndim - 1];
    const bool     use_32bits = CANUSE32BITS(s) && CANUSE32BITS(h) &&
                                CANUSE32BITS(g);
    const auto     code = static_cast<DLDataTypeCode>(s.dtype.code);
    const auto     bits = s.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    const bound::BoundVec bvec(bnd);
    const cudaStream_t cstream = _reg_stream(stream);

#define RX_ARGS bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(s), CVOIDPTR(h),     \
                CVOIDPTR(g), voxel_size, absolute, membrane, bending,          \
                nb_iter, s.shape, s.strides, h.strides, g.strides, cstream
    NDIM_SWITCH(RX_DT)
#undef RX_ARGS
}

void field_forward(
          DLTensor & out       ,
    const DLTensor & hes       ,
    const DLTensor & inp       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    sym_matvec(out, hes, inp, stream);
    field_matvec_add(out, inp, voxel_size, absolute, membrane, bending, bound, ndim, stream);
}

// `field_diag`'s regulariser diagonal doesn't depend on the operand being
// solved for, so `field_precond[_]` materialise it into a fresh contiguous
// device scratch buffer shaped like `grd`/`sol` and hand it to
// posdef::sym_solve[_] as the per-channel weight map. Caller owns the
// returned device pointer and must freeDevice() it.
static inline uint8_t * field_precond_diag(
    const DLTensor & like      ,
    const double    * voxel_size,
    const double    * absolute  ,
    const double    * membrane  ,
    const double    * bending   ,
          int8_t      bound     ,
          int         ndim      ,
          int         stream    ,
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

    field_diag(diag_t, voxel_size, absolute, membrane, bending, bound, ndim, stream);
    return diag_buf;
}

void field_precond(
          DLTensor & out       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    CHECK_NO_LANES(grd)
    if (grd.ndim - ndim - 1 < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");

    DLTensor diag_t;
    uint8_t * diag_buf = field_precond_diag(
        grd, voxel_size, absolute, membrane, bending, bound, ndim, stream, diag_t);
    try {
        sym_solve(out, hes, grd, diag_t, stream);
    } catch (...) {
        freeDevice(diag_buf);
        throw;
    }
    freeDevice(diag_buf);
}

void field_precond_(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    CHECK_NO_LANES(sol)
    if (sol.ndim - ndim - 1 < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");

    DLTensor diag_t;
    uint8_t * diag_buf = field_precond_diag(
        sol, voxel_size, absolute, membrane, bending, bound, ndim, stream, diag_t);
    try {
        sym_solve_(sol, hes, diag_t, stream);
    } catch (...) {
        freeDevice(diag_buf);
        throw;
    }
    freeDevice(diag_buf);
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
