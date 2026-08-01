#include <stdexcept>
#include <vector>
#include "reg_field.h"
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

// Reject the boundary conditions under which the requested operator is not
// self-adjoint (fastfields-kernels#50 decision 2, as corrected 2026-08-01).
//
// The difference-form stencil is exact at a boundary only where the fold is an
// involution on the tap set. Larger reach folds more taps, so which conditions
// survive depends on the energy, and the set is MEASURED (assemble `A`, take
// `max|A-A^T|/max|A|`) rather than argued:
//
//        bound      | absolute | membrane | bending
//        -----------+----------+----------+---------
//        Replicate  |    ok    |    ok    | REJECT   (0.042-0.13)
//        DCT1       |    ok    |  REJECT  | REJECT   (0.25-0.46 / 0.37-0.50)
//        all others |    ok    |    ok    |   ok     (exactly 0)
//
// DCT1's whole-sample fold lands the -1 tap of x=0 onto its own +1 tap, so
// A[0][1] picks up the fold and A[1][0] does not -- that bites from reach 1
// upwards, i.e. membrane as well as bending. Replicate's clamp is idempotent
// rather than involutive (at x=0 both x-1 and x-2 fold onto 0), which needs a
// +-2 tap to bite, so bending only. An asymmetric operator is not something CG
// or relaxation can solve; failing loudly beats converging to the wrong answer.
//
// The rule itself lives in `bound::supports_{absolute,membrane,bending}` on the
// kernels side, so there is exactly one definition of it.
//
// Checked ONCE here at the dispatch entry, never per voxel: past this point
// every voxel in the stencil loop may assume a self-adjoint-capable boundary
// with no runtime branching.
static inline void check_selfadjoint_bound(
    const double * membrane, const double * bending, bound::type bnd)
{
    // Mirror the wrappers' energy selection EXACTLY -- highest-order non-null
    // penalty wins -- or the check and the kernel it guards can disagree.
    if (bending) {
        if (!bound::supports_bending(bnd))
            throw std::invalid_argument(
                "the bending penalty is not self-adjoint under the Replicate "
                "and DCT1 boundary conditions; use Zero, DCT2, DST1, DST2, "
                "DFT or NoCheck");
    } else if (membrane) {
        if (!bound::supports_membrane(bnd))
            throw std::invalid_argument(
                "the membrane penalty is not self-adjoint under the DCT1 "
                "boundary condition; use Zero, Replicate, DCT2, DST1, DST2, "
                "DFT or NoCheck");
    }
    // absolute reads no neighbour, so it has no fold and no condition to
    // reject (`bound::supports_absolute` is true for all eight).
}

/***********************************************************************
 *                             WRAPPERS                                *
 ***********************************************************************/

namespace {

// build a length-nc reduce_t vector from a (possibly null) double array
static inline std::vector<reduce_t> as_weights(const double * w, int64_t nc)
{
    std::vector<reduce_t> v(static_cast<size_t>(nc), reduce_t(0));
    if (w) for (int64_t c = 0; c < nc; ++c) v[static_cast<size_t>(c)] = w[c];
    return v;
}

template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_matvec(
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
    const int64_t * stride_inp )
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
            static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, a.data(), m.data(), b.data());
    else if (membrane)
        reg_field::matvec_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, a.data(), m.data());
    else
        reg_field::matvec_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, a.data());

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_inp);
}

template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_diag(
          int64_t   nbatch     ,
          int64_t   nc         ,
          void    * out        ,
    const double  * voxel_size ,
    const double  * absolute   ,
    const double  * membrane   ,
    const double  * bending    ,
    const int64_t * size       ,
    const int64_t * stride_out )
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
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, a.data(), m.data(), b.data());
    else if (membrane)
        reg_field::diag_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, a.data(), m.data());
    else
        reg_field::diag_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, a.data());

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
}

// Materialise the Toeplitz convolution kernel (stencil) of the operator. The
// output is a per-channel vector stencil (*batch, *spatial, C) -- the field
// regulariser never couples channels, so there is no matrix case. Dispatch
// mirrors _field_diag: the highest-order non-null penalty selects the stencil.
template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_kernel(
          int64_t   nbatch     ,
          int64_t   nc         ,
          void    * out        ,
    const double  * voxel_size ,
    const double  * absolute   ,
    const double  * membrane   ,
    const double  * bending    ,
    const int64_t * size       ,
    const int64_t * stride_out )
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
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, a.data(), m.data(), b.data());
    else if (membrane)
        reg_field::kernel_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, a.data(), m.data());
    else
        // kernel_absolute takes no voxel_size (the L2 stencil is scale-free).
        reg_field::kernel_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, a.data());

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
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

#define BOUND_SWITCH(DT, NDIM, BND)                                     \
    switch (bnd) {                                                      \
        case bound::type::Zero:      DT(NDIM, BND(bound::type::Zero));      break; \
        case bound::type::Replicate: DT(NDIM, BND(bound::type::Replicate)); break; \
        case bound::type::DCT1:      DT(NDIM, BND(bound::type::DCT1));      break; \
        case bound::type::DCT2:      DT(NDIM, BND(bound::type::DCT2));      break; \
        case bound::type::DST1:      DT(NDIM, BND(bound::type::DST1));      break; \
        case bound::type::DST2:      DT(NDIM, BND(bound::type::DST2));      break; \
        case bound::type::DFT:       DT(NDIM, BND(bound::type::DFT));       break; \
        case bound::type::NoCheck:   DT(NDIM, BND(bound::type::NoCheck));   break; \
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
          int        /* stream <unused> */
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
    check_selfadjoint_bound(membrane, bending, bnd);

#define MV_ARGS static_cast<int64_t>(nbatch), nc, VOIDPTR(out), CVOIDPTR(inp), \
                voxel_size, absolute, membrane, bending,                       \
                out.shape, out.strides, inp.strides
    NDIM_SWITCH(MV_DT)
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
          int        /* stream <unused> */
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
    check_selfadjoint_bound(membrane, bending, bnd);

#define DG_ARGS static_cast<int64_t>(nbatch), nc, VOIDPTR(out), \
                voxel_size, absolute, membrane, bending,         \
                out.shape, out.strides
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
          int        /* stream <unused> */
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
    // Deliberately NOT check_selfadjoint_bound: `kernel_*` writes the interior
    // Toeplitz stencil at pure strides and never consults the boundary at all,
    // so there is a well-defined answer to return here even for a condition
    // under which the assembled operator would not be self-adjoint. The
    // rejection belongs on the operator entry points (field_matvec/field_diag).

#define KN_ARGS static_cast<int64_t>(nbatch), nc, VOIDPTR(out), \
                voxel_size, absolute, membrane, bending,         \
                out.shape, out.strides
    NDIM_SWITCH(KN_DT)
#undef KN_ARGS
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
