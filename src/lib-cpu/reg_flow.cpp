#include <stdexcept>
#include <string>
#include "reg_flow.h"
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

// Reject the boundary conditions under which the requested operator is not
// self-adjoint (fastfields-kernels#50 decision 2; the flow half is #59).
//
// The difference-form stencil is exact at a boundary only where the fold is an
// involution on the tap set, and WHICH conditions survive depends on which
// energy is active. Two independent mechanisms are in play here, so two
// predicates:
//
//   * the per-component SAME-AXIS stencil, keyed on reach -- more reach folds
//     more taps, so it can only lose conditions (`bound::supports_reach`);
//   * Lame's CROSS-CHANNEL block, which reads the other velocity component
//     through `bound::transpose(B)` (`bound::supports_lame_cross`). It is not
//     a reach question and its answer does not follow from the reach table in
//     either direction: Replicate is fine at reach 1 but not here, and DST1 is
//     fine at reach 2 for the same-axis stencil but not here.
//
// Measured, never argued: assemble `A` column by column (matvec on unit
// vectors) and take `max|A - A^T| / max|A|`; then, as an independent check,
// `|<Av,w> - <v,Aw>|` over random v, w with no matrix assembled at all. Both
// agree, D = 1..3, several grids, isotropic and anisotropic voxels:
//
//     bound      | absolute | membrane | bending | lame (1d / nd) | all (1d / nd)
//     -----------+----------+----------+---------+----------------+--------------
//     Zero       |    ok    |    ok    |   ok    |   ok  /  ok    |  ok  /  ok
//     Replicate  |    ok    |    ok    | REJECT  |   ok  / REJECT | REJ  / REJECT
//     DCT1       |    ok    |  REJECT  | REJECT  | REJECT/ REJECT | REJ  / REJECT
//     DCT2       |    ok    |    ok    |   ok    |   ok  /  ok    |  ok  /  ok
//     DST1       |    ok    |    ok    |   ok    |   ok  / REJECT |  ok  / REJECT
//     DST2       |    ok    |    ok    |   ok    |   ok  /  ok    |  ok  /  ok
//     DFT        |    ok    |    ok    |   ok    |   ok  /  ok    |  ok  /  ok
//     NoCheck    |    ok    |    ok    |   ok    |   ok  /  ok    |  ok  /  ok
//
// Every `ok` is a measured 0, not "small". `1d` and `nd` differ because there
// is no axis PAIR at D == 1, hence no cross block and no extra condition.
//
// An asymmetric operator is not something CG or relaxation can solve, so
// failing loudly beats converging to the wrong answer.
//
// Checked ONCE here at the dispatch entry, never per voxel: past this point
// every voxel in the stencil loop may assume a self-adjoint-capable boundary
// with no runtime branching.
static const char * const BOUND_NAME[8] = {
    "Zero", "Replicate", "DCT1", "DCT2", "DST1", "DST2", "DFT", "NoCheck"
};

// Spell the accepted set out of the predicate itself rather than transcribing
// it into a string literal, so the message cannot drift from the rule.
template <class Pred>
static std::string accepted_bounds(Pred ok)
{
    std::string s;
    for (int i = 0; i < 8; ++i)
        if (ok(static_cast<bound::type>(i))) {
            if (!s.empty()) s += ", ";
            s += BOUND_NAME[i];
        }
    return s;
}

template <class Pred>
static void reject_unless(Pred ok, bound::type bnd, const char * energy)
{
    if (ok(bnd)) return;
    throw std::invalid_argument(
        std::string("the ") + energy + " penalty is not self-adjoint under the "
        + BOUND_NAME[static_cast<int>(bnd)] + " boundary condition; use "
        + accepted_bounds(ok));
}

static inline void check_selfadjoint_bound(
    double membrane, double bending, double shears, double div,
    bound::type bnd, int ndim)
{
    // Mirror the wrappers' energy selection EXACTLY -- the Lame terms select
    // the C x C stencil, else the highest-order non-null penalty wins -- or the
    // check and the kernel it guards can disagree.
    if (shears != 0.0 || div != 0.0) {
        if (bending != 0.0)
            reject_unless([ndim](bound::type b)
                { return bound::supports_lame_bending(b, ndim); },
                bnd, "linear-elastic + bending");
        else
            reject_unless([ndim](bound::type b)
                { return bound::supports_lame(b, ndim); },
                bnd, "linear-elastic");
    } else if (bending != 0.0) {
        reject_unless(bound::supports_bending, bnd, "bending");
    } else if (membrane != 0.0) {
        reject_unless(bound::supports_membrane, bnd, "membrane");
    }
    // absolute reads no neighbour, so it has no fold and no condition to
    // reject (`bound::supports_absolute` is true for all eight).
}

/***********************************************************************
 *                             WRAPPERS                                *
 ***********************************************************************/

namespace {

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
          double    shears     ,
          double    div        ,
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

    // The linear-elastic (Lamé) terms `shears`/`div` couple the flow channels,
    // so any non-zero one selects the C x C matrix stencil (matvec_all when
    // bending is also on, else the cheaper matvec_lame). Otherwise fall back
    // to the per-channel single-penalty stencils (highest-order non-zero wins).
    if (shears != 0.0 || div != 0.0) {
        if (bending != 0.0)
            reg_flow::matvec_all<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out, _inp,
                _size, _stride_out, _stride_inp, vx,
                absolute, membrane, bending, shears, div);
        else
            reg_flow::matvec_lame<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out, _inp,
                _size, _stride_out, _stride_inp, vx,
                absolute, membrane, shears, div);
    } else if (bending != 0.0)
        reg_flow::matvec_bending<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute, membrane, bending);
    else if (membrane != 0.0)
        reg_flow::matvec_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute, membrane);
    else
        reg_flow::matvec_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute);

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
          double    shears     ,
          double    div        ,
    const int64_t * size       ,
    const int64_t * stride_out )
{
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nall1);
          scalar_t * _out = static_cast<scalar_t *>(out);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    if (shears != 0.0 || div != 0.0) {
        if (bending != 0.0)
            reg_flow::diag_all<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out,
                _size, _stride_out, vx, absolute, membrane, bending, shears, div);
        else
            reg_flow::diag_lame<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out,
                _size, _stride_out, vx, absolute, membrane, shears, div);
    } else if (bending != 0.0)
        reg_flow::diag_bending<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane, bending);
    else if (membrane != 0.0)
        reg_flow::diag_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane);
    else
        reg_flow::diag_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
}

// Materialise the Toeplitz convolution kernel (stencil) of the operator.
// `nfull` is the length of the size/stride arrays (== out.ndim): it is
// nbatch+ndim+1 for the vector (per-channel) stencil and nbatch+ndim+2 for the
// matrix (Lamé, cross-channel) stencil. Dispatch mirrors jitfields: a non-zero
// `shears`/`div` selects the C x C matrix stencil (kernel_all when bending is
// also on, else kernel_lame); otherwise the highest-order penalty picks the
// per-channel vector stencil.
template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _flow_kernel(
          int64_t   nbatch     ,
          void    * out        ,
    const double  * voxel_size ,
          double    absolute   ,
          double    membrane   ,
          double    bending    ,
          double    shears     ,
          double    div        ,
    const int64_t * size       ,
    const int64_t * stride_out ,
          int64_t   nfull      )
{
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nfull);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nfull);
          scalar_t * _out = static_cast<scalar_t *>(out);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    if (shears != 0.0 || div != 0.0) {
        if (bending != 0.0)
            reg_flow::kernel_all<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out,
                _size, _stride_out, vx, absolute, membrane, bending, shears, div);
        else
            reg_flow::kernel_lame<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out,
                _size, _stride_out, vx, absolute, membrane, shears, div);
    } else if (bending != 0.0)
        reg_flow::kernel_bending<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane, bending);
    else if (membrane != 0.0)
        reg_flow::kernel_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane);
    else
        reg_flow::kernel_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
}

// One or more relaxation (Gauss-Seidel) sweeps solving `(H + L) x = g` in
// place, where H is the per-voxel symmetric Hessian, L the flow regulariser,
// and x the warm-started `sol`. Dispatches to the impl relaxer matching the
// highest-order penalty (membrane covers the absolute-only case).
template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _flow_relax(
          int64_t   nbatch     ,
          void    * sol        ,
    const void    * hes        ,
    const void    * grd        ,
    const double  * voxel_size ,
          double    absolute   ,
          double    membrane   ,
          double    bending    ,
          double    shears     ,
          double    div        ,
          int       niter      ,
    const int64_t * size       ,
    const int64_t * stride_sol ,
    const int64_t * stride_hes ,
    const int64_t * stride_grd )
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

    if (shears != 0.0 || div != 0.0) {
        if (bending != 0.0)
            reg_flow::relax_all_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _sol, _hes, _grd,
                _size, _stride_sol, _stride_hes, _stride_grd, vx,
                absolute, membrane, bending, shears, div, niter);
        else
            reg_flow::relax_lame_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _sol, _hes, _grd,
                _size, _stride_sol, _stride_hes, _stride_grd, vx,
                absolute, membrane, shears, div, niter);
    } else if (bending != 0.0)
        reg_flow::relax_bending_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _sol, _hes, _grd,
            _size, _stride_sol, _stride_hes, _stride_grd, vx,
            absolute, membrane, bending, niter);
    else
        reg_flow::relax_membrane_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _sol, _hes, _grd,
            _size, _stride_sol, _stride_hes, _stride_grd, vx,
            absolute, membrane, niter);

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

// matvec dtype x offset dispatch, given ndim and the (repeated) bound pack.
#define MV_DT(NDIM, BNDS...)                                            \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_matvec<NDIM, float,  int32_t, BNDS>(MV_ARGS)    \
                : _flow_matvec<NDIM, float,  int64_t, BNDS>(MV_ARGS);   \
            case 64: return use_32bits                                  \
                ? _flow_matvec<NDIM, double, int32_t, BNDS>(MV_ARGS)    \
                : _flow_matvec<NDIM, double, int64_t, BNDS>(MV_ARGS);   \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define DG_DT(NDIM, BNDS...)                                            \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_diag<NDIM, float,  int32_t, BNDS>(DG_ARGS)      \
                : _flow_diag<NDIM, float,  int64_t, BNDS>(DG_ARGS);     \
            case 64: return use_32bits                                  \
                ? _flow_diag<NDIM, double, int32_t, BNDS>(DG_ARGS)      \
                : _flow_diag<NDIM, double, int64_t, BNDS>(DG_ARGS);     \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define KN_DT(NDIM, BNDS...)                                            \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_kernel<NDIM, float,  int32_t, BNDS>(KN_ARGS)    \
                : _flow_kernel<NDIM, float,  int64_t, BNDS>(KN_ARGS);   \
            case 64: return use_32bits                                  \
                ? _flow_kernel<NDIM, double, int32_t, BNDS>(KN_ARGS)    \
                : _flow_kernel<NDIM, double, int64_t, BNDS>(KN_ARGS);   \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define RX_DT(NDIM, BNDS...)                                            \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_relax<NDIM, float,  int32_t, BNDS>(RX_ARGS)     \
                : _flow_relax<NDIM, float,  int64_t, BNDS>(RX_ARGS);    \
            case 64: return use_32bits                                  \
                ? _flow_relax<NDIM, double, int32_t, BNDS>(RX_ARGS)     \
                : _flow_relax<NDIM, double, int64_t, BNDS>(RX_ARGS);    \
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
        default: throw std::invalid_argument("Only 1D, 2D and 3D flow are supported"); \
    }

void flow_matvec(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
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
    CHECK_SAME      (out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    CHECK_SAME_SHAPE(out, inp, out.ndim)

    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    check_selfadjoint_bound(membrane, bending, shears, div, bnd, ndim);

#define MV_ARGS static_cast<int64_t>(nbatch), VOIDPTR(out), CVOIDPTR(inp), \
                voxel_size, absolute, membrane, bending, shears, div,      \
                out.shape, out.strides, inp.strides
    NDIM_SWITCH(MV_DT)
#undef MV_ARGS
}

void flow_diag(
          DLTensor & out_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
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
    CHECK_SAME    (out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")

    const bool     use_32bits = CANUSE32BITS(out);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    check_selfadjoint_bound(membrane, bending, shears, div, bnd, ndim);

#define DG_ARGS static_cast<int64_t>(nbatch), VOIDPTR(out),          \
                voxel_size, absolute, membrane, bending, shears, div, \
                out.shape, out.strides
    NDIM_SWITCH(DG_DT)
#undef DG_ARGS
}

void flow_kernel(
          DLTensor & out_      ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_);
    DLTensor & out = _out.t;

    // The Lamé (shears/div) stencil is a C x C matrix of kernels (one extra
    // trailing axis); every other penalty gives a per-channel vector of
    // kernels. The output rank tells us which, and fixes nbatch.
    const bool is_matrix = (shears != 0.0 || div != 0.0);
    const int  ntrail    = is_matrix ? 2 : 1;
    const int32_t nbatch = out.ndim - ndim - ntrail;

    CHECK_NO_LANES(out)
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME(out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    if (is_matrix)
        CHECK_SAME(out.shape[out.ndim-2], (int64_t)ndim,
                   "Lamé kernel needs a trailing (ndim, ndim) matrix axis")

    const bool     use_32bits = CANUSE32BITS(out);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    // flow_kernel is EXEMPT: it materialises the interior Toeplitz stencil at
    // pure strides and never consults the boundary, so a well-defined answer
    // exists for every condition (same rule as field_kernel).

#define KN_ARGS static_cast<int64_t>(nbatch), VOIDPTR(out),          \
                voxel_size, absolute, membrane, bending, shears, div, \
                out.shape, out.strides, static_cast<int64_t>(out.ndim)
    NDIM_SWITCH(KN_DT)
#undef KN_ARGS
}

void flow_relax(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        nb_iter   ,
          int        /* stream <unused> */
)
{
    const int32_t nbatch = sol.ndim - ndim - 1;
    CHECK_NO_LANES  (sol)
    CHECK_SAME_DTYPE(sol, hes)
    CHECK_SAME_DTYPE(sol, grd)
    CHECK_SAME      (sol.ndim, grd.ndim, "Tensors do not have the same number of dimensions")
    CHECK_SAME      (sol.ndim, hes.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME      (sol.shape[sol.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    CHECK_SAME      (grd.shape[grd.ndim-1], (int64_t)ndim, "Gradient channel dimension must equal ndim")
    CHECK_SAME_SHAPE(sol, grd, sol.ndim)

    const bool     use_32bits = CANUSE32BITS(sol) && CANUSE32BITS(hes) &&
                                CANUSE32BITS(grd);
    const auto     code = static_cast<DLDataTypeCode>(sol.dtype.code);
    const auto     bits = sol.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    // relax SOLVES with this operator, so an asymmetric one is exactly what
    // it cannot handle -- checked here too (reg_field currently checks only
    // its matvec/diag entries; aligning it is a follow-up).
    check_selfadjoint_bound(membrane, bending, shears, div, bnd, ndim);

#define RX_ARGS static_cast<int64_t>(nbatch), VOIDPTR(sol), CVOIDPTR(hes),    \
                CVOIDPTR(grd), voxel_size, absolute, membrane, bending,       \
                shears, div, nb_iter, sol.shape, sol.strides, hes.strides,    \
                grd.strides
    NDIM_SWITCH(RX_DT)
#undef RX_ARGS
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
