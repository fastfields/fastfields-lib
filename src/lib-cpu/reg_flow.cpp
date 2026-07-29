#include <stdexcept>
#include <string>
#include <vector>
#include "reg_flow.h"
#include "posdef.h"
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

// length of the shape/stride arrays: (*batch, *spatial, C) == out.ndim
template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _flow_matvec(
    const bound::BoundVec & bvec,
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
    // so any non-zero one selects the full combined stencil (matvec_all, which
    // also folds in absolute/membrane/bending). Otherwise fall back to the
    // cheaper single-penalty stencils (highest-order non-zero wins).
    if (shears != 0.0 || div != 0.0)
        reg_flow::matvec_all<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx,
            absolute, membrane, bending, shears, div);
    else if (bending != 0.0)
        reg_flow::matvec_bending<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute, membrane, bending);
    else if (membrane != 0.0)
        reg_flow::matvec_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute, membrane);
    else
        reg_flow::matvec_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_inp);
}

// Accumulate variant of _flow_matvec: out += L(inp) (op='+') or out -= L(inp)
// (op='-'), instead of overwriting out. Mirrors _field_matvec_acc.
template <int ndim, char op, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _flow_matvec_acc(
    const bound::BoundVec & bvec,
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

    if (shears != 0.0 || div != 0.0)
        reg_flow::matvec_all<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx,
            absolute, membrane, bending, shears, div);
    else if (bending != 0.0)
        reg_flow::matvec_bending<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute, membrane, bending);
    else if (membrane != 0.0)
        reg_flow::matvec_membrane<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute, membrane);
    else
        reg_flow::matvec_absolute<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_inp);
}

template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _flow_diag(
    const bound::BoundVec & bvec,
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

    if (shears != 0.0 || div != 0.0)
        reg_flow::diag_all<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane, bending, shears, div);
    else if (bending != 0.0)
        reg_flow::diag_bending<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane, bending);
    else if (membrane != 0.0)
        reg_flow::diag_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane);
    else
        reg_flow::diag_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
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
    const bound::BoundVec & bvec,
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
                bvec, static_cast<offset_t>(nbatch), _out,
                _size, _stride_out, vx, absolute, membrane, bending, shears, div);
        else
            reg_flow::kernel_lame<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _out,
                _size, _stride_out, vx, absolute, membrane, shears, div);
    } else if (bending != 0.0)
        reg_flow::kernel_bending<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane, bending);
    else if (membrane != 0.0)
        reg_flow::kernel_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane);
    else
        reg_flow::kernel_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
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
    const bound::BoundVec & bvec,
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
                bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd,
                _size, _stride_sol, _stride_hes, _stride_grd, vx,
                absolute, membrane, bending, shears, div, niter);
        else
            reg_flow::relax_lame_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd,
                _size, _stride_sol, _stride_hes, _stride_grd, vx,
                absolute, membrane, shears, div, niter);
    } else if (bending != 0.0)
        reg_flow::relax_bending_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd,
            _size, _stride_sol, _stride_hes, _stride_grd, vx,
            absolute, membrane, bending, niter);
    else
        reg_flow::relax_membrane_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd,
            _size, _stride_sol, _stride_hes, _stride_grd, vx,
            absolute, membrane, niter);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_sol);
    free_if_needed<int64_t *>(_stride_hes);
    free_if_needed<int64_t *>(_stride_grd);
}

// Joint-reweighted-least-squares (JRLS) variant of `_flow_matvec`: an extra
// per-voxel weight map `wgt` (shared across all ndim components) modulates
// the penalty strength. Only membrane_jrls and lame_jrls exist at the impl
// layer (matching jitfields, which never wires an absolute-only or
// bending-aware jrls kernel) -- `bending` is rejected by the public wrapper
// before dispatch ever reaches here.
template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _flow_matvec_rls(
    const bound::BoundVec & bvec,
          int64_t   nbatch     ,
          void    * out        ,
    const void    * inp        ,
    const void    * wgt        ,
    const double  * voxel_size ,
          double    absolute   ,
          double    membrane   ,
          double    shears     ,
          double    div        ,
    const int64_t * size       ,
    const int64_t * stride_out ,
    const int64_t * stride_inp ,
    const int64_t * stride_wgt )
{
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nall1);
    const offset_t * _stride_inp = copy_if_needed<offset_t *>(stride_inp, nall1);
    const offset_t * _stride_wgt = copy_if_needed<offset_t *>(stride_wgt, nall1);
          scalar_t * _out = static_cast<      scalar_t *>(out);
    const scalar_t * _inp = static_cast<const scalar_t *>(inp);
    const scalar_t * _wgt = static_cast<const scalar_t *>(wgt);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    if (shears != 0.0 || div != 0.0)
        reg_flow::matvec_lame_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp, _wgt,
            _size, _stride_out, _stride_inp, _stride_wgt, vx,
            absolute, membrane, shears, div);
    else
        reg_flow::matvec_membrane_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp, _wgt,
            _size, _stride_out, _stride_inp, _stride_wgt, vx, absolute, membrane);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_inp);
    free_if_needed<int64_t *>(_stride_wgt);
}

template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _flow_diag_rls(
    const bound::BoundVec & bvec,
          int64_t   nbatch     ,
          void    * out        ,
    const void    * wgt        ,
    const double  * voxel_size ,
          double    absolute   ,
          double    membrane   ,
          double    shears     ,
          double    div        ,
    const int64_t * size       ,
    const int64_t * stride_out ,
    const int64_t * stride_wgt )
{
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nall1);
    const offset_t * _stride_wgt = copy_if_needed<offset_t *>(stride_wgt, nall1);
          scalar_t * _out = static_cast<scalar_t *>(out);
    const scalar_t * _wgt = static_cast<const scalar_t *>(wgt);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    if (shears != 0.0 || div != 0.0)
        reg_flow::diag_lame_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _wgt,
            _size, _stride_out, _stride_wgt, vx, absolute, membrane, shears, div);
    else
        reg_flow::diag_membrane_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _wgt,
            _size, _stride_out, _stride_wgt, vx, absolute, membrane);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_wgt);
}

template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _flow_relax_rls(
    const bound::BoundVec & bvec,
          int64_t   nbatch     ,
          void    * sol        ,
    const void    * hes        ,
    const void    * grd        ,
    const void    * wgt        ,
    const double  * voxel_size ,
          double    absolute   ,
          double    membrane   ,
          double    shears     ,
          double    div        ,
          int       niter      ,
    const int64_t * size       ,
    const int64_t * stride_sol ,
    const int64_t * stride_hes ,
    const int64_t * stride_grd ,
    const int64_t * stride_wgt )
{
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_sol = copy_if_needed<offset_t *>(stride_sol, nall1);
    const offset_t * _stride_hes = copy_if_needed<offset_t *>(stride_hes, nall1);
    const offset_t * _stride_grd = copy_if_needed<offset_t *>(stride_grd, nall1);
    const offset_t * _stride_wgt = copy_if_needed<offset_t *>(stride_wgt, nall1);
          scalar_t * _sol = static_cast<      scalar_t *>(sol);
    const scalar_t * _hes = static_cast<const scalar_t *>(hes);
    const scalar_t * _grd = static_cast<const scalar_t *>(grd);
    const scalar_t * _wgt = static_cast<const scalar_t *>(wgt);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    if (shears != 0.0 || div != 0.0)
        reg_flow::relax_lame_jrls_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd, _wgt,
            _size, _stride_sol, _stride_hes, _stride_grd, _stride_wgt, vx,
            absolute, membrane, shears, div, niter);
    else
        reg_flow::relax_membrane_jrls_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd, _wgt,
            _size, _stride_sol, _stride_hes, _stride_grd, _stride_wgt, vx,
            absolute, membrane, niter);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_sol);
    free_if_needed<int64_t *>(_stride_hes);
    free_if_needed<int64_t *>(_stride_grd);
    free_if_needed<int64_t *>(_stride_wgt);
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

#define ADD_MV_DT(NDIM, BNDS...)                                        \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_matvec_acc<NDIM, '+', float,  int32_t, BNDS>(MV_ARGS) \
                : _flow_matvec_acc<NDIM, '+', float,  int64_t, BNDS>(MV_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_matvec_acc<NDIM, '+', double, int32_t, BNDS>(MV_ARGS) \
                : _flow_matvec_acc<NDIM, '+', double, int64_t, BNDS>(MV_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define SUB_MV_DT(NDIM, BNDS...)                                        \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_matvec_acc<NDIM, '-', float,  int32_t, BNDS>(MV_ARGS) \
                : _flow_matvec_acc<NDIM, '-', float,  int64_t, BNDS>(MV_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_matvec_acc<NDIM, '-', double, int32_t, BNDS>(MV_ARGS) \
                : _flow_matvec_acc<NDIM, '-', double, int64_t, BNDS>(MV_ARGS); \
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

#define RLS_MV_DT(NDIM, BNDS...)                                        \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_matvec_rls<NDIM, float,  int32_t, BNDS>(RLS_MV_ARGS) \
                : _flow_matvec_rls<NDIM, float,  int64_t, BNDS>(RLS_MV_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_matvec_rls<NDIM, double, int32_t, BNDS>(RLS_MV_ARGS) \
                : _flow_matvec_rls<NDIM, double, int64_t, BNDS>(RLS_MV_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define RLS_DG_DT(NDIM, BNDS...)                                        \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_diag_rls<NDIM, float,  int32_t, BNDS>(RLS_DG_ARGS) \
                : _flow_diag_rls<NDIM, float,  int64_t, BNDS>(RLS_DG_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_diag_rls<NDIM, double, int32_t, BNDS>(RLS_DG_ARGS) \
                : _flow_diag_rls<NDIM, double, int64_t, BNDS>(RLS_DG_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define RLS_RX_DT(NDIM, BNDS...)                                        \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_relax_rls<NDIM, float,  int32_t, BNDS>(RLS_RX_ARGS) \
                : _flow_relax_rls<NDIM, float,  int64_t, BNDS>(RLS_RX_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_relax_rls<NDIM, double, int32_t, BNDS>(RLS_RX_ARGS) \
                : _flow_relax_rls<NDIM, double, int64_t, BNDS>(RLS_RX_ARGS); \
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
    const bound::BoundVec bvec(bnd);

#define MV_ARGS bvec, static_cast<int64_t>(nbatch), VOIDPTR(out), CVOIDPTR(inp), \
                voxel_size, absolute, membrane, bending, shears, div,      \
                out.shape, out.strides, inp.strides
    NDIM_SWITCH(MV_DT)
#undef MV_ARGS
}

/**
 * @brief `flow_matvec` variant that accumulates into `out`: `out += L(inp)`,
 *        instead of overwriting it. Same conventions otherwise.
 */
void flow_matvec_add(
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
    const bound::BoundVec bvec(bnd);

#define MV_ARGS bvec, static_cast<int64_t>(nbatch), VOIDPTR(out), CVOIDPTR(inp), \
                voxel_size, absolute, membrane, bending, shears, div,      \
                out.shape, out.strides, inp.strides
    NDIM_SWITCH(ADD_MV_DT)
#undef MV_ARGS
}

/**
 * @brief `flow_matvec` variant that subtracts from `out`: `out -= L(inp)`,
 *        instead of overwriting it. Same conventions otherwise.
 */
void flow_matvec_sub(
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
    const bound::BoundVec bvec(bnd);

#define MV_ARGS bvec, static_cast<int64_t>(nbatch), VOIDPTR(out), CVOIDPTR(inp), \
                voxel_size, absolute, membrane, bending, shears, div,      \
                out.shape, out.strides, inp.strides
    NDIM_SWITCH(SUB_MV_DT)
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
    const bound::BoundVec bvec(bnd);

#define DG_ARGS bvec, static_cast<int64_t>(nbatch), VOIDPTR(out),          \
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
    const bound::BoundVec bvec(bnd);

#define KN_ARGS bvec, static_cast<int64_t>(nbatch), VOIDPTR(out),          \
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
    const bound::BoundVec bvec(bnd);

#define RX_ARGS bvec, static_cast<int64_t>(nbatch), VOIDPTR(sol), CVOIDPTR(hes),    \
                CVOIDPTR(grd), voxel_size, absolute, membrane, bending,       \
                shears, div, nb_iter, sol.shape, sol.strides, hes.strides,    \
                grd.strides
    NDIM_SWITCH(RX_DT)
#undef RX_ARGS
}

void flow_forward(
          DLTensor & out       ,
    const DLTensor & hes       ,
    const DLTensor & inp       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    sym_matvec(out, hes, inp, stream);
    flow_matvec_add(out, inp, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
}

// `flow_diag`'s regulariser diagonal doesn't depend on the operand being
// solved for, so `flow_precond[_]` materialise it into a fresh contiguous
// scratch buffer shaped like `grd`/`sol` and hand it to posdef::sym_solve[_]
// as the per-channel weight map.
static inline std::vector<uint8_t> flow_precond_diag(
    const DLTensor & like      ,
    const double    * voxel_size,
          double      absolute  ,
          double      membrane  ,
          double      bending   ,
          double      shears    ,
          double      div       ,
          int8_t      bound     ,
          int         ndim      ,
          int         stream    ,
          DLTensor  & diag_t    )
{
    size_t numel = 1;
    for (int32_t d = 0; d < like.ndim; ++d)
        numel *= static_cast<size_t>(like.shape[d]);
    std::vector<uint8_t> diag_buf(numel * static_cast<size_t>(like.dtype.bits) / 8);

    diag_t.data        = diag_buf.data();
    diag_t.device       = like.device;
    diag_t.ndim         = like.ndim;
    diag_t.dtype        = like.dtype;
    diag_t.shape        = like.shape;
    diag_t.strides      = nullptr;
    diag_t.byte_offset  = 0;

    flow_diag(diag_t, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
    return diag_buf;
}

void flow_precond(
          DLTensor & out       ,
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
          int        stream    )
{
    CHECK_NO_LANES(grd)
    if (grd.ndim - ndim - 1 < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");

    DLTensor diag_t;
    std::vector<uint8_t> diag_buf = flow_precond_diag(
        grd, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream, diag_t);
    sym_solve(out, hes, grd, diag_t, stream);
}

void flow_precond_(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    CHECK_NO_LANES(sol)
    if (sol.ndim - ndim - 1 < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");

    DLTensor diag_t;
    std::vector<uint8_t> diag_buf = flow_precond_diag(
        sol, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream, diag_t);
    sym_solve_(sol, hes, diag_t, stream);
}

// `bending` has no jrls kernel wired at the impl layer (matching jitfields,
// which never exposes this combination either) -- reject it up front rather
// than silently ignoring it.
static inline void flow_rls_check_bending(double bending, const char * who)
{
    if (bending != 0.0)
        throw std::invalid_argument(
            std::string(who) + ": bending penalty is not supported with "
            "RLS/JRLS weighting");
}

void flow_matvec_rls(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const DLTensor & wgt_      ,
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
    flow_rls_check_bending(bending, "flow_matvec_rls");

    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _inp(inp_), _wgt(wgt_);
    DLTensor       & out = _out.t;
    const DLTensor & inp = _inp.t;
    const DLTensor & wgt = _wgt.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME_DTYPE(out, wgt)
    CHECK_SAME      (out.ndim, inp.ndim, "Tensors do not have the same number of dimensions")
    CHECK_SAME      (out.ndim, wgt.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME      (out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    CHECK_SAME      (wgt.shape[wgt.ndim-1], (int64_t)1, "Weight tensor must have a trailing size-1 channel axis")
    CHECK_SAME_SHAPE(out, inp, out.ndim)
    CHECK_SAME_SHAPE(out, wgt, out.ndim - 1)

    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp) && CANUSE32BITS(wgt);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    const bound::BoundVec bvec(bnd);

#define RLS_MV_ARGS bvec, static_cast<int64_t>(nbatch), VOIDPTR(out), CVOIDPTR(inp), \
                CVOIDPTR(wgt), voxel_size, absolute, membrane, shears, div,     \
                out.shape, out.strides, inp.strides, wgt.strides
    NDIM_SWITCH(RLS_MV_DT)
#undef RLS_MV_ARGS
}

void flow_diag_rls(
          DLTensor & out_      ,
    const DLTensor & wgt_      ,
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
    flow_rls_check_bending(bending, "flow_diag_rls");

    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _wgt(wgt_);
    DLTensor       & out = _out.t;
    const DLTensor & wgt = _wgt.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, wgt)
    CHECK_SAME      (out.ndim, wgt.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME      (out.shape[out.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    CHECK_SAME      (wgt.shape[wgt.ndim-1], (int64_t)1, "Weight tensor must have a trailing size-1 channel axis")
    CHECK_SAME_SHAPE(out, wgt, out.ndim - 1)

    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(wgt);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    const bound::BoundVec bvec(bnd);

#define RLS_DG_ARGS bvec, static_cast<int64_t>(nbatch), VOIDPTR(out), CVOIDPTR(wgt), \
                voxel_size, absolute, membrane, shears, div,                   \
                out.shape, out.strides, wgt.strides
    NDIM_SWITCH(RLS_DG_DT)
#undef RLS_DG_ARGS
}

void flow_relax_rls(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const DLTensor & wgt_      ,
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
    flow_rls_check_bending(bending, "flow_relax_rls");

    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _wgt(wgt_);
    const DLTensor & wgt = _wgt.t;

    const int32_t nbatch = sol.ndim - ndim - 1;
    CHECK_NO_LANES  (sol)
    CHECK_SAME_DTYPE(sol, hes)
    CHECK_SAME_DTYPE(sol, grd)
    CHECK_SAME_DTYPE(sol, wgt)
    CHECK_SAME      (sol.ndim, grd.ndim, "Tensors do not have the same number of dimensions")
    CHECK_SAME      (sol.ndim, hes.ndim, "Tensors do not have the same number of dimensions")
    CHECK_SAME      (sol.ndim, wgt.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME      (sol.shape[sol.ndim-1], (int64_t)ndim, "Channel dimension must equal ndim")
    CHECK_SAME      (grd.shape[grd.ndim-1], (int64_t)ndim, "Gradient channel dimension must equal ndim")
    CHECK_SAME      (wgt.shape[wgt.ndim-1], (int64_t)1, "Weight tensor must have a trailing size-1 channel axis")
    CHECK_SAME_SHAPE(sol, grd, sol.ndim)
    CHECK_SAME_SHAPE(sol, wgt, sol.ndim - 1)

    const bool     use_32bits = CANUSE32BITS(sol) && CANUSE32BITS(hes) &&
                                CANUSE32BITS(grd) && CANUSE32BITS(wgt);
    const auto     code = static_cast<DLDataTypeCode>(sol.dtype.code);
    const auto     bits = sol.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);
    const bound::BoundVec bvec(bnd);

#define RLS_RX_ARGS bvec, static_cast<int64_t>(nbatch), VOIDPTR(sol), CVOIDPTR(hes), \
                CVOIDPTR(grd), CVOIDPTR(wgt), voxel_size, absolute, membrane,   \
                shears, div, nb_iter, sol.shape, sol.strides, hes.strides,      \
                grd.strides, wgt.strides
    NDIM_SWITCH(RLS_RX_DT)
#undef RLS_RX_ARGS
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
