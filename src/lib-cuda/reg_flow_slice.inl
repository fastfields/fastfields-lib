/**
 * The instantiating half of `reg_flow`: the internal wrapper templates, the
 * dtype x offset x boundary dispatch they sit under, and the slice functions
 * `reg_flow.cpp` calls. See `reg_flow_slice.h` for why the seam exists.
 *
 * This file is included exactly once per slice translation unit, which selects
 * what it owns by defining a subset of these before the include:
 *
 *     FF_FLOW_SLICE_ND1 / _ND2 / _ND3     which spatial ranks   (default: none)
 *     FF_FLOW_SLICE_MATVEC                the matvec family     (default: off)
 *     FF_FLOW_SLICE_DIAG                  the diag family       (default: off)
 *     FF_FLOW_SLICE_KERNEL                the stencil family    (default: off)
 *     FF_FLOW_SLICE_RELAX                 the relaxation sweeps (default: off)
 *     FF_FLOW_SLICE_OP_SET / _OP_ADD / _OP_SUB   which `op` arms (default: all)
 *
 * The cross product of what is selected is what this TU instantiates, and a
 * template nobody in the TU calls costs nothing. Every slice function declared
 * in `reg_flow_slice.h` must be defined by exactly one TU in MODULES -- too
 * few and the library link fails on an undefined symbol, too many and it fails
 * on a duplicate. Both failures are at link time, which is the point: the
 * Makefile's MODULES list and this selection cannot silently disagree.
 *
 * `.inl` and not `.h` deliberately -- it defines functions and is included
 * once, the same arrangement `impl/kernels/threadpool.inl` uses. nvcc's -MMD
 * records it as a prerequisite of every slice object, so editing it rebuilds
 * all of them.
 */

#include <stdexcept>
#include <string>
#include <cstdint>
#include <fastfields/api/cuda/posdef.h>
#include <fastfields/core/autocast.h>
#include <fastfields/core/dispatch.h>
#include <fastfields/api/cuda/stream.h>
#include <fastfields/core/dlpack.h>
#include <fastfields/core/cuda_switch.h>
#include <fastfields/impl/kernels/bounds.h>
#include <fastfields/impl/kernels/utils.h>
#include <fastfields/impl/cuda/reg_flow.h>
#include "reg_flow_slice.h"

#ifndef FF_FLOW_SLICE_ND1
#  define FF_FLOW_SLICE_ND1 0
#endif
#ifndef FF_FLOW_SLICE_ND2
#  define FF_FLOW_SLICE_ND2 0
#endif
#ifndef FF_FLOW_SLICE_ND3
#  define FF_FLOW_SLICE_ND3 0
#endif
#ifndef FF_FLOW_SLICE_MATVEC
#  define FF_FLOW_SLICE_MATVEC 0
#endif
#ifndef FF_FLOW_SLICE_DIAG
#  define FF_FLOW_SLICE_DIAG 0
#endif
#ifndef FF_FLOW_SLICE_KERNEL
#  define FF_FLOW_SLICE_KERNEL 0
#endif
#ifndef FF_FLOW_SLICE_RELAX
#  define FF_FLOW_SLICE_RELAX 0
#endif
#ifndef FF_FLOW_SLICE_OP_SET
#  define FF_FLOW_SLICE_OP_SET 1
#endif
#ifndef FF_FLOW_SLICE_OP_ADD
#  define FF_FLOW_SLICE_OP_ADD 1
#endif
#ifndef FF_FLOW_SLICE_OP_SUB
#  define FF_FLOW_SLICE_OP_SUB 1
#endif

#if !(FF_FLOW_SLICE_ND1 || FF_FLOW_SLICE_ND2 || FF_FLOW_SLICE_ND3)
#  error "a reg_flow slice must select at least one of FF_FLOW_SLICE_ND1/2/3"
#endif
#if !(FF_FLOW_SLICE_MATVEC || FF_FLOW_SLICE_DIAG || \
      FF_FLOW_SLICE_KERNEL || FF_FLOW_SLICE_RELAX)
#  error "a reg_flow slice must select at least one operation family"
#endif

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)

// reduction / accumulation type (matches jitfields' float64 default)
typedef double reduce_t;

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

    // The linear-elastic (Lamé) terms `shears`/`div` couple the flow channels,
    // so any non-zero one selects the full combined stencil (matvec_all, which
    // also folds in absolute/membrane/bending). Otherwise fall back to the
    // cheaper single-penalty stencils (highest-order non-zero wins).
    if (shears != 0.0 || div != 0.0)
        reg_flow::matvec_all<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx,
            absolute, membrane, bending, shears, div, stream);
    else if (bending != 0.0)
        reg_flow::matvec_bending<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute, membrane, bending, stream);
    else if (membrane != 0.0)
        reg_flow::matvec_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute, membrane, stream);
    else
        reg_flow::matvec_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute, stream);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_inp);
}

// Accumulate variant of _flow_matvec: out += L(inp) (op='+') or out -= L(inp)
// (op='-'), instead of overwriting out. Mirrors the CPU `_flow_matvec_acc`.
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

    if (shears != 0.0 || div != 0.0)
        reg_flow::matvec_all<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx,
            absolute, membrane, bending, shears, div, stream);
    else if (bending != 0.0)
        reg_flow::matvec_bending<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute, membrane, bending, stream);
    else if (membrane != 0.0)
        reg_flow::matvec_membrane<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute, membrane, stream);
    else
        reg_flow::matvec_absolute<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, absolute, stream);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_inp);
}

template <int ndim, char op, typename scalar_t, typename offset_t, bound::type... BOUND>
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
    const int64_t * stride_out ,
          cudaStream_t stream  )
{
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nall1);
          scalar_t * _out = static_cast<scalar_t *>(out);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    if (shears != 0.0 || div != 0.0)
        reg_flow::diag_all<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane, bending, shears, div, stream);
    else if (bending != 0.0)
        reg_flow::diag_bending<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane, bending, stream);
    else if (membrane != 0.0)
        reg_flow::diag_membrane<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane, stream);
    else
        reg_flow::diag_absolute<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, stream);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
}

// Materialise the Toeplitz convolution kernel (stencil) of the operator (see
// cpu-lib). `nfull` is the length of the size/stride arrays (== out.ndim):
// nbatch+ndim+1 for the per-channel vector stencil, nbatch+ndim+2 for the Lamé
// (cross-channel) matrix stencil.
template <int ndim, char op, typename scalar_t, typename offset_t, bound::type... BOUND>
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
          int64_t   nfull      ,
          cudaStream_t stream  )
{
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nfull);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nfull);
          scalar_t * _out = static_cast<scalar_t *>(out);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    if (shears != 0.0 || div != 0.0) {
        if (bending != 0.0)
            reg_flow::kernel_all<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _out,
                _size, _stride_out, vx, absolute, membrane, bending, shears, div, stream);
        else
            reg_flow::kernel_lame<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _out,
                _size, _stride_out, vx, absolute, membrane, shears, div, stream);
    } else if (bending != 0.0)
        reg_flow::kernel_bending<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane, bending, stream);
    else if (membrane != 0.0)
        reg_flow::kernel_membrane<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, membrane, stream);
    else
        reg_flow::kernel_absolute<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, absolute, stream);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
}

// In-place relaxation sweeps solving `(H + L) x = g` (see cpu-lib).
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
    const int64_t * stride_grd ,
          cudaStream_t stream   )
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
                absolute, membrane, bending, shears, div, niter, stream);
        else
            reg_flow::relax_lame_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd,
                _size, _stride_sol, _stride_hes, _stride_grd, vx,
                absolute, membrane, shears, div, niter, stream);
    } else if (bending != 0.0)
        reg_flow::relax_bending_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd,
            _size, _stride_sol, _stride_hes, _stride_grd, vx,
            absolute, membrane, bending, niter, stream);
    else
        reg_flow::relax_membrane_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
            bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd,
            _size, _stride_sol, _stride_hes, _stride_grd, vx,
            absolute, membrane, niter, stream);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_sol);
    free_if_needed<int64_t *>(_stride_hes);
    free_if_needed<int64_t *>(_stride_grd);
}

} // anonymous namespace

/***********************************************************************
 *                            DISPATCH                                 *
 ***********************************************************************/

// Unchanged from the single-TU form, except that the ndim arm is now the
// slice's own compile-time constant instead of a `switch (ndim)`: the switch
// moved up into reg_flow.cpp, where it picks a slice rather than a template
// argument. Everything below one arm of that switch is what a slice TU is.

#define FF_FLOW_BND1(B) B
#define FF_FLOW_BND2(B) B, B
#define FF_FLOW_BND3(B) B, B, B

// matvec dtype x offset dispatch, given ndim and the (repeated) bound pack.
#define FF_FLOW_MV_DT(NDIM, BNDS...)                                    \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_matvec<NDIM, float,  int32_t, BNDS>(FF_FLOW_MV_ARGS)  \
                : _flow_matvec<NDIM, float,  int64_t, BNDS>(FF_FLOW_MV_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_matvec<NDIM, double, int32_t, BNDS>(FF_FLOW_MV_ARGS)  \
                : _flow_matvec<NDIM, double, int64_t, BNDS>(FF_FLOW_MV_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define FF_FLOW_ADD_MV_DT(NDIM, BNDS...)                                \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_matvec_acc<NDIM, '+', float,  int32_t, BNDS>(FF_FLOW_MV_ARGS) \
                : _flow_matvec_acc<NDIM, '+', float,  int64_t, BNDS>(FF_FLOW_MV_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_matvec_acc<NDIM, '+', double, int32_t, BNDS>(FF_FLOW_MV_ARGS) \
                : _flow_matvec_acc<NDIM, '+', double, int64_t, BNDS>(FF_FLOW_MV_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define FF_FLOW_SUB_MV_DT(NDIM, BNDS...)                                \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_matvec_acc<NDIM, '-', float,  int32_t, BNDS>(FF_FLOW_MV_ARGS) \
                : _flow_matvec_acc<NDIM, '-', float,  int64_t, BNDS>(FF_FLOW_MV_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_matvec_acc<NDIM, '-', double, int32_t, BNDS>(FF_FLOW_MV_ARGS) \
                : _flow_matvec_acc<NDIM, '-', double, int64_t, BNDS>(FF_FLOW_MV_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define FF_FLOW_DG_DT(NDIM, BNDS...)                                    \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_diag<NDIM, '=', float , int32_t, BNDS>(FF_FLOW_DG_ARGS) \
                : _flow_diag<NDIM, '=', float , int64_t, BNDS>(FF_FLOW_DG_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_diag<NDIM, '=', double, int32_t, BNDS>(FF_FLOW_DG_ARGS) \
                : _flow_diag<NDIM, '=', double, int64_t, BNDS>(FF_FLOW_DG_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define FF_FLOW_ADD_DG_DT(NDIM, BNDS...)                                \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_diag<NDIM, '+', float , int32_t, BNDS>(FF_FLOW_DG_ARGS) \
                : _flow_diag<NDIM, '+', float , int64_t, BNDS>(FF_FLOW_DG_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_diag<NDIM, '+', double, int32_t, BNDS>(FF_FLOW_DG_ARGS) \
                : _flow_diag<NDIM, '+', double, int64_t, BNDS>(FF_FLOW_DG_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define FF_FLOW_SUB_DG_DT(NDIM, BNDS...)                                \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_diag<NDIM, '-', float , int32_t, BNDS>(FF_FLOW_DG_ARGS) \
                : _flow_diag<NDIM, '-', float , int64_t, BNDS>(FF_FLOW_DG_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_diag<NDIM, '-', double, int32_t, BNDS>(FF_FLOW_DG_ARGS) \
                : _flow_diag<NDIM, '-', double, int64_t, BNDS>(FF_FLOW_DG_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define FF_FLOW_KN_DT(NDIM, BNDS...)                                    \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_kernel<NDIM, '=', float , int32_t, BNDS>(FF_FLOW_KN_ARGS) \
                : _flow_kernel<NDIM, '=', float , int64_t, BNDS>(FF_FLOW_KN_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_kernel<NDIM, '=', double, int32_t, BNDS>(FF_FLOW_KN_ARGS) \
                : _flow_kernel<NDIM, '=', double, int64_t, BNDS>(FF_FLOW_KN_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define FF_FLOW_ADD_KN_DT(NDIM, BNDS...)                                \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_kernel<NDIM, '+', float , int32_t, BNDS>(FF_FLOW_KN_ARGS) \
                : _flow_kernel<NDIM, '+', float , int64_t, BNDS>(FF_FLOW_KN_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_kernel<NDIM, '+', double, int32_t, BNDS>(FF_FLOW_KN_ARGS) \
                : _flow_kernel<NDIM, '+', double, int64_t, BNDS>(FF_FLOW_KN_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define FF_FLOW_SUB_KN_DT(NDIM, BNDS...)                                \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_kernel<NDIM, '-', float , int32_t, BNDS>(FF_FLOW_KN_ARGS) \
                : _flow_kernel<NDIM, '-', float , int64_t, BNDS>(FF_FLOW_KN_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_kernel<NDIM, '-', double, int32_t, BNDS>(FF_FLOW_KN_ARGS) \
                : _flow_kernel<NDIM, '-', double, int64_t, BNDS>(FF_FLOW_KN_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define FF_FLOW_RX_DT(NDIM, BNDS...)                                    \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _flow_relax<NDIM, float,  int32_t, BNDS>(FF_FLOW_RX_ARGS)  \
                : _flow_relax<NDIM, float,  int64_t, BNDS>(FF_FLOW_RX_ARGS); \
            case 64: return use_32bits                                  \
                ? _flow_relax<NDIM, double, int32_t, BNDS>(FF_FLOW_RX_ARGS)  \
                : _flow_relax<NDIM, double, int64_t, BNDS>(FF_FLOW_RX_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

// Which of these boundary conditions gets a dedicated (static) instantiation
// and which shares the single Dynamic (runtime) one is a build-time choice --
// see FF_STATIC_BOUND_* in kernels/bounds.h. `bvec` carries the runtime
// condition for whichever ones fall back to Dynamic.
#define FF_FLOW_BOUND_SWITCH(DT, NDIM, BND)                             \
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

/***********************************************************************
 *                          SLICE DEFINITIONS                          *
 ***********************************************************************/

// The prologues below are what each exported entry point used to compute
// inline, minus the argument checks (those stayed in reg_flow.cpp, where the
// error messages and their order are unchanged) and minus `nbatch` (passed in,
// so the rank arithmetic still happens exactly once).

#define FF_FLOW_MV_ARGS                                                       \
    bvec, static_cast<int64_t>(nbatch), FF_VOIDPTR(out), FF_CVOIDPTR(inp),    \
    voxel_size, absolute, membrane, bending, shears, div,                     \
    out.shape, out.strides, inp.strides, cstream

#define FF_FLOW_DG_ARGS                                                       \
    bvec, static_cast<int64_t>(nbatch), FF_VOIDPTR(out),                      \
    voxel_size, absolute, membrane, bending, shears, div,                     \
    out.shape, out.strides, cstream

#define FF_FLOW_KN_ARGS                                                       \
    bvec, static_cast<int64_t>(nbatch), FF_VOIDPTR(out),                      \
    voxel_size, absolute, membrane, bending, shears, div,                     \
    out.shape, out.strides, static_cast<int64_t>(out.ndim), cstream

#define FF_FLOW_RX_ARGS                                                       \
    bvec, static_cast<int64_t>(nbatch), FF_VOIDPTR(sol), FF_CVOIDPTR(hes),    \
    FF_CVOIDPTR(grd), voxel_size, absolute, membrane, bending,                \
    shears, div, nb_iter, sol.shape, sol.strides, hes.strides,                \
    grd.strides, cstream

#define FF_FLOW_PROLOGUE(T)                                                   \
    const auto code = static_cast<DLDataTypeCode>(T.dtype.code);              \
    const auto bits = T.dtype.bits;                                           \
    const bound::type bnd = static_cast<bound::type>(bound);                  \
    const bound::BoundVec bvec(bnd);                                          \
    const cudaStream_t cstream = _reg_stream(stream);

FF_NAMESPACE_BEGIN(flow_slice)

// Each of these is one arm of the old `NDIM_SWITCH`, given its own name so it
// can be given its own translation unit.
#define FF_FLOW_DEFINE_SLICE(NAME, SIG, PROLOGUE_T, DT, ND)                   \
    FF_FLOW_SLICE_HIDDEN SIG(NAME)                                            \
    {                                                                         \
        FF_FLOW_PROLOGUE(PROLOGUE_T)                                          \
        FF_FLOW_BOUND_SWITCH(DT, ND, FF_FLOW_BND##ND)                         \
    }

#if FF_FLOW_SLICE_MATVEC && FF_FLOW_SLICE_OP_SET
#  define FF_FLOW_EMIT_MATVEC_SET(ND) \
     FF_FLOW_DEFINE_SLICE(matvec_##ND##d, FF_FLOW_SLICE_SIG_MATVEC, out, FF_FLOW_MV_DT, ND)
#else
#  define FF_FLOW_EMIT_MATVEC_SET(ND)
#endif
#if FF_FLOW_SLICE_MATVEC && FF_FLOW_SLICE_OP_ADD
#  define FF_FLOW_EMIT_MATVEC_ADD(ND) \
     FF_FLOW_DEFINE_SLICE(addmatvec_##ND##d, FF_FLOW_SLICE_SIG_MATVEC, out, FF_FLOW_ADD_MV_DT, ND)
#else
#  define FF_FLOW_EMIT_MATVEC_ADD(ND)
#endif
#if FF_FLOW_SLICE_MATVEC && FF_FLOW_SLICE_OP_SUB
#  define FF_FLOW_EMIT_MATVEC_SUB(ND) \
     FF_FLOW_DEFINE_SLICE(submatvec_##ND##d, FF_FLOW_SLICE_SIG_MATVEC, out, FF_FLOW_SUB_MV_DT, ND)
#else
#  define FF_FLOW_EMIT_MATVEC_SUB(ND)
#endif

#if FF_FLOW_SLICE_DIAG && FF_FLOW_SLICE_OP_SET
#  define FF_FLOW_EMIT_DIAG_SET(ND) \
     FF_FLOW_DEFINE_SLICE(diag_##ND##d, FF_FLOW_SLICE_SIG_DIAG, out, FF_FLOW_DG_DT, ND)
#else
#  define FF_FLOW_EMIT_DIAG_SET(ND)
#endif
#if FF_FLOW_SLICE_DIAG && FF_FLOW_SLICE_OP_ADD
#  define FF_FLOW_EMIT_DIAG_ADD(ND) \
     FF_FLOW_DEFINE_SLICE(adddiag_##ND##d, FF_FLOW_SLICE_SIG_DIAG, out, FF_FLOW_ADD_DG_DT, ND)
#else
#  define FF_FLOW_EMIT_DIAG_ADD(ND)
#endif
#if FF_FLOW_SLICE_DIAG && FF_FLOW_SLICE_OP_SUB
#  define FF_FLOW_EMIT_DIAG_SUB(ND) \
     FF_FLOW_DEFINE_SLICE(subdiag_##ND##d, FF_FLOW_SLICE_SIG_DIAG, out, FF_FLOW_SUB_DG_DT, ND)
#else
#  define FF_FLOW_EMIT_DIAG_SUB(ND)
#endif

#if FF_FLOW_SLICE_KERNEL && FF_FLOW_SLICE_OP_SET
#  define FF_FLOW_EMIT_KERNEL_SET(ND) \
     FF_FLOW_DEFINE_SLICE(kernel_##ND##d, FF_FLOW_SLICE_SIG_KERNEL, out, FF_FLOW_KN_DT, ND)
#else
#  define FF_FLOW_EMIT_KERNEL_SET(ND)
#endif
#if FF_FLOW_SLICE_KERNEL && FF_FLOW_SLICE_OP_ADD
#  define FF_FLOW_EMIT_KERNEL_ADD(ND) \
     FF_FLOW_DEFINE_SLICE(addkernel_##ND##d, FF_FLOW_SLICE_SIG_KERNEL, out, FF_FLOW_ADD_KN_DT, ND)
#else
#  define FF_FLOW_EMIT_KERNEL_ADD(ND)
#endif
#if FF_FLOW_SLICE_KERNEL && FF_FLOW_SLICE_OP_SUB
#  define FF_FLOW_EMIT_KERNEL_SUB(ND) \
     FF_FLOW_DEFINE_SLICE(subkernel_##ND##d, FF_FLOW_SLICE_SIG_KERNEL, out, FF_FLOW_SUB_KN_DT, ND)
#else
#  define FF_FLOW_EMIT_KERNEL_SUB(ND)
#endif

// `relax` has no op axis -- it solves in place, there is nothing to accumulate
// into -- so FF_FLOW_SLICE_OP_* does not apply to it.
#if FF_FLOW_SLICE_RELAX
#  define FF_FLOW_EMIT_RELAX(ND) \
     FF_FLOW_DEFINE_SLICE(relax_##ND##d, FF_FLOW_SLICE_SIG_RELAX, sol, FF_FLOW_RX_DT, ND)
#else
#  define FF_FLOW_EMIT_RELAX(ND)
#endif

#define FF_FLOW_EMIT_ND(ND)                                                   \
    FF_FLOW_EMIT_MATVEC_SET(ND)                                               \
    FF_FLOW_EMIT_MATVEC_ADD(ND)                                               \
    FF_FLOW_EMIT_MATVEC_SUB(ND)                                               \
    FF_FLOW_EMIT_DIAG_SET(ND)                                                 \
    FF_FLOW_EMIT_DIAG_ADD(ND)                                                 \
    FF_FLOW_EMIT_DIAG_SUB(ND)                                                 \
    FF_FLOW_EMIT_KERNEL_SET(ND)                                               \
    FF_FLOW_EMIT_KERNEL_ADD(ND)                                               \
    FF_FLOW_EMIT_KERNEL_SUB(ND)                                               \
    FF_FLOW_EMIT_RELAX(ND)

#if FF_FLOW_SLICE_ND1
FF_FLOW_EMIT_ND(1)
#endif
#if FF_FLOW_SLICE_ND2
FF_FLOW_EMIT_ND(2)
#endif
#if FF_FLOW_SLICE_ND3
FF_FLOW_EMIT_ND(3)
#endif

FF_NAMESPACE_END(flow_slice)

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
