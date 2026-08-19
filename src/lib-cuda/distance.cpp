#include <stdexcept>
#include <cstdint>
#include "fastfields/api/cuda/distance.h"
#include "fastfields/core/autocast.h"
#include "fastfields/core/dispatch.h"
#include "fastfields/core/dlpack.h"
#include "fastfields/core/cuda_switch.h"
#include "fastfields/impl/kernels/utils.h"
#include "fastfields/impl/cuda/distance_euclidean.h"
#include "fastfields/impl/cuda/distance_l1.h"
#include "fastfields/impl/cuda/distance_spline.h"
#include "fastfields/impl/cuda/distance_mesh.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

/***********************************************************************
 *                              EUCLIDEAN                              *
 ***********************************************************************/

#define DISPATCH_DT(func, args...)                                     \
{                                                                      \
    const bool use_32bits = FF_CANUSE32BITS(inp_out);                  \
    const auto code = static_cast<DLDataTypeCode>(inp_out.dtype.code); \
    switch (code) {                                                    \
        case kDLFloat: switch (inp_out.dtype.bits) {                   \
            case 32: return (                                          \
                use_32bits                                             \
                ? func<float,int32_t>(args)                            \
                : func<float,int64_t>(args)                            \
            );                                                         \
            case 64: return (                                          \
                use_32bits                                             \
                ? func<double,int32_t>(args)                           \
                : func<double,int64_t>(args)                           \
            );                                                         \
            default: break;                                            \
        };                                                             \
        default: throw std::invalid_argument(                          \
            "only floating point data types are supported"             \
        );                                                             \
    };                                                                 \
}

namespace {
template <typename scalar_t = float, typename offset_t = int64_t>
inline void _dt_euclidean(
          int64_t   ndim      ,     // number of dimensions
          void    * f         ,     // pointer to data [*batch, n]
          double    w         ,     // pixel spacing
    const int64_t * size      ,     // [ndim] data shape   == (*batch, n)
    const int64_t * stride    ,     // [ndim] data strides
          intptr_t  stream    )     // CUDA stream (0 == default stream)
{
    const offset_t * _size   = copy_if_needed<offset_t *>(size,   ndim);
    const offset_t * _stride = copy_if_needed<offset_t *>(stride, ndim);
          scalar_t * _f      = static_cast<scalar_t *>(f);
    const offset_t   _ndim   = static_cast<offset_t  >(ndim);
    const scalar_t   _w      = static_cast<scalar_t  >(w);
    distance_e::dt(_ndim, _f, _w, _size, _stride, stream);
    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride);
}
}

void dt_euclidean(
          DLTensor & inp_out_,
          double     voxel_spacing,
          intptr_t   stream
)
{
    // Normalise a NULL strides field (compact row-major) before dispatch.
    ContiguousStrides _io(inp_out_);
    DLTensor & inp_out = _io.t;

    FF_CHECK_NO_LANES(inp_out)
    DISPATCH_DT(
        _dt_euclidean,
        inp_out.ndim,
        FF_VOIDPTR(inp_out),
        voxel_spacing,
        inp_out.shape,
        inp_out.strides,
        stream
    )
}

namespace {
template <typename scalar_t = float, typename offset_t = int64_t>
inline void _dt_l1(
          int64_t   ndim      ,     // number of dimensions
          void    * f         ,     // pointer to data [*batch, n]
          double    w         ,     // pixel spacing
    const int64_t * size      ,     // [ndim] data shape   == (*batch, n)
    const int64_t * stride    ,     // [ndim] data strides
          intptr_t  stream    )     // CUDA stream (0 == default stream)
{
    const offset_t * _size   = copy_if_needed<offset_t *>(size,   ndim);
    const offset_t * _stride = copy_if_needed<offset_t *>(stride, ndim);
          scalar_t * _f      = static_cast<scalar_t *>(f);
    const offset_t   _ndim   = static_cast<offset_t  >(ndim);
    const scalar_t   _w      = static_cast<scalar_t  >(w);
    distance_l1::dt(_ndim, _f, _w, _size, _stride, stream);
    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride);
}
}

void dt_l1(
          DLTensor & inp_out_,
          double     voxel_spacing,
          intptr_t   stream
)
{
    // Normalise a NULL strides field (compact row-major) before dispatch.
    ContiguousStrides _io(inp_out_);
    DLTensor & inp_out = _io.t;

    FF_CHECK_NO_LANES(inp_out)
    DISPATCH_DT(
        _dt_l1,
        inp_out.ndim,
        FF_VOIDPTR(inp_out),
        voxel_spacing,
        inp_out.shape,
        inp_out.strides,
        stream
    )
}

/***********************************************************************
 *                              SPLINE                                 *
 ***********************************************************************/

#define DISPATCH_SPLINE_DIM(D, func, args...)                           \
    switch (code) {                                                     \
        case kDLFloat: switch (loc.dtype.bits) {                        \
            case 32: return (                                           \
                use_32bits                                              \
                ? func<D,float,int32_t>(args)                           \
                : func<D,float,int64_t>(args)                           \
            );                                                          \
            case 64: return (                                           \
                use_32bits                                              \
                ? func<D,double,int32_t>(args)                          \
                : func<D,double,int64_t>(args)                          \
            );                                                          \
            default: break;                                             \
        };                                                              \
        default: throw std::invalid_argument(                           \
            "Only floating point data types are supported"              \
        );                                                              \
    }

#define DISPATCH_SPLINE(func, args...)                                  \
{                                                                       \
    const auto ndim = loc.shape[loc.ndim-1];                            \
    const auto code = static_cast<DLDataTypeCode>(loc.dtype.code);      \
    switch (ndim) {                                                     \
        case  1: DISPATCH_SPLINE_DIM(1, func, args);                    \
        case  2: DISPATCH_SPLINE_DIM(2, func, args);                    \
        case  3: DISPATCH_SPLINE_DIM(3, func, args);                    \
        default: throw std::invalid_argument(                           \
            "Only 1D, 2D and 3D splines are supported"                  \
        );                                                              \
    };                                                                  \
}

namespace {
template <int ndim, typename scalar_t, typename offset_t>
inline void _dt_spline_table(
          int64_t   nbatch         ,   // Number of batch dimensions
          void    * time           ,   // (*batch) tensor -> Best time
          void    * dist           ,   // (*batch) tensor -> Best sqdist
    const void    * loc            ,   // (*batch, ndim) tensor -> ND location of each point
    const void    * coeff          ,   // (*batch, npoints, ndim) tensor -> Spline coefficients
    const void    * times          ,   // (*batch) tensor -> Time values to try
          int64_t   ntimes         ,   // Number of times values to try
    const int64_t * size           ,   // [*batch, npoints, ndim] list -> Coeff shape
    const int64_t * int64_time    ,   // [*batch] list -> Strides of `time`
    const int64_t * stride_dist    ,   // [*batch] list -> Strides of `dist`
    const int64_t * stride_loc     ,   // [*batch, ndim] list -> Strides of `loc`
    const int64_t * stride_coeff   ,   // [*batch, npoints, ndim] list -> Strides or `coeff`
    const int64_t * int64_times   ,   // [*batch, ntimes] list -> Strides of `times`
          int8_t    spline         ,   // Spline order
          int8_t    bound          )   // Boundary condition
{
    // Array lengths follow how the impl (distance_spline::mindist_table)
    // indexes each buffer:
    //   size          -> coeff shape (*batch, npoints, ndim)  == nbatch+2
    //                    (reads size[nbatch] as npoints)
    //   stride_time   -> time  (*batch)                       == nbatch
    //   stride_dist   -> dist  (*batch)                       == nbatch
    //   stride_loc    -> loc   (*batch, ndim)                 == nbatch+1
    //   stride_coeff  -> coeff (*batch, npoints, ndim)        == nbatch+2
    //   stride_times  -> times (*batch, ntimes)               == nbatch+1
    const offset_t * _size        = copy_if_needed<offset_t *>(size,         nbatch+2);
    const offset_t * _int64_time = copy_if_needed<offset_t *>(int64_time,  nbatch);
    const offset_t * _stride_dist = copy_if_needed<offset_t *>(stride_dist,  nbatch);
    const offset_t * _stride_loc  = copy_if_needed<offset_t *>(stride_loc,   nbatch+1);
    const offset_t * _stride_coeff= copy_if_needed<offset_t *>(stride_coeff, nbatch+2);
    const offset_t * _int64_times= copy_if_needed<offset_t *>(int64_times, nbatch+1);
          scalar_t * _time        = static_cast<      scalar_t *>(time);
          scalar_t * _dist        = static_cast<      scalar_t *>(dist);
    const scalar_t * _loc         = static_cast<const scalar_t *>(loc);
    const scalar_t * _coeff       = static_cast<const scalar_t *>(coeff);
    const scalar_t * _times       = static_cast<const scalar_t *>(times);
    const offset_t   _nbatch      = static_cast<      offset_t  >(nbatch);
    const spline_t   _spline      = static_cast<      spline_t  >(spline);
    const bound_t    _bound       = static_cast<      bound_t   >(bound);
    const offset_t   _ntimes      = static_cast<      offset_t  >(ntimes);
    distance_spline::mindist_table<
        ndim, spline_t::Dynamic, bound_t::Dynamic, scalar_t, offset_t>
    (
        _nbatch, _time, _dist, _loc, _coeff, _times, _ntimes,
        _size, _int64_time, _stride_dist, _stride_loc, _stride_coeff, _int64_times,
        _spline, _bound
    );
    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_int64_time);
    free_if_needed<int64_t *>(_stride_dist);
    free_if_needed<int64_t *>(_stride_loc);
    free_if_needed<int64_t *>(_stride_coeff);
    free_if_needed<int64_t *>(_int64_times);
}
}

void dt_spline_table(
          DLTensor & time_,
          DLTensor & dist_,
    const DLTensor & loc_,
    const DLTensor & coeff_,
    const DLTensor & times_,
          int8_t     spline,
          int8_t     bound,
          intptr_t   /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _tm(time_), _di(dist_), _lo(loc_), _co(coeff_), _ti(times_);
    DLTensor       & time  = _tm.t;
    DLTensor       & dist  = _di.t;
    const DLTensor & loc   = _lo.t;
    const DLTensor & coeff = _co.t;
    const DLTensor & times = _ti.t;

    const bool use_32bits = (
        FF_CANUSE32BITS(time)  &&
        FF_CANUSE32BITS(dist)  &&
        FF_CANUSE32BITS(loc)   &&
        FF_CANUSE32BITS(coeff) &&
        FF_CANUSE32BITS(times)
    );
    const int32_t ndim   = loc.shape[loc.ndim-1];
    const int32_t nbatch = loc.ndim - 1;
    FF_CHECK_NO_LANES  (time)
    FF_CHECK_SAME_DTYPE(time, dist)
    FF_CHECK_SAME_DTYPE(time, loc)
    FF_CHECK_SAME_DTYPE(time, coeff)
    FF_CHECK_SAME_DTYPE(time, times)
    FF_CHECK_SAME      (time.ndim,  nbatch,   "Number of batch dimensions does not match")
    FF_CHECK_SAME      (dist.ndim,  nbatch,   "Number of batch dimensions does not match")
    // coeff is (*batch, npoints, ndim) -> nbatch+2 dims
    // times is (*batch, ntimes)        -> nbatch+1 dims
    FF_CHECK_SAME      (coeff.ndim, nbatch+2, "Number of coeff dimensions does not match")
    FF_CHECK_SAME      (times.ndim, nbatch+1, "Number of times dimensions does not match")
    FF_CHECK_SAME      (coeff.shape[coeff.ndim-1], ndim, "Dimensionality of coeff and location does not match")
    FF_CHECK_SAME_BATCH_ND(loc, time,  nbatch)
    FF_CHECK_SAME_BATCH_ND(loc, dist,  nbatch)
    FF_CHECK_SAME_BATCH_ND(loc, coeff, nbatch)
    FF_CHECK_SAME_BATCH_ND(loc, times, nbatch)

    DISPATCH_SPLINE(
        _dt_spline_table,
        nbatch,                             // nbatch
        FF_VOIDPTR(time),                      // time
        FF_VOIDPTR(dist),                      // dist
        FF_VOIDPTR(loc),                       // loc
        FF_VOIDPTR(coeff),                     // coeff
        FF_VOIDPTR(times),                     // times
        times.shape[times.ndim-1],          // ntimes
        coeff.shape,                        // size (coeff shape: *batch, npoints, ndim)
        time.strides,                       // int64_time
        dist.strides,                       // stride_dist
        loc.strides,                        // stride_loc
        coeff.strides,                      // stride_coeff
        times.strides,                      // int64_times
        spline,
        bound
    )
}

namespace {
template <int ndim, typename scalar_t, typename offset_t>
inline void _dt_spline_brent(
    int64_t   nbatch        , // Number of batch dimensions
    void    * time          , // (*batch) tensor -> Best time
    void    * dist          , // (*batch) tensor -> Best sqdist
    void    * loc           , // (*batch, ndim) tensor -> ND location of each point
    void    * coeff         , // (*batch, npoints, ndim) tensor -> Spline coefficients
    int64_t * size          , // [*batch, npoints, ndim] list -> Coeff shape
    int64_t * int64_time   , // [*batch] list -> Strides of `time`
    int64_t * stride_dist   , // [*batch] list -> Strides of `dist`
    int64_t * stride_loc    , // [*batch, ndim] list -> Strides of `loc`
    int64_t * stride_coeff  , // [*batch, npoints, ndim] list -> Strides or `coeff`
    int64_t   max_iter      ,
    double    tol           ,
    double    step          ,
    int8_t    spline        ,
    int8_t    bound         )
{
    // Array lengths follow how the impl indexes each buffer:
    //   size         -> coeff shape (*batch, npoints, ndim) == nbatch+2
    //                   (reads size[nbatch] as npoints)
    //   stride_time  -> time  (*batch)                      == nbatch
    //   stride_dist  -> dist  (*batch)                      == nbatch
    //   stride_loc   -> loc   (*batch, ndim)                == nbatch+1
    //   stride_coeff -> coeff (*batch, npoints, ndim)       == nbatch+2
    const offset_t * _size        = copy_if_needed<offset_t *>(size,         nbatch+2);
    const offset_t * _int64_time = copy_if_needed<offset_t *>(int64_time,  nbatch);
    const offset_t * _stride_dist = copy_if_needed<offset_t *>(stride_dist,  nbatch);
    const offset_t * _stride_loc  = copy_if_needed<offset_t *>(stride_loc,   nbatch+1);
    const offset_t * _stride_coeff= copy_if_needed<offset_t *>(stride_coeff, nbatch+2);
          scalar_t * _time        = static_cast<      scalar_t *>(time);
          scalar_t * _dist        = static_cast<      scalar_t *>(dist);
    const scalar_t * _loc         = static_cast<const scalar_t *>(loc);
    const scalar_t * _coeff       = static_cast<const scalar_t *>(coeff);
    const offset_t   _nbatch      = static_cast<      offset_t  >(nbatch);
    const spline_t   _spline      = static_cast<      spline_t  >(spline);
    const bound_t    _bound       = static_cast<      bound_t   >(bound);
    const offset_t   _max_iter    = static_cast<      offset_t  >(max_iter);
    const double     _tol         = static_cast<      double    >(tol);
    const double     _step        = static_cast<      double    >(step);
    distance_spline::mindist_brent<
        ndim, spline_t::Dynamic, bound_t::Dynamic, scalar_t, offset_t>
    (
        _nbatch, _time, _dist, _loc, _coeff,
        _size, _int64_time, _stride_dist, _stride_loc, _stride_coeff,
        _max_iter, _tol, _step, _spline, _bound
    );
    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_int64_time);
    free_if_needed<int64_t *>(_stride_dist);
    free_if_needed<int64_t *>(_stride_loc);
    free_if_needed<int64_t *>(_stride_coeff);
}
}

void dt_spline_brent(
          DLTensor & time_,
          DLTensor & dist_,
    const DLTensor & loc_,
    const DLTensor & coeff_,
          int64_t    max_iter,
          double     tol,
          double     step,
          int8_t     spline,
          int8_t     bound,
          intptr_t   /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _tm(time_), _di(dist_), _lo(loc_), _co(coeff_);
    DLTensor       & time  = _tm.t;
    DLTensor       & dist  = _di.t;
    const DLTensor & loc   = _lo.t;
    const DLTensor & coeff = _co.t;

    const bool use_32bits = (
        FF_CANUSE32BITS(time)  &&
        FF_CANUSE32BITS(dist)  &&
        FF_CANUSE32BITS(loc)   &&
        FF_CANUSE32BITS(coeff)
    );
    const int32_t ndim   = loc.shape[loc.ndim-1];
    const int32_t nbatch = loc.ndim - 1;
    FF_CHECK_NO_LANES  (time)
    FF_CHECK_SAME_DTYPE(time, dist)
    FF_CHECK_SAME_DTYPE(time, loc)
    FF_CHECK_SAME_DTYPE(time, coeff)
    FF_CHECK_SAME      (time.ndim,  nbatch,   "Number of batch dimensions does not match")
    FF_CHECK_SAME      (dist.ndim,  nbatch,   "Number of batch dimensions does not match")
    // coeff is (*batch, npoints, ndim) -> nbatch+2 dims
    FF_CHECK_SAME      (coeff.ndim, nbatch+2, "Number of coeff dimensions does not match")
    FF_CHECK_SAME      (coeff.shape[coeff.ndim-1], ndim, "Dimensionality of coeff and location does not match")
    FF_CHECK_SAME_BATCH_ND(loc, time,  nbatch)
    FF_CHECK_SAME_BATCH_ND(loc, dist,  nbatch)
    FF_CHECK_SAME_BATCH_ND(loc, coeff, nbatch)

    DISPATCH_SPLINE(
        _dt_spline_brent,
        nbatch,                             // nbatch
        FF_VOIDPTR(time),                      // time
        FF_VOIDPTR(dist),                      // dist
        FF_VOIDPTR(loc),                       // loc
        FF_VOIDPTR(coeff),                     // coeff
        coeff.shape,                        // size (coeff shape: *batch, npoints, ndim)
        time.strides,                       // int64_time
        dist.strides,                       // stride_dist
        loc.strides,                        // stride_loc
        coeff.strides,                      // stride_coeff
        max_iter,
        tol,
        step,
        spline,
        bound
    )
}

namespace {
template <int ndim, typename scalar_t, typename offset_t>
inline void _dt_spline_gaussnewton(
    int64_t   nbatch        , // Number of batch dimensions
    void    * time          , // (*batch) tensor -> Best time
    void    * dist          , // (*batch) tensor -> Best sqdist
    void    * loc           , // (*batch, ndim) tensor -> ND location of each point
    void    * coeff         , // (*batch, npoints, ndim) tensor -> Spline coefficients
    int64_t * size          , // [*batch, npoints, ndim] list -> Coeff shape
    int64_t * int64_time   , // [*batch] list -> Strides of `time`
    int64_t * stride_dist   , // [*batch] list -> Strides of `dist`
    int64_t * stride_loc    , // [*batch, ndim] list -> Strides of `loc`
    int64_t * stride_coeff  , // [*batch, npoints, ndim] list -> Strides or `coeff`
    int64_t   max_iter      ,
    double    tol           ,
    int8_t    spline        ,
    int8_t    bound         )
{
    // Array lengths follow how the impl indexes each buffer:
    //   size         -> coeff shape (*batch, npoints, ndim) == nbatch+2
    //                   (reads size[nbatch] as npoints)
    //   stride_time  -> time  (*batch)                      == nbatch
    //   stride_dist  -> dist  (*batch)                      == nbatch
    //   stride_loc   -> loc   (*batch, ndim)                == nbatch+1
    //   stride_coeff -> coeff (*batch, npoints, ndim)       == nbatch+2
    const offset_t * _size        = copy_if_needed<offset_t *>(size,         nbatch+2);
    const offset_t * _int64_time = copy_if_needed<offset_t *>(int64_time,  nbatch);
    const offset_t * _stride_dist = copy_if_needed<offset_t *>(stride_dist,  nbatch);
    const offset_t * _stride_loc  = copy_if_needed<offset_t *>(stride_loc,   nbatch+1);
    const offset_t * _stride_coeff= copy_if_needed<offset_t *>(stride_coeff, nbatch+2);
          scalar_t * _time        = static_cast<      scalar_t *>(time);
          scalar_t * _dist        = static_cast<      scalar_t *>(dist);
    const scalar_t * _loc         = static_cast<const scalar_t *>(loc);
    const scalar_t * _coeff       = static_cast<const scalar_t *>(coeff);
    const offset_t   _nbatch      = static_cast<      offset_t  >(nbatch);
    const spline_t   _spline      = static_cast<      spline_t  >(spline);
    const bound_t    _bound       = static_cast<      bound_t   >(bound);
    const offset_t   _max_iter    = static_cast<      offset_t  >(max_iter);
    const double     _tol         = static_cast<      double    >(tol);
    distance_spline::mindist_gaussnewton<
        ndim, spline_t::Dynamic, bound_t::Dynamic, scalar_t, offset_t>
    (
        _nbatch, _time, _dist, _loc, _coeff,
        _size, _int64_time, _stride_dist, _stride_loc, _stride_coeff,
        _max_iter, _tol, _spline, _bound
    );
    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_int64_time);
    free_if_needed<int64_t *>(_stride_dist);
    free_if_needed<int64_t *>(_stride_loc);
    free_if_needed<int64_t *>(_stride_coeff);
}
}

void dt_spline_gaussnewton(
          DLTensor & time_,
          DLTensor & dist_,
    const DLTensor & loc_,
    const DLTensor & coeff_,
          int64_t    max_iter,
          double     tol,
          int8_t     spline,
          int8_t     bound,
          intptr_t   /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _tm(time_), _di(dist_), _lo(loc_), _co(coeff_);
    DLTensor       & time  = _tm.t;
    DLTensor       & dist  = _di.t;
    const DLTensor & loc   = _lo.t;
    const DLTensor & coeff = _co.t;

    const bool use_32bits = (
        FF_CANUSE32BITS(time)  &&
        FF_CANUSE32BITS(dist)  &&
        FF_CANUSE32BITS(loc)   &&
        FF_CANUSE32BITS(coeff)
    );
    const int32_t ndim   = loc.shape[loc.ndim-1];
    const int32_t nbatch = loc.ndim - 1;
    FF_CHECK_NO_LANES  (time)
    FF_CHECK_SAME_DTYPE(time, dist)
    FF_CHECK_SAME_DTYPE(time, loc)
    FF_CHECK_SAME_DTYPE(time, coeff)
    FF_CHECK_SAME      (time.ndim,  nbatch,   "Number of batch dimensions does not match")
    FF_CHECK_SAME      (dist.ndim,  nbatch,   "Number of batch dimensions does not match")
    // coeff is (*batch, npoints, ndim) -> nbatch+2 dims
    FF_CHECK_SAME      (coeff.ndim, nbatch+2, "Number of coeff dimensions does not match")
    FF_CHECK_SAME      (coeff.shape[coeff.ndim-1], ndim, "Dimensionality of coeff and location does not match")
    FF_CHECK_SAME_BATCH_ND(loc, time,  nbatch)
    FF_CHECK_SAME_BATCH_ND(loc, dist,  nbatch)
    FF_CHECK_SAME_BATCH_ND(loc, coeff, nbatch)

    DISPATCH_SPLINE(
        _dt_spline_gaussnewton,
        nbatch,                             // nbatch
        FF_VOIDPTR(time),                      // time
        FF_VOIDPTR(dist),                      // dist
        FF_VOIDPTR(loc),                       // loc
        FF_VOIDPTR(coeff),                     // coeff
        coeff.shape,                        // size (coeff shape: *batch, npoints, ndim)
        time.strides,                       // int64_time
        dist.strides,                       // stride_dist
        loc.strides,                        // stride_loc
        coeff.strides,                      // stride_coeff
        max_iter,
        tol,
        spline,
        bound
    )
}

/***********************************************************************
 *                                MESH                                 *
 ***********************************************************************/

#define DISPATCH_MESH_SCALAR(D, S, O, func, args...)                       \
    switch (code_index) {                                               \
        case kDLInt: switch (loc.dtype.bits) {                          \
            case  8: return func<D,S,int8_t,  O>(args);                    \
            case 16: return func<D,S,int16_t, O>(args);                    \
            case 32: return func<D,S,int32_t, O>(args);                    \
            case 64: return func<D,S,int64_t, O>(args);                    \
            default: break;                                             \
        };                                                              \
        case kDLUInt: switch (loc.dtype.bits) {                         \
            case  8: return func<D,S,uint8_t,  O>(args);                   \
            case 16: return func<D,S,uint16_t, O>(args);                   \
            case 32: return func<D,S,uint32_t, O>(args);                   \
            case 64: return func<D,S,uint64_t, O>(args);                   \
            default: break;                                             \
        };                                                              \
        default: throw std::invalid_argument(                           \
            "Only integer data types are supported"                     \
        );                                                              \
    }

#define DISPATCH_MESH_OFFSET(D, S, func, args...)                       \
    if (use_32bits) DISPATCH_MESH_SCALAR(D, S, int32_t, func, args)     \
    else            DISPATCH_MESH_SCALAR(D, S, int64_t, func, args)     \

#define DISPATCH_MESH_DIM(D, func, args...)                             \
    switch (code_scalar) {                                              \
        case kDLFloat: switch (loc.dtype.bits) {                        \
            case 32: DISPATCH_MESH_OFFSET(D, float,  func, args);       \
            case 64: DISPATCH_MESH_OFFSET(D, double, func, args);       \
            default: break;                                             \
        };                                                              \
        default: throw std::invalid_argument(                           \
            "Only floating point data types are supported"              \
        );                                                              \
    }

#define DISPATCH_MESH(func, args...)                                    \
{                                                                       \
    const auto ndim        = loc.shape[loc.ndim-1];                     \
    const auto code_scalar = static_cast<DLDataTypeCode>(vertices.dtype.code); \
    const auto code_index  = static_cast<DLDataTypeCode>(faces.dtype.code);    \
    switch (ndim) {                                                     \
        case 2: DISPATCH_MESH_DIM(2, func, args);                       \
        case 3: DISPATCH_MESH_DIM(3, func, args);                       \
        default: throw std::invalid_argument(                           \
            "Only 1D, 2D and 3D splines are supported"                  \
        );                                                              \
    };                                                                  \
}

template <
    int      ndim,
    typename scalar_t = float,              // Value data type
    typename index_t  = int64_t,            // Index/Stride data type
    typename offset_t = int64_t             // Index/Stride data type
>
static inline void
_dt_mesh(
    int64_t   nbatch,                // Number of batch dimensions in coord
    void    * dist,                  // (*batch) tensor -> Output placeholder for distance
    void    * nearest_vertex,        // (*batch) tensor -> Output placeholder for index of nearest vertex
    void    * coord,                 // (*batch, D) tensor -> Coordinates at which to evaluate distance
    void    * vertices,              // (N, D) tensor -> All vertices
    void    * faces,                 // (M, D) tensor -> All faces (face = D vertex indices)
    int64_t * size,                  // [*batch] list -> Size of `dist`
    int64_t   nb_faces,
    int64_t   nb_vertices,
    int64_t * stride_dist,           // [*batch] list -> Strides of `dist`
    int64_t * stride_nearest,        // [*batch] list -> Strides of `nearest_vertex`
    int64_t * stride_coord,          // [*batch, D] list -> Strides of `coord`
    int64_t * stride_vertices,       // [N, D] list -> Strides of `vertices`
    int64_t * stride_faces,          // [M, D] list -> Strides of `faces`
    bool      _signed = false,
    bool      naive   = false,
    intptr_t  stream  = 0            // CUDA stream (0 == default stream)
)
{
    // vertices (N, D) and faces (M, D) are always 2D (the impl only reads
    // stride[0] and stride[1]); their length is independent of loc's batch rank.
    const offset_t * _size           = copy_if_needed<offset_t *>(size,           nbatch);
    const offset_t * _stride_dist    = copy_if_needed<offset_t *>(stride_dist,    nbatch);
    const offset_t * _stride_coord   = copy_if_needed<offset_t *>(stride_coord,   nbatch+1);
    const offset_t * _stride_vertices= copy_if_needed<offset_t *>(stride_vertices,2);
    const offset_t * _stride_faces   = copy_if_needed<offset_t *>(stride_faces,   2);
          scalar_t * _dist           = static_cast<      scalar_t *>(dist);
          index_t  * _nearest_vertex = static_cast<      index_t  *>(nearest_vertex);
    const scalar_t * _coord          = static_cast<const scalar_t *>(coord);
    const scalar_t * _vertices       = static_cast<const scalar_t *>(vertices);
    const index_t  * _faces          = static_cast<const index_t  *>(faces);
    const offset_t   _nbatch         = static_cast<      offset_t  >(nbatch);
    const offset_t   _nb_faces       = static_cast<      offset_t  >(nb_faces);
    const offset_t   _nb_vertices    = static_cast<      offset_t  >(nb_vertices);

    const offset_t * _stride_nearest = (
        stride_nearest
        ? copy_if_needed<offset_t *>(stride_nearest, nbatch)
        : nullptr
    );

    distance_mesh::dt<
        ndim, scalar_t, index_t, offset_t
    >(
        _nbatch, _dist, _nearest_vertex, _coord, _vertices, _faces,
        _size, _nb_faces, _nb_vertices,
        _stride_dist, _stride_nearest, _stride_coord, _stride_vertices, _stride_faces,
        _signed, naive, stream
    );

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_dist);
    free_if_needed<int64_t *>(_stride_coord);
    free_if_needed<int64_t *>(_stride_vertices);
    free_if_needed<int64_t *>(_stride_faces);
    if (_stride_nearest) free_if_needed<int64_t *>(_stride_nearest);
}

void dt_mesh(
          DLTensor & dist_,
          DLTensor & nearest_vertex_,
    const DLTensor & loc_,
    const DLTensor & vertices_,
    const DLTensor & faces_,
          bool       _signed,
          bool       naive,
          intptr_t   stream
)
{
    // Normalise NULL strides (compact row-major) before dispatch. nearest_vertex
    // is an optional output (null-data placeholder) and only normalised when set.
    ContiguousStrides _di(dist_), _lo(loc_), _ve(vertices_), _fa(faces_),
                      _nv(nearest_vertex_, nearest_vertex_.data != nullptr);
    DLTensor       & dist           = _di.t;
    DLTensor       & nearest_vertex = _nv.t;
    const DLTensor & loc            = _lo.t;
    const DLTensor & vertices       = _ve.t;
    const DLTensor & faces          = _fa.t;

    bool use_32bits = (
        FF_CANUSE32BITS(dist)              &&
        FF_CANUSE32BITS(loc)               &&
        FF_CANUSE32BITS(vertices)          &&
        FF_CANUSE32BITS(faces)
    );
    const int32_t ndim   = loc.shape[loc.ndim-1];
    const int32_t nbatch = loc.ndim - 1;
    FF_CHECK_NO_LANES  (dist)
    FF_CHECK_SAME_DTYPE(dist,  loc)
    FF_CHECK_SAME_DTYPE(dist,  vertices)
    FF_CHECK_SAME      (          dist.ndim, nbatch,   "Number of batch dimensions does not match")
    // vertices (N, D) and faces (M, D) describe a single shared mesh and are
    // always 2D; their leading axis is the vertex/face count, independent of
    // loc's point batch. Only `loc` and the per-point outputs share a batch.
    FF_CHECK_SAME      (      vertices.ndim, 2,        "Vertices must be a (N, D) tensor")
    FF_CHECK_SAME      (         faces.ndim, 2,        "Faces must be a (M, D) tensor")
    FF_CHECK_SAME_BATCH_ND(loc, dist,           nbatch)
    FF_CHECK_SAME      (vertices.shape[vertices.ndim-1], ndim, "Dimensionality of the vertices and location does not match")
    FF_CHECK_SAME      (faces.shape[faces.ndim-1],       ndim, "Dimensionality of the vertices and faces does not match")

    if (nearest_vertex.data)
    {
        FF_CHECK_SAME_DTYPE(faces, nearest_vertex)
        FF_CHECK_SAME      (nearest_vertex.ndim, nbatch,   "Number of batch dimensions does not match")
        FF_CHECK_SAME_BATCH_ND(loc, nearest_vertex, nbatch)
        use_32bits &= FF_CANUSE32BITS(nearest_vertex);
    }

    DISPATCH_MESH(
        _dt_mesh,
        nbatch,                             // nbatch
        FF_VOIDPTR(dist),                      // data
        FF_VOIDPTR(nearest_vertex),            // nearest_vertex
        FF_VOIDPTR(loc),                       // coord
        FF_VOIDPTR(vertices),                  // vertices
        FF_VOIDPTR(faces),                     // faces
        loc.shape,                          // size
        faces.shape[0],                     // nb_faces (M = faces.shape[0])
        vertices.shape[0],                  // nb_vertices (N = vertices.shape[0])
        dist.strides,                       // stride_dist
        nearest_vertex.strides,             // stride_nearest
        loc.strides,                        // stride_coord
        vertices.strides,                   // stride_vertices
        faces.strides,                      // stride_faces
        _signed,                            // signed
        naive,                              // naive
        stream                              // stream
    )
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
