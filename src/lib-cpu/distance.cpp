#include <stdexcept>
#include "distance.h"
#include "autocast.h"
#include "dlpack.h"
// R7 (TEENY-MIGRATION.md sec. 9): fastfields vendors DLPack v1.2, teeny v1.1,
// and both use the guard DLPACK_DLPACK_H_ -- so whichever is seen first wins
// for the whole TU. Our "dlpack.h" is included ABOVE on purpose; keep it there.
#include <teeny/dlpack.h>
#include "impl/kernels/cuda_switch.h"
#include "impl/kernels/utils.h"
#include "impl/distance_euclidean.h"
#include "impl/distance_l1.h"
#include "impl/distance_spline.h"
#include "impl/distance_mesh.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

#define VOIDPTR(x)      (static_cast<void*>(static_cast<char*>(x.data) + x.byte_offset))
#define CANUSE32BITS(x) (canUse32BitIndexMath(x.ndim, x.shape, x.strides))

/***********************************************************************
 *                              CHECKS                                 *
 ***********************************************************************/

#define CHECK_NO_LANES(tensor)                                          \
    if (tensor.dtype.lanes > 1)                                         \
        throw std::invalid_argument(                                    \
            "Only scalar data types are supported"                      \
        );

#define CHECK_SAME(X, Y, msg)                                           \
    if (X != Y) throw std::invalid_argument(msg);

#define CHECK_SAME_BATCH(X, Y, D)                                       \
    if (X.ndim < D || Y.ndim < D)                                       \
        throw std::invalid_argument(                                    \
            "Number of dimensions does not match"                       \
        );                                                              \
    for (int32_t d=0; d < D; ++d)                                       \
        if (X.shape[d] != Y.shape[d])                                   \
            throw std::invalid_argument(                                \
                "Tensors do not have the same batch shape"              \
            );                                                          \

#define CHECK_SAME_SHAPE(X, Y)                                          \
    if (X.ndim != Y.ndim)                                               \
        throw std::invalid_argument(                                    \
            "Tensors do not have the same number of dimensions"         \
        );                                                              \
    CHECK_SAME_BATCH(X, Y, X.ndim)

#define CHECK_SAME_DTYPE(X, Y)                                          \
    if (                                                                \
        (X.dtype.code  != Y.dtype.code) ||                              \
        (X.dtype.bits  != Y.dtype.bits) ||                              \
        (X.dtype.lanes != Y.dtype.lanes)                                \
    )                                                                   \
        throw std::invalid_argument(                                    \
            "Tensors do not have the same data type"                    \
        );

/***********************************************************************
 *                              EUCLIDEAN                              *
 ***********************************************************************/

// Offsets are computed host-side in int64 by the teeny impl (the anyrank folds
// the batch offset into each line's pointer). On the CPU the old int32 narrowing
// was a wash (64-bit ALU); the GPU port keeps whole-carrier host-side narrowing.
#define DISPATCH_DT(func, args...)                                      \
{                                                                       \
    const auto code = static_cast<DLDataTypeCode>(inp_out.dtype.code);  \
    switch (code) {                                                     \
        case kDLFloat: switch (inp_out.dtype.bits) {                    \
            case 32: return func<float >(args);                         \
            case 64: return func<double>(args);                         \
            default: break;                                             \
        };                                                              \
        default: throw std::invalid_argument(                           \
            "only floating point data types are supported"              \
        );                                                              \
    };                                                                  \
}

// The dtype arm's IMPORT POINT. `tny::from_dlpack` builds the teeny carrier
// once, straight off the bare DLTensor, and does by construction all three
// things this path used to do by hand: it folds `byte_offset` into the data
// pointer (was VOIDPTR), expands a NULL `strides` field to row-major (was
// ContiguousStrides), and copies the shape/stride metadata into the carrier --
// so the impl below needs no pointer, no ndim, and no size[]/stride[] arrays.
//
// These shims stay (rather than collapsing into the switch arms) because
// `scalar_t` LEADS their parameter list, which is what lets DISPATCH_DT's
// `func<float>` / `func<double>` spelling -- and its exact rejection message,
// behavioural ABI -- stay untouched. They are no longer void*-recasts; the
// import is their whole body.
namespace {
template <typename scalar_t = float>
inline void _dt_euclidean(
          DLTensor & inp_out  ,     // data [*batch, n], transformed in place
          double     w        )     // pixel spacing
{
    auto at = tny::from_dlpack<scalar_t>(&inp_out);
    distance_e::dt(at, static_cast<scalar_t>(w));
}
}

void dt_euclidean(
          DLTensor & inp_out,
          double     voxel_spacing,
          int        /* stream <unused> */
)
{
    CHECK_NO_LANES(inp_out)
    DISPATCH_DT(
        _dt_euclidean,
        inp_out,
        voxel_spacing
    )
}

namespace {
template <typename scalar_t = float>
inline void _dt_l1(
          DLTensor & inp_out  ,     // data [*batch, n], transformed in place
          double     w        )     // pixel spacing
{
    auto at = tny::from_dlpack<scalar_t>(&inp_out);
    distance_l1::dt(at, static_cast<scalar_t>(w));
}
}

void dt_l1(
          DLTensor & inp_out,
          double     voxel_spacing,
          int        /* stream <unused> */
)
{
    CHECK_NO_LANES(inp_out)
    DISPATCH_DT(
        _dt_l1,
        inp_out,
        voxel_spacing
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
          int        /* stream <unused> */
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
        CANUSE32BITS(time)  &&
        CANUSE32BITS(dist)  &&
        CANUSE32BITS(loc)   &&
        CANUSE32BITS(coeff) &&
        CANUSE32BITS(times)
    );
    const int32_t ndim   = loc.shape[loc.ndim-1];
    const int32_t nbatch = loc.ndim - 1;
    CHECK_NO_LANES  (time)
    CHECK_SAME_DTYPE(time, dist)
    CHECK_SAME_DTYPE(time, loc)
    CHECK_SAME_DTYPE(time, coeff)
    CHECK_SAME_DTYPE(time, times)
    CHECK_SAME      (time.ndim,  nbatch,   "Number of batch dimensions does not match")
    CHECK_SAME      (dist.ndim,  nbatch,   "Number of batch dimensions does not match")
    // coeff is (*batch, npoints, ndim) -> nbatch+2 dims
    // times is (*batch, ntimes)        -> nbatch+1 dims
    CHECK_SAME      (coeff.ndim, nbatch+2, "Number of coeff dimensions does not match")
    CHECK_SAME      (times.ndim, nbatch+1, "Number of times dimensions does not match")
    CHECK_SAME      (coeff.shape[coeff.ndim-1], ndim, "Dimensionality of coeff and location does not match")
    CHECK_SAME_BATCH(loc, time,  nbatch)
    CHECK_SAME_BATCH(loc, dist,  nbatch)
    CHECK_SAME_BATCH(loc, coeff, nbatch)
    CHECK_SAME_BATCH(loc, times, nbatch)

    DISPATCH_SPLINE(
        _dt_spline_table,
        nbatch,                             // nbatch
        VOIDPTR(time),                      // time
        VOIDPTR(dist),                      // dist
        VOIDPTR(loc),                       // loc
        VOIDPTR(coeff),                     // coeff
        VOIDPTR(times),                     // times
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
          int        /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _tm(time_), _di(dist_), _lo(loc_), _co(coeff_);
    DLTensor       & time  = _tm.t;
    DLTensor       & dist  = _di.t;
    const DLTensor & loc   = _lo.t;
    const DLTensor & coeff = _co.t;

    const bool use_32bits = (
        CANUSE32BITS(time)  &&
        CANUSE32BITS(dist)  &&
        CANUSE32BITS(loc)   &&
        CANUSE32BITS(coeff)
    );
    const int32_t ndim   = loc.shape[loc.ndim-1];
    const int32_t nbatch = loc.ndim - 1;
    CHECK_NO_LANES  (time)
    CHECK_SAME_DTYPE(time, dist)
    CHECK_SAME_DTYPE(time, loc)
    CHECK_SAME_DTYPE(time, coeff)
    CHECK_SAME      (time.ndim,  nbatch,   "Number of batch dimensions does not match")
    CHECK_SAME      (dist.ndim,  nbatch,   "Number of batch dimensions does not match")
    // coeff is (*batch, npoints, ndim) -> nbatch+2 dims
    CHECK_SAME      (coeff.ndim, nbatch+2, "Number of coeff dimensions does not match")
    CHECK_SAME      (coeff.shape[coeff.ndim-1], ndim, "Dimensionality of coeff and location does not match")
    CHECK_SAME_BATCH(loc, time,  nbatch)
    CHECK_SAME_BATCH(loc, dist,  nbatch)
    CHECK_SAME_BATCH(loc, coeff, nbatch)

    DISPATCH_SPLINE(
        _dt_spline_brent,
        nbatch,                             // nbatch
        VOIDPTR(time),                      // time
        VOIDPTR(dist),                      // dist
        VOIDPTR(loc),                       // loc
        VOIDPTR(coeff),                     // coeff
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
          int        /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _tm(time_), _di(dist_), _lo(loc_), _co(coeff_);
    DLTensor       & time  = _tm.t;
    DLTensor       & dist  = _di.t;
    const DLTensor & loc   = _lo.t;
    const DLTensor & coeff = _co.t;

    const bool use_32bits = (
        CANUSE32BITS(time)  &&
        CANUSE32BITS(dist)  &&
        CANUSE32BITS(loc)   &&
        CANUSE32BITS(coeff)
    );
    const int32_t ndim   = loc.shape[loc.ndim-1];
    const int32_t nbatch = loc.ndim - 1;
    CHECK_NO_LANES  (time)
    CHECK_SAME_DTYPE(time, dist)
    CHECK_SAME_DTYPE(time, loc)
    CHECK_SAME_DTYPE(time, coeff)
    CHECK_SAME      (time.ndim,  nbatch,   "Number of batch dimensions does not match")
    CHECK_SAME      (dist.ndim,  nbatch,   "Number of batch dimensions does not match")
    // coeff is (*batch, npoints, ndim) -> nbatch+2 dims
    CHECK_SAME      (coeff.ndim, nbatch+2, "Number of coeff dimensions does not match")
    CHECK_SAME      (coeff.shape[coeff.ndim-1], ndim, "Dimensionality of coeff and location does not match")
    CHECK_SAME_BATCH(loc, time,  nbatch)
    CHECK_SAME_BATCH(loc, dist,  nbatch)
    CHECK_SAME_BATCH(loc, coeff, nbatch)

    DISPATCH_SPLINE(
        _dt_spline_gaussnewton,
        nbatch,                             // nbatch
        VOIDPTR(time),                      // time
        VOIDPTR(dist),                      // dist
        VOIDPTR(loc),                       // loc
        VOIDPTR(coeff),                     // coeff
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
        case kDLInt: switch (faces.dtype.bits) {                        \
            case  8: return func<D,S,int8_t,  O>(args);                    \
            case 16: return func<D,S,int16_t, O>(args);                    \
            case 32: return func<D,S,int32_t, O>(args);                    \
            case 64: return func<D,S,int64_t, O>(args);                    \
            default: break;                                             \
        };                                                              \
        case kDLUInt: switch (faces.dtype.bits) {                       \
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
    bool      naive   = false
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
        _signed, naive
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
          int        /* stream <unused> */
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
        CANUSE32BITS(dist)              &&
        CANUSE32BITS(loc)               &&
        CANUSE32BITS(vertices)          &&
        CANUSE32BITS(faces)
    );
    const int32_t ndim   = loc.shape[loc.ndim-1];
    const int32_t nbatch = loc.ndim - 1;
    CHECK_NO_LANES  (dist)
    CHECK_SAME_DTYPE(dist,  loc)
    CHECK_SAME_DTYPE(dist,  vertices)
    CHECK_SAME      (          dist.ndim, nbatch,   "Number of batch dimensions does not match")
    // vertices (N, D) and faces (M, D) describe a single shared mesh and are
    // always 2D; their leading axis is the vertex/face count, independent of
    // loc's point batch. Only `loc` and the per-point outputs share a batch.
    CHECK_SAME      (      vertices.ndim, 2,        "Vertices must be a (N, D) tensor")
    CHECK_SAME      (         faces.ndim, 2,        "Faces must be a (M, D) tensor")
    CHECK_SAME_BATCH(loc, dist,           nbatch)
    CHECK_SAME      (vertices.shape[vertices.ndim-1], ndim, "Dimensionality of the vertices and location does not match")
    CHECK_SAME      (faces.shape[faces.ndim-1],       ndim, "Dimensionality of the vertices and faces does not match")

    if (nearest_vertex.data)
    {
        CHECK_SAME_DTYPE(faces, nearest_vertex)
        CHECK_SAME      (nearest_vertex.ndim, nbatch,   "Number of batch dimensions does not match")
        CHECK_SAME_BATCH(loc, nearest_vertex, nbatch)
        use_32bits &= CANUSE32BITS(nearest_vertex);
    }

    DISPATCH_MESH(
        _dt_mesh,
        nbatch,                             // nbatch
        VOIDPTR(dist),                      // data
        VOIDPTR(nearest_vertex),            // nearest_vertex
        VOIDPTR(loc),                       // coord
        VOIDPTR(vertices),                  // vertices
        VOIDPTR(faces),                     // faces
        loc.shape,                          // size
        faces.shape[0],                     // nb_faces (M = faces.shape[0])
        vertices.shape[0],                  // nb_vertices (N = vertices.shape[0])
        dist.strides,                       // stride_dist
        nearest_vertex.strides,             // stride_nearest
        loc.strides,                        // stride_coord
        vertices.strides,                   // stride_vertices
        faces.strides,                      // stride_faces
        _signed,                            // signed
        naive                               // naive
    )
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
