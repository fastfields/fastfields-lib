#ifndef FF_DISTANCE_SPLINE_CPU
#define FF_DISTANCE_SPLINE_CPU
#include "kernels/cuda_switch.h"
#include "kernels/distance.h"
#include "kernels/batch.h"
#include "kernels/parallel.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_spline)

template <class T, class U>
inline T unsafe_cast(U ptr)
{
    return static_cast<T>(static_cast<void *>(ptr));
}

// Compute the minimum distance from a set of points to a 1D spline
// using a dictionary of times
template <
    int      ndim,                          // Number of spatial dimensions
    spline_t S        = spline_t::Cubic,    // Spline order
    bound_t  B        = bound_t::DCT2,      // Boundary condition
    typename scalar_t = float,              // Value data type
    typename offset_t = int64_t             // Index/Stride data type
>
static inline void
_mindist_table(
          offset_t   nbatch,        // Number of batch dimensions
          scalar_t * time,          // (*batch) tensor -> Best time
          scalar_t * dist,          // (*batch) tensor -> Best sqdist
    const scalar_t * loc,           // (*batch, ndim) tensor -> ND location of each point
    const scalar_t * coeff,         // (*batch, npoints, ndim) tensor -> Spline coefficients
    const scalar_t * times,         // (*batch) tensor -> Time values to try
          offset_t   ntimes,        // Number of times values to try
    const offset_t * size,          // [*batch, npoints, ndim] list -> Coeff shape
    const offset_t * stride_time,   // [*batch] list -> Strides of `time`
    const offset_t * stride_dist,   // [*batch] list -> Strides of `dist`
    const offset_t * stride_loc,    // [*batch, ndim] list -> Strides of `loc`
    const offset_t * stride_coeff,  // [*batch, npoints, ndim] list -> Strides or `coeff`
    const offset_t * stride_times,  // [*batch, ntimes] list -> Strides of `times`
          spline_t   spline = S,
          bound_t    bound  = B
)
{
    using Kernel = Kernels<Config<ndim, S, B, scalar_t, offset_t>>;

    offset_t numel = prod(size, nbatch);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_time  = index2offset(i, nbatch, size, stride_time);
        offset_t offset_dist  = index2offset(i, nbatch, size, stride_dist);
        offset_t offset_loc   = index2offset(i, nbatch, size, stride_loc);
        offset_t offset_coeff = index2offset(i, nbatch, size, stride_coeff);
        offset_t offset_times = index2offset(i, nbatch, size, stride_times);

        Kernel::min_table(
            time  + offset_time,
            dist  + offset_dist,
            loc   + offset_loc,
            coeff + offset_coeff,
            times + offset_times,
            ntimes,
            stride_times [nbatch],
            stride_loc   [nbatch],
            size         [nbatch],
            stride_coeff [nbatch],
            stride_coeff [nbatch+1],
            spline,
            bound
        );
    }});
}

// Compute the minimum distance from a set of points to a 1D spline
// using Brent (gradient-free) optimization
template <
    int      ndim,                          // Number of spatial dimensions
    spline_t S        = spline_t::Cubic,    // Spline order
    bound_t  B        = bound_t::DCT2,      // Boundary condition
    typename scalar_t = float,              // Value data type
    typename offset_t = int64_t             // Index/Stride data type
>
static inline void
_mindist_brent(
          offset_t   nbatch,        // Number of batch dimensions
          scalar_t * time,          // (*batch) tensor -> Best time
          scalar_t * dist,          // (*batch) tensor -> Best sqdist
    const scalar_t * loc,           // (*batch, ndim) tensor -> ND location of each point
    const scalar_t * coeff,         // (*batch, npoints, ndim) tensor -> Spline coefficients
    const offset_t * size,          // [*batch, npoints, ndim] list -> Coeff shape
    const offset_t * stride_time,   // [*batch] list -> Strides of `time`
    const offset_t * stride_dist,   // [*batch] list -> Strides of `dist`
    const offset_t * stride_loc,    // [*batch, ndim] list -> Strides of `loc`
    const offset_t * stride_coeff,  // [*batch, npoints, ndim] list -> Strides or `coeff`
          offset_t   max_iter,
          scalar_t   tol,
          scalar_t   step,
          spline_t   spline = S,
          bound_t    bound  = B
)
{
    using Kernel = Kernels<Config<ndim, S, B, scalar_t, offset_t>>;

    offset_t numel = prod(size, nbatch);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_time  = index2offset(i, nbatch, size, stride_time);
        offset_t offset_dist  = index2offset(i, nbatch, size, stride_dist);
        offset_t offset_loc   = index2offset(i, nbatch, size, stride_loc);
        offset_t offset_coeff = index2offset(i, nbatch, size, stride_coeff);

        Kernel::min_brent(
            time  + offset_time,
            dist  + offset_dist,
            loc   + offset_loc,
            coeff + offset_coeff,
            stride_loc   [nbatch],
            size         [nbatch],
            stride_coeff [nbatch],
            stride_coeff [nbatch+1],
            max_iter,
            tol,
            step,
            spline,
            bound
        );
    }});
}


// Compute the minimum distance from a set of points to a 1D spline
// using Gauss-Newton optimization
template <
    int      ndim,                          // Number of spatial dimensions
    spline_t S        = spline_t::Cubic,    // Spline order
    bound_t  B        = bound_t::DCT2,      // Boundary condition
    typename scalar_t = float,              // Value data type
    typename offset_t = int64_t             // Index/Stride data type
>
static inline void
_mindist_gaussnewton(
          offset_t   nbatch,        // Number of batch dimensions
          scalar_t * time,          // (*batch) tensor -> Best time
          scalar_t * dist,          // (*batch) tensor -> Best sqdist
    const scalar_t * loc,           // (*batch, ndim) tensor -> ND location of each point
    const scalar_t * coeff,         // (*batch, npoints, ndim) tensor -> Spline coefficients
    const offset_t * size,          // [*batch, npoints, ndim] list -> Coeff shape
    const offset_t * stride_time,   // [*batch] list -> Strides of `time`
    const offset_t * stride_dist,   // [*batch] list -> Strides of `dist`
    const offset_t * stride_loc,    // [*batch, ndim] list -> Strides of `loc`
    const offset_t * stride_coeff,  // [*batch, npoints, ndim] list -> Strides or `coeff`
          offset_t   max_iter,
          scalar_t   tol,
          spline_t   spline = S,
          bound_t    bound  = B
)
{
    using Kernel = Kernels<Config<ndim, S, B, scalar_t, offset_t>>;

    offset_t numel = prod(size, nbatch);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_time  = index2offset(i, nbatch, size, stride_time);
        offset_t offset_dist  = index2offset(i, nbatch, size, stride_dist);
        offset_t offset_loc   = index2offset(i, nbatch, size, stride_loc);
        offset_t offset_coeff = index2offset(i, nbatch, size, stride_coeff);

        Kernel::min_gaussnewton(
            time  + offset_time,
            dist  + offset_dist,
            loc   + offset_loc,
            coeff + offset_coeff,
            stride_loc   [nbatch],
            size         [nbatch],
            stride_coeff [nbatch],
            stride_coeff [nbatch+1],
            max_iter,
            tol,
            spline,
            bound
        );
    }});
}

template <
    typename scalar_t = float,              // Value data type
    typename offset_t = int64_t             // Index/Stride data type
>
struct AutoCast {

    template <
        int32_t  D,
        spline_t S              = spline_t::Cubic,
        bound_t  B              = bound_t::DCT2,
        typename nbatch_t       = int32_t,
        typename time_t         = scalar_t,
        typename dist_t         = scalar_t,
        typename loc_t          = const scalar_t,
        typename coeff_t        = const scalar_t,
        typename times_t        = const scalar_t,
        typename ntimes_t       = offset_t,
        typename size_t         = const offset_t,
        typename stride_time_t  = const offset_t,
        typename stride_dist_t  = const offset_t,
        typename stride_loc_t   = const offset_t,
        typename stride_coeff_t = const offset_t,
        typename stride_times_t = const offset_t,
        typename spline_tt      = spline_t,
        typename bound_tt       = bound_t
    >
    static inline void
    mindist_table(
        nbatch_t         nbatch,        // Number of batch dimensions
        time_t         * time,          // (*batch) tensor -> Best time
        dist_t         * dist,          // (*batch) tensor -> Best sqdist
        loc_t          * loc,           // (*batch, ndim) tensor -> ND location of each point
        coeff_t        * coeff,         // (*batch, npoints, ndim) tensor -> Spline coefficients
        times_t        * times,         // (*batch) tensor -> Time values to try
        ntimes_t         ntimes,        // Number of times values to try
        size_t         * size,          // [*batch, npoints, ndim] list -> Coeff shape
        stride_time_t  * stride_time,   // [*batch] list -> Strides of `time`
        stride_dist_t  * stride_dist,   // [*batch] list -> Strides of `dist`
        stride_loc_t   * stride_loc,    // [*batch, ndim] list -> Strides of `loc`
        stride_coeff_t * stride_coeff,  // [*batch, npoints, ndim] list -> Strides or `coeff`
        stride_times_t * stride_times,  // [*batch, ntimes] list -> Strides of `times`
        spline_tt        spline = static_cast<spline_tt>(S),
        bound_tt         bound  = static_cast<bound_tt>(B)
    )
    {
        return _mindist_table<D>(
            static_cast<      offset_t  > (nbatch),
            unsafe_cast<      scalar_t *> (time),
            unsafe_cast<      scalar_t *> (dist),
            unsafe_cast<const scalar_t *> (loc),
            unsafe_cast<const scalar_t *> (coeff),
            unsafe_cast<const scalar_t *> (times),
            static_cast<      offset_t  > (ntimes),
            unsafe_cast<const offset_t *> (size),
            unsafe_cast<const offset_t *> (stride_time),
            unsafe_cast<const offset_t *> (stride_dist),
            unsafe_cast<const offset_t *> (stride_loc),
            unsafe_cast<const offset_t *> (stride_coeff),
            unsafe_cast<const offset_t *> (stride_times),
            static_cast<      spline_t  > (spline),
            static_cast<      bound_t   > (bound)
        );
    }

    template <
        int32_t  D,
        spline_t S              = spline_t::Cubic,
        bound_t  B              = bound_t::DCT2,
        typename nbatch_t       = offset_t,
        typename time_t         = scalar_t,
        typename dist_t         = scalar_t,
        typename loc_t          = const scalar_t,
        typename coeff_t        = const scalar_t,
        typename size_t         = const offset_t,
        typename stride_time_t  = const offset_t,
        typename stride_dist_t  = const offset_t,
        typename stride_loc_t   = const offset_t,
        typename stride_coeff_t = const offset_t,
        typename max_iter_t     = offset_t,
        typename tol_t          = scalar_t,
        typename step_t         = scalar_t,
        typename spline_tt      = spline_t,
        typename bound_tt       = bound_t
    >
    static inline void
    mindist_brent(
        nbatch_t         nbatch,        // Number of batch dimensions
        time_t         * time,          // (*batch) tensor -> Best time
        dist_t         * dist,          // (*batch) tensor -> Best sqdist
        loc_t          * loc,           // (*batch, ndim) tensor -> ND location of each point
        coeff_t        * coeff,         // (*batch, npoints, ndim) tensor -> Spline coefficients
        size_t         * size,          // [*batch, npoints, ndim] list -> Coeff shape
        stride_time_t  * stride_time,   // [*batch] list -> Strides of `time`
        stride_dist_t  * stride_dist,   // [*batch] list -> Strides of `dist`
        stride_loc_t   * stride_loc,    // [*batch, ndim] list -> Strides of `loc`
        stride_coeff_t * stride_coeff,  // [*batch, npoints, ndim] list -> Strides or `coeff`
        max_iter_t       max_iter,
        tol_t            tol,
        step_t           step,
        spline_tt        spline = static_cast<spline_tt>(S),
        bound_tt         bound  = static_cast<bound_tt>(B)
    )
    {
        return _mindist_brent<D>(
            static_cast<      offset_t  > (nbatch),
            unsafe_cast<      scalar_t *> (time),
            unsafe_cast<      scalar_t *> (dist),
            unsafe_cast<const scalar_t *> (loc),
            unsafe_cast<const scalar_t *> (coeff),
            unsafe_cast<const offset_t *> (size),
            unsafe_cast<const offset_t *> (stride_time),
            unsafe_cast<const offset_t *> (stride_dist),
            unsafe_cast<const offset_t *> (stride_loc),
            unsafe_cast<const offset_t *> (stride_coeff),
            static_cast<      offset_t  > (max_iter),
            static_cast<      scalar_t  > (tol),
            static_cast<      scalar_t  > (step),
            static_cast<      spline_t  > (spline),
            static_cast<      bound_t   > (bound)
        );
    }

    template <
        int32_t  D,
        spline_t S              = spline_t::Cubic,
        bound_t  B              = bound_t::DCT2,
        typename nbatch_t       = offset_t,
        typename time_t         = scalar_t,
        typename dist_t         = scalar_t,
        typename loc_t          = const scalar_t,
        typename coeff_t        = const scalar_t,
        typename size_t         = const offset_t,
        typename stride_time_t  = const offset_t,
        typename stride_dist_t  = const offset_t,
        typename stride_loc_t   = const offset_t,
        typename stride_coeff_t = const offset_t,
        typename max_iter_t     = offset_t,
        typename tol_t          = scalar_t,
        typename spline_tt      = spline_t,
        typename bound_tt       = bound_t
    >
    static inline void
    mindist_gaussnewton(
        nbatch_t         nbatch,        // Number of batch dimensions
        time_t         * time,          // (*batch) tensor -> Best time
        dist_t         * dist,          // (*batch) tensor -> Best sqdist
        loc_t          * loc,           // (*batch, ndim) tensor -> ND location of each point
        coeff_t        * coeff,         // (*batch, npoints, ndim) tensor -> Spline coefficients
        size_t         * size,          // [*batch, npoints, ndim] list -> Coeff shape
        stride_time_t  * stride_time,   // [*batch] list -> Strides of `time`
        stride_dist_t  * stride_dist,   // [*batch] list -> Strides of `dist`
        stride_loc_t   * stride_loc,    // [*batch, ndim] list -> Strides of `loc`
        stride_coeff_t * stride_coeff,  // [*batch, npoints, ndim] list -> Strides or `coeff`
        max_iter_t       max_iter,
        tol_t            tol,
        spline_tt        spline = static_cast<spline_tt>(S),
        bound_tt         bound  = static_cast<bound_tt>(B)
    )
    {
        return _mindist_gaussnewton<D>(
            static_cast<      offset_t  > (nbatch),
            unsafe_cast<      scalar_t *> (time),
            unsafe_cast<      scalar_t *> (dist),
            unsafe_cast<const scalar_t *> (loc),
            unsafe_cast<const scalar_t *> (coeff),
            unsafe_cast<const offset_t *> (size),
            unsafe_cast<const offset_t *> (stride_time),
            unsafe_cast<const offset_t *> (stride_dist),
            unsafe_cast<const offset_t *> (stride_loc),
            unsafe_cast<const offset_t *> (stride_coeff),
            static_cast<      offset_t  > (max_iter),
            static_cast<      scalar_t  > (tol),
            static_cast<      spline_t  > (spline),
            static_cast<      bound_t   > (bound)
        );
    }
};


FF_NAMESPACE_END(distance_spline)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_DISTANCE_SPLINE_CPU
