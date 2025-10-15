#ifndef FF_DISTANCE_SPLINE_CPU
#define FF_DISTANCE_SPLINE_CPU
#include "kernels/cuda_switch.h"
#include "kernels/distance.h"
#include "kernels/batch.h"
#include "kernels/parallel.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_spline)

// Compute the minimum distance from a set of points to a 1D spline
// using a dictionary of times
template <
    int      ndim,          // Number of spatial dimensions
    spline_t S,             // Spline order
    bound_t  B,             // Boundary condition
    typename scalar_t,      // Value data type
    typename offset_t       // Index/Stride data type
>
void mindist_table(
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
    const offset_t * stride_times   // [*batch, ntimes] list -> Strides of `times`
)
{
    using Klass = SplineDist<ndim, S, B, scalar_t, offset_t>;

    offset_t numel = prod(size, nbatch);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_time  = index2offset(i, nbatch, size, stride_time);
        offset_t offset_dist  = index2offset(i, nbatch, size, stride_dist);
        offset_t offset_loc   = index2offset(i, nbatch, size, stride_loc);
        offset_t offset_coeff = index2offset(i, nbatch, size, stride_coeff);
        offset_t offset_times = index2offset(i, nbatch, size, stride_times);

        Klass::min_table(
            time + offset_time,
            dist + offset_dist,
            loc + offset_loc,
            coeff + offset_coeff,
            times + offset_times,
            ntimes,
            stride_times[nbatch],
            stride_loc[nbatch],
            size[nbatch],
            stride_coeff[nbatch],
            stride_coeff[nbatch+1]
        );
    }});
}

// Compute the minimum distance from a set of points to a 1D spline
// using Brent (gradient-free) optimization
template <
    int      ndim,          // Number of spatial dimensions
    spline_t S,             // Spline order
    bound_t  B,             // Boundary condition
    typename scalar_t,      // Value data type
    typename offset_t       // Index/Stride data type
>
void mindist_brent(
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
          scalar_t   step
)
{
    using Klass = SplineDist<ndim, S, B, scalar_t, offset_t>;

    offset_t numel = prod(size, nbatch);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_time  = index2offset(i, nbatch, size, stride_time);
        offset_t offset_dist  = index2offset(i, nbatch, size, stride_dist);
        offset_t offset_loc   = index2offset(i, nbatch, size, stride_loc);
        offset_t offset_coeff = index2offset(i, nbatch, size, stride_coeff);

        Klass::min_brent(
            time + offset_time,
            dist + offset_dist,
            loc + offset_loc,
            coeff + offset_coeff,
            stride_loc[nbatch],
            size[nbatch],
            stride_coeff[nbatch],
            stride_coeff[nbatch+1],
            max_iter,
            tol,
            step
        );
    }});
}


// Compute the minimum distance from a set of points to a 1D spline
// using Gauss-Newton optimization
template <
    int      ndim,          // Number of spatial dimensions
    spline_t S,             // Spline order
    bound_t  B,             // Boundary condition
    typename scalar_t,      // Value data type
    typename offset_t       // Index/Stride data type
>
void mindist_gaussnewton(
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
          scalar_t   tol
)
{
    using Klass = SplineDist<ndim, S, B, scalar_t, offset_t>;

    offset_t numel = prod(size, nbatch);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_time  = index2offset(i, nbatch, size, stride_time);
        offset_t offset_dist  = index2offset(i, nbatch, size, stride_dist);
        offset_t offset_loc   = index2offset(i, nbatch, size, stride_loc);
        offset_t offset_coeff = index2offset(i, nbatch, size, stride_coeff);

        Klass::min_gaussnewton(
            time + offset_time,
            dist + offset_dist,
            loc + offset_loc,
            coeff + offset_coeff,
            stride_loc[nbatch],
            size[nbatch],
            stride_coeff[nbatch],
            stride_coeff[nbatch+1],
            max_iter,
            tol
        );
    }});
}

FF_NAMESPACE_END(distance_spline)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_DISTANCE_SPLINE_CPU
