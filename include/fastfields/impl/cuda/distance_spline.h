#ifndef FF_IMPL_CUDA_DISTANCE_SPLINE_H
#define FF_IMPL_CUDA_DISTANCE_SPLINE_H
#include <stdexcept>
#include "fastfields/core/cuda_switch.h"
#include "fastfields/impl/kernels/distance.h"
#include "fastfields/impl/kernels/batch.h"
#include "fastfields/impl/kernels/utils.h"

using namespace std;
FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_spline)

// Compute the minimum distance from a set of points to a 1D spline
// using a dictionary of times
template <
    int nbatch,             // Number of batch dimensions
    int ndim,               // Number of spatial dimensions
    spline::type S,         // Spline order
    bound::type B,          // Boundary condition
    typename scalar_t,      // Value data type
    typename offset_t       // Index/Stride data type
>
FF_CUGLOB void mindist_table(
    scalar_t * time,                // (*batch) tensor -> Best time
    scalar_t * dist,                // (*batch) tensor -> Best sqdist
    const scalar_t * loc,           // (*batch, ndim) tensor -> ND location of each point
    const scalar_t * coeff,         // (*batch, npoints, ndim) tensor -> Spline coefficients
    const scalar_t * times,         // (*batch) tensor -> Time values to try
    offset_t ntimes,                // Number of times values to try
    const offset_t * _size,         // [*batch, npoints, ndim] list -> Coeff shape
    const offset_t * _stride_time,  // [*batch] list -> Strides of `time`
    const offset_t * _stride_dist,  // [*batch] list -> Strides of `dist`
    const offset_t * _stride_loc,   // [*batch, ndim] list -> Strides of `loc`
    const offset_t * _stride_coeff, // [*batch, npoints, ndim] list -> Strides or `coeff`
    const offset_t * _stride_times  // [*batch, ntimes] list -> Strides of `times`
)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;
    using Klass = Kernels<Config<ndim, S, B, scalar_t, offset_t>>;

    offset_t size         [nbatch+2]; fillfrom<nbatch+2> (size,         _size);
    offset_t stride_time  [nbatch];   fillfrom<nbatch>   (stride_time,  _stride_time);
    offset_t stride_dist  [nbatch];   fillfrom<nbatch>   (stride_dist,  _stride_dist);
    offset_t stride_loc   [nbatch+1]; fillfrom<nbatch+1> (stride_loc,   _stride_loc);
    offset_t stride_coeff [nbatch+2]; fillfrom<nbatch+2> (stride_coeff, _stride_coeff);
    offset_t stride_times [nbatch+1]; fillfrom<nbatch+1> (stride_times, _stride_times);

    offset_t numel = prod<nbatch>(size);
    for (offset_t i=index; index < numel; index += stride, i=index)
    {
        offset_t offset_time  = index2offset<nbatch>(i, size, stride_time);
        offset_t offset_dist  = index2offset<nbatch>(i, size, stride_dist);
        offset_t offset_loc   = index2offset<nbatch>(i, size, stride_loc);
        offset_t offset_coeff = index2offset<nbatch>(i, size, stride_coeff);
        offset_t offset_times = index2offset<nbatch>(i, size, stride_times);

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
    }
}

// Compute the minimum distance from a set of points to a 1D spline
// using Brent (gradient-free) optimization
template <
    int nbatch,             // Number of batch dimensions
    int ndim,               // Number of spatial dimensions
    spline::type S,         // Spline order
    bound::type B,          // Boundary condition
    typename scalar_t,      // Value data type
    typename offset_t       // Index/Stride data type
>
FF_CUGLOB void mindist_brent(
    scalar_t * time,                // (*batch) tensor -> Best time
    scalar_t * dist,                // (*batch) tensor -> Best sqdist
    const scalar_t * loc,           // (*batch, ndim) tensor -> ND location of each point
    const scalar_t * coeff,         // (*batch, npoints, ndim) tensor -> Spline coefficients
    const offset_t * _size,         // [*batch, npoints, ndim] list -> Coeff shape
    const offset_t * _stride_time,  // [*batch] list -> Strides of `time`
    const offset_t * _stride_dist,  // [*batch] list -> Strides of `dist`
    const offset_t * _stride_loc,   // [*batch, ndim] list -> Strides of `loc`
    const offset_t * _stride_coeff, // [*batch, npoints, ndim] list -> Strides or `coeff`
    offset_t max_iter,
    scalar_t tol,
    scalar_t step
)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;
    using Klass = Kernels<Config<ndim, S, B, scalar_t, offset_t>>;

    offset_t size         [nbatch+2]; fillfrom<nbatch+2> (size,         _size);
    offset_t stride_time  [nbatch];   fillfrom<nbatch>   (stride_time,  _stride_time);
    offset_t stride_dist  [nbatch];   fillfrom<nbatch>   (stride_dist,  _stride_dist);
    offset_t stride_loc   [nbatch+1]; fillfrom<nbatch+1> (stride_loc,   _stride_loc);
    offset_t stride_coeff [nbatch+2]; fillfrom<nbatch+2> (stride_coeff, _stride_coeff);

    offset_t numel = prod<nbatch>(size);
    for (offset_t i=index; index < numel; index += stride, i=index)
    {
        offset_t offset_time  = index2offset<nbatch>(i, size, stride_time);
        offset_t offset_dist  = index2offset<nbatch>(i, size, stride_dist);
        offset_t offset_loc   = index2offset<nbatch>(i, size, stride_loc);
        offset_t offset_coeff = index2offset<nbatch>(i, size, stride_coeff);

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
    }
}


// Compute the minimum distance from a set of points to a 1D spline
// using Gauss-Newton optimization
template <
    int nbatch,             // Number of batch dimensions
    int ndim,               // Number of spatial dimensions
    spline::type S,         // Spline order
    bound::type B,          // Boundary condition
    typename scalar_t,      // Value data type
    typename offset_t       // Index/Stride data type
>
FF_CUGLOB void mindist_gaussnewton(
    scalar_t * time,                // (*batch) tensor -> Best time
    scalar_t * dist,                // (*batch) tensor -> Best sqdist
    const scalar_t * loc,           // (*batch, ndim) tensor -> ND location of each point
    const scalar_t * coeff,         // (*batch, npoints, ndim) tensor -> Spline coefficients
    const offset_t * _size,         // [*batch, npoints, ndim] list -> Coeff shape
    const offset_t * _stride_time,  // [*batch] list -> Strides of `time`
    const offset_t * _stride_dist,  // [*batch] list -> Strides of `dist`
    const offset_t * _stride_loc,   // [*batch, ndim] list -> Strides of `loc`
    const offset_t * _stride_coeff, // [*batch, npoints, ndim] list -> Strides or `coeff`
    offset_t max_iter,
    scalar_t tol
)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;
    using Klass = Kernels<Config<ndim, S, B, scalar_t, offset_t>>;

    offset_t size         [nbatch+2]; fillfrom<nbatch+2> (size,         _size);
    offset_t stride_time  [nbatch];   fillfrom<nbatch>   (stride_time,  _stride_time);
    offset_t stride_dist  [nbatch];   fillfrom<nbatch>   (stride_dist,  _stride_dist);
    offset_t stride_loc   [nbatch+1]; fillfrom<nbatch+1> (stride_loc,   _stride_loc);
    offset_t stride_coeff [nbatch+2]; fillfrom<nbatch+2> (stride_coeff, _stride_coeff);

    offset_t numel = prod<nbatch>(size);
    for (offset_t i=index; index < numel; index += stride, i=index)
    {
        offset_t offset_time  = index2offset<nbatch>(i, size, stride_time);
        offset_t offset_dist  = index2offset<nbatch>(i, size, stride_dist);
        offset_t offset_loc   = index2offset<nbatch>(i, size, stride_loc);
        offset_t offset_coeff = index2offset<nbatch>(i, size, stride_coeff);

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
    }
}

// ---------------------------------------------------------------------------
// Host launchers (mirror cpu-impl distance_spline). These wrap the FF_CUGLOB
// kernels above (device shape/stride copy, grid config, launch, stream). Not
// implemented yet — the CUDA point-to-spline launchers are pending. Provided
// with the cpu-impl signatures so the cuda-lib dispatch layer compiles + links;
// they throw until authored. Compile-verified under nvcc; runtime needs a GPU.
template <int ndim, spline_t spline, bound_t bound, typename scalar_t, typename offset_t>
FF_CUHOST inline void
mindist_table(
          offset_t nbatch, scalar_t* /*time*/, scalar_t* /*dist*/,
    const scalar_t* /*loc*/, const scalar_t* /*coeff*/, const scalar_t* /*times*/,
          offset_t /*ntimes*/, const offset_t* /*size*/,
    const offset_t* /*stride_time*/, const offset_t* /*stride_dist*/,
    const offset_t* /*stride_loc*/,  const offset_t* /*stride_coeff*/,
    const offset_t* /*stride_times*/,
          spline_t /*spline_dyn*/ = spline_t::Cubic,
          bound_t  /*bound_dyn*/  = bound_t::DCT2)
{
    throw std::logic_error("distance_spline::mindist_table (CUDA) not implemented");
}

template <int ndim, spline_t spline, bound_t bound, typename scalar_t, typename offset_t>
FF_CUHOST inline void
mindist_brent(
          offset_t nbatch, scalar_t* /*time*/, scalar_t* /*dist*/,
    const scalar_t* /*loc*/, const scalar_t* /*coeff*/, const offset_t* /*size*/,
    const offset_t* /*stride_time*/, const offset_t* /*stride_dist*/,
    const offset_t* /*stride_loc*/,  const offset_t* /*stride_coeff*/,
          offset_t /*max_iter*/, scalar_t /*tol*/, scalar_t /*step*/,
          spline_t /*spline_dyn*/ = spline_t::Cubic,
          bound_t  /*bound_dyn*/  = bound_t::DCT2)
{
    throw std::logic_error("distance_spline::mindist_brent (CUDA) not implemented");
}

template <int ndim, spline_t spline, bound_t bound, typename scalar_t, typename offset_t>
FF_CUHOST inline void
mindist_gaussnewton(
          offset_t nbatch, scalar_t* /*time*/, scalar_t* /*dist*/,
    const scalar_t* /*loc*/, const scalar_t* /*coeff*/, const offset_t* /*size*/,
    const offset_t* /*stride_time*/, const offset_t* /*stride_dist*/,
    const offset_t* /*stride_loc*/,  const offset_t* /*stride_coeff*/,
          offset_t /*max_iter*/, scalar_t /*tol*/,
          spline_t /*spline_dyn*/ = spline_t::Cubic,
          bound_t  /*bound_dyn*/  = bound_t::DCT2)
{
    throw std::logic_error("distance_spline::mindist_gaussnewton (CUDA) not implemented");
}

FF_NAMESPACE_END(distance_spline)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
#endif // FF_IMPL_CUDA_DISTANCE_SPLINE_H
