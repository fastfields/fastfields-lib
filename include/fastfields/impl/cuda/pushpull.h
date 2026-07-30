#include "kernels/cuda_switch.h"
#include "kernels/spline.h"
#include "kernels/bounds.h"
#include "kernels/batch.h"
#include "kernels/pushpull.h"
#include "utils.h"       // allocDevice / copyToDevice / freeDevice / GET_BLOCKS
#include <cstdint>       // std::intptr_t
#include <stdexcept>     // std::logic_error

using namespace std;
FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(pushpull)

// Runtime "in field of view" check. `extrapolate` is a runtime argument
// (rather than a template parameter) so that the interpolation kernels are
// not needlessly instantiated once per extrapolation mode -- the mode only
// gates a cheap bounds test and is independent of spline/bound. Mirrors
// fastfields-cpu-impl's `infov_dyn` exactly; it used to be a *compile-time*
// template parameter here only, a CUDA-specific 3x multiplier on every
// pushpull kernel with no matching benefit (the CPU side never paid it).
//   1  : always in FOV (extrapolate everywhere)
//   0  : reject coordinates past the first/last voxel *centres*
//  -1  : reject coordinates past the first/last voxel *edges*
template <int ndim, typename scalar_t, typename offset_t>
inline CUDEV bool infov_dyn(int extrapolate, const scalar_t * loc, const offset_t * size)
{
    if (extrapolate > 0)  return InFOV< 1, ndim>::infov(loc, size);
    if (extrapolate == 0) return InFOV< 0, ndim>::infov(loc, size);
    return                       InFOV<-1, ndim>::infov(loc, size);
}

template <int nbatch, int ndim,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
CUGLOB
void pull(
    bound::BoundVec  bnd,
    spline::SplineVec spl,
    int extrapolate,
    scalar_t * out,                // (*batch, *spatial_grid, C) tensor | Placeholder for the pulled volume
    const scalar_t * inp,          // (*batch, *spatial_spln, C) tensor | Input volume
    const scalar_t * grid,         // (*batch, *spatial_grid, D) tensor | Coordinates into the input volume
    const offset_t * _size_grid,   // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc, // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,  // [*batch, *spatial_grid, C] vector
    const offset_t * _stride_inp,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_grid) // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;
    const bound_t  _bnd[3] = { bnd[0], bnd[1], bnd[2] };
    const spline_t _spl[3] = { spl[0], spl[1], spl[2] };

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+1]; fillfrom<nall+1>(stride_out,  _stride_out);
    offset_t stride_inp  [nall+1]; fillfrom<nall+1>(stride_inp,  _stride_inp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc  = size_splinc[nall];
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t gsc = stride_grid[nall];

    auto pull = [&](const reduce_t * loc, offset_t out_offset, offset_t inp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ>::pull(
            out + out_offset, inp + inp_offset,
            loc, size_splinc + nbatch, stride_inp + nbatch, nc, osc, isc,
            _bnd, _spl);
    };

    offset_t numel = prod<nall>(size_grid);  // no outer loop across channels
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t out_offset = index2offset<nall>(i, size_grid, stride_out);
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);

        reduce_t loc[ndim]; fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!infov_dyn<ndim>(extrapolate, loc, size_splinc+nbatch))
        {
            for (offset_t c=0; c<nc; ++c)
                out[out_offset + c * osc] = static_cast<scalar_t>(0);
            continue;
        }
        offset_t inp_offset = index2offset<nbatch>(i, size_grid, stride_inp);

        pull(loc, out_offset, inp_offset);
    }
}

template <int nbatch, int ndim,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
CUGLOB
void push(
    bound::BoundVec  bnd,
    spline::SplineVec spl,
    int extrapolate,
    scalar_t * out,                   // (*batch, *spatial_spln, C) tensor | Placeholder for the splatted volume
    const scalar_t * inp,             // (*batch, *spatial_grid, C) tensor | Input volume
    const scalar_t * grid,            // (*batch, *spatial_grid, D) tensor | Coordinates into the output volume
    const offset_t * _size_grid,      // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc,    // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,     // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_inp,     // [*batch, *spatial_grid, C] vector
    const offset_t * _stride_grid)    // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;
    const bound_t  _bnd[3] = { bnd[0], bnd[1], bnd[2] };
    const spline_t _spl[3] = { spl[0], spl[1], spl[2] };

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+1]; fillfrom<nall+1>(stride_out,  _stride_out);
    offset_t stride_inp  [nall+1]; fillfrom<nall+1>(stride_inp,  _stride_inp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc  = size_splinc[nall];
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t gsc = stride_grid[nall];

    auto push = [&] (const reduce_t * loc, offset_t out_offset, offset_t inp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ>::push(
            out + out_offset, inp + inp_offset,
            loc, size_splinc + nbatch, stride_out + nbatch, nc, osc, isc,
            _bnd, _spl);
    };

    offset_t numel = prod<nall>(size_grid);  // no outer loop across channels
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);

        reduce_t loc[ndim]; fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!infov_dyn<ndim>(extrapolate, loc, _size_splinc+nbatch))
            continue;

        offset_t inp_offset = index2offset<nall>(i, size_grid, stride_inp);
        offset_t out_offset = index2offset<nbatch>(i, size_grid, stride_out);

        push(loc, out_offset, inp_offset);
    }
}

template <int nbatch, int ndim,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
CUGLOB
void count(
    bound::BoundVec  bnd,
    spline::SplineVec spl,
    int extrapolate,
    scalar_t * out,                  // (*batch, *spatial_spln, C) tensor | Placeholder for the count image
    const scalar_t * grid,           // (*batch, *spatial_grid, D) tensor | Coordinates into the output volume
    const offset_t * _size_grid,     // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc,   // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,    // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_grid)   // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;
    const bound_t  _bnd[3] = { bnd[0], bnd[1], bnd[2] };
    const spline_t _spl[3] = { spl[0], spl[1], spl[2] };

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+1]; fillfrom<nall+1>(stride_out,  _stride_out);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t gsc = stride_grid[nall];

    auto count = [&](const reduce_t * loc, offset_t out_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ>::count(
            out + out_offset, loc, size_splinc + nbatch, stride_out + nbatch,
            _bnd, _spl);
    };

    offset_t numel = prod<nall>(size_grid);  // no outer loop across channels
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);

        reduce_t loc[ndim]; fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!infov_dyn<ndim>(extrapolate, loc, size_splinc+nbatch))
            continue;

        offset_t out_offset = index2offset<nbatch>(i, size_grid, stride_out);

        count(loc, out_offset);
    }
}

template <int nbatch, int ndim, bool abs,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
CUGLOB
void grad(
    bound::BoundVec  bnd,
    spline::SplineVec spl,
    int extrapolate,
    scalar_t * out,                 // (*batch, *spatial_grid, C, D) tensor | Placeholder for the pulled gradients
    const scalar_t * inp,           // (*batch, *spatial_spln, C) tensor    | Input volume
    const scalar_t * grid,          // (*batch, *spatial_grid, D) tensor    | Coordinates into the input volume
    const offset_t * _size_grid,    // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial_grid, C, D] vector
    const offset_t * _stride_inp,   // [*batch, *spatial_spln, C] vector
            const offset_t * _stride_grid)  // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;
    const bound_t  _bnd[3] = { bnd[0], bnd[1], bnd[2] };
    const spline_t _spl[3] = { spl[0], spl[1], spl[2] };

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+2]; fillfrom<nall+2>(stride_out,  _stride_out);
    offset_t stride_inp  [nall+1]; fillfrom<nall+1>(stride_inp,  _stride_inp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc  = size_splinc[nall];
    offset_t osc = stride_out[nall];
    offset_t osg = stride_out[nall+1];
    offset_t isc = stride_inp[nall];
    offset_t gsc = stride_grid[nall];

    auto grad = [&](const reduce_t * loc, offset_t out_offset, offset_t inp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ, abs>::grad(
            out + out_offset, inp + inp_offset,
            loc, size_splinc + nbatch, stride_inp + nbatch,
            nc, osc, isc, osg, _bnd, _spl);
    };

    offset_t numel = prod<nall>(size_grid);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t out_offset = index2offset<nall>(i, size_grid, stride_out);
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);

        reduce_t loc[ndim];  fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!infov_dyn<ndim>(extrapolate, loc, size_splinc + nbatch))
        {
            for (offset_t c=0; c<nc; ++c)
                fill<ndim>(out + out_offset + c * osc, 0, osg);
            continue;
        }

        offset_t inp_offset = index2offset<nbatch>(i, size_grid, stride_inp);

        grad(loc, out_offset, inp_offset);
    }
}

template <int nbatch, int ndim, bool abs,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
CUGLOB
void hess(
    bound::BoundVec  bnd,
    spline::SplineVec spl,
    int extrapolate,
    scalar_t * out,                 // (*batch, *spatial_grid, C, D) tensor | Placeholder for the pulled gradients
    const scalar_t * inp,           // (*batch, *spatial_spln, C) tensor    | Input volume
    const scalar_t * grid,          // (*batch, *spatial_grid, D) tensor    | Coordinates into the input volume
    const offset_t * _size_grid,    // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial_grid, C, D] vector
    const offset_t * _stride_inp,   // [*batch, *spatial_spln, C] vector
            const offset_t * _stride_grid)  // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;
    const bound_t  _bnd[3] = { bnd[0], bnd[1], bnd[2] };
    const spline_t _spl[3] = { spl[0], spl[1], spl[2] };

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+2]; fillfrom<nall+2>(stride_out,  _stride_out);
    offset_t stride_inp  [nall+1]; fillfrom<nall+1>(stride_inp,  _stride_inp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc  = size_splinc[nall];
    offset_t osc = stride_out[nall];
    offset_t osg = stride_out[nall+1];
    offset_t isc = stride_inp[nall];
    offset_t gsc = stride_grid[nall];

    auto eval = [&](const reduce_t * loc, offset_t out_offset, offset_t inp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ, abs>::hess(
            out + out_offset, inp + inp_offset,
            loc, size_splinc + nbatch, stride_inp + nbatch,
            nc, osc, isc, osg, _bnd, _spl);
    };

    offset_t numel = prod<nall>(size_grid);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t out_offset = index2offset<nall>(i, size_grid, stride_out);
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);

        reduce_t loc[ndim];  fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!infov_dyn<ndim>(extrapolate, loc, size_splinc + nbatch))
        {
            for (offset_t c=0; c<nc; ++c)
                fill<(ndim*(ndim+1))/2>(out + out_offset + c * osc, 0, osg);
            continue;
        }

        offset_t inp_offset = index2offset<nbatch>(i, size_grid, stride_inp);

        eval(loc, out_offset, inp_offset);
    }
}

template <int nbatch, int ndim, bool abs,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
CUGLOB
void pull_backward(
    bound::BoundVec  bnd,
    spline::SplineVec spl,
    int extrapolate,
    scalar_t * out,                 // (*batch, *spatial_spln, C) tensor | Placeholder for the gradient wrt `inp`
    scalar_t * gout,                // (*batch, *spatial_grid, D) tensor | Placeholder for the gradient wrt `grid`
    const scalar_t * inp,           // (*batch, *spatial_spln, C) tensor | Input volume of the forward pass
    const scalar_t * ginp,          // (*batch, *spatial_grid, C) tensor | Gradient wrt to the output of the forward pass
    const scalar_t * grid,          // (*batch, *spatial_grid, D) tensor | Coordinates into the input volume
    const offset_t * _size_grid,    // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_gout,  // [*batch, *spatial_grid, D] vector
    const offset_t * _stride_inp,   // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_ginp,  // [*batch, *spatial_grid, C] vector
    const offset_t * _stride_grid)  // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;
    const bound_t  _bnd[3] = { bnd[0], bnd[1], bnd[2] };
    const spline_t _spl[3] = { spl[0], spl[1], spl[2] };

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+1]; fillfrom<nall+1>(stride_out,  _stride_out);
    offset_t stride_gout [nall+1]; fillfrom<nall+1>(stride_gout, _stride_gout);
    offset_t stride_inp  [nall+1]; fillfrom<nall+1>(stride_inp,  _stride_inp);
    offset_t stride_ginp [nall+1]; fillfrom<nall+1>(stride_ginp, _stride_ginp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc  = size_splinc[nall];
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t isg = stride_ginp[nall];
    offset_t osg = stride_gout[nall];
    offset_t gsc = stride_grid[nall];

    auto pull_backward = [&](
        const reduce_t * loc,
        offset_t out_offset,
        offset_t gout_offset,
        offset_t inp_offset,
        offset_t ginp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ, abs>::pull_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            loc, size_splinc + nbatch,
            stride_out + nbatch, stride_inp + nbatch,
            nc, osc, isc, osg, isg, _bnd, _spl);
    };

    offset_t numel = prod<nall>(size_grid);  // no outer loop across channels
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);
        offset_t gout_offset = index2offset<nall>(i, size_grid, stride_gout);

        reduce_t loc[ndim];  fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!infov_dyn<ndim>(extrapolate, loc, size_splinc + nbatch))
        {
            fill<ndim>(gout + gout_offset, 0, osg);
            continue;
        }

        offset_t inp_offset  = index2offset<nbatch>(i, size_grid, stride_inp);
        offset_t out_offset  = index2offset<nbatch>(i, size_grid, stride_out);
        offset_t ginp_offset = index2offset<nall>(i, size_grid, stride_ginp);

        pull_backward(loc, out_offset, gout_offset, inp_offset, ginp_offset);
    }
}

template <int nbatch, int ndim, bool abs,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
CUGLOB
void push_backward(
    bound::BoundVec  bnd,
    spline::SplineVec spl,
    int extrapolate,
    scalar_t * out,                 // (*batch, *spatial_grid, C) tensor | Placeholder for the gradient wrt `inp`
    scalar_t * gout,                // (*batch, *spatial_grid, D) tensor | Placeholder for the gradient wrt `grid`
    const scalar_t * inp,           // (*batch, *spatial_grid, C) tensor | Input volume of the forward pass
    const scalar_t * ginp,          // (*batch, *spatial_spln, C) tensor | Gradient wrt the output of the forward pass
    const scalar_t * grid,          // (*batch, *spatial_grid, D) tensor | Coordinates into the output of the forward pass
    const offset_t * _size_grid,    // [*batch, *spatial_spln, C] vector
    const offset_t * _size_splinc,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_gout,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_inp,   // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_ginp,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_grid)  // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;
    const bound_t  _bnd[3] = { bnd[0], bnd[1], bnd[2] };
    const spline_t _spl[3] = { spl[0], spl[1], spl[2] };

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+1]; fillfrom<nall+1>(stride_out,  _stride_out);
    offset_t stride_gout [nall+1]; fillfrom<nall+1>(stride_gout,  _stride_gout);
    offset_t stride_inp  [nall+1]; fillfrom<nall+1>(stride_inp,  _stride_inp);
    offset_t stride_ginp [nall+1]; fillfrom<nall+1>(stride_ginp, _stride_ginp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc  = size_splinc[nall];
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    offset_t isg = stride_ginp[nall];
    offset_t osg = stride_gout[nall];
    offset_t gsc = stride_grid[nall];

    auto push_backward = [&](
        const reduce_t * loc,
        offset_t out_offset,
        offset_t gout_offset,
        offset_t inp_offset,
        offset_t ginp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ, abs>::push_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            loc, size_splinc + nbatch, stride_inp + nbatch,
            nc, osc, isc, osg, isg, _bnd, _spl);
    };

    offset_t numel = prod<nall>(size_grid);  // no outer loop across channels
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);
        offset_t out_offset  = index2offset<nall>(i, size_grid, stride_out);
        offset_t gout_offset = index2offset<nall>(i, size_grid, stride_gout);

        reduce_t loc[ndim];  fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!infov_dyn<ndim>(extrapolate, loc, size_splinc+nbatch))
        {
            for (offset_t c=0; c<nc; ++c)
                out[out_offset + c * osc] = static_cast<scalar_t>(0);
            fill<ndim>(gout + gout_offset, 0, osg);
            continue;
        }

        offset_t inp_offset = index2offset<nall>(i, size_grid, stride_inp);
        offset_t ginp_offset = index2offset<nbatch>(i, size_grid, stride_ginp);
        push_backward(loc, out_offset, gout_offset, inp_offset, ginp_offset);
    }
}


template <int nbatch, int ndim, bool abs,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
CUGLOB
void count_backward(
    bound::BoundVec  bnd,
    spline::SplineVec spl,
    int extrapolate,
    scalar_t * gout,                // (*batch, *spatial_grid, D) tensor | Placeholder for the gradient wrt `grid`
    const scalar_t * ginp,          // (*batch, *spatial_spln, 1) tensor | Gradient wrt to the output of the forward pass
    const scalar_t * grid,          // (*batch, *spatial_grid, D) tensor | Coordinates into the output of the forward pass
    const offset_t * _size_grid,    // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc,  // [*batch, *spatial_spln, 1] vector
    const offset_t * _stride_gout,  // [*batch, *spatial_grid, D] vector
    const offset_t * _stride_ginp,  // [*batch, *spatial_spln, 1] vector
    const offset_t * _stride_grid)  // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;
    const bound_t  _bnd[3] = { bnd[0], bnd[1], bnd[2] };
    const spline_t _spl[3] = { spl[0], spl[1], spl[2] };

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_gout [nall+1]; fillfrom<nall+1>(stride_gout, _stride_gout);
    offset_t stride_ginp [nall+1]; fillfrom<nall+1>(stride_ginp, _stride_ginp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc  = size_splinc[nall];
    offset_t osg = stride_gout[nall];
    offset_t gsc = stride_grid[nall];

    auto count_backward = [&](
        const reduce_t * loc,
        offset_t gout_offset,
        offset_t ginp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ, abs>::count_backward(
            gout + gout_offset, ginp + ginp_offset,
            loc, size_splinc + nbatch, stride_ginp + nbatch, osg, _bnd, _spl);
    };

    offset_t numel = prod<nall>(size_grid);  // no outer loop across channels
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t grid_offset = index2offset<nall>(i, size_grid, stride_grid);
        offset_t gout_offset = index2offset<nall>(i, size_grid, stride_gout);

        reduce_t loc[ndim];  fillfrom<ndim>(loc, grid + grid_offset, gsc);
        if (!infov_dyn<ndim>(extrapolate, loc, size_splinc + nbatch))
        {
            fill<ndim>(gout + gout_offset, 0, osg);
            continue;
        }

        offset_t ginp_offset = index2offset<nbatch>(i, size_grid, stride_ginp);
        count_backward(loc, gout_offset, ginp_offset);
    }
}

template <int nbatch, int ndim, bool abs,
          typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
CUGLOB
void grad_backward(
    bound::BoundVec  bnd,
    spline::SplineVec spl,
    int extrapolate,
    scalar_t * out,                 // (*batch, *spatial_spln, C) tensor    | Placeholder for the gradient wrt `inp`
    scalar_t * gout,                // (*batch, *spatial_grid, D) tensor    | Placeholder for the gradient wrt `grid`
    const scalar_t * inp,           // (*batch, *spatial_spln, C) tensor    | Input of the forward pass
    const scalar_t * ginp,          // (*batch, *spatial_grid, C, D) tensor | Gradient wrt the output of the forward pass
    const scalar_t * grid,          // (*batch, *spatial_grid, D) tensor    | Coordinates into the input volume
    const offset_t * _size_grid,    // [*batch, *spatial_grid, D] vector
    const offset_t * _size_splinc,  // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_out,   // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_gout,  // [*batch, *spatial_grid, D] vector
    const offset_t * _stride_inp,   // [*batch, *spatial_spln, C] vector
    const offset_t * _stride_ginp,  // [*batch, *spatial_grid, C, D] vector
    const offset_t * _stride_grid)  // [*batch, *spatial_grid, D] vector
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    static constexpr int nall = ndim + nbatch;
    const bound_t  _bnd[3] = { bnd[0], bnd[1], bnd[2] };
    const spline_t _spl[3] = { spl[0], spl[1], spl[2] };

    // copy vectors to the stack
    offset_t size_grid   [nall+1]; fillfrom<nall+1>(size_grid,   _size_grid);
    offset_t size_splinc [nall+1]; fillfrom<nall+1>(size_splinc, _size_splinc);
    offset_t stride_out  [nall+1]; fillfrom<nall+1>(stride_out,  _stride_out);
    offset_t stride_gout [nall+1]; fillfrom<nall+1>(stride_gout, _stride_gout);
    offset_t stride_inp  [nall+1]; fillfrom<nall+1>(stride_inp,  _stride_inp);
    offset_t stride_ginp [nall+2]; fillfrom<nall+2>(stride_ginp, _stride_ginp);
    offset_t stride_grid [nall+1]; fillfrom<nall+1>(stride_grid, _stride_grid);
    offset_t nc   = size_splinc[nall];
    offset_t osc  = stride_out[nall];
    offset_t isc  = stride_inp[nall];
    offset_t isg  = stride_ginp[nall+1];
    offset_t gsc  = stride_ginp[nall];
    offset_t osg  = stride_gout[nall];
    offset_t grsc = stride_grid[nall];

    auto grad_backward = [&](
        const reduce_t * loc,
        offset_t out_offset,
        offset_t gout_offset,
        offset_t inp_offset,
        offset_t ginp_offset)
    {
        return PushPull<ndim, IX, BX, IY, BY, IZ, BZ, abs>::grad_backward(
            out + out_offset, gout + gout_offset,
            inp + inp_offset, ginp + ginp_offset,
            loc, size_splinc + nbatch,
            stride_out + nbatch, stride_inp + nbatch,
            nc, osc, isc, gsc, osg, isg, _bnd, _spl);
    };

    auto get_grid_offset = [&](offset_t i) {
        return index2offset<nall>(i, size_grid, stride_grid); };
    auto get_gout_offset = [&](offset_t i) {
        return index2offset<nall>(i, size_grid, stride_gout); };
    auto get_inp_offset = [&](offset_t i) {
        return index2offset<nbatch>(i, size_grid, stride_inp); };
    auto get_out_offset = [&](offset_t i) {
        return index2offset<nbatch>(i, size_grid, stride_out); };
    auto get_ginp_offset = [&](offset_t i) {
        return index2offset<nall>(i, size_grid, stride_ginp); };

    offset_t numel = prod<nall>(size_grid);  // no outer loop across channels
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t grid_offset = get_grid_offset(i);
        offset_t gout_offset = get_gout_offset(i);

        reduce_t loc[ndim];  fillfrom<ndim>(loc, grid + grid_offset, grsc);
        if (!infov_dyn<ndim>(extrapolate, loc, size_splinc + nbatch))
        {
            fill<ndim>(gout + gout_offset, 0, osg);
            continue;
        }

        grad_backward(loc, get_out_offset(i), gout_offset,
                      get_inp_offset(i), get_ginp_offset(i));
    }
}

/***********************************************************************
 *                          HOST LAUNCHERS                             *
 *                                                                     *
 * These mirror the fastfields-cpu-impl pushpull launchers, but launch *
 * the CUGLOB kernels above over the grid. `nbatch`, `extrapolate` and *
 * the bound/spline conditions are runtime arguments here exactly as   *
 * on the CPU side; only `ndim`, `abs` and whichever bound/spline axes *
 * the build compiles statically remain compile-time template          *
 * parameters, so the launcher only needs to dispatch the runtime      *
 * `nbatch` to its matching compile-time specialisation.                *
 *                                                                     *
 * `extrapolate` used to be a compile-time parameter too (dispatched   *
 * via FF_PP_EX below the runtime value {+1, 0, -1}), tripling every   *
 * pushpull kernel instantiation for a check that is a cheap runtime   *
 * branch on the CPU side and costs nothing to make runtime here       *
 * either -- removed; `nbatch` is bounded by FF_PP_MAX_NBATCH, larger  *
 * batch ranks throw std::logic_error.                                 *
 ***********************************************************************/

// Number of leading batch dimensions the device path instantiates.
#ifndef FF_PP_MAX_NBATCH
#define FF_PP_MAX_NBATCH 0
#endif

// Only instantiate the batch ranks up to FF_PP_MAX_NBATCH (each additional
// rank multiplies the already-large spline x bound x dtype x offset matrix of
// device-kernel specialisations, so the bound keeps nvcc compile time finite).
#if   FF_PP_MAX_NBATCH <= 0
#define FF_PP_NB_EXTRA(LAUNCH)
#elif FF_PP_MAX_NBATCH == 1
#define FF_PP_NB_EXTRA(LAUNCH) case 1: LAUNCH(1); break;
#elif FF_PP_MAX_NBATCH == 2
#define FF_PP_NB_EXTRA(LAUNCH) case 1: LAUNCH(1); break;                      \
                               case 2: LAUNCH(2); break;
#elif FF_PP_MAX_NBATCH == 3
#define FF_PP_NB_EXTRA(LAUNCH) case 1: LAUNCH(1); break;                      \
                               case 2: LAUNCH(2); break;                      \
                               case 3: LAUNCH(3); break;
#else
#define FF_PP_NB_EXTRA(LAUNCH) case 1: LAUNCH(1); break;                      \
                               case 2: LAUNCH(2); break;                      \
                               case 3: LAUNCH(3); break;                      \
                               case 4: LAUNCH(4); break;
#endif

// Dispatch the runtime `nbatch` to a compile-time constant.
#define FF_PP_DISPATCH(LAUNCH)                                                \
    switch (nbatch) {                                                         \
        case 0: LAUNCH(0); break;                                             \
        FF_PP_NB_EXTRA(LAUNCH)                                                \
        default: throw std::logic_error(                                      \
            "ff::cuda::pushpull: batch rank exceeds the compiled maximum "    \
            "(FF_PP_MAX_NBATCH)");                                            \
    }

// int -> cudaStream_t (0 == default stream).
CUHOST inline cudaStream_t _pp_stream(int stream)
{
    return reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream));
}

// -------------------------------------------------------------------- pull
template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
CUHOST void pull(
          bound::BoundVec  bnd,
          spline::SplineVec spl,
          offset_t   nbatch,
          int        extrapolate,
          scalar_t * out,
    const scalar_t * inp,
    const scalar_t * grid,
    const offset_t * size_grid,
    const offset_t * size_splinc,
    const offset_t * stride_out,
    const offset_t * stride_inp,
    const offset_t * stride_grid,
          int        stream = 0)
{
    const offset_t nall = ndim + nbatch;
    const offset_t n1   = nall + 1;
    offset_t numel = 1;
    for (offset_t d = 0; d < nall; ++d) numel *= size_grid[d];
    cudaStream_t cstream = _pp_stream(stream);

    offset_t * d_sg = nullptr, * d_ss = nullptr, * d_so = nullptr,
             * d_si = nullptr, * d_sgr = nullptr;
    try
    {
        d_sg  = copyToDevice(size_grid,   n1);
        d_ss  = copyToDevice(size_splinc, n1);
        d_so  = copyToDevice(stride_out,  n1);
        d_si  = copyToDevice(stride_inp,  n1);
        d_sgr = copyToDevice(stride_grid, n1);

#define FF_PP_PULL(NB)                                                       \
        pull<NB, ndim, reduce_t, scalar_t, offset_t,                         \
             IX, BX, IY, BY, IZ, BZ>                                         \
            <<<GET_BLOCKS(numel), CUDA_NUM_THREADS, 0, cstream>>>            \
            (bnd, spl, extrapolate, out, inp, grid, d_sg, d_ss, d_so, d_si, d_sgr)
        FF_PP_DISPATCH(FF_PP_PULL);
#undef FF_PP_PULL
    }
    catch (const std::exception &exc)
    {
        freeDevice(d_sg, d_ss, d_so, d_si, d_sgr);
        throw exc;
    }
    freeDevice(d_sg, d_ss, d_so, d_si, d_sgr);
}

// -------------------------------------------------------------------- push
template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
CUHOST void push(
          bound::BoundVec  bnd,
          spline::SplineVec spl,
          offset_t   nbatch,
          int        extrapolate,
          scalar_t * out,          // must be pre-zeroed by the caller
    const scalar_t * inp,
    const scalar_t * grid,
    const offset_t * size_grid,
    const offset_t * size_splinc,
    const offset_t * stride_out,
    const offset_t * stride_inp,
    const offset_t * stride_grid,
          int        stream = 0)
{
    const offset_t nall = ndim + nbatch;
    const offset_t n1   = nall + 1;
    offset_t numel = 1;
    for (offset_t d = 0; d < nall; ++d) numel *= size_grid[d];
    cudaStream_t cstream = _pp_stream(stream);

    offset_t * d_sg = nullptr, * d_ss = nullptr, * d_so = nullptr,
             * d_si = nullptr, * d_sgr = nullptr;
    try
    {
        d_sg  = copyToDevice(size_grid,   n1);
        d_ss  = copyToDevice(size_splinc, n1);
        d_so  = copyToDevice(stride_out,  n1);
        d_si  = copyToDevice(stride_inp,  n1);
        d_sgr = copyToDevice(stride_grid, n1);

#define FF_PP_PUSH(NB)                                                       \
        push<NB, ndim, reduce_t, scalar_t, offset_t,                         \
             IX, BX, IY, BY, IZ, BZ>                                         \
            <<<GET_BLOCKS(numel), CUDA_NUM_THREADS, 0, cstream>>>            \
            (bnd, spl, extrapolate, out, inp, grid, d_sg, d_ss, d_so, d_si, d_sgr)
        FF_PP_DISPATCH(FF_PP_PUSH);
#undef FF_PP_PUSH
    }
    catch (const std::exception &exc)
    {
        freeDevice(d_sg, d_ss, d_so, d_si, d_sgr);
        throw exc;
    }
    freeDevice(d_sg, d_ss, d_so, d_si, d_sgr);
}

// ------------------------------------------------------------------- count
template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
CUHOST void count(
          bound::BoundVec  bnd,
          spline::SplineVec spl,
          offset_t   nbatch,
          int        extrapolate,
          scalar_t * out,          // must be pre-zeroed by the caller
    const scalar_t * grid,
    const offset_t * size_grid,
    const offset_t * size_splinc,
    const offset_t * stride_out,
    const offset_t * stride_grid,
          int        stream = 0)
{
    const offset_t nall = ndim + nbatch;
    const offset_t n1   = nall + 1;
    offset_t numel = 1;
    for (offset_t d = 0; d < nall; ++d) numel *= size_grid[d];
    cudaStream_t cstream = _pp_stream(stream);

    offset_t * d_sg = nullptr, * d_ss = nullptr,
             * d_so = nullptr, * d_sgr = nullptr;
    try
    {
        d_sg  = copyToDevice(size_grid,   n1);
        d_ss  = copyToDevice(size_splinc, n1);
        d_so  = copyToDevice(stride_out,  n1);
        d_sgr = copyToDevice(stride_grid, n1);

#define FF_PP_COUNT(NB)                                                      \
        count<NB, ndim, reduce_t, scalar_t, offset_t,                        \
              IX, BX, IY, BY, IZ, BZ>                                        \
            <<<GET_BLOCKS(numel), CUDA_NUM_THREADS, 0, cstream>>>            \
            (bnd, spl, extrapolate, out, grid, d_sg, d_ss, d_so, d_sgr)
        FF_PP_DISPATCH(FF_PP_COUNT);
#undef FF_PP_COUNT
    }
    catch (const std::exception &exc)
    {
        freeDevice(d_sg, d_ss, d_so, d_sgr);
        throw exc;
    }
    freeDevice(d_sg, d_ss, d_so, d_sgr);
}

// -------------------------------------------------------------------- grad
template <int ndim, bool abs, typename reduce_t, typename scalar_t, typename offset_t,
          spline::type IX,    bound::type BX,
          spline::type IY=IX, bound::type BY=BX,
          spline::type IZ=IY, bound::type BZ=BY>
CUHOST void grad(
          bound::BoundVec  bnd,
          spline::SplineVec spl,
          offset_t   nbatch,
          int        extrapolate,
          scalar_t * out,
    const scalar_t * inp,
    const scalar_t * grid,
    const offset_t * size_grid,
    const offset_t * size_splinc,
    const offset_t * stride_out,   // has an extra trailing (D) axis: length nall+2
    const offset_t * stride_inp,
    const offset_t * stride_grid,
          int        stream = 0)
{
    const offset_t nall = ndim + nbatch;
    const offset_t n1   = nall + 1;
    offset_t numel = 1;
    for (offset_t d = 0; d < nall; ++d) numel *= size_grid[d];
    cudaStream_t cstream = _pp_stream(stream);

    offset_t * d_sg = nullptr, * d_ss = nullptr, * d_so = nullptr,
             * d_si = nullptr, * d_sgr = nullptr;
    try
    {
        d_sg  = copyToDevice(size_grid,   n1);
        d_ss  = copyToDevice(size_splinc, n1);
        d_so  = copyToDevice(stride_out,  n1 + 1);   // extra (D) axis
        d_si  = copyToDevice(stride_inp,  n1);
        d_sgr = copyToDevice(stride_grid, n1);

#define FF_PP_GRAD(NB)                                                       \
        grad<NB, ndim, abs, reduce_t, scalar_t, offset_t,                    \
             IX, BX, IY, BY, IZ, BZ>                                         \
            <<<GET_BLOCKS(numel), CUDA_NUM_THREADS, 0, cstream>>>            \
            (bnd, spl, extrapolate, out, inp, grid, d_sg, d_ss, d_so, d_si, d_sgr)
        FF_PP_DISPATCH(FF_PP_GRAD);
#undef FF_PP_GRAD
    }
    catch (const std::exception &exc)
    {
        freeDevice(d_sg, d_ss, d_so, d_si, d_sgr);
        throw exc;
    }
    freeDevice(d_sg, d_ss, d_so, d_si, d_sgr);
}

#undef FF_PP_DISPATCH
#undef FF_PP_NB_EXTRA

FF_NAMESPACE_END(pushpull)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
