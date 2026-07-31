#include "pushpull.h"
// VOIDPTR / CHECK_* / DISPATCH_PP and the reduce_t typedef, shared with
// pushpull_backward.cpp so the two translation units cannot drift apart on
// which (order, bound) pairs are statically instantiated.
#include "pushpull_dispatch.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

/***********************************************************************
 *                          LEAF WRAPPERS                             *
 ***********************************************************************/

namespace {

// Narrow the [n1]-length shape/stride arrays to `offset_t`, cast the data
// pointers and call the templated impl. `n1 == nbatch + ndim + 1`.
template <int ndim, spline::type I, bound::type B,
          typename scalar_t, typename offset_t>
inline void _pull(
    const bound::BoundVec  & bvec, const spline::SplineVec & svec,
          int64_t   nbatch, int64_t n1, int extrapolate,
          void    * out, const void * inp, const void * grid,
    const int64_t * size_grid,  const int64_t * size_splinc,
    const int64_t * stride_out, const int64_t * stride_inp, const int64_t * stride_grid)
{
    const offset_t * _sg  = copy_if_needed<offset_t *>(size_grid,   n1);
    const offset_t * _ss  = copy_if_needed<offset_t *>(size_splinc, n1);
    const offset_t * _so  = copy_if_needed<offset_t *>(stride_out,  n1);
    const offset_t * _si  = copy_if_needed<offset_t *>(stride_inp,  n1);
    const offset_t * _sgr = copy_if_needed<offset_t *>(stride_grid, n1);
          scalar_t * _out  = static_cast<      scalar_t *>(out);
    const scalar_t * _inp  = static_cast<const scalar_t *>(inp);
    const scalar_t * _grid = static_cast<const scalar_t *>(grid);

    pushpull::pull<ndim, reduce_t, scalar_t, offset_t, I, B>(
        static_cast<offset_t>(nbatch), extrapolate, _out, _inp, _grid,
        _sg, _ss, _so, _si, _sgr, bvec, svec);

    free_if_needed<int64_t *>(_sg);  free_if_needed<int64_t *>(_ss);
    free_if_needed<int64_t *>(_so);  free_if_needed<int64_t *>(_si);
    free_if_needed<int64_t *>(_sgr);
}

template <int ndim, spline::type I, bound::type B,
          typename scalar_t, typename offset_t>
inline void _push(
    const bound::BoundVec  & bvec, const spline::SplineVec & svec,
          int64_t   nbatch, int64_t n1, int extrapolate,
          void    * out, const void * inp, const void * grid,
    const int64_t * size_grid,  const int64_t * size_splinc,
    const int64_t * stride_out, const int64_t * stride_inp, const int64_t * stride_grid)
{
    const offset_t * _sg  = copy_if_needed<offset_t *>(size_grid,   n1);
    const offset_t * _ss  = copy_if_needed<offset_t *>(size_splinc, n1);
    const offset_t * _so  = copy_if_needed<offset_t *>(stride_out,  n1);
    const offset_t * _si  = copy_if_needed<offset_t *>(stride_inp,  n1);
    const offset_t * _sgr = copy_if_needed<offset_t *>(stride_grid, n1);
          scalar_t * _out  = static_cast<      scalar_t *>(out);
    const scalar_t * _inp  = static_cast<const scalar_t *>(inp);
    const scalar_t * _grid = static_cast<const scalar_t *>(grid);

    pushpull::push<ndim, reduce_t, scalar_t, offset_t, I, B>(
        static_cast<offset_t>(nbatch), extrapolate, _out, _inp, _grid,
        _sg, _ss, _so, _si, _sgr, bvec, svec);

    free_if_needed<int64_t *>(_sg);  free_if_needed<int64_t *>(_ss);
    free_if_needed<int64_t *>(_so);  free_if_needed<int64_t *>(_si);
    free_if_needed<int64_t *>(_sgr);
}

template <int ndim, spline::type I, bound::type B,
          typename scalar_t, typename offset_t>
inline void _count(
    const bound::BoundVec  & bvec, const spline::SplineVec & svec,
          int64_t   nbatch, int64_t n1, int extrapolate,
          void    * out, const void * grid,
    const int64_t * size_grid,  const int64_t * size_splinc,
    const int64_t * stride_out, const int64_t * stride_grid)
{
    const offset_t * _sg  = copy_if_needed<offset_t *>(size_grid,   n1);
    const offset_t * _ss  = copy_if_needed<offset_t *>(size_splinc, n1);
    const offset_t * _so  = copy_if_needed<offset_t *>(stride_out,  n1);
    const offset_t * _sgr = copy_if_needed<offset_t *>(stride_grid, n1);
          scalar_t * _out  = static_cast<      scalar_t *>(out);
    const scalar_t * _grid = static_cast<const scalar_t *>(grid);

    pushpull::count<ndim, reduce_t, scalar_t, offset_t, I, B>(
        static_cast<offset_t>(nbatch), extrapolate, _out, _grid,
        _sg, _ss, _so, _sgr, bvec, svec);

    free_if_needed<int64_t *>(_sg);  free_if_needed<int64_t *>(_ss);
    free_if_needed<int64_t *>(_so);  free_if_needed<int64_t *>(_sgr);
}

// grad: out has an extra trailing (D) axis, so stride_out has length n2 = n1+1.
template <int ndim, spline::type I, bound::type B,
          typename scalar_t, typename offset_t>
inline void _grad(
    const bound::BoundVec  & bvec, const spline::SplineVec & svec,
          int64_t   nbatch, int64_t n1, int extrapolate, bool abs,
          void    * out, const void * inp, const void * grid,
    const int64_t * size_grid,  const int64_t * size_splinc,
    const int64_t * stride_out, const int64_t * stride_inp, const int64_t * stride_grid)
{
    const offset_t * _sg  = copy_if_needed<offset_t *>(size_grid,   n1);
    const offset_t * _ss  = copy_if_needed<offset_t *>(size_splinc, n1);
    const offset_t * _so  = copy_if_needed<offset_t *>(stride_out,  n1 + 1);
    const offset_t * _si  = copy_if_needed<offset_t *>(stride_inp,  n1);
    const offset_t * _sgr = copy_if_needed<offset_t *>(stride_grid, n1);
          scalar_t * _out  = static_cast<      scalar_t *>(out);
    const scalar_t * _inp  = static_cast<const scalar_t *>(inp);
    const scalar_t * _grid = static_cast<const scalar_t *>(grid);

    if (abs)
        pushpull::grad<ndim, true,  reduce_t, scalar_t, offset_t, I, B>(
            static_cast<offset_t>(nbatch), extrapolate, _out, _inp, _grid,
            _sg, _ss, _so, _si, _sgr, bvec, svec);
    else
        pushpull::grad<ndim, false, reduce_t, scalar_t, offset_t, I, B>(
            static_cast<offset_t>(nbatch), extrapolate, _out, _inp, _grid,
            _sg, _ss, _so, _si, _sgr, bvec, svec);

    free_if_needed<int64_t *>(_sg);  free_if_needed<int64_t *>(_ss);
    free_if_needed<int64_t *>(_so);  free_if_needed<int64_t *>(_si);
    free_if_needed<int64_t *>(_sgr);
}

} // anonymous namespace

/***********************************************************************
 *                              PULL                                  *
 ***********************************************************************/

void pull(
          DLTensor & out_,
    const DLTensor & inp_,
    const DLTensor & grid_,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          int        /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _inp(inp_), _grid(grid_);
    DLTensor       & out  = _out.t;
    const DLTensor & inp  = _inp.t;
    const DLTensor & grid = _grid.t;

    const int      ndim   = static_cast<int>(grid.shape[grid.ndim - 1]);
    const int32_t  nbatch = grid.ndim - ndim - 1;
    const int64_t  n1     = grid.ndim;   // nbatch + ndim + 1
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME_DTYPE(out, grid)
    CHECK_SAME(out.ndim,  grid.ndim, "out and grid must have the same rank")
    CHECK_SAME(inp.ndim,  grid.ndim, "inp and grid must have the same rank")
    if (nbatch < 0)
        throw std::invalid_argument("grid rank is too small for the coordinate dim");
    CHECK_SAME(out.shape[out.ndim-1], inp.shape[inp.ndim-1], "channel counts differ")
    CHECK_SAME_BATCH(out, grid, nbatch)
    CHECK_SAME_BATCH(inp, grid, nbatch)

    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp) && CANUSE32BITS(grid);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const spline_t spl  = static_cast<spline_t>(spline);
    const bound_t  bnd  = static_cast<bound_t >(bound);
    // Runtime carriers: read by whichever axes were instantiated as
    // `Dynamic` by the build policy, ignored by the static ones.
    const bound::BoundVec   bvec(bnd);
    const spline::SplineVec svec(spl);
    const int      ex   = static_cast<int>(extrapolate);

    DISPATCH_PP(_pull,
        bvec, svec,
        static_cast<int64_t>(nbatch), n1, ex,
        VOIDPTR(out), CVOIDPTR(inp), CVOIDPTR(grid),
        grid.shape, inp.shape,
        out.strides, inp.strides, grid.strides)
}

/***********************************************************************
 *                              PUSH                                  *
 ***********************************************************************/

void push(
          DLTensor & out_,
    const DLTensor & inp_,
    const DLTensor & grid_,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          int        /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _inp(inp_), _grid(grid_);
    DLTensor       & out  = _out.t;
    const DLTensor & inp  = _inp.t;
    const DLTensor & grid = _grid.t;

    const int      ndim   = static_cast<int>(grid.shape[grid.ndim - 1]);
    const int32_t  nbatch = grid.ndim - ndim - 1;
    const int64_t  n1     = grid.ndim;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME_DTYPE(out, grid)
    CHECK_SAME(out.ndim,  grid.ndim, "out and grid must have the same rank")
    CHECK_SAME(inp.ndim,  grid.ndim, "inp and grid must have the same rank")
    if (nbatch < 0)
        throw std::invalid_argument("grid rank is too small for the coordinate dim");
    CHECK_SAME(out.shape[out.ndim-1], inp.shape[inp.ndim-1], "channel counts differ")
    CHECK_SAME_BATCH(out, grid, nbatch)
    CHECK_SAME_BATCH(inp, grid, nbatch)

    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp) && CANUSE32BITS(grid);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const spline_t spl  = static_cast<spline_t>(spline);
    const bound_t  bnd  = static_cast<bound_t >(bound);
    // Runtime carriers: read by whichever axes were instantiated as
    // `Dynamic` by the build policy, ignored by the static ones.
    const bound::BoundVec   bvec(bnd);
    const spline::SplineVec svec(spl);
    const int      ex   = static_cast<int>(extrapolate);

    // size_splinc = out (the splatted volume); size_grid = grid.
    DISPATCH_PP(_push,
        bvec, svec,
        static_cast<int64_t>(nbatch), n1, ex,
        VOIDPTR(out), CVOIDPTR(inp), CVOIDPTR(grid),
        grid.shape, out.shape,
        out.strides, inp.strides, grid.strides)
}

/***********************************************************************
 *                              COUNT                                 *
 ***********************************************************************/

void count(
          DLTensor & out_,
    const DLTensor & grid_,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          int        /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _grid(grid_);
    DLTensor       & out  = _out.t;
    const DLTensor & grid = _grid.t;

    const int      ndim   = static_cast<int>(grid.shape[grid.ndim - 1]);
    const int32_t  nbatch = grid.ndim - ndim - 1;
    const int64_t  n1     = grid.ndim;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, grid)
    CHECK_SAME(out.ndim,  grid.ndim, "out and grid must have the same rank")
    if (nbatch < 0)
        throw std::invalid_argument("grid rank is too small for the coordinate dim");
    CHECK_SAME_BATCH(out, grid, nbatch)

    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(grid);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const spline_t spl  = static_cast<spline_t>(spline);
    const bound_t  bnd  = static_cast<bound_t >(bound);
    // Runtime carriers: read by whichever axes were instantiated as
    // `Dynamic` by the build policy, ignored by the static ones.
    const bound::BoundVec   bvec(bnd);
    const spline::SplineVec svec(spl);
    const int      ex   = static_cast<int>(extrapolate);

    DISPATCH_PP(_count,
        bvec, svec,
        static_cast<int64_t>(nbatch), n1, ex,
        VOIDPTR(out), CVOIDPTR(grid),
        grid.shape, out.shape,
        out.strides, grid.strides)
}

/***********************************************************************
 *                              GRAD                                  *
 ***********************************************************************/

void grad(
          DLTensor & out_,
    const DLTensor & inp_,
    const DLTensor & grid_,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          bool       abs,
          int        /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _inp(inp_), _grid(grid_);
    DLTensor       & out  = _out.t;
    const DLTensor & inp  = _inp.t;
    const DLTensor & grid = _grid.t;

    const int      ndim   = static_cast<int>(grid.shape[grid.ndim - 1]);
    const int32_t  nbatch = grid.ndim - ndim - 1;
    const int64_t  n1     = grid.ndim;   // grid/inp rank; out rank == n1 + 1
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME_DTYPE(out, grid)
    CHECK_SAME(inp.ndim,  grid.ndim, "inp and grid must have the same rank")
    CHECK_SAME(out.ndim,  grid.ndim + 1, "grad output must have an extra trailing axis")
    if (nbatch < 0)
        throw std::invalid_argument("grid rank is too small for the coordinate dim");
    CHECK_SAME(out.shape[out.ndim-1], ndim, "grad output trailing axis must equal ndim")
    CHECK_SAME(out.shape[out.ndim-2], inp.shape[inp.ndim-1], "channel counts differ")
    CHECK_SAME_BATCH(inp, grid, nbatch)

    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp) && CANUSE32BITS(grid);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const spline_t spl  = static_cast<spline_t>(spline);
    const bound_t  bnd  = static_cast<bound_t >(bound);
    // Runtime carriers: read by whichever axes were instantiated as
    // `Dynamic` by the build policy, ignored by the static ones.
    const bound::BoundVec   bvec(bnd);
    const spline::SplineVec svec(spl);
    const int      ex   = static_cast<int>(extrapolate);

    DISPATCH_PP(_grad,
        bvec, svec,
        static_cast<int64_t>(nbatch), n1, ex, abs,
        VOIDPTR(out), CVOIDPTR(inp), CVOIDPTR(grid),
        grid.shape, inp.shape,
        out.strides, inp.strides, grid.strides)
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
