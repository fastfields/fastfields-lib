// Adjoints of the pushpull ops (see pushpull.h for the calling contract).
//
// A separate translation unit from `pushpull.cpp` purely for build cost:
// these instantiate the same ndim x order x bound x dtype matrix a second
// time, and `pushpull` is already the most expensive module here (and by a
// wide margin on the CUDA side, where ptxas memory -- not just wall time --
// is the binding constraint; cf. the reg_field / reg_field_rls split).
#include "pushpull.h"
#include "pushpull_dispatch.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

/***********************************************************************
 *                          LEAF WRAPPERS                             *
 ***********************************************************************/

namespace {

// ---------------------------------------------------------------------
// Backward leaves.
//
// `abs` is a compile-time flag on the impl templates. For the pull/push/
// count adjoints it is pinned to `false` (see pushpull.h: a signed spline
// derivative is what an adjoint means); only `grad_backward` branches on
// it, mirroring `_grad`.
// ---------------------------------------------------------------------

template <int ndim, spline::type I, bound::type B,
          typename scalar_t, typename offset_t>
inline void _pull_backward(
    const bound::BoundVec  & bvec, const spline::SplineVec & svec,
          int64_t   nbatch, int64_t n1, int extrapolate,
          void    * out,  void * gout,
    const void    * inp,  const void * ginp, const void * grid,
    const int64_t * size_grid,   const int64_t * size_splinc,
    const int64_t * stride_out,  const int64_t * stride_gout,
    const int64_t * stride_inp,  const int64_t * stride_ginp,
    const int64_t * stride_grid)
{
    const offset_t * _sg   = copy_if_needed<offset_t *>(size_grid,   n1);
    const offset_t * _ss   = copy_if_needed<offset_t *>(size_splinc, n1);
    const offset_t * _so   = copy_if_needed<offset_t *>(stride_out,  n1);
    const offset_t * _sgo  = copy_if_needed<offset_t *>(stride_gout, n1);
    const offset_t * _si   = copy_if_needed<offset_t *>(stride_inp,  n1);
    const offset_t * _sgi  = copy_if_needed<offset_t *>(stride_ginp, n1);
    const offset_t * _sgr  = copy_if_needed<offset_t *>(stride_grid, n1);
          scalar_t * _out  = static_cast<      scalar_t *>(out);
          scalar_t * _gout = static_cast<      scalar_t *>(gout);
    const scalar_t * _inp  = static_cast<const scalar_t *>(inp);
    const scalar_t * _ginp = static_cast<const scalar_t *>(ginp);
    const scalar_t * _grid = static_cast<const scalar_t *>(grid);

    pushpull::pull_backward<ndim, false, reduce_t, scalar_t, offset_t, I, B>(
        static_cast<offset_t>(nbatch), extrapolate, _out, _gout,
        _inp, _ginp, _grid,
        _sg, _ss, _so, _sgo, _si, _sgi, _sgr, bvec, svec);

    free_if_needed<int64_t *>(_sg);   free_if_needed<int64_t *>(_ss);
    free_if_needed<int64_t *>(_so);   free_if_needed<int64_t *>(_sgo);
    free_if_needed<int64_t *>(_si);   free_if_needed<int64_t *>(_sgi);
    free_if_needed<int64_t *>(_sgr);
}

template <int ndim, spline::type I, bound::type B,
          typename scalar_t, typename offset_t>
inline void _push_backward(
    const bound::BoundVec  & bvec, const spline::SplineVec & svec,
          int64_t   nbatch, int64_t n1, int extrapolate,
          void    * out,  void * gout,
    const void    * inp,  const void * ginp, const void * grid,
    const int64_t * size_grid,   const int64_t * size_splinc,
    const int64_t * stride_out,  const int64_t * stride_gout,
    const int64_t * stride_inp,  const int64_t * stride_ginp,
    const int64_t * stride_grid)
{
    const offset_t * _sg   = copy_if_needed<offset_t *>(size_grid,   n1);
    const offset_t * _ss   = copy_if_needed<offset_t *>(size_splinc, n1);
    const offset_t * _so   = copy_if_needed<offset_t *>(stride_out,  n1);
    const offset_t * _sgo  = copy_if_needed<offset_t *>(stride_gout, n1);
    const offset_t * _si   = copy_if_needed<offset_t *>(stride_inp,  n1);
    const offset_t * _sgi  = copy_if_needed<offset_t *>(stride_ginp, n1);
    const offset_t * _sgr  = copy_if_needed<offset_t *>(stride_grid, n1);
          scalar_t * _out  = static_cast<      scalar_t *>(out);
          scalar_t * _gout = static_cast<      scalar_t *>(gout);
    const scalar_t * _inp  = static_cast<const scalar_t *>(inp);
    const scalar_t * _ginp = static_cast<const scalar_t *>(ginp);
    const scalar_t * _grid = static_cast<const scalar_t *>(grid);

    pushpull::push_backward<ndim, false, reduce_t, scalar_t, offset_t, I, B>(
        static_cast<offset_t>(nbatch), extrapolate, _out, _gout,
        _inp, _ginp, _grid,
        _sg, _ss, _so, _sgo, _si, _sgi, _sgr, bvec, svec);

    free_if_needed<int64_t *>(_sg);   free_if_needed<int64_t *>(_ss);
    free_if_needed<int64_t *>(_so);   free_if_needed<int64_t *>(_sgo);
    free_if_needed<int64_t *>(_si);   free_if_needed<int64_t *>(_sgi);
    free_if_needed<int64_t *>(_sgr);
}

template <int ndim, spline::type I, bound::type B,
          typename scalar_t, typename offset_t>
inline void _count_backward(
    const bound::BoundVec  & bvec, const spline::SplineVec & svec,
          int64_t   nbatch, int64_t n1, int extrapolate,
          void    * gout, const void * ginp, const void * grid,
    const int64_t * size_grid,   const int64_t * size_splinc,
    const int64_t * stride_gout, const int64_t * stride_ginp,
    const int64_t * stride_grid)
{
    const offset_t * _sg   = copy_if_needed<offset_t *>(size_grid,   n1);
    const offset_t * _ss   = copy_if_needed<offset_t *>(size_splinc, n1);
    const offset_t * _sgo  = copy_if_needed<offset_t *>(stride_gout, n1);
    const offset_t * _sgi  = copy_if_needed<offset_t *>(stride_ginp, n1);
    const offset_t * _sgr  = copy_if_needed<offset_t *>(stride_grid, n1);
          scalar_t * _gout = static_cast<      scalar_t *>(gout);
    const scalar_t * _ginp = static_cast<const scalar_t *>(ginp);
    const scalar_t * _grid = static_cast<const scalar_t *>(grid);

    pushpull::count_backward<ndim, false, reduce_t, scalar_t, offset_t, I, B>(
        static_cast<offset_t>(nbatch), extrapolate, _gout, _ginp, _grid,
        _sg, _ss, _sgo, _sgi, _sgr, bvec, svec);

    free_if_needed<int64_t *>(_sg);   free_if_needed<int64_t *>(_ss);
    free_if_needed<int64_t *>(_sgo);  free_if_needed<int64_t *>(_sgi);
    free_if_needed<int64_t *>(_sgr);
}

// grad_backward: `ginp` carries the extra trailing (D) axis of `grad`'s
// output, so `stride_ginp` has length n1 + 1 (cf. `_grad`'s stride_out).
template <int ndim, spline::type I, bound::type B,
          typename scalar_t, typename offset_t>
inline void _grad_backward(
    const bound::BoundVec  & bvec, const spline::SplineVec & svec,
          int64_t   nbatch, int64_t n1, int extrapolate, bool abs,
          void    * out,  void * gout,
    const void    * inp,  const void * ginp, const void * grid,
    const int64_t * size_grid,   const int64_t * size_splinc,
    const int64_t * stride_out,  const int64_t * stride_gout,
    const int64_t * stride_inp,  const int64_t * stride_ginp,
    const int64_t * stride_grid)
{
    const offset_t * _sg   = copy_if_needed<offset_t *>(size_grid,   n1);
    const offset_t * _ss   = copy_if_needed<offset_t *>(size_splinc, n1);
    const offset_t * _so   = copy_if_needed<offset_t *>(stride_out,  n1);
    const offset_t * _sgo  = copy_if_needed<offset_t *>(stride_gout, n1);
    const offset_t * _si   = copy_if_needed<offset_t *>(stride_inp,  n1);
    const offset_t * _sgi  = copy_if_needed<offset_t *>(stride_ginp, n1 + 1);
    const offset_t * _sgr  = copy_if_needed<offset_t *>(stride_grid, n1);
          scalar_t * _out  = static_cast<      scalar_t *>(out);
          scalar_t * _gout = static_cast<      scalar_t *>(gout);
    const scalar_t * _inp  = static_cast<const scalar_t *>(inp);
    const scalar_t * _ginp = static_cast<const scalar_t *>(ginp);
    const scalar_t * _grid = static_cast<const scalar_t *>(grid);

    if (abs)
        pushpull::grad_backward<ndim, true,  reduce_t, scalar_t, offset_t, I, B>(
            static_cast<offset_t>(nbatch), extrapolate, _out, _gout,
            _inp, _ginp, _grid,
            _sg, _ss, _so, _sgo, _si, _sgi, _sgr, bvec, svec);
    else
        pushpull::grad_backward<ndim, false, reduce_t, scalar_t, offset_t, I, B>(
            static_cast<offset_t>(nbatch), extrapolate, _out, _gout,
            _inp, _ginp, _grid,
            _sg, _ss, _so, _sgo, _si, _sgi, _sgr, bvec, svec);

    free_if_needed<int64_t *>(_sg);   free_if_needed<int64_t *>(_ss);
    free_if_needed<int64_t *>(_so);   free_if_needed<int64_t *>(_sgo);
    free_if_needed<int64_t *>(_si);   free_if_needed<int64_t *>(_sgi);
    free_if_needed<int64_t *>(_sgr);
}

} // anonymous namespace

/***********************************************************************
 *                          PULL BACKWARD                              *
 ***********************************************************************/

void pull_backward(
          DLTensor & out_,
          DLTensor & gout_,
    const DLTensor & inp_,
    const DLTensor & ginp_,
    const DLTensor & grid_,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          int        /* stream <unused> */
)
{
    ContiguousStrides _out(out_), _gout(gout_), _inp(inp_), _ginp(ginp_), _grid(grid_);
    DLTensor       & out  = _out.t;
    DLTensor       & gout = _gout.t;
    const DLTensor & inp  = _inp.t;
    const DLTensor & ginp = _ginp.t;
    const DLTensor & grid = _grid.t;

    const int      ndim   = static_cast<int>(grid.shape[grid.ndim - 1]);
    const int32_t  nbatch = grid.ndim - ndim - 1;
    const int64_t  n1     = grid.ndim;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, gout)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME_DTYPE(out, ginp)
    CHECK_SAME_DTYPE(out, grid)
    CHECK_SAME(out.ndim,  grid.ndim, "out and grid must have the same rank")
    CHECK_SAME(gout.ndim, grid.ndim, "gout and grid must have the same rank")
    CHECK_SAME(inp.ndim,  grid.ndim, "inp and grid must have the same rank")
    CHECK_SAME(ginp.ndim, grid.ndim, "ginp and grid must have the same rank")
    if (nbatch < 0)
        throw std::invalid_argument("grid rank is too small for the coordinate dim");
    // `out` mirrors `inp` (the field), `gout` mirrors `grid`.
    for (int32_t d = 0; d < out.ndim; ++d)
        CHECK_SAME(out.shape[d], inp.shape[d], "out and inp must have the same shape")
    for (int32_t d = 0; d < gout.ndim; ++d)
        CHECK_SAME(gout.shape[d], grid.shape[d], "gout and grid must have the same shape")
    CHECK_SAME(ginp.shape[ginp.ndim-1], inp.shape[inp.ndim-1], "channel counts differ")
    CHECK_SAME_BATCH(inp,  grid, nbatch)
    CHECK_SAME_BATCH(ginp, grid, nbatch)

    const bool     use_32bits = CANUSE32BITS(out)  && CANUSE32BITS(gout)
                             && CANUSE32BITS(inp)  && CANUSE32BITS(ginp)
                             && CANUSE32BITS(grid);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const spline_t spl  = static_cast<spline_t>(spline);
    const bound_t  bnd  = static_cast<bound_t >(bound);
    const bound::BoundVec   bvec(bnd);
    const spline::SplineVec svec(spl);
    const int      ex   = static_cast<int>(extrapolate);

    DISPATCH_PP(_pull_backward,
        bvec, svec,
        static_cast<int64_t>(nbatch), n1, ex,
        VOIDPTR(out), VOIDPTR(gout),
        CVOIDPTR(inp), CVOIDPTR(ginp), CVOIDPTR(grid),
        grid.shape, inp.shape,
        out.strides, gout.strides, inp.strides, ginp.strides, grid.strides)
}

/***********************************************************************
 *                          PUSH BACKWARD                              *
 ***********************************************************************/

void push_backward(
          DLTensor & out_,
          DLTensor & gout_,
    const DLTensor & inp_,
    const DLTensor & ginp_,
    const DLTensor & grid_,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          int        /* stream <unused> */
)
{
    ContiguousStrides _out(out_), _gout(gout_), _inp(inp_), _ginp(ginp_), _grid(grid_);
    DLTensor       & out  = _out.t;
    DLTensor       & gout = _gout.t;
    const DLTensor & inp  = _inp.t;
    const DLTensor & ginp = _ginp.t;
    const DLTensor & grid = _grid.t;

    const int      ndim   = static_cast<int>(grid.shape[grid.ndim - 1]);
    const int32_t  nbatch = grid.ndim - ndim - 1;
    const int64_t  n1     = grid.ndim;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, gout)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME_DTYPE(out, ginp)
    CHECK_SAME_DTYPE(out, grid)
    CHECK_SAME(out.ndim,  grid.ndim, "out and grid must have the same rank")
    CHECK_SAME(gout.ndim, grid.ndim, "gout and grid must have the same rank")
    CHECK_SAME(inp.ndim,  grid.ndim, "inp and grid must have the same rank")
    CHECK_SAME(ginp.ndim, grid.ndim, "ginp and grid must have the same rank")
    if (nbatch < 0)
        throw std::invalid_argument("grid rank is too small for the coordinate dim");
    // Here both `out` and `inp` are grid-shaped; `ginp` is the field.
    for (int32_t d = 0; d < out.ndim; ++d)
        CHECK_SAME(out.shape[d], inp.shape[d], "out and inp must have the same shape")
    for (int32_t d = 0; d < gout.ndim; ++d)
        CHECK_SAME(gout.shape[d], grid.shape[d], "gout and grid must have the same shape")
    CHECK_SAME(ginp.shape[ginp.ndim-1], inp.shape[inp.ndim-1], "channel counts differ")
    CHECK_SAME_BATCH(inp,  grid, nbatch)
    CHECK_SAME_BATCH(ginp, grid, nbatch)

    const bool     use_32bits = CANUSE32BITS(out)  && CANUSE32BITS(gout)
                             && CANUSE32BITS(inp)  && CANUSE32BITS(ginp)
                             && CANUSE32BITS(grid);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const spline_t spl  = static_cast<spline_t>(spline);
    const bound_t  bnd  = static_cast<bound_t >(bound);
    const bound::BoundVec   bvec(bnd);
    const spline::SplineVec svec(spl);
    const int      ex   = static_cast<int>(extrapolate);

    // size_splinc is the *pushed volume*, i.e. ginp's shape.
    DISPATCH_PP(_push_backward,
        bvec, svec,
        static_cast<int64_t>(nbatch), n1, ex,
        VOIDPTR(out), VOIDPTR(gout),
        CVOIDPTR(inp), CVOIDPTR(ginp), CVOIDPTR(grid),
        grid.shape, ginp.shape,
        out.strides, gout.strides, inp.strides, ginp.strides, grid.strides)
}

/***********************************************************************
 *                         COUNT BACKWARD                              *
 ***********************************************************************/

void count_backward(
          DLTensor & gout_,
    const DLTensor & ginp_,
    const DLTensor & grid_,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          int        /* stream <unused> */
)
{
    ContiguousStrides _gout(gout_), _ginp(ginp_), _grid(grid_);
    DLTensor       & gout = _gout.t;
    const DLTensor & ginp = _ginp.t;
    const DLTensor & grid = _grid.t;

    const int      ndim   = static_cast<int>(grid.shape[grid.ndim - 1]);
    const int32_t  nbatch = grid.ndim - ndim - 1;
    const int64_t  n1     = grid.ndim;
    CHECK_NO_LANES  (gout)
    CHECK_SAME_DTYPE(gout, ginp)
    CHECK_SAME_DTYPE(gout, grid)
    CHECK_SAME(gout.ndim, grid.ndim, "gout and grid must have the same rank")
    CHECK_SAME(ginp.ndim, grid.ndim, "ginp and grid must have the same rank")
    if (nbatch < 0)
        throw std::invalid_argument("grid rank is too small for the coordinate dim");
    for (int32_t d = 0; d < gout.ndim; ++d)
        CHECK_SAME(gout.shape[d], grid.shape[d], "gout and grid must have the same shape")
    CHECK_SAME(ginp.shape[ginp.ndim-1], 1, "count gradient must have a single channel")
    CHECK_SAME_BATCH(ginp, grid, nbatch)

    const bool     use_32bits = CANUSE32BITS(gout) && CANUSE32BITS(ginp)
                             && CANUSE32BITS(grid);
    const auto     code = static_cast<DLDataTypeCode>(gout.dtype.code);
    const auto     bits = gout.dtype.bits;
    const spline_t spl  = static_cast<spline_t>(spline);
    const bound_t  bnd  = static_cast<bound_t >(bound);
    const bound::BoundVec   bvec(bnd);
    const spline::SplineVec svec(spl);
    const int      ex   = static_cast<int>(extrapolate);

    DISPATCH_PP(_count_backward,
        bvec, svec,
        static_cast<int64_t>(nbatch), n1, ex,
        VOIDPTR(gout), CVOIDPTR(ginp), CVOIDPTR(grid),
        grid.shape, ginp.shape,
        gout.strides, ginp.strides, grid.strides)
}

/***********************************************************************
 *                          GRAD BACKWARD                              *
 ***********************************************************************/

void grad_backward(
          DLTensor & out_,
          DLTensor & gout_,
    const DLTensor & inp_,
    const DLTensor & ginp_,
    const DLTensor & grid_,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          bool       abs,
          int        /* stream <unused> */
)
{
    ContiguousStrides _out(out_), _gout(gout_), _inp(inp_), _ginp(ginp_), _grid(grid_);
    DLTensor       & out  = _out.t;
    DLTensor       & gout = _gout.t;
    const DLTensor & inp  = _inp.t;
    const DLTensor & ginp = _ginp.t;
    const DLTensor & grid = _grid.t;

    const int      ndim   = static_cast<int>(grid.shape[grid.ndim - 1]);
    const int32_t  nbatch = grid.ndim - ndim - 1;
    const int64_t  n1     = grid.ndim;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, gout)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME_DTYPE(out, ginp)
    CHECK_SAME_DTYPE(out, grid)
    CHECK_SAME(out.ndim,  grid.ndim, "out and grid must have the same rank")
    CHECK_SAME(gout.ndim, grid.ndim, "gout and grid must have the same rank")
    CHECK_SAME(inp.ndim,  grid.ndim, "inp and grid must have the same rank")
    CHECK_SAME(ginp.ndim, grid.ndim + 1, "ginp must have an extra trailing axis")
    if (nbatch < 0)
        throw std::invalid_argument("grid rank is too small for the coordinate dim");
    for (int32_t d = 0; d < out.ndim; ++d)
        CHECK_SAME(out.shape[d], inp.shape[d], "out and inp must have the same shape")
    for (int32_t d = 0; d < gout.ndim; ++d)
        CHECK_SAME(gout.shape[d], grid.shape[d], "gout and grid must have the same shape")
    CHECK_SAME(ginp.shape[ginp.ndim-1], ndim, "ginp trailing axis must equal ndim")
    CHECK_SAME(ginp.shape[ginp.ndim-2], inp.shape[inp.ndim-1], "channel counts differ")
    CHECK_SAME_BATCH(inp,  grid, nbatch)
    CHECK_SAME_BATCH(ginp, grid, nbatch)

    const bool     use_32bits = CANUSE32BITS(out)  && CANUSE32BITS(gout)
                             && CANUSE32BITS(inp)  && CANUSE32BITS(ginp)
                             && CANUSE32BITS(grid);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const spline_t spl  = static_cast<spline_t>(spline);
    const bound_t  bnd  = static_cast<bound_t >(bound);
    const bound::BoundVec   bvec(bnd);
    const spline::SplineVec svec(spl);
    const int      ex   = static_cast<int>(extrapolate);

    DISPATCH_PP(_grad_backward,
        bvec, svec,
        static_cast<int64_t>(nbatch), n1, ex, abs,
        VOIDPTR(out), VOIDPTR(gout),
        CVOIDPTR(inp), CVOIDPTR(ginp), CVOIDPTR(grid),
        grid.shape, inp.shape,
        out.strides, gout.strides, inp.strides, ginp.strides, grid.strides)
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
