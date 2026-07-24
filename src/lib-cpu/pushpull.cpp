#include <stdexcept>
#include "pushpull.h"
#include "autocast.h"
#include "dlpack.h"
#include "impl/kernels/cuda_switch.h"
#include "impl/kernels/utils.h"
#include "impl/pushpull.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

#define VOIDPTR(x)      (static_cast<void*>(static_cast<char*>(x.data) + x.byte_offset))
#define CVOIDPTR(x)     (static_cast<const void*>(static_cast<const char*>(x.data) + x.byte_offset))
#define CANUSE32BITS(x) (canUse32BitIndexMath(x.ndim, x.shape, x.strides))

// reduce/accumulation type used by the sampling kernels. Match jitfields
// (float64) for CPU accuracy.
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
        (X.dtype.bits  != Y.dtype.bits) ||                              \
        (X.dtype.lanes != Y.dtype.lanes))                              \
        throw std::invalid_argument("Tensors do not have the same data type");

#define CHECK_SAME_BATCH(X, Y, D)                                       \
    for (int32_t d=0; d < D; ++d)                                       \
        if (X.shape[d] != Y.shape[d])                                   \
            throw std::invalid_argument("Tensors do not have the same batch shape");

/***********************************************************************
 *                          LEAF WRAPPERS                             *
 ***********************************************************************/

namespace {

// Narrow the [n1]-length shape/stride arrays to `offset_t`, cast the data
// pointers and call the templated impl. `n1 == nbatch + ndim + 1`.
template <int ndim, spline::type I, bound::type B,
          typename scalar_t, typename offset_t>
inline void _pull(
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
        _sg, _ss, _so, _si, _sgr);

    free_if_needed<int64_t *>(_sg);  free_if_needed<int64_t *>(_ss);
    free_if_needed<int64_t *>(_so);  free_if_needed<int64_t *>(_si);
    free_if_needed<int64_t *>(_sgr);
}

template <int ndim, spline::type I, bound::type B,
          typename scalar_t, typename offset_t>
inline void _push(
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
        _sg, _ss, _so, _si, _sgr);

    free_if_needed<int64_t *>(_sg);  free_if_needed<int64_t *>(_ss);
    free_if_needed<int64_t *>(_so);  free_if_needed<int64_t *>(_si);
    free_if_needed<int64_t *>(_sgr);
}

template <int ndim, spline::type I, bound::type B,
          typename scalar_t, typename offset_t>
inline void _count(
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
        _sg, _ss, _so, _sgr);

    free_if_needed<int64_t *>(_sg);  free_if_needed<int64_t *>(_ss);
    free_if_needed<int64_t *>(_so);  free_if_needed<int64_t *>(_sgr);
}

// grad: out has an extra trailing (D) axis, so stride_out has length n2 = n1+1.
template <int ndim, spline::type I, bound::type B,
          typename scalar_t, typename offset_t>
inline void _grad(
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
            _sg, _ss, _so, _si, _sgr);
    else
        pushpull::grad<ndim, false, reduce_t, scalar_t, offset_t, I, B>(
            static_cast<offset_t>(nbatch), extrapolate, _out, _inp, _grid,
            _sg, _ss, _so, _si, _sgr);

    free_if_needed<int64_t *>(_sg);  free_if_needed<int64_t *>(_ss);
    free_if_needed<int64_t *>(_so);  free_if_needed<int64_t *>(_si);
    free_if_needed<int64_t *>(_sgr);
}

} // anonymous namespace

/***********************************************************************
 *                            DISPATCH                                *
 ***********************************************************************/

#define PP_DTYPE(D, I, B, FN, args...)                                         \
    switch (code) {                                                            \
        case kDLFloat: switch (bits) {                                         \
            case 32: return (use_32bits ? FN<D,I,B,float, int32_t>(args)       \
                                        : FN<D,I,B,float, int64_t>(args));      \
            case 64: return (use_32bits ? FN<D,I,B,double,int32_t>(args)       \
                                        : FN<D,I,B,double,int64_t>(args));      \
            default: break;                                                    \
        }; default: break;                                                     \
    }

#define PP_BOUND(D, I, FN, args...)                                            \
    switch (bnd) {                                                             \
        case bound_t::Zero:      PP_DTYPE(D,I,bound_t::Zero,     FN,args); break; \
        case bound_t::Replicate: PP_DTYPE(D,I,bound_t::Replicate,FN,args); break; \
        case bound_t::DCT1:      PP_DTYPE(D,I,bound_t::DCT1,     FN,args); break; \
        case bound_t::DCT2:      PP_DTYPE(D,I,bound_t::DCT2,     FN,args); break; \
        case bound_t::DST1:      PP_DTYPE(D,I,bound_t::DST1,     FN,args); break; \
        case bound_t::DST2:      PP_DTYPE(D,I,bound_t::DST2,     FN,args); break; \
        case bound_t::DFT:       PP_DTYPE(D,I,bound_t::DFT,      FN,args); break; \
        case bound_t::NoCheck:   PP_DTYPE(D,I,bound_t::NoCheck,  FN,args); break; \
        default: throw std::invalid_argument("Unsupported boundary condition");\
    }

// Fast-test builds (`-DFF_TEST_SPARSE`) instantiate only a *covering* subset of
// the order x bound matrix instead of the full 8x8: all bounds for the two
// reference orders -- Linear (the specialised L/L path) and Cubic (the generic
// "Any" path) -- and only DCT2 for the remaining orders. The covering
// test_pushpull exercises exactly this set (each order once at DCT2; each bound
// once at Linear and at Cubic; Linear/Cubic x every ndim). It cuts the pushpull
// compile ~3x while still hitting every single-factor value and the order x
// ndim interaction class. The default (library / CI) build keeps the full
// matrix, so nothing ships uncompiled.
#ifdef FF_TEST_SPARSE
#define PP_BOUND_SPARSE(D, I, FN, args...)                                     \
    switch (bnd) {                                                             \
        case bound_t::DCT2: PP_DTYPE(D,I,bound_t::DCT2, FN,args); break;       \
        default: throw std::invalid_argument(                                 \
            "bound not instantiated in FF_TEST_SPARSE build");                \
    }
#define PP_ORDER(D, FN, args...)                                               \
    switch (spl) {                                                             \
        case spline_t::Nearest:      PP_BOUND_SPARSE(D,spline_t::Nearest,     FN,args); break; \
        case spline_t::Linear:       PP_BOUND(D,spline_t::Linear,      FN,args); break; \
        case spline_t::Quadratic:    PP_BOUND_SPARSE(D,spline_t::Quadratic,   FN,args); break; \
        case spline_t::Cubic:        PP_BOUND(D,spline_t::Cubic,       FN,args); break; \
        case spline_t::FourthOrder:  PP_BOUND_SPARSE(D,spline_t::FourthOrder, FN,args); break; \
        case spline_t::FifthOrder:   PP_BOUND_SPARSE(D,spline_t::FifthOrder,  FN,args); break; \
        case spline_t::SixthOrder:   PP_BOUND_SPARSE(D,spline_t::SixthOrder,  FN,args); break; \
        case spline_t::SeventhOrder: PP_BOUND_SPARSE(D,spline_t::SeventhOrder,FN,args); break; \
        default: throw std::invalid_argument("Unsupported spline order");      \
    }
#else
#define PP_ORDER(D, FN, args...)                                               \
    switch (spl) {                                                             \
        case spline_t::Nearest:      PP_BOUND(D,spline_t::Nearest,     FN,args); break; \
        case spline_t::Linear:       PP_BOUND(D,spline_t::Linear,      FN,args); break; \
        case spline_t::Quadratic:    PP_BOUND(D,spline_t::Quadratic,   FN,args); break; \
        case spline_t::Cubic:        PP_BOUND(D,spline_t::Cubic,       FN,args); break; \
        case spline_t::FourthOrder:  PP_BOUND(D,spline_t::FourthOrder, FN,args); break; \
        case spline_t::FifthOrder:   PP_BOUND(D,spline_t::FifthOrder,  FN,args); break; \
        case spline_t::SixthOrder:   PP_BOUND(D,spline_t::SixthOrder,  FN,args); break; \
        case spline_t::SeventhOrder: PP_BOUND(D,spline_t::SeventhOrder,FN,args); break; \
        default: throw std::invalid_argument("Unsupported spline order");      \
    }
#endif

#define DISPATCH_PP(FN, args...)                                               \
{                                                                              \
    switch (ndim) {                                                           \
        case 1: PP_ORDER(1, FN, args); break;                                 \
        case 2: PP_ORDER(2, FN, args); break;                                 \
        case 3: PP_ORDER(3, FN, args); break;                                 \
        default: throw std::invalid_argument("Only 1D, 2D and 3D are supported"); \
    };                                                                        \
    throw std::invalid_argument("Unsupported data type");                     \
}

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
    const int      ex   = static_cast<int>(extrapolate);

    DISPATCH_PP(_pull,
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
    const int      ex   = static_cast<int>(extrapolate);

    // size_splinc = out (the splatted volume); size_grid = grid.
    DISPATCH_PP(_push,
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
    const int      ex   = static_cast<int>(extrapolate);

    DISPATCH_PP(_count,
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
    const int      ex   = static_cast<int>(extrapolate);

    DISPATCH_PP(_grad,
        static_cast<int64_t>(nbatch), n1, ex, abs,
        VOIDPTR(out), CVOIDPTR(inp), CVOIDPTR(grid),
        grid.shape, inp.shape,
        out.strides, inp.strides, grid.strides)
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
