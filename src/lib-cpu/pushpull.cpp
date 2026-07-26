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
        (X.dtype.bits  != Y.dtype.bits) ||                             \
        (X.dtype.lanes != Y.dtype.lanes))                             \
        throw std::invalid_argument("Tensors do not have the same data type");

#define CHECK_SAME_BATCH(X, Y, D)                                       \
    for (int32_t d=0; d < D; ++d)                                       \
        if (X.shape[d] != Y.shape[d])                                   \
            throw std::invalid_argument("Tensors do not have the same batch shape");

// The teeny driver decodes out/grid (pull/grad) — or inp/grid (push) — over the
// SAME (*batch, *spatial) index, so their spatial extents must match (the old
// driver decoded everything over size_grid; the per-tensor decode relies on this
// equality instead). Checks dims [NB, NB+ND).
#define CHECK_SAME_SPATIAL(X, Y, NB, ND)                                \
    for (int32_t d=0; d < (ND); ++d)                                    \
        if (X.shape[(NB)+d] != Y.shape[(NB)+d])                         \
            throw std::invalid_argument("Tensors do not have the same spatial shape");

// as_anyrank(copy_meta) copies the shape/stride into an inline store capped at
// TNY_MAX_RANK; a larger rank would assert-abort (-O3, no -DNDEBUG). Reject it
// with a clear error instead. The Makefile bumps TNY_MAX_RANK to 64 (DLPack max).
#define CHECK_RANK_FITS(R)                                              \
    if ((R) > TNY_MAX_RANK)                                             \
        throw std::invalid_argument("tensor rank exceeds TNY_MAX_RANK");

/***********************************************************************
 *                          LEAF WRAPPERS                             *
 ***********************************************************************/
// Narrow each tensor's own shape/stride arrays to `offset_t` (32-bit when the
// spans fit), cast the data pointers, and call the teeny driver. Order O and
// boundary B are compile-time; `bnd` (the runtime bound) is forwarded as the
// driver's `rt` arg and consumed only when B == bound_t::Dynamic (the routed
// bounds). `n1 == nbatch + ndim + 1`; grad's output carries one extra axis.

namespace {

template <int D, int O, bound_t B, typename scalar_t, typename offset_t>
inline void _pull(
          int64_t nbatch, int64_t n1, int extrapolate, bound_t bnd,
          void * out, const void * inp, const void * grid,
    const int64_t * size_out, const int64_t * size_inp, const int64_t * size_grid,
    const int64_t * stride_out, const int64_t * stride_inp, const int64_t * stride_grid)
{
    const offset_t * _zo = copy_if_needed<offset_t *>(size_out,   n1);
    const offset_t * _zi = copy_if_needed<offset_t *>(size_inp,   n1);
    const offset_t * _zg = copy_if_needed<offset_t *>(size_grid,  n1);
    const offset_t * _to = copy_if_needed<offset_t *>(stride_out, n1);
    const offset_t * _ti = copy_if_needed<offset_t *>(stride_inp, n1);
    const offset_t * _tg = copy_if_needed<offset_t *>(stride_grid,n1);
    pushpull::pull<D, O, B, reduce_t, scalar_t, offset_t>(
        static_cast<offset_t>(nbatch), extrapolate, bnd,
        static_cast<scalar_t *>(out), static_cast<const scalar_t *>(inp),
        static_cast<const scalar_t *>(grid), _zo, _zi, _zg, _to, _ti, _tg);
    free_if_needed<int64_t *>(_zo); free_if_needed<int64_t *>(_zi); free_if_needed<int64_t *>(_zg);
    free_if_needed<int64_t *>(_to); free_if_needed<int64_t *>(_ti); free_if_needed<int64_t *>(_tg);
}

template <int D, int O, bound_t B, typename scalar_t, typename offset_t>
inline void _push(
          int64_t nbatch, int64_t n1, int extrapolate, bound_t bnd,
          void * out, const void * inp, const void * grid,
    const int64_t * size_out, const int64_t * size_inp, const int64_t * size_grid,
    const int64_t * stride_out, const int64_t * stride_inp, const int64_t * stride_grid)
{
    const offset_t * _zo = copy_if_needed<offset_t *>(size_out,   n1);
    const offset_t * _zi = copy_if_needed<offset_t *>(size_inp,   n1);
    const offset_t * _zg = copy_if_needed<offset_t *>(size_grid,  n1);
    const offset_t * _to = copy_if_needed<offset_t *>(stride_out, n1);
    const offset_t * _ti = copy_if_needed<offset_t *>(stride_inp, n1);
    const offset_t * _tg = copy_if_needed<offset_t *>(stride_grid,n1);
    pushpull::push<D, O, B, reduce_t, scalar_t, offset_t>(
        static_cast<offset_t>(nbatch), extrapolate, bnd,
        static_cast<scalar_t *>(out), static_cast<const scalar_t *>(inp),
        static_cast<const scalar_t *>(grid), _zo, _zi, _zg, _to, _ti, _tg);
    free_if_needed<int64_t *>(_zo); free_if_needed<int64_t *>(_zi); free_if_needed<int64_t *>(_zg);
    free_if_needed<int64_t *>(_to); free_if_needed<int64_t *>(_ti); free_if_needed<int64_t *>(_tg);
}

template <int D, int O, bound_t B, typename scalar_t, typename offset_t>
inline void _count(
          int64_t nbatch, int64_t n1, int extrapolate, bound_t bnd,
          void * out, const void * grid,
    const int64_t * size_out, const int64_t * size_grid,
    const int64_t * stride_out, const int64_t * stride_grid)
{
    const offset_t * _zo = copy_if_needed<offset_t *>(size_out,   n1);
    const offset_t * _zg = copy_if_needed<offset_t *>(size_grid,  n1);
    const offset_t * _to = copy_if_needed<offset_t *>(stride_out, n1);
    const offset_t * _tg = copy_if_needed<offset_t *>(stride_grid,n1);
    pushpull::count<D, O, B, reduce_t, scalar_t, offset_t>(
        static_cast<offset_t>(nbatch), extrapolate, bnd,
        static_cast<scalar_t *>(out), static_cast<const scalar_t *>(grid),
        _zo, _zg, _to, _tg);
    free_if_needed<int64_t *>(_zo); free_if_needed<int64_t *>(_zg);
    free_if_needed<int64_t *>(_to); free_if_needed<int64_t *>(_tg);
}

template <int D, int O, bound_t B, bool ABS, typename scalar_t, typename offset_t>
inline void _grad(
          int64_t nbatch, int64_t n1, int extrapolate, bound_t bnd,
          void * out, const void * inp, const void * grid,
    const int64_t * size_out, const int64_t * size_inp, const int64_t * size_grid,
    const int64_t * stride_out, const int64_t * stride_inp, const int64_t * stride_grid)
{
    const offset_t * _zo = copy_if_needed<offset_t *>(size_out,   n1 + 1);   // extra D axis
    const offset_t * _zi = copy_if_needed<offset_t *>(size_inp,   n1);
    const offset_t * _zg = copy_if_needed<offset_t *>(size_grid,  n1);
    const offset_t * _to = copy_if_needed<offset_t *>(stride_out, n1 + 1);
    const offset_t * _ti = copy_if_needed<offset_t *>(stride_inp, n1);
    const offset_t * _tg = copy_if_needed<offset_t *>(stride_grid,n1);
    pushpull::grad<D, O, B, ABS, reduce_t, scalar_t, offset_t>(
        static_cast<offset_t>(nbatch), extrapolate, bnd,
        static_cast<scalar_t *>(out), static_cast<const scalar_t *>(inp),
        static_cast<const scalar_t *>(grid), _zo, _zi, _zg, _to, _ti, _tg);
    free_if_needed<int64_t *>(_zo); free_if_needed<int64_t *>(_zi); free_if_needed<int64_t *>(_zg);
    free_if_needed<int64_t *>(_to); free_if_needed<int64_t *>(_ti); free_if_needed<int64_t *>(_tg);
}

}  // namespace

/***********************************************************************
 *                             DISPATCH                                *
 ***********************************************************************/
// Full library matrix: ndim(1/2/3) x order(0-7) x bound x dtype(f32/f64) x
// offset(i32/i64). The bound split (cpu-lib#22): DFT/DCT2/DST2/Zero/NoCheck are
// compiled statically; DCT1/DST1/Replicate route to the single Dynamic
// instantiation (the runtime `bnd` selects inside the kernel). Test builds trim
// order x bound with -DFF_TEST_SPARSE (DCT2 only outside Linear/Cubic), exactly
// like the pre-teeny dispatch.

// dtype x offset, given static D, O, and compile-time bound B.
#define PP_DTYPE(D, O, B, FN, args...)                                      \
    switch (bits) {                                                         \
        case 32: return (use_32bits ? FN<D,O,B,float, int32_t>(args)        \
                                    : FN<D,O,B,float, int64_t>(args));      \
        case 64: return (use_32bits ? FN<D,O,B,double,int32_t>(args)        \
                                    : FN<D,O,B,double,int64_t>(args));      \
        default: break;                                                     \
    }                                                                       \
    throw std::invalid_argument("only float32 / float64 are supported");

// bound -> compile-time B, applying the static/Dynamic split.
#define PP_BOUND(D, O, FN, args...)                                         \
    switch (bnd) {                                                          \
        case bound_t::DFT:       PP_DTYPE(D,O,bound_t::DFT,     FN,args); break; \
        case bound_t::DCT2:      PP_DTYPE(D,O,bound_t::DCT2,    FN,args); break; \
        case bound_t::DST2:      PP_DTYPE(D,O,bound_t::DST2,    FN,args); break; \
        case bound_t::Zero:      PP_DTYPE(D,O,bound_t::Zero,    FN,args); break; \
        case bound_t::NoCheck:   PP_DTYPE(D,O,bound_t::NoCheck, FN,args); break; \
        case bound_t::DCT1:                                                 \
        case bound_t::DST1:                                                 \
        case bound_t::Replicate: PP_DTYPE(D,O,bound_t::Dynamic, FN,args); break; \
        default: throw std::invalid_argument("unsupported boundary condition"); \
    }

#ifdef FF_TEST_SPARSE
// Covering subset: full bound sweep only at Linear + Cubic; DCT2-only elsewhere.
#define PP_BOUND_SPARSE(D, O, FN, args...)                                  \
    switch (bnd) {                                                          \
        case bound_t::DCT2: PP_DTYPE(D,O,bound_t::DCT2, FN,args); break;    \
        default: throw std::invalid_argument("bound not instantiated in FF_TEST_SPARSE build"); \
    }
#define PP_ORDER(D, FN, args...)                                            \
    switch (spl) {                                                          \
        case spline_t::Nearest:      PP_BOUND_SPARSE(D,0,FN,args); break;   \
        case spline_t::Linear:       PP_BOUND       (D,1,FN,args); break;   \
        case spline_t::Quadratic:    PP_BOUND_SPARSE(D,2,FN,args); break;   \
        case spline_t::Cubic:        PP_BOUND       (D,3,FN,args); break;   \
        case spline_t::FourthOrder:  PP_BOUND_SPARSE(D,4,FN,args); break;   \
        case spline_t::FifthOrder:   PP_BOUND_SPARSE(D,5,FN,args); break;   \
        case spline_t::SixthOrder:   PP_BOUND_SPARSE(D,6,FN,args); break;   \
        case spline_t::SeventhOrder: PP_BOUND_SPARSE(D,7,FN,args); break;   \
        default: throw std::invalid_argument("Unsupported spline order");   \
    }
#else
#define PP_ORDER(D, FN, args...)                                            \
    switch (spl) {                                                          \
        case spline_t::Nearest:      PP_BOUND(D,0,FN,args); break;          \
        case spline_t::Linear:       PP_BOUND(D,1,FN,args); break;          \
        case spline_t::Quadratic:    PP_BOUND(D,2,FN,args); break;          \
        case spline_t::Cubic:        PP_BOUND(D,3,FN,args); break;          \
        case spline_t::FourthOrder:  PP_BOUND(D,4,FN,args); break;          \
        case spline_t::FifthOrder:   PP_BOUND(D,5,FN,args); break;          \
        case spline_t::SixthOrder:   PP_BOUND(D,6,FN,args); break;          \
        case spline_t::SeventhOrder: PP_BOUND(D,7,FN,args); break;          \
        default: throw std::invalid_argument("Unsupported spline order");   \
    }
#endif

#define DISPATCH_PP(FN, args...)                                            \
{                                                                          \
    if (code != kDLFloat)                                                  \
        throw std::invalid_argument("only floating point data types are supported"); \
    switch (ndim) {                                                        \
        case 1: PP_ORDER(1, FN, args); break;                             \
        case 2: PP_ORDER(2, FN, args); break;                             \
        case 3: PP_ORDER(3, FN, args); break;                             \
        default: throw std::invalid_argument("Only 1D, 2D and 3D are supported"); \
    }                                                                      \
}

// ---- grad adds the runtime `abs` -> compile-time ABS axis -----------------
#define PPG_DTYPE(D, O, B, ABS, args...)                                    \
    switch (bits) {                                                         \
        case 32: return (use_32bits ? _grad<D,O,B,ABS,float, int32_t>(args) \
                                    : _grad<D,O,B,ABS,float, int64_t>(args));\
        case 64: return (use_32bits ? _grad<D,O,B,ABS,double,int32_t>(args) \
                                    : _grad<D,O,B,ABS,double,int64_t>(args));\
        default: break;                                                     \
    }                                                                       \
    throw std::invalid_argument("only float32 / float64 are supported");

#define PPG_ABS(D, O, B, args...)                                           \
    if (abs) { PPG_DTYPE(D,O,B,true,  args) } else { PPG_DTYPE(D,O,B,false, args) }

#define PPG_BOUND(D, O, args...)                                            \
    switch (bnd) {                                                          \
        case bound_t::DFT:       PPG_ABS(D,O,bound_t::DFT,     args); break; \
        case bound_t::DCT2:      PPG_ABS(D,O,bound_t::DCT2,    args); break; \
        case bound_t::DST2:      PPG_ABS(D,O,bound_t::DST2,    args); break; \
        case bound_t::Zero:      PPG_ABS(D,O,bound_t::Zero,    args); break; \
        case bound_t::NoCheck:   PPG_ABS(D,O,bound_t::NoCheck, args); break; \
        case bound_t::DCT1:                                                 \
        case bound_t::DST1:                                                 \
        case bound_t::Replicate: PPG_ABS(D,O,bound_t::Dynamic, args); break; \
        default: throw std::invalid_argument("unsupported boundary condition"); \
    }

#ifdef FF_TEST_SPARSE
#define PPG_BOUND_SPARSE(D, O, args...)                                     \
    switch (bnd) {                                                          \
        case bound_t::DCT2: PPG_ABS(D,O,bound_t::DCT2, args); break;        \
        default: throw std::invalid_argument("bound not instantiated in FF_TEST_SPARSE build"); \
    }
#define PPG_ORDER(D, args...)                                               \
    switch (spl) {                                                          \
        case spline_t::Nearest:      PPG_BOUND_SPARSE(D,0,args); break;     \
        case spline_t::Linear:       PPG_BOUND       (D,1,args); break;     \
        case spline_t::Quadratic:    PPG_BOUND_SPARSE(D,2,args); break;     \
        case spline_t::Cubic:        PPG_BOUND       (D,3,args); break;     \
        case spline_t::FourthOrder:  PPG_BOUND_SPARSE(D,4,args); break;     \
        case spline_t::FifthOrder:   PPG_BOUND_SPARSE(D,5,args); break;     \
        case spline_t::SixthOrder:   PPG_BOUND_SPARSE(D,6,args); break;     \
        case spline_t::SeventhOrder: PPG_BOUND_SPARSE(D,7,args); break;     \
        default: throw std::invalid_argument("Unsupported spline order");   \
    }
#else
#define PPG_ORDER(D, args...)                                               \
    switch (spl) {                                                          \
        case spline_t::Nearest:      PPG_BOUND(D,0,args); break;            \
        case spline_t::Linear:       PPG_BOUND(D,1,args); break;            \
        case spline_t::Quadratic:    PPG_BOUND(D,2,args); break;            \
        case spline_t::Cubic:        PPG_BOUND(D,3,args); break;            \
        case spline_t::FourthOrder:  PPG_BOUND(D,4,args); break;            \
        case spline_t::FifthOrder:   PPG_BOUND(D,5,args); break;            \
        case spline_t::SixthOrder:   PPG_BOUND(D,6,args); break;            \
        case spline_t::SeventhOrder: PPG_BOUND(D,7,args); break;            \
        default: throw std::invalid_argument("Unsupported spline order");   \
    }
#endif

#define DISPATCH_PP_GRAD(args...)                                           \
{                                                                          \
    if (code != kDLFloat)                                                  \
        throw std::invalid_argument("only floating point data types are supported"); \
    switch (ndim) {                                                        \
        case 1: PPG_ORDER(1, args); break;                                \
        case 2: PPG_ORDER(2, args); break;                                \
        case 3: PPG_ORDER(3, args); break;                                \
        default: throw std::invalid_argument("Only 1D, 2D and 3D are supported"); \
    }                                                                      \
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
    CHECK_SAME_SPATIAL(out, grid, nbatch, ndim)   // out is grid-shaped; inp is the sampled volume
    CHECK_RANK_FITS(n1)

    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp) && CANUSE32BITS(grid);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const spline_t spl  = static_cast<spline_t>(spline);
    const bound_t  bnd  = static_cast<bound_t >(bound);
    const int      ex   = static_cast<int>(extrapolate);

    DISPATCH_PP(_pull,
        static_cast<int64_t>(nbatch), n1, ex, bnd,
        VOIDPTR(out), CVOIDPTR(inp), CVOIDPTR(grid),
        out.shape, inp.shape, grid.shape,
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
    CHECK_SAME_SPATIAL(inp, grid, nbatch, ndim)   // inp is grid-shaped; out is the splatted volume
    CHECK_RANK_FITS(n1)

    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp) && CANUSE32BITS(grid);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const spline_t spl  = static_cast<spline_t>(spline);
    const bound_t  bnd  = static_cast<bound_t >(bound);
    const int      ex   = static_cast<int>(extrapolate);

    DISPATCH_PP(_push,
        static_cast<int64_t>(nbatch), n1, ex, bnd,
        VOIDPTR(out), CVOIDPTR(inp), CVOIDPTR(grid),
        out.shape, inp.shape, grid.shape,
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
    CHECK_RANK_FITS(n1)   // out is the splatted volume (its own spatial); no shared-spatial decode

    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(grid);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const spline_t spl  = static_cast<spline_t>(spline);
    const bound_t  bnd  = static_cast<bound_t >(bound);
    const int      ex   = static_cast<int>(extrapolate);

    DISPATCH_PP(_count,
        static_cast<int64_t>(nbatch), n1, ex, bnd,
        VOIDPTR(out), CVOIDPTR(grid),
        out.shape, grid.shape,
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
    CHECK_SAME_SPATIAL(out, grid, nbatch, ndim)   // out is grid-shaped (+C,D); inp is the sampled volume
    CHECK_RANK_FITS(n1 + 1)                        // grad output carries the extra D axis

    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp) && CANUSE32BITS(grid);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const spline_t spl  = static_cast<spline_t>(spline);
    const bound_t  bnd  = static_cast<bound_t >(bound);
    const int      ex   = static_cast<int>(extrapolate);

    DISPATCH_PP_GRAD(
        static_cast<int64_t>(nbatch), n1, ex, bnd,
        VOIDPTR(out), CVOIDPTR(inp), CVOIDPTR(grid),
        out.shape, inp.shape, grid.shape,
        out.strides, inp.strides, grid.strides)
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
