#ifndef FF_LIB_RESTRICT
#define FF_LIB_RESTRICT
#include "dlpack.h"
#include "defines.h"

#ifndef FF_LIB_BOUND_SPLINE_T
#define FF_LIB_BOUND_SPLINE_T
FF_NAMESPACE_BEGIN(FF)

FF_NAMESPACE_BEGIN(bound_t)
using T = int8_t;
static constexpr T Dynamic   = -1;
static constexpr T Zero      =  0;
static constexpr T Replicate =  1;
static constexpr T DCT1      =  2;
static constexpr T DCT2      =  3;
static constexpr T DST1      =  4;
static constexpr T DST2      =  5;
static constexpr T DFT       =  6;
static constexpr T NoCheck   =  7;
FF_NAMESPACE_END(bound_t)

FF_NAMESPACE_BEGIN(spline_t)
using T = int8_t;
static constexpr T Dynamic       = -1;
static constexpr T Nearest       =  0;
static constexpr T Linear        =  1;
static constexpr T Quadratic     =  2;
static constexpr T Cubic         =  3;
static constexpr T FourthOrder   =  4;
static constexpr T FifthOrder    =  5;
static constexpr T SixthOrder    =  6;
static constexpr T SeventhOrder  =  7;
FF_NAMESPACE_END(spline_t)

FF_NAMESPACE_END(FF)
#endif // FF_LIB_BOUND_SPLINE_T

FF_NAMESPACE_BEGIN(FF)

/**
 * @brief Restriction: the adjoint (transpose) of the resize prolongation.
 *
 * Maps a fine tensor `inp` to a coarse tensor `out`; `out` is ACCUMULATED into
 * and must be zeroed by the caller. Use the reciprocal scale of the matching
 * resize (resize: coarse/fine; restriction: fine/coarse) with the same shift.
 *
 * @param out     Output (coarse) tensor (*batch, *outshape), pre-zeroed
 * @param inp     Input (fine) tensor    (*batch, *inshape)
 * @param spline  Spline order applied to every spatial dim
 * @param bound   Boundary condition applied to every spatial dim
 * @param shift   Anchor shift (0 centres, 0.5 edges)
 * @param scale   [ndim] per-dim scaling (input-index per output-index)
 * @param ndim    Number of spatial dimensions (1, 2 or 3)
 * @param stream  Cuda stream on which to operate
 */
void restriction(
          DLTensor & out    ,
    const DLTensor & inp    ,
          int8_t     spline = spline_t::Quadratic,
          int8_t     bound  = bound_t::DCT2,
          double     shift  = 0.0,
    const double   * scale  = nullptr,
          int        ndim   = 1,
          int        stream = 0
);

FF_NAMESPACE_END(FF)

#endif // FF_LIB_RESTRICT
