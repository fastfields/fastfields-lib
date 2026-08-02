#ifndef FF_LIB_SPLINC
#define FF_LIB_SPLINC
#include "dlpack.h"
#include <cstdint>
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
 * @brief In-place spline coefficient prefilter along the last dimension.
 *
 * Filters the last axis (all leading axes are batch); prefiltering makes
 * spline interpolation reproduce the input samples. To filter several axes,
 * permute the tensor and call repeatedly.
 *
 * @param inp_out  Input/Output tensor in DLTensor format (float32/float64)
 * @param spline   Spline order (orders 0/1 are no-ops)
 * @param bound    Boundary condition
 * @param stream   Cuda stream on which to operate
 */
void spline_coeff(
          DLTensor & inp_out ,
          int8_t     spline   = spline_t::Cubic,
          int8_t     bound    = bound_t::DCT2,
          intptr_t   stream   = 0
);

FF_NAMESPACE_END(FF)

#endif // FF_LIB_SPLINC
