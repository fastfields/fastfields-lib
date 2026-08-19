#ifndef FF_LIB_PUSHPULL
#define FF_LIB_PUSHPULL
#include "fastfields/core/dlpack.h"
#include "fastfields/core/defines.h"

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
 * @brief Sample ("pull") a spline-encoded volume at arbitrary coordinates.
 *
 * Channel-last, x-first coordinate convention:
 *   inp  : (*batch, *inshape,  C)
 *   grid : (*batch, *outshape, D)   with D == the spatial rank (1, 2 or 3)
 *   out  : (*batch, *outshape, C)
 *
 * @param out          Output tensor (pulled samples)
 * @param inp          Input volume (spline coefficients)
 * @param grid         Sampling coordinates (in voxels, x-first)
 * @param spline       Spline order applied to every spatial dim
 * @param bound        Boundary condition applied to every spatial dim
 * @param extrapolate  1: always; 0: not past voxel centres; -1: not past edges
 * @param stream       Cuda stream on which to operate
 */
void pull(
          DLTensor & out,
    const DLTensor & inp,
    const DLTensor & grid,
          int8_t     spline      = spline_t::Quadratic,
          int8_t     bound       = bound_t::DCT2,
          int8_t     extrapolate = 1,
          int        stream      = 0
);

/**
 * @brief Splat ("push") values into a volume; numerical adjoint of `pull`.
 *        `out` must be pre-zeroed by the caller (values are accumulated).
 *
 *   inp  : (*batch, *outshape, C)
 *   grid : (*batch, *outshape, D)
 *   out  : (*batch, *inshape,  C)
 */
void push(
          DLTensor & out,
    const DLTensor & inp,
    const DLTensor & grid,
          int8_t     spline      = spline_t::Quadratic,
          int8_t     bound       = bound_t::DCT2,
          int8_t     extrapolate = 1,
          int        stream      = 0
);

/**
 * @brief Splat ones (== push of an all-ones input). `out` (*batch,*inshape,1)
 *        must be pre-zeroed.
 */
void count(
          DLTensor & out,
    const DLTensor & grid,
          int8_t     spline      = spline_t::Quadratic,
          int8_t     bound       = bound_t::DCT2,
          int8_t     extrapolate = 1,
          int        stream      = 0
);

/**
 * @brief Sample the spatial gradients of a spline-encoded volume.
 *
 *   inp  : (*batch, *inshape,  C)
 *   grid : (*batch, *outshape, D)
 *   out  : (*batch, *outshape, C, D)
 */
void grad(
          DLTensor & out,
    const DLTensor & inp,
    const DLTensor & grid,
          int8_t     spline      = spline_t::Quadratic,
          int8_t     bound       = bound_t::DCT2,
          int8_t     extrapolate = 1,
          bool       abs         = false,
          int        stream      = 0
);

FF_NAMESPACE_END(FF)

#endif // FF_LIB_PUSHPULL
