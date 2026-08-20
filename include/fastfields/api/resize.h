#pragma once
#include <fastfields/core/dlpack.h>
#include <cstdint>
#include <fastfields/core/defines.h>

#ifndef FF_LIB_BOUND_SPLINE_T
#define FF_LIB_BOUND_SPLINE_T
FF_NAMESPACE_BEGIN(FF_NS)

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

FF_NAMESPACE_END(FF_NS)
#endif // FF_LIB_BOUND_SPLINE_T

FF_NAMESPACE_BEGIN(FF_NS)

/**
 * @brief Resample (prolongation) a tensor to a new shape using spline
 *        interpolation. Prefilter the input with `spline_coeff` first for a
 *        proper interpolating resize.
 *
 * The `ndim` trailing dimensions are spatial; leading dimensions are batch.
 * For each output voxel `loc`, input is sampled at `x = scale*(loc+shift)-shift`.
 *
 * @param out     Output tensor (*batch, *outshape)
 * @param inp     Input tensor  (*batch, *inshape)
 * @param spline  Spline order applied to every spatial dim
 * @param bound   Boundary condition applied to every spatial dim
 * @param shift   Anchor shift (0 centres, 0.5 edges)
 * @param scale   [ndim] per-dim scaling (input-index per output-index)
 * @param ndim    Number of spatial dimensions (1, 2 or 3)
 * @param stream  Cuda stream on which to operate
 */
void resample(
          DLTensor & out    ,
    const DLTensor & inp    ,
          int8_t     spline = spline_t::Quadratic,
          int8_t     bound  = bound_t::DCT2,
          double     shift  = 0.0,
    const double   * scale  = nullptr,
          int        ndim   = 1,
          intptr_t   stream = 0
);

FF_NAMESPACE_END(FF_NS)
