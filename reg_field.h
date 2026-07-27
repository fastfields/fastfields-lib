#ifndef FF_LIB_REG_FIELD
#define FF_LIB_REG_FIELD
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
 * @brief Apply a spatial regulariser operator to a multi-channel field.
 *
 * Layout is (*batch, *spatial, C) with `C` channels in the last axis. The
 * penalties are per-channel weight vectors of length `C`; the highest-order
 * non-null penalty selects the finite-difference stencil. With `voxel_size == 1`
 * and only `absolute`, the result is a per-channel scaling
 * `out[...,c] = absolute[c] * inp[...,c]`; with only `membrane`, it is
 * `membrane[c]` times the discrete negative Laplacian of channel `c`.
 *
 * @param out         Output tensor (*batch, *spatial, C)
 * @param inp         Input  tensor (*batch, *spatial, C)
 * @param voxel_size  [ndim] spatial voxel size (nullptr -> all ones)
 * @param absolute    [C] absolute (L2) penalty weights (nullptr -> zeros)
 * @param membrane    [C] membrane penalty weights (nullptr -> disabled)
 * @param bending     [C] bending penalty weights (nullptr -> disabled)
 * @param bound       Boundary condition applied to every spatial dim
 * @param ndim        Number of spatial dimensions (1, 2 or 3)
 * @param stream      Cuda stream on which to operate
 */
void field_matvec(
          DLTensor & out       ,
    const DLTensor & inp       ,
    const double   * voxel_size = nullptr,
    const double   * absolute  = nullptr,
    const double   * membrane  = nullptr,
    const double   * bending   = nullptr,
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief Diagonal (preconditioner) of the regulariser operator, same
 *        conventions as `field_matvec`. Writes into `out` (*batch, *spatial, C).
 */
void field_diag(
          DLTensor & out       ,
    const double   * voxel_size = nullptr,
    const double   * absolute  = nullptr,
    const double   * membrane  = nullptr,
    const double   * bending   = nullptr,
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief Materialise the per-channel Toeplitz convolution kernel (stencil) of
 *        the field regulariser (same penalties/conventions as `field_matvec`).
 *
 * Writes the small per-channel stencil that, convolved with a field, reproduces
 * `field_matvec`. The output is a vector of per-channel kernels, shape
 * `(*batch, *spatial, C)` (the field regulariser never couples channels). The
 * spatial extent must be at least the stencil width (1 absolute / 3 membrane /
 * 5 bending) and is centred.
 */
void field_kernel(
          DLTensor & out       ,
    const double   * voxel_size = nullptr,
    const double   * absolute  = nullptr,
    const double   * membrane  = nullptr,
    const double   * bending   = nullptr,
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief In-place relaxation (Gauss-Seidel) sweeps solving `(H + L) x = g`.
 *
 * Refines the warm-started field `sol` towards the solution of the regularised
 * system, where `H` is the per-voxel compact-symmetric Hessian (`hes`, packed
 * `C*(C+1)/2` last axis), `L` the field regulariser (same per-channel penalties
 * as `field_matvec`), and `g` the gradient (`grd`, `C` last axis). Runs
 * `nb_iter` red-black sweeps and writes the refined solution back into `sol`.
 *
 * @param sol        Field to refine, in/out (*batch, *spatial, C)
 * @param hes        Compact-symmetric Hessian (*batch, *spatial, C*(C+1)/2)
 * @param grd        Gradient (*batch, *spatial, C)
 * @param nb_iter    Number of relaxation iterations
 */
void field_relax(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const double   * voxel_size = nullptr,
    const double   * absolute  = nullptr,
    const double   * membrane  = nullptr,
    const double   * bending   = nullptr,
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          int        nb_iter   = 1,
          int        stream    = 0
);

FF_NAMESPACE_END(FF)

#endif // FF_LIB_REG_FIELD
