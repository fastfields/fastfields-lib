#ifndef FF_LIB_REG_FLOW
#define FF_LIB_REG_FLOW
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
 * @brief Apply a spatial regulariser operator to a vector flow field.
 *
 * Layout is (*batch, *spatial, C) with `C == ndim` flow components in the last
 * axis. The operator is the sum of the requested penalties (absolute, membrane,
 * bending, and the linear-elastic `shears`/`div` Lamé terms); the highest-order
 * non-zero penalty selects the finite-difference stencil (a non-zero
 * `shears`/`div` selects the full combined stencil). With `voxel_size == 1` and
 * only `absolute`, the result is `absolute * inp`; with only `membrane`, it is
 * `membrane` times the discrete negative Laplacian of the field.
 *
 * @param out         Output tensor (*batch, *spatial, ndim)
 * @param inp         Input  tensor (*batch, *spatial, ndim)
 * @param voxel_size  [ndim] spatial voxel size (nullptr -> all ones)
 * @param absolute    Absolute (L2) penalty weight
 * @param membrane    Membrane (first-order) penalty weight
 * @param bending     Bending (second-order) penalty weight
 * @param shears      Linear-elastic shears (Lamé mu) penalty weight
 * @param div         Linear-elastic divergence (Lamé lambda) penalty weight
 * @param bound       Boundary condition applied to every spatial dim
 * @param ndim        Number of spatial dimensions (1, 2 or 3)
 * @param stream      Cuda stream on which to operate
 */
void flow_matvec(
          DLTensor & out       ,
    const DLTensor & inp       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief Diagonal (preconditioner) of the regulariser operator, same
 *        conventions as `flow_matvec`. Writes into `out` (*batch, *spatial, ndim).
 */
void flow_diag(
          DLTensor & out       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief Materialise the Toeplitz convolution kernel (stencil) of the flow
 *        regulariser operator (same penalties/conventions as `flow_matvec`).
 *
 * Writes the small stencil that, convolved with a flow field, reproduces
 * `flow_matvec`. Output is `(*batch, *spatial, ndim)` for the per-channel
 * vector stencil, or `(*batch, *spatial, ndim, ndim)` when `shears`/`div`
 * select the cross-channel Lamé matrix stencil. The spatial extent must be at
 * least the stencil width (1 absolute-only / 3 membrane+Lamé / 5 bending) and
 * is centred.
 */
void flow_kernel(
          DLTensor & out       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief In-place relaxation (Gauss-Seidel) sweeps solving `(H + L) x = g`,
 *        refining the warm-started flow `sol` (*batch, *spatial, ndim) given a
 *        per-voxel symmetric Hessian `hes` (*batch, *spatial, ndim*(ndim+1)/2)
 *        and gradient `grd` (*batch, *spatial, ndim). Penalties as in
 *        `flow_matvec`; runs `nb_iter` iterations.
 */
void flow_relax(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          int        nb_iter   = 1,
          int        stream    = 0
);

FF_NAMESPACE_END(FF)

#endif // FF_LIB_REG_FLOW
