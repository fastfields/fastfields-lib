#ifndef FF_LIB_DISTANCE
#define FF_LIB_DISTANCE
#include "fastfields/core/dlpack.h"
#include <cstdint>
#include "fastfields/core/defines.h"

FF_NAMESPACE_BEGIN(FF_NS)

#ifndef FF_LIB_BOUND_SPLINE_T
#define FF_LIB_BOUND_SPLINE_T
FF_NAMESPACE_BEGIN(bound_t)
using T = int8_t;
static constexpr T Dynamic   = -1; ///< Used to turn-off static implementations in templated classes
static constexpr T Zero      =  0; ///< Zero outside of the FOV
static constexpr T Replicate =  1; ///< Replicate last inbound value = clip coordinates
static constexpr T DCT1      =  2; ///< Symmetric w.r.t. center of the last inbound voxel
static constexpr T DCT2      =  3; ///< Symmetric w.r.t. edge of the last inbound voxel (= Neumann)
static constexpr T DST1      =  4; ///< Antisymmetric w.r.t. center of the last inbound voxel
static constexpr T DST2      =  5; ///< Antisymmetric w.r.t. edge of the last inbound voxel (= Dirichlet)
static constexpr T DFT       =  6; ///< Circular / Wrap around the FOV
static constexpr T NoCheck   =  7; ///< Checks disabled: assume coordinates are inbound
FF_NAMESPACE_END(bound_t)

FF_NAMESPACE_BEGIN(spline_t)
using T = int8_t;
static constexpr T Dynamic       = -1;  ///< Used to turn-off static implementations in templated classes
static constexpr T Nearest       =  0;
static constexpr T Linear        =  1;
static constexpr T Quadratic     =  2;
static constexpr T Cubic         =  3;
static constexpr T FourthOrder   =  4;
static constexpr T FifthOrder    =  5;
static constexpr T SixthOrder    =  6;
static constexpr T SeventhOrder  =  7;
FF_NAMESPACE_END(spline_t)
#endif // FF_LIB_BOUND_SPLINE_T

/**
 * @brief Compute the Euclidean distance transform of a tensor.
 *
 * Must be called with a tensor of type float32 or float64.
 * Must have zeros at the feature locations and +inf elsewhere.
 * The distance is taken along the last dimension.
 *
 * @param inp_out        Input/Output tensor in DLTensor format
 * @param voxel_spacing  Spacing between voxels
 * @param stream         Cuda stream on which to operate
 */
void dt_euclidean(
          DLTensor & inp_out,
          double     voxel_spacing,
          intptr_t   stream = 0
);

/**
 * @brief Compute the L1 distance transform of a tensor.
 *
 * Must be called with a tensor of type float32 or float64.
 * Must have zeros at the feature locations and +inf elsewhere.
 * The distance is taken along the last dimension.
 *
 * @param inp_out        Input/Output tensor in DLTensor format
 * @param voxel_spacing  Spacing between voxels
 * @param stream         Cuda stream on which to operate
 */
void dt_l1(
          DLTensor & inp_out,
          double     voxel_spacing,
          intptr_t   stream = 0
);

/**
 * @brief Compute the distance from a set of points to a spline
 *        using a dictionary approach.
 *
 * @param time      Output tensor for best time (*B,)
 * @param dist      Output tensor for best squared distance (*B,)
 * @param loc       Input tensor for ND location of each point (*B, D)
 * @param coeff     Input tensor for spline coefficients (N, D)
 * @param times     Input tensor for time values to try (K,)
 * @param spline    Spline type (see spline_t)
 * @param bound     Boundary condition type (see bound_t)
 * @param stream    Cuda stream on which to operate
 */
void dt_spline_table(
          DLTensor & time,
          DLTensor & dist,
    const DLTensor & loc,
    const DLTensor & coeff,
    const DLTensor & times,
          int8_t     spline = spline_t::Cubic,
          int8_t     bound  = bound_t::DCT2,
          intptr_t   stream = 0
);

/**
 * @brief Compute the distance from a set of points to a spline
 *        using Brent's method.
 *
 * @param time      Output tensor for best time (*B,)
 * @param dist      Output tensor for best squared distance (*B,)
 * @param loc       Input tensor for ND location of each point (*B, D)
 * @param coeff     Input tensor for spline coefficients (N, D)
 * @param max_iter  Maximum number of iterations
 * @param tol       Tolerance for convergence
 * @param step      Initial step size
 * @param spline    Spline type (see spline_t)
 * @param bound     Boundary condition type (see bound_t)
 * @param stream    Cuda stream on which to operate
 */
void dt_spline_brent(
          DLTensor & time,
          DLTensor & dist,
    const DLTensor & loc,
    const DLTensor & coeff,
          int64_t    max_iter,
          double     tol,
          double     step,
          int8_t     spline = spline_t::Cubic,
          int8_t     bound  = bound_t::DCT2,
          intptr_t   stream = 0
);

/**
 * @brief Compute the distance from a set of points to a spline
 *        using Gauss-Newton optimization.
 *
 * @param time      Output tensor for best time (*B,)
 * @param dist      Output tensor for best squared distance (*B,)
 * @param loc       Input tensor for ND location of each point (*B, D)
 * @param coeff     Input tensor for spline coefficients (N, D)
 * @param max_iter  Maximum number of iterations
 * @param tol       Tolerance for convergence
 * @param spline    Spline type (see spline_t)
 * @param bound     Boundary condition type (see bound_t)
 * @param stream    Cuda stream on which to operate
 */
void dt_spline_gaussnewton(
          DLTensor & time,
          DLTensor & dist,
    const DLTensor & loc,
    const DLTensor & coeff,
          int64_t    max_iter,
          double     tol,
          int8_t     spline = spline_t::Cubic,
          int8_t     bound  = bound_t::DCT2,
          intptr_t   stream = 0
);

/**
 * @brief Compute the distance from a set of points to a triangular mesh.
 *
 * @param dist              Output tensor for squared distances (*B,)
 * @param nearest_vertex    Output tensor for index of nearest vertex (*B,)
 * @param loc               Input tensor for ND location of each point (*B, D)
 * @param vertices          Input tensor for mesh vertices (N, D)
 * @param faces             Input tensor for mesh faces (M, D)
 * @param _signed           Whether to compute signed distances (inside negative)
 * @param naive             Whether to use the naive algorithm (no acceleration structure)
 * @param stream            Cuda stream on which to operate
 */
void dt_mesh(
          DLTensor & dist,
          DLTensor & nearest_vertex,
    const DLTensor & loc,
    const DLTensor & vertices,
    const DLTensor & faces,
          bool       _signed = true,
          bool       naive   = false,
          intptr_t   stream  = 0
);

FF_NAMESPACE_END(FF_NS)

#endif // FF_LIB_DISTANCE
