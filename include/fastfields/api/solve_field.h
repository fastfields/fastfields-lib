#pragma once
#include <fastfields/core/dlpack.h>
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
 * @brief Solve `(H + L) x = g` by Jacobi-preconditioned conjugate gradients.
 *
 * `H` is the per-voxel compact-symmetric data-term Hessian (`hes`, packed
 * `C*(C+1)/2` last axis) and `L` the field regulariser (same per-channel
 * `absolute`/`membrane`/`bending` penalties and boundary conventions as
 * `field_matvec`). This is the same system `field_relax` attacks with
 * Gauss-Seidel sweeps and that `field_forward` applies; CG is the Krylov
 * alternative, preconditioned by `field_precond`'s `(H + diag(L))^-1`.
 *
 * `sol` is both the warm start and the output, mirroring `field_relax`'s
 * in-place contract: pass a zero-filled buffer for a cold start, or a
 * previous estimate to continue from it.
 *
 * Iteration stops after `nb_iter` steps or as soon as the residual norm
 * `||g - (H + L) x||` drops below `tol * ||g||` (`tol <= 0` disables the
 * early exit). It also stops early if the operator turns out not to be
 * positive definite along the current search direction.
 *
 * @note CPU only for now -- the CUDA backend is not wired yet, and CUDA
 *       tensors are rejected. See fastfields-lib#34.
 *
 * @param sol          Warm start in, solution out (*batch, *spatial, C)
 * @param hes          Compact-symmetric Hessian (*batch, *spatial, C*(C+1)/2)
 * @param grd          Gradient / right-hand side (*batch, *spatial, C)
 * @param voxel_size   [ndim] spatial voxel size (nullptr -> all ones)
 * @param absolute     [C] absolute (L2) penalty weights (nullptr -> zeros)
 * @param membrane     [C] membrane penalty weights (nullptr -> disabled)
 * @param bending      [C] bending penalty weights (nullptr -> disabled)
 * @param bound        Boundary condition applied to every spatial dim
 * @param ndim         Number of spatial dimensions (1, 2 or 3)
 * @param nb_iter      Maximum number of CG iterations
 * @param tol          Relative residual tolerance (<= 0 -> run all iterations)
 * @param nb_iter_out  If non-null, receives the number of iterations run
 * @param residual_out If non-null, receives the final relative residual
 * @param stream       Cuda stream on which to operate
 */
void field_cg(
          DLTensor & sol         ,
    const DLTensor & hes         ,
    const DLTensor & grd         ,
    const double   * voxel_size  = nullptr,
    const double   * absolute    = nullptr,
    const double   * membrane    = nullptr,
    const double   * bending     = nullptr,
          int8_t     bound       = bound_t::DCT2,
          int        ndim        = 1,
          int        nb_iter     = 32,
          double     tol         = 1e-8,
          int      * nb_iter_out = nullptr,
          double   * residual_out= nullptr,
          intptr_t   stream      = 0
);

FF_NAMESPACE_END(FF_NS)
