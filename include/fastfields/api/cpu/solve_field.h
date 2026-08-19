#ifndef FF_CPU_SOLVE_FIELD
#define FF_CPU_SOLVE_FIELD
#include "fastfields/core/dlpack.h"
#include <cstdint>

namespace ff  {
namespace cpu {

/**
 * @brief Solve `(H + L) x = g` by Jacobi-preconditioned conjugate gradients.
 *
 * `H` is the per-voxel compact-symmetric data-term Hessian (`hes`, packed
 * `C*(C+1)/2` last axis) and `L` the field regulariser (same per-channel
 * `absolute`/`membrane`/`bending` penalties and boundary conventions as
 * `field_matvec`); together they form the same symmetric positive-definite
 * system that `field_relax` attacks with Gauss-Seidel sweeps. Applying the
 * operator is exactly `field_forward`, and the preconditioner is
 * `field_precond`'s `(H + diag(L))^-1` -- except that the regulariser
 * diagonal, which does not change between iterations, is computed once up
 * front and reused, so a CG iteration costs one `field_forward` and one
 * `sym_solve` rather than a fresh `field_diag` per step.
 *
 * `sol` is both the warm start and the output, mirroring `field_relax`'s
 * in-place contract: pass a zero-filled buffer for a cold start, or a
 * previous estimate to continue from it.
 *
 * Iteration stops after `nb_iter` steps or as soon as the residual norm
 * `||g - (H + L) x||` drops below `tol * ||g||` (`tol <= 0` disables the
 * early exit and always runs the full `nb_iter` steps). CG also stops
 * early -- reporting the iterations completed so far -- if the system
 * turns out not to be positive definite along the current search
 * direction (`<p, (H+L)p> <= 0`), which would otherwise divide by a
 * non-positive curvature.
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
 * @param stream       Cuda stream on which to operate (unused on CPU)
 */
void field_cg(
          DLTensor & sol         ,
    const DLTensor & hes         ,
    const DLTensor & grd         ,
    const double   * voxel_size  = nullptr,
    const double   * absolute    = nullptr,
    const double   * membrane    = nullptr,
    const double   * bending     = nullptr,
          int8_t     bound       = 0,
          int        ndim        = 1,
          int        nb_iter     = 32,
          double     tol         = 1e-8,
          int      * nb_iter_out = nullptr,
          double   * residual_out= nullptr,
          intptr_t   stream      = 0
);

} // namespace cpu
} // namespace ff

#endif // FF_CPU_SOLVE_FIELD
