#ifndef FF_CPU_REG_FIELD
#define FF_CPU_REG_FIELD
#include "dlpack.h"
#include <cstdint>

namespace ff  {
namespace cpu {

/**
 * @brief Apply a spatial regulariser operator to a multi-channel field.
 *
 * The tensor layout is (*batch, *spatial, C) where the last axis holds the `C`
 * field channels and the `ndim` axes before it are spatial. The penalties are
 * per-channel weight vectors of length `C`; the highest-order non-null penalty
 * selects the finite-difference stencil. With `voxel_size == 1` and only
 * `absolute`, the result is a per-channel scaling `out[...,c] = absolute[c] *
 * inp[...,c]`; with only `membrane`, it is `membrane[c]` times the discrete
 * negative Laplacian of channel `c`.
 *
 * @param out         Output tensor (*batch, *spatial, C)
 * @param inp         Input  tensor (*batch, *spatial, C)
 * @param voxel_size  [ndim] spatial voxel size (nullptr -> all ones)
 * @param absolute    [C] absolute (L2) penalty weights (nullptr -> zeros)
 * @param membrane    [C] membrane penalty weights (nullptr -> disabled)
 * @param bending     [C] bending penalty weights (nullptr -> disabled)
 * @param bound       Boundary condition applied to every spatial dim
 * @param ndim        Number of spatial dimensions (1, 2 or 3)
 * @param stream      Cuda stream on which to operate (unused on CPU)
 */
void field_matvec(
          DLTensor & out       ,
    const DLTensor & inp       ,
    const double   * voxel_size = nullptr,
    const double   * absolute  = nullptr,
    const double   * membrane  = nullptr,
    const double   * bending   = nullptr,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief `field_matvec` variant that accumulates into `out`: `out += L(inp)`,
 *        instead of overwriting it. Same conventions otherwise.
 */
void field_matvec_add(
          DLTensor & out       ,
    const DLTensor & inp       ,
    const double   * voxel_size = nullptr,
    const double   * absolute  = nullptr,
    const double   * membrane  = nullptr,
    const double   * bending   = nullptr,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief `field_matvec` variant that subtracts from `out`: `out -= L(inp)`,
 *        instead of overwriting it. Same conventions otherwise.
 */
void field_matvec_sub(
          DLTensor & out       ,
    const DLTensor & inp       ,
    const double   * voxel_size = nullptr,
    const double   * absolute  = nullptr,
    const double   * membrane  = nullptr,
    const double   * bending   = nullptr,
          int8_t     bound     = 0,
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
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief Materialise the Toeplitz convolution kernel (stencil) of the field
 *        regulariser operator (same penalties/conventions as `field_matvec`).
 *
 * Writes the small per-channel stencil that, convolved with a field, reproduces
 * `field_matvec`. The output is a **vector** of per-channel kernels, shape
 * `(*batch, *spatial, C)` (the field regulariser never couples channels). The
 * spatial extent must be at least the stencil width in every spatial dim
 * (1 if absolute-only, 3 if membrane, 5 if bending) and is centred.
 *
 * @param out         Output stencil (*batch, *spatial, C)
 * @param voxel_size  [ndim] spatial voxel size (nullptr -> all ones)
 * @param absolute    [C] absolute (L2) penalty weights (nullptr -> zeros)
 * @param membrane    [C] membrane penalty weights (nullptr -> disabled)
 * @param bending     [C] bending penalty weights (nullptr -> disabled)
 * @param bound       Boundary condition applied to every spatial dim
 * @param ndim        Number of spatial dimensions (1, 2 or 3)
 * @param stream      Cuda stream on which to operate (unused on CPU)
 */
void field_kernel(
          DLTensor & out       ,
    const double   * voxel_size = nullptr,
    const double   * absolute  = nullptr,
    const double   * membrane  = nullptr,
    const double   * bending   = nullptr,
          int8_t     bound     = 0,
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
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        nb_iter   = 1,
          int        stream    = 0
);

/**
 * @brief Forward application of the regularised system: `out = (H + L) x`,
 *        where `H` is the per-voxel compact-symmetric Hessian and `L` the
 *        field regulariser (same penalties/conventions as `field_matvec`).
 *        `field_relax` solves this system; `field_precond` approximates its
 *        inverse.
 *
 * @param out        Output tensor (*batch, *spatial, C)
 * @param hes        Compact-symmetric Hessian (*batch, *spatial, C*(C+1)/2)
 * @param inp        Input tensor (*batch, *spatial, C)
 */
void field_forward(
          DLTensor & out       ,
    const DLTensor & hes       ,
    const DLTensor & inp       ,
    const double   * voxel_size = nullptr,
    const double   * absolute  = nullptr,
    const double   * membrane  = nullptr,
    const double   * bending   = nullptr,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief Jacobi-type preconditioner solve: `out = (H + diag(L)) \ grd`,
 *        where `diag(L)` is `field_diag`'s regulariser diagonal (same
 *        penalties/conventions as `field_matvec`) and `H` the per-voxel
 *        compact-symmetric Hessian.
 *
 * @param out        Output tensor (*batch, *spatial, C)
 * @param hes        Compact-symmetric Hessian (*batch, *spatial, C*(C+1)/2)
 * @param grd        Gradient (*batch, *spatial, C)
 */
void field_precond(
          DLTensor & out       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const double   * voxel_size = nullptr,
    const double   * absolute  = nullptr,
    const double   * membrane  = nullptr,
    const double   * bending   = nullptr,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief In-place variant of `field_precond`: `sol` holds the gradient on
 *        entry and the preconditioned solution on exit.
 *
 * @param sol        Gradient in, preconditioned solution out (*batch, *spatial, C)
 * @param hes        Compact-symmetric Hessian (*batch, *spatial, C*(C+1)/2)
 */
void field_precond_(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const double   * voxel_size = nullptr,
    const double   * absolute  = nullptr,
    const double   * membrane  = nullptr,
    const double   * bending   = nullptr,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief Reweighted-least-squares (RLS/JRLS) variant of `field_matvec`.
 *
 * Same conventions as `field_matvec`, with an additional per-voxel weight
 * map `wgt` that spatially modulates the penalty strength (e.g. for
 * edge-preserving / robust regularisation). `wgt` has shape
 * `(*batch, *spatial, 1)` for a single weight shared across all channels
 * (RLS), or `(*batch, *spatial, C)` for a per-channel weight (JRLS,
 * `C` matching `out`'s channel count) -- the trailing dimension of `wgt`
 * selects which mode is used.
 *
 * All three orders (`absolute`, `membrane`, `bending`) are verified
 * self-adjoint under an arbitrary positive weight map, for RLS and JRLS,
 * under DCT2/DST2/DFT boundaries. Zero boundary is not yet covered for
 * `bending`: an out-of-bounds weight-map read at that boundary is a
 * separately-tracked issue (fastfields-kernels#34, finding S1).
 *
 * @param out         Output tensor (*batch, *spatial, C)
 * @param inp         Input  tensor (*batch, *spatial, C)
 * @param wgt         Weight tensor (*batch, *spatial, 1 or C)
 */
void field_matvec_rls(
          DLTensor & out       ,
    const DLTensor & inp       ,
    const DLTensor & wgt       ,
    const double   * voxel_size = nullptr,
    const double   * absolute  = nullptr,
    const double   * membrane  = nullptr,
    const double   * bending   = nullptr,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief RLS/JRLS variant of `field_diag`, same weight-map conventions as
 *        `field_matvec_rls`. Writes into `out` (*batch, *spatial, C).
 */
void field_diag_rls(
          DLTensor & out       ,
    const DLTensor & wgt       ,
    const double   * voxel_size = nullptr,
    const double   * absolute  = nullptr,
    const double   * membrane  = nullptr,
    const double   * bending   = nullptr,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief RLS/JRLS variant of `field_relax`, same weight-map conventions as
 *        `field_matvec_rls`.
 *
 * @param sol        Field to refine, in/out (*batch, *spatial, C)
 * @param hes        Compact-symmetric Hessian (*batch, *spatial, C*(C+1)/2)
 * @param grd        Gradient (*batch, *spatial, C)
 * @param wgt        Weight tensor (*batch, *spatial, 1 or C)
 * @param nb_iter    Number of relaxation iterations
 */
void field_relax_rls(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const DLTensor & wgt       ,
    const double   * voxel_size = nullptr,
    const double   * absolute  = nullptr,
    const double   * membrane  = nullptr,
    const double   * bending   = nullptr,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        nb_iter   = 1,
          int        stream    = 0
);

} // namespace cpu
} // namespace ff

#endif // FF_CPU_REG_FIELD
