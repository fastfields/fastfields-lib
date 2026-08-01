#ifndef FF_CUDA_REG_FIELD
#define FF_CUDA_REG_FIELD
#include "dlpack.h"
#include <cstdint>

namespace ff  {
namespace cuda {

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
void field_addmatvec_(
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
void field_submatvec_(
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
 * @brief `field_diag` variant that accumulates into `out`: `out += diag(L)`,
 *        instead of overwriting it. Same conventions otherwise.
 *
 * **In-place only**, mirroring the original jitfields C-level `op='+'`
 * entry point. There is deliberately no out-of-place counterpart: an
 * out-of-place accumulate is a caller-side clone followed by this same call.
 */
void field_adddiag_(
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
 * @brief `field_diag` variant that accumulates into `out`: `out -= diag(L)`,
 *        instead of overwriting it. Same conventions otherwise.
 *
 * **In-place only**, mirroring the original jitfields C-level `op='-'`
 * entry point. There is deliberately no out-of-place counterpart: an
 * out-of-place accumulate is a caller-side clone followed by this same call.
 */
void field_subdiag_(
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
 * @brief Materialise the per-channel Toeplitz convolution kernel (stencil) of
 *        the field regulariser. See the cpu-lib declaration; forwards a CUDA
 *        stream. Output is a vector stencil `(*batch, *spatial, C)` (no
 *        cross-channel matrix case).
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
 * @brief `field_kernel` variant that accumulates into `out`: `out += K (the stencil)`,
 *        instead of overwriting it. Same conventions otherwise.
 *
 * **In-place only**, mirroring the original jitfields C-level `op='+'`
 * entry point. There is deliberately no out-of-place counterpart: an
 * out-of-place accumulate is a caller-side clone followed by this same call.
 */
void field_addkernel_(
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
 * @brief `field_kernel` variant that accumulates into `out`: `out -= K (the stencil)`,
 *        instead of overwriting it. Same conventions otherwise.
 *
 * **In-place only**, mirroring the original jitfields C-level `op='-'`
 * entry point. There is deliberately no out-of-place counterpart: an
 * out-of-place accumulate is a caller-side clone followed by this same call.
 */
void field_subkernel_(
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
 *        Same conventions as the CPU `field_relax`; forwards the CUDA stream.
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
 *        inverse. Forwards a CUDA stream.
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
 *        compact-symmetric Hessian. Forwards a CUDA stream.
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
 * @brief Reweighted-least-squares (RLS/JRLS) variant of `field_matvec`. Same
 *        conventions as the CPU `field_matvec_rls`; forwards the CUDA stream.
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

} // namespace cuda
} // namespace ff

#endif // FF_CUDA_REG_FIELD
