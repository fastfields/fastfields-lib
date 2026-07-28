#ifndef FF_CPU_REG_FLOW
#define FF_CPU_REG_FLOW
#include "dlpack.h"
#include <cstdint>

namespace ff  {
namespace cpu {

/**
 * @brief Apply a spatial regulariser operator to a vector flow field.
 *
 * The tensor layout is (*batch, *spatial, C) where the last axis holds the
 * `C == ndim` flow components and the `ndim` axes before it are spatial.
 * The operator is the sum of the requested penalties (absolute, membrane,
 * bending, and the linear-elastic `shears`/`div` Lamé terms); the highest-order
 * non-zero penalty selects the stencil (a non-zero `shears`/`div` selects the
 * full combined stencil). With `voxel_size == 1` and only `absolute`, the
 * result is `absolute * inp`; with only `membrane`, it is `membrane` times the
 * discrete negative Laplacian of the field.
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
 * @param stream      Cuda stream on which to operate (unused on CPU)
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
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief `flow_matvec` variant that accumulates into `out`: `out += L(inp)`,
 *        instead of overwriting it. Same conventions otherwise.
 */
void flow_matvec_add(
          DLTensor & out       ,
    const DLTensor & inp       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief `flow_matvec` variant that subtracts from `out`: `out -= L(inp)`,
 *        instead of overwriting it. Same conventions otherwise.
 */
void flow_matvec_sub(
          DLTensor & out       ,
    const DLTensor & inp       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief Diagonal of the regulariser operator (same conventions as
 *        `flow_matvec`). Writes into `out` (*batch, *spatial, ndim).
 */
void flow_diag(
          DLTensor & out       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief Materialise the Toeplitz convolution kernel (stencil) of the flow
 *        regulariser operator.
 *
 * Writes the small stencil that, convolved with a flow field, reproduces
 * `flow_matvec` (same penalties/conventions). The highest-order non-zero
 * penalty selects the stencil; a non-zero `shears`/`div` selects the full
 * linear-elastic stencil, which couples the flow channels.
 *
 * The output rank depends on whether the Lamé terms are active:
 *   - `shears == div == 0`: `out` is a **vector** of per-channel kernels,
 *     shape `(*batch, *spatial, ndim)` (channels are independent).
 *   - `shears != 0 || div != 0`: `out` is a **matrix** of kernels,
 *     shape `(*batch, *spatial, ndim, ndim)` (cross-channel coupling).
 * The spatial extent must be at least the stencil width in every spatial dim
 * (1 if absolute-only, 3 if membrane/Lamé, 5 if bending) and is centred.
 *
 * @param out         Output stencil (*batch, *spatial, ndim[, ndim])
 * @param voxel_size  [ndim] spatial voxel size (nullptr -> all ones)
 * @param absolute    Absolute (L2) penalty weight
 * @param membrane    Membrane (first-order) penalty weight
 * @param bending     Bending (second-order) penalty weight
 * @param shears      Linear-elastic shears (Lamé mu) penalty weight
 * @param div         Linear-elastic divergence (Lamé lambda) penalty weight
 * @param bound       Boundary condition applied to every spatial dim
 * @param ndim        Number of spatial dimensions (1, 2 or 3)
 * @param stream      Cuda stream on which to operate (unused on CPU)
 */
void flow_kernel(
          DLTensor & out       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief In-place relaxation (Gauss-Seidel) sweeps solving `(H + L) x = g`.
 *
 * Refines the warm-started flow `sol` towards the solution of the regularised
 * system, where `H` is the per-voxel symmetric Hessian (`hes`, packed
 * `ndim*(ndim+1)/2` last axis), `L` the flow regulariser (same penalties as
 * `flow_matvec`), and `g` the gradient (`grd`, `ndim` last axis). Runs
 * `nb_iter` red-black sweeps and writes the refined solution back into `sol`.
 *
 * @param sol        Flow to refine, in/out (*batch, *spatial, ndim)
 * @param hes        Symmetric Hessian (*batch, *spatial, ndim*(ndim+1)/2)
 * @param grd        Gradient (*batch, *spatial, ndim)
 * @param nb_iter    Number of relaxation iterations
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
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        nb_iter   = 1,
          int        stream    = 0
);

/**
 * @brief Forward application of the regularised system: `out = (H + L) x`,
 *        where `H` is the per-voxel symmetric Hessian and `L` the flow
 *        regulariser (same penalties/conventions as `flow_matvec`).
 *        `flow_relax` solves this system; `flow_precond` approximates its
 *        inverse.
 *
 * @param out        Output tensor (*batch, *spatial, ndim)
 * @param hes        Symmetric Hessian (*batch, *spatial, ndim*(ndim+1)/2)
 * @param inp        Input tensor (*batch, *spatial, ndim)
 */
void flow_forward(
          DLTensor & out       ,
    const DLTensor & hes       ,
    const DLTensor & inp       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief Jacobi-type preconditioner solve: `out = (H + diag(L)) \ grd`,
 *        where `diag(L)` is `flow_diag`'s regulariser diagonal (same
 *        penalties/conventions as `flow_matvec`) and `H` the per-voxel
 *        symmetric Hessian.
 *
 * @param out        Output tensor (*batch, *spatial, ndim)
 * @param hes        Symmetric Hessian (*batch, *spatial, ndim*(ndim+1)/2)
 * @param grd        Gradient (*batch, *spatial, ndim)
 */
void flow_precond(
          DLTensor & out       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief In-place variant of `flow_precond`: `sol` holds the gradient on
 *        entry and the preconditioned solution on exit.
 *
 * @param sol        Gradient in, preconditioned solution out (*batch, *spatial, ndim)
 * @param hes        Symmetric Hessian (*batch, *spatial, ndim*(ndim+1)/2)
 */
void flow_precond_(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief Joint-reweighted-least-squares (JRLS) variant of `flow_matvec`.
 *
 * Same conventions as `flow_matvec`, with an additional per-voxel weight map
 * `wgt` (*batch, *spatial, 1), shared across all `ndim` flow components, that
 * spatially modulates the penalty strength (e.g. for edge-preserving / robust
 * regularisation). A non-zero `shears`/`div` selects the weighted Lamé
 * stencil (`membrane`/`absolute` still apply, folded into the same kernel);
 * otherwise the weighted membrane stencil is used (degenerating to a pure
 * weighted diagonal when `membrane == 0`). `bending` is **not supported**
 * with weighting (throws if non-zero) -- jitfields never wired this
 * combination at the loop level either.
 *
 * @param out         Output tensor (*batch, *spatial, ndim)
 * @param inp         Input  tensor (*batch, *spatial, ndim)
 * @param wgt         Weight tensor (*batch, *spatial, 1)
 */
void flow_matvec_rls(
          DLTensor & out       ,
    const DLTensor & inp       ,
    const DLTensor & wgt       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief JRLS variant of `flow_diag`, same weight-map conventions as
 *        `flow_matvec_rls`. Writes into `out` (*batch, *spatial, ndim).
 */
void flow_diag_rls(
          DLTensor & out       ,
    const DLTensor & wgt       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

/**
 * @brief JRLS variant of `flow_relax`, same weight-map conventions as
 *        `flow_matvec_rls`.
 *
 * @param sol        Flow to refine, in/out (*batch, *spatial, ndim)
 * @param hes        Symmetric Hessian (*batch, *spatial, ndim*(ndim+1)/2)
 * @param grd        Gradient (*batch, *spatial, ndim)
 * @param wgt        Weight tensor (*batch, *spatial, 1)
 * @param nb_iter    Number of relaxation iterations
 */
void flow_relax_rls(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const DLTensor & wgt       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        nb_iter   = 1,
          int        stream    = 0
);

} // namespace cpu
} // namespace ff

#endif // FF_CPU_REG_FLOW
