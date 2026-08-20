#ifndef FF_LIB_REG_FLOW
#define FF_LIB_REG_FLOW
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
          intptr_t   stream    = 0
);

/**
 * @brief `flow_matvec` variant that accumulates into `out`: `out += L(inp)`,
 *        instead of overwriting it. Same conventions otherwise.
 */
void flow_addmatvec_(
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
          intptr_t   stream    = 0
);

/**
 * @brief `flow_matvec` variant that subtracts from `out`: `out -= L(inp)`,
 *        instead of overwriting it. Same conventions otherwise.
 */
void flow_submatvec_(
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
          intptr_t   stream    = 0
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
          intptr_t   stream    = 0
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
          intptr_t   stream    = 0
);

/**
 * @brief In-place relaxation (Gauss-Seidel) sweeps solving `(H + L) x = g`,
 *        refining the warm-started flow `sol` (*batch, *spatial, ndim) given a
 *        per-voxel symmetric Hessian `hes` (*batch, *spatial, ndim*(ndim+1)/2)
 *        and gradient `grd` (*batch, *spatial, ndim). Penalties as in
 *        `flow_matvec`; runs `nb_iter` iterations.
 */
/**
 * @brief `flow_diag` variant that accumulates into `out`: `out += diag(L)`.
 *
 * **In-place only** (jitfields `op='+'`); see `flow_addmatvec_`.
 */
void flow_adddiag_(
          DLTensor & out       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          intptr_t   stream    = 0
);

/**
 * @brief `flow_diag` variant that accumulates into `out`: `out -= diag(L)`.
 *
 * **In-place only** (jitfields `op='-'`); see `flow_addmatvec_`.
 */
void flow_subdiag_(
          DLTensor & out       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          intptr_t   stream    = 0
);

/**
 * @brief `flow_kernel` variant that accumulates into `out`: `out += K (the stencil)`.
 *
 * **In-place only** (jitfields `op='+'`); see `flow_addmatvec_`.
 */
void flow_addkernel_(
          DLTensor & out       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          intptr_t   stream    = 0
);

/**
 * @brief `flow_kernel` variant that accumulates into `out`: `out -= K (the stencil)`.
 *
 * **In-place only** (jitfields `op='-'`); see `flow_addmatvec_`.
 */
void flow_subkernel_(
          DLTensor & out       ,
    const double   * voxel_size = nullptr,
          double     absolute  = 0.0,
          double     membrane  = 0.0,
          double     bending   = 0.0,
          double     shears    = 0.0,
          double     div       = 0.0,
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          intptr_t   stream    = 0
);

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
          intptr_t   stream    = 0
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
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          intptr_t   stream    = 0
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
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          intptr_t   stream    = 0
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
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          intptr_t   stream    = 0
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
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          intptr_t   stream    = 0
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
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          intptr_t   stream    = 0
);

/**
 * @brief JRLS variant of `flow_relax`, same weight-map conventions as
 *        `flow_matvec_rls`.
 *
 * @param sol        Flow to refine, in/out (*batch, *spatial, ndim)
 * @param hes        Symmetric Hessian (*batch, *spatial, ndim*(ndim+1)/2)
 * @param grd        Gradient (*batch, *spatial, ndim)
 * @param wgt        Weight tensor (*batch, *spatial, 1)
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
          int8_t     bound     = bound_t::DCT2,
          int        ndim      = 1,
          int        nb_iter   = 1,
          intptr_t   stream    = 0
);

FF_NAMESPACE_END(FF_NS)

#endif // FF_LIB_REG_FLOW
