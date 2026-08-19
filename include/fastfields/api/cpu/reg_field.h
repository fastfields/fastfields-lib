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
 * @throws std::invalid_argument if the selected energy is not self-adjoint
 *         under `bound`. The stencil's boundary fold has to be an involution on
 *         the tap set, and more reach folds more taps, so the answer depends on
 *         which penalty is highest-order:
 *
 *             bound      | absolute | membrane | bending
 *             -----------+----------+----------+---------
 *             Replicate  |    ok    |    ok    | REJECT
 *             DCT1       |    ok    |  REJECT  | REJECT
 *             all others |    ok    |    ok    |   ok
 *
 *         DCT1 reflects about the last inbound voxel, so at x=0 the -1 tap
 *         lands on the +1 tap and the operator loses symmetry from reach 1
 *         upwards; Replicate's clamp only bites once a +-2 tap exists.
 *         `absolute` reads no neighbour at all and is accepted everywhere. The
 *         rejection set is measured (assembled `A`, `max|A-A^T|/max|A|`), lives
 *         in `bound::supports_{absolute,membrane,bending}`, and is validated
 *         ONCE per call, never per voxel.
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
 * @brief Diagonal of the regulariser operator, same conventions as
 *        `field_matvec`. Writes into `out` (*batch, *spatial, C).
 *
 * This is the EXACT matrix diagonal at every voxel, boundary voxels included --
 * i.e. exactly what `field_matvec` returns when contracted against a unit
 * vector -- not an interior-only approximation extended to the edges.
 *
 * @throws std::invalid_argument on the same bending/boundary combinations as
 *         `field_matvec`.
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
 * Unlike `field_matvec` / `field_diag` this never rejects a bending + boundary
 * combination: the stencil is written at pure strides and does not consult the
 * boundary at all, so there is a well-defined answer for every condition.
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

} // namespace cpu
} // namespace ff

#endif // FF_CPU_REG_FIELD
