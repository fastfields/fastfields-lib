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

} // namespace cpu
} // namespace ff

#endif // FF_CPU_REG_FIELD
