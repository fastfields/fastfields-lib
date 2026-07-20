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
 * bending); the highest-order non-zero penalty selects the stencil. With
 * `voxel_size == 1` and only `absolute`, the result is `absolute * inp`;
 * with only `membrane`, it is `membrane` times the discrete negative
 * Laplacian of the field.
 *
 * @param out         Output tensor (*batch, *spatial, ndim)
 * @param inp         Input  tensor (*batch, *spatial, ndim)
 * @param voxel_size  [ndim] spatial voxel size (nullptr -> all ones)
 * @param absolute    Absolute (L2) penalty weight
 * @param membrane    Membrane (first-order) penalty weight
 * @param bending     Bending (second-order) penalty weight
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
          int8_t     bound     = 0,
          int        ndim      = 1,
          int        stream    = 0
);

} // namespace cpu
} // namespace ff

#endif // FF_CPU_REG_FLOW
