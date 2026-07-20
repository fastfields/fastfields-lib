#ifndef FF_CPU_RESTRICT
#define FF_CPU_RESTRICT
#include "dlpack.h"
#include <cstdint>

namespace ff  {
namespace cpu {

/**
 * @brief Restriction: the adjoint (transpose) of the resize prolongation.
 *
 * Maps a fine tensor `inp` to a coarse tensor `out` by scattering each fine
 * sample onto the coarse grid with spline weights. `out` is ACCUMULATED into,
 * so the caller must zero it beforehand.
 *
 * The `ndim` trailing dimensions are spatial; leading dimensions are batch and
 * must have identical shapes in `out` and `inp`. For consistency with resize,
 * pass the reciprocal scale: if resize (coarse->fine) used `scale = coarse/fine`,
 * restriction (fine->coarse) uses `scale = fine/coarse` with the same `shift`.
 *
 * @param out     Output (coarse) tensor (*batch, *outshape), pre-zeroed
 * @param inp     Input (fine) tensor    (*batch, *inshape)
 * @param spline  Spline order applied to every spatial dim (see spline_t)
 * @param bound   Boundary condition applied to every spatial dim (see bound_t)
 * @param shift   Anchor shift (0 aligns voxel centres, 0.5 aligns edges)
 * @param scale   [ndim] per-dim scaling (input-index per output-index)
 * @param ndim    Number of spatial dimensions (1, 2 or 3)
 * @param stream  Cuda stream on which to operate (unused on CPU)
 */
void restriction(
          DLTensor & out    ,
    const DLTensor & inp    ,
          int8_t     spline ,
          int8_t     bound  ,
          double     shift  ,
    const double   * scale  ,
          int        ndim   ,
          int        stream = 0
);

} // namespace cpu
} // namespace ff

#endif // FF_CPU_RESTRICT
