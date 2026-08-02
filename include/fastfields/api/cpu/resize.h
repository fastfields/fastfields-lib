#ifndef FF_CPU_RESIZE
#define FF_CPU_RESIZE
#include "dlpack.h"
#include <cstdint>

namespace ff  {
namespace cpu {

/**
 * @brief Resample (prolongation) a tensor to a new shape using spline
 *        interpolation. This is the raw interpolating "resize"; to obtain a
 *        proper (interpolating) resize, prefilter the input with `spline_coeff`
 *        beforehand.
 *
 * The `ndim` trailing dimensions are spatial; leading dimensions are batch and
 * must have identical shapes in `out` and `inp`. For each output voxel with
 * spatial index `loc`, the input is sampled at `x = scale*(loc + shift) - shift`.
 *
 * @param out     Output tensor (*batch, *outshape)
 * @param inp     Input tensor  (*batch, *inshape)
 * @param spline  Spline order applied to every spatial dim (see spline_t)
 * @param bound   Boundary condition applied to every spatial dim (see bound_t)
 * @param shift   Anchor shift (0 aligns voxel centres, 0.5 aligns edges)
 * @param scale   [ndim] per-dim scaling (input-index per output-index)
 * @param ndim    Number of spatial dimensions (1, 2 or 3)
 * @param stream  Cuda stream on which to operate (unused on CPU)
 */
void resample(
          DLTensor & out    ,
    const DLTensor & inp    ,
          int8_t     spline ,
          int8_t     bound  ,
          double     shift  ,
    const double   * scale  ,
          int        ndim   ,
          intptr_t   stream = 0
);

} // namespace cpu
} // namespace ff

#endif // FF_CPU_RESIZE
