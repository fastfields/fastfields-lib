#ifndef FF_CPU_PUSHPULL
#define FF_CPU_PUSHPULL
#include "dlpack.h"
#include <cstdint>

namespace ff  {
namespace cpu {

/**
 * @brief Sample (gather / "pull") a spline-encoded volume at arbitrary
 *        coordinates.
 *
 * Tensor layout (channel-last, x-first coordinates):
 *   inp  : (*batch, *inshape,  C)
 *   grid : (*batch, *outshape, D)   with D == ndim (the spatial rank)
 *   out  : (*batch, *outshape, C)
 *
 * @param out          Output tensor (pulled samples)
 * @param inp          Input volume (spline coefficients)
 * @param grid         Sampling coordinates (in voxels, x-first)
 * @param spline       Spline order applied to every spatial dim (see spline_t)
 * @param bound        Boundary condition applied to every spatial dim
 * @param extrapolate  1: always; 0: not past voxel centres; -1: not past edges
 * @param stream       Cuda stream (unused on CPU)
 */
void pull(
          DLTensor & out,
    const DLTensor & inp,
    const DLTensor & grid,
          int8_t     spline      = 2,
          int8_t     bound       = 3,
          int8_t     extrapolate = 1,
          int        stream      = 0
);

/**
 * @brief Splat ("push") values into a volume; numerical adjoint of `pull`.
 *        `out` must be pre-zeroed by the caller (values are accumulated).
 *
 *   inp  : (*batch, *outshape, C)
 *   grid : (*batch, *outshape, D)
 *   out  : (*batch, *inshape,  C)
 */
void push(
          DLTensor & out,
    const DLTensor & inp,
    const DLTensor & grid,
          int8_t     spline      = 2,
          int8_t     bound       = 3,
          int8_t     extrapolate = 1,
          int        stream      = 0
);

/**
 * @brief Splat ones into a volume (== push of an all-ones input).
 *        `out` (*batch, *inshape, 1) must be pre-zeroed.
 */
void count(
          DLTensor & out,
    const DLTensor & grid,
          int8_t     spline      = 2,
          int8_t     bound       = 3,
          int8_t     extrapolate = 1,
          int        stream      = 0
);

/**
 * @brief Sample the spatial gradients of a spline-encoded volume.
 *
 *   inp  : (*batch, *inshape,  C)
 *   grid : (*batch, *outshape, D)
 *   out  : (*batch, *outshape, C, D)
 *
 * @param abs  Whether to use the absolute value of the basis derivatives.
 */
void grad(
          DLTensor & out,
    const DLTensor & inp,
    const DLTensor & grid,
          int8_t     spline      = 2,
          int8_t     bound       = 3,
          int8_t     extrapolate = 1,
          bool       abs         = false,
          int        stream      = 0
);

} // namespace cpu
} // namespace ff

#endif // FF_CPU_PUSHPULL
