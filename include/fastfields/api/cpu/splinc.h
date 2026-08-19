#ifndef FF_CPU_SPLINC
#define FF_CPU_SPLINC
#include "dlpack.h"
#include <cstdint>

namespace ff  {
namespace cpu {

/**
 * @brief In-place spline coefficient prefilter along the last dimension.
 *
 * Solves the spline interpolation system so that spline interpolation of
 * the resulting coefficients reproduces the original samples. All leading
 * dimensions are treated as batch; filtering is applied along the last axis.
 * To filter along several axes, permute the tensor and call repeatedly.
 *
 * @param inp_out  Input/Output tensor in DLTensor format (float32/float64)
 * @param spline   Spline order (see spline_t; orders 0/1 are no-ops)
 * @param bound    Boundary condition (see bound_t)
 * @param stream   Cuda stream on which to operate (unused on CPU)
 */
void spline_coeff(
          DLTensor & inp_out ,
          int8_t     spline   = 3, // Cubic
          int8_t     bound    = 3, // DCT2
          int        stream   = 0
);

} // namespace cpu
} // namespace ff

#endif // FF_CPU_SPLINC
