// "Distance Transforms of Sampled Functions"
// Pedro F. Felzenszwalb & Daniel P. Huttenlocher
// Theory of Computing (2012)
// https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
#ifndef FF_DISTANCE_L1
#define FF_DISTANCE_L1
#include <fastfields/core/cuda_switch.h>
#include "../utils.h"

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_l1)

// Update the upper bound on the L1 distance.
//
// This function processes the data along a single dimension.
// Once it's been applied to all dimensions, `f` contains the L1 distance.
// Initially, `f` must contain "zero" in the background and "inf" in
// the foreground.
//
// f      - [inp] upper bound of the distance to nearest "0"
//        - [out] updated upper bound
// size   - Number of voxels along the current dimension
// stride - Stride between two voxels along the current dimension
// w      - Voxel size along the current dimension
template <typename offset_t, typename scalar_t>
FF_CUDEV
void kernel(scalar_t * f, offset_t size, offset_t stride, scalar_t w)
{
  if (size == 1) return;

  scalar_t tmp = *f;
  f += stride;
  for (offset_t i = 1; i < size; ++i, f += stride) {
     tmp = min(tmp + w, *f);
     *f = tmp;
  }
  f -= 2 * stride;
  for (offset_t i = size-2; i >= 0; --i, f -= stride) {
     tmp = min(tmp + w, *f);
     *f = tmp;
  }
}

FF_NAMESPACE_END(distance_l1)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
#endif // FF_DISTANCE_L1
