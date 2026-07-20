#ifndef FF_CPU_DISTANCE_L1
#define FF_CPU_DISTANCE_L1
#include "kernels/cuda_switch.h"
#include "kernels/distance.h"
#include "kernels/batch.h"
#include "kernels/parallel.h"
#include "kernels/utils.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_l1)

template <typename scalar_t = float, typename offset_t = int64_t>
inline void
dt(
          offset_t   ndim   ,   // number of dimensions
          scalar_t * f      ,   // pointer to data [*batch, n]
          scalar_t   w      ,   // pixel spacing
    const offset_t * size   ,   // [ndim] data shape   == (*batch, n)
    const offset_t * stride )   // [ndim] data strides
{
    offset_t nbatch = ndim - 1;
    offset_t n = size[nbatch];
    offset_t s = stride[nbatch];

    offset_t numel = prod(size, nbatch);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset = index2offset(i, nbatch, size, stride);
        kernel(f + offset, n, s, w);
    }});
}

FF_NAMESPACE_END(distance_l1)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_CPU_DISTANCE_L1
