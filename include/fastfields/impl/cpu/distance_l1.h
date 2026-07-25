#ifndef FF_CPU_DISTANCE_L1
#define FF_CPU_DISTANCE_L1
#include <teeny/teeny.h>
#include "kernels/cuda_switch.h"
#include "kernels/distance.h"
#include "kernels/parallel.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_l1)

// 1-D L1 distance sweep along the LAST axis, batched over every leading axis.
//
// teeny's peel replaces the hand-written index2offset batch plumbing: an
// `anyrank` over (*batch, n) hands out each rank-1 line as a view whose data
// pointer already has the (arbitrarily strided) batch offset folded in, so the
// sweep kernel is called unchanged on the raw pointer + stride. Add or drop
// batch axes by changing the data, not the loop.
template <typename scalar_t = float, typename offset_t = int64_t>
inline void
dt(
          offset_t   ndim   ,   // number of dimensions
          scalar_t * f      ,   // pointer to data [*batch, n]
          scalar_t   w      ,   // pixel spacing
    const offset_t * size   ,   // [ndim] data shape   == (*batch, n)
    const offset_t * stride )   // [ndim] data strides
{
    auto at = tny::as_anyrank(f, size, stride, static_cast<int>(ndim), tny::copy_meta);
    const offset_t nlines = at.template size_front<-1>();
    parallel_for(0, nlines, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = start; i < end; ++i) {
        auto line = at.template peel_front_at<-1>(i);
        kernel<offset_t, scalar_t>(line.data(), line.extent(0), line.stride(0), w);
    }});
}

FF_NAMESPACE_END(distance_l1)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_CPU_DISTANCE_L1
