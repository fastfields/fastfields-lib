#ifndef FF_CPU_DISTANCE_EUCLIDEAN
#define FF_CPU_DISTANCE_EUCLIDEAN
#include <teeny/teeny.h>
#include "kernels/cuda_switch.h"
#include "kernels/distance.h"
#include "kernels/parallel.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_e)

// Squared-Euclidean distance sweep (lower-envelope-of-parabolas) along the LAST
// axis, batched over every leading axis. As in distance_l1, teeny's `peel`
// replaces the index2offset batch plumbing (each rank-1 line already carries its
// batch offset in the pointer); the sweep kernel and its per-line scratch
// buffers (v/z/d, length n) are unchanged.
template <typename scalar_t, typename offset_t>
inline void
dt(
          offset_t   ndim     ,     // number of dimensions
          scalar_t * f        ,     // pointer to data [*batch, n]
          scalar_t   w        ,     // pixel spacing
    const offset_t * size     ,     // [ndim] data shape   == (*batch, n)
    const offset_t * stride   )     // [ndim] data strides
{
    const offset_t n = size[ndim - 1];   // transform-axis length (all lines share it)
    w = w * w;

    auto at = tny::as_anyrank(f, size, stride, static_cast<int>(ndim), tny::copy_meta);
    const offset_t nlines = at.template size_front<-1>();
    parallel_for(0, nlines, GRAIN_SIZE, [&](long start, long end)
    {
        offset_t * v = nullptr;
        scalar_t * z = nullptr, * d = nullptr;
        try
        {
            v = new offset_t[n];
            z = new scalar_t[n];
            d = new scalar_t[n];
            for (offset_t i = start; i < end; ++i)
            {
                auto line = at.template peel_front_at<-1>(i);
                kernel<offset_t, scalar_t>(
                    line.data(), v, z, d, w, line.extent(0), line.stride(0));
            }
        }
        catch (const std::exception &exc)
        {
            if (v) delete[] v;
            if (z) delete[] z;
            if (d) delete[] d;
            throw exc;
        }
        delete[] v;
        delete[] z;
        delete[] d;
    });
}

FF_NAMESPACE_END(distance_e)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_CPU_DISTANCE_EUCLIDEAN
