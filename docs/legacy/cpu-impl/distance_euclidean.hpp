#ifndef FF_DISTANCE_EUCLIDEAN_CPU
#define FF_DISTANCE_EUCLIDEAN_CPU
#include "kernels/cuda_switch.h"
#include "kernels/distance.h"
#include "kernels/batch.h"
#include "kernels/parallel.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_e)

template <typename scalar_t = float, typename offset_t = int64_t>
static void _dt(
          offset_t ndim,            // number of dimensions
          scalar_t f        [],     // pointer to data [*batch, n]
          scalar_t w,               // pixel spacing
    const offset_t size     [],     // [ndim] data shape   == (*batch, n)
    const offset_t stride   []      // [ndim] data strides
)
{
    offset_t nbatch = ndim - 1;
    offset_t n = size[nbatch];
    offset_t s = stride[nbatch];
    w = w*w;

    offset_t numel = prod(size, nbatch);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    auto v = new offset_t[n];
    auto z = new scalar_t[n];
    auto d = new scalar_t[n];
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset = index2offset(i, nbatch, size, stride);
        kernel(f + offset, v, z, d, w, n, s);
    }
    delete[] v;
    delete[] z;
    delete[] d;
    });
}

template <typename scalar_t = float, typename offset_t = int64_t>
struct AutoCast {

    template <
        typename ndim_t   = offset_t,
        typename f_t      = scalar_t,
        typename w_t      = scalar_t,
        typename size_t   = const offset_t
        typename stride_t = const offset_t
    >
    static inline void
    dt(
        ndim_t   ndim,            // number of dimensions
        f_t      f        [],     // pointer to data [*batch, n]
        w_t      w,               // pixel spacing
        size_t   size     [],     // [ndim] data shape   == (*batch, n)
        stride_t stride           // [ndim] data strides
    )
    {
        return _dt(
            static_cast<       offset_t   > (ndim),
            static_cast<       scalar_t * > (f),
            static_cast<       scalar_t   > (w),
            static_cast< const offset_t * > (size),
            static_cast< const offset_t * > (stride)
        );
    }
};

FF_NAMESPACE_END(distance_e)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_DISTANCE_EUCLIDEAN_CPU
