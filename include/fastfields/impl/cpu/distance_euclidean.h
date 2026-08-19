#ifndef FF_CPU_DISTANCE_EUCLIDEAN
#define FF_CPU_DISTANCE_EUCLIDEAN
#include <cstdint>
#include <type_traits>
#include <teeny/teeny.h>
#include "kernels/cuda_switch.h"
#include "kernels/distance.h"
#include "kernels/parallel.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_e)

// Squared-Euclidean distance sweep (lower-envelope-of-parabolas) along the LAST
// axis, batched over every leading axis.
//
// Same tensor-native boundary as distance_l1::dt -- see the note there for the
// template shape and for why the geometry is derived rather than passed. The
// (*batch, n) carrier IS the argument, built once by the caller; as in
// distance_l1, teeny's `peel` replaces the index2offset batch plumbing (each
// rank-1 line already carries its batch offset in the pointer). The sweep
// kernel and its per-line scratch buffers (v/z/d, length n) are unchanged.
template <class A, typename scalar_t>
inline void
dt(
    A          at ,   // (*batch, n) carrier -- transformed in place
    scalar_t   w  )   // pixel spacing
{
    using offset_t = decltype(at.size(0));   // the carrier's own offset type
    static_assert(
        std::is_same<scalar_t,
                     typename std::remove_pointer<decltype(A::data)>::type>::value,
        "distance_e::dt: spacing type must match the carrier's element type");

    // Transform-axis length (all lines share it). Read by explicit index, NOT
    // `at.shape(-1)`: the carrier's shape store is a fixed TNY_MAX_RANK vector
    // of which only the leading `ndim` slots are live, so a negative wrap would
    // land on the last SLOT rather than the last AXIS.
    const offset_t n = at.size(at.ndim - 1);
    w = w * w;

    const offset_t nlines = at.template size_front<-1>();
    parallel_for(0, nlines, GRAIN_SIZE, [&](int64_t start, int64_t end)
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
