#ifndef FF_SPLINC_LOOP
#define FF_SPLINC_LOOP
#include <teeny/teeny.h>
#include "kernels/cuda_switch.h"
#include "kernels/splinc.h"
#include "kernels/parallel.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(splinc)

// 1-D spline prefilter (causal/anticausal IIR pole recursion) along the LAST
// axis, batched over every leading axis.
//
// teeny's peel replaces the hand-written index2offset batch plumbing: an
// `anyrank` over (*batch, n) hands out each rank-1 line as a view whose data
// pointer already has the (arbitrarily strided) batch offset folded in, so the
// sweep kernel is called unchanged on the raw pointer + size + stride.
template <
    int npoles,
    bound::type B,
    typename scalar_t,
    typename offset_t,
    typename reduce_t
>
void loop(
          offset_t   nbatch,
          scalar_t * inp,
    const offset_t * size,
    const offset_t * stride,
    const reduce_t * _poles
)
{
    reduce_t poles[npoles];
    for (int k = 0; k < npoles; ++k) poles[k] = _poles[k];

    const int ndim = static_cast<int>(nbatch) + 1;
    auto at = tny::as_anyrank(inp, size, stride, ndim, tny::copy_meta);
    const offset_t nlines = at.template size_front<-1>();

    parallel_for(0, nlines, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = start; i < end; ++i) {
        auto line = at.template peel_front_at<-1>(i);
        splinc::filter<B, npoles>(line.data(), line.extent(0), line.stride(0), poles);
    }});
}

FF_NAMESPACE_END(splinc)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_SPLINC_LOOP
