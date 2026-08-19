#ifndef FF_SPLINC_LOOP
#define FF_SPLINC_LOOP
#include "kernels/cuda_switch.h"
#include "kernels/splinc.h"
#include "kernels/batch.h"
#include "kernels/utils.h"
#include "kernels/parallel.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(splinc)

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
    offset_t ndim = nbatch + 1;
    reduce_t poles  [npoles];  fillfrom<npoles>(poles, _poles);

    offset_t numel = prod(size, nbatch);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset = index2offset(i, nbatch, size, stride);
        splinc::filter<B,npoles>(
            inp + offset, size[nbatch], stride[nbatch], poles);
    }});
}

FF_NAMESPACE_END(splinc)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_SPLINC_LOOP
