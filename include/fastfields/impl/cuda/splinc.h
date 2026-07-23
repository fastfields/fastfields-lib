#include "kernels/cuda_switch.h"
#include "kernels/splinc.h"
#include "kernels/bounds.h"
#include "kernels/batch.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(splinc)

template <int nbatch, int npoles, bound::type B,
          typename scalar_t, typename offset_t, typename reduce_t>
CUGLOB
void kernel(
    scalar_t * inp,
    const offset_t * _size,
    const offset_t * _stride,
    const reduce_t * _poles)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;

    constexpr int ndim = nbatch + 1;
    reduce_t poles  [npoles];  fillfrom<npoles>(poles, _poles);
    offset_t size   [ndim];    fillfrom<ndim>(size,    _size);
    offset_t stride [ndim];    fillfrom<ndim>(stride,  _stride);

    offset_t numel = prod<nbatch>(size);
    for (offset_t i=index; index < numel;
         index += blockDim.x * gridDim.x, i=index)
    {
        offset_t offset = index2offset<nbatch>(i, size, stride);
        splinc::filter<B,npoles>(
            inp + offset, size[nbatch], stride[nbatch], poles);
    }
}

FF_NAMESPACE_END(splinc)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
