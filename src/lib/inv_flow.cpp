#include <stdexcept>
#include <cstdint>
#include <fastfields/api/inv_flow.h>
#include <fastfields/api/checks.h>
#include <fastfields/api/cpu/inv_flow.h>
#ifdef FF_WITH_CUDA
#include <fastfields/api/cuda/inv_flow.h>
#endif

FF_NAMESPACE_BEGIN(FF_NS)

void invert_flow(
          DLTensor & out    ,
          DLTensor & owner  ,
    const DLTensor & disp   ,
          int        ndim   ,
          intptr_t   stream )
{
    require_same_device(out, owner, disp);
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::invert_flow(out, owner, disp, ndim, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::invert_flow(out, owner, disp, ndim, stream);

    if (is_cuda(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

FF_NAMESPACE_END(FF_NS)
