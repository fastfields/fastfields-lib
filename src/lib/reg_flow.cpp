#include <stdexcept>
#include "reg_flow.h"
#include "checks.h"
#include "cpu/reg_flow.h"
#ifdef FF_WITH_CUDA
#include "cuda/reg_flow.h"
#endif

#define IS_CUDA(tensor) (tensor.device.device_type == DLDeviceType::kDLCUDA)
#define IS_CPU(tensor)  (tensor.device.device_type == DLDeviceType::kDLCPU || \
                         tensor.device.device_type == DLDeviceType::kDLCUDAHost)

FF_NAMESPACE_BEGIN(FF)

void flow_matvec(
          DLTensor & out       ,
    const DLTensor & inp       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    require_same_device(out, inp);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::flow_matvec(out, inp, voxel_size, absolute, membrane, bending, bound, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::flow_matvec(out, inp, voxel_size, absolute, membrane, bending, bound, ndim, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void flow_diag(
          DLTensor & out       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::flow_diag(out, voxel_size, absolute, membrane, bending, bound, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::flow_diag(out, voxel_size, absolute, membrane, bending, bound, ndim, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

FF_NAMESPACE_END(FF)
