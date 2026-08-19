#include <stdexcept>
#include "fastfields/api/splinc.h"
#include "fastfields/api/cpu/splinc.h"
#ifdef FF_WITH_CUDA
#include "fastfields/api/cuda/splinc.h"
#endif

#define IS_CUDA(tensor) (tensor.device.device_type == DLDeviceType::kDLCUDA)
#define IS_CPU(tensor)  (tensor.device.device_type == DLDeviceType::kDLCPU || \
                         tensor.device.device_type == DLDeviceType::kDLCUDAHost)

FF_NAMESPACE_BEGIN(FF)

void spline_coeff(
          DLTensor & inp_out ,
          int8_t     spline  ,
          int8_t     bound   ,
          int        stream  )
{
#ifdef FF_WITH_CUDA
    if (IS_CUDA(inp_out))
        return FF_CUDA::spline_coeff(inp_out, spline, bound, stream);
#endif
    if (IS_CPU(inp_out))
        return FF_CPU::spline_coeff(inp_out, spline, bound, stream);

    if (IS_CUDA(inp_out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

FF_NAMESPACE_END(FF)
