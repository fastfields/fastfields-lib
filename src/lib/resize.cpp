#include <stdexcept>
#include "fastfields/api/resize.h"
#include "fastfields/api/checks.h"
#include "fastfields/api/cpu/resize.h"
#ifdef FF_WITH_CUDA
#include "fastfields/api/cuda/resize.h"
#endif

#define IS_CUDA(tensor) (tensor.device.device_type == DLDeviceType::kDLCUDA)
#define IS_CPU(tensor)  (tensor.device.device_type == DLDeviceType::kDLCPU || \
                         tensor.device.device_type == DLDeviceType::kDLCUDAHost)

FF_NAMESPACE_BEGIN(FF)

void resample(
          DLTensor & out    ,
    const DLTensor & inp    ,
          int8_t     spline ,
          int8_t     bound  ,
          double     shift  ,
    const double   * scale  ,
          int        ndim   ,
          int        stream )
{
    require_same_device(out, inp);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::resample(out, inp, spline, bound, shift, scale, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::resample(out, inp, spline, bound, shift, scale, ndim, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

FF_NAMESPACE_END(FF)
