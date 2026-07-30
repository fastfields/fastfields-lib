#include <stdexcept>
#include "pushpull.h"
#include "checks.h"
#include "cpu/pushpull.h"
#ifdef FF_WITH_CUDA
#include "cuda/pushpull.h"
#endif

#define IS_CUDA(tensor) (tensor.device.device_type == DLDeviceType::kDLCUDA)
#define IS_CPU(tensor)  (tensor.device.device_type == DLDeviceType::kDLCPU || \
                         tensor.device.device_type == DLDeviceType::kDLCUDAHost)

FF_NAMESPACE_BEGIN(FF)

void pull(
          DLTensor & out,
    const DLTensor & inp,
    const DLTensor & grid,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          int        stream )
{
    require_same_device(out, inp, grid);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::pull(out, inp, grid, spline, bound, extrapolate, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::pull(out, inp, grid, spline, bound, extrapolate, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void push(
          DLTensor & out,
    const DLTensor & inp,
    const DLTensor & grid,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          int        stream )
{
    require_same_device(out, inp, grid);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::push(out, inp, grid, spline, bound, extrapolate, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::push(out, inp, grid, spline, bound, extrapolate, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void count(
          DLTensor & out,
    const DLTensor & grid,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          int        stream )
{
    require_same_device(out, grid);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::count(out, grid, spline, bound, extrapolate, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::count(out, grid, spline, bound, extrapolate, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void grad(
          DLTensor & out,
    const DLTensor & inp,
    const DLTensor & grid,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          bool       abs,
          int        stream )
{
    require_same_device(out, inp, grid);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::grad(out, inp, grid, spline, bound, extrapolate, abs, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::grad(out, inp, grid, spline, bound, extrapolate, abs, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

FF_NAMESPACE_END(FF)
