#include <stdexcept>
#include <cstdint>
#include "fastfields/api/pushpull.h"
#include "fastfields/api/checks.h"
#include "fastfields/api/cpu/pushpull.h"
#ifdef FF_WITH_CUDA
#include "fastfields/api/cuda/pushpull.h"
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
          intptr_t   stream )
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
          intptr_t   stream )
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
          intptr_t   stream )
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
          intptr_t   stream )
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

/***********************************************************************
 *                          BACKWARD PASSES                            *
 ***********************************************************************/

void pull_backward(
          DLTensor & out,
          DLTensor & gout,
    const DLTensor & inp,
    const DLTensor & ginp,
    const DLTensor & grid,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          intptr_t   stream )
{
    require_same_device(out, gout, inp, ginp, grid);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::pull_backward(out, gout, inp, ginp, grid, spline, bound, extrapolate, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::pull_backward(out, gout, inp, ginp, grid, spline, bound, extrapolate, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void push_backward(
          DLTensor & out,
          DLTensor & gout,
    const DLTensor & inp,
    const DLTensor & ginp,
    const DLTensor & grid,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          intptr_t   stream )
{
    require_same_device(out, gout, inp, ginp, grid);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::push_backward(out, gout, inp, ginp, grid, spline, bound, extrapolate, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::push_backward(out, gout, inp, ginp, grid, spline, bound, extrapolate, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void count_backward(
          DLTensor & gout,
    const DLTensor & ginp,
    const DLTensor & grid,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          intptr_t   stream )
{
    require_same_device(gout, ginp, grid);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(gout))
        return FF_CUDA::count_backward(gout, ginp, grid, spline, bound, extrapolate, stream);
#endif
    if (IS_CPU(gout))
        return FF_CPU::count_backward(gout, ginp, grid, spline, bound, extrapolate, stream);

    if (IS_CUDA(gout))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void grad_backward(
          DLTensor & out,
          DLTensor & gout,
    const DLTensor & inp,
    const DLTensor & ginp,
    const DLTensor & grid,
          int8_t     spline,
          int8_t     bound,
          int8_t     extrapolate,
          bool       abs,
          intptr_t   stream )
{
    require_same_device(out, gout, inp, ginp, grid);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::grad_backward(out, gout, inp, ginp, grid, spline, bound, extrapolate, abs, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::grad_backward(out, gout, inp, ginp, grid, spline, bound, extrapolate, abs, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

FF_NAMESPACE_END(FF)
