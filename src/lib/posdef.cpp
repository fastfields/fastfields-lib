#include <stdexcept>
#include "posdef.h"
#include "checks.h"
#include "cpu/posdef.h"
#ifdef FF_WITH_CUDA
#include "cuda/posdef.h"
#endif

FF_NAMESPACE_BEGIN(FF)

#define IS_CUDA(tensor) (tensor.device.device_type == DLDeviceType::kDLCUDA)
#define IS_CPU(tensor)  (tensor.device.device_type == DLDeviceType::kDLCPU || \
                         tensor.device.device_type == DLDeviceType::kDLCUDAHost)

void sym_matvec(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
          int        stream         )
{
    require_same_device(out, hessian, inp);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::sym_matvec(out, hessian, inp, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::sym_matvec(out, hessian, inp, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void sym_matvec_backward(
          DLTensor & out            ,
    const DLTensor & grd            ,
    const DLTensor & inp            ,
          int        stream         )
{
    require_same_device(out, grd, inp);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::sym_matvec_backward(out, grd, inp, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::sym_matvec_backward(out, grd, inp, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void sym_addmatvec_(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
          int        stream         )
{
    require_same_device(out, hessian, inp);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::sym_addmatvec_(out, hessian, inp, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::sym_addmatvec_(out, hessian, inp, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void sym_submatvec_(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
          int        stream         )
{
    require_same_device(out, hessian, inp);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::sym_submatvec_(out, hessian, inp, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::sym_submatvec_(out, hessian, inp, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void sym_solve(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
    const DLTensor & weight         ,
          int        stream         )
{
    require_same_device(out, hessian, inp);
    if (weight.data) require_same_device(out, weight);  // weight is optional (null-data placeholder)
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::sym_solve(out, hessian, inp, weight, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::sym_solve(out, hessian, inp, weight, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void sym_solve_(
          DLTensor & inp_out        ,
    const DLTensor & hessian        ,
    const DLTensor & weight         ,
          int        stream         )
{
    require_same_device(inp_out, hessian);
    if (weight.data) require_same_device(inp_out, weight);  // weight is optional (null-data placeholder)
#ifdef FF_WITH_CUDA
    if (IS_CUDA(inp_out))
        return FF_CUDA::sym_solve_(inp_out, hessian, weight, stream);
#endif
    if (IS_CPU(inp_out))
        return FF_CPU::sym_solve_(inp_out, hessian, weight, stream);

    if (IS_CUDA(inp_out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void sym_invert(
          DLTensor & out            ,
    const DLTensor & hessian        ,
          int        stream         )
{
    require_same_device(out, hessian);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::sym_invert(out, hessian, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::sym_invert(out, hessian, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void sym_invert_(
          DLTensor & hessian        ,
          int        stream         )
{
#ifdef FF_WITH_CUDA
    if (IS_CUDA(hessian))
        return FF_CUDA::sym_invert_(hessian, stream);
#endif
    if (IS_CPU(hessian))
        return FF_CPU::sym_invert_(hessian, stream);

    if (IS_CUDA(hessian))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

FF_NAMESPACE_END(FF)
