#include <stdexcept>
#include <cstdint>
#include "fastfields/api/posdef.h"
#include "fastfields/api/checks.h"
#include "fastfields/api/cpu/posdef.h"
#ifdef FF_WITH_CUDA
#include "fastfields/api/cuda/posdef.h"
#endif

FF_NAMESPACE_BEGIN(FF_NS)

void sym_matvec(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
          intptr_t   stream         )
{
    require_same_device(out, hessian, inp);
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::sym_matvec(out, hessian, inp, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::sym_matvec(out, hessian, inp, stream);

    if (is_cuda(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void sym_matvec_backward(
          DLTensor & out            ,
    const DLTensor & grd            ,
    const DLTensor & inp            ,
          intptr_t   stream         )
{
    require_same_device(out, grd, inp);
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::sym_matvec_backward(out, grd, inp, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::sym_matvec_backward(out, grd, inp, stream);

    if (is_cuda(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void sym_addmatvec_(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
          intptr_t   stream         )
{
    require_same_device(out, hessian, inp);
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::sym_addmatvec_(out, hessian, inp, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::sym_addmatvec_(out, hessian, inp, stream);

    if (is_cuda(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void sym_submatvec_(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
          intptr_t   stream         )
{
    require_same_device(out, hessian, inp);
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::sym_submatvec_(out, hessian, inp, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::sym_submatvec_(out, hessian, inp, stream);

    if (is_cuda(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void sym_solve(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
    const DLTensor & weight         ,
          intptr_t   stream         )
{
    require_same_device(out, hessian, inp);
    if (weight.data) require_same_device(out, weight);  // weight is optional (null-data placeholder)
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::sym_solve(out, hessian, inp, weight, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::sym_solve(out, hessian, inp, weight, stream);

    if (is_cuda(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void sym_solve_(
          DLTensor & inp_out        ,
    const DLTensor & hessian        ,
    const DLTensor & weight         ,
          intptr_t   stream         )
{
    require_same_device(inp_out, hessian);
    if (weight.data) require_same_device(inp_out, weight);  // weight is optional (null-data placeholder)
#ifdef FF_WITH_CUDA
    if (is_cuda(inp_out))
        return FF_CUDA::sym_solve_(inp_out, hessian, weight, stream);
#endif
    if (is_cpu(inp_out))
        return FF_CPU::sym_solve_(inp_out, hessian, weight, stream);

    if (is_cuda(inp_out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void sym_invert(
          DLTensor & out            ,
    const DLTensor & hessian        ,
          intptr_t   stream         )
{
    require_same_device(out, hessian);
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::sym_invert(out, hessian, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::sym_invert(out, hessian, stream);

    if (is_cuda(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void sym_invert_(
          DLTensor & hessian        ,
          intptr_t   stream         )
{
#ifdef FF_WITH_CUDA
    if (is_cuda(hessian))
        return FF_CUDA::sym_invert_(hessian, stream);
#endif
    if (is_cpu(hessian))
        return FF_CPU::sym_invert_(hessian, stream);

    if (is_cuda(hessian))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

FF_NAMESPACE_END(FF_NS)
