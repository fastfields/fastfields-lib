#include <stdexcept>
#include "reg_field.h"
#include "checks.h"
#include "cpu/reg_field.h"
#ifdef FF_WITH_CUDA
#include "cuda/reg_field.h"
#endif

#define IS_CUDA(tensor) (tensor.device.device_type == DLDeviceType::kDLCUDA)
#define IS_CPU(tensor)  (tensor.device.device_type == DLDeviceType::kDLCPU || \
                         tensor.device.device_type == DLDeviceType::kDLCUDAHost)

FF_NAMESPACE_BEGIN(FF)

void field_matvec(
          DLTensor & out       ,
    const DLTensor & inp       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    require_same_device(out, inp);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::field_matvec(out, inp, voxel_size, absolute, membrane, bending, bound, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::field_matvec(out, inp, voxel_size, absolute, membrane, bending, bound, ndim, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void field_diag(
          DLTensor & out       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::field_diag(out, voxel_size, absolute, membrane, bending, bound, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::field_diag(out, voxel_size, absolute, membrane, bending, bound, ndim, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void field_kernel(
          DLTensor & out       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::field_kernel(out, voxel_size, absolute, membrane, bending, bound, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::field_kernel(out, voxel_size, absolute, membrane, bending, bound, ndim, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void field_relax(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        nb_iter   ,
          int        stream    )
{
#ifdef FF_WITH_CUDA
    if (IS_CUDA(sol))
        return FF_CUDA::field_relax(sol, hes, grd, voxel_size, absolute, membrane, bending, bound, ndim, nb_iter, stream);
#endif
    if (IS_CPU(sol))
        return FF_CPU::field_relax(sol, hes, grd, voxel_size, absolute, membrane, bending, bound, ndim, nb_iter, stream);

    if (IS_CUDA(sol))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void field_forward(
          DLTensor & out       ,
    const DLTensor & hes       ,
    const DLTensor & inp       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    require_same_device(out, hes);
    require_same_device(out, inp);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::field_forward(out, hes, inp, voxel_size, absolute, membrane, bending, bound, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::field_forward(out, hes, inp, voxel_size, absolute, membrane, bending, bound, ndim, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void field_precond(
          DLTensor & out       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    require_same_device(out, hes);
    require_same_device(out, grd);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::field_precond(out, hes, grd, voxel_size, absolute, membrane, bending, bound, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::field_precond(out, hes, grd, voxel_size, absolute, membrane, bending, bound, ndim, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void field_precond_(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    require_same_device(sol, hes);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(sol))
        return FF_CUDA::field_precond_(sol, hes, voxel_size, absolute, membrane, bending, bound, ndim, stream);
#endif
    if (IS_CPU(sol))
        return FF_CPU::field_precond_(sol, hes, voxel_size, absolute, membrane, bending, bound, ndim, stream);

    if (IS_CUDA(sol))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void field_matvec_rls(
          DLTensor & out       ,
    const DLTensor & inp       ,
    const DLTensor & wgt       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    require_same_device(out, inp);
    require_same_device(out, wgt);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::field_matvec_rls(out, inp, wgt, voxel_size, absolute, membrane, bending, bound, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::field_matvec_rls(out, inp, wgt, voxel_size, absolute, membrane, bending, bound, ndim, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void field_diag_rls(
          DLTensor & out       ,
    const DLTensor & wgt       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    require_same_device(out, wgt);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::field_diag_rls(out, wgt, voxel_size, absolute, membrane, bending, bound, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::field_diag_rls(out, wgt, voxel_size, absolute, membrane, bending, bound, ndim, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void field_relax_rls(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const DLTensor & wgt       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        nb_iter   ,
          int        stream    )
{
    require_same_device(sol, wgt);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(sol))
        return FF_CUDA::field_relax_rls(sol, hes, grd, wgt, voxel_size, absolute, membrane, bending, bound, ndim, nb_iter, stream);
#endif
    if (IS_CPU(sol))
        return FF_CPU::field_relax_rls(sol, hes, grd, wgt, voxel_size, absolute, membrane, bending, bound, ndim, nb_iter, stream);

    if (IS_CUDA(sol))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

FF_NAMESPACE_END(FF)
