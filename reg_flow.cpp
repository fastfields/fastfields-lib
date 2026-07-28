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
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    require_same_device(out, inp);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::flow_matvec(out, inp, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::flow_matvec(out, inp, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

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
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::flow_diag(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::flow_diag(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void flow_kernel(
          DLTensor & out       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::flow_kernel(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::flow_kernel(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void flow_relax(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        nb_iter   ,
          int        stream    )
{
#ifdef FF_WITH_CUDA
    if (IS_CUDA(sol))
        return FF_CUDA::flow_relax(sol, hes, grd, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, nb_iter, stream);
#endif
    if (IS_CPU(sol))
        return FF_CPU::flow_relax(sol, hes, grd, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, nb_iter, stream);

    throw std::invalid_argument("unsupported device");
}

void flow_precond(
          DLTensor & out       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    require_same_device(out, hes);
    require_same_device(out, grd);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::flow_precond(out, hes, grd, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::flow_precond(out, hes, grd, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void flow_precond_(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    require_same_device(sol, hes);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(sol))
        return FF_CUDA::flow_precond_(sol, hes, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (IS_CPU(sol))
        return FF_CPU::flow_precond_(sol, hes, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (IS_CUDA(sol))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void flow_matvec_rls(
          DLTensor & out       ,
    const DLTensor & inp       ,
    const DLTensor & wgt       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    require_same_device(out, inp);
    require_same_device(out, wgt);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::flow_matvec_rls(out, inp, wgt, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::flow_matvec_rls(out, inp, wgt, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void flow_diag_rls(
          DLTensor & out       ,
    const DLTensor & wgt       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        stream    )
{
    require_same_device(out, wgt);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(out))
        return FF_CUDA::flow_diag_rls(out, wgt, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (IS_CPU(out))
        return FF_CPU::flow_diag_rls(out, wgt, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (IS_CUDA(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void flow_relax_rls(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const DLTensor & wgt       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          int        nb_iter   ,
          int        stream    )
{
    require_same_device(sol, wgt);
#ifdef FF_WITH_CUDA
    if (IS_CUDA(sol))
        return FF_CUDA::flow_relax_rls(sol, hes, grd, wgt, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, nb_iter, stream);
#endif
    if (IS_CPU(sol))
        return FF_CPU::flow_relax_rls(sol, hes, grd, wgt, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, nb_iter, stream);

    if (IS_CUDA(sol))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

FF_NAMESPACE_END(FF)
