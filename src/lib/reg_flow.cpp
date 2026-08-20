#include <stdexcept>
#include <cstdint>
#include <fastfields/api/reg_flow.h>
#include <fastfields/api/checks.h>
#include <fastfields/api/cpu/reg_flow.h>
#ifdef FF_WITH_CUDA
#include <fastfields/api/cuda/reg_flow.h>
#endif

FF_NAMESPACE_BEGIN(FF_NS)

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
          intptr_t   stream    )
{
    require_same_device(out, inp);
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::flow_matvec(out, inp, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::flow_matvec(out, inp, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (is_cuda(out))
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
          intptr_t   stream    )
{
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::flow_diag(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::flow_diag(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (is_cuda(out))
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
          intptr_t   stream    )
{
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::flow_kernel(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::flow_kernel(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (is_cuda(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

// `flow_matvec` variant that accumulates into `out`: `out += L(inp)`.
// In-place only (jitfields `op='+'`); an out-of-place accumulate is a
// caller-side clone followed by this same call.
void flow_addmatvec_(
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
          intptr_t   stream    )
{
    require_same_device(out, inp);
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::flow_addmatvec_(out, inp, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::flow_addmatvec_(out, inp, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (is_cuda(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

// `flow_matvec` variant that accumulates into `out`: `out -= L(inp)`.
// In-place only (jitfields `op='-'`); an out-of-place accumulate is a
// caller-side clone followed by this same call.
void flow_submatvec_(
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
          intptr_t   stream    )
{
    require_same_device(out, inp);
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::flow_submatvec_(out, inp, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::flow_submatvec_(out, inp, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (is_cuda(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

// `flow_diag` variant that accumulates into `out`: `out += diag(L)`.
// In-place only (jitfields `op='+'`); an out-of-place accumulate is a
// caller-side clone followed by this same call.
void flow_adddiag_(
          DLTensor & out       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream    )
{
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::flow_adddiag_(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::flow_adddiag_(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (is_cuda(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

// `flow_diag` variant that accumulates into `out`: `out -= diag(L)`.
// In-place only (jitfields `op='-'`); an out-of-place accumulate is a
// caller-side clone followed by this same call.
void flow_subdiag_(
          DLTensor & out       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream    )
{
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::flow_subdiag_(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::flow_subdiag_(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (is_cuda(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

// `flow_kernel` variant that accumulates into `out`: `out += K (the stencil)`.
// In-place only (jitfields `op='+'`); an out-of-place accumulate is a
// caller-side clone followed by this same call.
void flow_addkernel_(
          DLTensor & out       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream    )
{
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::flow_addkernel_(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::flow_addkernel_(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (is_cuda(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

// `flow_kernel` variant that accumulates into `out`: `out -= K (the stencil)`.
// In-place only (jitfields `op='-'`); an out-of-place accumulate is a
// caller-side clone followed by this same call.
void flow_subkernel_(
          DLTensor & out       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream    )
{
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::flow_subkernel_(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::flow_subkernel_(out, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (is_cuda(out))
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
          intptr_t   stream    )
{
#ifdef FF_WITH_CUDA
    if (is_cuda(sol))
        return FF_CUDA::flow_relax(sol, hes, grd, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, nb_iter, stream);
#endif
    if (is_cpu(sol))
        return FF_CPU::flow_relax(sol, hes, grd, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, nb_iter, stream);

    throw std::invalid_argument("unsupported device");
}

void flow_forward(
          DLTensor & out       ,
    const DLTensor & hes       ,
    const DLTensor & inp       ,
    const double   * voxel_size,
          double     absolute  ,
          double     membrane  ,
          double     bending   ,
          double     shears    ,
          double     div       ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream    )
{
    require_same_device(out, hes);
    require_same_device(out, inp);
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::flow_forward(out, hes, inp, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::flow_forward(out, hes, inp, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (is_cuda(out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
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
          intptr_t   stream    )
{
    require_same_device(out, hes);
    require_same_device(out, grd);
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::flow_precond(out, hes, grd, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::flow_precond(out, hes, grd, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (is_cuda(out))
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
          intptr_t   stream    )
{
    require_same_device(sol, hes);
#ifdef FF_WITH_CUDA
    if (is_cuda(sol))
        return FF_CUDA::flow_precond_(sol, hes, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (is_cpu(sol))
        return FF_CPU::flow_precond_(sol, hes, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (is_cuda(sol))
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
          intptr_t   stream    )
{
    require_same_device(out, inp);
    require_same_device(out, wgt);
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::flow_matvec_rls(out, inp, wgt, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::flow_matvec_rls(out, inp, wgt, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (is_cuda(out))
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
          intptr_t   stream    )
{
    require_same_device(out, wgt);
#ifdef FF_WITH_CUDA
    if (is_cuda(out))
        return FF_CUDA::flow_diag_rls(out, wgt, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);
#endif
    if (is_cpu(out))
        return FF_CPU::flow_diag_rls(out, wgt, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, stream);

    if (is_cuda(out))
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
          intptr_t   stream    )
{
    require_same_device(sol, wgt);
#ifdef FF_WITH_CUDA
    if (is_cuda(sol))
        return FF_CUDA::flow_relax_rls(sol, hes, grd, wgt, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, nb_iter, stream);
#endif
    if (is_cpu(sol))
        return FF_CPU::flow_relax_rls(sol, hes, grd, wgt, voxel_size, absolute, membrane, bending, shears, div, bound, ndim, nb_iter, stream);

    if (is_cuda(sol))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

FF_NAMESPACE_END(FF_NS)
