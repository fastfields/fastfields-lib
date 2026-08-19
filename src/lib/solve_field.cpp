#include <stdexcept>
#include "fastfields/api/solve_field.h"
#include "fastfields/api/checks.h"
#include "fastfields/api/cpu/solve_field.h"

FF_NAMESPACE_BEGIN(FF_NS)

// The CG driver is CPU-only for now: unlike the other modules there is no
// `FF_CUDA::field_cg` to forward to, because the solver's dot products need a
// device-side reduction that the cuda backend does not expose yet (tracked in
// fastfields-lib#34). CUDA tensors therefore fall through to the explicit
// "unsupported" error below rather than being silently run on the host.
void field_cg(
          DLTensor & sol         ,
    const DLTensor & hes         ,
    const DLTensor & grd         ,
    const double   * voxel_size  ,
    const double   * absolute    ,
    const double   * membrane    ,
    const double   * bending     ,
          int8_t     bound       ,
          int        ndim        ,
          int        nb_iter     ,
          double     tol         ,
          int      * nb_iter_out ,
          double   * residual_out,
          intptr_t   stream      )
{
    require_same_device(sol, hes, grd);

    if (is_cpu(sol))
        return FF_CPU::field_cg(sol, hes, grd, voxel_size, absolute, membrane,
                                bending, bound, ndim, nb_iter, tol,
                                nb_iter_out, residual_out, stream);

    if (is_cuda(sol))
        throw std::invalid_argument(
            "fastfields: field_cg is not implemented on CUDA yet");
    throw std::invalid_argument("unsupported device");
}

FF_NAMESPACE_END(FF_NS)
