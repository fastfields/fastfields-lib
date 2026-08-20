#include <stdexcept>
#include <cstdint>
#include <fastfields/api/splinc.h>
#include <fastfields/api/checks.h>
#include <fastfields/api/cpu/splinc.h>
#ifdef FF_WITH_CUDA
#include <fastfields/api/cuda/splinc.h>
#endif

FF_NAMESPACE_BEGIN(FF_NS)

void spline_coeff(
          DLTensor & inp_out ,
          int8_t     spline  ,
          int8_t     bound   ,
          intptr_t   stream  )
{
    // Reject boundary conditions the prefilter does not implement. Must run
    // before dispatch: neither backend validates `bound`, and both alias the
    // unimplemented ones onto the DCT1 recursion (fastfields-lib#65).
    require_splinc_bound(spline, bound);

#ifdef FF_WITH_CUDA
    if (is_cuda(inp_out))
        return FF_CUDA::spline_coeff(inp_out, spline, bound, stream);
#endif
    if (is_cpu(inp_out))
        return FF_CPU::spline_coeff(inp_out, spline, bound, stream);

    if (is_cuda(inp_out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

FF_NAMESPACE_END(FF_NS)
