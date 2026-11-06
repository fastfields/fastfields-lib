#pragma once
#include <fastfields/core/cuda_switch.h>
#include <fastfields/impl/kernels/splinc.h>
#include <fastfields/impl/kernels/batch.h>
#include <fastfields/impl/kernels/utils.h>
#include <fastfields/impl/kernels/parallel.h>

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(splinc)

// Prefilter every line along the last axis. `B` is the compile-time boundary
// condition, or `bound::type::Dynamic` to use `bnd` instead. The plan (initial
// condition horizons, edge-correction or small dense inverse) only depends on
// the axis length, the poles and the boundary, so it is built once here and
// shared read-only by every line.
template <
    int npoles,
    bound::type B,
    typename scalar_t,
    typename offset_t,
    typename reduce_t
>
void loop(
          offset_t      nbatch,
          scalar_t    * inp,
    const offset_t    * size,
    const offset_t    * stride,
    const reduce_t    * _poles,
          bound::type   bnd = B
)
{
    reduce_t poles  [npoles];  fillfrom<npoles>(poles, _poles);

    const plan<npoles, reduce_t, offset_t> p = make_plan<npoles>(
        B == bound::type::Dynamic ? bnd : B, size[nbatch], poles);
    const offset_t st = stride[nbatch];

    offset_t numel = prod(size, nbatch);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset = index2offset(i, nbatch, size, stride);
        splinc::filter(inp + offset, st, p);
    }});
}

FF_NAMESPACE_END(splinc)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
