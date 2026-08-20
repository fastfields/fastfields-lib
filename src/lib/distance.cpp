#include <stdexcept>
#include <cstdint>
#include <fastfields/api/distance.h>
#include <fastfields/api/checks.h>
#include <fastfields/api/cpu/distance.h>
#ifdef FF_WITH_CUDA
#include <fastfields/api/cuda/distance.h>
#endif

FF_NAMESPACE_BEGIN(FF_NS)

void dt_euclidean(
          DLTensor & inp_out        ,
          double     voxel_spacing  ,
          intptr_t   stream         )
{
#ifdef FF_WITH_CUDA
    if (is_cuda(inp_out))
        return FF_CUDA::dt_euclidean(inp_out, voxel_spacing, stream);
#endif
    if (is_cpu(inp_out))
        return FF_CPU::dt_euclidean(inp_out, voxel_spacing, stream);

    if (is_cuda(inp_out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void dt_l1(
          DLTensor & inp_out        ,
          double     voxel_spacing  ,
          intptr_t   stream         )
{
#ifdef FF_WITH_CUDA
    if (is_cuda(inp_out))
        return FF_CUDA::dt_l1(inp_out, voxel_spacing, stream);
#endif
    if (is_cpu(inp_out))
        return FF_CPU::dt_l1(inp_out, voxel_spacing, stream);

    if (is_cuda(inp_out))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void dt_spline_table(
          DLTensor & time   ,
          DLTensor & dist   ,
    const DLTensor & loc    ,
    const DLTensor & coeff  ,
    const DLTensor & times  ,
          int8_t     spline ,
          int8_t     bound  ,
          intptr_t   stream )
{
    require_same_device(loc, time, dist, coeff, times);
#ifdef FF_WITH_CUDA
    if (is_cuda(loc))
        return FF_CUDA::dt_spline_table(time, dist, loc, coeff, times, spline, bound, stream);
#endif
    if (is_cpu(loc))
        return FF_CPU::dt_spline_table(time, dist, loc, coeff, times, spline, bound, stream);

    if (is_cuda(loc))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void dt_spline_brent(
          DLTensor & time       ,
          DLTensor & dist       ,
    const DLTensor & loc        ,
    const DLTensor & coeff      ,
          int64_t    max_iter   ,
          double     tol        ,
          double     step       ,
          int8_t     spline     ,
          int8_t     bound      ,
          intptr_t   stream     )
{
    require_same_device(loc, time, dist, coeff);
#ifdef FF_WITH_CUDA
    if (is_cuda(loc))
        return FF_CUDA::dt_spline_brent(time, dist, loc, coeff, max_iter, tol, step, spline, bound, stream);
#endif
    if (is_cpu(loc))
        return FF_CPU::dt_spline_brent(time, dist, loc, coeff, max_iter, tol, step, spline, bound, stream);

    if (is_cuda(loc))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void dt_spline_gaussnewton(
          DLTensor & time       ,
          DLTensor & dist       ,
    const DLTensor & loc        ,
    const DLTensor & coeff      ,
          int64_t    max_iter   ,
          double     tol        ,
          int8_t     spline     ,
          int8_t     bound      ,
          intptr_t   stream    )
{
    require_same_device(loc, time, dist, coeff);
#ifdef FF_WITH_CUDA
    if (is_cuda(loc))
        return FF_CUDA::dt_spline_gaussnewton(time, dist, loc, coeff, max_iter, tol, spline, bound, stream);
#endif
    if (is_cpu(loc))
        return FF_CPU::dt_spline_gaussnewton(time, dist, loc, coeff, max_iter, tol, spline, bound, stream);

    if (is_cuda(loc))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

void dt_mesh(
          DLTensor & dist           ,
          DLTensor & nearest_vertex ,
    const DLTensor & loc            ,
    const DLTensor & vertices       ,
    const DLTensor & faces          ,
          bool       _signed        ,
          bool       naive          ,
          intptr_t   stream          )
{
    require_same_device(loc, dist, vertices, faces);
    // `nearest_vertex` is an optional output: callers signal "not wanted"
    // with a null-data placeholder DLTensor, whose device fields are zeroed
    // and would therefore never match the real operands. Only check it when
    // it carries data (same convention as posdef's optional `weight`).
    if (nearest_vertex.data) require_same_device(loc, nearest_vertex);
#ifdef FF_WITH_CUDA
    if (is_cuda(loc))
        return FF_CUDA::dt_mesh(dist, nearest_vertex, loc, vertices, faces, _signed, naive, stream);
#endif
    if (is_cpu(loc))
        return FF_CPU::dt_mesh(dist, nearest_vertex, loc, vertices, faces, _signed, naive, stream);

    if (is_cuda(loc))
        throw std::invalid_argument("fastfields: built without CUDA support, cannot operate on CUDA tensors");
    throw std::invalid_argument("unsupported device");
}

FF_NAMESPACE_END(FF_NS)
