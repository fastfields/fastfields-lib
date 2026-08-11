// Compile-only probe for the mesh distance launchers.
//
// `distance_mesh::sdt` is not reachable from the real build:
// fastfields-cuda-lib's `distance.cpp` still has `distance_mesh::dt` throw "not
// implemented", so nothing instantiates the launcher and nvcc never type-checks
// it. Two separate rounds of compile errors (fastfields-cuda-impl#23, #40)
// therefore sat on `main` unnoticed.
//
// This TU exists purely so that the mesh launchers are instantiated -- and
// hence type-checked -- by CI. It is compiled, never linked and never run
// (there is no GPU in CI).
//
// IMPORTANT -- instantiate by *calling* from a __host__ function, not with an
// explicit-instantiation definition (`template void ...sdt<...>(...);`). An
// explicit instantiation also instantiates the body in nvcc's *device* pass,
// which reports a spurious
//     error: calling a __device__ function("ff::cuda::prod") from a
//            __host__ function("sdt") is not allowed
// because `prod` is CUDEV. `distance_euclidean.h`'s `dt` reproduces that
// identically yet compiles fine in the real build, so it is an artifact of the
// probe technique, not a defect. See fastfields-cuda-impl#40.

#include "../distance_mesh.h"

namespace M = ff::cuda::distance_mesh;

template <int D, typename scalar_t, typename index_t, typename offset_t>
static void
probe_sdt(offset_t nbatch, scalar_t * dist, index_t * nearest_vertex,
          const scalar_t * coord, const scalar_t * vertices,
          const index_t * faces, const offset_t * size, offset_t nb_vertices,
          offset_t nb_faces, const offset_t * stride_dist,
          const offset_t * stride_nearest, const offset_t * stride_coord,
          const offset_t * stride_vertices, const offset_t * stride_faces)
{
    M::sdt<D, scalar_t, index_t, offset_t>(
        nbatch, dist, nearest_vertex, coord, vertices, faces, size, nb_vertices,
        nb_faces, stride_dist, stride_nearest, stride_coord, stride_vertices,
        stride_faces, 0);
}

// `sdt` above only reaches `sdt_kernel`. `udt_kernel` walks the same BVH but
// has no launcher yet, so nothing instantiates it -- and it sat on `main` with
// a hard error (pointer arithmetic on a `void*` trace buffer) that no build
// could see. Taking the address of a `__global__` specialization odr-uses it,
// which instantiates its body and type-checks it. The pointers are discarded;
// nothing here is ever launched.
template <int D, typename scalar_t, typename index_t, typename offset_t>
static const void * probe_udt_kernel()
{
    return reinterpret_cast<const void *>(
        &M::udt_kernel<D, scalar_t, index_t, offset_t>);
}

const void * ff_compile_probe_mesh_udt_kernel(int which)
{
    switch (which)
    {
        case 0:  return probe_udt_kernel<2, float,  int,  int >();
        case 1:  return probe_udt_kernel<3, float,  int,  int >();
        case 2:  return probe_udt_kernel<2, double, long, long>();
        default: return probe_udt_kernel<3, double, long, long>();
    }
}

// One host entry point per (ndim, scalar_t, index_t, offset_t) combination the
// cuda-lib dispatcher can select. Never called.
void ff_compile_probe_mesh_sdt(
    // 2D / float / int
    int nb32, float * d32, int * nv32, const float * c32, const float * v32,
    const int * f32, const int * sz32, const int * st32,
    // 3D / double / long
    long nb64, double * d64, long * nv64, const double * c64,
    const double * v64, const long * f64, const long * sz64, const long * st64)
{
    probe_sdt<2, float, int, int>(nb32, d32, nv32, c32, v32, f32, sz32, nb32,
                                  nb32, st32, st32, st32, st32, st32);
    probe_sdt<3, float, int, int>(nb32, d32, nv32, c32, v32, f32, sz32, nb32,
                                  nb32, st32, st32, st32, st32, st32);
    probe_sdt<2, double, long, long>(nb64, d64, nv64, c64, v64, f64, sz64, nb64,
                                     nb64, st64, st64, st64, st64, st64);
    probe_sdt<3, double, long, long>(nb64, d64, nv64, c64, v64, f64, sz64, nb64,
                                     nb64, st64, st64, st64, st64, st64);
}
