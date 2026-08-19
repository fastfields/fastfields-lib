// Compile-only probe for the mesh distance launchers.
//
// History: `distance_mesh::dt` used to throw "not implemented" on every branch,
// so nothing instantiated the `sdt` launcher and nvcc never type-checked it.
// Two separate rounds of compile errors (fastfields-cuda-impl#23, #40)
// therefore sat on `main` unnoticed, and this TU was added to instantiate the
// launchers explicitly.
//
// `dt` now dispatches its signed/tree-accelerated branch into `sdt`, so the
// real build (cuda-lib's `distance.cpp`) does instantiate that path. This TU is
// still the stricter gate, and still earns its place:
//
//   * it covers `dt` for all four (signed x naive) flag combinations and both
//     the with- and without-`nearest_vertex` shapes, whereas the real build
//     only instantiates whatever `distance.cpp` happens to reference;
//   * it covers `sdt_naive_kernel` / `udt_kernel` / `udt_naive_kernel` and
//     `copy_faces`, none of which has a host launcher and none of which any
//     build would otherwise instantiate;
//   * it applies `-Werror=shadow=local`, which the library build does not.
//
// It is compiled, never linked and never run (there is no GPU in CI).
//
// What this buys, concretely. Each of these was verified by reintroducing the
// defect and watching this TU fail to compile (fastfields-lib#5):
//
//   * a launch whose argument list does not match the kernel signature
//     (missing `_treetrace`/`treesize`)     -> "no instance of function
//                                              template ... matches";
//   * a host pointer passed where the kernel wants the device `DeviceNode *`
//     -> same diagnostic, but only since fastfields-cuda-impl#44 gave that
//        parameter a real type; while it was `const void *` no compiler could
//        have told the two apart;
//   * a stack array whose initialiser list is longer than its bound
//     (`offset_t stride_mat[2] = {a, b, c}`) -> "too many initializer values";
//   * a runtime value used as a template argument
//     (`index2offset<nbatch>(...)`)          -> "expression must have a
//                                               constant value";
//   * a local re-declaration that shadows an outer cleanup variable -> caught
//     by `-Werror=shadow=local`, which the workflow passes to the host
//     compiler for this TU. Nothing in the type system catches this one; it
//     leaks rather than mistypes.
//
// What it does NOT buy: anything semantic. A grid sized from the batch *rank*
// instead of the element count (`GET_BLOCKS(nbatch)` vs `GET_BLOCKS(numel)`)
// compiles perfectly and silently under-launches. That class needs review or
// real hardware.
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
          const index_t * faces, const offset_t * size, offset_t nb_faces,
          offset_t nb_vertices, const offset_t * stride_dist,
          const offset_t * stride_nearest, const offset_t * stride_coord,
          const offset_t * stride_vertices, const offset_t * stride_faces)
{
    M::sdt<D, scalar_t, index_t, offset_t>(
        nbatch, dist, nearest_vertex, coord, vertices, faces, size, nb_faces,
        nb_vertices, stride_dist, stride_nearest, stride_coord, stride_vertices,
        stride_faces, 0);
}

// The launcher above is not what real callers reach: cuda-lib's `distance.cpp`
// dispatches into `distance_mesh::dt`, which then selects a launcher. Probing
// only `sdt` left `dt` itself uninstantiated, so a mistake in the forwarding
// call -- a dropped argument, or the (nb_faces, nb_vertices) pair passed in the
// wrong order, both of which are `offset_t` -- would not have been type-checked
// by anything. Instantiate the actual entry point too, for every flag
// combination cuda-lib can select. The three unimplemented branches throw at
// *run* time, which does not stop nvcc from compiling them here.
template <int D, typename scalar_t, typename index_t, typename offset_t>
static void
probe_dt(offset_t nbatch, scalar_t * dist, index_t * nearest_vertex,
         const scalar_t * coord, const scalar_t * vertices,
         const index_t * faces, const offset_t * size, offset_t nb_faces,
         offset_t nb_vertices, const offset_t * stride_dist,
         const offset_t * stride_nearest, const offset_t * stride_coord,
         const offset_t * stride_vertices, const offset_t * stride_faces)
{
    for (int flags = 0; flags < 4; ++flags)
        M::dt<D, scalar_t, index_t, offset_t>(
            nbatch, dist, nearest_vertex, coord, vertices, faces, size,
            nb_faces, nb_vertices, stride_dist, stride_nearest, stride_coord,
            stride_vertices, stride_faces,
            /*_signed=*/(flags & 1) != 0, /*naive=*/(flags & 2) != 0,
            /*stream=*/0);
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

// Same reasoning for the two *naive* kernels (the brute-force references the
// signed/unsigned BVH paths are meant to be validated against) -- neither has a
// host launcher, so nothing instantiated them either.
template <int D, typename scalar_t, typename index_t, typename offset_t>
static const void * probe_naive_kernels(int which)
{
    const void * const addrs[2] = {
        reinterpret_cast<const void *>(
            &M::sdt_naive_kernel<D, scalar_t, index_t, offset_t>),
        reinterpret_cast<const void *>(
            &M::udt_naive_kernel<D, scalar_t, index_t, offset_t>),
    };
    return addrs[which & 1];
}

const void * ff_compile_probe_mesh_naive_kernels(int which)
{
    switch (which >> 1) {
        case 0: return probe_naive_kernels<2, float, int, int>(which);
        case 1: return probe_naive_kernels<3, float, int, int>(which);
        case 2: return probe_naive_kernels<2, double, long, long>(which);
        default: return probe_naive_kernels<3, double, long, long>(which);
    }
}

// `copy_faces` (and the `copy_faces_kernel` it launches) is a third member of
// the family with no caller: `sdt` uses `copyTensorToContiguous` instead. It
// sat on `main` with a hard error -- the output parameter was `const index_t *`
// and the body initialised an `index_t *` from it. Unlike the kernels above it
// is a `__host__` function, so calling it is enough to instantiate both it and
// the kernel it launches. Never called at runtime; the returned device pointer
// is discarded by the (never-invoked) caller.
template <int D, typename index_t, typename offset_t>
static index_t * probe_copy_faces(offset_t nb_faces, const index_t * faces,
                                  const offset_t * stride)
{
    return M::copy_faces<D, index_t, offset_t>(nb_faces, faces, stride, 0);
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

    // The real entry point, same four type combinations.
    probe_dt<2, float, int, int>(nb32, d32, nv32, c32, v32, f32, sz32, nb32,
                                 nb32, st32, st32, st32, st32, st32);
    probe_dt<3, float, int, int>(nb32, d32, nv32, c32, v32, f32, sz32, nb32,
                                 nb32, st32, st32, st32, st32, st32);
    probe_dt<2, double, long, long>(nb64, d64, nv64, c64, v64, f64, sz64, nb64,
                                    nb64, st64, st64, st64, st64, st64);
    probe_dt<3, double, long, long>(nb64, d64, nv64, c64, v64, f64, sz64, nb64,
                                    nb64, st64, st64, st64, st64, st64);

    // `nearest_vertex` is optional. Passing it null exercises the guarded
    // stride-upload path in `sdt` (and the null-check in `sdt_kernel`); it is a
    // runtime branch, so this is coverage of the *shape* of the call, not proof
    // the branch behaves -- but it keeps a null-typed argument in the probe.
    probe_dt<3, float, int, int>(nb32, d32, static_cast<int *>(nullptr), c32,
                                 v32, f32, sz32, nb32, nb32, st32, nullptr,
                                 st32, st32, st32);

    probe_copy_faces<2, int, int>(nb32, f32, st32);
    probe_copy_faces<3, int, int>(nb32, f32, st32);
    probe_copy_faces<2, long, long>(nb64, f64, st64);
    probe_copy_faces<3, long, long>(nb64, f64, st64);
}
