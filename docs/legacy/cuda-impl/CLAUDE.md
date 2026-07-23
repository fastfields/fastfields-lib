# fastfields-cuda-impl

Header-only **CUDA implementation** for the fastfields project: `__global__`
device kernels plus host launchers that iterate over array elements, calling the
single-element `fastfields-kernels`. Templated, dynamic sizes. The CUDA "impl"
layer (mirror of `fastfields-cpu-impl`).

```
kernels ─ cpu-impl  ─ cpu-lib ─────────────────┐
        └ cuda-impl ← (you are here) ─ cuda-lib ┴─ lib ─ ...
```

- Submodule `kernels -> fastfields-kernels` (symlink in this dev tree).
- Consumed as the submodule `impl` by `fastfields-cuda-lib`.

## Philosophy / role
- Owns the GPU loops over elements. Each module provides device kernels
  (`CUGLOB __global__`) and should provide **host launchers** (`CUHOST`) that
  allocate/copy shape+stride to the device, launch the kernel over the elements,
  and forward the CUDA `stream` — the cuda-lib layer calls those launchers.
- Header-only, templated, dynamic sizes. Namespace `ff::cuda::<module>` (via
  `FF_DEVICE = cuda`).

## Layout
One header per module: `distance_euclidean.h`, `distance_l1.h`,
`distance_mesh.h`, `distance_spline.h`, `posdef.h`, `pushpull.h`,
`reg_field.h`, `reg_flow.h`, `resize.h`, `restrict.h`, `splinc.h`, plus
`utils.h`. (No `tetrahedron.h` yet.)

## Build & test
- Built only with **nvcc**, and only transitively through `fastfields-cuda-lib`.
  There is no GPU in CI, so this layer is validated by **compile + link**, not
  by running — runtime correctness needs real hardware.
- CUDA build (needs nvcc): `make -C ../fastfields-cuda-lib CXX=clang++`.

## Conventions & caveats (IN PROGRESS — read before editing)
- **Host launchers are incomplete.** Only distance (euclidean / l1 / mesh) has
  the `CUHOST` launchers. `posdef`, `resize`, `restrict`, `splinc`, `pushpull`,
  `reg_field`, `reg_flow` currently provide **device kernels only**, so the
  corresponding cuda-lib sources cannot compile until launchers are added —
  those modules are intentionally omitted from the cuda-lib Makefile `MODULES`.
- Known unfixed items (need a CUDA toolchain to verify): wrong `"lib/…"`
  includes (should be `"kernels/…"`); `resize.h`'s `kernelnd` passes an
  undefined `x` (should be `loc`); `distance_euclidean.h` `dt()` scratch buffer
  looks short by a factor of `stride_buf` (possible OOB device write). See the
  "NOT yet fixed" section of MIGRATION.md.
- **C++11**; namespaces via the `FF_NAMESPACE_BEGIN` macros (→ `ff::cuda::…`).
- Keep it a structural mirror of `fastfields-cpu-impl` — port by analogy.

## Pointers
- Hierarchy: `/home/user/.github/profile/README.md`.
- Full status matrix, porting pattern, and the CUDA-specific TODO list:
  `/home/user/fastfields-lib/MIGRATION.md`.
