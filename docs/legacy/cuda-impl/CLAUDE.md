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

## Conventions & caveats
- **Host launchers exist for every module** (`CUHOST`): distance, posdef, resize,
  restrict, splinc, reg_field, reg_flow, pushpull. All compile under nvcc; all
  but pushpull are in the cuda-lib `MODULES` (pushpull's compile is ~40 min).
  The include paths (`"kernels/…"`), `resize.h` `x`→`loc`, and the
  `distance_euclidean.h` scratch-buffer sizing were fixed during integration /
  the fable review.
- **No GPU in CI** — everything here is validated by compile+link only. Runtime
  correctness (atomics, the CUDA `stream` plumbing, the mesh `sdt` launcher) is
  unvalidated; see the tracked `fastfields-lib` issues.
- **C++11**; namespaces via the `FF_NAMESPACE_BEGIN` macros (→ `ff::cuda::…`).
- Keep it a structural mirror of `fastfields-cpu-impl` — port by analogy.

## Pointers
- Hierarchy: `/home/user/.github/profile/README.md`.
- Full status matrix, porting pattern, and the CUDA-specific TODO list:
  `/home/user/fastfields-lib/MIGRATION.md`.
