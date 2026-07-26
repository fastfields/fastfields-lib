# fastfields-kernels

Voxelwise **kernels** for the fastfields project: functions that compute on a
**single element** (one voxel / one point / one small matrix). Header-only,
backend-agnostic (CPU or CUDA), templated. This is the bottom layer; everything
else builds on it.

```
kernels  ← (you are here)
  ├─ cpu-impl  ─ cpu-lib  ┐
  └─ cuda-impl ─ cuda-lib ┴─ lib ─ bind-py ─ {numpy,cupy,torch} ─ fastfields
```

Consumed as the git submodule `kernels` by both `fastfields-cpu-impl` and
`fastfields-cuda-impl`.

## Philosophy / role
- Single-element math only. **No device loops** over arrays live here — the
  impl layers own the parallel loops. A kernel is what runs per element.
- Header-only; all functions `static`/`inline` (hidden from the linker).
- Data and pointer types are templated; sizes may be static or dynamic.
- Backend-agnostic: the same header compiles for CPU or CUDA. `cuda_switch.h`
  defines the `FF_DEVICE` / `CUDEV` / `CUGLOB` / `CUHOST` macros keyed on
  `__CUDACC__`; when not compiling with nvcc, `FF_DEVICE` is `cpu` and the CUDA
  qualifiers expand to nothing. `defines.h` provides the `FF` (= `ff`) namespace
  macros; kernels live in `ff::<FF_DEVICE>::<module>`.

## Layout
- `cuda_switch.h`, `defines.h` — backend macros + namespace macros (the spine).
- `utils.h`, `meta.h`, `batch.h`, `bounds.h`, `atomic.h` — shared helpers
  (indexing, boundary conditions, atomics).
- `parallel.h` / `parallel_impl.h`, `threadpool.h` / `threadpool.inl` — CPU
  thread-pool primitives (used by cpu-impl).
- `vector/` — small vector/pointer abstractions (static & dynamic sizes).
- Modules: `distance/{euclidean,l1,spline,mesh}.h`, `posdef/`, `pushpull/`
  (`1d/2d/3d/nd`), `regularisers/{field,flow}/`, `resize.h`, `restrict.h`,
  `splinc.h`, `spline.h`, `tetrahedron.h`. Each has a top-level umbrella header
  (`distance.h`, `posdef.h`, ...).

## Build & test
Header-only — nothing to build here directly. Kernels are exercised through the
CPU library's tests:
```
make -C ../fastfields-cpu-lib test CXX=clang++
```
(Requires the submodule symlinks `cpu-impl/kernels -> kernels` and
`cpu-lib/impl -> cpu-impl` so the include nesting resolves.)

## Conventions & caveats
- **C++17** (all library Makefiles are `-std=c++17`, nvcc included) — `if
  constexpr`, inline `constexpr` members, fold expressions are all fair game
  (the teeny-based `pushpull/teeny.h` + `gather.h` use them). Note: existing
  namespace-scope state still uses Meyers-singleton accessors (`threadpool.inl`;
  a past bug was globals in a header breaking multi-module links) — inline
  variables are now available and fine for new such state.
- Includes are relative (`"../cuda_switch.h"`, `"../utils.h"`); keep the
  submodule directory name `kernels` intact.
- Namespaces are `FF_NAMESPACE_BEGIN(FF)` / `(FF_DEVICE)` / `(<module>)` — do
  **not** hard-code `ff::cpu`; use the macros so CUDA reuses the same source.
- CPU is the tested path; CUDA correctness needs real hardware (none in CI).

## Pointers
- Hierarchy overview: `/home/user/.github/profile/README.md`.
- Migration status, per-module porting pattern, and the list of kernel bugs
  already fixed here: `/home/user/fastfields-lib/MIGRATION.md`.
