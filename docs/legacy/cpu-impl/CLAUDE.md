# fastfields-cpu-impl

Header-only **CPU implementation** for the fastfields project: parallel loops
over array elements that call the single-element `fastfields-kernels`. Templated,
dynamic sizes. This is the CPU "impl" layer.

```
kernels ─ cpu-impl  ← (you are here) ─ cpu-lib ┐
        └ cuda-impl ─ cuda-lib ────────────────┴─ lib ─ ...
```

- Submodule `kernels -> fastfields-kernels` (a symlink in this dev tree).
- Consumed as the submodule `impl` by `fastfields-cpu-lib`.

## Philosophy / role
- Owns the **loops over elements** (thread pool / OpenMP), calling a kernel per
  element. The kernels do the math; this layer does the iteration + scheduling.
- Header-only; functions are templated on data/pointer types with **dynamic**
  (runtime) sizes.
- Runs on the CPU only. The namespace is `ff::cpu::<module>` (via the
  `FF_DEVICE = cpu` macro from the kernels).

## Layout
One header per module, each including its kernel from `kernels/`:
- `distance_euclidean.h`, `distance_l1.h`, `distance_spline.h`,
  `distance_mesh.h`
- `posdef.h` — fields of small positive-definite (compact-symmetric) matrices.
- `pushpull.h` — spline gather/scatter/count/grad.
- `reg_field.h`, `reg_flow.h` — regularisation operators (absolute/membrane/
  bending) for multi-channel fields and vector flows.
- `resize.h`, `restrict.h`, `splinc.h` — resampling + spline prefilter.
- `tetrahedron.h`.

## Build & test
Header-only — no build of its own. Validated through the CPU library, which
symlinks this repo as `impl`:
```
make -C ../fastfields-cpu-lib test CXX=clang++
```
Requires the submodule symlinks (`cpu-lib/impl -> cpu-impl`,
`cpu-impl/kernels -> kernels`) so include nesting resolves.

## Conventions & caveats
- **C++17** (all Makefiles are `-std=c++17`). Includes use the `kernels/…` prefix (a past bug used `lib/…`);
  keep the `kernels` submodule name.
- Namespace must be `FF_NAMESPACE_BEGIN(FF)/(FF_DEVICE)/(<module>)` (→
  `ff::cpu::…`), matching the kernels — not a bare `ff::<module>`.
- Sizes are dynamic here (kernels may also offer static-size variants).
- CPU is the tested path; this repo has no GPU code.

## Pointers
- Hierarchy: `/home/user/.github/profile/README.md`.
- Status matrix, porting pattern, and cpu-impl bugs already fixed (posdef,
  resize/restrict/splinc includes & namespaces, euclidean scratch buffers):
  `/home/user/fastfields-lib/MIGRATION.md`.
