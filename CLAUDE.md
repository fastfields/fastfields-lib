# fastfields-lib

The **hub**. Compiles to **`libfastfields.so`**. Public functions take **DLPack
tensors** (`DLTensor`), dispatch on `device.device_type` to the CPU or CUDA
library, and link against both. Namespace `ff::`.

```
kernels ─ cpu-impl ─ cpu-lib ┐
        ─ cuda-impl ─ cuda-lib ┴─ lib ← (you are here) ─ bind-py ─ {numpy,cupy,torch} ─ fastfields
```

- Submodules `cpu -> fastfields-cpu-lib` and `cuda -> fastfields-cuda-lib`
  (symlinks in this dev tree).
- Consumed as the submodule `_fastfields_lib` by `fastfields-bind-py`.

## Philosophy / role
- The device-dispatch boundary. Each public `ff::<fn>(DLTensor&, …)` inspects
  the tensor's device and forwards to `FF_CPU::` (`ff::cpu`) or `FF_CUDA::`
  (`ff::cuda`), the latter guarded by `FF_WITH_CUDA`.
- Part of a rewrite of the JIT-compiled `jitfields`, dropping the cppyy/torchlib
  dependency. Data crosses backends via DLPack; higher layers bind with
  nanobind.

## Exposed operation families (FEATURE level)
Describe capabilities, **not** exact DLTensor argument lists — the interface will
be refactored onto a new tensor library later.
- **Distance** — Euclidean & L1 distance transforms; point-to-1D-spline distance
  (table / Brent / Gauss-Newton methods); point-to-triangular-mesh distance.
- **Posdef** — fields of small positive-definite (compact-symmetric) matrices:
  matrix-vector product (+ backward), in-place add/sub matvec, linear solve,
  inverse.
- **Resampling** — spline resampling (`resample`), its adjoint
  (`restriction`, prolongation-transpose), and spline coefficient prefiltering
  (`spline_coeff`).
- **Pushpull** — spline-interpolation gather (pull), scatter (push), count, and
  spatial gradient — the building blocks of image warping/sampling.
- **Regularisers** — spatial regularization (absolute/membrane/bending energies)
  on multi-channel fields and on vector flows: matrix-vector product and
  diagonal (for preconditioning).

## Layout
- `<module>.{h,cpp}` per op family: `distance`, `posdef`, `resize`, `restrict`,
  `splinc`, `pushpull`, `reg_field`, `reg_flow`.
- `dlpack.h` (vendored), `defines.h` (`FF_CPU`/`FF_CUDA`/`FF_WITH_CUDA`),
  `Makefile`, `MIGRATION.md`, `NOTES.md` (dlpack/stream references).

## Build & test
```
make -C . all CXX=clang++      # builds libfastfields.so; also builds+installs libfastfields-cpu.so
```
No standalone tests at this level — correctness is gated by
`fastfields-cpu-lib`'s test suite. CUDA is compile/link-only (no GPU in CI).
Submodule symlinks must exist (`lib/cpu -> cpu-lib`, `cpu-lib/impl -> cpu-impl`,
`cpu-impl/kernels -> kernels`).

## Conventions & caveats
- **C++11**, clang-style Makefile flags, object rule needs `-fPIC`. Add a module
  to `MODULES` in cpu-lib, cuda-lib **and** lib. `libfastfields.so` links
  `-lfastfields-cpu` with an `$ORIGIN/../lib` rpath.
- Op renames from the impl (`resample`/`restriction`/`spline_coeff`);
  `restriction` accumulates into `out`.
- CUDA path compiles+links (`make USE_CUDA=1` builds the `FF_WITH_CUDA` variant
  against `libfastfields-cuda`). No GPU in CI, so CUDA is **compile+link
  only** — the CPU path is the tested source of truth.

## Pointers
- **`./MIGRATION.md`** — the canonical status matrix, the per-module porting
  pattern (`distance.{h,cpp}` is the template), the list of bugs fixed, and the
  open TODOs. Read it first when porting a module.
- Hierarchy: `/home/user/.github/profile/README.md`.
