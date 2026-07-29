# fastfields-cpu-lib

Compiles to **`libfastfields-cpu.so`**. Exports functions that take **unsafe
pointers** (no templates in the ABI) and dispatch dtype (and dim/spline/bound)
to the templated `fastfields-cpu-impl`. Built with gcc/clang. Namespace
`ff::cpu`.

```
kernels ─ cpu-impl ─ cpu-lib ← (you are here) ┐
        ─ cuda-impl ─ cuda-lib ───────────────┴─ lib ─ ...
```

- Submodule `impl -> fastfields-cpu-impl` (symlink in this dev tree), which in
  turn symlinks `kernels -> fastfields-kernels`.
- Consumed as the submodule `cpu` by `fastfields-lib`.

## Philosophy / role
- The dtype-dispatch boundary: public symbols take `void*`/pointers + shapes,
  pick `scalar_t`/`offset_t` (and dim/spline/bound), and call the templated
  impl. No templates leak into the exported ABI.
- Pointer/stride arrays are narrowed to 32-bit when `canUse32BitIndexMath`
  allows (`autocast.h`, `copy_if_needed`/`free_if_needed`).
- This is the **primary automated test gate** for the whole project (no GPU in
  CI), so each module ships a CPU correctness test vs. a brute-force reference.

## Layout
- `<module>.{h,cpp}` per module: `distance`, `posdef`, `resize`, `restrict`,
  `splinc`, `pushpull`, `reg_field`, `reg_flow` (the current `MODULES` list).
- `dlpack.h` — vendored DLPack header. `autocast.h` — 32-bit index narrowing.
- `tests/test_<module>.cpp` — one per module, each validated against a
  reference implementation. `Makefile`, `build/`.

## Build & test
```
make -C . test CXX=clang++     # builds libfastfields-cpu.so + compiles & runs tests/test_*.cpp
make -C . all  CXX=clang++     # library only
```
Each `make test` binary compiles a `tests/test_<name>.cpp` together with the
module sources and runs it (exit non-zero on failure).

## Conventions & caveats
- **`BOUNDFLAGS`** (empty by default = all eight boundary conditions statically
  instantiated, i.e. behaviour unchanged) exposes the static/dynamic bound
  policy the CUDA build needs; see fastfields-lib#43.
- **Public op renames** (a namespace can't share a name with a function inside
  `ff::cpu`): `resize -> resample`, `restrict -> restriction`,
  `splinc -> spline_coeff`. `restriction` **accumulates into `out`**, so callers
  pre-zero it.
- Makefiles use **clang-style flags** (`-ferror-limit`, `-ftemplate-backtrace-
  limit`); the object rule needs **`-fPIC`** (a past link bug). Add a module by
  appending it to `MODULES`; ports done in parallel can compile
  `tests/test_<m>.cpp <m>.cpp` directly and integrate `MODULES` in one commit.
- **C++11**. Namespace `ff::cpu`.

## Pointers
- Status matrix, porting pattern (use `distance.{h,cpp}` as the template), and
  the full CPU bug list: `/home/user/fastfields-lib/MIGRATION.md`.
- Hierarchy: `/home/user/.github/profile/README.md`.
