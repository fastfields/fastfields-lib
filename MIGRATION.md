# fastfields migration status & plan

This document tracks the port of the jitfields kernels into the layered,
DLPack-based `fastfields` C++/CUDA libraries. See `.github/profile/README.md`
for the repository hierarchy.

## Layer recap

```
kernels (voxelwise, header-only, templated)
  └─ {cpu,cuda}-impl  (loops over elements; CPU thread pool / CUDA kernels; templated, dynamic sizes)
       └─ {cpu,cuda}-lib  (exported symbols; unsafe pointers -> template dispatch on dtype/dim)
            └─ lib  (exported symbols; DLTensor in; dispatch on device -> cpu/cuda lib)
```

Each ported operation appears at three "library" levels:
- `{cpu,cuda}-lib/<module>.{h,cpp}` — `ff::cpu::` / `ff::cuda::` functions taking
  `DLTensor&`, dispatching on dtype (and dim/spline/bound where relevant) to the
  templated impl. Pointer/stride arrays are narrowed to 32-bit when
  `canUse32BitIndexMath` allows (see `autocast.h` / `copy_if_needed`).
- `lib/<module>.{h,cpp}` — public `ff::` functions taking `DLTensor&`, dispatching
  on `device.device_type` to the cpu or cuda lib.

## Status matrix

| module         | kernels | cpu-impl | cuda-impl | cpu-lib | cuda-lib | lib | CPU tested |
|----------------|:------:|:--------:|:---------:|:-------:|:--------:|:---:|:----------:|
| distance       |  ✓     |   ✓      |   ✓       |   ✓     |   ✓      | ✓   | **yes**    |
| posdef         |  ✓     |   ✓      |   ✓       |   ✗     |   ✗      | ✗   | no         |
| pushpull       |  ✓     |   ✓      |   ✓       |   ✗     |   ✗      | ✗   | no         |
| reg_field      |  ✓     |   ✓      |   ✓       |   ✗     |   ✗      | ✗   | no         |
| reg_flow       |  ✓     |   ✓      |   ✓       |   ✗     |   ✗      | ✗   | no         |
| resize         |  ✓     |   ✓      |   ✓       |   ✗     |   ✗      | ✗   | no         |
| restrict       |  ✓     |   ✓      |   ✓       |   ✗     |   ✗      | ✗   | no         |
| splinc         |  ✓     |   ✓      |   ✓       |   ✗     |   ✗      | ✗   | no         |
| tetrahedron    |  ✓     |   ✓      |   —       |   ✗     |   ✗      | ✗   | no         |

"✓" for a lib column means the dispatch layer exists; only `distance` is currently
compiled and tested (CPU). `tetrahedron` has no cuda-impl header yet.

## Build & test (CPU)

No `nvcc` is assumed available in CI; the CPU path is the primary automated gate.
Submodules must be checked out (or symlinked) so the include nesting resolves:
`cpu-lib/impl -> cpu-impl`, `cpu-impl/kernels -> kernels`, `lib/cpu -> cpu-lib`.

```
make -C fastfields-cpu-lib test CXX=clang++   # builds libfastfields-cpu.so + runs tests/test_*.cpp
make -C fastfields-lib      all  CXX=clang++   # builds libfastfields.so (links cpu lib)
```

`make test` compiles each `tests/test_<name>.cpp` together with the module sources
and runs it. New modules should ship a `tests/test_<module>.cpp` that validates the
CPU path against a brute-force / reference implementation, as `test_distance.cpp` does.

## Bugs found & fixed (CPU, verified)

1. **kernels/parallel_impl.h** — missing `<queue>`/`<string>`; would not compile.
2. **kernels/distance/mesh.h** — missing `<algorithm>` (`std::sort`).
3. **cpu-impl/distance_euclidean.h** — `z`/`d` scratch buffers declared `offset_t*`
   but assigned `new scalar_t[n]` and passed where `scalar_t*` is expected; failed
   to compile when `scalar_t != offset_t`.
4. **lib/distance.cpp** — `dt_l1` dispatched to `dt_euclidean` in both branches;
   the L1 transform was never called.
5. **cpu-lib/Makefile, lib/Makefile** — object compile rule missing `-fPIC`, so the
   shared-library link failed (`relocation R_X86_64_PC32 … recompile with -fPIC`).

## Bugs found, NOT yet fixed (need a CUDA build to verify)

- **cuda-impl/distance_euclidean.h** — `dt()` allocates a scratch buffer of
  `vector_size * (sizeof(offset_t) + 2*sizeof(scalar_t))` bytes, but `dt_kernel`
  addresses `z`/`d` at `buf + stride_buf * n * …` and indexes each of the
  `stride_buf` (= total launched threads) lanes with its own length-`n` `v/z/d`
  region. The allocation therefore appears short by a factor of `stride_buf`,
  which would be an out-of-bounds device write. Confirm and fix once a CUDA
  toolchain is available; add a CUDA test mirroring `test_distance.cpp`.
- **cpu-lib/distance.cpp** `_dt_spline_*` — the `copy_if_needed<offset_t*>(…, ndim)`
  calls pass the spatial dim `ndim` (1/2/3) as the array length for `size`/stride
  arrays whose true length is `nbatch (+1/+2)`. In the 32-bit index path this
  under-copies and the kernel may read past the narrowed arrays. Needs a spline
  test (CPU) to confirm before changing; euclidean/l1 are unaffected (their arrays
  really are length `ndim`).

## Porting pattern (per module)

Use `distance.{h,cpp}` at each level as the template.

1. **cpu-lib/`<module>`.h** — declare `ff::cpu::<fn>(DLTensor&, …)` for each op.
2. **cpu-lib/`<module>`.cpp** — for each op:
   - a `CHECK_*` block (shape/dtype/batch), reusing the macros in `distance.cpp`
     (consider hoisting these into a shared `checks.h`);
   - a `DISPATCH_*` macro selecting `scalar_t`/`offset_t` (+ `ndim`, spline, bound);
   - an anonymous-namespace `_<fn><…>` template that narrows pointers via
     `copy_if_needed`, casts `void*`, calls the impl, then `free_if_needed`.
3. **cuda-lib/`<module>`.{h,cpp}** — same as cpu-lib but `ff::cuda::`, forwarding
   `stream`, including `impl/…` from cuda-impl. (Cannot be compiled here.)
4. **lib/`<module>`.{h,cpp}** — public `ff::<fn>(DLTensor&, …)` dispatching on
   `device.device_type` to `FF_CPU::` / `FF_CUDA::` (guard cuda with `FF_WITH_CUDA`).
5. **Makefiles** — add `<module>` to `MODULES` in cpu-lib, cuda-lib and lib.
6. **tests/test_`<module>`.cpp** in cpu-lib — CPU correctness vs. reference.

### Avoiding merge conflicts when porting in parallel
Module `.cpp/.h` and `tests/test_<module>.cpp` are disjoint per module. The only
shared files are the three `MODULES` lists in the Makefiles. When several modules
are ported concurrently, add each module's `<module>.cpp` compiled *directly* with
its test (`clang++ -I. tests/test_<module>.cpp <module>.cpp -o …`) to validate
without editing the Makefile, and record the required `MODULES +=` line in the PR;
integrate the `MODULES` lists in a single follow-up commit.

## Task breakdown

- **T1 (done)** distance: CPU green + tested; fixes above; pushed.
- **T2 (this doc)** migration plan & status matrix.
- **T3** posdef: `sym_matvec[_backward]`, `sym_addmatvec_`, `sym_submatvec_`,
  `sym_solve[_]`, `sym_invert[_]` over `DLTensor`. Matrix layouts (full/sym/diag/
  estatics/eye) are selected by a runtime enum — dispatch on it. CPU test: compare
  `sym_matvec` then `sym_solve` round-trips a known SPD system.
- **T4** resize + restrict + splinc: smaller loops (`loop`/`loopnd`, `loop2`).
  resize/restrict take spline order + bound + a scale/anchor; splinc is the spline
  prefilter. CPU tests: resize by integer factor vs. manual interpolation; splinc
  followed by evaluation reproduces samples.
- **T5** pushpull: `pull/push/count/grad/hess` + `_backward`, dispatched on
  dim × spline × bound × dtype. Large; port `pull`/`push` first with a small
  gather/scatter CPU test, then the rest.
- **T6** regularisers (reg_field, reg_flow): `matvec/kernel/diag/relax` for
  absolute/membrane/bending (+ RLS). Dispatch on dim × dtype; energies parametrised
  by `absolute/membrane/bending` weights + voxel size. CPU test: `matvec` equals the
  finite-difference Laplacian for the membrane term.
- **T7** de-templating audit + Makefile/CI hardening: confirm which impl entry
  points still template runtime sizes that the migration intends to de-template
  (cross-check jitfields); wire `USE_OPENMP` (currently defined but unused) into
  `CXXFLAGS`; add cuda-lib `-Xcompiler -fPIC`; add a SessionStart/CI job that runs
  the CPU build + tests.
