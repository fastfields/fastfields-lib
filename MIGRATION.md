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
| posdef         |  ✓     |   ✓      |   ✓       |   ✓     |   ~      | ✓   | **yes**    |
| resize         |  ✓     |   ✓      |   ✓       |   ✓     |   ~      | ✓   | **yes**    |
| restrict       |  ✓     |   ✓      |   ✓       |   ✓     |   ~      | ✓   | **yes**    |
| splinc         |  ✓     |   ✓      |   ✓       |   ✓     |   ~      | ✓   | **yes**    |
| pushpull       |  ✓     |   ✓      |   ✓       |   ✓     |   ~      | ✓   | **yes**    |
| reg_field      |  ✓     |   ✓      |   ✓       |   ✓     |   ~      | ✓   | **yes**    |
| reg_flow       |  ✓     |   ✓      |   ✓       |   ✓     |   ~      | ✓   | **yes**    |
| tetrahedron    |  ✓     |   ✓      |   —       |   ✗     |   ✗      | ✗   | no         |

All eight CPU-buildable modules (distance, posdef, resize, restrict, splinc, pushpull,
reg_field, reg_flow) are ported through cpu-lib → lib, link into one `libfastfields-cpu.so`,
and pass their CPU test suites (distance 2350 / euclidean+l1, posdef 3580, resize 3885,
restrict 191, splinc 4576, distance_spline 510, distance_mesh 1280, pushpull 104,
reg_field 272, reg_flow 282). pushpull exposes pull/push/count/grad (hess + backward
variants remain in the impl, not yet exposed); regularisers expose matvec/diag for
absolute/membrane/bending (kernel/relax/RLS remain in the impl).

**CUDA branch (integrated; compile-gated):** nvcc (Ubuntu CUDA 12.0) compiles the kernels
and cuda-impl under `__CUDACC__`. Host launchers now exist for **every** module
(distance, posdef, resize, restrict, splinc, reg_field, reg_flow, pushpull). `cuda-lib`
links `libfastfields-cuda.so` with `MODULES = distance posdef resize restrict splinc
reg_field reg_flow` (pushpull is written + type-checked but omitted from `MODULES` for now
because its spline×bound×dim×dtype matrix takes ~40 min to compile — see the `whl`/T21 notes).
The hub `fastfields-lib` builds an optional `FF_WITH_CUDA` variant (`make USE_CUDA=1`) that
links both `-lfastfields-cpu` and `-lfastfields-cuda`; pushpull's CUDA path is gated behind
`FF_CUDA_NO_PUSHPULL` so the link resolves without it. **No GPU in CI**, so all of the above is
**compile+link only** — runtime CUDA correctness is unvalidated (see the tracked issues).

"✓" for a cpu-lib/lib column means the dispatch layer exists **and is CPU-compiled+tested**.
"✓(c)" for cuda-lib means it compiles+links under nvcc (no runtime validation).
`tetrahedron` has no cuda-impl header yet.

### Public API naming
`resize`/`restrict`/`splinc` would be an illegal same-name namespace+function inside `ff::cpu`,
so the public ops are named **`resample`** (resize), **`restriction`** (restrict), and
**`spline_coeff`** (splinc). `restriction` accumulates into `out`, so callers pre-zero it.

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
6. **kernels/threadpool.inl** — defined non-`inline` free functions and two
   namespace-scope globals (`internal::num_threads`, `internal::global_pool`) in a
   header. Fine for a single-module library, but a library with ≥2 module objects
   (e.g. distance + posdef) failed to link (`multiple definition of …`). Made the
   helpers `inline` and wrapped the two globals in Meyers-singleton accessors (C++11
   has no inline variables). This unblocks every multi-module library.
7. **kernels/posdef/utils.h** — three alias templates referenced undeclared names
   (`_as_points`, `left`/`right`, and `_return_type` with no args); any posdef include
   failed to parse. **kernels/posdef/sym.inl** — dynamic `Sym::invert` called the
   inherited `copy_` unqualified (two-phase lookup) → `this_type::copy_`.
8. **cpu-impl/posdef.h** — runtime `nbatch` passed as a template arg to `index2offset`;
   a dead static-C `else` branch that misbound the dynamic specialization under C++11
   two-arm instantiation (guarded with `(C<0?1:C)`); `delete`→`delete[]` at 4 sites.
9. **kernels/utils.h** — `prod<size>(x)` called `typed_prod<T,size>(x, size)` with a
   spurious extra arg (no matching overload); triggered by `restrict::loop`.
10. **cpu-impl/{resize,restrict,splinc}.h** — wrong include prefix `"lib/…"` →
    `"kernels/…"`; impl namespace was plain `ff::<module>` but the kernels live in
    `ff::cpu::` (`FF_DEVICE`) so it must be `FF_NAMESPACE_BEGIN(FF)/(FF_DEVICE)/(<module>)`
    like distance; `index2offset_nd<ndim>()` runtime-ndim → dynamic overload;
    `jf::has_atomic_add` → `has_atomic_add`.

## Fable correctness review — fixes landed

A thorough correctness review (vs the jitfields oracle) found and we fixed:

- **Live CPU bugs:** signed mesh distance queried the *unsorted* faces instead of
  the BVH-reordered `faces_copy` (`cpu-impl/distance_mesh.h`); the mesh index-dtype
  dispatch read the width from `loc` not `faces` (`cpu-lib/distance.cpp`);
  2D any-spline `pull` used `size[1]` for the x axis (`kernels/pushpull/2d.h`);
  `resample`/`restriction` silently no-op'd on an unsupported dtype (now throw);
  unsigned mesh distance never wrote `nearest_vertex`. Regression tests added
  (multi-triangle unsorted mesh, mixed float/int widths, 2D quad/cubic pull on a
  non-square grid, bad-dtype-throws).
- **CUDA (compile-verified):** `GET_BLOCKS` guard was inverted (every launch threw);
  the euclidean scratch buffer was short by the launched-thread factor (device OOB);
  the pushpull `hess` kernel called itself; the converting `copyToHost` derefed
  device memory on the host.
- **Latent/inherited:** cubic-1D `hess` aliased the gradient weights; posdef
  `Eye`/`ESTATICS` (bad `iadd` arity / null-`w` deref); `AtomicAdd<true>` was
  nonsense (now a correct C++11 CAS loop).

Earlier flagged items now resolved: the euclidean scratch under-alloc (fixed), the
`cpu-lib/distance.cpp` `_dt_spline_*` `copy_if_needed` lengths (verified correct —
`nbatch`/`nbatch+1`/`nbatch+2`), and the cuda-impl `"lib/…"` includes / `resize.h`
`x`→`loc` (fixed during the CUDA integration).

## Bugs found, tracked as issues (not fixed in the review pass)

Confirmed but unreachable-today or GPU-unverifiable — see `fastfields-lib` issues:
CUDA `stream` width (`int`→`intptr_t`) + forwarding in the distance launchers; the
internally-broken CUDA mesh `sdt` "complete" launcher (behind an honest `throw`); and
the latent C++ follow-ups (reg `<set>` hard-coding, pushpull Dynamic spline/bound
bypass, `strides==NULL` handling, `tetrahedron.h` rasterization math).

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
