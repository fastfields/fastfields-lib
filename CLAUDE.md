# fastfields

The consolidated **C++/CUDA** side of the fastfields project: kernels, both
backend implementations, both dtype-dispatch libraries, and the DLPack hub, in
one repository. Builds **`libfastfields.so`** (the hub),
**`libfastfields-cpu.so`** and, opt-in, **`libfastfields-cuda.so`**.

On **2026-08-19** six repositories — `fastfields-lib`, `fastfields-cpu-lib`,
`fastfields-cuda-lib`, `fastfields-cpu-impl`, `fastfields-cuda-impl`,
`fastfields-kernels` — were merged here and `main` was rewritten. The other
five are archived and read-only.

> **Read [`MIGRATION-PROVENANCE.md`](MIGRATION-PROVENANCE.md) before you read
> `git log`.** Commit subjects predating the merge carry `(#N)` numbers from
> their *original* repo, and GitHub silently renders them as links to
> `fastfields-lib` issues of the same number. 141 commits on `main` are
> affected. A mis-link looks correct, so nobody notices.

---

## Branches — read this before opening a PR

| Branch | What it is |
| --- | --- |
| `main` | the consolidated tree |
| `teeny` | the teeny refactor: 159 commits ahead of `main`, carries `external/teeny` (a submodule `main` does not have) |

**PRs in the `claude-fastfields-to-teeny` workstream must target `teeny`, never
`main`.** Three of that workstream's recent PRs were mis-filed against `main`
and had to be re-targeted; the rule predates the consolidation and must survive
it. Most open issues carrying the `claude-fastfields-to-teeny` label belong to
that workstream.

`pre-consolidation-main` and `pre-consolidation-teeny` preserve the pre-rewrite
tips. Do not delete them.

---

## The layer stack

```
impl/kernels ─ impl/cpu  ─ api/cpu  + src/lib-cpu  ┐
             ─ impl/cuda ─ api/cuda + src/lib-cuda ┴─ api + src/lib  ─ (bind-py ─ …)
```

- **`include/fastfields/impl/kernels/`** — single-element math (one voxel, one
  point, one small matrix). Header-only, backend-agnostic, no device loops.
  Namespace `ff::<FF_DEVICE>::<module>` via the `FF_NAMESPACE_BEGIN` macros —
  do **not** hard-code `ff::cpu`.
- **`include/fastfields/impl/cpu/`**, **`impl/cuda/`** — the loops over
  elements (thread pool / OpenMP; `__global__` kernels + `FF_CUHOST` launchers).
  Header-only, templated, dynamic sizes. `ff::cpu::…` / `ff::cuda::…`.
- **`include/fastfields/api/cpu/`**, **`api/cuda/`** + **`src/lib-cpu/`**,
  **`src/lib-cuda/`** — the dtype-dispatch boundary. Public symbols take
  `void*`/pointers + shapes and pick `scalar_t`/`offset_t` (and dim/spline/
  bound). No templates leak into the exported ABI.
- **`include/fastfields/api/*.h`** + **`src/lib/`** — the hub, the
  device-dispatch boundary. Each public `ff::<fn>(DLTensor&, …)` inspects the
  tensor's device and forwards to `FF_CPU::` or `FF_CUDA::`, the latter guarded
  by `FF_WITH_CUDA`.
- **`include/fastfields/core/`** — `dlpack.h` (vendored, do not edit),
  `autocast.h` (32-bit index narrowing), `defines.h`, `cuda_switch.h`,
  `dispatch.h` (the `FF_VOIDPTR` / `FF_CANUSE32BITS` / `FF_CHECK_*` helpers
  both dtype-dispatch layers share). `core/` is where anything used by
  **both** `src/lib-cpu` (host compiler) and `src/lib-cuda` (nvcc) has to
  live: it is the only directory that is backend-agnostic by contract.

Also: `make/common.mk` (the shared makefile fragment), `tools/consolidate.sh`
(the frozen consolidation rules), `ci/legacy/` and `docs/legacy/` (the six
repos' pre-consolidation workflows and docs, kept for reference — not live).

## Exposed operation families (FEATURE level)
Describe capabilities, **not** exact DLTensor argument lists — the interface
will be refactored onto a new tensor library later.
- **Distance** — Euclidean & L1 distance transforms; point-to-1D-spline
  distance (table / Brent / Gauss-Newton); point-to-triangular-mesh distance.
- **Posdef** — fields of small positive-definite (compact-symmetric) matrices:
  matrix-vector product (+ backward), in-place add/sub matvec, linear solve,
  inverse.
- **Resampling** — spline resampling (`resample`), its adjoint (`restriction`,
  prolongation-transpose), and spline coefficient prefiltering (`spline_coeff`).
- **Pushpull** — spline-interpolation gather (pull), scatter (push), count, and
  spatial gradient.
- **Regularisers** — spatial regularization (absolute/membrane/bending) on
  multi-channel fields and on vector flows: matvec and diagonal.

## Build & test

```
make                        # = make all = cpu + hub -> build/libfastfields.so
make cpu                    # build/lib/libfastfields-cpu.so   (never needs nvcc)
make cuda                   # build/lib/libfastfields-cuda.so  (needs nvcc)
make test                   # test-lib-cpu + test-lib (no GPU toolchain needed)
make test-lib-cpu           # THE gate: CPU suite vs. brute-force references
make test-lib               # hub argument-validation tests (link nothing, seconds)
make test-impl-cuda         # compile-only nvcc probe (tests/impl-cuda/*.cu)
```

`CXX=clang++` and `CXX=g++` both work. **There are no submodules on `main`**
and no symlinks to set up — that was the pre-consolidation arrangement.

`make/common.mk` sets `.DEFAULT_GOAL := all` explicitly. Do not remove it: every
Makefile here `include`s that fragment *before* declaring its own targets, so
without it make picks the output-directory rule as the default goal and a bare
`make` silently builds nothing. `fastfields-dlpack`'s `setup.py` invokes
`make` with no target and depends on this.

**Op correctness is gated by `make test-lib-cpu`** — there is no GPU in CI, so
the CPU path is the tested source of truth and CUDA is **compile+link only**.

## CI

`.github/workflows/ci.yml`, path-filtered. `codespell` always; `test-cpu` (a
3-leg `BOUNDFLAGS`/`SPLINEFLAGS` matrix + an `INDEXFLAGS` leg + a g++ leg),
`sanitize` (ASan+UBSan) and `tsan` on kernels/cpu/hub changes; `test-hub` on
hub changes; `build-cuda` (two legs, one per `FF_INDEX32` position) and
`compile-probe-cuda` on kernels/cuda changes.

**The `tsan` leg is the only one that runs anything in parallel.** With the
shipping `GRAIN_SIZE` (32768) every workload in `tests/lib-cpu/` is below the
threshold at which `parallel_for` hands work to the thread pool, so the whole
59,886-check suite executes single-threaded -- measured, zero `clone` syscalls
across all 13 binaries. That leg rebuilds the same suite with
`-DFF_GRAIN_SIZE=1` so the same checks run concurrently, under ThreadSanitizer.
It is not a performance knob: results must be identical at any grain size, and
32768 stays the shipping value.

`build-cuda` builds `libfastfields-cuda.so` **and then links the hub against it**
(`make lib USE_CUDA=1`). That second step is the point: building the CUDA
library alone cannot notice an entry point missing from it, because nothing
inside the library references those entry points — the hub does. It also prints
each module's peak `nvcc` RSS, so the memory figures quoted in
`src/lib-cuda/Makefile` can be re-checked rather than trusted.

**A change under `impl/kernels/` (or `core/`, or the build system) triggers
everything** — both backends compile them. Only `api/`-only or `src/lib`-only
changes may skip CUDA.

pushpull's fully-static order×bound compile is nightly
(`.github/workflows/nightly-pushpull.yml`), never on PRs.

## Conventions & caveats

- **C++11** for the CPU and hub layers; **C++14** for the CUDA layer (nvcc).
  Object rules need `-fPIC`. Adding a module means adding it to `MODULES` in
  `src/lib-cpu`, `src/lib-cuda` **and** `src/lib`.
- `libfastfields.so` links `-lfastfields-cpu` with an `$ORIGIN/../lib` rpath,
  and with `-Wl,--no-undefined` (`$(NO_UNDEFINED)` in `make/common.mk`, cleared
  on macOS/Windows where the linker has no such option). A shared object is
  otherwise allowed to carry unresolved symbols, which is how fastfields-lib#80
  hid: four modules missing from `src/lib-cuda`'s `MODULES` left eleven
  `FF_CUDA::` entry points undefined with every build green. The hub link is
  now the gate on backend completeness — forget a `MODULES` entry and it fails
  there.
- Op renames from the impl layer: `resize -> resample`,
  `restrict -> restriction`, `splinc -> spline_coeff` (a namespace cannot share
  a name with a function inside `ff::cpu`). **`restriction` accumulates into
  `out`**, so callers pre-zero it.
- **Boundary conditions and spline orders may be compile-time *or* runtime.**
  `bound::utils<B>` / `spline` statics versus `bound::dyn<B>` and
  `type::Dynamic`. Which are statically instantiated is a build-time choice via
  `BOUNDFLAGS` / `SPLINEFLAGS`. These live **outside** `CXXFLAGS` on purpose so
  that a `CXXFLAGS=` override (as CUDA CI does, to force `-O1`) cannot silently
  drop the policy.
- **So is the 32-bit index axis**, via `INDEXFLAGS` / `FF_INDEX32`
  (`core/dispatch.h`) — the third member of that family and the most expensive
  of the three: every templated kernel is templated on `offset_t`, whose two
  values are chosen per call by `canUse32BitIndexMath`, so the narrow path is
  exactly ×2 instantiations of everything below the dispatch layer.
  `INDEXFLAGS="-DFF_INDEX32=0"` collapses both arms onto `int64_t`. Same
  outside-`CXXFLAGS` rule, and **the default (on) is set separately in
  `src/lib-cpu/Makefile` and `src/lib-cuda/Makefile`** — that per-library
  default is what makes it a per-backend option, so do not hoist it into
  `make/common.mk`. The narrow path is an inherited ATen register-pressure
  optimisation that has never been benchmarked here (no GPU in CI); the
  default does not move without one.
- **CUDA memory limits are measured, not guessed** — and the numbers that used
  to be recorded here were wrong. CUDA CI forces `-O1` and `-j2` because
  `ptxas` was OOM-killed at ~16 GB on `reg_field.cpp`, and `src/lib-cuda`'s
  `MODULES` split (`reg_field`/`reg_field_rls`, `reg_flow`/`reg_flow_rls`) is
  what keeps that build possible. But the "~3.8 GB per module" figure this note
  carried is not what the build actually does: `build-cuda` now measures every
  module and **`reg_flow` peaks at 12.98 GB of a 16 GB runner**. The measured
  table lives above `MODULES` in `src/lib-cuda/Makefile`; read it before
  changing `-j`, `-O`, the bound/spline policy, or anything that makes a
  regulariser heavier. Do not recombine or "tidy" the split.
- **Macros in installed headers are `FF_`-prefixed.** Anything `#define`d under
  `include/` and not `#undef`'d before the end of that header is inherited by
  every downstream translation unit, so it must be namespaced by prefix. Macros
  private to a single `.cpp` are exempt — but the moment one is hoisted into a
  header it stops being private, which is why de-duplicating a helper and
  prefixing it are the same change. Two documented exemptions live in
  `core/cuda_switch.h` and must keep their spelling to work at all: the
  `__CUDACC_RTC__` fixed-width integer definitions (NVRTC ships no standard
  library, so those *are* `<cstdint>` there) and the non-nvcc `__device__` /
  `__host__` fallbacks. Prefer an `inline` function to a macro where one will
  do — a function in `ff::` is collision-safe without any prefix.
- `include/fastfields/core/dlpack.h` is vendored upstream code: do not edit it,
  and it is skipped by `codespell` (see `.codespellrc`).

## Pointers

- **[`MIGRATION-PROVENANCE.md`](MIGRATION-PROVENANCE.md)** — which source repo
  became which path, the `(#N)` mis-link hazard, and the issue old→new mapping.
- **[`MIGRATION.md`](MIGRATION.md)** — the status matrix, per-module porting
  pattern, bugs fixed, open TODOs.
- **[`tools/consolidate.sh`](tools/consolidate.sh)** — the frozen path-rewrite
  rules, with a warning about re-running them.
- Hierarchy overview: `/home/user/.github/profile/README.md`.
