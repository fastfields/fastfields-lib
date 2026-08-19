# fastfields → teeny migration — plan & living status

Tracking issue: **fastfields-lib#21**. This is the design + status companion for
reimplementing the fastfields C++/CUDA stack on **teeny**
([`balbasty/teeny`](https://github.com/balbasty/teeny)), a header-only C++17
tensor library on `cuda::std::mdspan`. It is the teeny-track sibling of
`MIGRATION.md` (which tracks the original jitfields→fastfields port).

**Branch model.** Every repo has a long-lived integration branch, `teeny` —
branched off `main`, and treated as *our main* for this effort until it's
eventually merged back into `main` once the migration is complete. It's a
shared dev branch, not the property of whichever agent/session is currently
pushing to it. Per-task work goes on `claude/<task>` branches → issue → PR →
review (fable on core/perf-sensitive diffs) → squash-merge into `teeny`. We
never touch `main` or the parallel non-teeny track
(`claude/jitfields-fastfields-migration-*`). Because the layers are separate
repos linked by submodule pins, one *module* port is a set of coordinated PRs
(kernels + impl + lib), developed together through the dev-tree symlinks but
committed per-repo; each PR references its own repo's task issue.

---

## 0. Thesis

teeny absorbs the machinery fastfields hand-rolls, so kernels read like the math
and a single source folds to immediates when shapes are static and stays generic
when dynamic. Goal: leaner / more readable / more maintainable code **at no
runtime cost**. GPU is untestable in CI, so correctness + perf are judged by
static analysis, the CPU oracle tests, and skeptical (fable) review — not runtime
timing alone.

teeny's floor is **C++17** (CCCL); the stack is **C++11** today. Adopting teeny in
the kernels moves the compiling layers to C++17. Verified low-risk: the current
C++11 distance chain already compiles + passes at `-std=c++17` (clang, 2350
checks; benign `_GLIBCXX_DEPRECATED` warning only).

## 1. What teeny deletes (measured against the source)

| current fastfields machinery | teeny replacement |
|---|---|
| `kernels/batch.h` `index2offset`/`sub2offset`/`index2sub` (274 LOC) | `peel_front<-Sr>()` / `peel_front_at<-Sr>(i)`; folded `t.stride(Int<d>())` |
| `kernels/utils.h` `StaticValue<T,V>` + ~40 operator overloads | `Int<V>` / `integral_constant` folding (built in) |
| `canUse32BitIndexMath` + `autocast.h` `copy_if_needed`/`ContiguousStrides` | `dispatch_index(v,f)` (int32 arm when `index_fits`) + `from_dlpack` |
| `kernels/utils.h` `min/max/abs/sign/mod/prod/square/pow/sqrt/fillfrom/fill` | teeny math members + `compute_type` |
| `kernels/vector/*` (11 headers of static/dynamic pointer+vector abstractions) | teeny views (`wrap`, `strides<S...>`, `local`) |
| `pushpull/{1d,2d,3d,nd}.h` per-rank hand-unrolled gather/scatter trees | ONE separable recursion over static `D` (teeny `examples/fastfields/pushpull.hpp`) |
| `has_atomic_add` fork | `at(i...).atomic_add_(v)` (atomic on device, `+=` on host) |
| per-op dtype×dim×spline×bound dispatch macros in `*-lib/*.cpp` | `dispatch_dlpack` / `dispatch_value<...>` at the ONE boundary |

**Stays in fastfields** (teeny deliberately omits): the CPU parallel-for driver
(`parallel.h` / `threadpool.*`); the boundary-condition index maps + spline
weight tables (domain code — reference in teeny `examples/fastfields/{bounds,spline}.hpp`);
the DLPack ABI surface at the `lib` boundary.

## 2. Layer mapping — keep the repos, refactor their internals

Collapsing repos is orthogonal to teeny adoption, collides with the parallel
track and the python chain, and risks the DLPack ABI — so the 6-tier structure is
preserved; each layer just gets leaner.

| layer | today | on teeny |
|---|---|---|
| **kernels** | single-element math on raw `scalar_t* + stride + size` | per-line/cell math on a teeny rank-k **view** (`template<class V> CUDEV void l1_line(V,w)`) — folds when static |
| **{cpu,cuda}-impl** | `parallel_for` + `index2offset` per element | `parallel_for` / CUDA launch over `peel_front<-Sr>` cells; the index math is gone |
| **{cpu,cuda}-lib** | dtype×width dispatch macros + `copy_if_needed` | `from_dlpack` + `dispatch_value` (dim/order/bound) at one point; offset-width narrowing on the CUDA side only (§9 R5). **Planned 2026-07, never executed** — `from_dlpack` appears nowhere in fastfields today; this row is what umbrella #57 is now actually doing (§9) |
| **lib** | device dispatch on `DLTensor` | unchanged in spirit (DLPack device dispatch) |
| **binds / numpy·cupy·torch** | DLPack in/out | **unchanged** — the ABI is preserved |

## 3. Build integration (verified locally)

The include scheme today is quote-includes resolved through the dev-tree symlink
nesting; `INCLUDES` is empty; only the cpu-lib **test** rule adds `-I.`.

- teeny enters as a git submodule of **`fastfields-kernels`** at `external/teeny`
  (recursive → brings CCCL), pinned to a teeny integration commit — mirrors
  teeny's own `external/cccl` and the "symlink in dev, real submodule in release"
  pattern. kernels is the shared leaf both impls include → the DRY home. Dev tree:
  `fastfields-kernels/external/teeny -> /home/user/teeny`.
- In the compiling Makefiles (`cpu-lib`, later `cuda-lib`/`lib`) add to `INCLUDES`:
  `-I impl/kernels/external/teeny/include` and
  `-I impl/kernels/external/teeny/external/cccl/libcudacxx/include`; bump
  `-std=c++11` → `-std=c++17`.
- CI already checks out `submodules: recursive` with `CI_SUBMODULE_TOKEN`, so
  teeny+CCCL fetch automatically once registered. Bump CI std to c++17; consider a
  g++ leg (teeny is a 2-compiler gate; fastfields CI is clang-only).

## 4. Roadmap & status

Ordered by risk × representativeness. Legend: ☐ todo · ◐ in progress · ☑ done.

| # | module | scope | status |
|---|---|---|---|
| — | **substrate** | teeny submodule under kernels (kernels#11) + C++17 bump (cpu-lib#16) | ☑ |
| 0 | **distance (L1 + euclidean)** | slice 0; done — cpu-impl#8 (peel) + cpu-lib#18 (dispatch); 2350-check oracle green on clang++/g++ + asan/ubsan | ☑ |
| 1 | **posdef** | all 5 SPD layouts on `matrix.h` — kernels#12/#13 + cpu-impl#9/#10 + cpu-lib#19/#20; 5090-check oracle green on clang++/g++ + asan/ubsan | ☑ |
| 2 | **pushpull** | flagship; one separable recursion replaces the 1d/2d/3d trees — kernels + cpu-impl + cpu-lib; 324-check oracle green on clang++/g++ + asan/ubsan/tsan | ☑ |
| 3 | **splinc / resize / restrict** | 1-D IIR + pull/push-with-scale | ☐ |
| 4 | **regularisers (field / flow)** | stencil operators | ☐ |
| 5 | **distance spline / mesh** | heavier distance variants | ☐ |

Rows 3–5 are now sequenced by the **tensor-native-boundaries umbrella (#57)**,
which reshapes every call boundary before/while porting them — its own phase
list (0/A–E) is in **§9.4**, and every phase there is gated by **§9.3**.

Compile-time note: pushpull's dim×spline×bound×dtype matrix is a ~40-min build
today; teeny's static folding must not worsen it, and the `dispatch_value`
boundary may help — measure, never trade runtime for it.

## 5. Slice 0 — distance, concretely

Current chain (read end to end):
- `kernels/distance/l1.h`::`kernel(f,size,stride,w)` — 1-D fwd+bwd sweep.
- `kernels/distance/euclidean.h`::`kernel(f,v,z,d,w2,size,stride,stride_buf)` —
  lower-envelope-of-parabolas + `intersection`/`fillin`, 3 scratch buffers.
- `cpu-impl/distance_{l1,euclidean}.h`::`dt(ndim,f,w,size,stride)` —
  `parallel_for(0, prod(size,nbatch))` → `index2offset` → `kernel(f+off,n,s,w)`;
  euclidean `new[]`s v/z/d per worker.
- `cpu-lib/distance.cpp`::`dt_{euclidean,l1}(DLTensor&,w,stream)` — normalise null
  strides, `DISPATCH_DT` on float32/64 × int32/64, `copy_if_needed`, call impl.
- `lib/distance.cpp` — device dispatch → cpu. Operates along the **last axis**;
  Python orchestrates the per-axis sweeps (contract preserved).

teeny target (skeleton; final form pending the §6 fable review):
```cpp
// kernels/distance/l1.h — 1-D sweep on ANY rank-1 view (folds when static)
template <class Line, class W>
CUDEV void l1_line(Line line, W w) {
  const auto n = line.extent(0);
  if (n <= 1) return;
  auto tmp = line(0);
  for (long i = 1;   i < n;  ++i) { tmp = min(tmp + w, line(i)); line(i) = tmp; }
  for (long i = n-2; i >= 0; --i) { tmp = min(tmp + w, line(i)); line(i) = tmp; }
}
// cpu-impl/distance_l1.h — batch loop = peel_front<-1> + the CPU driver
//   parallel_for(0, at.size_front<-1>(), GRAIN, [&](s,e){
//     for (i in [s,e)) l1_line(at.peel_front_at<-1>(i), w); });
// cpu-lib/distance.cpp — from_dlpack + dispatch_index, no autocast macros
```
Euclidean: same skeleton; scratch v/z/d become teeny rank-1 buffers (or keep
`new[]` for the first cut) indexed as views.

**Correctness gate:** the existing `cpu-lib/tests/test_distance.cpp` (brute-force
O(n²) oracle, float32/float64, euclidean+l1 — 2350 checks) runs unchanged against
the new impl (same `ff::cpu::dt_*` ABI). Add teeny static_asserts (folded stride
on a static-shape line). Sanitizers (asan/ubsan) on the host path.

Slice 0 lands as: **PR-A** (substrate) then **PR-B** (distance refactor).

## 6. Resolved by the fable review (distance slice)

1. **anyrank in the hot loop → RESOLVED: keep it (Candidate A).** Per-cell
   `peel_front_at<-1>(i)` is arithmetically *cheaper* than `index2offset` (one
   fewer multiply per batch dim); the anyrank cell is a trivially-copyable 24-byte
   `layout_stride` view. Rejected `dispatch_rank`/`fixed<R>()` (Candidate B): ~33
   instantiations, and `fixed<R>` yields all-dynamic `layout_stride` so *nothing
   folds* — pure compile-time bloat, zero runtime win for a last-axis sweep.
2. **Keep the raw-pointer sweep.** The kernel is called on the cell's
   `data()/extent(0)/stride(0)`, so the inner loop is byte-identical — teeny
   replaces only the batch plumbing. (A teeny-native `uget(i)` sweep is safe too
   but separable; deferred.)
3. **Euclidean scratch** — kept the existing per-worker `new[]` v/z/d (unchanged;
   teeny doesn't touch the scratch).
4. **int32 narrowing → CPU: dropped (a wash on 64-bit ALU); GPU: keep whole-carrier
   host-side narrowing.** `dispatch_index` is `_TNY_HOST` (unusable inside
   `__global__`), so per-cell narrowing can't be the CUDA mechanism — the CUDA port
   keeps the moral equivalent of `autocast.h` (host-side, whole-carrier).
5. **Layer boundary** — kept the kernel/impl split for slice 0; revisit after
   pushpull (where the per-rank trees actually collapse).

### Open teeny-side follow-up
- **teeny#181** — a rank-preserving DLPack dtype dispatch (dispatch on dtype, keep
  the `anyrank`; struct-agnostic so a downstream's own `DLTensor` works). Not
  needed for distance; design it against posdef/pushpull's richer dispatch.

## 7. posdef — concretely

The per-voxel SPD matrix ops, ported across all three CPU repos (kernels#12/#13,
cpu-impl#9/#10, cpu-lib#19/#20).

**kernels — one header, five layouts.** `posdef/matrix.h` replaces the
per-layout `.inl` template hacks with `ff::<dev>::posdef::{eye,diag,estatics,
sym,full}` + `chol`, each expressing its packed layout on teeny views. The
packed last-dim length selects the layout via `guess_type(C, CC)`: Eye (`CC=1`),
Diag (`CC=C`), ESTATICS (`CC=2C-1`), Sym (`CC=C(C+1)/2`, diag-then-rows), Full
(`CC=C²`). The `Eye>Diag>ESTATICS>Sym>Full` priority is the *efficiency* order,
and the only collisions are exact equivalents — C=1 (all layouts == a scalar)
and C=2 (ESTATICS == Sym: same matrix, same packing) — so picking the cheapest
never changes a result. `reduce_t = double` accumulation (regardless of
`scalar_t`), the `1.000001` diagonal ridge, and the `1e-40` pivot floor are
carried over from jitfields verbatim.

**impl — peel, no index math.** `posdef.h` builds a per-tensor `anyrank` over
`(*batch, trailing)` via a small `_any(ptr, size, nbatch, trailing, stride)`
helper (each tensor carries its *own* trailing extent — out/inp = C, hessian =
CC — so the shared `size` array can't force the wrong last dim) and hands each
cell to `matrix.h` through `peel_front_at<-1>`. matvec/addmatvec_/submatvec_ are
`<type Ty, int C, …>` (layout + static-C generic); solve/solve_ are layout-generic
with a dynamic C (Sym/Full route through a `double` Cholesky workspace, the rest
solve in place). matvec_backward and invert(_) stay Sym-typed — their channel
count is inferred from the packed length, which is only well-defined for Sym.

**lib — dispatch by layout.** `sym_matvec`/`addmatvec_`/`submatvec_` dispatch
layout × static-C{1,2,3}/dynamic × dtype × offset; `sym_solve`/`solve_` dispatch
layout × dtype × offset (dynamic C). The old Sym-only `CC == C(C+1)/2`
precondition is gone. A `CHECK_RANK` guard pins each posdef tensor to `nbatch+1`
dims — the impl reads exactly `nbatch+1` strides but derives `CC` from the true
last dim, so a longer-rank tensor would otherwise be decoded against the wrong
axis.

**Correctness gate.** `cpu-lib/tests/test_posdef.cpp` extended to build a dense
SPD per layout, pack it, and check matvec/solve vs a brute-force reference
across layouts — 5090 checks, 0 failures on clang++ and g++, clean under g++
asan+ubsan. fable independently confirmed the new kernels bit-exact against the
old ones (22,210 old-vs-new checks).

**Follow-ups.**
- cuda-impl still consumes the old `posdef/{eye,diag,estatics,full}.inl` +
  `cholesky.h`; port its `posdef.h` onto `matrix.h` by analogy (nvcc compile+link
  only, no GPU in CI), then delete those kernel files.
- **[perf] static-C solve/invert (cpu-lib#21).** matvec dispatches static-C
  {1,2,3}; solve/invert dispatch dynamic-C only (pre-existing — the original did
  the same). Static C would fold the per-voxel Cholesky/closed-form loops (the
  same win as matvec), at a compile-time cost (Cholesky × 5 layouts × 3 C × dtype
  × offset). The teeny cpu-impl port removed the dead `sym_{solve,invert}_tpl(_)`
  static-C scaffolding (cpu-impl#10); cpu-lib#21 records the optimisation so it's
  not lost. Measure (CPU `-O3 -S` + micro-timing) before wiring `DISPATCH_C`.

### posdef teeny-side follow-ups — all shipped, adoption pending a pin bump
teeny closed all three items posdef surfaced; the merged posdef was built against
the pre-feature pin (`/home/user/teeny` @ b3830b5), so adoption is a later slice —
fold it into the teeny-pin bump the pushpull port needs anyway, then re-run the
full CPU oracle.
- **teeny#181** (via teeny#182) — shipped the *rank-preserving* `dispatch_dlpack_dtype`.
  But fastfields vendors its **own** DLPack structs, so the piece our lib layer can
  actually consume is the *struct-agnostic* n-ary dispatch (`dispatch_enum` over
  dtype × layout × static-C), still **deferred pending a concrete spec from me** —
  bring it once the posdef + pushpull lib layers pin down the signature.
- **teeny#183** — uninitialised-stack construction (skip the zero-fill before an
  immediate overwrite: Cholesky workspace / matvec accumulators). *Done; adopt.*
  Parity holds today via DSE, so this is latent-safety, not a measured win.
- **teeny#184** — a portable `TNY_UNROLL` unroll-pragma macro. *Done; adopt* to
  delete posdef's hand-rolled `FF_POSDEF_UNROLL`. Pure DRY — `FF_POSDEF_UNROLL`
  already emits `#pragma GCC unroll` on gcc, so no fold is lost today.

## 8. pushpull — concretely (the flagship)

Spline gather/scatter/count/grad, ported across kernels + cpu-impl + cpu-lib.

**kernels — one recursion, no per-rank trees.** `pushpull/teeny.h` replaces the
hand-unrolled `1d/2d/3d.h` gather/scatter trees (`nd.h` was broken WIP) with a
single separable recursion over the static spatial rank D
(`vox::pull/push/count/grad`). Spatial rank D, interpolation order O, AND boundary
B are compile-time (the K=O+1 tap loops fully unroll — validated by a clang++/g++
`-O3 -S` codegen probe: zero tap backedges even at cubic; matches the old
hand-unrolled corner trees). The boundary **sign is folded into the weight** at
setup, so the hot loop is a branchless multiply-accumulate. `reduce_t=double`;
FOV/extrapolate preserved. **Hybrid bound:** `B == bound_t::Dynamic` takes a
runtime route (per-axis switch on `rt`) so the lib compiles common bounds
statically and routes rare ones through one Dynamic instantiation.

**cpu-impl — anyrank peel, flat scatter.** The driver deletes `index2offset` /
`fillfrom`: all four ops parallelise flat over the grid voxels (batch ×
spatial_grid) — out/grid peel the last 1 (grad: 2) dims to the voxel cell; inp
peels only the batch. Reads are voxel-parallel (NOT batch-cell-parallel, which
would single-thread nbatch≤1). Scatters (push/count) use `anyAtomicAdd`, a
lock-free CAS on the host (kernels#14) so they too parallelise flat, race-free
(TSan-verified). Each tensor is wrapped from its OWN shape (per-tensor decode);
fable-reviewed correct.

**cpu-lib — dispatch + bound split.** Full ndim(1/2/3) × order(0-7) × bound ×
dtype × offset matrix; `-DFF_TEST_SPARSE` trims order×bound for tests (DCT2 only
outside Linear/Cubic). Bound split (cpu-lib#22): static DFT/DCT2/DST2/Zero/NoCheck;
DCT1/DST1/Replicate → the Dynamic instantiation. `CHECK_SAME_SPATIAL` + a
`TNY_MAX_RANK=64` bump guard the per-tensor decode / anyrank meta cap.

**Gate.** `test_pushpull.cpp` extended with the classes the covering matrix missed
(nbatch≥2, anisotropic 3D catching axis swaps, extrapolate 0/-1): **324 checks, 0
failures on clang++ and g++, clean under asan+ubsan** (and the driver under TSan).

**Decisions/issues:** cpu-lib#22 (bound split), cpu-impl#11 (hess + the four
`*_backward` ops deferred — compiled but never exported; port when autograd
bindings need them), kernels#14 (the valid host atomic — a latent-race fix
surfaced here, cross-cutting to reg_*/restrict).

## 9. Tensor-native call boundaries (the umbrella-#57 convention)

Every module ported so far builds a teeny carrier *inside* its impl body and
throws it away at the door: impl entry points still take
`(nbatch, T*, size[], stride_out[], stride_inp[], …)`, and `*-lib` still explodes
a `DLTensor` into raw arrays through `autocast.h` to feed them. That was
migration inertia — each port stayed bit-verifiable against the pre-teeny ABI —
not a design decision, and the price is metadata built, torn down and rebuilt up
to three times per call chain. Umbrella **#57** fixes the boundaries themselves.

This section is the **reference an implementer checks a PR against mid-flight**.
The narrative, the survey it came from, and the rejected alternatives live in
#57; what follows is only the part that has to be quotable.

### 9.1 Three carriers, one per boundary kind

| boundary | carrier | existing in-tree model |
|---|---|---|
| **kernels ↔ impl** (rebuild together) | fixed-rank teeny **views**, template-deduced; `D`/`O`/`B`/dtype/offset-width stay compile-time template parameters | posdef `kernels/posdef/matrix.h` — `matvec(Ov&& o, const Hv& h, const Xv& x)`, five packed layouts, extents read as `x.extent(Int<0>())`; pushpull `kernels/pushpull/teeny.h` — `vox::pull<D,O,B,…>(VOut out, const VIn inp, const reduce_t loc[D], …)` — note the sub-voxel coordinate stays a plain `D`-array: it is a vector of *values*, not a tensor operand, and R2 does not apply to it |
| **impl ↔ \*-lib** (rebuild together) | typed **`anyrank` carriers**, one per tensor (+ small views/spans for parameter vectors); `nbatch`/`nc`/`size[]`/`stride_*[]` arguments disappear | cuda-impl `reg_field.h`/`reg_flow.h` — `_matvec_*_k(AO ao, AI ai, …)`, anyrank carriers passed **by value** into `__global__`, peeled with `peel_front_at<-1>(i)` inside |
| **\*-lib exported symbols and above** | **`DLTensor`, unchanged** — it already *is* the ABI-stable tensor; no teeny handle type is invented for the `.so` edge | `lib/*.cpp` → binds → numpy/cupy/torch, untouched |

`*-lib` builds the carriers once, via `tny::from_dlpack(&dlt)` inside its
existing dispatch, and hands them down. The pattern is not new — tiers 1 and 3
are already shipped and perf-proven; #57 applies tier 2 to the middle, which is
the only layer that never got it.

### 9.2 The rules

**R1 — Carrier-only refactor.** Change *what* a boundary passes, never *where*
dispatch happens: runtime→static dispatch (dtype, offset width, spatial rank
`D`, order, bound) stays in `*-lib`, and every impl entry keeps its template
parameters. *Why:* pushing rank/dtype dispatch down into impl was evaluated as
Candidate B in the distance-slice review (§6, item 1) and rejected — ~33
instantiations, and `fixed<R>()` yields an all-dynamic `layout_stride` so
*nothing folds*: compile-time bloat for zero runtime win.

**R2 — Derive, don't pass.** No `nbatch`, `nc`, `size[]` or `stride[]`
parameter where a carrier already answers the question. *Why:* every such
argument is a second source of truth for something the tensor knows, and it is
what forces the build/teardown/rebuild round trip in the first place. Spellings:
on an `anyrank` the rank is the public member **`at.ndim`** (there is no
`rank()` accessor on the carrier), the trailing channel count is
**`at.size(at.ndim - 1)`** — **not** `at.shape(-1)`: for a `copy_meta` carrier
(what `from_dlpack` produces) `shape` is a *fixed* `TNY_MAX_RANK`-extent tensor
with only the leading `ndim` slots live, so `operator()`'s negative-index wrap
lands on slot `TNY_MAX_RANK - 1` (uninitialised), not the last *live* axis —
found and fixed during Phase A (fastfields-cpu-impl#60/fastfields-cpu-lib#74),
where it would otherwise have silently read garbage. A batch count is
`at.ndim - D - 1`; on a peeled cell — which *is* a fixed-rank view — use
`cell.rank()`, `cell.extent(0)`, `cell.stride(0)` (`shape(-1)`-style negative
indexing IS safe there, since a fixed-rank view's shape tensor has exactly
`rank()` elements, not a padded store). This is about *geometry only*: `D`,
order, bound, dtype and offset width are not derivable from a carrier and stay
template parameters supplied by the `*-lib` dispatch (R1).

**R3 — Each tensor carries its own metadata.** One carrier per tensor, built
from *that* tensor's own shape and strides; never a `size[]`/`stride[]` array
shared across operands. *Why:* a shared array lets one operand's geometry decode
another's — posdef's out/inp trailing dim is `C` while the hessian's is `CC`,
which is exactly why cpu-impl's `_any` helper takes a per-tensor trailing extent
(§7) — and a shared *length* argument is the recurring hazard behind the
`copy_if_needed`-length audits in `distance.cpp`/`splinc.cpp`. Per-tensor
carriers retire the whole class structurally rather than by re-auditing it.

**R4 — Const-correctness crosses the boundary.** Read-only operands are
`anyrank` carriers / views of `const T`. *Why:* probed against the current teeny
pin on clang++ **and** g++ — `as_anyrank(const float*, …, copy_meta)` and
`from_dlpack<const float>(&dlt)` compile, peel and read correctly, and
write-through a peeled cell is a compile error — so a read-only signature costs
nothing at runtime and is enforced by the type system. There is **no teeny-side
gap** here; do not work around it with a `const_cast`.

**R5 — D1: no int32 offset dispatch on CPU; the GPU narrows whole-carrier,
host-side.** *Why:* the CPU int32 arm was measured a wash on a 64-bit ALU
(distance-slice review, §6 item 4; distance already runs int64-only), while the GPU keeps
it because occupancy is register-bound, the SM has no native 64-bit IMAD, and
the `copy_meta` store itself halves. Mechanism: **balbasty/teeny#467**
(`anyrank::index_fits`/`reindex`) — until it ships, cuda-lib keeps the moral
equivalent of `autocast.h` **on its narrowing path only**, and nowhere else.
`dispatch_index` is `_TNY_HOST`, so per-cell narrowing can never be the CUDA
mechanism.

**R6 — Device carriers stay trimmed.** A carrier passed by value into a
`__global__` is built with **`copy_meta`** (inline shape/stride store — a view
carrier holds *host* pointers and is UB on the device) and, where several ride
one launch, an explicitly **capped `MaxRank`** with a host-side rank check.
*Why:* kernel parameter space is 4 KiB and a carrier measures 1040 bytes at
`TNY_MAX_RANK=64` / 528 at 32, so the four carriers of a JRLS `relax_*` launch
overflow it outright at 64 (nvcc: *"Formal parameter space overflowed"*) and fit
in 2112 bytes when capped — the existing `FF_REG_{FIELD,FLOW}_MAX_RANK 32`
pattern is the rule, not an accident.

**R7 — DLPack include order: fastfields' vendored `dlpack.h` *before*
`<teeny/dlpack.h>`.** *Why:* fastfields vendors DLPack **v1.2**, teeny vendors
**v1.1**, both use the same include guard `DLPACK_DLPACK_H_`, so whichever is
seen first wins for the entire TU and the other is silently skipped. v1.2 is a
pure superset (verified: it only *adds* — `kDLTrn = 18` and the v1.2 exchange-API
declarations — and changes no shared enumerator value), so fastfields-first keeps
every TU on one consistent, newer set of definitions.

### 9.3 Per-phase performance gate (non-negotiable)

Quoted from #57 — the methodology this project already proved out: **object-code
identity where code doesn't change, oracle identity everywhere, measured never
argued.**

- Functions the phase doesn't touch (e.g. `kernel()` sweeps, stencil
  contractions): **byte-identical instantiations** at `-O2` (precedent:
  kernels#60's byte-identical `reg_field.o` under clang across a whole
  template-layer insertion).
- Functions the phase re-skins (driver loops): disassembly diff — **no new
  instructions inside the per-element loop**; anything added must be explained or
  eliminated before merge.
- Every existing CPU oracle suite green with **unchanged check counts**, clang++
  AND g++ (with a genuine `make clean` between — cpu-lib#56's trap), asan/ubsan
  on touched paths.
- CUDA: compile+link under nvcc (no GPU in CI), kernel-parameter budget watched
  (trimmed `MaxRank` carriers stay the rule — R6).

"Unchanged check counts" means unchanged, not "still passing": a suite that
silently runs fewer assertions is a failed gate.

### 9.4 Umbrella #57 phases

Legend as §4: ☐ todo · ◐ in progress · ☑ done.

| phase | scope | issues | status |
|---|---|---|---|
| **0** | this convention, written down before any code moves | lib#58 (+ teeny#467 filed) | ◐ |
| **A** | prototype + **gate** on distance l1/euclidean; one coordinated PR-set — everything downstream waits for this gate to actually pass | cpu-impl#58 + cpu-lib#72 | ☐ |
| **B** | fan-out per module (cpu + cuda, one coordinated PR-set each): posdef, pushpull, resize/restrict/splinc, reg_field, reg_flow — each drops its CPU int32 arm (R5), deletes its `VOIDPTR`/`copy_if_needed` block, and moves `as_anyrank` up into `*-lib` | filed after the Phase A gate passes | ☐ |
| **C** | kernel-driver re-skins to views (regulariser `nd.h`/`stencil.h` outer parameter lists) + deletions: `kernels/vector/` (already dead), `batch.h`/`utils.h` remnants | — | ☐ |
| **D** | mesh essence rewrite — delete `mesh_utils.h` (976 lines), point algebra onto teeny vocabulary, `distance_mesh` impl onto carriers/peel. A rewrite with a 1,280-check oracle, not a re-skin | — | ☐ |
| **E** | `distance_spline` **last**: test-hardening prerequisite first (weakest oracle in the project), then the `vox::` bridge (blocked on kernels#33, measured 2.3–2.36×), then carriers; retires legacy `pushpull/1d.h` | — | ☐ |

**Not at risk.** The regulariser engine (kernels#50 phases 1–6) is foundation,
not casualty: `stap.h` / `stencil.h` / `field/nd.h` / `flow/nd.h` math, the
`subsample` colour views, the `enumerate` loc decode and every verified numeric
stand. What changes is the ~36 impl entry-point signatures and their three-line
`as_anyrank` prologues — the layer those phases deliberately left alone — plus
(Phase C) the drivers' parameter lists.

---
_Living document — update in the same PR as the code it describes._
