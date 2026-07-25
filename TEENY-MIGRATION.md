# fastfields → teeny migration — plan & living status

Tracking issue: **fastfields-lib#21**. This is the design + status companion for
reimplementing the fastfields C++/CUDA stack on **teeny**
([`balbasty/teeny`](https://github.com/balbasty/teeny)), a header-only C++17
tensor library on `cuda::std::mdspan`. It is the teeny-track sibling of
`MIGRATION.md` (which tracks the original jitfields→fastfields port).

**Branch model.** Every repo has a long-lived integration branch
`claude/fastfields-teeny-refactor-js42id` — branched off `main`, and treated as
*our main* for this effort. Per-task work goes on `claude/<task>` branches → PR
→ review (fable on core/perf-sensitive diffs) → squash-merge into the integration
branch. We never touch `main` or the parallel non-teeny track
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
| **{cpu,cuda}-lib** | dtype×width dispatch macros + `copy_if_needed` | `from_dlpack` + `dispatch_index` (+ `dispatch_value` for dim/order/bound) at one point |
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
| 2 | **pushpull** | flagship; deletes 1d/2d/3d/nd trees; adjoint-identity gate | ☐ (next) |
| 3 | **splinc / resize / restrict** | 1-D IIR + pull/push-with-scale | ☐ |
| 4 | **regularisers (field / flow)** | stencil operators | ☐ |
| 5 | **distance spline / mesh** | heavier distance variants | ☐ |

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

**Follow-up.** cuda-impl still consumes the old `posdef/{eye,diag,estatics,
full}.inl` + `cholesky.h`; port its `posdef.h` onto `matrix.h` by analogy
(nvcc compile+link only, no GPU in CI), then delete those kernel files.

### posdef teeny-side follow-ups surfaced
- **teeny#183** — uninitialised-stack-tensor construction, to skip the zero-fill
  before an immediate overwrite (Cholesky workspace / matvec accumulators).
- **teeny#184** — a portable `TNY_UNROLL` unroll-pragma macro (posdef hand-rolls
  `FF_POSDEF_UNROLL` per compiler today).

---
_Living document — update in the same PR as the code it describes._
