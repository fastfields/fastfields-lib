# Proposal: what `core/` means, and how things are named

**Status: proposal. Nothing here is a decision.** The prototype on this branch
exists to turn cost estimates into measurements; it is not the migration.

Two questions were asked together because they are the same question twice:
*what is this layer for, and does its name say so?*

Every number below is measured on `85fdac7`, not estimated. The commands are
in [Appendix A](#appendix-a--how-the-numbers-were-measured).

---

## 0. Summary

| | Recommendation |
| --- | --- |
| **`core/` means** | everything more than one layer depends on that is not itself the maths of a named operation — device-specialised or not |
| **Move** | the owner's eight, plus `spline.h`; `bounds.h` and `spline.h` move whole |
| **Do not move** | `vector/` (superseded, unused), `tetrahedron.h` (dead), `splinc.h` (an operation) |
| **Namespaces** | add `vox` *inside* each module: `ff::<device>::<module>::vox` |
| **`atomic.h`** | real bug, not just inconsistency — fix during the move |
| **`_impl` functions** | adopt #147's `flow_slice::` precedent; keep `_x` for locals |
| **Thread API** | it already exists — export it; do **not** mirror it on CUDA |

The single most important finding is not in either question as asked:
**`include/fastfields/impl/cpu/tetrahedron.h` does not compile**, is referenced
by nothing, and is built by nothing. See §1.4.

---

## 1. Question 1 — what moves to `core/`

### 1.1 The definition

The owner has ruled that `core/` means *shared infrastructure, possibly
device-specialised* — the disqualifier is being a kernel implementation, not
being device-aware. That ruling is right, and the tree already assumed it:

- **`core/cuda_switch.h`** exists to branch on `__CUDACC__`, and pulls in
  `<cuda_fp16.h>` under nvcc. `core/` has never been free of device code.
- **`core/autocast.h` already opens `FF_NAMESPACE_BEGIN(FF_DEVICE)`.** So the
  premise that `core/` must not open the device namespace was never true; one
  of the five files there does it today.

So the constraint is narrower than "backend-agnostic". It is:

> A `core/` header must compile under the host compiler *and* nvcc, and must
> mean the right thing in each.

`ff::<FF_DEVICE>::` satisfies that — each translation unit resolves `FF_DEVICE`
to the one backend it is being compiled for, which is exactly why `autocast.h`
works. The proposed definition, which decides every borderline file:

> **`core/` holds every header that more than one layer depends on and that is
> not itself the computation of a named fastfields operation — device-specialised
> or not. `impl/kernels/` keeps only the per-element maths of a named operation.**

"Named operation" means the eight families the library exposes: distance,
posdef, resample, restriction, spline_coeff, pushpull, reg_field, reg_flow.
Boundary conditions and spline weights are not operations; they are the
vocabulary every operation is written in.

### 1.2 The file list

Ten files. The dependency closure is **clean**: every one of them includes only
`core/` headers and other files in this list, so `core/` never acquires a
dependency on `impl/kernels/`. That is checked mechanically, not asserted —
`tools/move-core-headers.py --check` fails if it stops being true.

| File | Lines | Namespace today | Why it moves |
| --- | ---: | --- | --- |
| `utils.h` | 701 | `ff::<dev>` | Numeric/type helpers with zero domain knowledge. **52 include sites, 17 of them in `src/`** — the dispatch layer already reaches into the kernels layer for it. |
| `bounds.h` | 800 | `ff::bound` **and** `ff::<dev>::bound` | `bound::type` is used **209 times across 18 files in `src/`**. Also carries the `FF_STATIC_BOUND_*` / `FF_BOUND_SEL` build-policy macros — the definition of "shared by both dispatch layers". |
| `spline.h` | 1447 | `ff::spline` **and** `ff::<dev>::spline` | Same shape as `bounds.h`: `spline::type` used 23× in `src/`. Not on the owner's list as stated, but it should be — see §1.3. |
| `batch.h` | 273 | `ff::` | Linear-index ↔ sub-index conversion. Used by **12 files in `impl/cpu` and 11 in `impl/cuda`** — genuinely cross-backend. |
| `atomic.h` | 366 | `ff::` / **global** | The accumulate primitive. Device-specialised by construction, which the new definition admits. Carries a real bug — §2.2. |
| `meta.h` | 43 | `ff::meta` | `Pack` / `Tuple` metaprogramming. Pure infrastructure. |
| `parallel.h` | 55 | `ff::` | `parallel_for` + the grain-size policy. |
| `parallel_impl.h` | 235 | `ff::`, `ff::internal` | Its backend selection (native / OpenMP / none). Travels with `parallel.h`. |
| `threadpool.h` | 260 | `ff::` | The work-stealing pool. |
| `threadpool.inl` | 79 | `ff::`, `ff::internal` | Its inline definitions. Travels with `threadpool.h`. |

**4,262 lines.** `core/` goes 5 → 15 files; `impl/kernels/` goes 57 → 47 and
then contains only per-operation implementations, plus the two exceptions in
§1.4.

**The pairs travel as pairs — confirmed, and it is stronger than "convention".**
`parallel.h` includes `parallel_impl.h`, `parallel_impl.h` includes
`threadpool.h`, and `threadpool.h` includes `threadpool.inl`. It is a single
include chain; there is no cut point. `parallel_impl.h`, `threadpool.h` and
`threadpool.inl` have **one include site each** — they are reached only through
`parallel.h`.

### 1.3 `spline.h` versus `splinc.h` — the owner's instinct is right

The owner's list moves `spline` and not `splinc`, and that is correct, but the
reason is worth stating because the names are one character apart and the
distinction is not visible from them:

- **`spline.h` (1447 lines)** is the B-spline *basis*: `weight0..7`,
  `grad0..7`, `hess0..7`, and the `spline::type` enum. It is a shared
  primitive that pushpull, resize and restrict all call. It is not an
  operation. → **moves.**
- **`splinc.h` (339 lines)** is the prefilter *implementation*. It is the
  operation exported as `spline_coeff`. → **stays.**

The one-character gap between the names is the trap. Since a rename is on the
table anyway, the cheapest fix is to make the moved file's new home carry the
distinction: `core/spline.h` (basis) versus `impl/kernels/splinc.h` (operation)
already reads better than the two sitting side by side. If more is wanted,
`splinc.h` → `spline_coeff.h` aligns it with its exported name (`resize.h` and
`restrict.h` have the same mismatch — see §2.4).

**On moving `bounds.h` and `spline.h` whole.** Both files are already two files
glued together: a device-free vocabulary half (`ff::bound`, `ff::spline` — the
`enum class type`, `BoundVecN`, `SplineVecN`) and a device-scoped maths half
(`ff::<dev>::bound`, `ff::<dev>::spline`). The device half already re-exports
the vocabulary:

```cpp
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(bound)
using FF_NS::bound::type;         // bounds.h:203
using FF_NS::bound::transpose;
using FF_NS::bound::BoundVec;
```

So a purist split — vocabulary to `core/`, maths stays — is *available* and the
bridge for it already exists. I recommend **against doing it now**: it turns
two renames git can follow into two deletions and four creations, loses blame
continuity on 2,247 lines, and buys nothing the whole-file move does not. It is
a clean follow-up if the owner ever wants `core/` to be small. Flagging it
because the seam is real and someone will otherwise rediscover it.

### 1.4 What I would **not** move — and two things that should not be there at all

**`vector/` — 11 files, 2,389 lines. Do not move. It is superseded, not
infrastructure.**

The brief asks whether it is infrastructure or implementation. Measured, it is
neither, because **nothing uses it**. Its only includer in the entire tree is
`tests/kernels/vector/test.cpp`, which is not in the gate — it is a hand-run
scratch target that compiles but never runs (`Makefile:50`, "A hand-run scratch
program … compiled (not run)").

It is also not merely unused, it is *duplicated*.
`impl/kernels/distance/mesh_utils.h` (1,158 lines, and in the gate) contains an
independent parallel hierarchy of the same abstraction — `StridedPoint`,
`RefPoint`, `SizedStridedPointer`, `Sized`, `ConstStridedPoint` — against
`vector/`'s `AbstractVector`, `WeakRef`, `AbstractSizedPointer`, `Sized`. Two
implementations of one idea; the used one is the other one.

Moving 2,389 unreferenced lines into the directory whose every edit triggers the
full CI matrix, CUDA included, is the wrong direction. Recommend deciding its
fate separately: either adopt it in `mesh_utils.h` (and *then* it is core
infrastructure) or delete it. 9 of its 11 files also hardcode `namespace ff {`
instead of the `FF_NS` macro, which is its own small argument that it has not
been maintained alongside the rest.

**`tetrahedron.h` — dead, in two divergent copies, one of which does not
compile.**

The brief asks where it sits. The answer is that this is not a placement
question:

- `impl/kernels/tetrahedron.h` (243 lines, `ff::<dev>::tetra`) — **zero**
  references anywhere in `include/`, `src/`, `tests/` or any Makefile.
- `impl/cpu/tetrahedron.h` (3.4 KB, `ff::tetra` — hardcoded, and missing the
  `cpu` level every other file in that directory has) — also zero references,
  **has no `#include` directives at all**, and does not compile:

```
include/fastfields/impl/cpu/tetrahedron.h:42:28:
    error: use of undeclared identifier 'GRAIN_SIZE'
   42 |     parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
```

Neither is compiled by anything, so neither is syntax-checked by CI. They are
the stub for `fastfields-kernels#30` ("Implement the tetrahedron rasterization
(invfield) kernel"). Whoever picks that issue up will start from a file that
does not build and will assume they broke it.

Recommend: leave `impl/kernels/tetrahedron.h` where it is (it *is*
per-element geometry for a future operation, so it is correctly placed), and
either fix or delete `impl/cpu/tetrahedron.h`. Either way, add both to a
compile-only target so the tree stops carrying headers nobody has ever built.
This is independent of the move and can land first.

**`splinc.h`, `resize.h`, `restrict.h`, `distance/`, `posdef/`, `pushpull/`,
`regularisers/`** — all operations. Stay.

### 1.5 Does this need a fourth directory, or `core/cpu` + `core/cuda`?

**No.** The brief asks, and the answer falls out of §1.1: `FF_DEVICE` already
partitions the namespace per compiler, so a device-specialised header in a
single `core/` directory produces the right symbols in each backend without any
directory split. `core/autocast.h` has been doing this the whole time.

A `core/cpu` + `core/cuda` split would also be actively wrong for
`atomic.h`, whose whole job is to present *one* interface
(`ff::anyAtomicAdd`) over two implementations. Splitting the directory would
split the file that exists to not be split.

The one honest wrinkle: `parallel.h` and `threadpool.h` are CPU-only in fact —
their 13 include sites are **all** in `impl/cpu/` (plus one test). Pure
dependency logic puts them in `impl/cpu/`, not `core/`. See §3.3, where the
owner's discoverability argument and Question 4 settle it.

---

## 2. Question 2 — naming

### 2.1 Namespaces: the problem is not "marking voxelwise-ness"

The owner's `vox` idea is right, but for a stronger reason than the one given.
The real defect is that **the kernels layer and the impl layer share a
namespace**. All seven module namespaces are opened by both:

| Namespace | opened in `impl/kernels/` | opened in `impl/{cpu,cuda}/` |
| --- | ---: | ---: |
| `ff::<dev>::pushpull` | 5 files | 2 |
| `ff::<dev>::reg_field` | 4 | 2 |
| `ff::<dev>::reg_flow` | 4 | 2 |
| `ff::<dev>::posdef` | 3 | 2 |
| `ff::<dev>::resize` | 1 | 2 |
| `ff::<dev>::restrict` | 1 | 2 |
| `ff::<dev>::splinc` | 1 | 2 |

**Today this does not collide** — and I want to be precise, because the obvious
grep overstates it. At namespace scope the overlap is **zero**. It is avoided
because the kernels do not expose free functions at all: every kernel entry
point is a static member of a class template — `Kernels` (pushpull, reg_field),
`RegFlow`, `Multiscale` (resize, restrict), `Child` (posdef) — while the impl
drivers are free functions in the same namespace. Members and free functions
cannot collide.

That class wrapper is the workaround. Two pieces of evidence that it is
load-bearing rather than incidental:

1. **`impl/cpu/pushpull.h:37-82`.** Inside the driver `void pull(...)`, there
   is a lambda *also called* `pull` that shadows it, whose body calls the
   kernel `PushPull<...>::pull`:

   ```cpp
   void pull( /* ...the driver... */ ) {
       auto pull = [&](const reduce_t * loc, offset_t o, offset_t i) {
           return PushPull<ndim, IX, BX, IY, BY, IZ, BZ>::pull(...);   // the kernel
       };
       ...
       pull(loc, out_offset, inp_offset);   // the lambda
   }
   ```

   Three different things called `pull` in one scope. Remove the shadowing
   lambda and line 82 recurses into the driver.

2. **The `teeny` branch already hit the collision and already fixed it this
   way.** Commit `b7dbd08`, *"pushpull: nest voxel kernels in
   `ff::cpu::pushpull::vox`"*: **"Avoids a name collision with the impl layer's
   loop drivers, which are also `ff::cpu::pushpull::pull/push/count/grad`."**
   On `teeny` the kernels had been refactored into free functions, at which
   point the shared namespace stopped being survivable.

So: `vox` is not cosmetics. It is what lets a kernel be a plain function.

### 2.2 Recommended spelling: `ff::<device>::<module>::vox`

`vox` goes **inside** the module, not above it.

| Option | Verdict |
| --- | --- |
| `ff::<dev>::<module>::vox` | **Recommended.** Smallest diff; the level where the collision actually is; keeps module cohesion; matches what `teeny` already did, so `main` and `teeny` converge rather than diverge further. |
| `ff::<dev>::vox::<module>` | Groups all kernels under one node and allows one `using`, but separates a kernel from its module and is a larger conceptual change for no extra safety. |
| `ff::vox::<module>` | **Wrong.** Kernels genuinely differ by device (`FF_CUDEV`, the `half` specialisation in `utils.h`), so they must stay device-scoped. |
| Neither | Leaves the class-template workaround permanently load-bearing and leaves `main` diverging from `teeny`. |

**`vox` versus `wise`.** `wise` is an adjective fragment;
`ff::cpu::pushpull::wise::pull` does not read as anything. `vox` reads, is
short at every call site, and has precedent in this codebase.

**The honest caveat, since the owner should get the trade and not a false
verdict:** `vox` is slightly inaccurate. Not everything in this layer is a
voxel — `posdef` operates on one small matrix, `distance/mesh` on one point
against a mesh, `tetrahedron` on a tetrahedron. The project's own prose says so:
`CLAUDE.md` describes the layer as *"single-element math (one voxel, one point,
one small matrix)"* and the impl layer as owning *"the loops over elements"*.
By the codebase's own vocabulary the accurate name is **`elem`**.

I recommend `vox` anyway, on two grounds: the `teeny` precedent exists and
choosing differently means renaming twice, and `vox` matches the project's
name and domain. But if the owner would rather be accurate than convergent,
**this is the only cheap moment to pick `elem`** — after the sweep it is
another 28-site rename. The `teeny` precedent is genuinely thin (one commit,
seven lines, one file), so it should not by itself decide this.

**Cost, measured.** 26 `FF_NAMESPACE_BEGIN(<module>)` sites across 25 files in
`impl/kernels/` — 52 line edits with the matching `END`s. Broken down:
`pushpull` 5, `reg_flow` 4, `reg_field` 4, `posdef` 3, `distance_mesh` 3, and
one each for `resize`, `restrict`, `splinc`, `tetra`, `distance_e`,
`distance_l1`, `distance_spline`. Call sites in the impl layers mostly need
*no* change,
because they reach kernels through the class templates (`Kernels<...>::`,
`Multiscale<...>::`), which are found by ordinary lookup from the enclosing
namespace either way. `posdef` is the exception at 168 qualified uses. This is
a small change that unlocks a later, larger one (turning kernels into free
functions); it does **not** have to be done at the same time as the file move,
and should not be.

### 2.3 `atomic.h` — the concrete test case, and a real bug

The coordinator asked me to confirm the scoping before asserting it. Confirmed,
and it is worse than described.

**CPU half** (`#ifndef __CUDACC__`, lines 13–89): opens `FF_NAMESPACE_BEGIN(FF_NS)`
only → `ff::`. No device level, unlike the rest of the layer. Exposes
`ff::anyAtomicAdd`, `ff::anyAtomicAddNoReturn`, and also `ff::AtomicAdd`,
`ff::has_atomic_add`, `ff::has_fetch_add`.

**CUDA half** (lines 94–364): contains **no namespace macros at all**. Verified
by brace-depth tracking, these sit at **global scope**:

```
  97: struct AtomicFPOp;
 100: struct AtomicFPOp<double>
 119: struct Atomic##NAME##IntegerImpl        (×4, via FF_ATOMIC_INTEGER_IMPL)
 263: static inline FF_CUDEV double atomicAdd(double* address, double val)
 279: static inline FF_CUDEV double gpuAtomicAdd(double*, double)
 283: static inline FF_CUDEV float  gpuAtomicAdd(float*, float)
 341: static inline FF_CUDEV void   gpuAtomicAddNoReturn(double*, double)
 345/347: gpuAtomicAddNoReturn(float*, float)
```

Only the two `anyAtomicAdd` wrappers are namespaced, and they use a hardcoded
`namespace ff {` rather than `FF_NAMESPACE_BEGIN(FF_NS)` — the one place in
`include/` outside `vector/` that does.

Three distinct problems:

1. **Line 263 declares an overload of CUDA's own built-in `atomicAdd` at global
   scope.** It is guarded by `__CUDA_ARCH__ < 600`, and the file's own comment
   admits the hazard: *"defining it for sm_60+ collides with the built-in"*. So
   the collision is understood to be real and is avoided by an arch guard
   rather than by scoping.
2. **`AtomicFPOp` and `gpuAtomicAdd` are ATen's names at ATen's scope**, in an
   *installed public header*. The file says so in its first line ("CUDA portion
   copied from PyTorch/ATen"). `fastfields-torch` is a planned downstream
   binding; a translation unit that includes both this header and PyTorch's
   CUDA atomics is the expected case, not an exotic one.
3. **`FF_GPU_ATOMIC_INTEGER` (line 200) is defined and never used anywhere**,
   and neither it nor `FF_ATOMIC_INTEGER_IMPL` is `#undef`'d, so both leak into
   every CUDA TU downstream. They are `FF_`-prefixed, so they satisfy
   `CLAUDE.md`'s rule — but a dead macro on the installed surface is still dead.

**Recommended fix, and it is the case that tests the scheme.** Under §2.2's
scheme, `atomic.h` becomes `core/atomic.h` with:

```cpp
FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)        // ff::cpu:: or ff::cuda::
FF_NAMESPACE_BEGIN(atomic)
  // both halves: the implementation details, whichever branch is live
FF_NAMESPACE_END(atomic)
  // ff::<dev>::anyAtomicAdd  -- one name, two implementations
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
```

Note what this does and does not change. `anyAtomicAdd` **moves from `ff::` to
`ff::<device>::`**. That is a behavioural improvement (the CPU and CUDA
implementations stop sharing a name at the same scope) but it is a real API
change for its callers — currently `core/bounds.h` and
`tests/kernels/atomic/test.cpp`, i.e. two files. Cheap now; not free.

`#undef FF_ATOMIC_INTEGER_IMPL` after its single use at line 209, and delete
`FF_GPU_ATOMIC_INTEGER`, are both unconditional wins and can land immediately,
independent of everything else here.

### 2.4 Kernel naming — what is actually there

Surveyed before proposing, as asked. The inconsistencies are real but they are
**not random drift**, and that changes what to do about them.

**(a) Case style splits exactly along provenance.** Everything project-original
is `snake_case`. Every `camelCase` identifier is in a vendored or adapted file:

| Identifier | File | Upstream |
| --- | --- | --- |
| `anyAtomicAdd`, `gpuAtomicAdd`, `atomicAddNoReturn` | `atomic.h` | PyTorch/ATen |
| `pushWork`, `requestSteal`, `stealWork`, `threadFunc`, `threadId` | `threadpool.h` | wstpool (MIT) |
| `canUse32BitIndexMath` | `utils.h` | ATen |

Recommend **leaving these spellings alone**. They are the traceability to
upstream, and renaming them makes future diffs against upstream harder for no
gain. Document the rule instead: *snake_case for our code; a vendored file
keeps its upstream spelling*. That converts an apparent inconsistency into a
stated convention, which is cheaper and more honest than a rename.

The one exception worth considering is `canUse32BitIndexMath`, because it is no
longer really theirs: `core/dispatch.h`'s `FF_CANUSE32BITS` macro calls it,
**250 times across 21 files in `src/`**. See §2.5 for a documentation bug attached to it.

**(b) "In-place" is spelled three ways.** This is genuine drift:

| Spelling | Where | Example |
| --- | --- | --- |
| trailing `_` | distance, posdef, public ABI | `add_`, `solve_`, `sym_invert_` |
| `i` prefix | `posdef/utils.h` | `iadd`, `isub`, `imul`, `idiv`, `iaddcmul`, `idivcadd` |
| `to_` infix | `distance/mesh_utils.h` | `addto_`, `multo_`, `crossto_`, `divto_` |

The trailing `_` is dominant and already the public ABI's convention
(`field_precond_`, `sym_solve_`). Recommend it as the rule. **But** `add_` and
`addto_` are not synonyms — `add_(other)` is `this += other`, `addto_(lhs, rhs)`
is `this = lhs + rhs`. That is an under-documented distinction, not a
redundancy, so a blind merge would be a behaviour change. Recommend renaming
only the `i`-prefix family (7 names, one file) and *documenting* the
`add_`/`addto_` pair.

**(c) Internal namespaces are spelled three ways**: `_sign`, `_splinc`,
`_spline` (leading underscore) versus `internal` (`parallel_impl.h`,
`threadpool.inl`) versus `posdef::internal`. Recommend `internal` everywhere —
it is already the majority, it needs no sigil rule, and it matches #147's
choice (§2.5). Five namespace renames.

**(d) Module namespace does not match directory**, in a way that will confuse:

| Directory | Namespace | Public name |
| --- | --- | --- |
| `distance/euclidean.h` | `distance_e` | `dt_euclidean` |
| `distance/l1.h` | `distance_l1` | `dt_l1` |
| `regularisers/field/` | `reg_field` | `field_*` |
| `resize.h` | `resize` | `resample` |
| `restrict.h` | `restrict` | `restriction` |
| `splinc.h` | `splinc` | `spline_coeff` |
| `tetrahedron.h` | `tetra` | — |

Three vocabularies for the same module. Note the constraint from `MIGRATION.md`:
the public names had to differ because *a namespace cannot share a name with a
function inside `ff::cpu`*, which is why `resize`→`resample` happened. That
constraint applies to the **flat exported surface**, not to the kernels layer —
which is precisely why the kernels layer solved the same problem by nesting
instead. Both solutions are defensible; having both, undocumented, is what
costs.

Recommend the low-risk half only: rename the file to match the exported name
where they differ (`resize.h`→`resample.h`, `restrict.h`→`restriction.h`,
`splinc.h`→`spline_coeff.h`, `distance_e`→`distance_euclidean`). Leave the
namespaces alone — renaming those is where the collision risk is, and it buys
less than the file rename does. **This should be a separate PR from the move**;
bundling a rename into a relocation makes both unreviewable.

### 2.5 "impl" functions — the sigil is overloaded, and #147 already fixed it

Measured across the 21 dispatch translation units (`src/lib-cpu`,
`src/lib-cuda`): **116 distinct `_`-prefixed identifiers, 2,943 mentions.**

The important finding is that the sigil means **two different things**:

| Role | Examples | Count |
| --- | --- | --- |
| Narrowed local — the typed/32-bit copy of an ABI argument | `_stride_out` (195), `_stride_inp` (112), `_out` (74), `_stride_wgt` (84) | the large majority |
| Internal dispatch template | `_flow_matvec` (16), `_field_diag` (28), `_flow_kernel` (28), `_sym_matvec` | a few dozen |

The brief notes correctly that this is *legal* — `_x` is reserved only at global
scope, and these are inside `ff::…`. The problem is not legality, it is that
one sigil marks two unrelated things in the same file.

**Recommendation: split the roles, and do not invent a convention — adopt the
one #147 is already landing.**

- **Keep `_x` for the narrowed local.** It is consistent, it is meaningful
  (`_out` is the typed `out`), and it is ~2,800 sites. Renaming it is pure
  churn.
- **Drop it for functions.** Put them in a named nested namespace.

That is exactly what PR #147 does. Its new seam declares
`ff::cuda::flow_slice::matvec_3d`, `diag_3d`, `relax_1d` — a named nested
namespace, hidden visibility, **no underscore**. So the convention is already
being chosen by in-flight work; the recommendation is to generalise it rather
than compete with it. `detail::` or `internal::` for the general case
(`internal` is already in use, per §2.4c).

Note **anonymous namespaces will not do the job here**, though all 21 files
already use them: an anonymous namespace nested inside `ff::cuda` cannot hold a
`flow_matvec` alongside the exported `ff::cuda::flow_matvec` without making
unqualified calls ambiguous. A *named* nested namespace is the mechanism that
works, which is presumably why #147 reached for one.

One small thing to flag to whoever reviews #147: its seam opens `namespace ff {`
/ `namespace cuda {` literally rather than via `FF_NAMESPACE_BEGIN(FF_NS)`.
Defensible in `src/lib-cuda` (where `cuda` really is fixed), but `FF_NS` exists
so the root namespace is spelled in one place.

**A documentation bug found on the way.** `core/dispatch.h:28` and `:48` state
that `canUse32BitIndexMath` comes from `core/autocast.h`. It does not — it is
defined at `impl/kernels/utils.h:668`, and `autocast.h` only mentions it in a
comment. That is why the 17 `src/*.cpp` files that need it also include
`impl/kernels/utils.h` directly. So **`core/` already depends on a symbol that
lives in `impl/kernels/`** — the inversion this proposal repairs is not
hypothetical, it is load-bearing today and mis-documented. Worth fixing the
comment even if nothing moves.

---

## 3. Question 4 (new) — a thread-count API

### 3.1 The CPU API already exists. It is not exported.

This is the headline. `include/fastfields/impl/kernels/threadpool.inl:58-69`
already defines, fully implemented:

```cpp
inline size_t set_num_threads(size_t nthreads);   // ff::
inline size_t get_num_threads();                  // ff::
```

and `parallel_impl.h:61-79` already wraps them in a **backend-abstracted** pair
that also covers the OpenMP and single-threaded builds:

```cpp
ff::get_parallel_threads();   // native pool | omp_get_max_threads() | 1
ff::set_parallel_threads(int);
ff::get_parallel_backend();   // "native" | "omp" | "none"
```

So the task is **export**, not design. The public API should forward to
`set_parallel_threads`, not `set_num_threads`, so that an OpenMP build answers
correctly.

**A name collision to resolve first.** The obvious public name `ff::set_num_threads`
is *already taken* by the internal header-only helper above. Declaring an
exported `ff::set_num_threads` in `api/` while the inline one is in scope is
asking for an ambiguity or an ODR problem. Recommend three distinct names for
three distinct roles:

| Role | Name | Where |
| --- | --- | --- |
| Exported public API | `ff::set_num_threads(int)` / `ff::get_num_threads()` | `api/threads.h` + `src/lib/threads.cpp` |
| Backend-abstracted | `ff::set_parallel_threads` / `get_parallel_threads` | `core/parallel_impl.h` (unchanged) |
| Native-pool internal | `ff::internal::set_pool_size` / `pool_size` | `core/threadpool.inl` (**renamed**) |

**Where it lives:** the hub (`api/` + `src/lib/`), because it is user-facing and
must be an exported symbol in `libfastfields.so`. It is *not* a device dispatch
— thread count is CPU-only — so `src/lib/threads.cpp` forwards straight to
`FF_CPU::` with no `device_type` switch. That is a deliberate asymmetry with
every other file in `src/lib/` and should be commented as such.

### 3.2 Three defects a public setter would expose

**(a) Negative input spawns unbounded threads.** `set_parallel_threads` takes
`int`, `set_num_threads` takes `size_t`, and nothing validates:

```cpp
set_parallel_threads(-1)
  -> set_num_threads(size_t(-1))          // 18446744073709551615
  -> if (nthreads == 0) ...                // false, guard misses
  -> num_threads() = static_cast<int>(...) // -1
  -> new ThreadPool(num_threads())         // int -1 -> size_t count
  -> for (size_t i = 0; i < count; i++) mWorkers.emplace_back(...)
```

Unbounded thread creation until the process dies. Harmless while the function
is internal and every caller passes a sane constant; not harmless the moment it
is public. **A public setter must validate**: reject `n <= 0`, and clamp to some
sane ceiling.

**(b) The setter is unsynchronised.** `internal::num_threads()` (an `int&` to a
function-local static) and `internal::global_pool()` (a `shared_ptr&`) are
mutated with no lock. Concurrent `set_num_threads` and `get_global_pool()` is a
data race on the control block. **#97 just fixed a data race and a lost-wakeup
deadlock in this pool that were reachable only at higher thread counts** — a
setter is precisely the thing that makes those counts reachable on purpose.
Recommend a mutex around the (count, pool) pair.

The *resize* semantics are already safe and worth keeping: `get_global_pool()`
returns `shared_ptr` **by value**, so an in-flight `parallel_for` holds a
reference and the old pool outlives the `reset()`. Document that as the
contract: *"takes effect for parallel regions that begin after it returns;
in-flight regions keep their pool."* That is the same guarantee
`omp_set_num_threads` and `torch.set_num_threads` give, so it needs no
explaining to users.

**(c) The default is `hardware_concurrency() / 2` — deliberate, but a
heuristic.** `threadpool.inl:30-36`:

```cpp
auto num_threads = std::thread::hardware_concurrency();
#if defined(_M_X64) || defined(__x86_64__)
    num_threads /= 2;
#endif
```

The halving is guarded on **x86_64 only**, which makes its intent unambiguous:
it is an SMT/hyper-threading correction (assume 2-way SMT, estimate physical
cores), not a "use half the machine" policy. It is inherited from ATen, and it
is correct in intent — physical cores usually beat logical ones for this kind of
work. It is still a *guess*: it halves wrongly on an x86_64 machine with SMT
disabled (common on cloud VMs pinned one thread per core), and it does nothing
on ARM, which has no SMT and needs no correction.

**Verdict: deliberate, keep it, but document it and let the setter override it.**
Worth noting the CI consequence: `ubuntu-latest` has 4 vCPUs, so the pool
defaults to **2 threads**. That is a thin margin for the `tsan` leg to find
concurrency bugs in — which is consistent with #97's bugs having needed higher
counts to surface.

**Test coverage.** Whatever lands should be exercised by the new
`test-cpu (tsan, grain=1)` leg at **more than one thread count** — the setter's
whole purpose is to reach counts CI otherwise never sees. `tests/kernels/atomic/`
and the `test-atomics` target are the natural place; note the gate is fixed at
59,886 checks / 13 suites, so this must go in a target outside `make test`,
exactly as `test-atomics` already does.

### 3.3 This is also what settles `threadpool` → `core/`

The owner leans `core/` for discoverability and worries the pool would be "lost
in the sea of actual implementations" in `impl/cpu/`. **Pure dependency logic
disagrees**: all 13 `parallel.h` include sites are in `impl/cpu/` (plus one
test), and `threadpool.h` has exactly one includer, `parallel_impl.h`. By
dependency alone, the whole group belongs in `impl/cpu/`.

But dependency logic is answering the wrong question once §3.1 lands. A file
that backs an **exported public API** is not an implementation of an operation
— it is infrastructure, by any reading of §1.1's definition. That converts the
owner's discoverability instinct into a structural argument, which is stronger
than the one they made.

**Recommendation: `core/`, and I agree with the owner — but the reasoning is
conditional.** If the thread-count API is exported, `core/` is correct on the
merits. If Question 4 is declined, `impl/cpu/` is the defensible home and the
argument is only discoverability. Since Question 4 looks likely to land, and
since moving it twice is worse than moving it once, `core/` now is the right
call either way.

### 3.4 CUDA: do not mirror it. But there is a real bug here.

**Allowed?** Yes — threads-per-block is purely a launch parameter, and
`impl/cuda/utils.h:20` already takes it as an argument
(`GET_BLOCKS(N, max_threads_per_block = CUDA_NUM_THREADS)`).

**Recommended as a user-facing knob?** No, and the coordinator's reasoning is
right: CPU thread count is a **resource-sharing** control (how much of a shared
machine you may consume); CUDA threads-per-block is a **work-shaping** control
(occupancy and efficiency, not consumption). Exposing them symmetrically
implies a parallel that does not exist and invites users to "limit GPU usage"
with a knob that cannot do that. Limiting GPU consumption is MPS/MIG/stream
territory.

**But the `CUDA_NUM_THREADS = 1024` question is more serious than the API
question, and it is a latent correctness bug.** Measured:

- `impl/cuda/utils.h:16` — `static constexpr int CUDA_NUM_THREADS = 1024`.
  1024 is the *architectural maximum* any current NVIDIA arch permits.
- **All 39 `<<<...>>>` launch sites in `impl/cuda/` use it.** The
  `max_threads_per_block` parameter of `GET_BLOCKS` is never once overridden —
  it is dead.
- **There is no launch error checking anywhere.** `cudaGetLastError` and
  `cudaPeekAtLastError` appear **zero** times in `include/fastfields/impl/cuda/`
  or `src/lib-cuda/`. Only `cudaMalloc`/`cudaMemcpy` return codes are checked.

Put together: a kernel whose register pressure puts
`cudaFuncGetAttributes().maxThreadsPerBlock` below 1024 fails at launch with
`cudaErrorLaunchOutOfResources`, **and the failure is silently discarded** — the
kernel does not run, the output tensor keeps whatever it held, and the caller
gets no error and no exception. With no GPU in CI, nothing would ever report it.

The regularisers are the obvious candidates: `reg_flow` and `reg_field` are the
12.98 GB and 8.11 GB nvcc compiles precisely because their 3-D bending kernels
are enormous, and register count tracks that.

**Recommended, in priority order — none of this is a user-facing API:**

1. **Add launch error checking.** Cheap, mechanical, and it converts a silent
   wrong answer into a thrown exception. This is worth doing on its own merits
   regardless of everything else in this document, and it is the only item here
   I would call urgent.
2. **Replace the constant with a per-kernel choice** — `cudaOccupancyMaxPotentialBlockSize`,
   or at minimum clamp to that kernel's `cudaFuncGetAttributes().maxThreadsPerBlock`.
   Both faster and immune to the failure by construction.
3. **If a knob is still wanted**, make it an optional *cap* — `ff::cuda::set_max_block_size(int)`,
   defaulting to auto and validated against the per-kernel attribute — never a
   mirror of the CPU thread count.

I have **no GPU to verify any of this on**, so items 1–2 rest on reading the
CUDA contract rather than on a reproduction. Flagging that explicitly: the
"1024 might be too many" claim is a well-founded suspicion, not a measured bug.
The "a failed launch is silently ignored" half **is** verified — it is a grep
result, not a judgement.

---

## 4. Sequencing

```
  #145 header guards ──┐
  #146 <fastfields/…>  ├──► THE MOVE (this proposal, §1)  ──► kernels-as-free-functions
  #143 FF_INDEX32   ───┘         │
  #147 TU split ─── independent ─┘
```

**Must land after:** #145 and #146. Both rewrite includes tree-wide; #146 alone
touches 149 files, 14 of them in `impl/kernels/`. Rebasing a file move under
them by hand is the failure mode the rewrite-script pattern exists to prevent.

**`tools/move-core-headers.py` is written to survive that rebase**, following
the pattern set by `rename-macros.py` and `dedup-dispatch-helpers.py`: it does
not hard-code an include delimiter, it *measures* the tree's dominant one and
matches it. Run before #146 it emits `"fastfields/…"`; run after, `<fastfields/…>`.
Either way `--check` passes on its own output. **Rebase by re-running, never by
hand.**

> **This was demonstrated, not assumed.** #146 and #143 landed on `main` while
> this proposal was being written. The rebase was performed exactly as the
> script's docstring prescribes — revert the prototype, merge `main`, re-run the
> script unedited — and it produced `<fastfields/…>` on the new base with no
> change to the script and no hand-editing:
>
> ```
> $ python3 tools/move-core-headers.py
> moved 10 header(s) to include/fastfields/core;
> rewrote includes in 81 file(s) using <fastfields/...>
>
> $ python3 tools/move-core-headers.py --check
> clean; 10 header(s) in include/fastfields/core,
> include delimiter <...>, no dependency leak
> ```
>
> The merge with #143 was clean: this change touches 17 files under `src/` with
> 23 line changes, all of them include lines, against #143's 19 — the same
> files, different lines, as predicted below.

**Independent of the move (verified, not assumed):**

- **#147** touches `src/lib-cuda/` only — 16 files, none under `include/`. Zero
  overlap. Its `flow_slice::` seam is an *input* to §2.5, not a conflict.
- **#143 (`FF_INDEX32`)** touches `make/`, the Makefiles and the dispatch layer.
  Its only intersection is the ~16 `src/*.cpp` whose `impl/kernels/utils.h`
  include line changes; different lines in the same files.

**CI is neutral.** `.github/workflows/ci.yml:102` already puts
`^include/fastfields/impl/kernels/` and `^include/fastfields/core/` in the *same*
"trigger EVERYTHING" clause, so nothing about the path filters changes.

**Can land independently, in any order, right now:**

| Item | § | Size |
| --- | --- | --- |
| Fix or delete `impl/cpu/tetrahedron.h`; add a compile-only target for orphan headers | 1.4 | small |
| `#undef FF_ATOMIC_INTEGER_IMPL`; delete unused `FF_GPU_ATOMIC_INTEGER` | 2.3 | trivial |
| Fix `core/dispatch.h`'s wrong comment about `canUse32BitIndexMath` | 2.5 | trivial |
| CUDA launch error checking | 3.4 | small, **highest value here** |

**Must land together:** the ten-file move, its include rewrite, the one-line
`tools/test-baseline.sh` probe update (§ below), and the five live prose
references. One commit, one script, one `--check`.

**Should wait:** everything in §2. The `vox` rename, the module/file renames and
the `_impl` convention are each a separate PR *after* the move settles.
Bundling a rename into a relocation makes both unreviewable, and §2.2's rename
is only worth its cost once the free-function refactor it enables is actually
scheduled.

**One thing the move breaks that is easy to miss:** `tools/test-baseline.sh:238`
probes for `include/fastfields/impl/kernels/bounds.h` to verify tree layout, and
dies if it is absent. The gate tool fails *before running any test*, with a
message that looks like a broken checkout. One line. Note that
`tools/consolidate.sh` and `test-baseline.sh:249` reference the same paths and
must **not** be updated — they describe the frozen pre-consolidation layout.
Similarly `MIGRATION.md`'s five references are historical and should be left as
history.

---

## 5. Where I am uncertain

1. **`bounds.h` / `spline.h`: whole-file move versus splitting at the existing
   seam (§1.3).** I recommend whole-file, but this is a judgement about how
   small `core/` should be, not a technical finding. The seam is real and the
   `using` bridge for it already exists, so the split stays cheap later. The
   owner's taste should decide.
2. **`vox` versus `elem` (§2.2).** `vox` is convergent with `teeny`; `elem`
   matches what the codebase's own prose says the layer does. I lean `vox` and
   the margin is thin. This is the trade, not a verdict.
3. **`threadpool` → `core/` (§3.3)** is conditional on the thread API being
   exported. If Question 4 is declined, dependency logic says `impl/cpu/`.
4. **The CUDA 1024 hazard (§3.4)** is reasoned from the CUDA contract, not
   reproduced — there is no GPU here. The *silent* part is verified; the
   *triggering* part is not.
5. **`vector/`'s fate (§1.4)** — I can show it is unused and duplicated, but not
   whether it was meant to replace `mesh_utils.h`'s hierarchy or was abandoned.
   That is history only the owner has.

---

## Appendix A — how the numbers were measured

Everything is reproducible on `85fdac7`.

| Claim | Command |
| --- | --- |
| include sites per file | resolve every `#include` (absolute *and* relative) to a repo path, count edges |
| `bound::type` 209× in `src/` | `grep -ro 'bound::type' src \| wc -l` |
| 39 CUDA launch sites | `grep -rho '<<<' include/fastfields/impl/cuda/*.h \| wc -l` |
| 0 launch error checks | `grep -rn 'cudaGetLastError\|cudaPeekAtLastError' include/fastfields/impl/cuda src/lib-cuda` |
| `vector/` used once | `grep -rn 'kernels/vector' include src tests` |
| `tetrahedron` unreferenced | `grep -rn 'tetrahedron\|tetra' include src tests make Makefile` |
| `impl/cpu/tetrahedron.h` does not compile | `clang++ -std=c++11 -fsyntax-only -I include` on a TU that includes it |
| atomic.h global scope | brace-depth scan of the `#else` branch |
| 116 `_` identifiers / 2,943 mentions | `grep -rhoE '\b_[a-z][a-z0-9_]*\b' src/lib-c* \| sort -u \| wc -l` |
| dependency closure clean | `tools/move-core-headers.py --check` |

**Prototype result.** `tools/move-core-headers.py` applied twice, unedited:

| Base | Delimiter emitted | Files rewritten |
| --- | --- | ---: |
| `85fdac7` (before #146) | `"fastfields/…"` | 81 |
| `de288a9` (after #146 + #143) | `<fastfields/…>` | 81 |

`--check` clean, idempotent and free of dependency leaks on both. The gate
result is recorded in the pull request that carries this document.
