# Teenifying fastfields — status report

Consolidated findings from the tool-by-tool deep review (Opus draft → Fable
adversarial revision, 5 tools × 2 passes), plus the 3 correctness bugs the
review surfaced and fixed along the way. Everything below is on
`claude/fastfields-teeny-refactor-js42id` across the fastfields repos unless
noted. See also [`MIGRATION.md`](./MIGRATION.md) for the underlying
per-module status matrix.

An interactive version of this report (with status pills and dark/light
theming) was also published as a Claude artifact during the review session.

## At a glance

- **3** correctness bugs found & fixed (independent of the teeny refactor itself)
- **5** tools deep-reviewed (euclidean+L1, mesh, spline, pushpull, regularisers)
- **~4,900** dead lines verified safe to delete (pushpull)
- **2.3×–2.36×** measured speedup available on the spline eval path

## Bugs found & fixed

None of these are teeny-refactor bugs — they're pre-existing correctness
issues the deep-review process surfaced while reading the regulariser code
closely. All three are fixed, tested, and have open PRs.

### 1. Diag boundary corner cross-term sign error — FIXED

PRs: [fastfields-kernels#32](https://github.com/fastfields/fastfields-kernels/pull/32), [fastfields-cpu-lib#39](https://github.com/fastfields/fastfields-cpu-lib/pull/39)

The boundary-corrected diagonal for `diag_bending` (field) and
`diag_bending`/`diag_all` (flow) computed the corner weight as
`w·(fx0·fy0 + fx1·fy0 + fx1·fy0 + fx1·fy1)` — `fx1·fy0` counted twice,
`fx0·fy1` dropped. The correct expansion of `(fx0+fx1)·(fy0+fy1)` is
`fx0·fy0 + fx1·fy0 + fx0·fy1 + fx1·fy1`, matching what
`matvec_bending`/`matvec_all` already compute.

Fixed in **6 functions, 12 wrong terms** across `field/{2,3}d.h` and
`flow/{2,3}d.h`. The bug cancels exactly at fully-symmetric corners (both
axes flip the same way) and only shows where the two axes' one-sided
boundary signs differ — invisible to the existing interior-only diag tests.
It corrupts the shipped `field_diag`/`flow_diag` public API (the
preconditioner diagonal exposed to numpy/torch/cupy) at boundary voxels
under sign-flipping bounds (DST1/DST2/Zero) whenever bending is active.

Regression tests assert `diag(0,j) == diag(j,0)` on a square domain with a
uniform boundary — confirmed to fail (16 mismatches) against the pre-fix
kernels and pass against the fix.

### 2. matvec_lame / diag_lame unreachable — FIXED

PR: [fastfields-cpu-lib#39](https://github.com/fastfields/fastfields-cpu-lib/pull/39)

`_flow_matvec`/`_flow_diag` in `reg_flow.cpp` routed any non-zero
`shears`/`div` straight to the full combined `matvec_all`/`diag_all`
stencil, even when `bending == 0` — leaving the cheaper Lamé-only
`matvec_lame`/`diag_lame` stencils (already exercised via `relax_lame_`)
unreachable from the public dispatch. `_flow_kernel` and `_flow_relax`
already branched correctly on `bending != 0`; this mirrors that pattern.

This is a **performance** bug, not a correctness one — confirmed by
reverting the fix and finding every existing test value unchanged (the two
stencils must mathematically agree when `bending == 0`). It wastes a
~25-tap stencil per voxel per channel instead of the ~9-tap one for any
elastic-only (non-bending) registration configuration, a common use case.

### 3. reg_field relax_membrane_ type mismatch — FIXED

PR: [fastfields-cpu-impl#34](https://github.com/fastfields/fastfields-cpu-impl/pull/34)

This is very likely **"the broken relax"** — `relax_membrane_` allocated
its scratch buffers as `scalar_t * val = new reduce_t[nc]`, a pointer type
mismatch that is a hard compile error whenever `scalar_t != reduce_t` —
i.e. whenever the caller's dtype is float32, since cpu-lib's `reduce_t` is
always `double`. Every other `relax_*` variant in the file
(`relax_bending_`, the RLS/JRLS family) already used `new scalar_t[nc]`
correctly — an isolated transcription slip.

This proves the whole `reg_field` relax family has never compiled for
float32. It's unwired (no `field_relax` public export exists yet, unlike
`reg_flow`'s relax, which is exercised via `flow_relax`), so it wasn't
caught by CI. Verified via a standalone probe: pre-fix fails to compile for
`float` with exactly the predicted error; post-fix compiles for both
dtypes and a Gauss-Seidel sweep on a trivial diagonal-Hessian system
converges correctly for both.

Wiring `field_relax` into the public API is a separate, larger decision —
not made here.

## Landed this session

| Item | Where | Status |
|---|---|---|
| jitfields-migration merge audit & port | teeny, all fastfields repos | merged |
| reg_flow full teenification (`kernel_*`, `matvec_all`/`diag_all`, and the project's first `relax_*` port) | [fastfields-cpu-impl#33](https://github.com/fastfields/fastfields-cpu-impl/pull/33) | PR open |
| Tetrahedron rasterization tracked (Python reference → per-element kernel plan) | [fastfields-kernels#30](https://github.com/fastfields/fastfields-kernels/issues/30) | low priority, deferred |

## Euclidean + L1 distance — the simplest tool, genuinely shrinks

Separable per-axis distance transforms (Felzenszwalb-style). The per-axis
scan itself is an inherently sequential 1-D recurrence teeny can't remove,
but the surrounding batch/axis bookkeeping — manual stride math for
iterating "every 1-D line along axis d" — is exactly what
`peel`/`peel_front` already do. The plan proposes a `scan_`-shaped teeny
primitive (an in-place sequential fold along one axis, keeping every other
axis batched) as the one missing piece; short of that, the existing peel
API already collapses most of the manual indexing.

## Mesh distance — where mesh_utils-style code should mostly disappear

Point-to-triangle-mesh nearest-distance search. The Opus draft's dead-code
deletion list included `SizedStridedPointer` — Fable caught that it's
actually alive (used by the CUDA treetrace buffer); deleting it would have
broken the CUDA build. Opus's claimed "`nearest_face` bug" was also false
(never exposed to callers). Fable's corrected plan keeps the phase ordering
but fixes these two premises before any code moves.

Real opportunity once corrected: most of the triangle/point vector algebra
(distances, projections, normalization) is a direct match for teeny's
`norm`/`normalize`/`cross`/`dot` (landed on teeny main as part of the
vector-algebra PR, teeny commit `5046b32`). Proposed gaps: `sqdist`/`dist`
convenience wrappers, `maximum_`/`minimum_` in-place, `index_select`, and a
zipped multi-tensor peel for walking triangle vertex arrays together.

## Spline distance — real speedup, real architecture gap

Point-to-1D-spline distance (table / Brent / Gauss-Newton methods). The
live code path is instantiated with `spline_t::Dynamic, bound_t::Dynamic`,
which routes through a **function-pointer interpreter** (13 indirect calls
per cubic eval) rather than the compile-time-unrolled
`vox::pull`/`push`/`grad` kernels teeny already has.

Bridging via `tny::dispatch_value<0..7>` to `vox::pull<1,O,B>` was measured
— real compile+run probes, not estimated — at bit-exact identical output
and a **2.3×–2.36× speedup**. But `vox::*` has no runtime-order dispatch
route (only runtime-*bound*), so this needs either a Dynamic-order escape
hatch in `vox::`/`_make_axis`, or an 8-way static-order dispatch in the
distance layer — a real design decision, not a mechanical port.

Also found: `vox::pull` takes its output **by value** — passing an owning
`tny::local` instead of its `.view()` silently writes into a throwaway copy
with no diagnostic. This footgun broke the Opus draft's own verification
probe (its "verified end-to-end" claim wasn't actually validated as
claimed). Fable's revised plan adds a Phase 0.5 `static_assert` guard
against it before any batch-loop work proceeds, and a Phase 0 of
test-hardening first (spline orders 0/4/6/7, most bounds beyond DCT2, and
Brent/Gauss-Newton all currently lack analytic references — smoke tests
only).

## Pushpull — biggest confirmed win: ~4,900 dead lines

Spline-interpolation gather (pull), scatter (push), count, and spatial
gradient. `gather_sep`/`row_k<K>`/`row_n` (`gather.h`) is deliberately
teeny-free — confirmed correct by two independent Fable passes, not for a
vague performance reason but because boundary-reflected taps are
**non-affine** (a teeny/mdspan view's addressing is always
`base + Σ stride_d·i_d`; boundary reflection maps taps through an arbitrary
per-axis index list no affine mapping can express), and because teeny's
reduction engines are flat (single linear index) and can't express
`gather_sep`'s per-axis hoisting of partial offset/weight computation.

Opus's "CUDA hard blocker" framing (taking a `__device__` function pointer
from host code) was wrong — `CUDEV` is plain `__device__`, so the
function-pointer path is legal and already compiles on device. It's a
severe *performance* hazard, not a compile blocker.

> **Ready now:** ~3,730 + ~1,180 = **~4,910 lines** across
> `pushpull/nd.h`, `2d.h`, `3d.h`, dead blocks in `utils.h`, and 1d.h's
> never-instantiated `Z/L/Q/C` specializations — verified symbol-by-symbol
> as unreachable, unlike the judgment-dependent phases. Fable flagged this
> as safe to execute immediately as its own PR, decoupled from everything
> else.

Proposed teeny gaps: **P1** — static-unrolled `dot`/`zipreduce_`
(confirmed real and valuable, likely mergeable with the regularisers' P2
filing below); **P2** — `slice_n<K>`; P3 downgraded ("mostly already
solved" by existing features); **P4** — host-atomic `fetch_add`, low
priority.

## Regularisers — where the 3 bugs above were found

Absolute/membrane/bending (+ Lamé shears/div for flow) energies:
`matvec_*`/`diag_*`/`kernel_*`/`relax_*` per term. Boundary conditions are
per-tap/per-axis/sign-carrying (`bound::cget(ptr, offset, sign)`), which
forces raw `(loc, size, stride)` kernel signatures that resist a naive
teeny rewrite — this is the densest hand-expanded code in the project
(`field/{1,2,3}d.h` + `flow/{1,2,3}d.h`, ~8,840 lines across `D ∈ {1,2,3}`).

The real opportunity: coloured Gauss-Seidel relax's
`patch1`/`patch2`/`patch3` predicates are confirmed literal affine
sub-lattices (`loc[d] % k == digit_d(n)`) for patch2/patch3,
decomposable-into-sub-lattices for patch1 (parity) — a genuine
teeny-slice-view opportunity using *existing* teeny features (multi-axis
strided slicing), no new primitives required.

Opus's Phase 5 claim that negative-index wrap does the ± window math for a
centred kernel was verified **wrong** — teeny's negative-index wrap is
end-relative, not centre-relative, so it can't express a centred window as
drafted; a corner-based indexing scheme is needed instead. Opus's
"`reg_field` leaks memory" claim was also verified false (every `new` has a
matching `delete[]`) — bug #3 above is the real, stronger finding in its
place.

Proposed teeny gaps: **P1** — `subsample` (axis-pack + starts + step sugar
for the patch-coloring sub-lattices); **P2** — `reduce_`/`zipreduce_` fast
path (likely the same filing as pushpull's P1 — merge before submitting);
**P3** — documentation only.

## Teeny gaps to file

None of these are filed as teeny issues yet — consolidating the 5 reviews'
proposals first, since P1 (static-unrolled reduce/zipreduce) is very likely
one filing shared by pushpull and regularisers, not two.

| Gap | From | Note |
|---|---|---|
| `scan_` — in-place sequential fold along one axis, rest batched | euclidean+L1 | only real gap for this tool |
| static-unrolled `dot`/`zipreduce_` | pushpull P1, regularisers P2 | confirmed real; file once |
| `slice_n<K>` | pushpull P2 | |
| host-atomic `fetch_add` | pushpull P4 | low priority |
| `subsample`(axis-pack, starts, step) | regularisers P1 | sugar only — existing slicing already expresses it |
| `sqdist`/`dist`, `maximum_`/`minimum_`, `index_select`, zipped multi-tensor peel | mesh | |
| Dynamic-order escape in `vox::` / 8-way static dispatch | spline | architecture decision, not a small gap |

## What needs a call

1. **Execute pushpull's Phase A dead-code deletion** (~4,910 lines) —
   verified safe, independent of every judgment call below.
2. **Spline's architecture decision** — Dynamic-order escape hatch in
   `vox::`, or an 8-way static dispatch in the distance layer, to unlock
   the measured 2.3× win. Both are real surface-area changes to teeny or to
   the distance dispatch shape.
3. **Whether to wire `field_relax`** into cpu-lib now (the porting pattern
   is fresh off reg_flow's relax port) or leave it fixed-but-unwired.
4. **File the teeny issues** above, then begin the remaining Fable-revised
   phase plans (mesh's corrected ordering, regularisers' patch-coloring
   slice work).

---
_Generated from the 5-tool Opus-draft → Fable-review pipeline plus a
bug-fixing pass, on `claude/fastfields-teeny-refactor-js42id`._
