# Migration provenance

On **2026-08-19** the six `fastfields` C++/CUDA repositories were consolidated
into this one. `main` was rewritten: its history is the six source histories
merged, with each repo's files moved to the path prefix it now occupies.

This file records where everything came from, and — more importantly — the two
ways the merged history will actively mislead you if you read it naively.

The rewrite itself was produced by [`tools/consolidate.sh`](tools/consolidate.sh),
which carries the frozen path-rewrite rules and a warning about re-running them.

---

## 1. Which source repo became which path

| Path prefix in this repo | Source repository |
| --- | --- |
| `include/fastfields/api/*.h` | `fastfields-lib` (the hub's public headers) |
| `include/fastfields/api/cpu/` | `fastfields-cpu-lib` (headers) |
| `include/fastfields/api/cuda/` | `fastfields-cuda-lib` (headers) |
| `include/fastfields/impl/cpu/` | `fastfields-cpu-impl` |
| `include/fastfields/impl/cuda/` | `fastfields-cuda-impl` |
| `include/fastfields/impl/kernels/` | `fastfields-kernels` |
| `include/fastfields/core/` | deduplicated — see below |
| `src/lib/` | `fastfields-lib` (`*.cpp`) |
| `src/lib-cpu/` | `fastfields-cpu-lib` (`*.cpp`) |
| `src/lib-cuda/` | `fastfields-cuda-lib` (`*.cpp`) |
| `tests/lib/` | `fastfields-lib` |
| `tests/lib-cpu/` | `fastfields-cpu-lib` |
| `tests/impl-cuda/` | `fastfields-cuda-impl` |
| `tests/kernels/` | `fastfields-kernels` |
| `make/common.mk` | new — the shared makefile fragment the six Makefiles collapsed into |
| `ci/legacy/<repo>/` | each repo's pre-consolidation workflows, kept for reference |
| `docs/legacy/<repo>/` | the five non-hub repos' docs |

### `include/fastfields/core/`

These four headers existed in more than one source repo and were deduplicated
into one copy during the consolidation:

| File | Existed in | Notes |
| --- | --- | --- |
| `dlpack.h` | `fastfields-lib`, `fastfields-cpu-lib`, `fastfields-cuda-lib` | vendored upstream DLPack header; the surviving copy is byte-identical to the hub's |
| `autocast.h` | `fastfields-cpu-lib`, `fastfields-cuda-lib` | 32-bit index narrowing |
| `defines.h` | `fastfields-lib`, `fastfields-kernels` | the two differed (`FF_LIB_DEFINES` vs `FF_DEFINES` guards); the consolidated file is a **merge of both**, identical to neither |
| `cuda_switch.h` | `fastfields-kernels` | backend macros |

`dlpack.h` is vendored third-party code. It is exempt from `codespell`
(see `.codespellrc`) and should not be edited locally.

---

## 2. Commit subjects predating the merge carry the WRONG issue numbers

**This is the thing to know before reading `git log`.**

Every source repo numbered its own issues and pull requests from 1. Those
numbers are baked into the commit subjects that came across, as bare `(#N)`
references:

```
$ git log --format='%s' main | grep -cE '\(#[0-9]+\)'
141          # commits (146 occurrences, 77 distinct numbers)
```

Including the `teeny` branch it is 177 commits / 182 occurrences.

GitHub renders every one of those as a link to **`fastfields-lib#N`** — an
issue or PR in *this* repo. For a commit that came from `fastfields-kernels`,
`fastfields-cpu-lib`, or any of the other four, that link is simply wrong.

> **A mis-link is worse than a broken link.** A dead link is visibly dead and
> someone investigates. `(#23)` resolving to a real, unrelated
> `fastfields-lib` issue of the same number looks correct and nobody notices.
> With 77 distinct numbers in a repo that is itself well past #80, collisions
> are the normal case, not the exception.

**How to resolve a `(#N)` correctly:** find which source repo the commit came
from — its file paths give it away, per the table above — and read `#N` against
*that* repo. The five absorbed repositories are archived, not deleted, so their
issues and PRs remain readable at their original URLs:

* `https://github.com/fastfields/fastfields-cpu-lib/issues/N`
* `https://github.com/fastfields/fastfields-cuda-lib/issues/N`
* `https://github.com/fastfields/fastfields-cpu-impl/issues/N`
* `https://github.com/fastfields/fastfields-cuda-impl/issues/N`
* `https://github.com/fastfields/fastfields-kernels/issues/N`

Commits made *after* the consolidation refer to this repo's numbering, as
normal. The merge commits themselves are the dividing line.

Note the same applies to `#N` references in commit *bodies* (307 occurrences on
`main`), not only in subjects.

---

## 3. Issue transfer: old → new mapping

Transferring an issue between GitHub repositories **renumbers it**. The mapping
below is the record of what moved where.

> **STATUS: NOT YET TRANSFERRED.**
> GitHub exposes issue transfer only through the GraphQL `transferIssue`
> mutation. The session that performed this migration was restricted to a
> pinned set of GraphQL operations and could not call it; the REST API has no
> transfer endpoint at all. The 41 open issues below therefore still live in
> their original repositories and **the five source repos must not be archived
> until they have moved** — archived repositories are read-only, and an issue
> cannot be transferred out of one without unarchiving it first.
>
> Whoever completes the transfer should fill in the "new" column here.

Only **open** issues are being transferred. Closed issues stay where they are,
as history, in the archived repos. **Pull requests cannot be transferred at
all** — every PR in the five absorbed repos stays at its original URL
permanently.

### `fastfields-cpu-lib` (8 open)

| Old | New | Title |
| --- | --- | --- |
| #78 | _pending_ | [teeny] Phase B4+B5: regulariser boundary — reg_field/reg_flow import via tny::from_dlpack |
| #66 | _pending_ | field_relax is not exported at all — 6 relax_* sites unreachable and untested |
| #55 | _pending_ | `make clean` leaves test objects/binaries behind |
| #22 | _pending_ | [perf] pushpull: decide the static-vs-Dynamic boundary split at dispatch |
| #21 | _pending_ | [perf] posdef solve/invert dispatch dynamic-C only |
| #19 | _pending_ | [teeny] guess_type dispatch of the posdef families to all 5 layouts |
| #17 | _pending_ | [teeny] Distance dispatch on teeny + teeny include paths |
| #15 | _pending_ | [teeny] Build cpu-lib at -std=c++17 |

### `fastfields-cuda-lib` (5 open)

| Old | New | Title |
| --- | --- | --- |
| #45 | _pending_ | [teeny] Phase B4+B5-cuda: regulariser boundary |
| #44 | _pending_ | [teeny] Phase B3-cuda: pushpull import via from_dlpack |
| #43 | _pending_ | [teeny] Phase B2-cuda: posdef sym_* import via from_dlpack |
| #42 | _pending_ | [teeny] Phase B1-cuda: resample/restriction/spline_coeff import |
| #41 | _pending_ | [teeny] Phase B0-cuda: dt_l1/dt_euclidean import via from_dlpack |

### `fastfields-cpu-impl` (6 open)

| Old | New | Title |
| --- | --- | --- |
| #65 | _pending_ | [teeny] Phase B5: reg_flow's 25 entry points take teeny carriers |
| #64 | _pending_ | [teeny] Phase B4: reg_field's 29 entry points take teeny carriers |
| #13 | _pending_ | [teeny] Port the splinc (spline prefilter) driver onto teeny peel |
| #11 | _pending_ | [teeny] pushpull impl ports only the 4 exported ops |
| #9 | _pending_ | [teeny] Port the posdef impl onto teeny peel; expose all 5 layouts |
| #7 | _pending_ | [teeny] Port the distance (l1 + euclidean) batch loop onto teeny |

### `fastfields-cuda-impl` (8 open)

| Old | New | Title |
| --- | --- | --- |
| #47 | _pending_ | Mesh distance: write the missing CUHOST launchers for sdt_naive, udt, udt_naive |
| #39 | _pending_ | [teeny] Phase B5-cuda: reg_flow's CUDA launchers take teeny carriers |
| #38 | _pending_ | [teeny] Phase B4-cuda: reg_field's CUDA launchers take teeny carriers |
| #37 | _pending_ | [teeny] Phase B3-cuda: pushpull launchers take teeny carriers |
| #36 | _pending_ | [teeny] Phase B2-cuda: posdef launchers take teeny carriers |
| #35 | _pending_ | [teeny] Phase B1-cuda: resize/restrict/splinc launchers take teeny carriers |
| #34 | _pending_ | [teeny] Phase B0-cuda: distance_{l1,euclidean} dt() launchers take carriers |
| #5 | _pending_ | Port CUDA restrict to the output-driven transpose |

### `fastfields-kernels` (14 open)

| Old | New | Title |
| --- | --- | --- |
| #71 | _pending_ | [teeny] `bound::index_stays_inbounds` is host-only under nvcc |
| #66 | _pending_ | coloured relax (patch1/patch3) is not a valid graph colouring under DFT bounds — data race |
| #62 | _pending_ | [perf] 3-D `matvec_all` (Lamé + bending) is 1.1–1.5x slower on the N-D flow engine |
| #57 | _pending_ | CI: `test-via-cpu-lib` hardcodes `cpu-lib@main` |
| #55 | _pending_ | [teeny] reg #50 phase 1: N-D tap-table engine for the FIELD regulariser |
| #53 | _pending_ | chore: bump the teeny submodule pin to teeny `main` |
| #51 | _pending_ | [teeny] distance/l1.h: rewrite the L1 line kernel on the two-pass scan_ idiom |
| #47 | _pending_ | 3D any-order hess() writes accxz into the zz slot, drops acczz |
| #41 | _pending_ | reg_field/reg_flow: diag_bending omits fold-back term under folding bounds |
| #33 | _pending_ | Spline distance: give vox::pull/push/grad a runtime-order dispatch route |
| #30 | _pending_ | Implement the tetrahedron rasterization (invfield) kernel |
| #14 | _pending_ | host anyAtomicAdd is non-atomic for float/double at C++17 |
| #12 | _pending_ | [teeny] Reimplement the posdef matrix kernels on teeny (all 5 layouts) |
| #10 | _pending_ | [teeny] Register teeny as a submodule (external/teeny) |

Note the numbering collisions this migration has to survive: `#55`, `#66` and
`#47` each name *two different open issues* in two different source repos, and
several more collide with existing `fastfields-lib` numbers. This is exactly
the hazard section 2 describes.

---

## 4. The `claude-fastfields-to-teeny` workstream targets `teeny`, not `main`

The teeny refactor lives on the **`teeny`** branch. It is 159 commits ahead of
`main` and carries `external/teeny` (a submodule `main` does not have).

**Every PR in the `claude-fastfields-to-teeny` workstream must be opened
against `teeny`.** Most of the issues listed in section 3 belong to it — they
carry the `claude-fastfields-to-teeny` label.

This is not a stylistic preference. Three of that workstream's recent PRs were
mis-filed against `main` and had to be re-targeted. The rule predates the
consolidation and has to survive it, which is why it is written down here and
in [`CLAUDE.md`](CLAUDE.md) rather than living in one workstream's memory.

`main` and `teeny` share a merge base (`main`'s pre-CI tip) and `teeny` is not
behind on content, so re-targeting a mis-filed PR is usually mechanical — but
only if it is caught before review.
