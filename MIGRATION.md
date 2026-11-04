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
| solve_field    |  —     |   ✓      |   ✗       |   ✓     |   ✗      | ✓   | **yes**    |
| invfield       |  ✓     |   ✓      |   ✓       |   ✓     |   ✓      | ✓   | **yes**    |

All eight CPU-buildable modules (distance, posdef, resize, restrict, splinc, pushpull,
reg_field, reg_flow) are ported through cpu-lib → lib, link into one `libfastfields-cpu.so`,
and pass their CPU test suites (distance 2350 / euclidean+l1, posdef 3580, resize 150,
restrict 28, splinc 4576, distance_spline 510, distance_mesh 1280, pushpull 104,
reg_field 272, reg_flow 282, pushpull_backward 6381). pushpull exposes
pull/push/count/grad **and** pull_backward/push_backward/count_backward/grad_backward
(`hess` remains in the impl, not yet exposed); regularisers expose matvec/diag for
absolute/membrane/bending (kernel/relax/RLS remain in the impl).

`solve_field` is the first slice of the pure-C++ solver umbrella (#34): a
Jacobi-preconditioned conjugate-gradient solve of `(H + L) x = g` for the *field*
family, layered on the existing `field_forward` / `field_diag` / `sym_solve`
operators. It has no kernels-layer code of its own -- cpu-impl contributes only
the two BLAS-1 element loops CG needs (`dot`, `axpby_`), and the driver lives in
cpu-lib next to `field_forward`/`field_precond`, which are composed the same way.
Still open on #34: the flow/"grid" flavour, the V-cycle/FMG driver, the CUDA
backend (needs a device-side reduction for the dot products), and the Python
wiring.

The four backward ops live in their own module (`pushpull_backward.{cpp}`, sharing
`pushpull_dispatch.h` with the forward ones) so that the second copy of the
ndim×order×bound×dtype instantiation matrix is a separate, parallelisable
translation unit: ~7 min at the library's fully-static policy on cpu-lib, ~3m40s
under cuda-lib's default policy. Exposing them turned up four latent bugs, all
inherited from jitfields and all caught by the new finite-difference oracle
(`tests/test_pushpull_backward.cpp`, which differences `<forward_op, ginp>` wrt
every element of `inp` and `grid`):
  * `push_backward` passed `stride_inp` where the kernel indexes `ginp`, so the
    gradient was wrong whenever the pushed volume and the grid had different
    spatial strides (impl layer, both CPU and CUDA);
  * 1D linear `grad_backward` returned a zero gradient wrt `inp`;
  * 2D and 3D linear `grad_backward` returned a zero gradient wrt `grid`, dropping
    the mixed second derivative (only the *pure* ones vanish for a linear basis).

**CUDA branch (integrated; compile-gated):** nvcc (Ubuntu CUDA 12.0) compiles the kernels
and cuda-impl under `__CUDACC__`. Host launchers now exist for **every** module
(distance, posdef, resize, restrict, splinc, reg_field, reg_flow, pushpull). `cuda-lib`
links `libfastfields-cuda.so` with `MODULES = distance reg_field reg_field_rls reg_flow
reg_flow_rls pushpull` (pushpull joined in fastfields-cuda-lib#30: its spline×bound×dim×dtype
matrix used to take ~40 min / risk a ptxas OOM under the old fully-static compile scheme —
fixed via `bound::type::Dynamic` (already used by reg_field/reg_flow, #28) plus a new
`spline::type::Dynamic` one axis further out (kernels/spline.h), each with its own
build-time static/dynamic policy — BOUNDFLAGS / SPLINEFLAGS in cuda-lib's Makefile).
The hub `fastfields-lib` builds an optional `FF_WITH_CUDA` variant (`make USE_CUDA=1`) that
links both `-lfastfields-cpu` and `-lfastfields-cuda`. **No GPU in CI**, so all of the above is
**compile+link only** — runtime CUDA correctness is unvalidated (see the tracked issues).

"✓" for a cpu-lib/lib column means the dispatch layer exists **and is CPU-compiled+tested**.
"✓(c)" for cuda-lib means it compiles+links under nvcc (no runtime validation).
`invfield` replaced the unwired `tetrahedron` scaffolding; see its own
section below.

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
    `ff::cpu::` (`FF_DEVICE`) so it must be `FF_NAMESPACE_BEGIN(FF_NS)/(FF_DEVICE)/(<module>)`
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
bypass, `strides==NULL` handling). `tetrahedron.h`'s rasterization math was on
that list too; both `tetrahedron.h` files are deleted and the op is now
`invfield` (#111) -- see below.

## Boundary conditions: static vs. runtime (`bound::type::Dynamic`)

Boundary conditions are template parameters, so each dispatch layer would
instantiate every kernel once per `(ndim x bound x dtype x offset_t)` — times
`nbatch x nc` again on the CUDA side. With all eight conditions static, nvcc's
`ptxas` needed **~16 GB** on `reg_field.cpp` / `reg_flow.cpp` and was OOM-killed
in CI. (nvcc's top-level `-O` only tunes *host* code, so lowering it did not
help, and splitting the translation unit did not either — the RLS-only half OOMs
on its own.)

`bound::type::Dynamic` — declared in `kernels/bounds.h` from the start but never
implemented — now selects a genuine **runtime** implementation: `bound::dyn<B>`
is an empty, zero-cost forwarder to `bound::utils<B>` for a real `B`, and holds
the condition as a data member (direct `switch`, no function pointers) for
`Dynamic`. Kernels that use it hold `dyn` members constructed from a runtime
`bound::BoundVec` and therefore have **non-static** methods.

Which conditions keep a dedicated static instantiation is a **build-time**
choice, via `BOUNDFLAGS` / `FF_STATIC_BOUNDS` / `FF_STATIC_BOUND_*`; dispatch
layers spell the template argument `FF_BOUND_<NAME>`. Defaults: **cpu-lib** all
eight static (unchanged); **cuda-lib** DCT2 static + the rest Dynamic. Results
are identical either way. Applied to the regularisers only so far —
`resize`/`restrict`/`splinc`/`pushpull` still use the static `bound::utils<B>`.
See fastfields-lib#43.

## The 32-bit index axis (`FF_INDEX32`)

The third build-time axis, and the widest. Every templated kernel below the
dispatch layer is templated on `offset_t`, and `offset_t` has exactly two
values, chosen per call by `canUse32BitIndexMath`: `int32_t` when every
operand's largest element offset fits in 32 bits, `int64_t` otherwise. So the
narrow path costs exactly **x2 instantiations of everything** underneath —
x2 device code, x2 SASS and x2 ptxas memory on the CUDA side. `core/autocast.h`
(`copy_if_needed` / `free_if_needed`) exists solely to feed that second
instantiation with narrowed shape and stride arrays.

`FF_INDEX32=0` (`core/dispatch.h`) makes the narrow arm of every index
dispatch name `int64_t` too: the two arms become one instantiation, the axis
collapses, and the `canUse32BitIndexMath` probe folds away with it. Results are
identical; code size, compile cost and per-voxel speed move.

Which position each backend takes is a **per-library** default —
`INDEXFLAGS` in `src/lib-cpu/Makefile` and `src/lib-cuda/Makefile`, exactly as
`BOUNDFLAGS`/`SPLINEFLAGS` are — so the CPU library can drop the axis while the
CUDA library keeps it, or the reverse. **Both default to on**, i.e. unchanged
behaviour, and CI builds both positions on both backends.

The open question is whether the narrow path is worth its cost. It is an ATen
inheritance (register pressure) and has never been benchmarked in this project,
because there is no GPU in CI. Note the shape of the trade before flipping
anything: the backend that keeps the axis is CUDA, and CUDA is also the one
with no build headroom left (`reg_flow` at 12.98 GB of a 16 GB runner). On the
narrow path CUDA additionally pays a `cudaMallocHost`/`cudaFreeHost` per array
per call, while **363 of the 391** `impl/cuda` upload sites use the
*synchronous* `copyToDevice` (only `distance_{euclidean,l1,mesh}.h` use
`copyToDeviceAsync`) — so for almost every op the pinning has no async copy to
enable, and its cost may partly offset the register-pressure win. Measured
numbers and the exact benchmark that would settle it: #94 and #144.

## `__host__` / `__device__` qualifiers under nvcc (#150)

**nvcc checks a host→device call only when the calling function is not itself
a template.** That single sentence is the whole hazard, and it is not
documented anywhere in the CUDA guide as a limitation.

| caller | callee | nvcc 12.0 |
| --- | --- | --- |
| plain `__host__` function | `__device__` function | **error** |
| plain `__host__` function | `__device__` function *template* | **error** |
| `__host__` function *template* | `__device__` function | **accepted silently** |
| `__host__` function *template* | `__device__` function *template* | **accepted silently** |

In the two bottom rows nvcc does not diagnose anything. `cudafe++` instead
emits, into the **host** object, this in place of the callee's real body (the
real one is kept next to it under `#if 0`):

```c
{int volatile ___ = 1; (void)args; ::exit(___);}
```

So the caller compiles, links, passes `-Wl,--no-undefined` and `ldd -r` — the
damage is intra-TU, nothing becomes undefined — and **terminates the process
with status 1** the first time it runs. `exit` is `noreturn`, so from `-O1`
upwards the host compiler additionally deletes every statement after the call
as unreachable. `-O0` keeps those statements; it is not any less broken, the
process still exits.

Every host caller in this tree is a template, which is why this went unseen:

* `impl/kernels/utils.h`'s `canUse32BitIndexMath` → `typed_prod`, the edge
  behind every `FF_CANUSE32BITS` in `src/lib-cuda`;
* every `FF_CUHOST` launcher in `impl/cuda/{reg_field,reg_flow,
  distance_euclidean,distance_l1,distance_mesh}.h` → `prod(size, n)`, on its
  first line, to size the grid;
* `impl/kernels/distance/mesh.h`'s `FF_CUHOST build_tree` → `max`.

**The rule.** Anything in `impl/kernels/utils.h` is `FF_CUHOSTDEV`, without
exception — the header holds only backend-agnostic helpers over scalars and
raw arrays, and host code calls them. More generally, `FF_CUDEV` means "this
genuinely cannot run on the host" (a device intrinsic, `__shared__`,
`threadIdx`), not "this is called from a kernel". When in doubt, `FF_CUHOSTDEV`
costs nothing: device codegen is identical and the host copy is one inline
function.

**The two gates**, added with the fix:

1. `tests/impl-cuda/compile_probe_hostdev.cu` — calls each helper from a
   plain, non-template `__host__` function, i.e. the shape in the top two rows
   of the table, so a re-qualified helper is an nvcc **error** again. It also
   explicitly instantiates one representative launcher per affected header,
   which type-checks the whole launcher body (an explicit instantiation
   instantiates it in the device pass, where the call edge *is* reported).
   Compiles in ~2 s in `compile-probe-cuda`; against the pre-fix header it
   produces 22 errors.
2. `tools/check-cuda-host-stubs.sh` — run in `build-cuda` over
   `build/obj/lib-cuda/*.o`, fails on any undefined `exit`. Nothing in this
   codebase calls `::exit`, so that symbol has exactly one source: the stub
   above. This is the catch-all for edges the probe does not enumerate.

## `invfield` -- displacement-field inversion (#111)

The last module ported, and the one with no ancestor in jitfields: it comes
from the `brainhops` Python reference
(`src/brainhops/_ext/invfield/{_inv2d,_inv3d}.py`), supplied by the repo owner
as their own code, so there is no licence to carry over -- only the papers,
which the headers cite (Ashburner, Andersson and Friston, NeuroImage 1999 for
two dimensions and Human Brain Mapping 2000, appendix, for three).

Despite the old file name it is a **rasteriser**, not a mesh op. The voxel grid
is a simplicial mesh (each cell cut into two triangles or five tetrahedra), the
nodal displacements make every simplex an affine map, and the field is inverted
by walking the *deformed* simplices and interpolating the *original* corner
coordinates at each integer point they cover. `out` is the result minus the
identity, i.e. the inverse displacement.

It replaces `impl/kernels/tetrahedron.h` and `impl/cpu/tetrahedron.h`, both of
which are **deleted**. They were a different op (a general tetrahedral mesh
with explicit connectivity and per-vertex values, not a grid), neither was
reachable from any `MODULES` list or any `#include`, the cpu-impl one did not
compile at all (no includes, and a bare `namespace ff { namespace tetra {`
instead of the `FF_NAMESPACE_BEGIN` macros), and the kernels one's
`barycoord1` was arithmetically wrong in several places (`v0z - v3y`, a divide
by the quantity just tested against zero). Fixing them would have been a
rewrite of every line, so they are gone rather than renamed.

### Three decisions the reference does not make for us

**The red-black decomposition is kept, but only the colouring.** The reference
splits the grid into shifted subgrids (four in two dimensions, eight in three)
so that batches of grid-aligned simplices can be extracted by slicing, which is
an artefact of writing the loop as array operations. Iterating cells directly
makes the subgrids unnecessary and they are gone. What is *not* an artefact is
the red-black colouring itself: the eight subgrids are exactly the eight parity
classes of the cell index, four "red" and four "black", and the colour chooses
which of two mirror-image five-tetrahedron decompositions a cube uses. That
matters, because the five-tetrahedron split cuts each face of the cube along a
diagonal and two cubes sharing a face must cut it the same way or the mesh is
not conforming. So the parity survives as `(i + j + k) & 1` in
`invfield::stencil`.

It is **not** doing write-conflict avoidance, and it could not: the deformed
simplices of one colour can overlap each other just as readily as two of
different colours, and the reference's own `out[xm, ym, zm] = vsrc` with
duplicate indices is order-undefined within a single batch. #102's colouring
problem is therefore not in play here either way.

**Overlapping simplices elect an owner, lowest index wins.** The reference
overwrites, so its winner is whichever batch ran last -- order-dependent in
NumPy and a genuine race in CUDA. Here the op writes no floating point during
rasterisation at all. The claim pass runs an **atomic minimum of the simplex
index** into a caller-provided `owner` map, and a second pass over *voxels*
reads the winner back and does the interpolation. The election is idempotent,
commutative and associative, so the result does not depend on thread count,
`FF_GRAIN_SIZE` or backend; and because each voxel has exactly one writer in
the second pass, the whole op is race-free rather than merely
atomically-serialised. The float math also drops from once per (simplex, voxel)
pair to once per covered voxel.

`anyAtomicMin` was added to `impl/kernels/atomic.h` for this, on both backends.
It has no non-atomic fallback on purpose: unlike accumulation, a non-atomic
minimum is not a slower implementation of the same thing, it is a race. Note
that it is `__device__`-only on CUDA (`atomicCAS` is), which makes the two
functions that reach it the only `FF_CUDEV` ones in the new kernels header --
see the #150 section above, and the note above `invfield::claim_line`.

**Hole filling stays with the caller.** The reference fills the voxels no
simplex covered by iterating a normalised Gaussian blur (sigma = 1/sqrt(8 ln 2),
a one-voxel FWHM) until every voxel is finite. That is not implemented here and
should not be: it is a policy with its own parameters, it needs a
data-dependent number of passes with a global "any voxel still NaN?" reduction
between them -- host-side control flow that does not belong inside a
dtype-dispatched kernel, and that this library has no reduction primitive for
anyway (the same gap that keeps `field_cg` CPU-only, #34). Instead the kernel
is honest about what it computed: uncovered voxels get a quiet NaN in `out` and
-1 in `owner`, so `owner >= 0` is the coverage mask, and filling is a
composition the caller performs. `owner` is not merely a mask, either -- it
names the winning simplex, which is what makes the result auditable.

### The rasterisation differs from the reference, and is more correct

The reference slices a deformed tetrahedron plane by plane, cuts each slice into
triangles and each triangle into segments, computing every polygon corner by
interpolating along the edges the plane crosses. That divides by an edge's
extent along the slicing axis -- which is **zero whenever a face of the simplex
lies in the slicing plane**. An undeformed grid is nothing but such cases
(every tetrahedron of the cube has three corners in one plane), so the
reference leaves holes in the one field whose inverse is known exactly:
measured, `inverse3d` on a zero field over a 9x10x11 grid leaves 135 of 990
voxels NaN and emits `invalid value encountered in divide`; `inverse2d` over
17x18 leaves 9 of 306.

So the slicing is done differently here. A barycentric coordinate is an
*affine* function of position, so along a line of varying axis-0 coordinate
each one is `a + b*x` and "inside the simplex" is an intersection of `ndim + 1`
half-lines -- an interval, at most one division per coordinate and none when
`b` is zero. The outer axes are bounded by the simplex's own corner extents,
which convexity makes exact. No cross-section polygons, no edge interpolation,
no degenerate division, and the identity field inverts to exactly zero at every
voxel (which `tests/lib-cpu/test_invfield.cpp` asserts, and which is also the
test that the cell stencils really tile their cell -- a wrong stencil leaves a
hole).

### What it was validated against

* `tests/lib-cpu/test_invfield.cpp`, 3,488 checks: closed-form cases (identity,
  translation, random affine -- an affine field's piecewise-linear interpolant
  is the field, so the expected inverse is the analytic one and independent of
  how the cells are cut); a brute-force sweep of every (voxel, simplex) pair
  that re-derives containment with its own Gaussian-elimination barycentric
  solve and checks the owner/coverage/value contract; and layout invariance
  (padded strides, and a batch stride past `INT32_MAX` to reach the `int64_t`
  `offset_t` arm).
* The Python reference, offline: over twenty 2D and 3D cases, the coverage
  masks are **identical** and the values agree to 1.4e-14 (float64). On folded
  fields the masks still agree exactly and the values differ only at
  multiply-covered voxels, which is the documented policy difference (lowest
  index versus last writer).
* Grain-size invariance, measured: `out` and `owner` are **bit-identical**
  between `FF_GRAIN_SIZE=32768` (single-threaded, zero `clone` syscalls) and
  `FF_GRAIN_SIZE=1` with four threads, and the latter is clean under
  ThreadSanitizer.
* CUDA: compile+link only, as everywhere else here. `tests/impl-cuda/
  compile_probe_invfield.cu` instantiates the launcher for all eight
  (ndim, dtype, offset) combinations and takes the address of each of the three
  `__global__` kernels. **Unvalidated on hardware:** that the three launches
  are correctly ordered by the stream, and that the `atomicCAS` loop in
  `anyAtomicMin` behaves as intended on a real device.

### What the Python binding will need

The hub entry point is `ff::invert_field(out, owner, disp, ndim, stream)` with
every output caller-allocated, so the wrapper allocates two tensors: `out`
shaped like `disp` (`(*batch, *spatial, ndim)`, float32 or float64) and `owner`
with one dimension fewer (`(*batch, *spatial)`, **int64**). Neither needs
zeroing. The wrapper then owns the hole filling, and `owner >= 0` is the mask
for it; the natural Python signature is `inverse(disp) -> field` with the mask
kept internal, or `inverse(disp, return_mask=True)` for callers that want to
fill differently. Nothing else in the signature is new: no bound, no spline
order, no `op` flag.

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
### In-place accumulate (`op` = set / add / sub) — restored from jitfields

jitfields templated every regulariser entry point on `char op`
(`Op<'='>` = set, `'+'` = `iadd`, `'-'` = `isub`, see
`jitfields/csrc/lib/regularisers/{field,flow}/utils.h`). All three write through
`out`, so the add/sub forms are **in-place only** at the C level — there was
never a separate "return a fresh tensor" C entry point. In jitfields' Python
layer, `field_matvec_add` clones (`out = inp.clone()`) and then calls the *same*
C function with `op='add'`; `field_addmatvec_` calls it straight on `inp`.

That structure survived the port at the bottom of the stack but was never
surfaced at the top:

| layer | before | after |
| --- | --- | --- |
| kernels | `Op<op>` present | unchanged |
| cpu-impl / cuda-impl | `char op` on matvec/kernel/diag | unchanged |
| cpu-lib / cuda-lib | only `{field,flow}_matvec_{add,sub}` (task #53) | + `diag`/`kernel`, all renamed `_add_`/`_sub_` |
| **lib (hub)** | **nothing** | all 12 entry points |
| bind-py / dlpack | nothing | all 12 |
| numpy / torch / cupy | composed `inp + field_matvec(...)` in Python | call the fused primitive |

`_field_diag` / `_field_kernel` (and the flow twins) hardcoded `'='`; they are now
templated on `op` like `_field_matvec_acc`, and the `DG_DT`/`KN_DT` dispatch
macros gained `ADD_`/`SUB_` variants.

**Naming.** These entry points are **verb-first**, with a trailing
underscore on the in-place spelling (`field_addmatvec_`, `field_subdiag_`, …)
because they are in-place only. This diverges from jitfields' own naming
(`field_matvec_add`/`field_matvec_add_`) but matches the existing `ff::`
convention already used for posdef (`sym_addmatvec_`, `sym_submatvec_`), and
the out-of-place/in-place pairing of `field_precond` / `field_precond_`.
Consistency of the C++ and Python surfaces with each other and with this
codebase's own naming won out over parity with jitfields (explicit repo-owner
decision); this renames the four symbols added by task #53
(`{field,flow}_matvec_{add,sub}` -> `{field,flow}_{add,sub}matvec_`). The
project is unreleased, so there is no compatibility cost, and the rename also
removes a genuine Python-level collision that the earlier `_add_`/`_sub_`-only
spelling still had: `fastfields.{numpy,torch,cupy}.field_matvec_add` would
otherwise be the *out-of-place* spelling under the same name as the in-place
C primitive.

**Python surface** (deliberately diverges from jitfields' naming, everything
else unchanged): `field_addmatvec` is out-of-place, `field_addmatvec_` is
in-place, and both go through the one in-place C primitive — out-of-place
just clones first.

- **T7** de-templating audit + Makefile/CI hardening: confirm which impl entry
  points still template runtime sizes that the migration intends to de-template
  (cross-check jitfields); wire `USE_OPENMP` (currently defined but unused) into
  `CXXFLAGS`; add cuda-lib `-Xcompiler -fPIC`; add a SessionStart/CI job that runs
  the CPU build + tests.
