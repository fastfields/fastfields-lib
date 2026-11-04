# API families

The public API lives in the `ff::` namespace. Every function takes DLPack
tensors (`DLTensor`) and dispatches on device to the CPU or CUDA backend.

!!! note
    This page describes the operations at the **feature** level. Exact
    `DLTensor` argument lists are deliberately omitted — the tensor interface
    will be refactored onto a new tensor library later, so documenting precise
    signatures now would go stale. For the current call shapes, read the module
    headers (`<module>.h`) and `MIGRATION.md`.

Modules are organized one `<module>.{h,cpp}` per operation family: `distance`,
`posdef`, `resize`, `restrict`, `splinc`, `pushpull`, `reg_field`, `reg_flow`,
`solve_field`, `inv_flow`.

## Distance

Distance transforms and point-to-geometry distances:

- **Euclidean distance transform** and **L1 distance transform** over dense
  fields.
- **Point-to-1D-spline distance** — computed by a choice of methods: a
  precomputed **table** lookup, **Brent** minimization, or **Gauss–Newton**
  iteration.
- **Point-to-triangular-mesh distance** — nearest-surface distance from points
  to a triangle mesh.

## Posdef

Operations on fields of small **positive-definite** (compact-symmetric)
matrices. Matrix layouts (full / symmetric / diagonal / and related compact
forms) are selected by a runtime enum. Exposed:

- **matrix–vector product** (and its **backward** pass);
- **in-place add / subtract** matvec (`addmatvec_` / `submatvec_`);
- **linear solve** of the SPD system;
- **inverse**.

## Resampling

Spline resampling and its building blocks:

- **`resample`** — spline resampling of a field (renamed from the impl's
  `resize`).
- **`restriction`** — the adjoint of resampling (prolongation-transpose; renamed
  from `restrict`). It **accumulates into `out`**, so callers pre-zero the
  output.
- **`spline_coeff`** — spline-coefficient **prefiltering** (renamed from
  `splinc`), the step that turns samples into interpolation coefficients.

## Pushpull

The building blocks of image warping and sampling — spline interpolation between
a field and a set of sample locations:

- **pull** — gather / sample a field at given locations (spline interpolation);
- **push** — scatter values back onto a field (the adjoint of pull);
- **count** — accumulate sample weights onto the grid;
- **grad** — spatial gradient of the sampled field.

(Hessian and backward variants exist in the impl layer but are not yet exposed
at this level.)

## Regularisers

Spatial **regularization** operators for multi-channel fields (`reg_field`) and
for vector flows (`reg_flow`). The energies — **absolute**, **membrane**, and
**bending** — are parametrised by per-term weights and voxel size. Exposed:

- **matrix–vector product** (apply the regularizer operator);
- **diagonal** (for use as a preconditioner).

(Kernel / relaxation / RLS variants remain in the impl layer.)

## Flow inversion

**Displacement-field inversion** (`invert_flow`, module `inv_flow`), in two
and three dimensions. The op is named for a *flow* rather than a *field*
because of the distinction `reg_field` and `reg_flow` already draw: a field
has an arbitrary channel count, a flow has `C == ndim`. A displacement
field's last axis is `[x, y]` or `[x, y, z]`, so it is a flow by that
definition.

The voxel grid is read as a simplicial mesh -- each cell cut into two triangles
or five tetrahedra -- so the displacements at the grid nodes define a
piecewise-affine map, and that map is inverted exactly, simplex by simplex: the
deformed simplices are rasterised and the original coordinates are interpolated
at every voxel they cover. A field intended to be interpolated with
higher-degree splines is therefore inverted as its piecewise-linear
interpolant, so the result is approximate between nodes and more so near the
border, where the mesh is clipped by the field of view.

The op writes two tensors, both allocated by the caller and neither needing to
be initialised:

- the **inverse field**, with the same shape as the input;
- a **coverage map**, one `int64` per voxel, holding the index of the simplex
  the voxel's value came from, or -1 if no simplex covered it.

Voxels that no deformed simplex covers -- which happens near the border for any
non-trivial field -- are left as a quiet NaN. Filling them in is the caller's
job, and `owner >= 0` is the mask for it. Where two deformed simplices cover
the same voxel, as a folded field makes them, the lowest-indexed simplex wins;
the choice is deterministic and does not depend on the backend, the thread
count or the grain size.
