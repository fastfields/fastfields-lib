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
`posdef`, `resize`, `restrict`, `splinc`, `pushpull`, `reg_field`, `reg_flow`.

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
