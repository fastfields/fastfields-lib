#pragma once
#include <fastfields/core/dlpack.h>
#include <cstdint>

namespace ff  {
namespace cuda {

/**
 * @brief Invert a dense displacement field by rasterising the deformed mesh.
 *
 * The voxel grid is read as a simplicial mesh -- each cell cut into two
 * triangles (2D) or five tetrahedra (3D) -- so that the displacements at the
 * grid nodes define a piecewise-affine map, which is inverted exactly simplex
 * by simplex. Only the nodal displacements are used, so a field meant to be
 * interpolated with higher-degree splines is inverted as its piecewise-linear
 * interpolant: the result is approximate between nodes, and more so near the
 * border, where the mesh is clipped by the field of view. See Ashburner,
 * Andersson and Friston, "High-Dimensional Image Registration Using Symmetric
 * Priors", NeuroImage, 1999 (2D) and "Image Registration Using a Symmetric
 * Prior - in Three Dimensions", Human Brain Mapping, 2000 (3D, the appendix).
 *
 * `out` and `owner` are written in full; neither needs to be initialised.
 *
 * Voxels that no deformed simplex covers get a quiet NaN in `out` and -1 in
 * `owner`. That happens near the border for any non-trivial field, and the
 * caller is expected to fill those voxels itself (the reference implementation
 * iterates a normalised one-voxel-FWHM Gaussian blur until none are left).
 * Filling them is deliberately NOT part of this op: it is a policy choice with
 * its own parameters, it needs a data-dependent number of passes with a global
 * reduction between them, and `owner >= 0` is the coverage mask it needs.
 *
 * Where two deformed simplices cover the same voxel -- which a folded field
 * does, and so does any voxel lying on a face two simplices share -- the
 * voxel goes to the one with the LOWEST index, `owner` says which that was,
 * and the choice does not depend on thread count or grain size. The simplex
 * index is `cell * S + k`, with `S` simplices per cell (2 in two dimensions,
 * 5 in three), `cell` indexing the cell grid (shape `spatial[d] - 1`) in
 * row-major order, and `k` the simplex within the cell.
 *
 * @param out     Output inverse displacements (*batch, *spatial, ndim)
 * @param owner   Output simplex index per voxel, -1 where uncovered,
 *                int64 (*batch, *spatial)
 * @param disp    Input displacements in voxels (*batch, *spatial, ndim)
 * @param ndim    Number of spatial dimensions (2 or 3)
 * @param stream  Cuda stream on which to operate
 */
void invert_flow(
          DLTensor & out    ,
          DLTensor & owner  ,
    const DLTensor & disp   ,
          int        ndim   ,
          intptr_t   stream = 0
);

} // namespace cuda
} // namespace ff
