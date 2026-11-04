#pragma once
#include <fastfields/core/dlpack.h>
#include <cstdint>
#include <fastfields/core/defines.h>

FF_NAMESPACE_BEGIN(FF_NS)

/**
 * @brief Invert a dense displacement field by rasterising the deformed mesh.
 *
 * The voxel grid is read as a simplicial mesh -- each cell cut into triangles
 * (2D) or five tetrahedra (3D) -- so that the displacements at the grid nodes
 * define a piecewise-affine map, which is inverted exactly simplex by simplex.
 * Only the nodal displacements are used, so a field meant to be interpolated
 * with higher-degree splines is inverted as its piecewise-linear interpolant.
 * See Ashburner, Andersson and Friston, "High-Dimensional Image Registration
 * Using Symmetric Priors", NeuroImage, 1999 (2D) and "Image Registration Using
 * a Symmetric Prior - in Three Dimensions", Human Brain Mapping, 2000 (3D).
 *
 * Both outputs are written in full and need no initialisation. Voxels that no
 * deformed simplex covers get a quiet NaN in `out` and -1 in `owner`; filling
 * them is the caller's job, and `owner >= 0` is the coverage mask for it.
 *
 * Where two deformed simplices cover the same voxel the lowest-indexed one
 * wins, deterministically and independently of thread count. `owner` names
 * the winner as `cell * S + k` (S = 2 or 5 simplices per cell, `cell` the
 * row-major index into the cell grid of shape `spatial[d] - 1`, `k` the
 * simplex within the cell).
 *
 * @param out     Output inverse displacements (*batch, *spatial, ndim)
 * @param owner   Output int64 simplex index per voxel (*batch, *spatial)
 * @param disp    Input displacements in voxels (*batch, *spatial, ndim)
 * @param ndim    Number of spatial dimensions (2 or 3)
 * @param stream  Cuda stream on which to operate
 */
void invert_field(
          DLTensor & out    ,
          DLTensor & owner  ,
    const DLTensor & disp   ,
          int        ndim   = 3,
          intptr_t   stream = 0
);

FF_NAMESPACE_END(FF_NS)
