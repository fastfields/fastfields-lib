#pragma once
/* The implicit mesh of a displacement flow ("inv_flow").
 *
 * Inverting a dense displacement flow is a rasterisation problem. The voxel
 * grid is read as a simplicial mesh -- every unit cell cut into two triangles
 * or five tetrahedra -- and the displacements at the cell's corners make each
 * simplex an affine map. Walk the *deformed* simplex, and at every integer
 * point it covers read back the point's barycentric coordinates and use them
 * to interpolate the simplex's *original* corner coordinates. That
 * interpolated coordinate is where the inverse flow sends the voxel.
 *
 * Reference: J. Ashburner, J. L. R. Andersson and K. J. Friston,
 * "High-Dimensional Image Registration Using Symmetric Priors", NeuroImage,
 * 1999 (two dimensions),
 * https://www.fil.ion.ucl.ac.uk/spm/doc/papers/john_high_dim.pdf
 * and "Image Registration Using a Symmetric Prior - in Three Dimensions",
 * Human Brain Mapping, 2000 (three dimensions, see the appendix),
 * https://pmc.ncbi.nlm.nih.gov/articles/PMC6871943/pdf/HBM-9-212.pdf
 *
 * What is in THIS header is only what makes the mesh implicit: the cell
 * stencils, and the decode from a simplex index back to the corner positions
 * and attributes that index stands for. The walk itself -- barycentric
 * coordinates, the scan over covered integer points, the owner election --
 * is in `simplex.h`, which knows nothing about grids and is shared with the
 * general explicit-mesh rasteriser that #111's follow-up will add. Read that
 * header first; it explains the two-pass structure this one plugs into.
 *
 * "Implicit attributes" is the other half of what makes this op the special
 * case: there is no attribute array, because each corner's attribute IS its
 * own grid coordinate. So the channel count is pinned to `ndim`, which is why
 * the op is `inv_flow` and not `inv_field` (see "Public API naming" in
 * MIGRATION.md).
 *
 * The loops over the mesh and over the grid live in impl/cpu/inv_flow.h and
 * impl/cuda/inv_flow.h. Nothing here allocates, and every function is
 * FF_CUHOSTDEV.
 */
#include <fastfields/core/cuda_switch.h>
#include "simplex.h"
#include "utils.h"

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(inv_flow)

/***********************************************************************
 *                          THE CELL STENCILS                          *
 ***********************************************************************/

/* How many simplices one grid cell is cut into. Two dimensions: the cell is
 * a square cut along a diagonal. Three: the cube is cut into five tetrahedra
 * -- four corner (trirectangular) ones plus the regular one spanned by the
 * four remaining corners. */
template <int ndim> struct Cell;
template <> struct Cell<2> { static constexpr int simplices = 2; };
template <> struct Cell<3> { static constexpr int simplices = 5; };

/* Which corners of a cell make up its k-th simplex.
 *
 * A corner is returned as a bit code: bit (ndim-1-d) of the code is the
 * offset along spatial axis `d`, so in three dimensions code 5 = 0b101 is the
 * corner at (+1, 0, +1) and in two dimensions code 2 = 0b10 is (+1, 0).
 *
 * `parity` is (i + j + k) & 1 over the cell's own index, which is a red-black
 * colouring of the cell grid. The colouring is NOT a scheduling device and it
 * is not an artefact of the array-at-a-time reference implementation: it
 * chooses *which* of the two mirror-image decompositions the cell uses, and
 * that is what makes the mesh conforming. The five-tetrahedron split of a cube
 * cuts each of the cube's six faces along one diagonal, and two cubes sharing
 * a face must cut it the same way or the mesh has a crack in it. Alternating
 * the decomposition with the parity of the cell index is exactly what
 * guarantees that. In two dimensions the diagonal is interior to the cell and
 * no such constraint exists, but the alternation is kept because it is what
 * the reference does and it keeps the mesh unbiased.
 *
 * Overloaded on the array extent so a caller templated on `ndim` can write
 * `int vtx[ndim+1]; stencil(parity, k, vtx);` and get the right one. */
inline FF_CUHOSTDEV void stencil(int parity, int k, int (&vtx)[3])
{
    // A red square is cut from (0,0) to (+1,+1); a black one along the other
    // diagonal, from (0,+1) to (+1,0).
    if (parity == 0) {
        //   0 --- 1          #1 has its tip at 00, #2 at 11.
        //   |  /  |
        //   2 --- 3
        if (k == 0) { vtx[0] = 0; vtx[1] = 1; vtx[2] = 2; }
        else        { vtx[0] = 3; vtx[1] = 1; vtx[2] = 2; }
    } else {
        //   0 --- 1          #1 has its tip at 01, #2 at 10.
        //   |  \  |
        //   2 --- 3
        if (k == 0) { vtx[0] = 1; vtx[1] = 0; vtx[2] = 3; }
        else        { vtx[0] = 2; vtx[1] = 0; vtx[2] = 3; }
    }
}

inline FF_CUHOSTDEV void stencil(int parity, int k, int (&vtx)[4])
{
    // A red cube puts the tips of its four trirectangular tetrahedra at 000,
    // 011, 101 and 110 -- two opposite corners of one face and two opposite
    // corners of the other -- and the fifth, regular, tetrahedron spans the
    // four corners that are nobody's tip. A black cube is the mirror image:
    // its tips are at the other four corners, 010, 001, 100 and 111.
    if (parity == 0) {
        switch (k) {
            case 0:  vtx[0] = 0; vtx[1] = 1; vtx[2] = 2; vtx[3] = 4; return;
            case 1:  vtx[0] = 3; vtx[1] = 2; vtx[2] = 1; vtx[3] = 7; return;
            case 2:  vtx[0] = 5; vtx[1] = 4; vtx[2] = 1; vtx[3] = 7; return;
            case 3:  vtx[0] = 6; vtx[1] = 4; vtx[2] = 2; vtx[3] = 7; return;
            default: vtx[0] = 7; vtx[1] = 1; vtx[2] = 2; vtx[3] = 4; return;
        }
    } else {
        switch (k) {
            case 0:  vtx[0] = 2; vtx[1] = 3; vtx[2] = 0; vtx[3] = 6; return;
            case 1:  vtx[0] = 1; vtx[1] = 0; vtx[2] = 3; vtx[3] = 5; return;
            case 2:  vtx[0] = 4; vtx[1] = 0; vtx[2] = 6; vtx[3] = 5; return;
            case 3:  vtx[0] = 7; vtx[1] = 3; vtx[2] = 5; vtx[3] = 6; return;
            default: vtx[0] = 0; vtx[1] = 3; vtx[2] = 6; vtx[3] = 5; return;
        }
    }
}

/***********************************************************************
 *                  SIMPLEX INDEX -> CORNERS AND ATTRIBUTES            *
 ***********************************************************************/

/* The corners of simplex `sid`, in the source domain (the grid coordinates
 * themselves) and in the deformed one (plus their displacement).
 *
 * `sid` is `cell * Cell<ndim>::simplices + k`, where `cell` indexes the cell
 * grid (of shape `size[d] - 1`) in row-major order and `k` selects one of the
 * cell's simplices. `disp` points at the start of this batch element, `stride`
 * holds its spatial strides and `stride_comp` the stride of its trailing
 * component axis. */
template <int ndim, typename scalar_t, typename reduce_t, typename offset_t>
inline FF_CUHOSTDEV void corners(
          reduce_t (&vsrc)[ndim + 1][ndim],
          reduce_t (&vdst)[ndim + 1][ndim],
          int64_t    sid,
    const scalar_t * disp,
    const offset_t * size,          // [ndim] spatial shape
    const offset_t * stride,        // [ndim] spatial strides of `disp`
          offset_t   stride_comp)
{
    const int k    = static_cast<int>(sid % Cell<ndim>::simplices);
    int64_t   cell = sid / Cell<ndim>::simplices;

    offset_t base[ndim];
    int      parity = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        const int64_t n = static_cast<int64_t>(size[d]) - 1;
        base[d] = static_cast<offset_t>(cell % n);
        cell /= n;
        parity += static_cast<int>(base[d]);
    }
    parity &= 1;

    int vtx[ndim + 1];
    stencil(parity, k, vtx);

    for (int i = 0; i <= ndim; ++i) {
        offset_t off = 0;
        for (int d = 0; d < ndim; ++d) {
            const offset_t c = base[d]
                + static_cast<offset_t>((vtx[i] >> (ndim - 1 - d)) & 1);
            vsrc[i][d] = static_cast<reduce_t>(c);
            off += c * stride[d];
        }
        for (int d = 0; d < ndim; ++d)
            vdst[i][d] = vsrc[i][d]
                       + static_cast<reduce_t>(disp[off + d * stride_comp]);
    }
}

/* Where the inverse field sends the voxel `vox`, according to simplex `sid`.
 *
 * Returns false when the deformed simplex is degenerate, in which case it has
 * no barycentric coordinates and the voxel must be treated as uncovered. */
template <int ndim, typename scalar_t, typename reduce_t, typename offset_t>
inline FF_CUHOSTDEV bool source(
          reduce_t (&srcp)[ndim],
          int64_t    sid,
    const offset_t   vox[ndim],
    const scalar_t * disp,
    const offset_t * size,          // [ndim] spatial shape
    const offset_t * stride,        // [ndim] spatial strides of `disp`
          offset_t   stride_comp)
{
    reduce_t vsrc[ndim + 1][ndim], vdst[ndim + 1][ndim];
    corners<ndim>(vsrc, vdst, sid, disp, size, stride, stride_comp);

    reduce_t grad[ndim + 1][ndim];
    if (simplex::gradients(grad, vdst) == static_cast<reduce_t>(0))
        return false;

    reduce_t point[ndim];
    for (int d = 0; d < ndim; ++d)
        point[d] = static_cast<reduce_t>(vox[d]);

    reduce_t bary[ndim + 1];
    simplex::evaluate<ndim>(bary, point, vdst[0], grad);

    for (int d = 0; d < ndim; ++d) {
        reduce_t acc = static_cast<reduce_t>(0);
        for (int i = 0; i <= ndim; ++i)
            acc += bary[i] * vsrc[i][d];
        srcp[d] = acc;
    }
    return true;
}


FF_NAMESPACE_END(inv_flow)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
