#pragma once
/* Rasterising one simplex into a dense grid.
 *
 * The single-element math shared by every op that takes a simplicial mesh and
 * writes a dense volume: the barycentric coordinates of a point in a simplex,
 * the walk over the integer points a deformed simplex covers, and the protocol
 * by which two simplices covering the same voxel settle which one owns it.
 *
 * Nothing here knows where the simplex came from or what is being interpolated
 * over it. It takes `ndim + 1` corner positions and nothing else -- no channel
 * count, no attributes, no mesh connectivity -- which is what lets the two ops
 * that need it share it:
 *
 *   `impl/kernels/inv_flow.h`  the voxel grid as an IMPLICIT mesh (each cell
 *                              cut into two triangles or five tetrahedra) with
 *                              IMPLICIT attributes (each corner's attribute is
 *                              its own grid coordinate), so `C == ndim`;
 *
 *   a general tetrahedral-mesh rasteriser (not yet written, fastfields-lib#111
 *                              follow-up) would take an EXPLICIT vertex array
 *                              plus a tetrahedron index array and EXPLICIT
 *                              per-vertex attributes with arbitrary `nc`.
 *
 * The two differ only in their loops and in where they get corners and
 * attributes from. Interpolating the attributes is deliberately NOT here: it
 * is a weighted sum of `ndim + 1` terms, three lines either way, and its
 * *addressing* is what differs -- a stack array for the implicit case, a
 * strided gather for the explicit one. Pushing a channel count down into this
 * header would be the wrong seam.
 *
 * Nothing here allocates, and every function is FF_CUHOSTDEV except the two
 * that write the owner buffer atomically, which cannot run on the host at all
 * (see the note above `claim_line`).
 *
 * ---------------------------------------------------------------------------
 * Two passes, and why rasterising is not a scatter
 * ---------------------------------------------------------------------------
 * Nothing stops two deformed simplices from covering the same voxel: a folded
 * field does exactly that, an explicit mesh may be self-intersecting outright,
 * and so does every voxel that happens to sit on a face two simplices share.
 * Writing the interpolated value straight out would therefore be a write
 * conflict -- order-dependent at best, a data race on a GPU -- so the work
 * splits in two:
 *
 *   `claim`   rasterises one deformed simplex and runs an atomic MINIMUM of
 *             the simplex's index into an `owner` buffer. Only an integer
 *             index is written, so the election is associative, commutative
 *             and idempotent: the winner is the lowest-indexed covering
 *             simplex whatever order the simplices run in, at any grain size,
 *             on either backend.
 *
 *   resolve   runs once per *voxel*, reads that voxel's winning index back,
 *             rebuilds that one simplex, and interpolates. One writer per
 *             output element, so no atomics and no race. This half is the
 *             op's own (`inv_flow::source` is the implicit-mesh version).
 *
 * The float math therefore happens once per covered voxel rather than once
 * per (simplex, voxel) pair, and the owner buffer doubles as the coverage mask
 * a caller needs in order to fill the voxels no simplex reached.
 *
 * ---------------------------------------------------------------------------
 * How a simplex is walked
 * ---------------------------------------------------------------------------
 * A barycentric coordinate is an *affine* function of position, so along a
 * line of varying axis-0 coordinate each one is `a + b*x`, and "inside the
 * simplex" is the intersection of `ndim + 1` half-lines -- an interval,
 * computed by at most one division per coordinate and none at all when `b` is
 * zero. The outer axes are bounded by the simplex's own corner extents, which
 * convexity makes exact.
 *
 * The obvious alternative -- slice the simplex plane by plane, cut each slice
 * into polygons, and interpolate each polygon corner along the edges the plane
 * crosses -- divides by an edge's extent along the slicing axis, which is zero
 * whenever a face of the simplex lies in the slicing plane. An undeformed grid
 * is nothing but such cases, so that formulation leaves holes in the one field
 * whose inverse is known exactly (the identity). See the `inv_flow` section of
 * MIGRATION.md for the measurement.
 */
#include <cstring>      // memcpy -- see `not_a_number`
#include <fastfields/core/cuda_switch.h>
#include "utils.h"
#include "atomic.h"

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(simplex)

/***********************************************************************
 *                     THE OWNER-ELECTION PROTOCOL                     *
 ***********************************************************************/

/* The `owner` value a voxel carries while no simplex has claimed it. The
 * election is an atomic minimum, so it has to compare greater than every real
 * simplex index; the largest int64_t does, and the resolve pass rewrites it as
 * -1 ("no simplex covered this voxel") on the way out. Spelled as a literal
 * rather than INT64_MAX because the fixed-width limit macros are not all
 * available in every compilation mode this header is built in (NVRTC ships no
 * <cstdint>; see core/cuda_switch.h). */
inline FF_CUHOSTDEV int64_t unclaimed()
{
    return 0x7fffffffffffffffLL;
}

/* The `owner` value the resolve pass writes for a voxel no simplex covered.
 * Negative, so `owner >= 0` is the coverage mask. */
inline FF_CUHOSTDEV int64_t uncovered()
{
    return -1;
}

/* A quiet NaN -- what `inv_flow` writes for an uncovered voxel.
 *
 * Float and double only, and that is on purpose rather than for want of
 * writing two more overloads. `inv_flow` can use a NaN because the thing it
 * interpolates is a coordinate, hence floating point; an op whose attributes
 * are, say, integer labels has no NaN to write and must say "uncovered"
 * through the `owner` map alone. Adding an integral overload here would
 * invite it to invent a sentinel instead.
 *
 * Built from its bit pattern because neither spelling that reads naturally is
 * available on both sides of the host/device split: `std::numeric_limits` is
 * host-only and a literal `0.0/0.0` is a constant expression a compiler may
 * reject outright. The punning goes through `memcpy` and not a union, for the
 * reason core/half.h gives at length: reading the inactive member of a union
 * is undefined behaviour in ISO C++, and `memcpy` is the conforming spelling
 * that every compiler folds away. Overloaded on an unused tag argument rather
 * than specialised on a return type, which keeps the call site
 * `not_a_number(scalar_t(0))` deducible under nvcc. */
inline FF_CUHOSTDEV float not_a_number(float)
{
    const uint32_t bits = 0x7fc00000u;
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

inline FF_CUHOSTDEV double not_a_number(double)
{
    const uint64_t bits = 0x7ff8000000000000ULL;
    double value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

/***********************************************************************
 *                        BARYCENTRIC COORDINATES                      *
 ***********************************************************************/

/* Three-vector dot and cross products. Local to this header rather than added
 * to utils.h: they take a raw three-element array, which is the shape the
 * barycentric formula below wants and nothing else in the tree asks for. */
template <typename reduce_t>
inline FF_CUHOSTDEV reduce_t dot3(const reduce_t a[3], const reduce_t b[3])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

template <typename reduce_t>
inline FF_CUHOSTDEV void cross3(
          reduce_t out[3],
    const reduce_t a[3],
    const reduce_t b[3])
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

/* The gradients of a simplex's barycentric coordinates.
 *
 * A barycentric coordinate is affine in position, so each one is fully
 * described by its gradient plus its value at one point, and the natural point
 * is the first corner, where the coordinates are (1, 0, ..., 0). So
 *
 *     bary[i](p) = (i == 0) + grad[i] . (p - v[0])
 *
 * which is all `evaluate` and `interval` below need. Returns a determinant
 * that is zero exactly when the simplex is degenerate -- in which case it has
 * no barycentric coordinates, `grad` is not written, and the caller must skip
 * the simplex. That zero test is the only thing any caller uses the value for,
 * and it is as well that it is: the two overloads do not return the same
 * quantity. In three dimensions it is the signed volume determinant
 * `e1 . (e2 x e3)`, six times the tetrahedron's signed volume; in two it is
 * the Gram determinant of the two edges, which is four times the *squared*
 * area and therefore never negative.
 *
 * Overloaded on the array extent, as `stencil` is. */
template <typename reduce_t>
inline FF_CUHOSTDEV reduce_t gradients(
          reduce_t (&grad)[3][2],
    const reduce_t   v[3][2])
{
    const reduce_t e1[2] = {v[1][0] - v[0][0], v[1][1] - v[0][1]};
    const reduce_t e2[2] = {v[2][0] - v[0][0], v[2][1] - v[0][1]};

    const reduce_t d11 = e1[0]*e1[0] + e1[1]*e1[1];
    const reduce_t d12 = e1[0]*e2[0] + e1[1]*e2[1];
    const reduce_t d22 = e2[0]*e2[0] + e2[1]*e2[1];
    const reduce_t det = d11 * d22 - d12 * d12;
    if (det == static_cast<reduce_t>(0)) return det;

    for (int d = 0; d < 2; ++d) {
        grad[1][d] = (d22 * e1[d] - d12 * e2[d]) / det;
        grad[2][d] = (d11 * e2[d] - d12 * e1[d]) / det;
        grad[0][d] = -(grad[1][d] + grad[2][d]);
    }
    return det;
}

template <typename reduce_t>
inline FF_CUHOSTDEV reduce_t gradients(
          reduce_t (&grad)[4][3],
    const reduce_t   v[4][3])
{
    reduce_t e1[3], e2[3], e3[3];
    for (int d = 0; d < 3; ++d) {
        e1[d] = v[1][d] - v[0][d];
        e2[d] = v[2][d] - v[0][d];
        e3[d] = v[3][d] - v[0][d];
    }

    // The gradient of the i-th coordinate is the opposite face's normal over
    // six times the tetrahedron's volume. Using the cyclic identity
    // a.(b x c) = b.(c x a) = c.(a x b), the three scalar triple products the
    // textbook formula evaluates per point all share the same `p - v[0]`
    // factor, and what multiplies it is exactly these three cross products.
    reduce_t n1[3], n2[3], n3[3];
    cross3(n1, e2, e3);
    cross3(n2, e3, e1);
    cross3(n3, e1, e2);

    const reduce_t det = dot3(e1, n1);
    if (det == static_cast<reduce_t>(0)) return det;

    for (int d = 0; d < 3; ++d) {
        grad[1][d] = n1[d] / det;
        grad[2][d] = n2[d] / det;
        grad[3][d] = n3[d] / det;
        grad[0][d] = -(grad[1][d] + grad[2][d] + grad[3][d]);
    }
    return det;
}

/* The barycentric coordinates of `p`, from the gradients. */
template <int ndim, typename reduce_t>
inline FF_CUHOSTDEV void evaluate(
          reduce_t (&bary)[ndim + 1],
    const reduce_t   p[ndim],
    const reduce_t   origin[ndim],
    const reduce_t   grad[ndim + 1][ndim])
{
    for (int i = 0; i <= ndim; ++i) {
        reduce_t b = (i == 0) ? static_cast<reduce_t>(1)
                              : static_cast<reduce_t>(0);
        for (int d = 0; d < ndim; ++d)
            b += grad[i][d] * (p[d] - origin[d]);
        bary[i] = b;
    }
}

/***********************************************************************
 *                      THE RASTERISATION PRIMITIVES                   *
 ***********************************************************************/

/* Split a row-major linear index over the `ndim` spatial dimensions back into
 * its per-axis coordinates. Row-major (the last axis fastest) and not the
 * Fortran order impl/kernels/batch.h uses, so that consecutive work items walk
 * consecutive elements of a contiguous tensor. */
template <int ndim, typename offset_t>
inline FF_CUHOSTDEV void unravel(
          offset_t   index,
    const offset_t * size,
          offset_t * sub)
{
    for (int d = ndim - 1; d >= 0; --d) {
        sub[d] = index % size[d];
        index /= size[d];
    }
}

/* The integer points of [0, n) that a real interval [lo, hi] contains.
 *
 * Returns false when there are none, which also covers every way the geometry
 * can go wrong: a NaN coordinate (every comparison against it is false, so the
 * very first test rejects it) and an interval that misses the grid entirely.
 * The clamp happens in floating point and *before* the cast, so the cast is
 * always of a value known to lie in [0, n-1]; casting an out-of-range or
 * non-finite double to an integer is undefined behaviour, and a deformed
 * simplex is caller data that can be arbitrarily far from the grid. */
template <typename reduce_t, typename offset_t>
inline FF_CUHOSTDEV bool span(
          reduce_t   lo,
          reduce_t   hi,
          offset_t   n,
          offset_t & first,
          offset_t & last)
{
    const reduce_t top = static_cast<reduce_t>(n - 1);
    if (!(lo <= hi))  return false;   // empty, or either end is NaN
    if (!(hi >= 0))   return false;   // entirely below the grid
    if (!(lo <= top)) return false;   // entirely above it

    const reduce_t a = lo < 0   ? static_cast<reduce_t>(0) : lo;
    const reduce_t b = hi > top ? top                      : hi;
    first = static_cast<offset_t>(ceil(a));
    last  = static_cast<offset_t>(floor(b));
    return first <= last;
}

/* The extent along axis 0, clipped to [0, nx-1], on which the whole line
 * through `point` lies inside the simplex.
 *
 * `point` is the line's position with its axis-0 entry already set to zero, so
 * that evaluating a barycentric coordinate there gives the constant term of
 * `a + b*x`. Each coordinate contributes one half-line; a coordinate whose `b`
 * is zero does not vary along the line at all and either admits every x or
 * none, so it is the one case that needs no division. Returns false when the
 * intersection is empty. */
template <int ndim, typename reduce_t, typename offset_t>
inline FF_CUHOSTDEV bool interval(
          reduce_t & lo,
          reduce_t & hi,
    const reduce_t   point[ndim],
    const reduce_t   origin[ndim],
    const reduce_t   grad[ndim + 1][ndim],
          offset_t   nx)
{
    lo = static_cast<reduce_t>(0);
    hi = static_cast<reduce_t>(nx - 1);
    for (int i = 0; i <= ndim; ++i) {
        reduce_t a = (i == 0) ? static_cast<reduce_t>(1)
                              : static_cast<reduce_t>(0);
        for (int d = 0; d < ndim; ++d)
            a += grad[i][d] * (point[d] - origin[d]);
        const reduce_t b = grad[i][0];
        if (b > 0) {
            const reduce_t t = -a / b;
            if (t > lo) lo = t;
        } else if (b < 0) {
            const reduce_t t = -a / b;
            if (t < hi) hi = t;
        } else if (!(a >= 0)) {
            return false;
        }
    }
    return lo <= hi;
}

/* Claim every integer point of one axis-0 line for `sid`.
 *
 * FF_CUDEV, and NOT FF_CUHOSTDEV like everything above it, because it reaches
 * `anyAtomicMin` -- which on the CUDA side is `__device__`-only, since
 * `atomicCAS` is. FF_CUDEV says "cannot run on the host", which is exactly
 * true of this function in a CUDA build and costs nothing in a CPU build,
 * where the qualifier expands to nothing. Marking it FF_CUHOSTDEV would be the
 * fastfields-lib#150 trap: nvcc does not diagnose a host->device call from a
 * template, it silently replaces the callee's host body with `::exit(1)`.
 * Nothing in a CUDA build calls this from the host -- `kernel_claim` in
 * impl/cuda/inv_flow.h is the only caller, and it is `__global__`. */
template <int ndim, typename reduce_t, typename offset_t>
inline FF_CUDEV void claim_line(
          int64_t  * owner,         // already offset to the line's start
          int64_t    sid,
    const reduce_t   point[ndim],
    const reduce_t   origin[ndim],
    const reduce_t   grad[ndim + 1][ndim],
          offset_t   nx,
          offset_t   sx)
{
    reduce_t lo, hi;
    if (!interval<ndim>(lo, hi, point, origin, grad, nx)) return;
    offset_t x0, x1;
    if (!span(lo, hi, nx, x0, x1)) return;
    for (offset_t x = x0; x <= x1; ++x)
        anyAtomicMin(owner + x * sx, sid);
}

/***********************************************************************
 *                         THE CLAIM PASS                              *
 ***********************************************************************/

/* Rasterise one deformed simplex. Specialised on `ndim` rather than written
 * once with `if (ndim == 2)`, because the two cases differ in the *types* of
 * their vertex arrays and both arms of a runtime branch would have to
 * compile -- and because the loop nest has a different depth.
 *
 * `claim` is FF_CUDEV for the reason given above `claim_line`: it writes the
 * owner buffer atomically, which cannot be done on the host. */
template <int ndim> struct Rasteriser;

template <>
struct Rasteriser<2>
{
    /* Claim every voxel the deformed triangle `vert` covers for `sid`. */
    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV void claim(
              int64_t  * owner,
              int64_t    sid,
        const reduce_t   vert[3][2],
        const offset_t * size,      // [2] spatial shape
        const offset_t * stride)    // [2] spatial strides of `owner`
    {
        reduce_t grad[3][2];
        if (gradients(grad, vert) == static_cast<reduce_t>(0)) return;

        reduce_t ylo = vert[0][1], yhi = vert[0][1];
        for (int i = 1; i < 3; ++i) {
            ylo = min(ylo, vert[i][1]);
            yhi = max(yhi, vert[i][1]);
        }
        offset_t y0, y1;
        if (!span(ylo, yhi, size[1], y0, y1)) return;

        for (offset_t y = y0; y <= y1; ++y) {
            const reduce_t point[2] = {static_cast<reduce_t>(0),
                                       static_cast<reduce_t>(y)};
            claim_line<2>(owner + y * stride[1], sid, point, vert[0], grad,
                          size[0], stride[0]);
        }
    }
};

template <>
struct Rasteriser<3>
{
    /* Claim every voxel the deformed tetrahedron `vert` covers for `sid`.
     *
     * The axis-1 and axis-2 ranges are the simplex's own corner extents, which
     * convexity makes exact bounds: a plane or line outside them cannot meet a
     * convex hull of those corners. They are not tight *per slice*, so some
     * lines are tested and found empty -- a few tens of flops each, against a
     * cell-sized simplex that spans two or three of them. */
    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV void claim(
              int64_t  * owner,
              int64_t    sid,
        const reduce_t   vert[4][3],
        const offset_t * size,      // [3] spatial shape
        const offset_t * stride)    // [3] spatial strides of `owner`
    {
        reduce_t grad[4][3];
        if (gradients(grad, vert) == static_cast<reduce_t>(0)) return;

        reduce_t lo[3], hi[3];
        for (int d = 1; d < 3; ++d) {
            lo[d] = hi[d] = vert[0][d];
            for (int i = 1; i < 4; ++i) {
                lo[d] = min(lo[d], vert[i][d]);
                hi[d] = max(hi[d], vert[i][d]);
            }
        }
        offset_t y0, y1, z0, z1;
        if (!span(lo[1], hi[1], size[1], y0, y1)) return;
        if (!span(lo[2], hi[2], size[2], z0, z1)) return;

        for (offset_t z = z0; z <= z1; ++z) {
            int64_t * plane = owner + z * stride[2];
            for (offset_t y = y0; y <= y1; ++y) {
                const reduce_t point[3] = {static_cast<reduce_t>(0),
                                           static_cast<reduce_t>(y),
                                           static_cast<reduce_t>(z)};
                claim_line<3>(plane + y * stride[1], sid, point, vert[0],
                              grad, size[0], stride[0]);
            }
        }
    }
};


FF_NAMESPACE_END(simplex)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
