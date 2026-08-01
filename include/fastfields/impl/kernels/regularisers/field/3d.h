#ifndef FF_REGULARISERS_FIELD_3D
#define FF_REGULARISERS_FIELD_3D
#include "../../cuda_switch.h"
#include "../../bounds.h"
#include "../../utils.h"
#include "utils.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_field)

//----------------------------------------------------------------------
//          low-level kernels for anything regularization
//----------------------------------------------------------------------

template <int _C, class... T>
struct Kernels<Config<three, _C, T...>>
{
    using _Config = Config<one, _C, T...>;
    using scalar_t = typename _Config::scalar_t;
    using reduce_t = typename _Config::reduce_t;
    using offset_t = typename _Config::offset_t;
    static constexpr offset_t D   = static_cast<offset_t>(three);
    static constexpr offset_t C   = static_cast<offset_t>(_C);
    static constexpr bound_t  BX  = _Config::Bound::template At<0>::Value;
    static constexpr bound_t  BY  = _Config::Bound::template At<1>::Value;
    static constexpr bound_t  BZ  = _Config::Bound::template At<2>::Value;
    // Boundary helpers. `bound::dyn<B>` is an empty, zero-cost forwarder when B
    // is a real condition, and carries the condition as a data member when B is
    // `bound::type::Dynamic` (single instantiation, runtime dispatch).
    bound::dyn<BX> bound_utils_x;
    bound::dyn<BY> bound_utils_y;
    bound::dyn<BZ> bound_utils_z;

    inline CUDEV Kernels() {}

    // Runtime boundary conditions; ignored by statically instantiated axes.
    explicit inline CUDEV Kernels(const ::FF::bound::BoundVec & bnd)
        : bound_utils_x(bnd[0])
        , bound_utils_y(bnd[1])
        , bound_utils_z(bnd[2]) {}
    typedef scalar_t & (*OpType)(scalar_t &, const reduce_t &);

    //------------------------------------------------------------------
    //                            ABSOLUTE
    //------------------------------------------------------------------

    static const offset_t kernelsize_absolute = C;

    CUDEV inline offset_t
    get_kernelsize_absolute(offset_t nc = C)
    { return C < 0 ? nc : C; }

    /// kernel <- [abs, ...]
    CUDEV inline void
    make_kernel_absolute(
              reduce_t kernel   [],
        const reduce_t absolute [],
              offset_t nc       = C
    )
    {
#       pragma unroll
        for (int c = 0; c < (C < 0 ? nc : C); ++c)
            kernel[c] = absolute[c];
    }

    // --- matvec ---

    template <OpType op = set>
    CUDEV inline void
    matvec_absolute(
              scalar_t out      [],
        const scalar_t inp      [],
              offset_t osc,
              offset_t isc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
#       pragma unroll
        for (offset_t c = 0; c < (C < 0 ? nc : C); ++c)
            op(out[osc*c], kernel[c] * inp[isc*c]);
    }

    // --- kernel ---

    template <OpType op = set>
    CUDEV inline  void
    kernel_absolute(
              scalar_t out      [],
              offset_t osc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
#       pragma unroll
        for (offset_t c = 0; c < (C < 0 ? nc : C); ++c)
            op(out[osc*c], kernel[c]);
    }

    // --- diagonal ---

    template <OpType op = set>
    CUDEV inline  void
    diag_absolute(
              scalar_t out      [],
              offset_t osc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        return kernel_absolute<op>(out, osc, kernel, nc);
    }

    //------------------------------------------------------------------
    //                            MEMBRANE
    //------------------------------------------------------------------

    static const offset_t kernelsize_membrane = (D+1)*C;

    CUDEV inline offset_t
    get_kernelsize_membrane(offset_t nc = C)
    { return (D+1) * (C < 0 ? nc : C); }

    /// kernel <- [abs, w100, w010, w001, ...]
    CUDEV inline void
    make_kernel_membrane(
              reduce_t kernel       [],
        const reduce_t absolute     [],
        const reduce_t membrane     [],
        const reduce_t voxel_size   [D],
              offset_t nc           = C
    )
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1], vz = voxel_size[2];
        vx = 1./(vx*vx); vy = 1./(vy*vy); vz = 1./(vz*vz);
        for (int c = 0; c < (C < 0 ? nc : C); ++c, kernel+=4)
        {
            reduce_t m = membrane[c];
            kernel[0] = absolute[c];
            kernel[1] = -m * vx;
            kernel[2] = -m * vy;
            kernel[3] = -m * vz;
        }
    }

    /// kernel <- [w00, w100, w010, w001, ...]
    CUDEV inline void
    make_fullkernel_membrane(
              reduce_t kernel       [],
        const reduce_t absolute     [],
        const reduce_t membrane     [],
        const reduce_t voxel_size   [D],
              offset_t nc           = C
    )
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1], vz = voxel_size[2];
        vx = 1./(vx*vx); vy = 1./(vy*vy); vz = 1./(vz*vz);
        for (int c = 0; c < (C < 0 ? nc : C); ++c, kernel+=4)
        {
            reduce_t m = membrane[c];
            kernel[0] = absolute[c] + 2 * m * (vx + vy + vz);
            kernel[1] = -m * vx;
            kernel[2] = -m * vy;
            kernel[3] = -m * vz;
        }
    }

    // --- matvec ---

    template <OpType op = set>
    CUDEV inline void
    matvec_membrane(
              scalar_t out      [],
        const scalar_t inp      [],
        const offset_t loc      [D],
        const offset_t size     [D],
        const offset_t stride   [D],
              offset_t osc,
              offset_t isc,
        const reduce_t kernel   [],
              offset_t nc           = C
    )
    {
        offset_t  x = loc[0],     y = loc[1],     z = loc[2];
        offset_t nx = size[0],   ny = size[1],   nz = size[2];
        offset_t sx = stride[0], sy = stride[1], sz = stride[2];

        offset_t x0 = x-1, x1 = x+1, y0 = y-1, y1 = y+1, z0 = z-1, z1 = z+1;
        int8_t fx0 = bound_utils_x.sign(x0, nx);
        int8_t fx1 = bound_utils_x.sign(x1, nx);
        int8_t fy0 = bound_utils_y.sign(y0, ny);
        int8_t fy1 = bound_utils_y.sign(y1, ny);
        int8_t fz0 = bound_utils_z.sign(z0, nz);
        int8_t fz1 = bound_utils_z.sign(z1, nz);
        x0 = (bound_utils_x.index(x0, nx) - x) * sx;
        x1 = (bound_utils_x.index(x1, nx) - x) * sx;
        y0 = (bound_utils_y.index(y0, ny) - y) * sy;
        y1 = (bound_utils_y.index(y1, ny) - y) * sy;
        z0 = (bound_utils_z.index(z0, nz) - z) * sz;
        z1 = (bound_utils_z.index(z1, nz) - z) * sz;

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t center = static_cast<reduce_t>(inp[0]);
            auto get = [&](offset_t o, int8_t f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out, kernel[0] * center +
                     kernel[1] * (get(x0, fx0) + get(x1, fx1)) +
                     kernel[2] * (get(y0, fy0) + get(y1, fy1)) +
                     kernel[3] * (get(z0, fz0) + get(z1, fz1)));
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            conv(out + osc*c, inp + isc*c, kernel + 4*c);
    }

    // --- kernel ---

    template <OpType op = set>
    CUDEV inline void
    kernel_membrane(
              scalar_t out      [],
              offset_t sc,
        const offset_t stride   [3],
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        offset_t sx = stride[0], sy = stride[1], sz = stride[2];

        auto setkernel = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0], w100 = kernel[1],
                     w010 = kernel[2], w001 = kernel[3];
            op(out[0],   w000);
            op(out[-sx], w100);
            op(out[+sx], w100);
            op(out[-sy], w010);
            op(out[+sy], w010);
            op(out[-sz], w001);
            op(out[+sz], w001);
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            setkernel(out + sc*c, kernel + 4*c);
    }

    // --- diagonal ---

    template <OpType op = set>
    CUDEV inline void
    diag_membrane(
              scalar_t out      [],
              offset_t osc,
        const offset_t loc      [D],
        const offset_t size     [D],
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        offset_t  x = loc[0],   y = loc[1],   z = loc[2];
        offset_t nx = size[0], ny = size[1], nz = size[2];

        int8_t fx = bound_utils_x.sign(x-1, nx)
                       + bound_utils_x.sign(x+1, nx);
        int8_t fy = bound_utils_y.sign(y-1, ny)
                       + bound_utils_y.sign(y+1, ny);
        int8_t fz = bound_utils_z.sign(z-1, nz)
                       + bound_utils_z.sign(z+1, nz);

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c, kernel += 4)
            op(out[osc*c], kernel[0] - kernel[1]*fx - kernel[2]*fy - kernel[3]*fz);
    }

    //------------------------------------------------------------------
    //                            BENDING
    //------------------------------------------------------------------

    static const offset_t kernelsize_bending = 10*C;

    CUDEV inline offset_t
    get_kernelsize_bending(offset_t nc = C)
    { return 10 * (C < 0 ? nc : C); }

    /// kernel <- [
    ///     abs, w100, w010, w001, w200, w020, w002, w110, w101, w011, ...]
    CUDEV inline void
    make_kernel_bending(
              reduce_t kernel       [],
        const reduce_t absolute     [],
        const reduce_t membrane     [],
        const reduce_t bending      [],
        const reduce_t voxel_size   [D],
              offset_t nc           = C
    )
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1], vz = voxel_size[2];
        vx = 1./(vx*vx); vy = 1./(vy*vy); vz = 1./(vz*vz);
        for (int c=0; c<(C < 0 ? nc : C); ++c, kernel+=10)
        {
            reduce_t m = membrane[c], b = bending[c];
            kernel[0] = absolute[c];
            kernel[1] = -4 * b * vx * (vx + vy + vz) - m * vx;
            kernel[2] = -4 * b * vy * (vx + vy + vz) - m * vy;
            kernel[3] = -4 * b * vz * (vx + vy + vz) - m * vz;
            kernel[4] = b * vx * vx;
            kernel[5] = b * vy * vy;
            kernel[6] = b * vz * vz;
            kernel[7] = 2 * b * vx * vy;
            kernel[8] = 2 * b * vx * vz;
            kernel[9] = 2 * b * vy * vz;
        }
    }

    /// kernel <- [
    ///     w000, w100, w010, w001, w200, w020, w002, w110, w101, w011, ...]
    CUDEV inline void
    make_fullkernel_bending(
              reduce_t kernel       [],
        const reduce_t absolute     [],
        const reduce_t membrane     [],
        const reduce_t bending      [],
        const reduce_t voxel_size   [D],
              offset_t nc           = C
    )
    {
        reduce_t vx = voxel_size[0], vy = voxel_size[1], vz = voxel_size[2];
        vx = 1./(vx*vx); vy = 1./(vy*vy); vz = 1./(vz*vz);
        for (int c=0; c<(C < 0 ? nc : C); ++c, kernel+=10)
        {
            reduce_t m = membrane[c], b = bending[c];
            kernel[1] = -4 * b * vx * (vx + vy + vz) - m * vx;
            kernel[2] = -4 * b * vy * (vx + vy + vz) - m * vy;
            kernel[3] = -4 * b * vz * (vx + vy + vz) - m * vz;
            kernel[4] = b * vx * vx;
            kernel[5] = b * vy * vy;
            kernel[6] = b * vz * vz;
            kernel[7] = 2 * b * vx * vy;
            kernel[8] = 2 * b * vx * vz;
            kernel[9] = 2 * b * vy * vz;
            kernel[0] = absolute[c]
                      - 2 * (kernel[1] + kernel[2] + kernel[3] +
                             kernel[4] + kernel[5] + kernel[6])
                      - 4 * (kernel[7] + kernel[8] + kernel[9]);
        }
    }

    // --- matvec ---

    template <OpType op = set>
    CUDEV inline void
    matvec_bending(
              scalar_t out      [],
        const scalar_t inp      [],
        const offset_t loc      [D],
        const offset_t size     [D],
        const offset_t stride   [D],
              offset_t osc,
              offset_t isc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        offset_t  x = loc[0],     y = loc[1],     z = loc[2];
        offset_t nx = size[0],   ny = size[1],   nz = size[2];
        offset_t sx = stride[0], sy = stride[1], sz = stride[2];

        int8_t   fx00 = bound_utils_x.sign(x-2, nx);
        int8_t   fx0  = bound_utils_x.sign(x-1, nx);
        int8_t   fx1  = bound_utils_x.sign(x+1, nx);
        int8_t   fx11 = bound_utils_x.sign(x+2, nx);
        int8_t   fy00 = bound_utils_y.sign(y-2, ny);
        int8_t   fy0  = bound_utils_y.sign(y-1, ny);
        int8_t   fy1  = bound_utils_y.sign(y+1, ny);
        int8_t   fy11 = bound_utils_y.sign(y+2, ny);
        int8_t   fz00 = bound_utils_z.sign(z-2, nz);
        int8_t   fz0  = bound_utils_z.sign(z-1, nz);
        int8_t   fz1  = bound_utils_z.sign(z+1, nz);
        int8_t   fz11 = bound_utils_z.sign(z+2, nz);
        offset_t x00 = (bound_utils_x.index(x-2, nx) - x) * sx;
        offset_t x0  = (bound_utils_x.index(x-1, nx) - x) * sx;
        offset_t x1  = (bound_utils_x.index(x+1, nx) - x) * sx;
        offset_t x11 = (bound_utils_x.index(x+2, nx) - x) * sx;
        offset_t y00 = (bound_utils_y.index(y-2, ny) - y) * sy;
        offset_t y0  = (bound_utils_y.index(y-1, ny) - y) * sy;
        offset_t y1  = (bound_utils_y.index(y+1, ny) - y) * sy;
        offset_t y11 = (bound_utils_y.index(y+2, ny) - y) * sy;
        offset_t z00 = (bound_utils_z.index(z-2, nz) - z) * sz;
        offset_t z0  = (bound_utils_z.index(z-1, nz) - z) * sz;
        offset_t z1  = (bound_utils_z.index(z+1, nz) - z) * sz;
        offset_t z11 = (bound_utils_z.index(z+2, nz) - z) * sz;

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t w000 = kernel[0],
                     w100 = kernel[1], w010 = kernel[2], w001 = kernel[3],
                     w200 = kernel[4], w020 = kernel[5], w002 = kernel[6],
                     w110 = kernel[7], w101 = kernel[8], w011 = kernel[9];

            reduce_t center = static_cast<reduce_t>(inp[0]);
            auto get = [&](offset_t o, int8_t f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out,
                  w000 * center
                + w100 * (get(x0, fx0) + get(x1, fx1))
                + w010 * (get(y0, fy0) + get(y1, fy1))
                + w001 * (get(z0, fz0) + get(z1, fz1))
                + w200 * (get(x00, fx00) + get(x11, fx11))
                + w020 * (get(y00, fy00) + get(y11, fy11))
                + w002 * (get(z00, fz00) + get(z11, fz11))
                + w110 * (get(x0+y0, fx0*fy0) + get(x1+y0, fx1*fy0) +
                          get(x0+y1, fx0*fy1) + get(x1+y1, fx1*fy1))
                + w101 * (get(x0+z0, fx0*fz0) + get(x1+z0, fx1*fz0) +
                          get(x0+z1, fx0*fz1) + get(x1+z1, fx1*fz1))
                + w011 * (get(y0+z0, fy0*fz0) + get(y1+z0, fy1*fz0) +
                          get(y0+z1, fy0*fz1) + get(y1+z1, fy1*fz1))
            );
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            conv(out + osc*c, inp + isc*c, kernel + 10*c);
    }

    // --- kernel ---

    template <OpType op = set>
    CUDEV inline void
     kernel_bending(
              scalar_t out      [],
              offset_t sc,
        const offset_t stride   [D],
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        offset_t sx = stride[0], sy = stride[1], sz = stride[2];

        auto setkernel = [&](scalar_t * o, const reduce_t * ker) {
            reduce_t w000 = ker[0],
                     w100 = ker[1], w010 = ker[2], w001 = ker[3],
                     w200 = ker[4], w020 = ker[5], w002 = ker[6],
                     w110 = ker[7], w101 = ker[8], w011 = ker[9];
            op(o[0],      w000);
            op(o[-sx],    w100);
            op(o[+sx],    w100);
            op(o[-sy],    w010);
            op(o[+sy],    w010);
            op(o[-sz],    w001);
            op(o[+sz],    w001);
            op(o[-sx*2],  w200);
            op(o[+sx*2],  w200);
            op(o[-sy*2],  w020);
            op(o[+sy*2],  w020);
            op(o[-sz*2],  w002);
            op(o[+sz*2],  w002);
            op(o[-sx-sy], w110);
            op(o[-sx+sy], w110);
            op(o[+sx-sy], w110);
            op(o[+sx+sy], w110);
            op(o[-sx-sz], w101);
            op(o[-sx+sz], w101);
            op(o[+sx-sz], w101);
            op(o[+sx+sz], w101);
            op(o[-sy-sz], w011);
            op(o[-sy+sz], w011);
            op(o[+sy-sz], w011);
            op(o[+sy+sz], w011);
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            setkernel(out + sc*c, kernel + 10*c);
    }

    // --- diagonal ---

    template <OpType op = set>
    inline CUDEV void
    diag_bending(
              scalar_t out      [],
              offset_t osc,
        const offset_t loc      [D],
        const offset_t size     [D],
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        offset_t  x = loc[0],     y = loc[1],     z = loc[2];
        offset_t nx = size[0],   ny = size[1],   nz = size[2];

        int8_t fx00 = bound_utils_x.sign(x-2, nx);
        int8_t fx0  = bound_utils_x.sign(x-1, nx);
        int8_t fx1  = bound_utils_x.sign(x+1, nx);
        int8_t fx11 = bound_utils_x.sign(x+2, nx);
        int8_t fy00 = bound_utils_y.sign(y-2, ny);
        int8_t fy0  = bound_utils_y.sign(y-1, ny);
        int8_t fy1  = bound_utils_y.sign(y+1, ny);
        int8_t fy11 = bound_utils_y.sign(y+2, ny);
        int8_t fz00 = bound_utils_z.sign(z-2, nz);
        int8_t fz0  = bound_utils_z.sign(z-1, nz);
        int8_t fz1  = bound_utils_z.sign(z+1, nz);
        int8_t fz11 = bound_utils_z.sign(z+2, nz);

        auto setdiag = [&](scalar_t & out, const reduce_t * kernel) {
            reduce_t w000 = kernel[0],
                     w100 = kernel[1], w010 = kernel[2], w001 = kernel[3],
                     w200 = kernel[4], w020 = kernel[5], w002 = kernel[6],
                     w110 = kernel[7], w101 = kernel[8], w011 = kernel[9];
            w000 -=   w100 * (fx0 + fx1)   + w010 * (fy0 + fy1)   + w001 * (fz0 + fz1)
                    + w200 * (fx00 + fx11) + w020 * (fy00 + fy11) + w002 * (fz00 + fz11)
                    + w110 * (fx0*fy0 + fx1*fy0 + fx1*fy0 + fx1*fy1)
                    + w101 * (fx0*fz0 + fx1*fz0 + fx1*fz0 + fx1*fz1)
                    + w011 * (fy0*fz0 + fy1*fz0 + fy1*fz0 + fy1*fz1);
            op(out, w000);
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            setdiag(out[osc*c], kernel + 10*c);
    }

    //------------------------------------------------------------------
    //                         ABSOLUTE RLS
    //------------------------------------------------------------------

    // --- matvec ---

    template <OpType op = set>
    inline CUDEV
    void matvec_absolute_rls(
              scalar_t out      [],
        const scalar_t inp      [],
        const scalar_t wgt      [],
              offset_t osc,
              offset_t isc,
              offset_t wsc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            op(out[osc*c], kernel[c] *
                           static_cast<reduce_t>(wgt[wsc*c]) *
                           static_cast<reduce_t>(inp[isc*c]));
    }

    // --- diagonal ---

    template <OpType op = set>
    CUDEV inline void
    diag_absolute_rls(
              scalar_t out      [],
        const scalar_t wgt      [],
              offset_t osc,
              offset_t wsc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            op(out[osc*c], kernel[c] * static_cast<reduce_t>(wgt[wsc*c]));
    }

    //------------------------------------------------------------------
    //                         ABSOLUTE JRLS
    //------------------------------------------------------------------

    // --- matvec ---

    template <OpType op = set>
    inline CUDEV
    void matvec_absolute_jrls(
              scalar_t out      [],
        const scalar_t inp      [],
        const scalar_t wgt      [],
              offset_t osc,
              offset_t isc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        reduce_t w = static_cast<reduce_t>(*wgt);
        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            op(out[osc*c], kernel[c] * w * static_cast<reduce_t>(inp[isc*c]));
    }

    // --- diagonal ---

    template <OpType op = set>
    CUDEV inline void
    diag_absolute_jrls(
              scalar_t out      [],
        const scalar_t wgt      [],
              offset_t osc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        reduce_t w = static_cast<reduce_t>(*wgt);
        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            op(out[osc*c], kernel[c] * w);
    }

    //------------------------------------------------------------------
    //                         MEMBRANE RLS
    //------------------------------------------------------------------

    static const offset_t kernelsize_membrane_rls = kernelsize_membrane;

    CUDEV inline offset_t
    get_kernelsize_membrane_rls(offset_t nc = C)
    { return get_kernelsize_membrane(nc); }

    CUDEV inline void
    make_kernel_membrane_rls(
              reduce_t kernel       [],
        const reduce_t absolute     [],
        const reduce_t membrane     [],
        const reduce_t voxel_size   [D],
              offset_t nc           = C
    )
    {
        make_kernel_membrane(kernel, absolute, membrane, voxel_size, nc);
        for (int k=0; k<get_kernelsize_membrane_rls(); ++k)
            kernel[k] *= 0.5;
    }

    // --- matvec ---

    template <OpType op = set>
    CUDEV inline void
    matvec_membrane_rls(
              scalar_t out      [],
        const scalar_t inp      [],
        const scalar_t wgt      [],
        const offset_t loc      [D],
        const offset_t size     [D],
        const offset_t istride  [D],
        const offset_t wstride  [D],
              offset_t osc,
              offset_t isc,
              offset_t wsc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        offset_t   x = loc[0],       y = loc[1],       z = loc[2];
        offset_t  nx = size[0],     ny = size[1],     nz = size[2];
        offset_t isx = istride[0], isy = istride[1], isz = istride[2];
        offset_t wsx = wstride[0], wsy = wstride[1], wsz = wstride[2];

        int8_t   fx0 = bound_utils_x.sign(x-1, nx);
        int8_t   fx1 = bound_utils_x.sign(x+1, nx);
        int8_t   fy0 = bound_utils_y.sign(y-1, ny);
        int8_t   fy1 = bound_utils_y.sign(y+1, ny);
        int8_t   fz0 = bound_utils_z.sign(z-1, nz);
        int8_t   fz1 = bound_utils_z.sign(z+1, nz);
        offset_t ix0 = (bound_utils_x.index(x-1, nx) - x);
        offset_t ix1 = (bound_utils_x.index(x+1, nx) - x);
        offset_t iy0 = (bound_utils_y.index(y-1, ny) - y);
        offset_t iy1 = (bound_utils_y.index(y+1, ny) - y);
        offset_t iz0 = (bound_utils_z.index(z-1, nz) - z);
        offset_t iz1 = (bound_utils_z.index(z+1, nz) - z);
        offset_t wx0 = ix0 * wsx;
        offset_t wx1 = ix1 * wsx;
        offset_t wy0 = iy0 * wsy;
        offset_t wy1 = iy1 * wsy;
        offset_t wz0 = iz0 * wsz;
        offset_t wz1 = iz1 * wsz;
        ix0 *= isx;
        ix1 *= isx;
        iy0 *= isy;
        iy1 *= isy;
        iz0 *= isz;
        iz1 *= isz;

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
        {
            // --- load weight map ---
            reduce_t w111 = static_cast<reduce_t>(wgt[wsc*c]);
            // f == 0 means there is no such neighbour (e.g. Zero boundary going
            // out of range) -- `index()` is unclamped there, so a raw read would
            // be out of bounds. Fall back to replicating the centre's own weight.
            auto wget = [&](offset_t o, int8_t f)
            {
                return f ? (bound::cget<reduce_t>(wgt + wsc*c, o) + w111)
                         : (w111 + w111);
            };
            reduce_t w011 = wget(wx0, fx0);
            reduce_t w211 = wget(wx1, fx1);
            reduce_t w101 = wget(wy0, fy0);
            reduce_t w121 = wget(wy1, fy1);
            reduce_t w110 = wget(wz0, fz0);
            reduce_t w112 = wget(wz1, fz1);

            // --- convolution ---

            auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
            {
                reduce_t m000 = kernel[0], m100 = kernel[1],
                         m010 = kernel[2], m001 = kernel[3];

                reduce_t center = static_cast<reduce_t>(*inp);
                auto get = [&](offset_t o, int8_t f)
                {
                    return bound::cget<reduce_t>(inp, o, f) - center;
                };

                op(*out,
                   (m000*w111*2)*center
                   + (m100*w011)*get(ix0, fx0) + (m100*w211)*get(ix1, fx1)
                   + (m010*w101)*get(iy0, fy0) + (m010*w121)*get(iy1, fy1)
                   + (m001*w110)*get(iz0, fz0) + (m001*w112)*get(iz1, fz1)
                );
            };

            conv(out + osc*c, inp + isc*c, kernel + 4*c);
        }
    }

    // --- diagonal ---

    template <OpType op = set>
    CUDEV inline void
    diag_membrane_rls(
              scalar_t out      [],
        const scalar_t wgt      [],
        const offset_t loc      [D],
        const offset_t size     [D],
        const offset_t wstride  [D],
              offset_t osc,
              offset_t wsc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        offset_t   x = loc[0],       y = loc[1],       z = loc[2];
        offset_t  nx = size[0],     ny = size[1],     nz = size[2];
        offset_t wsx = wstride[0], wsy = wstride[1], wsz = wstride[2];

        int8_t   fx0 = bound_utils_x.sign(x-1, nx);
        int8_t   fx1 = bound_utils_x.sign(x+1, nx);
        int8_t   fy0 = bound_utils_y.sign(y-1, ny);
        int8_t   fy1 = bound_utils_y.sign(y+1, ny);
        int8_t   fz0 = bound_utils_z.sign(z-1, nz);
        int8_t   fz1 = bound_utils_z.sign(z+1, nz);
        offset_t ix0 = (bound_utils_x.index(x-1, nx) - x) * wsx;
        offset_t ix1 = (bound_utils_x.index(x+1, nx) - x) * wsx;
        offset_t iy0 = (bound_utils_y.index(y-1, ny) - y) * wsy;
        offset_t iy1 = (bound_utils_y.index(y+1, ny) - y) * wsy;
        offset_t iz0 = (bound_utils_z.index(z-1, nz) - z) * wsz;
        offset_t iz1 = (bound_utils_z.index(z+1, nz) - z) * wsz;

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
        {
            // --- load weight map ---
            reduce_t w111 = static_cast<reduce_t>(wgt[wsc*c]);
            // f == 0 means there is no such neighbour (e.g. Zero boundary going
            // out of range) -- `index()` is unclamped there, so a raw read would
            // be out of bounds. Fall back to replicating the centre's own weight.
            auto wget = [&](offset_t o, int8_t f)
            {
                return f ? (bound::cget<reduce_t>(wgt + wsc*c, o) + w111)
                         : (w111 + w111);
            };
            reduce_t w011 = wget(ix0, fx0) * fx0;
            reduce_t w211 = wget(ix1, fx1) * fx1;
            reduce_t w101 = wget(iy0, fy0) * fy0;
            reduce_t w121 = wget(iy1, fy1) * fy1;
            reduce_t w110 = wget(iz0, fz0) * fz0;
            reduce_t w112 = wget(iz1, fz1) * fz1;

            // --- convolution ---

            auto conv = [&](scalar_t * out, const reduce_t * kernel)
            {
                reduce_t m000 = kernel[0], m100 = kernel[1],
                         m010 = kernel[2], m001 = kernel[3];
                op(*out,
                   m000*w111*2
                   - m100*(w011 + w211)
                   - m010*(w101 + w121)
                   - m001*(w110 + w112)
                );
            };

            conv(out + osc*c, kernel + 4*c);
        }
    }

    //------------------------------------------------------------------
    //                         MEMBRANE JRLS
    //------------------------------------------------------------------

    // --- matvec ---

    template <OpType op = set>
    CUDEV inline void
    matvec_membrane_jrls(
              scalar_t out      [],
        const scalar_t inp      [],
        const scalar_t wgt      [],
        const offset_t loc      [D],
        const offset_t size     [D],
        const offset_t istride  [D],
        const offset_t wstride  [D],
              offset_t osc,
              offset_t isc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        offset_t   x = loc[0],       y = loc[1],       z = loc[2];
        offset_t  nx = size[0],     ny = size[1],     nz = size[2];
        offset_t isx = istride[0], isy = istride[1], isz = istride[2];
        offset_t wsx = wstride[0], wsy = wstride[1], wsz = wstride[2];

        int8_t   fx0 = bound_utils_x.sign(x-1, nx);
        int8_t   fx1 = bound_utils_x.sign(x+1, nx);
        int8_t   fy0 = bound_utils_y.sign(y-1, ny);
        int8_t   fy1 = bound_utils_y.sign(y+1, ny);
        int8_t   fz0 = bound_utils_z.sign(z-1, nz);
        int8_t   fz1 = bound_utils_z.sign(z+1, nz);
        offset_t ix0 = (bound_utils_x.index(x-1, nx) - x);
        offset_t ix1 = (bound_utils_x.index(x+1, nx) - x);
        offset_t iy0 = (bound_utils_y.index(y-1, ny) - y);
        offset_t iy1 = (bound_utils_y.index(y+1, ny) - y);
        offset_t iz0 = (bound_utils_z.index(z-1, nz) - z);
        offset_t iz1 = (bound_utils_z.index(z+1, nz) - z);
        offset_t wx0 = ix0 * wsx;
        offset_t wx1 = ix1 * wsx;
        offset_t wy0 = iy0 * wsy;
        offset_t wy1 = iy1 * wsy;
        offset_t wz0 = iz0 * wsz;
        offset_t wz1 = iz1 * wsz;
        ix0 *= isx;
        ix1 *= isx;
        iy0 *= isy;
        iy1 *= isy;
        iz0 *= isz;
        iz1 *= isz;

        // --- load weight map ---
        reduce_t w111 = static_cast<reduce_t>(*wgt);
        // f == 0 means there is no such neighbour (e.g. Zero boundary going
        // out of range) -- `index()` is unclamped there, so a raw read would
        // be out of bounds. Fall back to replicating the centre's own weight.
        auto wget = [&](offset_t o, int8_t f)
        {
            return f ? (bound::cget<reduce_t>(wgt, o) + w111)
                     : (w111 + w111);
        };
        reduce_t w011 = wget(wx0, fx0);
        reduce_t w211 = wget(wx1, fx1);
        reduce_t w101 = wget(wy0, fy0);
        reduce_t w121 = wget(wy1, fy1);
        reduce_t w110 = wget(wz0, fz0);
        reduce_t w112 = wget(wz1, fz1);

        // --- convolution ---

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t m000 = kernel[0], m100 = kernel[1],
                     m010 = kernel[2], m001 = kernel[3];

            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [&](offset_t o, int8_t f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out,
               (m000*w111*2)*center
               + (m100*w011)*get(ix0, fx0) + (m100*w211)*get(ix1, fx1)
               + (m010*w101)*get(iy0, fy0) + (m010*w121)*get(iy1, fy1)
               + (m001*w110)*get(iz0, fz0) + (m001*w112)*get(iz1, fz1)
            );
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            conv(out + osc*c, inp + isc*c, kernel + 4*c);
    }

    // --- diagonal ---

    template <OpType op = set>
    CUDEV inline void
    diag_membrane_jrls(
              scalar_t out      [],
        const scalar_t wgt      [],
        const offset_t loc      [D],
        const offset_t size     [D],
        const offset_t wstride  [D],
              offset_t osc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        offset_t   x = loc[0],       y = loc[1],       z = loc[2];
        offset_t  nx = size[0],     ny = size[1],     nz = size[2];
        offset_t wsx = wstride[0], wsy = wstride[1], wsz = wstride[2];

        int8_t   fx0 = bound_utils_x.sign(x-1, nx);
        int8_t   fx1 = bound_utils_x.sign(x+1, nx);
        int8_t   fy0 = bound_utils_y.sign(y-1, ny);
        int8_t   fy1 = bound_utils_y.sign(y+1, ny);
        int8_t   fz0 = bound_utils_z.sign(z-1, nz);
        int8_t   fz1 = bound_utils_z.sign(z+1, nz);
        offset_t ix0 = (bound_utils_x.index(x-1, nx) - x) * wsx;
        offset_t ix1 = (bound_utils_x.index(x+1, nx) - x) * wsx;
        offset_t iy0 = (bound_utils_y.index(y-1, ny) - y) * wsy;
        offset_t iy1 = (bound_utils_y.index(y+1, ny) - y) * wsy;
        offset_t iz0 = (bound_utils_z.index(z-1, nz) - z) * wsz;
        offset_t iz1 = (bound_utils_z.index(z+1, nz) - z) * wsz;

        // --- load weight map ---
        reduce_t w111 = static_cast<reduce_t>(*wgt);
        // f == 0 means there is no such neighbour (e.g. Zero boundary going
        // out of range) -- `index()` is unclamped there, so a raw read would
        // be out of bounds. Fall back to replicating the centre's own weight.
        auto wget = [&](offset_t o, int8_t f)
        {
            return f ? (bound::cget<reduce_t>(wgt, o) + w111)
                     : (w111 + w111);
        };
        reduce_t w011 = wget(ix0, fx0) * fx0;
        reduce_t w211 = wget(ix1, fx1) * fx1;
        reduce_t w101 = wget(iy0, fy0) * fy0;
        reduce_t w121 = wget(iy1, fy1) * fy1;
        reduce_t w110 = wget(iz0, fz0) * fz0;
        reduce_t w112 = wget(iz1, fz1) * fz1;

        // --- convolution ---

        auto conv = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t m000 = kernel[0], m100 = kernel[1],
                     m010 = kernel[2], m001 = kernel[3];
            op(*out,
               m000*w111*2
               - m100*(w011 + w211)
               - m010*(w101 + w121)
               - m001*(w110 + w112)
            );
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            conv(out + osc*c, kernel + 4*c);
    }

    //------------------------------------------------------------------
    //                         BENDING RLS
    //------------------------------------------------------------------

    static const offset_t kernelsize_bending_rls = kernelsize_bending;

    CUDEV inline offset_t
    get_kernelsize_bending_rls(offset_t nc = C)
    { return get_kernelsize_bending(nc); }

    inline CUDEV void
    make_kernel_bending_rls(
              reduce_t kernel       [],
        const reduce_t absolute     [],
        const reduce_t membrane     [],
        const reduce_t bending      [],
        const reduce_t voxel_size   [D],
              offset_t nc           = C
    )
    {
        make_kernel_bending(kernel, absolute, membrane, bending, voxel_size, nc);
        for (int k=0; k<get_kernelsize_bending_rls(nc); ++k)
        {
            if (k % 10 == 0) continue;
            kernel[k] *= 0.25;
        }
    }

    // --- matvec ---

    template <OpType op = set>
    CUDEV inline void
    matvec_bending_rls(
        scalar_t * out,
        const scalar_t inp      [],
        const scalar_t wgt      [],
        const offset_t loc      [D],
        const offset_t size     [D],
        const offset_t istride  [D],
        const offset_t wstride  [D],
              offset_t osc,
              offset_t isc,
              offset_t wsc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        offset_t   x = loc[0],       y = loc[1],       z = loc[2];
        offset_t  nx = size[0],     ny = size[1],     nz = size[2];
        offset_t isx = istride[0], isy = istride[1], isz = istride[2];
        offset_t wsx = wstride[0], wsy = wstride[1], wsz = wstride[2];

        int8_t   fx0 = bound_utils_x.sign(x-2, nx);
        int8_t   fx1 = bound_utils_x.sign(x-1, nx);
        int8_t   fx3 = bound_utils_x.sign(x+1, nx);
        int8_t   fx4 = bound_utils_x.sign(x+2, nx);
        int8_t   fy0 = bound_utils_y.sign(y-2, ny);
        int8_t   fy1 = bound_utils_y.sign(y-1, ny);
        int8_t   fy3 = bound_utils_y.sign(y+1, ny);
        int8_t   fy4 = bound_utils_y.sign(y+2, ny);
        int8_t   fz0 = bound_utils_z.sign(z-2, nz);
        int8_t   fz1 = bound_utils_z.sign(z-1, nz);
        int8_t   fz3 = bound_utils_z.sign(z+1, nz);
        int8_t   fz4 = bound_utils_z.sign(z+2, nz);
        offset_t ix0 = (bound_utils_x.index(x-2, nx) - x);
        offset_t ix1 = (bound_utils_x.index(x-1, nx) - x);
        offset_t ix3 = (bound_utils_x.index(x+1, nx) - x);
        offset_t ix4 = (bound_utils_x.index(x+2, nx) - x);
        offset_t iy0 = (bound_utils_y.index(y-2, ny) - y);
        offset_t iy1 = (bound_utils_y.index(y-1, ny) - y);
        offset_t iy3 = (bound_utils_y.index(y+1, ny) - y);
        offset_t iy4 = (bound_utils_y.index(y+2, ny) - y);
        offset_t iz0 = (bound_utils_z.index(z-2, nz) - z);
        offset_t iz1 = (bound_utils_z.index(z-1, nz) - z);
        offset_t iz3 = (bound_utils_z.index(z+1, nz) - z);
        offset_t iz4 = (bound_utils_z.index(z+2, nz) - z);
        offset_t wx0 = ix0 * wsx;
        offset_t wx1 = ix1 * wsx;
        offset_t wx3 = ix3 * wsx;
        offset_t wx4 = ix4 * wsx;
        offset_t wy0 = iy0 * wsy;
        offset_t wy1 = iy1 * wsy;
        offset_t wy3 = iy3 * wsy;
        offset_t wy4 = iy4 * wsy;
        offset_t wz0 = iz0 * wsz;
        offset_t wz1 = iz1 * wsz;
        offset_t wz3 = iz3 * wsz;
        offset_t wz4 = iz4 * wsz;
        ix0 *= isx;
        ix1 *= isx;
        ix3 *= isx;
        ix4 *= isx;
        iy0 *= isy;
        iy1 *= isy;
        iy3 *= isy;
        iy4 *= isy;
        iz0 *= isz;
        iz1 *= isz;
        iz3 *= isz;
        iz4 *= isz;

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c, kernel+=10, out+=osc, inp+=isc, wgt+=wsc)
        {
            reduce_t b000 = kernel[0],
                     b100 = kernel[1], b010 = kernel[2], b001 = kernel[3],
                     b200 = kernel[4], b020 = kernel[5], b002 = kernel[6],
                     b110 = kernel[7], b101 = kernel[8], b011 = kernel[9];

            reduce_t w222 = static_cast<reduce_t>(*wgt);
            // f == 0 means there is no such neighbour (e.g. Zero boundary going
            // out of range) -- `index()` is unclamped there, so a raw read would
            // be out of bounds. `alt` is used instead: the nearest weight that
            // does exist along that direction -- the centre's own weight for a
            // first-order tap, and the first-order tap for the second-order and
            // diagonal ones. Replicating the *nearest* weight rather than always
            // the centre makes the implied extension of the weight map the same
            // no matter which voxel reads it, which is what keeps the operator
            // self-adjoint.
            auto wget = [&](offset_t o, int8_t f, reduce_t alt)
            {
                return f ? bound::cget<reduce_t>(wgt, o) : alt;
            };

            // first order neighbours
            reduce_t w122 = wget(wx1, fx1, w222);
            reduce_t w322 = wget(wx3, fx3, w222);
            reduce_t w212 = wget(wy1, fy1, w222);
            reduce_t w232 = wget(wy3, fy3, w222);
            reduce_t w221 = wget(wz1, fz1, w222);
            reduce_t w223 = wget(wz3, fz3, w222);

            // second order neighbours
            reduce_t w022 = wget(wx0, fx0, w122);
            reduce_t w422 = wget(wx4, fx4, w322);
            reduce_t w202 = wget(wy0, fy0, w212);
            reduce_t w242 = wget(wy4, fy4, w232);
            reduce_t w220 = wget(wz0, fz0, w221);
            reduce_t w224 = wget(wz4, fz4, w223);

            // diagonal neighbours
            reduce_t w112 = wget(wx1+wy1, fx1*fy1, fx1 ? w122 : w212);
            reduce_t w132 = wget(wx1+wy3, fx1*fy3, fx1 ? w122 : w232);
            reduce_t w312 = wget(wx3+wy1, fx3*fy1, fx3 ? w322 : w212);
            reduce_t w332 = wget(wx3+wy3, fx3*fy3, fx3 ? w322 : w232);
            reduce_t w121 = wget(wx1+wz1, fx1*fz1, fx1 ? w122 : w221);
            reduce_t w123 = wget(wx1+wz3, fx1*fz3, fx1 ? w122 : w223);
            reduce_t w321 = wget(wx3+wz1, fx3*fz1, fx3 ? w322 : w221);
            reduce_t w323 = wget(wx3+wz3, fx3*fz3, fx3 ? w322 : w223);
            reduce_t w211 = wget(wy1+wz1, fy1*fz1, fy1 ? w212 : w221);
            reduce_t w213 = wget(wy1+wz3, fy1*fz3, fy1 ? w212 : w223);
            reduce_t w231 = wget(wy3+wz1, fy3*fz1, fy3 ? w232 : w221);
            reduce_t w233 = wget(wy3+wz3, fy3*fz3, fy3 ? w232 : w223);

            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [&](offset_t o, int8_t f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            auto sum1 = [&]()
            {
                reduce_t m122 = (b100 - 2*b200) * (w222 + w122)
                                - 2*b200 * (w322 + w022)
                                - b110 * (w212 + w112 + w232 + w132)
                                - b101 * (w221 + w121 + w223 + w123);
                reduce_t m322 = (b100 - 2*b200) * (w222 + w322)
                                - 2*b200 * (w422 + w122)
                                - b110 * (w232 + w332 + w212 + w312)
                                - b101 * (w223 + w323 + w221 + w321);

                reduce_t m212 = (b010 - 2*b020) * (w222 + w212)
                                - 2*b020 * (w232 + w202)
                                - b110 * (w122 + w112 + w322 + w312)
                                - b011 * (w221 + w211 + w223 + w213);
                reduce_t m232 = (b010 - 2*b020) * (w222 + w232)
                                - 2*b020 * (w242 + w212)
                                - b110 * (w322 + w332 + w122 + w132)
                                - b011 * (w223 + w233 + w221 + w231);

                reduce_t m221 = (b001 - 2*b002) * (w222 + w221)
                                - 2*b002 * (w223 + w220)
                                - b101 * (w122 + w121 + w322 + w321)
                                - b011 * (w212 + w211 + w232 + w231);
                reduce_t m223 = (b001 - 2*b002) * (w222 + w223)
                                - 2*b002 * (w224 + w221)
                                - b101 * (w322 + w323 + w122 + w123)
                                - b011 * (w232 + w233 + w212 + w213);

                return (m122*get(ix1, fx1) +  m322*get(ix3, fx3) +
                        m212*get(iy1, fy1) +  m232*get(iy3, fy3) +
                        m221*get(iz1, fz1) +  m223*get(iz3, fz3));
            };

            auto sum2 = [&]()
            {
                reduce_t m022 = b200 * (2 * w122 + w022 + w222);
                reduce_t m422 = b200 * (2 * w322 + w422 + w222);
                reduce_t m202 = b020 * (2 * w212 + w202 + w222);
                reduce_t m242 = b020 * (2 * w232 + w242 + w222);
                reduce_t m220 = b002 * (2 * w221 + w220 + w222);
                reduce_t m224 = b002 * (2 * w223 + w224 + w222);

                return (m022*get(ix0, fx0) +  m422*get(ix4, fx4) +
                        m202*get(iy0, fy0) +  m242*get(iy4, fy4) +
                        m220*get(iz0, fz0) +  m224*get(iz4, fz4));
            };

            auto sumdiag = [&]()
            {
                reduce_t m112 = b110 * (w222 + w122 + w212 + w112);
                reduce_t m132 = b110 * (w222 + w122 + w232 + w132);
                reduce_t m312 = b110 * (w222 + w322 + w212 + w312);
                reduce_t m332 = b110 * (w222 + w322 + w232 + w332);

                reduce_t m121 = b101 * (w222 + w122 + w221 + w121);
                reduce_t m123 = b101 * (w222 + w122 + w223 + w123);
                reduce_t m321 = b101 * (w222 + w322 + w221 + w321);
                reduce_t m323 = b101 * (w222 + w322 + w223 + w323);

                reduce_t m211 = b011 * (w222 + w212 + w221 + w211);
                reduce_t m213 = b011 * (w222 + w212 + w223 + w213);
                reduce_t m231 = b011 * (w222 + w232 + w221 + w231);
                reduce_t m233 = b011 * (w222 + w232 + w223 + w233);

                return (m112*get(ix1+iy1, fx1*fy1) +  m132*get(ix1+iy3, fx1*fy3) +
                        m312*get(ix3+iy1, fx3*fy1) +  m332*get(ix3+iy3, fx3*fy3) +
                        m121*get(ix1+iz1, fx1*fz1) +  m123*get(ix1+iz3, fx1*fz3) +
                        m321*get(ix3+iz1, fx3*fz1) +  m323*get(ix3+iz3, fx3*fz3) +
                        m211*get(iy1+iz1, fy1*fz1) +  m213*get(iy1+iz3, fy1*fz3) +
                        m231*get(iy3+iz1, fy3*fz1) +  m233*get(iy3+iz3, fy3*fz3));
            };

            op(*out, b000*center + sum1() + sum2() + sumdiag());
        }
    }

    // --- diagonal ---

    template <OpType op = set>
    CUDEV inline void
    diag_bending_rls(
              scalar_t out      [],
        const scalar_t wgt      [],
        const offset_t loc      [D],
        const offset_t size     [D],
        const offset_t wstride  [D],
              offset_t osc,
              offset_t wsc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        offset_t   x = loc[0],       y = loc[1],       z = loc[2];
        offset_t  nx = size[0],     ny = size[1],     nz = size[2];
        offset_t wsx = wstride[0], wsy = wstride[1], wsz = wstride[2];

        int8_t   fx0 = bound_utils_x.sign(x-2, nx);
        int8_t   fx1 = bound_utils_x.sign(x-1, nx);
        int8_t   fx3 = bound_utils_x.sign(x+1, nx);
        int8_t   fx4 = bound_utils_x.sign(x+2, nx);
        int8_t   fy0 = bound_utils_y.sign(y-2, ny);
        int8_t   fy1 = bound_utils_y.sign(y-1, ny);
        int8_t   fy3 = bound_utils_y.sign(y+1, ny);
        int8_t   fy4 = bound_utils_y.sign(y+2, ny);
        int8_t   fz0 = bound_utils_z.sign(z-2, nz);
        int8_t   fz1 = bound_utils_z.sign(z-1, nz);
        int8_t   fz3 = bound_utils_z.sign(z+1, nz);
        int8_t   fz4 = bound_utils_z.sign(z+2, nz);
        offset_t ix0 = (bound_utils_x.index(x-2, nx) - x) * wsx;
        offset_t ix1 = (bound_utils_x.index(x-1, nx) - x) * wsx;
        offset_t ix3 = (bound_utils_x.index(x+1, nx) - x) * wsx;
        offset_t ix4 = (bound_utils_x.index(x+2, nx) - x) * wsx;
        offset_t iy0 = (bound_utils_y.index(y-2, ny) - y) * wsy;
        offset_t iy1 = (bound_utils_y.index(y-1, ny) - y) * wsy;
        offset_t iy3 = (bound_utils_y.index(y+1, ny) - y) * wsy;
        offset_t iy4 = (bound_utils_y.index(y+2, ny) - y) * wsy;
        offset_t iz0 = (bound_utils_z.index(z-2, nz) - z) * wsz;
        offset_t iz1 = (bound_utils_z.index(z-1, nz) - z) * wsz;
        offset_t iz3 = (bound_utils_z.index(z+1, nz) - z) * wsz;
        offset_t iz4 = (bound_utils_z.index(z+2, nz) - z) * wsz;

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c, kernel+=10, out+=osc, wgt+=wsc)
        {

            reduce_t b000 = kernel[0],
                     b100 = kernel[1], b010 = kernel[2], b001 = kernel[3],
                     b200 = kernel[4], b020 = kernel[5], b002 = kernel[6],
                     b110 = kernel[7], b101 = kernel[8], b011 = kernel[9];

            reduce_t w222 = static_cast<reduce_t>(*wgt);
            // f == 0 means there is no such neighbour (e.g. Zero boundary going
            // out of range) -- `index()` is unclamped there, so a raw read would
            // be out of bounds. `alt` is used instead: the nearest weight that
            // does exist along that direction -- the centre's own weight for a
            // first-order tap, and the first-order tap for the second-order and
            // diagonal ones. Replicating the *nearest* weight rather than always
            // the centre makes the implied extension of the weight map the same
            // no matter which voxel reads it, which is what keeps the operator
            // self-adjoint.
            auto wget = [&](offset_t o, int8_t f, reduce_t alt)
            {
                return f ? bound::cget<reduce_t>(wgt, o) : alt;
            };

            reduce_t w122 = wget(ix1, fx1, w222);
            reduce_t w322 = wget(ix3, fx3, w222);
            reduce_t w212 = wget(iy1, fy1, w222);
            reduce_t w232 = wget(iy3, fy3, w222);
            reduce_t w221 = wget(iz1, fz1, w222);
            reduce_t w223 = wget(iz3, fz3, w222);

            reduce_t w022 = wget(ix0, fx0, w122);
            reduce_t w422 = wget(ix4, fx4, w322);
            reduce_t w202 = wget(iy0, fy0, w212);
            reduce_t w242 = wget(iy4, fy4, w232);
            reduce_t w220 = wget(iz0, fz0, w221);
            reduce_t w224 = wget(iz4, fz4, w223);

            reduce_t w112 = wget(ix1+iy1, fx1*fy1, fx1 ? w122 : w212);
            reduce_t w132 = wget(ix1+iy3, fx1*fy3, fx1 ? w122 : w232);
            reduce_t w312 = wget(ix3+iy1, fx3*fy1, fx3 ? w322 : w212);
            reduce_t w332 = wget(ix3+iy3, fx3*fy3, fx3 ? w322 : w232);
            reduce_t w121 = wget(ix1+iz1, fx1*fz1, fx1 ? w122 : w221);
            reduce_t w123 = wget(ix1+iz3, fx1*fz3, fx1 ? w122 : w223);
            reduce_t w321 = wget(ix3+iz1, fx3*fz1, fx3 ? w322 : w221);
            reduce_t w323 = wget(ix3+iz3, fx3*fz3, fx3 ? w322 : w223);
            reduce_t w211 = wget(iy1+iz1, fy1*fz1, fy1 ? w212 : w221);
            reduce_t w213 = wget(iy1+iz3, fy1*fz3, fy1 ? w212 : w223);
            reduce_t w231 = wget(iy3+iz1, fy3*fz1, fy3 ? w232 : w221);
            reduce_t w233 = wget(iy3+iz3, fy3*fz3, fy3 ? w232 : w223);

            reduce_t m122 = (b100 - 2*b200) * (w222 + w122)
                            - 2*b200 * (w322 + w022)
                            - b110 * (w212 + w112 + w232 + w132)
                            - b101 * (w221 + w121 + w223 + w123);
            reduce_t m322 = (b100 - 2*b200) * (w222 + w322)
                            - 2*b200 * (w422 + w122)
                            - b110 * (w232 + w332 + w212 + w312)
                            - b101 * (w223 + w323 + w221 + w321);

            reduce_t m212 = (b010 - 2*b020) * (w222 + w212)
                            - 2*b020 * (w232 + w202)
                            - b110 * (w122 + w112 + w322 + w312)
                            - b011 * (w221 + w211 + w223 + w213);
            reduce_t m232 = (b010 - 2*b020) * (w222 + w232)
                            - 2*b020 * (w242 + w212)
                            - b110 * (w322 + w332 + w122 + w132)
                            - b011 * (w223 + w233 + w221 + w231);

            reduce_t m221 = (b001 - 2*b002) * (w222 + w221)
                            - 2*b002 * (w223 + w220)
                            - b101 * (w122 + w121 + w322 + w321)
                            - b011 * (w212 + w211 + w232 + w231);
            reduce_t m223 = (b001 - 2*b002) * (w222 + w223)
                            - 2*b002 * (w224 + w221)
                            - b101 * (w322 + w323 + w122 + w123)
                            - b011 * (w232 + w233 + w212 + w213);

            reduce_t m022 = b200 * (2 * w122 + w022 + w222);
            reduce_t m422 = b200 * (2 * w322 + w422 + w222);
            reduce_t m202 = b020 * (2 * w212 + w202 + w222);
            reduce_t m242 = b020 * (2 * w232 + w242 + w222);
            reduce_t m220 = b002 * (2 * w221 + w220 + w222);
            reduce_t m224 = b002 * (2 * w223 + w224 + w222);

            reduce_t m112 = b110 * (w222 + w122 + w212 + w112);
            reduce_t m132 = b110 * (w222 + w122 + w232 + w132);
            reduce_t m312 = b110 * (w222 + w322 + w212 + w312);
            reduce_t m332 = b110 * (w222 + w322 + w232 + w332);

            reduce_t m121 = b101 * (w222 + w122 + w221 + w121);
            reduce_t m123 = b101 * (w222 + w122 + w223 + w123);
            reduce_t m321 = b101 * (w222 + w322 + w221 + w321);
            reduce_t m323 = b101 * (w222 + w322 + w223 + w323);

            reduce_t m211 = b011 * (w222 + w212 + w221 + w211);
            reduce_t m213 = b011 * (w222 + w212 + w223 + w213);
            reduce_t m231 = b011 * (w222 + w232 + w221 + w231);
            reduce_t m233 = b011 * (w222 + w232 + w223 + w233);

            b000 -= (m122*fx1 +  m322*fx3 +
                     m212*fy1 +  m232*fy3 +
                     m221*fz1 +  m223*fz3) +
                    (m022*fx0 +  m422*fx4 +
                     m202*fy0 +  m242*fy4 +
                     m220*fz0 +  m224*fz4) +
                    (m112*(fx1*fy1) +  m132*(fx1*fy3) +
                     m312*(fx3*fy1) +  m332*(fx3*fy3) +
                     m121*(fx1*fz1) +  m123*(fx1*fz3) +
                     m321*(fx3*fz1) +  m323*(fx3*fz3) +
                     m211*(fy1*fz1) +  m213*(fy1*fz3) +
                     m231*(fy3*fz1) +  m233*(fy3*fz3));

            op(*out, b000);
        }
    }

    //------------------------------------------------------------------
    //                         BENDING JRLS
    //------------------------------------------------------------------

    // --- matvec ---

    template <OpType op = set>
    CUDEV inline void
    matvec_bending_jrls(
              scalar_t out      [],
        const scalar_t inp      [],
        const scalar_t wgt      [],
        const offset_t loc      [3],
        const offset_t size     [3],
        const offset_t istride  [3],
        const offset_t wstride  [3],
              offset_t osc,
              offset_t isc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        offset_t   x = loc[0],       y = loc[1],       z = loc[2];
        offset_t  nx = size[0],     ny = size[1],     nz = size[2];
        offset_t isx = istride[0], isy = istride[1], isz = istride[2];
        offset_t wsx = wstride[0], wsy = wstride[1], wsz = wstride[2];

        int8_t   fx0 = bound_utils_x.sign(x-2, nx);
        int8_t   fx1 = bound_utils_x.sign(x-1, nx);
        int8_t   fx3 = bound_utils_x.sign(x+1, nx);
        int8_t   fx4 = bound_utils_x.sign(x+2, nx);
        int8_t   fy0 = bound_utils_y.sign(y-2, ny);
        int8_t   fy1 = bound_utils_y.sign(y-1, ny);
        int8_t   fy3 = bound_utils_y.sign(y+1, ny);
        int8_t   fy4 = bound_utils_y.sign(y+2, ny);
        int8_t   fz0 = bound_utils_z.sign(z-2, nz);
        int8_t   fz1 = bound_utils_z.sign(z-1, nz);
        int8_t   fz3 = bound_utils_z.sign(z+1, nz);
        int8_t   fz4 = bound_utils_z.sign(z+2, nz);
        offset_t ix0 = (bound_utils_x.index(x-2, nx) - x);
        offset_t ix1 = (bound_utils_x.index(x-1, nx) - x);
        offset_t ix3 = (bound_utils_x.index(x+1, nx) - x);
        offset_t ix4 = (bound_utils_x.index(x+2, nx) - x);
        offset_t iy0 = (bound_utils_y.index(y-2, ny) - y);
        offset_t iy1 = (bound_utils_y.index(y-1, ny) - y);
        offset_t iy3 = (bound_utils_y.index(y+1, ny) - y);
        offset_t iy4 = (bound_utils_y.index(y+2, ny) - y);
        offset_t iz0 = (bound_utils_z.index(z-2, nz) - z);
        offset_t iz1 = (bound_utils_z.index(z-1, nz) - z);
        offset_t iz3 = (bound_utils_z.index(z+1, nz) - z);
        offset_t iz4 = (bound_utils_z.index(z+2, nz) - z);
        offset_t wx0 = ix0 * wsx;
        offset_t wx1 = ix1 * wsx;
        offset_t wx3 = ix3 * wsx;
        offset_t wx4 = ix4 * wsx;
        offset_t wy0 = iy0 * wsy;
        offset_t wy1 = iy1 * wsy;
        offset_t wy3 = iy3 * wsy;
        offset_t wy4 = iy4 * wsy;
        offset_t wz0 = iz0 * wsz;
        offset_t wz1 = iz1 * wsz;
        offset_t wz3 = iz3 * wsz;
        offset_t wz4 = iz4 * wsz;
        ix0 *= isx;
        ix1 *= isx;
        ix3 *= isx;
        ix4 *= isx;
        iy0 *= isy;
        iy1 *= isy;
        iy3 *= isy;
        iy4 *= isy;
        iz0 *= isz;
        iz1 *= isz;
        iz3 *= isz;
        iz4 *= isz;

        reduce_t w222 = static_cast<reduce_t>(*wgt);
        // f == 0 means there is no such neighbour (e.g. Zero boundary going
        // out of range) -- `index()` is unclamped there, so a raw read would
        // be out of bounds. `alt` is used instead: the nearest weight that
        // does exist along that direction -- the centre's own weight for a
        // first-order tap, and the first-order tap for the second-order and
        // diagonal ones. Replicating the *nearest* weight rather than always
        // the centre makes the implied extension of the weight map the same
        // no matter which voxel reads it, which is what keeps the operator
        // self-adjoint.
        auto wget = [&](offset_t o, int8_t f, reduce_t alt)
        {
            return f ? bound::cget<reduce_t>(wgt, o) : alt;
        };

        // first order neighbours
        reduce_t w122 = wget(wx1, fx1, w222);
        reduce_t w322 = wget(wx3, fx3, w222);
        reduce_t w212 = wget(wy1, fy1, w222);
        reduce_t w232 = wget(wy3, fy3, w222);
        reduce_t w221 = wget(wz1, fz1, w222);
        reduce_t w223 = wget(wz3, fz3, w222);

        // second order neighbours
        reduce_t w022 = wget(wx0, fx0, w122);
        reduce_t w422 = wget(wx4, fx4, w322);
        reduce_t w202 = wget(wy0, fy0, w212);
        reduce_t w242 = wget(wy4, fy4, w232);
        reduce_t w220 = wget(wz0, fz0, w221);
        reduce_t w224 = wget(wz4, fz4, w223);

        // diagonal neighbours
        reduce_t w112 = wget(wx1+wy1, fx1*fy1, fx1 ? w122 : w212);
        reduce_t w132 = wget(wx1+wy3, fx1*fy3, fx1 ? w122 : w232);
        reduce_t w312 = wget(wx3+wy1, fx3*fy1, fx3 ? w322 : w212);
        reduce_t w332 = wget(wx3+wy3, fx3*fy3, fx3 ? w322 : w232);
        reduce_t w121 = wget(wx1+wz1, fx1*fz1, fx1 ? w122 : w221);
        reduce_t w123 = wget(wx1+wz3, fx1*fz3, fx1 ? w122 : w223);
        reduce_t w321 = wget(wx3+wz1, fx3*fz1, fx3 ? w322 : w221);
        reduce_t w323 = wget(wx3+wz3, fx3*fz3, fx3 ? w322 : w223);
        reduce_t w211 = wget(wy1+wz1, fy1*fz1, fy1 ? w212 : w221);
        reduce_t w213 = wget(wy1+wz3, fy1*fz3, fy1 ? w212 : w223);
        reduce_t w231 = wget(wy3+wz1, fy3*fz1, fy3 ? w232 : w221);
        reduce_t w233 = wget(wy3+wz3, fy3*fz3, fy3 ? w232 : w223);

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t b000 = kernel[0],
                     b100 = kernel[1], b010 = kernel[2], b001 = kernel[3],
                     b200 = kernel[4], b020 = kernel[5], b002 = kernel[6],
                     b110 = kernel[7], b101 = kernel[8], b011 = kernel[9];

            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [&](offset_t o, int8_t f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            auto sum1 = [&]()
            {
                reduce_t m122 = (b100 - 2*b200) * (w222 + w122)
                                - 2*b200 * (w322 + w022)
                                - b110 * (w212 + w112 + w232 + w132)
                                - b101 * (w221 + w121 + w223 + w123);
                reduce_t m322 = (b100 - 2*b200) * (w222 + w322)
                                - 2*b200 * (w422 + w122)
                                - b110 * (w232 + w332 + w212 + w312)
                                - b101 * (w223 + w323 + w221 + w321);

                reduce_t m212 = (b010 - 2*b020) * (w222 + w212)
                                - 2*b020 * (w232 + w202)
                                - b110 * (w122 + w112 + w322 + w312)
                                - b011 * (w221 + w211 + w223 + w213);
                reduce_t m232 = (b010 - 2*b020) * (w222 + w232)
                                - 2*b020 * (w242 + w212)
                                - b110 * (w322 + w332 + w122 + w132)
                                - b011 * (w223 + w233 + w221 + w231);

                reduce_t m221 = (b001 - 2*b002) * (w222 + w221)
                                - 2*b002 * (w223 + w220)
                                - b101 * (w122 + w121 + w322 + w321)
                                - b011 * (w212 + w211 + w232 + w231);
                reduce_t m223 = (b001 - 2*b002) * (w222 + w223)
                                - 2*b002 * (w224 + w221)
                                - b101 * (w322 + w323 + w122 + w123)
                                - b011 * (w232 + w233 + w212 + w213);

                return (m122*get(ix1, fx1) +  m322*get(ix3, fx3) +
                        m212*get(iy1, fy1) +  m232*get(iy3, fy3) +
                        m221*get(iz1, fz1) +  m223*get(iz3, fz3));
            };

            auto sum2 = [&]()
            {
                reduce_t m022 = b200 * (2 * w122 + w022 + w222);
                reduce_t m422 = b200 * (2 * w322 + w422 + w222);
                reduce_t m202 = b020 * (2 * w212 + w202 + w222);
                reduce_t m242 = b020 * (2 * w232 + w242 + w222);
                reduce_t m220 = b002 * (2 * w221 + w220 + w222);
                reduce_t m224 = b002 * (2 * w223 + w224 + w222);

                return (m022*get(ix0, fx0) +  m422*get(ix4, fx4) +
                        m202*get(iy0, fy0) +  m242*get(iy4, fy4) +
                        m220*get(iz0, fz0) +  m224*get(iz4, fz4));
            };

            auto sumdiag = [&]()
            {
                reduce_t m112 = b110 * (w222 + w122 + w212 + w112);
                reduce_t m132 = b110 * (w222 + w122 + w232 + w132);
                reduce_t m312 = b110 * (w222 + w322 + w212 + w312);
                reduce_t m332 = b110 * (w222 + w322 + w232 + w332);

                reduce_t m121 = b101 * (w222 + w122 + w221 + w121);
                reduce_t m123 = b101 * (w222 + w122 + w223 + w123);
                reduce_t m321 = b101 * (w222 + w322 + w221 + w321);
                reduce_t m323 = b101 * (w222 + w322 + w223 + w323);

                reduce_t m211 = b011 * (w222 + w212 + w221 + w211);
                reduce_t m213 = b011 * (w222 + w212 + w223 + w213);
                reduce_t m231 = b011 * (w222 + w232 + w221 + w231);
                reduce_t m233 = b011 * (w222 + w232 + w223 + w233);

                return (m112*get(ix1+iy1, fx1*fy1) +  m132*get(ix1+iy3, fx1*fy3) +
                        m312*get(ix3+iy1, fx3*fy1) +  m332*get(ix3+iy3, fx3*fy3) +
                        m121*get(ix1+iz1, fx1*fz1) +  m123*get(ix1+iz3, fx1*fz3) +
                        m321*get(ix3+iz1, fx3*fz1) +  m323*get(ix3+iz3, fx3*fz3) +
                        m211*get(iy1+iz1, fy1*fz1) +  m213*get(iy1+iz3, fy1*fz3) +
                        m231*get(iy3+iz1, fy3*fz1) +  m233*get(iy3+iz3, fy3*fz3));
            };

            op(*out, b000*center + sum1() + sum2() + sumdiag());
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            conv(out + osc*c, inp + isc*c, kernel + 10*c);
    }

    // --- diagonal ---

    template <OpType op = set>
    CUDEV inline void
    diag_bending_jrls(
              scalar_t out      [],
        const scalar_t wgt      [],
        const offset_t loc      [D],
        const offset_t size     [D],
        const offset_t wstride  [D],
              offset_t osc,
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        offset_t   x = loc[0],       y = loc[1],       z = loc[2];
        offset_t  nx = size[0],     ny = size[1],     nz = size[2];
        offset_t wsx = wstride[0], wsy = wstride[1], wsz = wstride[2];

        int8_t fx0 = bound_utils_x.sign(x-2, nx);
        int8_t fx1 = bound_utils_x.sign(x-1, nx);
        int8_t fx3 = bound_utils_x.sign(x+1, nx);
        int8_t fx4 = bound_utils_x.sign(x+2, nx);
        int8_t fy0 = bound_utils_y.sign(y-2, ny);
        int8_t fy1 = bound_utils_y.sign(y-1, ny);
        int8_t fy3 = bound_utils_y.sign(y+1, ny);
        int8_t fy4 = bound_utils_y.sign(y+2, ny);
        int8_t fz0 = bound_utils_z.sign(z-2, nz);
        int8_t fz1 = bound_utils_z.sign(z-1, nz);
        int8_t fz3 = bound_utils_z.sign(z+1, nz);
        int8_t fz4 = bound_utils_z.sign(z+2, nz);
        offset_t ix0 = (bound_utils_x.index(x-2, nx) - x) * wsx;
        offset_t ix1 = (bound_utils_x.index(x-1, nx) - x) * wsx;
        offset_t ix3 = (bound_utils_x.index(x+1, nx) - x) * wsx;
        offset_t ix4 = (bound_utils_x.index(x+2, nx) - x) * wsx;
        offset_t iy0 = (bound_utils_y.index(y-2, ny) - y) * wsy;
        offset_t iy1 = (bound_utils_y.index(y-1, ny) - y) * wsy;
        offset_t iy3 = (bound_utils_y.index(y+1, ny) - y) * wsy;
        offset_t iy4 = (bound_utils_y.index(y+2, ny) - y) * wsy;
        offset_t iz0 = (bound_utils_z.index(z-2, nz) - z) * wsz;
        offset_t iz1 = (bound_utils_z.index(z-1, nz) - z) * wsz;
        offset_t iz3 = (bound_utils_z.index(z+1, nz) - z) * wsz;
        offset_t iz4 = (bound_utils_z.index(z+2, nz) - z) * wsz;

        reduce_t w222 = static_cast<reduce_t>(*wgt);
        // f == 0 means there is no such neighbour (e.g. Zero boundary going
        // out of range) -- `index()` is unclamped there, so a raw read would
        // be out of bounds. `alt` is used instead: the nearest weight that
        // does exist along that direction -- the centre's own weight for a
        // first-order tap, and the first-order tap for the second-order and
        // diagonal ones. Replicating the *nearest* weight rather than always
        // the centre makes the implied extension of the weight map the same
        // no matter which voxel reads it, which is what keeps the operator
        // self-adjoint.
        auto wget = [&](offset_t o, int8_t f, reduce_t alt)
        {
            return f ? bound::cget<reduce_t>(wgt, o) : alt;
        };

        reduce_t w122 = wget(ix1, fx1, w222);
        reduce_t w322 = wget(ix3, fx3, w222);
        reduce_t w212 = wget(iy1, fy1, w222);
        reduce_t w232 = wget(iy3, fy3, w222);
        reduce_t w221 = wget(iz1, fz1, w222);
        reduce_t w223 = wget(iz3, fz3, w222);

        reduce_t w022 = wget(ix0, fx0, w122);
        reduce_t w422 = wget(ix4, fx4, w322);
        reduce_t w202 = wget(iy0, fy0, w212);
        reduce_t w242 = wget(iy4, fy4, w232);
        reduce_t w220 = wget(iz0, fz0, w221);
        reduce_t w224 = wget(iz4, fz4, w223);

        reduce_t w112 = wget(ix1+iy1, fx1*fy1, fx1 ? w122 : w212);
        reduce_t w132 = wget(ix1+iy3, fx1*fy3, fx1 ? w122 : w232);
        reduce_t w312 = wget(ix3+iy1, fx3*fy1, fx3 ? w322 : w212);
        reduce_t w332 = wget(ix3+iy3, fx3*fy3, fx3 ? w322 : w232);
        reduce_t w121 = wget(ix1+iz1, fx1*fz1, fx1 ? w122 : w221);
        reduce_t w123 = wget(ix1+iz3, fx1*fz3, fx1 ? w122 : w223);
        reduce_t w321 = wget(ix3+iz1, fx3*fz1, fx3 ? w322 : w221);
        reduce_t w323 = wget(ix3+iz3, fx3*fz3, fx3 ? w322 : w223);
        reduce_t w211 = wget(iy1+iz1, fy1*fz1, fy1 ? w212 : w221);
        reduce_t w213 = wget(iy1+iz3, fy1*fz3, fy1 ? w212 : w223);
        reduce_t w231 = wget(iy3+iz1, fy3*fz1, fy3 ? w232 : w221);
        reduce_t w233 = wget(iy3+iz3, fy3*fz3, fy3 ? w232 : w223);

        auto conv = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t b000 = kernel[0],
                     b100 = kernel[1], b010 = kernel[2], b001 = kernel[3],
                     b200 = kernel[4], b020 = kernel[5], b002 = kernel[6],
                     b110 = kernel[7], b101 = kernel[8], b011 = kernel[9];

            reduce_t m122 = (b100 - 2*b200) * (w222 + w122)
                            - 2*b200 * (w322 + w022)
                            - b110 * (w212 + w112 + w232 + w132)
                            - b101 * (w221 + w121 + w223 + w123);
            reduce_t m322 = (b100 - 2*b200) * (w222 + w322)
                            - 2*b200 * (w422 + w122)
                            - b110 * (w232 + w332 + w212 + w312)
                            - b101 * (w223 + w323 + w221 + w321);

            reduce_t m212 = (b010 - 2*b020) * (w222 + w212)
                            - 2*b020 * (w232 + w202)
                            - b110 * (w122 + w112 + w322 + w312)
                            - b011 * (w221 + w211 + w223 + w213);
            reduce_t m232 = (b010 - 2*b020) * (w222 + w232)
                            - 2*b020 * (w242 + w212)
                            - b110 * (w322 + w332 + w122 + w132)
                            - b011 * (w223 + w233 + w221 + w231);

            reduce_t m221 = (b001 - 2*b002) * (w222 + w221)
                            - 2*b002 * (w223 + w220)
                            - b101 * (w122 + w121 + w322 + w321)
                            - b011 * (w212 + w211 + w232 + w231);
            reduce_t m223 = (b001 - 2*b002) * (w222 + w223)
                            - 2*b002 * (w224 + w221)
                            - b101 * (w322 + w323 + w122 + w123)
                            - b011 * (w232 + w233 + w212 + w213);

            reduce_t m022 = b200 * (2 * w122 + w022 + w222);
            reduce_t m422 = b200 * (2 * w322 + w422 + w222);
            reduce_t m202 = b020 * (2 * w212 + w202 + w222);
            reduce_t m242 = b020 * (2 * w232 + w242 + w222);
            reduce_t m220 = b002 * (2 * w221 + w220 + w222);
            reduce_t m224 = b002 * (2 * w223 + w224 + w222);

            reduce_t m112 = b110 * (w222 + w122 + w212 + w112);
            reduce_t m132 = b110 * (w222 + w122 + w232 + w132);
            reduce_t m312 = b110 * (w222 + w322 + w212 + w312);
            reduce_t m332 = b110 * (w222 + w322 + w232 + w332);

            reduce_t m121 = b101 * (w222 + w122 + w221 + w121);
            reduce_t m123 = b101 * (w222 + w122 + w223 + w123);
            reduce_t m321 = b101 * (w222 + w322 + w221 + w321);
            reduce_t m323 = b101 * (w222 + w322 + w223 + w323);

            reduce_t m211 = b011 * (w222 + w212 + w221 + w211);
            reduce_t m213 = b011 * (w222 + w212 + w223 + w213);
            reduce_t m231 = b011 * (w222 + w232 + w221 + w231);
            reduce_t m233 = b011 * (w222 + w232 + w223 + w233);

            b000 -= (m122*fx1 +  m322*fx3 +
                     m212*fy1 +  m232*fy3 +
                     m221*fz1 +  m223*fz3) +
                    (m022*fx0 +  m422*fx4 +
                     m202*fy0 +  m242*fy4 +
                     m220*fz0 +  m224*fz4) +
                    (m112*(fx1*fy1) +  m132*(fx1*fy3) +
                     m312*(fx3*fy1) +  m332*(fx3*fy3) +
                     m121*(fx1*fz1) +  m123*(fx1*fz3) +
                     m321*(fx3*fz1) +  m323*(fx3*fz3) +
                     m211*(fy1*fz1) +  m213*(fy1*fz3) +
                     m231*(fy3*fz1) +  m233*(fy3*fz3));

            op(*out, b000);
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            conv(out + osc*c, kernel + 10*c);
    }
};

FF_NAMESPACE_END(reg_field)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_REGULARISERS_FIELD_3D
