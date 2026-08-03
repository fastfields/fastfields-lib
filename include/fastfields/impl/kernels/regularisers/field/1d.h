#ifndef FF_REGULARISERS_FIELD_1D
#define FF_REGULARISERS_FIELD_1D
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
struct Kernels<Config<one, _C, T...>>
{
    using _Config = Config<one, _C, T...>;
    using scalar_t = typename _Config::scalar_t;
    using reduce_t = typename _Config::reduce_t;
    using offset_t = typename _Config::offset_t;
    static constexpr offset_t D   = static_cast<offset_t>(one);
    static constexpr offset_t C   = static_cast<offset_t>(_C);
    static constexpr bound_t  BX  = _Config::Bound::template At<0>::Value;
    // Boundary helpers. `bound::dyn<B>` is an empty, zero-cost forwarder when B
    // is a real condition, and carries the condition as a data member when B is
    // `bound::type::Dynamic` (single instantiation, runtime dispatch).
    bound::dyn<BX> bound_utils_x;

    inline CUDEV Kernels() {}

    // Runtime boundary conditions; ignored by statically instantiated axes.
    explicit inline CUDEV Kernels(const ::FF::bound::BoundVec & bnd)
        : bound_utils_x(bnd[0]) {}
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
        for (offset_t c = 0; c < (C < 0 ? nc : C); ++c)
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

    /// kernel <- [abs, w1, ...]
    CUDEV inline void
    make_kernel_membrane(
              reduce_t kernel       [],
        const reduce_t absolute     [],
        const reduce_t membrane     [],
        const reduce_t voxel_size   [D],
              offset_t nc           = C
    )
    {
        reduce_t vx = voxel_size[0];
        vx = 1./(vx*vx);
        for (offset_t c = 0; c < (C < 0 ? nc : C); ++c, kernel+=(D+1))
        {
            reduce_t m = membrane[c];
            kernel[0] = absolute[c];
            kernel[1] = -m * vx;
        }
    }

    /// kernel <- [w00, w10, w01, ...]
    CUDEV inline void
    make_fullkernel_membrane(
              reduce_t kernel       [],
        const reduce_t absolute     [],
        const reduce_t membrane     [],
        const reduce_t voxel_size   [D],
              offset_t nc           = C
    )
    {
        reduce_t vx = voxel_size[0];
        vx = 1./(vx*vx);
        for (offset_t c = 0; c < (C < 0 ? nc : C); ++c, kernel+=(D+1))
        {
            reduce_t m = membrane[c];
            kernel[0] = absolute[c] + 2 * m * vx;
            kernel[1] = -m * vx;
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
              offset_t nc       = C
    )
    {
        offset_t  x = loc[0];
        offset_t nx = size[0];
        offset_t sx = stride[0];

        offset_t  x0 = x-1, x1 = x+1;
        int8_t   fx0 = bound_utils_x.sign(x0, nx);
        int8_t   fx1 = bound_utils_x.sign(x1, nx);
        x0 = (bound_utils_x.index(x0, nx) - x) * sx;
        x1 = (bound_utils_x.index(x1, nx) - x) * sx;

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t center = static_cast<reduce_t>(inp[0]);
            auto get = [&](offset_t o, int8_t f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out, kernel[0] * center +
                     kernel[1] * (get(x0, fx0) + get(x1, fx1)));
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            conv(out + osc*c, inp + isc*c, kernel + (D+1)*c);
    }

    // --- kernel ---

    template <OpType op = set>
    CUDEV inline void
    kernel_membrane(
              scalar_t out      [],
              offset_t sc,
        const offset_t stride   [D],
        const reduce_t kernel   [],
              offset_t nc       = C
    )
    {
        offset_t sx = stride[0];

        auto setkernel = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t w0 = kernel[0], w1 = kernel[1];
            op(out[0],   w0);
            op(out[-sx], w1);
            op(out[+sx], w1);
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            setkernel(out + sc*c, kernel + (D+1)*c);
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
        offset_t  x = loc[0];
        offset_t nx = size[0];

        int8_t fx = bound_utils_x.sign(x-1, nx)
                       + bound_utils_x.sign(x+1, nx);

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c, kernel += (D+1))
            op(out[osc*c], kernel[0] - kernel[1]*fx);
    }

    //------------------------------------------------------------------
    //                            BENDING
    //------------------------------------------------------------------

    static const offset_t kernelsize_bending = 3*C;

    CUDEV inline offset_t
    get_kernelsize_bending(offset_t nc = C)
    { return 3 * (C < 0 ? nc : C); }

    /// kernel <- [abs, w1, w2, ...]
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
        reduce_t vx = voxel_size[0];
        vx = 1./(vx*vx);
        for (offset_t c=0; c<(C < 0 ? nc : C); ++c, kernel+=3)
        {
            reduce_t m = membrane[c], b = bending[c];
            kernel[0] = absolute[c];
            kernel[1] = -4 * b * vx * vx - m * vx;
            kernel[2] = b * vx * vx;
        }
    }

    /// kernel <- [w0, w1, w2, ...]
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
        reduce_t vx = voxel_size[0];
        vx = 1./(vx*vx);
        for (offset_t c=0; c<(C < 0 ? nc : C); ++c, kernel+=3)
        {
            reduce_t m = membrane[c], b = bending[c];
            kernel[1] = -4 * b * vx * vx - m * vx;
            kernel[2] = b * vx * vx;
            kernel[0] = absolute[c] - 2 * (kernel[1] + kernel[2]);
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
        offset_t  x = loc[0];
        offset_t nx = size[0];
        offset_t sx = stride[0];

        int8_t   fx00 = bound_utils_x.sign(x-2, nx);
        int8_t   fx0  = bound_utils_x.sign(x-1, nx);
        int8_t   fx1  = bound_utils_x.sign(x+1, nx);
        int8_t   fx11 = bound_utils_x.sign(x+2, nx);
        offset_t x00 = (bound_utils_x.index(x-2, nx) - x) * sx;
        offset_t x0  = (bound_utils_x.index(x-1, nx) - x) * sx;
        offset_t x1  = (bound_utils_x.index(x+1, nx) - x) * sx;
        offset_t x11 = (bound_utils_x.index(x+2, nx) - x) * sx;

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t w0 = kernel[0],
                     w1 = kernel[1],
                     w2 = kernel[2];

            reduce_t center = static_cast<reduce_t>(inp[0]);
            auto get = [&](offset_t o, int8_t f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out,
                  w0 * center
                + w1 * (get(x0, fx0) + get(x1, fx1))
                + w2 * (get(x00, fx00) + get(x11, fx11))
            );
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            conv(out + osc*c, inp + isc*c, kernel + 3*c);
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
        offset_t sx = stride[0];

        auto setkernel = [&](scalar_t * o, const reduce_t * ker)
        {
            reduce_t w0 = ker[0],
                     w1 = ker[1],
                     w2 = ker[2];
            op(o[0],      w0);
            op(o[-sx],    w1);
            op(o[+sx],    w1);
            op(o[-sx*2],  w2);
            op(o[+sx*2],  w2);
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            setkernel(out + sc*c, kernel + 3*c);
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
        offset_t  x = loc[0];
        offset_t nx = size[0];

        int8_t fx00 = bound_utils_x.sign(x-2, nx);
        int8_t fx0  = bound_utils_x.sign(x-1, nx);
        int8_t fx1  = bound_utils_x.sign(x+1, nx);
        int8_t fx11 = bound_utils_x.sign(x+2, nx);

        auto setdiag = [&](scalar_t & out, const reduce_t * kernel)
        {
            reduce_t w0 = kernel[0],
                     w1 = kernel[1],
                     w2 = kernel[2];
            w0 -= w1 * (fx0 + fx1) + w2 * (fx00 + fx11);
            op(out, w0);
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            setdiag(out[osc*c], kernel + 3*c);
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
        for (int k=0; k<get_kernelsize_membrane_rls(nc); ++k)
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
        offset_t   x = loc[0];
        offset_t  nx = size[0];
        offset_t isx = istride[0];
        offset_t wsx = wstride[0];

        int8_t   fx0 = bound_utils_x.sign(x-1, nx);
        int8_t   fx1 = bound_utils_x.sign(x+1, nx);
        offset_t ix0 = (bound_utils_x.index(x-1, nx) - x);
        offset_t ix1 = (bound_utils_x.index(x+1, nx) - x);
        offset_t wx0 = ix0 * wsx;
        offset_t wx1 = ix1 * wsx;
        ix0 *= isx;
        ix1 *= isx;

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
        {
            // --- load weight map ---
            reduce_t w1 = static_cast<reduce_t>(wgt[wsc*c]);
            // f == 0 means there is no such neighbour (e.g. Zero boundary going
            // out of range) -- `index()` is unclamped there, so a raw read would
            // be out of bounds. Fall back to replicating the centre's own weight.
            auto wget = [&](offset_t o, int8_t f)
            {
                return f ? (bound::cget<reduce_t>(wgt + wsc*c, o) + w1)
                         : (w1 + w1);
            };
            reduce_t w0 = wget(wx0, fx0);
            reduce_t w2 = wget(wx1, fx1);

            // --- convolution ---

            auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
            {
                reduce_t m0 = kernel[0], m1 = kernel[1];

                reduce_t center = static_cast<reduce_t>(*inp);
                auto get = [&](offset_t o, int8_t f)
                {
                    return bound::cget<reduce_t>(inp, o, f) - center;
                };

                op(*out,
                   (m0*w1*2)*center
                   + (m1*w0)*get(ix0, fx0) + (m1*w2)*get(ix1, fx1)
                );
            };

            conv(out + osc*c, inp + isc*c, kernel + (D+1)*c);
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
        offset_t   x = loc[0];
        offset_t  nx = size[0];
        offset_t wsx = wstride[0];

        int8_t   fx0 = bound_utils_x.sign(x-1, nx);
        int8_t   fx1 = bound_utils_x.sign(x+1, nx);
        offset_t ix0 = (bound_utils_x.index(x-1, nx) - x) * wsx;
        offset_t ix1 = (bound_utils_x.index(x+1, nx) - x) * wsx;

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
        {
            // --- load weight map ---
            reduce_t w1 = static_cast<reduce_t>(wgt[wsc*c]);
            // f == 0 means there is no such neighbour (e.g. Zero boundary going
            // out of range) -- `index()` is unclamped there, so a raw read would
            // be out of bounds. Fall back to replicating the centre's own weight.
            auto wget = [&](offset_t o, int8_t f)
            {
                return f ? (bound::cget<reduce_t>(wgt + wsc*c, o) + w1)
                         : (w1 + w1);
            };
            reduce_t w0 = wget(ix0, fx0) * fx0;
            reduce_t w2 = wget(ix1, fx1) * fx1;

            // --- convolution ---

            auto conv = [&](scalar_t * out, const reduce_t * kernel)
            {
                reduce_t m0 = kernel[0], m1 = kernel[1];
                op(*out, m0*w1*2 - m1*(w0 + w2));
            };

            conv(out + osc*c, kernel + (D+1)*c);
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
        offset_t   x = loc[0];
        offset_t  nx = size[0];
        offset_t isx = istride[0];
        offset_t wsx = wstride[0];

        int8_t   fx0 = bound_utils_x.sign(x-1, nx);
        int8_t   fx1 = bound_utils_x.sign(x+1, nx);
        offset_t ix0 = (bound_utils_x.index(x-1, nx) - x);
        offset_t ix1 = (bound_utils_x.index(x+1, nx) - x);
        offset_t wx0 = ix0 * wsx;
        offset_t wx1 = ix1 * wsx;
        ix0 *= isx;
        ix1 *= isx;

        // --- load weight map ---
        reduce_t w1 = static_cast<reduce_t>(*wgt);
        // f == 0 means there is no such neighbour (e.g. Zero boundary going
        // out of range) -- `index()` is unclamped there, so a raw read would
        // be out of bounds. Fall back to replicating the centre's own weight.
        auto wget = [&](offset_t o, int8_t f)
        {
            return f ? (bound::cget<reduce_t>(wgt, o) + w1)
                     : (w1 + w1);
        };
        reduce_t w0 = wget(wx0, fx0);
        reduce_t w2 = wget(wx1, fx1);

        // --- convolution ---

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t m0 = kernel[0], m1 = kernel[1];

            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [&](offset_t o, int8_t f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            op(*out,
               (m0*w1*2)*center
               + (m1*w0)*get(ix0, fx0) + (m1*w2)*get(ix1, fx1)
            );
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            conv(out + osc*c, inp + isc*c, kernel + (D+1)*c);
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
        offset_t   x = loc[0];
        offset_t  nx = size[0];
        offset_t wsx = wstride[0];

        int8_t   fx0 = bound_utils_x.sign(x-1, nx);
        int8_t   fx1 = bound_utils_x.sign(x+1, nx);
        offset_t ix0 = (bound_utils_x.index(x-1, nx) - x) * wsx;
        offset_t ix1 = (bound_utils_x.index(x+1, nx) - x) * wsx;

        // --- load weight map ---
        reduce_t w1 = static_cast<reduce_t>(*wgt);
        // f == 0 means there is no such neighbour (e.g. Zero boundary going
        // out of range) -- `index()` is unclamped there, so a raw read would
        // be out of bounds. Fall back to replicating the centre's own weight.
        auto wget = [&](offset_t o, int8_t f)
        {
            return f ? (bound::cget<reduce_t>(wgt, o) + w1)
                     : (w1 + w1);
        };
        reduce_t w0 = wget(ix0, fx0) * fx0;
        reduce_t w2 = wget(ix1, fx1) * fx1;

        // --- convolution ---

        auto conv = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t m0 = kernel[0], m1 = kernel[1];
            op(*out, m0*w1*2 - m1*(w0 + w2));
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            conv(out + osc*c, kernel + (D+1)*c);
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
        // The RLS matvec folds the weight map into each coefficient by summing
        // the weights of the voxels straddling the corresponding finite
        // difference, so the kernel must be pre-divided by that number of
        // weight samples. That count differs between the two penalties: the
        // second-order (bending) coefficients are shared by 4 weight samples,
        // the first-order (membrane) ones by only 2. Rescaling the whole table
        // by 0.25 -- as a single pass over `make_kernel_bending` would --
        // therefore runs the membrane term at half strength, since `kernel[1]`
        // mixes a bending *and* a membrane contribution. Build the table
        // directly instead, applying 0.25 to the bending part and 0.5 to the
        // membrane part. `kernel[0]` (absolute) is left unweighted.
        reduce_t vx = voxel_size[0];
        vx = 1. / (vx * vx);
        for (offset_t c = 0; c < (C < 0 ? nc : C); ++c, kernel += 3) {
            reduce_t m = membrane[c], b = bending[c];
            kernel[0] = absolute[c];
            kernel[1] = -b * vx * vx - 0.5 * m * vx;
            kernel[2] = 0.25 * b * vx * vx;
        }
    }

    // --- matvec ---

    template <OpType op = set>
    CUDEV inline void
    matvec_bending_rls(
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
        offset_t   x = loc[0];
        offset_t  nx = size[0];
        offset_t isx = istride[0];
        offset_t wsx = wstride[0];

        int8_t   fx0 = bound_utils_x.sign(x-2, nx);
        int8_t   fx1 = bound_utils_x.sign(x-1, nx);
        int8_t   fx3 = bound_utils_x.sign(x+1, nx);
        int8_t   fx4 = bound_utils_x.sign(x+2, nx);
        offset_t ix0 = (bound_utils_x.index(x-2, nx) - x);
        offset_t ix1 = (bound_utils_x.index(x-1, nx) - x);
        offset_t ix3 = (bound_utils_x.index(x+1, nx) - x);
        offset_t ix4 = (bound_utils_x.index(x+2, nx) - x);
        offset_t wx0 = ix0 * wsx;
        offset_t wx1 = ix1 * wsx;
        offset_t wx3 = ix3 * wsx;
        offset_t wx4 = ix4 * wsx;
        ix0 *= isx;
        ix1 *= isx;
        ix3 *= isx;
        ix4 *= isx;

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c, kernel+=3, out+=osc, inp+=isc, wgt+=wsc)
        {
            reduce_t b0 = kernel[0], b1 = kernel[1], b2 = kernel[2];

            reduce_t w2 = static_cast<reduce_t>(*wgt);
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
            reduce_t w1 = wget(wx1, fx1, w2);
            reduce_t w3 = wget(wx3, fx3, w2);

            // second order neighbours
            reduce_t w0 = wget(wx0, fx0, w1);
            reduce_t w4 = wget(wx4, fx4, w3);

            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [&](offset_t o, int8_t f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            auto sum1 = [&]()
            {
                reduce_t m1 = (b1 - 2*b2) * (w2 + w1) - 2*b2 * (w3 + w0);
                reduce_t m3 = (b1 - 2*b2) * (w2 + w3) - 2*b2 * (w4 + w1);
                return (m1*get(ix1, fx1) +  m3*get(ix3, fx3));
            };

            auto sum2 = [&]()
            {
                reduce_t m0 = b2 * (2 * w1 + w0 + w2);
                reduce_t m4 = b2 * (2 * w3 + w4 + w2);
                return (m0*get(ix0, fx0) +  m4*get(ix4, fx4));
            };

            op(*out, b0*center + sum1() + sum2());
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
        offset_t   x = loc[0];
        offset_t  nx = size[0];
        offset_t wsx = wstride[0];

        int8_t   fx0 = bound_utils_x.sign(x-2, nx);
        int8_t   fx1 = bound_utils_x.sign(x-1, nx);
        int8_t   fx3 = bound_utils_x.sign(x+1, nx);
        int8_t   fx4 = bound_utils_x.sign(x+2, nx);
        offset_t ix0 = (bound_utils_x.index(x-2, nx) - x) * wsx;
        offset_t ix1 = (bound_utils_x.index(x-1, nx) - x) * wsx;
        offset_t ix3 = (bound_utils_x.index(x+1, nx) - x) * wsx;
        offset_t ix4 = (bound_utils_x.index(x+2, nx) - x) * wsx;

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c, kernel+=3, out+=osc, wgt+=wsc)
        {
            reduce_t b0 = kernel[0], b1 = kernel[1], b2 = kernel[2];

            reduce_t w2 = static_cast<reduce_t>(*wgt);
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

            reduce_t w1 = wget(ix1, fx1, w2);
            reduce_t w3 = wget(ix3, fx3, w2);
            reduce_t w0 = wget(ix0, fx0, w1);
            reduce_t w4 = wget(ix4, fx4, w3);

            reduce_t m1 = (b1 - 2*b2) * (w2 + w1) - 2*b2 * (w3 + w0);
            reduce_t m3 = (b1 - 2*b2) * (w2 + w3) - 2*b2 * (w4 + w1);
            reduce_t m0 = b2 * (2 * w1 + w0 + w2);
            reduce_t m4 = b2 * (2 * w3 + w4 + w2);

            b0 -= (m1*fx1 +  m3*fx3) + (m0*fx0 +  m4*fx4);
            op(*out, b0);
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
        offset_t   x = loc[0];
        offset_t  nx = size[0];
        offset_t isx = istride[0];
        offset_t wsx = wstride[0];

        int8_t   fx0 = bound_utils_x.sign(x-2, nx);
        int8_t   fx1 = bound_utils_x.sign(x-1, nx);
        int8_t   fx3 = bound_utils_x.sign(x+1, nx);
        int8_t   fx4 = bound_utils_x.sign(x+2, nx);
        offset_t ix0 = (bound_utils_x.index(x-2, nx) - x);
        offset_t ix1 = (bound_utils_x.index(x-1, nx) - x);
        offset_t ix3 = (bound_utils_x.index(x+1, nx) - x);
        offset_t ix4 = (bound_utils_x.index(x+2, nx) - x);
        offset_t wx0 = ix0 * wsx;
        offset_t wx1 = ix1 * wsx;
        offset_t wx3 = ix3 * wsx;
        offset_t wx4 = ix4 * wsx;
        ix0 *= isx;
        ix1 *= isx;
        ix3 *= isx;
        ix4 *= isx;

        reduce_t w2 = static_cast<reduce_t>(*wgt);
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
        reduce_t w1 = wget(wx1, fx1, w2);
        reduce_t w3 = wget(wx3, fx3, w2);

        // second order neighbours
        reduce_t w0 = wget(wx0, fx0, w1);
        reduce_t w4 = wget(wx4, fx4, w3);

        auto conv = [&](scalar_t * out, const scalar_t * inp, const reduce_t * kernel)
        {
            reduce_t b0 = kernel[0], b1 = kernel[1], b2 = kernel[2];

            reduce_t center = static_cast<reduce_t>(*inp);
            auto get = [&](offset_t o, int8_t f)
            {
                return bound::cget<reduce_t>(inp, o, f) - center;
            };

            auto sum1 = [&]()
            {
                reduce_t m1 = (b1 - 2*b2) * (w2 + w1) - 2*b2 * (w3 + w0);
                reduce_t m3 = (b1 - 2*b2) * (w2 + w3) - 2*b2 * (w4 + w1);
                return (m1*get(ix1, fx1) +  m3*get(ix3, fx3));
            };

            auto sum2 = [&]()
            {
                reduce_t m0 = b2 * (2 * w1 + w0 + w2);
                reduce_t m4 = b2 * (2 * w3 + w4 + w2);
                return (m0*get(ix0, fx0) +  m4*get(ix4, fx4));
            };

            op(*out, b0*center + sum1() + sum2());
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            conv(out + osc*c, inp + isc*c, kernel + 3*c);
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
        offset_t   x = loc[0];
        offset_t  nx = size[0];
        offset_t wsx = wstride[0];

        int8_t   fx0 = bound_utils_x.sign(x-2, nx);
        int8_t   fx1 = bound_utils_x.sign(x-1, nx);
        int8_t   fx3 = bound_utils_x.sign(x+1, nx);
        int8_t   fx4 = bound_utils_x.sign(x+2, nx);
        offset_t ix0 = (bound_utils_x.index(x-2, nx) - x) * wsx;
        offset_t ix1 = (bound_utils_x.index(x-1, nx) - x) * wsx;
        offset_t ix3 = (bound_utils_x.index(x+1, nx) - x) * wsx;
        offset_t ix4 = (bound_utils_x.index(x+2, nx) - x) * wsx;

        reduce_t w2 = static_cast<reduce_t>(*wgt);
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

        reduce_t w1 = wget(ix1, fx1, w2);
        reduce_t w3 = wget(ix3, fx3, w2);
        reduce_t w0 = wget(ix0, fx0, w1);
        reduce_t w4 = wget(ix4, fx4, w3);

        auto conv = [&](scalar_t * out, const reduce_t * kernel)
        {
            reduce_t b0 = kernel[0], b1 = kernel[1], b2 = kernel[2];

            reduce_t m1 = (b1 - 2*b2) * (w2 + w1) - 2*b2 * (w3 + w0);
            reduce_t m3 = (b1 - 2*b2) * (w2 + w3) - 2*b2 * (w4 + w1);
            reduce_t m0 = b2 * (2 * w1 + w0 + w2);
            reduce_t m4 = b2 * (2 * w3 + w4 + w2);

            b0 -= (m1*fx1 +  m3*fx3) + (m0*fx0 +  m4*fx4);
            op(*out, b0);
        };

        for (offset_t c=0; c<(C < 0 ? nc : C); ++c)
            conv(out + osc*c, kernel + 3*c);
    }
};

FF_NAMESPACE_END(reg_field)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_REGULARISERS_FIELD_1D
