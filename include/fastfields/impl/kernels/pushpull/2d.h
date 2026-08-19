/***********************************************************************
 *
 *                                  2D
 *
 **********************************************************************/
#ifndef FF_PUSHPULL_2D
#define FF_PUSHPULL_2D
#include "../cuda_switch.h"
#include "../spline.h"
#include "../bounds.h"
#include "utils.h"

// TODO: quadratic and cubic specializations

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(pushpull)

/***********************************************************************
 *
 *                               NEAREST
 *
 **********************************************************************/
template <bound::type BX, bound::type BY, bool ABS>
struct Kernels<Config<two, Spline<Z,Z>, Bound<BX, BY>, ABS>> {
    using utils_x = PushPullUtils<Z, BX, ABS>;
    using utils_y = PushPullUtils<Z, BY, ABS>;
    using self = Kernels<Config<two, Spline<Z,Z>, Bound<BX, BY>, ABS>>;
    static constexpr bool isdynamicbx = (BX == bound_t::Dynamic);
    static constexpr bool isdynamicby = (BY == bound_t::Dynamic);

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0]) : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1]) : BY);
        offset_t ix, iy;
        int8_t   fx, fy;
        utils_x::index(loc[0], size[0], &ix, &fx, bx, Z);
        utils_y::index(loc[1], size[1], &iy, &fy, by, Z);
        offset_t i = ix * stride[0] + iy * stride[1];
        int8_t   f = fx * fy;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = bound::get(inp, i, f);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0]) : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1]) : BY);
        offset_t ix, iy;
        int8_t   fx, fy;
        utils_x::index(loc[0], size[0], &ix, &fx, bx, Z);
        utils_y::index(loc[1], size[1], &iy, &fy, by, Z);
        offset_t i = ix * stride[0] + iy * stride[1];
        int8_t   f = fx * fy;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            bound::add(out, i, *inp, f);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count(
              scalar_t out      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0]) : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1]) : BY);
        offset_t ix, iy;
        int8_t   fx, fy;
        utils_x::index(loc[0], size[0], &ix, &fx, bx, Z);
        utils_y::index(loc[1], size[1], &iy, &fy, by, Z);
        offset_t i = ix * stride[0] + iy * stride[1];
        int8_t   f = fx * fy;

        bound::add(out, i, 1, f);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        for (offset_t c = 0; c < nc; ++c, out += osc) {
            *out     = static_cast<scalar_t>(0);
            out[osg] = static_cast<scalar_t>(0);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void hess(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        for (offset_t c = 0; c < nc; ++c, out += osc) {
            *out       = static_cast<scalar_t>(0);
            out[osg]   = static_cast<scalar_t>(0);
            out[osg*2] = static_cast<scalar_t>(0);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [2],
        const offset_t size         [2],
        const offset_t stride_out   [2],
        const offset_t stride_inp   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound        [2] = nullptr,
        const spline_t spline       [2] = nullptr
    )
    {
        gout[0]   = static_cast<scalar_t>(0);
        gout[osg] = static_cast<scalar_t>(0);
        self::push(out, ginp, loc, size, stride_out, nc, osc, isc, bound, spline);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push_backward(
              scalar_t out      [],
              scalar_t gout     [],
        const scalar_t inp      [],
        const scalar_t ginp     [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        gout[0]   = static_cast<scalar_t>(0);
        gout[osg] = static_cast<scalar_t>(0);
        self::pull(out, ginp, loc, size, stride, nc, osc, isc, bound, spline);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count_backward(
              scalar_t gout     [],
        const scalar_t inp      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t osg,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        gout[0]   = static_cast<scalar_t>(0);
        gout[osg] = static_cast<scalar_t>(0);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [2],
        const offset_t size         [2],
        const offset_t stride_out   [2],
        const offset_t stride_inp   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t gsc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound        [2] = nullptr,
        const spline_t spline       [2] = nullptr
    )
    {
        gout[0]   = static_cast<scalar_t>(0);
        gout[osg] = static_cast<scalar_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }
};


/***********************************************************************
 *
 *                               LINEAR
 *
 **********************************************************************/
template <bound::type BX, bound::type BY, bool ABS>
struct Kernels<Config<two, Spline<L,L>, Bound<BX, BY>, ABS>> {
    using utils_x = PushPullUtils<L, BX, ABS>;
    using utils_y = PushPullUtils<L, BY, ABS>;
    static constexpr int8_t negate      = static_cast<int8_t>(ABS ? 1 : -1);
    static constexpr bool   isdynamicbx = (BX == bound_t::Dynamic);
    static constexpr bool   isdynamicby = (BY == bound_t::Dynamic);

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0]) : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1]) : BY);
        offset_t ix[2], &ix0 = ix[0], &ix1 = ix[1], iy[2], &iy0 = iy[0], &iy1 = iy[1];
        reduce_t wx[2], &wx0 = wx[0], &wx1 = wx[1], wy[2], &wy0 = wy[0], &wy1 = wy[1];
        int8_t   fx[2], &fx0 = fx[0], &fx1 = fx[1], fy[2], &fy0 = fy[0], &fy1 = fy[1];
        utils_x::index(loc[0], size[0], ix, wx, fx, bx, L);
        utils_y::index(loc[1], size[1], iy, wy, fy, by, L);
        ix0 *= stride[0]; ix1 *= stride[0];
        iy0 *= stride[1]; iy1 *= stride[1];
        offset_t i00 = ix0 + iy0;
        offset_t i01 = ix0 + iy1;
        offset_t i10 = ix1 + iy0;
        offset_t i11 = ix1 + iy1;
        reduce_t w00 = wx0 * wy0;
        reduce_t w01 = wx0 * wy1;
        reduce_t w10 = wx1 * wy0;
        reduce_t w11 = wx1 * wy1;
        int8_t   f00 = fx0 * fy0;
        int8_t   f01 = fx0 * fy1;
        int8_t   f10 = fx1 * fy0;
        int8_t   f11 = fx1 * fy1;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, i00, f00) * w00
                    + bound::cget<reduce_t>(inp, i01, f01) * w01
                    + bound::cget<reduce_t>(inp, i10, f10) * w10
                    + bound::cget<reduce_t>(inp, i11, f11) * w11);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0]) : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1]) : BY);
        offset_t ix[2], &ix0 = ix[0], &ix1 = ix[1], iy[2], &iy0 = iy[0], &iy1 = iy[1];
        reduce_t wx[2], &wx0 = wx[0], &wx1 = wx[1], wy[2], &wy0 = wy[0], &wy1 = wy[1];
        int8_t   fx[2], &fx0 = fx[0], &fx1 = fx[1], fy[2], &fy0 = fy[0], &fy1 = fy[1];
        utils_x::index(loc[0], size[0], ix, wx, fx, bx, L);
        utils_y::index(loc[1], size[1], iy, wy, fy, by, L);

        ix0 *= stride[0]; ix1 *= stride[0];
        iy0 *= stride[1]; iy1 *= stride[1];
        offset_t i00 = ix0 + iy0;
        offset_t i01 = ix0 + iy1;
        offset_t i10 = ix1 + iy0;
        offset_t i11 = ix1 + iy1;
        reduce_t w00 = wx0 * wy0;
        reduce_t w01 = wx0 * wy1;
        reduce_t w10 = wx1 * wy0;
        reduce_t w11 = wx1 * wy1;
        int8_t   f00 = fx0 * fy0;
        int8_t   f01 = fx0 * fy1;
        int8_t   f10 = fx1 * fy0;
        int8_t   f11 = fx1 * fy1;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            bound::add(out, i00, val * w00, f00);
            bound::add(out, i01, val * w01, f01);
            bound::add(out, i10, val * w10, f10);
            bound::add(out, i11, val * w11, f11);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count(
              scalar_t out      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0]) : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1]) : BY);
        offset_t ix[2], &ix0 = ix[0], &ix1 = ix[1], iy[2], &iy0 = iy[0], &iy1 = iy[1];
        reduce_t wx[2], &wx0 = wx[0], &wx1 = wx[1], wy[2], &wy0 = wy[0], &wy1 = wy[1];
        int8_t   fx[2], &fx0 = fx[0], &fx1 = fx[1], fy[2], &fy0 = fy[0], &fy1 = fy[1];
        utils_x::index(loc[0], size[0], ix, wx, fx, bx, L);
        utils_y::index(loc[1], size[1], iy, wy, fy, by, L);

        ix0 *= stride[0]; ix1 *= stride[0];
        iy0 *= stride[1]; iy1 *= stride[1];
        offset_t i00 = ix0 + iy0;
        offset_t i01 = ix0 + iy1;
        offset_t i10 = ix1 + iy0;
        offset_t i11 = ix1 + iy1;
        reduce_t w00 = wx0 * wy0;
        reduce_t w01 = wx0 * wy1;
        reduce_t w10 = wx1 * wy0;
        reduce_t w11 = wx1 * wy1;
        int8_t   f00 = fx0 * fy0;
        int8_t   f01 = fx0 * fy1;
        int8_t   f10 = fx1 * fy0;
        int8_t   f11 = fx1 * fy1;

        bound::add(out, i00, w00, f00);
        bound::add(out, i01, w01, f01);
        bound::add(out, i10, w10, f10);
        bound::add(out, i11, w11, f11);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0]) : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1]) : BY);
        offset_t ix[2], &ix0 = ix[0], &ix1 = ix[1], iy[2], &iy0 = iy[0], &iy1 = iy[1];
        reduce_t wx[2], &wx0 = wx[0], &wx1 = wx[1], wy[2], &wy0 = wy[0], &wy1 = wy[1];
        int8_t   fx[2], &fx0 = fx[0], &fx1 = fx[1], fy[2], &fy0 = fy[0], &fy1 = fy[1];
        utils_x::index(loc[0], size[0], ix, wx, fx, bx, L);
        utils_y::index(loc[1], size[1], iy, wy, fy, by, L);

        ix0 *= stride[0]; ix1 *= stride[0];
        iy0 *= stride[1]; iy1 *= stride[1];
        offset_t i00 = ix0 + iy0;
        offset_t i01 = ix0 + iy1;
        offset_t i10 = ix1 + iy0;
        offset_t i11 = ix1 + iy1;
        int8_t   f00 = fx0 * fy0;
        int8_t   f01 = fx0 * fy1;
        int8_t   f10 = fx1 * fy0;
        int8_t   f11 = fx1 * fy1;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t v00 = bound::cget<reduce_t>(inp, i00, f00);
            reduce_t v01 = bound::cget<reduce_t>(inp, i01, f01);
            reduce_t v10 = bound::cget<reduce_t>(inp, i10, f10);
            reduce_t v11 = bound::cget<reduce_t>(inp, i11, f11);
            out[0] = static_cast<scalar_t>(
                (v10 * wy0 + v11 * wy1) +
                (v00 * wy0 + v01 * wy1) * negate
            );
            out[osg] = static_cast<scalar_t>(
                (v01 * wx0 + v11 * wx1) +
                (v10 * wx1 + v00 * wx0) * negate
            );
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void hess(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0]) : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1]) : BY);
        offset_t ix[2], &ix0 = ix[0], &ix1 = ix[1], iy[2], &iy0 = iy[0], &iy1 = iy[1];
        reduce_t wx[2], &wx0 = wx[0], &wx1 = wx[1], wy[2], &wy0 = wy[0], &wy1 = wy[1];
        int8_t   fx[2], &fx0 = fx[0], &fx1 = fx[1], fy[2], &fy0 = fy[0], &fy1 = fy[1];
        utils_x::index(loc[0], size[0], ix, wx, fx, bx, L);
        utils_y::index(loc[1], size[1], iy, wy, fy, by, L);

        ix0 *= stride[0]; ix1 *= stride[0];
        iy0 *= stride[1]; iy1 *= stride[1];
        offset_t i00 = ix0 + iy0;
        offset_t i01 = ix0 + iy1;
        offset_t i10 = ix1 + iy0;
        offset_t i11 = ix1 + iy1;
        int8_t   f00 = fx0 * fy0;
        int8_t   f01 = fx0 * fy1;
        int8_t   f10 = fx1 * fy0;
        int8_t   f11 = fx1 * fy1;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t v00 = bound::cget<reduce_t>(inp, i00, f00);
            reduce_t v01 = bound::cget<reduce_t>(inp, i01, f01);
            reduce_t v10 = bound::cget<reduce_t>(inp, i10, f10);
            reduce_t v11 = bound::cget<reduce_t>(inp, i11, f11);
            reduce_t gxy = (v00 + v11) + (v01 + v01) * negate;

            out[0]     = static_cast<scalar_t>(0);
            out[osg]   = static_cast<scalar_t>(0);
            out[osg*2] = static_cast<scalar_t>(gxy);

        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [2],
        const offset_t size         [2],
        const offset_t stride_out   [2],
        const offset_t stride_inp   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound        [2] = nullptr,
        const spline_t spline       [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0]) : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1]) : BY);
        offset_t ix[2], &ix0 = ix[0], &ix1 = ix[1], iy[2], &iy0 = iy[0], &iy1 = iy[1];
        reduce_t wx[2], &wx0 = wx[0], &wx1 = wx[1], wy[2], &wy0 = wy[0], &wy1 = wy[1];
        int8_t   fx[2], &fx0 = fx[0], &fx1 = fx[1], fy[2], &fy0 = fy[0], &fy1 = fy[1];
        utils_x::index(loc[0], size[0], ix, wx, fx, bx, L);
        utils_y::index(loc[1], size[1], iy, wy, fy, by, L);

        offset_t osx = stride_out[0], osy = stride_out[1], osz = stride_out[2];
        offset_t isx = stride_inp[0], isy = stride_inp[1], isz = stride_inp[2];
        reduce_t w00 = wx0 * wy0;
        reduce_t w01 = wx0 * wy1;
        reduce_t w10 = wx1 * wy0;
        reduce_t w11 = wx1 * wy1;
        int8_t   f00 = fx0 * fy0;
        int8_t   f01 = fx0 * fy1;
        int8_t   f10 = fx1 * fy0;
        int8_t   f11 = fx1 * fy1;
        offset_t i00 = ix0 * isx + iy0 * isy;
        offset_t i01 = ix0 * isx + iy1 * isy;
        offset_t i10 = ix1 * isx + iy0 * isy;
        offset_t i11 = ix1 * isx + iy1 * isy;
        offset_t o00 = ix0 * osx + iy0 * osy;
        offset_t o01 = ix0 * osx + iy1 * osy;
        offset_t o10 = ix1 * osx + iy0 * osy;
        offset_t o11 = ix1 * osx + iy1 * osy;

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, o00, gval * w00, f00);
            bound::add(out, o01, gval * w01, f01);
            bound::add(out, o10, gval * w10, f10);
            bound::add(out, o11, gval * w11, f11);
            // compute input spatial gradient
            reduce_t v00 = bound::cget<reduce_t>(inp, i00, f00);
            reduce_t v01 = bound::cget<reduce_t>(inp, i01, f01);
            reduce_t v10 = bound::cget<reduce_t>(inp, i10, f10);
            reduce_t v11 = bound::cget<reduce_t>(inp, i11, f11);
            accx += gval * ((v10 * wy0 + v11 * wy1) +
                            (v00 * wy0 + v01 * wy1) * negate);
            accy += gval * ((v01 * wx0 + v11 * wx1) +
                            (v00 * wx0 + v10 * wx1) * negate);
        }
        gout[0]   = static_cast<scalar_t>(accx);
        gout[osg] = static_cast<scalar_t>(accy);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push_backward(
              scalar_t out      [],
              scalar_t gout     [],
        const scalar_t inp      [],
        const scalar_t ginp     [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0]) : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1]) : BY);
        offset_t ix[2], &ix0 = ix[0], &ix1 = ix[1], iy[2], &iy0 = iy[0], &iy1 = iy[1];
        reduce_t wx[2], &wx0 = wx[0], &wx1 = wx[1], wy[2], &wy0 = wy[0], &wy1 = wy[1];
        int8_t   fx[2], &fx0 = fx[0], &fx1 = fx[1], fy[2], &fy0 = fy[0], &fy1 = fy[1];
        utils_x::index(loc[0], size[0], ix, wx, fx, bx, L);
        utils_y::index(loc[1], size[1], iy, wy, fy, by, L);

        offset_t sx = stride[0], sy = stride[1], sz = stride[2];
        ix0 *= sx; ix1 *= sx;
        iy0 *= sy; iy1 *= sy;

        offset_t i00 = ix0 + iy0;
        offset_t i01 = ix0 + iy1;
        offset_t i10 = ix1 + iy0;
        offset_t i11 = ix1 + iy1;
        reduce_t w00 = wx0 * wy0;
        reduce_t w01 = wx0 * wy1;
        reduce_t w10 = wx1 * wy0;
        reduce_t w11 = wx1 * wy1;
        int8_t   f00 = fx0 * fy0;
        int8_t   f01 = fx0 * fy1;
        int8_t   f10 = fx1 * fy0;
        int8_t   f11 = fx1 * fy1;

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            // pull incoming gradient
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(ginp, i00, f00) * w00
                    + bound::cget<reduce_t>(ginp, i01, f01) * w01
                    + bound::cget<reduce_t>(ginp, i10, f10) * w10
                    + bound::cget<reduce_t>(ginp, i11, f11) * w11);
            // compute input spatial gradient
            reduce_t val = static_cast<reduce_t>(*inp);
            reduce_t v00 = bound::cget<reduce_t>(ginp, i00, f00);
            reduce_t v01 = bound::cget<reduce_t>(ginp, i01, f01);
            reduce_t v10 = bound::cget<reduce_t>(ginp, i10, f10);
            reduce_t v11 = bound::cget<reduce_t>(ginp, i11, f11);
            accx += val * ((v10 * wy0 + v11 * wy1) +
                           (v00 * wy0 + v01 * wy1) * negate);
            accy += val * ((v01 * wx0 + v11 * wx1) +
                           (v10 * wx1 + v00 * wx0) * negate);
        }
        gout[0]   = static_cast<scalar_t>(accx);
        gout[osg] = static_cast<scalar_t>(accy);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count_backward(
              scalar_t gout     [],
        const scalar_t ginp     [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t osg,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0]) : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1]) : BY);
        offset_t ix[2], &ix0 = ix[0], &ix1 = ix[1], iy[2], &iy0 = iy[0], &iy1 = iy[1];
        reduce_t wx[2], &wx0 = wx[0], &wx1 = wx[1], wy[2], &wy0 = wy[0], &wy1 = wy[1];
        int8_t   fx[2], &fx0 = fx[0], &fx1 = fx[1], fy[2], &fy0 = fy[0], &fy1 = fy[1];
        utils_x::index(loc[0], size[0], ix, wx, fx, bx, L);
        utils_y::index(loc[1], size[1], iy, wy, fy, by, L);

        offset_t sx = stride[0], sy = stride[1], sz = stride[2];
        ix0 *= sx; ix1 *= sx;
        iy0 *= sy; iy1 *= sy;

        offset_t i00 = ix0 + iy0;
        offset_t i01 = ix0 + iy1;
        offset_t i10 = ix1 + iy0;
        offset_t i11 = ix1 + iy1;
        reduce_t w00 = wx0 * wy0;
        reduce_t w01 = wx0 * wy1;
        reduce_t w10 = wx1 * wy0;
        reduce_t w11 = wx1 * wy1;
        int8_t   f00 = fx0 * fy0;
        int8_t   f01 = fx0 * fy1;
        int8_t   f10 = fx1 * fy0;
        int8_t   f11 = fx1 * fy1;

        // compute input spatial gradient
        reduce_t v00 = bound::cget<reduce_t>(ginp, i00, f00);
        reduce_t v01 = bound::cget<reduce_t>(ginp, i01, f01);
        reduce_t v10 = bound::cget<reduce_t>(ginp, i10, f10);
        reduce_t v11 = bound::cget<reduce_t>(ginp, i11, f11);
        gout[0]   = static_cast<scalar_t>(
                        (v10 * wy0 + v11 * wy1) +
                        (v00 * wy0 + v01 * wy1) * negate);
        gout[osg] = static_cast<scalar_t>(
                        (v01 * wx0 + v11 * wx1) +
                        (v00 * wx0 + v10 * wx1) * negate);
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [2],
        const offset_t size         [2],
        const offset_t stride_out   [2],
        const offset_t stride_inp   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t gsc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound        [2] = nullptr,
        const spline_t spline       [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0]) : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1]) : BY);
        offset_t ix[2], &ix0 = ix[0], &ix1 = ix[1], iy[2], &iy0 = iy[0], &iy1 = iy[1];
        reduce_t wx[2], &wx0 = wx[0], &wx1 = wx[1], wy[2], &wy0 = wy[0], &wy1 = wy[1];
        int8_t   fx[2], &fx0 = fx[0], &fx1 = fx[1], fy[2], &fy0 = fy[0], &fy1 = fy[1];
        utils_x::index(loc[0], size[0], ix, wx, fx, bx, L);
        utils_y::index(loc[1], size[1], iy, wy, fy, by, L);

        offset_t osx = stride_out[0], osy = stride_out[1];
        offset_t isx = stride_inp[0], isy = stride_inp[1];
        // `inp` offsets must be built before `ix`/`iy` are scaled in place by
        // the *output* strides (the two tensors need not share a layout).
        offset_t j00 = ix0 * isx + iy0 * isy;
        offset_t j01 = ix0 * isx + iy1 * isy;
        offset_t j10 = ix1 * isx + iy0 * isy;
        offset_t j11 = ix1 * isx + iy1 * isy;
        ix0 *= osx; ix1 *= osx;
        iy0 *= osy; iy1 *= osy;

        offset_t i00 = ix0 + iy0;
        offset_t i01 = ix0 + iy1;
        offset_t i10 = ix1 + iy0;
        offset_t i11 = ix1 + iy1;
        int8_t   f00 = fx0 * fy0;
        int8_t   f01 = fx0 * fy1;
        int8_t   f10 = fx1 * fy0;
        int8_t   f11 = fx1 * fy1;

        // Derivative weights of the linear basis (`negate` folds in ABS),
        // matching what `gindex` returns for the general-order kernels.
        const reduce_t g0 = static_cast<reduce_t>(negate);
        const reduce_t g1 = static_cast<reduce_t>(1);

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t oval;
            reduce_t gvalx = static_cast<reduce_t>(ginp[0]);
            reduce_t gvaly = static_cast<reduce_t>(ginp[isg]);

            oval = (gvalx * wy0 + gvaly * wx0) * negate;
            bound::add(out, i00, oval, f00);

            oval = gvalx * wy1 * negate + gvaly * wx0;
            bound::add(out, i01, oval, f01);

            oval = + gvalx * wy0 + gvaly * wx1 * negate;
            bound::add(out, i10, oval, f10);

            oval = + gvalx * wy1 + gvaly * wx1;
            bound::add(out, i11, oval, f11);

            // d/d(loc). The pure second derivatives d2/dx2, d2/dy2 vanish for
            // a linear basis, but the *mixed* one does not, so the grid
            // gradient is not zero (it used to be hard-coded to zero here,
            // inherited from jitfields):
            //   dL/dx = gvaly * d2(pull)/dxdy,  dL/dy = gvalx * d2(pull)/dxdy
            const reduce_t cross =
                  g0 * g0 * bound::cget<reduce_t>(inp, j00, f00)
                + g0 * g1 * bound::cget<reduce_t>(inp, j01, f01)
                + g1 * g0 * bound::cget<reduce_t>(inp, j10, f10)
                + g1 * g1 * bound::cget<reduce_t>(inp, j11, f11);
            accx += gvaly * cross;
            accy += gvalx * cross;
        }

        gout[0]   = static_cast<scalar_t>(accx);
        gout[osg] = static_cast<scalar_t>(accy);
    }
};


/***********************************************************************
 *
 *                                 ANY
 *
 **********************************************************************/
template <spline::type IX, bound::type BX,
          spline::type IY, bound::type BY,
          bool ABS>
struct Kernels<Config<two, Spline<IX,IY>, Bound<BX, BY>, ABS>> {
    using utils_x = PushPullUtils<IX, BX, ABS>;
    using utils_y = PushPullUtils<IY, BY, ABS>;
    static constexpr int Nx = utils_x::bufsize;
    static constexpr int Ny = utils_y::bufsize;
    static constexpr bool isdynamicbx = (BX == bound_t::Dynamic);
    static constexpr bool isdynamicby = (BY == bound_t::Dynamic);
    static constexpr bool isdynamicsx = (IX == spline_t::Dynamic);
    static constexpr bool isdynamicsy = (IY == spline_t::Dynamic);

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0])  : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1])  : BY);
        spline_t sx = (isdynamicsx ? static_cast<spline_t>(spline[0]) : IX);
        spline_t sy = (isdynamicsy ? static_cast<spline_t>(spline[1]) : IY);
        // Precompute weights and indices
        offset_t ix[Nx], iy[Ny];
        reduce_t wx[Nx], wy[Ny];
        int8_t   fx[Nx], fy[Ny];
        offset_t nx = utils_x::index(loc[0], size[0], ix, wx, fx, bx, sx);
        offset_t ny = utils_y::index(loc[1], size[1], iy, wy, fy, by, sy);
        for (offset_t i = 0, st = stride[0]; i < (isdynamicsx ? nx : Nx); ++i)
            ix[i] *= st;
        for (offset_t i = 0, st = stride[1]; i < (isdynamicsy ? ny : Ny); ++i)
            iy[i] *= st;

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t i = 0; i < (isdynamicsx ? nx : Nx); ++i)
            for (offset_t j = 0; j < (isdynamicsy ? ny : Ny); ++j)
                acc += bound::cget<reduce_t>(
                    inp, ix[i] + iy[j], fx[i] * fy[j]
                ) * (wx[i] * wy[j]);
            *out = static_cast<scalar_t>(acc);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx  ? static_cast<bound_t>(bound[0])  : BX);
        bound_t  by = (isdynamicby  ? static_cast<bound_t>(bound[1])  : BY);
        spline_t sx = (isdynamicsx ? static_cast<spline_t>(spline[0]) : IX);
        spline_t sy = (isdynamicsy ? static_cast<spline_t>(spline[1]) : IY);
        // Precompute weights and indices
        offset_t ix[Nx], iy[Ny];
        reduce_t wx[Nx], wy[Ny];
        int8_t   fx[Nx], fy[Ny];
        offset_t nx = utils_x::index(loc[0], size[0], ix, wx, fx, bx, sx);
        offset_t ny = utils_y::index(loc[1], size[1], iy, wy, fy, by, sy);
        for (offset_t i = 0, st = stride[0]; i < (isdynamicsx ? nx : Nx); ++i)
            ix[i] *= st;
        for (offset_t i = 0, st = stride[1]; i < (isdynamicsy ? ny : Ny); ++i)
            iy[i] *= st;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            for (offset_t i = 0; i < (isdynamicsx ? nx : Nx); ++i)
            for (offset_t j = 0; j < (isdynamicsy ? ny : Ny); ++j)
                bound::add(out, ix[i] + iy[j], val * (wx[i] * wy[j]), fx[i] * fy[j]);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count(
              scalar_t out      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0])  : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1])  : BY);
        spline_t sx = (isdynamicsx ? static_cast<spline_t>(spline[0]) : IX);
        spline_t sy = (isdynamicsy ? static_cast<spline_t>(spline[1]) : IY);
        // Precompute weights and indices
        offset_t ix[Nx], iy[Ny];
        reduce_t wx[Nx], wy[Ny];
        int8_t   fx[Nx], fy[Ny];
        offset_t nx = utils_x::index(loc[0], size[0], ix, wx, fx, bx, sx);
        offset_t ny = utils_y::index(loc[1], size[1], iy, wy, fy, by, sy);
        for (offset_t i = 0, st = stride[0]; i < (isdynamicsx ? nx : Nx); ++i)
            ix[i] *= st;
        for (offset_t i = 0, st = stride[1]; i < (isdynamicsy ? ny : Ny); ++i)
            iy[i] *= st;

        for (offset_t i = 0; i < (isdynamicsx ? nx : Nx); ++i)
        for (offset_t j = 0; j < (isdynamicsy ? ny : Ny); ++j)
            bound::add(out, ix[i] + iy[j], wx[i] * wy[j], fx[i] * fy[j]);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0])  : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1])  : BY);
        spline_t sx = (isdynamicsx ? static_cast<spline_t>(spline[0]) : IX);
        spline_t sy = (isdynamicsy ? static_cast<spline_t>(spline[1]) : IY);
        // Precompute weights and indices
        offset_t ix[Nx], iy[Ny];
        reduce_t wx[Nx], wy[Ny];
        reduce_t gx[Nx], gy[Ny];
        int8_t   fx[Nx], fy[Ny];
        offset_t nx = utils_x::gindex(loc[0], size[0], ix, wx, gx, fx, bx, sx);
        offset_t ny = utils_y::gindex(loc[1], size[1], iy, wy, gy, fy, by, sy);
        for (offset_t i = 0, st = stride[0]; i < (isdynamicsx ? nx : Nx); ++i)
            ix[i] *= st;
        for (offset_t i = 0, st = stride[1]; i < (isdynamicsy ? ny : Ny); ++i)
            iy[i] *= st;

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            reduce_t accx = static_cast<reduce_t>(0);
            reduce_t accy = static_cast<reduce_t>(0);
            for (offset_t i = 0; i < (isdynamicsx ? nx : Nx); ++i)
            for (offset_t j = 0; j < (isdynamicsy ? ny : Ny); ++j)
            {
                reduce_t val = bound::cget<reduce_t>(inp, ix[i] + iy[j], fx[i] * fy[j]);
                accx += val * (gx[i] * wy[j]);
                accy += val * (wx[i] * gy[j]);
            }
            out[0]   = static_cast<scalar_t>(accx);
            out[osg] = static_cast<scalar_t>(accy);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void hess(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0])  : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1])  : BY);
        spline_t sx = (isdynamicsx ? static_cast<spline_t>(spline[0]) : IX);
        spline_t sy = (isdynamicsy ? static_cast<spline_t>(spline[1]) : IY);
        // Precompute weights and indices
        offset_t ix[Nx], iy[Ny];
        reduce_t wx[Nx], wy[Ny];
        reduce_t gx[Nx], gy[Ny];
        reduce_t hx[Nx], hy[Ny];
        int8_t   fx[Nx], fy[Ny];
        offset_t nx = utils_x::hindex(loc[0], size[0], ix, wx, gx, hx, fx, bx, sx);
        offset_t ny = utils_y::hindex(loc[1], size[1], iy, wy, gy, hy, fy, by, sy);
        for (offset_t i = 0, st = stride[0]; i < (isdynamicsx ? nx : Nx); ++i)
            ix[i] *= st;
        for (offset_t i = 0, st = stride[1]; i < (isdynamicsy ? ny : Ny); ++i)
            iy[i] *= st;

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            reduce_t accxx = static_cast<reduce_t>(0);
            reduce_t accyy = static_cast<reduce_t>(0);
            reduce_t accxy = static_cast<reduce_t>(0);
            for (offset_t i = 0; i < (isdynamicsx ? nx : Nx); ++i)
            for (offset_t j = 0; j < (isdynamicsy ? ny : Ny); ++j)
            {
                reduce_t val = bound::cget<reduce_t>(inp, ix[i] + iy[j], fx[i] * fy[j]);
                accxx += val * (hx[i] * wy[j]);
                accyy += val * (wx[i] * hy[j]);
                accxy += val * (gx[i] * gy[j]);
            }
            out[0]     = static_cast<scalar_t>(accxx);
            out[osg]   = static_cast<scalar_t>(accyy);
            out[osg*2] = static_cast<scalar_t>(accxy);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [2],
        const offset_t size         [2],
        const offset_t stride_out   [2],
        const offset_t stride_inp   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound        [2] = nullptr,
        const spline_t spline       [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0])  : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1])  : BY);
        spline_t sx = (isdynamicsx ? static_cast<spline_t>(spline[0]) : IX);
        spline_t sy = (isdynamicsy ? static_cast<spline_t>(spline[1]) : IY);
        // Precompute weights and indices
        offset_t ix[Nx], iy[Ny];
        reduce_t wx[Nx], wy[Ny];
        reduce_t gx[Nx], gy[Ny];
        int8_t   fx[Nx], fy[Ny];
        offset_t nx = utils_x::gindex(loc[0], size[0], ix, wx, gx, fx, bx, sx);
        offset_t ny = utils_y::gindex(loc[1], size[1], iy, wy, gy, fy, by, sy);
        offset_t osx = stride_out[0], osy = stride_out[1];
        offset_t isx = stride_inp[0], isy = stride_inp[1];

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            reduce_t gval  = static_cast<reduce_t>(*ginp);
            reduce_t accx1 = static_cast<reduce_t>(0);
            reduce_t accy1 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i < (isdynamicsx ? nx : Nx); ++i)
            {
                offset_t ixo = ix[i] * osx;
                offset_t ixi = ix[i] * isx;
                int8_t   ffx = fx[i];
                reduce_t wwx = wx[i];
                reduce_t ggx = gx[i];
                for (offset_t j = 0; j < (isdynamicsy ? ny : Ny); ++j)
                {
                    offset_t iyo = ixo + iy[j] * osy;
                    offset_t iyi = ixi + iy[j] * isy;
                    int8_t    ff = ffx * fy[j];
                    // push incoming gradient
                    bound::add(out, iyo, gval * (wwx * wy[j]), ff);
                    // compute input spatial gradient
                    reduce_t val = bound::cget<reduce_t>(inp, iyi, ff);
                    accx1 += val * (ggx * wy[j]);
                    accy1 += val * (wwx * gy[j]);
                }
            }
            accx += gval * accx1;
            accy += gval * accy1;
        }
        gout[0]   = static_cast<scalar_t>(accx);
        gout[osg] = static_cast<scalar_t>(accy);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push_backward(
              scalar_t out      [],
              scalar_t gout     [],
        const scalar_t inp      [],
        const scalar_t ginp     [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0])  : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1])  : BY);
        spline_t sx = (isdynamicsx ? static_cast<spline_t>(spline[0]) : IX);
        spline_t sy = (isdynamicsy ? static_cast<spline_t>(spline[1]) : IY);
        // Precompute weights and indices
        offset_t ix[Nx], iy[Ny];
        reduce_t wx[Nx], wy[Ny];
        reduce_t gx[Nx], gy[Ny];
        int8_t   fx[Nx], fy[Ny];
        offset_t nx = utils_x::gindex(loc[0], size[0], ix, wx, gx, fx, bx, sx);
        offset_t ny = utils_y::gindex(loc[1], size[1], iy, wy, gy, fy, by, sy);
        for (offset_t i = 0, st = stride[0]; i < (isdynamicsx ? nx : Nx); ++i)
            ix[i] *= st;
        for (offset_t i = 0, st = stride[1]; i < (isdynamicsy ? ny : Ny); ++i)
            iy[i] *= st;

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            reduce_t val   = static_cast<reduce_t>(*inp);
            reduce_t acc1  = static_cast<reduce_t>(0);
            reduce_t accx2 = static_cast<reduce_t>(0);
            reduce_t accy2 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i < (isdynamicsx ? nx : Nx); ++i)
            for (offset_t j = 0; j < (isdynamicsy ? ny : Ny); ++j)
            {
                reduce_t gval = bound::cget<reduce_t>(ginp, ix[i] + iy[j], fx[i] * fy[j]);
                // pull incoming gradient
                acc1 += gval * (wx[i] * wy[j]);
                // compute incoming gradient spatial gradient
                accx2 += gval * (gx[i] * wy[j]);
                accy2 += gval * (wx[i] * gy[j]);
            }
            *out = static_cast<scalar_t>(acc1);
            accx += val * accx2;
            accy += val * accy2;
        }
        gout[0]   = static_cast<scalar_t>(accx);
        gout[osg] = static_cast<scalar_t>(accy);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count_backward(
              scalar_t gout     [],
        const scalar_t ginp     [],
        const reduce_t loc      [2],
        const offset_t size     [2],
        const offset_t stride   [2],
              offset_t osg,
        const bound_t  bound    [2] = nullptr,
        const spline_t spline   [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0])  : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1])  : BY);
        spline_t sx = (isdynamicsx ? static_cast<spline_t>(spline[0]) : IX);
        spline_t sy = (isdynamicsy ? static_cast<spline_t>(spline[1]) : IY);
        // Precompute weights and indices
        offset_t ix[Nx], iy[Ny];
        reduce_t wx[Nx], wy[Ny];
        reduce_t gx[Nx], gy[Ny];
        int8_t   fx[Nx], fy[Ny];
        offset_t nx = utils_x::gindex(loc[0], size[0], ix, wx, gx, fx, bx, sx);
        offset_t ny = utils_y::gindex(loc[1], size[1], iy, wy, gy, fy, by, sy);
        for (offset_t i = 0, s = stride[0]; i < (isdynamicsx ? nx : Nx); ++i)
            ix[i] *= s;
        for (offset_t i = 0, s = stride[1]; i < (isdynamicsy ? ny : Ny); ++i)
            iy[i] *= s;

        // compute input spatial gradient
        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        for (offset_t i = 0; i < (isdynamicsx ? nx : Nx); ++i)
        for (offset_t j = 0; j < (isdynamicsy ? ny : Ny); ++j)
        {
            reduce_t val = bound::cget<reduce_t>(ginp, ix[i] + iy[j], fx[i] * fy[j]);
            accx += val * (gx[i] * wy[j]);
            accy += val * (wx[i] * gy[j]);
        }
        gout[0]   = static_cast<scalar_t>(accx);
        gout[osg] = static_cast<scalar_t>(accy);
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [2],
        const offset_t size         [2],
        const offset_t stride_out   [2],
        const offset_t stride_inp   [2],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t gsc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound        [2] = nullptr,
        const spline_t spline       [2] = nullptr
    )
    {
        bound_t  bx = (isdynamicbx ? static_cast<bound_t>(bound[0])  : BX);
        bound_t  by = (isdynamicby ? static_cast<bound_t>(bound[1])  : BY);
        spline_t sx = (isdynamicsx ? static_cast<spline_t>(spline[0]) : IX);
        spline_t sy = (isdynamicsy ? static_cast<spline_t>(spline[1]) : IY);
        // Precompute weights and indices
        offset_t ix[Nx], iy[Ny];
        reduce_t wx[Nx], wy[Ny];
        reduce_t gx[Nx], gy[Ny];
        reduce_t hx[Nx], hy[Ny];
        int8_t   fx[Nx], fy[Ny];
        offset_t nx = utils_x::hindex(loc[0], size[0], ix, wx, gx, hx, fx, bx, sx);
        offset_t ny = utils_y::hindex(loc[1], size[1], iy, wy, gy, hy, fy, by, sy);
        offset_t osx = stride_out[0], osy = stride_out[1];
        offset_t isx = stride_inp[0], isy = stride_inp[1];

        reduce_t accx = static_cast<reduce_t>(0);
        reduce_t accy = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t gvalx = static_cast<reduce_t>(ginp[0]);
            reduce_t gvaly = static_cast<reduce_t>(ginp[isg]);
            reduce_t accxx1 = static_cast<reduce_t>(0);
            reduce_t accxy1 = static_cast<reduce_t>(0);
            reduce_t accyy1 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i < (isdynamicsx ? nx : Nx); ++i)
            for (offset_t j = 0; j < (isdynamicsy ? ny : Ny); ++j)
            {
                // push incoming gradient
                reduce_t oval = gvalx * (gx[i] * wy[j]) + gvaly * (wx[i] * gy[j]);
                bound::add(out, ix[i] * osx + iy[j] * osy, oval, fx[i] * fy[j]);
                // compute input spatial hessian
                reduce_t ival = bound::cget<reduce_t>(inp, ix[i] * isx + iy[j] * isy, fx[i] * fy[j]);
                accxx1 += ival * hx[i] * wy[j];
                accyy1 += ival * wx[i] * hy[j];
                accxy1 += ival * gx[i] * gy[j];
            }
            accx += gvalx * accxx1 + gvaly * accxy1;
            accy += gvaly * accyy1 + gvalx * accxy1;
        }
        gout[0]   = static_cast<scalar_t>(accx);
        gout[osg] = static_cast<scalar_t>(accy);
    }
};

FF_NAMESPACE_END(pushpull)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_PUSHPULL_2D
