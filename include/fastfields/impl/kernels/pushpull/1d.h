/***********************************************************************
 *
 *                                  1D
 *
 **********************************************************************/
#ifndef FF_PUSHPULL_1D
#define FF_PUSHPULL_1D
#include "../cuda_switch.h"
#include "../spline.h"
#include "../bounds.h"
#include "utils.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(pushpull)

/***********************************************************************
 *
 *                               NEAREST
 *
 **********************************************************************/
template <bound_t B, bool ABS>
struct PushPull<PushPullConfig<one, Spline<Z>, Bound<B>, ABS>> {
    using utils = PushPullUtils<Z, B, ABS>;
    using self  = PushPull<PushPullConfig<one, Spline<Z>, Bound<B>, ABS>>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    i;
        int8_t      f;
        utils::index(loc[0], size[0], &i, nullptr, &f, b, Z);
        i *= stride[0];

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = bound::get(inp, i, f);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    i;
        int8_t      f;
        utils::index(loc[0], size[0], &i, nullptr, &f, b, Z);
        i *= stride[0];

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            bound::add(out, i, *inp, f);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count(
              scalar_t out      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    i;
        int8_t      f;
        utils::index(loc[0], size[0], &i, nullptr, &f, b, Z);
        i *= stride[0];

        bound::add(out, i, 1, f);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void hess(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [1],
        const offset_t size         [1],
        const offset_t stride_out   [1],
        const offset_t stride_inp   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        *gout = static_cast<scalar_t>(0);
        self::push(out, ginp, loc, size, stride_out, nc, osc, isg, bound, spline);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push_backward(
              scalar_t out      [],
              scalar_t gout     [],
        const scalar_t inp      [],
        const scalar_t ginp     [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        *gout = static_cast<scalar_t>(0);
        self::pull(out, ginp, loc, size, stride, nc, osc, isc, bound, spline);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count_backward(
              scalar_t gout     [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t osg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        *gout = static_cast<scalar_t>(0);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [1],
        const offset_t size         [1],
        const offset_t stride_out   [1],
        const offset_t stride_inp   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t gsc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        *gout = static_cast<scalar_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }
};

/***********************************************************************
 *
 *                               LINEAR
 *
 **********************************************************************/
template <bound::type B, bool ABS>
struct PushPull<PushPullConfig<one, Spline<L>, Bound<B>, ABS>> {
    using utils = PushPullUtils<L, B, ABS>;
    static const int8_t negate = static_cast<int8_t>(ABS ? 1 : -1);

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[2], &x0 = x[0], &x1 = x[1];
        reduce_t    w[2], &w0 = w[0], &w1 = w[1];
        int8_t      f[2], &f0 = f[0], &f1 = f[1];
        utils::index(loc[0], size[0], x, w, f, b, L);
        x0 *= stride[0];
        x1 *= stride[0];

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                  static_cast<reduce_t>(bound::get(inp, x0, f0)) * w0
                + static_cast<reduce_t>(bound::get(inp, x1, f1)) * w1
            );
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[2], &x0 = x[0], &x1 = x[1];
        reduce_t    w[2], &w0 = w[0], &w1 = w[1];
        int8_t      f[2], &f0 = f[0], &f1 = f[1];
        utils::index(loc[0], size[0], x, w, f, b, L);
        x[0] *= stride[0];
        x[1] *= stride[0];

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            bound::add(out, x0, val * w0, f0);
            bound::add(out, x1, val * w1, f1);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count(
              scalar_t out      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[2], &x0 = x[0], &x1 = x[1];
        reduce_t    w[2], &w0 = w[0], &w1 = w[1];
        int8_t      f[2], &f0 = f[0], &f1 = f[1];
        utils::index(loc[0], size[0], x, w, f, b, L);
        x[0] *= stride[0];
        x[1] *= stride[0];

        bound::add(out, x0, w0, f0);
        bound::add(out, x1, w1, f1);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[2], &x0 = x[0], &x1 = x[1];
        reduce_t    w[2], &w0 = w[0], &w1 = w[1];
        int8_t      f[2], &f0 = f[0], &f1 = f[1];
        utils::index(loc[0], size[0], x, w, f, b, L);
        x0 *= stride[0];
        x1 *= stride[0];

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                bound::cget<reduce_t>(inp, x1, f1) +
                bound::cget<reduce_t>(inp, x0, f0) * negate
            );
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void hess(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [1],
        const offset_t size         [1],
        const offset_t stride_out   [1],
        const offset_t stride_inp   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[2], &x0 = x[0], &x1 = x[1];
        reduce_t    w[2], &w0 = w[0], &w1 = w[1];
        int8_t      f[2], &f0 = f[0], &f1 = f[1];
        utils::index(loc[0], size[0], x, w, f, b, L);
        offset_t osx = stride_out[0], isx = stride_inp[0];

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, x0 * osx, gval * w0, f0);
            bound::add(out, x1 * osx, gval * w1, f1);
            // compute input spatial gradient
            acc += gval * (
                bound::cget<reduce_t>(inp, x1 * isx, f1) +
                bound::cget<reduce_t>(inp, x0 * isx, f0) * negate
            );
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push_backward(
              scalar_t out      [],
              scalar_t gout     [],
        const scalar_t inp      [],
        const scalar_t ginp     [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[2], &x0 = x[0], &x1 = x[1];
        reduce_t    w[2], &w0 = w[0], &w1 = w[1];
        int8_t      f[2], &f0 = f[0], &f1 = f[1];
        utils::index(loc[0], size[0], x, w, f, b, L);
        x0 *= stride[0];
        x1 *= stride[0];

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            // pull incoming gradient
            *out = static_cast<scalar_t>(
                  bound::cget<reduce_t>(ginp, x0, f0) * w0
                + bound::cget<reduce_t>(ginp, x1, f1) * w1
            );
            // compute input spatial gradient
            reduce_t val = static_cast<reduce_t>(*inp);
            acc += val * (
                bound::cget<reduce_t>(ginp, x1, f1) +
                bound::cget<reduce_t>(ginp, x0, f0) * negate
            );
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count_backward(
              scalar_t gout     [],
        const scalar_t ginp     [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t osg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[2], &x0 = x[0], &x1 = x[1];
        reduce_t    w[2], &w0 = w[0], &w1 = w[1];
        int8_t      f[2], &f0 = f[0], &f1 = f[1];
        utils::index(loc[0], size[0], x, w, f, b, L);
        x0 *= stride[0];
        x1 *= stride[0];

        // compute input spatial gradient
        *gout = static_cast<scalar_t>(
            bound::cget<reduce_t>(ginp, x1, f1) +
            bound::cget<reduce_t>(ginp, x0, f0) * negate);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [1],
        const offset_t size         [1],
        const offset_t stride_out   [1],
        const offset_t stride_inp   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t gsc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        *gout = static_cast<scalar_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc)
            *out = static_cast<scalar_t>(0);
    }
};

/***********************************************************************
 *
 *                               QUADRATIC
 *
 **********************************************************************/
template <bound::type B, bool ABS>
struct PushPull<PushPullConfig<one, Spline<Q>, Bound<B>, ABS>> {
    using utils = PushPullUtils<Q, B, ABS>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[3], &x0 = x[0], &x1 = x[1], &x2 = x[2];
        reduce_t    w[3], &w0 = w[0], &w1 = w[1], &w2 = w[2];
        int8_t      f[3], &f0 = f[0], &f1 = f[1], &f2 = f[2];
        utils::index(loc[0], size[0], x, w, f, b, Q);
        x0 *= stride[0];
        x1 *= stride[0];
        x2 *= stride[0];

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, x0, f0) * w0
                    + bound::cget<reduce_t>(inp, x1, f1) * w1
                    + bound::cget<reduce_t>(inp, x2, f2) * w2
            );
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[3], &x0 = x[0], &x1 = x[1], &x2 = x[2];
        reduce_t    w[3], &w0 = w[0], &w1 = w[1], &w2 = w[2];
        int8_t      f[3], &f0 = f[0], &f1 = f[1], &f2 = f[2];
        utils::index(loc[0], size[0], x, w, f, b, Q);
        x0 *= stride[0];
        x1 *= stride[0];
        x2 *= stride[0];

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            bound::add(out, x0, val * w0, f0);
            bound::add(out, x1, val * w1, f1);
            bound::add(out, x2, val * w2, f2);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count(
              scalar_t out      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[3], &x0 = x[0], &x1 = x[1], &x2 = x[2];
        reduce_t    w[3], &w0 = w[0], &w1 = w[1], &w2 = w[2];
        int8_t      f[3], &f0 = f[0], &f1 = f[1], &f2 = f[2];
        utils::index(loc[0], size[0], x, w, f, b, Q);
        x0 *= stride[0];
        x1 *= stride[0];
        x2 *= stride[0];

        bound::add(out, x0, w0, f0);
        bound::add(out, x1, w1, f1);
        bound::add(out, x2, w2, f2);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[3], &x0 = x[0], &x1 = x[1], &x2 = x[2];
        reduce_t    w[3], &w0 = w[0], &w1 = w[1], &w2 = w[2];
        reduce_t    g[3], &g0 = g[0], &g1 = g[1], &g2 = g[2];
        int8_t      f[3], &f0 = f[0], &f1 = f[1], &f2 = f[2];
        utils::gindex(loc[0], size[0], x, w, f, g, b, Q);
        x0 *= stride[0];
        x1 *= stride[0];
        x2 *= stride[0];

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                  bound::cget<reduce_t>(inp, x0, f0) * g0
                + bound::cget<reduce_t>(inp, x1, f1) * g1
                + bound::cget<reduce_t>(inp, x2, f2) * g2
            );
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void hess(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[3], &x0 = x[0], &x1 = x[1], &x2 = x[2];
        reduce_t    w[3], &w0 = w[0], &w1 = w[1], &w2 = w[2];
        reduce_t    g[3], &g0 = g[0], &g1 = g[1], &g2 = g[2];
        reduce_t    h[3], &h0 = h[0], &h1 = h[1], &h2 = h[2];
        int8_t      f[3], &f0 = f[0], &f1 = f[1], &f2 = f[2];
        utils::hindex(loc[0], size[0], x, w, g, h, f, b, Q);
        x0 *= stride[0];
        x1 *= stride[0];
        x2 *= stride[0];

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                  bound::cget<reduce_t>(inp, x0, f0) * h0
                + bound::cget<reduce_t>(inp, x1, f1) * h1
                + bound::cget<reduce_t>(inp, x2, f2) * h2
            );
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [1],
        const offset_t size         [1],
        const offset_t stride_out   [1],
        const offset_t stride_inp   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[3], &x0 = x[0], &x1 = x[1], &x2 = x[2];
        reduce_t    w[3], &w0 = w[0], &w1 = w[1], &w2 = w[2];
        reduce_t    g[3], &g0 = g[0], &g1 = g[1], &g2 = g[2];
        int8_t      f[3], &f0 = f[0], &f1 = f[1], &f2 = f[2];
        utils::gindex(loc[0], size[0], x, w, g, f, b, Q);
        offset_t osx = stride_out[0], isx = stride_inp[0];

        reduce_t acc = 0;
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, x0 * osx, gval * w0, f0);
            bound::add(out, x1 * osx, gval * w1, f1);
            bound::add(out, x2 * osx, gval * w2, f2);
            // compute input spatial gradient
            acc += gval * (bound::cget<reduce_t>(inp, x0 * isx, f0) * g0
                         + bound::cget<reduce_t>(inp, x1 * isx, f1) * g1
                         + bound::cget<reduce_t>(inp, x2 * isx, f2) * g2);
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push_backward(
              scalar_t out      [],
              scalar_t gout     [],
        const scalar_t inp      [],
        const scalar_t ginp     [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[3], &x0 = x[0], &x1 = x[1], &x2 = x[2];
        reduce_t    w[3], &w0 = w[0], &w1 = w[1], &w2 = w[2];
        reduce_t    g[3], &g0 = g[0], &g1 = g[1], &g2 = g[2];
        int8_t      f[3], &f0 = f[0], &f1 = f[1], &f2 = f[2];
        utils::gindex(loc[0], size[0], x, w, g, f, b, Q);
        x0 *= stride[0];
        x1 *= stride[0];
        x2 *= stride[0];

        reduce_t acc = 0;
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            reduce_t ginp0 = bound::cget<reduce_t>(ginp, x0, f0);
            reduce_t ginp1 = bound::cget<reduce_t>(ginp, x1, f1);
            reduce_t ginp2 = bound::cget<reduce_t>(ginp, x2, f2);
            // pull incoming gradient
            *out = static_cast<scalar_t>(ginp0 * w0 + ginp1 * w1 + ginp2 * w2);
            // compute incoming gradient spatial gradient
            reduce_t val = static_cast<reduce_t>(*inp);
            acc += val * (ginp0 * g0 + ginp1 * g1 + ginp2 * g2);
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count_backward(
              scalar_t gout     [],
        const scalar_t ginp     [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t osg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[3], &x0 = x[0], &x1 = x[1], &x2 = x[2];
        reduce_t    w[3], &w0 = w[0], &w1 = w[1], &w2 = w[2];
        reduce_t    g[3], &g0 = g[0], &g1 = g[1], &g2 = g[2];
        int8_t      f[3], &f0 = f[0], &f1 = f[1], &f2 = f[2];
        utils::gindex(loc[0], size[0], x, w, g, f, b, Q);
        x0 *= stride[0];
        x1 *= stride[0];
        x2 *= stride[0];

        // compute input spatial gradient
        *gout = bound::cget<reduce_t>(ginp, x0, f0) * g0
              + bound::cget<reduce_t>(ginp, x1, f1) * g1
              + bound::cget<reduce_t>(ginp, x2, f2) * g2;
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [1],
        const offset_t size         [1],
        const offset_t stride_out   [1],
        const offset_t stride_inp   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t gsc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[3], &x0 = x[0], &x1 = x[1], &x2 = x[2];
        reduce_t    w[3], &w0 = w[0], &w1 = w[1], &w2 = w[2];
        reduce_t    g[3], &g0 = g[0], &g1 = g[1], &g2 = g[2];
        reduce_t    h[3], &h0 = h[0], &h1 = h[1], &h2 = h[2];
        int8_t      f[3], &f0 = f[0], &f1 = f[1], &f2 = f[2];
        utils::hindex(loc[0], size[0], x, w, g, h, f, b, Q);
        offset_t osx = stride_out[0], isx = stride_inp[0];

        reduce_t acc = 0;
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, x0 * osx, static_cast<scalar_t>(gval * g0), f0);
            bound::add(out, x1 * osx, static_cast<scalar_t>(gval * g1), f1);
            bound::add(out, x2 * osx, static_cast<scalar_t>(gval * g2), f2);
            // compute input spatial hessian
            acc += gval * (static_cast<reduce_t>(bound::get(inp, x0 * isx, f0)) * h0
                         + static_cast<reduce_t>(bound::get(inp, x1 * isx, f1)) * h1
                         + static_cast<reduce_t>(bound::get(inp, x2 * isx, f2)) * h2);
        }
        *gout = static_cast<scalar_t>(acc);
    }
};

/***********************************************************************
 *
 *                               CUBIC
 *
 **********************************************************************/
template <bound::type B, bool ABS>
struct PushPull<PushPullConfig<one, Spline<C>, Bound<B>, ABS>> {
    using utils = PushPullUtils<C, B, ABS>;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[4], &x0 = x[0], &x1 = x[1], &x2 = x[2], &x3 = x[3];
        reduce_t    w[4], &w0 = w[0], &w1 = w[1], &w2 = w[2], &w3 = w[3];
        int8_t      f[4], &f0 = f[0], &f1 = f[1], &f2 = f[2], &f3 = f[3];
        utils::index(loc[0], size[0], x, w, f, b, C);
        x0 *= stride[0];
        x1 *= stride[0];
        x2 *= stride[0];
        x3 *= stride[0];

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, x0, f0) * w0
                    + bound::cget<reduce_t>(inp, x1, f1) * w1
                    + bound::cget<reduce_t>(inp, x2, f2) * w2
                    + bound::cget<reduce_t>(inp, x3, f3) * w3
            );
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[4], &x0 = x[0], &x1 = x[1], &x2 = x[2], &x3 = x[3];
        reduce_t    w[4], &w0 = w[0], &w1 = w[1], &w2 = w[2], &w3 = w[3];
        int8_t      f[4], &f0 = f[0], &f1 = f[1], &f2 = f[2], &f3 = f[3];
        utils::index(loc[0], size[0], x, w, f, b, C);
        x0 *= stride[0];
        x1 *= stride[0];
        x2 *= stride[0];
        x3 *= stride[0];

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            bound::add(out, x0, val * w0, f0);
            bound::add(out, x1, val * w1, f1);
            bound::add(out, x2, val * w2, f2);
            bound::add(out, x3, val * w3, f3);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count(
              scalar_t out      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[4], &x0 = x[0], &x1 = x[1], &x2 = x[2], &x3 = x[3];
        reduce_t    w[4], &w0 = w[0], &w1 = w[1], &w2 = w[2], &w3 = w[3];
        int8_t      f[4], &f0 = f[0], &f1 = f[1], &f2 = f[2], &f3 = f[3];
        utils::index(loc[0], size[0], x, w, f, b, Q);
        x0 *= stride[0];
        x1 *= stride[0];
        x2 *= stride[0];
        x3 *= stride[0];

        bound::add(out, x0, w0, f0);
        bound::add(out, x1, w1, f1);
        bound::add(out, x2, w2, f2);
        bound::add(out, x3, w3, f3);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[4], &x0 = x[0], &x1 = x[1], &x2 = x[2], &x3 = x[3];
        reduce_t    w[4], &w0 = w[0], &w1 = w[1], &w2 = w[2], &w3 = w[3];
        reduce_t    g[4], &g0 = g[0], &g1 = g[1], &g2 = g[2], &g3 = g[3];
        int8_t      f[4], &f0 = f[0], &f1 = f[1], &f2 = f[2], &f3 = f[3];
        utils::gindex(loc[0], size[0], x, w, g, f, b, C);
        x0 *= stride[0];
        x1 *= stride[0];
        x2 *= stride[0];
        x3 *= stride[0];

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, x0, f0) * g0
                    + bound::cget<reduce_t>(inp, x1, f1) * g1
                    + bound::cget<reduce_t>(inp, x2, f2) * g2
                    + bound::cget<reduce_t>(inp, x3, f3) * g
            );
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void hess(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[4], &x0 = x[0], &x1 = x[1], &x2 = x[2], &x3 = x[3];
        reduce_t    w[4], &w0 = w[0], &w1 = w[1], &w2 = w[2], &w3 = w[3];
        reduce_t    g[4], &g0 = g[0], &g1 = g[1], &g2 = g[2], &g3 = g[3];
        reduce_t    h[4], &h0 = g[0], &h1 = g[1], &h2 = h[2], &h3 = h[3];
        int8_t      f[4], &f0 = f[0], &f1 = f[1], &f2 = f[2], &f3 = f[3];
        utils::hindex(loc[0], size[0], x, w, g, h, f, b, C);
        x0 *= stride[0];
        x1 *= stride[0];
        x2 *= stride[0];
        x3 *= stride[0];

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
            *out = static_cast<scalar_t>(
                      bound::cget<reduce_t>(inp, x0, f0) * h0
                    + bound::cget<reduce_t>(inp, x1, f1) * h1
                    + bound::cget<reduce_t>(inp, x2, f2) * h2
                    + bound::cget<reduce_t>(inp, x3, f3) * h3
            );
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [1],
        const offset_t size         [1],
        const offset_t stride_out   [1],
        const offset_t stride_inp   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[4], &x0 = x[0], &x1 = x[1], &x2 = x[2], &x3 = x[3];
        reduce_t    w[4], &w0 = w[0], &w1 = w[1], &w2 = w[2], &w3 = w[3];
        reduce_t    g[4], &g0 = g[0], &g1 = g[1], &g2 = g[2], &g3 = g[3];
        int8_t      f[4], &f0 = f[0], &f1 = f[1], &f2 = f[2], &f3 = f[3];
        utils::gindex(loc[0], size[0], x, w, g, f, b, C);
        offset_t osx = stride_out[0];
        offset_t isx = stride_inp[0];

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, x0 * osx, gval * w0, f0);
            bound::add(out, x1 * osx, gval * w1, f1);
            bound::add(out, x2 * osx, gval * w2, f2);
            bound::add(out, x3 * osx, gval * w3, f3);
            // compute input spatial gradient
            acc += gval * (bound::cget<reduce_t>(inp, x0 * isx, f0) * g0
                         + bound::cget<reduce_t>(inp, x1 * isx, f1) * g1
                         + bound::cget<reduce_t>(inp, x2 * isx, f2) * g2
                         + bound::cget<reduce_t>(inp, x3 * isx, f3) * g3);
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push_backward(
              scalar_t out      [],
              scalar_t gout     [],
        const scalar_t inp      [],
        const scalar_t ginp     [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[4], &x0 = x[0], &x1 = x[1], &x2 = x[2], &x3 = x[3];
        reduce_t    w[4], &w0 = w[0], &w1 = w[1], &w2 = w[2], &w3 = w[3];
        reduce_t    g[4], &g0 = g[0], &g1 = g[1], &g2 = g[2], &g3 = g[3];
        int8_t      f[4], &f0 = f[0], &f1 = f[1], &f2 = f[2], &f3 = f[3];
        utils::gindex(loc[0], size[0], x, w, g, f, b, C);
        x0 *= stride[0];
        x1 *= stride[0];
        x2 *= stride[0];
        x3 *= stride[0];

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            reduce_t ginp0 = bound::cget<reduce_t>(ginp, x0, f0);
            reduce_t ginp1 = bound::cget<reduce_t>(ginp, x1, f1);
            reduce_t ginp2 = bound::cget<reduce_t>(ginp, x2, f2);
            reduce_t ginp3 = bound::cget<reduce_t>(ginp, x3, f3);
            // pull incoming gradient
            *out = static_cast<scalar_t>(ginp0 * w0 + ginp1 * w1 +
                                         ginp2 * w2 + ginp3 * w3);
            // compute incoming gradient spatial gradient
            reduce_t val = static_cast<reduce_t>(*inp);
            acc += val * (ginp0 * g0 + ginp1 * g1 + ginp2 * g2 + ginp3 * g3);
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count_backward(
              scalar_t gout     [],
        const scalar_t ginp     [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t osg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[4], &x0 = x[0], &x1 = x[1], &x2 = x[2], &x3 = x[3];
        reduce_t    w[4], &w0 = w[0], &w1 = w[1], &w2 = w[2], &w3 = w[3];
        reduce_t    g[4], &g0 = g[0], &g1 = g[1], &g2 = g[2], &g3 = g[3];
        int8_t      f[4], &f0 = f[0], &f1 = f[1], &f2 = f[2], &f3 = f[3];
        utils::gindex(loc[0], size[0], x, w, g, f, b, C);
        x0 *= stride[0];
        x1 *= stride[0];
        x2 *= stride[0];
        x3 *= stride[0];

        // compute input spatial gradient
        *gout = bound::cget<reduce_t>(ginp, x0, f0) * g0
              + bound::cget<reduce_t>(ginp, x1, f1) * g1
              + bound::cget<reduce_t>(ginp, x2, f2) * g2
              + bound::cget<reduce_t>(ginp, x3, f3) * g3;
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [1],
        const offset_t size         [1],
        const offset_t stride_out   [1],
        const offset_t stride_inp   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t gsc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t     b = (B == bound_t::Dynamic  ? bound[0] : B);
        offset_t    x[4], &x0 = x[0], &x1 = x[1], &x2 = x[2], &x3 = x[3];
        reduce_t    w[4], &w0 = w[0], &w1 = w[1], &w2 = w[2], &w3 = w[3];
        reduce_t    g[4], &g0 = g[0], &g1 = g[1], &g2 = g[2], &g3 = g[3];
        reduce_t    h[4], &h0 = g[0], &h1 = g[1], &h2 = h[2], &h3 = h[3];
        int8_t      f[4], &f0 = f[0], &f1 = f[1], &f2 = f[2], &f3 = f[3];
        utils::hindex(loc[0], size[0], x, w, g, h, f, b, C);
        offset_t osx = stride_out[0], isx = stride_inp[0];

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            // push incoming gradient
            reduce_t gval = static_cast<reduce_t>(*ginp);
            bound::add(out, x0 * osx, gval * g0, f0);
            bound::add(out, x1 * osx, gval * g1, f1);
            bound::add(out, x2 * osx, gval * g2, f2);
            bound::add(out, x3 * osx, gval * g3, f3);
            // compute input spatial hessian
            acc += gval * (bound::cget<reduce_t>(inp, x0 * isx, f0) * h0
                         + bound::cget<reduce_t>(inp, x1 * isx, f1) * h1
                         + bound::cget<reduce_t>(inp, x2 * isx, f2) * h2
                         + bound::cget<reduce_t>(inp, x3 * isx, f3) * h3);
        }
        *gout = static_cast<scalar_t>(acc);
    }
};

/***********************************************************************
 *
 *                                 ANY
 *
 **********************************************************************/
template <spline::type I, bound::type B, bool ABS>
struct PushPull<PushPullConfig<one, Spline<I>, Bound<B>, ABS>> {
    using utils = PushPullUtils<I, B, ABS>;
    static constexpr int N = utils::bufsize;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t  b = (B == bound_t::Dynamic  ? bound[0]  : B);
        spline_t s = (I == spline_t::Dynamic ? spline[0] : I);

        // Precompute weights and indices
        offset_t ix[N];
        reduce_t wx[N];
        int8_t   fx[N];
        offset_t length = utils::index(loc[0], size[0], ix, wx, fx, b, s);
        for (offset_t i = 0, st = stride[0]; i < length; ++i)
            ix[i] *= st;

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t i = 0; i < length; ++i)
                acc += bound::cget<reduce_t>(inp, ix[i], fx[i]) * wx[i];
            *out = static_cast<scalar_t>(acc);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t  b = (B == bound_t::Dynamic  ? bound[0]  : B);
        spline_t s = (I == spline_t::Dynamic ? spline[0] : I);

        // Precompute weights and indices
        offset_t ix[N];
        reduce_t wx[N];
        int8_t   fx[N];
        offset_t length = utils::index(loc[0], size[0], ix, wx, fx);
        for (offset_t i = 0, s = stride[0]; i < length; ++i)
            ix[i] *= s;

        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc) {
            reduce_t val = static_cast<reduce_t>(*inp);
            for (offset_t i = 0; i < length; ++i)
                bound::add(out, ix[i], val * wx[i], fx[i]);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count(
              scalar_t out      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t  b = (B == bound_t::Dynamic  ? bound[0]  : B);
        spline_t s = (I == spline_t::Dynamic ? spline[0] : I);

        // Precompute weights and indices
        offset_t ix[N];
        reduce_t wx[N];
        int8_t   fx[N];
        offset_t length = utils::index(loc[0], size[0], ix, wx, fx, b, s);
        for (offset_t i = 0, st = stride[0]; i < length; ++i)
            ix[i] *= st;

        for (offset_t i = 0; i < length; ++i)
            bound::add(out, ix[i], wx[i], fx[i]);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t  b = (B == bound_t::Dynamic  ? bound[0]  : B);
        spline_t s = (I == spline_t::Dynamic ? spline[0] : I);

        // Precompute weights and indices
        offset_t ix[N];
        reduce_t wx[N];
        int8_t   fx[N];
        reduce_t gx[N];
        offset_t length = utils::gindex(loc[0], size[0], ix, wx, gx, fx, b, s);
        for (offset_t i = 0, st = stride[0]; i < length; ++i)
            ix[i] *= st;

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t i = 0; i < length; ++i)
                acc += static_cast<reduce_t>(bound::get(inp, ix[i], fx[i])) * gx[i];
            *out = static_cast<scalar_t>(acc);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void hess(
              scalar_t out      [],
        const scalar_t inp      [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t osg2,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t  b = (B == bound_t::Dynamic  ? bound[0]  : B);
        spline_t s = (I == spline_t::Dynamic ? spline[0] : I);

        // Precompute weights and indices
        offset_t ix[N];
        reduce_t wx[N];
        int8_t   fx[N];
        reduce_t gx[N];
        reduce_t hx[N];
        offset_t length = utils::hindex(loc[0], size[0], ix, wx, gx, hx, fx, b, s);
        for (offset_t i = 0, st = stride[0]; i < length; ++i)
            ix[i] *= st;

        // Convolve coefficients with basis functions
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc)
        {
            reduce_t acc = static_cast<reduce_t>(0);
            for (offset_t i = 0; i < length; ++i)
                acc += static_cast<reduce_t>(bound::get(inp, ix[i], fx[i])) * hx[i];
            *out = static_cast<scalar_t>(acc);
        }
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [1],
        const offset_t size         [1],
        const offset_t stride_out   [1],
        const offset_t stride_inp   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t  b = (B == bound_t::Dynamic  ? bound[0]  : B);
        spline_t s = (I == spline_t::Dynamic ? spline[0] : I);

        // Precompute weights and indices
        offset_t ix[N];
        reduce_t wx[N];
        int8_t   fx[N];
        reduce_t gx[N];
        offset_t length = utils::gindex(loc[0], size[0], ix, wx, gx, fx, b, s);
        offset_t osx = stride_out[0], isx = stride_inp[0];

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            reduce_t gval = static_cast<reduce_t>(*ginp);
            reduce_t acc1 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i < length; ++i) {
                // push incoming gradient
                bound::add(out, ix[i] * osx, gval * wx[i], fx[i]);
                // compute input spatial gradient
                acc1 += bound::cget<reduce_t>(inp, ix[i] * isx, fx[i]) * gx[i];
            }
            acc += gval * acc1;
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push_backward(
              scalar_t out      [],
              scalar_t gout     [],
        const scalar_t inp      [],
        const scalar_t ginp     [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t  b = (B == bound_t::Dynamic  ? bound[0]  : B);
        spline_t s = (I == spline_t::Dynamic ? spline[0] : I);

        // Precompute weights and indices
        offset_t ix[N];
        reduce_t wx[N];
        int8_t   fx[N];
        reduce_t gx[N];
        offset_t length = utils::gindex(loc[0], size[0], ix, wx, gx, fx, b, s);
        for (offset_t i = 0, st = stride[0]; i < length; ++i)
            ix[i] *= st;

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += isg)
        {
            reduce_t val  = static_cast<reduce_t>(*inp);
            reduce_t acc1 = static_cast<reduce_t>(0);
            reduce_t acc2 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i < length; ++i) {
                reduce_t gval = bound::cget<reduce_t>(ginp, ix[i], fx[i]);
                // pull incoming gradient
                acc1 += gval * wx[i];
                // compute incoming gradient spatial gradient
                acc2 += gval * gx[i];
            }
            *out = static_cast<scalar_t>(acc1);
            acc += val * acc2;
        }
        *gout = static_cast<scalar_t>(acc);
    }

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count_backward(
              scalar_t gout     [],
        const scalar_t ginp     [],
        const reduce_t loc      [1],
        const offset_t size     [1],
        const offset_t stride   [1],
              offset_t osg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t  b = (B == bound_t::Dynamic  ? bound[0]  : B);
        spline_t s = (I == spline_t::Dynamic ? spline[0] : I);

        // Precompute weights and indices
        offset_t ix[N];
        reduce_t wx[N];
        int8_t   fx[N];
        reduce_t gx[N];
        offset_t length = utils::gindex(loc[0], size[0], ix, wx, gx, fx, b, s);
        for (offset_t i = 0, st = stride[0]; i < length; ++i)
            ix[i] *= st;

        // compute input spatial gradient
        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t i = 0; i < length; ++i)
            acc += static_cast<reduce_t>(bound::get(ginp, ix[i], fx[i])) * gx[i];
        *gout = static_cast<scalar_t>(acc);
    }


    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [1],
        const offset_t size         [1],
        const offset_t stride_out   [1],
        const offset_t stride_inp   [1],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t gsc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [1] = nullptr,
        const spline_t spline   [1] = nullptr
    )
    {
        bound_t  b = (B == bound_t::Dynamic  ? bound[0]  : B);
        spline_t s = (I == spline_t::Dynamic ? spline[0] : I);

        // Precompute weights and indices
        offset_t ix[N];
        reduce_t wx[N];
        int8_t   fx[N];
        reduce_t gx[N];
        reduce_t hx[N];
        offset_t length = utils::hindex(loc[0], size[0], ix, wx, gx, hx, fx, b, s);
        offset_t osx = stride_out[0], isx = stride_inp[0];

        reduce_t acc = static_cast<reduce_t>(0);
        for (offset_t c = 0; c < nc; ++c, out += osc, inp += isc, ginp += gsc)
        {
            reduce_t gval = static_cast<reduce_t>(*ginp);
            reduce_t acc1 = static_cast<reduce_t>(0);
            for (offset_t i = 0; i < length; ++i) {
                // push incoming gradient
                bound::add(out, ix[i] * osx, gval * gx[i], fx[i]);
                // compute input spatial hessian
                acc1 += bound::cget<reduce_t>(inp, ix[i] * isx, fx[i]) * hx[i];
            }
            acc += gval * acc1;
        }
        *gout = static_cast<scalar_t>(acc);
    }
};

FF_NAMESPACE_END(pushpull)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_PUSHPULL_1D
