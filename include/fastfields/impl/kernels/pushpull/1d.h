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
 *                                 ANY
 *
 **********************************************************************/
template <spline::type I, bound::type B, bool ABS>
struct Kernels<Config<one, Spline<I>, Bound<B>, ABS>> {
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
        bound_t  b = (B == bound_t::Dynamic  ? static_cast<bound_t>(bound[0])  : B);
        spline_t s = (I == spline_t::Dynamic ? static_cast<spline_t>(spline[0]) : I);

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
        bound_t  b = (B == bound_t::Dynamic  ? static_cast<bound_t>(bound[0])  : B);
        spline_t s = (I == spline_t::Dynamic ? static_cast<spline_t>(spline[0]) : I);

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
        bound_t  b = (B == bound_t::Dynamic  ? static_cast<bound_t>(bound[0])  : B);
        spline_t s = (I == spline_t::Dynamic ? static_cast<spline_t>(spline[0]) : I);

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
        bound_t  b = (B == bound_t::Dynamic  ? static_cast<bound_t>(bound[0])  : B);
        spline_t s = (I == spline_t::Dynamic ? static_cast<spline_t>(spline[0]) : I);

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
        bound_t  b = (B == bound_t::Dynamic  ? static_cast<bound_t>(bound[0])  : B);
        spline_t s = (I == spline_t::Dynamic ? static_cast<spline_t>(spline[0]) : I);

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
        bound_t  b = (B == bound_t::Dynamic  ? static_cast<bound_t>(bound[0])  : B);
        spline_t s = (I == spline_t::Dynamic ? static_cast<spline_t>(spline[0]) : I);

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
        bound_t  b = (B == bound_t::Dynamic  ? static_cast<bound_t>(bound[0])  : B);
        spline_t s = (I == spline_t::Dynamic ? static_cast<spline_t>(spline[0]) : I);

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
        bound_t  b = (B == bound_t::Dynamic  ? static_cast<bound_t>(bound[0])  : B);
        spline_t s = (I == spline_t::Dynamic ? static_cast<spline_t>(spline[0]) : I);

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
        bound_t  b = (B == bound_t::Dynamic  ? static_cast<bound_t>(bound[0])  : B);
        spline_t s = (I == spline_t::Dynamic ? static_cast<spline_t>(spline[0]) : I);

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
