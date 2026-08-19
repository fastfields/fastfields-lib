#ifndef FF_PUSHPULL_UTILS
#define FF_PUSHPULL_UTILS
#include "fastfields/core/cuda_switch.h"
#include "../spline.h"
#include "../bounds.h"
#include "../meta.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(pushpull)

const spline_t Z = spline_t::Nearest;
const spline_t L = spline_t::Linear;
const spline_t Q = spline_t::Quadratic;
const spline_t C = spline_t::Cubic;
const bound_t B0 = bound_t::NoCheck;
const int zero   = 0;
const int one    = 1;
const int two    = 2;
const int three  = 3;
const int mone   = -1;

template <int D, class I, class B, bool ABS=false> struct Config {
    static constexpr int  dim       = D;
    static constexpr bool abs       = ABS;
    using index_t                   = I;
    using bound_t                   = B;
};

template <class Config>
struct Kernels {

    static constexpr int D = Config::dim;

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull(
              scalar_t out      [],             // pointer to output voxel
        const scalar_t inp      [],             // pointer to input tensor
        const reduce_t loc      [D],            // input location to sample
        const offset_t size     [D],            // spatial size of input tensor
        const offset_t stride   [D],            // spatial strides of input tensor
              offset_t nc,                      // number of channels
              offset_t osc,                     // output channel stride
              offset_t isc,                     // input channel stride
        const bound_t  bound    [D] = nullptr,  // boundary condition (if dynamic)
        const spline_t spline   [D] = nullptr   // interpolation (if dynamic)
    );

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push(
              scalar_t out      [],             // pointer to output tensor
        const scalar_t inp      [],             // pointer to input voxel
        const reduce_t loc      [D],            // output location in which to push input value
        const offset_t size     [D],            // spatial size of output tensor
        const offset_t stride   [D],            // spatial strides of output tensor
              offset_t nc,                      // number of channels
              offset_t osc,                     // output channel stride
              offset_t isc,                     // input channel stride
        const bound_t  bound    [D] = nullptr,  // boundary condition (if dynamic)
        const spline_t spline   [D] = nullptr   // interpolation (if dynamic)
    );

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count(
              scalar_t out      [],             // pointer to output tensor
        const reduce_t loc      [D],            // output location in which to push ones
        const offset_t size     [D],            // spatial size of output tensor
        const offset_t stride   [D],            // spatial strides of output tensor
        const bound_t  bound    [D] = nullptr,  // boundary condition (if dynamic)
        const spline_t spline   [D] = nullptr   // interpolation (if dynamic)
    );

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad(
              scalar_t out      [],             // pointer to output voxel
        const scalar_t inp      [],             // pointer to input tensor
        const reduce_t loc      [D],            // input location to sample
        const offset_t size     [D],            // spatial size of input tensor
        const offset_t stride   [D],            // spatial strides of input tensor
              offset_t nc,                      // number of channels
              offset_t osc,                     // output channel stride
              offset_t isc,                     // input channel stride
              offset_t osg,                     // output direction axis stride
        const bound_t  bound    [D] = nullptr,  // boundary condition (if dynamic)
        const spline_t spline   [D] = nullptr   // interpolation (if dynamic)
    );

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void hess(
              scalar_t out      [],             // pointer to output voxel
        const scalar_t inp      [],             // pointer to input tensor
        const reduce_t loc      [D],            // input location to sample
        const offset_t size     [D],            // spatial size of input tensor
        const offset_t stride   [D],            // spatial strides of input tensor
              offset_t nc,                      // number of channels
              offset_t osc,                     // output channel stride
              offset_t isc,                     // input channel stride
              offset_t osg,                     // output direction axis stride
        const bound_t  bound    [D] = nullptr,  // boundary condition (if dynamic)
        const spline_t spline   [D] = nullptr   // interpolation (if dynamic)
    );

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void pull_backward(
              scalar_t out          [],         // pointer to output tensor
              scalar_t gout         [],         // pointer to output gradient
        const scalar_t inp          [],         // pointer to input voxel
        const scalar_t ginp         [],         // pointer to input gradient
        const reduce_t loc          [D],        // output location in which to push input value
        const offset_t size         [D],        // spatial size of output tensor
        const offset_t stride_out   [D],        // spatial strides of output tensor
        const offset_t stride_inp   [D],        // spatial strides of input tensor
              offset_t nc,                      // number of channels
              offset_t osc,                     // output channel stride
              offset_t isc,                     // input channel stride
              offset_t osg,                     // output gradient direction axis stride
              offset_t isg,                     // input gradient direction axis stride
        const bound_t  bound    [D] = nullptr,  // boundary condition (if dynamic)
        const spline_t spline   [D] = nullptr   // interpolation (if dynamic)
    );

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void push_backward(
              scalar_t out      [],
              scalar_t gout     [],
        const scalar_t inp      [],
        const scalar_t ginp     [],
        const reduce_t loc      [D],
        const offset_t size     [D],
        const offset_t stride   [D],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [D] = nullptr,  // boundary condition (if dynamic)
        const spline_t spline   [D] = nullptr   // interpolation (if dynamic)
    );

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void count_backward(
              scalar_t gout     [],
        const scalar_t ginp     [],
        const reduce_t loc      [D],
        const offset_t size     [D],
        const offset_t stride   [D],
              offset_t osg,
        const bound_t  bound    [D] = nullptr,  // boundary condition (if dynamic)
        const spline_t spline   [D] = nullptr   // interpolation (if dynamic)
    );

    template <typename reduce_t, typename scalar_t, typename offset_t>
    CUDEV static inline
    void grad_backward(
              scalar_t out          [],
              scalar_t gout         [],
        const scalar_t inp          [],
        const scalar_t ginp         [],
        const reduce_t loc          [D],
        const offset_t size         [D],
        const offset_t stride_out   [D],
        const offset_t stride_inp   [D],
              offset_t nc,
              offset_t osc,
              offset_t isc,
              offset_t gsc,
              offset_t osg,
              offset_t isg,
        const bound_t  bound    [D] = nullptr,  // boundary condition (if dynamic)
        const spline_t spline   [D] = nullptr   // interpolation (if dynamic)
    );
};

/***********************************************************************
 *
 *                                UTILS
 *
 **********************************************************************/


template <bool ABS = false>
struct PushPullMaybe {
    template <typename T>
    static inline CUDEV
    const T& fabs(const T& val) { return val; }
};


template <>
struct PushPullMaybe<true> {
    template <typename T>
    static inline CUDEV
    T fabs(const T& val) { return ::fabs(val); }
};


/*** Bufsize ***********************************************************/

template <spline_t S=spline_t::Dynamic>
struct SplineBufSize
{
    static constexpr int bufsize = static_cast<int>(
        static_cast<signed char>(S) + 1
    );
};


template <>
struct SplineBufSize<spline_t::Dynamic>
{
    static constexpr int bufsize = 8;   // upper bound (max order = 7)
};

/*** Any **************************************************************/
template <spline_t S=Z, bound_t B=B0, bool ABS=false>
struct PushPullUtils {

    using bound_utils  = bound::utils<B>;
    using spline_utils = spline::utils<S>;
    using maybe        = PushPullMaybe<ABS>;
    static constexpr int bufsize = SplineBufSize<S>::bufsize;

    template <typename reduce_t, typename offset_t>
    static inline CUDEV offset_t
    index(
        reduce_t x,
        offset_t size,
        offset_t i  [],
        reduce_t w  [],
        int8_t   f  [],
        bound_t  b = B,
        spline_t s = S
    )
    {
        const auto bounds_fn = (
            S == spline_t::Dynamic
            ? spline::bounds_fn<reduce_t, offset_t>(s)
            : spline_utils::template bounds<reduce_t, offset_t>
        );
        const auto fastweight_fn = (
            S == spline_t::Dynamic
            ? spline::fastweight_fn<reduce_t>(s)
            : spline_utils::template fastweight<reduce_t>
        );
        const auto sign_fn = (
            B == bound_t::Dynamic
            ? bound::sign_fn<offset_t>(b)
            : bound_utils::template sign<offset_t>
        );
        const auto index_fn = (
            B == bound_t::Dynamic
            ? bound::index_fn<offset_t>(b)
            : bound_utils::template index<offset_t>
        );

        offset_t b0, b1;
        bounds_fn(x, b0, b1);
        reduce_t *ow = w;
        offset_t *oi = i;
        int8_t   *of = f;
        for (offset_t bi = b0; bi <= b1; ++bi) {
            reduce_t d = fabs(x - bi);
            *(ow++) = fastweight_fn(d);
            *(of++) = sign_fn(bi, size);
            *(oi++) = index_fn(bi, size);
        }
        return b1-b0+1;
    }

    template <typename reduce_t, typename offset_t>
    static inline CUDEV offset_t
    gindex(
        reduce_t x,
        offset_t size,
        offset_t i  [],
        reduce_t w  [],
        reduce_t g  [],
        int8_t   f  [],
        bound_t  b = B,
        spline_t s = S
    )
    {
        const auto bounds_fn = (
            S == spline_t::Dynamic
            ? spline::bounds_fn<reduce_t, offset_t>(s)
            : spline_utils::template bounds<reduce_t, offset_t>
        );
        const auto fastweight_fn = (
            S == spline_t::Dynamic
            ? spline::fastweight_fn<reduce_t>(s)
            : spline_utils::template fastweight<reduce_t>
        );
        const auto fastgrad_fn = (
            S == spline_t::Dynamic
            ? spline::fastgrad_fn<reduce_t>(s)
            : spline_utils::template fastgrad<reduce_t>
        );
        const auto sign_fn = (
            B == bound_t::Dynamic
            ? bound::sign_fn<offset_t>(b)
            : bound_utils::template sign<offset_t>
        );
        const auto index_fn = (
            B == bound_t::Dynamic
            ? bound::index_fn<offset_t>(b)
            : bound_utils::template index<offset_t>
        );

        offset_t b0, b1;
        bounds_fn(x, b0, b1);
        reduce_t *ow = w;
        reduce_t *og = g;
        offset_t *oi = i;
        int8_t   *of = f;
        for (offset_t bi = b0; bi <= b1; ++bi) {
            reduce_t d = x - bi;
            bool neg = d < 0;
            if (neg) d = -d;
            *(ow++)  = fastweight_fn(d);
            *(og++)  = maybe::fabs(fastgrad_fn(d) * (neg ? -1 : 1));
            *(of++)  = sign_fn(bi, size);
            *(oi++)  = index_fn(bi, size);
        }
        return b1-b0+1;
    }

    template <typename reduce_t, typename offset_t>
    static inline CUDEV offset_t
    hindex(
        reduce_t x,
        offset_t size,
        offset_t i  [],
        reduce_t w  [],
        reduce_t g  [],
        reduce_t h  [],
        int8_t   f  [],
        bound_t  b = B,
        spline_t s = S
    )
    {
        const auto bounds_fn = (
            S == spline_t::Dynamic
            ? spline::bounds_fn<reduce_t, offset_t>(s)
            : spline_utils::template bounds<reduce_t, offset_t>
        );
        const auto fastweight_fn = (
            S == spline_t::Dynamic
            ? spline::fastweight_fn<reduce_t>(s)
            : spline_utils::template fastweight<reduce_t>
        );
        const auto fastgrad_fn = (
            S == spline_t::Dynamic
            ? spline::fastgrad_fn<reduce_t>(s)
            : spline_utils::template fastgrad<reduce_t>
        );
        const auto fasthess_fn = (
            S == spline_t::Dynamic
            ? spline::fasthess_fn<reduce_t>(s)
            : spline_utils::template fasthess<reduce_t>
        );
        const auto sign_fn = (
            B == bound_t::Dynamic
            ? bound::sign_fn<offset_t>(b)
            : bound_utils::template sign<offset_t>
        );
        const auto index_fn = (
            B == bound_t::Dynamic
            ? bound::index_fn<offset_t>(b)
            : bound_utils::template index<offset_t>
        );

        offset_t b0, b1;
        bounds_fn(x, b0, b1);
        reduce_t *ow = w;
        reduce_t *og = g;
        reduce_t *oh = h;
        offset_t *oi = i;
        int8_t   *of = f;
        for (offset_t bi = b0; bi <= b1; ++bi) {
            reduce_t d = x - bi;
            bool neg = d < 0;
            if (neg) d = -d;
            *(ow++)  = fastweight_fn(d);
            *(og++)  = maybe::fabs(fastgrad_fn(d) * (neg ? -1 : 1));
            *(oh++)  = maybe::fabs(fasthess_fn(d));
            *(of++)  = sign_fn(bi, size);
            *(oi++)  = index_fn(bi, size);
        }
        return b1-b0+1;
    }

};

FF_NAMESPACE_END(pushpull)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_PUSHPULL_UTILS
