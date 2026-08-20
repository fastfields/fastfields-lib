#ifndef FF_PUSHPULL_UTILS
#define FF_PUSHPULL_UTILS
#include "fastfields/core/cuda_switch.h"
#include "fastfields/core/spline.h"
#include "fastfields/core/bounds.h"
#include "fastfields/core/meta.h"

FF_NAMESPACE_BEGIN(FF_NS)
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
    FF_CUDEV static inline
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
    FF_CUDEV static inline
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
    FF_CUDEV static inline
    void count(
              scalar_t out      [],             // pointer to output tensor
        const reduce_t loc      [D],            // output location in which to push ones
        const offset_t size     [D],            // spatial size of output tensor
        const offset_t stride   [D],            // spatial strides of output tensor
        const bound_t  bound    [D] = nullptr,  // boundary condition (if dynamic)
        const spline_t spline   [D] = nullptr   // interpolation (if dynamic)
    );

    template <typename reduce_t, typename scalar_t, typename offset_t>
    FF_CUDEV static inline
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
    FF_CUDEV static inline
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
    FF_CUDEV static inline
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
    FF_CUDEV static inline
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
    FF_CUDEV static inline
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
    FF_CUDEV static inline
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


template <
    spline_t IX=Z,  bound_t BX=B0,
    bool ABS=false
>
using PushPull1D = Kernels<Config<one, Spline<IX>, Bound<BX>, ABS>>;

template <
    spline_t IX=Z,  bound_t BX=B0,
    spline_t IY=IX, bound_t BY=BX,
    bool ABS=false
>
using PushPull2D = Kernels<Config<two, Spline<IX, IY>, Bound<BX, BY>, ABS>>;

template <
    spline_t IX=Z,  bound_t BX=B0,
    spline_t IY=IX, bound_t BY=BX,
    spline_t IZ=IY, bound_t BZ=BY,
    bool ABS=false
>
using PushPull3D = Kernels<Config<three, Spline<IX, IY, IZ>, Bound<BX, BY, BZ>, ABS>>;
// template <int D, bool ABS=false>
// using PushPullND = PushPull<D, Z, B0, Z, B0, Z, B0, ABS>;

// Dimension-parametrised dispatcher: maps a runtime-selected `ndim`
// (1/2/3) plus per-dim spline/bound to the corresponding `Kernels<...>`
// specialisation. The impl layer instantiates `PushPull<ndim, ...>`.
template <
    int D,
    spline_t IX=Z,  bound_t BX=B0,
    spline_t IY=IX, bound_t BY=BX,
    spline_t IZ=IY, bound_t BZ=BY,
    bool ABS=false
>
struct PushPullSelect;

template <spline_t IX, bound_t BX, spline_t IY, bound_t BY, spline_t IZ, bound_t BZ, bool ABS>
struct PushPullSelect<one, IX, BX, IY, BY, IZ, BZ, ABS> {
    using type = Kernels<Config<one, Spline<IX>, Bound<BX>, ABS>>;
};

template <spline_t IX, bound_t BX, spline_t IY, bound_t BY, spline_t IZ, bound_t BZ, bool ABS>
struct PushPullSelect<two, IX, BX, IY, BY, IZ, BZ, ABS> {
    using type = Kernels<Config<two, Spline<IX, IY>, Bound<BX, BY>, ABS>>;
};

template <spline_t IX, bound_t BX, spline_t IY, bound_t BY, spline_t IZ, bound_t BZ, bool ABS>
struct PushPullSelect<three, IX, BX, IY, BY, IZ, BZ, ABS> {
    using type = Kernels<Config<three, Spline<IX, IY, IZ>, Bound<BX, BY, BZ>, ABS>>;
};

template <
    int D,
    spline_t IX=Z,  bound_t BX=B0,
    spline_t IY=IX, bound_t BY=BX,
    spline_t IZ=IY, bound_t BZ=BY,
    bool ABS=false
>
using PushPull = typename PushPullSelect<D, IX, BX, IY, BY, IZ, BZ, ABS>::type;


/***********************************************************************
 *
 *                                UTILS
 *
 **********************************************************************/


/*** Check In/Out of Bounds *******************************************/
#define FF_EXTRAPOLATE_TINY 5E-2

template <int extrapolate, int D>
struct InFOV {};

template <int D>
struct InFOV<one, D> {
    template <typename scalar_t, typename offset_t>
    static inline FF_CUDEV bool
    infov(const scalar_t * loc, const offset_t * size, offset_t stride=1) {
        return true;
    }
};

template <int D>
struct InFOV<zero, D> { // Limits at voxel centers
    template <typename scalar_t, typename offset_t>
    static inline FF_CUDEV bool
    infov(const scalar_t * loc, const offset_t * size, offset_t stride=1) {
#       pragma unroll
        for (int d=0; d < D; ++d, loc += stride) {
            scalar_t loc1 = *loc;
            if (loc1 < -FF_EXTRAPOLATE_TINY)
                return false;
            if (loc1 > size[d] - 1 + FF_EXTRAPOLATE_TINY)
                return false;
        }
        return true;
    }
};

template <int D>
struct InFOV<mone, D> { // Limits at voxel edges
    template <typename scalar_t, typename offset_t>
    static inline FF_CUDEV bool
    infov(const scalar_t * loc, const offset_t * size, offset_t stride=1) {
#       pragma unroll
        for (int d=0; d < D; ++d, loc += stride) {
            scalar_t loc1 = *loc;
            if (loc1 < - 0.5 - FF_EXTRAPOLATE_TINY)
                return false;
            if (loc1 > size[d] - 0.5 + FF_EXTRAPOLATE_TINY)
                return false;
        }
        return true;
    }
};

/*
template <>
struct InFOV<one, one> {
    template <typename scalar_t, typename offset_t>
    static FF_CUDEV bool
    infov(scalar_t x, offset_t nx) {
        return true;
    }
};

template <>
struct InFOV<zero, one> { // Limits at voxel centers
    template <typename scalar_t, typename offset_t>
    static FF_CUDEV bool
    infov(scalar_t x, offset_t nx) {
        if (x < -FF_EXTRAPOLATE_TINY)
            return false;
        if (x > nx - 1 + FF_EXTRAPOLATE_TINY)
            return false;
        return true;
    }
};

template <>
struct InFOV<mone, one> { // Limits at voxel edges
    template <typename scalar_t, typename offset_t>
    static FF_CUDEV bool
    infov(scalar_t x, offset_t nx) {
        if (x < -0.5 - FF_EXTRAPOLATE_TINY)
            return false;
        if (x > nx - 0.5 + FF_EXTRAPOLATE_TINY)
            return false;
        return true;
    }
};

template <>
struct InFOV<one, two> {
    template <typename scalar_t, typename offset_t>
    static FF_CUDEV bool
    infov(scalar_t x, scalar_t y, offset_t nx, offset_t ny) {
        return true;
    }
};

template <>
struct InFOV<zero, two> {
    template <typename scalar_t, typename offset_t>
    static FF_CUDEV bool
    infov(scalar_t x, scalar_t y, offset_t nx, offset_t ny) {
        return InFOV<0, 1>::infov(x, nx) &&
               InFOV<0, 1>::infov(y, ny);
    }
};

template <>
struct InFOV<mone, two> {
    template <typename scalar_t, typename offset_t>
    static FF_CUDEV bool
    infov(scalar_t x, scalar_t y, offset_t nx, offset_t ny) {
        return InFOV<-1, 1>::infov(x, nx) &&
               InFOV<-1, 1>::infov(y, ny);
    }
};

template <>
struct InFOV<one, three> {
    template <typename scalar_t, typename offset_t>
    static FF_CUDEV bool
    infov(scalar_t x, scalar_t y, scalar_t z,
          offset_t nx, offset_t ny, offset_t nz) {
        return true;
    }
};

template <>
struct InFOV<zero, three> {
    template <typename scalar_t, typename offset_t>
    static FF_CUDEV bool
    infov(scalar_t x, scalar_t y, scalar_t z,
          offset_t nx, offset_t ny, offset_t nz) {
        return InFOV<0, 1>::infov(x, nx) &&
               InFOV<0, 1>::infov(y, ny) &&
               InFOV<0, 1>::infov(z, nz);
    }
};

template <>
struct InFOV<mone, three> {
    template <typename scalar_t, typename offset_t>
    static FF_CUDEV bool
    infov(scalar_t x, scalar_t y, scalar_t z,
          offset_t nx, offset_t ny, offset_t nz) {
        return InFOV<-1, 1>::infov(x, nx) &&
               InFOV<-1, 1>::infov(y, ny) &&
               InFOV<-1, 1>::infov(z, nz);
    }
};
*/


template <bool ABS = false>
struct PushPullMaybe {
    template <typename T>
    static inline FF_CUDEV
    const T& fabs(const T& val) { return val; }
};


template <>
struct PushPullMaybe<true> {
    template <typename T>
    static inline FF_CUDEV
    T fabs(const T& val) { return ::fabs(val); }
};


/**********************************************************************\
|*** Wrap out-of-bounds indices ***************************************|
\**********************************************************************/

template <spline_t I=Z,  bound_t B=B0, bool ABS=false>
struct PushPullAnyUtils {
    using spline_utils = spline::utils<I>;
    using bound_utils = bound::utils<B>;
    using maybe = PushPullMaybe<ABS>;


    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    index(reduce_t x, offset_t size, offset_t i[], reduce_t w[], int8_t s[])
    {
        offset_t b0, b1;
        spline_utils::bounds(x, b0, b1);
        offset_t db = b1-b0;
        reduce_t    *ow = w;
        offset_t    *oi = i;
        int8_t *os = s;
        for (offset_t b = b0; b <= b1; ++b) {
            reduce_t d = fabs(x - b);
            *(ow++)  = spline_utils::fastweight(d);
            *(os++)  = bound_utils::sign(b, size);
            *(oi++)  = bound_utils::index(b, size);
        }
        return db;
    }

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    gindex(reduce_t x, offset_t size, offset_t i[], reduce_t w[], reduce_t g[], int8_t s[])
    {
        offset_t b0, b1;
        spline_utils::bounds(x, b0, b1);
        offset_t db = b1-b0;
        reduce_t    *ow = w;
        reduce_t    *og = g;
        offset_t    *oi = i;
        int8_t *os = s;
        for (offset_t b = b0; b <= b1; ++b) {
            reduce_t d = x - b;
            bool neg = d < 0;
            if (neg) d = -d;
            *(ow++)  = spline_utils::fastweight(d);
            *(og++)  = maybe::fabs(spline_utils::fastgrad(d) * (neg ? -1 : 1));
            *(os++)  = bound_utils::sign(b, size);
            *(oi++)  = bound_utils::index(b, size);
        }
        return db;
    }

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    hindex(reduce_t x, offset_t size, offset_t i[],
           reduce_t w[], reduce_t g[], reduce_t h[], int8_t s[])
    {
        offset_t b0, b1;
        spline_utils::bounds(x, b0, b1);
        offset_t db = b1-b0;
        reduce_t    *ow = w;
        reduce_t    *og = g;
        reduce_t    *oh = h;
        offset_t    *oi = i;
        int8_t *os = s;
        for (offset_t b = b0; b <= b1; ++b) {
            reduce_t d = x - b;
            bool neg = d < 0;
            if (neg) d = -d;
            *(ow++)  = spline_utils::fastweight(d);
            *(og++)  = maybe::fabs(spline_utils::fastgrad(d) * (neg ? -1 : 1));
            *(oh++)  = maybe::fabs(spline_utils::fasthess(d));
            *(os++)  = bound_utils::sign(b, size);
            *(oi++)  = bound_utils::index(b, size);
        }
        return db;
    }
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
    static inline FF_CUDEV offset_t
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
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);

        offset_t b0, b1;
        su.bounds(x, b0, b1);
        reduce_t *ow = w;
        offset_t *oi = i;
        int8_t   *of = f;
        for (offset_t bi = b0; bi <= b1; ++bi) {
            reduce_t d = fabs(x - bi);
            *(ow++) = su.fastweight(d);
            *(of++) = bu.sign(bi, size);
            *(oi++) = bu.index(bi, size);
        }
        return b1-b0+1;
    }

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
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
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);

        offset_t b0, b1;
        su.bounds(x, b0, b1);
        reduce_t *ow = w;
        reduce_t *og = g;
        offset_t *oi = i;
        int8_t   *of = f;
        for (offset_t bi = b0; bi <= b1; ++bi) {
            reduce_t d = x - bi;
            bool neg = d < 0;
            if (neg) d = -d;
            *(ow++)  = su.fastweight(d);
            *(og++)  = maybe::fabs(su.fastgrad(d) * (neg ? -1 : 1));
            *(of++)  = bu.sign(bi, size);
            *(oi++)  = bu.index(bi, size);
        }
        return b1-b0+1;
    }

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
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
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);

        offset_t b0, b1;
        su.bounds(x, b0, b1);
        reduce_t *ow = w;
        reduce_t *og = g;
        reduce_t *oh = h;
        offset_t *oi = i;
        int8_t   *of = f;
        for (offset_t bi = b0; bi <= b1; ++bi) {
            reduce_t d = x - bi;
            bool neg = d < 0;
            if (neg) d = -d;
            *(ow++)  = su.fastweight(d);
            *(og++)  = maybe::fabs(su.fastgrad(d) * (neg ? -1 : 1));
            *(oh++)  = maybe::fabs(su.fasthess(d));
            *(of++)  = bu.sign(bi, size);
            *(oi++)  = bu.index(bi, size);
        }
        return b1-b0+1;
    }

};

/*** Nearest **********************************************************/
template <bound_t B, bool ABS>
struct PushPullUtils<Z,B,ABS> {
    using bound_utils = bound::utils<B>;
    using spline_utils = spline::utils<Z>;
    using maybe = PushPullMaybe<ABS>;
    static constexpr int bufsize = 1;
    static constexpr spline_t S = Z;

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    index(
        reduce_t    x,
        offset_t    size,
        offset_t    i [1],
        reduce_t    w [1],
        int8_t      f [1],
        bound_t     b = B,
        spline_t    s = Z
    )
    {
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);
        *i = static_cast<offset_t>(round(x));
        *f = bu.sign(*i, size);
        *i = bu.index(*i, size);
        if (w) *w = static_cast<reduce_t>(1);
        return static_cast<offset_t>(1);
    }

    // Weight-less overload: nearest interpolation has an implicit weight
    // of 1, so callers that do not need the weight buffer can omit it.
    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    index(
        reduce_t    x,
        offset_t    size,
        offset_t    i [1],
        int8_t      f [1],
        bound_t     b = B,
        spline_t    s = Z
    )
    {
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);
        return index(x, size, i, static_cast<reduce_t *>(nullptr), f, b, s);
    }

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    gindex(
        reduce_t x,
        offset_t size,
        offset_t i  [1],
        reduce_t w  [1],
        reduce_t g  [1],
        int8_t   f  [1],
        bound_t  b = B,
        spline_t s = S
    )
    {
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);
        index(x, size, i, w, f, b, s);
        if (g) *g = static_cast<reduce_t>(0);
        return static_cast<offset_t>(1);
    }

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    hindex(
        reduce_t x,
        offset_t size,
        offset_t i  [1],
        reduce_t w  [1],
        reduce_t g  [1],
        reduce_t h  [1],
        int8_t   f  [1],
        bound_t  b = B,
        spline_t s = S
    )
    {
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);
        gindex(x, size, i, w, f, b, s);
        if (h) *h = static_cast<reduce_t>(0);
        return static_cast<offset_t>(1);
    }
};

/*** Linear ***********************************************************/
template <bound_t B, bool ABS>
struct PushPullUtils<L,B,ABS> {
    using bound_utils = bound::utils<B>;
    using spline_utils = spline::utils<L>;
    using maybe = PushPullMaybe<ABS>;
    static constexpr int bufsize = 2;
    static constexpr spline_t S = L;

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    index(
        reduce_t    x,
        offset_t    size,
        offset_t    i [2],
        reduce_t    w [2],
        int8_t      f [2],
        bound_t     b = B,
        spline_t    s = L
    )
    {
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);
        i[0] = static_cast<offset_t>(floor(x));
        w[1] = x - i[0];
        w[0] = 1. - w[1];
        f[1] = bu.sign(i[0]+1, size);
        f[0] = bu.sign(i[0],   size);
        i[1] = bu.index(i[0]+1, size);
        i[0] = bu.index(i[0],   size);
        return static_cast<offset_t>(2);
    }

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    gindex(
        reduce_t x,
        offset_t size,
        offset_t i  [1],
        reduce_t w  [1],
        reduce_t g  [1],
        int8_t   f  [1],
        bound_t  b = B,
        spline_t s = S
    )
    {
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);
        index(x, size, i, w, f, b, s);
        if (g) {
            g[0] = static_cast<reduce_t>(-1);
            g[1] = static_cast<reduce_t>(1);
        }
        return static_cast<offset_t>(2);
    }

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    hindex(
        reduce_t x,
        offset_t size,
        offset_t i  [1],
        reduce_t w  [1],
        reduce_t g  [1],
        reduce_t h  [1],
        int8_t   f  [1],
        bound_t  b = B,
        spline_t s = S
    )
    {
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);
        gindex(x, size, i, w, f, b, s);
        if (h)
            h[0] = h[1] = static_cast<reduce_t>(0);
        return static_cast<offset_t>(2);
    }
};

/*** Quadratic ********************************************************/
template <bound_t B, bool ABS>
struct PushPullUtils<Q,B,ABS> {
    using bound_utils = bound::utils<B>;
    using spline_utils = spline::utils<Q>;
    using maybe = PushPullMaybe<ABS>;
    static constexpr int bufsize = 3;
    static constexpr spline_t S = Q;

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    index(
        reduce_t    x,
        offset_t    size,
        offset_t    i [3],
        reduce_t    w [3],
        int8_t      f [3],
        bound_t     b = B,
        spline_t    s = Q
    )
    {
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);
        i[1] = static_cast<offset_t>(round(x));
        i[0] = i[1] - 1;
        i[2] = i[1] + 1;
        w[0] = su.fastweight(x - i[0]);
        w[1] = su.weight(x - i[1]); // cannot use fast (sign unknown)
        w[2] = su.fastweight(i[2] - x);
        f[0] = bu.sign(i[0], size);
        f[1] = bu.sign(i[1], size);
        f[2] = bu.sign(i[2], size);
        i[0] = bu.index(i[0], size);
        i[1] = bu.index(i[1], size);
        i[2] = bu.index(i[2], size);
        return static_cast<offset_t>(3);
    }

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    gindex(
        reduce_t    x,
        offset_t    size,
        offset_t    i [3],
        reduce_t    w [3],
        reduce_t    g [3],
        int8_t      f [3],
        bound_t     b = B,
        spline_t    s = Q
    )
    {
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);
        i[1] = static_cast<offset_t>(round(x));
        i[0] = i[1] - 1;
        i[2] = i[1] + 1;
        w[0] = su.fastweight(x - i[0]);
        w[1] = su.weight(x - i[1]); // cannot use fast (sign unknown)
        w[2] = su.fastweight(i[2] - x);
        g[0] = maybe::fabs(su.fastgrad(x - i[0]));
        g[1] = maybe::fabs(su.grad(x - i[1])); // cannot use fast (sign unknown)
        g[2] = maybe::fabs(-su.fastgrad(i[2] - x));
        f[0] = bu.sign(i[0], size);
        f[1] = bu.sign(i[1], size);
        f[2] = bu.sign(i[2], size);
        i[0] = bu.index(i[0], size);
        i[1] = bu.index(i[1], size);
        i[2] = bu.index(i[2], size);
        return static_cast<offset_t>(3);
    }

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    hindex(
        reduce_t    x,
        offset_t    size,
        offset_t    i [3],
        reduce_t    w [3],
        reduce_t    g [3],
        reduce_t    h [3],
        int8_t      f [3],
        bound_t     b = B,
        spline_t    s = Q
    )
    {
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);
        i[1] = static_cast<offset_t>(round(x));
        i[0] = i[1] - 1;
        i[2] = i[1] + 1;
        w[0] = su.fastweight(x - i[0]);
        w[1] = su.weight(x - i[1]); // cannot use fast (sign unknown)
        w[2] = su.fastweight(i[2] - x);
        g[0] = maybe::fabs(su.fastgrad(x - i[0]));
        g[1] = maybe::fabs(su.grad(x - i[1])); // cannot use fast (sign unknown)
        g[2] = maybe::fabs(-su.fastgrad(i[2] - x));
        h[0] = maybe::fabs(su.fasthess(x - i[0]));
        h[1] = maybe::fabs(su.hess(x - i[1])); // cannot use fast (sign unknown)
        h[2] = maybe::fabs(su.fasthess(i[2] - x));
        f[0] = bu.sign(i[0], size);
        f[1] = bu.sign(i[1], size);
        f[2] = bu.sign(i[2], size);
        i[0] = bu.index(i[0], size);
        i[1] = bu.index(i[1], size);
        i[2] = bu.index(i[2], size);
        return static_cast<offset_t>(3);
    }
};

/*** Cubic ************************************************************/
template <bound_t B, bool ABS>
struct PushPullUtils<C,B,ABS> {
    using bound_utils = bound::utils<B>;
    using spline_utils = spline::utils<C>;
    using maybe = PushPullMaybe<ABS>;
    static constexpr int bufsize = 4;
    static constexpr spline_t S = C;

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    index(
        reduce_t    x,
        offset_t    size,
        offset_t    i [4],
        reduce_t    w [4],
        int8_t      f [4],
        bound_t     b = B,
        spline_t    s = C
    )
    {
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);
        i[1] = static_cast<offset_t>(floor(x));
        i[0] = i[1] - 1;
        i[2] = i[1] + 1;
        i[3] = i[1] + 2;
        w[0] = su.fastweight(x - i[0]);
        w[1] = su.fastweight(x - i[1]);
        w[2] = su.fastweight(i[2] - x);
        w[3] = su.fastweight(i[3] - x);
        f[0] = bu.sign(i[0], size);
        f[1] = bu.sign(i[1], size);
        f[2] = bu.sign(i[2], size);
        f[3] = bu.sign(i[3], size);
        i[0] = bu.index(i[0], size);
        i[1] = bu.index(i[1], size);
        i[2] = bu.index(i[2], size);
        i[3] = bu.index(i[3], size);
        return static_cast<offset_t>(4);
    }

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    gindex(
        reduce_t    x,
        offset_t    size,
        offset_t    i [4],
        reduce_t    w [4],
        reduce_t    g [4],
        int8_t      f [4],
        bound_t     b = B,
        spline_t    s = C
    )
    {
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);
        i[1] = static_cast<offset_t>(floor(x));
        i[0] = i[1] - 1;
        i[2] = i[1] + 1;
        i[3] = i[1] + 2;
        w[0] = su.fastweight(x - i[0]);
        w[1] = su.fastweight(x - i[1]);
        w[2] = su.fastweight(i[2] - x);
        w[3] = su.fastweight(i[3] - x);
        g[0] = maybe::fabs(su.fastgrad(x - i[0]));
        g[1] = maybe::fabs(su.fastgrad(x - i[1]));
        g[2] = maybe::fabs(-su.fastgrad(i[2] - x));
        g[3] = maybe::fabs(-su.fastgrad(i[3] - x));
        f[0] = bu.sign(i[0], size);
        f[1] = bu.sign(i[1], size);
        f[2] = bu.sign(i[2], size);
        f[3] = bu.sign(i[3], size);
        i[0] = bu.index(i[0], size);
        i[1] = bu.index(i[1], size);
        i[2] = bu.index(i[2], size);
        i[3] = bu.index(i[3], size);
        return static_cast<offset_t>(4);
    }

    template <typename reduce_t, typename offset_t>
    static inline FF_CUDEV offset_t
    hindex(
        reduce_t    x,
        offset_t    size,
        offset_t    i [4],
        reduce_t    w [4],
        reduce_t    g [4],
        reduce_t    h [4],
        int8_t      f [4],
        bound_t     b = B,
        spline_t    s = C
    )
    {
        const bound::dyn<B>  bu(b);
        const spline::dyn<S> su(s);
        i[1] = static_cast<offset_t>(floor(x));
        i[0] = i[1] - 1;
        i[2] = i[1] + 1;
        i[3] = i[1] + 2;
        w[0] = spline_utils::fastweight(x - i[0]);
        w[1] = spline_utils::fastweight(x - i[1]);
        w[2] = spline_utils::fastweight(i[2] - x);
        w[3] = spline_utils::fastweight(i[3] - x);
        g[0] = maybe::fabs(spline_utils::fastgrad(x - i[0]));
        g[1] = maybe::fabs(spline_utils::fastgrad(x - i[1]));
        g[2] = maybe::fabs(-spline_utils::fastgrad(i[2] - x));
        g[3] = maybe::fabs(-spline_utils::fastgrad(i[3] - x));
        h[0] = maybe::fabs(spline_utils::fasthess(x - i[0]));
        h[1] = maybe::fabs(spline_utils::fasthess(x - i[1]));
        h[2] = maybe::fabs(spline_utils::fasthess(i[2] - x));
        h[3] = maybe::fabs(spline_utils::fasthess(i[3] - x));
        f[0] = bound_utils::sign(i[0], size);
        f[1] = bound_utils::sign(i[1], size);
        f[2] = bound_utils::sign(i[2], size);
        f[3] = bound_utils::sign(i[3], size);
        i[0] = bound_utils::index(i[0], size);
        i[1] = bound_utils::index(i[1], size);
        i[2] = bound_utils::index(i[2], size);
        i[3] = bound_utils::index(i[3], size);
        return static_cast<offset_t>(4);
    }
};

FF_NAMESPACE_END(pushpull)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)

#endif // FF_PUSHPULL_UTILS
