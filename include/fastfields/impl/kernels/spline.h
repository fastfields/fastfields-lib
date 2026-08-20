// This file contains static functions for handling (0-7 order)
// spline weights.
// It also defines an enumerated types that encodes each boundary type.
// The entry points are:
// . spline::weight     -> node weight based on distance
// . spline::fastweight -> same, assuming x lies in support
// . spline::grad       -> weight derivative // oriented distance
// . spline::fastgrad   -> same, assuming x lies in support
// . spline::hess       -> weight 2nd derivative // oriented distance
// . spline::fasthess   -> same, assuming x lies in support
// . spline::bounds     -> min/max nodes

// NOTE:
// 1st derivatives used to be implemented with a recursive call, e.g.:
// scalar_t grad2(scalar_t x) {
//   if (x < 0) return -grad2(-x);
//   ...
// }
// However, this prevents nvcc to staticallly determine the stack size
// and leads to memory errors (because the allocated stack is too small).
// I now use a slightly less compact implementation that gets rid of
// recursive calls.

// TODO? other types of basis functions (gauss, sinc)

#ifndef FF_SPLINE
#define FF_SPLINE
#include <fastfields/core/cuda_switch.h>
#include "meta.h"

FF_NAMESPACE_BEGIN(FF_NS)

FF_NAMESPACE_BEGIN(spline)
enum class type : int8_t {
    Dynamic       = -1,  // Used to turn-off static implementations in templated classes
    Nearest       = 0,
    Linear        = 1,
    Quadratic     = 2,
    Cubic         = 3,
    FourthOrder   = 4,
    FifthOrder    = 5,
    SixthOrder    = 6,
    SeventhOrder  = 7
};

// Runtime companion of the compile-time `Spline<S...>` pack -- the exact
// analogue of `bound::BoundVec` (see bounds.h).
//
// Every axis whose compile-time interpolation order is `type::Dynamic` reads
// its actual order from this vector at run time; axes instantiated with a real
// `type` ignore it entirely (the corresponding `spline::dyn<S>` specialisation
// is stateless and its constructor discards the argument). Trivially copyable,
// so it can be passed by value all the way into a `__global__` kernel.
//
// Templated on `MaxNDim` (default 3) rather than hard-coding the array size,
// mirroring `bound::BoundVecN` -- see its comment in bounds.h for why 3 is
// today's *ceiling* (set by the 1D/2D/3D-only pushpull/regulariser kernels
// that consume this vector), not an architectural limit of `SplineVec`
// itself. A future n>3 kernel can instantiate `SplineVecN<N>` directly;
// `SplineVec` (used everywhere today) is just the `N=3` alias below.
template <int MaxNDim = 3>
struct SplineVecN {
  static const int max_ndim = MaxNDim;
  int8_t s[max_ndim];

  inline FF_CUHOSTDEV SplineVecN()
  { for (int d = 0; d < max_ndim; ++d) s[d] = static_cast<int8_t>(type::Linear); }

  // Isotropic: the same order on every axis (what the public ABI exposes).
  explicit inline FF_CUHOSTDEV SplineVecN(type v)
  { for (int d = 0; d < max_ndim; ++d) s[d] = static_cast<int8_t>(v); }

  // Anisotropic: one order per axis, `ndim <= max_ndim` of them meaningful.
  // Axes `d >= ndim` are padded with `type::Linear` purely so every element
  // of this trivially-copyable struct holds a deterministic, valid
  // `spline::type` -- never `Dynamic` (which would force every `dyn<S>`
  // consumer down its runtime-switch path for a value nothing ever reads)
  // and never an uninitialised byte. No kernel actually reads a padding
  // axis: every dispatch layer (kernels/pushpull/{1d,2d,3d}.h) loops
  // exactly `ndim` times, never `max_ndim`, so the specific pad value is
  // inert. `Linear` is used, specifically, because it is the cheapest real
  // order to evaluate if a padding axis were ever (incorrectly) read -- a
  // 2-tap linear weight/index computation, versus e.g. a 7-tap SeventhOrder
  // one -- making any such latent bug cheap rather than silently expensive.
  // Matches `bound::BoundVecN`'s analogous choice of `type::Zero` (that
  // enum's own semantic default) for the same never-read padding purpose.
  inline FF_CUHOSTDEV SplineVecN(const type * v, int ndim)
  {
    for (int d = 0; d < max_ndim; ++d)
      s[d] = static_cast<int8_t>(d < ndim ? v[d] : type::Linear);
  }

  inline FF_CUHOSTDEV type operator[] (int d) const
  { return static_cast<type>(s[d]); }
};

using SplineVec = SplineVecN<3>;

FF_NAMESPACE_END(spline)

using spline_t = spline::type;
template <spline_t... S> using Spline = meta::Tuple<spline_t, S...>;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                 STATIC / DYNAMIC SPLINE BUILD POLICY
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Same story as the boundary conditions in bounds.h, one axis further out.
// `pushpull` templates on the interpolation order *and* the boundary condition
// *and* ndim *and* the two element types, so instantiating all eight orders
// statically multiplies an already-large matrix by eight. That is affordable on
// the CPU but not under nvcc, where every combination becomes its own
// `__global__` kernel for `ptxas` to schedule and register-allocate.
//
// `spline::type::Dynamic` selects a *runtime* implementation instead: a single
// instantiation whose order is read from a `spline::SplineVec` at run time.
// Which orders keep a dedicated (faster) static instantiation is a build-time
// choice, for whoever compiles the library:
//
//     -DFF_STATIC_SPLINES=0                 // everything runtime (smallest)
//     -DFF_STATIC_SPLINES=1                 // everything static  (fastest)
//     -DFF_STATIC_SPLINES=0 \
//     -DFF_STATIC_SPLINE_LINEAR=1 \
//     -DFF_STATIC_SPLINE_CUBIC=1            // static fast path for two of them
//
// The per-order macros are FF_STATIC_SPLINE_{NEAREST,LINEAR,QUADRATIC,CUBIC,
// FOURTHORDER,FIFTHORDER,SIXTHORDER,SEVENTHORDER}; each defaults to
// FF_STATIC_SPLINES. Behaviour is identical either way -- only code size,
// compile cost and per-voxel speed change.
//
// Dispatch layers must write `FF_SPLINE_CUBIC` instead of
// `spline::type::Cubic` when choosing the template argument, and pass the
// *runtime* order along in a `spline::SplineVec` so the Dynamic instantiations
// can recover it.

#ifndef FF_STATIC_SPLINES
#  define FF_STATIC_SPLINES 1
#endif
#ifndef FF_STATIC_SPLINE_NEAREST
#  define FF_STATIC_SPLINE_NEAREST      FF_STATIC_SPLINES
#endif
#ifndef FF_STATIC_SPLINE_LINEAR
#  define FF_STATIC_SPLINE_LINEAR       FF_STATIC_SPLINES
#endif
#ifndef FF_STATIC_SPLINE_QUADRATIC
#  define FF_STATIC_SPLINE_QUADRATIC    FF_STATIC_SPLINES
#endif
#ifndef FF_STATIC_SPLINE_CUBIC
#  define FF_STATIC_SPLINE_CUBIC        FF_STATIC_SPLINES
#endif
#ifndef FF_STATIC_SPLINE_FOURTHORDER
#  define FF_STATIC_SPLINE_FOURTHORDER  FF_STATIC_SPLINES
#endif
#ifndef FF_STATIC_SPLINE_FIFTHORDER
#  define FF_STATIC_SPLINE_FIFTHORDER   FF_STATIC_SPLINES
#endif
#ifndef FF_STATIC_SPLINE_SIXTHORDER
#  define FF_STATIC_SPLINE_SIXTHORDER   FF_STATIC_SPLINES
#endif
#ifndef FF_STATIC_SPLINE_SEVENTHORDER
#  define FF_STATIC_SPLINE_SEVENTHORDER FF_STATIC_SPLINES
#endif

#define FF_SPLINE_IF_1(NAME)   ::FF_NS::spline::type::NAME
#define FF_SPLINE_IF_0(NAME)   ::FF_NS::spline::type::Dynamic
#define FF_SPLINE_CAT_(A, B)   A##B
#define FF_SPLINE_CAT(A, B)    FF_SPLINE_CAT_(A, B)
#define FF_SPLINE_SEL(FLAG, NAME) FF_SPLINE_CAT(FF_SPLINE_IF_, FLAG)(NAME)

// Template argument to use for each interpolation order:
// the order itself when it is statically compiled, `Dynamic` otherwise.
#define FF_SPLINE_NEAREST      FF_SPLINE_SEL(FF_STATIC_SPLINE_NEAREST,      Nearest)
#define FF_SPLINE_LINEAR       FF_SPLINE_SEL(FF_STATIC_SPLINE_LINEAR,       Linear)
#define FF_SPLINE_QUADRATIC    FF_SPLINE_SEL(FF_STATIC_SPLINE_QUADRATIC,    Quadratic)
#define FF_SPLINE_CUBIC        FF_SPLINE_SEL(FF_STATIC_SPLINE_CUBIC,        Cubic)
#define FF_SPLINE_FOURTHORDER  FF_SPLINE_SEL(FF_STATIC_SPLINE_FOURTHORDER,  FourthOrder)
#define FF_SPLINE_FIFTHORDER   FF_SPLINE_SEL(FF_STATIC_SPLINE_FIFTHORDER,   FifthOrder)
#define FF_SPLINE_SIXTHORDER   FF_SPLINE_SEL(FF_STATIC_SPLINE_SIXTHORDER,   SixthOrder)
#define FF_SPLINE_SEVENTHORDER FF_SPLINE_SEL(FF_STATIC_SPLINE_SEVENTHORDER, SeventhOrder)

FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(spline)

using FF_NS::spline::type;
using FF_NS::spline::SplineVec;

FF_NAMESPACE_BEGIN(_spline)

  // Forward declarations for the few basis functions that are referenced
  // by a lower/earlier-defined function (two-phase name lookup would
  // otherwise fail on these unqualified dependent calls).
  template <typename scalar_t> static inline FF_CUDEV scalar_t fastgrad1(scalar_t x);
  template <typename scalar_t> static inline FF_CUDEV scalar_t fasthess5(scalar_t x);
  template <typename scalar_t> static inline FF_CUDEV scalar_t fasthess6(scalar_t x);
  template <typename scalar_t> static inline FF_CUDEV scalar_t fasthess7(scalar_t x);

  // --- order 0 -------------------------------------------------------

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t weight0(scalar_t x) {
    x = fabs(x);
    return x < 0.5 ? static_cast<scalar_t>(1) : static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastweight0(scalar_t x) {
    x = fabs(x);
    return static_cast<scalar_t>(1);
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t grad0(scalar_t x) {
    return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastgrad0(scalar_t x) {
    return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t hess0(scalar_t x) {
    return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fasthess0(scalar_t x) {
    return static_cast<scalar_t>(0);
  }

  template <typename scalar_t, typename offset_t>
  static inline FF_CUDEV void bounds0(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(round(x));
    upp = low;
  }

  // --- order 1 -------------------------------------------------------

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t weight1(scalar_t x) {
    x = fabs(x);
    return x < 1 ? static_cast<scalar_t>(1) - x : static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastweight1(scalar_t x) {
    return static_cast<scalar_t>(1) - x;
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t grad1(scalar_t x) {
    if (fabs(x) >= 1) return static_cast<scalar_t>(0);
    return fastgrad1(x);
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastgrad1(scalar_t x) {
    return static_cast<scalar_t>(-1);
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t hess1(scalar_t x) {
    return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fasthess1(scalar_t x) {
    return static_cast<scalar_t>(0);
  }

  template <typename scalar_t, typename offset_t>
  static inline FF_CUDEV void bounds1(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(floor(x));
    upp = low + 1;
  }

  // --- order 2 -------------------------------------------------------

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t weight2(scalar_t x) {
    x = fabs(x);
    if ( x < 0.5 )
    {
      return 0.75 - x * x;
    }
    else if ( x < 1.5 )
    {
      x = 1.5 - x;
      return 0.5 * x * x;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastweight2(scalar_t x) {
    if ( x < 0.5 )
    {
      return 0.75 - x * x;
    }
    else
    {
      x = 1.5 - x;
      return 0.5 * x * x;
    }
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t grad2(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < 0.5 )
    {
      x = -2. * x;
    }
    else if ( x < 1.5 )
    {
      x = x - 1.5;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastgrad2(scalar_t x) {
    if ( x < 0.5 )
    {
      x = -2. * x;
    }
    else
    {
      x = x - 1.5;
    }
    return x;
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t hess2(scalar_t x) {
    x = fabs(x);
    if ( x < 0.5 )
    {
      return static_cast<scalar_t>(-2.);
    }
    else if ( x < 1.5 )
    {
      return static_cast<scalar_t>(1.);
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fasthess2(scalar_t x) {
    if ( x < 0.5 )
    {
      return static_cast<scalar_t>(-2.);
    }
    else
    {
      return static_cast<scalar_t>(1.);
    }
  }

  template <typename scalar_t, typename offset_t>
  static inline FF_CUDEV void bounds2(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(floor(x-.5));
    upp = low + 2;
  }

  // --- order 3 -------------------------------------------------------

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t weight3(scalar_t x) {
    x = fabs(x);
    if ( x < 1. )
    {
      return ( x * x * (x - 2.) * 3. + 4. ) / 6.;
    }
    else if ( x < 2. )
    {
      x = 2. - x;
      return ( x * x * x ) / 6.;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastweight3(scalar_t x) {
    if ( x < 1. )
    {
      return ( x * x * (x - 2.) * 3. + 4. ) / 6.;
    }
    else
    {
      x = 2. - x;
      return ( x * x * x ) / 6.;
    }
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t grad3(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < 1. )
    {
      x = x * ( x * 1.5 - 2. );
    }
    else if ( x < 2. )
    {
      x = 2. - x;
      x = - ( x * x ) * 0.5;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastgrad3(scalar_t x) {
    if ( x < 1. )
    {
      x = x * ( x * 1.5 - 2. );
    }
    else
    {
      x = 2. - x;
      x = - ( x * x ) * 0.5;
    }
    return x;
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t hess3(scalar_t x) {
    x = fabs(x);
    if ( x < 1. )
    {
      return x * 3. - 2.;
    }
    else if ( x < 2. )
    {
      return 2. - x;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fasthess3(scalar_t x) {
    if ( x < 1. )
    {
      return x * 3. - 2.;
    }
    else
    {
      return 2. - x;
    }
  }


  template <typename scalar_t, typename offset_t>
  static inline FF_CUDEV void bounds3(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(floor(x-1.));
    upp = low + 3;
  }

  // --- order 4 -------------------------------------------------------

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t weight4(scalar_t x) {
    x = fabs(x);
    if ( x < 0.5 )
    {
      x *= x;
      return x * ( x * 0.25 - 0.625 ) + 115. / 192.;
    }
    else if ( x < 1.5 )
    {
      return x * ( x * ( x * ( 5. - x ) / 6. - 1.25 ) + 5. / 24. ) + 55. / 96.;
    }
    else if ( x < 2.5 )
    {
      x -= 2.5;
      x *= x;
      return ( x * x ) / 24.;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastweight4(scalar_t x) {
    if ( x < 0.5 )
    {
      x *= x;
      return x * ( x * 0.25 - 0.625 ) + 115. / 192.;
    }
    else if ( x < 1.5 )
    {
      return x * ( x * ( x * ( 5. - x ) / 6. - 1.25 ) + 5. / 24. ) + 55. / 96.;
    }
    else
    {
      x -= 2.5;
      x *= x;
      return ( x * x ) / 24.;
    }
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t grad4(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < 0.5 )
    {
      x = x * ( x * x - 1.25 );
    }
    else if ( x < 1.5 )
    {
      x = x * ( x * ( x * ( -2. / 3. ) + 2.5 ) - 2.5 ) + 5. / 24.;
    }
    else if ( x < 2.5 )
    {
      x = x * 2. - 5.;
      x = ( x * x * x ) / 48.;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastgrad4(scalar_t x) {
    if ( x < 0.5 )
    {
      x = x * ( x * x - 1.25 );
    }
    else if ( x < 1.5 )
    {
      x = x * ( x * ( x * ( -2. / 3. ) + 2.5 ) - 2.5 ) + 5. / 24.;
    }
    else
    {
      x = x * 2. - 5.;
      x = ( x * x * x ) / 48.;
    }
    return x;
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t hess4(scalar_t x) {
    x = fabs(x);
    if ( x < 0.5 )
    {
      return ( x * x ) * 3. - 1.25;
    }
    else if ( x < 1.5 )
    {
      return  x * ( x * ( -2. ) + 5. ) - 2.5;
    }
    else if ( x < 2.5 )
    {
      x = x * 2. - 5.;
      return ( x * x ) / 8.;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fasthess4(scalar_t x) {
    if ( x < 0.5 )
    {
      return ( x * x ) * 3. - 1.25;
    }
    else if ( x < 1.5 )
    {
      return  x * ( x * ( -2. ) + 5. ) - 2.5;
    }
    else
    {
      x = x * 2. - 5.;
      return ( x * x ) / 8.;
    }
  }

  template <typename scalar_t, typename offset_t>
  static inline FF_CUDEV void bounds4(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(floor(x-1.5));
    upp = low + 4;
  }

  // --- order 5 -------------------------------------------------------

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t weight5(scalar_t x) {
    x = fabs(x);
    if ( x < 1. )
    {
      scalar_t f = x * x;
      return f * ( f * ( 0.25 - x * ( 1. / 12. ) ) - 0.5 ) + 0.55;
    }
    else if ( x < 2. )
    {
      return x * ( x * ( x * ( x * ( x * ( 1. / 24. ) - 0.375 ) + 1.25 ) -
             1.75 ) + 0.625 ) + 0.425;
    }
    else if ( x < 3. )
    {
      scalar_t f = 3. - x;
      x = f * f;
      return f * x * x * ( 1. / 120. );
    }
    else
      return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastweight5(scalar_t x) {
    if ( x < 1. )
    {
      scalar_t f = x * x;
      return f * ( f * ( 0.25 - x * ( 1. / 12. ) ) - 0.5 ) + 0.55;
    }
    else if ( x < 2. )
    {
      return x * ( x * ( x * ( x * ( x * ( 1. / 24. ) - 0.375 ) + 1.25 ) -
             1.75 ) + 0.625 ) + 0.425;
    }
    else
    {
      scalar_t f = 3. - x;
      x = f * f;
      return f * x * x * ( 1. / 120. );
    }
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t grad5(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < 1. )
    {
      x = x * ( x * ( x * ( x * ( -5. / 12. ) + 1. ) ) - 1. );
    }
    else if ( x < 2. )
    {
      x = x * ( x * ( x * ( x * ( 5. / 24. ) - 1.5 ) + 3.75 ) - 3.5 ) + 0.625;
    }
    else if ( x < 3. )
    {
      x -= 3.;
      x *= x;
      x = - ( x * x ) / 24.;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastgrad5(scalar_t x) {
    if ( x < 1. )
    {
      x = x * ( x * ( x * ( x * ( -5. / 12. ) + 1. ) ) - 1. );
    }
    else if ( x < 2. )
    {
      x = x * ( x * ( x * ( x * ( 5. / 24. ) - 1.5 ) + 3.75 ) - 3.5 ) + 0.625;
    }
    else
    {
      x -= 3.;
      x *= x;
      x = - ( x * x ) / 24.;
    }
    return x;
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t hess5(scalar_t x) {
    x = fabs(x);
    if ( x >= 3. )
        return static_cast<scalar_t>(0);
    else
        return fasthess5(x);
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fasthess5(scalar_t x) {
    if ( x < 1. )
        return - (x * x) * (x * (5./3.) - 3.) - 1.;
    else if ( x < 2. )
        return x * (x * (x * (5./6.) - 9./2.) + 15./2.) - 7./2.;
    else
        return 9./2. - x * (x * (x/6. - 3./2.) + 9./2.);
  }

  template <typename scalar_t, typename offset_t>
  static inline FF_CUDEV void bounds5(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(floor(x-2.));
    upp = low + 5;
  }

  // --- order 6 -------------------------------------------------------

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t weight6(scalar_t x) {
    x = fabs(x);
    if ( x < 0.5 )
    {
      x *= x;
      return x * ( x * ( 7. / 48. - x * ( 1. / 36. ) ) - 77. / 192. ) +
             5887. / 11520.0;
    }
    else if ( x < 1.5 )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( 1. / 48. ) - 7. / 48. ) +
             0.328125 ) - 35. / 288. ) - 91. / 256. ) - 7. / 768. ) +
             7861. / 15360.0;
    }
    else if ( x < 2.5 )
    {
      return x * ( x * ( x * ( x * ( x * ( 7. / 60. - x * ( 1. / 120. ) ) -
             0.65625 ) + 133. / 72. ) - 2.5703125 ) + 1267. / 960. ) +
             1379. / 7680.0;
    }
    else if ( x < 3.5 )
    {
      x -= 3.5;
      x *= x * x;
      return x * x * ( 1. / 720. );
    }
    else
      return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastweight6(scalar_t x) {
    if ( x < 0.5 )
    {
      x *= x;
      return x * ( x * ( 7. / 48. - x * ( 1. / 36. ) ) - 77. / 192. ) +
             5887. / 11520.0;
    }
    else if ( x < 1.5 )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( 1. / 48. ) - 7. / 48. ) +
             0.328125 ) - 35. / 288. ) - 91. / 256. ) - 7. / 768. ) +
             7861. / 15360.0;
    }
    else if ( x < 2.5 )
    {
      return x * ( x * ( x * ( x * ( x * ( 7. / 60. - x * ( 1. / 120. ) ) -
             0.65625 ) + 133. / 72. ) - 2.5703125 ) + 1267. / 960. ) +
             1379. / 7680.0;
    }
    else
    {
      x -= 3.5;
      x *= x * x;
      return x * x * ( 1. / 720. );
    }
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t grad6(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < .5 )
    {
      scalar_t x2 = x * x;
      x = x * ( x2 * ( 7. / 12. ) - ( x2 * x2 ) / 6.- 77./96. );
    }
    else if ( x < 1.5 )
    {
      x = x * ( x * ( x * ( x * ( x * 0.125 - 35./48. ) + 1.3125 )
             - 35./96. ) - 0.7109375 ) - 7.0/768.0;
    }
    else if ( x < 2.5 )
    {
      x = x * ( x * ( x * ( x * ( x * (-1./20.) + 7./12. )
             - 2.625 ) + 133./24. ) - 5.140625 ) + 1267./960.;
    }
    else if ( x < 3.5 )
    {
      x *= 2.;
      x -= 7.;
      scalar_t x2 = x*x;
      x = (x2 * x2 * x ) / 3840.;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastgrad6(scalar_t x) {
    if ( x < .5 )
    {
      scalar_t x2 = x * x;
      x = x * ( x2 * ( 7. / 12. ) - ( x2 * x2 ) / 6.- 77./96. );
    }
    else if ( x < 1.5 )
    {
      x = x * ( x * ( x * ( x * ( x * 0.125 - 35./48. ) + 1.3125 )
             - 35./96. ) - 0.7109375 ) - 7.0/768.0;
    }
    else if ( x < 2.5 )
    {
      x = x * ( x * ( x * ( x * ( x * (-1./20.) + 7./12. )
             - 2.625 ) + 133./24. ) - 5.140625 ) + 1267./960.;
    }
    else
    {
      x *= 2.;
      x -= 7.;
      scalar_t x2 = x*x;
      x = (x2 * x2 * x ) / 3840.;
    }
    return x;
  }


  template <typename scalar_t>
  static inline FF_CUDEV scalar_t hess6(scalar_t x) {
    x = fabs(x);
    if ( x >= 3.5 )
        return static_cast<scalar_t>(0);
    else
        return fasthess6(x);
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fasthess6(scalar_t x) {
    if ( x < 0.5 ) {
        x *= x;
        return - x * (x * (5./6) - 7./4.) - 77./96.;
    } else if ( x < 1.5 )
        return (x * (x * (x * (x * (5./8.) - 35./12.) + 63./16.) - 35./48.) - 91./128.);
    else if ( x < 2.5 )
        return -(x * (x * (x * (x/4. - 7./3.) + 63./8.) - 133./12.) + 329./64.);
    else
        return (x * (x * (x * (x/24. - 7./12.) + 49./16.) - 343./48.) + 2401./384.);

  }

  template <typename scalar_t, typename offset_t>
  static inline FF_CUDEV void bounds6(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(floor(x-2.5));
    upp = low + 6;
  }

  // --- order 7 -------------------------------------------------------

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t weight7(scalar_t x) {
    x = fabs(x);
    if ( x < 1. )
    {
      scalar_t f = x * x;
      return f * ( f * ( f * ( x * ( 1. / 144. ) - 1. / 36. ) + 1. / 9. ) -
             1. / 3. ) + 151. / 315.0;
    }
    else if ( x < 2. )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( 0.05 - x * ( 1. / 240. ) ) -
             7. / 30. ) + 0.5 ) - 7. / 18. ) - 0.1 ) - 7. / 90. ) +
             103. / 210.0;
    }
    else if ( x < 3. )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( x * ( 1. / 720. ) -
             1. / 36. ) + 7. / 30. ) - 19. / 18. ) + 49. / 18. ) -
             23. / 6. ) + 217. / 90. ) - 139. / 630.0;
    }
    else if ( x < 4. )
    {
      scalar_t f = 4. - x;
      x = f * f * f;
      return ( x * x * f ) / 5040.;
    }
    else
      return static_cast<scalar_t>(0);
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastweight7(scalar_t x) {
    if ( x < 1. )
    {
      scalar_t f = x * x;
      return f * ( f * ( f * ( x * ( 1. / 144. ) - 1. / 36. ) + 1. / 9. )
             - 1. / 3. ) + 151. / 315.0;
    }
    else if ( x < 2. )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( 0.05 - x * ( 1. / 240. ) )
             - 7. / 30. ) + 0.5 ) - 7. / 18. ) - 0.1 ) - 7. / 90. )
             + 103. / 210.0;
    }
    else if ( x < 3. )
    {
      return x * ( x * ( x * ( x * ( x * ( x * ( x * ( 1. / 720. )
             - 1. / 36. ) + 7. / 30. ) - 19. / 18. ) + 49. / 18. )
             - 23. / 6. ) + 217. / 90. ) - 139. / 630.0;
    }
    else
    {
      scalar_t f = 4. - x;
      x = f * f * f;
      return ( x * x * f ) / 5040.;
    }
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t grad7(scalar_t x) {
    bool neg = x < 0;
    if (neg) x = -x;
    if ( x < 1. )
    {
      scalar_t x2 = x * x;
      x = x * ( x2 *( x2 * ( x * ( 7. / 144. )
             - 1. / 6. ) + 4. / 9. ) - 2. / 3. );
    }
    else if ( x < 2. )
    {
      x = x * ( x * ( x * ( x * ( x * ( x * ( -7. / 240. ) + 3. / 10. )
             - 7. / 6. ) + 2. ) - 7. / 6. ) - 1. / 5. ) - 7. / 90.;
    }
    else if ( x < 3. )
    {
      x = x * ( x * (x * ( x * ( x * ( x * ( 7. / 720. ) - 1. / 6. )
             + 7. / 6. ) - 38. / 9. ) + 49. / 6. ) - 23. / 3. ) + 217. / 90.;
    }
    else if ( x < 4. )
    {
      x -= 4;
      x *= x*x;
      x *= x;
      x = - x / 720.;
    }
    else
    {
      return static_cast<scalar_t>(0);
    }
    if (neg) x = -x;
    return x;
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fastgrad7(scalar_t x) {
    if ( x < 1. )
    {
      scalar_t x2 = x * x;
      x = x * ( x2 *( x2 * ( x * ( 7. / 144. )
             - 1. / 6. ) + 4. / 9. ) - 2. / 3. );
    }
    else if ( x < 2. )
    {
      x = x * ( x * ( x * ( x * ( x * ( x * ( -7. / 240. ) + 3. / 10. )
             - 7. /6. ) + 2. ) - 7. / 6. ) - 1. / 5. ) - 7. / 90.;
    }
    else if ( x < 3. )
    {
      x = x * ( x * (x * ( x * ( x * ( x * ( 7. / 720. ) - 1. / 6. )
             + 7. / 6. ) - 38. / 9. ) + 49. / 6. ) - 23. / 3. ) + 217. / 90.;
    }
    else
    {
      x -= 4;
      x *= x*x;
      x *= x;
      x = - x / 720.;
    }
    return x;
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t hess7(scalar_t x) {
    x = fabs(x);
    if ( x >= 4. )
        return static_cast<scalar_t>(0);
    else
        return fasthess7(x);
  }

  template <typename scalar_t>
  static inline FF_CUDEV scalar_t fasthess7(scalar_t x) {
    if ( x < 1. ) {
        scalar_t x2 = x * x;
        return x2 * (x2 * (x * (7./24.) - 5./6.) + 4./3.) - 2./3.;
    } else if ( x < 2. )
        return - (x * (x * (x * (x * (x * (7./40.) - 3./2.) + 14./3.) - 6.) + 7./3.) + 1./5.);
    else if ( x < 3. )
        return (x * (x * (x * (x * (x * (7./120.) - 5./6.) + 14./3.) - 38./3.) + 49./3.) - 23./3.);
    else
        return - (x * (x * (x * (x * (x/120. - 1./6.) + 4./3.) - 16./3.) + 32./3.) - 128./15.);
  }

  template <typename scalar_t, typename offset_t>
  static inline FF_CUDEV void bounds7(scalar_t x, offset_t & low, offset_t & upp) {
    low = static_cast<offset_t>(floor(x-3.));
    upp = low + 7;
  }


FF_NAMESPACE_END(_spline)

template <typename scalar_t>
struct _value_fn { typedef scalar_t(*type)(scalar_t); };

template <typename scalar_t>
using _value_fn_t = typename _value_fn<scalar_t>::type;

template <typename scalar_t>
static inline FF_CUDEV _value_fn_t<scalar_t>
weight_fn(type spline_type) {
  switch (spline_type) {
    case type::Nearest:      return _spline::weight0<scalar_t>;
    case type::Linear:       return _spline::weight1<scalar_t>;
    case type::Quadratic:    return _spline::weight2<scalar_t>;
    case type::Cubic:        return _spline::weight3<scalar_t>;
    case type::FourthOrder:  return _spline::weight4<scalar_t>;
    case type::FifthOrder:   return _spline::weight5<scalar_t>;
    case type::SixthOrder:   return _spline::weight6<scalar_t>;
    case type::SeventhOrder: return _spline::weight7<scalar_t>;
    default:                 return _spline::weight1<scalar_t>;
  }
}

template <typename scalar_t>
static inline FF_CUDEV scalar_t
weight(type spline_type, scalar_t x) {
  return weight_fn<scalar_t>(spline_type)(x);
  // switch (spline_type) {
  //   case type::Nearest:      return _spline::weight0(x);
  //   case type::Linear:       return _spline::weight1(x);
  //   case type::Quadratic:    return _spline::weight2(x);
  //   case type::Cubic:        return _spline::weight3(x);
  //   case type::FourthOrder:  return _spline::weight4(x);
  //   case type::FifthOrder:   return _spline::weight5(x);
  //   case type::SixthOrder:   return _spline::weight6(x);
  //   case type::SeventhOrder: return _spline::weight7(x);
  //   default:                 return _spline::weight1(x);
  // }
}

template <typename scalar_t>
static inline FF_CUDEV _value_fn_t<scalar_t>
fastweight_fn(type spline_type) {
  switch (spline_type) {
    case type::Nearest:      return _spline::fastweight0<scalar_t>;
    case type::Linear:       return _spline::fastweight1<scalar_t>;
    case type::Quadratic:    return _spline::fastweight2<scalar_t>;
    case type::Cubic:        return _spline::fastweight3<scalar_t>;
    case type::FourthOrder:  return _spline::fastweight4<scalar_t>;
    case type::FifthOrder:   return _spline::fastweight5<scalar_t>;
    case type::SixthOrder:   return _spline::fastweight6<scalar_t>;
    case type::SeventhOrder: return _spline::fastweight7<scalar_t>;
    default:                 return _spline::fastweight1<scalar_t>;
  }
}

template <typename scalar_t>
static inline FF_CUDEV scalar_t
fastweight(type spline_type, scalar_t x) {
  return fastweight_fn<scalar_t>(spline_type)(x);
  // switch (spline_type) {
  //   case type::Nearest:      return _spline::fastweight0(x);
  //   case type::Linear:       return _spline::fastweight1(x);
  //   case type::Quadratic:    return _spline::fastweight2(x);
  //   case type::Cubic:        return _spline::fastweight3(x);
  //   case type::FourthOrder:  return _spline::fastweight4(x);
  //   case type::FifthOrder:   return _spline::fastweight5(x);
  //   case type::SixthOrder:   return _spline::fastweight6(x);
  //   case type::SeventhOrder: return _spline::fastweight7(x);
  //   default:                 return _spline::fastweight1(x);
  // }
}

template <typename scalar_t>
static inline FF_CUDEV _value_fn_t<scalar_t>
grad_fn(type spline_type) {
  switch (spline_type) {
    case type::Nearest:      return _spline::grad0<scalar_t>;
    case type::Linear:       return _spline::grad1<scalar_t>;
    case type::Quadratic:    return _spline::grad2<scalar_t>;
    case type::Cubic:        return _spline::grad3<scalar_t>;
    case type::FourthOrder:  return _spline::grad4<scalar_t>;
    case type::FifthOrder:   return _spline::grad5<scalar_t>;
    case type::SixthOrder:   return _spline::grad6<scalar_t>;
    case type::SeventhOrder: return _spline::grad7<scalar_t>;
    default:                 return _spline::grad1<scalar_t>;
  }
}

template <typename scalar_t>
static inline FF_CUDEV scalar_t
grad(type spline_type, scalar_t x) {
  return grad_fn<scalar_t>(spline_type)(x);
  // switch (spline_type) {
  //   case type::Nearest:      return _spline::grad0(x);
  //   case type::Linear:       return _spline::grad1(x);
  //   case type::Quadratic:    return _spline::grad2(x);
  //   case type::Cubic:        return _spline::grad3(x);
  //   case type::FourthOrder:  return _spline::grad4(x);
  //   case type::FifthOrder:   return _spline::grad5(x);
  //   case type::SixthOrder:   return _spline::grad6(x);
  //   case type::SeventhOrder: return _spline::grad7(x);
  //   default:                 return _spline::grad1(x);
  // }
}

template <typename scalar_t>
static inline FF_CUDEV _value_fn_t<scalar_t>
fastgrad_fn(type spline_type) {
  switch (spline_type) {
    case type::Nearest:      return _spline::fastgrad0<scalar_t>;
    case type::Linear:       return _spline::fastgrad1<scalar_t>;
    case type::Quadratic:    return _spline::fastgrad2<scalar_t>;
    case type::Cubic:        return _spline::fastgrad3<scalar_t>;
    case type::FourthOrder:  return _spline::fastgrad4<scalar_t>;
    case type::FifthOrder:   return _spline::fastgrad5<scalar_t>;
    case type::SixthOrder:   return _spline::fastgrad6<scalar_t>;
    case type::SeventhOrder: return _spline::fastgrad7<scalar_t>;
    default:                 return _spline::fastgrad1<scalar_t>;
  }
}

template <typename scalar_t>
static inline FF_CUDEV scalar_t
fastgrad(type spline_type, scalar_t x) {
  return fastgrad_fn<scalar_t>(spline_type)(x);
  // switch (spline_type) {
  //   case type::Nearest:      return _spline::fastgrad0(x);
  //   case type::Linear:       return _spline::fastgrad1(x);
  //   case type::Quadratic:    return _spline::fastgrad2(x);
  //   case type::Cubic:        return _spline::fastgrad3(x);
  //   case type::FourthOrder:  return _spline::fastgrad4(x);
  //   case type::FifthOrder:   return _spline::fastgrad5(x);
  //   case type::SixthOrder:   return _spline::fastgrad6(x);
  //   case type::SeventhOrder: return _spline::fastgrad7(x);
  //   default:                 return _spline::fastgrad1(x);
  // }
}

template <typename scalar_t>
static inline FF_CUDEV _value_fn_t<scalar_t>
hess_fn(type spline_type) {
  switch (spline_type) {
    case type::Nearest:      return _spline::hess0<scalar_t>;
    case type::Linear:       return _spline::hess1<scalar_t>;
    case type::Quadratic:    return _spline::hess2<scalar_t>;
    case type::Cubic:        return _spline::hess3<scalar_t>;
    case type::FourthOrder:  return _spline::hess4<scalar_t>;
    case type::FifthOrder:   return _spline::hess0<scalar_t>; // notimplemented
    case type::SixthOrder:   return _spline::hess0<scalar_t>; // notimplemented
    case type::SeventhOrder: return _spline::hess0<scalar_t>; // notimplemented
    default:                 return _spline::hess1<scalar_t>;
  }
}

template <typename scalar_t>
static inline FF_CUDEV scalar_t
hess(type spline_type, scalar_t x) {
  return hess_fn<scalar_t>(spline_type)(x);
  // switch (spline_type) {
  //   case type::Nearest:      return _spline::hess0(x);
  //   case type::Linear:       return _spline::hess1(x);
  //   case type::Quadratic:    return _spline::hess2(x);
  //   case type::Cubic:        return _spline::hess3(x);
  //   case type::FourthOrder:  return _spline::hess4(x);
  //   case type::FifthOrder:   return _spline::hess0(x); // notimplemented
  //   case type::SixthOrder:   return _spline::hess0(x); // notimplemented
  //   case type::SeventhOrder: return _spline::hess0(x); // notimplemented
  //   default:                 return _spline::hess1(x);
  // }
}

template <typename scalar_t>
static inline FF_CUDEV _value_fn_t<scalar_t>
fasthess_fn(type spline_type) {
  switch (spline_type) {
    case type::Nearest:      return _spline::fasthess0<scalar_t>;
    case type::Linear:       return _spline::fasthess1<scalar_t>;
    case type::Quadratic:    return _spline::fasthess2<scalar_t>;
    case type::Cubic:        return _spline::fasthess3<scalar_t>;
    case type::FourthOrder:  return _spline::fasthess4<scalar_t>;
    case type::FifthOrder:   return _spline::fasthess0<scalar_t>; // notimplemented
    case type::SixthOrder:   return _spline::fasthess0<scalar_t>; // notimplemented
    case type::SeventhOrder: return _spline::fasthess0<scalar_t>; // notimplemented
    default:                 return _spline::fasthess1<scalar_t>;
  }
}

template <typename scalar_t>
static inline FF_CUDEV scalar_t
fasthess(type spline_type, scalar_t x) {
  return fasthess_fn<scalar_t>(spline_type)(x);
  // switch (spline_type) {
  //   case type::Nearest:      return _spline::fasthess0(x);
  //   case type::Linear:       return _spline::fasthess1(x);
  //   case type::Quadratic:    return _spline::fasthess2(x);
  //   case type::Cubic:        return _spline::fasthess3(x);
  //   case type::FourthOrder:  return _spline::fasthess4(x);
  //   case type::FifthOrder:   return _spline::fasthess0(x); // notimplemented
  //   case type::SixthOrder:   return _spline::fasthess0(x); // notimplemented
  //   case type::SeventhOrder: return _spline::fasthess0(x); // notimplemented
  //   default:                 return _spline::fasthess1(x);
  // }
}

template <typename scalar_t, typename offset_t>
struct _bounds_fn { typedef void(*type)(scalar_t, offset_t &, offset_t &); };

template <typename scalar_t, typename offset_t>
using _bounds_fn_t = typename _bounds_fn<scalar_t, offset_t>::type;

template <typename scalar_t, typename offset_t>
static inline FF_CUDEV _bounds_fn_t<scalar_t, offset_t>
bounds_fn(type spline_type) {
  switch (spline_type) {
    case type::Nearest:      return _spline::bounds0<scalar_t, offset_t>;
    case type::Linear:       return _spline::bounds1<scalar_t, offset_t>;
    case type::Quadratic:    return _spline::bounds2<scalar_t, offset_t>;
    case type::Cubic:        return _spline::bounds3<scalar_t, offset_t>;
    case type::FourthOrder:  return _spline::bounds4<scalar_t, offset_t>;
    case type::FifthOrder:   return _spline::bounds5<scalar_t, offset_t>;
    case type::SixthOrder:   return _spline::bounds6<scalar_t, offset_t>;
    case type::SeventhOrder: return _spline::bounds7<scalar_t, offset_t>;
    default:                 return _spline::bounds1<scalar_t, offset_t>;
  }
}

template <typename scalar_t, typename offset_t>
static inline FF_CUDEV void
bounds(type spline_type, scalar_t x, offset_t & low, offset_t & upp)
{
  return bounds_fn<scalar_t, offset_t>(spline_type)(x, low, upp);
  // switch (spline_type) {
  //   case type::Nearest:      return _spline::bounds0(x, low, upp);
  //   case type::Linear:       return _spline::bounds1(x, low, upp);
  //   case type::Quadratic:    return _spline::bounds2(x, low, upp);
  //   case type::Cubic:        return _spline::bounds3(x, low, upp);
  //   case type::FourthOrder:  return _spline::bounds4(x, low, upp);
  //   case type::FifthOrder:   return _spline::bounds5(x, low, upp);
  //   case type::SixthOrder:   return _spline::bounds6(x, low, upp);
  //   case type::SeventhOrder: return _spline::bounds7(x, low, upp);
  //   default:                 return _spline::bounds1(x, low, upp);
  // }
}


template <type I> struct utils {};

#define FF_INTERPOL_UTILS(NAME, ORDER) \
template <> struct utils<type::NAME> { \
    template <typename scalar_t> \
    static inline FF_CUDEV scalar_t \
    weight(scalar_t x) { return _spline::weight##ORDER(x); } \
    template <typename scalar_t> \
    static inline FF_CUDEV scalar_t \
    fastweight(scalar_t x) { return _spline::fastweight##ORDER(x); } \
    template <typename scalar_t> \
    static inline FF_CUDEV scalar_t \
    grad(scalar_t x) { return _spline::grad##ORDER(x); } \
    template <typename scalar_t> \
    static inline FF_CUDEV scalar_t \
    fastgrad(scalar_t x) { return _spline::fastgrad##ORDER(x); } \
    template <typename scalar_t> \
    static inline FF_CUDEV scalar_t \
    hess(scalar_t x) { return _spline::hess##ORDER(x); } \
    template <typename scalar_t> \
    static inline FF_CUDEV scalar_t \
    fasthess(scalar_t x) { return _spline::fasthess##ORDER(x); } \
    template <typename scalar_t, typename offset_t> \
    static inline FF_CUDEV void \
    bounds(scalar_t x, offset_t & low, offset_t & upp) { return _spline::bounds##ORDER(x, low, upp); } \
};

FF_INTERPOL_UTILS(Nearest, 0)
FF_INTERPOL_UTILS(Linear, 1)
FF_INTERPOL_UTILS(Quadratic, 2)
FF_INTERPOL_UTILS(Cubic, 3)
FF_INTERPOL_UTILS(FourthOrder, 4)
FF_INTERPOL_UTILS(FifthOrder, 5)
FF_INTERPOL_UTILS(SixthOrder, 6)
FF_INTERPOL_UTILS(SeventhOrder, 7)

// NOTE: there is deliberately no `utils<type::Dynamic>` specialisation.
// There used to be one whose methods all returned 0 -- a placeholder that made
// `utils<Dynamic>` *compile* while silently producing all-zero weights (and an
// empty node range) for anything that actually called it. Use `dyn<S>` below
// for the runtime-order path; a `utils<Dynamic>` instantiation is now a
// compile error, which is what it should always have been.

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  STATIC / DYNAMIC SPLINE SELECTOR
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// `dyn<S>` is the stateful counterpart of `utils<S>`, mirroring `bound::dyn<B>`
// exactly: same `weight` / `fastweight` / `grad` / `fastgrad` / `hess` /
// `fasthess` / `bounds` interface, but as (const) member functions of an object
// that may carry the interpolation order at run time.
//
//  - for a real `S`, `dyn<S>` is an empty struct that forwards to `utils<S>`;
//    the compiler folds it away completely (zero cost, identical codegen);
//  - for `type::Dynamic`, `dyn` holds the order as a data member and branches
//    on it per call.
//
// Kernels therefore hold `dyn<IX> spline_utils_x;` members instead of
// `using spline_utils_x = utils<IX>;` aliases, and are constructed from a
// `SplineVec`. A single Dynamic instantiation replaces up to eight static ones.

template <type S> struct dyn
{
    inline FF_CUDEV dyn() {}
    explicit inline FF_CUDEV dyn(type) {}       // runtime value: not needed, ignored

    inline FF_CUDEV type value() const { return S; }

    template <typename scalar_t>
    inline FF_CUDEV scalar_t weight(scalar_t x) const
    { return utils<S>::template weight<scalar_t>(x); }

    template <typename scalar_t>
    inline FF_CUDEV scalar_t fastweight(scalar_t x) const
    { return utils<S>::template fastweight<scalar_t>(x); }

    template <typename scalar_t>
    inline FF_CUDEV scalar_t grad(scalar_t x) const
    { return utils<S>::template grad<scalar_t>(x); }

    template <typename scalar_t>
    inline FF_CUDEV scalar_t fastgrad(scalar_t x) const
    { return utils<S>::template fastgrad<scalar_t>(x); }

    template <typename scalar_t>
    inline FF_CUDEV scalar_t hess(scalar_t x) const
    { return utils<S>::template hess<scalar_t>(x); }

    template <typename scalar_t>
    inline FF_CUDEV scalar_t fasthess(scalar_t x) const
    { return utils<S>::template fasthess<scalar_t>(x); }

    template <typename scalar_t, typename offset_t>
    inline FF_CUDEV void bounds(scalar_t x, offset_t & low, offset_t & upp) const
    { return utils<S>::template bounds<scalar_t, offset_t>(x, low, upp); }

    // Number of nodes in the support: `bounds` always yields exactly this many.
    // Static here, so the surrounding loops keep their compile-time trip count.
    inline FF_CUDEV int nodes() const { return static_cast<int>(S) + 1; }
};

template <> struct dyn<type::Dynamic>
{
    type spl;

    inline FF_CUDEV dyn() : spl(type::Linear) {}
    explicit inline FF_CUDEV dyn(type s) : spl(s) {}

    inline FF_CUDEV type value() const { return spl; }

    // Direct switches, *not* the `weight_fn` / `bounds_fn` function-pointer
    // helpers above: an indirect call cannot be inlined and is expensive on the
    // GPU, whereas a switch over a warp-uniform value is close to free. This is
    // the same trade-off `bound::dyn<Dynamic>` makes.
#define FF_SPLINE_DYN_FWD(NAME)                                               \
    template <typename scalar_t>                                              \
    inline FF_CUDEV scalar_t NAME(scalar_t x) const                              \
    {                                                                         \
      switch (spl) {                                                          \
        case type::Nearest:      return _spline::NAME##0(x);                  \
        case type::Quadratic:    return _spline::NAME##2(x);                  \
        case type::Cubic:        return _spline::NAME##3(x);                  \
        case type::FourthOrder:  return _spline::NAME##4(x);                  \
        case type::FifthOrder:   return _spline::NAME##5(x);                  \
        case type::SixthOrder:   return _spline::NAME##6(x);                  \
        case type::SeventhOrder: return _spline::NAME##7(x);                  \
        default:                 return _spline::NAME##1(x);  /* Linear */    \
      }                                                                       \
    }

    FF_SPLINE_DYN_FWD(weight)
    FF_SPLINE_DYN_FWD(fastweight)
    FF_SPLINE_DYN_FWD(grad)
    FF_SPLINE_DYN_FWD(fastgrad)
    FF_SPLINE_DYN_FWD(hess)
    FF_SPLINE_DYN_FWD(fasthess)
#undef FF_SPLINE_DYN_FWD

    template <typename scalar_t, typename offset_t>
    inline FF_CUDEV void bounds(scalar_t x, offset_t & low, offset_t & upp) const
    {
      switch (spl) {
        case type::Nearest:      return _spline::bounds0(x, low, upp);
        case type::Quadratic:    return _spline::bounds2(x, low, upp);
        case type::Cubic:        return _spline::bounds3(x, low, upp);
        case type::FourthOrder:  return _spline::bounds4(x, low, upp);
        case type::FifthOrder:   return _spline::bounds5(x, low, upp);
        case type::SixthOrder:   return _spline::bounds6(x, low, upp);
        case type::SeventhOrder: return _spline::bounds7(x, low, upp);
        default:                 return _spline::bounds1(x, low, upp);
      }
    }

    // Runtime support size. Callers must use this (not a compile-time bound) to
    // size their loops; the buffers themselves are sized by `SplineBufSize`,
    // which reserves the worst case (8) for Dynamic.
    inline FF_CUDEV int nodes() const
    { return static_cast<int>(static_cast<signed char>(spl)) + 1; }
};


FF_NAMESPACE_END(spline)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)

#endif // FF_SPLINE
