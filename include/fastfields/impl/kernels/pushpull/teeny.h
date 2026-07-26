// Teeny-based pushpull kernels: one separable recursion over the static spatial
// rank D replaces the per-rank 1d/2d/3d hand-unrolled gather/scatter trees.
//
// Design (validated by a clang++/g++ -O3 codegen probe + a fable design pass,
// fastfields-lib#21):
//   * spatial rank D and interpolation order O are COMPILE-TIME (so the K=O+1
//     tap loops fully unroll — no runtime tap loop even at cubic); the boundary
//     condition is RUNTIME (it only drives the per-voxel O(K*D) neighbourhood
//     setup, never the O(K^D * C) accumulation).
//   * The boundary SIGN is folded into the weight at setup (w[k] *= sign), so the
//     recursion is a branchless multiply-accumulate and a runtime bound costs
//     nothing in the hot loop. A zero-boundary out-of-range tap gets weight 0 and
//     its offset clamped in-range (a harmless *0 gather); PUSH additionally skips
//     the zero-weight tap to avoid pointless scatter traffic.
//   * PUSH is the exact adjoint of PULL: identical neighbours/weights, but the
//     leaf scatter-accumulates (anyAtomicAdd) instead of reading.
//
// The kernels take teeny views at the boundary (element type + spatial
// strides/extents come from the view) and do raw offset math inside, mirroring
// teeny/examples/fastfields/pushpull.hpp. The impl layer peels the batch and
// loops the grid voxels; the channel loop lives here (neighbours are hoisted
// above it and reused across channels).
#ifndef FF_PUSHPULL_TEENY
#define FF_PUSHPULL_TEENY
// Standard math headers FIRST: spline.h/bounds.h call unqualified floor/round/
// fabs and rely on `::floor` etc. being visible at their definition point.
// teeny.h introduces tny::floor (tensor overloads), so if these came after it,
// two-phase lookup would wrongly bind the scalar calls to the tensor overloads.
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include "../cuda_switch.h"
#include "../spline.h"
#include "../bounds.h"
#include "../atomic.h"
#include <teeny/teeny.h>

// Portable full-unroll pragma. FF-local until teeny#184's TNY_UNROLL lands in the
// pinned teeny (clang/nvcc honour `#pragma unroll`; gcc needs `GCC unroll N`).
#if defined(__clang__) || defined(__CUDACC__)
#  define FF_PP_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#  define FF_PP_UNROLL _Pragma("GCC unroll 16")
#else
#  define FF_PP_UNROLL
#endif

#define FF_EXTRAPOLATE_TINY 5E-2

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(pushpull)

// max support: order 7 -> 8 taps
static constexpr int FF_PP_MAXK = 8;

// ---- compile-time (static order O) spline selectors -----------------------
// fastweight assumes the sample lies in the support (guaranteed here: taps are
// exactly the support nodes low..low+O), so the range check is skipped.
template <int O, typename scalar_t>
static inline CUDEV scalar_t _fastweight(scalar_t x) {
    if      constexpr (O==0) return spline::_spline::fastweight0<scalar_t>(x);
    else if constexpr (O==1) return spline::_spline::fastweight1<scalar_t>(x);
    else if constexpr (O==2) return spline::_spline::fastweight2<scalar_t>(x);
    else if constexpr (O==3) return spline::_spline::fastweight3<scalar_t>(x);
    else if constexpr (O==4) return spline::_spline::fastweight4<scalar_t>(x);
    else if constexpr (O==5) return spline::_spline::fastweight5<scalar_t>(x);
    else if constexpr (O==6) return spline::_spline::fastweight6<scalar_t>(x);
    else                     return spline::_spline::fastweight7<scalar_t>(x);
}

template <int O, typename scalar_t, typename offset_t>
static inline CUDEV offset_t _low(scalar_t x) {
    offset_t low = 0, upp = 0;
    if      constexpr (O==0) spline::_spline::bounds0(x, low, upp);
    else if constexpr (O==1) spline::_spline::bounds1(x, low, upp);
    else if constexpr (O==2) spline::_spline::bounds2(x, low, upp);
    else if constexpr (O==3) spline::_spline::bounds3(x, low, upp);
    else if constexpr (O==4) spline::_spline::bounds4(x, low, upp);
    else if constexpr (O==5) spline::_spline::bounds5(x, low, upp);
    else if constexpr (O==6) spline::_spline::bounds6(x, low, upp);
    else                     spline::_spline::bounds7(x, low, upp);
    return low;
}

// ---- field-of-view test (runtime extrapolate) -----------------------------
template <typename scalar_t, typename offset_t>
static inline CUDEV bool _infov(int extrapolate, scalar_t x, offset_t n) {
    if (extrapolate == 1) return true;                         // always in
    const scalar_t tiny = static_cast<scalar_t>(FF_EXTRAPOLATE_TINY);
    if (extrapolate == 0)  // limits at voxel centres [-tiny, n-1+tiny]
        return x >= -tiny && x <= static_cast<scalar_t>(n - 1) + tiny;
    // extrapolate == -1 : limits at voxel edges [-0.5-tiny, n-0.5+tiny]
    return x >= static_cast<scalar_t>(-0.5) - tiny
        && x <= static_cast<scalar_t>(n) - static_cast<scalar_t>(0.5) + tiny;
}

// ---- per-axis neighbourhood (sign folded into the weight) ------------------
// off[k] is a pre-multiplied strided offset; when the sign is 0 (zero boundary,
// out of range) the weight is 0 and the offset is clamped to 0 so the gather is
// an in-range multiply-by-zero.
template <typename reduce_t, typename offset_t>
struct _axis { reduce_t w[FF_PP_MAXK]; offset_t off[FF_PP_MAXK]; };

template <int O, typename reduce_t, typename offset_t>
static inline CUDEV _axis<reduce_t, offset_t>
_make_axis(bound_t b, reduce_t coord, offset_t n, offset_t stride) {
    _axis<reduce_t, offset_t> a;
    offset_t low = _low<O, reduce_t, offset_t>(coord);
    FF_PP_UNROLL
    for (int k = 0; k <= O; ++k) {
        offset_t nb = low + static_cast<offset_t>(k);
        int8_t   s  = bound::sign (b, nb, n);
        offset_t ix = bound::index(b, nb, n);
        a.w[k]   = static_cast<reduce_t>(s)
                 * _fastweight<O>(static_cast<reduce_t>(std::fabs(coord - static_cast<reduce_t>(nb))));
        a.off[k] = (s == 0 ? offset_t(0) : ix) * stride;
    }
    return a;
}

// ---- gather / scatter recursions over the static spatial rank --------------
template <int d, int D, int O, typename reduce_t, typename scalar_t, typename offset_t>
static inline CUDEV reduce_t
_pull_rec(const scalar_t * inp, const _axis<reduce_t, offset_t> * ax, offset_t off, reduce_t w) {
    reduce_t acc = 0;
    const _axis<reduce_t, offset_t> & a = ax[d];
    FF_PP_UNROLL
    for (int k = 0; k <= O; ++k) {
        offset_t o  = off + a.off[k];
        reduce_t ww = w * a.w[k];
        if constexpr (d + 1 == D) acc += static_cast<reduce_t>(inp[o]) * ww;
        else                      acc += _pull_rec<d + 1, D, O, reduce_t, scalar_t, offset_t>(inp, ax, o, ww);
    }
    return acc;
}

template <int d, int D, int O, typename reduce_t, typename scalar_t, typename offset_t>
static inline CUDEV void
_push_rec(scalar_t * out, const _axis<reduce_t, offset_t> * ax, offset_t off, reduce_t wv) {
    const _axis<reduce_t, offset_t> & a = ax[d];
    FF_PP_UNROLL
    for (int k = 0; k <= O; ++k) {
        reduce_t ww = a.w[k];
        if (ww == static_cast<reduce_t>(0)) continue;   // skip zero-boundary taps (no scatter traffic)
        offset_t o = off + a.off[k];
        if constexpr (d + 1 == D) anyAtomicAdd(out + o, static_cast<scalar_t>(wv * ww));
        else                      _push_rec<d + 1, D, O, reduce_t, scalar_t, offset_t>(out, ax, o, wv * ww);
    }
}

// ---- view helpers ----------------------------------------------------------
template <class V> using _elem_t =
    std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<V>().data())>>;

// Build the D spatial axes from a `(*spatial, C)` view + grid coordinate.
template <int D, int O, typename reduce_t, class VIn, typename offset_t>
static inline CUDEV void
_axes_from(_axis<reduce_t, offset_t> ax[D], const VIn & inp,
           const reduce_t loc[D], const bound_t bnd[D]) {
    FF_PP_UNROLL
    for (int d = 0; d < D; ++d)
        ax[d] = _make_axis<O>(bnd[d], loc[d],
                              static_cast<offset_t>(inp.extent(d)),
                              static_cast<offset_t>(inp.stride(d)));
}

// ===========================================================================
//                                  PULL
//   out (C,)  <-  gather from  inp (*spatial_in, C)  at `loc`
// ===========================================================================
template <int D, int O, typename reduce_t, typename offset_t, class VOut, class VIn>
static inline CUDEV void
pull(VOut out, const VIn inp, const reduce_t loc[D], const bound_t bnd[D], int extrapolate) {
    using scalar_t = _elem_t<VIn>;
    const offset_t nc  = static_cast<offset_t>(out.extent(0));
    const offset_t osc = static_cast<offset_t>(out.stride(0));
          scalar_t * op = out.data();

    // out-of-FOV -> write zeros
    bool inside = true;
    FF_PP_UNROLL
    for (int d = 0; d < D; ++d)
        inside = inside && _infov(extrapolate, loc[d], static_cast<offset_t>(inp.extent(d)));
    if (!inside) {
        for (offset_t c = 0; c < nc; ++c) op[c * osc] = static_cast<scalar_t>(0);
        return;
    }

    _axis<reduce_t, offset_t> ax[D];
    _axes_from<D, O, reduce_t>(ax, inp, loc, bnd);

    const scalar_t * ip  = inp.data();
    const offset_t   isc = static_cast<offset_t>(inp.stride(D));
    for (offset_t c = 0; c < nc; ++c)
        op[c * osc] = static_cast<scalar_t>(
            _pull_rec<0, D, O, reduce_t, scalar_t, offset_t>(ip + c * isc, ax, offset_t(0), static_cast<reduce_t>(1)));
}

// ===========================================================================
//                                  PUSH
//   out (*spatial_out, C)  <-  scatter-accumulate  inp (C,)  at `loc`
//   (out is pre-zeroed by the caller; adjoint of PULL)
// ===========================================================================
template <int D, int O, typename reduce_t, typename offset_t, class VOut, class VIn>
static inline CUDEV void
push(VOut out, const VIn inp, const reduce_t loc[D], const bound_t bnd[D], int extrapolate) {
    using scalar_t = _elem_t<VOut>;
    const offset_t nc  = static_cast<offset_t>(inp.extent(0));
    const offset_t isc = static_cast<offset_t>(inp.stride(0));
    const scalar_t * ip = inp.data();

    bool inside = true;
    FF_PP_UNROLL
    for (int d = 0; d < D; ++d)
        inside = inside && _infov(extrapolate, loc[d], static_cast<offset_t>(out.extent(d)));
    if (!inside) return;                                   // out-of-FOV -> nothing scattered

    _axis<reduce_t, offset_t> ax[D];
    _axes_from<D, O, reduce_t>(ax, out, loc, bnd);

    scalar_t * op = out.data();
    const offset_t osc = static_cast<offset_t>(out.stride(D));
    for (offset_t c = 0; c < nc; ++c)
        _push_rec<0, D, O, reduce_t, scalar_t, offset_t>(
            op + c * osc, ax, offset_t(0), static_cast<reduce_t>(ip[c * isc]));
}

// ===========================================================================
//                                  COUNT
//   out (*spatial_out, 1)  <-  scatter-accumulate the interpolation weights
//   (push of a constant 1; no channel loop; adjoint of pulling a ones field)
// ===========================================================================
template <int D, int O, typename reduce_t, typename offset_t, class VOut, class VGrid>
static inline CUDEV void
count(VOut out, const VGrid /*grid, unused: loc passed*/, const reduce_t loc[D],
      const bound_t bnd[D], int extrapolate) {
    using scalar_t = _elem_t<VOut>;
    bool inside = true;
    FF_PP_UNROLL
    for (int d = 0; d < D; ++d)
        inside = inside && _infov(extrapolate, loc[d], static_cast<offset_t>(out.extent(d)));
    if (!inside) return;

    _axis<reduce_t, offset_t> ax[D];
    _axes_from<D, O, reduce_t>(ax, out, loc, bnd);

    _push_rec<0, D, O, reduce_t, scalar_t, offset_t>(
        out.data(), ax, offset_t(0), static_cast<reduce_t>(1));
}

FF_NAMESPACE_END(pushpull)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_PUSHPULL_TEENY
