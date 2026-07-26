// Teeny-based pushpull kernels: one separable recursion over the static spatial
// rank D replaces the per-rank 1d/2d/3d hand-unrolled gather/scatter trees.
//
// Design (validated by a clang++/g++ -O3 codegen probe, fastfields-lib#21):
//   * spatial rank D, interpolation order O, AND boundary condition B are all
//     COMPILE-TIME (the K=O+1 tap loops fully unroll — no runtime tap loop even
//     at cubic; the boundary index/sign fold to the specific bound's arithmetic,
//     and the constant-sign bounds drop the sign handling entirely). The full
//     dim x order x bound matrix is compiled for the library; TEST builds trim it
//     with the sparse orthogonal-coverage matrix (-DFF_TEST_SPARSE), exactly like
//     the pre-teeny cpu-lib — compile time is a build concern, not a perf one.
//   * HYBRID escape: passing B == bound_t::Dynamic takes a RUNTIME boundary route
//     (the trailing `rt` arg drives a per-axis switch instead of folding). This
//     lets the lib compile the common bounds statically and route the rarely-used
//     ones through the slower runtime path, deciding the split later without
//     touching the kernel. `rt` is ignored for any concrete B (the compiler drops
//     it), so static call sites need not pass it.
//   * The boundary SIGN is folded into the weight at setup (w[k] *= sign), so the
//     recursion is a branchless multiply-accumulate. A zero-boundary out-of-range
//     tap gets weight 0 and its offset clamped in-range (a harmless *0 gather);
//     PUSH additionally skips the zero-weight tap to avoid pointless scatter.
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

template <int O, typename scalar_t>
static inline CUDEV scalar_t _fastgrad(scalar_t x) {
    if      constexpr (O==0) return spline::_spline::fastgrad0<scalar_t>(x);
    else if constexpr (O==1) return spline::_spline::fastgrad1<scalar_t>(x);
    else if constexpr (O==2) return spline::_spline::fastgrad2<scalar_t>(x);
    else if constexpr (O==3) return spline::_spline::fastgrad3<scalar_t>(x);
    else if constexpr (O==4) return spline::_spline::fastgrad4<scalar_t>(x);
    else if constexpr (O==5) return spline::_spline::fastgrad5<scalar_t>(x);
    else if constexpr (O==6) return spline::_spline::fastgrad6<scalar_t>(x);
    else                     return spline::_spline::fastgrad7<scalar_t>(x);
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
// an in-range multiply-by-zero. Boundary B is COMPILE-TIME (bound::utils<B>),
// so index/sign fold and the constant-sign bounds cost nothing.
template <typename reduce_t, typename offset_t>
struct _axis { reduce_t w[FF_PP_MAXK]; offset_t off[FF_PP_MAXK]; };

// Boundary index/sign for tap node `nb`. B == bound_t::Dynamic takes the RUNTIME
// route (`rt`, a per-axis switch) — the opt-in slower path for rarely-used bounds
// the lib chooses not to compile statically; any concrete B folds.
template <bound_t B, typename offset_t>
static inline CUDEV void _bound_at(bound_t rt, offset_t nb, offset_t n, int8_t & s, offset_t & ix) {
    if constexpr (B == bound_t::Dynamic) {
        s  = bound::sign (rt, nb, n);
        ix = bound::index(rt, nb, n);
    } else {
        s  = bound::utils<B>::template sign <offset_t>(nb, n);
        ix = bound::utils<B>::template index<offset_t>(nb, n);
    }
}

template <int O, bound_t B, typename reduce_t, typename offset_t>
static inline CUDEV _axis<reduce_t, offset_t>
_make_axis(reduce_t coord, offset_t n, offset_t stride, bound_t rt) {
    _axis<reduce_t, offset_t> a;
    offset_t low = _low<O, reduce_t, offset_t>(coord);
    FF_PP_UNROLL
    for (int k = 0; k <= O; ++k) {
        offset_t nb = low + static_cast<offset_t>(k);
        int8_t s; offset_t ix;
        _bound_at<B>(rt, nb, n, s, ix);
        a.w[k]   = static_cast<reduce_t>(s)
                 * _fastweight<O>(static_cast<reduce_t>(std::fabs(coord - static_cast<reduce_t>(nb))));
        a.off[k] = (s == 0 ? offset_t(0) : ix) * stride;
    }
    return a;
}

// axis neighbourhood carrying BOTH the weight and the oriented grad-weight, each
// sign-folded. g[k] = s * maybe_fabs(fastgrad(|d|) * orient), matching the
// existing gindex convention (utils.h): d=coord-node, orient = sign(d).
template <typename reduce_t, typename offset_t>
struct _axisg { reduce_t w[FF_PP_MAXK]; reduce_t g[FF_PP_MAXK]; offset_t off[FF_PP_MAXK]; };

template <int O, bound_t B, bool ABS, typename reduce_t, typename offset_t>
static inline CUDEV _axisg<reduce_t, offset_t>
_make_axis_g(reduce_t coord, offset_t n, offset_t stride, bound_t rt) {
    _axisg<reduce_t, offset_t> a;
    offset_t low = _low<O, reduce_t, offset_t>(coord);
    FF_PP_UNROLL
    for (int k = 0; k <= O; ++k) {
        offset_t nb   = low + static_cast<offset_t>(k);
        reduce_t d    = coord - static_cast<reduce_t>(nb);
        bool     neg  = d < 0;
        reduce_t ad   = neg ? -d : d;
        int8_t s; offset_t ix;
        _bound_at<B>(rt, nb, n, s, ix);
        reduce_t gg   = _fastgrad<O>(ad) * (neg ? static_cast<reduce_t>(-1) : static_cast<reduce_t>(1));
        if (ABS && gg < 0) gg = -gg;                       // maybe::fabs (abs basis derivative)
        a.w[k]   = static_cast<reduce_t>(s) * _fastweight<O>(ad);
        a.g[k]   = static_cast<reduce_t>(s) * gg;
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

template <int d, int D, int O, int GD, typename reduce_t, typename scalar_t, typename offset_t>
static inline CUDEV reduce_t
_grad_rec(const scalar_t * inp, const _axisg<reduce_t, offset_t> * ax, offset_t off, reduce_t w) {
    reduce_t acc = 0;
    const _axisg<reduce_t, offset_t> & a = ax[d];
    FF_PP_UNROLL
    for (int k = 0; k <= O; ++k) {
        offset_t o  = off + a.off[k];
        reduce_t ww = w * (d == GD ? a.g[k] : a.w[k]);     // d==GD folds (both compile-time)
        if constexpr (d + 1 == D) acc += static_cast<reduce_t>(inp[o]) * ww;
        else                      acc += _grad_rec<d + 1, D, O, GD, reduce_t, scalar_t, offset_t>(inp, ax, o, ww);
    }
    return acc;
}

// ---- view helpers ----------------------------------------------------------
template <class V> using _elem_t =
    std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<V>().data())>>;

// Build the D spatial axes from a `(*spatial, C)` view + grid coordinate.
template <int D, int O, bound_t B, typename reduce_t, typename offset_t, class VIn>
static inline CUDEV void
_axes_from(_axis<reduce_t, offset_t> ax[D], const VIn & inp, const reduce_t loc[D], bound_t rt) {
    FF_PP_UNROLL
    for (int d = 0; d < D; ++d)
        ax[d] = _make_axis<O, B, reduce_t, offset_t>(loc[d],
                                                     static_cast<offset_t>(inp.extent(d)),
                                                     static_cast<offset_t>(inp.stride(d)), rt);
}

// ===========================================================================
//                                  PULL
//   out (C,)  <-  gather from  inp (*spatial_in, C)  at `loc`
// ===========================================================================
template <int D, int O, bound_t B, typename reduce_t, typename offset_t, class VOut, class VIn>
static inline CUDEV void
pull(VOut out, const VIn inp, const reduce_t loc[D], int extrapolate, bound_t rt = bound_t::Dynamic) {
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
    _axes_from<D, O, B, reduce_t, offset_t>(ax, inp, loc, rt);

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
template <int D, int O, bound_t B, typename reduce_t, typename offset_t, class VOut, class VIn>
static inline CUDEV void
push(VOut out, const VIn inp, const reduce_t loc[D], int extrapolate, bound_t rt = bound_t::Dynamic) {
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
    _axes_from<D, O, B, reduce_t, offset_t>(ax, out, loc, rt);

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
template <int D, int O, bound_t B, typename reduce_t, typename offset_t, class VOut>
static inline CUDEV void
count(VOut out, const reduce_t loc[D], int extrapolate, bound_t rt = bound_t::Dynamic) {
    using scalar_t = _elem_t<VOut>;
    bool inside = true;
    FF_PP_UNROLL
    for (int d = 0; d < D; ++d)
        inside = inside && _infov(extrapolate, loc[d], static_cast<offset_t>(out.extent(d)));
    if (!inside) return;

    _axis<reduce_t, offset_t> ax[D];
    _axes_from<D, O, B, reduce_t, offset_t>(ax, out, loc, rt);

    _push_rec<0, D, O, reduce_t, scalar_t, offset_t>(
        out.data(), ax, offset_t(0), static_cast<reduce_t>(1));
}

// ===========================================================================
//                                  GRAD
//   out (C, D)  <-  spatial gradient of the pulled value w.r.t. the D coords
//   (component dd uses the grad-weight on axis dd, the weight elsewhere)
// ===========================================================================
template <int D, int O, bound_t B, bool ABS, typename reduce_t, typename offset_t, class VOut, class VIn>
static inline CUDEV void
grad(VOut out, const VIn inp, const reduce_t loc[D], int extrapolate, bound_t rt = bound_t::Dynamic) {
    using scalar_t = _elem_t<VIn>;
    const offset_t nc  = static_cast<offset_t>(out.extent(0));   // out is (C, D)
    const offset_t osc = static_cast<offset_t>(out.stride(0));
    const offset_t osg = static_cast<offset_t>(out.stride(1));
          scalar_t * op = out.data();

    bool inside = true;
    FF_PP_UNROLL
    for (int d = 0; d < D; ++d)
        inside = inside && _infov(extrapolate, loc[d], static_cast<offset_t>(inp.extent(d)));
    if (!inside) {
        for (offset_t c = 0; c < nc; ++c)
            for (int dd = 0; dd < D; ++dd) op[c * osc + dd * osg] = static_cast<scalar_t>(0);
        return;
    }

    _axisg<reduce_t, offset_t> ax[D];
    FF_PP_UNROLL
    for (int d = 0; d < D; ++d)
        ax[d] = _make_axis_g<O, B, ABS, reduce_t, offset_t>(loc[d],
                                                            static_cast<offset_t>(inp.extent(d)),
                                                            static_cast<offset_t>(inp.stride(d)), rt);

    const scalar_t * ip  = inp.data();
    const offset_t   isc = static_cast<offset_t>(inp.stride(D));
    for (offset_t c = 0; c < nc; ++c) {
        const scalar_t * ipc = ip + c * isc;
        scalar_t * opc = op + c * osc;
        opc[0 * osg] = static_cast<scalar_t>(_grad_rec<0, D, O, 0, reduce_t, scalar_t, offset_t>(ipc, ax, offset_t(0), static_cast<reduce_t>(1)));
        if constexpr (D > 1) opc[1 * osg] = static_cast<scalar_t>(_grad_rec<0, D, O, 1, reduce_t, scalar_t, offset_t>(ipc, ax, offset_t(0), static_cast<reduce_t>(1)));
        if constexpr (D > 2) opc[2 * osg] = static_cast<scalar_t>(_grad_rec<0, D, O, 2, reduce_t, scalar_t, offset_t>(ipc, ax, offset_t(0), static_cast<reduce_t>(1)));
    }
}

FF_NAMESPACE_END(pushpull)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_PUSHPULL_TEENY
