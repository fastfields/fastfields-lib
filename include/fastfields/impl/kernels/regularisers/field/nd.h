#ifndef FF_REGULARISERS_FIELD_ND
#define FF_REGULARISERS_FIELD_ND
// One N-D tap-table engine for the FIELD regulariser, replacing the
// hand-expanded `field/{1,2,3}d.h` stencil bodies (fastfields-kernels#50).
//
// ---------------------------------------------------------------------------
//  The one idea
// ---------------------------------------------------------------------------
// Every field energy is a small Toeplitz stencil written in DIFFERENCE FORM
//
//      out = w0 * centre  +  Sum_taps  w_tap * ( read(tap) - centre )
//
// with `read(tap)` a sign-carrying, boundary-folded neighbour read. Difference
// form is what makes the boundary come out right with no renormalisation: a tap
// that folds back onto the centre contributes `w*(sgn*centre - centre)`, a tap
// that reads nothing contributes `w*(0 - centre)`, and both are exactly the
// corresponding row of the symmetric operator.
//
// The taps are:
//
//    reach 1 (membrane)   +-1 along each axis d,          weight w1[d]
//    reach 2 (bending)  + +-2 along each axis d,          weight w2[d]
//                       + the 4 corners (+-1_d, +-1_e),   weight w11[d][e]
//
// and a corner tap is just the componentwise combination of two axis taps
// (`off_d + off_e`, `sgn_d * sgn_e`) because the boundary fold is SEPARABLE.
// That separability is the whole reason one engine can serve every D.
//
// `diag_*` is the same stencil contracted against the unit vector at the centre
// (`stap.h`'s `sdiag`), and `kernel_*` is the same stencil written out at pure
// strides with no folding at all. So all three operations, for all three
// energies, in all three weighting modes, come from ONE tap enumeration.
//
// ---------------------------------------------------------------------------
//  Weight table layout (per channel)
// ---------------------------------------------------------------------------
//      [ w0 | w1[0..D-1] | w2[0..D-1] | w11[pair 0..NPAIR-1] ]
//        1        D             D              D(D-1)/2
// absolute uses the first entry only, membrane the first 1+D, bending all of
// them. Pairs are in lexicographic (d<e) order -- for D=3: (0,1) (0,2) (1,2) --
// which reproduces the previous per-D layouts byte for byte.
//
// ---------------------------------------------------------------------------
//  Behaviour differences from the hand-expanded code (deliberate)
// ---------------------------------------------------------------------------
//  * `diag_*` now returns the EXACT matrix diagonal everywhere (#50 decision 1).
//    The old form subtracted `Sum_t w_t*sgn_t`, which is right in the interior
//    and wrong at every boundary voxel under every condition. See `sdiag`.
//  * The RLS/JRLS weight map is read through `smag`, which returns 0 where the
//    folded tap is not a real memory location, instead of dereferencing out of
//    bounds (kernels#39).
//  * The RLS kernel-table rescaling is applied for every D. The 2D-bending and
//    3D-membrane/bending variants previously called `get_kernelsize_*_rls()`
//    with no argument, which resolves to the STATIC channel count -- i.e. -1
//    whenever channels are dynamic, as they always are through the impl layer --
//    so the rescaling loop never ran and those kernels were left unscaled. Only
//    the 1D path was correct; generic code cannot reproduce a per-D slip.
//
// NOT fixed here (out of scope, tracked separately): kernels#40, the membrane
// term inside `make_kernel_bending_rls` being scaled by 1/4 like the bending
// term instead of by 1/2. It lives in the weight TABLE, not the stencil, and is
// flagged at its construction site below.
#include "../../cuda_switch.h"
#include "../../bounds.h"
#include "../../stap.h"
#include "../../utils.h"
#include "utils.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_field)

template <int _D, int _C, class... T>
struct Kernels<Config<_D, _C, T...>>
{
    using _Config  = Config<_D, _C, T...>;
    using scalar_t = typename _Config::scalar_t;
    using reduce_t = typename _Config::reduce_t;
    using offset_t = typename _Config::offset_t;

    static constexpr int      D = _D;
    static constexpr offset_t C = static_cast<offset_t>(_C);
    typedef scalar_t & (*OpType)(scalar_t &, const reduce_t &);

    static_assert(D >= 1, "the field regulariser needs at least one spatial axis");

    // ------------------------------------------------------------------
    //  weight-table geometry
    // ------------------------------------------------------------------

    static constexpr int NPAIR = D * (D - 1) / 2;      // # of (d<e) axis pairs
    static constexpr int NPAIRA = NPAIR > 0 ? NPAIR : 1;   // never a 0-size array

    static constexpr int KSIZE_ABSOLUTE = 1;
    static constexpr int KSIZE_MEMBRANE = 1 + D;
    static constexpr int KSIZE_BENDING  = 1 + 2*D + NPAIR;

    static inline CUDEV constexpr int kw1(int d) { return 1 + d; }
    static inline CUDEV constexpr int kw2(int d) { return 1 + D + d; }
    static inline CUDEV constexpr int kwc(int p) { return 1 + 2*D + p; }

    // index of the (d<e) pair in lexicographic order
    static inline CUDEV constexpr int pairidx(int d, int e)
    {
        int k = 0;
        for (int i = 0; i < d; ++i) k += (D - 1 - i);
        return k + (e - d - 1);
    }

    // ------------------------------------------------------------------
    //  per-axis boundary conditions
    // ------------------------------------------------------------------
    // Always through `bound::dyn<B>` (never `utils<B>`) so the static/dynamic
    // build policy of kernels#42 keeps working: for a real B this is an empty
    // object that folds away, for `Dynamic` it carries the condition at run
    // time and a single instantiation serves all eight.

    template <int d> struct axis_bound
    { static constexpr bound_t value = _Config::Bound::template At<d>::Value; };

    template <int R, int d = 0>
    static inline CUDEV FF_INLINE void
    fill_taps(stap<offset_t, R> * tp, const offset_t * loc,
              const offset_t * size, const offset_t * stride)
    {
        if constexpr (d < D)
        {
            tp[d] = make_stap<R>(bound::dyn<axis_bound<d>::value>(),
                                 loc[d], size[d], stride[d]);
            fill_taps<R, d + 1>(tp, loc, size, stride);
        }
    }

    // ------------------------------------------------------------------
    //  tap slots
    // ------------------------------------------------------------------
    // A reach-R axis has 2R non-centre taps. The loops below walk them by a
    // flat index j in [0, 2R); `tap_k` turns that into the tap's displacement
    // k (in -R..-1, +1..+R) and `stap::slot(k) == k + R` locates it in the
    // table. `tap_j` is the inverse, so a coefficient formula can ask for a
    // specific tap by name ("the +-2 tap of axis d").

    template <int R> static inline CUDEV constexpr int tap_k(int j)
    { return j < R ? j - R : j - R + 1; }
    template <int R> static inline CUDEV constexpr int tap_j(int k)
    { return k < 0 ? k + R : k + R - 1; }

    // ------------------------------------------------------------------
    //  the RLS/JRLS weight neighbourhood
    // ------------------------------------------------------------------
    // The reweighting map is sampled at exactly the stencil's own tap
    // positions, so one struct covers membrane (R=1) and bending (R=2). In
    // `wmode::none` the whole thing is replaced by an empty type and every
    // access below sits in a discarded `if constexpr` branch, so the plain
    // stencils carry no trace of it.

    struct wnone {};

    template <int R>
    struct wnbr
    {
        reduce_t c;                     // centre
        reduce_t ax[D][2*R];            // axis taps, tap_j order
        reduce_t cr[NPAIRA][4];         // corners, [pair][(s>0)*2 + (t>0)]
    };

    template <int R, wmode WM>
    using wtype = meta::If<WM == wmode::none, wnone, wnbr<R>>;

    template <int R>
    static inline CUDEV FF_INLINE reduce_t
    wax(const wnbr<R> & w, int d, int k) { return w.ax[d][tap_j<R>(k)]; }

    // corner value for "sign sd along axis d, sign se along axis e", for either
    // ordering of (d, e); the table is keyed on the (lo < hi) pair.
    template <int R>
    static inline CUDEV FF_INLINE reduce_t
    wcr(const wnbr<R> & w, int d, int sd, int e, int se)
    {
        const int lo  = d < e ? d  : e;
        const int hi  = d < e ? e  : d;
        const int slo = d < e ? sd : se;
        const int shi = d < e ? se : sd;
        return w.cr[pairidx(lo, hi)][(slo > 0 ? 2 : 0) + (shi > 0 ? 1 : 0)];
    }

    template <int R>
    static inline CUDEV FF_INLINE wnbr<R>
    load_wnbr(const scalar_t * wgt, const stap<offset_t, R> * tw)
    {
        wnbr<R> w;
        w.c = static_cast<reduce_t>(*wgt);
        for (int d = 0; d < D; ++d)
            for (int j = 0; j < 2*R; ++j) {
                const int sl = tap_k<R>(j) + R;
                w.ax[d][j] = smag<reduce_t>(wgt, tw[d].off[sl], tw[d].inb[sl]);
            }
        if constexpr (R >= 2)
            for (int d = 0; d < D; ++d)
            for (int e = d + 1; e < D; ++e)
            for (int si = 0; si < 2; ++si)
            for (int ti = 0; ti < 2; ++ti) {
                const int sd = (si ? 1 : -1) + R;
                const int se = (ti ? 1 : -1) + R;
                w.cr[pairidx(d, e)][si*2 + ti] = smag<reduce_t>(
                    wgt, tw[d].off[sd] + tw[e].off[se],
                    static_cast<int8_t>(tw[d].inb[sd] && tw[e].inb[se]));
            }
        return w;
    }

    // ------------------------------------------------------------------
    //  tap coefficients
    // ------------------------------------------------------------------
    // For WM == none the coefficient is the table entry itself and every `wnbr`
    // access below is discarded at compile time. For RLS/JRLS the coefficient is
    // the table entry times a local average of the reweighting map -- the same
    // formula the hand-expanded code inlined, derived once:
    //
    //   centre     (membrane)  2 * w0 * Wc            [w0 already carries 1/2]
    //   +-1 tap    (membrane)      w1[d] * (Wc + W(d,s))
    //   +-1 tap    (bending)   (w1[d] - 2 w2[d]) * (Wc + W(d,s))
    //                          - 2 w2[d] * (W(d,-s) + W2(d,s))
    //                          - sum_{e != d} w11[d][e]
    //                                * sum_{t=+-1} ( W(e,t) + Wcr(d,s,e,t) )
    //   +-2 tap    (bending)       w2[d] * (Wc + 2 W(d,s) + W2(d,s))
    //   corner     (bending)     w11[d][e] * (Wc + W(d,s) + W(e,t) + Wcr(..))
    //
    // Each bracket is an unnormalised local mean of the map over the cell the
    // tap spans; the 1/2 and 1/4 factors baked into the RLS kernel table by
    // `make_kernel_*_rls` are exactly their normalisations.

    template <int R, wmode WM, class W>
    static inline CUDEV FF_INLINE reduce_t
    coef_centre(const reduce_t * k, const W & w)
    {
        if constexpr (WM == wmode::none) return k[0];
        else if constexpr (R == 1)       return k[0] * w.c * static_cast<reduce_t>(2);
        else                             return k[0];   // bending: unweighted
    }

    // the +-1 tap of axis d (s = +-1)
    template <int R, wmode WM, class W>
    static inline CUDEV FF_INLINE reduce_t
    coef_ax1(const reduce_t * k, const W & w, int d, int s)
    {
        if constexpr (WM == wmode::none) return k[kw1(d)];
        else if constexpr (R == 1)       return k[kw1(d)] * (w.c + wax<R>(w, d, s));
        else {
            const reduce_t k1 = k[kw1(d)], k2 = k[kw2(d)];
            reduce_t acc = (k1 - 2*k2) * (w.c + wax<R>(w, d, s))
                         - 2*k2 * (wax<R>(w, d, -s) + wax<R>(w, d, 2*s));
            for (int e = 0; e < D; ++e) {
                if (e == d) continue;
                const reduce_t kc = k[kwc(pairidx(d < e ? d : e, d < e ? e : d))];
                for (int t = -1; t <= 1; t += 2)
                    acc -= kc * (wax<R>(w, e, t) + wcr<R>(w, d, s, e, t));
            }
            return acc;
        }
    }

    // the +-2 tap of axis d -- reach 2 only, so `kw2` is always in the table
    template <int R, wmode WM, class W>
    static inline CUDEV FF_INLINE reduce_t
    coef_ax2(const reduce_t * k, const W & w, int d, int s)
    {
        static_assert(R >= 2, "a reach-1 stencil has no second-order tap");
        if constexpr (WM == wmode::none) return k[kw2(d)];
        else return k[kw2(d)] * (w.c + 2*wax<R>(w, d, s) + wax<R>(w, d, 2*s));
    }

    template <int R, wmode WM, class W>
    static inline CUDEV FF_INLINE reduce_t
    coef_cr(const reduce_t * k, const W & w, int d, int s, int e, int t)
    {
        static_assert(R >= 2, "a reach-1 stencil has no corner tap");
        const reduce_t kc = k[kwc(pairidx(d, e))];
        if constexpr (WM == wmode::none) return kc;
        else return kc * (w.c + wax<R>(w, d, s) + wax<R>(w, e, t)
                              + wcr<R>(w, d, s, e, t));
    }

    // the order-m axis tap of axis d on side s, dispatching on m so that a
    // reach-1 stencil never even names the second-order table slot
    template <int R, wmode WM, class W>
    static inline CUDEV FF_INLINE reduce_t
    coef_axm(const reduce_t * k, const W & w, int d, int m, int s)
    {
        if constexpr (R == 1) return coef_ax1<R, WM>(k, w, d, s);
        else return m == 1 ? coef_ax1<R, WM>(k, w, d, s)
                           : coef_ax2<R, WM>(k, w, d, s);
    }

    // ------------------------------------------------------------------
    //  the two per-voxel contractions
    // ------------------------------------------------------------------
    // `stencil_matvec` walks the taps accumulating `w_tap * (read - centre)`;
    // `stencil_diag` walks the SAME taps accumulating `sdiag(w_tap, off, sgn)`.
    // Keeping them next to each other, over one tap enumeration, is what makes
    // a diag/matvec disagreement (the kernels#48/#49 bug class) structurally
    // impossible rather than merely unlikely.

    // The tap walk is ORDER-major (all reach-1 taps, then all reach-2, then the
    // corners), each axis' +- pair adjacent and its - side first, and a corner
    // block iterating the HIGHER axis' sign outermost. That is not cosmetic:
    // it is the exact summation order the hand-expanded bodies used, so the
    // plain path stays bit-identical rather than merely equal to a ULP. In the
    // unweighted case a +- pair also shares one coefficient, so the two deltas
    // are summed before the single multiply -- again as before, and half the
    // multiplies of a naive per-tap loop.

    template <int R, wmode WM, class W>
    static inline CUDEV FF_INLINE reduce_t
    stencil_matvec(const scalar_t * inp, const stap<offset_t, R> * ti,
                   const reduce_t * k, const W & w)
    {
        const reduce_t centre = static_cast<reduce_t>(*inp);
        reduce_t acc = coef_centre<R, WM>(k, w) * centre;

        for (int m = 1; m <= R; ++m)
            for (int d = 0; d < D; ++d)
            {
                const int lo = R - m, hi = R + m;
                const reduce_t dm = sdelta<reduce_t>(inp, ti[d].off[lo], ti[d].sgn[lo], centre);
                const reduce_t dp = sdelta<reduce_t>(inp, ti[d].off[hi], ti[d].sgn[hi], centre);
                if constexpr (WM == wmode::none)
                    acc += coef_axm<R, WM>(k, w, d, m, 1) * (dm + dp);
                else
                    acc += coef_axm<R, WM>(k, w, d, m, -1) * dm
                         + coef_axm<R, WM>(k, w, d, m,  1) * dp;
            }

        if constexpr (R >= 2)
            for (int d = 0; d < D; ++d)
            for (int e = d + 1; e < D; ++e)
            {
                reduce_t sum = static_cast<reduce_t>(0);
                for (int t = -1; t <= 1; t += 2)     // sign along the HIGHER axis
                for (int s = -1; s <= 1; s += 2)     // sign along the lower axis
                {
                    const int sd = s + R, se = t + R;
                    const reduce_t dl = sdelta<reduce_t>(
                        inp, ti[d].off[sd] + ti[e].off[se],
                        static_cast<int8_t>(ti[d].sgn[sd] * ti[e].sgn[se]), centre);
                    if constexpr (WM == wmode::none) sum += dl;
                    else acc += coef_cr<R, WM>(k, w, d, s, e, t) * dl;
                }
                if constexpr (WM == wmode::none) acc += k[kwc(pairidx(d, e))] * sum;
            }

        return acc;
    }

    template <int R, wmode WM, class W>
    static inline CUDEV FF_INLINE reduce_t
    stencil_diag(const stap<offset_t, R> * ti, const reduce_t * k, const W & w)
    {
        const offset_t z = static_cast<offset_t>(0);
        reduce_t acc = coef_centre<R, WM>(k, w);

        for (int m = 1; m <= R; ++m)
            for (int d = 0; d < D; ++d)
            {
                const int lo = R - m, hi = R + m;
                acc += sdiag<reduce_t>(coef_axm<R, WM>(k, w, d, m, -1),
                                       ti[d].off[lo] == z, ti[d].sgn[lo])
                     + sdiag<reduce_t>(coef_axm<R, WM>(k, w, d, m,  1),
                                       ti[d].off[hi] == z, ti[d].sgn[hi]);
            }

        if constexpr (R >= 2)
            for (int d = 0; d < D; ++d)
            for (int e = d + 1; e < D; ++e)
            for (int t = -1; t <= 1; t += 2)
            for (int s = -1; s <= 1; s += 2)
            {
                const int sd = s + R, se = t + R;
                // a corner folds onto the centre only when BOTH axes do
                acc += sdiag<reduce_t>(
                    coef_cr<R, WM>(k, w, d, s, e, t),
                    ti[d].off[sd] == z && ti[e].off[se] == z,
                    static_cast<int8_t>(ti[d].sgn[sd] * ti[e].sgn[se]));
            }
        return acc;
    }

    // materialise the stencil at pure strides (no folding, no boundary)
    template <int R, OpType op>
    static inline CUDEV FF_INLINE void
    stencil_write(scalar_t * out, const offset_t * stride, const reduce_t * k)
    {
        op(out[0], k[0]);
        for (int m = 1; m <= R; ++m)
            for (int d = 0; d < D; ++d)
            {
                reduce_t w;
                if constexpr (R == 1) w = k[kw1(d)];
                else w = (m == 1) ? k[kw1(d)] : k[kw2(d)];
                for (int s = -1; s <= 1; s += 2)
                    op(out[static_cast<offset_t>(s * m) * stride[d]], w);
            }
        if constexpr (R >= 2)
            for (int d = 0; d < D; ++d)
            for (int e = d + 1; e < D; ++e)
            for (int t = -1; t <= 1; t += 2)
            for (int s = -1; s <= 1; s += 2)
                op(out[static_cast<offset_t>(s) * stride[d] +
                       static_cast<offset_t>(t) * stride[e]], k[kwc(pairidx(d, e))]);
    }

    // ------------------------------------------------------------------
    //  the generic driver
    // ------------------------------------------------------------------
    // One channel loop for every (energy x weighting mode x matvec/diag). The
    // `joint` mode hoists the weight neighbourhood out of the loop -- it is the
    // same map for every channel -- which is the only thing that ever
    // distinguished a JRLS body from its RLS twin.

    static inline CUDEV FF_INLINE offset_t nchannels(offset_t nc)
    { return C < 0 ? nc : C; }

    template <int R, wmode WM, OpType op>
    static inline CUDEV FF_INLINE void
    run_matvec(scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
               const offset_t * loc, const offset_t * size,
               const offset_t * istride, const offset_t * wstride,
               offset_t osc, offset_t isc, offset_t wsc,
               const reduce_t * kernel, offset_t nc, int ksize)
    {
        stap<offset_t, R> ti[D];
        fill_taps<R>(ti, loc, size, istride);

        stap<offset_t, R> tw[D];
        wtype<R, WM> w{};
        if constexpr (WM != wmode::none) {
            fill_taps<R>(tw, loc, size, wstride);
            if constexpr (WM == wmode::joint) w = load_wnbr<R>(wgt, tw);
        }

        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c)
        {
            if constexpr (WM == wmode::split) w = load_wnbr<R>(wgt + wsc*c, tw);
            op(out[osc*c],
               stencil_matvec<R, WM>(inp + isc*c, ti, kernel + ksize*c, w));
        }
    }

    template <int R, wmode WM, OpType op>
    static inline CUDEV FF_INLINE void
    run_diag(scalar_t * out, const scalar_t * wgt,
             const offset_t * loc, const offset_t * size,
             const offset_t * wstride, offset_t osc, offset_t wsc,
             const reduce_t * kernel, offset_t nc, int ksize)
    {
        // the diagonal never reads the data, so the tap table only needs the
        // folded geometry -- a unit stride is enough to tell "folded onto the
        // centre" (off == 0) from "somewhere else".
        offset_t unit[D];
        for (int d = 0; d < D; ++d) unit[d] = static_cast<offset_t>(1);

        stap<offset_t, R> ti[D];
        fill_taps<R>(ti, loc, size, unit);

        stap<offset_t, R> tw[D];
        wtype<R, WM> w{};
        if constexpr (WM != wmode::none) {
            fill_taps<R>(tw, loc, size, wstride);
            if constexpr (WM == wmode::joint) w = load_wnbr<R>(wgt, tw);
        }

        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c)
        {
            if constexpr (WM == wmode::split) w = load_wnbr<R>(wgt + wsc*c, tw);
            op(out[osc*c], stencil_diag<R, WM>(ti, kernel + ksize*c, w));
        }
    }

    //==================================================================
    //                            ABSOLUTE
    //==================================================================
    // No spatial coupling: a per-channel scaling, optionally reweighted.

    static const offset_t kernelsize_absolute = C;

    CUDEV static inline offset_t
    get_kernelsize_absolute(offset_t nc = C)
    { return nchannels(nc); }

    /// kernel <- [abs, ...]
    CUDEV static inline void
    make_kernel_absolute(
              reduce_t kernel   [],
        const reduce_t absolute [],
              offset_t nc       = C
    )
    {
        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c) kernel[c] = absolute[c];
    }

    template <OpType op = set>
    CUDEV static inline void
    matvec_absolute(
              scalar_t out    [],
        const scalar_t inp    [],
              offset_t osc,
              offset_t isc,
        const reduce_t kernel [],
              offset_t nc     = C
    )
    {
        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c)
            op(out[osc*c], kernel[c] * static_cast<reduce_t>(inp[isc*c]));
    }

    template <OpType op = set>
    CUDEV static inline void
    kernel_absolute(
              scalar_t out    [],
              offset_t osc,
        const reduce_t kernel [],
              offset_t nc     = C
    )
    {
        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c) op(out[osc*c], kernel[c]);
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_absolute(
              scalar_t out    [],
              offset_t osc,
        const reduce_t kernel [],
              offset_t nc     = C
    )
    { return kernel_absolute<op>(out, osc, kernel, nc); }

    template <OpType op = set>
    CUDEV static inline void
    matvec_absolute_rls(
              scalar_t out    [],
        const scalar_t inp    [],
        const scalar_t wgt    [],
              offset_t osc,
              offset_t isc,
              offset_t wsc,
        const reduce_t kernel [],
              offset_t nc     = C
    )
    {
        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c)
            op(out[osc*c], kernel[c] * static_cast<reduce_t>(wgt[wsc*c])
                                     * static_cast<reduce_t>(inp[isc*c]));
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_absolute_rls(
              scalar_t out    [],
        const scalar_t wgt    [],
              offset_t osc,
              offset_t wsc,
        const reduce_t kernel [],
              offset_t nc     = C
    )
    {
        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c)
            op(out[osc*c], kernel[c] * static_cast<reduce_t>(wgt[wsc*c]));
    }

    template <OpType op = set>
    CUDEV static inline void
    matvec_absolute_jrls(
              scalar_t out    [],
        const scalar_t inp    [],
        const scalar_t wgt    [],
              offset_t osc,
              offset_t isc,
        const reduce_t kernel [],
              offset_t nc     = C
    )
    {
        const reduce_t w = static_cast<reduce_t>(*wgt);
        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c)
            op(out[osc*c], kernel[c] * w * static_cast<reduce_t>(inp[isc*c]));
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_absolute_jrls(
              scalar_t out    [],
        const scalar_t wgt    [],
              offset_t osc,
        const reduce_t kernel [],
              offset_t nc     = C
    )
    {
        const reduce_t w = static_cast<reduce_t>(*wgt);
        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c) op(out[osc*c], kernel[c] * w);
    }

    //==================================================================
    //                            MEMBRANE
    //==================================================================

    static const offset_t kernelsize_membrane = KSIZE_MEMBRANE * C;

    CUDEV static inline offset_t
    get_kernelsize_membrane(offset_t nc = C)
    { return KSIZE_MEMBRANE * nchannels(nc); }

    /// kernel <- [abs, w1[d]..., ...]
    CUDEV static inline void
    make_kernel_membrane(
              reduce_t kernel     [],
        const reduce_t absolute   [],
        const reduce_t membrane   [],
        const reduce_t voxel_size [D],
              offset_t nc         = C
    )
    {
        reduce_t v[D];
        for (int d = 0; d < D; ++d) v[d] = 1. / (voxel_size[d] * voxel_size[d]);

        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c, kernel += KSIZE_MEMBRANE)
        {
            const reduce_t m = membrane[c];
            kernel[0] = absolute[c];
            for (int d = 0; d < D; ++d) kernel[kw1(d)] = -m * v[d];
        }
    }

    /// kernel <- [w0 (all taps folded in), w1[d]..., ...]
    CUDEV static inline void
    make_fullkernel_membrane(
              reduce_t kernel     [],
        const reduce_t absolute   [],
        const reduce_t membrane   [],
        const reduce_t voxel_size [D],
              offset_t nc         = C
    )
    {
        make_kernel_membrane(kernel, absolute, membrane, voxel_size, nc);
        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c, kernel += KSIZE_MEMBRANE)
            for (int d = 0; d < D; ++d) kernel[0] -= 2 * kernel[kw1(d)];
    }

    template <OpType op = set>
    CUDEV static inline void
    matvec_membrane(
              scalar_t out    [],
        const scalar_t inp    [],
        const offset_t loc    [D],
        const offset_t size   [D],
        const offset_t stride [D],
              offset_t osc,
              offset_t isc,
        const reduce_t kernel [],
              offset_t nc     = C
    )
    {
        run_matvec<1, wmode::none, op>(out, inp, nullptr, loc, size, stride,
                                       nullptr, osc, isc, 0, kernel, nc,
                                       KSIZE_MEMBRANE);
    }

    template <OpType op = set>
    CUDEV static inline void
    kernel_membrane(
              scalar_t out    [],
              offset_t sc,
        const offset_t stride [D],
        const reduce_t kernel [],
              offset_t nc     = C
    )
    {
        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c)
            stencil_write<1, op>(out + sc*c, stride, kernel + KSIZE_MEMBRANE*c);
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_membrane(
              scalar_t out    [],
              offset_t osc,
        const offset_t loc    [D],
        const offset_t size   [D],
        const reduce_t kernel [],
              offset_t nc     = C
    )
    {
        run_diag<1, wmode::none, op>(out, nullptr, loc, size, nullptr, osc, 0,
                                     kernel, nc, KSIZE_MEMBRANE);
    }

    // --- membrane RLS / JRLS ---

    static const offset_t kernelsize_membrane_rls = kernelsize_membrane;

    CUDEV static inline offset_t
    get_kernelsize_membrane_rls(offset_t nc = C)
    { return get_kernelsize_membrane(nc); }

    CUDEV static inline void
    make_kernel_membrane_rls(
              reduce_t kernel     [],
        const reduce_t absolute   [],
        const reduce_t membrane   [],
        const reduce_t voxel_size [D],
              offset_t nc         = C
    )
    {
        make_kernel_membrane(kernel, absolute, membrane, voxel_size, nc);
        // 1/2 normalises the two-point mean of the weight map that every
        // membrane coefficient carries (see `coef_ax1`).
        const offset_t k = get_kernelsize_membrane_rls(nc);
        for (offset_t i = 0; i < k; ++i) kernel[i] *= 0.5;
    }

    template <OpType op = set>
    CUDEV static inline void
    matvec_membrane_rls(
              scalar_t out     [],
        const scalar_t inp     [],
        const scalar_t wgt     [],
        const offset_t loc     [D],
        const offset_t size    [D],
        const offset_t istride [D],
        const offset_t wstride [D],
              offset_t osc,
              offset_t isc,
              offset_t wsc,
        const reduce_t kernel  [],
              offset_t nc      = C
    )
    {
        run_matvec<1, wmode::split, op>(out, inp, wgt, loc, size, istride,
                                        wstride, osc, isc, wsc, kernel, nc,
                                        KSIZE_MEMBRANE);
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_membrane_rls(
              scalar_t out     [],
        const scalar_t wgt     [],
        const offset_t loc     [D],
        const offset_t size    [D],
        const offset_t wstride [D],
              offset_t osc,
              offset_t wsc,
        const reduce_t kernel  [],
              offset_t nc      = C
    )
    {
        run_diag<1, wmode::split, op>(out, wgt, loc, size, wstride, osc, wsc,
                                      kernel, nc, KSIZE_MEMBRANE);
    }

    template <OpType op = set>
    CUDEV static inline void
    matvec_membrane_jrls(
              scalar_t out     [],
        const scalar_t inp     [],
        const scalar_t wgt     [],
        const offset_t loc     [D],
        const offset_t size    [D],
        const offset_t istride [D],
        const offset_t wstride [D],
              offset_t osc,
              offset_t isc,
        const reduce_t kernel  [],
              offset_t nc      = C
    )
    {
        run_matvec<1, wmode::joint, op>(out, inp, wgt, loc, size, istride,
                                        wstride, osc, isc, 0, kernel, nc,
                                        KSIZE_MEMBRANE);
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_membrane_jrls(
              scalar_t out     [],
        const scalar_t wgt     [],
        const offset_t loc     [D],
        const offset_t size    [D],
        const offset_t wstride [D],
              offset_t osc,
        const reduce_t kernel  [],
              offset_t nc      = C
    )
    {
        run_diag<1, wmode::joint, op>(out, wgt, loc, size, wstride, osc, 0,
                                      kernel, nc, KSIZE_MEMBRANE);
    }

    //==================================================================
    //                             BENDING
    //==================================================================

    static const offset_t kernelsize_bending = KSIZE_BENDING * C;

    CUDEV static inline offset_t
    get_kernelsize_bending(offset_t nc = C)
    { return KSIZE_BENDING * nchannels(nc); }

    /// kernel <- [abs, w1[d]..., w2[d]..., w11[pair]..., ...]
    CUDEV static inline void
    make_kernel_bending(
              reduce_t kernel     [],
        const reduce_t absolute   [],
        const reduce_t membrane   [],
        const reduce_t bending    [],
        const reduce_t voxel_size [D],
              offset_t nc         = C
    )
    {
        reduce_t v[D], vsum = 0;
        for (int d = 0; d < D; ++d) {
            v[d] = 1. / (voxel_size[d] * voxel_size[d]);
            vsum += v[d];
        }

        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c, kernel += KSIZE_BENDING)
        {
            const reduce_t m = membrane[c], b = bending[c];
            kernel[0] = absolute[c];
            for (int d = 0; d < D; ++d) {
                kernel[kw1(d)] = -4 * b * v[d] * vsum - m * v[d];
                kernel[kw2(d)] = b * v[d] * v[d];
            }
            for (int d = 0; d < D; ++d)
                for (int e = d + 1; e < D; ++e)
                    kernel[kwc(pairidx(d, e))] = 2 * b * v[d] * v[e];
        }
    }

    /// kernel <- [w0 (all taps folded in), w1..., w2..., w11..., ...]
    CUDEV static inline void
    make_fullkernel_bending(
              reduce_t kernel     [],
        const reduce_t absolute   [],
        const reduce_t membrane   [],
        const reduce_t bending    [],
        const reduce_t voxel_size [D],
              offset_t nc         = C
    )
    {
        make_kernel_bending(kernel, absolute, membrane, bending, voxel_size, nc);
        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c, kernel += KSIZE_BENDING)
        {
            // every tap appears with multiplicity: 2 per axis tap, 4 per corner
            for (int d = 0; d < D; ++d)
                kernel[0] -= 2 * (kernel[kw1(d)] + kernel[kw2(d)]);
            for (int p = 0; p < NPAIR; ++p) kernel[0] -= 4 * kernel[kwc(p)];
        }
    }

    template <OpType op = set>
    CUDEV static inline void
    matvec_bending(
              scalar_t out    [],
        const scalar_t inp    [],
        const offset_t loc    [D],
        const offset_t size   [D],
        const offset_t stride [D],
              offset_t osc,
              offset_t isc,
        const reduce_t kernel [],
              offset_t nc     = C
    )
    {
        run_matvec<2, wmode::none, op>(out, inp, nullptr, loc, size, stride,
                                       nullptr, osc, isc, 0, kernel, nc,
                                       KSIZE_BENDING);
    }

    template <OpType op = set>
    CUDEV static inline void
    kernel_bending(
              scalar_t out    [],
              offset_t sc,
        const offset_t stride [D],
        const reduce_t kernel [],
              offset_t nc     = C
    )
    {
        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c)
            stencil_write<2, op>(out + sc*c, stride, kernel + KSIZE_BENDING*c);
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_bending(
              scalar_t out    [],
              offset_t osc,
        const offset_t loc    [D],
        const offset_t size   [D],
        const reduce_t kernel [],
              offset_t nc     = C
    )
    {
        run_diag<2, wmode::none, op>(out, nullptr, loc, size, nullptr, osc, 0,
                                     kernel, nc, KSIZE_BENDING);
    }

    // --- bending RLS / JRLS ---

    static const offset_t kernelsize_bending_rls = kernelsize_bending;

    CUDEV static inline offset_t
    get_kernelsize_bending_rls(offset_t nc = C)
    { return get_kernelsize_bending(nc); }

    CUDEV static inline void
    make_kernel_bending_rls(
              reduce_t kernel     [],
        const reduce_t absolute   [],
        const reduce_t membrane   [],
        const reduce_t bending    [],
        const reduce_t voxel_size [D],
              offset_t nc         = C
    )
    {
        make_kernel_bending(kernel, absolute, membrane, bending, voxel_size, nc);
        // 1/4 normalises the four-point mean of the weight map every bending
        // coefficient carries; `w0` (absolute) is left alone because the bending
        // centre coefficient is unweighted.
        //
        // NB kernels#40 lives here: `w1[d]` mixes a bending part (which wants
        // 1/4) and a membrane part `-m*v[d]` (which wants the membrane's 1/2),
        // and scaling the whole entry runs the membrane term at half strength.
        // Fixing it needs the two contributions kept apart in the table, which
        // is a weight-table change, not a stencil change -- deliberately NOT
        // done here so this rewrite stays behaviour-preserving away from #50's
        // two decisions.
        const offset_t n = nchannels(nc);
        for (offset_t c = 0; c < n; ++c, kernel += KSIZE_BENDING)
            for (int i = 1; i < KSIZE_BENDING; ++i) kernel[i] *= 0.25;
    }

    template <OpType op = set>
    CUDEV static inline void
    matvec_bending_rls(
              scalar_t out     [],
        const scalar_t inp     [],
        const scalar_t wgt     [],
        const offset_t loc     [D],
        const offset_t size    [D],
        const offset_t istride [D],
        const offset_t wstride [D],
              offset_t osc,
              offset_t isc,
              offset_t wsc,
        const reduce_t kernel  [],
              offset_t nc      = C
    )
    {
        run_matvec<2, wmode::split, op>(out, inp, wgt, loc, size, istride,
                                        wstride, osc, isc, wsc, kernel, nc,
                                        KSIZE_BENDING);
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_bending_rls(
              scalar_t out     [],
        const scalar_t wgt     [],
        const offset_t loc     [D],
        const offset_t size    [D],
        const offset_t wstride [D],
              offset_t osc,
              offset_t wsc,
        const reduce_t kernel  [],
              offset_t nc      = C
    )
    {
        run_diag<2, wmode::split, op>(out, wgt, loc, size, wstride, osc, wsc,
                                      kernel, nc, KSIZE_BENDING);
    }

    template <OpType op = set>
    CUDEV static inline void
    matvec_bending_jrls(
              scalar_t out     [],
        const scalar_t inp     [],
        const scalar_t wgt     [],
        const offset_t loc     [D],
        const offset_t size    [D],
        const offset_t istride [D],
        const offset_t wstride [D],
              offset_t osc,
              offset_t isc,
        const reduce_t kernel  [],
              offset_t nc      = C
    )
    {
        run_matvec<2, wmode::joint, op>(out, inp, wgt, loc, size, istride,
                                        wstride, osc, isc, 0, kernel, nc,
                                        KSIZE_BENDING);
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_bending_jrls(
              scalar_t out     [],
        const scalar_t wgt     [],
        const offset_t loc     [D],
        const offset_t size    [D],
        const offset_t wstride [D],
              offset_t osc,
        const reduce_t kernel  [],
              offset_t nc      = C
    )
    {
        run_diag<2, wmode::joint, op>(out, wgt, loc, size, wstride, osc, 0,
                                      kernel, nc, KSIZE_BENDING);
    }
};

FF_NAMESPACE_END(reg_field)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_REGULARISERS_FIELD_ND
