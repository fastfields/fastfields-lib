#ifndef FF_REGULARISERS_STENCIL
#define FF_REGULARISERS_STENCIL
// The per-component N-D stencil engine shared by the regulariser kernels.
//
// This is the layer between `stap.h` (per-axis boundary-folded tap tables --
// energy-agnostic, knows nothing about weights) and the per-module kernel
// classes (`regularisers/field/nd.h`, `regularisers/flow/nd.h` -- which own the
// entry points, the weight tables and the channel loops).
//
// ---------------------------------------------------------------------------
//  The one idea
// ---------------------------------------------------------------------------
// Every separable regulariser energy is a small Toeplitz stencil written in
// DIFFERENCE FORM
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
// `stencil_diag` is the same stencil contracted against the unit vector at the
// centre (`stap.h`'s `sdiag`), and `stencil_write` is the same stencil written
// out at pure strides with no folding at all. So all three operations, for
// every energy, in every weighting mode, come from ONE tap enumeration -- which
// is what makes a diag/matvec disagreement (the kernels#48/#49 bug class)
// structurally impossible rather than merely unlikely.
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
// Extracted verbatim from `regularisers/field/nd.h` (fastfields-kernels#56) so
// the flow engine (#59) can build on the same code rather than a second copy of
// it. Behaviour-preserving: the tap walk, the summation order and the shared
// multiplies are unchanged, and `field/nd.h` still produces bit-identical
// results.
#include "../cuda_switch.h"
#include "../bounds.h"
#include "../stap.h"
#include "../meta.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg)

/// The per-component stencil engine.
///
///   D          number of spatial axes
///   BoundTuple `Bound<B0, B1, ...>` (a `meta::Tuple<bound_t, ...>`) with one
///              entry per axis
template <int D, class scalar_t, class reduce_t, class offset_t, class BoundTuple>
struct stencil
{
    typedef scalar_t & (*OpType)(scalar_t &, const reduce_t &);

    static_assert(D >= 1, "a regulariser stencil needs at least one spatial axis");

    // ------------------------------------------------------------------
    //  weight-table geometry
    // ------------------------------------------------------------------

    static constexpr int NPAIR  = D * (D - 1) / 2;      // # of (d<e) axis pairs
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
    { static constexpr bound_t value = BoundTuple::template At<d>::Value; };

    // The ADJOINT condition of the same axis. The transpose of a first
    // difference flips the parity of the extension (DCT<->DST, DFT self-
    // adjoint) -- see `bound::transpose`. Used by flow's Lame cross-coupling
    // block, which must apply D_d^T D_e rather than D_d D_e.
    template <int d> struct axis_bound_adjoint
    { static constexpr bound_t value = bound::transpose(axis_bound<d>::value); };

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

    template <int R, int d = 0>
    static inline CUDEV FF_INLINE void
    fill_taps_adjoint(stap<offset_t, R> * tp, const offset_t * loc,
                      const offset_t * size, const offset_t * stride)
    {
        if constexpr (d < D)
        {
            tp[d] = make_stap<R>(bound::dyn<axis_bound_adjoint<d>::value>(),
                                 loc[d], size[d], stride[d]);
            fill_taps_adjoint<R, d + 1>(tp, loc, size, stride);
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
    //  the two generic drivers
    // ------------------------------------------------------------------
    // The `joint` mode hoists the weight neighbourhood out of the channel loop
    // -- it is the same map for every channel -- which is the only thing that
    // ever distinguished a JRLS body from its RLS twin.
    //
    // `Post` post-multiplies each channel's result (1 by default). Flow's
    // per-component weight tables are the shared table divided by the channel's
    // own voxel size, and one of its entry points divides the SUM rather than
    // the table; passing the reciprocal here reproduces that association.

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

        for (offset_t c = 0; c < nc; ++c)
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

        for (offset_t c = 0; c < nc; ++c)
        {
            if constexpr (WM == wmode::split) w = load_wnbr<R>(wgt + wsc*c, tw);
            op(out[osc*c], stencil_diag<R, WM>(ti, kernel + ksize*c, w));
        }
    }
};

FF_NAMESPACE_END(reg)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_REGULARISERS_STENCIL
