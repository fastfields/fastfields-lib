#ifndef FF_REGULARISERS_FLOW_ND
#define FF_REGULARISERS_FLOW_ND
// One N-D tap-table engine for the FLOW regulariser, replacing the
// hand-expanded `flow/{1,2,3}d.h` stencil bodies (fastfields-kernels#50 phase 2,
// tracked as #59). The per-component stencil, the tap tables and the RLS/JRLS
// neighbourhood come from `regularisers/stencil.h`, shared with the field
// engine; this header adds only what is flow-specific.
//
// ---------------------------------------------------------------------------
//  What "flow" adds over "field"
// ---------------------------------------------------------------------------
// A flow is a VECTOR field: one channel per spatial axis (C == D), where the
// field regulariser has an arbitrary channel count. Two consequences.
//
// 1. **Per-component weight tables.** Component `c` is penalised in units of
//    its own voxel size, so its table is the shared table divided by
//    `v[c] = 1 / voxel_size[c]^2`. Everything else about the per-component
//    stencil -- absolute, membrane, bending, and their JRLS variants -- is the
//    field stencil verbatim, so it is the shared engine with `nc = D`.
//
// 2. **The Lame (linear-elastic) cross-coupling block**, which has no field
//    analogue. The elastic energy couples DIFFERENT components:
//
//        out_c  +=  w2 * sum_{e != c}  sum_{s,t = +-1}  -s*t * v_e(x + s*1_c + t*1_e)
//
//    i.e. a 4-corner gather per axis pair, reading the OTHER component. The
//    off-diagonal block is a product of two first differences, D_c^T D_e, so
//    for the whole operator to be self-adjoint the axis-`c` half of each corner
//    must be folded by the ADJOINT boundary condition -- `bound::transpose`,
//    which swaps DCT1<->DST1 and DCT2<->DST2 and leaves the rest alone
//    (fastfields-lib#26). In the interior the two coincide, so the interior
//    stencil is unchanged. There is no cross block at D == 1 (no axis pair).
//
//    Because the fold is separable per axis, that 4-corner gather is just two
//    reach-1 tap tables -- one adjoint, one natural -- combined componentwise,
//    exactly as a bending corner is. It costs about fifteen lines against the
//    tap table where the hand-expanded form cost ~300 per D.
//
// ---------------------------------------------------------------------------
//  Weight table layout
// ---------------------------------------------------------------------------
//      [ component 0 | component 1 | ... | component D-1 | ww ]
// with each component block laid out as `regularisers/stencil.h` describes
//      [ w0 | w1[0..D-1] | w2[0..D-1] | w11[pair...] ]
// and `ww` present only for the Lame energies at D >= 2. This reproduces the
// previous per-D layouts entry for entry, so `cpu-impl` / `cuda-impl` compile
// and behave unchanged.
//
// ---------------------------------------------------------------------------
//  Behaviour differences from the hand-expanded code (deliberate)
// ---------------------------------------------------------------------------
//  * `diag_*` now returns the EXACT matrix diagonal everywhere (#50 decision 1),
//    as for field. The old form subtracted `Sum_t w_t*sgn_t`, which is right in
//    the interior and wrong at every boundary voxel under every condition. The
//    Lame cross block contributes nothing here: it writes `A[(x,c)][(x',e)]`
//    with `e != c`, never a diagonal entry -- verified by measurement, not by
//    this sentence (see the diagonal probe in the PR).
//  * The JRLS weight map is read through `smag`, which returns 0 where the
//    folded tap is not a real memory location, instead of dereferencing out of
//    bounds (kernels#39).
//  * `make_fullkernel_lame` at D == 1 dropped the `/ v[0]` on its centre entry,
//    so a non-unit voxel size gave a centre weight inconsistent with
//    `make_kernel_lame`'s. Every other D, and every other energy, divides.
//    Generic code cannot reproduce a per-D transcription slip.
//  * `matvec_bending_jrls` (D == 3, the only D that had one) built its four
//    (y,z)-corner coefficients from the (x,y) corner entries of the weight map
//    -- `w112`/`w132`/`w312`/`w332` where `w211`/`w213`/`w231`/`w233` were
//    meant, the latter being computed and then never used. Same class of
//    copy-paste slip.
//  * `bending_jrls` now exists for D == 1 and D == 2 as well. It was 3-D only
//    (2-D's was `#if 0`-ed out, 1-D never had one); a generic engine has no way
//    to omit it. Nothing calls it yet.
#include "fastfields/core/cuda_switch.h"
#include "../../bounds.h"
#include "../../stap.h"
#include "../../utils.h"
#include "../stencil.h"
#include "utils.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_flow)

template <int _D, typename _scalar_t, typename _reduce_t, typename _offset_t,
          bound::type... _B>
struct RegFlow
{
    using scalar_t = _scalar_t;
    using reduce_t = _reduce_t;
    using offset_t = _offset_t;
    typedef scalar_t & (*OpType)(scalar_t &, const reduce_t &);

    static constexpr int D = _D;
    static constexpr int C = _D;   // a flow carries one channel per spatial axis

    static_assert(D >= 1, "the flow regulariser needs at least one spatial axis");
    static_assert(sizeof...(_B) == _D,
                  "one boundary condition per spatial axis");

    using S = reg::stencil<D, scalar_t, reduce_t, offset_t, Bound<_B...>>;

    // Zero / Replicate / DFT / NoCheck are their own transpose, so the Lame
    // cross block's adjoint tap table is then the natural one and is never
    // built (see `run_matvec_lame`).
    static constexpr bool adjoint_is_natural =
        ((bound::transpose(_B) == _B) && ...);

    static constexpr int NPAIR = S::NPAIR;
    static constexpr int KMEM  = S::KSIZE_MEMBRANE;      // per component
    static constexpr int KBEND = S::KSIZE_BENDING;       // per component
    // the shared Lame coefficient `ww` exists only where an axis pair does
    static constexpr int NWW   = NPAIR > 0 ? 1 : 0;

    static inline CUDEV constexpr int kw1(int d) { return S::kw1(d); }
    static inline CUDEV constexpr int kw2(int d) { return S::kw2(d); }
    static inline CUDEV constexpr int kwc(int p) { return S::kwc(p); }
    static inline CUDEV constexpr int pairidx(int d, int e) { return S::pairidx(d, e); }

    // ------------------------------------------------------------------
    //  kernel sizes  (identical to the per-D tables they replace)
    // ------------------------------------------------------------------

    static constexpr int kernelsize_absolute      = C;
    static constexpr int kernelsize_membrane      = KMEM  * C;
    static constexpr int kernelsize_bending       = KBEND * C;
    static constexpr int kernelsize_lame          = KMEM  * C + NWW;
    static constexpr int kernelsize_all           = KBEND * C + NWW;
    static constexpr int kernelsize_membrane_jrls = kernelsize_membrane;
    // the JRLS Lame cross carries the two Lame constants separately rather than
    // their sum, so it has two trailing entries where the plain form has one
    static constexpr int kernelsize_lame_jrls     = KMEM  * C + 2*NWW;
    // ... and the JRLS bending table is SHARED across components (the per-
    // component division happens on the result), preceded by the D voxel terms
    static constexpr int kernelsize_bending_jrls  = D + KBEND;

    // ------------------------------------------------------------------
    //  weight-table construction
    // ------------------------------------------------------------------
    // `v[d] = 1 / voxel_size[d]^2`, and every component's table is the shared
    // one divided by `v[c]`.
    //
    // The arithmetic below reproduces the hand-expanded tables' own operation
    // ASSOCIATIONS, not merely their values, so a bit-identical result survives
    // an anisotropic voxel size. That is why, for instance, the bending
    // first-order weight has two spellings: `make_kernel_bending` at D >= 2
    // expanded the product and `make_kernel_all` (and 1-D bending) factored it,
    // and the two differ in the last ULP.

    static inline CUDEV void
    lambdas(reduce_t * v, reduce_t & vsum, const reduce_t * voxel_size)
    {
        vsum = static_cast<reduce_t>(0);
        for (int d = 0; d < D; ++d) {
            const reduce_t s = voxel_size[d];
            v[d] = static_cast<reduce_t>(1) / (s * s);
            vsum += v[d];
        }
    }

    template <bool Factored>
    static inline CUDEV reduce_t
    w1_bending(reduce_t b, reduce_t m, const reduce_t * v, reduce_t vsum, int d)
    {
        if constexpr (Factored) return (-4 * b * vsum - m) * v[d];
        else                    return -4 * b * v[d] * vsum - m * v[d];
    }

    /// the shared (component-independent) bending weights
    template <bool Factored>
    static inline CUDEV void
    bending_weights(reduce_t * w1, reduce_t * w2, reduce_t * w11,
                    reduce_t b, reduce_t m, const reduce_t * v, reduce_t vsum)
    {
        for (int d = 0; d < D; ++d) {
            w1[d] = w1_bending<Factored>(b, m, v, vsum, d);
            w2[d] = b * v[d] * v[d];
        }
        for (int d = 0; d < D; ++d)
            for (int e = d + 1; e < D; ++e)
                w11[pairidx(d, e)] = 2 * b * v[d] * v[e];
    }

    /// `w0` with every tap folded in: what `make_fullkernel_*` wants, before
    /// the per-component division. Multiplicity 2 per axis tap, 4 per corner.
    static inline CUDEV reduce_t
    fold_taps(reduce_t absolute, const reduce_t * w1, const reduce_t * w2,
              const reduce_t * w11, bool reach2)
    {
        reduce_t s1 = static_cast<reduce_t>(0), s2 = static_cast<reduce_t>(0);
        for (int d = 0; d < D; ++d) s1 += w1[d];
        if (reach2) {
            for (int d = 0; d < D; ++d) s1 += w2[d];
            for (int p = 0; p < NPAIR; ++p) s2 += w11[p];
            return absolute - 2 * s1 - 4 * s2;
        }
        return absolute - 2 * s1;
    }

    /// the first-order weight of component `c` along axis `d`, once the Lame
    /// terms are folded in. The subtraction CHAIN (rather than adding a
    /// precomputed `-2*shears - div`) is the association the hand-expanded
    /// tables used, and the two differ in the last ULP.
    static inline CUDEV reduce_t
    lame_w1(reduce_t w1, reduce_t shears, reduce_t div,
            const reduce_t * v, int c, int d)
    {
        const reduce_t x = w1 / v[c];
        return d == c ? x - 2*shears - div : x - shears * (v[d]/v[c]);
    }

    /// the shears part of the extra centre weight the Lame terms contribute
    /// once every tap is folded in: `2*shears*(2*v[c] + sum_{d != c} v[d])/v[c]`.
    /// The bracket is left-folded over d with the c-th term doubled in place,
    /// which is exactly how the hand-expanded tables spelled it (`2*vx+vy+vz`,
    /// `vx+2*vy+vz`, ...), so the result is bit-identical for any voxel size.
    /// `+ 2*div` is added at the call site rather than here, again to keep the
    /// original association.
    static inline CUDEV reduce_t
    lame_centre(reduce_t shears, const reduce_t * v, int c)
    {
        reduce_t acc = static_cast<reduce_t>(0);
        for (int d = 0; d < D; ++d) acc = d == 0 ? (d == c ? 2*v[d] : v[d])
                                                 : acc + (d == c ? 2*v[d] : v[d]);
        return 2 * shears * acc / v[c];
    }

    //==================================================================
    //                            ABSOLUTE
    //==================================================================

    /// kernel <- [abs/v[0], abs/v[1], ...]
    CUDEV static inline void
    make_kernel_absolute(
        reduce_t * kernel, reduce_t absolute, const reduce_t voxel_size[D])
    {
        reduce_t v[D], vsum;
        lambdas(v, vsum, voxel_size);
        for (int c = 0; c < C; ++c) kernel[c] = absolute / v[c];
    }

    template <OpType op = set>
    CUDEV static inline void
    matvec_absolute(
        scalar_t * out, const scalar_t * inp,
        offset_t osc, offset_t isc, const reduce_t kernel[])
    {
        for (int c = 0; c < C; ++c)
            op(out[osc*c], kernel[c] * static_cast<reduce_t>(inp[isc*c]));
    }

    template <OpType op = set>
    CUDEV static inline void
    kernel_absolute(scalar_t * out, offset_t osc, const reduce_t kernel[])
    {
        for (int c = 0; c < C; ++c) op(out[osc*c], kernel[c]);
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_absolute(scalar_t * out, offset_t osc, const reduce_t kernel[])
    {
        return kernel_absolute<op>(out, osc, kernel);
    }

    //==================================================================
    //                            MEMBRANE
    //==================================================================

    /// kernel <- [ (abs, w1[d]...) per component ]
    CUDEV static inline void
    make_kernel_membrane(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        const reduce_t voxel_size[D])
    {
        reduce_t v[D], vsum;
        lambdas(v, vsum, voxel_size);
        for (int c = 0; c < C; ++c, kernel += KMEM) {
            kernel[0] = absolute / v[c];
            for (int d = 0; d < D; ++d) kernel[kw1(d)] = -membrane * (v[d]/v[c]);
        }
    }

    /// kernel <- [ (w0 with every tap folded in, w1[d]...) per component ]
    CUDEV static inline void
    make_fullkernel_membrane(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        const reduce_t voxel_size[D])
    {
        reduce_t v[D], vsum;
        lambdas(v, vsum, voxel_size);
        const reduce_t w0 = absolute + 2 * membrane * vsum;
        for (int c = 0; c < C; ++c, kernel += KMEM) {
            kernel[0] = w0 / v[c];
            for (int d = 0; d < D; ++d) kernel[kw1(d)] = -membrane * (v[d]/v[c]);
        }
    }

    template <OpType op = set>
    CUDEV static inline void
    matvec_membrane(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[D], const offset_t size[D], const offset_t stride[D],
        offset_t osc, offset_t isc, const reduce_t kernel[])
    {
        S::template run_matvec<1, wmode::none, op>(
            out, inp, nullptr, loc, size, stride, nullptr,
            osc, isc, 0, kernel, C, KMEM);
    }

    template <OpType op = set>
    CUDEV static inline void
    kernel_membrane(
        scalar_t * out, offset_t sc, const offset_t stride[D],
        const reduce_t kernel[])
    {
        for (int c = 0; c < C; ++c)
            S::template stencil_write<1, op>(out + sc*c, stride, kernel + KMEM*c);
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_membrane(
        scalar_t * out, offset_t osc,
        const offset_t loc[D], const offset_t size[D],
        const reduce_t kernel[])
    {
        S::template run_diag<1, wmode::none, op>(
            out, nullptr, loc, size, nullptr, osc, 0, kernel, C, KMEM);
    }

    //==================================================================
    //                            BENDING
    //==================================================================

    /// kernel <- [ (abs, w1[d]..., w2[d]..., w11[pair]...) per component ]
    CUDEV static inline void
    make_kernel_bending(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        const reduce_t voxel_size[D])
    {
        reduce_t v[D], vsum;
        lambdas(v, vsum, voxel_size);
        reduce_t w1[D], w2[D], w11[S::NPAIRA];
        // `make_kernel_bending` expanded this product at D >= 2 and factored it
        // at D == 1; the two differ in the last ULP, so each is kept.
        bending_weights<D == 1>(w1, w2, w11, bending, membrane, v, vsum);

        for (int c = 0; c < C; ++c, kernel += KBEND) {
            kernel[0] = absolute / v[c];
            for (int d = 0; d < D; ++d) {
                kernel[kw1(d)] = w1[d] / v[c];
                kernel[kw2(d)] = w2[d] / v[c];
            }
            for (int p = 0; p < NPAIR; ++p) kernel[kwc(p)] = w11[p] / v[c];
        }
    }

    CUDEV static inline void
    make_fullkernel_bending(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        const reduce_t voxel_size[D])
    {
        reduce_t v[D], vsum;
        lambdas(v, vsum, voxel_size);
        reduce_t w1[D], w2[D], w11[S::NPAIRA];
        bending_weights<D == 1>(w1, w2, w11, bending, membrane, v, vsum);
        const reduce_t w0 = fold_taps(absolute, w1, w2, w11, /*reach2=*/true);

        for (int c = 0; c < C; ++c, kernel += KBEND) {
            kernel[0] = w0 / v[c];
            for (int d = 0; d < D; ++d) {
                kernel[kw1(d)] = w1[d] / v[c];
                kernel[kw2(d)] = w2[d] / v[c];
            }
            for (int p = 0; p < NPAIR; ++p) kernel[kwc(p)] = w11[p] / v[c];
        }
    }

    template <OpType op = set>
    CUDEV static inline void
    matvec_bending(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[D], const offset_t size[D], const offset_t stride[D],
        offset_t osc, offset_t isc, const reduce_t kernel[])
    {
        S::template run_matvec<2, wmode::none, op>(
            out, inp, nullptr, loc, size, stride, nullptr,
            osc, isc, 0, kernel, C, KBEND);
    }

    template <OpType op = set>
    CUDEV static inline void
    kernel_bending(
        scalar_t * out, offset_t sc, const offset_t stride[D],
        const reduce_t kernel[])
    {
        for (int c = 0; c < C; ++c)
            S::template stencil_write<2, op>(out + sc*c, stride, kernel + KBEND*c);
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_bending(
        scalar_t * out, offset_t osc,
        const offset_t loc[D], const offset_t size[D],
        const reduce_t kernel[])
    {
        S::template run_diag<2, wmode::none, op>(
            out, nullptr, loc, size, nullptr, osc, 0, kernel, C, KBEND);
    }

    //==================================================================
    //                     THE LAME CROSS-COUPLING BLOCK
    //==================================================================
    // Shared by `lame` and `all`. One reach-1 tap table per axis under the
    // ADJOINT condition (used for the block's OWN axis) and one under the
    // natural condition (used for the other axis).
    //
    // The four reads of a pair share ONE multiply by `ww`, and the sum is
    // walked in the order the hand-expanded bodies used -- over the axis PAIR
    // (lo < hi):  + (lo+1, hi-1)  + (lo-1, hi+1)  - (lo-1, hi-1)  - (lo+1, hi+1),
    // other axes in increasing order -- so the plain path stays bit-identical
    // rather than merely equal to a ULP.

    // Exactly one of the pair's two axes is the block's OWN axis; that is the
    // half read from the ADJOINT table. `RA`/`RN` are the two tables' reaches:
    // where a condition is its own transpose the caller passes the SAME table
    // twice and no second one is ever built.
    template <int c, int RA, int RN>
    static inline CUDEV FF_INLINE reduce_t
    cross_lame(const scalar_t * inp, offset_t isc,
               const stap<offset_t, RA> * tadj, const stap<offset_t, RN> * tnat,
               reduce_t ww)
    {
        reduce_t sum = static_cast<reduce_t>(0);
        for (int e = 0; e < D; ++e)
        {
            if (e == c) continue;
            // which of the pair is the block's own axis decides which table
            // each half reads; hoist that out of the four-term walk
            const bool own_lo = c < e;
            const offset_t * ao = tadj[c].off;   const int8_t * as = tadj[c].sgn;
            const offset_t * no = tnat[e].off;   const int8_t * ns = tnat[e].sgn;
            FF_STAP_UNROLL
            for (int q = 0; q < 4; ++q)
            {
                const int sl = (q == 0 || q == 3) ?  1 : -1;   // lower axis
                const int sh = (q == 1 || q == 3) ?  1 : -1;   // higher axis
                const int so = own_lo ? sl : sh;               // own (adjoint)
                const int se = own_lo ? sh : sl;               // other (natural)
                const reduce_t val = bound::cget<reduce_t>(
                    inp + isc*e, ao[so + RA] + no[se + RN],
                    static_cast<int8_t>(as[so + RA] * ns[se + RN]));
                sum = (q < 2) ? sum + val : sum - val;
            }
        }
        return ww * sum;
    }

    /// the cross block of the materialised (C, C) Toeplitz stencil: for each
    /// axis pair, both off-diagonal blocks carry `-ww*s*t` at `s*1_d + t*1_e`
    template <OpType op>
    static inline CUDEV void
    cross_lame_write(scalar_t * out, const offset_t sc[2],
                     const offset_t * stride, reduce_t ww)
    {
        for (int d = 0; d < D; ++d)
        for (int e = d + 1; e < D; ++e)
        {
            scalar_t * b0 = out + sc[0]*e + sc[1]*d;   // block (e, d)
            scalar_t * b1 = out + sc[0]*d + sc[1]*e;   // block (d, e)
            for (int t = -1; t <= 1; t += 2)
            for (int s = -1; s <= 1; s += 2)
            {
                const offset_t o = static_cast<offset_t>(s) * stride[d]
                                 + static_cast<offset_t>(t) * stride[e];
                const reduce_t w = -ww * static_cast<reduce_t>(s * t);
                op(b0[o], w);
                op(b1[o], w);
            }
        }
    }

    /// the shared driver for `lame` and `all`: the per-component stencil of
    /// reach R, plus the cross block, in one pass
    template <int R, OpType op>
    static inline CUDEV FF_INLINE void
    run_matvec_lame(scalar_t * out, const scalar_t * inp,
                    const offset_t * loc, const offset_t * size,
                    const offset_t * stride, offset_t osc, offset_t isc,
                    const reduce_t * kernel, int ksize)
    {
        stap<offset_t, R> ti[D];
        S::template fill_taps<R>(ti, loc, size, stride);

        // The cross block's natural half is the reach-1 window of `ti`, so it
        // needs no table of its own; and where EVERY axis is its own transpose
        // (Zero / Replicate / DFT / NoCheck) the adjoint half is `ti` too, so
        // the whole extra table folds away at compile time.
        stap<offset_t, 1> tadj[D];
        reduce_t ww = static_cast<reduce_t>(0);
        if constexpr (NPAIR > 0) {
            ww = kernel[ksize * C];
            if constexpr (!adjoint_is_natural)
                S::template fill_taps_adjoint<1>(tadj, loc, size, stride);
        }

        if constexpr (adjoint_is_natural) chan_lame<R, op, 0>(out, inp, osc, isc, ti, ti, kernel, ksize, ww);
        else                             chan_lame<R, op, 0>(out, inp, osc, isc, ti, tadj, kernel, ksize, ww);
    }

    /// the channel loop of `run_matvec_lame`, unrolled. `C == D` is a
    /// compile-time constant and the cross block's control flow is entirely a
    /// function of the channel, so making `c` a template parameter turns
    /// "which axis is mine" into immediates instead of a per-read select.
    template <int R, OpType op, int c, int RA>
    static inline CUDEV FF_INLINE void
    chan_lame(scalar_t * out, const scalar_t * inp, offset_t osc, offset_t isc,
              const stap<offset_t, R> * ti, const stap<offset_t, RA> * tadj,
              const reduce_t * kernel, int ksize, reduce_t ww)
    {
        if constexpr (c < C)
        {
            typename S::wnone w{};
            reduce_t acc = S::template stencil_matvec<R, wmode::none>(
                inp + isc*c, ti, kernel + ksize*c, w);
            if constexpr (NPAIR > 0)
                acc += cross_lame<c, RA, R>(inp, isc, tadj, ti, ww);
            op(out[osc*c], acc);
            chan_lame<R, op, c + 1>(out, inp, osc, isc, ti, tadj, kernel, ksize, ww);
        }
    }

    //==================================================================
    //                              LAME
    //==================================================================

    /// kernel <- [ (abs, w1[d]...) per component, ww ]
    CUDEV static inline void
    make_kernel_lame(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[D])
    {
        reduce_t v[D], vsum;
        lambdas(v, vsum, voxel_size);
        reduce_t w1[D];
        for (int d = 0; d < D; ++d) w1[d] = -membrane * v[d];

        reduce_t * k = kernel;
        for (int c = 0; c < C; ++c, k += KMEM) {
            k[0] = absolute / v[c];
            for (int d = 0; d < D; ++d) k[kw1(d)] = lame_w1(w1[d], shears, div, v, c, d);
        }
        if constexpr (NPAIR > 0) kernel[KMEM * C] = 0.25 * (shears + div);
    }

    CUDEV static inline void
    make_fullkernel_lame(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[D])
    {
        reduce_t v[D], vsum;
        lambdas(v, vsum, voxel_size);
        reduce_t w1[D];
        for (int d = 0; d < D; ++d) w1[d] = -membrane * v[d];
        const reduce_t w0 = fold_taps(absolute, w1, nullptr, nullptr,
                                      /*reach2=*/false);

        reduce_t * k = kernel;
        for (int c = 0; c < C; ++c, k += KMEM) {
            k[0] = w0 / v[c] + lame_centre(shears, v, c) + 2 * div;
            for (int d = 0; d < D; ++d) k[kw1(d)] = lame_w1(w1[d], shears, div, v, c, d);
        }
        if constexpr (NPAIR > 0) kernel[KMEM * C] = 0.25 * (shears + div);
    }

    template <OpType op = set>
    CUDEV static inline void
    matvec_lame(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[D], const offset_t size[D], const offset_t stride[D],
        offset_t osc, offset_t isc, const reduce_t kernel[])
    {
        run_matvec_lame<1, op>(out, inp, loc, size, stride, osc, isc,
                               kernel, KMEM);
    }

    template <OpType op = set>
    CUDEV static inline void
    kernel_lame(
        scalar_t * out, const offset_t sc[2], const offset_t stride[D],
        const reduce_t kernel[])
    {
        for (int c = 0; c < C; ++c)
            S::template stencil_write<1, op>(out + (sc[0] + sc[1])*c, stride,
                                             kernel + KMEM*c);
        if constexpr (NPAIR > 0)
            cross_lame_write<op>(out, sc, stride, kernel[KMEM * C]);
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_lame(
        scalar_t * out, offset_t osc,
        const offset_t loc[D], const offset_t size[D],
        const reduce_t kernel[])
    {
        // the cross block never touches a diagonal entry (it writes channel
        // e != c), so the diagonal is the per-component stencil's alone
        S::template run_diag<1, wmode::none, op>(
            out, nullptr, loc, size, nullptr, osc, 0, kernel, C, KMEM);
    }

    //==================================================================
    //                          LAME + BENDING
    //==================================================================

    /// kernel <- [ (abs, w1[d]..., w2[d]..., w11[pair]...) per component, ww ]
    static inline CUDEV void
    make_kernel_all(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[D])
    {
        reduce_t v[D], vsum;
        lambdas(v, vsum, voxel_size);
        reduce_t w1[D], w2[D], w11[S::NPAIRA];
        bending_weights<true>(w1, w2, w11, bending, membrane, v, vsum);

        reduce_t * k = kernel;
        for (int c = 0; c < C; ++c, k += KBEND) {
            k[0] = absolute / v[c];
            for (int d = 0; d < D; ++d) {
                k[kw1(d)] = lame_w1(w1[d], shears, div, v, c, d);
                k[kw2(d)] = w2[d] / v[c];
            }
            for (int p = 0; p < NPAIR; ++p) k[kwc(p)] = w11[p] / v[c];
        }
        if constexpr (NPAIR > 0) kernel[KBEND * C] = 0.25 * (shears + div);
    }

    static inline CUDEV void
    make_fullkernel_all(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[D])
    {
        reduce_t v[D], vsum;
        lambdas(v, vsum, voxel_size);
        reduce_t w1[D], w2[D], w11[S::NPAIRA];
        bending_weights<true>(w1, w2, w11, bending, membrane, v, vsum);
        const reduce_t w0 = fold_taps(absolute, w1, w2, w11, /*reach2=*/true);

        reduce_t * k = kernel;
        for (int c = 0; c < C; ++c, k += KBEND) {
            k[0] = w0 / v[c] + lame_centre(shears, v, c) + 2 * div;
            for (int d = 0; d < D; ++d) {
                k[kw1(d)] = lame_w1(w1[d], shears, div, v, c, d);
                k[kw2(d)] = w2[d] / v[c];
            }
            for (int p = 0; p < NPAIR; ++p) k[kwc(p)] = w11[p] / v[c];
        }
        if constexpr (NPAIR > 0) kernel[KBEND * C] = 0.25 * (shears + div);
    }

    template <OpType op = set>
    CUDEV static inline void
    matvec_all(
        scalar_t * out, const scalar_t * inp,
        const offset_t loc[D], const offset_t size[D], const offset_t stride[D],
        offset_t osc, offset_t isc, const reduce_t kernel[])
    {
        run_matvec_lame<2, op>(out, inp, loc, size, stride, osc, isc,
                               kernel, KBEND);
    }

    template <OpType op = set>
    CUDEV static inline void
    kernel_all(
        scalar_t * out, const offset_t sc[2], const offset_t stride[D],
        const reduce_t kernel[])
    {
        for (int c = 0; c < C; ++c)
            S::template stencil_write<2, op>(out + (sc[0] + sc[1])*c, stride,
                                             kernel + KBEND*c);
        if constexpr (NPAIR > 0)
            cross_lame_write<op>(out, sc, stride, kernel[KBEND * C]);
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_all(
        scalar_t * out, offset_t osc,
        const offset_t loc[D], const offset_t size[D],
        const reduce_t kernel[])
    {
        S::template run_diag<2, wmode::none, op>(
            out, nullptr, loc, size, nullptr, osc, 0, kernel, C, KBEND);
    }

    //==================================================================
    //                          ABSOLUTE JRLS
    //==================================================================
    // No spatial coupling: one shared weight scalar times the per-component
    // absolute table. No separate kernel table -- `make_kernel_absolute`'s.

    template <OpType op = set>
    static inline CUDEV void
    matvec_absolute_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        offset_t osc, offset_t isc, const reduce_t kernel[])
    {
        const reduce_t w = static_cast<reduce_t>(*wgt);
        for (int c = 0; c < C; ++c)
            op(out[osc*c], kernel[c] * w * static_cast<reduce_t>(inp[isc*c]));
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_absolute_jrls(
        scalar_t * out, const scalar_t * wgt,
        offset_t osc, const reduce_t kernel[])
    {
        const reduce_t w = static_cast<reduce_t>(*wgt);
        for (int c = 0; c < C; ++c) op(out[osc*c], kernel[c] * w);
    }

    //==================================================================
    //                          MEMBRANE JRLS
    //==================================================================

    CUDEV static inline void
    make_kernel_membrane_jrls(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        const reduce_t voxel_size[D])
    {
        make_kernel_membrane(kernel, absolute, membrane, voxel_size);
        // 1/2 normalises the two-point mean of the weight map that every
        // membrane coefficient carries (see `stencil.h`'s `coef_ax1`)
        for (int i = 0; i < kernelsize_membrane_jrls; ++i) kernel[i] *= 0.5;
    }

    template <OpType op = set>
    CUDEV static inline void
    matvec_membrane_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        const offset_t loc[D], const offset_t size[D],
        const offset_t istride[D], const offset_t wstride[D],
        offset_t osc, offset_t isc, const reduce_t kernel[])
    {
        S::template run_matvec<1, wmode::joint, op>(
            out, inp, wgt, loc, size, istride, wstride,
            osc, isc, 0, kernel, C, KMEM);
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_membrane_jrls(
        scalar_t * out, const scalar_t * wgt,
        const offset_t loc[D], const offset_t size[D],
        const offset_t wstride[D], offset_t osc, const reduce_t kernel[])
    {
        S::template run_diag<1, wmode::joint, op>(
            out, wgt, loc, size, wstride, osc, 0, kernel, C, KMEM);
    }

    //==================================================================
    //                          BENDING JRLS
    //==================================================================
    // The one energy whose table is SHARED across components: the leading D
    // entries are the `v[d]`, the rest is a single unscaled stencil table, and
    // the per-component `1/v[c]` is applied to the RESULT rather than to the
    // table.

    CUDEV static inline void
    make_kernel_bending_jrls(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane, reduce_t bending,
        const reduce_t voxel_size[D])
    {
        reduce_t v[D], vsum;
        lambdas(v, vsum, voxel_size);
        for (int d = 0; d < D; ++d) kernel[d] = v[d];

        reduce_t * k = kernel + D;
        reduce_t w1[D], w2[D], w11[S::NPAIRA];
        bending_weights<true>(w1, w2, w11, bending, membrane, v, vsum);
        k[0] = absolute;
        for (int d = 0; d < D; ++d) { k[kw1(d)] = w1[d]; k[kw2(d)] = w2[d]; }
        for (int p = 0; p < NPAIR; ++p) k[kwc(p)] = w11[p];

        // 1/4 normalises the four-point mean of the weight map every bending
        // coefficient carries; `k[0]` (absolute) is left alone because the
        // bending centre coefficient is unweighted.
        for (int i = 1; i < KBEND; ++i) k[i] *= 0.25;
    }

    template <OpType op = set>
    CUDEV static inline void
    matvec_bending_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        const offset_t loc[D], const offset_t size[D],
        const offset_t istride[D], const offset_t wstride[D],
        offset_t osc, offset_t isc, const reduce_t kernel[])
    {
        stap<offset_t, 2> ti[D], tw[D];
        S::template fill_taps<2>(ti, loc, size, istride);
        S::template fill_taps<2>(tw, loc, size, wstride);
        const auto w = S::template load_wnbr<2>(wgt, tw);

        for (int c = 0; c < C; ++c)
            op(out[osc*c],
               S::template stencil_matvec<2, wmode::joint>(
                   inp + isc*c, ti, kernel + D, w) / kernel[c]);
    }

    template <OpType op = set>
    CUDEV static inline void
    diag_bending_jrls(
        scalar_t * out, const scalar_t * wgt,
        const offset_t loc[D], const offset_t size[D],
        const offset_t wstride[D], offset_t osc, const reduce_t kernel[])
    {
        offset_t unit[D];
        for (int d = 0; d < D; ++d) unit[d] = static_cast<offset_t>(1);

        stap<offset_t, 2> ti[D], tw[D];
        S::template fill_taps<2>(ti, loc, size, unit);
        S::template fill_taps<2>(tw, loc, size, wstride);
        const auto w = S::template load_wnbr<2>(wgt, tw);

        for (int c = 0; c < C; ++c)
            op(out[osc*c],
               S::template stencil_diag<2, wmode::joint>(ti, kernel + D, w)
               / kernel[c]);
    }

    //==================================================================
    //                            LAME JRLS
    //==================================================================
    // The reweighted cross block carries the two Lame constants SEPARATELY --
    // `dw = shears/4` scales the block's own axis, `sw = div/4` the other's --
    // so it cannot share the plain form's single `ww`.
    //
    // Two things differ from the plain Lame cross and are reproduced rather
    // than "corrected", since nothing reaches this path and there is no oracle
    // for a change: it folds through the NATURAL condition on both axes (no
    // `bound::transpose`, so fastfields-lib#26 never reached the JRLS variant),
    // and its four terms run in the order ( -(s-,t-) -(s+,t+) +(s-,t+) +(s+,t-) )
    // with no shared multiply. Both are flagged in #59 as follow-up candidates.

    CUDEV static inline void
    make_kernel_lame_jrls(
        reduce_t * kernel, reduce_t absolute, reduce_t membrane,
        reduce_t shears, reduce_t div, const reduce_t voxel_size[D])
    {
        make_kernel_lame(kernel, absolute, membrane, shears, div, voxel_size);
        for (int i = 0; i < KMEM * C; ++i) kernel[i] *= 0.5;
        if constexpr (NPAIR > 0) {
            kernel[KMEM * C]     = 0.25 * shears;
            kernel[KMEM * C + 1] = 0.25 * div;
        }
    }

    template <OpType op = set>
    static inline CUDEV void
    matvec_lame_jrls(
        scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
        const offset_t loc[D], const offset_t size[D],
        const offset_t istride[D], const offset_t wstride[D],
        offset_t osc, offset_t isc, const reduce_t kernel[])
    {
        stap<offset_t, 1> ti[D], tw[D];
        S::template fill_taps<1>(ti, loc, size, istride);
        S::template fill_taps<1>(tw, loc, size, wstride);
        const auto w = S::template load_wnbr<1>(wgt, tw);

        reduce_t dw = static_cast<reduce_t>(0), sw = static_cast<reduce_t>(0);
        if constexpr (NPAIR > 0) {
            dw = kernel[KMEM * C];
            sw = kernel[KMEM * C + 1];
        }

        for (int c = 0; c < C; ++c)
        {
            reduce_t acc = S::template stencil_matvec<1, wmode::joint>(
                inp + isc*c, ti, kernel + KMEM*c, w);

            if constexpr (NPAIR > 0)
                for (int e = 0; e < D; ++e)
                {
                    if (e == c) continue;
                    for (int q = 0; q < 4; ++q)
                    {
                        const int s = (q == 1 || q == 3) ?  1 : -1;
                        const int t = (q == 1 || q == 2) ?  1 : -1;
                        const reduce_t coef = dw * S::template wax<1>(w, c, s)
                                            + sw * S::template wax<1>(w, e, t);
                        const int ic = s + 1, ie = t + 1;
                        const reduce_t val = bound::cget<reduce_t>(
                            inp + isc*e, ti[c].off[ic] + ti[e].off[ie],
                            static_cast<int8_t>(ti[c].sgn[ic] * ti[e].sgn[ie]));
                        acc = (q < 2) ? acc - coef * val : acc + coef * val;
                    }
                }

            op(out[osc*c], acc);
        }
    }

    template <OpType op = set>
    static inline CUDEV void
    diag_lame_jrls(
        scalar_t * out, const scalar_t * wgt,
        const offset_t loc[D], const offset_t size[D],
        const offset_t wstride[D], offset_t osc, const reduce_t kernel[])
    {
        S::template run_diag<1, wmode::joint, op>(
            out, wgt, loc, size, wstride, osc, 0, kernel, C, KMEM);
    }
};

FF_NAMESPACE_END(reg_flow)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_REGULARISERS_FLOW_ND
