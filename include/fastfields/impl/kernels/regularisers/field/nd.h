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
#include "fastfields/core/cuda_switch.h"
#include "../../bounds.h"
#include "../../stap.h"
#include "../../utils.h"
#include "../stencil.h"
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

    // The shared per-component stencil engine: tap tables, weight-table
    // geometry, the RLS/JRLS neighbourhood and the three contractions
    // (`regularisers/stencil.h`). This class adds only what is field-specific:
    // the weight tables and the entry points.
    using S = reg::stencil<D, scalar_t, reduce_t, offset_t, typename _Config::Bound>;

    static constexpr int NPAIR          = S::NPAIR;
    static constexpr int KSIZE_ABSOLUTE = S::KSIZE_ABSOLUTE;
    static constexpr int KSIZE_MEMBRANE = S::KSIZE_MEMBRANE;
    static constexpr int KSIZE_BENDING  = S::KSIZE_BENDING;

    static inline CUDEV constexpr int kw1(int d) { return S::kw1(d); }
    static inline CUDEV constexpr int kw2(int d) { return S::kw2(d); }
    static inline CUDEV constexpr int kwc(int p) { return S::kwc(p); }
    static inline CUDEV constexpr int pairidx(int d, int e) { return S::pairidx(d, e); }

    static inline CUDEV FF_INLINE offset_t nchannels(offset_t nc)
    { return C < 0 ? nc : C; }

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
        S::template run_matvec<1, wmode::none, op>(out, inp, nullptr, loc, size, stride,
                                       nullptr, osc, isc, 0, kernel, nchannels(nc),
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
            S::template stencil_write<1, op>(out + sc*c, stride, kernel + KSIZE_MEMBRANE*c);
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
        S::template run_diag<1, wmode::none, op>(out, nullptr, loc, size, nullptr, osc, 0,
                                     kernel, nchannels(nc), KSIZE_MEMBRANE);
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
        S::template run_matvec<1, wmode::split, op>(out, inp, wgt, loc, size, istride,
                                        wstride, osc, isc, wsc, kernel, nchannels(nc),
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
        S::template run_diag<1, wmode::split, op>(out, wgt, loc, size, wstride, osc, wsc,
                                      kernel, nchannels(nc), KSIZE_MEMBRANE);
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
        S::template run_matvec<1, wmode::joint, op>(out, inp, wgt, loc, size, istride,
                                        wstride, osc, isc, 0, kernel, nchannels(nc),
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
        S::template run_diag<1, wmode::joint, op>(out, wgt, loc, size, wstride, osc, 0,
                                      kernel, nchannels(nc), KSIZE_MEMBRANE);
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
        S::template run_matvec<2, wmode::none, op>(out, inp, nullptr, loc, size, stride,
                                       nullptr, osc, isc, 0, kernel, nchannels(nc),
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
            S::template stencil_write<2, op>(out + sc*c, stride, kernel + KSIZE_BENDING*c);
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
        S::template run_diag<2, wmode::none, op>(out, nullptr, loc, size, nullptr, osc, 0,
                                     kernel, nchannels(nc), KSIZE_BENDING);
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
        S::template run_matvec<2, wmode::split, op>(out, inp, wgt, loc, size, istride,
                                        wstride, osc, isc, wsc, kernel, nchannels(nc),
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
        S::template run_diag<2, wmode::split, op>(out, wgt, loc, size, wstride, osc, wsc,
                                      kernel, nchannels(nc), KSIZE_BENDING);
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
        S::template run_matvec<2, wmode::joint, op>(out, inp, wgt, loc, size, istride,
                                        wstride, osc, isc, 0, kernel, nchannels(nc),
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
        S::template run_diag<2, wmode::joint, op>(out, wgt, loc, size, wstride, osc, 0,
                                      kernel, nchannels(nc), KSIZE_BENDING);
    }
};

FF_NAMESPACE_END(reg_field)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_REGULARISERS_FIELD_ND
