#ifndef FF_STAP
#define FF_STAP
// Per-axis boundary-folded STencil TAP tables.
//
// A finite-difference stencil of reach R reads, along each axis d, the taps at
// x-R .. x+R. Near a boundary those reads must be FOLDED by the axis' boundary
// condition, which turns each tap into a pair
//
//      off = (index(x+k, n) - x) * stride      // where to read, relative to x
//      sgn = sign (x+k, n)                     // 0 = read nothing, -1 = negate
//
// `stap<offset_t,R>` is that pair list for ONE axis, built once per voxel.
// Everything downstream is arithmetic on the table: an N-D stencil is the
// per-axis tables combined, and a CORNER tap (the ±1/±1 diagonals of a bending
// or Lamé stencil) is just the componentwise combination
//
//      off = off_d + off_e,   sgn = sgn_d * sgn_e
//
// which is valid precisely because the boundary fold is SEPARABLE: folding a
// multi-index folds each axis independently. That separability is what lets one
// N-D engine replace the per-D hand-expanded stencil bodies.
//
// This header is deliberately ENERGY-AGNOSTIC: it knows about reach and
// boundary folding, and nothing about absolute/membrane/bending/Lamé or about
// weight tables. The field engine (`regularisers/field/nd.h`) and the flow
// engine both build on it.
//
// Like `gather.h` this is a pointer-level primitive -- flat little arrays and a
// base pointer, no tensor library needed -- so it composes with a teeny-based
// caller or a raw-pointer one, on host or device, verbatim.
#include "cuda_switch.h"
#include "bounds.h"

// portable full-unroll hint (same policy as gather.h's FF_GATHER_UNROLL)
#if defined(__CUDACC__) || defined(__clang__)
#  define FF_STAP_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#  define FF_STAP_UNROLL _Pragma("GCC unroll 8")
#else
#  define FF_STAP_UNROLL
#endif

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

// One axis' folded taps at k = -R .. +R, stored at slot `k + R`.
// Slot R is the CENTRE: always {off = 0, sgn = 1, inb = 1}, kept in the table so
// that corner combination and the diagonal formula need no special case.
//
// `inb` says whether the folded tap is a REAL memory location. A signed read
// (`bound::cget(ptr, off, sgn)`) does not need it -- it returns 0 without
// dereferencing when `sgn == 0` -- but an UNSIGNED read of a companion array
// (a strictly-positive RLS weight map) does. Under every condition whose
// `index()` cannot leave [0, n) this is the compile-time constant 1 and the
// gate disappears.
template <typename offset_t, int R>
struct stap
{
    static constexpr int reach = R;
    static constexpr int size  = 2*R + 1;
    static constexpr int mid   = R;         // slot of the centre tap

    offset_t off[2*R + 1];
    int8_t   sgn[2*R + 1];
    int8_t   inb[2*R + 1];

    // slot of tap k (k in [-R, R])
    static inline CUDEV constexpr int slot(int k) { return k + R; }
};

// How a reweighting map (RLS / JRLS) is laid out relative to the stencil.
// Energy-agnostic: it only says whether there IS a companion map and whether it
// is indexed per channel or shared across them.
enum class wmode
{
    none,     // no map -- the weight is a constant 1 and folds away entirely
    split,    // one map PER CHANNEL   (RLS):  advance by wsc with the channel
    joint     // one map for ALL channels (JRLS): read once, hoisted out of the
              //                                  channel loop
};

// Build the table for one axis.
//   bu : the axis' boundary condition, ALWAYS through `bound::dyn<B>` so the
//        static/dynamic build policy (fastfields-kernels#42) keeps working;
//        for a real B it is an empty object and folds away entirely.
//   x  : the voxel's coordinate along this axis
//   n  : the axis' extent
//   s  : the axis' stride, in ELEMENTS of whatever array will be read
//
// Note the table is stride-scaled, so a caller that reads two arrays with
// different strides (an input and a weight map, say) builds two tables.
template <int R, bound::type B, typename offset_t>
static inline CUDEV FF_INLINE stap<offset_t, R>
make_stap(bound::dyn<B> bu, offset_t x, offset_t n, offset_t s)
{
    constexpr bool always_inb = bound::index_stays_inbounds(B);
    stap<offset_t, R> t;
    FF_STAP_UNROLL
    for (int k = -R; k <= R; ++k)
    {
        const int i = k + R;
        if (k == 0) {
            t.off[i] = static_cast<offset_t>(0); t.sgn[i] = 1; t.inb[i] = 1;
            continue;
        }
        const offset_t c = x + static_cast<offset_t>(k);
        t.sgn[i] = bu.sign(c, n);
        t.off[i] = (bu.index(c, n) - x) * s;
        t.inb[i] = always_inb ? int8_t(1) : int8_t(t.sgn[i] != 0);
    }
    return t;
}

// Unsigned read of a companion array at a tap: the value where the tap is a real
// memory location, 0 where it is not. Used for the RLS/JRLS weight map, whose
// entries are magnitudes and must NOT pick up the boundary sign.
template <typename reduce_t, typename scalar_t, typename offset_t>
static inline CUDEV FF_INLINE reduce_t
smag(const scalar_t * ptr, offset_t off, int8_t inb)
{
    return inb ? static_cast<reduce_t>(ptr[off]) : static_cast<reduce_t>(0);
}

// The difference-form read: `sgn`-corrected value at a tap, MINUS the centre.
// Writing every stencil as `w0*centre + Σ_t w_t * delta(t)` is what makes the
// sign-carrying boundary reads reproduce the exact symmetric operator at the
// boundary with no explicit renormalisation.
template <typename reduce_t, typename scalar_t, typename offset_t>
static inline CUDEV FF_INLINE reduce_t
sdelta(const scalar_t * ptr, offset_t off, int8_t sgn, reduce_t centre)
{
    return bound::cget<reduce_t>(ptr, off, sgn) - centre;
}

// d(out)/d(centre) contributed by ONE difference-form tap `w * delta(t)`.
//
// Contract the stencil against the unit vector at the centre voxel: the tap
// reads `sgn * ptr[off]`, which is `sgn` when the tap folded back ONTO the
// centre and 0 otherwise; the `- centre` of the difference form always
// contributes -1. Hence
//
//      diag = w0 + Σ_t w_t * ((folded onto centre ? sgn_t : 0) - 1)
//
// In the interior nothing folds and this degenerates to the familiar
// `w0 - Σ_t w_t`. At a boundary it does NOT: a folding condition (DCT2, DFT, …)
// makes some taps land back on the centre and CONTRIBUTE +w*sgn, while a Zero
// condition zeroes the read but still keeps the -1. That is the exact matrix
// diagonal (fastfields-kernels#50 decision 1); it is not the same quantity as
// the older `-Σ_t w_t*sgn_t` approximation.
//
// `onto` is passed explicitly rather than derived from `off == 0` because a
// multi-axis (corner) tap only folds onto the centre when EVERY axis does --
// `off_d + off_e == 0` would also be true for two offsets that merely cancel.
template <typename reduce_t>
static inline CUDEV FF_INLINE reduce_t
sdiag(reduce_t w, bool onto, int8_t sgn)
{
    const reduce_t fold = onto ? static_cast<reduce_t>(sgn)
                               : static_cast<reduce_t>(0);
    return w * (fold - static_cast<reduce_t>(1));
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_STAP
