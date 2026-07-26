#ifndef FF_GATHER
#define FF_GATHER
// One separable weighted gather shared by pull / resize / restrict. A gather is
// a sum over the per-axis PRODUCT of taps: each spatial axis contributes a set of
// (offset, weight) pairs (a "row"), and the result accumulates
//   inp[ Σ_d off_d ] * Π_d weight_d
// over the Cartesian product of the per-axis rows. The only thing that differs
// between callers is how many taps a row has:
//   * row_k<K>  -- K taps, K a COMPILE-TIME count (the O+1 spline taps of pull /
//                  resize). The innermost loop bound is constant, so it fully
//                  unrolls -- identical codegen to a hand-unrolled gather.
//   * row_n     -- n taps, n a RUNTIME count (restrict's dilated CSR slice).
// Both share the SAME recursion. This is a pointer-level primitive: a gather
// over (offset, weight) tap lists only touches a base pointer + flat arrays, so
// it needs no tensor library. That is a layering choice, NOT a device
// constraint -- teeny is host+device and would run here fine; there is simply
// nothing for mdspan/extents/views to do in a raw weighted gather. Being CUDEV
// and dependency-light, it composes with a teeny-based caller or a raw-pointer
// one, on host or device, verbatim.
#include "cuda_switch.h"

// portable full-unroll hint (same policy as pushpull's FF_PP_UNROLL)
#if defined(__CUDACC__) || defined(__clang__)
#  define FF_GATHER_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#  define FF_GATHER_UNROLL _Pragma("GCC unroll 16")
#else
#  define FF_GATHER_UNROLL
#endif

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

// --- per-axis tap rows ------------------------------------------------------
// A row exposes flat weight/offset arrays and a `count` (static for row_k, so
// the loop unrolls; runtime for row_n). The recursion is generic over the type.
template <typename reduce_t, typename offset_t, int K>
struct row_k {                                   // K taps, compile-time count
    const reduce_t * w;
    const offset_t * o;
    static constexpr offset_t count = K;
};
template <typename reduce_t, typename offset_t>
struct row_n {                                   // n taps, runtime count
    const reduce_t * w;
    const offset_t * o;
    offset_t         count;
};

// --- the shared recursion ---------------------------------------------------
template <int d, int D, class Row, typename scalar_t, typename offset_t, typename reduce_t>
static inline CUDEV reduce_t
_sepgather(const scalar_t * inp, const Row * rows, offset_t base, reduce_t w) {
    reduce_t acc = static_cast<reduce_t>(0);
    const Row & r = rows[d];
    FF_GATHER_UNROLL
    for (offset_t k = 0; k < r.count; ++k) {     // r.count constant for row_k -> unrolls
        const offset_t o  = base + r.o[k];
        const reduce_t ww = w * r.w[k];
        if constexpr (d + 1 == D) acc += static_cast<reduce_t>(inp[o]) * ww;
        else                      acc += _sepgather<d + 1, D, Row, scalar_t, offset_t, reduce_t>(inp, rows, o, ww);
    }
    return acc;
}

// gather over the D per-axis rows (offsets relative to `inp`).
template <int D, class Row, typename scalar_t, typename offset_t, typename reduce_t>
static inline CUDEV reduce_t
gather_sep(const scalar_t * inp, const Row * rows) {
    return _sepgather<0, D, Row, scalar_t, offset_t, reduce_t>(
               inp, rows, static_cast<offset_t>(0), static_cast<reduce_t>(1));
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_GATHER
