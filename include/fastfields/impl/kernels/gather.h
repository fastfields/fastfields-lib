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
// Both share the SAME recursion. Device-capable (CUDEV) and C++11 (struct
// recursion, no `if constexpr`, no std / teeny), so CPU and the CUDA port use it
// verbatim over host or device pointers.
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
// A row exposes flat weight/offset arrays and a static count(row) accessor; the
// recursion is generic over the row type.
template <typename reduce_t, typename offset_t, int K>
struct row_k {                                   // K taps, compile-time count
    const reduce_t * w;
    const offset_t * o;
    static inline CUDEV offset_t count(const row_k &) { return static_cast<offset_t>(K); }
};
template <typename reduce_t, typename offset_t>
struct row_n {                                   // n taps, runtime count
    const reduce_t * w;
    const offset_t * o;
    offset_t         n;
    static inline CUDEV offset_t count(const row_n & r) { return r.n; }
};

// --- the shared recursion ---------------------------------------------------
template <int d, int D, class Row, typename scalar_t, typename offset_t, typename reduce_t>
struct _sepgather {
    static inline CUDEV reduce_t
    go(const scalar_t * inp, const Row * rows, offset_t base, reduce_t w) {
        reduce_t acc = static_cast<reduce_t>(0);
        const Row & r = rows[d];
        const offset_t n = Row::count(r);        // constant for row_k -> loop unrolls
        FF_GATHER_UNROLL
        for (offset_t k = 0; k < n; ++k)
            acc += _sepgather<d + 1, D, Row, scalar_t, offset_t, reduce_t>::go(
                       inp, rows, base + r.o[k], w * r.w[k]);
        return acc;
    }
};
template <int D, class Row, typename scalar_t, typename offset_t, typename reduce_t>
struct _sepgather<D, D, Row, scalar_t, offset_t, reduce_t> {   // past the last axis
    static inline CUDEV reduce_t
    go(const scalar_t * inp, const Row *, offset_t base, reduce_t w) {
        return static_cast<reduce_t>(inp[base]) * w;
    }
};

// gather over the D per-axis rows (offsets relative to `inp`).
template <int D, class Row, typename scalar_t, typename offset_t, typename reduce_t>
static inline CUDEV reduce_t
gather_sep(const scalar_t * inp, const Row * rows) {
    return _sepgather<0, D, Row, scalar_t, offset_t, reduce_t>::go(
               inp, rows, static_cast<offset_t>(0), static_cast<reduce_t>(1));
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_GATHER
