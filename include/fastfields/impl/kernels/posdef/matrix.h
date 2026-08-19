// matrix.h — compact positive-definite matrix kernels on teeny.
//
// One generic implementation per (layout, op), parameterised on the channel
// count C through the views' extents: static C folds/unrolls to the same code
// the old hand-written C=1/2/3 specializations produced, dynamic C runs the same
// source as a runtime loop. Replaces the CRTP `common<>`/`utils<>` scaffolding,
// the `internal::Pointer` strided-pointer abstraction, the template-recursive
// `sub2pak_rows<K>`, and the flat-buffer Cholesky of posdef/{cholesky.h,*.inl}.
//
// LAYOUTS (packed length CC, chosen from (C, CC) by guess_type):
//   Eye(1)  Diag(C)  ESTATICS(2C-1)  Sym(C(C+1)/2, diag-then-rows)  Full(C^2)
//
// ACCUMULATION CONTRACT (matches the old kernels bit-for-bit-ish): reduce_t is
// ALWAYS double, independent of the views' element type (float32/float64). Every
// accumulator and the Cholesky workspace is double; loads cast scalar->double,
// the final store casts back. teeny's elementwise engines (compute_type<float> ==
// float) are deliberately NOT used here — the kernels are explicit double loops
// and use teeny only for typed views, static folding, and the stack workspace.
//
// Numerics preserved: the 1.000001 diagonal ridge (to_full) and the 1e-40 pivot
// floor (cholesky_). Bugs fixed vs the old kernels: ESTATICS weighted solve now
// applies w[c] to every channel (the old one dropped it for c<C-1); Full loads
// the whole C^2 matrix and treats the weight as optional (the old one under-copied
// C(C+1)/2 into the C^2 buffer and dereferenced a null weight).
#ifndef FF_POSDEF_MATRIX
#define FF_POSDEF_MATRIX
#include <teeny/teeny.h>
#include "fastfields/core/cuda_switch.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>

// Full-unroll request for the static-C loops. NB gcc ignores `#pragma unroll`
// (what the old kernels used) — it needs `#pragma GCC unroll`.
#if defined(__clang__) || defined(__CUDACC__)
#  define FF_POSDEF_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#  define FF_POSDEF_UNROLL _Pragma("GCC unroll 16")
#else
#  define FF_POSDEF_UNROLL
#endif

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(posdef)

namespace tn = tny;
namespace cs = cuda::std;

using reduce_t = double;                                 // fixed accumulation type
inline constexpr reduce_t one_plus_tiny = 1.000001;      // JFH_OnePlusTiny ridge
inline constexpr reduce_t pivot_floor   = 1e-40;

// The storage layouts, selected at runtime from the packed length.
enum class type : uint8_t { None, Eye, Diag, ESTATICS, Sym, Full };

template <typename offset_t>
CUHOST inline type guess_type(offset_t C, offset_t CC)
{
    if      (CC == 0)             return type::None;
    else if (CC == 1)             return type::Eye;
    else if (CC == C)             return type::Diag;
    else if (CC == 2 * C - 1)     return type::ESTATICS;
    else if (CC == (C * (C + 1)) / 2) return type::Sym;
    else if (CC == C * C)         return type::Full;
    else throw std::runtime_error("Input does not look like a known matrix form");
}

// writer for the matvec family: o = Hx / o += Hx / o -= Hx
enum class wr { set, add, sub };

template <wr W, class Ov>
CUDEV inline void _write(Ov&& o, long c, reduce_t s)
{
    if      (W == wr::set) o(c) = s;
    else if (W == wr::add) o(c) = reduce_t(o(c)) + s;
    else                   o(c) = reduce_t(o(c)) - s;
}

// static extent (or dynamic_extent) of a view's axis 0
template <class V>
inline constexpr long _s0 = long(cs::decay_t<V>::shape_type::static_extent(0));
template <class V>
inline constexpr bool _is_static = _s0<V> != long(cs::dynamic_extent);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Eye
// h is a single scalar acting as h*I.
namespace eye {

template <wr W = wr::set, class Ov, class Hv, class Xv>
CUDEV void matvec(Ov&& o, const Hv& h, const Xv& x)
{
    const long C = long(x.extent(tn::Int<0>()));
    const reduce_t hh = h(0);
    for (long c = 0; c < C; ++c) _write<W>(o, c, hh * reduce_t(x(c)));
}
template <class Vv, class Hv>
CUDEV void solve_(Vv&& v, const Hv& h)
{
    const long C = long(v.extent(tn::Int<0>()));
    const reduce_t hh = h(0);
    for (long c = 0; c < C; ++c) v(c) = reduce_t(v(c)) / hh;
}
template <class Vv, class Hv, class Wv>
CUDEV void solve_(Vv&& v, const Hv& h, const Wv& w)
{
    const long C = long(v.extent(tn::Int<0>()));
    const reduce_t hh = h(0);
    for (long c = 0; c < C; ++c) v(c) = reduce_t(v(c)) / (hh + reduce_t(w(c)));
}
template <class Ov, class Hv>
CUDEV void invert(Ov&& o, const Hv& h) { o(0) = reduce_t(1) / reduce_t(h(0)); }

} // namespace eye

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Diag
namespace diag {

template <wr W = wr::set, class Ov, class Hv, class Xv>
CUDEV void matvec(Ov&& o, const Hv& h, const Xv& x)
{
    const long C = long(x.extent(tn::Int<0>()));
    for (long c = 0; c < C; ++c) _write<W>(o, c, reduce_t(h(c)) * reduce_t(x(c)));
}
template <class Vv, class Hv>
CUDEV void solve_(Vv&& v, const Hv& h)
{
    const long C = long(v.extent(tn::Int<0>()));
    for (long c = 0; c < C; ++c) v(c) = reduce_t(v(c)) / reduce_t(h(c));
}
template <class Vv, class Hv, class Wv>
CUDEV void solve_(Vv&& v, const Hv& h, const Wv& w)
{
    const long C = long(v.extent(tn::Int<0>()));
    for (long c = 0; c < C; ++c) v(c) = reduce_t(v(c)) / (reduce_t(h(c)) + reduce_t(w(c)));
}
template <class Ov, class Hv>
CUDEV void invert(Ov&& o, const Hv& h)
{
    const long C = long(h.extent(tn::Int<0>()));
    for (long c = 0; c < C; ++c) o(c) = reduce_t(1) / reduce_t(h(c));
}

} // namespace diag

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ESTATICS
// Packed 2C-1 = [ d0..d(C-1) diagonals | e0..e(C-2) couplings to the last chan ].
// Row c<C-1:  d[c]*x[c] + e[c]*x[C-1];   Row C-1: sum_c e[c]*x[c] + d[C-1]*x[C-1].
namespace estatics {

template <wr W = wr::set, class Ov, class Hv, class Xv>
CUDEV void matvec(Ov&& o, const Hv& h, const Xv& x)
{
    const long C  = long(x.extent(tn::Int<0>()));
    const reduce_t xl = x(C - 1);                       // last input
    reduce_t oo = (W == wr::set) ? reduce_t(0) : reduce_t(o(C - 1));
    for (long c = 0; c < C - 1; ++c) {
        const reduce_t d = h(c), e = h(C + c), xc = x(c);
        _write<W>(o, c, d * xc + e * xl);
        oo = (W == wr::sub) ? oo - e * xc : oo + e * xc;
    }
    const reduce_t last = reduce_t(h(C - 1)) * xl;
    o(C - 1) = (W == wr::sub) ? oo - last : oo + last;  // W==set: oo started 0
}

// v <- H \ v, via the Schur complement of the last channel.
template <class Vv, class Hv>
CUDEV void solve_(Vv&& v, const Hv& h)
{
    const long C = long(v.extent(tn::Int<0>()));
    reduce_t ov = 0, oh = reduce_t(h(C - 1));
    for (long c = 0; c < C - 1; ++c) {
        const reduce_t e = h(C + c), dc = h(c), t = e / dc;
        oh -= e * t;
        ov += reduce_t(v(c)) * t;
    }
    const reduce_t xl = (reduce_t(v(C - 1)) - ov) / oh;
    v(C - 1) = xl;
    for (long c = 0; c < C - 1; ++c) {
        const reduce_t e = h(C + c), dc = h(c);
        v(c) = (reduce_t(v(c)) - xl * e) / dc;
    }
}
// v <- (H + diag(w)) \ v — weight applied to EVERY channel (old kernel dropped
// it for c<C-1).
template <class Vv, class Hv, class Wv>
CUDEV void solve_(Vv&& v, const Hv& h, const Wv& w)
{
    const long C = long(v.extent(tn::Int<0>()));
    reduce_t ov = 0, oh = reduce_t(h(C - 1)) + reduce_t(w(C - 1));
    for (long c = 0; c < C - 1; ++c) {
        const reduce_t e = h(C + c), dc = reduce_t(h(c)) + reduce_t(w(c)), t = e / dc;
        oh -= e * t;
        ov += reduce_t(v(c)) * t;
    }
    const reduce_t xl = (reduce_t(v(C - 1)) - ov) / oh;
    v(C - 1) = xl;
    for (long c = 0; c < C - 1; ++c) {
        const reduce_t e = h(C + c), dc = reduce_t(h(c)) + reduce_t(w(c));
        v(c) = (reduce_t(v(c)) - xl * e) / dc;
    }
}

} // namespace estatics

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cholesky
// Shared by Sym and Full. All on the CxC DOUBLE workspace M.
namespace chol {

// in-place Cholesky-Banachiewicz; reads the upper triangle, writes L into lower.
template <class Mv>
CUDEV void factor_(Mv& M)
{
    const long C = long(M.extent(tn::Int<0>()));
    for (long c = 0; c < C; ++c)
        for (long b = c; b < C; ++b) {
            reduce_t sm = M(c, b);
            for (long d = c - 1; d >= 0; --d) sm -= reduce_t(M(c, d)) * reduce_t(M(b, d));
            if (c == b) M(c, c) = std::sqrt(sm > pivot_floor ? sm : pivot_floor);
            else        M(b, c) = sm / reduce_t(M(c, c));
        }
}
// solve L L^T x = x in place (x rounds through its scalar_t, like the original).
template <class Mv, class Xv>
CUDEV void solve_(const Mv& M, Xv&& x)
{
    const long C = long(M.extent(tn::Int<0>()));
    for (long c = 0; c < C; ++c) {
        reduce_t sm = x(c);
        for (long cc = c - 1; cc >= 0; --cc) sm -= reduce_t(M(c, cc)) * reduce_t(x(cc));
        x(c) = sm / reduce_t(M(c, c));
    }
    for (long c = C - 1; c >= 0; --c) {
        reduce_t sm = x(c);
        for (long cc = c + 1; cc < C; ++cc) sm -= reduce_t(M(cc, c)) * reduce_t(x(cc));
        x(c) = sm / reduce_t(M(c, c));
    }
}

} // namespace chol

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sym
// Packed C(C+1)/2, diag-then-rows. From the fable-validated design.
namespace sym {

// packed "diag then rows" index: (i,j) -> k. Closed form of sub2pak_rows<K>.
template <class Idx>
CUDEV constexpr Idx sub2pak(Idx C, Idx i, Idx j)
{
    const Idx a = i < j ? i : j, b = i < j ? j : i;
    return a == b ? a : C + a * (C - 1) - (a * (a + 1)) / 2 + (b - 1);
}
// row-major packed (no diag-first shuffle) — used by invert's in-place dance.
template <class Idx>
CUDEV constexpr Idx sub2pak_rows(Idx C, Idx i, Idx j)
{
    const Idx a = i < j ? i : j, b = i < j ? j : i;
    return a * C - (a * (a + 1)) / 2 + b;
}

template <wr W = wr::set, class Ov, class Hv, class Xv>
CUDEV void matvec(Ov&& o, const Hv& h, const Xv& x)
{
    const long C = long(x.extent(tn::Int<0>()));
    if constexpr (_is_static<Xv>) {
        // static C: snapshot inputs into double locals before writing outputs
        // (o may alias h/x for the compiler; the snapshots SROA to registers).
        constexpr long Cs = _s0<Xv>, CCs = Cs * (Cs + 1) / 2;
        auto hr = tn::local<reduce_t, tn::shape<CCs>>();
        auto xr = tn::local<reduce_t, tn::shape<Cs>>();
        FF_POSDEF_UNROLL
        for (long k = 0; k < CCs; ++k) hr(k) = h(k);
        FF_POSDEF_UNROLL
        for (long c = 0; c < Cs; ++c) xr(c) = x(c);
        FF_POSDEF_UNROLL
        for (long c = 0; c < Cs; ++c) {
            reduce_t s = hr(sub2pak(Cs, c, 0L)) * xr(0);
            FF_POSDEF_UNROLL
            for (long cc = 1; cc < Cs; ++cc) s += hr(sub2pak(Cs, c, cc)) * xr(cc);
            _write<W>(o, c, s);
        }
    } else {
        for (long c = 0; c < C; ++c) {
            reduce_t s = 0;
            for (long cc = 0; cc < C; ++cc) s += reduce_t(h(sub2pak(C, c, cc))) * reduce_t(x(cc));
            _write<W>(o, c, s);
        }
    }
}

// h(diag c) = x(c)*y(c);  h(off c,cc) = x(c)*y(cc) + x(cc)*y(c)
template <class Hv, class Xv, class Yv>
CUDEV void matvec_backward(Hv&& h, const Xv& x, const Yv& y)
{
    const long C = long(x.extent(tn::Int<0>()));
    for (long c = 0; c < C; ++c) h(c) = reduce_t(x(c)) * reduce_t(y(c));
    long k = C;
    for (long c = 0; c < C; ++c)
        for (long cc = c + 1; cc < C; ++cc, ++k)
            h(k) = reduce_t(x(c)) * reduce_t(y(cc)) + reduce_t(x(cc)) * reduce_t(y(c));
}

// packed -> full CxC double workspace: ridge on the diagonal, BOTH triangles.
template <class Mv, class Hv>
CUDEV void to_full(Mv& M, const Hv& h)
{
    const long C = long(M.extent(tn::Int<0>()));
    for (long c = 0; c < C; ++c) M(c, c) = reduce_t(h(c)) * one_plus_tiny;
    long k = C;
    for (long c = 0; c < C; ++c)
        for (long cc = c + 1; cc < C; ++cc, ++k) {
            const reduce_t v = h(k);
            M(cc, c) = v;
            M(c, cc) = v;
        }
}
template <class Mv, class Wv>
CUDEV void _add_diag(Mv& M, const Wv& w)
{
    const long C = long(M.extent(tn::Int<0>()));
    for (long c = 0; c < C; ++c) M(c, c) = reduce_t(M(c, c)) + reduce_t(w(c));
}

// v <- (H + diag(w)) \ v with an explicit rank-2 double workspace (any C).
template <class Vv, class Hv, class Mv>
CUDEV void solve_w_(Vv&& v, const Hv& h, Mv& M) { to_full(M, h); chol::factor_(M); chol::solve_(M, v); }
template <class Vv, class Hv, class Wv, class Mv>
CUDEV void solve_w_(Vv&& v, const Hv& h, const Wv& w, Mv& M)
{ to_full(M, h); _add_diag(M, w); chol::factor_(M); chol::solve_(M, v); }

// static-C convenience: makes its own double workspace.
template <class Vv, class Hv>
CUDEV void solve_(Vv&& v, const Hv& h)
{
    constexpr long Cs = _s0<Vv>;
    static_assert(_is_static<Vv>, "dynamic C: use solve_w_ with a workspace");
    auto M = tn::local<reduce_t, tn::shape<Cs, Cs>>();
    solve_w_(v, h, M);
}
template <class Vv, class Hv, class Wv>
CUDEV void solve_(Vv&& v, const Hv& h, const Wv& w)
{
    constexpr long Cs = _s0<Vv>;
    static_assert(_is_static<Vv>, "dynamic C: use solve_w_ with a workspace");
    auto M = tn::local<reduce_t, tn::shape<Cs, Cs>>();
    solve_w_(v, h, w, M);
}

// h <- packed(inv(H)) in place (basis-vector solves + rows-packed shuffle).
template <class Hv, class Mv>
CUDEV void invert_w_(Hv&& h, Mv& M)
{
    const long C = long(M.extent(tn::Int<0>()));
    to_full(M, h);
    chol::factor_(M);
    for (long kk = C - 1; kk > 0; --kk) {
        for (long c = 0; c < C; ++c) h(c) = (c == kk) ? reduce_t(1) : reduce_t(0);
        chol::solve_(M, h);
        for (long j = 1; j <= kk; ++j) h(sub2pak_rows(C, j, kk)) = reduce_t(h(j));
    }
    h(0) = 1;
    for (long c = 1; c < C; ++c) h(c) = 0;
    chol::solve_(M, h);
    long i = 0;
    for (long c = 0; c < C; ++c)
        for (long cc = c; cc < C; ++cc, ++i) M(c, cc) = reduce_t(h(i));
    i = 0;
    for (long c = 0; c < C; ++c, ++i) h(i) = reduce_t(M(c, c));
    for (long c = 0; c < C; ++c)
        for (long cc = c + 1; cc < C; ++cc, ++i) h(i) = reduce_t(M(c, cc));
}
template <class Hv>
CUDEV void invert_(Hv&& h)
{
    constexpr long CCs = _s0<Hv>;
    static_assert(CCs != long(cs::dynamic_extent), "dynamic C: use invert_w_");
    constexpr long Cs = [] { long c = 1; while (c * (c + 1) / 2 < CCs) ++c; return c; }();
    static_assert(Cs * (Cs + 1) / 2 == CCs, "packed length must be C(C+1)/2");
    auto M = tn::local<reduce_t, tn::shape<Cs, Cs>>();
    invert_w_(h, M);
}
// out-of-place: o <- packed(inv(H)). Static-C convenience (own workspace).
template <class Ov, class Hv>
CUDEV void invert(Ov&& o, const Hv& h)
{
    const long CC = long(h.extent(tn::Int<0>()));
    for (long k = 0; k < CC; ++k) o(k) = reduce_t(h(k));
    invert_(o);
}

} // namespace sym

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Full
// Dense C^2 row-major, assumed symmetric. Reuses the Cholesky core.
namespace full {

template <wr W = wr::set, class Ov, class Hv, class Xv>
CUDEV void matvec(Ov&& o, const Hv& h, const Xv& x)
{
    const long C = long(x.extent(tn::Int<0>()));
    for (long c = 0; c < C; ++c) {
        reduce_t s = 0;
        for (long cc = 0; cc < C; ++cc) s += reduce_t(h(c * C + cc)) * reduce_t(x(cc));
        _write<W>(o, c, s);
    }
}
template <class Mv, class Hv>
CUDEV void _to_full(Mv& M, const Hv& h)     // load C^2, ridge on the diagonal
{
    const long C = long(M.extent(tn::Int<0>()));
    for (long c = 0; c < C; ++c)
        for (long cc = 0; cc < C; ++cc)
            M(c, cc) = (c == cc) ? reduce_t(h(c * C + cc)) * one_plus_tiny
                                 : reduce_t(h(c * C + cc));
}
template <class Vv, class Hv, class Mv>
CUDEV void solve_w_(Vv&& v, const Hv& h, Mv& M) { _to_full(M, h); chol::factor_(M); chol::solve_(M, v); }
template <class Vv, class Hv, class Wv, class Mv>
CUDEV void solve_w_(Vv&& v, const Hv& h, const Wv& w, Mv& M)
{ _to_full(M, h); sym::_add_diag(M, w); chol::factor_(M); chol::solve_(M, v); }

template <class Vv, class Hv>
CUDEV void solve_(Vv&& v, const Hv& h)
{
    constexpr long Cs = _s0<Vv>;
    static_assert(_is_static<Vv>, "dynamic C: use solve_w_ with a workspace");
    auto M = tn::local<reduce_t, tn::shape<Cs, Cs>>();
    solve_w_(v, h, M);
}
template <class Vv, class Hv, class Wv>
CUDEV void solve_(Vv&& v, const Hv& h, const Wv& w)
{
    constexpr long Cs = _s0<Vv>;
    static_assert(_is_static<Vv>, "dynamic C: use solve_w_ with a workspace");
    auto M = tn::local<reduce_t, tn::shape<Cs, Cs>>();
    solve_w_(v, h, w, M);
}

} // namespace full

FF_NAMESPACE_END(posdef)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_POSDEF_MATRIX
