#ifndef FF_POSDEF_CPU
#define FF_POSDEF_CPU
#include <cstdint>
#include <type_traits>
#include <vector>
#include <teeny/teeny.h>
#include "kernels/cuda_switch.h"
#include "kernels/posdef/matrix.h"
#include "kernels/parallel.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(posdef)

// The batch loop is teeny's peel: an `anyrank` over (*batch, packed) hands each
// voxel's rank-1 cell (the matrix/vector for that voxel) to the single-voxel
// kernels in kernels/posdef/matrix.h. peel_front_at<-1> folds the (arbitrarily
// strided) batch offset into each cell's pointer, so the old index2offset /
// internal::Pointer plumbing is gone. Static C recasts each cell to a static
// extent (the kernels then fold/unroll); dynamic C keeps the runtime extent and
// passes a heap CxC double workspace for the Cholesky path.
//
// The TENSORS are the arguments. Each entry point takes one teeny `anyrank`
// carrier per operand, built by the caller (*-lib) from that operand's OWN
// DLPack shape and strides -- so `nbatch`, `nchannel`, the shared `size[]` and
// every `stride_*[]` array are gone (TEENY-MIGRATION.md sec. 9, R2/R3). That
// retires the `_any` helper this file used to open with, whose whole job was to
// rebuild each carrier from ONE shared batch-shape array plus a per-tensor
// trailing extent -- the vectors trail C while the packed hessian trails
// C(C+1)/2, so a single shared `size[]` could decode one operand against
// another's geometry. Per-tensor carriers retire that hazard structurally.
//
// TEMPLATE SHAPE (Phase A's, fastfields-cpu-impl#60): one parameter per TENSOR,
// deduced. The element type is NOT a template parameter any more -- there is no
// scalar argument left that needs to name it, and the carriers carry it -- but
// dtype dispatch itself still happens in *-lib (R1): it is *-lib that picks
// `from_dlpack<float>` vs `<double>`, exactly as it used to pick `scalar_t`.
// `reduce_t` (the double accumulation type), the layout `Ty`, the static
// channel count `C` and the write mode `W` all still arrive from there.
//
// READ-ONLY operands are carriers of `const scalar_t` (R4); writing through a
// peeled cell of one is then a compile error rather than a convention.

// Element type of a carrier, const stripped -- read-only operands are carriers
// of `const scalar_t`, so the raw member type differs by cv-qualification only.
template <class A>
using _elem_t = typename std::remove_cv<
                    typename std::remove_pointer<decltype(A::data)>::type>::type;

// All operands of one call must agree on the element type. Reported here, in
// one line, rather than as a pointer-conversion cascade inside the kernels.
template <class A, class... Rest>
constexpr bool _same_elem = (std::is_same<_elem_t<A>, _elem_t<Rest>>::value && ...);

// The RUNTIME packed-length helper `_packed<Ty, C>(nchannel)` is gone with the
// prologue it served: its only callers were the `_any` calls that rebuilt the
// hessian carrier from the shared batch shape plus a computed trailing extent,
// and *-lib now hands that carrier down already shaped by the DLPack descriptor.
// The COMPILE-TIME twin below stays verbatim -- the static-C recast still needs
// CC as a constant expression.
template <type Ty, int C>   // compile-time CC for the static path (C>0)
static constexpr long _packed_s()
{
    return Ty == type::Eye ? 1 : Ty == type::Diag ? C
         : Ty == type::ESTATICS ? 2L * C - 1 : Ty == type::Sym ? long(C) * (C + 1) / 2
         : long(C) * C;
}

// C from the compact-symmetric packed length CC = C(C+1)/2 -- the inverse of
// _packed<type::Sym, -1>. invert[_]'s operands are ALL packed, so C is the one
// piece of geometry no carrier's extent spells out directly; it is needed only
// to size the CxC Cholesky workspace. *-lib validates CC as a triangular number
// (channels_from_packed) and throws before dispatching, so this inverse is exact
// on every value that reaches here. Runs once per call, outside the voxel loop.
template <typename offset_t>
static inline offset_t _channels_sym(offset_t CC)
{
    offset_t c = 0;
    while (c * (c + 1) / 2 < CC) ++c;
    return c;
}

template <type Ty, wr W, class Ov, class Hv, class Xv>
static inline void _dispatch_matvec(Ov&& o, const Hv& h, const Xv& x)
{
    if constexpr      (Ty == type::Eye)      eye::matvec<W>(o, h, x);
    else if constexpr (Ty == type::Diag)     diag::matvec<W>(o, h, x);
    else if constexpr (Ty == type::ESTATICS) estatics::matvec<W>(o, h, x);
    else if constexpr (Ty == type::Sym)      sym::matvec<W>(o, h, x);
    else                                     full::matvec<W>(o, h, x);
}

// in-place solve v <- (H + diag(w)) \ v for the selected layout. Sym/Full take
// the CxC double workspace M (Cholesky); Eye/Diag/ESTATICS ignore it.
template <type Ty, class Vv, class Hv, class Mv>
static inline void _dispatch_solve(Vv&& v, const Hv& h, Mv& M)
{
    if constexpr      (Ty == type::Eye)      eye::solve_(v, h);
    else if constexpr (Ty == type::Diag)     diag::solve_(v, h);
    else if constexpr (Ty == type::ESTATICS) estatics::solve_(v, h);
    else if constexpr (Ty == type::Sym)      sym::solve_w_(v, h, M);
    else                                     full::solve_w_(v, h, M);
}
template <type Ty, class Vv, class Hv, class Wv, class Mv>
static inline void _dispatch_solve(Vv&& v, const Hv& h, const Wv& w, Mv& M)
{
    if constexpr      (Ty == type::Eye)      eye::solve_(v, h, w);
    else if constexpr (Ty == type::Diag)     diag::solve_(v, h, w);
    else if constexpr (Ty == type::ESTATICS) estatics::solve_(v, h, w);
    else if constexpr (Ty == type::Sym)      sym::solve_w_(v, h, w, M);
    else                                     full::solve_w_(v, h, w, M);
}

// ---- matvec family (set / add / sub), any layout --------------------------
// ao: (*batch, C)   ah: (*batch, CC)   ai: (*batch, C)
template <type Ty, wr W, int C, typename reduce_t, class AO, class AH, class AI>
static void _matvec(AO ao, const AH ah, const AI ai)
{
    static_assert(_same_elem<AO, AH, AI>,
                  "posdef::matvec: out/hessian/input carriers must share an element type");
    using offset_t = decltype(ao.size(0));   // the carriers' own offset type
    const offset_t nvox = ao.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](int64_t start, int64_t end) {
    for (offset_t i = start; i < end; ++i) {
        auto o = ao.template peel_front_at<-1>(i);
        auto h = ah.template peel_front_at<-1>(i);
        auto x = ai.template peel_front_at<-1>(i);
        if constexpr (C > 0)
            _dispatch_matvec<Ty, W>(o.recast(tny::shape<C>{}),
                                    h.recast(tny::shape<_packed_s<Ty, C>()>{}),
                                    x.recast(tny::shape<C>{}));
        else
            _dispatch_matvec<Ty, W>(o, h, x);
    }});
}

template <type Ty, int C, typename reduce_t, class AO, class AH, class AI>
void matvec(AO ao, const AH ah, const AI ai)
{ _matvec<Ty, wr::set, C, reduce_t>(ao, ah, ai); }

template <type Ty, int C, typename reduce_t, class AO, class AH, class AI>
void addmatvec_(AO ao, const AH ah, const AI ai)
{ _matvec<Ty, wr::add, C, reduce_t>(ao, ah, ai); }

template <type Ty, int C, typename reduce_t, class AO, class AH, class AI>
void submatvec_(AO ao, const AH ah, const AI ai)
{ _matvec<Ty, wr::sub, C, reduce_t>(ao, ah, ai); }

// ---- matvec_backward: out(packed) = grad wrt H of <grd, H inp> -------------
// ao: (*batch, C(C+1)/2)   ag: (*batch, C)   ai: (*batch, C)
template <int C, typename reduce_t, class AO, class AG, class AI>
void sym_matvec_backward(AO ao, const AG ag, const AI ai)
{
    static_assert(_same_elem<AO, AG, AI>,
                  "posdef::sym_matvec_backward: all carriers must share an element type");
    using offset_t = decltype(ao.size(0));
    const offset_t nvox = ag.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](int64_t start, int64_t end) {
    for (offset_t i = start; i < end; ++i) {
        auto o = ao.template peel_front_at<-1>(i);
        auto g = ag.template peel_front_at<-1>(i);
        auto x = ai.template peel_front_at<-1>(i);
        if constexpr (C > 0) {
            constexpr long Cs = C, CCs = static_cast<long>(C) * (C + 1) / 2;
            sym::matvec_backward(o.recast(tny::shape<CCs>{}), x.recast(tny::shape<Cs>{}),
                                 g.recast(tny::shape<Cs>{}));
        } else {
            sym::matvec_backward(o, x, g);
        }
    }});
}

// ---- solve: out = (H + diag(w)) \ inp --------------------------------------
// The weight is OPTIONAL, and that is spelled as an arity -- two `solve`
// overloads -- not as a carrier with a null data pointer. A null-data sentinel
// would smuggle the pointer-era convention straight through the tensor
// boundary; *-lib's `has_wgt` test picks the overload instead, so the branch
// stops where the DLPack descriptor stops. One driver serves both arities: the
// weight rides a pack of size 0 or 1 that expands directly into the dispatch
// call, so `have_w` is no longer a runtime test inside the voxel loop either.
template <type Ty, typename reduce_t, class AO, class AI, class AH, class... AW>
static void _solve(AO ao, const AI ai, const AH ah, const AW &... aw)
{
    static_assert(sizeof...(AW) <= 1, "posdef::solve: at most one weight carrier");
    static_assert(_same_elem<AO, AI, AH, AW...>,
                  "posdef::solve: all carriers must share an element type");
    using offset_t = decltype(ao.size(0));
    // R2: the channel count is the vectors' trailing extent. NOT `ao.shape(-1)`
    // -- a from_dlpack carrier's `shape` is a fixed TNY_MAX_RANK store whose
    // leading `ndim` slots alone are live, so a negative wrap reads past them.
    const offset_t nchannel = ao.size(ao.ndim - 1);
    const offset_t nvox     = ao.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](int64_t start, int64_t end) {
    std::vector<reduce_t> b(nchannel * nchannel);   // Cholesky workspace (Sym/Full)
    auto M = tny::wrap(b.data(), tny::shape<-1,-1>{nchannel, nchannel});
    for (offset_t i = start; i < end; ++i) {
        auto o = ao.template peel_front_at<-1>(i);
        auto x = ai.template peel_front_at<-1>(i);
        auto h = ah.template peel_front_at<-1>(i);
        o.copy_(x);
        _dispatch_solve<Ty>(o, h, aw.template peel_front_at<-1>(i)..., M);
    }});
}

template <type Ty, typename reduce_t, class AO, class AI, class AH>
void solve(AO ao, const AI ai, const AH ah)
{ _solve<Ty, reduce_t>(ao, ai, ah); }

template <type Ty, typename reduce_t, class AO, class AI, class AH, class AW>
void solve(AO ao, const AI ai, const AH ah, const AW aw)
{ _solve<Ty, reduce_t>(ao, ai, ah, aw); }

// ---- solve_: in place, inp_out = (H + diag(w)) \ inp_out -------------------
template <type Ty, typename reduce_t, class AO, class AH, class... AW>
static void _solve_(AO ao, const AH ah, const AW &... aw)
{
    static_assert(sizeof...(AW) <= 1, "posdef::solve_: at most one weight carrier");
    static_assert(_same_elem<AO, AH, AW...>,
                  "posdef::solve_: all carriers must share an element type");
    using offset_t = decltype(ao.size(0));
    const offset_t nchannel = ao.size(ao.ndim - 1);
    const offset_t nvox     = ao.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](int64_t start, int64_t end) {
    std::vector<reduce_t> b(nchannel * nchannel);   // Cholesky workspace (Sym/Full)
    auto M = tny::wrap(b.data(), tny::shape<-1,-1>{nchannel, nchannel});
    for (offset_t i = start; i < end; ++i) {
        auto o = ao.template peel_front_at<-1>(i);
        auto h = ah.template peel_front_at<-1>(i);
        _dispatch_solve<Ty>(o, h, aw.template peel_front_at<-1>(i)..., M);
    }});
}

template <type Ty, typename reduce_t, class AO, class AH>
void solve_(AO ao, const AH ah)
{ _solve_<Ty, reduce_t>(ao, ah); }

template <type Ty, typename reduce_t, class AO, class AH, class AW>
void solve_(AO ao, const AH ah, const AW aw)
{ _solve_<Ty, reduce_t>(ao, ah, aw); }

// ---- invert: out = inv(H) (out-of-place) ----------------------------------
// ao, ah: (*batch, C(C+1)/2) -- both packed, so C comes from _channels_sym.
template <typename reduce_t, class AO, class AH>
void sym_invert(AO ao, const AH ah)
{
    static_assert(_same_elem<AO, AH>,
                  "posdef::sym_invert: all carriers must share an element type");
    using offset_t = decltype(ao.size(0));
    const offset_t CC       = ah.size(ah.ndim - 1);
    const offset_t nchannel = _channels_sym(CC);
    const offset_t nvox     = ao.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](int64_t start, int64_t end) {
    std::vector<reduce_t> b(nchannel * nchannel);
    auto M = tny::wrap(b.data(), tny::shape<-1,-1>{nchannel, nchannel});
    for (offset_t i = start; i < end; ++i) {
        auto o = ao.template peel_front_at<-1>(i);
        auto h = ah.template peel_front_at<-1>(i);
        for (offset_t k = 0; k < CC; ++k) o(k) = static_cast<reduce_t>(h(k));
        sym::invert_w_(o, M);
    }});
}

// ---- invert_: in place, hes = inv(hes) ------------------------------------
template <typename reduce_t, class AH>
void sym_invert_(AH ah)
{
    using offset_t = decltype(ah.size(0));
    const offset_t nchannel = _channels_sym(ah.size(ah.ndim - 1));
    const offset_t nvox     = ah.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](int64_t start, int64_t end) {
    std::vector<reduce_t> b(nchannel * nchannel);
    auto M = tny::wrap(b.data(), tny::shape<-1,-1>{nchannel, nchannel});
    for (offset_t i = start; i < end; ++i) {
        auto h = ah.template peel_front_at<-1>(i);
        sym::invert_w_(h, M);
    }});
}

FF_NAMESPACE_END(posdef)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_POSDEF_CPU
