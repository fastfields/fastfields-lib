#ifndef FF_POSDEF_CPU
#define FF_POSDEF_CPU
#include <teeny/teeny.h>
#include <vector>
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
// The impl receives ONE `size` array (shared batch dims + one trailing), but the
// tensors differ in trailing length (vectors = C, packed hessian = C(C+1)/2), so
// each anyrank is built with its OWN trailing extent over the shared batch dims.

template <typename T, typename offset_t>
static inline auto _any(T* p, const offset_t* size, offset_t nbatch,
                        offset_t trailing, const offset_t* stride)
{
    std::vector<offset_t> sz(size, size + nbatch);
    sz.push_back(trailing);
    return tny::as_anyrank(p, sz.data(), stride, static_cast<int>(nbatch + 1), tny::copy_meta);
}

// packed length CC for a layout at channel count C (static path uses C>0).
template <type Ty, int C, typename offset_t>
static inline offset_t _packed(offset_t nchannel)
{
    const offset_t c = (C > 0) ? offset_t(C) : nchannel;
    switch (Ty) {
        case type::Eye:      return 1;
        case type::Diag:     return c;
        case type::ESTATICS: return 2 * c - 1;
        case type::Sym:      return c * (c + 1) / 2;
        default:             return c * c;         // Full
    }
}
template <type Ty, int C>   // compile-time CC for the static path (C>0)
static constexpr long _packed_s()
{
    return Ty == type::Eye ? 1 : Ty == type::Diag ? C
         : Ty == type::ESTATICS ? 2L * C - 1 : Ty == type::Sym ? long(C) * (C + 1) / 2
         : long(C) * C;
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

// ---- matvec family (set / add / sub), any layout --------------------------
template <type Ty, wr W, int C, typename reduce_t, typename scalar_t, typename offset_t>
static void _matvec(
          offset_t   nbatch,   offset_t nchannel,
          scalar_t * out, const scalar_t * hes, const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out, const offset_t * stride_hes, const offset_t * stride_inp)
{
    const offset_t CC = _packed<Ty, C>(nchannel);
    auto ao = _any(out, size, nbatch, nchannel, stride_out);
    auto ah = _any(hes, size, nbatch, CC,       stride_hes);
    auto ai = _any(inp, size, nbatch, nchannel, stride_inp);
    const offset_t nvox = ao.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
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

template <type Ty, int C, typename reduce_t, typename scalar_t, typename offset_t>
void matvec(offset_t nbatch, offset_t nchannel, scalar_t* out,
                const scalar_t* hes, const scalar_t* inp, const offset_t* size,
                const offset_t* stride_out, const offset_t* stride_hes, const offset_t* stride_inp)
{ _matvec<Ty, wr::set, C, reduce_t>(nbatch, nchannel, out, hes, inp, size, stride_out, stride_hes, stride_inp); }

template <type Ty, int C, typename reduce_t, typename scalar_t, typename offset_t>
void addmatvec_(offset_t nbatch, offset_t nchannel, scalar_t* out,
                const scalar_t* hes, const scalar_t* inp, const offset_t* size,
                const offset_t* stride_out, const offset_t* stride_hes, const offset_t* stride_inp)
{ _matvec<Ty, wr::add, C, reduce_t>(nbatch, nchannel, out, hes, inp, size, stride_out, stride_hes, stride_inp); }

template <type Ty, int C, typename reduce_t, typename scalar_t, typename offset_t>
void submatvec_(offset_t nbatch, offset_t nchannel, scalar_t* out,
                const scalar_t* hes, const scalar_t* inp, const offset_t* size,
                const offset_t* stride_out, const offset_t* stride_hes, const offset_t* stride_inp)
{ _matvec<Ty, wr::sub, C, reduce_t>(nbatch, nchannel, out, hes, inp, size, stride_out, stride_hes, stride_inp); }

// ---- matvec_backward: out(packed) = grad wrt H of <grd, H inp> -------------
template <int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_matvec_backward(offset_t nbatch, offset_t nchannel, scalar_t* out,
                const scalar_t* grd, const scalar_t* inp, const offset_t* size,
                const offset_t* stride_out, const offset_t* stride_grd, const offset_t* stride_inp)
{
    const offset_t CC = _packed<type::Sym, C>(nchannel);
    auto ao = _any(out, size, nbatch, CC,       stride_out);
    auto ag = _any(grd, size, nbatch, nchannel, stride_grd);
    auto ai = _any(inp, size, nbatch, nchannel, stride_inp);
    const offset_t nvox = ag.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
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

// ---- solve: out = (H + diag(w)) \ inp  (w optional via null wgt) -----------
template <int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_solve_tpl(offset_t nbatch, scalar_t* out, const scalar_t* inp,
                const scalar_t* hes, const scalar_t* wgt, const offset_t* size,
                const offset_t* stride_out, const offset_t* stride_inp,
                const offset_t* stride_hes, const offset_t* stride_wgt)
{
    constexpr long Cs = C, CCs = static_cast<long>(C) * (C + 1) / 2;
    auto ao = _any(out, size, nbatch, offset_t(C),   stride_out);
    auto ai = _any(inp, size, nbatch, offset_t(C),   stride_inp);
    auto ah = _any(hes, size, nbatch, offset_t(CCs), stride_hes);
    const offset_t nvox = ao.template size_front<-1>();
    const bool have_w = (wgt != nullptr);
    auto aw = _any(have_w ? wgt : inp, size, nbatch, offset_t(C), have_w ? stride_wgt : stride_inp);
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = start; i < end; ++i) {
        auto o = ao.template peel_front_at<-1>(i).recast(tny::shape<Cs>{});
        auto x = ai.template peel_front_at<-1>(i).recast(tny::shape<Cs>{});
        auto h = ah.template peel_front_at<-1>(i).recast(tny::shape<CCs>{});
        o.copy_(x);
        if (have_w) sym::solve_(o, h, aw.template peel_front_at<-1>(i).recast(tny::shape<Cs>{}));
        else        sym::solve_(o, h);
    }});
}

template <typename reduce_t, typename scalar_t, typename offset_t>
void sym_solve(offset_t nbatch, offset_t nchannel, scalar_t* out, const scalar_t* inp,
                const scalar_t* hes, const scalar_t* wgt, const offset_t* size,
                const offset_t* stride_out, const offset_t* stride_inp,
                const offset_t* stride_hes, const offset_t* stride_wgt)
{
    const offset_t CC = nchannel * (nchannel + 1) / 2;
    auto ao = _any(out, size, nbatch, nchannel, stride_out);
    auto ai = _any(inp, size, nbatch, nchannel, stride_inp);
    auto ah = _any(hes, size, nbatch, CC,       stride_hes);
    const offset_t nvox = ao.template size_front<-1>();
    const bool have_w = (wgt != nullptr);
    auto aw = _any(have_w ? wgt : inp, size, nbatch, nchannel, have_w ? stride_wgt : stride_inp);
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    std::vector<reduce_t> b(nchannel * nchannel);
    auto M = tny::wrap(b.data(), tny::shape<-1,-1>{nchannel, nchannel});
    for (offset_t i = start; i < end; ++i) {
        auto o = ao.template peel_front_at<-1>(i);
        auto x = ai.template peel_front_at<-1>(i);
        auto h = ah.template peel_front_at<-1>(i);
        o.copy_(x);
        if (have_w) sym::solve_w_(o, h, aw.template peel_front_at<-1>(i), M);
        else        sym::solve_w_(o, h, M);
    }});
}

// ---- solve_: in place, inp_out = (H + diag(w)) \ inp_out -------------------
template <int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_solve_tpl_(offset_t nbatch, scalar_t* out, const scalar_t* hes,
                const scalar_t* wgt, const offset_t* size,
                const offset_t* stride_out, const offset_t* stride_hes, const offset_t* stride_wgt)
{
    constexpr long Cs = C, CCs = static_cast<long>(C) * (C + 1) / 2;
    auto ao = _any(out, size, nbatch, offset_t(C),   stride_out);
    auto ah = _any(hes, size, nbatch, offset_t(CCs), stride_hes);
    const offset_t nvox = ao.template size_front<-1>();
    const bool have_w = (wgt != nullptr);
    auto aw = _any(have_w ? wgt : out, size, nbatch, offset_t(C), have_w ? stride_wgt : stride_out);
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = start; i < end; ++i) {
        auto o = ao.template peel_front_at<-1>(i).recast(tny::shape<Cs>{});
        auto h = ah.template peel_front_at<-1>(i).recast(tny::shape<CCs>{});
        if (have_w) sym::solve_(o, h, aw.template peel_front_at<-1>(i).recast(tny::shape<Cs>{}));
        else        sym::solve_(o, h);
    }});
}

template <typename reduce_t, typename scalar_t, typename offset_t>
void sym_solve_(offset_t nbatch, offset_t nchannel, scalar_t* out, const scalar_t* hes,
                const scalar_t* wgt, const offset_t* size,
                const offset_t* stride_out, const offset_t* stride_hes, const offset_t* stride_wgt)
{
    const offset_t CC = nchannel * (nchannel + 1) / 2;
    auto ao = _any(out, size, nbatch, nchannel, stride_out);
    auto ah = _any(hes, size, nbatch, CC,       stride_hes);
    const offset_t nvox = ao.template size_front<-1>();
    const bool have_w = (wgt != nullptr);
    auto aw = _any(have_w ? wgt : out, size, nbatch, nchannel, have_w ? stride_wgt : stride_out);
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    std::vector<reduce_t> b(nchannel * nchannel);
    auto M = tny::wrap(b.data(), tny::shape<-1,-1>{nchannel, nchannel});
    for (offset_t i = start; i < end; ++i) {
        auto o = ao.template peel_front_at<-1>(i);
        auto h = ah.template peel_front_at<-1>(i);
        if (have_w) sym::solve_w_(o, h, aw.template peel_front_at<-1>(i), M);
        else        sym::solve_w_(o, h, M);
    }});
}

// ---- invert: out = inv(H) (out-of-place) ----------------------------------
template <int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_invert_tpl(offset_t nbatch, scalar_t* out, const scalar_t* hes,
                const offset_t* size, const offset_t* stride_out, const offset_t* stride_hes)
{
    constexpr long CCs = static_cast<long>(C) * (C + 1) / 2;
    auto ao = _any(out, size, nbatch, offset_t(CCs), stride_out);
    auto ah = _any(hes, size, nbatch, offset_t(CCs), stride_hes);
    const offset_t nvox = ao.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = start; i < end; ++i) {
        auto o = ao.template peel_front_at<-1>(i).recast(tny::shape<CCs>{});
        auto h = ah.template peel_front_at<-1>(i).recast(tny::shape<CCs>{});
        sym::invert(o, h);
    }});
}

template <typename reduce_t, typename scalar_t, typename offset_t>
void sym_invert(offset_t nbatch, offset_t nchannel, scalar_t* out, const scalar_t* hes,
                const offset_t* size, const offset_t* stride_out, const offset_t* stride_hes)
{
    const offset_t CC = nchannel * (nchannel + 1) / 2;
    auto ao = _any(out, size, nbatch, CC, stride_out);
    auto ah = _any(hes, size, nbatch, CC, stride_hes);
    const offset_t nvox = ao.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
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
template <int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_invert_tpl_(offset_t nbatch, scalar_t* hes, const offset_t* size, const offset_t* stride)
{
    constexpr long CCs = static_cast<long>(C) * (C + 1) / 2;
    auto ah = _any(hes, size, nbatch, offset_t(CCs), stride);
    const offset_t nvox = ah.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = start; i < end; ++i) {
        auto h = ah.template peel_front_at<-1>(i).recast(tny::shape<CCs>{});
        sym::invert_(h);
    }});
}

template <typename reduce_t, typename scalar_t, typename offset_t>
void sym_invert_(offset_t nbatch, offset_t nchannel, scalar_t* hes,
                const offset_t* size, const offset_t* stride)
{
    const offset_t CC = nchannel * (nchannel + 1) / 2;
    auto ah = _any(hes, size, nbatch, CC, stride);
    const offset_t nvox = ah.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
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
