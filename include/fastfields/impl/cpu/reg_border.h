#pragma once
#include <vector>
#include <fastfields/core/cuda_switch.h>
#include <fastfields/impl/kernels/bounds.h>
#include <fastfields/impl/kernels/utils.h>
#include <fastfields/impl/kernels/batch.h>
#include <fastfields/impl/kernels/parallel.h>
#include <fastfields/impl/kernels/posdef.h>
#include <fastfields/impl/kernels/regularisers/border.h>
#include <fastfields/impl/kernels/regularisers/field/utils.h>

/***********************************************************************
 *
 *  CPU side of the definition-F border pass (see
 *  kernels/regularisers/border.h for the convention).
 *
 *  `Partition` splits the (*batch, *spatial) lattice once into an interior
 *  box and disjoint border slabs, from a per-axis slab thickness. The impl
 *  loops run their unchanged stencil body over the interior box -- the body
 *  receives the same flat voxel index it always did, so nothing inside it
 *  changes and no voxel is tested -- and then one of the `*_border` passes
 *  below over the slabs. With every thickness zero the interior box is the
 *  whole lattice and there are no slabs: the loop is the one it replaced.
 *
 **********************************************************************/

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_border)

// One box of the flat (*batch, *spatial) index space, dimension 0 fastest --
// the order `index2offset{,_v2}` decompose a flat index in.
template <typename offset_t>
struct Box {
    std::vector<offset_t> lo, ext, mul;
    offset_t numel;

    // box-local flat index -> lattice flat index
    inline offset_t operator()(offset_t j) const
    {
        offset_t i = 0;
        for (size_t d = 0; d < ext.size(); ++d) {
            offset_t x = j % ext[d];
            j /= ext[d];
            i += (lo[d] + x) * mul[d];
        }
        return i;
    }
};

template <int ndim, typename offset_t>
struct Partition {
    Box<offset_t>              interior;
    std::vector<Box<offset_t>> border;

    // size: [*batch, *spatial, ...]; T: [ndim] slab thickness per axis
    Partition(offset_t nbatch, const offset_t * size, const offset_t * T)
    {
        const offset_t nall = nbatch + ndim;
        std::vector<offset_t> mul(nall), full_lo(nall, 0), full_hi(nall),
                              in_lo(nall, 0), in_hi(nall);
        offset_t acc = 1;
        for (offset_t d = 0; d < nall; ++d) {
            mul[d] = acc; acc *= size[d];
            full_hi[d] = in_hi[d] = size[d];
        }
        for (int d = 0; d < ndim; ++d) {
            offset_t n = size[nbatch + d], t = T[d];
            if (t <= 0) continue;
            in_lo[nbatch + d] = t < n ? t : n;
            in_hi[nbatch + d] = n - t > in_lo[nbatch + d] ? n - t : in_lo[nbatch + d];
        }
        interior = make(in_lo, in_hi, mul);
        for (int d = 0; d < ndim; ++d) {
            offset_t n = size[nbatch + d], t = T[d];
            if (t <= 0) continue;
            std::vector<offset_t> lo(nall), hi(nall);
            for (offset_t e = 0; e < nall; ++e) {
                bool inner = e >= nbatch && e < nbatch + d;
                lo[e] = inner ? in_lo[e] : full_lo[e];
                hi[e] = inner ? in_hi[e] : full_hi[e];
            }
            offset_t a = t < n ? t : n;
            offset_t b = n - t > a ? n - t : a;
            lo[nbatch + d] = 0; hi[nbatch + d] = a;
            push(lo, hi, mul);
            lo[nbatch + d] = b; hi[nbatch + d] = n;
            push(lo, hi, mul);
        }
    }

private:
    static Box<offset_t> make(const std::vector<offset_t> & lo,
                              const std::vector<offset_t> & hi,
                              const std::vector<offset_t> & mul)
    {
        Box<offset_t> b;
        b.lo = lo; b.mul = mul; b.ext.resize(lo.size());
        b.numel = 1;
        for (size_t d = 0; d < lo.size(); ++d) {
            b.ext[d] = hi[d] > lo[d] ? hi[d] - lo[d] : 0;
            b.numel *= b.ext[d];
        }
        return b;
    }
    void push(const std::vector<offset_t> & lo, const std::vector<offset_t> & hi,
              const std::vector<offset_t> & mul)
    {
        Box<offset_t> b = make(lo, hi, mul);
        if (b.numel > 0) border.push_back(b);
    }
};

// Run `chunk(start, end, box)` over a box, in parallel.
template <typename offset_t, class F>
inline void for_box(const Box<offset_t> & box, const F & chunk)
{
    parallel_for(0, box.numel, GRAIN_SIZE, [&](long start, long end) {
        chunk(static_cast<offset_t>(start), static_cast<offset_t>(end), box);
    });
}

/// Per-axis slab thickness (see `thickness` in kernels/regularisers/border.h).
template <int ndim, typename offset_t>
inline void thicknesses(const bound::BoundVec & bnd, const offset_t * spatial,
                        int order, bool lame, bool diag, offset_t * T)
{
    for (int d = 0; d < ndim; ++d)
        T[d] = thickness<offset_t>(bnd[d], spatial[d], order, lame, diag);
}

template <char op, typename scalar_t, typename reduce_t>
inline void apply(scalar_t & out, reduce_t v)
{
    if (op == '+')      out = static_cast<scalar_t>(static_cast<reduce_t>(out) + v);
    else if (op == '-') out = static_cast<scalar_t>(static_cast<reduce_t>(out) - v);
    else                out = static_cast<scalar_t>(v);
}

template <int ndim, typename offset_t>
inline bool colour(int patch, const offset_t * loc, offset_t n)
{
    using namespace reg_field;
    return patch == 1 ? patch1<ndim, offset_t>(loc, n)
         : patch == 2 ? patch2<ndim, offset_t>(loc, n)
         :              patch3<ndim, offset_t>(loc, n);
}

template <int ndim, typename offset_t>
inline offset_t spatial_offset(const offset_t * loc, const offset_t * stride)
{
    offset_t o = 0;
    for (int d = 0; d < ndim; ++d) o += loc[d] * stride[d];
    return o;
}

//----------------------------------------------------------------------
//                              FIELDS
//----------------------------------------------------------------------

// Penalties and weights of a field energy, as the border pass needs them.
template <typename reduce_t, typename scalar_t, typename offset_t>
struct FieldEnergy {
    const reduce_t * vx;
    const reduce_t * a;
    const reduce_t * m;
    const reduce_t * b;            // nullptr: no bending
    const scalar_t * wgt;          // nullptr: unweighted
    const offset_t * stride_wgt;   // [*batch, *spatial, C]
    bool             joint;        // one weight for all channels (JRLS)
};

/// out op= A_F inp on the border slabs
template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t>
void field_matvec_border(
    const bound::BoundVec & bnd, const Partition<ndim, offset_t> & P,
    offset_t nbatch, const offset_t * size,
    scalar_t * out, const offset_t * stride_out,
    const scalar_t * inp, const offset_t * stride_inp,
    const FieldEnergy<reduce_t, scalar_t, offset_t> & E)
{
    const offset_t nall = nbatch + ndim, nc = size[nall];
    const offset_t osc = stride_out[nall], isc = stride_inp[nall];
    const offset_t wsc = (E.wgt && !E.joint) ? E.stride_wgt[nall] : 0;
    for (const Box<offset_t> & box : P.border)
    for_box(box, [&](offset_t start, offset_t end, const Box<offset_t> & bx) {
        std::vector<reduce_t> mv(static_cast<size_t>(nc));
        offset_t loc[ndim];
        for (offset_t j = start; j < end; ++j) {
            offset_t i = bx(j);
            offset_t io = index2offset_v2<ndim>(i, nall, size, stride_inp, loc);
            offset_t oo = index2offset(i, nall, size, stride_out);
            const scalar_t * in0 = inp + (io - spatial_offset<ndim>(loc, stride_inp + nbatch));
            const scalar_t * w0  = nullptr;
            if (E.wgt) {
                offset_t wo = index2offset(i, nall, size, E.stride_wgt);
                w0 = E.wgt + (wo - spatial_offset<ndim>(loc, E.stride_wgt + nbatch));
            }
            field_at<ndim, reduce_t, scalar_t, offset_t>(
                bnd, loc, size + nbatch, in0, stride_inp + nbatch, isc,
                w0, E.wgt ? E.stride_wgt + nbatch : nullptr, wsc,
                nc, E.vx, E.a, E.m, E.b, mv.data(), nullptr);
            for (offset_t c = 0; c < nc; ++c)
                apply<op>(out[oo + osc * c], mv[c]);
        }
    });
}

/// out op= diag(A_F) on the border slabs
template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t>
void field_diag_border(
    const bound::BoundVec & bnd, const Partition<ndim, offset_t> & P,
    offset_t nbatch, const offset_t * size,
    scalar_t * out, const offset_t * stride_out,
    const FieldEnergy<reduce_t, scalar_t, offset_t> & E)
{
    const offset_t nall = nbatch + ndim, nc = size[nall];
    const offset_t osc = stride_out[nall];
    const offset_t wsc = (E.wgt && !E.joint) ? E.stride_wgt[nall] : 0;
    for (const Box<offset_t> & box : P.border)
    for_box(box, [&](offset_t start, offset_t end, const Box<offset_t> & bx) {
        std::vector<reduce_t> dg(static_cast<size_t>(nc));
        offset_t loc[ndim];
        for (offset_t j = start; j < end; ++j) {
            offset_t i = bx(j);
            offset_t oo = index2offset_v2<ndim>(i, nall, size, stride_out, loc);
            const scalar_t * w0 = nullptr;
            if (E.wgt) {
                offset_t wo = index2offset(i, nall, size, E.stride_wgt);
                w0 = E.wgt + (wo - spatial_offset<ndim>(loc, E.stride_wgt + nbatch));
            }
            field_at<ndim, reduce_t, scalar_t, offset_t>(
                bnd, loc, size + nbatch, static_cast<const scalar_t *>(nullptr),
                stride_out + nbatch, static_cast<offset_t>(0),
                w0, E.wgt ? E.stride_wgt + nbatch : nullptr, wsc,
                nc, E.vx, E.a, E.m, E.b, nullptr, dg.data());
            for (offset_t c = 0; c < nc; ++c)
                apply<op>(out[oo + osc * c], dg[c]);
        }
    });
}

/// One colour of relaxation on the border slabs: sol += (H + diag A_F) \ (g - (H + A_F) sol)
template <int ndim, typename reduce_t, typename scalar_t, typename offset_t>
void field_relax_border(
    const bound::BoundVec & bnd, const Partition<ndim, offset_t> & P,
    int patch, offset_t n,
    offset_t nbatch, const offset_t * size,
    scalar_t * sol, const offset_t * stride_sol,
    const scalar_t * hes, const offset_t * stride_hes,
    const scalar_t * grd, const offset_t * stride_grd,
    const FieldEnergy<reduce_t, scalar_t, offset_t> & E)
{
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    const offset_t nall = nbatch + ndim, nc = size[nall];
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall], gsc = stride_grd[nall];
    const offset_t wsc = (E.wgt && !E.joint) ? E.stride_wgt[nall] : 0;
    const offset_t ncc = PosDef::work_size(nc);
    for (const Box<offset_t> & box : P.border)
    for_box(box, [&](offset_t start, offset_t end, const Box<offset_t> & bx) {
        std::vector<reduce_t> mv(static_cast<size_t>(nc)), dg(static_cast<size_t>(nc));
        std::vector<scalar_t> val(static_cast<size_t>(nc)), diag(static_cast<size_t>(nc));
        std::vector<scalar_t> buf(static_cast<size_t>(ncc ? ncc : 1));
        offset_t loc[ndim];
        for (offset_t j = start; j < end; ++j) {
            offset_t i = bx(j);
            offset_t so = index2offset_v2<ndim>(i, nall, size, stride_sol, loc);
            if (!colour<ndim>(patch, loc, n)) continue;
            offset_t go = index2offset(i, nall, size, stride_grd);
            offset_t ho = index2offset(i, nall, size, stride_hes);
            const scalar_t * s0 = sol + (so - spatial_offset<ndim>(loc, stride_sol + nbatch));
            const scalar_t * w0 = nullptr;
            if (E.wgt) {
                offset_t wo = index2offset(i, nall, size, E.stride_wgt);
                w0 = E.wgt + (wo - spatial_offset<ndim>(loc, E.stride_wgt + nbatch));
            }
            field_at<ndim, reduce_t, scalar_t, offset_t>(
                bnd, loc, size + nbatch, s0, stride_sol + nbatch, osc,
                w0, E.wgt ? E.stride_wgt + nbatch : nullptr, wsc,
                nc, E.vx, E.a, E.m, E.b, mv.data(), dg.data());
            for (offset_t c = 0; c < nc; ++c) {
                val[c]  = static_cast<scalar_t>(static_cast<reduce_t>(grd[go + gsc * c]) - mv[c]);
                diag[c] = static_cast<scalar_t>(dg[c]);
            }
            PosDef::relax_(
                nc, Strided(sol + so, osc), StridedConst(hes + ho, hsc),
                val.data(), diag.data(), ncc ? buf.data() : nullptr,
                static_cast<reduce_t>(0));
        }
    });
}

//----------------------------------------------------------------------
//                              FLOWS
//----------------------------------------------------------------------

template <typename reduce_t, typename scalar_t, typename offset_t>
struct FlowEnergy {
    const reduce_t * vx;
    reduce_t a, m, b, sh, dv;
    const scalar_t * wgt;          // nullptr: unweighted (JRLS otherwise)
    const offset_t * stride_wgt;
};

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t>
void flow_matvec_border(
    const bound::BoundVec & bnd, const Partition<ndim, offset_t> & P,
    offset_t nbatch, const offset_t * size,
    scalar_t * out, const offset_t * stride_out,
    const scalar_t * inp, const offset_t * stride_inp,
    const FlowEnergy<reduce_t, scalar_t, offset_t> & E)
{
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_out[nall], isc = stride_inp[nall];
    for (const Box<offset_t> & box : P.border)
    for_box(box, [&](offset_t start, offset_t end, const Box<offset_t> & bx) {
        reduce_t mv[3];
        offset_t loc[ndim];
        for (offset_t j = start; j < end; ++j) {
            offset_t i = bx(j);
            offset_t io = index2offset_v2<ndim>(i, nall, size, stride_inp, loc);
            offset_t oo = index2offset(i, nall, size, stride_out);
            const scalar_t * in0 = inp + (io - spatial_offset<ndim>(loc, stride_inp + nbatch));
            const scalar_t * w0  = nullptr;
            if (E.wgt) {
                offset_t wo = index2offset(i, nall, size, E.stride_wgt);
                w0 = E.wgt + (wo - spatial_offset<ndim>(loc, E.stride_wgt + nbatch));
            }
            flow_at<ndim, reduce_t, scalar_t, offset_t>(
                bnd, loc, size + nbatch, in0, stride_inp + nbatch, isc,
                w0, E.wgt ? E.stride_wgt + nbatch : nullptr,
                E.vx, E.a, E.m, E.b, E.sh, E.dv, mv, nullptr);
            for (int c = 0; c < ndim; ++c)
                apply<op>(out[oo + osc * c], mv[c]);
        }
    });
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t>
void flow_diag_border(
    const bound::BoundVec & bnd, const Partition<ndim, offset_t> & P,
    offset_t nbatch, const offset_t * size,
    scalar_t * out, const offset_t * stride_out,
    const FlowEnergy<reduce_t, scalar_t, offset_t> & E)
{
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_out[nall];
    for (const Box<offset_t> & box : P.border)
    for_box(box, [&](offset_t start, offset_t end, const Box<offset_t> & bx) {
        reduce_t dg[3];
        offset_t loc[ndim];
        for (offset_t j = start; j < end; ++j) {
            offset_t i = bx(j);
            offset_t oo = index2offset_v2<ndim>(i, nall, size, stride_out, loc);
            const scalar_t * w0 = nullptr;
            if (E.wgt) {
                offset_t wo = index2offset(i, nall, size, E.stride_wgt);
                w0 = E.wgt + (wo - spatial_offset<ndim>(loc, E.stride_wgt + nbatch));
            }
            flow_at<ndim, reduce_t, scalar_t, offset_t>(
                bnd, loc, size + nbatch, static_cast<const scalar_t *>(nullptr),
                stride_out + nbatch, static_cast<offset_t>(0),
                w0, E.wgt ? E.stride_wgt + nbatch : nullptr,
                E.vx, E.a, E.m, E.b, E.sh, E.dv, nullptr, dg);
            for (int c = 0; c < ndim; ++c)
                apply<op>(out[oo + osc * c], dg[c]);
        }
    });
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t>
void flow_relax_border(
    const bound::BoundVec & bnd, const Partition<ndim, offset_t> & P,
    int patch, offset_t n,
    offset_t nbatch, const offset_t * size,
    scalar_t * sol, const offset_t * stride_sol,
    const scalar_t * hes, const offset_t * stride_hes,
    const scalar_t * grd, const offset_t * stride_grd,
    const FlowEnergy<reduce_t, scalar_t, offset_t> & E)
{
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    constexpr int DD = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall], gsc = stride_grd[nall];
    for (const Box<offset_t> & box : P.border)
    for_box(box, [&](offset_t start, offset_t end, const Box<offset_t> & bx) {
        reduce_t mv[3], dg[3];
        scalar_t val[ndim], diag[ndim];
        reduce_t buf[DD ? DD : 1];
        offset_t loc[ndim];
        for (offset_t j = start; j < end; ++j) {
            offset_t i = bx(j);
            offset_t so = index2offset_v2<ndim>(i, nall, size, stride_sol, loc);
            if (!colour<ndim>(patch, loc, n)) continue;
            offset_t go = index2offset(i, nall, size, stride_grd);
            offset_t ho = index2offset(i, nall, size, stride_hes);
            const scalar_t * s0 = sol + (so - spatial_offset<ndim>(loc, stride_sol + nbatch));
            const scalar_t * w0 = nullptr;
            if (E.wgt) {
                offset_t wo = index2offset(i, nall, size, E.stride_wgt);
                w0 = E.wgt + (wo - spatial_offset<ndim>(loc, E.stride_wgt + nbatch));
            }
            flow_at<ndim, reduce_t, scalar_t, offset_t>(
                bnd, loc, size + nbatch, s0, stride_sol + nbatch, osc,
                w0, E.wgt ? E.stride_wgt + nbatch : nullptr,
                E.vx, E.a, E.m, E.b, E.sh, E.dv, mv, dg);
            for (int c = 0; c < ndim; ++c) {
                val[c]  = static_cast<scalar_t>(static_cast<reduce_t>(grd[go + gsc * c]) - mv[c]);
                diag[c] = static_cast<scalar_t>(dg[c]);
            }
            PosDef::relax_(
                Strided(sol + so, osc), StridedConst(hes + ho, hsc),
                val, diag, buf, static_cast<reduce_t>(0));
        }
    });
}

FF_NAMESPACE_END(reg_border)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
