#pragma once
#include <fastfields/core/cuda_switch.h>
#include "../bounds.h"
#include "../utils.h"

/***********************************************************************
 *
 *   Regulariser borders: definition F, evaluated row by row
 *
 *   Every quadratic regulariser here is a sum of squared rows,
 *
 *       E(v) = sum_r  c_r w_r (a_r . E v)^2 ,
 *
 *   where E extends v beyond the field of view (FOV) through the boundary
 *   condition, a_r is one finite difference, c_r its energy coefficient and
 *   w_r its RLS weight. **Definition F** keeps exactly the rows *centred on a
 *   FOV voxel*: their taps may read extended (mirrored / ghost) samples, but
 *   there are no ghost rows and no half-weighted border rows. Its operator
 *
 *       A_F = E^T D^T F^T W F D E
 *
 *   is symmetric positive semi-definite for every boundary condition, mixed
 *   per-axis conditions included.
 *
 *   The interior stencils (field/{1,2,3}d.h, flow/{1,2,3}d.h) evaluate
 *   R.M.E -- the stencil applied to the extended field and read back on the
 *   FOV -- which equals A_F away from the borders but not on them for dct1,
 *   dst1, zero, replicate (bending) and the Lamé cross terms; and their
 *   diagonals are the stencil's centre tap, which is not the operator diagonal
 *   wherever a neighbour folds back onto the voxel itself. So the impl loops
 *   split the lattice once into an interior box, which runs the stencils
 *   unchanged, and border slabs, which run the functions below: they rebuild
 *   (A_F v)[p] and A_F[p,p] at one voxel p by enumerating the rows that touch
 *   p. That is exact by construction (no closed-form special cases to get
 *   wrong per bound), costs nothing in the interior, and gives relaxation the
 *   exact border diagonal, so its fixed point is the F system and H + A_F
 *   stays SPD. The slab thickness per axis is `thickness()` below.
 *
 *   Which rows touch p: every row tap lies within one sample of the row's
 *   centre on every axis, so along an axis of size n a row centred at q
 *   reaches p either directly (q in {p-1, p, p+1}) or through the image of
 *   one ghost sample (-1 when q == 0, n when q == n-1). `make_axis` collects
 *   those centres once per axis; a row on the axis set S is enumerated over
 *   the product of the candidate centres of S (p itself on the other axes).
 *
 *   The rows (vx = voxel size; s, t in {forward, backward}; flow component k
 *   scaled by vx_k^2 in absolute / membrane / bending):
 *     absolute  w_p v_p^2                                  (bound-independent)
 *     membrane  1/2 w_q (D_a^s v)_q^2 / vx_a^2
 *     bending   wb_q (C2_a v)_q^2 / vx_a^4,  wb = [1 2 1]/4 of w along a
 *               1/2 wc_q (D_a^s D_b^t v)_q^2 / (vx_a vx_b)^2, wc = mean of w
 *               over the four corners of the row
 *     shears    (D_i^s u_i)^2  and
 *               1/4 ((vx_i/vx_j) D_j^s u_i + (vx_j/vx_i) D_i^t u_j)^2
 *     div       2^-D (sum_i D_i^{s_i} u_i)^2
 *   which are exactly the rows the interior stencils implement (checked
 *   densely by tests/lib-cpu/test_reg_borders.cpp). RLS weights are read
 *   through the even counterpart of the bound (`bound::even_counterpart`).
 *
 *   Everything here is FF_CUDEV, allocation-free and uses runtime bounds
 *   (`bound::dyn<Dynamic>`): the border pass is not the hot path, and one
 *   instantiation per (D, dtype, index width) keeps the CUDA build small.
 *
 **********************************************************************/

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_border)

static const int MAXQ   = 5;   // candidate row centres per axis
static const int MAXTAP = 8;   // taps per row (div in 3D has 6)

template <typename offset_t>
struct AxisTable {
    int      nq;                // number of candidate centres; q[0] == p
    offset_t q   [MAXQ];
    offset_t idx [MAXQ][3];     // E index of q + k, k in {-1, 0, 1}
    int8_t   sgn [MAXQ][3];     // E sign  of q + k
    int8_t   hit [MAXQ][3];     // sgn if E(q + k) == p, else 0
    offset_t widx[MAXQ][3];     // even-counterpart index of q + k (weights)
};

template <typename offset_t>
FF_CUDEV inline void axis_push(AxisTable<offset_t> & T, offset_t q, offset_t n)
{
    if (q < 0 || q >= n) return;
    for (int j = 0; j < T.nq; ++j) if (T.q[j] == q) return;
    T.q[T.nq++] = q;
}

template <typename offset_t>
FF_CUDEV inline void make_axis(AxisTable<offset_t> & T, offset_t p, offset_t n,
                               bound::type b)
{
    bound::dyn<bound::type::Dynamic> B(b), W(bound::even_counterpart(b));
    T.nq = 0;
    axis_push(T, p, n);
    axis_push(T, p - 1, n);
    axis_push(T, p + 1, n);
    const offset_t lo = -1, hi = n;
    if (B.sign(lo, n) && B.index(lo, n) == p) axis_push(T, static_cast<offset_t>(0), n);
    if (B.sign(hi, n) && B.index(hi, n) == p) axis_push(T, n - 1, n);
    for (int j = 0; j < T.nq; ++j)
    for (int k = 0; k < 3; ++k) {
        offset_t t = T.q[j] + static_cast<offset_t>(k - 1);
        int8_t   s = B.sign(t, n);
        offset_t i = s ? B.index(t, n) : T.q[j];
        T.sgn[j][k]  = s;
        T.idx[j][k]  = i;
        T.hit[j][k]  = (s && i == p) ? s : static_cast<int8_t>(0);
        T.widx[j][k] = W.index(t, n);
    }
}

template <typename reduce_t>
struct Tap { int8_t k[3]; int8_t comp; reduce_t c; };

// Shared state of one voxel's evaluation.
template <int D, typename reduce_t, typename scalar_t, typename offset_t>
struct Ctx {
    AxisTable<offset_t> ax[D];
    const offset_t * istride;
    const scalar_t * wgt;          // weight map at the origin, or nullptr
    const offset_t * wstride;
};

// Accumulate one row centred at candidate centre `jq`:
//   mv[c] += cw * alpha_c * (a_r . E v),   dg[c] += cw * alpha_c^2
// where alpha_c is the coefficient of (p, component c) in the row. `inp`
// points at the origin (voxel 0) of component 0; components are `isc` apart.
template <int D, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUDEV inline void row(
    const Ctx<D, reduce_t, scalar_t, offset_t> & X, const int * jq,
    const Tap<reduce_t> * tp, int ntap, reduce_t cw,
    const scalar_t * inp, offset_t isc, reduce_t * mv, reduce_t * dg)
{
    reduce_t al[3] = {0, 0, 0};
    bool any = false;
    for (int t = 0; t < ntap; ++t) {
        int h = 1;
        for (int d = 0; d < D; ++d) h *= X.ax[d].hit[jq[d]][tp[t].k[d] + 1];
        if (h) { al[tp[t].comp] += tp[t].c * static_cast<reduce_t>(h); any = true; }
    }
    if (!any || cw == static_cast<reduce_t>(0)) return;
    if (mv && inp) {
        reduce_t rho = 0;
        for (int t = 0; t < ntap; ++t) {
            int s = 1; offset_t off = 0;
            for (int d = 0; d < D; ++d) {
                s   *= X.ax[d].sgn[jq[d]][tp[t].k[d] + 1];
                off += X.ax[d].idx[jq[d]][tp[t].k[d] + 1] * X.istride[d];
            }
            if (s) rho += tp[t].c * static_cast<reduce_t>(s)
                        * static_cast<reduce_t>(inp[off + isc * tp[t].comp]);
        }
        for (int c = 0; c < 3; ++c) if (al[c] != 0) mv[c] += cw * al[c] * rho;
    }
    if (dg) for (int c = 0; c < 3; ++c) dg[c] += cw * al[c] * al[c];
}

// Weight-map sample at q + k (k per axis), through the even counterpart.
template <int D, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUDEV inline reduce_t wsample(
    const Ctx<D, reduce_t, scalar_t, offset_t> & X, const scalar_t * w,
    const int * jq, const int8_t * k)
{
    offset_t off = 0;
    for (int d = 0; d < D; ++d) off += X.ax[d].widx[jq[d]][k[d] + 1] * X.wstride[d];
    return static_cast<reduce_t>(w[off]);
}

template <typename reduce_t>
FF_CUDEV inline void set_tap(Tap<reduce_t> & t, int ax, int k, int comp, reduce_t c)
{
    t.k[0] = t.k[1] = t.k[2] = 0;
    t.k[ax] = static_cast<int8_t>(k);
    t.comp = static_cast<int8_t>(comp);
    t.c = c;
}

// first difference along `ax`, direction s (0 forward, 1 backward), scaled
template <typename reduce_t>
FF_CUDEV inline int diff_taps(Tap<reduce_t> * t, int ax, int s, int comp, reduce_t c)
{
    if (s == 0) { set_tap(t[0], ax, +1, comp,  c); set_tap(t[1], ax,  0, comp, -c); }
    else        { set_tap(t[0], ax,  0, comp,  c); set_tap(t[1], ax, -1, comp, -c); }
    return 2;
}

//----------------------------------------------------------------------
//          Scalar (per-component) energies: absolute, membrane, bending
//----------------------------------------------------------------------

// Rows of one scalar component: `inp`/`w` point at that component's origin,
// `mv`/`dg` at its accumulator. `scale` multiplies every coefficient (flows:
// vx_k^2). The absolute term is handled by the caller (it reads no bound).
template <int D, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUDEV inline void scalar_rows(
    const Ctx<D, reduce_t, scalar_t, offset_t> & X,
    const scalar_t * inp, offset_t isc, const scalar_t * w,
    const reduce_t * vx, reduce_t scale, reduce_t m, reduce_t b,
    reduce_t * mv, reduce_t * dg)
{
    Tap<reduce_t> tp[MAXTAP];
    int jq[3] = {0, 0, 0};
    int8_t k0[3] = {0, 0, 0};
    if (m != 0)
    for (int a = 0; a < D; ++a)
    for (int s = 0; s < 2; ++s) {
        int nt = diff_taps(tp, a, s, 0, static_cast<reduce_t>(1));
        const reduce_t c = static_cast<reduce_t>(0.5) * m * scale / (vx[a] * vx[a]);
        for (int j = 0; j < X.ax[a].nq; ++j) {
            jq[a] = j;
            reduce_t cw = w ? c * wsample(X, w, jq, k0) : c;
            row(X, jq, tp, nt, cw, inp, isc, mv, dg);
        }
        jq[a] = 0;
    }
    if (b == 0) return;
    for (int a = 0; a < D; ++a) {
        set_tap(tp[0], a, -1, 0, static_cast<reduce_t>(1));
        set_tap(tp[1], a,  0, 0, static_cast<reduce_t>(-2));
        set_tap(tp[2], a, +1, 0, static_cast<reduce_t>(1));
        const reduce_t v2 = vx[a] * vx[a];
        const reduce_t c = b * scale / (v2 * v2);
        for (int j = 0; j < X.ax[a].nq; ++j) {
            jq[a] = j;
            reduce_t cw = c;
            if (w) {
                int8_t km[3] = {0, 0, 0}, kp[3] = {0, 0, 0};
                km[a] = -1; kp[a] = +1;
                cw *= static_cast<reduce_t>(0.25) * (wsample(X, w, jq, km) + wsample(X, w, jq, kp))
                    + static_cast<reduce_t>(0.5) * wsample(X, w, jq, k0);
            }
            row(X, jq, tp, 3, cw, inp, isc, mv, dg);
        }
        jq[a] = 0;
    }
    for (int a = 0; a < D; ++a)
    for (int e = a + 1; e < D; ++e)
    for (int s = 0; s < 2; ++s)
    for (int t = 0; t < 2; ++t) {
        // D_a^s D_e^t: product of the two first differences
        Tap<reduce_t> da[2], de[2];
        diff_taps(da, a, s, 0, static_cast<reduce_t>(1));
        diff_taps(de, e, t, 0, static_cast<reduce_t>(1));
        int nt = 0;
        for (int i = 0; i < 2; ++i) for (int l = 0; l < 2; ++l) {
            tp[nt] = da[i];
            tp[nt].k[e] = de[l].k[e];
            tp[nt].c = da[i].c * de[l].c;
            ++nt;
        }
        const reduce_t c = static_cast<reduce_t>(0.5) * b * scale
                         / (vx[a] * vx[a] * vx[e] * vx[e]);
        for (int ja = 0; ja < X.ax[a].nq; ++ja)
        for (int je = 0; je < X.ax[e].nq; ++je) {
            jq[a] = ja; jq[e] = je;
            reduce_t cw = c;
            if (w) {
                reduce_t acc = 0;
                for (int i = 0; i < nt; ++i) acc += wsample(X, w, jq, tp[i].k);
                cw *= static_cast<reduce_t>(0.25) * acc;
            }
            row(X, jq, tp, nt, cw, inp, isc, mv, dg);
        }
        jq[a] = 0; jq[e] = 0;
    }
}

template <int D, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUDEV inline void init_ctx(
    Ctx<D, reduce_t, scalar_t, offset_t> & X, const bound::BoundVec & bnd,
    const offset_t * loc, const offset_t * size, const offset_t * istride,
    const scalar_t * wgt, const offset_t * wstride)
{
    for (int d = 0; d < D; ++d) make_axis(X.ax[d], loc[d], size[d], bnd[d]);
    X.istride = istride;
    X.wgt     = wgt;
    X.wstride = wstride;
}

template <int D, typename offset_t>
FF_CUDEV inline offset_t voxel_offset(const offset_t * loc, const offset_t * stride)
{
    offset_t o = 0;
    for (int d = 0; d < D; ++d) o += loc[d] * stride[d];
    return o;
}

/**
 * Field (independent channels): (A_F v)[p, c] into mv[c] and A_F[p,p] (per
 * channel) into dg[c], for c < nc. Either output may be nullptr; `inp` may be
 * nullptr when only the diagonal is wanted.
 *
 * `inp` / `wgt` point at the origin (voxel 0) of the current batch element.
 * `wsc` is the weight map's channel stride; 0 means one weight shared by all
 * channels (JRLS); `wgt == nullptr` means unweighted.
 * `b` may be nullptr (no bending).
 */
template <int D, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUDEV inline void field_at(
    const bound::BoundVec & bnd, const offset_t * loc, const offset_t * size,
    const scalar_t * inp, const offset_t * istride, offset_t isc,
    const scalar_t * wgt, const offset_t * wstride, offset_t wsc,
    offset_t nc, const reduce_t * vx,
    const reduce_t * a, const reduce_t * m, const reduce_t * b,
    reduce_t * mv, reduce_t * dg)
{
    Ctx<D, reduce_t, scalar_t, offset_t> X;
    init_ctx(X, bnd, loc, size, istride, wgt, wstride);
    const offset_t ip = voxel_offset<D>(loc, istride);
    const offset_t wp = wgt ? voxel_offset<D>(loc, wstride) : 0;
    for (offset_t c = 0; c < nc; ++c) {
        reduce_t acc[3] = {0, 0, 0}, dac[3] = {0, 0, 0};
        const scalar_t * w = wgt ? wgt + wsc * c : nullptr;
        const scalar_t * v = inp ? inp + isc * c : nullptr;
        reduce_t ww = w ? static_cast<reduce_t>(w[wp]) : static_cast<reduce_t>(1);
        if (a[c] != 0) {
            dac[0] += a[c] * ww;
            if (v) acc[0] += a[c] * ww * static_cast<reduce_t>(v[ip]);
        }
        scalar_rows(X, v, static_cast<offset_t>(0), w, vx, static_cast<reduce_t>(1),
                    m[c], b ? b[c] : static_cast<reduce_t>(0),
                    mv ? acc : nullptr, dg ? dac : nullptr);
        if (mv) mv[c] = acc[0];
        if (dg) dg[c] = dac[0];
    }
}

/**
 * Flow (D components, coupled by the Lamé terms): (A_F u)[p, k] into mv[k]
 * and A_F[(p,k),(p,k)] into dg[k]. One weight map shared by all components
 * (JRLS) or `wgt == nullptr`.
 */
template <int D, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUDEV inline void flow_at(
    const bound::BoundVec & bnd, const offset_t * loc, const offset_t * size,
    const scalar_t * inp, const offset_t * istride, offset_t isc,
    const scalar_t * wgt, const offset_t * wstride,
    const reduce_t * vx, reduce_t a, reduce_t m, reduce_t b,
    reduce_t sh, reduce_t dv,
    reduce_t * mv, reduce_t * dg)
{
    Ctx<D, reduce_t, scalar_t, offset_t> X;
    init_ctx(X, bnd, loc, size, istride, wgt, wstride);
    const offset_t ip = voxel_offset<D>(loc, istride);
    const offset_t wp = wgt ? voxel_offset<D>(loc, wstride) : 0;
    reduce_t acc[3] = {0, 0, 0}, dac[3] = {0, 0, 0};
    reduce_t ww = wgt ? static_cast<reduce_t>(wgt[wp]) : static_cast<reduce_t>(1);
    reduce_t * pm = (mv && inp) ? acc : nullptr;
    reduce_t * pd = dg ? dac : nullptr;
    for (int k = 0; k < D; ++k) {
        const reduce_t scale = vx[k] * vx[k];
        reduce_t ak[3] = {0, 0, 0}, dk[3] = {0, 0, 0};
        const scalar_t * v = inp ? inp + isc * k : nullptr;
        if (a != 0) {
            dk[0] += a * scale * ww;
            if (v) ak[0] += a * scale * ww * static_cast<reduce_t>(v[ip]);
        }
        scalar_rows(X, v, static_cast<offset_t>(0), wgt, vx, scale, m, b,
                    pm ? ak : nullptr, pd ? dk : nullptr);
        acc[k] += ak[0]; dac[k] += dk[0];
    }
    Tap<reduce_t> tp[MAXTAP];
    int jq[3] = {0, 0, 0};
    int8_t k0[3] = {0, 0, 0};
    if (sh != 0) {
        for (int i = 0; i < D; ++i)
        for (int s = 0; s < 2; ++s) {
            int nt = diff_taps(tp, i, s, i, static_cast<reduce_t>(1));
            for (int j = 0; j < X.ax[i].nq; ++j) {
                jq[i] = j;
                reduce_t cw = wgt ? sh * wsample(X, wgt, jq, k0) : sh;
                row(X, jq, tp, nt, cw, inp, isc, pm, pd);
            }
            jq[i] = 0;
        }
        for (int i = 0; i < D; ++i)
        for (int j = i + 1; j < D; ++j)
        for (int s = 0; s < 2; ++s)
        for (int t = 0; t < 2; ++t) {
            int nt = diff_taps(tp, j, s, i, vx[i] / vx[j]);
            nt += diff_taps(tp + nt, i, t, j, vx[j] / vx[i]);
            const reduce_t c = static_cast<reduce_t>(0.25) * sh;
            for (int ji = 0; ji < X.ax[i].nq; ++ji)
            for (int jj = 0; jj < X.ax[j].nq; ++jj) {
                jq[i] = ji; jq[j] = jj;
                reduce_t cw = wgt ? c * wsample(X, wgt, jq, k0) : c;
                row(X, jq, tp, nt, cw, inp, isc, pm, pd);
            }
            jq[i] = 0; jq[j] = 0;
        }
    }
    if (dv != 0) {
        const reduce_t c = dv / static_cast<reduce_t>(1 << D);
        for (int sc = 0; sc < (1 << D); ++sc) {
            int nt = 0;
            for (int i = 0; i < D; ++i)
                nt += diff_taps(tp + nt, i, (sc >> i) & 1, i, static_cast<reduce_t>(1));
            int n0 = X.ax[0].nq;
            int n1 = D > 1 ? X.ax[D > 1 ? 1 : 0].nq : 1;
            int n2 = D > 2 ? X.ax[D > 2 ? 2 : 0].nq : 1;
            for (int j0 = 0; j0 < n0; ++j0)
            for (int j1 = 0; j1 < n1; ++j1)
            for (int j2 = 0; j2 < n2; ++j2) {
                jq[0] = j0; jq[1] = j1; jq[2] = j2;
                reduce_t cw = wgt ? c * wsample(X, wgt, jq, k0) : c;
                row(X, jq, tp, nt, cw, inp, isc, pm, pd);
            }
            jq[0] = jq[1] = jq[2] = 0;
        }
    }
    for (int k = 0; k < D; ++k) {
        if (mv) mv[k] = acc[k];
        if (dg) dg[k] = dac[k];
    }
}

//----------------------------------------------------------------------
//                     Border slab thickness
//----------------------------------------------------------------------

/**
 * Thickness of the border slab of one axis: the voxels, counted from each
 * end, whose value the border pass must produce instead of the stencil.
 *
 *   order   stencil radius of the energy: 1 (membrane, Lamé), 2 (bending)
 *   lame    the energy has Lamé cross terms
 *   diag    the caller needs the diagonal (diag, relax), not only matvec
 *
 * matvec: R.M.E == A_F already for dft, and for dct2/dst2 when there is no
 * Lamé coupling; everything else needs the slab. dct1 needs one more voxel
 * than the stencil radius for membrane/Lamé: the image of ghost sample -1 is
 * voxel 1, so the row centred on voxel 0 also reaches voxel 1.
 * diag: the stencil diagonal is exact only where no tap leaves the FOV, so
 * every non-periodic axis needs `order` voxels; a periodic axis only when it
 * is so short that a neighbour wraps onto the voxel itself.
 */
template <typename offset_t>
FF_CUHOSTDEV inline offset_t thickness(bound::type b, offset_t n, int order,
                                       bool lame, bool diag)
{
    if (b == bound::type::DFT)
        return (diag && n <= static_cast<offset_t>(order)) ? n : static_cast<offset_t>(0);
    if (!diag && !lame && (b == bound::type::DCT2 || b == bound::type::DST2))
        return 0;
    offset_t T = static_cast<offset_t>(order);
    if (b == bound::type::DCT1 && T < 2) T = 2;
    return T < n ? T : n;
}

FF_NAMESPACE_END(reg_border)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
