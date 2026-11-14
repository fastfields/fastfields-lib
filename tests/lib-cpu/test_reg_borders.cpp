// CPU tests: regularisers at the borders, against an explicit definition-F
// reference.
//
// Definition F ("FOV average"): the energy is a sum of squared rows, and only
// rows *centred on a voxel of the field of view* are kept. A row's taps may
// read extended (ghost / mirrored) values through the boundary condition E;
// there are no ghost rows and no half-weighted border rows. The operator is
//
//     A_F = E^T D^T F^T W F D E,
//
// built here densely, row by row, from nothing but the energies' definitions
// and an independent implementation of E. Every regulariser operator in
// `ff::cpu` -- matvec, diag, the relaxation fixed point, the RLS/JRLS
// variants, fields and flows -- is compared against it, for every boundary
// condition the library supports, mixed per-axis conditions, anisotropic
// voxels, and axes of size 1 and 2. See MIGRATION.md ("Regulariser borders")
// for the convention and fastfields-lib#26 for what it replaced.
//
// The energies (vx = voxel size; s, t range over {forward, backward}):
//   absolute : sum_q  w_q v_q^2
//   membrane : sum_q  sum_a sum_s  1/2 w_q (D_a^s v)_q^2 / vx_a^2
//   bending  : sum_q  sum_a  wb_q (C2_a v)_q^2 / vx_a^4
//            + sum_q  sum_{a<b} sum_{s,t} 1/2 wc_q (D_a^s D_b^t v)_q^2 / (vx_a vx_b)^2
//   shears   : sum_q  sum_i sum_s (D_i^s u_i)^2
//            + sum_q  sum_{i<j} sum_{s,t} 1/4 ((vx_i/vx_j) D_j^s u_i + (vx_j/vx_i) D_i^t u_j)^2
//   div      : sum_q  2^-D sum_{s_1..s_D} (sum_i D_i^{s_i} u_i)^2
// with flow component k scaled by vx_k^2 in absolute/membrane/bending. The RLS
// row weights are fastfields' (unchanged) interior ones: w_q for first-order
// rows, the [1 2 1]/4 average along the axis for second-order rows, and the
// mean of the four corners for cross rows -- all read through the *even*
// counterpart of the boundary condition (dst2 -> dct2, dst1/zero -> replicate).
//
// Set FF_REG_BORDERS_AUDIT=1 in the environment to print, per (energy,
// bound), which definition the library implements (F, the period average P,
// the uncorrected stencil readout R.M.E, or something else) and whether the
// operator is symmetric, instead of only pass/fail counts; and
// FF_REG_BORDERS_VERBOSE=1 for the relative distance of each problem's
// operator to F, R.M.E and P.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <fastfields/core/dlpack.h>
#include <fastfields/api/cpu/reg_field.h>
#include <fastfields/api/cpu/reg_flow.h>
#include <fastfields/impl/cpu/reg_field.h>
#include <fastfields/impl/cpu/reg_flow.h>

namespace {

using ff::bound::type;
using ff::bound::BoundVec;

enum { B_ZERO = 0, B_REPL = 1, B_DCT1 = 2, B_DCT2 = 3, B_DST1 = 4,
       B_DST2 = 5, B_DFT = 6 };
const char * bname(int b)
{
    static const char * n[] = {"zero", "replicate", "dct1", "dct2",
                               "dst1", "dst2", "dft"};
    return n[b];
}

int g_checks = 0, g_failures = 0;
bool g_audit = false;

void check(bool ok, const std::string & what)
{
    ++g_checks;
    if (!ok) {
        ++g_failures;
        if (g_failures <= 60) std::printf("  FAIL %s\n", what.c_str());
    }
}

//----------------------------------------------------------------------
//          E: the extension, written independently of bounds.h
//----------------------------------------------------------------------

int64_t pmod(int64_t a, int64_t m) { return ((a % m) + m) % m; }

// Extended sample t of an axis of n voxels -> (FOV index, sign); sign 0 means
// the extended sample is identically zero.
void ext(int64_t t, int64_t n, int b, int64_t & idx, int & sgn)
{
    sgn = 1;
    switch (b) {
    case B_DFT:  idx = pmod(t, n); return;
    case B_DCT2: { int64_t m = pmod(t, 2*n); idx = m < n ? m : 2*n-1-m; return; }
    case B_DST2: { int64_t m = pmod(t, 2*n); idx = m < n ? m : 2*n-1-m;
                   sgn = m < n ? 1 : -1; return; }
    case B_DCT1: { if (n == 1) { idx = 0; return; }
                   int64_t m = pmod(t, 2*n-2); idx = m < n ? m : 2*n-2-m; return; }
    case B_DST1: { int64_t u = pmod(t+1, 2*n+2);
                   if (u == 0 || u == n+1) { idx = 0; sgn = 0; return; }
                   if (u <= n) { idx = u-1; return; }
                   idx = 2*n+1-u; sgn = -1; return; }
    case B_ZERO: if (t < 0 || t >= n) { idx = 0; sgn = 0; } else idx = t; return;
    case B_REPL: idx = t < 0 ? 0 : t >= n ? n-1 : t; return;
    }
    idx = 0; sgn = 0;
}

// Even counterpart, used for even quantities (weights).
int even_bound(int b)
{
    if (b == B_DST2) return B_DCT2;
    if (b == B_DST1 || b == B_ZERO) return B_REPL;
    return b;
}

//----------------------------------------------------------------------
//                       Rows of the energies
//----------------------------------------------------------------------

struct Tap { int o[3]; int comp; double c; };
struct WS  { int o[3]; double f; };
struct RowT {
    double coef;              // energy coefficient of this row
    int wchan;                // weight-map channel (RLS)
    std::vector<WS>  ws;      // row weight = sum f * w~(q + o)
    std::vector<Tap> taps;
    bool absolute;            // reads no neighbour (bound-independent)
};

struct Lattice {
    int D; int64_t n[3]; int b[3]; int C;
    int64_t N() const { int64_t k = 1; for (int d = 0; d < D; ++d) k *= n[d]; return k; }
    int64_t spatial_index(const int64_t * x) const
    { int64_t k = 0; for (int d = 0; d < D; ++d) k = k * n[d] + x[d]; return k; }
};

void unit(int D, int a, int * o, int v) { for (int d = 0; d < 3; ++d) o[d] = 0; if (a < D) o[a] = v; }

// first difference along a with direction s (0 = forward, 1 = backward)
std::vector<std::pair<int, double> > dtaps(int s)
{
    std::vector<std::pair<int, double> > t;
    if (s == 0) { t.push_back({+1, 1.}); t.push_back({0, -1.}); }
    else        { t.push_back({0, 1.});  t.push_back({-1, -1.}); }
    return t;
}

RowT mkrow(double coef, int wchan) { RowT r; r.coef = coef; r.wchan = wchan; r.absolute = false; return r; }
WS ws0(double f) { WS w; w.o[0] = w.o[1] = w.o[2] = 0; w.f = f; return w; }

void add_scalar_rows(std::vector<RowT> & rows, int D, int comp, int wchan,
                     const double * vx, double scale,
                     double a, double m, double bnd)
{
    if (a != 0) {
        RowT r = mkrow(a * scale, wchan); r.absolute = true;
        r.ws.push_back(ws0(1));
        Tap t; unit(D, 0, t.o, 0); t.comp = comp; t.c = 1; r.taps.push_back(t);
        rows.push_back(r);
    }
    if (m != 0)
    for (int ax = 0; ax < D; ++ax) for (int s = 0; s < 2; ++s) {
        RowT r = mkrow(0.5 * m * scale / (vx[ax]*vx[ax]), wchan);
        r.ws.push_back(ws0(1));
        for (auto & dt : dtaps(s)) { Tap t; unit(D, ax, t.o, dt.first); t.comp = comp; t.c = dt.second; r.taps.push_back(t); }
        rows.push_back(r);
    }
    if (bnd != 0) {
        for (int ax = 0; ax < D; ++ax) {
            RowT r = mkrow(bnd * scale / std::pow(vx[ax], 4), wchan);
            int k[3] = {-1, 0, 1}; double c[3] = {1, -2, 1}, f[3] = {.25, .5, .25};
            for (int i = 0; i < 3; ++i) {
                Tap t; unit(D, ax, t.o, k[i]); t.comp = comp; t.c = c[i]; r.taps.push_back(t);
                WS w; unit(D, ax, w.o, k[i]); w.f = f[i]; r.ws.push_back(w);
            }
            rows.push_back(r);
        }
        for (int ax = 0; ax < D; ++ax) for (int bx = ax+1; bx < D; ++bx)
        for (int s = 0; s < 2; ++s) for (int u = 0; u < 2; ++u) {
            RowT r = mkrow(0.5 * bnd * scale / (vx[ax]*vx[ax]*vx[bx]*vx[bx]), wchan);
            for (auto & da : dtaps(s)) for (auto & db : dtaps(u)) {
                Tap t; unit(D, ax, t.o, da.first); t.o[bx] = db.first;
                t.comp = comp; t.c = da.second * db.second; r.taps.push_back(t);
                WS w; for (int d = 0; d < 3; ++d) w.o[d] = t.o[d]; w.f = .25; r.ws.push_back(w);
            }
            rows.push_back(r);
        }
    }
}

std::vector<RowT> field_rows(int D, int C, const double * vx,
                             const double * a, const double * m, const double * b,
                             bool joint)
{
    std::vector<RowT> rows;
    for (int c = 0; c < C; ++c)
        add_scalar_rows(rows, D, c, joint ? 0 : c, vx, 1., a[c], m[c], b ? b[c] : 0.);
    return rows;
}

std::vector<RowT> flow_rows(int D, const double * vx, double a, double m,
                            double b, double sh, double dv)
{
    std::vector<RowT> rows;
    for (int k = 0; k < D; ++k)
        add_scalar_rows(rows, D, k, 0, vx, vx[k]*vx[k], a, m, b);
    if (sh != 0) {
        for (int i = 0; i < D; ++i) for (int s = 0; s < 2; ++s) {
            RowT r = mkrow(sh, 0); r.ws.push_back(ws0(1));
            for (auto & dt : dtaps(s)) { Tap t; unit(D, i, t.o, dt.first); t.comp = i; t.c = dt.second; r.taps.push_back(t); }
            rows.push_back(r);
        }
        for (int i = 0; i < D; ++i) for (int j = i+1; j < D; ++j)
        for (int s = 0; s < 2; ++s) for (int u = 0; u < 2; ++u) {
            RowT r = mkrow(0.25 * sh, 0); r.ws.push_back(ws0(1));
            for (auto & dt : dtaps(s)) { Tap t; unit(D, j, t.o, dt.first); t.comp = i; t.c = dt.second * vx[i]/vx[j]; r.taps.push_back(t); }
            for (auto & dt : dtaps(u)) { Tap t; unit(D, i, t.o, dt.first); t.comp = j; t.c = dt.second * vx[j]/vx[i]; r.taps.push_back(t); }
            rows.push_back(r);
        }
    }
    if (dv != 0)
    for (int sc = 0; sc < (1 << D); ++sc) {
        RowT r = mkrow(dv / (1 << D), 0); r.ws.push_back(ws0(1));
        for (int i = 0; i < D; ++i)
            for (auto & dt : dtaps((sc >> i) & 1)) { Tap t; unit(D, i, t.o, dt.first); t.comp = i; t.c = dt.second; r.taps.push_back(t); }
        rows.push_back(r);
    }
    return rows;
}

//----------------------------------------------------------------------
//                       Dense operators
//----------------------------------------------------------------------

typedef std::vector<double> Mat;   // row-major (NC x NC)

// Weight map: (N, WC) row-major, or empty for unweighted.
struct Weights { std::vector<double> w; int WC = 0; };

double row_weight(const Lattice & L, const RowT & r, const int64_t * q, const Weights & W)
{
    if (W.WC == 0) return 1.;
    double acc = 0;
    for (const WS & s : r.ws) {
        int64_t x[3];
        for (int d = 0; d < L.D; ++d) {
            int sg; ext(q[d] + s.o[d], L.n[d], even_bound(L.b[d]), x[d], sg);
        }
        acc += s.f * W.w[L.spatial_index(x) * W.WC + r.wchan];
    }
    return acc;
}

// Row vector a = (taps read through E), as (dense index, value) pairs.
void row_vec(const Lattice & L, const RowT & r, const int64_t * q,
             std::vector<std::pair<int64_t, double> > & a)
{
    a.clear();
    for (const Tap & t : r.taps) {
        int64_t x[3]; int sg = 1;
        for (int d = 0; d < L.D; ++d) {
            int s; ext(q[d] + t.o[d], L.n[d], L.b[d], x[d], s); sg *= s;
        }
        if (sg == 0) continue;
        a.push_back({L.spatial_index(x) * L.C + t.comp, t.c * sg});
    }
}

enum Def { DEF_F, DEF_P, DEF_RME };

// Iterate centres q over per-axis ranges [lo_d, hi_d).
template <class F>
void for_box(int D, const int64_t * lo, const int64_t * hi, F f)
{
    int64_t q[3] = {0, 0, 0};
    for (q[0] = lo[0]; q[0] < hi[0]; ++q[0])
    for (q[1] = (D > 1 ? lo[1] : 0); q[1] < (D > 1 ? hi[1] : 1); ++q[1])
    for (q[2] = (D > 2 ? lo[2] : 0); q[2] < (D > 2 ? hi[2] : 1); ++q[2])
        f(q);
}

// Returns false if the definition is not computable here (P of an
// unbounded-energy configuration).
bool build(const Lattice & L, const std::vector<RowT> & rows, const Weights & W,
           Def def, Mat & A)
{
    const int64_t NC = L.N() * L.C;
    A.assign(NC * NC, 0.);
    std::vector<std::pair<int64_t, double> > a;
    int64_t lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
    double mult = 1.;
    const int R = 3;
    for (int d = 0; d < L.D; ++d) {
        int64_t n = L.n[d];
        if (def == DEF_F) { lo[d] = 0; hi[d] = n; }
        else if (def == DEF_RME) { lo[d] = -R; hi[d] = n + R; }
        else switch (L.b[d]) {
            case B_DFT:  lo[d] = 0; hi[d] = n; break;
            case B_DCT2: case B_DST2: lo[d] = 0; hi[d] = 2*n; mult *= 2; break;
            case B_DCT1: if (n == 1) { lo[d] = 0; hi[d] = 1; }
                         else { lo[d] = 0; hi[d] = 2*n-2; mult *= 2; } break;
            case B_DST1: lo[d] = -1; hi[d] = 2*n+1; mult *= 2; break;
            default: lo[d] = -R; hi[d] = n + R; break;   // zero, replicate
        }
    }
    bool ok = true;
    for_box(L.D, lo, hi, [&](const int64_t * q) {
        bool infov = true;
        for (int d = 0; d < L.D; ++d) infov = infov && q[d] >= 0 && q[d] < L.n[d];
        for (const RowT & r : rows) {
            if (r.absolute && !infov) continue;   // absolute never reads the bound
            row_vec(L, r, q, a);
            if (a.empty()) continue;
            double w = r.coef * row_weight(L, r, q, W) / (r.absolute ? 1. : mult);
            if (def == DEF_RME) {
                for (const Tap & t : r.taps) {
                    int64_t x[3]; bool in = true;
                    for (int d = 0; d < L.D; ++d) { x[d] = q[d] + t.o[d]; in = in && x[d] >= 0 && x[d] < L.n[d]; }
                    if (!in) continue;
                    int64_t p = L.spatial_index(x) * L.C + t.comp;
                    for (auto & e : a) A[p * NC + e.first] += w * t.c * e.second;
                }
            } else {
                if (def == DEF_P && !infov) {
                    // replicate: a row far outside reads a constant, so only
                    // rows that are identically zero there are summable.
                    bool repl = false;
                    for (int d = 0; d < L.D; ++d) repl = repl || (L.b[d] == B_REPL && (q[d] < 0 || q[d] >= L.n[d]));
                    if (repl) {
                        // a row that does not vanish on a constant would make
                        // P infinite over an unbounded replicate extension
                        std::map<int64_t, double> acc; for (auto & e : a) acc[e.first] += e.second;
                        bool zero = true; for (auto & e : acc) zero = zero && std::fabs(e.second) < 1e-12;
                        // rows this far out only read the clamped constant
                        bool far = false;
                        for (int d = 0; d < L.D; ++d)
                            far = far || (L.b[d] == B_REPL && (q[d] < -2 || q[d] > L.n[d] + 1));
                        if (!zero && far) ok = false;
                    }
                }
                for (auto & e : a) for (auto & f : a)
                    A[e.first * NC + f.first] += w * e.second * f.second;
            }
        }
    });
    return ok;
}

double maxabs(const Mat & A) { double m = 0; for (double v : A) m = std::max(m, std::fabs(v)); return m; }
double reldiff(const Mat & A, const Mat & B)
{
    double m = 0; for (size_t i = 0; i < A.size(); ++i) m = std::max(m, std::fabs(A[i] - B[i]));
    return m / std::max(1e-300, std::max(maxabs(A), maxabs(B)));
}
double asym(const Mat & A, int64_t NC)
{
    double m = 0;
    for (int64_t i = 0; i < NC; ++i) for (int64_t j = 0; j < NC; ++j)
        m = std::max(m, std::fabs(A[i*NC+j] - A[j*NC+i]));
    return m / std::max(1e-300, maxabs(A));
}

//----------------------------------------------------------------------
//          Library calls: impl layer, runtime (Dynamic) bounds
//----------------------------------------------------------------------

const type DYN = type::Dynamic;

struct Strides {
    std::vector<int64_t> size, str;
    Strides(const Lattice & L, int64_t last)
    {
        for (int d = 0; d < L.D; ++d) size.push_back(L.n[d]);
        size.push_back(last);
        str.assign(size.size(), 1);
        for (int d = (int)size.size() - 2; d >= 0; --d) str[d] = str[d+1] * size[d+1];
    }
};

struct FieldPen { std::vector<double> a, m, b; bool has_b = false, has_m = false; };
struct FlowPen  { double a = 0, m = 0, b = 0, sh = 0, dv = 0; };

BoundVec bvec_of(const Lattice & L)
{
    type t[3]; for (int d = 0; d < 3; ++d) t[d] = static_cast<type>(d < L.D ? L.b[d] : 0);
    return BoundVec(t, L.D);
}

// Each op: out = A inp (matvec), out = diag(A) (diag), relax niter.
// `wgt` null -> unweighted.
enum Op { OP_MATVEC, OP_DIAG, OP_RELAX };

template <int D, type... B>
void field_call(Op op, const Lattice & L, const FieldPen & P, const double * vx,
                double * out, const double * inp, const Weights & W,
                const double * hes, const double * grd, int niter)
{
    namespace rf = ff::cpu::reg_field;
    BoundVec bv = bvec_of(L);
    Strides S(L, L.C), SW(L, W.WC ? W.WC : 1), SH(L, L.C*(L.C+1)/2);
    const int64_t * sz = S.size.data(); const int64_t * st = S.str.data();
    const double *a = P.a.data(), *m = P.m.data(), *b = P.b.data();
    const double * w = W.w.data();
    const int64_t * sw = SW.str.data();
    const bool joint = W.WC == 1;
    if (W.WC == 0) {
        if (op == OP_MATVEC) {
            if (P.has_b) rf::matvec_bending<D,'=',double,double,int64_t,B...>(bv, 0, out, inp, sz, st, st, vx, a, m, b);
            else if (P.has_m) rf::matvec_membrane<D,'=',double,double,int64_t,B...>(bv, 0, out, inp, sz, st, st, vx, a, m);
            else rf::matvec_absolute<D,'=',double,double,int64_t,B...>(bv, 0, out, inp, sz, st, st, a);
        } else if (op == OP_DIAG) {
            if (P.has_b) rf::diag_bending<D,'=',double,double,int64_t,B...>(bv, 0, out, sz, st, vx, a, m, b);
            else if (P.has_m) rf::diag_membrane<D,'=',double,double,int64_t,B...>(bv, 0, out, sz, st, vx, a, m);
            else rf::diag_absolute<D,'=',double,double,int64_t,B...>(bv, 0, out, sz, st, a);
        } else {
            const int64_t * sh = SH.str.data();
            if (P.has_b) rf::relax_bending_<D,double,double,int64_t,B...>(bv, 0, out, hes, grd, sz, st, sh, st, vx, a, m, b, niter);
            else rf::relax_membrane_<D,double,double,int64_t,B...>(bv, 0, out, hes, grd, sz, st, sh, st, vx, a, m, niter);
        }
        return;
    }
    if (op == OP_MATVEC) {
        if (P.has_b) { if (joint) rf::matvec_bending_jrls<D,'=',double,double,int64_t,B...>(bv, 0, out, inp, w, sz, st, st, sw, vx, a, m, b);
                       else       rf::matvec_bending_rls <D,'=',double,double,int64_t,B...>(bv, 0, out, inp, w, sz, st, st, sw, vx, a, m, b); }
        else if (P.has_m) { if (joint) rf::matvec_membrane_jrls<D,'=',double,double,int64_t,B...>(bv, 0, out, inp, w, sz, st, st, sw, vx, a, m);
                            else       rf::matvec_membrane_rls <D,'=',double,double,int64_t,B...>(bv, 0, out, inp, w, sz, st, st, sw, vx, a, m); }
        else { if (joint) rf::matvec_absolute_jrls<D,'=',double,double,int64_t,B...>(bv, 0, out, inp, w, sz, st, st, sw, a);
               else       rf::matvec_absolute_rls <D,'=',double,double,int64_t,B...>(bv, 0, out, inp, w, sz, st, st, sw, a); }
    } else if (op == OP_DIAG) {
        if (P.has_b) { if (joint) rf::diag_bending_jrls<D,'=',double,double,int64_t,B...>(bv, 0, out, w, sz, st, sw, vx, a, m, b);
                       else       rf::diag_bending_rls <D,'=',double,double,int64_t,B...>(bv, 0, out, w, sz, st, sw, vx, a, m, b); }
        else if (P.has_m) { if (joint) rf::diag_membrane_jrls<D,'=',double,double,int64_t,B...>(bv, 0, out, w, sz, st, sw, vx, a, m);
                            else       rf::diag_membrane_rls <D,'=',double,double,int64_t,B...>(bv, 0, out, w, sz, st, sw, vx, a, m); }
        else { if (joint) rf::diag_absolute_jrls<D,'=',double,double,int64_t,B...>(bv, 0, out, w, sz, st, sw, a);
               else       rf::diag_absolute_rls <D,'=',double,double,int64_t,B...>(bv, 0, out, w, sz, st, sw, a); }
    } else {
        const int64_t * sh = SH.str.data();
        if (P.has_b) { if (joint) rf::relax_bending_jrls_<D,double,double,int64_t,B...>(bv, 0, out, hes, grd, w, sz, st, sh, st, sw, vx, a, m, b, niter);
                       else       rf::relax_bending_rls_ <D,double,double,int64_t,B...>(bv, 0, out, hes, grd, w, sz, st, sh, st, sw, vx, a, m, b, niter); }
        else { if (joint) rf::relax_membrane_jrls_<D,double,double,int64_t,B...>(bv, 0, out, hes, grd, w, sz, st, sh, st, sw, vx, a, m, niter);
               else       rf::relax_membrane_rls_ <D,double,double,int64_t,B...>(bv, 0, out, hes, grd, w, sz, st, sh, st, sw, vx, a, m, niter); }
    }
}

template <int D, type... B>
void flow_call(Op op, const Lattice & L, const FlowPen & P, const double * vx,
               double * out, const double * inp, const Weights & W,
               const double * hes, const double * grd, int niter)
{
    namespace rf = ff::cpu::reg_flow;
    BoundVec bv = bvec_of(L);
    Strides S(L, L.C), SW(L, 1), SH(L, L.C*(L.C+1)/2);
    const int64_t * sz = S.size.data(); const int64_t * st = S.str.data();
    const int64_t * sw = SW.str.data(); const int64_t * sh = SH.str.data();
    const double * w = W.w.data();
    const bool lame = P.sh != 0 || P.dv != 0;
    if (W.WC == 0) {
        if (op == OP_MATVEC) {
            if (lame && P.b != 0) rf::matvec_all<D,'=',double,double,int64_t,B...>(bv, 0, out, inp, sz, st, st, vx, P.a, P.m, P.b, P.sh, P.dv);
            else if (lame) rf::matvec_lame<D,'=',double,double,int64_t,B...>(bv, 0, out, inp, sz, st, st, vx, P.a, P.m, P.sh, P.dv);
            else if (P.b != 0) rf::matvec_bending<D,'=',double,double,int64_t,B...>(bv, 0, out, inp, sz, st, st, vx, P.a, P.m, P.b);
            else if (P.m != 0) rf::matvec_membrane<D,'=',double,double,int64_t,B...>(bv, 0, out, inp, sz, st, st, vx, P.a, P.m);
            else rf::matvec_absolute<D,'=',double,double,int64_t,B...>(bv, 0, out, inp, sz, st, st, vx, P.a);
        } else if (op == OP_DIAG) {
            if (lame && P.b != 0) rf::diag_all<D,'=',double,double,int64_t,B...>(bv, 0, out, sz, st, vx, P.a, P.m, P.b, P.sh, P.dv);
            else if (lame) rf::diag_lame<D,'=',double,double,int64_t,B...>(bv, 0, out, sz, st, vx, P.a, P.m, P.sh, P.dv);
            else if (P.b != 0) rf::diag_bending<D,'=',double,double,int64_t,B...>(bv, 0, out, sz, st, vx, P.a, P.m, P.b);
            else if (P.m != 0) rf::diag_membrane<D,'=',double,double,int64_t,B...>(bv, 0, out, sz, st, vx, P.a, P.m);
            else rf::diag_absolute<D,'=',double,double,int64_t,B...>(bv, 0, out, sz, st, vx, P.a);
        } else {
            if (lame && P.b != 0) rf::relax_all_<D,double,double,int64_t,B...>(bv, 0, out, hes, grd, sz, st, sh, st, vx, P.a, P.m, P.b, P.sh, P.dv, niter);
            else if (lame) rf::relax_lame_<D,double,double,int64_t,B...>(bv, 0, out, hes, grd, sz, st, sh, st, vx, P.a, P.m, P.sh, P.dv, niter);
            else if (P.b != 0) rf::relax_bending_<D,double,double,int64_t,B...>(bv, 0, out, hes, grd, sz, st, sh, st, vx, P.a, P.m, P.b, niter);
            else rf::relax_membrane_<D,double,double,int64_t,B...>(bv, 0, out, hes, grd, sz, st, sh, st, vx, P.a, P.m, niter);
        }
        return;
    }
    if (op == OP_MATVEC) {
        if (lame) rf::matvec_lame_jrls<D,'=',double,double,int64_t,B...>(bv, 0, out, inp, w, sz, st, st, sw, vx, P.a, P.m, P.sh, P.dv);
        else rf::matvec_membrane_jrls<D,'=',double,double,int64_t,B...>(bv, 0, out, inp, w, sz, st, st, sw, vx, P.a, P.m);
    } else if (op == OP_DIAG) {
        if (lame) rf::diag_lame_jrls<D,'=',double,double,int64_t,B...>(bv, 0, out, w, sz, st, sw, vx, P.a, P.m, P.sh, P.dv);
        else rf::diag_membrane_jrls<D,'=',double,double,int64_t,B...>(bv, 0, out, w, sz, st, sw, vx, P.a, P.m);
    } else {
        if (lame) rf::relax_lame_jrls_<D,double,double,int64_t,B...>(bv, 0, out, hes, grd, w, sz, st, sh, st, sw, vx, P.a, P.m, P.sh, P.dv, niter);
        else rf::relax_membrane_jrls_<D,double,double,int64_t,B...>(bv, 0, out, hes, grd, w, sz, st, sh, st, sw, vx, P.a, P.m, niter);
    }
}

struct Problem {
    bool flow; Lattice L; double vx[3]; FieldPen fp; FlowPen wp; Weights W;
    std::string name;
};

void call(Op op, const Problem & pb, double * out, const double * inp,
          const double * hes = nullptr, const double * grd = nullptr, int niter = 1)
{
    const Lattice & L = pb.L;
    if (pb.flow) {
        if (L.D == 1) flow_call<1, DYN>(op, L, pb.wp, pb.vx, out, inp, pb.W, hes, grd, niter);
        if (L.D == 2) flow_call<2, DYN, DYN>(op, L, pb.wp, pb.vx, out, inp, pb.W, hes, grd, niter);
        if (L.D == 3) flow_call<3, DYN, DYN, DYN>(op, L, pb.wp, pb.vx, out, inp, pb.W, hes, grd, niter);
    } else {
        if (L.D == 1) field_call<1, DYN>(op, L, pb.fp, pb.vx, out, inp, pb.W, hes, grd, niter);
        if (L.D == 2) field_call<2, DYN, DYN>(op, L, pb.fp, pb.vx, out, inp, pb.W, hes, grd, niter);
        if (L.D == 3) field_call<3, DYN, DYN, DYN>(op, L, pb.fp, pb.vx, out, inp, pb.W, hes, grd, niter);
    }
}

std::vector<RowT> rows_of(const Problem & pb)
{
    if (pb.flow) return flow_rows(pb.L.D, pb.vx, pb.wp.a, pb.wp.m, pb.wp.b, pb.wp.sh, pb.wp.dv);
    return field_rows(pb.L.D, pb.L.C, pb.vx, pb.fp.a.data(), pb.fp.m.data(),
                      pb.fp.has_b ? pb.fp.b.data() : nullptr, pb.W.WC == 1);
}

void library_matrix(const Problem & pb, Mat & A)
{
    const int64_t NC = pb.L.N() * pb.L.C;
    A.assign(NC * NC, 0.);
    std::vector<double> e(NC, 0.), o(NC, 0.);
    for (int64_t j = 0; j < NC; ++j) {
        std::fill(e.begin(), e.end(), 0.); e[j] = 1.;
        std::fill(o.begin(), o.end(), 0.);
        call(OP_MATVEC, pb, o.data(), e.data());
        for (int64_t i = 0; i < NC; ++i) A[i * NC + j] = o[i];
    }
}

//----------------------------------------------------------------------
//                       Problem construction
//----------------------------------------------------------------------

uint64_t g_rng = 88172645463325252ull;
double urand() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (g_rng >> 11) * (1.0 / 9007199254740992.0); }

Lattice lattice(int D, const int64_t * n, const int * b, int C)
{
    Lattice L; L.D = D; L.C = C;
    for (int d = 0; d < 3; ++d) { L.n[d] = d < D ? n[d] : 1; L.b[d] = d < D ? b[d] : 0; }
    return L;
}

// energy kinds
enum EKind { E_ABS, E_MEM, E_BEN, E_SHEARS, E_DIV, E_ALL };
const char * ekname(EKind e)
{
    static const char * n[] = {"absolute", "membrane", "bending", "shears", "div", "all"};
    return n[e];
}

Problem make_problem(bool flow, EKind e, int rls /*0 none,1 rls,2 jrls*/,
                     int D, const int64_t * n, const int * b, bool aniso)
{
    Problem pb; pb.flow = flow;
    int C = flow ? D : 2;
    pb.L = lattice(D, n, b, C);
    double vxs[3] = {1., 1., 1.};
    if (aniso) { vxs[0] = 0.8; vxs[1] = 1.3; vxs[2] = 1.1; }
    for (int d = 0; d < 3; ++d) pb.vx[d] = vxs[d];
    if (flow) {
        pb.wp.a = 0.3;
        if (e >= E_MEM) pb.wp.m = 0.7;
        if (e == E_BEN || e == E_ALL) pb.wp.b = 0.4;
        if (e == E_SHEARS || e == E_ALL) pb.wp.sh = 0.9;
        if (e == E_DIV || e == E_ALL) pb.wp.dv = 0.6;
        if (e == E_SHEARS || e == E_DIV) pb.wp.m = 0.;
    } else {
        pb.fp.a = {0.3, 0.5};
        pb.fp.m = {0., 0.};
        pb.fp.b = {0., 0.};
        if (e >= E_MEM) { pb.fp.m = {0.7, 1.1}; pb.fp.has_m = true; }
        if (e == E_BEN) { pb.fp.b = {0.4, 0.25}; pb.fp.has_b = true; }
    }
    if (rls) {
        pb.W.WC = (rls == 1 && !flow) ? C : 1;
        pb.W.w.resize(pb.L.N() * pb.W.WC);
        for (double & v : pb.W.w) v = 0.5 + urand();
    }
    char buf[256];
    std::string sizes, bnds;
    for (int d = 0; d < D; ++d) { sizes += (d ? "x" : "") + std::to_string(n[d]); bnds += std::string(d ? "/" : "") + bname(b[d]); }
    std::snprintf(buf, sizeof(buf), "%s %s%s %dD n=%s bound=%s%s", flow ? "flow" : "field",
                  ekname(e), rls == 1 ? "+rls" : rls == 2 ? "+jrls" : "", D, sizes.c_str(), bnds.c_str(),
                  aniso ? " aniso" : "");
    pb.name = buf;
    return pb;
}

//----------------------------------------------------------------------
//                       Checks on one problem
//----------------------------------------------------------------------

struct Verdict { bool F = false, P = false, RME = false, Pdef = false, sym = false;
                 bool diagF = false, relaxF = false; };

const double TOL = 1e-10;

Verdict examine(const Problem & pb, bool do_relax)
{
    Verdict v;
    const int64_t NC = pb.L.N() * pb.L.C;
    std::vector<RowT> rows = rows_of(pb);
    Mat A, AF, AP, AR;
    library_matrix(pb, A);
    build(pb.L, rows, pb.W, DEF_F, AF);
    build(pb.L, rows, pb.W, DEF_RME, AR);
    v.Pdef = build(pb.L, rows, pb.W, DEF_P, AP);
    v.sym = asym(A, NC) < TOL;
    v.F   = reldiff(A, AF) < TOL;
    v.RME = reldiff(A, AR) < TOL;
    v.P   = v.Pdef && reldiff(A, AP) < TOL;
    if (std::getenv("FF_REG_BORDERS_VERBOSE"))
        std::printf("  %-60s dF=%.2e dRME=%.2e dP=%.2e asym=%.2e\n", pb.name.c_str(),
                    reldiff(A, AF), reldiff(A, AR), v.Pdef ? reldiff(A, AP) : -1., asym(A, NC));

    // diag
    std::vector<double> dg(NC, 0.);
    call(OP_DIAG, pb, dg.data(), nullptr);
    double dmax = 0, dscale = 0;
    for (int64_t i = 0; i < NC; ++i) { dmax = std::max(dmax, std::fabs(dg[i] - AF[i*NC+i])); dscale = std::max(dscale, std::fabs(AF[i*NC+i])); }
    v.diagF = dmax <= TOL * std::max(1., dscale);

    if (do_relax) {
        // Hessian = h * I (compact symmetric: C diagonal entries then zeros);
        // the fixed point of relaxation must solve (H + A_F) x = g exactly.
        const int C = pb.L.C, K = C*(C+1)/2;
        std::vector<double> hes(pb.L.N() * K, 0.), g(NC), x(NC, 0.);
        for (int64_t i = 0; i < pb.L.N(); ++i) for (int c = 0; c < C; ++c) hes[i*K + c] = 1.0;
        for (double & t : g) t = urand() - 0.5;
        call(OP_RELAX, pb, x.data(), nullptr, hes.data(), g.data(), 400);
        double rn = 0, gn = 0;
        for (int64_t i = 0; i < NC; ++i) {
            double r = g[i] - x[i];
            for (int64_t j = 0; j < NC; ++j) r -= AF[i*NC+j] * x[j];
            rn += r*r; gn += g[i]*g[i];
        }
        v.relaxF = std::sqrt(rn / gn) < 1e-8;
    }
    return v;
}

std::string verdict_str(const Verdict & v)
{
    std::string s = v.F ? "F" : v.P ? "P" : v.RME ? "RME" : "other";
    if (v.F && v.RME) s = "F(=RME)";
    if (v.F && v.P && !v.RME) s = "F(=P)";
    return s;
}

// Audit table accumulation: (energy, bound) -> set of verdicts.
std::map<std::string, std::map<std::string, int> > g_table;

void run(const Problem & pb, bool do_relax, const std::string & key)
{
    Verdict v = examine(pb, do_relax);
    check(v.F,   pb.name + ": matvec == A_F");
    check(v.sym, pb.name + ": symmetric");
    check(v.diagF, pb.name + ": diag == diag(A_F)");
    if (do_relax) check(v.relaxF, pb.name + ": relax fixed point solves (H + A_F) x = g");
    if (g_audit) {
        std::string s = verdict_str(v) + (v.sym ? "" : ",asym") + (v.diagF ? "" : ",diag!=F")
                      + (do_relax ? (v.relaxF ? "" : ",relax!=F") : "");
        g_table[key][s]++;
    }
}

//----------------------------------------------------------------------
//    Public API: static-bound instantiations, batches, ops, float32
//----------------------------------------------------------------------

template <typename T>
DLTensor tensor(T * data, std::vector<int64_t> & shape, std::vector<int64_t> & strides)
{
    DLTensor t;
    t.data = data; t.device.device_type = kDLCPU; t.device.device_id = 0;
    t.ndim = (int32_t)shape.size();
    t.dtype.code = kDLFloat; t.dtype.bits = sizeof(T) * 8; t.dtype.lanes = 1;
    t.shape = shape.data(); t.strides = strides.data(); t.byte_offset = 0;
    return t;
}

std::vector<int64_t> cstrides(const std::vector<int64_t> & s)
{
    std::vector<int64_t> r(s.size(), 1);
    for (int d = (int)s.size() - 2; d >= 0; --d) r[d] = r[d+1] * s[d+1];
    return r;
}

// y_b = A x_b for each batch element b, with A dense (NC x NC)
std::vector<double> apply_batched(const Mat & A, const std::vector<double> & x, int64_t NC, int B)
{
    std::vector<double> y(x.size(), 0.);
    for (int b = 0; b < B; ++b)
    for (int64_t i = 0; i < NC; ++i) {
        double s = 0; for (int64_t j = 0; j < NC; ++j) s += A[i*NC+j] * x[b*NC+j];
        y[b*NC+i] = s;
    }
    return y;
}

template <typename T>
void expect(const std::vector<T> & got, const std::vector<double> & ref,
            double tol, const std::string & what)
{
    double m = 0, sc = 0;
    for (size_t i = 0; i < ref.size(); ++i) { m = std::max(m, std::fabs((double)got[i] - ref[i])); sc = std::max(sc, std::fabs(ref[i])); }
    check(m <= tol * std::max(1., sc), what);
}

// One batch element's weights must be the same for both, so the reference is
// built per batch element when weighted.
template <typename T>
void api_field(int bnd, bool bending, bool rls, double tol)
{
    const int B = 2, D = 2, C = 2;
    int64_t n[3] = {6, 5, 1}; int b3[3] = {bnd, bnd, bnd};
    Problem pb = make_problem(false, bending ? E_BEN : E_MEM, rls ? 1 : 0, D, n, b3, true);
    const int64_t N = pb.L.N(), NC = N * C;
    std::vector<int64_t> shape = {B, n[0], n[1], C}, st = cstrides(shape);
    std::vector<int64_t> wshape = {B, n[0], n[1], C}, wst = cstrides(wshape);
    std::vector<double> x(B * NC), r0(B * NC), wd(B * NC);
    for (double & v : x) v = urand() - 0.5;
    for (double & v : r0) v = urand() - 0.5;
    for (double & v : wd) v = 0.5 + urand();
    std::vector<T> xt(x.begin(), x.end()), out(B * NC), wt(wd.begin(), wd.end());
    // dense references, per batch element
    std::vector<double> ref(B * NC), refd(B * NC);
    std::vector<RowT> rows = rows_of(pb);
    for (int b = 0; b < B; ++b) {
        Weights W;
        if (rls) { W.WC = C; W.w.assign(wd.begin() + b*NC, wd.begin() + (b+1)*NC); }
        Mat A; build(pb.L, rows, W, DEF_F, A);
        for (int64_t i = 0; i < NC; ++i) {
            double s = 0; for (int64_t j = 0; j < NC; ++j) s += A[i*NC+j] * x[b*NC+j];
            ref[b*NC+i] = s; refd[b*NC+i] = A[i*NC+i];
        }
    }
    DLTensor to = tensor(out.data(), shape, st), tx = tensor(xt.data(), shape, st);
    DLTensor tw = tensor(wt.data(), wshape, wst);
    const double * a = pb.fp.a.data(), * m = pb.fp.m.data();
    const double * bb = bending ? pb.fp.b.data() : nullptr;
    std::string tag = std::string("api field ") + (bending ? "bending" : "membrane")
                    + (rls ? "+rls" : "") + (sizeof(T) == 4 ? " f32 " : " f64 ") + bname(bnd);
    if (rls) {
        ff::cpu::field_matvec_rls(to, tx, tw, pb.vx, a, m, bb, (int8_t)bnd, D);
        expect(out, ref, tol, tag + ": matvec_rls (batched)");
        ff::cpu::field_diag_rls(to, tw, pb.vx, a, m, bb, (int8_t)bnd, D);
        expect(out, refd, tol, tag + ": diag_rls (batched)");
        return;
    }
    ff::cpu::field_matvec(to, tx, pb.vx, a, m, bb, (int8_t)bnd, D);
    expect(out, ref, tol, tag + ": matvec (batched)");
    std::copy(r0.begin(), r0.end(), out.begin());
    ff::cpu::field_addmatvec_(to, tx, pb.vx, a, m, bb, (int8_t)bnd, D);
    std::vector<double> e(B * NC);
    for (int64_t i = 0; i < B * NC; ++i) e[i] = r0[i] + ref[i];
    expect(out, e, tol, tag + ": addmatvec_ (batched)");
    std::copy(r0.begin(), r0.end(), out.begin());
    ff::cpu::field_submatvec_(to, tx, pb.vx, a, m, bb, (int8_t)bnd, D);
    for (int64_t i = 0; i < B * NC; ++i) e[i] = r0[i] - ref[i];
    expect(out, e, tol, tag + ": submatvec_ (batched)");
    ff::cpu::field_diag(to, pb.vx, a, m, bb, (int8_t)bnd, D);
    expect(out, refd, tol, tag + ": diag (batched)");
    std::copy(r0.begin(), r0.end(), out.begin());
    ff::cpu::field_adddiag_(to, pb.vx, a, m, bb, (int8_t)bnd, D);
    for (int64_t i = 0; i < B * NC; ++i) e[i] = r0[i] + refd[i];
    expect(out, e, tol, tag + ": adddiag_ (batched)");
}

void api_flow(int bnd, bool jrls)
{
    const int B = 2, D = 2;
    int64_t n[3] = {5, 6, 1}; int b3[3] = {bnd, bnd, bnd};
    Problem pb = make_problem(true, jrls ? E_SHEARS : E_ALL, jrls ? 2 : 0, D, n, b3, true);
    if (jrls) pb.wp.dv = 0.6, pb.wp.m = 0.7;   // shears + div + membrane, weighted
    const int64_t N = pb.L.N(), NC = N * D;
    std::vector<int64_t> shape = {B, n[0], n[1], D}, st = cstrides(shape);
    std::vector<int64_t> wshape = {B, n[0], n[1], 1}, wst = cstrides(wshape);
    std::vector<double> x(B * NC), r0(B * NC), w(B * N), out(B * NC);
    for (double & v : x) v = urand() - 0.5;
    for (double & v : r0) v = urand() - 0.5;
    for (double & v : w) v = 0.5 + urand();
    std::vector<double> ref(B * NC), refd(B * NC);
    std::vector<RowT> rows = flow_rows(D, pb.vx, pb.wp.a, pb.wp.m, pb.wp.b, pb.wp.sh, pb.wp.dv);
    for (int b = 0; b < B; ++b) {
        Weights W;
        if (jrls) { W.WC = 1; W.w.assign(w.begin() + b*N, w.begin() + (b+1)*N); }
        Mat A; build(pb.L, rows, W, DEF_F, A);
        for (int64_t i = 0; i < NC; ++i) {
            double s = 0; for (int64_t j = 0; j < NC; ++j) s += A[i*NC+j] * x[b*NC+j];
            ref[b*NC+i] = s; refd[b*NC+i] = A[i*NC+i];
        }
    }
    DLTensor to = tensor(out.data(), shape, st), tx = tensor(x.data(), shape, st);
    DLTensor tw = tensor(w.data(), wshape, wst);
    const FlowPen & P = pb.wp;
    std::string tag = std::string("api flow ") + (jrls ? "lame+jrls " : "all ") + bname(bnd);
    if (jrls) {
        ff::cpu::flow_matvec_rls(to, tx, tw, pb.vx, P.a, P.m, 0., P.sh, P.dv, (int8_t)bnd, D);
        expect(out, ref, 1e-10, tag + ": matvec_rls (batched)");
        ff::cpu::flow_diag_rls(to, tw, pb.vx, P.a, P.m, 0., P.sh, P.dv, (int8_t)bnd, D);
        expect(out, refd, 1e-10, tag + ": diag_rls (batched)");
        return;
    }
    ff::cpu::flow_matvec(to, tx, pb.vx, P.a, P.m, P.b, P.sh, P.dv, (int8_t)bnd, D);
    expect(out, ref, 1e-10, tag + ": matvec (batched)");
    std::copy(r0.begin(), r0.end(), out.begin());
    ff::cpu::flow_submatvec_(to, tx, pb.vx, P.a, P.m, P.b, P.sh, P.dv, (int8_t)bnd, D);
    std::vector<double> e(B * NC);
    for (int64_t i = 0; i < B * NC; ++i) e[i] = r0[i] - ref[i];
    expect(out, e, 1e-10, tag + ": submatvec_ (batched)");
    ff::cpu::flow_diag(to, pb.vx, P.a, P.m, P.b, P.sh, P.dv, (int8_t)bnd, D);
    expect(out, refd, 1e-10, tag + ": diag (batched)");
}

} // namespace

int main()
{
    const char * env = std::getenv("FF_REG_BORDERS_AUDIT");
    g_audit = env && env[0] && env[0] != '0';
    std::printf("regulariser borders: definition-F reference\n");

    const int bounds[] = {B_ZERO, B_REPL, B_DCT1, B_DCT2, B_DST1, B_DST2, B_DFT};

    struct Shape { int D; int64_t n[3]; };
    const Shape shapes[] = {
        {1, {7, 1, 1}}, {1, {1, 1, 1}}, {1, {2, 1, 1}}, {1, {3, 1, 1}},
        {2, {6, 5, 1}}, {2, {2, 5, 1}}, {2, {1, 4, 1}},
        {3, {5, 4, 3}}, {3, {2, 4, 5}},
    };

    // field energies x rls kinds; flow energies x {none, jrls}
    for (int flow = 0; flow < 2; ++flow)
    for (int ek = 0; ek <= (flow ? E_ALL : E_BEN); ++ek)
    for (int rls = 0; rls <= 2; ++rls) {
        if (flow && rls == 1) continue;                    // flows: JRLS only
        if (flow && rls && (ek == E_BEN || ek == E_ALL)) continue;  // no bending jrls for flows
        if (flow && rls && ek == E_ABS) continue;          // no absolute-only jrls entry for flows
        for (int bi = 0; bi < 7; ++bi)
        for (const Shape & sh : shapes) {
            int b[3] = {bounds[bi], bounds[bi], bounds[bi]};
            bool aniso = sh.D > 1;
            Problem pb = make_problem(flow, (EKind)ek, rls, sh.D, sh.n, b, aniso);
            std::string key = std::string(flow ? "flow  " : "field ") + ekname((EKind)ek)
                            + (rls == 1 ? "+rls" : rls == 2 ? "+jrls" : "") + " | " + bname(bounds[bi]);
            // relaxation has no absolute-only entry point; 3D is left to
            // matvec/diag to keep the dense solve cheap
            run(pb, ek != E_ABS && sh.D <= 2, key);
        }
        // mixed per-axis bounds
        const int mixes[][3] = {
            {B_DCT1, B_ZERO, B_DST2}, {B_DFT, B_DCT1, B_REPL}, {B_DST1, B_DCT2, B_DFT},
            {B_ZERO, B_DST2, B_DCT1}, {B_REPL, B_DST1, B_DCT2},
        };
        for (const auto & mx : mixes) {
            int64_t n2[3] = {5, 4, 1}, n3[3] = {4, 3, 5};
            Problem p2 = make_problem(flow, (EKind)ek, rls, 2, n2, mx, true);
            Problem p3 = make_problem(flow, (EKind)ek, rls, 3, n3, mx, true);
            std::string key = std::string(flow ? "flow  " : "field ") + ekname((EKind)ek)
                            + (rls == 1 ? "+rls" : rls == 2 ? "+jrls" : "") + " | mixed";
            run(p2, false, key);
            run(p3, false, key);
        }
    }

    // public API: static-bound instantiations, a batch dimension, the
    // accumulate/subtract ops, RLS, float32
    for (int bi = 0; bi < 7; ++bi) {
        for (int ben = 0; ben < 2; ++ben) {
            api_field<double>(bounds[bi], ben, false, 1e-10);
            api_field<double>(bounds[bi], ben, true, 1e-10);
            api_field<float>(bounds[bi], ben, false, 1e-5);
        }
        api_flow(bounds[bi], false);
        api_flow(bounds[bi], true);
    }

    if (g_audit) {
        std::printf("\n%-34s | %s\n", "energy | bound", "verdicts (count)");
        for (auto & row : g_table) {
            std::printf("%-34s |", row.first.c_str());
            for (auto & v : row.second) std::printf(" %s(%d)", v.first.c_str(), v.second);
            std::printf("\n");
        }
    }
    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    std::printf(g_failures ? "FAILED\n" : "PASSED\n");
    return g_failures ? 1 : 0;
}
