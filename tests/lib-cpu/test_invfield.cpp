// CPU unit tests for the invfield module (displacement-field inversion).
//
// The op rasterises the deformed simplicial mesh of the voxel grid, so it is
// checked from three independent directions:
//
//   1. Cases with a closed-form answer. An identity field must invert to zero
//      at EVERY voxel, which is also the test that the five-tetrahedron (and
//      two-triangle) cell stencils really tile their cell: a wrong stencil
//      leaves holes, and a hole shows up as an uncovered voxel. An affine
//      field inverts to the exact affine inverse, whatever the stencil is, so
//      that one checks the barycentric interpolation against arithmetic the
//      test does itself.
//
//   2. A brute-force check of the owner/coverage contract on random smooth
//      fields: for every voxel, every simplex is tested for containment with
//      a barycentric solve the test performs by Gaussian elimination (not with
//      the kernel's formula), and the result is compared against what the op
//      reported -- that the winner contains the voxel, that it is the
//      lowest-indexed simplex that does, that the written value is that
//      simplex's interpolation, and that a voxel no simplex contains is NaN.
//
//   3. Invariance to layout: a batched, non-contiguous, 64-bit-offset call
//      must agree with per-element contiguous calls.
//
// The module is additionally cross-checked offline against the brainhops
// Python reference it was ported from; see the pull request for the numbers.
// That comparison is what validates the stencils against an outside source,
// and it cannot live here because it needs NumPy.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <fastfields/core/dlpack.h>
#include <fastfields/api/cpu/invfield.h>

namespace {

int g_failures = 0, g_checks = 0;

void check(bool ok, const char * what)
{
    ++g_checks;
    if (!ok) {
        ++g_failures;
        if (g_failures < 40) std::printf("  FAIL [%s]\n", what);
    }
}

template <typename T>
DLTensor make_cpu_tensor(T * data, std::vector<int64_t> & shape,
                         std::vector<int64_t> & strides,
                         uint8_t code, uint8_t bits)
{
    DLTensor t;
    t.data               = static_cast<void*>(data);
    t.device.device_type = kDLCPU;
    t.device.device_id   = 0;
    t.ndim               = static_cast<int32_t>(shape.size());
    t.dtype.code         = code;
    t.dtype.bits         = bits;
    t.dtype.lanes        = 1;
    t.shape              = shape.data();
    t.strides            = strides.data();
    t.byte_offset        = 0;
    return t;
}

std::vector<int64_t> cstrides(const std::vector<int64_t> & shape)
{
    std::vector<int64_t> s(shape.size());
    int64_t acc = 1;
    for (int64_t d = (int64_t)shape.size() - 1; d >= 0; --d) {
        s[d] = acc;
        acc *= shape[d];
    }
    return s;
}

/* ------------------------------------------------------------------ */
/*  A self-contained harness: build the three tensors and run the op  */
/* ------------------------------------------------------------------ */

template <typename T>
struct Case
{
    int                  ndim;
    std::vector<int64_t> spatial;
    int64_t              nvox;
    std::vector<T>       disp, out;
    std::vector<int64_t> owner;

    Case(int nd, const std::vector<int64_t> & sp)
        : ndim(nd), spatial(sp), nvox(1)
    {
        for (int d = 0; d < ndim; ++d) nvox *= spatial[d];
        disp.assign((size_t)(nvox * ndim), (T)0);
        out.assign((size_t)(nvox * ndim), (T)0);
        owner.assign((size_t)nvox, 0);
    }

    // Row-major voxel index from per-axis coordinates.
    int64_t vindex(const int64_t * sub) const
    {
        int64_t i = 0;
        for (int d = 0; d < ndim; ++d) i = i * spatial[d] + sub[d];
        return i;
    }

    void unravel(int64_t i, int64_t * sub) const
    {
        for (int d = ndim - 1; d >= 0; --d) {
            sub[d] = i % spatial[d];
            i /= spatial[d];
        }
    }

    void run(uint8_t bits)
    {
        std::vector<int64_t> shf = spatial; shf.push_back(ndim);
        std::vector<int64_t> stf = cstrides(shf);
        std::vector<int64_t> sho = spatial;
        std::vector<int64_t> sto = cstrides(sho);
        DLTensor td = make_cpu_tensor(disp.data(),  shf, stf, kDLFloat, bits);
        DLTensor to = make_cpu_tensor(out.data(),   shf, stf, kDLFloat, bits);
        DLTensor tw = make_cpu_tensor(owner.data(), sho, sto, kDLInt,   64);
        ff::cpu::invert_field(to, tw, td, ndim, 0);
    }
};

/* ------------------------------------------------------------------ */
/*  Reference geometry, written out independently of the kernels      */
/* ------------------------------------------------------------------ */

// Corner codes of each simplex of a cell, as bit patterns over the spatial
// axes (bit ndim-1-d is the offset along axis d). Red cells (even index sum)
// and black cells (odd) use mirror-image decompositions, which is what makes
// the mesh conforming across a shared cell face.
const int TRI_RED [2][3] = {{0, 1, 2}, {3, 1, 2}};
const int TRI_BLK [2][3] = {{1, 0, 3}, {2, 0, 3}};
const int TET_RED [5][4] = {{0, 1, 2, 4}, {3, 2, 1, 7}, {5, 4, 1, 7},
                            {6, 4, 2, 7}, {7, 1, 2, 4}};
const int TET_BLK [5][4] = {{2, 3, 0, 6}, {1, 0, 3, 5}, {4, 0, 6, 5},
                            {7, 3, 5, 6}, {0, 3, 6, 5}};

int simplices_per_cell(int ndim) { return ndim == 3 ? 5 : 2; }

// The corners (as grid coordinates) of simplex `sid`, which is
// cell * simplices_per_cell + k with the cell grid indexed row-major.
void simplex_corners(int64_t sid, int ndim, const int64_t * spatial,
                     int64_t corners[4][3])
{
    const int S = simplices_per_cell(ndim);
    const int k = (int)(sid % S);
    int64_t cell = sid / S;
    int64_t base[3] = {0, 0, 0};
    int parity = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        const int64_t n = spatial[d] - 1;
        base[d] = cell % n;
        cell /= n;
        parity += (int)base[d];
    }
    parity &= 1;
    for (int i = 0; i <= ndim; ++i) {
        const int code = ndim == 3
            ? (parity ? TET_BLK[k][i] : TET_RED[k][i])
            : (parity ? TRI_BLK[k][i] : TRI_RED[k][i]);
        for (int d = 0; d < ndim; ++d)
            corners[i][d] = base[d] + ((code >> (ndim - 1 - d)) & 1);
    }
}

// Barycentric coordinates of `p` in the simplex `v`, by solving the
// (ndim+1)-square system [v; 1] w = [p; 1] with Gaussian elimination and
// partial pivoting. Returns false if the simplex is degenerate.
bool barycentric(const double * p, const double v[4][3], int ndim,
                 double * w)
{
    const int n = ndim + 1;
    double a[4][5];
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < ndim; ++d) a[d][i] = v[i][d];
        a[ndim][i] = 1.0;
    }
    for (int d = 0; d < ndim; ++d) a[d][n] = p[d];
    a[ndim][n] = 1.0;

    for (int c = 0; c < n; ++c) {
        int piv = c;
        for (int r = c + 1; r < n; ++r)
            if (std::fabs(a[r][c]) > std::fabs(a[piv][c])) piv = r;
        if (std::fabs(a[piv][c]) < 1e-14) return false;
        for (int j = 0; j <= n; ++j) { double t = a[c][j]; a[c][j] = a[piv][j]; a[piv][j] = t; }
        for (int r = 0; r < n; ++r) {
            if (r == c) continue;
            const double f = a[r][c] / a[c][c];
            for (int j = c; j <= n; ++j) a[r][j] -= f * a[c][j];
        }
    }
    for (int i = 0; i < n; ++i) w[i] = a[i][n] / a[i][i];
    return true;
}

/* ------------------------------------------------------------------ */
/*  1. closed-form cases                                             */
/* ------------------------------------------------------------------ */

// An identity field must invert to zero, and -- the point of the test -- must
// cover every single voxel, because the simplices of a cell tile the cell and
// the cells tile the grid.
template <typename T>
void test_identity(int ndim, const std::vector<int64_t> & spatial,
                   uint8_t bits, double tol)
{
    Case<T> c(ndim, spatial);
    c.run(bits);

    int64_t uncovered = 0;
    double worst = 0.0;
    for (int64_t v = 0; v < c.nvox; ++v) {
        if (c.owner[(size_t)v] < 0) { ++uncovered; continue; }
        for (int d = 0; d < ndim; ++d)
            worst = std::max(worst, std::fabs((double)c.out[(size_t)(v*ndim+d)]));
    }
    check(uncovered == 0, "identity.full_coverage");
    if (uncovered)
        std::printf("    (%lld of %lld voxels uncovered)\n",
                    (long long)uncovered, (long long)c.nvox);
    check(worst <= tol, "identity.zero");
}

// A constant displacement is a pure translation: wherever a simplex covers the
// voxel, the inverse displacement is the negated translation.
template <typename T>
void test_translation(int ndim, const std::vector<int64_t> & spatial,
                      const double * shift, uint8_t bits, double tol)
{
    Case<T> c(ndim, spatial);
    for (int64_t v = 0; v < c.nvox; ++v)
        for (int d = 0; d < ndim; ++d)
            c.disp[(size_t)(v*ndim+d)] = (T)shift[d];
    c.run(bits);

    int64_t covered = 0;
    double worst = 0.0;
    for (int64_t v = 0; v < c.nvox; ++v) {
        if (c.owner[(size_t)v] < 0) continue;
        ++covered;
        for (int d = 0; d < ndim; ++d)
            worst = std::max(worst,
                std::fabs((double)c.out[(size_t)(v*ndim+d)] + shift[d]));
    }
    check(covered > 0, "translation.some_coverage");
    check(worst <= tol, "translation.exact");
}

bool invert_matrix(const double * m, int n, double * inv)
{
    double a[3][6] = {{0}};
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) a[r][c] = m[r*n+c];
        for (int c = 0; c < n; ++c) a[r][n+c] = (r == c) ? 1.0 : 0.0;
    }
    for (int c = 0; c < n; ++c) {
        int piv = c;
        for (int r = c+1; r < n; ++r)
            if (std::fabs(a[r][c]) > std::fabs(a[piv][c])) piv = r;
        if (std::fabs(a[piv][c]) < 1e-12) return false;
        for (int j = 0; j < 2*n; ++j) { double t=a[c][j]; a[c][j]=a[piv][j]; a[piv][j]=t; }
        const double p = a[c][c];
        for (int j = 0; j < 2*n; ++j) a[c][j] /= p;
        for (int r = 0; r < n; ++r) {
            if (r == c) continue;
            const double f = a[r][c];
            for (int j = 0; j < 2*n; ++j) a[r][j] -= f * a[c][j];
        }
    }
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) inv[r*n+c] = a[r][n+c];
    return true;
}

// An affine field's piecewise-linear interpolant IS the field, so the inverse
// is the affine inverse and the expected answer is independent of how the
// cells are cut up.
template <typename T>
void test_affine(int ndim, const std::vector<int64_t> & spatial,
                 unsigned seed, uint8_t bits, double tol)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-0.15, 0.15);

    // dst = M x + b, with M = I + A.
    double M[9] = {0}, Minv[9] = {0}, b[3] = {0};
    for (int r = 0; r < ndim; ++r) {
        for (int c = 0; c < ndim; ++c) M[r*ndim+c] = (r == c ? 1.0 : 0.0) + u(rng);
        b[r] = u(rng) * 2.0;
    }
    if (!invert_matrix(M, ndim, Minv)) return;

    Case<T> c(ndim, spatial);
    for (int64_t v = 0; v < c.nvox; ++v) {
        int64_t sub[3];
        c.unravel(v, sub);
        for (int r = 0; r < ndim; ++r) {
            double y = b[r];
            for (int k = 0; k < ndim; ++k) y += M[r*ndim+k] * (double)sub[k];
            c.disp[(size_t)(v*ndim+r)] = (T)(y - (double)sub[r]);
        }
    }
    c.run(bits);

    int64_t covered = 0;
    double worst = 0.0;
    for (int64_t v = 0; v < c.nvox; ++v) {
        if (c.owner[(size_t)v] < 0) continue;
        ++covered;
        int64_t sub[3];
        c.unravel(v, sub);
        for (int r = 0; r < ndim; ++r) {
            double x = 0.0;
            for (int k = 0; k < ndim; ++k)
                x += Minv[r*ndim+k] * ((double)sub[k] - b[k]);
            worst = std::max(worst,
                std::fabs((double)c.out[(size_t)(v*ndim+r)] - (x - (double)sub[r])));
        }
    }
    check(covered > 0, "affine.some_coverage");
    check(worst <= tol, "affine.exact");
    if (worst > tol) std::printf("    (worst affine error %.3e)\n", worst);
}

/* ------------------------------------------------------------------ */
/*  2. brute force over every (voxel, simplex) pair                  */
/* ------------------------------------------------------------------ */

// A smooth-ish random field: a few low-frequency sinusoids, so the map is
// non-trivial without needing a Gaussian blur.
template <typename T>
void fill_smooth(Case<T> & c, unsigned seed, double magnitude)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    double ph[3][4][3], am[3][4];
    for (int d = 0; d < c.ndim; ++d)
        for (int h = 0; h < 4; ++h) {
            am[d][h] = u(rng);
            for (int k = 0; k < c.ndim; ++k) ph[d][h][k] = u(rng) * 3.14159265358979;
        }
    double peak = 0.0;
    for (int64_t v = 0; v < c.nvox; ++v) {
        int64_t sub[3];
        c.unravel(v, sub);
        for (int d = 0; d < c.ndim; ++d) {
            double s = 0.0;
            for (int h = 0; h < 4; ++h) {
                double a = ph[d][h][0];
                for (int k = 0; k < c.ndim; ++k)
                    a += (double)sub[k] * (0.3 + 0.17 * h) * (k + 1) * 0.37;
                s += am[d][h] * std::sin(a);
            }
            c.disp[(size_t)(v*c.ndim+d)] = (T)s;
            peak = std::max(peak, std::fabs(s));
        }
    }
    if (peak > 0)
        for (size_t i = 0; i < c.disp.size(); ++i)
            c.disp[i] = (T)((double)c.disp[i] * magnitude / peak);
}

// `tol_in` is how far inside a simplex a voxel must be before the test insists
// the op found it, and `tol_out` how far outside before the test complains
// that the op claimed it. Voxels in the shell between the two get no verdict
// on containment: whether a point within rounding of a face is "inside" is not
// a question either implementation can answer, and it does not matter -- the
// mesh is conforming, so both sides of a shared face interpolate to the same
// value.
template <typename T>
void test_bruteforce(int ndim, const std::vector<int64_t> & spatial,
                     unsigned seed, double magnitude, uint8_t bits,
                     double tol)
{
    const double tol_in = 1e-7, tol_out = 1e-7;

    Case<T> c(ndim, spatial);
    fill_smooth(c, seed, magnitude);
    c.run(bits);

    int64_t ncell = 1;
    for (int d = 0; d < ndim; ++d) ncell *= spatial[d] - 1;
    const int64_t nsimplex = ncell * simplices_per_cell(ndim);

    for (int64_t v = 0; v < c.nvox; ++v) {
        int64_t sub[3];
        c.unravel(v, sub);
        double p[3];
        for (int d = 0; d < ndim; ++d) p[d] = (double)sub[d];

        const int64_t sid = c.owner[(size_t)v];

        int64_t first_in = -1;        // lowest simplex that clearly contains v
        double  claim_lo = 0.0;       // how far inside the winner v sits
        double  want[3]  = {0, 0, 0}; // the winner's own interpolation
        bool    saw_claim = false;

        for (int64_t s = 0; s < nsimplex; ++s) {
            int64_t corners[4][3];
            simplex_corners(s, ndim, spatial.data(), corners);
            double vdst[4][3] = {{0}};
            for (int i = 0; i <= ndim; ++i) {
                const int64_t ci = c.vindex(corners[i]);
                for (int d = 0; d < ndim; ++d)
                    vdst[i][d] = (double)corners[i][d]
                               + (double)c.disp[(size_t)(ci*ndim+d)];
            }
            double w[4];
            if (!barycentric(p, vdst, ndim, w)) continue;
            double lo = w[0];
            for (int i = 1; i <= ndim; ++i) lo = std::min(lo, w[i]);

            if (lo >= tol_in && first_in < 0) first_in = s;
            if (s == sid) {
                saw_claim = true;
                claim_lo  = lo;
                for (int d = 0; d < ndim; ++d) {
                    double x = 0.0;
                    for (int i = 0; i <= ndim; ++i)
                        x += w[i] * (double)corners[i][d];
                    want[d] = x - p[d];
                }
            }
        }

        bool ok = true;
        const bool finite = std::isfinite((double)c.out[(size_t)(v*ndim)]);
        if (sid < 0) {
            // Uncovered: no simplex may clearly contain the voxel, and the
            // output must be NaN.
            ok = (first_in < 0) && !finite;
        } else {
            // The winner must exist, must contain the voxel, must be the
            // lowest-indexed simplex that does (or lower, if it won on a
            // shared face), and its value must be its own interpolation.
            ok = saw_claim && finite && (claim_lo >= -tol_out);
            if (first_in >= 0) ok = ok && (sid <= first_in);
            for (int d = 0; ok && d < ndim; ++d) {
                const double got = (double)c.out[(size_t)(v*ndim+d)];
                const double scale = std::max(1.0, std::fabs(want[d]));
                ok = std::fabs(got - want[d]) / scale <= tol;
            }
        }
        check(ok, "bruteforce");
        if (!ok && g_failures < 40)
            std::printf("    voxel %lld: owner=%lld first_in=%lld finite=%d"
                        " claim_lo=%.3e\n",
                        (long long)v, (long long)sid, (long long)first_in,
                        (int)finite, claim_lo);
    }
}

/* ------------------------------------------------------------------ */
/*  3. layout invariance                                             */
/* ------------------------------------------------------------------ */

// Per-batch-element reference: run each element on its own, contiguously.
struct Expected
{
    std::vector<float>   disp, out;
    std::vector<int64_t> owner;
    int64_t              nvox;
};

Expected reference_batch(int ndim, const std::vector<int64_t> & spatial,
                         int64_t B)
{
    Expected e;
    e.nvox = 1;
    for (int d = 0; d < ndim; ++d) e.nvox *= spatial[d];
    for (int64_t bt = 0; bt < B; ++bt) {
        Case<float> c(ndim, spatial);
        fill_smooth(c, 7 + (unsigned)bt, 1.3);
        c.run(32);
        e.disp.insert(e.disp.end(), c.disp.begin(), c.disp.end());
        e.out.insert(e.out.end(), c.out.begin(), c.out.end());
        e.owner.insert(e.owner.end(), c.owner.begin(), c.owner.end());
    }
    return e;
}

bool same_field(const float * got, int64_t stride, const float * want,
                int64_t n)
{
    for (int64_t i = 0; i < n; ++i) {
        const float a = got[i * stride], b = want[i];
        const bool na = !std::isfinite(a), nb = !std::isfinite(b);
        if (na != nb) return false;
        if (!na && a != b) return false;
    }
    return true;
}

// A batch of two with padded (non-contiguous) strides on all three tensors
// must give exactly what two separate contiguous calls give. This is what
// catches a loop that assumes a compact layout, and it is cheap enough to run
// unconditionally.
void test_batched_padded(int ndim, const std::vector<int64_t> & spatial)
{
    const int64_t B = 2;
    const Expected e = reference_batch(ndim, spatial, B);
    const int64_t nvox = e.nvox;

    // Pad every axis's stride so nothing is contiguous anywhere.
    std::vector<int64_t> shf; shf.push_back(B);
    for (int d = 0; d < ndim; ++d) shf.push_back(spatial[d]);
    shf.push_back(ndim);
    std::vector<int64_t> stf(shf.size());
    {
        int64_t acc = 1;
        for (int64_t d = (int64_t)shf.size() - 1; d >= 0; --d) {
            stf[d] = acc;
            acc *= shf[d];
            acc += 3;                     // the padding
        }
    }
    std::vector<int64_t> sho(shf.begin(), shf.end() - 1);
    std::vector<int64_t> sto(sho.size());
    {
        int64_t acc = 1;
        for (int64_t d = (int64_t)sho.size() - 1; d >= 0; --d) {
            sto[d] = acc;
            acc *= sho[d];
            acc += 2;
        }
    }
    const size_t nf = (size_t)(stf[0] * B + 8);
    const size_t nw = (size_t)(sto[0] * B + 8);

    std::vector<float>   dbuf(nf, 0.0f), obuf(nf, 0.0f);
    std::vector<int64_t> wbuf(nw, 0);

    // Scatter the reference input into the padded layout.
    for (int64_t bt = 0; bt < B; ++bt)
        for (int64_t v = 0; v < nvox; ++v) {
            int64_t sub[3], rest = v;
            for (int d = ndim - 1; d >= 0; --d) {
                sub[d] = rest % spatial[d];
                rest /= spatial[d];
            }
            int64_t off = bt * stf[0];
            for (int d = 0; d < ndim; ++d) off += sub[d] * stf[1 + d];
            for (int d = 0; d < ndim; ++d)
                dbuf[(size_t)(off + d * stf[ndim + 1])] =
                    e.disp[(size_t)((bt * nvox + v) * ndim + d)];
        }

    DLTensor td = make_cpu_tensor(dbuf.data(), shf, stf, kDLFloat, 32);
    DLTensor to = make_cpu_tensor(obuf.data(), shf, stf, kDLFloat, 32);
    DLTensor tw = make_cpu_tensor(wbuf.data(), sho, sto, kDLInt,   64);
    ff::cpu::invert_field(to, tw, td, ndim, 0);

    bool same = true;
    for (int64_t bt = 0; bt < B && same; ++bt)
        for (int64_t v = 0; v < nvox && same; ++v) {
            int64_t sub[3], rest = v;
            for (int d = ndim - 1; d >= 0; --d) {
                sub[d] = rest % spatial[d];
                rest /= spatial[d];
            }
            int64_t foff = bt * stf[0], woff = bt * sto[0];
            for (int d = 0; d < ndim; ++d) {
                foff += sub[d] * stf[1 + d];
                woff += sub[d] * sto[1 + d];
            }
            same = same
                && wbuf[(size_t)woff] == e.owner[(size_t)(bt * nvox + v)]
                && same_field(&obuf[(size_t)foff], stf[ndim + 1],
                              &e.out[(size_t)((bt * nvox + v) * ndim)], ndim);
        }
    check(same, "batched_padded.matches_contiguous");
}

// The int64_t offset_t arm. A batch stride past INT32_MAX on the input makes
// canUse32BitIndexMath refuse the narrow path for the whole call (it is the
// AND over the operands), so this is the only way to reach that instantiation.
// The buffer is lazily allocated -- only the two touched regions commit pages
// -- and the test is skipped if the virtual allocation is refused.
void test_inflated_stride(int ndim, const std::vector<int64_t> & spatial)
{
    const int64_t B = 2;
    const Expected e = reference_batch(ndim, spatial, B);
    const int64_t nvox = e.nvox;
    const int64_t P = (int64_t)1 << 31;        // > INT32_MAX elements

    const size_t nd = (size_t)(P * (B - 1)) + (size_t)(nvox * ndim);
    float * dbig = static_cast<float *>(std::calloc(nd, sizeof(float)));
    if (!dbig) {
        std::printf("  [invfield inflated-stride] skipped"
                    " (lazy alloc of %.1f GB refused)\n",
                    nd * sizeof(float) / 1e9);
        return;
    }
    for (int64_t bt = 0; bt < B; ++bt)
        for (int64_t i = 0; i < nvox * ndim; ++i)
            dbig[bt * P + i] = e.disp[(size_t)(bt * nvox * ndim + i)];

    std::vector<float>   obuf((size_t)(B * nvox * ndim), 0.0f);
    std::vector<int64_t> wbuf((size_t)(B * nvox), 0);

    std::vector<int64_t> shf; shf.push_back(B);
    for (int d = 0; d < ndim; ++d) shf.push_back(spatial[d]);
    shf.push_back(ndim);
    std::vector<int64_t> stf = cstrides(shf);
    std::vector<int64_t> sti = stf; sti[0] = P;      // the inflated input
    std::vector<int64_t> sho(shf.begin(), shf.end() - 1);
    std::vector<int64_t> sto = cstrides(sho);

    DLTensor td = make_cpu_tensor(dbig,        shf, sti, kDLFloat, 32);
    DLTensor to = make_cpu_tensor(obuf.data(), shf, stf, kDLFloat, 32);
    DLTensor tw = make_cpu_tensor(wbuf.data(), sho, sto, kDLInt,   64);
    ff::cpu::invert_field(to, tw, td, ndim, 0);

    bool same = true;
    for (int64_t i = 0; i < B * nvox && same; ++i)
        same = wbuf[(size_t)i] == e.owner[(size_t)i];
    if (same)
        same = same_field(obuf.data(), 1, e.out.data(), B * nvox * ndim);
    check(same, "inflated_stride.matches_contiguous");

    std::free(dbig);
}

/* ------------------------------------------------------------------ */
/*  4. argument validation                                           */
/* ------------------------------------------------------------------ */

template <class F>
void check_throws(F f, const char * what)
{
    bool threw = false;
    try { f(); } catch (const std::exception &) { threw = true; }
    check(threw, what);
}

void test_validation()
{
    const int ndim = 2;
    std::vector<int64_t> sp = {5, 6};
    Case<double> c(ndim, sp);

    std::vector<int64_t> shf = sp; shf.push_back(ndim);
    std::vector<int64_t> stf = cstrides(shf);
    std::vector<int64_t> sho = sp;
    std::vector<int64_t> sto = cstrides(sho);

    check_throws([&] {
        DLTensor td = make_cpu_tensor(c.disp.data(),  shf, stf, kDLFloat, 64);
        DLTensor to = make_cpu_tensor(c.out.data(),   shf, stf, kDLFloat, 64);
        DLTensor tw = make_cpu_tensor(c.owner.data(), sho, sto, kDLInt,   64);
        ff::cpu::invert_field(to, tw, td, 1, 0);
    }, "validation.ndim1_throws");

    check_throws([&] {
        DLTensor td = make_cpu_tensor(c.disp.data(),  shf, stf, kDLFloat, 16);
        DLTensor to = make_cpu_tensor(c.out.data(),   shf, stf, kDLFloat, 16);
        DLTensor tw = make_cpu_tensor(c.owner.data(), sho, sto, kDLInt,   64);
        ff::cpu::invert_field(to, tw, td, ndim, 0);
    }, "validation.float16_throws");

    check_throws([&] {
        DLTensor td = make_cpu_tensor(c.disp.data(),  shf, stf, kDLFloat, 64);
        DLTensor to = make_cpu_tensor(c.out.data(),   shf, stf, kDLFloat, 64);
        DLTensor tw = make_cpu_tensor(c.owner.data(), sho, sto, kDLInt,   32);
        ff::cpu::invert_field(to, tw, td, ndim, 0);
    }, "validation.owner_int32_throws");

    check_throws([&] {
        // owner given the field's rank instead of one fewer
        DLTensor td = make_cpu_tensor(c.disp.data(),  shf, stf, kDLFloat, 64);
        DLTensor to = make_cpu_tensor(c.out.data(),   shf, stf, kDLFloat, 64);
        DLTensor tw = make_cpu_tensor(c.owner.data(), shf, stf, kDLInt,   64);
        ff::cpu::invert_field(to, tw, td, ndim, 0);
    }, "validation.owner_rank_throws");

    // A degenerate spatial axis has no cells at all.
    std::vector<int64_t> sp1 = {1, 6};
    Case<double> c1(ndim, sp1);
    std::vector<int64_t> shf1 = sp1; shf1.push_back(ndim);
    std::vector<int64_t> stf1 = cstrides(shf1);
    std::vector<int64_t> sho1 = sp1;
    std::vector<int64_t> sto1 = cstrides(sho1);
    check_throws([&] {
        DLTensor td = make_cpu_tensor(c1.disp.data(),  shf1, stf1, kDLFloat, 64);
        DLTensor to = make_cpu_tensor(c1.out.data(),   shf1, stf1, kDLFloat, 64);
        DLTensor tw = make_cpu_tensor(c1.owner.data(), sho1, sto1, kDLInt,   64);
        ff::cpu::invert_field(to, tw, td, ndim, 0);
    }, "validation.thin_axis_throws");

    check_throws([&] {
        // last axis of the field must hold ndim components
        std::vector<int64_t> bad = sp; bad.push_back(ndim + 1);
        std::vector<int64_t> bst = cstrides(bad);
        std::vector<double> big((size_t)(c.nvox * (ndim + 1)), 0.0);
        DLTensor td = make_cpu_tensor(big.data(),     bad, bst, kDLFloat, 64);
        DLTensor to = make_cpu_tensor(big.data(),     bad, bst, kDLFloat, 64);
        DLTensor tw = make_cpu_tensor(c.owner.data(), sho, sto, kDLInt,   64);
        ff::cpu::invert_field(to, tw, td, ndim, 0);
    }, "validation.component_count_throws");
}

} // namespace

int main()
{
    std::printf("invfield module CPU tests\n");

    test_validation();

    // --- closed-form cases, 2D and 3D, float64 and float32 ---
    const std::vector<int64_t> sp2 = {17, 18};
    const std::vector<int64_t> sp3 = {9, 10, 11};

    test_identity<double>(2, sp2, 64, 1e-12);
    test_identity<double>(3, sp3, 64, 1e-12);
    test_identity<float >(2, sp2, 32, 1e-5);
    test_identity<float >(3, sp3, 32, 1e-5);

    const double t2[2] = {1.25, -2.5};
    const double t3[3] = {0.75, -1.5, 2.25};
    test_translation<double>(2, sp2, t2, 64, 1e-10);
    test_translation<double>(3, sp3, t3, 64, 1e-10);
    test_translation<float >(2, sp2, t2, 32, 1e-4);
    test_translation<float >(3, sp3, t3, 32, 1e-4);

    for (unsigned s = 1; s <= 4; ++s) {
        test_affine<double>(2, sp2, s,      64, 1e-10);
        test_affine<double>(3, sp3, s + 10, 64, 1e-10);
        test_affine<float >(2, sp2, s,      32, 1e-3);
        test_affine<float >(3, sp3, s + 10, 32, 1e-3);
    }

    // --- brute force: the owner / coverage / value contract ---
    const std::vector<int64_t> bf2 = {13, 14};
    const std::vector<int64_t> bf3 = {7, 8, 9};
    for (unsigned s = 1; s <= 2; ++s) {
        test_bruteforce<double>(2, bf2, s,      0.7, 64, 1e-9);
        test_bruteforce<double>(2, bf2, s + 10, 2.5, 64, 1e-9);
        test_bruteforce<double>(3, bf3, s,      0.7, 64, 1e-9);
        test_bruteforce<double>(3, bf3, s + 10, 2.5, 64, 1e-9);
    }
    test_bruteforce<float>(2, bf2, 5, 1.1, 32, 1e-4);
    test_bruteforce<float>(3, bf3, 5, 1.1, 32, 1e-4);

    // --- layout invariance ---
    test_batched_padded(2, {11, 12});
    test_batched_padded(3, {6, 7, 8});
    test_inflated_stride(2, {11, 12});
    test_inflated_stride(3, {6, 7, 8});

    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
