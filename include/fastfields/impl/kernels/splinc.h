#pragma once
// Compute spline interpolating coefficients (the spline prefilter)
//
// A spline of order 2 or more does not pass through its coefficients. Along
// one axis of length n, sampling coefficients `c` at the integer nodes gives
// `A @ c`, where the banded matrix `A` depends on the order and on the boundary
// condition, and the prefilter computes `c = inv(A) @ f` from the values `f`.
//
// For DCT1, DCT2, DST1, DST2 and DFT the inverse is a recursive filter: one
// causal and one anticausal pass per pole of the spline (Unser, Aldroubi &
// Eden). These boundaries extend the input of every pass the same way as they
// extend the coefficients, so the first value of each pass is a sum over one
// period of the extended input, which `causal_init` / `anticausal_init`
// compute in closed form -- exactly, for any axis length. Past a pole's
// horizon K (|z|^K < 1e-17) the sum is truncated, which changes it by less
// than its own rounding error.
//
// Zero and Replicate have no such rule from order 4: the inputs of the passes
// after the first do not follow the extension of the coefficients. At orders 2
// and 3 their `A` is that of DST1 / DCT2, whose filters therefore give their
// coefficients exactly. From order 4 the coefficients are computed with the
// DST1 / DCT2 filter and then corrected near the edges: away from the edges
// every row of `A` is the same convolution, so the correction is a sum of
// powers of the poles decaying from each edge, whose 2h weights (h = order/2)
// solve a small linear system written on the first and last h rows of `A`.
// Axes too short for that (n < 2h) are solved directly (n <= 5).
//
// The initial conditions and the edge correction are a port of brainhops'
// `interpol/_coeff.py`, itself a NumPy/CuPy port of torch-interpol. The
// recursion goes back to John Ashburner's C code in SPM (bsplines.c), derived
// from Philippe Thevenaz's.
//
// The correction matrices only depend on (n, poles, bound), so they live in a
// `plan` that the loops build once per call (CPU) or once per thread (CUDA),
// not once per line. Everything here is fixed-size and allocation-free: at
// most 3 poles, systems of size at most 6, every loop bounded by n or by a
// pole's horizon (<= 63 samples at order 7).
//
// Boundary semantics follow `bounds.h` exactly -- in particular DST1 on an
// axis of length 1 is treated there as a constant extension (sign 1, index 0),
// so its prefilter is the identity, like DCT1/DCT2/DFT/Replicate at n = 1.
//
// References
// ----------
// ..[1]  M. Unser, A. Aldroubi and M. Eden.
//        "B-Spline Signal Processing: Part I-Theory,"
//        IEEE Transactions on Signal Processing 41(2):821-832 (1993).
// ..[2]  M. Unser, A. Aldroubi and M. Eden.
//        "B-Spline Signal Processing: Part II-Efficient Design and Applications,"
//        IEEE Transactions on Signal Processing 41(2):834-848 (1993).
// ..[3]  M. Unser.
//        "Splines: A Perfect Fit for Signal and Image Processing,"
//        IEEE Signal Processing Magazine 16(6):22-38 (1999).

#include <fastfields/core/cuda_switch.h>
#include "spline.h"
#include "bounds.h"
#include "utils.h"

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(splinc)

/***********************************************************************
 *                              POLES                                  *
 ***********************************************************************/

// The poles of the prefilter of each order, in (-1, 0), largest in magnitude
// first. Computed with mpmath at 60 digits and rounded to 30 significant
// digits (brainhops `_POLES`): the closed forms with square roots that exist
// up to order 5 subtract nearly equal numbers and lose up to 2e-13 in float64.
template <typename scalar_t>
inline FF_CUDEV int get_poles(int order, scalar_t * poles)
{
    switch (order) {
        case 0:
        case 1:
            return 0;
        case 2:
            poles[0] = static_cast<scalar_t>(-0.171572875253809902396622551581);
            return 1;
        case 3:
            poles[0] = static_cast<scalar_t>(-0.267949192431122706472553658494);
            return 1;
        case 4:
            poles[0] = static_cast<scalar_t>(-0.361341225900220177092212841326);
            poles[1] = static_cast<scalar_t>(-0.0137254292973391213603312269391);
            return 2;
        case 5:
            poles[0] = static_cast<scalar_t>(-0.430575347099973791851434783494);
            poles[1] = static_cast<scalar_t>(-0.0430962882032646538227123768226);
            return 2;
        case 6:
            poles[0] = static_cast<scalar_t>(-0.488294589303044755130118038884);
            poles[1] = static_cast<scalar_t>(-0.0816792710762375125979377657371);
            poles[2] = static_cast<scalar_t>(-0.00141415180832581775108724397656);
            return 3;
        case 7:
            poles[0] = static_cast<scalar_t>(-0.535280430796438165542403781682);
            poles[1] = static_cast<scalar_t>(-0.122554615192326690515272264359);
            poles[2] = static_cast<scalar_t>(-0.00914869480960827692859302165165);
            return 3;
    }
    return -1;
}

// The gain makes the filter return a constant signal unchanged: the product of
// (1 - z) * (1 - 1/z) over the poles, which is 1 / b[h] (see `plan::b`).
template <typename scalar_t>
inline FF_CUDEV scalar_t get_gain(const scalar_t * poles, int npoles)
{
    double lam = 1.;
    for (int i=0; i<npoles; ++i, ++poles) {
        double pole = static_cast<double>(*poles);
        lam *= (1. - pole) * (1. - 1./pole);
    }
    return static_cast<scalar_t>(lam);
}

FF_NAMESPACE_BEGIN(_splinc)

/***********************************************************************
 *                        SMALL HELPERS                                *
 ***********************************************************************/

// z**k by binary exponentiation (no libm: exact enough, and host/device-safe).
template <typename reduce_t, typename offset_t>
inline FF_CUDEV reduce_t ipow(reduce_t z, offset_t k)
{
    reduce_t out = 1, base = z;
    while (k > 0) {
        if (k & 1) out *= base;
        base *= base;
        k >>= 1;
    }
    return out;
}

// Smallest K such that |z|**K < 1e-17, i.e. negligible next to 1 in float64.
// At most 63 for the poles above; the cap only guards against |z| >= 1.
template <typename reduce_t>
inline FF_CUDEV int horizon(reduce_t z)
{
    reduce_t a = z < 0 ? -z : z, p = 1;
    int k = 0;
    while (p >= static_cast<reduce_t>(1e-17) && k < 1024) { p *= a; ++k; }
    return k;
}

template <typename reduce_t, typename scalar_t, typename offset_t>
inline FF_CUDEV reduce_t ld(const scalar_t * x, offset_t i, offset_t stride)
{ return static_cast<reduce_t>(x[i * stride]); }

/***********************************************************************
 *                      INITIAL CONDITIONS                             *
 ***********************************************************************/
//
// A pole z makes a causal pass y[k] = x[k] + z * y[k-1], then an anticausal
// pass w[k] = z * (w[k+1] - y[k]). When the input is extended past the axis by
// the boundary, the exact first values of the two passes are
//
//     y[0]   = sum_{j>=0} z^j x[-j],
//     w[n-1] = -z / (1 - z^2) * (y[n-1] + sum_{j>=1} z^j y[n-1+j]),
//
// written below as sums over one period of the extension. `K` is the pole's
// horizon: when n - 1 >= K the powers z^K and above are dropped.

template <typename reduce_t, typename scalar_t, typename offset_t>
inline FF_CUDEV reduce_t
causal_init(bound::type bnd, const scalar_t * x, offset_t n, offset_t st,
            reduce_t z, offset_t K)
{
    const reduce_t x0 = ld<reduce_t>(x, offset_t(0), st);
    if (n - 1 >= K) {
        if (bnd == bound::type::DFT) {
            // x[0] + z x[n-1] + z^2 x[n-2] + ...
            reduce_t out = x0, zk = z;
            for (offset_t j = 1; j <= K; ++j, zk *= z)
                out += zk * ld<reduce_t>(x, n - j, st);
            return out;
        }
        reduce_t sum = 0, zk = 1;
        for (offset_t j = 0; j < K; ++j, zk *= z)
            sum += zk * ld<reduce_t>(x, j, st);
        // x[0] plus the first samples reflected to the left of the axis:
        // about 0 for DCT1, about -1/2 for DCT2/DST2, about -1 for DST1, with
        // a change of sign for the antisymmetric two.
        switch (bnd) {
            case bound::type::DCT1: return sum;
            case bound::type::DST2: return x0 - z * sum;
            case bound::type::DST1: return x0 - z * z * sum;
            default:                return x0 + z * sum;   // DCT2
        }
    }

    // Short axis: the whole period, in closed form. The forward powers z^k
    // and the reversed ones z^(M-k) are accumulated by multiplication.
    switch (bnd) {
        case bound::type::DCT1: {
            // One period has 2n-2 samples; the first and last appear once.
            if (n == 1) return x0;
            const offset_t M = 2 * n - 2;
            reduce_t out = x0 + ipow(z, n - 1) * ld<reduce_t>(x, n - 1, st);
            reduce_t zk = z;
            for (offset_t k = 1; k < n - 1; ++k, zk *= z)
                out += zk * ld<reduce_t>(x, k, st);
            reduce_t zr = ipow(z, n);               // z^(M-k) at k = n-2
            for (offset_t k = n - 2; k >= 1; --k, zr *= z)
                out += zr * ld<reduce_t>(x, k, st);
            return out / (1 - ipow(z, M));
        }
        case bound::type::DCT2:
        case bound::type::DST2: {
            // One period has 2n samples, sign-flipped in its second half for
            // DST2: z^k +- z^(2n-1-k).
            const reduce_t s = bnd == bound::type::DST2 ? -1 : 1;
            reduce_t sum = 0, zk = 1;
            for (offset_t k = 0; k < n; ++k, zk *= z)
                sum += zk * ld<reduce_t>(x, k, st);
            reduce_t rev = 0, zr = zk;              // zk == z^n == z^(2n-1-k) at k = n-1
            for (offset_t k = n - 1; k >= 0; --k, zr *= z)
                rev += zr * ld<reduce_t>(x, k, st);
            return x0 + s * z * (sum + s * rev) / (1 - zk * zk);
        }
        case bound::type::DST1: {
            // One period has 2n+2 samples, two of which are zero:
            // z^2 (z^(2n-k) - z^k).
            reduce_t sum = 0, zk = 1;
            for (offset_t k = 0; k < n; ++k, zk *= z)
                sum += zk * ld<reduce_t>(x, k, st);
            reduce_t rev = 0, zr = zk * z;          // z^(n+1) == z^(2n-k) at k = n-1
            for (offset_t k = n - 1; k >= 0; --k, zr *= z)
                rev += zr * ld<reduce_t>(x, k, st);
            const reduce_t zn1 = zk * z;            // z^(n+1)
            return x0 + z * z * (rev - sum) / (1 - zn1 * zn1);
        }
        default: {
            // DFT: one period has n samples, x[0] + sum_{k>=1} z^(n-k) x[k].
            reduce_t out = x0, zr = z;
            for (offset_t k = n - 1; k >= 1; --k, zr *= z)
                out += zr * ld<reduce_t>(x, k, st);
            return out / (1 - ipow(z, n));
        }
    }
}

// `y` holds the output of the causal pass.
template <typename reduce_t, typename scalar_t, typename offset_t>
inline FF_CUDEV reduce_t
anticausal_init(bound::type bnd, const scalar_t * y, offset_t n, offset_t st,
                reduce_t z, offset_t K)
{
    const reduce_t yl = ld<reduce_t>(y, n - 1, st);
    switch (bnd) {
        case bound::type::DCT1:
            // The samples after the last one are y[n-2], y[n-3], ...
            return (z * z * ld<reduce_t>(y, n - 2, st) + z * yl) / (z * z - 1);
        case bound::type::DCT2: return  z / (z - 1) * yl;
        case bound::type::DST2: return -z / (1 + z) * yl;
        case bound::type::DST1: return -z * yl;
        default: break;
    }
    // DFT: the causal pass keeps the input periodic, so the samples after
    // the last one are y[0], y[1], ...
    if (n - 1 >= K) {
        reduce_t out = -z * yl, zk = z * z;
        for (offset_t j = 0; j < K - 1; ++j, zk *= z)
            out -= zk * ld<reduce_t>(y, j, st);
        return out;
    }
    reduce_t out = -z * yl, zk = z * z;
    for (offset_t j = 0; j < n - 1; ++j, zk *= z)
        out -= zk * ld<reduce_t>(y, j, st);
    return out / (1 - ipow(z, n));
}

/***********************************************************************
 *                   SAMPLING MATRIX / SMALL SOLVES                    *
 ***********************************************************************/

// Map a sample index j, at most `n` past either edge, into the axis for the
// four boundaries the edge correction needs. Returns the sign (0 = the sample
// is zero). Matches bounds.h for every j it is called with (|overshoot| <= h
// and n >= 1 for Zero/Replicate; n >= 2h for DST1/DCT2).
template <typename offset_t>
inline FF_CUDEV int extend(bound::type bnd, offset_t j, offset_t n, offset_t & idx)
{
    switch (bnd) {
        case bound::type::Zero:
            idx = j;
            return (j >= 0 && j < n) ? 1 : 0;
        case bound::type::Replicate:
            idx = j < 0 ? offset_t(0) : j >= n ? n - 1 : j;
            return 1;
        case bound::type::DST1:
            // antisymmetric about -1 and n
            if (j == -1 || j == n) { idx = 0; return 0; }
            if (j < -1) { idx = -j - 2;    return -1; }
            if (j >  n) { idx = 2 * n - j; return -1; }
            idx = j; return 1;
        default:
            // DCT2: symmetric about -1/2 and n-1/2
            idx = j < 0 ? -j - 1 : j >= n ? 2 * n - 1 - j : j;
            return 1;
    }
}

// In-place inverse of a small dense matrix (row-major, size m <= 6) by
// Gauss-Jordan elimination with partial pivoting.
template <typename reduce_t>
inline FF_CUDEV void invert_small(reduce_t * a, int m)
{
    reduce_t inv[36];
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            inv[i*m+j] = i == j ? reduce_t(1) : reduce_t(0);
    for (int c = 0; c < m; ++c) {
        int p = c;
        reduce_t best = a[c*m+c] < 0 ? -a[c*m+c] : a[c*m+c];
        for (int r = c + 1; r < m; ++r) {
            reduce_t v = a[r*m+c] < 0 ? -a[r*m+c] : a[r*m+c];
            if (v > best) { best = v; p = r; }
        }
        if (p != c)
            for (int j = 0; j < m; ++j) {
                reduce_t t = a[c*m+j]; a[c*m+j] = a[p*m+j]; a[p*m+j] = t;
                t = inv[c*m+j]; inv[c*m+j] = inv[p*m+j]; inv[p*m+j] = t;
            }
        const reduce_t d = 1 / a[c*m+c];
        for (int j = 0; j < m; ++j) { a[c*m+j] *= d; inv[c*m+j] *= d; }
        for (int r = 0; r < m; ++r) {
            if (r == c) continue;
            const reduce_t f = a[r*m+c];
            if (f == 0) continue;
            for (int j = 0; j < m; ++j) {
                a[r*m+j]   -= f * a[c*m+j];
                inv[r*m+j] -= f * inv[c*m+j];
            }
        }
    }
    for (int i = 0; i < m*m; ++i) a[i] = inv[i];
}

FF_NAMESPACE_END(_splinc)

/***********************************************************************
 *                                PLAN                                 *
 ***********************************************************************/

FF_NAMESPACE_BEGIN(_splinc)
enum class mode : int8_t {
    Identity  = 0,  // A == identity (n == 1 with a constant extension)
    Recursive = 1,  // exact recursive filter of `base`
    Corrected = 2,  // recursive filter of `base`, then the edge correction
    Dense     = 3   // n < 2h: c = inv(A) @ f, with the n x n inverse in `mat`
};
FF_NAMESPACE_END(_splinc)

// Everything about the prefilter of one axis that does not depend on the data:
// built once per call, shared by every line. Trivially copyable.
template <int npoles, typename reduce_t, typename offset_t>
struct plan {
    static const int h = npoles;           // order / 2

    _splinc::mode mode;
    bound::type   bnd;                     // the boundary asked for
    bound::type   base;                    // the boundary whose recursion runs
    offset_t      n;
    reduce_t      gain;
    reduce_t      poles[npoles];
    offset_t      K[npoles];               // horizon of each pole
    offset_t      Kmax;
    // Values of the spline at the integers -h..h (the rows of `A`), rebuilt
    // from the poles: z^h B(z) = b[h] prod_s (z - z_s)(z - 1/z_s), B(1) = 1.
    reduce_t      b[2*npoles+1];
    // Corrected: (2h x 2h) map from the base coefficients [c[0..h-1],
    //            c[n-h..n-1]] to the weights of the sequences
    //            [z_s^k ..., z_s^(n-1-k) ...].
    // Dense:     the (n x n) inverse of A.
    reduce_t      mat[36];
};

template <int npoles, typename reduce_t, typename offset_t>
inline FF_CUDEV plan<npoles, reduce_t, offset_t>
make_plan(bound::type bnd, offset_t n, const reduce_t * poles)
{
    using namespace _splinc;
    const int h = npoles;
    plan<npoles, reduce_t, offset_t> p;
    p.bnd  = bnd;
    p.n    = n;
    p.gain = get_gain(poles, npoles);
    p.Kmax = 0;
    for (int s = 0; s < h; ++s) {
        p.poles[s] = poles[s];
        p.K[s]     = static_cast<offset_t>(horizon(poles[s]));
        if (p.K[s] > p.Kmax) p.Kmax = p.K[s];
    }

    // Stencil from the poles: prod_s (z^2 - (z_s + 1/z_s) z + 1) / gain.
    {
        reduce_t c[2*npoles+1];
        for (int i = 0; i <= 2*h; ++i) c[i] = 0;
        c[0] = 1;
        for (int s = 0; s < h; ++s) {
            const reduce_t a = -(poles[s] + 1 / poles[s]);
            const int deg = 2 * s;                 // current degree
            for (int i = deg + 2; i >= 0; --i) {
                reduce_t v = c[i];
                if (i >= 1) v += a * c[i-1];
                if (i >= 2) v += c[i-2];
                c[i] = v;
            }
        }
        for (int i = 0; i <= 2*h; ++i) p.b[i] = c[i] / p.gain;
    }

    const bool is_zr = bnd == bound::type::Zero ||
                       bnd == bound::type::Replicate;
    p.base = bnd == bound::type::Zero      ? bound::type::DST1 :
             bnd == bound::type::Replicate ? bound::type::DCT2 : bnd;

    if (n == 1 && bnd != bound::type::Zero && bnd != bound::type::DST2) {
        // DCT1/DCT2/DFT/Replicate (and DST1, per bounds.h) extend a single
        // sample as a constant, and the stencil sums to one: A = [1].
        p.mode = mode::Identity;
        return p;
    }
    if (!is_zr || h == 1) {
        p.mode = mode::Recursive;
        return p;
    }

    if (n < 2 * h) {
        // Too short for the edge correction: no row is away from the edges.
        const int m = static_cast<int>(n);
        for (int i = 0; i < m*m; ++i) p.mat[i] = 0;
        for (int r = 0; r < m; ++r)
            for (int t = -h; t <= h; ++t) {
                offset_t idx;
                int sgn = extend(bnd, offset_t(r - t), n, idx);
                if (sgn) p.mat[r*m + static_cast<int>(idx)] += sgn * p.b[t+h];
            }
        invert_small(p.mat, m);
        p.mode = mode::Dense;
        return p;
    }

    // Edge correction. Rows i < h are r = i; rows i >= h are r = n - 2h + i.
    // S[i][s]: row r of A applied to sequence s (s < h: z_s^j, s >= h:
    // z_(s-h)^(n-1-j)), with A's own extension. D[i][k]: row r of (B - A)
    // applied to the base coefficients, as weights on cb[k] (k < h) and on
    // cb[n-2h+k] (k >= h). Out-of-axis taps of row r stay on its own edge,
    // and land within h samples of it, because n >= 2h.
    reduce_t S[36], D[36];
    const int m = 2 * h;
    for (int i = 0; i < m*m; ++i) { S[i] = 0; D[i] = 0; }
    for (int i = 0; i < m; ++i) {
        const offset_t r = i < h ? offset_t(i) : n - 2*h + i;
        for (int t = -h; t <= h; ++t) {
            const offset_t j  = r - t;
            const reduce_t bt = p.b[t+h];
            offset_t idx;
            const int sgn = extend(bnd, j, n, idx);
            if (sgn)
                for (int s = 0; s < h; ++s) {
                    S[i*m + s]     += sgn * bt * ipow(poles[s], idx);
                    S[i*m + h + s] += sgn * bt * ipow(poles[s], n - 1 - idx);
                }
            if (j >= 0 && j < n) continue;          // B and A agree inside
            offset_t bidx;
            const int bsgn = extend(p.base, j, n, bidx);
            if (bsgn) {
                const int k = bidx < h ? static_cast<int>(bidx)
                                       : static_cast<int>(bidx - (n - 2*h));
                D[i*m + k] += bsgn * bt;
            }
            if (sgn) {
                const int k = idx < h ? static_cast<int>(idx)
                                      : static_cast<int>(idx - (n - 2*h));
                D[i*m + k] -= sgn * bt;
            }
        }
    }
    invert_small(S, m);
    for (int i = 0; i < m; ++i)
        for (int k = 0; k < m; ++k) {
            reduce_t v = 0;
            for (int j = 0; j < m; ++j) v += S[i*m+j] * D[j*m+k];
            p.mat[i*m+k] = v;
        }
    p.mode = mode::Corrected;
    return p;
}

/***********************************************************************
 *                               FILTER                                *
 ***********************************************************************/

FF_NAMESPACE_BEGIN(_splinc)

template <int npoles, typename scalar_t, typename reduce_t, typename offset_t>
inline FF_CUDEV void recursive(scalar_t * inp, offset_t size, offset_t stride,
                               const plan<npoles, reduce_t, offset_t> & p)
{
    const reduce_t lam = p.gain;
    for (offset_t i = 0; i < size; ++i)
        inp[i * stride] *= lam;

    scalar_t *prev, *cur;
    for (int k = 0; k < npoles; ++k) {
        const reduce_t pole = p.poles[k];
        cur = inp;

        *cur = static_cast<scalar_t>(
            causal_init(p.base, inp, size, stride, pole, p.K[k]));

        prev = cur;
        cur = prev + stride;
        for (offset_t i = 1; i < size; ++i, cur += stride) {
            *cur += pole * (*prev);
            prev = cur;
        }

        cur -= stride;
        *cur = static_cast<scalar_t>(
            anticausal_init(p.base, inp, size, stride, pole, p.K[k]));

        prev = cur;
        cur = prev - stride;
        for (offset_t i = size-2; i >= 0; --i, cur -= stride) {
            *cur = pole * ((*prev) - (*cur));
            prev = cur;
        }
    }
}

FF_NAMESPACE_END(_splinc)

// Prefilter one line with a prebuilt plan (`p.n` must equal the line length).
template <int npoles, typename scalar_t, typename reduce_t, typename offset_t>
inline FF_CUDEV void filter(scalar_t * inp, offset_t stride,
                            const plan<npoles, reduce_t, offset_t> & p)
{
    using _splinc::mode;
    const int h = npoles;
    const offset_t n = p.n;

    switch (p.mode) {
        case mode::Identity:
            return;
        case mode::Dense: {
            const int m = static_cast<int>(n);
            reduce_t f[2*npoles];
            for (int i = 0; i < m; ++i) f[i] = _splinc::ld<reduce_t>(inp, offset_t(i), stride);
            for (int i = 0; i < m; ++i) {
                reduce_t v = 0;
                for (int j = 0; j < m; ++j) v += p.mat[i*m+j] * f[j];
                inp[i * stride] = static_cast<scalar_t>(v);
            }
            return;
        }
        case mode::Recursive:
            _splinc::recursive(inp, n, stride, p);
            return;
        case mode::Corrected:
            break;
    }

    _splinc::recursive(inp, n, stride, p);

    const int m = 2 * h;
    reduce_t e[2*npoles], w[2*npoles];
    for (int k = 0; k < h; ++k) {
        e[k]     = _splinc::ld<reduce_t>(inp, offset_t(k),         stride);
        e[h + k] = _splinc::ld<reduce_t>(inp, n - 2*h + offset_t(h + k), stride);
    }
    for (int i = 0; i < m; ++i) {
        reduce_t v = 0;
        for (int k = 0; k < m; ++k) v += p.mat[i*m+k] * e[k];
        w[i] = v;
    }
    // The sequences z^k start at the first sample, z^(n-1-k) at the last.
    const offset_t L = n < p.Kmax ? n : p.Kmax;
    reduce_t zl[npoles], zr[npoles];
    for (int s = 0; s < h; ++s) { zl[s] = w[s]; zr[s] = w[h + s]; }
    for (offset_t k = 0; k < L; ++k) {
        reduce_t dl = 0, dr = 0;
        for (int s = 0; s < h; ++s) {
            dl += zl[s]; zl[s] *= p.poles[s];
            dr += zr[s]; zr[s] *= p.poles[s];
        }
        inp[k * stride] += static_cast<scalar_t>(dl);
        inp[(n - 1 - k) * stride] += static_cast<scalar_t>(dr);
    }
}

// Prefilter one line. `B` is the compile-time boundary condition, or
// `bound::type::Dynamic` to read it from `bnd` at run time. Builds the plan
// for this line alone -- loops over many lines should build it once with
// `make_plan` and call the overload above.
template <bound::type B, int npoles,
          typename scalar_t, typename reduce_t, typename offset_t>
inline FF_CUDEV void filter(scalar_t * inp, offset_t size, offset_t stride,
                            const reduce_t * poles, bound::type bnd = B)
{
    const plan<npoles, reduce_t, offset_t> p = make_plan<npoles>(
        B == bound::type::Dynamic ? bnd : B, size, poles);
    filter(inp, stride, p);
}

FF_NAMESPACE_END(splinc)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
