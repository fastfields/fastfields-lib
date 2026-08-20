// CPU unit tests for the solve_field module (Jacobi-preconditioned CG).
//
// The oracle is a *manufactured solution*: pick an arbitrary field `x_true`,
// form the right-hand side `g = (H + L) x_true` with the already-gated
// `sym_matvec` / `field_matvec` operators, then hand `g` to `field_cg` and
// require that it recovers `x_true`. That is a strictly stronger statement
// than "the residual got smaller" -- a solver that converged to the wrong
// fixed point, or that quietly dropped the Hessian or the regulariser from
// the operator, would still shrink *its own* residual but would not land on
// `x_true`.
//
// Each case also cross-checks the residual `field_cg` reports against one
// recomputed here from the returned solution, and exercises the in-place
// warm-start contract (`sol` in = initial guess, `sol` out = solution).
//
// Build (from fastfields-cpu-lib):
//   clang++ -std=c++11 -O2 -ferror-limit=5 -I. tests/test_solve_field.cpp \
//           solve_field.cpp reg_field.cpp posdef.cpp -o build/test_solve_field
//   ./build/test_solve_field

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>
#include <fastfields/core/dlpack.h>
#include <fastfields/api/cpu/solve_field.h>
#include <fastfields/api/cpu/reg_field.h>
#include <fastfields/api/cpu/posdef.h>

namespace {

enum { B_ZERO = 0, B_DCT2 = 3, B_DST2 = 5, B_DFT = 6 };

template <typename T>
DLTensor make_cpu_tensor(T* data, std::vector<int64_t>& shape,
                         std::vector<int64_t>& strides, uint8_t bits)
{
    DLTensor t;
    t.data               = static_cast<void*>(data);
    t.device.device_type = kDLCPU;
    t.device.device_id   = 0;
    t.ndim               = static_cast<int32_t>(shape.size());
    t.dtype.code         = static_cast<uint8_t>(kDLFloat);
    t.dtype.bits         = bits;
    t.dtype.lanes        = 1;
    t.shape              = shape.data();
    t.strides            = strides.data();
    t.byte_offset        = 0;
    return t;
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> s(shape.size());
    int64_t acc = 1;
    for (int64_t d = (int64_t)shape.size() - 1; d >= 0; --d) { s[d] = acc; acc *= shape[d]; }
    return s;
}

int g_failures = 0;
int g_checks   = 0;

void check_close(double a, double b, const char* what, double tol = 1e-5)
{
    ++g_checks;
    double diff = std::fabs(a - b);
    double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    if (diff / scale > tol) {
        ++g_failures;
        std::printf("  MISMATCH [%s]: got %.8g expected %.8g\n", what, a, b);
    }
}

void check_true(bool ok, const char* what)
{
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAILED [%s]\n", what);
    }
}

// Packed compact-symmetric layout is "diagonal first, then rows":
//   [h00 h11 ... h(C-1)(C-1) | h01 h02 ... | h12 ...]
// Fill a strictly diagonally dominant (hence positive definite) matrix so
// that both `sym_matvec` and the preconditioner's `sym_solve` are well posed,
// and the off-diagonal entries actually couple the channels (a purely
// diagonal Hessian would not exercise the compact-symmetric paths).
template <typename scalar_t>
void fill_hessian(std::vector<scalar_t>& hes, int64_t nvox, int64_t C,
                  double scale)
{
    const int64_t K = C * (C + 1) / 2;
    for (int64_t v = 0; v < nvox; ++v) {
        scalar_t* h = hes.data() + v * K;
        // off-diagonals first, so the diagonal can dominate their row sums
        std::vector<double> off(static_cast<size_t>(K - C), 0.0);
        for (int64_t k = 0; k < K - C; ++k)
            off[static_cast<size_t>(k)] =
                0.35 * std::sin(0.7 * (double)v + 1.3 * (double)k + 0.4);
        std::vector<double> rowsum(static_cast<size_t>(C), 0.0);
        int64_t k = 0;
        for (int64_t i = 0; i < C; ++i)
            for (int64_t j = i + 1; j < C; ++j, ++k) {
                double o = off[static_cast<size_t>(k)];
                rowsum[static_cast<size_t>(i)] += std::fabs(o);
                rowsum[static_cast<size_t>(j)] += std::fabs(o);
                h[C + k] = (scalar_t)o;
            }
        for (int64_t c = 0; c < C; ++c)
            h[c] = (scalar_t)(scale * (1.0 + 0.25 * std::sin(0.3 * (double)v + (double)c))
                              + rowsum[static_cast<size_t>(c)]);
    }
}

// ---------------------------------------------------------------------
// field_cg: manufactured-solution solve of (H + L) x = g.
// ---------------------------------------------------------------------
template <typename scalar_t>
void run_cg(const char* label, int ndim, const std::vector<int64_t>& spatial,
            int64_t C, int order,
            const std::vector<double>& absolute,
            const std::vector<double>& membrane,
            const std::vector<double>& bending,
            uint8_t bits, int bound, double hscale,
            int nb_iter, double tol, double xtol,
            bool warm_start = false)
{
    const int64_t K = C * (C + 1) / 2;
    std::vector<int64_t> fshape(spatial), hshape(spatial);
    fshape.push_back(C);
    hshape.push_back(K);
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    std::vector<int64_t> hstr = contiguous_strides(hshape);

    int64_t nvox = 1;
    for (size_t d = 0; d < spatial.size(); ++d) nvox *= spatial[d];
    const int64_t fnum = nvox * C, hnum = nvox * K;

    std::vector<scalar_t> xtrue(fnum), hes(hnum, 0), grd(fnum, 0), Lx(fnum, 0);
    for (int64_t i = 0; i < fnum; ++i)
        xtrue[i] = (scalar_t)(std::sin(0.37 * (double)i + 0.21)
                              + 0.4 * std::cos(0.11 * (double)i));
    fill_hessian(hes, nvox, C, hscale);

    const double* ap = absolute.data();
    const double* mp = (order >= 2) ? membrane.data() : nullptr;
    const double* bp = (order >= 3) ? bending.data()  : nullptr;

    DLTensor txt  = make_cpu_tensor(xtrue.data(), fshape, fstr, bits);
    DLTensor thes = make_cpu_tensor(hes.data(),   hshape, hstr, bits);
    DLTensor tgrd = make_cpu_tensor(grd.data(),   fshape, fstr, bits);
    DLTensor tLx  = make_cpu_tensor(Lx.data(),    fshape, fstr, bits);

    // g = H x_true + L x_true, built from the two operators separately (not
    // from field_forward) so the right-hand side does not go through the same
    // composition the solver uses to apply the operator.
    ff::cpu::sym_matvec(tgrd, thes, txt, 0);
    ff::cpu::field_matvec(tLx, txt, nullptr, ap, mp, bp, (int8_t)bound, ndim, 0);
    for (int64_t i = 0; i < fnum; ++i)
        grd[i] = (scalar_t)((double)grd[i] + (double)Lx[i]);

    // Warm start: either cold (zeros) or a coarse perturbation of the answer.
    std::vector<scalar_t> sol(fnum, 0);
    if (warm_start)
        for (int64_t i = 0; i < fnum; ++i)
            sol[i] = (scalar_t)((double)xtrue[i]
                                + 0.25 * std::sin(0.9 * (double)i));
    DLTensor tsol = make_cpu_tensor(sol.data(), fshape, fstr, bits);

    int    iters = -1;
    double resid = -1;
    ff::cpu::field_cg(tsol, thes, tgrd, nullptr, ap, mp, bp, (int8_t)bound,
                      ndim, nb_iter, tol, &iters, &resid, 0);

    // 1. the solver recovered the manufactured solution
    double err = 0, nrm = 0;
    for (int64_t i = 0; i < fnum; ++i) {
        double d = (double)sol[i] - (double)xtrue[i];
        err += d * d;
        nrm += (double)xtrue[i] * (double)xtrue[i];
    }
    const double rel_err = std::sqrt(err / nrm);

    // 2. the residual it reported matches one recomputed from the solution
    std::vector<scalar_t> Ax(fnum, 0), Lax(fnum, 0);
    DLTensor tAx  = make_cpu_tensor(Ax.data(),  fshape, fstr, bits);
    DLTensor tLax = make_cpu_tensor(Lax.data(), fshape, fstr, bits);
    ff::cpu::sym_matvec(tAx, thes, tsol, 0);
    ff::cpu::field_matvec(tLax, tsol, nullptr, ap, mp, bp, (int8_t)bound, ndim, 0);
    double res = 0, gnrm = 0;
    for (int64_t i = 0; i < fnum; ++i) {
        double r = (double)grd[i] - ((double)Ax[i] + (double)Lax[i]);
        res  += r * r;
        gnrm += (double)grd[i] * (double)grd[i];
    }
    const double rel_res = std::sqrt(res / gnrm);

    std::printf("  %-46s iters=%3d resid=%.3e err=%.3e\n",
                label, iters, resid, rel_err);

    std::string what;
    what = std::string("cg_solution[") + label + "]";
    check_close(rel_err, 0.0, what.c_str(), xtol);
    what = std::string("cg_reported_residual[") + label + "]";
    check_close(resid, rel_res, what.c_str(), 1e-3);
    what = std::string("cg_iters_bounded[") + label + "]";
    check_true(iters >= 0 && iters <= nb_iter, what.c_str());
}

// A converged CG solution must be a fixed point: solving again from it must
// terminate immediately (zero iterations) rather than wandering off.
template <typename scalar_t>
void run_cg_fixed_point(int64_t N, int64_t C, uint8_t bits, int bound)
{
    const int64_t K = C * (C + 1) / 2;
    std::vector<int64_t> fshape = {N, N, C}, hshape = {N, N, K};
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    std::vector<int64_t> hstr = contiguous_strides(hshape);
    const int64_t nvox = N * N, fnum = nvox * C, hnum = nvox * K;

    std::vector<double> absolute(C, 0.4), membrane(C, 1.0);
    std::vector<scalar_t> sol(fnum, 0), grd(fnum), hes(hnum, 0);
    for (int64_t i = 0; i < fnum; ++i)
        grd[i] = (scalar_t)std::sin(0.4 * (double)i + 0.2);
    fill_hessian(hes, nvox, C, 3.0);

    DLTensor tsol = make_cpu_tensor(sol.data(), fshape, fstr, bits);
    DLTensor thes = make_cpu_tensor(hes.data(), hshape, hstr, bits);
    DLTensor tgrd = make_cpu_tensor(grd.data(), fshape, fstr, bits);

    int it1 = -1, it2 = -1;
    double r1 = -1, r2 = -1;
    ff::cpu::field_cg(tsol, thes, tgrd, nullptr, absolute.data(),
                      membrane.data(), nullptr, (int8_t)bound, 2,
                      200, 1e-10, &it1, &r1, 0);
    std::vector<scalar_t> first(sol);
    ff::cpu::field_cg(tsol, thes, tgrd, nullptr, absolute.data(),
                      membrane.data(), nullptr, (int8_t)bound, 2,
                      200, 1e-10, &it2, &r2, 0);

    std::printf("  %-46s iters=%3d then %d\n", "2d_fixed_point", it1, it2);
    check_true(it1 > 0, "cg_fixed_point_first_pass_iterated");
    check_true(it2 == 0, "cg_fixed_point_restart_is_noop");
    for (int64_t i = 0; i < fnum; ++i)
        check_close((double)sol[i], (double)first[i], "cg_fixed_point_stable");
}

// nb_iter <= 0 must leave the warm start untouched (and report 0 iterations),
// and a capped nb_iter must stop exactly there.
template <typename scalar_t>
void run_cg_iteration_cap(int64_t N, int64_t C, uint8_t bits)
{
    const int64_t K = C * (C + 1) / 2;
    std::vector<int64_t> fshape = {N, N, C}, hshape = {N, N, K};
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    std::vector<int64_t> hstr = contiguous_strides(hshape);
    const int64_t nvox = N * N, fnum = nvox * C, hnum = nvox * K;

    std::vector<double> absolute(C, 0.2), membrane(C, 1.0);
    std::vector<scalar_t> sol(fnum), grd(fnum), hes(hnum, 0);
    for (int64_t i = 0; i < fnum; ++i) {
        grd[i] = (scalar_t)std::sin(0.4 * (double)i + 0.2);
        sol[i] = (scalar_t)(0.5 * std::cos(0.3 * (double)i));
    }
    std::vector<scalar_t> warm(sol);
    fill_hessian(hes, nvox, C, 3.0);

    DLTensor tsol = make_cpu_tensor(sol.data(), fshape, fstr, bits);
    DLTensor thes = make_cpu_tensor(hes.data(), hshape, hstr, bits);
    DLTensor tgrd = make_cpu_tensor(grd.data(), fshape, fstr, bits);

    int it = -1;
    double rs = -1;
    ff::cpu::field_cg(tsol, thes, tgrd, nullptr, absolute.data(),
                      membrane.data(), nullptr, (int8_t)B_DCT2, 2,
                      0, 1e-10, &it, &rs, 0);
    check_true(it == 0, "cg_zero_iter_reports_zero");
    for (int64_t i = 0; i < fnum; ++i)
        check_close((double)sol[i], (double)warm[i], "cg_zero_iter_untouched");
    check_true(rs > 0, "cg_zero_iter_still_reports_residual");

    // tol = 0 disables the early exit: the cap is what stops it.
    it = -1;
    ff::cpu::field_cg(tsol, thes, tgrd, nullptr, absolute.data(),
                      membrane.data(), nullptr, (int8_t)B_DCT2, 2,
                      3, 0.0, &it, &rs, 0);
    check_true(it == 3, "cg_iteration_cap_respected");
}

// The solver must not assume compact row-major buffers: run it on a channel-
// major view (strides permuted) and compare against the contiguous answer.
template <typename scalar_t>
void run_cg_strided(int64_t N, int64_t C, uint8_t bits, int bound)
{
    const int64_t K = C * (C + 1) / 2;
    std::vector<int64_t> fshape = {N, N, C}, hshape = {N, N, K};
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    std::vector<int64_t> hstr = contiguous_strides(hshape);
    const int64_t nvox = N * N, fnum = nvox * C, hnum = nvox * K;

    std::vector<double> absolute(C, 0.3), membrane(C, 0.8);
    std::vector<scalar_t> grd(fnum), hes(hnum, 0);
    for (int64_t i = 0; i < fnum; ++i)
        grd[i] = (scalar_t)std::sin(0.4 * (double)i + 0.2);
    fill_hessian(hes, nvox, C, 3.0);

    DLTensor thes = make_cpu_tensor(hes.data(), hshape, hstr, bits);
    DLTensor tgrd = make_cpu_tensor(grd.data(), fshape, fstr, bits);

    std::vector<scalar_t> ref(fnum, 0);
    DLTensor tref = make_cpu_tensor(ref.data(), fshape, fstr, bits);
    ff::cpu::field_cg(tref, thes, tgrd, nullptr, absolute.data(),
                      membrane.data(), nullptr, (int8_t)bound, 2,
                      200, 1e-10, nullptr, nullptr, 0);

    // Same (N, N, C) logical tensor, physically stored channel-major as
    // (C, N, N): element (y, x, c) lives at c*N*N + y*N + x. That makes the
    // channel stride N*N rather than 1, which is what the dot/axpby element
    // loops read out of `stride[nall]`.
    std::vector<scalar_t> perm(fnum, 0);
    std::vector<int64_t>  pstr = {N, 1, N * N};
    DLTensor tperm = make_cpu_tensor(perm.data(), fshape, pstr, bits);
    ff::cpu::field_cg(tperm, thes, tgrd, nullptr, absolute.data(),
                      membrane.data(), nullptr, (int8_t)bound, 2,
                      200, 1e-10, nullptr, nullptr, 0);

    for (int64_t y = 0; y < N; ++y)
        for (int64_t x = 0; x < N; ++x)
            for (int64_t c = 0; c < C; ++c)
                check_close((double)perm[c * N * N + y * N + x],
                            (double)ref[(y * N + x) * C + c],
                            "cg_strided_matches_contiguous", 1e-4);
}

void test_bad_args_throw()
{
    std::vector<int64_t> fshape = {4, 4, 2}, hshape = {4, 4, 3};
    std::vector<int64_t> fstr = contiguous_strides(fshape);
    std::vector<int64_t> hstr = contiguous_strides(hshape);
    std::vector<double> sol(32, 0), grd(32, 0), hes(48, 1);
    std::vector<double> absolute(2, 1.0);

    DLTensor tsol = make_cpu_tensor(sol.data(), fshape, fstr, 64);
    DLTensor thes = make_cpu_tensor(hes.data(), hshape, hstr, 64);
    DLTensor tgrd = make_cpu_tensor(grd.data(), fshape, fstr, 64);

    bool threw = false;
    try {
        // ndim larger than the tensor rank
        ff::cpu::field_cg(tsol, thes, tgrd, nullptr, absolute.data(), nullptr,
                          nullptr, (int8_t)B_DCT2, 3, 10, 1e-8, nullptr,
                          nullptr, 0);
    } catch (const std::invalid_argument&) { threw = true; }
    check_true(threw, "cg_rejects_ndim_over_rank");

    threw = false;
    try {
        // integer dtype
        DLTensor tbad = tsol;
        tbad.dtype.code = (uint8_t)kDLInt;
        DLTensor tbadh = thes; tbadh.dtype.code = (uint8_t)kDLInt;
        DLTensor tbadg = tgrd; tbadg.dtype.code = (uint8_t)kDLInt;
        ff::cpu::field_cg(tbad, tbadh, tbadg, nullptr, absolute.data(), nullptr,
                          nullptr, (int8_t)B_DCT2, 2, 10, 1e-8, nullptr,
                          nullptr, 0);
    } catch (const std::invalid_argument&) { threw = true; }
    check_true(threw, "cg_rejects_integer_dtype");

    threw = false;
    try {
        // Hessian trailing dimension is not C*(C+1)/2
        std::vector<int64_t> wshape = {4, 4, 2};
        std::vector<int64_t> wstr = contiguous_strides(wshape);
        DLTensor twrong = make_cpu_tensor(hes.data(), wshape, wstr, 64);
        ff::cpu::field_cg(tsol, twrong, tgrd, nullptr, absolute.data(), nullptr,
                          nullptr, (int8_t)B_DCT2, 2, 10, 1e-8, nullptr,
                          nullptr, 0);
    } catch (const std::invalid_argument&) { threw = true; }
    check_true(threw, "cg_rejects_bad_hessian_width");
}

} // namespace

int main()
{
    std::printf("solve_field module CPU tests\n");

    test_bad_args_throw();

    // --- 2D, the three penalty orders, DCT2 --------------------------------
    std::printf(" 2D (8x8), DCT2:\n");
    run_cg<double>("2d_absolute[C=2]", 2, {8, 8}, 2, 1,
                   {0.7, 0.4}, {0, 0}, {0, 0}, 64, B_DCT2, 3.0,
                   200, 1e-10, 1e-6);
    run_cg<double>("2d_membrane[C=2]", 2, {8, 8}, 2, 2,
                   {0.3, 0.2}, {1.0, 0.7}, {0, 0}, 64, B_DCT2, 3.0,
                   200, 1e-10, 1e-6);
    run_cg<double>("2d_bending[C=2]", 2, {8, 8}, 2, 3,
                   {0.3, 0.2}, {0.5, 0.6}, {1.0, 0.8}, 64, B_DCT2, 3.0,
                   200, 1e-10, 1e-6);
    // membrane with no absolute term: L alone is singular (constants are in
    // its null space under DCT2), the Hessian is what makes (H + L) definite.
    run_cg<double>("2d_membrane_no_absolute[C=2]", 2, {8, 8}, 2, 2,
                   {0.0, 0.0}, {1.0, 1.0}, {0, 0}, 64, B_DCT2, 3.0,
                   200, 1e-10, 1e-6);

    // --- boundary conditions ----------------------------------------------
    std::printf(" 2D (8x8), boundary conditions:\n");
    for (int bnd : {B_ZERO, B_DCT2, B_DST2, B_DFT}) {
        char buf[80];
        std::snprintf(buf, sizeof(buf), "2d_membrane[C=2 bound=%d]", bnd);
        run_cg<double>(buf, 2, {8, 8}, 2, 2, {0.3, 0.2}, {1.0, 0.7}, {0, 0},
                       64, bnd, 3.0, 200, 1e-10, 1e-6);
    }
    for (int bnd : {B_ZERO, B_DCT2, B_DFT}) {
        char buf[80];
        std::snprintf(buf, sizeof(buf), "2d_bending[C=1 bound=%d]", bnd);
        run_cg<double>(buf, 2, {8, 8}, 1, 3, {0.2}, {0.5}, {1.0},
                       64, bnd, 3.0, 200, 1e-10, 1e-6);
    }

    // --- 1D / 3D, channel counts, dtypes -----------------------------------
    std::printf(" 1D / 3D / dtypes:\n");
    run_cg<double>("1d_membrane[C=1]", 1, {32}, 1, 2, {0.5}, {1.0}, {0},
                   64, B_DCT2, 3.0, 200, 1e-10, 1e-6);
    run_cg<double>("1d_bending[C=3]", 1, {24}, 3, 3,
                   {0.2, 0.3, 0.4}, {0.5, 0.6, 0.7}, {1.0, 0.9, 0.8},
                   64, B_DCT2, 4.0, 300, 1e-10, 1e-6);
    run_cg<double>("3d_membrane[C=2]", 3, {8, 8, 8}, 2, 2,
                   {0.3, 0.2}, {1.0, 0.7}, {0, 0}, 64, B_DCT2, 3.0,
                   300, 1e-10, 1e-6);
    run_cg<double>("3d_bending[C=1]", 3, {6, 6, 6}, 1, 3,
                   {0.2}, {0.5}, {1.0}, 64, B_DCT2, 3.0, 300, 1e-10, 1e-6);
    run_cg<float >("2d_membrane_float[C=2]", 2, {8, 8}, 2, 2,
                   {0.3, 0.2}, {1.0, 0.7}, {0, 0}, 32, B_DCT2, 3.0,
                   200, 1e-6, 2e-3);

    // --- stiff systems (weak Hessian, strong regulariser) ------------------
    std::printf(" stiff systems:\n");
    run_cg<double>("2d_stiff_membrane[C=1 h=0.05]", 2, {8, 8}, 1, 2,
                   {0.01}, {10.0}, {0}, 64, B_DCT2, 0.05, 400, 1e-10, 1e-5);
    run_cg<double>("2d_stiff_bending[C=1 h=0.05]", 2, {8, 8}, 1, 3,
                   {0.01}, {1.0}, {10.0}, 64, B_DCT2, 0.05, 600, 1e-10, 1e-5);

    // --- warm start --------------------------------------------------------
    std::printf(" warm start:\n");
    run_cg<double>("2d_membrane_warm[C=2]", 2, {8, 8}, 2, 2,
                   {0.3, 0.2}, {1.0, 0.7}, {0, 0}, 64, B_DCT2, 3.0,
                   200, 1e-10, 1e-6, /*warm_start=*/true);
    run_cg<double>("3d_bending_warm[C=1]", 3, {6, 6, 6}, 1, 3,
                   {0.2}, {0.5}, {1.0}, 64, B_DCT2, 3.0, 300, 1e-10, 1e-6,
                   /*warm_start=*/true);

    // --- contracts ---------------------------------------------------------
    std::printf(" contracts:\n");
    run_cg_fixed_point<double>(8, 2, 64, B_DCT2);
    run_cg_iteration_cap<double>(8, 2, 64);
    run_cg_strided<double>(8, 2, 64, B_DCT2);

    std::printf("%d/%d checks passed\n", g_checks - g_failures, g_checks);
    if (g_failures) {
        std::printf("FAILED (%d)\n", g_failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
