// CPU unit tests for the posdef module.
//
// Exercises the full stack (dispatch -> impl -> kernels) for the compact
// symmetric ("Sym") Hessian operations: sym_matvec, sym_addmatvec_,
// sym_submatvec_, sym_matvec_backward, sym_solve[_], sym_invert[_].
//
// The compact-symmetric layout is "diagonal-then-rows":
//   C=2 -> [h00, h11, h01]
//   C=3 -> [h00, h11, h22, h01, h02, h12]
//
// Build (from fastfields-cpu-lib):
//   clang++ -std=c++11 -O2 -I. tests/test_posdef.cpp posdef.cpp -o build/test_posdef
//   ./build/test_posdef

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include "dlpack.h"
#include "posdef.h"

namespace {

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

DLTensor null_tensor()
{
    DLTensor t;
    t.data               = nullptr;
    t.device.device_type = kDLCPU;
    t.device.device_id   = 0;
    t.ndim               = 0;
    t.dtype.code         = static_cast<uint8_t>(kDLFloat);
    t.dtype.bits         = 32;
    t.dtype.lanes        = 1;
    t.shape              = nullptr;
    t.strides            = nullptr;
    t.byte_offset        = 0;
    return t;
}

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> s(shape.size());
    int64_t acc = 1;
    for (int64_t d = (int64_t)shape.size() - 1; d >= 0; --d) {
        s[d] = acc;
        acc *= shape[d];
    }
    return s;
}

// pack a full symmetric CxC matrix (row-major) into the compact layout
void pack_sym(const std::vector<double>& full, int C, std::vector<double>& packed)
{
    packed.clear();
    for (int c = 0; c < C; ++c) packed.push_back(full[c*C + c]);       // diagonal
    for (int c = 0; c < C; ++c)
        for (int cc = c+1; cc < C; ++cc)
            packed.push_back(full[c*C + cc]);                          // upper rows
}

int g_failures = 0;
int g_checks   = 0;

void check_close(double a, double b, const char* what, double tol = 1e-3)
{
    ++g_checks;
    double diff  = std::fabs(a - b);
    double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    if (diff / scale > tol) {
        ++g_failures;
        std::printf("  MISMATCH [%s]: got %.8g expected %.8g\n", what, a, b);
    }
}

template <typename scalar_t>
void run_case(int C, int64_t nbatch, uint8_t bits, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> off(-0.5, 0.5);
    std::uniform_real_distribution<double> vec(-2.0, 2.0);

    const int CC = C*(C+1)/2;

    // reference (double precision) data, per batch element
    std::vector<std::vector<double>> full(nbatch);   // CxC symmetric SPD
    std::vector<std::vector<double>> x(nbatch);       // vector
    std::vector<std::vector<double>> y(nbatch);       // second vector (for add/sub)

    // device buffers (scalar_t)
    std::vector<scalar_t> hbuf(nbatch * CC);
    std::vector<scalar_t> xbuf(nbatch * C);
    std::vector<scalar_t> bbuf(nbatch * C);  // outputs / matvec result

    for (int64_t bt = 0; bt < nbatch; ++bt) {
        full[bt].assign(C*C, 0.0);
        // symmetric, diagonally dominant -> SPD
        for (int c = 0; c < C; ++c)
            for (int cc = c; cc < C; ++cc) {
                double v = (c == cc) ? (C + 2.0 + off(rng)) : off(rng);
                full[bt][c*C + cc] = v;
                full[bt][cc*C + c] = v;
            }
        x[bt].resize(C);
        y[bt].resize(C);
        for (int c = 0; c < C; ++c) { x[bt][c] = vec(rng); y[bt][c] = vec(rng); }

        std::vector<double> packed;
        pack_sym(full[bt], C, packed);
        for (int k = 0; k < CC; ++k) hbuf[bt*CC + k] = static_cast<scalar_t>(packed[k]);
        for (int c = 0; c < C; ++c)  xbuf[bt*C + c]  = static_cast<scalar_t>(x[bt][c]);
    }

    std::vector<int64_t> vshape = {nbatch, (int64_t)C};
    std::vector<int64_t> vstr   = contiguous_strides(vshape);
    std::vector<int64_t> hshape = {nbatch, (int64_t)CC};
    std::vector<int64_t> hstr   = contiguous_strides(hshape);

    DLTensor H = make_cpu_tensor(hbuf.data(), hshape, hstr, bits);
    DLTensor X = make_cpu_tensor(xbuf.data(), vshape, vstr, bits);
    DLTensor B = make_cpu_tensor(bbuf.data(), vshape, vstr, bits);
    DLTensor W = null_tensor();

    // --- matvec: B = H * X ---
    ff::cpu::sym_matvec(B, H, X, 0);
    for (int64_t bt = 0; bt < nbatch; ++bt)
        for (int c = 0; c < C; ++c) {
            double ref = 0.0;
            for (int cc = 0; cc < C; ++cc) ref += full[bt][c*C + cc] * x[bt][cc];
            check_close((double)bbuf[bt*C + c], ref, "matvec");
        }

    // --- solve: X2 = H \ B  (should recover x) ---
    std::vector<scalar_t> x2buf(nbatch * C);
    DLTensor X2 = make_cpu_tensor(x2buf.data(), vshape, vstr, bits);
    ff::cpu::sym_solve(X2, H, B, W, 0);
    for (int64_t bt = 0; bt < nbatch; ++bt)
        for (int c = 0; c < C; ++c)
            check_close((double)x2buf[bt*C + c], x[bt][c], "solve");

    // --- solve_ in place: buf = B, then buf = H \ buf ---
    std::vector<scalar_t> sbuf = bbuf;
    DLTensor S = make_cpu_tensor(sbuf.data(), vshape, vstr, bits);
    ff::cpu::sym_solve_(S, H, W, 0);
    for (int64_t bt = 0; bt < nbatch; ++bt)
        for (int c = 0; c < C; ++c)
            check_close((double)sbuf[bt*C + c], x[bt][c], "solve_");

    // --- invert: Hinv (packed) then matvec(Hinv, B) should recover x ---
    std::vector<scalar_t> hinv(nbatch * CC);
    DLTensor Hinv = make_cpu_tensor(hinv.data(), hshape, hstr, bits);
    ff::cpu::sym_invert(Hinv, H, 0);
    std::vector<scalar_t> zbuf(nbatch * C);
    DLTensor Z = make_cpu_tensor(zbuf.data(), vshape, vstr, bits);
    ff::cpu::sym_matvec(Z, Hinv, B, 0);
    for (int64_t bt = 0; bt < nbatch; ++bt)
        for (int c = 0; c < C; ++c)
            check_close((double)zbuf[bt*C + c], x[bt][c], "invert+matvec");

    // --- invert_ in place should match invert ---
    std::vector<scalar_t> hinv2(hbuf.begin(), hbuf.end());
    DLTensor Hinv2 = make_cpu_tensor(hinv2.data(), hshape, hstr, bits);
    ff::cpu::sym_invert_(Hinv2, 0);
    for (int64_t bt = 0; bt < nbatch; ++bt)
        for (int k = 0; k < CC; ++k)
            check_close((double)hinv2[bt*CC + k], (double)hinv[bt*CC + k], "invert_");

    // --- addmatvec_: out = Y, then out += H * X ---
    std::vector<scalar_t> abuf(nbatch * C);
    for (int64_t bt = 0; bt < nbatch; ++bt)
        for (int c = 0; c < C; ++c) abuf[bt*C + c] = static_cast<scalar_t>(y[bt][c]);
    DLTensor A = make_cpu_tensor(abuf.data(), vshape, vstr, bits);
    ff::cpu::sym_addmatvec_(A, H, X, 0);
    for (int64_t bt = 0; bt < nbatch; ++bt)
        for (int c = 0; c < C; ++c) {
            double ref = y[bt][c];
            for (int cc = 0; cc < C; ++cc) ref += full[bt][c*C + cc] * x[bt][cc];
            check_close((double)abuf[bt*C + c], ref, "addmatvec_");
        }

    // --- submatvec_: out = Y, then out -= H * X ---
    std::vector<scalar_t> ubuf(nbatch * C);
    for (int64_t bt = 0; bt < nbatch; ++bt)
        for (int c = 0; c < C; ++c) ubuf[bt*C + c] = static_cast<scalar_t>(y[bt][c]);
    DLTensor U = make_cpu_tensor(ubuf.data(), vshape, vstr, bits);
    ff::cpu::sym_submatvec_(U, H, X, 0);
    for (int64_t bt = 0; bt < nbatch; ++bt)
        for (int c = 0; c < C; ++c) {
            double ref = y[bt][c];
            for (int cc = 0; cc < C; ++cc) ref -= full[bt][c*C + cc] * x[bt][cc];
            check_close((double)ubuf[bt*C + c], ref, "submatvec_");
        }

    // --- matvec_backward: G(CC) from grad=Y, inp=X ---
    // diag c   : y[c]*x[c]
    // off (c,cc): y[c]*x[cc] + y[cc]*x[c]
    std::vector<scalar_t> gbuf(nbatch * CC);
    for (int64_t bt = 0; bt < nbatch; ++bt)
        for (int c = 0; c < C; ++c) xbuf[bt*C + c] = static_cast<scalar_t>(x[bt][c]);
    std::vector<scalar_t> ybuf(nbatch * C);
    for (int64_t bt = 0; bt < nbatch; ++bt)
        for (int c = 0; c < C; ++c) ybuf[bt*C + c] = static_cast<scalar_t>(y[bt][c]);
    DLTensor G  = make_cpu_tensor(gbuf.data(), hshape, hstr, bits);
    DLTensor Yv = make_cpu_tensor(ybuf.data(), vshape, vstr, bits);
    ff::cpu::sym_matvec_backward(G, Yv, X, 0);
    for (int64_t bt = 0; bt < nbatch; ++bt) {
        std::vector<double> ref;
        for (int c = 0; c < C; ++c) ref.push_back(y[bt][c]*x[bt][c]);
        for (int c = 0; c < C; ++c)
            for (int cc = c+1; cc < C; ++cc)
                ref.push_back(y[bt][c]*x[bt][cc] + y[bt][cc]*x[bt][c]);
        for (int k = 0; k < CC; ++k)
            check_close((double)gbuf[bt*CC + k], ref[k], "matvec_backward");
    }
}

} // namespace

int main()
{
    std::printf("posdef module CPU tests\n");
    for (unsigned seed = 1; seed <= 5; ++seed) {
        run_case<float >(2, 7,  32, seed);
        run_case<double>(2, 7,  64, seed + 100);
        run_case<float >(3, 5,  32, seed + 200);
        run_case<double>(3, 5,  64, seed + 300);
        run_case<float >(1, 4,  32, seed + 400);
        run_case<double>(4, 3,  64, seed + 500);  // dynamic C=-1 path
    }
    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
