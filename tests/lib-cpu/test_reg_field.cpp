// CPU unit tests for the reg_field module.
//
// Exercises ff::cpu::field_matvec / field_diag against hand-written references:
//   * absolute-only matvec == per-channel elementwise scaling (exact)
//   * membrane-only matvec == membrane[c] * discrete negative Laplacian (exact,
//     with explicit Zero and DCT2 boundary references)
//   * diag == matvec on unit vectors in the interior
//
// Build (from fastfields-cpu-lib):
//   clang++ -std=c++11 -O2 -ferror-limit=5 -I. tests/test_reg_field.cpp reg_field.cpp -o build/test_reg_field
//   ./build/test_reg_field

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "dlpack.h"
#include "reg_field.h"

namespace {

enum { B_ZERO = 0, B_DCT2 = 3 };

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

double neigh_1d(const std::vector<double>& f, int64_t idx, int bound)
{
    int64_t n = (int64_t)f.size();
    if (idx >= 0 && idx < n) return f[idx];
    if (bound == B_ZERO) return 0.0;
    int64_t r = idx;
    if (r < 0)  r = -r - 1;
    if (r >= n) r = 2*n - r - 1;
    if (r < 0)  r = 0;
    if (r >= n) r = n - 1;
    return f[r];
}

// 1D field with C channels, contiguous (N, C).
template <typename scalar_t>
void run_1d(int64_t N, int64_t C, const std::vector<double>& absolute,
            const std::vector<double>& membrane, bool use_membrane,
            int bound, uint8_t bits)
{
    std::vector<int64_t> shape = {N, C};
    std::vector<int64_t> str   = contiguous_strides(shape);
    int64_t numel = N * C;
    std::vector<scalar_t> inp(numel), out(numel, scalar_t(-777));
    // per-channel field
    std::vector<std::vector<double>> f(C, std::vector<double>(N));
    for (int64_t x = 0; x < N; ++x)
        for (int64_t c = 0; c < C; ++c) {
            double v = std::sin(0.4 * x + 0.7 * c + 1.0) + 0.3 * x - 0.2 * c;
            inp[x * C + c] = (scalar_t)v;
            f[c][x] = v;
        }

    DLTensor tin  = make_cpu_tensor(inp.data(), shape, str, bits);
    DLTensor tout = make_cpu_tensor(out.data(), shape, str, bits);

    const double* mem = use_membrane ? membrane.data() : nullptr;
    ff::cpu::field_matvec(tout, tin, nullptr, absolute.data(), mem, nullptr,
                          (int8_t)bound, 1, 0);

    for (int64_t x = 0; x < N; ++x)
        for (int64_t c = 0; c < C; ++c) {
            double center = f[c][x];
            double expect = absolute[c] * center;
            if (use_membrane) {
                double L = neigh_1d(f[c], x-1, bound);
                double R = neigh_1d(f[c], x+1, bound);
                expect += membrane[c] * (2*center - L - R);
            }
            check_close((double)out[x * C + c], expect, "field1d_matvec");
        }
}

template <typename scalar_t>
void run_1d_diag(int64_t N, int64_t C, const std::vector<double>& absolute,
                 const std::vector<double>& membrane, int bound, uint8_t bits)
{
    std::vector<int64_t> shape = {N, C};
    std::vector<int64_t> str   = contiguous_strides(shape);
    int64_t numel = N * C;

    std::vector<scalar_t> diag(numel, 0);
    DLTensor td = make_cpu_tensor(diag.data(), shape, str, bits);
    ff::cpu::field_diag(td, nullptr, absolute.data(), membrane.data(), nullptr,
                        (int8_t)bound, 1, 0);

    // interior only (see note in test_reg_flow.cpp about the boundary convention)
    for (int64_t x = 1; x < N - 1; ++x)
        for (int64_t c = 0; c < C; ++c) {
            std::vector<scalar_t> e(numel, 0), o(numel, 0);
            e[x * C + c] = 1;
            DLTensor te = make_cpu_tensor(e.data(), shape, str, bits);
            DLTensor to = make_cpu_tensor(o.data(), shape, str, bits);
            ff::cpu::field_matvec(to, te, nullptr, absolute.data(), membrane.data(),
                                  nullptr, (int8_t)bound, 1, 0);
            check_close((double)diag[x * C + c], (double)o[x * C + c], "field1d_diag_interior");
        }
}

// 2D field (H, W, C), absolute-only exact scaling.
template <typename scalar_t>
void run_2d_absolute(int64_t H, int64_t W, int64_t C,
                     const std::vector<double>& absolute, uint8_t bits)
{
    std::vector<int64_t> shape = {H, W, C};
    std::vector<int64_t> str   = contiguous_strides(shape);
    int64_t numel = H * W * C;
    std::vector<scalar_t> inp(numel), out(numel, scalar_t(-1));
    for (int64_t i = 0; i < numel; ++i) inp[i] = (scalar_t)(0.05 * i - 2.0);

    DLTensor tin  = make_cpu_tensor(inp.data(), shape, str, bits);
    DLTensor tout = make_cpu_tensor(out.data(), shape, str, bits);
    ff::cpu::field_matvec(tout, tin, nullptr, absolute.data(), nullptr, nullptr,
                          (int8_t)B_DCT2, 2, 0);

    for (int64_t i = 0; i < numel; ++i) {
        int64_t c = i % C;
        check_close((double)out[i], absolute[c] * (double)inp[i], "field2d_abs");
    }
}

// 1D field bending interior == bending[c] * biharmonic stencil [1,-4,6,-4,1].
template <typename scalar_t>
void run_1d_bending(int64_t N, int64_t C, const std::vector<double>& bending, uint8_t bits)
{
    std::vector<int64_t> shape = {N, C};
    std::vector<int64_t> str   = contiguous_strides(shape);
    int64_t numel = N * C;
    std::vector<scalar_t> inp(numel), out(numel, scalar_t(-1));
    std::vector<double> zero(C, 0.0);
    std::vector<std::vector<double>> f(C, std::vector<double>(N));
    for (int64_t x = 0; x < N; ++x)
        for (int64_t c = 0; c < C; ++c) {
            double v = std::sin(0.35*x + 0.6*c + 0.2);
            inp[x*C+c] = (scalar_t)v; f[c][x] = v;
        }
    DLTensor ti = make_cpu_tensor(inp.data(), shape, str, bits);
    DLTensor to = make_cpu_tensor(out.data(), shape, str, bits);
    ff::cpu::field_matvec(to, ti, nullptr, zero.data(), zero.data(), bending.data(),
                          (int8_t)B_DCT2, 1, 0);
    for (int64_t x = 2; x < N-2; ++x)
        for (int64_t c = 0; c < C; ++c) {
            double e = bending[c] * (f[c][x-2] - 4*f[c][x-1] + 6*f[c][x]
                                     - 4*f[c][x+1] + f[c][x+2]);
            check_close((double)out[x*C+c], e, "field1d_bending");
        }
}

// 3D field membrane interior == membrane[c] * (6*center - 6 face-neighbours).
template <typename scalar_t>
void run_3d_membrane(int64_t N, int64_t C, const std::vector<double>& membrane, uint8_t bits)
{
    std::vector<int64_t> shape = {N, N, N, C};
    std::vector<int64_t> str   = contiguous_strides(shape);
    int64_t numel = N * N * N * C;
    std::vector<scalar_t> inp(numel), out(numel, scalar_t(-1));
    std::vector<double> absolute(C, 0.0);
    auto idx = [&](int64_t x,int64_t y,int64_t z,int64_t c){ return ((x*N+y)*N+z)*C+c; };
    for (int64_t i = 0; i < numel; ++i) inp[i] = (scalar_t)std::sin(0.17*i + 0.3);

    DLTensor ti = make_cpu_tensor(inp.data(), shape, str, bits);
    DLTensor to = make_cpu_tensor(out.data(), shape, str, bits);
    ff::cpu::field_matvec(to, ti, nullptr, absolute.data(), membrane.data(), nullptr,
                          (int8_t)B_DCT2, 3, 0);

    for (int64_t x = 1; x < N-1; ++x)
    for (int64_t y = 1; y < N-1; ++y)
    for (int64_t z = 1; z < N-1; ++z)
    for (int64_t c = 0; c < C; ++c) {
        double ctr = (double)inp[idx(x,y,z,c)];
        double s = (double)inp[idx(x-1,y,z,c)] + (double)inp[idx(x+1,y,z,c)]
                 + (double)inp[idx(x,y-1,z,c)] + (double)inp[idx(x,y+1,z,c)]
                 + (double)inp[idx(x,y,z-1,c)] + (double)inp[idx(x,y,z+1,c)];
        check_close((double)out[idx(x,y,z,c)], membrane[c]*(6*ctr - s), "field3d_membrane");
    }
}

// --- B4: negative / validation tests --------------------------------------
// Bad dtype: float16 must throw at the dtype dispatch, not silently no-op.
void test_bad_dtype_throws()
{
    const int64_t N = 6, C = 2;
    std::vector<uint16_t> inp(N*C,0), out(N*C,0);          // float16 payload
    std::vector<double> absolute(C, 1.0);
    std::vector<int64_t> sh={N,C}, st=contiguous_strides(sh);
    DLTensor tin =make_cpu_tensor(inp.data(),sh,st,16);
    DLTensor tout=make_cpu_tensor(out.data(),sh,st,16);
    bool threw = false;
    try { ff::cpu::field_matvec(tout, tin, nullptr, absolute.data(), nullptr, nullptr,
                                (int8_t)B_DCT2, 1, 0); }
    catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [reg_field.bad_dtype_throws]\n"); }
}

// Shape mismatch: out and inp with different spatial extents must throw.
void test_shape_mismatch_throws()
{
    const int64_t N = 6, C = 2;
    std::vector<double> inp((N+1)*C,0), out(N*C,0);        // differing spatial dim
    std::vector<double> absolute(C, 1.0);
    std::vector<int64_t> shi={N+1,C}, sti=contiguous_strides(shi);
    std::vector<int64_t> sho={N,C},   sto=contiguous_strides(sho);
    DLTensor tin =make_cpu_tensor(inp.data(),shi,sti,64);
    DLTensor tout=make_cpu_tensor(out.data(),sho,sto,64);
    bool threw = false;
    try { ff::cpu::field_matvec(tout, tin, nullptr, absolute.data(), nullptr, nullptr,
                                (int8_t)B_DCT2, 1, 0); }
    catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [reg_field.shape_mismatch_throws]\n"); }
}

// field_kernel: the materialised per-channel stencil must equal the operator's
// impulse response in the interior. Apply field_matvec to a unit impulse at the
// centre of a domain large enough that the response never touches a boundary,
// then compare the centred window to the kernel returned by field_kernel. Field
// channels are independent, so an impulse in channel c0 only affects output
// channel c0. `order` selects which penalty pointers are passed (and the
// stencil width): 1 absolute, 3 membrane, 5 bending.
template <typename scalar_t>
void run_2d_kernel_impulse(int64_t kd, int64_t C, int order,
                           const std::vector<double>& absolute,
                           const std::vector<double>& membrane,
                           const std::vector<double>& bending,
                           uint8_t bits, int bound = B_DCT2)
{
    const double* ap = absolute.data();
    const double* mp = (order >= 2) ? membrane.data() : nullptr;
    const double* bp = (order >= 3) ? bending.data()  : nullptr;

    std::vector<int64_t> kshape = {kd, kd, C};
    std::vector<int64_t> kstr   = contiguous_strides(kshape);
    std::vector<scalar_t> K(kd * kd * C, scalar_t(0));
    DLTensor tK = make_cpu_tensor(K.data(), kshape, kstr, bits);
    ff::cpu::field_kernel(tK, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);

    const int64_t N = 2 * kd + 1, cc = N / 2, half = kd / 2;
    std::vector<int64_t> fshape = {N, N, C};
    std::vector<int64_t> fstr   = contiguous_strides(fshape);
    int64_t fnumel = N * N * C;
    auto fidx = [&](int64_t i, int64_t j, int64_t c){ return (i * N + j) * C + c; };

    for (int64_t c0 = 0; c0 < C; ++c0) {
        std::vector<scalar_t> x(fnumel, scalar_t(0)), o(fnumel, scalar_t(0));
        x[fidx(cc, cc, c0)] = 1;
        DLTensor tx = make_cpu_tensor(x.data(), fshape, fstr, bits);
        DLTensor to = make_cpu_tensor(o.data(), fshape, fstr, bits);
        ff::cpu::field_matvec(to, tx, nullptr, ap, mp, bp, (int8_t)bound, 2, 0);
        for (int64_t a = 0; a < kd; ++a)
        for (int64_t b = 0; b < kd; ++b)
        for (int64_t c = 0; c < C; ++c) {
            double got  = (double)o[fidx(cc + (a - half), cc + (b - half), c)];
            double kern = (c == c0) ? (double)K[(a * kd + b) * C + c] : 0.0;
            char buf[96];
            std::snprintf(buf, sizeof(buf),
                "field2d_kernel_impulse[kd=%lld C=%lld order=%d bnd=%d]",
                (long long)kd, (long long)C, order, bound);
            check_close(got, kern, buf);
        }
    }
}

} // namespace

int main()
{
    std::printf("reg_field module CPU tests\n");
    test_bad_dtype_throws();
    test_shape_mismatch_throws();

    // absolute-only, per-channel scaling, 1 and 2 channels
    run_1d<double>(9, 1, {2.5}, {0.0}, false, B_ZERO, 64);
    run_1d<float >(9, 2, {2.5, -1.5}, {0,0}, false, B_DCT2, 32);

    // membrane-only, negative Laplacian per channel, Zero + DCT2
    run_1d<double>(11, 1, {0.0}, {1.0}, true, B_ZERO, 64);
    run_1d<double>(11, 2, {0.0, 0.0}, {1.0, 2.0}, true, B_DCT2, 64);
    run_1d<float >(13, 2, {0.0, 0.0}, {1.3, 0.7}, true, B_ZERO, 32);

    // combined absolute + membrane
    run_1d<double>(13, 2, {0.7, 0.4}, {1.3, 0.9}, true, B_DCT2, 64);

    // diagonal consistency (interior)
    run_1d_diag<double>(11, 2, {0.7, 0.4}, {1.3, 0.9}, B_DCT2, 64);
    run_1d_diag<double>(11, 1, {0.0}, {1.0}, B_ZERO, 64);

    // 2D absolute
    run_2d_absolute<double>(4, 5, 2, {1.75, -0.5}, 64);
    run_2d_absolute<float >(3, 4, 1, {0.5}, 32);

    // 1D bending (interior biharmonic stencil, per channel)
    run_1d_bending<double>(11, 2, {1.0, 0.5}, 64);
    run_1d_bending<double>(9,  1, {2.0}, 64);

    // 3D membrane (interior negative Laplacian, per channel)
    run_3d_membrane<double>(5, 2, {1.0, 2.5}, 64);
    run_3d_membrane<double>(4, 1, {1.3}, 64);

    // field_kernel: the per-channel stencil == the operator's impulse response.
    run_2d_kernel_impulse<double>(1, 2, 1, {2.5, 1.5}, {0, 0}, {0, 0}, 64);
    run_2d_kernel_impulse<double>(3, 2, 2, {0.3, 0.4}, {1.0, 0.7}, {0, 0}, 64);
    run_2d_kernel_impulse<double>(5, 2, 3, {0.3, 0.4}, {0.5, 0.6},
                                  {1.0, 0.8}, 64);
    run_2d_kernel_impulse<float >(3, 1, 2, {0.0}, {1.0}, {0.0}, 32);
    run_2d_kernel_impulse<double>(3, 2, 2, {0.3, 0.4}, {1.0, 0.7}, {0, 0}, 64,
                                  B_ZERO);

    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
