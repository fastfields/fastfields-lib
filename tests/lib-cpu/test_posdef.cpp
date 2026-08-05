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
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
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

// --- B3: weighted solve -- (H + diag(w)) \ b -------------------------------
// The unweighted run_case always passes W=null_tensor(), so the
// (H+diag(w))\x branch (weight normalization + kernel diagonal augmentation)
// had zero coverage. Here we build a random positive weight vector w, form the
// RHS b = (H + diag(w)) x by brute force (INCLUDING the diagonal weight), and
// verify sym_solve / sym_solve_ recover x.
template <typename scalar_t>
void run_weighted_case(int C, int64_t nbatch, uint8_t bits, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> off(-0.5, 0.5);
    std::uniform_real_distribution<double> vec(-2.0, 2.0);
    std::uniform_real_distribution<double> wd (0.1, 3.0);   // strictly positive weight

    const int CC = C*(C+1)/2;

    std::vector<std::vector<double>> full(nbatch);   // CxC SPD
    std::vector<std::vector<double>> x(nbatch);       // solution
    std::vector<std::vector<double>> w(nbatch);       // per-channel diagonal weight

    std::vector<scalar_t> hbuf(nbatch * CC);
    std::vector<scalar_t> bbuf(nbatch * C);  // RHS = (H + diag(w)) x
    std::vector<scalar_t> wbuf(nbatch * C);

    for (int64_t bt = 0; bt < nbatch; ++bt) {
        full[bt].assign(C*C, 0.0);
        for (int c = 0; c < C; ++c)
            for (int cc = c; cc < C; ++cc) {
                double v = (c == cc) ? (C + 2.0 + off(rng)) : off(rng);
                full[bt][c*C + cc] = v;
                full[bt][cc*C + c] = v;
            }
        x[bt].resize(C);
        w[bt].resize(C);
        for (int c = 0; c < C; ++c) { x[bt][c] = vec(rng); w[bt][c] = wd(rng); }

        std::vector<double> packed;
        pack_sym(full[bt], C, packed);
        for (int k = 0; k < CC; ++k) hbuf[bt*CC + k] = static_cast<scalar_t>(packed[k]);

        // RHS b = (H + diag(w)) x  (brute-force reference, diagonal weight added)
        for (int c = 0; c < C; ++c) {
            double b = 0.0;
            for (int cc = 0; cc < C; ++cc) b += full[bt][c*C + cc] * x[bt][cc];
            b += w[bt][c] * x[bt][c];
            bbuf[bt*C + c] = static_cast<scalar_t>(b);
            wbuf[bt*C + c] = static_cast<scalar_t>(w[bt][c]);
        }
    }

    std::vector<int64_t> vshape = {nbatch, (int64_t)C};
    std::vector<int64_t> vstr   = contiguous_strides(vshape);
    std::vector<int64_t> hshape = {nbatch, (int64_t)CC};
    std::vector<int64_t> hstr   = contiguous_strides(hshape);

    DLTensor H = make_cpu_tensor(hbuf.data(), hshape, hstr, bits);
    DLTensor B = make_cpu_tensor(bbuf.data(), vshape, vstr, bits);
    DLTensor W = make_cpu_tensor(wbuf.data(), vshape, vstr, bits);

    // out-of-place weighted solve: X2 = (H + diag(w)) \ B  == x
    std::vector<scalar_t> x2buf(nbatch * C);
    DLTensor X2 = make_cpu_tensor(x2buf.data(), vshape, vstr, bits);
    ff::cpu::sym_solve(X2, H, B, W, 0);
    for (int64_t bt = 0; bt < nbatch; ++bt)
        for (int c = 0; c < C; ++c)
            check_close((double)x2buf[bt*C + c], x[bt][c], "weighted_solve");

    // in-place weighted solve: buf = B, then buf = (H + diag(w)) \ buf == x
    std::vector<scalar_t> sbuf = bbuf;
    DLTensor S = make_cpu_tensor(sbuf.data(), vshape, vstr, bits);
    ff::cpu::sym_solve_(S, H, W, 0);
    for (int64_t bt = 0; bt < nbatch; ++bt)
        for (int c = 0; c < C; ++c)
            check_close((double)sbuf[bt*C + c], x[bt][c], "weighted_solve_");
}

// --- B2: 64-bit index + non-contiguous stride path -------------------------
// Every other case uses tiny contiguous tensors, so canUse32BitIndexMath()
// always returns true and the int64_t offset_t instantiations + strided-index
// loops are never run. canUse32BitIndexMath keys on the max element OFFSET, so
// giving the hessian a deliberately inflated batch stride (>= INT32_MAX) forces
// use_32bits=false. The buffer is lazily allocated (calloc) and only the two
// batch planes are touched, so RSS stays tiny; if the (virtual) allocation is
// refused we skip rather than fail. scalar_t=float keeps it to ~8.6 GB virtual.
template <typename scalar_t>
void run_inflated_stride_case(int C, uint8_t bits, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> off(-0.5, 0.5);
    std::uniform_real_distribution<double> vec(-2.0, 2.0);

    const int64_t nbatch = 2;
    const int     CC     = C*(C+1)/2;
    // Inflated batch stride: (nbatch-1)*P + (CC-1) >= INT32_MAX -> 64-bit path.
    const int64_t P = (int64_t)1 << 31;                 // 2147483648 > INT32_MAX

    const size_t nelem = (size_t)P * (nbatch - 1) + (size_t)CC;
    scalar_t * hbig = static_cast<scalar_t*>(std::calloc(nelem, sizeof(scalar_t)));
    if (!hbig) {
        std::printf("  [posdef inflated-stride] skipped (lazy alloc of %.1f GB refused)\n",
                    nelem * sizeof(scalar_t) / 1e9);
        return;
    }

    std::vector<std::vector<double>> full(nbatch);
    std::vector<std::vector<double>> x(nbatch);
    std::vector<scalar_t> xbuf(nbatch * C), bbuf(nbatch * C);

    for (int64_t bt = 0; bt < nbatch; ++bt) {
        full[bt].assign(C*C, 0.0);
        for (int c = 0; c < C; ++c)
            for (int cc = c; cc < C; ++cc) {
                double v = (c == cc) ? (C + 2.0 + off(rng)) : off(rng);
                full[bt][c*C + cc] = v;
                full[bt][cc*C + c] = v;
            }
        x[bt].resize(C);
        for (int c = 0; c < C; ++c) x[bt][c] = vec(rng);

        std::vector<double> packed;
        pack_sym(full[bt], C, packed);
        // write the packed hessian at the *inflated* batch offset bt*P
        for (int k = 0; k < CC; ++k) hbig[bt*P + k] = static_cast<scalar_t>(packed[k]);
        for (int c = 0; c < C; ++c)  xbuf[bt*C + c] = static_cast<scalar_t>(x[bt][c]);
    }

    // H tensor: shape [2, CC], strides [P, 1] (non-contiguous, huge offset).
    std::vector<int64_t> hshape = {nbatch, (int64_t)CC};
    std::vector<int64_t> hstr   = {P, 1};
    std::vector<int64_t> vshape = {nbatch, (int64_t)C};
    std::vector<int64_t> vstr   = contiguous_strides(vshape);

    DLTensor H = make_cpu_tensor(hbig,       hshape, hstr, bits);
    DLTensor X = make_cpu_tensor(xbuf.data(), vshape, vstr, bits);
    DLTensor B = make_cpu_tensor(bbuf.data(), vshape, vstr, bits);

    // matvec through the strided hessian: B = H * X
    ff::cpu::sym_matvec(B, H, X, 0);
    for (int64_t bt = 0; bt < nbatch; ++bt)
        for (int c = 0; c < C; ++c) {
            double ref = 0.0;
            for (int cc = 0; cc < C; ++cc) ref += full[bt][c*C + cc] * x[bt][cc];
            check_close((double)bbuf[bt*C + c], ref, "inflated_matvec");
        }

    // solve back through the strided hessian: X2 = H \ B == x
    std::vector<scalar_t> x2buf(nbatch * C);
    DLTensor X2 = make_cpu_tensor(x2buf.data(), vshape, vstr, bits);
    DLTensor Wn = null_tensor();
    ff::cpu::sym_solve(X2, H, B, Wn, 0);
    for (int64_t bt = 0; bt < nbatch; ++bt)
        for (int c = 0; c < C; ++c)
            check_close((double)x2buf[bt*C + c], x[bt][c], "inflated_solve");

    std::free(hbig);
}

// --- B4: negative / validation tests --------------------------------------
// Bad dtype: float16 must throw at the dtype dispatch, not silently no-op.
void test_bad_dtype_throws()
{
    const int64_t nbatch = 3; const int C = 2, CC = 3;
    std::vector<uint16_t> hbuf(nbatch*CC,0), xbuf(nbatch*C,0), bbuf(nbatch*C,0);
    std::vector<int64_t> vs={nbatch,(int64_t)C}, vst=contiguous_strides(vs);
    std::vector<int64_t> hs={nbatch,(int64_t)CC}, hst=contiguous_strides(hs);
    DLTensor H=make_cpu_tensor(hbuf.data(),hs,hst,16);
    DLTensor X=make_cpu_tensor(xbuf.data(),vs,vst,16);
    DLTensor Bo=make_cpu_tensor(bbuf.data(),vs,vst,16);
    bool threw = false;
    try { ff::cpu::sym_matvec(Bo, H, X, 0); } catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [posdef.bad_dtype_throws]\n"); }
}

// Shape mismatch: input channel count inconsistent with output must throw.
void test_shape_mismatch_throws()
{
    const int64_t nbatch = 3; const int C = 2, CC = 3;
    std::vector<double> hbuf(nbatch*CC,0), xbuf(nbatch*(C+1),0), bbuf(nbatch*C,0);
    std::vector<int64_t> vs ={nbatch,(int64_t)C},     vst =contiguous_strides(vs);
    std::vector<int64_t> vsx={nbatch,(int64_t)(C+1)}, vstx=contiguous_strides(vsx); // mismatched
    std::vector<int64_t> hs ={nbatch,(int64_t)CC},    hst =contiguous_strides(hs);
    DLTensor H=make_cpu_tensor(hbuf.data(),hs,hst,64);
    DLTensor X=make_cpu_tensor(xbuf.data(),vsx,vstx,64);  // C+1 channels
    DLTensor Bo=make_cpu_tensor(bbuf.data(),vs,vst,64);   // C channels
    bool threw = false;
    try { ff::cpu::sym_matvec(Bo, H, X, 0); } catch (const std::exception&) { threw = true; }
    ++g_checks;
    if (!threw) { ++g_failures; std::printf("  FAIL [posdef.shape_mismatch_throws]\n"); }
}


// --- B5: DLPack descriptor variants ---------------------------------------
// Two descriptor features that every posdef entry point used to normalise by
// hand before dispatch:
//
//   * `strides == NULL` -- DLPack's compact row-major shorthand, previously
//     expanded by autocast.h's ContiguousStrides;
//   * `byte_offset != 0` -- previously folded into the data pointer by the
//     VOIDPTR / CVOIDPTR macros.
//
// tny::from_dlpack now does both by construction. NEITHER had a single check on
// ANY posdef entry point before, so these cases pin the behaviour against the
// same brute-force oracle the contiguous cases use. They pass identically on the
// pre-refactor implementation -- they describe the ABI, not the new internals.
//
// mode bit0 -> strides == NULL ; bit1 -> byte_offset != 0 (so 3 == both).

template <typename scalar_t>
struct Padded {
    std::vector<scalar_t> buf;       // [pad sentinels | logical data]
    std::vector<int64_t>  shape;
    std::vector<int64_t>  strides;
    int64_t               pad = 0;
    DLTensor              t;

    void init(const std::vector<int64_t>& shp, uint8_t bits, int mode, int64_t padn)
    {
        shape   = shp;
        strides = contiguous_strides(shp);
        int64_t n = 1;
        for (size_t d = 0; d < shp.size(); ++d) n *= shp[d];
        pad = (mode & 2) ? padn : 0;
        buf.assign((size_t)(pad + n), scalar_t(0));
        for (int64_t k = 0; k < pad; ++k) buf[(size_t)k] = PAD_SENTINEL;
        t.data               = static_cast<void*>(buf.data());
        t.device.device_type = kDLCPU;
        t.device.device_id   = 0;
        t.ndim               = static_cast<int32_t>(shp.size());
        t.dtype.code         = static_cast<uint8_t>(kDLFloat);
        t.dtype.bits         = bits;
        t.dtype.lanes        = 1;
        t.shape              = shape.data();
        t.strides            = (mode & 1) ? nullptr : strides.data();
        t.byte_offset        = pad * (int64_t)sizeof(scalar_t);
    }
    scalar_t* d() { return buf.data() + pad; }        // logical element 0

    static const scalar_t PAD_SENTINEL;
};
template <typename scalar_t>
const scalar_t Padded<scalar_t>::PAD_SENTINEL = static_cast<scalar_t>(-12345.5);

// One check per tensor: nothing IN FRONT of byte_offset may be written. A
// mis-folded offset that reads right and writes left fails here instead of
// silently corrupting the caller's memory.
template <typename scalar_t>
void check_pad(const Padded<scalar_t>& p, const char* what)
{
    ++g_checks;
    for (int64_t k = 0; k < p.pad; ++k)
        if (p.buf[(size_t)k] != Padded<scalar_t>::PAD_SENTINEL) {
            ++g_failures;
            std::printf("  FAIL [posdef.pad_written:%s] slot %lld\n", what, (long long)k);
            return;
        }
}

template <typename scalar_t>
void run_descriptor_variants(uint8_t bits, int mode, unsigned seed)
{
    const int     C  = 3, CC = 6;
    const int64_t nb = 4, PAD = 5;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> off(-0.5, 0.5);
    std::uniform_real_distribution<double> vec(-2.0, 2.0);
    std::uniform_real_distribution<double> wdi(0.1, 3.0);

    std::vector<std::vector<double>> full(nb), x(nb), y(nb), w(nb);
    const std::vector<int64_t> vshape = {nb, (int64_t)C};
    const std::vector<int64_t> hshape = {nb, (int64_t)CC};

    Padded<scalar_t> H, X, Y, W;
    H.init(hshape, bits, mode, PAD);
    X.init(vshape, bits, mode, PAD);
    Y.init(vshape, bits, mode, PAD);
    W.init(vshape, bits, mode, PAD);

    for (int64_t bt = 0; bt < nb; ++bt) {
        full[bt].assign(C*C, 0.0);
        for (int c = 0; c < C; ++c)
            for (int cc = c; cc < C; ++cc) {
                double v = (c == cc) ? (C + 2.0 + off(rng)) : off(rng);
                full[bt][c*C + cc] = v;
                full[bt][cc*C + c] = v;
            }
        x[bt].resize(C); y[bt].resize(C); w[bt].resize(C);
        for (int c = 0; c < C; ++c) {
            x[bt][c] = vec(rng); y[bt][c] = vec(rng); w[bt][c] = wdi(rng);
        }
        std::vector<double> packed;
        pack_sym(full[bt], C, packed);
        for (int k = 0; k < CC; ++k) H.d()[bt*CC + k] = (scalar_t)packed[k];
        for (int c = 0; c < C; ++c) {
            X.d()[bt*C + c] = (scalar_t)x[bt][c];
            Y.d()[bt*C + c] = (scalar_t)y[bt][c];
            W.d()[bt*C + c] = (scalar_t)w[bt][c];
        }
    }

    // --- sym_matvec: B = H X ---
    Padded<scalar_t> B; B.init(vshape, bits, mode, PAD);
    ff::cpu::sym_matvec(B.t, H.t, X.t, 0);
    for (int64_t bt = 0; bt < nb; ++bt)
        for (int c = 0; c < C; ++c) {
            double ref = 0.0;
            for (int cc = 0; cc < C; ++cc) ref += full[bt][c*C + cc] * x[bt][cc];
            check_close((double)B.d()[bt*C + c], ref, "desc.matvec");
        }
    check_pad(B, "matvec");

    // --- sym_addmatvec_ / sym_submatvec_ : out = Y +/- H X ---
    Padded<scalar_t> A, U;
    A.init(vshape, bits, mode, PAD); U.init(vshape, bits, mode, PAD);
    for (int64_t bt = 0; bt < nb; ++bt)
        for (int c = 0; c < C; ++c) {
            A.d()[bt*C + c] = (scalar_t)y[bt][c];
            U.d()[bt*C + c] = (scalar_t)y[bt][c];
        }
    ff::cpu::sym_addmatvec_(A.t, H.t, X.t, 0);
    ff::cpu::sym_submatvec_(U.t, H.t, X.t, 0);
    for (int64_t bt = 0; bt < nb; ++bt)
        for (int c = 0; c < C; ++c) {
            double hx = 0.0;
            for (int cc = 0; cc < C; ++cc) hx += full[bt][c*C + cc] * x[bt][cc];
            check_close((double)A.d()[bt*C + c], y[bt][c] + hx, "desc.addmatvec_");
            check_close((double)U.d()[bt*C + c], y[bt][c] - hx, "desc.submatvec_");
        }
    check_pad(A, "addmatvec_");
    check_pad(U, "submatvec_");

    // --- sym_matvec_backward: G(CC) from grad=Y, inp=X ---
    Padded<scalar_t> G; G.init(hshape, bits, mode, PAD);
    ff::cpu::sym_matvec_backward(G.t, Y.t, X.t, 0);
    for (int64_t bt = 0; bt < nb; ++bt) {
        std::vector<double> ref;
        for (int c = 0; c < C; ++c) ref.push_back(y[bt][c]*x[bt][c]);
        for (int c = 0; c < C; ++c)
            for (int cc = c+1; cc < C; ++cc)
                ref.push_back(y[bt][c]*x[bt][cc] + y[bt][cc]*x[bt][c]);
        for (int k = 0; k < CC; ++k)
            check_close((double)G.d()[bt*CC + k], ref[k], "desc.matvec_backward");
    }
    check_pad(G, "matvec_backward");

    // --- sym_solve (unweighted): X2 = H \ B == x ---
    Padded<scalar_t> X2; X2.init(vshape, bits, mode, PAD);
    DLTensor Wn = null_tensor();
    ff::cpu::sym_solve(X2.t, H.t, B.t, Wn, 0);
    for (int64_t bt = 0; bt < nb; ++bt)
        for (int c = 0; c < C; ++c)
            check_close((double)X2.d()[bt*C + c], x[bt][c], "desc.solve");
    check_pad(X2, "solve");

    // --- sym_solve (weighted): Bw = (H + diag(w)) x, then solve back ---
    Padded<scalar_t> Bw, Xw;
    Bw.init(vshape, bits, mode, PAD); Xw.init(vshape, bits, mode, PAD);
    for (int64_t bt = 0; bt < nb; ++bt)
        for (int c = 0; c < C; ++c) {
            double r = w[bt][c] * x[bt][c];
            for (int cc = 0; cc < C; ++cc) r += full[bt][c*C + cc] * x[bt][cc];
            Bw.d()[bt*C + c] = (scalar_t)r;
        }
    ff::cpu::sym_solve(Xw.t, H.t, Bw.t, W.t, 0);
    for (int64_t bt = 0; bt < nb; ++bt)
        for (int c = 0; c < C; ++c)
            check_close((double)Xw.d()[bt*C + c], x[bt][c], "desc.wsolve");
    check_pad(Xw, "wsolve");

    // --- sym_solve_ in place ---
    Padded<scalar_t> S; S.init(vshape, bits, mode, PAD);
    for (int64_t bt = 0; bt < nb; ++bt)
        for (int c = 0; c < C; ++c) S.d()[bt*C + c] = B.d()[bt*C + c];
    ff::cpu::sym_solve_(S.t, H.t, Wn, 0);
    for (int64_t bt = 0; bt < nb; ++bt)
        for (int c = 0; c < C; ++c)
            check_close((double)S.d()[bt*C + c], x[bt][c], "desc.solve_");
    check_pad(S, "solve_");

    // --- sym_invert then matvec recovers x ---
    Padded<scalar_t> Hi, Z;
    Hi.init(hshape, bits, mode, PAD); Z.init(vshape, bits, mode, PAD);
    ff::cpu::sym_invert(Hi.t, H.t, 0);
    ff::cpu::sym_matvec(Z.t, Hi.t, B.t, 0);
    for (int64_t bt = 0; bt < nb; ++bt)
        for (int c = 0; c < C; ++c)
            check_close((double)Z.d()[bt*C + c], x[bt][c], "desc.invert+matvec");
    check_pad(Hi, "invert");
    check_pad(Z,  "invert+matvec");

    // --- sym_invert_ in place must match sym_invert ---
    Padded<scalar_t> Hi2; Hi2.init(hshape, bits, mode, PAD);
    for (int64_t bt = 0; bt < nb; ++bt)
        for (int k = 0; k < CC; ++k) Hi2.d()[bt*CC + k] = H.d()[bt*CC + k];
    ff::cpu::sym_invert_(Hi2.t, 0);
    for (int64_t bt = 0; bt < nb; ++bt)
        for (int k = 0; k < CC; ++k)
            check_close((double)Hi2.d()[bt*CC + k], (double)Hi.d()[bt*CC + k],
                        "desc.invert_");
    check_pad(Hi2, "invert_");

    // The INPUTS must come back untouched too (a mis-folded offset on a
    // read-only operand would show up as a written sentinel here).
    check_pad(H, "hessian.in");
    check_pad(X, "inp.in");
    check_pad(Y, "grd.in");
    check_pad(W, "wgt.in");
}

} // namespace

// Exercise the guess_type dispatch for the NON-Sym layouts through the public
// sym_matvec / sym_solve ABI. At C=3 the packed lengths are all distinct (Eye 1,
// Diag 3, ESTATICS 5, Sym 6, Full 9), so guess_type unambiguously selects the
// layout. layout: 0=Eye 1=Diag 2=ESTATICS 3=Full.
static int layout_CC(int layout, int C)
{ return layout == 0 ? 1 : layout == 1 ? C : layout == 2 ? 2*C-1 : C*C; }

// Fill per-batch dense SPD matrices M (row-major CxC) that fit `layout`, and the
// matching packed buffer hbuf (nb x CC) in the given element type.
template <typename T>
void fill_layout(int layout, int C, int64_t nb, std::mt19937& rng,
                 std::vector<std::vector<double>>& M, std::vector<T>& hbuf)
{
    std::uniform_real_distribution<double> u(-0.4, 0.4);
    const int CC = layout_CC(layout, C);
    M.assign(nb, std::vector<double>(C*C, 0.0));
    hbuf.assign(nb*CC, T(0));
    for (int64_t b = 0; b < nb; ++b) {
        auto& m = M[b];
        if (layout == 0) {                                   // Eye
            double d = 2.0 + std::fabs(u(rng)); for (int c=0;c<C;++c) m[c*C+c]=d;
            hbuf[b*CC] = (T)d;
        } else if (layout == 1) {                            // Diag
            for (int c=0;c<C;++c){ double d=2.0+std::fabs(u(rng)); m[c*C+c]=d; hbuf[b*CC+c]=(T)d; }
        } else if (layout == 2) {                            // ESTATICS
            double se=0;
            for (int c=0;c<C-1;++c){ double e=u(rng); m[c*C+(C-1)]=m[(C-1)*C+c]=e;
                m[c*C+c]=1.5+std::fabs(e); hbuf[b*CC+c]=(T)m[c*C+c]; hbuf[b*CC+C+c]=(T)e; se+=std::fabs(e); }
            m[(C-1)*C+(C-1)]=1.5+se; hbuf[b*CC+(C-1)]=(T)m[(C-1)*C+(C-1)];
        } else {                                             // Full (dense symmetric)
            for (int c=0;c<C;++c){ m[c*C+c]=C+2.0+u(rng);
                for (int cc=c+1;cc<C;++cc){ double v=u(rng); m[c*C+cc]=m[cc*C+c]=v; } }
            for (int i=0;i<C*C;++i) hbuf[b*CC+i]=(T)m[i];
        }
    }
}

template <typename T>
void run_layout_matvec(int layout, int C, int64_t nb, uint8_t bits, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int CC = layout_CC(layout, C);
    std::vector<std::vector<double>> M; std::vector<T> hbuf;
    fill_layout<T>(layout, C, nb, rng, M, hbuf);
    std::vector<std::vector<double>> xd(nb, std::vector<double>(C));
    std::vector<T> xbuf(nb*C), obuf(nb*C, T(0));
    for (int64_t b=0;b<nb;++b) for (int c=0;c<C;++c){ xd[b][c]=u(rng)*3.0; xbuf[b*C+c]=(T)xd[b][c]; }
    std::vector<int64_t> osh={nb,C}, hsh={nb,(int64_t)CC}, xsh={nb,C};
    auto os=contiguous_strides(osh), hs=contiguous_strides(hsh), xs=contiguous_strides(xsh);
    DLTensor ot=make_cpu_tensor(obuf.data(),osh,os,bits),
             ht=make_cpu_tensor(hbuf.data(),hsh,hs,bits),
             xt=make_cpu_tensor(xbuf.data(),xsh,xs,bits);
    ff::cpu::sym_matvec(ot, ht, xt, 0);
    for (int64_t b=0;b<nb;++b) for (int c=0;c<C;++c){
        double r=0; for (int cc=0;cc<C;++cc) r += M[b][c*C+cc]*xd[b][cc];
        check_close((double)obuf[b*C+c], r, "layout_matvec");
    }
}

// Solve (H + diag(w)) x = b for each layout: build b = (M + diag(w)) x from a
// known x, solve, recover x. weighted=false uses no weight (null tensor).
template <typename T>
void run_layout_solve(int layout, int C, int64_t nb, uint8_t bits, unsigned seed, bool weighted)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int CC = layout_CC(layout, C);
    std::vector<std::vector<double>> M; std::vector<T> hbuf;
    fill_layout<T>(layout, C, nb, rng, M, hbuf);
    std::vector<std::vector<double>> xd(nb, std::vector<double>(C));
    std::vector<T> wbuf(nb*C), bbuf(nb*C), obuf(nb*C, T(0));
    for (int64_t b=0;b<nb;++b) {
        std::vector<double> wd(C, 0.0);
        for (int c=0;c<C;++c){ xd[b][c]=u(rng)*2.0; wd[c] = weighted ? 0.2+std::fabs(u(rng)) : 0.0; wbuf[b*C+c]=(T)wd[c]; }
        for (int c=0;c<C;++c){                              // b = (M + diag(w)) x
            double r = wd[c]*xd[b][c];
            for (int cc=0;cc<C;++cc) r += M[b][c*C+cc]*xd[b][cc];
            bbuf[b*C+c]=(T)r;
        }
    }
    std::vector<int64_t> osh={nb,C}, hsh={nb,(int64_t)CC}, xsh={nb,C};
    auto os=contiguous_strides(osh), hs=contiguous_strides(hsh), xs=contiguous_strides(xsh);
    DLTensor ot=make_cpu_tensor(obuf.data(),osh,os,bits),
             ht=make_cpu_tensor(hbuf.data(),hsh,hs,bits),
             bt=make_cpu_tensor(bbuf.data(),xsh,xs,bits),
             wt=make_cpu_tensor(wbuf.data(),xsh,xs,bits);
    ff::cpu::sym_solve(ot, ht, bt, weighted ? wt : null_tensor(), 0);
    for (int64_t b=0;b<nb;++b) for (int c=0;c<C;++c)
        check_close((double)obuf[b*C+c], xd[b][c], weighted ? "layout_wsolve" : "layout_solve");
}

int main()
{
    std::printf("posdef module CPU tests\n");
    // guess_type dispatch of the non-Sym layouts (C=3, packed lengths distinct):
    // matvec + solve (unweighted and weighted), both dtypes.
    for (unsigned s = 1; s <= 3; ++s) {
        for (int L = 0; L <= 3; ++L) {                       // Eye, Diag, ESTATICS, Full
            run_layout_matvec<float >(L, 3, 6, 32, s + 1000 + 10*L);
            run_layout_matvec<double>(L, 3, 6, 64, s + 1050 + 10*L);
            run_layout_solve <float >(L, 3, 6, 32, s + 2000 + 10*L, false);
            run_layout_solve <double>(L, 3, 6, 64, s + 2050 + 10*L, false);
            run_layout_solve <double>(L, 3, 6, 64, s + 2100 + 10*L, true);   // weighted
        }
    }
    test_bad_dtype_throws();
    test_shape_mismatch_throws();
    for (unsigned seed = 1; seed <= 5; ++seed) {
        run_case<float >(2, 7,  32, seed);
        run_case<double>(2, 7,  64, seed + 100);
        run_case<float >(3, 5,  32, seed + 200);
        run_case<double>(3, 5,  64, seed + 300);
        run_case<float >(1, 4,  32, seed + 400);
        run_case<double>(4, 3,  64, seed + 500);  // dynamic C=-1 path

        // B3: weighted solve (H + diag(w)) \ b, both dtypes and static/dynamic C.
        run_weighted_case<float >(2, 7, 32, seed + 600);
        run_weighted_case<double>(3, 5, 64, seed + 700);
        run_weighted_case<double>(4, 3, 64, seed + 800);  // dynamic C=-1 path
    }

    // B2: 64-bit index + non-contiguous stride path (float keeps virtual mem low).
    run_inflated_stride_case<float>(2, 32, 900);
    run_inflated_stride_case<float>(3, 32, 901);

    // B5: DLPack descriptor variants -- NULL strides / non-zero byte_offset /
    // both, on every exported entry point, both dtypes.
    for (int mode = 1; mode <= 3; ++mode) {
        run_descriptor_variants<float >(32, mode, 3000u + 10u*mode);
        run_descriptor_variants<double>(64, mode, 3500u + 10u*mode);
    }
    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
