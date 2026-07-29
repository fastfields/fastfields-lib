// CPU unit tests for the `op` (assign / add / subtract) dispatch in the
// regulariser impl layer (fastfields-lib issue #6, item a).
//
// The public cpu-lib API (field_matvec / flow_matvec) always passes op '='.
// The bug being guarded against is that the CPU impl used to hard-code the
// kernel op as <set> regardless of the `char op` template parameter, so a
// future '+' / '-' would have silently OVERWRITTEN instead of accumulating.
//
// These tests call the templated impl functions
//   ff::cpu::reg_field::* / ff::cpu::reg_flow::*
// directly with op '=', '+' and '-', and check that
//   op '='  writes            out = K(inp)
//   op '+'  accumulates       out = base + K(inp)
//   op '-'  subtracts         out = base - K(inp)
// where K(inp) is the same operator obtained from the (unchanged) '=' path.
//
// Build (from fastfields-cpu-lib): picked up automatically by `make test`.

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

#include "impl/kernels/cuda_switch.h"
#include "impl/kernels/bounds.h"
#include "impl/kernels/utils.h"
#include "impl/reg_field.h"
#include "impl/reg_flow.h"

namespace {

typedef double  reduce_t;
typedef double  scalar_t;
typedef int64_t offset_t;

const ff::cpu::bound::type BND = ff::cpu::bound::type::DCT2;   // 1D bound

int g_failures = 0;
int g_checks   = 0;

void check_close(double a, double b, const char* what, double tol = 1e-6)
{
    ++g_checks;
    double diff  = std::fabs(a - b);
    double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    if (diff / scale > tol) {
        ++g_failures;
        std::printf("  MISMATCH [%s]: got %.10g expected %.10g\n", what, a, b);
    }
}

std::vector<scalar_t> make_inp(int64_t N, int64_t C)
{
    std::vector<scalar_t> v(N * C);
    for (int64_t x = 0; x < N; ++x)
        for (int64_t c = 0; c < C; ++c)
            v[x*C+c] = (scalar_t)(std::sin(0.4*x + 0.7*c + 1.0) + 0.3*x - 0.2*c);
    return v;
}

std::vector<scalar_t> make_base(int64_t N, int64_t C)
{
    std::vector<scalar_t> v(N * C);
    for (int64_t i = 0; i < N*C; ++i)
        v[i] = (scalar_t)(1.0 + 0.13 * i - 0.05 * (i % 3));
    return v;
}

// Given a runner with a member `template <char op> void apply(out, inp)`,
// verify the three op paths are consistent (the actual assertion of the fix):
//   ref  = K(inp)             via op '='
//   plus  == base + ref       via op '+'
//   minus == base - ref       via op '-'
template <typename Runner>
void check_ops(const Runner& run, int64_t N, int64_t C, const char* tag)
{
    std::vector<scalar_t> inp  = make_inp(N, C);
    std::vector<scalar_t> base = make_base(N, C);

    std::vector<scalar_t> ref(N*C, scalar_t(0));
    run.template apply<'='>(ref, inp);

    double mag = 0;
    for (int64_t i = 0; i < N*C; ++i) mag += std::fabs((double)ref[i]);
    if (mag < 1e-9) { ++g_failures; std::printf("  ZERO-OP [%s]: reference all zero\n", tag); }

    std::vector<scalar_t> plus = base;
    run.template apply<'+'>(plus, inp);
    for (int64_t i = 0; i < N*C; ++i)
        check_close((double)plus[i], (double)base[i] + (double)ref[i], tag);

    std::vector<scalar_t> minus = base;
    run.template apply<'-'>(minus, inp);
    for (int64_t i = 0; i < N*C; ++i)
        check_close((double)minus[i], (double)base[i] - (double)ref[i], tag);
}

//===========================================================================
//  reg_field runners (multi-channel field, ndim = 1)
//===========================================================================

struct FieldAbsMatvec {
    int64_t N, C; std::vector<reduce_t> absolute;
    template <char op>
    void apply(std::vector<scalar_t>& out, const std::vector<scalar_t>& inp) const {
        offset_t size[2]   = {N, C};
        offset_t stride[2] = {C, 1};
        ff::cpu::reg_field::matvec_absolute<1, op, reduce_t, scalar_t, offset_t, BND>(
            ff::bound::BoundVec(BND), 0, out.data(), inp.data(), size, stride, stride, absolute.data());
    }
};

struct FieldMembraneMatvec {
    int64_t N, C; std::vector<reduce_t> absolute, membrane;
    template <char op>
    void apply(std::vector<scalar_t>& out, const std::vector<scalar_t>& inp) const {
        offset_t size[2]   = {N, C};
        offset_t stride[2] = {C, 1};
        reduce_t vx[1]     = {1.0};
        ff::cpu::reg_field::matvec_membrane<1, op, reduce_t, scalar_t, offset_t, BND>(
            ff::bound::BoundVec(BND), 0, out.data(), inp.data(), size, stride, stride, vx,
            absolute.data(), membrane.data());
    }
};

// NOTE: the diag path is exercised via `diag_membrane`. `diag_absolute` also
// takes `op` and the impl now threads it, but the *kernel* `diag_absolute<op>`
// (fastfields-kernels field|flow/{1,2,3}d.h) forwards to `kernel_absolute(...)`
// dropping `<op>`, so it cannot accumulate regardless of the impl fix. That is a
// separate, pre-existing kernel-layer bug (shared with the CUDA impl); see the
// PR description. diag_membrane honours `op` and validates the impl threading.
struct FieldMembraneDiag {
    int64_t N, C; std::vector<reduce_t> absolute, membrane;
    template <char op>   // diag ignores `inp`; op still selects assign/add/sub
    void apply(std::vector<scalar_t>& out, const std::vector<scalar_t>&) const {
        offset_t size[2]   = {N, C};
        offset_t stride[2] = {C, 1};
        reduce_t vx[1]     = {1.0};
        ff::cpu::reg_field::diag_membrane<1, op, reduce_t, scalar_t, offset_t, BND>(
            ff::bound::BoundVec(BND), 0, out.data(), size, stride, vx, absolute.data(), membrane.data());
    }
};

//===========================================================================
//  reg_flow runners (vector flow, ndim = 1 -> 1 component)
//===========================================================================

struct FlowAbsMatvec {
    int64_t N; reduce_t absolute;
    template <char op>
    void apply(std::vector<scalar_t>& out, const std::vector<scalar_t>& inp) const {
        offset_t size[2]   = {N, 1};
        offset_t stride[2] = {1, 1};
        reduce_t vx[1]     = {1.0};
        ff::cpu::reg_flow::matvec_absolute<1, op, reduce_t, scalar_t, offset_t, BND>(
            ff::bound::BoundVec(BND), 0, out.data(), inp.data(), size, stride, stride, vx, absolute);
    }
};

struct FlowMembraneMatvec {
    int64_t N; reduce_t absolute, membrane;
    template <char op>
    void apply(std::vector<scalar_t>& out, const std::vector<scalar_t>& inp) const {
        offset_t size[2]   = {N, 1};
        offset_t stride[2] = {1, 1};
        reduce_t vx[1]     = {1.0};
        ff::cpu::reg_flow::matvec_membrane<1, op, reduce_t, scalar_t, offset_t, BND>(
            ff::bound::BoundVec(BND), 0, out.data(), inp.data(), size, stride, stride, vx, absolute, membrane);
    }
};

struct FlowMembraneDiag {
    int64_t N; reduce_t absolute, membrane;
    template <char op>
    void apply(std::vector<scalar_t>& out, const std::vector<scalar_t>&) const {
        offset_t size[2]   = {N, 1};
        offset_t stride[2] = {1, 1};
        reduce_t vx[1]     = {1.0};
        ff::cpu::reg_flow::diag_membrane<1, op, reduce_t, scalar_t, offset_t, BND>(
            ff::bound::BoundVec(BND), 0, out.data(), size, stride, vx, absolute, membrane);
    }
};

} // namespace

int main()
{
    std::printf("reg op (=/+/-) dispatch CPU tests\n");

    // ---- reg_field ----
    { FieldAbsMatvec r; r.N = 9; r.C = 2; r.absolute = {2.5, -1.5};
      check_ops(r, r.N, r.C, "field_abs_matvec"); }
    { FieldMembraneMatvec r; r.N = 11; r.C = 2; r.absolute = {0.7, 0.4}; r.membrane = {1.3, 0.9};
      check_ops(r, r.N, r.C, "field_membrane_matvec"); }
    { FieldMembraneDiag r; r.N = 11; r.C = 2; r.absolute = {0.7, 0.4}; r.membrane = {1.3, 0.9};
      check_ops(r, r.N, r.C, "field_membrane_diag"); }

    // ---- reg_flow ----
    { FlowAbsMatvec r; r.N = 9; r.absolute = 2.5;
      check_ops(r, r.N, 1, "flow_abs_matvec"); }
    { FlowMembraneMatvec r; r.N = 11; r.absolute = 0.7; r.membrane = 1.3;
      check_ops(r, r.N, 1, "flow_membrane_matvec"); }
    { FlowMembraneDiag r; r.N = 11; r.absolute = 0.7; r.membrane = 1.3;
      check_ops(r, r.N, 1, "flow_membrane_diag"); }

    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
