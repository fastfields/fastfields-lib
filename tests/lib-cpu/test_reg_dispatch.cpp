// Routing-identity tests for the regulariser dispatch chain
// (fastfields-cpu-lib#69).
//
// `reg_field.cpp` / `reg_flow.cpp` used to route a runtime
// (ndim, bound, dtype, offset-width) tuple to a templated worker through a
// hand-rolled nested macro pyramid (`NDIM_SWITCH` / `BOUND_SWITCH` / the
// per-op `*_DT` macros). That pyramid is now `reg_dispatch.h`, built on
// teeny's `dispatch_values`.
//
// This is a PURE codegen refactor, so the bar is not "the numbers still look
// right" -- a misrouted ndim or bound could easily land on a neighbouring
// instantiation whose output passes a loose numeric check. The bar is that the
// new chain selects the IDENTICAL template instantiation the old pyramid did,
// for every input the old one handled, and throws the identical exception for
// every input it rejected.
//
// So this file keeps the deleted macro pyramid alive, verbatim, as the
// reference implementation, points BOTH it and the real `reg_dispatch.h` chain
// at the same tracer, and compares:
//
//   * the instantiation actually reached -- recorded as (D, the full boundary
//     PACK, sizeof(scalar_t), sizeof(offset_t)); the pack is recorded element
//     by element, so a pack of the wrong LENGTH (the `BND1`/`BND2`/`BND3`
//     repeat that `dispatch_values` cannot do by itself) is caught, not just a
//     wrong value;
//   * or, where nothing is reached, the exception type and message text.
//
// swept over the full cross product, including out-of-range ndim, out-of-range
// bound, and non-float dtypes.
//
// Build (from fastfields-cpu-lib): picked up automatically by `make test`.

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "impl/kernels/cuda_switch.h"
#include "impl/kernels/bounds.h"
#include "dlpack.h"
#include "reg_dispatch.h"

namespace {

using ff::cpu::bound::type;
namespace reg_dispatch = ff::cpu::reg_dispatch;

int g_failures = 0;
int g_checks   = 0;

static void check(bool ok, const char * what)
{
    ++g_checks;
    if (!ok) { ++g_failures; std::printf("  FAIL: %s\n", what); }
}

/***********************************************************************
 *                             THE TRACER                              *
 ***********************************************************************/

// The identity of one templated instantiation, as reached by a dispatch.
struct Reached {
    bool             fired  = false;   // did any instantiation run at all?
    int              ndim   = -1;
    std::vector<int> bounds;           // the BOUND... pack, in order
    int              sbits  = 0;       // sizeof(scalar_t) * 8
    int              obits  = 0;       // sizeof(offset_t) * 8

    bool operator==(const Reached & o) const
    {
        return fired == o.fired && ndim == o.ndim && bounds == o.bounds
            && sbits == o.sbits && obits == o.obits;
    }
};

// Stand-in for `_field_matvec` / `_flow_relax` / ...: same template parameter
// shape (`int ndim`, scalar_t, offset_t, a variadic bound pack), records what
// it was instantiated as instead of computing anything.
template <int ndim, typename scalar_t, typename offset_t, type... BOUND>
inline void trace(Reached & r)
{
    r.fired  = true;
    r.ndim   = ndim;
    r.bounds = { static_cast<int>(BOUND)... };
    r.sbits  = static_cast<int>(sizeof(scalar_t)) * 8;
    r.obits  = static_cast<int>(sizeof(offset_t)) * 8;
}

// The adapter the production entry points write, pointed at the tracer.
struct trace_op {
    template <int D, typename scalar_t, typename offset_t, type... BOUND, typename... Args>
    static void run(Args &&... args)
    { trace<D, scalar_t, offset_t, BOUND...>(std::forward<Args>(args)...); }
};

/***********************************************************************
 *        REFERENCE: the pre-#69 macro pyramid, verbatim               *
 ***********************************************************************/
// Copied unchanged from reg_field.cpp/reg_flow.cpp as they stood before #69
// (only the leaf call is redirected to `trace`, and `TR_ARGS` stands in for
// the `MV_ARGS`/`DG_ARGS`/... argument macros). Do not "tidy" this: its whole
// value is being the old code, not a paraphrase of it.

#define BND1(B) B
#define BND2(B) B, B
#define BND3(B) B, B, B

#define TR_DT(NDIM, BNDS...)                                            \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? trace<NDIM, float,  int32_t, BNDS>(TR_ARGS)           \
                : trace<NDIM, float,  int64_t, BNDS>(TR_ARGS);          \
            case 64: return use_32bits                                  \
                ? trace<NDIM, double, int32_t, BNDS>(TR_ARGS)           \
                : trace<NDIM, double, int64_t, BNDS>(TR_ARGS);          \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define BOUND_SWITCH(DT, NDIM, BND)                                     \
    switch (bnd) {                                                      \
        case type::Zero:      DT(NDIM, BND(type::Zero));      break;     \
        case type::Replicate: DT(NDIM, BND(type::Replicate)); break;     \
        case type::DCT1:      DT(NDIM, BND(type::DCT1));      break;     \
        case type::DCT2:      DT(NDIM, BND(type::DCT2));      break;     \
        case type::DST1:      DT(NDIM, BND(type::DST1));      break;     \
        case type::DST2:      DT(NDIM, BND(type::DST2));      break;     \
        case type::DFT:       DT(NDIM, BND(type::DFT));       break;     \
        case type::NoCheck:   DT(NDIM, BND(type::NoCheck));   break;     \
        default: throw std::invalid_argument("Unsupported boundary condition"); \
    }

#define NDIM_SWITCH(DT)                                                 \
    switch (ndim) {                                                     \
        case 1: BOUND_SWITCH(DT, 1, BND1); break;                       \
        case 2: BOUND_SWITCH(DT, 2, BND2); break;                       \
        case 3: BOUND_SWITCH(DT, 3, BND3); break;                       \
        default: throw std::invalid_argument(ndim_msg);                 \
    }

static void old_dispatch(
    int            ndim       ,
    type           bnd        ,
    const char *   ndim_msg   ,
    DLDataTypeCode code       ,
    uint8_t        bits       ,
    bool           use_32bits ,
    Reached &      out        )
{
#define TR_ARGS out
    NDIM_SWITCH(TR_DT)
#undef TR_ARGS
}

#undef NDIM_SWITCH
#undef BOUND_SWITCH
#undef TR_DT
#undef BND3
#undef BND2
#undef BND1

/***********************************************************************
 *                     THE COMPARISON HARNESS                          *
 ***********************************************************************/

// What one dispatch call did: either it reached an instantiation, or it threw.
struct Outcome {
    Reached     reached;
    bool        threw = false;
    std::string message;

    bool operator==(const Outcome & o) const
    {
        return threw == o.threw && message == o.message && reached == o.reached;
    }
};

template <class Fn>
static Outcome observe(Fn && fn)
{
    Outcome o;
    try { fn(o.reached); }
    catch (const std::invalid_argument & e) { o.threw = true; o.message = e.what(); }
    return o;
}

static std::string describe(
    int ndim, int bnd, DLDataTypeCode code, uint8_t bits, bool use_32bits)
{
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "ndim=%d bound=%d code=%d bits=%u use_32bits=%d",
                  ndim, bnd, static_cast<int>(code),
                  static_cast<unsigned>(bits), static_cast<int>(use_32bits));
    return std::string(buf);
}

// The sweep. `ndim_msg` is threaded through both paths so the field ("...3D
// field...") and flow ("...3D flow...") wordings are each checked.
static void sweep(const char * ndim_msg, const char * label)
{
    // Deliberately wider than the supported sets, on every axis.
    const int            ndims[] = { -1, 0, 1, 2, 3, 4, 7 };
    const int            bnds [] = { -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 127 };
    const DLDataTypeCode codes[] = { kDLFloat, kDLInt, kDLUInt, kDLBfloat };
    const uint8_t        bitss[] = { 8, 16, 32, 64 };

    int n_reached = 0, n_threw = 0;

    for (int ndim : ndims)
    for (int b    : bnds)
    for (auto code: codes)
    for (auto bits: bitss)
    for (int u    : { 0, 1 })
    {
        const type bnd  = static_cast<type>(b);
        const bool u32  = (u != 0);

        const Outcome old_o = observe([&](Reached & r) {
            old_dispatch(ndim, bnd, ndim_msg, code, bits, u32, r);
        });
        const Outcome new_o = observe([&](Reached & r) {
            reg_dispatch::dispatch_nd_bound<trace_op>(
                ndim, bnd, ndim_msg, code, bits, u32, r);
        });

        const std::string ctx = std::string(label) + " " + describe(ndim, b, code, bits, u32);

        // 1. Routing identity: same instantiation, or same rejection.
        if (!(old_o == new_o)) {
            std::printf("  old: fired=%d ndim=%d npack=%zu sbits=%d obits=%d threw=%d msg='%s'\n",
                        (int)old_o.reached.fired, old_o.reached.ndim,
                        old_o.reached.bounds.size(), old_o.reached.sbits,
                        old_o.reached.obits, (int)old_o.threw, old_o.message.c_str());
            std::printf("  new: fired=%d ndim=%d npack=%zu sbits=%d obits=%d threw=%d msg='%s'\n",
                        (int)new_o.reached.fired, new_o.reached.ndim,
                        new_o.reached.bounds.size(), new_o.reached.sbits,
                        new_o.reached.obits, (int)new_o.threw, new_o.message.c_str());
        }
        check(old_o == new_o, ("routing identity: " + ctx).c_str());

        // 2. Absolute check on the new path, independent of the old one, so a
        //    shared mistake cannot pass: an accepted call must land on exactly
        //    ndim copies of the requested boundary, at the requested widths.
        if (new_o.reached.fired) {
            ++n_reached;
            bool ok = new_o.reached.ndim == ndim
                   && static_cast<int>(new_o.reached.bounds.size()) == ndim
                   && new_o.reached.sbits == static_cast<int>(bits)
                   && new_o.reached.obits == (u32 ? 32 : 64);
            for (int v : new_o.reached.bounds) ok = ok && (v == b);
            check(ok, ("pack shape: " + ctx).c_str());
        } else {
            ++n_threw;
        }
    }

    // 3. Coverage: every one of the 3 x 8 x 2 x 2 supported combinations must
    //    have reached an instantiation (a dispatch that silently threw
    //    everywhere would otherwise "agree" trivially).
    check(n_reached == 3 * 8 * 2 * 2, "supported cross product fully reached");
    std::printf("  %s: %d reached, %d rejected\n", label, n_reached, n_threw);
}

// The three rejection messages, checked by text at their exact trigger.
static void check_messages(const char * ndim_msg, const char * label)
{
    Reached r;
    auto call = [&](int ndim, int b, DLDataTypeCode code, uint8_t bits) {
        return observe([&](Reached & rr) {
            reg_dispatch::dispatch_nd_bound<trace_op>(
                ndim, static_cast<type>(b), ndim_msg, code, bits, false, rr);
        });
    };
    (void) r;

    const std::string ctx = std::string(" (") + label + ")";

    Outcome o = call(4, 0, kDLFloat, 32);
    check(o.threw && o.message == ndim_msg, ("bad ndim message" + ctx).c_str());

    o = call(2, 8, kDLFloat, 32);
    check(o.threw && o.message == "Unsupported boundary condition",
          ("bad bound message" + ctx).c_str());

    o = call(2, 0, kDLInt, 32);
    check(o.threw && o.message == "only floating point data types are supported",
          ("bad dtype code message" + ctx).c_str());

    o = call(2, 0, kDLFloat, 16);
    check(o.threw && o.message == "only floating point data types are supported",
          ("bad dtype width message" + ctx).c_str());

    // Wrong on more than one axis: the OUTERMOST problem is reported, exactly
    // as the nested switches did (ndim beats bound beats dtype).
    o = call(4, 8, kDLInt, 16);
    check(o.threw && o.message == ndim_msg, ("outermost wins: ndim" + ctx).c_str());
    o = call(2, 8, kDLInt, 16);
    check(o.threw && o.message == "Unsupported boundary condition",
          ("outermost wins: bound" + ctx).c_str());
}

} // namespace

int main()
{
    std::printf("reg dispatch routing-identity CPU tests\n");

    sweep("Only 1D, 2D and 3D field are supported", "field");
    sweep("Only 1D, 2D and 3D flow are supported",  "flow");

    check_messages("Only 1D, 2D and 3D field are supported", "field");
    check_messages("Only 1D, 2D and 3D flow are supported",  "flow");

    std::printf("checks: %d, failures: %d\n", g_checks, g_failures);
    if (g_failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
