// Standalone unit test for the spline_coeff bound guard (fastfields-lib #65).
//
// `bound=zero` (and dst1/dst2/nocheck) has no prefilter recursion of its own:
// the kernels fall through to the DCT1 initial/final conditions, so the caller
// silently got whole-point mirroring -- results bit-identical to dct1 -- for a
// boundary condition they never asked for. ff::require_splinc_bound rejects
// those up front, mirroring jitfields' `splinc.checkbound`.
//
// Header-only: it exercises the guard itself, not the dispatch, so no library
// link is needed. Compile + run from the repo root:
//     clang++ -std=c++11 -I.. tests/test_splinc_bound.cpp -o /tmp/tsb && /tmp/tsb
// or via `make test`.

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include "../splinc.h"

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; } \
    else         { std::printf("ok:   %s\n", msg); } \
} while (0)

// Returns the message if require_splinc_bound threw, or an empty string.
static std::string threw_with(int8_t spline, int8_t bound) {
    try { ff::require_splinc_bound(spline, bound); }
    catch (const std::invalid_argument & e) { return std::string(e.what()); }
    return std::string();
}

static bool contains(const std::string & hay, const char * needle) {
    return hay.find(needle) != std::string::npos;
}

int main() {
    // The four orders that actually run the recursion, plus the two identity
    // orders and an out-of-range one.
    const int8_t real_orders[6] = {
        ff::spline_t::Quadratic,   ff::spline_t::Cubic,
        ff::spline_t::FourthOrder, ff::spline_t::FifthOrder,
        ff::spline_t::SixthOrder,  ff::spline_t::SeventhOrder
    };

    // 1. The implemented bounds must never throw, at any real order.
    {
        const int8_t ok_bounds[4] = {
            ff::bound_t::DCT1, ff::bound_t::DCT2,
            ff::bound_t::DFT,  ff::bound_t::Replicate
        };
        bool any_threw = false;
        for (int o = 0; o < 6; ++o)
            for (int b = 0; b < 4; ++b)
                if (!threw_with(real_orders[o], ok_bounds[b]).empty())
                    any_threw = true;
        CHECK(!any_threw, "dct1/dct2/dft/replicate accepted at every order 2..7");
    }

    // 2. bound=zero must throw at every order that runs the prefilter.
    //    This is the regression: it used to be silently aliased to dct1.
    {
        bool all_threw = true;
        for (int o = 0; o < 6; ++o)
            if (threw_with(real_orders[o], ff::bound_t::Zero).empty())
                all_threw = false;
        CHECK(all_threw, "bound=zero throws at every order 2..7");
    }

    // 3. The message names the offending bound and the implemented set.
    {
        std::string msg = threw_with(ff::spline_t::Cubic, ff::bound_t::Zero);
        CHECK(contains(msg, "zero"), "error message names the rejected bound (zero)");
        CHECK(contains(msg, "dct1") && contains(msg, "dct2") &&
              contains(msg, "dft")  && contains(msg, "replicate"),
              "error message lists the implemented bounds");
    }

    // 4. The other unimplemented bounds throw too (dst1/dst2 already raised
    //    from the torch binding; the guard makes that uniform across backends).
    {
        CHECK(contains(threw_with(ff::spline_t::Cubic, ff::bound_t::DST1), "dst1"),
              "bound=dst1 throws");
        CHECK(contains(threw_with(ff::spline_t::Cubic, ff::bound_t::DST2), "dst2"),
              "bound=dst2 throws");
        CHECK(contains(threw_with(ff::spline_t::Cubic, ff::bound_t::NoCheck), "nocheck"),
              "bound=nocheck throws");
    }

    // 5. Orders 0/1 are the identity -- the bound is never consulted, so even
    //    an unimplemented one must be accepted (matches jitfields' checkbound).
    {
        bool threw = !threw_with(ff::spline_t::Nearest, ff::bound_t::Zero).empty()
                  || !threw_with(ff::spline_t::Linear,  ff::bound_t::Zero).empty();
        CHECK(!threw, "orders 0/1 accept any bound (prefilter is the identity)");
    }

    // 6. An out-of-range order is not this guard's business: it must stay
    //    silent so the backend's "unsupported spline order" is what surfaces.
    {
        bool threw = !threw_with((int8_t)8,  ff::bound_t::Zero).empty()
                  || !threw_with((int8_t)-1, ff::bound_t::Zero).empty();
        CHECK(!threw, "out-of-range orders defer to the backend's order error");
    }

    if (failures) {
        std::printf("\n%d check(s) FAILED\n", failures);
        return 1;
    }
    std::printf("\nAll checks passed.\n");
    return 0;
}
