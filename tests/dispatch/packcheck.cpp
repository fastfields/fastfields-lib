// Checks that api/dispatch.h hands the leaf exactly what the macro pyramid
// used to: the right ndim, the right scalar_t, the right offset_t, and a
// boundary pack of exactly `ndim` copies (BND1/BND2/BND3) -- the pack length is
// load-bearing, because bound::getutils<B> / <B,B> / <B,B,B> detect isotropy
// from it. Also checks the three rejection paths still throw.
//
// NOT wired into any make target, on purpose: `make test` globs
// tests/lib-cpu/test_*.cpp and tests/lib/test_*.cpp, so this file cannot change
// the recorded baseline (13 suites / 59886 checks, plus 2 hub suites). Run it
// by hand, the way tests/kernels/vector/test.cpp is run:
//
//   clang++ -std=c++11 -O1 -Iinclude -o /tmp/packcheck tests/dispatch/packcheck.cpp && /tmp/packcheck
//   clang++ -std=c++11 -O1 -Iinclude -DFF_INDEX32=0 -o /tmp/packcheck0 tests/dispatch/packcheck.cpp && /tmp/packcheck0
//   g++     -std=c++11 -O1 -Iinclude -o /tmp/packcheckg tests/dispatch/packcheck.cpp && /tmp/packcheckg
#include <cstdio>
#include <cstdint>
#include <vector>
#include "fastfields/api/dispatch.h"

using namespace ff::cpu;

static std::vector<int> g_ndim, g_npack, g_off, g_sca;

struct probe_op {
    template <int ndim, class scalar_t, class offset_t, ff::bound::type... B>
    static void run(int /* sentinel */) {
        g_ndim .push_back(ndim);
        g_npack.push_back(static_cast<int>(sizeof...(B)));
        g_off  .push_back(static_cast<int>(sizeof(offset_t)));
        g_sca  .push_back(static_cast<int>(sizeof(scalar_t)));
    }
};

int main()
{
    int bad = 0, checks = 0;
    DLDataType f32; f32.code = kDLFloat; f32.bits = 32; f32.lanes = 1;
    DLDataType f64; f64.code = kDLFloat; f64.bits = 64; f64.lanes = 1;

    for (int nd = 1; nd <= 3; ++nd)
    for (int b  = 0; b  <  8; ++b)
    for (int w  = 0; w  <  2; ++w)
    {
        g_ndim.clear(); g_npack.clear(); g_off.clear(); g_sca.clear();
        DispatchKey k32(nd, static_cast<int8_t>(b), f32, w != 0);
        dispatch_nbd<probe_op>(k32, 0);
        DispatchKey k64(nd, static_cast<int8_t>(b), f64, w != 0);
        dispatch_nbd<probe_op>(k64, 0);
        ++checks;
        if (g_ndim.size() != 2) { ++bad; continue; }
        if (g_ndim [0] != nd || g_ndim [1] != nd) ++bad;   // ndim threaded through
        if (g_npack[0] != nd || g_npack[1] != nd) ++bad;   // BND<ndim>
        if (g_sca  [0] != 4  || g_sca  [1] != 8 ) ++bad;   // float / double
#if FF_INDEX32
        const int want = w ? 4 : 8;
#else
        const int want = 8;                                 // axis compiled out
#endif
        if (g_off[0] != want || g_off[1] != want) ++bad;
    }

    int threw = 0;
    DLDataType i32; i32.code = kDLInt; i32.bits = 32; i32.lanes = 1;
    try { DispatchKey k(2, 3, i32, true);   dispatch_nbd<probe_op>(k, 0); }
    catch (const std::invalid_argument &) { ++threw; }      // bad dtype
    try { DispatchKey k(4, 3, f32, true);   dispatch_nbd<probe_op>(k, 0); }
    catch (const std::invalid_argument &) { ++threw; }      // bad ndim
    try { DispatchKey k(2, 99, f32, true);  dispatch_nbd<probe_op>(k, 0); }
    catch (const std::invalid_argument &) { ++threw; }      // bad bound
    if (threw != 3) ++bad;

    std::printf("packcheck: %s (%d configurations, %d problems)\n",
                bad ? "FAIL" : "OK", checks, bad);
    return bad != 0;
}
