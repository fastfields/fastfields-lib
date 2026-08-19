/**
 * Conversion-correctness test for core/half.h (the teeny half/bfloat16 port).
 *
 * NOT part of `make test` -- run it with `make test-half`. It gates the header
 * alone, and deliberately does not touch the 13-suite lib-cpu gate.
 *
 * The interesting part is the ORACLE. clang++ 15+ and g++ 12+ both provide
 * `_Float16` and `__bf16` as real arithmetic types on x86-64, in C++11 mode,
 * with IEEE round-to-nearest-even conversions done by the hardware (F16C) or by
 * compiler-rt/libgcc. That is a genuine ground truth to diff against, so rather
 * than spot-checking a handful of values this test is EXHAUSTIVE:
 *
 *   - all 2^16 half bit patterns  -> float, bit-compared against the oracle
 *   - all 2^16 bf16 bit patterns  -> float, bit-compared against the oracle
 *   - all 2^32 float bit patterns -> half, bit-compared against the oracle
 *   - all 2^32 float bit patterns -> bf16, bit-compared against the oracle
 *
 * That covers every subnormal, every overflow boundary and every
 * round-to-nearest-even tie by construction, with nothing left to a hand-picked
 * list. NaN is the one documented exception: the software conversions
 * canonicalise the payload where the hardware preserves it, so a NaN input is
 * only required to produce *a* NaN.
 *
 * A named-case section follows, so a failure reports something more useful than
 * "bit pattern 0x3c01 differs", and so the boundary values a reader wants to
 * see stated are actually stated.
 */
#include "fastfields/core/half.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <type_traits>

using ff::bfloat16;
using ff::half;

// ---------------------------------------------------------------------------
// oracle: the compiler's own 16-bit float types
// ---------------------------------------------------------------------------
#if !defined(__FLT16_MAX__)
#error "this test needs the compiler's native _Float16 as an oracle"
#endif

typedef _Float16 oracle_h;
typedef __bf16 oracle_b;

static uint16_t oracle_f32_to_f16(float f)
{
    oracle_h h = static_cast<oracle_h>(f);
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}
static float oracle_f16_to_f32(uint16_t u)
{
    oracle_h h;
    memcpy(&h, &u, 2);
    return static_cast<float>(h);
}
static uint16_t oracle_f32_to_bf16(float f)
{
    oracle_b b = static_cast<oracle_b>(f);
    uint16_t u;
    memcpy(&u, &b, 2);
    return u;
}
static float oracle_bf16_to_f32(uint16_t u)
{
    oracle_b b;
    memcpy(&b, &u, 2);
    return static_cast<float>(b);
}

// ---------------------------------------------------------------------------
static uint32_t bits_of(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}
static float float_of(uint32_t u)
{
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static uint16_t to_h(float f) { return half(f).bits; }
static uint16_t to_b(float f) { return bfloat16(f).bits; }
static float of_h(uint16_t u)
{
    half h;
    h.bits = u;
    return static_cast<float>(h);
}
static float of_b(uint16_t u)
{
    bfloat16 b;
    b.bits = u;
    return static_cast<float>(b);
}

static bool h_is_nan(uint16_t u)
{
    return ((u >> 10) & 0x1f) == 0x1f && (u & 0x3ff);
}
static bool b_is_nan(uint16_t u)
{
    return ((u >> 7) & 0xff) == 0xff && (u & 0x7f);
}

static int failures = 0;
static const int MAX_REPORT = 12;

static void fail(const char * what, unsigned long long in,
                 unsigned long long got, unsigned long long want)
{
    if (failures < MAX_REPORT)
        std::printf("  FAIL %-24s in=0x%llx got=0x%llx want=0x%llx\n", what, in,
                    got, want);
    ++failures;
}

static void check(bool ok, const char * what)
{
    if (!ok) {
        std::printf("  FAIL %s\n", what);
        ++failures;
    }
}

// ---------------------------------------------------------------------------
// The two 2^32 sweeps take ~6 min at -O2. `test_half <stride>` subsamples them
// (use an ODD stride: an even one would never test an odd float bit pattern at
// all, and the round-to-nearest-even ties live in the low bits). Stride 1 --
// the default, and what `make test-half` runs -- is the real exhaustive proof;
// a stride is only for a quick smoke run while iterating on the header.
int main(int argc, char ** argv)
{
    uint32_t stride = 1;
    if (argc > 1) {
        long s = std::strtol(argv[1], 0, 0);
        if (s <= 0 || !(s & 1)) {
            std::printf("stride must be a positive odd number\n");
            return 2;
        }
        stride = static_cast<uint32_t>(s);
    }

    // ---- layout / type properties ----------------------------------------
    static_assert(sizeof(half) == 2, "half is 16-bit");
    static_assert(sizeof(bfloat16) == 2, "bfloat16 is 16-bit");
    // trivially copyable: what DLPack buffers, memcpy and by-value CUDA kernel
    // arguments actually require.
    static_assert(std::is_trivially_copyable<half>::value, "half memcpy-able");
    static_assert(std::is_trivially_copyable<bfloat16>::value,
                  "bfloat16 memcpy-able");
    // trivially default-constructible: where this port deviates from teeny
    // (which uses an NSDMI). `half x;` must not emit a store, and __shared__
    // arrays under nvcc require it.
    static_assert(std::is_trivially_default_constructible<half>::value,
                  "half default-ctor must stay trivial (no NSDMI)");
    static_assert(std::is_trivially_default_constructible<bfloat16>::value,
                  "bfloat16 default-ctor must stay trivial (no NSDMI)");
    // the accumulate-wider-than-you-store rule
    static_assert(std::is_same<ff::compute_type<half>::type, float>::value,
                  "half computes in float");
    static_assert(std::is_same<ff::compute_type<bfloat16>::type, float>::value,
                  "bf16 computes in float");
    static_assert(std::is_same<ff::compute_type<double>::type, double>::value,
                  "double computes in double");

    std::printf(
        "core/half.h -- conversion check vs native _Float16 / __bf16\n");

    // ---- 1. half -> float, all 65536 patterns -----------------------------
    {
        int local = failures;
        for (uint32_t u = 0; u < 0x10000u; ++u) {
            uint16_t h = static_cast<uint16_t>(u);
            float got = of_h(h), want = oracle_f16_to_f32(h);
            if (h_is_nan(h)) {
                if (!std::isnan(got)) fail("f16->f32 nan", h, 0, 0);
                continue;
            }
            if (bits_of(got) != bits_of(want))
                fail("f16->f32", h, bits_of(got), bits_of(want));
        }
        std::printf("  [%s] f16  -> f32   65536/65536 bit patterns\n",
                    failures == local ? "ok" : "!!");
    }

    // ---- 2. bfloat16 -> float, all 65536 patterns -------------------------
    {
        int local = failures;
        for (uint32_t u = 0; u < 0x10000u; ++u) {
            uint16_t b = static_cast<uint16_t>(u);
            float got = of_b(b), want = oracle_bf16_to_f32(b);
            if (b_is_nan(b)) {
                if (!std::isnan(got)) fail("bf16->f32 nan", b, 0, 0);
                continue;
            }
            if (bits_of(got) != bits_of(want))
                fail("bf16->f32", b, bits_of(got), bits_of(want));
        }
        std::printf("  [%s] bf16 -> f32   65536/65536 bit patterns\n",
                    failures == local ? "ok" : "!!");
    }

    // ---- 3. float -> half, all 2^32 patterns ------------------------------
    // subnormals, the 65504/65520 overflow boundary and every tie-to-even are
    // all in here by construction.
    {
        int local = failures;
        uint64_t n = 0;
        for (uint64_t k = 0; k < 0x100000000ull; k += stride) {
            uint32_t u = static_cast<uint32_t>(k);
            float f = float_of(u);
            ++n;
            uint16_t got = to_h(f), want = oracle_f32_to_f16(f);
            if (std::isnan(f)) {
                if (!h_is_nan(got)) fail("f32->f16 nan", u, got, 0);
            }
            else if (got != want)
                fail("f32->f16", u, got, want);
        }
        std::printf(
            "  [%s] f32  -> f16   %llu/4294967296 bit patterns (stride %u)\n",
            failures == local ? "ok" : "!!", (unsigned long long)n, stride);
    }

    // ---- 4. float -> bfloat16, all 2^32 patterns --------------------------
    {
        int local = failures;
        uint64_t n = 0;
        for (uint64_t k = 0; k < 0x100000000ull; k += stride) {
            uint32_t u = static_cast<uint32_t>(k);
            float f = float_of(u);
            ++n;
            uint16_t got = to_b(f), want = oracle_f32_to_bf16(f);
            if (std::isnan(f)) {
                if (!b_is_nan(got)) fail("f32->bf16 nan", u, got, 0);
            }
            else if (got != want)
                fail("f32->bf16", u, got, want);
        }
        std::printf(
            "  [%s] f32  -> bf16  %llu/4294967296 bit patterns (stride %u)\n",
            failures == local ? "ok" : "!!", (unsigned long long)n, stride);
    }

    // ---- 5. named cases ---------------------------------------------------
    // Redundant with the sweeps, but they name the properties a reader wants
    // stated, and they fail readably.
    {
        int local = failures;

        // exact round-trip of exactly-representable values
        check(static_cast<float>(half(1.5f)) == 1.5f, "half 1.5 round-trip");
        check(static_cast<float>(half(-0.25f)) == -0.25f,
              "half -0.25 round-trip");
        check(static_cast<float>(bfloat16(1.5f)) == 1.5f,
              "bf16 1.5 round-trip");
        check(static_cast<float>(bfloat16(-256.f)) == -256.f,
              "bf16 -256 round-trip");

        // normal-range boundaries
        check(to_h(65504.0f) == 0x7bff, "half max normal 65504");
        check(to_h(65519.0f) == 0x7bff, "half 65519 rounds down, not to inf");
        check(to_h(65520.0f) == 0x7c00, "half 65520 ties up to inf");
        check(to_h(65536.0f) == 0x7c00, "half overflow -> +inf");
        check(to_h(-65536.0f) == 0xfc00, "half overflow -> -inf");

        // subnormals
        check(to_h(6.103515625e-05f) == 0x0400, "half smallest normal 2^-14");
        check(to_h(5.9604644775390625e-08f) == 0x0001,
              "half smallest subnormal 2^-24");
        check(to_h(std::ldexp(1.0f, -25)) == 0x0000,
              "half 2^-25 ties to even -> 0");
        check(to_h(std::ldexp(1.5f, -25)) == 0x0001,
              "half 1.5*2^-25 rounds up");
        check(to_h(std::ldexp(3.0f, -25)) == 0x0002,
              "half 3*2^-25 ties to even -> 2");
        check(of_h(0x0001) == std::ldexp(1.0f, -24),
              "half subnormal decode 2^-24");
        check(of_h(0x03ff) == std::ldexp(1023.0f, -24),
              "half largest subnormal decode");

        // round-to-nearest-EVEN ties (not round-half-away).
        // 1 + 2^-11 sits exactly halfway between half(1.0)=0x3c00 and 0x3c01;
        // 0x3c00 has an even mantissa, so it must stay.
        check(to_h(1.0f + std::ldexp(1.0f, -11)) == 0x3c00,
              "half tie 1+2^-11 -> even (down)");
        // 1 + 3*2^-11 sits halfway between 0x3c01 and 0x3c02 -> up, to even.
        check(to_h(1.0f + std::ldexp(3.0f, -11)) == 0x3c02,
              "half tie 1+3*2^-11 -> even (up)");
        // same for bf16: 1 + 2^-8 sits between 0x3f80 and 0x3f81.
        check(to_b(1.0f + std::ldexp(1.0f, -8)) == 0x3f80,
              "bf16 tie 1+2^-8 -> even (down)");
        check(to_b(1.0f + std::ldexp(3.0f, -8)) == 0x3f82,
              "bf16 tie 1+3*2^-8 -> even (up)");

        // inf / nan / signed zero
        const float inf = std::ldexp(1.0f, 200) * std::ldexp(1.0f, 200);
        check(to_h(inf) == 0x7c00 && to_h(-inf) == 0xfc00, "half +/-inf");
        check(to_b(inf) == 0x7f80 && to_b(-inf) == 0xff80, "bf16 +/-inf");
        check(std::isinf(of_h(0x7c00)) && of_h(0x7c00) > 0, "half decode +inf");
        check(std::isinf(of_b(0xff80)) && of_b(0xff80) < 0, "bf16 decode -inf");
        check(h_is_nan(to_h(inf - inf)), "half nan stays nan");
        check(b_is_nan(to_b(inf - inf)), "bf16 nan stays nan");
        check(std::isnan(of_h(0x7e00)) && std::isnan(of_b(0x7fc0)),
              "nan decode");
        check(to_h(0.0f) == 0x0000 && to_h(-0.0f) == 0x8000,
              "half signed zero");
        check(to_b(0.0f) == 0x0000 && to_b(-0.0f) == 0x8000,
              "bf16 signed zero");

        // bf16 keeps the float exponent range, so no overflow where half
        // overflows -- but it has only 7 mantissa bits, so FLT_MAX (0x7f7fffff)
        // is ABOVE the largest bf16 normal (0x7f7f) and rounds UP to infinity.
        // Confirmed against the oracle; this is not a bug.
        check(to_b(65536.0f) == 0x4780, "bf16 65536 is an ordinary normal");
        check(of_b(0x7f7f) == 3.3895313892515355e+38f,
              "bf16 max normal is 0x7f7f");
        check(to_b(3.4028235e38f) == 0x7f80, "FLT_MAX overflows bf16 -> +inf");

        // arithmetic goes through float, and half OP half stays half
        static_assert(std::is_same<decltype(half(1) + half(1)), half>::value,
                      "half + half -> half");
        check(static_cast<float>(half(1.5f) * half(2.0f)) == 3.0f,
              "half multiply");
        half acc(0.0f);
        acc += half(0.5f);
        acc += half(0.25f);
        check(static_cast<float>(acc) == 0.75f, "half operator+=");
        check(half(2.0f) > half(1.0f) && !(half(2.0f) < half(1.0f)),
              "half compare");
        check(static_cast<float>(-half(3.0f)) == -3.0f, "half unary minus");

        // mixed-type arithmetic must (a) compile and (b) widen, not narrow.
        // Without FF_HALF_MIXED every one of these is ambiguous, which is what
        // stops the kernels compiling at all.
        static_assert(std::is_same<decltype(half(1) * 2.0f), float>::value,
                      "half * float  -> float");
        static_assert(std::is_same<decltype(2.0 * half(1)), double>::value,
                      "double * half -> double");
        static_assert(std::is_same<decltype(half(1) - 1), int>::value,
                      "half - int    -> int");
        check(half(1.5f) * 2.0f == 3.0f, "half * float value");
        check(2.0 * half(1.5f) == 3.0, "double * half value");
        check(half(2.0f) > 1.0f, "half vs float compare");

        // 16-bit accumulation loses; float accumulation does not. 2049 ones:
        // half cannot represent 2049 (the gap above 2048 is 2), so a half
        // accumulator sticks. compute_type<half> == float is why we don't.
        {
            ff::compute_type<half>::type wide = 0;
            half narrow(0.0f);
            for (int i = 0; i < 2049; ++i) {
                wide += 1.0f;
                narrow += half(1.0f);
            }
            check(wide == 2049.0f, "float accumulator reaches 2049");
            check(
                static_cast<float>(narrow) == 2048.0f,
                "half accumulator saturates at 2048 (why compute_type exists)");
        }

        std::printf("  [%s] named cases\n", failures == local ? "ok" : "!!");
    }

    if (failures) {
        std::printf("FAILED: %d check(s)%s\n", failures,
                    failures > MAX_REPORT ? " (output truncated)" : "");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
