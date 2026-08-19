#pragma once
#ifndef FF_HALF
#define FF_HALF
/**
 * Half-precision element types: `half` (IEEE binary16) and `bfloat16` -- the
 * two 16-bit floats PyTorch exposes, and the two DLPack can describe
 * (`kDLFloat` at 16 bits, and `kDLBfloat`).
 *
 * Under nvcc these ARE the native CUDA types (`__half` / `__nv_bfloat16`), so
 * device kernels get hardware half math. On a host compiler (g++/clang++, no
 * CUDA) we fall back to portable software structs with the identical 16-bit
 * layout, so the same buffer means the same thing on both sides of the
 * device-dispatch boundary.
 *
 * Force the portable types even under nvcc with `-DFF_PORTABLE_HALF` -- the
 * escape hatch that makes the software conversions testable under nvcc, and
 * lets a host-side reference be diffed against the hardware one on a real GPU.
 *
 * `compute_type<T>` names the type math should ACCUMULATE in: `float` for both
 * half types (the usual mixed-precision rule -- accumulate wider than you
 * store), `T` for everything else. The kernels already separate the
 * accumulator (`reduce_t`) from the storage type (`scalar_t`), so this trait is
 * what a dispatch layer would use to pick `reduce_t` when `scalar_t` is 16-bit.
 *
 * STATUS: prototype. Nothing else in the tree includes this header yet; see the
 * cost analysis in the PR that introduced it before wiring it into a dispatch
 * layer.
 *
 * ---------------------------------------------------------------------------
 * PROVENANCE. Ported from `balbasty/teeny`, `include/teeny/half.h` (MIT,
 * (c) 2023-2025 Yael Balbastre). teeny and this repository are both MIT and
 * share a copyright holder, so the copy carries no obligation beyond keeping
 * this notice. Changes made for fastfields:
 *   - `<cuda/std/type_traits>` / `<cuda/std/cstdint>` -> `<type_traits>` and
 *     the `<cstdint>` that `core/cuda_switch.h` already provides. libcu++
 *     (CCCL) enforces a C++17 floor; fastfields is C++11 (C++14 for nvcc) and
 *     has no CCCL dependency.
 *   - `cs::enable_if_t<...>` (C++14) -> `typename std::enable_if<...>::type`.
 *   - `_TNY_API` -> `CUHOSTDEV`; `TNY_*` guards/macros -> `FF_*`.
 *   - float<->bits punning goes through `memcpy`, not a union; see `f2u` below.
 *   - the NSDMI on `bits` is dropped; see the note above FF_HALF_TYPE.
 *   - mixed-type (half OP float/double/int) operators added; see FF_HALF_MIXED.
 *     Without them the kernels do not compile at all -- teeny does not need
 *     them because its engines convert to `compute_type` before any arithmetic.
 *   - namespace `tny` -> `ff` (device-INdependent, like `core/autocast.h`, not
 *     `ff::<FF_DEVICE>`).
 * ---------------------------------------------------------------------------
 */
#include "fastfields/core/cuda_switch.h"

#if defined(__CUDACC__) && !defined(FF_PORTABLE_HALF)
#  define FF_CUDA_HALF 1
// cuda_switch.h already pulls in <cuda_fp16.h> under nvcc; bf16 is separate.
#  include <cuda_fp16.h>
#  include <cuda_bf16.h>
#else
#  include <cstring>      // memcpy
#  include <type_traits>  // enable_if, is_arithmetic
#endif

FF_NAMESPACE_BEGIN(FF)

#ifdef FF_CUDA_HALF

/** IEEE binary16 -- the native CUDA `__half` under nvcc. */
typedef __half        half;
/** bfloat16 -- the native CUDA `__nv_bfloat16` under nvcc. */
typedef __nv_bfloat16 bfloat16;

#else  // ---- portable software fallback ----------------------------------

namespace _half_detail {

// float <-> raw bits.
//
// teeny punned through `union { float f; uint32_t u; }`. Reading the inactive
// member of a union is undefined behaviour in ISO C++ (it is *defined* in C,
// and gcc/clang/msvc/nvcc all document it as a supported extension, which is
// why teeny gets away with it). memcpy is the conforming spelling and every one
// of those compilers folds it to the same single register move at -O1 and
// above, so there is no reason to keep the UB.
inline CUHOSTDEV uint32_t f2u(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

inline CUHOSTDEV float u2f(uint32_t u)
{
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

// float32 -> IEEE binary16 (round to nearest, ties to even)
inline CUHOSTDEV uint16_t f32_to_f16(float f)
{
    uint32_t x    = f2u(f);
    uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
    uint32_t exp  = (x >> 23) & 0xffu;
    uint32_t mant = x & 0x7fffffu;
    if (exp == 0xff)                                  // inf / nan
        return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
    int e = static_cast<int>(exp) - 127 + 15;
    if (e >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00u);  // overflow -> inf
    if (e <= 0) {                                     // subnormal / zero
        if (e < -10) return sign;
        mant |= 0x800000u;
        uint32_t shift = static_cast<uint32_t>(14 - e);
        uint16_t h     = static_cast<uint16_t>(mant >> shift);
        uint32_t rem   = mant & ((1u << shift) - 1u);
        uint32_t halfw = 1u << (shift - 1);
        if (rem > halfw || (rem == halfw && (h & 1))) ++h;
        return static_cast<uint16_t>(sign | h);
    }
    uint16_t h   = static_cast<uint16_t>((static_cast<uint32_t>(e) << 10) | (mant >> 13));
    uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;
    return static_cast<uint16_t>(sign | h);
}

// IEEE binary16 -> float32 (always exact)
inline CUHOSTDEV float f16_to_f32(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {                                        // subnormal -> normal
            int e = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; --e; }
            mant &= 0x3ffu;
            out = sign | (static_cast<uint32_t>(e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    return u2f(out);
}

// float32 <-> bfloat16 (round to nearest, ties to even)
inline CUHOSTDEV uint16_t f32_to_bf16(float f)
{
    uint32_t x = f2u(f);
    if (((x >> 23) & 0xffu) == 0xffu && (x & 0x7fffffu))   // nan -> quiet nan
        return static_cast<uint16_t>((x >> 16) | 0x40u);
    uint32_t r = x + 0x7fffu + ((x >> 16) & 1u);
    return static_cast<uint16_t>(r >> 16);
}

inline CUHOSTDEV float bf16_to_f32(uint16_t h)
{
    return u2f(static_cast<uint32_t>(h) << 16);
}

} // namespace _half_detail

// Mixed-type arithmetic. Without these, `half * 2.0f` -- and every
// `reduce_t OP scalar_t` the kernels are full of -- is AMBIGUOUS: the implicit
// converting constructor makes `operator*(half, half)` viable, the implicit
// `operator float()` makes the built-in `operator*(float, float)` viable, and
// neither is better than the other. Spelling the heterogeneous overloads out
// picks the one the kernels want (the WIDER type: the mixed-precision rule
// again) and does it without an `explicit` constructor, which would instead
// break every `scalar_t x = <reduce_t expr>` site. This is what PyTorch's
// c10::Half does, for the same reason.
#define FF_HALF_MIXED(NAME, W)                                                                      \
inline CUHOSTDEV W operator+(NAME a, W b) { return static_cast<W>(static_cast<float>(a)) + b; }     \
inline CUHOSTDEV W operator+(W a, NAME b) { return a + static_cast<W>(static_cast<float>(b)); }     \
inline CUHOSTDEV W operator-(NAME a, W b) { return static_cast<W>(static_cast<float>(a)) - b; }     \
inline CUHOSTDEV W operator-(W a, NAME b) { return a - static_cast<W>(static_cast<float>(b)); }     \
inline CUHOSTDEV W operator*(NAME a, W b) { return static_cast<W>(static_cast<float>(a)) * b; }     \
inline CUHOSTDEV W operator*(W a, NAME b) { return a * static_cast<W>(static_cast<float>(b)); }     \
inline CUHOSTDEV W operator/(NAME a, W b) { return static_cast<W>(static_cast<float>(a)) / b; }     \
inline CUHOSTDEV W operator/(W a, NAME b) { return a / static_cast<W>(static_cast<float>(b)); }     \
inline CUHOSTDEV bool operator==(NAME a, W b) { return static_cast<W>(static_cast<float>(a)) == b; }\
inline CUHOSTDEV bool operator==(W a, NAME b) { return a == static_cast<W>(static_cast<float>(b)); }\
inline CUHOSTDEV bool operator!=(NAME a, W b) { return static_cast<W>(static_cast<float>(a)) != b; }\
inline CUHOSTDEV bool operator!=(W a, NAME b) { return a != static_cast<W>(static_cast<float>(b)); }\
inline CUHOSTDEV bool operator< (NAME a, W b) { return static_cast<W>(static_cast<float>(a)) <  b; }\
inline CUHOSTDEV bool operator< (W a, NAME b) { return a <  static_cast<W>(static_cast<float>(b)); }\
inline CUHOSTDEV bool operator> (NAME a, W b) { return static_cast<W>(static_cast<float>(a)) >  b; }\
inline CUHOSTDEV bool operator> (W a, NAME b) { return a >  static_cast<W>(static_cast<float>(b)); }\
inline CUHOSTDEV bool operator<=(NAME a, W b) { return static_cast<W>(static_cast<float>(a)) <= b; }\
inline CUHOSTDEV bool operator<=(W a, NAME b) { return a <= static_cast<W>(static_cast<float>(b)); }\
inline CUHOSTDEV bool operator>=(NAME a, W b) { return static_cast<W>(static_cast<float>(a)) >= b; }\
inline CUHOSTDEV bool operator>=(W a, NAME b) { return a >= static_cast<W>(static_cast<float>(b)); }

// A portable half type: a uint16 store plus convert-through-float arithmetic.
//
// NOTE on `bits`. teeny spells this `uint16_t bits{};` with `NAME() = default;`
// -- legal C++11, but the NSDMI makes the defaulted default constructor
// NON-TRIVIAL, so `half x;` in a kernel silently zero-stores and
// `__shared__ half tile[N];` (which requires a trivial default constructor)
// would not compile under nvcc. Dropping the NSDMI restores a trivial default
// constructor, so the type behaves exactly as `float` does. It costs nothing:
// the class is a non-aggregate in C++11 either way, because the user-provided
// templated converting constructor below already disqualifies it -- the NSDMI
// is not what pays for that. The type stays TRIVIALLY COPYABLE, which is the
// property DLPack buffers, memcpy and by-value kernel arguments actually need;
// tests/core/test_half.cpp static_asserts both.
#define FF_HALF_TYPE(NAME, TO_F, FROM_F)                                                            \
struct NAME {                                                                                       \
    uint16_t bits;                                                                                  \
    NAME() = default;                                                                               \
    template <class U, typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>        \
    CUHOSTDEV NAME(U v) : bits(FROM_F(static_cast<float>(v))) {}                                    \
    CUHOSTDEV operator float() const { return TO_F(bits); }                                         \
    CUHOSTDEV NAME operator-() const { return NAME(-TO_F(bits)); }                                  \
    CUHOSTDEV NAME & operator+=(NAME o) { *this = NAME(TO_F(bits) + TO_F(o.bits)); return *this; }  \
    CUHOSTDEV NAME & operator-=(NAME o) { *this = NAME(TO_F(bits) - TO_F(o.bits)); return *this; }  \
    CUHOSTDEV NAME & operator*=(NAME o) { *this = NAME(TO_F(bits) * TO_F(o.bits)); return *this; }  \
    CUHOSTDEV NAME & operator/=(NAME o) { *this = NAME(TO_F(bits) / TO_F(o.bits)); return *this; }  \
};                                                                                                  \
inline CUHOSTDEV NAME operator+(NAME a, NAME b) { return NAME(TO_F(a.bits) + TO_F(b.bits)); }       \
inline CUHOSTDEV NAME operator-(NAME a, NAME b) { return NAME(TO_F(a.bits) - TO_F(b.bits)); }       \
inline CUHOSTDEV NAME operator*(NAME a, NAME b) { return NAME(TO_F(a.bits) * TO_F(b.bits)); }       \
inline CUHOSTDEV NAME operator/(NAME a, NAME b) { return NAME(TO_F(a.bits) / TO_F(b.bits)); }       \
inline CUHOSTDEV bool operator==(NAME a, NAME b) { return TO_F(a.bits) == TO_F(b.bits); }           \
inline CUHOSTDEV bool operator!=(NAME a, NAME b) { return TO_F(a.bits) != TO_F(b.bits); }           \
inline CUHOSTDEV bool operator< (NAME a, NAME b) { return TO_F(a.bits) <  TO_F(b.bits); }           \
inline CUHOSTDEV bool operator> (NAME a, NAME b) { return TO_F(a.bits) >  TO_F(b.bits); }           \
inline CUHOSTDEV bool operator<=(NAME a, NAME b) { return TO_F(a.bits) <= TO_F(b.bits); }           \
inline CUHOSTDEV bool operator>=(NAME a, NAME b) { return TO_F(a.bits) >= TO_F(b.bits); }           \
FF_HALF_MIXED(NAME, float)                                                                          \
FF_HALF_MIXED(NAME, double)                                                                         \
FF_HALF_MIXED(NAME, int)

FF_HALF_TYPE(half,     _half_detail::f16_to_f32,  _half_detail::f32_to_f16)
FF_HALF_TYPE(bfloat16, _half_detail::bf16_to_f32, _half_detail::f32_to_bf16)
#undef FF_HALF_TYPE
#undef FF_HALF_MIXED

#endif // FF_CUDA_HALF

static_assert(sizeof(half) == 2 && sizeof(bfloat16) == 2,
              "half types must be 16-bit: the DLPack buffer layout depends on it");

/**
 * The type math should ACCUMULATE / compute in for element type `T`.
 * Half types compute in `float`; everything else computes in itself.
 */
template <typename T> struct compute_type           { typedef T     type; };
template <>           struct compute_type<half>     { typedef float type; };
template <>           struct compute_type<bfloat16> { typedef float type; };

FF_NAMESPACE_END(FF)

#endif // FF_HALF
