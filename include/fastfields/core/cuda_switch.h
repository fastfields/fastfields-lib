#ifndef FF_CUDA_SWITCH
#define FF_CUDA_SWITCH


#ifndef __CUDACC__

// replace __device__ with empty symbol
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#define CUGLOB
#define CUHOST
#define CUDEV
#define CUHOSTDEV
#define FF_DEVICE cpu
#include <cstdint>

#else

#define CUGLOB __global__
#define CUHOST __host__
#define CUDEV  __device__
#define CUHOSTDEV __host__ __device__
#define FF_DEVICE cuda

#ifdef __CUDACC_RTC__
// NVRTC (runtime / JIT compilation) does not ship the standard library
// headers, so define the fixed-width integer types by hand.
#define int8_t      signed char
#define int16_t     short
#define int32_t     int
#define int64_t     long
#define uint8_t     unsigned char
#define uint16_t    unsigned short
#define uint32_t    unsigned int
#define uint64_t    unsigned long
#else
// Ahead-of-time nvcc compilation has <cstdint>; #defining the integer type
// names as macros would collide with its typedefs ("invalid combination of
// type specifiers" inside <cstdint>).
#include <cstdint>
#endif

#include <cuda_fp16.h>

#endif

// Force inlining.
//
// `inline` is a linkage keyword, not an inlining request. In a deeply templated
// stencil engine the cost heuristics routinely decide AGAINST inlining a helper
// that has to be inlined for the surrounding compile-time-bounded loops to
// unroll and the little fixed-size tap tables to stay in registers -- turning
// straight-line code into a real call per channel. Where that is a correctness-
// of-codegen requirement rather than a hint, say so.
//
// Spelled as an attribute only (never as `inline`), so it composes with an
// explicit `inline` / `CUDEV` without duplicating the specifier. nvcc accepts
// the GNU attribute on `__device__` functions on every platform we build for;
// anywhere else it degrades to the ordinary heuristics.
#if defined(__GNUC__) || defined(__clang__)
#  define FF_INLINE __attribute__((always_inline))
#else
#  define FF_INLINE
#endif

#include "defines.h"

#endif // FF_CUDA_SWITCH
