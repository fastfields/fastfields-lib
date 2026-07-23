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

#include "defines.h"

#endif // FF_CUDA_SWITCH
