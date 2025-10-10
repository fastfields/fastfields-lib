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

#include <cuda_fp16.h>
#define CUGLOB __global__
#define CUHOST __host__
#define CUDEV  __device__
#define CUHOSTDEV __host__ __device__
#define FF_DEVICE cuda

#define int8_t      signed char
#define int16_t     short
#define int32_t     int
#define int64_t     long
#define uint8_t     unsigned char
#define uint16_t    unsigned short
#define uint32_t    unsigned int
#define uint64_t    unsigned long

#endif

#include "defines.h"

#endif // FF_CUDA_SWITCH
