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
#else
#include <cuda_fp16.h>
#define CUGLOB __global__
#define CUHOST __host__
#define CUDEV __device__
#define CUHOSTDEV __host__ __device__
#endif

#endif // FF_CUDA_SWITCH
