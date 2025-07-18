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
#else
#include <cuda_fp16.h>
#endif

#endif // FF_CUDA_SWITCH
