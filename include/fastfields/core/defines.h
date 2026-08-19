#pragma once
#ifndef FF_DEFINES
#define FF_DEFINES

// Merged from fastfields-kernels/defines.h and fastfields-lib/defines.h during
// the six-repo consolidation. The first three macros were token-identical in
// both; the namespace-device pair came from kernels and the FF_CPU/FF_CUDA pair
// from the hub. One file, one guard -- so a bare `#include "defines.h"` can no
// longer resolve to a different header than the author meant.

#define FF                          ff
#define FF_NAMESPACE_BEGIN(NAME)    namespace NAME {
#define FF_NAMESPACE_END(NAME)      }
#define FF_NAMESPACE_BEGIN_DEVICE   FF_NAMESPACE_BEGIN(FF_DEVICE)
#define FF_NAMESPACE_END_DEVICE     FF_NAMESPACE_END(FF_DEVICE)

#define FF_CPU cpu
#ifndef FF_WITH_CUDA
#  define FF_CUDA notimplemented
#else
#  define FF_CUDA cuda
#endif

#endif // FF_DEFINES
