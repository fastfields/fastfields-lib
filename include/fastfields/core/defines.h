#pragma once
#ifndef FF_DEFINES
#define FF_DEFINES

// Merged from fastfields-kernels/defines.h and fastfields-lib/defines.h during
// the six-repo consolidation. The first three macros were token-identical in
// both; the namespace-device pair came from kernels and the FF_CPU/FF_CUDA pair
// from the hub. One file, one guard -- so a bare `#include "defines.h"` can no
// longer resolve to a different header than the author meant.

// `FF_NS` is the project's root namespace spelled once, so that
// `FF_NAMESPACE_BEGIN(FF_NS)` is the only place any header names it. It was
// called `FF` until the public-macro prefixing pass: a two-letter, all-caps
// macro in an installed header takes that name away from every translation
// unit downstream of us, and `FF` is an entirely plausible downstream
// identifier. `#undef`-ing it at the end of this header is NOT an alternative
// -- it is used by ~105 other files *after* including this one, and undefining
// it would quietly turn every `FF_NAMESPACE_BEGIN(FF)` into a namespace
// literally named `FF` rather than into an error.
#define FF_NS                       ff
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
