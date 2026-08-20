#pragma once
#ifndef FF_CUDA_STREAM
#define FF_CUDA_STREAM

/**
 * The public ABI carries a CUDA stream as an `intptr_t` (no CUDA types leak
 * into the exported signatures); the cuda-impl launchers take a real
 * `cudaStream_t`. This is the one-line conversion between them, which had been
 * copied verbatim into all four regulariser dispatch sources
 * (`reg_field.cpp`, `reg_field_rls.cpp`, `reg_flow.cpp`, `reg_flow_rls.cpp`).
 *
 * CUDA-only by construction -- it names `cudaStream_t` -- so it lives under
 * `api/cuda/` rather than in `core/`, and only `src/lib-cuda` includes it.
 * `pushpull` has its own `_pp_stream` in the cuda-impl layer; the two are left
 * separate deliberately, as that one sits a layer below.
 */

#include <cstdint>
#include <fastfields/core/cuda_switch.h>

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)

// intptr_t -> cudaStream_t (0 == the default stream).
static inline cudaStream_t _reg_stream(intptr_t stream)
{
    return reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream));
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)

#endif // FF_CUDA_STREAM
