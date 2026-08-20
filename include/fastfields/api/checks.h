#ifndef FF_LIB_CHECKS
#define FF_LIB_CHECKS
#include <stdexcept>
#include <fastfields/core/dlpack.h>
#include <fastfields/core/defines.h>

FF_NAMESPACE_BEGIN(FF_NS)

/**
 * Assert that a set of DLTensors all live on the same device.
 *
 * The hub dispatches each op on a single operand's device and forwards every
 * tensor to that backend. If the operands disagree (e.g. `out` on CPU but
 * `inp` on CUDA), the wrong backend runs and reads a device pointer as host
 * memory (or vice-versa) -> segfault / silent garbage. Call this at the top of
 * every public entry, with the dispatch tensor as `ref` and every other real
 * tensor argument as the variadic tail, to reject the mismatch up front.
 *
 * Compares both `device_type` and `device_id`. Only pass tensors that carry
 * real data; optional/placeholder tensors (null data) should be excluded or
 * guarded by the caller (`if (w.data) require_same_device(ref, w)`).
 */
inline void require_same_device(const DLTensor & /*ref*/) {}

/**
 * The hub's device predicates.
 *
 * Every `src/lib/<module>.cpp` carried its own `IS_CPU` / `IS_CUDA` macro pair
 * -- nine identical copies, and two more unprefixed macros leaking out of the
 * translation units that defined them. They are plain predicates over a POD
 * field, so they are stated here as inline functions instead: a function in
 * `ff::` cannot collide with a downstream identifier the way a bare `IS_CPU`
 * macro can, and so it needs no prefix to be safe. Two of the 31 unprefixed
 * macros are thus removed rather than renamed.
 *
 * `kDLCUDAHost` is pinned (page-locked) *host* memory: addressable by the CPU,
 * so it belongs on the CPU side of the dispatch, not the CUDA one.
 */
inline bool is_cuda(const DLTensor & t)
{
    return t.device.device_type == DLDeviceType::kDLCUDA;
}

inline bool is_cpu(const DLTensor & t)
{
    return t.device.device_type == DLDeviceType::kDLCPU ||
           t.device.device_type == DLDeviceType::kDLCUDAHost;
}

template <class... Rest>
inline void require_same_device(const DLTensor & ref, const DLTensor & t, const Rest &... rest) {
    if (t.device.device_type != ref.device.device_type ||
        t.device.device_id   != ref.device.device_id)
        throw std::invalid_argument("fastfields: all tensors must be on the same device");
    require_same_device(ref, rest...);
}

FF_NAMESPACE_END(FF_NS)

#endif // FF_LIB_CHECKS
