#ifndef FF_LIB_CHECKS
#define FF_LIB_CHECKS
#include <stdexcept>
#include "dlpack.h"
#include "defines.h"

FF_NAMESPACE_BEGIN(FF)

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

template <class... Rest>
inline void require_same_device(const DLTensor & ref, const DLTensor & t, const Rest &... rest) {
    if (t.device.device_type != ref.device.device_type ||
        t.device.device_id   != ref.device.device_id)
        throw std::invalid_argument("fastfields: all tensors must be on the same device");
    require_same_device(ref, rest...);
}

FF_NAMESPACE_END(FF)

#endif // FF_LIB_CHECKS
