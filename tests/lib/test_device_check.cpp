// Standalone unit test for the require_same_device guard (fastfields-lib #16).
//
// Compile + run directly (no full library link needed), from the repo root:
//     clang++ -std=c++11 -I.. tests/test_device_check.cpp -o /tmp/test_device_check
//     /tmp/test_device_check
// or from within tests/:  clang++ -std=c++11 -I.. test_device_check.cpp && ./a.out
//
// It constructs DLTensors on different devices and checks that
// require_same_device throws on mismatch and stays silent on agreement.

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include "../checks.h"

static DLTensor make_tensor(DLDeviceType type, int32_t id) {
    DLTensor t;             // zero-init the fields we don't care about
    t.data           = (void*)0x1;   // non-null: a "real" tensor
    t.device.device_type = type;
    t.device.device_id   = id;
    t.ndim           = 0;
    t.dtype.code     = 0;
    t.dtype.bits     = 32;
    t.dtype.lanes    = 1;
    t.shape          = nullptr;
    t.strides        = nullptr;
    t.byte_offset    = 0;
    return t;
}

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; } \
    else         { std::printf("ok:   %s\n", msg); } \
} while (0)

int main() {
    DLTensor cpu_a  = make_tensor(kDLCPU,  0);
    DLTensor cpu_b  = make_tensor(kDLCPU,  0);
    DLTensor cuda0  = make_tensor(kDLCUDA, 0);
    DLTensor cuda1  = make_tensor(kDLCUDA, 1);

    // 1. Matching devices must NOT throw.
    {
        bool threw = false;
        try { ff::require_same_device(cpu_a, cpu_b); }
        catch (const std::invalid_argument &) { threw = true; }
        CHECK(!threw, "matching CPU devices do not throw");
    }

    // 2. CPU vs CUDA (different device_type) must throw.
    {
        bool threw = false;
        try { ff::require_same_device(cpu_a, cuda0); }
        catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "CPU vs CUDA throws (device_type mismatch)");
    }

    // 3. Same type, different device_id must throw.
    {
        bool threw = false;
        try { ff::require_same_device(cuda0, cuda1); }
        catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "CUDA:0 vs CUDA:1 throws (device_id mismatch)");
    }

    // 4. Variadic: mismatch anywhere in the tail is caught.
    {
        bool threw = false;
        try { ff::require_same_device(cpu_a, cpu_b, cuda0); }
        catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "mismatch in variadic tail throws");
    }

    // 5. Single tensor (no tail) never throws.
    {
        bool threw = false;
        try { ff::require_same_device(cpu_a); }
        catch (const std::invalid_argument &) { threw = true; }
        CHECK(!threw, "single tensor does not throw");
    }

    if (failures) { std::printf("\n%d check(s) FAILED\n", failures); return 1; }
    std::printf("\nAll device-check assertions passed.\n");
    return 0;
}
