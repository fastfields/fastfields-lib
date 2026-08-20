// Compile-only probe: every helper in impl/kernels/utils.h must stay
// host-callable (FF_CUHOSTDEV), and this TU is what makes nvcc say so.
//
// Why this exists (fastfields-lib#150)
// ---------------------------------------------------------------------------
// `typed_prod` and `prod` were `inline FF_CUDEV`, i.e. `__device__` only, while
// three separate host-side callers used them:
//
//   * `utils.h`'s own `FF_CUHOST canUse32BitIndexMath` -> `typed_prod`, which
//     every `FF_CANUSE32BITS` in src/lib-cuda goes through;
//   * every `FF_CUHOST` launcher in impl/cuda/{reg_field,reg_flow,
//     distance_euclidean,distance_l1,distance_mesh}.h -> `prod(size, n)`,
//     to size its grid;
//   * `impl/kernels/distance/mesh.h`'s `FF_CUHOST build_tree` -> `max`.
//
// nvcc did not reject any of them, and the reason is the whole point of this
// file: **nvcc diagnoses a host->device call only when the calling function is
// not itself a template.** All three callers above are templates, so the check
// was skipped, and cudafe++ emitted into the *host* object, in place of the
// callee's real body,
//
//     {int volatile ___ = 1; (void)args; ::exit(___);}
//
// Everything compiled, linked, passed `-Wl,--no-undefined` and `ldd -r`, and
// would have terminated the calling process with status 1 on the first call.
// At -O1 and above (which is what CI builds CUDA with) the host compiler also
// deleted every statement after the call, `exit` being `noreturn`, so the
// dispatch downstream of it vanished from the object entirely.
//
// What this file does
// ---------------------------------------------------------------------------
// It calls each helper from a **plain, non-template `__host__` function** --
// the one shape nvcc *does* check. Mark any of them `FF_CUDEV` again and this
// TU fails with nvcc's own
//
//     error: calling a __device__ function("ff::cuda::prod<...>") from a
//            __host__ function("...") is not allowed
//
// which is exactly the diagnostic the real code could not produce. It is
// compiled, never linked, never run, and costs a couple of seconds.
//
// The second half explicitly instantiates one representative `FF_CUHOST`
// launcher per impl/cuda header that calls into these helpers. An explicit
// instantiation type-checks the launcher body in nvcc's device pass, where the
// host->device call edge *is* reported -- so this half catches a bad edge
// introduced anywhere in a launcher, not just the helpers enumerated above.
// Note that compile_probe_mesh.cu documents this same technique as a "spurious
// artifact of the probe technique, not a defect". That reading was wrong: the
// error it saw was #150, reported correctly by the compiler and dismissed.
//
// What it does NOT buy: anything about device code, and anything semantic. A
// helper that is host-callable but computes the wrong thing compiles happily.

#include <fastfields/impl/kernels/utils.h>
#include <fastfields/impl/cuda/distance_euclidean.h>
#include <fastfields/impl/cuda/distance_l1.h>

namespace U = ff::cuda;

// ---------------------------------------------------------------------------
// utils.h, one non-template host caller per helper.
// ---------------------------------------------------------------------------

double ff_probe_hostdev_scalars(double a, double b, int n)
{
    double x = a, y = b;
    U::swap(x, y);
    double acc = 0;
    acc += U::square(x);
    acc += U::sqrt(y < 0 ? -y : y);
    acc += U::pow<3>(x);
    acc += U::pow(x, n);
    acc += U::min(x, y);
    acc += U::max(x, y);
    acc += U::abs(x);
    acc += U::sign(x);
    acc += U::mod(x, y);
    return acc;
}

long ff_probe_hostdev_arrays(const long * size, unsigned long ndim,
                             long * out, const long * inp)
{
    long acc = 0;

    // The two `typed_prod` overloads and the two `prod` overloads. The dynamic
    // `typed_prod` is the exact edge `canUse32BitIndexMath` takes.
    acc += U::typed_prod<long>(size, ndim);
    acc += U::typed_prod<long, 3>(size);
    acc += U::prod(size, ndim);
    acc += U::prod<3>(size);

    // fill / fillfrom, static and dynamic rank, with and without a stride.
    long buf[4];
    U::fillfrom<4>(buf, inp);
    U::fillfrom<4>(buf, inp, 2L);
    U::fillfrom(4, out, inp);
    U::fillfrom(4, out, inp, 2L);
    U::fill<4>(out, 1L);
    U::fill<4>(out, 1L, 2L);

    acc += buf[0];
    return acc;
}

// `canUse32BitIndexMath` itself: a host-only function template, so nvcc cannot
// check *its* body -- but instantiating it from here at least keeps the
// signature honest, and it is the call every src/lib-cuda dispatch makes.
bool ff_probe_hostdev_canuse32(int ndim, const long * size, const long * stride)
{
    return U::canUse32BitIndexMath(ndim, size, stride);
}

// ---------------------------------------------------------------------------
// One explicit instantiation per launcher header that reaches those helpers.
// This is the half that generalises: it re-checks the whole launcher body.
// ---------------------------------------------------------------------------

template void ff::cuda::distance_e::dt<float, int>(
    int, float *, float, const int *, const int *, intptr_t);
template void ff::cuda::distance_l1::dt<float, int>(
    int, float *, float, const int *, const int *, intptr_t);
