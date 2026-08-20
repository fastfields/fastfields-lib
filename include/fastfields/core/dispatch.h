#pragma once
#ifndef FF_CORE_DISPATCH
#define FF_CORE_DISPATCH

/**
 * The helpers shared by both dtype-dispatch layers (`src/lib-cpu` and
 * `src/lib-cuda`): unpack the public pointer ABI's arguments into what the
 * templated impl layer wants, and reject the combinations it cannot express.
 *
 * Every `src/lib-cpu/<module>.cpp` and `src/lib-cuda/<module>.cpp` did this
 * the same way -- add `byte_offset` to `data`, decide whether the shape and
 * stride arrays fit in 32 bits, validate the operands -- and each one carried
 * its own copy of the macros. There were 19 copies of `VOIDPTR`, 19 of
 * `CANUSE32BITS`, 19 of `CHECK_NO_LANES` and 17 of `CHECK_SAME`, and they had
 * already drifted apart; see the three preserved divergences below.
 *
 * Why `core/` and not `api/`
 * --------------------------------------------------------------------------
 * `src/lib-cuda` is compiled by **nvcc** while `src/lib-cpu` and `src/lib` use
 * the host compiler, so anything all of them share has to be backend-agnostic.
 * That is exactly what `include/fastfields/core/` is for. `api/checks.h` is
 * the hub's (host-only) validation header and is the wrong home for macros
 * nvcc must also digest. Note the CI consequence, which is intended: a change
 * under `core/` triggers every job, CUDA included.
 *
 * Usage notes
 * --------------------------------------------------------------------------
 * * `FF_CANUSE32BITS` calls `canUse32BitIndexMath` unqualified, so it must be
 *   expanded from inside `ff::cpu` / `ff::cuda` (i.e. `ff::FF_DEVICE`, where
 *   `core/autocast.h` declares it). Every dispatch source already is. It is
 *   left unqualified on purpose: qualifying it would change name lookup, and
 *   this header's contract is that the macros expand token-for-token to what
 *   the 19 local copies expanded to.
 * * The `FF_CHECK_*` macros expand to bare `if` / `for` statements, NOT to a
 *   `do { ... } while (0)`. That is deliberate: it is what the copies did, and
 *   wrapping them would silently change which statements a brace-less
 *   `if (cond) FF_CHECK_...(...);` guards. They are therefore not safe as the
 *   body of an unbraced `if` / `else`. Tightening that is a behaviour change
 *   and belongs in its own commit, measured against the CPU suite.
 */

#include <cstddef>      // size_t
#include <cstdint>      // int32_t (the shape loops), int64_t
#include <stdexcept>    // std::invalid_argument
#include <vector>       // as_weights
#include "fastfields/core/dlpack.h"
#include "fastfields/core/defines.h"     // FF_NAMESPACE_*
#include "fastfields/core/autocast.h"    // canUse32BitIndexMath

/***********************************************************************
 *                        POINTER MARSHALLING                          *
 ***********************************************************************/

// DLPack keeps the element offset out of `data`, so every call into the impl
// layer has to fold it back in.
#define FF_VOIDPTR(x)   (static_cast<void*>(static_cast<char*>(x.data) + x.byte_offset))
#define FF_CVOIDPTR(x)  (static_cast<const void*>(static_cast<const char*>(x.data) + x.byte_offset))

// PRESERVED DIVERGENCE 1/3.  `posdef.cpp` -- and only posdef.cpp -- guarded
// its CVOIDPTR against a null `data`: an absent optional operand is passed as
// a descriptor with `data == nullptr`, and offsetting a null pointer is
// undefined behaviour even when the result is never dereferenced. Given its
// own name rather than folded into FF_CVOIDPTR, so the difference is visible
// at the call site instead of hiding in one file's private prologue.
#define FF_CVOIDPTR_OR_NULL(x)  (x.data ? FF_CVOIDPTR(x) : nullptr)

// Can this tensor's shape/stride arithmetic be narrowed to 32-bit offsets?
// See core/autocast.h.
#define FF_CANUSE32BITS(x) (canUse32BitIndexMath(x.ndim, x.shape, x.strides))

/***********************************************************************
 *                              CHECKS                                 *
 ***********************************************************************/

#define FF_CHECK_NO_LANES(tensor)                                       \
    if (tensor.dtype.lanes > 1)                                         \
        throw std::invalid_argument("Only scalar data types are supported");

#define FF_CHECK_SAME(X, Y, msg)                                        \
    if (X != Y) throw std::invalid_argument(msg);

#define FF_CHECK_SAME_DTYPE(X, Y)                                       \
    if ((X.dtype.code  != Y.dtype.code) ||                              \
        (X.dtype.bits  != Y.dtype.bits) ||                              \
        (X.dtype.lanes != Y.dtype.lanes))                               \
        throw std::invalid_argument("Tensors do not have the same data type");

// Agreement on the leading `D` (batch) dimensions.
#define FF_CHECK_SAME_BATCH(X, Y, D)                                    \
    for (int32_t d=0; d < D; ++d)                                       \
        if (X.shape[d] != Y.shape[d])                                   \
            throw std::invalid_argument("Tensors do not have the same batch shape");

// PRESERVED DIVERGENCE 2/3.  As above, but first rejects tensors with fewer
// than `D` dimensions rather than reading past the end of `shape`.
// `distance.cpp` and `posdef.cpp` are the two that need it -- they derive `D`
// from an operand's own `ndim` -- and were the only two that had it.
#define FF_CHECK_SAME_BATCH_ND(X, Y, D)                                 \
    if (X.ndim < D || Y.ndim < D)                                       \
        throw std::invalid_argument("Number of dimensions does not match");  \
    FF_CHECK_SAME_BATCH(X, Y, D)

// PRESERVED DIVERGENCE 3/3.  `CHECK_SAME_SHAPE` was two *different macros
// sharing one name*: a 3-argument leading-D check in the regularisers and
// solve_field (this one), and a 2-argument whole-shape check in distance.cpp
// (below). Merging them under one name would have silently changed one set of
// call sites, which is the sharpest illustration of why 17 private copies of a
// macro is a hazard rather than a tidiness complaint.
#define FF_CHECK_SAME_SHAPE_N(X, Y, D)                                  \
    for (int32_t d=0; d < D; ++d)                                       \
        if (X.shape[d] != Y.shape[d])                                   \
            throw std::invalid_argument("Tensors do not have the same shape");

// Agreement on the WHOLE shape, rank included.
#define FF_CHECK_SAME_SHAPE(X, Y)                                       \
    if (X.ndim != Y.ndim)                                               \
        throw std::invalid_argument("Tensors do not have the same number of dimensions"); \
    FF_CHECK_SAME_BATCH_ND(X, Y, X.ndim)

/***********************************************************************
 *                     NON-TENSOR ARGUMENT MARSHALLING                 *
 ***********************************************************************/

FF_NAMESPACE_BEGIN(FF_NS)

/**
 * Build a length-`nc` penalty-weight vector from the ABI's `const double *`.
 *
 * The regulariser entry points take each energy term's weight as a raw pointer
 * that is either an `nc`-long array or null ("this term is off"); the impl
 * layer takes a filled vector. Was copied verbatim into
 * `src/lib-cpu/reg_field.cpp` and `src/lib-cuda/reg_field{,_rls}.cpp`.
 *
 * Returns `std::vector<double>` rather than the dispatch sources' local
 * `reduce_t` typedef, which is `double` in every one of them. If a backend
 * ever changes its accumulation type, the assignment at the call site stops
 * compiling -- which is the failure you want, rather than a silent narrowing
 * inside a shared header.
 */
inline std::vector<double> as_weights(const double * w, int64_t nc)
{
    std::vector<double> v(static_cast<size_t>(nc), 0.0);
    if (w) for (int64_t c = 0; c < nc; ++c) v[static_cast<size_t>(c)] = w[c];
    return v;
}

FF_NAMESPACE_END(FF_NS)

#endif // FF_CORE_DISPATCH
