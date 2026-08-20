#pragma once
/**
 * Private (not installed) header: the ndim x bound x dtype x index-width
 * dispatch pyramid shared by the regulariser API surface -- `reg_field.cpp`
 * (multi-channel fields) and `reg_flow.cpp` (vector flows).
 *
 * Why one header for two modules
 * --------------------------------------------------------------------------
 * These two are one *surface*, not two: their pyramids were byte-identical.
 * Before this header, `BND1`/`BND2`/`BND3` and `BOUND_SWITCH` were literally
 * the same text in all six regulariser translation units (the two here, plus
 * `src/lib-cuda/reg_{field,flow}{,_rls}.cpp`), and `NDIM_SWITCH` differed in
 * exactly one word -- the noun in its diagnostic, "field" versus "flow". That
 * word is now the `NOUN` parameter, and it is the only thing the two modules
 * did not already share.
 *
 * The dtype x index-width fan is the part that had really multiplied: it was
 * copied once per *entry point*, not once per module -- thirteen times in
 * `reg_field.cpp` alone, fifty-two across the six regulariser TUs, out of
 * seventy-one in the whole tree. Every copy differed only in the leaf's name
 * and, for nine of them, one extra `char` template argument. Both variations
 * are parameters here, so the fan is written twice (once per leaf shape)
 * instead of fifty-two times.
 *
 * This follows `api/{cpu,cuda}/pushpull_dispatch.h`, which did the same for
 * the pushpull surface, and it is deliberately NOT a dispatcher generic over
 * all modules: the seven API surfaces have eight different leaf
 * template-argument orders and do not even agree on the order the axes are
 * resolved in (`splinc` dispatches dtype *outermost*; every surface here
 * dispatches it innermost). See fastfields-lib#94 for the measurement.
 *
 * Not in `core/`
 * --------------------------------------------------------------------------
 * Per the definition fastfields-lib#149 settled on, `core/` holds what more
 * than one layer needs and what is not the computation of a named fastfields
 * operation. This is tied to one named operation family's template signatures,
 * so it belongs in `api/`, beside the entry points whose leaves it names.
 *
 * Variadic macro syntax
 * --------------------------------------------------------------------------
 * The argument lists use ISO `...` / `__VA_ARGS__`, not the GNU named-variadic
 * `args...` extension the rest of the dispatch macros still use. That is a
 * portability down payment, not a fix: MSVC's *traditional* preprocessor also
 * mis-forwards `__VA_ARGS__` to a nested macro (it arrives as one argument),
 * so an MSVC build needs `/Zc:preprocessor` regardless. Nothing in this tree
 * builds under MSVC today and that claim is untested here.
 */
#include <stdexcept>
#include <cstdint>
#include <fastfields/core/dispatch.h>
#include <fastfields/core/dlpack.h>
#include <fastfields/core/cuda_switch.h>
#include <fastfields/impl/kernels/bounds.h>

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)

/***********************************************************************
 *                        THE BOUNDARY PACK                            *
 ***********************************************************************/

// The regulariser leaves take the boundary condition as a pack of `ndim`
// copies, and the length is load-bearing: `bound::getutils<B>` / `<B,B>` /
// `<B,B,B>` detect isotropy from it. These three expand one condition to the
// pack the leaf expects. They are macro *arguments* at every call site, so
// the commas they introduce are protected until they land in the template
// argument list itself.
#define FF_REG_BND1(B) B
#define FF_REG_BND2(B) B, B
#define FF_REG_BND3(B) B, B, B

/***********************************************************************
 *                  LEVEL 1 -- dtype x index width                     *
 ***********************************************************************/

// Two leaf shapes exist in this surface and there is no way to write them as
// one: nine entry points thread a compile-time `char` op ('=', '+', '-') as
// the leaf's second template argument, and seven (the `relax` and `_rls`
// families) have no op concept at all. Both macros take `OP` so that the
// levels above can forward one argument list to either; FF_REG_DT ignores it.
//
// `off32_t` is `int32_t` or, under `FF_INDEX32=0`, `int64_t` -- which is how
// the index axis collapses onto one instantiation. See core/dispatch.h.

// Leaf shape A: FN<ndim, scalar_t, offset_t, BOUND...>
#define FF_REG_DT(FN, NDIM, OP, BNDS, ...)                              \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? FN<NDIM, float,  off32_t, BNDS>(__VA_ARGS__)          \
                : FN<NDIM, float,  int64_t, BNDS>(__VA_ARGS__);         \
            case 64: return use_32bits                                  \
                ? FN<NDIM, double, off32_t, BNDS>(__VA_ARGS__)          \
                : FN<NDIM, double, int64_t, BNDS>(__VA_ARGS__);         \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

// Leaf shape B: FN<ndim, op, scalar_t, offset_t, BOUND...>
#define FF_REG_DT_OP(FN, NDIM, OP, BNDS, ...)                           \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? FN<NDIM, OP, float,  off32_t, BNDS>(__VA_ARGS__)      \
                : FN<NDIM, OP, float,  int64_t, BNDS>(__VA_ARGS__);     \
            case 64: return use_32bits                                  \
                ? FN<NDIM, OP, double, off32_t, BNDS>(__VA_ARGS__)      \
                : FN<NDIM, OP, double, int64_t, BNDS>(__VA_ARGS__);     \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

/***********************************************************************
 *                     LEVEL 2 -- boundary condition                   *
 ***********************************************************************/

// Which of these conditions gets a dedicated (static) instantiation and which
// shares the single Dynamic (runtime) one is a build-time choice -- see
// FF_STATIC_BOUND_* in impl/kernels/bounds.h. The switch labels stay
// exhaustive on the *runtime* value either way; only the instantiated
// template argument collapses onto the shared Dynamic path. `bvec`, which the
// call site passes through, carries the runtime condition for the ones that
// do.
#define FF_REG_BOUND(DT, FN, NDIM, OP, BND, ...)                                    \
    switch (bnd) {                                                                  \
        case bound::type::Zero:      DT(FN,NDIM,OP,BND(FF_BOUND_ZERO),     __VA_ARGS__); break; \
        case bound::type::Replicate: DT(FN,NDIM,OP,BND(FF_BOUND_REPLICATE),__VA_ARGS__); break; \
        case bound::type::DCT1:      DT(FN,NDIM,OP,BND(FF_BOUND_DCT1),     __VA_ARGS__); break; \
        case bound::type::DCT2:      DT(FN,NDIM,OP,BND(FF_BOUND_DCT2),     __VA_ARGS__); break; \
        case bound::type::DST1:      DT(FN,NDIM,OP,BND(FF_BOUND_DST1),     __VA_ARGS__); break; \
        case bound::type::DST2:      DT(FN,NDIM,OP,BND(FF_BOUND_DST2),     __VA_ARGS__); break; \
        case bound::type::DFT:       DT(FN,NDIM,OP,BND(FF_BOUND_DFT),      __VA_ARGS__); break; \
        case bound::type::NoCheck:   DT(FN,NDIM,OP,BND(FF_BOUND_NOCHECK),  __VA_ARGS__); break; \
        default: throw std::invalid_argument("Unsupported boundary condition");     \
    }

/***********************************************************************
 *                          LEVEL 3 -- ndim                            *
 ***********************************************************************/

// NOUN is the word the rank diagnostic uses ("field" / "flow"). It is the
// only text that was ever module-specific in this pyramid.
#define FF_REG_NDIM(DT, FN, OP, NOUN, ...)                                            \
    switch (ndim) {                                                                   \
        case 1: FF_REG_BOUND(DT,FN,1,OP,FF_REG_BND1,__VA_ARGS__); break;              \
        case 2: FF_REG_BOUND(DT,FN,2,OP,FF_REG_BND2,__VA_ARGS__); break;              \
        case 3: FF_REG_BOUND(DT,FN,3,OP,FF_REG_BND3,__VA_ARGS__); break;              \
        default: throw std::invalid_argument("Only 1D, 2D and 3D " NOUN " are supported"); \
    }

/***********************************************************************
 *                         THE TWO CALL FORMS                          *
 ***********************************************************************/

/**
 * Dispatch `FN` over ndim x bound x dtype x index width.
 *
 * The call site must already have `ndim`, `bnd`, `code`, `bits` and
 * `use_32bits` in scope -- the same five locals every regulariser entry point
 * computed before this header existed.
 *
 *   FF_DISPATCH_REG   (_field_relax,      "field", args...)   FN<D,S,O,B...>
 *   FF_DISPATCH_REG_OP(_field_matvec_acc, '+', "field", args...) FN<D,op,S,O,B...>
 */
#define FF_DISPATCH_REG(FN, NOUN, ...) \
    FF_REG_NDIM(FF_REG_DT, FN, '=', NOUN, __VA_ARGS__)

#define FF_DISPATCH_REG_OP(FN, OP, NOUN, ...) \
    FF_REG_NDIM(FF_REG_DT_OP, FN, OP, NOUN, __VA_ARGS__)

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
