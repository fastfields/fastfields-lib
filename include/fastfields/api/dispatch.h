#ifndef FF_API_DISPATCH
#define FF_API_DISPATCH
/**
 * Private (not installed) header: the ndim x bound x dtype x index-width
 * dispatch pyramid, stated once.
 *
 * PROTOTYPE -- see the accompanying proposal. Every module under src/lib-cpu
 * and src/lib-cuda carries its own hand-written copy of that pyramid as nested
 * function-like macros: `NDIM_SWITCH`, `BOUND_SWITCH`, `BND1`/`BND2`/`BND3`,
 * one `<OP>_DT` per entry point and one `<OP>_ARGS` per entry point. The
 * switches are byte-identical across modules; only the leaf call differs. This
 * header factors the switches out and leaves the leaf as a template argument.
 *
 * Usage
 * -----
 * Wrap the module's leaf in a struct with a single static `run`, templated on
 * exactly the axes the pyramid resolves:
 *
 *     struct field_matvec_op {
 *         template <int ndim, class scalar_t, class offset_t, bound::type... B>
 *         static void run(<the wrapper's concrete parameter list>) { ... }
 *     };
 *
 * and call it:
 *
 *     dispatch_nbd<field_matvec_op>(key, arg0, arg1, ...);
 *
 * The arguments are an ordinary function-call argument list, so the
 * `#define <OP>_ARGS ... #undef` pairs disappear -- and with them the GNU
 * `args...` named-variadic-macro syntax, which MSVC's traditional preprocessor
 * does not accept.
 *
 * The boundary condition is expanded to a pack of `ndim` copies, exactly as
 * BND1/BND2/BND3 did: `bound::getutils<B>` / `<B,B>` / `<B,B,B>` detect
 * isotropy from the pack length, so the length is load-bearing.
 * tests/dispatch/packcheck.cpp checks that it still is.
 *
 * C++11: no `if constexpr`, no fold expressions, no variable templates, no
 * generic lambdas. Compiles under clang++ and g++ at -std=c++11 and
 * -std=c++14 (the CUDA layer's standard).
 */
#include <stdexcept>
#include <cstdint>
#include "fastfields/core/dlpack.h"
#include "fastfields/core/cuda_switch.h"
#include "fastfields/impl/kernels/bounds.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

/***********************************************************************
 *                        INDEX-WIDTH POLICY                           *
 ***********************************************************************/
//
// Whether the 32-bit index specialisation (`offset_t == int32_t`) is compiled
// at all.
//
// This is the single largest multiplier in the whole build. Every templated
// kernel below the dispatch layer takes `offset_t`, and `offset_t` has exactly
// two values, chosen at run time by `canUse32BitIndexMath`. So the axis costs
// exactly x2 instantiations of everything -- and on the CUDA side "everything"
// is device code, which is ptxas memory and SASS in the shipped library.
//
// It is stated once, here, instead of being hard-coded into ~40 `<OP>_DT`
// macros across 21 translation units, so that turning it off is one flag:
//
//     make cpu  CXXFLAGS='-std=c++11 -O3 -DFF_INDEX32=0'
//
// With FF_INDEX32=0 the key's `use_32bits` field is ignored, `offset_t` is
// always int64_t, and the whole narrowing path in core/autocast.h collapses to
// a no-op passthrough.
//
// Nothing in this repository measures what the 32-bit path buys. It is
// inherited from ATen's `canUse32BitIndexMath`, where it is a GPU
// register-pressure optimisation; here it is applied to the CPU backend too,
// and on the CUDA side it is paid for with a `cudaMallocHost` per narrowed
// array per call (see core/autocast.h).
#ifndef FF_INDEX32
#  define FF_INDEX32 1
#endif

/***********************************************************************
 *                             THE KEY                                 *
 ***********************************************************************/

// The runtime axes the pyramid switches on. Built once per public entry point,
// immediately after the argument checks.
struct DispatchKey
{
    int            ndim;
    bound::type    bound;
    DLDataTypeCode code;
    uint8_t        bits;
    bool           use_32bits;

    inline DispatchKey(int ndim_, int8_t bound_, const DLDataType & dtype,
                       bool use_32bits_)
        : ndim(ndim_),
          bound(static_cast<bound::type>(bound_)),
          code(static_cast<DLDataTypeCode>(dtype.code)),
          bits(dtype.bits),
          use_32bits(use_32bits_)
    {}
};

/***********************************************************************
 *                    LEVEL 1 -- dtype x index width                   *
 ***********************************************************************/
// Innermost level. `Op::run` is called with every compile-time axis fixed; the
// boundary pack `B...` already holds `ndim` entries.
template <class Op, int ndim, bound::type... B>
struct _dispatch_dtype
{
    template <class... A>
    static inline void call(const DispatchKey & key, A &&... args)
    {
        switch (key.code) {
            case kDLFloat:
                switch (key.bits) {
#if FF_INDEX32
                    case 32: return key.use_32bits
                        ? Op::template run<ndim, float,  int32_t, B...>(args...)
                        : Op::template run<ndim, float,  int64_t, B...>(args...);
                    case 64: return key.use_32bits
                        ? Op::template run<ndim, double, int32_t, B...>(args...)
                        : Op::template run<ndim, double, int64_t, B...>(args...);
#else
                    case 32: return Op::template run<ndim, float,  int64_t, B...>(args...);
                    case 64: return Op::template run<ndim, double, int64_t, B...>(args...);
#endif
                    default: break;
                }
                break;
            default: break;
        }
        throw std::invalid_argument("only floating point data types are supported");
    }
};

/***********************************************************************
 *                 LEVEL 2 -- expand the boundary pack                 *
 ***********************************************************************/
// BND1/BND2/BND3, as three explicit specialisations rather than a recursion:
// the recursive form instantiates `ndim + 1` class templates (and as many
// member-function templates) per (op, ndim, bound) triple, which is pure
// front-end cost for no benefit.
template <class Op, int ndim, bound::type B> struct _expand_bound;

template <class Op, bound::type B>
struct _expand_bound<Op, 1, B>
{
    template <class... A>
    static inline void call(const DispatchKey & key, A &&... args)
    { _dispatch_dtype<Op, 1, B>::call(key, args...); }
};

template <class Op, bound::type B>
struct _expand_bound<Op, 2, B>
{
    template <class... A>
    static inline void call(const DispatchKey & key, A &&... args)
    { _dispatch_dtype<Op, 2, B, B>::call(key, args...); }
};

template <class Op, bound::type B>
struct _expand_bound<Op, 3, B>
{
    template <class... A>
    static inline void call(const DispatchKey & key, A &&... args)
    { _dispatch_dtype<Op, 3, B, B, B>::call(key, args...); }
};

/***********************************************************************
 *                    LEVEL 3 -- boundary condition                    *
 ***********************************************************************/
// The template argument is `FF_BOUND_<NAME>` (kernels/bounds.h): the condition
// itself when it is statically compiled, `Dynamic` otherwise, per BOUNDFLAGS.
// The switch labels stay exhaustive on the runtime value either way; only the
// instantiated template argument collapses onto the shared Dynamic path.
template <class Op, int ndim>
struct _dispatch_bound
{
    template <class... A>
    static inline void call(const DispatchKey & key, A &&... args)
    {
        switch (key.bound) {
            case bound::type::Zero:
                return _expand_bound<Op, ndim, FF_BOUND_ZERO     >::call(key, args...);
            case bound::type::Replicate:
                return _expand_bound<Op, ndim, FF_BOUND_REPLICATE>::call(key, args...);
            case bound::type::DCT1:
                return _expand_bound<Op, ndim, FF_BOUND_DCT1     >::call(key, args...);
            case bound::type::DCT2:
                return _expand_bound<Op, ndim, FF_BOUND_DCT2     >::call(key, args...);
            case bound::type::DST1:
                return _expand_bound<Op, ndim, FF_BOUND_DST1     >::call(key, args...);
            case bound::type::DST2:
                return _expand_bound<Op, ndim, FF_BOUND_DST2     >::call(key, args...);
            case bound::type::DFT:
                return _expand_bound<Op, ndim, FF_BOUND_DFT      >::call(key, args...);
            case bound::type::NoCheck:
                return _expand_bound<Op, ndim, FF_BOUND_NOCHECK  >::call(key, args...);
            default:
                throw std::invalid_argument("Unsupported boundary condition");
        }
    }
};

/***********************************************************************
 *                          LEVEL 4 -- ndim                            *
 ***********************************************************************/

/**
 * @brief ndim x bound x dtype x index-width dispatch.
 *
 * Replaces `NDIM_SWITCH(<OP>_DT)` together with the `<OP>_DT` and `<OP>_ARGS`
 * macro pair.
 */
template <class Op, class... A>
inline void dispatch_nbd(const DispatchKey & key, A &&... args)
{
    switch (key.ndim) {
        case 1: return _dispatch_bound<Op, 1>::call(key, args...);
        case 2: return _dispatch_bound<Op, 2>::call(key, args...);
        case 3: return _dispatch_bound<Op, 3>::call(key, args...);
        default:
            throw std::invalid_argument("Only 1D, 2D and 3D are supported");
    }
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_API_DISPATCH
