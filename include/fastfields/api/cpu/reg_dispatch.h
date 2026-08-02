#ifndef FF_REG_DISPATCH_H
#define FF_REG_DISPATCH_H

/***********************************************************************
 * Runtime (ndim, bound, dtype, offset-width) -> compile-time dispatch,
 * shared by `reg_field.cpp` and `reg_flow.cpp`.
 *
 * This replaces the hand-rolled `NDIM_SWITCH` / `BOUND_SWITCH` / per-op
 * `*_DT` macro pyramid that used to be copy-pasted into every regulariser
 * translation unit (fastfields-cpu-lib#69). The ndim x bound half is now
 * teeny's `dispatch_values` (teeny#465); the dtype x offset half stays an
 * ordinary branch, because it selects TYPES (`float`/`double`,
 * `int32_t`/`int64_t`) and `candidates<...>` only turns runtime values into
 * NON-TYPE template arguments.
 *
 * Routing is bit-for-bit the routing the macros performed, including the
 * order in which unsupported inputs are diagnosed:
 *
 *      ndim outside {1,2,3}  -> `ndim_msg` (caller-supplied: the field and
 *                               flow entry points word it differently)
 *      bound outside {0..7}  -> "Unsupported boundary condition"
 *      anything else         -> "only floating point data types are supported"
 *
 * i.e. a call that is wrong in more than one way reports the OUTERMOST
 * problem, exactly as the nested switches did.
 *
 * A caller supplies one `Op` per entry point: a four-line adapter whose
 * `run<D, scalar_t, offset_t, BOUND...>` names the templated worker to call.
 * Everything else -- the axis-rank match, the boundary match, repeating the
 * matched boundary across the axes, the dtype/offset leaf, and every throw --
 * lives here once.
 ***********************************************************************/

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <teeny/teeny.h>

#include "impl/kernels/cuda_switch.h"
#include "impl/kernels/bounds.h"
#include "dlpack.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_dispatch)

/***********************************************************************
 *              dtype x offset-width  (the old `*_DT` macros)          *
 ***********************************************************************/

// `D` and the boundary pack are already compile-time here; pick `scalar_t`
// from (code, bits) and `offset_t` from `use_32bits`, then call `Op`. The
// float32/float64 x int32/int64 fan-out and the rejection message are the
// ones the `MV_DT`/`DG_DT`/`KN_DT`/`RX_DT` macros carried.
template <class Op, int D, bound::type... BOUND, class... Args>
inline void dispatch_dtype(
    DLDataTypeCode code       ,
    uint8_t        bits       ,
    bool           use_32bits ,
    Args &&...     args       )
{
    if (code == kDLFloat) {
        if (bits == 32)
            return use_32bits
                ? Op::template run<D, float,  int32_t, BOUND...>(std::forward<Args>(args)...)
                : Op::template run<D, float,  int64_t, BOUND...>(std::forward<Args>(args)...);
        if (bits == 64)
            return use_32bits
                ? Op::template run<D, double, int32_t, BOUND...>(std::forward<Args>(args)...)
                : Op::template run<D, double, int64_t, BOUND...>(std::forward<Args>(args)...);
    }
    throw std::invalid_argument("only floating point data types are supported");
}

/***********************************************************************
 *      one boundary condition, repeated across the axes (BND1/2/3)    *
 ***********************************************************************/

// The impl layer takes one boundary condition PER AXIS (`bound::type...
// BOUND`); this dispatch boundary applies a single isotropic condition to all
// of them, which is what the old `BND1(B)` / `BND2(B) B,B` / `BND3(B) B,B,B`
// macros spelled out. `dispatch_values` hands back one compile-time value per
// DISTINCT runtime parameter, so turning the single matched boundary into a
// D-long pack -- where D is itself one of the dispatched parameters -- is this
// index-sequence fold. `same_bound<B, I>::value` is just `B` for every `I`, so
// expanding it over `make_index_sequence<D>` yields `B` repeated D times.
template <bound::type B, std::size_t>
struct same_bound { static constexpr bound::type value = B; };

template <class Op, int D, bound::type B, std::size_t... I, class... Args>
inline void repeat_bound(std::index_sequence<I...>, Args &&... args)
{
    dispatch_dtype<Op, D, same_bound<B, I>::value...>(std::forward<Args>(args)...);
}

/***********************************************************************
 *          ndim x bound  (the old `NDIM_SWITCH`/`BOUND_SWITCH`)       *
 ***********************************************************************/

// The whole pyramid, in one call: 3 axis ranks x 8 boundary conditions, with
// the candidate lists sitting next to each other so the instantiation budget
// is visible in one place. `candidates` takes the `bound::type` enum directly
// (no hand-written cast at the call site).
//
// `args` is (code, bits, use_32bits, <the worker's own arguments>) -- i.e. the
// old `MV_ARGS`/`DG_ARGS`/... macro bodies, passed as an ordinary pack.
template <class Op, class... Args>
inline void dispatch_nd_bound(
    int          ndim     ,
    bound::type  bnd      ,
    const char * ndim_msg ,
    Args &&...   args     )
{
    const bool matched = tny::dispatch_values(
        [&](auto D, auto B) {
            repeat_bound<Op, D.value, static_cast<bound::type>(B.value)>(
                std::make_index_sequence<static_cast<std::size_t>(D.value)>{},
                std::forward<Args>(args)...);
        },
        tny::candidates<1, 2, 3>(ndim),
        tny::candidates<static_cast<int>(bound::type::Zero)     ,
                        static_cast<int>(bound::type::Replicate),
                        static_cast<int>(bound::type::DCT1)     ,
                        static_cast<int>(bound::type::DCT2)     ,
                        static_cast<int>(bound::type::DST1)     ,
                        static_cast<int>(bound::type::DST2)     ,
                        static_cast<int>(bound::type::DFT)      ,
                        static_cast<int>(bound::type::NoCheck)  >(bnd));

    if (!matched) {
        // `dispatch_values` reports only THAT some parameter missed its list,
        // so name the culprit here -- outermost first, the order the nested
        // switches diagnosed in.
        if (ndim < 1 || ndim > 3)
            throw std::invalid_argument(ndim_msg);
        throw std::invalid_argument("Unsupported boundary condition");
    }
}

FF_NAMESPACE_END(reg_dispatch)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_REG_DISPATCH_H
