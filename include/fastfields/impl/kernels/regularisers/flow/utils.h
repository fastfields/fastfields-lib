#ifndef FF_REGULARISERS_FLOW_UTILS
#define FF_REGULARISERS_FLOW_UTILS
#include "../../cuda_switch.h"
#include "../../bounds.h"
#include "../../utils.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_flow)

// The flow kernel class. ONE definition, generic in D, in `flow/nd.h` -- it
// used to be three hand-expanded partial specialisations (`flow/{1,2,3}d.h`),
// selected by the `one`/`two`/`three` tags that lived here (fastfields-
// kernels#59). Declared rather than defined here so `utils.h` stays the
// vocabulary header it was.
template <int D, typename scalar_t, typename reduce_t, typename offset_t,
          bound::type... B>
struct RegFlow;

//----------------------------------------------------------------------
//          Helpers to implement generic variants that either
//          assign to, add or subtract from the output pointer.
//----------------------------------------------------------------------

template <typename T, typename IT>
inline CUDEV T & set(T & out, const IT & in)
{
    out = static_cast<T>(in);
    return out;
}

template <typename T, typename IT>
inline CUDEV T & iadd(T & out, const IT & in)
{
    out = static_cast<T>(static_cast<IT>(out) + in);
    return out;
}

template <typename T, typename IT>
inline CUDEV T & isub(T & out, const IT & in)
{
    out = static_cast<T>(static_cast<IT>(out) - in);
    return out;
}

template <typename T, typename IT>
inline CUDEV T add(const T & out, const IT & in)
{
    return static_cast<T>(static_cast<IT>(out) + in);
}

template <typename T, typename IT>
inline CUDEV T sub(const T & out, const IT & in)
{
    return static_cast<T>(static_cast<IT>(out) - in);
}

template <char op, typename scalar_t, typename reduce_t = scalar_t>
struct Op {
    typedef scalar_t & (*FuncType)(scalar_t &, const reduce_t &);
    static constexpr FuncType f = set;
};

template <typename scalar_t, typename reduce_t>
struct Op<'+', scalar_t, reduce_t> {
    typedef scalar_t & (*FuncType)(scalar_t &, const reduce_t &);
    static constexpr FuncType f = iadd;
};

template <typename scalar_t, typename reduce_t>
struct Op<'-', scalar_t, reduce_t> {
    typedef scalar_t & (*FuncType)(scalar_t &, const reduce_t &);
    static constexpr FuncType f = isub;
};

//----------------------------------------------------------------------
//                  Helpers to implement the loops
//----------------------------------------------------------------------

template <int N, typename U>
CUDEV inline
U center_offset(const U * size, const U * stride)
{
    U offset = 0;
#   pragma unroll
    for (int d=0; d < N; ++d)
        offset += (size[d]-1)/2 * stride[d];
    return offset;
}

template <int N, typename offset_t>
CUDEV inline
bool patch1(const offset_t loc[N], offset_t n)
{
    offset_t acc = 0;
#   pragma unroll
    for (int d=0; d < N; ++d)
        acc += loc[d];
    return acc % 2 == n % 2;
}

template <int N, typename offset_t>
CUDEV inline
bool patch2(const offset_t loc[N], offset_t n)
{
    offset_t acc = 0;
    offset_t mul = 1;
#   pragma unroll
    for (int d=0; d < N; ++d, mul *= 2)
        acc += (loc[d] % 2) * mul;
    return acc == n % mul;
}

template <int N, typename offset_t>
CUDEV inline
bool patch3(const offset_t loc[N], offset_t n)
{
    offset_t acc = 0;
    offset_t mul = 1;
#   pragma unroll
    for (int d=0; d < N; ++d, mul *= 3)
        acc += (loc[d] % 3) * mul;
    return acc == n % mul;
}

FF_NAMESPACE_END(reg_flow)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_REGULARISERS_FLOW_UTILS
