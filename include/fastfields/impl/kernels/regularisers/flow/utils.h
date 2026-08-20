#ifndef FF_REGULARISERS_FLOW_UTILS
#define FF_REGULARISERS_FLOW_UTILS
#include <fastfields/core/cuda_switch.h>
#include <fastfields/core/bounds.h>
#include <fastfields/core/utils.h>

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_flow)

const bound::type B0 = bound::type::NoCheck;
const int zero  = 0;
const int one   = 1;
const int two   = 2;
const int three = 3;

template <int, typename, typename, typename, bound::type...>
struct RegFlow {};

//----------------------------------------------------------------------
//          Helpers to implement generic variants that either
//          assign to, add or subtract from the output pointer.
//----------------------------------------------------------------------

template <typename T, typename IT>
inline FF_CUDEV T & set(T & out, const IT & in)
{
    out = static_cast<T>(in);
    return out;
}

template <typename T, typename IT>
inline FF_CUDEV T & iadd(T & out, const IT & in)
{
    out = static_cast<T>(static_cast<IT>(out) + in);
    return out;
}

template <typename T, typename IT>
inline FF_CUDEV T & isub(T & out, const IT & in)
{
    out = static_cast<T>(static_cast<IT>(out) - in);
    return out;
}

template <typename T, typename IT>
inline FF_CUDEV T add(const T & out, const IT & in)
{
    return static_cast<T>(static_cast<IT>(out) + in);
}

template <typename T, typename IT>
inline FF_CUDEV T sub(const T & out, const IT & in)
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
FF_CUDEV inline
U center_offset(const U * size, const U * stride)
{
    U offset = 0;
#   pragma unroll
    for (int d=0; d < N; ++d)
        offset += (size[d]-1)/2 * stride[d];
    return offset;
}

template <int N, typename offset_t>
FF_CUDEV inline
bool patch1(const offset_t loc[N], offset_t n)
{
    offset_t acc = 0;
#   pragma unroll
    for (int d=0; d < N; ++d)
        acc += loc[d];
    return acc % 2 == n % 2;
}

template <int N, typename offset_t>
FF_CUDEV inline
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
FF_CUDEV inline
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
FF_NAMESPACE_END(FF_NS)

#endif // FF_REGULARISERS_FLOW_UTILS
