#ifndef FF_UTILS
#define FF_UTILS
#include <limits>
#include "fastfields/core/cuda_switch.h"

#ifndef __CUDACC__
#   include <cmath>
#endif // __CUDACC__

#ifndef __CUDA_ARCH__
#   include <cstdint>   // std::int32_t
#   include <cstddef>   // std::ptrdiff_t
#   include <limits>    // std::numeric_limits
#endif // __CUDA_ARCH__

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)

// static check for floating types
template <typename T>
struct is_floating_point { static constexpr bool value = false; };
template <>
struct is_floating_point<float> { static constexpr bool value = true; };
template <>
struct is_floating_point<double> { static constexpr bool value = true; };
#ifdef __CUDACC__
template <>
struct is_floating_point<half> { static constexpr bool value = true; };
#endif


template <typename T>
inline FF_CUDEV
void swap(T& a, T& b)
{
    T c(a); a=b; b=c;
}

template <typename T>
inline FF_CUDEV
T square(T a)
{
    return a*a;
}

#ifdef __CUDACC__

template <typename T>
inline FF_CUDEV
T sqrt(T a)
{}

template <>
inline FF_CUDEV
float sqrt(float a)
{
    return ::sqrtf(a);
}

template <>
inline FF_CUDEV
double sqrt(double a)
{
    return ::sqrt(a);
}

template <>
inline FF_CUDEV
half sqrt(half a)
{
    // hsqrt is not visible at global scope in every CUDA/arch combination;
    // compute in float and narrow back (half math promotes to float anyway).
    return static_cast<half>(::sqrtf(static_cast<float>(a)));
}

#else

template <typename T>
inline FF_CUDEV
T sqrt(T a)
{
    return std::sqrt(a);
}

#endif


template <int N, typename T>
inline FF_CUDEV
T pow(T a) {
    T p = a;
#   pragma unroll
    for(int d = 0; d < N-1; ++d)
        p *= a;
    return p;
}

template <typename T>
inline FF_CUDEV
T pow(T a, int N) {
    T p = a;
#   pragma unroll
    for(int d = 0; d < N-1; ++d)
        p *= a;
    return p;
}

template <typename T>
inline FF_CUDEV
T min(T a, T b)
{
    return (a < b ? a : b);
}

template <typename T>
inline FF_CUDEV
T max(T a, T b)
{
    return (a > b ? a : b);
}

template <typename T>
inline FF_CUDEV
T abs(T a)
{
    return static_cast<T>(a < 0 ? -a : a);
}

template <typename T>
inline FF_CUDEV
signed char sign(T a)
{
    return static_cast<signed char>(a == 0 ? 0 : a < 0 ? -1 : 1);
}

#ifdef __CUDACC__
template <>
inline FF_CUDEV
half min<>(half a, half b)
{
    // Compare via float: half has multiple implicit conversions to built-in
    // types, so `a < b` is ambiguous; comparing the float forms is not.
    float af = static_cast<float>(a);
    float bf = static_cast<float>(b);
    return (af < bf ? a : b);
}
template <>
inline FF_CUDEV
half max<>(half a, half b)
{
    float af = static_cast<float>(a);
    float bf = static_cast<float>(b);
    return (af > bf ? a : b);
}
#endif

// fmod
template <typename T, typename U,
          bool is_float_T = is_floating_point<T>::value,
          bool is_float_U = is_floating_point<U>::value >
struct _mod
{
    inline FF_CUDEV static
    T f(T x, U d)
    {
        signed char sx = sign(x);
        signed char sd = sign(d);

        long ratio = (sx*sd)*static_cast<long>(trunc(abs(x)/abs(d)));
        return (x - ratio * d);
    }
};


template <typename T, typename U>
struct _mod<T, U, false, false>
{
    inline FF_CUDEV static
    T f(T x, U d)
    {
        return x % d;
    }
};

template <typename T, typename U>
inline FF_CUDEV
T mod(T x, U d)
{
    return _mod<T,U>::f(x, d);
}

template <typename OT, typename IT, typename size_t>
inline FF_CUDEV
OT typed_prod(const IT * x, size_t size)
{
    if (size == 0)
        return static_cast<OT>(1);
    OT tmp = static_cast<OT>(x[0]);
    for (size_t d = 1; d < size; ++d)
        tmp *= static_cast<OT>(x[d]);
    return tmp;
}

template <typename OT, unsigned long size, typename IT>
inline FF_CUDEV
OT typed_prod(const IT * x)
{
    if (size == 0)
        return static_cast<OT>(1);
    OT tmp = static_cast<OT>(x[0]);
#   pragma unroll
    for (unsigned long d = 1; d < size; ++d)
        tmp *= static_cast<OT>(x[d]);
    return tmp;
}

template <typename T, typename size_t>
inline FF_CUDEV
T prod(const T * x, size_t size)
{
    return typed_prod<T>(x, size);
}

template <unsigned long size, typename T>
inline FF_CUDEV
T prod(const T * x)
{
    return typed_prod<T, size>(x);
}

template <int N, typename U, typename V>
inline FF_CUDEV
void fillfrom(U out[N], const V * inp)
{
#   pragma unroll
    for (int n=0; n < N; ++ n)
        out[n] = static_cast<U>(inp[n]);
}

template <int N, typename U, typename V, typename W>
inline FF_CUDEV
void fillfrom(U out[N], const V * inp, W stride)
{
#   pragma unroll
    for (int n=0; n < N; ++n, inp += stride)
        out[n] = static_cast<U>(*inp);
}

template <typename U, typename V>
inline FF_CUDEV
void fillfrom(int N, U out[], const V * inp)
{
    for (int n=0; n < N; ++ n)
        out[n] = static_cast<U>(inp[n]);
}

template <typename U, typename V, typename W>
inline FF_CUDEV
void fillfrom(int N, U out[], const V * inp, W stride)
{
    for (int n=0; n < N; ++n, inp += stride)
        out[n] = static_cast<U>(*inp);
}

template <int N, typename U, typename V>
inline FF_CUDEV
void fill(U * out, V inp)
{
    auto val = static_cast<U>(inp);
#   pragma unroll
    for (int n=0; n < N; ++n)
        out[n] = val;
}

template <int N, typename U, typename V, typename W>
inline FF_CUDEV
void fill(U * out, V inp, W stride)
{
    auto val = static_cast<U>(inp);
#   pragma unroll
    for (int n=0; n < N; ++n, out += stride)
        (*out) = val;
}

// --- static value ---

template <typename T, T Value>
struct StaticValue
{
    FF_CUHOSTDEV constexpr StaticValue(const T & value = Value) {}
    FF_CUHOSTDEV constexpr StaticValue(const StaticValue<T,Value> & value) {}
    FF_CUHOSTDEV constexpr operator T() const { return Value; }
};

template <typename T, T V>
FF_CUHOSTDEV constexpr StaticValue<decltype(+V),+V>
operator+(const StaticValue<T,V> & v)
{
    return StaticValue<decltype(+V),+V>();
}

template <typename T, T V>
FF_CUHOSTDEV constexpr StaticValue<decltype(-V),-V>
operator-(const StaticValue<T,V> & v)
{
    return StaticValue<decltype(-V),-V>();
}

template <typename T, T V>
FF_CUHOSTDEV constexpr StaticValue<decltype(!V),!V>
operator!(const StaticValue<T,V> & v)
{
    return StaticValue<decltype(!V),!V>();
}

template <typename T, T V>
FF_CUHOSTDEV constexpr StaticValue<decltype(~V),~V>
operator~(const StaticValue<T,V> & v)
{
    return StaticValue<decltype(~V),~V>();
}

// static vs static

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<decltype(V+W),V+W>
operator+(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V+W),V+W>();
}

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<decltype(V-W),V-W>
operator-(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V-W),V-W>();
}

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<decltype(V*W),V*W>
operator*(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V*W),V*W>();
}

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<decltype(V/W),V/W>
operator/(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V/W),V/W>();
}

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<decltype(V%W),V%W>
operator%(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V%W),V%W>();
}

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<decltype(V&W),V&W>
operator&(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V&W),V&W>();
}

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<decltype(V|W),V|W>
operator|(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V|W),V|W>();
}

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<decltype(V^W),V^W>
operator^(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V^W),V^W>();
}

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<decltype(V>>W),(V>>W)>
operator>>(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V>>W),(V>>W)>();
}

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<decltype(V<<W),(V<<W)>
operator<<(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V<<W),(V<<W)>();
}

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<bool,V==W>
operator==(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<bool,V==W>();
}

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<bool,V!=W>
operator!=(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<bool,V!=W>();
}

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<bool,V<=W>
operator<=(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<bool,V<=W>();
}

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<bool,V>=W>
operator>=(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<bool,V>=W>();
}

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<bool,(V<W)>
operator<(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<bool,(V<W)>();
}

template <typename T, T V, typename U, U W>
FF_CUHOSTDEV constexpr StaticValue<bool,(V>W)>
operator>(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<bool,(V>W)>();
}

// static vs dynamic

template <typename T, T V, typename U>
FF_CUHOSTDEV inline decltype(V+U())
operator+(const StaticValue<T,V> & v, const U & w)
{
    return v + w;
}

template <typename T, T V, typename U>
FF_CUHOSTDEV inline decltype(V-U())
operator-(const StaticValue<T,V> & v, const U & w)
{
    return v - w;
}

template <typename T, T V, typename U>
FF_CUHOSTDEV inline decltype(V*U())
operator*(const StaticValue<T,V> & v, const U & w)
{
    return v * w;
}

template <typename T, T V, typename U>
FF_CUHOSTDEV inline decltype(V/U())
operator/(const StaticValue<T,V> & v, const U & w)
{
    return v / w;
}

template <typename T, T V, typename U>
FF_CUHOSTDEV inline decltype(V%U())
operator%(const StaticValue<T,V> & v, const U & w)
{
    return v % w;
}

template <typename T, T V, typename U>
FF_CUHOSTDEV inline decltype(V&U())
operator&(const StaticValue<T,V> & v, const U & w)
{
    return v & w;
}

template <typename T, T V, typename U>
FF_CUHOSTDEV inline decltype(V|U())
operator|(const StaticValue<T,V> & v, const U & w)
{
    return v | w;
}

template <typename T, T V, typename U>
FF_CUHOSTDEV inline decltype(V^U())
operator^(const StaticValue<T,V> & v, const U & w)
{
    return v ^ w;
}

template <typename T, T V, typename U>
FF_CUHOSTDEV inline decltype(V>>U())
operator>>(const StaticValue<T,V> & v, const U & w)
{
    return v >> w;
}

template <typename T, T V, typename U>
FF_CUHOSTDEV inline decltype(V<<U())
operator<<(const StaticValue<T,V> & v, const U & w)
{
    return v << w;
}

template <typename T, T V, typename U>
FF_CUHOSTDEV inline bool
operator==(const StaticValue<T,V> & v, const U & w)
{
    return v == w;
}

template <typename T, T V, typename U>
FF_CUHOSTDEV inline bool
operator!=(const StaticValue<T,V> & v, const U & w)
{
    return v != w;
}

template <typename T, T V, typename U>
FF_CUHOSTDEV inline bool
operator>=(const StaticValue<T,V> & v, const U & w)
{
    return v >= w;
}

template <typename T, T V, typename U>
FF_CUHOSTDEV inline bool
operator<=(const StaticValue<T,V> & v, const U & w)
{
    return v <= w;
}

template <typename T, T V, typename U>
FF_CUHOSTDEV inline bool
operator>(const StaticValue<T,V> & v, const U & w)
{
    return v > w;
}

template <typename T, T V, typename U>
FF_CUHOSTDEV inline bool
operator<(const StaticValue<T,V> & v, const U & w)
{
    return v < w;
}

// dynamic vs static

template <typename T, typename U, U W>
FF_CUHOSTDEV inline decltype(T()+W)
operator+(const T & v, const StaticValue<U,W> & w)
{
    return v + w;
}

template <typename T, typename U, U W>
FF_CUHOSTDEV inline decltype(T()-W)
operator-(const T & v, const StaticValue<U,W> & w)
{
    return v - w;
}

template <typename T, typename U, U W>
FF_CUHOSTDEV inline decltype(T()/W)
operator/(const T & v, const StaticValue<U,W> & w)
{
    return v / w;
}

template <typename T, typename U, U W>
FF_CUHOSTDEV inline decltype(T()*W)
operator*(const T & v, const StaticValue<U,W> & w)
{
    return v * w;
}

template <typename T, typename U, U W>
FF_CUHOSTDEV inline decltype(T()%W)
operator%(const T & v, const StaticValue<U,W> & w)
{
    return v % w;
}

template <typename T, typename U, U W>
FF_CUHOSTDEV inline decltype(T()&W)
operator&(const T & v, const StaticValue<U,W> & w)
{
    return v & w;
}

template <typename T, typename U, U W>
FF_CUHOSTDEV inline decltype(T()|W)
operator|(const T & v, const StaticValue<U,W> & w)
{
    return v | w;
}

template <typename T, typename U, U W>
FF_CUHOSTDEV inline decltype(T()^W)
operator^(const T & v, const StaticValue<U,W> & w)
{
    return v ^ w;
}

template <typename T, typename U, U W>
FF_CUHOSTDEV inline decltype(T()>>W)
operator>>(const T & v, const StaticValue<U,W> & w)
{
    return v >> w;
}

template <typename T, typename U, U W>
FF_CUHOSTDEV inline decltype(T()<<W)
operator<<(const T & v, const StaticValue<U,W> & w)
{
    return v << w;
}

template <typename T, typename U, U W>
FF_CUHOSTDEV inline bool
operator==(const T & v, const StaticValue<U,W> & w)
{
    return v == w;
}

template <typename T, typename U, U W>
FF_CUHOSTDEV inline bool
operator!=(const T & v, const StaticValue<U,W> & w)
{
    return v != w;
}

template <typename T, typename U, U W>
FF_CUHOSTDEV inline bool
operator>=(const T & v, const StaticValue<U,W> & w)
{
    return v >= w;
}

template <typename T, typename U, U W>
FF_CUHOSTDEV inline bool
operator<=(const T & v, const StaticValue<U,W> & w)
{
    return v <= w;
}

template <typename T, typename U, U W>
FF_CUHOSTDEV inline bool
operator>(const T & v, const StaticValue<U,W> & w)
{
    return v > w;
}

template <typename T, typename U, U W>
FF_CUHOSTDEV inline bool
operator<(const T & v, const StaticValue<U,W> & w)
{
    return v < w;
}

// - 32 bit index math check

template <class ndim_t, class size_t, class stride_t>
FF_CUHOST inline bool canUse32BitIndexMath(
          ndim_t     ndim,
    const size_t   * size,
    const stride_t * stride
)
{
    int64_t max32 = std::numeric_limits<int32_t>::max();

    int64_t numel = typed_prod<int64_t>(size, ndim);
    if (numel >= max32) return false;
    if (numel == 0)     return max32 > 0;

    // A null stride array is DLPack's compact row-major tensor: the largest
    // offset is numel-1, already known < max32 here, so 32-bit math is safe.
    if (stride == nullptr) return true;

    int64_t offset    = 0;
    int64_t lin_index = numel - 1;

    // NOTE: Assumes all strides are positive, which is true for now
    for (ndim_t i = ndim - 1; i >= 0; --i) {
        int64_t cur_index  = lin_index % size[i];
        int64_t cur_offset = cur_index * stride[i];
        offset += cur_offset;
        lin_index /= size[i];
    }

    return offset < max32;
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)

#endif // FF_UTILS
