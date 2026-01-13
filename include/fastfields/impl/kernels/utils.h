#ifndef FF_UTILS
#define FF_UTILS
#include "cuda_switch.h"

#ifndef __CUDACC__
#   include <cmath>
#endif

FF_NAMESPACE_BEGIN(FF)
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
inline CUDEV
void swap(T& a, T& b)
{
    T c(a); a=b; b=c;
}

template <typename T>
inline CUDEV
T square(T a)
{
    return a*a;
}

#ifdef __CUDACC__

template <typename T>
inline CUDEV
T sqrt(T a)
{}

template <>
inline CUDEV
float sqrt(float a)
{
    return ::sqrtf(a);
}

template <>
inline CUDEV
double sqrt(double a)
{
    return ::sqrt(a);
}

template <>
inline CUDEV
half sqrt(half a)
{
    return ::hsqrt(a);
}

#else

template <typename T>
inline CUDEV
T sqrt(T a)
{
    return std::sqrt(a);
}

#endif


template <int N, typename T>
inline CUDEV
T pow(T a) {
    T p = a;
#   pragma unroll
    for(int d = 0; d < N-1; ++d)
        p *= a;
    return p;
}

template <typename T>
inline CUDEV
T pow(T a, int N) {
    T p = a;
#   pragma unroll
    for(int d = 0; d < N-1; ++d)
        p *= a;
    return p;
}

template <typename T>
inline CUDEV
T min(T a, T b)
{
    return (a < b ? a : b);
}

template <typename T>
inline CUDEV
T max(T a, T b)
{
    return (a > b ? a : b);
}

template <typename T>
inline CUDEV
T abs(T a)
{
    return static_cast<T>(a < 0 ? -a : a);
}

template <typename T>
inline CUDEV
signed char sign(T a)
{
    return static_cast<signed char>(a == 0 ? 0 : a < 0 ? -1 : 1);
}

#ifdef __CUDACC__
template <>
inline CUDEV
half min<>(half a, half b)
{
    float af = static_cast<float>(a);
    float bf = static_cast<float>(b);
    return (a < b ? a : b);
}
template <>
inline CUDEV
half max<>(half a, half b)
{
    float af = static_cast<float>(a);
    float bf = static_cast<float>(b);
    return (a > b ? a : b);
}
#endif

// fmod
template <typename T, typename U,
          bool is_float_T = is_floating_point<T>::value,
          bool is_float_U = is_floating_point<U>::value >
struct _mod
{
    inline CUDEV static
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
    inline CUDEV static
    T f(T x, U d)
    {
        return x % d;
    }
};

template <typename T, typename U>
inline CUDEV
T mod(T x, U d)
{
    return _mod<T,U>::f(x, d);
}

template <typename T, typename size_t>
inline CUDEV
T prod(const T * x, size_t size)
{
    if (size == 0)
        return static_cast<T>(1);
    T tmp = x[0];
    for (size_t d = 1; d < size; ++d)
        tmp *= x[d];
    return tmp;
}

template <unsigned long size, typename T>
inline CUDEV
T prod(const T * x)
{
    if (size == 0)
        return static_cast<T>(1);
    T tmp = x[0];
#   pragma unroll
    for (size_t d = 1; d < size; ++d)
        tmp *= x[d];
    return tmp;
}

template <int N, typename U, typename V>
inline CUDEV
void fillfrom(U out[N], const V * inp)
{
#   pragma unroll
    for (int n=0; n < N; ++ n)
        out[n] = static_cast<U>(inp[n]);
}

template <int N, typename U, typename V, typename W>
inline CUDEV
void fillfrom(U out[N], const V * inp, W stride)
{
#   pragma unroll
    for (int n=0; n < N; ++n, inp += stride)
        out[n] = static_cast<U>(*inp);
}

template <typename U, typename V>
inline CUDEV
void fillfrom(int N, U out[], const V * inp)
{
    for (int n=0; n < N; ++ n)
        out[n] = static_cast<U>(inp[n]);
}

template <typename U, typename V, typename W>
inline CUDEV
void fillfrom(int N, U out[], const V * inp, W stride)
{
    for (int n=0; n < N; ++n, inp += stride)
        out[n] = static_cast<U>(*inp);
}

template <int N, typename U, typename V>
inline CUDEV
void fill(U * out, V inp)
{
    auto val = static_cast<U>(inp);
#   pragma unroll
    for (int n=0; n < N; ++n)
        out[n] = val;
}

template <int N, typename U, typename V, typename W>
inline CUDEV
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
    CUHOSTDEV constexpr StaticValue(const T & value = Value) {}
    CUHOSTDEV constexpr StaticValue(const StaticValue<T,Value> & value) {}
    CUHOSTDEV constexpr operator T() const { return Value; }
};

template <typename T, T V>
CUHOSTDEV constexpr StaticValue<decltype(+V),+V>
operator+(const StaticValue<T,V> & v)
{
    return StaticValue<decltype(+V),+V>();
}

template <typename T, T V>
CUHOSTDEV constexpr StaticValue<decltype(-V),-V>
operator-(const StaticValue<T,V> & v)
{
    return StaticValue<decltype(-V),-V>();
}

template <typename T, T V>
CUHOSTDEV constexpr StaticValue<decltype(!V),!V>
operator!(const StaticValue<T,V> & v)
{
    return StaticValue<decltype(!V),!V>();
}

template <typename T, T V>
CUHOSTDEV constexpr StaticValue<decltype(~V),~V>
operator~(const StaticValue<T,V> & v)
{
    return StaticValue<decltype(~V),~V>();
}

// static vs static

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<decltype(V+W),V+W>
operator+(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V+W),V+W>();
}

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<decltype(V-W),V-W>
operator-(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V-W),V-W>();
}

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<decltype(V*W),V*W>
operator*(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V*W),V*W>();
}

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<decltype(V/W),V/W>
operator/(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V/W),V/W>();
}

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<decltype(V%W),V%W>
operator%(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V%W),V%W>();
}

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<decltype(V&W),V&W>
operator&(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V&W),V&W>();
}

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<decltype(V|W),V|W>
operator|(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V|W),V|W>();
}

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<decltype(V^W),V^W>
operator^(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V^W),V^W>();
}

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<decltype(V>>W),(V>>W)>
operator>>(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V>>W),(V>>W)>();
}

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<decltype(V<<W),(V<<W)>
operator<<(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<decltype(V<<W),(V<<W)>();
}

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<bool,V==W>
operator==(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<bool,V==W>();
}

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<bool,V!=W>
operator!=(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<bool,V!=W>();
}

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<bool,V<=W>
operator<=(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<bool,V<=W>();
}

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<bool,V>=W>
operator>=(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<bool,V>=W>();
}

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<bool,(V<W)>
operator<(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<bool,(V<W)>();
}

template <typename T, T V, typename U, U W>
CUHOSTDEV constexpr StaticValue<bool,(V>W)>
operator>(const StaticValue<T,V> & v, const StaticValue<U,W> & w)
{
    return StaticValue<bool,(V>W)>();
}

// static vs dynamic

template <typename T, T V, typename U>
CUHOSTDEV decltype(V+U())
operator+(const StaticValue<T,V> & v, const U & w)
{
    return v + w;
}

template <typename T, T V, typename U>
CUHOSTDEV decltype(V-U())
operator-(const StaticValue<T,V> & v, const U & w)
{
    return v - w;
}

template <typename T, T V, typename U>
CUHOSTDEV decltype(V*U())
operator*(const StaticValue<T,V> & v, const U & w)
{
    return v * w;
}

template <typename T, T V, typename U>
CUHOSTDEV decltype(V/U())
operator/(const StaticValue<T,V> & v, const U & w)
{
    return v / w;
}

template <typename T, T V, typename U>
CUHOSTDEV decltype(V%U())
operator%(const StaticValue<T,V> & v, const U & w)
{
    return v % w;
}

template <typename T, T V, typename U>
CUHOSTDEV decltype(V&U())
operator&(const StaticValue<T,V> & v, const U & w)
{
    return v & w;
}

template <typename T, T V, typename U>
CUHOSTDEV decltype(V|U())
operator|(const StaticValue<T,V> & v, const U & w)
{
    return v | w;
}

template <typename T, T V, typename U>
CUHOSTDEV decltype(V^U())
operator^(const StaticValue<T,V> & v, const U & w)
{
    return v ^ w;
}

template <typename T, T V, typename U>
CUHOSTDEV decltype(V>>U())
operator>>(const StaticValue<T,V> & v, const U & w)
{
    return v >> w;
}

template <typename T, T V, typename U>
CUHOSTDEV decltype(V<<U())
operator<<(const StaticValue<T,V> & v, const U & w)
{
    return v << w;
}

template <typename T, T V, typename U>
CUHOSTDEV bool
operator==(const StaticValue<T,V> & v, const U & w)
{
    return v == w;
}

template <typename T, T V, typename U>
CUHOSTDEV bool
operator!=(const StaticValue<T,V> & v, const U & w)
{
    return v != w;
}

template <typename T, T V, typename U>
CUHOSTDEV bool
operator>=(const StaticValue<T,V> & v, const U & w)
{
    return v >= w;
}

template <typename T, T V, typename U>
CUHOSTDEV bool
operator<=(const StaticValue<T,V> & v, const U & w)
{
    return v <= w;
}

template <typename T, T V, typename U>
CUHOSTDEV bool
operator>(const StaticValue<T,V> & v, const U & w)
{
    return v > w;
}

template <typename T, T V, typename U>
CUHOSTDEV bool
operator<(const StaticValue<T,V> & v, const U & w)
{
    return v < w;
}

// dynamic vs static

template <typename T, typename U, U W>
CUHOSTDEV decltype(T()+W)
operator+(const T & v, const StaticValue<U,W> & w)
{
    return v + w;
}

template <typename T, typename U, U W>
CUHOSTDEV decltype(T()-W)
operator-(const T & v, const StaticValue<U,W> & w)
{
    return v - w;
}

template <typename T, typename U, U W>
CUHOSTDEV decltype(T()/W)
operator/(const T & v, const StaticValue<U,W> & w)
{
    return v / w;
}

template <typename T, typename U, U W>
CUHOSTDEV decltype(T()*W)
operator*(const T & v, const StaticValue<U,W> & w)
{
    return v * w;
}

template <typename T, typename U, U W>
CUHOSTDEV decltype(T()%W)
operator%(const T & v, const StaticValue<U,W> & w)
{
    return v % w;
}

template <typename T, typename U, U W>
CUHOSTDEV decltype(T()&W)
operator&(const T & v, const StaticValue<U,W> & w)
{
    return v & w;
}

template <typename T, typename U, U W>
CUHOSTDEV decltype(T()|W)
operator|(const T & v, const StaticValue<U,W> & w)
{
    return v | w;
}

template <typename T, typename U, U W>
CUHOSTDEV decltype(T()^W)
operator^(const T & v, const StaticValue<U,W> & w)
{
    return v ^ w;
}

template <typename T, typename U, U W>
CUHOSTDEV decltype(T()>>W)
operator>>(const T & v, const StaticValue<U,W> & w)
{
    return v >> w;
}

template <typename T, typename U, U W>
CUHOSTDEV decltype(T()<<W)
operator<<(const T & v, const StaticValue<U,W> & w)
{
    return v << w;
}

template <typename T, typename U, U W>
CUHOSTDEV bool
operator==(const T & v, const StaticValue<U,W> & w)
{
    return v == w;
}

template <typename T, typename U, U W>
CUHOSTDEV bool
operator!=(const T & v, const StaticValue<U,W> & w)
{
    return v != w;
}

template <typename T, typename U, U W>
CUHOSTDEV bool
operator>=(const T & v, const StaticValue<U,W> & w)
{
    return v >= w;
}

template <typename T, typename U, U W>
CUHOSTDEV bool
operator<=(const T & v, const StaticValue<U,W> & w)
{
    return v <= w;
}

template <typename T, typename U, U W>
CUHOSTDEV bool
operator>(const T & v, const StaticValue<U,W> & w)
{
    return v > w;
}

template <typename T, typename U, U W>
CUHOSTDEV bool
operator<(const T & v, const StaticValue<U,W> & w)
{
    return v < w;
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_UTILS
