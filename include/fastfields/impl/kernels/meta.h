#ifndef FF_META
#define FF_META
#include "defines.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(meta)

template <class... T>                 struct Pack;
template <int I, class P>             struct _PackAt {};
template <int I, class T, class... U> struct _PackAt<I, Pack<T, U...>> { using Value = _PackAt<I-1, Pack<U...>>; };
template <class T, class... U>        struct _PackAt<0, Pack<T, U...>> { using Value = T; };

template <class... T>
struct Pack
{
    template <class... V> using Append = Pack<T..., V...>;
    template <int I=0>    using At     = typename _PackAt<I, Pack<T...>>::Value;
};

template <class T, T... X>             struct Tuple;
template <int I, class T>              struct _TupleAt {};
template <int I, class T, T X, T... Y> struct _TupleAt<I, Tuple<T, X, Y...>> { static constexpr T Value = _TupleAt<I-1, Tuple<T, Y...>>::Value; };
template <class T, T X, T... Y>        struct _TupleAt<0, Tuple<T, X, Y...>> { static constexpr T Value = X; };

template <class T, T... X>
struct Tuple
{
    template <T... Y>  using Append = Tuple<T, X..., Y...>;
    template <int I=0> using At     = _TupleAt<I, Tuple<T, X...>>;
};

template <int N, class T>      struct _NPack         { using Type = typename Pack<T>::template Append<typename _NPack<N-1, T>::Type>; };
template <class T>             struct _NPack<1,T>    { using Type = Pack<T>; };
template <class T>             struct _NPack<0,T>    { using Type = Pack<>;  };
template <int N, class T>      using NPack           = typename _NPack<N,T>::Type;
template <int N, class T, T X> struct _NTuple        { using Type = typename Tuple<T,X>::template Append<typename _NTuple<N-1, T, X>::Type>; };
template <class T, T X>        struct _NTuple<1,T,X> { using Type = Tuple<T,X>; };
template <class T, T X>        struct _NTuple<0,T,X> { using Type = Tuple<T>;  };
template <int N, class T, T X> using NTuple          = typename _NTuple<N,T,X>::Type;

template <int...      D>       using Int             = Tuple<int,      D...>;

// compile-time type selection (hand-rolled rather than <type_traits>, like the
// `is_floating_point` in utils.h, so the headers stay NVRTC-friendly)
template <bool C, class A, class B> struct _If            { using Type = A; };
template <class A, class B>         struct _If<false,A,B> { using Type = B; };
template <bool C, class A, class B> using  If             = typename _If<C,A,B>::Type;

FF_NAMESPACE_END(meta)
FF_NAMESPACE_END(FF)


#endif // FF_META
