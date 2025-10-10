#ifndef FF_META
#define FF_META
#include "defines.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(meta)

template <class... T>          struct Pack           { template <class... V> using Append = Pack<T..., V...>; };
template <int N, class T>      struct _NPack         { using Type = typename Pack<T>::Append<typename _NPack<N-1, T>::Type>; };
template <class T>             struct _NPack<1,T>    { using Type = Pack<T>; };
template <class T>             struct _NPack<0,T>    { using Type = Pack<>;  };
template <int N, class T>      using NPack           = typename _NPack<N,T>::Type;
template <class T, T... X>     struct Tuple          { template <T... Y> using Append = Tuple<T, X..., Y...>; };
template <int N, class T, T X> struct _NTuple        { using Type = typename Tuple<T,X>::Append<typename _NTuple<N-1, T, X>::Type>; };
template <class T, T X>        struct _NTuple<1,T,X> { using Type = Tuple<T,X>; };
template <class T, T X>        struct _NTuple<0,T,X> { using Type = Tuple<T>;  };
template <int N, class T, T X> using NTuple          = typename _NTuple<N,T,X>::Type;

template <int...      D>       using Int             = Tuple<int,      D...>;

FF_NAMESPACE_END(meta)
FF_NAMESPACE_END(FF)


#endif // FF_META
