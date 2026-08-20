#ifndef FF_POSDEF_UTILS
#define FF_POSDEF_UTILS
#include <fastfields/core/cuda_switch.h>
#include "../utils.h"

#define FF_ONE_PLUS_TINY 1.000001
#define FF_UNUSED __attribute__((unused))

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                              INTERNALS
//
// ---------------------------------------------------------------------
//
// We first define a bunch of internal utilities that allow:
// 1) to work with strided pointers. Strided pointers point to memory
//    in which elements of interest are not separated by `sizeof(T)`
//    but by `S * sizeof(T)`. This classes implement operators that are
//    classically used on pointers (dereference, access, ++, +=, --, -=).
// 2) we define traits that work on both classical and strided pointers:
//      elem_type<T>::value -> Type of referenced elements
//      is_const<T>::value -> Whether referenced elements are const
//      return_type<T...>::value -> Upcast of types T...
// 3) we define inplace operators that convert all values to a "reduction"
//    type to carry the computation before downcasting to the output type.
//    E.g.: iadd<reduce_t>, isub<reduce_t>, iaddcmul<reduce_t>, ...
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(posdef)
FF_NAMESPACE_BEGIN(internal)

//----------------------------------------------------------------------
//
//                          Strided pointers
//
//    We define a class Pointer<T, S, L>  in which the stride
//    length is statically known, and a specialized version Pointer<T, 0>
//    for strides that are only known dynamically.
//
//----------------------------------------------------------------------

template <typename ST, long S=1, typename OT=long>
struct Pointer {
    using this_type = Pointer<ST, S, OT>;
    using scalar_t = ST;
    using offset_t = OT;

    scalar_t * data;
    static constexpr offset_t stride = static_cast<offset_t>(S);

    // FF_CUHOSTDEV so device kernels can construct these (empty on non-nvcc).
    FF_CUHOSTDEV Pointer(scalar_t * ptr): data(ptr) {}
    FF_CUHOSTDEV Pointer(const this_type & ptr): data(ptr.data) {}

    inline FF_CUDEV scalar_t& operator[] (offset_t i) const { return data[i*stride]; }
    // inline FF_CUDEV const scalar_t& operator[] (offset_t i) const { return data[i*stride]; }
    inline FF_CUDEV scalar_t& operator* () const { return *data; }
    inline FF_CUDEV operator bool () const { return data != nullptr; }

    inline FF_CUDEV this_type & operator++ () { data += stride; return *this; }
    inline FF_CUDEV this_type operator++ (int) { this_type prev = *this; data += stride; return prev; }
    inline FF_CUDEV this_type & operator-- () { data -= stride; return *this; }
    inline FF_CUDEV this_type operator-- (int) { this_type prev = *this; data -= stride; return prev; }
    inline FF_CUDEV this_type & operator += (offset_t N) { data += N * stride; return *this; }
    inline FF_CUDEV this_type & operator -= (offset_t N) { data -= N * stride; return *this; }
};

template <typename ST, typename OT>
struct Pointer<ST, 0, OT> {
    using this_type = Pointer<ST, 0, OT>;
    using scalar_t = ST;
    using offset_t = OT;

    scalar_t * data;
    offset_t stride;

    // FF_CUHOSTDEV so device kernels can construct these (empty on non-nvcc).
    FF_CUHOSTDEV Pointer(scalar_t * ptr): data(ptr), stride(1) {}
    FF_CUHOSTDEV Pointer(scalar_t * ptr, offset_t str): data(ptr), stride(str) {}

    template <typename inp_offset_t, long S>
    FF_CUHOSTDEV Pointer(const Pointer<scalar_t, S, inp_offset_t> & ptr):
        data(ptr.data), stride(static_cast<offset_t>(ptr.stride)) {}

    inline FF_CUDEV scalar_t& operator[] (offset_t i) const { return data[i*stride]; }
    // inline FF_CUDEV const scalar_t& operator[] (offset_t i) const { return data[i*stride]; }
    inline FF_CUDEV scalar_t& operator* () const { return *data; }
    inline FF_CUDEV operator bool () const { return data != nullptr; }

    inline FF_CUDEV this_type & operator++ () { data += stride; return *this; }
    inline FF_CUDEV this_type operator++ (int) { this_type prev = *this; data += stride; return prev; }
    inline FF_CUDEV this_type & operator-- () { data -= stride; return *this; }
    inline FF_CUDEV this_type operator-- (int) { this_type prev = *this; data -= stride; return prev; }
    inline FF_CUDEV this_type & operator += (offset_t N) { data += N * stride; return *this; }
    inline FF_CUDEV this_type & operator -= (offset_t N) { data -= N * stride; return *this; }
};

template <typename ST, typename OT>
using StridedPointer = Pointer<ST, 0, OT>;

#if 0
template <typename scalar_t, long S, typename offset_t>
std::ostream& operator<< (std::ostream& os, const Pointer<scalar_t, S, offset_t> & ptr)
{
    os << "Pointer[" << ptr.data << " (" << ptr.stride << ")]";
    return os;
}
#endif

template <typename scalar_t, long S, typename offset_t>
inline FF_CUDEV
Pointer<scalar_t, S, offset_t> operator+ (Pointer<scalar_t, S, offset_t> prev, offset_t N)
{
    Pointer<scalar_t, S, offset_t> next = prev;
    next += N;
    return next;
}

template <typename scalar_t, long S, typename offset_t>
inline FF_CUDEV
Pointer<scalar_t, S, offset_t> operator- (Pointer<scalar_t, S, offset_t> prev, offset_t N)
{
    Pointer<scalar_t, S, offset_t> next = prev;
    next -= N;
    return next;
}

template <typename scalar_t, long S, typename offset_t>
inline FF_CUDEV
Pointer<scalar_t, S, offset_t> pointer(Pointer<scalar_t, S, offset_t> ptr)
{
    return ptr;
}

template <typename scalar_t, typename offset_t>
inline FF_CUDEV
Pointer<scalar_t, 0, offset_t> pointer(scalar_t * ptr, offset_t stride)
{
    return Pointer<scalar_t, 0, offset_t>(ptr, stride);
}

template <typename scalar_t>
inline FF_CUDEV
Pointer<scalar_t, 0, long> pointer(scalar_t * ptr)
{
    return Pointer<scalar_t, 0, long>(ptr);
}


//----------------------------------------------------------------------
//
//                              Traits
//
//----------------------------------------------------------------------

// ----------------
// traits: is_pointer, as_pointer
// Convert classic pointer types into our Pointer type
// ----------------

template <typename T>
struct _as_pointer {};

template <typename T>
struct is_pointer { static constexpr bool value = false; };

template <typename scalar_t, long S, typename offset_t>
struct _as_pointer<Pointer<scalar_t, S, offset_t> > {
    using value = Pointer<scalar_t, S, offset_t>;
};

template <typename scalar_t, long S, typename offset_t>
struct is_pointer<Pointer<scalar_t, S, offset_t> > {
    static constexpr bool value = true;
};

template <typename scalar_t>
struct _as_pointer<scalar_t *> {
    using value = Pointer<scalar_t>;
};

template <typename scalar_t>
struct is_pointer<scalar_t *> {
    static constexpr bool value = true;
};

template <typename T>
using as_pointer = typename _as_pointer<T>::value;


// ----------------
// traits: deconst
// Remove constness from scalar type
// ----------------

template <typename T>
struct _deconst {
    using value = T;
};

template <typename T>
struct _deconst<const T> {
    using value = T;
};


template <typename T>
using deconst = typename _deconst<T>::value;

// ----------------
// traits: elem_type
// Return the element-type (without constness) referenced by a pointer
// ----------------

template <typename T, bool is_pointer_type = is_pointer<T>::value>
struct _elem_type {
    using value = deconst<typename as_pointer<T>::scalar_t>;
};

template <typename T>
struct _elem_type<T, false> {
    using value = deconst<T>;
};

template <typename T>
using elem_type = typename _elem_type<T>::value;


// ----------------
// traits: upcast
// Return the output type of a binary (or +) operation on two (or +) types
// ----------------

// -- Helper for dealing with a single type

template <typename left_t>
struct _return_type1 {
    using value = elem_type<left_t>;
};

template <typename left_t>
using return_type1 = typename _return_type1<left_t>::value;

// -- Helper for dealing with a pair of types

template <typename left_t, typename right_t>
struct _return_type2;

template <typename left_t, typename right_t>
using return_type2 = typename _return_type2<left_t, right_t>::value;


template <typename left_t, typename right_t>
struct _return_type2 {
    using value = _return_type2<return_type1<left_t>, return_type1<right_t>>;
};

template <typename same_t>
struct _return_type2<same_t, same_t> {
    using value = return_type1<same_t>;
};

// void gets skipped
template <typename scalar_t>
struct _return_type2<scalar_t, void> {
    using value = return_type1<scalar_t>;
};
template <typename scalar_t>
struct _return_type2<void, scalar_t> {
    using value = return_type1<scalar_t>;
};


// <float, double> -> double
template <>
struct _return_type2<float, double> {
    using value = double;
};
template <>
struct _return_type2<double, float> {
    using value = double;
};

#ifdef __CUDACC__
    // <half, double> -> double
    template <>
    struct _return_type2<double, half> {
        using value = double;
    };
    template <>
    struct _return_type2<half, double> {
        using value = double;
    };

    // <half, float> -> float
    template <>
    struct _return_type2<float, half> {
        using value = float;
    };
    template <>
    struct _return_type2<half, float> {
        using value = float;
    };

#endif // __CUDACC__


// -- Generic declaration

// Should never be called unless there is only one type
// Then, we fallback to the first type.
template <typename left_t, typename... scalar_t>
struct _return_type;

template <typename left_t, typename... scalar_t>
using return_type = typename _return_type<left_t, scalar_t...>::value;

template <typename left_t, typename... scalar_t>
struct _return_type {
    using value = return_type1<left_t>;
};

// 2 types -> defer to return_type2
template <typename left_t, typename right_t>
struct _return_type<left_t, right_t> {
    using value = return_type2<left_t, right_t>;
};

// 3+ types -> collapse first two types and recurse
template <typename left_t, typename right_t, typename next_t, typename... other_t>
struct _return_type<left_t, right_t, next_t, other_t...> {
    using _lr   = return_type2<left_t, right_t>;
    using value = return_type<_lr, next_t, other_t...>;
};

//----------------------------------------------------------------------
//
//        In-place operators with upcast to reduction type
//
//----------------------------------------------------------------------

// left = right
template <typename left_t, typename right_t>
inline FF_CUDEV void set(left_t & left, const right_t & right)
{
    left = static_cast<left_t>(right);
}

// left += right
template <typename reduce_t, typename left_t, typename right_t>
inline FF_CUDEV void iadd(left_t & left, const right_t & right)
{
    left = static_cast<left_t>(static_cast<reduce_t>(left) +
                               static_cast<reduce_t>(right));
}

// left -= right
template <typename reduce_t, typename left_t, typename right_t>
inline FF_CUDEV void isub(left_t & left, const right_t & right)
{
    left = static_cast<left_t>(static_cast<reduce_t>(left) -
                               static_cast<reduce_t>(right));
}

// left *= right
template <typename reduce_t, typename left_t, typename right_t>
inline FF_CUDEV void imul(left_t & left, const right_t & right)
{
    left = static_cast<left_t>(static_cast<reduce_t>(left) *
                               static_cast<reduce_t>(right));
}

// left /= right
template <typename reduce_t, typename left_t, typename right_t>
inline FF_CUDEV void idiv(left_t & left, const right_t & right)
{
    left = static_cast<left_t>(static_cast<reduce_t>(left) /
                               static_cast<reduce_t>(right));
}

// out += left * right
template <typename reduce_t, typename out_t, typename left_t, typename right_t>
inline FF_CUDEV void iaddcmul(out_t & out, const left_t & left, const right_t & right)
{
    out = static_cast<out_t>(static_cast<reduce_t>(out) +
                               static_cast<reduce_t>(left) *
                               static_cast<reduce_t>(right));
}

// out -= left * right
template <typename reduce_t, typename out_t, typename left_t, typename right_t>
inline FF_CUDEV void isubcmul(out_t & out, const left_t & left, const right_t & right)
{
    out = static_cast<out_t>(static_cast<reduce_t>(out) -
                               static_cast<reduce_t>(left) *
                               static_cast<reduce_t>(right));
}

// out /= left + right
template <typename reduce_t, typename out_t, typename left_t, typename right_t>
inline FF_CUDEV void idivcadd(out_t & out, const left_t & left, const right_t & right)
{
    out = static_cast<out_t>(static_cast<reduce_t>(out) /
                               (static_cast<reduce_t>(left) +
                                static_cast<reduce_t>(right)));
}

// out = left + right
template <typename reduce_t, typename out_t, typename left_t, typename right_t>
inline FF_CUDEV void add(out_t & out, const left_t & left, const right_t & right)
{
    out = static_cast<out_t>(static_cast<reduce_t>(left) +
                             static_cast<reduce_t>(right));
}

// out = left - right
template <typename reduce_t, typename out_t, typename left_t, typename right_t>
inline FF_CUDEV void sub(out_t & out, const left_t & left, const right_t & right)
{
    out = static_cast<out_t>(static_cast<reduce_t>(left) -
                             static_cast<reduce_t>(right));
}

// out = left * right
template <typename reduce_t, typename out_t, typename left_t, typename right_t>
inline FF_CUDEV void mul(out_t & out, const left_t & left, const right_t & right)
{
    out = static_cast<out_t>(static_cast<reduce_t>(left) *
                             static_cast<reduce_t>(right));
}

// out = left / right
template <typename reduce_t, typename out_t, typename left_t, typename right_t>
inline FF_CUDEV void div(out_t & out, const left_t & left, const right_t & right)
{
    out = static_cast<out_t>(static_cast<reduce_t>(left) /
                             static_cast<reduce_t>(right));
}

FF_NAMESPACE_END(internal)
FF_NAMESPACE_END(posdef)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)

#endif // FF_POSDEF_UTILS
