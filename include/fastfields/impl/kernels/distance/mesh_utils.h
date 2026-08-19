#ifndef FF_DISTANCE_MESH_UTILS_H
#define FF_DISTANCE_MESH_UTILS_H
#include "fastfields/core/cuda_switch.h"
#include "../utils.h"
#include <type_traits>
#include <teeny/teeny.h>

// =============================================================================
//
//                          VECTOR MATH HELPERS
//
// =============================================================================

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_mesh)

template <typename value_t, typename offset_t>
struct StridedPointer {

    CUHOSTDEV StridedPointer(value_t * ptr, offset_t stride):
        ptr(ptr), stride(stride) {}

    CUHOSTDEV value_t & operator[] (offset_t n) { return ptr[n*stride]; }
    CUHOSTDEV const value_t & operator[] (offset_t n) const { return ptr[n*stride]; }

    value_t * ptr;
    offset_t stride;
};

template <typename value_t, typename offset_t>
struct SizedStridedPointer {

    CUHOSTDEV SizedStridedPointer(value_t * ptr, offset_t stride, offset_t size):
        ptr(ptr), stride(stride), size(size) {}

    CUHOSTDEV value_t & operator[] (offset_t n) { return ptr[n*stride]; }
    CUHOSTDEV const value_t & operator[] (offset_t n) const { return ptr[n*stride]; }

    value_t * ptr;
    offset_t stride;
    offset_t size;
};

// -----------------------------------------------------------------------------
// SFINAE helper: match a "point-like" argument (anything non-arithmetic) so the
// templated vector overloads never collide with the scalar overloads. This is
// what lets us drop the AnyPoint / AnyConstPoint pure-virtual bases: the mixins
// accept any concrete point type by template instead of by virtual reference.
// -----------------------------------------------------------------------------
template <class P>
using _if_point = typename std::enable_if<
    !std::is_arithmetic<typename std::decay<P>::type>::value, bool>::type;

// =============================================================================
//     1D VECTORS
// =============================================================================

template <typename offset_t>
struct Sized {

    CUHOSTDEV Sized(offset_t length): length(length) {}

    CUHOSTDEV inline int size() const { return length; }

    offset_t length;
};

template <long N>
struct StaticSized {

    static constexpr long length = N;

    CUHOSTDEV inline int size() const { return length; }
};


template <int D, typename scalar_t>
struct StaticPoint;
template <int D, typename scalar_t>
struct RefPoint;
template <int D, typename scalar_t>
struct ConstRefPoint;
template <int D, typename scalar_t, typename offset_t>
struct StridedPoint;
template <int D, typename scalar_t, typename offset_t>
struct ConstStridedPoint;


// -----------------------------------------------------------------------------
// Vector-algebra mixins, now TEENY-BACKED. Every leaf point type exposes a
// `t()` returning a `cuda::std::mdspan`-based teeny VIEW over its D elements
// (a `tny::local` for the stack point, a strided/contiguous `tny::wrap` view
// for the reference points). The mixin method bodies route ALL vector math
// through teeny's host+device vector ops (`copy_`/`add_`/`sub_`/`mul_`/`div_`,
// fused axpy `add_(x,alpha)`, `maximum`/`minimum` into self, `dot`/`sqnorm`/
// `norm`, `cross`) instead of the former hand-rolled per-`d` loops. Reductions
// pin the accumulator to `scalar_t` (`dot<scalar_t>` etc.) so branch decisions
// stay bit-identical to the original loops. The final (derived) type is reached
// via static_cast (offset-correct under multiple inheritance).
template <int D, typename scalar_t, typename FinalType = void>
struct PointMixin {

    using this_type         = PointMixin<D, scalar_t, FinalType>;
    using final_type        = FinalType;
    using static_type       = StaticPoint<D, scalar_t>;

    // reference to final type

    CUHOSTDEV inline
    final_type & thisref() { return static_cast<final_type&>(*this); }
    CUHOSTDEV inline
    const final_type & thisref() const { return static_cast<const final_type&>(*this); }

    // in-place operations

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    final_type& copy_ (const P & other)
    { thisref().t().copy_(other.t()); return thisref(); }

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    final_type& copy_ (const P & other, scalar_t alpha)
    { auto self = thisref().t(); self.zero_(); self.add_(other.t(), alpha); return thisref(); }

    CUHOSTDEV inline
    final_type& copy_ (scalar_t alpha)
    { thisref().t().fill_(alpha); return thisref(); }

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    final_type& operator = (const P & other)
    { return this->copy_(other); }

    CUHOSTDEV inline
    final_type& operator = (scalar_t alpha)
    { return this->copy_(alpha); }

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    final_type& add_ (const P & other)
    { thisref().t().add_(other.t()); return thisref(); }

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    final_type& add_ (const P & other, scalar_t alpha)
    { thisref().t().add_(other.t(), alpha); return thisref(); }

    CUHOSTDEV inline
    final_type& add_ (scalar_t alpha)
    { thisref().t().add_(alpha); return thisref(); }

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    final_type& operator += (const P & other)
    { return this->add_(other); }

    CUHOSTDEV inline
    final_type& operator += (scalar_t alpha)
    { return this->add_(alpha); }

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    final_type& sub_ (const P & other)
    { thisref().t().sub_(other.t()); return thisref(); }

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    final_type& sub_ (const P & other, scalar_t alpha)
    { thisref().t().sub_(other.t(), alpha); return thisref(); }

    CUHOSTDEV inline
    final_type& sub_ (scalar_t alpha)
    { thisref().t().sub_(alpha); return thisref(); }

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    final_type& operator -= (const P & other)
    { return this->sub_(other); }

    CUHOSTDEV inline
    final_type& operator -= (scalar_t alpha)
    { return this->sub_(alpha); }

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    final_type& mul_ (const P & other)
    { thisref().t().mul_(other.t()); return thisref(); }

    CUHOSTDEV inline
    final_type& mul_ (scalar_t alpha)
    { thisref().t().mul_(alpha); return thisref(); }

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    final_type& operator *= (const P & other)
    { return this->mul_(other); }

    CUHOSTDEV inline
    final_type& operator *= (scalar_t alpha)
    { return this->mul_(alpha); }

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    final_type& div_ (const P & other)
    { thisref().t().div_(other.t()); return thisref(); }

    CUHOSTDEV inline
    final_type& div_ (scalar_t alpha)
    { thisref().t().div_(alpha); return thisref(); }

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    final_type& operator /= (const P & other)
    { return this->div_(other); }

    CUHOSTDEV inline
    final_type& operator /= (scalar_t alpha)
    { return this->div_(alpha); }

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    final_type& max_(const P & other)
    { auto self = thisref().t(); self.maximum(other.t(), tny::into(self)); return thisref(); }

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    final_type& min_(const P & other)
    { auto self = thisref().t(); self.minimum(other.t(), tny::into(self)); return thisref(); }

    CUHOSTDEV inline
    final_type& normalize_()
    {
        auto self = thisref().t();
        scalar_t nrm = tny::norm<scalar_t>(self);
        self.div_(nrm);
        return thisref();
    }


    // out-of-place operations (fill self); self does not alias lhs/rhs

    template <class L, class R, _if_point<L> = true>
    CUHOSTDEV inline
    final_type& addto_(const L & lhs, const R & rhs)
    { auto self = thisref().t(); self.copy_(lhs.t()); self.add_(rhs.t()); return thisref(); }

    template <class L, class R, _if_point<L> = true>
    CUHOSTDEV inline
    final_type& addto_(const L & lhs, const R & rhs, scalar_t alpha)
    { auto self = thisref().t(); self.copy_(lhs.t()); self.add_(rhs.t(), alpha); return thisref(); }

    template <class L, class R, _if_point<L> = true>
    CUHOSTDEV inline
    final_type& subto_(const L & lhs, const R & rhs)
    { auto self = thisref().t(); self.copy_(lhs.t()); self.sub_(rhs.t()); return thisref(); }

    template <class L, class R, _if_point<L> = true>
    CUHOSTDEV inline
    final_type& subto_(const L & lhs, const R & rhs, scalar_t alpha)
    { auto self = thisref().t(); self.copy_(lhs.t()); self.sub_(rhs.t(), alpha); return thisref(); }

    template <class L, class R, _if_point<L> = true>
    CUHOSTDEV inline
    final_type& crossto_(const L & lhs, const R & rhs)
    {
        // !! only works in 3D
        auto self = thisref().t();
        tny::cross(lhs.t(), rhs.t(), tny::into(self));
        return thisref();
    }

};


// CRTP mixin providing const (read-only / out-of-place) vector ops.
template <int D, typename scalar_t, typename FinalType = void>
struct ConstPointMixin {

    using this_type         = ConstPointMixin<D, scalar_t, FinalType>;
    using final_type        = FinalType;
    using static_type       = StaticPoint<D, scalar_t>;

    // reference to final type (const)

    CUHOSTDEV inline
    const final_type & cthisref() const { return static_cast<const final_type&>(*this); }

    // out-of-place operations (return static point)

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    static_type sub(const P & other) const
    { static_type out; out.copy_(cthisref()); out.sub_(other); return out; }

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    static_type operator-(const P & rhs) const
    { return this->sub(rhs); }

    // operations that return a scalar

    template <class P, _if_point<P> = true>
    CUHOSTDEV inline
    scalar_t dot(const P & other) const
    { return tny::dot<scalar_t>(cthisref().t(), other.t()); }

    CUHOSTDEV inline
    scalar_t sqnorm() const
    { return tny::sqnorm<scalar_t>(cthisref().t()); }

    CUHOSTDEV inline
    scalar_t norm() const
    { return tny::norm<scalar_t>(cthisref().t()); }

};

// -----------------------------------------------------------------------------
// Leaf point types. Each holds only its raw storage handle (so the strided
// list / array carriers and the std::sort FaceIterator can still bump the
// pointer directly), exposes operator[] for element access, and a `t()` that
// materialises the teeny view the mixins run their math on.
// -----------------------------------------------------------------------------

template <int D, typename scalar_t>
struct StaticPoint:
    public PointMixin <D, scalar_t, StaticPoint <D, scalar_t> >,
    public ConstPointMixin <D, scalar_t, StaticPoint <D, scalar_t> >
{
    using PointMixin<D, scalar_t, StaticPoint<D, scalar_t> >::operator=;

    CUHOSTDEV StaticPoint() = default;

    template <class P, _if_point<P> = true>
    CUHOSTDEV StaticPoint(const P & other)
    { this->copy_(other); }

    CUHOSTDEV inline scalar_t& operator[] (int d) { return _data.data()[d]; };
    CUHOSTDEV inline const scalar_t& operator[] (int d) const { return _data.data()[d]; };

    CUHOSTDEV inline auto t()       { return _data.view(); }
    CUHOSTDEV inline auto t() const { return _data.view(); }

    tny::local<scalar_t, tny::shape<D> > _data;

    // reference view over a static [begin,end) sub-range (used by the strided
    // array carriers to slice their inline stride vector).

    template <int begin, int end>
    CUHOSTDEV inline
    RefPoint<end-begin, scalar_t> view()
    {
        return RefPoint<end-begin, scalar_t>(_data.data() + begin);
    }

    template <int begin, int end>
    CUHOSTDEV inline
    ConstRefPoint<end-begin, scalar_t> view() const
    {
        return ConstRefPoint<end-begin, scalar_t>(_data.data() + begin);
    }

};

template <int D, typename scalar_t>
struct RefPoint:
    public PointMixin <D, scalar_t, RefPoint <D, scalar_t> >,
    public ConstPointMixin <D, scalar_t, RefPoint <D, scalar_t> >
{
    using PointMixin<D, scalar_t, RefPoint<D, scalar_t> >::operator=;

    CUHOSTDEV RefPoint(scalar_t * data): data(data) {}

    CUHOSTDEV inline scalar_t& operator[] (int d) { return data[d]; };
    CUHOSTDEV inline const scalar_t& operator[] (int d) const { return data[d]; };

    CUHOSTDEV inline auto t()       { return tny::wrap(data, tny::shape<D>{}); }
    CUHOSTDEV inline auto t() const { return tny::wrap(const_cast<const scalar_t*>(data), tny::shape<D>{}); }

    scalar_t * data;
};

template <int D, typename scalar_t>
struct ConstRefPoint:
    public ConstPointMixin <D, scalar_t, ConstRefPoint<D, scalar_t> >
{
    CUHOSTDEV ConstRefPoint(const scalar_t * data): data(data) {}

    CUHOSTDEV inline const scalar_t& operator[] (int d) const { return data[d]; };

    CUHOSTDEV inline auto t() const { return tny::wrap(data, tny::shape<D>{}); }

    const scalar_t * data;
};

template <int D, typename scalar_t, typename offset_t>
struct StridedPoint:
    public PointMixin <D, scalar_t, StridedPoint<D, scalar_t, offset_t> >,
    public ConstPointMixin <D, scalar_t, StridedPoint<D, scalar_t, offset_t> >
{
    CUHOSTDEV StridedPoint(scalar_t * data, offset_t stride): data(data), stride(stride) {}

    CUHOSTDEV inline scalar_t& operator[] (int d) { return data[d*stride]; };
    CUHOSTDEV inline const scalar_t& operator[] (int d) const { return data[d*stride]; };

    CUHOSTDEV inline auto t()       { return tny::wrap(data, tny::shape<D>{}, {static_cast<int64_t>(stride)}); }
    CUHOSTDEV inline auto t() const { return tny::wrap(const_cast<const scalar_t*>(data), tny::shape<D>{}, {static_cast<int64_t>(stride)}); }

    // Elementwise assignment from any (non-strided) point. A StridedPoint rhs
    // instead selects the implicitly-declared copy-assignment (pointer rebind),
    // exactly as before the AnyConstPoint base was removed.
    template <class P, _if_point<P> = true>
    CUHOSTDEV inline StridedPoint<D, scalar_t, offset_t> & operator= (const P & other)
    {
        this->t().copy_(other.t());
        return *this;
    }

    CUHOSTDEV inline StridedPoint<D, scalar_t, offset_t> & operator= (scalar_t alpha)
    {
        this->t().fill_(alpha);
        return *this;
    }

    scalar_t * data;
    offset_t stride;
};

template <int D, typename scalar_t, typename offset_t>
struct ConstStridedPoint:
    public ConstPointMixin <D, scalar_t, ConstStridedPoint<D, scalar_t, offset_t> >
{
    CUHOSTDEV ConstStridedPoint(const scalar_t * data, offset_t stride): data(data), stride(stride) {}

    CUHOSTDEV inline const scalar_t& operator[] (int d) const { return data[d*stride]; };

    CUHOSTDEV inline auto t() const { return tny::wrap(data, tny::shape<D>{}, {static_cast<int64_t>(stride)}); }

    const scalar_t * data;
    offset_t stride;
};

// =============================================================================
//     VECTORS OF VECTORS
// =============================================================================

template <int N, int D, typename scalar_t>
struct StaticPointList: public StaticSized<N> {

    using PointType = RefPoint<D, scalar_t>;
    using ConstPointType = ConstRefPoint<D, scalar_t>;

    CUHOSTDEV inline int size() const { return N; }

    CUHOSTDEV inline PointType operator[] (int n)
    { return PointType(data + n*D); };
    CUHOSTDEV inline ConstPointType operator[] (int n)  const
    { return ConstPointType(data + n*D); };

    scalar_t data[N*D];
};

template <int D, typename scalar_t>
struct RefPointList {

    using PointType = RefPoint<D, scalar_t>;
    using ConstPointType = ConstRefPoint<D, scalar_t>;

    CUHOSTDEV RefPointList(scalar_t * data): data(data) {}

    CUHOSTDEV inline PointType operator[] (int n)
    { return PointType(data + n*D); };
    CUHOSTDEV inline ConstPointType operator[] (int n)  const
    { return ConstPointType(data + n*D); };

    scalar_t * data = nullptr;
};

template <int D, typename scalar_t, typename offset_t = long>
struct RefPointListSized:
    public RefPointList<D, scalar_t>,
    public Sized<offset_t>
{
    using BaseList = RefPointList<D, scalar_t>;
    using BaseSized = Sized<offset_t>;

    CUHOSTDEV RefPointListSized(scalar_t * data, offset_t length):
        BaseList(data), BaseSized(length) {}
};

template <int D, typename scalar_t>
struct ConstRefPointList {

    using ConstPointType = ConstRefPoint<D, scalar_t>;

    CUHOSTDEV ConstRefPointList(const scalar_t * data): data(data) {}

    CUHOSTDEV inline ConstPointType operator[] (int n)  const
    { return ConstPointType(data + n*D); };

    const scalar_t * data = nullptr;
};

template <int D, typename scalar_t, typename offset_t = long>
struct ConstRefPointListSized:
    public ConstRefPointList<D, scalar_t>,
    public Sized<offset_t>
{
    using BaseList = ConstRefPointList<D, scalar_t>;
    using BaseSized = Sized<offset_t>;

    CUHOSTDEV ConstRefPointListSized(const scalar_t * data, offset_t length):
        BaseList(data), BaseSized(length) {}

};

template <int D, typename scalar_t, typename offset_t>
struct StridedPointList {

    using PointType = StridedPoint<D, scalar_t, offset_t>;
    using ConstPointType = ConstStridedPoint<D, scalar_t, offset_t>;

    CUHOSTDEV
    StridedPointList(scalar_t * data,
                     offset_t stride_elem,
                     offset_t stride_channel):
        data(data), stride_elem(stride_elem), stride_channel(stride_channel) {}

    CUHOSTDEV inline PointType operator[] (int n)
    { return PointType(data + n*stride_elem, stride_channel); };

    CUHOSTDEV inline ConstPointType operator[] (int n)  const
    { return ConstPointType(data + n*stride_elem, stride_channel); };

    scalar_t * data = nullptr;
    offset_t stride_elem = static_cast<offset_t>(1);
    offset_t stride_channel = static_cast<offset_t>(1);
};

template <int D, typename scalar_t, typename offset_t = long>
struct StridedPointListSized:
    public StridedPointList<D, scalar_t, offset_t>,
    public Sized<offset_t>
{
    using BaseList = StridedPointList<D, scalar_t, offset_t>;
    using BaseSized = Sized<offset_t>;

    CUHOSTDEV StridedPointListSized(scalar_t * data,
                     offset_t stride_elem,
                     offset_t stride_channel,
                     offset_t length):
        BaseList(data, stride_elem, stride_channel), BaseSized(length) {}
};

template <int D, typename scalar_t, typename offset_t>
struct ConstStridedPointList {

    using ConstPointType = ConstStridedPoint<D, scalar_t, offset_t>;

    CUHOSTDEV
    ConstStridedPointList(const scalar_t * data,
                          offset_t stride_elem,
                          offset_t stride_channel):
        data(data), stride_elem(stride_elem), stride_channel(stride_channel) {}

    CUHOSTDEV inline ConstPointType operator[] (int n)  const
    { return ConstPointType(data + n*stride_elem, stride_channel); };

    const scalar_t * data = nullptr;
    offset_t stride_elem = static_cast<offset_t>(1);
    offset_t stride_channel = static_cast<offset_t>(1);
};

template <int D, typename scalar_t, typename offset_t = long>
struct ConstStridedPointListSized:
    public ConstStridedPointList<D, scalar_t, offset_t>,
    public Sized<offset_t>
{
    using BaseList = ConstStridedPointList<D, scalar_t, offset_t>;
    using BaseSized = Sized<offset_t>;

    CUHOSTDEV ConstStridedPointListSized(const scalar_t * data,
                     offset_t stride_elem,
                     offset_t stride_channel,
                     offset_t length):
        BaseList(data, stride_elem, stride_channel), BaseSized(length) {}
};


// =============================================================================
//     ARRAYS OF VECTORS
// =============================================================================

template <int... N>
struct _Prod {};
template <int N0, int... N>
struct _Prod<N0, N...> { static constexpr long value = N0 * _Prod<N...>::value; };
template <int N0>
struct _Prod<N0> { static constexpr long value = N0; };
template <>
struct _Prod<> { static constexpr long value = 1; };

template <typename... N>
struct _Count {};
template <typename N0, typename... N>
struct _Count<N0, N...> { static constexpr int value = 1 + _Count<N...>::value; };
template <typename N0>
struct _Count<N0> { static constexpr int value = 1; };
template <>
struct _Count<> { static constexpr int value = 0; };

template <int... N>
struct _CountInt {};
template <int N0, int... N>
struct _CountInt<N0, N...> { static constexpr int value = 1 + _CountInt<N...>::value; };
template <int N0>
struct _CountInt<N0> { static constexpr int value = 1; };
template <>
struct _CountInt<> { static constexpr int value = 0; };

template <int D, typename scalar_t, int N0, int... N>
struct StaticPointArray {

    using PointType    = StaticPoint<D, scalar_t>;
    using SubArrayType = StaticPointArray<D, scalar_t, N...>;
    static constexpr long stride0 = _Prod<N...>::value * D;
    static constexpr int  nbatch  = _CountInt<N...>::value + 1;

    template <int COUNT, bool dummy=true> // need dummy parameter to avoid explicit specialization
    struct returned { using type = typename SubArrayType::template returned<COUNT-1>::type; };
    template <bool dummy>
    struct returned<nbatch-1, dummy> { using type = PointType; };
    template <bool dummy>
    struct returned<0, dummy> { using type = SubArrayType; };

    template <typename... T>
    CUHOSTDEV  inline
    typename returned<_Count<T...>::value>::type & at(int n0, T... n)
    {
        return (*this)[n0].at(n...);
    };
    CUHOSTDEV inline
    SubArrayType & at (int n0)
    {
        return reinterpret_cast<SubArrayType&>(data + n0 * stride0);
    };
    CUHOSTDEV inline
    SubArrayType & operator[] (int n0)
    {
        return this->at(n0);
    };

    template <typename... T>
    CUHOSTDEV  inline
    const typename returned<_Count<T...>::value>::type & at(int n0, T... n) const
    {
        return (*this)[n0].at(n...);
    };
    CUHOSTDEV inline
    const SubArrayType& at(int n0) const
    {
        return reinterpret_cast<const SubArrayType&>(data + n0 * stride0);
    };
    CUHOSTDEV inline
    const SubArrayType& operator[] (int n0) const
    {
        return this->at(n0);
    };

    scalar_t data[N0*stride0];
};

template <int D, typename scalar_t, int N0>
struct StaticPointArray<D, scalar_t, N0> {

    using PointType = StaticPoint<D, scalar_t>;

    template <int COUNT>
    struct returned { using type = PointType; };

    CUHOSTDEV inline
    PointType& at (int n0)
    {
        return reinterpret_cast<PointType&>(data + n0 * D);
    };
    CUHOSTDEV inline
    PointType& operator[] (int n0)
    {
        return this->at(n0);
    };

    CUHOSTDEV inline
    const PointType& at (int n0) const
    {
        return reinterpret_cast<const PointType&>(data + n0 * D);
    };
    CUHOSTDEV inline
    const PointType& operator[] (int n0) const
    {
        return this->at(n0);
    };

    scalar_t data[N0*D];
};

template <int D, typename scalar_t, int... N>
struct RefPointArray {};

template <int D, typename scalar_t, int N1, int... N>
struct RefPointArray<D, scalar_t, N1, N...> {

    using PointType    = StaticPoint<D, scalar_t>;
    using SubArrayType = RefPointArray<D, scalar_t, N...>;
    static constexpr long stride0 = _Prod<N...>::value * D * N1;
    static constexpr int  nbatch  = _CountInt<N...>::value + 2;

    template <int COUNT, bool dummy = true>
    struct returned { using type = typename SubArrayType::template returned<COUNT-1>::type; };
    template <bool dummy>
    struct returned<nbatch-1, dummy> { using type = PointType; };
    template <bool dummy>
    struct returned<0, dummy> { using type = SubArrayType; };

    template <typename... T>
    CUHOSTDEV  inline
    typename returned<_Count<T...>::value>::type & at(int n0, T... n)
    {
        return (*this)[n0].at(n...);
    };
    CUHOSTDEV inline
    SubArrayType & at(int n0)
    {
        return reinterpret_cast<SubArrayType&>(data + n0 * stride0);
    };
    CUHOSTDEV inline
    SubArrayType & operator[](int n0)
    {
        return this->at(n0);
    };


    template <typename... T>
    CUHOSTDEV  inline
    const typename returned<_Count<T...>::value>::type & at(int n0, T... n) const
    {
        return (*this)[n0].at(n...);
    };
    CUHOSTDEV inline
    const SubArrayType& at(int n0) const
    {
        return reinterpret_cast<const SubArrayType&>(data + n0 * stride0);
    };
    CUHOSTDEV inline
    const SubArrayType& operator[] (int n0) const
    {
        return this->at(n0);
    };

    scalar_t * data;
};

template <int D, typename scalar_t>
struct RefPointArray<D, scalar_t> {

    using PointType = StaticPoint<D, scalar_t>;

    template <int COUNT>
    struct returned { using type = PointType; };

    CUHOSTDEV inline
    PointType& at(int n0)
    {
        return reinterpret_cast<PointType&>(data + n0 * D);
    };
    CUHOSTDEV inline
    PointType& operator[] (int n0)
    {
        return this->at(n0);
    };

    CUHOSTDEV inline
    const PointType& at(int n0) const
    {
        return reinterpret_cast<const PointType&>(data + n0 * D);
    };
    CUHOSTDEV inline
    const PointType& operator[] (int n0) const
    {
        return this->at(n0);
    };

    scalar_t * data;
};

template <int D, typename scalar_t, typename offset_t, int... N>
struct StridedPointArray {};

template <int D, typename scalar_t, typename offset_t, int N1, int... N>
struct StridedPointArray<D, scalar_t, offset_t, N1, N...> {

    using PointType    = StaticPoint<D, scalar_t>;
    using SubArrayType = StridedPointArray<D, scalar_t, offset_t, N...>;
    static constexpr long stride0 = _Prod<N...>::value * D * N1;
    static constexpr int  nbatch  = _CountInt<N...>::value + 2;

    template <int COUNT, bool dummy = true>
    struct returned { using type = typename SubArrayType::template returned<COUNT-1>::type; };
    template <bool dummy>
    struct returned<nbatch-1, dummy> { using type = PointType; };
    template <bool dummy>
    struct returned<0, dummy> { using type = SubArrayType; };

    template <typename Stride>
    CUHOSTDEV
    StridedPointArray(scalar_t * data, const Stride & stride):
        data(data), stride(stride) {}

    CUHOSTDEV
    StridedPointArray(scalar_t * data = nullptr):
        data(data), stride() { stride.copy_(1); }

    template <typename... T>
    CUHOSTDEV  inline
    typename returned<_Count<T...>::value>::type at(int n0, T... n)
    {
        return (*this)[n0].at(n...);
    };
    CUHOSTDEV inline
    SubArrayType at(int n0)
    {
        return SubArrayType(data + n0 * stride[0], stride.template view<1,nbatch+1>());
    };
    CUHOSTDEV inline
    SubArrayType operator[] (int n0)
    {
        return this->at(n0);
    };

    template <typename... T>
    CUHOSTDEV  inline
    const typename returned<_Count<T...>::value>::type at(int n0, T... n) const
    {
        return (*this)[n0].at(n...);
    };
    CUHOSTDEV inline
    const SubArrayType at(int n0) const
    {
        return SubArrayType(data + n0 * stride[0], stride.template view<1,nbatch+1>());
    };
    CUHOSTDEV inline
    const SubArrayType operator[] (int n0) const
    {
        return this->at(n0);
    };

    scalar_t * data;
    StaticPoint<nbatch+1, offset_t> stride;
};

template <int D, typename scalar_t, typename offset_t>
struct StridedPointArray<D, scalar_t, offset_t> {

    using PointType = StridedPoint<D, scalar_t, offset_t>;
    using ConstPointType = StridedPoint<D, const scalar_t, offset_t>;

    template <typename Stride>
    CUHOSTDEV
    StridedPointArray(scalar_t * data, const Stride & stride):
        data(data), stride(stride) {}

    CUHOSTDEV  inline PointType at(int n)
    {
        return PointType(data + n*stride[0], stride[1]);
    };
    CUHOSTDEV  inline PointType operator[] (int n)
    {
        return this->at(n);
    };

    CUHOSTDEV  inline ConstPointType at(int n) const
    {
        return ConstPointType(data + n*stride[0], stride[1]);
    };
    CUHOSTDEV  inline ConstPointType operator[] (int n) const
    {
        return this->at(n);
    };

    scalar_t * data;
    StaticPoint<2, offset_t> stride;
};

template <int D, typename scalar_t, typename offset_t, int... N>
struct ConstStridedPointArray {};

template <int D, typename scalar_t, typename offset_t, int N1, int... N>
struct ConstStridedPointArray<D, scalar_t, offset_t, N1, N...> {

    using PointType    = StaticPoint<D, scalar_t>;
    using SubArrayType = ConstStridedPointArray<D, scalar_t, offset_t, N...>;
    static constexpr long stride0 = _Prod<N...>::value * D * N1;
    static constexpr int  nbatch  = _CountInt<N...>::value + 2;

    template <int COUNT, bool dummy = true>
    struct returned { using type = typename SubArrayType::template returned<COUNT-1>::type; };
    template <bool dummy>
    struct returned<nbatch-1, dummy> { using type = PointType; };
    template <bool dummy>
    struct returned<0, dummy> { using type = SubArrayType; };

    template <typename Stride>
    CUHOSTDEV
    ConstStridedPointArray(const scalar_t * data, const Stride & stride):
        data(data), stride(stride) {}

    CUHOSTDEV
    ConstStridedPointArray(scalar_t * data = nullptr):
        data(data), stride() { stride.copy_(1); }

    template <typename... T>
    CUHOSTDEV  inline
    const typename returned<_Count<T...>::value>::type at(int n0, T... n) const
    {
        return (*this)[n0].at(n...);
    };
    CUHOSTDEV inline
    const SubArrayType at(int n0) const
    {
        return SubArrayType(data + n0 * stride[0], stride.template view<1,nbatch+1>());
    };
    CUHOSTDEV inline
    const SubArrayType operator[] (int n0) const
    {
        return this->at(n0);
    };

    const scalar_t * data;
    StaticPoint<nbatch+1, offset_t> stride;
};

template <int D, typename scalar_t, typename offset_t>
struct ConstStridedPointArray<D, scalar_t, offset_t> {

    using PointType = StridedPoint<D, scalar_t, offset_t>;
    using ConstPointType = ConstStridedPoint<D, scalar_t, offset_t>;

    template <typename Stride>
    CUHOSTDEV
    ConstStridedPointArray(const scalar_t * data, const Stride & stride):
        data(data), stride(stride) {}

    CUHOSTDEV
    ConstStridedPointArray(scalar_t * data = nullptr):
        data(data), stride() { stride.copy_(1); }


    CUHOSTDEV  inline ConstPointType at(int n) const
    {
        return ConstPointType(data + n*stride[0], stride[1]);
    };
    CUHOSTDEV  inline ConstPointType operator[] (int n) const
    {
        return this->at(n);
    };

    const scalar_t * data;
    StaticPoint<2, offset_t> stride;
};


FF_NAMESPACE_END(distance_mesh)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_DISTANCE_MESH_UTILS_H
