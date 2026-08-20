#ifndef FF_DISTANCE_MESH_UTILS_H
#define FF_DISTANCE_MESH_UTILS_H
#include "fastfields/core/cuda_switch.h"
#include "fastfields/core/utils.h"

// =============================================================================
//
//                          VECTOR MATH HELPERS
//
// =============================================================================

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_mesh)

template <typename value_t, typename offset_t>
struct StridedPointer {

    FF_CUHOSTDEV StridedPointer(value_t * ptr, offset_t stride):
        ptr(ptr), stride(stride) {}

    FF_CUHOSTDEV value_t & operator[] (offset_t n) { return ptr[n*stride]; }
    FF_CUHOSTDEV const value_t & operator[] (offset_t n) const { return ptr[n*stride]; }

    value_t * ptr;
    offset_t stride;
};

template <typename value_t, typename offset_t>
struct SizedStridedPointer {

    FF_CUHOSTDEV SizedStridedPointer(value_t * ptr, offset_t stride, offset_t size):
        ptr(ptr), stride(stride), size(size) {}

    FF_CUHOSTDEV value_t & operator[] (offset_t n) { return ptr[n*stride]; }
    FF_CUHOSTDEV const value_t & operator[] (offset_t n) const { return ptr[n*stride]; }

    value_t * ptr;
    offset_t stride;
    offset_t size;
};

// =============================================================================
//     1D VECTORS
// =============================================================================

template <typename offset_t>
struct Sized {

    FF_CUHOSTDEV virtual ~Sized() {}

    FF_CUHOSTDEV Sized(offset_t length): length(length) {}

    FF_CUHOSTDEV inline int size() const { return length; }

    offset_t length;
};

template <long N>
struct StaticSized {

    static constexpr long length = N;

    FF_CUHOSTDEV virtual ~StaticSized() {}

    FF_CUHOSTDEV inline int size() const { return length; }
};


template <int D, typename scalar_t>
struct AnyConstPoint {

    FF_CUHOSTDEV virtual ~AnyConstPoint() {}

    FF_CUHOSTDEV virtual const scalar_t& operator[] (int d) const = 0;
};

template <int D, typename scalar_t>
struct AnyPoint {

    FF_CUHOSTDEV virtual ~AnyPoint() {}

    FF_CUHOSTDEV virtual scalar_t& operator[] (int d) = 0;
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


template <int D, typename scalar_t, typename FinalType = void>
struct PointMixin: public AnyPoint<D, scalar_t> {

    using this_type         = PointMixin<D, scalar_t, FinalType>;
    using final_type        = FinalType;
    using static_type       = StaticPoint<D, scalar_t>;
    using point_type        = AnyPoint<D, scalar_t>;
    using const_point_type  = AnyConstPoint<D, scalar_t>;

    FF_CUHOSTDEV virtual ~PointMixin() {}

    // reference to final type

    FF_CUHOSTDEV inline
    final_type * thisptr() { return reinterpret_cast<final_type*>(this); }
    FF_CUHOSTDEV inline
    const final_type * thisptr() const { return reinterpret_cast<const final_type*>(this); }
    FF_CUHOSTDEV inline
    final_type & thisref() { return reinterpret_cast<final_type&>(*this); }
    FF_CUHOSTDEV inline
    const final_type & thisref() const { return reinterpret_cast<const final_type&>(*this); }

    // in-place operations

    FF_CUHOSTDEV inline
    final_type& copy_ (const const_point_type & other)
    { for (int d=0; d < D; ++d) (*this)[d] = other[d]; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& copy_ (const const_point_type & other, scalar_t alpha)
    { for (int d=0; d < D; ++d) (*this)[d] = other[d] * alpha; return thisref(); }


    FF_CUHOSTDEV inline
    final_type& copy_ (scalar_t alpha)
    { for (int d=0; d < D; ++d) (*this)[d] = alpha; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& operator = (const const_point_type & other)
    { return this->copy_(other); }

    FF_CUHOSTDEV inline
    final_type& operator = (scalar_t alpha)
    { return this->copy_(alpha); }

    FF_CUHOSTDEV inline
    final_type& add_ (const const_point_type & other)
    { for (int d=0; d < D; ++d) (*this)[d] += other[d]; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& add_ (const const_point_type & other, scalar_t alpha)
    { for (int d=0; d < D; ++d) (*this)[d] += other[d] * alpha; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& add_ (scalar_t alpha)
    { for (int d=0; d < D; ++d) (*this)[d] += alpha; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& operator += (const const_point_type & other)
    { return this->add_(other); }

    FF_CUHOSTDEV inline
    final_type& operator += (scalar_t alpha)
    { return this->add_(alpha); }

    FF_CUHOSTDEV inline
    final_type& sub_ (const const_point_type & other)
    { for (int d=0; d < D; ++d) (*this)[d] -= other[d]; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& sub_ (const const_point_type & other, scalar_t alpha)
    { for (int d=0; d < D; ++d) (*this)[d] -= other[d] * alpha; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& sub_ (scalar_t alpha)
    { for (int d=0; d < D; ++d) (*this)[d] -= alpha; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& operator -= (const const_point_type & other)
    { return this->sub_(other); }

    FF_CUHOSTDEV inline
    final_type& operator -= (scalar_t alpha)
    { return this->sub_(alpha); }

    FF_CUHOSTDEV inline
    final_type& mul_ (const const_point_type & other)
    { for (int d=0; d < D; ++d) (*this)[d] *= other[d]; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& mul_ (const const_point_type & other, scalar_t alpha)
    { for (int d=0; d < D; ++d) (*this)[d] *= other[d] * alpha; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& mul_ (scalar_t alpha)
    { for (int d=0; d < D; ++d) (*this)[d] *= alpha; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& operator *= (const const_point_type & other)
    { return this->mul_(other); }

    FF_CUHOSTDEV inline
    final_type& operator *= (scalar_t alpha)
    { return this->mul_(alpha); }

    FF_CUHOSTDEV inline
    final_type& div_ (const const_point_type & other)
    { for (int d=0; d < D; ++d) (*this)[d] /= other[d]; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& div_ (const const_point_type & other, scalar_t alpha)
    { for (int d=0; d < D; ++d) (*this)[d] /= other[d] * alpha; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& div_ (scalar_t alpha)
    { for (int d=0; d < D; ++d) (*this)[d] /= alpha; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& operator /= (const const_point_type & other)
    { return this->div_(other); }

    FF_CUHOSTDEV inline
    final_type& operator /= (scalar_t alpha)
    { return this->div_(alpha); }

    FF_CUHOSTDEV inline
    final_type& max_(const const_point_type & other)
    { for (int d=0; d < D; ++d) (*this)[d] = FF_NS::FF_DEVICE::max((*this)[d], other[d]); return thisref(); }

    FF_CUHOSTDEV inline
    final_type& min_(const const_point_type & other)
    { for (int d=0; d < D; ++d) (*this)[d] = FF_NS::FF_DEVICE::min((*this)[d], other[d]); return thisref(); }
    FF_CUHOSTDEV inline
    final_type& normalize_()
    {
        scalar_t nrm = static_cast<scalar_t>(0);
        for (int d=0; d < D; ++d) nrm += (*this)[d] * (*this)[d];
        nrm = sqrt(nrm);
        for (int d=0; d < D; ++d) (*this)[d] /= nrm;
        return thisref();
      }


    // out-of-place operations (fill self)

    FF_CUHOSTDEV inline
    final_type& addto_(const const_point_type & lhs, const const_point_type & rhs)
    { for (int d=0; d < D; ++d) (*this)[d] = lhs[d] + rhs[d]; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& addto_(const const_point_type & lhs, const const_point_type & rhs, scalar_t alpha)
    { for (int d=0; d < D; ++d) (*this)[d] = lhs[d] + rhs[d] * alpha; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& subto_(const const_point_type & lhs, const const_point_type & rhs)
    { for (int d=0; d < D; ++d) (*this)[d] = lhs[d] - rhs[d]; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& subto_(const const_point_type & lhs, const const_point_type & rhs, scalar_t alpha)
    { for (int d=0; d < D; ++d) (*this)[d] = lhs[d] - rhs[d] * alpha; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& multo_(const const_point_type & lhs, const const_point_type & rhs)
    { for (int d=0; d < D; ++d) (*this)[d] = lhs[d] * rhs[d]; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& multo_(const const_point_type & lhs, const const_point_type & rhs, scalar_t alpha)
    { for (int d=0; d < D; ++d) (*this)[d] = lhs[d] * rhs[d] * alpha; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& divto_(const const_point_type & lhs, const const_point_type & rhs)
    { for (int d=0; d < D; ++d) (*this)[d] = lhs[d] / rhs[d]; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& divto_(const const_point_type & lhs, const const_point_type & rhs, scalar_t alpha)
    { for (int d=0; d < D; ++d) (*this)[d] = lhs[d] / rhs[d] * alpha; return thisref(); }

    FF_CUHOSTDEV inline
    final_type& maxto_(const const_point_type & lhs, const const_point_type & rhs)
    { for (int d=0; d < D; ++d) (*this)[d] = FF_NS::FF_DEVICE::max(lhs[d], rhs[d]); return thisref(); }

    FF_CUHOSTDEV inline
    final_type& minto_(const const_point_type & lhs, const const_point_type & rhs)
    { for (int d=0; d < D; ++d) (*this)[d] = FF_NS::FF_DEVICE::min(lhs[d], rhs[d]); return thisref(); }

    FF_CUHOSTDEV inline
    final_type& crossto_(const const_point_type & lhs, const const_point_type & rhs)
    {
        // !! only works in 3D
        (*this)[0] =  lhs[1]*rhs[2] - lhs[2]*rhs[1];
        (*this)[1] = -lhs[0]*rhs[2] + lhs[2]*rhs[0];
        (*this)[2] =  lhs[0]*rhs[1] - lhs[1]*rhs[0];
        return thisref();
    }

};


template <int D, typename scalar_t, typename FinalType = void>
struct ConstPointMixin: public AnyConstPoint<D, scalar_t> {

    using this_type         = PointMixin<D, scalar_t, FinalType>;
    using final_type        = FinalType;
    using static_type       = StaticPoint<D, scalar_t>;
    using point_type        = AnyPoint<D, scalar_t>;
    using const_point_type  = AnyConstPoint<D, scalar_t>;

    FF_CUHOSTDEV virtual ~ConstPointMixin() {}

    // out-of-place operations (return static point)

    FF_CUHOSTDEV inline
    static_type copy () const
    { return static_type(*this); }

    FF_CUHOSTDEV inline
    static_type add(const const_point_type & other) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = (*this)[d] + other[d]; return out; }

    FF_CUHOSTDEV inline
    static_type add(const const_point_type & other, scalar_t alpha) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = (*this)[d] + other[d] * alpha; return out; }

    FF_CUHOSTDEV inline
    static_type add(scalar_t alpha) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = (*this)[d] + alpha; return out; }

    FF_CUHOSTDEV inline
    static_type operator+(const const_point_type & rhs) const
    { return this->add(rhs); }

    FF_CUHOSTDEV inline
    static_type operator+(scalar_t alpha) const
    { return this->add(alpha); }

    FF_CUHOSTDEV inline
    static_type sub(const const_point_type & other) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = (*this)[d] - other[d]; return out; }

    FF_CUHOSTDEV inline
    static_type sub(const const_point_type & other, scalar_t alpha) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = (*this)[d] - other[d] * alpha; return out; }

    FF_CUHOSTDEV inline
    static_type sub(scalar_t alpha) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = (*this)[d] - alpha; return out; }

    FF_CUHOSTDEV inline
    static_type operator-(const const_point_type & rhs) const
    { return this->sub(rhs); }

    FF_CUHOSTDEV inline
    static_type operator-(scalar_t alpha) const
    { return this->sub(alpha); }

    FF_CUHOSTDEV inline
    static_type mul(const const_point_type & other) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = (*this)[d] * other[d]; return out; }

    FF_CUHOSTDEV inline
    static_type mul(const const_point_type & other, scalar_t alpha) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = (*this)[d] * other[d] * alpha; return out; }

    FF_CUHOSTDEV inline
    static_type mul(scalar_t alpha) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = (*this)[d] * alpha; return out; }

    FF_CUHOSTDEV inline
    static_type operator*(const const_point_type & rhs) const
    { return this->mul(rhs); }

    FF_CUHOSTDEV inline
    static_type operator*(scalar_t alpha) const
    { return this->mul(alpha); }

    FF_CUHOSTDEV inline
    static_type div(const const_point_type & other) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = (*this)[d] / other[d]; return out; }

    FF_CUHOSTDEV inline
    static_type div(const const_point_type & other, scalar_t alpha) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = (*this)[d] / (other[d] * alpha); return out; }

    FF_CUHOSTDEV inline
    static_type div(scalar_t alpha) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = (*this)[d] / alpha; return out; }

    FF_CUHOSTDEV inline
    static_type operator/(const const_point_type & rhs) const
    { return this->div(rhs); }

    FF_CUHOSTDEV inline
    static_type operator/(scalar_t alpha) const
    { return this->div(alpha); }

    FF_CUHOSTDEV inline
    static_type max(const const_point_type & other) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = FF_NS::FF_DEVICE::max((*this)[d], other[d]); return out; }

    FF_CUHOSTDEV inline
    static_type max(scalar_t alpha) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = FF_NS::FF_DEVICE::max((*this)[d], alpha); return out; }

    FF_CUHOSTDEV inline
    static_type min(const const_point_type & other) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = FF_NS::FF_DEVICE::min((*this)[d], other[d]); return out; }

    FF_CUHOSTDEV inline
    static_type min(scalar_t alpha) const
    { static_type out; for (int d=0; d < D; ++d) out[d] = FF_NS::FF_DEVICE::min((*this)[d], alpha); return out; }

    FF_CUHOSTDEV inline
    static_type cross(const const_point_type & other) const
    {
        // !! only works in 3D
        static_type out;
        out[0] =  (*this)[1]*other[2] - (*this)[2]*other[1];
        out[1] = -(*this)[0]*other[2] + (*this)[2]*other[0];
        out[2] =  (*this)[0]*other[1] - (*this)[1]*other[0];
        return out;
    }

    // operations that return a scalar

    FF_CUHOSTDEV inline
    scalar_t dot(const const_point_type & other) const
    { scalar_t out = static_cast<scalar_t>(0); for (int d=0; d < D; ++d) out += (*this)[d] * other[d]; return out; }

    FF_CUHOSTDEV inline
    scalar_t sum() const
    { scalar_t out = static_cast<scalar_t>(0); for (int d=0; d < D; ++d) out += (*this)[d]; return out; }

    FF_CUHOSTDEV inline
    scalar_t prod() const
    { scalar_t out = static_cast<scalar_t>(1); for (int d=0; d < D; ++d) out *= (*this)[d]; return out; }

    FF_CUHOSTDEV inline
    scalar_t sqnorm() const
    { return this->dot(*this); }

    FF_CUHOSTDEV inline
    scalar_t norm() const
    { return sqrt(this->sqnorm()); }

    // view

    template <int begin, int end>
    FF_CUHOSTDEV inline
    StaticPoint<end-begin, scalar_t> copyview()
    {
        StaticPoint<end-begin, scalar_t> out;
        for (int d=0; d<(end-begin); ++d)
            out[d] = (*this)[begin+d];
        return out;
    }

};

template <int D, typename scalar_t>
struct StaticPoint:
    public PointMixin <D, scalar_t, StaticPoint <D, scalar_t> >,
    public ConstPointMixin <D, scalar_t, StaticPoint <D, scalar_t> >
{
    using any_const_point = AnyConstPoint<D, scalar_t>;

    FF_CUHOSTDEV virtual ~StaticPoint() {}
    FF_CUHOSTDEV StaticPoint() = default;
    FF_CUHOSTDEV StaticPoint(const any_const_point & other)
    { this->copy_(other); }

    FF_CUHOSTDEV inline scalar_t& operator[] (int d) { return data[d]; };
    FF_CUHOSTDEV inline const scalar_t& operator[] (int d) const { return data[d]; };

    scalar_t data[D];

    // reference view

    template <int begin, int end>
    FF_CUHOSTDEV inline
    RefPoint<end-begin, scalar_t> view()
    {
        return RefPoint<end-begin, scalar_t>(data + begin);
    }

    template <int begin, int end>
    FF_CUHOSTDEV inline
    ConstRefPoint<end-begin, scalar_t> view() const
    {
        return ConstRefPoint<end-begin, scalar_t>(data + begin);
    }

};

template <int D, typename scalar_t>
struct RefPoint:
    public PointMixin <D, scalar_t, RefPoint <D, scalar_t> >,
    public ConstPointMixin <D, scalar_t, RefPoint <D, scalar_t> >
{
    FF_CUHOSTDEV virtual ~RefPoint() {}
    FF_CUHOSTDEV RefPoint(scalar_t * data): data(data) {}

    FF_CUHOSTDEV inline scalar_t& operator[] (int d) { return data[d]; };
    FF_CUHOSTDEV inline const scalar_t& operator[] (int d) const { return data[d]; };

    scalar_t * data;


    // reference view

    template <int begin, int end>
    FF_CUHOSTDEV inline
    RefPoint<end-begin, scalar_t> view()
    {
        return RefPoint<end-begin, scalar_t>(data + begin);
    }

    template <int begin, int end>
    FF_CUHOSTDEV inline
    ConstRefPoint<end-begin, scalar_t> view() const
    {
        return ConstRefPoint<end-begin, scalar_t>(data + begin);
    }
};

template <int D, typename scalar_t>
struct ConstRefPoint:
    public ConstPointMixin <D, scalar_t, ConstRefPoint<D, scalar_t> >
{
    FF_CUHOSTDEV virtual ~ConstRefPoint() {}
    FF_CUHOSTDEV ConstRefPoint(const scalar_t * data): data(data) {}

    FF_CUHOSTDEV inline const scalar_t& operator[] (int d) const { return data[d]; };

    const scalar_t * data;

    // reference view

    template <int begin, int end>
    FF_CUHOSTDEV inline
    ConstRefPoint<end-begin, scalar_t> view() const
    {
        return ConstRefPoint<end-begin, scalar_t>(data + begin);
    }
};

template <int D, typename scalar_t, typename offset_t>
struct StridedPoint:
    public PointMixin <D, scalar_t, StridedPoint<D, scalar_t, offset_t> >,
    public ConstPointMixin <D, scalar_t, StridedPoint<D, scalar_t, offset_t> >
{
    FF_CUHOSTDEV virtual ~StridedPoint() {}
    FF_CUHOSTDEV StridedPoint(scalar_t * data, offset_t stride): data(data), stride(stride) {}

    FF_CUHOSTDEV inline scalar_t& operator[] (int d) { return data[d*stride]; };
    FF_CUHOSTDEV inline const scalar_t& operator[] (int d) const { return data[d*stride]; };

    FF_CUHOSTDEV inline StridedPoint<D, scalar_t, offset_t> & operator= (const AnyConstPoint<D, scalar_t> & other)
    {
        printf("assign (%ld)\n", data);
        for (int d=0; d<D; ++d) (*this)[d] = other[d];
        return *this;
    }

    scalar_t * data;
    offset_t stride;

    // reference view

    template <int begin, int end>
    FF_CUHOSTDEV inline
    StridedPoint<end-begin, scalar_t, offset_t> view()
    {
        return StridedPoint<end-begin, scalar_t, offset_t>(data + begin, stride);
    }

    template <int begin, int end>
    FF_CUHOSTDEV inline
    ConstStridedPoint<end-begin, scalar_t, offset_t> view() const
    {
        return ConstStridedPoint<end-begin, scalar_t, offset_t>(data + begin, stride);
    }
};

template <int D, typename scalar_t, typename offset_t>
struct ConstStridedPoint:
    public ConstPointMixin <D, scalar_t, ConstStridedPoint<D, scalar_t, offset_t> >
{
    FF_CUHOSTDEV virtual ~ConstStridedPoint() {}
    FF_CUHOSTDEV ConstStridedPoint(const scalar_t * data, offset_t stride): data(data), stride(stride) {}

    FF_CUHOSTDEV inline const scalar_t& operator[] (int d) const { return data[d*stride]; };

    const scalar_t * data;
    offset_t stride;

    // reference view

    template <int begin, int end>
    FF_CUHOSTDEV inline
    const ConstStridedPoint<end-begin, scalar_t, offset_t> view() const
    {
        return ConstStridedPoint<end-begin, scalar_t, offset_t>(data + begin, stride);
    }
};

// =============================================================================
//     VECTORS OF VECTORS
// =============================================================================

template <int N, int D, typename scalar_t>
struct StaticPointList: public StaticSized<N> {

    using PointType = RefPoint<D, scalar_t>;
    using ConstPointType = ConstRefPoint<D, scalar_t>;
    FF_CUHOSTDEV virtual ~StaticPointList() {}

    FF_CUHOSTDEV inline int size() const { return N; }

    FF_CUHOSTDEV inline PointType operator[] (int n)
    { return PointType(data + n*D); };
    FF_CUHOSTDEV inline ConstPointType operator[] (int n)  const
    { return ConstPointType(data + n*D); };

    scalar_t data[N*D];
};

template <int D, typename scalar_t>
struct RefPointList {

    using PointType = RefPoint<D, scalar_t>;
    using ConstPointType = ConstRefPoint<D, scalar_t>;

    FF_CUHOSTDEV virtual ~RefPointList() {}
    FF_CUHOSTDEV RefPointList(scalar_t * data): data(data) {}

    FF_CUHOSTDEV inline PointType operator[] (int n)
    { return PointType(data + n*D); };
    FF_CUHOSTDEV inline ConstPointType operator[] (int n)  const
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

    FF_CUHOSTDEV virtual ~RefPointListSized() {}
    FF_CUHOSTDEV RefPointListSized(scalar_t * data, offset_t length):
        BaseList(data), BaseSized(length) {}
};

template <int D, typename scalar_t>
struct ConstRefPointList {

    using ConstPointType = ConstRefPoint<D, scalar_t>;

    FF_CUHOSTDEV virtual ~ConstRefPointList() {}
    FF_CUHOSTDEV ConstRefPointList(const scalar_t * data): data(data) {}

    FF_CUHOSTDEV inline ConstPointType operator[] (int n)  const
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

    FF_CUHOSTDEV virtual ~ConstRefPointListSized() {}
    FF_CUHOSTDEV ConstRefPointListSized(const scalar_t * data, offset_t length):
        BaseList(data), BaseSized(length) {}

};

template <int D, typename scalar_t, typename offset_t>
struct StridedPointList {

    using PointType = StridedPoint<D, scalar_t, offset_t>;
    using ConstPointType = ConstStridedPoint<D, scalar_t, offset_t>;

    FF_CUHOSTDEV virtual ~StridedPointList() {}
    FF_CUHOSTDEV
    StridedPointList(scalar_t * data,
                     offset_t stride_elem,
                     offset_t stride_channel):
        data(data), stride_elem(stride_elem), stride_channel(stride_channel) {}

    FF_CUHOSTDEV inline PointType operator[] (int n)
    { return PointType(data + n*stride_elem, stride_channel); };

    FF_CUHOSTDEV inline ConstPointType operator[] (int n)  const
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

    FF_CUHOSTDEV virtual ~StridedPointListSized() {}
    FF_CUHOSTDEV StridedPointListSized(scalar_t * data,
                     offset_t stride_elem,
                     offset_t stride_channel,
                     offset_t length):
        BaseList(data, stride_elem, stride_channel), BaseSized(length) {}
};

template <int D, typename scalar_t, typename offset_t>
struct ConstStridedPointList {

    using ConstPointType = ConstStridedPoint<D, scalar_t, offset_t>;

    FF_CUHOSTDEV virtual ~ConstStridedPointList() {}
    FF_CUHOSTDEV
    ConstStridedPointList(const scalar_t * data,
                          offset_t stride_elem,
                          offset_t stride_channel):
        data(data), stride_elem(stride_elem), stride_channel(stride_channel) {}

    FF_CUHOSTDEV inline ConstPointType operator[] (int n)  const
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

    FF_CUHOSTDEV virtual ~ConstStridedPointListSized() {}
    FF_CUHOSTDEV ConstStridedPointListSized(const scalar_t * data,
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

    FF_CUHOSTDEV virtual ~StaticPointArray() {}

    template <typename... T>
    FF_CUHOSTDEV  inline
    typename returned<_Count<T...>::value>::type & at(int n0, T... n)
    {
        return (*this)[n0].at(n...);
    };
    FF_CUHOSTDEV inline
    SubArrayType & at (int n0)
    {
        return reinterpret_cast<SubArrayType&>(data + n0 * stride0);
    };
    FF_CUHOSTDEV inline
    SubArrayType & operator[] (int n0)
    {
        return this->at(n0);
    };

    template <typename... T>
    FF_CUHOSTDEV  inline
    const typename returned<_Count<T...>::value>::type & at(int n0, T... n) const
    {
        return (*this)[n0].at(n...);
    };
    FF_CUHOSTDEV inline
    const SubArrayType& at(int n0) const
    {
        return reinterpret_cast<const SubArrayType&>(data + n0 * stride0);
    };
    FF_CUHOSTDEV inline
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

    FF_CUHOSTDEV virtual ~StaticPointArray() {}

    FF_CUHOSTDEV inline
    PointType& at (int n0)
    {
        return reinterpret_cast<PointType&>(data + n0 * D);
    };
    FF_CUHOSTDEV inline
    PointType& operator[] (int n0)
    {
        return this->at(n0);
    };

    FF_CUHOSTDEV inline
    const PointType& at (int n0) const
    {
        return reinterpret_cast<const PointType&>(data + n0 * D);
    };
    FF_CUHOSTDEV inline
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

    FF_CUHOSTDEV virtual ~RefPointArray() {}

    template <typename... T>
    FF_CUHOSTDEV  inline
    typename returned<_Count<T...>::value>::type & at(int n0, T... n)
    {
        return (*this)[n0].at(n...);
    };
    FF_CUHOSTDEV inline
    SubArrayType & at(int n0)
    {
        return reinterpret_cast<SubArrayType&>(data + n0 * stride0);
    };
    FF_CUHOSTDEV inline
    SubArrayType & operator[](int n0)
    {
        return this->at(n0);
    };


    template <typename... T>
    FF_CUHOSTDEV  inline
    const typename returned<_Count<T...>::value>::type & at(int n0, T... n) const
    {
        return (*this)[n0].at(n...);
    };
    FF_CUHOSTDEV inline
    const SubArrayType& at(int n0) const
    {
        return reinterpret_cast<const SubArrayType&>(data + n0 * stride0);
    };
    FF_CUHOSTDEV inline
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

    FF_CUHOSTDEV virtual ~RefPointArray() {}

    FF_CUHOSTDEV inline
    PointType& at(int n0)
    {
        return reinterpret_cast<PointType&>(data + n0 * D);
    };
    FF_CUHOSTDEV inline
    PointType& operator[] (int n0)
    {
        return this->at(n0);
    };

    FF_CUHOSTDEV inline
    const PointType& at(int n0) const
    {
        return reinterpret_cast<const PointType&>(data + n0 * D);
    };
    FF_CUHOSTDEV inline
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

    FF_CUHOSTDEV virtual ~StridedPointArray() {}

    template <typename Stride>
    FF_CUHOSTDEV
    StridedPointArray(scalar_t * data, const Stride & stride):
        data(data), stride(stride) {}

    FF_CUHOSTDEV
    StridedPointArray(scalar_t * data = nullptr):
        data(data), stride() { stride.copy_(1); }

    template <typename... T>
    FF_CUHOSTDEV  inline
    typename returned<_Count<T...>::value>::type at(int n0, T... n)
    {
        return (*this)[n0].at(n...);
    };
    FF_CUHOSTDEV inline
    SubArrayType at(int n0)
    {
        return SubArrayType(data + n0 * stride[0], stride.template view<1,nbatch+1>());
    };
    FF_CUHOSTDEV inline
    SubArrayType operator[] (int n0)
    {
        return this->at(n0);
    };

    template <typename... T>
    FF_CUHOSTDEV  inline
    const typename returned<_Count<T...>::value>::type at(int n0, T... n) const
    {
        return (*this)[n0].at(n...);
    };
    FF_CUHOSTDEV inline
    const SubArrayType at(int n0) const
    {
        return SubArrayType(data + n0 * stride[0], stride.template view<1,nbatch+1>());
    };
    FF_CUHOSTDEV inline
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

    FF_CUHOSTDEV virtual ~StridedPointArray() {}
    template <typename Stride>
    FF_CUHOSTDEV
    StridedPointArray(scalar_t * data, const Stride & stride):
        data(data), stride(stride) {}

    FF_CUHOSTDEV  inline PointType at(int n)
    {
        return PointType(data + n*stride[0], stride[1]);
    };
    FF_CUHOSTDEV  inline PointType operator[] (int n)
    {
        return this->at(n);
    };

    FF_CUHOSTDEV  inline ConstPointType at(int n) const
    {
        return ConstPointType(data + n*stride[0], stride[1]);
    };
    FF_CUHOSTDEV  inline ConstPointType operator[] (int n) const
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

    FF_CUHOSTDEV virtual ~ConstStridedPointArray() {}

    template <typename Stride>
    FF_CUHOSTDEV
    ConstStridedPointArray(const scalar_t * data, const Stride & stride):
        data(data), stride(stride) {}

    FF_CUHOSTDEV
    ConstStridedPointArray(scalar_t * data = nullptr):
        data(data), stride() { stride.copy_(1); }

    template <typename... T>
    FF_CUHOSTDEV  inline
    const typename returned<_Count<T...>::value>::type at(int n0, T... n) const
    {
        return (*this)[n0].at(n...);
    };
    FF_CUHOSTDEV inline
    const SubArrayType at(int n0) const
    {
        return SubArrayType(data + n0 * stride[0], stride.template view<1,nbatch+1>());
    };
    FF_CUHOSTDEV inline
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

    FF_CUHOSTDEV virtual ~ConstStridedPointArray() {}
    template <typename Stride>
    FF_CUHOSTDEV
    ConstStridedPointArray(const scalar_t * data, const Stride & stride):
        data(data), stride(stride) {}

    FF_CUHOSTDEV
    ConstStridedPointArray(scalar_t * data = nullptr):
        data(data), stride() { stride.copy_(1); }


    FF_CUHOSTDEV  inline ConstPointType at(int n) const
    {
        return ConstPointType(data + n*stride[0], stride[1]);
    };
    FF_CUHOSTDEV  inline ConstPointType operator[] (int n) const
    {
        return this->at(n);
    };

    const scalar_t * data;
    StaticPoint<2, offset_t> stride;
};


FF_NAMESPACE_END(distance_mesh)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)

#endif // FF_DISTANCE_MESH_UTILS_H
