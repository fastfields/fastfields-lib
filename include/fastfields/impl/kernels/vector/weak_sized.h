#ifndef FF_VECTOR_WEAKSIZEDREF_H
#define FF_VECTOR_WEAKSIZEDREF_H
#include "forward.h"
#include "abstract_sized.h"
#include "../cuda_switch.h"

namespace ff {

template <typename T, unsigned long N, long S, typename D>
class WeakSizedRef:
public AbstractSizedPointer<T, N, S, WeakSizedRef<T,N,S,D> >
{
public:

    //--- types --------------------------------------------------------

    using this_type = WeakSizedRef<T,N,S,D>;
    using this_type_final = WeakSizedRef<T,N,S>;
    using final_type = typename internal::guess_type<this_type_final, D>::value;

    // inherited
    using parent_type = AbstractSizedPointer<T, N, S, this_type>;
    using value_type = typename parent_type::value_type;
    using size_type = typename parent_type::size_type;
    using offset_type = typename parent_type::offset_type;
    using reference = typename parent_type::reference;
    using const_reference = typename parent_type::const_reference;
    using pointer = typename parent_type::pointer;
    using const_pointer = typename parent_type::const_pointer;
    using iterator = typename parent_type::iterator;
    using const_iterator = typename parent_type::const_iterator;
    using reverse_iterator = typename parent_type::reverse_iterator;
    using const_reverse_iterator = typename parent_type::const_reverse_iterator;
    static constexpr offset_type static_stride = parent_type::static_stride;

    //--- destructor ---------------------------------------------------

    CUHOSTDEV
    virtual ~WeakSizedRef() {}

    //--- constructors -------------------------------------------------

    CUHOSTDEV
    WeakSizedRef(size_type size, offset_type stride):
        parent_type(size, stride), _data(nullptr)
    {}

    CUHOSTDEV
    WeakSizedRef(size_type size, T * other = nullptr, offset_type stride = S):
        parent_type(size, stride), _data(other)
    {}

    CUHOSTDEV
    WeakSizedRef(T * other, offset_type stride = S):
        parent_type(N, stride), _data(other)
    {}

    CUHOSTDEV
    template <unsigned long ON, long OS, typename OD>
    WeakSizedRef(const AbstractSizedPointer<T, ON, OS, OD> & other):
        parent_type(other.size(), other.stride()), _data(other.data())
    {}

    //--- virtual ------------------------------------------------------

    CUHOSTDEV
    inline T * data() const
    {
        return _data;
    }

    //--- unbind -------------------------------------------------------

    template <typename U, typename... V>
    CUHOSTDEV
    void unbind(U& x, V&... y) const
    {
        if (this->size() == 0) return;
        x = (*this)[0];
        WeakSizedRef<T, N == DynamicSize ? DynamicSize : N-1, S> next(
            this->size()-1, this->data() + this->stride(), this->stride());
        return next.unbind(y...);
    }

    template <typename U>
    CUHOSTDEV
    void unbind(U& x) const
    {
        if (this->size() == 0) return;
        x = (*this)[0];
    }

    CUHOSTDEV
    void unbind() const
    {}

protected:
    T * _data;
};

template <typename T>
CUHOSTDEV
WeakSizedRef<T> weak_ref(unsigned long N, T * ptr)
{
    return WeakSizedRef<T>(N, ptr);
}

template <typename T, unsigned long N>
CUHOSTDEV
WeakSizedRef<T,N> weak_ref(T ptr[N])
{
    return WeakSizedRef<T,N>(N, ptr);
}

template <typename T>
CUHOSTDEV
WeakSizedRef<T,DynamicSize,DynamicStride> weak_ref(unsigned long N, T * ptr, long stride)
{
    return WeakSizedRef<T,DynamicSize,DynamicStride>(N, ptr, stride);
}

template <typename T, unsigned long N>
CUHOSTDEV
WeakSizedRef<T,N,DynamicStride> weak_ref(T ptr[N], long stride)
{
    return WeakSizedRef<T,N,DynamicStride>(N, ptr, stride);
}

template <typename T, unsigned N, long S, typename D>
CUHOSTDEV
WeakSizedRef<T,N,S> weak_ref(const AbstractSizedPointer<T,S,N,D> & ptr)
{
    return WeakSizedRef<T,N,S>(ptr);
}


} // namespace ff

#endif // FF_VECTOR_WEAKSIZEDREF_H
