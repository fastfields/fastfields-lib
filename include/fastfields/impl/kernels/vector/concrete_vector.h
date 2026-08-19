#ifndef FF_VECTOR_CONCRETEVECTOR_H
#define FF_VECTOR_CONCRETEVECTOR_H
#include <initializer_list>
#include "forward.h"
#include "traits.h"
#include "abstract_vector.h"
#include "weak_vector.h"
#include "../cuda_switch.h"

namespace ff {

/***********************************************************************
 *
 *                      STATIC-SIZED VECTOR
 *
 **********************************************************************/

template <typename T, unsigned long N, typename D>
class Vector: public AbstractVector<T, N, 1, Vector<T, N, D> > {
public:

    //--- types --------------------------------------------------------

    using this_type = Vector<T, N, D>;
    using this_type_final = Vector<T, N>;
    using final_type = typename internal::guess_type<this_type_final, D>::value;

    // inherited
    using parent_type = AbstractVector<T, N, 1, this_type>;
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
    virtual ~Vector() {}

    //--- constructors -------------------------------------------------

    CUHOSTDEV
    Vector() {}

    CUHOSTDEV
    Vector(const T & value)
    {
        this->fill(value);
    }

    CUHOSTDEV
    Vector(std::initializer_list<T> list)
    {
        this->copy(list.begin(), list.end());
    }

    template <typename U>
    CUHOSTDEV
    Vector(const U * other) // assume stride 1 and length N
    {
        this->copy(other);
    }

    template <typename Iterator>
    CUHOSTDEV
    Vector(Iterator begin, const Iterator & end)
    {
        this->copy(begin, end);
    }

    template <typename OT, unsigned long ON, long OS, typename OD>
    CUHOSTDEV
    Vector(const AbstractSizedPointer<OT,ON,OS,OD> & other)
    {
        this->copy(other);
    }

    //--- other --------------------------------------------------------

    CUHOSTDEV
    inline const T * data() const
    {
        return this->_data;
    }

    CUHOSTDEV
    inline T * data()
    {
        return this->_data;
    }

protected:
    T _data[N];
};

/***********************************************************************
 *
 *                      DYNAMIC-SIZED VECTOR
 *
 **********************************************************************/

template <typename T, typename D>
class Vector<T,DynamicSize,D>:
public AbstractVector<T, DynamicSize, 1, Vector<T,DynamicSize,D> >
{
public:

    //--- types --------------------------------------------------------

    using this_type = Vector<T, DynamicSize, D>;
    using this_type_final = Vector<T, DynamicSize>;
    using final_type = typename internal::guess_type<this_type_final, D>::value;

    // inherited
    using parent_type = AbstractVector<T, DynamicSize, 1, this_type>;
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

    CUHOST
    virtual ~Vector()
    {
        delete this->_data;
    }

    //--- constructors -------------------------------------------------

    CUHOST
    Vector(size_type size):
        parent_type(size), _data(new T[size])
    {}

    CUHOST
    Vector(size_type size, const T & value):
        parent_type(size), _data(new T[size])
    {
        this->fill(value);
    }

    CUHOST
    template <typename U>
    Vector(size_type size, const U * other):
        parent_type(size), _data(new T[size])
    {
        this->copy(other);
    }

    CUHOST
    template <typename Iterator>
    Vector(size_type size, Iterator begin, const Iterator & end):
        parent_type(size), _data(new T[size])
    {
        this->copy(begin, end);
    }

    CUHOST
    Vector(std::initializer_list<T> other):
        parent_type(other.size()), _data(new T[other.size()])
    {
        this->copy(other.begin(), other.end());
    }

    CUHOST
    template <typename OT, unsigned long ON, long OS, typename  OD>
    Vector(const AbstractVector<OT,ON,OS,OD> & other):
        parent_type(other.size()), _data(new T[other.size()])
    {
        this->copy(other);
    }

    //--- other --------------------------------------------------------

    CUHOST
    inline const T * data() const
    {
        return this->_data;
    }

    CUHOST
    inline T * data()
    {
        return this->_data;
    }

protected:
    T * _data;
};

} // namespace ff

// =====================================================================
//  Unpack API
// =====================================================================

template <typename T, unsigned long N, typename D>
struct std::tuple_size< ff::Vector<T, N, D> >
{
    using value_type = std::size_t;
    static constexpr value_type value = static_cast<std::size_t>(N);
    using type = std::integral_constant<value_type, value>;
};

template <std::size_t I, typename T, unsigned long N, typename D>
struct std::tuple_element< I, ff::Vector<T, N, D> >
{
    using type = T;
};

#endif // FF_VECTOR_CONCRETEVECTOR_H
