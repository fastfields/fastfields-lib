#ifndef FF_CPU_AUTOCAST
#define FF_CPU_AUTOCAST
#include <cstddef>
#include <cstdint>
#include "dlpack.h"
#include "impl/kernels/cuda_switch.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

// DLPack allows DLTensor.strides == NULL to mean "compact row-major". The
// dispatch layer dereferences strides unconditionally (canUse32BitIndexMath,
// copy_if_needed, and the impl loops), so a producer that omits strides would
// otherwise segfault. This RAII wrapper holds a normalised copy of the (POD)
// descriptor: when strides are missing it synthesises explicit contiguous
// strides (in elements, DLPack's unit) that outlive the call, so every
// downstream user sees non-null strides. The data pointer is copied verbatim,
// so writes through the normalised tensor still hit the caller's buffer.
struct ContiguousStrides {
    DLTensor   t;
    int64_t  * owned;

    // `normalize` lets callers skip a placeholder tensor (e.g. an optional
    // weight passed as a null-data descriptor), whose shape may be invalid and
    // whose strides are never read.
    explicit ContiguousStrides(const DLTensor & src, bool normalize = true)
        : t(src), owned(nullptr)
    {
        if (normalize && t.strides == nullptr) {
            const int32_t n = t.ndim > 0 ? t.ndim : 1;
            owned = new int64_t[n];
            int64_t s = 1;
            for (int32_t i = t.ndim - 1; i >= 0; --i) {
                owned[i] = s;
                s *= t.shape[i];
            }
            t.strides = owned;
        }
    }

    ~ContiguousStrides() { delete[] owned; }

    ContiguousStrides(const ContiguousStrides &)             = delete;
    ContiguousStrides & operator=(const ContiguousStrides &) = delete;
};

template <class T> struct RemovePointer        { using Type = T; };
template <class T> struct RemovePointer<T*>    { using Type = T; };
template <class T> struct RemoveConst          { using Type = T; };
template <class T> struct RemoveConst<const T> { using Type = T; };
template <class T> struct RemoveConstPointer   { using Type = typename RemoveConst<typename RemovePointer<T>::Type>::Type; };

template <class OutPointer, class InpPointer>
struct _copy_if_needed {
    using NonConstOutPointer = typename RemoveConst<OutPointer>::Type;
    using OutElemType = typename RemoveConstPointer<OutPointer>::Type;

    static inline OutPointer copy(InpPointer ptr, size_t numel)
    {
        NonConstOutPointer out = new OutElemType[numel];
        for (size_t i=0; i < numel; ++i)
            out[i] = static_cast<OutElemType>(ptr[i]);
        return const_cast<OutPointer>(out);
    }

    static inline void free(OutPointer ptr)
    {
        delete[] const_cast<NonConstOutPointer>(ptr);
    }
};

template <class SamePointer>
struct _copy_if_needed<SamePointer, SamePointer> {
    static inline SamePointer copy(SamePointer ptr, size_t /* numel */)
    {
        return ptr;
    }

    static inline void free(SamePointer /* ptr */)
    {
        // do nothing
    }
};

template <class NonConstPointer>
struct _copy_if_needed<const NonConstPointer, NonConstPointer> {
    using ConstPointer = const NonConstPointer;

    static inline ConstPointer copy(NonConstPointer ptr, size_t /* numel */)
    {
        return const_cast<ConstPointer>(ptr);
    }

    static inline void free(ConstPointer /* ptr */)
    {
        // do nothing
    }
};

template <class OutPointer, class InpPointer>
inline OutPointer copy_if_needed(InpPointer ptr, size_t numel)
{
    return _copy_if_needed<OutPointer, InpPointer>::copy(ptr, numel);
}

template <class InpPointer, class OutPointer>
inline void free_if_needed(OutPointer ptr)
{
    _copy_if_needed<OutPointer, InpPointer>::free(ptr);
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_CPU_AUTOCAST
