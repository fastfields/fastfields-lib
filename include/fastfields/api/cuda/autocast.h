#ifndef FF_CUDA_AUTOCAST
#define FF_CUDA_AUTOCAST
#include <cstddef>
#include <cstdint>
#include "dlpack.h"
#include "impl/kernels/cuda_switch.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

// DLPack allows DLTensor.strides == NULL to mean "compact row-major". The
// dispatch layer dereferences strides unconditionally (canUse32BitIndexMath,
// copy_if_needed, and the impl launchers), so a producer that omits strides
// would otherwise segfault. This RAII wrapper holds a normalised copy of the
// (POD) descriptor: when strides are missing it synthesises explicit contiguous
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

template <class ElemType>
inline ElemType * hostNew(size_t numel)
{
    void * out;
    if (cudaMallocHost(&out, numel * sizeof(ElemType)))
        throw std::runtime_error("cudaMallocHost failed");
    return static_cast<ElemType*>(out);
}

template <class ElemType>
inline void hostDelete(ElemType * ptr)
{
    if (cudaFreeHost(const_cast<void*>(static_cast<const void*>(ptr))))
        throw std::runtime_error("cudaFreeHost failed");
}

template <class OutPointer, class InpPointer>
struct _copy_if_needed {
    using NonConstOutPointer = typename RemoveConst<OutPointer>::Type;
    using OutElemType        = typename RemoveConstPointer<OutPointer>::Type;

    static inline OutPointer copy(InpPointer ptr, size_t numel)
    {
        auto out = hostNew<OutElemType>(numel);
        for (size_t i=0; i < numel; ++i)
            out[i] = static_cast<OutElemType>(ptr[i]);
        return const_cast<OutPointer>(out);
    }

    static inline void free(OutPointer ptr)
    {
        hostDelete(ptr);
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

// A4: no-op passthrough when only constness differs (same element type, no
// dtype narrowing). Callers do `copy_if_needed<offset_t*>(size /*const T**/, n)`,
// so on the 64-bit path (offset_t == int64_t) the copy pair is <T*, const T*>:
// there is nothing to narrow, so hand the source array straight through instead
// of a cudaMallocHost + copy. `free_if_needed<int64_t*>(_size /*const T**/)`
// instantiates the mirror pair <const T*, T*>, which must also be a no-op so the
// borrowed pointer is never cudaFreeHost-d. Both directions are provided so the
// copy and free stay symmetric; the 32-bit (narrowing) path keeps the primary
// template. `T` is deduced to a non-pointer element type, so these never overlap
// the SamePointer / <const NonConstPointer, NonConstPointer> specializations.
template <class T>
struct _copy_if_needed<T*, const T*> {
    static inline T* copy(const T* ptr, size_t /* numel */)
    {
        return const_cast<T*>(ptr);
    }

    static inline void free(T* /* ptr */)
    {
        // do nothing
    }
};

template <class T>
struct _copy_if_needed<const T*, T*> {
    static inline const T* copy(T* ptr, size_t /* numel */)
    {
        return ptr;
    }

    static inline void free(const T* /* ptr */)
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

#endif // FF_CUDA_AUTOCAST
