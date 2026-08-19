#pragma once
#ifndef FF_AUTOCAST
#define FF_AUTOCAST
#include <cstddef>
#include <cstdint>
#include "fastfields/core/dlpack.h"
#include "fastfields/core/cuda_switch.h"

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

// ------------------------------------------------------------------ host
// The staging buffers below are *host* memory, and the two backends want them
// allocated differently: CUDA wants them page-locked (cudaMallocHost) so the
// following H2D copy can be async, the CPU backend just wants new[]. That is
// the only difference there has ever been between the two autocast headers, so
// it is injected here rather than by shipping the header twice.
//
// FF_AUTOCAST_PINNED_HOST may be set explicitly; by default it follows the
// compiler, because nvcc compiles the CUDA library and nothing else.
#ifndef FF_AUTOCAST_PINNED_HOST
#  ifdef __CUDACC__
#    define FF_AUTOCAST_PINNED_HOST 1
#  else
#    define FF_AUTOCAST_PINNED_HOST 0
#  endif
#endif

#if FF_AUTOCAST_PINNED_HOST

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

#else

template <class ElemType>
inline ElemType * hostNew(size_t numel)
{
    return new ElemType[numel];
}

template <class ElemType>
inline void hostDelete(ElemType * ptr)
{
    delete[] const_cast<typename RemoveConst<ElemType>::Type *>(ptr);
}

#endif // FF_AUTOCAST_PINNED_HOST

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

// ------------------------------------------------------------------ RAII
//
// PROTOTYPE -- see the accompanying proposal. `IndexArray<offset_t>` is the
// RAII form of the hand-managed `copy_if_needed` / `free_if_needed` pair. It
// replaces
//
//     const offset_t * _size = copy_if_needed<offset_t*>(size, n);
//     ... 30 lines, any of which may throw ...
//     free_if_needed<int64_t*>(_size);          // not reached if one did
//
// with
//
//     IndexArray<offset_t> _size(size, n);
//
// Three things follow.
//
//  1. The leak on the throwing path goes away. It is not hypothetical: every
//     reg_* impl wrapper does `new reduce_t[...]` after the copy, and every
//     CUDA launcher throws `std::bad_alloc` / `std::range_error` from
//     `copyToDevice` / `GET_BLOCKS`.
//
//  2. The allocator leaves the narrowing path. The arrays being narrowed are
//     shape/stride vectors of length `nbatch + ndim (+1)`; FF_INDEX_SBO covers
//     every rank the library dispatches (ndim <= 3) with a heap fallback for
//     anything larger.
//
//  3. On the CUDA build that deletes a `cudaMallocHost` + `cudaFreeHost` pair
//     per narrowed array per call. Page-locked allocation is among the most
//     expensive host-side CUDA calls there is and it synchronises the device.
//     The header's justification for it -- "so the following H2D copy can be
//     async" -- does not hold for the launchers that actually exist: 365 of
//     the impl/cuda upload sites call the *synchronous* `copyToDevice`, which
//     gains nothing from a pinned source, and the handful that call
//     `copyToDeviceAsync` document pageable sources as safe.
//
// A stack buffer is safe for both: `copyToDevice` is synchronous, and
// `copyToDeviceAsync` from pageable memory stages through a driver buffer
// before returning (its own comment says so).
#ifndef FF_INDEX_SBO
#  define FF_INDEX_SBO 8
#endif

template <class offset_t>
class IndexArray
{
    offset_t         _sbo[FF_INDEX_SBO];
    offset_t       * _heap;
    const offset_t * _ptr;

public:
    IndexArray(const int64_t * src, size_t numel) : _heap(nullptr), _ptr(nullptr)
    {
        if (!src) return;
        offset_t * dst = _sbo;
        if (numel > FF_INDEX_SBO)
            dst = _heap = new offset_t[numel];
        for (size_t i = 0; i < numel; ++i)
            dst[i] = static_cast<offset_t>(src[i]);
        _ptr = dst;
    }

    ~IndexArray() { delete[] _heap; }

    inline operator const offset_t * () const { return _ptr; }
    inline const offset_t * get() const { return _ptr; }

    IndexArray(const IndexArray &)             = delete;
    IndexArray & operator=(const IndexArray &) = delete;
};

// Nothing to narrow: borrow the caller's array. Same interface, zero cost.
// This is the only specialisation instantiated when FF_INDEX32 == 0, at which
// point the five `_copy_if_needed` specialisations above have no callers left.
template <>
class IndexArray<int64_t>
{
    const int64_t * _ptr;

public:
    IndexArray(const int64_t * src, size_t /* numel */) : _ptr(src) {}

    inline operator const int64_t * () const { return _ptr; }
    inline const int64_t * get() const { return _ptr; }

    IndexArray(const IndexArray &)             = delete;
    IndexArray & operator=(const IndexArray &) = delete;
};

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_AUTOCAST
