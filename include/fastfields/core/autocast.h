#pragma once
#include <cstddef>
#include <cstdint>
// `hostNew` / `hostDelete` throw std::runtime_error on the pinned-host
// (cudaMallocHost) path. This header never included <stdexcept> for them: it
// compiled only because every translation unit that includes it happens to
// include <stdexcept> first. A standalone .cu that includes just this header
// fails to compile without this line.
#include <stdexcept>
#include <fastfields/core/dlpack.h>
#include <fastfields/core/cuda_switch.h>

FF_NAMESPACE_BEGIN(FF_NS)
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

/***********************************************************************
 *                    THE SAME THING, BUT SCOPED                       *
 ***********************************************************************/

/**
 * `IndexArray<offset_t>` is the RAII form of the `copy_if_needed` /
 * `free_if_needed` pair above, for the one job that pair is actually used
 * for: handing a `const int64_t *` shape or stride array to a kernel
 * templated on `offset_t`.
 *
 * Two things it fixes.
 *
 * **The manual pair leaks whenever anything between the two throws**, and in
 * every dispatch wrapper in this tree something between them can. The shape
 * is always
 *
 *     const offset_t * _size = copy_if_needed<offset_t*>(size, n);   // allocates
 *     ... as_weights(), new reduce_t[], the impl call ...            // can throw
 *     free_if_needed<int64_t*>(_size);                               // skipped
 *
 * and on the CUDA side the impl call throws by design -- every
 * `FF_CUDA_LAUNCH` does, and so does every `copyToDevice`. Measured with
 * ASan/LSan on the `reg_field` wrapper shape: 32 bytes in 2 allocations
 * escape per throwing call on the narrow arm. It is a per-call leak on a
 * path user code is expected to hit (an invalid argument), not a
 * once-per-process one.
 *
 * **It does not allocate at all for the sizes that actually occur.** These
 * arrays are `nbatch + ndim + 1` long -- single digits in every caller -- so
 * the elements live in the object itself and the narrow arm costs no
 * allocator traffic. Only a genuinely large rank falls back to `hostNew`.
 * That matters most under `FF_AUTOCAST_PINNED_HOST`, where the fallback is
 * `cudaMallocHost`: a page-locking syscall, on the per-call path, three to
 * five times per launch.
 *
 * The wide arm (`offset_t == int64_t`) is specialised below to borrow the
 * caller's array verbatim -- no copy, no storage -- exactly as
 * `_copy_if_needed<T*, const T*>` already does for that case. So with
 * `FF_INDEX32=0`, where both arms name `int64_t`, this whole class is a
 * pointer copy.
 *
 * Deliberately not convertible to a *non*-const pointer: nothing in the
 * dispatch layer writes through a shape or stride array, and the wide arm
 * aliases the caller's memory.
 */

// How many elements live inside the object before it reaches for the heap.
// `nbatch + ndim + 1` with ndim <= 3; 8 covers every shape this tree builds
// and keeps the object at 32 bytes on the narrow arm.
#ifndef FF_INDEX_ARRAY_INLINE
#  define FF_INDEX_ARRAY_INLINE 8
#endif

template <class offset_t>
class IndexArray
{
public:
    // A null `src` yields a null array and copies nothing. An absent
    // optional operand is passed as a null-data descriptor whose stride
    // array is never read, and the call sites that have one spelled it
    // `wgt ? copy_if_needed<offset_t*>(stride_wgt, n) : nullptr`.
    IndexArray(const int64_t * src, size_t numel)
        : _ptr(nullptr), _heap(nullptr)
    {
        if (!src) return;
        offset_t * dst;
        if (numel <= FF_INDEX_ARRAY_INLINE) {
            dst = _inln;
        } else {
            dst = _heap = hostNew<offset_t>(numel);
        }
        for (size_t i = 0; i < numel; ++i)
            dst[i] = static_cast<offset_t>(src[i]);
        _ptr = dst;
    }

    ~IndexArray() { if (_heap) hostDelete<offset_t>(_heap); }

    operator const offset_t * () const { return _ptr; }
    const offset_t * get()       const { return _ptr; }

private:
    IndexArray(const IndexArray &);
    IndexArray & operator=(const IndexArray &);

    const offset_t * _ptr;
    offset_t       * _heap;
    offset_t         _inln[FF_INDEX_ARRAY_INLINE];
};

// Wide arm: there is nothing to narrow to, so borrow the caller's array.
// Same no-op the `_copy_if_needed<T*, const T*>` specialisation provides, and
// the reason `FF_INDEX32=0` costs nothing here either.
template <>
class IndexArray<int64_t>
{
public:
    IndexArray(const int64_t * src, size_t /* numel */) : _ptr(src) {}

    operator const int64_t * () const { return _ptr; }
    const int64_t * get()       const { return _ptr; }

private:
    IndexArray(const IndexArray &);
    IndexArray & operator=(const IndexArray &);

    const int64_t * _ptr;
};

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
