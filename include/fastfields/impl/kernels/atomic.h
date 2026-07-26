/***********************************************************************
 * CUDA portion copied from PyTorch/ATen
 * https://github.com/pytorch/pytorch/blob/master/LICENSE
 **********************************************************************/

#ifndef FF_ATOMIC
#define FF_ATOMIC
#include "cuda_switch.h"

/***********************************************************************
 *                              CPU
 **********************************************************************/
#ifndef __CUDACC__
#include <atomic>

FF_NAMESPACE_BEGIN(FF)

template <typename T>
class has_fetch_add
{
    // This class helps us check if atomic += is defined for
    // floating point types.
    // https://stackoverflow.com/questions/257288
    typedef char one;
    struct two { char x[2]; };

    template <typename C> static one test( decltype(&C::fetch_add) ) ;
    template <typename C> static two test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

template <typename scalar_t>
struct has_atomic_add
{
    // A lock-free CAS atomic add (AtomicAdd<true>, below) works for ANY type
    // whose std::atomic is lock-free — including float/double: std::atomic<float>
    // ::fetch_add is C++20-only, but compare_exchange_weak has been available
    // since C++11. Gate on lock-freedom, NOT fetch_add existence, so float/double
    // get a VALID host atomic at C++17 instead of the non-atomic AtomicAdd<false>
    // (which was reached only because fetch_add is absent pre-C++20). This lets
    // scatter kernels (pushpull push/count, reg_* accumulate) parallelise flat
    // over all elements instead of falling back to a batch-serial structure.
    enum { value = has_fetch_add<std::atomic<scalar_t> >::value
                || std::atomic<scalar_t>::is_always_lock_free };
};

template <bool has_atom=false>
struct AtomicAdd {
    template <typename scalar_t>
    static inline scalar_t atomicAdd(scalar_t * address, scalar_t val) {
        val += *address;
        *address = val;
        return val;
    }

    template <typename scalar_t>
    static inline void atomicAddNoReturn(scalar_t * address, scalar_t val) {
        *address += val;
    }
};

template <>
struct AtomicAdd<true> {
    // Implemented in c++ 11+ for integral types
    // Implemented in c++ 20+ for floating types

    // Lock-free add via compare-exchange. std::atomic<float>::fetch_add only
    // exists in C++20, so use a CAS loop (available since C++11) on the object
    // viewed in place as an atomic. Selected whenever std::atomic<scalar_t> is
    // lock-free (integers AND float/double) — see has_atomic_add above. Viewing
    // the object in place as a std::atomic is the C++17 stand-in for C++20's
    // std::atomic_ref (valid where the atomic is lock-free and same-layout,
    // which holds for the arithmetic types used here).
    template <typename scalar_t>
    static inline scalar_t atomicAdd(scalar_t * address, scalar_t val) {
        std::atomic<scalar_t> * aptr =
            reinterpret_cast<std::atomic<scalar_t> *>(address);
        scalar_t old = aptr->load(std::memory_order_relaxed);
        while (!aptr->compare_exchange_weak(old, old + val,
                                            std::memory_order_relaxed)) {}
        return old + val;   // new value, matching AtomicAdd<false>::atomicAdd
    }

    template <typename scalar_t>
    static inline void atomicAddNoReturn(scalar_t * address, scalar_t val) {
        atomicAdd(address, val);
    }
};

template <typename T>
static inline T anyAtomicAdd(T *address, T val) {
    return AtomicAdd<has_atomic_add<T>::value>::atomicAdd(address, val);
}

template <typename T>
static inline void anyAtomicAddNoReturn(T *address, T val) {
    return AtomicAdd<has_atomic_add<T>::value>::atomicAddNoReturn(address, val);
}

FF_NAMESPACE_END(FF)

/***********************************************************************
 *                              CUDA
 **********************************************************************/
#else

template <typename T>
struct AtomicFPOp;

template <>
struct AtomicFPOp<double> {
  template <typename func_t>
  inline CUDEV double operator() (double * address, double val, const func_t& func) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, func(val, assumed));
      // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
  }
};

#define ATOMIC_INTEGER_IMPL(NAME)                                                                                      \
template <typename T, size_t n>                                                                                        \
struct Atomic##NAME##IntegerImpl;                                                                                      \
                                                                                                                       \
template<typename T>                                                                                                   \
struct Atomic##NAME##IntegerImpl<T, 1> {                                                                               \
  template <typename func_t>                                                                                           \
  inline CUDEV void operator()(T *address, T val, const func_t& func) {                                           \
    size_t offset = (size_t)address & 3;                                                                               \
    unsigned int * address_as_ui = (unsigned int *)((char *)address - offset);                                         \
    unsigned int old = *address_as_ui;                                                                                 \
    unsigned int shift = offset * 8;                                                                                   \
    unsigned int old_byte;                                                                                             \
    unsigned int newval;                                                                                               \
    unsigned int assumed;                                                                                              \
                                                                                                                       \
    do {                                                                                                               \
      assumed = old;                                                                                                   \
      old_byte = (old >> shift) & 0xff;                                                                                \
      newval = static_cast<char>(func(val, static_cast<T>(old_byte)));                                                 \
      newval = (old & ~(0x000000ff << shift)) | (newval << shift);                                                     \
      old = atomicCAS(address_as_ui, assumed, newval);                                                                 \
    } while (assumed != old);                                                                                          \
  }                                                                                                                    \
};                                                                                                                     \
                                                                                                                       \
template<typename T>                                                                                                   \
struct Atomic##NAME##IntegerImpl<T, 2> {                                                                               \
  template <typename func_t>                                                                                           \
  inline CUDEV void operator()(T *address, T val, const func_t& func) {                                           \
    size_t offset = (size_t)address & 2;                                                                               \
    unsigned int * address_as_ui = (unsigned int *)((char *)address - offset);                                         \
    bool is_32_align = offset;                                                                                         \
    unsigned int old = *address_as_ui;                                                                                 \
    unsigned int old_bytes;                                                                                            \
    unsigned int newval;                                                                                               \
    unsigned int assumed;                                                                                              \
                                                                                                                       \
    do {                                                                                                               \
      assumed = old;                                                                                                   \
      old_bytes = is_32_align ? old >> 16 : old & 0xffff;                                                              \
      newval = static_cast<unsigned short>(func(val, static_cast<T>(old_bytes)));                                      \
      newval = is_32_align ? (old & 0xffff) | (newval << 16) : (old & 0xffff0000) | newval;                            \
      old = atomicCAS(address_as_ui, assumed, newval);                                                                 \
    } while (assumed != old);                                                                                          \
  }                                                                                                                    \
};                                                                                                                     \
                                                                                                                       \
template<typename T>                                                                                                   \
struct Atomic##NAME##IntegerImpl<T, 4> {                                                                               \
  template <typename func_t>                                                                                           \
  inline CUDEV void operator()(T *address, T val, const func_t& func) {                                           \
    unsigned int * address_as_ui = (unsigned int *) (address);                                                         \
    unsigned int old = *address_as_ui;                                                                                 \
    unsigned int newval;                                                                                               \
    unsigned int assumed;                                                                                              \
                                                                                                                       \
    do {                                                                                                               \
      assumed = old;                                                                                                   \
      newval = static_cast<unsigned int>(func(val, static_cast<T>(old)));                                              \
      old = atomicCAS(address_as_ui, assumed, newval);                                                                 \
    } while (assumed != old);                                                                                          \
  }                                                                                                                    \
};                                                                                                                     \
                                                                                                                       \
template<typename T>                                                                                                   \
struct Atomic##NAME##IntegerImpl<T, 8> {                                                                               \
  template <typename func_t>                                                                                           \
  inline CUDEV void operator()(T *address, T val, const func_t& func) {                                           \
    unsigned long long * address_as_ui = (unsigned long long *) (address);                                             \
    unsigned long long old = *address_as_ui;                                                                           \
    unsigned long long newval;                                                                                         \
    unsigned long long assumed;                                                                                        \
                                                                                                                       \
    do {                                                                                                               \
      assumed = old;                                                                                                   \
      newval = static_cast<unsigned long>(func(val, static_cast<T>(old)));                                             \
      old = atomicCAS(address_as_ui, assumed, newval);                                                                 \
    } while (assumed != old);                                                                                          \
  }                                                                                                                    \
};


# define GPU_ATOMIC_INTEGER(NAME, OP, DTYPE)                                                                           \
static inline CUDEV void gpuAtomic##NAME(DTYPE *address, DTYPE val) {                                             \
Atomic##NAME##IntegerImpl<DTYPE, sizeof(DTYPE)>()(address,                                                             \
                                                      val,                                                             \
                                                      [](DTYPE a, DTYPE b) {                                           \
                                                          return OP;                                                   \
                                                      });                                                              \
}                                                                                                                      \

ATOMIC_INTEGER_IMPL(Add)

/*
// Don't instantiate gpuAtomicAdd with the macro as it seems non-standard (see int32, int64)
static inline CUDEV void gpuAtomicAdd(char *address, char val) {
  AtomicAddIntegerImpl<char, sizeof(char)>()(address,
                                                   val,
                                                   [](char a, char b) {
                                                      return a + b;
                                                   });
}

static inline  CUDEV void gpuAtomicAdd(signed char *address, signed char val) {
  AtomicAddIntegerImpl<signed char, sizeof(signed char)>()(address,
                                                 val,
                                                 [](signed char a, signed char b) {
                                                   return a + b;
                                                 });
}

static inline  CUDEV void gpuAtomicAdd(short *address, short val) {
  AtomicAddIntegerImpl<short, sizeof(short)>()(address,
                                                   val,
                                                   [](short a, short b) {
                                                     return a + b;
                                                   });
}

static inline CUDEV int gpuAtomicAdd(int *address, int val) {
  return atomicAdd(address, val);
}

static inline CUDEV void gpuAtomicAdd(long *address, long val) {
#if defined(USE_ROCM)
  __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
#else
  AtomicAddIntegerImpl<long, sizeof(long)>()(address,
                                                   val,
                                                   [](long a, long b) {
                                                      return a + b;
                                                   });
#endif
}

static inline CUDEV void gpuAtomicAdd(bool *address, bool val) {
  *address = address && val;
}
*/

// from CUDA C Programmic Guide.
// Native double atomicAdd exists for compute capability >= 6.0, so only
// provide this fallback for older device architectures (defining it for
// sm_60+ collides with the built-in "atomicAdd(double*, double)").
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
static inline CUDEV double atomicAdd(double* address, double val)
#if defined(__clang__) && defined(__CUDA__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgcc-compat"
    __attribute__((enable_if(true, "")))
#pragma GCC diagnostic pop
#endif // defined(__clang__) && defined(__CUDA__)
{

  return AtomicFPOp<double>()(address, val,
                              [](double val, unsigned long long int assumed) {
                                return __double_as_longlong(val + __longlong_as_double(assumed));
                              });
}
#endif // __CUDA_ARCH__ < 600

static inline CUDEV double gpuAtomicAdd(double *address, double val) {
  return atomicAdd(address, val);
}

static inline CUDEV float gpuAtomicAdd(float *address, float val) {
  return atomicAdd(address, val);
}

/*
template<typename T>
static inline CUDEV void gpuAtomicAdd(complex<T> *address, complex<T> val) {
  gpuAtomicAdd(&address->real_, val.real_);
  gpuAtomicAdd(&address->imag_, val.imag_);
}
*/

/* Note [gpuAtomicAdd vs atomicAdd]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Some extensions such as torchvision call atomicAdd()
 * directly and require non-library provided data type support. Only for these, we
 * continue to provide atomicAdd overloads.
 */

/*
static inline CUDEV void atomicAdd(char *address, char val) {
  gpuAtomicAdd(address, val);
}

static inline  CUDEV void atomicAdd(signed char *address, signed char val) {
  gpuAtomicAdd(address, val);
}

static inline  CUDEV void atomicAdd(short *address, short val) {
  gpuAtomicAdd(address, val);
}

static inline CUDEV void atomicAdd(long *address, long val) {
  gpuAtomicAdd(address, val);
}

static inline CUDEV void atomicAdd(bool *address, bool val) {
  gpuAtomicAdd(address, val);
}
*/

/* Note [explicitly non-returning atomics]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * AMD's MI100 (gfx908) provides an optimized fp32 atomicAdd, exposed via atomicAddNoRet().
 * Due to compiler limitations, callers must opt-in to guarantee the optimized instruction.
 * This non-returning atomicAddNoRet cannot be used to implement the returning atomicAdd,
 * therefore we need a new API 'gpuAtomicAddNoReturn'.
 */
/*
template<typename T>
static inline CUDEV void gpuAtomicAddNoReturn(complex<T> *address, complex<T> val) { gpuAtomicAdd(address, val); }
static inline CUDEV void gpuAtomicAddNoReturn(char *address, char val) { gpuAtomicAdd(address, val); }
static inline CUDEV void gpuAtomicAddNoReturn(signed char *address, signed char val) { gpuAtomicAdd(address, val); }
static inline CUDEV void gpuAtomicAddNoReturn(short *address, short val) { gpuAtomicAdd(address, val); }
static inline CUDEV void gpuAtomicAddNoReturn(int *address, int val) { gpuAtomicAdd(address, val); }
static inline CUDEV void gpuAtomicAddNoReturn(long *address, long val) { gpuAtomicAdd(address, val); }
static inline CUDEV void gpuAtomicAddNoReturn(bool *address, bool val) { gpuAtomicAdd(address, val); }
*/
static inline CUDEV void gpuAtomicAddNoReturn(double *address, double val) { gpuAtomicAdd(address, val); }

/* Special case fp32 atomic. */
#if defined(USE_ROCM)
static inline CUDEV void gpuAtomicAddNoReturn(float *address, float val) { atomicAddNoRet(address, val); }
#else
static inline CUDEV void gpuAtomicAddNoReturn(float *address, float val) { gpuAtomicAdd(address, val); }
#endif

namespace ff {

template <typename T>
static inline CUDEV T anyAtomicAdd(T *address, T val) {
    return gpuAtomicAdd(address, val);
}

template <typename T>
static inline CUDEV void anyAtomicAddNoReturn(T *address, T val) {
    return gpuAtomicAddNoReturn(address, val);
}

} // namespace ff

#endif // __CUDA__

#endif // FF_ATOMIC
