#ifndef FF_BOUNDS
#define FF_BOUNDS
#include "cuda_switch.h"
#include "atomic.h"
#include "utils.h"
#include "meta.h"

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                             INDEXING
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FF_NAMESPACE_BEGIN(FF)

FF_NAMESPACE_BEGIN(bound)
enum class type : int8_t {
  Dynamic   = -1, // Used to turn-off static implementations in templated classes
  Zero      = 0,  // Zero outside of the FOV
  Replicate = 1,  // Replicate last inbound value = clip coordinates
  DCT1      = 2,  // Symmetric w.r.t. center of the last inbound voxel
  DCT2      = 3,  // Symmetric w.r.t. edge of the last inbound voxel (= Neumann)
  DST1      = 4,  // Antisymmetric w.r.t. center of the last inbound voxel
  DST2      = 5,  // Antisymmetric w.r.t. edge of the last inbound voxel (= Dirichlet)
  DFT       = 6,  // Circular / Wrap around the FOV
  NoCheck   = 7   // /!\ Checks disabled: assume coordinates are inbound
};

// Boundary condition of the transpose of a first-difference operator.
//
// The adjoint of a derivative flips the parity of the extension: an even
// (DCT / Neumann) reflection becomes an odd (DST / Dirichlet) one and vice
// versa, while a periodic (DFT) extension is self-adjoint. Used by the
// linear-elastic (Lamé) flow regulariser so its cross-channel coupling block
// is a genuine transpose of the mirror block (self-adjoint operator, required
// by CG / relaxation solvers). DCT1<->DST1 and DCT2<->DST2 swap; every other
// condition is its own transpose.
//
// This yields an exactly self-adjoint operator for the half-sample-symmetric
// family (DCT2<->DST2, both reflect_N) and for DFT / Zero / Replicate / NoCheck.
// It is NOT exact for the whole-sample-symmetric family (DCT1 reflect_{N-1},
// DST1 reflect_{N+1}): forward and adjoint use different reflection centres, so
// a single companion-boundary read cannot reproduce D^T there. The Lamé
// operator is therefore not SPD under DCT1/DST1 — a documented limitation
// (flow regularisation uses DCT2/Neumann or DFT in practice). See fastfields-
// lib#26.
// __host__ __device__: used both as a compile-time template argument and, for a
// `type::Dynamic` axis, evaluated at run time inside a device-side constructor.
CUHOSTDEV constexpr inline type transpose(type b)
{
  return b == type::DCT1 ? type::DST1 :
         b == type::DST1 ? type::DCT1 :
         b == type::DCT2 ? type::DST2 :
         b == type::DST2 ? type::DCT2 :
         b;
}
// Runtime companion of the compile-time `Bound<B...>` pack.
//
// Every axis whose compile-time boundary condition is `type::Dynamic` reads its
// actual condition from this vector at run time; axes instantiated with a real
// `type` ignore it entirely (the corresponding `bound::dyn<B>` specialisation is
// stateless and its constructor discards the argument). Trivially copyable, so
// it can be passed by value all the way into a `__global__` kernel.
//
// Templated on `MaxNDim` (default 3) rather than hard-coding the array size:
// every pushpull/regulariser kernel in the library today is written for
// ndim in {1,2,3} (see kernels/pushpull/{1d,2d,3d}.h -- `nd.h` exists but is
// currently unreachable dead code), so 3 is today's *ceiling*, not an
// architectural limit of `BoundVec` itself. A future n>3 kernel can
// instantiate `BoundVecN<N>` for its own `N` without touching this struct;
// `BoundVec` (used everywhere today) is just the `N=3` alias below.
template <int MaxNDim = 3>
struct BoundVecN {
  static const int max_ndim = MaxNDim;
  int8_t b[max_ndim];

  inline CUHOSTDEV BoundVecN()
  { for (int d = 0; d < max_ndim; ++d) b[d] = static_cast<int8_t>(type::Zero); }

  // Isotropic: the same condition on every axis (what the public ABI exposes).
  explicit inline CUHOSTDEV BoundVecN(type v)
  { for (int d = 0; d < max_ndim; ++d) b[d] = static_cast<int8_t>(v); }

  // Anisotropic: one condition per axis, `ndim <= max_ndim` of them meaningful.
  // Axes `d >= ndim` are padded with `type::Zero` (rather than left
  // uninitialised) purely so every element of the trivially-copyable struct
  // has a deterministic, valid `bound::type` value -- no kernel ever reads a
  // padding axis (every dispatch layer loops exactly `ndim` times, never
  // `max_ndim`), so the specific pad value is inert; `Zero` is used because
  // it is this enum's own semantic default/identity value, matching `type()`
  // default-constructing to 0.
  inline CUHOSTDEV BoundVecN(const type * v, int ndim)
  {
    for (int d = 0; d < max_ndim; ++d)
      b[d] = static_cast<int8_t>(d < ndim ? v[d] : type::Zero);
  }

  inline CUHOSTDEV type operator[] (int d) const
  { return static_cast<type>(b[d]); }
};

using BoundVec = BoundVecN<3>;

FF_NAMESPACE_END(bound)

using bound_t = bound::type;
template <bound_t...  B> using Bound = meta::Tuple<bound_t,  B...>;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  STATIC / DYNAMIC BOUND BUILD POLICY
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Every boundary condition used to be a *compile-time* template parameter, so
// the dispatch layers instantiated the whole (ndim x bound x dtype x offset_t)
// matrix of every kernel. That is fine for the CPU backend but makes nvcc's
// `ptxas` blow past 16 GB of RAM on the regularisers alone.
//
// `bound::type::Dynamic` selects a *runtime* implementation instead: a single
// instantiation whose boundary condition is read from a `bound::BoundVec` at
// run time. Which conditions keep a dedicated (faster) static instantiation and
// which share the Dynamic one is a **build-time** choice, for whoever compiles
// the library:
//
//     -DFF_STATIC_BOUNDS=0                 // everything runtime (smallest build)
//     -DFF_STATIC_BOUNDS=1                 // everything static  (fastest code)
//     -DFF_STATIC_BOUNDS=0 \
//     -DFF_STATIC_BOUND_DCT2=1 \
//     -DFF_STATIC_BOUND_DFT=1              // static fast path for two of them
//
// The per-condition macros are FF_STATIC_BOUND_{ZERO,REPLICATE,DCT1,DCT2,DST1,
// DST2,DFT,NOCHECK}; each defaults to FF_STATIC_BOUNDS. Behaviour is identical
// either way -- only code size, compile cost and per-voxel speed change.
//
// Dispatch layers must write `FF_BOUND_DCT2` instead of `bound::type::DCT2`
// when choosing the template argument, and pass the *runtime* condition along
// in a `bound::BoundVec` so the Dynamic instantiations can recover it.

#ifndef FF_STATIC_BOUNDS
#  define FF_STATIC_BOUNDS 1
#endif
#ifndef FF_STATIC_BOUND_ZERO
#  define FF_STATIC_BOUND_ZERO      FF_STATIC_BOUNDS
#endif
#ifndef FF_STATIC_BOUND_REPLICATE
#  define FF_STATIC_BOUND_REPLICATE FF_STATIC_BOUNDS
#endif
#ifndef FF_STATIC_BOUND_DCT1
#  define FF_STATIC_BOUND_DCT1      FF_STATIC_BOUNDS
#endif
#ifndef FF_STATIC_BOUND_DCT2
#  define FF_STATIC_BOUND_DCT2      FF_STATIC_BOUNDS
#endif
#ifndef FF_STATIC_BOUND_DST1
#  define FF_STATIC_BOUND_DST1      FF_STATIC_BOUNDS
#endif
#ifndef FF_STATIC_BOUND_DST2
#  define FF_STATIC_BOUND_DST2      FF_STATIC_BOUNDS
#endif
#ifndef FF_STATIC_BOUND_DFT
#  define FF_STATIC_BOUND_DFT       FF_STATIC_BOUNDS
#endif
#ifndef FF_STATIC_BOUND_NOCHECK
#  define FF_STATIC_BOUND_NOCHECK   FF_STATIC_BOUNDS
#endif

#define FF_BOUND_IF_1(NAME)   ::FF::bound::type::NAME
#define FF_BOUND_IF_0(NAME)   ::FF::bound::type::Dynamic
#define FF_BOUND_CAT_(A, B)   A##B
#define FF_BOUND_CAT(A, B)    FF_BOUND_CAT_(A, B)
#define FF_BOUND_SEL(FLAG, NAME) FF_BOUND_CAT(FF_BOUND_IF_, FLAG)(NAME)

// Template argument to use for each boundary condition:
// the condition itself when it is statically compiled, `Dynamic` otherwise.
#define FF_BOUND_ZERO       FF_BOUND_SEL(FF_STATIC_BOUND_ZERO,      Zero)
#define FF_BOUND_REPLICATE  FF_BOUND_SEL(FF_STATIC_BOUND_REPLICATE, Replicate)
#define FF_BOUND_DCT1       FF_BOUND_SEL(FF_STATIC_BOUND_DCT1,      DCT1)
#define FF_BOUND_DCT2       FF_BOUND_SEL(FF_STATIC_BOUND_DCT2,      DCT2)
#define FF_BOUND_DST1       FF_BOUND_SEL(FF_STATIC_BOUND_DST1,      DST1)
#define FF_BOUND_DST2       FF_BOUND_SEL(FF_STATIC_BOUND_DST2,      DST2)
#define FF_BOUND_DFT        FF_BOUND_SEL(FF_STATIC_BOUND_DFT,       DFT)
#define FF_BOUND_NOCHECK    FF_BOUND_SEL(FF_STATIC_BOUND_NOCHECK,   NoCheck)

FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(bound)

using FF::bound::type;
using FF::bound::transpose;
using FF::bound::BoundVec;

// These function act on floating point coordinates and simply
// apply the periodicity and reflection conditions of each boundary.
//
// This means that some of them output coordinates outside of the
// array support [0, n-1]. Coordinates would typically be converted to
// integer (round/floor/ceil), then clamped between [0, n-1], before
// being used to index into an array.
template <typename offset_t, bool is_float = is_floating_point<offset_t>::value >
struct _index
{
    template <typename size_t>
    static inline CUDEV
    offset_t inbounds(offset_t coord, size_t size)
    {
      return coord;
    }

    // Periodic (0, N-1)*2 + Reflect (0, N-1)
    // Support length = N-1
    // -> Boundary condition of a DCT-I
    template <typename size_t>
    static inline CUDEV
    offset_t reflect_Nminus1(offset_t coord, size_t size)
    {
      if (size == 1) return static_cast<offset_t>(0);
      size -= 1;
      size_t size_twice = size*2;
      coord = mod(abs(coord) % size_twice);                // period
      coord = coord > size ? size_twice - coord : coord;   // reflect
      return coord;
    }

    // Periodic (-1, N)*2 + Reflect (1, N)
    // Support length = N+1
    // -> Boundary condition of a DST-I
    template <typename size_t>
    static inline CUDEV
    offset_t reflect_Nplus1(offset_t coord, size_t size)
    {
      if (size == 1) static_cast<offset_t>(0);
      size += 1;
      size_t size_twice = size*2;
      coord += 1;
      coord = mod(abs(coord) % size_twice);                // period
      coord = coord > size ? size_twice - coord : coord;   // reflect
      coord -= 1;
      return coord;
    }

    // Periodic (-1/2, N-1/2)*2 + Reflect (-1/2, N-1/2)
    // Support length = N
    // -> Boundary condition of a DCT-II or DST-II
    template <typename size_t>
    static inline CUDEV
    offset_t reflect_N(offset_t coord, size_t size)
    {
      if (size == 1) static_cast<offset_t>(0);
      size_t size_twice = size*2;
      coord += 0.5;
      coord = mod(abs(coord) % size_twice);                // period
      coord = coord > size ? size_twice - coord : coord;   // reflect
      coord -= 0.5;
      return coord;
    }

    // Periodic (-1/2, N-1/2)
    // Support length = N
    // -> Boundary condition of a DFT
    template <typename size_t>
    static inline CUDEV
    offset_t circular(offset_t coord, size_t size)
    {
      if (size == 1) static_cast<offset_t>(0);
      coord += 0.5;
      coord = mod(coord % size);
      coord -= 0.5;
      return coord;
    }

    // Clamped to (-1/2, N-1/2)
    // Support length = N
    template <typename size_t>
    static inline CUDEV
    offset_t replicate(offset_t coord, size_t size)
    {
      coord = coord <= -0.5     ? static_cast<offset_t>(-0.5)
            : coord >= size-0.5 ? static_cast<offset_t>(size - 0.5) : coord;
      return coord;
    }
};

// These functions are specialized for integral coordinates
template <typename offset_t>
struct _index<offset_t, false>
{
    template <typename size_t>
    static inline CUDEV
    offset_t inbounds(offset_t coord, size_t size)
    {
      return coord;
    }

    // Boundary condition of a DCT-I (periodicity: (n-1)*2)
    // Indices are reflected about the centre of the border elements:
    //    -1 --> 1
    //     n --> n-2
    template <typename size_t>
    static inline CUDEV
    offset_t reflect_Nminus1(offset_t coord, size_t size)
    {
      if (size == 1) return 0;
      size_t size_twice = (size-1)*2;
      coord = abs(coord);
      coord = coord % size_twice;
      coord = coord >= size ? size_twice - coord : coord;
      return coord;
    }

    // Boundary condition of a DST-I (periodicity: (n+1)*2)
    // Indices are reflected about the centre of the first out-of-bound
    // element:
    //    -1 --> undefined [0]
    //    -2 --> 0
    //     n --> undefined [n-1]
    //   n+1 --> n-1
    template <typename size_t>
    static inline CUDEV
    offset_t reflect_Nplus1(offset_t coord, size_t size)
    {
      if (size == 1) return static_cast<offset_t>(0);
      size_t size_twice = (size+1)*2;
      coord = coord == -1 ? static_cast<offset_t>(0) : coord < 0 ? -coord-2 : coord;
      coord = coord % size_twice;
      coord = coord == size ? static_cast<offset_t>(size-1)
            : coord > size  ? size_twice-coord-2 : coord;
      return coord;
    }

    // Boundary condition of a DCT/DST-II (periodicity: n*2)
    // Indices are reflected about the edge of the border elements:
    //    -1 --> 0
    //     n --> n-1
    template <typename size_t>
    static inline CUDEV
    offset_t reflect_N(offset_t coord, size_t size)
    {
      size_t size_twice = size*2;
      coord = coord < 0 ? size_twice - ((-coord-1) % size_twice) - 1
                        : coord % size_twice;
      coord = coord >= size ? size_twice - coord - 1 : coord;
      return coord;
    }

    // Boundary condition of a DFT (periodicity: n)
    // Indices wrap about the edges:
    //    -1 --> n-1
    //     n --> 0
    template <typename size_t>
    static inline CUDEV
    offset_t circular(offset_t coord, size_t size)
    {
      coord = coord < 0 ? (size + coord%size) % size : coord % size;
      return coord;
    }

    // Replicate edge values:
    //    -1 --> 0
    //    -2 --> 0
    //     n --> n-1
    //   n+1 --> n-1
    template <typename size_t>
    static inline CUDEV
    offset_t replicate(offset_t coord, size_t size)
    {
      coord = coord <= 0    ? static_cast<offset_t>(0)
            : coord >= size ? static_cast<offset_t>(size - 1) : coord;
      return coord;
    }
};


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                          SIGN MODIFICATION
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FF_NAMESPACE_BEGIN(_sign)

template <typename offset_t, typename size_t>
inline CUDEV int8_t inbounds(offset_t coord, size_t size) {
  return coord < 0 || coord >= size ? 0 : 1;
}

// Boundary condition of a DCT/DFT
// No sign modification based on coordinates
template <typename offset_t, typename size_t>
constexpr inline CUDEV int8_t constant(offset_t coord, size_t size) {
  return static_cast<int8_t>(1);
}

// Boundary condition of a DST-I
// Periodic sign change based on coordinates
template <typename offset_t, typename size_t>
inline CUDEV int8_t periodic1(offset_t coord, size_t size) {
  if (size == 1) return 1;
  size_t size_twice = (size+1)*2;
  coord = coord < 0 ? size - coord - 1 : coord;
  coord = coord % size_twice;
  if (coord % (size+1) == size)   return  static_cast<int8_t>(0);
  else if ((coord/(size+1)) % 2)  return  static_cast<int8_t>(-1);
  else                            return  static_cast<int8_t>(1);
}

// Boundary condition of a DST-II
// Periodic sign change based on coordinates
template <typename offset_t, typename size_t>
inline CUDEV int8_t periodic2(offset_t coord, size_t size) {
  coord = (coord < 0 ? size - coord - 1 : coord);
  return static_cast<int8_t>((coord/size) % 2 ? -1 : 1);
}

FF_NAMESPACE_END(_sign)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                                BOUND
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Check if coordinates within bounds
template <typename size_t>
inline CUDEV
bool inbounds(size_t coord, size_t size)
{
  return coord >= 0 && coord < size;
}

template <typename scalar_t, typename size_t>
inline CUDEV
bool inbounds(scalar_t coord, size_t size, scalar_t tol)
{
  return coord >= -tol && coord < (scalar_t)(size-1)+tol;
}

template <typename scalar_t, typename offset_t>
inline CUDEV
scalar_t get(const scalar_t * ptr, offset_t offset, int8_t sign)
{
  if (sign == -1)  return -ptr[offset];
  else if (sign)   return  ptr[offset];
  else             return  static_cast<scalar_t>(0);
}

template <typename scalar_t, typename offset_t>
inline CUDEV
scalar_t get(const scalar_t * ptr, offset_t offset)
{
  return ptr[offset];
}

template <typename val_t, typename scalar_t, typename offset_t>
inline CUDEV
scalar_t cget(const scalar_t * ptr, offset_t offset, int8_t sign)
{
  return static_cast<val_t>(get(ptr, offset, sign));
}

template <typename val_t, typename scalar_t, typename offset_t>
inline CUDEV
scalar_t cget(const scalar_t * ptr, offset_t offset)
{
  return static_cast<val_t>(get(ptr, offset));
}

template <typename scalar_t, typename offset_t, typename val_t>
inline CUDEV
void add(scalar_t *ptr, offset_t offset, val_t val, int8_t sign)
{
  scalar_t cval = static_cast<scalar_t>(val);
  if (sign == -1)  anyAtomicAdd(ptr + offset, -cval);
  else if (sign)   anyAtomicAdd(ptr + offset,  cval);
}

template <typename scalar_t, typename offset_t, typename val_t>
inline CUDEV
void add(scalar_t *ptr, offset_t offset, val_t val)
{
  anyAtomicAdd(ptr + offset,  static_cast<scalar_t>(val));
}

template <type B> struct utils {
    template <typename offset_t, typename size_t = offset_t>
    static inline CUDEV offset_t index(offset_t coord, size_t size)
    { return _index<offset_t>::inbounds(coord, size); }
    template <typename offset_t, typename size_t = offset_t>
    static inline CUDEV int8_t sign(offset_t coord, size_t size)
    { return _sign::inbounds(coord, size); }
};

template <> struct utils<type::Replicate> {
    template <typename offset_t, typename size_t = offset_t>
    static inline CUDEV offset_t index(offset_t coord, size_t size)
    { return _index<offset_t>::replicate(coord, size); }
    template <typename offset_t, typename size_t = offset_t>
    static constexpr inline CUDEV int8_t sign(offset_t coord, size_t size)
    { return _sign::constant(coord, size); }
};

template <> struct utils<type::DCT1> {
    template <typename offset_t, typename size_t = offset_t>
    static inline CUDEV offset_t index(offset_t coord, size_t size)
    { return _index<offset_t>::reflect_Nminus1(coord, size); }
    template <typename offset_t, typename size_t = offset_t>
    static constexpr inline CUDEV int8_t sign(offset_t coord, size_t size)
    { return _sign::constant(coord, size); }
};

template <> struct utils<type::DCT2> {
    template <typename offset_t, typename size_t = offset_t>
    static inline CUDEV offset_t index(offset_t coord, size_t size)
    { return _index<offset_t>::reflect_N(coord, size); }
    template <typename offset_t, typename size_t = offset_t>
    static constexpr inline CUDEV int8_t sign(offset_t coord, size_t size)
    { return _sign::constant(coord, size); }
};

template <> struct utils<type::DST1> {
    template <typename offset_t, typename size_t = offset_t>
    static inline CUDEV offset_t index(offset_t coord, size_t size)
    { return _index<offset_t>::reflect_Nplus1(coord, size); }
    template <typename offset_t, typename size_t = offset_t>
    static inline CUDEV int8_t sign(offset_t coord, size_t size)
    { return _sign::periodic1(coord, size); }
};

template <> struct utils<type::DST2> {
    template <typename offset_t, typename size_t = offset_t>
    static inline CUDEV offset_t index(offset_t coord, size_t size)
    { return _index<offset_t>::reflect_N(coord, size); }
    template <typename offset_t, typename size_t = offset_t>
    static inline CUDEV int8_t sign(offset_t coord, size_t size)
    { return _sign::periodic2(coord, size); }
};

template <> struct utils<type::DFT> {
    template <typename offset_t, typename size_t = offset_t>
    static inline CUDEV offset_t index(offset_t coord, size_t size)
    { return _index<offset_t>::circular(coord, size); }
    template <typename offset_t, typename size_t = offset_t>
    static constexpr inline CUDEV int8_t sign(offset_t coord, size_t size)
    { return _sign::constant(coord, size); }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                  STATIC / DYNAMIC BOUND SELECTOR
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// `dyn<B>` is the stateful counterpart of `utils<B>`: same `index` / `sign`
// interface, but as (const) member functions of an object that may carry the
// boundary condition at run time.
//
//  - for a real `B`, `dyn<B>` is an empty struct that forwards to `utils<B>`;
//    the compiler folds it away completely (zero cost, identical codegen);
//  - for `type::Dynamic`, `dyn` holds the condition as a data member and
//    branches on it per call.
//
// Kernels therefore hold `dyn<BX> bound_utils_x;` members instead of
// `using bound_utils_x = utils<BX>;` aliases, and are constructed from a
// `BoundVec`. A single Dynamic instantiation replaces the eight static ones --
// which is what keeps nvcc's `ptxas` inside a sane memory budget.

template <type B> struct dyn
{
    inline CUDEV dyn() {}
    explicit inline CUDEV dyn(type) {}       // runtime value: not needed, ignored

    inline CUDEV type value() const { return B; }

    template <typename offset_t, typename size_t = offset_t>
    inline CUDEV offset_t index(offset_t coord, size_t size) const
    { return utils<B>::template index<offset_t, size_t>(coord, size); }

    template <typename offset_t, typename size_t = offset_t>
    inline CUDEV int8_t sign(offset_t coord, size_t size) const
    { return utils<B>::template sign<offset_t, size_t>(coord, size); }
};

template <> struct dyn<type::Dynamic>
{
    type bnd;

    inline CUDEV dyn() : bnd(type::Zero) {}
    explicit inline CUDEV dyn(type b) : bnd(b) {}

    inline CUDEV type value() const { return bnd; }

    // Direct switches (rather than the `index_fn` / `sign_fn` function-pointer
    // helpers below): an indirect call cannot be inlined and is expensive on
    // the GPU, whereas a switch over a warp-uniform value is close to free.
    template <typename offset_t, typename size_t = offset_t>
    inline CUDEV offset_t index(offset_t coord, size_t size) const
    {
      switch (bnd) {
        case type::Replicate:  return _index<offset_t>::replicate(coord, size);
        case type::DCT1:       return _index<offset_t>::reflect_Nminus1(coord, size);
        case type::DCT2:       return _index<offset_t>::reflect_N(coord, size);
        case type::DST1:       return _index<offset_t>::reflect_Nplus1(coord, size);
        case type::DST2:       return _index<offset_t>::reflect_N(coord, size);
        case type::DFT:        return _index<offset_t>::circular(coord, size);
        default:               return _index<offset_t>::inbounds(coord, size);
      }
    }

    template <typename offset_t, typename size_t = offset_t>
    inline CUDEV int8_t sign(offset_t coord, size_t size) const
    {
      switch (bnd) {
        case type::Replicate:  return _sign::constant(coord, size);
        case type::DCT1:       return _sign::constant(coord, size);
        case type::DCT2:       return _sign::constant(coord, size);
        case type::DST1:       return _sign::periodic1(coord, size);
        case type::DST2:       return _sign::periodic2(coord, size);
        case type::DFT:        return _sign::constant(coord, size);
        // `Zero` and `NoCheck` both fall through to the primary `utils<B>`
        // template (bounds-checked sign); keep the Dynamic path bit-identical.
        default:               return _sign::inbounds(coord, size);
      }
    }
};

// Not iso -> use sign
template <type... B> struct getutils {
    template <typename val_t, typename scalar_t, typename offset_t>
    static inline CUDEV scalar_t
    cget(const scalar_t * ptr, offset_t offset, int8_t sign)
    { return cget<val_t>(ptr, offset, sign); }
    template <typename scalar_t, typename offset_t, typename val_t>
    static inline CUDEV void
    add(scalar_t *ptr, offset_t offset, val_t val, int8_t sign)
    { return add(ptr, offset, val, sign); }
};

// iso -> no need for sign

template <type B> struct getutils<B> {
    template <typename val_t, typename scalar_t, typename offset_t>
    static inline CUDEV scalar_t
    cget(const scalar_t * ptr, offset_t offset, int8_t)
    { return bound::cget<val_t>(ptr, offset); }
    template <typename scalar_t, typename offset_t, typename val_t>
    static inline CUDEV void
    add(scalar_t *ptr, offset_t offset, val_t val, int8_t)
    { return bound::add(ptr, offset, val); }
};

template <type B> struct getutils<B,B> {
    template <typename val_t, typename scalar_t, typename offset_t>
    static inline CUDEV scalar_t
    cget(const scalar_t * ptr, offset_t offset, int8_t)
    { return bound::cget<val_t>(ptr, offset); }
    template <typename scalar_t, typename offset_t, typename val_t>
    static inline CUDEV void
    add(scalar_t *ptr, offset_t offset, val_t val, int8_t)
    { return bound::add(ptr, offset, val); }
};

template <type B> struct getutils<B,B,B> {
    template <typename val_t, typename scalar_t, typename offset_t>
    static inline CUDEV scalar_t
    cget(const scalar_t * ptr, offset_t offset, int8_t)
    { return bound::cget<val_t>(ptr, offset); }
    template <typename scalar_t, typename offset_t, typename val_t>
    static inline CUDEV void
    add(scalar_t *ptr, offset_t offset, val_t val, int8_t)
    { return bound::add(ptr, offset, val); }
};

// unless dst/zero

#define FF_ISO_SIGN(B) \
    template <> struct getutils<B> { \
        template <typename val_t, typename scalar_t, typename offset_t> \
        static inline CUDEV scalar_t \
        cget(const scalar_t * ptr, offset_t offset, int8_t sign) \
        { return bound::cget<val_t>(ptr, offset, sign); } \
        template <typename scalar_t, typename offset_t, typename val_t> \
        static inline CUDEV void \
        add(scalar_t *ptr, offset_t offset, val_t val, int8_t sign) \
        { return bound::add(ptr, offset, val, sign); } \
    }; \
    template <> struct getutils<B,B> { \
        template <typename val_t, typename scalar_t, typename offset_t> \
        static inline CUDEV scalar_t \
        cget(const scalar_t * ptr, offset_t offset, int8_t sign) \
        { return bound::cget<val_t>(ptr, offset, sign); } \
        template <typename scalar_t, typename offset_t, typename val_t> \
        static inline CUDEV void \
        add(scalar_t *ptr, offset_t offset, val_t val, int8_t sign) \
        { return bound::add(ptr, offset, val, sign); } \
    }; \
    template <> struct getutils<B,B,B> { \
        template <typename val_t, typename scalar_t, typename offset_t> \
        static inline CUDEV scalar_t \
        cget(const scalar_t * ptr, offset_t offset, int8_t sign) \
        { return bound::cget<val_t>(ptr, offset, sign); } \
        template <typename scalar_t, typename offset_t, typename val_t> \
        static inline CUDEV void \
        add(scalar_t *ptr, offset_t offset, val_t val, int8_t sign) \
        { return bound::add(ptr, offset, val, sign); } \
    };

FF_ISO_SIGN(type::DST1)
FF_ISO_SIGN(type::DST2)
FF_ISO_SIGN(type::Zero)
// A Dynamic axis may turn out to be DST1/DST2/Zero at run time, so it must keep
// the sign-aware `cget`/`add` path.
FF_ISO_SIGN(type::Dynamic)

template <typename offset_t, typename size_t = offset_t>
struct _index_fn { typedef offset_t(*type)(offset_t, size_t); };

template <typename offset_t, typename size_t = offset_t>
using _index_fn_t = typename _index_fn<offset_t, size_t>::type;

template <typename offset_t, typename size_t = offset_t>
static inline CUDEV _index_fn_t<offset_t, size_t>
index_fn(type bound_type) {
  switch (bound_type) {
    case type::Replicate:  return _index<offset_t>::template replicate<size_t>;
    case type::DCT1:       return _index<offset_t>::template reflect_Nminus1<size_t>;
    case type::DCT2:       return _index<offset_t>::template reflect_N<size_t>;
    case type::DST1:       return _index<offset_t>::template reflect_Nplus1<size_t>;
    case type::DST2:       return _index<offset_t>::template reflect_N<size_t>;
    case type::DFT:        return _index<offset_t>::template circular<size_t>;
    case type::Zero:       return _index<offset_t>::template inbounds<size_t>;
    default:               return _index<offset_t>::template inbounds<size_t>;
  }
}

template <typename offset_t, typename size_t = offset_t>
static inline CUDEV offset_t
index(type bound_type, offset_t coord, size_t size) {
  return index_fn<offset_t, size_t>(bound_type)(coord, size);
  // switch (bound_type) {
  //   case type::Replicate:  return _index<offset_t>::replicate(coord, size);
  //   case type::DCT1:       return _index<offset_t>::reflect_Nminus1(coord, size);
  //   case type::DCT2:       return _index<offset_t>::reflect_N(coord, size);
  //   case type::DST1:       return _index<offset_t>::reflect_Nplus1(coord, size);
  //   case type::DST2:       return _index<offset_t>::reflect_N(coord, size);
  //   case type::DFT:        return _index<offset_t>::circular(coord, size);
  //   case type::Zero:       return _index<offset_t>::inbounds(coord, size);
  //   default:               return _index<offset_t>::inbounds(coord, size);
  // }
}

template <typename offset_t, typename size_t = offset_t>
struct _sign_fn { typedef int8_t(*type)(offset_t, size_t); };

template <typename offset_t, typename size_t = offset_t>
using _sign_fn_t = typename _sign_fn<offset_t, size_t>::type;

template <typename offset_t, typename size_t = offset_t>
static inline CUDEV _sign_fn_t<offset_t, size_t>
sign_fn(type bound_type) {
  switch (bound_type) {
    case type::Replicate:  return _sign::constant<offset_t, size_t>;
    case type::DCT1:       return _sign::constant<offset_t, size_t>;
    case type::DCT2:       return _sign::constant<offset_t, size_t>;
    case type::DST1:       return _sign::periodic1<offset_t, size_t>;
    case type::DST2:       return _sign::periodic2<offset_t, size_t>;
    case type::DFT:        return _sign::constant<offset_t, size_t>;
    case type::Zero:       return _sign::inbounds<offset_t, size_t>;
    default:               return _sign::inbounds<offset_t, size_t>;
  }
}

template <typename offset_t, typename size_t = offset_t>
static inline CUDEV int8_t
sign(type bound_type, offset_t coord, size_t size) {
  return sign_fn<offset_t, size_t>(bound_type)(coord, size);
  // switch (bound_type) {
  //   case type::Replicate:  return _sign::constant(coord, size);
  //   case type::DCT1:       return _sign::constant(coord, size);
  //   case type::DCT2:       return _sign::constant(coord, size);
  //   case type::DST1:       return _sign::periodic1(coord, size);
  //   case type::DST2:       return _sign::periodic2(coord, size);
  //   case type::DFT:        return _sign::constant(coord, size);
  //   case type::Zero:       return _sign::inbounds(coord, size);
  //   default:               return _sign::inbounds(coord, size);
  // }
}

FF_NAMESPACE_END(bound)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_BOUNDS
