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
constexpr inline type transpose(type b)
{
  return b == type::DCT1 ? type::DST1 :
         b == type::DST1 ? type::DCT1 :
         b == type::DCT2 ? type::DST2 :
         b == type::DST2 ? type::DCT2 :
         b;
}

// Can a REACH-2 energy (field `bending`, flow Lamé) build an exactly
// self-adjoint operator under this boundary condition?
//
// The difference-form stencil reproduces the exact symmetric operator at a
// boundary as long as the boundary FOLD is an involution on the tap set: the
// tap that voxel `p` folds onto must fold back onto `p` with the reciprocal
// sign.
//
// CORRECTION (measured, not assumed -- see fastfields-kernels#56's review):
// reach-1 energies (`absolute`, `membrane`) are NOT self-adjoint under every
// condition either. DCT1's whole-sample fold lands the -1 tap of x=0 onto its
// own +1 tap, so A[0][1] picks up the fold while A[1][0] does not -- measured
// relative asymmetry 0.29-0.47 depending on D, identical old vs. new engine
// (pre-existing, not introduced by this rewrite). This predicate currently
// covers only the reach-2 (bending) case below; a reach-1 predicate rejecting
// DCT1 for membrane does not exist yet and is tracked as a follow-up to
// fastfields-kernels#50's Decision 2, alongside the DST1 correction next.
//
// Reach 2 also folds ±2 taps and the ±1/±1 corners:
//
//   * Replicate  -- clamping is idempotent, not involutive: both x-1 and x-2
//                   fold onto 0 at x=0, so the (0,-2) matrix entry has no
//                   (-2,0) partner to mirror. Measured asymmetric (confirmed).
//   * DCT1       -- same whole-sample-fold mechanism as membrane above.
//                   Measured asymmetric (confirmed).
//   * DST1       -- CORRECTION: measured EXACTLY self-adjoint for field
//                   bending at every D (0 relative asymmetry, to the last
//                   bit) -- the ±2 fold lands back on the centre voxel (a
//                   diagonal entry) and the ±1 fold hits the sign-0 phantom
//                   node, so no unmatched off-diagonal entry is created.
//                   Included in this predicate's rejection set below anyway,
//                   conservatively, pending the fastfields-kernels#50
//                   follow-up decision -- the exclusion may belong to flow's
//                   Lamé cross-coupling block (`transpose()`, phase 2) rather
//                   than to field's plain bending term.
//
// The half-sample family (DCT2/DST2, both reflect_N), DFT, Zero and NoCheck are
// involutive at every reach and stay exact for both membrane and bending. See
// fastfields-kernels#43 (the original matvec_bending symmetry evidence,
// partially superseded by the correction above) and fastfields-lib#26 (the
// Lamé mirror).
//
// This is only a PREDICATE: it is `constexpr` and device-safe so a kernel can
// `static_assert` on it, but the runtime rejection belongs at the host dispatch
// entry, checked ONCE per call rather than once per voxel (fastfields-kernels#50
// decision 2).
constexpr inline bool supports_bending(type b)
{
  return !(b == type::Replicate || b == type::DCT1 || b == type::DST1);
}

// Is `utils<b>::index` guaranteed to land inside [0, n)?
//
// A stencil read of the DATA is always safe -- `cget(ptr, off, sgn)` returns 0
// without dereferencing when `sgn == 0`. But a read that does NOT carry a sign
// (a strictly-positive RLS weight map, say) has no such guard, so it must be
// gated on the folded index actually being a real memory location.
//
//   * Replicate / DCT1 / DCT2 / DST2 / DFT map every coordinate into [0, n).
//   * Zero and NoCheck pass the coordinate through unchanged (`inbounds`).
//   * DST1's `reflect_Nplus1` has support N+1 and can return -1 or n -- the
//     phantom Dirichlet nodes, exactly where `periodic1` returns sign 0.
//
// For the three that can leave the support, `sign(...) == 0` is precisely the
// out-of-range test, so gating on the sign is both necessary and sufficient.
// `Dynamic` conservatively reports false, which keeps the runtime path correct
// for whichever condition it ends up carrying.
constexpr inline bool index_stays_inbounds(type b)
{
  return b == type::Replicate || b == type::DCT1 || b == type::DCT2
      || b == type::DST2      || b == type::DFT;
}
FF_NAMESPACE_END(bound)

using bound_t = bound::type;
template <bound_t...  B> using Bound = meta::Tuple<bound_t,  B...>;

FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(bound)

using FF::bound::type;
using FF::bound::transpose;
using FF::bound::supports_bending;
using FF::bound::index_stays_inbounds;

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
// `using bound_utils_x = utils<BX>;` aliases. A single Dynamic instantiation
// replaces the eight static ones -- which is what keeps nvcc's `ptxas` inside a
// sane memory budget (fastfields-kernels#42). Ported verbatim from `main` so the
// two tracks merge without a conflict; the `FF_STATIC_BOUND_*` build policy and
// `BoundVec` that drive it from the dispatch layers arrive with that branch.

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
