#ifndef FF_BOUNDS
#define FF_BOUNDS
#include "fastfields/core/cuda_switch.h"
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
// Which conditions this actually yields an exactly self-adjoint operator for is
// MEASURED, in `supports_lame_cross` below -- an earlier version of this comment
// argued the answer (DCT2/DST2/DFT/Zero/Replicate/NoCheck exact, DCT1/DST1 not)
// and was right about DCT1/DST1 but wrong about Replicate. See fastfields-lib#26
// for the fix this function implements and fastfields-kernels#59 for the
// measurement.
constexpr inline type transpose(type b)
{
  return b == type::DCT1 ? type::DST1 :
         b == type::DST1 ? type::DCT1 :
         b == type::DCT2 ? type::DST2 :
         b == type::DST2 ? type::DCT2 :
         b;
}

// Can a separable finite-difference energy of the given REACH build an exactly
// self-adjoint operator under this boundary condition?
//
// The difference-form stencil reproduces the exact symmetric operator at a
// boundary as long as the boundary FOLD is an involution on the tap set: the
// tap that voxel `p` folds onto must fold back onto `p` with the reciprocal
// sign. Larger reach folds more taps, so it can only ever lose conditions --
// which is why one predicate keyed on reach covers all three energies.
//
// The rejection set is MEASURED, never argued: assemble `A` column by column
// (matvec on unit vectors) and take `max|A - A^T| / max|A|`. Two independent
// from-scratch measurements on different grids agree, and both agree old engine
// vs. new -- these are pre-existing properties of the discretisation, not
// artefacts of the tap-table rewrite. Relative asymmetry, D = 1..3:
//
//        bound      | reach 0 (absolute) | reach 1 (membrane) | reach 2 (bending)
//        -----------+--------------------+--------------------+------------------
//        Zero       |          0         |          0         |         0
//        Replicate  |          0         |          0         |   0.042 - 0.13
//        DCT1       |          0         |    0.25 - 0.46     |   0.37 - 0.50
//        DCT2       |          0         |          0         |         0
//        DST1       |          0         |          0         |         0
//        DST2       |          0         |          0         |         0
//        DFT        |          0         |          0         |         0
//        NoCheck    |          0         |          0         |         0
//
// so:
//
//   * reach 0 -- `absolute` reads no neighbour at all, so there is no fold to
//                be non-involutive. Exact under every condition. (Measured
//                rather than assumed: "it is diagonal so it must be fine" is
//                the same shape of argument that was wrong twice below.)
//   * DCT1    -- whole-sample symmetry reflects about the last INBOUND voxel,
//                so at x=0 the -1 tap lands on the +1 tap: A[0][1] picks up the
//                fold and A[1][0] does not. Breaks from reach 1 upwards.
//   * Replicate -- clamping is idempotent, not involutive: at x=0 both x-1 and
//                x-2 fold onto 0, so the (0,-2) entry has no (-2,0) partner.
//                Needs a +-2 tap to bite, so reach 2 only.
//   * DST1    -- exact at every reach for these energies. Its +-2 fold lands
//                back on the centre voxel (a diagonal entry) and its +-1 fold
//                hits the sign-0 phantom node, so no unmatched off-diagonal
//                entry is ever created.
//
// Superseded claims, recorded so they are not re-derived: fastfields-kernels#43
// held that reach-1 energies were self-adjoint under every condition (wrong for
// DCT1), and fastfields-kernels#50's original Decision 2 rejected DST1 for
// bending (wrong for field). Both corrected 2026-08-01 after two independent
// measurements; see #50's updated Decision 2.
//
// SCOPE -- the SAME-AXIS stencil. Flow's membrane and bending are the same
// separable per-component stencil and do share this table (measured, #59, not
// inherited). Lame's cross-channel block is a different fold and has its own
// predicate below.
//
// These are only PREDICATES: `constexpr` and device-safe so a kernel can
// `static_assert` on one, but the runtime rejection belongs at the host
// dispatch entry, checked ONCE per call rather than once per voxel (#50
// decision 2).
constexpr inline bool supports_reach(type b, int reach)
{
  return reach <= 0 ? true                                  // no taps, no fold
       : reach == 1 ? b != type::DCT1
       :              !(b == type::DCT1 || b == type::Replicate);
}

// Named for the three field energies, so a dispatch site reads as the energy it
// is about rather than as a magic number. `reach >= 3` does not occur in this
// project and is not covered by the measurement above.
constexpr inline bool supports_absolute(type b) { return supports_reach(b, 0); }
constexpr inline bool supports_membrane(type b) { return supports_reach(b, 1); }
constexpr inline bool supports_bending (type b) { return supports_reach(b, 2); }

// The table above, executable. Costs nothing at run time and stops the set
// drifting away from the measurement the next time someone edits the comment.
#define FF_BOUND_SA_ROW(B, A, M, D)                                     \
    static_assert(supports_absolute(type::B) == A, #B " absolute");     \
    static_assert(supports_membrane(type::B) == M, #B " membrane");     \
    static_assert(supports_bending (type::B) == D, #B " bending");
FF_BOUND_SA_ROW(Zero,      true, true,  true )
FF_BOUND_SA_ROW(Replicate, true, true,  false)
FF_BOUND_SA_ROW(DCT1,      true, false, false)
FF_BOUND_SA_ROW(DCT2,      true, true,  true )
FF_BOUND_SA_ROW(DST1,      true, true,  true )
FF_BOUND_SA_ROW(DST2,      true, true,  true )
FF_BOUND_SA_ROW(DFT,       true, true,  true )
FF_BOUND_SA_ROW(NoCheck,   true, true,  true )
#undef FF_BOUND_SA_ROW

// Can the LAME (linear-elastic) CROSS-CHANNEL block be exactly self-adjoint
// under this condition?
//
// A DIFFERENT mechanism from `supports_reach`, which is why it is a different
// predicate rather than another reach value. The cross block is a product of
// two first differences, D_c^T D_e: its 4-corner gather folds the axis-`c` half
// through `transpose(b)` and the axis-`e` half through `b`. Whether that pair is
// a genuine transpose is not a question about reach at all, and the answer does
// not follow from the same-axis table either way -- `Replicate` is exact at
// reach 1 but NOT here, and `DST1` is exact at reach 2 for field's bending but
// NOT here.
//
// MEASURED, like the table above and by the same two independent methods
// (assemble `A` and take `max|A - A^T|/max|A|`; and, without assembling
// anything, `|<Av,w> - <v,Aw>|` over random v, w). Both agree, on several grids,
// D = 2 and 3, with and without the diagonal-block energies, isotropic and
// anisotropic voxels. Relative asymmetry of the pure Lame operator:
//
//        bound      |   D = 2   |   D = 3
//        -----------+-----------+---------
//        Zero       |     0     |     0
//        Replicate  |   0.110   |   0.086
//        DCT1       |   0.360   |   0.281
//        DCT2       |     0     |     0
//        DST1       |   0.055   |   0.043
//        DST2       |     0     |     0
//        DFT        |     0     |     0
//        NoCheck    |     0     |     0
//
//   * DCT1 / DST1 -- whole-sample symmetry: forward and adjoint reflect about
//                different centres (reflect_{N-1} vs reflect_{N+1}), so a single
//                companion-boundary read cannot reproduce D^T there.
//   * Replicate -- self-transpose, and clamping is idempotent rather than
//                involutive, so a corner whose two taps both clamp onto the
//                centre voxel has no partner entry. Exact at reach 1 on the
//                SAME axis (nothing pairs two clamped taps there), which is why
//                inheriting `supports_membrane` would have been wrong.
//   * DCT2 / DST2 -- half-sample symmetry, both reflect_N, and each other's
//                transpose: exact.
//
// Removing the `transpose()` from the cross block flips this table -- DCT2 and
// DST2 become asymmetric and DST1 becomes exact -- which is both the pre-#26
// behaviour and a check that the fix is load-bearing.
constexpr inline bool supports_lame_cross(type b)
{
  return !(b == type::Replicate || b == type::DCT1 || b == type::DST1);
}

// The Lame energies as a DISPATCH site sees them: the per-component blocks
// (reach 1 for `lame`, reach 2 for `lame + bending`) AND, from D >= 2, the cross
// block. There is no axis pair at D == 1, so no cross block and no extra
// condition -- measured, not assumed.
constexpr inline bool supports_lame(type b, int ndim)
{ return supports_reach(b, 1) && (ndim < 2 || supports_lame_cross(b)); }

constexpr inline bool supports_lame_bending(type b, int ndim)
{ return supports_reach(b, 2) && (ndim < 2 || supports_lame_cross(b)); }

// The measurement, executable -- `lame` and `lame+bending` at D = 1 (no cross
// block) and at D >= 2 (with one).
#define FF_BOUND_LAME_ROW(B, L1, A1, L2, A2)                            \
    static_assert(supports_lame(type::B, 1)         == L1, #B " lame 1d");     \
    static_assert(supports_lame_bending(type::B, 1) == A1, #B " all 1d");      \
    static_assert(supports_lame(type::B, 2)         == L2, #B " lame nd");     \
    static_assert(supports_lame_bending(type::B, 2) == A2, #B " all nd");
//                     lame1d  all1d  lameNd  allNd
FF_BOUND_LAME_ROW(Zero,      true,  true,  true,  true )
FF_BOUND_LAME_ROW(Replicate, true,  false, false, false)
FF_BOUND_LAME_ROW(DCT1,      false, false, false, false)
FF_BOUND_LAME_ROW(DCT2,      true,  true,  true,  true )
FF_BOUND_LAME_ROW(DST1,      true,  true,  false, false)
FF_BOUND_LAME_ROW(DST2,      true,  true,  true,  true )
FF_BOUND_LAME_ROW(DFT,       true,  true,  true,  true )
FF_BOUND_LAME_ROW(NoCheck,   true,  true,  true,  true )
#undef FF_BOUND_LAME_ROW

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
CUHOSTDEV constexpr inline bool index_stays_inbounds(type b)
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
using FF::bound::supports_reach;
using FF::bound::supports_absolute;
using FF::bound::supports_membrane;
using FF::bound::supports_bending;
using FF::bound::supports_lame_cross;
using FF::bound::supports_lame;
using FF::bound::supports_lame_bending;
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
