#include <stdexcept>
#include <cmath>
#include "splinc.h"
#include "dlpack.h"
// R7 (TEENY-MIGRATION.md sec. 9): fastfields vendors DLPack v1.2, teeny v1.1,
// and both use the guard DLPACK_DLPACK_H_ -- so whichever is seen first wins
// for the whole TU. Our "dlpack.h" is included ABOVE on purpose; keep it there.
#include <teeny/dlpack.h>
#include "impl/kernels/cuda_switch.h"
#include "impl/kernels/utils.h"
#include "impl/splinc.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

/***********************************************************************
 *                              CHECKS                                 *
 ***********************************************************************/

#define CHECK_NO_LANES(tensor)                                          \
    if (tensor.dtype.lanes > 1)                                         \
        throw std::invalid_argument(                                    \
            "Only scalar data types are supported"                      \
        );

/***********************************************************************
 *                              POLES                                  *
 ***********************************************************************/

// Host-side poles / npoles (mirrors kernels/splinc.h get_poles).
static inline int get_poles_host(int order, double * poles)
{
    switch (order) {
        case 0:
        case 1:
            return 0;
        case 2:
            poles[0] = sqrt(8.) - 3.;
            return 1;
        case 3:
            poles[0] = sqrt(3.) - 2.;
            return 1;
        case 4:
            poles[0] = sqrt(664. - sqrt(438976.)) + sqrt(304.) - 19.;
            poles[1] = sqrt(664. + sqrt(438976.)) - sqrt(304.) - 19.;
            return 2;
        case 5:
            poles[0] = sqrt(67.5 - sqrt(4436.25)) + sqrt(26.25) - 6.5;
            poles[1] = sqrt(67.5 + sqrt(4436.25)) - sqrt(26.25) - 6.5;
            return 2;
        case 6:
            poles[0] = -0.48829458930304475513011803888378906211227916123937760839;
            poles[1] = -0.081679271076237512597937765737059080653379610398148178525368;
            poles[2] = -0.00141415180832581775108724397655859252786416905534669851652709;
            return 3;
        case 7:
            poles[0] = -0.5352804307964381655424037816816460718339231523426924148812;
            poles[1] = -0.122554615192326690515272264359357343605486549427295558490763;
            poles[2] = -0.0091486948096082769285930216516478534156925639545994482648003;
            return 3;
    }
    throw std::invalid_argument("Unsupported spline order (must be 0..7)");
}

/***********************************************************************
 *                            DISPATCH                                 *
 ***********************************************************************/

// The dtype arm's IMPORT POINT. `tny::from_dlpack` builds the teeny carrier
// once, straight off the bare DLTensor, and does by construction all three
// things this path used to do by hand: it folds `byte_offset` into the data
// pointer (was VOIDPTR), expands a NULL `strides` field to row-major (was
// ContiguousStrides), and copies the shape/stride metadata into the carrier --
// so the impl below needs no pointer, no nbatch, and no size[]/stride[] arrays.
//
// D1/R5: the int32 offset arm is gone. The CPU narrowing was measured a wash on
// a 64-bit ALU (distance-slice review), so this is now the single int64
// instantiation; `offset_t` is whatever the carrier carries (int64 off DLPack).
namespace {
template <int npoles, bound::type B, typename scalar_t>
inline void _splinc(
          DLTensor & inp_out,   // data [*batch, n], prefiltered in place
    const double   * poles  )   // [npoles] filter poles
{
    auto at = tny::from_dlpack<scalar_t>(&inp_out);
    splinc::loop<npoles, B>(at, poles);
}
} // anonymous namespace

// npoles x bound -> compile-time; scalar_t already fixed.
#define DISPATCH_SPLINC_BOUND(NP, S, args...)                           \
    switch (bnd) {                                                      \
        case bound_t::Zero:      return _splinc<NP, bound_t::Zero,      S>(args); \
        case bound_t::Replicate: return _splinc<NP, bound_t::Replicate, S>(args); \
        case bound_t::DCT1:      return _splinc<NP, bound_t::DCT1,      S>(args); \
        case bound_t::DCT2:      return _splinc<NP, bound_t::DCT2,      S>(args); \
        case bound_t::DFT:       return _splinc<NP, bound_t::DFT,       S>(args); \
        default: throw std::invalid_argument(                          \
            "splinc only supports zero/replicate/dct1/dct2/dft bounds"); \
    }

#define DISPATCH_SPLINC_NPOLES(S, args...)                             \
    switch (npoles) {                                                   \
        case 1: DISPATCH_SPLINC_BOUND(1, S, args);                      \
        case 2: DISPATCH_SPLINC_BOUND(2, S, args);                      \
        case 3: DISPATCH_SPLINC_BOUND(3, S, args);                      \
        default: throw std::invalid_argument("Unsupported npoles");     \
    }

#define DISPATCH_SPLINC(args...)                                        \
{                                                                       \
    const auto code = static_cast<DLDataTypeCode>(inp_out.dtype.code);  \
    switch (code) {                                                     \
        case kDLFloat: switch (inp_out.dtype.bits) {                    \
            case 32: DISPATCH_SPLINC_NPOLES(float,  args)               \
            case 64: DISPATCH_SPLINC_NPOLES(double, args)               \
            default: break;                                             \
        };                                                              \
        default: break;                                                 \
    };                                                                  \
    throw std::invalid_argument("only floating point data types are supported"); \
}

void spline_coeff(
          DLTensor & inp_out,
          int8_t     spline  ,
          int8_t     bound   ,
          int        /* stream <unused> */
)
{
    // A NULL strides field (DLPack's compact row-major shorthand) and a non-zero
    // byte_offset are now normalised by `tny::from_dlpack` inside the dtype arm,
    // so there is nothing to pre-normalise here. The check ORDER is unchanged:
    // lanes -> spline-order throw -> dtype throw.
    CHECK_NO_LANES(inp_out)

    double poles[3];
    const int npoles = get_poles_host(static_cast<int>(spline), poles);
    if (npoles == 0) return;  // orders 0/1: identity

    const bound_t  bnd    = static_cast<bound_t>(bound);

    DISPATCH_SPLINC(
        inp_out,
        poles
    )
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
