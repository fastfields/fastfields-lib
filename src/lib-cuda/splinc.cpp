#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <fastfields/api/cuda/splinc.h>
#include <fastfields/core/autocast.h>
#include <fastfields/core/dispatch.h>
#include <fastfields/core/dlpack.h>
#include <fastfields/core/cuda_switch.h>
#include <fastfields/impl/kernels/utils.h>
#include <fastfields/impl/cuda/splinc.h>

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)

/***********************************************************************
 *                              POLES                                  *
 ***********************************************************************/

// Host-side poles / npoles: the same 30-digit values as kernels/splinc.h
// `get_poles` (mpmath at 60 digits; the square-root closed forms that exist up
// to order 5 lose up to 2e-13 in float64 to cancellation).
static inline int get_poles_host(int order, double * poles)
{
    switch (order) {
        case 0:
        case 1:
            return 0;
        case 2: poles[0] = -0.171572875253809902396622551581;  return 1;
        case 3: poles[0] = -0.267949192431122706472553658494;  return 1;
        case 4:
            poles[0] = -0.361341225900220177092212841326;
            poles[1] = -0.0137254292973391213603312269391;
            return 2;
        case 5:
            poles[0] = -0.430575347099973791851434783494;
            poles[1] = -0.0430962882032646538227123768226;
            return 2;
        case 6:
            poles[0] = -0.488294589303044755130118038884;
            poles[1] = -0.0816792710762375125979377657371;
            poles[2] = -0.00141415180832581775108724397656;
            return 3;
        case 7:
            poles[0] = -0.535280430796438165542403781682;
            poles[1] = -0.122554615192326690515272264359;
            poles[2] = -0.00914869480960827692859302165165;
            return 3;
    }
    throw std::invalid_argument("Unsupported spline order (must be 0..7)");
}

/***********************************************************************
 *                            DISPATCH                                 *
 ***********************************************************************/

namespace {
template <int npoles, bound::type B, typename scalar_t, typename offset_t>
inline void _splinc(
          int64_t   nbatch  ,   // number of batch dimensions (ndim = nbatch + 1)
          void    * inp     ,   // pointer to data [*batch, n]
    const int64_t * size    ,   // [ndim] data shape   == (*batch, n)
    const int64_t * stride  ,   // [ndim] data strides
    const double  * poles   ,   // [npoles] filter poles
          bound::type bnd   )   // runtime bound (read when B is Dynamic)
{
    const int64_t    ndim    = nbatch + 1;
    const IndexArray<offset_t> _size   (size, ndim);
    const IndexArray<offset_t> _stride (stride, ndim);
          scalar_t * _inp    = static_cast<scalar_t *>(inp);
    splinc::loop<npoles, B, scalar_t, offset_t, double>(
        static_cast<offset_t>(nbatch), _inp, _size, _stride, poles, bnd);
}
} // anonymous namespace

// npoles x bound -> compile-time; scalar_t/offset_t already fixed. The
// FF_BOUND_* template arguments follow the BOUNDFLAGS build policy (bounds.h):
// a condition that is not statically compiled maps to `Dynamic`, and the
// runtime `bnd` passed along is what the kernels then read. NoCheck has no
// extension to prefilter against and is refused (the hub refuses it first).
#define DISPATCH_SPLINC_BOUND(NP, S, O, args...)                        \
    switch (bnd) {                                                      \
        case bound_t::Zero:      return _splinc<NP, FF_BOUND_ZERO,      S, O>(args, bnd); \
        case bound_t::Replicate: return _splinc<NP, FF_BOUND_REPLICATE, S, O>(args, bnd); \
        case bound_t::DCT1:      return _splinc<NP, FF_BOUND_DCT1,      S, O>(args, bnd); \
        case bound_t::DCT2:      return _splinc<NP, FF_BOUND_DCT2,      S, O>(args, bnd); \
        case bound_t::DST1:      return _splinc<NP, FF_BOUND_DST1,      S, O>(args, bnd); \
        case bound_t::DST2:      return _splinc<NP, FF_BOUND_DST2,      S, O>(args, bnd); \
        case bound_t::DFT:       return _splinc<NP, FF_BOUND_DFT,       S, O>(args, bnd); \
        default: throw std::invalid_argument(                          \
            "splinc only supports zero/replicate/dct1/dct2/dst1/dst2/dft bounds"); \
    }

#define DISPATCH_SPLINC_NPOLES(S, O, args...)                          \
    switch (npoles) {                                                   \
        case 1: DISPATCH_SPLINC_BOUND(1, S, O, args);                   \
        case 2: DISPATCH_SPLINC_BOUND(2, S, O, args);                   \
        case 3: DISPATCH_SPLINC_BOUND(3, S, O, args);                   \
        default: throw std::invalid_argument("Unsupported npoles");     \
    }

#define DISPATCH_SPLINC(args...)                                                 \
{                                                                                \
    const bool use_32bits = FF_CANUSE32BITS(inp_out);                            \
    const auto code = static_cast<DLDataTypeCode>(inp_out.dtype.code);           \
    switch (code) {                                                              \
        case kDLFloat: switch (inp_out.dtype.bits) {                             \
            case 32:                                                             \
                if (use_32bits) DISPATCH_SPLINC_NPOLES(float,  off32_t, args)    \
                else            DISPATCH_SPLINC_NPOLES(float,  int64_t, args)    \
            case 64:                                                             \
                if (use_32bits) DISPATCH_SPLINC_NPOLES(double, off32_t, args)    \
                else            DISPATCH_SPLINC_NPOLES(double, int64_t, args)    \
            default: break;                                                      \
        };                                                                       \
        default: break;                                                          \
    };                                                                           \
    throw std::invalid_argument("only floating point data types are supported"); \
}

void spline_coeff(
          DLTensor & inp_out_,
          int8_t     spline  ,
          int8_t     bound   ,
          intptr_t   /* stream <unused> */
)
{
    // Normalise a NULL strides field (compact row-major) before dispatch.
    ContiguousStrides _io(inp_out_);
    DLTensor & inp_out = _io.t;

    FF_CHECK_NO_LANES(inp_out)

    double poles[3];
    const int npoles = get_poles_host(static_cast<int>(spline), poles);
    if (npoles == 0) return;  // orders 0/1: identity

    const int64_t  nbatch = inp_out.ndim - 1;
    const bound_t  bnd    = static_cast<bound_t>(bound);

    DISPATCH_SPLINC(
        nbatch,
        FF_VOIDPTR(inp_out),
        inp_out.shape,
        inp_out.strides,
        poles
    )
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
