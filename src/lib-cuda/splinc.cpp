#include <stdexcept>
#include <cstdint>
#include <cmath>
#include "splinc.h"
#include "autocast.h"
#include "dlpack.h"
#include "impl/kernels/cuda_switch.h"
#include "impl/kernels/utils.h"
#include "impl/splinc.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

#define VOIDPTR(x)      (static_cast<void*>(static_cast<char*>(x.data) + x.byte_offset))
#define CANUSE32BITS(x) (canUse32BitIndexMath(x.ndim, x.shape, x.strides))

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

namespace {
template <int npoles, bound::type B, typename scalar_t, typename offset_t>
inline void _splinc(
          int64_t   nbatch  ,   // number of batch dimensions (ndim = nbatch + 1)
          void    * inp     ,   // pointer to data [*batch, n]
    const int64_t * size    ,   // [ndim] data shape   == (*batch, n)
    const int64_t * stride  ,   // [ndim] data strides
    const double  * poles   )   // [npoles] filter poles
{
    const int64_t    ndim    = nbatch + 1;
    const offset_t * _size   = copy_if_needed<offset_t *>(size,   ndim);
    const offset_t * _stride = copy_if_needed<offset_t *>(stride, ndim);
          scalar_t * _inp    = static_cast<scalar_t *>(inp);
    splinc::loop<npoles, B, scalar_t, offset_t, double>(
        static_cast<offset_t>(nbatch), _inp, _size, _stride, poles);
    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride);
}
} // anonymous namespace

// npoles x bound -> compile-time; scalar_t/offset_t already fixed.
#define DISPATCH_SPLINC_BOUND(NP, S, O, args...)                        \
    switch (bnd) {                                                      \
        case bound_t::Zero:      return _splinc<NP, bound_t::Zero,      S, O>(args); \
        case bound_t::Replicate: return _splinc<NP, bound_t::Replicate, S, O>(args); \
        case bound_t::DCT1:      return _splinc<NP, bound_t::DCT1,      S, O>(args); \
        case bound_t::DCT2:      return _splinc<NP, bound_t::DCT2,      S, O>(args); \
        case bound_t::DFT:       return _splinc<NP, bound_t::DFT,       S, O>(args); \
        default: throw std::invalid_argument(                          \
            "splinc only supports zero/replicate/dct1/dct2/dft bounds"); \
    }

#define DISPATCH_SPLINC_NPOLES(S, O, args...)                          \
    switch (npoles) {                                                   \
        case 1: DISPATCH_SPLINC_BOUND(1, S, O, args);                   \
        case 2: DISPATCH_SPLINC_BOUND(2, S, O, args);                   \
        case 3: DISPATCH_SPLINC_BOUND(3, S, O, args);                   \
        default: throw std::invalid_argument("Unsupported npoles");     \
    }

#define DISPATCH_SPLINC(args...)                                        \
{                                                                       \
    const bool use_32bits = CANUSE32BITS(inp_out);                      \
    const auto code = static_cast<DLDataTypeCode>(inp_out.dtype.code);  \
    switch (code) {                                                     \
        case kDLFloat: switch (inp_out.dtype.bits) {                    \
            case 32:                                                    \
                if (use_32bits) DISPATCH_SPLINC_NPOLES(float,  int32_t, args) \
                else            DISPATCH_SPLINC_NPOLES(float,  int64_t, args) \
            case 64:                                                    \
                if (use_32bits) DISPATCH_SPLINC_NPOLES(double, int32_t, args) \
                else            DISPATCH_SPLINC_NPOLES(double, int64_t, args) \
            default: break;                                             \
        };                                                              \
        default: break;                                                 \
    };                                                                  \
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

    CHECK_NO_LANES(inp_out)

    double poles[3];
    const int npoles = get_poles_host(static_cast<int>(spline), poles);
    if (npoles == 0) return;  // orders 0/1: identity

    const int64_t  nbatch = inp_out.ndim - 1;
    const bound_t  bnd    = static_cast<bound_t>(bound);

    DISPATCH_SPLINC(
        nbatch,
        VOIDPTR(inp_out),
        inp_out.shape,
        inp_out.strides,
        poles
    )
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
