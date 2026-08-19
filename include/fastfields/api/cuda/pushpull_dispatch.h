#ifndef FF_CUDA_PUSHPULL_DISPATCH
#define FF_CUDA_PUSHPULL_DISPATCH
/**
 * Private (not installed) header: the argument-marshalling helpers, the
 * argument checks and the ndim x order x bound x dtype dispatch matrix
 * shared by `pushpull.cpp` (the forward ops) and `pushpull_backward.cpp`
 * (their adjoints).
 *
 * The two live in separate translation units for the same reason
 * `reg_field` / `reg_field_rls` are split: the backward ops instantiate the
 * whole template matrix a second time, and ptxas memory (not just wall
 * time) is the binding constraint on this module. They must nevertheless
 * agree exactly on which (order, bound) combinations are statically
 * instantiated and which share the `Dynamic` path, so the dispatch macros
 * are defined once, here, rather than copied.
 */
#include <stdexcept>
#include <cstdint>
#include "fastfields/core/autocast.h"
#include "fastfields/core/dispatch.h"
#include "fastfields/core/dlpack.h"
#include "fastfields/core/cuda_switch.h"
#include "fastfields/impl/kernels/utils.h"
#include "fastfields/impl/cuda/pushpull.h"

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)

// reduce/accumulation type used by the sampling kernels. Match jitfields
// (float64) for CPU accuracy.
typedef double reduce_t;

/***********************************************************************
 *                            DISPATCH                                *
 ***********************************************************************/

#define FF_PP_DTYPE(D, I, B, FN, args...)                                         \
    switch (code) {                                                            \
        case kDLFloat: switch (bits) {                                         \
            case 32: return (use_32bits ? FN<D,I,B,float, int32_t>(args)       \
                                        : FN<D,I,B,float, int64_t>(args));      \
            case 64: return (use_32bits ? FN<D,I,B,double,int32_t>(args)       \
                                        : FN<D,I,B,double,int64_t>(args));      \
            default: break;                                                    \
        }; default: break;                                                     \
    }

// The template argument fed to FF_PP_DTYPE is FF_BOUND_<NAME> / FF_SPLINE_<NAME>
// (kernels/bounds.h, kernels/spline.h) -- the condition/order itself when it
// is statically compiled, `Dynamic` otherwise per BOUNDFLAGS/SPLINEFLAGS. The
// switch labels stay exhaustive on the *runtime* value either way; only the
// instantiated template argument collapses onto the shared Dynamic path.
// This is what actually keeps ptxas's memory bounded -- see BOUNDFLAGS/
// SPLINEFLAGS in the Makefile.
#define FF_PP_BOUND(D, I, FN, args...)                                            \
    switch (bnd) {                                                             \
        case bound_t::Zero:      FF_PP_DTYPE(D,I,FF_BOUND_ZERO,     FN,args); break; \
        case bound_t::Replicate: FF_PP_DTYPE(D,I,FF_BOUND_REPLICATE,FN,args); break; \
        case bound_t::DCT1:      FF_PP_DTYPE(D,I,FF_BOUND_DCT1,     FN,args); break; \
        case bound_t::DCT2:      FF_PP_DTYPE(D,I,FF_BOUND_DCT2,     FN,args); break; \
        case bound_t::DST1:      FF_PP_DTYPE(D,I,FF_BOUND_DST1,     FN,args); break; \
        case bound_t::DST2:      FF_PP_DTYPE(D,I,FF_BOUND_DST2,     FN,args); break; \
        case bound_t::DFT:       FF_PP_DTYPE(D,I,FF_BOUND_DFT,      FN,args); break; \
        case bound_t::NoCheck:   FF_PP_DTYPE(D,I,FF_BOUND_NOCHECK,  FN,args); break; \
        default: throw std::invalid_argument("Unsupported boundary condition");\
    }

#define FF_PP_ORDER(D, FN, args...)                                               \
    switch (spl) {                                                             \
        case spline_t::Nearest:      FF_PP_BOUND(D,FF_SPLINE_NEAREST,     FN,args); break; \
        case spline_t::Linear:       FF_PP_BOUND(D,FF_SPLINE_LINEAR,      FN,args); break; \
        case spline_t::Quadratic:    FF_PP_BOUND(D,FF_SPLINE_QUADRATIC,   FN,args); break; \
        case spline_t::Cubic:        FF_PP_BOUND(D,FF_SPLINE_CUBIC,       FN,args); break; \
        case spline_t::FourthOrder:  FF_PP_BOUND(D,FF_SPLINE_FOURTHORDER, FN,args); break; \
        case spline_t::FifthOrder:   FF_PP_BOUND(D,FF_SPLINE_FIFTHORDER,  FN,args); break; \
        case spline_t::SixthOrder:   FF_PP_BOUND(D,FF_SPLINE_SIXTHORDER,  FN,args); break; \
        case spline_t::SeventhOrder: FF_PP_BOUND(D,FF_SPLINE_SEVENTHORDER,FN,args); break; \
        default: throw std::invalid_argument("Unsupported spline order");      \
    }

#define FF_DISPATCH_PP(FN, args...)                                               \
{                                                                              \
    switch (ndim) {                                                           \
        case 1: FF_PP_ORDER(1, FN, args); break;                                 \
        case 2: FF_PP_ORDER(2, FN, args); break;                                 \
        case 3: FF_PP_ORDER(3, FN, args); break;                                 \
        default: throw std::invalid_argument("Only 1D, 2D and 3D are supported"); \
    };                                                                        \
    throw std::invalid_argument("Unsupported data type");                     \
}


FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)

#endif // FF_CUDA_PUSHPULL_DISPATCH
