#pragma once
#include <fastfields/core/dlpack.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <fastfields/core/defines.h>

#ifndef FF_LIB_BOUND_SPLINE_T
#define FF_LIB_BOUND_SPLINE_T
FF_NAMESPACE_BEGIN(FF_NS)

FF_NAMESPACE_BEGIN(bound_t)
using T = int8_t;
static constexpr T Dynamic   = -1;
static constexpr T Zero      =  0;
static constexpr T Replicate =  1;
static constexpr T DCT1      =  2;
static constexpr T DCT2      =  3;
static constexpr T DST1      =  4;
static constexpr T DST2      =  5;
static constexpr T DFT       =  6;
static constexpr T NoCheck   =  7;
FF_NAMESPACE_END(bound_t)

FF_NAMESPACE_BEGIN(spline_t)
using T = int8_t;
static constexpr T Dynamic       = -1;
static constexpr T Nearest       =  0;
static constexpr T Linear        =  1;
static constexpr T Quadratic     =  2;
static constexpr T Cubic         =  3;
static constexpr T FourthOrder   =  4;
static constexpr T FifthOrder    =  5;
static constexpr T SixthOrder    =  6;
static constexpr T SeventhOrder  =  7;
FF_NAMESPACE_END(spline_t)

FF_NAMESPACE_END(FF_NS)
#endif // FF_LIB_BOUND_SPLINE_T

FF_NAMESPACE_BEGIN(FF_NS)

/**
 * @brief Assert that `bound` is a boundary condition the prefilter implements.
 *
 * The prefilter is exact -- spline interpolation of its output with the same
 * boundary reproduces the input -- for every boundary condition that defines
 * an extension: `Zero`, `Replicate`, `DCT1`, `DCT2`, `DST1`, `DST2` and `DFT`.
 * The last five use a recursive filter with exact initial conditions; `Zero`
 * and `Replicate` use the `DST1` / `DCT2` filter, exact for them at orders 2
 * and 3, plus an exact edge correction from order 4 (see
 * impl/kernels/splinc.h).
 *
 * `NoCheck` (and any value outside the enumeration) has no extension, so there
 * is no system to invert. It used to be silently aliased onto the `DCT1`
 * recursion -- as `Zero`, `DST1` and `DST2` were before they were implemented
 * (fastfields-lib#65) -- so reject it up front. This is the hub's single
 * dispatch point for `spline_coeff`, so the check covers the CPU and CUDA
 * backends and every downstream binding.
 *
 * Orders 0 and 1 are exempt: the prefilter is the identity there, never runs
 * the recursion, and so never touches the boundary. Orders outside 0..7 are
 * left alone too, so the backend's "unsupported spline order" stays the error
 * the caller sees.
 *
 * @param spline  Spline order (see spline_t)
 * @param bound   Boundary condition (see bound_t)
 */
inline void require_splinc_bound(int8_t spline, int8_t bound)
{
    // Identity orders (0/1) and out-of-range orders: nothing to validate here.
    if (spline < spline_t::Quadratic || spline > spline_t::SeventhOrder) return;

    switch (bound) {
        case bound_t::Zero:
        case bound_t::Replicate:
        case bound_t::DCT1:
        case bound_t::DCT2:
        case bound_t::DST1:
        case bound_t::DST2:
        case bound_t::DFT: return;
        default: break;
    }

    const char * name = bound == bound_t::NoCheck ? "nocheck" : "unknown";
    throw std::invalid_argument(
        std::string("fastfields: `spline_coeff` is only implemented for bounds "
                    "(zero, replicate, dct1, dct2, dst1, dst2, dft) but got: ") +
        name);
}

/**
 * @brief In-place spline coefficient prefilter along the last dimension.
 *
 * Filters the last axis (all leading axes are batch); prefiltering makes
 * spline interpolation reproduce the input samples. To filter several axes,
 * permute the tensor and call repeatedly.
 *
 * @param inp_out  Input/Output tensor in DLTensor format (float32/float64)
 * @param spline   Spline order (orders 0/1 are no-ops)
 * @param bound    Boundary condition: zero/replicate/dct1/dct2/dst1/dst2/dft.
 *                 NoCheck (or an unknown value) throws `std::invalid_argument`
 *                 (see require_splinc_bound). Ignored for orders 0/1.
 * @param stream   Cuda stream on which to operate
 */
void spline_coeff(
          DLTensor & inp_out ,
          int8_t     spline   = spline_t::Cubic,
          int8_t     bound    = bound_t::DCT2,
          intptr_t   stream   = 0
);

FF_NAMESPACE_END(FF_NS)
