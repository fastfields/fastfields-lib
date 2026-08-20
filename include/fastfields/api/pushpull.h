#ifndef FF_LIB_PUSHPULL
#define FF_LIB_PUSHPULL
#include <fastfields/core/dlpack.h>
#include <cstdint>
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
 * @brief Sample ("pull") a spline-encoded volume at arbitrary coordinates.
 *
 * Channel-last, x-first coordinate convention:
 *   inp  : (*batch, *inshape,  C)
 *   grid : (*batch, *outshape, D)   with D == the spatial rank (1, 2 or 3)
 *   out  : (*batch, *outshape, C)
 *
 * @param out          Output tensor (pulled samples)
 * @param inp          Input volume (spline coefficients)
 * @param grid         Sampling coordinates (in voxels, x-first)
 * @param spline       Spline order applied to every spatial dim
 * @param bound        Boundary condition applied to every spatial dim
 * @param extrapolate  1: always; 0: not past voxel centres; -1: not past edges
 * @param stream       Cuda stream on which to operate
 */
void pull(
          DLTensor & out,
    const DLTensor & inp,
    const DLTensor & grid,
          int8_t     spline      = spline_t::Quadratic,
          int8_t     bound       = bound_t::DCT2,
          int8_t     extrapolate = 1,
          intptr_t   stream      = 0
);

/**
 * @brief Splat ("push") values into a volume; numerical adjoint of `pull`.
 *        `out` must be pre-zeroed by the caller (values are accumulated).
 *
 *   inp  : (*batch, *outshape, C)
 *   grid : (*batch, *outshape, D)
 *   out  : (*batch, *inshape,  C)
 */
void push(
          DLTensor & out,
    const DLTensor & inp,
    const DLTensor & grid,
          int8_t     spline      = spline_t::Quadratic,
          int8_t     bound       = bound_t::DCT2,
          int8_t     extrapolate = 1,
          intptr_t   stream      = 0
);

/**
 * @brief Splat ones (== push of an all-ones input). `out` (*batch,*inshape,1)
 *        must be pre-zeroed.
 */
void count(
          DLTensor & out,
    const DLTensor & grid,
          int8_t     spline      = spline_t::Quadratic,
          int8_t     bound       = bound_t::DCT2,
          int8_t     extrapolate = 1,
          intptr_t   stream      = 0
);

/**
 * @brief Sample the spatial gradients of a spline-encoded volume.
 *
 *   inp  : (*batch, *inshape,  C)
 *   grid : (*batch, *outshape, D)
 *   out  : (*batch, *outshape, C, D)
 */
void grad(
          DLTensor & out,
    const DLTensor & inp,
    const DLTensor & grid,
          int8_t     spline      = spline_t::Quadratic,
          int8_t     bound       = bound_t::DCT2,
          int8_t     extrapolate = 1,
          bool       abs         = false,
          intptr_t   stream      = 0
);

/***********************************************************************
 *                            BACKWARD PASSES                          *
 ***********************************************************************
 *
 * Adjoints of the four sampling ops above, with respect to *both* the
 * sampled field and the sampling coordinates.  They exist so that a
 * higher layer (e.g. `fastfields.torch`'s autograd Functions) can
 * differentiate through `grid`, which the plain `push`/`pull` adjoint
 * pair cannot express.
 *
 * Convention for every function below:
 *   * `out`  (gradient wrt the forward `inp`) is **accumulated** into
 *     whenever the op scatters (`pull_backward`) and **overwritten**
 *     when it gathers (`push_backward`, `grad_backward` scatters).  The
 *     safe rule for callers is: always pre-zero `out`.
 *   * `gout` (gradient wrt `grid`) is **overwritten**, never accumulated
 *     -- one grid point maps to exactly one output element.
 *   * `ginp` is the incoming gradient, i.e. the gradient with respect to
 *     the *output* of the corresponding forward op, and therefore has
 *     that output's shape.
 *   * Coordinates that fall outside the field of view (per `extrapolate`)
 *     contribute nothing and get a zero `gout`.
 *
 * Unlike `grad`, the `pull`/`push`/`count` adjoints take no `abs` flag:
 * `abs` replaces the signed spline derivative by its absolute value, a
 * majorisation trick that only makes sense for the `grad` operator
 * itself.  A true adjoint always needs the signed derivative, so these
 * three are instantiated with `abs = false`.
 */

/**
 * @brief Adjoint of `pull` wrt both `inp` and `grid`.
 *
 *   out  : (*batch, *inshape,  C)   gradient wrt `inp`   (pre-zero; accumulated)
 *   gout : (*batch, *outshape, D)   gradient wrt `grid`  (overwritten)
 *   inp  : (*batch, *inshape,  C)   forward input volume
 *   ginp : (*batch, *outshape, C)   gradient wrt the output of `pull`
 *   grid : (*batch, *outshape, D)   forward sampling coordinates
 */
void pull_backward(
          DLTensor & out,
          DLTensor & gout,
    const DLTensor & inp,
    const DLTensor & ginp,
    const DLTensor & grid,
          int8_t     spline      = spline_t::Quadratic,
          int8_t     bound       = bound_t::DCT2,
          int8_t     extrapolate = 1,
          intptr_t   stream      = 0
);

/**
 * @brief Adjoint of `push` wrt both `inp` and `grid`.
 *
 *   out  : (*batch, *outshape, C)   gradient wrt `inp`   (overwritten)
 *   gout : (*batch, *outshape, D)   gradient wrt `grid`  (overwritten)
 *   inp  : (*batch, *outshape, C)   forward input (the splatted values)
 *   ginp : (*batch, *inshape,  C)   gradient wrt the output of `push`
 *   grid : (*batch, *outshape, D)   forward sampling coordinates
 */
void push_backward(
          DLTensor & out,
          DLTensor & gout,
    const DLTensor & inp,
    const DLTensor & ginp,
    const DLTensor & grid,
          int8_t     spline      = spline_t::Quadratic,
          int8_t     bound       = bound_t::DCT2,
          int8_t     extrapolate = 1,
          intptr_t   stream      = 0
);

/**
 * @brief Adjoint of `count` wrt `grid` (its only input).
 *
 *   gout : (*batch, *outshape, D)   gradient wrt `grid`  (overwritten)
 *   ginp : (*batch, *inshape,  1)   gradient wrt the output of `count`
 *   grid : (*batch, *outshape, D)   forward sampling coordinates
 */
void count_backward(
          DLTensor & gout,
    const DLTensor & ginp,
    const DLTensor & grid,
          int8_t     spline      = spline_t::Quadratic,
          int8_t     bound       = bound_t::DCT2,
          int8_t     extrapolate = 1,
          intptr_t   stream      = 0
);

/**
 * @brief Adjoint of `grad` wrt both `inp` and `grid`.
 *
 *   out  : (*batch, *inshape,  C)      gradient wrt `inp`  (pre-zero; accumulated)
 *   gout : (*batch, *outshape, D)      gradient wrt `grid` (overwritten)
 *   inp  : (*batch, *inshape,  C)      forward input volume
 *   ginp : (*batch, *outshape, C, D)   gradient wrt the output of `grad`
 *   grid : (*batch, *outshape, D)      forward sampling coordinates
 *
 * @param abs  Must match the `abs` used by the corresponding `grad` call.
 */
void grad_backward(
          DLTensor & out,
          DLTensor & gout,
    const DLTensor & inp,
    const DLTensor & ginp,
    const DLTensor & grid,
          int8_t     spline      = spline_t::Quadratic,
          int8_t     bound       = bound_t::DCT2,
          int8_t     extrapolate = 1,
          bool       abs         = false,
          intptr_t   stream      = 0
);

FF_NAMESPACE_END(FF_NS)

#endif // FF_LIB_PUSHPULL
