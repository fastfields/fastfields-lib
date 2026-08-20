#include <stdexcept>
#include <vector>
#include <cmath>
#include "fastfields/api/cpu/solve_field.h"
#include "fastfields/api/cpu/reg_field.h"
#include "fastfields/api/cpu/posdef.h"
#include "fastfields/core/autocast.h"
#include "fastfields/core/dispatch.h"
#include "fastfields/core/dlpack.h"
#include "fastfields/core/cuda_switch.h"
#include "fastfields/core/utils.h"
#include "fastfields/impl/cpu/solve_field.h"

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)

typedef double reduce_t;

/***********************************************************************
 *                        VECTOR PRIMITIVES                            *
 ***********************************************************************/

namespace {

// Number of elements described by a tensor's shape.
static inline size_t tensor_numel(const DLTensor & t)
{
    size_t numel = 1;
    for (int32_t d = 0; d < t.ndim; ++d)
        numel *= static_cast<size_t>(t.shape[d]);
    return numel;
}

// A contiguous work buffer shaped like a reference tensor. CG needs four of
// them (`r`, `z`, `p`, `Ap`) plus one for the regulariser diagonal; they are
// owned for the duration of the solve and freed on the way out, which follows
// `field_precond`'s existing "materialise scratch into a std::vector<uint8_t>
// and wrap it in a DLTensor" pattern rather than inventing an allocator.
//
// `shape` is *borrowed* from the reference tensor, so a Scratch must not
// outlive the tensor it was built from (they are all locals of `field_cg`).
struct Scratch {
    std::vector<uint8_t> buf;
    std::vector<int64_t> strides;
    DLTensor             t;

    explicit Scratch(const DLTensor & like)
        : buf(tensor_numel(like) * static_cast<size_t>(like.dtype.bits) / 8, 0),
          strides(static_cast<size_t>(like.ndim > 0 ? like.ndim : 1), 1)
    {
        int64_t s = 1;
        for (int32_t d = like.ndim - 1; d >= 0; --d) {
            strides[static_cast<size_t>(d)] = s;
            s *= like.shape[d];
        }
        t.data        = buf.data();
        t.device      = like.device;
        t.ndim        = like.ndim;
        t.dtype       = like.dtype;
        t.shape       = like.shape;
        t.strides     = strides.data();
        t.byte_offset = 0;
    }

    Scratch(const Scratch &)             = delete;
    Scratch & operator=(const Scratch &) = delete;
};

template <typename scalar_t, typename offset_t>
inline reduce_t _dot(
          int64_t   nall     ,
    const void    * x        ,
    const void    * y        ,
    const int64_t * size     ,
    const int64_t * stride_x ,
    const int64_t * stride_y )
{
    const int64_t nall1 = nall + 1;
    const offset_t * _size     = copy_if_needed<offset_t *>(size,     nall1);
    const offset_t * _stride_x = copy_if_needed<offset_t *>(stride_x, nall1);
    const offset_t * _stride_y = copy_if_needed<offset_t *>(stride_y, nall1);

    reduce_t out = solve_field::dot<reduce_t, scalar_t, offset_t>(
        static_cast<offset_t>(nall),
        static_cast<const scalar_t *>(x),
        static_cast<const scalar_t *>(y),
        _size, _stride_x, _stride_y);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_x);
    free_if_needed<int64_t *>(_stride_y);
    return out;
}

template <typename scalar_t, typename offset_t>
inline void _axpby_(
          int64_t   nall     ,
          void    * y        ,
    const void    * x        ,
          reduce_t  a        ,
          reduce_t  b        ,
    const int64_t * size     ,
    const int64_t * stride_y ,
    const int64_t * stride_x )
{
    const int64_t nall1 = nall + 1;
    const offset_t * _size     = copy_if_needed<offset_t *>(size,     nall1);
    const offset_t * _stride_y = copy_if_needed<offset_t *>(stride_y, nall1);
    const offset_t * _stride_x = copy_if_needed<offset_t *>(stride_x, nall1);

    solve_field::axpby_<reduce_t, scalar_t, offset_t>(
        static_cast<offset_t>(nall),
        static_cast<      scalar_t *>(y),
        static_cast<const scalar_t *>(x),
        a, b, _size, _stride_y, _stride_x);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_y);
    free_if_needed<int64_t *>(_stride_x);
}

// dtype/offset dispatch for the two primitives. Both operands are always
// field-shaped `(*batch, *spatial, C)` tensors of the same dtype, so `nall`
// (everything but the channel axis) and the shape come from the first one.
#define SOLVE_DT_SWITCH(CALL)                                                \
    switch (static_cast<DLDataTypeCode>(x.dtype.code)) {                     \
        case kDLFloat: switch (x.dtype.bits) {                               \
            case 32: return use_32bits ? CALL(float,  int32_t)               \
                                       : CALL(float,  int64_t);              \
            case 64: return use_32bits ? CALL(double, int32_t)               \
                                       : CALL(double, int64_t);              \
            default: break;                                                  \
        } break;                                                             \
        default: break;                                                      \
    }                                                                        \
    throw std::invalid_argument("only floating point data types are supported");

static inline reduce_t dot(const DLTensor & x, const DLTensor & y)
{
    const int64_t nall       = static_cast<int64_t>(x.ndim) - 1;
    const bool    use_32bits = FF_CANUSE32BITS(x) && FF_CANUSE32BITS(y);
#define DOT_CALL(SCALAR, OFFSET) \
    _dot<SCALAR, OFFSET>(nall, FF_CVOIDPTR(x), FF_CVOIDPTR(y), x.shape, x.strides, y.strides)
    SOLVE_DT_SWITCH(DOT_CALL)
#undef DOT_CALL
}

// y = a * x + b * y
static inline void axpby_(DLTensor & y, const DLTensor & x,
                          reduce_t a, reduce_t b)
{
    const int64_t nall       = static_cast<int64_t>(x.ndim) - 1;
    const bool    use_32bits = FF_CANUSE32BITS(x) && FF_CANUSE32BITS(y);
#define AXPBY_CALL(SCALAR, OFFSET)                                     \
    _axpby_<SCALAR, OFFSET>(nall, FF_VOIDPTR(y), FF_CVOIDPTR(x), a, b, \
                            x.shape, y.strides, x.strides)
    SOLVE_DT_SWITCH(AXPBY_CALL)
#undef AXPBY_CALL
}

} // anonymous namespace

/***********************************************************************
 *                        CONJUGATE GRADIENTS                          *
 ***********************************************************************/

void field_cg(
          DLTensor & sol_        ,
    const DLTensor & hes         ,
    const DLTensor & grd_        ,
    const double   * voxel_size  ,
    const double   * absolute    ,
    const double   * membrane    ,
    const double   * bending     ,
          int8_t     bound       ,
          int        ndim        ,
          int        nb_iter     ,
          double     tol         ,
          int      * nb_iter_out ,
          double   * residual_out,
          intptr_t   stream      )
{
    // Normalise NULL strides (compact row-major) before anything reads them.
    ContiguousStrides _sol(sol_), _grd(grd_);
    DLTensor       & sol = _sol.t;
    const DLTensor & grd = _grd.t;

    FF_CHECK_NO_LANES  (sol)
    FF_CHECK_SAME_DTYPE(sol, grd)
    FF_CHECK_SAME_DTYPE(sol, hes)
    FF_CHECK_SAME      (sol.ndim, grd.ndim, "Tensors do not have the same number of dimensions")
    FF_CHECK_SAME      (sol.ndim, hes.ndim, "Tensors do not have the same number of dimensions")
    if (sol.ndim - ndim - 1 < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    FF_CHECK_SAME_SHAPE_N(sol, grd, sol.ndim)

    const int64_t nc = sol.shape[sol.ndim - 1];
    FF_CHECK_SAME(hes.shape[hes.ndim - 1], nc * (nc + 1) / 2,
               "The Hessian's trailing dimension must be C*(C+1)/2")

    if (nb_iter_out)  *nb_iter_out  = 0;
    if (residual_out) *residual_out = 0.;
    if (tensor_numel(sol) == 0) return;

    // Work buffers: residual, preconditioned residual, search direction, and
    // the operator applied to it -- plus the regulariser diagonal, which is
    // hoisted out of the loop because it depends only on the penalties, not on
    // the iterate (this is what makes a CG step cheaper than calling
    // `field_precond`, which recomputes `field_diag` every time).
    Scratch r(sol), z(sol), p(sol), ap(sol), diag(sol);

    field_diag(diag.t, voxel_size, absolute, membrane, bending, bound, ndim, stream);

    // r = g - (H + L) x0
    field_forward(r.t, hes, sol, voxel_size, absolute, membrane, bending,
                  bound, ndim, stream);
    axpby_(r.t, grd, 1.0, -1.0);

    // Residuals are reported (and tested) relative to ||g||; a zero
    // right-hand side falls back to an absolute threshold so that the ratio
    // stays well defined.
    const reduce_t gnorm = std::sqrt(dot(grd, grd));
    const reduce_t scale = gnorm > 0 ? gnorm : reduce_t(1);
          reduce_t rnorm = std::sqrt(dot(r.t, r.t));

    if (residual_out) *residual_out = rnorm / scale;
    if (nb_iter <= 0 || rnorm == 0 || (tol > 0 && rnorm <= tol * scale))
        return;

    // z = M^-1 r, with M = H + diag(L) (Jacobi / block-Jacobi over channels)
    sym_solve(z.t, hes, r.t, diag.t, stream);
    axpby_(p.t, z.t, 1.0, 0.0);             // p = z
    reduce_t rz = dot(r.t, z.t);

    // `rz` is `<r, M^-1 r>`, which is strictly positive for as long as the
    // preconditioner M = H + diag(L) is positive definite and `r` is non-zero.
    // Zero means the residual vanished; negative means M is not positive
    // definite, at which point CG's step lengths stop being descent steps --
    // both are reasons to stop rather than to keep iterating.
    int it = 0;
    while (it < nb_iter && rz > 0)
    {
        field_forward(ap.t, hes, p.t, voxel_size, absolute, membrane, bending,
                      bound, ndim, stream);

        const reduce_t pap = dot(p.t, ap.t);
        // Non-positive curvature means (H + L) is not positive definite along
        // p -- CG has no valid step length, so stop with what we have instead
        // of dividing by it.
        if (!(pap > 0)) break;

        const reduce_t alpha = rz / pap;
        axpby_(sol,  p.t,  alpha, 1.0);     // x += alpha * p
        axpby_(r.t,  ap.t, -alpha, 1.0);    // r -= alpha * A p
        ++it;

        rnorm = std::sqrt(dot(r.t, r.t));
        if (tol > 0 && rnorm <= tol * scale) break;
        if (it >= nb_iter) break;

        sym_solve(z.t, hes, r.t, diag.t, stream);
        const reduce_t rz_new = dot(r.t, z.t);
        const reduce_t beta   = rz_new / rz;
        axpby_(p.t, z.t, 1.0, beta);        // p = z + beta * p
        rz = rz_new;
    }

    if (nb_iter_out)  *nb_iter_out  = it;
    if (residual_out) *residual_out = rnorm / scale;
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
