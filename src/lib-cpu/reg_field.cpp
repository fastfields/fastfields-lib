#include <stdexcept>
#include <cstdint>
#include <string>
#include <vector>
#include "fastfields/api/cpu/reg_field.h"
#include "fastfields/api/cpu/posdef.h"
#include "fastfields/api/dispatch.h"
#include "fastfields/core/autocast.h"
#include "fastfields/core/dlpack.h"
#include "fastfields/core/cuda_switch.h"
#include "fastfields/impl/kernels/bounds.h"
#include "fastfields/impl/kernels/utils.h"
#include "fastfields/impl/cpu/reg_field.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)

#define VOIDPTR(x)      (static_cast<void*>(static_cast<char*>(x.data) + x.byte_offset))
#define CVOIDPTR(x)     (static_cast<const void*>(static_cast<const char*>(x.data) + x.byte_offset))
#define CANUSE32BITS(x) (canUse32BitIndexMath(x.ndim, x.shape, x.strides))

typedef double reduce_t;

/***********************************************************************
 *                              CHECKS                                 *
 ***********************************************************************/

#define CHECK_NO_LANES(tensor)                                          \
    if (tensor.dtype.lanes > 1)                                         \
        throw std::invalid_argument("Only scalar data types are supported");

#define CHECK_SAME(X, Y, msg)                                           \
    if (X != Y) throw std::invalid_argument(msg);

#define CHECK_SAME_DTYPE(X, Y)                                          \
    if ((X.dtype.code  != Y.dtype.code) ||                              \
        (X.dtype.bits  != Y.dtype.bits) ||                             \
        (X.dtype.lanes != Y.dtype.lanes))                              \
        throw std::invalid_argument("Tensors do not have the same data type");

#define CHECK_SAME_SHAPE(X, Y, D)                                       \
    for (int32_t d=0; d < D; ++d)                                       \
        if (X.shape[d] != Y.shape[d])                                   \
            throw std::invalid_argument("Tensors do not have the same shape");

/***********************************************************************
 *                             WRAPPERS                                *
 ***********************************************************************/

namespace {

// build a length-nc reduce_t vector from a (possibly null) double array
static inline std::vector<reduce_t> as_weights(const double * w, int64_t nc)
{
    std::vector<reduce_t> v(static_cast<size_t>(nc), reduce_t(0));
    if (w) for (int64_t c = 0; c < nc; ++c) v[static_cast<size_t>(c)] = w[c];
    return v;
}

// Accumulate variant of _field_matvec: out += L(inp) (op='+') or out -= L(inp)
// (op='-'), instead of overwriting out. The impl-layer matvec_* templates
// already take the op char (see op_apply, issue #6a); this just threads a
// compile-time '+'/'-' instead of the '=' _field_matvec hardcodes.
template <char op>
struct field_matvec_op
{
    template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
    static void run(
        const bound::BoundVec & bvec,
              int64_t   nbatch     ,
              int64_t   nc         ,
              void    * out        ,
        const void    * inp        ,
        const double  * voxel_size ,
        const double  * absolute   ,
        const double  * membrane   ,
        const double  * bending    ,
        const int64_t * size       ,
        const int64_t * stride_out ,
        const int64_t * stride_inp )
    {
        const size_t nall1 = static_cast<size_t>(nbatch + ndim + 1);
        IndexArray<offset_t> _size       (size,       nall1);
        IndexArray<offset_t> _stride_out (stride_out, nall1);
        IndexArray<offset_t> _stride_inp (stride_inp, nall1);
              scalar_t * _out = static_cast<      scalar_t *>(out);
        const scalar_t * _inp = static_cast<const scalar_t *>(inp);

        reduce_t vx[ndim];
        for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

        std::vector<reduce_t> a = as_weights(absolute, nc);
        std::vector<reduce_t> m = as_weights(membrane, nc);
        std::vector<reduce_t> b = as_weights(bending,  nc);

        if (bending)
            reg_field::matvec_bending<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _out, _inp,
                _size, _stride_out, _stride_inp, vx, a.data(), m.data(), b.data());
        else if (membrane)
            reg_field::matvec_membrane<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _out, _inp,
                _size, _stride_out, _stride_inp, vx, a.data(), m.data());
        else
            reg_field::matvec_absolute<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _out, _inp,
                _size, _stride_out, _stride_inp, a.data());

    }
};

template <char op>
struct field_diag_op
{
    template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
    static void run(
        const bound::BoundVec & bvec,
              int64_t   nbatch     ,
              int64_t   nc         ,
              void    * out        ,
        const double  * voxel_size ,
        const double  * absolute   ,
        const double  * membrane   ,
        const double  * bending    ,
        const int64_t * size       ,
        const int64_t * stride_out )
    {
        const size_t nall1 = static_cast<size_t>(nbatch + ndim + 1);
        IndexArray<offset_t> _size       (size,       nall1);
        IndexArray<offset_t> _stride_out (stride_out, nall1);
              scalar_t * _out = static_cast<scalar_t *>(out);

        reduce_t vx[ndim];
        for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

        std::vector<reduce_t> a = as_weights(absolute, nc);
        std::vector<reduce_t> m = as_weights(membrane, nc);
        std::vector<reduce_t> b = as_weights(bending,  nc);

        if (bending)
            reg_field::diag_bending<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _out,
                _size, _stride_out, vx, a.data(), m.data(), b.data());
        else if (membrane)
            reg_field::diag_membrane<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _out,
                _size, _stride_out, vx, a.data(), m.data());
        else
            reg_field::diag_absolute<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _out,
                _size, _stride_out, a.data());

    }
};

// Materialise the Toeplitz convolution kernel (stencil) of the operator. The
// output is a per-channel vector stencil (*batch, *spatial, C) -- the field
// regulariser never couples channels, so there is no matrix case. Dispatch
// mirrors _field_diag: the highest-order non-null penalty selects the stencil.
template <char op>
struct field_kernel_op
{
    template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
    static void run(
        const bound::BoundVec & bvec,
              int64_t   nbatch     ,
              int64_t   nc         ,
              void    * out        ,
        const double  * voxel_size ,
        const double  * absolute   ,
        const double  * membrane   ,
        const double  * bending    ,
        const int64_t * size       ,
        const int64_t * stride_out )
    {
        const size_t nall1 = static_cast<size_t>(nbatch + ndim + 1);
        IndexArray<offset_t> _size       (size,       nall1);
        IndexArray<offset_t> _stride_out (stride_out, nall1);
              scalar_t * _out = static_cast<scalar_t *>(out);

        reduce_t vx[ndim];
        for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

        std::vector<reduce_t> a = as_weights(absolute, nc);
        std::vector<reduce_t> m = as_weights(membrane, nc);
        std::vector<reduce_t> b = as_weights(bending,  nc);

        if (bending)
            reg_field::kernel_bending<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _out,
                _size, _stride_out, vx, a.data(), m.data(), b.data());
        else if (membrane)
            reg_field::kernel_membrane<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _out,
                _size, _stride_out, vx, a.data(), m.data());
        else
            // kernel_absolute takes no voxel_size (the L2 stencil is scale-free).
            reg_field::kernel_absolute<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _out,
                _size, _stride_out, a.data());

    }
};

// One or more relaxation (Gauss-Seidel) sweeps solving `(H + L) x = g` in
// place, where H is the per-voxel compact-symmetric Hessian, L the field
// regulariser, and x the warm-started `sol`. Dispatches to the impl relaxer
// matching the highest-order penalty (membrane covers the absolute-only case).
struct field_relax_op
{
    template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
    static void run(
        const bound::BoundVec & bvec,
              int64_t   nbatch     ,
              int64_t   nc         ,
              void    * sol        ,
        const void    * hes        ,
        const void    * grd        ,
        const double  * voxel_size ,
        const double  * absolute   ,
        const double  * membrane   ,
        const double  * bending    ,
              int       niter      ,
        const int64_t * size       ,
        const int64_t * stride_sol ,
        const int64_t * stride_hes ,
        const int64_t * stride_grd )
    {
        const size_t nall1 = static_cast<size_t>(nbatch + ndim + 1);
        IndexArray<offset_t> _size       (size,       nall1);
        IndexArray<offset_t> _stride_sol (stride_sol, nall1);
        IndexArray<offset_t> _stride_hes (stride_hes, nall1);
        IndexArray<offset_t> _stride_grd (stride_grd, nall1);
              scalar_t * _sol = static_cast<      scalar_t *>(sol);
        const scalar_t * _hes = static_cast<const scalar_t *>(hes);
        const scalar_t * _grd = static_cast<const scalar_t *>(grd);

        reduce_t vx[ndim];
        for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

        std::vector<reduce_t> a = as_weights(absolute, nc);
        std::vector<reduce_t> m = as_weights(membrane, nc);
        std::vector<reduce_t> b = as_weights(bending,  nc);

        if (bending)
            reg_field::relax_bending_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd,
                _size, _stride_sol, _stride_hes, _stride_grd, vx,
                a.data(), m.data(), b.data(), niter);
        else
            reg_field::relax_membrane_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
                bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd,
                _size, _stride_sol, _stride_hes, _stride_grd, vx,
                a.data(), m.data(), niter);

    }
};

// Reweighted-least-squares (RLS/JRLS) variant of `_field_matvec`: an extra
// per-voxel weight map `wgt` modulates the penalty strength. `is_jrls`
// selects between the per-channel-weight (RLS) and single-shared-weight
// (JRLS) impl variants -- both take an identical argument list, so the
// dispatch is a plain runtime branch rather than a template parameter.
struct field_matvec_rls_op
{
    template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
    static void run(
        const bound::BoundVec & bvec,
              int64_t   nbatch     ,
              int64_t   nc         ,
              bool      is_jrls    ,
              void    * out        ,
        const void    * inp        ,
        const void    * wgt        ,
        const double  * voxel_size ,
        const double  * absolute   ,
        const double  * membrane   ,
        const double  * bending    ,
        const int64_t * size       ,
        const int64_t * stride_out ,
        const int64_t * stride_inp ,
        const int64_t * stride_wgt )
    {
        const size_t nall1 = static_cast<size_t>(nbatch + ndim + 1);
        IndexArray<offset_t> _size       (size,       nall1);
        IndexArray<offset_t> _stride_out (stride_out, nall1);
        IndexArray<offset_t> _stride_inp (stride_inp, nall1);
        IndexArray<offset_t> _stride_wgt (stride_wgt, nall1);
              scalar_t * _out = static_cast<      scalar_t *>(out);
        const scalar_t * _inp = static_cast<const scalar_t *>(inp);
        const scalar_t * _wgt = static_cast<const scalar_t *>(wgt);

        reduce_t vx[ndim];
        for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

        std::vector<reduce_t> a = as_weights(absolute, nc);
        std::vector<reduce_t> m = as_weights(membrane, nc);
        std::vector<reduce_t> b = as_weights(bending,  nc);

        if (bending) {
            if (is_jrls)
                reg_field::matvec_bending_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _out, _inp, _wgt,
                    _size, _stride_out, _stride_inp, _stride_wgt, vx, a.data(), m.data(), b.data());
            else
                reg_field::matvec_bending_rls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _out, _inp, _wgt,
                    _size, _stride_out, _stride_inp, _stride_wgt, vx, a.data(), m.data(), b.data());
        } else if (membrane) {
            if (is_jrls)
                reg_field::matvec_membrane_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _out, _inp, _wgt,
                    _size, _stride_out, _stride_inp, _stride_wgt, vx, a.data(), m.data());
            else
                reg_field::matvec_membrane_rls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _out, _inp, _wgt,
                    _size, _stride_out, _stride_inp, _stride_wgt, vx, a.data(), m.data());
        } else {
            if (is_jrls)
                reg_field::matvec_absolute_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _out, _inp, _wgt,
                    _size, _stride_out, _stride_inp, _stride_wgt, a.data());
            else
                reg_field::matvec_absolute_rls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _out, _inp, _wgt,
                    _size, _stride_out, _stride_inp, _stride_wgt, a.data());
        }

    }
};

struct field_diag_rls_op
{
    template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
    static void run(
        const bound::BoundVec & bvec,
              int64_t   nbatch     ,
              int64_t   nc         ,
              bool      is_jrls    ,
              void    * out        ,
        const void    * wgt        ,
        const double  * voxel_size ,
        const double  * absolute   ,
        const double  * membrane   ,
        const double  * bending    ,
        const int64_t * size       ,
        const int64_t * stride_out ,
        const int64_t * stride_wgt )
    {
        const size_t nall1 = static_cast<size_t>(nbatch + ndim + 1);
        IndexArray<offset_t> _size       (size,       nall1);
        IndexArray<offset_t> _stride_out (stride_out, nall1);
        IndexArray<offset_t> _stride_wgt (stride_wgt, nall1);
              scalar_t * _out = static_cast<scalar_t *>(out);
        const scalar_t * _wgt = static_cast<const scalar_t *>(wgt);

        reduce_t vx[ndim];
        for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

        std::vector<reduce_t> a = as_weights(absolute, nc);
        std::vector<reduce_t> m = as_weights(membrane, nc);
        std::vector<reduce_t> b = as_weights(bending,  nc);

        if (bending) {
            if (is_jrls)
                reg_field::diag_bending_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _out, _wgt,
                    _size, _stride_out, _stride_wgt, vx, a.data(), m.data(), b.data());
            else
                reg_field::diag_bending_rls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _out, _wgt,
                    _size, _stride_out, _stride_wgt, vx, a.data(), m.data(), b.data());
        } else if (membrane) {
            if (is_jrls)
                reg_field::diag_membrane_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _out, _wgt,
                    _size, _stride_out, _stride_wgt, vx, a.data(), m.data());
            else
                reg_field::diag_membrane_rls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _out, _wgt,
                    _size, _stride_out, _stride_wgt, vx, a.data(), m.data());
        } else {
            if (is_jrls)
                reg_field::diag_absolute_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _out, _wgt,
                    _size, _stride_out, _stride_wgt, a.data());
            else
                reg_field::diag_absolute_rls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _out, _wgt,
                    _size, _stride_out, _stride_wgt, a.data());
        }

    }
};

struct field_relax_rls_op
{
    template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
    static void run(
        const bound::BoundVec & bvec,
              int64_t   nbatch     ,
              int64_t   nc         ,
              bool      is_jrls    ,
              void    * sol        ,
        const void    * hes        ,
        const void    * grd        ,
        const void    * wgt        ,
        const double  * voxel_size ,
        const double  * absolute   ,
        const double  * membrane   ,
        const double  * bending    ,
              int       niter      ,
        const int64_t * size       ,
        const int64_t * stride_sol ,
        const int64_t * stride_hes ,
        const int64_t * stride_grd ,
        const int64_t * stride_wgt )
    {
        const size_t nall1 = static_cast<size_t>(nbatch + ndim + 1);
        IndexArray<offset_t> _size       (size,       nall1);
        IndexArray<offset_t> _stride_sol (stride_sol, nall1);
        IndexArray<offset_t> _stride_hes (stride_hes, nall1);
        IndexArray<offset_t> _stride_grd (stride_grd, nall1);
        IndexArray<offset_t> _stride_wgt (stride_wgt, nall1);
              scalar_t * _sol = static_cast<      scalar_t *>(sol);
        const scalar_t * _hes = static_cast<const scalar_t *>(hes);
        const scalar_t * _grd = static_cast<const scalar_t *>(grd);
        const scalar_t * _wgt = static_cast<const scalar_t *>(wgt);

        reduce_t vx[ndim];
        for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

        std::vector<reduce_t> a = as_weights(absolute, nc);
        std::vector<reduce_t> m = as_weights(membrane, nc);
        std::vector<reduce_t> b = as_weights(bending,  nc);

        if (bending) {
            if (is_jrls)
                reg_field::relax_bending_jrls_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd, _wgt,
                    _size, _stride_sol, _stride_hes, _stride_grd, _stride_wgt, vx,
                    a.data(), m.data(), b.data(), niter);
            else
                reg_field::relax_bending_rls_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd, _wgt,
                    _size, _stride_sol, _stride_hes, _stride_grd, _stride_wgt, vx,
                    a.data(), m.data(), b.data(), niter);
        } else {
            if (is_jrls)
                reg_field::relax_membrane_jrls_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd, _wgt,
                    _size, _stride_sol, _stride_hes, _stride_grd, _stride_wgt, vx,
                    a.data(), m.data(), niter);
            else
                reg_field::relax_membrane_rls_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
                    bvec, static_cast<offset_t>(nbatch), _sol, _hes, _grd, _wgt,
                    _size, _stride_sol, _stride_hes, _stride_grd, _stride_wgt, vx,
                    a.data(), m.data(), niter);
        }

    }
};

} // anonymous namespace

/***********************************************************************
 *                            DISPATCH                                 *
 ***********************************************************************/

//
// `dispatch_nbd<Op>(key, ...)` (api/dispatch.h) is the entire pyramid:
// ndim (1/2/3) x boundary condition (8, each expanded to a pack of `ndim`
// copies) x dtype (f32/f64) x index width. It replaces the twelve `<OP>_DT`
// macros, `BOUND_SWITCH`, `BND1`/`BND2`/`BND3`, `NDIM_SWITCH` and the six
// `#define <OP>_ARGS ... #undef` pairs this file used to carry.

void field_matvec(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _inp(inp_);
    DLTensor       & out = _out.t;
    const DLTensor & inp = _inp.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME      (out.ndim, inp.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME_SHAPE(out, inp, out.ndim)

    const int64_t    nc = out.shape[out.ndim - 1];
    const bound::BoundVec bvec(static_cast<bound::type>(bound));
    const DispatchKey key(ndim, bound, out.dtype,
                          CANUSE32BITS(out) && CANUSE32BITS(inp));

    dispatch_nbd<field_matvec_op<'='> >(key,
        bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(out), CVOIDPTR(inp), voxel_size, absolute, membrane, bending, out.shape, out.strides, inp.strides);
}

/**
 * @brief `field_matvec` variant that accumulates: `out += L(inp)`.
 */
void field_addmatvec_(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _inp(inp_);
    DLTensor       & out = _out.t;
    const DLTensor & inp = _inp.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME      (out.ndim, inp.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME_SHAPE(out, inp, out.ndim)

    const int64_t    nc = out.shape[out.ndim - 1];
    const bound::BoundVec bvec(static_cast<bound::type>(bound));
    const DispatchKey key(ndim, bound, out.dtype,
                          CANUSE32BITS(out) && CANUSE32BITS(inp));

    dispatch_nbd<field_matvec_op<'+'> >(key,
        bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(out), CVOIDPTR(inp), voxel_size, absolute, membrane, bending, out.shape, out.strides, inp.strides);
}

/**
 * @brief `field_matvec` variant that subtracts: `out -= L(inp)`.
 */
void field_submatvec_(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _inp(inp_);
    DLTensor       & out = _out.t;
    const DLTensor & inp = _inp.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME      (out.ndim, inp.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME_SHAPE(out, inp, out.ndim)

    const int64_t    nc = out.shape[out.ndim - 1];
    const bound::BoundVec bvec(static_cast<bound::type>(bound));
    const DispatchKey key(ndim, bound, out.dtype,
                          CANUSE32BITS(out) && CANUSE32BITS(inp));

    dispatch_nbd<field_matvec_op<'-'> >(key,
        bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(out), CVOIDPTR(inp), voxel_size, absolute, membrane, bending, out.shape, out.strides, inp.strides);
}

void field_diag(
          DLTensor & out_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_);
    DLTensor & out = _out.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES(out)
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");

    const int64_t    nc = out.shape[out.ndim - 1];
    const bound::BoundVec bvec(static_cast<bound::type>(bound));
    const DispatchKey key(ndim, bound, out.dtype,
                          CANUSE32BITS(out));

    dispatch_nbd<field_diag_op<'='> >(key,
        bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(out), voxel_size, absolute, membrane, bending, out.shape, out.strides);
}

/**
 * @brief `field_diag` variant that accumulates: `out += diag(L)`.
 *
 * In-place only, matching the jitfields `op='+'` C-level entry point: the
 * caller owns `out` and this reads-modifies-writes it. An out-of-place
 * "return a fresh tensor" form is a caller-side clone, not a second kernel.
 */
void field_adddiag_(
          DLTensor & out_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_);
    DLTensor & out = _out.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES(out)
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");

    const int64_t    nc = out.shape[out.ndim - 1];
    const bound::BoundVec bvec(static_cast<bound::type>(bound));
    const DispatchKey key(ndim, bound, out.dtype,
                          CANUSE32BITS(out));

    dispatch_nbd<field_diag_op<'+'> >(key,
        bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(out), voxel_size, absolute, membrane, bending, out.shape, out.strides);
}

/**
 * @brief `field_diag` variant that subtracts: `out -= diag(L)`. In-place only.
 */
void field_subdiag_(
          DLTensor & out_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_);
    DLTensor & out = _out.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES(out)
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");

    const int64_t    nc = out.shape[out.ndim - 1];
    const bound::BoundVec bvec(static_cast<bound::type>(bound));
    const DispatchKey key(ndim, bound, out.dtype,
                          CANUSE32BITS(out));

    dispatch_nbd<field_diag_op<'-'> >(key,
        bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(out), voxel_size, absolute, membrane, bending, out.shape, out.strides);
}

void field_kernel(
          DLTensor & out_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_);
    DLTensor & out = _out.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES(out)
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");

    const int64_t    nc = out.shape[out.ndim - 1];
    const bound::BoundVec bvec(static_cast<bound::type>(bound));
    const DispatchKey key(ndim, bound, out.dtype,
                          CANUSE32BITS(out));

    dispatch_nbd<field_kernel_op<'='> >(key,
        bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(out), voxel_size, absolute, membrane, bending, out.shape, out.strides);
}

/**
 * @brief `field_kernel` variant that accumulates the stencil: `out += K`.
 *        In-place only (jitfields `op='+'`).
 */
void field_addkernel_(
          DLTensor & out_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_);
    DLTensor & out = _out.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES(out)
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");

    const int64_t    nc = out.shape[out.ndim - 1];
    const bound::BoundVec bvec(static_cast<bound::type>(bound));
    const DispatchKey key(ndim, bound, out.dtype,
                          CANUSE32BITS(out));

    dispatch_nbd<field_kernel_op<'+'> >(key,
        bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(out), voxel_size, absolute, membrane, bending, out.shape, out.strides);
}

/**
 * @brief `field_kernel` variant that subtracts the stencil: `out -= K`.
 *        In-place only (jitfields `op='-'`).
 */
void field_subkernel_(
          DLTensor & out_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_);
    DLTensor & out = _out.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES(out)
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");

    const int64_t    nc = out.shape[out.ndim - 1];
    const bound::BoundVec bvec(static_cast<bound::type>(bound));
    const DispatchKey key(ndim, bound, out.dtype,
                          CANUSE32BITS(out));

    dispatch_nbd<field_kernel_op<'-'> >(key,
        bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(out), voxel_size, absolute, membrane, bending, out.shape, out.strides);
}

void field_relax(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        nb_iter   ,
          intptr_t   /* stream <unused> */
)
{
    const int32_t nbatch = sol.ndim - ndim - 1;
    CHECK_NO_LANES  (sol)
    CHECK_SAME_DTYPE(sol, hes)
    CHECK_SAME_DTYPE(sol, grd)
    CHECK_SAME      (sol.ndim, grd.ndim, "Tensors do not have the same number of dimensions")
    CHECK_SAME      (sol.ndim, hes.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME_SHAPE(sol, grd, sol.ndim)

    const int64_t    nc = sol.shape[sol.ndim - 1];
    const bound::BoundVec bvec(static_cast<bound::type>(bound));
    const DispatchKey key(ndim, bound, sol.dtype,
                          CANUSE32BITS(sol) && CANUSE32BITS(hes) && CANUSE32BITS(grd));

    dispatch_nbd<field_relax_op >(key,
        bvec, static_cast<int64_t>(nbatch), nc, VOIDPTR(sol), CVOIDPTR(hes), CVOIDPTR(grd), voxel_size, absolute, membrane, bending, nb_iter, sol.shape, sol.strides, hes.strides, grd.strides);
}

void field_forward(
          DLTensor & out       ,
    const DLTensor & hes       ,
    const DLTensor & inp       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream    )
{
    sym_matvec(out, hes, inp, stream);
    field_addmatvec_(out, inp, voxel_size, absolute, membrane, bending, bound, ndim, stream);
}

// `field_diag`'s regulariser diagonal doesn't depend on the operand being
// solved for, so `field_precond[_]` materialise it into a fresh contiguous
// scratch buffer shaped like `grd`/`sol` and hand it to posdef::sym_solve[_]
// as the per-channel weight map.
static inline std::vector<uint8_t> field_precond_diag(
    const DLTensor & like      ,
    const double    * voxel_size,
    const double    * absolute  ,
    const double    * membrane  ,
    const double    * bending   ,
          int8_t      bound     ,
          int         ndim      ,
          intptr_t    stream    ,
          DLTensor  & diag_t    )
{
    size_t numel = 1;
    for (int32_t d = 0; d < like.ndim; ++d)
        numel *= static_cast<size_t>(like.shape[d]);
    std::vector<uint8_t> diag_buf(numel * static_cast<size_t>(like.dtype.bits) / 8);

    diag_t.data        = diag_buf.data();
    diag_t.device       = like.device;
    diag_t.ndim         = like.ndim;
    diag_t.dtype        = like.dtype;
    diag_t.shape        = like.shape;
    diag_t.strides      = nullptr;
    diag_t.byte_offset  = 0;

    field_diag(diag_t, voxel_size, absolute, membrane, bending, bound, ndim, stream);
    return diag_buf;
}

void field_precond(
          DLTensor & out       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream    )
{
    CHECK_NO_LANES(grd)
    if (grd.ndim - ndim - 1 < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");

    DLTensor diag_t;
    std::vector<uint8_t> diag_buf = field_precond_diag(
        grd, voxel_size, absolute, membrane, bending, bound, ndim, stream, diag_t);
    sym_solve(out, hes, grd, diag_t, stream);
}

void field_precond_(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   stream    )
{
    CHECK_NO_LANES(sol)
    if (sol.ndim - ndim - 1 < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");

    DLTensor diag_t;
    std::vector<uint8_t> diag_buf = field_precond_diag(
        sol, voxel_size, absolute, membrane, bending, bound, ndim, stream, diag_t);
    sym_solve_(sol, hes, diag_t, stream);
}

// Determine whether `wgt`'s trailing (channel) dimension selects the RLS
// (one weight per channel) or JRLS (single weight shared/"joint" across all
// `nc` channels) impl variant, validating it against the operand's channel
// count. NB: RLS = per-channel (wc == nc), JRLS = broadcast (wc == 1) --
// matches the original jitfields/nitorch semantics. See fastfields-cpu-lib#65:
// this predicate previously had the two cases backwards.
static inline bool field_rls_is_jrls(const DLTensor & wgt, int64_t nc,
                                     const char * who)
{
    const int64_t wc = wgt.shape[wgt.ndim - 1];
    if (wc == nc) return false;
    if (wc == 1) return true;
    throw std::invalid_argument(std::string(who) +
                                ": weight tensor's trailing dimension must be "
                                "1 (JRLS) or match the channel count (RLS)");
}

void field_matvec_rls(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const DLTensor & wgt_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _inp(inp_), _wgt(wgt_);
    DLTensor       & out = _out.t;
    const DLTensor & inp = _inp.t;
    const DLTensor & wgt = _wgt.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, inp)
    CHECK_SAME_DTYPE(out, wgt)
    CHECK_SAME      (out.ndim, inp.ndim, "Tensors do not have the same number of dimensions")
    CHECK_SAME      (out.ndim, wgt.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME_SHAPE(out, inp, out.ndim)
    CHECK_SAME_SHAPE(out, wgt, out.ndim - 1)

    const int64_t    nc = out.shape[out.ndim - 1];
    const bool     is_jrls = field_rls_is_jrls(wgt, nc, "field_matvec_rls");
    const bound::BoundVec bvec(static_cast<bound::type>(bound));
    const DispatchKey key(ndim, bound, out.dtype,
                          CANUSE32BITS(out) && CANUSE32BITS(inp) && CANUSE32BITS(wgt));

    dispatch_nbd<field_matvec_rls_op >(key,
        bvec, static_cast<int64_t>(nbatch), nc, is_jrls, VOIDPTR(out), CVOIDPTR(inp), CVOIDPTR(wgt), voxel_size, absolute, membrane, bending, out.shape, out.strides, inp.strides, wgt.strides);
}

void field_diag_rls(
          DLTensor & out_      ,
    const DLTensor & wgt_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          intptr_t   /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _out(out_), _wgt(wgt_);
    DLTensor       & out = _out.t;
    const DLTensor & wgt = _wgt.t;

    const int32_t nbatch = out.ndim - ndim - 1;
    CHECK_NO_LANES  (out)
    CHECK_SAME_DTYPE(out, wgt)
    CHECK_SAME      (out.ndim, wgt.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME_SHAPE(out, wgt, out.ndim - 1)

    const int64_t    nc = out.shape[out.ndim - 1];
    const bool     is_jrls = field_rls_is_jrls(wgt, nc, "field_diag_rls");
    const bound::BoundVec bvec(static_cast<bound::type>(bound));
    const DispatchKey key(ndim, bound, out.dtype,
                          CANUSE32BITS(out) && CANUSE32BITS(wgt));

    dispatch_nbd<field_diag_rls_op >(key,
        bvec, static_cast<int64_t>(nbatch), nc, is_jrls, VOIDPTR(out), CVOIDPTR(wgt), voxel_size, absolute, membrane, bending, out.shape, out.strides, wgt.strides);
}

void field_relax_rls(
          DLTensor & sol       ,
    const DLTensor & hes       ,
    const DLTensor & grd       ,
    const DLTensor & wgt_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        nb_iter   ,
          intptr_t   /* stream <unused> */
)
{
    // Normalise NULL strides (compact row-major) before dispatch.
    ContiguousStrides _wgt(wgt_);
    const DLTensor & wgt = _wgt.t;

    const int32_t nbatch = sol.ndim - ndim - 1;
    CHECK_NO_LANES  (sol)
    CHECK_SAME_DTYPE(sol, hes)
    CHECK_SAME_DTYPE(sol, grd)
    CHECK_SAME_DTYPE(sol, wgt)
    CHECK_SAME      (sol.ndim, grd.ndim, "Tensors do not have the same number of dimensions")
    CHECK_SAME      (sol.ndim, hes.ndim, "Tensors do not have the same number of dimensions")
    CHECK_SAME      (sol.ndim, wgt.ndim, "Tensors do not have the same number of dimensions")
    if (nbatch < 0)
        throw std::invalid_argument("ndim is larger than the tensor rank");
    CHECK_SAME_SHAPE(sol, grd, sol.ndim)
    CHECK_SAME_SHAPE(sol, wgt, sol.ndim - 1)

    const int64_t    nc = sol.shape[sol.ndim - 1];
    const bool     is_jrls = field_rls_is_jrls(wgt, nc, "field_relax_rls");
    const bound::BoundVec bvec(static_cast<bound::type>(bound));
    const DispatchKey key(ndim, bound, sol.dtype,
                          CANUSE32BITS(sol) && CANUSE32BITS(hes) && CANUSE32BITS(grd) && CANUSE32BITS(wgt));

    dispatch_nbd<field_relax_rls_op >(key,
        bvec, static_cast<int64_t>(nbatch), nc, is_jrls, VOIDPTR(sol), CVOIDPTR(hes), CVOIDPTR(grd), CVOIDPTR(wgt), voxel_size, absolute, membrane, bending, nb_iter, sol.shape, sol.strides, hes.strides, grd.strides, wgt.strides);
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
