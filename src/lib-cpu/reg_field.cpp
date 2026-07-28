#include <stdexcept>
#include <string>
#include <vector>
#include "reg_field.h"
#include "autocast.h"
#include "dlpack.h"
#include "impl/kernels/cuda_switch.h"
#include "impl/kernels/bounds.h"
#include "impl/kernels/utils.h"
#include "impl/reg_field.h"

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

template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_matvec(
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
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nall1);
    const offset_t * _stride_inp = copy_if_needed<offset_t *>(stride_inp, nall1);
          scalar_t * _out = static_cast<      scalar_t *>(out);
    const scalar_t * _inp = static_cast<const scalar_t *>(inp);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    std::vector<reduce_t> a = as_weights(absolute, nc);
    std::vector<reduce_t> m = as_weights(membrane, nc);
    std::vector<reduce_t> b = as_weights(bending,  nc);

    if (bending)
        reg_field::matvec_bending<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, a.data(), m.data(), b.data());
    else if (membrane)
        reg_field::matvec_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, a.data(), m.data());
    else
        reg_field::matvec_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, a.data());

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_inp);
}

// Accumulate variant of _field_matvec: out += L(inp) (op='+') or out -= L(inp)
// (op='-'), instead of overwriting out. The impl-layer matvec_* templates
// already take the op char (see op_apply, issue #6a); this just threads a
// compile-time '+'/'-' instead of the '=' _field_matvec hardcodes.
template <int ndim, char op, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_matvec_acc(
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
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nall1);
    const offset_t * _stride_inp = copy_if_needed<offset_t *>(stride_inp, nall1);
          scalar_t * _out = static_cast<      scalar_t *>(out);
    const scalar_t * _inp = static_cast<const scalar_t *>(inp);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    std::vector<reduce_t> a = as_weights(absolute, nc);
    std::vector<reduce_t> m = as_weights(membrane, nc);
    std::vector<reduce_t> b = as_weights(bending,  nc);

    if (bending)
        reg_field::matvec_bending<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, a.data(), m.data(), b.data());
    else if (membrane)
        reg_field::matvec_membrane<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, vx, a.data(), m.data());
    else
        reg_field::matvec_absolute<ndim, op, reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out, _inp,
            _size, _stride_out, _stride_inp, a.data());

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_inp);
}

template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_diag(
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
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nall1);
          scalar_t * _out = static_cast<scalar_t *>(out);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    std::vector<reduce_t> a = as_weights(absolute, nc);
    std::vector<reduce_t> m = as_weights(membrane, nc);
    std::vector<reduce_t> b = as_weights(bending,  nc);

    if (bending)
        reg_field::diag_bending<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, a.data(), m.data(), b.data());
    else if (membrane)
        reg_field::diag_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, a.data(), m.data());
    else
        reg_field::diag_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, a.data());

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
}

// Materialise the Toeplitz convolution kernel (stencil) of the operator. The
// output is a per-channel vector stencil (*batch, *spatial, C) -- the field
// regulariser never couples channels, so there is no matrix case. Dispatch
// mirrors _field_diag: the highest-order non-null penalty selects the stencil.
template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_kernel(
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
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nall1);
          scalar_t * _out = static_cast<scalar_t *>(out);

    reduce_t vx[ndim];
    for (int d = 0; d < ndim; ++d) vx[d] = voxel_size ? voxel_size[d] : 1.0;

    std::vector<reduce_t> a = as_weights(absolute, nc);
    std::vector<reduce_t> m = as_weights(membrane, nc);
    std::vector<reduce_t> b = as_weights(bending,  nc);

    if (bending)
        reg_field::kernel_bending<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, a.data(), m.data(), b.data());
    else if (membrane)
        reg_field::kernel_membrane<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, vx, a.data(), m.data());
    else
        // kernel_absolute takes no voxel_size (the L2 stencil is scale-free).
        reg_field::kernel_absolute<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _out,
            _size, _stride_out, a.data());

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
}

// One or more relaxation (Gauss-Seidel) sweeps solving `(H + L) x = g` in
// place, where H is the per-voxel compact-symmetric Hessian, L the field
// regulariser, and x the warm-started `sol`. Dispatches to the impl relaxer
// matching the highest-order penalty (membrane covers the absolute-only case).
template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_relax(
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
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_sol = copy_if_needed<offset_t *>(stride_sol, nall1);
    const offset_t * _stride_hes = copy_if_needed<offset_t *>(stride_hes, nall1);
    const offset_t * _stride_grd = copy_if_needed<offset_t *>(stride_grd, nall1);
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
            static_cast<offset_t>(nbatch), _sol, _hes, _grd,
            _size, _stride_sol, _stride_hes, _stride_grd, vx,
            a.data(), m.data(), b.data(), niter);
    else
        reg_field::relax_membrane_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
            static_cast<offset_t>(nbatch), _sol, _hes, _grd,
            _size, _stride_sol, _stride_hes, _stride_grd, vx,
            a.data(), m.data(), niter);

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_sol);
    free_if_needed<int64_t *>(_stride_hes);
    free_if_needed<int64_t *>(_stride_grd);
}

// Reweighted-least-squares (RLS/JRLS) variant of `_field_matvec`: an extra
// per-voxel weight map `wgt` modulates the penalty strength. `is_jrls`
// selects between the single-shared-weight (RLS) and per-channel-weight
// (JRLS) impl variants -- both take an identical argument list, so the
// dispatch is a plain runtime branch rather than a template parameter.
template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_matvec_rls(
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
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nall1);
    const offset_t * _stride_inp = copy_if_needed<offset_t *>(stride_inp, nall1);
    const offset_t * _stride_wgt = copy_if_needed<offset_t *>(stride_wgt, nall1);
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
                static_cast<offset_t>(nbatch), _out, _inp, _wgt,
                _size, _stride_out, _stride_inp, _stride_wgt, vx, a.data(), m.data(), b.data());
        else
            reg_field::matvec_bending_rls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out, _inp, _wgt,
                _size, _stride_out, _stride_inp, _stride_wgt, vx, a.data(), m.data(), b.data());
    } else if (membrane) {
        if (is_jrls)
            reg_field::matvec_membrane_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out, _inp, _wgt,
                _size, _stride_out, _stride_inp, _stride_wgt, vx, a.data(), m.data());
        else
            reg_field::matvec_membrane_rls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out, _inp, _wgt,
                _size, _stride_out, _stride_inp, _stride_wgt, vx, a.data(), m.data());
    } else {
        if (is_jrls)
            reg_field::matvec_absolute_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out, _inp, _wgt,
                _size, _stride_out, _stride_inp, _stride_wgt, a.data());
        else
            reg_field::matvec_absolute_rls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out, _inp, _wgt,
                _size, _stride_out, _stride_inp, _stride_wgt, a.data());
    }

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_inp);
    free_if_needed<int64_t *>(_stride_wgt);
}

template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_diag_rls(
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
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_out = copy_if_needed<offset_t *>(stride_out, nall1);
    const offset_t * _stride_wgt = copy_if_needed<offset_t *>(stride_wgt, nall1);
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
                static_cast<offset_t>(nbatch), _out, _wgt,
                _size, _stride_out, _stride_wgt, vx, a.data(), m.data(), b.data());
        else
            reg_field::diag_bending_rls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out, _wgt,
                _size, _stride_out, _stride_wgt, vx, a.data(), m.data(), b.data());
    } else if (membrane) {
        if (is_jrls)
            reg_field::diag_membrane_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out, _wgt,
                _size, _stride_out, _stride_wgt, vx, a.data(), m.data());
        else
            reg_field::diag_membrane_rls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out, _wgt,
                _size, _stride_out, _stride_wgt, vx, a.data(), m.data());
    } else {
        if (is_jrls)
            reg_field::diag_absolute_jrls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out, _wgt,
                _size, _stride_out, _stride_wgt, a.data());
        else
            reg_field::diag_absolute_rls<ndim, '=', reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _out, _wgt,
                _size, _stride_out, _stride_wgt, a.data());
    }

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_out);
    free_if_needed<int64_t *>(_stride_wgt);
}

template <int ndim, typename scalar_t, typename offset_t, bound::type... BOUND>
inline void _field_relax_rls(
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
    const int64_t nall1 = nbatch + ndim + 1;
    const offset_t * _size       = copy_if_needed<offset_t *>(size,       nall1);
    const offset_t * _stride_sol = copy_if_needed<offset_t *>(stride_sol, nall1);
    const offset_t * _stride_hes = copy_if_needed<offset_t *>(stride_hes, nall1);
    const offset_t * _stride_grd = copy_if_needed<offset_t *>(stride_grd, nall1);
    const offset_t * _stride_wgt = copy_if_needed<offset_t *>(stride_wgt, nall1);
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
                static_cast<offset_t>(nbatch), _sol, _hes, _grd, _wgt,
                _size, _stride_sol, _stride_hes, _stride_grd, _stride_wgt, vx,
                a.data(), m.data(), b.data(), niter);
        else
            reg_field::relax_bending_rls_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _sol, _hes, _grd, _wgt,
                _size, _stride_sol, _stride_hes, _stride_grd, _stride_wgt, vx,
                a.data(), m.data(), b.data(), niter);
    } else {
        if (is_jrls)
            reg_field::relax_membrane_jrls_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _sol, _hes, _grd, _wgt,
                _size, _stride_sol, _stride_hes, _stride_grd, _stride_wgt, vx,
                a.data(), m.data(), niter);
        else
            reg_field::relax_membrane_rls_<ndim, reduce_t, scalar_t, offset_t, BOUND...>(
                static_cast<offset_t>(nbatch), _sol, _hes, _grd, _wgt,
                _size, _stride_sol, _stride_hes, _stride_grd, _stride_wgt, vx,
                a.data(), m.data(), niter);
    }

    free_if_needed<int64_t *>(_size);
    free_if_needed<int64_t *>(_stride_sol);
    free_if_needed<int64_t *>(_stride_hes);
    free_if_needed<int64_t *>(_stride_grd);
    free_if_needed<int64_t *>(_stride_wgt);
}

} // anonymous namespace

/***********************************************************************
 *                            DISPATCH                                 *
 ***********************************************************************/

#define BND1(B) B
#define BND2(B) B, B
#define BND3(B) B, B, B

#define MV_DT(NDIM, BNDS...)                                            \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _field_matvec<NDIM, float,  int32_t, BNDS>(MV_ARGS)   \
                : _field_matvec<NDIM, float,  int64_t, BNDS>(MV_ARGS);  \
            case 64: return use_32bits                                  \
                ? _field_matvec<NDIM, double, int32_t, BNDS>(MV_ARGS)   \
                : _field_matvec<NDIM, double, int64_t, BNDS>(MV_ARGS);  \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define ADD_MV_DT(NDIM, BNDS...)                                        \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _field_matvec_acc<NDIM, '+', float,  int32_t, BNDS>(MV_ARGS) \
                : _field_matvec_acc<NDIM, '+', float,  int64_t, BNDS>(MV_ARGS); \
            case 64: return use_32bits                                  \
                ? _field_matvec_acc<NDIM, '+', double, int32_t, BNDS>(MV_ARGS) \
                : _field_matvec_acc<NDIM, '+', double, int64_t, BNDS>(MV_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define SUB_MV_DT(NDIM, BNDS...)                                        \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _field_matvec_acc<NDIM, '-', float,  int32_t, BNDS>(MV_ARGS) \
                : _field_matvec_acc<NDIM, '-', float,  int64_t, BNDS>(MV_ARGS); \
            case 64: return use_32bits                                  \
                ? _field_matvec_acc<NDIM, '-', double, int32_t, BNDS>(MV_ARGS) \
                : _field_matvec_acc<NDIM, '-', double, int64_t, BNDS>(MV_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define DG_DT(NDIM, BNDS...)                                            \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _field_diag<NDIM, float,  int32_t, BNDS>(DG_ARGS)     \
                : _field_diag<NDIM, float,  int64_t, BNDS>(DG_ARGS);    \
            case 64: return use_32bits                                  \
                ? _field_diag<NDIM, double, int32_t, BNDS>(DG_ARGS)     \
                : _field_diag<NDIM, double, int64_t, BNDS>(DG_ARGS);    \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define KN_DT(NDIM, BNDS...)                                            \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _field_kernel<NDIM, float,  int32_t, BNDS>(KN_ARGS)   \
                : _field_kernel<NDIM, float,  int64_t, BNDS>(KN_ARGS);  \
            case 64: return use_32bits                                  \
                ? _field_kernel<NDIM, double, int32_t, BNDS>(KN_ARGS)   \
                : _field_kernel<NDIM, double, int64_t, BNDS>(KN_ARGS);  \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define RX_DT(NDIM, BNDS...)                                            \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _field_relax<NDIM, float,  int32_t, BNDS>(RX_ARGS)    \
                : _field_relax<NDIM, float,  int64_t, BNDS>(RX_ARGS);   \
            case 64: return use_32bits                                  \
                ? _field_relax<NDIM, double, int32_t, BNDS>(RX_ARGS)    \
                : _field_relax<NDIM, double, int64_t, BNDS>(RX_ARGS);   \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define RLS_MV_DT(NDIM, BNDS...)                                        \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _field_matvec_rls<NDIM, float,  int32_t, BNDS>(RLS_MV_ARGS) \
                : _field_matvec_rls<NDIM, float,  int64_t, BNDS>(RLS_MV_ARGS); \
            case 64: return use_32bits                                  \
                ? _field_matvec_rls<NDIM, double, int32_t, BNDS>(RLS_MV_ARGS) \
                : _field_matvec_rls<NDIM, double, int64_t, BNDS>(RLS_MV_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define RLS_DG_DT(NDIM, BNDS...)                                        \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _field_diag_rls<NDIM, float,  int32_t, BNDS>(RLS_DG_ARGS) \
                : _field_diag_rls<NDIM, float,  int64_t, BNDS>(RLS_DG_ARGS); \
            case 64: return use_32bits                                  \
                ? _field_diag_rls<NDIM, double, int32_t, BNDS>(RLS_DG_ARGS) \
                : _field_diag_rls<NDIM, double, int64_t, BNDS>(RLS_DG_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define RLS_RX_DT(NDIM, BNDS...)                                        \
    switch (code) {                                                     \
        case kDLFloat: switch (bits) {                                  \
            case 32: return use_32bits                                  \
                ? _field_relax_rls<NDIM, float,  int32_t, BNDS>(RLS_RX_ARGS) \
                : _field_relax_rls<NDIM, float,  int64_t, BNDS>(RLS_RX_ARGS); \
            case 64: return use_32bits                                  \
                ? _field_relax_rls<NDIM, double, int32_t, BNDS>(RLS_RX_ARGS) \
                : _field_relax_rls<NDIM, double, int64_t, BNDS>(RLS_RX_ARGS); \
            default: break;                                             \
        } break;                                                        \
        default: break;                                                 \
    }                                                                   \
    throw std::invalid_argument("only floating point data types are supported");

#define BOUND_SWITCH(DT, NDIM, BND)                                     \
    switch (bnd) {                                                      \
        case bound::type::Zero:      DT(NDIM, BND(bound::type::Zero));      break; \
        case bound::type::Replicate: DT(NDIM, BND(bound::type::Replicate)); break; \
        case bound::type::DCT1:      DT(NDIM, BND(bound::type::DCT1));      break; \
        case bound::type::DCT2:      DT(NDIM, BND(bound::type::DCT2));      break; \
        case bound::type::DST1:      DT(NDIM, BND(bound::type::DST1));      break; \
        case bound::type::DST2:      DT(NDIM, BND(bound::type::DST2));      break; \
        case bound::type::DFT:       DT(NDIM, BND(bound::type::DFT));       break; \
        case bound::type::NoCheck:   DT(NDIM, BND(bound::type::NoCheck));   break; \
        default: throw std::invalid_argument("Unsupported boundary condition"); \
    }

#define NDIM_SWITCH(DT)                                                 \
    switch (ndim) {                                                     \
        case 1: BOUND_SWITCH(DT, 1, BND1); break;                       \
        case 2: BOUND_SWITCH(DT, 2, BND2); break;                       \
        case 3: BOUND_SWITCH(DT, 3, BND3); break;                       \
        default: throw std::invalid_argument("Only 1D, 2D and 3D field are supported"); \
    }

void field_matvec(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        /* stream <unused> */
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
    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);

#define MV_ARGS static_cast<int64_t>(nbatch), nc, VOIDPTR(out), CVOIDPTR(inp), \
                voxel_size, absolute, membrane, bending,                       \
                out.shape, out.strides, inp.strides
    NDIM_SWITCH(MV_DT)
#undef MV_ARGS
}

/**
 * @brief `field_matvec` variant that accumulates: `out += L(inp)`.
 */
void field_matvec_add(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        /* stream <unused> */
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
    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);

#define MV_ARGS static_cast<int64_t>(nbatch), nc, VOIDPTR(out), CVOIDPTR(inp), \
                voxel_size, absolute, membrane, bending,                       \
                out.shape, out.strides, inp.strides
    NDIM_SWITCH(ADD_MV_DT)
#undef MV_ARGS
}

/**
 * @brief `field_matvec` variant that subtracts: `out -= L(inp)`.
 */
void field_matvec_sub(
          DLTensor & out_      ,
    const DLTensor & inp_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        /* stream <unused> */
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
    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);

#define MV_ARGS static_cast<int64_t>(nbatch), nc, VOIDPTR(out), CVOIDPTR(inp), \
                voxel_size, absolute, membrane, bending,                       \
                out.shape, out.strides, inp.strides
    NDIM_SWITCH(SUB_MV_DT)
#undef MV_ARGS
}

void field_diag(
          DLTensor & out_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        /* stream <unused> */
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
    const bool     use_32bits = CANUSE32BITS(out);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);

#define DG_ARGS static_cast<int64_t>(nbatch), nc, VOIDPTR(out), \
                voxel_size, absolute, membrane, bending,         \
                out.shape, out.strides
    NDIM_SWITCH(DG_DT)
#undef DG_ARGS
}

void field_kernel(
          DLTensor & out_      ,
    const double   * voxel_size,
    const double   * absolute  ,
    const double   * membrane  ,
    const double   * bending   ,
          int8_t     bound     ,
          int        ndim      ,
          int        /* stream <unused> */
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
    const bool     use_32bits = CANUSE32BITS(out);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);

#define KN_ARGS static_cast<int64_t>(nbatch), nc, VOIDPTR(out), \
                voxel_size, absolute, membrane, bending,         \
                out.shape, out.strides
    NDIM_SWITCH(KN_DT)
#undef KN_ARGS
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
          int        /* stream <unused> */
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
    const bool     use_32bits = CANUSE32BITS(sol) && CANUSE32BITS(hes) &&
                                CANUSE32BITS(grd);
    const auto     code = static_cast<DLDataTypeCode>(sol.dtype.code);
    const auto     bits = sol.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);

#define RX_ARGS static_cast<int64_t>(nbatch), nc, VOIDPTR(sol), CVOIDPTR(hes), \
                CVOIDPTR(grd), voxel_size, absolute, membrane, bending,        \
                nb_iter, sol.shape, sol.strides, hes.strides, grd.strides
    NDIM_SWITCH(RX_DT)
#undef RX_ARGS
}

// Determine whether `wgt`'s trailing (channel) dimension selects the RLS
// (single weight shared across all `nc` channels) or JRLS (one weight per
// channel) impl variant, validating it against the operand's channel count.
static inline bool field_rls_is_jrls(const DLTensor & wgt, int64_t nc,
                                     const char * who)
{
    const int64_t wc = wgt.shape[wgt.ndim - 1];
    if (wc == 1)  return false;
    if (wc == nc) return true;
    throw std::invalid_argument(
        std::string(who) + ": weight tensor's trailing dimension must be "
        "1 (RLS) or match the channel count (JRLS)");
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
          int        /* stream <unused> */
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
    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(inp) && CANUSE32BITS(wgt);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);

#define RLS_MV_ARGS static_cast<int64_t>(nbatch), nc, is_jrls, VOIDPTR(out), \
                CVOIDPTR(inp), CVOIDPTR(wgt),                                \
                voxel_size, absolute, membrane, bending,                     \
                out.shape, out.strides, inp.strides, wgt.strides
    NDIM_SWITCH(RLS_MV_DT)
#undef RLS_MV_ARGS
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
          int        /* stream <unused> */
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
    const bool     use_32bits = CANUSE32BITS(out) && CANUSE32BITS(wgt);
    const auto     code = static_cast<DLDataTypeCode>(out.dtype.code);
    const auto     bits = out.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);

#define RLS_DG_ARGS static_cast<int64_t>(nbatch), nc, is_jrls, VOIDPTR(out), \
                CVOIDPTR(wgt),                                               \
                voxel_size, absolute, membrane, bending,                     \
                out.shape, out.strides, wgt.strides
    NDIM_SWITCH(RLS_DG_DT)
#undef RLS_DG_ARGS
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
          int        /* stream <unused> */
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
    const bool     use_32bits = CANUSE32BITS(sol) && CANUSE32BITS(hes) &&
                                CANUSE32BITS(grd) && CANUSE32BITS(wgt);
    const auto     code = static_cast<DLDataTypeCode>(sol.dtype.code);
    const auto     bits = sol.dtype.bits;
    const bound::type bnd = static_cast<bound::type>(bound);

#define RLS_RX_ARGS static_cast<int64_t>(nbatch), nc, is_jrls, VOIDPTR(sol), \
                CVOIDPTR(hes), CVOIDPTR(grd), CVOIDPTR(wgt),                 \
                voxel_size, absolute, membrane, bending,                    \
                nb_iter, sol.shape, sol.strides, hes.strides, grd.strides,  \
                wgt.strides
    NDIM_SWITCH(RLS_RX_DT)
#undef RLS_RX_ARGS
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
