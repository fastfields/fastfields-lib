#pragma once
#include <stdexcept>
#include <fastfields/core/cuda_switch.h>
#include <fastfields/impl/kernels/bounds.h>
#include <fastfields/impl/kernels/utils.h>
#include <fastfields/impl/kernels/posdef.h>
#include <fastfields/impl/kernels/regularisers/border.h>
#include <fastfields/impl/kernels/regularisers/field/utils.h>
#include "utils.h"       // GET_BLOCKS / CUDA_NUM_THREADS
#include "launch.h"      // FF_CUDA_LAUNCH -- the only checked kernel launch

/***********************************************************************
 *
 *  CUDA side of the definition-F border pass (see
 *  kernels/regularisers/border.h for the convention, and
 *  impl/cpu/reg_border.h for the CPU twin this mirrors).
 *
 *  The stencil kernels in reg_{field,flow}.h keep their flat loop and skip
 *  every voxel outside the interior box (`Slabs::interior`, passed to them by
 *  value; a compare-only `contains`, no index remapping -- see DevBox). The
 *  border slabs run `border_kernel` below, which rebuilds (A_F v)[p] /
 *  A_F[p,p] row by row through `field_at` / `flow_at`. It is deliberately
 *  light on nvcc: NOT templated on nbatch, the channel count, the boundary
 *  condition, the op or the kind (matvec / diag / relax) -- all runtime, read
 *  from `Args`, the BoundVec and two scalars -- so a module instantiates it
 *  once per (ndim, field/flow, dtype, index width) however many stencil
 *  instantiations it has. Every argument travels by value in one
 *  trivially-copyable struct: no device allocation, no copies.
 *
 *  Relaxation: the launcher runs, per colour, the interior stencil kernel
 *  then these border kernels for the same colour -- the CPU order, so the
 *  fixed point is the same (H + A_F) x = g.
 *
 **********************************************************************/

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_border)

template <typename scalar_t, typename reduce_t, typename offset_t>
struct Args {
    int      nbatch, nall;
    offset_t nc;
    offset_t size [MAXALL + 1];
    offset_t s_out[MAXALL + 1];   // out, or sol (relax)
    offset_t s_inp[MAXALL + 1];   // inp (matvec)
    offset_t s_wgt[MAXALL + 1];
    offset_t s_hes[MAXALL + 1];
    offset_t s_grd[MAXALL + 1];
    reduce_t vx[3];
    reduce_t a[3], m[3], b[3];    // per channel (fields) / [0] (flows)
    reduce_t sh, dv;              // flows only
    bool     has_b;
    bool     joint;               // fields: one weight for every channel
    scalar_t       * out;
    const scalar_t * inp;
    const scalar_t * wgt;         // nullptr: unweighted
    const scalar_t * hes;
    const scalar_t * grd;
};

template <typename offset_t>
FF_CUDEV inline void decompose(offset_t i, int nall, const offset_t * size, offset_t * x)
{
    for (int d = 0; d < nall; ++d) { x[d] = i % size[d]; i /= size[d]; }
}

template <typename offset_t>
FF_CUDEV inline offset_t dot_range(const offset_t * x, const offset_t * s, int lo, int hi)
{
    offset_t o = 0;
    for (int d = lo; d < hi; ++d) o += x[d] * s[d];
    return o;
}

template <int C, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUDEV inline void relax_point(
    scalar_t * sol, offset_t osc, const scalar_t * hes, offset_t hsc,
    const reduce_t * val, const reduce_t * dg)
{
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t, C>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, C>::work_size;
    scalar_t v[C], d[C];
    reduce_t buf[CC ? CC : 1];
    for (int c = 0; c < C; ++c) {
        v[c] = static_cast<scalar_t>(val[c]);
        d[c] = static_cast<scalar_t>(dg[c]);
    }
    PosDef::relax_(Strided(sol, osc), StridedConst(hes, hsc), v, d, buf,
                   static_cast<reduce_t>(0));
}

template <int ndim, typename offset_t>
FF_CUDEV inline bool colour(int patch, const offset_t * loc, offset_t n)
{
    return patch == 1 ? reg_field::patch1<ndim, offset_t>(loc, n)
         : patch == 2 ? reg_field::patch2<ndim, offset_t>(loc, n)
         :              reg_field::patch3<ndim, offset_t>(loc, n);
}

template <typename scalar_t, typename reduce_t>
FF_CUDEV inline void apply_rt(char op, scalar_t & out, reduce_t v)
{
    if (op == '+')      apply_op<'+'>(out, v);
    else if (op == '-') apply_op<'-'>(out, v);
    else                apply_op<'='>(out, v);
}

// kind: 0 matvec, 1 diag, 2 relax (one colour). `kind` and `op` are runtime
// on purpose: one instantiation per (ndim, field/flow, dtype, index width)
// instead of seven -- the border pass is a few percent of the voxels, a
// uniform branch costs nothing there, and nvcc memory is the scarce resource
// in these modules (src/lib-cuda/Makefile).
template <int ndim, bool flow, typename reduce_t, typename scalar_t, typename offset_t>
FF_CUGLOB void border_kernel(
    bound::BoundVec bnd, DevBox<offset_t> box,
    Args<scalar_t, reduce_t, offset_t> A, int kind, char op, int patch, int col)
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t index_stride = blockDim.x * gridDim.x;
    const int nb = A.nbatch, na = A.nall;
    for (; index < box.numel; index += index_stride)
    {
        offset_t x[MAXALL], loc[ndim];
        decompose(box(index), na, A.size, x);
        for (int d = 0; d < ndim; ++d) loc[d] = x[nb + d];
        if (kind == 2 && !colour<ndim>(patch, loc, static_cast<offset_t>(col)))
            continue;

        // batch-element origins and the voxel's own offsets
        const scalar_t * v0 = nullptr;
        offset_t vsc = 0;
        const offset_t * vstr = A.s_inp;
        if (kind == 0) { v0 = A.inp + dot_range(x, A.s_inp, 0, nb); vsc = A.s_inp[na]; }
        if (kind == 2) { v0 = A.out + dot_range(x, A.s_out, 0, nb); vsc = A.s_out[na]; vstr = A.s_out; }
        const scalar_t * w0 = A.wgt ? A.wgt + dot_range(x, A.s_wgt, 0, nb) : nullptr;
        const offset_t oo   = dot_range(x, A.s_out, 0, na);
        const offset_t osc  = A.s_out[na];

        reduce_t mv[3] = {0, 0, 0}, dg[3] = {0, 0, 0};
        if (flow)
            flow_at<ndim, reduce_t, scalar_t, offset_t>(
                bnd, loc, A.size + nb, v0, vstr + nb, vsc,
                w0, A.s_wgt + nb, A.vx, A.a[0], A.m[0], A.b[0], A.sh, A.dv,
                kind == 1 ? nullptr : mv, kind == 0 ? nullptr : dg);
        else
            field_at<ndim, reduce_t, scalar_t, offset_t>(
                bnd, loc, A.size + nb, v0, vstr + nb, vsc,
                w0, A.s_wgt + nb, (A.wgt && !A.joint) ? A.s_wgt[na] : static_cast<offset_t>(0),
                A.nc, A.vx, A.a, A.m, A.has_b ? A.b : nullptr,
                kind == 1 ? nullptr : mv, kind == 0 ? nullptr : dg);

        if (kind == 0) {
            for (offset_t c = 0; c < A.nc; ++c) apply_rt(op, A.out[oo + osc * c], mv[c]);
        } else if (kind == 1) {
            for (offset_t c = 0; c < A.nc; ++c) apply_rt(op, A.out[oo + osc * c], dg[c]);
        } else {
            const offset_t go = dot_range(x, A.s_grd, 0, na), gsc = A.s_grd[na];
            const offset_t ho = dot_range(x, A.s_hes, 0, na), hsc = A.s_hes[na];
            reduce_t val[3];
            for (offset_t c = 0; c < A.nc; ++c)
                val[c] = static_cast<reduce_t>(A.grd[go + gsc * c]) - mv[c];
            if (A.nc == 1)      relax_point<1>(A.out + oo, osc, A.hes + ho, hsc, val, dg);
            else if (A.nc == 2) relax_point<2>(A.out + oo, osc, A.hes + ho, hsc, val, dg);
            else                relax_point<3>(A.out + oo, osc, A.hes + ho, hsc, val, dg);
        }
    }
}

//----------------------------------------------------------------------
//                          Host side
//----------------------------------------------------------------------

/// Slab partition for one launcher call (see `thickness` in border.h).
template <int ndim, typename offset_t>
inline Slabs<offset_t> cuda_slabs(const bound::BoundVec & bnd, offset_t nbatch,
                                  const offset_t * size, int order, bool lame, bool diag)
{
    if (nbatch + ndim > MAXALL)
        throw std::logic_error("ff::cuda::reg_border: nbatch > 1 is not "
                               "supported by the CUDA launcher");
    offset_t T[ndim];
    for (int d = 0; d < ndim; ++d)
        T[d] = thickness<offset_t>(bnd[d], size[nbatch + d], order, lame, diag);
    return make_slabs<ndim, offset_t>(static_cast<int>(nbatch), size, T);
}

template <typename scalar_t, typename reduce_t, typename offset_t>
inline Args<scalar_t, reduce_t, offset_t> make_args(offset_t nbatch, int ndim, const offset_t * size)
{
    Args<scalar_t, reduce_t, offset_t> A;
    A.nbatch = static_cast<int>(nbatch);
    A.nall   = static_cast<int>(nbatch) + ndim;
    A.nc     = size[A.nall];
    if (A.nc > 3)
        throw std::logic_error("ff::cuda::reg_border: channel count outside "
                               "[1, 3] is not supported by the CUDA launcher");
    for (int d = 0; d <= MAXALL; ++d)
        A.size[d] = A.s_out[d] = A.s_inp[d] = A.s_wgt[d] = A.s_hes[d] = A.s_grd[d] = 0;
    for (int d = 0; d <= A.nall; ++d) A.size[d] = size[d];
    for (int d = 0; d < 3; ++d) A.vx[d] = A.a[d] = A.m[d] = A.b[d] = 0;
    A.sh = A.dv = 0;
    A.has_b = false; A.joint = false;
    A.out = nullptr; A.inp = A.wgt = A.hes = A.grd = nullptr;
    return A;
}

template <typename offset_t>
inline void fill(offset_t * dst, const offset_t * src, int n)
{ if (src) for (int d = 0; d < n; ++d) dst[d] = src[d]; }

template <int ndim, bool flow, int kind, char op,
          typename reduce_t, typename scalar_t, typename offset_t>
inline void launch_border(const bound::BoundVec & bnd, const Slabs<offset_t> & S,
                          const Args<scalar_t, reduce_t, offset_t> & A,
                          int patch, int col, cudaStream_t stream)
{
    for (int k = 0; k < S.nborder; ++k) {
        const int blocks = GET_BLOCKS(S.border[k].numel);
        FF_CUDA_LAUNCH((border_kernel<ndim, flow, reduce_t, scalar_t, offset_t>),
                       blocks, CUDA_NUM_THREADS, 0, stream,
                       bnd, S.border[k], A, kind, op, patch, col);
    }
}

// Each wrapper below takes the launcher's own *host* arguments. A weight
// stride vector is read on its batch + spatial entries, plus the channel
// entry only for a per-channel (non-joint) weight map -- JRLS launchers carry
// `nall` entries, RLS ones `nall + 1`.

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t>
inline void field_matvec_border(
    const bound::BoundVec & bnd, const Slabs<offset_t> & S, offset_t nbatch,
    scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
    const offset_t * size, const offset_t * stride_out, const offset_t * stride_inp,
    const offset_t * stride_wgt, bool joint, const reduce_t * vx,
    const reduce_t * a, const reduce_t * m, const reduce_t * b, cudaStream_t stream)
{
    if (!S.nborder) return;
    auto A = make_args<scalar_t, reduce_t, offset_t>(nbatch, ndim, size);
    fill(A.s_out, stride_out, A.nall + 1);
    fill(A.s_inp, stride_inp, A.nall + 1);
    if (wgt) fill(A.s_wgt, stride_wgt, joint ? A.nall : A.nall + 1);
    for (int d = 0; d < ndim; ++d) A.vx[d] = vx[d];
    for (offset_t c = 0; c < A.nc; ++c) { A.a[c] = a[c]; A.m[c] = m ? m[c] : 0; A.b[c] = b ? b[c] : 0; }
    A.has_b = b != nullptr; A.joint = joint;
    A.out = out; A.inp = inp; A.wgt = wgt;
    launch_border<ndim, false, 0, op>(bnd, S, A, 0, 0, stream);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t>
inline void field_diag_border(
    const bound::BoundVec & bnd, const Slabs<offset_t> & S, offset_t nbatch,
    scalar_t * out, const scalar_t * wgt,
    const offset_t * size, const offset_t * stride_out,
    const offset_t * stride_wgt, bool joint, const reduce_t * vx,
    const reduce_t * a, const reduce_t * m, const reduce_t * b, cudaStream_t stream)
{
    if (!S.nborder) return;
    auto A = make_args<scalar_t, reduce_t, offset_t>(nbatch, ndim, size);
    fill(A.s_out, stride_out, A.nall + 1);
    if (wgt) fill(A.s_wgt, stride_wgt, joint ? A.nall : A.nall + 1);
    for (int d = 0; d < ndim; ++d) A.vx[d] = vx[d];
    for (offset_t c = 0; c < A.nc; ++c) { A.a[c] = a[c]; A.m[c] = m ? m[c] : 0; A.b[c] = b ? b[c] : 0; }
    A.has_b = b != nullptr; A.joint = joint;
    A.out = out; A.wgt = wgt;
    launch_border<ndim, false, 1, op>(bnd, S, A, 0, 0, stream);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t>
inline void field_relax_border(
    const bound::BoundVec & bnd, const Slabs<offset_t> & S, int patch, int col,
    offset_t nbatch, scalar_t * sol, const scalar_t * hes, const scalar_t * grd,
    const scalar_t * wgt, const offset_t * size, const offset_t * stride_sol,
    const offset_t * stride_hes, const offset_t * stride_grd,
    const offset_t * stride_wgt, bool joint, const reduce_t * vx,
    const reduce_t * a, const reduce_t * m, const reduce_t * b, cudaStream_t stream)
{
    if (!S.nborder) return;
    auto A = make_args<scalar_t, reduce_t, offset_t>(nbatch, ndim, size);
    fill(A.s_out, stride_sol, A.nall + 1);
    fill(A.s_hes, stride_hes, A.nall + 1);
    fill(A.s_grd, stride_grd, A.nall + 1);
    if (wgt) fill(A.s_wgt, stride_wgt, joint ? A.nall : A.nall + 1);
    for (int d = 0; d < ndim; ++d) A.vx[d] = vx[d];
    for (offset_t c = 0; c < A.nc; ++c) { A.a[c] = a[c]; A.m[c] = m ? m[c] : 0; A.b[c] = b ? b[c] : 0; }
    A.has_b = b != nullptr; A.joint = joint;
    A.out = sol; A.hes = hes; A.grd = grd; A.wgt = wgt;
    launch_border<ndim, false, 2, '='>(bnd, S, A, patch, col, stream);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t>
inline Args<scalar_t, reduce_t, offset_t> flow_args(
    offset_t nbatch, const offset_t * size, const reduce_t * vx,
    reduce_t a, reduce_t m, reduce_t b, reduce_t sh, reduce_t dv,
    const scalar_t * wgt, const offset_t * stride_wgt)
{
    auto A = make_args<scalar_t, reduce_t, offset_t>(nbatch, ndim, size);
    for (int d = 0; d < ndim; ++d) A.vx[d] = vx[d];
    A.a[0] = a; A.m[0] = m; A.b[0] = b; A.sh = sh; A.dv = dv;
    A.wgt = wgt;
    if (wgt) fill(A.s_wgt, stride_wgt, A.nall);
    return A;
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t>
inline void flow_matvec_border(
    const bound::BoundVec & bnd, const Slabs<offset_t> & S, offset_t nbatch,
    scalar_t * out, const scalar_t * inp, const scalar_t * wgt,
    const offset_t * size, const offset_t * stride_out, const offset_t * stride_inp,
    const offset_t * stride_wgt, const reduce_t * vx,
    reduce_t a, reduce_t m, reduce_t b, reduce_t sh, reduce_t dv, cudaStream_t stream)
{
    if (!S.nborder) return;
    auto A = flow_args<ndim>(nbatch, size, vx, a, m, b, sh, dv, wgt, stride_wgt);
    fill(A.s_out, stride_out, A.nall + 1);
    fill(A.s_inp, stride_inp, A.nall + 1);
    A.out = out; A.inp = inp;
    launch_border<ndim, true, 0, op>(bnd, S, A, 0, 0, stream);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t>
inline void flow_diag_border(
    const bound::BoundVec & bnd, const Slabs<offset_t> & S, offset_t nbatch,
    scalar_t * out, const scalar_t * wgt,
    const offset_t * size, const offset_t * stride_out,
    const offset_t * stride_wgt, const reduce_t * vx,
    reduce_t a, reduce_t m, reduce_t b, reduce_t sh, reduce_t dv, cudaStream_t stream)
{
    if (!S.nborder) return;
    auto A = flow_args<ndim>(nbatch, size, vx, a, m, b, sh, dv, wgt, stride_wgt);
    fill(A.s_out, stride_out, A.nall + 1);
    A.out = out;
    launch_border<ndim, true, 1, op>(bnd, S, A, 0, 0, stream);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t>
inline void flow_relax_border(
    const bound::BoundVec & bnd, const Slabs<offset_t> & S, int patch, int col,
    offset_t nbatch, scalar_t * sol, const scalar_t * hes, const scalar_t * grd,
    const scalar_t * wgt, const offset_t * size, const offset_t * stride_sol,
    const offset_t * stride_hes, const offset_t * stride_grd,
    const offset_t * stride_wgt, const reduce_t * vx,
    reduce_t a, reduce_t m, reduce_t b, reduce_t sh, reduce_t dv, cudaStream_t stream)
{
    if (!S.nborder) return;
    auto A = flow_args<ndim>(nbatch, size, vx, a, m, b, sh, dv, wgt, stride_wgt);
    fill(A.s_out, stride_sol, A.nall + 1);
    fill(A.s_hes, stride_hes, A.nall + 1);
    fill(A.s_grd, stride_grd, A.nall + 1);
    A.out = sol; A.hes = hes; A.grd = grd;
    launch_border<ndim, true, 2, '='>(bnd, S, A, patch, col, stream);
}

FF_NAMESPACE_END(reg_border)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
