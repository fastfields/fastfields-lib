#ifndef FF_POSDEF_CUDA
#define FF_POSDEF_CUDA
// Teeny-based CUDA posdef impl -- the device mirror of the CPU launcher
// (fastfields-cpu-impl/posdef.h). Same math, same representation:
//
//   * POINTWISE over (*batch, packed): teeny's peel hands each voxel's rank-1
//     cell (the matrix/vector for that voxel) to the shared single-voxel kernels
//     in kernels/posdef/matrix.h. `peel_front_at<-1>` folds the (arbitrarily
//     strided) batch offset into each cell's pointer -- the same call the CPU
//     body uses, so "CPU works + CUDA compiles" gives real confidence they
//     compute the same thing. NO host precompute, NO atomics (each voxel is
//     independent -> disjoint writes).
//
// Device port vs. the CPU version:
//   * the batch loop is a `__global__` grid-stride loop over the voxels instead
//     of parallel_for; every tensor is wrapped as a DEVICE-PASSABLE teeny anyrank
//     carrier (`as_anyrank<TNY_MAX_RANK, storage::gpu_view>(..., copy_meta)` --
//     the shape/stride travel INLINE with the carrier, so it is trivially
//     copyable and passes into the kernel BY VALUE; no separate device copy of
//     shape/stride);
//   * the Cholesky path (Sym/Full solve / invert) needs a per-voxel CxC double
//     workspace. The CPU body heap-allocates it per thread; on the device we use
//     a STACK buffer sized to a compile-time cap FF_POSDEF_MAX_C and wrap a
//     runtime CxC view over it. The host launcher rejects nchannel > the cap.
//
// The impl receives ONE `size` array (shared batch dims + one trailing), but the
// tensors differ in trailing length (vectors = C, packed hessian = C(C+1)/2), so
// each anyrank is built with its OWN trailing extent over the shared batch dims.
#include <teeny/teeny.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "kernels/cuda_switch.h"
// kernels/posdef/matrix.h aliases `namespace cs = cuda::std;` at the (ff::cuda)
// FF_DEVICE namespace scope, where unqualified `cuda` binds to the enclosing
// ff::cuda namespace rather than the global ::cuda -- a latent bug that only
// surfaces under nvcc (it is fine for the ff::cpu host build, which never
// shadows `cuda`). Inject ff::cuda::cuda -> ::cuda so `cuda::std` resolves to
// the global ::cuda::std, WITHOUT touching the shared kernels header and WITHOUT
// shadowing the real ::std (an ff::cuda::std alias would break every bare
// std:: use inside ff::cuda). Teeny above already pulls in ::cuda::std.
#ifdef __CUDACC__
namespace ff { namespace cuda { namespace cuda = ::cuda; } }
#endif
#include "kernels/posdef/matrix.h"
#include "utils.h"                    // GET_BLOCKS / CUDA_NUM_THREADS

// Largest channel count the Cholesky (Sym/Full) solve / invert device paths
// support: they carry a per-voxel CxC double workspace on the stack, sized to
// this cap. The host launcher throws for a larger channel count.
#ifndef FF_POSDEF_MAX_C
#define FF_POSDEF_MAX_C 16
#endif

// Largest total tensor rank (batch dims + 1 trailing) a device-passable anyrank
// carrier is sized for. A carrier holds its shape+stride INLINE (~2*MaxRank*8
// bytes), and a kernel gets at most 4 KiB of by-value parameter space -- solve
// passes FOUR carriers, so the full TNY_MAX_RANK (64) would overflow it. This
// cap keeps four carriers well within the limit while staying far more generous
// than the legacy launcher (which capped nbatch at 7). The host launcher throws
// for a deeper tensor.
#ifndef FF_POSDEF_MAX_RANK
#define FF_POSDEF_MAX_RANK 32
#endif

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(posdef)

// Device-passable anyrank carrier over the shared batch dims + one trailing dim.
// The DATA pointer lives in device memory (storage::gpu_view); shape/stride are
// COPIED inline (copy_meta) so the carrier passes into the kernel by value.
template <typename T, typename offset_t>
static inline auto _any(T* p, const offset_t* size, offset_t nbatch,
                        offset_t trailing, const offset_t* stride)
{
    if (nbatch + 1 > FF_POSDEF_MAX_RANK)
        throw std::logic_error("posdef: tensor rank too large for the CUDA launcher");
    std::vector<offset_t> sz(size, size + nbatch);
    sz.push_back(trailing);
    return tny::as_anyrank<FF_POSDEF_MAX_RANK, tny::storage::gpu_view>(
        p, sz.data(), stride, static_cast<int>(nbatch + 1), tny::copy_meta);
}

// packed length CC for a layout at channel count C (static path uses C>0).
template <type Ty, int C, typename offset_t>
static inline offset_t _packed(offset_t nchannel)
{
    const offset_t c = (C > 0) ? offset_t(C) : nchannel;
    switch (Ty) {
        case type::Eye:      return 1;
        case type::Diag:     return c;
        case type::ESTATICS: return 2 * c - 1;
        case type::Sym:      return c * (c + 1) / 2;
        default:             return c * c;         // Full
    }
}
template <type Ty, int C>   // compile-time CC for the static path (C>0)
CUHOSTDEV static constexpr long _packed_s()
{
    return Ty == type::Eye ? 1 : Ty == type::Diag ? C
         : Ty == type::ESTATICS ? 2L * C - 1 : Ty == type::Sym ? long(C) * (C + 1) / 2
         : long(C) * C;
}

template <type Ty, wr W, class Ov, class Hv, class Xv>
CUDEV static inline void _dispatch_matvec(Ov&& o, const Hv& h, const Xv& x)
{
    if constexpr      (Ty == type::Eye)      eye::matvec<W>(o, h, x);
    else if constexpr (Ty == type::Diag)     diag::matvec<W>(o, h, x);
    else if constexpr (Ty == type::ESTATICS) estatics::matvec<W>(o, h, x);
    else if constexpr (Ty == type::Sym)      sym::matvec<W>(o, h, x);
    else                                     full::matvec<W>(o, h, x);
}

// in-place solve v <- (H + diag(w)) \ v for the selected layout. Sym/Full take
// the CxC double workspace M (Cholesky); Eye/Diag/ESTATICS ignore it.
template <type Ty, class Vv, class Hv, class Mv>
CUDEV static inline void _dispatch_solve(Vv&& v, const Hv& h, Mv& M)
{
    if constexpr      (Ty == type::Eye)      eye::solve_(v, h);
    else if constexpr (Ty == type::Diag)     diag::solve_(v, h);
    else if constexpr (Ty == type::ESTATICS) estatics::solve_(v, h);
    else if constexpr (Ty == type::Sym)      sym::solve_w_(v, h, M);
    else                                     full::solve_w_(v, h, M);
}
template <type Ty, class Vv, class Hv, class Wv, class Mv>
CUDEV static inline void _dispatch_solve(Vv&& v, const Hv& h, const Wv& w, Mv& M)
{
    if constexpr      (Ty == type::Eye)      eye::solve_(v, h, w);
    else if constexpr (Ty == type::Diag)     diag::solve_(v, h, w);
    else if constexpr (Ty == type::ESTATICS) estatics::solve_(v, h, w);
    else if constexpr (Ty == type::Sym)      sym::solve_w_(v, h, w, M);
    else                                     full::solve_w_(v, h, w, M);
}

// ---- matvec family (set / add / sub), any layout --------------------------
template <type Ty, wr W, int C, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AH, class AI>
CUGLOB void _matvec_kernel(AO ao, AH ah, AI ai, offset_t nvox)
{
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox;
         i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto o = ao.template peel_front_at<-1>(i);
        auto h = ah.template peel_front_at<-1>(i);
        auto x = ai.template peel_front_at<-1>(i);
        if constexpr (C > 0)
            _dispatch_matvec<Ty, W>(o.recast(tny::shape<C>{}),
                                    h.recast(tny::shape<_packed_s<Ty, C>()>{}),
                                    x.recast(tny::shape<C>{}));
        else
            _dispatch_matvec<Ty, W>(o, h, x);
    }
}

template <type Ty, wr W, int C, typename reduce_t, typename scalar_t, typename offset_t>
static void _matvec(
          offset_t   nbatch,   offset_t nchannel,
          scalar_t * out, const scalar_t * hes, const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out, const offset_t * stride_hes, const offset_t * stride_inp,
          int        stream)
{
    const offset_t CC = _packed<Ty, C>(nchannel);
    auto ao = _any(out, size, nbatch, nchannel, stride_out);
    auto ah = _any(hes, size, nbatch, CC,       stride_hes);
    auto ai = _any(inp, size, nbatch, nchannel, stride_inp);
    const offset_t nvox = ao.template size_front<-1>();

    cudaStream_t s = reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream));
    const int blocks  = GET_BLOCKS(nvox);
    const int threads = CUDA_NUM_THREADS;
    _matvec_kernel<Ty, W, C, reduce_t, scalar_t, offset_t>
        <<<blocks, threads, 0, s>>>(ao, ah, ai, nvox);
}

template <type Ty, int C, typename reduce_t, typename scalar_t, typename offset_t>
void matvec(offset_t nbatch, offset_t nchannel, scalar_t* out,
                const scalar_t* hes, const scalar_t* inp, const offset_t* size,
                const offset_t* stride_out, const offset_t* stride_hes, const offset_t* stride_inp,
                int stream = 0)
{ _matvec<Ty, wr::set, C, reduce_t>(nbatch, nchannel, out, hes, inp, size, stride_out, stride_hes, stride_inp, stream); }

template <type Ty, int C, typename reduce_t, typename scalar_t, typename offset_t>
void addmatvec_(offset_t nbatch, offset_t nchannel, scalar_t* out,
                const scalar_t* hes, const scalar_t* inp, const offset_t* size,
                const offset_t* stride_out, const offset_t* stride_hes, const offset_t* stride_inp,
                int stream = 0)
{ _matvec<Ty, wr::add, C, reduce_t>(nbatch, nchannel, out, hes, inp, size, stride_out, stride_hes, stride_inp, stream); }

template <type Ty, int C, typename reduce_t, typename scalar_t, typename offset_t>
void submatvec_(offset_t nbatch, offset_t nchannel, scalar_t* out,
                const scalar_t* hes, const scalar_t* inp, const offset_t* size,
                const offset_t* stride_out, const offset_t* stride_hes, const offset_t* stride_inp,
                int stream = 0)
{ _matvec<Ty, wr::sub, C, reduce_t>(nbatch, nchannel, out, hes, inp, size, stride_out, stride_hes, stride_inp, stream); }

// ---- matvec_backward: out(packed) = grad wrt H of <grd, H inp> -------------
template <int C, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AG, class AI>
CUGLOB void _matvec_backward_kernel(AO ao, AG ag, AI ai, offset_t nvox)
{
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox;
         i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto o = ao.template peel_front_at<-1>(i);
        auto g = ag.template peel_front_at<-1>(i);
        auto x = ai.template peel_front_at<-1>(i);
        if constexpr (C > 0) {
            constexpr long Cs = C, CCs = static_cast<long>(C) * (C + 1) / 2;
            sym::matvec_backward(o.recast(tny::shape<CCs>{}), x.recast(tny::shape<Cs>{}),
                                 g.recast(tny::shape<Cs>{}));
        } else {
            sym::matvec_backward(o, x, g);
        }
    }
}

template <int C, typename reduce_t, typename scalar_t, typename offset_t>
void sym_matvec_backward(offset_t nbatch, offset_t nchannel, scalar_t* out,
                const scalar_t* grd, const scalar_t* inp, const offset_t* size,
                const offset_t* stride_out, const offset_t* stride_grd, const offset_t* stride_inp,
                int stream = 0)
{
    const offset_t CC = _packed<type::Sym, C>(nchannel);
    auto ao = _any(out, size, nbatch, CC,       stride_out);
    auto ag = _any(grd, size, nbatch, nchannel, stride_grd);
    auto ai = _any(inp, size, nbatch, nchannel, stride_inp);
    const offset_t nvox = ag.template size_front<-1>();

    cudaStream_t s = reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream));
    const int blocks  = GET_BLOCKS(nvox);
    const int threads = CUDA_NUM_THREADS;
    _matvec_backward_kernel<C, reduce_t, scalar_t, offset_t>
        <<<blocks, threads, 0, s>>>(ao, ag, ai, nvox);
}

// ---- solve: out = (H + diag(w)) \ inp  (w optional via null wgt) -----------
template <type Ty, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, class AH, class AW>
CUGLOB void _solve_kernel(AO ao, AI ai, AH ah, AW aw,
                          offset_t nvox, offset_t nchannel, bool have_w)
{
    reduce_t buf[FF_POSDEF_MAX_C * FF_POSDEF_MAX_C];   // per-voxel Cholesky workspace (Sym/Full)
    auto M = tny::wrap(buf, tny::shape<-1,-1>{nchannel, nchannel});
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox;
         i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto o = ao.template peel_front_at<-1>(i);
        auto x = ai.template peel_front_at<-1>(i);
        auto h = ah.template peel_front_at<-1>(i);
        o.copy_(x);
        if (have_w) _dispatch_solve<Ty>(o, h, aw.template peel_front_at<-1>(i), M);
        else        _dispatch_solve<Ty>(o, h, M);
    }
}

template <type Ty, typename reduce_t, typename scalar_t, typename offset_t>
void solve(offset_t nbatch, offset_t nchannel, scalar_t* out, const scalar_t* inp,
                const scalar_t* hes, const scalar_t* wgt, const offset_t* size,
                const offset_t* stride_out, const offset_t* stride_inp,
                const offset_t* stride_hes, const offset_t* stride_wgt,
                int stream = 0)
{
    if (nchannel > FF_POSDEF_MAX_C)
        throw std::logic_error("posdef: nchannel too large for the CUDA solve workspace");
    const offset_t CC = _packed<Ty, -1>(nchannel);
    auto ao = _any(out, size, nbatch, nchannel, stride_out);
    auto ai = _any(inp, size, nbatch, nchannel, stride_inp);
    auto ah = _any(hes, size, nbatch, CC,       stride_hes);
    const offset_t nvox = ao.template size_front<-1>();
    const bool have_w = (wgt != nullptr);
    auto aw = _any(have_w ? wgt : inp, size, nbatch, nchannel, have_w ? stride_wgt : stride_inp);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream));
    const int blocks  = GET_BLOCKS(nvox);
    const int threads = CUDA_NUM_THREADS;
    _solve_kernel<Ty, reduce_t, scalar_t, offset_t>
        <<<blocks, threads, 0, s>>>(ao, ai, ah, aw, nvox, nchannel, have_w);
}

// ---- solve_: in place, inp_out = (H + diag(w)) \ inp_out -------------------
template <type Ty, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AH, class AW>
CUGLOB void _solve__kernel(AO ao, AH ah, AW aw,
                           offset_t nvox, offset_t nchannel, bool have_w)
{
    reduce_t buf[FF_POSDEF_MAX_C * FF_POSDEF_MAX_C];   // per-voxel Cholesky workspace (Sym/Full)
    auto M = tny::wrap(buf, tny::shape<-1,-1>{nchannel, nchannel});
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox;
         i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto o = ao.template peel_front_at<-1>(i);
        auto h = ah.template peel_front_at<-1>(i);
        if (have_w) _dispatch_solve<Ty>(o, h, aw.template peel_front_at<-1>(i), M);
        else        _dispatch_solve<Ty>(o, h, M);
    }
}

template <type Ty, typename reduce_t, typename scalar_t, typename offset_t>
void solve_(offset_t nbatch, offset_t nchannel, scalar_t* out, const scalar_t* hes,
                const scalar_t* wgt, const offset_t* size,
                const offset_t* stride_out, const offset_t* stride_hes, const offset_t* stride_wgt,
                int stream = 0)
{
    if (nchannel > FF_POSDEF_MAX_C)
        throw std::logic_error("posdef: nchannel too large for the CUDA solve workspace");
    const offset_t CC = _packed<Ty, -1>(nchannel);
    auto ao = _any(out, size, nbatch, nchannel, stride_out);
    auto ah = _any(hes, size, nbatch, CC,       stride_hes);
    const offset_t nvox = ao.template size_front<-1>();
    const bool have_w = (wgt != nullptr);
    auto aw = _any(have_w ? wgt : out, size, nbatch, nchannel, have_w ? stride_wgt : stride_out);

    cudaStream_t s = reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream));
    const int blocks  = GET_BLOCKS(nvox);
    const int threads = CUDA_NUM_THREADS;
    _solve__kernel<Ty, reduce_t, scalar_t, offset_t>
        <<<blocks, threads, 0, s>>>(ao, ah, aw, nvox, nchannel, have_w);
}

// ---- invert: out = inv(H) (out-of-place) ----------------------------------
template <typename reduce_t, typename scalar_t, typename offset_t, class AO, class AH>
CUGLOB void _invert_kernel(AO ao, AH ah, offset_t nvox, offset_t nchannel, offset_t CC)
{
    reduce_t buf[FF_POSDEF_MAX_C * FF_POSDEF_MAX_C];
    auto M = tny::wrap(buf, tny::shape<-1,-1>{nchannel, nchannel});
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox;
         i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto o = ao.template peel_front_at<-1>(i);
        auto h = ah.template peel_front_at<-1>(i);
        for (offset_t k = 0; k < CC; ++k) o(k) = static_cast<reduce_t>(h(k));
        sym::invert_w_(o, M);
    }
}

template <typename reduce_t, typename scalar_t, typename offset_t>
void sym_invert(offset_t nbatch, offset_t nchannel, scalar_t* out, const scalar_t* hes,
                const offset_t* size, const offset_t* stride_out, const offset_t* stride_hes,
                int stream = 0)
{
    if (nchannel > FF_POSDEF_MAX_C)
        throw std::logic_error("posdef: nchannel too large for the CUDA invert workspace");
    const offset_t CC = nchannel * (nchannel + 1) / 2;
    auto ao = _any(out, size, nbatch, CC, stride_out);
    auto ah = _any(hes, size, nbatch, CC, stride_hes);
    const offset_t nvox = ao.template size_front<-1>();

    cudaStream_t s = reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream));
    const int blocks  = GET_BLOCKS(nvox);
    const int threads = CUDA_NUM_THREADS;
    _invert_kernel<reduce_t, scalar_t, offset_t>
        <<<blocks, threads, 0, s>>>(ao, ah, nvox, nchannel, CC);
}

// ---- invert_: in place, hes = inv(hes) ------------------------------------
template <typename reduce_t, typename scalar_t, typename offset_t, class AH>
CUGLOB void _invert__kernel(AH ah, offset_t nvox, offset_t nchannel)
{
    reduce_t buf[FF_POSDEF_MAX_C * FF_POSDEF_MAX_C];
    auto M = tny::wrap(buf, tny::shape<-1,-1>{nchannel, nchannel});
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox;
         i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto h = ah.template peel_front_at<-1>(i);
        sym::invert_w_(h, M);
    }
}

template <typename reduce_t, typename scalar_t, typename offset_t>
void sym_invert_(offset_t nbatch, offset_t nchannel, scalar_t* hes,
                const offset_t* size, const offset_t* stride,
                int stream = 0)
{
    if (nchannel > FF_POSDEF_MAX_C)
        throw std::logic_error("posdef: nchannel too large for the CUDA invert workspace");
    const offset_t CC = nchannel * (nchannel + 1) / 2;
    auto ah = _any(hes, size, nbatch, CC, stride);
    const offset_t nvox = ah.template size_front<-1>();

    cudaStream_t s = reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream));
    const int blocks  = GET_BLOCKS(nvox);
    const int threads = CUDA_NUM_THREADS;
    _invert__kernel<reduce_t, scalar_t, offset_t>
        <<<blocks, threads, 0, s>>>(ah, nvox, nchannel);
}

FF_NAMESPACE_END(posdef)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_POSDEF_CUDA
