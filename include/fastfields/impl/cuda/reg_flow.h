#ifndef FF_REGULARISERS_FLOW_CUDA
#define FF_REGULARISERS_FLOW_CUDA
// Teeny-based CUDA reg_flow impl -- the device mirror of the CPU launcher
// (fastfields-cpu-impl/reg_flow.h). Same math, same representation:
//
//   * ABSOLUTE is POINTWISE: teeny's peel hands each (*batch,*spatial) voxel's
//     rank-1 channel cell to the shared single-voxel kernel (kernels/
//     regularisers/flow). `peel_front_at<-1>` folds the (arbitrarily strided)
//     batch/spatial offset into each cell's pointer -- the same call the CPU body
//     uses, so "CPU works + CUDA compiles" gives real confidence they compute the
//     same thing. NO host precompute of offsets, NO atomics (disjoint writes).
//   * MEMBRANE / BENDING are STENCIL ops: they gather spatial NEIGHBOURS with
//     boundary conditions, so each voxel needs its spatial multi-index `loc` and
//     the spatial size/stride. We peel the (*spatial,C) volume of a batch cell
//     with `peel_front_at<-(ndim+1)>(b)`, decode `loc` within it, offset each base
//     pointer, and call the single-voxel kernel -- exactly the CPU loop body.
//
// Device port vs. the CPU version:
//   * the parallel_for becomes a `__global__` grid-stride loop over the voxels;
//     each tensor is wrapped as a DEVICE-PASSABLE teeny anyrank carrier
//     (`as_anyrank<TNY_MAX_RANK, storage::gpu_view>(..., copy_meta)` -- shape/
//     stride travel INLINE, so the carrier is trivially copyable and passes into
//     the kernel BY VALUE; no separate device copy of shape/stride);
//   * the (small, compile-time-sized) convolution kernel table is built ON THE
//     HOST (identical to the CPU impl) and cudaMemcpy'd to the device;
//   * the spatial size/stride the stencil single-voxel kernels index are copied
//     into a tiny by-value POD (`flow_sp`) passed in the launch -- ndim <= 3.
//   * this teeny launcher handles an ARBITRARY batch rank (the batch offset folds
//     into the peel), unlike the legacy launcher which capped nbatch at 3.
//
//   * LAME / ALL are stencil ops too, with the same shape as membrane/bending;
//     `kernel_lame`/`kernel_all` write a (C,C) block, so they take the TWO
//     trailing strides rather than one scalar channel stride.
//   * RELAX is a COLOURED Gauss-Seidel sweep. The CPU launcher (phase 4) walks
//     one dense `subsample` sub-lattice per colour; here each colour becomes its
//     OWN kernel LAUNCH on the caller's stream -- see `colour_lattice` below for
//     why a launch (and not an in-kernel colour loop) is the only correct
//     mirror. Because a flow's channel count IS ndim, every per-voxel relax
//     scratch buffer is COMPILE-TIME sized and lives on the device stack -- no
//     channel cap and no device allocation (contrast reg_field, whose runtime C
//     needs both).
//   * JRLS (membrane, lame) adds a per-voxel WEIGHT map, which rides along as
//     one more anyrank carrier plus its spatial stride row in the by-value POD.
//
// The op ('=','+','-') is threaded through exactly as the CPU `op_apply` does:
// `Op<op,scalar_t,reduce_t>::f` is the function-pointer non-type template arg the
// single-voxel kernels take (the C++17 device path, same as the legacy launcher).
// The relax bodies want the CPU's bare `isub`/`set` op ids; `Op<'-',...>::f` and
// `Op<'=',...>::f` ARE those two functions (kernels/regularisers/flow/utils.h),
// so they are spelled that way here to stay on this file's existing idiom.
#include <teeny/teeny.h>
#include <cstdint>
#include <stdexcept>
#include "kernels/cuda_switch.h"
#include "kernels/bounds.h"
#include "kernels/utils.h"
#include "kernels/regularisers/flow.h"
// See posdef.h / reg_field.h for this guard: kernels/posdef/matrix.h aliases
// `namespace cs = cuda::std;` at the (ff::cuda) FF_DEVICE namespace scope, where
// unqualified `cuda` binds to the enclosing ff::cuda namespace instead of the
// global ::cuda -- fine for the ff::cpu host build, a hard error under nvcc.
// Inject ff::cuda::cuda -> ::cuda before pulling that header in.
#ifdef __CUDACC__
namespace ff { namespace cuda { namespace cuda = ::cuda; } }
#endif
#include "kernels/posdef.h"
#include "utils.h"                    // GET_BLOCKS / CUDA_NUM_THREADS / copyToDevice / freeDevice

// Largest total tensor rank a relax_* device-passable carrier is sized for. A
// carrier holds its shape+stride INLINE (measured: 1040 bytes at TNY_MAX_RANK=64,
// 528 at 32), and a kernel gets at most 4 KiB of by-value parameter space -- the
// JRLS relax kernels pass FOUR carriers (sol/hes/grd/wgt), so the uncapped
// TNY_MAX_RANK=64 would need 4168 bytes and nvcc rejects the launch outright
// ("Formal parameter space overflowed"). Capping the relax carriers at 32 puts
// four of them at 2112 bytes, comfortably inside it. The host launcher throws
// for a deeper tensor. matvec/diag/kernel pass at most three carriers of the
// uncapped kind and are left exactly as they were.
#ifndef FF_REG_FLOW_MAX_RANK
#define FF_REG_FLOW_MAX_RANK 32
#endif

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_flow)

// Device-passable anyrank carrier over (*batch, *spatial, C). Shape/stride are
// COPIED inline (copy_meta) so the carrier passes into the kernel by value; the
// DATA pointer lives in device memory (storage::gpu_view).
template <typename T, typename offset_t>
static inline auto _any(T* p, const offset_t* size, const offset_t* stride, offset_t nall)
{
    return tny::as_anyrank<TNY_MAX_RANK, tny::storage::gpu_view>(
        p, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);
}

// Tiny by-value spatial metadata (spatial extents + spatial out/inp strides) the
// stencil single-voxel kernels index. ndim <= 3, so passing it by value into the
// kernel is trivial (no device copy of the shape/stride arrays).
template <int ndim, typename offset_t>
struct flow_sp {
    offset_t size[ndim];   // spatial extents  (size[nbatch + d])
    offset_t sout[ndim];   // spatial strides of out
    offset_t sinp[ndim];   // spatial strides of inp
};

// Device-passable carrier over (*batch, *spatial) -- NO trailing channel axis.
// The kernel_* launchers only need each batch cell's origin pointer (they write
// the stencil at the spatial centre), so their index domain is one rank shorter.
template <typename T, typename offset_t>
static inline auto _any_nb(T* p, const offset_t* size, const offset_t* stride, offset_t nall)
{
    return tny::as_anyrank<TNY_MAX_RANK, tny::storage::gpu_view>(
        p, size, stride, static_cast<int>(nall), tny::copy_meta);
}

// Rank-CAPPED variant, used by the relax_* launchers only -- they pass up to
// FOUR carriers in one launch (see FF_REG_FLOW_MAX_RANK).
template <typename T, typename offset_t>
static inline auto _any_rx(T* p, const offset_t* size, const offset_t* stride, offset_t rank)
{
    if (rank > static_cast<offset_t>(FF_REG_FLOW_MAX_RANK))
        throw std::logic_error("reg_flow: tensor rank too large for the CUDA relax launcher");
    return tny::as_anyrank<FF_REG_FLOW_MAX_RANK, tny::storage::gpu_view>(
        p, size, stride, static_cast<int>(rank), tny::copy_meta);
}

// Spatial stride row of the JRLS weight map. Kept separate from flow_sp so the
// existing non-weighted kernels keep their exact parameter list.
template <int ndim, typename offset_t>
struct flow_wp {
    offset_t swgt[ndim];   // spatial strides of wgt (stride_wgt[nbatch + d])
};

// By-value metadata for the kernel_* launchers: the spatial extents/strides the
// centre offset is computed from, plus the trailing channel stride(s). The
// absolute/membrane/bending kernel writers take ONE scalar channel stride (`sc`);
// lame/all write a (C,C) block and take a POINTER to the two trailing strides
// (`sc2`), which is why both spellings are carried here.
template <int ndim, typename offset_t>
struct flow_kp {
    offset_t size[ndim];
    offset_t sout[ndim];
    offset_t sc;
    offset_t sc2[2];
};

//----------------------------------------------------------------------
//  Coloured relaxation: the colour -> dense sub-lattice decomposition
//----------------------------------------------------------------------
// Transcribed from the CPU launcher (fastfields-cpu-impl/reg_flow.h, phase 4) so
// the two produce the SAME colour classes in the SAME order. A coloured
// Gauss-Seidel sweep relaxes one colour class per pass; each class is a union of
// *Cartesian* sub-lattices `loc[d] % k == start[d]`.
//
//   AXIS-INDEPENDENT (`patch2`, k=2; `patch3`, k=3) -- k^ndim colours, and a
//   colour IS a single sub-lattice: `start[d]` = digit `d` of the colour in
//   base k. One sub-lattice, no union.
//
//   CHECKERBOARD (`patch1`, k=2) -- 2 colours, `(sum_d loc[d]) % 2 == colour`.
//   That is diagonal, not Cartesian, so it is the UNION of the 2^(ndim-1)
//   sub-lattices whose start vector has parity `colour` (at most 4, ndim <= 3).
//   Keeping the 2 checkerboard classes -- rather than the finer 2^ndim
//   axis-independent colouring -- is deliberate: it preserves the exact per-pass
//   iterate sequence the predicate form produced for niter > 1. That design
//   decision (and its known pre-existing limitation under DFT boundaries with
//   incompatible axis lengths) is inherited here unchanged from cpu-impl#49/#50
//   and kernels#66 -- it is not re-litigated or "fixed" in the device port.
//
// WHY ONE KERNEL LAUNCH PER COLOUR, and not one launch with an in-kernel colour
// loop: the colouring exists precisely because colour c+1 READS voxels colour c
// just WROTE, so every colour boundary is a true GLOBAL barrier across the whole
// grid. `__syncthreads()` only synchronises within a block, so an in-kernel
// colour loop would be racy across blocks -- silently producing a different (and
// wrong) iterate sequence rather than failing loudly. A grid-wide barrier would
// need cooperative-groups launches, which this codebase does not use anywhere
// and which would cap the grid at the resident-block count. Successive launches
// on ONE stream are ordered by the stream itself, so the launch boundary IS the
// barrier, for free and with no occupancy cost. That also means only a single
// `cudaStreamSynchronize` is needed, after the last colour.

constexpr int colour_ipow(int k, int n)
{
    return n <= 0 ? 1 : k * colour_ipow(k, n - 1);
}

template <int ndim, int k, bool checker>
struct colour_lattice
{
    static constexpr int ncolour = checker ? 2 : colour_ipow(k, ndim);
    static constexpr int nsub    = checker ? (1 << (ndim - 1)) : 1;

    // Fill `start[0..ndim)` for sub-lattice `j` of the colour that pass `n`
    // selects. HOST-only: it runs in the launcher, between launches.
    template <typename offset_t>
    CUHOST static void starts(offset_t n, int j, offset_t start[ndim])
    {
        if (checker)
        {
            int par = 0;
            for (int d = 0; d < ndim - 1; ++d)
            {
                const int s = (j >> d) & 1;
                start[d] = static_cast<offset_t>(s);
                par ^= s;
            }
            start[ndim-1] = static_cast<offset_t>(static_cast<int>(n % 2) ^ par);
        }
        else
        {
            offset_t c = n % static_cast<offset_t>(ncolour);
            for (int d = 0; d < ndim; ++d, c /= static_cast<offset_t>(k))
                start[d] = c % static_cast<offset_t>(k);
        }
    }
};

// Per-axis extents of the spatial sub-lattice `loc[d] % k == start[d]`, and
// their product -- the CPU's `sublattice_numel` with the per-axis extents kept
// as well, because a grid-stride loop needs random access by flat index and so
// decodes the sub-lattice index itself. Same formula, one place.
template <int ndim, typename offset_t>
CUHOST inline offset_t sublattice_extents(const offset_t * size, offset_t k,
                                          const offset_t * start, offset_t * nsub)
{
    offset_t n = 1;
    for (int d = 0; d < ndim; ++d)
    {
        const offset_t e = size[d] - start[d];
        nsub[d] = (e > 0) ? (e + k - 1) / k : static_cast<offset_t>(0);
        n *= nsub[d];
    }
    return n;
}

// By-value spatial metadata for ONE coloured relax launch.
template <int ndim, typename offset_t>
struct flow_rx {
    offset_t size  [ndim];   // FULL spatial extents (the kernels' `size + nbatch`)
    offset_t nsub  [ndim];   // sub-lattice extents  (this colour)
    offset_t cstart[ndim];   // sub-lattice origin   (this colour)
    offset_t kstep;          // sub-lattice step
    offset_t ssol  [ndim];
    offset_t shes  [ndim];
    offset_t sgrd  [ndim];
    offset_t swgt  [ndim];   // JRLS only; else unused
};

// Decode a flat sub-lattice index into the voxel's FULL spatial index `loc`.
// Device twin of the CPU's `loc[d] = cstart[d] + kstep * v.index[d]`, where
// `v.index` is the multi-index of teeny's peel over the `subsample` view. Peeled
// axes vary row-major (last axis fastest), so the decode runs last-axis-first to
// match. NB the visiting ORDER within a colour is immaterial -- every voxel of
// one colour is independent by construction -- so this only has to enumerate the
// same SET.
template <int ndim, typename offset_t>
CUDEV inline void decode_sublattice(offset_t s, const flow_rx<ndim, offset_t> & rx,
                                    offset_t * loc)
{
    for (int d = ndim - 1; d >= 0; --d)
    {
        const offset_t c = s % rx.nsub[d];
        s /= rx.nsub[d];
        loc[d] = rx.cstart[d] + rx.kstep * c;
    }
}

template <int ndim, typename offset_t>
CUDEV inline offset_t spatial_offset(const offset_t * loc, const offset_t * stride)
{
    offset_t o = 0;
    for (int d = 0; d < ndim; ++d) o += loc[d] * stride[d];
    return o;
}

//======================================================================
//                              ABSOLUTE  (pointwise)
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, bound::type... BOUND>
CUGLOB void _matvec_absolute_k(AO ao, AI ai, const reduce_t* kernel,
                               offset_t osc, offset_t isc, offset_t nvox)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        auto ic = ai.template peel_front_at<-1>(i);
        Impl::template matvec_absolute<opfunc>(oc.data(), ic.data(), osc, isc, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void matvec_absolute(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_out[nall], isc = stride_inp[nall];

    reduce_t kbuf[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kbuf, absolute, voxel_size);

    auto ao = _any(out, size, stride_out, nall);
    auto ai = _any(inp, size, stride_inp, nall);
    const offset_t nvox = ao.template size_front<-1>();

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_absolute));
    try {
        _matvec_absolute_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), decltype(ai), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, ai, d_kernel, osc, isc, nvox);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _diag_absolute_k(AO ao, const reduce_t* kernel, offset_t sc, offset_t nvox)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        Impl::template diag_absolute<opfunc>(oc.data(), sc, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_absolute(
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t sc = stride[nall];

    reduce_t kbuf[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kbuf, absolute, voxel_size);

    auto ao = _any(out, size, stride, nall);
    const offset_t nvox = ao.template size_front<-1>();

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_absolute));
    try {
        _diag_absolute_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, sc, nvox);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

//======================================================================
//                              MEMBRANE  (stencil)
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, bound::type... BOUND>
CUGLOB void _matvec_membrane_k(AO ao, AI ai, const reduce_t* kernel,
                               flow_sp<ndim, offset_t> sp, offset_t osc, offset_t isc,
                               offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0, io = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d]; io += c * sp.sinp[d];
        }
        Impl::template matvec_membrane<opfunc>(
            vo.data() + oo, vi.data() + io, loc, sp.size, sp.sinp, osc, isc, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void matvec_membrane(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_out[nall], isc = stride_inp[nall];

    reduce_t kbuf[Impl::kernelsize_membrane];
    Impl::make_kernel_membrane(kbuf, absolute, membrane, voxel_size);

    flow_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_inp[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto ai = _any(inp, size, stride_inp, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_membrane));
    try {
        _matvec_membrane_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), decltype(ai), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, ai, d_kernel, sp, osc, isc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _diag_membrane_k(AO ao, const reduce_t* kernel,
                             flow_sp<ndim, offset_t> sp, offset_t sc,
                             offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d];
        }
        Impl::template diag_membrane<opfunc>(vo.data() + oo, sc, loc, sp.size, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_membrane(
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t sc = stride[nall];

    reduce_t kbuf[Impl::kernelsize_membrane];
    Impl::make_kernel_membrane(kbuf, absolute, membrane, voxel_size);

    flow_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride[nbatch + d];
        sp.sinp[d] = stride[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_membrane));
    try {
        _diag_membrane_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, sp, sc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

//======================================================================
//                              BENDING  (stencil)
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, bound::type... BOUND>
CUGLOB void _matvec_bending_k(AO ao, AI ai, const reduce_t* kernel,
                              flow_sp<ndim, offset_t> sp, offset_t osc, offset_t isc,
                              offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0, io = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d]; io += c * sp.sinp[d];
        }
        Impl::template matvec_bending<opfunc>(
            vo.data() + oo, vi.data() + io, loc, sp.size, sp.sinp, osc, isc, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void matvec_bending(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     bending,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_out[nall], isc = stride_inp[nall];

    reduce_t kbuf[Impl::kernelsize_bending];
    Impl::make_kernel_bending(kbuf, absolute, membrane, bending, voxel_size);

    flow_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_inp[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto ai = _any(inp, size, stride_inp, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_bending));
    try {
        _matvec_bending_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), decltype(ai), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, ai, d_kernel, sp, osc, isc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _diag_bending_k(AO ao, const reduce_t* kernel,
                            flow_sp<ndim, offset_t> sp, offset_t sc,
                            offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d];
        }
        Impl::template diag_bending<opfunc>(vo.data() + oo, sc, loc, sp.size, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_bending(
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     bending,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t sc = stride[nall];

    reduce_t kbuf[Impl::kernelsize_bending];
    Impl::make_kernel_bending(kbuf, absolute, membrane, bending, voxel_size);

    flow_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride[nbatch + d];
        sp.sinp[d] = stride[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_bending));
    try {
        _diag_bending_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, sp, sc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}


//======================================================================
//                          kernel_* (stencil at the batch centre)
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _kernel_absolute_k(AO ao, const reduce_t* kernel,
                      flow_kp<ndim, offset_t> kp, offset_t ncell)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    const offset_t center = center_offset<ndim>(kp.size, kp.sout);
    for (offset_t b = blockIdx.x * blockDim.x + threadIdx.x;
         b < ncell; b += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto vo = ao.template peel_front_at<-ndim>(b);
        Impl::template kernel_absolute<opfunc>(vo.data() + center, kp.sc, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void kernel_absolute(
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;

    // absolute is pointwise, so there is no "full" (unfolded) kernel variant:
    // make_kernel_absolute IS the stencil the CPU kernel_absolute writes.
    reduce_t kbuf[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kbuf, absolute, voxel_size);

    flow_kp<ndim, offset_t> kp;
    kp.sc = stride[nall];
    // kp.sc2 is unused here (kernel_absolute reads only kp.sc below) -- leave it
    // uninitialised rather than reading stride[nall + 1], which is past the end
    // of this tensor's (*batch,*spatial,C) stride array (nall + 1 entries,
    // indices 0..nall). Only kernel_lame/kernel_all's (*batch,*spatial,C,C)
    // tensors have a real nall+1 entry to read into sc2[1].
    for (int d = 0; d < ndim; ++d) {
        kp.size[d] = size[nbatch + d]; kp.sout[d] = stride[nbatch + d];
    }

    // batch-only index domain: no trailing channel axis (peel_front_at<-ndim>).
    auto ao = _any_nb(out, size, stride, nall);
    const offset_t ncell = ao.template size_front<-ndim>();

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_absolute));
    try {
        _kernel_absolute_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(ncell), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, kp, ncell);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _kernel_membrane_k(AO ao, const reduce_t* kernel,
                      flow_kp<ndim, offset_t> kp, offset_t ncell)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    const offset_t center = center_offset<ndim>(kp.size, kp.sout);
    for (offset_t b = blockIdx.x * blockDim.x + threadIdx.x;
         b < ncell; b += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto vo = ao.template peel_front_at<-ndim>(b);
        Impl::template kernel_membrane<opfunc>(vo.data() + center, kp.sc, kp.sout, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void kernel_membrane(
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;

    reduce_t kbuf[Impl::kernelsize_membrane];
    Impl::make_fullkernel_membrane(kbuf, absolute, membrane, voxel_size);

    flow_kp<ndim, offset_t> kp;
    kp.sc = stride[nall];
    // kp.sc2 unused here -- see kernel_absolute's comment above.
    for (int d = 0; d < ndim; ++d) {
        kp.size[d] = size[nbatch + d]; kp.sout[d] = stride[nbatch + d];
    }

    // batch-only index domain: no trailing channel axis (peel_front_at<-ndim>).
    auto ao = _any_nb(out, size, stride, nall);
    const offset_t ncell = ao.template size_front<-ndim>();

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_membrane));
    try {
        _kernel_membrane_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(ncell), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, kp, ncell);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _kernel_bending_k(AO ao, const reduce_t* kernel,
                      flow_kp<ndim, offset_t> kp, offset_t ncell)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    const offset_t center = center_offset<ndim>(kp.size, kp.sout);
    for (offset_t b = blockIdx.x * blockDim.x + threadIdx.x;
         b < ncell; b += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto vo = ao.template peel_front_at<-ndim>(b);
        Impl::template kernel_bending<opfunc>(vo.data() + center, kp.sc, kp.sout, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void kernel_bending(
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     bending,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;

    reduce_t kbuf[Impl::kernelsize_bending];
    Impl::make_fullkernel_bending(kbuf, absolute, membrane, bending, voxel_size);

    flow_kp<ndim, offset_t> kp;
    kp.sc = stride[nall];
    // kp.sc2 unused here -- see kernel_absolute's comment above.
    for (int d = 0; d < ndim; ++d) {
        kp.size[d] = size[nbatch + d]; kp.sout[d] = stride[nbatch + d];
    }

    // batch-only index domain: no trailing channel axis (peel_front_at<-ndim>).
    auto ao = _any_nb(out, size, stride, nall);
    const offset_t ncell = ao.template size_front<-ndim>();

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_bending));
    try {
        _kernel_bending_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(ncell), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, kp, ncell);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _kernel_lame_k(AO ao, const reduce_t* kernel,
                      flow_kp<ndim, offset_t> kp, offset_t ncell)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    const offset_t center = center_offset<ndim>(kp.size, kp.sout);
    for (offset_t b = blockIdx.x * blockDim.x + threadIdx.x;
         b < ncell; b += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto vo = ao.template peel_front_at<-ndim>(b);
        Impl::template kernel_lame<opfunc>(vo.data() + center, kp.sc2, kp.sout, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void kernel_lame(
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     shears,
          reduce_t     div,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;

    reduce_t kbuf[Impl::kernelsize_lame];
    Impl::make_fullkernel_lame(kbuf, absolute, membrane, shears, div, voxel_size);

    flow_kp<ndim, offset_t> kp;
    kp.sc = stride[nall];
    kp.sc2[0] = stride[nall];
    kp.sc2[1] = stride[nall + 1];
    for (int d = 0; d < ndim; ++d) {
        kp.size[d] = size[nbatch + d]; kp.sout[d] = stride[nbatch + d];
    }

    // batch-only index domain: no trailing channel axis (peel_front_at<-ndim>).
    auto ao = _any_nb(out, size, stride, nall);
    const offset_t ncell = ao.template size_front<-ndim>();

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_lame));
    try {
        _kernel_lame_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(ncell), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, kp, ncell);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _kernel_all_k(AO ao, const reduce_t* kernel,
                      flow_kp<ndim, offset_t> kp, offset_t ncell)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    const offset_t center = center_offset<ndim>(kp.size, kp.sout);
    for (offset_t b = blockIdx.x * blockDim.x + threadIdx.x;
         b < ncell; b += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto vo = ao.template peel_front_at<-ndim>(b);
        Impl::template kernel_all<opfunc>(vo.data() + center, kp.sc2, kp.sout, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void kernel_all(
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     bending,
          reduce_t     shears,
          reduce_t     div,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;

    reduce_t kbuf[Impl::kernelsize_all];
    Impl::make_fullkernel_all(kbuf, absolute, membrane, bending, shears, div, voxel_size);

    flow_kp<ndim, offset_t> kp;
    kp.sc = stride[nall];
    kp.sc2[0] = stride[nall];
    kp.sc2[1] = stride[nall + 1];
    for (int d = 0; d < ndim; ++d) {
        kp.size[d] = size[nbatch + d]; kp.sout[d] = stride[nbatch + d];
    }

    // batch-only index domain: no trailing channel axis (peel_front_at<-ndim>).
    auto ao = _any_nb(out, size, stride, nall);
    const offset_t ncell = ao.template size_front<-ndim>();

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_all));
    try {
        _kernel_all_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(ncell), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, kp, ncell);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

//======================================================================
//                          LAME  (stencil)
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, bound::type... BOUND>
CUGLOB void _matvec_lame_k(AO ao, AI ai, const reduce_t* kernel,
                      flow_sp<ndim, offset_t> sp, offset_t osc, offset_t isc,
                      offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0, io = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d]; io += c * sp.sinp[d];
        }
        Impl::template matvec_lame<opfunc>(
            vo.data() + oo, vi.data() + io, loc, sp.size, sp.sinp, osc, isc, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void matvec_lame(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     shears,
          reduce_t     div,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_out[nall], isc = stride_inp[nall];

    reduce_t kbuf[Impl::kernelsize_lame];
    Impl::make_kernel_lame(kbuf, absolute, membrane, shears, div, voxel_size);

    flow_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_inp[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto ai = _any(inp, size, stride_inp, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_lame));
    try {
        _matvec_lame_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), decltype(ai), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, ai, d_kernel, sp, osc, isc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _diag_lame_k(AO ao, const reduce_t* kernel,
                    flow_sp<ndim, offset_t> sp, offset_t sc,
                    offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d];
        }
        Impl::template diag_lame<opfunc>(vo.data() + oo, sc, loc, sp.size, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_lame(
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     shears,
          reduce_t     div,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t sc = stride[nall];

    reduce_t kbuf[Impl::kernelsize_lame];
    Impl::make_kernel_lame(kbuf, absolute, membrane, shears, div, voxel_size);

    flow_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride[nbatch + d];
        sp.sinp[d] = stride[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_lame));
    try {
        _diag_lame_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, d_kernel, sp, sc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

//======================================================================
//                          ALL  (stencil)
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, bound::type... BOUND>
CUGLOB void _matvec_all_k(AO ao, AI ai, const reduce_t* kernel,
                      flow_sp<ndim, offset_t> sp, offset_t osc, offset_t isc,
                      offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0, io = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d]; io += c * sp.sinp[d];
        }
        Impl::template matvec_all<opfunc>(
            vo.data() + oo, vi.data() + io, loc, sp.size, sp.sinp, osc, isc, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void matvec_all(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     bending,
          reduce_t     shears,
          reduce_t     div,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_out[nall], isc = stride_inp[nall];

    reduce_t kbuf[Impl::kernelsize_all];
    Impl::make_kernel_all(kbuf, absolute, membrane, bending, shears, div, voxel_size);

    flow_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_inp[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto ai = _any(inp, size, stride_inp, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_all));
    try {
        _matvec_all_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), decltype(ai), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, ai, d_kernel, sp, osc, isc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _diag_all_k(AO ao, const reduce_t* kernel,
                    flow_sp<ndim, offset_t> sp, offset_t sc,
                    offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d];
        }
        Impl::template diag_all<opfunc>(vo.data() + oo, sc, loc, sp.size, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_all(
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     bending,
          reduce_t     shears,
          reduce_t     div,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t sc = stride[nall];

    reduce_t kbuf[Impl::kernelsize_all];
    Impl::make_kernel_all(kbuf, absolute, membrane, bending, shears, div, voxel_size);

    flow_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride[nbatch + d];
        sp.sinp[d] = stride[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_all));
    try {
        _diag_all_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, d_kernel, sp, sc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

//======================================================================
//                          COLOURED RELAX
//======================================================================

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          class AS, class AH, class AG, bound::type... BOUND>
CUGLOB void _relax_membrane_k(AS as, AH ah, AG ag, const reduce_t* kernel,
                     flow_rx<ndim, offset_t> rx,
                     offset_t osc, offset_t hsc, offset_t gsc,
                     offset_t nvox, offset_t nsp)
{
    using Impl         = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    static constexpr auto subfunc = Op<'-', scalar_t, reduce_t>::f;   // CPU's `isub`
    static constexpr auto setfunc = Op<'=', scalar_t, reduce_t>::f;   // CPU's `set`
    constexpr int DD = PosDef::work_size;

    // C == ndim here, so every scratch buffer is compile-time sized: no channel
    // cap and no device allocation, unlike reg_field's runtime-C relax.
    scalar_t val[ndim], diag[ndim];
    reduce_t buf[DD ? DD : 1];
    offset_t loc[ndim];

    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        decode_sublattice<ndim>(i - b * nsp, rx, loc);

        scalar_t       * solp = as.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.ssol);
        const scalar_t * hesp = ah.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.shes);
        const scalar_t * grdp = ag.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.sgrd);

#       pragma unroll
        for (int d = 0; d < ndim; ++d) val[d] = grdp[gsc * d];

        Impl::template matvec_membrane<subfunc>(
            val, solp, loc, rx.size, rx.ssol, static_cast<offset_t>(1), osc, kernel);
        Impl::template diag_membrane<setfunc>(
            diag, static_cast<offset_t>(1), loc, rx.size, kernel);

        PosDef::relax_(Strided(solp, osc), StridedConst(hesp, hsc),
                       val, diag, buf, static_cast<reduce_t>(0));
    }
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void relax_membrane_(
          offset_t     nbatch,
          scalar_t   * sol,
    const scalar_t   * hes,
    const scalar_t   * grd,
    const offset_t   * size,
    const offset_t   * stride_sol,
    const offset_t   * stride_hes,
    const offset_t   * stride_grd,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          int          niter = 1,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall], gsc = stride_grd[nall];

    reduce_t kbuf[Impl::kernelsize_membrane];
    Impl::make_kernel_membrane(kbuf, absolute, membrane, voxel_size);

    using Colour = colour_lattice<ndim, 2, true>;   // was patch1
    flow_rx<ndim, offset_t> rx;
    rx.kstep = 2;
    for (int d = 0; d < ndim; ++d) {
        rx.size[d] = size[nbatch + d];
        rx.ssol[d] = stride_sol[nbatch + d];
        rx.shes[d] = stride_hes[nbatch + d];
        rx.sgrd[d] = stride_grd[nbatch + d];
        rx.swgt[d] = 0;
    }

    auto as = _any_rx(sol, size, stride_sol, nall);
    auto ah = _any_rx(hes, size, stride_hes, nall);
    auto ag = _any_rx(grd, size, stride_grd, nall);
    const offset_t ncell = as.template size_front<-ndim>();

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_membrane));
    try {
        // Colour::ncolour == the CPU's own pass count for this scheme
        // (2*niter).
        for (offset_t n = 0; n < static_cast<offset_t>(Colour::ncolour) * niter; ++n)
        for (int j = 0; j < Colour::nsub; ++j)
        {
            Colour::starts(n, j, rx.cstart);
            const offset_t nsp  = sublattice_extents<ndim>(rx.size, rx.kstep, rx.cstart, rx.nsub);
            const offset_t nvox = ncell * nsp;
            if (nvox <= 0) continue;
            _relax_membrane_k<ndim, reduce_t, scalar_t, offset_t,
                     decltype(as), decltype(ah), decltype(ag), BOUND...>
                <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                    as, ah, ag, d_kernel, rx, osc, hsc, gsc, nvox, nsp);
        }
        // one sync after the LAST colour: same-stream launches are already
        // ordered, so the per-colour barrier costs nothing extra.
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          class AS, class AH, class AG, bound::type... BOUND>
CUGLOB void _relax_bending_k(AS as, AH ah, AG ag, const reduce_t* kernel,
                     flow_rx<ndim, offset_t> rx,
                     offset_t osc, offset_t hsc, offset_t gsc,
                     offset_t nvox, offset_t nsp)
{
    using Impl         = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    static constexpr auto subfunc = Op<'-', scalar_t, reduce_t>::f;   // CPU's `isub`
    static constexpr auto setfunc = Op<'=', scalar_t, reduce_t>::f;   // CPU's `set`
    constexpr int DD = PosDef::work_size;

    // C == ndim here, so every scratch buffer is compile-time sized: no channel
    // cap and no device allocation, unlike reg_field's runtime-C relax.
    scalar_t val[ndim], diag[ndim];
    reduce_t buf[DD ? DD : 1];
    offset_t loc[ndim];

    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        decode_sublattice<ndim>(i - b * nsp, rx, loc);

        scalar_t       * solp = as.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.ssol);
        const scalar_t * hesp = ah.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.shes);
        const scalar_t * grdp = ag.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.sgrd);

#       pragma unroll
        for (int d = 0; d < ndim; ++d) val[d] = grdp[gsc * d];

        Impl::template matvec_bending<subfunc>(
            val, solp, loc, rx.size, rx.ssol, static_cast<offset_t>(1), osc, kernel);
        Impl::template diag_bending<setfunc>(
            diag, static_cast<offset_t>(1), loc, rx.size, kernel);

        PosDef::relax_(Strided(solp, osc), StridedConst(hesp, hsc),
                       val, diag, buf, static_cast<reduce_t>(0));
    }
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void relax_bending_(
          offset_t     nbatch,
          scalar_t   * sol,
    const scalar_t   * hes,
    const scalar_t   * grd,
    const offset_t   * size,
    const offset_t   * stride_sol,
    const offset_t   * stride_hes,
    const offset_t   * stride_grd,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     bending,
          int          niter = 1,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall], gsc = stride_grd[nall];

    reduce_t kbuf[Impl::kernelsize_bending];
    Impl::make_kernel_bending(kbuf, absolute, membrane, bending, voxel_size);

    using Colour = colour_lattice<ndim, 3, false>;   // was patch3
    flow_rx<ndim, offset_t> rx;
    rx.kstep = 3;
    for (int d = 0; d < ndim; ++d) {
        rx.size[d] = size[nbatch + d];
        rx.ssol[d] = stride_sol[nbatch + d];
        rx.shes[d] = stride_hes[nbatch + d];
        rx.sgrd[d] = stride_grd[nbatch + d];
        rx.swgt[d] = 0;
    }

    auto as = _any_rx(sol, size, stride_sol, nall);
    auto ah = _any_rx(hes, size, stride_hes, nall);
    auto ag = _any_rx(grd, size, stride_grd, nall);
    const offset_t ncell = as.template size_front<-ndim>();

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_bending));
    try {
        // Colour::ncolour == the CPU's own pass count for this scheme
        // (pow<ndim>(3)*niter).
        for (offset_t n = 0; n < static_cast<offset_t>(Colour::ncolour) * niter; ++n)
        for (int j = 0; j < Colour::nsub; ++j)
        {
            Colour::starts(n, j, rx.cstart);
            const offset_t nsp  = sublattice_extents<ndim>(rx.size, rx.kstep, rx.cstart, rx.nsub);
            const offset_t nvox = ncell * nsp;
            if (nvox <= 0) continue;
            _relax_bending_k<ndim, reduce_t, scalar_t, offset_t,
                     decltype(as), decltype(ah), decltype(ag), BOUND...>
                <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                    as, ah, ag, d_kernel, rx, osc, hsc, gsc, nvox, nsp);
        }
        // one sync after the LAST colour: same-stream launches are already
        // ordered, so the per-colour barrier costs nothing extra.
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          class AS, class AH, class AG, bound::type... BOUND>
CUGLOB void _relax_lame_k(AS as, AH ah, AG ag, const reduce_t* kernel,
                     flow_rx<ndim, offset_t> rx,
                     offset_t osc, offset_t hsc, offset_t gsc,
                     offset_t nvox, offset_t nsp)
{
    using Impl         = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    static constexpr auto subfunc = Op<'-', scalar_t, reduce_t>::f;   // CPU's `isub`
    static constexpr auto setfunc = Op<'=', scalar_t, reduce_t>::f;   // CPU's `set`
    constexpr int DD = PosDef::work_size;

    // C == ndim here, so every scratch buffer is compile-time sized: no channel
    // cap and no device allocation, unlike reg_field's runtime-C relax.
    scalar_t val[ndim], diag[ndim];
    reduce_t buf[DD ? DD : 1];
    offset_t loc[ndim];

    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        decode_sublattice<ndim>(i - b * nsp, rx, loc);

        scalar_t       * solp = as.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.ssol);
        const scalar_t * hesp = ah.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.shes);
        const scalar_t * grdp = ag.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.sgrd);

#       pragma unroll
        for (int d = 0; d < ndim; ++d) val[d] = grdp[gsc * d];

        Impl::template matvec_lame<subfunc>(
            val, solp, loc, rx.size, rx.ssol, static_cast<offset_t>(1), osc, kernel);
        Impl::template diag_lame<setfunc>(
            diag, static_cast<offset_t>(1), loc, rx.size, kernel);

        PosDef::relax_(Strided(solp, osc), StridedConst(hesp, hsc),
                       val, diag, buf, static_cast<reduce_t>(0));
    }
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void relax_lame_(
          offset_t     nbatch,
          scalar_t   * sol,
    const scalar_t   * hes,
    const scalar_t   * grd,
    const offset_t   * size,
    const offset_t   * stride_sol,
    const offset_t   * stride_hes,
    const offset_t   * stride_grd,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     shears,
          reduce_t     div,
          int          niter = 1,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall], gsc = stride_grd[nall];

    reduce_t kbuf[Impl::kernelsize_lame];
    Impl::make_kernel_lame(kbuf, absolute, membrane, shears, div, voxel_size);

    using Colour = colour_lattice<ndim, 2, false>;   // was patch2
    flow_rx<ndim, offset_t> rx;
    rx.kstep = 2;
    for (int d = 0; d < ndim; ++d) {
        rx.size[d] = size[nbatch + d];
        rx.ssol[d] = stride_sol[nbatch + d];
        rx.shes[d] = stride_hes[nbatch + d];
        rx.sgrd[d] = stride_grd[nbatch + d];
        rx.swgt[d] = 0;
    }

    auto as = _any_rx(sol, size, stride_sol, nall);
    auto ah = _any_rx(hes, size, stride_hes, nall);
    auto ag = _any_rx(grd, size, stride_grd, nall);
    const offset_t ncell = as.template size_front<-ndim>();

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_lame));
    try {
        // Colour::ncolour == the CPU's own pass count for this scheme
        // (pow<ndim>(2)*niter).
        for (offset_t n = 0; n < static_cast<offset_t>(Colour::ncolour) * niter; ++n)
        for (int j = 0; j < Colour::nsub; ++j)
        {
            Colour::starts(n, j, rx.cstart);
            const offset_t nsp  = sublattice_extents<ndim>(rx.size, rx.kstep, rx.cstart, rx.nsub);
            const offset_t nvox = ncell * nsp;
            if (nvox <= 0) continue;
            _relax_lame_k<ndim, reduce_t, scalar_t, offset_t,
                     decltype(as), decltype(ah), decltype(ag), BOUND...>
                <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                    as, ah, ag, d_kernel, rx, osc, hsc, gsc, nvox, nsp);
        }
        // one sync after the LAST colour: same-stream launches are already
        // ordered, so the per-colour barrier costs nothing extra.
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          class AS, class AH, class AG, bound::type... BOUND>
CUGLOB void _relax_all_k(AS as, AH ah, AG ag, const reduce_t* kernel,
                     flow_rx<ndim, offset_t> rx,
                     offset_t osc, offset_t hsc, offset_t gsc,
                     offset_t nvox, offset_t nsp)
{
    using Impl         = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    static constexpr auto subfunc = Op<'-', scalar_t, reduce_t>::f;   // CPU's `isub`
    static constexpr auto setfunc = Op<'=', scalar_t, reduce_t>::f;   // CPU's `set`
    constexpr int DD = PosDef::work_size;

    // C == ndim here, so every scratch buffer is compile-time sized: no channel
    // cap and no device allocation, unlike reg_field's runtime-C relax.
    scalar_t val[ndim], diag[ndim];
    reduce_t buf[DD ? DD : 1];
    offset_t loc[ndim];

    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        decode_sublattice<ndim>(i - b * nsp, rx, loc);

        scalar_t       * solp = as.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.ssol);
        const scalar_t * hesp = ah.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.shes);
        const scalar_t * grdp = ag.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.sgrd);

#       pragma unroll
        for (int d = 0; d < ndim; ++d) val[d] = grdp[gsc * d];

        Impl::template matvec_all<subfunc>(
            val, solp, loc, rx.size, rx.ssol, static_cast<offset_t>(1), osc, kernel);
        Impl::template diag_all<setfunc>(
            diag, static_cast<offset_t>(1), loc, rx.size, kernel);

        PosDef::relax_(Strided(solp, osc), StridedConst(hesp, hsc),
                       val, diag, buf, static_cast<reduce_t>(0));
    }
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void relax_all_(
          offset_t     nbatch,
          scalar_t   * sol,
    const scalar_t   * hes,
    const scalar_t   * grd,
    const offset_t   * size,
    const offset_t   * stride_sol,
    const offset_t   * stride_hes,
    const offset_t   * stride_grd,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     bending,
          reduce_t     shears,
          reduce_t     div,
          int          niter = 1,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall], gsc = stride_grd[nall];

    reduce_t kbuf[Impl::kernelsize_all];
    Impl::make_kernel_all(kbuf, absolute, membrane, bending, shears, div, voxel_size);

    using Colour = colour_lattice<ndim, 3, false>;   // was patch3
    flow_rx<ndim, offset_t> rx;
    rx.kstep = 3;
    for (int d = 0; d < ndim; ++d) {
        rx.size[d] = size[nbatch + d];
        rx.ssol[d] = stride_sol[nbatch + d];
        rx.shes[d] = stride_hes[nbatch + d];
        rx.sgrd[d] = stride_grd[nbatch + d];
        rx.swgt[d] = 0;
    }

    auto as = _any_rx(sol, size, stride_sol, nall);
    auto ah = _any_rx(hes, size, stride_hes, nall);
    auto ag = _any_rx(grd, size, stride_grd, nall);
    const offset_t ncell = as.template size_front<-ndim>();

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_all));
    try {
        // Colour::ncolour == the CPU's own pass count for this scheme
        // (pow<ndim>(3)*niter).
        for (offset_t n = 0; n < static_cast<offset_t>(Colour::ncolour) * niter; ++n)
        for (int j = 0; j < Colour::nsub; ++j)
        {
            Colour::starts(n, j, rx.cstart);
            const offset_t nsp  = sublattice_extents<ndim>(rx.size, rx.kstep, rx.cstart, rx.nsub);
            const offset_t nvox = ncell * nsp;
            if (nvox <= 0) continue;
            _relax_all_k<ndim, reduce_t, scalar_t, offset_t,
                     decltype(as), decltype(ah), decltype(ag), BOUND...>
                <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                    as, ah, ag, d_kernel, rx, osc, hsc, gsc, nvox, nsp);
        }
        // one sync after the LAST colour: same-stream launches are already
        // ordered, so the per-colour barrier costs nothing extra.
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

//======================================================================
//                          MEMBRANE JRLS
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, class AW, bound::type... BOUND>
CUGLOB void _matvec_membrane_jrls_k(AO ao, AI ai, AW aw, const reduce_t* kernel,
                           flow_sp<ndim, offset_t> sp, flow_wp<ndim, offset_t> wp,
                           offset_t osc, offset_t isc, offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
        auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0, io = 0, wo = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d]; io += c * sp.sinp[d]; wo += c * wp.swgt[d];
        }
        Impl::template matvec_membrane_jrls<opfunc>(
            vo.data() + oo, vi.data() + io, vw.data() + wo,
            loc, sp.size, sp.sinp, wp.swgt, osc, isc, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void matvec_membrane_jrls(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_out[nall], isc = stride_inp[nall];

    reduce_t kbuf[Impl::kernelsize_membrane_jrls];
    Impl::make_kernel_membrane_jrls(kbuf, absolute, membrane, voxel_size);

    flow_sp<ndim, offset_t> sp;
    flow_wp<ndim, offset_t> wp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_inp[nbatch + d]; wp.swgt[d] = stride_wgt[nbatch + d];
        nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto ai = _any(inp, size, stride_inp, nall);
    auto aw = _any(wgt, size, stride_wgt, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_membrane_jrls));
    try {
        _matvec_membrane_jrls_k<ndim, op, reduce_t, scalar_t, offset_t,
                       decltype(ao), decltype(ai), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, ai, aw, d_kernel, sp, wp, osc, isc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AW, bound::type... BOUND>
CUGLOB void _diag_membrane_jrls_k(AO ao, AW aw, const reduce_t* kernel,
                         flow_sp<ndim, offset_t> sp, flow_wp<ndim, offset_t> wp,
                         offset_t osc, offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0, wo = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d]; wo += c * wp.swgt[d];
        }
        Impl::template diag_membrane_jrls<opfunc>(
            vo.data() + oo, vw.data() + wo, loc, sp.size, wp.swgt, osc, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_membrane_jrls(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_out[nall];

    reduce_t kbuf[Impl::kernelsize_membrane_jrls];
    Impl::make_kernel_membrane_jrls(kbuf, absolute, membrane, voxel_size);

    flow_sp<ndim, offset_t> sp;
    flow_wp<ndim, offset_t> wp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_out[nbatch + d]; wp.swgt[d] = stride_wgt[nbatch + d];
        nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto aw = _any(wgt, size, stride_wgt, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_membrane_jrls));
    try {
        _diag_membrane_jrls_k<ndim, op, reduce_t, scalar_t, offset_t,
                     decltype(ao), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, aw, d_kernel, sp, wp, osc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          class AS, class AH, class AG, class AW, bound::type... BOUND>
CUGLOB void _relax_membrane_jrls_k(AS as, AH ah, AG ag, AW aw, const reduce_t* kernel,
                          flow_rx<ndim, offset_t> rx,
                          offset_t osc, offset_t hsc, offset_t gsc,
                          offset_t nvox, offset_t nsp)
{
    using Impl         = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    static constexpr auto subfunc = Op<'-', scalar_t, reduce_t>::f;
    static constexpr auto setfunc = Op<'=', scalar_t, reduce_t>::f;
    constexpr int DD = PosDef::work_size;

    scalar_t val[ndim], diag[ndim];
    reduce_t buf[DD ? DD : 1];
    offset_t loc[ndim];

    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        decode_sublattice<ndim>(i - b * nsp, rx, loc);

        scalar_t       * solp = as.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.ssol);
        const scalar_t * hesp = ah.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.shes);
        const scalar_t * grdp = ag.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.sgrd);
        const scalar_t * wgtp = aw.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.swgt);

#       pragma unroll
        for (int d = 0; d < ndim; ++d) val[d] = grdp[gsc * d];

        Impl::template matvec_membrane_jrls<subfunc>(
            val, solp, wgtp, loc, rx.size, rx.ssol, rx.swgt,
            static_cast<offset_t>(1), osc, kernel);
        Impl::template diag_membrane_jrls<setfunc>(
            diag, wgtp, loc, rx.size, rx.swgt, static_cast<offset_t>(1), kernel);

        PosDef::relax_(Strided(solp, osc), StridedConst(hesp, hsc),
                       val, diag, buf, static_cast<reduce_t>(0));
    }
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void relax_membrane_jrls_(
          offset_t     nbatch,
          scalar_t   * sol,
    const scalar_t   * hes,
    const scalar_t   * grd,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_sol,
    const offset_t   * stride_hes,
    const offset_t   * stride_grd,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          int          niter = 1,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall], gsc = stride_grd[nall];

    reduce_t kbuf[Impl::kernelsize_membrane_jrls];
    Impl::make_kernel_membrane_jrls(kbuf, absolute, membrane, voxel_size);

    using Colour = colour_lattice<ndim, 2, true>;   // was patch1
    flow_rx<ndim, offset_t> rx;
    rx.kstep = 2;
    for (int d = 0; d < ndim; ++d) {
        rx.size[d] = size[nbatch + d];
        rx.ssol[d] = stride_sol[nbatch + d];
        rx.shes[d] = stride_hes[nbatch + d];
        rx.sgrd[d] = stride_grd[nbatch + d];
        rx.swgt[d] = stride_wgt[nbatch + d];
    }

    auto as = _any_rx(sol, size, stride_sol, nall);
    auto ah = _any_rx(hes, size, stride_hes, nall);
    auto ag = _any_rx(grd, size, stride_grd, nall);
    auto aw = _any_rx(wgt, size, stride_wgt, nall);
    const offset_t ncell = as.template size_front<-ndim>();

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_membrane_jrls));
    try {
        for (offset_t n = 0; n < static_cast<offset_t>(Colour::ncolour) * niter; ++n)
        for (int j = 0; j < Colour::nsub; ++j)
        {
            Colour::starts(n, j, rx.cstart);
            const offset_t nsp  = sublattice_extents<ndim>(rx.size, rx.kstep, rx.cstart, rx.nsub);
            const offset_t nvox = ncell * nsp;
            if (nvox <= 0) continue;
            _relax_membrane_jrls_k<ndim, reduce_t, scalar_t, offset_t,
                          decltype(as), decltype(ah), decltype(ag), decltype(aw), BOUND...>
                <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                    as, ah, ag, aw, d_kernel, rx, osc, hsc, gsc, nvox, nsp);
        }
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

//======================================================================
//                          LAME JRLS
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, class AW, bound::type... BOUND>
CUGLOB void _matvec_lame_jrls_k(AO ao, AI ai, AW aw, const reduce_t* kernel,
                           flow_sp<ndim, offset_t> sp, flow_wp<ndim, offset_t> wp,
                           offset_t osc, offset_t isc, offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
        auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0, io = 0, wo = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d]; io += c * sp.sinp[d]; wo += c * wp.swgt[d];
        }
        Impl::template matvec_lame_jrls<opfunc>(
            vo.data() + oo, vi.data() + io, vw.data() + wo,
            loc, sp.size, sp.sinp, wp.swgt, osc, isc, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void matvec_lame_jrls(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     shears,
          reduce_t     div,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_out[nall], isc = stride_inp[nall];

    reduce_t kbuf[Impl::kernelsize_lame_jrls];
    Impl::make_kernel_lame_jrls(kbuf, absolute, membrane, shears, div, voxel_size);

    flow_sp<ndim, offset_t> sp;
    flow_wp<ndim, offset_t> wp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_inp[nbatch + d]; wp.swgt[d] = stride_wgt[nbatch + d];
        nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto ai = _any(inp, size, stride_inp, nall);
    auto aw = _any(wgt, size, stride_wgt, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_lame_jrls));
    try {
        _matvec_lame_jrls_k<ndim, op, reduce_t, scalar_t, offset_t,
                       decltype(ao), decltype(ai), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, ai, aw, d_kernel, sp, wp, osc, isc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AW, bound::type... BOUND>
CUGLOB void _diag_lame_jrls_k(AO ao, AW aw, const reduce_t* kernel,
                         flow_sp<ndim, offset_t> sp, flow_wp<ndim, offset_t> wp,
                         offset_t osc, offset_t nvox, offset_t nsp)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
        offset_t s = i - b * nsp, loc[ndim], oo = 0, wo = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            const offset_t c = s % sp.size[d]; s /= sp.size[d]; loc[d] = c;
            oo += c * sp.sout[d]; wo += c * wp.swgt[d];
        }
        Impl::template diag_lame_jrls<opfunc>(
            vo.data() + oo, vw.data() + wo, loc, sp.size, wp.swgt, osc, kernel);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_lame_jrls(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     shears,
          reduce_t     div,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_out[nall];

    reduce_t kbuf[Impl::kernelsize_lame_jrls];
    Impl::make_kernel_lame_jrls(kbuf, absolute, membrane, shears, div, voxel_size);

    flow_sp<ndim, offset_t> sp;
    flow_wp<ndim, offset_t> wp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_out[nbatch + d]; wp.swgt[d] = stride_wgt[nbatch + d];
        nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto aw = _any(wgt, size, stride_wgt, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_lame_jrls));
    try {
        _diag_lame_jrls_k<ndim, op, reduce_t, scalar_t, offset_t,
                     decltype(ao), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, aw, d_kernel, sp, wp, osc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          class AS, class AH, class AG, class AW, bound::type... BOUND>
CUGLOB void _relax_lame_jrls_k(AS as, AH ah, AG ag, AW aw, const reduce_t* kernel,
                          flow_rx<ndim, offset_t> rx,
                          offset_t osc, offset_t hsc, offset_t gsc,
                          offset_t nvox, offset_t nsp)
{
    using Impl         = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    static constexpr auto subfunc = Op<'-', scalar_t, reduce_t>::f;
    static constexpr auto setfunc = Op<'=', scalar_t, reduce_t>::f;
    constexpr int DD = PosDef::work_size;

    scalar_t val[ndim], diag[ndim];
    reduce_t buf[DD ? DD : 1];
    offset_t loc[ndim];

    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b = (nsp > 0) ? i / nsp : offset_t(0);
        decode_sublattice<ndim>(i - b * nsp, rx, loc);

        scalar_t       * solp = as.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.ssol);
        const scalar_t * hesp = ah.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.shes);
        const scalar_t * grdp = ag.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.sgrd);
        const scalar_t * wgtp = aw.template peel_front_at<-ndim>(b).data()
                              + spatial_offset<ndim>(loc, rx.swgt);

#       pragma unroll
        for (int d = 0; d < ndim; ++d) val[d] = grdp[gsc * d];

        Impl::template matvec_lame_jrls<subfunc>(
            val, solp, wgtp, loc, rx.size, rx.ssol, rx.swgt,
            static_cast<offset_t>(1), osc, kernel);
        Impl::template diag_lame_jrls<setfunc>(
            diag, wgtp, loc, rx.size, rx.swgt, static_cast<offset_t>(1), kernel);

        PosDef::relax_(Strided(solp, osc), StridedConst(hesp, hsc),
                       val, diag, buf, static_cast<reduce_t>(0));
    }
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void relax_lame_jrls_(
          offset_t     nbatch,
          scalar_t   * sol,
    const scalar_t   * hes,
    const scalar_t   * grd,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_sol,
    const offset_t   * stride_hes,
    const offset_t   * stride_grd,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
          reduce_t     absolute,
          reduce_t     membrane,
          reduce_t     shears,
          reduce_t     div,
          int          niter = 1,
          cudaStream_t stream = 0)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall], gsc = stride_grd[nall];

    reduce_t kbuf[Impl::kernelsize_lame_jrls];
    Impl::make_kernel_lame_jrls(kbuf, absolute, membrane, shears, div, voxel_size);

    using Colour = colour_lattice<ndim, 2, false>;   // was patch2
    flow_rx<ndim, offset_t> rx;
    rx.kstep = 2;
    for (int d = 0; d < ndim; ++d) {
        rx.size[d] = size[nbatch + d];
        rx.ssol[d] = stride_sol[nbatch + d];
        rx.shes[d] = stride_hes[nbatch + d];
        rx.sgrd[d] = stride_grd[nbatch + d];
        rx.swgt[d] = stride_wgt[nbatch + d];
    }

    auto as = _any_rx(sol, size, stride_sol, nall);
    auto ah = _any_rx(hes, size, stride_hes, nall);
    auto ag = _any_rx(grd, size, stride_grd, nall);
    auto aw = _any_rx(wgt, size, stride_wgt, nall);
    const offset_t ncell = as.template size_front<-ndim>();

    reduce_t* d_kernel = copyToDevice(kbuf, static_cast<offset_t>(Impl::kernelsize_lame_jrls));
    try {
        for (offset_t n = 0; n < static_cast<offset_t>(Colour::ncolour) * niter; ++n)
        for (int j = 0; j < Colour::nsub; ++j)
        {
            Colour::starts(n, j, rx.cstart);
            const offset_t nsp  = sublattice_extents<ndim>(rx.size, rx.kstep, rx.cstart, rx.nsub);
            const offset_t nvox = ncell * nsp;
            if (nvox <= 0) continue;
            _relax_lame_jrls_k<ndim, reduce_t, scalar_t, offset_t,
                          decltype(as), decltype(ah), decltype(ag), decltype(aw), BOUND...>
                <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                    as, ah, ag, aw, d_kernel, rx, osc, hsc, gsc, nvox, nsp);
        }
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

FF_NAMESPACE_END(reg_flow)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_REGULARISERS_FLOW_CUDA
