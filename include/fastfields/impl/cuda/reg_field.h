#ifndef FF_REGULARISERS_FIELD_CUDA
#define FF_REGULARISERS_FIELD_CUDA
// Teeny-based CUDA reg_field impl -- the device mirror of the CPU launcher
// (fastfields-cpu-impl/reg_field.h). Same math, same representation:
//
//   * ABSOLUTE is POINTWISE: teeny's peel hands each (*batch,*spatial) voxel's
//     rank-1 channel cell to the shared single-voxel kernel (kernels/
//     regularisers/field). `peel_front_at<-1>` folds the (arbitrarily strided)
//     batch/spatial offset into each cell's pointer -- the same call the CPU body
//     uses. NO atomics (each output voxel is written once -> disjoint writes).
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
//     stride travel INLINE, so the carrier passes into the kernel BY VALUE);
//   * the convolution kernel table is PER-CHANNEL (runtime length
//     get_kernelsize_*(nc)); it is built ON THE HOST (identical to the CPU impl,
//     into a std::vector) and cudaMemcpy'd to the device -- the CPU heap
//     `new reduce_t[...]` becomes a device buffer, freed after the launch;
//   * the spatial size/stride the stencil single-voxel kernels index are copied
//     into a tiny by-value POD (`field_sp`) passed in the launch -- ndim <= 3.
//   * this teeny launcher handles an ARBITRARY batch rank AND channel count (both
//     fold into the peel / the runtime kernel table), unlike the legacy launcher
//     which capped nbatch at 1 and nc at 3.
//
//   * RELAX is a COLOURED Gauss-Seidel sweep. The CPU launcher (phase 4) walks
//     one dense `subsample` sub-lattice per colour; here each colour becomes its
//     OWN kernel LAUNCH on the caller's stream -- see `colour_lattice` below for
//     why a launch (and not an in-kernel colour loop) is the only correct
//     mirror. Each launch grid-strides over that colour's sub-lattice, decodes
//     the voxel's FULL spatial index `loc` from the sub-lattice index, and runs
//     the same minus-convolution / diagonal / `PosDef::relax_` triple the CPU
//     loop body runs.
//   * RLS / JRLS add a per-voxel WEIGHT map. It is an ordinary device tensor, so
//     it rides along as one more anyrank carrier (plus its spatial stride row in
//     the by-value POD); nothing is allocated on the device path.
//
// The op ('=','+','-') is threaded through exactly as the CPU `op_apply` does:
// `Op<op,scalar_t,reduce_t>::f` is the function-pointer non-type template arg the
// single-voxel kernels take (the C++17 device path, same as the legacy launcher).
// The relax bodies want the CPU's bare `isub`/`set` op ids; `Op<'-',...>::f` and
// `Op<'=',...>::f` ARE those two functions (kernels/regularisers/field/utils.h),
// so they are spelled that way here to stay on this file's existing idiom.
#include <teeny/teeny.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "fastfields/core/cuda_switch.h"
#include "fastfields/impl/kernels/bounds.h"
#include "fastfields/impl/kernels/utils.h"
#include "fastfields/impl/kernels/regularisers/field.h"
// kernels/posdef/matrix.h (pulled in by kernels/posdef.h, which the relax_*
// bodies need for the per-voxel linear solve) aliases `namespace cs =
// cuda::std;` at the (ff::cuda) FF_DEVICE namespace scope, where unqualified
// `cuda` binds to the enclosing ff::cuda namespace rather than the global
// ::cuda -- a latent bug that only surfaces under nvcc (it is fine for the
// ff::cpu host build, which never shadows `cuda`). Inject ff::cuda::cuda ->
// ::cuda so `cuda::std` resolves to the global ::cuda::std, WITHOUT touching
// the shared kernels header and WITHOUT shadowing the real ::std. This is the
// same guard posdef.h documents and applies for the same header.
#ifdef __CUDACC__
namespace ff { namespace cuda { namespace cuda = ::cuda; } }
#endif
#include "fastfields/impl/kernels/posdef.h"
#include "utils.h"                    // GET_BLOCKS / CUDA_NUM_THREADS / copyToDevice / freeDevice

// Largest channel count the relax_* device paths support. Unlike matvec/diag --
// which stream one channel vector at a time and need no scratch -- a relax step
// needs three per-VOXEL scratch buffers (the residual `val[C]`, the stencil
// diagonal `diag[C]`, and posdef's CxC Cholesky workspace). The CPU heap-
// allocates those per thread; on the device they are stack arrays sized to this
// compile-time cap, exactly as posdef.h's own device solve/invert paths do. The
// host launcher throws for a larger channel count.
#ifndef FF_REG_FIELD_MAX_C
#define FF_REG_FIELD_MAX_C 16
#endif

// Largest total tensor rank a relax_* device-passable carrier is sized for. A
// carrier holds its shape+stride INLINE (measured: 1040 bytes at TNY_MAX_RANK=64,
// 528 at 32), and a kernel gets at most 4 KiB of by-value parameter space -- the
// RLS/JRLS relax kernels pass FOUR carriers (sol/hes/grd/wgt), so the full
// TNY_MAX_RANK=64 would need 4160 bytes and OVERFLOW that limit. Capping the
// relax carriers at 32 puts four of them at 2112 bytes, comfortably inside it,
// while staying far more generous than the legacy launcher. The host launcher
// throws for a deeper tensor. (matvec/diag pass at most three carriers of the
// uncapped kind and are left exactly as they were.)
#ifndef FF_REG_FIELD_MAX_RANK
#define FF_REG_FIELD_MAX_RANK 32
#endif

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_field)

// Device-passable anyrank carrier over (*batch, *spatial, C). Shape/stride are
// COPIED inline (copy_meta) so the carrier passes into the kernel by value; the
// DATA pointer lives in device memory (storage::gpu_view).
template <typename T, typename offset_t>
static inline auto _any(T* p, const offset_t* size, const offset_t* stride, offset_t nall)
{
    return tny::as_anyrank<TNY_MAX_RANK, tny::storage::gpu_view>(
        p, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);
}

// Rank-CAPPED device-passable carrier, used by the relax_* launchers only. They
// pass up to FOUR carriers (sol/hes/grd/wgt) in one launch, which at the
// uncapped TNY_MAX_RANK would overflow the 4 KiB kernel parameter budget (see
// FF_REG_FIELD_MAX_RANK). `rank` is the number of leading axes to wrap: the
// STENCIL relax bodies want (*batch, *spatial) = nall (they take a base pointer
// plus an explicit channel stride and never index the channel axis), while the
// POINTWISE absolute relax bodies want (*batch, *spatial, C) = nall + 1 so that
// `peel_front_at<-1>` lands directly on each voxel's channel cell.
template <typename T, typename offset_t>
static inline auto _any_rx(T* p, const offset_t* size, const offset_t* stride, offset_t rank)
{
    if (rank > static_cast<offset_t>(FF_REG_FIELD_MAX_RANK))
        throw std::logic_error("reg_field: tensor rank too large for the CUDA relax launcher");
    return tny::as_anyrank<FF_REG_FIELD_MAX_RANK, tny::storage::gpu_view>(
        p, size, stride, static_cast<int>(rank), tny::copy_meta);
}

// Tiny by-value spatial metadata (spatial extents + spatial out/inp strides) the
// stencil single-voxel kernels index. ndim <= 3, so passing it by value into the
// kernel is trivial (no device copy of the shape/stride arrays).
template <int ndim, typename offset_t>
struct field_sp {
    offset_t size[ndim];   // spatial extents  (size[nbatch + d])
    offset_t sout[ndim];   // spatial strides of out
    offset_t sinp[ndim];   // spatial strides of inp
};

// Spatial stride row of the RLS/JRLS weight map. Kept separate from field_sp so
// the existing non-weighted kernels keep their exact parameter list.
template <int ndim, typename offset_t>
struct field_wp {
    offset_t swgt[ndim];   // spatial strides of wgt (stride_wgt[nbatch + d])
};

//----------------------------------------------------------------------
//  Coloured relaxation: the colour -> dense sub-lattice decomposition
//----------------------------------------------------------------------
// Transcribed from the CPU launcher (fastfields-cpu-impl/reg_field.h, phase 4)
// so the two produce the SAME colour classes in the SAME order. A coloured
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
    // colours in the scheme (the modulus the predicates applied to the pass
    // counter), and how many Cartesian sub-lattices one colour is made of
    static constexpr int ncolour = checker ? 2 : colour_ipow(k, ndim);
    static constexpr int nsub    = checker ? (1 << (ndim - 1)) : 1;

    // Fill `start[0..ndim)` for sub-lattice `j` of the colour that pass `n`
    // selects. `n` is the raw pass counter, taken exactly as the predicates
    // took it (they reduced it mod `ncolour` themselves). HOST-only: it runs in
    // the launcher, between launches.
    template <typename offset_t>
    CUHOST static void starts(offset_t n, int j, offset_t start[ndim])
    {
        if (checker)
        {
            // the bits of `j` give the leading ndim-1 starts; the last start is
            // whatever makes the parity of the whole vector match the colour
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
// their product. This is the CPU's `sublattice_numel` with the per-axis extents
// kept as well: the CPU hands the sub-lattice to teeny's `subsample` and lets
// its peel walk it, whereas a grid-stride loop needs random access by flat
// index, so the device kernel decodes the sub-lattice index itself and needs the
// extents to decode against. Same formula, one place -- an axis whose start runs
// past its extent contributes 0 and empties the whole sub-lattice.
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

// By-value spatial metadata for ONE coloured relax launch: the full spatial
// extents (what the stencil kernels index), the sub-lattice this launch sweeps,
// and the spatial stride row of each participating tensor.
template <int ndim, typename offset_t>
struct field_rx {
    offset_t size  [ndim];   // FULL spatial extents (the kernels' `size + nbatch`)
    offset_t nsub  [ndim];   // sub-lattice extents  (this colour)
    offset_t cstart[ndim];   // sub-lattice origin   (this colour)
    offset_t kstep;          // sub-lattice step
    offset_t ssol  [ndim];   // spatial strides of sol
    offset_t shes  [ndim];   // spatial strides of hes
    offset_t sgrd  [ndim];   // spatial strides of grd
    offset_t swgt  [ndim];   // spatial strides of wgt (RLS/JRLS only; else unused)
};

// Decode a flat sub-lattice index into the voxel's FULL spatial index `loc`, and
// accumulate the matching offset into each tensor's spatial stride row. This is
// the device twin of the CPU's `loc[d] = cstart[d] + kstep * v.index[d]`, where
// `v.index` is the multi-index of teeny's peel over the `subsample` view. Peeled
// axes vary row-major (last axis fastest), so the decode runs last-axis-first to
// match. NB the visiting ORDER within a colour is immaterial to the result --
// every voxel of one colour is independent by construction, which is the whole
// point of the colouring -- so this only has to enumerate the same SET.
template <int ndim, typename offset_t>
CUDEV inline void decode_sublattice(offset_t s, const field_rx<ndim, offset_t> & rx,
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
                               offset_t osc, offset_t isc, offset_t nc, offset_t nvox)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        auto ic = ai.template peel_front_at<-1>(i);
        Impl::template matvec_absolute<opfunc>(oc.data(), ic.data(), osc, isc, kernel, nc);
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
    const reduce_t   * absolute,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_out[nall], isc = stride_inp[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_absolute(nc)));
    Impl::make_kernel_absolute(kbuf.data(), absolute, nc);

    auto ao = _any(out, size, stride_out, nall);
    auto ai = _any(inp, size, stride_inp, nall);
    const offset_t nvox = ao.template size_front<-1>();

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _matvec_absolute_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), decltype(ai), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, ai, d_kernel, osc, isc, nc, nvox);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _diag_absolute_k(AO ao, const reduce_t* kernel, offset_t sc, offset_t nc, offset_t nvox)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        Impl::template diag_absolute<opfunc>(oc.data(), sc, kernel, nc);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_absolute(
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride,
    const reduce_t   * absolute,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t sc = stride[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_absolute(nc)));
    Impl::make_kernel_absolute(kbuf.data(), absolute, nc);

    auto ao = _any(out, size, stride, nall);
    const offset_t nvox = ao.template size_front<-1>();

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _diag_absolute_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, sc, nc, nvox);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

// --- ABSOLUTE: kernel ------------------------------------------------
// Writes the stencil at the CENTRE spatial voxel of each batch cell, so the
// grid-stride loop runs over BATCH CELLS, not voxels. `center_offset` is CUDEV
// (device-only under nvcc), so unlike the CPU launcher -- which computes the
// centre once on the host -- it is evaluated inside the kernel from the by-value
// spatial metadata. Cheap: ndim <= 3 adds, once per batch cell.

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _kernel_absolute_k(AO ao, const reduce_t* kernel,
                               field_sp<ndim, offset_t> sp, offset_t sc,
                               offset_t nc, offset_t ncell)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    const offset_t center = center_offset<ndim>(sp.size, sp.sout);
    for (offset_t b = blockIdx.x * blockDim.x + threadIdx.x;
         b < ncell; b += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto vol = ao.template peel_front_at<-(ndim + 1)>(b);
        Impl::template kernel_absolute<opfunc>(vol.data() + center, sc, kernel, nc);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void kernel_absolute(
          offset_t     nbatch,
          scalar_t   * out,
    const offset_t   * size,
    const offset_t   * stride,
    const reduce_t   * absolute,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t sc = stride[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_absolute(nc)));
    Impl::make_kernel_absolute(kbuf.data(), absolute, nc);

    field_sp<ndim, offset_t> sp;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride[nbatch + d];
        sp.sinp[d] = stride[nbatch + d];
    }

    auto ao = _any(out, size, stride, nall);
    const offset_t ncell = ao.template size_front<-(ndim + 1)>();

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _kernel_absolute_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(ncell), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, sp, sc, nc, ncell);
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
                               field_sp<ndim, offset_t> sp, offset_t osc, offset_t isc,
                               offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
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
            vo.data() + oo, vi.data() + io, loc, sp.size, sp.sinp, osc, isc, kernel, nc);
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_out[nall], isc = stride_inp[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_membrane(nc)));
    Impl::make_kernel_membrane(kbuf.data(), absolute, membrane, voxel_size, nc);

    field_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_inp[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto ai = _any(inp, size, stride_inp, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _matvec_membrane_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), decltype(ai), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, ai, d_kernel, sp, osc, isc, nc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _diag_membrane_k(AO ao, const reduce_t* kernel,
                             field_sp<ndim, offset_t> sp, offset_t sc,
                             offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
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
        Impl::template diag_membrane<opfunc>(vo.data() + oo, sc, loc, sp.size, kernel, nc);
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t sc = stride[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_membrane(nc)));
    Impl::make_kernel_membrane(kbuf.data(), absolute, membrane, voxel_size, nc);

    field_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride[nbatch + d];
        sp.sinp[d] = stride[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _diag_membrane_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, sp, sc, nc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

// --- MEMBRANE: kernel ------------------------------------------------

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _kernel_membrane_k(AO ao, const reduce_t* kernel,
                               field_sp<ndim, offset_t> sp, offset_t sc,
                               offset_t nc, offset_t ncell)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    const offset_t center = center_offset<ndim>(sp.size, sp.sout);
    for (offset_t b = blockIdx.x * blockDim.x + threadIdx.x;
         b < ncell; b += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto vol = ao.template peel_front_at<-(ndim + 1)>(b);
        Impl::template kernel_membrane<opfunc>(vol.data() + center, sc, sp.sout, kernel, nc);
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t sc = stride[nall];

    // NB the FULL kernel (make_fullkernel_*), not the folded one matvec/diag use.
    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_membrane(nc)));
    Impl::make_fullkernel_membrane(kbuf.data(), absolute, membrane, voxel_size, nc);

    field_sp<ndim, offset_t> sp;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride[nbatch + d];
        sp.sinp[d] = stride[nbatch + d];
    }

    auto ao = _any(out, size, stride, nall);
    const offset_t ncell = ao.template size_front<-(ndim + 1)>();

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _kernel_membrane_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(ncell), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, sp, sc, nc, ncell);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

// --- MEMBRANE: relax -------------------------------------------------
// One launch per colour (see colour_lattice). Each thread owns one voxel of the
// colour's sub-lattice and runs the CPU loop body verbatim: seed the residual
// from the gradient, subtract the stencil convolution, take the stencil
// diagonal, then the per-voxel `PosDef::relax_` linear solve.

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          class AS, class AH, class AG, bound::type... BOUND>
CUGLOB void _relax_membrane_k(AS as, AH ah, AG ag, const reduce_t* kernel,
                              field_rx<ndim, offset_t> rx,
                              offset_t osc, offset_t hsc, offset_t gsc,
                              offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl         = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    static constexpr auto subfunc = Op<'-', scalar_t, reduce_t>::f;   // CPU's `isub`
    static constexpr auto setfunc = Op<'=', scalar_t, reduce_t>::f;   // CPU's `set`

    scalar_t val [FF_REG_FIELD_MAX_C];
    scalar_t diag[FF_REG_FIELD_MAX_C];
    reduce_t buf [FF_REG_FIELD_MAX_C * FF_REG_FIELD_MAX_C];
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

        for (offset_t c = 0; c < nc; ++c) val[c] = grdp[gsc * c];

        Impl::template matvec_membrane<subfunc>(
            val, solp, loc, rx.size, rx.ssol, static_cast<offset_t>(1), osc, kernel, nc);
        Impl::template diag_membrane<setfunc>(
            diag, static_cast<offset_t>(1), loc, rx.size, kernel, nc);

        PosDef::relax_(nc, Strided(solp, osc), StridedConst(hesp, hsc),
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          int          niter = 1,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall], gsc = stride_grd[nall];
    if (nc > static_cast<offset_t>(FF_REG_FIELD_MAX_C))
        throw std::logic_error("reg_field: too many channels for the CUDA relax launcher");

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_membrane(nc)));
    Impl::make_kernel_membrane(kbuf.data(), absolute, membrane, voxel_size, nc);

    using Colour = colour_lattice<ndim, 2, true>;   // was patch1
    field_rx<ndim, offset_t> rx;
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

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
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
                    as, ah, ag, d_kernel, rx, osc, hsc, gsc, nc, nvox, nsp);
        }
        // one sync after the LAST colour: same-stream launches are already
        // ordered, so the per-colour barrier costs nothing extra.
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
                              field_sp<ndim, offset_t> sp, offset_t osc, offset_t isc,
                              offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
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
            vo.data() + oo, vi.data() + io, loc, sp.size, sp.sinp, osc, isc, kernel, nc);
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_out[nall], isc = stride_inp[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_bending(nc)));
    Impl::make_kernel_bending(kbuf.data(), absolute, membrane, bending, voxel_size, nc);

    field_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_inp[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto ai = _any(inp, size, stride_inp, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _matvec_bending_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), decltype(ai), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, ai, d_kernel, sp, osc, isc, nc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _diag_bending_k(AO ao, const reduce_t* kernel,
                            field_sp<ndim, offset_t> sp, offset_t sc,
                            offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
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
        Impl::template diag_bending<opfunc>(vo.data() + oo, sc, loc, sp.size, kernel, nc);
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t sc = stride[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_bending(nc)));
    Impl::make_kernel_bending(kbuf.data(), absolute, membrane, bending, voxel_size, nc);

    field_sp<ndim, offset_t> sp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride[nbatch + d];
        sp.sinp[d] = stride[nbatch + d]; nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _diag_bending_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, sp, sc, nc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

// --- BENDING: kernel -------------------------------------------------

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, bound::type... BOUND>
CUGLOB void _kernel_bending_k(AO ao, const reduce_t* kernel,
                              field_sp<ndim, offset_t> sp, offset_t sc,
                              offset_t nc, offset_t ncell)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    const offset_t center = center_offset<ndim>(sp.size, sp.sout);
    for (offset_t b = blockIdx.x * blockDim.x + threadIdx.x;
         b < ncell; b += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto vol = ao.template peel_front_at<-(ndim + 1)>(b);
        Impl::template kernel_bending<opfunc>(vol.data() + center, sc, sp.sout, kernel, nc);
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t sc = stride[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_bending(nc)));
    Impl::make_fullkernel_bending(kbuf.data(), absolute, membrane, bending, voxel_size, nc);

    field_sp<ndim, offset_t> sp;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride[nbatch + d];
        sp.sinp[d] = stride[nbatch + d];
    }

    auto ao = _any(out, size, stride, nall);
    const offset_t ncell = ao.template size_front<-(ndim + 1)>();

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _kernel_bending_k<ndim, op, reduce_t, scalar_t, offset_t, decltype(ao), BOUND...>
            <<<GET_BLOCKS(ncell), CUDA_NUM_THREADS, 0, stream>>>(ao, d_kernel, sp, sc, nc, ncell);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

// --- BENDING: relax --------------------------------------------------

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          class AS, class AH, class AG, bound::type... BOUND>
CUGLOB void _relax_bending_k(AS as, AH ah, AG ag, const reduce_t* kernel,
                             field_rx<ndim, offset_t> rx,
                             offset_t osc, offset_t hsc, offset_t gsc,
                             offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl         = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    static constexpr auto subfunc = Op<'-', scalar_t, reduce_t>::f;
    static constexpr auto setfunc = Op<'=', scalar_t, reduce_t>::f;

    scalar_t val [FF_REG_FIELD_MAX_C];
    scalar_t diag[FF_REG_FIELD_MAX_C];
    reduce_t buf [FF_REG_FIELD_MAX_C * FF_REG_FIELD_MAX_C];
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

        for (offset_t c = 0; c < nc; ++c) val[c] = grdp[gsc * c];

        Impl::template matvec_bending<subfunc>(
            val, solp, loc, rx.size, rx.ssol, static_cast<offset_t>(1), osc, kernel, nc);
        Impl::template diag_bending<setfunc>(
            diag, static_cast<offset_t>(1), loc, rx.size, kernel, nc);

        PosDef::relax_(nc, Strided(solp, osc), StridedConst(hesp, hsc),
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          int          niter = 1,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall], gsc = stride_grd[nall];
    if (nc > static_cast<offset_t>(FF_REG_FIELD_MAX_C))
        throw std::logic_error("reg_field: too many channels for the CUDA relax launcher");

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_bending(nc)));
    Impl::make_kernel_bending(kbuf.data(), absolute, membrane, bending, voxel_size, nc);

    using Colour = colour_lattice<ndim, 3, false>;   // was patch3
    field_rx<ndim, offset_t> rx;
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

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        // Colour::ncolour == 3^ndim == the CPU's pow<ndim>(3) pass count.
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
                    as, ah, ag, d_kernel, rx, osc, hsc, gsc, nc, nvox, nsp);
        }
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}


//======================================================================
//                          ABSOLUTE RLS
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, class AW, bound::type... BOUND>
CUGLOB void _matvec_absolute_rls_k(AO ao, AI ai, AW aw, const reduce_t* kernel,
                               offset_t osc, offset_t isc, offset_t wsc,
                               offset_t nc, offset_t nvox)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    (void)wsc;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        auto ic = ai.template peel_front_at<-1>(i);
        auto wc = aw.template peel_front_at<-1>(i);
        Impl::template matvec_absolute_rls<opfunc>(
            oc.data(), ic.data(), wc.data(), osc, isc, wsc, kernel, nc);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void matvec_absolute_rls(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const offset_t   * stride_wgt,
    const reduce_t   * absolute,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_out[nall], isc = stride_inp[nall], wsc = stride_wgt[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_absolute(nc)));
    Impl::make_kernel_absolute(kbuf.data(), absolute, nc);

    auto ao = _any(out, size, stride_out, nall);
    auto ai = _any(inp, size, stride_inp, nall);
    auto aw = _any(wgt, size, stride_wgt, nall);
    const offset_t nvox = ao.template size_front<-1>();

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _matvec_absolute_rls_k<ndim, op, reduce_t, scalar_t, offset_t,
                            decltype(ao), decltype(ai), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, ai, aw, d_kernel, osc, isc, wsc, nc, nvox);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AW, bound::type... BOUND>
CUGLOB void _diag_absolute_rls_k(AO ao, AW aw, const reduce_t* kernel,
                             offset_t osc, offset_t wsc, offset_t nc, offset_t nvox)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    (void)wsc;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        auto wc = aw.template peel_front_at<-1>(i);
        Impl::template diag_absolute_rls<opfunc>(
            oc.data(), wc.data(), osc, wsc, kernel, nc);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_absolute_rls(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_wgt,
    const reduce_t   * absolute,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_out[nall], wsc = stride_wgt[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_absolute(nc)));
    Impl::make_kernel_absolute(kbuf.data(), absolute, nc);

    auto ao = _any(out, size, stride_out, nall);
    auto aw = _any(wgt, size, stride_wgt, nall);
    const offset_t nvox = ao.template size_front<-1>();

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _diag_absolute_rls_k<ndim, op, reduce_t, scalar_t, offset_t,
                          decltype(ao), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, aw, d_kernel, osc, wsc, nc, nvox);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          class AS, class AH, class AG, class AW, bound::type... BOUND>
CUGLOB void _relax_absolute_rls_k(AS as, AH ah, AG ag, AW aw, const reduce_t* kernel,
                              offset_t osc, offset_t hsc, offset_t gsc, offset_t wsc,
                              offset_t nc, offset_t nvox)
{
    using Impl         = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    static constexpr auto subfunc = Op<'-', scalar_t, reduce_t>::f;
    static constexpr auto setfunc = Op<'=', scalar_t, reduce_t>::f;
    (void)wsc;

    scalar_t val [FF_REG_FIELD_MAX_C];
    scalar_t diag[FF_REG_FIELD_MAX_C];
    reduce_t buf [FF_REG_FIELD_MAX_C * FF_REG_FIELD_MAX_C];

    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        scalar_t       * solp = as.template peel_front_at<-1>(i).data();
        const scalar_t * hesp = ah.template peel_front_at<-1>(i).data();
        const scalar_t * grdp = ag.template peel_front_at<-1>(i).data();
        const scalar_t * wgtp = aw.template peel_front_at<-1>(i).data();

        for (offset_t c = 0; c < nc; ++c) val[c] = grdp[gsc * c];

        Impl::template matvec_absolute_rls<subfunc>(
            val, solp, wgtp, static_cast<offset_t>(1), osc, wsc, kernel, nc);
        Impl::template diag_absolute_rls<setfunc>(
            diag, wgtp, static_cast<offset_t>(1), wsc, kernel, nc);

        PosDef::relax_(nc, Strided(solp, osc), StridedConst(hesp, hsc),
                       val, diag, buf, static_cast<reduce_t>(0));
    }
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void relax_absolute_rls_(
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
    const reduce_t   * absolute,
          int          niter = 1,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall];
    const offset_t gsc = stride_grd[nall], wsc = stride_wgt[nall];
    if (nc > static_cast<offset_t>(FF_REG_FIELD_MAX_C))
        throw std::logic_error("reg_field: too many channels for the CUDA relax launcher");

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_absolute(nc)));
    Impl::make_kernel_absolute(kbuf.data(), absolute, nc);

    auto as = _any_rx(sol, size, stride_sol, nall + 1);
    auto ah = _any_rx(hes, size, stride_hes, nall + 1);
    auto ag = _any_rx(grd, size, stride_grd, nall + 1);
    auto aw = _any_rx(wgt, size, stride_wgt, nall + 1);
    const offset_t nvox = as.template size_front<-1>();

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        for (offset_t n = 0; n < static_cast<offset_t>(niter); ++n)
            _relax_absolute_rls_k<ndim, reduce_t, scalar_t, offset_t,
                              decltype(as), decltype(ah), decltype(ag), decltype(aw), BOUND...>
                <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                    as, ah, ag, aw, d_kernel, osc, hsc, gsc, wsc, nc, nvox);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

//======================================================================
//                          ABSOLUTE JRLS
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, class AW, bound::type... BOUND>
CUGLOB void _matvec_absolute_jrls_k(AO ao, AI ai, AW aw, const reduce_t* kernel,
                               offset_t osc, offset_t isc, offset_t wsc,
                               offset_t nc, offset_t nvox)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    (void)wsc;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        auto ic = ai.template peel_front_at<-1>(i);
        auto wc = aw.template peel_front_at<-1>(i);
        Impl::template matvec_absolute_jrls<opfunc>(
            oc.data(), ic.data(), wc.data(), osc, isc, kernel, nc);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void matvec_absolute_jrls(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const offset_t   * stride_wgt,
    const reduce_t   * absolute,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_out[nall], isc = stride_inp[nall], wsc = stride_wgt[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_absolute(nc)));
    Impl::make_kernel_absolute(kbuf.data(), absolute, nc);

    auto ao = _any(out, size, stride_out, nall);
    auto ai = _any(inp, size, stride_inp, nall);
    auto aw = _any(wgt, size, stride_wgt, nall);
    const offset_t nvox = ao.template size_front<-1>();

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _matvec_absolute_jrls_k<ndim, op, reduce_t, scalar_t, offset_t,
                            decltype(ao), decltype(ai), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, ai, aw, d_kernel, osc, isc, wsc, nc, nvox);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AW, bound::type... BOUND>
CUGLOB void _diag_absolute_jrls_k(AO ao, AW aw, const reduce_t* kernel,
                             offset_t osc, offset_t wsc, offset_t nc, offset_t nvox)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    (void)wsc;
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        auto wc = aw.template peel_front_at<-1>(i);
        Impl::template diag_absolute_jrls<opfunc>(
            oc.data(), wc.data(), osc, kernel, nc);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_absolute_jrls(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_wgt,
    const reduce_t   * absolute,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_out[nall], wsc = stride_wgt[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_absolute(nc)));
    Impl::make_kernel_absolute(kbuf.data(), absolute, nc);

    auto ao = _any(out, size, stride_out, nall);
    auto aw = _any(wgt, size, stride_wgt, nall);
    const offset_t nvox = ao.template size_front<-1>();

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _diag_absolute_jrls_k<ndim, op, reduce_t, scalar_t, offset_t,
                          decltype(ao), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, aw, d_kernel, osc, wsc, nc, nvox);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          class AS, class AH, class AG, class AW, bound::type... BOUND>
CUGLOB void _relax_absolute_jrls_k(AS as, AH ah, AG ag, AW aw, const reduce_t* kernel,
                              offset_t osc, offset_t hsc, offset_t gsc, offset_t wsc,
                              offset_t nc, offset_t nvox)
{
    using Impl         = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    static constexpr auto subfunc = Op<'-', scalar_t, reduce_t>::f;
    static constexpr auto setfunc = Op<'=', scalar_t, reduce_t>::f;
    (void)wsc;

    scalar_t val [FF_REG_FIELD_MAX_C];
    scalar_t diag[FF_REG_FIELD_MAX_C];
    reduce_t buf [FF_REG_FIELD_MAX_C * FF_REG_FIELD_MAX_C];

    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox; i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        scalar_t       * solp = as.template peel_front_at<-1>(i).data();
        const scalar_t * hesp = ah.template peel_front_at<-1>(i).data();
        const scalar_t * grdp = ag.template peel_front_at<-1>(i).data();
        const scalar_t * wgtp = aw.template peel_front_at<-1>(i).data();

        for (offset_t c = 0; c < nc; ++c) val[c] = grdp[gsc * c];

        Impl::template matvec_absolute_jrls<subfunc>(
            val, solp, wgtp, static_cast<offset_t>(1), osc, kernel, nc);
        Impl::template diag_absolute_jrls<setfunc>(
            diag, wgtp, static_cast<offset_t>(1), kernel, nc);

        PosDef::relax_(nc, Strided(solp, osc), StridedConst(hesp, hsc),
                       val, diag, buf, static_cast<reduce_t>(0));
    }
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void relax_absolute_jrls_(
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
    const reduce_t   * absolute,
          int          niter = 1,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall];
    const offset_t gsc = stride_grd[nall], wsc = stride_wgt[nall];
    if (nc > static_cast<offset_t>(FF_REG_FIELD_MAX_C))
        throw std::logic_error("reg_field: too many channels for the CUDA relax launcher");

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_absolute(nc)));
    Impl::make_kernel_absolute(kbuf.data(), absolute, nc);

    auto as = _any_rx(sol, size, stride_sol, nall + 1);
    auto ah = _any_rx(hes, size, stride_hes, nall + 1);
    auto ag = _any_rx(grd, size, stride_grd, nall + 1);
    auto aw = _any_rx(wgt, size, stride_wgt, nall + 1);
    const offset_t nvox = as.template size_front<-1>();

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        for (offset_t n = 0; n < static_cast<offset_t>(niter); ++n)
            _relax_absolute_jrls_k<ndim, reduce_t, scalar_t, offset_t,
                              decltype(as), decltype(ah), decltype(ag), decltype(aw), BOUND...>
                <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                    as, ah, ag, aw, d_kernel, osc, hsc, gsc, wsc, nc, nvox);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

//======================================================================
//                          MEMBRANE RLS
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, class AW, bound::type... BOUND>
CUGLOB void _matvec_membrane_rls_k(AO ao, AI ai, AW aw, const reduce_t* kernel,
                          field_sp<ndim, offset_t> sp, field_wp<ndim, offset_t> wp,
                          offset_t osc, offset_t isc, offset_t wsc,
                          offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    (void)wsc;
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
        Impl::template matvec_membrane_rls<opfunc>(
            vo.data() + oo, vi.data() + io, vw.data() + wo,
            loc, sp.size, sp.sinp, wp.swgt, osc, isc, wsc, kernel, nc);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void matvec_membrane_rls(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_out[nall], isc = stride_inp[nall], wsc = stride_wgt[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_membrane_rls(nc)));
    Impl::make_kernel_membrane_rls(kbuf.data(), absolute, membrane, voxel_size, nc);

    field_sp<ndim, offset_t> sp;
    field_wp<ndim, offset_t> wp;
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

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _matvec_membrane_rls_k<ndim, op, reduce_t, scalar_t, offset_t,
                      decltype(ao), decltype(ai), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, ai, aw, d_kernel, sp, wp, osc, isc, wsc, nc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AW, bound::type... BOUND>
CUGLOB void _diag_membrane_rls_k(AO ao, AW aw, const reduce_t* kernel,
                        field_sp<ndim, offset_t> sp, field_wp<ndim, offset_t> wp,
                        offset_t osc, offset_t wsc,
                        offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    (void)wsc;
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
        Impl::template diag_membrane_rls<opfunc>(
            vo.data() + oo, vw.data() + wo,
            loc, sp.size, wp.swgt, osc, wsc, kernel, nc);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_membrane_rls(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_out[nall], wsc = stride_wgt[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_membrane_rls(nc)));
    Impl::make_kernel_membrane_rls(kbuf.data(), absolute, membrane, voxel_size, nc);

    field_sp<ndim, offset_t> sp;
    field_wp<ndim, offset_t> wp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_out[nbatch + d]; wp.swgt[d] = stride_wgt[nbatch + d];
        nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto aw = _any(wgt, size, stride_wgt, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _diag_membrane_rls_k<ndim, op, reduce_t, scalar_t, offset_t,
                    decltype(ao), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, aw, d_kernel, sp, wp, osc, wsc, nc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          class AS, class AH, class AG, class AW, bound::type... BOUND>
CUGLOB void _relax_membrane_rls_k(AS as, AH ah, AG ag, AW aw, const reduce_t* kernel,
                         field_rx<ndim, offset_t> rx,
                         offset_t osc, offset_t hsc, offset_t gsc, offset_t wsc,
                         offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl         = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    static constexpr auto subfunc = Op<'-', scalar_t, reduce_t>::f;
    static constexpr auto setfunc = Op<'=', scalar_t, reduce_t>::f;
    (void)wsc;

    scalar_t val [FF_REG_FIELD_MAX_C];
    scalar_t diag[FF_REG_FIELD_MAX_C];
    reduce_t buf [FF_REG_FIELD_MAX_C * FF_REG_FIELD_MAX_C];
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

        for (offset_t c = 0; c < nc; ++c) val[c] = grdp[gsc * c];

        Impl::template matvec_membrane_rls<subfunc>(
            val, solp, wgtp, loc, rx.size, rx.ssol, rx.swgt,
            static_cast<offset_t>(1), osc, wsc, kernel, nc);
        Impl::template diag_membrane_rls<setfunc>(
            diag, wgtp, loc, rx.size, rx.swgt,
            static_cast<offset_t>(1), wsc, kernel, nc);

        PosDef::relax_(nc, Strided(solp, osc), StridedConst(hesp, hsc),
                       val, diag, buf, static_cast<reduce_t>(0));
    }
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void relax_membrane_rls_(
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          int          niter = 1,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall];
    const offset_t gsc = stride_grd[nall], wsc = stride_wgt[nall];
    if (nc > static_cast<offset_t>(FF_REG_FIELD_MAX_C))
        throw std::logic_error("reg_field: too many channels for the CUDA relax launcher");

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_membrane_rls(nc)));
    Impl::make_kernel_membrane_rls(kbuf.data(), absolute, membrane, voxel_size, nc);

    using Colour = colour_lattice<ndim, 2, true>;   // was patch1
    field_rx<ndim, offset_t> rx;
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

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        for (offset_t n = 0; n < static_cast<offset_t>(Colour::ncolour) * niter; ++n)
        for (int j = 0; j < Colour::nsub; ++j)
        {
            Colour::starts(n, j, rx.cstart);
            const offset_t nsp  = sublattice_extents<ndim>(rx.size, rx.kstep, rx.cstart, rx.nsub);
            const offset_t nvox = ncell * nsp;
            if (nvox <= 0) continue;
            _relax_membrane_rls_k<ndim, reduce_t, scalar_t, offset_t,
                         decltype(as), decltype(ah), decltype(ag), decltype(aw), BOUND...>
                <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                    as, ah, ag, aw, d_kernel, rx, osc, hsc, gsc, wsc, nc, nvox, nsp);
        }
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
                          field_sp<ndim, offset_t> sp, field_wp<ndim, offset_t> wp,
                          offset_t osc, offset_t isc, offset_t wsc,
                          offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    (void)wsc;
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
            loc, sp.size, sp.sinp, wp.swgt, osc, isc, kernel, nc);
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_out[nall], isc = stride_inp[nall], wsc = stride_wgt[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_membrane_rls(nc)));
    Impl::make_kernel_membrane_rls(kbuf.data(), absolute, membrane, voxel_size, nc);

    field_sp<ndim, offset_t> sp;
    field_wp<ndim, offset_t> wp;
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

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _matvec_membrane_jrls_k<ndim, op, reduce_t, scalar_t, offset_t,
                      decltype(ao), decltype(ai), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, ai, aw, d_kernel, sp, wp, osc, isc, wsc, nc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AW, bound::type... BOUND>
CUGLOB void _diag_membrane_jrls_k(AO ao, AW aw, const reduce_t* kernel,
                        field_sp<ndim, offset_t> sp, field_wp<ndim, offset_t> wp,
                        offset_t osc, offset_t wsc,
                        offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    (void)wsc;
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
            vo.data() + oo, vw.data() + wo,
            loc, sp.size, wp.swgt, osc, kernel, nc);
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_out[nall], wsc = stride_wgt[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_membrane_rls(nc)));
    Impl::make_kernel_membrane_rls(kbuf.data(), absolute, membrane, voxel_size, nc);

    field_sp<ndim, offset_t> sp;
    field_wp<ndim, offset_t> wp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_out[nbatch + d]; wp.swgt[d] = stride_wgt[nbatch + d];
        nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto aw = _any(wgt, size, stride_wgt, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _diag_membrane_jrls_k<ndim, op, reduce_t, scalar_t, offset_t,
                    decltype(ao), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, aw, d_kernel, sp, wp, osc, wsc, nc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          class AS, class AH, class AG, class AW, bound::type... BOUND>
CUGLOB void _relax_membrane_jrls_k(AS as, AH ah, AG ag, AW aw, const reduce_t* kernel,
                         field_rx<ndim, offset_t> rx,
                         offset_t osc, offset_t hsc, offset_t gsc, offset_t wsc,
                         offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl         = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    static constexpr auto subfunc = Op<'-', scalar_t, reduce_t>::f;
    static constexpr auto setfunc = Op<'=', scalar_t, reduce_t>::f;
    (void)wsc;

    scalar_t val [FF_REG_FIELD_MAX_C];
    scalar_t diag[FF_REG_FIELD_MAX_C];
    reduce_t buf [FF_REG_FIELD_MAX_C * FF_REG_FIELD_MAX_C];
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

        for (offset_t c = 0; c < nc; ++c) val[c] = grdp[gsc * c];

        Impl::template matvec_membrane_jrls<subfunc>(
            val, solp, wgtp, loc, rx.size, rx.ssol, rx.swgt,
            static_cast<offset_t>(1), osc, kernel, nc);
        Impl::template diag_membrane_jrls<setfunc>(
            diag, wgtp, loc, rx.size, rx.swgt,
            static_cast<offset_t>(1), kernel, nc);

        PosDef::relax_(nc, Strided(solp, osc), StridedConst(hesp, hsc),
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
          int          niter = 1,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall];
    const offset_t gsc = stride_grd[nall], wsc = stride_wgt[nall];
    if (nc > static_cast<offset_t>(FF_REG_FIELD_MAX_C))
        throw std::logic_error("reg_field: too many channels for the CUDA relax launcher");

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_membrane_rls(nc)));
    Impl::make_kernel_membrane_rls(kbuf.data(), absolute, membrane, voxel_size, nc);

    using Colour = colour_lattice<ndim, 2, true>;   // was patch1
    field_rx<ndim, offset_t> rx;
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

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
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
                    as, ah, ag, aw, d_kernel, rx, osc, hsc, gsc, wsc, nc, nvox, nsp);
        }
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

//======================================================================
//                          BENDING RLS
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, class AW, bound::type... BOUND>
CUGLOB void _matvec_bending_rls_k(AO ao, AI ai, AW aw, const reduce_t* kernel,
                          field_sp<ndim, offset_t> sp, field_wp<ndim, offset_t> wp,
                          offset_t osc, offset_t isc, offset_t wsc,
                          offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    (void)wsc;
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
        Impl::template matvec_bending_rls<opfunc>(
            vo.data() + oo, vi.data() + io, vw.data() + wo,
            loc, sp.size, sp.sinp, wp.swgt, osc, isc, wsc, kernel, nc);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void matvec_bending_rls(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_out[nall], isc = stride_inp[nall], wsc = stride_wgt[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_bending_rls(nc)));
    Impl::make_kernel_bending_rls(kbuf.data(), absolute, membrane, bending, voxel_size, nc);

    field_sp<ndim, offset_t> sp;
    field_wp<ndim, offset_t> wp;
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

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _matvec_bending_rls_k<ndim, op, reduce_t, scalar_t, offset_t,
                      decltype(ao), decltype(ai), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, ai, aw, d_kernel, sp, wp, osc, isc, wsc, nc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AW, bound::type... BOUND>
CUGLOB void _diag_bending_rls_k(AO ao, AW aw, const reduce_t* kernel,
                        field_sp<ndim, offset_t> sp, field_wp<ndim, offset_t> wp,
                        offset_t osc, offset_t wsc,
                        offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    (void)wsc;
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
        Impl::template diag_bending_rls<opfunc>(
            vo.data() + oo, vw.data() + wo,
            loc, sp.size, wp.swgt, osc, wsc, kernel, nc);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_bending_rls(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_out[nall], wsc = stride_wgt[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_bending_rls(nc)));
    Impl::make_kernel_bending_rls(kbuf.data(), absolute, membrane, bending, voxel_size, nc);

    field_sp<ndim, offset_t> sp;
    field_wp<ndim, offset_t> wp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_out[nbatch + d]; wp.swgt[d] = stride_wgt[nbatch + d];
        nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto aw = _any(wgt, size, stride_wgt, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _diag_bending_rls_k<ndim, op, reduce_t, scalar_t, offset_t,
                    decltype(ao), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, aw, d_kernel, sp, wp, osc, wsc, nc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          class AS, class AH, class AG, class AW, bound::type... BOUND>
CUGLOB void _relax_bending_rls_k(AS as, AH ah, AG ag, AW aw, const reduce_t* kernel,
                         field_rx<ndim, offset_t> rx,
                         offset_t osc, offset_t hsc, offset_t gsc, offset_t wsc,
                         offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl         = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    static constexpr auto subfunc = Op<'-', scalar_t, reduce_t>::f;
    static constexpr auto setfunc = Op<'=', scalar_t, reduce_t>::f;
    (void)wsc;

    scalar_t val [FF_REG_FIELD_MAX_C];
    scalar_t diag[FF_REG_FIELD_MAX_C];
    reduce_t buf [FF_REG_FIELD_MAX_C * FF_REG_FIELD_MAX_C];
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

        for (offset_t c = 0; c < nc; ++c) val[c] = grdp[gsc * c];

        Impl::template matvec_bending_rls<subfunc>(
            val, solp, wgtp, loc, rx.size, rx.ssol, rx.swgt,
            static_cast<offset_t>(1), osc, wsc, kernel, nc);
        Impl::template diag_bending_rls<setfunc>(
            diag, wgtp, loc, rx.size, rx.swgt,
            static_cast<offset_t>(1), wsc, kernel, nc);

        PosDef::relax_(nc, Strided(solp, osc), StridedConst(hesp, hsc),
                       val, diag, buf, static_cast<reduce_t>(0));
    }
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void relax_bending_rls_(
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          int          niter = 1,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall];
    const offset_t gsc = stride_grd[nall], wsc = stride_wgt[nall];
    if (nc > static_cast<offset_t>(FF_REG_FIELD_MAX_C))
        throw std::logic_error("reg_field: too many channels for the CUDA relax launcher");

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_bending_rls(nc)));
    Impl::make_kernel_bending_rls(kbuf.data(), absolute, membrane, bending, voxel_size, nc);

    using Colour = colour_lattice<ndim, 3, false>;   // was patch3
    field_rx<ndim, offset_t> rx;
    rx.kstep = 3;
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

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        // NB `2*niter`, NOT Colour::ncolour*niter -- mirrors the CPU launcher's
        // known bug exactly (fastfields-cpu-impl#51: this leaves most of the
        // volume unrelaxed at ndim=3). Not fixed here on purpose: a fix needs
        // to land on cpu-impl and cuda-impl together, per #51.
        for (offset_t n = 0; n < 2 * niter; ++n)
        for (int j = 0; j < Colour::nsub; ++j)
        {
            Colour::starts(n, j, rx.cstart);
            const offset_t nsp  = sublattice_extents<ndim>(rx.size, rx.kstep, rx.cstart, rx.nsub);
            const offset_t nvox = ncell * nsp;
            if (nvox <= 0) continue;
            _relax_bending_rls_k<ndim, reduce_t, scalar_t, offset_t,
                         decltype(as), decltype(ah), decltype(ag), decltype(aw), BOUND...>
                <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                    as, ah, ag, aw, d_kernel, rx, osc, hsc, gsc, wsc, nc, nvox, nsp);
        }
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

//======================================================================
//                          BENDING JRLS
//======================================================================

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AI, class AW, bound::type... BOUND>
CUGLOB void _matvec_bending_jrls_k(AO ao, AI ai, AW aw, const reduce_t* kernel,
                          field_sp<ndim, offset_t> sp, field_wp<ndim, offset_t> wp,
                          offset_t osc, offset_t isc, offset_t wsc,
                          offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    (void)wsc;
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
        Impl::template matvec_bending_jrls<opfunc>(
            vo.data() + oo, vi.data() + io, vw.data() + wo,
            loc, sp.size, sp.sinp, wp.swgt, osc, isc, kernel, nc);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void matvec_bending_jrls(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * inp,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_inp,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_out[nall], isc = stride_inp[nall], wsc = stride_wgt[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_bending_rls(nc)));
    Impl::make_kernel_bending_rls(kbuf.data(), absolute, membrane, bending, voxel_size, nc);

    field_sp<ndim, offset_t> sp;
    field_wp<ndim, offset_t> wp;
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

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _matvec_bending_jrls_k<ndim, op, reduce_t, scalar_t, offset_t,
                      decltype(ao), decltype(ai), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, ai, aw, d_kernel, sp, wp, osc, isc, wsc, nc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          class AO, class AW, bound::type... BOUND>
CUGLOB void _diag_bending_jrls_k(AO ao, AW aw, const reduce_t* kernel,
                        field_sp<ndim, offset_t> sp, field_wp<ndim, offset_t> wp,
                        offset_t osc, offset_t wsc,
                        offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    static constexpr auto opfunc = Op<op, scalar_t, reduce_t>::f;
    (void)wsc;
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
        Impl::template diag_bending_jrls<opfunc>(
            vo.data() + oo, vw.data() + wo,
            loc, sp.size, wp.swgt, osc, kernel, nc);
    }
}

template <int ndim, char op, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void diag_bending_jrls(
          offset_t     nbatch,
          scalar_t   * out,
    const scalar_t   * wgt,
    const offset_t   * size,
    const offset_t   * stride_out,
    const offset_t   * stride_wgt,
    const reduce_t   * voxel_size,
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_out[nall], wsc = stride_wgt[nall];

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_bending_rls(nc)));
    Impl::make_kernel_bending_rls(kbuf.data(), absolute, membrane, bending, voxel_size, nc);

    field_sp<ndim, offset_t> sp;
    field_wp<ndim, offset_t> wp;
    offset_t nsp = 1;
    for (int d = 0; d < ndim; ++d) {
        sp.size[d] = size[nbatch + d]; sp.sout[d] = stride_out[nbatch + d];
        sp.sinp[d] = stride_out[nbatch + d]; wp.swgt[d] = stride_wgt[nbatch + d];
        nsp *= sp.size[d];
    }

    auto ao = _any(out, size, stride_out, nall);
    auto aw = _any(wgt, size, stride_wgt, nall);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        _diag_bending_jrls_k<ndim, op, reduce_t, scalar_t, offset_t,
                    decltype(ao), decltype(aw), BOUND...>
            <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                ao, aw, d_kernel, sp, wp, osc, wsc, nc, nvox, nsp);
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          class AS, class AH, class AG, class AW, bound::type... BOUND>
CUGLOB void _relax_bending_jrls_k(AS as, AH ah, AG ag, AW aw, const reduce_t* kernel,
                         field_rx<ndim, offset_t> rx,
                         offset_t osc, offset_t hsc, offset_t gsc, offset_t wsc,
                         offset_t nc, offset_t nvox, offset_t nsp)
{
    using Impl         = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef       = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided      = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;
    static constexpr auto subfunc = Op<'-', scalar_t, reduce_t>::f;
    static constexpr auto setfunc = Op<'=', scalar_t, reduce_t>::f;
    (void)wsc;

    scalar_t val [FF_REG_FIELD_MAX_C];
    scalar_t diag[FF_REG_FIELD_MAX_C];
    reduce_t buf [FF_REG_FIELD_MAX_C * FF_REG_FIELD_MAX_C];
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

        for (offset_t c = 0; c < nc; ++c) val[c] = grdp[gsc * c];

        Impl::template matvec_bending_jrls<subfunc>(
            val, solp, wgtp, loc, rx.size, rx.ssol, rx.swgt,
            static_cast<offset_t>(1), osc, kernel, nc);
        Impl::template diag_bending_jrls<setfunc>(
            diag, wgtp, loc, rx.size, rx.swgt,
            static_cast<offset_t>(1), kernel, nc);

        PosDef::relax_(nc, Strided(solp, osc), StridedConst(hesp, hsc),
                       val, diag, buf, static_cast<reduce_t>(0));
    }
}

template <int ndim, typename reduce_t, typename scalar_t, typename offset_t,
          bound::type... BOUND>
void relax_bending_jrls_(
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
    const reduce_t   * absolute,
    const reduce_t   * membrane,
    const reduce_t   * bending,
          int          niter = 1,
          cudaStream_t stream = 0)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    const offset_t nall = nbatch + ndim;
    const offset_t nc = size[nall];
    const offset_t osc = stride_sol[nall], hsc = stride_hes[nall];
    const offset_t gsc = stride_grd[nall], wsc = stride_wgt[nall];
    if (nc > static_cast<offset_t>(FF_REG_FIELD_MAX_C))
        throw std::logic_error("reg_field: too many channels for the CUDA relax launcher");

    std::vector<reduce_t> kbuf(static_cast<size_t>(Impl::get_kernelsize_bending_rls(nc)));
    Impl::make_kernel_bending_rls(kbuf.data(), absolute, membrane, bending, voxel_size, nc);

    using Colour = colour_lattice<ndim, 3, false>;   // was patch3
    field_rx<ndim, offset_t> rx;
    rx.kstep = 3;
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

    reduce_t* d_kernel = copyToDevice(kbuf.data(), static_cast<offset_t>(kbuf.size()));
    try {
        // NB `2*niter`, NOT Colour::ncolour*niter -- mirrors the CPU launcher's
        // known bug exactly (fastfields-cpu-impl#51: this leaves most of the
        // volume unrelaxed at ndim=3). Not fixed here on purpose: a fix needs
        // to land on cpu-impl and cuda-impl together, per #51.
        for (offset_t n = 0; n < 2 * niter; ++n)
        for (int j = 0; j < Colour::nsub; ++j)
        {
            Colour::starts(n, j, rx.cstart);
            const offset_t nsp  = sublattice_extents<ndim>(rx.size, rx.kstep, rx.cstart, rx.nsub);
            const offset_t nvox = ncell * nsp;
            if (nvox <= 0) continue;
            _relax_bending_jrls_k<ndim, reduce_t, scalar_t, offset_t,
                         decltype(as), decltype(ah), decltype(ag), decltype(aw), BOUND...>
                <<<GET_BLOCKS(nvox), CUDA_NUM_THREADS, 0, stream>>>(
                    as, ah, ag, aw, d_kernel, rx, osc, hsc, gsc, wsc, nc, nvox, nsp);
        }
        cudaStreamSynchronize(stream);
    } catch (...) { freeDevice(d_kernel); throw; }
    freeDevice(d_kernel);
}

FF_NAMESPACE_END(reg_field)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_REGULARISERS_FIELD_CUDA
