#ifndef FF_REGULARISERS_FIELD_CPU
#define FF_REGULARISERS_FIELD_CPU
#include <stdexcept>
#include <utility>
#include <teeny/teeny.h>
#include "kernels/cuda_switch.h"
#include "kernels/bounds.h"
#include "kernels/utils.h"
#include "kernels/batch.h"
#include "kernels/parallel.h"
#include "kernels/regularisers/field.h"
#include "kernels/posdef.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_field)

//----------------------------------------------------------------------
//  op dispatch helper (issue #6a)
//----------------------------------------------------------------------
// Thread the compile-time op ('=', '+', '-') through a function-template
// wrapper. Mirrors the CUDA impl's `Op<op,scalar_t,reduce_t>::f` dispatch, but
// as a named function template because C++11 (unlike the CUDA/C++17 path) only
// accepts a *function id-expression* -- not a constexpr pointer variable -- as
// a non-type template argument of function-pointer type.
template <char op, typename scalar_t, typename reduce_t>
inline scalar_t & op_apply(scalar_t & out, const reduce_t & in)
{
    return Op<op, scalar_t, reduce_t>::f(out, in);
}

//----------------------------------------------------------------------
//  Coloured relaxation: the colour -> dense sub-lattice decomposition
//----------------------------------------------------------------------
// A coloured Gauss-Seidel sweep relaxes one colour class of voxels per pass.
// The three colourings in use here are the ones the kernel layer spells as the
// `patch1`/`patch2`/`patch3` predicates, and each colour class is a union of
// *Cartesian* sub-lattices `loc[d] % k == start[d]` -- that is, of teeny
// `subsample<0..ndim-1>(k, start...)` views. Sweeping those views directly
// visits exactly the colour's voxels, where the predicate form visited every
// voxel on every pass and discarded the ones that did not match.
//
//   AXIS-INDEPENDENT (`patch2`, k=2; `patch3`, k=3) -- k^ndim colours, and a
//   colour IS a single sub-lattice: `start[d]` = digit `d` of the colour in
//   base k. One `subsample` call, no union.
//
//   CHECKERBOARD (`patch1`, k=2) -- 2 colours, `(sum_d loc[d]) % 2 == colour`.
//   That is diagonal, not Cartesian, so it is not any single `subsample` call.
//   It is the UNION of the 2^(ndim-1) sub-lattices whose start vector has
//   parity `colour` (at most 4, since ndim <= 3). Keeping the 2 checkerboard
//   classes -- rather than relaxing on the finer 2^ndim axis-independent
//   colouring, which would also be a valid Gauss-Seidel separation -- is
//   deliberate: it preserves the exact per-pass iterate sequence the predicate
//   form produced for niter > 1.

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
    // took it (they reduced it mod `ncolour` themselves).
    template <typename offset_t>
    static void starts(offset_t n, int j, offset_t start[ndim])
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

// Number of voxels in the spatial sub-lattice `loc[d] % k == start[d]` -- i.e.
// the element count of the `subsample` view below, computed without building
// it (an axis whose start runs past its extent contributes 0, and the whole
// sub-lattice is then empty).
template <int ndim, typename offset_t>
inline offset_t sublattice_numel(const offset_t * size, offset_t k,
                                 const offset_t * start)
{
    offset_t n = 1;
    for (int d = 0; d < ndim; ++d)
    {
        const offset_t e = size[d] - start[d];
        n *= (e > 0) ? (e + k - 1) / k : static_cast<offset_t>(0);
    }
    return n;
}

// Bind all ndim spatial axes of `vol` to that sub-lattice, as one teeny
// `subsample` view (the axis pack is what needs the index_sequence).
template <typename Tensor, typename offset_t, std::size_t... D>
inline auto sublattice_(Tensor vol, offset_t k, const offset_t * start,
                        std::index_sequence<D...>)
{
    return vol.template subsample<static_cast<long>(D)...>(k, start[D]...);
}

template <int ndim, typename Tensor, typename offset_t>
inline auto sublattice(Tensor vol, offset_t k, const offset_t * start)
{
    return sublattice_(vol, k, start, std::make_index_sequence<ndim>{});
}

//======================================================================
//                              ABSOLUTE
//======================================================================

// --- ABSOLUTE: matvec -----------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_absolute(
          offset_t   nbatch,
          scalar_t * out,            // (*batch, *spatial, channels) tensor
    const scalar_t * inp,            // (*batch, *spatial, channels) tensor
    const offset_t * size,           // [*batch, *spatial, channels] vector
    const offset_t * stride_out,     // [*batch, *spatial, channels] vector
    const offset_t * stride_inp,     // [*batch, *spatial, channels] vector
    const reduce_t * absolute
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    const offset_t nall = nbatch + ndim;
    const offset_t nc   = size[nall];

    // teeny peel replaces the index2offset plumbing: wrap (*batch, *spatial, C)
    // and peel the last axis -> each cell is the channel vector at one voxel;
    // cell.data()/stride(0)/extent(0) give the base ptr, channel stride, C.
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);

    const offset_t nvox = ao.template size_front<-1>();   // batch x spatial voxels
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        auto oc = ao.template peel_front_at<-1>(i);       // (C,)
        auto ic = ai.template peel_front_at<-1>(i);       // (C,)
        Impl::template matvec_absolute<op_apply<op, scalar_t, reduce_t> >(
            oc.data(), ic.data(), oc.stride(0), ic.stride(0), kernel, nc);
    }});
    delete[] kernel;
}

// --- ABSOLUTE: kernel ------------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void kernel_absolute(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride,        // [*batch, *spatial, channels] vector
    const reduce_t * absolute
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    const offset_t nall = nbatch + ndim;
    const offset_t sc   = stride[nall];
    const offset_t nc   = size[nall];

    // kernel writes the stencil at the CENTRE spatial voxel of each batch: peel
    // the batch (keep the *spatial+C volume), offset to the spatial centre.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);
    const offset_t center = center_offset<ndim>(size + nbatch, stride + nbatch);

    const offset_t ncell = ao.template size_front<-(ndim + 1)>();   // batch cells
    parallel_for(0, ncell, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t b=start; b < end; ++b)
    {
        auto vol = ao.template peel_front_at<-(ndim + 1)>(b);       // (*spatial, C) volume
        Impl::template kernel_absolute<op_apply<op, scalar_t, reduce_t> >(
            vol.data() + center, sc, kernel, nc);
    }});
    delete[] kernel;
}

// --- ABSOLUTE: diagonal ----------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_absolute(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride,        // [*batch, *spatial, channels] vector
    const reduce_t * absolute
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    const offset_t nall = nbatch + ndim;
    const offset_t nc   = size[nall];

    // absolute has no spatial dependence -> peel the channel axis per voxel.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);

    const offset_t nvox = ao.template size_front<-1>();   // batch x spatial voxels
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        auto oc = ao.template peel_front_at<-1>(i);       // (C,)
        Impl::template diag_absolute<op_apply<op, scalar_t, reduce_t> >(
            oc.data(), oc.stride(0), kernel, nc);
    }});
    delete[] kernel;
}

//======================================================================
//                              MEMBRANE
//======================================================================

// --- MEMBRANE: matvec -----------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_membrane(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * inp,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_inp,    // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t isc  = stride_inp[nall];
    const offset_t nc   = size[nall];

    // Stencil op: the kernel gathers spatial NEIGHBOURS with boundary conditions,
    // so it needs the voxel's spatial multi-index `loc`. Peel the batch (keep the
    // *spatial+C volume, cached per cell), then sweep that volume's spatial axes
    // with teeny's own peel: `enumerate()` hands out the multi-index `loc` next to
    // the already-offset cell, so neither is computed by hand (teeny #213).
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane(nc)];
    Impl::make_kernel_membrane(kernel, absolute, membrane, voxel_size, nc);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    // Each parallel chunk is a flat [start,end) slice of the (batch x spatial)
    // voxel range; walk it one batch run at a time -- peel that batch's volume
    // once (the cursor the chunk used to carry), then let teeny's spatial peel
    // sweep the [lo,hi) sub-range inside it.
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
    {
        const offset_t b = i / nsp, lo = i - b * nsp;
        const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
        i += hi - lo;
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);   // (*spatial, C) this batch
        auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
        auto ci = tny::peel_front<-1>(vi).subrange(lo, hi).begin();
        for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
        {
            Impl::template matvec_membrane<op_apply<op, scalar_t, reduce_t> >(
                v.cell.data(), (*ci).data(),
                v.index.data(), size + nbatch, stride_inp + nbatch, osc, isc, kernel, nc);
            ++ci;
        }
    }});
    delete[] kernel;
}

// --- MEMBRANE: kernel ------------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void kernel_membrane(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride,        // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t sc   = stride[nall];
    const offset_t nc   = size[nall];

    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane(nc)];
    Impl::make_fullkernel_membrane(kernel, absolute, membrane, voxel_size, nc);
    const offset_t center = center_offset<ndim>(size + nbatch, stride + nbatch);

    const offset_t ncell = ao.template size_front<-(ndim + 1)>();   // batch cells
    parallel_for(0, ncell, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t b=start; b < end; ++b)
    {
        auto vol = ao.template peel_front_at<-(ndim + 1)>(b);       // (*spatial, C) volume
        Impl::template kernel_membrane<op_apply<op, scalar_t, reduce_t> >(
            vol.data() + center, sc, stride + nbatch, kernel, nc);
    }});
    delete[] kernel;
}

// --- MEMBRANE: diagonal ----------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_membrane(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride,        // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t sc   = stride[nall];
    const offset_t nc   = size[nall];

    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane(nc)];
    Impl::make_kernel_membrane(kernel, absolute, membrane, voxel_size, nc);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
    {
        const offset_t b = i / nsp, lo = i - b * nsp;
        const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
        i += hi - lo;
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
            Impl::template diag_membrane<op_apply<op, scalar_t, reduce_t> >(
                v.cell.data(), sc, v.index.data(), size + nbatch, kernel, nc);
    }});
    delete[] kernel;
}

// --- MEMBRANE: relax -------------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_membrane_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
          int        niter=1
)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall  = nbatch + ndim;
    offset_t osc   = stride_sol[nall];
    offset_t hsc   = stride_hes[nall];
    offset_t gsc   = stride_grd[nall];
    offset_t nc    = size[nall];

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane(nc)];
    Impl::make_kernel_membrane(kernel, absolute, membrane, voxel_size, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size(nc);

    // Coloured sweep over dense sub-lattice views (see colour_lattice
    // above): one `subsample` per pass, so a pass touches only its own
    // colour, instead of visiting every voxel and discarding the rest. The
    // wrapped index domain is (*batch, *spatial) -- the channel axis is left
    // off, since every kernel below takes a base pointer plus its own channel
    // stride and never indexes that axis here.
    using Colour = colour_lattice<ndim, 2, true>;   // was patch1
    const offset_t kstep = 2;
    auto asol = tny::as_anyrank(sol, size, stride_sol, static_cast<int>(nall), tny::copy_meta);
    auto agrd = tny::as_anyrank(grd, size, stride_grd, static_cast<int>(nall), tny::copy_meta);
    auto ahes = tny::as_anyrank(hes, size, stride_hes, static_cast<int>(nall), tny::copy_meta);
    const offset_t ncell = asol.template size_front<-ndim>();

    for (offset_t n=0; n<2*niter; ++n) {
    for (int j=0; j<Colour::nsub; ++j) {
        offset_t cstart[ndim];
        Colour::starts(n, j, cstart);
        const offset_t nsp  = sublattice_numel<ndim>(size + nbatch, kstep, cstart);
        const offset_t nvox = ncell * nsp;
        if (nvox <= 0) continue;
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t * val  = new reduce_t[nc];
        scalar_t * diag = new reduce_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vsol = sublattice<ndim>(asol.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vgrd = sublattice<ndim>(agrd.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vhes = sublattice<ndim>(ahes.template peel_front_at<-ndim>(b), kstep, cstart);
            auto cgrd = tny::peel_front<ndim>(vgrd).subrange(lo, hi).begin();
            auto ches = tny::peel_front<ndim>(vhes).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<ndim>(vsol).enumerate().subrange(lo, hi))
            {
                for (int d=0; d<ndim; ++d)
                    loc[d] = cstart[d] + kstep * v.index[d];
                scalar_t       * solp = v.cell.data();
                const scalar_t * grdp = (*cgrd).data();
                const scalar_t * hesp = (*ches).data();

                // gradient
                for (offset_t c=0; c<nc; ++c)
                    val[c] = grdp[gsc*c];

                // minus convolution
                Impl::template matvec_membrane<isub>(
                    val, solp,
                    loc, size + nbatch, stride_sol + nbatch,
                    static_cast<offset_t>(1), osc, kernel, nc);

                // diagonal
                Impl::template diag_membrane<set>(
                    diag, static_cast<offset_t>(1), loc, size + nbatch, kernel, nc);

                // sol += (hes + diag) \ (grad - conv(sol))
                PosDef::relax_(
                    nc,
                    Strided(solp, osc),
                    StridedConst(hesp, hsc),
                    val, diag, buf, static_cast<reduce_t>(0)
                );
                ++cgrd; ++ches;
            }
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    }
    delete[] kernel;
}

//======================================================================
//                              BENDING
//======================================================================

// --- BENDING: matvec ------------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_bending(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, 0) tensor
    const scalar_t * inp,           // (*batch, *spatial, 0) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_out,    // [*batch, *spatial, 0] vector
    const offset_t * stride_inp,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t isc  = stride_inp[nall];
    const offset_t nc   = size[nall];

    // biharmonic stencil -> needs the voxel's spatial multi-index (see matvec_membrane).
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending(nc)];
    Impl::make_kernel_bending(kernel, absolute, membrane, bending, voxel_size, nc);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
    {
        const offset_t b = i / nsp, lo = i - b * nsp;
        const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
        i += hi - lo;
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
        auto ci = tny::peel_front<-1>(vi).subrange(lo, hi).begin();
        for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
        {
            Impl::template matvec_bending<op_apply<op, scalar_t, reduce_t> >(
                v.cell.data(), (*ci).data(),
                v.index.data(), size + nbatch, stride_inp + nbatch, osc, isc, kernel, nc);
            ++ci;
        }
    }});
    delete[] kernel;
}

// --- BENDING: kernel -------------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void kernel_bending(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride,        // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t sc   = stride[nall];
    const offset_t nc   = size[nall];

    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending(nc)];
    Impl::make_fullkernel_bending(kernel, absolute, membrane, bending, voxel_size, nc);
    const offset_t center = center_offset<ndim>(size + nbatch, stride + nbatch);

    const offset_t ncell = ao.template size_front<-(ndim + 1)>();   // batch cells
    parallel_for(0, ncell, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t b=start; b < end; ++b)
    {
        auto vol = ao.template peel_front_at<-(ndim + 1)>(b);       // (*spatial, C) volume
        Impl::template kernel_bending<op_apply<op, scalar_t, reduce_t> >(
            vol.data() + center, sc, stride + nbatch, kernel, nc);
    }});
    delete[] kernel;
}

// --- BENDING: diagonal -----------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_bending(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride,        // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t sc   = stride[nall];
    const offset_t nc   = size[nall];

    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending(nc)];
    Impl::make_kernel_bending(kernel, absolute, membrane, bending, voxel_size, nc);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
    {
        const offset_t b = i / nsp, lo = i - b * nsp;
        const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
        i += hi - lo;
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
            Impl::template diag_bending<op_apply<op, scalar_t, reduce_t> >(
                v.cell.data(), sc, v.index.data(), size + nbatch, kernel, nc);
    }});
    delete[] kernel;
}

// --- BENDING: relax --------------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_bending_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending,
          int        niter=1)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t, 0>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall  = nbatch + ndim;
    offset_t osc   = stride_sol[nall];
    offset_t hsc   = stride_hes[nall];
    offset_t gsc   = stride_grd[nall];
    offset_t nc    = size[nall];

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending(nc)];
    Impl::make_kernel_bending(kernel, absolute, membrane, bending, voxel_size, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size(nc);

    // Coloured sweep over dense sub-lattice views (see colour_lattice
    // above): one `subsample` per pass, so a pass touches only its own
    // colour, instead of visiting every voxel and discarding the rest. The
    // wrapped index domain is (*batch, *spatial) -- the channel axis is left
    // off, since every kernel below takes a base pointer plus its own channel
    // stride and never indexes that axis here.
    using Colour = colour_lattice<ndim, 3, false>;   // was patch3
    const offset_t kstep = 3;
    auto asol = tny::as_anyrank(sol, size, stride_sol, static_cast<int>(nall), tny::copy_meta);
    auto agrd = tny::as_anyrank(grd, size, stride_grd, static_cast<int>(nall), tny::copy_meta);
    auto ahes = tny::as_anyrank(hes, size, stride_hes, static_cast<int>(nall), tny::copy_meta);
    const offset_t ncell = asol.template size_front<-ndim>();

    for (offset_t n = 0; n < pow<ndim>(3)*niter; ++n) {
    for (int j=0; j<Colour::nsub; ++j) {
        offset_t cstart[ndim];
        Colour::starts(n, j, cstart);
        const offset_t nsp  = sublattice_numel<ndim>(size + nbatch, kstep, cstart);
        const offset_t nvox = ncell * nsp;
        if (nvox <= 0) continue;
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t * val  = new scalar_t[nc];
        scalar_t * diag = new scalar_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vsol = sublattice<ndim>(asol.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vgrd = sublattice<ndim>(agrd.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vhes = sublattice<ndim>(ahes.template peel_front_at<-ndim>(b), kstep, cstart);
            auto cgrd = tny::peel_front<ndim>(vgrd).subrange(lo, hi).begin();
            auto ches = tny::peel_front<ndim>(vhes).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<ndim>(vsol).enumerate().subrange(lo, hi))
            {
                for (int d=0; d<ndim; ++d)
                    loc[d] = cstart[d] + kstep * v.index[d];
                scalar_t       * solp = v.cell.data();
                const scalar_t * grdp = (*cgrd).data();
                const scalar_t * hesp = (*ches).data();

                // gradient
                for (offset_t c=0; c<nc; ++c)
                    val[c] = grdp[gsc*c];

                // minus convolution
                Impl::template matvec_bending<isub>(
                    val, solp,
                    loc, size + nbatch, stride_sol + nbatch,
                    static_cast<offset_t>(1), osc, kernel, nc);

                // diagonal
                Impl::template diag_bending<set>(
                    diag, static_cast<offset_t>(1), loc, size + nbatch, kernel, nc);

                // sol += (hes + diag) \ (grad - conv(sol))
                PosDef::relax_(
                    nc,
                    Strided(solp, osc),
                    StridedConst(hesp, hsc),
                    val, diag, buf, static_cast<reduce_t>(0)
                );
                ++cgrd; ++ches;
            }
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    }
    delete[] kernel;
}

//======================================================================
//         WEIGHTED (RLS / JRLS) ENTRY POINTS -- READ THIS FIRST
//======================================================================
//
// Parameter-comment notation: the last slot of `(*batch, *spatial, N)` /
// `[*batch, *spatial, N]` is the trailing CHANNEL axis' extent. `0` here,
// `C` in reg_flow.h and `channels` in the `diag_*` entry points are three
// spellings of ONE thing -- the field's own channel count -- not three
// different shapes.
//
// `wgt` is the one parameter whose channel extent is not the field's, and
// it differs between the two weighting modes:
//
//   *_rls   RLS  -- one weight map PER CHANNEL, (*batch, *spatial, C).
//                   The kernel advances the weight pointer by `wsc` with
//                   the channel (`wmode::split`, fastfields-kernels
//                   `regularisers/stencil.h`).
//   *_jrls  JRLS -- ONE weight map shared by every channel,
//                   (*batch, *spatial, 1), read once and hoisted out of
//                   the channel loop (`wmode::joint`).
//
// This is not a cosmetic distinction: handing an RLS entry point a
// single-channel map makes the per-channel advance walk off the end of
// the buffer -- a heap out-of-bounds read, diagnosed in
// fastfields-kernels#67 and traced back to these comments being wrong.
//
//======================================================================
//                           ABSOLUTE RLS
//======================================================================

// --- ABSOLUTE+RLS: matvec --------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_absolute_rls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, 0) tensor
    const scalar_t * inp,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 0) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_out,    // [*batch, *spatial, 0] vector
    const offset_t * stride_inp,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * absolute
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    const offset_t nall = nbatch + ndim;
    const offset_t nc   = size[nall];

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);

    const offset_t nvox = ao.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        auto ic = ai.template peel_front_at<-1>(i);
        auto wc = aw.template peel_front_at<-1>(i);
        Impl::template matvec_absolute_rls<op_apply<op, scalar_t, reduce_t> >(
            oc.data(), ic.data(), wc.data(),
            oc.stride(0), ic.stride(0), wc.stride(0), kernel, nc);
    }});
    delete[] kernel;
}

// --- ABSOLUTE+RLS: diagonal ------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_absolute_rls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, channels] vector
    const reduce_t * absolute
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    const offset_t nall = nbatch + ndim;
    const offset_t nc   = size[nall];

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);

    const offset_t nvox = ao.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        auto wc = aw.template peel_front_at<-1>(i);
        Impl::template diag_absolute_rls<op_apply<op, scalar_t, reduce_t> >(
            oc.data(), wc.data(), oc.stride(0), wc.stride(0), kernel, nc);
    }});
    delete[] kernel;
}

// --- ABSOLUTE+RLS: relax ---------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_absolute_rls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 0) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * absolute,
          int        niter=1
)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];
    offset_t wsc    = stride_wgt[nall];
    offset_t nc    = size[nall];
    offset_t numel  = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size(nc);

    for (offset_t n=0; n<niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        scalar_t * val  = new scalar_t[nc];
        scalar_t * diag = new scalar_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t sol_offset = index2offset(i, nall, size, stride_sol);
            offset_t grd_offset = index2offset(i, nall, size, stride_grd);
            offset_t hes_offset = index2offset(i, nall, size, stride_hes);
            offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

            // gradient
            for (offset_t c=0; c<nc; ++c)
                val[c] = grd[grd_offset + gsc*c];

            // minus convolution
            Impl::template matvec_absolute_rls<isub>(
                val, sol + sol_offset, wgt + wgt_offset,
                static_cast<offset_t>(1), osc, wsc, kernel, nc);

            // diagonal
            Impl::template diag_absolute_rls<set>(
                diag, wgt + wgt_offset,
                static_cast<offset_t>(1), wsc, kernel, nc);

            // sol += (hes + diag) \ (grad - conv(sol))
            PosDef::relax_(
                nc,
                Strided(sol + sol_offset, osc),
                StridedConst(hes + hes_offset, hsc),
                val, diag, buf, static_cast<reduce_t>(0)
            );
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    delete[] kernel;
}

//======================================================================
//                           ABSOLUTE JRLS
//======================================================================

// --- ABSOLUTE+JRLS: matvec -------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_absolute_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, 0) tensor
    const scalar_t * inp,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_out,    // [*batch, *spatial, 0] vector
    const offset_t * stride_inp,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 1] vector
    const reduce_t * absolute
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    const offset_t nall = nbatch + ndim;
    const offset_t nc   = size[nall];

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);

    const offset_t nvox = ao.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        auto ic = ai.template peel_front_at<-1>(i);
        auto wc = aw.template peel_front_at<-1>(i);
        Impl::template matvec_absolute_jrls<op_apply<op, scalar_t, reduce_t> >(
            oc.data(), ic.data(), wc.data(),
            oc.stride(0), ic.stride(0), kernel, nc);
    }});
    delete[] kernel;
}

// --- ABSOLUTE+JRLS: diagonal -----------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_absolute_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 1] vector
    const reduce_t * absolute
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    const offset_t nall = nbatch + ndim;
    const offset_t nc   = size[nall];

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);

    const offset_t nvox = ao.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        auto wc = aw.template peel_front_at<-1>(i);
        Impl::template diag_absolute_jrls<op_apply<op, scalar_t, reduce_t> >(
            oc.data(), wc.data(), oc.stride(0), kernel, nc);
    }});
    delete[] kernel;
}

// --- ABSOLUTE+JRLS: relax --------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_absolute_jrls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 1] vector
    const reduce_t * absolute,
          int        niter=1
)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t, 0>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];
    offset_t nc     = size[nall];
    offset_t numel  = prod(size, nall);    // no outer loop across channels

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_absolute(nc)];
    Impl::make_kernel_absolute(kernel, absolute, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size(nc);

    for (offset_t n=0; n<niter; ++n) {
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
        scalar_t * val  = new scalar_t[nc];
        scalar_t * diag = new scalar_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i=start; i < end; ++i)
        {
            offset_t sol_offset = index2offset(i, nall, size, stride_sol);
            offset_t grd_offset = index2offset(i, nall, size, stride_grd);
            offset_t hes_offset = index2offset(i, nall, size, stride_hes);
            offset_t wgt_offset = index2offset(i, nall, size, stride_wgt);

            // gradient
            for (offset_t c=0; c<nc; ++c)
                val[c] = grd[grd_offset + gsc*c];

            // minus convolution
            Impl::template matvec_absolute_jrls<isub>(
                val, sol + sol_offset, wgt + wgt_offset,
                static_cast<offset_t>(1), osc, kernel, nc);

            // diagonal
            Impl::template diag_absolute_jrls<set>(
                diag, wgt + wgt_offset,
                static_cast<offset_t>(1), kernel, nc);

            // sol += (hes + diag) \ (grad - conv(sol))
            PosDef::relax_(
                nc,
                Strided(sol + sol_offset, osc),
                StridedConst(hes + hes_offset, hsc),
                val, diag, buf, static_cast<reduce_t>(0)
            );
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    delete[] kernel;
}

//======================================================================
//                           MEMBRANE RLS
//======================================================================

// --- MEMBRANE+RLS: matvec --------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_membrane_rls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, 0) tensor
    const scalar_t * inp,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 0) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_out,    // [*batch, *spatial, 0] vector
    const offset_t * stride_inp,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t isc  = stride_inp[nall];
    const offset_t wsc  = stride_wgt[nall];
    const offset_t nc   = size[nall];

    // stencil + weight field -> peel out/inp/wgt volumes, decode loc, offset each.
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane_rls(nc)];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size, nc);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
    {
        const offset_t b = i / nsp, lo = i - b * nsp;
        const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
        i += hi - lo;
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
        auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
        auto ci = tny::peel_front<-1>(vi).subrange(lo, hi).begin();
        auto cw = tny::peel_front<-1>(vw).subrange(lo, hi).begin();
        for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
        {
            Impl::template matvec_membrane_rls<op_apply<op, scalar_t, reduce_t> >(
                v.cell.data(), (*ci).data(), (*cw).data(),
                v.index.data(), size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
                osc, isc, wsc, kernel, nc);
            ++ci;
            ++cw;
        }
    }});
    delete[] kernel;
}

// --- MEMBRANE+RLS: diagonal ------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_membrane_rls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t wsc  = stride_wgt[nall];
    const offset_t nc   = size[nall];

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane_rls(nc)];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size, nc);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
    {
        const offset_t b = i / nsp, lo = i - b * nsp;
        const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
        i += hi - lo;
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
        auto cw = tny::peel_front<-1>(vw).subrange(lo, hi).begin();
        for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
        {
            Impl::template diag_membrane_rls<op_apply<op, scalar_t, reduce_t> >(
                v.cell.data(), (*cw).data(),
                v.index.data(), size + nbatch, stride_wgt + nbatch, osc, wsc, kernel, nc);
            ++cw;
        }
    }});
    delete[] kernel;
}

// --- MEMBRANE+RLS: relax ---------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_membrane_rls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 0) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
          int        niter=1
)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t, 0>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];
    offset_t wsc    = stride_wgt[nall];
    offset_t nc     = size[nall];

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane_rls(nc)];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size;

    // Coloured sweep over dense sub-lattice views (see colour_lattice
    // above): one `subsample` per pass, so a pass touches only its own
    // colour, instead of visiting every voxel and discarding the rest. The
    // wrapped index domain is (*batch, *spatial) -- the channel axis is left
    // off, since every kernel below takes a base pointer plus its own channel
    // stride and never indexes that axis here.
    using Colour = colour_lattice<ndim, 2, true>;   // was patch1
    const offset_t kstep = 2;
    auto asol = tny::as_anyrank(sol, size, stride_sol, static_cast<int>(nall), tny::copy_meta);
    auto agrd = tny::as_anyrank(grd, size, stride_grd, static_cast<int>(nall), tny::copy_meta);
    auto ahes = tny::as_anyrank(hes, size, stride_hes, static_cast<int>(nall), tny::copy_meta);
    auto awgt = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall), tny::copy_meta);
    const offset_t ncell = asol.template size_front<-ndim>();

    for (offset_t n=0; n<2*niter; ++n) {
    for (int j=0; j<Colour::nsub; ++j) {
        offset_t cstart[ndim];
        Colour::starts(n, j, cstart);
        const offset_t nsp  = sublattice_numel<ndim>(size + nbatch, kstep, cstart);
        const offset_t nvox = ncell * nsp;
        if (nvox <= 0) continue;
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t * val  = new scalar_t[nc];
        scalar_t * diag = new scalar_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vsol = sublattice<ndim>(asol.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vgrd = sublattice<ndim>(agrd.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vhes = sublattice<ndim>(ahes.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vwgt = sublattice<ndim>(awgt.template peel_front_at<-ndim>(b), kstep, cstart);
            auto cgrd = tny::peel_front<ndim>(vgrd).subrange(lo, hi).begin();
            auto ches = tny::peel_front<ndim>(vhes).subrange(lo, hi).begin();
            auto cwgt = tny::peel_front<ndim>(vwgt).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<ndim>(vsol).enumerate().subrange(lo, hi))
            {
                for (int d=0; d<ndim; ++d)
                    loc[d] = cstart[d] + kstep * v.index[d];
                scalar_t       * solp = v.cell.data();
                const scalar_t * grdp = (*cgrd).data();
                const scalar_t * hesp = (*ches).data();
                const scalar_t * wgtp = (*cwgt).data();

                // gradient
                for (offset_t c=0; c<nc; ++c)
                    val[c] = grdp[gsc*c];

                // minus convolution
                Impl::template matvec_membrane_rls<isub>(
                    val, solp, wgtp,
                    loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
                    static_cast<offset_t>(1), osc, wsc, kernel, nc);

                // diagonal
                Impl::template diag_membrane_rls<set>(
                    diag, wgtp, loc,
                    size + nbatch, stride_wgt + nbatch,
                    static_cast<offset_t>(1), wsc, kernel, nc);

                // sol += (hes + diag) \ (grad - conv(sol))
                PosDef::relax_(
                    nc,
                    Strided(solp, osc),
                    StridedConst(hesp, hsc),
                    val, diag, buf, static_cast<reduce_t>(0)
                );
                ++cgrd; ++ches; ++cwgt;
            }
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    }
    delete[] kernel;
}

//======================================================================
//                           MEMBRANE JRLS
//======================================================================

// --- MEMBRANE+JRLS: matvec -------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_membrane_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, 0) tensor
    const scalar_t * inp,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_out,    // [*batch, *spatial, 0] vector
    const offset_t * stride_inp,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 1] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t isc  = stride_inp[nall];
    const offset_t nc   = size[nall];

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane_rls(nc)];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size, nc);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
    {
        const offset_t b = i / nsp, lo = i - b * nsp;
        const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
        i += hi - lo;
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
        auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
        auto ci = tny::peel_front<-1>(vi).subrange(lo, hi).begin();
        auto cw = tny::peel_front<-1>(vw).subrange(lo, hi).begin();
        for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
        {
            Impl::template matvec_membrane_jrls<op_apply<op, scalar_t, reduce_t> >(
                v.cell.data(), (*ci).data(), (*cw).data(),
                v.index.data(), size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
                osc, isc, kernel, nc);
            ++ci;
            ++cw;
        }
    }});
    delete[] kernel;
}

// --- MEMBRANE+JRLS: diagonal -----------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_membrane_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 1] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t nc   = size[nall];

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane_rls(nc)];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size, nc);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
    {
        const offset_t b = i / nsp, lo = i - b * nsp;
        const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
        i += hi - lo;
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
        auto cw = tny::peel_front<-1>(vw).subrange(lo, hi).begin();
        for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
        {
            Impl::template diag_membrane_jrls<op_apply<op, scalar_t, reduce_t> >(
                v.cell.data(), (*cw).data(),
                v.index.data(), size + nbatch, stride_wgt + nbatch, osc, kernel, nc);
            ++cw;
        }
    }});
    delete[] kernel;
}

// --- MEMBRANE+JRLS: relax ---------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_membrane_jrls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 1] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
          int        niter=1
)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t, 0>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall  = nbatch + ndim;
    offset_t osc   = stride_sol[nall];
    offset_t hsc   = stride_hes[nall];
    offset_t gsc   = stride_grd[nall];
    offset_t nc    = size[nall];

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_membrane_rls(nc)];
    Impl::make_kernel_membrane_rls(kernel, absolute, membrane, voxel_size, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size(nc);

    // Coloured sweep over dense sub-lattice views (see colour_lattice
    // above): one `subsample` per pass, so a pass touches only its own
    // colour, instead of visiting every voxel and discarding the rest. The
    // wrapped index domain is (*batch, *spatial) -- the channel axis is left
    // off, since every kernel below takes a base pointer plus its own channel
    // stride and never indexes that axis here.
    using Colour = colour_lattice<ndim, 2, true>;   // was patch1
    const offset_t kstep = 2;
    auto asol = tny::as_anyrank(sol, size, stride_sol, static_cast<int>(nall), tny::copy_meta);
    auto agrd = tny::as_anyrank(grd, size, stride_grd, static_cast<int>(nall), tny::copy_meta);
    auto ahes = tny::as_anyrank(hes, size, stride_hes, static_cast<int>(nall), tny::copy_meta);
    auto awgt = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall), tny::copy_meta);
    const offset_t ncell = asol.template size_front<-ndim>();

    for (offset_t n=0; n<2*niter; ++n) {
    for (int j=0; j<Colour::nsub; ++j) {
        offset_t cstart[ndim];
        Colour::starts(n, j, cstart);
        const offset_t nsp  = sublattice_numel<ndim>(size + nbatch, kstep, cstart);
        const offset_t nvox = ncell * nsp;
        if (nvox <= 0) continue;
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        offset_t   loc[ndim];
        scalar_t * val  = new scalar_t[nc];
        scalar_t * diag = new scalar_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vsol = sublattice<ndim>(asol.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vgrd = sublattice<ndim>(agrd.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vhes = sublattice<ndim>(ahes.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vwgt = sublattice<ndim>(awgt.template peel_front_at<-ndim>(b), kstep, cstart);
            auto cgrd = tny::peel_front<ndim>(vgrd).subrange(lo, hi).begin();
            auto ches = tny::peel_front<ndim>(vhes).subrange(lo, hi).begin();
            auto cwgt = tny::peel_front<ndim>(vwgt).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<ndim>(vsol).enumerate().subrange(lo, hi))
            {
                for (int d=0; d<ndim; ++d)
                    loc[d] = cstart[d] + kstep * v.index[d];
                scalar_t       * solp = v.cell.data();
                const scalar_t * grdp = (*cgrd).data();
                const scalar_t * hesp = (*ches).data();
                const scalar_t * wgtp = (*cwgt).data();

                // gradient
                for (offset_t c=0; c<nc; ++c)
                    val[c] = grdp[gsc*c];

                // minus convolution
                Impl::template matvec_membrane_jrls<isub>(
                    val, solp, wgtp,
                    loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
                    static_cast<offset_t>(1), osc, kernel, nc);

                // diagonal
                Impl::template diag_membrane_jrls<set>(
                    diag, wgtp, loc,
                    size + nbatch, stride_wgt + nbatch,
                    static_cast<offset_t>(1), kernel, nc);

                // sol += (hes + diag) \ (grad - conv(sol))
                PosDef::relax_(
                    nc,
                    Strided(solp, osc),
                    StridedConst(hesp, hsc),
                    val, diag, buf, static_cast<reduce_t>(0)
                );
                ++cgrd; ++ches; ++cwgt;
            }
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    }
    delete[] kernel;
}

//======================================================================
//                           BENDING RLS
//======================================================================

// --- BENDING+RLS: matvec ---------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_bending_rls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, 0) tensor
    const scalar_t * inp,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 0) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_out,    // [*batch, *spatial, 0] vector
    const offset_t * stride_inp,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t isc  = stride_inp[nall];
    const offset_t wsc  = stride_wgt[nall];
    const offset_t nc   = size[nall];

    // stencil + weight field -> peel out/inp/wgt volumes, decode loc, offset each.
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending_rls(nc)];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size, nc);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
    {
        const offset_t b = i / nsp, lo = i - b * nsp;
        const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
        i += hi - lo;
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
        auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
        auto ci = tny::peel_front<-1>(vi).subrange(lo, hi).begin();
        auto cw = tny::peel_front<-1>(vw).subrange(lo, hi).begin();
        for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
        {
            Impl::template matvec_bending_rls<op_apply<op, scalar_t, reduce_t> >(
                v.cell.data(), (*ci).data(), (*cw).data(),
                v.index.data(), size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
                osc, isc, wsc, kernel, nc);
            ++ci;
            ++cw;
        }
    }});
    delete[] kernel;
}

// --- BENDING+RLS: diagonal -------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_bending_rls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, channels) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, channels] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t wsc  = stride_wgt[nall];
    const offset_t nc   = size[nall];

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending_rls(nc)];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size, nc);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
    {
        const offset_t b = i / nsp, lo = i - b * nsp;
        const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
        i += hi - lo;
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
        auto cw = tny::peel_front<-1>(vw).subrange(lo, hi).begin();
        for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
        {
            Impl::template diag_bending_rls<op_apply<op, scalar_t, reduce_t> >(
                v.cell.data(), (*cw).data(),
                v.index.data(), size + nbatch, stride_wgt + nbatch, osc, wsc, kernel, nc);
            ++cw;
        }
    }});
    delete[] kernel;
}

// --- BENDING+RLS: relax ----------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_bending_rls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 0) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 0] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending,
          int        niter=1
)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall    = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];
    offset_t wsc    = stride_wgt[nall];
    offset_t nc     = size[nall];

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending_rls(nc)];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size(nc);

    // Coloured sweep over dense sub-lattice views (see colour_lattice
    // above): one `subsample` per pass, so a pass touches only its own
    // colour, instead of visiting every voxel and discarding the rest. The
    // wrapped index domain is (*batch, *spatial) -- the channel axis is left
    // off, since every kernel below takes a base pointer plus its own channel
    // stride and never indexes that axis here.
    using Colour = colour_lattice<ndim, 3, false>;   // was patch3
    const offset_t kstep = 3;
    auto asol = tny::as_anyrank(sol, size, stride_sol, static_cast<int>(nall), tny::copy_meta);
    auto agrd = tny::as_anyrank(grd, size, stride_grd, static_cast<int>(nall), tny::copy_meta);
    auto ahes = tny::as_anyrank(hes, size, stride_hes, static_cast<int>(nall), tny::copy_meta);
    auto awgt = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall), tny::copy_meta);
    const offset_t ncell = asol.template size_front<-ndim>();

    for (offset_t n=0; n<2*niter; ++n) {
    for (int j=0; j<Colour::nsub; ++j) {
        offset_t cstart[ndim];
        Colour::starts(n, j, cstart);
        const offset_t nsp  = sublattice_numel<ndim>(size + nbatch, kstep, cstart);
        const offset_t nvox = ncell * nsp;
        if (nvox <= 0) continue;
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t * val  = new scalar_t[nc];
        scalar_t * diag = new scalar_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vsol = sublattice<ndim>(asol.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vgrd = sublattice<ndim>(agrd.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vhes = sublattice<ndim>(ahes.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vwgt = sublattice<ndim>(awgt.template peel_front_at<-ndim>(b), kstep, cstart);
            auto cgrd = tny::peel_front<ndim>(vgrd).subrange(lo, hi).begin();
            auto ches = tny::peel_front<ndim>(vhes).subrange(lo, hi).begin();
            auto cwgt = tny::peel_front<ndim>(vwgt).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<ndim>(vsol).enumerate().subrange(lo, hi))
            {
                for (int d=0; d<ndim; ++d)
                    loc[d] = cstart[d] + kstep * v.index[d];
                scalar_t       * solp = v.cell.data();
                const scalar_t * grdp = (*cgrd).data();
                const scalar_t * hesp = (*ches).data();
                const scalar_t * wgtp = (*cwgt).data();

                // gradient
                for (offset_t c=0; c<nc; ++c)
                    val[c] = grdp[gsc*c];

                // minus convolution
                Impl::template matvec_bending_rls<isub>(
                    val, solp, wgtp,
                    loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
                    static_cast<offset_t>(1), osc, wsc, kernel, nc);

                // diagonal
                Impl::template diag_bending_rls<set>(
                    diag, wgtp, loc,
                    size + nbatch, stride_wgt + nbatch,
                    static_cast<offset_t>(1), wsc, kernel, nc);

                // sol += (hes + diag) \ (grad - conv(sol))
                PosDef::relax_(
                    nc,
                    Strided(solp, osc),
                    StridedConst(hesp, hsc),
                    val, diag, buf, static_cast<reduce_t>(0)
                );
                ++cgrd; ++ches; ++cwgt;
            }
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    }
    delete[] kernel;
}

//======================================================================
//                           BENDING JRLS
//======================================================================

// --- BENDING+JRLS: matvec --------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_bending_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, 0) tensor
    const scalar_t * inp,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_out,    // [*batch, *spatial, 0] vector
    const offset_t * stride_inp,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 1] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t isc  = stride_inp[nall];
    const offset_t nc   = size[nall];

    // stencil + weight field -> peel out/inp/wgt volumes, decode loc, offset each.
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending_rls(nc)];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size, nc);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
    {
        const offset_t b = i / nsp, lo = i - b * nsp;
        const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
        i += hi - lo;
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vi = ai.template peel_front_at<-(ndim + 1)>(b);
        auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
        auto ci = tny::peel_front<-1>(vi).subrange(lo, hi).begin();
        auto cw = tny::peel_front<-1>(vw).subrange(lo, hi).begin();
        for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
        {
            Impl::template matvec_bending_jrls<op_apply<op, scalar_t, reduce_t> >(
                v.cell.data(), (*ci).data(), (*cw).data(),
                v.index.data(), size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
                osc, isc, kernel, nc);
            ++ci;
            ++cw;
        }
    }});
    delete[] kernel;
}

// --- BENDING+JRLS: diagonal ------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_bending_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 1] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending
)
{
    using Impl = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t nc   = size[nall];

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending_rls(nc)];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size, nc);

    const offset_t nsp  = prod(size + nbatch, ndim);
    const offset_t nvox = ao.template size_front<-(ndim + 1)>() * nsp;

    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
    {
        const offset_t b = i / nsp, lo = i - b * nsp;
        const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
        i += hi - lo;
        auto vo = ao.template peel_front_at<-(ndim + 1)>(b);
        auto vw = aw.template peel_front_at<-(ndim + 1)>(b);
        auto cw = tny::peel_front<-1>(vw).subrange(lo, hi).begin();
        for (auto v : tny::peel_front<-1>(vo).enumerate().subrange(lo, hi))
        {
            Impl::template diag_bending_jrls<op_apply<op, scalar_t, reduce_t> >(
                v.cell.data(), (*cw).data(),
                v.index.data(), size + nbatch, stride_wgt + nbatch, osc, kernel, nc);
            ++cw;
        }
    }});
    delete[] kernel;
}

// --- BENDING+JRLS: relax ---------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_bending_jrls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, 0) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, 0) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, 0] vector
    const offset_t * stride_sol,    // [*batch, *spatial, 0] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, 0] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 1] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
    const reduce_t * absolute,
    const reduce_t * membrane,
    const reduce_t * bending,
          int        niter=1
)
{
    using Impl          = RegField<0, ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall  = nbatch + ndim;
    offset_t osc   = stride_sol[nall];
    offset_t hsc   = stride_hes[nall];
    offset_t gsc   = stride_grd[nall];
    offset_t nc    = size[nall];

    reduce_t * kernel = new reduce_t[Impl::get_kernelsize_bending_rls(nc)];
    Impl::make_kernel_bending_rls(kernel, absolute, membrane, bending, voxel_size, nc);
    offset_t ncc = posdef::utils<posdef::type::Sym, offset_t>::work_size(nc);

    // Coloured sweep over dense sub-lattice views (see colour_lattice
    // above): one `subsample` per pass, so a pass touches only its own
    // colour, instead of visiting every voxel and discarding the rest. The
    // wrapped index domain is (*batch, *spatial) -- the channel axis is left
    // off, since every kernel below takes a base pointer plus its own channel
    // stride and never indexes that axis here.
    using Colour = colour_lattice<ndim, 3, false>;   // was patch3
    const offset_t kstep = 3;
    auto asol = tny::as_anyrank(sol, size, stride_sol, static_cast<int>(nall), tny::copy_meta);
    auto agrd = tny::as_anyrank(grd, size, stride_grd, static_cast<int>(nall), tny::copy_meta);
    auto ahes = tny::as_anyrank(hes, size, stride_hes, static_cast<int>(nall), tny::copy_meta);
    auto awgt = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall), tny::copy_meta);
    const offset_t ncell = asol.template size_front<-ndim>();

    for (offset_t n=0; n<2*niter; ++n) {
    for (int j=0; j<Colour::nsub; ++j) {
        offset_t cstart[ndim];
        Colour::starts(n, j, cstart);
        const offset_t nsp  = sublattice_numel<ndim>(size + nbatch, kstep, cstart);
        const offset_t nvox = ncell * nsp;
        if (nvox <= 0) continue;
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        offset_t   loc[ndim];
        scalar_t * val  = new scalar_t[nc];
        scalar_t * diag = new scalar_t[nc];
        reduce_t * buf  = ncc ? new reduce_t[ncc] : nullptr;
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vsol = sublattice<ndim>(asol.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vgrd = sublattice<ndim>(agrd.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vhes = sublattice<ndim>(ahes.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vwgt = sublattice<ndim>(awgt.template peel_front_at<-ndim>(b), kstep, cstart);
            auto cgrd = tny::peel_front<ndim>(vgrd).subrange(lo, hi).begin();
            auto ches = tny::peel_front<ndim>(vhes).subrange(lo, hi).begin();
            auto cwgt = tny::peel_front<ndim>(vwgt).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<ndim>(vsol).enumerate().subrange(lo, hi))
            {
                for (int d=0; d<ndim; ++d)
                    loc[d] = cstart[d] + kstep * v.index[d];
                scalar_t       * solp = v.cell.data();
                const scalar_t * grdp = (*cgrd).data();
                const scalar_t * hesp = (*ches).data();
                const scalar_t * wgtp = (*cwgt).data();

                // gradient
                for (offset_t c=0; c<nc; ++c)
                    val[c] = grdp[gsc*c];

                // minus convolution
                Impl::template matvec_bending_jrls<isub>(
                    val, solp, wgtp,
                    loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
                    static_cast<offset_t>(1), osc, kernel, nc);

                // diagonal
                Impl::template diag_bending_jrls<set>(
                    diag, wgtp, loc,
                    size + nbatch, stride_wgt + nbatch,
                    static_cast<offset_t>(1), kernel, nc);

                // sol += (hes + diag) \ (grad - conv(sol))
                PosDef::relax_(
                    nc,
                    Strided(solp, osc),
                    StridedConst(hesp, hsc),
                    val, diag, buf, static_cast<reduce_t>(0)
                );
                ++cgrd; ++ches; ++cwgt;
            }
        }
        delete[] val;
        delete[] diag;
        if (ncc) delete[] buf;
    });
    }
    }
    delete[] kernel;
}

FF_NAMESPACE_END(reg_field)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_REGULARISERS_FIELD_CPU
