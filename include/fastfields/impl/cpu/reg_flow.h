#ifndef FF_REGULARISERS_FLOW_CPU
#define FF_REGULARISERS_FLOW_CPU
#include <stdexcept>
#include <utility>
#include <teeny/teeny.h>
#include "kernels/cuda_switch.h"
#include "kernels/bounds.h"
#include "kernels/utils.h"
#include "kernels/batch.h"
#include "kernels/parallel.h"
#include "kernels/regularisers/flow.h"
#include "kernels/posdef.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(reg_flow)

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
    const reduce_t * _voxel_size,    // [*spatial] vector
          reduce_t   absolute
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t isc  = stride_inp[nall];

    // compute kernel
    reduce_t kernel[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kernel, absolute, voxel_size);

    // pointwise: peel each (*batch,*spatial,C) tensor down to its per-voxel C cell.
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nvox = ao.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        auto oc = ao.template peel_front_at<-1>(i);
        auto ic = ai.template peel_front_at<-1>(i);
        Impl::template matvec_absolute<op_apply<op, scalar_t, reduce_t> >(
            oc.data(), ic.data(), osc, isc, kernel);
    }});
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
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size [ndim];   fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t sc     = stride[nall];

    reduce_t kernel[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kernel, absolute, voxel_size);

    offset_t offset = center_offset<ndim>(size+nbatch, stride+nbatch);

    // batch-only: peel each (*batch,*spatial) cell to its batch origin, write
    // the kernel at the spatial center (no per-voxel spatial decode needed).
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall), tny::copy_meta);
    const offset_t nbcell = ao.template size_front<-ndim>();

    parallel_for(0, nbcell, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        auto vo = ao.template peel_front_at<-ndim>(i);
        Impl::template kernel_absolute<op_apply<op, scalar_t, reduce_t> >(vo.data() + offset, sc, kernel);
    }});
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
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t sc   = stride[nall];

    reduce_t kernel[Impl::kernelsize_absolute];
    Impl::make_kernel_absolute(kernel, absolute, voxel_size);

    // pointwise: peel to the per-voxel C cell (the diagonal is position-independent).
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

    const offset_t nvox = ao.template size_front<-1>();
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            auto oc = ao.template peel_front_at<-1>(i);
            Impl::template diag_absolute<op_apply<op, scalar_t, reduce_t> >(oc.data(), sc, kernel);
        }
    });
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
          reduce_t   absolute,
          reduce_t   membrane
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t isc  = stride_inp[nall];

    reduce_t kernel[Impl::kernelsize_membrane];
    Impl::make_kernel_membrane(kernel, absolute, membrane, voxel_size);

    // stencil: peel out/inp volumes per batch cell, decode loc + spatial offset per voxel.
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);

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
                Impl::template matvec_membrane<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), (*ci).data(),
                    v.index.data(), size + nbatch, stride_inp + nbatch, osc, isc, kernel);
                ++ci;
            }
        }
    });
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
          reduce_t   absolute,
          reduce_t   membrane
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t sc     = stride[nall];

    reduce_t kernel[Impl::kernelsize_membrane];
    Impl::make_fullkernel_membrane(kernel, absolute, membrane, voxel_size);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    // batch-only: peel each (*batch,*spatial) cell to its batch origin, write
    // the full kernel at the spatial center.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall), tny::copy_meta);
    const offset_t nbcell = ao.template size_front<-ndim>();

    parallel_for(0, nbcell, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            auto vo = ao.template peel_front_at<-ndim>(i);

            Impl::template kernel_membrane<op_apply<op, scalar_t, reduce_t> >(
                vo.data() + offset, sc, stride + nbatch, kernel);
        }
    });
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
          reduce_t   absolute,
          reduce_t   membrane
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t,  offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t sc   = stride[nall];

    reduce_t kernel[Impl::kernelsize_membrane];
    Impl::make_kernel_membrane(kernel, absolute, membrane, voxel_size);

    // stencil (boundary-aware diagonal): peel per batch cell, decode loc + offset per voxel.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

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
                    v.cell.data(), sc, v.index.data(), size + nbatch, kernel);
        }
    });
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
          scalar_t * sol,           // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_sol,    // [*batch, *spatial, C] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          int        niter=1
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];

    reduce_t kernel[Impl::kernelsize_membrane];
    Impl::make_kernel_membrane(kernel, absolute, membrane, voxel_size);
    constexpr int DD = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    // Peel the (*batch, *spatial) index domain per batch cell (rank
    // nbatch+ndim carriers -- no trailing channel axis needed, only the
    // batch origin pointer matters here).
    auto as = tny::as_anyrank(sol, size, stride_sol, static_cast<int>(nall), tny::copy_meta);
    auto ah = tny::as_anyrank(hes, size, stride_hes, static_cast<int>(nall), tny::copy_meta);
    auto ag = tny::as_anyrank(grd, size, stride_grd, static_cast<int>(nall), tny::copy_meta);

    // Coloured sweep over dense sub-lattice views (see colour_lattice
    // above): one `subsample` per pass, so a pass touches only its own
    // colour, instead of visiting every voxel and discarding the rest.
    using Colour = colour_lattice<ndim, 2, true>;   // was patch1
    const offset_t kstep = 2;
    const offset_t ncell = as.template size_front<-ndim>();

    for (offset_t n=0; n<2*niter; ++n) {
    for (int j=0; j<Colour::nsub; ++j) {
        offset_t cstart[ndim];
        Colour::starts(n, j, cstart);
        const offset_t nsp  = sublattice_numel<ndim>(size + nbatch, kstep, cstart);
        const offset_t nvox = ncell * nsp;
        if (nvox <= 0) continue;
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t val[ndim], diag[ndim];
        reduce_t buf[DD ? DD : 1];
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vs = sublattice<ndim>(as.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vh = sublattice<ndim>(ah.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vg = sublattice<ndim>(ag.template peel_front_at<-ndim>(b), kstep, cstart);
            auto cvh = tny::peel_front<ndim>(vh).subrange(lo, hi).begin();
            auto cvg = tny::peel_front<ndim>(vg).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<ndim>(vs).enumerate().subrange(lo, hi))
            {
                for (int d=0; d<ndim; ++d)
                    loc[d] = cstart[d] + kstep * v.index[d];
                scalar_t       * sol_ptr = v.cell.data();
                const scalar_t * hes_ptr = (*cvh).data();
                const scalar_t * grd_ptr = (*cvg).data();

                // gradient
#               pragma unroll
                for (int d=0; d<ndim; ++d)
                    val[d] = grd_ptr[gsc*d];

                // minus convolution
                Impl::template matvec_membrane<isub>(
                    val, sol_ptr,
                    loc, size + nbatch, stride_sol + nbatch,
                    static_cast<offset_t>(1), osc, kernel);

                // diagonal
                Impl::template diag_membrane<set>(
                    diag, static_cast<offset_t>(1), loc, size + nbatch, kernel);

                // sol += (hes + diag) \ (grad - conv(sol))
                PosDef::relax_(
                    Strided(sol_ptr, osc),
                    StridedConst(hes_ptr, hsc),
                    val, diag, buf, static_cast<reduce_t>(0)
                );
                ++cvh; ++cvg;
            }
        }
    });
    }
    }
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
          scalar_t * out,           // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_out,    // [*batch, *spatial, C] vector
    const offset_t * stride_inp,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t absolute,
          reduce_t membrane,
          reduce_t bending
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t isc  = stride_inp[nall];

    reduce_t kernel[Impl::kernelsize_bending];
    Impl::make_kernel_bending(kernel, absolute, membrane, bending, voxel_size);

    // stencil: peel out/inp volumes per batch cell, decode loc + spatial offset per voxel.
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);

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
                    v.index.data(), size + nbatch, stride_inp + nbatch, osc, isc, kernel);
                ++ci;
            }
        }
    });
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
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   bending
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall = nbatch + ndim;
    offset_t sc = stride[nall];

    reduce_t kernel[Impl::kernelsize_bending];
    Impl::make_fullkernel_bending(kernel, absolute, membrane, bending, voxel_size);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    // batch-only: peel each (*batch,*spatial) cell to its batch origin, write
    // the full kernel at the spatial center.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall), tny::copy_meta);
    const offset_t nbcell = ao.template size_front<-ndim>();

    parallel_for(0, nbcell, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            auto vo = ao.template peel_front_at<-ndim>(i);

            Impl::template kernel_bending<op_apply<op, scalar_t, reduce_t> >(
                vo.data() + offset, sc, stride + nbatch, kernel);
        }
    });
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
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   bending
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t sc   = stride[nall];

    reduce_t kernel[Impl::kernelsize_bending];
    Impl::make_kernel_bending(kernel, absolute, membrane, bending, voxel_size);

    // stencil (boundary-aware diagonal): peel per batch cell, decode loc + offset per voxel.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

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
                    v.cell.data(), sc, v.index.data(), size + nbatch, kernel);
        }
    });
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
          scalar_t * sol,           // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_sol,    // [*batch, *spatial, C] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   bending,
          int        niter=1
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall = nbatch + ndim;
    offset_t osc = stride_sol[nall];
    offset_t hsc = stride_hes[nall];
    offset_t gsc = stride_grd[nall];

    reduce_t kernel[Impl::kernelsize_bending];
    Impl::make_kernel_bending(kernel, absolute, membrane, bending, voxel_size);
    constexpr int DD = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    auto as = tny::as_anyrank(sol, size, stride_sol, static_cast<int>(nall), tny::copy_meta);
    auto ah = tny::as_anyrank(hes, size, stride_hes, static_cast<int>(nall), tny::copy_meta);
    auto ag = tny::as_anyrank(grd, size, stride_grd, static_cast<int>(nall), tny::copy_meta);

    // Coloured sweep over dense sub-lattice views (see colour_lattice
    // above): one `subsample` per pass, so a pass touches only its own
    // colour, instead of visiting every voxel and discarding the rest.
    using Colour = colour_lattice<ndim, 3, false>;   // was patch3
    const offset_t kstep = 3;
    const offset_t ncell = as.template size_front<-ndim>();

    for (offset_t n = 0; n < pow<ndim>(3)*niter; ++n) {
    for (int j=0; j<Colour::nsub; ++j) {
        offset_t cstart[ndim];
        Colour::starts(n, j, cstart);
        const offset_t nsp  = sublattice_numel<ndim>(size + nbatch, kstep, cstart);
        const offset_t nvox = ncell * nsp;
        if (nvox <= 0) continue;
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t val[ndim], diag[ndim];
        reduce_t buf[DD ? DD : 1];
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vs = sublattice<ndim>(as.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vh = sublattice<ndim>(ah.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vg = sublattice<ndim>(ag.template peel_front_at<-ndim>(b), kstep, cstart);
            auto cvh = tny::peel_front<ndim>(vh).subrange(lo, hi).begin();
            auto cvg = tny::peel_front<ndim>(vg).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<ndim>(vs).enumerate().subrange(lo, hi))
            {
                for (int d=0; d<ndim; ++d)
                    loc[d] = cstart[d] + kstep * v.index[d];
                scalar_t       * sol_ptr = v.cell.data();
                const scalar_t * hes_ptr = (*cvh).data();
                const scalar_t * grd_ptr = (*cvg).data();

                // gradient
#               pragma unroll
                for (int d=0; d<ndim; ++d)
                    val[d] = grd_ptr[gsc*d];

                // minus convolution
                Impl::template matvec_bending<isub>(
                    val, sol_ptr,
                    loc, size + nbatch, stride_sol + nbatch,
                    static_cast<offset_t>(1), osc, kernel);

                // diagonal
                Impl::template diag_bending<set>(
                    diag, static_cast<offset_t>(1), loc, size + nbatch, kernel);

                // sol += (hes + diag) \ (grad - conv(sol))
                PosDef::relax_(
                    Strided(sol_ptr, osc),
                    StridedConst(hes_ptr, hsc),
                    val, diag, buf, static_cast<reduce_t>(0)
                );
                ++cvh; ++cvg;
            }
        }
    });
    }
    }
}

//======================================================================
//                              LAME
//======================================================================

// --- LAME: matvec ---------------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_lame(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_out,    // [*batch, *spatial, C] vector
    const offset_t * stride_inp,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   shears,
          reduce_t   div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t isc    = stride_inp[nall];
    reduce_t kernel[Impl::kernelsize_lame];
    Impl::make_kernel_lame(kernel, absolute, membrane, shears, div, voxel_size);

    // stencil: peel out/inp volumes per batch cell, then let teeny's spatial
    // peel supply `loc` and the per-voxel base pointers (same shape as
    // matvec_membrane; replaces index2offset_v2).
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);

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
                Impl::template matvec_lame<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), (*ci).data(),
                    v.index.data(), size + nbatch, stride_inp + nbatch, osc, isc, kernel);
                ++ci;
            }
        }
    });
}

// --- LAME: kernel ----------------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void kernel_lame(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, C, C) tensor
    const offset_t * size,          // [*batch, *spatial, C, C] vector
    const offset_t * stride,        // [*batch, *spatial, C, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   shears,
          reduce_t   div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall = nbatch + ndim;

    reduce_t kernel[Impl::kernelsize_lame];
    Impl::make_fullkernel_lame(kernel, absolute, membrane, shears, div, voxel_size);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    // batch-only: peel each (*batch,*spatial) cell to its batch origin, write
    // the full kernel at the spatial center.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall), tny::copy_meta);
    const offset_t nbcell = ao.template size_front<-ndim>();

    parallel_for(0, nbcell, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            auto vo = ao.template peel_front_at<-ndim>(i);

            Impl::template kernel_lame<op_apply<op, scalar_t, reduce_t> >(
                vo.data() + offset, stride + nall, stride + nbatch, kernel);
        }
    });
}

// --- LAME: diagonal --------------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_lame(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride,        // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   shears,
          reduce_t   div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t sc     = stride[nall];
    reduce_t kernel[Impl::kernelsize_lame];
    Impl::make_kernel_lame(kernel, absolute, membrane, shears, div, voxel_size);

    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

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
                Impl::template diag_lame<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), sc, v.index.data(), size + nbatch, kernel);
        }
    });
}
// --- LAME: relax -----------------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_lame_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_sol,    // [*batch, *spatial, C] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   shears,
          reduce_t   div,
          int        niter=1
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];

    reduce_t kernel[Impl::kernelsize_lame];
    Impl::make_kernel_lame(kernel, absolute, membrane, shears, div, voxel_size);
    constexpr int DD = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    auto as = tny::as_anyrank(sol, size, stride_sol, static_cast<int>(nall), tny::copy_meta);
    auto ah = tny::as_anyrank(hes, size, stride_hes, static_cast<int>(nall), tny::copy_meta);
    auto ag = tny::as_anyrank(grd, size, stride_grd, static_cast<int>(nall), tny::copy_meta);

    // Coloured sweep over dense sub-lattice views (see colour_lattice
    // above): one `subsample` per pass, so a pass touches only its own
    // colour, instead of visiting every voxel and discarding the rest.
    using Colour = colour_lattice<ndim, 2, false>;   // was patch2
    const offset_t kstep = 2;
    const offset_t ncell = as.template size_front<-ndim>();

    for (offset_t n = 0; n < pow<ndim>(2)*niter; ++n) {
    for (int j=0; j<Colour::nsub; ++j) {
        offset_t cstart[ndim];
        Colour::starts(n, j, cstart);
        const offset_t nsp  = sublattice_numel<ndim>(size + nbatch, kstep, cstart);
        const offset_t nvox = ncell * nsp;
        if (nvox <= 0) continue;
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t val[ndim], diag[ndim];
        reduce_t buf[DD ? DD : 1];
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vs = sublattice<ndim>(as.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vh = sublattice<ndim>(ah.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vg = sublattice<ndim>(ag.template peel_front_at<-ndim>(b), kstep, cstart);
            auto cvh = tny::peel_front<ndim>(vh).subrange(lo, hi).begin();
            auto cvg = tny::peel_front<ndim>(vg).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<ndim>(vs).enumerate().subrange(lo, hi))
            {
                for (int d=0; d<ndim; ++d)
                    loc[d] = cstart[d] + kstep * v.index[d];
                scalar_t       * sol_ptr = v.cell.data();
                const scalar_t * hes_ptr = (*cvh).data();
                const scalar_t * grd_ptr = (*cvg).data();

                // gradient
#               pragma unroll
                for (int d=0; d<ndim; ++d)
                    val[d] = grd_ptr[gsc*d];

                // minus convolution
                Impl::template matvec_lame<isub>(
                    val, sol_ptr,
                    loc, size + nbatch, stride_sol + nbatch,
                    static_cast<offset_t>(1), osc, kernel);

                // diagonal
                Impl::template diag_lame<set>(
                    diag, static_cast<offset_t>(1), loc, size + nbatch, kernel);

                // sol += (hes + diag) \ (grad - conv(sol))
                PosDef::relax_(
                    Strided(sol_ptr, osc),
                    StridedConst(hes_ptr, hsc),
                    val, diag, buf, static_cast<reduce_t>(0)
                );
                ++cvh; ++cvg;
            }
        }
    });
    }
    }
}

//======================================================================
//                          LAME + BENDING
//======================================================================

// --- BENDING+LAME: matvec -------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_all(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_out,    // [*batch, *spatial, C] vector
    const offset_t * stride_inp,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   bending,
          reduce_t   shears,
          reduce_t   div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t osc  = stride_out[nall];
    const offset_t isc  = stride_inp[nall];

    reduce_t kernel[Impl::kernelsize_all];
    Impl::make_kernel_all(kernel, absolute, membrane, bending, shears, div, voxel_size);

    // stencil: peel out/inp volumes per batch cell, decode loc + spatial offset per voxel.
    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);

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
                Impl::template matvec_all<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), (*ci).data(),
                    v.index.data(), size + nbatch, stride_inp + nbatch, osc, isc, kernel);
                ++ci;
            }
        }
    });
}

// --- BENDING+LAME: kernel --------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void kernel_all(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, C, C) tensor
    const offset_t * size,          // [*batch, *spatial, C, C] vector
    const offset_t * stride,        // [*batch, *spatial, C, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   bending,
          reduce_t   shears,
          reduce_t   div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall = nbatch + ndim;

    reduce_t kernel[Impl::kernelsize_all];
    Impl::make_fullkernel_all(kernel, absolute, membrane, bending, shears, div, voxel_size);

    offset_t offset = center_offset<ndim>(size + nbatch, stride + nbatch);

    // batch-only: peel each (*batch,*spatial) cell to its batch origin, write
    // the full kernel at the spatial center.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall), tny::copy_meta);
    const offset_t nbcell = ao.template size_front<-ndim>();

    parallel_for(0, nbcell, GRAIN_SIZE, [&](long start, long end) {
        for (offset_t i=start; i < end; ++i)
        {
            auto vo = ao.template peel_front_at<-ndim>(i);

            Impl::template kernel_all<op_apply<op, scalar_t, reduce_t> >(
                vo.data() + offset, stride + nall, stride + nbatch, kernel);
        }
    });
}

// --- BENDING+LAME: diagonal ------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_all(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride,        // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   bending,
          reduce_t   shears,
          reduce_t   div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    const offset_t nall = nbatch + ndim;
    const offset_t sc   = stride[nall];

    reduce_t kernel[Impl::kernelsize_all];
    Impl::make_kernel_all(kernel, absolute, membrane, bending, shears, div, voxel_size);

    // stencil (boundary-aware diagonal): peel per batch cell, decode loc + offset per voxel.
    auto ao = tny::as_anyrank(out, size, stride, static_cast<int>(nall) + 1, tny::copy_meta);

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
                Impl::template diag_all<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), sc, v.index.data(), size + nbatch, kernel);
        }
    });
}

// --- BENDING+LAME: relax ---------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_all_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_sol,    // [*batch, *spatial, C] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, C] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   bending,
          reduce_t   shears,
          reduce_t   div,
          int        niter=1
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];

    reduce_t kernel[Impl::kernelsize_all];
    Impl::make_kernel_all(kernel, absolute, membrane, bending, shears, div, voxel_size);
    constexpr int CC = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    auto as = tny::as_anyrank(sol, size, stride_sol, static_cast<int>(nall), tny::copy_meta);
    auto ah = tny::as_anyrank(hes, size, stride_hes, static_cast<int>(nall), tny::copy_meta);
    auto ag = tny::as_anyrank(grd, size, stride_grd, static_cast<int>(nall), tny::copy_meta);

    // Coloured sweep over dense sub-lattice views (see colour_lattice
    // above): one `subsample` per pass, so a pass touches only its own
    // colour, instead of visiting every voxel and discarding the rest.
    using Colour = colour_lattice<ndim, 3, false>;   // was patch3
    const offset_t kstep = 3;
    const offset_t ncell = as.template size_front<-ndim>();

    for (offset_t n = 0; n < pow<ndim>(3)*niter; ++n) {
    for (int j=0; j<Colour::nsub; ++j) {
        offset_t cstart[ndim];
        Colour::starts(n, j, cstart);
        const offset_t nsp  = sublattice_numel<ndim>(size + nbatch, kstep, cstart);
        const offset_t nvox = ncell * nsp;
        if (nvox <= 0) continue;
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t val[ndim], diag[ndim];
        reduce_t buf[CC ? CC : 1];
        for (offset_t i = static_cast<offset_t>(start), e = static_cast<offset_t>(end); i < e; )
        {
            const offset_t b = i / nsp, lo = i - b * nsp;
            const offset_t hi = (e - b * nsp < nsp) ? e - b * nsp : nsp;
            i += hi - lo;
            auto vs = sublattice<ndim>(as.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vh = sublattice<ndim>(ah.template peel_front_at<-ndim>(b), kstep, cstart);
            auto vg = sublattice<ndim>(ag.template peel_front_at<-ndim>(b), kstep, cstart);
            auto cvh = tny::peel_front<ndim>(vh).subrange(lo, hi).begin();
            auto cvg = tny::peel_front<ndim>(vg).subrange(lo, hi).begin();
            for (auto v : tny::peel_front<ndim>(vs).enumerate().subrange(lo, hi))
            {
                for (int d=0; d<ndim; ++d)
                    loc[d] = cstart[d] + kstep * v.index[d];
                scalar_t       * sol_ptr = v.cell.data();
                const scalar_t * hes_ptr = (*cvh).data();
                const scalar_t * grd_ptr = (*cvg).data();

                // gradient
#               pragma unroll
                for (int d=0; d<ndim; ++d)
                    val[d] = grd_ptr[gsc*d];

                // minus convolution
                Impl::template matvec_all<isub>(
                    val, sol_ptr,
                    loc, size + nbatch, stride_sol + nbatch,
                    static_cast<offset_t>(1), osc, kernel);

                // diagonal
                Impl::template diag_all<set>(
                    diag, static_cast<offset_t>(1), loc, size + nbatch, kernel);

                // sol += (hes + diag) \ (grad - conv(sol))
                PosDef::relax_(
                    Strided(sol_ptr, osc),
                    StridedConst(hes_ptr, hsc),
                    val, diag, buf, static_cast<reduce_t>(0)
                );
                ++cvh; ++cvg;
            }
        }
    });
    }
    }
}

//======================================================================
//          WEIGHTED (JRLS) ENTRY POINTS -- READ THIS FIRST
//======================================================================
//
// Parameter-comment notation: the last slot of `(*batch, *spatial, N)` /
// `[*batch, *spatial, N]` is the trailing CHANNEL axis' extent. `C` here
// and `channels` in the `diag_*` entry points are two spellings of ONE
// thing -- the flow's own channel count (== ndim) -- not two shapes.
//
// Flow is weighted in JOINT mode only: there is no `*_rls` sibling, so
// `wgt` is always ONE weight map shared by every channel,
// (*batch, *spatial, 1), read once and hoisted out of the channel loop
// (`wmode::joint`, fastfields-kernels `regularisers/stencil.h`). Its
// channel extent is therefore 1, never `C` -- see fastfields-kernels#67
// for what a mismatch between map and declared geometry costs.
//
//======================================================================
//                           MEMBRANE JRLS
//======================================================================

// --- MEMBRANE+JRLS: matvec ------------------------------------------

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
          scalar_t * out,           // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_out,    // [*batch, *spatial, C] vector
    const offset_t * stride_inp,    // [*batch, *spatial, C] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 1] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    offset_t isc    = stride_inp[nall];
    reduce_t kernel[Impl::kernelsize_membrane_jrls];
    Impl::make_kernel_membrane_jrls(kernel, absolute, membrane, voxel_size);

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

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
                    osc, isc, kernel);
                ++ci;
                ++cw;
            }
        }
    });
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
          reduce_t   absolute,
          reduce_t   membrane
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    reduce_t kernel[Impl::kernelsize_membrane_jrls];
    Impl::make_kernel_membrane_jrls(kernel, absolute, membrane, voxel_size);

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

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
                    v.index.data(), size + nbatch, stride_wgt + nbatch, osc, kernel);
                ++cw;
            }
        }
    });
}

// --- MEMBRANE+JRLS: relax --------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_membrane_jrls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_sol,    // [*batch, *spatial, C] vector
    const offset_t * stride_hes,    // [*batch, *spatial, K] vector
    const offset_t * stride_grd,    // [*batch, *spatial, C] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 1] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          int        niter=1
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];

    reduce_t kernel[Impl::kernelsize_membrane_jrls];
    Impl::make_kernel_membrane_jrls(kernel, absolute, membrane, voxel_size);
    constexpr int DD = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

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
        scalar_t val[ndim], diag[ndim];
        reduce_t buf[DD ? DD : 1];
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
#               pragma unroll
                for (int d=0; d<ndim; ++d)
                    val[d] = grdp[gsc*d];

                // minus convolution
                Impl::template matvec_membrane_jrls<isub>(
                    val, solp, wgtp,
                    loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
                    static_cast<offset_t>(1), osc, kernel);

                // diagonal
                Impl::template diag_membrane_jrls<set>(
                    diag, wgtp, loc,
                    size + nbatch, stride_wgt + nbatch,
                    static_cast<offset_t>(1), kernel);

                // sol += (hes + diag) \ (grad - conv(sol))
                PosDef::relax_(
                    Strided(solp, osc),
                    StridedConst(hesp, hsc),
                    val, diag, buf, static_cast<reduce_t>(0)
                );
                ++cgrd; ++ches; ++cwgt;
            }
        }
    });
    }
    }
}

//======================================================================
//                           LAME JRLS
//======================================================================

// --- LAME+JRLS: matvec ----------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void matvec_lame_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, C) tensor
    const scalar_t * inp,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, C] vector
    const offset_t * stride_out,    // [*batch, *spatial, C] vector
    const offset_t * stride_inp,    // [*batch, *spatial, C] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 1] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t absolute,
          reduce_t membrane,
          reduce_t shears,
          reduce_t div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall = nbatch + ndim;
    offset_t osc = stride_out[nall];
    offset_t isc = stride_inp[nall];
    reduce_t kernel[Impl::kernelsize_lame_jrls];
    Impl::make_kernel_lame_jrls(kernel, absolute, membrane, shears, div, voxel_size);

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto ai = tny::as_anyrank(inp, size, stride_inp, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

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
                Impl::template matvec_lame_jrls<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), (*ci).data(), (*cw).data(),
                    v.index.data(), size + nbatch, stride_inp + nbatch, stride_wgt + nbatch,
                    osc, isc, kernel);
                ++ci;
                ++cw;
            }
        }
    });
}

// --- LAME+JRLS: diagonal ---------------------------------------------

template <
    int ndim,
    char op,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void diag_lame_jrls(
          offset_t   nbatch,
          scalar_t * out,           // (*batch, *spatial, channels) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,          // [*batch, *spatial, channels] vector
    const offset_t * stride_out,    // [*batch, *spatial, channels] vector
    const offset_t * stride_wgt,    // [*batch, *spatial, 1] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   shears,
          reduce_t   div
)
{
    using Impl = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_out[nall];
    reduce_t kernel[Impl::kernelsize_lame_jrls];
    Impl::make_kernel_lame_jrls(kernel, absolute, membrane, shears, div, voxel_size);

    auto ao = tny::as_anyrank(out, size, stride_out, static_cast<int>(nall) + 1, tny::copy_meta);
    auto aw = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall) + 1, tny::copy_meta);

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
                Impl::template diag_lame_jrls<op_apply<op, scalar_t, reduce_t> >(
                    v.cell.data(), (*cw).data(),
                    v.index.data(), size + nbatch, stride_wgt + nbatch, osc, kernel);
                ++cw;
            }
        }
    });
}

// --- LAME+JRLS: relax ------------------------------------------------

template <
    int ndim,
    typename reduce_t,
    typename scalar_t,
    typename offset_t,
    bound::type... BOUND
>
void relax_lame_jrls_(
          offset_t   nbatch,
          scalar_t * sol,           // (*batch, *spatial, C) tensor
    const scalar_t * hes,           // (*batch, *spatial, K) tensor
    const scalar_t * grd,           // (*batch, *spatial, C) tensor
    const scalar_t * wgt,           // (*batch, *spatial, 1) tensor
    const offset_t * size,         // [*batch, *spatial, C] vector
    const offset_t * stride_sol,   // [*batch, *spatial, C] vector
    const offset_t * stride_hes,   // [*batch, *spatial, K] vector
    const offset_t * stride_grd,   // [*batch, *spatial, C] vector
    const offset_t * stride_wgt,   // [*batch, *spatial, 1] vector
    const reduce_t * _voxel_size,   // [*spatial] vector
          reduce_t   absolute,
          reduce_t   membrane,
          reduce_t   shears,
          reduce_t   div,
          int        niter=1
)
{
    using Impl          = RegFlow<ndim, scalar_t, reduce_t, offset_t, BOUND...>;
    using PosDef        = posdef::utils<posdef::type::Sym, offset_t, ndim>;
    using Strided       = posdef::internal::StridedPointer<scalar_t, offset_t>;
    using StridedConst  = posdef::internal::StridedPointer<const scalar_t, offset_t>;

    // copy vectors to the stack
    reduce_t voxel_size[ndim];    fillfrom<ndim>(voxel_size, _voxel_size);
    offset_t nall   = nbatch + ndim;
    offset_t osc    = stride_sol[nall];
    offset_t hsc    = stride_hes[nall];
    offset_t gsc    = stride_grd[nall];

    reduce_t kernel[Impl::kernelsize_lame_jrls];
    Impl::make_kernel_lame_jrls(kernel, absolute, membrane, shears, div, voxel_size);
    constexpr int DD = posdef::utils<posdef::type::Sym, offset_t, ndim>::work_size;

    // Coloured sweep over dense sub-lattice views (see colour_lattice
    // above): one `subsample` per pass, so a pass touches only its own
    // colour, instead of visiting every voxel and discarding the rest. The
    // wrapped index domain is (*batch, *spatial) -- the channel axis is left
    // off, since every kernel below takes a base pointer plus its own channel
    // stride and never indexes that axis here.
    using Colour = colour_lattice<ndim, 2, false>;   // was patch2
    const offset_t kstep = 2;
    auto asol = tny::as_anyrank(sol, size, stride_sol, static_cast<int>(nall), tny::copy_meta);
    auto agrd = tny::as_anyrank(grd, size, stride_grd, static_cast<int>(nall), tny::copy_meta);
    auto ahes = tny::as_anyrank(hes, size, stride_hes, static_cast<int>(nall), tny::copy_meta);
    auto awgt = tny::as_anyrank(wgt, size, stride_wgt, static_cast<int>(nall), tny::copy_meta);
    const offset_t ncell = asol.template size_front<-ndim>();

    for (offset_t n = 0; n < pow<ndim>(2)*niter; ++n) {
    for (int j=0; j<Colour::nsub; ++j) {
        offset_t cstart[ndim];
        Colour::starts(n, j, cstart);
        const offset_t nsp  = sublattice_numel<ndim>(size + nbatch, kstep, cstart);
        const offset_t nvox = ncell * nsp;
        if (nvox <= 0) continue;
    parallel_for(0, nvox, GRAIN_SIZE, [&](long start, long end) {
        offset_t loc[ndim];
        scalar_t val[ndim], diag[ndim];
        reduce_t buf[DD ? DD : 1];
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
#               pragma unroll
                for (int d=0; d<ndim; ++d)
                    val[d] = grdp[gsc*d];

                // minus convolution
                Impl::template matvec_lame_jrls<isub>(
                    val, solp, wgtp,
                    loc, size + nbatch, stride_sol + nbatch, stride_wgt + nbatch,
                    static_cast<offset_t>(1), osc, kernel);

                // diagonal
                Impl::template diag_lame_jrls<set>(
                    diag, wgtp, loc,
                    size + nbatch, stride_wgt + nbatch,
                    static_cast<offset_t>(1), kernel);

                // sol += (hes + diag) \ (grad - conv(sol))
                PosDef::relax_(
                    Strided(solp, osc),
                    StridedConst(hesp, hsc),
                    val, diag, buf, static_cast<reduce_t>(0)
                );
                ++cgrd; ++ches; ++cwgt;
            }
        }
    });
    }
    }
}

FF_NAMESPACE_END(reg_flow)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)


#endif // FF_REGULARISERS_FLOW_CPU
