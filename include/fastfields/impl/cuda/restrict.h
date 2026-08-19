#ifndef FF_RESTRICT_CUDA
#define FF_RESTRICT_CUDA
// Teeny-based CUDA restrict (spline restriction) impl -- the device mirror of the
// CPU launcher (fastfields-cpu-impl/restrict.h). Same math, same representation:
//
//   * restriction is the exact ADJOINT of resize's prolongation, built by
//     TRANSPOSING resize's pull tap enumeration -> adjoint-exact for every
//     shift/order/scale/boundary.
//   * SEPARABLE, FLAT CSR weight tables per axis: row[d][0..nc] + (foff[d], fwt[d])
//     = the (fine-offset, signed-weight) taps landing on each coarse output index m
//     along axis d. The tables are built ONCE on the HOST (the pull weights depend
//     only on the per-axis coordinate), reused across every batch cell and voxel.
//   * OUTPUT-DRIVEN: one thread per coarse output voxel -> disjoint accumulates,
//     NO atomics (a scatter would contend). restriction ACCUMULATES into the
//     pre-zeroed `out` (the documented contract; matches the CPU path).
//
// Device port vs. the CPU version:
//   * the flat CSR buffers are cudaMemcpy'd to the device (they are FLAT arrays,
//     not nested std containers, exactly so this transfer is trivial);
//   * both tensors are wrapped as DEVICE-PASSABLE teeny anyrank carriers
//     (`as_anyrank<TNY_MAX_RANK, storage::gpu_view>(..., copy_meta)` -- the
//     shape/stride travel INLINE with the carrier, so it is trivially copyable and
//     passes into the kernel BY VALUE; no separate device copy of shape/stride);
//   * the kernel runs a grid-stride loop over the coarse voxels, peels the batch
//     cell with `peel_front_at<-D>` (device-safe, _TNY_API), and runs the SAME
//     shared `gather_sep`/`row_n` (kernels/gather.h) the CPU body uses -- so
//     "CPU works + CUDA compiles" gives real confidence they compute the same thing.
//
// The device kernel is INDEPENDENT of the spline order O and boundary B (those are
// baked into the CSR weights on the host), so the whole O x B matrix folds to a
// single device instantiation per (D, dtype, offset) -- the host `loop` stays
// templated on O/B only for the table build.
#include "fastfields/core/cuda_switch.h"
#include "fastfields/impl/kernels/pushpull/teeny.h"   // _low / _fastweight / _bound_at + gather_sep / row_n (+ teeny.h)
#include "utils.h"                    // GET_BLOCKS / CUDA_NUM_THREADS / copyToDevice / freeDevice
#include <cmath>
#include <vector>
#include <cstdint>

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(restrict)

// Device-side view of the flat per-axis CSR tables. A trivially-copyable POD
// passed into the kernel BY VALUE; the pointers address device memory.
template <int D, typename offset_t, typename reduce_t>
struct csr_dev {
    const offset_t * row [D];   // [osize[d] + 1] CSR row offsets
    const offset_t * foff[D];   // [tot[d]] fine offsets (i * fine_stride[d])
    const reduce_t * fwt [D];   // [tot[d]] signed weights
    offset_t         osize[D];  // coarse extents (for the spatial multi-index decode)
};

// One coarse output voxel per (grid-stride) iteration. Independent of O/B: the
// weights/offsets were baked into the CSR on the host. `CO`/`CI` are the (device-
// passable) anyrank carrier types for the coarse out / fine inp tensors.
template <int D, typename reduce_t, typename scalar_t, typename offset_t,
          class CO, class CI>
CUGLOB void
_restrict_kernel(CO ao, CI ai, csr_dev<D, offset_t, reduce_t> csr,
                 offset_t nvox, offset_t nsp)
{
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox;
         i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b  = (nsp > 0) ? i / nsp : offset_t(0);   // batch cell
        auto oc = ao.template peel_front_at<-D>(b);               // coarse out volume
        auto ic = ai.template peel_front_at<-D>(b);               // fine inp volume

        offset_t sp = i - b * nsp;
        offset_t m[D];                                            // coarse spatial multi-index (row-major)
        for (int d = D - 1; d >= 0; --d) { m[d] = sp % csr.osize[d]; sp /= csr.osize[d]; }

        // view each axis's CSR slice as a runtime-count row and run the shared
        // separable gather (gather.h) -- the same recursion resize/pull use.
        row_n<reduce_t, offset_t> rows[D];
        for (int d = 0; d < D; ++d) {
            const offset_t lo = csr.row[d][m[d]], hi = csr.row[d][m[d] + 1];
            rows[d].w = csr.fwt[d] + lo; rows[d].o = csr.foff[d] + lo; rows[d].count = hi - lo;
        }
        const reduce_t acc = gather_sep<D, row_n<reduce_t, offset_t>,
                                        scalar_t, offset_t, reduce_t>(ic.data(), rows);

        if      constexpr (D == 1) oc(m[0])             += static_cast<scalar_t>(acc);
        else if constexpr (D == 2) oc(m[0], m[1])       += static_cast<scalar_t>(acc);
        else                       oc(m[0], m[1], m[2]) += static_cast<scalar_t>(acc);
    }
}

// Host launcher. Builds the flat CSR tables on the host (identical to the CPU
// impl), copies them to the device, wraps out/inp as device-passable anyrank
// carriers, launches the grid-stride kernel over the coarse output voxels on
// `stream`, then synchronises and frees the temporaries.
//
// `scale` has length D; the shape/stride vectors have length nall = D + nbatch
// (host arrays). `out`/`inp` are DEVICE pointers. Order O and boundary B are
// compile-time (B == bound_t::Dynamic routes the runtime `bound` through
// _bound_at); reduce_t is the accumulation type (double).
template <
    int D, int O, bound_t B,
    typename reduce_t, typename scalar_t, typename offset_t
>
CUHOST void loop(
          offset_t   nbatch,
          scalar_t * out,             // (*batch, *out_spatial) coarse tensor (pre-zeroed; accumulated)
    const scalar_t * inp,             // (*batch, *inp_spatial) fine tensor
          reduce_t   shift,
    const reduce_t * _scale,          // [D] per-axis scaling (fine / coarse)
    const offset_t * size_out,        // [nbatch + D] output shape
    const offset_t * size_inp,        // [nbatch + D] input shape
    const offset_t * stride_out,      // [nbatch + D] output strides
    const offset_t * stride_inp,      // [nbatch + D] input strides
          bound_t    bound = bound_t::Dynamic,   // runtime bound (B == Dynamic route)
          int        stream = 0
)
{
    reduce_t scale[D];
    offset_t osize[D], isize[D], fstride[D];
    for (int d = 0; d < D; ++d) {
        scale[d]   = _scale[d];
        osize[d]   = size_out[nbatch + d];   // coarse
        isize[d]   = size_inp[nbatch + d];   // fine
        fstride[d] = stride_inp[nbatch + d];
    }

    // Per-axis FLAT CSR transpose tables (built ON THE HOST). A tap on output m
    // along d is the exact transpose of pushpull::_make_axis (same _low, tap nb,
    // s/index, weight), so restrict is adjoint-exact. Two passes: count taps per
    // output index, prefix-sum into row offsets, then scatter (fine offset,
    // signed weight) into place.
    std::vector<offset_t> row[D], foff[D];
    std::vector<reduce_t> fwt[D];
    for (int d = 0; d < D; ++d) {
        const offset_t nc = osize[d], nf = isize[d];
        row[d].assign(static_cast<size_t>(nc) + 1, 0);
        for (offset_t i = 0; i < nf; ++i) {
            const reduce_t c   = (static_cast<reduce_t>(i) + shift) / scale[d] - shift;
            const offset_t low = pushpull::_low<O, reduce_t, offset_t>(c);
            for (int k = 0; k <= O; ++k) {
                int8_t s; offset_t ix;
                pushpull::_bound_at<B>(bound, low + static_cast<offset_t>(k), nc, s, ix);
                if (s != 0) row[d][static_cast<size_t>(ix) + 1] += 1;
            }
        }
        for (offset_t m = 0; m < nc; ++m) row[d][m + 1] += row[d][m];   // -> CSR offsets
        const offset_t tot = row[d][nc];
        foff[d].resize(static_cast<size_t>(tot));
        fwt[d].resize(static_cast<size_t>(tot));
        std::vector<offset_t> cur(row[d].begin(), row[d].begin() + nc);  // write cursor per row
        for (offset_t i = 0; i < nf; ++i) {
            const reduce_t c   = (static_cast<reduce_t>(i) + shift) / scale[d] - shift;
            const offset_t low = pushpull::_low<O, reduce_t, offset_t>(c);
            for (int k = 0; k <= O; ++k) {
                const offset_t nb = low + static_cast<offset_t>(k);
                int8_t s; offset_t ix;
                pushpull::_bound_at<B>(bound, nb, nc, s, ix);
                if (s == 0) continue;
                const reduce_t w = static_cast<reduce_t>(s)
                    * pushpull::_fastweight<O>(
                        static_cast<reduce_t>(std::fabs(c - static_cast<reduce_t>(nb))));
                const offset_t e = cur[static_cast<size_t>(ix)]++;
                foff[d][static_cast<size_t>(e)] = i * fstride[d];
                fwt[d][static_cast<size_t>(e)]  = w;
            }
        }
    }

    offset_t nsp = 1;
    for (int d = 0; d < D; ++d) nsp *= osize[d];

    // Device-passable teeny carriers: shape/stride are COPIED inline (copy_meta),
    // so the carrier passes into the kernel by value; the DATA pointers live in
    // device memory (storage::gpu_view).
    const int rank = static_cast<int>(nbatch) + D;
    auto ao = tny::as_anyrank<TNY_MAX_RANK, tny::storage::gpu_view>(
                  out, size_out, stride_out, rank, tny::copy_meta);
    auto ai = tny::as_anyrank<TNY_MAX_RANK, tny::storage::gpu_view>(
                  inp, size_inp, stride_inp, rank, tny::copy_meta);

    const offset_t ncell = ao.template size_front<-D>();   // #batch cells
    const offset_t nvox  = ncell * nsp;                    // total coarse output voxels

    // Copy the flat CSR tables to the device.
    offset_t * d_row [D]; offset_t * d_foff[D]; reduce_t * d_fwt[D];
    for (int d = 0; d < D; ++d) { d_row[d] = nullptr; d_foff[d] = nullptr; d_fwt[d] = nullptr; }

    try
    {
        for (int d = 0; d < D; ++d) {
            d_row[d]  = copyToDevice(row[d].data(),  static_cast<offset_t>(row[d].size()));
            d_foff[d] = copyToDevice(foff[d].data(), static_cast<offset_t>(foff[d].size()));
            d_fwt[d]  = copyToDevice(fwt[d].data(),  static_cast<offset_t>(fwt[d].size()));
        }

        csr_dev<D, offset_t, reduce_t> csr;
        for (int d = 0; d < D; ++d) {
            csr.row[d] = d_row[d]; csr.foff[d] = d_foff[d]; csr.fwt[d] = d_fwt[d];
            csr.osize[d] = osize[d];
        }

        cudaStream_t s = reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream));
        const int blocks  = GET_BLOCKS(nvox);
        const int threads = CUDA_NUM_THREADS;

        _restrict_kernel<D, reduce_t, scalar_t, offset_t>
            <<<blocks, threads, 0, s>>>(ao, ai, csr, nvox, nsp);

        // The kernel reads the CSR device buffers, so wait before freeing them.
        cudaStreamSynchronize(s);
    }
    catch (...)
    {
        for (int d = 0; d < D; ++d) freeDevice(d_row[d], d_foff[d], d_fwt[d]);
        throw;
    }
    for (int d = 0; d < D; ++d) freeDevice(d_row[d], d_foff[d], d_fwt[d]);
}

FF_NAMESPACE_END(restrict)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_RESTRICT_CUDA
