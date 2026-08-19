#ifndef FF_RESIZE_CUDA
#define FF_RESIZE_CUDA
// Teeny-based CUDA resize (spline resampling) impl -- the device mirror of the
// CPU launcher (fastfields-cpu-impl/resize.h). Same math, same representation:
//
//   * resize is a PULL whose sampling coordinate is an AFFINE map of the output
//     voxel index: loc[d] = scale[d] * (idx[d] + shift) - shift.
//   * SEPARABLE per-axis weight tables (grid regularity): because loc[d] depends
//     only on the d-th output coordinate, the per-axis neighbourhood (the O+1
//     sign-folded weights + strided offsets, pushpull::_make_axis) has only
//     size_out[d] distinct values along axis d. Precompute those tables ONCE on
//     the HOST (batch-invariant: the folded offsets are relative to each cell's
//     base pointer, the input spatial extents/strides do not vary per batch),
//     reused across every batch cell and voxel.
//   * OUTPUT-DRIVEN: one thread per output voxel -> disjoint writes, NO atomics.
//     resize has no channel axis (every leading dim is batch, only the last D are
//     spatial) and no FOV test (boundary is the kernel's job -> always in-bounds).
//
// Device port vs. the CPU version:
//   * the per-axis tables are NESTED std::vector<std::vector<_axis>> on the CPU;
//     a nested vector cannot be cudaMemcpy'd, so here each axis's table is
//     FLATTENED into two contiguous host buffers -- wt[d] (O+1 weights per output
//     index, row-major by output index) and off[d] (the matching strided offsets)
//     -- exactly as restrict flattens its CSR, then cudaMemcpy'd to the device;
//   * both tensors are wrapped as DEVICE-PASSABLE teeny anyrank carriers
//     (`as_anyrank<TNY_MAX_RANK, storage::gpu_view>(..., copy_meta)` -- the
//     shape/stride travel INLINE with the carrier, so it is trivially copyable and
//     passes into the kernel BY VALUE; no separate device copy of shape/stride);
//   * the kernel runs a grid-stride loop over the output voxels, peels the batch
//     cell with `peel_front_at<-D>` (device-safe, _TNY_API), views the O+1 taps of
//     each axis as a compile-time-count `row_k<O+1>`, and runs the SAME shared
//     `gather_sep` (kernels/gather.h) the CPU body uses -- so "CPU works + CUDA
//     compiles" gives real confidence they compute the same thing.
//
// The device kernel depends on the spline order O (the O+1 tap count folds into
// `row_k`) but is INDEPENDENT of the boundary B (baked into the weights on the
// host), so the O x B matrix folds to one device instantiation per (D, O, dtype,
// offset) -- the host `loop` stays templated on O/B for the table build.
#include "fastfields/core/cuda_switch.h"
#include "fastfields/impl/kernels/pushpull/teeny.h"   // _axis / _make_axis + gather_sep / row_k (+ teeny.h)
#include "utils.h"                    // GET_BLOCKS / CUDA_NUM_THREADS / copyToDevice / freeDevice
#include <cmath>
#include <vector>
#include <cstdint>

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(resize)

// Device-side view of the flat per-axis weight/offset tables. A trivially-
// copyable POD passed into the kernel BY VALUE; the pointers address device
// memory. Each table holds O+1 taps per output index (row-major by output index).
template <int D, int O, typename offset_t, typename reduce_t>
struct axis_dev {
    const reduce_t * wt [D];   // [osize[d] * (O+1)] sign-folded weights
    const offset_t * off[D];   // [osize[d] * (O+1)] strided input offsets
    offset_t         osize[D]; // output extents (for the spatial multi-index decode)
};

// One output voxel per (grid-stride) iteration. Depends on O (the O+1 tap count),
// but independent of B: the weights/offsets were baked into the tables on the
// host. `CO`/`CI` are the (device-passable) anyrank carrier types for the out /
// inp tensors.
template <int D, int O, typename reduce_t, typename scalar_t, typename offset_t,
          class CO, class CI>
CUGLOB void
_resize_kernel(CO ao, CI ai, axis_dev<D, O, offset_t, reduce_t> tab,
               offset_t nvox, offset_t nsp)
{
    for (offset_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nvox;
         i += static_cast<offset_t>(gridDim.x) * blockDim.x)
    {
        const offset_t b  = (nsp > 0) ? i / nsp : offset_t(0);   // batch cell
        auto oc = ao.template peel_front_at<-D>(b);               // out spatial volume
        auto ic = ai.template peel_front_at<-D>(b);               // inp spatial volume

        offset_t sp = i - b * nsp;
        offset_t m[D];                                            // out spatial multi-index (row-major)
        for (int d = D - 1; d >= 0; --d) { m[d] = sp % tab.osize[d]; sp /= tab.osize[d]; }

        // view each axis's precomputed neighbourhood as a compile-time-count row
        // and run the shared separable gather (gather.h) -- O+1 taps per axis unroll.
        row_k<reduce_t, offset_t, O + 1> rows[D];
        for (int d = 0; d < D; ++d) {
            const offset_t base = m[d] * static_cast<offset_t>(O + 1);
            rows[d].w = tab.wt[d]  + base;
            rows[d].o = tab.off[d] + base;
        }
        const reduce_t val = gather_sep<D, row_k<reduce_t, offset_t, O + 1>,
                                        scalar_t, offset_t, reduce_t>(ic.data(), rows);

        if      constexpr (D == 1) oc(m[0])              = static_cast<scalar_t>(val);
        else if constexpr (D == 2) oc(m[0], m[1])        = static_cast<scalar_t>(val);
        else                       oc(m[0], m[1], m[2])  = static_cast<scalar_t>(val);
    }
}

// Host launcher. Builds the per-axis neighbourhood tables on the host (identical
// math to the CPU impl), FLATTENS each into two contiguous buffers, copies them to
// the device, wraps out/inp as device-passable anyrank carriers, launches the
// grid-stride kernel over the output voxels on `stream`, then synchronises and
// frees the temporaries.
//
// `scale` has length D; the shape/stride vectors have length nall = D + nbatch
// (host arrays). `out`/`inp` are DEVICE pointers. Order O and boundary B are
// compile-time (B == bound_t::Dynamic routes the runtime `bound` through
// _make_axis / _bound_at); reduce_t is the accumulation type (double).
template <
    int D, int O, bound_t B,
    typename reduce_t, typename scalar_t, typename offset_t
>
CUHOST void loop(
          offset_t   nbatch,
          scalar_t * out,             // (*batch, *out_spatial) tensor
    const scalar_t * inp,             // (*batch, *inp_spatial) tensor
          reduce_t   shift,
    const reduce_t * _scale,          // [D] per-axis scaling
    const offset_t * size_out,        // [nbatch + D] output shape
    const offset_t * size_inp,        // [nbatch + D] input shape
    const offset_t * stride_out,      // [nbatch + D] output strides
    const offset_t * stride_inp,      // [nbatch + D] input strides
          bound_t    bound = bound_t::Dynamic,   // runtime bound (B == Dynamic route)
          int        stream = 0
)
{
    reduce_t scale[D];
    offset_t osize[D], iext[D], istr[D];
    for (int d = 0; d < D; ++d) {
        scale[d] = _scale[d];
        osize[d] = size_out[nbatch + d];
        iext[d]  = size_inp[nbatch + d];    // input spatial extent (batch-invariant)
        istr[d]  = stride_inp[nbatch + d];
    }

    // Per-axis neighbourhood tables, built once (grid regularity), FLATTENED into
    // contiguous buffers so they can be cudaMemcpy'd (a nested std::vector cannot).
    // Layout per axis d: O+1 consecutive taps per output index, row-major by index.
    using axis_t = pushpull::_axis<reduce_t, offset_t>;
    const offset_t K = static_cast<offset_t>(O) + 1;
    std::vector<reduce_t> hwt [D];
    std::vector<offset_t> hoff[D];
    for (int d = 0; d < D; ++d) {
        hwt[d].resize(static_cast<size_t>(osize[d]) * static_cast<size_t>(K));
        hoff[d].resize(static_cast<size_t>(osize[d]) * static_cast<size_t>(K));
        for (offset_t idx = 0; idx < osize[d]; ++idx) {
            const reduce_t coord = scale[d] * (static_cast<reduce_t>(idx) + shift) - shift;
            const axis_t a =
                pushpull::_make_axis<O, B, reduce_t, offset_t>(coord, iext[d], istr[d], bound);
            const size_t base = static_cast<size_t>(idx) * static_cast<size_t>(K);
            for (int k = 0; k <= O; ++k) {
                hwt[d][base + static_cast<size_t>(k)]  = a.w[k];
                hoff[d][base + static_cast<size_t>(k)] = a.off[k];
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
    const offset_t nvox  = ncell * nsp;                    // total output voxels

    // Copy the flat per-axis tables to the device.
    reduce_t * d_wt[D]; offset_t * d_off[D];
    for (int d = 0; d < D; ++d) { d_wt[d] = nullptr; d_off[d] = nullptr; }

    try
    {
        for (int d = 0; d < D; ++d) {
            d_wt[d]  = copyToDevice(hwt[d].data(),  static_cast<offset_t>(hwt[d].size()));
            d_off[d] = copyToDevice(hoff[d].data(), static_cast<offset_t>(hoff[d].size()));
        }

        axis_dev<D, O, offset_t, reduce_t> tab;
        for (int d = 0; d < D; ++d) {
            tab.wt[d] = d_wt[d]; tab.off[d] = d_off[d];
            tab.osize[d] = osize[d];
        }

        cudaStream_t s = reinterpret_cast<cudaStream_t>(static_cast<std::intptr_t>(stream));
        const int blocks  = GET_BLOCKS(nvox);
        const int threads = CUDA_NUM_THREADS;

        _resize_kernel<D, O, reduce_t, scalar_t, offset_t>
            <<<blocks, threads, 0, s>>>(ao, ai, tab, nvox, nsp);

        // The kernel reads the device tables, so wait before freeing them.
        cudaStreamSynchronize(s);
    }
    catch (...)
    {
        for (int d = 0; d < D; ++d) freeDevice(d_wt[d], d_off[d]);
        throw;
    }
    for (int d = 0; d < D; ++d) freeDevice(d_wt[d], d_off[d]);
}

FF_NAMESPACE_END(resize)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)

#endif // FF_RESIZE_CUDA
