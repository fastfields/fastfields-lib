/* Utilities to convert contiguous linear indices to
 * - sub-indices, and/or
 * - strided linear indices
 *
 * All functions assume that the input index is a linear index into a
 * contiguous volume of size `size` (with `ndim` dimensions), with a
 * Fortran layout. That is
 *      index = i + size[0] * (j + size[1] * (k + ...))
 *
 * The `stride` vector is then used to build the corresponding
 * strided linear index
 *      strided = i * stride[0] + j * stride[1] + k * stride[2] + ...
 *
 * Functions postfixed 1d/2d/3d only convert the leading "batch"
 * dimensions to a strided index, and return the remaining sub-indices
 * (x, y, z) in placeholders.
 *
 * Each function comes in two flavors:
 * - a dynamically sized version, where `ndim` is a function argument
 * - a statically sized version, where `ndim` is a template parameter
 */
#ifndef FF_BATCH
#define FF_BATCH
#include "fastfields/core/cuda_switch.h"
#include "utils.h"

FF_NAMESPACE_BEGIN(FF)

template <typename offset_t>
inline CUDEV
offset_t index2offset(
          offset_t   index,
          int        _ndim,
    const offset_t * size,
    const offset_t * stride
)
{
    offset_t ndim = static_cast<offset_t>(_ndim);
    offset_t new_index  = 0, new_index1;
    offset_t cur_stride = 1, nxt_stride = 1;
    for (offset_t i = 0; i < ndim; ++i) {
        new_index1 = index;
        nxt_stride = cur_stride * size[i];
        new_index1 = index % nxt_stride;
        new_index1 = new_index1 / cur_stride;
        cur_stride = nxt_stride;
        new_index += new_index1 * stride[i];
    }
    return new_index;
}

template <int _ndim, typename offset_t>
inline CUDEV
offset_t index2offset(
          offset_t   index,
    const offset_t * size,
    const offset_t * stride
)
{
    static constexpr offset_t ndim = static_cast<offset_t>(_ndim);
    offset_t new_index  = 0, new_index1;
    offset_t cur_stride = 1, nxt_stride = 1;
#   pragma unroll
    for (offset_t i = 0; i < ndim; ++i) {
        new_index1 = index;
        nxt_stride = cur_stride * size[i];
        new_index1 = index % nxt_stride;
        new_index1 = new_index1 / cur_stride;
        cur_stride = nxt_stride;
        new_index += new_index1 * stride[i];
    }
    return new_index;
}

template <typename offset_t>
inline CUDEV
offset_t index2offset_nd(
          offset_t   index,
          int        _nall,
    const offset_t * size,
    const offset_t * stride,
          offset_t * x,
          int        _ndim
)
{
    offset_t nall = static_cast<offset_t>(_nall);
    offset_t ndim = static_cast<offset_t>(_ndim);
    offset_t new_index  = 0, new_index1;
    offset_t cur_stride = 1, nxt_stride = 1;
    for (offset_t i = 0; i < nall; ++i) {
        new_index1 = index;
        if (i < nall-1) {
            nxt_stride = cur_stride * size[i];
            new_index1 = index % nxt_stride;
        }
        new_index1 = new_index1 / cur_stride;
        cur_stride = nxt_stride;
        if (i < nall-ndim)
            new_index += new_index1 * stride[i];
        else
            x[i-(nall-ndim)] = new_index1;
    }
    return new_index;
}


template <int _ndim, int _nall, typename offset_t>
inline CUDEV
offset_t index2offset_nd(
          offset_t   index,
    const offset_t * size,
    const offset_t * stride,
          offset_t * x
)
{
    static constexpr offset_t ndim = static_cast<offset_t>(_ndim);
    static constexpr offset_t nall = static_cast<offset_t>(_nall);
    offset_t new_index  = 0, new_index1;
    offset_t cur_stride = 1, nxt_stride = 1;
#   pragma unroll
    for (offset_t i = 0; i < nall; ++i) {
        new_index1 = index;
        if (i < nall-1)  {
            nxt_stride = cur_stride * size[i];
            new_index1 = index % nxt_stride;
        }
        new_index1 = new_index1 / cur_stride;
        cur_stride = nxt_stride;
        if (i < nall-ndim)
            new_index += new_index1 * stride[i];
        else
            x[i-(nall-ndim)] = new_index1;
    }
    return new_index;
}

// This version build the entire offset + extract the last N sub-indices
// It differs from `index2offset_nd<n,ndim>`, which only build the batch
// offset when it extracts sub-indices.
//
// This should be called index2offset_nd_v2.
template <int _ndim, int _nall, typename offset_t>
inline CUDEV
offset_t index2offset_v2(
          offset_t   index,
    const offset_t * size,
    const offset_t * stride,
          offset_t * x = nullptr
)
{
    static constexpr offset_t ndim = static_cast<offset_t>(_ndim);
    static constexpr offset_t nall = static_cast<offset_t>(_nall);
    offset_t new_index  = 0, new_index1;
    offset_t cur_stride = 1, nxt_stride = 1;
#   pragma unroll
    for (offset_t i = 0; i < nall; ++i) {
        new_index1 = index;
        if (i < nall-1)  {
            nxt_stride = cur_stride * size[i];
            new_index1 = index % nxt_stride;
        }
        new_index1 = new_index1 / cur_stride;
        cur_stride = nxt_stride;
        if (i >= nall-ndim)
            x[i-(nall-ndim)] = new_index1;
        new_index += new_index1 * stride[i];
    }
    return new_index;
}

template <int ndim, typename offset_t>
inline CUDEV
offset_t index2offset_v2(
    offset_t index,
    offset_t nall,
    const offset_t * size,
    const offset_t * stride,
    offset_t * x = nullptr)
{
    offset_t new_index = 0, new_index1;
    offset_t current_stride = 1, next_stride = 1;
    for (int i = 0; i < nall; ++i) {
        new_index1 = index;
        if (i < nall-1)  {
            next_stride = current_stride * size[i];
            new_index1 = index % next_stride;
        }
        new_index1 = new_index1 / current_stride;
        current_stride = next_stride;
        if (i >= nall-ndim)
            x[i-(nall-ndim)] = new_index1;
        new_index += new_index1 * stride[i];
    }
    return new_index;
}

template <int _ndim, typename offset_t>
inline CUDEV
offset_t sub2offset(
    const offset_t * sub,
    const offset_t * stride
)
{
    static constexpr offset_t ndim = static_cast<offset_t>(_ndim);
    offset_t offset = 0;
#   pragma unroll
    for (offset_t d=0; d < ndim; ++d)
        offset += sub[d] * stride[d];
    return offset;
}

template <typename offset_t>
inline CUDEV
offset_t sub2offset(offset_t ndim, const offset_t * sub, const offset_t * stride)
{
    offset_t offset = 0;
    for (int d=0; d < ndim; ++d)
        offset += sub[d] * stride[d];
    return offset;
}

template <int _nall, typename offset_t>
inline CUDEV
void index2sub(
          offset_t   index,
    const offset_t * size,
          offset_t * x
)
{
    static constexpr offset_t nall = static_cast<offset_t>(_nall);
    offset_t new_index1;
    offset_t cur_stride = 1, nxt_stride = 1;
#   pragma unroll
    for (offset_t i = 0; i < nall; ++i) {
        new_index1 = index;
        if (i < nall-1)
        {
            nxt_stride = cur_stride * size[i];
            new_index1 = index % nxt_stride;
        }
        new_index1 = new_index1 / cur_stride;
        cur_stride = nxt_stride;
        x[i] = new_index1;
    }
}

template <typename offset_t>
inline CUDEV
void index2sub(
          offset_t   nall,
          offset_t   index,
    const offset_t * size,
          offset_t * x
)
{
    offset_t new_index1;
    offset_t current_stride = 1, next_stride = 1;
    for (int i = 0; i < nall; ++i) {
        new_index1 = index;
        if (i < nall-1)
        {
            next_stride = current_stride * size[i];
            new_index1 = index % next_stride;
        }
        new_index1 = new_index1 / current_stride;
        current_stride = next_stride;
        x[i] = new_index1;
    }
}

FF_NAMESPACE_END(FF)

#endif // FF_BATCH
