#pragma once
#include <fastfields/core/cuda_switch.h>
#include <fastfields/impl/kernels/distance.h>
#include <fastfields/core/batch.h>
#include <fastfields/core/utils.h>
#include "utils.h"
#include <cstdint>
#include <memory>       // std::unique_ptr
#include <type_traits>  // std::is_trivially_copyable

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_mesh)

/***********************************************************************
 *                         CPU PRECOMPUTATION
 ***********************************************************************/

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
FF_CUHOST inline void
build_tree(
          // (2M) array of *constructed* `Node` objects. It used to be a raw
          // `void*` byte buffer that was `reinterpret_cast` here; `Node` is a
          // polymorphic type, so casting raw storage to it and writing through
          // the result is UB (no `Node` was ever constructed, so no vptr was
          // ever set). The caller now owns a real `Node[]`.
          typename MeshDist<ndim, scalar_t, index_t, offset_t>::Node * tree,
          index_t  * _faces           ,  // (M, D) tensor -> All faces (face = D vertex indices)
    const scalar_t * _vertices        ,  // (N, D) tensor -> All vertices
          offset_t   nb_faces         ,  // M
          offset_t   nb_vertices      ,  // N
    const offset_t * stride_faces     ,  // [M, D] list -> Strides of `faces`
    const offset_t * stride_vertices  )  // [N, D] list -> Strides of `vertices`
{
    using Klass      = MeshDist<ndim, scalar_t, index_t, offset_t>;
    using FaceList   = StridedPointList<ndim, index_t, offset_t>;
    using VertexList = ConstStridedPointList<ndim, scalar_t, offset_t>;

    auto  faces      = FaceList     (_faces,    stride_faces    [0], stride_faces   [1]);
    auto  vertices   = VertexList   (_vertices, stride_vertices [0], stride_vertices[1]);

    index_t node_id = 0;
    Klass::build_tree(tree, node_id, -1, 0, nb_faces, faces, vertices);
}

/***********************************************************************
 *                      POD MIRROR OF THE BVH NODE
 ***********************************************************************/

// `MeshDist<...>::Node` cannot be shipped to the device with `cudaMemcpy`.
//
// A `Node` holds two `BoundingSphere`s, each of which holds a
// `StaticPoint<D, scalar_t>` centre. `StaticPoint` derives from `PointMixin`
// and `ConstPointMixin`, whose `operator[]` is pure-virtual in
// `AnyPoint`/`AnyConstPoint` -- so every `StaticPoint` carries vtable
// pointers and `Node` is *not* trivially copyable. Measured at the currently
// pinned kernels (3e38c85):
//
//     D=2 float/int   : sizeof(Node) = 104, is_trivially_copyable = false
//     D=3 float/int   : sizeof(Node) = 120, is_trivially_copyable = false
//     D=2 double/long : sizeof(Node) = 128, is_trivially_copyable = false
//     D=3 double/long : sizeof(Node) = 144, is_trivially_copyable = false
//
// Byte-copying such an object is UB even between two host processes; copying
// it to a *device* address space, where the host vtables do not exist at all,
// is also concretely wrong -- the device would dispatch through host
// pointers. So we do not transfer `Node` at all. Instead the host BVH builder
// (`MeshDist::build_tree`, unchanged and still `FF_CUHOST`) writes real `Node`
// objects into host memory, and a translation pass flattens them into this
// plain struct, which *is* trivially copyable and is what actually crosses
// the H2D boundary.
//
// NB: this mirror has no counterpart in `fastfields-cpu-impl`, and that
// divergence from the "structural mirror" rule is deliberate -- the CPU path
// walks the very `Node` objects it built, in one address space, so it needs
// no transfer representation. `fastfields-cuda-impl#43` lists collapsing the
// two representations (building the BVH in POD form on both paths) as a
// larger follow-up.
template <int D, typename scalar_t, typename index_t>
struct DeviceNode
{
    // `BoundingSphere bv_left`  -> {center_left,  radius_left}
    // `BoundingSphere bv_right` -> {center_right, radius_right}
    scalar_t center_left [D];
    scalar_t radius_left;
    scalar_t center_right[D];
    scalar_t radius_right;
    // A leaf is encoded exactly as in `Node`: `left == -1`, and `right` is
    // then the face index rather than a node index.
    index_t  left;
    index_t  right;
    index_t  parent;
};

// Flatten the host-built `Node` tree into the POD mirror that is uploaded.
// One pass over the `2M` nodes; `nodes` must be a fully constructed array.
template <int ndim, typename scalar_t, typename index_t, typename offset_t>
FF_CUHOST inline void
flatten_tree(
          DeviceNode<ndim, scalar_t, index_t> * out,
    const typename MeshDist<ndim, scalar_t, index_t, offset_t>::Node * nodes,
          offset_t nb_nodes)
{
    using Node = typename MeshDist<ndim, scalar_t, index_t, offset_t>::Node;
    using Pod  = DeviceNode<ndim, scalar_t, index_t>;

    // The whole point of the mirror: what we memcpy must be memcpy-able.
    static_assert(std::is_trivially_copyable<Pod>::value,
                  "DeviceNode must be trivially copyable -- it is memcpy'd to "
                  "the device");
    static_assert(std::is_standard_layout<Pod>::value,
                  "DeviceNode must be standard layout");
    // Deliberately NOT asserted: `!is_trivially_copyable<Node>`. If `Node` ever
    // becomes trivially copyable (i.e. `StaticPoint` loses its virtuals) this
    // mirror turns redundant, not wrong -- so that should read as "you can
    // simplify now", not as a build break in this repo.

    for (offset_t n = 0; n < nb_nodes; ++n)
    {
        const Node & node = nodes[n];
        Pod        & pod  = out[n];

        for (int d = 0; d < ndim; ++d)
        {
            pod.center_left [d] = node.bv_left .center[d];
            pod.center_right[d] = node.bv_right.center[d];
        }
        pod.radius_left  = node.bv_left .radius;
        pod.radius_right = node.bv_right.radius;
        pod.left         = node.left;
        pod.right        = node.right;
        pod.parent       = node.parent;
    }
}

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
FF_CUHOST inline void
build_normals(
          scalar_t * _normfaces          ,  // (M, D) tensor
          scalar_t * _normvertices       ,  // (N, D) tensor
          scalar_t * _normedges          ,  // (M, D, D) tensor
    const index_t  * _faces              ,  // (M, D) tensor -> All faces (face = D vertex indices)
    const scalar_t * _vertices           ,  // (N, D) tensor -> All vertices
          offset_t   nb_faces            ,  // M
          offset_t   nb_vertices         ,  // N
    const offset_t * stride_normfaces    ,  // [M, D] list
    const offset_t * stride_normvertices ,  // [N, D] list
    const offset_t * stride_normedges    ,  // [M, D, D] list
    const offset_t * stride_faces        ,  // [M, D] list -> Strides of `faces`
    const offset_t * stride_vertices     )  // [N, D] list -> Strides of `vertices`
{
    using Klass          = MeshDistUtil<ndim, scalar_t, offset_t>;
    using FaceList       = ConstStridedPointListSized<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointListSized<ndim, scalar_t, offset_t>;
    using NormalList     = StridedPointList<ndim, scalar_t, offset_t>;
    using EdgeNormalList = StridedPointArray<ndim, scalar_t, offset_t, ndim>;
    using EdgeStride     = StaticPoint<3, offset_t>;

    // In 2D -> no edges
    auto _stride_normedges = EdgeStride();
    if (stride_normedges)
        _stride_normedges.copy_(ConstRefPoint<3, offset_t>(stride_normedges));

    auto faces        = FaceList        (_faces,        stride_faces        [0], stride_faces       [1], nb_faces);
    auto vertices     = VertexList      (_vertices,     stride_vertices     [0], stride_vertices    [1], nb_vertices);
    auto normfaces    = NormalList      (_normfaces,    stride_normfaces    [0], stride_normfaces   [1]);
    auto normvertices = NormalList      (_normvertices, stride_normvertices [0], stride_normvertices[1]);
    auto normedges    = EdgeNormalList  (_normedges,    _stride_normedges);

    Klass::build_normals(normfaces, normvertices, normedges, faces, vertices);
}

/***********************************************************************
 *                         COPY UTILITY
 ***********************************************************************/

// Host-only: it returns an `allocHost` buffer, and its only caller
// (`copyTensorToContiguous`) is itself FF_CUHOST. Declaring it FF_CUHOSTDEV made
// nvcc emit `warning #20014-D: calling a __host__ function from a
// __host__ __device__ function is not allowed` for every instantiation.
template <typename offset_t>
FF_CUHOST inline offset_t * contiguousStrides(const offset_t * size, int ndim)
{
    offset_t * stride = allocHost<offset_t>(ndim);
    stride[ndim-1] = static_cast<offset_t>(1);
    for (int d=ndim-2; d >= 0; --d)
        stride[d] = size[d+1] * stride[d+1];
    return stride;
}

template <typename scalar_t, typename offset_t>
FF_CUGLOB inline void
copy_tensor_kernel(
          offset_t   ndim,
          scalar_t * out,
    const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_out,
    const offset_t * stride_inp)
{
    offset_t index  = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;

    offset_t numel = prod(size, ndim);
    for (offset_t i=index; i < numel; i += stride)
    {
        offset_t offset_inp = index2offset(i, ndim, size, stride_inp);
        offset_t offset_out = index2offset(i, ndim, size, stride_out);
        out[offset_out] = inp[offset_inp];
    }
}

template <typename scalar_t, typename offset_t>
FF_CUHOST inline
scalar_t * copyTensorToContiguous(
          offset_t     ndim,
    const scalar_t   * inp,
    const offset_t   * size,
    const offset_t   * stride_inp,
          cudaStream_t stream = 0)
{
    scalar_t * out             = nullptr;
    offset_t * size_copy       = nullptr;
    offset_t * stride_out      = nullptr;
    offset_t * stride_out_copy = nullptr;
    offset_t * stride_inp_copy = nullptr;
    try
    {
        offset_t numel = prod(size, ndim);
        // Compute contiguous output strides
        stride_out = contiguousStrides(size, ndim);
        // Allocate device memory and copy metadata
        out             = allocDevice<scalar_t>(numel);
        size_copy       = copyToDeviceAsync(size, ndim, stream);
        stride_out_copy = copyToDeviceAsync(stride_out, ndim, stream);
        stride_inp_copy = copyToDeviceAsync(stride_inp, ndim, stream);
        // Copy data
        copy_tensor_kernel<scalar_t, offset_t>
            <<<GET_BLOCKS(numel), CUDA_NUM_THREADS, 0, stream>>>
            (ndim, out, inp, size_copy, stride_out_copy, stride_inp_copy);
    }
    catch (const std::exception & e)
    {
        freeHost(stride_out);
        freeDevice(out, size_copy, stride_out_copy, stride_inp_copy);
        throw e;
    }
    freeHost(stride_out);
    freeDevice(size_copy, stride_out_copy, stride_inp_copy);

    return out;
}

// `faces_out` is written through, so it cannot be `const`: it was declared
// `const index_t *` and the body then initialised an `index_t *` from it, which
// is ill-formed. Nothing instantiated this kernel -- `copy_faces` has no
// caller, `sdt` copies via `copyTensorToContiguous` -- so the error was never
// diagnosed. `tests/compile_probe_mesh.cu` now instantiates it.
template <int ndim,         // Number of spatial dimensions
          typename index_t, // Index (faces) data type
          typename offset_t // Index/Stride data type
          >
FF_CUGLOB inline void copy_faces_kernel(
    offset_t nb_faces,         // Number of faces (M)
    index_t * faces_out,       // (M, D) output (contiguous) tensor of faces
    const index_t * faces_inp, // (M, D) input tensor of faces
    offset_t stride_elem,      // Input stride between elements
    offset_t stride_channel)   // Input stride between channels
{
    offset_t index  = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;

    for (offset_t i=index; i < nb_faces; i += stride)
    {
        const index_t * ptr_inp = faces_inp + i * stride_elem;
              index_t * ptr_out = faces_out + i * static_cast<offset_t>(ndim);
        for (offset_t c = 0; c < ndim; ++c, ++ptr_out, ptr_inp += stride_channel)
        {
            *ptr_out = *ptr_inp;
        }
    }
}

template <
    int      ndim,          // Number of spatial dimensions
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
FF_CUHOST inline
index_t * copy_faces(
          offset_t     nb_faces   ,
    const index_t    * faces      ,
    const offset_t   * stride     ,
          cudaStream_t stream = 0 )
{
    offset_t stride0 = stride[0], stride1 = stride[1];
    index_t * faces_out = allocDevice<index_t>(nb_faces * ndim);
    copy_faces_kernel<ndim, index_t, offset_t>
        <<<GET_BLOCKS(nb_faces), CUDA_NUM_THREADS, 0, stream>>>
        (nb_faces, faces_out, faces, stride0, stride1);
    return faces_out;
}

/***********************************************************************
 *              DEVICE-SIDE BVH TRAVERSAL OVER THE POD MIRROR
 ***********************************************************************/

// The kernels-level traversal (`MeshDist::query_dist_loop`) and its two
// wrappers (`_unsigned_dist`, `unsigned_dist`, `signed_dist`) all take a
// `const Node *`, so once the device no longer receives `Node` objects they
// can no longer be used from a CUDA kernel. The three functions below are
// structural mirrors of those, reading `DeviceNode` instead.
//
// Everything that is *not* tree access -- `Utils::sqdist_unsigned`,
// `Utils::sign`, `MeshDist::get_nearest_vertex` -- is still called straight
// out of `fastfields-kernels`; only the node field accesses change:
//
//     node->bv_left.center[d]  ->  node->center_left[d]
//     node->bv_left.radius     ->  node->radius_left
//     node->left/right/parent  ->  unchanged
//
// Keep them in step with `kernels/distance/mesh.h` when that traversal
// changes: nothing in the build enforces it. Collapsing the duplication (by
// giving the kernels traversal a node-type template parameter, or by building
// the BVH in POD form on both paths) is the follow-up noted in
// fastfields-cuda-impl#43.

// Mirror of `MeshDist::query_dist_loop`.
template <
    int      ndim,
    typename scalar_t,
    typename index_t,
    typename offset_t,
    typename NearestPoint, typename Point, typename Vertices, typename Faces,
    typename Trace
>
FF_CUDEV inline void
query_dist_loop_pod(
          index_t       & nearest_face,
          scalar_t      & nearest_dist,
          NearestEntity & nearest_entity,
          NearestPoint  & nearest_point,
    const Point         & point,
    const Vertices      & vertices,
    const Faces         & faces,
    const DeviceNode<ndim, scalar_t, index_t> * nodes,
          Trace         & trace)
{
    using Klass             = MeshDist<ndim, scalar_t, index_t, offset_t>;
    using Utils             = typename Klass::Utils;
    using StaticPointScalar = typename Klass::StaticPointScalar;
    using Node              = DeviceNode<ndim, scalar_t, index_t>;

    const Node * node = nullptr;
    index_t node_id = 0;
    index_t level = 0;

    // we use the first four bits of trace (at each level) as follow:
    // higher bit -> lower bit
    // [current_side_is_right, current_side_is_left, right_was_visited, left_was_visited]

    auto fast_dist = [&](const scalar_t * center, scalar_t radius)
    {
        scalar_t dist = 0;
        for (int d=0; d<ndim; ++d)
        {
            scalar_t tmp = point[d] - center[d];
            dist += tmp*tmp;
        }
        dist = sqrt(dist);
        dist -= radius;
        return dist;
    };

    while (1)
    {
        if (node_id < 0)
            break;

        node = nodes + node_id;

        if (node->left == -1)
        {
            // leaf

            const offset_t face_id = static_cast<offset_t>(node->right);
            auto face = faces[face_id];
            auto facevertices = StaticPointList<ndim, ndim, scalar_t>();
            for (offset_t d=0; d<ndim; ++d)
                facevertices[d].copy_(vertices[face[d]]);

            NearestEntity       maybe_entity;
            StaticPointScalar   maybe_point;
            scalar_t maybe_dist = Utils::sqdist_unsigned(
                maybe_entity, maybe_point, point, facevertices);

            if (maybe_dist < nearest_dist * nearest_dist)
            {
                nearest_face   = face_id;
                nearest_dist   = sqrt(maybe_dist);
                nearest_entity = maybe_entity;
                nearest_point.copy_(maybe_point);
            }
            level -= 1;
            trace[level] |= trace[level] >> 2; // set current side as "visited"
            node_id = node->parent;
        }
        else if ((trace[level] & 1) && (trace[level] & 2))
        {
            // left and right already visited
            if (level == 0)
                break;
            for (index_t l=level; l<trace.size; ++l)
                trace[l] = 0;
            node_id = node->parent;
            level -= 1;
            trace[level] |= trace[level] >> 2; // set current side as "visited"
        }
        else if(trace[level] & 2)
        {
            // already visited right, now visit left
            const scalar_t d_left = fast_dist(node->center_left, node->radius_left);

            if (d_left < nearest_dist)
            {
                trace[level] &= 3;      // erase side bits
                trace[level] |= 1 << 2; // set current side as "left"
                node_id = node->left;
                level += 1;
                continue;
            }
            else
            {
                trace[level] |= 1;     // set left as "visited"
            }
        }
        else if(trace[level] & 1)
        {
            // already visited left, now visit right
            const scalar_t d_right = fast_dist(node->center_right, node->radius_right);

            if (d_right < nearest_dist)
            {
                trace[level] &= 3;      // erase side bits
                trace[level] |= 2 << 2; // set current side as "right"
                node_id = node->right;
                level += 1;
                continue;
            }
            else
            {
                trace[level] |= 2;     // set right as "visited"
            }
        }
        else
        {
            // none visited - decide whether to start with left or right
            scalar_t d_left  = fast_dist(node->center_left,  node->radius_left);
            scalar_t d_right = fast_dist(node->center_right, node->radius_right);

            if (d_left < d_right)
            {
                if (d_left < nearest_dist)
                {
                    trace[level] &= 3;      // erase side bits
                    trace[level] |= 1 << 2; // set current side as "left"
                    node_id = node->left;
                    level += 1;
                    continue;
                }
                else
                {
                    trace[level] |= 1;     // set left as "visited"
                    if (d_right < nearest_dist)
                    {
                        trace[level] &= 3;      // erase side bits
                        trace[level] |= 2 << 2; // set current side as "right"
                        node_id = node->right;
                        level += 1;
                        continue;
                    }
                    else
                    {
                        trace[level] |= 2;     // set right as "visited"
                    }
                }
            }
            else
            {
                if (d_right < nearest_dist)
                {
                    trace[level] &= 3;      // erase side bits
                    trace[level] |= 2 << 2; // set current side as "right"
                    node_id = node->right;
                    level += 1;
                    continue;
                }
                else
                {
                    trace[level] |= 2;     // set right as "visited"
                    if (d_left < nearest_dist)
                    {
                        trace[level] &= 3;      // erase side bits
                        trace[level] |= 1 << 2; // set current side as "left"
                        node_id = node->left;
                        level += 1;
                        continue;
                    }
                    else
                    {
                        trace[level] |= 1;     // set left as "visited"
                    }
                }
            }
        }
    }
}

// Mirror of `MeshDist::unsigned_dist` (which wraps `_unsigned_dist`).
template <
    int      ndim,
    typename scalar_t,
    typename index_t,
    typename offset_t,
    typename Point, typename Vertices, typename Faces, typename Trace
>
FF_CUDEV inline scalar_t
unsigned_dist_pod(
    const Point    & point,
    const Vertices & vertices,
    const Faces    & faces,
    const DeviceNode<ndim, scalar_t, index_t> * tree,
          Trace    & treetrace,
          index_t  * nearest_vertex = nullptr,
          index_t  * _nearest_face = nullptr,
          NearestEntity * _nearest_entity = nullptr,
          typename MeshDist<ndim, scalar_t, index_t, offset_t>::StaticPointScalar
                 * _nearest_point = nullptr)
{
    using Klass             = MeshDist<ndim, scalar_t, index_t, offset_t>;
    using StaticPointScalar = typename Klass::StaticPointScalar;

    index_t             nearest_face;
    StaticPointScalar   nearest_point;
    NearestEntity       nearest_entity;
    scalar_t            nearest_dist = static_cast<scalar_t>(1./0.);

    query_dist_loop_pod<ndim, scalar_t, index_t, offset_t>(
        nearest_face, nearest_dist, nearest_entity, nearest_point,
        point, vertices, faces, tree, treetrace);

    // get index of vertex nearest to the projection (must run BEFORE the
    // return: fastfields exposes nearest_vertex on the unsigned path).
    if (nearest_vertex)
        *nearest_vertex = Klass::get_nearest_vertex(
            faces[nearest_face], nearest_point, vertices);

    if (_nearest_face)   *_nearest_face   = nearest_face;
    if (_nearest_entity) *_nearest_entity = nearest_entity;
    if (_nearest_point)  _nearest_point->copy_(nearest_point);

    return nearest_dist;
}

// Mirror of `MeshDist::signed_dist`.
template <
    int      ndim,
    typename scalar_t,
    typename index_t,
    typename offset_t,
    typename Point, typename Vertices, typename Faces,
    typename NormFaces, typename NormEdges, typename NormVertices,
    typename Trace
>
FF_CUDEV inline scalar_t
signed_dist_pod(
    const Point         & point,
    const Vertices      & vertices,
    const Faces         & faces,
    const DeviceNode<ndim, scalar_t, index_t> * tree,
          Trace         & treetrace,
    const NormFaces     & normfaces,
    const NormEdges     & normedges,
    const NormVertices  & normvertices,
          index_t       * nearest_vertex = nullptr)
{
    using Klass             = MeshDist<ndim, scalar_t, index_t, offset_t>;
    using Utils             = typename Klass::Utils;
    using StaticPointScalar = typename Klass::StaticPointScalar;

    index_t             nearest_face;
    StaticPointScalar   nearest_point;
    NearestEntity       nearest_entity;

    // compute unsigned distance and return index of nearest triangle
    scalar_t dist = unsigned_dist_pod<ndim, scalar_t, index_t, offset_t>(
        point, vertices, faces, tree, treetrace,
        nearest_vertex, &nearest_face, &nearest_entity, &nearest_point);

    // load normals into a compact array
    auto face    = faces[nearest_face];
    auto normals = StaticPointList<ndim+1+(ndim == 3 ? ndim : 0), ndim, scalar_t>();
    normals[0].copy_(normfaces[nearest_face]);
    if (ndim == 3)
    {
        auto normedge = normedges[nearest_face];
        for (offset_t d=0; d<ndim; ++d)
        {
            normals[1+d].copy_(normvertices[face[d]]);
            normals[1+ndim+d].copy_(normedge[d]);
        }
    }
    else
    {
        for (offset_t d=0; d<ndim; ++d)
            normals[1+d].copy_(normvertices[face[d]]);
    }

    // compute sign from dot product <ray, normal>
    scalar_t sign = Utils::sign(point, nearest_point, normals, nearest_entity);

    return dist * sign;
}

/***********************************************************************
 *                           CUDA KERNELS
 ***********************************************************************/

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
FF_CUGLOB inline void sdt_kernel(
          offset_t   nbatch             ,  // Number of batch dimensions in coord
          scalar_t * dist               ,  // (*batch) tensor -> Output placeholder for distance
          index_t  * nearest_vertex     ,  // (*batch) tensor -> Output placeholder for index of nearest vertex
    const scalar_t * coord              ,  // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices          ,  // (N, D) tensor -> All vertices
    const index_t  * _faces             ,  // (M, D) tensor -> All faces (face = D vertex indices)
    // (2M) array of POD nodes -- see `DeviceNode`. This used to be a
    // `const void *` that was `reinterpret_cast` to `MeshDist::Node`, a
    // polymorphic host type whose bytes are meaningless on the device.
    const DeviceNode<ndim, scalar_t, index_t> * tree,
          void     * _treetrace         ,  // (nb_levels * nb_threads) trace buffer
          offset_t   treesize           ,
    const scalar_t * _normfaces         ,  // (M, D) tensor
    const scalar_t * _normvertices      ,  // (N, D) tensor
    const scalar_t * _normedges         ,  // (M, D, D) tensor
    const offset_t * size               ,  // [*batch] list -> Size of `dist`
    const offset_t * stride_dist        ,  // [*batch] list -> Strides of `dist`
    const offset_t * stride_nearest     ,  // [*batch] list -> Strides of `nearest_vertex`
    const offset_t * stride_coord       ,  // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices    ,  // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces       ,  // [M, D] list -> Strides of `faces`
    const offset_t * stride_normfaces   ,  // [M, D] list
    const offset_t * stride_normvertices,  // [N, D] list
    const offset_t * stride_normedges   )  // [M, D, D] list
{
    offset_t index  = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;

    using FaceList       = ConstStridedPointList<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using NormalList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using EdgeNormalList = ConstStridedPointArray<ndim, scalar_t, offset_t, ndim>;
    using EdgeStride     = StaticPoint<3, offset_t>;
    using RefPoint       = ConstStridedPoint<ndim, scalar_t, offset_t>;
    using ClonedPoint    = StaticPoint<ndim, scalar_t>;

    // Each lane owns a `treesize`-long trace, interleaved across the launched
    // threads: lane `index` reads/writes trace element j at `[index + j*stride]`.
    // `_treetrace` is `void*`, so it must be cast before the arithmetic --
    // pointer arithmetic on `void*` is not valid C++.
    auto treetrace = SizedStridedPointer<char, offset_t>(
        static_cast<char*>(_treetrace) + index, stride, treesize);

    // In 2D -> no edges
    auto _stride_normedges = EdgeStride();
    if (stride_normedges)
        _stride_normedges.copy_(ConstRefPoint<3, offset_t>(stride_normedges));

    auto faces        = FaceList      (_faces,        stride_faces        [0], stride_faces        [1]);
    auto vertices     = VertexList    (_vertices,     stride_vertices     [0], stride_vertices     [1]);
    auto normfaces    = NormalList    (_normfaces,    stride_normfaces    [0], stride_normfaces    [1]);
    auto normvertices = NormalList    (_normvertices, stride_normvertices [0], stride_normvertices [1]);
    auto normedges    = EdgeNormalList(_normedges,    _stride_normedges);

    offset_t numel = prod(size, nbatch);
    for (offset_t i=index; i < numel; i += stride)
    {
        // if (i != 0) return;

        for (offset_t j=0; j<treetrace.size; ++j)
            treetrace[j] = static_cast<char>(0);

        offset_t offset_coord = index2offset(i, nbatch, size, stride_coord);
        offset_t offset_dist  = index2offset(i, nbatch, size, stride_dist);
        offset_t offset_nearest = 0;
        if (nearest_vertex)
            offset_nearest  = index2offset(i, nbatch, size, stride_nearest);
        ClonedPoint point;
        point.copy_(RefPoint(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = signed_dist_pod<ndim, scalar_t, index_t, offset_t>(
            point,
            vertices,
            faces,
            tree,
            treetrace,
            normfaces,
            normedges,
            normvertices,
            nearest_vertex + offset_nearest
        );
    }
}

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
FF_CUGLOB inline void sdt_naive_kernel(
          offset_t   nbatch             ,  // Number of batch dimensions in coord
          scalar_t * dist               ,  // (*batch) tensor -> Output placeholder for distance
          index_t  * nearest_vertex     ,  // (*batch) tensor -> Output placeholder for index of nearest vertex
    const scalar_t * coord              ,  // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices          ,  // (N, D) tensor -> All vertices
    const index_t  * _faces             ,  // (M, D) tensor -> All faces (face = D vertex indices)
    const scalar_t * _normfaces         ,  // (M, D) tensor
    const scalar_t * _normvertices      ,  // (N, D) tensor
    const scalar_t * _normedges         ,  // (M, D, D) tensor
    const offset_t * size               ,  // [*batch] list -> Size of `dist`
          offset_t   nb_faces           ,
    const offset_t * stride_dist        ,  // [*batch] list -> Strides of `dist`
    const offset_t * stride_nearest     ,  // [*batch] list -> Strides of `nearest_vertex`
    const offset_t * stride_coord       ,  // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices    ,  // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces       ,  // [M, D] list -> Strides of `faces`
    const offset_t * stride_normfaces   ,  // [M, D] list
    const offset_t * stride_normvertices,  // [N, D] list
    const offset_t * stride_normedges   )  // [M, D, D] list
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;

    using Klass          = MeshDist<ndim, scalar_t, index_t, offset_t>;
    using FaceList       = ConstStridedPointListSized<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using NormalList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using EdgeNormalList = ConstStridedPointArray<ndim, scalar_t, offset_t, ndim>;
    using EdgeStride     = StaticPoint<3, offset_t>;

    // In 2D -> no edges
    auto _stride_normedges = EdgeStride();
    if (stride_normedges)
        _stride_normedges.copy_(ConstRefPoint<3, offset_t>(stride_normedges));

    auto faces        = FaceList       (_faces,        stride_faces        [0], stride_faces        [1], nb_faces);
    auto vertices     = VertexList     (_vertices,     stride_vertices     [0], stride_vertices     [1]);
    auto normfaces    = NormalList     (_normfaces,    stride_normfaces    [0], stride_normfaces    [1]);
    auto normvertices = NormalList     (_normvertices, stride_normvertices [0], stride_normvertices [1]);
    auto normedges    = EdgeNormalList (_normedges,    _stride_normedges);

    offset_t numel = prod(size, nbatch);
    for (offset_t i=index; i < numel; i += stride)
    {
        offset_t offset_coord = index2offset(i, nbatch, size, stride_coord);
        offset_t offset_dist  = index2offset(i, nbatch, size, stride_dist);
        // `nearest_vertex` is optional, exactly as in `sdt_kernel` and in
        // cpu-impl's `build_sdt_naive`: when it is null the stride array is
        // null too and must not be read.
        offset_t offset_nearest = 0;
        if (nearest_vertex)
            offset_nearest  = index2offset(i, nbatch, size, stride_nearest);

        StaticPoint<ndim, scalar_t> point(
            ConstStridedPoint<ndim, scalar_t, offset_t>(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = Klass::signed_dist_naive(
            point,
            vertices,
            faces,
            normfaces,
            normedges,
            normvertices,
            nearest_vertex + offset_nearest
        );
    }
}

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
FF_CUGLOB inline void udt_kernel(
          offset_t   nbatch         ,  // Number of batch dimensions in coord
          scalar_t * dist           ,  // (*batch) tensor -> Output placeholder for distance
    const scalar_t * coord          ,  // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices      ,  // (N, D) tensor -> All vertices
    const index_t  * _faces         ,  // (M, D) tensor -> All faces (face = D vertex indices)
    const DeviceNode<ndim, scalar_t, index_t> * tree,  // (2M) POD node array
          void     * _treetrace     ,  // (nb_levels * nb_threads) trace buffer
          offset_t   treesize       ,
    const offset_t * size           ,  // [*batch] list -> Size of `dist`
    const offset_t * stride_dist    ,  // [*batch] list -> Strides of `dist`
    const offset_t * stride_coord   ,  // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,  // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces   )  // [M, D] list -> Strides of `faces`
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;

    using FaceList       = ConstStridedPointList<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;

    // `_treetrace` is `void*`, so it must be cast before the arithmetic --
    // pointer arithmetic on `void*` is not valid C++. (Same fix as
    // `sdt_kernel`; this kernel is never instantiated today, so the ill-formed
    // expression was never diagnosed.)
    auto treetrace = SizedStridedPointer<char, offset_t>(
        static_cast<char*>(_treetrace) + index, stride, treesize);

    auto faces        = FaceList  (_faces,    stride_faces   [0], stride_faces   [1]);
    auto vertices     = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);

    offset_t numel = prod(size, nbatch);
    for (offset_t i=index; index < numel; index += stride, i=index)
    {
        for (offset_t j=0; j<treetrace.size; ++j)
            treetrace[j] = static_cast<char>(0);

        offset_t offset_coord = index2offset(i, nbatch, size, stride_coord);
        offset_t offset_dist  = index2offset(i, nbatch, size, stride_dist);

        StaticPoint<ndim, scalar_t> point(
            ConstStridedPoint<ndim, scalar_t, offset_t>(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = unsigned_dist_pod<ndim, scalar_t, index_t, offset_t>(
            point,
            vertices,
            faces,
            tree,
            treetrace
        );
    }
}

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
FF_CUGLOB inline void udt_naive_kernel(
          offset_t   nbatch         ,  // Number of batch dimensions in coord
          scalar_t * dist           ,  // (*batch) tensor -> Output placeholder for distance
    const scalar_t * coord          ,  // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices      ,  // (N, D) tensor -> All vertices
    const index_t  * _faces         ,  // (M, D) tensor -> All faces (face = D vertex indices)
    const offset_t * size           ,  // [*batch] list -> Size of `dist`
          offset_t   nb_faces       ,
    const offset_t * stride_dist    ,  // [*batch] list -> Strides of `dist`
    const offset_t * stride_coord   ,  // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,  // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces   )  // [M, D] list -> Strides of `faces`
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;

    using Klass          = MeshDist<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointListSized<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;

    auto faces        = FaceList  (_faces,    stride_faces   [0], stride_faces   [1], nb_faces);
    auto vertices     = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);

    offset_t numel = prod(size, nbatch);
    for (offset_t i=index; i < numel; i += stride)
    {
        offset_t offset_coord = index2offset(i, nbatch, size, stride_coord);
        offset_t offset_dist  = index2offset(i, nbatch, size, stride_dist);

        StaticPoint<ndim, scalar_t> point(
            ConstStridedPoint<ndim, scalar_t, offset_t>(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = Klass::unsigned_dist_naive(
            point,
            vertices,
            faces
        );
    }
}

/***********************************************************************
 *             Templated entrypoints that launch the CUDA kernels
 ***********************************************************************/

// Signed mesh distance transform: builds the BVH and the vertex/face/edge
// normals on the host, uploads a POD mirror of the tree, and launches
// `sdt_kernel` over the batch elements on `stream`.
//
// NB: the mesh counts are taken as (nb_faces, nb_vertices), matching cpu-impl's
// `sdt` and the `dt` dispatcher below. They used to be declared the other way
// round here, and only here -- both are `offset_t`, so a call site passing them
// in `dt`'s order would have compiled silently and built the mesh from the
// wrong counts.
template <
    int      _ndim,         // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
FF_CUHOST inline void
sdt(
          offset_t   nbatch,                // Number of batch dimensions in coord
          scalar_t * dist,                  // (*batch)     tensor  -> Output placeholder for distance
          index_t  * nearest_vertex,        // (*batch)     tensor  -> Output placeholder for index of nearest vertex
    const scalar_t * coord,                 // (*batch, D)  tensor  -> Coordinates at which to evaluate distance
    const scalar_t * vertices,              // (N, D)       tensor  -> All vertices
    const index_t  * faces,                 // (M, D)       tensor  -> All faces (face = D vertex indices)
    const offset_t * size,                  // [*batch]     list    -> Size of `dist`
          offset_t   nb_faces,              // M                    -> Number of faces
          offset_t   nb_vertices,           // N                    -> Number of vertices
    const offset_t * stride_dist,           // [*batch]     list    -> Strides of `dist`
    const offset_t * stride_nearest,        // [*batch]     list    -> Strides of `nearest_vertex`
    const offset_t * stride_coord,          // [*batch, D]  list    -> Strides of `coord`
    const offset_t * stride_vertices,       // [N, D]       list    -> Strides of `vertices`
    const offset_t * stride_faces   ,       // [M, D]       list    -> Strides of `faces`
          intptr_t   stream = 0           // CUDA stream (0 == default stream)
)
{
    static const offset_t ndim = static_cast<offset_t>(_ndim);
    const cudaStream_t s = (cudaStream_t)(std::intptr_t)stream;

    using Node = typename MeshDist<_ndim, scalar_t, index_t, offset_t>::Node;
    using Pod  = DeviceNode<_ndim, scalar_t, index_t>;

    index_t  * faces_device      = nullptr;
    scalar_t * verts_device      = nullptr;
    Pod      * tree_device       = nullptr;
    char     * treetrace_device  = nullptr;
    index_t  * faces_host        = nullptr;
    scalar_t * verts_host        = nullptr;
    // Owning, exception-safe: neither of these is `cudaMallocHost` memory, so
    // they must not go through `freeHost` (`cudaFreeHost`) below.
    //  * `tree_host` is an array of *constructed* `Node` objects -- the host
    //    BVH builder writes real objects into it, so raw storage will not do.
    //  * `tree_pod_host` is the POD mirror that is actually uploaded. It is
    //    deliberately pageable rather than pinned: `copyToDeviceAsync` only
    //    synchronises the stream on its converting path, so a pinned source
    //    freed right after the enqueue would race with the in-flight DMA,
    //    whereas a pageable source is staged by the driver before the call
    //    returns.
    std::unique_ptr<Node[]> tree_host;
    std::unique_ptr<Pod[]>  tree_pod_host;
    scalar_t * normfaces_host    = nullptr;
    scalar_t * normverts_host    = nullptr;
    scalar_t * normedges_host    = nullptr;
    scalar_t * normfaces_device  = nullptr;
    scalar_t * normverts_device  = nullptr;
    scalar_t * normedges_device  = nullptr;
    offset_t * stride_vec_device = nullptr;
    offset_t * stride_mat_device = nullptr;
    offset_t * stride_dist_device    = nullptr;
    offset_t * stride_nearest_device = nullptr;
    offset_t * stride_coord_device   = nullptr;
    offset_t * size_device           = nullptr;

    try
    {

        // Make a copy of the faces on the device with contiguous layout
        offset_t   size_faces    [2] = {nb_faces,    ndim};
        offset_t   size_verts    [2] = {nb_vertices, ndim};
        offset_t   stride_vec    [2] = {ndim, 1};
        // `stride_mat` strides the (M, D, D) edge-normal tensor, so it holds
        // three values -- it was declared `[2]` with a 3-element initializer.
        offset_t   stride_mat    [3] = {ndim*ndim, ndim, 1};
        // `faces` and `vertices` are always *2-D* tensors -- (M, D) and (N, D)
        // -- so the rank handed to `copyTensorToContiguous` is 2, never `ndim`.
        // Passing `ndim` made the 3-D case (the main triangular-mesh case) read
        // `size_faces[2]` / `stride[2]`, one past these 2-element stack arrays:
        // `prod`, `contiguousStrides` and the metadata `copyToDeviceAsync` all
        // ran over rank 3, so `numel` -- and hence the `cudaMalloc` size --
        // picked up a garbage third factor.
        static const offset_t rank = static_cast<offset_t>(2);
        // NB: the following assign the cleanup variables declared above.
        // Redeclaring them here shadowed those, so both cleanup paths saw
        // nullptr and the real allocations leaked.
        //
        // The *input* strides must be the caller's real ones. `stride_vec` is
        // the contiguous layout, i.e. the layout of the copy's destination --
        // handing it in as `stride_inp` assumed the input was already
        // contiguous and silently misread non-contiguous DLTensors.
        faces_device =
            copyTensorToContiguous(rank, faces, size_faces, stride_faces, s);
        verts_device = copyTensorToContiguous(rank, vertices, size_verts,
                                              stride_vertices, s);

        // Copy to host
        faces_host = copyToHost(faces_device, nb_faces    * ndim);
        verts_host = copyToHost(verts_device, nb_vertices * ndim);

        // Allocate tree.
        //
        // `nb_levels` bounds the *depth* of the BVH and is still needed below
        // to size the per-lane traversal trace (`treesize`).
        //
        // The node *count*, however, is exact: `MeshDist::build_tree` splits
        // at the median and emits one node per recursion
        // (`nodes(1) = 1`, `nodes(M) = 1 + nodes(M/2) + nodes(M-M/2)`), so a
        // mesh of M faces yields exactly 2M-1 nodes.
        //
        // It used to be estimated as a *full* binary tree of `nb_levels`
        // levels -- `2^nb_levels - 1` ~= 8M nodes, ~4x the 2M-1 actually
        // emitted -- whose per-node size was a hand-rolled POD formula
        //
        //     nb_features = sizeof(scalar_t)*2*(ndim+1) + sizeof(index_t)*3
        //
        // that under-counts `sizeof(Node)` by 1.6-2.9x depending on the dtype
        // pair (Node is polymorphic: two vptr-carrying `StaticPoint` centres,
        // plus its own vptr). Two compensating errors -- the node over-count
        // hid the per-node shortfall, which is why this never overflowed.
        // Both are gone: `2*nb_faces` real `Node` objects, sized by the
        // compiler. See fastfields-kernels#75 / fastfields-cuda-impl#43.
        offset_t nb_levels = static_cast<offset_t>(ceil(log2(static_cast<scalar_t>(nb_faces)))) + 3;
        offset_t nb_nodes  = 2 * nb_faces;
        tree_host.reset(new Node[nb_nodes]);

        // Build tree
        build_tree<_ndim>(
            tree_host.get(),
            faces_host,
            verts_host,
            nb_faces,
            nb_vertices,
            stride_vec,
            stride_vec
        );

        // Allocate normals
        normfaces_host = allocHost<scalar_t>(nb_faces    * ndim);
        normverts_host = allocHost<scalar_t>(nb_vertices * ndim);
        normedges_host = allocHost<scalar_t>(nb_faces    * ndim * ndim);
        // The vertex normals are ACCUMULATED into, not assigned:
        // `MeshDistUtil::build_normals` does `normvertices[v].add_(normal)`
        // once per incident face and normalises at the end. cpu-impl allocates
        // them with `new scalar_t[...]()` for exactly this reason; `allocHost`
        // is `cudaMallocHost`, which does not zero, so without this loop the
        // pseudonormals accumulate on top of whatever the pinned allocation
        // happened to contain. That does not fail loudly -- it perturbs the
        // vertex/edge pseudonormals, i.e. the *sign* of the returned distance
        // near vertices and edges. Face normals (`copy_`) and edge normals
        // (built in a local map, then `copy_`) are assigned, so only this one
        // buffer needs zeroing.
        for (offset_t i = 0; i < nb_vertices * ndim; ++i)
            normverts_host[i] = static_cast<scalar_t>(0);

        // Build normals
        build_normals<ndim>(
            normfaces_host,
            normverts_host,
            normedges_host,
            faces_host,
            verts_host,
            nb_faces,
            nb_vertices,
            stride_vec,
            stride_vec,
            stride_mat,
            stride_vec,
            stride_vec
        );

        // Flatten the polymorphic host `Node` tree into its POD mirror.
        //
        // The `Node` array itself is never uploaded: it is not trivially
        // copyable (its `StaticPoint` centres carry vtable pointers), so
        // memcpy'ing it to the device would hand the device host vtable
        // pointers to dispatch through. `DeviceNode` holds the same fields as
        // plain data, and `sdt_kernel` walks that. See `DeviceNode` above and
        // fastfields-cuda-impl#43.
        tree_pod_host.reset(new Pod[nb_nodes]);
        flatten_tree<_ndim, scalar_t, index_t, offset_t>(
            tree_pod_host.get(), tree_host.get(), nb_nodes);

        // Copy to device
        faces_device            = copyToDeviceAsync(faces_host,     nb_faces    * ndim, s, faces_device);
        tree_device             = copyToDeviceAsync(tree_pod_host.get(), nb_nodes, s);
        normfaces_device        = copyToDeviceAsync(normfaces_host, nb_faces    * ndim, s);
        normverts_device        = copyToDeviceAsync(normverts_host, nb_vertices * ndim, s);
        normedges_device        = copyToDeviceAsync(normedges_host, nb_faces    * ndim * ndim, s);
        stride_vec_device       = copyToDeviceAsync(stride_vec,     2, s);
        stride_mat_device       = copyToDeviceAsync(stride_mat,     3, s);
        stride_dist_device      = copyToDeviceAsync(stride_dist,    nbatch, s);
        // `nearest_vertex` is an *optional* output: `ff::dt_mesh` leaves it
        // null unless the caller asked for nearest-vertex indices, and cuda-lib
        // then passes a null `stride_nearest` too. `sdt_kernel` already guards
        // on `nearest_vertex` before touching `stride_nearest`, but the upload
        // has to be guarded as well -- `copyToDeviceAsync(nullptr, ...)` would
        // reach `cudaMemcpyAsync` with a null source, which fails with
        // `cudaErrorInvalidValue` and is rethrown here as `std::bad_alloc`.
        // That is the *default* call pattern, so it would have failed for
        // nearly every caller.
        stride_nearest_device = nullptr;
        if (nearest_vertex)
            stride_nearest_device =
                copyToDeviceAsync(stride_nearest, nbatch, s);
        stride_coord_device     = copyToDeviceAsync(stride_coord,   nbatch + 1, s);
        size_device             = copyToDeviceAsync(size,           nbatch, s);

        // The grid must cover the number of points being evaluated, i.e. the
        // element count of the batch -- not `nbatch`, which is the batch *rank*
        // (a handful of dimensions). Mirrors distance_euclidean.h.
        offset_t numel      = prod(size, nbatch);
        int      num_blocks = GET_BLOCKS(numel);

        // Recursion is unavailable on device, so `sdt_kernel` walks the tree
        // iteratively with an explicit trace of one byte per tree level. Every
        // lane needs its own trace and they are interleaved across the launched
        // lanes, so the buffer scales with the thread count -- the same shape as
        // the euclidean scratch buffer.
        offset_t treesize   = nb_levels;
        offset_t stride_buf = static_cast<offset_t>(num_blocks) * CUDA_NUM_THREADS;
        treetrace_device    = allocDevice<char>(stride_buf * treesize);

        // Compute SDT
        sdt_kernel<ndim, scalar_t, index_t, offset_t>
            <<<num_blocks, CUDA_NUM_THREADS, 0, s>>>
            (
                nbatch,
                dist,
                nearest_vertex,
                coord,
                verts_device,
                faces_device,
                tree_device,
                treetrace_device,
                treesize,
                normfaces_device,
                normverts_device,
                normedges_device,
                size_device,
                stride_dist_device,
                stride_nearest_device,
                stride_coord_device,
                stride_vec_device,
                stride_vec_device,
                stride_vec_device,
                stride_vec_device,
                stride_mat_device
            );
    }
    catch (const std::exception & e)
    {
        freeDevice(
            faces_device,
            verts_device,
            tree_device,
            treetrace_device,
            normfaces_device,
            normverts_device,
            normedges_device,
            stride_vec_device,
            stride_mat_device,
            stride_dist_device,
            stride_nearest_device,
            stride_coord_device,
            size_device
        );
        // NB: `tree_host` / `tree_pod_host` are `unique_ptr`s (plain `new[]`,
        // not `cudaMallocHost`) and free themselves on the way out.
        freeHost(
            faces_host,
            verts_host,
            normfaces_host,
            normverts_host,
            normedges_host
        );
        throw e;
    }

    freeDevice(
        faces_device,
        verts_device,
        tree_device,
        treetrace_device,
        normfaces_device,
        normverts_device,
        normedges_device,
        stride_vec_device,
        stride_mat_device,
        stride_dist_device,
        stride_nearest_device,
        stride_coord_device,
        size_device
    );
    freeHost(
        faces_host,
        verts_host,
        normfaces_host,
        normverts_host,
        normedges_host
    );
}

// Naive signed mesh distance transform: no acceleration structure. Structural
// mirror of cpu-impl's `sdt_naive` -- it builds the vertex/face/edge normals on
// the host, uploads them, and launches `sdt_naive_kernel`, which brute-forces
// every face for every point.
//
// This is the reference the tree-accelerated `sdt` above is meant to be
// validated against on real hardware (fastfields-lib#5), so it deliberately
// shares as little machinery with it as possible: no BVH, no POD mirror, no
// per-lane traversal trace. What the two do share is the host-side normal
// construction, which is the same `fastfields-kernels` builder both backends
// use.
//
// Two differences from `sdt` above, both inherited from cpu-impl:
//   * no `build_tree`, so the faces are never reordered and the contiguous
//     device copy uploaded once at the top stays valid for the launch (`sdt`
//     has to re-upload `faces_host` after the in-place BVH sort);
//   * `nb_levels` / `treesize` / the trace buffer do not exist here at all.
template <
    int      _ndim,         // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
FF_CUHOST inline void
sdt_naive(
          offset_t   nbatch,                // Number of batch dimensions in coord
          scalar_t * dist,                  // (*batch)     tensor  -> Output placeholder for distance
          index_t  * nearest_vertex,        // (*batch)     tensor  -> Output placeholder for index of nearest vertex
    const scalar_t * coord,                 // (*batch, D)  tensor  -> Coordinates at which to evaluate distance
    const scalar_t * vertices,              // (N, D)       tensor  -> All vertices
    const index_t  * faces,                 // (M, D)       tensor  -> All faces (face = D vertex indices)
    const offset_t * size,                  // [*batch]     list    -> Size of `dist`
          offset_t   nb_faces,              // M                    -> Number of faces
          offset_t   nb_vertices,           // N                    -> Number of vertices
    const offset_t * stride_dist,           // [*batch]     list    -> Strides of `dist`
    const offset_t * stride_nearest,        // [*batch]     list    -> Strides of `nearest_vertex`
    const offset_t * stride_coord,          // [*batch, D]  list    -> Strides of `coord`
    const offset_t * stride_vertices,       // [N, D]       list    -> Strides of `vertices`
    const offset_t * stride_faces   ,       // [M, D]       list    -> Strides of `faces`
          intptr_t   stream = 0           // CUDA stream (0 == default stream)
)
{
    static const offset_t ndim = static_cast<offset_t>(_ndim);
    const cudaStream_t s = (cudaStream_t)(std::intptr_t)stream;

    index_t  * faces_device      = nullptr;
    scalar_t * verts_device      = nullptr;
    index_t  * faces_host        = nullptr;
    scalar_t * verts_host        = nullptr;
    scalar_t * normfaces_host    = nullptr;
    scalar_t * normverts_host    = nullptr;
    scalar_t * normedges_host    = nullptr;
    scalar_t * normfaces_device  = nullptr;
    scalar_t * normverts_device  = nullptr;
    scalar_t * normedges_device  = nullptr;
    offset_t * stride_vec_device = nullptr;
    offset_t * stride_mat_device = nullptr;
    offset_t * stride_dist_device    = nullptr;
    offset_t * stride_nearest_device = nullptr;
    offset_t * stride_coord_device   = nullptr;
    offset_t * size_device           = nullptr;

    try
    {
        offset_t   size_faces [2] = {nb_faces,    ndim};
        offset_t   size_verts [2] = {nb_vertices, ndim};
        offset_t   stride_vec [2] = {ndim, 1};
        // Strides of the (M, D, D) edge-normal tensor: three values.
        offset_t   stride_mat [3] = {ndim*ndim, ndim, 1};
        // `faces` and `vertices` are 2-D tensors -- (M, D) and (N, D) -- so the
        // rank handed to `copyTensorToContiguous` is 2, never `ndim`.
        static const offset_t rank = static_cast<offset_t>(2);

        // The normals are built by a host function, so the mesh has to be on
        // the host; the kernel then reads the contiguous device copies with
        // `stride_vec`. Same route as `sdt` above: device-side gather into a
        // contiguous buffer (which honours the caller's real input strides),
        // then one D2H copy.
        faces_device =
            copyTensorToContiguous(rank, faces, size_faces, stride_faces, s);
        verts_device = copyTensorToContiguous(rank, vertices, size_verts,
                                              stride_vertices, s);

        faces_host = copyToHost(faces_device, nb_faces    * ndim);
        verts_host = copyToHost(verts_device, nb_vertices * ndim);

        // Allocate normals. `normverts_host` must be zeroed: the builder
        // accumulates into it (see the same comment in `sdt`).
        normfaces_host = allocHost<scalar_t>(nb_faces    * ndim);
        normverts_host = allocHost<scalar_t>(nb_vertices * ndim);
        normedges_host = allocHost<scalar_t>(nb_faces    * ndim * ndim);
        for (offset_t i = 0; i < nb_vertices * ndim; ++i)
            normverts_host[i] = static_cast<scalar_t>(0);

        // Build normals
        build_normals<_ndim>(
            normfaces_host,
            normverts_host,
            normedges_host,
            faces_host,
            verts_host,
            nb_faces,
            nb_vertices,
            stride_vec,
            stride_vec,
            stride_mat,
            stride_vec,
            stride_vec
        );

        // Copy to device
        normfaces_device        = copyToDeviceAsync(normfaces_host, nb_faces    * ndim, s);
        normverts_device        = copyToDeviceAsync(normverts_host, nb_vertices * ndim, s);
        normedges_device        = copyToDeviceAsync(normedges_host, nb_faces    * ndim * ndim, s);
        stride_vec_device       = copyToDeviceAsync(stride_vec,     2, s);
        stride_mat_device       = copyToDeviceAsync(stride_mat,     3, s);
        stride_dist_device      = copyToDeviceAsync(stride_dist,    nbatch, s);
        // Guarded exactly as in `sdt`: `nearest_vertex` is optional and its
        // stride array is null when it is, which `copyToDeviceAsync` cannot be
        // handed (a null `cudaMemcpyAsync` source fails with
        // `cudaErrorInvalidValue`). That is the default call pattern.
        stride_nearest_device = nullptr;
        if (nearest_vertex)
            stride_nearest_device =
                copyToDeviceAsync(stride_nearest, nbatch, s);
        stride_coord_device     = copyToDeviceAsync(stride_coord,   nbatch + 1, s);
        size_device             = copyToDeviceAsync(size,           nbatch, s);

        // The grid covers the number of points evaluated -- the batch element
        // count, not `nbatch`, which is the batch *rank*.
        offset_t numel      = prod(size, nbatch);
        int      num_blocks = GET_BLOCKS(numel);

        // Compute SDT
        sdt_naive_kernel<ndim, scalar_t, index_t, offset_t>
            <<<num_blocks, CUDA_NUM_THREADS, 0, s>>>
            (
                nbatch,
                dist,
                nearest_vertex,
                coord,
                verts_device,
                faces_device,
                normfaces_device,
                normverts_device,
                normedges_device,
                size_device,
                nb_faces,
                stride_dist_device,
                stride_nearest_device,
                stride_coord_device,
                stride_vec_device,
                stride_vec_device,
                stride_vec_device,
                stride_vec_device,
                stride_mat_device
            );
    }
    catch (const std::exception & e)
    {
        freeDevice(
            faces_device,
            verts_device,
            normfaces_device,
            normverts_device,
            normedges_device,
            stride_vec_device,
            stride_mat_device,
            stride_dist_device,
            stride_nearest_device,
            stride_coord_device,
            size_device
        );
        freeHost(
            faces_host,
            verts_host,
            normfaces_host,
            normverts_host,
            normedges_host
        );
        throw e;
    }

    freeDevice(
        faces_device,
        verts_device,
        normfaces_device,
        normverts_device,
        normedges_device,
        stride_vec_device,
        stride_mat_device,
        stride_dist_device,
        stride_nearest_device,
        stride_coord_device,
        size_device
    );
    freeHost(
        faces_host,
        verts_host,
        normfaces_host,
        normverts_host,
        normedges_host
    );
}

template <int ndim, typename scalar_t, typename index_t, typename offset_t>
FF_CUHOST inline void sdt(
          offset_t   nbatch             ,  // Number of batch dimensions in coord
          scalar_t * dist               ,  // (*batch) tensor -> Output placeholder for distance
          index_t  * nearest_vertex     ,  // (*batch) tensor -> Output placeholder for index of nearest vertex
    const scalar_t * coord              ,  // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * vertices           ,  // (N, D) tensor -> All vertices
    const index_t  * faces              ,  // (M, D) tensor -> All faces (face = D vertex indices)
    const void     * tree               ,  // (log2(M) * F) tensor -> Binary tree
          void     * treetrace          ,  // (log2(M) * F) tensor -> Binary tree
          offset_t   treesize           ,
    const scalar_t * normfaces          ,  // (M, D) tensor
    const scalar_t * normvertices       ,  // (N, D) tensor
    const scalar_t * normedges          ,  // (M, D, D) tensor
    const offset_t * size               ,  // [*batch] list -> Size of `dist`
    const offset_t * stride_dist        ,  // [*batch] list -> Strides of `dist`
    const offset_t * stride_nearest     ,  // [*batch] list -> Strides of `nearest_vertex`
    const offset_t * stride_coord       ,  // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices    ,  // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces       ,  // [M, D] list -> Strides of `faces`
    const offset_t * stride_normfaces   ,  // [M, D] list
    const offset_t * stride_normvertices,  // [N, D] list
    const offset_t * stride_normedges   ,  // [M, D, D] list
          intptr_t   stream = 0        )  // CUDA stream (0 == default stream)
{
    // NOT IMPLEMENTED, and deliberately still a throw. Read this before
    // "finishing" it -- the missing piece is a contract, not code.
    //
    // History: the previous body was an erroneous copy/paste of a Euclidean
    // distance-transform launcher (it referenced a non-existent
    // `allocBuffer`/`freeBuffers` callback API and identifiers `f`/`w`/`stride`
    // that are not parameters here, and launched `sdt_kernel` with the wrong
    // signature). It was replaced by this throw.
    //
    // What is actually missing. Mechanically the body is short -- upload the
    // stride arrays, size the grid from `prod(size, nbatch)`, launch
    // `sdt_kernel` -- but three things this signature does not state have to be
    // decided before any of that can be *correct*, and none of them is
    // discoverable from a call site:
    //
    //   1. `const void * tree`. `sdt_kernel` takes
    //      `const DeviceNode<ndim, scalar_t, index_t> *`: a POD mirror, in
    //      *device* memory. A `void *` cannot distinguish that from the
    //      polymorphic host `MeshDist::Node[]` that `build_tree` produces --
    //      and passing the latter is precisely the defect
    //      fastfields-cuda-impl#44 fixed in `sdt_kernel` by giving the
    //      parameter a real type. Nothing in this repository hands a caller a
    //      `DeviceNode` array: `flatten_tree` + the upload are private to the
    //      `sdt` overload above. So there is no producer, and no caller.
    //   2. `void * treetrace` / `treesize`. The trace is per *lane*, not per
    //      point, and is interleaved across the launched lanes, so the buffer
    //      must be at least `GET_BLOCKS(numel) * CUDA_NUM_THREADS * treesize`
    //      bytes of device memory -- a size that depends on the launch
    //      configuration this function picks. A caller cannot size it without
    //      replicating that choice.
    //   3. `faces` must be the BVH-sorted face list. `MeshDist::build_tree`
    //      reorders faces in place and its leaves store indices into the
    //      *sorted* order (see the comment on cpu-impl's `sdt`), so the faces
    //      and pseudonormals passed here have to be the ones the tree was
    //      built from, not the caller's originals.
    //
    // The options, for whoever picks this up (fastfields-lib#5):
    //
    //   a. Give it a real signature and an in-repo caller: type `tree` as
    //      `const DeviceNode<ndim, scalar_t, index_t> *` and `treetrace` as
    //      `char *`, then make the `sdt` overload above delegate to it -- which
    //      is exactly how cpu-impl is layered (`sdt` builds, `build_sdt`
    //      queries). The contract then has a compiled user, and the name should
    //      follow cpu-impl too (`build_sdt`).
    //   b. Drop this overload. It has no caller, no producer for its tree
    //      argument, and no test.
    //
    // Either is a design change with a reviewable blast radius, so it is not
    // being made silently here. Until then this throws rather than pretending:
    // a launcher that compiles but cannot be given a valid `tree` would be
    // worse than none.
    throw std::logic_error("distance_mesh::sdt (precomputed tree) not implemented");
}

// Top-level mesh distance dispatcher (mirrors cpu-impl distance_mesh::dt).
//
// Coverage is PARTIAL -- of the four branches cpu-impl dispatches, two have a
// CUDA host launcher:
//
//   _signed  naive   CPU          CUDA
//   -------  -----   ----------   -------------------------------------------
//   true     false   sdt          sdt()            -> dispatched below
//   true     true    sdt_naive    sdt_naive()      -> dispatched below
//   false    false   udt          udt_kernel       -> no launcher, throws
//   false    true    udt_naive    udt_naive_kernel -> no launcher, throws
//
// The two unsigned device kernels exist and are type-checked by
// `tests/impl-cuda/compile_probe_mesh.cu`, but neither has a `FF_CUHOST` launcher
// to upload the mesh and size the grid. Writing those is tracked separately;
// until then those branches throw rather than silently returning garbage. See
// fastfields-lib#5.
//
// Verification status of the branches that *are* dispatched: the per-element
// math (`MeshDist`, `signed_dist`, `signed_dist_naive`, ...) is the shared
// kernels source under `impl/kernels/`, compiled here and on the CPU alike and
// exhaustively covered by `tests/lib-cpu/test_distance_mesh.cpp`. What is
// CUDA-only is the launcher glue -- the host BVH build and normal build, the
// POD flatten/upload, the grid sizing and the trace buffer -- and that is
// backed by nvcc compile+link evidence only, since there is no GPU in CI.
// Nothing here has ever been executed: `sdt` vs `sdt_naive` agreement on real
// hardware is still the open acceptance bar of fastfields-lib#5. Same bar as
// every other module in src/lib-cuda's MODULES.
template <int ndim, typename scalar_t, typename index_t, typename offset_t>
FF_CUHOST inline void
dt(
          offset_t   nbatch,
          scalar_t * dist,
          index_t  * nearest_vertex,
    const scalar_t * coord,
    const scalar_t * vertices,
    const index_t  * faces,
    const offset_t * size,
          offset_t   nb_faces,
          offset_t   nb_vertices,
    const offset_t * stride_dist,
    const offset_t * stride_nearest,
    const offset_t * stride_coord,
    const offset_t * stride_vertices,
    const offset_t * stride_faces,
          bool       _signed = false,
          bool       naive   = false,
          intptr_t   stream  = 0
)
{
    // Signed, tree-accelerated.
    //
    // `sdt` builds the BVH and the vertex/face/edge normals on the host,
    // uploads a POD mirror of the tree, sizes the grid from the batch element
    // count and launches `sdt_kernel` on `stream`. `nearest_vertex` may be
    // null, in which case the nearest-vertex output (and its stride upload) is
    // skipped.
    //
    // NB: `sdt` takes (nb_faces, nb_vertices) in that order, same as here.
    if (_signed && !naive)
        return sdt<ndim, scalar_t, index_t, offset_t>(
            nbatch, dist, nearest_vertex, coord, vertices, faces, size,
            nb_faces, nb_vertices, stride_dist, stride_nearest, stride_coord,
            stride_vertices, stride_faces, stream);

    // Signed, brute-force: the reference the branch above is meant to be
    // validated against. Builds and uploads the normals, then launches
    // `sdt_naive_kernel`; no BVH, no traversal trace. Same argument order.
    if (_signed && naive)
        return sdt_naive<ndim, scalar_t, index_t, offset_t>(
            nbatch, dist, nearest_vertex, coord, vertices, faces, size,
            nb_faces, nb_vertices, stride_dist, stride_nearest, stride_coord,
            stride_vertices, stride_faces, stream);

    // The two unsigned branches have device kernels but no host launcher -- see
    // the table above this function. Throw a message that names the missing
    // piece rather than the generic "not implemented", so a caller who hits one
    // knows which variant to ask for. (No `(void)` casts needed: every
    // parameter is used by the dispatched branches above.)
    if (!_signed && !naive)
        throw std::logic_error(
            "distance_mesh::dt (CUDA): the unsigned mesh distance (udt) has "
            "no host launcher yet");
    throw std::logic_error(
        "distance_mesh::dt (CUDA): the naive unsigned mesh distance "
        "(udt_naive) has no host launcher yet");
}


FF_NAMESPACE_END(distance_mesh)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)
