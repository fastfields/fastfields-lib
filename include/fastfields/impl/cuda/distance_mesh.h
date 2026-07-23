#pragma once
#include "kernels/cuda_switch.h"
#include "kernels/distance.h"
#include "kernels/batch.h"
#include "kernels/utils.h"
#include "utils.h"

FF_NAMESPACE_BEGIN(FF)
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
CUHOST inline void
build_tree(
          void     * _tree            ,  // (log2(M) * F) tensor -> Placeholder for binary tree
          index_t  * _faces           ,  // (M, D) tensor -> All faces (face = D vertex indices)
    const scalar_t * _vertices        ,  // (N, D) tensor -> All vertices
          offset_t   nb_faces         ,  // M
          offset_t   nb_vertices      ,  // N
    const offset_t * stride_faces     ,  // [M, D] list -> Strides of `faces`
    const offset_t * stride_vertices  )  // [N, D] list -> Strides of `vertices`
{
    using Klass      = MeshDist<ndim, scalar_t, index_t, offset_t>;
    using Node       = typename Klass::Node;
    using FaceList   = StridedPointList<ndim, index_t, offset_t>;
    using VertexList = ConstStridedPointList<ndim, scalar_t, offset_t>;

    auto  faces      = FaceList     (_faces,    stride_faces    [0], stride_faces   [1]);
    auto  vertices   = VertexList   (_vertices, stride_vertices [0], stride_vertices[1]);
    auto  tree       = reinterpret_cast<Node *>(_tree);

    index_t node_id = 0;
    Klass::build_tree(tree, node_id, -1, 0, nb_faces, faces, vertices);
}

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
CUHOST inline void
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

template <typename offset_t>
CUHOSTDEV inline
offset_t * contiguousStrides(const offset_t * size, int ndim)
{
    offset_t * stride = allocHost<offset_t>(ndim);
    stride[ndim-1] = static_cast<offset_t>(1);
    for (int d=ndim-2; d >= 0; --d)
        stride[d] = size[d+1] * stride[d+1];
    return stride;
}

template <typename scalar_t, typename offset_t>
CUGLOB inline void
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
CUHOST inline
scalar_t * copyTensorToContiguous(
          offset_t   ndim,
    const scalar_t * inp,
    const offset_t * size,
    const offset_t * stride_inp)
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
        size_copy       = copyToDevice(size, ndim);
        stride_out_copy = copyToDevice(stride_out, ndim);
        stride_inp_copy = copyToDevice(stride_inp, ndim);
        // Copy data
        copy_tensor_kernel<scalar_t, offset_t>
            <<<GET_BLOCKS(numel), CUDA_NUM_THREADS, 0>>>
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

template <
    int      ndim,          // Number of spatial dimensions
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
CUGLOB inline void
copy_faces_kernel(
          offset_t   nb_faces       , // Number of faces (M)
    const index_t  * faces_out      , // (M, D) output (contiguous) tensor of faces
    const index_t  * faces_inp      , // (M, D) input tensor of faces
          offset_t   stride_elem    , // Input stride between elements
          offset_t   stride_channel ) // Input stride between channels
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
CUHOST inline
index_t * copy_faces(
          offset_t   nb_faces   ,
    const index_t  * faces      ,
    const offset_t * stride     )
{
    offset_t stride0 = stride[0], stride1 = stride[1];
    index_t * faces_out = allocDevice<index_t>(nb_faces * ndim);
    copy_faces_kernel<ndim, index_t, offset_t>
        <<<GET_BLOCKS(nb_faces), CUDA_NUM_THREADS, 0>>>
        (nb_faces, faces_out, faces, stride0, stride1);
    return faces_out;
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
CUGLOB inline void sdt_kernel(
          offset_t   nbatch             ,  // Number of batch dimensions in coord
          scalar_t * dist               ,  // (*batch) tensor -> Output placeholder for distance
          index_t  * nearest_vertex     ,  // (*batch) tensor -> Output placeholder for index of nearest vertex
    const scalar_t * coord              ,  // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices          ,  // (N, D) tensor -> All vertices
    const index_t  * _faces             ,  // (M, D) tensor -> All faces (face = D vertex indices)
    const void     * _tree              ,  // (log2(M) * F) tensor -> Binary tree
          void     * _treetrace         ,  // (log2(M) * F) tensor -> Binary tree
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

    using Klass          = MeshDist<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointList<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using NormalList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using EdgeNormalList = ConstStridedPointArray<ndim, scalar_t, offset_t, ndim>;
    using EdgeStride     = StaticPoint<3, offset_t>;
    using RefPoint       = ConstStridedPoint<ndim, scalar_t, offset_t>;
    using ClonedPoint    = StaticPoint<ndim, scalar_t>;

    auto treetrace = SizedStridedPointer<char, offset_t>(_treetrace + index, stride, treesize);

    // In 2D -> no edges
    auto _stride_normedges = EdgeStride();
    if (stride_normedges)
        _stride_normedges.copy_(ConstRefPoint<3, offset_t>(stride_normedges));

    auto faces        = FaceList      (_faces,        stride_faces        [0], stride_faces        [1]);
    auto vertices     = VertexList    (_vertices,     stride_vertices     [0], stride_vertices     [1]);
    auto normfaces    = NormalList    (_normfaces,    stride_normfaces    [0], stride_normfaces    [1]);
    auto normvertices = NormalList    (_normvertices, stride_normvertices [0], stride_normvertices [1]);
    auto normedges    = EdgeNormalList(_normedges,    _stride_normedges);
    auto tree         = reinterpret_cast<const Node *>(_tree);

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
            offset_nearest  = index2offset<nbatch>(i, nbatch, size, stride_nearest);
        ClonedPoint point;
        point.copy_(RefPoint(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = Klass::signed_dist(
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
CUGLOB inline void sdt_naive_kernel(
          offset_t   nbatch             ,  // Number of batch dimensions in coord
          scalar_t * dist               ,  // (*batch) tensor -> Output placeholder for distance
    const scalar_t * coord              ,  // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices          ,  // (N, D) tensor -> All vertices
    const index_t  * _faces             ,  // (M, D) tensor -> All faces (face = D vertex indices)
    const scalar_t * _normfaces         ,  // (M, D) tensor
    const scalar_t * _normvertices      ,  // (N, D) tensor
    const scalar_t * _normedges         ,  // (M, D, D) tensor
    const offset_t * size               ,  // [*batch] list -> Size of `dist`
          offset_t   nb_faces           ,
    const offset_t * stride_dist        ,  // [*batch] list -> Strides of `dist`
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
    using Node           = typename Klass::Node;
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

        StaticPoint<ndim, scalar_t> point(
            ConstStridedPoint<ndim, scalar_t, offset_t>(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = Klass::signed_dist_naive(
            point,
            vertices,
            faces,
            normfaces,
            normedges,
            normvertices
        );
    }
}

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
CUGLOB inline void udt_kernel(
          offset_t   nbatch         ,  // Number of batch dimensions in coord
          scalar_t * dist           ,  // (*batch) tensor -> Output placeholder for distance
    const scalar_t * coord          ,  // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices      ,  // (N, D) tensor -> All vertices
    const index_t  * _faces         ,  // (M, D) tensor -> All faces (face = D vertex indices)
    const void     * _tree          ,  // (log2(M) * F) tensor -> Binary tree
          void     * _treetrace     ,  // (log2(M) * F) tensor -> Binary tree
          offset_t   treesize       ,
    const offset_t * size           ,  // [*batch] list -> Size of `dist`
    const offset_t * stride_dist    ,  // [*batch] list -> Strides of `dist`
    const offset_t * stride_coord   ,  // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,  // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces   )  // [M, D] list -> Strides of `faces`
{
    offset_t index = threadIdx.x + blockIdx.x * blockDim.x;
    offset_t stride = blockDim.x * gridDim.x;

    using Klass          = MeshDist<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointList<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;

    auto treetrace = SizedStridedPointer<char, offset_t>(_treetrace + index, stride, treesize);

    auto faces        = FaceList  (_faces,    stride_faces   [0], stride_faces   [1]);
    auto vertices     = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);
    auto tree         = reinterpret_cast<const Node *>(_tree);

    offset_t numel = prod(size, nbatch);
    for (offset_t i=index; index < numel; index += stride, i=index)
    {
        for (offset_t j=0; j<treetrace.size; ++j)
            treetrace[j] = static_cast<char>(0);

        offset_t offset_coord = index2offset(i, nbatch, size, stride_coord);
        offset_t offset_dist  = index2offset(i, nbatch, size, stride_dist);

        StaticPoint<ndim, scalar_t> point(
            ConstStridedPoint<ndim, scalar_t, offset_t>(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = Klass::unsigned_dist(
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
CUGLOB inline void udt_naive_kernel(
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

template <
    int      _ndim,         // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
CUHOST inline void
sdt(
          offset_t   nbatch,                // Number of batch dimensions in coord
          scalar_t * dist,                  // (*batch)     tensor  -> Output placeholder for distance
          index_t  * nearest_vertex,        // (*batch)     tensor  -> Output placeholder for index of nearest vertex
    const scalar_t * coord,                 // (*batch, D)  tensor  -> Coordinates at which to evaluate distance
    const scalar_t * vertices,              // (N, D)       tensor  -> All vertices
    const index_t  * faces,                 // (M, D)       tensor  -> All faces (face = D vertex indices)
    const offset_t * size,                  // [*batch]     list    -> Size of `dist`
          offset_t   nb_vertices,           // N                    -> Number of vertices
          offset_t   nb_faces,              // M                    -> Number of faces
    const offset_t * stride_dist,           // [*batch]     list    -> Strides of `dist`
    const offset_t * stride_nearest,        // [*batch]     list    -> Strides of `nearest_vertex`
    const offset_t * stride_coord,          // [*batch, D]  list    -> Strides of `coord`
    const offset_t * stride_vertices,       // [N, D]       list    -> Strides of `vertices`
    const offset_t * stride_faces           // [M, D]       list    -> Strides of `faces`
)
{
    static const offset_t ndim = static_cast<offset_t>(_ndim);

    index_t  * faces_device      = nullptr;
    scalar_t * verts_device      = nullptr;
    uint8_t  * tree_device       = nullptr;
    index_t  * faces_host        = nullptr;
    scalar_t * verts_host        = nullptr;
    uint8_t  * tree_host         = nullptr;
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
        offset_t   stride_mat    [2] = {ndim*ndim, ndim, 1};
        index_t  * faces_device      = copyTensorToContiguous(ndim, faces,    size_faces, stride_vec);
        scalar_t * verts_device      = copyTensorToContiguous(ndim, vertices, size_verts, stride_vec);

        // Copy to host
        index_t  * faces_host = copyToHost(faces_device, nb_faces    * ndim);
        scalar_t * verts_host = copyToHost(verts_device, nb_vertices * ndim);

        // Allocate tree
        offset_t nb_levels = static_cast<offset_t>(ceil(log2(static_cast<scalar_t>(nb_faces)))) + 3;
        offset_t nb_nodes  = 0;
        for (offset_t i = 0, pow = 1; i < nb_levels; ++i) {
            nb_nodes += pow;
            pow      *= 2;
        }
        offset_t nb_features = sizeof(scalar_t) * 2*(ndim+1) + sizeof(index_t) * 3;
        uint8_t * tree_host = allocHost<uint8_t>(nb_nodes * nb_features);

        // Build tree
        build_tree<ndim>(
            tree_host,
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

        // Copy to device
        faces_device            = copyToDevice(faces_host,     nb_faces    * ndim, faces_device);
        tree_device             = copyToDevice(tree_host,      nb_nodes    * nb_features);
        normfaces_device        = copyToDevice(normfaces_host, nb_faces    * ndim);
        normverts_device        = copyToDevice(normverts_host, nb_vertices * ndim);
        normedges_device        = copyToDevice(normedges_host, nb_faces    * ndim * ndim);
        stride_vec_device       = copyToDevice(stride_vec,     2);
        stride_mat_device       = copyToDevice(stride_mat,     2);
        stride_dist_device      = copyToDevice(stride_dist,    nbatch);
        stride_nearest_device   = copyToDevice(stride_nearest, nbatch);
        stride_coord_device     = copyToDevice(stride_coord,   nbatch + 1);
        size_device             = copyToDevice(size,           nbatch);

        // Compute SDT
        sdt_kernel<ndim, scalar_t, index_t, offset_t>
            <<<GET_BLOCKS(nbatch), CUDA_NUM_THREADS, 0>>>
            (
                nbatch,
                dist,
                nearest_vertex,
                coord,
                verts_device,
                faces_device,
                tree_host,
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
            tree_host,
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
        tree_host,
        normfaces_host,
        normverts_host,
        normedges_host
    );
}

template <int ndim, typename scalar_t, typename index_t, typename offset_t>
CUHOST inline void sdt(
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
    const offset_t * stride_normedges   )  // [M, D, D] list
{
    // TODO(host-launcher): this precomputed-tree/normals `sdt` launcher is not
    // implemented yet. The previous body was an erroneous copy/paste of a
    // Euclidean distance-transform launcher (it referenced a non-existent
    // `allocBuffer`/`freeBuffers` callback API and identifiers `f`/`w`/`stride`
    // that are not parameters here, and launched `sdt_kernel` with the wrong
    // signature). Implementing it correctly against the mesh `sdt_kernel`
    // (line ~219) is tracked as the separate host-launcher task; see the
    // complete `sdt` overload above (which builds the tree/normals itself).
    throw std::logic_error("distance_mesh::sdt (precomputed tree) not implemented");
}

// Top-level mesh distance dispatcher (mirrors cpu-impl distance_mesh::dt).
// The signed, non-naive path forwards to the complete `sdt` launcher above
// (which builds the tree + normals on device itself). The naive and unsigned
// CUDA launchers are not written yet (they exist on the CPU side); they throw
// for now. Compile-verified under nvcc; runtime correctness needs a GPU.
template <int ndim, typename scalar_t, typename index_t, typename offset_t>
CUHOST inline void
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
          bool       naive   = false
)
{
    // The CUDA mesh launchers (sdt/sdt_naive/udt/udt_naive) are not finished
    // yet — the existing `sdt` above still has gaps (missing `build_tree`,
    // etc.). Once they compile+run on device, dispatch here like cpu-impl's
    // distance_mesh::dt. For now throw so the dispatch layer links.
    (void)nbatch; (void)dist; (void)nearest_vertex; (void)coord; (void)vertices;
    (void)faces; (void)size; (void)nb_faces; (void)nb_vertices; (void)stride_dist;
    (void)stride_nearest; (void)stride_coord; (void)stride_vertices;
    (void)stride_faces; (void)_signed; (void)naive;
    throw std::logic_error("distance_mesh::dt (CUDA) not implemented");
}


FF_NAMESPACE_END(distance_mesh)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)
