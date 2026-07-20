#ifndef FF_CPU_DISTANCE_MESH
#define FF_CPU_DISTANCE_MESH
#include "kernels/cuda_switch.h"
#include "kernels/distance.h"
#include "kernels/distance/mesh_utils.h"
#include "kernels/batch.h"
#include "kernels/parallel.h"

FF_NAMESPACE_BEGIN(FF)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(distance_mesh)

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
inline void
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
inline void
build_normals(
          scalar_t * _normfaces,            // (M, D) tensor
          scalar_t * _normvertices,         // (N, D) tensor
          scalar_t * _normedges,            // (M, D, D) tensor
    const index_t  * _faces,                // (M, D) tensor -> All faces (face = D vertex indices)
    const scalar_t * _vertices,             // (N, D) tensor -> All vertices
          offset_t   nb_faces,              // M
          offset_t   nb_vertices,           // N
    const offset_t * stride_normfaces,      // [M, D] list
    const offset_t * stride_normvertices,   // [N, D] list
    const offset_t * stride_normedges,      // [M, D, D] list
    const offset_t * stride_faces,          // [M, D] list -> Strides of `faces`
    const offset_t * stride_vertices        // [N, D] list -> Strides of `vertices`
)
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

template <
    int      ndim,          // Number of spatial dimensions
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
inline
index_t * copy_faces(
          offset_t   nb_faces,
    const index_t  * faces,
    const offset_t * stride
)
{
    offset_t stride0 = stride[0], stride1 = stride[1];
    index_t * faces_out  = new index_t[nb_faces * ndim];
    parallel_for(0, nb_faces, GRAIN_SIZE, [&](long start, long end) {
    const index_t * ptr_inp = faces     + start * stride0;
    index_t * ptr_out = faces_out + start * ndim;
    for (offset_t i = start; i < end; ++i, ptr_inp += stride0)
    {
        const index_t * ptr_inp_c = ptr_inp;
        for (offset_t c = 0; c < ndim; ++c, ++ptr_out, ptr_inp_c += stride1)
        {
            *ptr_out = *ptr_inp_c;
        }
    }});
    return faces_out;
}

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
inline void
build_sdt(
          offset_t   nbatch,                // Number of batch dimensions in coord
          scalar_t * dist,                  // (*batch) tensor -> Output placeholder for distance
          index_t  * nearest_vertex,        // (*batch) tensor -> Output placeholder for index of nearest vertex
//        uint8_t  * _nearest_entity,
    const scalar_t * coord,                 // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices,             // (N, D) tensor -> All vertices
    const index_t  * _faces,                // (M, D) tensor -> All faces (face = D vertex indices)
    const uint8_t  * _tree,                 // (log2(M) * F) tensor -> Binary tree
    const scalar_t * _normfaces,            // (M, D) tensor
    const scalar_t * _normvertices,         // (N, D) tensor
    const scalar_t * _normedges,            // (M, D, D) tensor
    const offset_t * size,                  // [*batch] list -> Size of `dist`
    const offset_t * stride_dist,           // [*batch] list -> Strides of `dist`
    const offset_t * stride_nearest,        // [*batch] list -> Strides of `nearest_vertex`
//  const offset_t * stride_nearest_e,      // [*batch] list -> Strides of `nearest_vertex`
    const offset_t * stride_coord,          // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,       // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces,          // [M, D] list -> Strides of `faces`
    const offset_t * stride_normfaces,      // [M, D] list
    const offset_t * stride_normvertices,   // [N, D] list
    const offset_t * stride_normedges       // [M, D, D] list
)
{
    using Klass          = MeshDist<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointList<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using NormalList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using EdgeNormalList = ConstStridedPointArray<ndim, scalar_t, offset_t, ndim>;
    using EdgeStride     = StaticPoint<3, offset_t>;
    using StridedPointND = ConstStridedPoint<ndim, scalar_t, offset_t>;
    using StaticPointND  = StaticPoint<ndim, scalar_t>;

    // auto nearest_entity = reinterpret_cast<NearestEntity*>(_nearest_entity);

    // In 2D -> no edges
    auto _stride_normedges = EdgeStride();
    if (stride_normedges)
        _stride_normedges.copy_(ConstRefPoint<3, offset_t>(stride_normedges));

    auto faces        = FaceList        (_faces,        stride_faces        [0], stride_faces        [1]);
    auto vertices     = VertexList      (_vertices,     stride_vertices     [0], stride_vertices     [1]);
    auto normfaces    = NormalList      (_normfaces,    stride_normfaces    [0], stride_normfaces    [1]);
    auto normvertices = NormalList      (_normvertices, stride_normvertices [0], stride_normvertices [1]);
    auto normedges    = EdgeNormalList  (_normedges,    _stride_normedges);
    auto tree         = reinterpret_cast<const Node *>(_tree);

    offset_t numel = prod(size, nbatch);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_coord = index2offset(i, nbatch, size, stride_coord);
        offset_t offset_dist  = index2offset(i, nbatch, size, stride_dist);
        offset_t offset_nearest = 0;
        if (nearest_vertex)
            offset_nearest  = index2offset(i, nbatch, size, stride_nearest);
        // offset_t offset_nearest_e = 0;
        // if (nearest_entity)
        //     offset_nearest_e  = index2offset(i, nbatch, size, stride_nearest_e);

        StaticPointND point(StridedPointND(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = Klass::signed_dist(
            point,
            vertices,
            faces,
            tree,
            normfaces,
            normedges,
            normvertices,
            nearest_vertex + offset_nearest /*,
            nullptr,
            nearest_entity + offset_nearest_e
            */
        );
    }});
}

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
inline void
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
    const offset_t * stride_faces           // [M, D]       list    -> Strides of `faces`
)
{
    // Make a copy of the faces, so that it can be modified in-place
    index_t * faces_copy = copy_faces<ndim>(nb_faces, faces, stride_faces);
    offset_t  stride_faces_copy[2] = {ndim, 1};

    // Allocate tree
    offset_t nb_levels = static_cast<offset_t>(ceil(log2(static_cast<scalar_t>(nb_faces)))) + 3;
    offset_t nb_nodes  = 0;
    for (offset_t i = 0, pow = 1; i < nb_levels; ++i) {
        nb_nodes += pow;
        pow      *= 2;
    }
    offset_t nb_features = sizeof(scalar_t) * 2*(ndim+1) + sizeof(index_t) * 3;
    uint8_t * tree = new uint8_t[nb_nodes * nb_features];

    // Build tree
    build_tree<ndim>(
        tree,
        faces_copy,
        vertices,
        nb_faces,
        nb_vertices,
        stride_faces_copy,
        stride_vertices
    );

    // Allocate normals
    scalar_t * normfaces    = new scalar_t[nb_faces    * ndim],
             * normvertices = new scalar_t[nb_vertices * ndim](), // zero-init: accumulated
             * normedges    = new scalar_t[nb_faces    * ndim * ndim];
    offset_t stride_normfaces    [2] = {ndim, 1},
             stride_normvertices [2] = {ndim, 1},
             stride_normedges    [3] = {ndim*ndim, ndim, 1};

    // Build normals
    build_normals<ndim>(
        normfaces,
        normvertices,
        normedges,
        faces_copy,
        vertices,
        nb_faces,
        nb_vertices,
        stride_normfaces,
        stride_normvertices,
        stride_normedges,
        stride_faces_copy,
        stride_vertices
    );

    // Compute SDT
    build_sdt<ndim>(
        nbatch,
        dist,
        nearest_vertex,
        coord,
        vertices,
        faces,
        tree,
        normfaces,
        normvertices,
        normedges,
        size,
        stride_dist,
        stride_nearest,
        stride_coord,
        stride_vertices,
        stride_faces,
        stride_normfaces,
        stride_normvertices,
        stride_normedges
    );

    // Delete buffers
    delete[] faces_copy;
    delete[] tree;
    delete[] normfaces;
    delete[] normvertices;
    delete[] normedges;
}

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
inline void
build_sdt_naive(
          offset_t   nbatch,                // Number of batch dimensions in coord
          scalar_t * dist,                  // (*batch) tensor -> Output placeholder for distance
          index_t  * nearest_vertex,        // (*batch) tensor -> Output placeholder for index of nearest vertex
    const scalar_t * coord,                 // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices,             // (N, D) tensor -> All vertices
    const index_t  * _faces,                // (M, D) tensor -> All faces (face = D vertex indices)
    const scalar_t * _normfaces,            // (M, D) tensor
    const scalar_t * _normvertices,         // (N, D) tensor
    const scalar_t * _normedges,            // (M, D, D) tensor
    const offset_t * size,                  // [*batch] list -> Size of `dist`
          offset_t   nb_faces,
    const offset_t * stride_dist,           // [*batch] list -> Strides of `dist`
    const offset_t * stride_nearest,        // [*batch] list -> Strides of `nearest_vertex`
    const offset_t * stride_coord,          // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,       // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces,          // [M, D] list -> Strides of `faces`
    const offset_t * stride_normfaces,      // [M, D] list
    const offset_t * stride_normvertices,   // [N, D] list
    const offset_t * stride_normedges       // [M, D, D] list
)
{
    using Klass          = MeshDist<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointListSized<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using NormalList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using EdgeNormalList = ConstStridedPointArray<ndim, scalar_t, offset_t, ndim>;
    using EdgeStride     = StaticPoint<3, offset_t>;
    using StridedPointND = ConstStridedPoint<ndim, scalar_t, offset_t>;
    using StaticPointND  = StaticPoint<ndim, scalar_t>;

    // In 2D -> no edges
    auto _stride_normedges = EdgeStride();
    if (stride_normedges)
        _stride_normedges.copy_(ConstRefPoint<3, offset_t>(stride_normedges));

    auto faces        = FaceList(_faces, stride_faces[0], stride_faces[1], nb_faces);
    auto vertices     = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);
    auto normfaces    = NormalList(_normfaces, stride_normfaces[0], stride_normfaces[1]);
    auto normvertices = NormalList(_normvertices, stride_normvertices[0], stride_normvertices[1]);
    auto normedges    = EdgeNormalList(_normedges, _stride_normedges);

    offset_t numel = prod(size, nbatch);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_coord = index2offset(i, nbatch, size, stride_coord);
        offset_t offset_dist  = index2offset(i, nbatch, size, stride_dist);
        offset_t offset_nearest = 0;
        if (nearest_vertex)
            offset_nearest  = index2offset(i, nbatch, size, stride_nearest);

        StaticPointND point(StridedPointND(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = Klass::signed_dist_naive(
            point,
            vertices,
            faces,
            normfaces,
            normedges,
            normvertices,
            nearest_vertex + offset_nearest
        );
    }});
}

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
inline void
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
    const offset_t * stride_faces           // [M, D]       list    -> Strides of `faces`
)
{
    // Allocate normals
    scalar_t * normfaces    = new scalar_t[nb_faces    * ndim],
             * normvertices = new scalar_t[nb_vertices * ndim](), // zero-init: accumulated
             * normedges    = new scalar_t[nb_faces    * ndim * ndim];
    offset_t stride_normfaces    [2] = {ndim, 1},
             stride_normvertices [2] = {ndim, 1},
             stride_normedges    [3] = {ndim*ndim, ndim, 1};

    // Build normals
    build_normals<ndim>(
        normfaces,
        normvertices,
        normedges,
        faces,
        vertices,
        nb_faces,
        nb_vertices,
        stride_normfaces,
        stride_normvertices,
        stride_normedges,
        stride_faces,
        stride_vertices
    );

    // Compute SDT
    build_sdt_naive<ndim>(
        nbatch,
        dist,
        nearest_vertex,
        coord,
        vertices,
        faces,
        normfaces,
        normvertices,
        normedges,
        size,
        nb_faces,
        stride_dist,
        stride_nearest,
        stride_coord,
        stride_vertices,
        stride_faces,
        stride_normfaces,
        stride_normvertices,
        stride_normedges
    );

    // Delete buffers
    delete[] normfaces;
    delete[] normvertices;
    delete[] normedges;
}

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
inline void
build_udt(
          offset_t   nbatch,            // Number of batch dimensions in coord
          scalar_t * dist,              // (*batch) tensor -> Output placeholder for distance
          index_t  * nearest_vertex,    // (*batch) tensor -> Output placeholder for index of nearest vertex
    const scalar_t * coord,             // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices,         // (N, D) tensor -> All vertices
    const index_t  * _faces,            // (M, D) tensor -> All faces (face = D vertex indices)
    const uint8_t  * _tree,             // (log2(M) * F) tensor -> Binary tree
    const offset_t * size,              // [*batch] list -> Size of `dist`
    const offset_t * stride_dist,       // [*batch] list -> Strides of `dist`
    const offset_t * stride_nearest,    // [*batch] list -> Strides of `nearest_vertex`
    const offset_t * stride_coord,      // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,   // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces       // [M, D] list -> Strides of `faces`
)
{
    using Klass          = MeshDist<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointList<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using StridedPointND = ConstStridedPoint<ndim, scalar_t, offset_t>;
    using StaticPointND  = StaticPoint<ndim, scalar_t>;


    auto faces    = FaceList(_faces, stride_faces[0], stride_faces[1]);
    auto vertices = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);
    auto tree     = reinterpret_cast<const Node *>(_tree);

    offset_t numel = prod(size, nbatch);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_coord = index2offset(i, nbatch, size, stride_coord);
        offset_t offset_dist  = index2offset(i, nbatch, size, stride_dist);
        offset_t offset_nearest = 0;
        if (nearest_vertex)
            offset_nearest  = index2offset(i, nbatch, size, stride_nearest);

        StaticPointND point(StridedPointND(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = Klass::unsigned_dist(
            point,
            vertices,
            faces,
            tree,
            nearest_vertex + offset_nearest
        );
    }});
}

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
inline void
udt(
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
    const offset_t * stride_faces           // [M, D]       list    -> Strides of `faces`
)
{
    // Make a copy of the faces, so that it can be modified in-place
    index_t * faces_copy = copy_faces<ndim>(nb_faces, faces, stride_faces);
    offset_t  stride_faces_copy[2] = {ndim, 1};

    // Allocate tree
    offset_t nb_levels = static_cast<offset_t>(ceil(log2(static_cast<scalar_t>(nb_faces)))) + 3;
    offset_t nb_nodes  = 0;
    for (offset_t i = 0, pow = 1; i < nb_levels; ++i) {
        nb_nodes += pow;
        pow      *= 2;
    }
    offset_t nb_features = sizeof(scalar_t) * 2*(ndim+1) + sizeof(index_t) * 3;
    uint8_t * tree = new uint8_t[nb_nodes * nb_features];

    // Build tree
    build_tree<ndim>(
        tree,
        faces_copy,
        vertices,
        nb_faces,
        nb_vertices,
        stride_faces_copy,
        stride_vertices
    );

    // Compute DT
    build_udt<ndim>(
        nbatch,
        dist,
        nearest_vertex,
        coord,
        vertices,
        faces_copy,
        tree,
        size,
        stride_dist,
        stride_nearest,
        stride_coord,
        stride_vertices,
        stride_faces_copy
    );

    // Delete buffers
    delete[] faces_copy;
    delete[] tree;
}

template <
    int      ndim,          // Number of spatial dimensions
    typename scalar_t,      // Value data type
    typename index_t,       // Index (faces) data type
    typename offset_t       // Index/Stride data type
>
inline void
udt_naive(
          offset_t   nbatch,            // Number of batch dimensions in coord
          scalar_t * dist,              // (*batch) tensor -> Output placeholder for distance
          index_t  * nearest_vertex,    // (*batch) tensor -> Output placeholder for index of nearest vertex
    const scalar_t * coord,             // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * _vertices,         // (N, D) tensor -> All vertices
    const index_t  * _faces,            // (M, D) tensor -> All faces (face = D vertex indices)
    const offset_t * size,              // [*batch] list -> Size of `dist`
          offset_t   nb_faces,
          offset_t   nb_vertices,       // UNUSED
    const offset_t * stride_dist,       // [*batch] list -> Strides of `dist`
    const offset_t * stride_nearest,    // [*batch] list -> Strides of `nearest_vertex`
    const offset_t * stride_coord,      // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,   // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces       // [M, D] list -> Strides of `faces`
)
{
    using Klass          = MeshDist<ndim, scalar_t, index_t, offset_t>;
    using Node           = typename Klass::Node;
    using FaceList       = ConstStridedPointListSized<ndim, index_t, offset_t>;
    using VertexList     = ConstStridedPointList<ndim, scalar_t, offset_t>;
    using StridedPointND = ConstStridedPoint<ndim, scalar_t, offset_t>;
    using StaticPointND  = StaticPoint<ndim, scalar_t>;


    auto faces    = FaceList(_faces, stride_faces[0], stride_faces[1], nb_faces);
    auto vertices = VertexList(_vertices, stride_vertices[0], stride_vertices[1]);

    offset_t numel = prod(size, nbatch);
    parallel_for(0, numel, GRAIN_SIZE, [&](long start, long end) {
    for (offset_t i=start; i < end; ++i)
    {
        offset_t offset_coord = index2offset(i, nbatch, size, stride_coord);
        offset_t offset_dist  = index2offset(i, nbatch, size, stride_dist);
        offset_t offset_nearest = 0;
        if (nearest_vertex)
            offset_nearest  = index2offset(i, nbatch, size, stride_nearest);

        StaticPointND point(StridedPointND(coord + offset_coord, stride_coord[nbatch]));

        dist[offset_dist] = Klass::unsigned_dist_naive(
            point,
            vertices,
            faces,
            nearest_vertex + offset_nearest
        );
    }});
}

template <
    int      ndim,
    typename scalar_t = float,              // Value data type
    typename index_t  = int64_t,            // Index/Stride data type
    typename offset_t = int64_t             // Index/Stride data type
>
inline void
dt(
          offset_t   nbatch,                // Number of batch dimensions in coord
          scalar_t * dist,                  // (*batch) tensor -> Output placeholder for distance
          index_t  * nearest_vertex,        // (*batch) tensor -> Output placeholder for index of nearest vertex
    const scalar_t * coord,                 // (*batch, D) tensor -> Coordinates at which to evaluate distance
    const scalar_t * vertices,              // (N, D) tensor -> All vertices
    const index_t  * faces,                 // (M, D) tensor -> All faces (face = D vertex indices)
    const offset_t * size,                  // [*batch] list -> Size of `dist`
          offset_t   nb_faces,
          offset_t   nb_vertices,
    const offset_t * stride_dist,           // [*batch] list -> Strides of `dist`
    const offset_t * stride_nearest,        // [*batch] list -> Strides of `nearest_vertex`
    const offset_t * stride_coord,          // [*batch, D] list -> Strides of `coord`
    const offset_t * stride_vertices,       // [N, D] list -> Strides of `vertices`
    const offset_t * stride_faces,          // [M, D] list -> Strides of `faces`
          bool      _signed = false,
          bool      naive   = false
)
{
    typedef void (*func_t)(
              offset_t  ,
              scalar_t *,
              index_t  *,
        const scalar_t *,
        const scalar_t *,
        const index_t  *,
        const offset_t *,
              offset_t  ,
              offset_t  ,
        const offset_t *,
        const offset_t *,
        const offset_t *,
        const offset_t *,
        const offset_t *
    );
    auto call = [&] (func_t func) {
        return func(
            nbatch,
            dist,
            nearest_vertex,
            coord,
            vertices,
            faces,
            size,
            nb_faces,
            nb_vertices,
            stride_dist,
            stride_nearest,
            stride_coord,
            stride_vertices,
            stride_faces
        );
    };

    if (_signed && !naive)
        return call(sdt         <ndim, scalar_t, index_t, offset_t>);
    else if (_signed && naive)
        return call(sdt_naive   <ndim, scalar_t, index_t, offset_t>);
    else if (!_signed && !naive)
        return call(udt         <ndim, scalar_t, index_t, offset_t>);
    else if (!_signed && naive)
        return call(udt_naive   <ndim, scalar_t, index_t, offset_t>);
}

FF_NAMESPACE_END(distance_mesh)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF)


#endif // FF_DISTANCE_MESH
