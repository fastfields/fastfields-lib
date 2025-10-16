#include <exception>
#include "distance.h"
#include "dlpack.h"
#include "impl/kernels/cuda_switch.h"
#include "impl/distance_euclidean.hpp"
#include "impl/distance_l1.hpp"
#include "impl/distance_spline.hpp"
#include "impl/distance_mesh.hpp"

using namespace FF;
using namespace FF::FF_DEVICE;

/***********************************************************************
 *                              CHECKS                                 *
 ***********************************************************************/

#define CHECK_NO_LANES(tensor)                                          \
    if (tensor.dtype.lanes > 1)                                         \
        throw std::invalid_argument(                                    \
            "Only scalar data types are supported"                      \
        );

#define CHECK_SAME(X, Y, msg)                                           \
    if (X != Y) throw std::invalid_argument(msg);

#define CHECK_SAME_BATCH(X, Y, D)                                       \
    if (X.ndim < D || Y.ndim < D)                                       \
        throw std::invalid_argument(                                    \
            "Number of dimensions does not match"                       \
        );                                                              \
    for (int32_t d=0; d < D; ++d)                                       \
        if (X.shape[d] != Y.shape[d])                                   \
            throw std::invalid_argument(                                \
                "Tensors do not have the same batch shape"              \
            );                                                          \

#define CHECK_SAME_SHAPE(X, Y)                                          \
    if (X.ndim != Y.ndim)                                               \
        throw std::invalid_argument(                                    \
            "Tensors do not have the same number of dimensions"         \
        );                                                              \
    CHECK_SAME_BATCH(X, Y, X.ndim)

#define CHECK_SAME_DTYPE(X, Y)                                          \
    if (                                                                \
        (X.dtype.code  != Y.dtype.code) ||                              \
        (X.dtype.bits  != Y.dtype.bits) ||                              \
        (X.dtype.lanes != Y.dtype.lanes)                                \
    )                                                                   \
        throw std::invalid_argument(                                    \
            "Tensors do not have the same data type"                    \
        );

/***********************************************************************
 *                              EUCLIDEAN                              *
 ***********************************************************************/

#define DISPATCH_DT(autocast, func, args...)                            \
{                                                                       \
    DLDataTypeCode code = static_cast<DLDataTypeCode>(inp_out.dtype.code); \
    switch (code) {                                                     \
        case kDLFloat: switch (inp_out.dtype.bits) {                    \
            case 32: return autocast<float>::func(args);                \
            case 64: return autocast<double>::func(args);               \
        };                                                              \
        throw std::invalid_argument(                                    \
            "only floating point data types are supported"              \
        );                                                              \
    };                                                                  \
}

void dt_euclidean(
          DLTensor & inp_out,
    const double   & voxel_spacing
)
{
    CHECK_NO_LANES(inp_out)
    DISPATCH_DT(
        FF_DEVICE::distance_e::AutoCast,
        dt,
        inp_out.ndim,
        inp_out.data + inp_out.byte_offset,
        voxel_spacing,
        inp_out.shape,
        inp_out.strides
    )
}

void dt_l1(
          DLTensor & inp_out,
    const double   & voxel_spacing
)
{
    CHECK_NO_LANES(inp_out)
    DISPATCH_DT(
        FF_DEVICE::distance_l1::AutoCast,
        dt,
        inp_out.ndim,
        inp_out.data + inp_out.byte_offset,
        voxel_spacing,
        inp_out.shape,
        inp_out.strides
    )
}

/***********************************************************************
 *                              SPLINE                                 *
 ***********************************************************************/

#define DISPATCH_SPLINE_DIM(D, autocast, func, args...)                 \
    switch (code) {                                                     \
        case kDLFloat: switch (loc.dtype.bits) {                        \
            case 32: return autocast<float>::func<D,S,B>(args);         \
            case 64: return autocast<double>::func<D,S,B>(args);        \
        };                                                              \
        throw std::invalid_argument(                                    \
            "Only floating point data types are supported"              \
        );                                                              \
    }

#define DISPATCH_SPLINE(autocast, func, args...)                        \
{                                                                       \
    static const bound_t  B    = bound_t::Dynamic;                      \
    static const spline_t S    = spline_t::Dynamic;                     \
    const int32_t         ndim = loc.shape[loc.ndim-1];                 \
    const DLDataTypeCode  code = static_cast<DLDataTypeCode>(loc.dtype.code); \
    switch (ndim) {                                                     \
        case 1: DISPATCH_SPLINE_DIM(1, autocast, func, args);           \
        case 2: DISPATCH_SPLINE_DIM(2, autocast, func, args);           \
        case 3: DISPATCH_SPLINE_DIM(2, autocast, func, args);           \
        throw std::invalid_argument(                                    \
            "Only 1D, 2D and 3D splines are supported"                  \
        );                                                              \
    };                                                                  \
}

void dt_spline_table(
          DLTensor & time,
          DLTensor & dist,
    const DLTensor & loc,
    const DLTensor & coeff,
    const DLTensor & times,
    const spline_t & spline,
    const bound_t  & bound
)
{
    const int32_t ndim   = loc.shape[loc.ndim-1];
    const int32_t nbatch = loc.ndim - 1;
    CHECK_NO_LANES  (time)
    CHECK_SAME_DTYPE(time, dist)
    CHECK_SAME_DTYPE(time, loc)
    CHECK_SAME_DTYPE(time, coeff)
    CHECK_SAME_DTYPE(time, times)
    CHECK_SAME      (time.ndim,  nbatch,   "Number of batch dimensions does not match")
    CHECK_SAME      (dist.ndim,  nbatch,   "Number of batch dimensions does not match")
    CHECK_SAME      (coeff.ndim, nbatch+1, "Number of batch dimensions does not match")
    CHECK_SAME      (times.ndim, nbatch+1, "Number of batch dimensions does not match")
    CHECK_SAME_BATCH(loc, time,  nbatch)
    CHECK_SAME_BATCH(loc, dist,  nbatch)
    CHECK_SAME_BATCH(loc, coeff, nbatch)
    CHECK_SAME_BATCH(loc, times, nbatch)

    DISPATCH_SPLINE(
        FF_DEVICE::distance_spline::AutoCast,
        mindist_table,
        nbatch,                             // nbatch
        time.data  + time.byte_offset,      // time
        dist.data  + dist.byte_offset,      // dist
        loc.data   + loc.byte_offset,       // loc
        coeff.data + coeff.byte_offset,     // coeff
        times.data + times.byte_offset,     // times
        times.shape[times.ndim-1],          // ntimes
        loc.shape,                          // size
        time.strides,                       // stride_time
        dist.strides,                       // stride_dist
        loc.strides,                        // stride_loc
        coeff.strides,                      // stride_coeff
        times.strides,                      // stride_times
        spline,
        bound
    )
}

void dt_spline_brent(
          DLTensor & time,
          DLTensor & dist,
    const DLTensor & loc,
    const DLTensor & coeff,
    const long     & max_iter,
    const double   & tol,
    const double   & step,
    const spline_t & spline,
    const bound_t  & bound
)
{
    const int32_t ndim   = loc.shape[loc.ndim-1];
    const int32_t nbatch = loc.ndim - 1;
    CHECK_NO_LANES  (time)
    CHECK_SAME_DTYPE(time, dist)
    CHECK_SAME_DTYPE(time, loc)
    CHECK_SAME_DTYPE(time, coeff)
    CHECK_SAME      (time.ndim,  nbatch,   "Number of batch dimensions does not match")
    CHECK_SAME      (dist.ndim,  nbatch,   "Number of batch dimensions does not match")
    CHECK_SAME      (coeff.ndim, nbatch+1, "Number of batch dimensions does not match")
    CHECK_SAME_BATCH(loc, time,  nbatch)
    CHECK_SAME_BATCH(loc, dist,  nbatch)
    CHECK_SAME_BATCH(loc, coeff, nbatch)

    DISPATCH_SPLINE(
        FF_DEVICE::distance_spline::AutoCast,
        mindist_brent,
        nbatch,                             // nbatch
        time.data  + time.byte_offset,      // time
        dist.data  + dist.byte_offset,      // dist
        loc.data   + loc.byte_offset,       // loc
        coeff.data + coeff.byte_offset,     // coeff
        loc.shape,                          // size
        time.strides,                       // stride_time
        dist.strides,                       // stride_dist
        loc.strides,                        // stride_loc
        coeff.strides,                      // stride_coeff
        max_iter,
        tol,
        step,
        spline,
        bound
    )
}

void dt_spline_gaussnewton(
          DLTensor & time,
          DLTensor & dist,
    const DLTensor & loc,
    const DLTensor & coeff,
    const long     & max_iter,
    const double   & tol,
    const spline_t & spline,
    const bound_t  & bound
)
{
    const int32_t ndim   = loc.shape[loc.ndim-1];
    const int32_t nbatch = loc.ndim - 1;
    CHECK_NO_LANES  (time)
    CHECK_SAME_DTYPE(time, dist)
    CHECK_SAME_DTYPE(time, loc)
    CHECK_SAME_DTYPE(time, coeff)
    CHECK_SAME      (time.ndim,  nbatch,   "Number of batch dimensions does not match")
    CHECK_SAME      (dist.ndim,  nbatch,   "Number of batch dimensions does not match")
    CHECK_SAME      (coeff.ndim, nbatch+1, "Number of batch dimensions does not match")
    CHECK_SAME_BATCH(loc, time,  nbatch)
    CHECK_SAME_BATCH(loc, dist,  nbatch)
    CHECK_SAME_BATCH(loc, coeff, nbatch)

    DISPATCH_SPLINE(
        FF_DEVICE::distance_spline::AutoCast,
        mindist_gaussnewton,
        nbatch,                             // nbatch
        time.data  + time.byte_offset,      // time
        dist.data  + dist.byte_offset,      // dist
        loc.data   + loc.byte_offset,       // loc
        coeff.data + coeff.byte_offset,     // coeff
        loc.shape,                          // size
        time.strides,                       // stride_time
        dist.strides,                       // stride_dist
        loc.strides,                        // stride_loc
        coeff.strides,                      // stride_coeff
        max_iter,
        tol,
        spline,
        bound
    )
}

/***********************************************************************
 *                                MESH                                 *
 ***********************************************************************/

#define DISPATCH_MESH_SCALAR(D, S, autocast, func, args...)         \
    switch (code_index) {                                               \
        case kDLInt: switch (loc.dtype.bits) {                          \
            case  8: return autocast<S,int8_t >::func<D>(args);         \
            case 16: return autocast<S,int16_t>::func<D>(args);         \
            case 32: return autocast<S,int32_t>::func<D>(args);         \
            case 64: return autocast<S,int64_t>::func<D>(args);         \
        };                                                              \
        case kDLUInt: switch (loc.dtype.bits) {                         \
            case  8: return autocast<S,uint8_t >::func<D>(args);        \
            case 16: return autocast<S,uint16_t>::func<D>(args);        \
            case 32: return autocast<S,uint32_t>::func<D>(args);        \
            case 64: return autocast<S,uint64_t>::func<D>(args);        \
        };                                                              \
        throw std::invalid_argument(                                    \
            "Only integer data types are supported"                     \
        );                                                              \
    }

#define DISPATCH_MESH_DIM(D, autocast, func, args...)                   \
    switch (code_scalar) {                                              \
        case kDLFloat: switch (loc.dtype.bits) {                        \
            case 32: DISPATCH_MESH_SCALAR(D, float,  autocast, func, args); \
            case 64: DISPATCH_MESH_SCALAR(D, double, autocast, func, args);  \
        };                                                              \
        throw std::invalid_argument(                                    \
            "Only floating point data types are supported"              \
        );                                                              \
    }

#define DISPATCH_MESH(autocast, func, args...)                          \
{                                                                       \
    const int32_t         ndim        = loc.shape[loc.ndim-1];          \
    const DLDataTypeCode  code_scalar = static_cast<DLDataTypeCode>(vertices.dtype.code); \
    const DLDataTypeCode  code_index  = static_cast<DLDataTypeCode>(faces.dtype.code);    \
    switch (ndim) {                                                     \
        case 2: DISPATCH_MESH_DIM(2, autocast, func, args);             \
        case 3: DISPATCH_MESH_DIM(3, autocast, func, args);             \
        throw std::invalid_argument(                                    \
            "Only 1D, 2D and 3D splines are supported"                  \
        );                                                              \
    };                                                                  \
}

void dt_mesh(
          DLTensor & dist,
          DLTensor & nearest_vertex,
    const DLTensor & loc,
    const DLTensor & vertices,
    const DLTensor & faces,
          bool       _signed = true,
          bool       naive   = false
)
{
    const int32_t ndim   = loc.shape[loc.ndim-1];
    const int32_t nbatch = loc.ndim - 1;
    CHECK_NO_LANES  (dist)
    CHECK_SAME_DTYPE(dist,  loc)
    CHECK_SAME_DTYPE(dist,  vertices)
    CHECK_SAME      (          dist.ndim, nbatch,   "Number of batch dimensions does not match")
    CHECK_SAME      (      vertices.ndim, nbatch+1, "Number of batch dimensions does not match")
    CHECK_SAME      (         faces.ndim, nbatch+1, "Number of batch dimensions does not match")
    CHECK_SAME_BATCH(loc, dist,           nbatch)
    CHECK_SAME_BATCH(loc, vertices,       nbatch)
    CHECK_SAME_BATCH(loc, faces,          nbatch)
    CHECK_SAME      (vertices.shape[vertices.ndim-1], ndim, "Dimensionality of the vertices and location does not match")
    CHECK_SAME      (faces.shape[faces.ndim-1],       ndim, "Dimensionality of the vertices and faces does not match")

    if (nearest_vertex.data)
    {
        CHECK_SAME_DTYPE(faces, nearest_vertex)
        CHECK_SAME      (nearest_vertex.ndim, nbatch,   "Number of batch dimensions does not match")
        CHECK_SAME_BATCH(loc, nearest_vertex, nbatch)
    }

    DISPATCH_MESH(
        FF_DEVICE::distance_mesh::AutoCast,
        dt,
        nbatch,                                             // nbatch
        dist.data           + dist.byte_offset,             // data
        nearest_vertex.data + nearest_vertex.byte_offset,   // nearest_vertex
        loc.data            + loc.byte_offset,              // coord
        vertices.data       + vertices.byte_offset,         // vertices
        faces.data          + faces.byte_offset,            // faces
        loc.shape,                                          // size
        faces.shape[faces.ndim-1],                          // nb_faces
        vertices.shape[vertices.ndim-1],                    // nb_vertices
        dist.strides,                                       // stride_dist
        nearest_vertex.strides,                             // stride_nearest
        loc.strides,                                        // stride_coord
        vertices.strides,                                   // stride_vertices
        faces.strides,                                      // stride_faces
        _signed,                                            // signed
        naive                                               // naive
    )
}
