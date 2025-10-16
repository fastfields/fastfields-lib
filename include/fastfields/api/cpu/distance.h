#ifndef FF_DISTANCE_CPU
#define FF_DISTANCE_CPU
#include "dlpack.h"
#include <cstdint>

namespace ff  {
namespace cpu {

void dt_euclidean(
          DLTensor & inp_out,
    const double   & voxel_spacing
);
void dt_l1(
          DLTensor & inp_out,
    const double   & voxel_spacing
);
void dt_spline_table(
          DLTensor & time,
          DLTensor & dist,
    const DLTensor & loc,
    const DLTensor & coeff,
    const DLTensor & times,
    const int8_t   & spline,
    const int8_t   & bound
);
void dt_spline_brent(
          DLTensor & time,
          DLTensor & dist,
    const DLTensor & loc,
    const DLTensor & coeff,
    const int64_t  & max_iter,
    const double   & tol,
    const double   & step,
    const int8_t   & spline,
    const int8_t   & bound
);
void dt_spline_gaussnewton(
          DLTensor & time,
          DLTensor & dist,
    const DLTensor & loc,
    const DLTensor & coeff,
    const int64_t  & max_iter,
    const double   & tol,
    const int8_t   & spline,
    const int8_t   & bound
);
void dt_mesh(
          DLTensor & dist,
          DLTensor & nearest_vertex,
    const DLTensor & loc,
    const DLTensor & vertices,
    const DLTensor & faces,
          bool       _signed = true,
          bool       naive   = false
);

} // namespace cpu
} // namespace ff

#endif // FF_DISTANCE_CPU
