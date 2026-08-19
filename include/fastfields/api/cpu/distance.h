#ifndef FF_CPU_DISTANCE
#define FF_CPU_DISTANCE
#include "dlpack.h"
#include <cstdint>

namespace ff  {
namespace cpu {

void dt_euclidean(
          DLTensor & inp_out        ,
          double     voxel_spacing  = 1.0,
          int        stream         = 0
);
void dt_l1(
          DLTensor & inp_out        ,
          double     voxel_spacing  = 1.0,
          int        stream         = 0
);
void dt_spline_table(
          DLTensor & time           ,
          DLTensor & dist           ,
    const DLTensor & loc            ,
    const DLTensor & coeff          ,
    const DLTensor & times          ,
          int8_t     spline         = 3, // Cubic
          int8_t     bound          = 3, // DCT2
          int        stream         = 0
);
void dt_spline_brent(
          DLTensor & time           ,
          DLTensor & dist           ,
    const DLTensor & loc            ,
    const DLTensor & coeff          ,
          int64_t    max_iter       ,
          double     tol            ,
          double     step           ,
          int8_t     spline         = 3, // Cubic
          int8_t     bound          = 3, // DCT2
          int        stream         = 0
);
void dt_spline_gaussnewton(
          DLTensor & time           ,
          DLTensor & dist           ,
    const DLTensor & loc            ,
    const DLTensor & coeff          ,
          int64_t    max_iter       ,
          double     tol            ,
          int8_t     spline         = 3, // Cubic
          int8_t     bound          = 3, // DCT2
          int        stream         = 0
);
void dt_mesh(
          DLTensor & dist           ,
          DLTensor & nearest_vertex ,
    const DLTensor & loc            ,
    const DLTensor & vertices       ,
    const DLTensor & faces          ,
          bool       _signed        = true,
          bool       naive          = false,
          int        stream         = 0
);

} // namespace cpu
} // namespace ff

#endif // FF_CPU_DISTANCE
