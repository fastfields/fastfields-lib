#ifndef FF_CPU_POSDEF
#define FF_CPU_POSDEF
#include "dlpack.h"
#include <cstdint>

namespace ff  {
namespace cpu {

// All tensors below store the "value" axis in their last dimension.
//   - vectors (inp / out / grd / wgt) have a trailing axis of length C
//   - compact symmetric matrices (hessian) have a trailing axis of length
//     C*(C+1)/2, stored in a "diagonal-then-rows" layout:
//         [ h00 h11 h22 ... | h01 h02 ... h0(C-1) | h12 ... ]
//     e.g. C=2 -> [h00, h11, h01]; C=3 -> [h00, h11, h22, h01, h02, h12].
// The leading dimensions are treated as batch and iterated in parallel.

// out = H * inp
void sym_matvec(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
          int        stream         = 0
);

// out = d(g' H x)/dH  (compact-symmetric gradient of matvec wrt the matrix)
// out : (*batch, C*(C+1)/2) ; grd, inp : (*batch, C)
void sym_matvec_backward(
          DLTensor & out            ,
    const DLTensor & grd            ,
    const DLTensor & inp            ,
          int        stream         = 0
);

// out += H * inp
void sym_addmatvec_(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
          int        stream         = 0
);

// out -= H * inp
void sym_submatvec_(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
          int        stream         = 0
);

// out = (H + diag(weight)) \ inp
// weight is optional: pass a DLTensor whose .data is null to disable it.
void sym_solve(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
    const DLTensor & weight         ,
          int        stream         = 0
);

// inp_out = (H + diag(weight)) \ inp_out   (in place)
void sym_solve_(
          DLTensor & inp_out        ,
    const DLTensor & hessian        ,
    const DLTensor & weight         ,
          int        stream         = 0
);

// out = inv(H)   (both in compact-symmetric layout)
void sym_invert(
          DLTensor & out            ,
    const DLTensor & hessian        ,
          int        stream         = 0
);

// hessian = inv(hessian)   (in place, compact-symmetric layout)
void sym_invert_(
          DLTensor & hessian        ,
          int        stream         = 0
);

} // namespace cpu
} // namespace ff

#endif // FF_CPU_POSDEF
