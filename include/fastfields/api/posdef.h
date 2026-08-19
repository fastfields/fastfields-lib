#ifndef FF_LIB_POSDEF
#define FF_LIB_POSDEF
#include "fastfields/core/dlpack.h"
#include <cstdint>
#include "fastfields/core/defines.h"

FF_NAMESPACE_BEGIN(FF)

/**
 * Compact symmetric ("Sym") positive-definite matrix operations.
 *
 * The trailing dimension of a vector tensor holds the C channels; the
 * trailing dimension of a matrix tensor holds the C*(C+1)/2 unique entries
 * of a symmetric CxC matrix in a "diagonal-then-rows" layout:
 *     [ h00 h11 ... h(C-1)(C-1) | h01 h02 ... h0(C-1) | h12 ... ]
 * e.g. C=2 -> [h00, h11, h01]; C=3 -> [h00, h11, h22, h01, h02, h12].
 * Every leading dimension is treated as batch.
 */

/** out = H * inp */
void sym_matvec(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
          intptr_t   stream         = 0
);

/** Backward of matvec wrt the matrix: out (*batch, C*(C+1)/2) from grd, inp (*batch, C) */
void sym_matvec_backward(
          DLTensor & out            ,
    const DLTensor & grd            ,
    const DLTensor & inp            ,
          intptr_t   stream         = 0
);

/** out += H * inp */
void sym_addmatvec_(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
          intptr_t   stream         = 0
);

/** out -= H * inp */
void sym_submatvec_(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
          intptr_t   stream         = 0
);

/** out = (H + diag(weight)) \ inp   (weight optional: pass a null-data DLTensor) */
void sym_solve(
          DLTensor & out            ,
    const DLTensor & hessian        ,
    const DLTensor & inp            ,
    const DLTensor & weight         ,
          intptr_t   stream         = 0
);

/** inp_out = (H + diag(weight)) \ inp_out   (in place) */
void sym_solve_(
          DLTensor & inp_out        ,
    const DLTensor & hessian        ,
    const DLTensor & weight         ,
          intptr_t   stream         = 0
);

/** out = inv(H)   (both compact-symmetric) */
void sym_invert(
          DLTensor & out            ,
    const DLTensor & hessian        ,
          intptr_t   stream         = 0
);

/** hessian = inv(hessian)   (in place, compact-symmetric) */
void sym_invert_(
          DLTensor & hessian        ,
          intptr_t   stream         = 0
);

FF_NAMESPACE_END(FF)

#endif // FF_LIB_POSDEF
