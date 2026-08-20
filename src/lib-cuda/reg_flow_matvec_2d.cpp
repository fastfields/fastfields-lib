// One reg_flow instantiation slice: flow_matvec / flow_addmatvec_ / flow_submatvec_, 2D.
//
// The exported symbols live in reg_flow.cpp; this translation unit holds
// only the template instantiations that one arm of its `switch (ndim)`
// reaches. See reg_flow_slice.h for why the seam exists and
// reg_flow_slice.inl for the bodies these two defines select.

#define FF_FLOW_SLICE_MATVEC 1
#define FF_FLOW_SLICE_ND2 1

#include "reg_flow_slice.inl"
