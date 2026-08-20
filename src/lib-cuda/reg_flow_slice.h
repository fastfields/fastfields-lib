#pragma once
/**
 * `reg_flow`'s internal seam: the boundary between the exported entry points
 * (`reg_flow.cpp`, which instantiates nothing) and the template instantiations
 * (`reg_flow_<group>_<n>d.cpp`, one translation unit per slice).
 *
 * WHY THIS EXISTS
 * ----------------------------------------------------------------------
 * `reg_flow` is the most expensive translation unit in the project: nvcc peaks
 * at 12.98 GB compiling it, 81% of a 16 GB CI runner, and it is one
 * indivisible TU (see the measured table above MODULES in the Makefile). That
 * single number is what pins the CUDA build at `-j2`, and `-j2` is what makes
 * a multi-architecture `-gencode` set unaffordable against the 120-minute
 * timeout.
 *
 * The cost is template instantiation, and it factors cleanly. Each exported
 * entry point selects exactly one internal wrapper template and dispatches it
 * over ndim x boundary x dtype x offset width. Nothing is shared between one
 * entry point's leaves and another's, so a TU that only ever calls
 * `_flow_diag<2, ...>` pays for `_flow_diag<2, ...>` and nothing else. Cutting
 * the file along those seams cuts the instantiation set the same way.
 *
 * WHY A FORWARDING SEAM AND NOT `-D` ON ONE SOURCE
 * ----------------------------------------------------------------------
 * The obvious cheap trick -- compile `reg_flow.cpp` N times with different
 * `-D` flags, the way BOUNDFLAGS already varies a build -- produces N objects
 * that each define `ff::cuda::flow_matvec`, and the link fails on duplicate
 * symbols. So the exported entry point has to live in exactly one TU, and the
 * per-slice TUs have to export something else.
 *
 * That something else is declared here: one ordinary function per
 * (entry point, ndim), named for the slice it covers. `reg_flow.cpp` keeps
 * every exported symbol, every argument check and every error message, does
 * its `switch (ndim)`, and calls one of these. Each slice TU defines the ones
 * it owns, and instantiates only what those need.
 *
 * The alternative -- `extern template` declarations here with explicit
 * instantiation definitions in the slice TUs -- was rejected. It has to name
 * every leaf, and the leaves are not knowable from the source: with
 * `FF_STATIC_BOUNDS=0` six of the eight `FF_BOUND_*` selectors collapse onto
 * `bound::type::Dynamic`, so the enumeration would contain the same
 * specialization six times, and explicitly instantiating one specialization
 * more than once is ill-formed ([temp.explicit]/5). Which leaves collapse is a
 * build-flag decision, so the enumeration would have to be written differently
 * per BOUNDFLAGS setting. A `switch` has no such problem: duplicate
 * *implicit* instantiations across its arms are simply the same instantiation.
 *
 * THE SYMBOLS DECLARED HERE ARE NOT ABI
 * ----------------------------------------------------------------------
 * They are hidden-visibility, so `libfastfields-cuda.so`'s dynamic symbol
 * table is byte-identical to what it was before the split -- verify with
 * `nm -D --defined-only`. This header is private to `src/lib-cuda/`; it is not
 * installed and nothing outside this directory may include it.
 */

#include <cstdint>
#include <fastfields/core/dlpack.h>

// Internal linkage would defeat the purpose (the definition and the call are
// in different TUs), but these must not reach the .so's dynamic symbol table:
// the exported ABI is `ff::cuda::flow_*` and nothing else. Hidden visibility
// is the difference. Non-GNU toolchains fall back to ordinary external
// linkage, which costs a few exported symbols and nothing else -- nothing
// implements the Windows build (see make/common.mk), so this only keeps the
// door open the same way that block does.
#if defined(__GNUC__) || defined(__clang__)
#  define FF_FLOW_SLICE_HIDDEN __attribute__((visibility("hidden")))
#else
#  define FF_FLOW_SLICE_HIDDEN
#endif

/***********************************************************************
 *                          SLICE SIGNATURES                           *
 ***********************************************************************/

// The arguments a slice needs that it cannot cheaply re-derive. Everything
// the entry point validated -- rank, channel count, dtype agreement -- has
// already been checked by the time a slice is called, and `nbatch` is passed
// rather than recomputed so that the rank arithmetic still happens in exactly
// one place.
//
// Spelled as macros so the declaration here and the definition in
// `reg_flow_slice.inl` cannot drift: there is one text for each shape.

#define FF_FLOW_SLICE_SIG_MATVEC(NAME)                                        \
    void NAME(                                                                \
              DLTensor & out        ,                                         \
        const DLTensor & inp        ,                                         \
        const double   * voxel_size ,                                         \
              double     absolute   ,                                         \
              double     membrane   ,                                         \
              double     bending    ,                                         \
              double     shears     ,                                         \
              double     div        ,                                         \
              int8_t     bound      ,                                         \
              int32_t    nbatch     ,                                         \
              bool       use_32bits ,                                         \
              intptr_t   stream     )

#define FF_FLOW_SLICE_SIG_DIAG(NAME)                                          \
    void NAME(                                                                \
              DLTensor & out        ,                                         \
        const double   * voxel_size ,                                         \
              double     absolute   ,                                         \
              double     membrane   ,                                         \
              double     bending    ,                                         \
              double     shears     ,                                         \
              double     div        ,                                         \
              int8_t     bound      ,                                         \
              int32_t    nbatch     ,                                         \
              bool       use_32bits ,                                         \
              intptr_t   stream     )

// Same shape as DIAG; named separately because the two families' arguments
// mean different things (`out` is the operator diagonal versus its Toeplitz
// stencil) and are free to diverge.
#define FF_FLOW_SLICE_SIG_KERNEL(NAME) FF_FLOW_SLICE_SIG_DIAG(NAME)

#define FF_FLOW_SLICE_SIG_RELAX(NAME)                                         \
    void NAME(                                                                \
              DLTensor & sol        ,                                         \
        const DLTensor & hes        ,                                         \
        const DLTensor & grd        ,                                         \
        const double   * voxel_size ,                                         \
              double     absolute   ,                                         \
              double     membrane   ,                                         \
              double     bending    ,                                         \
              double     shears     ,                                         \
              double     div        ,                                         \
              int8_t     bound      ,                                         \
              int        nb_iter    ,                                         \
              int32_t    nbatch     ,                                         \
              bool       use_32bits ,                                         \
              intptr_t   stream     )

namespace ff {
namespace cuda {
namespace flow_slice {

// One declaration per (entry point, ndim). The `add`/`sub` prefixes are the
// `op` template argument ('+' / '-') the corresponding exported entry point
// passes; it is part of the slice's identity rather than a runtime argument
// precisely so that a build may put each op in its own TU.
#define FF_FLOW_SLICE_DECL_ND(ND)                                             \
    FF_FLOW_SLICE_HIDDEN FF_FLOW_SLICE_SIG_MATVEC(matvec_##ND##d);            \
    FF_FLOW_SLICE_HIDDEN FF_FLOW_SLICE_SIG_MATVEC(addmatvec_##ND##d);         \
    FF_FLOW_SLICE_HIDDEN FF_FLOW_SLICE_SIG_MATVEC(submatvec_##ND##d);         \
    FF_FLOW_SLICE_HIDDEN FF_FLOW_SLICE_SIG_DIAG  (diag_##ND##d);              \
    FF_FLOW_SLICE_HIDDEN FF_FLOW_SLICE_SIG_DIAG  (adddiag_##ND##d);           \
    FF_FLOW_SLICE_HIDDEN FF_FLOW_SLICE_SIG_DIAG  (subdiag_##ND##d);           \
    FF_FLOW_SLICE_HIDDEN FF_FLOW_SLICE_SIG_KERNEL(kernel_##ND##d);            \
    FF_FLOW_SLICE_HIDDEN FF_FLOW_SLICE_SIG_KERNEL(addkernel_##ND##d);         \
    FF_FLOW_SLICE_HIDDEN FF_FLOW_SLICE_SIG_KERNEL(subkernel_##ND##d);         \
    FF_FLOW_SLICE_HIDDEN FF_FLOW_SLICE_SIG_RELAX (relax_##ND##d);

FF_FLOW_SLICE_DECL_ND(1)
FF_FLOW_SLICE_DECL_ND(2)
FF_FLOW_SLICE_DECL_ND(3)

#undef FF_FLOW_SLICE_DECL_ND

} // namespace flow_slice
} // namespace cuda
} // namespace ff
