#ifndef FF_POSDEF_CHOLESKY
#define FF_POSDEF_CHOLESKY
#include <fastfields/core/cuda_switch.h>
#include "../utils.h"
#include "utils.h"

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)
FF_NAMESPACE_BEGIN(posdef)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                             CHOLESKY
//
// ---------------------------------------------------------------------
//
// Internals that perform a Cholesky decomposition and solve a linear
// system from a Cholesky decomposition
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename offset_t=int64_t, int C=-1>
struct cholesky {

    template <
        typename ptr_t,
        typename reduce_t = internal::elem_type<ptr_t>
    >
    inline FF_CUDEV static void
    decompose_(
        ptr_t    a,
        reduce_t unused = static_cast<reduce_t>(0)
    )
    {
        return cholesky<offset_t>::decompose_(static_cast<offset_t>(C), a, unused);
    }

    template <
        typename aptr_t,
        typename xptr_t,
        typename reduce_t = internal::return_type<aptr_t, xptr_t>
    >
    inline FF_CUDEV static
    void solve_(
        aptr_t   a,
        xptr_t   x,
        reduce_t unused = static_cast<reduce_t>(0)
    )
    {
        return cholesky<offset_t>::solve_(static_cast<offset_t>(C), a, x, unused);
    }

}; // struct cholesky

template <typename offset_t>
struct cholesky<offset_t, -1> {

    /// In-place Cholesky decomposition (Cholesky–Banachiewicz)
    ///
    /// @param[in]     C:  (u)int
    /// @param[inout]  a:  CxC matrix
    ///
    /// https://en.wikipedia.org/wiki/Cholesky_decomposition
    template <
        typename ptr_t,
        typename reduce_t = internal::elem_type<ptr_t>
    >
    inline FF_CUDEV static
    void decompose_(
        offset_t C,
        ptr_t    a,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        using scalar_t = internal::elem_type<ptr_t>;

        reduce_t sm, sm0;
        sm0 = 1e-40;

#if 0
        for(offset_t c = 0; c < C; ++c)
            internal::iadd<reduce_t>(sm0, a[c*C+c]);
        sm0 *= 1e-7;
        sm0 *= sm0;
#endif

        for (offset_t c = 0; c < C; ++c) {
            for (offset_t b = c; b < C; ++b) {
                sm = static_cast<reduce_t>(a[c*C+b]);
                for(offset_t d = c-1; d >= 0; --d)
                    internal::isubcmul<reduce_t>(sm, a[c*C+d], a[b*C+d]);
                if (c == b)
                    a[c*C+c] = static_cast<scalar_t>(sqrt(max(sm, sm0)));
                else
                    internal::div<reduce_t>(a[b*C+c], sm, a[c*C+c]);
            }
        }
    }

    /// Cholesky solver (inplace)
    ///
    /// @param[in]    C:  (u)int
    /// @param[in]    a:  CxC matrix (output from `decompose`)
    /// @param[inout] x:  C vector
    template <
        typename aptr_t,
        typename xptr_t,
        typename reduce_t =
            internal::return_type<aptr_t, xptr_t>
        >
    inline FF_CUDEV static
    void solve_(
        offset_t C,
        aptr_t a,
        xptr_t x,
        reduce_t /*unused*/ = static_cast<reduce_t>(0))
    {
        using scalar_t = internal::elem_type<xptr_t>;

        reduce_t sm;
        for (offset_t c = 0; c < C; ++c)
        {
            sm = static_cast<reduce_t>(x[c]);
            for (offset_t cc = c-1; cc >= 0; --cc)
                internal::isubcmul<reduce_t>(sm, a[c*C+cc], x[cc]);
            internal::div<reduce_t>(x[c], sm, a[c*C+c]);
        }
        for (offset_t c = C-1; c >= 0; --c)
        {
            sm = static_cast<reduce_t>(x[c]);
            for(offset_t cc = c+1; cc < C; ++cc)
                internal::isubcmul<reduce_t>(sm, a[cc*C+c], x[cc]);
            internal::div<reduce_t>(x[c], sm, a[c*C+c]);
        }
    }

}; // struct cholesky

FF_NAMESPACE_END(posdef)
FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)

#endif // FF_POSDEF_CHOLESKY
