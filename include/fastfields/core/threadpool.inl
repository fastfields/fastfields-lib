#ifndef FF_THREADPOOL_INL
#define FF_THREADPOOL_INL
#include <memory>
#include <string>
#include <thread>
#include <fastfields/core/defines.h>

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(internal)

    // Some of this is copied from pytorch/aten

    // NOTE: everything in this header is `inline` (functions) or wrapped in a
    // Meyers singleton (the two pieces of mutable global state). This header is
    // included by every module translation unit; without inline linkage the
    // functions and globals would be multiply-defined once a library contains
    // more than one module object file. C++11 has no inline variables, hence
    // the function-local statics below rather than namespace-scope globals.

    inline size_t get_env_num_threads(const char* var_name, size_t def_value = 0) {
        try {
            if (auto* value = std::getenv(var_name)) {
                size_t nthreads = static_cast<size_t>(std::stoi(value));
                if (nthreads) return nthreads;
            }
        } catch (const std::exception& e) {}
        return def_value;
    }

    inline size_t default_num_threads_from_hardware() {
        auto num_threads = std::thread::hardware_concurrency();
#       if defined(_M_X64) || defined(__x86_64__)
            num_threads /= 2;
#       endif
        return num_threads;
    }

    inline size_t default_num_threads() {
        size_t nthreads = get_env_num_threads("FF_NUM_THREADS", 0);
        if (nthreads == 0) nthreads = get_env_num_threads("OMP_NUM_THREADS", nthreads);
        if (nthreads == 0) nthreads = get_env_num_threads("MKL_NUM_THREADS", nthreads);
        if (nthreads == 0) nthreads = default_num_threads_from_hardware();
        if (nthreads == 0) nthreads = 1;
        return nthreads;
    }

    inline int& num_threads() {
        static int value = static_cast<int>(default_num_threads());
        return value;
    }
    inline std::shared_ptr<ThreadPool>& global_pool() {
        static std::shared_ptr<ThreadPool> value(nullptr);
        return value;
    }

FF_NAMESPACE_END(internal)

inline size_t set_num_threads(size_t nthreads) {
    if (nthreads == 0) nthreads = 1;
    size_t old_num_threads = internal::num_threads();
    internal::num_threads() = static_cast<int>(nthreads);
    if (old_num_threads != static_cast<size_t>(internal::num_threads()))
        internal::global_pool().reset(new ThreadPool(internal::num_threads()));
    return internal::num_threads();
}

inline size_t get_num_threads() {
    return internal::num_threads();
}

inline std::shared_ptr<ThreadPool> get_global_pool() {
    if (!internal::global_pool())
        internal::global_pool().reset(new ThreadPool(internal::num_threads()));
    return internal::global_pool();
}

FF_NAMESPACE_END(FF_NS)
#endif // FF_THREADPOOL_INL
