#pragma once
#include <fastfields/core/cuda_switch.h>
#include <cstdio>       // std::snprintf
#include <cstdlib>      // std::getenv
#include <cstring>      // std::strcmp
#include <stdexcept>    // std::runtime_error
#include <string>       // std::string
#include <utility>      // std::forward

/***********************************************************************
 *                                                                     *
 *                    THE ONE CHECKED KERNEL LAUNCH                    *
 *                                                                     *
 ***********************************************************************
 *
 * A CUDA kernel launch is asynchronous and does not throw. It reports
 * failure by setting the runtime's error state, which somebody has to go
 * and look at. Nothing in this tree ever did: there were 39 `<<<` sites
 * and zero `cudaGetLastError` / `cudaPeekAtLastError` calls, so a kernel
 * that never ran was indistinguishable from one that ran correctly --
 * the launcher returned normally and the caller read whatever was
 * already in the output buffer (fastfields-lib#152).
 *
 * The fix is structural rather than conventional. `<<<` appears in
 * exactly ONE place in this repository -- `launchKernel` below -- and
 * every launcher goes through the `FF_CUDA_LAUNCH` macro that wraps it.
 * `tools/check-cuda-launches.py --check` fails the build if a `<<<`
 * shows up anywhere else, which is what makes this survive contact with
 * future patches: a check nobody *can* bypass beats one people have to
 * remember. (The same reasoning as `-Wl,--no-undefined` in
 * make/common.mk and the `FF_MEM_BUDGET_KB` gate in CI; #152 exists
 * precisely because "remember to check" was never applied even once.)
 *
 * WHAT THIS COSTS AT RUNTIME: nothing measurable, and in particular no
 * device synchronisation. `cudaGetLastError` is a host-side read of a
 * thread-local error word that the driver wrote while enqueueing the
 * launch; it does not touch the device and does not wait for the kernel.
 * The stream ordering of the calling code is completely unchanged.
 *
 * WHAT IT THEREFORE CANNOT SEE: errors raised by the kernel while it
 * *executes* (illegal address, misaligned access, device-side assert).
 * Those only become observable at a synchronisation point, and
 * synchronising after every launch would serialise the pipeline and
 * destroy exactly the asynchrony the design depends on. So that is an
 * opt-in debug mode, off by default -- see `FF_CUDA_LAUNCH_SYNC` below,
 * which is this project's `CUDA_LAUNCH_BLOCKING`.
 */

FF_NAMESPACE_BEGIN(FF_NS)
FF_NAMESPACE_BEGIN(FF_DEVICE)

/***********************************************************************
 *                   OPT-IN POST-LAUNCH SYNCHRONISATION                *
 ***********************************************************************/

// Compile-time default for the debug sync described above. It is 0, and it
// must stay 0: turning it on inserts a `cudaStreamSynchronize` after every
// single launch, which makes every launcher blocking and removes any overlap
// between the host and the device. It exists so that a debug build can be
// produced with `-DFF_CUDA_LAUNCH_SYNC=1` without editing sources.
#ifndef FF_CUDA_LAUNCH_SYNC
#  define FF_CUDA_LAUNCH_SYNC 0
#endif

// Runtime override, read from the environment variable of the same name.
// "0", "false" and the empty value mean off; anything else means on. The
// environment wins over the compile-time default in both directions, so a
// release build can be asked for the diagnosis once without a rebuild:
//
//     FF_CUDA_LAUNCH_SYNC=1 python -c "import fastfields; ..."
//
// This is deliberately the same shape as the CUDA runtime's own
// CUDA_LAUNCH_BLOCKING, which is the knob a CUDA programmer already reaches
// for. The two compose: CUDA_LAUNCH_BLOCKING makes the launch itself
// synchronous, this makes the *error check* synchronous.
FF_CUHOST inline bool _launchSyncFromEnv()
{
    const char * v = std::getenv("FF_CUDA_LAUNCH_SYNC");
    if (!v || !*v)                return FF_CUDA_LAUNCH_SYNC != 0;
    if (std::strcmp(v, "0") == 0) return false;
    if (std::strcmp(v, "false") == 0) return false;
    return true;
}

// C++11 guarantees the initialisation of a function-local static is run once
// and is thread-safe, so the getenv is paid once per process rather than once
// per launch. (Namespace-scope state in a header is what fastfields-kernels'
// threadpool bug was, hence the accessor rather than a variable.)
FF_CUHOST inline bool launchSyncEnabled()
{
    static const bool enabled = _launchSyncFromEnv();
    return enabled;
}

/***********************************************************************
 *                  STICKY vs. NON-STICKY CLASSIFICATION               *
 ***********************************************************************/

// The difference matters enormously to whoever reads the report, and it is
// not visible from the error string:
//
//   * a launch-CONFIGURATION error is raised before the kernel starts. The
//     context is untouched, the next launch can succeed, and the fix is on
//     the caller's side ("your block size is too big for this kernel").
//   * an EXECUTION error poisons the CUDA context. Every subsequent CUDA
//     call in the process -- including in code that has nothing to do with
//     fastfields -- returns the same error until the process exits. The
//     honest report is "this process is dead", not "retry with less".
//
// Anything not on either list is reported as unclassified rather than
// guessed at; a wrong claim about stickiness is worse than no claim.
enum class LaunchErrorClass { Configuration, Execution, Unclassified };

FF_CUHOST inline LaunchErrorClass launchErrorClass(cudaError_t err)
{
    switch (err)
    {
        // --- rejected before the kernel ran: recoverable ----------------
        case cudaErrorInvalidConfiguration:     // grid/block out of range
        case cudaErrorLaunchOutOfResources:     // registers or shared mem
        case cudaErrorInvalidDeviceFunction:    // no such kernel on device
        case cudaErrorNoKernelImageForDevice:   // no cubin for this arch
        case cudaErrorInvalidValue:
        case cudaErrorInvalidDevice:
        case cudaErrorInvalidPtx:
        case cudaErrorUnsupportedPtxVersion:
        // Not launch *configuration* so much as "there was nothing to launch
        // on", but they belong on this side of the line all the same: no
        // kernel ran, so no kernel poisoned anything.
        case cudaErrorNoDevice:
        case cudaErrorInsufficientDriver:
            return LaunchErrorClass::Configuration;

        // --- raised by the running kernel: sticky ------------------------
        case cudaErrorLaunchFailure:
        case cudaErrorIllegalAddress:
        case cudaErrorMisalignedAddress:
        case cudaErrorIllegalInstruction:
        case cudaErrorInvalidAddressSpace:
        case cudaErrorInvalidPc:
        case cudaErrorHardwareStackError:
        case cudaErrorLaunchTimeout:
        case cudaErrorAssert:
        case cudaErrorECCUncorrectable:
            return LaunchErrorClass::Execution;

        default:
            return LaunchErrorClass::Unclassified;
    }
}

/***********************************************************************
 *                          THE ERROR REPORT                           *
 ***********************************************************************/

// Deliberately NOT a template: it is called from the cold path of a function
// template that is instantiated once per (kernel signature, dtype, offset
// type, ndim, spline, bound) combination, and none of this text needs to be
// duplicated across those. Everything kernel-specific arrives as an argument.
//
// `max_threads` / `num_regs` are `cudaFuncGetAttributes` results, or -1 when
// the query itself failed. They are what turns "too many resources requested
// for launch" into an actionable sentence.
FF_CUHOST inline void _throwLaunchError(
    const char  * name,
    dim3          grid,
    dim3          block,
    size_t        shared_mem,
    cudaStream_t  stream,
    cudaError_t   err,
    bool          preexisting,
    bool          from_sync,
    int           max_threads,
    int           num_regs)
{
    // `#KERNEL` stringises a parenthesised template-id (see FF_CUDA_LAUNCH),
    // so strip the wrapping parentheses for readability.
    std::string kernel(name ? name : "<unknown>");
    if (kernel.size() >= 2 && kernel.front() == '(' && kernel.back() == ')')
        kernel = kernel.substr(1, kernel.size() - 2);

    const char * classification;
    switch (launchErrorClass(err))
    {
        case LaunchErrorClass::Configuration:
            classification =
                "the launch was REJECTED before the kernel started, so no "
                "kernel has poisoned the CUDA context. The fix is in the "
                "launch configuration, the inputs, or the environment";
            break;
        case LaunchErrorClass::Execution:
            classification =
                "this is an EXECUTION error raised by a running kernel. It is "
                "STICKY: the CUDA context is now unusable and every subsequent "
                "CUDA call in this process will fail with the same error";
            break;
        default:
            classification =
                "this error is not one that fastfields classifies, so whether "
                "the CUDA context survived it is unknown";
            break;
    }

    char buf[2048];
    int n = std::snprintf(
        buf, sizeof(buf),
        "ff::cuda: CUDA kernel launch failed.\n"
        "  kernel        : %s\n"
        "  configuration : grid=(%u,%u,%u) block=(%u,%u,%u) "
        "shared=%llu bytes stream=%p -- %llu threads total\n"
        "  error         : %s (%d): %s\n"
        "  meaning       : %s.\n"
        "  observed      : %s",
        kernel.c_str(),
        grid.x, grid.y, grid.z, block.x, block.y, block.z,
        static_cast<unsigned long long>(shared_mem),
        static_cast<void*>(stream),
        static_cast<unsigned long long>(grid.x) * grid.y * grid.z *
            block.x * block.y * block.z,
        cudaGetErrorName(err), static_cast<int>(err), cudaGetErrorString(err),
        classification,
        from_sync
            ? "on synchronising the stream after the launch "
              "(FF_CUDA_LAUNCH_SYNC is on), so it was raised by the kernel "
              "itself rather than by the launch configuration"
            : "immediately after the launch, without synchronising");

    // The register budget, when we could get it and it is the thing that went
    // wrong. `CUDA_NUM_THREADS` is 1024 -- the *architectural maximum*, not a
    // safe default -- so a register-heavy instantiation whose
    // maxThreadsPerBlock is below that fails here, and this line says so in
    // as many words rather than leaving it to be rediscovered with a
    // debugger. See fastfields-lib#152 and `GET_BLOCKS`'s block-size
    // parameter in utils.h.
    if (n > 0 && n < static_cast<int>(sizeof(buf)) && max_threads > 0)
    {
        const bool over = static_cast<int>(block.x * block.y * block.z)
                        > max_threads;
        std::snprintf(
            buf + n, sizeof(buf) - static_cast<size_t>(n),
            "\n  this kernel   : maxThreadsPerBlock=%d, %d registers/thread"
            "%s",
            max_threads, num_regs,
            over ? " -- THE LAUNCH ASKED FOR MORE THREADS PER BLOCK THAN THIS "
                   "KERNEL CAN TAKE. Its register usage is the limit. Read the "
                   "note above CUDA_NUM_THREADS in impl/cuda/utils.h before "
                   "changing the block size: two launchers size a device "
                   "scratch buffer from the launch geometry, so the grid may "
                   "not simply be scaled up to compensate."
                 : "");
    }

    if (preexisting)
    {
        // See the note at the peek in `launchKernel`. Say it out loud rather
        // than quietly attributing somebody else's failure to this launch.
        const size_t len = std::strlen(buf);
        std::snprintf(
            buf + len, sizeof(buf) - len,
            "\n  ATTRIBUTION   : the CUDA error state was ALREADY set with "
            "this same code before this launch was issued, so it may have "
            "been raised by an earlier CUDA call outside fastfields. It is "
            "reported here because this is the first place in the process "
            "that inspects it.");
    }

    throw std::runtime_error(buf);
}

/***********************************************************************
 *                            THE LAUNCH                               *
 ***********************************************************************/

// The only `<<<` in the tree. Launches `kernel` on the CALLER'S STREAM with
// the given configuration, then checks the error state and throws on failure.
//
// `kernel` is a pointer to a __global__ function; nvcc lowers `ptr<<<...>>>()`
// through the same host stub as a direct launch, so routing every site through
// this template costs nothing and gains a compile-time check that the argument
// list matches the kernel signature.
template <class Kernel, class... Args>
FF_CUHOST inline void launchKernel(
    const char   * name,
    Kernel         kernel,
    dim3           grid,
    dim3           block,
    size_t         shared_mem,
    cudaStream_t   stream,
    Args &&...     args)
{
    // MISATTRIBUTION, handled rather than ignored.
    //
    // `cudaGetLastError` returns *and clears* the last error from any
    // preceding CUDA call in this thread -- not just from a launch. So if
    // something entirely unrelated failed earlier and nobody looked, the
    // check below would find that error sitting there and blame this launch
    // for it. `cudaPeekAtLastError` reads the same word WITHOUT clearing it,
    // so we can tell the two apart: if the state was already dirty with the
    // same code, the report says so explicitly instead of asserting that this
    // kernel is what failed.
    //
    // Why observe here rather than draining at the dispatch boundary
    // (src/lib-cuda/*.cpp): draining would *discard* an error belonging to
    // the host application -- PyTorch, CuPy -- which is entitled to see it on
    // its own next check. fastfields is a library in someone else's process
    // and does not own that state. Peeking takes nothing away. We do clear on
    // the failure path below, but only because we immediately convert it into
    // an exception: the error is escalated, never swallowed.
    const cudaError_t prior = cudaPeekAtLastError();

    kernel<<<grid, block, shared_mem, stream>>>(std::forward<Args>(args)...);

    // Host-side only. No `cudaDeviceSynchronize`, no `cudaStreamSynchronize`,
    // nothing that waits on the device -- see the header comment.
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        int max_threads = -1, num_regs = -1;
        cudaFuncAttributes attr;
        if (cudaFuncGetAttributes(&attr, kernel) == cudaSuccess)
        {
            max_threads = attr.maxThreadsPerBlock;
            num_regs    = attr.numRegs;
        }
        // The query above sets the error state if it failed; drop that so the
        // state we leave behind reflects the launch, not our own diagnosis.
        cudaGetLastError();

        _throwLaunchError(name, grid, block, shared_mem, stream, err,
                          /*preexisting=*/err == prior, /*from_sync=*/false,
                          max_threads, num_regs);
    }

    // Opt-in, off by default, and never on by default: this serialises the
    // caller against the device on every launch. It is the only way to
    // observe a fault raised while the kernel runs, which is a different and
    // strictly larger class of failure than the check above can reach.
    if (launchSyncEnabled())
    {
        const cudaError_t serr = cudaStreamSynchronize(stream);
        if (serr != cudaSuccess)
        {
            cudaGetLastError();   // drain: we are about to report it
            _throwLaunchError(name, grid, block, shared_mem, stream, serr,
                              /*preexisting=*/false, /*from_sync=*/true,
                              -1, -1);
        }
    }
}

FF_NAMESPACE_END(FF_DEVICE)
FF_NAMESPACE_END(FF_NS)

/***********************************************************************
 *                          THE CALL SITE FORM                         *
 ***********************************************************************/

// Replaces
//     kernel<A, B><<<grid, block, shmem, stream>>>(x, y, z);
// with
//     FF_CUDA_LAUNCH((kernel<A, B>), grid, block, shmem, stream, x, y, z);
//
// The macro exists for exactly one reason: `#KERNEL` records the kernel's
// spelling so the failure report can name it. Everything else is the function
// template above, per this project's "prefer an inline function to a macro"
// rule -- but a function cannot stringise its own argument.
//
// PARENTHESES ARE REQUIRED around a template-id: the commas inside
// `kernel<A, B>` would otherwise split it across macro parameters. A
// parenthesised function name still decays to a function pointer, and
// `_throwLaunchError` strips the parentheses back out of the printed name.
#define FF_CUDA_LAUNCH(KERNEL, GRID, BLOCK, SHMEM, STREAM, ...)              \
    ::FF_NS::FF_DEVICE::launchKernel(                                        \
        #KERNEL, KERNEL, (GRID), (BLOCK), (SHMEM), (STREAM), __VA_ARGS__)
