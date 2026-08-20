#!/bin/sh
# Fail if any nvcc-produced object references `exit` -- the signature of a
# __host__ function calling a __device__-only one (fastfields-lib#150).
#
# WHAT IS BEING DETECTED
# ---------------------------------------------------------------------------
# nvcc rejects a host->device call only when the CALLING function is not itself
# a template. Every host caller in this tree is a template (the FF_CUHOST
# launchers in impl/cuda/, `canUse32BitIndexMath`, mesh.h's `build_tree`), so
# the check is skipped and cudafe++ instead emits, into the HOST object, this
# body in place of the callee's real one:
#
#     {int volatile ___ = 1; (void)args; ::exit(___);}
#
# That compiles, links, and passes both `-Wl,--no-undefined` and `ldd -r`,
# because the damage is intra-TU: nothing becomes undefined. It terminates the
# calling process with status 1 at the first call, and -- since `exit` is
# `noreturn` -- at -O1 and above the host compiler deletes every statement
# after it, so the dtype/dim/bound dispatch downstream vanishes from the object
# too. `libfastfields-cuda.so` shipped in that state for months with every job
# green.
#
# WHY `exit` IS A SOUND ORACLE HERE
# ---------------------------------------------------------------------------
# Nothing in this codebase calls `::exit` (or `std::exit`) -- checked; the only
# `exit` in the tree is `ff::cpu::ThreadPool::Worker::exit()`, a member
# function with a different symbol. So an undefined `exit` in a lib-cuda object
# has exactly one source: a cudafe++ stub. If a deliberate `::exit` call is
# ever added below `impl/`, this check has to be revisited rather than
# silenced.
#
# This is the catch-all net. The precise, earlier gate is
# tests/impl-cuda/compile_probe_hostdev.cu, which turns the same mistake into
# an nvcc error at compile time by calling each helper from a non-template host
# function -- the one shape nvcc does diagnose.
#
# Usage: tools/check-cuda-host-stubs.sh <object> [<object> ...]

set -eu

[ $# -gt 0 ] || { echo "usage: $0 <object> [...]" >&2; exit 2; }

status=0
checked=0
for obj in "$@"; do
    [ -f "$obj" ] || { echo "no such object: $obj" >&2; status=1; continue; }
    checked=$((checked + 1))
    if nm -u "$obj" 2>/dev/null | grep -qw 'exit'; then
        echo "FAIL $obj -- undefined reference to exit()"
        echo "     A __host__ function in this TU calls a __device__-only one."
        echo "     cudafe++ replaced the callee's host body with exit(1); every"
        echo "     statement after the call was then dropped as unreachable."
        echo "     Find it with:"
        echo "       nvcc ... --keep --keep-dir /tmp/keep  # then grep the"
        echo "       # generated .cudafe1.cpp for '::exit(___)' and read the"
        echo "       # function above it -- that is the mis-qualified callee."
        echo "     Fix: give that function FF_CUHOSTDEV, not FF_CUDEV, and add"
        echo "     it to tests/impl-cuda/compile_probe_hostdev.cu."
        status=1
    else
        echo "ok   $obj"
    fi
done

if [ "$status" -eq 0 ]; then
    echo "no host-pass exit(1) stubs in $checked object(s)"
fi
exit "$status"
