#!/usr/bin/env bash
#
# test-baseline.sh -- record / verify the fastfields CPU test baseline.
#
# The six C++/CUDA repos are being consolidated into one, which rewrites git
# history and relocates every file. The only mechanical proof that such a
# migration changed nothing is that the test suite produces *identical* results
# before and after. This script produces that evidence in a diff-able form.
#
# It builds fastfields-cpu-lib's suite once per configuration leg (the legs
# .github/workflows/test.yaml actually runs) and emits one sorted TSV line per
# (suite, config) with the exact check and failure counts:
#
#     suite<TAB>config<TAB>checks<TAB>failures
#
# Two runs of this script are compared with plain `diff`. A byte-identical
# report means the tree under test behaves identically.
#
# Usage
#   tools/test-baseline.sh --tree DIR   [options]   # measure an existing tree
#   tools/test-baseline.sh --ref REF    [options]   # clone fresh at REF
#   tools/test-baseline.sh --tree DIR --check tools/test-baseline.expected
#
# Options
#   --tree DIR      Either layout, detected automatically:
#                     * the CONSOLIDATED tree -- recognised by make/common.mk.
#                       Sources live in src/lib-cpu and src/lib, tests in
#                       tests/lib-cpu and tests/lib, and every group shares the
#                       single build/ at the root.
#                     * the six-repo tree -- a directory holding the wired repo
#                       checkouts. Must contain fastfields-cpu-lib/ (with
#                       impl -> ../fastfields-cpu-impl and impl/kernels ->
#                       ../fastfields-kernels). If the symlinks are missing but
#                       sibling checkouts exist, they are created.
#                   Measuring both and diffing the two reports is the point of
#                   this script: the migration's correctness argument is that
#                   they are byte-identical.
#   --cpu-lib DIR   Point directly at the cpu-lib checkout (skips the layout
#                   guessing above). Overrides --tree for locating the Makefile.
#   --ref REF       Clone the repos fresh at REF into a work dir and wire them.
#                   REF applies to fastfields-cpu-lib; the submodule chain is
#                   resolved from the recorded gitlink pins (what CI does).
#   --workdir DIR   Where --ref clones to. Default: a temp dir (removed on exit).
#   --cxx CXX       Compiler command. Default: clang++.
#   --jobs N        make -j. Default: number of CPUs.
#   --legs LIST     Comma-separated legs to run. Default: static,dynamic,
#                   cuda-default. Also available: sanitize; `default` (a bare
#                   `make test`, i.e. the Makefile's own target-specific sparse
#                   default); and `lib` (fastfields-lib's own two standalone
#                   argument-validation tests). Use `all` for every leg.
#   --lib DIR       fastfields-lib checkout for the `lib` leg. Defaults to
#                   <tree>/fastfields-lib. These tests include only hub headers,
#                   so they need no submodules and run in seconds.
#   --out FILE      Write the report here as well as to stdout.
#   --check FILE    Compare the report against FILE and exit non-zero on any
#                   difference. Comment lines are ignored on both sides, so the
#                   recorded file can carry provenance; every data row must
#                   still match exactly. Implies the failure gate below.
#   --keep          Do not delete a --ref workdir on exit.
#   -h|--help       This text.
#
# Exit status
#   0  every suite ran, every leg reported 0 failures, and (with --check) the
#      report matched FILE exactly.
#   1  a suite reported failures, a suite failed to build/run, or --check found
#      a difference.
#   2  usage / environment error.
#
# Notes that are load-bearing -- read before changing anything here.
#
#   * Every leg goes through `make test`, never `make build/test_<x>`. The
#     Makefile sets BOUNDFLAGS/SPLINEFLAGS with *target-specific plain `=`*
#     assignments on the `test:` target (they cannot be `?=`, because the
#     global `?=` defaults already count as "set" at parse time, so a
#     target-specific `?=` would never fire). Those target-specific values
#     propagate to the prerequisites `make test` builds -- but building a test
#     binary by its own path does NOT enter the `test:` context, and would
#     silently compile the fully-static policy instead of the requested leg.
#
#   * `make test` always compiles with -DFF_TEST_SPARSE (it is hard-coded into
#     TESTCPPFLAGS, not a leg variable). It is therefore NOT a configuration
#     axis: it is on for every leg here and for every leg in CI. It makes
#     resize.cpp/restrict.cpp instantiate only a covering subset of the
#     order x bound matrix; the tests are written to stay inside that subset.
#     `make all` (the library) omits it and compiles the full matrix.
#
#   * The sanitize leg passes CXXFLAGS on the command line, which overrides the
#     Makefile's `CXXFLAGS +=` wholesale (a command-line variable outranks any
#     in-makefile assignment, `+=` included), so everything the Makefile would
#     have contributed is restated there.
#
#   * Suites report in three different formats: twelve of cpu-lib's print
#     "checks: N, failures: M"; test_solve_field prints "P/N checks passed";
#     fastfields-lib's two print one "ok:"/"FAIL:" line per assertion and no
#     total. All three are parsed. A suite using a fourth format is reported as
#     UNPARSED and fails the run rather than being silently skipped.

set -uo pipefail

die() { printf 'test-baseline: %s\n' "$*" >&2; exit 2; }

# ---------------------------------------------------------------- arguments

TREE=""
CPU_LIB=""
CONSOLIDATED=""
LIB_DIR=""
REF=""
WORKDIR=""
CXX_CMD="clang++"
JOBS=""
LEGS="static,dynamic,cuda-default"
OUT=""
CHECK=""
KEEP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --tree)     TREE="${2:-}"; shift 2 ;;
        --cpu-lib)  CPU_LIB="${2:-}"; shift 2 ;;
        --lib)      LIB_DIR="${2:-}"; shift 2 ;;
        --ref)      REF="${2:-}"; shift 2 ;;
        --workdir)  WORKDIR="${2:-}"; shift 2 ;;
        --cxx)      CXX_CMD="${2:-}"; shift 2 ;;
        --jobs)     JOBS="${2:-}"; shift 2 ;;
        --legs)     LEGS="${2:-}"; shift 2 ;;
        --out)      OUT="${2:-}"; shift 2 ;;
        --check)    CHECK="${2:-}"; shift 2 ;;
        --keep)     KEEP=1; shift ;;
        -h|--help)  sed -n '2,/^set -uo/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1; exit 0 ;;
        *)          die "unknown argument: $1 (try --help)" ;;
    esac
done

[ -n "$JOBS" ] || JOBS="$( (command -v nproc >/dev/null && nproc) || echo 2)"

# Resolve to absolute paths immediately: every path below is used after `cd`,
# so the script must not depend on the directory it was invoked from.
abspath() { # $1 need not exist yet
    local d b
    d="$(dirname -- "$1")"; b="$(basename -- "$1")"
    d="$(cd -- "$d" 2>/dev/null && pwd)" || die "no such directory: $(dirname -- "$1")"
    printf '%s/%s\n' "${d%/}" "$b"
}

[ -n "$OUT" ]   && OUT="$(abspath "$OUT")"
[ -n "$CHECK" ] && { CHECK="$(abspath "$CHECK")"; [ -f "$CHECK" ] || die "no such baseline file: $CHECK"; }

GITHUB="https://github.com/fastfields"
CLEANUP_DIR=""
cleanup() { [ -n "$CLEANUP_DIR" ] && [ "$KEEP" -eq 0 ] && rm -rf -- "$CLEANUP_DIR"; }
trap cleanup EXIT

# ------------------------------------------------------------ tree assembly

# Wire the dev-tree symlink convention: cpu-lib/impl -> ../fastfields-cpu-impl
# and cpu-impl/kernels -> ../fastfields-kernels. Both repos also track these as
# real git submodules, so a `--recursive` checkout works too; this only fills in
# what is missing. A wrong or absent wiring compiles the wrong sources (or
# nothing), so it is verified afterwards rather than assumed.
wire_tree() {
    local root="$1"
    if [ ! -e "$root/fastfields-cpu-lib/impl/kernels" ] \
       && [ -d "$root/fastfields-cpu-impl" ] && [ -d "$root/fastfields-kernels" ]; then
        [ -d "$root/fastfields-cpu-lib/impl" ] && [ ! -L "$root/fastfields-cpu-lib/impl" ] \
            && rmdir -- "$root/fastfields-cpu-lib/impl" 2>/dev/null
        [ -e "$root/fastfields-cpu-lib/impl" ] \
            || ln -s ../fastfields-cpu-impl "$root/fastfields-cpu-lib/impl"
        [ -d "$root/fastfields-cpu-impl/kernels" ] && [ ! -L "$root/fastfields-cpu-impl/kernels" ] \
            && rmdir -- "$root/fastfields-cpu-impl/kernels" 2>/dev/null
        [ -e "$root/fastfields-cpu-impl/kernels" ] \
            || ln -s ../fastfields-kernels "$root/fastfields-cpu-impl/kernels"
    fi
}

clone_at_ref() {
    local root="$1" ref="$2"
    mkdir -p -- "$root" || die "cannot create workdir $root"
    git clone --quiet "$GITHUB/fastfields-cpu-lib" "$root/fastfields-cpu-lib" \
        || die "clone of fastfields-cpu-lib failed"
    git -C "$root/fastfields-cpu-lib" checkout --quiet "$ref" \
        || die "no such ref in fastfields-cpu-lib: $ref"
    # Follow the recorded gitlink pins down the chain -- this is what CI's
    # `submodules: recursive` checkout resolves to, and it is what makes a
    # --ref run reproducible. Note the pins may lag the submodules' own main.
    local impl_pin kern_pin
    impl_pin="$(git -C "$root/fastfields-cpu-lib" ls-tree HEAD impl | awk '{print $3}')"
    git clone --quiet "$GITHUB/fastfields-cpu-impl" "$root/fastfields-cpu-impl" \
        || die "clone of fastfields-cpu-impl failed"
    [ -n "$impl_pin" ] && git -C "$root/fastfields-cpu-impl" checkout --quiet "$impl_pin"
    kern_pin="$(git -C "$root/fastfields-cpu-impl" ls-tree HEAD kernels | awk '{print $3}')"
    git clone --quiet "$GITHUB/fastfields-kernels" "$root/fastfields-kernels" \
        || die "clone of fastfields-kernels failed"
    [ -n "$kern_pin" ] && git -C "$root/fastfields-kernels" checkout --quiet "$kern_pin"
    wire_tree "$root"
}

if [ -n "$REF" ]; then
    [ -n "$TREE$CPU_LIB" ] && die "--ref cannot be combined with --tree/--cpu-lib"
    if [ -n "$WORKDIR" ]; then
        WORKDIR="$(mkdir -p -- "$WORKDIR" && cd -- "$WORKDIR" && pwd)" || die "bad --workdir"
    else
        WORKDIR="$(mktemp -d)" || die "mktemp failed"
        CLEANUP_DIR="$WORKDIR"
    fi
    clone_at_ref "$WORKDIR" "$REF"
    TREE="$WORKDIR"
fi

if [ -z "$CPU_LIB" ]; then
    [ -n "$TREE" ] || die "one of --tree, --cpu-lib or --ref is required (try --help)"
    TREE="$(cd -- "$TREE" 2>/dev/null && pwd)" || die "no such directory: $TREE"
    wire_tree "$TREE"
    if   [ -f "$TREE/make/common.mk" ] && [ -f "$TREE/src/lib-cpu/Makefile" ]; then
        CONSOLIDATED="$TREE"; CPU_LIB="$TREE/src/lib-cpu"
        [ -n "$LIB_DIR" ] || LIB_DIR="$TREE/src/lib"
    elif [ -f "$TREE/fastfields-cpu-lib/Makefile" ]; then CPU_LIB="$TREE/fastfields-cpu-lib"
    elif [ -f "$TREE/Makefile" ] && [ -d "$TREE/tests" ]; then CPU_LIB="$TREE"
    else die "cannot find a cpu-lib checkout under $TREE (pass --cpu-lib)"
    fi
fi
CPU_LIB="$(cd -- "$CPU_LIB" 2>/dev/null && pwd)" || die "no such directory: $CPU_LIB"
[ -f "$CPU_LIB/Makefile" ] || die "$CPU_LIB has no Makefile"

# Where the tests live and where the build lands differ between the two
# layouts: the six-repo tree gives each repo its own tests/ and build/, the
# consolidated tree has one tests/<group>/ and one build/ for everything.
# Everything below reads these two variables rather than assuming either.
if [ -n "$CONSOLIDATED" ]; then
    CONSOLIDATED="$(cd -- "$CONSOLIDATED" && pwd)"
    CPU_TESTS_DIR="$CONSOLIDATED/tests/lib-cpu"
    BUILD_ROOT="$CONSOLIDATED/build"
    LIB_BUILD_ROOT="$BUILD_ROOT"
    # The include chain has to resolve, or the build tests the wrong code (or
    # fails in a way that looks like a source bug). Check a file from each
    # layer -- for the consolidated tree that is the -I root, not a symlink.
    PROBES="include/fastfields/impl/cpu/pushpull.h
            include/fastfields/core/bounds.h
            include/fastfields/impl/kernels/restrict.h
            include/fastfields/core/cuda_switch.h"
    for probe in $PROBES; do
        [ -f "$CONSOLIDATED/$probe" ] \
            || die "consolidated tree is missing $CONSOLIDATED/$probe"
    done
else
    CPU_TESTS_DIR="$CPU_LIB/tests"
    BUILD_ROOT="$CPU_LIB/build"
    LIB_BUILD_ROOT=""   # set once LIB_DIR is known, in run_lib_leg
    for probe in impl/pushpull.h impl/kernels/bounds.h impl/kernels/restrict.h; do
        [ -f "$CPU_LIB/$probe" ] || die "submodule chain not wired: missing $CPU_LIB/$probe
  expected cpu-lib/impl -> fastfields-cpu-impl and cpu-impl/kernels -> fastfields-kernels
  (either as symlinks, or via a --recursive submodule checkout)"
    done
fi
[ -d "$CPU_TESTS_DIR" ] || die "no test sources at $CPU_TESTS_DIR"


# ------------------------------------------------------------------- legs

# Each leg is: name | BOUNDFLAGS | SPLINEFLAGS | extra make args.
# The first three mirror .github/workflows/test.yaml's `test` matrix exactly;
# `sanitize` mirrors its separate ASan+UBSan job; `default` is a bare
# `make test`, i.e. whatever the Makefile's target-specific assignment says.
leg_boundflags() {
    case "$1" in
        static)       printf '%s' "" ;;
        dynamic)      printf '%s' "-DFF_STATIC_BOUNDS=0" ;;
        cuda-default) printf '%s' "-DFF_STATIC_BOUNDS=0 -DFF_STATIC_BOUND_DCT2=1 -DFF_STATIC_BOUND_DST2=1" ;;
        sanitize)     printf '%s' "-DFF_STATIC_BOUNDS=0" ;;
    esac
}
leg_splineflags() {
    case "$1" in
        static)       printf '%s' "" ;;
        dynamic)      printf '%s' "-DFF_STATIC_SPLINES=0" ;;
        cuda-default) printf '%s' "-DFF_STATIC_SPLINES=0 -DFF_STATIC_SPLINE_NEAREST=1 -DFF_STATIC_SPLINE_LINEAR=1 -DFF_STATIC_SPLINE_QUADRATIC=1 -DFF_STATIC_SPLINE_CUBIC=1" ;;
        sanitize)     printf '%s' "-DFF_STATIC_SPLINES=0" ;;
    esac
}

ALL_LEGS="static dynamic cuda-default sanitize default lib"
[ "$LEGS" = "all" ] && LEGS="$(printf '%s' "$ALL_LEGS" | tr ' ' ',')"
LEG_LIST="$(printf '%s' "$LEGS" | tr ',' ' ')"
for leg in $LEG_LIST; do
    case " $ALL_LEGS " in *" $leg "*) ;; *) die "unknown leg: $leg (known: $ALL_LEGS)" ;; esac
done

# ------------------------------------------------------------------- run

RESULTS="$(mktemp)" || die "mktemp failed"
NOTES="$(mktemp)"   || die "mktemp failed"
trap 'cleanup; rm -f -- "$RESULTS" "$NOTES"' EXIT

# fastfields-lib's own two tests: standalone argument-validation checks that
# live in the hub repo and nowhere below it. They include only hub headers, so
# they neither need the submodule chain nor link libfastfields.so. They print
# one "ok:"/"FAIL:" line per assertion and no total, so the count is the number
# of those lines. Note lib's `make test` loop is fail-fast (`|| exit 1`), unlike
# cpu-lib's (`|| status=1`), so a failure truncates the report -- the
# same-suite-count gate below catches that.
run_lib_leg() {
    local log rc
    [ -n "$LIB_DIR" ] || LIB_DIR="$TREE/fastfields-lib"
    local lib_tests="$LIB_DIR/tests"
    [ -n "$CONSOLIDATED" ] && lib_tests="$CONSOLIDATED/tests/lib"
    if [ ! -f "$LIB_DIR/Makefile" ] || [ ! -d "$lib_tests" ]; then
        printf 'leg lib: no fastfields-lib checkout at %s (pass --lib DIR)\n' "$LIB_DIR" >>"$NOTES"
        printf '#count\tlib\t0\n' >>"$RESULTS"
        return 1
    fi
    LIB_DIR="$(cd -- "$LIB_DIR" && pwd)"
    log="$(mktemp)" || die "mktemp failed"
    printf '== leg lib (CXX=%s)\n' "$CXX_CMD" >&2
    rm -rf -- "${LIB_BUILD_ROOT:-$LIB_DIR/build}"
    make -j"$JOBS" -C "$LIB_DIR" test CXX="$CXX_CMD" \
        ${COMMON_MAKEARGS[@]+"${COMMON_MAKEARGS[@]}"} >"$log" 2>&1
    rc=$?
    awk '
        /^--- / {
            n = split($2, p, "/"); suite = p[n]; sub(/^test_/, "", suite)
            suite = "lib_" suite
            order[++nord] = suite; c[suite] = 0; f[suite] = 0; next
        }
        suite != "" && /^ok: /   { c[suite]++; next }
        suite != "" && /^FAIL: / { c[suite]++; f[suite]++; next }
        END {
            for (i = 1; i <= nord; i++) {
                s = order[i]
                if (c[s] > 0) printf "%s\tlib\t%d\t%d\n", s, c[s], f[s]
                else          printf "%s\tlib\tUNPARSED\tUNPARSED\n", s
            }
            printf "#count\tlib\t%d\n", nord
        }
    ' "$log" >>"$RESULTS"
    if [ $rc -ne 0 ]; then
        printf 'leg lib: make exited %d -- last lines:\n' "$rc" >>"$NOTES"
        tail -n 25 "$log" | sed 's/^/    /' >>"$NOTES"
    fi
    rm -f -- "$log"
    return $rc
}

run_leg() {
    local leg="$1" log rc bf sf
    if [ "$leg" = "lib" ]; then run_lib_leg; return $?; fi
    log="$(mktemp)" || die "mktemp failed"
    bf="$(leg_boundflags "$leg")"
    sf="$(leg_splineflags "$leg")"

    printf '== leg %s (CXX=%s, -j%s)\n' "$leg" "$CXX_CMD" "$JOBS" >&2

    # Always start from a clean build dir: leg flags change object content but
    # the object *paths* are identical, and the Makefile's dependency files
    # track headers, not flags. A stale object from the previous leg would be
    # reused and the leg would silently measure the wrong policy.
    rm -rf -- "$BUILD_ROOT"

    if [ "$leg" = "default" ]; then
        # No BOUNDFLAGS/SPLINEFLAGS on the command line: exercise the
        # Makefile's own target-specific defaults for the `test:` target.
        make -j"$JOBS" -C "$CPU_LIB" test CXX="$CXX_CMD" \
            ${COMMON_MAKEARGS[@]+"${COMMON_MAKEARGS[@]}"} >"$log" 2>&1
        rc=$?
    elif [ "$leg" = "sanitize" ]; then
        # CXXFLAGS on the command line replaces the Makefile's `CXXFLAGS +=`
        # entirely, so every flag it would have added is restated here.
        # -fno-sanitize-recover=all is what makes UBSan a gate rather than a
        # report: without it UBSan prints and continues, and the run exits 0.
        UBSAN_OPTIONS=print_stacktrace=1 ASAN_OPTIONS=detect_stack_use_after_return=1 \
        make -j"$JOBS" -C "$CPU_LIB" test CXX="$CXX_CMD" \
            CXXFLAGS="-std=c++11 -O1 -g -fPIC -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer $DIAGFLAGS" \
            TESTFLAGS="" BOUNDFLAGS="$bf" SPLINEFLAGS="$sf" >"$log" 2>&1
        rc=$?
    else
        make -j"$JOBS" -C "$CPU_LIB" test CXX="$CXX_CMD" \
            ${COMMON_MAKEARGS[@]+"${COMMON_MAKEARGS[@]}"} BOUNDFLAGS="$bf" SPLINEFLAGS="$sf" >"$log" 2>&1
        rc=$?
    fi

    # Parse the run section. `make test`'s recipe is a sequential shell loop, so
    # "running <bin>" and that binary's own output stay in order even under -j.
    awk -v leg="$leg" '
        /^running / {
            n = split($2, p, "/"); suite = p[n]; sub(/^test_/, "", suite)
            seen[suite] = 1; order[++nord] = suite; next
        }
        suite != "" && /^checks: [0-9]+, failures: [0-9]+/ {
            c = $2; sub(/,$/, "", c); f = $4
            checks[suite] = c; fails[suite] = f; got[suite] = 1; next
        }
        suite != "" && /^[0-9]+\/[0-9]+ checks passed/ {
            split($1, q, "/"); checks[suite] = q[2]; fails[suite] = q[2] - q[1]
            got[suite] = 1; next
        }
        END {
            for (i = 1; i <= nord; i++) {
                s = order[i]
                if (got[s]) printf "%s\t%s\t%s\t%s\n", s, leg, checks[s], fails[s]
                else        printf "%s\t%s\tUNPARSED\tUNPARSED\n", s, leg
            }
            printf "#count\t%s\t%d\n", leg, nord
        }
    ' "$log" >>"$RESULTS"

    if [ $rc -ne 0 ]; then
        printf 'leg %s: make exited %d -- last lines:\n' "$leg" "$rc" >>"$NOTES"
        tail -n 25 "$log" | sed 's/^/    /' >>"$NOTES"
    fi
    rm -f -- "$log"
    return $rc
}

# Compiler awareness. The Makefiles are clang-only by default: CXXFLAGS picks up
# -ferror-limit / -ftemplate-backtrace-limit (and fastfields-lib's TESTFLAGS the
# same two), which gcc rejects outright -- so `make CXX=g++` fails on the *flags*
# long before it reaches any source. To build with anything else we must replace
# CXXFLAGS wholesale on the command line (a command-line variable outranks the
# makefile's `+=`), restating the flags that actually matter and dropping the
# clang-specific diagnostics. TESTFLAGS is emptied for the same reason (it is
# vestigial in cpu-lib but real in fastfields-lib's test rule).
#
# This is deliberately keyed on the compiler rather than on a user flag, so that
# --cxx g++ just works and the two compilers' reports stay directly comparable.
CXX_IS_CLANG=0
case "$($CXX_CMD --version 2>&1 | head -1)" in *clang*) CXX_IS_CLANG=1 ;; esac

DIAGFLAGS="-ferror-limit=1 -ftemplate-backtrace-limit=0"
[ "$CXX_IS_CLANG" -eq 0 ] && DIAGFLAGS=""

# Extra `make` arguments for every non-sanitize leg. Kept in an array so a value
# containing spaces stays a single argument.
set_common_makeargs() {
    COMMON_MAKEARGS=()
    if [ "$CXX_IS_CLANG" -eq 0 ]; then
        COMMON_MAKEARGS=(CXXFLAGS="-std=c++11 -O3 -fPIC" TESTFLAGS="")
    fi
}
set_common_makeargs

overall=0
for leg in $LEG_LIST; do
    run_leg "$leg" || overall=1
done

# --------------------------------------------------------------- report

# Sorted with LC_ALL=C so the byte ordering does not depend on the machine's
# locale -- the whole point is that two reports compare with plain `diff`.
REPORT="$(mktemp)" || die "mktemp failed"
{
    printf '# fastfields cpu-lib test baseline (format v1)\n'
    printf '# columns: suite\tconfig\tchecks\tfailures\n'
    grep -v '^#count' -- "$RESULTS" | LC_ALL=C sort
} >"$REPORT"

cat -- "$REPORT"
[ -n "$OUT" ] && { cp -- "$REPORT" "$OUT"; printf 'wrote %s\n' "$OUT" >&2; }

# ----------------------------------------------------------- the gate

# 1. every leg must have produced a result line for every suite it started, and
#    no suite may report failures.
if grep -q 'UNPARSED' -- "$REPORT"; then
    printf 'FAIL: a suite produced output this parser does not understand\n' >&2
    overall=1
fi
if awk -F'\t' '/^#/ {next} $4 != 0 {exit 0} END {exit 1}' "$REPORT"; then
    printf 'FAIL: at least one suite reported failures\n' >&2
    overall=1
fi
# 2. every cpu-lib leg must have run the same number of suites as every other
#    cpu-lib leg: a suite that fails to build simply never prints "running", and
#    would otherwise vanish from the report without failing it. The `lib` leg is
#    a different repo with its own (smaller) suite set, so it is compared only
#    against itself -- it must be non-empty.
nsuites="$(awk -F'\t' '/^#count/ && $2 != "lib" {print $3}' "$RESULTS" | LC_ALL=C sort -u | wc -l)"
if [ "$nsuites" -gt 1 ]; then
    printf 'FAIL: cpu-lib legs ran different numbers of suites:\n' >&2
    awk -F'\t' '/^#count/ && $2 != "lib" {printf "    %s: %s suites\n", $2, $3}' "$RESULTS" >&2
    overall=1
fi
if awk -F'\t' '/^#count/ && $2 == "lib" && $3 == 0 {exit 0} END {exit 1}' "$RESULTS"; then
    printf 'FAIL: the lib leg ran no suites\n' >&2
    overall=1
fi

if [ -s "$NOTES" ]; then printf '\n-- build/run diagnostics --\n' >&2; cat -- "$NOTES" >&2; fi

# 3. with --check, every data line must match the recorded baseline exactly.
#    Comment lines (`#...`) are stripped from both sides first, so the recorded
#    file can carry provenance (which tree, which commits, which compiler) while
#    the comparison itself stays a strict line-for-line match on the data.
if [ -n "$CHECK" ]; then
    exp_data="$(mktemp)"; got_data="$(mktemp)"
    grep -v '^#' -- "$CHECK"  >"$exp_data" 2>/dev/null
    grep -v '^#' -- "$REPORT" >"$got_data" 2>/dev/null
    # A subset of legs would otherwise diff as "every recorded row vanished",
    # which reads like catastrophic breakage rather than the operator error it
    # is. Compare the config sets first and say so plainly.
    exp_cfg="$(cut -f2 -- "$exp_data" | LC_ALL=C sort -u | tr '\n' ' ')"
    got_cfg="$(cut -f2 -- "$got_data" | LC_ALL=C sort -u | tr '\n' ' ')"
    if [ "$exp_cfg" != "$got_cfg" ]; then
        printf '\nBASELINE NOT COMPARABLE against %s\n' "$CHECK" >&2
        printf '  recorded configs: %s\n  measured configs: %s\n' "$exp_cfg" "$got_cfg" >&2
        printf '  --check compares whole reports, so re-run with the same --legs\n' >&2
        printf '  the recording used.\n' >&2
        overall=1
    elif diff -u -- "$exp_data" "$got_data" >/dev/null 2>&1; then
        printf '\nBASELINE MATCH: %d rows identical to %s\n' \
            "$(wc -l <"$got_data")" "$CHECK" >&2
    else
        printf '\nBASELINE MISMATCH against %s:\n' "$CHECK" >&2
        diff -u --label expected --label measured -- "$exp_data" "$got_data" >&2
        overall=1
    fi
    rm -f -- "$exp_data" "$got_data"
fi

rm -f -- "$REPORT"
exit $overall
