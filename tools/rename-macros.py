#!/usr/bin/env python3
"""
rename-macros.py -- give every macro on the installed public surface an FF_
prefix.

THE RULE THIS APPLIES
--------------------------------------------------------------------------
Every macro that survives preprocessing of a header under `include/` -- i.e.
every `#define` that is not `#undef`'d before the end of the header that
defined it -- must be spelled `FF_*`. Installing a header means every
translation unit downstream of it inherits its macros, and an unprefixed one
is a name this project has silently taken from everybody who includes us.
184 macros here already follow the convention; these are the stragglers.

Macros defined and consumed inside a single `.cpp` are NOT covered: they never
leave the translation unit, so they cannot collide with anyone. (A macro that
gets *moved into* a header is covered from the moment it moves -- which is why
de-duplication and prefixing are one job, done in that order. That half landed
in the preceding commit.)

Where an `inline` function will do the job, prefer it and delete the macro
outright: a function in `ff::` is collision-safe with no prefix at all. The
preceding commit did that for `IS_CPU` / `IS_CUDA`; this one does it for
`uchar_t`.

DELIBERATELY EXEMPT
--------------------------------------------------------------------------
`include/fastfields/core/cuda_switch.h` keeps two families unprefixed, because
prefixing them would destroy the thing they exist to do:

  * `#define int8_t signed char` ... `#define uint64_t unsigned long`, under
    `#ifdef __CUDACC_RTC__`. NVRTC ships no standard library, so these hand
    definitions ARE `<cstdint>` in that mode. They must keep the standard
    spellings or nothing downstream compiles.
  * `#define __device__` / `#define __host__`, under `#ifndef __CUDACC__`.
    These erase nvcc's qualifiers when a host compiler sees CUDA-annotated
    code. Renaming them would leave the real names undefined.

`include/fastfields/core/dlpack.h` is vendored upstream code (`DLPACK_*`) and
is never rewritten by this script.

USAGE
--------------------------------------------------------------------------
    python3 tools/rename-macros.py           # apply
    python3 tools/rename-macros.py --check   # report residue, change nothing

Idempotent, and whole-identifier: `CUHOST` cannot eat `CUHOSTDEV`, and `FF`
cannot eat `FF_DEVICE`. Every rename is compiler-verified -- a missed site is
an undeclared identifier, not a silent behaviour change.

If this lands on a base that has moved, do NOT resolve conflicts by hand:
reset, re-run the script on the new base, and commit that. It is deterministic,
which is the whole reason it is committed rather than described.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIRS = ("include", "src", "tests")
SOURCE_EXTS = (".h", ".hpp", ".inl", ".cpp", ".cu", ".cuh")

# Vendored third-party code: never rewritten.
SKIP = {"include/fastfields/core/dlpack.h"}

RENAMES = [
    # ---- the namespace spine -------------------------------------------
    # `FF` is a two-letter, all-caps macro in an installed header -- the worst
    # offender on the list. See the commit message for why it is renamed
    # rather than left alone or #undef'd (the #undef option is not merely
    # awkward here, it is silently wrong).
    ("FF", "FF_NS"),
    # ---- core/cuda_switch.h: the CUDA qualifier abbreviations -----------
    ("CUGLOB", "FF_CUGLOB"),
    ("CUHOSTDEV", "FF_CUHOSTDEV"),
    ("CUHOST", "FF_CUHOST"),
    ("CUDEV", "FF_CUDEV"),
    # ---- api/{cpu,cuda}/pushpull_dispatch.h ----------------------------
    # Renamed only. The dispatch pyramid's *design* is a separate change in
    # someone else's hands; this is spelling. DISPATCH_PP becomes
    # FF_DISPATCH_PP rather than FF_PP_DISPATCH because impl/cuda/pushpull.h
    # already uses the latter name (it #undef's it, so there is no actual
    # collision -- but reusing the spelling would be gratuitously confusing).
    ("DISPATCH_PP", "FF_DISPATCH_PP"),
    ("PP_BOUND", "FF_PP_BOUND"),
    ("PP_DTYPE", "FF_PP_DTYPE"),
    ("PP_ORDER", "FF_PP_ORDER"),
    # ---- impl/kernels/ --------------------------------------------------
    ("ATOMIC_INTEGER_IMPL", "FF_ATOMIC_INTEGER_IMPL"),
    ("GPU_ATOMIC_INTEGER", "FF_GPU_ATOMIC_INTEGER"),
    ("INTERPOL_UTILS", "FF_INTERPOL_UTILS"),
    ("DIST_USE_LOOP", "FF_DIST_USE_LOOP"),
    # `JFH_` is a leftover prefix from the jitfields headers this code came
    # from; it is a prefix, but not this project's prefix.
    ("JFH_OnePlusTiny", "FF_ONE_PLUS_TINY"),
]

# `uchar_t` is deleted rather than renamed. A lowercase macro that
# impersonates a typedef is worse than a shouty one: a downstream
# `typedef unsigned char uchar_t;` does not merely collide, it fails to
# compile with a diagnostic pointing at the wrong file. Six uses, two headers.
TYPE_MACROS = {"uchar_t": "unsigned char"}

# JFH_OnePlusTiny is #defined identically in three sibling headers. Only
# posdef/utils.h keeps the definition -- the other two both include it.
DEDUP_DEFINE = {
    "FF_ONE_PLUS_TINY": (
        "include/fastfields/impl/kernels/posdef/utils.h",
        (
            "include/fastfields/impl/kernels/posdef/cholesky.h",
            "include/fastfields/impl/kernels/posdef/posdef.h",
        ),
    )
}


def sources():
    for d in SOURCE_DIRS:
        for dirpath, _, filenames in os.walk(os.path.join(ROOT, d)):
            for name in sorted(filenames):
                if not name.endswith(SOURCE_EXTS):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, name), ROOT)
                if rel not in SKIP:
                    yield rel


def rename(text, old, new):
    return re.sub(
        r"(?<![A-Za-z0-9_])" + re.escape(old) + r"(?![A-Za-z0-9_])", new, text
    )


def drop_define(text, name):
    return re.sub(
        r"^[ \t]*#[ \t]*define[ \t]+" + re.escape(name) + r"\b"
        r"(?:[^\n]*\\\n)*[^\n]*\n"
        r"(?:[ \t]*\n)?",
        "",
        text,
        flags=re.MULTILINE,
    )


def apply(check_only=False):
    changed = 0
    for rel in sources():
        path = os.path.join(ROOT, rel)
        with open(path, encoding="utf-8") as fh:
            text = original = fh.read()
        for old, new in RENAMES:
            text = rename(text, old, new)
        for old, expansion in TYPE_MACROS.items():
            text = drop_define(text, old)
            text = rename(text, old, expansion)
        for name, (_keep, drop_from) in DEDUP_DEFINE.items():
            if rel in drop_from:
                text = drop_define(text, name)
        if text != original:
            changed += 1
            if check_only:
                print("would rewrite %s" % rel)
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
    print("%d file(s)%s" % (changed, " would change" if check_only else " rewritten"))
    return changed


def residue():
    """Every macro still defined unprefixed under include/, minus the
    documented exemptions. Must be empty after a run."""
    exempt = re.compile(r"^(?:DLPACK_|__device__$|__host__$|u?int(?:8|16|32|64)_t$)")
    hits = []
    for rel in sources():
        if not rel.startswith("include/"):
            continue
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                m = re.match(r"\s*#\s*define\s+([A-Za-z_]\w*)", line)
                if m and not m.group(1).startswith("FF_") and not exempt.match(
                    m.group(1)
                ):
                    hits.append("%s:%d: %s" % (rel, n, m.group(1)))
    return hits


if __name__ == "__main__":
    check = "--check" in sys.argv
    apply(check_only=check)
    left = residue()
    if left:
        print("\nSTILL UNPREFIXED in include/:")
        for h in left:
            print("  " + h)
    else:
        print(
            "\ninclude/ is clean: every surviving macro is FF_-prefixed or "
            "documented-exempt."
        )
    sys.exit(1 if left else 0)
