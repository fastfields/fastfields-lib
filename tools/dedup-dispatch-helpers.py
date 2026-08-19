#!/usr/bin/env python3
"""
dedup-dispatch-helpers.py -- hoist the copy-pasted dispatch helpers into the
shared headers, and rewrite every call site onto their FF_-prefixed names.

This is the script that produced the de-duplication commit. It is committed
rather than merely described because the change touches 32 files and ~1100 call
sites: a reviewer who wants to confirm the rewrite was mechanical can re-run it
against the parent commit and diff, instead of reading 1100 renamed lines.

    git checkout <parent> -- include src
    python3 tools/dedup-dispatch-helpers.py
    git diff        # must be empty

It is idempotent: on an already-converted tree the local #defines are gone and
the call sites already carry the prefix, so a second run is a no-op.

WHAT MOVES WHERE, AND WHY
--------------------------------------------------------------------------
Placement is decided by *audience*, and the audience question that actually
bites here is which compiler sees the code:

include/fastfields/core/dispatch.h   (new)
    FF_VOIDPTR / FF_CVOIDPTR / FF_CVOIDPTR_OR_NULL / FF_CANUSE32BITS, the
    FF_CHECK_* family, and as_weights(). Shared by src/lib-cpu (host compiler)
    AND src/lib-cuda (nvcc) AND the two api/*/pushpull_dispatch.h headers, so
    it must be backend-agnostic -> core/, which is the only directory that is
    backend-agnostic by contract.

include/fastfields/api/cuda/stream.h (new)
    _reg_stream(). Names cudaStream_t, so it is CUDA-only by construction and
    belongs under api/cuda/, not core/. Putting it in core/ would also have
    mis-tagged it for CI's path filter.

include/fastfields/api/checks.h      (extended)
    is_cpu() / is_cuda(), replacing the IS_CPU / IS_CUDA macros. Used only by
    src/lib (the hub), which is host-compiled -- the hub's own validation
    header is the right home, and functions are collision-safe without a
    prefix, so these two macros are deleted rather than renamed.

THE DIVERGENCES, PRESERVED
--------------------------------------------------------------------------
Three of the duplicated macros were NOT identical across their copies. Each
keeps its own name so the difference is visible at the call site:

  CVOIDPTR          posdef.cpp alone guarded against a null `data` (optional
                    tensors) -> FF_CVOIDPTR_OR_NULL there, FF_CVOIDPTR
                    everywhere else.
  CHECK_SAME_BATCH  distance.cpp / posdef.cpp alone also rejected ndim < D
                    -> FF_CHECK_SAME_BATCH_ND there, FF_CHECK_SAME_BATCH
                    everywhere else.
  CHECK_SAME_SHAPE  TWO DIFFERENT MACROS SHARING ONE NAME: a 2-argument
                    whole-shape check in distance.cpp, a 3-argument leading-D
                    check in the regularisers and solve_field
                    -> FF_CHECK_SAME_SHAPE and FF_CHECK_SAME_SHAPE_N.

CHECK_NO_LANES and CHECK_SAME_DTYPE also had two spellings each, but those
differ only in line wrapping and expand identically.

Equivalence is not asserted, it is proved: tools/macro-equivalence.py
preprocesses every (file, macro) pair on both sides and compares token streams.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Which files carry which variant
# --------------------------------------------------------------------------

BACKENDS = ("lib-cpu", "lib-cuda")

DISPATCH_SOURCES = sorted(
    os.path.join("src", b, f)
    for b in BACKENDS
    for f in os.listdir(os.path.join(ROOT, "src", b))
    if f.endswith(".cpp")
) + [
    "include/fastfields/api/cpu/pushpull_dispatch.h",
    "include/fastfields/api/cuda/pushpull_dispatch.h",
]

HUB_SOURCES = sorted(
    os.path.join("src", "lib", f)
    for f in os.listdir(os.path.join(ROOT, "src", "lib"))
    if f.endswith(".cpp")
)

# The null-tolerant CVOIDPTR lived only in posdef.
CVOIDPTR_OR_NULL = {"src/lib-cpu/posdef.cpp", "src/lib-cuda/posdef.cpp"}

# The ndim-guarded CHECK_SAME_BATCH lived only in distance and posdef.
BATCH_ND = {
    "src/lib-cpu/distance.cpp",
    "src/lib-cpu/posdef.cpp",
    "src/lib-cuda/distance.cpp",
    "src/lib-cuda/posdef.cpp",
}

# CHECK_SAME_SHAPE's 2-argument whole-shape form lived only in distance; every
# other user had the 3-argument leading-D form.
SHAPE_WHOLE = {"src/lib-cpu/distance.cpp", "src/lib-cuda/distance.cpp"}

# Macros hoisted into core/dispatch.h. Order is irrelevant to correctness (the
# renames are whole-identifier) but longest-first reads more obviously safe.
HOISTED = [
    "CHECK_SAME_DTYPE",
    "CHECK_SAME_BATCH",
    "CHECK_SAME_SHAPE",
    "CHECK_NO_LANES",
    "CHECK_SAME",
    "CANUSE32BITS",
    "CVOIDPTR",
    "VOIDPTR",
]

HUB_MACROS = ["IS_CUDA", "IS_CPU"]

# The macro names this script introduces. A backslash-continuation block that
# mentions one of them is a block whose hand-alignment the rename just broke.
NEW_NAMES = re.compile(
    r"(?<![A-Za-z0-9_])FF_(?:VOIDPTR|CVOIDPTR|CVOIDPTR_OR_NULL|CANUSE32BITS"
    r"|CHECK_[A-Z_]+)(?![A-Za-z0-9_])"
)

# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def write(rel, text):
    with open(os.path.join(ROOT, rel), "w", encoding="utf-8") as fh:
        fh.write(text)


def drop_define(text, name):
    """Delete `#define <name> ...`, its backslash continuations, and the single
    blank line that separated it from whatever came next."""
    return re.sub(
        r"^[ \t]*#[ \t]*define[ \t]+" + re.escape(name) + r"\b"
        r"(?:[^\n]*\\\n)*[^\n]*\n"
        r"(?:[ \t]*\n)?",
        "",
        text,
        flags=re.MULTILINE,
    )


def rename(text, old, new):
    """Whole-identifier rename."""
    return re.sub(
        r"(?<![A-Za-z0-9_])" + re.escape(old) + r"(?![A-Za-z0-9_])", new, text
    )


def drop_banner(text, title):
    """Remove a `/*** TITLE ***/` box comment that no longer covers anything."""
    return re.sub(
        r"^/\*{5,}\*?\n"
        r"^[ \t]*\*[ \t]*" + re.escape(title) + r"[ \t]*\*[ \t]*\n"
        r"^[ \t]*\*{5,}/\n"
        r"(?:[ \t]*\n)?",
        "",
        text,
        flags=re.MULTILINE,
    )


def add_include(text, header, after):
    """Insert `#include "<header>"` right after the `after` include, once."""
    line = '#include "%s"\n' % header
    if line in text:
        return text
    anchor = '#include "%s"\n' % after
    if anchor not in text:
        raise SystemExit("no anchor %r to insert %r after" % (after, header))
    return text.replace(anchor, anchor + line, 1)


def realign_continuations(text):
    """Re-align the trailing backslash of every multi-line macro this rewrite
    made longer. The tree is hand-column-aligned and CI lints the lines a PR
    touches, so leaving hundreds of renamed lines with their backslashes shoved
    out of column would be the most visible thing in the diff."""
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        if not lines[i].endswith("\\"):
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].endswith("\\"):
            j += 1
        block = lines[i : j + 1]  # include the block's final, unbackslashed line
        if not any(NEW_NAMES.search(ln) for ln in block):
            out.extend(block)
        else:
            bodies = [ln[:-1].rstrip() if ln.endswith("\\") else ln for ln in block]
            width = max(len(b) for b in bodies[:-1]) + 1
            for k, body in enumerate(bodies):
                out.append(body if k == len(bodies) - 1 else body.ljust(width) + "\\")
        i = j + 1
    return "\n".join(out)


# --------------------------------------------------------------------------
# Pass 1 -- the two dtype-dispatch layers
# --------------------------------------------------------------------------


def convert_dispatch(rel):
    text = original = read(rel)

    for name in HOISTED:
        text = drop_define(text, name)
    # pushpull.cpp / pushpull_backward.cpp only *use* these macros -- their
    # definitions live in the shared pushpull_dispatch.h, which is itself in
    # this list. Such a file gets its call sites renamed but no new include:
    # the dispatch header already pulls core/dispatch.h in.
    defined_here = text != original

    # Per-file variant selection, before the generic renames below.
    if rel in CVOIDPTR_OR_NULL:
        text = rename(text, "CVOIDPTR", "FF_CVOIDPTR_OR_NULL")
    if rel in BATCH_ND:
        text = rename(text, "CHECK_SAME_BATCH", "FF_CHECK_SAME_BATCH_ND")
    if rel in SHAPE_WHOLE:
        text = rename(text, "CHECK_SAME_SHAPE", "FF_CHECK_SAME_SHAPE")
    else:
        text = rename(text, "CHECK_SAME_SHAPE", "FF_CHECK_SAME_SHAPE_N")

    for name in HOISTED:
        text = rename(text, name, "FF_" + name)

    # as_weights: three byte-identical copies -> core/dispatch.h.
    text = re.sub(
        r"\n// build a length-nc reduce_t vector from a \(possibly null\) double array\n"
        r"static inline std::vector<reduce_t> as_weights\(const double \* w, int64_t nc\)\n"
        r"\{\n(?:[^\n]*\n)*?\}\n(?:[ \t]*\n)?",
        "\n",
        text,
    )

    # _reg_stream: four byte-identical copies -> api/cuda/stream.h.
    text = re.sub(
        r"\n// int -> cudaStream_t \(0 == default stream\)\.[^\n]*\n"
        r"(?:// [^\n]*\n)*"
        r"static inline cudaStream_t _reg_stream\(intptr_t stream\)\n"
        r"\{\n(?:[^\n]*\n)*?\}\n(?:[ \t]*\n)?",
        "\n",
        text,
    )

    if text == original:
        return False

    if defined_here:
        text = drop_banner(text, "CHECKS")
        text = add_include(
            text, "fastfields/core/dispatch.h", "fastfields/core/autocast.h"
        )
    if rel.startswith("src/lib-cuda/") and "_reg_stream" in text:
        text = add_include(
            text, "fastfields/api/cuda/stream.h", "fastfields/core/dispatch.h"
        )
    text = realign_continuations(text)
    write(rel, text)
    return True


# --------------------------------------------------------------------------
# Pass 2 -- the hub
# --------------------------------------------------------------------------


def convert_hub(rel):
    text = original = read(rel)

    for name in HUB_MACROS:
        text = drop_define(text, name)
    text = rename(text, "IS_CUDA", "is_cuda")
    text = rename(text, "IS_CPU", "is_cpu")

    if text == original:
        return False

    # Every hub source but splinc.cpp already includes api/checks.h for
    # require_same_device -- splinc's single in/out tensor has no second
    # operand to disagree with it, so it never needed the header before.
    if '#include "fastfields/api/checks.h"' not in text:
        module = os.path.basename(rel)[: -len(".cpp")]
        text = add_include(
            text, "fastfields/api/checks.h", "fastfields/api/%s.h" % module
        )
    write(rel, text)
    return True


def main():
    changed = 0
    for rel in DISPATCH_SOURCES:
        if convert_dispatch(rel):
            changed += 1
            print("dispatch  %s" % rel)
    for rel in HUB_SOURCES:
        if convert_hub(rel):
            changed += 1
            print("hub       %s" % rel)
    print("%d file(s) rewritten" % changed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
