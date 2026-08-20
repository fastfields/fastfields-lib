#!/usr/bin/env python3
"""
normalise-header-guards.py -- one include-guard convention for every header.

THE RULE THIS APPLIES
--------------------------------------------------------------------------
Every header in this repository is bracketed by an `#ifndef`/`#define`/
`#endif` include guard, and no header uses `#pragma once`.

    #ifndef FF_SOMETHING_H
    #define FF_SOMETHING_H
    ...
    #endif // FF_SOMETHING_H

The guard must open before any other preprocessor directive and close as the
last directive in the file -- a guard that does not span the whole file is not
a guard. Its name must be `FF_`-prefixed (the rule the public-macro pass
established: a guard is a macro on the installed surface like any other) and
unique across the tree.

WHY GUARDS AND NOT `#pragma once`
--------------------------------------------------------------------------
Both work; the tree had drifted into four conventions at once (guard-only,
pragma-only, both, and six headers with neither), split by which of the six
pre-consolidation repositories a file came from rather than by any decision.
The deciding argument is what `include/fastfields/` *is*: the public installed
interface, copied to an install prefix and consumed from there by
`fastfields-dlpack`.

`#pragma once` keys on file identity. Two *copies* of the same logical header
-- the build-tree one reached through `-I include` and the installed one
reached through the prefix, in a single translation unit -- are two files, so
`#pragma once` includes both and the second one redefines everything the first
defined. A macro guard is the mechanism that makes those two copies collapse
into one inclusion, because it keys on a name rather than on an inode.

That is not a hypothetical worry we invented for this sweep: it is exactly the
reason `core/dlpack.h` must keep its upstream `DLPACK_DLPACK_H_` guard -- so
that our vendored copy and a system DLPack header interoperate. The property
the vendored header needs is the property an installed header needs, and there
is no reason for the public surface to hold itself to a weaker rule than the
one file everybody already agrees must be guarded. Choosing guards is also the
only choice under which `dlpack.h` is *conformant as it stands* rather than a
carve-out: the convention costs this tree zero exemptions.

(The historical objection to `#pragma once` here -- that every cross-repo
dependency was a symlink, so "the same file" had several identities -- is
indeed gone: `main` has no `.gitmodules`, no gitlinks and no symlinks. It is
just not the argument that decides it. Nor is portability: clang, g++, nvcc
and MSVC all support the pragma. Belt-and-braces "both" is defensible too, and
was in use in five headers, but it buys nothing over the guard alone -- the
guard is what does the work in every case where the two differ -- at the cost
of a second thing to keep in sync in 117 files.)

GUARD NAMES
--------------------------------------------------------------------------
A guard this script *adds* is derived from the file's path under
`include/fastfields/`, uppercased, with every non-alphanumeric character
(including the extension dot) turned into `_`:

    include/fastfields/impl/cuda/utils.h  ->  FF_IMPL_CUDA_UTILS_H

Derivation makes the name unique by construction and gives new headers a rule
to follow instead of a precedent to guess at.

Guards that already exist keep their names. They are unique and already
`FF_`-prefixed, so renaming 99 working macros would be churn: it buys nothing
a reader can use, it invalidates the literal guard text quoted in the frozen
`tools/consolidate.sh`, and it multiplies the conflict surface against the
long-running `teeny` branch. The convention this script enforces is the
*shape* of the guard, which is what had actually drifted; `--check` enforces
the properties that matter (present, whole-file, prefixed, unique), not a
spelling.

VENDORED FILES
--------------------------------------------------------------------------
`include/fastfields/core/dlpack.h` is verbatim upstream code. It is never
rewritten, and `--check` exempts it from the `FF_` prefix requirement only --
it must still carry a whole-file guard and must still not use `#pragma once`,
both of which are already true of it today.

It is the only vendored file. `impl/kernels/threadpool.h` carries a
third-party copyright (wstpool, MIT) but is adapted rather than verbatim -- it
already uses this project's namespace macros and an `FF_`-prefixed guard -- so
it is treated as project code.

USAGE
--------------------------------------------------------------------------
    python3 tools/normalise-header-guards.py           # apply
    python3 tools/normalise-header-guards.py --check   # verify, change nothing

Idempotent and deterministic: re-running over an already-normalised tree
rewrites nothing, and running over the pre-sweep tree reproduces the sweep
exactly.

If this lands on a base that has moved, do NOT resolve conflicts by hand:
reset, re-run the script on the new base, and commit that.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIRS = ("include", "src", "tests")
HEADER_EXTS = (".h", ".hpp", ".inl", ".cuh")

# Verbatim third-party code: never rewritten, and exempt from the FF_ prefix
# requirement (only from that -- see the module docstring).
VENDORED = {"include/fastfields/core/dlpack.h"}

PUBLIC_ROOT = os.path.join("include", "fastfields")


def headers():
    for d in SOURCE_DIRS:
        for dirpath, _, filenames in os.walk(os.path.join(ROOT, d)):
            for name in sorted(filenames):
                if name.endswith(HEADER_EXTS):
                    yield os.path.relpath(os.path.join(dirpath, name), ROOT)


def derived_guard(rel):
    """FF_ + the path under include/fastfields/, uppercased, non-alnum -> _."""
    stem = rel[len(PUBLIC_ROOT) + 1:] if rel.startswith(PUBLIC_ROOT + os.sep) else rel
    return "FF_" + re.sub(r"[^A-Za-z0-9]", "_", stem).upper()


def strip_comments(text):
    """Blank out block/line comments so directive scanning cannot be fooled by
    a `#endif` inside a comment. Newlines are preserved, so line numbers and
    line count are unchanged."""
    out = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(c if c == "\n" else " " for c in text[i:j]))
            i = j
        elif text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def directives(text):
    """(line index, directive, rest) for every preprocessor directive, with
    line continuations respected and comments already blanked."""
    lines = strip_comments(text).split("\n")
    out = []
    i = 0
    while i < len(lines):
        m = re.match(r"\s*#\s*(\w+)\s*(.*)$", lines[i])
        start = i
        while lines[i].rstrip().endswith("\\") and i + 1 < len(lines):
            i += 1
        if m:
            out.append((start, m.group(1), m.group(2).strip()))
        i += 1
    return out


def find_guard(text):
    """The whole-file guard name, or None.

    Requires: the first directive is `#ifndef N`, the second is `#define N`
    with an empty replacement list, and the conditional nesting introduced by
    that `#ifndef` closes only at the file's last directive.

    A leading `#pragma once` is stepped over rather than rejected, so that a
    "both styles" header is recognised as guarded and keeps its guard name
    when the pragma is dropped."""
    ds = [d for d in directives(text) if d[1] != "pragma"]
    if len(ds) < 3:
        return None
    if ds[0][1] != "ifndef":
        return None
    name = ds[0][2]
    if not re.fullmatch(r"\w+", name or ""):
        return None
    if ds[1][1] != "define" or ds[1][2] != name:
        return None
    depth = 0
    for k, (_, d, _rest) in enumerate(ds):
        if d in ("if", "ifdef", "ifndef"):
            depth += 1
        elif d == "endif":
            depth -= 1
            if depth == 0:
                return name if k == len(ds) - 1 else None
    return None


def prologue_end(lines):
    """Index of the first line that is neither blank nor part of the leading
    comment block -- i.e. where the guard goes."""
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("//"):
            i += 1
            continue
        if s.startswith("/*"):
            while i < n and "*/" not in lines[i]:
                i += 1
            i += 1
            continue
        break
    return i


def normalise(rel, text):
    name = find_guard(text) or derived_guard(rel)

    lines = text.split("\n")
    trailing_newline = lines and lines[-1] == ""
    if trailing_newline:
        lines.pop()

    had_guard = find_guard(text) is not None

    # 1. drop every `#pragma once`, and the blank line it may leave behind at
    #    the very top of the file.
    kept = [l for l in lines if not re.match(r"\s*#\s*pragma\s+once\s*$", l)]
    if len(kept) != len(lines):
        lines = kept
        while lines and not lines[0].strip():
            lines.pop(0)

    # 2. open the guard if the file has none.
    if not had_guard:
        at = prologue_end(lines)
        lines[at:at] = ["#ifndef %s" % name, "#define %s" % name]

    # 3. close it, and normalise the closing comment. `#endif FF_X` (no `//`)
    #    is ill-formed -- extra tokens after #endif -- and was in the tree.
    if had_guard:
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                lines[i] = "#endif // %s" % name
                break
    else:
        lines.append("#endif // %s" % name)

    return "\n".join(lines) + ("\n" if trailing_newline or not had_guard else "")


def apply(check_only=False):
    changed = 0
    for rel in headers():
        if rel in VENDORED:
            continue
        path = os.path.join(ROOT, rel)
        with open(path, encoding="utf-8") as fh:
            original = fh.read()
        text = normalise(rel, original)
        if text != original:
            changed += 1
            if check_only:
                print("would rewrite %s" % rel)
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
    print("%d header(s)%s" % (changed, " would change" if check_only else " rewritten"))
    return changed


def violations():
    """Every header that does not satisfy the convention. Must be empty."""
    hits = []
    seen = {}
    for rel in headers():
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            text = fh.read()

        for n, line in enumerate(text.split("\n"), 1):
            if re.match(r"\s*#\s*pragma\s+once\s*$", line):
                hits.append("%s:%d: #pragma once" % (rel, n))

        name = find_guard(text)
        if name is None:
            hits.append("%s: no whole-file include guard" % rel)
            continue
        if rel not in VENDORED and not name.startswith("FF_"):
            hits.append("%s: guard %s is not FF_-prefixed" % (rel, name))
        if name in seen:
            hits.append("%s: guard %s collides with %s" % (rel, name, seen[name]))
        seen[name] = rel

        last = [l for l in text.split("\n") if l.strip()][-1].strip()
        if rel not in VENDORED and last != "#endif // %s" % name:
            hits.append("%s: closes with %r, want '#endif // %s'" % (rel, last, name))
    return hits


if __name__ == "__main__":
    check = "--check" in sys.argv
    apply(check_only=check)
    left = violations()
    if left:
        print("\nHEADERS NOT MATCHING THE CONVENTION:")
        for h in left:
            print("  " + h)
    else:
        print(
            "\nevery header carries a whole-file, FF_-prefixed, unique include "
            "guard and no #pragma once."
        )
    sys.exit(1 if left else 0)
