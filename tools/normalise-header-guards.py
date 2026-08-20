#!/usr/bin/env python3
"""
normalise-header-guards.py -- `#pragma once` in every header, one convention.

THE RULE THIS APPLIES
--------------------------------------------------------------------------
Every header in this repository has `#pragma once` as its first line and
carries no whole-file `#ifndef`/`#define`/`#endif` guard. Line 1, with no
exception for a licence or provenance comment: "is line 1 `#pragma once`" is
a rule that needs no judgement to apply and none to check, and it is what the
headers already using the pragma did. Attribution comments keep their text
and simply sit one line lower.

The tree had drifted into four conventions at once -- 94 headers with a guard
only, 12 with the pragma only, 5 with both, and 6 with neither -- split by
which of the six pre-consolidation repositories a file came from rather than
by any decision. `#pragma once` is now the single convention.

`#pragma once` is not ISO C++, but every compiler this project targets or
plans to target supports it: clang, g++, nvcc and MSVC. The historical hazard
here -- that it keys on file identity, which was ambiguous when every
cross-repo dependency was a symlink -- is gone: `main` has no `.gitmodules`,
no gitlinks and no symlinks.

VENDORED FILES KEEP THEIR UPSTREAM GUARDS
--------------------------------------------------------------------------
`include/fastfields/core/dlpack.h` is verbatim upstream code and is never
touched. Its `DLPACK_DLPACK_H_` guard is the *upstream* macro, which is what
lets our copy and a system DLPack header with the same guard collapse into a
single inclusion -- something `#pragma once` cannot do for two distinct files.
Replacing it would break that interoperation, so it stays. `--check` treats it
as exempt but still reports it, and fails loudly if the path disappears (e.g.
after a file move) rather than silently dropping the exemption.

Six other headers carry third-party provenance, and all six are ADAPTATIONS
rather than drop-in vendored copies, so the pragma applies to them:

    impl/kernels/atomic.h            "CUDA portion copied from PyTorch/ATen"
    impl/kernels/parallel.h          "adapted from PyTorch/ATen ParallelNative"
    impl/kernels/parallel_impl.h     ditto
    impl/kernels/threadpool.h        YasserAsmi/wstpool (MIT)
    impl/kernels/threadpool.inl      "some of this is copied from pytorch/aten"
    impl/cuda/utils.h                two helpers "(Copied from PyTorch)"
    impl/kernels/distance/mesh.h     InteractiveComputerGraphics/TriangleMeshDistance

The test that settles it is not how much text came from upstream, it is
whether the file carries an upstream *guard macro* to interoperate with. None
of them does: every one is guarded by a name this project invented (`FF_ATOMIC`,
`FF_PARALLEL_H`, `FF_THREADPOOL_H`, ...), or -- in `impl/cuda/utils.h`'s case --
by nothing at all, since it already used `#pragma once` before this sweep. They
are also re-namespaced into `ff::` via `FF_NAMESPACE_BEGIN`, use this project's
`FF_CUHOST`/`FF_CUDEV` qualifiers, and include project headers, so no upstream
copy could substitute for them and none is on any include path. There is
nothing for a guard to interoperate with, and `dlpack.h` remains the only
vendored file. (Copyright notices are untouched either way -- this is about
include mechanics, not attribution.)

REMOVING A GUARD IS NOT AUTOMATICALLY INERT
--------------------------------------------------------------------------
A guard macro can be *tested* from outside the header that defines it. A
`#ifdef` on a guard name somewhere else in the tree makes that `#define`
load-bearing, and deleting it silently changes what compiles -- the one way
a sweep like this can break something without any diagnostic. So the script
does not assume: before removing anything it greps the whole repository for
every guard macro, and refuses to run at all if one is mentioned in a
compiled source outside its own header. `--check` runs the same audit, so a
future header that starts testing a guard name turns the check red.

Mentions outside compiled sources cannot affect what compiles, so they are
reported rather than blocking.

On the tree as of this commit the audit is clean: all 99 guard macros are
referenced exactly once, by their own `#ifndef`, with no `#ifdef` on a guard
name anywhere. The only mentions elsewhere are inert text in the frozen
`tools/consolidate.sh` (which quotes header text it generated), one prose
line in `MIGRATION-PROVENANCE.md`, and this script's own docstring.

WHAT IS NOT A HEADER GUARD, AND SURVIVES
--------------------------------------------------------------------------
Only a guard that brackets the *whole file* is removed. `#ifndef` blocks that
guard part of a file, or that exist to make a definition idempotent across
several files, are untouched. Two families matter here:

  * `FF_LIB_BOUND_SPLINE_T` -- eight `api/*.h` headers each wrap the shared
    `bound_t`/`spline_t` declarations in it so they can be co-included. This
    is exactly the job `#pragma once` cannot do (one macro, eight files), and
    it is depended on downstream: `fastfields-dlpack/src/ext.cpp` includes all
    eight and says so in a comment.
  * `FF_POSDEF_MAX_NBATCH`, `FF_PP_MAX_NBATCH`, `FF_RESIZE_MAX_NBATCH`,
    `FF_RESTRICT_MAX_NBATCH`, `FF_SPLINC_MAX_NBATCH`, `FF_AUTOCAST_PINNED_HOST`
    -- `#ifndef X / #define X <value>` overridable build knobs.

The guard finder only accepts an `#ifndef` that is the file's first directive,
whose `#define` has an empty replacement list, and whose `#endif` is the file's
last directive, so none of the above can be mistaken for a guard.

USAGE
--------------------------------------------------------------------------
    python3 tools/normalise-header-guards.py           # apply
    python3 tools/normalise-header-guards.py --check   # verify, change nothing

Idempotent and deterministic: re-running over a normalised tree rewrites
nothing, and running over the pre-sweep tree reproduces the sweep exactly.

If this lands on a base that has moved, do NOT resolve conflicts by hand:
reset, re-run the script on the new base, and commit that.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIRS = ("include", "src", "tests")
HEADER_EXTS = (".h", ".hpp", ".inl", ".cuh")
COMPILED_EXTS = HEADER_EXTS + (".cpp", ".cu")

# Verbatim third-party code: never rewritten, keeps its upstream guard.
VENDORED = {"include/fastfields/core/dlpack.h"}


def headers():
    for d in SOURCE_DIRS:
        for dirpath, _, filenames in os.walk(os.path.join(ROOT, d)):
            for name in sorted(filenames):
                if name.endswith(HEADER_EXTS):
                    yield os.path.relpath(os.path.join(dirpath, name), ROOT)


def repo_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", "build")]
        for name in sorted(filenames):
            yield os.path.relpath(os.path.join(dirpath, name), ROOT)


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def strip_comments(text):
    """Blank out comments so directive scanning cannot be fooled by an
    `#endif` inside one. Line structure is preserved."""
    out, i, n = [], 0, len(text)
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
    out, i = [], 0
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
    """(name, ifndef line, define line, endif line) for a whole-file guard, or
    None.

    Requires the first directive to be `#ifndef N`, the second to be
    `#define N` with an empty replacement list, and the conditional nesting
    that `#ifndef` opens to close only at the file's last directive -- so a
    partial-file `#ifndef`, or a `#define N <value>` build knob, is never
    mistaken for a guard. A leading `#pragma once` is stepped over."""
    ds = [d for d in directives(text) if d[1] != "pragma"]
    if len(ds) < 3 or ds[0][1] != "ifndef":
        return None
    name = ds[0][2]
    if not re.fullmatch(r"\w+", name or ""):
        return None
    if ds[1][1] != "define" or ds[1][2] != name:
        return None
    depth = 0
    for k, (line, d, _rest) in enumerate(ds):
        if d in ("if", "ifdef", "ifndef"):
            depth += 1
        elif d == "endif":
            depth -= 1
            if depth == 0:
                if k != len(ds) - 1:
                    return None
                return name, ds[0][0], ds[1][0], line
    return None


def external_references(names):
    """Every mention of a guard macro outside the header that defines it,
    split into (blocking, inert).

    Blocking = a mention in a compiled source, where it could be a `#ifdef`
    that the guard's `#define` is load-bearing for. Inert = a mention anywhere
    else (a script that quotes header text, a prose line in a doc, this
    script's own docstring), which cannot affect what compiles but is still
    reported so nobody has to wonder."""
    owners = {name: rel for name, (rel, _) in names.items()}
    pats = {n: re.compile(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(n))
            for n in names}
    blocking, inert = {}, {}
    for rel in repo_files():
        try:
            text = read(rel)
        except OSError:
            continue
        compiled = rel.startswith(SOURCE_DIRS) and rel.endswith(COMPILED_EXTS)
        for name, pat in pats.items():
            if owners[name] == rel or name not in text:
                continue
            for n, line in enumerate(text.split("\n"), 1):
                if pat.search(line):
                    bucket = blocking if compiled else inert
                    bucket.setdefault(name, []).append((rel, n, line.strip()[:90]))
    return blocking, inert


def guarded_headers():
    """{guard name: (rel, guard tuple)} for every non-vendored header that has
    a whole-file guard."""
    out = {}
    for rel in headers():
        if rel in VENDORED:
            continue
        g = find_guard(read(rel))
        if g:
            out[g[0]] = (rel, g)
    return out


def normalise(text):
    lines = text.split("\n")
    while lines and lines[-1] == "":
        lines.pop()

    g = find_guard(text)
    drop = set()
    if g:
        _name, ifndef_at, define_at, endif_at = g
        drop |= {ifndef_at, define_at, endif_at}
    for i, line in enumerate(lines):
        if re.match(r"\s*#\s*pragma\s+once\s*$", line):
            drop.add(i)
    lines = [l for i, l in enumerate(lines) if i not in drop]

    # A guard's `#define` was often followed by a blank line, and its `#endif`
    # preceded by one; drop what that leaves dangling rather than keeping a
    # gap where a directive used to be.
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)

    lines.insert(0, "#pragma once")
    return "\n".join(lines) + "\n"


def apply(check_only=False):
    guarded = guarded_headers()
    blocking, inert = external_references(guarded)
    if blocking:
        print("REFUSING TO REMOVE -- guard macro used in a compiled source:")
        for name in sorted(blocking):
            print("  %s (defined in %s)" % (name, guarded[name][0]))
            for rel, n, line in blocking[name]:
                print("      %s:%d: %s" % (rel, n, line))
        return -1
    if inert:
        print("guard macros mentioned outside compiled sources (inert):")
        for name in sorted(inert):
            for rel, n, line in inert[name]:
                print("  %s at %s:%d: %s" % (name, rel, n, line))

    changed = 0
    for rel in headers():
        if rel in VENDORED:
            continue
        path = os.path.join(ROOT, rel)
        original = read(rel)
        text = normalise(original)
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

    for rel in sorted(VENDORED):
        if not os.path.isfile(os.path.join(ROOT, rel)):
            hits.append(
                "%s: listed as vendored but missing -- if it moved, update "
                "VENDORED, do not drop the exemption" % rel
            )

    for rel in headers():
        text = read(rel)
        if rel in VENDORED:
            if find_guard(text) is None:
                hits.append("%s: vendored file lost its upstream guard" % rel)
            continue

        g = find_guard(text)
        if g:
            hits.append("%s: still has a whole-file guard (%s)" % (rel, g[0]))

        if text.split("\n", 1)[0].strip() != "#pragma once":
            hits.append("%s: first line is not #pragma once" % rel)
        if sum(1 for d in directives(text)
               if d[1] == "pragma" and d[2] == "once") != 1:
            hits.append("%s: not exactly one #pragma once" % rel)

    blocking, _inert = external_references(guarded_headers())
    for name in sorted(blocking):
        for rel, n, line in blocking[name]:
            hits.append("guard %s used at %s:%d: %s" % (name, rel, n, line))
    return hits


if __name__ == "__main__":
    check = "--check" in sys.argv
    if apply(check_only=check) < 0:
        sys.exit(1)
    left = violations()
    if left:
        print("\nHEADERS NOT MATCHING THE CONVENTION:")
        for h in left:
            print("  " + h)
    else:
        print(
            "\nevery header opens with #pragma once and carries no whole-file "
            "guard; %s keeps its upstream guard." % ", ".join(sorted(VENDORED))
        )
    sys.exit(1 if left else 0)
