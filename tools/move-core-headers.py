#!/usr/bin/env python3
"""
move-core-headers.py -- relocate the shared-infrastructure headers out of
`include/fastfields/impl/kernels/` and into `include/fastfields/core/`.

WHAT THIS APPLIES
--------------------------------------------------------------------------
`impl/kernels/` is meant to hold single-element math and nothing else, but it
also carries the helpers every layer above it uses: atomics, indexing, the
thread pool, the numeric utilities, the boundary/spline vocabulary. This
script moves those helpers to `core/` -- the directory that already holds
what `src/lib`, `src/lib-cpu` (host compiler) and `src/lib-cuda` (nvcc) all
share -- and rewrites every `#include` that named them.

After it runs, `impl/kernels/` contains only per-operation implementations.

The move is closed under dependency: every file in MOVES includes only
`core/` headers and other files in MOVES, so `core/` never acquires a
dependency on `impl/kernels/`. `--check` re-verifies that property rather
than trusting this comment.

INCLUDE REWRITING
--------------------------------------------------------------------------
Includes inside `impl/kernels/` are spelled relative ("utils.h",
"../bounds.h", "../../utils.h"); includes from `src/`, `tests/` and the other
`include/` subtrees are spelled absolutely ("fastfields/impl/kernels/utils.h").
Both forms are handled by resolving every include to a repository path,
applying the move map, and re-emitting:

  * same-directory after the move  -> quoted bare name  ("utils.h")
  * anything else                  -> the absolute form, in whichever
                                      delimiter the tree already uses for
                                      `fastfields/...` includes

That last point is what makes this script survive the in-flight
include-delimiter sweep (PR #146, `"fastfields/..."` -> `<fastfields/...>`):
it does not hard-code a delimiter, it measures the tree's dominant one and
matches it. Run it before that sweep and it emits quotes; run it after and it
emits angle brackets. Either way `--check` passes on its own output.

HEADER GUARDS ARE NOT TOUCHED
--------------------------------------------------------------------------
The guard-normalisation pass (PR #145) derives a guard from the file's path
only for guards it *adds*; existing guards keep their names, and its `--check`
enforces the guard's shape (present, whole-file, `FF_`-prefixed, unique), not
its spelling. So `FF_UTILS` stays `FF_UTILS` after moving to `core/`, and the
two passes do not fight. Nothing here renames a macro.

USAGE
    python3 tools/move-core-headers.py           # apply
    python3 tools/move-core-headers.py --check   # verify, change nothing

Idempotent: re-running over an already-moved tree rewrites nothing. If this
lands on a base that has moved, do NOT resolve conflicts by hand -- reset,
re-run the script on the new base, and commit that.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KERNELS = "include/fastfields/impl/kernels"
CORE = "include/fastfields/core"

# The shared infrastructure. Each entry is a bare filename under KERNELS that
# moves to the same filename under CORE. See the proposal for the per-file
# rationale; the short version is "used by layers above the kernels, and not
# the implementation of any one operation".
MOVES = [
    "atomic.h",          # ff::anyAtomicAdd -- the CPU/CUDA accumulate primitive
    "batch.h",           # linear index <-> sub-index / strided-index conversion
    "bounds.h",          # bound::type + BoundVec vocabulary, and the bound math
    "meta.h",            # Pack / Tuple template metaprogramming
    "parallel.h",        # parallel_for + the grain-size policy
    "parallel_impl.h",   # its backend selection (native / OpenMP / none)
    "spline.h",          # spline::type + SplineVec vocabulary, and weight math
    "threadpool.h",      # the work-stealing pool
    "threadpool.inl",    # its inline definitions + the thread-count accessors
    "utils.h",           # numeric/type helpers, canUse32BitIndexMath
]

SOURCE_DIRS = ("include", "src", "tests")
EXTS = (".h", ".hpp", ".inl", ".cuh", ".cpp", ".cu")

INCLUDE_RE = re.compile(r'(#\s*include\s*)([<"])([^">]+)([">])')


def move_map():
    """old repo-relative path -> new repo-relative path."""
    return {f"{KERNELS}/{n}": f"{CORE}/{n}" for n in MOVES}


def sources():
    for d in SOURCE_DIRS:
        for dirpath, _, filenames in os.walk(os.path.join(ROOT, d)):
            for name in sorted(filenames):
                if name.endswith(EXTS):
                    yield os.path.relpath(os.path.join(dirpath, name), ROOT)


def dominant_delimiter():
    """Whichever bracket the tree already uses for `fastfields/...` includes.

    Ties and empty trees fall back to quotes, which is what `main` uses today.
    """
    angle = quote = 0
    for rel in sources():
        with open(os.path.join(ROOT, rel), encoding="utf8", errors="replace") as fh:
            for m in INCLUDE_RE.finditer(fh.read()):
                if m.group(3).startswith("fastfields/"):
                    if m.group(2) == "<":
                        angle += 1
                    else:
                        quote += 1
    return ("<", ">") if angle > quote else ('"', '"')


def resolve(including_file, spelling):
    """Repo-relative path an include names, or None if it is not ours."""
    if spelling.startswith("fastfields/"):
        return "include/" + spelling
    if spelling.startswith(("<", "/")) or "." not in spelling:
        return None
    cand = os.path.normpath(os.path.join(os.path.dirname(including_file), spelling))
    return cand if os.path.exists(os.path.join(ROOT, cand)) else None


def rewrite(rel, mapping, delim):
    """Return the file's new text, or None if unchanged."""
    path = os.path.join(ROOT, rel)
    with open(path, encoding="utf8", errors="replace") as fh:
        text = fh.read()

    # Where this file itself ends up, so same-directory includes stay bare.
    new_self = mapping.get(rel, rel)

    def sub(m):
        pre, open_d, spelling, close_d = m.groups()
        target = resolve(rel, spelling)
        if target is None:
            return m.group(0)
        new_target = mapping.get(target, target)

        # Leave any edge alone whose EXISTING spelling still resolves to the
        # right file from the new location. Two files that move together keep
        # their relative position, so `parallel.h`'s `"parallel_impl.h"` needs
        # no edit at all -- and not editing it is what keeps the moved files
        # pure renames rather than modified files. That matters for more than
        # tidiness: `git-clang-format --diff` treats a renamed-and-modified
        # file as wholly new and demands a whole-file reformat, so a stray
        # one-line rewrite here costs hundreds of lines of unrelated churn.
        if not spelling.startswith("fastfields/"):
            still = os.path.normpath(
                os.path.join(os.path.dirname(new_self), spelling))
            if still == new_target:
                return m.group(0)
        elif new_target == target:
            return m.group(0)

        # Otherwise spell it absolutely. Every destination here is under the
        # public root, and `core/` already refers to its own siblings that way
        # (`core/dispatch.h` -> <fastfields/core/autocast.h>), so this matches
        # the convention of the directory the files land in rather than
        # importing `impl/kernels/`'s relative style along with them.
        if new_target.startswith("include/fastfields/"):
            pub = new_target[len("include/"):]
            return f"{pre}{delim[0]}{pub}{delim[1]}"
        newrel = os.path.relpath(new_target, os.path.dirname(new_self))
        return f'{pre}"{newrel}"'

    out = INCLUDE_RE.sub(sub, text)
    return out if out != text else None


def check_closure(mapping):
    """core/ must not end up depending on impl/kernels/."""
    problems = []
    for old, new in mapping.items():
        with open(os.path.join(ROOT, old if os.path.exists(os.path.join(ROOT, old)) else new),
                  encoding="utf8", errors="replace") as fh:
            text = fh.read()
        src = old if os.path.exists(os.path.join(ROOT, old)) else new
        for m in INCLUDE_RE.finditer(text):
            target = resolve(src, m.group(3))
            if target is None:
                continue
            target = mapping.get(target, target)
            if target.startswith(KERNELS):
                problems.append(f"{new} still depends on {target}")
    return problems


def main():
    check = "--check" in sys.argv[1:]
    mapping = move_map()
    delim = dominant_delimiter()

    missing = [o for o in mapping if not os.path.exists(os.path.join(ROOT, o))]
    done = all(os.path.exists(os.path.join(ROOT, n)) for n in mapping.values())

    problems = check_closure(mapping)

    changed = []
    for rel in sources():
        new = rewrite(rel, mapping, delim)
        if new is None:
            continue
        changed.append(rel)
        if not check:
            with open(os.path.join(ROOT, rel), "w", encoding="utf8") as fh:
                fh.write(new)

    if not check:
        for old, new in mapping.items():
            if os.path.exists(os.path.join(ROOT, old)):
                os.makedirs(os.path.join(ROOT, os.path.dirname(new)), exist_ok=True)
                subprocess.check_call(["git", "mv", old, new], cwd=ROOT)

    if check:
        if problems:
            for p in problems:
                print("DEPENDENCY LEAK:", p)
            return 1
        if changed:
            print(f"{len(changed)} file(s) would change:")
            for c in changed[:40]:
                print("   ", c)
            if len(changed) > 40:
                print(f"    ... and {len(changed) - 40} more")
            return 1
        if missing and not done:
            print("move map names files that do not exist:", missing)
            return 1
        print(f"clean; {len(MOVES)} header(s) in {CORE}, "
              f"include delimiter {delim[0]}...{delim[1]}, no dependency leak")
        return 0

    print(f"moved {len(MOVES)} header(s) to {CORE}; "
          f"rewrote includes in {len(changed)} file(s) "
          f"using {delim[0]}fastfields/...{delim[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
