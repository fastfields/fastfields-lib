#!/usr/bin/env python3
"""
normalise-include-delimiters.py -- angle brackets for the public interface,
quotes for headers private to the including component.

THE RULE THIS APPLIES
--------------------------------------------------------------------------
    #include <fastfields/...>    the installed public interface, found on the
                                 include path (-I include)
    #include "sibling.h"         a header that lives next to the including
                                 file and is reached relative to it

`include/fastfields/` *is* the public installed interface: it is copied to an
install prefix and consumed from there by `fastfields-dlpack`, whose setup.py
puts it on the compiler's include path via `include_dirs`. The ordinary C++
convention spells a dependency found on the include path with `<>` and a
dependency found next to the includer with `""`, and this tree was writing
both categories the same way -- 437 `#include "fastfields/..."` and not one
`<fastfields/...>`.

That is carryover, not a decision: the consolidation moved every header under
one prefix and rewrote the *path* in every `#include`, leaving the delimiter
untouched because that was the minimal textual change. The clearest evidence
that the result reads wrong is downstream, in `fastfields-dlpack/src/ext.cpp`,
which writes `<nanobind/nanobind.h>` and `"fastfields/api/distance.h"` ten
lines apart -- two spellings for the same category of dependency.

WHY THIS IS BEHAVIOUR-PRESERVING HERE
--------------------------------------------------------------------------
`""` searches the including file's own directory first and then falls back to
exactly the `<>` search. So the two differ only when a quoted spelling is
resolved by that first step. `make/common.mk` sets

    INCLUDES += -I$(ROOTDIR)/include

and that is the whole include configuration -- every compile rule in the four
Makefiles uses `$(INCLUDES)` and nothing else, and there is **no `-iquote`
anywhere in the tree**, so `""` and `<>` share one search path. No directory
named `fastfields/` exists beside any source file, so no `fastfields/...`
include was ever being resolved by the relative step. `--check` re-derives
that guarantee from the sources rather than asking you to trust it, by
verifying that every quoted include really does resolve next to its includer.

WHAT IS NOT TOUCHED
--------------------------------------------------------------------------
The ~135 genuinely relative includes -- `"../utils.h"`, `"flow/2d.h"`,
`"utils.h"` -- are private to their component, correctly quoted, and left
exactly as they are. Standard-library and third-party includes already use
`<>`.

The two frozen one-shot scripts, `tools/consolidate.sh` and
`tools/dedup-dispatch-helpers.py`, quote include lines as *literals of the
tree at their own parent commit* -- their documented contract is to be
replayed against that parent, not against today's `main`. They are
deliberately not rewritten.

USAGE
--------------------------------------------------------------------------
    python3 tools/normalise-include-delimiters.py           # apply
    python3 tools/normalise-include-delimiters.py --check   # verify only

Idempotent and deterministic: re-running over a converted tree rewrites
nothing, and running over the pre-sweep tree reproduces the sweep exactly.

If this lands on a base that has moved, do NOT resolve conflicts by hand:
reset, re-run the script on the new base, and commit that.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIRS = ("include", "src", "tests")
SOURCE_EXTS = (".h", ".hpp", ".inl", ".cpp", ".cu", ".cuh")

# The one directory on the include path, per make/common.mk.
INCLUDE_PATH = (os.path.join(ROOT, "include"),)

# Verbatim third-party code: never rewritten. (It includes nothing of ours.)
VENDORED = {"include/fastfields/core/dlpack.h"}

PUBLIC_PREFIX = "fastfields/"

INCLUDE_RE = re.compile(r'^([ \t]*#[ \t]*include[ \t]*)(["<])([^">]+)([">])(.*)$')


def sources():
    for d in SOURCE_DIRS:
        for dirpath, _, filenames in os.walk(os.path.join(ROOT, d)):
            for name in sorted(filenames):
                if name.endswith(SOURCE_EXTS):
                    rel = os.path.relpath(os.path.join(dirpath, name), ROOT)
                    if rel not in VENDORED:
                        yield rel


def includes(rel):
    """(line number, delimiter, target) for every #include in the file."""
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            m = INCLUDE_RE.match(line.rstrip("\n"))
            if m:
                yield n, m.group(2), m.group(3).strip()


def on_include_path(target):
    return any(os.path.isfile(os.path.join(d, target)) for d in INCLUDE_PATH)


def beside(rel, target):
    return os.path.isfile(os.path.join(ROOT, os.path.dirname(rel), target))


def convert(text):
    out = []
    for line in text.split("\n"):
        m = INCLUDE_RE.match(line)
        if m and m.group(2) == '"' and m.group(3).strip().startswith(PUBLIC_PREFIX):
            line = "%s<%s>%s" % (m.group(1), m.group(3), m.group(5))
        out.append(line)
    return "\n".join(out)


def apply(check_only=False):
    changed = 0
    for rel in sources():
        path = os.path.join(ROOT, rel)
        with open(path, encoding="utf-8") as fh:
            original = fh.read()
        text = convert(original)
        if text != original:
            changed += 1
            if check_only:
                print("would rewrite %s" % rel)
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
    print("%d file(s)%s" % (changed, " would change" if check_only else " rewritten"))
    return changed


def violations():
    """Every include that does not match the convention. Must be empty.

    Three things are checked, and the last two are what make the first one
    safe rather than merely tidy:

      1. no `fastfields/...` include is still quoted;
      2. every quoted include resolves beside its includer -- which is the
         property that makes quotes-vs-angles a real distinction here, and
         proves no quoted include was relying on the include path;
      3. no angle-bracket include of ours resolves beside its includer, i.e.
         nothing private has been promoted to the public spelling.
    """
    hits = []
    for rel in sources():
        for n, delim, target in includes(rel):
            where = "%s:%d: %s" % (rel, n, target)
            if delim == '"':
                if target.startswith(PUBLIC_PREFIX):
                    hits.append("%s -- public header, must use <>" % where)
                elif not beside(rel, target):
                    hits.append("%s -- quoted but not beside the includer" % where)
            else:
                if beside(rel, target) and not on_include_path(target):
                    hits.append("%s -- private header, must use \"\"" % where)
    return hits


if __name__ == "__main__":
    check = "--check" in sys.argv
    apply(check_only=check)
    left = violations()
    if left:
        print("\nINCLUDES NOT MATCHING THE CONVENTION:")
        for h in left:
            print("  " + h)
    else:
        print(
            "\nevery public include uses <fastfields/...> and every quoted "
            "include resolves beside its includer."
        )
    sys.exit(1 if left else 0)
