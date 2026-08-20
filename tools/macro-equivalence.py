#!/usr/bin/env python3
"""
macro-equivalence.py -- prove that each hoisted FF_* macro expands
token-for-token to the local macro it replaced.

WHY THIS EXISTS
--------------------------------------------------------------------------
De-duplicating a macro that exists in 19 copies is only safe if the surviving
copy expands to the same tokens as every copy it replaced. Eyeballing 19 macro
bodies does not establish that -- three of these had genuinely diverged, and
one pair were two *different* macros sharing a name. So do not eyeball: run
the preprocessor on both sides and compare the token streams.

    python3 tools/macro-equivalence.py <old-tree> <new-tree>

`<old-tree>` is a checkout of the parent commit (a `git worktree add --detach`
is the cheapest way to get one); `<new-tree>` is the converted tree. Exits
non-zero on any mismatch, so it can be dropped into CI or a pre-merge check.

WHAT IT COMPARES
--------------------------------------------------------------------------
For every (dispatch source, macro) pair that existed on the old tree, it
preprocesses `MACRO(<placeholder args>)` twice:

  * once with only that file's own local #defines in scope (the old body), and
  * once with only core/dispatch.h's FF_* #defines in scope (the new body),

then tokenises both expansions and compares. Placeholder arguments (`T`, `A`,
`B`, `D`, `MSG`) are used rather than real expressions, so the comparison is of
the macro bodies alone and cannot be confounded by the call sites.

WHY NOT JUST DIFF WHOLE PREPROCESSED TRANSLATION UNITS
--------------------------------------------------------------------------
Tried that first; it is the wrong instrument. Two unavoidable, harmless
differences swamp the signal:

  * core/dispatch.h pulls in <vector>, which reorders standard-library
    declarations in the -E output of every TU that now includes it;
  * as_weights() is now visible in every dispatch TU rather than only in the
    three that defined it.

Both are additive and irrelevant to behaviour, but they make a whole-TU token
diff useless. Comparing the macro expansions in isolation asks the question
that actually matters.

THE ONE ACCEPTED DIFFERENCE
--------------------------------------------------------------------------
FF_CVOIDPTR_OR_NULL is defined by composing FF_CVOIDPTR, so its expansion
carries one extra *balanced parenthesis pair* around the non-null branch that
posdef.cpp's hand-written CVOIDPTR did not have:

    old:  (x.data ?  static_cast<const void*>(...)  : nullptr)
    new:  (x.data ? (static_cast<const void*>(...)) : nullptr)

That is reported as OK(parens) after checking the two token streams are
identical once parentheses are removed -- i.e. only grouping changed, and
grouping that was already unambiguous. Every other pair must match exactly.
"""

import difflib
import os
import re
import subprocess
import sys

TOK = re.compile(r'"(?:\\.|[^"\\])*"|[A-Za-z_]\w*|\d[\w.]*|[^\s]')

# Placeholder invocation per macro (keyed by the *old* name, with the two
# same-named CHECK_SAME_SHAPE variants disambiguated).
CALLS = {
    "VOIDPTR":           "(T)",
    "CVOIDPTR":          "(T)",
    "CANUSE32BITS":      "(T)",
    "CHECK_NO_LANES":    "(T)",
    "CHECK_SAME":        "(A, B, MSG)",
    "CHECK_SAME_DTYPE":  "(A, B)",
    "CHECK_SAME_BATCH":  "(A, B, D)",
    "CHECK_SAME_SHAPE2": "(A, B)",
    "CHECK_SAME_SHAPEN": "(A, B, D)",
}

HOISTED = (
    "VOIDPTR", "CVOIDPTR", "CANUSE32BITS", "CHECK_NO_LANES",
    "CHECK_SAME", "CHECK_SAME_DTYPE", "CHECK_SAME_BATCH", "CHECK_SAME_SHAPE",
)

# The three preserved divergences (kept in sync with
# tools/dedup-dispatch-helpers.py).
CVOIDPTR_OR_NULL = {"src/lib-cpu/posdef.cpp", "src/lib-cuda/posdef.cpp"}
BATCH_ND = {
    "src/lib-cpu/distance.cpp", "src/lib-cpu/posdef.cpp",
    "src/lib-cuda/distance.cpp", "src/lib-cuda/posdef.cpp",
}
SHAPE_WHOLE = {"src/lib-cpu/distance.cpp", "src/lib-cuda/distance.cpp"}


def defines_in(path):
    """name -> the full `#define` text (continuations included), for one file."""
    lines = open(path, encoding="utf-8").read().split("\n")
    out, i = {}, 0
    while i < len(lines):
        m = re.match(r"\s*#\s*define\s+([A-Za-z_]\w*)", lines[i])
        if m:
            body = [lines[i]]
            while body[-1].rstrip().endswith("\\") and i + 1 < len(lines):
                i += 1
                body.append(lines[i])
            out[m.group(1)] = "\n".join(body)
        i += 1
    return out


def expand(prelude, call):
    """Preprocess `call` with `prelude` in scope; return its token list."""
    src = prelude + "\nSTART " + call + " END\n"
    r = subprocess.run(
        ["clang++", "-E", "-P", "-x", "c++", "-"],
        input=src, capture_output=True, text=True,
    )
    if r.returncode:
        return None
    t = TOK.findall(r.stdout)
    return t[t.index("START") + 1 : len(t) - 1 - t[::-1].index("END")]


def new_name_for(rel, old):
    if old == "CHECK_SAME_SHAPE":
        return ("FF_CHECK_SAME_SHAPE", "CHECK_SAME_SHAPE2") if rel in SHAPE_WHOLE \
               else ("FF_CHECK_SAME_SHAPE_N", "CHECK_SAME_SHAPEN")
    if old == "CVOIDPTR":
        return ("FF_CVOIDPTR_OR_NULL" if rel in CVOIDPTR_OR_NULL
                else "FF_CVOIDPTR", old)
    if old == "CHECK_SAME_BATCH":
        return ("FF_CHECK_SAME_BATCH_ND" if rel in BATCH_ND
                else "FF_CHECK_SAME_BATCH", old)
    return "FF_" + old, old


def main(base, new):
    newdefs = defines_in(
        os.path.join(new, "include/fastfields/core/dispatch.h"))
    new_prelude = "\n".join(v for k, v in newdefs.items() if k.startswith("FF_"))

    files = []
    for d in ("src/lib-cpu", "src/lib-cuda"):
        files += [d + "/" + f
                  for f in sorted(os.listdir(os.path.join(base, d)))
                  if f.endswith(".cpp")]
    files += ["include/fastfields/api/cpu/pushpull_dispatch.h",
              "include/fastfields/api/cuda/pushpull_dispatch.h"]

    checked = bad = parens = 0
    for rel in files:
        old = defines_in(os.path.join(base, rel))
        old_prelude = "\n".join(v for k, v in old.items() if k in HOISTED)
        for name in HOISTED:
            if name not in old:
                continue
            new_name, call_key = new_name_for(rel, name)
            args = CALLS[call_key]
            a = expand(old_prelude, name + args)
            b = expand(new_prelude, new_name + args)
            checked += 1
            if a == b:
                continue
            if (new_name == "FF_CVOIDPTR_OR_NULL" and a and b and
                    [t for t in a if t not in "()"] ==
                    [t for t in b if t not in "()"]):
                parens += 1
                print("  OK(parens)  %-24s %s -> %s" % (rel, name, new_name))
                continue
            bad += 1
            print("MISMATCH %s  %s -> %s" % (rel, name, new_name))
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                    None, a or [], b or [], autojunk=False).get_opcodes():
                if tag == "equal":
                    continue
                print("    OLD: %s" % " ".join((a or [])[i1:i2])[:200])
                print("    NEW: %s" % " ".join((b or [])[j1:j2])[:200])

    print("\n%d macro expansions compared across %d files: "
          "%d exact, %d balanced-paren-only, %d MISMATCHED"
          % (checked, len(files), checked - parens - bad, parens, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
