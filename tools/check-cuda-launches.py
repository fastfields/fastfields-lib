#!/usr/bin/env python3
"""
check-cuda-launches.py -- every CUDA kernel launch goes through one helper.

THE RULE THIS ENFORCES
--------------------------------------------------------------------------
A CUDA kernel launch is asynchronous and does not throw. It reports failure
by setting the runtime's error state, and if nobody inspects that state the
failure is simply discarded: the launcher returns normally and the caller
reads whatever was already in the output buffer. That was the state of this
tree in fastfields-lib#152 -- 39 `<<<` sites, zero `cudaGetLastError`.

The fix was structural rather than conventional. `<<<` now appears in
exactly ONE place:

    include/fastfields/impl/cuda/launch.h    ff::cuda::launchKernel

and every launcher reaches it through the `FF_CUDA_LAUNCH` macro. This
script is what keeps that true. It fails if:

  1. `<<<` appears in code anywhere outside the helper;
  2. the helper does not contain exactly one `<<<`;
  3. the helper stops calling `cudaGetLastError` -- i.e. the funnel is
     still a funnel but no longer checks anything;
  4. `cudaLaunchKernel` / `cudaLaunchCooperativeKernel` is called outside
     the helper, which is the way to launch a kernel without writing `<<<`.

WHY A SCRIPT AND NOT A CONVENTION
--------------------------------------------------------------------------
There is no GPU in CI, so nothing here can *execute* a launch; compile+link
is the whole CUDA gate. A regression is still perfectly preventable, but
only mechanically. This project has learned the same lesson twice already --
`-Wl,--no-undefined` (#87) and the `FF_MEM_BUDGET_KB` budget (#95) both
exist because a rule people were supposed to remember was not remembered --
and #152 exists precisely because "check the launch" was never applied even
once, at any of the 39 sites, over the life of the code.

COMMENTS AND STRINGS ARE NOT CODE
--------------------------------------------------------------------------
Comments and string literals are blanked before scanning, so prose may
discuss `<<<` freely (launch.h's own header comment does, six times). That
is not laxity: it is what lets the "exactly one" half of the rule be stated
about the helper itself rather than waived for it.

USAGE
--------------------------------------------------------------------------
    python3 tools/check-cuda-launches.py            # check (the default)
    python3 tools/check-cuda-launches.py --check    # same, explicit
    python3 tools/check-cuda-launches.py --selftest # verify the checker

`--selftest` runs the analyser over synthetic inputs and asserts that it
both flags the violations it is supposed to flag and passes clean code. A
lint that has quietly stopped matching anything reports success forever;
this is what makes that failure mode visible. CI runs both modes.

Exit status
    0  the tree obeys the rule (or, with --selftest, the checker works)
    1  a violation was found (or the checker is broken)
    2  usage / environment error
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIRS = ("include", "src", "tests")
SOURCE_EXTS = (".h", ".hpp", ".inl", ".cpp", ".cu", ".cuh")

# The one file allowed to contain a launch.
HELPER = os.path.join("include", "fastfields", "impl", "cuda", "launch.h")

# The check the helper must still be performing. `cudaPeekAtLastError` alone
# is not enough: it does not clear, so a helper that only peeked would report
# the same stale error on every subsequent launch.
REQUIRED_IN_HELPER = "cudaGetLastError"

# Launching without writing `<<<`.
BACKDOORS = ("cudaLaunchKernel", "cudaLaunchCooperativeKernel")


# ---------------------------------------------------------------- stripping

def blank_comments_and_strings(text):
    """Replace comments and string/char literals with spaces.

    Newlines are preserved so reported line numbers stay meaningful, and the
    result has the same length as the input so column offsets do too.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
        elif c == "/" and nxt == "*":
            out.append("  ")
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            out.append("  ")
            i += 2
        elif c in ('"', "'"):
            quote = c
            out.append(" ")
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\" and i + 1 < n:
                    out.append("  ")
                    i += 2
                    continue
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            out.append(" ")
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def code_hits(text, needle):
    """[(line number, source line)] for `needle` outside comments/strings."""
    blanked = blank_comments_and_strings(text)
    raw = text.split("\n")
    hits = []
    for k, line in enumerate(blanked.split("\n")):
        if needle in line:
            hits.append((k + 1, raw[k].strip() if k < len(raw) else ""))
    return hits


# ------------------------------------------------------------------- checks

def iter_sources():
    for d in SOURCE_DIRS:
        top = os.path.join(ROOT, d)
        if not os.path.isdir(top):
            continue
        for dirpath, _dirnames, filenames in os.walk(top):
            for fn in sorted(filenames):
                if fn.endswith(SOURCE_EXTS):
                    path = os.path.join(dirpath, fn)
                    yield os.path.relpath(path, ROOT)


def check_tree():
    """Returns (list of problem strings, stats dict)."""
    problems = []
    launch_sites = 0
    helper_seen = False
    scanned = 0

    for rel in iter_sources():
        with open(os.path.join(ROOT, rel), "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        scanned += 1
        blanked = blank_comments_and_strings(text)
        is_helper = rel.replace("\\", "/") == HELPER.replace("\\", "/")
        if not is_helper:
            # The helper's own `#define` is not a call site.
            launch_sites += len(re.findall(r"\bFF_CUDA_LAUNCH\s*\(", blanked))

        if is_helper:
            helper_seen = True
            hits = code_hits(text, "<<<")
            if len(hits) != 1:
                problems.append(
                    "%s: expected exactly one `<<<` in code, found %d%s"
                    % (rel, len(hits),
                       "".join("\n      line %d: %s" % h for h in hits)))
            if REQUIRED_IN_HELPER not in blanked:
                problems.append(
                    "%s: the launch helper no longer calls `%s` -- every launch "
                    "still funnels through it, but nothing checks the result "
                    "(this is exactly fastfields-lib#152)"
                    % (rel, REQUIRED_IN_HELPER))
            continue

        for lineno, src in code_hits(text, "<<<"):
            problems.append(
                "%s:%d: kernel launch outside the helper\n      %s\n"
                "      use FF_CUDA_LAUNCH((kernel<...>), grid, block, shmem, "
                "stream, args...) from <fastfields/impl/cuda/launch.h>"
                % (rel, lineno, src))

        for backdoor in BACKDOORS:
            for lineno, src in code_hits(text, backdoor):
                problems.append(
                    "%s:%d: `%s` outside the helper -- that launches a kernel "
                    "without the error check\n      %s"
                    % (rel, lineno, backdoor, src))

    if not helper_seen:
        problems.append(
            "%s is missing: the launch helper is the whole mechanism this "
            "check exists to protect" % HELPER)
    if launch_sites == 0:
        problems.append(
            "no FF_CUDA_LAUNCH call sites found anywhere -- either the funnel "
            "was removed or this check is looking in the wrong place")

    return problems, {"scanned": scanned, "launch_sites": launch_sites}


# ----------------------------------------------------------------- selftest

_SELFTEST_CASES = [
    # (name, source, needle, expected number of code hits)
    ("bare launch", "void f() { k<<<1, 2, 0, s>>>(x); }", "<<<", 1),
    ("launch in a // comment", "// k<<<1,2>>>(x) is what this replaces\n", "<<<", 0),
    ("launch in a /* */ comment", "/* k<<<1,2>>>(x)\n   more */\nint x;\n", "<<<", 0),
    ("launch in a string", 'const char* s = "k<<<1,2>>>(x)";\n', "<<<", 0),
    ("stream operators are not launches", "std::cout << x << y;\n", "<<<", 0),
    ("shift then template", "a = b << c;\n", "<<<", 0),
    ("escaped quote does not swallow code",
     'const char* s = "\\"";\nvoid f() { k<<<1,1,0,s>>>(x); }\n', "<<<", 1),
    ("comment marker inside a string is not a comment",
     'const char* s = "// not a comment";\nvoid f() { k<<<1,1,0,s>>>(x); }\n',
     "<<<", 1),
    ("backdoor call", "cudaLaunchKernel(f, g, b, a, 0, s);\n", "cudaLaunchKernel", 1),
    ("backdoor in a comment", "// cudaLaunchKernel is the other way in\n",
     "cudaLaunchKernel", 0),
]


def selftest():
    failures = []
    for name, src, needle, expected in _SELFTEST_CASES:
        got = len(code_hits(src, needle))
        if got != expected:
            failures.append(
                "  %-45s expected %d hit(s) for %r, got %d"
                % (name, expected, needle, got))

    # Line numbers must survive the blanking, or a report points at the wrong
    # place -- which is how a lint stops being actionable.
    hits = code_hits("/* a\n   b */\nint x;\nvoid f(){ k<<<1,1,0,s>>>(y); }\n", "<<<")
    if hits != [] and hits[0][0] != 4:
        failures.append("  %-45s expected the hit on line 4, got line %d"
                        % ("line numbers survive blanking", hits[0][0]))

    # And the whole-tree check has to actually be able to fail.
    if not code_hits("k<<<1,1,0,s>>>(x);", "<<<"):
        failures.append("  %-45s the detector matches nothing at all"
                        % "detector is live")

    if failures:
        print("check-cuda-launches: SELFTEST FAILED -- the checker is broken, "
              "so a green run below would mean nothing:")
        print("\n".join(failures))
        return 1
    print("check-cuda-launches: selftest passed (%d cases)" % (len(_SELFTEST_CASES) + 2))
    return 0


# --------------------------------------------------------------------- main

def main(argv):
    mode = "check"
    for arg in argv[1:]:
        if arg in ("--check", "--selftest"):
            mode = arg[2:]
        elif arg in ("-h", "--help"):
            print(__doc__.strip())
            return 0
        else:
            sys.stderr.write("check-cuda-launches: unknown argument: %s\n" % arg)
            return 2

    if mode == "selftest":
        return selftest()

    problems, stats = check_tree()
    if problems:
        print("check-cuda-launches: %d problem(s):\n" % len(problems))
        for p in problems:
            print("  " + p)
        print("\nEvery CUDA kernel launch must go through "
              "ff::cuda::launchKernel in %s, which checks cudaGetLastError and "
              "throws with the kernel name and launch configuration. See "
              "fastfields-lib#152." % HELPER)
        return 1

    print("check-cuda-launches: clean -- %d source file(s) scanned, "
          "1 launch site in %s, %d FF_CUDA_LAUNCH call site(s)."
          % (stats["scanned"], HELPER, stats["launch_sites"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
