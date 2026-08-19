#!/usr/bin/env bash
#
# consolidate.sh -- build the consolidated fastfields tree from the six source
# repositories, deterministically and repeatably.
#
# ############################################################################
# #                                                                          #
# #   WARNING -- THE PATH-REWRITE RULES BELOW MUST NEVER BE RE-RUN WITH      #
# #   VARIATIONS AGAINST A SUBSET OF REFS.                                   #
# #                                                                          #
# #   RUN git filter-repo EXACTLY ONCE PER SOURCE REPOSITORY, OVER `main`    #
# #   AND `teeny` TOGETHER IN A SINGLE PASS (--refs main teeny).             #
# #                                                                          #
# #   THIS WAS PROVEN BY EXPERIMENT, NOT ASSUMED. IDENTICAL RULES APPLIED    #
# #   TO A REPO'S `main` AND `teeny` REWRITE THEIR SHARED ANCESTORS TO       #
# #   IDENTICAL SHAs, SO THE MERGE-BASE SURVIVES AND `teeny` STAYS           #
# #   MERGEABLE. CHANGING A SINGLE PATH RULE BETWEEN RUNS PRODUCED 0 SHARED  #
# #   COMMITS AND AN EMPTY MERGE-BASE -- `teeny` PERMANENTLY UN-MERGEABLE.   #
# #                                                                          #
# #   RULE DRIFT IS THE FATAL FAILURE MODE, NOT BRANCH STRUCTURE. IF THE     #
# #   RULES MUST CHANGE, THROW AWAY EVERY REWRITTEN REF AND RE-RUN THIS      #
# #   SCRIPT FROM PRISTINE CLONES -- DO NOT PATCH A REWRITTEN TREE IN PLACE. #
# #                                                                          #
# ############################################################################
#
# What it does
# ------------
#   1. clone the six source repos pristine (or reuse --src clones)
#   2. per repo: ONE `git filter-repo --refs main teeny` pass that moves paths
#      into the consolidated layout. Path moves only -- no content is touched,
#      because content edits would have to be replayed identically on `teeny`
#      and they cannot be (its content differs).
#   3. merge the six rewritten `main`s into one history
#      (`--allow-unrelated-histories`; the six path sets are disjoint by
#      construction, so the merges cannot conflict)
#   4. content commits on `main` only: core/ dedupe, include rewrite, build
#      system. `teeny` is left rewritten-but-unbuilt for phase 2.
#
# Target layout
# -------------
#   include/fastfields/api/            fastfields-lib      headers
#   include/fastfields/api/cpu/        fastfields-cpu-lib  headers
#   include/fastfields/api/cuda/       fastfields-cuda-lib headers
#   include/fastfields/impl/cpu/       fastfields-cpu-impl
#   include/fastfields/impl/cuda/      fastfields-cuda-impl
#   include/fastfields/impl/kernels/   fastfields-kernels
#   include/fastfields/core/           dlpack.h autocast.h defines.h cuda_switch.h
#   src/lib/  src/lib-cpu/  src/lib-cuda/          *.cpp + per-group Makefile
#   tests/lib/ tests/lib-cpu/ tests/impl-cuda/ tests/kernels/
#   make/common.mk                                  shared makefile fragment
#   ci/legacy/<repo>/                               the six repos' old workflows
#   docs/legacy/<repo>/                             the five non-hub repos' docs
#
# Usage
#   tools/consolidate.sh --out DIR [--src DIR] [--keep-src]
#
#   --out DIR     where the consolidated repo is created (wiped if it exists)
#   --src DIR     directory holding/receiving the six pristine clones
#                 (default: <out>/../src). Clones are made only if absent.
#   --keep-src    do not re-clone even if the clones look dirty
#   -h|--help     this text
#
# Requires: git, git-filter-repo (pip install git-filter-repo), python3.

set -euo pipefail

die() { printf 'consolidate: %s\n' "$*" >&2; exit 2; }
say() { printf '\n=== %s\n' "$*" >&2; }

GITHUB="https://github.com/fastfields"
REPOS="fastfields-lib fastfields-cpu-lib fastfields-cuda-lib fastfields-cpu-impl fastfields-cuda-impl fastfields-kernels"

OUT=""; SRC=""; KEEP_SRC=0
while [ $# -gt 0 ]; do
    case "$1" in
        --out)      OUT="${2:-}"; shift 2 ;;
        --src)      SRC="${2:-}"; shift 2 ;;
        --keep-src) KEEP_SRC=1; shift ;;
        -h|--help)  sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//' | head -n -1; exit 0 ;;
        *)          die "unknown argument: $1 (try --help)" ;;
    esac
done
[ -n "$OUT" ] || die "--out is required (try --help)"

command -v git-filter-repo >/dev/null || die "git-filter-repo not found (pip install git-filter-repo)"
command -v python3         >/dev/null || die "python3 not found"

mkdir -p -- "$OUT"; OUT="$(cd -- "$OUT" && pwd)"
[ -n "$SRC" ] || SRC="$(dirname -- "$OUT")/src"
mkdir -p -- "$SRC"; SRC="$(cd -- "$SRC" && pwd)"

# ---------------------------------------------------------------------------
# 1. pristine clones
# ---------------------------------------------------------------------------
say "stage 1: pristine clones in $SRC"
for r in $REPOS; do
    if [ -d "$SRC/$r/.git" ]; then
        printf '  reuse %s\n' "$r" >&2
    else
        git clone --quiet "$GITHUB/$r" "$SRC/$r" || die "clone failed: $r"
    fi
    git -C "$SRC/$r" fetch --quiet origin main teeny 2>/dev/null || true
done

# ---------------------------------------------------------------------------
# 2. path rewrite -- ONE filter-repo pass per repo, over main AND teeny
# ---------------------------------------------------------------------------
#
# Every rule below is a pure path move. The rules are written so that the six
# repos' output path sets are pairwise DISJOINT, which is what makes step 3's
# merges conflict-free without any -X strategy option.
#
# Files that are not source (README, LICENSE, lint config) are parked under
# docs/legacy/<repo>/ rather than dropped, so `git log`/`git blame` on them
# still works. Workflows are parked under ci/legacy/<repo>/ -- outside
# .github/, so they are inert until phase 3 writes the real CI.
#
# The trailing catch-all in each callback matters: history contains paths that
# no longer exist on the tip, and every one of them must land somewhere
# deterministic.

rules_for() {
    # $1 = repo name. Emits a python filename-callback body on stdout.
    local repo="$1" short
    case "$repo" in
        fastfields-lib)       short=lib ;;
        fastfields-cpu-lib)   short=cpu-lib ;;
        fastfields-cuda-lib)  short=cuda-lib ;;
        fastfields-cpu-impl)  short=cpu-impl ;;
        fastfields-cuda-impl) short=cuda-impl ;;
        fastfields-kernels)   short=kernels ;;
    esac

    # Shared preamble: drop submodule gitlinks + .gitmodules (there are no
    # submodules on the consolidated main), park workflows.
    cat <<PYPRE
f = filename.decode('utf-8', 'surrogateescape')
if f in ('.gitmodules', 'cpu', 'cuda', 'impl', 'kernels'):
    return None
if f.startswith('.github/'):
    return ('ci/legacy/$short/' + f[len('.github/'):]).encode()
def legacy(p):
    return ('docs/legacy/$short/' + p).encode()
PYPRE

    case "$repo" in
    fastfields-lib)
        # The hub keeps the repository root: its README/LICENSE/CLAUDE.md and
        # its docs/, tools/, and lint config become the consolidated repo's.
        cat <<'PY'
if f.startswith('tests/'):
    return ('tests/lib/' + f[len('tests/'):]).encode()
if f.startswith('docs/') or f.startswith('tools/'):
    return filename
if '/' not in f and f.endswith('.cpp'):
    return ('src/lib/' + f).encode()
if '/' not in f and f.endswith('.h'):
    return ('include/fastfields/api/' + f).encode()
if '/' not in f:
    return filename
return legacy(f)
PY
        ;;
    fastfields-cpu-lib|fastfields-cuda-lib)
        case "$repo" in
            fastfields-cpu-lib)  api=cpu;  grp=lib-cpu ;;
            fastfields-cuda-lib) api=cuda; grp=lib-cuda ;;
        esac
        cat <<PY
if f.startswith('tests/'):
    return ('tests/$grp/' + f[len('tests/'):]).encode()
if f == 'Makefile':
    return b'src/$grp/Makefile'
if '/' not in f and f.endswith('.cpp'):
    return ('src/$grp/' + f).encode()
if '/' not in f and (f.endswith('.h') or f.endswith('.inl')):
    return ('include/fastfields/api/$api/' + f).encode()
return legacy(f)
PY
        ;;
    fastfields-cpu-impl|fastfields-cuda-impl)
        case "$repo" in
            fastfields-cpu-impl)  api=cpu ;;
            fastfields-cuda-impl) api=cuda ;;
        esac
        cat <<PY
if f.startswith('tests/'):
    return ('tests/impl-$api/' + f[len('tests/'):]).encode()
if f.endswith('.h') or f.endswith('.inl'):
    return ('include/fastfields/impl/$api/' + f).encode()
return legacy(f)
PY
        ;;
    fastfields-kernels)
        # vector/test.cpp is a hand-run scratch program, not a header; it is the
        # only non-header source in this repo and it belongs under tests/.
        cat <<'PY'
if f == 'vector/test.cpp':
    return b'tests/kernels/vector/test.cpp'
if f.endswith('.h') or f.endswith('.inl'):
    return ('include/fastfields/impl/kernels/' + f).encode()
return legacy(f)
PY
        ;;
    esac
}

say "stage 2: path rewrite (one filter-repo pass per repo, --refs main teeny)"
RW="$OUT/.rewritten"
rm -rf -- "$RW"; mkdir -p -- "$RW"
for r in $REPOS; do
    printf '  %s\n' "$r" >&2
    rm -rf -- "$RW/$r"
    git clone --quiet --no-local --bare "$SRC/$r" "$RW/$r" >/dev/null
    # Keep only the two refs we rewrite, so no stray branch/tag carries an
    # un-rewritten (or differently-rewritten) copy of the same commits.
    git -C "$RW/$r" for-each-ref --format='%(refname)' \
        | grep -vE '^refs/heads/(main|teeny)$' \
        | while read -r ref; do git -C "$RW/$r" update-ref -d "$ref"; done
    rules_for "$r" > "$RW/$r.rules.py"
    ( cd "$RW/$r" && git filter-repo --force --refs main teeny \
        --filename-callback "$(cat "$RW/$r.rules.py")" >/dev/null )
done

# Record the pre/post merge-base of every repo: proof that the single-pass
# rewrite kept main and teeny sharing ancestry. An empty post-merge-base here
# means the run is poisoned -- stop and re-run from pristine clones.
say "stage 2 check: main/teeny merge-base survived the rewrite"
mb_fail=0
for r in $REPOS; do
    pre="$(git -C "$SRC/$r"  merge-base origin/main origin/teeny 2>/dev/null || true)"
    post="$(git -C "$RW/$r" merge-base main teeny 2>/dev/null || true)"
    if [ -z "$post" ]; then
        printf '  FAIL %-22s pre=%s post=<none>\n' "$r" "${pre:0:8}" >&2; mb_fail=1
    else
        printf '  ok   %-22s pre=%s post=%s\n' "$r" "${pre:0:8}" "${post:0:8}" >&2
    fi
done
[ "$mb_fail" -eq 0 ] || die "a repo lost its main/teeny merge-base -- rule drift; re-run from pristine clones"

# ---------------------------------------------------------------------------
# 3. merge the six rewritten mains
# ---------------------------------------------------------------------------
say "stage 3: merge the six rewritten mains"
rm -rf -- "$OUT/repo"
git init --quiet -b main "$OUT/repo"
cd "$OUT/repo"
git config user.name  "fastfields consolidation"
git config user.email "noreply@fastfields.invalid"

for r in $REPOS; do
    git remote add "$r" "$RW/$r"
    git fetch --quiet "$r" 'refs/heads/main:refs/heads/src/'"$r"'/main' \
                           'refs/heads/teeny:refs/heads/src/'"$r"'/teeny'
done

first=1
for r in $REPOS; do
    if [ "$first" -eq 1 ]; then
        git checkout --quiet -B main "src/$r/main"; first=0
    else
        git merge --quiet --allow-unrelated-histories --no-edit \
            -m "merge $r into the consolidated tree" "src/$r/main" \
            || die "merge of $r conflicted -- the path rule sets are not disjoint"
    fi
done

# ---------------------------------------------------------------------------
# 4. content: core/ dedupe, include rewrite, build system
# ---------------------------------------------------------------------------
say "stage 4: core/ dedupe"
python3 - <<'PYEOF'
import os, re, subprocess, sys

INC  = 'include/fastfields'
CORE = os.path.join(INC, 'core')
os.makedirs(CORE, exist_ok=True)

def git(*a):
    subprocess.run(['git'] + list(a), check=True)

def read(p):
    with open(p, encoding='utf-8') as fh: return fh.read()

def write(p, s):
    with open(p, 'w', encoding='utf-8') as fh: fh.write(s)

# --- dlpack.h ------------------------------------------------------------
# Three byte-identical vendored copies. Collapse to one, and keep the UPSTREAM
# include guard DLPACK_DLPACK_H_ exactly as-is -- no #pragma once. The matching
# guard is what lets a consumer who also has a system dlpack.h include both
# harmlessly; #pragma once keys on the file identity instead and would defeat
# that.
copies = [f'{INC}/api/dlpack.h', f'{INC}/api/cpu/dlpack.h', f'{INC}/api/cuda/dlpack.h']
have = [p for p in copies if os.path.exists(p)]
assert have, 'no dlpack.h found'
import hashlib
digests = {hashlib.md5(open(p,'rb').read()).hexdigest() for p in have}
assert len(digests) == 1, f'dlpack copies diverged: {digests}'
git('mv', have[0], f'{CORE}/dlpack.h')
for p in have[1:]:
    git('rm', '-q', p)
assert 'DLPACK_DLPACK_H_' in read(f'{CORE}/dlpack.h')
assert '#pragma once' not in read(f'{CORE}/dlpack.h')

# --- defines.h -----------------------------------------------------------
# Two colliding defines.h: kernels' namespace macros and the hub's
# FF_CPU/FF_CUDA/FF_WITH_CUDA. The three overlapping macros (FF,
# FF_NAMESPACE_BEGIN, FF_NAMESPACE_END) are token-identical, so the merge is
# mechanical; the rest are disjoint. Merging them into ONE file is also what
# defuses the cuda_switch.h hazard: cuda_switch.h used to end in a *bare*
# `#include "defines.h"` that resolved to whichever defines.h was nearest, and
# once both are reachable via -I that choice becomes silent and wrong
# (FF_DEVICE undefined, FF_CPU/FF_CUDA leaking into kernel namespaces). With a
# single merged file and a fully-qualified include there is nothing to pick.
kern = f'{INC}/impl/kernels/defines.h'
hub  = f'{INC}/api/defines.h'
assert os.path.exists(kern) and os.path.exists(hub)
write(f'{CORE}/defines.h', '''#pragma once
#ifndef FF_DEFINES
#define FF_DEFINES

// Merged from fastfields-kernels/defines.h and fastfields-lib/defines.h during
// the six-repo consolidation. The first three macros were token-identical in
// both; the namespace-device pair came from kernels and the FF_CPU/FF_CUDA pair
// from the hub. One file, one guard -- so a bare `#include "defines.h"` can no
// longer resolve to a different header than the author meant.

#define FF                          ff
#define FF_NAMESPACE_BEGIN(NAME)    namespace NAME {
#define FF_NAMESPACE_END(NAME)      }
#define FF_NAMESPACE_BEGIN_DEVICE   FF_NAMESPACE_BEGIN(FF_DEVICE)
#define FF_NAMESPACE_END_DEVICE     FF_NAMESPACE_END(FF_DEVICE)

#define FF_CPU cpu
#ifndef FF_WITH_CUDA
#  define FF_CUDA notimplemented
#else
#  define FF_CUDA cuda
#endif

#endif // FF_DEFINES
''')
git('rm', '-q', kern, hub)
git('add', f'{CORE}/defines.h')

# --- cuda_switch.h -------------------------------------------------------
cs = f'{INC}/impl/kernels/cuda_switch.h'
assert os.path.exists(cs)
git('mv', cs, f'{CORE}/cuda_switch.h')
s = read(f'{CORE}/cuda_switch.h')
# The bare include is the hazard described above: make it fully qualified.
assert '#include "defines.h"' in s
s = s.replace('#include "defines.h"', '#include "fastfields/core/defines.h"')
s = '#pragma once\n' + s
write(f'{CORE}/cuda_switch.h', s)
git('add', f'{CORE}/cuda_switch.h')

# --- autocast.h ----------------------------------------------------------
# NOT a duplicate pair. The CUDA copy routes the staging buffers through
# cudaMallocHost/cudaFreeHost so the H2D copy can be async; the CPU copy uses
# new[]/delete[]. That is a real functional difference, so this is a refactor
# to ONE header with the host allocator injected -- not two files under tidier
# names, which would preserve the duplication.
cpu_ac  = f'{INC}/api/cpu/autocast.h'
cuda_ac = f'{INC}/api/cuda/autocast.h'
assert os.path.exists(cpu_ac) and os.path.exists(cuda_ac)
src = read(cuda_ac)          # the CUDA copy is the superset (it has the hooks)

src = src.replace('#ifndef FF_CUDA_AUTOCAST\n#define FF_CUDA_AUTOCAST\n',
                  '#pragma once\n#ifndef FF_AUTOCAST\n#define FF_AUTOCAST\n', 1)
src = src.replace('#endif // FF_CUDA_AUTOCAST', '#endif // FF_AUTOCAST')

hook_old = '''template <class ElemType>
inline ElemType * hostNew(size_t numel)
{
    void * out;
    if (cudaMallocHost(&out, numel * sizeof(ElemType)))
        throw std::runtime_error("cudaMallocHost failed");
    return static_cast<ElemType*>(out);
}

template <class ElemType>
inline void hostDelete(ElemType * ptr)
{
    if (cudaFreeHost(const_cast<void*>(static_cast<const void*>(ptr))))
        throw std::runtime_error("cudaFreeHost failed");
}
'''
hook_new = '''// ------------------------------------------------------------------ host
// The staging buffers below are *host* memory, and the two backends want them
// allocated differently: CUDA wants them page-locked (cudaMallocHost) so the
// following H2D copy can be async, the CPU backend just wants new[]. That is
// the only difference there has ever been between the two autocast headers, so
// it is injected here rather than by shipping the header twice.
//
// FF_AUTOCAST_PINNED_HOST may be set explicitly; by default it follows the
// compiler, because nvcc compiles the CUDA library and nothing else.
#ifndef FF_AUTOCAST_PINNED_HOST
#  ifdef __CUDACC__
#    define FF_AUTOCAST_PINNED_HOST 1
#  else
#    define FF_AUTOCAST_PINNED_HOST 0
#  endif
#endif

#if FF_AUTOCAST_PINNED_HOST

template <class ElemType>
inline ElemType * hostNew(size_t numel)
{
    void * out;
    if (cudaMallocHost(&out, numel * sizeof(ElemType)))
        throw std::runtime_error("cudaMallocHost failed");
    return static_cast<ElemType*>(out);
}

template <class ElemType>
inline void hostDelete(ElemType * ptr)
{
    if (cudaFreeHost(const_cast<void*>(static_cast<const void*>(ptr))))
        throw std::runtime_error("cudaFreeHost failed");
}

#else

template <class ElemType>
inline ElemType * hostNew(size_t numel)
{
    return new ElemType[numel];
}

template <class ElemType>
inline void hostDelete(ElemType * ptr)
{
    delete[] const_cast<typename RemoveConst<ElemType>::Type *>(ptr);
}

#endif // FF_AUTOCAST_PINNED_HOST
'''
assert hook_old in src, 'autocast.h CUDA allocator hook not found verbatim'
src = src.replace(hook_old, hook_new, 1)
write(f'{CORE}/autocast.h', src)
git('rm', '-q', cpu_ac, cuda_ac)
git('add', f'{CORE}/autocast.h')

print('core/ ->', sorted(os.listdir(CORE)))
PYEOF

git add -A
git commit --quiet -m "core: deduplicate dlpack/defines/cuda_switch and inject autocast's host allocator

dlpack.h had three byte-identical vendored copies (lib, cpu-lib, cuda-lib);
they collapse to one, keeping the upstream DLPACK_DLPACK_H_ guard so a
consumer that also has a system dlpack.h can include both.

defines.h had two colliding definitions. Merging them is what makes the
cuda_switch.h hazard impossible by construction: it ended in a bare
#include \"defines.h\" that silently resolved to whichever copy was nearest,
which with both on the -I path would leave FF_DEVICE undefined and leak
FF_CPU/FF_CUDA into the kernel namespaces -- a miscompile, not an error. The
include is now fully qualified as well.

autocast.h was NOT a duplicate: the CUDA copy staged through pinned host
memory (cudaMallocHost/cudaFreeHost) for async H2D copies, the CPU copy used
new[]/delete[]. One header now, with the host allocator injected behind
FF_AUTOCAST_PINNED_HOST."

say "stage 5: include rewrite"
python3 - <<'PYEOF'
import os, re, sys

ROOT = '.'
INC  = 'include/fastfields'

# Longest-prefix-first. Order is load-bearing twice over:
#   * the cuda_switch.h exact rules must fire before the "impl/kernels/ and
#     "kernels/ prefix rules, or cuda_switch.h would be sent to
#     impl/kernels/ where it no longer lives;
#   * "impl/kernels/ must fire before "impl/ for the same reason.
GLOBAL = [
    ('"impl/kernels/cuda_switch.h"', '"fastfields/core/cuda_switch.h"'),
    ('"kernels/cuda_switch.h"',      '"fastfields/core/cuda_switch.h"'),
    ('"../../cuda_switch.h"',        '"fastfields/core/cuda_switch.h"'),
    ('"../cuda_switch.h"',           '"fastfields/core/cuda_switch.h"'),
    ('"cuda_switch.h"',              '"fastfields/core/cuda_switch.h"'),
    ('"impl/kernels/',               '"fastfields/impl/kernels/'),
    ('"kernels/',                    '"fastfields/impl/kernels/'),
    ('"dlpack.h"',                   '"fastfields/core/dlpack.h"'),
    ('"defines.h"',                  '"fastfields/core/defines.h"'),
    ('"autocast.h"',                 '"fastfields/core/autocast.h"'),
]

# Per-area rules, applied after the global ones. `impl/` resolves to a
# different directory depending on which library is including it -- that is the
# one rule the old symlink layout encoded implicitly and the new one must state.
AREA = {
    'include/fastfields/api/cpu':  [('"impl/', '"fastfields/impl/cpu/')],
    'src/lib-cpu':                 [('"impl/', '"fastfields/impl/cpu/')],
    'tests/lib-cpu':               [('"impl/', '"fastfields/impl/cpu/')],
    'include/fastfields/api/cuda': [('"impl/', '"fastfields/impl/cuda/')],
    'src/lib-cuda':                [('"impl/', '"fastfields/impl/cuda/')],
    'src/lib':                     [('"cpu/', '"fastfields/api/cpu/'),
                                    ('"cuda/', '"fastfields/api/cuda/')],
    'tests/lib':                   [('"cpu/', '"fastfields/api/cpu/'),
                                    ('"cuda/', '"fastfields/api/cuda/')],
}

# Sources moved out of their headers' directory lose same-directory adjacency,
# so every bare include of a sibling header has to be qualified. Headers that
# stayed adjacent are deliberately left alone -- the smaller the diff, the
# stronger the "nothing changed" argument.
DETACHED = {
    'src/lib':       f'{INC}/api',
    'tests/lib':     f'{INC}/api',
    'src/lib-cpu':   f'{INC}/api/cpu',
    'tests/lib-cpu': f'{INC}/api/cpu',
    'src/lib-cuda':  f'{INC}/api/cuda',
}

EXTRA = {
    # the cuda-impl compile probe moved from cuda-impl/tests/ to tests/impl-cuda/
    'tests/impl-cuda': [('"../', '"fastfields/impl/cuda/')],
    # kernels' scratch vector program moved from vector/ to tests/kernels/vector/
    'tests/kernels':   [('"vector.h"', '"fastfields/impl/kernels/vector/vector.h"'),
                        ('"stream.h"', '"fastfields/impl/kernels/vector/stream.h"')],
}

def sources():
    for base, _dirs, files in os.walk(ROOT):
        if '.git' in base.split(os.sep):
            continue
        for fn in files:
            if fn.endswith(('.h', '.inl', '.cpp', '.cu')):
                yield os.path.normpath(os.path.join(base, fn))

def area_of(path):
    for a in sorted(list(AREA) + list(DETACHED) + list(EXTRA), key=len, reverse=True):
        if path.startswith(a + os.sep):
            return a
    return None

changed = files_changed = 0
for path in sources():
    with open(path, encoding='utf-8') as fh:
        text = orig = fh.read()
    rules = list(GLOBAL)
    a = area_of(path)
    if a:
        rules += AREA.get(a, []) + EXTRA.get(a, [])
        hdrdir = DETACHED.get(a)
        if hdrdir and os.path.isdir(hdrdir):
            qual = hdrdir[len('include/'):]
            for h in sorted(os.listdir(hdrdir)):
                if h.endswith('.h'):
                    rules.append((f'"{h}"', f'"{qual}/{h}"'))
                    rules.append((f'"../{h}"', f'"{qual}/{h}"'))
    rules.sort(key=lambda kv: -len(kv[0]))

    out, i = [], 0
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith('#include'):
            for old, new in rules:
                if old in line and 'fastfields/' not in line:
                    line = line.replace(old, new, 1)
                    i += 1
                    break
        out.append(line)
    text = ''.join(out)
    if text != orig:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(text)
        files_changed += 1
        changed += i

print(f'rewrote {changed} include lines across {files_changed} files')

# Guard-collision check: once every header is reachable from one -I, two
# headers sharing an include guard would silently suppress one another.
guards = {}
dupes = []
for path in sources():
    if not path.startswith('include' + os.sep):
        continue
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            m = re.match(r'\s*#\s*ifndef\s+(\w+)', line)
            if m:
                guards.setdefault(m.group(1), []).append(path)
                break
for g, ps in sorted(guards.items()):
    if len(ps) > 1:
        dupes.append((g, ps))
if dupes:
    print('GUARD COLLISIONS:')
    for g, ps in dupes:
        print(' ', g, ps)
    sys.exit(1)
print(f'no include-guard collisions across {len(guards)} guarded headers')
PYEOF

git add -A
git commit --quiet -m "includes: rewrite quoted includes for the consolidated layout

Longest-prefix-first substitutions; \"impl/kernels/ must be substituted before
\"impl/, and the cuda_switch.h rules before both, because cuda_switch.h no
longer lives under kernels/. \"impl/ resolves to impl/cpu/ or impl/cuda/
depending on which library is doing the including -- the one thing the old
symlink layout encoded implicitly.

Includes that kept same-directory adjacency are deliberately untouched."

say "stage 6: build system"

# The three source Makefiles were path-moved in stage 2 (cpu-lib's to
# src/lib-cpu/, cuda-lib's to src/lib-cuda/, the hub's stayed at the root), so
# rewriting them in place keeps their history. make/common.mk and
# src/lib/Makefile are new.
mkdir -p make src/lib

cat > make/common.mk <<'MKEOF'
# Shared makefile fragment for the fastfields build.
#
# Included by the root Makefile and by each of src/lib/, src/lib-cpu/ and
# src/lib-cuda/. It owns everything the three groups must agree on: where the
# build output goes, how the platform is detected, and which diagnostic flags
# the detected compiler actually understands.
#
# A group sets GROUP before including this file:
#     GROUP := lib-cpu
#     include ../../make/common.mk

define verb
	@ echo "_____________________________________________________________"
	@ echo ""
	@ echo "        " $(1)
	@ echo "_____________________________________________________________"
	@ echo ""
endef

COMMON_MK := $(lastword $(MAKEFILE_LIST))
ROOTDIR   := $(patsubst %/,%,$(dir $(abspath $(COMMON_MK))))/..
ROOTDIR   := $(abspath $(ROOTDIR))

COPY      ?= cp -f
DEL       ?= rm -f
MOVE      ?= mv -f
MKDIR     ?= mkdir -p
UNAME     ?= uname
GET_ARCH  ?= $(UNAME) -m
NVCC      ?= nvcc

# Output paths. build/ and build/lib/ are load-bearing: fastfields-dlpack's
# setup.py hardcodes both, so libfastfields.so must land in build/ and the
# backend libraries in build/lib/ regardless of which subdirectory's Makefile
# produced them. Objects go under build/obj/<group>/ so the three groups'
# same-named objects (distance.o, posdef.o, ...) cannot collide now that they
# share one build tree.
BUILDDIR  ?= $(ROOTDIR)/build
LIBDIR    ?= $(BUILDDIR)/lib
OBJDIR    ?= $(BUILDDIR)/obj/$(GROUP)
TESTDIR   ?= $(BUILDDIR)/test/$(GROUP)

INCLUDES  += -I$(ROOTDIR)/include

MOSUF      = o
SOSUF      = so
SONAME     = soname
OMPFLAG    = -fopenmp
USE_OPENMP ?= 0

# Position-independent code + shared-library soname flags. Both are POSIX-only:
# on Windows the PE/COFF linker has no soname and code is position-independent
# by default, so they are cleared in the Windows block below and referenced via
# these variables (never hard-coded) in the link rules.
PICFLAG       = -fPIC
SONAME_PREFIX = -Wl,-$(SONAME),
SONAME_FLAG   = $(if $(SONAME_PREFIX),$(SONAME_PREFIX)$(@F))
RPATH         = -Wl,-rpath,'$$ORIGIN'/../lib

########################################################################
#	Compiler detection
########################################################################

# The diagnostic flags are spelled differently by the two compilers, and each
# rejects the other's outright -- so `make CXX=g++` used to fail on the *flags*
# before it reached a single line of source, and only worked via a command-line
# CXXFLAGS= override that dropped them. Detect instead.
CXX_VERSION := $(shell $(CXX) --version 2>/dev/null | head -1)
ifneq (,$(findstring clang,$(CXX_VERSION)))
  CXX_IS_CLANG := 1
endif

ifdef CXX_IS_CLANG
  DIAGFLAGS ?= -ferror-limit=1 -ftemplate-backtrace-limit=0
else
  DIAGFLAGS ?= -fmax-errors=1 -ftemplate-backtrace-limit=0
endif

########################################################################
#	Platform-specific settings
########################################################################

# Native Windows (cmd/pwsh) has no `uname`; GNU make sets OS=Windows_NT there.
# Under a Unix-like shell on Windows (git bash / MSYS, which is what CI uses),
# `uname` returns MINGW*/MSYS and the block further down matches instead.
ifeq ($(OS),Windows_NT)
  PLATFORM   = Windows
endif
ifndef PLATFORM
  PLATFORM   = $(shell $(UNAME))
  ifeq (Darwin,$(PLATFORM))
    ifeq (arm64,$(shell $(GET_ARCH))) # Check for Apple Silicon
      PLATFORM = arm64
    endif
  endif
endif

##### macOS #####
ifeq (Darwin,$(PLATFORM))
  OMPFLAG    = -fopenmp=libiomp5
  SOSUF      = dylib
  SONAME     = install_name
  RPATH      = -Wl,-rpath,@loader_path/../lib
endif
ifeq (arm64,$(PLATFORM))
  OMPFLAG    = -fopenmp=libiomp5
  SOSUF      = dylib
  SONAME     = install_name
  RPATH      = -Wl,-rpath,@loader_path/../lib
endif

##### Windows (native OS=Windows_NT, or a Unix-like shell: MINGW*/MSYS) #####
# Built with clang++ (as CI does): objects stay .o, shared libs are .dll, and
# there is no soname / -fPIC / rpath (the PE/COFF linker rejects -Wl,-soname and
# code is position-independent by default). A Unix-like shell (git bash / MSYS)
# is required so the recipe commands (cp/rm/mkdir/uname) resolve.
#
# This block used to be triplicated across the three repos and had already
# drifted -- cuda-lib's copy knew nothing about IS_WINDOWS or PICFLAG and set
# MOSUF=obj where the other two set o. It is stated once here, in cpu-lib's
# (non-stale) form. Nothing implements Windows support; this only keeps the
# door open.
IS_WINDOWS =
ifeq (Windows,$(PLATFORM))
  IS_WINDOWS = 1
endif
ifeq (MINGW32,$(word 1,$(subst _, ,$(PLATFORM))))
  IS_WINDOWS = 1
endif
ifeq (MINGW64,$(word 1,$(subst _, ,$(PLATFORM))))
  IS_WINDOWS = 1
endif
ifeq (MSYS,$(word 1,$(subst _, ,$(PLATFORM))))
  IS_WINDOWS = 1
endif
ifdef IS_WINDOWS
  SOSUF         = dll
  MOSUF         = o
  PICFLAG       =
  SONAME_PREFIX =
  RPATH         =
endif

########################################################################
#	Output directories
########################################################################

$(BUILDDIR) $(LIBDIR) $(OBJDIR) $(TESTDIR):
	$(MKDIR) $@
MKEOF

cat > Makefile <<'MKEOF'
# fastfields -- top-level delegation.
#
#   make cpu     build build/lib/libfastfields-cpu.so   (never needs nvcc)
#   make cuda    build build/lib/libfastfields-cuda.so  (needs nvcc)
#   make lib     build build/libfastfields.so           (implies cpu)
#   make all     the default: cpu + lib
#   make test    run every test group that does not need a GPU toolchain
#   make clean   remove build/

GROUP := root
include make/common.mk

.PHONY: all cpu cuda lib test test-lib test-lib-cpu test-impl-cuda test-kernels \
        clean install

# `all` produces both shared objects -- build/lib/libfastfields-cpu.so and
# build/libfastfields.so -- exactly as the hub repo's `all` did. CUDA stays
# opt-in (USE_CUDA=1, or `make cuda`) because CI has no GPU and the CPU path is
# the tested source of truth.
all: lib

cpu:
	$(MAKE) -C src/lib-cpu all

cuda:
	$(MAKE) -C src/lib-cuda all

lib: cpu
	$(MAKE) -C src/lib all

install:
	$(MAKE) -C src/lib install

########################################################################
#	Tests
########################################################################

test: test-lib-cpu test-lib

test-lib-cpu:
	$(MAKE) -C src/lib-cpu test

test-lib:
	$(MAKE) -C src/lib test

# Needs nvcc; not part of `make test`.
test-impl-cuda:
	$(MAKE) -C src/lib-cuda test-probe

# A hand-run scratch program for the vector/ abstractions; compiled (not run)
# as a check that the headers still stand alone.
test-kernels: | $(TESTDIR)
	$(CXX) $(CXXFLAGS) $(DIAGFLAGS) $(INCLUDES) -std=c++11 \
	  -o $(BUILDDIR)/test/kernels_vector tests/kernels/vector/test.cpp

clean:
	$(DEL) -r $(BUILDDIR)
MKEOF

cat > src/lib-cpu/Makefile <<'MKEOF'
# libfastfields-cpu.so -- the dtype-dispatch layer.
#
# Never invokes nvcc: `make cpu` must work on a machine with no CUDA toolchain
# at all, which is what keeps a CPU-only build of the whole project possible.

GROUP := lib-cpu
include ../../make/common.mk

CXXFLAGS += -std=c++11 -O3 $(DIAGFLAGS)
TESTFLAGS += $(DIAGFLAGS)

# Boundary-condition compile policy (see impl/kernels/bounds.h). The CPU backend
# compiles the full static matrix comfortably, so every boundary condition keeps
# its own (fastest) instantiation by default. Set e.g.
#   make BOUNDFLAGS="-DFF_STATIC_BOUNDS=0 -DFF_STATIC_BOUND_DCT2=1"
# to route the rarely used conditions through the shared `bound::type::Dynamic`
# runtime implementation instead -- smaller library, faster build, identical
# results. Deliberately a separate variable from CXXFLAGS so that overriding
# CXXFLAGS on the command line does not silently drop the policy.
BOUNDFLAGS  ?=

# Interpolation-order compile policy (see impl/kernels/spline.h) -- the same
# idea one axis further out, and for the same reason: `pushpull` templates on
# the spline order as well as the boundary condition, so all eight orders static
# means an 8x8 matrix of instantiations. The CPU backend compiles that
# comfortably, so every order keeps its own (fastest) instantiation by default.
# Set e.g.
#   make SPLINEFLAGS="-DFF_STATIC_SPLINES=0 -DFF_STATIC_SPLINE_LINEAR=1"
# to route the remaining orders through the shared `spline::type::Dynamic`
# runtime implementation instead -- identical results, smaller/faster build.
# Kept out of CXXFLAGS for the same reason as BOUNDFLAGS.
SPLINEFLAGS ?=

# `test` target only: a sparser *default* BOUNDFLAGS/SPLINEFLAGS than the
# library's (empty = fully static). pushpull's own dispatch used to carry a
# second, hand-duplicated switch behind `-DFF_TEST_SPARSE` purely to keep a
# bare `make test` fast (a *covering* subset of the order x bound matrix --
# all bounds for Linear/Cubic, DCT2 only for the rest -- that threw on
# anything else). That is now redundant with, and strictly weaker than,
# routing through `bound::type::Dynamic`/`spline::type::Dynamic`: it is the
# same compile-cost win (fewer static instantiations), but every combination
# stays fully *functional* via the shared Dynamic instantiation rather than
# throwing -- so pushpull.cpp no longer branches on FF_TEST_SPARSE at all;
# there is exactly one PP_ORDER/PP_BOUND, and "sparse vs. full" is purely this
# target-specific default. It matches the `cuda-default` CI matrix leg, so
# `make test` (no override) exercises the same mixed policy CUDA ships.
# An explicit `make test BOUNDFLAGS=... SPLINEFLAGS=...` (as the CI matrix's
# three legs do) always overrides this default -- GNU Make command-line
# variables outrank both plain and target-specific `?=`.
# NB: plain `=`, not `?=` -- BOUNDFLAGS/SPLINEFLAGS are already `?=`-defaulted
# (to empty) above, which "sets" them at parse time, so a target-specific
# `?=` here would see them as already-set and never fire. Plain `=` still
# yields to an explicit command-line override (`make test BOUNDFLAGS=...`):
# command-line-origin variables outrank *any* in-makefile assignment, target-
# specific or not, unless the makefile uses `override` (which this does not).
test: BOUNDFLAGS  = -DFF_STATIC_BOUNDS=0 -DFF_STATIC_BOUND_DCT2=1 -DFF_STATIC_BOUND_DST2=1
test: SPLINEFLAGS = -DFF_STATIC_SPLINES=0 -DFF_STATIC_SPLINE_NEAREST=1 \
                    -DFF_STATIC_SPLINE_LINEAR=1 -DFF_STATIC_SPLINE_QUADRATIC=1 \
                    -DFF_STATIC_SPLINE_CUBIC=1

MODULES = \
	distance \
	posdef \
	resize \
	restrict \
	splinc \
	pushpull \
	pushpull_backward \
	reg_field \
	reg_flow \
	solve_field

OBJECTS = $(addprefix $(OBJDIR)/,$(addsuffix .$(MOSUF),$(MODULES)))
TARGET  = $(LIBDIR)/libfastfields-cpu.$(SOSUF)

.PHONY: all install test clean clean-obj clean-lib

all: verb.build.lib $(TARGET) verb.build.lib.done

# Kept as a target for compatibility: the library is already built into
# build/lib/, so there is nothing left to install.
install: all

$(TARGET): $(OBJECTS) | $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared $(PICFLAG) $(SONAME_FLAG) -o $@ $^

# -MMD -MP emit header dependency files (*.d) so that editing a kernel/impl
# header (this is a header-only codebase) rebuilds the affected library object
# instead of leaving a stale binary.
$(OBJDIR)/%.$(MOSUF): %.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(BOUNDFLAGS) $(SPLINEFLAGS) $(INCLUDES) $(PICFLAG) -MMD -MP -c -o $@ $<

########################################################################
#	Tests
########################################################################
# Each tests/lib-cpu/test_<name>.cpp is linked against the module objects and
# run. Usage: `make test CXX=clang++` (or CXX=g++).
#
# Every module .cpp is compiled to an object ONCE (in build/obj/lib-cpu/test/)
# and linked into every test binary, instead of being recompiled together with
# each test. This turns ~(#tests x #modules) module compiles into just
# #modules, and the single-source `-c` compiles let ccache cache each object.
#
# Test objects build with -DFF_TEST_SPARSE: the heavy order x bound modules
# (pushpull, resize, restrict, splinc) then instantiate only a covering subset
# of the matrix, cutting test-compile time. The library build (`make all`) omits
# the flag and compiles the full matrix, so it is also the compile gate. The
# test objects live in their own dir so they never collide with the library
# objects built without FF_TEST_SPARSE.

TESTSRCDIR = $(ROOTDIR)/tests/lib-cpu
TESTOBJDIR = $(OBJDIR)/test
TESTSRC    = $(wildcard $(TESTSRCDIR)/test_*.cpp)
TESTBIN    = $(patsubst $(TESTSRCDIR)/%.cpp,$(TESTDIR)/%,$(TESTSRC))

TESTMODOBJ = $(addprefix $(TESTOBJDIR)/,$(addsuffix .$(MOSUF),$(MODULES)))
TESTDRVOBJ = $(patsubst $(TESTSRCDIR)/%.cpp,$(TESTOBJDIR)/%.$(MOSUF),$(TESTSRC))

# These objects are built via pattern rules, so make would treat them as
# intermediate and delete them after linking -- recompiling everything on the
# next `make test`. Mark them SECONDARY so they persist (real incremental
# rebuilds + warm ccache).
.SECONDARY: $(TESTMODOBJ) $(TESTDRVOBJ)

# -MMD -MP emit header dependency files (*.d) so header edits trigger rebuilds.
TESTCPPFLAGS = $(CXXFLAGS) $(BOUNDFLAGS) $(SPLINEFLAGS) -DFF_TEST_SPARSE $(INCLUDES) -MMD -MP

$(TESTOBJDIR):
	$(MKDIR) $(TESTOBJDIR)

$(TESTOBJDIR)/%.$(MOSUF): %.cpp | $(TESTOBJDIR)
	$(CXX) $(TESTCPPFLAGS) -c -o $@ $<

$(TESTOBJDIR)/test_%.$(MOSUF): $(TESTSRCDIR)/test_%.cpp | $(TESTOBJDIR)
	$(CXX) $(TESTCPPFLAGS) -c -o $@ $<

# Link each test binary from its driver object + ALL shared module objects.
# Linking the full module set means cross-module symbol references (e.g.
# test_restrict -> resample in resize.cpp) resolve automatically.
$(TESTDIR)/test_%: $(TESTOBJDIR)/test_%.$(MOSUF) $(TESTMODOBJ) | $(TESTDIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

test: $(TESTBIN)
	@ status=0; for t in $(TESTBIN); do \
	    echo "running $$t"; $$t || status=1; \
	done; exit $$status

-include $(wildcard $(OBJDIR)/*.d)
-include $(wildcard $(TESTOBJDIR)/*.d)

clean: clean-obj clean-lib
clean-obj:
	$(DEL) -r $(OBJDIR)
clean-lib:
	$(DEL) $(TARGET)

verb.build.lib:
	$(call verb, "Building CPU library...")
verb.build.lib.done:
	$(call verb, "Building CPU library: done.")
MKEOF

cat > src/lib-cuda/Makefile <<'MKEOF'
# libfastfields-cuda.so -- the CUDA dtype-dispatch layer.
#
# Every source here is a .cpp compiled by nvcc with `-x cu`; there are no .cu
# files. Nothing in `make cpu` or `make lib` reaches this Makefile unless
# USE_CUDA=1 asks for it.

GROUP := lib-cuda
include ../../make/common.mk

# nvcc flags: -std/-O3 pass straight through. The host compiler's diagnostic
# flags (-ferror-limit / -fmax-errors / -ftemplate-backtrace-limit) are NOT
# understood by nvcc, so DIAGFLAGS is deliberately not added here. nvcc 12
# builds these CUDA headers under -std=c++14.
CXXFLAGS += -std=c++14 -O3
TESTFLAGS += $(DIAGFLAGS)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Boundary-condition compile policy  (see impl/kernels/bounds.h)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Every boundary condition used to be a compile-time template parameter, so
# each regulariser kernel was instantiated 8 times per (ndim, dtype, offset_t)
# combination. On the CUDA side that pushes `ptxas` past 16 GB of RAM -- it gets
# OOM-killed on a standard 16 GB CI runner.
#
# `bound::type::Dynamic` provides the same operator with the condition read at
# run time: one instantiation shared by every condition that does not keep a
# static fast path. The default below keeps dedicated instantiations for two
# conditions and routes the other six through the Dynamic implementation:
#   DCT2 -- Neumann, the library-wide default boundary condition, and the
#           symmetric case (no sign flip at the boundary);
#   DST2 -- Dirichlet, the antisymmetric case, so the sign-flipping code path
#           (`_sign::periodic2` + the sign-aware `cget`/`add` selected by
#           FF_ISO_SIGN) also keeps a real compile-time instantiation rather
#           than only ever being reached through the shared Dynamic one.
#
# Whoever compiles the library can trade compile cost against per-voxel speed:
#   make BOUNDFLAGS="-DFF_STATIC_BOUNDS=1"                     # all 8 static
#   make BOUNDFLAGS="-DFF_STATIC_BOUNDS=0"                     # all dynamic
#   make BOUNDFLAGS="-DFF_STATIC_BOUNDS=0 -DFF_STATIC_BOUND_DFT=1 \
#                    -DFF_STATIC_BOUND_DCT2=1"                 # pick your own
# Results are identical either way; only code size, compile cost and speed move.
# Kept out of CXXFLAGS on purpose so that overriding CXXFLAGS on the command
# line (as CI does, to force -O1) does not silently drop the policy.
BOUNDFLAGS  ?= -DFF_STATIC_BOUNDS=0 -DFF_STATIC_BOUND_DCT2=1 -DFF_STATIC_BOUND_DST2=1

# Interpolation-order compile policy (see impl/kernels/spline.h) -- the same
# idea one axis further out. `pushpull` templates on the spline order as well as
# the boundary condition (and ndim, dtype, offset_t), so all eight orders static
# would multiply the already-Dynamic-bound matrix by up to eight again.
#
# `spline::type::Dynamic` provides the same operator with the order read at run
# time. The default below keeps dedicated instantiations for the four lowest
# (and by far the most commonly used) orders -- Nearest/Linear/Quadratic/Cubic
# -- and routes FourthOrder-SeventhOrder through the Dynamic implementation:
#   make SPLINEFLAGS="-DFF_STATIC_SPLINES=1"                    # all 8 static
#   make SPLINEFLAGS="-DFF_STATIC_SPLINES=0"                    # all dynamic
#   make SPLINEFLAGS="-DFF_STATIC_SPLINES=0 \
#                     -DFF_STATIC_SPLINE_CUBIC=1"               # pick your own
# Results are identical either way; only code size, compile cost and speed move.
# Kept out of CXXFLAGS for the same reason as BOUNDFLAGS.
SPLINEFLAGS ?= -DFF_STATIC_SPLINES=0 \
               -DFF_STATIC_SPLINE_NEAREST=1 -DFF_STATIC_SPLINE_LINEAR=1 \
               -DFF_STATIC_SPLINE_QUADRATIC=1 -DFF_STATIC_SPLINE_CUBIC=1

# The regularisers are split into a core module and an `_rls` module (the
# reweighted-least-squares ops) -- a CUDA-only split that lib-cpu does not need.
# Rationale: measured with the default BOUNDFLAGS, `ptxas` peaks at ~3.8 GB per
# split module but ~6-7 GB for the combined file. Two ~4 GB jobs fit a 16 GB CI
# runner under `make -j2`; two ~7 GB ones do not. The split is a memory/
# parallelism measure, not a correctness one -- the boundary-condition policy
# above is what actually brought this build back from a ~16 GB ptxas OOM.
# CI additionally passes -O1 on the command line, for the same reason.
MODULES = \
	distance \
	reg_field \
	reg_field_rls \
	reg_flow \
	reg_flow_rls \
	pushpull \
	pushpull_backward

OBJECTS = $(addprefix $(OBJDIR)/,$(addsuffix .$(MOSUF),$(MODULES)))
TARGET  = $(LIBDIR)/libfastfields-cuda.$(SOSUF)

.PHONY: all install test-probe clean clean-obj clean-lib

all: verb.build.lib $(TARGET) verb.build.lib.done

install: all

$(TARGET): $(OBJECTS) | $(LIBDIR)
	$(NVCC) $(CXXFLAGS) -shared -Xcompiler -fPIC \
	  -Xlinker -$(SONAME)=libfastfields-cuda.$(SOSUF) -o $@ $^

# Header dependency files, so that editing a kernel/impl header rebuilds the
# affected object. This layer had no dependency tracking at all, which in a
# header-only codebase meant stale objects survived header edits.
#
# These MUST be nvcc's own -MMD/-MF, NOT `-Xcompiler -MMD`. nvcc does not hand
# the host compiler the original .cpp: it hands it a generated
# /tmp/tmpxft_*.cudafe1.cpp, so a host-generated .d names that temporary as the
# prerequisite. The temp file is deleted when nvcc exits, so the NEXT `make`
# reads the stale .d and dies with "No rule to make target
# '/tmp/tmpxft_....cudafe1.cpp'". nvcc's own -MMD resolves dependencies against
# the real source and its headers.
#
# nvcc has no -MP equivalent, so a *deleted* header still breaks the build until
# the stale .d is removed -- the tradeoff for having any dependency tracking
# here at all. lib-cpu, compiled by the host compiler directly, keeps -MP.
$(OBJDIR)/%.$(MOSUF): %.cpp | $(OBJDIR)
	$(NVCC) $(CXXFLAGS) $(BOUNDFLAGS) $(SPLINEFLAGS) $(INCLUDES) \
	  -x cu -Xcompiler -fPIC -MMD -MF $(@:.$(MOSUF)=.d) -c -o $@ $<

# Compile-only probe for the CUDA impl layer: there is no GPU in CI, so the
# mesh launcher is validated by nvcc accepting it, not by running it.
PROBESRC = $(wildcard $(ROOTDIR)/tests/impl-cuda/*.cu)
test-probe: | $(TESTDIR)
	@ for p in $(PROBESRC); do \
	    echo "compiling $$p"; \
	    $(NVCC) $(CXXFLAGS) $(BOUNDFLAGS) $(SPLINEFLAGS) $(INCLUDES) \
	      -c -o $(TESTDIR)/$$(basename $$p .cu).$(MOSUF) $$p || exit 1; \
	done

-include $(wildcard $(OBJDIR)/*.d)

clean: clean-obj clean-lib
clean-obj:
	$(DEL) -r $(OBJDIR)
clean-lib:
	$(DEL) $(TARGET)

verb.build.lib:
	$(call verb, "Building CUDA library...")
verb.build.lib.done:
	$(call verb, "Building CUDA library: done.")
MKEOF

cat > src/lib/Makefile <<'MKEOF'
# libfastfields.so -- the device-dispatch hub.
#
# Links against libfastfields-cpu.so always, and libfastfields-cuda.so when
# USE_CUDA=1.

GROUP := lib
include ../../make/common.mk

CXXFLAGS += -std=c++11 -O3 $(DIAGFLAGS)
TESTFLAGS += $(DIAGFLAGS)

# Build the CUDA backend and link it in (needs nvcc). Default off: the CPU path
# is the tested source of truth and CI has no GPU.
USE_CUDA ?= 0
ifeq ($(USE_CUDA),1)
  CXXFLAGS    += -DFF_WITH_CUDA
  CUDA_LDFLAGS = -L$(LIBDIR) -lfastfields-cuda
  CUDA_DEP     = $(LIBDIR)/libfastfields-cuda.$(SOSUF)
endif

MODULES = \
	distance \
	posdef \
	resize \
	restrict \
	splinc \
	pushpull \
	reg_field \
	reg_flow \
	solve_field

OBJECTS = $(addprefix $(OBJDIR)/,$(addsuffix .$(MOSUF),$(MODULES)))
TARGET  = $(BUILDDIR)/libfastfields.$(SOSUF)
CPU_DEP = $(LIBDIR)/libfastfields-cpu.$(SOSUF)

.PHONY: all install test clean clean-obj clean-lib

all: verb.build.lib $(TARGET) verb.build.lib.done

install: all

$(CPU_DEP):
	$(MAKE) -C ../lib-cpu all

$(LIBDIR)/libfastfields-cuda.$(SOSUF):
	$(MAKE) -C ../lib-cuda all

# Real prerequisites (not just siblings under a recipe-less target) so `make -j`
# cannot run this link step before the backend libraries it links against
# actually exist. ($^ would pull the .so prerequisites into the link line too,
# so the recipe lists $(OBJECTS) explicitly instead.)
$(TARGET): $(OBJECTS) $(CPU_DEP) $(CUDA_DEP) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -shared $(PICFLAG) $(SONAME_FLAG) $(RPATH) \
	  -L$(LIBDIR) -lfastfields-cpu $(CUDA_LDFLAGS) \
	  -o $@ $(OBJECTS)

# -MMD -MP emit header dependency files (*.d). This layer had none, which in a
# header-only codebase meant editing a kernel header left a stale object.
$(OBJDIR)/%.$(MOSUF): %.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PICFLAG) -MMD -MP -c -o $@ $<

########################################################################
#	Tests
########################################################################
# Standalone, header-only tests: each tests/lib/test_<x>.cpp includes only the
# hub headers it exercises (checks.h, splinc.h, ...), so it compiles and runs on
# its own without linking libfastfields.so. Op correctness stays gated by
# lib-cpu's suite -- these cover the argument validation that lives in this
# layer and nowhere below it.

TESTSRCDIR = $(ROOTDIR)/tests/lib
TESTSRC    = $(wildcard $(TESTSRCDIR)/test_*.cpp)
TESTBINS   = $(patsubst $(TESTSRCDIR)/%.cpp,$(TESTDIR)/%,$(TESTSRC))

test: $(TESTBINS)
	$(call verb, "Running tests...")
	@ for t in $(TESTBINS); do echo "--- $$t"; $$t || exit 1; done
	$(call verb, "Running tests: done.")

$(TESTDIR)/test_%: $(TESTSRCDIR)/test_%.cpp | $(TESTDIR)
	$(CXX) $(CXXFLAGS) $(TESTFLAGS) $(INCLUDES) -MMD -MP -o $@ $<

-include $(wildcard $(OBJDIR)/*.d)
-include $(wildcard $(TESTDIR)/*.d)

clean: clean-obj clean-lib
clean-obj:
	$(DEL) -r $(OBJDIR)
clean-lib:
	$(DEL) $(TARGET)

verb.build.lib:
	$(call verb, "Building library...")
verb.build.lib.done:
	$(call verb, "Building library: done.")
MKEOF

git add -A
git commit --quiet -m "build: one Make tree for the consolidated layout

Four makefiles plus make/common.mk. common.mk owns what the three groups must
agree on -- output paths, platform detection, and the diagnostic flags -- so
the Windows block is stated once instead of three times (cuda-lib's copy had
already drifted: no IS_WINDOWS, no PICFLAG clearing, MOSUF=obj).

DIAGFLAGS is now compiler-detected: clang gets -ferror-limit=1, gcc gets
-fmax-errors=1, both get -ftemplate-backtrace-limit=0. \`make CXX=g++\` used to
fail on the flags themselves, before reaching any source, and only worked via a
command-line CXXFLAGS= override.

src/lib/ and src/lib-cuda/ gain -MMD -MP; in a header-only codebase their
absence meant a kernel-header edit left stale objects. lib-cpu already had it.

Preserved deliberately: build/ and build/lib/ as the output paths (downstream
setup.py hardcodes both), \`all\` as the default target producing both shared
objects, the BOUNDFLAGS/SPLINEFLAGS/FF_TEST_SPARSE semantics including the
target-specific plain-\` = \` assignments and the comment explaining why they
cannot be \`?=\`, and the split CUDA MODULES list."

say "stage 7: teach the test-baseline gate the new layout"
python3 - <<'PYBASE'
#!/usr/bin/env python3
"""Adapt tools/test-baseline.sh to the consolidated tree.

The gate has to be able to measure BOTH trees -- the six-repo layout and the
consolidated one -- because the whole correctness argument is a diff between
them. So this teaches the existing script the new layout by auto-detection
rather than forking it: `--tree DIR` now recognises a consolidated root by the
presence of make/common.mk, and everything downstream reads two new variables
(CPU_TESTS_DIR / BUILD_ROOT) instead of assuming <cpu-lib>/tests and
<cpu-lib>/build.
"""
import sys

p = 'tools/test-baseline.sh'
s = open(p, encoding='utf-8').read()
orig = s

def sub(old, new, n=1):
    global s
    assert s.count(old) == n, f'expected {n} occurrence(s) of:\n{old!r}\ngot {s.count(old)}'
    s = s.replace(old, new)

# ---- 1. document the new layout in --help -------------------------------
sub("""#   --tree DIR      Directory holding the wired repo checkouts. Must contain
#                   fastfields-cpu-lib/ (with impl -> ../fastfields-cpu-impl and
#                   impl/kernels -> ../fastfields-kernels). If the symlinks are
#                   missing but sibling checkouts exist, they are created.
#                   Alternatively DIR may itself be a consolidated tree that
#                   already resolves the include chain -- see --cpu-lib.""",
"""#   --tree DIR      Either layout, detected automatically:
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
#                   they are byte-identical.""")

# ---- 2. detect the consolidated layout ----------------------------------
sub("""    if   [ -f "$TREE/fastfields-cpu-lib/Makefile" ]; then CPU_LIB="$TREE/fastfields-cpu-lib"
    elif [ -f "$TREE/Makefile" ] && [ -d "$TREE/tests" ]; then CPU_LIB="$TREE"
    else die "cannot find a cpu-lib checkout under $TREE (pass --cpu-lib)"
    fi
fi""",
"""    if   [ -f "$TREE/make/common.mk" ] && [ -f "$TREE/src/lib-cpu/Makefile" ]; then
        CONSOLIDATED="$TREE"; CPU_LIB="$TREE/src/lib-cpu"
        [ -n "$LIB_DIR" ] || LIB_DIR="$TREE/src/lib"
    elif [ -f "$TREE/fastfields-cpu-lib/Makefile" ]; then CPU_LIB="$TREE/fastfields-cpu-lib"
    elif [ -f "$TREE/Makefile" ] && [ -d "$TREE/tests" ]; then CPU_LIB="$TREE"
    else die "cannot find a cpu-lib checkout under $TREE (pass --cpu-lib)"
    fi
fi""")

sub('TREE=""\nCPU_LIB=""', 'TREE=""\nCPU_LIB=""\nCONSOLIDATED=""')

# ---- 3. layout-dependent paths ------------------------------------------
sub("""CPU_LIB="$(cd -- "$CPU_LIB" 2>/dev/null && pwd)" || die "no such directory: $CPU_LIB"
[ -f "$CPU_LIB/Makefile" ] || die "$CPU_LIB has no Makefile"
[ -d "$CPU_LIB/tests" ]    || die "$CPU_LIB has no tests/ directory"

# The include chain has to resolve, or the build tests the wrong code (or fails
# in a way that looks like a source bug). Check a file from each layer.
for probe in impl/pushpull.h impl/kernels/bounds.h impl/kernels/restrict.h; do
    [ -f "$CPU_LIB/$probe" ] || die "submodule chain not wired: missing $CPU_LIB/$probe
  expected cpu-lib/impl -> fastfields-cpu-impl and cpu-impl/kernels -> fastfields-kernels
  (either as symlinks, or via a --recursive submodule checkout)"
done""",
"""CPU_LIB="$(cd -- "$CPU_LIB" 2>/dev/null && pwd)" || die "no such directory: $CPU_LIB"
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
            include/fastfields/impl/kernels/bounds.h
            include/fastfields/impl/kernels/restrict.h
            include/fastfields/core/cuda_switch.h"
    for probe in $PROBES; do
        [ -f "$CONSOLIDATED/$probe" ] \\
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
[ -d "$CPU_TESTS_DIR" ] || die "no test sources at $CPU_TESTS_DIR" """.rstrip() + "\n")

# ---- 4. the lib leg -----------------------------------------------------
sub("""    if [ ! -f "$LIB_DIR/Makefile" ] || [ ! -d "$LIB_DIR/tests" ]; then""",
    """    local lib_tests="$LIB_DIR/tests"
    [ -n "$CONSOLIDATED" ] && lib_tests="$CONSOLIDATED/tests/lib"
    if [ ! -f "$LIB_DIR/Makefile" ] || [ ! -d "$lib_tests" ]; then""")

sub('    rm -rf -- "$LIB_DIR/build"',
    '    rm -rf -- "${LIB_BUILD_ROOT:-$LIB_DIR/build}"')

# ---- 5. the cpu-lib legs ------------------------------------------------
sub("""    rm -rf -- "$CPU_LIB/build\"""", """    rm -rf -- "$BUILD_ROOT\"""")

assert s != orig
open(p, 'w', encoding='utf-8').write(s)
print('patched tools/test-baseline.sh for dual-layout support')
PYBASE

git add -A
git commit --quiet -m "tools: teach test-baseline.sh the consolidated layout

The gate has to measure BOTH trees -- the six-repo one and this one -- because
the migration's whole correctness argument is that the two reports are
byte-identical. So the script auto-detects the layout (make/common.mk) rather
than being forked, and the layout-dependent paths it used to assume
(<cpu-lib>/tests, <cpu-lib>/build) become variables."

say "done: $OUT/repo"
git -C "$OUT/repo" log --oneline -8
