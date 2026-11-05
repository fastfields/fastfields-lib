#!/usr/bin/env bash
#
# Publish one built docs site into a subtree of the shared GitHub Pages branch.
#
# The fastfields documentation is one site assembled from four repositories:
#
#     /          fastfields.github.io   (the landing page + the shared theme)
#     /python/   fastfields
#     /dlpack/   fastfields-dlpack
#     /lib/      fastfields-lib
#     /whl/      whl -- NOT here; its own project Pages site, independent
#
# All but `whl` publish into the `gh-pages` branch of fastfields.github.io,
# which is what Pages serves. The one invariant that makes that safe:
#
#     *** A publisher writes ONLY its own subtree. ***
#
# Get that wrong with a full checkout-and-replace and each deploy silently
# deletes the other three sections. So the sync below is scoped: a subdirectory
# publisher rsyncs inside its own directory and can reach nothing else, and the
# root publisher excludes every subtree owned by someone else. An excluded path
# is also protected from --delete by rsync, which is the property being relied
# on here.
#
# Concurrency: `concurrency:` groups in GitHub Actions are per-repository, so
# they cannot serialize publishers that live in different repos. Two repos can
# and will deploy at the same time. That is handled the only way it can be --
# optimistically: fetch, re-apply onto the current tip, push, and on a
# non-fast-forward rejection start over against the new tip. The sync is
# idempotent, so a retry is always safe.
#
# Usage:  publish-docs.sh --subdir <name|""> [--site site]
# Needs:  GH_TOKEN with `contents: write` on the target repo.

set -euo pipefail

SUBDIR=""
SITE="site"
TARGET_REPO="${TARGET_REPO:-fastfields/fastfields.github.io}"
BRANCH="${BRANCH:-gh-pages}"
ATTEMPTS="${ATTEMPTS:-5}"

# Subtrees owned by other publishers. Only consulted for the root publisher.
OTHERS=(python dlpack lib)

while [ $# -gt 0 ]; do
  case "$1" in
    --subdir) SUBDIR="${2:-}"; shift 2 ;;
    --site)   SITE="${2:-}";   shift 2 ;;
    *) echo "publish-docs.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

[ -d "$SITE" ] || { echo "publish-docs.sh: '$SITE' is not a directory -- did the build run?" >&2; exit 1; }

# In CI the remote is built from GH_TOKEN. PUBLISH_REMOTE overrides it, so the
# scoping and the retry can be exercised for real against a scratch repo or a
# fork without minting a token.
if [ -z "${PUBLISH_REMOTE:-}" ] && [ -z "${GH_TOKEN:-}" ]; then
  echo "publish-docs.sh: set GH_TOKEN (or PUBLISH_REMOTE)." >&2
  exit 1
fi

# Reject anything that could escape the intended subtree.
case "$SUBDIR" in
  */*|.|..|/*|*..*) echo "publish-docs.sh: refusing unsafe --subdir '$SUBDIR'" >&2; exit 2 ;;
esac

if [ -z "$SUBDIR" ]; then
  echo "publish-docs.sh: publishing the SITE ROOT of $TARGET_REPO@$BRANCH"
else
  echo "publish-docs.sh: publishing /$SUBDIR/ of $TARGET_REPO@$BRANCH"
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
REMOTE="${PUBLISH_REMOTE:-https://x-access-token:${GH_TOKEN}@github.com/${TARGET_REPO}.git}"

for attempt in $(seq 1 "$ATTEMPTS"); do
  CLONE="$WORK/try$attempt"

  # A fresh clone each attempt is what makes the retry correct: the re-apply
  # happens onto whatever tip won the race, not onto a stale index.
  if git clone --quiet --depth 1 --branch "$BRANCH" "$REMOTE" "$CLONE" 2>/dev/null; then
    :
  else
    echo "  branch '$BRANCH' does not exist yet; creating it"
    git clone --quiet --depth 1 "$REMOTE" "$CLONE"
    git -C "$CLONE" checkout --quiet --orphan "$BRANCH"
    git -C "$CLONE" rm -rq --cached . 2>/dev/null || true
    find "$CLONE" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
  fi

  if [ -z "$SUBDIR" ]; then
    # --delete at the clone root would otherwise delete .git itself, since it
    # is not part of the built site. A subdirectory publisher never hits this
    # because it syncs into $CLONE/$SUBDIR/.
    EXCLUDES=( "--exclude=/.git" )
    for d in "${OTHERS[@]}"; do EXCLUDES+=( "--exclude=/$d" ); done
    rsync -a --delete "${EXCLUDES[@]}" "$SITE/" "$CLONE/"
  else
    mkdir -p "$CLONE/$SUBDIR"
    rsync -a --delete "$SITE/" "$CLONE/$SUBDIR/"
  fi

  git -C "$CLONE" add -A
  if git -C "$CLONE" diff --cached --quiet; then
    echo "publish-docs.sh: no change to publish."
    exit 0
  fi

  git \
    -C "$CLONE" \
    -c user.name="github-actions[bot]" \
    -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
    commit --quiet \
    -m "docs: publish ${SUBDIR:-/} from ${GITHUB_REPOSITORY:-local}@${GITHUB_SHA:-manual}"

  if git -C "$CLONE" push --quiet origin "$BRANCH"; then
    echo "publish-docs.sh: published ${SUBDIR:-/} (attempt $attempt)."
    exit 0
  fi

  echo "  push rejected (another publisher won the race); retrying against the new tip"
  sleep $(( attempt * 5 ))
done

echo "publish-docs.sh: giving up after $ATTEMPTS attempts." >&2
exit 1
