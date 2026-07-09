#!/usr/bin/env bash
# Pre-submission anonymity self-check for a double-blind release
# (ESWA / anonymous GitHub). Run before every push:  bash tools/prepare_submission.sh
# - scans working-tree code for personal identifiers
# - confirms sensitive dirs (data / checkpoints / results / paper) are gitignored
# Exits non-zero if anything would leak identity or bloat the release.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1. Personal-identifier scan (tracked files only = what actually gets pushed) =="
HITS=$(git ls-files '*.py' '*.sh' '*.md' '*.txt' \
  | grep -v '^tools/prepare_submission.sh$' \
  | xargs grep -nIE "chouheiwa|849131492|@qq\.com|@icloud\.com|/home/[a-z]+/experiment" 2>/dev/null || true)
if [ -n "$HITS" ]; then
  echo "!! IDENTITY LEAK (in tracked files):"; echo "$HITS"; echo; echo "== FAIL =="; exit 1
fi
echo "OK: no personal identifiers in tracked files (gitignored dirs are not scanned; they are not pushed)."

echo "== 2. Sensitive dirs must be gitignored (not pushed) =="
FAIL=0
for d in output result hf_data paper wandb logs; do
  if git check-ignore -q "$d" 2>/dev/null; then
    echo "OK ignored : $d"
  elif [ ! -e "$d" ]; then
    echo "OK absent  : $d"
  else
    echo "!! TRACKED : $d  (add to .gitignore before pushing)"; FAIL=1
  fi
done

echo "== 3. What would be published =="
echo "tracked files: $(git ls-files | wc -l)"

[ "$FAIL" -eq 0 ] && { echo; echo "== PASS: safe to push to the anonymous remote =="; } \
                   || { echo; echo "== FAIL: fix the items above =="; exit 1; }
