#!/usr/bin/env bash
# Rerun studies and fail if their committed files drift.
#
# A study is a directory studies/<name>/ with measure.py and requirements.txt.
# Each study runs in its own fresh virtualenv with its pinned requirements,
# from inside its own directory. After the run, nothing tracked in the study
# may change and no new untracked file may appear (files covered by
# .gitignore, such as __pycache__/, are allowed).
#
#     scripts/check_studies.sh                       # every study
#     scripts/check_studies.sh 01-tokenizer-waste    # named studies
#
# PYTHON picks the interpreter for the virtualenvs (default: python3).
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"
python=${PYTHON:-python3}

dirs=()
if [ "$#" -gt 0 ]; then
  for name in "$@"; do dirs+=("studies/${name%/}"); done
else
  shopt -s nullglob
  for d in studies/*/; do dirs+=("${d%/}"); done
  shopt -u nullglob
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

checked=0
failed=()
for dir in "${dirs[@]}"; do
  name=$(basename "$dir")
  if [ ! -f "$dir/measure.py" ] || [ ! -f "$dir/requirements.txt" ]; then
    if [ "$#" -gt 0 ]; then
      echo "error: $dir has no measure.py and requirements.txt" >&2
      failed+=("$name")
    else
      echo "skip $dir: no measure.py and requirements.txt"
    fi
    continue
  fi
  if [ -n "$(git status --porcelain -- "$dir")" ]; then
    echo "error: $dir has uncommitted changes; commit or stash them first" >&2
    git status --short -- "$dir" >&2
    failed+=("$name")
    continue
  fi

  echo "::group::$name"
  venv="$work/$name"
  "$python" -m venv "$venv"
  "$venv/bin/python" -m pip install --quiet --disable-pip-version-check -r "$dir/requirements.txt"
  status=0
  (cd "$dir" && "$venv/bin/python" measure.py) || status=$?
  echo "::endgroup::"

  if [ "$status" -ne 0 ]; then
    echo "FAIL $name: measure.py exited $status" >&2
    failed+=("$name")
  elif [ -n "$(git status --porcelain -- "$dir")" ]; then
    echo "FAIL $name: rerun does not reproduce the committed files" >&2
    git status --short -- "$dir" >&2
    git --no-pager diff --stat -- "$dir" >&2
    git --no-pager diff -- "$dir" | head -n 200 >&2
    failed+=("$name")
  else
    echo "ok $name: rerun matches the committed files"
  fi
  checked=$((checked + 1))
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
echo "checked $checked studies"
