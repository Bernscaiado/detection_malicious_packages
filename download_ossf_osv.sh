#!/usr/bin/env bash
set -euo pipefail

# Where your project lives:
PROJECT_ROOT="$(pwd)"

# Match your notebook layout:
DATA_DIR="$PROJECT_ROOT/data"
BASE_DIR="$DATA_DIR/external/ossf-malicious-packages"
OSV_BASE="$BASE_DIR/osv/malicious"

mkdir -p "$OSV_BASE"

# GitHub API endpoint for full tree of the repo
TREE_API="https://api.github.com/repos/ossf/malicious-packages/git/trees/main?recursive=1"

# Optional: GitHub token to avoid API limits (strongly recommended)
#   export GITHUB_TOKEN=ghp_xxx...
AUTH_OPTS=()
if [ -n "${GITHUB_TOKEN:-}" ]; then
  AUTH_OPTS=( -H "Authorization: Bearer ${GITHUB_TOKEN}" )
fi

echo "=== Fetching repository tree from OpenSSF malicious-packages ==="
echo "API: $TREE_API"
tree_json="$(curl -sSL "${AUTH_OPTS[@]}" "$TREE_API")"

# Sanity-check response
if ! echo "$tree_json" | jq -e '.tree and (.tree | type=="array")' >/dev/null 2>&1; then
  echo "ERROR: Unexpected JSON from GitHub tree API (maybe rate limited?):"
  echo "$tree_json" | head -c 400
  echo
  exit 1
fi

# Extract paths for OSV malicious JSON for PyPI + npm only
echo "=== Filtering OSV malicious JSON paths for PyPI + npm ==="
echo "$tree_json" \
  | jq -r '.tree[]
           | select(.type == "blob")
           | .path' \
  | grep '^osv/malicious/' \
  | grep -E '/(pypi|npm)/' \
  > /tmp/osv_paths.txt

total_paths="$(wc -l < /tmp/osv_paths.txt || echo 0)"
echo "Found $total_paths JSON files under osv/malicious/{pypi,npm}/ in the repo."

if [ "$total_paths" -eq 0 ]; then
  echo "No paths found – something is wrong (check filters / repo)."
  exit 1
fi

echo "=== Downloading OSV JSON files ==="
i=0
while read -r path; do
  i=$((i+1))

  # path looks like: osv/malicious/pypi/pkgname/MAL-0000-0000.json
  rel="${path#osv/malicious/}"         # pypi/pkgname/MAL-....json
  eco="${rel%%/*}"                     # pypi or npm
  sub="${rel#*/}"                      # pkgname/MAL-....json

  out_path="$OSV_BASE/$eco/$sub"
  out_dir="$(dirname "$out_path")"
  mkdir -p "$out_dir"

  # Raw URL is much simpler & less rate-limited than contents API
  raw_url="https://raw.githubusercontent.com/ossf/malicious-packages/main/$path"

  printf '(%5d/%5d) %s -> %s\n' "$i" "$total_paths" "$raw_url" "$out_path"
  curl -sSL "$raw_url" -o "$out_path"

done < /tmp/osv_paths.txt

echo "=== Done ==="
echo "Local OSV directory: $OSV_BASE"
echo "PyPI JSON files: $(find "$OSV_BASE/pypi" -name '*.json' 2>/dev/null | wc -l || echo 0)"
echo "npm  JSON files: $(find "$OSV_BASE/npm"  -name '*.json' 2>/dev/null | wc -l || echo 0)"
