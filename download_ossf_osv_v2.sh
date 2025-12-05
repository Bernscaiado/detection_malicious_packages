# Still in: ~/Desktop/"CSI 4900 Jupyter"/data/external

# 1) Download tarball of the repo
TMP_TAR="$(mktemp /tmp/ossf-malpkgs-XXXXXX.tar.gz)"
echo "Downloading malicious-packages tarball..."
curl -L "https://github.com/ossf/malicious-packages/archive/refs/heads/main.tar.gz" -o "$TMP_TAR"

# 2) Create target directory
mkdir -p ossf-malicious-packages

echo "Extracting tarball with Windows-safe path transformations..."
# Inside the tar, the top-level directory is 'malicious-packages-main/'
# We strip that, and apply two transforms:
#   - replace ':' with '_'   (for vscode:open-vsx.org etc.)
#   - strip trailing '.'     (for 'mad-1.0.0.2.2.8.' etc.)
tar -xzf "$TMP_TAR" \
  -C ossf-malicious-packages \
  --strip-components=1 \
  --transform='s/:/_/g' \
  --transform='s/\.$//g'

# 3) Clean up temp tarball
rm -f "$TMP_TAR"

echo "Done extracting."
echo "Root of repo (sanitized) is: $(pwd)/ossf-malicious-packages"

echo "Checking OSV malicious directories:"
find ossf-malicious-packages/osv/malicious -maxdepth 3 -type d

echo "Counting JSON files for PyPI and npm:"
find ossf-malicious-packages/osv/malicious/pypi -name '*.json' 2>/dev/null | wc -l
find ossf-malicious-packages/osv/malicious/npm  -name '*.json' 2>/dev/null | wc -l
