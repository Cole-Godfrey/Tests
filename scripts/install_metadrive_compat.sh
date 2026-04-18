#!/usr/bin/env bash

set -euo pipefail

repo_url="${1:-https://github.com/HenryLHH/metadrive_clean.git}"
repo_ref="${2:-main}"

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

echo "[metadrive] cloning $repo_url@$repo_ref"
git clone --depth 1 --branch "$repo_ref" "$repo_url" "$workdir/metadrive_clean"

echo "[metadrive] patching stale dependency pins for Python 3.10/Linux"
python - <<'PY' "$workdir/metadrive_clean/setup.py"
from pathlib import Path
import sys

setup_path = Path(sys.argv[1])
text = setup_path.read_text()
replacements = {
    '"setuptools<65.0",': '"setuptools<82",',
    '"panda3d==1.10.8",': '"panda3d==1.10.16",',
    '"protobuf==3.20.1",': '"protobuf>=3.19.6,<4",',
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f"expected token not found in {setup_path}: {old}")
    text = text.replace(old, new)
setup_path.write_text(text)
PY

echo "[metadrive] installing patched metadrive package"
pip install "$workdir/metadrive_clean"
