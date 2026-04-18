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

echo "[metadrive] installing a compatible dependency set without upgrading OSRL core packages"
pip install --upgrade --force-reinstall --no-deps \
  "setuptools<82" \
  "numpy==1.24.4" \
  "scipy==1.10.1" \
  "numba==0.57.1" \
  "protobuf==3.19.6" \
  "panda3d==1.10.16" \
  "panda3d-gltf==1.3.0" \
  "panda3d-simplepbr==0.13.1" \
  "pandas==2.1.4" \
  "pytz==2024.2" \
  "tzdata==2024.2" \
  "seaborn==0.13.2" \
  "pillow==10.4.0" \
  "opencv-python-headless==4.8.1.78" \
  "lxml==5.4.0"

echo "[metadrive] installing patched metadrive package"
pip install --force-reinstall --no-deps "$workdir/metadrive_clean"
