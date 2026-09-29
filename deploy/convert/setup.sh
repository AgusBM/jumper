#!/usr/bin/env bash
# Build deploy/convert/.venv, the conversion environment.
#
# It is a separate virtualenv on purpose. rknn-toolkit2 pins numpy<=1.26.4,
# torch<=2.4.0 and (in practice) onnx==1.16.1; installing it into the
# repository's .venv downgrades all three and breaks training -- silently, since
# pip reports success and the damage only shows up the next time something
# imports mjlab.
#
# Usage:
#     bash deploy/convert/setup.sh
#
# Then, from the repository root:
#     deploy/convert/.venv/bin/python deploy/convert/onnx2rknn.py --bundle <bundle dir>

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

os=$(uname -s)
arch=$(uname -m)
if [ "$os" != "Linux" ] || { [ "$arch" != "x86_64" ] && [ "$arch" != "aarch64" ]; }; then
    echo "rknn-toolkit2 publishes Linux x86_64 / aarch64 wheels only; this is $os $arch." >&2
    echo "Convert on a Linux PC and copy the .rknn to the board." >&2
    exit 1
fi

# rknn-toolkit2 2.3.2 has wheels for cp36 through cp312. 3.13 finds nothing and
# pip's message ("no matching distribution") does not say why.
py_minor=$(python3 -c 'import sys; print(sys.version_info[1])')
if [ "$py_minor" -gt 12 ] || [ "$py_minor" -lt 8 ]; then
    echo "python3.$py_minor found; rknn-toolkit2 2.3.2 ships wheels for 3.8 - 3.12 only." >&2
    echo "Point this script at one of those: python3.12 -m venv $VENV, then pip install -r ..." >&2
    exit 1
fi

if [ ! -d "$VENV" ]; then
    echo "creating $VENV"
    python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install --upgrade pip --quiet
"$VENV/bin/pip" install -r "$HERE/requirements.txt"

echo
echo "--- verifying ---"
"$VENV/bin/python" - <<'PY'
import platform
from importlib.metadata import version
from rknn.api import RKNN

print(f"rknn-toolkit2 {version('rknn-toolkit2')}  "
      f"onnx {version('onnx')}  numpy {version('numpy')}  torch {version('torch')}")
print(f"host {platform.system()} {platform.machine()}  python {platform.python_version()}")
RKNN(verbose=False).release()
print("ok")
PY

echo
echo "Convert a bundle with:"
echo "    deploy/convert/.venv/bin/python deploy/convert/onnx2rknn.py --bundle <bundle dir>"
