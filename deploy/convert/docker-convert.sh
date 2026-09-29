#!/usr/bin/env bash
# Run the ONNX -> RKNN conversion inside a container, from any host.
#
# Takes the same arguments as onnx2rknn.py and passes them straight through:
#
#     bash deploy/convert/docker-convert.sh --bundle tasks/jumper/tripod/out/<name>
#     bash deploy/convert/docker-convert.sh --bundle <dir> --target rk3588
#
# The image is built on first use and reused after that. Force a rebuild after
# editing the converter or its requirements:
#
#     MJRL_RKNN_REBUILD=1 bash deploy/convert/docker-convert.sh --bundle <dir>
#
# Three details this wraps, each of which is a wrong answer waiting to happen:
#
#   - **The repository is mounted, not the bundle.** Bundle paths are given
#     relative to the repository root, and mounting only the bundle would break
#     every path someone copies out of the README.
#   - **The container runs as the calling user.** Docker's default is root, and
#     the .rknn it writes would land in the tree owned by root -- on Linux the
#     next export, running as you, then cannot overwrite it.
#   - **HOME is redirected to /tmp.** That user has no home inside the image, and
#     several of the toolkit's dependencies write caches to ~ on import.
#
# On macOS and Windows this is the only way to convert: rknn-toolkit2 has no
# wheel for either. On Linux it works too, but `setup.sh` is lighter.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# Tagged with the toolkit version, because that version has to match the board's
# librknnrt.so: two images side by side should be distinguishable at a glance.
TOOLKIT="$(sed -n 's/^rknn-toolkit2==\(.*\)$/\1/p' "$HERE/requirements.txt" | head -1)"
IMAGE="mjrl-rknn:${TOOLKIT:-latest}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found. Install Docker Desktop (macOS / Windows) or docker.io (Linux)." >&2
    exit 1
fi

if [ -n "${MJRL_RKNN_REBUILD:-}" ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[docker] building $IMAGE (first run pulls ~1.5 GB of wheels)"
    docker build -t "$IMAGE" "$HERE"
fi

# --user keeps output files owned by the caller. On macOS and Windows the bind
# mount already maps ownership and the flag is harmless.
exec docker run --rm -i \
    --user "$(id -u):$(id -g)" \
    --env HOME=/tmp \
    --volume "$REPO:/work" \
    --workdir /work \
    "$IMAGE" "$@"
