#!/usr/bin/env bash
# Fetch Rockchip's NPU runtime into this directory, pinned by release and by
# content:
#
#     bash deploy/fsm/vendor/rknpu2/fetch.sh
#
# `docker-build.sh` runs this itself, so only a native `cargo build` / `cargo test`
# with the `device` feature needs it run by hand. See README.md here for why the
# files are fetched rather than committed.
#
# The sha256 is checked on every run, not only after a download: a header or a
# library from another release compiles and links cleanly, and the mismatch shows
# up on the board as a wrong struct layout or a model that will not load.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Must match rknn-toolkit2 in deploy/convert/requirements.txt: a .rknn does not
# load under a librknnrt older than the toolkit that produced it.
RELEASE=v2.3.2
BASE="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/$RELEASE/rknpu2/runtime/Linux/librknn_api"

# local path | path under $BASE | sha256
FILES=(
    "include/rknn_api.h|include/rknn_api.h|c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a"
    "lib/aarch64/librknnrt.so|aarch64/librknnrt.so|d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8"
)

# sha256sum on Linux and Git Bash, shasum on macOS.
sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

for entry in "${FILES[@]}"; do
    IFS='|' read -r dest src want <<<"$entry"
    path="$HERE/$dest"

    if [ -f "$path" ]; then
        got="$(sha256 "$path")"
        if [ "$got" != "$want" ]; then
            echo "[rknpu2] $dest is not the $RELEASE file (sha256 $got, expected $want)." >&2
            echo "         Delete it and run this script again." >&2
            exit 1
        fi
        continue
    fi

    # Download beside the destination and verify before moving it into place, so
    # an interrupted or wrong download never sits at the name build.rs looks for.
    mkdir -p "$(dirname "$path")"
    echo "[rknpu2] fetching $dest ($RELEASE)"
    curl -fsSL --retry 3 -o "$path.part" "$BASE/$src"
    got="$(sha256 "$path.part")"
    if [ "$got" != "$want" ]; then
        rm -f "$path.part"
        echo "[rknpu2] $BASE/$src: sha256 $got, expected $want." >&2
        echo "         The upstream file has changed under the $RELEASE tag;" >&2
        echo "         nothing was installed." >&2
        exit 1
    fi
    mv "$path.part" "$path"
done

echo "[rknpu2] $RELEASE present and verified"
