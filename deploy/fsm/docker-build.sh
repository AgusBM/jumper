#!/usr/bin/env bash
# Cross-build the FSM for the RK3576 board, in a container.
#
#     bash deploy/fsm/docker-build.sh              # release build for aarch64
#     bash deploy/fsm/docker-build.sh test         # run the host test suite instead
#     MJRL_FSM_REBUILD=1 bash deploy/fsm/docker-build.sh
#
#     # `play --app` on Windows and on macOS: the controller as a CPython extension
#     bash deploy/fsm/docker-build.sh build --release --target x86_64-pc-windows-gnu \
#         --no-default-features --features py
#     bash deploy/fsm/docker-build.sh zigbuild --release --target universal2-apple-darwin \
#         --no-default-features --features py
#
# Anything after the script name is passed to cargo, so `test`, `clippy` and
# `build --features ...` all work.
#
# The image is built on first use and takes a while: it compiles CycloneDDS
# twice, once for the host (to get `idlc`, which runs at build time) and once for
# aarch64 (the library the binary links against).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
IMAGE="mjrl-fsm-build:latest"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found. Install Docker Desktop (macOS / Windows) or docker.io (Linux)." >&2
    exit 1
fi

# An image built before the Windows or macOS layer was added to the Dockerfile
# has every other layer and lacks that one, and cargo would say only that the
# target "may not be installed" -- or that `zigbuild` is no command. Rebuilding
# it reuses the layers it has.
stale=""
need=()
case " $* " in *x86_64-pc-windows-gnu*) need+=(x86_64-pc-windows-gnu) ;; esac
case " $* " in *apple-darwin*) need+=(aarch64-apple-darwin x86_64-apple-darwin cargo-zigbuild) ;; esac
if [ ${#need[@]} -gt 0 ] && docker image inspect "$IMAGE" >/dev/null 2>&1; then
    # Read whole, then searched: `grep -q` in a pipe under `pipefail` can close
    # it early and turn a found line into a failed pipeline. And `if`, not `&&`:
    # a missing command would end the probe non-zero, and `set -e` this script
    # with it, silently.
    have=$(docker run --rm --entrypoint sh "$IMAGE" -c \
        'rustup target list --installed; if command -v cargo-zigbuild >/dev/null; then echo cargo-zigbuild; fi')
    for n in "${need[@]}"; do
        grep -qx "$n" <<<"$have" || stale=1
    done
fi

if [ -n "${MJRL_FSM_REBUILD:-}" ] || [ -n "$stale" ] \
    || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[docker] building $IMAGE (compiles CycloneDDS twice; this takes a while)"
    docker build -t "$IMAGE" "$HERE"

    # The board carries libddsc.so.11. A CycloneDDS whose soname differs links
    # here and then fails to load on the robot, which is a bad place to find out.
    soname=$(docker run --rm --entrypoint sh "$IMAGE" -c \
        'ls /opt/cyclonedds-aarch64/lib/libddsc.so.* 2>/dev/null | head -1 | xargs -r basename')
    echo "[docker] aarch64 CycloneDDS soname: ${soname:-<none found>}"
    case "$soname" in
        libddsc.so.11*) ;;
        *) echo "[docker] WARNING: expected libddsc.so.11 to match the board; got '${soname}'." >&2
           echo "         Rebuild with --build-arg CYCLONEDDS_REF=<tag matching the board>." >&2 ;;
    esac
fi

# Docker creates a named volume owned by root, and the build below runs as the
# calling user, which then cannot write its download cache or its object files
# into them. Chown once, at creation: doing it on every run would cost a
# container start, and doing it inside the build container would need root there.
for vol in mjrl-fsm-target mjrl-fsm-cargo; do
    if ! docker volume inspect "$vol" >/dev/null 2>&1; then
        docker volume create "$vol" >/dev/null
        docker run --rm --user 0 --volume "$vol:/v" --entrypoint chown "$IMAGE" \
            -R "$(id -u):$(id -g)" /v
    fi
done

# Set the default as positional parameters rather than with `${@:-...}`, which
# would pass the whole string as one argument and reach cargo as an unknown
# subcommand.
if [ $# -eq 0 ]; then
    set -- build --release --target aarch64-unknown-linux-gnu --features rknn
fi

# Rockchip's header and runtime are fetched, not committed (vendor/rknpu2/README.md).
# Every build with the default `device` feature compiles against the header, so
# fetch unless the build has turned `device` off -- the browser and `play --app`
# builds, which should not need the network to succeed.
case " $* " in
    *" --no-default-features "*) ;;
    *) bash "$HERE/vendor/rknpu2/fetch.sh" ;;
esac

# The image carries CycloneDDS twice, and `build.rs` picks one through
# CYCLONEDDS_HOME. The image's default is the aarch64 tree, which is right for a
# cross build and wrong for anything running here: `cargo test` builds for
# x86_64 and would then link the aarch64 library ("incompatible with
# elf64-x86-64"). So the tree follows the target named in the arguments.
#
# LD_LIBRARY_PATH matters for the same reason but at run time: `cargo test`
# executes what it builds, and CycloneDDS is under /opt rather than anywhere the
# dynamic linker looks by default.
case " $* " in
    *aarch64*) DDS_HOME=/opt/cyclonedds-aarch64 ;;
    *)         DDS_HOME=/opt/cyclonedds-host ;;
esac

# Two named volumes, both because the container runs as the calling user:
#   - the target dir, so a root-owned target/ in the source tree cannot block the
#     next host-side `cargo build`;
#   - the cargo registry, because the image's CARGO_HOME belongs to root and a
#     non-root build cannot write its download cache there.
# Being volumes rather than tmpfs is what keeps the crate downloads across runs.
# Where a cross build's binary is left for `scripts/deploy.py`.
#
# The container's target dir is a named volume, on purpose -- see above -- so
# nothing it produces is visible from the host. Nobody had cross-built before,
# so nobody had noticed that the bundler therefore never found a `controller`
# and every board bundle shipped without one, saying so in a note that read like
# a step not yet taken.
#
# Under `out/deploy/`, which is already where that script keeps its artifacts,
# rather than back inside cargo's target tree: writing there from a container is
# what the volume exists to avoid.
OUT="$REPO/out/deploy/controller-aarch64"

docker run --rm -i \
    --user "$(id -u):$(id -g)" \
    --env HOME=/tmp \
    --env CARGO_HOME=/cargo \
    --env CYCLONEDDS_HOME="$DDS_HOME" \
    --env LD_LIBRARY_PATH="$DDS_HOME/lib" \
    --env CARGO_TARGET_DIR=/target \
    --volume "$REPO:/work" \
    --volume mjrl-fsm-target:/target \
    --volume mjrl-fsm-cargo:/cargo \
    --workdir /work/deploy/fsm \
    "$IMAGE" "$@"

# Only for a cross build. `test` and `clippy` produce nothing to hand over.
#
# The Windows DLL is renamed `.pyd` on the way out: CPython on Windows finds an
# extension by that suffix and by no other, and `scripts/deploy.py` ships it
# under the name it lands with here. The macOS dylib becomes `.so`, which is
# what CPython there looks for.
WIN_OUT="$REPO/out/deploy/controller-win-amd64.pyd"
MAC_OUT="$REPO/out/deploy/controller-macosx-universal2.so"
case " $* " in
    *universal2-apple-darwin*)
        docker run --rm --entrypoint sh \
            --user "$(id -u):$(id -g)" \
            --volume "$REPO:/work" \
            --volume mjrl-fsm-target:/target \
            --workdir /work \
            "$IMAGE" -c \
            'mkdir -p out/deploy && cp /target/universal2-apple-darwin/release/libmjrl_fsm.dylib \
             out/deploy/controller-macosx-universal2.so'
        # cargo-zigbuild's probe of zig, left in the crate; see `.gitignore`.
        rm -f "$HERE/.intentionally-empty-file.o"
        echo "[docker] Mach-O universal: $(file -b "$MAC_OUT" | grep -oE 'Mach-O 64-bit [a-z0-9_]+' \
            | sed 's/Mach-O 64-bit //' | tr '\n' ' ')"
        echo "[docker] -> ${MAC_OUT#"$REPO"/}   ($(du -h "$MAC_OUT" | cut -f1))"
        echo "[docker]    scripts/deploy.py --bundle copies it from there into runtime/mjlab/macosx-universal2/."
        ;;
    *x86_64-pc-windows-gnu*)
        docker run --rm --entrypoint sh \
            --user "$(id -u):$(id -g)" \
            --volume "$REPO:/work" \
            --volume mjrl-fsm-target:/target \
            --workdir /work \
            "$IMAGE" -c \
            'mkdir -p out/deploy && cp /target/x86_64-pc-windows-gnu/release/mjrl_fsm.dll \
             out/deploy/controller-win-amd64.pyd'
        echo "[docker] $(file -b "$WIN_OUT" | cut -d, -f1-2)"
        echo "[docker] -> ${WIN_OUT#"$REPO"/}   ($(du -h "$WIN_OUT" | cut -f1))"
        echo "[docker]    scripts/deploy.py --bundle copies it from there into runtime/mjlab/win-amd64/."
        ;;
    *aarch64*)
        # `--entrypoint`: the image's is `cargo`, and this step is a copy.
        docker run --rm --entrypoint sh \
            --user "$(id -u):$(id -g)" \
            --volume "$REPO:/work" \
            --volume mjrl-fsm-target:/target \
            --workdir /work \
            "$IMAGE" -c \
            'mkdir -p out/deploy && cp /target/aarch64-unknown-linux-gnu/release/controller \
             out/deploy/controller-aarch64'
        echo "[docker] $(file -b "$OUT" | cut -d, -f1-2)"
        echo "[docker] -> ${OUT#"$REPO"/}   ($(du -h "$OUT" | cut -f1))"
        echo "[docker]    scripts/deploy.py --bundle copies it from there into runtime/board/controller."
        ;;
esac
