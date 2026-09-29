# Cross-build the FSM for the RK3576 board, from a host of any kind.
#
# Cross-compiling, not emulating. Running an arm64 container under qemu needs a
# `binfmt` registration carrying the `F` flag, which is host-wide state a build
# should not be quietly changing -- and it is far slower. This image is x86_64
# and targets aarch64 through a normal cross toolchain.
#
#     bash deploy/fsm/docker-build.sh
#
# ## Why ubuntu:22.04 and not something newer
#
# **The base image's glibc is a hard floor on the board.** The RK3576 image is
# Ubuntu 22.04 (glibc 2.35); a binary linked against Debian bookworm's 2.36 dies
# on the board with `version GLIBC_2.36 not found` -- at startup, on the robot,
# nowhere near the build that caused it. 22.04 here matches the rootfs the board
# actually runs.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates pkg-config \
        gcc-aarch64-linux-gnu g++-aarch64-linux-gnu libc6-dev-arm64-cross \
        clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*

# CycloneDDS is needed twice, for two different reasons:
#
#   - **x86_64**, for `idlc`. It is a code generator that runs on the build
#     machine, so it must be a host binary whatever the target is.
#   - **aarch64**, for `libddsc.so`, which the binary links against.
#
# The ref is an argument because the shared-library soname has to match the
# board's. The board carries `libddsc.so.11`; a CycloneDDS that produces a
# different soname builds fine here and then cannot find its library at runtime.
# `docker-build.sh` checks the built soname rather than trusting this default.
ARG CYCLONEDDS_REF=master
ARG CYCLONEDDS_URL=https://github.com/eclipse-cyclonedds/cyclonedds.git

RUN git clone --depth 1 --branch "${CYCLONEDDS_REF}" "${CYCLONEDDS_URL}" /src/cyclonedds

# 1) Host build -- only idlc is kept.
RUN cmake -S /src/cyclonedds -B /build/host \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds-host \
        -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_IDLC=ON \
    && cmake --build /build/host -j"$(nproc)" \
    && cmake --install /build/host \
    && rm -rf /build/host

# 2) Target build -- the library the FSM links against. `-DBUILD_IDLC=OFF`
#    because an aarch64 idlc could not run here anyway.
RUN cmake -S /src/cyclonedds -B /build/target \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/cyclonedds-aarch64 \
        -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
        -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
        -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
        -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_IDLC=OFF \
    && cmake --build /build/target -j"$(nproc)" \
    && cmake --install /build/target \
    && rm -rf /build/target /src/cyclonedds

ENV PATH="/opt/cyclonedds-host/bin:/usr/local/cargo/bin:${PATH}"

ARG RUST_VERSION=1.98.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | env CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup \
          sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}" \
              --target aarch64-unknown-linux-gnu --no-modify-path
ENV CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup

# `build.rs` reads CYCLONEDDS_HOME for the headers and the link path. It points
# at the **aarch64** tree; idlc comes from PATH, which is the host tree above.
ENV CYCLONEDDS_HOME=/opt/cyclonedds-aarch64 \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc \
    CXX_aarch64_unknown_linux_gnu=aarch64-linux-gnu-g++ \
    AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar
# bindgen parses the generated headers with libclang, which defaults to the host
# target. Without this it would compute x86_64 struct layouts for an aarch64
# binary -- the two agree here, but silently, and would stop agreeing the moment
# a message gained a pointer or a long.
ENV BINDGEN_EXTRA_CLANG_ARGS_aarch64_unknown_linux_gnu="--target=aarch64-linux-gnu --sysroot=/usr/aarch64-linux-gnu"

# The second cross build: `play --app` on Windows, the controller as a CPython
# extension for `win-amd64`. The GNU target rather than MSVC, because MinGW is
# an apt package and MSVC's libraries come only with a licence to accept. The
# DLL it makes imports `python3.dll` and system DLLs and nothing else: no
# libgcc or winpthread to ship beside it.
#
# Its own layer, after the board's, so that adding it to an image built before
# it reuses every layer above -- CycloneDDS is not compiled again. The macOS
# layer below is added the same way.
RUN apt-get update && apt-get install -y --no-install-recommends gcc-mingw-w64-x86-64 \
    && rm -rf /var/lib/apt/lists/* \
    && rustup target add x86_64-pc-windows-gnu

# The third: `play --app` on macOS, one universal2 file carrying arm64 and
# x86_64, so every Mac and every Python build on it -- Homebrew's, conda's,
# python.org's universal2 one, one under Rosetta -- loads the same file and
# dyld picks the slice. Linked by zig through cargo-zigbuild, which carries
# libSystem's stubs, so no Apple SDK is needed and none has a licence to accept;
# rustc's warning that `xcrun` cannot find one is expected and harmless. zig
# also signs the arm64 slice ad hoc, without which Apple Silicon kills the
# process that loads it. Minimum macOS 11.7.1, zig 0.13's floor.
#
# Both pinned: which zig links a binary is not visible in it.
ARG ZIG_VERSION=0.13.0
ARG CARGO_ZIGBUILD_VERSION=0.23.4
RUN curl -fsSL "https://ziglang.org/download/${ZIG_VERSION}/zig-linux-x86_64-${ZIG_VERSION}.tar.xz" \
        | tar -xJ -C /opt \
    && ln -s "/opt/zig-linux-x86_64-${ZIG_VERSION}/zig" /usr/local/bin/zig \
    && cargo install --locked "cargo-zigbuild@${CARGO_ZIGBUILD_VERSION}" \
    && rm -rf /usr/local/cargo/registry \
    && rustup target add aarch64-apple-darwin x86_64-apple-darwin

WORKDIR /work
ENTRYPOINT ["cargo"]
CMD ["build", "--release", "--target", "aarch64-unknown-linux-gnu", "--features", "rknn"]
