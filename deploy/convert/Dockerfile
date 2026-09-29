# Conversion image: actor.onnx -> actor.rknn, on a host of any kind.
#
# rknn-toolkit2 publishes Linux wheels only (x86_64 and aarch64, no macOS, no
# Windows), so on anything else the toolkit cannot be installed at all. This
# image is the way those machines convert: Docker Desktop on macOS or Windows
# runs a Linux VM, and the build below produces an image for whatever
# architecture the host runs -- arm64 on Apple Silicon, amd64 elsewhere. Both are
# architectures the toolkit ships for, so the same Dockerfile serves both without
# a `--platform` flag.
#
# Build and run through `docker-convert.sh`, which handles the mount and the file
# ownership; see README.md.

# Debian rather than Alpine: the rknn-toolkit2 wheels are manylinux_2_17, which
# is glibc. On musl they install and then fail to load their .so.
FROM python:3.12-slim

# rknn-toolkit2 declares opencv-python, which links libGL and libglib. Neither is
# in a slim image, and without them `import rknn.api` dies on a missing
# libGL.so.1 -- an error that reads like a broken package rather than a missing
# system library. ~35 MB, and it is the whole reason this is not a one-line
# Dockerfile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/convert

# Requirements first, so editing the converter does not re-download ~1.5 GB of
# wheels. The torch pin inside is selected by `platform_machine`, which is what
# lets one file serve both architectures.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY onnx2rknn.py .

# Bundles are mounted here; see docker-convert.sh. Left as the working directory
# so that a relative bundle path on the command line means the same thing inside
# the container as it did outside.
WORKDIR /work

# HOME is set at run time to a writable directory: the container runs as the
# calling user (so output files are not root-owned), and that user has no home
# inside the image. Several libraries write caches to ~ and fail without it.
ENTRYPOINT ["python", "/opt/convert/onnx2rknn.py"]
