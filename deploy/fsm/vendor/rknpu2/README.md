# rknpu2 — Rockchip's NPU runtime, fetched rather than committed

The controller talks to the RK3576's NPU through two files from Rockchip:

| File | Used by |
|---|---|
| `include/rknn_api.h` | `build.rs`, on every build with the `device` feature: `src/rknn_layout_probe.c` includes it so the Rust mirror of `rknn_tensor_attr` is checked against the real layout on any host |
| `lib/aarch64/librknnrt.so` | `build.rs`, under `--features rknn` only: the board binary links against it |

Neither is in the repository. Both are Rockchip's, distributed under the
[RKNN SDK License](https://github.com/airockchip/rknn-toolkit2/blob/v2.3.2/LICENSE),
which is proprietary: it permits use for developing applications for Rockchip
products and cannot be re-granted under this repository's Apache-2.0. Rockchip
publishes both files in [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2),
so the build takes them from there.

```bash
bash deploy/fsm/vendor/rknpu2/fetch.sh
```

`docker-build.sh` runs it before every build, so the cross-build and
`docker-build.sh test` need nothing extra. A native `cargo build` or `cargo test`
with the default `device` feature needs it run once by hand; `build.rs` stops
with a message pointing here when the header is missing. The builds that
`scripts/deploy.py` runs on the host (`--no-default-features`, for the browser and
`play --app`) do not include the header and do not need it.

## The pin

`fetch.sh` takes release **v2.3.2** and checks both files' sha256. The version is
not free to float:

- `deploy/convert/requirements.txt` pins `rknn-toolkit2==2.3.2`, and a `.rknn`
  does not load under a `librknnrt` older than the toolkit that produced it;
- the board carries `librknnrt` 2.3.2 (`deploy/README.md`, *Verified against the
  board*), and `.claude/skills/deploy/scripts/check_board.py` checks it.

Moving to another release means changing all three together, and the two sha256
values in `fetch.sh` with them.

The files `fetch.sh` installs are byte-identical to the copies that were
committed here before the repository was opened (checked 2026-09-29), so a build
from a fresh clone links what the board was verified against.
