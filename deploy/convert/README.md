# convert — ONNX to RKNN for the RK3576 NPU

The last link in the chain that starts at a checkpoint:

```
model_*.pt  ──scripts/export.py──▶  actor.onnx  ──deploy/convert/onnx2rknn.py──▶  actor.rknn
                                    layout.json                              actor.rknn.json
```

`scripts/export.py` writes an exported policy and checks the ONNX against the torch
policy. This directory converts that ONNX for the Rockchip NPU and checks the result
against the ONNX. The two checks together pin the whole chain: a policy that behaves on
the board the way it behaved in training, or a conversion that fails loudly.

## Setup

Two ways in. Docker works everywhere and is the only option off Linux; the virtualenv is
lighter if you are already on Linux.

### Docker — any host

```bash
bash deploy/convert/docker-convert.sh --bundle tasks/jumper/tripod/out/<name>
```

Builds the image on first use, then takes the same arguments as `onnx2rknn.py` and passes
them through. `MJRL_RKNN_REBUILD=1` forces a rebuild after editing the converter.

rknn-toolkit2 publishes **Linux wheels only** — x86_64 and aarch64, no macOS, no Windows —
so on a Mac or a Windows box this is not a convenience, it is the only way to convert.
Docker Desktop runs a Linux VM and [`Dockerfile`](Dockerfile) builds for whatever
architecture the host is, arm64 on Apple Silicon and amd64 elsewhere. Both are
architectures the toolkit ships for, so no `--platform` flag is needed.

The wrapper exists because three details are each a wrong answer waiting to happen: the
repository is mounted rather than the bundle (so paths mean the same inside and out), the
container runs as the calling user (or the `.rknn` lands in your tree owned by root), and
`HOME` is redirected (that user has no home in the image, and dependencies write caches
to `~`).

### Virtualenv — Linux only

```bash
bash deploy/convert/setup.sh
```

Builds `deploy/convert/.venv` (~1.5 GB) and verifies the toolkit imports. Python
3.8 – 3.12.

**It is a separate virtualenv on purpose.** rknn-toolkit2 pins `numpy<=1.26.4`,
`torch<=2.4.0`, `onnx==1.16.1` and `setuptools<81`; the training environment is on torch
2.11+cu128, numpy 2.5 and onnx 1.22. Installing the toolkit there downgrades all of them —
including replacing the CUDA build of torch with a CPU one — and pip reports success. The
damage shows up the next time something imports mjlab. Never `pip install rknn-toolkit2`
into the training environment.

[`requirements.txt`](requirements.txt) records why every pin is there; four of them are
bugs or gaps in rknn-toolkit2's own metadata and each was found by a fresh install
failing.

## Converting

```bash
deploy/convert/.venv/bin/python deploy/convert/onnx2rknn.py \
    --bundle tasks/jumper/tripod/out/2026-09-05_14-53-00
```

Writes `actor.rknn` and `actor.rknn.json` beside the bundle's `actor.onnx`. Real output,
from the run the next section measures:

```
[rknn] source  .../actor.onnx  [1, 74] -> [1, 20]
[rknn] bundle consistent  obs=74 act=20
[rknn] test observations drawn from the normaliser baked into the graph (std 0.039 .. 2.623)
[rknn] building  target=rk3576  fp16
[rknn] agreement with onnxruntime over 64 observations: max 1.094e-03, mean 2.319e-04,
       against outputs up to 1.995
[rknn] done -> .../actor.rknn
```

Useful switches — `--onnx FILE` for a bare ONNX with no contract beside it, `--target`
for another Rockchip part, `--out`, `--samples`, `--seed`, `--tolerance`, `--verbose` for
the toolkit's own log. `--quantize` exists, and `--calib-obs` feeds it a `[N, obs_dim]`
`.npy` of recorded observations instead of sampled ones; read the next section before
reaching for either.

## Why the default is fp16 and not int8

i9-14900KF, rknn-toolkit2 2.3.2, target rk3576, the `jumper.tripod` actor **as it was
then** (74 → 512 → 256 → 128 → 20, elu), 64 in-distribution observations. The last column
converts the action error to joint angle at that task's `action_scale = 0.25`. That actor
has since grown — five frames of history make the observation 411 wide, and the hidden
layers are (512, 256, 128, 64) — and the measurement has not been repeated, so it is left
as it was taken rather than restated for a model it was never run on.

| build                | max abs difference from ONNX | joint angle |
|----------------------|------------------------------|-------------|
| fp16 (default)       | 1.1e-3                       | 0.016°      |
| int8 (`--quantize`)  | 2.7e-1                       | 3.9°        |

int8 is a 13% error against outputs of magnitude ~2 — 3.9° of error on every joint,
every control step. The model is a 250 KB MLP that the NPU runs in fp16 in well under a
millisecond; quantization buys nothing worth that. The default `--tolerance` (2e-2) sits
between the two numbers deliberately, so `--quantize` fails the check unless the
tolerance is also raised by hand.

The RK3576 NPU has no fp32 mode. The 1.1e-3 above is the fp16 floor, not conversion
damage.

## What the check actually checks

The converted model is run in the toolkit's simulator against onnxruntime on the same
observations, and the outputs compared element-wise. It is built to one side and moved
into place only after it passes, so **a failed check leaves no `.rknn` on disk** — there
is nothing for someone to pick up later without the scrollback that condemned it. Three
more things are worth knowing.

**The observations come from the policy's own training distribution.** The exported
actor carries its `EmpiricalNormalization` inside the graph — the leading `Sub`/`Div`
pair, whose constants are the observation mean and standard deviation measured during
training. `onnx2rknn.py` reads them and samples `mean + std * N(0,1)`. Feeding raw
`N(0,1)` instead puts terms whose training standard deviation is 0.039 about 25σ from
their mean, the activations blow up, and fp16 error grows with them: 5.0e-3 rather than
1.1e-3. That is measuring the test inputs, not the conversion.

**The simulator is not the board.** It models the NPU's arithmetic closely enough to
catch a conversion that changed the graph, which is what this check is for. It is not a
substitute for running the policy once on real hardware before trusting it.

The toolkit prints two warnings during a good conversion, both expected:

- `The config.mean_values is None, zeros will be set for input 0` — correct. The
  normalisation is already inside the graph; setting `mean_values` would apply it twice.
  This warning is the confirmation that it is not.
- `The 'data_format' is not set, and its default value is 'nhwc'` — harmless for a
  `[1, obs_dim]` input, which has no spatial layout to reorder. The 1.1e-3 agreement is
  the evidence: a permuted input would not land within 1e-3 of the ONNX.

## On the board

The `.rknn` is loaded by `librknnrt.so`, which is part of the board image rather than
this repository. **It must be at least as new as the toolkit that built the model** —
2.3.2 here. A model built by a newer toolkit than the runtime fails to load, usually
with an unhelpful error:

```bash
strings /usr/lib/librknnrt.so | grep -i 'librknnrt version'
```

Inference itself is `rknn_init` → `rknn_inputs_set` → `rknn_run` → `rknn_outputs_get`
from the C API, or `RKNNLite` from `rknn-toolkit-lite2` in Python. Feed the observation
as a flat `float32` vector of the contract's own `observation.dim` — 411 for `jumper.tripod`
today — in the layout `layout.json` describes, and apply the returned action exactly as
the bundle's own `README.md` says — `target = action * action_scale +
default_joint_pos[joint]`, paired joint by joint in `action_joint_order`. Nothing about
the conversion changes that contract.

`actor.rknn.json` records what produced the model: the source ONNX's sha256, the toolkit
version, the target, and the verification numbers. It answers "is this `.rknn` the one
built from that `.onnx`?" without relying on the two files having stayed in the same
directory.

**It records the ONNX's checksum rather than the `.rknn`'s, because the `.rknn` does not
have a stable one.** Converting the same ONNX twice on the same machine, back to back,
produces files of identical length that differ in ~14 kB — always the same byte range,
filled with what look like leaked heap pointers, i.e. uninitialised padding the toolkit
serialises. Both builds reported the same agreement with onnxruntime to four figures, so
the models are equivalent; only the bytes are not. The container and the virtualenv differ
from each other in exactly the same way and for the same reason, so a byte difference
between them means nothing either.

The practical consequence: **do not checksum a `.rknn` to decide whether to redeploy.**
It changes every build. Compare `onnx_sha256` in the sidecar instead.

## Reconverting

`.rknn` files are build products and are not committed (`.gitignore` covers `*.rknn`
alongside `*.onnx`). Reconvert whenever `actor.onnx` changes. `scripts/deploy.py` does
it for you: it reconverts any `.rknn` whose sidecar `onnx_sha256` no longer matches before
bundling. Nothing on the board checks, so a `.rknn` copied there by hand beside a fresh
`.onnx` is still silent.
