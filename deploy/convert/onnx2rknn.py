#!/usr/bin/env python3
"""Convert an exported `actor.onnx` into an `.rknn` for the RK3576 NPU.

`scripts/export.py` produces an exported policy -- `actor.onnx` plus
`layout.json`. This takes the ONNX out of that bundle, converts it for the
Rockchip NPU, and **checks the result numerically** before calling it done.

Run it with this directory's own interpreter, never the repository's:

    deploy/convert/.venv/bin/python deploy/convert/onnx2rknn.py \
        --bundle tasks/jumper/tripod/out/2026-09-05_14-53-00

rknn-toolkit2 pins `numpy<=1.26.4`, `torch<=2.4.0` and `onnx==1.16.1`.
Installing it into the training environment downgrades all three, so it gets a
virtualenv of its own; `setup.sh` builds it.

## The two silent failures this guards against

**The RKNN API returns error codes, it does not raise.** `build()` answering -1
and the script carrying on leaves the previous `.rknn` on disk -- or no file at
all -- and exits 0. Every toolkit call here is checked.

**A conversion can succeed and still be wrong.** An op fused into something with
different arithmetic, a layout the toolkit guessed, quantization left on: all of
these produce a valid `.rknn` that behaves differently from the ONNX. So the
converted model is run in the toolkit's simulator against onnxruntime on the same
observations and the outputs are compared element-wise. This mirrors
`scripts/export.py::_validate_onnx`, which does the same one link earlier in the
chain; together the two pin torch -> ONNX -> RKNN.

## Which observations the check is run on

The exported actor carries its `EmpiricalNormalization` inside the graph -- the
leading `Sub` / `Div` pair, whose constants are the observation mean and standard
deviation measured during training. This script reads them and draws test
observations from `mean + std * N(0, 1)`, i.e. from the distribution the policy
actually saw.

That is not cosmetic. Feeding raw `N(0, 1)` instead puts terms whose training
standard deviation is 0.039 about 25 sigma from their mean; the activations blow
up and fp16 rounding error grows with them, so the comparison measures the test
inputs rather than the conversion. Measured on the jumper.tripod actor: 5.0e-3 max
absolute difference on raw `N(0, 1)`, 1.1e-3 on in-distribution observations.

If the graph does not start with that `Sub` / `Div` pair the script says so and
falls back to `N(0, 1)`, rather than pretending the check is as tight.

## Measurements

i9-14900KF, rknn-toolkit2 2.3.2, target rk3576, the jumper.tripod actor
(74 -> 512 -> 256 -> 128 -> 20, elu), 64 in-distribution observations. The last
column converts the action error into joint angle at that task's
`action_scale = 0.25`.

| build                       | max abs difference from ONNX | joint angle |
|-----------------------------|------------------------------|-------------|
| fp16 (the default)          | 1.1e-3                       | 0.016 deg   |
| int8 (`--quantize`)         | 2.7e-1                       | 3.9 deg     |

int8 is a 13% error against outputs of magnitude ~2. **Do not quantize this
class of model**: it is a 250 KB MLP, the NPU runs it in fp16 in well under a
millisecond, and there is nothing to buy with that accuracy. The flag exists so
that the number above can be reproduced rather than argued about, and the
default `--tolerance` is set below the int8 error on purpose -- quantizing
requires raising it by hand.

The RK3576 NPU has no fp32 mode; fp16 is the floor, and the 1.1e-3 above is
that floor, not conversion damage.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

#: Default agreement required between the RKNN simulator and onnxruntime.
#: Chosen to sit an order of magnitude above the fp16 floor (1.1e-3, measured
#: above) and an order of magnitude below the int8 error (2.7e-1), so it passes
#: rounding noise and fails anything that changed the arithmetic.
DEFAULT_TOLERANCE = 2e-2

#: Calibration samples for `--quantize`. rknn's own examples use a few hundred.
CALIBRATION_SAMPLES = 256


# ──────────────────────────────────────────────────────────────────────
# The host has to be right before anything else is worth trying
# ──────────────────────────────────────────────────────────────────────


def _require_host() -> None:
    """rknn-toolkit2 ships Linux x86_64 wheels only.

    Said plainly here because the alternative is an ImportError from a
    `.so` several frames deep, which reads like a broken install rather than an
    unsupported machine.
    """
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise SystemExit(
            f"[rknn] rknn-toolkit2 runs on Linux x86_64 (or aarch64); this is "
            f"{platform.system()} {platform.machine()}. Convert on a Linux PC and "
            f"copy the .rknn to the board."
        )


def _shown(path: Path) -> str:
    """A path as the reader would type it: relative to where they are, if it can be.

    The commands in the message below are meant to be copied, and this file sits
    two directories down. Naming the directory (`convert/setup.sh`) is wrong from
    anywhere, and an absolute path is right but unreadable; relative to the
    working directory is what someone standing at the repository root can paste.
    """
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _require_toolkit():
    """Import the toolkit, or say which interpreter to use instead.

    The likeliest way to reach this is running the script with the repository's
    `.venv`, where rknn-toolkit2 is deliberately absent.
    """
    if importlib.util.find_spec("rknn") is None:
        here = Path(__file__).resolve().parent
        raise SystemExit(
            f"[rknn] rknn-toolkit2 is not installed in this interpreter.\n"
            f"       It must not be installed into the training environment -- it "
            f"pins numpy<=1.26.4, torch<=2.4.0 and onnx==1.16.1.\n"
            f"       Build its own environment:  bash {_shown(here / 'setup.sh')}\n"
            f"       Then run:  {_shown(here / '.venv/bin/python')} "
            f"{_shown(Path(__file__).resolve())} ..."
        )
    from rknn.api import RKNN

    return RKNN


def _check(ret: int, what: str) -> None:
    """Every RKNN call returns a status code instead of raising."""
    if ret != 0:
        raise SystemExit(f"[rknn] {what} failed (rknn returned {ret}); see the log above")


# ──────────────────────────────────────────────────────────────────────
# Reading what the ONNX and the contract say about themselves
# ──────────────────────────────────────────────────────────────────────


def _io_shapes(model) -> tuple[list[int], list[int]]:
    def dims(value):
        return [d.dim_value for d in value.type.tensor_type.shape.dim]

    return dims(model.graph.input[0]), dims(model.graph.output[0])


def _observation_stats(model, obs_dim: int):
    """The observation mean and standard deviation baked into the graph head.

    `rsl_rl`'s `_OnnxMLPModel.forward` runs the `EmpiricalNormalization` before
    the MLP, so an exported actor starts with `Sub(obs, mean)` then
    `Div(_, std)`. Only the first two nodes are examined: a `Sub` further in is
    part of the network and its constants are not observation statistics.
    """
    import numpy as np
    from onnx import numpy_helper

    nodes = list(model.graph.node)
    if len(nodes) < 2 or nodes[0].op_type != "Sub" or nodes[1].op_type != "Div":
        return None
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    mean, std = init.get(nodes[0].input[1]), init.get(nodes[1].input[1])
    if mean is None or std is None or mean.size != obs_dim or std.size != obs_dim:
        return None
    return np.asarray(mean, np.float32).reshape(1, obs_dim), np.asarray(
        std, np.float32
    ).reshape(1, obs_dim)


def _sample_observations(n: int, obs_dim: int, stats, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    if stats is None:
        return rng.normal(0.0, 1.0, (n, obs_dim)).astype(np.float32)
    mean, std = stats
    return (mean + std * rng.normal(0.0, 1.0, (n, obs_dim))).astype(np.float32)


def _cross_check_contract(contract: dict, ishape: list[int], oshape: list[int]) -> None:
    """The bundle's contract and its ONNX must agree.

    They come out of the same run of `scripts/export.py`, so disagreement means
    the directory holds files from two different exports -- the one case where
    converting the ONNX would produce a model that does not match the joint order
    and gains the deployment will use.
    """
    want_obs = contract["observation"]["dim"]
    want_act = contract["action"]["dim"]
    if ishape[-1] != want_obs or oshape[-1] != want_act:
        raise SystemExit(
            f"[rknn] the bundle is inconsistent: the layout says "
            f"obs={want_obs} act={want_act}, actor.onnx is {ishape} -> {oshape}. "
            f"Re-export the bundle."
        )
    print(f"[rknn] bundle consistent  obs={want_obs} act={want_act}")


# ──────────────────────────────────────────────────────────────────────
# Conversion
# ──────────────────────────────────────────────────────────────────────


def _write_calibration(observations, directory: Path) -> Path:
    """rknn takes calibration data as a text file of per-sample file paths."""
    import numpy as np

    lines = []
    for i, obs in enumerate(observations):
        path = directory / f"obs_{i:04d}.npy"
        np.save(path, obs.reshape(1, -1))
        lines.append(str(path))
    listing = directory / "dataset.txt"
    listing.write_text("\n".join(lines), encoding="utf-8")
    return listing


def convert(onnx_path: Path, out_path: Path, target: str, quantize: bool,
            calibration, verbose: bool):
    """Build the `.rknn` and return an initialised simulator session."""
    RKNN = _require_toolkit()
    rknn = RKNN(verbose=verbose)

    # mean_values / std_values are deliberately left unset. The normalisation is
    # already inside the graph (see the module docstring); setting them here
    # would apply it a second time, and the toolkit's "zeros will be set for
    # input 0" warning is the confirmation that it is not being applied twice.
    rknn.config(target_platform=target)
    _check(rknn.load_onnx(model=str(onnx_path)), f"load_onnx({onnx_path})")

    with tempfile.TemporaryDirectory(prefix="rknn-calib-") as tmp:
        dataset = str(_write_calibration(calibration, Path(tmp))) if quantize else None
        _check(rknn.build(do_quantization=quantize, dataset=dataset), "build")

    _check(rknn.export_rknn(str(out_path)), f"export_rknn({out_path})")
    _check(rknn.init_runtime(), "init_runtime (simulator)")
    return rknn


def verify(rknn, onnx_path: Path, observations) -> tuple[float, float, float]:
    """Compare the converted model against onnxruntime, observation by observation.

    Returns (max abs difference, mean abs difference, max abs ONNX output). The
    third is what makes the first two readable: a 1e-3 difference means one thing
    against outputs of order 2 and another against outputs of order 200.
    """
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    worst, total, magnitude = 0.0, 0.0, 0.0
    for obs in observations:
        x = obs.reshape(1, -1)
        y_onnx = session.run(None, {name: x})[0]
        y_rknn = np.asarray(rknn.inference(inputs=[x])[0]).reshape(y_onnx.shape)
        diff = np.abs(y_rknn - y_onnx)
        worst = max(worst, float(diff.max()))
        total += float(diff.mean())
        magnitude = max(magnitude, float(np.abs(y_onnx).max()))
    return worst, total / len(observations), magnitude


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="onnx2rknn.py",
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--bundle", type=Path,
        help="an exported policy from scripts/export.py (actor.onnx + layout.json)",
    )
    source.add_argument("--onnx", type=Path, help="a bare .onnx file, no contract to check against")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output .rknn; defaults to <bundle>/actor.rknn, or <onnx>.rknn",
    )
    parser.add_argument(
        "--target", default="rk3576",
        help="target platform, validated by rknn-toolkit2 (default: rk3576)",
    )
    parser.add_argument(
        "--samples", type=int, default=64, help="observations used for the check (default: 64)"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for those observations")
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help=f"max allowed difference from onnxruntime (default: {DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "--quantize", action="store_true",
        help="build int8 instead of fp16. Measured at a 13%% error on an actor MLP; "
             "read the module docstring before using it",
    )
    parser.add_argument(
        "--calib-obs", type=Path, default=None,
        help="[N, obs_dim] float32 .npy of recorded observations for --quantize; "
             "without it, calibration is drawn from the baked-in normaliser",
    )
    parser.add_argument("--verbose", action="store_true", help="the toolkit's own verbose log")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.calib_obs is not None and not args.quantize:
        # Silently ignoring it would read as "calibration data was used"
        raise SystemExit("[rknn] --calib-obs is calibration data for --quantize, which is off")
    _require_host()
    _require_toolkit()

    import numpy as np
    import onnx

    if args.bundle is not None:
        onnx_path = args.bundle / "actor.onnx"
        if not onnx_path.is_file():
            raise SystemExit(f"[rknn] no actor.onnx in {args.bundle} -- is that a bundle?")
        # `contract.json` is what scripts/export.py wrote before the file took the
        # name the on-robot importer looks for. Bundles under that name are still
        # readable here; nothing regenerates an old export just to rename it.
        contract_path = next(
            (p for p in (args.bundle / "layout.json", args.bundle / "isaac_layout.json",
                         args.bundle / "contract.json")
             if p.is_file()),
            None,
        )
    else:
        onnx_path = args.onnx
        if not onnx_path.is_file():
            raise SystemExit(f"[rknn] no such file: {onnx_path}")
        contract_path = None

    out_path = args.out or onnx_path.with_suffix(".rknn")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(onnx_path))
    ishape, oshape = _io_shapes(model)
    if len(ishape) != 2 or ishape[0] != 1:
        raise SystemExit(
            f"[rknn] expected a [1, obs_dim] input, got {ishape}. This script converts "
            f"the single-observation actor that scripts/export.py writes."
        )
    obs_dim, act_dim = ishape[-1], oshape[-1]
    print(f"[rknn] source  {onnx_path}  {ishape} -> {oshape}")

    contract = None
    if contract_path is not None and contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        _cross_check_contract(contract, ishape, oshape)

    stats = _observation_stats(model, obs_dim)
    if stats is None:
        print(
            "[rknn] WARNING: the graph does not start with the Sub/Div normaliser pair, "
            "so the check runs on raw N(0,1) observations. That is a looser check -- "
            "out-of-distribution inputs inflate fp16 error (5.0e-3 against 1.1e-3, "
            "measured); read the tolerance below with that in mind."
        )
    else:
        print(
            f"[rknn] test observations drawn from the normaliser baked into the graph "
            f"(std {float(stats[1].min()):.3f} .. {float(stats[1].max()):.3f})"
        )

    observations = _sample_observations(args.samples, obs_dim, stats, args.seed)

    if args.quantize and args.calib_obs is not None:
        calibration = np.load(args.calib_obs).astype(np.float32)
        if calibration.ndim != 2 or calibration.shape[1] != obs_dim:
            raise SystemExit(
                f"[rknn] --calib-obs must be [N, {obs_dim}], got {calibration.shape}"
            )
        print(f"[rknn] calibrating on {len(calibration)} recorded observations")
    elif args.quantize:
        calibration = _sample_observations(
            CALIBRATION_SAMPLES, obs_dim, stats, args.seed + 1
        )
        print(
            f"[rknn] WARNING: calibrating on {CALIBRATION_SAMPLES} sampled observations "
            f"rather than recorded ones. Pass --calib-obs to do better."
        )
    else:
        calibration = None

    print(f"[rknn] building  target={args.target}  "
          f"{'int8 (quantized)' if args.quantize else 'fp16'}")

    # Built to one side and moved into place only once it has been checked. A
    # model that failed verification must not be left on disk under the name the
    # deployment reads: the next person to look sees a plausible .rknn and a
    # scrollback they did not write.
    staging = out_path.with_name(out_path.name + ".partial")
    try:
        rknn = convert(onnx_path, staging, args.target, args.quantize, calibration, args.verbose)
        try:
            worst, mean, magnitude = verify(rknn, onnx_path, observations)
        finally:
            rknn.release()

        print(
            f"[rknn] agreement with onnxruntime over {args.samples} observations: "
            f"max {worst:.3e}, mean {mean:.3e}, against outputs up to {magnitude:.3f}"
        )
        if worst > args.tolerance:
            raise SystemExit(
                f"[rknn] the converted model disagrees with the ONNX by {worst:.3e}, over "
                f"the {args.tolerance:.3e} tolerance. Nothing was written.\n"
                + (
                    "       int8 quantization is the likely cause: measured at 2.7e-1 on "
                    "an actor MLP, which is 3.9 degrees of joint error. Drop --quantize.\n"
                    if args.quantize
                    else "       fp16 rounding alone measures 1.1e-3; a difference this "
                         "large means the conversion changed the arithmetic.\n"
                )
            )
        staging.replace(out_path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise

    meta = {
        "_source": "deploy/convert/onnx2rknn.py -- reconvert whenever actor.onnx changes",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "onnx": str(onnx_path),
        # So that "which ONNX is this .rknn from?" has an answer that does not
        # depend on the two files having stayed in the same directory
        "onnx_sha256": _sha256(onnx_path),
        "target_platform": args.target,
        "quantized": bool(args.quantize),
        "precision": "int8" if args.quantize else "float16",
        "toolkit_version": _toolkit_version(),
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "verification": {
            "samples": args.samples,
            "seed": args.seed,
            "observations": "baked_normalizer" if stats is not None else "standard_normal",
            "max_abs_diff": worst,
            "mean_abs_diff": mean,
            "max_abs_output": magnitude,
            "tolerance": args.tolerance,
        },
        **({"task_contract": {"action_joint_order": contract["action_joint_order"]}}
           if contract else {}),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[rknn] done -> {out_path}")
    for f in (out_path, meta_path):
        print(f"    {f.name:<20}{f.stat().st_size / 1024:8.1f} KB")
    print(
        "\n[rknn] the board's librknnrt.so must be at least as new as the toolkit "
        f"({_toolkit_version()}); check it with\n"
        "       strings /usr/lib/librknnrt.so | grep -i 'librknnrt version'"
    )


def _toolkit_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("rknn-toolkit2")
    except PackageNotFoundError:
        return "unknown"


if __name__ == "__main__":
    main()
