#!/usr/bin/env python3
"""Raw throughput benchmark: native (multi-threaded CPU) vs warp (cuda) vs
warp (cpu).

This is a **standalone** script that does not depend on the mjrl framework. What
it measures is raw physics throughput (env-steps/s), the number that decides
whether real CPU training is viable.

WARNING: **the initial state is not a training condition.** This script starts
from `mj_resetData` with `ctrl=0`, so the robot is uncontrolled, mostly in free
fall, and its feet are not on the ground.

The backend comparison (native vs warp/cuda vs warp/cpu) is **unaffected**: all
three share the same initial state and the comparison still holds, which is the
reason this script exists.

But **do not use it to compare collision geometry**. An earlier version had a
`--collision` switch and measured "hybrid gains nothing on GPU, primitive is 9.7x
faster"; that data is invalid, because after 400 steps the three schemes are not
in the same condition at all (hybrid and primitive are still airborne while mesh
has landed). The switch has been removed, and conclusions about collision geometry
come from measurements on a real training task instead; see docs/DESIGN.md.

Usage:
    python tools/bench_phase0.py --model path/to/robot.mjcf
    python tools/bench_phase0.py --model m.xml --envs 16,64,256 --threads 1,8,16,32
"""

from __future__ import annotations

import argparse
import platform
import time

import mujoco
import numpy as np
from mujoco import rollout

# ── Collision geometry ────────────────────────────────────────────────────
# jumper.xml carries several geom sets side by side, distinguished by name (see
# assets/jumper/tools/build_jumper.py):
#   <body>_visual    display only     <body>_meshcol   original meshes
#   <body>_collision capsules / spheres / boxes
# Compiled as-is all of them are active, which corresponds to no real
# configuration and makes the measurement meaningless. So this collapses them, on
# the **already compiled MjModel**, down to the hybrid scheme the tasks actually
# use (link capsules plus foot meshes). There is deliberately **no option** here.
# contype/conaffinity are edited directly rather than going through mjlab, to keep
# this script standalone.
FOOT_BODIES = (
    "LF_palm_pad_b_link", "RF_palm_pad_b_link",
    "LM_foot_tip_link", "RM_foot_tip_link",
    "LR_foot_tip_link", "RR_foot_tip_link",
)


def apply_hybrid_collision(m: mujoco.MjModel) -> int:
    """Collapse the coexisting geom sets to hybrid; return the colliding geom count.

    A model that does not follow the `_meshcol` / `_collision` naming convention
    (smoke_biped, for instance, has only one set) is **returned unchanged** --
    otherwise all of its collision geometry would be turned off.
    """
    names = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)
    ]
    if not any(n.endswith("_meshcol") for n in names):
        return int((m.geom_contype != 0).sum())

    foot_bids = set()
    for b in FOOT_BODIES:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if bid >= 0:
            foot_bids.add(bid)

    n_on = 0
    for g, name in enumerate(names):
        if m.geom_bodyid[g] == 0:  # leave worldbody geoms (the ground) alone
            continue
        is_foot = m.geom_bodyid[g] in foot_bids
        on = (name.endswith("_meshcol") and is_foot) or (
            name.endswith("_collision") and not is_foot
        )
        m.geom_contype[g] = 1 if on else 0
        m.geom_conaffinity[g] = 1 if on else 0
        n_on += int(on)
    return n_on


# ── Timing ────────────────────────────────────────────────────────────────


def _timed(fn, warmup: int, repeat: int) -> float:
    """Run warmup times, then time repeat runs; return mean seconds per run."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat


# ── The native backend (native MuJoCo + a rollout thread pool) ────────────


def bench_native(
    path: str,
    n_envs: int,
    n_step: int,
    n_thread: int,
    warmup: int,
    repeat: int,
    per_env_model: bool = True,
) -> float:
    """Return env-steps/s.

    The conventions that matter (established by measurement, not from the
    documentation) in `mujoco.rollout`:
      - the `data` list must be **exactly nthread long** -- MjData is per-thread
        scratch
      - `model` may be a single instance or a list of length nbatch, the latter
        being per-environment domain randomisation
      - batched state lives in the `initial_state` / `state` numpy arrays

    So the native backend needs no N MjData copies, and its memory grows only with
    the thread count.

    With per_env_model=True, N MjModel copies are built to reproduce the real cost
    of domain randomisation.
    """
    m0 = mujoco.MjModel.from_xml_path(path)
    apply_hybrid_collision(m0)
    if per_env_model:
        models = []
        for _ in range(n_envs):
            mi = mujoco.MjModel.from_xml_path(path)
            apply_hybrid_collision(mi)
            models.append(mi)
    else:
        models = m0
    datas = [mujoco.MjData(m0) for _ in range(n_thread)]  # one scratch per thread

    n_state = mujoco.mj_stateSize(m0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    d0 = mujoco.MjData(m0)
    mujoco.mj_resetData(m0, d0)
    mujoco.mj_forward(m0, d0)
    one = np.zeros(n_state, dtype=np.float64)
    mujoco.mj_getState(m0, d0, one, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    initial = np.tile(one, (n_envs, 1))

    control = np.zeros((n_envs, n_step, m0.nu), dtype=np.float64)
    # Preallocate the output buffers: the native backend holds these for its
    # lifetime and wraps torch tensors directly around them
    state_out = np.zeros((n_envs, n_step, n_state), dtype=np.float64)
    sensor_out = np.zeros((n_envs, n_step, m0.nsensordata), dtype=np.float64)

    pool = rollout.Rollout(nthread=n_thread)

    def _once() -> None:
        pool.rollout(
            models,
            datas,
            initial,
            control,
            nstep=n_step,
            state=state_out,
            sensordata=sensor_out,
        )

    try:
        sec = _timed(_once, warmup, repeat)
    finally:
        pool.close()
    return n_envs * n_step / sec


# ── The warp backend (MuJoCo Warp, cuda or cpu) ───────────────────────────


def bench_warp(
    path: str,
    n_envs: int,
    n_step: int,
    device: str,
    warmup: int,
    repeat: int,
    njmax: int | None = None,
    nconmax: int | None = None,
) -> float:
    """njmax and nconmax are **per-world** limits.

    mujoco_warp infers defaults from the mjData passed in, and a resting pose has
    far fewer contacts than a moving one, so a contact-dense model (a hexapod with
    all six legs down, for instance) hits
    ``nefc overflow - please increase njmax to N`` partway through training. Hence
    the option to set them explicitly.
    """
    import warp as wp
    import mujoco_warp as mjwarp

    wp.init()
    with wp.ScopedDevice(device):
        mj_model = mujoco.MjModel.from_xml_path(path)
        apply_hybrid_collision(mj_model)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)

        kw = {}
        if njmax is not None:
            kw["njmax"] = njmax
        if nconmax is not None:
            kw["nconmax"] = nconmax
        m = mjwarp.put_model(mj_model)
        d = mjwarp.put_data(mj_model, mj_data, nworld=n_envs, **kw)

        def _once() -> None:
            for _ in range(n_step):
                mjwarp.step(m, d)
            wp.synchronize()

        sec = _timed(_once, warmup, repeat)
    return n_envs * n_step / sec


# ── Reporting ─────────────────────────────────────────────────────────────


def _fmt(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:8.2f}M"
    if v >= 1e3:
        return f"{v / 1e3:8.1f}k"
    return f"{v:9.0f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="path to the MJCF")
    ap.add_argument("--envs", default="16,64,256",
                    help="environment counts, comma separated")
    ap.add_argument("--threads", default="1,8,16,32",
                    help="native thread counts, comma separated")
    ap.add_argument("--nstep", type=int, default=4,
                    help="physics steps advanced per call (= decimation)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--skip-warp", action="store_true")
    ap.add_argument("--njmax", type=int, default=None,
                    help="warp: per-world constraint row limit")
    ap.add_argument("--nconmax", type=int, default=None,
                    help="warp: per-world contact limit")
    ap.add_argument(
        "--shared-model",
        action="store_true",
        help="share one MjModel across all environments (disabling per-environment "
             "domain randomisation). Required for mesh-heavy models, which would "
             "otherwise exhaust memory",
    )
    args = ap.parse_args()

    envs = [int(x) for x in args.envs.split(",")]
    threads = [int(x) for x in args.threads.split(",")]

    m = mujoco.MjModel.from_xml_path(args.model)
    print(f"model      {args.model}")
    print(f"          nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}")
    n_on = apply_hybrid_collision(m)
    print(f"          timestep={m.opt.timestep} nstep/call={args.nstep}")
    print(f"collision  {n_on} colliding geoms (multi-scheme models collapsed to "
          f"hybrid)")
    print(f"host       {platform.processor() or platform.machine()}  "
          f"mujoco {mujoco.__version__}")
    print()
    print("units: env-steps/s (higher is better)")
    print()

    # native
    header = "  ".join(f"{f'{t} thr':>10}" for t in threads)
    print(f"{'native/cpu':>14}  {header}")
    native_best: dict[int, float] = {}
    for n in envs:
        cells = []
        for t in threads:
            try:
                v = bench_native(
                    args.model, n, args.nstep, t, args.warmup, args.repeat,
                    per_env_model=not args.shared_model,
                )
                native_best[n] = max(native_best.get(n, 0.0), v)
                cells.append(_fmt(v))
            except Exception as e:  # noqa: BLE001
                cells.append(f"{type(e).__name__[:9]:>9}")
        print(f"{f'{n} envs':>14}  " + "  ".join(f"{c:>10}" for c in cells))
    print()

    if args.skip_warp:
        return

    for device in ("cuda", "cpu"):
        print(f"{f'warp/{device}':>14}")
        for n in envs:
            try:
                v = bench_warp(
                    args.model, n, args.nstep, device, args.warmup, args.repeat,
                    njmax=args.njmax, nconmax=args.nconmax,
                )
                extra = ""
                if device == "cuda" and n in native_best and native_best[n] > 0:
                    extra = f"   ({v / native_best[n]:.1f}x the best native)"
                print(f"{f'{n} envs':>14}  {_fmt(v):>10}{extra}")
            except Exception as e:  # noqa: BLE001
                print(f"{f'{n} envs':>14}  {type(e).__name__}: {str(e)[:60]}")
        print()


if __name__ == "__main__":
    main()
