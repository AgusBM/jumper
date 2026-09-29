#!/usr/bin/env python3
"""Solve the standing pose for a new robot revision, and prove it is holdable.

A model revision moves the joint anchors by a few millimetres, which is enough to
leave the six feet no longer coplanar: the robot then starts every episode on
four or five legs, settles onto the rest over the first few steps, and nothing
reports it. This produces the two numbers that fix that -- `HOME` and `STAND_Z`
in `tasks/jumper/common/constants.py` -- and then checks the result three ways.

## The solve

`HOME` from constants is the **seed**, not the answer: across a revision the
previous pose is usually close, and the smallest change that restores coplanarity
is the one to prefer, because the seed encodes choices this script knows nothing
about (how extended the stance is, how the arms are folded, what margin was
wanted). So one joint per leg is adjusted -- the most distal actuated joint,
whose only real effect at this scale is to raise or lower that foot -- until
every foot's lowest contact point sits at a common height.

The target height is the mean of where the feet already are, so the six
corrections sum to roughly zero and no leg is dragged far from the seed.

## The three checks

**Coplanarity.** How far the six contact points span in z. Read this as a
convergence check and nothing more: the solver drove all six to one height, so a
number near zero says the root finds succeeded, not that the pose is good. The
figure that actually carries information is the **seed spread** printed first --
how far the previous revision's pose had drifted, and therefore how much of a
refinement this is rather than a new stance.

**Limit margin.** Every active joint's distance to its nearest limit, as a
fraction of its range. A home pose is the one configuration every run starts
from, and a joint pinned against a limit there is invisible: MuJoCo clamps it,
the PD pushes into the constraint forever, and no reward or termination looks at
it. The previous model shipped for months with a gripper joint a full radian
outside its range.

**Holdability.** The pose is simulated on a ground plane under the real PD gains
for two seconds. If the base sinks, the pose is not a stance. The reported worst
joint torque is what the servo has to supply to stand still, and it should be a
small fraction of the plateau -- the previous model needed 0.2499 N*m of 1.7464.

Usage:
    python assets/jumper/tools/calibrate_home.py
    python assets/jumper/tools/calibrate_home.py --model assets/jumper/jumper.xml
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
from scipy.optimize import brentq

from tasks.jumper.common.constants import (
    DAMPING,
    EFFORT_LIMIT,
    FEET,
    GAIT_JOINTS,
    HOME,
    STAND_Z,
    STIFFNESS,
)

#: The joint adjusted to change each foot's height: the last actuated joint in
#: the chain. On the 3-DoF legs that is the ankle; on the 5-DoF arms it is the
#: wrist, not the finger -- the finger carries the gripper, not the jaw the arm
#: stands on, and moving it would not lift the foot at all.
HEIGHT_JOINT = {
    "LF": "LF_J3_joint", "RF": "RF_J3_joint",
    "LM": "LM_J2_joint", "RM": "RM_J2_joint",
    "LR": "LR_J2_joint", "RR": "RR_J2_joint",
}

#: How far the solver is allowed to move a joint from the seed, in radians. A
#: revision that needs more than this is not a refinement of the previous pose,
#: and silently returning something far away would hide that.
MAX_ADJUST = 0.35


def apply_pose(
    model: mujoco.MjModel, data: mujoco.MjData, pose: dict[str, float], base_z: float
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, base_z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, value in pose.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)


def lowest_contact_z(model: mujoco.MjModel, data: mujoco.MjData, body: str) -> float:
    """The lowest world-frame point of this body's collision mesh."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    best = np.inf
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != bid:
            continue
        if not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").endswith("_meshcol"):
            continue
        mid = model.geom_dataid[gid]
        adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[adr : adr + num]
        rot = data.geom_xmat[gid].reshape(3, 3)
        best = min(best, float((rot @ verts.T).T[:, 2].min()) + float(data.geom_xpos[gid][2]))
    return best


def solve(model: mujoco.MjModel, seed: dict[str, float]) -> tuple[dict[str, float], float]:
    """Adjust one joint per leg until the six feet are coplanar."""
    data = mujoco.MjData(model)
    pose = dict(seed)

    apply_pose(model, data, pose, STAND_Z)
    start = {foot: lowest_contact_z(model, data, foot) for foot in FEET}
    target = float(np.mean(list(start.values())))
    print(f"seed spread {(max(start.values()) - min(start.values())) * 1000:.3f} mm, "
          f"levelling to z = {target:+.5f}")

    for foot in FEET:
        leg = foot.split("_")[0]
        joint = HEIGHT_JOINT[leg]
        seed_value = seed[joint]

        def error(value: float, joint: str = joint, foot: str = foot) -> float:
            # joint / foot are bound as defaults, not captured: brentq holds on to
            # this closure and a late-binding capture would silently solve every
            # leg against the last one.
            trial = dict(pose, **{joint: value})
            apply_pose(model, data, trial, STAND_Z)
            return lowest_contact_z(model, data, foot) - target

        lo, hi = seed_value - MAX_ADJUST, seed_value + MAX_ADJUST
        if error(lo) * error(hi) > 0:
            raise SystemExit(
                f"{joint}: no solution within {MAX_ADJUST} rad of the seed "
                f"({error(lo) * 1000:+.2f} mm to {error(hi) * 1000:+.2f} mm). "
                "The new model is too far from the old pose for a refinement; "
                "solve the stance from scratch instead."
            )
        pose[joint] = float(brentq(error, lo, hi, xtol=1e-10))
        print(f"  {joint:24s} {seed_value:+.4f} -> {pose[joint]:+.4f}  "
              f"({(pose[joint] - seed_value) * 1000:+7.2f} mrad)")

    apply_pose(model, data, pose, STAND_Z)
    final = {foot: lowest_contact_z(model, data, foot) for foot in FEET}
    stand_z = STAND_Z - min(final.values())
    return pose, stand_z


def report_margins(model: mujoco.MjModel, pose: dict[str, float]) -> float:
    print("\nlimit margins (fraction of range to the nearest limit)")
    worst = 1.0
    for name, value in pose.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[jid]
        margin = min(value - lo, hi - value) / (hi - lo)
        if name in GAIT_JOINTS:
            worst = min(worst, margin)
        flag = "  OUTSIDE" if not (lo <= value <= hi) else ("  tight" if margin < 0.12 else "")
        print(f"  {name:26s} {value:+8.4f}  in [{lo:+.2f}, {hi:+.2f}]  {margin * 100:5.1f}%{flag}")
    print(f"worst margin over the {len(GAIT_JOINTS)} gait joints: {worst * 100:.1f}%")
    return worst


def report_holdability(
    model_path: str, pose: dict[str, float], stand_z: float, seconds: float = 2.0
) -> None:
    """Stand the robot on a plane under the real PD gains and see if it holds."""
    spec = mujoco.MjSpec.from_file(model_path)
    spec.worldbody.add_geom(
        name="_calib_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5.0, 5.0, 0.1],
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    apply_pose(model, data, pose, stand_z)

    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in pose]
    qadr = [model.jnt_qposadr[i] for i in ids]
    vadr = [model.jnt_dofadr[i] for i in ids]
    target = np.array(list(pose.values()))

    peak = np.zeros(len(ids))
    for _ in range(int(seconds / model.opt.timestep)):
        tau = STIFFNESS * (target - data.qpos[qadr]) - DAMPING * data.qvel[vadr]
        tau = np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)
        data.qfrc_applied[vadr] = tau
        mujoco.mj_step(model, data)
        peak = np.maximum(peak, np.abs(tau))

    drift = float(data.qpos[2]) - stand_z
    worst = int(np.argmax(peak))
    print(f"\nheld for {seconds:.0f} s at kp={STIFFNESS} / kd={DAMPING}")
    print(f"  base height drift {drift * 1000:+.2f} mm"
          f"{'   <-- it is not standing' if abs(drift) > 0.005 else ''}")
    print(f"  worst joint torque {peak[worst]:.4f} N*m at {list(pose)[worst]} "
          f"({peak[worst] / EFFORT_LIMIT * 100:.1f}% of the {EFFORT_LIMIT:.4f} N*m plateau)")
    saturated = [n for n, p in zip(pose, peak) if p >= EFFORT_LIMIT - 1e-9]
    print(f"  saturated actuators: {saturated if saturated else 'none'}")
    print(f"  self-contacts at rest: {data.ncon - 6} beyond the six feet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="assets/jumper/jumper.xml")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    pose, stand_z = solve(model, HOME)

    data = mujoco.MjData(model)
    apply_pose(model, data, pose, stand_z)
    heights = [lowest_contact_z(model, data, foot) for foot in FEET]
    print(f"\ncoplanarity {(max(heights) - min(heights)) * 1000:.4f} mm "
          f"at STAND_Z = {stand_z:.5f}")

    report_margins(model, pose)
    report_holdability(args.model, pose, stand_z)

    print("\n--- paste into tasks/jumper/common/constants.py ---")
    print("HOME: dict[str, float] = {")
    for leg in ("LF", "RF", "LM", "RM", "LR", "RR"):
        entries = [f'"{n}": {v:.4f},' for n, v in pose.items() if n.startswith(f"{leg}_")]
        for i in range(0, len(entries), 3):
            print("    " + " ".join(entries[i : i + 3]))
    print("}")
    print(f"\nSTAND_Z = {stand_z:.5f}")


if __name__ == "__main__":
    main()
