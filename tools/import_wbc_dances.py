#!/usr/bin/env python3
"""Import the choreographies rl-wbc-fsm plays into this repository's dance tasks.

    python tools/import_wbc_dances.py --wbc <rl-wbc-fsm checkout> --ref origin/main
    python tools/import_wbc_dances.py --wbc <checkout> --ref origin/main --only brazilian,maze

The sources are read out of git at `--ref` rather than from the working tree, so the
checkout is only read, whatever branch it is on. The committed clips were imported
from rl-wbc-fsm's `origin/main` at 5bc8055 (2026-09-28).

rl-wbc-fsm, the C++ controller this repository's `deploy/fsm` crate replaced, ships
five dances. One of them, the crab dance, is `jumper.dance`'s demonstration clip
already (its playback track is `demo.npz`'s commanded channels, row for row). The
other four are written here, each into its own task's `media/`:

    dance       rl-wbc-fsm source                                   task
    brazilian   dance_choreo4.npz                                   jumper.dance_brazilian
                (rl-wbc-fsm's `dance_baxi`, *baxi* being the pinyin for Brazil)
    dream_wings dance_choreo5.npz                                   jumper.dance_dream_wings
    waist       policy/hexa/dance/dance_waist.reference_clip.json   jumper.dance_waist
                (its `q_cmd` table; the npz it came from is lost)
    maze        policy/hexa/dance/dance_maze.reference_clip.json    jumper.dance_maze
                (its `joint_pos` table; no npz of it exists)

Each is the track the board plays, not a policy's reference: brazilian, dream_wings and
waist run there as open-loop playback of exactly these rows, and maze has nothing
else.

## The commands are taken as the state

The output is the schema `tasks/jumper/common/dance/motion.py` reads, and its joint
channels, `meas_*` included, are **the commanded joint angles, copied**. That is a
decision, and it is the opposite of the one `jumper.dance` made for its own clip:
there the commanded channels are PD targets and the measured ones are imitated
instead. These sources carry no measured channels to choose -- and the joint angles
are exactly what the board sends the servos when it plays them.

**The base pose is not taken from the source.** The two npz sources do carry one,
and it is not a pose the robot takes: the board plays joint angles only, and under
the commanded base pose the jumper's support feet wander over the floor -- 50 mm
sideways and from 14 mm below it to 12 mm above for dream_wings, 95 mm and 36 below
to 35 above for the Brazilian dance -- while relative to each other they barely move (0.07 mm
between rows 50 ms apart, median). The pose is the planner's internal body target:
its x is 0 throughout while the middle feet travel 100 mm fore and aft beneath it. So every source's base pose is
solved from its joints (`solve_base_pose`):

- roll, pitch and height from the plane through the four support feet, which is
  the floor -- they are within 0.46 mm of one plane on every row of every source;
- x, y and yaw by holding planted feet in place from one row to the next.

**The control** is `jumper.dance`'s own clip, whose `meas_*` base pose MuJoCo
produced with the contacts resolved (`--check`). Solved from its joints alone:
height within 2.5 mm on the median row (5.7 mm worst), roll/pitch/yaw within
0.2/2.9/2.8 degrees -- and x off by up to 107 mm over the 148 mm the robot walked
in that simulation. Where the feet slide, how far the body goes is decided by
friction, which the joints do not record; that is the one part of the solution to
hold loosely. Over the four dances here the solved base travels at most 60 mm in
any 10 s episode.

## What else is changed on the way, and why

- **The first row of both npz sources is dropped.** It is the pose before the
  dance -- all twelve support-leg joints step 0.12-0.21 rad into the second row in
  one 2 ms tick -- and a reference that opens with a teleport is scored against
  from the first frame of every episode sampled at its start.
- **Every joint is clamped to the jumper's own range.** A state cannot lie past a
  mechanical stop. Only the Brazilian dance reaches one: its `LF_J3_joint` (the left wrist) is
  commanded to -2.618 against a limit of -2.100 on 16.7% of the track, which
  rl-wbc-fsm's own README records as present in the older recording too. The
  amount clamped is printed per joint, so a new source that needs it says so.

Nothing about the choreography is retimed, smoothed or re-choreographed. The Brazilian dance's
support legs straighten to exactly 0.000 wherever the planner's IK ran out of
reach, which is a 0.30 rad step in one 2 ms tick; it is kept, because it is what
the board plays, and at the 50 Hz the policy sees it is 22 rad/s -- inside what
the servo can do (see the continuity check in `motion.py`).

## Provenance

Each file records the rl-wbc-fsm commit and the sha256 of the source it was cut
from (`source`, `source_sha256`), so a re-import from a moved checkout can be
compared against what is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

#: The source's leg-major channel order under the names rl-wbc-fsm's JSON uses,
#: which are the robot's joint names before the v1.6.1 rename. Copied from the
#: generator's `JOINT_NAMES` table (motion-planner `tools/dance_mujoco.py`), which
#: is also where `motion.py`'s `LEG_JOINTS` comes from: leg `k`, joint `j` of this
#: table is channel `L{k}_j{j}` of the schema.
OLD_JOINT_NAMES: dict[int, tuple[str, ...]] = {
    0: ("LF_shoulder_yaw_joint", "LF_shoulder_pitch_joint", "LF_elbow_joint",
        "LF_wrist_joint", "LF_finger_joint"),
    1: ("LM_hip_joint", "LM_knee_joint", "LM_ankle_joint"),
    2: ("LR_hip_joint", "LR_knee_joint", "LR_ankle_joint"),
    3: ("RR_hip_joint", "RR_knee_joint", "RR_ankle_joint"),
    4: ("RM_hip_joint", "RM_knee_joint", "RM_ankle_joint"),
    5: ("RF_shoulder_yaw_joint", "RF_shoulder_pitch_joint", "RF_elbow_joint",
        "RF_wrist_joint", "RF_finger_joint"),
}

AXES = ("x", "y", "z", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class Source:
    """One dance: where it is in rl-wbc-fsm, and how to read it."""

    name: str
    path: str
    #: None for an npz; for a JSON clip, the table and the key holding its rate.
    table: tuple[str, str] | None = None
    #: Leading rows to drop. See the module docstring.
    skip: int = 0


SOURCES: dict[str, Source] = {
    s.name: s
    for s in (
        Source("brazilian", "dance_choreo4.npz", skip=1),
        Source("dream_wings", "dance_choreo5.npz", skip=1),
        Source("waist", "policy/hexa/dance/dance_waist.reference_clip.json",
               table=("q_cmd", "cmd_hz")),
        Source("maze", "policy/hexa/dance/dance_maze.reference_clip.json",
               table=("joint_pos", "fps")),
    )
}


def _keys() -> list[str]:
    return [f"L{leg}_j{j}" for leg in sorted(OLD_JOINT_NAMES)
            for j in range(len(OLD_JOINT_NAMES[leg]))]


def _read_npz(raw: bytes, name: str) -> tuple[np.ndarray, float]:
    """(joints [T, 22] in channel order, dt). The commanded base pose is not read:
    see the module docstring for why it is not a pose the robot takes."""
    with np.load(io.BytesIO(raw)) as z:
        t = np.asarray(z["time"], dtype=np.float64)
        steps = np.diff(t)
        dt = float(np.median(steps))
        if np.ptp(steps) > 1e-9:
            raise ValueError(f"{name}: time is not uniformly sampled")
        q = np.stack([z[k] for k in _keys()], axis=1).astype(np.float64)
    return q, dt


def _read_json(raw: bytes, name: str, table: str, rate_key: str) -> tuple[np.ndarray, float]:
    d = json.loads(raw)
    by_old = {name: f"L{leg}_j{j}" for leg, names in OLD_JOINT_NAMES.items()
              for j, name in enumerate(names)}
    order = [by_old[n] for n in d["joint_order"]]
    if sorted(order) != sorted(_keys()):
        raise ValueError(f"{name}: joint_order is not the 22 joints")
    a = np.asarray(d[table], dtype=np.float64)
    q = a[:, [order.index(k) for k in _keys()]]
    return q, 1.0 / float(d[rate_key])


class _Robot:
    """The jumper, compiled, and the two things this tool asks of it."""

    def __init__(self) -> None:
        import mujoco

        from tasks.jumper.common.constants import get_spec
        from tasks.jumper.common.dance.motion import LEG_JOINTS, SUPPORT_LEGS

        self.mujoco = mujoco
        self.model = get_spec(None).compile()
        self.data = mujoco.MjData(self.model)
        # Channel k of the source is joint `names[k]` of the entity.
        self.names = [n for leg in sorted(LEG_JOINTS) for n in LEG_JOINTS[leg]]
        jid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
               for n in self.names]
        self.q_adr = self.model.jnt_qposadr[jid]
        self.range = self.model.jnt_range[jid]
        self.sites = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s)
                      for s in SUPPORT_LEGS]
        self.home_z = self._home_z()

    def _home_z(self) -> float:
        from tasks.jumper.common.constants import HOME, STAND_Z

        d = self.data
        d.qpos[:] = 0.0
        d.qpos[2] = STAND_Z
        d.qpos[3] = 1.0
        d.qpos[self.q_adr] = [HOME[n] for n in self.names]
        self.mujoco.mj_kinematics(self.model, d)
        return float(np.mean(d.site_xpos[self.sites][:, 2]))

    def feet_in_base(self, q: np.ndarray) -> np.ndarray:
        """[T, 4, 3]: the support feet, in the base frame, for each row of joints."""
        d = self.data
        out = np.zeros((len(q), len(self.sites), 3))
        for k, row in enumerate(q):
            d.qpos[:] = 0.0
            d.qpos[3] = 1.0
            d.qpos[self.q_adr] = row
            self.mujoco.mj_kinematics(self.model, d)
            out[k] = d.site_xpos[self.sites]
        return out


def _rpy(R: np.ndarray) -> np.ndarray:
    """[T, 3, 3] -> [T, 3] roll, pitch, yaw for R = Rz(yaw) Ry(pitch) Rx(roll), the
    schema's convention (`motion.py::_rpy_to_quat`)."""
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    pitch = -np.arcsin(np.clip(R[:, 2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return np.stack([roll, pitch, yaw], axis=1)


def _tilt(n: np.ndarray) -> tuple[float, float]:
    """(roll, pitch) of a base whose frame sees world-up as `n`, yaw aside.

    For R = Rz(yaw) Ry(pitch) Rx(roll), world-up in the base frame is
    R^T e_z = (-sin pitch, sin roll cos pitch, cos roll cos pitch).
    """
    return float(np.arctan2(n[1], n[2])), float(-np.arcsin(np.clip(n[0], -1.0, 1.0)))


def _rot_rp(roll: float, pitch: float) -> np.ndarray:
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    return np.array([[cp, sp * sr, sp * cr], [0.0, cr, -sr], [-sp, cp * sr, cp * cr]])


def solve_base_pose(robot: _Robot, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The base pose that keeps the support feet on the ground and in place.

    Two halves, because the joints alone fix them differently:

    - **Roll, pitch and height, per row, absolutely.** The four support feet are
      on flat ground, so the plane through them is the floor: its normal is
      world-up as the base sees it, and the height puts the feet where they sit
      when the jumper stands at `HOME`.
    - **x, y and yaw, row to row.** Nothing in a single row says where the robot is
      on the floor or which way it faces, but feet that are planted do not move
      between two rows, so each row is placed where its feet best cover the
      previous row's (a 2-D Kabsch fit). The first row is at the origin, facing +x.

    Returns ([T, 6] x y z roll pitch yaw, [T] how far out of the floor's plane the
    four feet are, in metres).
    """
    feet = robot.feet_in_base(q)
    pose = np.zeros((len(q), 6))
    misfit = np.zeros(len(q))
    world_prev = None
    for k, p in enumerate(feet):
        c = p.mean(axis=0)
        _, sv, vt = np.linalg.svd(p - c)
        n = vt[-1] if np.dot(vt[-1], -c) > 0.0 else -vt[-1]
        misfit[k] = sv[-1] / 2.0
        roll, pitch = _tilt(n)
        level = p @ _rot_rp(roll, pitch).T
        z = robot.home_z - float(level[:, 2].mean())
        if world_prev is None:
            yaw, xy = 0.0, np.zeros(2)
        else:
            a, b = level[:, :2], world_prev[:, :2]
            am, bm = a.mean(axis=0), b.mean(axis=0)
            h = (a - am).T @ (b - bm)
            yaw = float(np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1]))
            cy, sy = np.cos(yaw), np.sin(yaw)
            xy = bm - np.array([[cy, -sy], [sy, cy]]) @ am
        cy, sy = np.cos(yaw), np.sin(yaw)
        world = level.copy()
        world[:, :2] = level[:, :2] @ np.array([[cy, -sy], [sy, cy]]).T + xy
        world[:, 2] += z
        world_prev = world
        pose[k] = (xy[0], xy[1], z, roll, pitch, yaw)
    pose[:, 5] = np.unwrap(pose[:, 5])
    return pose, misfit


def check_against_measured(robot: _Robot, clip: Path) -> str:
    """The control: solve a clip whose base pose is known, and compare.

    `jumper.dance`'s `demo.npz` carries `meas_*`, a trajectory MuJoCo produced with
    the contacts resolved, so its base pose is one the feet really stood under.
    Solved from its joints alone, it has to come back.
    """
    with np.load(clip) as z:
        q = np.stack([z[f"meas_{k}"] for k in _keys()], axis=1).astype(np.float64)
        known = np.stack([z[f"meas_body_{a}"] for a in AXES], axis=1).astype(np.float64)
    q, known = q[::20], known[::20]  # 1 kHz -> 50 Hz is plenty to compare poses
    solved, _ = solve_base_pose(robot, q)
    # The solved trajectory starts at the origin facing +x; the known one does not.
    y0 = known[0, 5]
    c, s = np.cos(y0), np.sin(y0)
    xy = solved[:, :2] @ np.array([[c, -s], [s, c]]).T + known[0, :2]
    err_xy = np.abs(xy - known[:, :2]).max(axis=0)
    err_z = np.abs(solved[:, 2] - known[:, 2])
    err_rpy = np.abs(solved[:, 3:] + [0.0, 0.0, y0] - np.unwrap(known[:, 3:], axis=0))
    return (
        f"control ({clip.name} meas_*, {len(q)} rows): x/y at most "
        f"{err_xy.round(4) * 1000} mm off over a {np.ptp(known[:, 0]) * 1000:.0f} mm "
        f"walk, z median {np.median(err_z) * 1000:.1f} max {err_z.max() * 1000:.1f} mm, "
        f"roll/pitch/yaw at most {np.degrees(err_rpy.max(axis=0)).round(2)} deg"
    )


def _git(wbc: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(wbc), *args], capture_output=True, check=True
    ).stdout


def import_one(src: Source, wbc: Path, commit: str, robot: _Robot) -> Path:
    raw = _git(wbc, "show", f"{commit}:{src.path}")
    name = Path(src.path).name
    q, dt = _read_npz(raw, name) if src.table is None else _read_json(raw, name, *src.table)
    q = q[src.skip:]

    lo, hi = robot.range[:, 0], robot.range[:, 1]
    over = np.maximum(lo - q.min(axis=0), q.max(axis=0) - hi)
    clamped = [(robot.names[i], over[i], float(((q[:, i] < lo[i]) | (q[:, i] > hi[i])).mean()))
               for i in np.nonzero(over > 0.0)[0]]
    q = np.clip(q, lo, hi)

    pose, misfit = solve_base_pose(robot, q)
    how = (f"base pose solved from the feet, which are "
           f"{np.median(misfit) * 1000:.2f} mm out of plane on the median row and "
           f"{misfit.max() * 1000:.2f} mm at worst")

    n = len(q)
    f32 = np.float32
    out: dict[str, np.ndarray] = {
        "time": (np.arange(n) * dt).astype(f32),
        "dt": np.array(dt),
        "phase": np.ones(n, dtype=f32),
    }
    for prefix in ("", "meas_"):
        for i, a in enumerate(AXES):
            out[f"{prefix}body_{a}"] = pose[:, i].astype(f32)
        for k, key in enumerate(_keys()):
            out[f"{prefix}{key}"] = q[:, k].astype(f32)
    digest = hashlib.sha256(raw).hexdigest()
    out["source"] = np.array(f"rl-wbc-fsm@{commit[:12]}:{src.path}")
    out["source_sha256"] = np.array(digest)

    dest = REPO / "tasks" / "jumper" / f"dance_{src.name}" / "media" / f"{src.name}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)

    print(f"{src.name}: {name} @ {commit[:12]} -> {dest.relative_to(REPO)}")
    print(f"  {n} rows at {1.0 / dt:.0f} Hz = {n * dt:.2f} s, first {src.skip} dropped; {how}")
    for name, by, frac in clamped:
        print(f"  clamped {name} by up to {by * 1000:.1f} mrad ({frac * 100:.1f}% of rows)")
    print(f"  {dest.stat().st_size / 1e6:.1f} MB")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wbc", type=Path, required=True, help="an rl-wbc-fsm checkout")
    ap.add_argument("--ref", required=True,
                    help="the rl-wbc-fsm commit to read the sources from, e.g. origin/main")
    ap.add_argument("--only", default=",".join(SOURCES),
                    help=f"comma-separated subset of {', '.join(SOURCES)}")
    ap.add_argument("--check", action="store_true",
                    help="first solve jumper.dance's demo clip, whose base pose is "
                         "known, and print how far the solution is from it")
    args = ap.parse_args(argv)
    names = [n.strip() for n in args.only.split(",") if n.strip()]
    unknown = sorted(set(names) - set(SOURCES))
    if unknown:
        ap.error(f"unknown dance(s): {', '.join(unknown)}")
    commit = _git(args.wbc, "rev-parse", "--verify", f"{args.ref}^{{commit}}").decode().strip()
    robot = _Robot()
    if args.check:
        print(check_against_measured(robot, REPO / "tasks/jumper/dance/media/demo.npz"))
    for name in names:
        import_one(SOURCES[name], args.wbc, commit, robot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
