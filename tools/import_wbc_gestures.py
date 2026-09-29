#!/usr/bin/env python3
"""Import the one-shot gestures rl-wbc-fsm plays into this repository's gesture tasks.

    python tools/import_wbc_gestures.py --wbc <rl-wbc-fsm checkout> --ref origin/main
    python tools/import_wbc_gestures.py --wbc <checkout> --ref origin/main --only hello,salute

The sources are read out of git at `--ref`, as `import_wbc_dances.py` reads them. The
committed clips were imported from rl-wbc-fsm's `origin/main` at 5bc8055
(2026-09-28).

rl-wbc-fsm puts four body actions on the d-pad, and none of them is a policy: each
is a `[[fsm.state]]` with `execution = "trajectory"`, which replays the `q_cmd`
table of a `reference_clip.json` as PD targets, open loop, and hands back to
locomotion when the table runs out. This repository's controller has no such state
-- it refuses to play a recording with no policy behind it
(`deploy/fsm/src/device.rs`) -- so each gesture becomes what the dances became: a
task that learns to track it, deployed as a mode whose recording starts on entry
and ends by itself.

    gesture  rl-wbc-fsm state (d-pad)  source table                          task
    hello    action_hello  (up)        policy/hexa/action_hello/...json      jumper.gesture_hello
    bow      action_bow    (down)      policy/hexa/action_bow/...json        jumper.gesture_bow
    paw      action_shake  (left)      policy/hexa/action_shake/...json      jumper.gesture_paw
    salute   action_salute (right)     policy/hexa/action_salute/...json     jumper.gesture_salute

`paw` is `action_shake`. rl-wbc-fsm's config calls it a shake and notes beside it
that what it plays for now is offering a paw, which is what the clip does: the
robot sits back, lifts the left front paw forward and holds it out for two
seconds. hello, bow and salute were cut from motion-planner choreography recordings
(`tools/choreo_npz_to_action_clip.py` there); paw has no recording and was
interpolated from hand-placed keyframes (`tools/synth_action_clip.py`).

The table read is `q_cmd`, the track the board plays, and the joint names, the leg
order and the reader are `import_wbc_dances.py`'s -- as is the decision to take the
commands as the state, for the reason given there. Three things are done
differently, each because of something measured on these four tables.

## Played at the board's speed

The board does not play these at the speed they were recorded: each state's
`playback_speed` in `config/hexa/rl-wbc-fsm.toml` scales the clock the table is
read on -- 3 for hello, 2 for bow and salute, 1 for paw. The clip written here is
the motion the robot makes, so its timebase is divided by that factor, read from
the same commit as the table rather than typed a second time here. hello is 15.8 s
as recorded and 5.3 s as performed.

The state's `arm_side` is read too. Only paw sets it, to `"left"`, which is the
side the clip was authored for; `"right"` mirrors the track on the board, and this
tool refuses it rather than import the unmirrored side under the mirrored name.

## Starting and ending at HOME

rl-wbc-fsm ramps into the table's first row and walks away from its last. This
repository's controller ramps to the policy's home pose, `HOME`, starts the
recording once it is there (`starts_on_entry`), and hands back to the walking
policy -- which starts from `HOME` too -- when it ends. The three recorded gestures
neither start nor end at `HOME`: they were cut from choreographies whose rest
stance is a different one (the rear hips 0.44-0.57 rad from `HOME`'s, the body 8 mm
higher), and hello's table opens with the arm already raised. Measured on the tables, the
first row is up to 0.79 rad from `HOME` (salute's right shoulder) and the last up
to 0.57 rad (bow's rear hips). paw's is 0.15 rad, `HOME` having moved with the
v1.6.1 model since its keyframes were placed.

So each clip gets a lead-in from `HOME` to its first row and a lead-out from its
last row back to `HOME`, smoothstep in every joint, each as long as the joint with
the furthest to go needs at `LEAD_SPEED` on average, and then `HOLD_S` standing at
`HOME`, so that the hand-back happens from a still stance the walking policy knows
rather than at the end of a moving one. The table in between is not touched.

## The base pose, from all six feet

`import_wbc_dances.solve_base_pose` puts the floor under the four legs, which is
right for dances, where the two front arms dance in the air. These stand on the
arms as well. At the choreographies' rest stance the paws are on the floor, and
under the four legs' floor they are 7-8 mm into it -- at hello's and salute's ends
and at both of bow's -- and 28 mm into it through salute's crouched opening. That
is a reference no robot can take, and the tracking reward would score every episode
against it.

So the floor is taken to be the plane the jumper would rest on, if it were rigid,
in each row's configuration (`resting_floor`): of the planes through three of the
six feet with no foot below them and the whole-body centre of mass above their
triangle, the one nearest the previous row's. Roll, pitch and height follow from
it as the dances' do from theirs. Where the paws are up and the four legs stand
level it is the dances' plane, and the tool checks that it is on every such row
(`four-leg control` in its output: within 0.01 mm and 0.01 degrees).

**The control**, a one-off run outside the repository: each imported clip played in
MuJoCo on stiff position servos (kp 400, median joint error 0.003-0.009 rad), free
base, a floor, contacts resolved. Against the height and tilt physics produced, the
six-foot floor is within 0.2 mm on the median row (1.2 mm for paw), 2.9 mm on the
worst, and 2.3 degrees of pitch everywhere. The four-leg floor is at most 2.1 mm
further off on hello, bow and paw, and 9.6 mm and 6.6 degrees off on salute, whose
crouch and the stance after its salute rest on the left paw and the rear legs with
the middle legs up. The same run shows bow's two claws pressing into each other from
3.0 to 5.1 s, 0.27 rad short of the table at worst: the gesture is hands pressed
together, and the v1.6.1 claws meet before the table expects them to. It is left as
it is; the task's self-collision term decides how hard the policy presses.

**x, y and yaw are the loose part**, as they are for the dances. The legs do not
stay planted in these tables: the four legs' feet change their distances to each
other by up to 49 mm within hello and bow themselves (their middle hips swing 0.42
and 0.61 rad), and by 32-56 mm across hello's, bow's and salute's lead-ins and
lead-outs, where the rear hips travel to and from the rest stance. On the board they
slide. The placement here is the least-sliding one -- each row fitted to the
previous row's feet that are within `DOWN_TOL` of the floor in both -- and the base
travels 58, 59, 2 and 36 mm over hello, bow, paw and salute. Moving `DOWN_TOL` from
1 mm to 0.1 or 3 mm moves x and y by up to 17 mm and yaw by up to 0.7 degrees, and
height, roll and pitch not at all.

## Provenance

Each file records the rl-wbc-fsm commit and the sha256 of both sources -- the table
and the config the speed came from -- and the speed, the lead times and the hold,
so a re-import can be compared against what is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

# `tools/` is not a package; run as a script, its own directory is on `sys.path`.
import import_wbc_dances as dances
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    # Python 3.10, as `rl/mjrl/app_play.py` handles it: `tomli` is the same parser
    # under its original name, and a dependency there.
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[1]

#: rl-wbc-fsm's config for the hexa, where each state's `playback_speed` and
#: `arm_side` are.
CONFIG = "config/hexa/rl-wbc-fsm.toml"

#: Average joint speed of the lead-in and lead-out, rad/s. Smoothstep peaks at 1.5
#: times its average, so the furthest-travelling joint tops out at 1.2 rad/s --
#: under every gesture's own peak at the 50 Hz control rate (hello 3.3 rad/s, bow
#: 4.6, paw 6.9, salute 7.8), so the transitions are never the fastest thing in a
#: clip.
LEAD_SPEED = 0.8

#: Shortest lead-in or lead-out, seconds, however little there is to cover.
LEAD_MIN_S = 0.4

#: Seconds standing at `HOME` after the lead-out, before the clip ends.
HOLD_S = 0.5

#: A foot this close to the floor, metres, is down, and is held in place from one
#: row to the next. **A choice, not a measurement**: the rigid body rocks, so the
#: feet leave the floor continuously and there is no gap to put it in. It moves x,
#: y and yaw only; the module docstring has by how much.
DOWN_TOL = 0.001

#: The six foot sites, in `constants.LEGS` order.
FEET = ("LF", "RF", "LM", "RM", "LR", "RR")


@dataclass(frozen=True)
class Source:
    """One gesture: which rl-wbc-fsm state plays it, and where its table is."""

    name: str
    state: str

    @property
    def path(self) -> str:
        return f"policy/hexa/{self.state}/{self.state}.reference_clip.json"


SOURCES: dict[str, Source] = {
    s.name: s
    for s in (
        Source("hello", "action_hello"),
        Source("bow", "action_bow"),
        Source("paw", "action_shake"),
        Source("salute", "action_salute"),
    )
}


class _Robot(dances._Robot):
    """The dances' robot, with all six feet and the centre of mass."""

    def __init__(self) -> None:
        super().__init__()
        mj = self.mujoco
        self.feet = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, s) for s in FEET]
        if min(self.feet) < 0:
            raise ValueError(f"the model is missing one of the foot sites {FEET}")

    def feet_and_com(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """([T, 6, 3] feet, [T, 3] whole-body centre of mass), in the base frame."""
        mj, d = self.mujoco, self.data
        feet = np.zeros((len(q), len(self.feet), 3))
        com = np.zeros((len(q), 3))
        for k, row in enumerate(q):
            d.qpos[:] = 0.0
            d.qpos[3] = 1.0
            d.qpos[self.q_adr] = row
            mj.mj_kinematics(self.model, d)
            mj.mj_comPos(self.model, d)
            feet[k] = d.site_xpos[self.feet]
            com[k] = d.subtree_com[1]  # body 1 is the base; its subtree is the robot
        return feet, com


def _inside(c: np.ndarray, a: np.ndarray, b: np.ndarray, p: np.ndarray, n: np.ndarray) -> bool:
    """Whether `c`, projected along `n`, falls inside the triangle a b p. `n` must be
    the triangle's own winding, `(b - a) x (p - a)`, not its flip."""
    for u, v in ((a, b), (b, p), (p, a)):
        if np.dot(np.cross(v - u, c - u), n) < -1e-9:
            return False
    return True


def resting_floor(
    feet: np.ndarray, com: np.ndarray, up: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The floor a rigid jumper in this configuration rests on.

    A plane through three of the feet with no foot below it and the centre of mass
    above its triangle is a floor it can rest on -- and there is more than one: a
    jumper tipped 37 degrees onto its paws and a middle leg is at rest too. The one
    it is on is the one it does not have to tip over to reach, the nearest to `up`,
    the previous row's world-up in the base frame. Returns (world-up in the base
    frame, each foot's height above that floor in metres).
    """
    best, best_dot = None, -np.inf
    for i, j, k in itertools.combinations(range(len(feet)), 3):
        a, b, p = feet[i], feet[j], feet[k]
        n = np.cross(b - a, p - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        if not _inside(com, a, b, p, n):
            continue
        if np.dot(com - a, n) < 0.0:
            n = -n
        h = (feet - a) @ n
        if h.min() < -1e-9 or np.dot(n, up) <= best_dot:
            continue
        best, best_dot = (n, h), float(np.dot(n, up))
    if best is None:
        raise ValueError("no three feet hold the centre of mass up: this row falls over")
    return best


def solve_resting_pose(robot: _Robot, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The base pose under which the jumper rests on the floor and its planted feet
    stay put. Returns ([T, 6] x y z roll pitch yaw, [T, 6] each foot's height above
    the floor, metres)."""
    feet, com = robot.feet_and_com(q)
    pose = np.zeros((len(q), 6))
    height = np.zeros((len(q), len(FEET)))
    world_prev, down_prev = None, None
    # The base frame's own up at HOME, where every foot is down.
    up = np.array([0.0, 0.0, 1.0])
    for k in range(len(q)):
        n, h = resting_floor(feet[k], com[k], up)
        up = n
        down = h < DOWN_TOL
        roll, pitch = dances._tilt(n)
        level = feet[k] @ dances._rot_rp(roll, pitch).T
        # The floor is the lowest foot's level: the feet on it sit where the support
        # feet sit at HOME.
        z = robot.home_z - float(level[:, 2].min())
        planted = down if down_prev is None else down & down_prev
        if world_prev is None:
            yaw, xy = 0.0, np.zeros(2)
        elif planted.sum() >= 2:
            a, b = level[planted, :2], world_prev[planted, :2]
            am, bm = a.mean(axis=0), b.mean(axis=0)
            m = (a - am).T @ (b - bm)
            yaw = float(np.arctan2(m[0, 1] - m[1, 0], m[0, 0] + m[1, 1]))
            cy, sy = np.cos(yaw), np.sin(yaw)
            xy = bm - np.array([[cy, -sy], [sy, cy]]) @ am
        else:
            raise ValueError(f"row {k}: fewer than two feet stay down into it")
        cy, sy = np.cos(yaw), np.sin(yaw)
        world = level.copy()
        world[:, :2] = level[:, :2] @ np.array([[cy, -sy], [sy, cy]]).T + xy
        world[:, 2] += z
        world_prev, down_prev = world, down
        pose[k] = (xy[0], xy[1], z, roll, pitch, yaw)
        height[k] = h
    pose[:, 5] = np.unwrap(pose[:, 5])
    return pose, height


def _smoothstep(u: np.ndarray) -> np.ndarray:
    return u * u * (3.0 - 2.0 * u)


def _lead(a: np.ndarray, b: np.ndarray, dt: float) -> tuple[np.ndarray, float]:
    """Rows from `a` towards `b`, `a` included and `b` not, and the time they take."""
    span = max(LEAD_MIN_S, float(np.abs(b - a).max()) / LEAD_SPEED)
    n = max(1, round(span / dt))
    u = _smoothstep(np.arange(n) / n)[:, None]
    return a + (b - a) * u, n * dt


def _state(config: dict, name: str) -> dict:
    states = [s for s in config["fsm"]["state"] if s.get("name") == name]
    if len(states) != 1:
        raise ValueError(f"{CONFIG} has {len(states)} states named {name!r}")
    return states[0]


def import_one(src: Source, wbc: Path, commit: str, robot: _Robot) -> Path:
    from tasks.jumper.common.constants import HOME

    raw = dances._git(wbc, "show", f"{commit}:{src.path}")
    raw_cfg = dances._git(wbc, "show", f"{commit}:{CONFIG}")
    state = _state(tomllib.loads(raw_cfg.decode()), src.state)
    if state.get("execution") != "trajectory":
        raise ValueError(f"{src.state} is not an open-loop playback on the board")
    side = state.get("arm_side", "left")
    if side != "left":
        raise ValueError(f"{src.state} plays its clip mirrored (arm_side = {side!r})")
    speed = float(state.get("playback_speed", 1.0))

    name = Path(src.path).name
    q, dt = dances._read_json(raw, name, "q_cmd", "cmd_hz")
    dt /= speed

    lo, hi = robot.range[:, 0], robot.range[:, 1]
    over = np.maximum(lo - q.min(axis=0), q.max(axis=0) - hi)
    clamped = [(robot.names[i], over[i]) for i in np.nonzero(over > 0.0)[0]]
    q = np.clip(q, lo, hi)

    home = np.array([HOME[n] for n in robot.names])
    lead_in, t_in = _lead(home, q[0], dt)
    lead_out, t_out = _lead(q[-1], home, dt)
    hold = np.repeat(home[None], round(HOLD_S / dt) + 1, axis=0)
    q = np.concatenate([lead_in, q, lead_out, hold])

    pose, height = solve_resting_pose(robot, q)
    lifted = {f: float((height[:, i] >= DOWN_TOL).mean()) for i, f in enumerate(FEET)}
    # The control: wherever the arms are up and the four legs are down, the floor is
    # the dances' four-leg plane, so both solvers must give the same tilt and height.
    four = (height[:, 2:] < DOWN_TOL).all(axis=1) & (height[:, :2] >= DOWN_TOL).all(axis=1)
    if four.any():
        ref, _ = dances.solve_base_pose(robot, q[four])
        err = np.abs(ref[:, 2:5] - pose[four, 2:5]).max(axis=0)
        control = (f"four-leg control on {four.sum()} rows: z within "
                   f"{err[0] * 1000:.3f} mm, roll/pitch within "
                   f"{np.degrees(err[1:]).round(4)} deg")
    else:
        control = "four-leg control: no row has both arms up"

    n = len(q)
    f32 = np.float32
    out: dict[str, np.ndarray] = {
        "time": (np.arange(n) * dt).astype(f32),
        "dt": np.array(dt),
        "phase": np.ones(n, dtype=f32),
    }
    for prefix in ("", "meas_"):
        for i, a in enumerate(dances.AXES):
            out[f"{prefix}body_{a}"] = pose[:, i].astype(f32)
        for k, key in enumerate(dances._keys()):
            out[f"{prefix}{key}"] = q[:, k].astype(f32)
    out["source"] = np.array(f"rl-wbc-fsm@{commit[:12]}:{src.path}")
    out["source_sha256"] = np.array(hashlib.sha256(raw).hexdigest())
    out["config"] = np.array(f"rl-wbc-fsm@{commit[:12]}:{CONFIG}")
    out["config_sha256"] = np.array(hashlib.sha256(raw_cfg).hexdigest())
    out["playback_speed"] = np.array(speed)
    out["lead_in_s"] = np.array(t_in)
    out["lead_out_s"] = np.array(t_out)
    out["hold_s"] = np.array(HOLD_S)

    dest = REPO / "tasks" / "jumper" / f"gesture_{src.name}" / "media" / f"{src.name}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)

    body = n - len(lead_in) - len(lead_out) - len(hold)
    print(f"{src.name}: {name} @ {commit[:12]} -> {dest.relative_to(REPO)}")
    print(f"  {body} rows at {1.0 / dt:.0f} Hz (played at x{speed:g}) = {body * dt:.2f} s, "
          f"plus {t_in:.2f} s in, {t_out:.2f} s out, {HOLD_S:.2f} s held = {n * dt:.2f} s")
    print(f"  base z {pose[:, 2].min() * 1000:.0f}..{pose[:, 2].max() * 1000:.0f} mm, "
          f"tilt at most {np.degrees(np.abs(pose[:, 3:5]).max()):.1f} deg, "
          f"travels {np.hypot(*np.ptp(pose[:, :2], axis=0)) * 1000:.0f} mm")
    print("  off the floor: " + ", ".join(f"{f} {v * 100:.0f}%" for f, v in lifted.items()))
    print(f"  {control}")
    for jname, by in clamped:
        print(f"  clamped {jname} by up to {by * 1000:.1f} mrad")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wbc", type=Path, required=True, help="an rl-wbc-fsm checkout")
    ap.add_argument("--ref", required=True,
                    help="the rl-wbc-fsm commit to read the sources from, e.g. origin/main")
    ap.add_argument("--only", default=",".join(SOURCES),
                    help=f"comma-separated subset of {', '.join(SOURCES)}")
    args = ap.parse_args(argv)
    names = [n.strip() for n in args.only.split(",") if n.strip()]
    unknown = sorted(set(names) - set(SOURCES))
    if unknown:
        ap.error(f"unknown gesture(s): {', '.join(unknown)}")
    commit = dances._git(args.wbc, "rev-parse", "--verify",
                         f"{args.ref}^{{commit}}").decode().strip()
    robot = _Robot()
    for name in names:
        import_one(SOURCES[name], args.wbc, commit, robot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
