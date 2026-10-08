#!/usr/bin/env python3
"""Step 2: Jumper writes a character on the floor with a brush.

    python tasks/jumper/calligraphy/tools/write.py                   # 无, as large as fits
    python tasks/jumper/calligraphy/tools/write.py --char 无 --plan-only

`jumper.five_foot`'s trained policy walks and stands; this moves the carried arm,
which that task leaves out of the policy's action, and drives the policy's
velocity command. Nothing is trained and nothing in the policy changes.

## The loop

**Every stroke is written by the arm alone, the trunk standing still.** A stroke
cut into pieces with a walk in between shows every seam -- the trunk never stops
exactly where it was sent -- and a trunk that moves while the arm writes makes
the line wander: 无's long middle stroke did both. So the character is made as
large as it can be with every stroke whole in the arm's reach from one place
(`stations.fit_size`): the longest stroke sets the size and the rest is scaled
with it. For each stroke:

    walk     the arm held still, the velocity command steering the trunk to the
             stretch's station: position and heading, then a short settle
    reach    the tip moves, `HOVER` above the floor, to above the stretch's first
             point -- in joint space from the stow on the first stretch, in a
             straight line otherwise
    lower    down onto the floor
    write    along the stroke at `--speed`, the hair sunk into the floor as deep
             as the stroke's width there calls for (`brush.section_width`)
    lift     back up to `HOVER`, then fold the arm to walk to the next stroke

Every target is solved against the trunk **as it is**, each control step
(`arm.py`): the trunk is never exactly where it was sent and drifts while the arm
writes, and solving against the measured pose is what keeps that out of the ink.

## What it writes

`logs/calligraphy/<u65e0>/<time>/`: `stretches.json` (the cut, as written --
replanned stretches included), `log.npz` (one row per control step: phase, stroke,
target and measured tip, whether the hair is in the floor and the width it leaves
there, the trunk's pose, the IK
residual, the command; and the full `qpos`), `model.mjb` (the compiled model that
`qpos` belongs to, for `render.py`) and
`topview.png`, the plan beside where the hair actually went into the floor.

The policy runs on `native:cpu` by default -- one environment, so a GPU buys
nothing here.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
CACHE = REPO / "logs" / "calligraphy" / "cache"

# Phases, as logged.
WALK, SETTLE, REACH, LOWER, WRITE, LIFT, FOLD, DONE = range(8)
PHASES = ("walk", "settle", "reach", "lower", "write", "lift", "fold", "done")

#: Walking: P gains and limits on the velocity command.
K_POS, V_MAX = 1.5, 0.12          # 1/s, m/s
K_YAW, W_MAX = 2.0, 0.6           # 1/s, rad/s
#: Below this the policy stands rather than steps; a correction is sent at least this fast.
V_MIN = 0.08
#: After a walk, the stroke has to be this far inside the band from where the trunk
#: actually stopped. The walk gets within 5 mm, but stopping, the trunk sways back
#: 5-7 mm with the last step (run 8, 2026-10-08), and walking again repeats it. So
#: the rest is closed by nudges -- a short push of the velocity command towards the
#: station, then a settle and a new measurement -- up to NUDGES times; only then is
#: the stroke split (a seam, reported).
RUN_MARGIN = 0.003                # m
NUDGES = 5
#: A stroke that is out of reach by a little after unfolding is moved, whole, by
#: up to this much -- the smallest shift that fits -- instead of being cut.
SHIFT_MAX = 0.005                 # m
SHIFT_STEP = 0.001                # m
#: The one way the trunk moves once the arm is out -- and only before a stroke
#: starts, never during it: the policy's body-pose command turns the trunk over
#: its planted feet ("twist", +-30 deg standing). Unfolding leaves the trunk
#: settled 5-10 mm from where it stood, and the arm out, five_foot does not walk;
#: but a 2-3 deg twist swings the whole band a centimetre at the stroke's distance.
#: So after unfolding, the smallest twist that brings the whole stroke into reach
#: is commanded, held through the stroke, and released before the arm folds.
TWIST_MAX = math.radians(15.0)
TWIST_STEP = math.radians(1.0)
TWIST_SETTLE_S = 1.2
#: Off: the trunk turned more than asked (+6 deg -> +10.3) and, twisted with the
#: arm out, stumbled in the middle of a stroke -- 73 mm and 13 deg while 无's third
#: stroke was written (run 14). Without it the trunk held within 12 mm and 2 deg
#: through every stroke of run 15 (2026-10-08, native:cpu).
TWIST_TRIES = 0
#: Folding, nudging and unfolding again this many times before a seam.
UNFOLD_TRIES = 3
#: When less than this many samples are in reach, the stroke is not split there:
#: it gets a new station, up to MAX_REPLANS times per stroke.
MIN_PIECE = 15
MAX_REPLANS = 2
NUDGE_V = 0.07                    # m/s commanded
NUDGE_RATE = 0.035                # m/s the trunk actually moves under it (assumed)
NUDGE_T = (0.12, 0.8)             # s, the shortest and longest push
#: While writing, this many consecutive IK misses end the stretch where it is.
MISS_STEPS = 3
#: The arm comes to rest for this long after unfolding or folding; going straight
#: on, the swing carried the tip 15 mm below the line it was sent along.
ARM_SETTLE_S = 1.0
#: Warn when no fold path keeps the arm this far off the floor.
FOLD_CLEARANCE = 0.02
#: Arrived: within this of the station, and of its heading.
POS_TOL, YAW_TOL = 0.006, math.radians(2.0)
WALK_TIMEOUT = 12.0               # s
SETTLE_S = 0.6
#: Unfolding takes this long. At 1.5 s the swing staggered the robot: the trunk
#: slid +-20 mm, turned 15 deg and rose to 136 mm, and wrote 无's second stroke
#: from there (run 9, 2026-10-08).
UNFOLD_S = 3.0
REACH_SPEED = 0.06                # m/s, hover moves between stretches
LOWER_S, LIFT_S = 0.35, 0.30
#: Unfolding and folding go through a point this high above the floor, and the
#: tip travels between it and HOVER in a straight line. Interpolated in joint space
#: all the way, the tip swept the floor on 35-56 of ~100 steps of each unfold and
#: up to 32 of 60 of each fold -- ink where no stroke is.
HIGH = 0.06                       # m
#: A solve that misses by more than this is retried from the reach map's seed.
IK_RETRY = 0.002                  # m
#: Tracking. The arm's PD lags a moving target -- 5.7 mm behind at 4 cm/s, 2.1 mm
#: across, measured on the first full run without either term (2026-10-08,
#: native:cpu, 无 at 0.30 m: median error 6.8 mm). So the target is commanded
#: LEAD_S ahead along the stroke, and what is left is integrated out in xy.
LEAD_S = 0.12                     # s
KI = 4.0                          # 1/s
CORR_MAX = 0.015                  # m
#: Pressing. The hair is a cone that may go into the floor (`brush.py`), and the
#: ink is its section there. The depth that gives a width depends on how the brush
#: leans, so the depth starts from a guess -- vertical depth = width, right for a
#: brush leaning ~40 deg -- and an integrator on the measured section moves the
#: tip up or down until the width is `press * width_full`.
KW = 3.0                          # m of depth per m of width error per s
ZCORR = (-0.006, 0.006)           # m, how far the integrator may move the tip


def _shift_to_fit(s, c, base, yaw, rm, dist):
    """The smallest shift of stroke `s` (within SHIFT_MAX) that brings samples
    [c.start, c.end] into reach from the trunk at `base`, `yaw`; or None."""
    from tasks.jumper.calligraphy import hanzi, stations

    r = np.arange(-SHIFT_MAX, SHIFT_MAX + 1e-9, SHIFT_STEP)
    offsets = sorted(((dx, dy) for dx in r for dy in r if math.hypot(dx, dy) <= SHIFT_MAX),
                     key=lambda d: math.hypot(*d))
    for d in offsets:
        moved = hanzi.Stroke(s.index, s.xy + np.asarray(d), s.press)
        if stations.reachable_until(moved, c.start, c.end, base, yaw, rm, RUN_MARGIN,
                                    dist) == c.end:
            return np.asarray(d), moved
    return None


def _yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--char", default="无")
    ap.add_argument("--size", type=float, default=None,
                    help="em square, metres; default: as large as fits stroke by stroke")
    ap.add_argument("--origin", type=float, nargs=2, default=(0.45, 0.0),
                    help="world xy of the character's centre; the robot starts at 0,0 "
                         "facing +x")
    ap.add_argument("--speed", type=float, default=0.03, help="writing speed, m/s")
    ap.add_argument("--width", type=float, default=0.065,
                    help="the stroke's full width, as a fraction of the character's size")
    ap.add_argument("--margin", type=float, default=0.010,
                    help="how far inside the reach band every written point must be, m")
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO / "tasks/jumper/five_foot/out/example/model_86600.pt")
    ap.add_argument("--plan-only", action="store_true",
                    help="cut the plan into stretches, draw them, and stop")
    ap.add_argument("--max-stretches", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import torch

    from tasks.jumper.calligraphy import arm as armmod
    from tasks.jumper.calligraphy import brush, hanzi, sim, stations
    from tasks.jumper.five_foot.claw import ARM_JOINTS, LF_GRASP

    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    out = args.out or REPO / "logs" / "calligraphy" / f"u{ord(args.char):04x}" / stamp
    out.mkdir(parents=True, exist_ok=True)

    # ── The environment: five_foot's replay config, plus the brush ───────────
    sm = sim.build(1, args.checkpoint)
    env, wrapped, policy, robot, jids = sm.env, sm.wrapped, sm.policy, sm.robot, sm.joint_ids
    steer, pose_cmd, dt = sm.steer, sm.pose, sm.dt

    # ── Where the arm can write, and the cut ─────────────────────────────────
    arm = armmod.Arm(env.sim.mj_model)
    standing = sim.standing_qpos(env.sim.mj_model)
    t0 = time.time()
    rm = armmod.cached_reach_map(arm, standing, CACHE)
    print(f"[write] reach band: {rm.ok.sum() * rm.cell**2 * 1e4:.0f} cm^2 "
          f"({time.time() - t0:.1f} s)")
    stable = CACHE / f"stable_{rm.key}.npz"
    if stable.exists():
        rm.ok &= np.load(stable)["ok"]
        print(f"[write] where the policy also stands still (tools/stability.py): "
              f"{rm.ok.sum() * rm.cell**2 * 1e4:.0f} cm^2")
    else:
        print(f"[write] no stability map ({stable.name}); run tools/stability.py -- "
              "without it the band includes arm poses the trunk will not hold still under")
    if args.size is None:
        size, plan, cut = stations.fit_size(
            lambda z: hanzi.plan(args.char, z, tuple(args.origin)), rm, args.margin)
        print(f"[write] the largest {args.char} with every stroke whole: {size * 100:.1f} cm")
    else:
        plan = hanzi.plan(args.char, args.size, tuple(args.origin))
        cut = stations.whole_strokes(plan, rm, args.margin)
        if cut is None:
            print(f"[write] at {args.size * 100:.0f} cm some stroke does not fit whole; "
                  "cutting it into stretches")
            cut = stations.plan_stretches(plan, rm, margin=args.margin)
    width_full = args.width * plan.size
    if args.max_stretches:
        cut = cut[: args.max_stretches]
    (out / "stretches.json").write_text(json.dumps([
        {"stroke": c.stroke, "start": c.start, "end": c.end,
         "base": np.round(c.base, 4).tolist(), "yaw_deg": round(math.degrees(c.yaw), 1),
         "margin_mm": round(c.margin * 1000, 1)} for c in cut
    ], indent=1))
    (out / "plan.json").write_text(json.dumps(plan.to_json()))
    print(f"[write] {plan.character}: {len(plan.strokes)} strokes -> {len(cut)} stretches")
    for k, c in enumerate(cut):
        print(f"  {k:2d} stroke {c.stroke + 1} [{c.start:3d}, {c.end:3d}] at "
              f"({c.base[0]:+.3f}, {c.base[1]:+.3f}) yaw {math.degrees(c.yaw):+5.1f}  "
              f"margin {c.margin * 1000:.0f} mm")
    np.savez(out / "reach.npz", ok=rm.ok, x0=rm.x0, y0=rm.y0, cell=rm.cell)
    if args.plan_only:
        _topview(out, plan, cut, None, rm)
        env.close()
        return 0

    # ── The loop ─────────────────────────────────────────────────────────────
    stow = np.array([LF_GRASP[j] for j in ARM_JOINTS])
    q_cmd = stow.copy()
    rows: list[list[float]] = []
    #: The whole simulation state each step, for replaying it into a renderer.
    qpos_rows: list[np.ndarray] = []
    obs = wrapped.get_observations()
    sim_t = 0.0

    def state():
        qpos = env.sim.data.qpos[0].cpu().numpy().astype(np.float64)
        return qpos, qpos[0:3].copy(), _yaw(qpos[3:7])

    def step(phase, k, sample, target, residual):
        nonlocal obs, sim_t
        hold = torch.tensor([[*q_cmd, brush.FINGER_HOLD]], dtype=torch.float32)
        robot.set_joint_position_target(hold, joint_ids=jids)
        with torch.inference_mode():
            obs, _, dones, _ = wrapped.step(policy(obs))
        sim_t += dt
        if bool(dones[0]):
            raise RuntimeError(f"the episode ended at t={sim_t:.2f}s in {PHASES[phase]} "
                               f"(stretch {k}): the robot fell or was reset")
        qpos, base, yaw = state()
        arm.set_state(qpos)
        tip = arm.d.site_xpos[arm.tip]
        base_site = arm.d.site_xpos[arm.base]
        tip_now[:] = tip
        width_now[0] = brush.section_width(tip, base_site)
        ink_now[:] = brush.ink_point(tip, base_site)
        tgt = target if target is not None else (np.nan, np.nan, np.nan)
        qpos_rows.append(qpos.astype(np.float32))
        rows.append([sim_t, phase, done[k].stroke if 0 <= k < len(done) else -1, k, sample,
                     *tgt, *tip, *ink_now[:2], float(width_now[0] > 0), width_now[0],
                     -min(tip[2], 0.0),
                     *base, yaw, residual, *steer.values])

    def solve(target):
        """IK from the last command; from the reach map's seed if that misses.

        Continuing from the last command keeps the arm on one branch, and is what
        almost always runs. When the arm was left somewhere the target cannot be
        reached from smoothly -- one unfold in the second run did, and the arm
        then wrote 12 cm off the stroke for its whole length -- the seed of the
        target's cell is a known-good branch.
        """
        nonlocal q_cmd
        qpos, base, yaw = state()
        arm.set_state(qpos)
        target = np.asarray(target)
        q, err = arm.ik(target, q_cmd)
        if err > IK_RETRY:
            q2, err2 = arm.ik(target, seed_for(target[:2], base, yaw))
            if err2 < err:
                q, err = q2, err2
        q_cmd = q
        return err

    def seed_for(xy_world, base, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        d = np.asarray(xy_world) - base[:2]
        local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        i, j = rm.index(local)
        i = int(np.clip(i, 0, rm.ok.shape[0] - 1))
        j = int(np.clip(j, 0, rm.ok.shape[1] - 1))
        return rm.seed[i, j]

    tip_now = np.zeros(3)
    width_now = np.zeros(1)
    ink_now = np.zeros(3)
    corr = np.zeros(2)
    zcorr = np.zeros(1)
    dist_map = rm.margin()

    def track(true_xy, lead_xy, z, press=None):
        """Solve for `lead_xy` plus the integrated correction; log `true_xy`.
        With `press`, the height is corrected to hold the ink's width."""
        nonlocal corr
        # The ink, not the apex, follows the stroke: aim the apex off by how far the
        # ink sits from it now, and integrate out what that misses.
        corr = np.clip(corr + KI * dt * (true_xy - ink_now[:2]), -CORR_MAX, CORR_MAX)
        lead_xy = lead_xy + (tip_now[:2] - ink_now[:2])
        if press is not None:
            zcorr[0] = np.clip(zcorr[0] + KW * dt * (width_now[0] - press * width_full),
                               *ZCORR)
            z = max(z + zcorr[0], -armmod.DEPTH_MAX)
        return solve(np.array([*(lead_xy + corr), z]))

    def unfold_to(q_goal, k, sample, target):
        """Stow -> `q_goal` (or back) along the floor-safe joint path."""
        nonlocal q_cmd
        qpos, _, _ = state()
        arm.set_state(qpos)
        legs, low = arm.fold_path(q_cmd.copy(), q_goal)
        if low < FOLD_CLEARANCE:
            print(f"\n[write] stretch {k}: the best fold path passes {low * 1000:.0f} mm "
                  "above the floor")
        for leg in legs:
            q_from = q_cmd.copy()
            n = int(UNFOLD_S / len(legs) / dt) + 1
            for i in range(1, n + 1):
                f = i / n
                f = f * f * (3 - 2 * f)
                q_cmd = q_from + f * (leg - q_from)
                step(FOLD if target is None else REACH, k, sample, target, np.nan)
        for _ in range(int(ARM_SETTLE_S / dt)):
            step(FOLD if target is None else REACH, k, sample, target, np.nan)

    def settle(k, sample):
        steer.values = [0.0, 0.0, 0.0]
        for _ in range(int(SETTLE_S / dt)):
            step(SETTLE, k, sample, None, np.nan)

    def line(frm, to, speed, phase, k, sample):
        n = max(1, int(np.linalg.norm(to - frm) / speed / dt))
        for i in range(1, n + 1):
            tgt = frm + (to - frm) * (i / n)
            step(phase, k, sample, tgt, solve(tgt))

    todo = deque(cut)
    done: list = []
    replans = dict.fromkeys(range(len(plan.strokes)), 0)
    shifts: dict[int, np.ndarray] = {}

    def hold_tip(k, sample, seconds):
        """Let the trunk turn under the arm while the tip stays where it is in the
        world: with the joints held instead, the tip rides the turning trunk and
        dipped into the floor (ink on 10 settle steps of run 13)."""
        qpos, _, _ = state()
        arm.set_state(qpos)
        here = arm.tip_pos(q_cmd).copy()
        for _ in range(int(seconds / dt)):
            step(SETTLE, k, sample, here, solve(here))

    def untwist(k, sample):
        if pose_cmd.values[2] != 0.0:
            pose_cmd.values[2] = 0.0
            hold_tip(k, sample, TWIST_SETTLE_S)
    folded = True
    while todo:
        c = todo.popleft()
        k = len(done)
        done.append(c)
        s = plan.strokes[c.stroke]
        p0 = s.xy[c.start]
        above = np.array([*p0, armmod.HOVER])

        def walk_to(c=c, k=k):
            """Walk, folded, to the station; then nudge until the stroke fits."""
            t_start = sim_t
            while True:
                _, base, yaw = state()
                e = c.base - base[:2]
                eyaw = _wrap(c.yaw - yaw)
                dist = float(np.linalg.norm(e))
                if (dist < POS_TOL and abs(eyaw) < YAW_TOL) or sim_t - t_start > WALK_TIMEOUT:
                    break
                v = K_POS * e
                n = np.linalg.norm(v)
                if n > V_MAX:
                    v *= V_MAX / n
                elif n < V_MIN:
                    v *= V_MIN / max(n, 1e-9)
                if dist < POS_TOL:
                    v[:] = 0.0
                cy, sy = math.cos(yaw), math.sin(yaw)
                steer.values = [float(cy * v[0] + sy * v[1]), float(-sy * v[0] + cy * v[1]),
                                float(np.clip(K_YAW * eyaw, -W_MAX, W_MAX))]
                step(WALK, k, c.start, None, np.nan)
            settle(k, c.start)
            nudge_to(c, k)

        def nudge_to(c, k, s=s, shift=(0.0, 0.0, 0.0)):
            """Nudge until the stroke fits from where the trunk will be once the arm
            is out: here, plus the `shift` that unfolding was seen to cause."""
            sx, sy_, syaw = shift
            _, base, yaw = state()
            for _ in range(NUDGES):
                ahead = base[:2] + (sx, sy_)
                if stations.reachable_until(s, c.start, c.end, ahead, yaw + syaw, rm,
                                            RUN_MARGIN, dist_map) == c.end:
                    return
                e = c.base - ahead
                eyaw = _wrap(c.yaw - yaw - syaw)
                t_push = float(np.clip(np.linalg.norm(e) / NUDGE_RATE, *NUDGE_T))
                cy, sy = math.cos(yaw), math.sin(yaw)
                v = NUDGE_V * e / max(np.linalg.norm(e), 1e-9)
                steer.values = [float(cy * v[0] + sy * v[1]), float(-sy * v[0] + cy * v[1]),
                                float(np.clip(eyaw / t_push, -W_MAX, W_MAX))]
                for _ in range(int(t_push / dt)):
                    step(WALK, k, c.start, None, np.nan)
                settle(k, c.start)
                _, base, yaw = state()

        # Position, unfold, and check from where the trunk is once the arm is out:
        # unfolding moves the trunk too, so the check before it is not the one
        # that counts. If the stroke no longer fits, fold, nudge and unfold again.
        shift = np.zeros(3)
        for cycle in range(UNFOLD_TRIES):
            if folded:
                if cycle == 0:
                    walk_to()
                else:
                    nudge_to(c, k, shift=tuple(shift))
                qpos, base, yaw = state()
                before = np.array([base[0], base[1], yaw])
                arm.set_state(qpos)
                q_goal, _ = arm.ik(above, seed_for(p0, base, yaw))
                # As high as the arm reaches over this point, up to HIGH: near the
                # edge of the band it does not reach 60 mm up (15.6 mm short at
                # LF_J3's limit, at 无's fourth stroke).
                high = above
                for z in (HIGH, 0.045, 0.03):
                    q_up, err = arm.ik(np.array([*p0, z]), q_goal)
                    if err < IK_RETRY:
                        q_goal, high = q_up, np.array([*p0, z])
                        break
                unfold_to(q_goal, k, c.start, high)
                folded = False
                start = high
                # Unfolding to the same pose shoves the trunk the same way each time:
                # remember by how much, and aim the next nudge that much short.
                _, b2, y2 = state()
                moved = np.array([b2[0] - before[0], b2[1] - before[1], _wrap(y2 - before[2])])
                shift = moved if cycle == 0 else 0.5 * (shift + moved)
            else:
                qpos, _, _ = state()
                arm.set_state(qpos)
                start = arm.tip_pos(q_cmd)
            _, base, yaw = state()
            last = stations.reachable_until(s, c.start, c.end, base, yaw, rm, RUN_MARGIN,
                                            dist_map)
            for _ in range(TWIST_TRIES):
                if last == c.end:
                    break
                fits = [d for d in np.arange(-TWIST_MAX, TWIST_MAX + 1e-9, TWIST_STEP)
                        if stations.reachable_until(s, c.start, c.end, base, yaw + d, rm,
                                                    RUN_MARGIN, dist_map) == c.end]
                if not fits:
                    break
                d = min(fits, key=abs)
                # Positive twist turns the trunk positive about z: +2, +6 and -3 deg
                # turned it +3.6, +5.4 and -2.9 (run 13).
                pose_cmd.values[2] = float(np.clip(pose_cmd.values[2] + d,
                                                   -2 * TWIST_MAX, 2 * TWIST_MAX))
                hold_tip(k, c.start, TWIST_SETTLE_S)
                _, base, yaw_after = state()
                turned = _wrap(yaw_after - yaw)
                print(f"\n[write] stretch {k}: twisted the trunk {math.degrees(d):+.0f} deg "
                      f"to bring the stroke into reach (turned {math.degrees(turned):+.1f})")
                yaw = yaw_after
                last = stations.reachable_until(s, c.start, c.end, base, yaw, rm, RUN_MARGIN,
                                                dist_map)
            if last < c.end:
                # A few millimetres short: move the whole stroke rather than cut it.
                # A stroke 5 mm off its place in a 10 cm character is a slip of the
                # hand; a seam in the middle of it is a mistake.
                shifted = _shift_to_fit(s, c, base, yaw, rm, dist_map)
                if shifted is not None:
                    delta, s = shifted
                    shifts[k] = delta
                    p0 = s.xy[c.start]
                    above = np.array([*p0, armmod.HOVER])
                    last = c.end
                    print(f"\n[write] stretch {k}: stroke {c.stroke + 1} moved "
                          f"({delta[0] * 1000:+.0f}, {delta[1] * 1000:+.0f}) mm to fit the reach")
            if last == c.end or cycle == UNFOLD_TRIES - 1:
                break
            print(f"\n[write] stretch {k}: the stroke left the reach as the arm unfolded; "
                  f"folding to reposition ({cycle + 1}/{UNFOLD_TRIES - 1})")
            line(start, np.array([*start[:2], HIGH]), REACH_SPEED, FOLD, k, c.start)
            untwist(k, c.start)
            unfold_to(stow, k, c.start, None)
            folded = True
        if last < c.end and last - c.start < MIN_PIECE and replans[c.stroke] < MAX_REPLANS:
            # Hardly any of the stroke is in reach from here: writing a sliver and
            # walking for the rest would make a seam for nothing (run 10 wrote 无's
            # third stroke as six pieces, four of them one sample long). Fold, and
            # go to a new station for the whole of what is left.
            replans[c.stroke] += 1
            rest = stations.cut_stroke(s, c.start, rm, args.margin, stations.YAWS, c.yaw,
                                       dist_map, tail=False)
            print(f"\n[write] stretch {k}: out of reach from here; a new station for the "
                  f"rest of stroke {c.stroke + 1} ({replans[c.stroke]}/{MAX_REPLANS})")
            c.end = c.start
            todo.extendleft(reversed(rest))
            qpos, _, _ = state()
            arm.set_state(qpos)
            here = arm.tip_pos(q_cmd)
            line(here, np.array([*here[:2], HIGH]), REACH_SPEED, FOLD, k, c.start)
            untwist(k, c.start)
            unfold_to(stow, k, c.start, None)
            folded = True
            continue
        if last < c.end:
            # Still out of reach: write what is reachable from here, cut the rest
            # again from where this stops. A seam; reported.
            if last <= c.start:
                last = c.start + 1
            rest = stations.cut_stroke(s, last, rm, args.margin, stations.YAWS, c.yaw, dist_map)
            print(f"\n[write] stretch {k}: SEAM -- from where the trunk stopped it reaches "
                  f"sample {last} of [{c.start}, {c.end}]; replanned the rest as {len(rest)}")
            c.end = last
            todo.extendleft(reversed(rest))

        line(start, above, REACH_SPEED, REACH, k, c.start)

        # lower
        z0 = -s.press[c.start] * width_full
        n = int(LOWER_S / dt)
        corr[:] = 0.0
        zcorr[:] = 0.0
        for i in range(1, n + 1):
            tgt = np.array([*p0, armmod.HOVER + (z0 - armmod.HOVER) * (i / n)])
            err = track(p0, p0, tgt[2])
            step(LOWER, k, c.start, tgt, err)

        # write: along the samples at --speed
        per_step = args.speed * dt / plan.ds  # samples per control step
        lead = args.speed * LEAD_S / plan.ds    # samples

        def at(u, c=c, s=s):
            u = min(float(c.end), u)
            i0 = int(u)
            f = u - i0
            i1 = min(i0 + 1, len(s.xy) - 1)
            return (1 - f) * s.xy[i0] + f * s.xy[i1], (1 - f) * s.press[i0] + f * s.press[i1]

        u = float(c.start)
        misses = 0
        while u < c.end:
            u = min(float(c.end), u + per_step)
            xy, press = at(u)
            tgt = np.array([*xy, -press * width_full])
            err = track(xy, at(u + lead)[0], tgt[2], press)
            step(WRITE, k, int(u), tgt, err)
            misses = misses + 1 if err > IK_RETRY else 0
            if misses >= MISS_STEPS and u < c.end:
                # The trunk drifted while the arm wrote and the stroke ran out of
                # reach: stop here and cut the rest again.
                stop = max(c.start + 1, int(u) - int(lead))
                rest = stations.cut_stroke(s, stop, rm, args.margin, stations.YAWS, c.yaw,
                                           dist_map)
                print(f"\n[write] stretch {k}: SEAM -- out of reach at sample {stop} "
                      f"(trunk drifted); replanned the rest as {len(rest)}")
                c.end = stop
                todo.extendleft(reversed(rest))
                break

        # lift
        pe = s.xy[c.end]
        ze = -s.press[c.end] * width_full + zcorr[0]
        n = int(LIFT_S / dt)
        for i in range(1, n + 1):
            tgt = np.array([*pe, ze + (armmod.HOVER - ze) * (i / n)])
            err = track(pe, pe, tgt[2])
            step(LIFT, k, c.end, tgt, err)

        # fold, unless the next stretch can be written from where the trunk is
        nxt = todo[0] if todo else None
        if nxt is not None:
            _, base, yaw = state()
            if stations.reachable_until(plan.strokes[nxt.stroke], nxt.start, nxt.end, base,
                                        yaw, rm, args.margin / 2, dist_map) == nxt.end:
                nxt.base, nxt.yaw = base[:2].copy(), yaw
                nxt = None
        if nxt is not None:
            line(np.array([*pe, armmod.HOVER]), np.array([*pe, HIGH]), REACH_SPEED, FOLD, k,
                 c.end)
            untwist(k, c.end)
            unfold_to(stow, k, c.end, None)
            folded = True
        print(f"[write] stretch {k + 1} done at t={sim_t:.1f}s ({len(todo)} to go)", flush=True)

    for _ in range(int(1.0 / dt)):
        step(DONE, len(done) - 1, -1, None, np.nan)
    cut = done
    (out / "stretches.json").write_text(json.dumps([
        {"stroke": c.stroke, "start": c.start, "end": c.end,
         "base": np.round(c.base, 4).tolist(), "yaw_deg": round(math.degrees(c.yaw), 1),
         "margin_mm": round(c.margin * 1000, 1),
         "shift_mm": np.round(shifts.get(k, np.zeros(2)) * 1000, 1).tolist()}
        for k, c in enumerate(cut)
    ], indent=1))

    cols = ["t", "phase", "stroke", "stretch", "sample", "tx", "ty", "tz", "px", "py", "pz",
            "ix", "iy", "contact", "width", "depth", "bx", "by", "bz", "byaw", "ik_residual",
            "cmd_vx", "cmd_vy", "cmd_wz"]
    log = np.asarray(rows, dtype=np.float64)
    np.savez(out / "log.npz", log=log, columns=np.array(cols), phases=np.array(PHASES),
             qpos=np.asarray(qpos_rows), dt=dt)
    # The compiled model the log's qpos belongs to: `render.py` replays it in plain
    # MuJoCo, without mjlab or torch (OSMesa, the CPU renderer, crashes the process
    # when torch is loaded beside it).
    import mujoco

    mujoco.mj_saveModel(env.sim.mj_model, str(out / "model.mjb"))
    _report(log, cols)
    _topview(out, plan, cut, log, rm, cols)
    print(f"[write] wrote {out}")
    env.close()
    return 0


def _report(log: np.ndarray, cols: list[str]) -> None:
    c = {n: i for i, n in enumerate(cols)}
    w = log[log[:, c["phase"]] == WRITE]
    err = np.linalg.norm(w[:, [c["ix"], c["iy"]]] - w[:, [c["tx"], c["ty"]]], axis=1)
    touching = w[:, c["contact"]] > 0
    print(f"[write] {log[-1, c['t']]:.1f} s simulated, {len(w)} writing steps")
    print(f"  ink centre to stroke while writing: median {np.median(err) * 1000:.1f} mm, "
          f"p95 {np.percentile(err, 95) * 1000:.1f} mm, max {err.max() * 1000:.1f} mm")
    print(f"  hair in the floor while writing: {touching.mean() * 100:.0f}% of steps")
    if touching.any():
        wd = w[touching, c["width"]] * 1000
        print(f"  ink width while writing: median {np.median(wd):.1f} mm, "
              f"p5-p95 {np.percentile(wd, 5):.1f}-{np.percentile(wd, 95):.1f} mm, "
              f"max depth {w[:, c['depth']].max() * 1000:.1f} mm")
    off = log[~np.isin(log[:, c["phase"]], (WRITE, LOWER, LIFT))]
    print(f"  hair in the floor outside lower/write/lift: {(off[:, c['contact']] > 0).sum()} "
          "steps")
    for k in np.unique(w[:, c["stretch"]]).astype(int):
        b = w[w[:, c["stretch"]] == k]
        drift = np.linalg.norm(b[-1, [c["bx"], c["by"]]] - b[0, [c["bx"], c["by"]]])
        turn = math.degrees(_wrap(b[-1, c["byaw"]] - b[0, c["byaw"]]))
        print(f"  stroke {int(b[0, c['stroke']]) + 1}: trunk moved {drift * 1000:.1f} mm and "
              f"turned {turn:+.1f} deg while the arm wrote")
    print(f"  IK residual while writing: max {np.nanmax(w[:, c['ik_residual']]) * 1000:.1f} mm")


def _topview(out: Path, plan, cut, log, rm, cols=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    for s in plan.strokes:
        ax.plot(-s.xy[:, 1], s.xy[:, 0], color="#c9b99a", lw=6, solid_capstyle="round",
                zorder=1)
    cmap = plt.get_cmap("tab10")
    for k, c in enumerate(cut):
        s = plan.strokes[c.stroke]
        seg = s.xy[c.start: c.end + 1]
        ax.plot(-seg[:, 1], seg[:, 0], color=cmap(k % 10), lw=1.5, zorder=2)
        ax.plot(-c.base[1], c.base[0], marker=(3, 0, -math.degrees(c.yaw)),
                color=cmap(k % 10), ms=9, zorder=3)
    if log is not None:
        ci = {n: i for i, n in enumerate(cols)}
        touch = log[:, ci["contact"]] > 0
        inked = np.isin(log[:, ci["phase"]], (LOWER, WRITE, LIFT))
        for sel, colour, label in ((touch & inked, "k", "hair in the floor"),
                                   (touch & ~inked, "r", "stray ink")):
            if sel.any():
                ax.scatter(-log[sel, ci["iy"]], log[sel, ci["ix"]], s=4, c=colour, zorder=4,
                           label=label)
        ax.plot(-log[:, ci["by"]], log[:, ci["bx"]], color="#888", lw=0.8, zorder=2,
                label="trunk")
        ax.legend(loc="lower left")
    ax.set_aspect("equal")
    ax.set_xlabel("-y (m)")
    ax.set_ylabel("x (m)")
    ax.set_title(f"U+{ord(plan.character):04X}, {plan.size * 100:.1f} cm: plan (tan), "
                 "strokes (colour), stations (triangles)")
    ax.grid(alpha=0.3)
    fig.savefig(out / "topview.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
