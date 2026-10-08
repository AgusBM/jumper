#!/usr/bin/env python3
"""Step 2: Jumper writes a character on the floor with a brush.

    python tasks/jumper/calligraphy/tools/write.py                   # 无, 0.30 m
    python tasks/jumper/calligraphy/tools/write.py --char 无 --size 0.30 --plan-only

`jumper.five_foot`'s trained policy walks and stands; this moves the carried arm,
which that task leaves out of the policy's action, and drives the policy's
velocity command. Nothing is trained and nothing in the policy changes.

## The loop

The plan (`hanzi.py`) is cut into stretches the arm can reach from one place
(`stations.py`). For each stretch:

    walk     the arm held still, the velocity command steering the trunk to the
             stretch's station: position and heading, then a short settle
    reach    the tip moves, `HOVER` above the floor, to above the stretch's first
             point -- in joint space from the stow on the first stretch, in a
             straight line otherwise
    lower    down onto the floor
    write    along the stretch at `--speed`, pressed in by `press * --depth`
    lift     back up to `HOVER`

Every target is solved against the trunk **as it is**, each control step
(`arm.py`): the trunk is never exactly where it was sent and drifts while the arm
writes, and solving against the measured pose is what keeps that out of the ink.

## What it writes

`logs/calligraphy/<u65e0>/<time>/`: `stretches.json` (the cut, as written --
replanned stretches included), `log.npz` (one row per control step: phase, stroke,
target and measured tip, the brush's contact and force, the trunk's pose, the IK
residual, the command; and the full `qpos`, to replay into a renderer) and
`topview.png`, the plan beside where the brush actually touched the floor.

The policy runs on `native:cpu` by default -- one environment, so a GPU buys
nothing here.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
CHECKPOINT = REPO / "tasks/jumper/five_foot/out/example/model_86600.pt"
TASK = "jumper.five_foot"

# Phases, as logged.
WALK, SETTLE, REACH, LOWER, WRITE, LIFT, FOLD, DONE = range(8)
PHASES = ("walk", "settle", "reach", "lower", "write", "lift", "fold", "done")

#: Walking: P gains and limits on the velocity command.
K_POS, V_MAX = 1.5, 0.12          # 1/s, m/s
K_YAW, W_MAX = 2.0, 0.6           # 1/s, rad/s
#: Below this the policy stands rather than steps; a correction is sent at least this fast.
V_MIN = 0.08
#: A stretch is replanned when, from where the trunk stopped, fewer than this many
#: samples are this far inside the band; the walk is retried first.
RUN_MARGIN = 0.004                # m
MIN_SAMPLES = 20
WALK_ATTEMPTS = 2
#: While writing, this many consecutive IK misses end the stretch where it is.
MISS_STEPS = 3
#: The arm comes to rest for this long after unfolding or folding; going straight
#: on, the swing carried the tip 15 mm below the line it was sent along.
ARM_SETTLE_S = 0.3
#: Warn when no fold path keeps the arm this far off the floor.
FOLD_CLEARANCE = 0.01
#: Arrived: within this of the station, and of its heading.
POS_TOL, YAW_TOL = 0.010, math.radians(3.0)
WALK_TIMEOUT = 12.0               # s
SETTLE_S = 0.6
UNFOLD_S = 1.5                    # joint-space reach from the stow
FOLD_S = 1.2                      # and back to it before walking
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
#: Pressing. A fixed depth below the floor pushed with 0-6 N as the trunk moved
#: under the arm, and on 无's long middle stroke that push turned the trunk from
#: -8 to -56 deg and lifted it 15 mm while the brush bounced (contact on 81% of
#: steps; 2026-10-08, native:cpu). So the depth is only where the press starts:
#: an integrator moves the tip up or down to hold the normal force at
#: `press * --force`.
KF = 0.004                        # m per N*s
ZCORR = (-0.004, 0.006)           # m, how far the integrator may move the tip


class _Steer:
    """Stands in for the operator on the twist command: returns our command."""

    def __init__(self):
        self.values = [0.0, 0.0, 0.0]

    def command(self, term, stamp=None):
        return list(self.values)

    def task_control(self, name, stamp=None):
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
    ap.add_argument("--size", type=float, default=0.30, help="em square, metres")
    ap.add_argument("--origin", type=float, nargs=2, default=(0.45, 0.0),
                    help="world xy of the character's centre; the robot starts at 0,0 "
                         "facing +x")
    ap.add_argument("--speed", type=float, default=0.04, help="writing speed, m/s")
    ap.add_argument("--depth", type=float, default=0.003,
                    help="how far the tip is pressed below the floor at full press, m")
    ap.add_argument("--force", type=float, default=1.0,
                    help="normal force at full press, N")
    ap.add_argument("--margin", type=float, default=0.012,
                    help="how far inside the reach band every written point must be, m")
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--plan-only", action="store_true",
                    help="cut the plan into stretches, draw them, and stop")
    ap.add_argument("--max-stretches", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    import torch
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    use_backend(resolve(backend="native", device="cpu", num_envs=1))

    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    import tasks
    from tasks.jumper.calligraphy import arm as armmod
    from tasks.jumper.calligraphy import brush, hanzi, stations
    from tasks.jumper.common.constants import HOME, STAND_Z
    from tasks.jumper.five_foot.claw import ARM_JOINTS, LF_GRASP

    plan = hanzi.plan(args.char, args.size, tuple(args.origin))
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    out = args.out or REPO / "logs" / "calligraphy" / f"u{ord(args.char):04x}" / stamp
    out.mkdir(parents=True, exist_ok=True)

    # ── The environment: five_foot's replay config, plus the brush ───────────
    cfg = tasks.load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    brush.apply(cfg)
    pose = cfg.commands["body_pose"]
    pose.rel_neutral_envs = 1.0  # level trunk throughout
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    agent = tasks.load_agent_cfg(TASK)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner_cls = tasks.load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent), device="cpu")
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")
    steer = _Steer()
    env.command_manager.get_term("twist")._operator = steer

    robot = env.scene["robot"]
    carried = list(ARM_JOINTS) + [armmod.FINGER_JOINT]
    jids = [robot.joint_names.index(j) for j in carried]
    sensor = env.scene.sensors["brush_ground"]
    dt = env.step_dt

    # ── Where the arm can write, and the cut ─────────────────────────────────
    arm = armmod.Arm(env.sim.mj_model)
    # Standing at the origin, level, legs at HOME, the brush in the claw: fixed,
    # not read from the simulation, whose reset randomises the joints.
    mjm = env.sim.mj_model
    standing = mjm.qpos0.astype(np.float64).copy()
    standing[0:3] = (0.0, 0.0, STAND_Z)
    standing[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, value in {**HOME, armmod.FINGER_JOINT: brush.FINGER_HOLD}.items():
        standing[mjm.jnt_qposadr[mjm.joint(armmod.PREFIX + name).id]] = value
    t0 = time.time()
    rm = armmod.cached_reach_map(arm, standing, REPO / "logs" / "calligraphy" / "cache")
    print(f"[write] reach band: {rm.ok.sum() * rm.cell**2 * 1e4:.0f} cm^2 "
          f"({time.time() - t0:.1f} s)")
    cut = stations.plan_stretches(plan, rm, margin=args.margin)
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
        tip_now[:] = tip
        force_now[0] = abs(float(sensor.data.force[0].reshape(-1)[2]))
        found = float(sensor.data.found[0].reshape(-1)[0])
        force = sensor.data.force[0].reshape(-1)[:3].cpu().numpy()
        tgt = target if target is not None else (np.nan, np.nan, np.nan)
        qpos_rows.append(qpos.astype(np.float32))
        rows.append([sim_t, phase, done[k].stroke if 0 <= k < len(done) else -1, k, sample,
                     *tgt, *tip, found, *force, *base, yaw, residual, *steer.values])

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
    force_now = np.zeros(1)
    corr = np.zeros(2)
    zcorr = np.zeros(1)
    dist_map = rm.margin()

    def track(true_xy, lead_xy, z, press=None):
        """Solve for `lead_xy` plus the integrated correction; log `true_xy`.
        With `press`, the height is corrected to hold the normal force."""
        nonlocal corr
        corr = np.clip(corr + KI * dt * (true_xy - tip_now[:2]), -CORR_MAX, CORR_MAX)
        if press is not None:
            zcorr[0] = np.clip(zcorr[0] + KF * dt * (force_now[0] - press * args.force),
                               *ZCORR)
            z = z + zcorr[0]
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

    def line(frm, to, speed, phase, k, sample):
        n = max(1, int(np.linalg.norm(to - frm) / speed / dt))
        for i in range(1, n + 1):
            tgt = frm + (to - frm) * (i / n)
            step(phase, k, sample, tgt, solve(tgt))

    todo = deque(cut)
    done: list = []
    folded = True
    while todo:
        c = todo.popleft()
        k = len(done)
        done.append(c)
        s = plan.strokes[c.stroke]
        # walk (folded: five_foot does not walk with the arm held out)
        for attempt in range(WALK_ATTEMPTS):
            t_start = sim_t
            while True:
                _, base, yaw = state()
                e = c.base - base[:2]
                eyaw = _wrap(c.yaw - yaw)
                dist = float(np.linalg.norm(e))
                if dist < POS_TOL and abs(eyaw) < YAW_TOL:
                    break
                if sim_t - t_start > WALK_TIMEOUT:
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
                vb = (cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1])
                w = float(np.clip(K_YAW * eyaw, -W_MAX, W_MAX))
                steer.values = [float(vb[0]), float(vb[1]), w]
                step(WALK, k, c.start, None, np.nan)
            steer.values = [0.0, 0.0, 0.0]
            for _ in range(int(SETTLE_S / dt)):
                step(SETTLE, k, c.start, None, np.nan)
            _, base, yaw = state()
            last = stations.reachable_until(s, c.start, c.end, base, yaw, rm,
                                            RUN_MARGIN, dist_map)
            if last - c.start >= MIN_SAMPLES or last == c.end:
                break
            print(f"\n[write] stretch {k}: arrived {np.linalg.norm(c.base - base[:2]) * 100:.1f}"
                  f" cm off; walking again ({attempt + 1}/{WALK_ATTEMPTS})")
        if last < c.end:
            # Arrived somewhere else: write what is reachable from here, cut the
            # rest again from where this stops.
            if last <= c.start:
                last = c.start + 1
            rest = stations.cut_stroke(s, last, rm, args.margin, stations.YAWS, c.yaw, dist_map)
            print(f"\n[write] stretch {k}: from where the trunk stopped it reaches sample "
                  f"{last} of [{c.start}, {c.end}]; replanned the rest as {len(rest)}")
            c.end = last
            todo.extendleft(reversed(rest))

        # reach: unfold through HIGH, then down to HOVER above the first point
        p0 = s.xy[c.start]
        above = np.array([*p0, armmod.HOVER])
        _, base, yaw = state()
        if folded:
            qpos, _, _ = state()
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
        else:
            qpos, _, _ = state()
            arm.set_state(qpos)
            start = arm.tip_pos(q_cmd)
        line(start, above, REACH_SPEED, REACH, k, c.start)

        # lower
        z0 = -s.press[c.start] * args.depth
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
            tgt = np.array([*xy, -press * args.depth])
            err = track(xy, at(u + lead)[0], tgt[2], press)
            step(WRITE, k, int(u), tgt, err)
            misses = misses + 1 if err > IK_RETRY else 0
            if misses >= MISS_STEPS and u < c.end:
                # The trunk drifted while the arm wrote and the stroke ran out of
                # reach: stop here and cut the rest again.
                stop = max(c.start + 1, int(u) - int(lead))
                rest = stations.cut_stroke(s, stop, rm, args.margin, stations.YAWS, c.yaw,
                                           dist_map)
                print(f"\n[write] stretch {k}: out of reach at sample {stop} "
                      f"(trunk drifted); replanned the rest as {len(rest)}")
                c.end = stop
                todo.extendleft(reversed(rest))
                break

        # lift
        pe = s.xy[c.end]
        ze = -s.press[c.end] * args.depth
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
            unfold_to(stow, k, c.end, None)
            folded = True
        print(f"[write] stretch {k + 1} done at t={sim_t:.1f}s ({len(todo)} to go)", flush=True)

    for _ in range(int(1.0 / dt)):
        step(DONE, len(done) - 1, -1, None, np.nan)
    cut = done
    (out / "stretches.json").write_text(json.dumps([
        {"stroke": c.stroke, "start": c.start, "end": c.end,
         "base": np.round(c.base, 4).tolist(), "yaw_deg": round(math.degrees(c.yaw), 1),
         "margin_mm": round(c.margin * 1000, 1)} for c in cut
    ], indent=1))

    cols = ["t", "phase", "stroke", "stretch", "sample", "tx", "ty", "tz", "px", "py", "pz",
            "contact", "fx", "fy", "fz", "bx", "by", "bz", "byaw", "ik_residual",
            "cmd_vx", "cmd_vy", "cmd_wz"]
    log = np.asarray(rows, dtype=np.float64)
    np.savez(out / "log.npz", log=log, columns=np.array(cols), phases=np.array(PHASES),
             qpos=np.asarray(qpos_rows), dt=dt)
    _report(log, cols)
    _topview(out, plan, cut, log, rm, cols)
    print(f"[write] wrote {out}")
    env.close()
    return 0


def _report(log: np.ndarray, cols: list[str]) -> None:
    c = {n: i for i, n in enumerate(cols)}
    w = log[log[:, c["phase"]] == WRITE]
    err = np.linalg.norm(w[:, [c["px"], c["py"]]] - w[:, [c["tx"], c["ty"]]], axis=1)
    touching = w[:, c["contact"]] > 0
    print(f"[write] {log[-1, c['t']]:.1f} s simulated, {len(w)} writing steps")
    print(f"  tip xy error while writing: median {np.median(err) * 1000:.1f} mm, "
          f"p95 {np.percentile(err, 95) * 1000:.1f} mm, max {err.max() * 1000:.1f} mm")
    print(f"  brush on the floor while writing: {touching.mean() * 100:.0f}% of steps")
    if touching.any():
        fz = np.abs(w[touching, c["fz"]])
        print(f"  normal force while touching: median {np.median(fz):.2f} N, "
              f"p95 {np.percentile(fz, 95):.2f} N")
    off = log[~np.isin(log[:, c["phase"]], (WRITE, LOWER, LIFT))]
    print(f"  brush on the floor outside lower/write: {(off[:, c['contact']] > 0).sum()} steps")
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
        for sel, colour, label in ((touch & inked, "k", "brush on the floor"),
                                   (touch & ~inked, "r", "stray contact")):
            if sel.any():
                ax.scatter(-log[sel, ci["py"]], log[sel, ci["px"]], s=4, c=colour, zorder=4,
                           label=label)
        ax.plot(-log[:, ci["by"]], log[:, ci["bx"]], color="#888", lw=0.8, zorder=2,
                label="trunk")
        ax.legend(loc="lower left")
    ax.set_aspect("equal")
    ax.set_xlabel("-y (m)")
    ax.set_ylabel("x (m)")
    ax.set_title(f"U+{ord(plan.character):04X}: plan (tan), stretches (colour) and stations (triangles)")
    ax.grid(alpha=0.3)
    fig.savefig(out / "topview.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
