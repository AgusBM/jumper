#!/usr/bin/env python3
"""Velocity tracking, on a ruler that does not move between runs.

    python tasks/jumper/five_foot/tools/vel_check.py --checkpoint <run>/model_N.pt
    python tasks/jumper/five_foot/tools/vel_check.py --checkpoint <pt> --seed 1

**`Episode_Reward/track_angular_velocity` is not comparable across runs.** The
tracking terms score `exp(-err^2 / std^2)`, and `std` is `ratio * ang_range *
scale` -- so a task that changes `command_ang_ceiling` or `command_ang_std_ratio`
changes the ruler with it, and two runs report the same number for different
errors. Worse, the term **saturates**: at std 0.5333 an error of 0.09 and one of
0.12 both read about 0.97, which is how a policy drifted from tracking 2.00 rad/s
exactly to tracking 1.92, and from 0.50 to 0.40, with the dashboard flat
throughout (`env_cfg.py::command_ang_std_ratio` has that measurement).

This measures the raw error instead, in m/s and rad/s, over a fixed list of
commands held one at a time, so two checkpoints can be compared whatever ruler
each was trained on. Per command it reports:

  * **cmd / act** -- commanded against the mean tracked value, which separates a
    systematic bias from noise: a policy that under-turns by 8% and one that is
    centred but jittery can carry the same mean error;
  * **|v_xy| err** -- the mean distance between the commanded and actual planar
    velocity, in the body frame;
  * **wz err** -- the mean absolute yaw-rate error.

Both are means over environments and over the recorded window, and the command is
pinned for the whole run (`resampling_time_range` is set past the episode), so
what comes back is steady-state tracking, not the response to a step.

Repeatable to about 0.01 rad/s: the same checkpoint at `--seed 0` and `--seed 1`
tracked 0.46 / 0.69 / 1.25 / 1.90 and 0.46 / 0.70 / 1.25 / 1.90 on the four turns
(`2026-09-23_19-56-20/model_2500`, warp, 32 envs). A difference smaller than that
is not one.
"""
from __future__ import annotations

import argparse
import warnings

CMDS = (
    ("stand",        (0.0, 0.0, 0.0)),
    ("fwd 0.25",     (0.25, 0.0, 0.0)),
    ("fwd 0.50",     (0.5, 0.0, 0.0)),
    ("back 0.30",    (-0.3, 0.0, 0.0)),
    ("left 0.25",    (0.0, 0.25, 0.0)),
    ("turn 0.50",    (0.0, 0.0, 0.5)),
    ("turn 0.75",    (0.0, 0.0, 0.75)),
    ("turn 1.33",    (0.0, 0.0, 1.3333)),
    ("turn 2.00",    (0.0, 0.0, 2.0)),
    ("fwd+turn",     (0.3, 0.0, 0.75)),
)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_envs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    import torch
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    import tasks
    res = resolve(backend="warp", device="cuda", num_envs=args.num_envs)
    use_backend(res)
    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    cfg = tasks.load_env_cfg("jumper.five_foot", play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    tw = cfg.commands["twist"]
    tw.rel_standing_envs, tw.rel_heading_envs = 0.0, 0.0
    tw.resampling_time_range = (1e6, 1e6)
    pose = cfg.commands["body_pose"]
    pose.rel_neutral_envs, pose.resampling_time_range = 1.0, (1e6, 1e6)

    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    agent = tasks.load_agent_cfg("jumper.five_foot")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = (tasks.load_runner_cls("jumper.five_foot") or MjlabOnPolicyRunner)(
        wrapped, asdict(agent), device=res.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=res.device)
    policy = runner.get_inference_policy(device=res.device)
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("twist")
    settle, record = 100, 250

    print(f"{args.checkpoint}\n{args.num_envs} envs, {record*env.step_dt:g}s recorded "
          f"after {settle*env.step_dt:g}s settle\n")
    print(f"{'command':<10} | {'vx cmd/act':>16} | {'wz cmd/act':>16} | "
          f"{'|v_xy| err':>10} | {'wz err':>8}")
    rows = []
    for name, (vx, vy, wz) in CMDS:
        cmd.cfg.ranges.lin_vel_x = (vx, vx)
        cmd.cfg.ranges.lin_vel_y = (vy, vy)
        cmd.cfg.ranges.ang_vel_z = (wz, wz)
        lin, ang = [], []
        with torch.inference_mode():
            obs, _ = wrapped.reset()
            for step in range(settle + record):
                obs, *_ = wrapped.step(policy(obs))
                if step < settle:
                    continue
                lin.append(robot.data.root_link_lin_vel_b[:, :2].clone())
                ang.append(robot.data.root_link_ang_vel_b[:, 2].clone())
        L, A = torch.stack(lin, 1), torch.stack(ang, 1)
        tgt = torch.tensor([vx, vy], device=L.device)
        lin_err = float((L - tgt).norm(dim=-1).mean())
        ang_err = float((A - wz).abs().mean())
        print(f"{name:<10} | {vx:+6.2f} / {float(L[...,0].mean()):+6.2f} | "
              f"{wz:+6.2f} / {float(A.mean()):+6.2f} | {lin_err:10.3f} | {ang_err:8.3f}")
        rows.append((name, lin_err, ang_err))
    walk = [r for r in rows if r[0] != "stand"]
    print(f"\nmean over the 9 moving commands: |v_xy| err {sum(r[1] for r in walk)/len(walk):.3f} m/s, "
          f"wz err {sum(r[2] for r in walk)/len(walk):.3f} rad/s")
    env.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
