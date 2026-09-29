#!/usr/bin/env python3
"""Walking: does RF step in its window, step outside it, or drag?

    python tasks/jumper/five_foot/tools/gait_check.py --checkpoint <run>/model_N.pt

The gait is LM+RR against RM+LR, with RF stepping on its own while the four are
planted (`mdp/rewards.py::GROUP_A`). Each of those can go wrong without a number
on the dashboard saying so -- RF lifting with a pair, RF never lifting and sliding
along instead, the pairs alternating with no four-planted moment for RF to use --
so this drives a trained policy through five walking commands and reports:

  * **RF up** -- the fraction of control steps RF is off the ground;
  * **out of window** -- the fraction RF is up with a gait leg up too, which is
    what `rf_step_window` charges;
  * **four down** -- the fraction the four gait legs are all planted, the window;
  * **lift** -- each foot's mean peak height per swing, in mm;
  * **drag** -- each foot's mean horizontal speed while touching the ground, in
    mm/s: a foot that is carried forward without lifting shows up here.

Airborne is contact lost (`found`), the test the gait terms read.
"""

from __future__ import annotations

import argparse
import warnings

COMMANDS = (
    ("forward 0.4", (0.4, 0.0, 0.0)),
    ("forward 0.25", (0.25, 0.0, 0.0)),
    ("back 0.3", (-0.3, 0.0, 0.0)),
    ("left 0.25", (0.0, 0.25, 0.0)),
    ("turn 0.5", (0.0, 0.0, 0.5)),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_envs", type=int, default=32)
    ap.add_argument("--backend", choices=("warp", "native"), default="warp")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    import torch
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    import tasks

    res = resolve(backend=args.backend,
                  device="cuda" if args.backend == "warp" else "cpu", num_envs=args.num_envs)
    use_backend(res)

    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    from tasks.jumper.five_foot.claw import FIVE_FOOT_LEGS
    from tasks.jumper.five_foot.mdp.rewards import GAIT_FEET, RF_FOOT, _site_pos_w

    cfg = tasks.load_env_cfg("jumper.five_foot", play=True)
    cfg.scene.num_envs = args.num_envs
    twist_cfg = cfg.commands["twist"]
    twist_cfg.rel_standing_envs, twist_cfg.rel_heading_envs = 0.0, 0.0
    twist_cfg.resampling_time_range = (1e6, 1e6)
    pose_cfg = cfg.commands["body_pose"]
    pose_cfg.rel_neutral_envs, pose_cfg.resampling_time_range = 1.0, (1e6, 1e6)

    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    agent = tasks.load_agent_cfg("jumper.five_foot")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = (tasks.load_runner_cls("jumper.five_foot") or MjlabOnPolicyRunner)(
        wrapped, asdict(agent), device=res.device
    )
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=res.device)
    policy = runner.get_inference_policy(device=res.device)

    robot = env.scene["robot"]
    sites = [list(robot.site_names).index(leg) for leg in FIVE_FOOT_LEGS]
    sensor = env.scene.sensors["feet_ground_contact"]
    command = env.command_manager.get_term("twist")
    settle, record = 100, 250

    print(f"{args.checkpoint}\n{args.num_envs} envs per command, "
          f"{record * env.step_dt:g} s recorded after {settle * env.step_dt:g} s\n")
    legs = "  ".join(f"{leg:>4}" for leg in FIVE_FOOT_LEGS)
    print(f"{'command':<13} | {'RF up':>5} | {'out of window':>13} | {'four down':>9} | "
          f"lift mm  {legs} | drag mm/s  {legs}")
    for name, (vx, vy, wz) in COMMANDS:
        command.cfg.ranges.lin_vel_x = (vx, vx)
        command.cfg.ranges.lin_vel_y = (vy, vy)
        command.cfg.ranges.ang_vel_z = (wz, wz)
        down, height, speed = [], [], []
        with torch.inference_mode():
            obs, _ = wrapped.reset()
            for step in range(settle + record):
                obs, *_ = wrapped.step(policy(obs))
                if step < settle:
                    continue
                found = sensor.data.found
                found = found[..., 0] if found.dim() == 3 else found
                down.append((found > 0).float().clone())
                height.append(_site_pos_w(robot)[:, sites, 2].clone())
                speed.append(robot.data.site_lin_vel_w[:, sites, :2].norm(dim=-1).clone())
        c, h, v = torch.stack(down, 1), torch.stack(height, 1), torch.stack(speed, 1)
        rf_up = 1.0 - c[..., RF_FOOT]
        gait_up = (1.0 - c[..., list(GAIT_FEET)]).amax(dim=-1)
        ground = h.amin(dim=1, keepdim=True)
        lift, drag = [], []
        for k in range(len(FIVE_FOOT_LEGS)):
            peaks = []
            for e in range(args.num_envs):
                air = (c[e, :, k] < 0.5).tolist()
                t = 0
                while t < len(air):
                    if air[t]:
                        t0 = t
                        while t < len(air) and air[t]:
                            t += 1
                        if t0 > 0 and t < len(air):
                            peaks.append(float((h[e, t0:t, k] - ground[e, 0, k]).max()))
                    else:
                        t += 1
            lift.append(1000.0 * sum(peaks) / len(peaks) if peaks else 0.0)
            touching = c[..., k] > 0.5
            drag.append(1000.0 * float(v[..., k][touching].mean()) if touching.any() else 0.0)
        print(f"{name:<13} | {rf_up.mean():5.0%} | {(rf_up * gait_up).mean():13.0%} | "
              f"{(1.0 - gait_up).mean():9.0%} | "
              f"{'':7}{'  '.join(f'{x:4.1f}' for x in lift)} | "
              f"{'':9}{'  '.join(f'{x:4.0f}' for x in drag)}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
