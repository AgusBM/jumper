#!/usr/bin/env python3
"""Standing, twist only: does the trunk turn over planted feet, or do the feet walk round?

    python tasks/jumper/five_foot/tools/twist_check.py --checkpoint <run>/model_N.pt
    python tasks/jumper/five_foot/tools/twist_check.py --checkpoint <pt> --twists 0,15,-15

`body_twist` scores the offset between the trunk and its footholds, and the feet
can make that offset as well as the trunk can. Both read as the same number on the
dashboard, so this is the check that separates them. Every environment is told to
stand (velocity command zero), settles for 1.2 s with the attitude level, and is
then held at one twist for 4 s:

  * **trunk turned** -- the trunk's yaw in the world over those 4 s;
  * **feet turned** -- the five feet's rotation about their own centre in the world,
    a 2-D Procrustes fit; the intended behaviour is zero;
  * **feet moved** -- each foot's horizontal displacement, mean and worst;
  * **lift-offs** -- contact lost, per foot, over the 4 s;
  * **sum|v_xy|** -- the feet's summed horizontal speed, averaged: what
    `mdp/rewards.py::feet_still_when_standing` charges, per second, before its
    weight.

Trunk turned minus feet turned is the twist `body_twist` sees. A policy obeying the
command puts all of it in the first column.
"""

from __future__ import annotations

import argparse
import math
import warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--twists", default="0,10,20,30,-20,-30", help="degrees, comma separated")
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
    from tasks.jumper.five_foot.mdp.pose_command import TWIST
    from tasks.jumper.five_foot.mdp.rewards import _site_pos_w

    # Standing everywhere, and the attitude level until this script says otherwise.
    cfg = tasks.load_env_cfg("jumper.five_foot", play=True)
    cfg.scene.num_envs = args.num_envs
    twist_cfg = cfg.commands["twist"]
    twist_cfg.ranges.lin_vel_x = twist_cfg.ranges.lin_vel_y = (0.0, 0.0)
    twist_cfg.ranges.ang_vel_z = (0.0, 0.0)
    twist_cfg.rel_standing_envs, twist_cfg.rel_heading_envs = 1.0, 0.0
    twist_cfg.resampling_time_range = (1e6, 1e6)
    pose_cfg = cfg.commands["body_pose"]
    for field in ("stand_pitch", "stand_roll", "stand_twist",
                  "move_pitch", "move_roll", "move_twist"):
        setattr(pose_cfg, field, (0.0, 0.0))
    pose_cfg.rel_neutral_envs, pose_cfg.resampling_time_range = 0.0, (1e6, 1e6)

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
    contact = env.scene.sensors["feet_ground_contact"]
    pose = env.command_manager.get_term("body_pose")
    settle, hold = 60, 200

    def yaw() -> torch.Tensor:
        w, x, y, z = robot.data.root_link_quat_w.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def down() -> torch.Tensor:
        found = contact.data.found
        return (found[..., 0] if found.dim() == 3 else found) > 0

    print(f"{args.checkpoint}\n{args.num_envs} envs standing; twist held {hold * env.step_dt:g} s "
          f"after a {settle * env.step_dt:g} s settle\n")
    print(f"{'twist':>6} | {'trunk turned':>12} | {'feet turned':>11} | "
          f"{'feet moved mean / worst':>23} | {'lift-offs':>9} | {'sum|v_xy|':>9}")
    for deg in (float(t) for t in args.twists.split(",")):
        with torch.inference_mode():
            obs, _ = wrapped.reset()
            for _ in range(settle):
                obs, *_ = wrapped.step(policy(obs))
            yaw0 = yaw().clone()
            feet0 = _site_pos_w(robot)[:, sites, :2].clone()
            was_down = down()
            lifts = torch.zeros(args.num_envs, len(sites), device=res.device)
            path = torch.zeros(args.num_envs, device=res.device)
            for _ in range(hold):
                pose.pose_target_b[:, TWIST] = math.radians(deg)
                obs, *_ = wrapped.step(policy(obs))
                now = down()
                lifts += (was_down & ~now).float()
                was_down = now
                speed = robot.data.site_lin_vel_w[:, sites, :2].norm(dim=-1).sum(dim=1)
                path += speed * env.step_dt
            turned = torch.rad2deg((yaw() - yaw0 + math.pi) % (2 * math.pi) - math.pi)
            feet1 = _site_pos_w(robot)[:, sites, :2]
            moved = (feet1 - feet0).norm(dim=-1) * 1000.0
            a = feet0 - feet0.mean(dim=1, keepdim=True)
            b = feet1 - feet1.mean(dim=1, keepdim=True)
            cross = (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).sum(dim=1)
            feet_turned = torch.rad2deg(torch.atan2(cross, (a * b).sum(dim=(1, 2))))
        print(f"{deg:+5.0f}° | {turned.mean():+11.1f}° | {feet_turned.mean():+10.1f}° | "
              f"{moved.mean():8.1f} / {moved.max():5.1f} mm      | {lifts.mean():9.2f} | "
              f"{path.mean() / (hold * env.step_dt):7.3f}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
