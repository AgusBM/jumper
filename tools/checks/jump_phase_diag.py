#!/usr/bin/env python3
"""Diagnose foot contact in jumper.jump by phase: actual vs reference.

    python tools/checks/jump_phase_diag.py --ckpt logs/jumper/jumper.jump/<ts>/model_2999.pt

All envs spawn from the motion start, run a deterministic policy, and bucket every step by phase:
- min-foot-z: off-ground height of the highest of the 6 foot tips (>0 means a foot is airborne)
- planted: fraction of feet actually in contact (force > 1 N)
- ref_planted: fraction the reference schedule requires to be in contact
In crouch/stance phases, ref_planted=1 while planted is clearly lower is direct evidence of
"feet not planted".
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()

    import torch

    import tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from tasks.jumper.common.constants import FEET

    env_cfg = tasks.load_env_cfg("jumper.jump", play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 3.0
    env_cfg.events["reset_from_reference"].params["phase_range"] = (0.0, 0.01)

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env)

    runner = MjlabOnPolicyRunner(
        wrapped, asdict(tasks.load_agent_cfg("jumper.jump")), "logs/diag", "cuda:0"
    )
    runner.load(str(args.ckpt), load_cfg={"actor": True}, strict=True,
                map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")

    cmd = env.command_manager.get_term("jump")
    ref = cmd.reference
    robot = env.scene["robot"]
    # The bodies the imitation reward measures. Not `meta["ik_ee"]`: that names the
    # generator model's bodies, which this robot does not have.
    foot_ids = torch.as_tensor(
        robot.find_bodies(list(FEET), preserve_order=True)[0],
        device=env.device, dtype=torch.long,
    )

    N_BINS = 10
    sums = torch.zeros(N_BINS, 3, device=env.device)  # min_z, planted, ref_planted
    counts = torch.zeros(N_BINS, device=env.device)

    obs, _ = wrapped.reset()
    with torch.inference_mode():
        for _ in range(args.steps):
            obs, _, done, _ = wrapped.step(policy(obs))
            tsg = cmd.time_since_go
            phase = ref.phase(tsg)

            foot_z = robot.data.body_link_pos_w[:, foot_ids, 2]
            min_z = foot_z.max(dim=1).values  # the highest foot

            forces = env.scene.sensors["feet_ground_contact"].data.force
            planted = (forces.norm(dim=-1) > 1.0).float().mean(dim=1)

            _, ref_s, _, _ = __import__(
                "tasks.jumper.jump.mdp.reference", fromlist=["ref_state"]
            ).ref_state(env, "jump")
            ref_planted = ref_s["contact"].float().mean(dim=1)

            bin_idx = (phase * N_BINS).long().clamp(0, N_BINS - 1)
            for b in range(N_BINS):
                m = bin_idx == b
                counts[b] += m.sum()
                sums[b, 0] += (min_z[m]).sum()
                sums[b, 1] += (planted[m]).sum()
                sums[b, 2] += (ref_planted[m]).sum()

    print(f"{'phase':>10} {'min-foot-z(mm)':>15} {'planted':>9} {'ref_planted':>12}")
    for b in range(N_BINS):
        n = counts[b].clamp(min=1)
        print(
            f"{b / N_BINS:4.1f}-{(b + 1) / N_BINS:4.1f} "
            f"{(sums[b, 0] / n * 1000).item():15.1f} "
            f"{(sums[b, 1] / n).item():9.3f} "
            f"{(sums[b, 2] / n).item():12.3f}"
        )

    env.close()


if __name__ == "__main__":
    main()
