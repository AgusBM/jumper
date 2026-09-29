#!/usr/bin/env python3
"""Zero-action pure reference replay diagnostics: are the foot tips planted in our simulation?

    python tools/checks/jump_ref_play_diag.py

No policy is involved (a zero residual action = the position target is the reference q_cmd). If
the middle legs still leave the ground during the crouch, it means our model's kinematics/contact
geometry do not match the recording, not that the policy learned wrong.
"""

from __future__ import annotations

import argparse

LEG_NAMES = ("LF", "RF", "LM", "RM", "LR", "RR")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()

    import torch

    import tasks
    from mjlab.envs import ManagerBasedRlEnv
    from tasks.jumper.common.constants import FEET

    env_cfg = tasks.load_env_cfg("jumper.jump", play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 3.0
    env_cfg.events["reset_from_reference"].params["phase_range"] = (0.0, 0.01)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    try:
        robot = env.scene["robot"]
        cmd = env.command_manager.get_term("jump")
        ref = cmd.reference
        # The bodies the imitation reward measures. Not `meta["ik_ee"]`: that names
        # the generator model's bodies, which this robot does not have.
        foot_ids = torch.as_tensor(
            robot.find_bodies(list(FEET), preserve_order=True)[0],
            device=env.device, dtype=torch.long,
        )
        zero = torch.zeros(args.num_envs, env.action_space.shape[-1], device=env.device)

        NB = 10
        sums = torch.zeros(NB, 4 + 6, device=env.device)
        counts = torch.zeros(NB, device=env.device)
        obs, _ = env.reset()
        with torch.inference_mode():
            for _ in range(args.steps):
                obs, _, _, _, _ = env.step(zero)
                phase = ref.phase(cmd.time_since_go)
                z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
                z_ref = ref.sample(ref.index_of(cmd.time_since_go))["base_pos"][:, 2]
                fz = robot.data.body_link_pos_w[:, foot_ids, 2]
                forces = env.scene.sensors["feet_ground_contact"].data.force
                planted = (forces.norm(dim=-1) > 1.0).float()
                b = (phase * NB).long().clamp(0, NB - 1)
                for k in range(NB):
                    m = b == k
                    counts[k] += m.sum()
                    sums[k, 0] += (z[m] - z_ref[m]).abs().sum()
                    sums[k, 1] += planted[m].mean(dim=1).sum()
                    sums[k, 2] += fz[m].min(dim=1).values.sum()
                    sums[k, 3] += fz[m].max(dim=1).values.sum()
                    for j in range(6):
                        sums[k, 4 + j] += planted[m][:, j].sum()

        print(f"{'phase':>9} {'z_err(mm)':>10} {'planted':>8} {'minZ':>7} {'maxZ':>7}  "
              + " ".join(LEG_NAMES))
        for k in range(NB):
            n = counts[k].clamp(min=1)
            per_leg = " ".join(f"{(sums[k, 4 + j] / n).item():.2f}" for j in range(6))
            print(
                f"{k / NB:4.1f}-{(k + 1) / NB:4.1f} "
                f"{(sums[k, 0] / n * 1000).item():10.1f} "
                f"{(sums[k, 1] / n).item():8.3f} "
                f"{(sums[k, 2] / n * 1000).item():7.1f} "
                f"{(sums[k, 3] / n * 1000).item():7.1f}  {per_leg}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
