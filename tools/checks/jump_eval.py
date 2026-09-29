#!/usr/bin/env python3
"""Evaluate a trained jumper.jump policy by replay: measure the jump and compare with the reference.

Usage:
    python tools/checks/jump_eval.py --ckpt logs/jumper/jumper.jump/<timestamp>/model_2999.pt

All envs spawn from the motion start (phase 0); a deterministic policy (distribution mean) runs 3
episodes, reporting the base apex / airborne time / push-off speed, compared with the npz
reference's stats.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    import torch

    import tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from tasks.jumper.jump.mdp.reference import get_reference

    env_cfg = tasks.load_env_cfg("jumper.jump", play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 3.0  # play mode defaults to 1e9; here we need episode boundaries
    # All spawn from the motion start so the jump metrics are comparable
    env_cfg.events["reset_from_reference"].params["phase_range"] = (0.0, 0.01)

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env)

    from tasks import load_agent_cfg, load_runner_cls

    runner_cls = load_runner_cls("jumper.jump") or None
    if runner_cls is None:
        from mjlab.rl.runner import MjlabOnPolicyRunner as runner_cls
    runner = runner_cls(wrapped, asdict(load_agent_cfg("jumper.jump")), "logs/eval", "cuda:0")
    runner.load(str(args.ckpt))
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = wrapped.reset()
    ref = get_reference(env)
    ref_peak = float(ref.base_pos[:, 2].max())
    ref_stance = ref.stand_base_z
    ref_flight = float(ref.meta["stats"]["flight_s"])
    ref_vz = float(ref.meta["stats"]["vz_takeoff"])
    print(
        f"reference: apex z={ref_peak:.3f} m (rise {ref_peak - ref_stance:.3f} m), "
        f"airborne {ref_flight:.3f} s, push-off vz={ref_vz:.2f} m/s"
    )

    max_z = torch.zeros(args.num_envs, device=env.device)
    flight_s = torch.zeros(args.num_envs, device=env.device)
    takeoff_vz = torch.zeros(args.num_envs, device=env.device)
    episodes_done = 0

    with torch.no_grad():
        while episodes_done < args.episodes:
            action = policy(obs)
            obs, _, dones, _ = wrapped.step(action)
            z = env.scene["robot"].data.root_link_pos_w[:, 2]
            max_z = torch.maximum(max_z, z)
            forces = env.scene.sensors["feet_ground_contact"].data.force
            airborne = (forces.norm(dim=-1) <= 1.0).all(dim=1)
            flight_s += airborne.float() * env.step_dt
            grounded = ~airborne
            vz = env.scene["robot"].data.root_link_lin_vel_w[:, 2]
            takeoff_vz[grounded] = torch.maximum(takeoff_vz[grounded], vz[grounded])

            done = dones.bool()
            if done.any():
                done_ids = done.nonzero(as_tuple=False).flatten()
                print(
                    f"episode {episodes_done}: apex z={max_z[done_ids].mean().item():.3f} m"
                    f" (rise {(max_z[done_ids] - ref_stance).mean().item():.3f} m), "
                    f"airborne {flight_s[done_ids].mean().item():.3f} s, "
                    f"push-off vz={takeoff_vz[done_ids].mean().item():.2f} m/s"
                )
                max_z[done_ids] = 0.0
                flight_s[done_ids] = 0.0
                takeoff_vz[done_ids] = 0.0
                episodes_done += 1  # one wave of done = one episode (envs are synchronized)

    env.close()


if __name__ == "__main__":
    main()
