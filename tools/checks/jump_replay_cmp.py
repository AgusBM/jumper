#!/usr/bin/env python3
"""Compare two replay methods: targeting the npz q (actual) vs q_cmd (the controller command).

    python tools/checks/jump_replay_cmp.py --mode q_cmd
    python tools/checks/jump_replay_cmp.py --mode q

During recording, PD tracked q_cmd; q is the actual state with lag. Replaying with q as the
target amounts to double lag, giving a weak push-off (the foot tip rises only ~30mm); q_cmd is
the target the recording controller actually commanded, and the baseline the task's residual
action adds to -- so `--mode q_cmd` is the task itself with a zero residual.

Both modes run through `env.step`, so the decimation, the servo curve and the sensors are the
task's own, and `--mode q` differs only by a residual that moves the target from q_cmd onto q.
This used to step the physics by hand, and that loop had drifted from the task three ways: one
physics step per control step (the reference ran at twice the speed of the physics), a 300 rpm
joint-speed wall the task no longer has, and q and q_cmd read on clocks 0.3 s apart.
"""

from __future__ import annotations

import argparse

import torch

LEG_NAMES = ("LF", "RF", "LM", "RM", "LR", "RR")


def run(env, mode: str, steps: int, nb: int = 10):
    from tasks.jumper.common.constants import FEET

    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("jump")
    ref = cmd.reference
    term = env.action_manager.get_term("joint_pos")
    # The bodies the imitation reward measures. Not `meta["ik_ee"]`: that names the
    # generator model's bodies, which this robot does not have.
    foot_ids = torch.as_tensor(
        robot.find_bodies(list(FEET), preserve_order=True)[0],
        device=env.device, dtype=torch.long,
    )

    sums = torch.zeros(nb, 10, device=env.device)
    counts = torch.zeros(nb, device=env.device)
    zero = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)
    env.reset()
    with torch.inference_mode():
        for _ in range(steps):
            action = zero
            if mode == "q":
                # The action term holds baseline + residual over the coming interval,
                # its baseline read at the interval's end; read q there too.
                ids = term.target_ids
                q = ref.sample(ref.index_of(cmd.time_since_go + env.step_dt))["q"]
                action = (q[:, ids] - term._get_ref_q()[:, ids]) / term.scale
            env.step(action)

            # Per env, off the command's own clock, so an env the terminations
            # reset is binned from its new start rather than from a global count.
            tsg = cmd.time_since_go
            phase = ref.phase(tsg)
            z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
            z_ref = ref.sample(ref.index_of(tsg))["base_pos"][:, 2]
            fz = robot.data.body_link_pos_w[:, foot_ids, 2]
            forces = env.scene.sensors["feet_ground_contact"].data.force
            planted = (forces.norm(dim=-1) > 1.0).float()
            b = (phase * nb).long().clamp(0, nb - 1)
            for k in range(nb):
                m = b == k
                counts[k] += m.sum()
                sums[k, 0] += (z[m] - z_ref[m]).abs().sum()
                sums[k, 1] += planted[m].mean(dim=1).sum()
                sums[k, 2] += fz[m].min(dim=1).values.sum()
                sums[k, 3] += fz[m].max(dim=1).values.sum()
                for j in range(6):
                    sums[k, 4 + j] += planted[m][:, j].sum()
    return sums, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--mode", choices=["q", "q_cmd"], default="q_cmd")
    args = parser.parse_args()

    import tasks
    from mjlab.envs import ManagerBasedRlEnv

    env_cfg = tasks.load_env_cfg("jumper.jump", play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = 1e9
    env_cfg.events["reset_from_reference"].params["phase_range"] = (0.0, 0.01)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    try:
        sums, counts = run(env, args.mode, args.steps)
        print(f"== mode={args.mode} ==")
        print(f"{'phase':>9} {'z_err(mm)':>10} {'planted':>8} {'minZ':>7} {'maxZ':>7}  "
              + " ".join(LEG_NAMES))
        for k in range(10):
            n = counts[k].clamp(min=1)
            per_leg = " ".join(f"{(sums[k, 4 + j] / n).item():.2f}" for j in range(6))
            print(
                f"{k / 10:4.1f}-{(k + 1) / 10:4.1f} "
                f"{(sums[k, 0] / n * 1000).item():10.1f} "
                f"{(sums[k, 1] / n).item():8.3f} "
                f"{(sums[k, 2] / n * 1000).item():7.1f} "
                f"{(sums[k, 3] / n * 1000).item():7.1f}  {per_leg}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
