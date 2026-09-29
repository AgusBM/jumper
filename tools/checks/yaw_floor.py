#!/usr/bin/env python3
"""Measure the irreducible part of the tracking error, which is what `gate_floor`
and `ang_floor` in `tasks/jumper/common/mdp/curriculum.py` are supposed to be.

**Why this exists.** Both curricula promote when the error falls under
`floor + ratio * range`. The floor pays for the part of the error that does not
scale with the command range and that no amount of learning removes: the gait's
own wobble, and the exploration noise PPO adds to every action it collects a
sample with. Set it too low and the bar sits under the noise, the run stalls at
a level, and nothing reports a problem -- which happened on the linear axis for
37500 iterations, and is happening on the angular axis now:

    measured on logs/jumper/jumper.tripod/2026-09-10_16-10-18 at iteration 15697
    ang_err 0.2874 against ang_err_bar 0.2375, terrain parked at level 3.6

`lin_floor = 0.03` came from a measurement of this shape. `ang_floor = 0.05` did
not -- the module says so itself: it is "provisional and deliberately
conservative", taken from a standing reading of 0.0417 because every other yaw
number available at the time came from a policy trained against a yaw reward that
returned zero. That policy no longer exists. This script is the re-measurement
that note asks for.

**What it measures.** The same quantity the curriculum reads, in the same units:
the mean per-step absolute error over an episode window. mjlab accumulates
`|cmd - actual| / max_command_step` every step and the curricula multiply back by
`max_command_step / steps`, so it is simply a mean, in m/s and rad/s.

Four conditions, and the differences between them are the answer:

    standing, deterministic   the gait's own ripple: the command is exactly zero,
                              so every rad/s of yaw is the robot's own wobble
    standing, stochastic      the same plus exploration noise, with no tracking
                              asked for -- this is the clean floor candidate
    moving,   deterministic   what the policy can actually track
    moving,   stochastic      what training collects, and what the curriculum
                              compares against its bar

`stochastic - deterministic` is the exploration component; the standing pair is
the part that does not scale with the range. A floor should cover what the robot
cannot remove and no more, or it stops being a bar at all.

**It does not change anything.** Read-only: it builds an environment, runs a
policy and prints numbers.

Usage:

    python tools/checks/yaw_floor.py --checkpoint logs/.../model_15650.pt

Defaults to `native:cpu` with few environments, because the machine this is
wanted on is usually the machine that is training. `--backend warp --device
cuda:0 --num-envs 512` when the GPU is free; the numbers are the same and it
takes seconds rather than minutes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _measure(env, wrapped, policy, steps: int, warmup: int):
    """Mean per-step |error| for both axes, over `steps` after `warmup`.

    The warm-up matters: `env.reset()` puts every robot in its home pose with a
    freshly sampled command, and the first tenth of a second of settling is not
    what the metric sees in training, where resets are staggered across a
    population that has been running for thousands of episodes.
    """
    import torch

    obs = wrapped.get_observations()
    lin_sum = ang_sum = 0.0
    counted = 0
    with torch.inference_mode():
        for i in range(warmup + steps):
            obs, _, _, _ = wrapped.step(policy(obs))
            if i < warmup:
                continue
            cmd = env.command_manager.get_command("twist")
            robot = env.scene["robot"]
            lin = torch.norm(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=-1)
            ang = torch.abs(cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
            lin_sum += float(lin.mean())
            ang_sum += float(ang.mean())
            counted += 1
    return lin_sum / counted, ang_sum / counted


def _build(task: str, standing: bool, res, checkpoint: Path | None):
    """An environment and a (deterministic, stochastic) policy pair."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    import tasks

    cfg = tasks.load_env_cfg(task)
    cfg.scene.num_envs = res.num_envs
    # **The curriculum is removed, not left running.** It would move the ranges
    # and the std under the measurement, and the point here is one controlled
    # condition per number. The config's own ranges are the curriculum's last
    # level, which is where a stalled run is sitting anyway.
    cfg.curriculum = {}
    if standing:
        # Every environment commanded to stand, which is the mechanism the command
        # term already has -- it writes the command to exactly zero. Forcing the
        # ranges to zero instead would leave `is_standing_env` false and the
        # observation subtly different from a real standing env.
        cfg.commands["twist"].rel_standing_envs = 1.0

    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=tasks.load_agent_cfg(task).clip_actions)

    if checkpoint is None:
        shape = env.action_space.shape

        def zero(_obs):
            return torch.zeros(shape, device=env.device)

        return env, wrapped, zero, zero

    from dataclasses import asdict

    runner_cls = tasks.load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(tasks.load_agent_cfg(task)), device=res.device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True,
                map_location=res.device)
    deterministic = runner.get_inference_policy(device=res.device)

    def stochastic(obs):
        # What training actually collects. `alg.act` samples from the policy's
        # distribution; the inference policy returns its mean. The gap between
        # them is the exploration noise this script exists to size.
        return runner.alg.act(obs)

    return env, wrapped, deterministic, stochastic


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="jumper.tripod")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="policy to measure. Omitted, a zero-action agent is used, "
                         "which measures the ripple of a robot that does nothing")
    ap.add_argument("--backend", default="native")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()

    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    res = resolve(backend=args.backend, device=args.device, num_envs=args.num_envs)
    use_backend(res)
    print(f"[yaw_floor] backend={res.backend} device={res.device} "
          f"num_envs={res.num_envs} steps={args.steps} (+{args.warmup} warm-up)")
    if args.checkpoint is None:
        print("[yaw_floor] no --checkpoint: measuring a zero-action agent")

    rows = []
    for standing in (True, False):
        env, wrapped, deterministic, stochastic = _build(
            args.task, standing, res, args.checkpoint
        )
        try:
            for label, policy in (("deterministic", deterministic),
                                  ("stochastic", stochastic)):
                lin, ang = _measure(env, wrapped, policy, args.steps, args.warmup)
                rows.append(("standing" if standing else "moving", label, lin, ang))
        finally:
            env.close()

    print(f"\n{'command':<10}{'policy':<16}{'lin err (m/s)':>16}{'ang err (rad/s)':>18}")
    print("-" * 60)
    for cmd, pol, lin, ang in rows:
        print(f"{cmd:<10}{pol:<16}{lin:>16.4f}{ang:>18.4f}")

    by = {(c, p): (lin, ang) for c, p, lin, ang in rows}
    ripple = by[("standing", "deterministic")]
    rest = by[("standing", "stochastic")]
    move_d = by[("moving", "deterministic")]
    move_s = by[("moving", "stochastic")]

    print("\nwhat the differences mean")
    print(f"  gait ripple, no command      lin {ripple[0]:.4f}   ang {ripple[1]:.4f}")
    print(f"  + exploration, no command    lin {rest[0]:.4f}   ang {rest[1]:.4f}"
          f"   <- floor candidate")
    print(f"  exploration alone (standing) lin {rest[0] - ripple[0]:+.4f}   "
          f"ang {rest[1] - ripple[1]:+.4f}")
    print(f"  exploration alone (moving)   lin {move_s[0] - move_d[0]:+.4f}   "
          f"ang {move_s[1] - move_d[1]:+.4f}")
    print(f"  what the curriculum sees     lin {move_s[0]:.4f}   ang {move_s[1]:.4f}")

    from tasks.jumper.common.mdp.curriculum import ANG_LEVELS, LEVELS

    lin_bar = 0.03 + 0.25 * LEVELS[-1]
    ang_bar = 0.05 + 0.25 * ANG_LEVELS[-1]
    print("\nagainst the terrain curriculum's bars (up_ratio 0.25, top level)")
    print(f"  linear   {move_s[0]:.4f} vs bar {lin_bar:.4f}   "
          f"{'PASS' if move_s[0] < lin_bar else 'FAIL'}")
    print(f"  angular  {move_s[1]:.4f} vs bar {ang_bar:.4f}   "
          f"{'PASS' if move_s[1] < ang_bar else 'FAIL'}")
    print("\n  A floor has to cover the 'no command' row and no more: that is the "
          "part\n  no policy can remove. What is left over is what the bar is "
          "entitled to ask for.")


if __name__ == "__main__":
    main()
