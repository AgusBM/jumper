#!/usr/bin/env python3
"""Pure reference-motion replay: loads no policy and plays the npz trajectory as-is.

    python tools/checks/jump_ref_play.py

The reference motion lifts the two middle legs (LM/RM) 0.112 s after go and holds them up until
push-off at 0.3 s — this is the trajectory generator's own loading design. This script exists to
see that with your own eyes: what is on screen is the npz motion, independent of any policy.

It takes the same code path as training: the residual action term's position target is
"reference joint angles + 0.25 × action", so feeding a zero action is pure PD tracking of the
reference. The episode resets every `--loop` seconds for looping playback.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--loop", type=float, default=3.0, help="replay period (s)")
    args = parser.parse_args()

    import torch

    import tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjrl.viewer.live import LiveViewer

    env_cfg = tasks.load_env_cfg("jumper.jump", play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.loop
    # Spawn from the motion start to watch a full crouch-jump-land
    env_cfg.events["reset_from_reference"].params["phase_range"] = (0.0, 0.01)

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    try:
        action_dim = env.action_space.shape[-1]
        zero = torch.zeros(args.num_envs, action_dim, device=env.device)

        obs, _ = env.reset()
        with LiveViewer(env.sim, env_index=0, fps=30.0):
            while True:
                with torch.inference_mode():
                    obs, _, _, _, _ = env.step(zero)
    finally:
        env.close()


if __name__ == "__main__":
    main()
