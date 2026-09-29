"""How an episode of jumper.ref_free_jump ends early: back at HOME.

The episode is the jump and the return: it starts with the robot at HOME and
the jump under way, and it is over once the jump is done and the robot has
stood at HOME for `hold_steps` in a row -- see `mdp/commands.py` for what
counts. `env_cfg.py` marks it a time-out, so PPO bootstraps from the value of
the state it ends in: the landed apex goes on paying per step, and an episode
that ended by coming home must not lose that stream to one that stayed away.
"""

from __future__ import annotations

import torch


def back_home(env, command_name: str = "jump", hold_steps: int = 20) -> torch.Tensor:
    """True once the jump is done and the robot has been back at HOME for
    `hold_steps` consecutive control steps."""
    cmd = env.command_manager.get_term(command_name)
    return cmd.home_steps >= hold_steps


__all__ = ["back_home"]
