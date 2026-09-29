"""Critic-only observations: what the rewards are computed from.

The actor sees its body and the go clock (`jumper.jump`'s `jump_phase`) and
nothing else, because nothing else will exist on the robot. The critic is
thrown away at export, so it may see what the rewards read -- and should: a
value function asked to predict `apex_height`, which is paid for a peak
latched a second earlier, without being shown the latch is fitting noise.

Both declare their mirror kind in `params`, as `common/mdp/symmetry.py` allows
a term to, so the symmetry augmentation needs no entry of theirs in `common/`.
"""

from __future__ import annotations

import torch


def jump_state(
    env,
    command_name: str = "jump",
    scale: float = 0.10,
    mirror_kind: str | None = None,
) -> torch.Tensor:
    """(can_jump, in_flight, landed, own apex rise / scale): the flight
    bookkeeping `mdp/commands.py` keeps, which the height rewards are gated on.
    The rise is the robot's lowest point's, above HOME; see `mdp/commands.py`.
    It reads 0 once a second jump has forfeited it, which is when `apex_height`
    stops paying for it -- the same four columns, so a critic from before this
    reads the same shape.

    `mirror_kind` is read off the params by the symmetry augmentation, not here.
    """
    del mirror_kind
    cmd = env.command_manager.get_term(command_name)
    rise = cmd.apex_rise.clamp(min=0.0) / scale * (~cmd.second_jump).float()
    return torch.stack(
        [cmd.can_jump.float(), cmd.in_flight.float(), cmd.landed.float(), rise], dim=1
    )


def prior_target(
    env,
    command_name: str = "jump",
    action_name: str = "joint_pos",
    mirror_kind: str | None = None,
) -> torch.Tensor:
    """The recording's command relative to HOME, over the policy's joints: what
    the feedforward adds to the next target, times `feedforward`.

    Read one control step ahead, as the action term reads it: the next target is
    held over the interval that starts now, and read at that interval's end.
    """
    del mirror_kind
    cmd = env.command_manager.get_term(command_name)
    term = env.action_manager.get_term(action_name)
    ref = cmd.reference
    q_cmd = ref.sample_cmd(cmd.time_since_go + ref.t_go + env.step_dt)
    return q_cmd[:, term.target_ids] - term.offset


def feedforward(
    env, action_name: str = "joint_pos", mirror_kind: str | None = None
) -> torch.Tensor:
    """How much of the recording's command the action term is adding, 1 -> 0
    over training. The same action lands on a different target as it falls, so
    a value function that cannot see it is fitting two tasks as one."""
    del mirror_kind
    term = env.action_manager.get_term(action_name)
    return torch.full((env.num_envs, 1), term.feedforward, device=env.device)


__all__ = ["feedforward", "jump_state", "prior_target"]
