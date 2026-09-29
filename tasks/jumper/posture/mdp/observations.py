"""What the policy is shown about its posture.

Two terms, and the split between them is the asymmetric actor-critic arrangement
the rest of this repository uses:

    posture_command   the four numbers being asked for      actor and critic
    posture_state     the four numbers being achieved       critic only

**The command is an input the robot genuinely has on hardware** -- it comes down
the wire from the operator, exactly like the velocity command does -- so the
actor may depend on it.

**The measurement is not, and that is why it is privileged.** Pitch and roll a
real IMU gives directly, but the twist and the height are reconstructions from
leg kinematics and foot contact that this robot's controller does not compute,
and `scripts/export.py::_validate_measurable` exists because a policy that
depends on something the robot cannot measure is one that trains, replays and
then cannot be deployed at all. The critic is thrown away at export, so it costs
the deployed policy nothing.

That the actor is not shown its own posture is not a handicap either: it sees the
20 joint positions with five frames of history, and the posture is a function of
those. What it has to do is learn that function, which is exactly the kind of
thing the value function does not need to and the policy does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from . import state as posture_state

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def posture_command(
    env: "ManagerBasedRlEnv",
    command_name: str,
    neutral_height: float,
    mirror_kind: str | None = None,
) -> torch.Tensor:
    """The commanded (twist, pitch, roll, height), [num_envs, 4].

    `mirror_kind` is read by `common/mdp/symmetry.py` off the term's params and
    is accepted and ignored here -- the observation manager passes every param to
    the function, so a term that declares its own mirror has to take the argument.

    `mjlab.envs.mdp.generated_commands` would do exactly this and is what the
    velocity command's observation uses. It is spelled out here so that the
    height channel can be **centred on the neutral height** before it reaches the
    network: the other three are signed quantities around zero, and a raw height
    of 0.07 to 0.15 is an input whose variation is a third of its own magnitude.
    The
    observation normaliser would eventually learn that offset, and until it does
    the posture command reads as one near-constant channel and three informative
    ones.

    **`neutral_height` is a parameter and not `term.cfg.neutral_height`**, which
    is what it used to read. The value is the same either way; what changes is
    that the *contract* can see it. `scripts/export.py` writes each term's params
    into `layout.json`, and a number reached across into another manager's config
    is a number the deployment is not told about -- so a builder working from the
    contract alone fed the raw height and was a constant 0.107 m out on that
    channel, for ever, with nothing to show for it.

    That was found by `play.py --bundle` (since removed), which rebuilt the
    observation from the contract and compared: eight terms matched to the last bit and this one did
    not. `posture_state_obs` below already took the parameter, which is why it did
    not have the same hole.
    """
    cmd = env.command_manager.get_command(command_name)
    assert cmd is not None, f"command {command_name!r} not found"
    out = cmd.clone()
    out[:, 3] -= neutral_height
    return out


def posture_state_obs(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    neutral_height: float,
    mirror_kind: str | None = None,
) -> torch.Tensor:
    """The measured (twist, pitch, roll, height), [num_envs, 4].

    Centred the same way `posture_command` is, and for the same reason. The two
    are then in the same units on the same origin, so their difference -- the
    tracking error -- is something the critic can form with one subtraction
    rather than having to discover an offset first.
    """
    out = posture_state.posture_state(env, asset_cfg)
    out[:, 3] -= neutral_height
    return out


__all__ = ["posture_command", "posture_state_obs"]
