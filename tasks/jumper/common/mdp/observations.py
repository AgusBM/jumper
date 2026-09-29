"""Observation terms the jumper tasks share.

mjlab's `envs/mdp/observations.py` covers the usual proprioception -- base
velocities, projected gravity, joint positions and velocities, the last action,
the command. What it has no term for is **actuator force**, and that is the one
this robot most wants: a 1.75 N*m actuator on a 2 kg machine spends much of a
stride near its limit, and torque is where that shows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["actuator_force"]


def actuator_force(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The force each actuator is producing, [num_envs, num_actuators].

    **`actuator_force`, not `joint_torques`.** `EntityData.joint_torques` raises
    on purpose, because "joint torque" is ambiguous on a model like this one: it
    could mean the actuator's own scalar output or the generalised force it
    contributes in DoF space, and the two differ wherever a gear ratio or a
    non-trivial transmission sits between them. This is the actuator's output, in
    actuation space -- the same space the policy's actions live in, so an action
    and the force it produced are directly comparable.

    Restricted to `asset_cfg.actuator_names` when given. That matters here: the
    robot has 22 actuators and the policy drives 20 of them, the two grippers being
    held closed by their PD. Feeding their force in would add two columns that are
    a constant, and constants that vary by less than the observation normaliser's
    epsilon are the sort of input that looks harmless and quietly wastes capacity.

    On hardware this is a current measurement, scaled. Included with that in mind:
    a term that cannot be reproduced on the robot is a term that makes the policy
    undeployable, which is why the deployment contract lists the observation terms
    by name.
    """
    asset = env.scene[asset_cfg.name]
    force = asset.data.actuator_force
    if asset_cfg.actuator_ids is not None:
        return force[:, asset_cfg.actuator_ids]
    return force
