"""Termination conditions for the reference-guided high jump."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .reference import ref_state


def reference_diverged(
    env,
    command_name: str = "jump",
    threshold: float = 6.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the sum of squared joint-space tracking errors exceeds the limit.

    The reference is the only target, and a large deviation means there is no value in
    continuing to sample -- stopping early gives away the hundreds of steps of dense
    imitation score to someone else, which is itself the strongest penalty signal
    (imitation weights sum to ~13.5/step, an episode is about 700 steps).
    """
    _, ref, _, _ = ref_state(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    e = ((robot.data.joint_pos - ref["q"]) ** 2).sum(dim=1)
    return e > threshold


__all__ = ["reference_diverged"]
