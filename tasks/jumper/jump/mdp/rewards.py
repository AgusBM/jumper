"""Imitation reward for the reference-guided high jump.

DeepMimic style: every channel is a bounded exponential kernel ``exp(-e / sigma_sq)``
taking values in (0, 1], which cannot be gamed and does not swamp the other terms
early in training the way an unbounded negative quadratic would. The reference is a
recorded MuJoCo rollout and a full score is physically attainable -- the whole scheme
rests on that.

sigma_sq is each channel's error scale (the squared error at which the kernel drops
to 1/e), derived from the reference's own amplitude rather than a hand-tuned knob.
Ported from kk-rl-lab `tasks/hexa_5d/jump/mdp/rewards_ref.py`, keeping its weight per
term.

Design rule: **the reference defines the motion, the reward only measures distance to
the reference**. Any "beyond the reference" positive score reopens the door to scoring
without jumping.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from ...common.constants import FEET
from .reference import ref_state


def ref_joint_pos(
    env,
    command_name: str = "jump",
    sigma_sq: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint-space pose tracking -- the term that actually constrains the jump pose."""
    _, ref, _, _ = ref_state(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    e = ((robot.data.joint_pos - ref["q"]) ** 2).mean(dim=1)
    return torch.exp(-e / sigma_sq)


def ref_joint_vel(
    env,
    command_name: str = "jump",
    sigma_sq: float = 40.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint velocity tracking. sigma is large: joint angular velocity approaches
    the servo limit during extension, so the squared error is naturally in the
    hundreds and a tight sigma would saturate to 0."""
    _, ref, _, _ = ref_state(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    e = ((robot.data.joint_vel - ref["qd"]) ** 2).mean(dim=1)
    return torch.exp(-e / sigma_sq)


def ref_base_height(
    env,
    command_name: str = "jump",
    sigma_sq: float = 0.0025,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base height tracking -- it carries the jump itself (squat depth, apex,
    landing height).

    sigma_sq = 0.0025 m^2: the reward drops to 1/e at a 5 cm height error.
    """
    _, ref, _, _ = ref_state(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    e = (z - ref["base_pos"][:, 2]) ** 2
    return torch.exp(-e / sigma_sq)


def ref_base_ori(
    env,
    command_name: str = "jump",
    sigma_sq: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Orientation tracking via the gravity direction in the base frame -- what the
    IMU actually observes, so it is still meaningful on the real robot."""
    _, ref, _, _ = ref_state(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    e = ((robot.data.projected_gravity_b - ref["proj_g"]) ** 2).sum(dim=1)
    return torch.exp(-e / sigma_sq)


def ref_base_lin_vel(
    env,
    command_name: str = "jump",
    sigma_sq: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base linear velocity tracking -- it carries the extension (reference peak
    1.87 m/s).

    Tracking position alone would let "slowly rise to the same height" score full,
    and that is not a jump.
    """
    _, ref, _, _ = ref_state(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    e = ((robot.data.root_link_lin_vel_w - ref["base_lin_vel"]) ** 2).sum(dim=1)
    return torch.exp(-e / sigma_sq)


def _foot_body_ids(env) -> torch.Tensor:
    """The six foot body indices, resolved once in LEGS order.

    Uses the V1.6 `FEET`, whose order matches the reference's `foot_pos`/`contact`
    columns (both walk LF, RF, LM, RM, LR, RR). The npz was recorded on the
    previous model, whose front bodies were `*F_palm_pad_b_link`; the foot ORDER is
    unchanged, so the columns line up without a remap.
    """
    if not hasattr(env, "_jump_ref_foot_ids"):
        robot: Entity = env.scene["robot"]
        ids, _ = robot.find_bodies(list(FEET), preserve_order=True)
        env._jump_ref_foot_ids = torch.as_tensor(ids, device=env.device, dtype=torch.long)
    return env._jump_ref_foot_ids


def ref_foot_pos(
    env,
    command_name: str = "jump",
    sigma_sq: float = 0.004,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Foot position tracking (base frame). Pure joint-space tracking tolerates
    mutually compensating errors that add up to a misplaced foot; this term pins
    down the leg geometry."""
    from mjlab.utils.lab_api.math import quat_apply_inverse

    _, ref, _, _ = ref_state(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    ids = _foot_body_ids(env)
    rel = robot.data.body_link_pos_w[:, ids] - robot.data.root_link_pos_w.unsqueeze(1)
    q = robot.data.root_link_quat_w.unsqueeze(1).expand(-1, rel.shape[1], -1).reshape(-1, 4)
    foot_b = quat_apply_inverse(q, rel.reshape(-1, 3)).reshape(rel.shape)
    e = ((foot_b - ref["foot_b"]) ** 2).sum(dim=-1).mean(dim=1)
    return torch.exp(-e / sigma_sq)


def ref_contact(
    env,
    command_name: str = "jump",
    sensor_name: str = "feet_ground_contact",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Fraction of feet whose contact state matches the reference schedule.

    The reference has a definite contact schedule -- six feet planted, all
    airborne, then planted again -- and when the feet leave and when they land IS
    the substance of the jump. Pose tracking only implies it indirectly: the
    policy can stay close in joint space yet take off two steps late, to which the
    exponential kernel is nearly insensitive, while the whole airborne phase is
    therefore wrong. This is the only discrete channel and it keeps signalling
    exactly at the contact transitions.
    """
    r, ref, _, _ = ref_state(env, command_name)
    del r
    sensor: ContactSensor = env.scene[sensor_name]
    f = sensor.data.force
    assert f is not None
    in_contact = (f.norm(dim=-1) > force_threshold).float()
    return (in_contact == ref["contact"]).float().mean(dim=1)


def undesired_contacts(
    env,
    sensor_name: str = "body_ground_contact",
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Fraction of non-foot geoms touching the ground (knees, forearms, chassis,
    ...), with a negative weight."""
    sensor: ContactSensor = env.scene[sensor_name]
    f = sensor.data.force
    assert f is not None
    contact = (f.norm(dim=-1) > force_threshold).float()
    return contact.mean(dim=1)


__all__ = [
    "ref_base_height",
    "ref_base_lin_vel",
    "ref_base_ori",
    "ref_contact",
    "ref_foot_pos",
    "ref_joint_pos",
    "ref_joint_vel",
    "undesired_contacts",
]
