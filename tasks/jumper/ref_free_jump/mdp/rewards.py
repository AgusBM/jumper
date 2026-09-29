"""Rewards for the reference-free high jump.

Two groups, and they are kept apart on purpose.

**The prior** (`prior_joint_pos`, `prior_joint_vel`) is the only pair of terms
that reads the recording, and it is annealed to zero weight by
`mdp/curriculum.py`. It compares the body with the recording at the same
moment -- the joint angles and the joint speeds the recording *reached*, not the
commands its controller sent -- so what it asks for is the motion, and the
policy is free to find its own commands for it on this robot.

**The objective** is everything else, and none of it knows the recording
exists. It pays for how high the robot's lowest point rises -- uncapped, so
the recording's 0.12 m is not a ceiling -- for being upright, and for standing at HOME around the jump; it
charges for drifting away from where the jump started, for touching the
ground with anything that is not a foot, and for legs touching each other. The two height terms
come in as the prior goes out, on the same schedule, so they weigh nothing
while the policy is still learning the recording's jump.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg


def _prior(env, command_name: str, asset_cfg: SceneEntityCfg, field: str, sigma_sq: float):
    """exp(-e / sigma_sq) over the recording's span, 0 outside it; e the mean
    squared gap between the body's `field` and the recording's over the joints.

    Both are read at `time_since_go`: rewards are computed after the step, on
    the state the step ended in, and that is the moment it is compared at.

    Zero before the go and after the span. Outside it the recording holds its
    own stand, which is not HOME, and `home_pose` is what speaks there.
    """
    cmd = env.command_manager.get_term(command_name)
    robot: Entity = env.scene[asset_cfg.name]
    ref = cmd.reference
    tsg = cmd.time_since_go
    want = ref.sample(ref.index_of(tsg))[field][:, asset_cfg.joint_ids]
    have = (robot.data.joint_pos if field == "q" else robot.data.joint_vel)[:, asset_cfg.joint_ids]
    e = ((have - want) ** 2).mean(dim=1)
    window = (tsg >= 0.0) & (tsg <= ref.span)
    return torch.exp(-e / sigma_sq) * window.float()


def prior_joint_pos(
    env, asset_cfg: SceneEntityCfg, command_name: str = "jump", sigma_sq: float = 0.20
) -> torch.Tensor:
    """The joint angles against the recording's, `q`."""
    return _prior(env, command_name, asset_cfg, "q", sigma_sq)


def prior_joint_vel(
    env, asset_cfg: SceneEntityCfg, command_name: str = "jump", sigma_sq: float = 40.0
) -> torch.Tensor:
    """The joint speeds against the recording's, `qd`. The recording's peak is
    35 rad/s at the push, so the kernel is wide: a tight one would read 0
    everywhere the robot is not already pushing."""
    return _prior(env, command_name, asset_cfg, "qd", sigma_sq)


def apex_height(env, command_name: str = "jump", scale: float = 0.10) -> torch.Tensor:
    """The episode's own apex -- how far its lowest point, of six feet and the
    base, rose above HOME; see `mdp/commands.py` -- in units of `scale`, paid
    every step once it has landed. Linear and uncapped: higher is always worth
    more.

    **Until it jumps again.** On the step after a second jump is seen this
    returns minus every payment it has made this episode, and 0 from then on, so
    an episode that jumped twice is paid nothing for height. The payments are
    counted in steps and the apex is latched, so the clawback is exact at a
    fixed weight; while the schedule is still raising it, the weight at the
    clawback exceeds the ones paid at by under 1% over a whole episode.
    """
    cmd = env.command_manager.get_term(command_name)
    rise = cmd.apex_rise.clamp(min=0.0) / scale
    paying = (cmd.landed & ~cmd.second_jump).float()
    return rise * (paying - cmd.clawback_pending.float() * cmd.paid_steps)


def flight_height(env, command_name: str = "jump", scale: float = 0.10) -> torch.Tensor:
    """How far the robot's lowest point is above HOME while in the episode's own
    flight -- the apex term's dense half, so that a higher arc pays before the
    landing does."""
    cmd = env.command_manager.get_term(command_name)
    rise = cmd.lowest_rise().clamp(min=0.0) / scale
    return rise * cmd.in_flight.float()


def home_pose(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str = "jump",
    sigma_sq: float = 0.05,
) -> torch.Tensor:
    """Standing at HOME once the jump is done: from its landing, or from the end
    of the recording's span for an episode that never had a jump of its own.

    HOME is where the jump starts and where the episode ends -- `back_home`
    ends it there -- so it is what the robot is asked to come back to. Before
    the landing nothing asks for it: the crouch, the push and the flight are
    the policy's to shape.

    `asset_cfg` has no default on purpose. The reward manager resolves joint
    names into ids only for a `SceneEntityCfg` it finds in the term's params; one
    left in a default argument is never resolved and reads every joint, the
    grippers included, without a word.
    """
    cmd = env.command_manager.get_term(command_name)
    robot: Entity = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    e = ((q - home) ** 2).mean(dim=1)
    return torch.exp(-e / sigma_sq) * cmd.jump_done.float()


def upright(env, sigma_sq: float = 0.10) -> torch.Tensor:
    """exp(-|g_xy|^2 / sigma_sq), g the gravity direction in the base frame."""
    robot: Entity = env.scene["robot"]
    g = robot.data.projected_gravity_b[:, :2]
    return torch.exp(-(g**2).sum(dim=1) / sigma_sq)


def horizontal_drift(env, scale: float = 0.10) -> torch.Tensor:
    """(d / scale)^2, d the base's horizontal distance from the env origin.

    Every spawn is at the origin -- HOME's by `reset_scene_to_default`, the
    recording's by RSI, which itself strays 7.9 mm at most and ends 0.7 mm out --
    so this is the distance from where the jump started. A vertical jump has no
    direction to travel in.
    """
    robot: Entity = env.scene["robot"]
    d = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return (d**2).sum(dim=1) / scale**2


def illegal_contact(
    env, sensor_name: str = "body_ground_contact", force_threshold: float = 1.0
) -> torch.Tensor:
    """1 while any part that is not a foot is on the ground, 0 otherwise.

    The sensor holds every collision geom but the six feet (`env_cfg.py` builds
    it by exclusion, not by listing), so this is every part a hexapod should not
    stand, push or land on. Binary on purpose: a knee down is as illegal as the
    belly, and the mean over the 35 parts it replaced made one of them worth
    1/35 of its weight -- nothing, against a jump paying over 4 a step.
    """
    sensor = env.scene[sensor_name]
    f = sensor.data.force
    assert f is not None
    return (f.norm(dim=-1) > force_threshold).any(dim=1).float()


def leg_collision(
    env, sensor_name: str = "self_collision", force_threshold: float = 1.0
) -> torch.Tensor:
    """1 in a step in which any two legs pushed on each other harder than
    `force_threshold` in any of its physics substeps, 0 otherwise.

    Which pairs can touch at all is the collision masks' business
    (`common/constants.py`): different legs collide, one leg's own links and the
    base never do, so the whole-robot sensor sees leg against leg and nothing
    else. Read over the substeps because a knock that starts and ends between
    two control steps is still a knock. Binary for the reason `illegal_contact`
    is: two calves leaning on each other are as much a collision as six.
    """
    sensor = env.scene[sensor_name]
    h = sensor.data.force_history
    assert h is not None, f"{sensor_name} keeps no force history"
    return (h.norm(dim=-1) > force_threshold).flatten(1).any(dim=1).float()


__all__ = [
    "apex_height",
    "flight_height",
    "home_pose",
    "horizontal_drift",
    "illegal_contact",
    "leg_collision",
    "prior_joint_pos",
    "prior_joint_vel",
    "upright",
]
