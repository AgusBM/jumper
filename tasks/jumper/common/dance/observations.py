"""What the dance policy sees of the choreography.

Every term here is a thin wrapper over mjlab's `MotionCommand`. They exist for one
reason: **the names**.

`scripts/export.py` keeps `_DEPLOY_TERMS`, the set of observation term names the
deployment's observation builder can construct, and refuses to export a contract
containing anything else -- a term the robot cannot build is a policy that throws
on the robot rather than in the export. That set already reserves the
reference-playback names, `clip_phase`, `ref_joint_pos`, `ref_joint_vel`,
`ref_tilt_error` and `ref_future`, and `deploy/fsm/src/obs.rs` names the same
group when it explains which terms belong to a trajectory mode it does not yet
have.

mjlab's own tracking task calls them `command`, `motion_anchor_pos_b` and
`motion_anchor_ori_b`. **None of those three are in the whitelist**, so mounting
its observation group unchanged gives a task that trains perfectly and cannot be
exported -- discovered at the end, after the training. These wrappers are the
translation, and `tests/test_dance_motion.py` pins every actor term name against
`export.py`'s set so a future term cannot quietly reintroduce the problem.

Note what is *not* here: `base_lin_vel`. It is in `export.py`'s
`_UNMEASURABLE_TERMS` -- this robot has no state estimator, its IMU gives angular
velocity only -- so it belongs to the critic, which is thrown away at export.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "FUTURE_HORIZONS",
    "clip_phase",
    "ref_future",
    "ref_tilt_error",
    "ref_joint_pos",
    "ref_joint_vel",
]

#: How far ahead `ref_future` looks, in control steps. At 50 Hz: 40 ms, 100 ms and
#: 200 ms.
#:
#: The point is anticipation. Given only the reference at the current instant a
#: policy can react but cannot prepare, and a choreography is full of moves that
#: have to be set up a beat early -- at 129.88 bpm a beat is 462 ms, so 200 ms is
#: about half a beat and covers the wind-up of a step without reaching so far
#: forward that the near-term signal is diluted.
#:
#: Three horizons rather than one because the useful lead time is not the same for
#: an arm sweep and an ankle adjustment, and rather than many because each costs a
#: full joint vector of width.
FUTURE_HORIZONS: tuple[int, ...] = (2, 5, 10)

#: mjlab's convention for a default asset (`rl/mjlab/envs/mdp/rewards.py:19`): one
#: module-level instance rather than a call in the signature's default.
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
    return cast(MotionCommand, env.command_manager.get_term(command_name))


def clip_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Where in the choreography this environment is, as [num_envs, 2] (sin, cos).

    **Its job is to locate the section, not the frame.** Over a clip of several
    minutes one sine wave resolves position only coarsely -- adjacent control steps
    differ by about 1e-4 of a cycle -- and that is the intended division of labour:
    the local shape of the motion arrives through `ref_joint_pos` and `ref_future`,
    which are exact, while this says which part of the dance is being danced. For a
    fixed choreography that is worth having, because it lets the policy specialise
    rather than treat every passage as the same tracking problem.

    sin/cos rather than the raw fraction for the reason the gait clock uses them
    (`tasks/jumper/common/mdp/phase.py`): the quantity is circular, and a raw value
    in [0, 1) puts a discontinuity at the wrap where 0.999 and 0.001 are adjacent
    in time and maximally far apart as numbers. The command loops the clip, so that
    wrap is reached.

    No noise: on the robot the controller generates this and knows it exactly, so
    corrupting it would model nothing real.
    """
    command = _command(env, command_name)
    total = max(int(command.motion.time_step_total), 1)
    angle = command.time_steps.to(torch.float32) * (2.0 * math.pi / total)
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def ref_joint_pos(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """The reference joint positions now, [num_envs, num_joints].

    **Relative to the default pose**, matching mjlab's `joint_pos_rel`, which is
    how the robot's own joint positions reach the policy. Mixing the two frames --
    the reference absolute and the measurement relative -- would leave the policy
    to learn the constant offset between them, which it can, and which quietly
    spends capacity on arithmetic instead of on dancing.
    """
    asset = env.scene[asset_cfg.name]
    return _command(env, command_name).joint_pos - asset.data.default_joint_pos


def ref_joint_vel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """The reference joint velocities now, [num_envs, num_joints]."""
    return _command(env, command_name).joint_vel


def ref_future(
    env: ManagerBasedRlEnv,
    command_name: str,
    horizons: tuple[int, ...] = FUTURE_HORIZONS,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """The reference joint positions at each horizon, [num_envs, len(horizons) * n].

    Clamped at the end of the clip rather than wrapped. Wrapping would show the
    policy the opening bars while it is finishing the closing ones, which is a
    discontinuity in the one signal whose whole purpose is to be smooth and
    predictive; holding the last frame instead says "nothing further is coming",
    which is true of the end of a dance.
    """
    command = _command(env, command_name)
    asset = env.scene[asset_cfg.name]
    last = command.motion.time_step_total - 1
    frames = [
        command.motion.joint_pos[(command.time_steps + h).clamp(max=last)]
        for h in horizons
    ]
    return torch.cat(frames, dim=-1) - asset.data.default_joint_pos.repeat(
        1, len(horizons)
    )


#: World "down", for turning a body orientation into the gravity it sees.
_GRAVITY_W = (0.0, 0.0, -1.0)


def ref_tilt_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """How far the robot's base is tilted from the reference's, [num_envs, 3].

    The reference's gravity direction minus the robot's own, both in their own
    body frames. **This is the term that makes the tracking closed-loop**:
    without it the policy is shown the reference and its own joints but never
    the difference between where it is and where it should be.

    It replaced a 9-wide `ref_error` -- mjlab's `motion_anchor_pos_b` and
    `motion_anchor_ori_b` concatenated -- and the two things it dropped were
    dropped for different reasons, both measured.

    **Position, because there is nothing to measure it with.** The robot has no
    base state estimator, so three numbers that are exact in simulation have no
    counterpart on hardware. They cost almost nothing here: the choreography's
    root travels 0.148 m in x, 0.011 in y and 0.032 in z across 235.7 s, which
    is a dance in place.

    **Yaw, because the IMU integrates it.** Gravity is the only absolute
    orientation reference this robot has, and it fixes roll and pitch only. A
    six-axis IMU's yaw is a free-running integration, and over the 235.7 s this
    clip runs its drift is plausibly the same order as the 22.3 deg of yaw the
    choreography actually uses -- so a yaw error term would be mostly drift, and
    a policy correcting it would slowly turn the robot while tracking perfectly.
    The 16.4 deg of pitch and 3.1 deg of roll are real and are kept.

    Expressing it as projected gravity is what removes the yaw: a pure rotation
    about the world's vertical leaves the gravity direction in the body frame
    unchanged, so this quantity cannot see yaw at all. That is a property of the
    formula rather than a step that could be forgotten.

    Three numbers for two degrees of freedom, which is the same redundancy
    `projected_gravity` already carries, and the deployment builds it from the
    IMU quaternion it already reads.
    """
    command = _command(env, command_name)
    g = torch.tensor(_GRAVITY_W, device=command.anchor_quat_w.device).expand(
        command.anchor_quat_w.shape[0], 3
    )
    ref = quat_apply_inverse(command.anchor_quat_w, g)
    robot = quat_apply_inverse(command.robot_anchor_quat_w, g)
    return ref - robot
