"""Reward terms only the dance tasks use.

Three of the four things asked of this task beyond following the choreography --
not falling, low energy, smooth motion -- are already covered by terms that exist:
early termination, `mdp.electrical_power_cost`, and `mdp.action_rate_l2` with
`mdp.action_acc_l2`. The fourth, **keeping torque peaks small**, has no term
anywhere in mjlab or in this repository, and the obvious candidate is the wrong
one. That is what this module is for.

The joint-space tracking terms are here for a different reason: mjlab's tracking
recipe has none. All six of its terms are body or root quantities, which is the
right abstraction for a retargeted humanoid clip and leaves a gap on a robot with
this much redundancy -- see `motion_joint_pos_error_exp`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand

from ..actuator import CONTINUOUS_TORQUE, PLATEAU_TORQUE

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "actuator_power_cost",
    "motion_joint_pos_error_exp",
    "motion_joint_vel_error_exp",
    "peak_actuator_force",
    "torque_headroom_cost",
]

#: mjlab's convention for a default asset (`rl/mjlab/envs/mdp/rewards.py:19`): one
#: module-level instance rather than a call in the signature's default.
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _motion(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
    return cast(MotionCommand, env.command_manager.get_term(command_name))


def torque_headroom_cost(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    threshold: float = CONTINUOUS_TORQUE,
) -> torch.Tensor:
    """How far the actuators went above the torque they can hold, squared and summed.

        cost = sum_j  relu(|tau_j| - threshold)^2          [num_envs]

    Zero below the threshold, quadratic above it.

    ## Why not `joint_torques_l2`

    The obvious term is mjlab's `joint_torques_l2`, mean-square actuator force. It
    does not do this job, and it is worth being precise about why, because it
    *looks* like it should.

    Most of the torque this robot produces is spent standing up. At `HOME` the
    worst joint (LM_knee) holds 0.2296 N*m against gravity and it has no choice
    about it (measured on the V1.6 model; `constants.py`, and
    `assets/jumper/motor/README.md`). The figure was 0.2499 on the model before
    it; the argument below does not turn on which. A mean-square penalty charges for
    that every step of every episode, so the cheapest way to satisfy it is to sag
    -- to find a posture that needs less holding torque -- which is a direct
    subtraction from the choreography. Meanwhile it barely notices a 50 ms spike,
    because one control step in twenty contributes a twentieth of the mean. It
    charges hardest for the thing the robot cannot avoid and least for the thing
    being asked about.

    ## Why this threshold

    The servo has two ratings, not one. `PLATEAU_TORQUE` (1.7464 N*m) is a **peak**
    and is holdable for 300 ms; after that the actuator's thermal state derates the
    ceiling to `CONTINUOUS_TORQUE` (1.2 N*m). Both are measured, and
    `tasks/jumper/common/actuator.py` models the transition.

    So there is a physically meaningful line, and it is not zero: torque below
    1.2 N*m costs the servo nothing it cannot sustain indefinitely, and torque
    above it is borrowed against a 300 ms budget. "Keep the peaks small" means
    "stay under the rating you can hold", and that is exactly what this measures --
    which is also why it does not fight the dance. The standing cost sits at 14% of
    the plateau, far below the threshold, so a policy holding a pose pays nothing
    at all.

    Squared rather than linear so that one joint far over costs more than four
    joints slightly over. The failure being prevented is a single actuator
    saturating through a fast move, not a general warmth.

    ## What it reads

    `actuator_force`, not `qfrc_actuator`: the actuator's own scalar output, in the
    space the policy acts in. `EntityData.joint_torques` raises on purpose because
    "joint torque" is ambiguous on this model -- see the note in
    `tasks/jumper/common/mdp/observations.py`.
    """
    asset = env.scene[asset_cfg.name]
    force = asset.data.actuator_force
    if asset_cfg.actuator_ids is not None:
        force = force[:, asset_cfg.actuator_ids]
    over = (force.abs() - threshold).clamp(min=0.0)
    return torch.sum(torch.square(over), dim=1)


def actuator_power_cost(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mechanical power the actuators are putting out, [num_envs], in watts.

        cost = sum_j  max(tau_j * qd_j, 0)

    ## Why this exists rather than `mdp.electrical_power_cost`

    mjlab has exactly this term already, and **it cannot run on the native
    backend**. It reads `asset.data.qfrc_actuator`, which `native_sim.py` does not
    provide -- the backend exposes `actuator_force` and raises `AttributeError`
    naming the missing field for anything else. So the task would train on warp and
    die on the first step under `--backend native`.

    That is the seam's characteristic failure and it is easy to walk into, because
    a two-backend repository lets a task be verified on one of them and look
    finished. This one was: the crash arrived from a real run, not from the tests.

    `actuator_force` is the actuator's own scalar output rather than the
    generalised force it contributes in DoF space. For this robot the two describe
    the same thing -- one direct-drive actuator per joint, and `Entity` reports
    `actuator_names == joint_names` exactly, so column j of one lines up with
    column j of the other. It is also the same signal `torque_headroom_cost` reads,
    so the energy and peak-torque terms are now measured off one quantity rather
    than two that could drift apart.

    The positive part only, following mjlab: negative mechanical power is the joint
    being back-driven, and charging for it would penalise the robot for absorbing
    energy on a landing -- which is the behaviour worth having.
    """
    asset = env.scene[asset_cfg.name]
    force = asset.data.actuator_force
    vel = asset.data.joint_vel
    if asset_cfg.actuator_ids is not None:
        force = force[:, asset_cfg.actuator_ids]
        vel = vel[:, asset_cfg.actuator_ids]
    return torch.sum(torch.clamp(force * vel, min=0.0), dim=1)


def peak_actuator_force(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """The largest actuator force any joint is producing, [num_envs], in N*m.

    A metric, not a reward. Mounted with `MetricsTermCfg(reduce="max")` it reports
    the episode's true peak torque, which is the quantity `torque_headroom_cost` is
    aimed at and which no reward can report: a reward is a sum over the episode, so
    a policy that halves its peak and doubles its time above threshold logs the
    same number.

    Read it against the two ratings it sits between --
    `CONTINUOUS_TORQUE` = %.4f N*m, holdable indefinitely, and
    `PLATEAU_TORQUE` = %.4f N*m, the most the hardware can ever produce.
    """
    asset = env.scene[asset_cfg.name]
    force = asset.data.actuator_force
    if asset_cfg.actuator_ids is not None:
        force = force[:, asset_cfg.actuator_ids]
    return force.abs().amax(dim=1)


peak_actuator_force.__doc__ = peak_actuator_force.__doc__ % (
    CONTINUOUS_TORQUE,
    PLATEAU_TORQUE,
)


def motion_joint_pos_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Track the choreography in **joint space**, as `exp(-mean_sq_error / std^2)`.

    mjlab's tracking recipe has six terms and every one of them is a body or a
    root quantity: anchor position and orientation, and per-body position,
    orientation, linear and angular velocity. For a retargeted humanoid clip
    that is the right abstraction -- what matters is where the limbs are in
    space, not which joint angles got them there.

    **It leaves a gap on this robot.** Twenty-two joints put many configurations
    on nearly the same set of body-link poses, so a policy can satisfy every
    body term while the legs are folded differently than the clip asks. For
    walking that is fine and often desirable. For a dance it is the thing
    being watched: the shape of the limbs *is* the choreography.

    The command already carries both sides of this -- `joint_pos` from the clip
    and `robot_joint_pos` from the simulation -- and already reports
    `error_joint_pos` as a metric. Only the reward was missing.

    All twenty-two joints, including the two fingers. `env_cfg.py` drives them
    deliberately (the choreography sweeps them +/-0.262 rad) and the note there
    anticipates exactly this term: excluding them would leave the reference
    asking for finger motion with no channel to produce it, which is a floor in
    this reward that no log would attribute to the grippers.

    The kernel and its shape are mjlab's, so the two sets of terms compose
    without a second convention.
    """
    command = _motion(env, command_name)
    error = torch.square(command.joint_pos - command.robot_joint_pos)
    return torch.exp(-error.mean(dim=-1) / std**2)


def motion_joint_vel_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Track the choreography's joint **velocities**, same kernel.

    Position alone is satisfied by a policy that reaches every pose late and
    slowly, which reads as a robot doing the right moves without the music.
    Velocity is where the timing lives, and on a clip at 129.88 bpm the timing
    is most of what makes it a dance.

    It pulls against `action_rate_l2` and `action_acc_l2`, deliberately and not
    accidentally: those suppress motion that is fast for no reason, and this
    asks for motion that is fast for a reason. The clip's joint velocity peaks
    at 12.9 rad/s, so there is real signal for them to disagree about, and the
    weights are where that argument is settled -- see `env_cfg.py`.
    """
    command = _motion(env, command_name)
    error = torch.square(command.joint_vel - command.robot_joint_vel)
    return torch.exp(-error.mean(dim=-1) / std**2)
