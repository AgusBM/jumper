"""Reward terms only `jumper.five_foot` uses.

Everything the family shares -- the gating (`cycling_gate`, `moving_gate`), the
phase matching (`phase_match`) and the contact-state normalisation
(`contact_state`) -- comes from `tasks/jumper/common/mdp/rewards.py`, exactly as the
three six-foot gait tasks take it. What is here is what changes when the
left-front leg stops being a leg.

## The port, and what it kept

kk-rl-lab's five-foot task (`tasks/hexa_5d/locomotion_5foot/mdp/rewards_cfg.py`)
carries **41 reward terms at the commit this was ported from** (`fb93b00`): 11
inherited from Isaac Lab's stock velocity `RewardsCfg`, two of which the task
re-declares, plus 32 of its own. **Eight are at weight 0**, so 33 are live.

Counted rather than remembered, because the number moved a lot: the file grew
from 23 terms to 41 over the 24 commits of that branch, and an earlier reading of
it here said "34, seven at zero" -- which is exactly commit `2049afd`, eighteen
commits in, and not the tip. The count at the initial commit was 27.

That growth is the thing to notice, not the endpoint. Seven of the eight
zero-weight terms are experiments switched off rather than removed
(`feet_air_time_max`, `mid_roll_symmetry`, `rear_stride_penalty_lr` / `_rr`,
`feet_contact_duty`, `energy`, `flat_orientation_l2`), each still carrying its
parameters and its rationale.

The other reason the list is that long is structural: Isaac Lab's stock velocity
rewards have no foot-clearance, foot-slip, swing-height, soft-landing or posture
terms, so the task had to supply them. mjlab's does, and the family has already
calibrated them for this robot, so most of that list is either already present or
already covered:

| kk-rl-lab term | here |
|---|---|
| track_lin_vel_xy / ang_vel_z / lin_vel_y | `track_linear_velocity`, `track_yaw_velocity` |
| flat_orientation / track_base_orientation | `upright` |
| ang_vel_xy_l2, pitch/roll oscillation | `body_tilt_rate` (deadbanded), `standing_sway` |
| stand_still, stand_still_joint_vel, joint deadzones | `pose` (variable_posture) |
| feet_slide, feet_air_time, feet_air_time_max | `foot_slip`, `air_time`, `foot_clearance`, `foot_swing_height` |
| action_rate_l2, dof_acc, dof_torques, energy, braking_work | `action_rate_l2` |
| base_pose command tracking (pitch/roll/twist/height) | no such command here |
| diagonal_gait | `five_foot_gait` below |
| feet_duty_balance, mid_rear_duty_match | `stance_duty_balance` below |
| left_right_symmetry | `group_load_balance` below -- by gait group, not by side |
| feet_home_position | `foot_home_position` below |
| forward_pitch | `nose_pitch` below -- one-sided twice, now symmetric |
| bad_base_height, base_height_target, base_height_smooth | `low_stance` below |
| undesired_contacts | a contact sensor on the non-foot geoms + `self_collision_cost` |
| peak_torque_margin | dropped; see the note in `env_cfg.py` |

So six functions live here, and the reason each survived the crossing is that it
scores something the six-foot skeleton has no term for **because a six-foot robot
does not have the problem**.

## Positive terms and the standing-still basin

Two of the six are positive, and on this robot that needs justifying every time.
A hexapod is statically stable, so standing still is free, and the family's
history is a series of runs that discovered exactly that (the measured breakdown
is in `common/velocity_env.py`). `group_load_balance` and `stance_duty_balance`
are both **perfect while standing** -- a robot that never moves has ideal load
symmetry and ideal duty balance -- so ungated they would each be a constant added
to the do-nothing score. kk-rl-lab runs them ungated at weights 3.0 and 3.0 of a
9.0 tracking budget. Here both are multiplied by `moving_gate`, so they are
exactly zero for a standing command and cannot tilt the walk/stand trade-off at
all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from ...common.mdp.rewards import (
    contact_state,
    moving_gate,
    phase_match,
)

# **The pitch/roll convention lives in one place.** These terms and the attitude
# command have to agree about which sign is nose-down, and a sign that disagrees
# still runs, still produces a plausible number and simply pushes the robot the
# wrong way -- the failure `nose_pitch` was already written twice for.
from .pose_command import PITCH, ROLL, TWIST, base_pitch_roll

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

#: mjlab's own idiom for a default entity (see `velocity/mdp/rewards.py`): one
#: module-level instance rather than a constructor call in a default argument,
#: which would build a new one per call and per import.
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# ── The gait grouping ─────────────────────────────────────────────────────
#
# Indices into `FIVE_FOOT_LEGS` = (RF, LM, RM, LR, RR) -- **not** into the
# family's six-leg `LEGS`. The contact sensor has five columns in this task, so
# these are the columns.
#
# **Two diagonal pairs of the middle and rear legs, and RF in neither.** It was
# the tripod grouping with the carried leg deleted -- {RF, LM, RR} against
# {RM, LR} -- and the policy trained on it did not walk it. Measured on
# `2026-09-22_15-31-02/model_16997`, contact lost counted as swing, walking at
# 0.4 m/s: RF swung inside every RR swing (100%) but lifted 5.5 mm for 64 ms,
# where LM and RR lifted 23 and 13 mm for 81 and 104 ms; backing up and turning,
# it lifted 2-3 mm for 25-28 ms and sat out a fifth of group A's swings. The
# machine was walking LM+RR against RM+LR and tapping RF to collect the grouping.
#
# The tap was the physics winning. With LF carried, RF is the only front support,
# and `grasp_pose.py --stance` on the stance these groups start from:
#
#     swing           stance          margin
#     RF+LM+RR        RM+LR           -85.7 mm    the old group A: a line, 86 mm off
#     LM+RR           RF+RM+LR        +42.6 mm    group A now
#     RM+LR           RF+LM+RR        +86.3 mm    group B, unchanged
#     RF alone        LM+RM+LR+RR     -18.4 mm
#
# Take RF out of the swing and both halves of the alternation stand on a
# triangle. RF still has to step -- it is a leg, and the trunk moves over it --
# and it steps **on its own, with the four planted**: a third phase, 2+2+1.
# With a group it stands on a line 86 mm from the centre of mass, which tips the
# trunk at about 45 rad/s^2 -- 15 mm of drop at RF within 60 ms, which is why the
# lifts it made there were 2-5 mm. Alone it is 18 mm short of static, and 30 mm
# of trunk shift backwards makes it +5 mm: a step that can be made, where the
# other two cannot. `five_foot_gait` scores that phase as it scores the other
# two, and `rf_step_window` charges RF for lifting in either group's.
GROUP_A: tuple[int, ...] = (1, 4)     # LM, RR
GROUP_B: tuple[int, ...] = (2, 3)     # RM, LR

#: RF's column in `FIVE_FOOT_LEGS`: the leg that steps alone.
RF_FOOT: int = 0

#: The legs the gait terms score, in `FIVE_FOOT_LEGS` column order: every leg but
#: RF. `five_foot_gait` matches and gates on these columns only, so RF's stance
#: and swing neither earn nor cost it anything.
GAIT_FEET: tuple[int, ...] = tuple(sorted(GROUP_A + GROUP_B))

__all__ = [
    "GAIT_FEET",
    "GROUP_A",
    "GROUP_B",
    "RF_FOOT",
    "body_height_hold",
    "body_roll",
    "body_twist",
    "feet_contact_without_cmd",
    "feet_still_when_standing",
    "five_foot_gait",
    "foot_home_position",
    "group_load_balance",
    "low_stance",
    "nose_pitch",
    "rf_drag",
    "rf_step_window",
    "stance_duty_balance",
    "track_body_pose",
]


def _gait_cycling_gate(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    feet: tuple[int, ...],
    max_air_time: float,
    max_contact_time: float,
) -> torch.Tensor:
    """`common/mdp/rewards.py::cycling_gate` over `feet` only; a [num_envs] 0/1 gate.

    The family's gate takes every column, so under it RF standing longer than
    `max_contact_time` would zero the gait reward -- which is what made RF tap:
    a leg that must not lift was being charged for not lifting. Same test, same
    thresholds, on the legs the gait is made of. The shared gate is left as it is:
    the other jumper tasks walk every leg.
    """
    sensor = env.scene.sensors[sensor_name]
    contact = contact_state(env, sensor_name)  # [B, F], every column
    t_air, t_con = sensor.data.current_air_time, sensor.data.current_contact_time
    assert t_air is not None and t_con is not None, (
        f"sensor {sensor_name!r} does not track air and contact time"
    )
    cols = list(feet)
    airborne = (t_air.reshape(contact.shape) * (1.0 - contact))[:, cols]
    stance = (t_con.reshape(contact.shape) * contact)[:, cols]
    return (
        (airborne.amax(dim=1) < max_air_time) & (stance.amax(dim=1) < max_contact_time)
    ).float()


def five_foot_gait(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.05,
    sigma: float = 0.7,
    max_air_time: float = 0.5,
    max_contact_time: float = 1.0,
) -> torch.Tensor:
    """Reward the three phases of a 2+2+1 gait: A in the air, B, or RF alone.

        reward = max(match(A airborne), match(B airborne),       over GAIT_FEET
                     [RF airborne] * match(all four planted))
                 * moving_gate * cycling_gate(GAIT_FEET)         in (0, 1]

    `phase_match` scores by **how many legs are in the wrong state**, so the
    grouping binds: any wrong grouping is at least two legs wrong and scores
    0.0003 at the default sigma, where the right one scores 1.0. That is what
    kk-rl-lab's `|mean(A) - mean(B)|` cannot do -- a mean dilutes the grouping
    away, and `common/mdp/rewards.py::phase_match` has the measurement showing a
    plainly wrong grouping collecting 37.5% under that form.

    **RF is outside both groups' match and outside the gate.** Scored over all five
    columns, the match required RF airborne with one group and planted with the
    other, and the family's `cycling_gate` zeroed the term whenever RF stood longer
    than `max_contact_time`; together they made RF lift once a cycle whether or
    not the trunk could stand without it, and it did -- by 2-5 mm, for one to three
    control steps (`GROUP_A` has the measurement and the support table). Here the
    groups are matched over `GAIT_FEET` and so is the gate (`_gait_cycling_gate`),
    so RF's state neither earns nor costs a group's phase anything -- lifting in
    one is `rf_step_window`'s to charge.

    **The third phase is RF's step, and it has to be scored here.** With only the
    two groups, the four planted together is two legs wrong for either and scores
    0.0003 -- exactly the window RF is meant to step in. A charge on RF lifting
    with a group would then leave it nowhere to step without losing something,
    and a leg with nowhere to step drags. So the four planted scores in full --
    **while RF is off the ground**, and only then: all five down is standing, and
    paying for it would pay the policy to stop walking.

    **No clock, unlike the other three gait tasks**, and the reason is measured
    rather than stylistic: a fixed-frequency clock commands a cadence on a
    schedule, which is the mistake `jumper.tripod`'s notes describe costing 37500
    iterations -- a cadence the machine cannot hold is a reward that can never be
    collected. Taking `max` leaves the policy to find when to do it, and the gate
    still forbids both ways of cheating -- a group held up (`max_air_time`) and
    marching in place with the rest planted (`max_contact_time`).

    If the gait term stays low, the measured alternative is a **ripple order** --
    one leg airborne at a time, which the stance table scores at +96.8 / +87.0 /
    +85.2 / +65.4 mm for RM / LR / RR / LM on the model before the bottom-shell
    revision. It is more stable and slower, and it is a different task rather than
    a tuning of this one, exactly as `jumper.ripple` is a different task from
    `jumper.tripod`. This term scores a single-leg swing at only 0.13 (one leg
    wrong), so the two cannot be blended.
    """
    full = contact_state(env, sensor_name)
    contact = full[:, list(GAIT_FEET)]
    a = tuple(GAIT_FEET.index(i) for i in GROUP_A)
    b = tuple(GAIT_FEET.index(i) for i in GROUP_B)
    rf_step = (1.0 - full[:, RF_FOOT]) * phase_match(contact, (), sigma)
    match = torch.maximum(
        torch.maximum(phase_match(contact, a, sigma), phase_match(contact, b, sigma)),
        rf_step,
    )
    return (
        match
        * moving_gate(env, command_name, command_threshold)
        * _gait_cycling_gate(env, sensor_name, GAIT_FEET, max_air_time, max_contact_time)
    )


def rf_step_window(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """RF off the ground while any of the four gait legs is too; a [num_envs] 0/1.

        cost = [RF airborne] * [any of GAIT_FEET airborne]        larger is worse

    RF steps alone, with the four planted (`GROUP_A` has why: with either group it
    stands on a line 86 mm from the centre of mass). `five_foot_gait` pays for that
    phase; this charges the alternative, so a lift made in a group's swing costs
    rather than merely failing to earn. Every control step it happens counts, so
    the charge is the time RF spends airborne out of its window.

    Ungated on the command: standing, the four are down and it is zero anyway, and
    a trunk stepping to recover from a push is exactly when RF should not lift
    with a group. Contact is the sensor's `found`, the same test `five_foot_gait`
    reads, so the two terms agree about what airborne means.
    """
    contact = contact_state(env, sensor_name)
    rf_up = 1.0 - contact[:, RF_FOOT]
    gait_up = (1.0 - contact[:, list(GAIT_FEET)]).amax(dim=1)
    return rf_up * gait_up


def rf_drag(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """RF's horizontal speed while it touches the ground, walking; m/s, larger is worse.

        cost = |v_xy(RF)| * [RF touching] * [the command asks for motion]

    `rf_step_window` decides **when** RF may lift; this is what makes it lift at
    all. A foot that is only allowed to step in a narrow window, and pays nothing
    for sliding, slides -- and it already did. Measured on
    `2026-09-22_17-40-51/model_21996` (`tools/gait_check.py`), RF's mean speed
    while touching was 134 mm/s walking forward at 0.4 m/s, 177 backing up at
    0.3 and 65 turning, against 46-73 mm/s for LM and RM; its lifts were 3-7 mm.
    The trunk was carrying RF forward along the floor more than RF was stepping.
    `foot_slip` sees it and prices it at almost nothing: -0.1 on the *square*,
    0.0018 per second for 134 mm/s.

    **Linear**, so a slow drag is charged from its first millimetre per second,
    and **RF only**: it is the leg the gait no longer lifts on a schedule, and the
    one with the new reason to stay down. **In contact only**: RF moving through
    the air is the step this term is asking for. **Walking only**: standing, every
    foot's movement is `feet_still_when_standing`'s, and charging it twice would
    make standing feet dearer than walking ones. The contact test is the sensor's
    `found`, the one the gait terms read, so "touching" means the same thing in
    all three.
    """
    asset = env.scene[asset_cfg.name]
    speed = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2].norm(dim=-1).sum(dim=1)
    touching = contact_state(env, sensor_name)[:, RF_FOOT]
    return speed * touching * moving_gate(env, command_name, command_threshold)




class stance_duty_balance:
    """Reward the middle and rear feet spending the same fraction of time loaded.

        duty_i  = EMA of [foot i is carrying more than force_threshold]
        reward  = exp(-var(duty) / std^2) * moving_gate                 in (0, 1]

    Where `group_load_balance` asks who is carrying the weight **now**, this
    asks who has been carrying it **over the last few seconds**, and the two can
    disagree: a leg that plants briefly and hard has the force but not the duty.
    Duty is stance fraction, so equalising it equalises the swing-to-stance ratio
    -- which is the quantity that goes wrong in the failure this is aimed at, the
    frog-like gait where the rear pair swings long and far and the middle pair
    does the carrying.

    Scoped to the four middle and rear feet, and not to the front one: those
    four are mirror images of each other, so equal duty is the right target for
    them, while the front foot walks on a claw and bears through both phases of
    the gait. `group_load_balance` leaves it out for that second reason.

    **kk-rl-lab splits this in two and here it does not need to be.** There,
    `feet_duty_balance` flattens the variance across all five feet and
    `mid_rear_duty_match` separately pins the middle pair's mean against the rear
    pair's, with the note that the first cannot imply the second -- because the
    front foot's duty is free to offset a middle/rear split and keep the overall
    variance low. Once the front foot is out of the term that escape is gone:
    with only these four columns, low variance across them *is* middle equals
    rear. One term, and it is the same statement.

    The EMA is seeded at 0.5 so a fresh episode starts matched, and `alpha`
    decides the window: 0.01 per control step at 50 Hz is a time constant of about
    2 s, several gait cycles, which is what makes this a duty measure rather than
    a slower copy of the force one.

    `force_threshold` is what counts as carrying: 2 N against a fair share of
    about 6 N per foot on five feet, so a grazing touch does not count as stance.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        self._feet = list(cfg.params["feet"])
        self.duty = torch.full(
            (env.num_envs, len(self._feet)), 0.5, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self.duty[slice(None) if env_ids is None else env_ids] = 0.5

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        feet: tuple[int, ...],
        command_name: str,
        std: float = 0.15,
        alpha: float = 0.01,
        force_threshold: float = 2.0,
        command_threshold: float = 0.05,
    ) -> torch.Tensor:
        sensor = env.scene[sensor_name]
        force = sensor.data.force
        assert force is not None, f"sensor {sensor_name!r} has no force field"
        bearing = (
            torch.linalg.norm(force[:, self._feet], dim=-1) > force_threshold
        ).float()
        self.duty = (1.0 - alpha) * self.duty + alpha * bearing
        spread = self.duty.var(dim=1, unbiased=False)
        return torch.exp(-spread / std**2) * moving_gate(
            env, command_name, command_threshold
        )



def _commanded_angle(
    env: ManagerBasedRlEnv, command_name: str | None, index: int
) -> torch.Tensor | float:
    """The commanded angle for one attitude axis, or 0.0 when nothing commands it.

    **The attitude terms have to work both ways**, because they predate the
    command: with no `command_name` they charge the angle itself, which is what
    six-foot behaviour and every checkpoint before the attitude command assume.
    With one they charge the *error*, which is the only form that does not fight
    the command -- an absolute penalty and a command asking for 11 degrees of
    nose-down are two terms pulling in opposite directions, and the policy would
    settle wherever they balance rather than where either asked.
    """
    if command_name is None:
        return 0.0
    return env.command_manager.get_command(command_name)[:, index]


def _site_pos_w(asset) -> torch.Tensor:
    """Every site's world position, `[B, S, 3]`, and nothing else.

    **Not `EntityData.site_pos_w`**, which slices `site_pose_w` and so converts
    `site_xmat` to quaternions first. Upstream's conversion raises on the native
    backend, where MuJoCo's matrices arrive flat ([B, S, 9]) and `quat_from_matrix`
    wants [..., 3, 3]; mjwarp hands them over already shaped, so only CPU runs --
    which is to say the whole test suite -- ever see it. A position needs no
    rotation, so this reads `site_xpos` directly, on both backends, and the
    vendored `rl/mjlab/entity/data.py` stays upstream's.
    """
    return asset.data.data.site_xpos[:, asset.data.indexing.site_ids]


def _footprint_xy(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str | None,
    twist_index: int = TWIST,
) -> torch.Tensor:
    """Foot site xy in the **commanded footprint frame**, `[B, F, 2]`.

    The feet are taken into the yaw-aligned base frame, exactly as
    `foot_home_position` has always done, and then rotated *back* by the commanded
    twist so that a robot obeying the twist command reads as having its feet at
    home.

    **Without this, the twist command and the two foot-geometry terms are a
    contradiction.** A trunk twisted by `theta` over stationary feet sees every
    foot rotate by `-theta` in its own frame; at the 30 degree standing command
    and a 200 mm foot radius that is 103 mm of apparent displacement, against
    `foot_home`'s 40 mm deadzone -- so obeying a twist command would cost about
    0.3 per step of `foot_home` and trip `rf_outboard` as well. Rotating the
    measurement by the same angle makes the two terms measure the shape of the
    stance rather than its heading, which is what they were always for.
    """
    asset = env.scene[asset_cfg.name]
    positions = _site_pos_w(asset)[:, asset_cfg.site_ids, :]  # [B, F, 3]
    n_feet = positions.shape[1]
    root_pos = asset.data.root_link_pos_w.unsqueeze(1)
    yaw = yaw_quat(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, n_feet, -1)
    local = quat_apply_inverse(yaw, positions - root_pos)[:, :, :2]  # [B, F, 2]
    if command_name is None:
        return local
    theta = env.command_manager.get_command(command_name)[:, twist_index]
    cos, sin = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
    x, y = local[:, :, 0], local[:, :, 1]
    return torch.stack((cos * x - sin * y, sin * x + cos * y), dim=-1)


def foot_home_position(
    env: ManagerBasedRlEnv,
    home_xy: tuple[tuple[float, float], ...],
    dead_radius: float = 0.04,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalise feet standing further than `dead_radius` from where they belong.

        cost = sum_i max(0, |xy_i - home_i| - dead_radius)

    in metres, larger is worse. Positions are the foot **sites**, taken into the
    yaw-aligned base frame, so the measure is where the feet are *under the body*
    and is unaffected by where the body is or which way it is facing. `home_xy`
    is read off the model by `grasp_pose.py --stance` rather than written down.

    Sites, and it has to be sites: `HOME_TOE_XY` is itself read off `site_xpos`
    by `grasp_pose.py --stance` (`Claw.site_xy`), so measuring anything else
    against it compares two different points. They are also the sites
    `foot_clearance` and `foot_slip` take, which is what makes the three foot
    terms agree about where a foot is.

    This read the foot **bodies** for a while, because `EntityData.site_pose_w`
    raised -- it handed MuJoCo's flat [B, S, 9] `site_xmat` to
    `quat_from_matrix`, which wants [..., 3, 3]. That is an upstream bug, and
    `_site_pos_w` reads the positions without going through it.

    **The workaround was closer to free than it looked, but not for the reason
    recorded at the time.** The note here used to say `build_jumper.py` puts each
    site at its foot body's origin; it does not. The site sits 2.4 mm (RF) to
    5.8 mm (the toe tips) from the origin *in the body frame*, and what makes
    them agree in xy at HOME is that the offset points almost straight down
    there -- a property of the stance, not of the geometry. Measured on the
    five-foot env, 8 environments, native CPU backend: **0.0001 mm apart in xy at
    reset, 1.26 mm worst case** over 200 steps of settling and random action, and
    bounded above by the 5.8 mm offset itself once a leg turns far enough.
    Against a 40 mm deadzone the term's cost came out identical to the last bit
    across that whole rollout, because the feet never reached the deadzone edge
    where the difference could show -- which is the honest version of "the same
    numbers": true here, and not true by construction.

    **What it is for, and why `pose` is not already it.** mjlab's `pose` term is
    the same idea in joint space, and on a 3-DoF leg the two are nearly the same
    constraint -- but only nearly, and the difference is where a five-legged robot
    fails. The stance is redundant: a leg can be well away from its home joint
    angles with the foot in the right place, and the whole *stance* can splay
    outwards while every joint stays within its posture kernel. Measured on
    kk-rl-lab's policy, asked to lean while stationary, the feet splayed 107 mm --
    2.7 times its 40 mm deadzone -- while joint-space terms were satisfied.

    A deadzone, not a kernel: a walking robot has to move its feet, and the
    deadzone is what keeps this from being a tax on stepping. 40 mm is
    kk-rl-lab's, and it is the one number in this term with a history: it was
    briefly 25 mm while standing, and that change was never validated (the run
    was stopped at 900 iterations).

    A circular deadzone rather than kk-rl-lab's independent x and y ones. Theirs
    is a square whose corner allows 57 mm of displacement in the diagonal
    direction against 40 on the axes; a foot is not more free to move diagonally,
    and the square shape was an artefact of writing the deadzone per axis.
    """
    local = _footprint_xy(env, asset_cfg, command_name)  # [B, F, 2]
    n_feet = local.shape[1]
    if n_feet != len(home_xy):
        raise ValueError(
            f"foot_home_position got {len(home_xy)} home positions for {n_feet} sites"
        )
    home = torch.tensor(home_xy, device=local.device, dtype=local.dtype)
    offset = torch.linalg.norm(local - home, dim=-1)  # [B, F]
    return (offset - dead_radius).clamp(min=0.0).sum(dim=1)


def nose_pitch(
    env: ManagerBasedRlEnv,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalise trunk pitch away from the command, **in either direction**.

        cost = (pitch - pitch_cmd)^2

    ## The shape is kk-rl-lab's `pitch_error_l2`, and the deadzone is gone

    A plain square -- no deadzone, no linear term -- and the difference is not
    cosmetic. This task carried `excess + 12 * excess^2` past a deadzone, which at
    the same weight prices the same error **13 times higher**: 8 degrees cost 3.81
    per step against 0.29, and 20 degrees 23.1 against 1.83. Three runs froze
    solid under it -- `2026-09-10_15-47-30`, `17-07-12` and `18-16-19`, the last
    swept at iteration 2999 and still 0.000 m/s against every command in six
    directions -- because a policy that cannot walk yet spends its first stumbling
    steps at exactly the 5 to 12 degrees where that multiple bites hardest.

    A square never saturates, which is the property the deadzone version reached
    for, and it is negligible at a small error on its own with no free band to
    tune: at weight 15, 1 degree costs 0.005, 5 degrees 0.11, 20 degrees 1.83.

    Pitch is read from projected gravity rather than from Euler angles, which is
    what makes it well behaved through any yaw.

    ## This term has been one-sided twice, in both directions, and neither held

    It began as `nose_down_pitch`. The argument was mechanical and still holds as
    far as it goes: with LF carried the front of the machine is held up by RF
    alone, so pitching *down* loads that one foot and walks the centre of mass
    toward the edge of the polygon it stands on -- `grasp_pose.py --stance`
    reports a -17.1 mm margin with RF airborne.

    It was then measured and reversed. On `model_6400` the policy sat at -2.98
    degrees mean with 50.5% of steps past 3 degrees **nose-up**, against 1.49
    degrees the other way, and RF was down 66% of the time carrying 17.9% of the
    load: leaning back is how the machine takes weight off its single front foot,
    and a nose-down-only penalty left that free. So the sign flipped to
    `nose_up_pitch`.

    That worked, and then the drift went back the other way. Measured on
    `model_2200`: mean pitch **+2.93 degrees, nose-DOWN**, with a maximum nose-up
    of 0.19 degrees. The penalty had eliminated the direction it charged for and
    the policy had simply moved into the direction it did not.

    **Twice is a pattern, and the pattern is the lesson.** What the one-sided form
    was protecting was never really a direction; it was the machine's tendency to
    park at whatever pitch offloaded the front foot, and it will find whichever
    side is free. So the term is symmetric now: any sustained pitch away from
    level costs, and the mechanism-specific asymmetry -- that nose-down is the
    more dangerous half -- is left to `upright`, `low_stance` and the support
    polygon, which is where it was always really enforced.

    ## Why a square and not a linear cost

    A purely linear cost charges the same per degree at 2 degrees as at 20, which
    treats one large lean as twenty small ones. The square makes the last degrees
    cost far more, which is what the mechanism does: 5 of 5 feet planted at 10
    degrees nose-down, 4 of 5 at 14, 2 of 5 at 17. `upright` is gone -- it was a
    second opinion about the same axis with level as its only target -- so this
    and `track_body_pose` are what price pitch now.

    kk-rl-lab's version takes an optional command name and measures the error
    relative to a commanded body pose, because absolute it fought their
    body-attitude command. That objection does not apply here: this framework has
    one command, the velocity twist, and nothing ever asks the robot to pitch.
    """
    asset = env.scene[asset_cfg.name]
    pitch, _ = base_pitch_roll(asset.data.projected_gravity_b)
    error = pitch - _commanded_angle(env, command_name, PITCH)
    return error.square()


def body_roll(
    env: ManagerBasedRlEnv,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalise trunk roll away from the command, either way. `nose_pitch` for
    the other axis, same shape -- see its docstring for why that shape is a plain
    square and not a deadzone.

        cost = (roll - roll_cmd)^2

    Roll comes from the base-frame gravity vector as `atan2(-g_y, -g_z)`, which is
    yaw-independent and therefore computable on the robot from its IMU alone.
    Positive is the right side down.

    ## Why roll needed its own term

    `upright` scores both axes -- `exp(-(g_x^2 + g_y^2) / std^2)` -- and was the
    only thing scoring roll *angle* at all. At its std of 0.447 rad it is very
    loose, and against `nose_pitch` at -15 the two axes were priced 40-odd times
    apart:

        tilt        upright loses (w 1.0)   nose_pitch costs (w -15)   ratio
         2 deg          0.0061                    0.078                13x
         5 deg          0.0373                    1.449                39x
        10 deg          0.1400                    5.928                42x
        20 deg          0.4428                   23.111                52x

    A policy that is charged 42 times more for pitching than for rolling will
    roll, and measurement found exactly that. `model_2200`, seven commanded
    directions, 9 s each: **standing produced a +2.39 degree list to the right
    that never moved** -- peak-to-peak 0.05 degrees, so a posture rather than a
    wobble. On `model_2400`, after `nose_pitch` became symmetric and pitch fell,
    standing straightened to +0.38 but the list reappeared under command: -3.36
    degrees turning left, -2.92 strafing right, again nearly constant within each
    phase.

    That is the same behaviour `nose_pitch` was written twice to catch, on the
    axis nothing was watching: the machine parks at whatever tilt is cheapest, and
    it has a standing reason to prefer one side, since LF is carried out to the
    left and `rf_outboard` pushes RF out to the right.

    Same threshold and same quadratic as `nose_pitch`; the weight is set in
    `env_cfg.py` and is two thirds of pitch's, because roll's useful range is the
    narrower of the two -- nothing here ever wants the trunk rolled, whereas a
    little nose-down is part of accelerating. What matters is that the gap is
    closed rather than inverted: a factor of 1.5 between the axes instead of 42.
    """
    asset = env.scene[asset_cfg.name]
    _, roll = base_pitch_roll(asset.data.projected_gravity_b)
    error = roll - _commanded_angle(env, command_name, ROLL)
    return error.square()



class body_twist:
    """Score the trunk's yaw **relative to its own footholds** against the command.

        reward = exp(-(twist - twist_cmd)^2 / std^2) - 1

    kk-rl-lab's `track_base_twist` at the same 15 degree std, minus one: there it
    is a positive kernel paying up to its weight, here it tops out at **zero**.
    The difference matters on this task, where three runs have ended parked -- a
    term that pays for holding an attitude pays most standing still, which is the
    one posture the robot must not be rewarded into. Minus one keeps the gradient
    and removes the bonus.

    A Gaussian rather than a deadzone because **a walking gait rotates the trunk
    against its own footholds by construction**: measured at 4.9 degrees walking
    and 2.3 standing on a policy that tracks at 95.4%. That measurement was taken
    under the 3+2 grouping this task ran then, and the number is quoted as it was
    measured; the property is the walking, not the grouping, and the gait is
    2+2+1 now (`GROUP_A`). At 15 degrees of std the
    kernel is still 0.89 of its peak there, where the quadratic-past-a-deadzone
    form this replaces charged -1.75 per step for the same gait.

    ## Where the measurement comes from

    Not from the root quaternion -- that is heading, and heading is what
    `ang_vel_z` commands. What this axis asks for is the *offset* between the trunk
    and the feet it is standing on, which is a shape, not a direction. When the
    trunk turns by `theta` over stationary feet, every foot's position in the
    yaw-aligned base frame rotates by `-theta`, so the angle that best maps the
    home footprint onto the current one recovers it:

        phi   = atan2(sum_i w_i (home_i x cur_i), sum_i w_i (home_i . cur_i))
        twist = -phi

    a 2-D Procrustes fit, which is a closed form rather than an optimisation. It is
    also deployable: foot positions come from the joint encoders through forward
    kinematics, and no part of it needs to know where the robot is in the world.

    `w_i` is 1 for a foot **bearing load** and 0 otherwise, so a swinging leg --
    which is not a foothold and is often exactly where the trunk is not -- does not
    drag the estimate. With nothing bearing (all five airborne, which on this
    machine means it is falling) every foot is weighted instead, because a fit of
    nothing has no angle at all and returning a hard zero there would read as
    "perfectly on command" at the moment the robot is least under control.

    ## Why a penalty and not a tracking kernel

    kk-rl-lab scores this axis with `exp(-err^2/std^2)`, a term that pays up to 1.0
    per step. That form is what put this task's earlier attitude attempt into a
    standing basin: a robot standing still holds any attitude perfectly, so it
    collected the attitude budget for doing nothing and stopped walking. Twice.
    A penalty cannot be collected, so an idle robot gains nothing here.

    ## While standing, only the trunk's own turn counts

    The offset above has two ways to appear, and only one of them is the command:
    the trunk turns over planted feet, or the feet walk round under a trunk that
    stays put. Measured, the policy trained on the offset alone did the second
    for two thirds of every twist (`feet_still_when_standing` has the table) --
    and was paid in full, because the offset cannot tell them apart:

        offset = (trunk's turn in the world) - (footprint's turn in the world)

    So once the robot has stood for `settle_time` (the swing in progress when it
    stopped has landed), the footprint is recorded where it stands, and from then
    on the footprint's own turn in the world since that record is added back:

        twist = offset + footprint's turn since the record = the trunk's own turn

    A footprint that walks round cancels exactly and earns nothing; a trunk that
    turns over planted feet is scored as before. Walking, the feet have to move
    and the record follows them every step, so the offset is scored unchanged.
    The turn is a Procrustes fit of the five feet's world positions against the
    record, all five weighted: standing, every one of them is meant to be down.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        settle_time = cfg.params.get("settle_time", 0.5)
        self._settle_steps = max(1.0, settle_time / env.step_dt)
        self._idle_steps = torch.zeros(env.num_envs, device=env.device)
        n_feet = len(cfg.params["home_xy"])
        self._record = torch.zeros(env.num_envs, n_feet, 2, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._idle_steps[env_ids] = 0.0   # the record follows the feet until settled

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        home_xy: tuple[tuple[float, float], ...],
        sensor_name: str,
        command_name: str,
        std: float = 0.2618,
        force_threshold: float = 2.0,
        twist_index: int = TWIST,
        velocity_command_name: str = "twist",
        command_threshold: float = 0.05,
        settle_time: float = 0.5,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del settle_time  # read once, in __init__
        local = _footprint_xy(env, asset_cfg, None)  # [B, F, 2], as measured
        n_feet = local.shape[1]
        if n_feet != len(home_xy):
            raise ValueError(f"body_twist got {len(home_xy)} home positions for {n_feet} sites")
        home = torch.tensor(home_xy, device=local.device, dtype=local.dtype).unsqueeze(0)

        force = env.scene.sensors[sensor_name].data.force
        assert force is not None, f"sensor {sensor_name!r} has no force field"
        bearing = (torch.linalg.norm(force, dim=-1) > force_threshold).to(local.dtype)  # [B, F]
        bearing = torch.where(
            bearing.sum(dim=1, keepdim=True) > 0.0, bearing, torch.ones_like(bearing)
        )
        cross = (bearing * (home[..., 0] * local[..., 1] - home[..., 1] * local[..., 0])).sum(dim=1)
        dot = (bearing * (home[..., 0] * local[..., 0] + home[..., 1] * local[..., 1])).sum(dim=1)
        offset = -torch.atan2(cross, dot)

        # The footprint's own turn in the world since the record, zero until settled.
        asset = env.scene[asset_cfg.name]
        feet = _site_pos_w(asset)[:, asset_cfg.site_ids, :2]  # [B, F, 2]
        standing = 1.0 - moving_gate(env, velocity_command_name, command_threshold)
        self._idle_steps = (self._idle_steps + 1.0) * standing
        settled = self._idle_steps > self._settle_steps
        self._record = torch.where(settled[:, None, None], self._record, feet)
        was = self._record - self._record.mean(dim=1, keepdim=True)
        now = feet - feet.mean(dim=1, keepdim=True)
        turned = torch.atan2(
            (was[..., 0] * now[..., 1] - was[..., 1] * now[..., 0]).sum(dim=1),
            (was * now).sum(dim=(1, 2)),
        )
        twist = offset + turned * settled.to(offset.dtype)

        command = env.command_manager.get_command(command_name)[:, twist_index]
        return torch.exp(-(twist - command).square() / std**2) - 1.0


class feet_still_when_standing:
    """How fast the feet move **while the velocity command asks the robot to stand**.

        cost = sum_i |v_xy,i| * settle        m/s, larger is worse
        settle = min(1, steps since the velocity command went idle / settle steps)

    The gate the attitude command was missing. `body_twist` measured the twist as
    the offset between the trunk and its footholds, and an offset has two ways to
    appear: the trunk turns over planted feet, which is the command, or the feet
    walk round under a trunk that stays put, which is not. It could not tell them
    apart and the policy found the second. `body_twist` now credits only the
    trunk's own turn while standing, which takes the payment away; this term
    charges the movement itself, including the shuffling a twist measure cannot
    see because it turns nothing. Measured on
    `2026-09-22_15-31-02/model_16997`, standing, twist held for 4 s
    (`tools/twist_check.py`):

        twist cmd   trunk turned   feet turned   feet moved   lift-offs/foot   sum|v_xy|
          +10 deg      +4.9 deg      -3.4 deg      15 mm          1.7          0.056 m/s
          +20 deg      +6.2 deg     -12.4 deg      44 mm          2.7          0.098 m/s
          +30 deg      +9.4 deg     -18.5 deg      65 mm          3.6          0.126 m/s

    Two thirds of every twist walked in by the feet. **Nothing in the reward asked
    for that.** Held by PD over planted feet, the intended posture scores as well
    as standing square on every term but two: `pose`, which the policy never
    collects anyway (0.008 standing square, 0.000 twisted -- its standing std is
    0.05 rad), and `dof_pos_limits` at 30 degrees, -0.09 per step as the mid legs'
    hips pass their soft limit. So walking the feet round was a habit the policy
    fell into and nothing ever charged it for, and one charge is the whole fix.

    **Speed, not displacement from where the feet were**, so there is no anchor to
    keep: the time integral of this is the distance the feet travelled, and a foot
    that steps away and back pays for both legs of it. Linear in speed, so a slow
    creep is charged from its first millimetre -- the same policy standing with no
    command at all drifts its feet 5 mm in 4 s.

    **Horizontal only.** What must not move is where each foot stands; a pad
    lifting and landing on the same spot changes nothing an operator aimed.

    **The gate is the velocity command, not the attitude.** Standing with no
    attitude asked for is standing too, and the feet should not creep then either.
    It is the complement of `feet_slip`'s gate (`moving_gate` at the same
    threshold): that term charges sliding while walking, this one charges moving
    while standing.

    **It ramps in over `settle_time` after the command goes idle**, and that is
    what lets the weight be larger than `standing_sway`'s. The note on that term in
    `velocity_env.py` is the reason: the moment a walking robot is told to stop,
    its swing feet are still in the air and have to land, and a gate that charges
    them from the first idle step teaches an abrupt stop rather than a still
    stance. 0.5 s is this task's own longest swing (`five_foot_gait`'s
    `max_air_time`), so a step in progress when the command stops has landed
    before the charge is whole. A fresh episode starts the count at zero too, so
    the drop from the reset pose is not charged either. An attitude resample while
    standing does not restart it: the feet should hold through that.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        settle_time = cfg.params.get("settle_time", 0.5)
        self._settle_steps = max(1.0, settle_time / env.step_dt)
        self._idle_steps = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._idle_steps[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.05,
        settle_time: float = 0.5,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del settle_time  # read once, in __init__
        standing = 1.0 - moving_gate(env, command_name, command_threshold)
        self._idle_steps = (self._idle_steps + 1.0) * standing
        settle = (self._idle_steps / self._settle_steps).clamp(max=1.0)
        asset = env.scene[asset_cfg.name]
        speed = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2].norm(dim=-1).sum(dim=1)
        return speed * settle


def stand_height_hold(
    env: ManagerBasedRlEnv,
    command_name: str,
    nominal_height: float,
    deadzone: float = 0.006,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Height, held **while standing** -- where there is no gait to excuse it.

    `body_height_hold` covers walking and has to: the trunk bobs 20 mm either way
    through a stride, so its deadzone is 15 mm and its excess is squared, and at
    that setting a standing robot can rise 28 mm for 0.13 of a step's reward.
    Which is what it does. Measured on `model_6999`, commanded to stand and to
    pitch nose-down:

        command      base height    mouth height
        neutral        109 mm          62 mm
        5 deg          118             44
        10 deg         130             37
        15 deg         138             30

    **The robot stands up to pitch down**, and the operator pays for it twice:
    the trunk is not where it was asked to be, and the mouth -- the thing the
    pitch was commanded *for* -- drops 2.4 mm per degree instead of the 4.5 that
    the same pitch at a fixed stance is worth (kk-rl-lab measures that curve).
    Twist does it too, 17 mm by 20 degrees.

    None of that is the mechanism. Damped-least-squares IK on the real model,
    five feet pinned on their home contact points and the base held at
    `STAND_Z`: nose-down 15 degrees, roll 10 and twist 20 all solve to a **0.0 mm
    foot residual**. Every attitude the operator can command is reachable with
    the body exactly where it should be.

    So this is the standing half of the same idea, and two things differ from its
    walking twin: the deadzone is 6 mm rather than 15, and the excess is
    **linear**. Squared, a small excursion is worth almost nothing and the policy
    sits in it; linear pulls as hard at 10 mm as at 30.
    """
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    excess = ((height - nominal_height).abs() - deadzone).clamp(min=0.0)
    command = env.command_manager.get_command(command_name)[:, :3]
    standing = (command.norm(dim=1) < command_threshold).float()
    return (excess / nominal_height) * standing


def stand_still_when_idle(
    env: ManagerBasedRlEnv,
    command_name: str,
    pose_command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.06,
    pose_threshold: float = 0.01,
) -> torch.Tensor:
    """How far the joints have wandered from their stance **while asked to stand**.

    kk-rl-lab's `stand_still_when_idle`, at its weight
    (`locomotion_5foot/mdp/rewards_cfg.py`), and it is the term this task was
    missing rather than a new idea. Measured on `model_6999`, commanded zero for
    4 seconds: the robot wanders **26.5 mm and 3.6 degrees of yaw** away from
    where it was told to stop. Nothing charged it for that. `standing_sway` is
    the only other idle-gated term and it squares the body velocity, so a 20 mm/s
    creep costs it 4e-4 before its own -0.1 weight -- four parts in a hundred
    thousand of one step's reward, which is not a gradient.

    This measures the joints instead of the body, and linearly, which is what
    makes it bite: a foot that shuffles has to leave the stance to do it, and an
    L1 penalty pulls just as hard at 1 degree from home as at 10.

        idle = sum |q - q_home| * alpha,   alpha = clamp(1 - |cmd| / threshold)

    `alpha` ramps rather than switching, so there is no cliff at the threshold
    for a policy to sit on the edge of.

    **The gate is the whole command, not its linear half.** The reference takes
    the norm of `cmd[:, :2]`, which leaves a pure turn -- vx = vy = 0, wz = 0.5 --
    reading as idle: the term would then charge the policy for every joint it
    moves to obey the turn it was asked for. Yaw rate is in the norm here for
    that reason, and it is the one place this departs from the reference.

    Commanding an attitude switches it off entirely, as it does there: pitching
    the nose down **is** leaving the stance, and a policy cannot both hold its
    home joints and put the trunk where it was asked to.
    """
    twist = env.command_manager.get_command(command_name)[:, :3]
    alpha = (1.0 - twist.norm(dim=1) / command_threshold).clamp(0.0, 1.0)
    pose = env.command_manager.get_command(pose_command_name)
    idle_pose = (pose.norm(dim=1) < pose_threshold).float()
    asset = env.scene[asset_cfg.name]
    deviation = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ).abs()
    return deviation.sum(dim=1) * alpha * idle_pose


def body_height_hold(
    env: ManagerBasedRlEnv,
    nominal_height: float,
    deadzone: float = 0.015,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalise the trunk leaving its stance height, **in either direction**.

        excess = max(0, |z - nominal| - deadzone)
        cost   = (excess / nominal)^2

    ## Why `low_stance` does not already do this

    It is one-sided -- it charges the body for sinking and nothing for standing
    tall -- and the direction the height actually moves is **up**. Measured on
    `2026-09-11_11-02-10/model_4500`, 32 environments holding each command for 5 s
    after 5 s of settling:

        command                trunk height    against level    low_stance
        level                    111.3 mm          +0.0           -0.0000
        nose-down 15 deg         152.0 mm         +40.8            0.0000
        nose-up 15 deg           115.2 mm          +3.9            0.0000
        roll 10 deg              103.9 mm          -7.4           -0.0008
        twist 20 deg             109.8 mm          -1.4           -0.0000
        forward 0.3, level       114.9 mm          +3.7            0.0000
        forward 0.3, nose-down   143.5 mm         +32.2            0.0000

    **A nose-down command raises the trunk by 41 mm and `low_stance` logs
    0.0000** through every one of them: it cannot see a body that is too high, and
    a body that is too high is what happens. Nothing else in the task scores
    height at all.

    ## The deadzone is the stance the policy actually holds

    The nominal is `STAND_Z`, 105 mm, and a level policy sits at 111 -- six above.
    A two-sided term anchored at 105 with no deadzone would charge the posture the
    machine chooses for itself when nothing is wrong, which is the failure the
    attitude terms were just rescued from. 15 mm covers that six with room, and
    leaves the 41 mm case paying for 26.

    Normalised by the nominal, like `low_stance`, so the weight reads as a
    fraction of the stance height rather than in square metres: the measured 41 mm
    excursion costs 0.055 per unit weight, 25 mm costs 0.008, and anything inside
    the deadzone costs nothing.

    Not gated on either command. Height is not something the operator asks for in
    this framework -- kk-rl-lab carries a commanded `height` channel on its
    attitude command and tracks it, and `dds.rs` already forwards one, but nothing
    here samples or observes it -- so the only target available is the stance the
    robot stands at, and that applies whether it is walking or still.
    """
    asset = env.scene[asset_cfg.name]
    excess = (
        (asset.data.root_link_pos_w[:, 2] - nominal_height).abs() - deadzone
    ).clamp(min=0.0)
    return (excess / nominal_height).square()


def low_stance(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalise the body sitting **below** `target_height`, quadratically.

        cost = (max(0, target - z) / target)^2

    a dimensionless fraction of the stance height, larger is worse. One-sided:
    standing tall is not a fault and is not charged.

    Normalised by the target so the weight reads directly: 1.0 is the body flat
    on the ground, 0.25 is half height, and 0.036 is 20 mm low. Left in m^2 the
    same three cases would be 0.011, 0.0028 and 0.0004, and the weight needed to
    make any of them matter would be in the hundreds -- a number nobody can sanity
    check against the 2.0 on velocity tracking.

    **A penalty rather than kk-rl-lab's three height terms**, and the reasons are
    one measurement and one structural difference:

    - `base_height_target` there tracks a *commanded* stand height. There is no
      such command in this framework, so the tracking half has nothing to track
      and only the nominal survives.
    - `base_height_smooth` penalises frame-to-frame height change. kk-rl-lab
      measured its contribution at -0.0073 per step and called it negligible;
      what remains of its job -- vertical bobbing while standing -- is already in
      `standing_sway`, which squares the body's linear velocity under a standing
      command.
    - `bad_base_height` is a large penalty below 0.05 m, i.e. a soft termination.
      That is a termination here (`too_low` in `env_cfg.py`), where it belongs:
      an episode spent lying down teaches nothing and should end rather than be
      charged for.

    One-sided and quadratic rather than an exponential kernel because a kernel is
    a *positive* term, and a positive term that a motionless robot collects in
    full is precisely what this robot's reward design spends its effort avoiding.
    At the nominal stance this costs exactly zero.

    Height is measured against the environment origin, so it is the height above
    that environment's own ground rather than a world coordinate.
    """
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return ((target_height - height).clamp(min=0.0) / target_height).square()


def foot_outboard_of_trunk(
    env: ManagerBasedRlEnv,
    foot_index: int,
    min_offset: float,
    outboard_sign: float,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalise one foot for standing **inside the body's own footprint**.

        outboard = outboard_sign * y            (metres, yaw-aligned base frame)
        cost     = max(0, min_offset - outboard)

    One-sided: past `min_offset` the term is exactly zero and the foot is free to
    go as far out as the leg reaches. It only ever pushes outwards.

    ## Why one foot, and why this one

    Four of the five ground feet already stand well clear of the trunk -- LM and
    RM at 200 mm, LR and RR at 161, against a body that is 90.7 mm wide either
    side (`claw.py::TRUNK_HALF_WIDTH`). **RF stands at 36.5 mm**, forty percent
    of the way in from the body's edge, and with LF carried as a claw it is the
    *entire* front of the machine. The support polygon that results has a
    -17.1 mm margin with RF airborne, and that negative number is this geometry
    seen from the other side: the front support is a single point, and it is a
    single point tucked under the body rather than out at its edge.

    Widening it is a matter of one joint. Measured by sweeping
    `RF_J0_joint` with everything else at HOME:

        yaw     -0.700  -0.355  -0.010  +0.335  +0.552  +0.680
        foot y  -0.195  -0.163  -0.120  -0.069  -0.037  -0.018

    HOME is +0.552, the joint's range runs to -0.700, and the body's edge is
    reached at about +0.15. So the outboard half of the range is 0.40 rad of
    travel away and the leg keeps its forward reach doing it (x goes 0.207 ->
    0.197 at the edge). The term is asking for something the machine can do
    easily, which is what makes a one-sided penalty the right shape: there is no
    case where it is charging for a limit.

    ## Not gated on contact, deliberately

    Scoring only the loaded foot would make *lifting it* a way to stop paying,
    which on a five-legged machine with one front foot is the last behaviour to
    incentivise. Charging the position whenever it is inboard also shapes the
    swing -- the leg sweeps out and stays out -- which is the same thing asked
    for and the reason `foot_clearance` and friends do not need to be involved.

    ## What it costs against `foot_home`

    `foot_home_position` pulls every foot back towards `HOME_TOE_XY` outside a
    40 mm deadzone, and moving RF from -36.5 mm to the body's edge is 54 mm, so
    the two do disagree -- by 14 mm past the deadzone, which at that term's -1.0
    is 0.014 per step. Against this term's own arithmetic (see `env_cfg.py`) that
    is a twentieth of the pull, so the deadzone is not what decides where the foot
    ends up. Left alone rather than re-centred, because `HOME_TOE_XY` is a
    *measurement* of where the feet are at HOME, not a target, and editing it to
    win an argument would make it a lie about the pose.
    """
    local = _footprint_xy(env, asset_cfg, command_name)  # [B, F, 2]
    n_feet = local.shape[1]
    if not 0 <= foot_index < n_feet:
        raise ValueError(f"foot_index {foot_index} is outside the {n_feet} sites given")
    outboard = outboard_sign * local[:, foot_index, 1]
    return (min_offset - outboard).clamp(min=0.0)


def feet_contact_without_cmd(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward the **number** of feet on the ground while nothing is commanded.

    A standing hexapod that shuffles its feet is not resting, and on hardware it
    is wearing out gearboxes to hold still. This pays per planted foot, so lifting
    one loses a fifth of the term.

    **This is a positive term that a motionless robot collects in full**, which is
    exactly the shape this robot's reward design otherwise refuses (see the module
    docstring). It is admitted here because the alternative reading is the right
    one: the term is not paying the robot for *standing*, it is paying it for
    standing **still** once standing is what was asked for. The gate is on the
    command, so it can never make standing more attractive than obeying a
    non-zero command -- the two are never both available.

    Weight matters more than usual for that reason, and it is why this sits at
    kk-rl-lab's 1.0 against five feet rather than being scaled up.
    """
    contact = contact_state(env, sensor_name)  # [B, F]
    cmd = env.command_manager.get_command(command_name)
    standing = (torch.linalg.norm(cmd[:, :3], dim=1) < command_threshold).float()
    return contact.sum(dim=1) * standing


class group_load_balance:
    """Reward the two gait groups carrying the same force **while each bears**.

        bearing_g = mean(|force| over group g's feet) > force_threshold
        load_g    = EMA of that mean, advanced only on the frames where group g
                    is bearing
        reward    = exp(-|load_A - load_B| / mean(load)) * moving_gate   in (0, 1]

    **It replaced a left/right version**, `lateral_load_balance`, which asked the
    same question of the two sides -- LM+LR against RM+RR. The sides are not what
    the gait alternates; the groups are. So this asks whether each *phase* pulls
    its weight, over `GROUP_A` (LM, RR) against `GROUP_B` (RM, LR), the partition
    `five_foot_gait` scores. RF is in neither: it bears through both phases. The
    old function is deleted rather than kept beside this -- nothing configured it
    once this existed, and a reward nothing wires is a reward nobody checks.

    ## The average is conditional on bearing, and that is the whole design

    **Instantaneously the two groups are supposed to be unbalanced.** The gait
    has one group in stance while the other swings, so the swinging group's force
    is near zero and the instantaneous gap is near its maximum. Scored per step
    this term would be minimised exactly where `five_foot_gait` is maximised.

    An unconditional average fixes that and introduces a subtler error: it divides
    each group's stance force by that group's **duty cycle**. A group down 60% of
    the time and one down 40% average out at 0.6 and 0.4 of their true stance
    loads, so the term reports an imbalance that is really a difference in *how
    long* each group stands, which `stance_duty_balance` already scores. Two terms
    would then be measuring the same thing, and this one would be measuring it
    badly.

    Advancing each group's average only while that group bears removes the duty
    cycle from the comparison entirely. What is left is the question worth asking:
    **when it is your turn, do you carry as much as the other group carries on
    its turn?**

    `alpha = 0.01` is a time constant of about 2 s of *bearing* frames -- longer
    in wall-clock, since only some frames advance it -- which is several gait
    cycles either way.

    ## Until both groups have borne, there is nothing to compare

    The term returns exactly 0 until each group has been seen bearing at least
    once. Without that, an episode opens comparing a real stance load against an
    uninitialised zero and reports a maximal imbalance the robot has not earned;
    seeding from the first frame instead would compare against whatever the other
    group happened to be doing at t=0. Neither is a measurement, so the term
    declines to make one.

    ## The floor is exp(-2), not zero

    The gap is relative, so one group carrying everything gives
    `|A - 0| / (A/2) = 2` whatever A is: the worst *comparable* score is
    `exp(-2) = 0.135` and the usable range is [0.135, 1]. Reading the logged
    value, 0.4 is not "40% of the way to balanced" but a little over a third of
    the way up a scale that starts at 0.135. The left/right term this replaced
    had the same floor and never wrote it down.

    Relative rather than absolute, and gated on the command, for the same two
    reasons as the term it replaces: a gap divided by the mean means "a tenth out
    of balance" at any load rather than becoming a penalty on stepping lightly,
    and a standing robot is trivially balanced and must not be paid for it.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
        self._a = list(cfg.params["group_a"])
        self._b = list(cfg.params["group_b"])
        self.load = torch.zeros(env.num_envs, 2, device=env.device)
        self._seen = torch.zeros(
            env.num_envs, 2, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        self._seen[idx] = False
        self.load[idx] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        group_a: tuple[int, ...],
        group_b: tuple[int, ...],
        command_name: str,
        alpha: float = 0.01,
        force_threshold: float = 2.0,
        command_threshold: float = 0.05,
    ) -> torch.Tensor:
        sensor = env.scene[sensor_name]
        force = sensor.data.force
        assert force is not None, f"sensor {sensor_name!r} has no force field"
        magnitude = torch.linalg.norm(force, dim=-1)  # [B, F]
        now = torch.stack(
            (magnitude[:, self._a].mean(dim=1), magnitude[:, self._b].mean(dim=1)),
            dim=1,
        )  # [B, 2]

        # `force_threshold` is on the group *mean*, not per foot. On five feet the
        # robot's 26.6 N gives a fair share near 6 N each, and a bearing pair
        # averages well above that -- 2.0 N is well under it and well over solver
        # noise, the same reasoning and the same number `stance_duty_balance` uses.
        bearing = now > force_threshold
        seeded = self._seen & bearing
        first = (~self._seen) & bearing
        self.load = torch.where(seeded, (1.0 - alpha) * self.load + alpha * now, self.load)
        self.load = torch.where(first, now, self.load)
        self._seen |= bearing

        scale = self.load.mean(dim=1).clamp(min=0.5)
        gap = (self.load[:, 0] - self.load[:, 1]).abs() / scale
        comparable = self._seen.all(dim=1).float()
        return (
            torch.exp(-gap)
            * comparable
            * moving_gate(env, command_name, command_threshold)
        )


def track_body_pose(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """`exp(-(pitch_err^2 + roll_err^2) / std^2)` -- the attitude command, tracked.

    The positive half of the attitude command, in the same form and at the same
    weight as the two velocity tracking terms: what the velocity terms do for
    where the robot goes, this does for how it holds itself while going there.

    Pitch and roll only. The third axis is scored by `body_twist`, which needs the
    footholds to measure at all -- twist is an offset between the trunk and the
    feet, not something the IMU can see -- and is a penalty rather than a payment.

    ## std, and what standing at level is worth

    0.175 rad, ten degrees: the same rule the family applies to velocity, half the
    top of the commanded range (`BodyPoseCommandCfg.stand_pitch` is +-20 degrees).
    That makes the term read as a fraction of the attitude actually achieved:

        error off the command   0 deg     5      10      15      20
        term pays               1.000   0.780   0.370   0.107   0.019

    **It saturates, and that is covered rather than fixed here.** Past about 15
    degrees off the command this term is nearly flat, which on its own would be
    the failure `kk-rl-lab`'s `pitch_error_l2` was written to escape -- a policy
    20 degrees out sees no gradient. What supplies the far field is `nose_pitch`
    and `body_roll`, which measure the same error and are linear-plus-quadratic in
    it, so they keep growing exactly where this stops. This term is the near-field
    shaping and the readout; the penalties are the pull.

    ## The `- 1.0`, and the basin it closes

    **A positive term a motionless robot can collect in full** is the shape that
    put three of this task's cold starts into a standing basin: hold still, hold
    any attitude perfectly, collect. The run that carried it held 0.957 of its 1.0
    while tracking sat at 0.385 of a possible 2.0 -- the attitude half solved, the
    walking half abandoned.

    The `- 1.0` is what removes the payment without removing the scoring. The
    kernel is unchanged and so is its gradient -- a constant offset differentiates
    away -- so the policy is pulled toward the commanded attitude exactly as
    before. What changes is what the pull is *worth*: perfect tracking is now 0
    rather than +1, and everything else is negative.

        error off the command   0 deg     5      10      15      20
        term pays, std 0.175    0.000  -0.220  -0.630  -0.893  -0.981
        term pays, std 0.10     0.000  -0.533  -0.952  -0.999  -1.000   <- configured

    **Gating it on the velocity command was the other answer, and it is the wrong
    one for this robot.** A gate makes attitude unscored at a standstill, and
    standing still is precisely when this machine is asked to hold an attitude:
    it carries a claw, and aiming happens with the feet planted. The offset scores
    attitude in both regimes -- moving and parked -- and pays for neither.

    What it costs: a policy far off the command bleeds up to `weight` per step
    whatever it does, which is an incentive to end the episode early if the total
    per-step reward ever goes negative. It does not here -- the positive terms
    reachable while upright are 3.0 before any tracking -- but that is the number
    to check if this weight is ever raised.

    Walking is still worth more, so the basin is a local optimum rather than the
    global one -- but that was true the last two times as well. The readout that
    tells them apart is `Episode_Reward/track_linear_velocity` against
    `Episode_Reward/track_body_pose`: a run that has parked shows this one high
    and that one flat, within a few hundred iterations.
    """
    asset = env.scene[asset_cfg.name]
    pitch, roll = base_pitch_roll(asset.data.projected_gravity_b)
    command = env.command_manager.get_command(command_name)
    error = (pitch - command[:, PITCH]).square() + (roll - command[:, ROLL]).square()
    return torch.exp(-error / std**2) - 1.0
