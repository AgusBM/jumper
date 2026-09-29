"""Environment config for jumper.five_foot.

The skeleton is the family's (`tasks/jumper/common/velocity_env.py`), so everything
this robot has already been calibrated for -- action scale, contact solver,
command curriculum, tracking stds, foot friction, push magnitudes -- is inherited
rather than restated. What this file does is the diff, and it is a bigger diff
than a gait task's because a leg has left the ground:

    1. five legs carry the robot        `_ground_the_five_legs`
    2. the action drops to 16 joints    the carried arm is not driven
    3. the observation grows to 21      the carried arm's angles *are* measured
    4. the arm is held, per episode     `mdp/events.py::hold_carried_arm`
    5. it may be carrying something     a mass on the claw, once it walks
    6. six reward terms are added       `mdp/rewards.py`
    7. one termination is added         lying down ends the episode
    8. in replay, the claw is driven    `mdp/gripper.py`, from its trigger, and
                                        it starts wide open rather than sampled

Each is one block below, in that order, with the reason next to it.

The measurements quoted throughout come from `tasks/jumper/five_foot/tools/grasp_pose.py`,
which reads them off `assets/jumper/jumper.xml` -- run it rather than trusting the
numbers here if the asset has moved.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp

from ..common.constants import (
    HOME,
    STAND_Z,
)
from ..common.mdp.controls import load_controls
from ..common.mdp.curriculum import STD_LIN_RATIO
from ..common.tof import tof_sensor
from ..common.velocity_env import velocity_env_cfg
from .claw import (
    ARM_JOINTS,
    CARRIED_JOINTS,
    FIVE_FOOT_JOINTS,
    FIVE_FOOT_LEGS,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_TOE_XY,
    LF_GRASP,
    LF_GRASP_BOX,
    OBSERVED_JOINTS,
    PAYLOAD_BODY,
    PAYLOAD_RANGE,
    PAYLOAD_UNIT_INERTIA,
    TRUNK_HALF_WIDTH,
    add_payload_body,
    foot_geoms_for,
)
from .jaws import JAW_COLLISION, decompose_jaws
from .mdp.curriculum import PayloadCurriculum
from .mdp.events import claw_payload, hold_carried_arm
from .mdp.grasp import claw_hold_event
from .mdp.gripper import gripper_teleop_event
from .mdp.pose_command import BodyPoseCommandCfg, TeleopBodyPoseCommandCfg
from .mdp.rewards import (
    GROUP_A,
    GROUP_B,
    body_height_hold,
    body_roll,
    body_twist,
    feet_contact_without_cmd,
    feet_still_when_standing,
    five_foot_gait,
    foot_home_position,
    foot_outboard_of_trunk,
    group_load_balance,
    low_stance,
    nose_pitch,
    rf_drag,
    rf_step_window,
    stance_duty_balance,
    track_body_pose,
)
from .objects import apply_objects
from .rl_cfg import NUM_STEPS_PER_ENV

#: The foot geoms of the five legs that carry the robot.
GROUND_FOOT_GEOMS: tuple[str, ...] = foot_geoms_for(FIVE_FOOT_LEGS)

#: Every other collision geom on the robot -- limbs, trunk, the claw, and the
#: left-front jaw that is a foot in the six-foot tasks and is not one here.
#:
#: Derived by subtraction from the collision scheme's own geom list rather than
#: written out, so a change to `constants.py` (a link gaining collision, the tips
#: being brought in as they were once before) reaches this term instead of
#: silently leaving the new geometry unpenalised. It is this task's scheme,
#: `jaws.py::JAW_COLLISION`, so the jaw pieces are charged like the hulls they
#: replace.
NON_FOOT_GEOMS: tuple[str, ...] = tuple(
    g for g in JAW_COLLISION.geom_names_expr if g not in GROUND_FOOT_GEOMS
)

#: Sensor name for those geoms against the terrain.
GROUND_CONTACT_SENSOR = "non_foot_ground_contact"

#: The attitude command's key, named once because six reward terms, every
#: observation group and the replay swap all have to agree on it.
POSE_COMMAND = "body_pose"

#: The task's operator controls, both commands: the walk and the attitude. Parsed
#: twice from one file -- by `velocity_env_cfg` for the walk and by the replay
#: block below for the attitude -- and the two share one operator.
CONTROLS = Path(__file__).parent / "controls.yaml"

#: Contact force, in newtons, above which a non-foot geom touching the ground
#: counts. The same 1.0 N the family's self-collision penalty uses, and for the
#: same reason: a 1.75 N*m actuator on a 0.16 m limb tops out near 10.9 N, so
#: 1.0 N is comfortably above solver noise and well below a limb bearing weight.
#: (The family's note says 12.5 N against the old 2.0 N*m cap; the measured servo
#: plateau moved it, and 1.0 N is 9% of the ceiling either way.)
GROUND_CONTACT_FORCE_THRESHOLD = 1.0

#: All four middle and rear feet, for `stance_duty_balance`.
MID_REAR_FEET: tuple[int, ...] = (1, 2, 3, 4)

# ── Soft ground is a scene, not a switch ──────────────────────────────────
#
# kk-rl-lab has a stage-3 setting that turns the ground into foam (PhysX compliant
# contact on the terrain material, 75 N/m per contact point, a measured 14.2 mm
# sink). The equivalent here is `--scene soft`, which is not this task's business:
#
#     python scripts/train.py --task jumper.five_foot --scene soft --resume
#
# `scenes/soft.py` carries the mechanism and the calibration. Two findings from
# porting this are worth keeping where a reader of *this* task will meet them,
# because both are dead ends that cost time:
#
# **Softening the FEET does not model soft ground, and it fails silently.** The
# foot geoms carry `priority=1` and so decide the foot-vs-ground pair, which makes
# them the obvious place to put the compliance. Measured here (256 envs, 5 s of
# zero action, seed 0, native CPU), the sink saturates at 4.7 mm however soft they
# are made -- `solimp` d0 0.8 -> 0.2 buys 0.22 mm, widening `width` to 0.06 buys
# 4.5 mm, `solref` timeconst out to 0.20 buys 4.7 mm. At the softest setting the
# feet carry **0.0 N of the robot's 26.6 N** while five non-foot geoms carry up to
# 11.2 N: the feet have gone through a plane that stayed rigid for every other
# geom, and the robot is standing on its shins. The compliance has to be on the
# terrain, at a priority above the feet's -- which is exactly what `scenes/soft.py`
# does and why its docstring leads with `priority`.
#
# **The five-foot stance is what makes it matter here.** On rigid ground this
# robot rests 8 mm under STAND_Z with the claw carried; the soft scene adds its
# sink on top of that, and `low_stance` is the term that will notice. Expect it to
# be non-zero on soft ground and do not read that as the policy collapsing.
#
# Soft ground is still a stage: train on rigid ground and resume into it.
# kk-rl-lab measured what starting from random weights on the hard setting does
# -- 8000 iterations to a policy that stands perfectly still. The payload was a
# stage for the same reason and is now a curriculum (`mdp/curriculum.py`); this
# one has not been made one, because it is a scene and a scene is chosen when the
# run starts.

#: Height below which the episode ends, in metres.
#:
#: kk-rl-lab spends a -20 reward term on this ("bad_base_height"). It is a
#: termination here because that is what it is: at 0.05 m against a 0.107 m
#: stance the robot is on its belly, it is not going to walk out of it, and the
#: rest of the episode is a thousand steps of a policy learning what lying down
#: is worth. `fell_over` does not cover it -- that fires at 50 degrees of tilt,
#: and a machine that sinks straight down stays perfectly level while doing it.
MIN_STAND_HEIGHT = 0.05

#: The payload curriculum's rungs, in kg: an empty claw, then `PAYLOAD_RANGE`.
#:
#: Two rungs, "carrying or not", because that is the question the two failed cold
#: starts answered and the only one with a measurement on both sides of it: the
#: stage-two run carried (0, 0.6) from a policy trained empty and walked, at a
#: tracking error of 0.105 against the empty claw's 0.074 (`mdp/curriculum.py`).
#: A (0, 0.3) rung between them would need a bar of its own, and nothing has
#: measured what the error is there.
PAYLOAD_LEVELS: tuple[tuple[float, float], ...] = ((0.0, 0.0), PAYLOAD_RANGE)

#: The linear tracking error, m/s, the policy has to get under at the top command
#: rung before the claw is loaded. The command curriculum's own bar there is 0.18
#: and is met on arrival, while the policy is still learning to walk: 0.164 at
#: iteration 1507 of the cold start in `mdp/curriculum.py`, where 0.12 falls
#: between 3014 (0.130) and 4521 (0.106). It is also above the 0.105 the policy
#: settles at *with* the load, so it does not ask for tracking the loaded policy
#: would not keep. Chosen against that run, not yet measured with the curriculum
#: in place.
PAYLOAD_LIN_ERR_BAR = 0.12

#: Iterations the commands have to have been on their top rung before the claw
#: is loaded. The error average is an EMA seeded when the commands max out, so
#: without a dwell a lucky early stretch could load the claw on a few hundred
#: episodes' evidence; 500 iterations is 12 episodes per environment. On the cold
#: start the bar is the binding condition either way (commands maxed at 933).
PAYLOAD_DWELL_ITERATIONS = 500


def _ground_the_five_legs(cfg: ManagerBasedRlEnvCfg) -> None:
    """Narrow the six-legged skeleton to the five legs that carry this robot.

    **One tuple decides five things, and they have to agree**: the geoms the
    `feet_ground_contact` sensor matches, the sites the foot height scan attaches
    to, the sites `foot_clearance` / `foot_slip` read velocities from, and the foot
    geoms `foot_friction` randomises. Every per-foot quantity downstream -- the
    contact state the gait rewards score, the critic's `foot_height` /
    `foot_air_time` / `foot_contact` / `foot_contact_forces` -- is then that many
    columns wide, in this order.

    Getting it wrong is silent in the way this repository's tests are written for:
    five legs in the sensor and six sites in `foot_slip` leaves both shapes
    plausible and pairs foot *i* of one with foot *i* of the other.
    `claw.py::foot_geoms_for` is the single derivation, and
    `tests/test_five_foot.py` pins the widths. `foot_friction` is the one that
    would not show up anywhere at all: left at six, the carried claw's pad gets a
    friction sample every episode and nothing reads it.

    What it does **not** change: the collision masks and the self-collision sensor
    still cover all six legs (a carried claw can still hit another leg), and the
    robot's `EntityCfg` is untouched here.

    These were a `ground_legs` argument on `velocity_env_cfg` until the task moved
    everything it needs inside its own directory. The config this produces is the
    one that argument did: compared field by field when it moved, train and play.
    """
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=leg, entity="robot") for leg in FIVE_FOOT_LEGS
            )
        elif sensor.name == "feet_ground_contact":
            assert isinstance(sensor, ContactSensorCfg)
            sensor.primary = ContactMatch(
                mode="geom", pattern=GROUND_FOOT_GEOMS, entity="robot"
            )
    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = FIVE_FOOT_LEGS
    if "foot_friction" in cfg.events:
        cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
            "robot", geom_names=GROUND_FOOT_GEOMS
        )

    # njmax is the **per-world** constraint row limit and nconmax the contact
    # limit. The failure is loud but early and obscure: `mujoco_warp.put_data`
    # raises `njmax overflow (njmax must be >= N)` while the environment is being
    # built, on the GPU only -- the native backend never preallocates and so never
    # notices.
    #
    # **What binds is the model as constructed, not the robot as it runs.**
    # mjlab hands `put_data` the `MjData` it built the model with, before any
    # reset, and there the floating base sits at the origin: trunk at floor
    # level, legs through the floor. Measured on `jumper.five_foot`, one world:
    #
    #                                         contacts   constraint rows
    #     as constructed, before 2026-09-12     108          454
    #     as constructed, now                   123          514
    #     after the first reset                   0           22   (dropped in)
    #     2 s later with nothing driven          18           95
    #
    # The jump between the first two is the fixed jaw: convex-decomposing
    # `LF_palm_link` (`jaws.py`) turned one collision geom into six, each carrying
    # the grip face's condim 3, and under mjlab's pyramidal cone a condim-3 contact
    # costs four rows where the catch-all condim 1 costs one. The skeleton's 512 is
    # under that pile-up; 640 and 192 clear it. Slots are per world and cheap beside
    # a model copy. Here and not in the skeleton because the jaws are this task's.
    cfg.sim.njmax = 640
    cfg.sim.nconmax = 192


def env_cfg(
    asset: Path | None = None,
    play: bool = False,
    objects: bool = False,
    hold: str | None = None,
) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config.

    Args:
        asset: model XML path, from `--model`. None uses the default jumper.xml.
        play: replay mode -- observation noise and external disturbances off,
            longer episodes.
        objects: `--objects`, the pick-place row in front of the robot
            (`objects.py::apply_objects`).
        hold: `--hold`, a prop the claw starts the episode holding.
    """
    cfg = velocity_env_cfg(
        # This task's own operator controls: the command axes, their signs, the
        # pad -- jumper.posture's layout, the walk on the left stick and the
        # attitude command below on the right -- the keyboard beside it, and the
        # claw and the arm, which the task answers itself.
        controls=CONTROLS,
        asset=asset,
        play=play,
        gait_reward=five_foot_gait,
        gait_name="five_foot",
        # **Back on at the family's 1.0**, after a measured detour. Turning it off
        # was tried on the argument that the grouping is already right without it:
        # the 3500-iteration policy alternated correctly on 93.8% of steps (group
        # A at a stance duty of 0.86 against group B's 0.16) while the term only
        # scored 0.67, the missing third being `cycling_gate`'s
        # `max_contact_time = 1.0 s` charging RF/LM/RR for a stance longer than a
        # six-foot gait would hold -- a prior calibrated on 3+3 applied to 3+2.
        #
        # **The argument was wrong about cause and effect.** Measured after the
        # term came off, over 600 steps at a commanded 0.50 m/s:
        #
        #     policy                          gait weight   alternating   A / B duty
        #     3500-iteration (term on)            1.0          93.8%      0.86 / 0.16
        #     5699 resumed from it (term off)     0.0          71.5%      0.63 / 0.41
        #     5999 cold start (term off)          0.0          27.2%      0.54 / 0.38
        #
        # The clean grouping was what this term had been *holding*, not something
        # it was merely observing. With it off, a policy that inherits the grouping
        # decays towards synchrony, and one that starts from random weights never
        # finds it at all -- 27% is worse than chance-adjacent. Both walked well
        # (91% and 92% of a 0.50 m/s command), so nothing in the tracking numbers
        # says the gait has gone; only this term does.
        #
        # `cycling_gate`'s threshold is still a 3+3 prior on a 3+2 gait and still
        # worth relaxing. That is a separate change from switching the term off,
        # and switching it off is not a way to make it.
        #
        # **Back to 0.5 after that second attempt.**
        #
        # **Off again, and the second attempt is not the first one repeated.**
        # What decayed last time was the support pattern with nothing else holding
        # it: `lateral_load` and `duty_balance` were at 0.5 each and `feet_planted`
        # did not exist. They are now 1.5, 1.75 and 1.0, so 4.25 of gated support
        # shaping stands where 1.0 did. Whether that is enough to hold a 3+2
        # grouping without a term that names the grouping is the open question --
        # and the readout is `Episode_Reward/five_foot_gait`, which still runs at
        # weight 0.
        #
        # The 0.5 step in between is worth recording: halving the weight cost the
        # gait nothing. Measured over 2000 iterations resumed from `model_8599`,
        # the term's own score held at 0.60 either side of the change while
        # angular-velocity tracking rose 10% (1.245 -> 1.373). The term was being
        # paid twice for what it was already achieving.
        #
        # **The grouping is 2+2 now, and the 3+3 prior on the gate went with it.**
        # Everything above is about {RF, LM, RR} against {RM, LR}. `model_16997`
        # walked that as LM+RR against RM+LR with RF tapping 2-5 mm inside group
        # A's swing -- the only way to collect a grouping whose group A leaves the
        # trunk on the RM-LR line, 86 mm outside it. The term now scores the two
        # diagonal pairs only, and gates on them only, so the stance longer than
        # `max_contact_time` it used to charge RF for is RF's to hold
        # (`mdp/rewards.py::GROUP_A` has the support table).
        gait_weight=0.5,
        # ── The parameters `velocity_env_cfg` no longer defaults ──
        # Named here since the family moved every tuning number into its tasks
        # (`CLAUDE.md`, "Where a parameter lives"). These are the values this task
        # trained under while they were still shared defaults, so naming them
        # changes nothing about what it learns: the "stand still" barrier at
        # mjlab's -0.1 / 1.0, and a 30 mm foot-lift target.
        action_rate_weight=-0.1,
        pose_weight=1.0,
        foot_target_height=0.03,
        # The linear ceiling reproduces the ladder this task climbed: `ladder(0.5,
        # ...)` gives rungs 0.25 / 0.35 / 0.50, which were `LEVELS` exactly, plus
        # the precision rung the shared ladder now adds on top -- the last range
        # again, with the tracking std tightened.
        #
        # **The angular ceiling is 2.0, and was 0.75.** A turn of one revolution
        # in 5 s is 1.26 rad/s, above the old ceiling, so the machine was never
        # asked for it. Measured on `2026-09-23_11-32-43/model_8500`, commanded
        # beyond its training range on eight environments (native, 4 s held):
        #
        #     commanded   tracked        per revolution   torque p99   busiest joint RMS
        #       0.75      0.71 (97%)         8.8 s          1.18 N*m       0.94 N*m
        #       1.26      1.19 (95%)         5.3 s          1.28           0.90
        #       2.00      1.92 (96%)         3.3 s          1.47           0.82
        #       3.00      2.92 (98%)         2.1 s          1.75 <- plateau 0.75
        #       5.00      4.66 (93%)         1.3 s          1.75           -
        #       6.00      0.00 ( 0%)         never          1.20           -
        #
        # Nothing mechanical stops it until 3 rad/s, where the peak torque pins to
        # the servo curve's plateau, and nothing stops the policy until 6, where it
        # simply stops turning -- an input that far outside what it trained on. 2.0
        # is chosen against neither: it covers the 5 s revolution with 60% to
        # spare, keeps the peak off the plateau, and keeps the busiest joint's RMS
        # torque at 0.82 N*m against a 1.2 N*m continuous rating, so a sustained
        # spin is thermally safe. A ceiling at the measured cliff would spend the
        # command range on speeds nothing asks for and train the gait at torques
        # that are saturated throughout.
        command_lin_ceiling=0.5,
        command_ang_ceiling=2.0,
        command_lin_std_ratio=STD_LIN_RATIO,
        # **0.15, where the family's is 0.4** -- because the ceiling above moved,
        # and this ratio is what the ceiling gets multiplied by. `ang_std = ratio
        # * ang_range * scale`, so raising the ceiling from 0.75 to 2.0 widened
        # the command range and loosened the ruler by the same 2.67x. That was not
        # intended, and `Episode_Reward/track_angular_velocity` cannot show it: at
        # std 0.5333 the term saturates and reads the same for an error of 0.09 as
        # for 0.12.
        #
        # What the loose ruler bought, measured at matched iterations
        # (`tools/vel_check.py`, warp, 32 envs, 5 s held per command, 2026-09-23)
        # as the tracked yaw rate over the commanded one:
        #
        #     commanded   09-05-38/7999   17-36-56/2800   17-36-56/7950
        #       0.50           98%             96%             80%
        #       0.75           97%             99%             89%
        #       1.33           96%            100%             95%
        #       2.00           96%            100%             96%
        #
        # The policy under-turns, worst where the error is smallest beside the std
        # -- and it learned to **during** that run: at 2800 iterations it tracked
        # every rate to within 4%. Turning 0.1 rad/s short costs 3.4% of the term
        # at std 0.5333 against 22% at 0.2, while `feet_still`, `rf_drag` and
        # `ground_contact` charge for the movement either way. The cheaper trade
        # was to turn slightly less than asked.
        #
        # 0.15 is not a fresh guess: 0.4 / 0.15 is 2.0 / 0.75, so the absolute std
        # returns to what the old ceiling gave at **every** rung -- 0.12, 0.20,
        # 0.30, 0.20 -- and the range alone is 2.67x wider. It asks of a 2.0 rad/s
        # turn the accuracy the old ladder asked of a 0.75 one, which is the
        # demand a wider range should carry; a wider range that also grades more
        # loosely asks for less than the range it replaced.
        command_ang_std_ratio=0.15,
    )

    # ── 1. Which legs are on the ground ──
    # The family's skeleton is written for six, so this task narrows it to five
    # itself rather than asking the shared code for a parameter only it uses.
    _ground_the_five_legs(cfg)

    robot = cfg.scene.entities["robot"]

    # ── 2. The action: 16 joints, not 20 ──────────────────────────────────
    # The field is `actuator_names`, **not** `asset_cfg`. `BaseActionCfg` is a
    # plain dataclass, so a wrong field name is accepted in silence and the action
    # simply stays 20 wide -- the trap the family's own note records hitting.
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.actuator_names = list(FIVE_FOOT_JOINTS)
    action.scale = {j: action.scale[j] for j in FIVE_FOOT_JOINTS}

    # ── 3. The observation: 21 joints, not 20 ─────────────────────────────
    # Four more than the action (the carried arm) plus the claw's finger. This is
    # where kk-rl-lab's `gripper` command observation ends up: the deployment
    # vocabulary is fixed on the far side of the contract, so a new command term
    # cannot be exported, but the arm's angles and the claw's aperture are
    # encoder readings and travel perfectly well inside `joint_pos`. `claw.py`
    # has the full argument.
    #
    # All three per-joint terms take the same set, because the deploy contract
    # has one `obs_joint_order` and the on-robot builder sizes `joint_pos`,
    # `joint_vel` and `joint_torque` from it (`deploy/fsm/src/obs.rs`). Give
    # them different sets and the observation is rebuilt at the wrong offsets on
    # the robot, with nothing raised anywhere.
    observed = SceneEntityCfg("robot", joint_names=list(OBSERVED_JOINTS))
    observed_actuators = SceneEntityCfg("robot", actuator_names=list(OBSERVED_JOINTS))
    for group in cfg.observations.values():
        for name in ("joint_pos", "joint_vel"):
            term = group.terms.get(name)
            if term is not None:
                term.params = dict(term.params or {}) | {"asset_cfg": observed}
        force = group.terms.get("actuator_force")
        if force is not None:
            force.params = dict(force.params or {}) | {"asset_cfg": observed_actuators}

    # ── 4. The carried arm: where it starts, and what holds it there ──────
    # The default pose is the hold -- stowed since 2026-09-28, `claw.py::LF_GRASP`
    # under its old name -- which makes it the arm's entry in `default_joint_pos`
    # and therefore the `default_joint_pos` the deploy contract carries: the pose
    # the robot ramps to, and the one its `joint_pos` observation is measured
    # from. `get_jumper_robot_cfg` returns a fresh EntityCfg per call,
    # so this does not reach the other four tasks.
    robot.init_state.joint_pos = dict(HOME) | dict(LF_GRASP)

    # Re-scope the inherited joint randomiser to the driven joints, so it does not
    # move the arm out from under the hold. (It samples from a zero-width range by
    # default, so today this changes nothing -- but a task that widens it would
    # otherwise be randomising the claw as well, which is what the hold below is
    # for.)
    joints = cfg.events.get("reset_robot_joints")
    if joints is not None:
        joints.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(FIVE_FOOT_JOINTS)
        )

    # The hold itself. Added last, so it runs after `reset_robot_joints` -- the
    # event manager keeps declaration order and dicts keep insertion order, so
    # "added last" is the whole ordering mechanism here.
    #
    # The finger gets a range of its own rather than a box around a nominal: it is
    # the claw's aperture, and the point of sampling it is that the policy sees
    # the whole open-to-shut travel across episodes instead of one angle. The two
    # ends are the measured working range (`claw.py`), so nothing outside the
    # linear part of the aperture curve is ever asked for.
    # (low, high), and on this joint more-open is the *lower* number: the finger
    # travels from GRIPPER_OPEN (-0.65) up to GRIPPER_CLOSED (+0.10), which on the
    # corrected aperture curve is the whole of the claw's travel rather than the
    # middle third of it.
    #
    # **In replay it is not sampled at all: the claw starts wide open.** An
    # operator wants to find the claw in one known state, and a demonstration that
    # begins with the jaws half shut for no visible reason reads as a fault. It is
    # also the state the rest of the task already names as the claw's nominal --
    # `LF_GRASP[FINGER_JOINT]` is GRIPPER_OPEN, so this only makes the reset agree
    # with the `default_joint_pos` the config has been declaring all along -- and
    # it is where `gripper_teleop` puts the claw for a trigger at rest, so the
    # operator's first touch finds it where the trigger says it is.
    #
    # **Training keeps the randomisation.** The finger is an observed joint, so
    # pinning it there would leave a constant column in `joint_pos` -- what the
    # note in `common/mdp/observations.py::actuator_force` warns against -- and,
    # now that an operator can close the claw, would train a policy that has never
    # walked with a closed claw and then let one close it.
    finger_range = (GRIPPER_OPEN, GRIPPER_OPEN) if play else (GRIPPER_OPEN, GRIPPER_CLOSED)
    hold_ranges = tuple(LF_GRASP_BOX[j] for j in ARM_JOINTS) + (finger_range,)
    cfg.events["hold_carried_arm"] = EventTermCfg(
        func=hold_carried_arm,
        mode="reset",
        params={
            # `preserve_order` is load-bearing: `ranges` is paired with these
            # joints positionally.
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(CARRIED_JOINTS), preserve_order=True
            ),
            "ranges": hold_ranges,
        },
    )

    # ── 5. What the claw is carrying ──────────────────────────────────────
    # A load in the mouth, fixed per environment once it is loaded: a mass **and
    # the inertia of an object that mass** (`mdp/events.py::claw_payload`), on a
    # body parked where a held object actually sits. `claw.py::PAYLOAD_BODY`
    # records what the point mass it replaced was doing instead, and
    # `PAYLOAD_RANGE` carries the measured cost of the load and the reason it
    # cannot be there from the first iteration.
    #
    # The body is added to the robot's spec here rather than in the asset,
    # because it belongs to this task: a jumper walking on six feet carries
    # nothing, and a body with mass on every robot in the repository is not a
    # thing one task gets to do. Wrapping `spec_fn` is how a task edits the model
    # it was handed -- the same shape `scenes/registry.py` uses for scenes.
    robot_cfg = cfg.scene.entities["robot"]
    _base_spec_fn = robot_cfg.spec_fn
    robot_cfg.spec_fn = lambda: add_payload_body(decompose_jaws(_base_spec_fn()))

    # The jaws, decomposed so the claw can pinch (`jaws.py` has the measurement),
    # and by the same argument as the payload: the shared asset gives each jaw one
    # hull, which is right for the tasks that walk on this claw. The scheme changes
    # with the spec -- under the shared one every jaw face is condim 1 and a pinch
    # runs at the prop's friction rather than the rubber's (`jaws.py::JAW_COLLISION`).
    robot_cfg.collisions = (JAW_COLLISION,)

    # **Training only.** In replay the claw is empty until the operator puts
    # something in it: a prop it grasps, or `play --hold`. An invisible 0-600 g
    # sampled at reset is a robot whose posture is wrong for a reason nothing on
    # screen accounts for -- and when the operator does then pick something up,
    # the two masses stack. Measured before this gate: a replay session was
    # carrying 311 g of nothing.
    #
    # **In training the claw starts empty and a curriculum loads it**
    # (`mdp/curriculum.py`): once the commands are on their top rung, have been
    # for `PAYLOAD_DWELL_ITERATIONS`, and the policy tracks there under
    # `PAYLOAD_LIN_ERR_BAR`, every environment takes a payload from
    # `PAYLOAD_LEVELS[1]` at its next reset. It used to be a stage you resumed
    # into by editing `PAYLOAD_RANGE`, and master sat on that stage's value, so a
    # fresh run was a cold start with the load on -- the one measured to stand
    # still.
    #
    # The startup event stays, sampling rung 0's (0, 0). What it contributes is
    # its `requires_model_fields`: that is what makes `body_mass` and
    # `body_inertia` per-world, and the curriculum loads through it and refuses
    # to run without it.
    if not play:
        cfg.events["claw_payload"] = EventTermCfg(
            func=claw_payload,
            mode="startup",
            params={
                "body_name": PAYLOAD_BODY,
                "ranges": PAYLOAD_LEVELS[0],
                "unit_inertia": PAYLOAD_UNIT_INERTIA,
            },
        )
        # After the command term, which it reads: the curriculum manager calls
        # terms in insertion order, so a promotion of the commands is seen on the
        # same reset rather than the next.
        assert "command" in cfg.curriculum, "the payload curriculum waits on it"
        cfg.curriculum["payload"] = CurriculumTermCfg(
            func=PayloadCurriculum,
            params={
                "levels": PAYLOAD_LEVELS,
                "lin_err_bar": PAYLOAD_LIN_ERR_BAR,
                "dwell_steps": PAYLOAD_DWELL_ITERATIONS * NUM_STEPS_PER_ENV,
            },
        )

    # ── 6a. Re-scope the inherited terms that would score the carried arm ──
    # `pose` rewards joints for staying near `default_joint_pos` and
    # `dof_pos_limits` penalises them for approaching their limits. Both would
    # charge the policy for the arm, which it does not drive: the hold is
    # re-sampled inside a +-0.10 rad box every episode, so `pose` would be reading
    # a random offset the policy can neither cause nor correct. Left in, it is a
    # noise term on the reward with no gradient attached.
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=list(FIVE_FOOT_JOINTS)
    )
    cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=list(FIVE_FOOT_JOINTS)
    )

    # ── 6a2. Drop the family's three dead terms ───────────────────────────
    # `body_ang_vel`, `angular_momentum` and `air_time` arrive from the skeleton
    # at weight 0 and have never been switched on in this task. At weight 0 a term
    # still runs, so each one costs a tensor op per step and puts a flat line into
    # `Episode_Reward/*` for a reader to scroll past.
    #
    # **Zero is not always the same as absent, which is why this is a deliberate
    # removal and not a tidy-up.** `five_foot_gait` spent a run at weight 0 on
    # purpose, because its value is the readout of whether the gait grouping holds
    # even when nothing is optimising it. These three have no such use here: no
    # measurement in this task's history has ever quoted one. If one is wanted
    # back, put it back at weight 0 rather than leaving it as a permanent
    # placeholder for a term nobody reads.
    for dead in ("body_ang_vel", "angular_momentum", "air_time"):
        cfg.rewards.pop(dead, None)

    # ── 6b. Non-foot ground contact ───────────────────────────────────────
    # `constants.py` prepared `TIP_GEOMS` for exactly this and left the decision
    # to training results: colliding is physical correctness, penalising is a
    # change to the learning problem. On five feet the case is no longer open.
    # kk-rl-lab runs the equivalent term at -4 and records why it had to double it
    # from the six-foot value: at a standstill the policy was **sitting on its
    # shins and its claw** instead of standing on its feet. That is available here
    # in the same way, and it is worse here, because the machine has one fewer leg
    # to stand on and a claw hanging 40 mm off the floor at the front.
    #
    # The sensor mirrors the self-collision one in shape -- one buffer entry per
    # physics substep -- so `self_collision_cost` reads it unchanged: it counts
    # substeps in which anything on this list pushed against the ground harder
    # than the threshold, 0..4 per policy step.
    ground_contact = ContactSensorCfg(
        name=GROUND_CONTACT_SENSOR,
        primary=ContactMatch(mode="geom", pattern=NON_FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=cfg.decimation,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (ground_contact,)
    # This sensor watches many more geoms than the feet one, and `maxmatch` is a
    # per-world cap on contact matches across every contact sensor. A robot on its
    # side can put a dozen of these geoms on the ground at once.
    cfg.sim.contact_sensor_maxmatch = 256

    # -0.5 per colliding substep, so at most -2.0 per policy step against the 4.0
    # the tracking terms are worth. Twice the family's self-collision weight: legs
    # brushing each other is untidy, a limb taking the robot's weight is the
    # failure this is here to prevent.
    cfg.rewards["ground_contact"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.5,
        params={
            "sensor_name": GROUND_CONTACT_SENSOR,
            "force_threshold": GROUND_CONTACT_FORCE_THRESHOLD,
        },
    )

    # ── 6c. The five-foot terms ───────────────────────────────────────────
    # Each is derived and argued in `mdp/rewards.py`; here is only what the
    # weights are and why they are that size. The scale to keep in mind: velocity
    # tracking is 2.0 + 2.0, `upright` and `pose` are 1.0 each, and the gait prior
    # is 1.0.
    #
    # kk-rl-lab's weights are not carried over, and cannot be: its reward set has
    # a positive budget of about 9 and a set of penalties an order of magnitude
    # larger than this one's, so a weight means a different fraction of the total
    # there. What is carried over is the *ordering* -- which of these matters more
    # than which -- and the fact that none of them is meant to outweigh tracking.
    #
    # **None of these weights has been trained with.** They are sized by
    # arithmetic against the terms around them, which is the honest description;
    # the first run's reward breakdown is what should move them.

    # +1.5. Load balance between the two **gait groups** -- `GROUP_A` (LM, RR)
    # against `GROUP_B` (RM, LR), the same partition `five_foot_gait` scores. RF is
    # in neither: it bears through both phases.
    #
    # This replaces a left/right version (LM+LR against RM+RR), which asked
    # whether the machine was listing to one side. The question here is different
    # and closer to what has actually been going wrong: **does each phase of the
    # gait take its turn carrying the robot, or does one group work while the
    # other rides along?** Measured on `model_8599`, RF -- then in group A -- was
    # on the ground 53% of the time and took 8.3% of the load against an even share
    # of 20%, and nothing in the reward set saw it.
    #
    # **Averaged, and the average is conditional on bearing.** Two things had to
    # be got right. Instantaneously the groups are *supposed* to be unbalanced --
    # one is in stance while the other swings -- so a per-step version would be
    # minimised exactly where `five_foot_gait` is maximised. And an
    # *unconditional* average divides each group's stance force by its duty cycle,
    # so it reports as a load imbalance what is really a difference in how long
    # each group stands -- which `duty_balance` already scores, making the two
    # terms measure one thing, one of them badly.
    #
    # Advancing each group's average only while it bears removes the duty cycle
    # from the comparison. What is left is the question worth asking: when it is
    # your turn, do you carry as much as the other group carries on its turn?
    #
    # Same weight as the term it replaces, and the same shape -- a relative gap
    # through an exponential, gated on the command -- so the two are comparable
    # across runs. What is not comparable is the *value*: this one averages, so it
    # does not swing with the gait the way the old one did.
    cfg.rewards["lateral_load"] = RewardTermCfg(
        func=group_load_balance,
        weight=1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "group_a": GROUP_A,
            "group_b": GROUP_B,
            "command_name": "twist",
            "alpha": 0.01,
            "force_threshold": 2.0,
            "command_threshold": 0.05,
        },
    )

    cfg.rewards["duty_balance"] = RewardTermCfg(
        func=stance_duty_balance,
        weight=1.75,
        params={
            "sensor_name": "feet_ground_contact",
            "feet": MID_REAR_FEET,
            "command_name": "twist",
            "std": 0.15,
            "alpha": 0.01,
            "force_threshold": 2.0,
            "command_threshold": 0.05,
        },
    )

    # -1.0. RF's stepping window: it lifts only with the four gait legs planted.
    # With either diagonal pair it stands on a line 86 mm from the centre of mass;
    # alone it is 18 mm short of static and a 30 mm trunk shift back makes it +5
    # (`mdp/rewards.py::GROUP_A`). `five_foot_gait` pays for that third phase; this
    # charges the other two, per control step RF spends airborne out of it.
    #
    # Sized against the habit it has to break. Measured on
    # `2026-09-22_17-40-51/model_21996`, 32 environments walking, contact lost as
    # airborne: RF was off the ground 41% of the time at 0.4 m/s, 33% at 0.25, 20%
    # backing up and sideways, 15% turning -- and **every** control step of it with
    # a gait leg off the ground too, while the four were planted together only
    # 0-1.7% of the time walking. So the window barely exists in the gait as
    # learned, and the policy has to make one: at -1.0 the old timing costs 0.15 to
    # 0.41 per second, the same order as `five_foot_gait`'s whole 0.5.
    #
    # What this does not do is make RF step: a leg with nowhere cheap to lift drags
    # instead. `rf_drag` below is the other half.
    cfg.rewards["rf_step_window"] = RewardTermCfg(
        func=rf_step_window,
        weight=-1.0,
        params={"sensor_name": "feet_ground_contact"},
    )

    # -2.0. RF may not be carried along the floor. It already was: 134 mm/s of
    # slide while touching at 0.4 m/s forward, 177 backing up, lifts of 3-7 mm
    # (`model_21996`, `tools/gait_check.py`), and `foot_slip` prices that at 0.0018
    # per second. Linear and in contact only, so the step through the air is free
    # and the slide on the floor is not; walking only, because standing is
    # `feet_still`'s. At -2.0 the measured habit costs 0.16 to 0.28 per second,
    # and dragging RF the whole way at 0.4 m/s would cost 0.8 -- the price of
    # never lifting it, which the window alone left at zero.
    cfg.rewards["rf_drag"] = RewardTermCfg(
        func=rf_drag,
        weight=-2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", site_names=("RF",)),
        },
    )


    # -1.0 on a cost that is metres of foot displacement summed over five feet,
    # outside a 40 mm deadzone. A stance splayed 40 mm past the zone on every foot
    # costs 0.2 per step -- noticeable against tracking's 4.0 and nowhere near
    # able to outbid it, which is the intended size: this is a shape constraint on
    # a walking policy, not a reason to stop walking.
    # **The one term that scores a standing stance.** Every other support term
    # here -- `five_foot_gait`, `lateral_load`, `duty_balance` -- is multiplied by
    # `moving_gate` and is therefore exactly zero when the command is ~0, so until
    # this went in nothing in the task had an opinion about a stationary robot's
    # feet at all. It pays per planted foot, so lifting one at a standstill loses
    # a fifth of it.
    #
    # **A positive term a motionless robot collects in full is the shape this
    # reward set otherwise refuses**, and the exemption has to be argued rather
    # than assumed. The argument is that the gate is on the *command*: the term is
    # not paying the robot for standing, it is paying it for standing **still**
    # once standing is what was asked for. It can never make standing more
    # attractive than obeying a non-zero command, because the two are never
    # simultaneously available.
    #
    # +1.0, kk-rl-lab's weight, against five feet rather than scaled up -- for the
    # same reason the exemption needs arguing.
    #
    # **What it does not do is fix load sharing.** It counts feet in *contact*, so
    # a foot resting on the ground bearing almost nothing still scores. Measured
    # on `model_8599`, RF is down 53% of the time and carries 8.3% of the load
    # against an even share of 20% -- this term would score that stance as
    # perfect. The quantity that sees it is per-foot force share, which nothing
    # here currently measures.
    cfg.rewards["feet_planted"] = RewardTermCfg(
        func=feet_contact_without_cmd,
        weight=1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.1,
        },
    )

    cfg.rewards["foot_home"] = RewardTermCfg(
        func=foot_home_position,
        weight=-1.0,
        params={
            # The foot sites, in `FIVE_FOOT_LEGS` order to match `HOME_TOE_XY`;
            # `preserve_order` because that pairing is positional. `build_jumper.py`
            # names each foot site by its leg prefix alone, so the leg tuple is
            # already the site tuple -- the same sites `foot_clearance` and
            # `foot_slip` are given in `common/velocity_env.py`.
            "asset_cfg": SceneEntityCfg(
                "robot",
                site_names=FIVE_FOOT_LEGS,
                preserve_order=True,
            ),
            "home_xy": HOME_TOE_XY,
            "dead_radius": 0.04,
            # **The commanded footprint frame**, or this term vetoes the twist
            # command: a trunk twisted 30 degrees over stationary feet sees every
            # foot move 103 mm at a 200 mm foot radius, against a 40 mm deadzone.
            # `mdp/rewards.py::_footprint_xy` rotates the measurement by the same
            # angle, so what is scored stays the *shape* of the stance. It is tied
            # to `body_twist` being registered: with nothing scoring twist, this
            # rotation would charge the term for a command nobody follows.
            "command_name": POSE_COMMAND,
        },
    )

    # RF's foothold, pushed out from under the body. -10.0 on metres of shortfall
    # against the trunk's own half-width, one-sided: zero once the foot is outside
    # the body's footprint, and free to go further.
    #
    # RF is the only ground foot that stands inside the machine's shadow -- 36.5 mm
    # against a body 90.7 mm wide either side, where the other four are at 161 and
    # 200 mm -- and with LF carried it is the *whole* front support. Per step:
    #
    #     foot y   +0.050   0.000   -0.037   -0.060   -0.091   -0.120
    #     cost     -1.407  -0.907   -0.542   -0.307    0.000    0.000
    #
    # So the HOME stance costs 0.54 per step against tracking's 4.0 -- enough to
    # be worth 0.4 rad of shoulder yaw, which is what the move takes, and nowhere
    # near enough to outbid walking. Crossing the centreline costs more the further
    # it goes, which is the case the signed form exists for: an unsigned |y| would
    # score a foot that had swung right across the body as if it were fine.
    cfg.rewards["rf_outboard"] = RewardTermCfg(
        func=foot_outboard_of_trunk,
        weight=-10.0,
        params={
            # Index 0 of FIVE_FOOT_LEGS = (RF, LM, RM, LR, RR); the same sites, in
            # the same order, as `foot_home` above.
            "foot_index": 0,
            "min_offset": TRUNK_HALF_WIDTH,
            # RF is on the right, where outboard is -y.
            "outboard_sign": -1.0,
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=FIVE_FOOT_LEGS, preserve_order=True
            ),
            # The same frame as `foot_home`, and for the same reason.
            "command_name": POSE_COMMAND,
        },
    )

    # ── The attitude command is scored for accuracy, on all three axes ────
    # All three penalties were switched off for one run after
    # `2026-09-10_12-03-37` stopped walking, and all three are back, because the
    # thing that paid a parked robot was not them: it was `track_body_pose`
    # collecting 0.957 of its 1.0 for standing still and holding any attitude
    # perfectly. That is fixed at the term itself, by an offset that makes perfect
    # attitude worth zero instead of one. These three are penalties; a standing
    # robot cannot collect them, and the toll they put on walking is the price of
    # attitude being accurate while the robot moves, which is what this command
    # exists for.
    #
    # **They charge the error against the command, not the angle.** `command_name`
    # is what makes that so; without it a term charges absolute pitch and then
    # fights `body_pose`, which asks for up to 15 degrees of it -- an absolute
    # penalty and a command pulling opposite ways settle wherever they balance,
    # which is neither. With it, obeying the command is free and only deviation
    # costs, measured: 10 degrees nose-down costs 0.395 against a level command
    # and exactly 0.000 against a command asking for 10 degrees nose-down.
    #
    # **The deadzone is 0.01 rad, 0.57 degrees.** Inside it these three terms are
    # exactly zero. It was 0.03 rad -- 1.72 degrees -- for most of this task's
    # history, and that showed: `2400.pt` held a +0.85 degree standing pitch bias
    # in every one of seven commanded directions, parked inside its own free band,
    # because nothing asked for better.
    #
    # It is not zero, which is where kk-rl-lab's plain `err^2` sits, because these
    # terms carry a *linear* piece as well as a quadratic one. At threshold 0 the
    # linear piece has a kink at the origin -- a constant 0.26 per degree pull at
    # any error whatever -- and a walking trunk is never still: `2400.pt` wobbled
    # about a degree in the mean with peaks near 4. A small deadzone keeps that
    # kink off the wobble and still asks for three times the accuracy the old one
    # did.
    #
    # **The far field is these three; the near field is `track_body_pose`**, which
    # has no deadzone at all and is steepest inside 3 degrees. That is what makes a
    # small deadzone affordable here and would not have been true before it was
    # registered.
    #
    # kk-rl-lab prices the same two axes at -20 roll and -15 pitch on a plain l2,
    # with no deadzone, alongside +2.5 of positive attitude tracking. The weights
    # here are this task's own -- the shapes differ -- but the ordering it chose,
    # roll no cheaper than pitch, is the one thing measurement here agrees with
    # loudly: at 42x apart the policy parked in a 2.39 degree list.
    #
    # -15.0 on radians of pitch past 1.72 degrees **in either direction**,
    # quadratic past that. Per step:
    #
    #     |pitch err|  1 deg   1.72 deg   2.93 deg   5 deg   10 deg
    #     cost          -0.00    -0.00    -0.40    -1.45    -5.93
    #
    #     |pitch|    1 deg   1.7 deg   2.93 deg   5 deg   6.62 deg   10 deg
    #     cost       0.000    0.000     -0.398   -1.449    -2.600    -5.928
    #
    # **This has been one-sided twice, in both directions, and neither held.** It
    # started as a nose-down penalty on the mechanical argument that with LF
    # carried, pitching down loads a front held up by RF alone. Measurement
    # reversed it: `model_6400` sat at -2.98 degrees mean *nose-up* with 50.5% of
    # steps past 3 degrees, because leaning back is how the machine offloads that
    # same foot, and a nose-down-only penalty left it free. So the sign flipped --
    # and then `model_2200` came back at **+2.93 degrees mean nose-DOWN** with a
    # maximum nose-up of 0.19. The penalty eliminated the half it charged for and
    # the policy moved into the half it did not, twice.
    #
    # What the term is really guarding is not a direction: it is the machine's
    # tendency to park at whatever pitch takes weight off its single front foot,
    # and it will use whichever side is free. Symmetric closes both.
    #
    # 2.93 degrees costs 0.40 per step under this form, against 0 before -- so it
    # is a live cost on the posture the current policy actually holds, not a guard
    # rail. The asymmetry that nose-down is the more dangerous half is left to
    # `upright`, `low_stance` and the support polygon, which is where it was
    # always really enforced.
    cfg.rewards["nose_pitch"] = RewardTermCfg(
        func=nose_pitch,
        weight=-15.0,
        # **Against the command, not against level.** The weight is unchanged, the
        # reference is not: with a command in play an absolute penalty and a
        # request for 11 degrees of nose-down are two terms pulling opposite ways,
        # and the policy settles where they balance rather than where either
        # asked. At the neutral command -- which is what a quarter of the field is
        # pinned to, and what every run before this one had -- this is exactly the
        # term that has been running all along.
        params={"command_name": POSE_COMMAND},
    )

    # Back on with the other two -- see the note above `nose_pitch`. Roll off and
    # pitch on was measured for one config and priced ten degrees of pitch at 6.56
    # against roll's 0.63, a factor of 10.4: the same shape as the 42x that
    # produced a parked 2.39-degree list, one order smaller. At -12 the two axes
    # are 1.2 apart. What -12 costs is also measured and is not nothing -- at -10,
    # roll improved in all seven commanded directions and linear tracking fell
    # from 95.4% to 88.1% -- and it is spent deliberately here, because attitude
    # accuracy is what this command is for. `Metrics/body_pose/error_roll` is the
    # readout that says whether it bought anything.
    #
    # -12.0 on radians of **roll** past 1.72 degrees, either way -- `nose_pitch`
    # for the other axis, same threshold and same quadratic, four fifths the
    # weight.
    #
    #     |roll err|   1 deg   1.72 deg   2.93 deg   5 deg   10 deg
    #     cost          -0.00    -0.00    -0.32    -1.16    -4.74
    #
    # **-10 was tried first and it worked, at a price.** Measured over seven
    # commanded directions, 450 steps each, against `model_2400` which had no roll
    # term at all:
    #
    #     |roll| mean      -10       none        vx / vy / wz tracking
    #     standing        0.43      0.71         86% / 65% / 76%   against
    #     forward         0.42      0.99         94% / 93% / 96%
    #     strafe left     0.31      1.69
    #     turn left       0.32      2.32
    #
    # Roll improved in all seven -- the standing list and the -3.36 degree turning
    # list both went -- and **tracking fell in all six moving ones**, worst on
    # lateral (93% -> 65%) and turning (96% -> 76%). Those are precisely the
    # directions that need the trunk to lean to balance, so the term was buying
    # attitude with manoeuvrability.
    #
    # Measured on the two runs' checkpoints under held commands, seven directions,
    # 16 environments each, 3 s to settle and 7 s to measure:
    #
    #     roll weight    |roll| mean    linear tracking (mean of 7)
    #     none (2400)       1.27 deg          95.4%
    #     -10  (2000)       0.79 deg          88.1%
    #
    # -10 bought 38% of the roll for 52% more tracking error (0.061 -> 0.093 m/s),
    # worst going backwards, 90% -> 73%. **-12 goes past that deliberately.** A
    # level trunk is what this machine is for -- it carries a claw out to the left
    # and will be asked to hold something in it -- and the tracking is the price
    # named for it. What the weight must not do is make the transient lean of a
    # turn expensive, and it does not: 1 degree is still free, 3 degrees costs
    # 0.34 against the 4.0 tracking is worth. **The run after this one is read
    # against that 88.1%**; -5 and -3 are the cheaper answers if it falls further
    # than -10 did.
    #
    # **The two axes were priced 42 times apart and the policy noticed.**
    # `upright` was the only term scoring roll angle, and at std 0.447 it is very
    # loose: 10 degrees of roll cost it 0.14 while 10 degrees of pitch cost
    # `nose_pitch` 5.93. Measured on `model_2200` over seven commanded directions,
    # standing produced a **+2.39 degree list to the right that never moved** --
    # 0.05 degrees peak to peak, a posture rather than a wobble. On `model_2400`,
    # once symmetric `nose_pitch` had pulled pitch down, standing straightened to
    # +0.38 and the list moved to the commanded phases instead: -3.36 degrees
    # turning left, -2.92 strafing right, near-constant within each.
    #
    # That is the behaviour `nose_pitch` was written twice to catch, on the axis
    # nothing was watching. And this machine has a reason to prefer one side: LF is
    # carried out to the left, `rf_outboard` pushes RF out to the right.
    # -12.0, four fifths of `nose_pitch`, and both are back at the scale they had
    # when the target was level rather than commanded.
    #
    # **They were cut to a tenth for one run, and the ramp is what undoes the
    # cut.** The reason for cutting them was the step: `body_pose` resampled every
    # 5 seconds straight to its new value, up to 40 degrees at once, and a
    # linear-plus-quadratic penalty then charges for a target the trunk cannot
    # reach -- 1.4 per step out of a 4.0 tracking budget, measured on
    # `2026-09-10_14-20-45`, with the cheapest answer available to the policy being
    # to stop walking and mind the trunk. Cutting the weight made that cheap, and
    # made the command cheap to ignore with it: `Metrics/body_pose/error_pitch`
    # went from 1.2 degrees at -15 to 3.0 at -1.5, while walking barely moved
    # (0.406 to 0.426, against 0.633 for this same reward set with no attitude
    # command at all).
    #
    # `BodyPoseCommand.max_rate` fixes the transient at its source -- the command
    # walks to its target at 30 deg/s, so what these terms charge is tracking error
    # rather than a teleport -- which is what makes the full weight affordable
    # again. If a run still trades walking for attitude, the ramp rate and these
    # weights are the two knobs, in that order.
    #
    #     |err|       0.57 deg   1 deg   2 deg   3 deg    5 deg   10 deg
    #     nose_pitch    -0.00   -0.12   -0.49   -0.96   -2.23   -7.34
    #     body_roll     -0.00   -0.10   -0.39   -0.77   -1.79   -5.87
    #
    # Roll is registered again with it, because the two axes have to be priced
    # together: with `nose_pitch` alone, ten degrees of pitch costs 0.59 and ten of
    # roll costs `upright`'s 0.14, and a policy charged four times more for one
    # axis than the other leans on the cheap one -- measured at 42x it produced a
    # parked 2.39 degree list.
    cfg.rewards["body_roll"] = RewardTermCfg(
        func=body_roll,
        weight=-20.0,
        params={"command_name": POSE_COMMAND},
    )

    # -3.0 on a squared fraction of the stance height, so a body 20 mm low costs
    # 0.11 per step, 50 mm low costs 0.68, and one flat on the ground costs 3.0 --
    # at which point `too_low` has already ended the episode. Zero at the nominal
    # stance, which is what makes it a penalty rather than another thing a
    # motionless robot collects.
    cfg.rewards["low_stance"] = RewardTermCfg(
        func=low_stance,
        weight=-3.0,
        params={"target_height": STAND_Z},
    )

    # **Standing still, when standing still is what was asked for.** Measured on
    # `model_6999` with the command held at zero for four seconds: the robot
    # wanders 26.5 mm and 3.6 degrees of yaw from where it stopped, and nothing
    # in the reward charges it -- `standing_sway` squares the body velocity, so
    # 20 mm/s of creep costs it 4e-4 before its -0.1 weight. This is kk-rl-lab's
    # answer to the same thing (`stand_still_when_idle`, -1.5 there): an L1 pull
    # on the joints back to the stance, ramped in as the command approaches zero
    # and off entirely while an attitude is commanded.
    #
    # -1.5 is the reference's weight and it lands in the same place here.
    # Measured on that policy: the term reads 0.421 rad idle (16 joints, about
    # 1.5 degrees each), 0.002 walking and 0.000 turning -- so it costs -0.63 of
    # a standing step's ~12 and **nothing at all** to a walking one. That last
    # number is the one that matters on this robot: a term that made standing
    # cheaper than walking would feed the failure this whole reward set is
    # arranged against.
    # **The reward set is the one `2026-09-11_15-11-16` trained with**, on
    # request: that run is the walk this task is judged against, and the two
    # idle-gated terms added since it -- `stand_height_hold` for the stance
    # height the attitude command was being paid for with, and
    # `stand_still_when_idle` for the 26 mm of creep at a zero command -- are not
    # registered here.
    #
    # Both are still in `mdp/rewards.py` with their measurements, and both are
    # tested; putting them back is two `RewardTermCfg` blocks. See the commit
    # that removed them for what they cost and what they bought.

    # ── 7. Lying down ends the episode ────────────────────────────────────
    cfg.terminations["too_low"] = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": MIN_STAND_HEIGHT},
    )

    # ── 8. In replay, the operator drives the claw ────────────────────────
    # `claw_left` closes it as far as it is held -- `LT` on the pad, Space on the
    # keyboard -- because `controls.yaml` declares it for the claw. Replay-only, exactly like the operator's walk and for the same two
    # reasons: during training there is no viewer or pad for an input to arrive
    # from, and a term that can be driven from outside the process makes a run
    # unreproducible. **Nothing here touches the training problem** -- no
    # reward, observation, action or command changes -- which is why it survives a
    # revert of the reward set.
    #
    # The finger stays out of the action. `mdp/gripper.py` opens with why: putting
    # it back would change the policy's input and output widths, invalidate every
    # checkpoint trained so far, and hand a joint to a network that was never
    # rewarded for anything it does with it.
    if play:
        # The walk needs nothing here: `velocity_env_cfg` already puts it on the
        # family's operator, from `controls.yaml`. A press is a tenth of a
        # stick's travel on every axis, so the top of this task's 2.0 rad/s yaw
        # is ten presses away, like the top of its walk -- the reason this task
        # once set its own `ang_step`.
        cfg.events["gripper_teleop"] = gripper_teleop_event()
        # **There is no grasp term.** The claw holds what it squeezes, by
        # friction, under the friction settings of the scene its props come with
        # (`tasks/jumper/five_foot/objects.py::PROP_CONE`). What is left is `play --hold`, which
        # starts an episode with a prop already squeezed in the mouth and does
        # nothing until asked. After the teleop, so that on the steps it shuts the
        # claw its target is the one that stands. See `mdp/grasp.py`.
        cfg.events["claw_hold"] = claw_hold_event()

    # ── 9. The trunk is told how to hold itself ───────────────────────────
    # A second command beside the velocity twist: pitch, roll, and a yaw the trunk
    # holds **over its own stationary footholds**. `mdp/pose_command.py` carries
    # the conventions, the bands and the keyboard; what is decided here is what it
    # costs and what the robot can see.
    #
    # **Why the machine needs it.** It carries a claw. A robot that can only walk
    # level can only reach what is already at its own height and angle; aiming is
    # done by leaning, and until now nothing ever asked this robot to lean.
    cfg.commands[POSE_COMMAND] = BodyPoseCommandCfg(resampling_time_range=(5.0, 5.0))

    # Observed by the actor **and** the critic, and it has to be both: a command
    # the policy cannot see is noise it is punished for not obeying. The actor's
    # observation grows by three, so **every checkpoint trained before it is
    # unloadable** -- that part is unavoidable and is the cost of the feature.
    #
    # **`base_pose` is the name, and it is not a free choice.** The deployment
    # vocabulary is fixed by the consumer: `deploy/fsm/src/obs.rs` builds each
    # term by name and already builds this one -- three channels, `base_pitch`,
    # `base_roll`, `base_twist`, in that order, which is `PITCH, ROLL, TWIST` --
    # and `scripts/export.py::_DEPLOY_TERMS` already lists it. Under any other
    # name the export refuses the contract and the feature stops at the last
    # metre, with a controller that could have filled it sitting right there.
    #
    # What that leaves is an order agreement between two languages that cannot
    # check each other. Three channels filled in the wrong order still load, still
    # run and lean the robot the wrong way; `tests/test_five_foot_pose_command.py` reads the
    # Rust and pins it.
    _pose_obs = ObservationTermCfg(
        func=mdp.generated_commands, params={"command_name": POSE_COMMAND}
    )
    for _group in cfg.observations.values():
        _group.terms["base_pose"] = _pose_obs

    # 1.0, on a kernel that tops out at **zero** rather than at one: the term is
    # `exp(-err^2/std^2) - 1`, so perfect attitude is worth nothing and everything
    # else is a cost. Pitch and roll only; twist is below, and it needs the
    # footholds rather than the IMU.
    #
    # **std 0.10 rather than 0.175**, which is kk-rl-lab's value for the same
    # kernel and is the accuracy setting: half the top of the commanded range
    # instead of the whole of it, so the curve is steep where the errors this
    # command cares about live.
    #
    #     |err|      0.5 deg   1 deg   2 deg    5 deg   10 deg
    #     std 0.175   -0.002  -0.010  -0.039   -0.220  -0.630
    #     std 0.10    -0.008  -0.030  -0.115   -0.533  -0.952
    #
    # **This is a positive term a standing robot can collect in full**, which is
    # exactly the shape that put two of this task's cold starts into a standing
    # basin: hold still, hold any attitude perfectly, collect. Those two runs had
    # 4.75 reachable without moving against 4.0 of tracking, and they sat down and
    # took it.
    #
    #     reachable standing   upright 1.0 + pose 1.0 + feet_planted 1.0 + this
    #     positive, weight 2.0   5.0
    #     positive, weight 1.0   4.0
    #     offset by -1 (now)     3.0    against 7.75 that requires moving
    #                                   (4.0 tracking + 3.75 gait and load, gated)
    #
    # Lowering the weight moved that number; it could not remove it, because a
    # positive attitude term is collectable standing at any weight. The offset
    # removes it: the most a parked robot can now get from attitude is zero.
    #
    # **Gating it on the velocity command was the other way to do that and is not
    # available here.** A gate scores attitude only while walking, and this robot
    # is asked to hold an attitude precisely when it is *not* walking -- it aims a
    # claw with its feet planted. The offset keeps both regimes scored.
    #
    # **The readout that separates them is
    # `Episode_Reward/track_linear_velocity` against `Episode_Reward/track_body_pose`:
    # a parked run shows this one high and that one flat, inside a few hundred
    # iterations.** Watch it early rather than at the end.
    # 2.5, kk-rl-lab's weight for the same kernel and the same std -- **minus one**.
    # `track_body_pose` is `exp(-(dpitch^2 + droll^2)/std^2) - 1`, so it tops out
    # at zero instead of at the weight.
    #
    # The reference pays this term, and its twist twin, +2.5 each: 5.0 of attitude
    # a robot collects in full by standing still and holding whatever it is
    # holding. **This task has parked on exactly that shape twice**, at 4.75
    # against 4.0 of tracking. The offset keeps the reference's gradient -- a
    # constant differentiates away, so the pull toward the commanded attitude is
    # bit-identical -- and changes only what the pull is worth: obeying is 0,
    # everything else is negative.
    #
    #     |err|      0.57 deg   1 deg   2 deg   5 deg   10 deg
    #     cost         -0.025   -0.075   -0.287   -1.333   -2.381
    #
    # Why a near-field term at all, when `nose_pitch` and `body_roll` already
    # charge the error: those two have a 1.72 degree deadzone and are nearly flat
    # inside 3 degrees, which is precisely the band this command has to be accurate
    # in. This kernel is steep exactly there and saturates past 15 degrees, where
    # they take over. The two halves are complementary by construction.
    #
    # Pitch and roll only -- twist needs the footholds rather than the IMU, and is
    # scored by `body_twist` below.
    cfg.rewards["track_body_pose"] = RewardTermCfg(
        func=track_body_pose,
        weight=2.5,
        params={"command_name": POSE_COMMAND, "std": 0.10},
    )

    # The third axis, in the same shape as `nose_pitch` and `body_roll` -- a 1.7
    # degree deadzone, then linear plus quadratic in the error -- and priced
    # between them, as kk-rl-lab's port priced it between its own two:
    #
    #     twist error   1 deg   1.7 deg   5 deg    15 deg    30 deg
    #     costs         0.000    0.000    1.304    11.834    46.133
    #
    # **A penalty, never a payment**, unlike kk-rl-lab's `exp(-err^2/std^2)` for
    # this axis. Theirs saturates past about two std, so a policy 30 degrees out
    # sees no gradient -- and 30 degrees of twist is the one attitude on this
    # machine that costs nothing mechanically, feet planted and legs shearing, so
    # it is a target worth still pulling on.
    # Back on with the other two -- see the note above `nose_pitch`. This one is
    # **the whole of the twist axis's scoring**: `track_body_pose` is pitch and
    # roll only, because twist is not visible to an IMU, so at 0.0 the third
    # channel of `body_pose` was a command the policy was shown and never graded
    # on. Per step, past the 1.72 degree deadzone:
    #
    #     |twist err| 0.57 deg   1 deg   2 deg   3 deg    5 deg   10 deg
    #     cost          -0.00   -0.10   -0.39   -0.77   -1.79   -5.87
    # -1.2 with roll, and on the same scale for the same reason -- see the note at
    # `body_roll`. This term is **the whole of the twist axis's scoring**:
    # `track_body_pose` is not registered and would not help if it were, because it
    # is pitch and roll only. With it off, the third channel of `body_pose` was
    # sampled, observed and graded by nothing.
    #
    #     |twist err| 0.57 deg   1 deg   2 deg   3 deg    5 deg   10 deg
    #     cost          -0.00   -0.10   -0.39   -0.77   -1.79   -5.87
    #
    # Twist is the axis that costs this mechanism the least: the feet stay planted
    # and the legs shear, so unlike pitch there is no foot to lose. That is why it
    # is priced with roll rather than with pitch.
    # ── The deadzones are what a walking robot may spend ──────────────────
    # **These three terms froze two runs solid.** `2026-09-10_15-47-30` and
    # `2026-09-10_17-07-12` both stood perfectly still: swept at iteration 600 and
    # 700 over six commands, every one came back 0% -- 0.000 m/s against 0.5
    # commanded -- while `2026-09-09_16-46-37`, the same task without an attitude
    # command, was at 88% forward and 37% turning by iteration 700.
    #
    # The cause is not the weights, it is the deadzones. Measured on
    # `16-46-37/model_2400`, a policy that tracks at 95.4%, holding each command
    # for 6 s with the attitude command at neutral -- so every degree below is a
    # *walking gait*, not a command being disobeyed -- and priced at the weights
    # the frozen runs trained under (-15/-12/-12, deadzones 0.01 rad):
    #
    #     command       |pitch| |roll| |twist|   nose  roll  twist   total/step
    #     standing        0.39   0.78   2.27    -0.02 -0.08 -0.71    -0.81
    #     forward 0.35    0.83   1.11   4.87    -0.11 -0.16 -1.75    -2.01
    #     strafe 0.30     0.80   1.31   3.89    -0.11 -0.23 -1.20    -1.54
    #     turn 0.50       0.55   1.58   4.94    -0.05 -0.28 -1.97    -2.30
    #
    # **Walking cost 1.2 to 1.5 per step more than standing, and nine tenths of it
    # was `body_twist`.** Against that, walking earns at most the 2.0 of
    # `track_linear_velocity`, and a robot standing still under a small command
    # already collects 0.3 to 0.4 of it. The margin was near zero for a policy
    # that already walks perfectly and negative for one taking its first steps, so
    # the first steps never happened.
    #
    # `body_twist` is the one that has to move, because **a 3+2 gait turns the
    # trunk against its own footholds by construction**: two legs leave the
    # ground, the support polygon changes shape, and the Procrustes fit reads 4.9
    # degrees of twist that no policy can avoid and no operator asked for.
    # Charging that through a 0.57 degree deadzone and a quadratic at weight 12 is
    # a fine on taking a step.
    #
    # So: 0.09 rad (5.2 deg) of deadzone on twist and the weight down to -3, and
    # the other two back to the 0.03 rad (1.72 deg) they had for every run before
    # the command existed. Re-priced on the same measurement, the tax on walking
    # is gone -- -0.037 per step forward and -0.112 turning, against -0.016
    # standing -- while disobeying is still expensive: 20 degrees of pitch ignored
    # costs 23.1, and a 30 degree twist command ignored costs 8.1.
    # ── Velocity tracking carries kk-rl-lab's weight, not the family's ────
    # **The attitude block has to be outweighed by the thing it modifies.** The
    # reference five-legged task prices velocity at 4.0 linear + 5.0 angular +
    # 2.0 lateral = 11.0, against 40 of attitude (2.5 orientation, -20 roll, -15
    # pitch, 2.5 twist): a ratio of 3.6. This task was at 2.0 + 2.0 = 4.0 against
    # 32.5, a ratio of 8.1 -- twice as attitude-dominated as the configuration
    # these weights were measured on, on top of a penalty shape that priced the
    # same error 13 times higher (see `rewards.nose_pitch`).
    #
    # 4.0 and 5.0 are the reference's own numbers for the two terms this task
    # has. The lateral third is not reintroduced here: splitting linear tracking
    # per axis was tried and reverted, and `track_linear_velocity` is the family's
    # lumped term with the curriculum writing its std by that name.
    #
    # **5.0 is safe only because the angular term is yaw alone.** mjlab's scores
    # `exp(-(yaw_err^2 + roll_rate^2 + pitch_rate^2)/std^2)`, and at 5.0 that
    # would be a heavy tax on tilting to a new attitude command -- the same trap
    # kk-rl-lab avoids by relaxing `ang_vel_xy_l2` from -0.15 to -0.05.
    # `velocity_env.py` already swaps the func for `track_yaw_velocity`, so what
    # is being weighted here is the yaw error and nothing else.
    cfg.rewards["track_linear_velocity"].weight = 4.0
    cfg.rewards["track_angular_velocity"].weight = 5.0

    # ── The trunk keeps its height while the attitude moves ───────────────
    # **A nose-down command raises the body by 41 mm, and nothing charged for
    # it.** `low_stance` is the only other term that mentions height and it is
    # one-sided -- it charges sinking, not standing tall -- so it logged 0.0000
    # through every attitude in the measurement in `rewards.body_height_hold`,
    # including the 152 mm case against a 111 mm level stance.
    #
    # -5.0 is a nudge and not a hammer, on purpose. **Some of that rise is the
    # mechanism, not a fault**: with LF carried, pitching the trunk nose-down over
    # five planted feet needs the legs to extend, and a term heavy enough to
    # forbid it would be paid by the attitude command it is supposed to be
    # cleaning up after -- the trade this task has already lost twice, once to
    # `body_roll` at -10 and once to the whole attitude block. At -5 the measured
    # 41 mm excursion costs 0.27 per step against the 2.5 that `track_body_pose`
    # pays for the attitude itself, so obeying still wins and the height is
    # shaped rather than pinned.
    #
    # Whether 0.27 is enough to move it is a run, not an argument. The readout is
    # `Episode_Reward/body_height_hold` against the same held-command sweep.
    cfg.rewards["body_height_hold"] = RewardTermCfg(
        func=body_height_hold,
        weight=-5.0,
        params={"nominal_height": STAND_Z, "deadzone": 0.015},
    )

    # Standing, only the trunk's own turn counts: once the feet have settled, the
    # footprint's turn in the world is added back, so a twist the feet walk in
    # earns nothing (`mdp/rewards.py::body_twist`). The standing test and the
    # settle time are `feet_still`'s, so the two terms agree on when the robot is
    # standing and when its feet have landed.
    cfg.rewards["body_twist"] = RewardTermCfg(
        func=body_twist,
        weight=2.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=FIVE_FOOT_LEGS, preserve_order=True
            ),
            "home_xy": HOME_TOE_XY,
            "sensor_name": "feet_ground_contact",
            "command_name": POSE_COMMAND,
            "std": math.radians(15.0),
            "velocity_command_name": "twist",
            "command_threshold": 0.05,
            "settle_time": 0.5,
        },
    )

    # **The feet stay where they stand while only the attitude is commanded.**
    # `body_twist` scores the offset between the trunk and its footholds, which the
    # feet can produce as well as the trunk, and they did: two thirds of every
    # commanded twist was walked in by the feet (`mdp/rewards.py::
    # feet_still_when_standing` has the table). Gated on the velocity command, so
    # it is the complement of `feet_slip` and costs a walking policy nothing, and
    # ramped in over the half second after the command stops so that finishing a
    # step is not charged as moving while standing.
    #
    # -2.0 because nothing pays for the stepping it has to outweigh: held over
    # planted feet, the twist scores as well as standing square (0.00 per step at
    # 20 degrees, -0.09 at 30 from the mid hips' soft limit). The measured habit,
    # 0.10 m/s of summed foot speed at 20 degrees and 0.13 at 30, costs 0.20 and
    # 0.25 per second here -- three times the soft-limit charge it would save at
    # the edge of the band, and 5% of `track_linear_velocity`'s 4.0.
    cfg.rewards["feet_still"] = RewardTermCfg(
        func=feet_still_when_standing,
        weight=-2.0,
        params={
            "command_name": "twist",
            "command_threshold": 0.05,
            # This task's longest swing (`five_foot_gait`'s `max_air_time`): a step
            # in progress when the command stops lands before the charge is whole.
            "settle_time": 0.5,
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=FIVE_FOOT_LEGS, preserve_order=True
            ),
        },
    )

    # **`upright` is deleted, because it is now a second opinion about the axis
    # `nose_pitch` and `body_roll` already price -- and the wrong one.** It pays
    # `exp(-(gx^2 + gy^2)/0.447^2)`, maximised by standing level whatever the
    # command says, so every degree the operator asks for is a degree it charges
    # for. Measured on `2026-09-10_15-47-30/model_600`, 32 environments held at a
    # standstill, 5 s to settle and 6 s to average, per step:
    #
    #     term            level     15 deg nose-down, 10 deg roll     obeying costs
    #     feet_planted    4.038     3.053                             -0.985
    #     upright         0.999     0.602                             -0.397
    #     ground_contact -0.015    -0.285                             -0.271
    #     foot_home      -0.000    -0.059                             -0.059
    #     pose            0.043     0.000                             -0.043
    #
    # The robot obeyed -- 15.59 and 10.23 degrees against 15 and 10 -- so this is
    # the price of a command being followed, not of one being ignored.
    #
    # Only `upright` is removed. What it was for on this machine, a trunk that
    # does not tip onto its single front foot, is what `nose_pitch` at -15 and
    # `body_roll` at -12 do, against the command rather than against level.
    # `pose` stays: at 0.043 it is not what limits anything, and it is the only
    # term keeping sixteen joints near a posture. `feet_planted` and
    # `ground_contact` are the two larger opponents and are **not** deleted here:
    # the first is the five-legged standstill support this task was built around,
    # the second is the trunk scraping the floor, and neither is an opinion about
    # attitude -- they are the physical cost of holding one.
    cfg.rewards.pop("upright")

    # In replay the operator flies it, from `controls.yaml`: the right stick, R3
    # and B, laid out as jumper.posture lays out its own pose command, because the
    # two share the robot's pitch, roll and twist channels. Replay-only, exactly
    # like the walk and the claw: in training there is no viewer for a key to
    # reach, and a term drivable from outside the process makes a run
    # unreproducible. Built from the training term's own fields, so the bands and
    # the ramp the operator drives are the ones the policy trained under -- and
    # the operator refuses to run if the file describes a command with no term
    # listening, so leaving this out stops `play` rather than binding nothing.
    if play:
        from dataclasses import fields as _dc_fields

        pose = cfg.commands[POSE_COMMAND]
        cfg.commands[POSE_COMMAND] = TeleopBodyPoseCommandCfg(
            **{f.name: getattr(pose, f.name) for f in _dc_fields(pose)},
            controls=load_controls(CONTROLS),
            term=POSE_COMMAND,
        )

    # ── 10. The row, and a prop already in the claw ───────────────────────
    # Last, because the row stills every command term there is -- the attitude
    # command above included -- and it was applied after the whole config when it
    # was a scene.
    if objects:
        apply_objects(cfg)
    if hold is not None:
        _start_holding(cfg, hold)

    # The dToF, which the model carries (`common/tof.py`). Replay only, where the
    # live viewer shows it in the window's corner; nothing in training reads it,
    # and there it would be 2268 rays per environment per step spent on nothing.
    if play:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

    return cfg


def _start_holding(cfg: ManagerBasedRlEnvCfg, name: str) -> None:
    """Start the episode with `name` already in the claw -- `--hold`.

    The whole of it is one parameter on the claw-hold term, which is where holding
    is implemented (`mdp/grasp.py`); this exists so the flag fails **here**, with
    the list of props in the message, rather than inside an event term three
    hundred steps into a run.
    """
    term = cfg.events.get("claw_hold")
    if term is None:
        raise SystemExit(
            "--hold is for replay: the claw-hold term is installed only with play"
        )
    props = sorted(cfg.scene.entities or {})
    if name not in props:
        raise SystemExit(
            f"--hold {name!r}: this run has no such prop. Add --objects, or pick "
            f"from: {', '.join(p for p in props if p != 'robot') or '(none)'}"
        )
    term.params["start_holding"] = name
    print(f"[five_foot] the claw starts holding {name!r}")


__all__ = [
    "GROUND_CONTACT_SENSOR",
    "GROUND_FOOT_GEOMS",
    "MID_REAR_FEET",
    "MIN_STAND_HEIGHT",
    "NON_FOOT_GEOMS",
    "env_cfg",
]
