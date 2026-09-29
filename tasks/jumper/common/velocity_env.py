"""The jumper hexapod flat-ground velocity-tracking task.

The reward / observation / termination skeleton comes straight from mjlab's
velocity task (`make_velocity_env_cfg`), the one used for the Unitree Go1. Changes
are confined to where a hexapod genuinely differs structurally from a quadruped:

- **six legs rather than four**, and heterogeneous: the front two are 5-DoF arms
  (in FOOT mode walking on the rear jaw pads, `*F_palm_pad_b_link`) and the middle
  and rear four are 3-DoF legs
- the pose reward's std is grouped into "forearms" and "middle/rear legs", whose
  ranges of motion differ considerably
- the body is named `base_link` rather than `trunk`
- **the contact sensor's geom names must match the collision scheme** -- under
  hybrid the feet are `*_meshcol`. Get this wrong and contact detection fails
  entirely, without raising

The collision scheme is fixed to hybrid; see the module docstring in `constants`.
Flat ground only; rough terrain is later work.

## This file holds mechanism. It must not hold tuning.

A number written here is a number every jumper task inherits **without choosing it**,
and that reads as agreement between the tasks when it is only a default. It has
already happened: `jumper.tetrapod` ran on a command ladder written for a different
gait, and nothing recorded that it had never agreed to one.

So the parameters a task owns are **required arguments, and this raises without
them** rather than falling back:

    foot_target_height                    how high this gait lifts a foot
    command_lin_ceiling / _ang_ceiling    the fastest it may be commanded
    command_lin_std_ratio / _ang_...      how strictly it is marked
    action_rate_weight / pose_weight      which side of the stand-still barrier

Facts about the *robot* go the other way, into `constants.py` -- one chassis, one
IMU, one set of current sensors, so one value: `EFFORT_LIMIT`, the noise levels,
`PUSH_VELOCITY_RANGE`. The test of which pile a number belongs in is whether two
gaits on this robot could reasonably want different values.

**The four locomotion tasks are each other's controls**, and may differ only in the
gait and in the ceiling the gait implies. `tests/test_task_parity.py` fails on
anything else and names it. See `docs/DESIGN.md` section 14.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from .constants import (
    ACTUATOR_FORCE_NOISE,
    FOOT_GEOMS,
    GAIT_JOINTS,
    IMU_ANG_VEL_NOISE,
    IMU_GRAVITY_NOISE,
    JUMPER_ACTION_SCALE,
    JUMPER_FOOT_FRICTION_RANGE,
    LEGS,
    PUSH_VELOCITY_RANGE,
    get_jumper_robot_cfg,
)
from .mdp.controls import controller_contract, load_controls
from .mdp.curriculum import (
    CommandRangeCurriculum,
    TerrainLevels,
    ang_std,
    ladder,
    lin_std,
)
from .mdp.observations import actuator_force
from .mdp.operator import OperatorVelocityCommandCfg
from .mdp.rewards import body_tilt_rate, standing_sway, track_yaw_velocity

# The action scale that `action_rate_weight` is quoted at. When a task passes a
# different `action_scale`, the weight scales by (new/ref)^2 so the penalty per
# unit of joint motion stays constant -- see the detailed note in
# `velocity_env_cfg`. The weight itself is no longer here: it is the task's, and a
# reference weight beside a required argument would just be a second answer.
_ACTION_RATE_REF_SCALE = 0.25

# Contact force above which a leg-vs-leg touch counts as a collision, in newtons.
#
# mjlab uses 10 N for the G1, a ~35 kg humanoid. jumper weighs about 2 kg (29.5 N
# all up) and its limbs are 0.16 m on a 2 N-m actuator, so a leg pushing into
# another leg at full torque tops out near 2 / 0.16 = 12.5 N. Keeping G1's 10 N
# would therefore only ever fire at near-saturated torque and let every ordinary
# collision through. 1.0 N is about 8% of that ceiling: comfortably above solver
# noise and a grazing brush, well below a leg actually driving into its neighbour.
_SELF_COLLISION_FORCE_THRESHOLD = 1.0

#: Roll^2 + pitch^2 a correct gait is allowed for free, in (rad/s)^2.
#:
#: Measured on the trained tripod policy at 3.125 Hz over a full episode: 0.182
#: with the deterministic policy, 0.533 with the stochastic one training actually
#: collects. 0.5 forgives the cadence's own bob and charges only what exceeds it.
#: Tripod's number; tetrapod and ripple bob less and can pass their own.
_BODY_TILT_DEADBAND = 0.5

#: Frames of history, for the terms that get any.
#:
#: Measured, by predicting the critic's height scan from the actor's observation
#: -- the privileged information history is meant to substitute for:
#:
#:     frames   actor width   R^2     of what 5 frames get
#:     1             94       0.351          85.8%
#:     2            186       0.372          90.9%
#:     3            278       0.387          94.6%
#:     5            462       0.409         100.0%
#:
#: On that table alone two frames looks like the better trade, and it was set to
#: two for a while. It went back to five for a reason the table cannot show: the
#: width was suspected of slowing convergence, and **it was not the cause**. The
#: cause was the critic's terrain scan, 117 columns with a standard deviation of
#: 2.99e-3 while every other term measured 0.43 to 2.2, whose noise the observation
#: normaliser then scaled back up to unit variance. Cutting the history did not
#: speed anything up, which is what pointed at the scan.
#:
#: With that removed the width is not known to cost anything, and the 9 points
#: between two frames and five are a **lower bound** -- a linear fit against one
#: target, blind to whatever history offers non-linearly, or for actuator lag and
#: contact transients.
OBS_HISTORY = 5

#: Which terms get history at all. The per-joint and per-foot signals, and
#: nothing else -- from the same measurement, the contribution of each term's
#: history to that R^2:
#:
#:     actuator_force +0.0062   projected_gravity +0.0020
#:     actions        +0.0051   base_ang_vel      +0.0016
#:     joint_pos      +0.0048   base_lin_vel      +0.0013
#:     joint_vel      +0.0034   command           +0.0001
#:
#: There is a clean break after the fourth. `command` is the clearest case: it
#: resamples every 3 to 8 s, so its history is the same number copied, and it
#: measures as such. The critic's foot terms are included by kind rather than by
#: measurement -- they are the contact signal, which is the transient history is
#: for, and they were not in the actor group the fit used.
HISTORY_TERMS = frozenset({
    "joint_pos", "joint_vel", "actions", "actuator_force",
    "foot_height", "foot_air_time", "foot_contact", "foot_contact_forces",
})

#: The terrain scan's grid, scaled from mjlab's 1.6 x 1.0 m by the standing-height
#: ratio (0.385). The grid is centred on `base_link` and spans +/- size/2, so
#: 0.6 x 0.4 m at 0.05 m is 13 x 9 = 117 rays reaching 0.30 m ahead -- about 0.11 m
#: past the front feet, a third of a stance.
#:
#: **Two claims that used to sit here were wrong and are worth not restoring.** The
#: 0.05 m spacing is *coarser* than the 0.025 m foot lift, not finer; and 0.30 m
#: ahead is well under one body length, not two.
#:
#: The ratio is also arguably the wrong one. This is a horizontal footprint, and
#: `scenes/rough.py` now scales horizontal dimensions by the stance ratio (0.880)
#: rather than the height ratio, which would make this grid 1.41 x 0.88 m. It is
#: left as it is here because widening it changes the critic's input width for all
#: four locomotion tasks, which is a decision for whoever next trains on terrain.
SCAN_SIZE = (0.6, 0.4)
SCAN_RES = 0.05
SCAN_MAX_DIST = 2.0

# Joint-name groups for the front arms (5-DoF) and the middle/rear legs (3-DoF).
#
# These match on the joint **index** rather than on a part name, because the
# v1.6.1 export renamed every joint to `<leg>_J<n>_joint` and a pattern spelling
# out `shoulder_yaw|elbow|wrist` matched nothing afterwards. mjlab raises on a
# regex that matches no joint, so this failed loudly -- but the index form is
# also the one the naming convention promises to keep: `URDF_CONVENTIONS.md`
# fixes J0-J4 on the front pair and J0-J2 on the walking legs, and says the
# numbering is stable across exports.
#
# `_ARM_JOINTS` stops at J3 on purpose: J4 is the gripper, which FOOT mode holds
# at home and `_FINGER_JOINTS` drives separately.
_ARM_JOINTS = r"^[LR]F_J[0-3]_joint$"
_LEG_JOINTS = r"^[LR][MR]_J[0-2]_joint$"
_FINGER_JOINTS = r"^[LR]F_J4_joint$"


def velocity_env_cfg(
    play: bool = False,
    asset: Path | None = None,
    effort_limit: float | None = None,
    action_scale: float | None = None,
    compensate_action_rate: bool = True,
    gait_reward: Callable | None = None,
    gait_name: str | None = None,
    gait_weight: float = 1.0,
    action_rate_weight: float | None = None,
    pose_weight: float | None = None,
    self_collision_weight: float = -0.25,
    standing_sway_weight: float = -0.1,
    foot_target_height: float | None = None,
    command_lin_ceiling: float | None = None,
    command_ang_ceiling: float | None = None,
    command_lin_std_ratio: float | None = None,
    command_ang_std_ratio: float | None = None,
    controls: Path | None = None,
    #: Whether a person can drive this task. `False` for one that removes the
    #: twist command -- `jumper.swing` deletes it and every term that reads it --
    #: and then `controls` is neither required nor exported, because there is no
    #: command for a controls file to describe.
    operator_command: bool = True,
    teleop: bool = True,
    gamepad: bool = True,
    terrain_scan: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Flat-ground velocity tracking.

    Args:
            **This parameter decides both the robot's collision geometry and the
            geom names the contact sensor must match**, and the two have to agree
            or the feet will not touch the ground and contact detection will read
            nothing.
        play: replay mode -- observation noise and external disturbances off,
            longer episodes.
        asset: model XML path, from `--model`. None uses the default jumper.xml.
        effort_limit: override the flat cap sitting on top of the servo's
            torque-speed curve, **for diagnostic experiments only**. The hardware's
            plateau is 1.7464 N*m; a policy trained with any other value cannot go
            on hardware. Raising it buys nothing, since the curve bounds the torque
            below this cap at every speed.
        action_scale: override the action scale (rad). Physically it is the joint
            target offset corresponding to a 1-sigma action. Its relation to torque
            is tau = kp * scale (the worst case, before the joint has caught up), so
            scale = effort/kp = 0.175 makes 1 sigma exactly saturate the plateau.
        compensate_action_rate: whether to compensate the action_rate_l2 weight when
            action_scale changes. **On by default, and it should almost always be
            on** -- see the note at `_ACTION_RATE_REF` below.
        gait_reward: the gait reward function, passed in by the calling task from
            its own `mdp/rewards.py` (see `tasks/jumper/<task>/env_cfg.py`).
            ``None`` means no gait prior, i.e. `jumper.flat`.
            There is deliberately **no dispatch by name**: each gait is used by one
            task and the function belongs to that task; the skeleton only mounts it
            as a RewardTerm.
        gait_name: the key prefix for that reward term in `cfg.rewards`, which is
            how it is distinguished in the logs. Required whenever `gait_reward` is
            given.
        gait_weight: the gait reward's weight, 1.0 by default.
            It started at 2.0 (the same order as a single velocity tracking term),
            which measured too high: the policy would rather hold a perfect gait in
            place than walk (displacement 1/26 of the control group's). The
            temptation grows with the weight, so it was brought down to 1.0 and
            paired with `cycling_gate`'s two-ended gating.

            **0.5 measurably does not work**: tetrapod at 0.5 trained to 4000
            iterations with mean feet in contact stuck at 2.37-2.38 (2.19 for the
            base task, target 4.0) while tracking kept improving over the same span
            (vx 88%->93%, displacement 1.64->1.89 m). Tracking rising while the
            gait does not rules out "not trained enough" and shows the policy has
            found its equilibrium at that weight -- **the weight decides the
            equilibrium; training time only decides how fast it is reached**.

            The case for 2.0: velocity tracking carries a combined weight of 4.0,
            and for the gait term to bargain with it, it has to be the same order.
            It also widens the gap between a legal pairing and an illegal grouping
            from 0.31 to 1.25 (tetrapod gives an illegal grouping 0.375 in partial
            credit; see mdp/rewards.py).
        self_collision_weight: Weight of the leg-vs-leg collision penalty.
            Negative, and applied per colliding substep. Only collisions between
            two *different* legs reach this term; links on one leg cannot collide
            with each other at all. See the reward block below for the default.
        action_rate_weight: Weight of `action_rate_l2`, and `pose_weight` of the
            home-posture reward. **Both required**, because together they are the
            barrier this task family keeps falling into: measured on a policy that
            never moved, `pose` paid +0.75 for joints staying home while
            `action_rate_l2` charged -0.95 for moving at all, about 1.7 of reward
            arguing against walking. A gait's answer to that is its own. Note that
            `action_rate_weight` is also the reference the `action_scale`
            compensation scales from, so it has to be the value at
            `_ACTION_RATE_REF_SCALE`.
        command_levels: The linear command range at each curriculum rung, and
            `command_ang_levels` the angular one. **Required**, and per task: what
            a gait should be asked to do is the task's decision, and a shared
            default is how `jumper.tetrapod` came to inherit a ladder nobody chose
            for it. A task wanting the shared rungs says so by naming them --
            `jumper.flat` appends to `LEVELS` deliberately, so that the four remain
            each other's controls, and that intent is legible where it is written.
        command_lin_std_ratio: std as a fraction of the range, and
            `command_ang_std_ratio` the same for yaw. **Required**: how strictly a
            gait is marked is a choice, and holding it constant across the rungs is
            what makes the widening legitimate at all. `curriculum.STD_LIN_RATIO` /
            `STD_ANG_RATIO` are the calibrated pair to name when the answer is "the
            same as the others".
        command_lin_std_scales: Per-rung multiplier on the std the ratio produces,
            `None` meaning 1.0 throughout. A rung below 1 is a **precision rung**:
            same range as the one below it, marked more strictly. Only
            `jumper.tripod` has one.
        command_ang_std_scales: The same, for the angular axis.
        foot_target_height: How high a foot should be lifted, in metres, for both
            `foot_clearance` and `foot_swing_height`. **Required**, and per task on
            purpose: it is a property of the gait rather than of the robot, and the
            terms punish overshoot as well as shortfall, so it also sets a ceiling
            on the terrain the policy can ever climb.
        standing_sway_weight: Weight of the body-sway penalty, which is active
            **only while the command asks the robot to stand**. Negative. Set it to
            0.0 to disable. See the reward block below for how -0.1 was calibrated.
        teleop: whether `play` hands the command to the operator -- the pad and
            the keyboard the task's `controls.yaml` describes. Ignored during
            training. On by default and harmless off-screen: the term passes the
            sampled command through until a control is actually touched.
        gamepad: whether a connected pad is opened at all. Both devices are live
            and the one touched last drives; a task that does not want pad
            control sets this False and keeps the keys.
    """
    from .constants import EFFORT_LIMIT

    eff = EFFORT_LIMIT if effort_limit is None else effort_limit
    cfg = make_velocity_env_cfg()

    # A hexapod with all feet down has far more contacts than a quadruped. njmax
    # is the **per-world** constraint row limit, its default inferred from a
    # resting mjData, and it overflows once the robot moves (measured at around
    # 500 needed).
    cfg.sim.njmax = 512
    cfg.sim.nconmax = 128
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 128

    cfg.scene.entities = {"robot": get_jumper_robot_cfg(effort_limit=eff, asset=asset)}

    # ── Flat ground: drop the terrain generator and the terrain ray casts ──
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    # Where robots start when a scene *does* put generated terrain underneath.
    # mjlab's 5 scatters them across the first six difficulty rows, which assumes
    # a policy that already walks on rough ground. Measured what it costs when
    # that assumption is wrong: a flat-trained policy resumed onto `--scene rough`
    # at a mean level of 2.47 fell within **15 control steps**, 0.3 s, on its
    # first episode. From there the terrain curriculum demoted it to 0.44, the
    # command curriculum fell from its top level back to level 1, and yaw
    # tracking -- 0.766 of 2.0 after the flat stage -- collapsed to 0.004 and did
    # not recover. Two curricula unwound together because the first thing the
    # policy met was terrain it had no chance on.
    #
    # 0 starts everyone on the flattest row and lets the terrain curriculum earn
    # the rest, which is what a curriculum is for.
    cfg.scene.terrain.max_init_terrain_level = 0

    foot_geoms = FOOT_GEOMS

    # The foot height scan attaches to the six foot sites (generated by
    # build_jumper.py and named by leg prefix)
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=leg, entity="robot") for leg in LEGS
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.02, num_samples=4)
        elif sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "base_link"
            # mjlab scans 1.6 x 1.0 m at 0.1 m for a robot 0.278 m tall. This one
            # is 0.3 m across and lifts its feet 0.025 m, so that grid looks five
            # body-widths ahead at a resolution coarser than the obstacles it is
            # meant to reveal. Scaled by the standing-height ratio, then rounded to
            # a resolution that divides the size exactly -- see `SCAN_SIZE`, which
            # records what that scaling does and does not buy.
            sensor.pattern = GridPatternCfg(size=SCAN_SIZE, resolution=SCAN_RES)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=foot_geoms, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    # Leg-vs-leg contacts. Both sides are the whole-robot subtree, which is how
    # mjlab wires the G1's self-collision sensor; terrain is outside the subtree
    # so ground contacts do not leak in. The filtering that makes this mean
    # "leg vs a *different* leg" lives in the collision masks, not here: geoms on
    # one leg share a bit and therefore never generate a contact at all, so a
    # gripper resting on its own palm is invisible to this sensor. See the mask
    # derivation in `constants`.
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        # One buffer entry per physics substep of a policy step, so a collision
        # that starts and resolves between two policy steps is still counted.
        history_length=cfg.decimation,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    # ── The terrain scan: off by default, critic-only when on ─────────────
    # **The actor never gets it, on purpose.** The robot has no way to measure
    # the ground ahead, so an actor that depends on it is an actor that cannot be
    # deployed. The critic is a different matter: it exists only during training,
    # it is thrown away at export, and giving the value function something the
    # policy cannot see is the standard asymmetric arrangement -- the same one
    # `foot_height`, `foot_air_time`, `foot_contact` and `foot_contact_forces`
    # already use here.
    #
    # Off by default because the skeleton sets flat ground, where every ray
    # returns the same number. A constant input is not free: it occupies width in
    # the critic and contributes nothing, the same objection as feeding in the two
    # gripper actuators. A task that wants rough terrain turns it on.
    del cfg.observations["actor"].terms["height_scan"]
    if terrain_scan:
        cfg.observations["critic"].terms["height_scan"].scale = 1.0 / SCAN_MAX_DIST
    else:
        cfg.scene.sensors = tuple(
            s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
        )
        del cfg.observations["critic"].terms["height_scan"]

    # ── Base linear velocity: critic only, for the same reason ────────────
    # The robot cannot measure it. An IMU gives angular velocity directly from
    # the gyro, and `base_ang_vel` therefore stays in the actor -- but linear
    # velocity only comes out of a state estimator integrating that IMU against
    # foot contacts, and this robot has none. An actor trained on the simulator's
    # exact `base_lin_vel` is an actor with no counterpart on hardware.
    #
    # This is the same asymmetric arrangement as `height_scan` above, and it costs
    # the deployed policy nothing: the critic is thrown away at export.
    #
    # It reads as a small change and is not -- the actor loses three inputs, so
    # **every checkpoint trained before it is unusable** and the tasks have to be
    # retrained. Kept anyway, because the alternative is a policy that trains and
    # replays beautifully and cannot be put on the robot at all. The three
    # deployed layouts on the hexapod controller agree: `base_ang_vel` present,
    # `base_lin_vel` absent, in all of them.
    #
    # `scripts/export.py::_validate_measurable` refuses to export an actor that
    # carries it, so this cannot quietly come back.
    del cfg.observations["actor"].terms["base_lin_vel"]

    # ── Actions and observations: exclude the two grippers ────────────────
    # In FOOT mode the forearms walk on the fixed jaws and the grippers stay closed
    # (0.0 in HOME). They belong in neither the action space nor the observation
    # dimension -- the actuator PD holds them at 0.
    #
    # The field that restricts joints is `actuator_names`, **not `asset_cfg`**.
    # BaseActionCfg is a plain dataclass and a wrong field name is silently
    # accepted: written as asset_cfg, the action dimension stayed at 22 instead of
    # 20 with no sign of anything wrong.
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.actuator_names = list(GAIT_JOINTS)
    joint_pos_action.scale = (
        JUMPER_ACTION_SCALE
        if action_scale is None
        else {j: action_scale for j in GAIT_JOINTS}
    )


    # joint_pos / joint_vel in the observations likewise take only these 20
    # joints. Both the actor and critic groups have to change.
    _gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))
    for _group in cfg.observations.values():
        for _name in ("joint_pos", "joint_vel"):
            _term = _group.terms.get(_name)
            if _term is not None:
                _term.params = dict(_term.params or {}) | {"asset_cfg": _gait_cfg}

    # ── What the IMU is actually like ─────────────────────────────────────
    # mjlab's velocity skeleton sets these (+/-0.2 on the rate, +/-0.05 on the
    # gravity direction) and they are inherited, not chosen. Replaced with this
    # robot's, which live in `constants.py` with their provenance: they describe
    # the IMU rather than the gait, so one set serves all four tasks and a task
    # that differs has to say so rather than drift.
    for _group in cfg.observations.values():
        if "base_ang_vel" in _group.terms:
            _group.terms["base_ang_vel"].noise = Unoise(
                n_min=-IMU_ANG_VEL_NOISE, n_max=IMU_ANG_VEL_NOISE
            )
        if "projected_gravity" in _group.terms:
            _group.terms["projected_gravity"].noise = Unoise(
                n_min=-IMU_GRAVITY_NOISE, n_max=IMU_GRAVITY_NOISE
            )

    # ── Actuator force, and a window of the recent past ───────────────────
    # Torque is what a 1.75 N*m actuator on a 2 kg robot spends most of a stride
    # near the limit of, and mjlab has no term for it. Restricted to the 20 driven
    # actuators: the two grippers are held closed by their PD and would contribute
    # constant columns.
    _force_cfg = SceneEntityCfg("robot", actuator_names=list(GAIT_JOINTS))
    for _group in cfg.observations.values():
        _group.terms["actuator_force"] = ObservationTermCfg(
            func=actuator_force,
            params={"asset_cfg": _force_cfg},
            # This robot's current sensor, not this gait's -- so the number
            # and the experiment behind it are in `constants.py`, beside the two
            # IMU figures it belongs with.
            noise=Unoise(
                n_min=-ACTUATOR_FORCE_NOISE, n_max=ACTUATOR_FORCE_NOISE
            ),
        )

    # `history_length` on the **group** overwrites every term's own value during
    # `_prepare_terms`, which is why it is set here rather than per term -- and why
    # anything reading the layout back has to read it from the manager rather than
    # from this config. `mdp/symmetry.py` does exactly that.
    #
    # One frame is 0.02 s, so five spans 0.1 s: a third of a 3.125 Hz gait cycle,
    # enough to carry the direction a contact is developing rather than only its
    # present value. The cost is the width -- every term is five times itself, so
    # the actor's input goes from 74 to 470.
    # Per term rather than on the group, because a group-level `history_length`
    # **overwrites every term's own value** in `_prepare_terms` -- all or nothing.
    # `HISTORY_TERMS` decides who gets any. The height scan is the clearest of the
    # exclusions -- a slow-varying picture of the ground, whose extra copies would
    # be 117 rays becoming 234 for almost no new information.
    # Note the ordering: this runs before a task adds terms of its own, so a term
    # mounted after `velocity_env_cfg` returns keeps whatever `history_length` it
    # was built with. That is a trap worth knowing about rather than working
    # around -- see `tasks/jumper/tripod/env_cfg.py`, which sets it explicitly.
    for _group in cfg.observations.values():
        for _name, _term in _group.terms.items():
            _term.history_length = OBS_HISTORY if _name in HISTORY_TERMS else 0

    # ── Per-robot reward parameters ───────────────────────────────────────
    # Tighter when standing, looser when walking. The forearms have a larger range
    # of motion than the middle and rear legs, so their std is larger.
    cfg.rewards["pose"].params["std_standing"] = {_ARM_JOINTS: 0.10, _LEG_JOINTS: 0.05}
    cfg.rewards["pose"].params["std_walking"] = {_ARM_JOINTS: 0.60, _LEG_JOINTS: 0.30}
    cfg.rewards["pose"].params["std_running"] = {_ARM_JOINTS: 0.60, _LEG_JOINTS: 0.30}
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(_ARM_JOINTS, _LEG_JOINTS)
    )

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
    cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = LEGS

    # How high a foot should be lifted. **Required from the task**, because it is a
    # number about a gait rather than about the robot: how high a foot goes is a
    # choice each gait makes against its own cadence and stride, and the four tasks
    # agreeing on it today is a coincidence rather than a shared fact. Go1's 0.1 m
    # would raise a foot to body height on a 0.107 m robot, so mjlab's default is
    # not a fallback either.
    #
    # Both terms punish **overshoot** as well: `feet_swing_height` costs
    # `(peak / target - 1)^2` and `feet_clearance` costs `|h - target| * speed`, so
    # this is a target and not a floor -- a foot lifted twice the target pays what
    # one lifted half of it pays.
    if action_rate_weight is None or pose_weight is None:
        raise ValueError(
            "action_rate_weight and pose_weight are required: together they are "
            "the 'stand still' barrier this robot keeps falling into, worth about "
            "1.7 of reward, and which side of it a gait sits on is that gait's "
            "problem rather than a shared default"
        )
    if command_lin_std_ratio is None or command_ang_std_ratio is None:
        raise ValueError(
            "command_lin_std_ratio and command_ang_std_ratio are required: how "
            "strictly a gait is marked is the task's decision, and "
            "curriculum.STD_LIN_RATIO / STD_ANG_RATIO are the values to name if "
            "the answer is 'the same as everyone else'"
        )
    if command_lin_ceiling is None or command_ang_ceiling is None:
        raise ValueError(
            "command_lin_ceiling and command_ang_ceiling are required: how fast a "
            "gait can be asked to go is the one thing the gait genuinely fixes, and "
            "a shared default is how a task ends up on a ceiling nobody chose for it"
        )
    # **The shape is shared, the ceiling is not.** Every gait climbs
    # `LEVEL_FRACTIONS` of its own ceiling and then meets one precision rung, so
    # level 1 means the same thing in every task. They used to be built by clamping
    # or appending to the shared *absolute* rungs, which gave three different
    # curricula under one name -- see `curriculum.ladder`.
    (
        command_levels,
        command_ang_levels,
        command_lin_std_scales,
        command_ang_std_scales,
    ) = ladder(command_lin_ceiling, command_ang_ceiling)

    if operator_command and controls is None:
        raise ValueError(
            "controls is required: it is the task's `controls.yaml`, naming which "
            "way each command axis counts as positive and what a stick at full "
            "deflection means. A shared default would be four tasks inheriting a "
            "decision none of them made, and a sign that is wrong by inheritance "
            "looks exactly like one that is right.\n"
            "A task that keeps no twist command for anyone to drive says "
            "`operator_command=False` instead -- see `jumper.swing`."
        )
    parsed_controls = load_controls(controls) if operator_command else None

    if foot_target_height is None:
        raise ValueError(
            "foot_target_height is required: the foot-lift target belongs to the "
            "gait, and a shared default would make four tasks look like they agree "
            "on purpose"
        )
    cfg.rewards["foot_clearance"].params["target_height"] = foot_target_height
    cfg.rewards["foot_swing_height"].params["target_height"] = foot_target_height

    # ── Dismantling the "stand still" barrier ─────────────────────────────
    # The evidence is the reward breakdown from the final iteration of a 2 N*m run
    # whose policy genuinely never moved, making that table a direct measurement of
    # what standing still collects:
    #   upright  0.9999  standing is the global optimum and entirely free
    #   pose     0.7476  **actively rewards joints staying at home** -- walking
    #                    necessarily departs from it, so this is a reverse incentive
    #   action_rate_l2  -0.9468  **penalises moving at all** -- and walking is by
    #                    definition periodically changing the action
    # Positive terms total 3.1475 of a theoretical 6.0: standing still takes 52%.
    # pose (+0.75) and action_rate (-0.95) together form a barrier of about 1.7.
    #
    # These two were once weakened to action_rate -0.02 / pose 0.5 to dismantle
    # that barrier. But that judgement was made **before the collision-box bug was
    # fixed** (the robot was sitting on a body box sunk into the ground and could
    # not possibly walk), so the conclusion did not hold. After the physics fix they
    # were rolled back to mjlab's values, to establish whether weakening them is
    # necessary at all. Walking after the rollback would mean these two are
    # unrelated to the problem and this departure from upstream should not be
    # kept.
    cfg.rewards["action_rate_l2"].weight = action_rate_weight
    cfg.rewards["pose"].weight = pose_weight

    # ── The hidden coupling between action_scale and action_rate_l2 ───────
    # mjlab's action_rate_l2 penalises the rate of change of the **raw policy
    # output** (its docstring says so explicitly: "Operates on raw policy output
    # (before per-term scale/offset)"), not of the joint angles.
    #
    # So producing the same joint motion dq needs a raw action change of
    # da = dq / scale, and the penalty goes as da^2, i.e. 1/scale^2. **Shrink scale
    # by k and the penalty for the same physical motion grows by k^2.**
    #
    #   scale 0.25 -> 0.20 : penalty x1.6
    #   scale 0.25 -> 0.05 : penalty x25
    #
    # That explains why a small action_scale cannot learn to walk: it is not only
    # that the action range is too small, but that the cost of moving at all has
    # been amplified by one to two orders of magnitude. Changing action_scale
    # without compensating the weight therefore changes two variables at once and
    # invalidates any comparison.
    if action_scale is not None and compensate_action_rate:
        ratio = (action_scale / _ACTION_RATE_REF_SCALE) ** 2
        cfg.rewards["action_rate_l2"].weight = action_rate_weight * ratio

    # Fix air_time's threshold first: its command_threshold is 0.5 while the mean
    # command is only 0.22, so even with a non-zero weight it would almost never
    # trigger (the other foot terms all use 0.05). **The weight stays at 0 here**;
    # this only puts the parameter in place.
    cfg.rewards["air_time"].params["command_threshold"] = 0.05

    # These three match mjlab's Go1 config, carried over as-is and **not evaluated
    # separately for a hexapod**. For reference: Go1 has all three at 0; G1 (the
    # humanoid) sets body_ang_vel to -0.05 and angular_momentum to -0.02, with
    # air_time also 0.
    #
    # What each does, to guide later tuning:
    #   air_time         -- a positive term, and the **only one in this set that
    #                       pays out solely while a foot is airborne**. Go1 and G1
    #                       walk with it off, so it is not what separates a walking
    #                       quadruped from a non-walking hexapod; but a hexapod is
    #                       not a quadruped, and to fight the "stand still" local
    #                       optimum it is the only candidate here.
    #   body_ang_vel     -- a penalty suppressing torso sway; enabled for the
    #                       humanoid only.
    #   angular_momentum -- a penalty, likewise.
    #                       Turning these two on only **increases the cost of
    #                       moving**, making standing still more attractive still.
    #                       For the current exploration problem that is the wrong
    #                       direction; do not reach for them as a rescue.
    cfg.rewards["air_time"].weight = 0.0
    cfg.rewards["body_ang_vel"].weight = 0.0
    cfg.rewards["angular_momentum"].weight = 0.0

    # ── Body sway, but only while the command says stand ──────────────────
    # The two terms just above stay at 0 because they penalise torso motion
    # *always*, which raises the cost of walking. This one is gated on a near-zero
    # command, so it is exactly zero for a walking policy and cannot tilt that
    # trade-off; it only separates standing well from standing badly -- which
    # nothing else in this task does, since both tracking rewards hand out
    # exp(0) = 1 to a zero command however the body is behaving (the "floor that
    # cannot be lowered" in the angular-std note below, covering ~28% of commands).
    #
    # **At -0.1 this term is a monitor, not an incentive**, and that is deliberate.
    # The measured sway of a trained policy standing still is 0.005, so the penalty
    # is 0.0005 per step -- 0.03% of the 2.0 a standing environment collects for
    # free. Even a body rocking hard, |w| ~ 0.5 rad/s, pays only 0.025 per step,
    # about 1.2%. Nothing at this weight will reshape behaviour; what it does is put
    # the quantity in the reward breakdown where it can be watched.
    #
    # That is the right size **because the sway had a cause, and the cause is fixed
    # elsewhere**. The residual rocking was the phase clock: it used to keep ticking
    # in the observation while the robot was commanded to stand, and the roll rate's
    # autocorrelation peaked at 2.50 Hz, the clock's own frequency, with only 5.17
    # of 6 feet on the ground. `tasks/jumper/tripod/mdp/phase.py` now zeroes the clock
    # while standing. Pushing on a penalty instead would have been fighting an input
    # that should not have been there.
    #
    # Raising it is not free, and the ceiling is set by a transient rather than by
    # the sway: the gate is a hard threshold, so an environment whose command flips
    # from walking to standing is charged for its deceleration, and that transient
    # measures **17x the steady state** (0.081 against 0.005). Anything large enough
    # to bite on steady-state sway therefore lands 17x harder on a robot doing the
    # physically unavoidable. To go much beyond ~-1 the gate needs to become "the
    # command has been near zero for N steps" -- measured, settling takes about 50
    # steps -- so that the stopping transient is excluded rather than punished.
    cfg.rewards["standing_sway"] = RewardTermCfg(
        func=standing_sway,
        weight=standing_sway_weight,
        params={"command_name": "twist", "command_threshold": 0.05},
    )

    # ── Leg-vs-leg collision penalty ────────────────────────────────────
    # `self_collision_cost` returns the number of substeps (0..decimation, so
    # 0..4 here) in which some leg pair pushed harder than the force threshold,
    # so the term costs at most `decimation * weight` per policy step.
    #
    # The weight is deliberately modest. Standing still at HOME has zero
    # self-collisions, so every newton of collision penalty also deepens the
    # "stand still and collect the free tracking score" local optimum this task
    # has repeatedly fallen into (see the long notes on the tracking stds above).
    # At -0.25 the worst case is -1.0 per step against 4.0 of tracking reward:
    # enough to make legs swinging through each other a bad deal, not enough to
    # make not moving the safe answer. Raise it via `self_collision_weight` if
    # trained policies still scrape their legs together.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=self_collision_weight,
        params={
            "sensor_name": self_collision_cfg.name,
            "force_threshold": _SELF_COLLISION_FORCE_THRESHOLD,
        },
    )

    # ── Terminations ──────────────────────────────────────────────────────
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations.pop("illegal_contact", None)
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(50.0)},
    )
    # Replaced rather than dropped: `mdp/curriculum.py`'s wrapper is inert on a
    # plane and live once `--scene rough` puts generated terrain underneath, which
    # is a decision that cannot be made here because the scene is applied after
    # this function returns.
    # The key is `terrain`, not mjlab's `terrain_levels`, so the two cannot be
    # confused in the log -- and the pop is what makes the replacement a
    # replacement. Drop the pop and both terms run: mjlab's distance rule would
    # promote (it is met by any hexapod, see `TerrainLevels`) while this one
    # demotes, on the same envs, in an order nothing defines.
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum["terrain"] = CurriculumTermCfg(
        func=TerrainLevels, params={"command_name": "twist"}
    )

    # ── Domain randomisation ──────────────────────────────────────────────
    if "foot_friction" in cfg.events:
        cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
            "robot", geom_names=foot_geoms
        )
        # The term's `operation` is "abs", so this **replaces** `_FOOT_FRICTION`
        # rather than scaling it: whatever the asset declares, training only ever
        # sees a sample from this range. Leaving mjlab's (0.3, 1.2) here meant the
        # nominal value was dead config -- and that the band sat entirely at or
        # below what Shore A 60 silicone gives on a clean dry floor, so every
        # environment was more slippery than anything the robot walks on.
        cfg.events["foot_friction"].params["ranges"] = JUMPER_FOOT_FRICTION_RANGE
    if "base_com" in cfg.events:
        cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

    # ── Command ranges ──
    # mjlab's defaults are tuned for the Unitree Go1 (0.278 m standing height,
    # +/-1.0 m/s linear velocity). The hexapod stands only 0.107 m tall, so
    # inheriting them puts commands beyond what it can reach -- measured,
    # error_vel_xy sat around 1.6 for long stretches (the command magnitude only
    # reaches 1.41), the tracking reward stayed near the floor and the learning
    # signal was weak. These bring it within this robot's actual capability.
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    # **Taken from the curriculum's last level, not written out again.** The two
    # have to agree, and when they were two independent literals they silently
    # stopped agreeing the moment the angular levels were extended to 0.75: the
    # curriculum trained the policy up to +/-0.75 of yaw while this file still
    # said 0.5. Nothing failed. What it changed was `play`, which builds with
    # `curriculum = {}` and so uses exactly these numbers -- so a pad could only
    # ask for two thirds of what the policy had been trained to do, and the
    # replay quietly under-reported the robot.
    #
    # Deriving it also restores what `curriculum.py` claims about itself: that
    # finishing the curriculum lands on the configuration this file argues for.
    _lin, _ang = command_levels[-1], command_ang_levels[-1]
    twist.ranges.lin_vel_x = (-_lin, _lin)
    twist.ranges.lin_vel_y = (-_lin, _lin)
    twist.ranges.ang_vel_z = (-_ang, _ang)

    # ── Forward-only envs carry a hardcoded 0.3 m/s floor; turn them off ──
    # `UniformVelocityCommand._resample_command` gives the `rel_forward_envs`
    # fraction a positive-x, zero-lateral command -- and floors it:
    #
    #     self.vel_command_b[fwd_ids, 0] = ...abs().clamp(min=0.3)
    #
    # **0.3 is a literal, unrelated to `ranges.lin_vel_x`.** It is sized as a
    # fraction of Go1's +/-1.0, and nothing -- no range setting, no curriculum level
    # -- can lower it. That is the problem, and it is about consistency rather than
    # reachability: 0.3 m/s is comfortably within this robot's ability (a trained
    # tripod policy commanded 0.3 achieves 0.257), but it is **twice the first
    # curriculum level's whole range**. Left on, 20% of environments would ignore
    # the level they are supposed to be training at, which defeats the point of
    # starting narrow (measured: with the curriculum at +/-0.15, sampled |vx| still
    # reached exactly 0.30).
    #
    # What is lost is upstream's coverage of straight-line walking, which its
    # comment motivates by stair climbing. These tasks are flat-ground only, and
    # `rel_heading_envs` (0.3) still produces sustained directed motion. If forward
    # coverage is wanted back, it needs a command term whose floor scales with the
    # range rather than this flag.
    twist.rel_forward_envs = 0.0

    # ── And drop the curriculum that would undo the three lines above ──
    # `mdp.commands_vel` does not *extend* the ranges, it **overwrites
    # `cfg.ranges` in place**, and its first stage is `step: 0` -- so it applies
    # before the first iteration and the assignments above never reach training:
    #
    #     step 0         lin_vel_x (-1.0, 1.0)   ang_vel_z (-0.5, 0.5)
    #     step 5000x24   lin_vel_x (-1.5, 2.0)   ang_vel_z (-0.7, 0.7)
    #     step 10000x24  lin_vel_x (-2.0, 3.0)
    #
    # Measured in a training log rather than inferred:
    # `Curriculum/command_vel/lin_vel_x_max: 1.0000` on the very first iteration.
    # lin_vel_y kept its -0.5/0.5 only because no stage mentions it, and ang_vel_z
    # agreed by coincidence -- so the visible symptom was confined to x, which is
    # exactly the axis the argument above is about.
    #
    # Those stage values are Go1's, and the reasoning above applies to them
    # unchanged: this robot stands 0.107 m against Go1's 0.278 m. A trained tripod
    # policy tracks about 85% of the commanded speed and reaches 0.427 m/s at a
    # command of 0.5, so the final stage's (-2.0, 3.0) would be asking for roughly
    # seven times what it can do. A command it cannot reach contributes no usable
    # gradient -- it just holds `error_vel_xy` high, which is the symptom the note
    # above was written about in the first place.
    #
    # This is dropped for **all four jumper tasks**, so they remain each other's
    # controls, and replaced below by one written against this robot.
    cfg.curriculum.pop("command_vel", None)

    # ── The replacement: a performance-gated range curriculum ──
    # `mdp/curriculum.py` carries the derivation. The two properties mjlab's
    # version lacks: the level advances only when the policy actually tracks at
    # the current level, and `std` scales with the range so that widening it does
    # not also loosen the ruler (measured: with std held fixed, narrowing the
    # range from 0.5 to 0.15 lets a robot that never moves collect 1.35 -> 3.11 of
    # the 4.0 the tracking terms are worth; with std scaled it stays at ~1.28).
    #
    # It starts at +/-0.25 and ends at +/-0.50 -- the value argued for above, with
    # std back at 0.25/0.20. So the curriculum's endpoint *is* the calibrated
    # configuration; what it changes is only how the policy gets there.
    #
    # The start was +/-0.15 until a run sat on that rung for all 37500 of its
    # iterations. The bar there is 0.045 and 0.03 of it is PPO's exploration
    # noise, so the rung was measuring the entropy coefficient rather than the
    # policy. `mdp/curriculum.py` carries the per-level numbers.
    #
    # An older note here read "a run cleared all three promotions by iteration 482
    # ... which says the levels are on the easy side rather than the hard side".
    # That run was on flat ground, and the reading did not survive contact with
    # `--scene rough`: the same `gate_ratio` of 0.3 then made the first rung
    # unclearable. Keep both facts in mind before tightening it -- and note that
    # the numbers behind the current bars come from a policy trained while
    # `track_angular_velocity` returned zero, so they are due a re-measurement
    # rather than a tightening.
    cfg.curriculum["command"] = CurriculumTermCfg(
        func=CommandRangeCurriculum,
        params={
            "command_name": "twist",
            "levels": command_levels,
            "ang_levels": command_ang_levels,
            "lin_std_scales": command_lin_std_scales,
            "ang_std_scales": command_ang_std_scales,
            "lin_std_ratio": command_lin_std_ratio,
            "ang_std_ratio": command_ang_std_ratio,
        },
    )

    # With the command ranges changed, the tracking reward's std has to tighten in
    # step or the reward has merely become easier. The formula is
    # r = exp(-err^2 / std^2), where std decides how much error still counts as
    # tracking. Halving the range while leaving std alone doubles the relative
    # tolerance -- and that is exactly what was measured: error_vel_xy fell by only
    # 21% (1.640 -> 1.296) while the track_linear_velocity reward rose 35% (0.611 ->
    # 0.826). The extra was not a better policy but a looser ruler.
    #
    # The case for 0.25: make tracking as hard here as it is for Go1. Go1 has
    # std=0.5 with typical commands around 0.7, so standing still scores
    # exp(-0.49/0.25) ~= 0.14. The hexapod's typical command is about 0.35, and
    # getting the same 0.14 needs std^2 = 0.0625, i.e. std = 0.25 -- which is also
    # exactly the proportional "halve the range, halve the std".
    #
    # The risk: std must not go below the precision the robot can achieve, or the
    # reward sits at 0 and the gradient vanishes. If training stalls, back off to
    # 0.35 (standing still scores 0.13) and look again.
    # Taken from the curriculum's own final level, for the same reason the range
    # is: a std that does not move with its range is a ruler that changes length,
    # which `mdp/curriculum.py` measures the cost of. On the shared ladder this is
    # 0.25, the value the paragraph above argues for.
    #
    # Through `lin_std` rather than `STD_LIN_RATIO * _lin` so that a task appending
    # its own rungs -- `jumper.flat` a wider one, `jumper.tripod` a stricter one -- has
    # one definition to override and not an open-coded formula to keep in step.
    cfg.rewards["track_linear_velocity"].params["std"] = lin_std(
        -1, command_levels, command_lin_std_scales, command_lin_std_ratio
    )

    # ── Yaw tracking scores yaw, and not the torso's roll and pitch rate ──
    # mjlab's `track_angular_velocity` scores
    # `exp(-(yaw_err^2 + roll^2 + pitch^2) / std^2)`. Everything below this line
    # about choosing the angular std was reasoned as if the exponent were the yaw
    # error alone, and on a Go1 that is nearly true. On this hexapod at 3.125 Hz
    # the two torso terms measure 0.518 against the yaw term's 0.237 -- **2.2x the
    # quantity the reward is named after** -- and the term is then identically zero
    # at every std the curriculum ever sets. Logged: 0.0017 against
    # track_linear_velocity's 1.11, for 37500 iterations, at weight 2.0.
    #
    # So none of the calibration below was ever in effect. The tables are kept
    # because the argument they make is the right one and now applies to a term
    # that can actually pay out -- but they were written against a function that
    # returned zero regardless, and the "measured" free scores in them are free
    # scores of the *linear* term and of a yaw-only reward computed offline, not of
    # what training was running. Treat the numbers as the design intent, and
    # re-measure once a policy has trained with this term alive.
    #
    # The derivation is in `mdp/rewards.py::track_yaw_velocity`.
    cfg.rewards["track_angular_velocity"].func = track_yaw_velocity

    # ── And the torso rates get their own term, rather than none ──────────
    # Taking roll and pitch out of the tracking exponent above left them scored by
    # nothing. This puts them back as a separate penalty with a **deadband**, which
    # is what stops it being a tax on walking: a correct gait has to bob, so
    # charging for the bob would make standing still -- this task's documented local
    # optimum -- the better deal. Above the deadband a correct gait pays nothing and
    # so does standing; only excess sway is charged. See the measurement in
    # `mdp/rewards.py::body_tilt_rate`.
    #
    # **The weight is deliberately small and is not measured.** Nothing has trained
    # with this term yet. -0.05 against tracking's 4.0 is enough to break a tie
    # between two gaits that track equally well and not enough to change which
    # behaviour is worth learning; if trained policies still wallow, raise it, and
    # watch `Episode_Reward/upright` (0.9868, about 3 degrees, before this existed)
    # and the standing-still termination rate for the failure it could cause.
    cfg.rewards["body_tilt_rate"] = RewardTermCfg(
        func=body_tilt_rate,
        weight=-0.05,
        params={"deadband": _BODY_TILT_DEADBAND},
    )

    # The angular std has to tighten too, and it is the largest block of free score.
    #
    # Evidence: with the old stds (linear 0.5, angular 0.707), 3000 iterations
    # taught the policy to **stand still** -- the torque audit showed vx/vy/wz
    # tracking rates all at 0% and mean torque at 2% of the limit. Per step,
    # standing still collected:
    #     track_linear  0.368 x 2 = 0.74
    #     track_angular 0.607 x 2 = 1.21   <- the largest block
    #     upright + pose        ~= 2.00
    #     total ~= 3.95, against a perfect tracking score of only 4.0 -- and walking
    #     additionally pays action_rate / foot_slip / foot_clearance penalties.
    # So not moving collected nearly full marks at zero risk, and 500 iterations
    # locked into that local optimum.
    #
    # Note that mjlab's Go1 and G1 both use 0.707 and do learn gaits, so this is not
    # an upstream bug. The difference is probably that a hexapod is **statically
    # stable**: standing costs nothing, which makes not moving especially
    # attractive, while a quadruped pays to stand at all.
    #
    # The arithmetic above is **wrong**: 0.607, and the 0.13 below, were computed at
    # the endpoint of |command|, whereas commands are uniform over the interval and
    # a typical value is half the endpoint. Recomputed over the actual sampling
    # distribution:
    #
    #     std=0.707 -> standing still collects 0.86    (believed to be 0.607)
    #     std=0.35  -> standing still collects 0.676   (believed to be 0.13, out by 5x)
    #
    # The consequence is that tightening to 0.35 did not solve anything. Measured:
    # the soft reward at weight 2.0, trained to 1577 iterations, reached
    # track_angular_velocity = 1.3390 while the free score for never turning at all
    # is 1.3512 -- **on that term the trained policy is indistinguishable from a
    # robot that never turns**.
    #
    # And lowering std helps the overall mean only so much: 28.3% of commands are
    # near zero, where exp(0)=1 regardless of std, forming a floor that cannot be
    # lowered (even std=0.15 only takes 0.676 to 0.416).
    #
    # **What matters is not the overall mean but the incentive gap when turning is
    # required** (the mean is diluted by zero commands). Measured by command
    # magnitude, what not turning collects:
    #
    #     bucket                share    std=0.35   std=0.25   std=0.20   std=0.15
    #     ~0 (should not turn)  28.3%      0.998      0.996      0.994      0.990
    #     small                 23.9%      0.871      0.769      0.672      0.515
    #     medium                23.9%      0.540      0.309      0.169      0.051
    #     large                 23.9%      0.235      0.063      0.015      0.001
    #
    # At medium commands, not turning currently takes 54% and the incentive gap is
    # only 0.46 -- turning costs action_rate and foot_slip, so of course the policy
    # does not turn.
    #
    # The case for 0.20 is **parity with linear velocity**, not a guess: for
    # |command|>0.2 (48% of commands),
    #     std=0.35 -> gap 0.612
    #     std=0.20 -> gap 0.908   <- linear velocity is 0.90 on the same basis
    # The zero bucket still scores 0.994, so "do not turn when you should not" still
    # pays.
    #
    # Reproduce with: tools/checks/track_free_score.py
    # Likewise taken from the final level. At the shared ladder's angular top of
    # 0.75 this is 0.30 rather than the 0.20 the note above computes -- and it is
    # the *same* ruler: 0.20 was right for a range of 0.5, and the incentive gap
    # that argument is built on is preserved by scaling, not by holding the number
    # still.
    cfg.rewards["track_angular_velocity"].params["std"] = ang_std(
        -1, command_ang_levels, command_ang_std_scales, command_ang_std_ratio
    )

    # ── External disturbances ──
    # Scaled to this robot's mass, which is why the numbers are in `constants.py`
    # with the rest of what the hardware is rather than here.
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"].update(PUSH_VELOCITY_RANGE)


    # ── The gait reward (optional) ────────────────────────────────────────
    # None of these depends on body velocity, so they provide gradient before the
    # body has moved -- a hexapod is statically stable and with all six feet down
    # the body is over-constrained, leaving velocity rewards with almost no gradient
    # near that state. They are a hexapod-specific prior with no counterpart in
    # mjlab's quadruped or humanoid tasks.
    if gait_reward is not None:
        if not gait_name:
            raise ValueError(
                "gait_name is required whenever gait_reward is given; without it "
                "the reward term has no key"
            )
        cfg.rewards[f"{gait_name}_gait"] = RewardTermCfg(
            func=gait_reward,
            weight=gait_weight,
            params={
                "sensor_name": feet_ground_cfg.name,
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        )

    # ── Viewer ────────────────────────────────────────────────────────────
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 0.9
    cfg.viewer.elevation = -15.0

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        # Replay keeps the training command ranges, so nothing outside what was
        # trained on is ever commanded

        # ── Hand the command to the operator ──
        # Replay-only, and never during training: there is no operator there, and
        # a command term that could be overridden from outside would make a run
        # unreproducible.
        #
        # The operator's term is a drop-in subclass, so it inherits every field
        # of the sampling command -- including the ranges, which it scales the
        # sticks to. Until a control is touched it *is* the sampling command,
        # which is what keeps `play.py --headless` and scripted audits behaving
        # exactly as before; see `mdp/operator.py`.
        #
        # Pad and keyboard are both live, and the one touched last drives. Each
        # binds its own inputs to what they do, in the same file, and the file
        # is checked for each reaching every axis or saying why not. A task
        # with a second command -- the
        # posture of `jumper.posture` -- installs that term itself and shares
        # this term's operator.
        if teleop:
            from dataclasses import fields as _dc_fields

            twist_fields = {f.name: getattr(twist, f.name) for f in _dc_fields(twist)}
            cfg.commands["twist"] = OperatorVelocityCommandCfg(
                **twist_fields, controls=parsed_controls, term="twist", pad=gamepad
            )

    # How an operator's input becomes the command the policy observes, carried
    # in an exported policy's contract. It is attached to the env config rather
    # than to a command term because the bindings are the robot's, not this
    # particular environment's: an export built without teleop still needs to
    # tell a consumer which way is forward. `scripts/export.py` picks it up.
    #
    # A callable, built at export: the file may describe command terms this
    # function does not install -- `jumper.posture` adds its posture term after
    # this returns -- and the block carries their rests as numbers read off
    # those terms, the way a reference contract is attached.
    if parsed_controls is not None:
        cfg.controller_contract = functools.partial(controller_contract, parsed_controls, cfg)

    return cfg


__all__ = ["velocity_env_cfg"]
