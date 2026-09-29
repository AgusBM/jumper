"""The environment every dance task builds: `dance_env_cfg`.

The skeleton is mjlab's tracking task (`make_tracking_env_cfg`), the BeyondMimic
re-implementation that ships in the vendored tree, wired for the Unitree G1. This is
the hexapod's version of `rl/mjlab/tasks/tracking/config/g1/env_cfgs.py`, and the
changes fall into four groups.

**1. The robot.** Six legs rather than two, `base_link` rather than `torso_link`,
a hexapod's contact count, and **all 22 joints in the action space** -- the
velocity tasks hold the two grippers closed at `HOME`, and a dance sweeps them.

**2. The scale.** Every threshold in the tracking task was chosen for a 1.3 m
humanoid. This robot stands 0.105 m tall, so each one is a parameter here, and each
dance task re-derives it against something measured about this robot or its clip.

**3. The observation names.** mjlab's tracking group is unexportable here; see
`observations.py`. The actor is rebuilt from names `scripts/export.py` accepts.

**4. The four extra objectives**, stated as four reward terms plus the
terminations: don't fall, low energy, smooth motion, small torque peaks.

## Mechanism here, numbers in the task

It was `jumper.dance`'s `env_cfg.py` while that was the only dance. What moved here
is what does not depend on the clip or on a choice: which bodies are tracked, which
terms the policy observes, the robot's contact sizing, the reference contract the
export writes. **Every weight, threshold, range and noise level is a required
argument**, with no default, so a dance task states all of them -- the rule in
`CLAUDE.md` that a tuning number in `common/` is one every task inherits without
choosing. The reasoning for each lives beside the number, in the task.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from ..constants import (
    ACTION_SCALE,
    ACTUATOR_FORCE_NOISE,
    FEET,
    FOOT_GEOMS,
    HOME,
    IMU_ANG_VEL_NOISE,
    IMU_GRAVITY_NOISE,
    JUMPER_FOOT_FRICTION_RANGE,
    get_jumper_robot_cfg,
)
from ..mdp.observations import actuator_force
from ..tof import tof_sensor
from .motion import ensure_motion_npz
from .observations import (
    FUTURE_HORIZONS,
    clip_phase,
    ref_future,
    ref_joint_pos,
    ref_joint_vel,
    ref_tilt_error,
)
from .rewards import (
    actuator_power_cost,
    motion_joint_pos_error_exp,
    motion_joint_vel_error_exp,
    peak_actuator_force,
    torque_headroom_cost,
)

__all__ = ["COMMAND", "SUPPORT_FEET", "TRACKED_BODIES", "dance_env_cfg"]

#: The command's name, used by every term that reads the clip.
COMMAND = "motion"

#: The bodies scored against the reference: the base and the six limb tips.
#:
#: Not all 31. The relative-body-position reward exists to stop joint errors
#: compounding down a chain, so it wants the ends of the chains, and the ends here
#: are the six feet -- two of which, `LF_palm_pad_b_link` and `RF_palm_pad_b_link`, are the front
#: arms' jaws and therefore carry most of the choreography (they lift to 0.29 m
#: while the four legs stay planted). Adding the intermediate links would score the
#: same errors again, once per joint they pass through, and widen the critic's
#: observation for it.
TRACKED_BODIES: tuple[str, ...] = ("base_link",) + FEET

#: The four legs that stay on the ground, as body names. `SUPPORT_LEGS` in
#: `motion.py` is the same four by leg prefix.
SUPPORT_FEET: tuple[str, ...] = (
    "LM_foot_tip_link", "RM_foot_tip_link", "LR_foot_tip_link", "RR_foot_tip_link",
)


def dance_env_cfg(
    *,
    media: Path,
    asset: Path | None,
    play: bool,
    episode_s: float,
    rsi_pose_range: dict[str, tuple[float, float]],
    rsi_velocity_range: dict[str, tuple[float, float]],
    rsi_joint_range: tuple[float, float],
    joint_pos_noise: float,
    joint_vel_noise: float,
    joint_pos_std: float,
    joint_pos_weight: float,
    joint_vel_std: float,
    joint_vel_weight: float,
    action_rate_weight: float,
    action_acc_weight: float,
    power_weight: float,
    torque_headroom_weight: float,
    joint_limit_weight: float,
    self_collision_weight: float,
    self_collision_force: float,
    anchor_height_error: float,
    anchor_tilt_error: float,
    support_foot_error: float,
    push_interval_s: tuple[float, float],
    push_velocity_range: dict[str, tuple[float, float]],
) -> ManagerBasedRlEnvCfg:
    """Build a dance task's environment config.

    Args:
        media: the task's material directory; see `motion.find_material`.
        asset: model XML path, from `--model`. None uses the default jumper.xml.
            The clip is written against the default asset's joints, so a
            structurally different model fails during conversion, by name.
        play: replay mode -- observation noise and disturbances off, the clip
            played from its first frame rather than sampled, unbounded episodes.
        episode_s: episode length, seconds. The clip is usually longer, so an
            episode is a window into it and the command's adaptive sampler chooses
            where the window starts, weighting the passages that have been failing.
        rsi_pose_range, rsi_velocity_range, rsi_joint_range: reference-state
            initialisation, the perturbation around the reference at every reset
            (`MotionCommandCfg.pose_range`, `.velocity_range`,
            `.joint_position_range`).
        joint_pos_noise, joint_vel_noise: the actor's uniform noise on the joint
            readings. The IMU and current sensor's come from `constants.py`.
        joint_pos_std, joint_pos_weight: joint-space position tracking,
            `exp(-mse / std^2)`.
        joint_vel_std, joint_vel_weight: joint-space velocity tracking, same form.
        action_rate_weight, action_acc_weight: first and second difference of the
            raw action.
        power_weight: positive electrical power, W.
        torque_headroom_weight: torque above the servo's continuous rating.
        joint_limit_weight: mjlab's soft joint-limit penalty.
        self_collision_weight, self_collision_force: leg-on-leg contact above
            `self_collision_force` newtons.
        anchor_height_error: termination on the base's height error, metres.
        anchor_tilt_error: termination on tilt error, as `1 - cos(tilt)`.
        support_foot_error: termination on a support foot's height error, metres.
        push_interval_s, push_velocity_range: the random shove's period and size.
    """
    cfg = make_tracking_env_cfg()

    # ── The robot ─────────────────────────────────────────────────────────
    cfg.scene.entities = {"robot": get_jumper_robot_cfg(asset=asset)}

    # A hexapod with all feet down has far more contacts than a biped, and
    # mjlab's tracking defaults (nconmax 35, njmax 250) are sized for the G1.
    # These are the velocity skeleton's, measured on this robot: njmax is the
    # per-world constraint row limit and overflows around 500 once it moves.
    cfg.sim.njmax = 512
    cfg.sim.nconmax = 128
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 128

    # Leg-vs-leg contact. Both sides are the whole-robot subtree, as mjlab wires
    # the G1's; geoms on one leg share a collision-mask bit and so never generate a
    # contact at all, which is what makes this mean "one leg against a *different*
    # leg". See the mask derivation in `common/constants.py`.
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=cfg.decimation,
        ),
    )

    # ── Actions: all 22 joints ────────────────────────────────────────────
    # The velocity tasks drive 20 and hold the two grippers at `HOME` by PD.
    # **A choreography moves them** -- the crab dance's fingers sweep +/-0.262 rad
    # -- so excluding them here would silently drop that part of the dance: the
    # reference would ask for finger motion, the policy would have no channel to
    # produce it, and the tracking reward would carry a permanent floor no log
    # would attribute to the grippers.
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.actuator_names = list(HOME)
    joint_pos_action.scale = {j: ACTION_SCALE for j in HOME}

    # ── The clip ──────────────────────────────────────────────────────────
    # Converted at build time from whatever is in the task's `media/`, cached, and
    # rebuilt when the source changes. See `motion.py` for the contract and for
    # why the reference is the measured channels rather than the commanded ones.
    control_dt = cfg.decimation * cfg.sim.mujoco.timestep

    motion_cmd = cfg.commands[COMMAND]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(ensure_motion_npz(control_dt, media, asset=asset))
    motion_cmd.anchor_body_name = "base_link"
    motion_cmd.body_names = TRACKED_BODIES
    motion_cmd.pose_range = dict(rsi_pose_range)
    motion_cmd.velocity_range = dict(rsi_velocity_range)
    motion_cmd.joint_position_range = tuple(rsi_joint_range)

    # ── Observations ──────────────────────────────────────────────────────
    # Rebuilt rather than edited: mjlab's group is keyed by names the deployment
    # cannot build (`command`, `motion_anchor_pos_b`, `motion_anchor_ori_b`), and
    # `command` would additionally be renamed to `velocity_commands` on export --
    # a dance command described as a velocity command. See `observations.py`.
    #
    # No history on any term. The velocity skeleton gives five frames to the
    # per-joint signals because it is inferring contact transients it cannot see;
    # here the policy is handed the reference trajectory outright, including three
    # future horizons, so history would mostly repeat what `ref_future` states
    # exactly.
    _clip_terms = {
        "clip_phase": ObservationTermCfg(
            func=clip_phase, params={"command_name": COMMAND}
        ),
        "ref_joint_pos": ObservationTermCfg(
            func=ref_joint_pos, params={"command_name": COMMAND}
        ),
        "ref_joint_vel": ObservationTermCfg(
            func=ref_joint_vel, params={"command_name": COMMAND}
        ),
        "ref_future": ObservationTermCfg(
            func=ref_future,
            params={"command_name": COMMAND, "horizons": FUTURE_HORIZONS},
        ),
        # Tilt only: no base state estimator for the position half, and a
        # six-axis IMU for the yaw half. `observations.py` has the measurements --
        # the crab dance's root travels 0.148 m across the clip, and the 22.3 deg of
        # yaw it uses is the same order as the drift would be.
        "ref_tilt_error": ObservationTermCfg(
            func=ref_tilt_error, params={"command_name": COMMAND}
        ),
    }

    def _proprioception(noisy: bool) -> dict[str, ObservationTermCfg]:
        """The terms that describe the robot itself.

        A fresh `ObservationTermCfg` per group, never one instance shared: the
        observation manager resolves term configs in place (it turns `scale` into a
        tensor and collects terms carrying a `reset`), so a shared object would
        have the two groups resolve over each other.
        """
        return {
            # This robot's IMU and current sensor, from `constants.py`. The dance
            # tasks are not built on `common/velocity_env.py`, which applies these
            # to the walking tasks, so this has to apply them itself.
            # `tests/test_hardware_facts.py` pins them.
            "base_ang_vel": ObservationTermCfg(
                func=mdp.builtin_sensor,
                params={"sensor_name": "robot/imu_ang_vel"},
                noise=Unoise(n_min=-IMU_ANG_VEL_NOISE, n_max=IMU_ANG_VEL_NOISE) if noisy else None,
            ),
            "projected_gravity": ObservationTermCfg(
                func=mdp.projected_gravity,
                noise=Unoise(n_min=-IMU_GRAVITY_NOISE, n_max=IMU_GRAVITY_NOISE) if noisy else None,
            ),
            "joint_pos": ObservationTermCfg(
                func=mdp.joint_pos_rel,
                noise=Unoise(n_min=-joint_pos_noise, n_max=joint_pos_noise) if noisy else None,
            ),
            "joint_vel": ObservationTermCfg(
                func=mdp.joint_vel_rel,
                noise=Unoise(n_min=-joint_vel_noise, n_max=joint_vel_noise) if noisy else None,
            ),
            "actions": ObservationTermCfg(func=mdp.last_action),
            # Exported as `joint_torque`; `scripts/export.py` renames it. On
            # hardware it is a scaled current measurement, so it carries the
            # current sensor's noise -- it was observed clean until that was noticed.
            "actuator_force": ObservationTermCfg(
                func=actuator_force,
                noise=(
                    Unoise(n_min=-ACTUATOR_FORCE_NOISE, n_max=ACTUATOR_FORCE_NOISE)
                    if noisy else None
                ),
            ),
        }

    cfg.observations["actor"] = ObservationGroupCfg(
        terms={**_clip_terms, **_proprioception(noisy=True)},
        concatenate_terms=True,
        enable_corruption=True,
    )
    # The critic additionally gets what the robot cannot measure: base linear
    # velocity (no state estimator -- `export.py` lists it as unmeasurable) and the
    # per-body pose error the reward is actually computed from. Standard asymmetric
    # actor-critic; the critic is thrown away at export, so this costs the deployed
    # policy nothing.
    cfg.observations["critic"] = ObservationGroupCfg(
        terms={
            **_clip_terms,
            **_proprioception(noisy=False),
            "base_lin_vel": ObservationTermCfg(
                func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
            ),
            "body_pos": ObservationTermCfg(
                func=mdp.robot_body_pos_b, params={"command_name": COMMAND}
            ),
            "body_ori": ObservationTermCfg(
                func=mdp.robot_body_ori_b, params={"command_name": COMMAND}
            ),
        },
        concatenate_terms=True,
        enable_corruption=False,
    )

    # ── Rewards ───────────────────────────────────────────────────────────
    # mjlab's six tracking terms keep its weights and stds; what follows replaces
    # its regularisers. `jumper.dance`'s env_cfg.py has the budget arithmetic.
    for name in ("action_rate_l2", "joint_limit", "self_collisions"):
        cfg.rewards.pop(name, None)

    # Joint-space tracking, which mjlab's recipe does not have: all six of its
    # terms are body or root quantities, and on a robot with twenty-two joints a
    # policy can satisfy every one of them with the legs folded differently than
    # the clip asks. For a dance the shape of the limbs *is* the thing watched.
    cfg.rewards["motion_joint_pos"] = RewardTermCfg(
        func=motion_joint_pos_error_exp,
        weight=joint_pos_weight,
        params={"command_name": COMMAND, "std": joint_pos_std},
    )
    cfg.rewards["motion_joint_vel"] = RewardTermCfg(
        func=motion_joint_vel_error_exp,
        weight=joint_vel_weight,
        params={"command_name": COMMAND, "std": joint_vel_std},
    )
    # Smoothness in two orders: a policy can hold a low first difference while
    # alternating direction every step, and only the second difference sees it.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=mdp.action_rate_l2, weight=action_rate_weight
    )
    cfg.rewards["action_acc_l2"] = RewardTermCfg(
        func=mdp.action_acc_l2, weight=action_acc_weight
    )
    # `actuator_power_cost`, not mjlab's `electrical_power_cost`: that one reads
    # `qfrc_actuator`, which the native backend does not provide. See the docstring
    # in `rewards.py` -- it is the same seam trap as `nan_detection` below.
    cfg.rewards["power"] = RewardTermCfg(
        func=actuator_power_cost,
        weight=power_weight,
        params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
    )
    # Torque above what the servo can hold. See `rewards.py` for why this is not
    # `joint_torques_l2`.
    cfg.rewards["torque_headroom"] = RewardTermCfg(
        func=torque_headroom_cost,
        weight=torque_headroom_weight,
        params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
    )
    cfg.rewards["joint_limit"] = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=joint_limit_weight,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=self_collision_weight,
        params={"sensor_name": "self_collision", "force_threshold": self_collision_force},
    )

    # ── Terminations ──────────────────────────────────────────────────────
    # There is no "alive" bonus, deliberately. Every tracking term is positive, so
    # ending the episode forfeits all of them -- falling is already the most
    # expensive thing the policy can do, and a survival bonus on top would pay it
    # to stand still through the passages it finds hard.
    cfg.terminations["anchor_pos"].params["threshold"] = anchor_height_error
    cfg.terminations["anchor_ori"].params["threshold"] = anchor_tilt_error
    # Bounded on **the four support legs only**: the two front arms are the dance,
    # so bounding them would terminate on the choreography itself.
    cfg.terminations["ee_body_pos"].params["body_names"] = SUPPORT_FEET
    cfg.terminations["ee_body_pos"].params["threshold"] = support_foot_error

    # **`mdp.nan_detection` is not mounted here, and must not be.** It looks like a
    # free safety net -- catch a diverged environment before it poisons the batch --
    # and it was added on that reasoning, then removed.
    #
    # It reads `data.qacc_warmstart`, which the native backend does not provide:
    # `native_sim.py` raises `AttributeError` naming the field. So the task trains
    # on warp and dies on the first step under `--backend native`, which is exactly
    # the asymmetry the two-backend seam exists to avoid, and it is invisible to
    # anyone testing on one backend only. Nothing else in this repository mounts it
    # -- not the velocity tasks, not mjlab's own velocity or tracking configs.
    #
    # If a NaN guard is wanted, `qacc_warmstart` has to reach the native backend
    # first (its error message says where), and that is a change to `rl/`, not to a
    # task.

    # ── Events ────────────────────────────────────────────────────────────
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = list(FOOT_GEOMS)
    # The silicone tips' band, from `constants.py`. `operation` is "abs", so this
    # replaces the feet's friction outright; the skeleton's (0.3, 1.2) is a generic
    # rubber band at or below what the tips give on a clean dry floor.
    cfg.events["foot_friction"].params["ranges"] = JUMPER_FOOT_FRICTION_RANGE
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
    # `base_com`'s offsets and `encoder_bias`'s +/-0.01 rad are the tracking
    # skeleton's, inherited as they are. Neither is measured on this robot or in
    # `constants.py`, and the walking tasks inherit different ones from theirs
    # (+/-0.025/0.025/0.03 m, +/-0.015 rad): a known gap, not a choice.
    cfg.events["push_robot"].interval_range_s = tuple(push_interval_s)
    cfg.events["push_robot"].params["velocity_range"] = dict(push_velocity_range)

    # ── Metrics ───────────────────────────────────────────────────────────
    # The peak the torque term is aimed at, reported directly. A reward is a sum
    # over the episode, so it cannot distinguish one bad spike from a mild
    # persistent overshoot; this can, and it is the number to read when deciding
    # whether `torque_headroom`'s weight is right.
    cfg.metrics["peak_torque"] = MetricsTermCfg(
        func=peak_actuator_force,
        reduce="max",
        params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
    )

    cfg.viewer.body_name = "base_link"
    # 0.105 m tall, against the G1's 1.3: mjlab's 2.8 m camera distance frames an
    # empty floor with a speck in the middle.
    cfg.viewer.distance = 0.9
    cfg.episode_length_s = episode_s

    if play:
        # Effectively unbounded, so the whole dance plays through rather than
        # cutting at the episode length.
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        # From the first frame, unperturbed: a replay is meant to show the dance,
        # and reference-state initialisation would start it in the middle.
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.joint_position_range = (0.0, 0.0)
        motion_cmd.sampling_mode = "start"

    def _reference_contract() -> dict:
        """The recording the deployment plays, and how to read it.

        Every number is read back off the converted clip rather than restated
        here: a rate or a frame count typed a second time is a board playing a
        slightly different recording than the policy trained against, and
        nothing would say so.

        Three tables, one per thing the policy observes. `q` feeds
        `ref_joint_pos` and `ref_future`, `qd` feeds `ref_joint_vel`, and
        `root_quat` -- the anchor body's orientation -- feeds `ref_tilt_error`.
        The controller refuses a term whose table is absent rather than reading
        zeros, which would be a motionless reference a standing robot tracks
        perfectly.
        """
        import numpy as np

        clip = np.load(motion_cmd.motion_file)
        anchor = TRACKED_BODIES.index(motion_cmd.anchor_body_name)
        n = int(clip["joint_pos"].shape[0])
        fps = float(clip["fps"])
        return {
            "name": Path(motion_cmd.motion_file).stem,
            "npz": motion_cmd.motion_file,
            "joint_order": [str(j) for j in clip["joint_names"]],
            # The policy drives the joints outright; the recording is what it is
            # scored against, not a baseline it corrects. `jumper.jump`'s is the
            # other kind.
            "residual_action": False,
            # A piece of music: it starts when you switch into the mode, and it
            # ends, after which the cascade hands back to the default gait.
            "starts_on_entry": True,
            "rec_hz": fps,
            "duration": n / fps,
            # `FUTURE_HORIZONS` is in **control steps**, and the deployment reads
            # seconds. Converted here, once, where the rate is known -- the same
            # numbers in two units is how they come to disagree. `jumper.jump`'s
            # are in seconds already and mean something different by the same
            # term name, which is why the export gate checks the width against
            # this list.
            "lookahead_s": [h / fps for h in FUTURE_HORIZONS],
            "reference": {
                "q": clip["joint_pos"],
                "qd": clip["joint_vel"],
                "root_quat": clip["body_quat_w"][:, anchor],
            },
        }

    cfg.reference_contract = _reference_contract

    # The dToF, which the model carries (`common/tof.py`). Replay only, where the
    # live viewer shows it in the window's corner; nothing in training reads it,
    # and there it would be 2268 rays per environment per step spent on nothing.
    if play:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

    return cfg
