"""jumper.jump -- reference-guided high jump.

The robot performs one in-place high jump following
`tasks/jumper/jump/ref/high_jump_flat.npz` (crouch-jump-tuck-land, 1.47 s). The
reward / termination / RSI reset are ported from `tasks/hexa_5d/jump/` in
kk-rl-lab (a scheme tuned on the same robot).

Key differences from the locomotion task (`common/velocity_env.py`):

- **200 Hz control** (timestep 0.0025 x decimation 2). The push-off phase of
  the reference lasts only 40 ms; at 50 Hz the whole push-off is just 2
  command steps, physically impossible to reproduce.
- **kp=20** (the real robot uses kp=10). The reference was recorded and
  verified reachable at kp=20/kd=0.5; the torque limit is untouched, at
  `EFFORT_LIMIT` = 1.7464 N*m. The user decided: tracking takes priority over
  sim2real consistency. (This said 2.0 N*m, which was the placeholder the
  effort limit carried before it was measured off the motor curve.)
- **Residual action**: position target = reference joint angles + scale x
  policy output. During push-off the target deviates from HOME by on the order
  of 1 rad; with HOME as the baseline it will not learn.
- **RSI reset**: every env is born at a random reference phase (0~0.9).
- No velocity command: the reference defines all motion, and the reward only
  measures the distance from the reference.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from ..common.constants import (
    CHASSIS_LINKS,
    EFFORT_LIMIT,
    FOOT_GEOMS,
    GAIT_JOINTS,
    HOME,
    IMU_ANG_VEL_NOISE,
    IMU_GRAVITY_NOISE,
    JUMPER_FOOT_FRICTION_RANGE,
    LIMB_LINKS,
    LEGS,
    PUSH_VELOCITY_RANGE,
    TIP_LINKS,
    get_jumper_robot_cfg,
)
from ..common.tof import tof_sensor
from . import mdp as jump_mdp

# ── Physics settings for this task ─────────────────────────────────────────────
#: 200 Hz control: timestep x decimation = 0.005 s
_SIM_TIMESTEP = 0.0025
_DECIMATION = 2

#: PD stiffness at which the reference was validated. The real robot is kp=10;
#: tracking takes priority (see the module docstring).
_JUMP_KP = 20.0

#: Offsets of the future reference preview in the observation; see mdp/observations.py
_FUTURE_OFFSETS = jump_mdp.observations.FUTURE_OFFSETS

#: On a board the jump starts when the controller hands this mode its policy --
#: once the switch-in ramp has reached the home pose -- not on a button pressed
#: inside the mode. Switching into the mode is the whole of asking for a jump,
#: and which control does that belongs to the deploy manifest
#: (`deploy/manifests.json`), like every other binding.
#:
#: It is also what the policy trained on. Every spawn is reference-state
#: initialised past its go (`rsi_fraction=1.0`, a phase in [0, 0.9)) with its
#: history fresh, and the command resamples only at reset, so `time_since_go`
#: is never negative in training: a start at phase 0 on the first tick after the
#: hand-off is the edge of that range, while standing in the go frame waiting
#: for a press is a state the policy never saw.
#:
#: It waited for a `go_event`, `jump_go` on A's release, until 2026-09-26. A
#: recording that wants a person to pick the moment inside the mode passes
#: `go_event` instead; the controller refuses a contract carrying both.
_STARTS_ON_ENTRY = True

#: Termination threshold on the sum of squared joint errors (measured in the lab)
_DIVERGE_THRESHOLD = 6.0

#: Non-foot collision geoms -- the objects of `undesired_contacts`. The feet are
#: the `*_palm_pad_b_link` / `*_foot_tip_link` meshes; everything else (chassis,
#: limb links, and the front `*_finger_tip_link` / `*_palm_pad_f_link` pads)
#: touching the ground is undesired.
_BODY_GEOMS = tuple(
    f"{link}_meshcol" for link in CHASSIS_LINKS + LIMB_LINKS + TIP_LINKS
)


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Reference-guided high jump.

    Args:
        asset: path to the model XML, from `--model`. None uses the default jumper.xml.
        play: replay mode -- observation noise and external disturbances off, episode stretched.
    """
    cfg = make_velocity_env_cfg()

    # ── Sim: 200 Hz control; jumper walk contact budget, as measured ─────────────
    cfg.sim.mujoco.timestep = _SIM_TIMESTEP
    cfg.sim.njmax = 512
    cfg.sim.nconmax = 128
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.decimation = _DECIMATION

    # ── Scene: flat ground + robot at kp=20 ────────────────────────────────────
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    robot = get_jumper_robot_cfg(effort_limit=EFFORT_LIMIT, asset=asset)
    # The gain belongs to this task, not to the robot: `constants.STIFFNESS` is a
    # hardware fact the other tasks share, and a `stiffness=` parameter on
    # `get_jumper_robot_cfg` would invite the next task to pick its own without
    # saying why. `get_jumper_robot_cfg` returns a fresh instance every call
    # precisely so it can be overridden in place here. Damping stays at the
    # hardware value -- only the stiffness the recording was made at differs.
    for actuator in robot.articulation.actuators:
        actuator.stiffness = _JUMP_KP
    cfg.scene.entities = {"robot": robot}

    foot_geoms = FOOT_GEOMS

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=leg, entity="robot") for leg in LEGS
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.02, num_samples=4)

    # Foot contact (for ref_contact's contact-schedule comparison + airborne metric)
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=foot_geoms, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    # Non-foot contact (undesired_contacts)
    body_ground_cfg = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(mode="geom", pattern=_BODY_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, body_ground_cfg)
    # No terrain ray casts needed on flat ground
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )

    # ── Observations: drop command/terrain/foot terms; add phase + future preview
    for group in cfg.observations.values():
        for name in ("command", "height_scan", "foot_height", "foot_air_time",
                     "foot_contact", "foot_contact_forces"):
            group.terms.pop(name, None)

    # Base linear velocity is critic-only, for the reason `common/velocity_env.py`
    # gives at length: an IMU yields angular velocity from the gyro, but linear
    # velocity comes only from a state estimator integrating it against foot
    # contacts, and this robot has none. The critic is discarded at export, so
    # this costs the deployed policy nothing.
    #
    # Repeated here rather than inherited because this task builds on mjlab's
    # `make_velocity_env_cfg`, not on the jumper skeleton, so the skeleton's own
    # deletion never runs. That was not noticed until `export.py` refused the
    # policy: it trained and replayed perfectly and could not go on the robot.
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["actor"].terms["jump_phase"] = ObservationTermCfg(
        func=jump_mdp.observations.jump_phase,
        params={"command_name": "jump"},
    )
    cfg.observations["actor"].terms["ref_future"] = ObservationTermCfg(
        func=jump_mdp.observations.ref_future,
        params={"command_name": "jump", "offsets": _FUTURE_OFFSETS},
    )
    cfg.observations["critic"].terms["jump_phase"] = ObservationTermCfg(
        func=jump_mdp.observations.jump_phase,
        params={"command_name": "jump"},
    )
    cfg.observations["critic"].terms["ref_future"] = ObservationTermCfg(
        func=jump_mdp.observations.ref_future,
        params={"command_name": "jump", "offsets": _FUTURE_OFFSETS},
    )

    # joint_pos / joint_vel take only the 20 controlled joints (as in the locomotion task)
    _gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))
    for _group in cfg.observations.values():
        for _name in ("joint_pos", "joint_vel"):
            _term = _group.terms.get(_name)
            if _term is not None:
                _term.params = dict(_term.params or {}) | {"asset_cfg": _gait_cfg}

    # ── What the IMU is actually like ──────────────────────────────────────────
    # This robot's noise levels, not the +/-0.2 rad/s and +/-0.05 mjlab's skeleton
    # gives Go1. `common/velocity_env.py` makes the same replacement for every
    # task built on it; this one is built on mjlab's skeleton directly, so it
    # inherited Go1's instead -- a gyro half as noisy as the one every walking
    # policy on this robot trains against, and nothing said so.
    # `tests/test_hardware_facts.py` pins it, and the friction and push further down.
    for group in cfg.observations.values():
        if "base_ang_vel" in group.terms:
            group.terms["base_ang_vel"].noise = Unoise(
                n_min=-IMU_ANG_VEL_NOISE, n_max=IMU_ANG_VEL_NOISE
            )
        if "projected_gravity" in group.terms:
            group.terms["projected_gravity"].noise = Unoise(
                n_min=-IMU_GRAVITY_NOISE, n_max=IMU_GRAVITY_NOISE
            )

    # ── Actions: residual position target (reference + scale x output) ─────────
    cfg.actions["joint_pos"] = jump_mdp.actions.ReferenceResidualJointPositionActionCfg(
        entity_name="robot",
        actuator_names=list(GAIT_JOINTS),
        scale={j: 0.25 for j in GAIT_JOINTS},
        command_name="jump",
    )

    # ── Command: reference motion player, replacing twist ──────────────────────
    cfg.commands = {
        "jump": jump_mdp.commands.JumpMotionCommandCfg(
            entity_name="robot",
            motion_file=str(jump_mdp.reference.JUMP_REF_NPZ),
            command_delay_range=(0.2, 1.0),
            rsi_fraction=1.0,
        )
    }

    # ── Reward: imitation channel + hard-property terms (per-term weights from the lab) ──
    # The exp kernel's perfect score is physically reachable; apart from action_rate
    # (which smooths the action sequence), any positive score outside the reference
    # reopens the loophole of scoring without jumping.
    cfg.rewards = {
        "ref_joint_pos": RewardTermCfg(
            func=jump_mdp.rewards.ref_joint_pos, weight=3.0,
            params={"command_name": "jump", "sigma_sq": 0.20},
        ),
        "ref_base_height": RewardTermCfg(
            func=jump_mdp.rewards.ref_base_height, weight=3.0,
            params={"command_name": "jump", "sigma_sq": 0.0025},
        ),
        "ref_base_lin_vel": RewardTermCfg(
            func=jump_mdp.rewards.ref_base_lin_vel, weight=2.0,
            params={"command_name": "jump", "sigma_sq": 0.30},
        ),
        "ref_foot_pos": RewardTermCfg(
            func=jump_mdp.rewards.ref_foot_pos, weight=2.0,
            params={"command_name": "jump", "sigma_sq": 0.004},
        ),
        "ref_base_ori": RewardTermCfg(
            func=jump_mdp.rewards.ref_base_ori, weight=1.5,
            params={"command_name": "jump", "sigma_sq": 0.05},
        ),
        "ref_contact": RewardTermCfg(
            func=jump_mdp.rewards.ref_contact, weight=2.5,
            params={"command_name": "jump", "force_threshold": 1.0},
        ),
        "ref_joint_vel": RewardTermCfg(
            func=jump_mdp.rewards.ref_joint_vel, weight=1.0,
            params={"command_name": "jump", "sigma_sq": 40.0},
        ),
        "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=0.5),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "undesired_contacts": RewardTermCfg(
            func=jump_mdp.rewards.undesired_contacts, weight=-1.0,
            params={"sensor_name": "body_ground_contact", "force_threshold": 1.0},
        ),
        # The lab used -0.01 at scale 0.3; this project uses scale 0.25, so by
        # the project convention we compensate by (0.3/0.25)**2, keeping the
        # penalty per unit of joint motion unchanged.
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2, weight=-0.01 * (0.30 / 0.25) ** 2
        ),
    }

    # ── Termination ────────────────────────────────────────────────────────────
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": 1.2},
        ),
        "reference_diverged": TerminationTermCfg(
            func=jump_mdp.terminations.reference_diverged,
            params={"command_name": "jump", "threshold": _DIVERGE_THRESHOLD},
        ),
    }

    # ── Events: DR from the locomotion task, reset swapped for RSI ─────────────
    # The ranges are the robot's, from `constants.py`, as `common/velocity_env.py`
    # sets them for the walking tasks: what the feet are made of and what the
    # robot weighs are not things a jump changes.
    if "foot_friction" in cfg.events:
        cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
            "robot", geom_names=foot_geoms
        )
        # `operation` is "abs", so this replaces the feet's friction outright.
        # mjlab's (0.3, 1.2) is a generic rubber band that sits at or below what
        # the silicone tips give on a clean dry floor.
        cfg.events["foot_friction"].params["ranges"] = JUMPER_FOOT_FRICTION_RANGE
    if "base_com" in cfg.events:
        cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
    if "push_robot" in cfg.events:
        # jumper weighs ~2 kg; Go1's ±0.5 m/s would tip it over. These were once
        # the same numbers written out here, which kept them equal only until
        # either copy moved.
        cfg.events["push_robot"].params["velocity_range"].update(PUSH_VELOCITY_RANGE)

    cfg.events = {
        # First fall back to the default pose, then override the RSI envs with reference phases.
        "reset_to_default": EventTermCfg(
            func=envs_mdp.reset_scene_to_default,
            mode="reset",
        ),
        "reset_from_reference": EventTermCfg(
            func=jump_mdp.events.reset_from_reference_phase,
            mode="reset",
            params={"command_name": "jump", "rsi_fraction": 1.0,
                    "phase_range": (0.0, 0.9)},
        ),
        # No joint-speed clamp. How fast a joint turns is the servo curve's to say
        # -- `ServoCurveActuator`, from `assets/jumper/motor/motor_config.yaml` --
        # as in every other task. A 300 rpm wall stood here: the generator's old
        # rectangular motor model. `high_jump_flat.npz` was generated against the
        # curve (`meta["motor"]`, no speed limit short of the 611 rpm cutoff) and
        # reaches 338 rpm itself, which the wall forbade.
        #
        # Measured with zero residual, 64 envs, RTX 5090 D, warp:cuda: rise
        # 0.134 m with the wall and 0.133 m without, peak joint speed 31.42 and
        # 39.3 rad/s. It was not what held the jump down; it was a limit the
        # robot does not have.
        **cfg.events,
    }
    # The locomotion task's reset_base / reset_robot_joints are replaced by the RSI scheme above.
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)

    # ── Curriculum: no terrain, no command progression ─────────────────────────
    cfg.curriculum = {}

    # ── Episode and viewer ─────────────────────────────────────────────────────
    # Trajectory span is ~1.22 s plus a settling margin after landing; 3.5 s is enough.
    cfg.episode_length_s = 3.5
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 0.9
    cfg.viewer.elevation = -15.0

    if play:
        # The usual play convention -- stretch the episode to forever -- is wrong for a
        # one-shot motion: the policy would jump once and then hold the landing pose
        # indefinitely. Keep the 3.5 s episode so replay loops whole attempts, and spawn
        # at the start of the motion rather than at the training-time random phase, so
        # what is on screen is a crouch-jump-land and not its second half.
        cfg.events["reset_from_reference"].params["phase_range"] = (0.0, 0.01)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}

    # ── The recording, for a board ─────────────────────────────────────────────
    # A residual policy is not runnable without the trajectory it corrects, so
    # the trajectory is part of the contract. Attached the way
    # `controller_contract` is -- a callable, because loading the npz is work
    # that `--list` and a dry run must not do.
    #
    # Every number comes off the loaded reference rather than being restated
    # here: a rate typed twice is a board playing a slightly different recording
    # than the policy trained against, and nothing would say so.
    def _reference_contract() -> dict:
        # **This robot's joints, not the recording's.** It used to pass the npz's
        # own `meta["joint_order"]`, which worked only while the two agreed: the
        # contract's `joint_order` is what a board resolves its wire indices
        # against by name, so handing it the recording's names ships a bundle
        # naming joints no model has. The V1.6.1 rename broke that agreement; the
        # npz has been relabelled to agree again (`reference.py::JUMP_REF_NPZ`),
        # but agreement is what the recording happens to have, not something
        # the contract may rest on. `HOME`'s key order is the model's
        # declaration order, which is the order of the action vector
        # (`tests/test_joint_order.py` pins both).
        ref = jump_mdp.reference.JumpReference(
            jump_mdp.reference.JUMP_REF_NPZ, "cpu", list(HOME)
        )
        # One control step: `mdp/actions.py` samples the baseline at the
        # interval's **end**, because the target is held over the interval.
        return ref.deploy_contract(_FUTURE_OFFSETS, _SIM_TIMESTEP * _DECIMATION,
                                   starts_on_entry=_STARTS_ON_ENTRY)

    cfg.reference_contract = _reference_contract

    # The dToF, which the model carries (`common/tof.py`). Replay only, where the
    # live viewer shows it in the window's corner; nothing in training reads it,
    # and there it would be 2268 rays per environment per step spent on nothing.
    if play:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

    return cfg


__all__ = ["env_cfg"]
