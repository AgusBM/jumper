"""jumper.ref_free_jump -- a high jump whose policy needs no recording.

`jumper.jump` is a residual: its target is the recording's command plus the
policy's correction, and its actor watches the recording's next 0.16 s. Both put
the trajectory on the robot, every step, for as long as the policy runs. This
task removes both from the policy the robot runs. The action is an absolute
joint target around HOME, as in the walking tasks, and the actor observes its
body and nothing else: no recording, and no clock. It jumps when it starts to
run, and the episode is that jump and the way back to HOME.

## Where the recording is used, and only while training

- **As a feedforward** (`mdp/actions.py`): inside the span the action term adds
  the recording's command, times a factor that is 1 until
  `PRIOR_HOLD_ITERATIONS` and 0 from `PRIOR_END_ITERATIONS`. At the start a zero
  action is the recording's jump; by the end the policy makes it alone.
- **As a prior on the joints** (`mdp/rewards.py::prior_joint_pos`,
  `prior_joint_vel`): rewards for the joint angles and speeds the recording
  reached at the same moment, weighted down to zero over the same window
  (`mdp/curriculum.py`).
- **As spawn states**: `RSI_FRACTION` of episodes start somewhere along it, so
  the flight and the landing are practised before the policy can reach them.
  The rest start at HOME, where the deployment will hand over.
- **In the critic**: the feedforward's size and direction and the flight
  bookkeeping, which the value function needs and the deployed actor never sees.

## Why a feedforward, and not only a reward

The first design had the recording as a reward alone -- a prior on the commanded
targets -- and the policy learnt to crouch and stay crouched. `model_300` of
`2026-09-26_18-21-37` matched it at 0.91 in the crouch and 0.99 in the landing,
at 0.39 and 0.33 in the push and the flight -- no better than standing at HOME
-- and never left the ground. `mdp/actions.py` has the measurement, and why a
prior on the joints' state alone has the same shape. What gets past it is
starting from the jump, which is what the feedforward does.

## What it is trained to do

First the recording's jump, then as high as it can go -- landing, and back at
HOME both times. An episode starts at HOME with the jump already under way --
there is no waiting for a go -- and ends, as a time-out, once the jump is done
and the robot has stood at HOME for `HOME_HOLD_STEPS` (`mdp/terminations.py`).
The height terms pay for how far the robot's lowest point -- the least of the
six feet and the base, each against HOME -- rises on a flight of the episode's
own making: pushed from no later than the bottom of the crouch, see
`mdp/commands.py`. So the recording's 0.12 m is not a ceiling, feet flicked up
over a base that stayed put are not height, and an episode spawned mid-air
cannot collect a jump it did not make.

**The two are cross-faded, not added.** Until `PRIOR_HOLD_ITERATIONS` the prior
is the only thing that pays for moving and both height terms weigh nothing; then
the prior falls to zero by `PRIOR_END_ITERATIONS` while they rise to their full
weight. Paid from the start, as in the first run (`2026-09-26_18-05-48`, 4096
envs, RTX 5090 D, warp:cuda), height was pulling the policy off the recording by
iteration 75: from there to 141 that run's prior -- then on the commanded
targets -- fell 0.62 -> 0.53 while `apex_height` rose 0.16 -> 0.67. Once it can jump at all, a landed apex pays
about 1.2 x 4 per step, more than following the prior perfectly does. What
is learnt first has to be the recording's technique; height is only worth asking
for once the policy has it.

## Kept from jumper.jump, and why

200 Hz and kp=20. The recording was made there, and its commands reproduce its
jump only there: fed through `jumper.jump` with a zero residual they rise 0.133 m
against the recording's 0.141, while at kp=10 and 50 Hz they do not leave the
ground (measured on the `hexa.jump2` branch). A feedforward of commands that do
not jump would start the policy somewhere other than this jump.

## Before it goes on the robot

The actor used to watch a go clock, `jump_phase`, which the controller can build
only from a contract's `reference` block -- which this task deliberately does not
ship, so `scripts/export.py` refused it. The actor has no clock now; the jump
starts when the policy does, and the export goes through: `model_5999` of
`2026-09-28_11-54-00` is in `out/example/`: 406 observations of six
terms the controller can build and the robot can measure, and the ONNX within
7e-7 of the torch policy.

**What the robot must do instead is stop it.** Training ends every episode once
the robot is back at HOME, so the policy has never been asked to stay there:
left running at HOME it sees the state it jumps from. The controller has to
leave the jump mode after the landing.

**Nor is every checkpoint a candidate.** One from before `PRIOR_END_ITERATIONS`
still leans on the feedforward, which the robot will not have.

## On the numbers below

The physics and the hardware facts are measured, or taken from where they are.
**The reward weights, the prior's schedule and the RSI share started as
guesses.** Where one has since been set against a trained policy, the note
beside it names the checkpoint and how it was measured; the rest still say only
what they were set against.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from ..common.constants import (
    ACTUATOR_FORCE_NOISE,
    EFFORT_LIMIT,
    FOOT_GEOMS,
    GAIT_JOINTS,
    IMU_ANG_VEL_NOISE,
    IMU_GRAVITY_NOISE,
    JUMPER_FOOT_FRICTION_RANGE,
    PUSH_VELOCITY_RANGE,
    get_jumper_robot_cfg,
)
from ..common.mdp.observations import actuator_force
from ..common.tof import tof_sensor
from ..jump import mdp as jump_mdp
from ..posture.mdp.history import StridedHistory
from . import mdp as ref_free_mdp
from .rl_cfg import NUM_STEPS_PER_ENV

# ── Physics: jumper.jump's, for the prior's sake ───────────────────────────────
#: 200 Hz control. The recording's push lasts about 50 ms: ten control steps
#: here, two or three at 50 Hz.
_SIM_TIMESTEP = 0.0025
_DECIMATION = 2

#: The gain the recording was made at. See the module docstring: fed forward, its
#: commands are *this* jump only at the gain that produced them. The damping
#: stays the hardware's; the torque-speed curve is the servo's.
_KP = 20.0

# ── The action ─────────────────────────────────────────────────────────────────
#: Absolute targets around HOME. The recording goes up to 1.8 rad from HOME -- the
#: middle and rear knees at the push -- which the policy must command itself by
#: the end: at the walking tasks' 0.25 an output of 7.2, at 0.5 one of 3.6. `rl_cfg.py` halves the initial std to match,
#: so exploration stays 0.15 rad, as in jumper.jump.
ACTION_SCALE = 0.5

# ── The observation's history ──────────────────────────────────────────────────
#: Frames of history, and the control steps between them: five frames 20 ms
#: apart, reaching 80 ms back. That is the window `OBS_HISTORY = 5` gives the
#: walking tasks at 50 Hz, kept at 200 Hz by striding rather than by widening, as
#: jumper.posture does -- `posture/mdp/history.py` has the argument. Five
#: consecutive frames here would reach back 20 ms.
OBS_HISTORY = 5
OBS_HISTORY_STRIDE = 4

#: The terms that carry it, in every group: the walking tasks' per-joint set
#: (`HISTORY_TERMS` in `common/velocity_env.py`, where `actuator_force`'s history
#: measured as the largest contribution). The actor has no linear velocity, no
#: contact and no preview of the recording, so the takeoff and the touchdown are
#: things it has to infer, and transients are what history is for. Chosen by that
#: analogy; not measured on this task.
HISTORY_TERMS = frozenset({"joint_pos", "joint_vel", "actions", "actuator_force"})

# ── The go ─────────────────────────────────────────────────────────────────────
#: Seconds from a HOME spawn to the go: none. The policy jumps when it starts to
#: run, so the go is the episode's first step. The command keeps the go instant
#: only for the feedforward, the prior and the height bookkeeping to be read
#: against; the actor does not see it. It was jumper.jump's (0.2, 1.0), and the
#: wait taught `model_1700` of `2026-09-27_18-35-23` to move before the go -- a
#: summed square action change of 5.7 a step there, against 0.001 after its
#: landing -- and `model_3200` of `2026-09-28_08-28-59` to wait 0.23 rad from
#: HOME at 38 (rad/s)^2.
GO_DELAY_RANGE = (0.0, 0.0)

#: Share of episodes spawned on the recording, and the phases they spawn at.
#: Half, so that half start where the robot will: at HOME, the jump beginning.
#: Phases up to 0.9 cover the crouch, the push, the flight and the landing.
RSI_FRACTION = 0.5
RSI_PHASE_RANGE = (0.0, 0.9)

# ── The prior ──────────────────────────────────────────────────────────────────
#: The two joint-tracking terms while they hold. jumper.jump's weights and kernel
#: widths for the same two quantities -- `ref_joint_pos` 3.0 at 0.20 rad^2 over
#: the mean squared angle error, `ref_joint_vel` 1.0 at 40 (rad/s)^2 over the
#: speed error -- which is also the budget of 4 the first design's prior had.
PRIOR_POS_WEIGHT = 3.0
PRIOR_POS_SIGMA_SQ = 0.20
PRIOR_VEL_WEIGHT = 1.0
PRIOR_VEL_SIGMA_SQ = 40.0

#: Iterations at full weight and full feedforward, and the iteration both reach
#: zero. One window for all of it: the recording hands over to the objective at
#: once, in the action and in the reward.
PRIOR_HOLD_ITERATIONS = 1000
PRIOR_END_ITERATIONS = 3000

# ── The objective ──────────────────────────────────────────────────────────────
#: Metres the robot's lowest point rises per unit of height reward: 0.10 m above
#: HOME pays 1. A unit rather than a target -- both height terms are linear and
#: uncapped.
HEIGHT_SCALE = 0.10

#: The latched apex, paid every step after the landing, and the height in flight.
#: The apex is the objective; the flight term is its dense half. A jump as high as
#: the recording's pays the apex about 1.2 x 4 per step for the second or more
#: that follows it, against the 2 per step standing still earns.
#:
#: **These are the weights they end at**, from `PRIOR_END_ITERATIONS` on. Both are
#: 0 until `PRIOR_HOLD_ITERATIONS` and rise linearly between -- the prior's
#: schedule, mirrored. See the module docstring for why.
APEX_WEIGHT = 4.0
FLIGHT_WEIGHT = 2.0

#: Standing at HOME once the jump is done, and upright throughout.
#:
#: 0.01 was tried for `HOME_SIGMA_SQ` and taken back. The stance it was meant to
#: fix was a mis-measurement -- a probe that held the step counter still, which
#: freezes every history term -- and far from HOME a kernel that tight stops
#: pulling: `model_3200` of `2026-09-28_08-28-59`, trained at 0.01, waited for the
#: go 0.23 rad from HOME, where it pays 0.006 and has next to no gradient.
HOME_WEIGHT = 1.0
HOME_SIGMA_SQ = 0.05

UPRIGHT_WEIGHT = 1.0
UPRIGHT_SIGMA_SQ = 0.10

#: Horizontal distance from the spawn, charged as (d / 0.10 m)^2: 10 cm of
#: drift costs what upright pays.
DRIFT_WEIGHT = -1.0
DRIFT_SCALE = 0.10

#: Every step anything but a foot is on the ground. About a step of what the
#: jump pays -- the apex alone is 4 x 1.2 a step once landed, for a jump as high
#: as the recording's -- so that no contact is worth what it buys. Measured before it existed, on
#: `model_6150` of `2026-09-27_15-01-47` (trained on 4096 envs, RTX 5090 D,
#: warp:cuda; measured on 64 envs, native:cpu, HOME spawns, DR on, feedforward 0):
#: some part other than a foot was down in 27 to 39% of steps in every phase,
#: standing at HOME before the go included -- the middle calves and the front
#: palms, mostly. The term it replaced was the share of the 35 parts touching,
#: at -1, which charged a single calf 0.03 a step.
#:
#: **It departs from the recording, and knowingly.** Fed forward on this model
#: the recording itself has a calf or a palm down in 57% of the push's steps and
#: 10% of the landing's, so from the start the policy is charged for part of the
#: recording's own push and has to find one that keeps to its feet.
ILLEGAL_CONTACT_WEIGHT = -5.0

#: Every step two legs push on each other, over 1 N in any physics substep: what
#: `illegal_contact` costs, for the same reason. Measured on 64 envs, native:cpu,
#: HOME spawns, DR and observation noise on, the policy's mean: `model_7400` of
#: `2026-09-28_08-08-34`, trained before this existed, had legs touching in 26% of
#: the steps before the go, rising to 97% of the landing's -- the front upper
#: arms on the middle hips, mostly; `model_3200` of `2026-09-28_08-28-59`, trained
#: with it, in 0.4% of steps at most. The recording, fed forward, has none in any
#: phase, so this asks nothing of the prior.
LEG_COLLISION_WEIGHT = -5.0

#: `action_rate_l2`, per squared radian a joint target moves in one step. mjlab's
#: term charges the raw action, and a target moved by dq is dq / `ACTION_SCALE`
#: of it, so the weight is this times the scale squared: a cost of motion, not of
#: a number.
#:
#: jumper.jump's realised value -- its -0.0144 at scale 0.25 -- the cost a jump has
#: been trained at on this robot, at 200 Hz and kp=20. The weight this replaced,
#: -0.01 * (0.3 / scale)^2, was meant to keep kk-rl-lab's -0.01 at scale 0.3 per
#: radian (0.111) and turned the compensation the wrong way: here it charged
#: 0.0144, a 7.7th of that. jumper.jump's own is turned the same way, which is
#: why its realised cost is 2.07x the lab's.
#:
#: What it had been charging too little for, on `model_6150` of
#: `2026-09-27_15-01-47`, measured on 64 envs, native:cpu, HOME spawns, DR on,
#: feedforward 0: the policy's mean changed its raw action by a summed square of
#: 68 a step -- 0.9 rad of target per joint, RMS -- and by 93 standing before the
#: go; by 56 with the observation noise off, so the chatter is the policy's own.
#: Its mean output was 2.5, 1.2 rad of target from HOME, where 0.087 rad of error
#: saturates the torque plateau: the targets were a switch for the torque's sign.
#: With its std of 1.96 on top, training paid 247 a step. At the old weight those
#: cost 0.24 and 0.89 a second, against the 4.4 apex_height paid then, on the
#: base's height; at this one 3.9 and
#: 14.2. The recording's own command, at 200 Hz, sums to 30 over the whole
#: episode, 0.002 a second here: what this charges is chatter, not the jump.
ACTION_RATE_PER_RAD_SQ = -0.2304

#: Tilt at which an episode ends: the controller's own. `deploy/jumper.controller.toml`
#: drops to `safe` at 50 degrees in every mode, the jump's included, so a policy
#: that learnt to recover from 60 would be cut off on the robot at 50.
FELL_OVER_RAD = math.radians(50.0)

#: Back at HOME, which ends the episode: the jump is done, the 20 driven joints
#: are within `HOME_TOLERANCE` rad RMS of HOME and moving at under
#: `HOME_SPEED_SQ` (rad/s)^2, and have been for `HOME_HOLD_STEPS` in a row -- 0.1
#: s, so that swinging through HOME is not arriving there. HOME held sits 0.012
#: rad off it. Measured before it existed, on `model_3200` of
#: `2026-09-28_08-28-59` (64 HOME spawns, native:cpu, DR and observation noise
#: on, no pushes, its mean): all 64 were back 1.25 s after the go on average, 0.71
#: at the earliest and 1.84 at the latest; without the 0.1 s hold, 1.14.
HOME_TOLERANCE = 0.05
HOME_SPEED_SQ = 0.5
HOME_HOLD_STEPS = 20

#: The longest an episode runs: the jump, the landing and the way home, with
#: margin on that policy's 1.84 s. One that has not come home by then times out.
EPISODE_S = 2.5

#: Every collision geom, by name: exactly the `*_meshcol` ones on this model. The
#: illegal-contact sensor takes these minus the feet -- by exclusion, because
#: listing the links covered the same 35 parts today and would have missed one
#: added tomorrow without a word.
_COLLISION_GEOMS = r".*_meshcol"


def _stride_history(cfg: ManagerBasedRlEnvCfg) -> None:
    """Give every group's `HISTORY_TERMS` `OBS_HISTORY` frames, `OBS_HISTORY_STRIDE`
    control steps apart, with jumper.posture's `StridedHistory`.

    jumper.posture's handover, and the three parts of it that each fail silently:

    - the manager's `history_length` stays **0**, or the frames are stacked twice;
    - the term's noise moves **into** the wrapper, which applies it once, as a
      frame is written. Left to the manager, every stored frame is re-noised each
      step and the policy can average away an error no robot can. It moves only
      where the group is corrupted, since clearing a term's noise is how the
      manager honours `enable_corruption = False`;
    - `common/mdp/symmetry.py` and `scripts/export.py` read the count and the
      stride off the wrapper (`history_frames`, `history_stride`), because the
      config's `history_length` says 0.

    The critic's terms too: the critic is mirrored as well, and a value function
    fitted on another window than the actor's is fitted on another observation.

    **Each group's term is copied before it is wrapped.** mjlab's skeleton builds
    the critic as `{**actor_terms, ...}`, so `joint_vel` and `actions` are one
    object in both groups. Unwrapped that is harmless, the manager clearing the
    noise per group; wrapped, the first group's noise travels in the shared params
    and the critic -- uncorrupted by design -- reads the actor's +/-1.5 rad/s,
    as jumper.posture's did until MR !70 gave it the same copy.
    """
    p = StridedHistory.PREFIX
    for group_name, group in cfg.observations.items():
        if group.history_length:
            raise ValueError(
                f"{group_name}: a group-level history_length overwrites every "
                f"term's, and would stack these frames a second time"
            )
        missing = HISTORY_TERMS - set(group.terms)
        if missing:
            raise ValueError(f"{group_name} has no {sorted(missing)} to give history to")
        for name in HISTORY_TERMS:
            term = group.terms[name] = copy.copy(group.terms[name])
            assert not term.history_length, f"{group_name}/{name} is already stacked"
            assert not any(k.startswith(p) for k in term.params or {}), (
                f"{group_name}/{name} has a param colliding with {p!r}"
            )
            term.params = {
                **(term.params or {}),
                p + "func": term.func,
                p + "frames": OBS_HISTORY,
                p + "stride": OBS_HISTORY_STRIDE,
                p + "noise": term.noise if group.enable_corruption else None,
            }
            term.func = StridedHistory
            term.noise = None


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Reference-free high jump.

    Args:
        asset: path to the model XML, from `--model`. None uses the default jumper.xml.
        play: replay mode -- noise and pushes off, and every episode starts at HOME
            and jumps on the go, as the deployment will.
    """
    cfg = make_velocity_env_cfg()

    # ── Sim ───────────────────────────────────────────────────────────────────
    cfg.sim.mujoco.timestep = _SIM_TIMESTEP
    cfg.decimation = _DECIMATION
    # The velocity skeleton's contact budget for this robot, as measured there.
    cfg.sim.njmax = 512
    cfg.sim.nconmax = 128
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 128

    # ── Scene ─────────────────────────────────────────────────────────────────
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    robot = get_jumper_robot_cfg(effort_limit=EFFORT_LIMIT, asset=asset)
    for actuator in robot.articulation.actuators:
        actuator.stiffness = _KP
    cfg.scene.entities = {"robot": robot}

    # Neither of the skeleton's scans is read by anything here; the two contact
    # sets are what the flight bookkeeping and the penalties read.
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ())
        if s.name not in ("terrain_scan", "foot_height_scan")
    ) + (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            # Read by nothing here. It is what marks a contact sensor as the
            # feet's for `play.py --measure` (`mjrl/viewer/monitor.py`), which
            # without it replays with no foot-force panel and says nothing.
            track_air_time=True,
        ),
        # Leg against leg: the walking tasks' sensor. The whole robot on both
        # sides, and the collision masks leave only different legs able to touch.
        # One entry per physics substep, so a knock between control steps counts.
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=_DECIMATION,
        ),
        ContactSensorCfg(
            name="body_ground_contact",
            primary=ContactMatch(mode="geom", pattern=_COLLISION_GEOMS, entity="robot",
                                 exclude=FOOT_GEOMS),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
            num_slots=1,
        ),
    )

    # ── Observations ──────────────────────────────────────────────────────────
    for group in cfg.observations.values():
        for name in ("command", "height_scan", "foot_height", "foot_air_time",
                     "foot_contact", "foot_contact_forces"):
            group.terms.pop(name, None)
    # No state estimator on the robot, so no linear velocity for the actor; the
    # critic keeps it.
    del cfg.observations["actor"].terms["base_lin_vel"]

    # The 20 driven joints, as everywhere else; the grippers are held at HOME.
    gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))
    for group in cfg.observations.values():
        for name in ("joint_pos", "joint_vel"):
            term = group.terms.get(name)
            if term is not None:
                term.params = dict(term.params or {}) | {"asset_cfg": gait_cfg}

    # This robot's IMU, from `constants.py`, not the Go1's the skeleton carries.
    for group in cfg.observations.values():
        if "base_ang_vel" in group.terms:
            group.terms["base_ang_vel"].noise = Unoise(
                n_min=-IMU_ANG_VEL_NOISE, n_max=IMU_ANG_VEL_NOISE
            )
        if "projected_gravity" in group.terms:
            group.terms["projected_gravity"].noise = Unoise(
                n_min=-IMU_GRAVITY_NOISE, n_max=IMU_GRAVITY_NOISE
            )

    # The current sensor, as the walking tasks read it: the 20 driven actuators,
    # with this robot's noise. On the ground the legs carry the body and in the
    # air only themselves -- the difference the takeoff and the touchdown make.
    force_cfg = SceneEntityCfg("robot", actuator_names=list(GAIT_JOINTS))
    for group in cfg.observations.values():
        group.terms["actuator_force"] = ObservationTermCfg(
            func=actuator_force,
            params={"asset_cfg": force_cfg},
            noise=Unoise(n_min=-ACTUATOR_FORCE_NOISE, n_max=ACTUATOR_FORCE_NOISE),
        )

    # The go clock, for the critic alone: the recording's progress since the go,
    # which the prior, the feedforward and the height bookkeeping are read
    # against. The actor jumps when it starts to run and needs no clock -- and a
    # clock is what the robot could not supply without the recording.
    cfg.observations["critic"].terms["jump_phase"] = ObservationTermCfg(
        func=jump_mdp.observations.jump_phase, params={"command_name": "jump"}
    )
    # And for the critic alone, what the rewards are computed from.
    cfg.observations["critic"].terms["jump_state"] = ObservationTermCfg(
        func=ref_free_mdp.observations.jump_state,
        params={"command_name": "jump", "scale": HEIGHT_SCALE, "mirror_kind": "scalar"},
    )
    cfg.observations["critic"].terms["prior_target"] = ObservationTermCfg(
        func=ref_free_mdp.observations.prior_target,
        params={"command_name": "jump", "mirror_kind": "joint"},
    )
    cfg.observations["critic"].terms["feedforward"] = ObservationTermCfg(
        func=ref_free_mdp.observations.feedforward, params={"mirror_kind": "scalar"},
    )

    # ── Action: absolute targets around HOME, the recording fed forward ───────
    cfg.actions["joint_pos"] = ref_free_mdp.actions.PriorFeedforwardJointPositionActionCfg(
        entity_name="robot",
        actuator_names=list(GAIT_JOINTS),
        scale={j: ACTION_SCALE for j in GAIT_JOINTS},
        use_default_offset=True,
        command_name="jump",
        hold_step=NUM_STEPS_PER_ENV * PRIOR_HOLD_ITERATIONS,
        end_step=NUM_STEPS_PER_ENV * PRIOR_END_ITERATIONS,
    )

    # ── Command: the go ───────────────────────────────────────────────────────
    cfg.commands = {
        "jump": ref_free_mdp.commands.RefFreeJumpCommandCfg(
            entity_name="robot",
            motion_file=str(jump_mdp.reference.JUMP_REF_NPZ),
            command_delay_range=GO_DELAY_RANGE,
            rsi_fraction=RSI_FRACTION,
            rsi_phase_range=RSI_PHASE_RANGE,
            home_tolerance=HOME_TOLERANCE,
            home_speed_sq=HOME_SPEED_SQ,
        )
    }

    # ── Rewards ───────────────────────────────────────────────────────────────
    # The four scheduled weights are written as they stand at iteration 0; the
    # curriculum below moves them.
    cfg.rewards = {
        "prior_joint_pos": RewardTermCfg(
            func=ref_free_mdp.rewards.prior_joint_pos, weight=PRIOR_POS_WEIGHT,
            params={"asset_cfg": gait_cfg, "command_name": "jump",
                    "sigma_sq": PRIOR_POS_SIGMA_SQ},
        ),
        "prior_joint_vel": RewardTermCfg(
            func=ref_free_mdp.rewards.prior_joint_vel, weight=PRIOR_VEL_WEIGHT,
            params={"asset_cfg": gait_cfg, "command_name": "jump",
                    "sigma_sq": PRIOR_VEL_SIGMA_SQ},
        ),
        "apex_height": RewardTermCfg(
            func=ref_free_mdp.rewards.apex_height, weight=0.0,
            params={"command_name": "jump", "scale": HEIGHT_SCALE},
        ),
        "flight_height": RewardTermCfg(
            func=ref_free_mdp.rewards.flight_height, weight=0.0,
            params={"command_name": "jump", "scale": HEIGHT_SCALE},
        ),
        "home_pose": RewardTermCfg(
            func=ref_free_mdp.rewards.home_pose, weight=HOME_WEIGHT,
            params={"command_name": "jump", "sigma_sq": HOME_SIGMA_SQ,
                    "asset_cfg": gait_cfg},
        ),
        "upright": RewardTermCfg(
            func=ref_free_mdp.rewards.upright, weight=UPRIGHT_WEIGHT,
            params={"sigma_sq": UPRIGHT_SIGMA_SQ},
        ),
        "horizontal_drift": RewardTermCfg(
            func=ref_free_mdp.rewards.horizontal_drift, weight=DRIFT_WEIGHT,
            params={"scale": DRIFT_SCALE},
        ),
        "illegal_contact": RewardTermCfg(
            func=ref_free_mdp.rewards.illegal_contact, weight=ILLEGAL_CONTACT_WEIGHT,
            params={"sensor_name": "body_ground_contact", "force_threshold": 1.0},
        ),
        "leg_collision": RewardTermCfg(
            func=ref_free_mdp.rewards.leg_collision, weight=LEG_COLLISION_WEIGHT,
            params={"sensor_name": "self_collision", "force_threshold": 1.0},
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        # Per radian of target, so times scale^2: see ACTION_RATE_PER_RAD_SQ.
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2, weight=ACTION_RATE_PER_RAD_SQ * ACTION_SCALE**2
        ),
    }

    # ── Curriculum: the prior comes down as the height goes up ────────────────
    def _cross_fade(term: str, start: float, end: float) -> CurriculumTermCfg:
        return CurriculumTermCfg(
            func=ref_free_mdp.curriculum.anneal_reward_weight,
            params={
                "term_name": term,
                "start_weight": start,
                "end_weight": end,
                "start_step": NUM_STEPS_PER_ENV * PRIOR_HOLD_ITERATIONS,
                "end_step": NUM_STEPS_PER_ENV * PRIOR_END_ITERATIONS,
            },
        )

    cfg.curriculum = {
        "prior_joint_pos_weight": _cross_fade("prior_joint_pos", PRIOR_POS_WEIGHT, 0.0),
        "prior_joint_vel_weight": _cross_fade("prior_joint_vel", PRIOR_VEL_WEIGHT, 0.0),
        "apex_height_weight": _cross_fade("apex_height", 0.0, APEX_WEIGHT),
        "flight_height_weight": _cross_fade("flight_height", 0.0, FLIGHT_WEIGHT),
        # Not a schedule of its own -- the action term reads the feedforward off
        # the step counter -- only its value in the log, beside the weights.
        "feedforward": CurriculumTermCfg(func=ref_free_mdp.curriculum.report_feedforward),
    }

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        # A time-out, not a failure: see `mdp/terminations.py`.
        "back_home": TerminationTermCfg(
            func=ref_free_mdp.terminations.back_home, time_out=True,
            params={"command_name": "jump", "hold_steps": HOME_HOLD_STEPS},
        ),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": FELL_OVER_RAD}
        ),
    }

    # ── Events ────────────────────────────────────────────────────────────────
    # The robot's own ranges, from `constants.py`, as `common/velocity_env.py`
    # sets them for the walking tasks.
    cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
        "robot", geom_names=FOOT_GEOMS
    )
    cfg.events["foot_friction"].params["ranges"] = JUMPER_FOOT_FRICTION_RANGE
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
    cfg.events["push_robot"].params["velocity_range"].update(PUSH_VELOCITY_RANGE)
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    cfg.events = {
        # HOME first, then RSI overwrites the share it spawns on the recording.
        "reset_to_default": EventTermCfg(
            func=envs_mdp.reset_scene_to_default, mode="reset"
        ),
        "reset_from_reference": EventTermCfg(
            func=jump_mdp.events.reset_from_reference_phase,
            mode="reset",
            params={"command_name": "jump", "rsi_fraction": RSI_FRACTION,
                    "phase_range": RSI_PHASE_RANGE},
        ),
        **cfg.events,
    }

    # ── Episode and viewer ────────────────────────────────────────────────────
    cfg.episode_length_s = EPISODE_S
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 0.9
    cfg.viewer.elevation = -15.0

    if play:
        # Every episode as the deployment will run it: from HOME, the jump at
        # once, and home again. Each ends there, so replay loops whole attempts.
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.events["reset_from_reference"].params["rsi_fraction"] = 0.0
        cfg.commands["jump"].rsi_fraction = 0.0
        cfg.curriculum = {}

    # The dToF, which the model carries (`common/tof.py`). Replay only, where the
    # live viewer shows it in the window's corner; nothing in training reads it,
    # and there it would be 2268 rays per environment per step spent on nothing.
    if play:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

    # Last, after `play`: the history takes each term's noise with it, and whether
    # there is any to take is the group's `enable_corruption`.
    _stride_history(cfg)
    return cfg


__all__ = ["env_cfg"]
