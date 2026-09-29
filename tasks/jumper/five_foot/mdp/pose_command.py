"""A second command next to the velocity twist: how the trunk holds itself.

Three numbers, sampled per command window and held: **pitch**, **roll** and a
**twist** the trunk keeps over its own stationary footholds. Ported from
kk-rl-lab's `tasks/hexa_5d/mdp/commands/body_pose_command.py`, which is where this
robot's attitude command was designed and measured.

## Why the machine needs one

It carries a claw. A robot that can only walk level can only reach what is already
at its own height and angle; aiming is done by leaning, and nothing in this task
has ever asked this robot to lean. The velocity command says where to go, this one
says how to arrive.

## What each axis is

    index  name    axis                       neutral
      0    pitch   nose up / nose down        level
      1    roll    lean left / lean right     level
      2    twist   trunk yaw **over its own footholds**   square

`twist` is not heading. Heading is what `ang_vel_z` commands, and a turn moves the
whole robot, feet included. Twist is the *offset* between the trunk and the feet it
is standing on -- a shape rather than a direction, bounded rather than continuous,
and the one attitude on this machine that costs nothing mechanically: the feet stay
planted and the legs only shear.

**Three, not four.** kk-rl-lab's version carries stand height as a fourth channel.
It is left out here because height already has a term in this task (`low_stance`,
against `STAND_Z`) and because every channel added to this command is a channel the
on-robot controller in `deploy/fsm/src/obs.rs` has to learn to fill. The layout
puts the three angles first for the same reason theirs does: a fourth can be
appended without moving an index anything reads positionally.

## Ranges depend on whether the robot is walking

Standing still, the legs are not swinging and the trunk can be asked for more --
kk-rl-lab measured and trains +-20 degrees of pitch, +-15 of roll and +-30 of
twist standing, and +-15 in every axis while moving, because a leg that is
mid-swing needs its amplitude back. Those are the bands here.

+-15 in every axis, standing included, was tried for one configuration and put
back: it is one number rather than two, but it also narrows what the machine can
be asked for in exactly the state where the ask is cheapest -- feet planted,
nothing swinging, which is when a claw is aimed. The walking bands are where the
caution belongs, and they already have it.

The velocity command is what decides which band applies, so this term reads it:
`stand_threshold` is a norm on `[vx, vy, wz]`.

## The command ramps

`pose_command_b` -- what the observation carries and the reward terms score --
moves toward the sampled value at `max_rate`, 30 degrees per second, rather than
jumping to it. **This is not smoothing for its own sake; a step command charges
the policy for a target it cannot reach.** The attitude terms are linear plus
quadratic in the error, so the instant after a resample a 30 degree step costs
`nose_pitch` 51 per step at -15, or 5.1 at -1.5, against a whole velocity
tracking budget of 4.0 -- for a trunk that physically cannot teleport. Measured
on `2026-09-10_14-50-24`, which had the step: 3 degrees of steady pitch error and
0.433 of linear tracking, against the 0.633 the same reward set reaches with no
attitude command at all.

A ramped command also makes the three observation channels continuous, which is
what the rest of this observation looks like -- gravity, joint angles, the
velocity command's own smooth resample. 30 deg/s crosses the widest band in 1.3
seconds, so the command still spends most of its 5 second window held.

## The command is observed, never paid for

Every term in `env_cfg.py` that scores an axis of this command is a **penalty on
the error**, and that is not a style choice. A robot standing still holds any
attitude perfectly, so an exponential attitude reward pays a motionless machine
for doing nothing -- 4.75 per step against tracking's 4.0, measured, twice, and
both cold starts sat down and collected it. A penalty cannot be collected by
standing.

The mirror of that: an attitude penalty measured against a command the policy
**cannot see** is noise it is punished for not obeying. Wiring this command in
means adding it to the observation, which grows the actor's input by three and
makes every checkpoint trained before it unloadable.

## Replay: the operator flies it

`TeleopBodyPoseCommandCfg` hands the command to the operator of `controls.yaml`,
the same one that drives the walk, and for the reason every teleop here gives:
during training there is no viewer for a key to reach, and a term drivable from
outside the process makes a run unreproducible.

    pad                                               keyboard
    right stick     up/down: nose down/up             I K
                    left/right: twist, then turn      J L turn; no twist
    R3 held         left/right rolls instead          U O roll
    B               hand both commands back           B
    (hands off)     level and square, standing still  (let go)

The keyboard has no twist here: Shift, posture's twist modifier, holds the arm
out in this task (Control-agent 3.1's layout, asked for on 2026-09-29).

**The pad's layout is `jumper.posture`'s, binding for binding.** The two tasks' pose
commands reach the robot on one set of channels, `Command::base_pitch`,
`base_roll` and `base_twist`, so a stick that tipped the nose one way in one mode
and the other way in the next would be one control with two meanings.
`tests/test_five_foot_pose_command.py` fails when the two files part.

**Full deflection is the standing edge, and walking narrows it.** A stick pushed
all the way asks for the edge of `cfg.ranges`, the standing bands; while the
velocity command walks, `hold_to_band` holds the target to the moving bands, and
the ramp carries the observed command there at `max_rate`. So pushing the right
stick past half travel into a turn brings the twist back from 30 degrees to 15 --
a turning robot is a moving one -- and on to zero at three quarters, where the
turn is at half its top rate and the stick's own mapping has unwound it
(`controls.yaml`). The controller does the same on the robot and in a browser,
from the bands and the rate the contract carries.

The attitude used to be on the numeric keypad, kk-rl-lab's layout key for key --
`KP 8` / `KP 2` pitch, `KP 4` / `KP 6` roll, `KP 7` / `KP 9` twist, `KP 5` level,
`KP 0` hand back -- which drove it in replay and nowhere else: `controls.yaml`
said nothing about it, so on the robot `base_pose` sat at level and square
whatever the operator did. The signs did not change with the move. `pitch+` is
nose down and `roll+` is the right side down, as `base_pitch_roll` measures them,
and a stick forward or right is `+` on both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from ...common.mdp.controls import Controls
from ...common.mdp.operator import connect

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "PITCH",
    "POSE_DIM",
    "ROLL",
    "TWIST",
    "BodyPoseCommand",
    "BodyPoseCommandCfg",
    "TeleopBodyPoseCommand",
    "TeleopBodyPoseCommandCfg",
    "base_pitch_roll",
]

#: Positions in the command vector. The reward terms index it positionally, so
#: these are the only place the layout is written down.
PITCH = 0
ROLL = 1
TWIST = 2
POSE_DIM = 3

_DEG = math.pi / 180.0


def base_pitch_roll(
    projected_gravity_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trunk pitch and roll in radians, from the gravity direction in the base frame.

        pitch = asin(g_x)          roll = atan2(-g_y, -g_z)

    Derived rather than asserted: for a body at `Rz(yaw) Ry(pitch) Rx(roll)` the
    unit gravity direction expressed in the base frame is

        g = [sin p, -cos p sin r, -cos p cos r]

    so the two formulas above invert it exactly for `|pitch| < 90 degrees`. Sign
    convention: `g_x > 0` is nose-down, and positive roll is the right side down.

    **Yaw-independent, and that is the point.** Both are functions of the IMU
    gravity vector alone, so the robot can compute them from what it actually has;
    anything read off the root quaternion would be a simulator-only quantity and
    would not survive `scripts/export.py`.
    """
    g = projected_gravity_b
    pitch = torch.asin(g[:, 0].clamp(-1.0, 1.0))
    roll = torch.atan2(-g[:, 1], -g[:, 2])
    return pitch, roll


@dataclass(kw_only=True)
class BodyPoseCommandCfg(CommandTermCfg):
    """The sampled attitude command. Defaults are kk-rl-lab's measured bands."""

    entity_name: str = "robot"

    velocity_command_name: str = "twist"
    """Which command decides between the standing and moving bands. Reading it
    rather than the robot's speed is deliberate: the bands protect *swing
    amplitude*, and what the legs are about to be asked for is the command, not
    the velocity the trunk happens to have this step."""

    stand_threshold: float = 0.06
    """Norm of `[vx, vy, wz]` below which the standing bands apply."""

    max_rate: float | None = 30.0 * _DEG
    """How fast the *observed* command may move, rad/s. See "The command ramps".

    30 deg/s crosses the widest band, 40 degrees of standing pitch, in 1.3
    seconds -- comfortably inside the 5 second command window, so a ramped command
    still spends most of its window held.

    **None is the reference implementation's behaviour**: kk-rl-lab steps, with no
    ramp anywhere in `body_pose_command.py`, and copes with the transient by
    scoring the axes with a plain `err^2` -- no deadzone, no linear term -- at -15
    and -20. That is a coherent alternative to this and the two are one parameter
    apart."""

    rel_neutral_envs: float = 0.25
    """Fraction of environments pinned to level and square. Without it every
    environment is always being asked for *some* lean, and the policy never sees
    that holding the neutral pose is also a thing it is asked to do -- which is
    the pose it spends most of its life in on the robot."""

    stand_pitch: tuple[float, float] = (-20.0 * _DEG, 20.0 * _DEG)
    stand_roll: tuple[float, float] = (-15.0 * _DEG, 15.0 * _DEG)
    stand_twist: tuple[float, float] = (-30.0 * _DEG, 30.0 * _DEG)
    """Standing bands. Wider, because nothing is mid-swing."""

    move_pitch: tuple[float, float] = (-15.0 * _DEG, 15.0 * _DEG)
    move_roll: tuple[float, float] = (-15.0 * _DEG, 15.0 * _DEG)
    move_twist: tuple[float, float] = (-15.0 * _DEG, 15.0 * _DEG)
    """Walking bands. Twist narrows most, 30 to 15 degrees -- it is the axis that
    shears the legs sideways out of their swing plane, and 30 degrees of it over a
    moving gait is the one this robot cannot afford. Pitch narrows 20 to 15, and
    roll never was the binding axis and stays at 15. (This used to say pitch and
    roll were the same in both bands; the numbers above were already 20 and 15.)"""

    @dataclass
    class Band:
        """One band's three axes as `(low, high)` in radians -- the shape
        `scripts/export.py::_command_ranges` and the operator's `spans` read."""

        pitch: tuple[float, float]
        roll: tuple[float, float]
        twist: tuple[float, float]

    @property
    def ranges(self) -> Band:
        """The standing bands: the widest the command goes, what the contract
        carries as `command_ranges.body_pose`, and what a stick at full deflection
        reaches. A view rather than a field, so the bands stay written once."""
        return BodyPoseCommandCfg.Band(
            pitch=self.stand_pitch, roll=self.stand_roll, twist=self.stand_twist
        )

    @property
    def moving(self) -> Band:
        """The walking bands, the same view: what the command is held to while
        the velocity command is at or above `stand_threshold`."""
        return BodyPoseCommandCfg.Band(
            pitch=self.move_pitch, roll=self.move_roll, twist=self.move_twist
        )

    def build(self, env: ManagerBasedRlEnv) -> BodyPoseCommand:
        return BodyPoseCommand(self, env)


class BodyPoseCommand(CommandTerm):
    """Uniform in each axis, from the band the velocity command selects."""

    cfg: BodyPoseCommandCfg

    def __init__(self, cfg: BodyPoseCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.entity_name]
        self.pose_command_b = torch.zeros(self.num_envs, POSE_DIM, device=self.device)
        # What the sampler (or the operator) asks for. `pose_command_b` is what is
        # observed and scored, and it walks toward this at `max_rate`.
        self.pose_target_b = torch.zeros(self.num_envs, POSE_DIM, device=self.device)
        self.is_neutral_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Pitch and roll only: the twist error needs the footholds, which the
        # reward term already assembles from the contact sensor and the foot
        # sites. Recomputing that here to log it would duplicate the one piece of
        # geometry in this feature that is easy to get wrong.
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_roll"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "BodyPoseCommand:\n"
            f"\tdimension: {POSE_DIM} (pitch, roll, twist)\n"
            f"\tresampling: {self.cfg.resampling_time_range}\n"
            f"\tstanding bands (deg): {self._deg(self.cfg.stand_pitch)} "
            f"{self._deg(self.cfg.stand_roll)} {self._deg(self.cfg.stand_twist)}\n"
            f"\tmoving bands (deg): {self._deg(self.cfg.move_pitch)} "
            f"{self._deg(self.cfg.move_roll)} {self._deg(self.cfg.move_twist)}\n"
            f"\tneutral fraction: {self.cfg.rel_neutral_envs}"
        )

    @staticmethod
    def _deg(band: tuple[float, float]) -> str:
        return f"[{band[0] / _DEG:+.0f},{band[1] / _DEG:+.0f}]"

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        standing = self._standing()[env_ids]
        r = torch.empty(len(env_ids), device=self.device)
        bands = (
            (PITCH, self.cfg.stand_pitch, self.cfg.move_pitch),
            (ROLL, self.cfg.stand_roll, self.cfg.move_roll),
            (TWIST, self.cfg.stand_twist, self.cfg.move_twist),
        )
        for index, stand, move in bands:
            lo = torch.where(standing, stand[0], move[0])
            hi = torch.where(standing, stand[1], move[1])
            self.pose_target_b[env_ids, index] = lo + r.uniform_(0.0, 1.0) * (hi - lo)
        self.is_neutral_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_neutral_envs

    def _standing(self) -> torch.Tensor:
        """Per environment: is the velocity command asking for a standstill?

        **The velocity command may not exist yet.** Command terms are built in
        config order, and a manager that builds this one first would see no
        `twist` at all. Everything standing is the safe answer there -- the wider
        band, for one window, until the first resample after both terms exist.
        """
        try:
            vel = self._env.command_manager.get_command(self.cfg.velocity_command_name)
        except (KeyError, AttributeError):
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return torch.norm(vel[:, :3], dim=1) < self.cfg.stand_threshold

    def hold_to_band(self) -> None:
        """Clamp every walking environment's target to the moving bands.

        For a target that something other than the sampler writes -- the
        operator in replay, whose full deflection is the standing edge. Every
        step rather than at a resample, because a stick moves when it likes, and
        a walking robot handed the standing band has been asked for an attitude
        its swinging legs were never trained to hold. The ramp then carries the
        observed command down to it at `max_rate`.
        """
        walking = ~self._standing()
        if not walking.any():
            return
        for index, (low, high) in (
            (PITCH, self.cfg.move_pitch),
            (ROLL, self.cfg.move_roll),
            (TWIST, self.cfg.move_twist),
        ):
            self.pose_target_b[walking, index] = self.pose_target_b[walking, index].clamp(
                low, high
            )

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids  # a pure function of the sampled state; all envs is safe
        self.pose_target_b[self.is_neutral_env] = 0.0
        # **The ramp.** Measured on `2026-09-10_14-50-24`: a step command left the
        # policy 3 degrees off in pitch and walking at 0.433 of the tracking its
        # own reward set reaches at 0.633 without an attitude command at all.
        if self.cfg.max_rate is None:
            self.pose_command_b[:] = self.pose_target_b
            return
        step = self.cfg.max_rate * self._env.step_dt
        delta = (self.pose_target_b - self.pose_command_b).clamp(-step, step)
        self.pose_command_b += delta

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        """Start each episode **at** the neutral pose, not at the new sample.

        The robot is spawned level and square, so a command that begins anywhere
        else begins with an error the policy did nothing to earn -- the same
        transient the ramp exists to remove, reintroduced once per episode. The
        first window then ramps from level to whatever was sampled.
        """
        extras = super().reset(env_ids)
        assert isinstance(env_ids, torch.Tensor)
        self.pose_command_b[env_ids] = 0.0
        return extras

    def _update_metrics(self) -> None:
        max_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        pitch, roll = base_pitch_roll(self.robot.data.projected_gravity_b)
        self.metrics["error_pitch"] += (
            (self.pose_command_b[:, PITCH] - pitch).abs() / max_step
        )
        self.metrics["error_roll"] += (
            (self.pose_command_b[:, ROLL] - roll).abs() / max_step
        )


@dataclass(kw_only=True)
class TeleopBodyPoseCommandCfg(BodyPoseCommandCfg):
    """A body pose command the operator can take over, from `controls.yaml`.

    A drop-in subclass of the sampling term, so every band, the ramp and the
    resampling clock carry over untouched -- and until somebody touches a
    control, this *is* the sampling term.
    """

    controls: Controls
    """The task's parsed `controls.yaml`, which describes this command."""

    term: str = "body_pose"
    """This term's key in `cfg.commands`, which is how the file names it."""

    pad: bool = True
    """Open a pad if one is connected. Must match the velocity term's: one
    person, one device."""

    pad_name: str | None = None

    def build(self, env: ManagerBasedRlEnv) -> TeleopBodyPoseCommand:
        return TeleopBodyPoseCommand(self, env)


class TeleopBodyPoseCommand(BodyPoseCommand):
    """The sampler until the operator touches a control, the operator after it."""

    cfg: TeleopBodyPoseCommandCfg

    #: `base_pose`'s columns, which is the command's own order.
    LAYOUT = ("pitch", "roll", "twist")

    def __init__(self, cfg: TeleopBodyPoseCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self._operator = connect(cfg, env, self.LAYOUT)
        self._driving = False

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Nothing to draw while the operator drives: the clock still runs, but a
        sample landing under the operator's hands would be one ramp step toward
        a posture nobody asked for, and a neutral flag drawn with it would snap
        a quarter of the field back to level."""
        if not self._driving:
            super()._resample_command(env_ids)

    def compute(
        self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        """Drive the **target**, and let the base class ramp toward it.

        Writing `pose_command_b` directly would hand the policy a step the
        training distribution has never contained -- `max_rate` is part of what
        it learned to track, and an operator who can teleport the command is
        testing something the robot was never asked to do. The controller ramps
        at the same rate on the robot, from the contract.

        Full deflection is the standing edge (`cfg.ranges`); a walking robot is
        held to the moving bands (`hold_to_band`), which is also what the
        controller does on the robot.
        """
        values = self._operator.command(self.cfg.term, self._env.common_step_counter)
        if values is None:
            if self._driving:
                # **Handing back has to undo what taking over did.** The operator
                # drives `pose_target_b` and clears the neutral fraction; both
                # would outlive the release until the next resample, up to five
                # seconds of the sampler appearing not to have taken the term back.
                self._driving = False
                self._resample(torch.arange(self.num_envs, device=self.device))
        else:
            self._driving = True
            # Every environment gets the same attitude, for the reason the
            # velocity term gives: a field where one robot obeys and fifteen
            # wander is harder to read, not richer.
            self.is_neutral_env[:] = False
            self.pose_target_b[:] = torch.tensor(
                values, device=self.device, dtype=torch.float32
            )
            self.hold_to_band()
        super().compute(dt, env_ids)
