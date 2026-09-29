"""The posture command: twist, pitch, roll and height, sampled per environment.

A second command term running alongside `twist`, the velocity one. It is a
separate term rather than four more channels on that one for a reason that is not
tidiness: `twist` is mjlab's `UniformVelocityCommand` and half this repository
reads it by position -- `moving_gate` takes `cmd[:, :3]`, the gait rewards, the
curriculum, the teleop and gamepad subclasses and `scripts/export.py`'s contract
all assume a three-vector. Widening it would have been a silent change to every
one of them.

## What it samples

    twist    body yaw over the feet             uniform over its band, below
    pitch    nose down positive                 uniform over its band
    roll     left side up positive              uniform over its band
    height   base above the ground              uniform over `ranges.height`

Each independently, all four redrawn together on the same clock. Independence is
worth stating because it is not free: a robot asked for the edge of pitch, the
edge of roll and 0.07 m of height at once is being asked for all three corners
at their limit, which is the hardest posture in the space and is drawn about as
often as any other. That is the intent -- the commands are meant to compose --
but if training stalls it is the first thing to look at, and the fix is a
correlated draw or a ranges curriculum rather than a smaller box.

## Two bands for the angles, and they are `jumper.five_foot`'s

`ranges` holds the **standing** half-ranges and `moving` the walking ones, and a
resample draws each angle from whichever band the velocity command is in:
standing when the norm of its `[vx, vy, wz]` is under `stand_threshold`. Standing,
no leg is swinging and the body can be asked for more; a leg in mid-swing needs
its amplitude back. The numbers match `jumper.five_foot`'s on purpose. Both
tasks' pose commands reach the robot on one set of channels, a stick at full
deflection is the standing edge in both, and the controller holds the angles to
`moving` while the velocity command walks -- in replay through `hold_to_band`,
on the robot and in a browser from the contract.

The band is chosen at the resample, as five_foot chooses it: the velocity command
resamples on its own clock, so an environment can draw a standing posture and
start walking before its next one. That blurs the edge rather than moving it, and
the operator's path is the stricter of the two -- it clamps every step.

**The norm decides, not the velocity term's `is_standing_env` flag**, which
`_standing` reads for the neutral rates. The flag exists only where mjlab's
sampler set it; a pad and a robot have the command's value and nothing else, and
one rule on every host is what makes a band mean the same thing on all of them.

## `rel_neutral_envs`, and why a neutral posture has to be over-sampled

A uniform draw over a four-dimensional box essentially never produces the
posture the robot actually holds most of the time: "level, square and at
standing height" is one point, and a continuous draw gives it probability zero.
Left alone, the task would train a robot that can hold any commanded lean and
has never once been asked to simply stand normally -- and `jumper.tripod`'s whole
gait, which this task inherits, was learned at exactly that posture.

So a fraction of environments are pinned to the neutral posture exactly, the same
device `UniformVelocityCommand` uses for `rel_standing_envs` and for the same
reason. The flag is redrawn at every resample, so the fraction of environments is
also the fraction of time.

## Two neutral rates, because the standing case is what this task is for

The rate is **conditional on whether the velocity command has that environment
parked**, and the reason is arithmetic. Drawn independently, "not moving and
holding a commanded posture" gets `rel_standing_envs x (1 - rel_neutral_envs)`,
which at 0.2 and 0.2 is 16% of the time -- and a fifth of the standing time is
spent on the one posture that asks nothing of the robot, which is precisely the
case the rest of the task already covers.

`rel_neutral_standing_envs` separates the two. At 0.0 every parked environment
holds a posture it has to work for; the idle-at-home case then comes only from
the moving draw happening to land near neutral, which is why it is a knob rather
than a hardcoded zero.

## The metrics are the four tracking errors

Logged as `Metrics/posture/error_*`, accumulated per step and normalised by the
longest resampling interval, which is the convention `UniformVelocityCommand`
uses. They are what to read to tell "the policy cannot reach this" apart from
"the policy is not being paid enough to try": the first shows an error that
plateaus above zero, the second shows an error that tracks the weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import wrap_to_pi

from . import state as posture_state

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

#: The command's first three columns, the angles that have a standing and a
#: moving band. The fourth, height, has one range.
ANGLES = ("twist", "pitch", "roll")


class PostureCommand(CommandTerm):
    """Four scalars: (twist, pitch, roll, height). See the module docstring."""

    cfg: "PostureCommandCfg"

    def __init__(self, cfg: "PostureCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        # `preserve_order` is not optional here: without it the resolver is free
        # to collapse the selection to a slice in model order, and the twist
        # estimate aligns the footprint row by row. `state._check_foot_order`
        # catches it either way.
        self.asset_cfg = SceneEntityCfg(
            cfg.entity_name, site_names=cfg.foot_sites, preserve_order=True
        )
        self.asset_cfg.resolve(env.scene)

        self.posture_command = torch.zeros(self.num_envs, 4, device=self.device)
        # Where the **feet** were, in the world, when the robot was parked, and
        # where the footprint was pointing. `_update_command` maintains both;
        # `foot_displacement` and `footprint_rotation` read them.
        self.foot_anchor = torch.zeros(
            self.num_envs, len(cfg.foot_sites), 2, device=self.device
        )
        self.footprint_anchor = torch.zeros(self.num_envs, device=self.device)
        self._robot = env.scene[cfg.entity_name]
        self.is_neutral_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        for name in ("twist", "pitch", "roll", "height"):
            self.metrics[f"error_{name}"] = torch.zeros(
                self.num_envs, device=self.device
            )
        # Not errors, and **not scored by anything**: what the feet did while the
        # body was being asked to turn. The term that charged the first of them
        # was removed on request, so these two are now the only thing in a run
        # that says whether a tracked twist was delivered by the body or by the
        # feet -- which `track_twist` is structurally unable to distinguish.
        for name in ("foot_displacement", "footprint_rotation"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.posture_command

    def measured(self) -> torch.Tensor:
        """What the body is actually doing, in the command's own layout."""
        return posture_state.posture_state(self._env, self.asset_cfg)

    def _update_metrics(self) -> None:
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        error = (self.posture_command - self.measured()).abs() / max_command_step
        for i, name in enumerate(("twist", "pitch", "roll", "height")):
            self.metrics[f"error_{name}"] += error[:, i]
        # The same normalisation, so all six read as an average over the interval.
        self.metrics["foot_displacement"] += (
            self.foot_displacement.mean(dim=1) / max_command_step
        )
        self.metrics["footprint_rotation"] += (
            self.footprint_rotation.abs() / max_command_step
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        r = torch.empty(len(env_ids), device=self.device)
        ranges, moving = self.cfg.ranges, self.cfg.moving
        standing = self._in_standing_band(env_ids)
        for i, name in enumerate(ANGLES):
            half = torch.where(
                standing,
                torch.full_like(r, getattr(ranges, name)),
                torch.full_like(r, getattr(moving, name)),
            )
            self.posture_command[env_ids, i] = (2.0 * r.uniform_(0.0, 1.0) - 1.0) * half
        self.posture_command[env_ids, 3] = r.uniform_(*ranges.height)

        # Which of the two neutral rates applies depends on whether the velocity
        # command has this environment parked **right now**. The two terms
        # resample on their own clocks, so the pairing is not exact: an
        # environment can be handed a posture while moving and be parked a second
        # later, keeping the posture it drew as a moving one. Over a run that
        # blurs the share rather than biasing it, and the alternative -- one term
        # owning both draws -- would mean this task no longer uses mjlab's
        # velocity command at all.
        standing = self._standing(env_ids)
        rate = torch.where(
            standing,
            torch.full_like(r, self.cfg.rel_neutral_standing_envs),
            torch.full_like(r, self.cfg.rel_neutral_envs),
        )
        self.is_neutral_env[env_ids] = r.uniform_(0.0, 1.0) <= rate
        neutral = env_ids[self.is_neutral_env[env_ids]]
        if len(neutral) > 0:
            self.posture_command[neutral, :3] = 0.0
            self.posture_command[neutral, 3] = self.cfg.neutral_height


    def _standing(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Whether the velocity command has each of `env_ids` parked.

        Returns all-False when the velocity term is not there or does not carry
        the flag, which makes `rel_neutral_standing_envs` inert rather than
        making the task fail to build. A task pairing this command with a
        different velocity term gets the documented behaviour of the other rate,
        not a crash halfway through a run.
        """
        try:
            term = self._env.command_manager.get_term(self.cfg.velocity_command)
        except (KeyError, AttributeError):
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        flag = getattr(term, "is_standing_env", None)
        if flag is None:
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        return flag[env_ids]

    def _in_standing_band(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Whether the velocity command is slow enough for the standing band.

        The norm of its `[vx, vy, wz]` under `stand_threshold` -- `jumper.five_foot`'s
        rule, mixed units and all, so that the two tasks and every host agree on
        it. All True while the velocity term does not exist yet: command terms are
        built in config order, and the wider band for one window, until the first
        resample after both exist, is the answer five_foot gives too.
        """
        try:
            vel = self._env.command_manager.get_command(self.cfg.velocity_command)
        except (KeyError, AttributeError):
            return torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        return torch.norm(vel[env_ids, :3], dim=1) < self.cfg.stand_threshold

    def hold_to_band(self) -> None:
        """Clamp every walking environment's three angles to `moving`.

        For a command that something other than the sampler writes -- the
        operator in replay, whose full deflection is the standing edge. Every
        step rather than at a resample, because a stick moves when it likes, and
        a walking robot handed the standing band has been asked for a posture
        its swinging legs were never trained to hold.
        """
        all_ids = torch.arange(self.num_envs, device=self.device)
        walking = ~self._in_standing_band(all_ids)
        if not walking.any():
            return
        for i, name in enumerate(ANGLES):
            half = getattr(self.cfg.moving, name)
            self.posture_command[walking, i] = self.posture_command[walking, i].clamp(
                -half, half
            )

    @property
    def footprint_yaw(self) -> torch.Tensor:
        """Where the footprint points in the world, in radians, [num_envs].

        `heading - body_twist`. `body_twist` is the body relative to its own
        footprint, so subtracting it from the body's world heading leaves the
        footprint's -- no second estimator, and the same Procrustes fit the
        tracking reward is scored on.
        """
        return wrap_to_pi(
            self._robot.data.heading_w
            - posture_state.body_twist(self._env, self.asset_cfg)
        )

    @property
    def foot_xy(self) -> torch.Tensor:
        """The foot sites in the world, x and y only, [num_envs, 6, 2]."""
        return posture_state.foot_pos_w(self._env, self.asset_cfg)[..., :2]

    @property
    def foot_displacement(self) -> torch.Tensor:
        """How far each foot has moved since the anchor, metres, [num_envs, 6].

        **Nothing charges this.** It is reported as
        `Metrics/posture/foot_displacement` and is the only reading, together with
        `footprint_rotation`, on which end of a twist actually turned --
        `mdp/rewards.py::track_twist` scores the body against the footprint and
        cannot tell "the body turned +theta" from "the footprint turned -theta".

        Per foot rather than fitted, and the measurement that forces it: the six
        feet sit at nearly one radius, so a rigid rotation moves them all by
        nearly the same distance. Parked, 32 environments, twist stepped from
        zero and held 200 steps on `08-23-12/model_47350`:

            leg                      LF     RF     LM     RM     LR     RR
            radius (mm)            188.5  188.6  196.9  196.9  219.7  219.7
            1.55 deg would move      5.1    5.1    5.3    5.3    5.9    5.9
            measured at +15 deg     66.3   77.5   24.2   20.8   12.1   45.3
            measured at -15 deg     74.2   63.6   19.4   27.6   51.0   11.0

        **The two feet with the smallest radius move the most, by three to six
        times** -- which is the one thing a rotation cannot do. It is the front
        pair being sheared apart, and they are *dragged* rather than stepped:
        airborne 0.6 to 3.3% of those 200 steps. A Procrustes fit read 1.55
        degrees of that, so `footprint_rotation` alone would have called it
        nothing happening.

        The pattern invites the wrong diagnosis on top of it. A front pair moving
        opposite ways looks like a gait, but `LF` is in `TRIPOD_A` and `RF` is in
        `TRIPOD_B` -- the tripod puts a left-right mirror pair in *different*
        groups, so it is the one pair the gait never moves together. Nothing
        groups the feet while parked: every gait term is gated off below a 0.05
        command.
        """
        return (self.foot_xy - self.foot_anchor).norm(dim=-1)

    def _anchor(self, env_ids: torch.Tensor) -> None:
        """Record where `env_ids`' feet are and where their footprints point."""
        self.foot_anchor[env_ids] = self.foot_xy[env_ids]
        self.footprint_anchor[env_ids] = self.footprint_yaw[env_ids]

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        """Keep the anchors under every environment that is free to move.

        The command itself needs no update -- it is held as sampled until the next
        resample. What this hook is for here is the anchors, which **track while
        moving and freeze while standing**. That needs no edge detection: an
        environment just parked carries the anchor from the step before, which is
        where it was when it was parked. It matters because the two terms resample
        on different clocks, so this one cannot see the moment the velocity
        command decides to stand.

        **A fresh episode is anchored here rather than in `_resample_command`**,
        and the difference is the whole point of the term that reads it. A resample
        is where a *new twist command* arrives, and re-anchoring there would hand
        the policy a free relocation at exactly the moment it is being asked to
        turn its body instead -- which is the behaviour being charged for. A reset
        is a different event and does need one, or an environment that restarts
        straight into standing is held to the previous episode's stance, somewhere
        else entirely. `episode_length_buf` is zeroed by `_reset_idx` and the
        managers run in the order `reset -> commands`, so a zero here is exactly
        "this environment restarted in this step".
        """
        del env_ids
        all_ids = torch.arange(self.num_envs, device=self.device)
        free = ~self._standing(all_ids) | (self._env.episode_length_buf == 0)
        if free.any():
            self._anchor(free.nonzero(as_tuple=False).flatten())

    @property
    def footprint_rotation(self) -> torch.Tensor:
        """How far the footprint has turned since the robot was parked, radians.

        **This is the constraint that makes a twist command mean anything.**

        `body_twist` is the body against its own footprint, which is the right
        frame -- `mdp/rewards.py::track_twist` has why, and
        `motion-planner`'s heading frame is the same thing. What it cannot see is
        which of the two rotated. "The body turned +theta over a fixed footprint"
        and "the footprint turned -theta under a fixed body" are one number, and
        the second is far cheaper while parked: turning the body means overcoming
        six friction cones, while standing in a joint configuration whose stance
        is rotated costs nothing. Measured on model_38000, commanded 10 degrees
        while parked:

            measured body_twist   +10.05 deg     track_twist scores full marks
            base heading change    -0.00 deg     the body did not turn at all

        In the planner that never happens because the feet are placed by the gait
        and can only move when the commanded yaw rate moves them. A policy has no
        such structure, so the constraint has to be paid for.

        The anchor tracks while the robot is free to move and freezes when it is
        parked, so this is "since you were told to stand" -- and while parked the
        commanded yaw rate is zero, which is what makes a *fixed* anchor the right
        reference and keeps any drift out of it. The walking case is not this
        quantity and is not handled.

        **Nothing is paid for this**, and nothing is paid for `foot_displacement`
        either since the term that did was removed. This is the quantity the
        requirement is *written* in -- "the footprint did not turn" -- and
        `foot_displacement` is the one that sees the shear a rigid-rotation fit
        cannot. Read them together or neither means much.
        """
        return wrap_to_pi(self.footprint_yaw - self.footprint_anchor)


@dataclass(kw_only=True)
class PostureCommandCfg(CommandTermCfg):
    """Config for `PostureCommand`.

    Every range is **required**: what posture a task asks for is the task's
    decision, and a default here would be four numbers inherited without being
    chosen -- the failure `CLAUDE.md` names and `tasks/jumper/common/` is arranged
    around.
    """

    entity_name: str = "robot"

    foot_sites: tuple[str, ...]
    """The six foot sites, in `constants.LEGS` order. The order matters: the twist
    estimate aligns them against `constants.NOMINAL_FOOT_XY` row by row, and a
    permuted tuple silently measures the alignment of one footprint onto a
    different robot's."""

    rel_neutral_envs: float = 0.0
    """Fraction of **moving** environments commanded to the neutral posture
    exactly. See the module docstring for why a uniform draw alone is not
    enough."""

    rel_neutral_standing_envs: float = 0.0
    """The same, for environments the velocity command has parked.

    **Separate from `rel_neutral_envs` because "stand still and hold a pose" is
    what this task is for**, and drawing the two commands independently spends
    part of the standing time on the one posture that asks nothing. With
    `rel_standing_envs` at 0.3 and both neutral rates at 0.2, the share of time
    that is "not moving, holding a commanded posture" is 0.3 x 0.8 = 24%; at 0.0
    here it is the whole 30%.

    Set it above zero to keep some idle-at-home time -- a robot standing
    normally with nobody asking for anything is also a real case, and it is the
    one `velocity_env_cfg`'s standing terms (`stance_load_stand`,
    `standing_sway`) were calibrated on."""

    velocity_command: str = "twist"
    """The command term that decides both which neutral rate applies (by its
    standing flag) and which band the angles are drawn from (by its norm). Read
    by name at resample time rather than held as a reference: command terms are
    built in `cfg.commands` order and this one is built second, so the other
    exists by then -- but a task that reorders them would otherwise get a
    construction-time crash instead of a lookup that simply finds nothing."""

    stand_threshold: float
    """Below this norm of the velocity command's `[vx, vy, wz]` the angles come
    from the standing band, `ranges`; at or above it, from `moving`. Required like
    the ranges: which band a slow walk gets is the task's call."""

    neutral_height: float
    """The height the neutral posture holds, in metres. `constants.STAND_Z` is
    what the robot stands at, and anything else is a deliberate statement that the
    task's idea of neutral is not the robot's."""

    @dataclass
    class Ranges:
        twist: float
        """Half-range of the body yaw over the feet **while standing**, in
        radians -- the widest the command goes, and what a stick at full
        deflection asks for. Symmetric by construction -- an asymmetric twist
        range would be a standing bias, and the left-right mirror augmentation
        assumes there is not one."""

        pitch: float
        """Half-range of pitch while standing, in radians. Symmetric, but for a
        different reason than the twist: the robot is *not* fore-aft symmetric --
        two 5-DoF arms in front, four 3-DoF legs behind -- so nose-up and
        nose-down are genuinely different postures and one may well turn out
        harder than the other. That difference belongs in what the policy learns,
        not in what it is asked for."""

        roll: float
        """Half-range of roll while standing, in radians."""

        height: tuple[float, float]
        """Height above the ground, in metres, as (low, high). Not symmetric about
        anything -- it is an absolute height, and `neutral_height` is where the
        robot stands within it. One range, walking or not."""

    ranges: Ranges

    @dataclass
    class Moving:
        """The half-ranges the three angles are held to while walking, radians.
        Each inside its standing counterpart in `ranges`."""

        twist: float
        pitch: float
        roll: float

    moving: Moving

    def build(self, env: "ManagerBasedRlEnv") -> PostureCommand:
        return PostureCommand(self, env)

    def __post_init__(self) -> None:
        low, high = self.ranges.height
        if not low <= self.neutral_height <= high:
            raise ValueError(
                f"neutral_height {self.neutral_height} is outside ranges.height "
                f"{self.ranges.height}, so the posture the neutral environments "
                f"hold is one no sampled environment is ever asked for"
            )
        if not 0.0 <= self.rel_neutral_envs <= 1.0:
            raise ValueError(
                f"rel_neutral_envs is a fraction of environments, got "
                f"{self.rel_neutral_envs}"
            )
        if not self.stand_threshold > 0.0:
            raise ValueError(
                f"stand_threshold is a norm, got {self.stand_threshold}; at zero no "
                "command is ever standing and the standing band is never drawn"
            )
        for name in ANGLES:
            walk, stand = getattr(self.moving, name), getattr(self.ranges, name)
            if not 0.0 < walk <= stand:
                raise ValueError(
                    f"moving.{name} is {walk} against a standing {stand}: the walking "
                    "band has to sit inside the standing one, or a robot that starts "
                    "walking is handed a wider posture than it was asked to hold still"
                )


__all__ = ["PostureCommand", "PostureCommandCfg"]
