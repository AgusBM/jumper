"""A command-range curriculum written for this hexapod, not inherited from Go1.

mjlab's `commands_vel` was removed in `velocity_env.py` for two independent
reasons: its values are sized for a 0.278 m quadruped, and it is **open-loop** --
the range expands on a step schedule whether or not the policy learned anything.
This replaces it with the two properties that were missing.

## 1. std moves with the range, so the ruler never changes

The tracking reward is `exp(-err^2 / std^2)`, so `std` is what decides how much
error still counts as tracking. Moving the range without moving `std` silently
changes the difficulty -- already recorded in `velocity_env.py`: halving the range
alone cut `error_vel_xy` by 21% while the tracking reward rose 35%, and "the extra
was not a better policy but a looser ruler".

Measured over the real command distribution, sweeping the range from 0.50 down to
0.15 (**what a robot that never moves collects**, of the 4.0 the two tracking terms
are worth):

    range   std fixed at 0.25/0.20      std scaled with the range
    0.50            1.35                        1.34
    0.35            1.87                        1.29
    0.25            2.43                        1.28
    0.15            3.11                        1.28

Held fixed, `std` lets the free score nearly treble as the range narrows -- the
task gets easier in exactly the way this robot must not be allowed to exploit,
since standing still is its documented local optimum. Scaled, the free score is
flat to within 5% across the whole sweep. **That invariance is what makes a range
curriculum legitimate here**: the level changes what is asked, not how strictly it
is marked.

The ratios are not free parameters. They are read off the pair `velocity_env.py`
already calibrated (0.25 and 0.20 at range 0.5, chosen so that standing still
collects the same 0.14 it does on Go1):

    lin_std = 0.50 * lin_range      ang_std = 0.40 * ang_range

so the **final shared level reproduces the calibrated configuration exactly**. The
curriculum's endpoint is where the task would have been without it.

## Rungs a task adds, and the one kind that breaks the invariance

`levels` and `ang_levels` are parameters, so a task can append to the shared
ladder. `jumper.flat` adds a wider rung, which is more of the same and keeps the
ratio. `jumper.tripod` adds a **precision rung** -- the top range repeated, marked
more strictly, via `lin_std_scales` -- and that one is a deliberate exception to
everything above.

It is legitimate only in that position. The invariance argument says the ruler must
not change *while the range is changing*, because then a level confounds two
difficulties; a rung that repeats the range changes nothing else, so the tightening
is the only variable and a policy that fails it has failed precision and nothing
else. Which rung is which is visible in the logs -- `lin_std` and `ang_std` are
logged per level, and on a precision rung `lin_range` stays put while `lin_std`
drops.

## The bar has a floor, because the error does

    bar = gate_floor + gate_ratio * range

and the floor is not a fudge factor. `error_vel_xy` has a component that does not
scale with the command range at all, so a bar that is purely proportional gets
**tighter than the noise** as the range narrows -- which made the first rung the
hardest one and left a 37500-iteration run pinned at level 0 for its whole life.

Measured on that run's policy, one full 1000-step episode per range at 512 envs,
with the stochastic policy training actually collects (the deterministic policy is
not what the metric sees):

    range   err_vel_xy   0.3*range   floor+0.3*range
    0.15      0.0550       0.0450  x      0.0750  ok    <- removed from LEVELS
    0.25      0.0586       0.0750  ok     0.1050  ok
    0.35      0.0717       0.1050  ok     0.1350  ok
    0.50      0.1435       0.1500  ok        --   (top, its bar never gates)

0.15 was the only rung the policy could not clear, and it failed by 22%. Note the
shape: the error is nearly flat from 0.15 to 0.35 and only then starts to track
the range, which is exactly what a fixed noise floor plus a proportional term
looks like.

0.15 is **also gone from `LEVELS`** -- the floor alone would have made it passable,
and it was dropped anyway. See the note there; briefly, a rung where two thirds of
the bar is exploration noise is not measuring the policy, and the run has better
things to spend a dwell period on.

Only the bars at 0.25 and 0.35 now decide anything: 0.50 is the top, and arriving
there is what sets `commands_maxed`, not clearing its gate.

**The two fixes overlap, and the result is loose.** Removing the 0.15 rung and
adding the floor each solve the same problem, and doing both leaves:

    level 0  range 0.25  bar 0.1050  err 0.0586   margin 44%
    level 1  range 0.35  bar 0.1350  err 0.0717   margin 47%

At those margins both promotions fire as soon as the dwell allows, so the
curriculum is closer to a 200-iteration schedule than to a performance gate --
which is the property this module exists to provide. The obvious response is to
tighten `gate_ratio`, and it is deliberately **not** being taken yet: every error
above was measured on a policy trained while `track_angular_velocity` returned
zero, so it never spent anything on turning. A policy that does will have a
different xy error, most likely a worse one early on. Tightening a gate against
numbers from a broken reward is exactly the sequence that produced the stall this
whole section is about.

So: run it once with the reward fixed, read `lin_err` against `lin_err_bar` on the
new run, and tighten then. If the margin really is ~45% on a healthy policy,
`gate_ratio` of 0.2 puts it back near 27%.

**Where the floor comes from.** Most of it is PPO's own exploration noise, not
anything about the policy's competence. The same episodes with the deterministic
policy:

    level 0   stochastic 0.0550   deterministic 0.0221   difference 0.033
    level 3   stochastic 0.1435   deterministic 0.1169   difference 0.027

-- roughly constant, as a fixed action-noise std should be, and `Policy/mean_std`
sat at 0.295 all run without decaying. The rest is the gait's own body-velocity
ripple: envs commanded to stand still, where the command is exactly zero, read
0.0113. So 0.03 is the exploration component measured at both ends of the range,
and it is deliberately the smaller reading of the two.

**This will need re-measuring.** These numbers come from a policy trained with
`track_angular_velocity` returning zero (see `mdp/rewards.py::track_yaw_velocity`),
so it never learned to turn. A policy that does turn moves differently and its
floor may not be 0.03. The structure -- a floor plus a proportional part -- is what
the measurement establishes; the constant is the part to check again.

## 2. The gate is performance, not step count

A level is held until the policy actually tracks at that level, rather than on a
step schedule that advances whether or not anything was learned.

Measured on a run that used this: the three promotions landed at iterations 282,
382 and 482, and the last two are exactly one dwell period apart -- so **only the
first was limited by performance**; after that the gate was already satisfied and
the dwell was what held the level. Together with the finished policy tracking 85%
of the commanded speed at the top level, that says `gate_ratio` is on the
permissive side. It is the first thing to tighten if the levels go by too fast to
be doing any work.

The measure is the mean per-step tracking error over the episode that just ended,
recovered exactly from mjlab's own accumulator:

    mean_err = metrics["error_vel_xy"] * max_command_step / episode_length_buf

`error_vel_xy` adds `err / max_command_step` every step, so multiplying it back by
`max_command_step` and dividing by the number of steps gives the plain mean in m/s.
This is read in `CurriculumManager.compute`, which
`ManagerBasedRlEnv._reset_idx` calls **first**, before the sim, the metrics and the
command manager are reset -- so both the accumulator and `episode_length_buf` still
hold the finished episode's values. Reading them anywhere later gets zeros.

The threshold scales with the range, `mean_err < gate_ratio * range`, which keeps
it meaningful at every level. The reference point is measured, not assumed: a
robot that never moves scores **0.71 * range** (300 steps of zero actions gave a
mean error of 0.108 at range 0.15, against a mean command magnitude of 0.107 --
for a motionless robot the tracking error simply *is* the command). So the default
`gate_ratio` of 0.3 asks for the error to be cut to 42% of what standing still
gives: a bar standing cannot clear, and one that does not demand good tracking
either.

The same measurement validates the recovery formula end to end -- the value
reconstructed from the accumulator was 0.1083 against 0.1079 accumulated directly
step by step, with a worst-case per-env deviation of 0.0014 (one step of lag).

## What the two curricula log, and why the names are shared

Both terms in this module return a dict that mjlab logs as
``Curriculum/<term>/<key>``, and the two use **one vocabulary**:

    level          the level now in force (mean over envs, for terrain)
    level_max      terrain only: the highest level any env has reached
    lin_err        the linear tracking error the decision is made on
    lin_err_bar    what `lin_err` has to get under
    ang_err        terrain only: the same for yaw
    ang_err_bar    what `ang_err` has to get under
    settled        the dwell is satisfied (0/1, or the fraction of envs)
    promoted       this call moved a level up (0/1, or the fraction)
    demoted        terrain only: this call moved a level down

The rule is **every error is logged next to the bar it is judged against**, and
a key that appears in both groups means the same thing in both.

That is not tidiness. The previous names were `err_mean` against `gate` in one
group and `err_xy` / `err_yaw` against nothing at all in the other -- so the
question "why has the level not moved" could not be answered from the dashboard.
It took a stalled 37500-iteration run to notice, and the answer was sitting in
two numbers that were both being logged, under names that did not suggest they
should be compared: `err_mean` 0.0548 against `gate` 0.0450.

The term keys are `command` and `terrain` rather than `command_range` and
`terrain_levels`. `terrain_levels` is also mjlab's own key for the term this one
replaces, and `velocity_env.py` pops it -- so if a future edit drops the pop, the
duplicate shows up as a second group in the log instead of silently running two
terrain curricula against each other.

## What this deliberately does not do

**No demotion.** Levels only advance. A promote/demote pair oscillates around its
threshold, and each oscillation moves `std`, i.e. the reward function, which is
hostile to the value function PPO is fitting.

## Resuming

**The level is checkpointed, by the task's runner.** mjlab checkpoints the policy
and the step counter; `common/runner.py` adds every `CheckpointedLadder`'s
`state_dict()`, so `train.py --resume` carries on at the rung the previous run had
reached, with its error average and its promotion clock.

`MJRL_COMMAND_LEVEL` **overrides** the checkpoint rather than standing in for it --
a run moved onto different ground may want a lower rung than the one it left:

    MJRL_COMMAND_LEVEL=2 python scripts/train.py ... --resume

Set, it wins. Where it names another rung than the checkpoint's, the error average
is dropped (it was measured against a different range) and the dwell counts from
the resume, as after a promotion. It is also the only source for a checkpoint
written before levels were saved; the runner says so when it loads one.

**The top level depends on the task**, because a task may append rungs. The shared
ladder tops out at 2; `jumper.tripod` adds a precision rung and tops out at 3;
`jumper.flat` adds a wider one and also tops out at 3. It became 2 when the 0.15 rung
was removed, so a value carried over from a run that predates any of these changes
points at a different rung than it did. Out of range is clamped to that task's top
rather than raising, which is the safe direction but silent. Set it by the *range
and ruler* you want, not by the number you remember -- for `jumper.tripod`:

    level 0  range 0.25 / 0.30   std 0.125 / 0.120
    level 1  range 0.35 / 0.50   std 0.175 / 0.200
    level 2  range 0.50 / 0.75   std 0.250 / 0.300   <- the calibrated pair
    level 3  range 0.50 / 0.75   std 0.125 / 0.200   <- same range, stricter ruler

A policy that finished the curriculum as it stood before the precision rung existed
is at **level 2**, not 3: it earned the top range on the old ruler, and that is
exactly what level 2 is. Resuming it at 3 hands it the tightened ruler with none of
the dwell that is meant to precede it.

Left at 0, a resume from such an older checkpoint re-runs the whole curriculum,
which is not merely slow: it trains a policy that had outgrown the narrow ranges
back down onto them. Note also that `play.py` builds with `curriculum = {}`, so
replay uses the config's own range (the final level's), not whatever level
training had reached.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

#: The linear range at each level, for lin_vel_x and lin_vel_y. The last is
#: `velocity_env.py`'s calibrated value, so finishing the curriculum lands exactly
#: on the configuration that file argues for.
#:
#: **There used to be a 0.15 rung below this one, and it was removed.** Not
#: because it was too hard for the policy -- because it was too small for the
#: *measurement*. The bar at 0.15 was 0.045 and 0.03 of that was PPO's own
#: exploration noise, so two thirds of the budget went to something no amount of
#: learning removes and the rung was unclearable (see the table in the module
#: docstring: 0.0550 against 0.0450, the only rung the policy failed).
#:
#: The floor added to the bar makes 0.15 passable again, so this is belt and
#: braces -- deliberately. The floor is one measured constant standing between the
#: run and a repeat of the same stall, and at 0.15 it would be carrying 67% of the
#: bar; at 0.25 it carries 40%. Do not put the rung back without re-measuring what
#: fraction of the bar the noise is eating at that range.
#:
#: Nothing was lost in bootstrapping terms. The stalled run spent its **entire
#: 37500 iterations** at 0.15 and the policy walked perfectly well from it -- 0.15
#: was where it got stuck, not what got it started -- and mjlab's Go1 default is
#: +/-1.0 with no curriculum at all, on a robot 2.6x taller.
LEVELS: tuple[float, ...] = (0.25, 0.35, 0.50)

#: The angular range at each level, which **goes further than the linear one**.
#:
#: The two were the same list to begin with, and that was a convenience rather
#: than a finding. Turning is the thing this robot was worst at, and it turned out
#: to have more room than the ceiling allowed: measured on a policy trained to
#: 30000 iterations, yaw tracking holds a steady 65-70% of the command and **does
#: not saturate** anywhere in the range tested --
#:
#:     commanded   0.20   0.35   0.50   0.75   1.00  rad/s
#:     achieved    0.138  0.227  0.350  0.486  0.660
#:
#: -- so a ceiling of 0.50 was leaving capability unused. 0.75 is the new top;
#: 1.00 is reachable too and is the obvious next step if this one is comfortable.
#:
#: Level 1 is 0.50, which keeps the old top as a rung rather than removing it, so
#: the extra range is a level added on the end rather than a shifted scale.
#:
#: The 0.15 rung went with `LEVELS`' -- one level is one rung on both axes, and the
#: assert below is what stops the two lists drifting apart.
ANG_LEVELS: tuple[float, ...] = (0.30, 0.50, 0.75)

assert len(LEVELS) == len(ANG_LEVELS), (
    "one level is one rung on both axes; a mismatch would silently promote the "
    "two at different rates"
)

#: std as a fraction of the range, from the calibrated pair at range 0.5. Applied
#: to each axis' own range, so the angular std reaches 0.30 at the widening levels
#: -- the ruler stays exactly as strict while the range grows, which is what makes
#: the widening part of this curriculum legitimate. See the sweep in the module
#: docstring.
#:
#: **Reference values, and each task names them rather than inheriting them.**
#: `velocity_env_cfg` requires the ratio as an argument; these are what the four
#: jumper tasks pass today, and what a fifth should pass unless it has a reason not
#: to. The derivation is this robot's, so a different gait on the same robot has
#: every reason to reuse it -- and should say that it is doing so.
STD_LIN_RATIO = 0.25 / 0.5
STD_ANG_RATIO = 0.20 / 0.5

#: The command ladder as fractions of a task's own ceiling, and the shape every
#: gait climbs. **The ceiling is the task's; the shape is not** -- a gait that can
#: only reach 0.4 m/s should still spend its first rung at the same *fraction* of
#: what it can do as one that reaches 0.8, or the two are not running the same
#: curriculum and their levels are not comparable.
#:
#: These are `jumper.tripod`'s, which were the undistorted pair: its rungs divided by
#: its ceiling give exactly this. `jumper.ripple` and `jumper.flat` used to be built by
#: clamping and appending to the shared *absolute* rungs, which squeezed one ladder
#: to (0.625, 0.875, 1.0) and stretched the other to (0.312, 0.437, 0.625, 1.0) --
#: three different curricula wearing one name.
#:
#: The linear and angular fractions differ from each other, and that is inherited:
#: turning was the thing this robot was worst at, and `ANG_LEVELS` was widened at
#: the top for the measured reason recorded there.
LEVEL_FRACTIONS: tuple[float, ...] = (0.5, 0.7, 1.0)
ANG_LEVEL_FRACTIONS: tuple[float, ...] = (0.4, 2.0 / 3.0, 1.0)

#: The precision rung's std multipliers -- the top range repeated and marked more
#: strictly. See the note on `STD_LIN_RATIO` for the measurement, and the module
#: docstring for why a rung that widens nothing is the only place the ruler may
#: change.
PRECISION_LIN_SCALE = 0.5
PRECISION_ANG_SCALE = 0.20 / 0.30


def ladder(lin_ceiling: float, ang_ceiling: float):
    """`(levels, ang_levels, lin_scales, ang_scales)` for a task's ceiling.

    One curriculum, scaled. The widening rungs are `LEVEL_FRACTIONS` of the
    ceiling, then a precision rung repeats the top and tightens the ruler -- so
    every gait climbs the same shape and differs only in how fast it is asked to
    go, which is the one thing a gait genuinely fixes.
    """
    lin = tuple(round(lin_ceiling * f, 6) for f in LEVEL_FRACTIONS)
    ang = tuple(round(ang_ceiling * f, 6) for f in ANG_LEVEL_FRACTIONS)
    return (
        (*lin, lin[-1]),
        (*ang, ang[-1]),
        (*(1.0,) * len(lin), PRECISION_LIN_SCALE),
        (*(1.0,) * len(ang), PRECISION_ANG_SCALE),
    )


def lin_std(
    level: int,
    levels: tuple[float, ...] = LEVELS,
    scales: tuple[float, ...] | None = None,
    ratio: float = STD_LIN_RATIO,
) -> float:
    """The linear tracking std at `level`.

    A function because three places need it and they must not drift: the curriculum
    writes it into the live reward manager, `velocity_env.py` sets the config
    default from `lin_std(-1)`, and a task that appends its own rungs sets it from
    its own lists. The config default is what `play.py` replays with -- it builds
    with `curriculum = {}` and never runs this class -- so a second copy of
    `ratio * range * scale` anywhere is a silent mismatch between what training
    ended on and what replay shows.

    `scales` is a per-level multiplier on the ratio-derived std, `None` meaning 1.0
    everywhere. It is how a **precision rung** is expressed: a level that repeats
    the range below it and marks it more strictly. Nothing here requires the
    tightening to be last, but see the module docstring for why it should be.
    """
    scale = 1.0 if scales is None else scales[level]
    return ratio * levels[level] * scale


def ang_std(
    level: int,
    levels: tuple[float, ...] = ANG_LEVELS,
    scales: tuple[float, ...] | None = None,
    ratio: float = STD_ANG_RATIO,
) -> float:
    """The angular tracking std at `level`. See `lin_std`."""
    scale = 1.0 if scales is None else scales[level]
    return ratio * levels[level] * scale


class CheckpointedLadder:
    """What a ladder carries across `train.py --resume`, and which source wins.

    Mixed into every curriculum term that earns its level -- this module's command
    ladder, `jumper.posture`'s and `jumper.five_foot`'s payload. `common/runner.py`
    finds them by this type, saves `state_dict()` into each checkpoint and hands it
    back to `load_state_dict()` on a resume; nothing else calls either.

    A subclass sets `LEVEL_ENV`, takes its starting level from
    `_level_from_env(top)` and keeps `level` and `last_promotion_step` up to date.

    The samples pooled towards the next error average are not saved: fewer than
    `min_samples` of them, and the average they feed is.
    """

    #: The environment variable that sets the level by hand.
    LEVEL_ENV: str
    #: Restored with the level, and **only when the level is the one they were
    #: measured at**: an error average is against one rung's range, and carried
    #: onto another it would decide a promotion on evidence from a different test.
    EVIDENCE: tuple[str, ...] = ("err_ema", "last_promotion_step")

    level: int
    last_promotion_step: int

    def _level_from_env(self, top: int) -> int:
        """The starting level: `LEVEL_ENV` clamped to `[0, top]`, or 0 if unset."""
        # `.strip()`: an unset variable and one exported empty are the same
        # intent, and `int("")` raises. `MJRL_COMMAND_LEVEL=` is what a shell
        # produces when the value is a variable that happens to be empty.
        raw = (os.environ.get(self.LEVEL_ENV) or "").strip()
        self._forced = bool(raw)
        self._top = top
        return max(0, min(int(raw or "0"), top))

    def state_dict(self) -> dict:
        """The level and its evidence, as plain Python for `torch.save`."""
        return {
            "level": self.level,
            **{key: copy.deepcopy(getattr(self, key)) for key in self.EVIDENCE},
        }

    def load_state_dict(self, state: dict | None, step: int) -> str:
        """Resume from `state`; return where the level came from, for the log.

        Args:
            state: what `state_dict()` saved, or None for a checkpoint without this
                ladder -- written before levels were saved, or by a run whose
                task did not have this term yet. The constructor's level stands.
            step: the step counter the checkpoint restored, the clock
                `last_promotion_step` is on.
        """
        if hasattr(self, "_applied"):
            # The wrapper's first reset already wrote the constructor's level into
            # the managers; this makes the next call write the restored one.
            self._applied = False
        env = f"${self.LEVEL_ENV}"
        if state is None:
            origin = env if self._forced else "the default"
            return f"level {self.level} from {origin}; the checkpoint does not carry it"
        saved = int(state["level"])
        level = self.level if self._forced else max(0, min(saved, self._top))
        self.level = level
        if level != saved:
            self.last_promotion_step = step
            why = env if self._forced else f"this ladder's top ({self._top})"
            return (
                f"level {level} from {why}, not the checkpoint's {saved}; the error "
                f"average restarts and the dwell counts from here"
            )
        for key in self.EVIDENCE:
            if key in state:
                setattr(self, key, copy.deepcopy(state[key]))
        return f"level {level} from the checkpoint" + (f" ({env} agrees)" if self._forced else "")


class CommandRangeCurriculum(CheckpointedLadder):
    """Widen the velocity command range as tracking improves, scaling `std` with it.

    A class term rather than a function because the level, the error average and
    the promotion time have to survive between calls; mjlab instantiates it once
    with ``(cfg, env)`` and then calls it like a function.
    """

    LEVEL_ENV = "MJRL_COMMAND_LEVEL"

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
        del env
        # **Where to start.** Fresh training starts at 0 and earns its way up.
        # A run resumed from a policy that already finished this curriculum must
        # not: dropping it back to the narrowest range retrains it on commands it
        # outgrew, and the levels it has to re-climb cost one dwell each.
        #
        # Measured on the run this was added for -- a flat-trained policy resumed
        # onto rough terrain -- the command range fell from its top level back to
        # level 1 and stayed, while yaw tracking went from 0.766 of 2.0 to 0.004.
        # The terrain was the trigger, but re-running this curriculum from zero is
        # what kept the command range narrow while the policy was being rewritten.
        #
        # The checkpoint carries it now (`CheckpointedLadder`, and the note at the
        # top of this module); `MJRL_COMMAND_LEVEL` sets it by hand, for a
        # checkpoint that predates that or a run that wants another rung. The
        # level belongs to the *run*, not to the configuration.
        #
        # **There is no `start_level` param.** mjlab also splats every param into
        # `__call__`, which takes no such keyword, so a read of it here was a knob
        # that raised TypeError on the first reset.
        params = getattr(cfg, "params", None) or {}
        # Clamped against **this task's** ladder, not the shared one. A task may
        # append rungs (`jumper.flat` adds a wider one, `jumper.tripod` a stricter one)
        # by overriding `levels`, and clamping to `len(LEVELS) - 1` would silently
        # refuse to resume into any of them -- the top of the ladder would be
        # unreachable by the only mechanism there is for reaching it.
        top = len(params.get("levels", LEVELS)) - 1
        self.level = self._level_from_env(top)
        self._start_level = self.level
        self.err_ema: float | None = None
        # Samples pooled across resets until `min_samples` have arrived.
        self._pending: float = 0.0
        self._pending_n: int = 0
        self.last_promotion_step = 0
        self._applied = False

    def _apply(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        levels: tuple[float, ...],
        ang_levels: tuple[float, ...],
        lin_reward: str,
        ang_reward: str,
        lin_scales: tuple[float, ...] | None = None,
        ang_scales: tuple[float, ...] | None = None,
        lin_ratio: float = STD_LIN_RATIO,
        ang_ratio: float = STD_ANG_RATIO,
    ) -> float:
        """Write the current level's range and std into the live managers.

        **Through the managers, never through `env.cfg`.** `RewardManager.__init__`
        does `self.cfg = deepcopy(cfg)`, so assigning to `env.cfg.rewards[...]`
        changes an object nothing reads; `get_term_cfg` returns the manager's own
        term, and `compute()` re-reads `term_cfg.params` every step, so mutating it
        takes effect on the next step. The failure mode of getting this wrong is
        silence: training continues at the unchanged std.
        """
        r = levels[self.level]
        a = ang_levels[self.level]
        ranges = env.command_manager.get_term(command_name).cfg.ranges
        ranges.lin_vel_x = (-r, r)
        ranges.lin_vel_y = (-r, r)
        ranges.ang_vel_z = (-a, a)
        env.reward_manager.get_term_cfg(lin_reward).params["std"] = lin_std(
            self.level, levels, lin_scales, lin_ratio
        )
        env.reward_manager.get_term_cfg(ang_reward).params["std"] = ang_std(
            self.level, ang_levels, ang_scales, ang_ratio
        )
        return r

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        command_name: str = "twist",
        levels: tuple[float, ...] = LEVELS,
        ang_levels: tuple[float, ...] = ANG_LEVELS,
        lin_std_scales: tuple[float, ...] | None = None,
        ang_std_scales: tuple[float, ...] | None = None,
        lin_std_ratio: float = STD_LIN_RATIO,
        ang_std_ratio: float = STD_ANG_RATIO,
        gate_ratio: float = 0.3,
        gate_floor: float = 0.03,
        ema_alpha: float = 0.05,
        dwell_steps: int = 24 * 100,
        min_steps: int = 50,
        min_samples: int = 8,
        warmup_steps: int = 0,
        lin_reward: str = "track_linear_velocity",
        ang_reward: str = "track_angular_velocity",
    ) -> dict[str, torch.Tensor | float]:
        """Advance at most one level per call; return the state for logging.

        Args:
            gate_ratio: the part of the bar that scales with the range. **Measured,
                a robot that never moves sits at 0.71 * range** (300 steps of zero
                actions: mean error 0.108 at range 0.15, against a mean command
                magnitude of 0.107 -- a motionless robot's error simply *is* the
                command). So 0.3 asks for the error to be cut to 42% of what
                standing still gives, before the floor below is added.
            gate_floor: the part of the bar that does **not** scale with the range,
                because the thing it is paying for does not either. See the note
                on this in the module docstring; the short version is that
                `gate_ratio * range` alone made level 0 the hardest rung in the
                curriculum and no policy could ever leave it.
            ema_alpha: smoothing for the error average across reset batches. Each
                call sees only the envs resetting on this step, which is a small
                and noisy sample.
            dwell_steps: minimum environment steps between promotions. A promotion
                widens the range, so the error jumps; without a dwell the level
                could climb again on stale evidence before that shows up. The
                default is 100 iterations at 24 steps each.
            min_steps: ignore envs whose episode was shorter than this. An env that
                fell over after a few steps says nothing about tracking, and its
                near-zero accumulator would drag the average down -- towards
                promotion.
            warmup_steps: no promotion at all before this many environment steps
                have been taken, whatever the error says. **0 disables it and is
                the default**, because the four locomotion tasks have never
                needed it: their dwell is 2400 environment steps and their
                rollout is 24, so their first promotion cannot land before
                iteration 100 anyway.

                A task with a longer rollout gets a shorter wait from the same
                dwell -- `hexa.posture` runs 48 steps an iteration, so 2400 steps
                is iteration 50 -- and that is where its first promotion landed,
                on evidence from the handful of environments that had ended early
                when the first full episode was still 33 iterations away. The
                dwell is a gap between promotions; this is a floor under the
                first one, and the two are different questions.
            min_samples: ignore a batch with fewer than this many valid envs.
                Roughly 10% of envs are commanded to stand (`rel_standing_envs`),
                and a motionless robot tracks a zero command *perfectly* -- so a
                batch of one or two envs that happened to be standing reports a
                near-zero error. Observed directly: a 3-iteration run, where only
                one or two envs ever reset, reported a mean error of 0.0021 against
                the true 0.108. In a real run resets arrive continuously and the
                average is sound; this keeps the early numbers from being nonsense
                as well.
            lin_std_scales: per-level multiplier on the std the ratio produces,
                `None` meaning 1.0 at every level. A level whose scale is below 1
                is a **precision rung**: same range as the one below it, marked
                more strictly. `jumper.tripod` appends one. Widening and tightening
                are separate difficulties, and a policy that meets both at once
                cannot tell you which one it failed.
            ang_std_scales: the same, for the angular axis.
        """
        # A scale list shorter than the ladder would raise here rather than at the
        # promotion that first indexes past its end, which could be hours in.
        for name, scales, ladder in (
            ("lin_std_scales", lin_std_scales, levels),
            ("ang_std_scales", ang_std_scales, ang_levels),
        ):
            assert scales is None or len(scales) == len(ladder), (
                f"{name} has {len(scales)} entries for a {len(ladder)}-rung ladder"
            )

        if not self._applied:
            # Applied on the first call rather than in __init__, which avoids
            # depending on the order the managers are constructed in.
            # `_reset_idx` runs the curriculum before the command manager resamples,
            # so level 0 is in force for the very first commands.
            self._apply(
                env, command_name, levels, ang_levels, lin_reward, ang_reward,
                lin_std_scales, ang_std_scales, lin_std_ratio, ang_std_ratio,
            )
            self._applied = True

        r = levels[self.level]
        command_term = env.command_manager.get_term(command_name)

        # The first reset happens before any stepping: the accumulator and the
        # step counter are both zero, and a zero error reads as perfect tracking.
        # mjlab's own terrain curriculum guards the same moment the same way.
        if env.common_step_counter > 0 and len(env_ids) > 0:
            steps = env.episode_length_buf[env_ids]
            valid = steps >= min_steps
            n_valid = int(valid.sum())
            # **Accumulated across resets, not required within one.**
            # `on_policy_runner.learn` randomises the initial episode lengths to
            # decorrelate the environments, so they time out **one at a time**:
            # the expected batch is `num_envs / episode_steps`, which is about 1
            # at any size worth running. Asking for `min_samples` in a single
            # batch therefore asks for a Poisson coincidence, not a sample size,
            # and the whole gate was decided by how often one happened to occur.
            #
            # Measured on `hexa.posture` at 200 Hz, where the 4000-step episode
            # made the coincidence too rare to happen at all: 1143 iterations,
            # every reset a batch of exactly 1, and **not one sample taken** --
            # `lin_err` logged nan from the first iteration to the last and the
            # ladder could never leave level 0. At 100 Hz the same code filled the
            # average, which is what hid it: a 2000-step episode makes the
            # coincidence a hundred times likelier, and one hit is enough to seed
            # an EMA that then never returns to None.
            #
            # Pooling instead keeps what `min_samples` is for -- its note is about
            # one or two environments that happened to draw the neutral posture
            # deciding a promotion -- while letting the envs arrive one by one.
            if n_valid:
                max_command_step = (
                    command_term.cfg.resampling_time_range[1] / env.step_dt
                )
                err = command_term.metrics["error_vel_xy"][env_ids][valid]
                self._pending += float(
                    (err * max_command_step / steps[valid].float()).sum()
                )
                self._pending_n += n_valid
            if self._pending_n >= min_samples:
                mean_err = self._pending / self._pending_n
                self._pending, self._pending_n = 0.0, 0
                self.err_ema = (
                    mean_err
                    if self.err_ema is None
                    else (1 - ema_alpha) * self.err_ema + ema_alpha * mean_err
                )

        # Kept as three separate conditions rather than one `and` chain so each
        # can be logged on its own. Which of them is false is the entire content
        # of "why has the level not moved", and deriving it from `level` alone
        # costs a source dive -- as it did, for 37500 iterations.
        settled = env.common_step_counter - self.last_promotion_step >= dwell_steps
        bar = gate_floor + gate_ratio * r
        tracked = self.err_ema is not None and self.err_ema < bar
        warmed_up = env.common_step_counter >= warmup_steps
        promotable = (
            self.level + 1 < len(levels) and tracked and settled and warmed_up
        )
        if promotable:
            self.level += 1
            self.last_promotion_step = env.common_step_counter
            r = self._apply(
                env, command_name, levels, ang_levels, lin_reward, ang_reward,
                lin_std_scales, ang_std_scales, lin_std_ratio, ang_std_ratio,
            )
            # Start the average again: the old value was measured against a
            # narrower range and is not comparable with what follows.
            self.err_ema = None
            # The pooled samples were measured against the narrower range
            # and are not comparable with what follows either.
            self._pending, self._pending_n = 0.0, 0
            bar = gate_floor + gate_ratio * r

        # Key names are shared with `TerrainLevels` on purpose: `level`,
        # `lin_err` / `lin_err_bar`, `settled` and `promoted` mean the same thing
        # in both groups, so the two curricula read as one dashboard. See the
        # note on naming at the top of this module.
        return {
            "level": float(self.level),
            "lin_range": r,
            "ang_range": ang_levels[self.level],
            "lin_std": lin_std(self.level, levels, lin_std_scales, lin_std_ratio),
            "ang_std": ang_std(self.level, ang_levels, ang_std_scales, ang_std_ratio),
            "lin_err": float("nan") if self.err_ema is None else self.err_ema,
            "lin_err_bar": bar,
            "settled": float(settled),
            "warmed_up": float(warmed_up),
            "promoted": float(promotable),
        }


__all__ = [
    "ANG_LEVELS",
    "ANG_LEVEL_FRACTIONS",
    "LEVELS",
    "LEVEL_FRACTIONS",
    "STD_ANG_RATIO",
    "STD_LIN_RATIO",
    "CheckpointedLadder",
    "CommandRangeCurriculum",
    "TerrainLevels",
    "ang_std",
    "ladder",
    "lin_std",
]


class TerrainLevels:
    """Promote a robot to harder ground when it **kept up with the command**, and
    not before it has spent long enough on the one it is on.

    On generated terrain this is what makes rough ground trainable: a robot that
    copes is moved to a harder row, one that does not is moved to an easier one,
    so nobody spends the run stuck on a slope they cannot climb. Without it the
    initial assignment is where they stay.

    ## Why not upstream's rule

    `terrain_levels_vel` promotes on `distance > terrain size / 2` -- an absolute
    number of metres, derived from the tile. That works for a quadruped, which
    falls over when the ground defeats it and then plainly has not travelled. **A
    hexapod is statically stable and does not fall**, so it keeps shuffling
    forward on terrain it is not actually handling, and an absolute distance bar
    is met either way.

    Measured when `scenes/rough.py` used 3.02 m tiles, which put that bar at
    1.51 m: a robot tracking its command at 40% still covers 2.76 m in an episode
    and promotes. Every level, every episode, until it runs out of levels. The
    demotion branch never fires at all -- it is masked by `~move_up`, and `move_up`
    is effectively always true.

    **The tile has since grown to 12 m, and that does not rescue the rule, it
    breaks it the other way.** The bar is now 6.0 m, and the displacement table in
    `scenes/rough.py` puts a robot at the 0.8 promotion standard at 5.50 m in a
    20 s episode -- so upstream's rule would go from promoting everyone every
    episode to promoting nobody ever, on the same policies, because a layout
    dimension moved. Neither reading has anything to do with whether the robot
    handled the ground, which is the objection; the sign of the error is incidental.

    So the bar here is **tracking error, on both axes**, averaged over the whole
    episode:

        err_xy  = metrics["error_vel_xy"]  * max_command_step / episode_length_buf
        err_yaw = metrics["error_vel_yaw"] * max_command_step / episode_length_buf

    and a level is earned only when *both* are under `up_ratio` of the range
    currently being commanded. Turning matters as much as travelling: a robot told
    to turn on the spot is asked for no displacement at all, so a
    distance-based rule scores it as a failure or ignores it, and on a slope
    turning is the harder of the two.

    Error rather than displacement also drops three approximations that a
    distance rule cannot avoid. It does not care that the command **resamples**
    every 3 to 8 s, because it is accumulated per step against whatever command
    was live at the time. It does not under-count a **curved path**, which
    straight-line displacement does. And it needs no special case for a
    **near-zero command**, where "distance asked for" is a division by almost
    nothing.

    The bar is relative to the command range rather than absolute, so it stays
    meaningful as `CommandRangeCurriculum` widens that range underneath it. For
    orientation: a robot that never moves scores a mean error of about 0.71 of the
    range, measured -- for a motionless robot the error simply *is* the command --
    so `up_ratio` of 0.25 asks the error down to about a third of that, on top of
    a floor for the part of the error that does not scale with the range at all
    (`lin_floor` / `ang_floor`, and the module docstring for why).

    ## The dwell, and why only promotion waits

    ## Order: commands first, then terrain

    Promotion is blocked entirely until `CommandRangeCurriculum` has reached its
    last level. Two curricula climbing together make each other unreadable -- a
    tracking error that rises could be the wider command range or the rougher
    ground, and each curriculum would react to the other's difficulty as if it
    were its own.

    It also collapses the flat-then-rough plan into a single run: there is no flat
    stage to train and resume from, because the terrain does not move until the
    commands have stopped moving.

    ## The dwell

    A level is also held for `dwell_episodes` before it can be left upward. Meeting
    the ratio once is a single episode on one tile of one difficulty; it says the
    robot got across, not that it has learned the ground. The dwell is per
    environment, because terrain levels are per environment -- each robot counts
    its own episodes, and the count is zeroed whenever its level changes.

    At roughly 41 iterations per episode (1000 control steps of 24 per iteration),
    12 episodes is about 500 iterations on a level.

    **Demotion is not gated.** The dwell exists to stop the population racing
    upward, and applying it downward would do something else entirely: hold a
    robot on terrain it has already shown it cannot cross, for hundreds of
    iterations, collecting nothing. A minimum time at a level is worth having; a
    minimum time stuck is not.

    A class rather than a function because the timer has to survive between calls;
    mjlab instantiates it once with ``(cfg, env)`` and then calls it like one.
    """

    def __init__(self, cfg, env) -> None:
        del cfg, env
        #: Episodes finished at the current level, per environment.
        self._episodes_here = None

    @staticmethod
    def _command_level(env) -> tuple[int, int] | None:
        """`(level, top)` of the command curriculum, or None if there is not one.

        Found by identity rather than by term name: the manager replaces a class
        term's `func` with the instance, so the object is right there, and looking
        for the type cannot be broken by renaming the term in a config.

        The top is read from **that term's own ladder**, the same one its
        constructor clamps against. Every task that climbs one appends rungs to
        the shared shape, so `len(LEVELS) - 1` is 2 where theirs is 3 (or 5, for
        `jumper.posture`), and it let terrain start climbing before the last
        command rung -- the two curricula moving at once that this class forbids.
        """
        cm = getattr(env, "curriculum_manager", None)
        for cfg in getattr(cm, "_term_cfgs", None) or ():
            if isinstance(getattr(cfg, "func", None), CommandRangeCurriculum):
                levels = (getattr(cfg, "params", None) or {}).get("levels", LEVELS)
                return cfg.func.level, len(levels) - 1
        return None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        command_name: str = "twist",
        up_ratio: float = 0.25,
        down_ratio: float = 0.5,
        lin_floor: float = 0.03,
        ang_floor: float = 0.18,
        dwell_episodes: int = 12,
        min_steps: int = 50,
        after_commands: bool = True,
    ) -> dict:
        terrain = getattr(env.scene, "terrain", None)
        if terrain is None or getattr(terrain.cfg, "terrain_generator", None) is None:
            return {}

        if self._episodes_here is None:
            self._episodes_here = torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            )
        # This function is called once per reset batch, so every environment in
        # `env_ids` has just finished an episode at its current level.
        self._episodes_here[env_ids] += 1

        term = env.command_manager.get_term(command_name)
        ranges = term.cfg.ranges
        # Recovered exactly from mjlab's accumulator, which adds
        # `err / max_command_step` every step; see CommandRangeCurriculum, which
        # uses the same recovery and validates it against a direct sum.
        max_command_step = term.cfg.resampling_time_range[1] / env.step_dt
        steps = env.episode_length_buf[env_ids].clamp(min=1).float()
        err_xy = term.metrics["error_vel_xy"][env_ids] * max_command_step / steps
        err_yaw = term.metrics["error_vel_yaw"][env_ids] * max_command_step / steps

        # Both bars carry the same floor `CommandRangeCurriculum` does, for the
        # same reason and from the same measurement -- see the module docstring.
        # It is not optional here either: measured at the top command level, where
        # this term is the only one still gating, the linear bar was 0.125 against
        # an error of 0.1435, so terrain could not have promoted **even after the
        # command curriculum finished**.
        #
        # `lin_floor` is the well-supported one: 0.033 and 0.027 of exploration
        # noise measured at the two ends of the range, against 0.0113 of pure gait
        # ripple with the command at exactly zero.
        #
        # `ang_floor` **was 0.05, and that made this level unclearable.** It was
        # taken from a standing reading of 0.0417 -- ripple with no exploration
        # component -- because every other yaw number available at the time came
        # from a policy trained against a yaw reward that returned zero. The note
        # here asked for a re-measurement once a policy had trained with yaw
        # tracking alive. It has, and this is that measurement:
        #
        #   tools/checks/yaw_floor.py, jumper.tripod model_15800 of
        #   logs/jumper/jumper.tripod/2026-09-10_16-10-18, 32 envs x 200 steps, cpu
        #
        #     command   policy          lin err (m/s)   ang err (rad/s)
        #     standing  deterministic       0.0082           0.0265
        #     standing  stochastic          0.0549           0.2418
        #     moving    deterministic       0.0336           0.0951
        #     moving    stochastic          0.0674           0.2705   <- what this sees
        #
        # **The policy tracks yaw to 0.0951.** It is more than twice as good as the
        # 0.2375 bar it was failing. What it is judged on is the stochastic row,
        # and 0.1755 of that is exploration noise -- 74% of the old bar spent on
        # something no amount of learning removes, against 22% on the linear axis
        # where the floor covers it. That is the 0.15 rung's failure again, in the
        # other axis: measured on the run above, terrain reached level 3.6 by
        # iteration 9737 and was drifting *down* by 15697, with `settled` at 0.982
        # and `promoted` at 0.0002.
        #
        # 0.18 is the moving-case exploration component, chosen the same way 0.03
        # was: cover the noise and nothing else. It leaves 0.2705 against a bar of
        # 0.3675, a 26% margin where linear has 56%, and the demotion bar at
        # 0.555 stays far above anything measured.
        #
        # Two things it does not fix. The floor is now 49% of the bar (the note on
        # `LEVELS` calls 40% acceptable and 67% not), and the exploration noise is
        # 23% of the full yaw command range, which is a fact about the policy's
        # action distribution rather than about this constant. Re-measure at 512
        # envs when a GPU is free; 32 was to stay out of a running job's memory.
        lin_bar = lin_floor + up_ratio * ranges.lin_vel_x[1]
        ang_bar = ang_floor + up_ratio * ranges.ang_vel_z[1]
        # An episode too short to have shown anything -- a robot that fell over in
        # the first second -- is neither promoted nor demoted on that evidence.
        long_enough = env.episode_length_buf[env_ids] >= min_steps
        tracked = (err_xy < lin_bar) & (err_yaw < ang_bar)
        # The demotion bar carries the floor too. Without it, "coping badly enough
        # to go back a level" would be judged against noise the robot cannot remove
        # -- and it was: `demoted` sat at 1.0 for the whole run, every environment
        # every episode, while the robot was in fact tracking at half of what
        # standing still would have scored.
        lost = (err_xy > lin_floor + down_ratio * ranges.lin_vel_x[1]) | (
            err_yaw > ang_floor + down_ratio * ranges.ang_vel_z[1]
        )

        # Long enough on this level to have learned it, not just crossed it once.
        # Promotion only: see the note in the class docstring.
        #
        # **Counted in episodes at this level, not in global steps.** The obvious
        # implementation compares `common_step_counter` against when the level
        # last changed, and that version measurably did not hold under `--resume`:
        # isolated to a two-iteration run, `settled` was 0.00 without `--resume`
        # and 1.00 with it, on the same code and the same checkpoint. An in-process
        # probe showed the counter at 0 and the comparison returning False, so what
        # the log recorded and what the function computed disagreed and the reason
        # was never established.
        #
        # This counts something the term owns outright: one increment per reset
        # batch, zeroed when the level changes. It cannot disagree with the env
        # about where the run started, because it never asks.
        settled = self._episodes_here[env_ids] >= dwell_episodes

        # **Terrain waits for the command curriculum.** Two curricula climbing at
        # once makes each one's evidence ambiguous: a tracking error that rises
        # could be the wider command range or the rougher ground, and neither
        # curriculum can tell which, so both react to the other's difficulty. Held
        # in order, the policy first learns the full command envelope on ground it
        # can handle, and only then is asked to keep it on worse ground.
        #
        # It also makes the two-stage plan one run. There is no flat stage to
        # train and then resume from -- the terrain simply does not move until the
        # commands stop moving.
        cmd = self._command_level(env) if after_commands else None
        commands_maxed = cmd is None or cmd[0] >= cmd[1]

        move_up = long_enough & settled & tracked
        if not commands_maxed:
            move_up = torch.zeros_like(move_up)
        move_down = long_enough & lost & ~move_up

        # The first reset happens before any stepping, so the accumulators are
        # empty. Upstream guards the same moment for the same reason: without it
        # every environment leaves the level `max_init_terrain_level` placed it on.
        if env.common_step_counter == 0:
            move_up = torch.zeros_like(move_up)
            move_down = torch.zeros_like(move_down)

        changed = move_up | move_down
        if bool(changed.any()):
            self._episodes_here[env_ids[changed]] = 0

        terrain.update_env_origins(env_ids, move_up, move_down)

        levels = terrain.terrain_levels.float()
        # Key names are shared with `CommandRangeCurriculum`; see the note on
        # naming at the top of this module. **Each error is logged beside the bar
        # it is being judged against**, on both axes, so a stall is legible from
        # the dashboard alone: which axis is failing, and by how much. Logging the
        # error without the bar is what made the last stall take a source dive --
        # the two numbers existed, under different names, in different groups.
        return {
            "level": float(levels.mean()),
            "level_max": float(levels.max()),
            "lin_err": float(err_xy.mean()),
            "lin_err_bar": float(lin_bar),
            "ang_err": float(err_yaw.mean()),
            "ang_err_bar": float(ang_bar),
            "settled": float(settled.float().mean()),
            "promoted": float(move_up.float().mean()),
            "demoted": float(move_down.float().mean()),
            # Why promotion is or is not available, without cross-referencing the
            # other curriculum's log.
            "commands_maxed": float(commands_maxed),
        }
