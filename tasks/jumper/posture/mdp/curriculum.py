"""The posture command's range curriculum: three rungs, a precision rung, and the ruler held.

The same object `common/mdp/curriculum.py::CommandRangeCurriculum` is for the
velocity command, written against this command instead, and it inherits that
module's two properties deliberately:

**The level advances on measured tracking, not on a step schedule.** mjlab's
`commands_vel` widens on a timer whether or not the policy learned anything;
what makes a curriculum a curriculum is the gate.

**`std` moves with the range, so the ruler never changes length.** The tracking
reward is `exp(-err^2 / std^2)`, so narrowing the range without narrowing `std`
does not make the task easier to *do*, it makes it easier to *score* -- measured
on the velocity command, holding `std` fixed let a robot that never moved nearly
treble its free score as the range narrowed. Here the same thing would pay a
robot that never leans, on a task whose whole objective is leaning. Every std
below is `POSTURE_STD_RATIO` times that level's own half-range, which is
`STD_LIN_RATIO`'s rule applied to four more axes.

## The rungs

    level   twist / pitch / roll      height            std: angle / height
      0           +/- 7.5 deg     0.0885 - 0.1285 m      3.75 deg / 10.0 mm
      1           +/- 10 deg      0.0823 - 0.1357 m      5.00 deg / 13.3 mm
      2           +/- 15 deg      0.0700 - 0.1500 m      7.50 deg / 20.0 mm
      3           +/- 15 deg      0.0700 - 0.1500 m      3.75 deg / 10.0 mm   precision

The angles in that table are the **walking** band, and they are the ruler: every
std and every bar in this module is taken from them. The standing band climbs the
same fractions of its own half-ranges (`band_levels`) -- twist 15, 20, 30
degrees, pitch 10, 13.3, 20, roll 7.5, 10, 15 -- so a parked robot at level `k` is
asked for the same share of what it can do standing as a walking one is of what
it can do walking. Held to the walking band's ruler, a standing axis is marked no
more loosely for being asked for more.

The height follows the *same fractions* of its own excursion either side of the
standing height -- a half, two thirds, all of it. One level is one rung on every
axis, for the reason `ANG_LEVELS` gives for the velocity pair: two ladders
climbing at different rates make `Curriculum/posture/level` mean nothing.

The first rung was 5 degrees and was widened; `POSTURE_LEVEL_FRACTIONS` has the
measurement that moved it.

The height's excursion is slightly asymmetric -- 37 mm of squat against 43 mm of
extension, because the robot stands at 0.107 and the two ends were each taken
from what it was measured to hold -- so its rungs are asymmetric too. Scaling both
sides by the same fraction is what keeps "level 1" the same *fraction of
capability* on the height axis as on the angle axes.

**Level 3 is a precision rung**, as the velocity ladder ends with one: the top
range again, marked by half the std. It was left out until something had
measured how much tighter this command could be marked, and `POSTURE_STD_SCALES`
has the measurement that put it in -- in short, the policy holding level 2 falls
short of every command by a fixed *fraction*, 16-21% of a twist and up to half of
a small roll, which is a tracking term losing a tug of war rather than a robot at
its limit. The rung is the same range, so it asks nothing new of the gait; it
changes only how much a shortfall costs.

## The bar, and the measurement it comes from

    bar = floor + ratio * (that level's half-range)

and the floor is the part that does not scale, because the thing it pays for does
not either. Getting this wrong in the other direction is recorded in
`common/mdp/curriculum.py`: a purely proportional bar made the *first* rung the
hardest one, and a 37500-iteration run sat at level 0 for its entire life.

Measured on this task before any policy existed -- 128 environments, every one
commanded the neutral posture, random actions, 350 steps after the drop settles,
mean absolute error per axis:

    action sigma   |twist|   |pitch|   |roll|    |height - 0.107|
        0.0         0.127     0.269     0.144        3.57 mm
        0.4         0.573     0.731     0.525        3.73 mm
        0.8         1.137     1.542     1.136        5.02 mm
                    (degrees)

Two things in that table, and they are read differently.

**The sigma-0 row is bias, not floor.** A robot holding HOME with zero actions
sags about 3.6 mm under its own weight, and shows a fraction of a degree of tilt
from the model's own asymmetry. A policy removes all of that by pushing up
slightly; it is an error the curriculum is entitled to ask for.

**The floor is what the noise adds on top**, sigma 0.8 minus sigma 0: about 1.0
degrees of twist, 1.3 of pitch, 1.0 of roll and 1.4 mm of height. That is PPO's
exploration, which no policy can drive out of the metric while it is still
exploring, and it is what `GATE_FLOOR_*` are set from.

**A robot that ignores the command entirely sits at half the range**, which is
what the bar has to beat: for a uniform draw over `+/- r` the mean absolute error
of holding neutral is `r / 2`. So at level 0 the numbers are

    level   axis     never-tries   bar = floor + 0.20 * half-range
      0     angle      3.75 deg    2.50 deg     67% of never-tries
            height    10.05 mm     5.50 mm      55%
      1     angle      5.00 deg    3.00 deg     60%
            height    13.41 mm     6.83 mm      51%
      2     angle      7.50 deg    4.00 deg     53%   (gates the step to the
            height    20.11 mm     9.50 mm      47%    precision rung)
      3     the same range and so the same bar, which never gates anything:
            it is the top

Every bar sits above the noise -- 1.3 degrees and 1.4 mm -- and below what
ignoring the command gives, which is the window a gate has to land in. Outside it
in either direction the level measures nothing: below the floor no policy can
clear it, above never-tries it promotes without anything having been learned.
`tests/test_posture.py` checks that window rung by rung, because it is the
property that decides whether this is a curriculum or a timer.

**The first rung used to be 5 degrees and did not have that window**: its bar
was 2.00 degrees against 2.50 of never-trying and 1.3 of noise, so most of what
it measured was PPO's exploration. That was the reading to make if level 0
stalled, and it was made before the first run rather than after it.

If a level still refuses to move, `Curriculum/posture/<axis>_err` against
`<axis>_err_bar` is what says which axis is holding it and by how much.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ...common.mdp.curriculum import CheckpointedLadder
from ..rl_cfg import NUM_STEPS_PER_ENV

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

#: The fractions of the full command range each rung asks for. Applied to the
#: angles and to the height alike, so one level is one rung everywhere.
#:
#: Written as fractions rather than as angles so that the ceilings
#: (`env_cfg.MOVE_LEAN` and the standing band's `STAND_*` beside it) are the only
#: place the scale lives -- the same relation `LEVEL_FRACTIONS` has to a task's
#: speed ceiling.
#:
#: **The rungs are 7.5, 10 and 15 degrees, not the even 5, 10, 15 this started
#: as, and the first one is why.** A 5 degree rung asks for a mean error
#: of 2.5 degrees from a robot that ignores the command, against 1.3 degrees of
#: measured exploration noise -- so roughly half of what the bar was measuring
#: was PPO's action sigma rather than the policy. That is the same fault the
#: velocity ladder found in its own 0.15 rung, where "two thirds of the bar is
#: exploration noise is not measuring the policy", and the answer there was the
#: same: widen the rung rather than loosen the gate.
#:
#: At 7.5 degrees the first bar is 2.5 degrees against 3.75 of never-trying and
#: 1.3 of noise, which is a window rather than a coin toss.
#:
#: **Only the first rung moved**; 10 and 15 are where they were. That leaves an
#: uneven ladder -- 2.5 degrees to the second rung and 5 to the third -- which is
#: deliberate and has precedent: `ANG_LEVEL_FRACTIONS` is `(0.4, 2/3, 1.0)` for
#: the same kind of reason, a rung placed by what was measured rather than by
#: arithmetic. Even spacing is not the property that matters; what matters is
#: that each rung's bar sits between the noise and what ignoring the command
#: gives, and `tests/test_posture.py` checks exactly that, rung by rung.
#:
#: **The last 1.0 is the precision rung**: the full range once more, marked by
#: `POSTURE_STD_SCALES`' tighter std rather than by a wider command.
POSTURE_LEVEL_FRACTIONS: tuple[float, ...] = (0.5, 2.0 / 3.0, 1.0, 1.0)

#: The std multiplier on the precision rung, both the angles and the height.
#:
#: **Measured, on `2026-09-28_20-40-55/model_70450`** -- the deterministic policy
#: after 23000 iterations at level 2, 16 environments, each command held 3.5 s
#: and read over the last 1.5. The error is command minus measured:
#:
#:     parked                 short by                of the command
#:     twist  10 / 20 / 30    1.6 / 3.4 / 6.2 deg     16 / 17 / 21 %
#:     roll    5 / 10 / 15    2.5 / 1.8 / 3.5 deg     50 / 18 / 23 %
#:     pitch  +/-7 to 20      under 1 deg, except 2.2 at 7 nose-up
#:     height 70 / 85 mm      8.3 / 4.8 mm high       the squat not reached
#:     height 150 mm          4.9 mm low, with 3.1 deg of pitch alongside
#:
#:     walking, 0.3-0.6 m/s at the moving band's 15 degrees
#:     twist / pitch / roll   1.6-2.0 / 2.6-3.4 / 2.9-3.9 deg short
#:     height +23 mm          6.8 mm short
#:     ripple                 0.2-1.0 deg and 0.6-2.7 mm, the gait's own
#:
#: **A shortfall proportional to the command** is a tracking term being outpulled
#: -- by `pose`, which charges every joint a posture moves -- not a robot at the
#: edge of what it can do: the ends of the range are held, just not all the way.
#: At the level-2 std a 30-degree twist 6.2 degrees short still scores 0.50, and
#: the policy settled there. Halving the std makes that 0.06.
#:
#: **The ripple is why this is safe while walking**, and it was the open question:
#: `track_twist` warns that a std tight enough to resolve a degree would charge the
#: gait's own ripple, which nothing can remove. Measured at 0.2-1.0 degrees, it
#: costs at most 1 - exp(-(1.0 / 3.75)^2) = 0.07 a step at the halved std, so the
#: rung is on both bands rather than parked only.
#:
#: **What it is not for is the twist near zero.** Training averages there are 1.2
#: degrees against the 1.0 of exploration noise (`GATE_FLOOR_ANGLE`); halving the
#: std charges that noise 0.10 rather than 0.03, and that is the cost this pays
#: for the ends of the range. If `Policy/mean_std` climbs once it is on, that is
#: the same tax-on-noise loop `rl_cfg.py` records for `action_rate_l2`, and the
#: first thing to try is 0.7 rather than 0.5.
PRECISION_STD_SCALE = 0.5
POSTURE_STD_SCALES: tuple[float, ...] = (
    *(1.0,) * (len(POSTURE_LEVEL_FRACTIONS) - 1),
    PRECISION_STD_SCALE,
)

#: std as a fraction of a level's half-range. `STD_LIN_RATIO`'s 0.5, named here
#: rather than imported so that this task's ruler is its own decision -- but it is
#: the same number, and deliberately: the calibration behind it ("standing still
#: collects the same 0.14 a Go1 does") is about the shape of `exp(-err^2/std^2)`
#: against a uniform command, which is exactly the situation here.
POSTURE_STD_RATIO = 0.5

#: The part of the promotion bar that does not scale with the range, in radians.
#: 1.0 degrees -- the exploration-noise component measured in the module
#: docstring, taken at the noisiest axis's neighbour rather than at pitch's 1.3,
#: because pitch is also the axis with the most static bias in the same table and
#: a floor set at the largest number would be forgiving the bias too.
GATE_FLOOR_ANGLE = math.radians(1.0)

#: The same for the height, in metres. 1.5 mm, the measured noise component --
#: **not** the 3.6 mm a passive robot sags, which is a bias a policy can remove.
GATE_FLOOR_HEIGHT = 0.0015

#: The part of the bar that scales. A robot ignoring the command sits at half the
#: range, so 0.20 asks for the error to be cut to 40% of that before the floor is
#: added. The velocity curriculum's 0.3 against its own "never moves" figure of
#: 0.71 * range is the same ask -- 42% -- and this is that number carried over
#: rather than a second opinion.
GATE_RATIO = 0.20

#: Minimum environment steps between promotions, and **the number that actually
#: paces this ladder**.
#:
#: 25 iterations, against the 100 the velocity curriculum uses. That coupling to
#: `rl_cfg.py` is real and is inherited from `CommandRangeCurriculum`: change
#: `num_steps_per_env` and a dwell counted in environment steps changes in
#: iterations without anything saying so.
#:
#: **It happened.** The 200 Hz change doubled the rollout to 48 and left this at
#: a literal `24 * 25`, so the dwell became 12.5 iterations -- half the measured
#: time-to-clear below, which is the one number it is calibrated against. The
#: warning in the paragraph above was already written and was not enough, so the
#: literal is gone and the coupling is expressed instead. `tests/test_posture.py`
#: asserts the iteration count rather than the step count, so either factor
#: moving alone now fails.
#:
#: **Measured, on the run of 2026-09-16 that this task's first 6000 iterations
#: came from.** Both promotions fired at exactly the dwell minimum -- level 1 at
#: iteration 100, level 2 at 200 -- so the gate never decided anything and the
#: ladder was a 300-iteration schedule wearing a curriculum's name.
#:
#: The obvious reading is that the bars are too loose, and it is wrong. Measured
#: against a robot running **zero actions** at level 0, which is what "ignored the
#: command" means:
#:
#:     axis     bar        zero-action error   of the bar
#:     twist    2.50 deg       2.94 deg          118%
#:     pitch    2.50 deg       2.90 deg          116%
#:     roll     2.50 deg       3.12 deg          125%
#:     height   5.50 mm        8.64 mm           157%
#:
#: Every bar is below what doing nothing gives, so clearing one is not free. What
#: is actually happening is that **posture tracking is learnt in tens of
#: iterations** -- it is close to a static map from joint offsets to body pose --
#: and the dwell was five times longer than that. The pacing constraint was the
#: clock, not the skill.
#:
#: So this moves and the bars do not. Tightening them is the change that looks
#: equivalent and is not: the top rung's converged errors are 1.3 / 1.6 / 1.9
#: degrees against a measured exploration-noise floor of 1.3, so a bar tight
#: enough to bite there would be measuring PPO's action sigma. That is the failure
#: `common/mdp/curriculum.py` records costing a run all 37500 of its iterations.
#:
#: 25 is the measured time-to-clear, so the ladder now takes about 75 iterations
#: and a rung the policy has *not* cleared still holds it. If promotions start
#: firing at 25 again -- the same signature as before, one level per dwell -- the
#: next question is whether the ladder is worth keeping at all, not whether the
#: dwell should be 10.
#:
#: **One episode is 1000 steps, which is 42 iterations**, so at 25 the first
#: promotion decision is made before every environment has reset once. That is
#: thin, and it is the cost being accepted here: `min_samples` keeps the estimate
#: from being built on two or three environments, and `ema_alpha` keeps one
#: unlucky batch from carrying it, but neither makes a half-episode sample into a
#: full one. It is the number to raise first if promotions start looking random.
DWELL_STEPS = NUM_STEPS_PER_ENV * 25

#: No promotion at all before this many iterations, on either ladder.
#:
#: `DWELL_STEPS`'s note above accepts a third of an episode as the evidence
#: behind the *first* promotion decision. That was the wrong trade and this is
#: the retraction: at 200 Hz an episode is 83 iterations, and both ladders were
#: promoting before one had finished anywhere in the field.
#:
#: What the run actually did, at a dwell of 25 iterations for posture and 50 for
#: the velocity ladder -- 2400 environment steps is 50 iterations at this task's
#: 48-step rollout, not the 100 it is at the locomotion tasks' 24:
#:
#:     iteration 25   posture level 0 -> 1
#:     iteration 49   command level 0 -> 1, on `lin_err` 0.131 against a 0.1515
#:                    bar, while `lin_err` and `ang_err` were both still rising
#:
#: The promotions were legal. The evidence was not: it came from the environments
#: that had *ended early*, because those are the only ones that had ended at all,
#: and an environment that terminates in its first second contributes a small
#: accumulated error that reads exactly like good tracking. `min_steps` removes
#: the worst of them, but a curriculum whose only samples are early terminations
#: is measuring the failure mode and calling it skill.
#:
#: 200 iterations is about two and a half episodes, so every ladder's first
#: decision is made on environments that have run at least one to the end. It is
#: a floor under the first promotion and nothing else -- the dwell still paces
#: everything after it, and the bars are unchanged.
#:
#: **A resumed run is not guarded, and that is the intended behaviour.** This is
#: counted in `common_step_counter`, which `mjlab/rl/runner.py` restores on
#: `--resume` precisely "to preserve curricula state" -- so a resumed run is past
#: the floor on its first iteration, and its level comes back from the checkpoint
#: (`common/runner.py`). What the floor is protecting against is a curriculum whose only samples
#: come from environments that *terminated early*, and a resumed policy already
#: runs episodes to the timeout. The guard is for a cold start.
#:
#: **This was written once and lost.** It landed on the 200 Hz branch that was
#: rolled back to `cbdf70c`, and came back only when the operator noticed the
#: posture ladder promoting at iteration 25 and remembered that it should not.
WARMUP_ITERATIONS = 200
WARMUP_STEPS = WARMUP_ITERATIONS * NUM_STEPS_PER_ENV

#: The environment key that sets the posture level by hand. Same role as
#: `MJRL_COMMAND_LEVEL` for the velocity ladder: the checkpoint carries the level
#: (`CheckpointedLadder`), and this overrides it -- or stands in for it, on a
#: checkpoint written before levels were saved, where a resumed policy that
#: already holds 15 degrees would otherwise be dropped back to 5 and have to
#: re-earn every rung, one dwell each.
POSTURE_LEVEL_ENV = "MJRL_POSTURE_LEVEL"


def posture_levels(
    max_lean: float, height_range: tuple[float, float], neutral_height: float
):
    """The ladder for a task's ceiling: `(angles, heights)`.

    `angles` is one half-range per rung, in radians. `heights` is one `(low,
    high)` per rung, each side of `neutral_height` scaled by the same fraction --
    which is what makes the asymmetric height excursion climb at the same rate as
    the symmetric angles.
    """
    low, high = height_range
    angles = tuple(max_lean * f for f in POSTURE_LEVEL_FRACTIONS)
    heights = tuple(
        (
            neutral_height - (neutral_height - low) * f,
            neutral_height + (high - neutral_height) * f,
        )
        for f in POSTURE_LEVEL_FRACTIONS
    )
    return angles, heights



def band_levels(half_ranges: dict[str, float]) -> dict[str, tuple[float, ...]]:
    """One ladder per axis, for a band whose half-range differs by axis.

    The standing band is 30 degrees of twist, 20 of pitch and 15 of roll, so it
    cannot share `posture_levels`' single `angles` tuple. The fractions are the
    same ones, which is what makes level `k` the same share of both bands.
    """
    return {
        axis: tuple(half * f for f in POSTURE_LEVEL_FRACTIONS)
        for axis, half in half_ranges.items()
    }

def angle_std(
    level: int,
    angles: tuple[float, ...],
    scales: tuple[float, ...] | None = None,
    ratio: float = POSTURE_STD_RATIO,
) -> float:
    """The twist / tilt tracking std at `level`.

    A function because two places need it and must not drift -- this curriculum
    writes it into the live reward manager, and `env_cfg.py` sets the config
    default from the top rung. The config default is what `play.py` replays with,
    since it builds with `curriculum = {}`, so a second copy of `ratio * range`
    anywhere is a silent mismatch between what training ended on and what a replay
    shows. That is the failure `lin_std` exists to prevent for the velocity pair.
    """
    scale = 1.0 if scales is None else scales[level]
    return ratio * angles[level] * scale


def height_std(
    level: int,
    heights: tuple[tuple[float, float], ...],
    scales: tuple[float, ...] | None = None,
    ratio: float = POSTURE_STD_RATIO,
) -> float:
    """The height tracking std at `level`. See `angle_std`.

    The half-range is `(high - low) / 2` rather than either side on its own: the
    rung is asymmetric about the standing height, and a std taken from the larger
    side would mark the squat and the extension by different rulers.
    """
    low, high = heights[level]
    scale = 1.0 if scales is None else scales[level]
    return ratio * (high - low) / 2.0 * scale


class PostureRangeCurriculum(CheckpointedLadder):
    """Widen the posture command as the posture is tracked, scaling `std` with it.

    A class rather than a function because the level, the error averages and the
    promotion time have to survive between calls; mjlab instantiates it once with
    ``(cfg, env)`` and then calls it like a function.
    """

    LEVEL_ENV = POSTURE_LEVEL_ENV

    #: The command's four channels, in order, paired with the metric each one is
    #: judged by. The metric names are `PostureCommand`'s own.
    _AXES = ("twist", "pitch", "roll", "height")

    def __init__(self, cfg: "CurriculumTermCfg", env: "ManagerBasedRlEnv") -> None:
        del env
        params = getattr(cfg, "params", None) or {}
        top = len(params["angles"]) - 1
        # No `start_level` param, as the velocity ladder has none: it would also
        # reach `__call__`, which does not take it, and raise TypeError on the
        # first reset.
        self.level = self._level_from_env(top)
        self.err_ema: dict[str, float | None] = {a: None for a in self._AXES}
        # Samples pooled across resets until `min_samples` have arrived.
        self._pending: dict[str, float] = {a: 0.0 for a in self._AXES}
        self._pending_n: int = 0
        self.last_promotion_step = 0
        self._applied = False

    def _apply(
        self,
        env: "ManagerBasedRlEnv",
        command_name: str,
        angles: tuple[float, ...],
        heights: tuple[tuple[float, float], ...],
        stand_angles: dict[str, tuple[float, ...]],
        rewards: tuple[str, str, str],
        std_scales: tuple[float, ...] | None,
        std_ratio: float,
    ) -> None:
        """Write this level's ranges and stds into the live managers.

        `angles` is the **walking** band's ladder and the ruler every std and bar
        is taken from; `stand_angles` climbs the standing band by the same
        fractions, one ladder per axis, so level `k` is the same share of both.

        **Through the managers, never through `env.cfg`.** `RewardManager.__init__`
        deep-copies its config, so assigning to `env.cfg.rewards[...]` changes an
        object nothing reads. The command term is the other way round -- it keeps
        its own `cfg` and re-reads `cfg.ranges` at every resample -- so that one is
        written directly. Getting either wrong is silent: training simply
        continues at the range or the std that was already there.
        """
        term_cfg = env.command_manager.get_term(command_name).cfg
        for axis in ("twist", "pitch", "roll"):
            setattr(term_cfg.moving, axis, angles[self.level])
            setattr(term_cfg.ranges, axis, stand_angles[axis][self.level])
        term_cfg.ranges.height = heights[self.level]

        twist_reward, tilt_reward, height_reward = rewards
        a_std = angle_std(self.level, angles, std_scales, std_ratio)
        for name in (twist_reward, tilt_reward):
            env.reward_manager.get_term_cfg(name).params["std"] = a_std
        env.reward_manager.get_term_cfg(height_reward).params["std"] = height_std(
            self.level, heights, std_scales, std_ratio
        )

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor,
        command_name: str,
        angles: tuple[float, ...],
        heights: tuple[tuple[float, float], ...],
        stand_angles: dict[str, tuple[float, ...]],
        std_scales: tuple[float, ...] | None = None,
        std_ratio: float = POSTURE_STD_RATIO,
        gate_ratio: float = GATE_RATIO,
        gate_floor_angle: float = GATE_FLOOR_ANGLE,
        gate_floor_height: float = GATE_FLOOR_HEIGHT,
        ema_alpha: float = 0.05,
        dwell_steps: int = DWELL_STEPS,
        min_steps: int = 100,
        min_samples: int = 8,
        warmup_steps: int = WARMUP_STEPS,
        rewards: tuple[str, str, str] = ("track_twist", "track_tilt", "track_height"),
    ) -> dict[str, torch.Tensor | float]:
        """Advance at most one level per call; return the state for logging.

        Args:
            stand_angles: the standing band's ladder, one tuple per angle axis and
                each as long as `angles`. It moves the command's `ranges`; the
                bars and stds below stay on `angles`, the walking band, which is
                the ruler every measurement in this module was taken against.
            gate_ratio: the part of the bar that scales with the range, and
                `gate_floor_angle` / `gate_floor_height` the parts that do not.
                All three are measured; see the module docstring.
            ema_alpha: smoothing across reset batches. Each call sees only the
                envs resetting on this step, which is a small and noisy sample.
            dwell_steps: minimum environment steps between promotions. A promotion
                widens the range so the error jumps; without a dwell the level
                could climb again on stale evidence before that shows up. See
                `DWELL_STEPS`: it is 25 iterations rather than the velocity
                ladder's 100, and the measurement that moved it is there.
            min_steps: ignore envs whose episode was shorter than this. An env
                that fell over after a few steps says nothing about posture, and
                its near-zero accumulator would drag the average towards
                promotion. **A duration, so it moved with the control rate**:
                50 steps was 0.5 s at 100 Hz and is 0.25 s at 200, which is a
                fall this would have started counting as evidence.
            warmup_steps: no promotion before this many environment steps,
                whatever the error says. See `WARMUP_STEPS`: a dwell is a gap
                between promotions and this is a floor under the first one, and
                at this task's rollout length the dwell alone put that floor at
                iteration 25 -- before a single environment had finished an
                episode.
            min_samples: ignore a batch with fewer valid envs than this. A fifth
                of environments are commanded the *neutral* posture
                (`rel_neutral_envs`), and a robot simply standing tracks that
                nearly perfectly -- so a batch of one or two that happened to be
                neutral reports an error that is not about the policy at all.
            std_scales: per-level multiplier on the std the ratio produces, `None`
                meaning 1.0 throughout. This is how the precision rung is
                expressed: `POSTURE_STD_SCALES`, which has the measurement.
        """
        assert std_scales is None or len(std_scales) == len(angles), (
            f"std_scales has {len(std_scales)} entries for a {len(angles)}-rung "
            f"ladder"
        )
        assert len(angles) == len(heights), (
            "one level is one rung on every axis; a mismatch would promote the "
            "angles and the height at different rates"
        )
        assert all(len(stand_angles[a]) == len(angles) for a in ("twist", "pitch", "roll")), (
            "the standing band climbs the same rungs as the walking one; a ladder "
            "of another length would put the two bands on different levels"
        )

        if not self._applied:
            # On the first call rather than in `__init__`, which avoids depending
            # on the order the managers are constructed in. `_reset_idx` runs the
            # curriculum before the command manager resamples, so level 0 is in
            # force for the very first commands.
            self._apply(
                env, command_name, angles, heights, stand_angles, rewards,
                std_scales, std_ratio,
            )
            self._applied = True

        command_term = env.command_manager.get_term(command_name)

        # The first reset happens before any stepping: the accumulators and the
        # step counter are both zero, and a zero error reads as perfect tracking.
        if env.common_step_counter > 0 and len(env_ids) > 0:
            steps = env.episode_length_buf[env_ids]
            valid = steps >= min_steps
            n_valid = int(valid.sum())
            # **Pooled across resets, not required within one.** The environments
            # time out one at a time -- `on_policy_runner.learn` randomises their
            # initial episode lengths -- so `min_samples` in a single batch is a
            # coincidence rather than a sample size. See the longer note in
            # `common/mdp/curriculum.py`, and the measurement that found it: 1143
            # iterations of this task at 200 Hz, every batch of size 1, not one
            # sample taken on any of the four axes.
            if n_valid:
                # The metrics are accumulated per step and divided by the longest
                # resampling interval, so this undoes that normalisation against
                # the episode each env actually ran -- the same arithmetic
                # `CommandRangeCurriculum` does for `error_vel_xy`.
                per_step = (
                    command_term.cfg.resampling_time_range[1] / env.step_dt
                ) / steps[valid].float()
                for axis in self._AXES:
                    err = command_term.metrics[f"error_{axis}"][env_ids][valid]
                    self._pending[axis] += float((err * per_step).sum())
                self._pending_n += n_valid
            if self._pending_n >= min_samples:
                for axis in self._AXES:
                    mean_err = self._pending[axis] / self._pending_n
                    previous = self.err_ema[axis]
                    self.err_ema[axis] = (
                        mean_err
                        if previous is None
                        else (1 - ema_alpha) * previous + ema_alpha * mean_err
                    )
                self._pending = {axis: 0.0 for axis in self._AXES}
                self._pending_n = 0

        half_angle = angles[self.level]
        low, high = heights[self.level]
        bars = {
            "twist": gate_floor_angle + gate_ratio * half_angle,
            "pitch": gate_floor_angle + gate_ratio * half_angle,
            "roll": gate_floor_angle + gate_ratio * half_angle,
            "height": gate_floor_height + gate_ratio * (high - low) / 2.0,
        }

        # **Every axis, not the mean of them.** A level the policy has met on
        # three axes and not the fourth is a level the fourth has not trained at,
        # and averaging lets a well-tracked twist carry an untracked height up the
        # ladder -- at which point the height rung it never learned is behind it
        # and the range has widened. The cost is that the slowest axis sets the
        # pace, which is why each error and each bar is logged separately: "why
        # has the level not moved" should be one glance rather than a source dive.
        tracked = all(
            self.err_ema[axis] is not None and self.err_ema[axis] < bars[axis]
            for axis in self._AXES
        )
        settled = env.common_step_counter - self.last_promotion_step >= dwell_steps
        warmed_up = env.common_step_counter >= warmup_steps
        promotable = (
            self.level + 1 < len(angles) and tracked and settled and warmed_up
        )
        if promotable:
            self.level += 1
            self.last_promotion_step = env.common_step_counter
            self._apply(
                env, command_name, angles, heights, stand_angles, rewards,
                std_scales, std_ratio,
            )
            # Start the averages again: they were measured against a narrower
            # range and are not comparable with what follows.
            self.err_ema = {axis: None for axis in self._AXES}
            self._pending = {axis: 0.0 for axis in self._AXES}
            self._pending_n = 0
            half_angle = angles[self.level]
            low, high = heights[self.level]

        out: dict[str, torch.Tensor | float] = {
            "level": float(self.level),
            "angle_range": half_angle,
            "height_low": low,
            "height_high": high,
            "angle_std": angle_std(self.level, angles, std_scales, std_ratio),
            "height_std": height_std(self.level, heights, std_scales, std_ratio),
            "settled": float(settled),
            "warmed_up": float(warmed_up),
            "promoted": float(promotable),
        }
        for axis in self._AXES:
            value = self.err_ema[axis]
            out[f"{axis}_err"] = float("nan") if value is None else value
            out[f"{axis}_err_bar"] = bars[axis]
        return out


__all__ = [
    "WARMUP_ITERATIONS",
    "WARMUP_STEPS",
    "DWELL_STEPS",
    "GATE_FLOOR_ANGLE",
    "GATE_FLOOR_HEIGHT",
    "GATE_RATIO",
    "POSTURE_LEVEL_FRACTIONS",
    "POSTURE_STD_RATIO",
    "POSTURE_STD_SCALES",
    "PRECISION_STD_SCALE",
    "PostureRangeCurriculum",
    "angle_std",
    "height_std",
    "band_levels",
    "posture_levels",
]
