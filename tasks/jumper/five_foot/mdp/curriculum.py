"""The payload curriculum: an empty claw until the robot walks, then a load.

`claw.py::PAYLOAD_RANGE` is a mass the policy has to learn to walk under, and it
cannot be learned from random weights: two cold starts here with it on stood
still and lifted their feet in place (the table beside `PAYLOAD_RANGE`). It used
to be a second stage -- train empty, edit the constant, resume -- and master sat
on the second stage's value from the day the task landed (1b8e81d), so the
README's "from scratch" command was a cold start with the payload on. Nothing
about that raised; it produced a policy that stands.

This makes the two stages one run. The claw is empty until the velocity
curriculum has reached its last rung and the policy tracks there, and then every
environment takes on a load at its next reset.

## The gate, and why it is not the command curriculum's

Three conditions, each logged on its own because which of them is false is the
whole answer to "why is the claw still empty":

    commands_maxed   `CommandRangeCurriculum` is on its last rung
    settled          ... and has been for `dwell_steps`
    lin_err          the episode-mean linear tracking error there, under `lin_err_bar`

**Commands first**, for the reason `TerrainLevels` gives for the same ordering:
two curricula climbing together make each other unreadable. A tracking error
that rises after a promotion could be the wider command range or the added mass,
and each would react to the other's difficulty as if it were its own.

**The bar is its own number, because the command curriculum's is met on
arrival.** Measured on the cold start this was designed against
(`logs/jumper/jumper.five_foot/2026-09-26_11-56-38` in the
five-foot-master-training worktree, `PAYLOAD_RANGE` at (0, 0) per its stored
diff, warp/CUDA at 116k steps/s), the top rung's bar is 0.18 and the error at
the top was:

    iteration   933    1507    3014    4521    6028    7535   10549
    lin_err     (top)  0.164   0.130   0.106   0.096   0.087  0.080

So 0.18 is cleared the moment the policy arrives, while it is still learning to
walk -- which is the state the two failed cold starts were in. And the payload's
own cost, measured on the stage-two run resumed from `model_20300` with
`PAYLOAD_RANGE` at (0, 0.6) (`2026-09-26_16-46-50`): the error at the top rung
went from **0.074 to about 0.105**, and `track_linear_velocity` from 2.48 to
1.85. A policy that already walks gives up about 0.03 m/s to the load.

`lin_err_bar` and `dwell_steps` are the task's (`env_cfg.py`), and the values it
passes are chosen against that table, not measured with this curriculum in
place: on the cold start above they would have switched the payload on at around
iteration 3500 to 4000.

The measure is the command curriculum's own -- `error_vel_xy` recovered to a
per-step mean over the episode that just ended, pooled across resets until
`min_samples` have arrived -- and is only taken **while the commands are
maxed**. An error measured at a narrower range is not comparable with the bar,
and would carry a lower number into the average than the top rung will produce.

## Loading at reset, not at promotion

A promotion does not touch an environment mid-episode. Each one takes on its
payload at its next reset, sampled once from the new rung and held from then on,
which is what `mdp/events.py::claw_payload` has always promised: a load that
changed halfway through an episode would be teaching the policy to expect
something no operator can do.

`on_policy_runner.learn` randomises the initial episode lengths, so resets
arrive a few at a time and the whole population is loaded within one episode
(1000 steps, about 42 iterations). `Curriculum/payload/loaded` is the fraction
that has been. Each batch that loads anything recomputes the model's derived
constants (`set_const`), which is a full pass over every world on warp; it
happens only while `loaded` is below 1, and not at all before the promotion or
after the transition.

## Why the startup event stays, writing zero

`claw_payload` is still registered as a startup event, sampling (0, 0) for the
first rung. It is not decoration: its `requires_model_fields` is what makes
`body_mass` and `body_inertia` **per-world** (`EventManager` collects the fields
and `sim.expand_model_fields` allocates them). A curriculum term declares no
fields, so without the event the writes below would land in a single shared
model field on warp, every environment would carry one mass, and nothing would
say so. So the curriculum looks the event up by name and refuses to run without
it, and it loads through the event's own function and parameters, with only the
range replaced -- one body, one inertia, one sampler.

## Resuming

The level is checkpointed, as the command level is (`CheckpointedLadder`,
`common/runner.py`): `--resume` carries on with the claw loaded if the run it
resumes had loaded it, and every environment takes that payload at its first
reset. Which environment carried what is not saved and does not need to be --
the physics starts every environment at rung 0, and the first reset corrects it.

`MJRL_PAYLOAD_LEVEL` overrides the checkpoint, and is the only source for one
written before levels were saved:

    MJRL_COMMAND_LEVEL=3 MJRL_PAYLOAD_LEVEL=1 python scripts/train.py \\
        --task jumper.five_foot --resume

Such an older checkpoint resumed with neither set starts empty and earns the
payload again -- after the command ladder has re-climbed (which took 200
iterations on the stage-two run, one dwell per rung) and `dwell_steps` has
passed. That is the right path for a policy that was trained empty; for one that
already carries the load it spends the dwell unlearning it.

No demotion, for the family's reason: levels only advance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import RecomputeLevel

from ...common.mdp.curriculum import LEVELS, CheckpointedLadder, CommandRangeCurriculum

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

__all__ = ["PAYLOAD_LEVEL_ENV", "PayloadCurriculum"]

#: The environment key that sets the payload level by hand, over the checkpoint's.
#: Same role as `MJRL_COMMAND_LEVEL` and `MJRL_POSTURE_LEVEL`.
PAYLOAD_LEVEL_ENV = "MJRL_PAYLOAD_LEVEL"


def _command_curriculum(env: ManagerBasedRlEnv) -> tuple[int, int] | None:
    """`(level, top)` of the command curriculum, or None if there is not one.

    Found by identity, as `TerrainLevels` finds it, so renaming the term in a
    config cannot break it. The top is read from **that term's own ladder**
    rather than the shared `LEVELS`: this task appends a precision rung, so its
    top is 3 where the shared ladder's is 2, and judging it against the shared
    one would call the commands maxed a rung early.
    """
    cm = getattr(env, "curriculum_manager", None)
    for cfg in getattr(cm, "_term_cfgs", None) or ():
        if isinstance(getattr(cfg, "func", None), CommandRangeCurriculum):
            levels = (getattr(cfg, "params", None) or {}).get("levels", LEVELS)
            return cfg.func.level, len(levels) - 1
    return None


class PayloadCurriculum(CheckpointedLadder):
    """Load the claw once the policy walks at the top command range.

    A class term because the level, the error average and the per-environment
    record of what each one carries have to survive between calls; mjlab
    instantiates it once with ``(cfg, env)`` and then calls it like a function.
    """

    LEVEL_ENV = PAYLOAD_LEVEL_ENV
    #: `_maxed_since` as well: it is on the same clock as `last_promotion_step`,
    #: and a resume that dropped it would restart the dwell at the top command rung.
    EVIDENCE = ("err_ema", "last_promotion_step", "_maxed_since")

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
        del env
        params = getattr(cfg, "params", None) or {}
        # Only the environment and the checkpoint, not a `start_level` param:
        # every param is also passed to `__call__`, so one read here and not
        # accepted there is a TypeError on the first reset.
        top = len(params["levels"]) - 1
        self.level = self._level_from_env(top)
        self.err_ema: float | None = None
        self._pending: float = 0.0
        self._pending_n: int = 0
        #: The step the commands were first seen maxed, or None while they are not.
        self._maxed_since: int | None = None
        self.last_promotion_step = 0
        #: Per environment, the rung whose payload it carries. The startup event
        #: loaded rung 0 into every one; allocated on the first call, when the
        #: device and the environment count are certain.
        self._carrying: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice | None,
        levels: tuple[tuple[float, float], ...],
        lin_err_bar: float,
        dwell_steps: int,
        event_name: str = "claw_payload",
        command_name: str = "twist",
        ema_alpha: float = 0.05,
        min_steps: int = 50,
        min_samples: int = 8,
    ) -> dict[str, float]:
        """Promote at most one rung, load whoever is resetting; return the state.

        Args:
            levels: one `(low, high)` mass range per rung, in kg. Rung 0 has to be
                the range the startup event samples, because that is what every
                environment carries before this term has loaded anything.
            lin_err_bar: the episode-mean linear tracking error, in m/s, that the
                policy has to get under at the top command rung. Absolute rather
                than a ratio of the range, because it is only ever judged at one
                range.
            dwell_steps: environment steps the commands have to have been maxed
                (or since the last promotion) before a promotion.
            event_name: the startup event whose function and parameters load the
                claw, and whose registration makes the fields per-world.
            ema_alpha, min_steps, min_samples: as `CommandRangeCurriculum`'s.
        """
        # Raises if the event is gone, which is the point: see the module docstring.
        payload = env.event_manager.get_term_cfg(event_name)
        if tuple(payload.params["ranges"]) != tuple(levels[0]):
            raise ValueError(
                f"the startup event samples {payload.params['ranges']} and rung 0 "
                f"is {levels[0]}: every environment starts carrying the event's "
                f"range, so rung 0 has to be it"
            )
        if self._carrying is None:
            self._carrying = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        # `CurriculumManager.compute` turns None into `slice(None)`; the reset paths
        # pass a tensor, but indexing the per-env record by a boolean mask needs one.
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.arange(env.num_envs, device=env.device)[
                slice(None) if env_ids is None else env_ids
            ]

        cmd = _command_curriculum(env)
        commands_maxed = cmd is None or cmd[0] >= cmd[1]
        step = env.common_step_counter
        if not commands_maxed:
            self._maxed_since = None
        elif self._maxed_since is None:
            self._maxed_since = step

        # The same recovery as `CommandRangeCurriculum`, which validates it
        # against a direct per-step sum; only taken at the top command range.
        if commands_maxed and step > 0 and len(env_ids) > 0:
            steps = env.episode_length_buf[env_ids]
            valid = steps >= min_steps
            n_valid = int(valid.sum())
            if n_valid:
                term = env.command_manager.get_term(command_name)
                max_command_step = term.cfg.resampling_time_range[1] / env.step_dt
                err = term.metrics["error_vel_xy"][env_ids][valid]
                self._pending += float((err * max_command_step / steps[valid].float()).sum())
                self._pending_n += n_valid
            if self._pending_n >= min_samples:
                mean_err = self._pending / self._pending_n
                self._pending, self._pending_n = 0.0, 0
                self.err_ema = (
                    mean_err
                    if self.err_ema is None
                    else (1 - ema_alpha) * self.err_ema + ema_alpha * mean_err
                )

        since = max(self._maxed_since or 0, self.last_promotion_step)
        settled = commands_maxed and step - since >= dwell_steps
        tracked = self.err_ema is not None and self.err_ema < lin_err_bar
        promotable = self.level + 1 < len(levels) and commands_maxed and settled and tracked
        if promotable:
            self.level += 1
            self.last_promotion_step = step
            # Measured without the load; not comparable with what follows.
            self.err_ema = None
            self._pending, self._pending_n = 0.0, 0

        # Whoever is resetting now and carries an older rung's payload takes this
        # one. Scoped to `env_ids`: an environment mid-episode keeps what it has.
        due = env_ids[self._carrying[env_ids] != self.level]
        if len(due) > 0:
            params = dict(payload.params)
            params["ranges"] = levels[self.level]
            payload.func(env, due, **params)
            # **The event manager recomputes after an event fires; nothing does
            # after a curriculum term.** Without this the mass lands in the model
            # and `body_subtreemass`, the invweights and the rest stay at their
            # empty-claw values -- a robot whose dynamics disagree with its own
            # inertia, and no error.
            env.sim.recompute_constants(
                getattr(payload.func, "recompute", RecomputeLevel.set_const)
            )
            self._carrying[due] = self.level

        low, high = levels[self.level]
        return {
            "level": float(self.level),
            "mass_low": float(low),
            "mass_high": float(high),
            "loaded": float((self._carrying == self.level).float().mean()),
            "lin_err": float("nan") if self.err_ema is None else self.err_ema,
            "lin_err_bar": float(lin_err_bar),
            "commands_maxed": float(commands_maxed),
            "settled": float(settled),
            "promoted": float(promotable),
        }
