"""Hyper-parameters for jumper.posture.

`jumper.tripod`'s, unchanged, and that is the point rather than laziness: this task
exists to be read against tripod, and a second variable in the optimiser would
make "the posture command cost 20% of the tracking score" an unanswerable
question.

The one thing that is worth reconsidering after a first run is the network width.
The actor gains 4 inputs and the critic 8 on a 470-wide observation, which is
nothing -- but what it is being asked to learn is a policy conditioned on a
four-dimensional posture command on top of a three-dimensional velocity one, and
that is a genuinely larger function class than tripod's. If the run tracks
velocity as well as tripod does and never learns the posture, widen before
touching the reward weights.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

#: Environment steps per policy update.
#:
#: **A rollout is a duration.** 24 steps at 200 Hz is 0.12 s, two thirds of one
#: gait cycle, so an advantage estimate would rarely span a whole stride. 48
#: keeps the 0.24 s the 100 Hz run had.
#:
#: It is a module constant rather than a literal in `agent_cfg` because
#: `mdp/curriculum.py` needs it: every dwell in this task is expressed in
#: *environment* steps, so the number of them in one iteration is what converts a
#: dwell into the iterations its calibration was measured in. Changing this
#: without changing the dwells halves or doubles the curriculum's pace and
#: nothing reports it -- which is exactly what the 24 -> 48 change did before
#: `DWELL_STEPS` was tied to it.
NUM_STEPS_PER_ENV = 48


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config.

    `experiment_name` is the middle segment of the log path: the full structure is
    `logs/<model>/jumper.posture/<date-time>`, with the outermost segment decided by
    `--model`.
    """
    return jumper_ppo_baseline(
        experiment_name="jumper.posture",
        # ── Network ──
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        # ── PPO ──
        # See `jumper.tripod`'s note: this is the knob that matters most for a gait
        # driven by a fixed phase clock. **0.003 rather than its 0.01**, and the
        # measurement is a loop this task turned out to be inside.
        #
        # `action_rate_l2` is the largest penalty here, and on
        # `2026-09-21_08-20-49` its raw quantity tracked the action noise almost
        # exactly -- not the terrain, and not what the policy had learned:
        #
        #     iteration   std    ar raw   ar/std^2   surrogate   mean reward
        #        336     0.583    10.95     32.2      +0.0074       -4.5
        #        504     0.621     9.35     24.3      +0.0060      +31.0  <- peak
        #        840     0.838    18.04     25.7      -0.0003      -57.8
        #       1512     0.794    17.06     27.0      -0.0057      -59.7
        #
        # `ar/std^2` is flat at 24-34 across the run, so **the penalty is a tax on
        # exploration noise**: raw ~= 26 * std^2, and the cost doubling from 11 to
        # 18 came entirely from `std` rising, not from rough ground making the
        # robot move more. That was the first explanation and the ratio refutes it.
        #
        # Which closes a loop: the entropy bonus lifts `std`, the penalty grows as
        # its square, the return falls, the advantages flatten, and the bonus wins
        # more easily. The peak reward sits at iteration 504 -- exactly where
        # `std` bottomed -- and both move together after it. `Loss/surrogate` goes
        # negative from 840 on, so the updates were reducing PPO's own objective.
        #
        # The magnitudes agree: `entropy_coef * Loss/entropy` is 0.01 x 25 = 0.25
        # against a surrogate of 0.005, fifty times larger. Entropy near 25 is
        # ordinary for a 20-dimensional action space; 0.01 against it is not.
        #
        # 0.003 leaves the bonus at 0.075, still an order above the surrogate, so
        # this throttles the loop rather than closing exploration down.
        #
        # **What says whether it worked** is not `Loss/entropy`, which falls
        # because the coefficient did. It is `Policy/mean_std` settling near 0.6,
        # `action_rate_l2` raw following it to about 9.4 by that ratio,
        # `Loss/surrogate` returning to positive, and the reward climbing back
        # towards the +31 it reached once. If `std` falls and the reward does not
        # follow, `std` was a symptom and `action_rate_weight` is the next knob.
        #
        # **Halved again, to 0.0015, when the precision rung went in.** Resumed at
        # level 3 from model_70550 (2026-09-29_08-35-39), `Policy/mean_std` went
        # 0.267 -> 0.293 within the first 400 iterations and stayed there, and
        # `track_linear_velocity` fell 1.52 -> 1.38 while the deterministic policy
        # walked exactly as fast (0.26 / 0.57 / 0.73 m/s at 0.3 / 0.6 / 0.8,
        # against 0.25 / 0.56 / 0.71 before) -- a fall in what the *noise* costs,
        # not in the gait, which is the same loop as above arriving at a quarter
        # of that std. At std 0.29 the bonus is 0.003 x 3.8 = 0.011 against a
        # surrogate near 0.005; 0.0015 puts it level with the surrogate.
        # **What says whether it worked**: `mean_std` back under 0.27, and
        # `track_linear_velocity` back towards 1.5 with the precision rung's
        # `track_*` terms holding.
        entropy_coef=0.0015,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        # **Both corrected for a 200 Hz control loop**, where the skeleton and
        # every other hexa task run at 50. These are per *step*, so the horizon
        # they buy is in steps and shrinks in seconds as the rate rises: 0.99 is
        # a 1.99 s horizon at 50 Hz and 0.50 s at 200, which is a different task,
        # not a faster one.
        #
        # The correction is `g ** (dt_new / dt_old)`, here the fourth root:
        #
        #     gamma  0.99  -> 0.997491    horizon 1.99 s, unchanged
        #     lam    0.95  -> 0.987259    GAE's own averaging window, likewise
        #
        # Left alone, the value function would have been fitted over a quarter of
        # the future this robot's gait cycle occupies -- 0.18 s against a 0.18 s
        # cycle at the 5.56 Hz ceiling, so exactly one stride and no more -- and
        # nothing would have reported it. `desired_kl` and `learning_rate` are
        # per update rather than per step and need no equivalent.
        gamma=0.99 ** 0.25,
        lam=0.95 ** 0.25,
        num_learning_epochs=5,
        num_mini_batches=4,
        # ── Runner ──
        # `NUM_STEPS_PER_ENV` above, and the note there says why it is not a
        # literal: every curriculum dwell in this task counts environment steps,
        # so this number is what turns one into iterations.
        num_steps_per_env=NUM_STEPS_PER_ENV,
        max_iterations=10_000,
    )


def runner_cls() -> type:
    """mjlab's velocity runner, which logs extra velocity tracking metrics, with
    the curriculum levels added to its checkpoints (`common/runner.py`)."""
    from ..common.runner import CurriculumRunner

    return CurriculumRunner


def play_status():
    """The line `scripts/play.py` prints while replaying: achieved speed against
    commanded. Shared by the four velocity tasks; see `common/play.py`."""
    from ..common.play import velocity_status

    return velocity_status
