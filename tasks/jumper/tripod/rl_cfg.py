"""Hyper-parameters for jumper.tripod.

**Only this file changes this task's training hyper-parameters**; no other task
is affected, and nothing this task does not state is inherited silently -- the
values below are the complete set that matters, spelled out.

The four tasks all start from the same numbers. That is a starting point, not a
controlled comparison: **they are not each other's control groups.** Only two of
the six pairings differ by a single variable --

    flat vs tetrapod   +tetrapod_gait 1.0                          one variable
    flat vs ripple     +ripple_gait 1.0                            one variable
    flat vs tripod     +tripod_gait 1.0, +stance_load 0.5, +obs    three
    tripod vs either   as above, in reverse                        three
    tetrapod vs ripple same budget, structurally different terms   not a control

-- so identical hyper-parameters were never what made a comparison valid, and
changing a number here does not break one that held. `common/ppo.py` has the
measured table.

This is the fastest of the three gaits (3 swinging / 3 in support) and the only
one driven by a fixed phase clock, so it is also the one whose reward has the
most high-frequency structure to get right. `entropy_coef` is the knob that
matters most here; see the note next to it.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config.

    `experiment_name` is the middle segment of the log path: the full structure is
    `logs/<model>/jumper.tripod/<date-time>`, with the outermost segment decided by
    `--model`.
    """
    return jumper_ppo_baseline(
        experiment_name="jumper.tripod",
        # ── Network ──
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        # ── PPO ──
        # **Watch this one on rough terrain.** Measured on a resumed rough run,
        # `Policy/mean_std` climbed 0.239 -> 0.429 over ~300 iterations and
        # settled there, and everything that fell with it was a precision term:
        #
        #     tripod_gait          0.716 -> 0.567   -21%
        #     action_rate penalty -0.464 -> -0.821  +77%
        #     track_linear         1.552 -> 1.549    -0.2%
        #     upright              0.962 -> 0.978    +1.7%
        #     episode length         972 ->   997    +2.5%
        #
        # The coarse objectives held or improved while the fine ones collapsed,
        # and `action_rate_l2` -- which is literally the squared step-to-step
        # action difference, i.e. a noise meter -- grew by very nearly the factor
        # sigma^2 grew by (1.77 against 1.93). That is injected action noise, not
        # a locomotion failure, and 0.01 is its coefficient.
        #
        # It was **not** lowered, because `Train/mean_reward` is collected under
        # that noise and so falls mechanically whether or not the deployed
        # deterministic policy got worse. Establish that first -- replay two
        # checkpoints with `play.py`, which is deterministic -- then tune.
        entropy_coef=0.01,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        # ── Runner ──
        num_steps_per_env=24,
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
