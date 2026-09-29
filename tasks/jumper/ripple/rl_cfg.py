"""Hyper-parameters for jumper.ripple.

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

The most stable gait: 1 swinging / 5 in support, six legs offset by a sixth of a
cycle each. It is also the slowest, and the one where "stand still" is closest to
a correct answer -- with five feet always down, the gait reward is nearly
satisfied by not moving. Worth remembering before reading a flat reward curve
here as convergence.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config.

    `experiment_name` is the middle segment of the log path: the full structure is
    `logs/<model>/jumper.ripple/<date-time>`, with the outermost segment decided by
    `--model`.
    """
    return jumper_ppo_baseline(
        experiment_name="jumper.ripple",
        # ── Network ──
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        # ── PPO ──
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
