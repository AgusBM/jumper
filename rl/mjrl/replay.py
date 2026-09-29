"""Work a replay leaves out that training needs.

`scripts/play.py` builds the task with `play=True`, which is the task's own idea of
a replay: no observation noise, no pushes, long episodes. This module is the
framework's half -- things the environment does on every step for the learner that
nothing in a replay reads.

## The reward terms

**Every step evaluates every reward term**, and a replay throws the result away:
`play.py`'s loop discards what `env.step` returns but the observation, and neither
the live viewer, `--measure` nor a task's status line reads a reward. On
jumper.posture with one environment on warp (RTX 5090 D, 2026-09-26) the reward
manager was **4.7 of 11.3 ms** a control step, the largest single cost of a replay
step, and skipping it took the headless rate from 92 to 121 steps a second. That
is the difference between a 200 Hz task watched at 0.46x and at 0.61x of real time.

**What makes skipping safe is not that nothing reads the rewards; it is that the
policy's observation does not depend on a reward term having run.** A term can
leave state behind -- `soft_touchdown` banks a foot velocity, jumper.jump caches
its reference on the step counter, jumper.posture's gait clock advances lazily for
whoever asks first. If a reward term is the first to advance something an
observation reads, skipping it can move the observation by a step, and nothing
would say so. For every task registered today it does not. The jump cache is
filled first by its termination, which runs before the rewards. The posture clock
*is* advanced first by a reward term -- the actor, the critic and the rewards read
one clock -- but it gives the same number whoever asks: it integrates the tempo the
observation recorded, and a reset brings it up to date before zeroing (see
`CadenceClock` in `tasks/jumper/posture/mdp/cadence.py`). `tests/test_replay.py`
checks this per task, observation by observation, so a task that changes it fails
there rather than in someone's replay; made to integrate the live command at
whoever asks first, the posture clock fails it at step 0.
"""

from __future__ import annotations

__all__ = ["skip_rewards"]


def skip_rewards(env) -> int:
    """Stop `env` evaluating its reward terms. Returns how many it had.

    `compute` is replaced on this **instance** only -- the same way the live viewer
    wraps `sim.step` -- and hands back a buffer of zeros, so `env.step` still
    returns a reward of the right shape. The rest of the manager is untouched:
    `reset` still runs, and still resets the terms that carry state, which is what
    keeps them coherent if anything later evaluates them again.
    """
    import torch

    manager = env.reward_manager
    zeros = torch.zeros(env.num_envs, device=env.device)

    def compute(dt: float) -> torch.Tensor:
        del dt  # nothing to scale
        return zeros

    manager.compute = compute
    return len(manager.active_terms)
