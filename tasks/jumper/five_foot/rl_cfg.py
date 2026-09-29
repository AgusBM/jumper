"""Hyper-parameters for jumper.five_foot.

**Only this file changes this task's training hyper-parameters**; no other task
is affected, and nothing this task does not state is inherited silently -- the
values below are the complete set that matters, spelled out.

The numbers are the family's, deliberately: the four six-foot tasks all start
from them, nothing measured here says they are wrong for five legs, and diverging
without a reason would only make a later comparison harder to read.

**One thing is not the family's, and it is not optional: symmetry is off.**
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

#: Environment steps per iteration. A module constant rather than a literal in
#: `agent_cfg` because `env_cfg.py` needs it: the payload curriculum's dwell is in
#: environment steps and was chosen in iterations, so this is what converts one
#: into the other, and the dwell stays 500 iterations whatever the rollout length.
#: A second literal there would drift from this one and change the curriculum's
#: pace with nothing reporting it (`tasks/jumper/posture/rl_cfg.py` has the run
#: where that happened).
NUM_STEPS_PER_ENV = 24


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config.

    `experiment_name` is the middle segment of the log path: the full structure is
    `logs/<model>/jumper.five_foot/<date-time>`, with the outermost segment decided
    by `--model`.
    """
    return jumper_ppo_baseline(
        experiment_name="jumper.five_foot",
        # ── Network ──
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        # ── PPO ──
        entropy_coef=0.005,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        # ── Runner ──
        num_steps_per_env=NUM_STEPS_PER_ENV,
        max_iterations=10_000,
        # ── Symmetry: off, and this is a correctness matter ──
        # The other four jumper tasks augment every mini-batch with its left-right
        # mirror and add a mirror loss, on the grounds that the robot has strict
        # sagittal symmetry. **This task's robot does not.** The left-front leg is
        # carried as a claw and the right-front one walks, so the mirror of a
        # valid state is a machine that does not exist -- and the mirror of a
        # valid action is a policy driving four joints it has no control over.
        #
        # Concretely, `common/mdp/symmetry.py::mirror_jumper` would map the action's
        # 16 joints through a 20-long permutation and the five-column foot terms
        # through a six-leg one. Both raise, which is the good case; the bad case
        # is a future edit that makes them line up numerically and quietly teaches
        # the policy that its two front legs are interchangeable.
        #
        # What is given up is real -- the mirror doubles the effective sample
        # count -- and there is no partial version available: a mirror that fixed
        # the front pair and swapped the middle and rear ones is not a symmetry of
        # this robot either, because the claw's mass sits on the left.
        symmetry=False,
    )


def runner_cls() -> type:
    """mjlab's velocity runner, which logs extra velocity tracking metrics, with
    the curriculum levels added to its checkpoints (`common/runner.py`)."""
    from ..common.runner import CurriculumRunner

    return CurriculumRunner
