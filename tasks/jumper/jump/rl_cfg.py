"""PPO hyperparameters for jumper.jump."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = jumper_ppo_baseline(experiment_name="jumper.jump")
    # This task's observation is narrower than the locomotion one (no actuator
    # force, no height scan), so it does not follow the shared baseline's fourth
    # hidden layer, which was added when that observation grew. Pin the
    # three-layer structure so the shape does not move under a master update.
    #
    # It used to say 170-dim, and to justify the pin by keeping existing
    # checkpoints loadable. Both expired together: the actor is 167 now, because
    # `base_lin_vel` left it (this robot has no state estimator, and
    # `export.py::_validate_measurable` refuses a policy that observes it), and
    # every checkpoint from before that change is three inputs wider and cannot
    # be loaded anyway. The pin is kept for the first reason, not the second.
    cfg.actor.hidden_dims = (512, 256, 128)
    cfg.critic.hidden_dims = (512, 256, 128)
    # Values from the lab's reference-guided scheme: the reward signal is dense and
    # low-variance, so the exploration need is lower than for velocity tracking and
    # convergence is far faster.
    cfg.algorithm.entropy_coef = 0.005
    cfg.actor.distribution_cfg["init_std"] = 0.6
    return cfg
