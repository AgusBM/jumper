"""Move one reward term's weight over training, and report the feedforward.

`env_cfg.py` uses `anneal_reward_weight` four times, on one window: the two
prior terms go from their weights to zero while the two height terms go from
zero to theirs -- a cross-fade from the recording's jump to the task's own
objective. The action term's feedforward falls to zero over the same window;
it reads its own value off the step counter, and `report_feedforward` only
puts that value in the log beside the weights.

A schedule, not a curriculum in the performance-gated sense
`common/mdp/curriculum.py` means: the weight is a function of how far training
has gone and of nothing the policy does. That is the intent. The prior is
scaffolding, and scaffolding that stays up for as long as the policy leans on it
never comes down.

**It writes the reward manager's own term config.** `RewardManager.compute`
reads each term's weight from the list `get_term_cfg` returns an element of, on
every step, so an assignment here is the weight the next step uses. Written to
`env.cfg.rewards` instead it would change nothing and log as though it had.

Counted in `common_step_counter`, which `mjlab/rl/runner.py` restores on
`--resume`, so a resumed run picks the schedule up where it left off.
"""

from __future__ import annotations

import torch


def anneal_reward_weight(
    env,
    env_ids: torch.Tensor | None,
    term_name: str,
    start_weight: float,
    end_weight: float,
    start_step: int,
    end_step: int,
) -> float:
    """`start_weight` until `start_step`, `end_weight` from `end_step`, linear
    between. Returns the weight, which the curriculum manager logs as
    `Curriculum/<term>`."""
    del env_ids
    if not 0 <= start_step < end_step:
        raise ValueError(
            f"need 0 <= start_step < end_step, got {start_step} and {end_step}"
        )
    s = env.common_step_counter
    frac = min(max((s - start_step) / (end_step - start_step), 0.0), 1.0)
    weight = start_weight + (end_weight - start_weight) * frac
    env.reward_manager.get_term_cfg(term_name).weight = weight
    return weight


def report_feedforward(env, env_ids: torch.Tensor | None, action_name: str = "joint_pos") -> float:
    """The action term's feedforward, for `Curriculum/<term>`. Sets nothing."""
    del env_ids
    return env.action_manager.get_term(action_name).feedforward


__all__ = ["anneal_reward_weight", "report_feedforward"]
