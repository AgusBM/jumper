"""MDP pieces shared by the three gait tasks.

- `rewards`    -- gating and phase matching. The gait reward functions themselves
  are not here; they live in each task's own `tasks/jumper/<task>/mdp/rewards.py`
- `observations` -- terms mjlab has none of; actuator force
- `symmetry`   -- the sagittal mirror transform, used by all four tasks for PPO's
  symmetry augmentation
- `curriculum` -- the command-range curriculum, which scales the tracking std with
  the range so that widening the range does not also loosen the ruler
- `controls`   -- the reader of a task's `controls.yaml`: which stick drives which
  command axis, and the keyboard as a virtual copy of the pad
- `operator`   -- replay-only: one operator drives every command a task has, from
  a pad or the keyboard, instead of it being sampled. The device side of the
  pad is the top-level `controller/` package
"""

from . import controls, curriculum, observations, operator, rewards, symmetry

__all__ = [
    "controls",
    "curriculum",
    "observations",
    "operator",
    "rewards",
    "symmetry",
]
