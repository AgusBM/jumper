"""MDP terms specific to jumper.tetrapod.

Rewards, observations and terminations only this task uses live here; the gating
and phase-matching pieces the three gait tasks share are in
`tasks/jumper/common/mdp/`.
"""

from . import rewards

__all__ = ["rewards"]
