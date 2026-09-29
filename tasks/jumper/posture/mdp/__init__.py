"""MDP terms specific to jumper.posture.

The posture command, what it is measured against, and the three terms that score
it. The gait itself is not here: this task walks with `jumper.tripod`'s gait reward
and clock, imported from that task rather than copied -- see the note in
`../env_cfg.py`.
"""

from . import commands, observations, rewards, state

__all__ = ["commands", "observations", "rewards", "state"]
