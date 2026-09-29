"""MDP terms only `jumper.swing` uses.

Everything here needs the swing, so none of it belongs in `common/` -- see the
note in `tasks/__init__.py` and `tests/test_registry.py`. The geometry all four
modules share is in `state.py`, computed once so that the reward, the observation
and the termination cannot drift into measuring three slightly different swings.
"""

from __future__ import annotations

from . import events, observations, rewards, state, terminations

__all__ = ["events", "observations", "rewards", "state", "terminations"]
