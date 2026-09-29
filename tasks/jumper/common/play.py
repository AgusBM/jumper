"""The replay readout the velocity tasks want.

`scripts/play.py` prints one rewritten line while it replays and asks the task
what should be on it (`tasks.load_play_status`). What belongs there is a task's
business: a velocity task wants the commanded speed beside the achieved one, and
a task with no command wants something else entirely.

This lives in `common/` because all four velocity tasks want the same line. A
task with a readout of its own puts it in its own directory -- see
`tasks/jumper/swing/rl_cfg.py`.
"""

from __future__ import annotations

from typing import Any


def velocity_status(entity: Any, index: int, env: Any) -> str:
    """"speed 0.31 of 0.35 m/s" -- achieved against commanded.

    Falls back to the achieved speed alone when there is no command to read, which
    is what `play --agent zero` on a config built without one does. That guard is
    kept here rather than left to `play.py`, because "is there a twist command"
    is a question only a task that has one can ask.
    """
    if entity is None:
        return ""
    import torch

    with torch.inference_mode():
        speed = float(torch.norm(entity.data.root_link_lin_vel_b[index, :2]))
    command = getattr(getattr(env, "unwrapped", env), "command_manager", None)
    asked = None if command is None else command.get_command("twist")
    if asked is None:
        return f"speed {speed:.2f} m/s"
    return f"speed {speed:.2f} of {float(torch.norm(asked[index, :2])):.2f} m/s"
