"""Finding a checkpoint in the training log tree.

Training writes `logs/<model>/<task>/<date-time>/model_<iteration>.pt`; the first
two segments come from `tasks.paths.log_root_for`, the third from the entry point
and the last from `rsl_rl`. Three commands have to turn "which checkpoint" into a
path -- train resumes from one, play replays one, export converts one -- and they
used to disagree: play globbed the tree itself, train could not do it at all. The
rule lives here once.

**Nothing here knows about tasks or assets.** It is handed a directory and
searches it, so `mjrl` (the framework) keeps its independence from `tasks` (the
application). Composing `logs/<model>/<task>` is the entry points' job, in
`scripts/_cli.py`.

Paths come back exactly as they were found rather than resolved to absolutes, so
a discovered checkpoint prints as the short `logs/...` path people recognise.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: What a checkpoint is called. `rsl_rl` writes `model_<iteration>.pt` every
#: `save_interval` iterations and once more when training stops.
CHECKPOINT_GLOB = "model_*.pt"

_ITERATION = re.compile(r"^model_(\d+)\.pt$")


def _sort_key(path: Path) -> tuple[float, int]:
    """Oldest first, by modification time, with the iteration number breaking ties.

    **By time rather than by the number in the name.** Resuming from an early
    checkpoint writes a new run whose numbers start below the highest one already
    on disk, so the largest number is not necessarily the latest training -- while
    "the one I just trained" always is the most recent file.

    The iteration number is the tie-breaker because some filesystems keep mtime to
    a whole second, and two checkpoints of a small model can land inside the same
    one; without it the winner of that tie would be whichever order the directory
    happened to be read in.
    """
    match = _ITERATION.match(path.name)
    return (path.stat().st_mtime, int(match.group(1)) if match else -1)


def checkpoints_in(directory: Path) -> list[Path]:
    """Every checkpoint in `directory` or one level below it, oldest last-modified first.

    One level below as well as inside, so both halves of the tree work with one
    call: a **run** directory holds the checkpoints themselves, while the **task**
    root holds one directory per run.
    """
    if not directory.is_dir():
        return []
    found = [*directory.glob(CHECKPOINT_GLOB), *directory.glob(f"*/{CHECKPOINT_GLOB}")]
    return sorted(found, key=_sort_key)


def newest_checkpoint(directory: Path) -> Path | None:
    """The most recently written checkpoint in `directory`, or None if there is none."""
    found = checkpoints_in(directory)
    return found[-1] if found else None


def find_checkpoint(wanted: str | os.PathLike[str] | None, task_root: Path) -> Path:
    """Resolve a request for a checkpoint into a file that exists.

    Args:
        wanted: what was asked for -- a `model_*.pt` file, a directory holding
            some (a run directory, or a whole task root), or None for "the newest
            one under `task_root`".
        task_root: `logs/<model>/<task>`, the tree every run of this task and this
            model writes into.

    Raises:
        FileNotFoundError: naming what was searched. Every caller is a
            command-line entry point and turns this into an exit with a message,
            which is why the text reads as advice rather than as a stack trace.
    """
    if wanted is None:
        newest = newest_checkpoint(task_root)
        if newest is None:
            raise FileNotFoundError(
                f"no checkpoint under {task_root}/. Train one first with "
                f"scripts/train.py, or name one with --checkpoint"
            )
        return newest

    path = Path(wanted).expanduser()
    if path.is_dir():
        newest = newest_checkpoint(path)
        if newest is None:
            raise FileNotFoundError(f"no {CHECKPOINT_GLOB} in {path}/")
        return newest
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    return path


__all__ = [
    "CHECKPOINT_GLOB",
    "checkpoints_in",
    "newest_checkpoint",
    "find_checkpoint",
]
