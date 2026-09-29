"""Fixed paths inside the repository.

A module of its own so that "where the repository root is" is derived exactly
once. Scattered `Path(__file__).resolve().parents[N]` expressions go wrong when a
file moves, and **raise nothing** -- they simply point at a directory that does
not exist, and only blow up when something tries to read from it.
"""

from __future__ import annotations

from pathlib import Path

#: The repository root. This file lives in `tasks/`, hence one level up.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

#: The model asset root. One subdirectory per asset, each with its own `tools/`
#: for generation and calibration scripts.
ASSETS_DIR: Path = REPO_ROOT / "assets"

#: The training log root directory name. Relative to the **current working
#: directory** rather than the repository root, matching mjlab's `--log-root`
#: semantics: training started elsewhere puts its logs there.
LOGS_DIRNAME = "logs"

#: Placeholder for the `<model>` segment when a task registers no assets, keeping
#: the directory depth constant so that globs like `logs/*/<task>/` always match.
NO_MODEL = "no_model"


def log_root_for(asset: Path | str | None) -> str:
    """The training log root: `logs/<model>`.

    The full structure is **`logs/<model>/<task>/<date-time>`**; mjlab supplies the
    last two segments (`experiment_name` is the task id, plus a timestamp; see
    `log_root_path` in `rl/mjlab/scripts/train.py`). This function is responsible
    only for the first.

    `<model>` is the model file's stem, so training one task against different
    `--model` values splits naturally into separate subtrees rather than mixing --
    which is the reason model is the outermost segment.
    """
    model = Path(asset).stem if asset else NO_MODEL
    return f"{LOGS_DIRNAME}/{model}"


#: Directory name for a task's kept exports, inside the task's own directory.
OUT_DIRNAME = "out"


def out_dir_for(task_id: str) -> Path:
    """Where a task's exports go: **`tasks/<task path>/out/`**.

    Beside the task rather than in one pile at the repository root, for the same
    reason a task's rewards live beside its config: an export belongs to the task
    it came from. `logs/` is the opposite case and stays where it is -- it holds
    every run, most of which are noise, and it is written by training rather than
    curated by hand. This directory is the small set someone decided to keep.

    A task id is its module path (`jumper.tripod` -> `tasks/jumper/tripod/`), the same
    rule the registry loads configs by, so nothing new has to be kept in step.

    One directory per export, named `<date-time>`. Not after the checkpoint: two
    exports of one checkpoint differ whenever anything between the config and
    `export.py` has changed, and a shared name meant the second silently
    replaced the first. Which checkpoint an export came from is recorded inside
    it, in `layout.json`'s `_checkpoint`, where it can hold the run as well as
    the iteration and where something can check it.

    **Ignored by git by default.** `.gitignore`'s `out/` has no leading slash, so
    it matches at every depth and covers this too, and `*.onnx` covers the file
    even where the directory does not. Putting an export into the repository is
    therefore a deliberate act:

        git add -f tasks/jumper/tripod/out/model_39998/actor.onnx

    That is the point rather than an inconvenience. Which exports are worth
    keeping is a judgement about a training run, not something a tool should infer
    -- and `-f` makes the judgement visible in the command.
    """
    return REPO_ROOT / "tasks" / Path(*task_id.split(".")) / OUT_DIRNAME


__all__ = [
    "ASSETS_DIR",
    "LOGS_DIRNAME",
    "NO_MODEL",
    "OUT_DIRNAME",
    "REPO_ROOT",
    "log_root_for",
    "out_dir_for",
]
