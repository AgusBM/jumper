"""Resuming training from a checkpoint (`train.py --resume`).

Two things can go wrong here, and **neither of them raises**:

- The *wrong* checkpoint is picked up. Training runs, the curves look plausible,
  and it continued from a run nobody meant -- or from a lower iteration than the
  one that was just trained.
- The *right* checkpoint is picked up but only partly loaded. Weights without the
  optimiser state and the iteration counter is a warm restart, not a resumption:
  the adaptive learning rate goes back to its initial value and the run logs from
  iteration 0 again.

So the assertions below are about which file "the newest checkpoint" means, and
about the entry points loading the whole training state through one shared
resolver rather than each searching the tree its own way.

Pure filesystem and source-text checking; no simulation dependency is imported.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mjrl.checkpoint import CHECKPOINT_GLOB, checkpoints_in, find_checkpoint

REPO = Path(__file__).resolve().parents[1]


def _checkpoint(run: Path, iteration: int, mtime: float) -> Path:
    """A stand-in for `model_<iteration>.pt` with a known modification time."""
    run.mkdir(parents=True, exist_ok=True)
    path = run / f"model_{iteration}.pt"
    path.write_bytes(b"")
    os.utime(path, (mtime, mtime))
    return path


# ── Which file "the newest checkpoint" means ────────────────────────────


def test_newest_is_by_time_not_by_number(tmp_path: Path) -> None:
    """The largest iteration number is **not** the latest training.

    Resuming from an early checkpoint starts a new run whose numbers begin below
    the highest already on disk. Sorting by name would then send the next
    `--resume` back to the abandoned run, and every subsequent one after it --
    training would look like it was progressing while going nowhere.
    """
    old = _checkpoint(tmp_path / "2026-09-01_10-00-00", 4999, mtime=1000)
    new = _checkpoint(tmp_path / "2026-09-04_10-00-00", 120, mtime=2000)

    assert find_checkpoint(None, tmp_path) == new
    assert checkpoints_in(tmp_path) == [old, new]


def test_ties_are_broken_by_the_iteration_number(tmp_path: Path) -> None:
    """Two checkpoints written inside the same second must still order.

    Some filesystems keep mtime to a whole second, and a small model saved often
    enough lands two checkpoints in one. Without the tie-break the winner would be
    whichever order the directory happened to be read in -- which differs between
    machines, so it would pass here and pick the wrong file there.
    """
    run = tmp_path / "2026-09-04_10-00-00"
    _checkpoint(run, 50, mtime=1000)
    later = _checkpoint(run, 100, mtime=1000)

    assert find_checkpoint(None, tmp_path) == later


def test_a_run_directory_narrows_to_that_run(tmp_path: Path) -> None:
    """`--checkpoint <run dir>` resumes that run rather than the newest one.

    The whole point of naming a directory is to go back to a run that is *not*
    the most recent, so the search must not fall back to the task root.
    """
    wanted = _checkpoint(tmp_path / "2026-09-01_10-00-00", 4999, mtime=1000)
    _checkpoint(tmp_path / "2026-09-04_10-00-00", 120, mtime=2000)

    assert find_checkpoint(wanted.parent, tmp_path) == wanted


def test_a_named_file_is_taken_as_given(tmp_path: Path) -> None:
    """An explicit path is never second-guessed, wherever it lives."""
    outside = _checkpoint(tmp_path / "somewhere-else", 7, mtime=1000)
    _checkpoint(tmp_path / "logs" / "2026-09-04_10-00-00", 120, mtime=2000)

    assert find_checkpoint(str(outside), tmp_path / "logs") == outside


def test_the_search_covers_both_levels_of_the_tree(tmp_path: Path) -> None:
    """One call handles a task root (one directory per run) and a run directory
    (the checkpoints themselves), because `--checkpoint` accepts either."""
    inside = _checkpoint(tmp_path / "run", 10, mtime=1000)

    assert checkpoints_in(tmp_path) == [inside]
    assert checkpoints_in(tmp_path / "run") == [inside]


def test_only_checkpoints_are_considered(tmp_path: Path) -> None:
    """A run directory also holds `params/`, `policy.onnx`, `tensorboard.log` and
    the event files; none of them is a checkpoint."""
    run = tmp_path / "2026-09-04_10-00-00"
    ckpt = _checkpoint(run, 10, mtime=1000)
    for stray in ("policy.onnx", "tensorboard.log", "events.out.tfevents.1"):
        (run / stray).write_bytes(b"")
    os.utime(run / "policy.onnx", (9999, 9999))

    assert find_checkpoint(None, tmp_path) == ckpt


# ── Failing with something to act on ────────────────────────────────────


def test_an_empty_tree_says_what_to_do(tmp_path: Path) -> None:
    """With nothing trained yet, `--resume` must not read as a broken install."""
    with pytest.raises(FileNotFoundError) as e:
        find_checkpoint(None, tmp_path / "logs" / "jumper" / "jumper.flat")
    message = str(e.value)
    assert "jumper.flat" in message, "the message must name the tree it searched"
    assert "--checkpoint" in message and "train" in message


def test_a_missing_file_is_not_reported_as_an_empty_tree(tmp_path: Path) -> None:
    """A typo in `--checkpoint` must name the path, not the directory searched."""
    with pytest.raises(FileNotFoundError, match="model_999.pt"):
        find_checkpoint(tmp_path / "model_999.pt", tmp_path)


def test_a_directory_with_no_checkpoints_says_so(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    with pytest.raises(FileNotFoundError, match=CHECKPOINT_GLOB.replace("*", r"\*")):
        find_checkpoint(tmp_path / "run", tmp_path)


# ── How the entry points wire it up ─────────────────────────────────────


def source(name: str) -> str:
    return (REPO / "scripts" / name).read_text(encoding="utf-8")


def test_train_restores_the_whole_training_state() -> None:
    """`train.py` must load without a `load_cfg`.

    play.py deliberately passes `load_cfg={"actor": True}` -- replay needs the
    policy and nothing else. The same call in train would load the weights and
    silently drop the optimiser state, the iteration counter and the environment's
    step counter: the learning rate would restart from its initial value, the run
    would log from iteration 0 over the top of the earlier curve, and every
    curriculum counting env steps would rewind. It trains, so nothing looks wrong.
    """
    train = source("train.py")
    call = train[train.index("runner.load(") : train.index("runner.load(") + 200]
    assert "load_cfg" not in call, (
        "train.py resumes with a partial load; that is a warm restart, not a resumption"
    )
    assert 'load_cfg={"actor": True}' in source("play.py"), (
        "play.py should still load the policy alone -- if this moved, re-check the above"
    )


def test_train_resolves_the_checkpoint_before_it_builds_anything() -> None:
    """A bad `--checkpoint` must cost a second, not the minute of scene assembly
    and model compilation that precedes the load."""
    train = source("train.py")
    assert train.index("resolve_checkpoint(") < train.index("ManagerBasedRlEnv(cfg=")


def test_train_loads_the_checkpoint_before_it_learns() -> None:
    train = source("train.py")
    assert train.index("_resume(runner") < train.index("runner.learn(")


def test_no_entry_point_searches_the_log_tree_itself() -> None:
    """One implementation of "which checkpoint", in `mjrl.checkpoint`.

    train and play used to disagree, because train had no answer at all and play
    globbed the tree in a helper of its own. Two answers to "the newest
    checkpoint" is one too many: whichever command someone checks with is the one
    they will trust.
    """
    for name in ("train.py", "play.py", "export.py", "_cli.py"):
        assert ".glob(" not in source(name), f"{name} searches for checkpoints itself"
