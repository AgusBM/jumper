"""The task registry.

The entry points under `scripts/` (train / play / export) **import no task's
config classes**; they know only this table.

## One task = one directory = one fixed name

A task id **is its module path relative to `tasks/`**, and that rule is
load-bearing: configs are not stored in the registry but loaded by convention from
the task's own directory -- `load_env_cfg` / `load_agent_cfg` turn the id straight
into `tasks.<id>.env_cfg` and import it.

    tasks/jumper/tripod/           task id is "jumper.tripod"
    |-- __init__.py     calls register(id="jumper.tripod", ...); names and assets only
    |-- env_cfg.py      def env_cfg(asset: Path | None, play: bool) -> env config
    |-- rl_cfg.py       def agent_cfg() -> rsl_rl config; def runner_cls() -> class|None
    |-- mdp/            (optional) rewards/observations/terminations **only this task uses**
    `-- export_media.py (optional) def export_media(out, ...) -> extra export artifacts

Dots in an id are directory separators: `jumper.tripod` <-> `tasks/jumper/tripod/`.
Tasks are grouped by robot family, and what a family shares goes in
`tasks/<family>/common/`.

To add a task: create the directory, write those three files, and import it in
`tasks/__init__.py`. **Nothing under `scripts/` changes.**

## No hyper-parameters in the registry

Hyper-parameters are **each task's own business**, written in its `rl_cfg.py`, so
changing one task means editing one file. The registry records only metadata --
names, assets, descriptions, tags. It is a directory, not a config container.

## Why configs are lazy

Registration happens at import time, when the backend has not been resolved and
torch / warp should not yet be pulled in. `register()` stores only strings and
paths; `load_*` is what actually imports a config module. Commands like `--list`
therefore work on a machine with no GPU, or with the simulation dependencies not
installed at all.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "error_text",
    "AssetSpec",
    "TaskSpec",
    "register",
    "get",
    "list_ids",
    "all_specs",
    "load_env_cfg",
    "load_agent_cfg",
    "load_cli_args",
    "load_play_status",
    "load_runner_cls",
    "load_export_media",
]

#: Suffixes recognised as model files. A `--model` value carrying one of these, or
#: containing a path separator, is treated as a file path; anything else is treated
#: as a registered asset name.
_MODEL_SUFFIXES = frozenset({".xml", ".mjcf", ".mjb"})


def error_text(exc: Exception) -> str:
    """Get an exception's readable message.

    `str()` on a `KeyError` re-reprs the message (adding a layer of quotes and
    turning newlines into \\n), which reads badly when printed to a user. A failed
    lookup raising KeyError is semantically right, so this is fixed on the display
    side instead.
    """
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


@dataclass(frozen=True)
class AssetSpec:
    """A model asset that can be fed to a task."""

    name: str
    """The short name used on `--model`, e.g. ``jumper``."""

    path: Path
    """Absolute path to the MJCF / XML."""

    description: str = ""
    """A one-line description, shown by `--list`."""


@dataclass(frozen=True)
class TaskSpec:
    """A task's metadata. **No config values** -- those live under `tasks/<id>/`."""

    id: str
    """The fixed task name, used to select it on the command line. Must match the
    directory under `tasks/`."""

    assets: tuple[AssetSpec, ...] = ()
    """The model assets this task supports; the first is the default. Empty means
    `--model` is not accepted."""

    description: str = ""
    """A one-line description, shown by `--list`."""

    tags: tuple[str, ...] = field(default_factory=tuple)
    """Free-form tags for filtering, e.g. ``("smoke",)`` / ``("locomotion", "jumper")``."""

    @property
    def default_asset(self) -> AssetSpec | None:
        return self.assets[0] if self.assets else None

    def resolve_asset(self, requested: str | None) -> Path | None:
        """Resolve a `--model` value into an XML path.

        Three forms:

            None                 use the task's default asset
            a registered name    e.g. ``jumper``
            a file path          e.g. ``assets/jumper/jumper.xml``, or any .xml / .mjcf

        The path form is a back door for experiments: **a task's joint names, HOME
        pose and foot geom names are written against its default asset**, so a
        structurally different model fails while the environment is being built
        (joint not found, sensor matching no geom) rather than quietly producing
        wrong results.
        """
        if requested is None:
            return self.default_asset.path if self.default_asset else None

        for asset in self.assets:
            if asset.name == requested:
                return asset.path

        candidate = Path(requested).expanduser()
        if "/" in requested or candidate.suffix.lower() in _MODEL_SUFFIXES:
            if not candidate.is_file():
                raise FileNotFoundError(f"model file does not exist: {candidate}")
            return candidate.resolve()

        known = ", ".join(a.name for a in self.assets) or "(this task registers no assets)"
        raise KeyError(
            f"task {self.id!r} has no asset named {requested!r}. "
            f"Registered: {known}. A path to an .xml also works."
        )


_REGISTRY: dict[str, TaskSpec] = {}


def register(
    id: str,  # noqa: A002 - matches gym.register's parameter name
    assets: tuple[AssetSpec, ...] = (),
    description: str = "",
    tags: tuple[str, ...] = (),
) -> TaskSpec:
    """Register a task. Registering the same id twice raises, so nothing is
    silently overwritten.

    ``id`` must match the task's directory, or `load_env_cfg` will not find the
    config.
    """
    if id in _REGISTRY:
        raise ValueError(
            f"task id registered twice: {id!r} (an earlier import already defined it)"
        )
    names = [a.name for a in assets]
    if len(names) != len(set(names)):
        raise ValueError(f"task {id!r} has duplicate asset names: {names}")
    spec = TaskSpec(id=id, assets=assets, description=description, tags=tags)
    _REGISTRY[id] = spec
    return spec


def get(task_id: str) -> TaskSpec:
    """Look up a TaskSpec by id, listing every available id on failure rather than
    raising a bare KeyError."""
    try:
        return _REGISTRY[task_id]
    except KeyError:
        available = "\n  ".join(sorted(_REGISTRY)) or "(the registry is empty)"
        raise KeyError(
            f"unknown task {task_id!r}. Available tasks:\n  {available}"
        ) from None


def list_ids(tag: str | None = None) -> list[str]:
    """List every task id, optionally filtered by tag."""
    ids = sorted(_REGISTRY)
    if tag is None:
        return ids
    return [i for i in ids if tag in _REGISTRY[i].tags]


def all_specs() -> list[TaskSpec]:
    return [_REGISTRY[i] for i in sorted(_REGISTRY)]


# ── Load configs by convention from tasks/<id>/ ───────────────────────────


def _module(task_id: str, name: str) -> Any:
    """Import `tasks.<task_id>.<name>`, turning "the directory does not follow the
    convention" into a readable message."""
    get(task_id)  # confirm the id is registered first, for a better error
    package = f"{__package__}.{task_id}"
    dotted = f"{package}.{name}"
    try:
        return importlib.import_module(dotted)
    except ModuleNotFoundError as e:
        # What is missing may be the target module itself or any parent package
        # along the path (a directory that was never created). Anything else is a
        # dependency of the config module (mjlab, say) not being installed, and
        # re-raising that unchanged is more useful.
        if not (e.name == dotted or dotted.startswith(f"{e.name}.")):
            raise
        raise ModuleNotFoundError(
            f"{dotted} not found. A task id must match a directory under tasks/, "
            f"and that directory must contain {name}.py -- see tasks/registry.py."
        ) from None


def load_env_cfg(
    task_id: str,
    asset: Path | None = None,
    play: bool = False,
    task_args: dict[str, Any] | None = None,
) -> Any:
    """Build this task's environment config.

    Args:
        asset: model path from `--model`, resolved by `TaskSpec.resolve_asset`.
            ``None`` means the task's default asset.
        play: replay mode -- observation noise and disturbances off, longer
            episodes.
        task_args: values for the arguments this task added to the parser itself,
            keyed by their `dest`. See `load_cli_args`. Empty for every task that
            adds none, which is why `env_cfg` signatures do not have to change.
    """
    return _module(task_id, "env_cfg").env_cfg(
        asset=asset, play=play, **(task_args or {})
    )


def load_agent_cfg(task_id: str) -> Any:
    """Get this task's own hyper-parameters (PPO, network, iteration count)."""
    return _module(task_id, "rl_cfg").agent_cfg()


def load_runner_cls(task_id: str) -> type | None:
    """Get the runner class this task uses; None means the default OnPolicyRunner."""
    return getattr(_module(task_id, "rl_cfg"), "runner_cls", lambda: None)()


def load_cli_args(task_id: str) -> Any:
    """The command-line arguments this task adds for itself, or None.

    A `(group) -> tuple[str, ...]` callable: it adds arguments to the group it is
    handed and returns their `dest` names, so the entry point knows which of the
    parsed values belong to the task and should be forwarded to `load_env_cfg`.

    **This is how a task gets a flag without `scripts/` growing one.**
    `scripts/_cli.py` holds one parser for all three entry points, so a
    `--swing-angle` written there would be a task's vocabulary in a file that is
    not allowed any -- the rule `tests/test_log_layout.py` pins. Here the parser
    stays empty of task names and asks the selected task what else it takes.

    It lives in the task's `__init__.py`, which is the one module of a task that
    is always imported and must stay free of simulation dependencies. Declaring an
    argument needs `argparse` and nothing else, so that is not a constraint.

    Returning the names rather than having the caller diff the parser is
    deliberate: argparse's introspection is private API, and a task that knows
    what it added can say so in one line.
    """
    get(task_id)  # confirm the id is registered first, for a better error
    return getattr(
        importlib.import_module(f"{__package__}.{task_id}"), "cli_args", None
    )


def load_play_status(task_id: str) -> Any:
    """The replay readout this task wants, or None for the generic one.

    A `(env, index) -> str` callable. `scripts/play.py` prints a line while it
    replays, and what belongs on that line is a task's business: a velocity task
    wants the commanded speed beside the achieved one, and a task with no command
    wants something else entirely.

    **This exists because `play.py` used to reach for `command_manager
    .get_command("twist")` itself.** That is a task's vocabulary in a file that is
    supposed to have none -- `scripts/` does not change when a task is added, which
    is the rule `tests/test_log_layout.py` enforces -- and it failed exactly as a
    layering violation does: `jumper.swing` has no command, `get_command` returned
    None rather than raising, the `except (AttributeError, KeyError)` written for
    this very case did not catch a `TypeError`, and replay died on the first
    frame for the first task that was not about velocity.

    Optional, like `runner_cls` and for the same reason: a task that has nothing
    to add says nothing.
    """
    return getattr(_module(task_id, "rl_cfg"), "play_status", lambda: None)()


def load_export_media(task_id: str) -> Any | None:
    """Get this task's extra-artifact hook, or None if it has none.

    `scripts/export.py` writes the same three files for every task -- `actor.onnx`,
    `layout.json`, `README.md` -- because that set *is* the deployment contract and
    the robot-side importer reads exactly it. Some tasks additionally have
    artifacts that are not for the robot at all: `jumper.dance` has a performance to
    render, with the music it was choreographed to.

    Those cannot go in the bundle (the importer copies the bundle to the board, and
    a 100 MB video would ride along), and they cannot go in `export.py` (it is
    shared by every task, and `tasks/registry.py`'s contract is that **nothing
    under `scripts/` changes when adding a task**). So this is the seam: a task may
    define `export_media.py` with an `export_media()` function, and `export.py`
    calls it knowing only that it exists -- never what it writes.

    **Absent by default and that is not an error.** Four of the five tasks have no
    such file; `find_spec` returning None is the normal case, not a
    misconfiguration, which is why this does not go through `_module` (whose job is
    to turn a missing *required* module into an explanation).
    """
    get(task_id)  # confirm the id is registered first, for a better error
    dotted = f"{__package__}.{task_id}.export_media"
    if importlib.util.find_spec(dotted) is None:
        return None
    return getattr(importlib.import_module(dotted), "export_media", None)
