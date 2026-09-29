"""The task registry and the "one task, one directory" convention.

Besides the ordinary registration and lookup semantics, two guarantees are pinned
here because they fail silently under refactoring:

1. **`import tasks` must not pull in simulation dependencies.** `--list` working on
   a machine without mjlab installed rests on each task's `__init__.py` touching
   only the registry and lightweight asset declarations. One `from .env_cfg import
   ...` at the top of a task's `__init__.py` and the guarantee is gone, with no
   error anywhere.
2. **A task id is its module path relative to `tasks/`** (`jumper.tripod` <->
   `tasks/jumper/tripod/`). Configs are not in the registry; `load_env_cfg` turns the
   id straight into `tasks.<id>.env_cfg` and imports it -- a mismatch means the
   config is not found.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterator

import pytest

import tasks
from tasks import registry

REPO = Path(__file__).resolve().parents[1]

#: A directory is a task if and only if it has both of these files. Family
#: directories (`jumper/`) and shared packages (`jumper/common/`) have neither and
#: cannot be mistaken for one.
TASK_MARKERS = ("env_cfg.py", "rl_cfg.py")


def task_dirs() -> list[Path]:
    """Find every task directory under `tasks/`, returned relative to tasks/."""
    root = REPO / "tasks"
    return sorted(
        d.relative_to(root)
        for d in root.rglob("*")
        if d.is_dir() and all((d / m).exists() for m in TASK_MARKERS)
    )


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """The registry is a module-level global and has to be restored, or tests
    pollute each other."""
    snapshot = dict(registry._REGISTRY)
    try:
        yield
    finally:
        registry._REGISTRY.clear()
        registry._REGISTRY.update(snapshot)


# ── Laziness ──────────────────────────────────────────────────────────────


def test_import_tasks_does_not_pull_in_simulation_deps() -> None:
    """Importing tasks in a clean interpreter must not pull in mjlab, torch or
    mujoco.

    Run in a subprocess, because other tests in the same pytest process may already
    have imported them.
    """
    probe = (
        "import sys; import tasks; "
        "leaked = [m for m in ('mjlab', 'torch', 'mujoco', 'warp') if m in sys.modules]; "
        "print(','.join(leaked))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert not out, f"import tasks also pulled in: {out}"


# ── One task = one directory = one fixed name ─────────────────────────────


def test_every_task_id_maps_to_its_directory() -> None:
    """Dots in an id are directory separators, and the directory has all three
    files."""
    for spec in tasks.all_specs():
        d = REPO / "tasks" / spec.id.replace(".", "/")
        assert d.is_dir(), (
            f"task {spec.id!r} has no directory tasks/{spec.id.replace('.', '/')}/"
        )
        for required in ("__init__.py", *TASK_MARKERS):
            assert (d / required).exists(), f"{d.relative_to(REPO)} lacks {required}"


def test_every_task_directory_is_registered() -> None:
    """The converse must hold too: every task directory on disk must be registered.

    Forget the import in `tasks/__init__.py` and that task quietly does not exist:
    `--list` does not show it and nothing reports an error.
    """
    registered = set(tasks.list_ids())
    for d in task_dirs():
        task_id = d.as_posix().replace("/", ".")
        assert task_id in registered, (
            f"tasks/{d}/ is a task directory but is not registered (expected id "
            f"{task_id!r}) -- check that tasks/__init__.py imports it"
        )


def test_shared_packages_are_not_registered() -> None:
    """`common/` is a shared package, not a task, and must not appear in the list."""
    for task_id in tasks.list_ids():
        assert "common" not in task_id.split("."), f"{task_id} looks like a shared package"


def test_task_owned_mdp_is_not_in_common() -> None:
    """An MDP term used by one task belongs to that task.

    Each of the three gait rewards serves exactly one task and lives in its
    directory; `common/mdp/` holds only the pieces they share (gating, phase
    matching) and the family-wide mirror transform.
    """
    common_mdp = REPO / "tasks" / "jumper" / "common" / "mdp" / "rewards.py"
    text = common_mdp.read_text(encoding="utf-8")
    for gait in ("tripod", "tetrapod", "ripple"):
        assert f"def {gait}_gait" not in text, (
            f"{gait}_gait is used only by jumper.{gait} and belongs in "
            f"tasks/jumper/{gait}/mdp/rewards.py"
        )
        own = REPO / "tasks" / "jumper" / gait / "mdp" / "rewards.py"
        assert f"def {gait}_gait" in own.read_text(encoding="utf-8")


# ── Loading configs by convention ─────────────────────────────────────────


def test_missing_task_directory_gives_actionable_error(clean_registry: None) -> None:
    """An id registered without its directory must say the convention was broken,
    not raise a bare ImportError."""
    registry.register(id="no_such_dir_task")
    with pytest.raises(ModuleNotFoundError, match="must match a directory under tasks/"):
        registry.load_env_cfg("no_such_dir_task")


def test_missing_nested_task_directory_also_reported(clean_registry: None) -> None:
    """A nested id missing an intermediate parent package must give the same
    convention-level hint."""
    registry.register(id="nosuchfamily.nosuchtask")
    with pytest.raises(ModuleNotFoundError, match="must match a directory under tasks/"):
        registry.load_agent_cfg("nosuchfamily.nosuchtask")


def test_runner_cls_defaults_to_none(monkeypatch) -> None:
    """Without `runner_cls` in rl_cfg.py, rsl_rl's default runner is used.

    Exercised against a module rather than a task, because every task in the
    repository defines one. `smoke.balance` was the subject until it was
    deleted, and the fallback it covered is a real branch -- `load_runner_cls`
    returns None and `train.py` builds `MjlabOnPolicyRunner`. A task added
    tomorrow without a `runner_cls` must not silently get nothing.
    """
    import types

    monkeypatch.setattr(registry, "_module", lambda *_: types.SimpleNamespace())
    assert registry.load_runner_cls("anything") is None

    # Control group: with one, it is called and returned, so the assertion above
    # cannot pass because the lookup is broken in both directions.
    sentinel = type("Runner", (), {})
    monkeypatch.setattr(
        registry, "_module", lambda *_: types.SimpleNamespace(runner_cls=lambda: sentinel)
    )
    assert registry.load_runner_cls("anything") is sentinel


# ── Registration and lookup ───────────────────────────────────────────────


def test_duplicate_id_raises(clean_registry: None) -> None:
    """Silent overwriting is among the hardest errors to trace and must raise."""
    registry.register(id="dup_task")
    with pytest.raises(ValueError, match="registered twice"):
        registry.register(id="dup_task")


def test_duplicate_asset_name_raises(clean_registry: None) -> None:
    dup = (
        registry.AssetSpec("same", Path("a.xml")),
        registry.AssetSpec("same", Path("b.xml")),
    )
    with pytest.raises(ValueError, match="duplicate asset names"):
        registry.register(id="dup_asset_task", assets=dup)


def test_unknown_id_lists_available_ones() -> None:
    """A miss must list the available ids rather than raise a bare KeyError."""
    with pytest.raises(KeyError) as excinfo:
        registry.get("nope")
    message = str(excinfo.value)
    for task_id in registry.list_ids():
        assert task_id in message


def test_list_ids_filters_by_tag(clean_registry: None) -> None:
    registry.register(id="tagged_task", tags=("marker",))
    assert registry.list_ids(tag="marker") == ["tagged_task"]


def test_jumper_gaits_are_four_independent_tasks() -> None:
    """The four gaits are four separate tasks, each with its own directory and
    hyper-parameter file."""
    ids = set(registry.list_ids())
    assert {"jumper.flat", "jumper.tripod", "jumper.tetrapod", "jumper.ripple"} <= ids


def test_each_task_has_its_own_hyperparameter_file() -> None:
    """Hyper-parameters belong to each task: changing one must not affect another."""
    for spec in tasks.all_specs():
        rl_cfg = REPO / "tasks" / spec.id.replace(".", "/") / "rl_cfg.py"
        assert "def agent_cfg" in rl_cfg.read_text(encoding="utf-8")


def test_every_spec_has_a_description() -> None:
    """`--list` shows only the id and this one line; leaving it empty is as good
    as not registering."""
    for spec in tasks.all_specs():
        assert spec.description, f"{spec.id} has no description"


# ── Resolving --model ─────────────────────────────────────────────────────


def test_default_asset_is_the_first_one() -> None:
    spec = tasks.get("jumper.tetrapod")
    assert spec.resolve_asset(None) == spec.assets[0].path


def test_asset_by_registered_name() -> None:
    spec = tasks.get("jumper.flat")
    assert spec.resolve_asset("jumper") == spec.assets[0].path


def test_asset_by_path() -> None:
    spec = tasks.get("jumper.flat")
    path = REPO / "assets" / "jumper" / "jumper.xml"
    assert spec.resolve_asset(str(path)) == path.resolve()


def test_unknown_asset_name_lists_the_registered_ones() -> None:
    spec = tasks.get("jumper.flat")
    with pytest.raises(KeyError, match="jumper"):
        spec.resolve_asset("nope")


def test_missing_model_file_is_reported_as_such() -> None:
    """Something that looks like a path must fail as a path, not as "no such asset
    name" -- that would send someone looking in the wrong place."""
    spec = tasks.get("jumper.flat")
    with pytest.raises(FileNotFoundError, match="model file does not exist"):
        spec.resolve_asset("assets/jumper/does_not_exist.xml")


def test_task_without_assets_resolves_to_none(clean_registry: None) -> None:
    spec = registry.register(id="assetless_task")
    assert spec.resolve_asset(None) is None
