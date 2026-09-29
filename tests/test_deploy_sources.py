"""`deploy.py`'s provenance: the commit an app says its controller came from,
and the wasm-bindgen that wrote its glue.

Apart from `test_bundle.py`, which skips whole without an exported policy to
bundle; this needs none, only cargo and git.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/deploy.py"


def _deploy_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("deploy_for_sources", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tracked(*pathspecs: str) -> set[str]:
    """The files a set of git pathspecs names, committed or not (ignored ones
    aside) -- what `source_commit` dates and checks for `-dirty`."""
    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "--", *pathspecs],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    return set(out.split())


def _uncovered(sources: tuple[str, ...], inputs: list[Path]) -> set[str]:
    """The inputs, as repository files, that `sources` does not name.

    A directory is every file in it -- except `tasks/`, which `build.rs`
    watches only to find new hooks. The hooks it compiles are listed file by
    file, as `#[path]` modules, so of `tasks/` only the deploy directories are
    inputs.
    """
    covered = _tracked(*sources)
    crate = REPO / "deploy/fsm"
    files: set[str] = set()
    for path in inputs:
        path = path.resolve()
        if REPO not in path.parents or crate / "target" in path.parents or path == crate / "target":
            continue  # a registry crate, or what the build itself generated
        rel = path.relative_to(REPO).as_posix()
        if path.is_file():
            files.add(rel)
        elif rel == "tasks":
            files |= {f for f in _tracked(rel) if "/deploy/" in f}
        else:
            files |= _tracked(rel)
    return files - covered


def test_the_provenance_dates_everything_the_controller_is_built_from() -> None:
    """`bundle.json`'s `runtime.commit` is the commit that last touched
    `deploy.py::SOURCES`, `-dirty` when any of it is uncommitted. So SOURCES
    has to name **every file the build reads**: one it misses changes the
    controller under an unchanged commit, and ships uncommitted without
    `-dirty`, and nothing anywhere looks wrong.

    That happened. SOURCES was the crate's `Cargo.toml`, `Cargo.lock` and
    `src/`, while `build.rs` compiles in every task's deploy hook: the jumper
    app built from 3fab20a carried the arm of 723d8e6, the commit after, and
    an uncommitted hook built clean. The same was true of the dictionary
    `src/vocabulary.rs` includes.

    What the build reads is taken from the build, not from a list written
    beside this one: rustc's dep-info for the default features -- the
    board's, the widest a host builds -- and every path `build.rs` joins onto
    the crate's directory, which covers what only an aarch64 build reads (the
    NPU header). The old SOURCES is the control group: the same check has to
    name the hook and the dictionary it missed.
    """
    import re

    deploy = _deploy_module()
    cargo = deploy.which("cargo")
    if cargo is None:
        pytest.skip("cargo is not installed, so what the controller reads cannot be asked")
    crate = REPO / "deploy/fsm"
    built = subprocess.run(
        [cargo, "build", "--lib", "--message-format=json", "--manifest-path",
         str(crate / "Cargo.toml")],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert built.returncode == 0, built.stderr[-2000:]
    rlib = next(
        Path(f)
        for line in built.stdout.splitlines() if line.startswith("{")
        for message in [json.loads(line)]
        if message.get("reason") == "compiler-artifact"
        and message["target"]["name"] == "mjrl_fsm"
        for f in message["filenames"] if f.endswith(".rlib")
    )
    depinfo = rlib.with_suffix(".d").read_text("utf-8")
    inputs = [Path(p) for p in depinfo.split(":", 1)[1].split()]
    build_rs = (crate / "build.rs").read_text("utf-8")
    inputs += [crate / rel for rel in re.findall(r'manifest\.join\("([^"]+)"\)', build_rs)]
    assert any(p.name == "lib.rs" and "deploy" in p.parts for p in inputs), (
        "no task hook among what the build read: this test lost what it is for"
    )

    missing = _uncovered(deploy.SOURCES, inputs)
    assert not missing, f"the controller is built from files SOURCES does not date: {sorted(missing)}"
    before = ("deploy/fsm/Cargo.toml", "deploy/fsm/Cargo.lock", "deploy/fsm/src")
    missed = _uncovered(before, inputs)
    assert {"tasks/jumper/five_foot/deploy/lib.rs", "controller/vocabulary.json"} <= missed, missed


def test_the_wasm_bindgen_cli_is_held_to_the_lock_exactly(monkeypatch) -> None:
    """`check_tooling` refuses every CLI but the one `Cargo.lock` builds.

    It compared against `Cargo.toml`'s `"0.2"` until 2026-09-29, so a CLI one
    release off passed, and the command it printed on a mismatch,
    `--version 0.2`, is one cargo refuses. The CLI is a stand-in here; the
    control is one patch release off the lock, which the old prefix check let
    through.
    """
    import re

    deploy = _deploy_module()
    locked = deploy.crate_wasm_bindgen_version()
    assert re.fullmatch(r"\d+\.\d+\.\d+", locked), f"not a version cargo installs: {locked}"

    asked: list[str] = []
    monkeypatch.setattr(deploy, "require", lambda name, install: asked.append(install) or name)
    monkeypatch.setattr(deploy, "tool_version", lambda path: f"wasm-bindgen {locked}")
    assert deploy.check_tooling() == ("wasm-bindgen", f"wasm-bindgen {locked}")
    assert asked == [f"cargo install wasm-bindgen-cli --version {locked} --locked"]

    major, minor, patch = locked.split(".")
    off = f"{major}.{minor}.{int(patch) + 1}"
    monkeypatch.setattr(deploy, "tool_version", lambda path: f"wasm-bindgen {off}")
    with pytest.raises(SystemExit) as refused:
        deploy.check_tooling()
    assert f"--version {locked} --locked" in str(refused.value)
