#!/usr/bin/env python3
"""Build the app: the controller and the policies it drives, for every host at once.

`deploy/fsm` is the observation, the action decode and the state machine. The
robot runs it cross-compiled, a browser as WebAssembly, `play` as a Python
extension. A **bundle** is all three builds plus the policies they drive, in one
directory and one `.app` of it, and building one is what this does:

    python scripts/deploy.py                      # the bundle deploy/manifests.json defines
    python scripts/deploy.py --manifest jumper    # one of several, by name
    python scripts/deploy.py --mode walk=tasks/jumper/tripod/out/<export>   # a one-off

By default it first makes what the robot needs and this script does not build
itself, then refuses an app that still lacks any of it:

  1. the board's controller, and `play`'s for Windows and macOS, cross-built in
     docker (`deploy/fsm/docker-build.sh`) -- every time, so each binary is this
     source's and not whatever was left;
  2. each policy's `.rknn`, converted in docker (`deploy/convert/docker-convert.sh`)
     -- where it is missing, or was made from another `actor.onnx`;
  3. the controller for the browser and for `play`, the reference vectors and the
     English manual, then the check, the schema and the `.app`.

The Chinese manual is the one step left, to an agent: the build cannot write
natural Chinese. The `bundle-manual` skill writes it and `--translate` adds it.
`--allow-incomplete` skips 1 and 2, takes whatever is there, and says in the
notes what is missing -- for a browser, a bench, or the tests.

Two more things run against an app that is already built:

    python scripts/deploy.py --check-reference out/bundle_<timestamp>/<name>
    python scripts/deploy.py --translate out/bundle_<timestamp>/<name> --language zh --manual <file>

It also used to publish the wasm alone into a consumer's asset directory
(`--out`, `--check`) and install the Python extension into site-packages
(`--python`). Both went on 2026-09-28: the app carries the same wasm under
`runtime/web/` and the same extension under `runtime/mjlab/`, and a copy in
site-packages was a build of any age that nothing tied to what shipped.

## Why this is a script and not two commands in a README

It was two commands in a README for one day, and in that day the procedure told
a lie. The artifact was built, `provenance.json` was written from `git
rev-parse HEAD`, and *then* the source fix it contained was committed -- so the
file recorded a commit whose source would produce a **different** controller,
one that reports zero joint torque. The bytes were right and the only thing
wrong was the line saying where they came from, in the file whose whole job is
to be believed.

So three things happen here that a person doing it by hand will not do reliably:

  * **A dirty tree is recorded, not refused.** An artifact containing
    uncommitted code cannot be reproduced by anybody, including the person who
    built it tomorrow, so the provenance says `-dirty` and the build says so as
    it runs. `--require-clean` refuses instead, for a build that is going
    somewhere. Refusing by default made the test suite red for as long as
    anybody was editing the crate.
  * **The commit is read after the check, not before.** With the tree clean,
    `HEAD` is the source, and it cannot drift from the bytes.
  * **The wasm-bindgen CLI is held against the crate**, exactly, at the version
    `Cargo.lock` builds. They share an ABI and are versioned separately; a
    mismatch produces glue that loads and then misreads every argument.

## Not part of `export.py`

`export.py` turns one checkpoint into one exported policy -- a different thing
at a different cadence, and its governing rule is that every value comes from a
live environment. Nothing here reads an environment. Sharing a command would
mean `--checkpoint` for a build that has no policy in it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CRATE = REPO / "deploy/fsm"
WASM = CRATE / "target/wasm32-unknown-unknown/release/mjrl_fsm.wasm"

#: What lands in `runtime/web/`. Named rather than globbed so a stale file left
#: by a previous wasm-bindgen version is not shipped as if it belonged.
#:
#: `controller`, not the crate's own `mjrl_fsm`: what a consumer receives is a
#: controller, and the crate's name is this repository's business. wasm-bindgen
#: writes `<name>_bg.wasm`; the `_bg` is its convention for "the wasm behind the
#: glue" and means nothing outside it, so it is dropped -- see `bindgen`.
ARTIFACTS = ("controller.js", "controller.wasm")

#: The Python extension `play` imports, inside a bundle, beside the wasm.
#:
#: A bundle without it is not self-contained: `play --app` would load whatever
#: `mjrl_fsm` happens to be installed, which is a *different build of the same
#: source* and can be any age. The two silently diverge the moment somebody edits
#: the crate and rebuilds one of them, and nothing says so.
#:
#: Platform-specific where the wasm is not -- `abi3` frees it from the Python
#: version, not from the machine -- so the manifest records which, and a host on
#: another platform is told rather than left with a load error. `.pyd` on
#: Windows, whose CPython imports an extension by that suffix and no other:
#: see `extension_file`.
BUNDLE_EXTENSION = "controller.so"

#: A bundle's schema. A consumer that does not know this string should refuse
#: rather than guess which fields it is missing.
BUNDLE_SCHEMA = "kk-policy-bundle/1"

#: The controller, cross-compiled for the board. One build of one crate serves
#: all three hosts; this is the third compilation of it, and on the robot it is
#: the whole program rather than a library something else links.
BOARD_BINARY = "controller"
STAGING = REPO / "out/deploy/.staging"

#: Where `docker-build.sh` leaves the cross-built binary.
#:
#: Not inside cargo's target tree: the container builds into a named volume so
#: that nothing it owns can block a host-side `cargo build`, which means its
#: target directory is invisible from here. The build copies the one artifact
#: out to this path instead.
CROSS_BUILT = REPO / "out/deploy" / f"{BOARD_BINARY}-aarch64"

#: `play --app` on the machines that do not build apps: the extension,
#: cross-built in the board's container (`deploy/fsm/Dockerfile` says how), by
#: its key in `runtimes.mjlab.extensions` -> (who it is for, where
#: `docker-build.sh` leaves it, `docker-build.sh`'s arguments).
#:
#: A Windows machine or a Mac is where an app is played by somebody who did not
#: build it, and a bundle carried only the build machine's extension -- so there
#: it did not play at all. `runtimes.mjlab.extensions` was a map for this, and
#: nothing wrote a second entry.
#:
#: macOS is one universal2 file for both CPUs, and keyed without a version:
#: `sysconfig.get_platform()` there carries whatever deployment target the
#: interpreter was built for (Homebrew `macosx-14.0-arm64`, python.org
#: `macosx-10.9-universal2`), which says nothing about what this file needs.
#: `mjrl.app_play.extension_for` is the lookup, for both of its callers.
PLAY_CROSS_BUILDS: dict[str, tuple[str, Path, tuple[str, ...]]] = {
    "win-amd64": ("Windows", REPO / "out/deploy/controller-win-amd64.pyd",
                  ("build", "--release", "--target", "x86_64-pc-windows-gnu",
                   "--no-default-features", "--features", "py")),
    "macosx-universal2": ("macOS", REPO / "out/deploy/controller-macosx-universal2.so",
                          ("zigbuild", "--release", "--target", "universal2-apple-darwin",
                           "--no-default-features", "--features", "py")),
}

#: The FSM config a bundle ships. `controller.toml` rather than the `fsm.toml`
#: it used to be, because it configures the thing next to it: `controller`,
#: `controller.wasm` and `controller.so` all read this file.
BUNDLE_CONFIG = "controller.toml"

#: Who a bundle is for: all three, in one. They read the same FSM config, the
#: same contracts and the same `reference.json` -- one copy of each -- and
#: differ only in the model format and which compilation of the controller
#: they take from it:
#:
#:   board   `models/*.rknn` for the NPU, and `runtime/board/controller` (aarch64)
#:   web     `models/*.onnx`, and `runtime/web/controller.{wasm,js}`
#:   mjlab   `models/*.onnx`, and `runtime/mjlab/<platform>/controller.so`
#:           (`.pyd` on Windows), which `play --app` imports
#:
#: It was three directories, one per host, hashed against each other to prove
#: they agreed. One directory cannot disagree with itself.
HOSTS = ("board", "web", "mjlab")

#: The integration guide that travels inside every bundle.
#:
#: A template beside the crate rather than a string in this script: it documents
#: `WebFsm`'s API, so it belongs next to `web.rs` and changes when that does.
#: Prose in a Python literal stops being edited.
#:
#: What it covers is the half a consumer cannot infer -- the frames, the units,
#: the joint order and the two-phase step -- every one of which fails silently.
README_TEMPLATE = CRATE / "BUNDLE_README.md"

#: How many frames of reference go in a bundle.
#:
#: The frames start the robot away from its home pose and converge on it, so a
#: config with a warm start or a mode-switch ramp spends its first frames there
#: and the rest running the policy. The synthesised one-mode config has neither,
#: and then all 24 are inferences -- which the note says, because "how many
#: frames covered which regime" is the only way to know what the reference is
#: actually worth.
REFERENCE_FRAMES = 24

#: The reference's own schema. A host that does not know it should skip the
#: check and say so, not compare against fields it is guessing at.
REFERENCE_SCHEMA = "kk-policy-reference/1"

#: The cargo feature that compiles the browser host.
FEATURES = "web"

#: What the controller is built from, for dating it: git pathspecs, from the
#: repository's root, covering every file the crate's build reads.
#:
#: Not the whole crate: a README edit is not a new controller, and two apps
#: built either side of one should say they carry the same controller. Not
#: narrower either -- `src/dds.rs` is `#[cfg]`-ed out of the browser's build and
#: into the board's, and tracking which module reaches which build means teaching
#: this script what the feature flags exclude. Being wrong about it would give two
#: different controllers one commit. Conservative on purpose: a needless new
#: commit costs nothing, a shared one hides the difference it exists to record.
#:
#: **Not only the crate's directory, either.** This was `Cargo.toml`,
#: `Cargo.lock` and `src/` until 2026-09-28, and the controller was also built
#: from files outside them: `build.rs`, which compiles in every task's deploy
#: hook -- so a commit that changed only `tasks/jumper/five_foot/deploy/lib.rs`
#: shipped a different arm under the previous commit's name, and an uncommitted
#: one without `-dirty` -- the dictionary `src/vocabulary.rs` includes, the IDL
#: the board's bus is generated from, and the NPU header `vendor/` holds.
#: `tests/test_bundle.py` holds this list to what rustc and `build.rs` actually
#: read.
SOURCES = (
    "deploy/fsm/Cargo.toml",
    "deploy/fsm/Cargo.lock",
    "deploy/fsm/build.rs",
    "deploy/fsm/src",
    "deploy/fsm/vendor",
    # A task's hook: `tasks/<family>/<task>/deploy/`, found by `build.rs`.
    ":(glob)tasks/**/deploy/**",
    "controller/vocabulary.json",
    "deploy/dds/idl",
)


def which(name: str) -> str | None:
    """Find a tool on PATH, then where `cargo install` puts things.

    Both tools here arrive through `cargo install`, which writes to
    `$CARGO_HOME/bin` -- a directory a login shell often has and a script
    inherited from an editor, a cron job or an IDE often does not. Looking there
    is not a courtesy: without it this refuses to run with "not installed" for a
    tool that is installed, and the fix it prints is to install it again.
    """
    found = shutil.which(name)
    if found:
        return found
    home = Path(os.environ.get("CARGO_HOME") or (Path.home() / ".cargo"))
    candidate = home / "bin" / name
    return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None


def require(name: str, install: str) -> str:
    path = which(name)
    if path:
        return path
    home = os.environ.get("CARGO_HOME") or "~/.cargo"
    raise SystemExit(
        f"[deploy] {name} is not on PATH and not in {home}/bin.\n"
        f"           {install}"
    )


def run(cmd: list[str], cwd: Path | None = None, capture: bool = False) -> str:
    # What this has printed goes out before what the child prints: into a pipe
    # or a log, `print` is buffered and the docker steps' own output overtook it.
    sys.stdout.flush()
    where = cwd or REPO
    result = subprocess.run(cmd, cwd=where, text=True,
                            capture_output=capture, check=False)
    if result.returncode != 0:
        if capture and result.stderr:
            print(result.stderr, file=sys.stderr)
        raise SystemExit(f"[deploy] failed: {' '.join(cmd)}")
    return (result.stdout or "").strip()


def tool_version(path: str) -> str:
    out = subprocess.run([path, "--version"], text=True, capture_output=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"[deploy] {path} --version failed:\n{out.stderr.strip()}")
    return out.stdout.strip()


def crate_wasm_bindgen_version() -> str:
    """The version the crate compiles against, from `Cargo.lock`.

    Not from `Cargo.toml`, which says `"0.2"`: a range, which every 0.2.x
    satisfies, and which `cargo install --version` refuses outright, because a
    bare `0.2` is not a version. The lock names the one wasm-bindgen the build
    compiles, and that is the one the CLI has to match.
    """
    text = (CRATE / "Cargo.lock").read_text("utf-8")
    match = re.search(r'^name = "wasm-bindgen"\nversion = "([^"]+)"', text, re.M)
    if not match:
        raise SystemExit("[deploy] Cargo.lock has no wasm-bindgen")
    return match.group(1)


def check_tooling() -> tuple[str, str]:
    """Hold the CLI against the crate, and say what to do when they differ.

    `wasm-bindgen` the crate writes the wasm's custom sections; `wasm-bindgen`
    the CLI reads them. They are one format with two release trains, and a CLI
    older than the crate fails loudly while a newer one has, historically, done
    worse -- so this refuses on any difference at all rather than trying to
    decide which mismatches are benign.

    Until 2026-09-29 it compared against `Cargo.toml`'s `"0.2"`, which held
    nothing: every 0.2.x passed, and the command it printed on a mismatch,
    `--version 0.2`, is one cargo refuses.
    """
    wanted = crate_wasm_bindgen_version()
    install = f"cargo install wasm-bindgen-cli --version {wanted} --locked"
    path = require("wasm-bindgen", install)
    cli = tool_version(path)  # e.g. "wasm-bindgen 0.2.128"
    installed = cli.split()[-1]
    if installed != wanted:
        raise SystemExit(
            f"[deploy] wasm-bindgen CLI is {installed}, the crate builds {wanted} "
            f"(deploy/fsm/Cargo.lock).\n"
            f"           They share an ABI and version separately; a mismatch produces "
            f"glue that loads and then misreads every argument.\n"
            f"           {install}"
        )
    return path, cli


def short(commit: str) -> str:
    """A commit for a human, keeping `-dirty`.

    `commit[:12]` dropped it, which was survivable while an uncommitted build
    needed a flag and is not now that it is the default: the line a person
    actually reads would show a clean hash for a build that is not.
    """
    return commit[:12] + ("-dirty" if commit.endswith("-dirty") else "")


def source_commit(require_clean: bool) -> str:
    """The commit the artifact is built from, with `-dirty` when it is.

    Read **after** the cleanliness check, never before. The one incident this
    script exists for was a provenance written from a `HEAD` that did not yet
    contain the source being built.

    The commit that last touched `SOURCES`, not the repository's `HEAD` and not
    the crate directory either. They differ constantly -- a task config, a
    document, this script, the crate's own README -- and a commit that cannot
    change a byte of the controller must not make two apps carrying the same
    controller look as if they differ.
    """
    paths = list(SOURCES)
    dirty = run(["git", "status", "--porcelain", "--", *paths], capture=True)
    if dirty and require_clean:
        raise SystemExit(
            "[deploy] --require-clean, and the controller's source has uncommitted changes:\n"
            + "\n".join("  " + line for line in dirty.splitlines())
            + "\n           An artifact built from them cannot be reproduced by "
              "anybody, including you tomorrow."
        )
    if dirty:
        # Said, not refused. This used to be fatal unless `--allow-dirty` was
        # passed, which is right for something being published and wrong for
        # every other build: it made the test suite red for exactly as long as
        # somebody was editing the crate, and two commits went in on a red run
        # because the failure looked like cargo contention and was this.
        #
        # The discipline it was protecting survives in the `-dirty` suffix
        # below, which lands in `bundle.json`'s `runtime.commit` and cannot be
        # mistaken for something reproducible. (`built_from` is the per-mode
        # provenance -- which checkpoint each policy came from -- and is a
        # different thing.) `--require-clean` is for the
        # build that is going somewhere.
        print("[deploy] building from an uncommitted controller source; the provenance "
              "will say '-dirty':")
        for line in dirty.splitlines():
            print("           " + line)
    last = run(["git", "log", "-1", "--format=%H", "--", *paths], capture=True)
    if not last:
        raise SystemExit(f"[deploy] {CRATE} has no commits")
    return last + ("-dirty" if dirty else "")


def compile_extension(cargo: str, verbose: bool) -> Path:
    """Build the extension `play` imports and return where cargo left it.

    Not maturin. The crate is already a `cdylib`, `abi3-py310` makes one file
    serve every interpreter this repository supports, and `cargo` is the whole
    of it -- adding a build backend to `pip install -e .` for that would be a
    dependency nobody could skip on a machine that only trains.
    """
    quiet = [] if verbose else ["--quiet"]
    run([cargo, "build", "--release", "--no-default-features", "--features", "py", *quiet],
        cwd=CRATE)
    for name in ("libmjrl_fsm.so", "libmjrl_fsm.dylib"):
        built = CRATE / "target/release" / name
        if built.is_file():
            return built
    raise SystemExit(f"[deploy] cargo produced no extension under {CRATE}/target/release")


def build(cargo: str, verbose: bool) -> None:
    """Test the core, then compile it.

    `--no-default-features` is the same build the wasm gets: no DDS, no NPU, and
    so no CycloneDDS needed on the machine doing the publishing. Shipping a
    controller that fails its own tests is the one failure worth a few seconds.
    """
    quiet = [] if verbose else ["--quiet"]
    run([cargo, "test", *quiet, "--no-default-features"], cwd=CRATE)
    run([cargo, "build", "--release", "--target", "wasm32-unknown-unknown",
         "--no-default-features", "--features", FEATURES, *quiet], cwd=CRATE)


def bindgen(tool: str, staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    run([tool, "--target", "web", "--no-typescript", "--out-name", "controller",
         "--out-dir", str(staging), str(WASM)], cwd=CRATE)

    # wasm-bindgen writes `controller_bg.wasm` and points the glue's fallback at
    # that name. Renaming without rewriting the reference would leave a bundle
    # whose two files disagree about each other: harmless for a consumer that
    # passes bytes -- both of ours do -- and a trap for the first one that does
    # not, which would fetch a 404 and see "failed to compile module".
    generated = staging / "controller_bg.wasm"
    if not generated.is_file():
        raise SystemExit(f"[deploy] wasm-bindgen produced no {generated.name}")
    generated.rename(staging / ARTIFACTS[1])
    glue = staging / ARTIFACTS[0]
    text = glue.read_text("utf-8")
    if text.count("controller_bg.wasm") != 1:
        raise SystemExit(
            f"[deploy] expected exactly one reference to controller_bg.wasm in "
            f"{glue.name}, found {text.count('controller_bg.wasm')}. wasm-bindgen's "
            f"output shape has changed; check what else moved before renaming."
        )
    glue.write_text(text.replace("controller_bg.wasm", ARTIFACTS[1]), "utf-8")

    missing = [name for name in ARTIFACTS if not (staging / name).is_file()]
    if missing:
        raise SystemExit(f"[deploy] wasm-bindgen produced no {missing}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_record(commit: str, cli: str) -> dict:
    """Which source the controller builds came from, and how the wasm was made.

    `runtimes.web` in `bundle.json` carries the `build` half; the README the
    commit.
    """
    return {
        "source": {"repo": "KingKongRobotics/jumper", "path": "deploy/fsm", "commit": commit},
        "build": {
            "tool": "scripts/deploy.py",
            "cargo": f"build --release --target wasm32-unknown-unknown "
                     f"--no-default-features --features {FEATURES}",
            "wasmBindgen": cli,
        },
    }


# ── Upload bundles ───────────────────────────────────────────────────────


#: The build's input, maintained by hand and checked in.
#:
#: Everything a bundle is made of used to live on the command line -- a
#: `--mode name=<dir>` per mode, pointing at a directory somebody exported at
#: some point -- and the one fact nobody wrote down anywhere was *which
#: checkpoint that was*. A bundle recorded its own contents and not its
#: ancestry, so "rebuild the thing we shipped" meant remembering.
#:
#: It names **exports**, never checkpoints. A checkpoint is not something a
#: robot can run: exporting is where the contract is built and where the
#: validations decide whether it can run at all. Which checkpoint an export came
#: from is recorded inside it and read back into `built_from`, so the ancestry
#: survives without the manifest holding a second copy of it.
#:
#: It also splits the two halves of a control scheme that had drifted apart.
#: `tasks/<task>/controls.yaml` says how a person drives a policy -- which key
#: nudges which axis -- and lives with the task. Which key *switches into* that
#: task is a property of the bundle, not of the task, and until now existed only
#: as a hand-written binding with nothing tying it to the mode it names.
MANIFESTS = CRATE.parent / "manifests.json"
MANIFESTS_SCHEMA = "kk-deploy-manifests/1"


def read_manifests(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"[deploy] no manifest file at {path}")
    data = json.loads(path.read_text("utf-8"))
    if data.get("schema") != MANIFESTS_SCHEMA:
        raise SystemExit(
            f"[deploy] {path} says schema {data.get('schema')!r}; "
            f"this reads {MANIFESTS_SCHEMA!r}"
        )
    bundles = data.get("bundles") or {}
    if not bundles:
        raise SystemExit(f"[deploy] {path} defines no bundles")
    return bundles


def resolve_manifest(name: str | None, path: Path) -> tuple[str, dict]:
    """Pick one bundle out of the file, or the only one there is."""
    bundles = read_manifests(path)
    if name is None:
        if len(bundles) > 1:
            raise SystemExit(
                f"[deploy] {path} defines {len(bundles)} bundles; --manifest <name>.\n"
                + "".join(f"           {n:<16}{b.get('description', '')}\n"
                          for n, b in sorted(bundles.items()))
            )
        name = next(iter(bundles))
    if name not in bundles:
        raise SystemExit(
            f"[deploy] {path} has no bundle {name!r}. It has: "
            + ", ".join(sorted(bundles))
        )
    return name, bundles[name]


def manifest_modes(spec: dict, path: Path) -> tuple[dict[str, Path], dict[str, dict]]:
    """Turn a manifest's modes into export directories.

    A mode names a `task` and an `export` directory, and only that.

    Not a checkpoint. A checkpoint is not a policy a robot can run -- exporting
    is where the contract is built and where ten validations decide whether it
    can run at all -- so naming one here means the bundler picks an export on
    the reader's behalf. It did, briefly: the newest matching one. That is a
    silent choice about what ships, made at build time, from a field that looks
    like it names a thing.

    The ancestry is not lost by leaving it out. Each export records the
    checkpoint it came from, and `built_from` reads it back, so `bundle.json`
    still says which run produced what shipped -- without the manifest carrying
    a second copy that could disagree with the directory beside it.

    A missing export is not an error about the manifest. The manifest is right
    and the export has not been run, so it prints the command.

    A mode may also carry a `hook`: its task's deploy configuration, a JSON
    object handed verbatim to `tasks/<task>/deploy/lib.rs` on the robot
    (`deploy/fsm/src/hook.rs`). `jumper.five_foot`'s is `{"side": "right"}`,
    the claw carried on the side the policy was not trained on.

    And the switches into it, one per device: `pad` and `keys` (see
    `_switch`). Until 2026-09-29 a mode had one `button`, one `key` and one
    gesture and modifier for both, and the bundle a `keyboard` table making keys
    pad buttons; both are refused by name.
    """
    modes: dict[str, Path] = {}
    detail: dict[str, dict] = {}
    for mode, entry in (spec.get("modes") or {}).items():
        task = entry.get("task")
        if not task:
            raise SystemExit(f"[deploy] {path}: mode {mode!r} names no task")
        if "checkpoint" in entry:
            raise SystemExit(
                f"[deploy] {path}: mode {mode!r} names a checkpoint. This takes an "
                f"export.\n"
                f"           A checkpoint is not something a robot can run: exporting "
                f"is where the contract is built and where the validations decide "
                f"whether it can run at all.\n"
                f"           python scripts/export.py --task {task} "
                f"--checkpoint {entry['checkpoint']}\n"
                f"           then put the directory it prints under `policy`."
            )
        if "policy" not in entry:
            raise SystemExit(f"[deploy] {path}: mode {mode!r} names no `policy`")
        where = (REPO / entry["policy"]).resolve()
        if not where.is_dir():
            raise SystemExit(
                f"[deploy] {path}: mode {mode!r} names an export that is not there:\n"
                f"           {where}\n"
                f"           Exports are timestamped and not tracked, so a manifest "
                f"from another machine names one this one has never built.\n"
                f"           python scripts/export.py --task {task} --checkpoint "
                f"<ckpt>, then put the directory it prints under `policy`."
            )
        # The checkpoint, from the export itself. Read rather than restated, so
        # `bundle.json` cannot say one run while the directory beside it came
        # from another.
        try:
            recorded = json.loads((where / "layout.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            recorded = {}
        ckpt = (recorded.get("_checkpoint") or {}).get("path")

        switches = {device: _switch(entry[device], device, f"mode {mode!r}", path)
                    for device in ("pad", "keys") if entry.get(device) is not None}

        hook = entry.get("hook")
        if hook is not None and not isinstance(hook, dict):
            raise SystemExit(
                f"[deploy] {path}: mode {mode!r}'s `hook` is its task's deploy "
                f"configuration, a JSON object; this one is {hook!r}.\n"
                f"           tasks/{task.replace('.', '/')}/deploy/lib.rs says what it "
                f"takes."
            )

        modes[mode] = where
        detail[mode] = {"task": task, "from": ckpt or "(the export records no checkpoint)",
                        "export": str(where.relative_to(REPO))
                        if where.is_relative_to(REPO) else str(where),
                        # `null` reads as "not bound", so the field can be
                        # present and say so. A mode with neither is reachable
                        # only from the cascade -- fine for one mode, and an
                        # unreachable state for any of the others.
                        **({"hook": hook} if hook is not None else {}),
                        **switches}
    if not modes:
        raise SystemExit(f"[deploy] {path}: this bundle has no modes")
    return modes, detail


#: What a switch may say, per device: the control it is on, the gesture, the
#: modifier held with it, and the modes it may be pressed in.
_SWITCH_FIELDS = {"pad": ("button", "on", "with", "from"), "keys": ("key", "on", "with", "from")}


def _switch(entry, device: str, where: str, path: Path) -> dict:
    """One device's switch, as the manifest writes it, checked for its shape.

    `pad: {"button": "A", "on": "toggle", "from": ["locomotion"]}` and
    `keys: {"key": ["key_1", "keypad_1"], "with": "ctrl"}`: a pad button or a
    list of keys, the gesture (`toggle` when left out), the modifier held with
    it -- a pad button for the pad, `ctrl` / `shift` / `alt` for the keys --
    and `from`, the modes it may be pressed in (any, left out). What each word
    may be is the crate's to check, against the dictionary, when it loads the
    composed file; here only the shape, so a typo in a field name is refused
    rather than a switch silently losing its `from`.
    """
    fields = _SWITCH_FIELDS[device]
    control = fields[0]
    if not isinstance(entry, dict) or control not in entry:
        raise SystemExit(f"[deploy] {path}: {where}'s `{device}` is {entry!r}; it needs "
                         f"`{control}` and may carry {list(fields[1:])}")
    stray = sorted(k for k in entry if k not in fields and not k.startswith("_"))
    if stray:
        raise SystemExit(f"[deploy] {path}: {where}'s `{device}` has {stray}; a switch "
                         f"says {list(fields)}")
    keys = entry[control]
    if device == "keys" and (not isinstance(keys, list) or not keys
                             or not all(isinstance(k, str) for k in keys)):
        raise SystemExit(f"[deploy] {path}: {where}'s `keys.key` is {keys!r}; it is a "
                         f"non-empty list of key names")
    frm = entry.get("from")
    if frm is not None and (not isinstance(frm, list) or not all(isinstance(m, str) for m in frm)):
        raise SystemExit(f"[deploy] {path}: {where}'s `{device}.from` is {frm!r}; it is a "
                         f"list of mode names")
    return {k: entry[k] for k in fields if entry.get(k) is not None}


def _bindings(name: str, switches: dict, *, event: bool = False,
              leaves: list | None = None) -> str:
    """The `[[fsm.button]]` entries for one name: one for the pad's switch and
    one per key of the keyboard's, all named alike, so they share one latch."""
    out = ""
    for device, sw in switches.items():
        controls = [("pad", sw["button"])] if device == "pad" else [("key", k) for k in sw["key"]]
        for field, control in controls:
            out += f'\n[[fsm.button]]\nname = "{name}"\n{field} = {json.dumps(control)}\n'
            out += f'on = "{sw.get("on", "toggle")}"\n'
            if "with" in sw:
                out += f'with = {json.dumps(sw["with"])}\n'
            if "from" in sw:
                out += f'from = {json.dumps(sw["from"])}\n'
            if leaves:
                out += f"leaves = {json.dumps(leaves)}\n"
            if event:
                out += "event = true\n"
    return out


def compose_fsm(spec: dict, detail: dict[str, dict], models: dict[str, str],
                path: Path) -> str:
    """Merge the manifest's modes and bindings into the FSM the manifest names.

    The split is the point. The **manifest** owns which modes exist, where each
    one came from and which key or button reaches it. The **FSM file** owns the
    cascade, the safety parameters and any state that runs no model. Neither may
    say the other's half, because the failure that produces is two files
    disagreeing about which key does what and nothing reading both.
    """
    fsm_path = (path.parent / spec["fsm"]).resolve() if "fsm" in spec else None
    if fsm_path is None:
        raise SystemExit(f"[deploy] {path}: this bundle names no `fsm`")
    if not fsm_path.is_file():
        raise SystemExit(f"[deploy] {path}: no FSM at {fsm_path}")
    text = fsm_path.read_text("utf-8")

    stale = [t for t in ("[fsm.key]", "[fsm.keyboard]", "[fsm.button]", "[[fsm.button]]")
             if t in text]
    if model_states(text):
        stale.append("a [[fsm.state]] with a model")
    if stale:
        raise SystemExit(
            f"[deploy] {fsm_path.name} declares {', and '.join(stale)}, which the "
            f"manifest owns.\n"
            f"           Delete them there: two files naming the same key is how the "
            f"two come to disagree."
        )

    # Each mode's task goes in beside its model, and its `hook` if it has one:
    # the controller finds a task's deploy library by that id and hands it the
    # table (`deploy/fsm/src/hook.rs`). A mode without `hook` runs without one.
    states = "".join(
        f'\n[[fsm.state]]\nname = "{mode}"\nmodel = "{models[mode]}"\n'
        f'task = "{detail[mode]["task"]}"\n'
        + (f'hook = {_toml_value(detail[mode]["hook"], f"{mode}.hook")}\n'
           if "hook" in detail[mode] else "")
        for mode in sorted(detail)
    )
    # Each device's switch into a mode is a binding of its own, named after the
    # mode, so the pad's A and the keyboard's Space share the jump's one latch;
    # `from` goes with each (since 2026-09-29, when the keyboard stopped being a
    # table of pad buttons).
    #
    # `on` defaults to `toggle`, which is what switching into a mode means --
    # you press it and stay there. A manifest can say `rise`, `fall`, `hold` or
    # a click count (`double`, ...) instead; press-to-arm and release-to-go is
    # one mode named twice with two gestures, and the crate checks the
    # vocabulary either way.
    #
    # Bindings that are not mode switches. A recorded motion's `go` is the
    # case: the jump's push-off is 40 ms wide, so it is an event *inside* a
    # mode rather than a transition between them. Still an ordinary binding --
    # same dictionary, same gestures, same modifier rules -- but latched by a
    # contract's `go_event` instead of by a cascade rule, which is what
    # `event = true` tells the crate. It checks both directions: an event no
    # mode reads, and a `go_event` no button declares, are each a control that
    # loads and does nothing.
    tables = ""
    for i, b in enumerate(spec.get("buttons", [])):
        where = f"buttons[{i}]"
        if "name" not in b or ("pad" not in b and "keys" not in b):
            raise SystemExit(
                f"[deploy] {path}: {where} needs `name` and a `pad` or `keys` switch; it "
                f"has {sorted(b)}.\n           python -m controller --vocabulary"
            )
        switches = {d: _switch(b[d], d, where, path) for d in ("pad", "keys") if d in b}
        tables += _bindings(b["name"], switches, event=True)

    # Leaving: a moment that lets go of the modes it names, and the cascade
    # falls through to its `always` mode. The jumper bundle's Menu and Ctrl
    # interrupt its dances and fixed actions this way.
    for i, b in enumerate(spec.get("leave", [])):
        where = f"leave[{i}]"
        leaves = b.get("leaves")
        if "name" not in b or not isinstance(leaves, list) or not leaves or (
                "pad" not in b and "keys" not in b):
            raise SystemExit(
                f"[deploy] {path}: {where} needs `name`, `leaves` (the modes it lets go "
                f"of) and a `pad` or `keys` switch; it has {sorted(b)}"
            )
        stray = sorted(m for m in leaves if m not in detail)
        if stray:
            raise SystemExit(f"[deploy] {path}: {where} leaves {stray}, which are not modes "
                             f"of this bundle ({sorted(detail)})")
        switches = {d: _switch(b[d], d, where, path) for d in ("pad", "keys") if d in b}
        tables += _bindings(b["name"], switches, leaves=leaves)

    for mode, d in sorted(detail.items()):
        switches = {device: d[device] for device in ("pad", "keys") if device in d}
        tables += _bindings(mode, switches)

    return (
        f"# Composed by scripts/deploy.py from {MANIFESTS.name} and {fsm_path.name}.\n"
        f"# The modes, the keys and the buttons come from the manifest; everything\n"
        f"# else is {fsm_path.name}'s. Editing this file edits neither.\n"
        + text.rstrip("\n") + "\n" + tables + states
    )


def _toml_value(value, where: str) -> str:
    """One JSON value as TOML, inline: a hook's configuration, into its state.

    Strings go through `json.dumps`, whose escapes are all valid in a TOML basic
    string; a table stays inline, so a state is still one block. A JSON `null`
    has no TOML spelling and is refused rather than dropped -- a key that
    silently vanished is a configuration the hook never saw.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v, f"{where}[]") for v in value) + "]"
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            key = k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)
            items.append(f"{key} = {_toml_value(v, f'{where}.{k}')}")
        return "{ " + ", ".join(items) + " }" if items else "{}"
    raise SystemExit(f"[deploy] {where} is {value!r}, which a TOML file cannot carry")


def synthesise_fsm(mode: str) -> str:
    """The one-mode FSM, for a bundle that carries a single policy.

    Written into the bundle rather than left to the consumer's default, so the
    bundle says what it runs under. A consumer with its own default still has
    one -- for bundles made before this existed -- but a bundle that brought a
    file and a bundle that did not should not be indistinguishable.

    No tilt rule. The robot drops to a damping hold past a limit because what it
    is protecting is hardware; a simulator has none, and a robot that gives up
    when it tips over is worse to watch than one that keeps trying. A bundle
    aimed at hardware writes its own file.
    """
    return f"""# Written by scripts/deploy.py for a single-policy bundle.
# Replace this with your own to declare more modes, the keys that switch
# between them, or a tilt fallback.
[fsm]
initial_state = "{mode}"
warm_start_ref = "{mode}"
safe_state = "safe"
tilt_limit = 3.15
state_timeout_ms = 200
command_timeout_ms = 3600000
mode_switch_ramp_s = 0.0
pose_reach_tol = 0.10
warm_start_duration_s = 0.0
ramp_kp = 0.15
ramp_kd = 0.01

[[fsm.state]]
name = "safe"
hold_current = true
kd = 0.01

[[fsm.state]]
name = "{mode}"
model = "{mode}.onnx"

[[fsm.rule]]
when = "feedback_stale"
enter = "safe"

[[fsm.rule]]
when = "in_state:safe"
enter = "@initial"

[[fsm.rule]]
when = "always"
enter = "{mode}"
"""


def model_states(fsm_text: str) -> dict[str, str]:
    """`{state name: model filename}` for every state that names one.

    Parsed here only to cross-check the `--mode` arguments against the file. The
    authority on whether the FSM is valid is `FsmConfig::parse` inside the
    controller, which the consumer runs and which refuses loudly -- this is a
    courtesy that catches a typo before the upload rather than after.
    """
    import tomllib

    try:
        doc = tomllib.loads(fsm_text)
    except tomllib.TOMLDecodeError as error:
        raise SystemExit(f"[deploy] the FSM config is not valid TOML: {error}") from None
    states = doc.get("fsm", {}).get("state", [])
    return {s["name"]: s["model"] for s in states if s.get("model")}


def parse_modes(entries: list[str]) -> dict[str, Path]:
    modes: dict[str, Path] = {}
    for entry in entries:
        name, _, where = entry.partition("=")
        if not name or not where:
            raise SystemExit(f"[deploy] --mode wants NAME=DIR, got {entry!r}")
        if name in modes:
            raise SystemExit(f"[deploy] --mode {name} given twice")
        modes[name] = Path(where)
    return modes


def verify_bundle(cargo: str, out: Path, verbose: bool) -> list[str]:
    """Load the bundle through the crate, the way a consumer will.

    The bundler writes an FSM config and a contract per mode and has no way to
    know either is loadable: Python cannot run `FsmConfig::parse`, and
    re-implementing it here would be the second implementation this whole crate
    exists to avoid. So it asks the crate -- the same code the browser runs,
    minus the wasm.

    A bundle that fails here would have failed on upload, with the visitor left
    to work out why.
    """
    result = subprocess.run(
        [cargo, "run", "--quiet", "--example", "check_bundle",
         "--no-default-features", "--", str(out)],
        cwd=CRATE, text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        print(result.stdout, end="")
        print(result.stderr, file=sys.stderr, end="")
        # A directory that exists is a directory somebody will try.
        shutil.rmtree(out, ignore_errors=True)
        raise SystemExit(f"[deploy] the bundle does not load; {out} was removed")
    if verbose:
        print(result.stdout, end="")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def write_reference(out: Path, extension: Path | None,
                    modes: dict[str, Path]) -> list[str]:
    """Record what this controller does, frame by frame, so three hosts can be
    held against it instead of against each other.

    The three hosts run the same crate, which is the reason to believe they
    agree and not a reason to stop checking: they compile it with different
    toolchains for different architectures, and two of them run the policy on a
    different inference backend. "Same source" survives all of that; "same
    numbers" is a measurement, and nobody had taken it.

    Each frame carries **two independent comparisons**, and keeping them apart
    is the point -- a wrong observation and a quantised model produce the same
    symptom, a robot that walks slightly wrong:

      * `obs` and `target` check the *controller*. Same input in, same
        observation and same joint targets out. Any difference at all is this
        crate behaving differently on that platform.
      * `act` checks the *model*. A host runs its own backend on `obs` and
        compares. RKNN is quantised and will differ in the third decimal;
        onnxruntime-web should not differ at all.

    The states are synthetic and seeded, not a recorded rollout: this script
    reads no environment, and what is being measured is agreement rather than
    realism. They start away from the home pose and converge on it, so a config
    with a warm start or a ramp has both on the record alongside the policy --
    a reference covering only the steady state agrees everywhere and says
    nothing. The note reports how many frames landed in each, because a
    reference of 24 inferences and no ramp is worth less than it looks.

    Returns notes. Absent rather than fatal when the extension or onnxruntime is
    missing: a bundle without a reference is a bundle nobody can check, which is
    exactly where this started, so it says so.
    """
    manifest = json.loads((out / "bundle.json").read_text("utf-8"))
    if extension is None:
        return ["no reference vectors: this bundle carries no native controller to "
                + "generate them with"]
    try:
        import numpy as np
        import onnxruntime
    except ImportError as e:
        return [f"no reference vectors: {e.name} is not installed"]

    fsm = _load_for_reference(out, manifest, extension)
    if isinstance(fsm, str):
        return [f"no reference vectors: {fsm}"]

    # The mode the cascade runs with nobody touching anything -- the one these
    # frames converge on and infer in -- rather than the first mode by name,
    # whose home may be another policy's entirely: `jumper`'s `claw_left` sorts
    # first and carries an arm 1.6 rad from the walking stance, so the frames
    # converged on a pose the warm start never reaches and not one inferred.
    import tomllib

    cfg = tomllib.loads((out / "controller.toml").read_text("utf-8")).get("fsm", {})
    start = next((m for m in (cfg.get("warm_start_ref"), cfg.get("initial_state"))
                  if m in manifest["modes"]), next(iter(manifest["modes"])))
    contract = json.loads((out / manifest["modes"][start]["contract"]).read_text("utf-8"))
    wire = contract["wire_joint_order"]
    # Every channel a controls block names, so the reference drives
    # them too. A reference commanding the twist alone would leave
    # `jumper.posture`'s posture channels at rest in every frame, and a builder
    # that dropped them -- or ordered them like `base_pose` -- would agree with
    # it everywhere. The running mode's own block first: each mode reads its
    # own controls, and another mode's axes are not the ones it observes.
    operator = next(
        (driven for mode in [start, *manifest["modes"]]
         for driven in [json.loads((out / manifest["modes"][mode]["contract"])
                                   .read_text("utf-8"))]
         if (driven.get("controller") or {}).get("schema") == "operator_controller/2"),
        None,
    )
    # Away from rest on every axis, and inside every trained range: a quarter of
    # the way along the positive end, from each axis's own rest.
    axes = {}
    if operator:
        for command in operator["controller"]["command"]:
            ranges = operator.get("command_ranges", {}).get(command["term"], {})
            for axis in command["axes"]:
                rest = float(axis.get("rest", 0.0))
                hi = ranges.get(axis["name"], [rest, rest])[1]
                axes[axis["name"]] = rest + 0.25 * (hi - rest)
    home = [contract["default_joint_pos"][j] for j in wire]
    n = len(wire)

    # From the **export directory**, not from the bundle: a device bundle ships
    # `.rknn`, which only an NPU runs, and the ONNX is exactly what that `.rknn`
    # should be held against. So both packagings get a reference computed the
    # same way, and the board's check is "does my NPU agree with the ONNX".
    sessions = {
        mode: onnxruntime.InferenceSession(str(source / "actor.onnx"),
                                           providers=["CPUExecutionProvider"])
        for mode, source in modes.items()
    }

    # Seeded, so two builds of one source produce identical bytes and `--check`
    # keeps meaning something.
    rng = np.random.default_rng(20260920)
    away = (np.array(home) + rng.uniform(-0.25, 0.25, n)).tolist()

    frames = []
    for i in range(REFERENCE_FRAMES):
        # Toward home, so the ramp completes partway through and the policy runs
        # for the rest. Both regimes, one reference.
        blend = min(1.0, i / (REFERENCE_FRAMES / 3))
        q = [a * (1 - blend) + h * blend for a, h in zip(away, home)]
        qd = rng.uniform(-0.5, 0.5, n).tolist()
        tau = rng.uniform(-0.4, 0.4, n).tolist()
        # Near upright, normalised: a quaternion that is not unit length is a
        # different input, and the drift would look like a controller bug.
        quat = rng.normal([1.0, 0.0, 0.0, 0.0], [0.0, 0.03, 0.03, 0.03])
        quat = (quat / np.linalg.norm(quat)).tolist()
        gyro = rng.uniform(-0.3, 0.3, 3).tolist()
        cmd = [0.3, 0.0, 0.15]
        now = i * 20_000

        fsm.set_state(q, qd, tau, quat, gyro, now)
        if axes:
            fsm.set_command_axes(list(axes), list(axes.values()), now_us=now)
        else:
            fsm.set_command(cmd[0], cmd[1], cmd[2], now_us=now)
        mode = fsm.tick(now)
        frame = {"now_us": now, "q": q, "qd": qd, "tau": tau, "quat": quat,
                 "gyro": gyro, "cmd": cmd, "mode": fsm.mode}
        if axes:
            frame["axes"] = axes
        if mode is not None:
            obs = list(fsm.observation())
            act = sessions[mode].run(
                None, {sessions[mode].get_inputs()[0].name:
                       np.asarray([obs], dtype=np.float32)})[0][0].tolist()
            fsm.resume(act)
            frame |= {"infer": mode, "obs": obs, "act": act}
        frame["target"] = list(fsm.positions())
        frames.append(frame)

    inferred = sum("obs" in f for f in frames)
    (out / "reference.json").write_text(
        json.dumps({
            "schema": REFERENCE_SCHEMA,
            "source": "scripts/deploy.py, this crate's own extension plus onnxruntime",
            # Every runtime is built from one commit; this read a `controller`
            # key no bundle has ever had, and recorded null.
            "commit": next((r.get("commit") for r in manifest.get("runtimes", {}).values()),
                           None),
            "note": "`obs` and `target` check the controller and should match exactly. "
                    "`act` checks the inference backend: RKNN is quantised and differs in "
                    "the third decimal, onnxruntime-web should not differ at all.",
            "joints": wire,
            # Where each term sits in the observation, so a host can say *which*
            # term disagrees instead of which index. The terms whose source
            # differs between hosts by design -- `joint_torque` is measured in
            # mjlab and reconstructed on the robot -- are the reason this is not
            # one number: without it a real difference and a designed one are
            # the same failure.
            "terms": _term_offsets(contract),
            "frames": frames,
        }, separators=(",", ":")) + "\n",
        "utf-8",
    )
    return [f"reference: {len(frames)} frames, {inferred} of them inferences "
            + f"({REFERENCE_FRAMES - inferred} in warm start or ramp)"]


def check_reference(bundle: Path) -> int:
    """Replay a bundle's reference through the controller the bundle carries.

    This is the `play` host's side of the three-way check -- the same extension
    `play --app` imports, driven through the same frames the robot replays with
    `controller --check-reference`. Two hosts, one set of numbers, and the
    difference between them is now a measurement rather than an assumption.
    """
    if not (bundle / "bundle.json").is_file():
        # Pointed at the directory `--bundle` writes, which holds the bundle
        # and its `.app` side by side.
        below = sorted(p.parent for p in bundle.glob("*/bundle.json"))
        if len(below) == 1:
            bundle = below[0]
    manifest_path = bundle / "bundle.json"
    if not manifest_path.is_file():
        raise SystemExit(f"[deploy] {bundle} has no bundle.json")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    reference = bundle / (manifest.get("reference") or "reference.json")
    if not reference.is_file():
        raise SystemExit(
            f"[deploy] {bundle} carries no reference.json, so there is nothing to "
            f"check against.\n           Rebuild it on a machine with onnxruntime."
        )
    # This interpreter checks the play host's build for this machine. The
    # board checks its own with `runtime/board/controller --check-reference`,
    # a browser with `checkReference`.
    from mjrl.app_play import extension_for

    platform = sysconfig.get_platform()
    built = (manifest.get("runtimes", {}).get("mjlab") or {}).get("extensions", {})
    key = extension_for(built, platform)
    entry = built[key] if key else None
    extension = bundle / entry["file"] if entry else None
    if extension is None or not extension.is_file():
        raise SystemExit(
            f"[deploy] {bundle} carries no controller extension for {platform}"
            + (f" (it has: {', '.join(sorted(built))})" if built else "")
            + ", so there is nothing here this interpreter can check.\n"
            "           The board checks its own: runtime/board/controller --bundle "
            "<dir> --check-reference"
        )

    fsm = _load_for_reference(bundle, manifest, extension)
    if isinstance(fsm, str):
        raise SystemExit(f"[deploy] {fsm}")
    report = fsm.check_reference(reference.read_text("utf-8"))

    print(f"[deploy] {bundle.parent.name}/{bundle.name}: "
          f"{report.frames} frames, {report.inferences} inferences")
    print(f"           observation  worst {report.observation:.3e}")
    print(f"           targets      worst {report.target:.3e}")
    differ = [(t, w) for t, w in report.divergent if w > 0.0]
    for term, worst in differ:
        print(f"           {term:<12} worst {worst:.3e} -- sourced differently here")
    for frame, want, got in report.mode_mismatches:
        print(f"           frame {frame}: the cascade chose '{got}', the reference has "
              f"'{want}'")
    # The controller is the same f32 arithmetic in the same order on this host,
    # so the honest expectation is bit-exact. `controller` holds the board to the
    # same number.
    tol = 1e-6
    bad = report.observation > tol or report.target > tol or report.mode_mismatches
    if bad:
        print(f"[deploy] this host does not match the reference (tolerance {tol:.0e})")
        return 1
    print("[deploy] this host matches the reference")
    return 0


def _term_offsets(contract: dict) -> list[dict]:
    offsets, at = [], 0
    for term in contract["observation"]["terms"]:
        offsets.append({"name": term["name"], "offset": at, "dim": term["dim"]})
        at += term["dim"]
    return offsets


def _load_for_reference(out: Path, manifest: dict, extension: Path, *,
                        need_limits: bool = True):
    """Import the bundle's own extension and build the controller from it.

    The bundle's, not the installed one: a reference generated by a different
    build of the same source is a reference that measures nothing, and those two
    were already two commits apart once.

    Returns the controller, or a string saying why not.
    """
    import importlib.machinery
    import importlib.util

    # The spec name is not free: CPython looks for `PyInit_<name>` and the
    # crate's `#[pymodule]` is `mjrl_fsm`, so any other name fails with
    # "dynamic module does not define module export function" -- which reads
    # like a broken build rather than a naming rule. The loader is named rather
    # than inferred from the suffix, which `importlib` matches against this
    # interpreter's own list. `mjrl/app_play.py` has the long version of this
    # comment; it is the other caller.
    loader = importlib.machinery.ExtensionFileLoader("mjrl_fsm", str(extension))
    spec = importlib.util.spec_from_file_location("mjrl_fsm", extension, loader=loader)
    if spec is None or spec.loader is None:
        return f"cannot load {extension}"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        return f"cannot import the controller: {e}"

    contracts = {m: (out / e["contract"]).read_text("utf-8")
                 for m, e in manifest["modes"].items()}
    first = json.loads(next(iter(contracts.values())))
    wire = first["wire_joint_order"]
    limits = first.get("joint_limits")
    if limits is None and need_limits:
        return "this bundle's contracts carry no joint_limits"
    if limits is None:
        # Asked only for what it binds (`pad_guide`), which no clamp changes.
        # Whether a host may run it without the stops is the host's check.
        limits = {j: [-3.2, 3.2] for j in wire}
    # From **whichever** mode observes `gait_phase`, not from the first contract
    # that happens to be read. `Bundle::gait` says the same thing in Rust, and
    # the two have to agree or the reference vectors are generated against a
    # different clock than the one a host will run. A bundle whose first mode is
    # a jump has no gait term at all, and this read `None` off it and handed
    # that to the extension, which reported it as a type error about a real
    # number -- true, and not a sentence anybody could act on.
    gait: dict = {}
    for text in contracts.values():
        params = next((t.get("params", {}) for t in json.loads(text)["observation"]["terms"]
                       if t["name"] == "gait_phase"), None)
        if params and params.get("period") is not None:
            gait = params
            break

    # Omitted rather than passed as null: the extension's fields are optional,
    # and absent means "this bundle has no gait clock" where an explicit null is
    # a number that failed to arrive.
    robot = {
        "joint_names": wire,
        "joint_pos_lo": [limits[j][0] for j in wire],
        "joint_pos_hi": [limits[j][1] for j in wire],
        "output_rate_hz": first["control"]["control_hz"],
    }
    if gait.get("period") is not None:
        robot["gait_period"] = gait["period"]
    if gait.get("command_threshold") is not None:
        robot["gait_gate_threshold"] = gait["command_threshold"]
    # The tables a residual policy is a residual *on*. Read here because the
    # crate reads no files, which is the same reason the contracts are passed as
    # text: one code path for a browser with no filesystem and for this.
    trajectories = {}
    for mode, text in contracts.items():
        spec = json.loads(text).get("reference")
        if spec:
            f = out / spec["file"]
            if not f.is_file():
                return f"mode '{mode}': the bundle has no {spec['file']}"
            trajectories[mode] = f.read_text("utf-8")
    try:
        return module.Fsm(
            (out / "controller.toml").read_text("utf-8"),
            contracts,
            robot,
            0,
            trajectories,
        )
    except ValueError as e:
        return str(e)


#: Where each host's build of the controller travels inside a bundle, and where
#: the models do. Everything else -- `controller.toml`, the contracts, the
#: recordings, `reference.json` -- sits at the top, once, because every host
#: reads the same bytes of it.
RUNTIME_DIR = "runtime"
MODELS_DIR = "models"


#: The two containers that make the board's half of an app. Neither is run from
#: here by hand-rolled `docker` calls: each script carries the details -- the
#: image, the volumes, who owns the output -- that are wrong answers waiting to
#: happen, and says so in its own header.
CROSS_BUILD_SCRIPT = CRATE / "docker-build.sh"
CONVERT_SCRIPT = REPO / "deploy/convert/docker-convert.sh"


def by_export(modes: dict[str, Path]) -> dict[Path, list[str]]:
    """`{export directory: its modes}`. Two modes on one export share one model."""
    grouped: dict[Path, list[str]] = {}
    for mode, source in sorted(modes.items()):
        grouped.setdefault(source.resolve(), []).append(mode)
    return grouped


def converted(source: Path) -> bool:
    """Whether `source` carries an `.rknn` made from the `actor.onnx` beside it.

    The converter writes `actor.rknn.json` with the digest of the ONNX it
    converted. An `.rknn` without that record, or with another ONNX's, is
    converted again rather than trusted: a policy exported again into the same
    directory leaves the old `.rknn` beside it looking finished.
    """
    rknn, record = source / "actor.rknn", source / "actor.rknn.json"
    if not (rknn.is_file() and record.is_file() and (source / "actor.onnx").is_file()):
        return False
    try:
        made_from = json.loads(record.read_text("utf-8")).get("onnx_sha256")
    except (OSError, json.JSONDecodeError):
        return False
    return made_from == sha256(source / "actor.onnx")


def preflight(modes: dict[str, Path]) -> list[str]:
    """What would stop the board's half of this app being made here.

    Everything after this takes minutes -- a cross build, a conversion per
    policy, the controller built three times -- so what can be known now is asked
    now, all of it at once rather than one refusal per run.

    An export directory that is not there is left to `assemble`, which says so
    in its own words.
    """
    problems: list[str] = []
    if shutil.which("docker") is None:
        problems.append("docker is not installed, and the board's controller and play's "
                        "Windows and macOS ones are cross-built, and each policy converted, "
                        "in a container")
    for module in ("numpy", "onnxruntime"):
        if importlib.util.find_spec(module) is None:
            problems.append(f"{module} is not installed, and the reference vectors are "
                            "recorded with it. It is a dependency: pip install -e .")
    for source, names in by_export(modes).items():
        if source.is_dir() and not converted(source) and not source.is_relative_to(REPO):
            problems.append(
                f"{', '.join(names)}: {source} has no current actor.rknn and is outside "
                f"the repository, which is all the converter's container can see. "
                f"Export it under tasks/, or convert it there first."
            )
    return problems


def prepare_board(modes: dict[str, Path]) -> None:
    """Cross-build the board's controller and `play`'s Windows and macOS ones,
    and convert every policy that needs it.

    The cross builds run every time, not only when there is none. The bundler
    takes whatever is at `CROSS_BUILT` and in `PLAY_CROSS_BUILDS` and files it
    under this source's commit, so a binary from last week's source would ship
    as this week's -- and nothing about a binary says which source it came
    from. Unchanged, each is an incremental cargo build in the container's own
    target volume.

    A conversion runs only where `converted` says the `.rknn` is missing or was
    made from another ONNX: it is minutes per policy, and its result is checked
    against onnxruntime before it is written.
    """
    print("[deploy] cross-building the board's controller (deploy/fsm/docker-build.sh)")
    run(["bash", str(CROSS_BUILD_SCRIPT)])
    if not CROSS_BUILT.is_file():
        raise SystemExit(f"[deploy] {CROSS_BUILD_SCRIPT.name} succeeded and left no {CROSS_BUILT}")
    for who, built, cargo in PLAY_CROSS_BUILDS.values():
        print(f"[deploy] cross-building play's controller for {who} "
              "(deploy/fsm/docker-build.sh)")
        run(["bash", str(CROSS_BUILD_SCRIPT), *cargo])
        if not built.is_file():
            raise SystemExit(f"[deploy] {CROSS_BUILD_SCRIPT.name} succeeded and left no {built}")
    for source, names in by_export(modes).items():
        if not source.is_dir() or converted(source):
            continue
        where = source.relative_to(REPO)
        print(f"[deploy] converting {', '.join(names)} for the NPU ({where})")
        run(["bash", str(CONVERT_SCRIPT), "--bundle", str(where)])
        if not converted(source):
            raise SystemExit(f"[deploy] {CONVERT_SCRIPT.name} succeeded and {where} still has no "
                             f"actor.rknn made from its actor.onnx")


def gaps(bundle: dict) -> list[str]:
    """What a complete app carries and this one, as written, does not.

    Read off `bundle.json` rather than off the steps that should have made each
    part, because what counts is what a consumer receives.
    """
    lacking = [] if "board" in bundle["runtimes"] else ["no board controller"]
    extensions = (bundle["runtimes"].get("mjlab") or {}).get("extensions", {})
    lacking += [f"no {who} controller for play"
                for platform, (who, _, _) in PLAY_CROSS_BUILDS.items() if platform not in extensions]
    lacking += [f"{mode}: no .rknn" for mode, entry in sorted(bundle["modes"].items())
                if "rknn" not in entry["models"]]
    if bundle.get("reference") is None:
        lacking.append("no reference vectors")
    return lacking


def extension_file(platform: str) -> str:
    """Where `platform`'s extension travels in a bundle.

    `.pyd` on Windows: CPython there imports an extension by that suffix and by
    no other, and a `controller.so` beside a Windows interpreter is a file
    `importlib` makes no spec for.
    """
    name = "controller.pyd" if platform.startswith("win") else BUNDLE_EXTENSION
    return f"{RUNTIME_DIR}/mjlab/{platform}/{name}"


def runtimes(meta: dict, extension: Path | None) -> tuple[dict[str, Path], dict, list[str]]:
    """Every host's build of this crate that this machine produced.

    Returns `{path in the bundle: source}`, the `runtimes` block of
    `bundle.json`, and notes for what could not be built. A host whose build is
    missing has no entry, and a note says how to add it -- a bundle that looks
    complete is one somebody copies to a robot.

    Keyed by host, not by platform: a host knows which it is. The platform is
    recorded inside each entry, and the play host's is a map, because a native
    extension is built per machine and a bundle may carry one for several.
    """
    commit = meta["source"]["commit"]
    files: dict[str, Path] = {}
    blocks: dict[str, dict] = {}
    notes: list[str] = []

    glue, wasm = (f"{RUNTIME_DIR}/web/{name}" for name in ARTIFACTS)
    files[glue], files[wasm] = STAGING / ARTIFACTS[0], STAGING / ARTIFACTS[1]
    blocks["web"] = {
        "model": "onnx", "glue": glue, "wasm": wasm,
        **meta["build"], "commit": commit,
        "unverifiedByTheConsumer": (
            "A consumer's own controller is held against its own implementation by a "
            "parity test. This one is not held against anything: it arrived with the "
            "upload. A consumer that runs it should say so where the person can see it."
        ),
    }

    extensions: dict[str, dict] = {}
    if extension is not None:
        platform = sysconfig.get_platform()
        so = extension_file(platform)
        files[so] = extension
        extensions[platform] = {"file": so, "abi": "abi3-py310"}
    for platform, (who, built, cargo) in PLAY_CROSS_BUILDS.items():
        if built.is_file():
            files[extension_file(platform)] = built
            extensions[platform] = {"file": extension_file(platform), "abi": "abi3-py310"}
        else:
            notes.append(
                f"no {who} runtime for play: nothing has cross-built {built.name}. "
                f"bash deploy/fsm/docker-build.sh {' '.join(cargo)} -- then build this "
                "bundle again."
            )
    if extensions:
        blocks["mjlab"] = {
            "model": "onnx",
            "extensions": extensions,
            "commit": commit,
            "note": "The same source as the wasm, built for a host that imports it. "
                    "Platform-specific: `abi3` frees it from the Python version, not from "
                    "the machine, so it is keyed by platform: the build machine's, and "
                    "win-amd64 and macosx-universal2, cross-built. macOS is keyed without "
                    "a version and the one file carries arm64 and x86_64.",
        }

    if CROSS_BUILT.is_file():
        exe = f"{RUNTIME_DIR}/board/{BOARD_BINARY}"
        files[exe] = CROSS_BUILT
        blocks["board"] = {
            "model": "rknn", "file": exe, "platform": "aarch64-unknown-linux-gnu",
            "commit": commit,
            "note": "The robot's whole program: this crate, cross-compiled. From the "
                    f"bundle's directory: `./{exe} --bundle . --machine <toml>`.",
        }
    else:
        notes.append(
            f"no board runtime: nothing has cross-built {BOARD_BINARY}. "
            "bash deploy/fsm/docker-build.sh -- then build this bundle again."
        )
    return files, blocks, notes


def assemble(out: Path, modes: dict[str, Path], fsm_path: Path | None, meta: dict,
             extension: Path | None, fsm_text: str | None = None,
             provenance: dict | None = None) -> dict:
    """One FSM design, every host, one directory.

    It was three, one per host, that had to agree about the state machine, the
    contracts and the reference vectors -- and were hashed against each other
    to prove it. Now there is one copy of each, and what differs by host is
    only which model format it loads and which build of the controller it is:
    `bundle.json`'s `runtimes` says, and each host takes its own.
    """
    if not modes:
        raise SystemExit("[deploy] an app needs at least one mode")

    if fsm_text is not None:
        pass
    elif fsm_path is not None:
        fsm_text = fsm_path.read_text("utf-8")
    elif len(modes) == 1:
        fsm_text = synthesise_fsm(next(iter(modes)))
    else:
        raise SystemExit(
            f"[deploy] {len(modes)} modes need an FSM to switch between them.\n"
            f"           --fsm <file.toml>. One mode is synthesised; more than one is a "
            f"decision about priority order, and there is no sensible default for it."
        )

    declared = model_states(fsm_text)
    extra = sorted(set(modes) - set(declared))
    missing = sorted(set(declared) - set(modes))
    if extra or missing:
        raise SystemExit(
            "[deploy] the FSM and the modes given do not match:\n"
            + "".join(f"  --mode {m} is not a [[fsm.state]] with a model\n" for m in extra)
            + "".join(f"  state '{m}' names a model but no --mode {m}=<dir> was given\n"
                      for m in missing)
        )

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / BUNDLE_CONFIG).write_text(fsm_text, "utf-8")

    files, blocks, notes = runtimes(meta, extension)
    for name, source in files.items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, out / name)
        if name.endswith(f"/{BOARD_BINARY}"):
            (out / name).chmod(0o755)

    # One copy of a model per export directory. `claw_left` and `claw_right`
    # are one export carried on two sides, and shipped them twice.
    stored: dict[Path, dict[str, str]] = {}
    listed: dict[str, dict] = {}
    unconverted: list[str] = []
    for mode, source in sorted(modes.items()):
        onnx, contract_path = source / "actor.onnx", source / "layout.json"
        for path in (onnx, contract_path):
            if not path.is_file():
                raise SystemExit(
                    f"[deploy] {source} is not an export directory: no {path.name}.\n"
                    f"           python scripts/export.py --task <id> --checkpoint <ckpt>"
                )
        key = source.resolve()
        if key not in stored:
            (out / MODELS_DIR).mkdir(exist_ok=True)
            entry = {"onnx": f"{MODELS_DIR}/{mode}.onnx"}
            shutil.copy2(onnx, out / entry["onnx"])
            # Where `onnx2rknn.py --bundle <dir>` leaves it. Looked for rather
            # than converted here: the toolkit pins numpy, torch and onnx to
            # versions that would downgrade the training environment, which is
            # why it has a virtualenv of its own and why this does not import it.
            rknn = source / "actor.rknn"
            if rknn.is_file():
                entry["rknn"] = f"{MODELS_DIR}/{mode}.rknn"
                shutil.copy2(rknn, out / entry["rknn"])
            stored[key] = entry
        models = stored[key]
        if "rknn" not in models:
            unconverted.append(mode)
        shutil.copy2(contract_path, out / f"{mode}.json")
        contract = json.loads(contract_path.read_text("utf-8"))

        # A reference-guided policy is a residual on a recording, so the
        # recording is as required as the model. The contract names the file and
        # the controller looks for it beside the contract, so it keeps its name
        # -- which means two modes carrying different recordings under one name
        # would overwrite each other silently, and one of the two would be
        # danced by a policy trained on the other.
        spec = contract.get("reference")
        if spec:
            src = source / spec["file"]
            if not src.is_file():
                raise SystemExit(
                    f"[deploy] {mode}: its contract names the reference trajectory "
                    f"{spec['file']}, which is not in {source}.\n"
                    f"           The policy cannot run without it. Re-export the task."
                )
            dst = out / spec["file"]
            if dst.is_file() and dst.read_bytes() != src.read_bytes():
                raise SystemExit(
                    f"[deploy] two modes ship different recordings both named "
                    f"{spec['file']}; {mode} is the second.\n"
                    f"           One would overwrite the other and the robot would "
                    f"run a motion it was not trained on."
                )
            shutil.copy2(src, dst)

        if "controller" not in contract:
            notes.append(
                f"{mode}: its contract has no `controller` block, so a consumer falls "
                f"back to its own key bindings. Re-export to carry the task's controls.yaml."
            )
        listed[mode] = {"models": dict(models), "contract": f"{mode}.json",
                        "observation": contract.get("observation", {}).get("dim"),
                        "action": contract.get("action", {}).get("dim")}
        if spec:
            listed[mode]["trajectory"] = spec["file"]

    if unconverted:
        notes.append(
            "not ready for the board: " + ", ".join(unconverted) + " carry no .rknn, and "
            "the board runs them on a stub. "
            + "; ".join(
                f"deploy/convert/.venv/bin/python deploy/convert/onnx2rknn.py "
                f"--bundle {modes[m]}" for m in unconverted
            )
            + " -- then build this bundle again."
        )

    bundle = {
        "schema": BUNDLE_SCHEMA,
        "runtimes": blocks,
        "fsm": BUNDLE_CONFIG,
        "modes": listed,
        "notes": notes,
        "files": {},
    }
    # A draft first: the reference is generated from a readable manifest.
    (out / "bundle.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", "utf-8")
    bundle["notes"] += write_reference(out, extension, modes)
    bundle["reference"] = "reference.json" if (out / "reference.json").is_file() else None
    # Every bundle carries its manual, from the controller it carries.
    fsm = (_load_for_reference(out, bundle, extension, need_limits=False)
           if extension is not None else "no native controller was built to ask")
    if isinstance(fsm, str):
        # A directory that exists is a directory somebody will try.
        shutil.rmtree(out, ignore_errors=True)
        raise SystemExit(f"[deploy] cannot write the bundle's manual: {fsm}\n"
                         f"           {out} was removed")
    bundle["manuals"] = {"en": write_manual(out, fsm)}
    # Where every mode came from: the task, and the checkpoint or export
    # directory. A bundle used to record its contents and not its ancestry,
    # so "rebuild what we shipped" meant somebody remembering.
    if provenance:
        bundle["built_from"] = provenance
    write_readme(out, bundle, meta)
    # Not `bundle.json` itself: it exists on disk by now, and hashing it here
    # would record the hash of the draft, a number that never matches the file
    # it names.
    bundle["files"] = {
        p.relative_to(out).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "bundle.json"
    }
    (out / "bundle.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return bundle


def write_readme(out: Path, bundle: dict, meta: dict) -> None:
    """The guide that travels inside the bundle, from `BUNDLE_README.md`.

    A template beside the crate rather than prose here: it documents `WebFsm`
    and the board's command line, so it belongs next to the code it describes
    and changes when that does.
    """
    def col(path: str) -> str:
        """A path in the listing's first column, and always a space after it."""
        return f"{path} ".ljust(34)

    # A file two modes share -- one export's model, one recording -- is listed
    # once, under the first of them.
    def earlier(mode: str) -> list[dict]:
        return [e for m, e in sorted(bundle["modes"].items()) if m < mode]
    mode_files = "".join(
        f"{col(entry['contract'])}{mode}'s contract: observation layout, joint order, gains\n"
        + "".join(f"{col(path)}{mode}'s policy, {fmt}\n" for fmt, path in entry["models"].items()
                  if not any(path == e["models"].get(fmt) for e in earlier(mode)))
        + (f"{col(entry['trajectory'])}the recording {mode} is driven by\n"
           if entry.get("trajectory") and not any(
               entry["trajectory"] == e.get("trajectory") for e in earlier(mode)) else "")
        for mode, entry in sorted(bundle["modes"].items())
    )
    runtime_files = "".join(
        {
            "web": lambda r: f"{col(r['wasm'])}the controller for a browser, and its "
                             f"binding\n{col(r['glue'])}  (wasm-bindgen "
                             f"{meta['build'].get('wasmBindgen', 'n/a').split()[-1]})\n",
            "mjlab": lambda r: "".join(f"{col(e['file'])}the controller for `play --app` "
                                       f"on {platform}\n"
                                       for platform, e in r["extensions"].items()),
            "board": lambda r: f"{col(r['file'])}the robot's whole program, for "
                               f"{r['platform']}\n",
        }[host](bundle["runtimes"][host])
        for host in ("web", "mjlab", "board") if host in bundle["runtimes"]
    )
    text = README_TEMPLATE.read_text("utf-8")
    (out / "README.md").write_text(
        text.replace("{BUNDLE}", out.name)
        .replace("{COMMIT}", meta["source"]["commit"])
        .replace("{RUNTIME_FILES}", runtime_files)
        .replace("{MODE_FILES}", mode_files)
        .replace("{{", "{").replace("}}", "}"),
        "utf-8",
    )


#: A bundle's manual: every key and button it reads, what each does in each mode
#: and how each mode is reached, in English, generated here on every build. A
#: translation travels beside it as `manual.<language>.json`, added afterwards
#: with `--translate` -- the `bundle-manual` skill writes one in Chinese.
MANUAL_SCHEMA = "kk-bundle-manual/1"
MANUAL = "manual.{}.json"

#: The fields a translation may change. Everything else in a manual -- a mode's
#: name, a control, a key -- is an identifier, and a translation that changed
#: one would be describing another bundle.
MANUAL_TEXT = ("source", "keyboard", "summary", "enter", "leave", "pad", "does")

#: A command axis pushed positive, and negative. The contracts' signs: `pitch`
#: positive is nose down, `roll` positive is the right side down, `twist` and
#: `ang_vel_z` positive are to the left.
_AXIS_WORDS = {
    "lin_vel_x": ("walk forward", "walk back"),
    "lin_vel_y": ("step left", "step right"),
    "ang_vel_z": ("turn left", "turn right"),
    "twist": ("twist left", "twist right"),
    "pitch": ("nose down", "nose up"),
    "roll": ("roll, right side down", "roll, left side down"),
    "height": ("stand taller", "crouch lower"),
}

#: A modifier as printed on the key.
_MODIFIER_LABELS = {"ctrl": "Ctrl", "shift": "Shift", "alt": "Alt"}

_PAD_NAMES = {
    "Lx": "left stick, left and right", "Ly": "left stick, up and down",
    "Rx": "right stick, left and right", "Ry": "right stick, up and down",
    "LT": "left trigger (LT)", "RT": "right trigger (RT)",
    "LB": "left bumper (LB)", "RB": "right bumper (RB)",
    "L3": "left stick click (L3)", "R3": "right stick click (R3)",
    "menu": "Menu", "home": "Home",
    "dpad_up": "D-pad up", "dpad_down": "D-pad down",
    "dpad_left": "D-pad left", "dpad_right": "D-pad right",
}


#: A click gesture as the manual says it, of a control or a key: the gesture is
#: the difference between two switches on one button, so it is in the words.
_CLICK_WORDS = {
    "single": ("click {} once", "click it once again"),
    "double": ("double-click {}", "double-click again"),
    "triple": ("triple-click {}", "triple-click again"),
    "quadruple": ("click {} four times", "click four times again"),
    "quintuple": ("click {} five times", "click five times again"),
}


def _key_label(code: str) -> str:
    """A browser key code as printed on the key: `KeyW` is W, both Ctrls are Ctrl."""
    named = {"Space": "Space", "ControlLeft": "Ctrl", "ControlRight": "Ctrl",
             "ShiftLeft": "Shift", "ShiftRight": "Shift", "AltLeft": "Alt", "AltRight": "Alt",
             "ArrowUp": "↑", "ArrowDown": "↓", "ArrowLeft": "←", "ArrowRight": "→",
             "NumpadEnter": "Num Enter", "Semicolon": ";", "Escape": "Esc"}
    if code in named:
        return named[code]
    for prefix, shown in (("Key", ""), ("Digit", ""), ("Numpad", "Num ")):
        if code.startswith(prefix):
            return shown + code[len(prefix):]
    return code


def _axis_word(axis: str, positive: bool) -> str:
    words = _AXIS_WORDS.get(axis)
    return words[0 if positive else 1] if words else f"{axis} {'+' if positive else '-'}"


def _stroke_label(entry: dict) -> str:
    """A keystroke from the guide as printed on the keys: `Shift + J`, `Ctrl`, `1`."""
    key = entry["key"]
    shown = (_MODIFIER_LABELS.get(key) or
             "/".join(dict.fromkeys(_key_label(c) for c in entry["codes"]["key"])) or key)
    return f"{_MODIFIER_LABELS[entry['modifier']]} + {shown}" if entry["modifier"] else shown


def write_manual(out: Path, fsm: object) -> str:
    """Write `manual.en.json` from the controller's own account of its controls.

    Not from the files a person wrote the bindings in: the controller is what
    reads the pad and the keys, and `pad_guide()` is its answer -- each task's
    controls.yaml as its contract carries it, the bundle's mode switches as
    controller.toml composes them from the manifest. A manual written from those
    files by a second reader would be the copy that drifts.

    The pad and the keyboard are two lists in each mode and two kinds of switch
    (since 2026-09-29): a key is what it does, never "the pad's A".

    Returns the file name. A controller that cannot answer refuses the bundle:
    every bundle carries its manual.
    """
    import tomllib

    if not hasattr(fsm, "pad_guide"):
        raise SystemExit("[deploy] this controller build has no pad_guide(), so the bundle "
                         "cannot be given its manual. Rebuild the extension.")
    guide = json.loads(fsm.pad_guide())
    fsm_cfg = tomllib.loads((out / BUNDLE_CONFIG).read_text("utf-8"))["fsm"]
    manifest = json.loads((out / "bundle.json").read_text("utf-8"))
    default = next((r["enter"] for r in fsm_cfg.get("rule", []) if r["when"] == "always"), None)
    tasks = {s["name"]: s.get("task") for s in fsm_cfg.get("state", []) if s.get("model")}

    def switch_does(s: dict) -> str:
        if s["leaves"]:
            does = f"leave {', '.join(s['leaves'])} -- whichever is on -- back to {default}"
        elif s["event"]:
            does = f"trigger {s['binding']}"
        else:
            _, again = _CLICK_WORDS.get(s["on"], (None, None))
            does = f"switch to {s['enters']}" + (
                "; press again to switch back" if s["on"] == "toggle" else
                f"; {again} to switch back" if again else "")
        if s["from"]:
            does += f" (from {', '.join(s['from'])} only)"
        return does

    # The switches, each device's apart.
    switches, reached, left_by = [], {}, {}
    for c in guide["controls"]:
        for s in c["switches"]:
            pad = _PAD_NAMES.get(c["control"], c["control"])
            click, _ = _CLICK_WORDS.get(s["on"], (None, None))
            if click:
                pad = click.format(pad)
            elif s["on"] == "fall":
                pad = f"{pad}, pressed and let go on its own"
            if s["with"]:
                pad = f"hold {_PAD_NAMES.get(s['with'], s['with'])}, then {pad}"
            switches.append({"device": "pad", "control": c["control"], "with": s["with"],
                             "keys": [], "pad": pad, "does": switch_does(s), "from": s["from"]})
            if s["enters"] and not s["event"]:
                reached.setdefault(s["enters"], []).append(pad)
            for m in s["leaves"]:
                left_by.setdefault(m, []).append(pad)
    for k in guide["keyboard"]:
        cap = _stroke_label(k)
        for s in k["switches"]:
            click, _ = _CLICK_WORDS.get(s["on"], (None, None))
            shown = click.format(cap) if click else (
                f"{cap}, pressed and let go on its own" if s["on"] == "fall" else cap)
            switches.append({"device": "keyboard", "control": k["stroke"], "with": s["with"],
                             "keys": [cap], "pad": None, "does": switch_does(s),
                             "from": s["from"]})
            if s["enters"] and not s["event"]:
                reached.setdefault(s["enters"], []).append(f"keyboard: {shown}")
            for m in s["leaves"]:
                left_by.setdefault(m, []).append(f"keyboard: {shown}")

    # The default first, as a person meets them; a stick's up and down before
    # its left and right, and its click after both.
    order = ["Ly", "Lx", "L3", "Ry", "Rx", "R3"]
    rank = {c["control"]: order.index(c["control"]) if c["control"] in order else len(order) + i
            for i, c in enumerate(guide["controls"])}
    # The stick's two ends as a person pushes them. evdev's `Ly` and `Ry` are
    # positive towards the operator, so up is their negative end.
    ends = {"Ly": ("up", "down"), "Ry": ("up", "down"), "Lx": ("left", "right"),
            "Rx": ("left", "right")}
    modes = []
    for mode in sorted(guide["modes"], key=lambda m: (m != default, guide["modes"].index(m))):
        contract = json.loads((out / manifest["modes"][mode]["contract"]).read_text("utf-8"))
        reference = contract.get("reference") or {}
        controller = contract.get("controller") or {}
        full = controller.get("devices", {}).get("keyboard", {}).get("full_after_s")
        # A held shift's layer is in "with R3 held"; a toggled one "after R3".
        shift = controller.get("devices", {}).get("gamepad", {}).get("shift") or {}
        held = shift.get("gesture") == "hold"
        layer = _PAD_NAMES.get(shift.get("button", ""), shift.get("button", ""))

        def axis_words(axis: str, positive: bool, moved: bool) -> str:
            words = _axis_word(axis, positive)
            return f"{words} (moves it; let go, it stays)" if moved else words

        rows = []
        for c in sorted(guide["controls"], key=lambda c: rank[c["control"]]):
            does = c["does"].get(mode)
            if not does:
                continue
            words = []
            axes = [d for d in does if d["kind"] == "axis"]
            for shifted in (False, True):
                these = sorted((d for d in axes if d["shifted"] is shifted), key=lambda d: d["travel"][0])
                if not these:
                    continue
                if c["control"] in ends:
                    neg, pos = ends[c["control"]]
                    said = "; ".join(
                        f"{end}: " + ", then ".join(axis_words(d["axis"], d["sign"] * push > 0, d["moved"])
                                                   for d in these)
                        for end, push in ((neg, -1), (pos, 1)))
                else:
                    said = ", then ".join(axis_words(d["axis"], d["sign"] > 0, d["moved"]) for d in these)
                words.append(f"with {layer} held -- {said}" if shifted and held else
                             f"after {layer} -- {said}" if shifted else said)
            for d in does:
                if d["kind"] == "task":
                    words.append(d["what"])
                elif d["kind"] == "release":
                    words.append("let go of everything: every control back to rest")
                elif d["kind"] == "shift":
                    words.append("hold for the other stick layer" if held
                                 else "toggle the other stick layer")
                elif d["kind"] == "reset":
                    words.append("tapped -- pressed and let go without the stick moving -- "
                                 "puts the moved axes back at rest")
            rows.append({"control": c["control"], "pad": _PAD_NAMES.get(c["control"], c["control"]),
                         "does": "; ".join(words)})

        keys = []
        for k in guide["keyboard"]:
            does = k["does"].get(mode)
            if not does:
                continue
            words = []
            for d in does:
                if d["kind"] == "axis":
                    # A key places what it drives, a moved axis included.
                    words.append(axis_words(d["axis"], d["dir"] == "+", False))
                elif d["kind"] == "task":
                    words.append(d["what"])
                elif d["kind"] == "release":
                    words.append("let go of everything: every control back to rest")
            keys.append({"key": _stroke_label(k), "does": "; ".join(words)})

        if mode == default:
            summary = "The mode the robot starts in, and the one it returns to when no other is asked for."
        elif reference.get("starts_on_entry"):
            summary = (f"A recorded motion, {reference.get('duration', 0):.1f} s long, that "
                       f"starts the moment this mode is entered.")
        else:
            summary = "A mode of its own, with its own controls."
        interrupt = (f" {' or '.join(left_by[mode])} interrupts it and goes back to {default}."
                     if mode in left_by else "")
        leave = (f"Hands back to {default} by itself when the motion ends; the switch pressed "
                 f"again stops it early.{interrupt}" if reference.get("starts_on_entry") else
                 f"Press its switch again to go back to {default}.{interrupt}" if mode != default else
                 "Leaving it is switching to another mode.")
        moving = [a["name"] for c in controller.get("command", []) for a in c.get("axes", [])
                  if a.get("integrate_s")]
        keyboard = ("This mode reads no sticks, triggers or keys." if not full else
                    f"A key held pushes what it drives further the longer it is held -- fully "
                    f"after {full:g} s -- and lets it back the moment it comes up"
                    + (f", the {' and the '.join(moving)} included: the keys place what the "
                       f"stick moves." if moving else "."))
        modes.append({
            "mode": mode, "task": tasks.get(mode), "default": mode == default,
            "summary": summary,
            "enter": ("Where the robot starts." if mode == default and mode not in reached else
                      " or ".join(reached.get(mode, [])) or "No switch reaches it."),
            "leave": leave,
            "keyboard": keyboard,
            "controls": rows,
            "keys": keys,
        })

    doc = {
        "schema": MANUAL_SCHEMA,
        "language": "en",
        "bundle": out.name,
        "source": "Generated by scripts/deploy.py when this bundle was built: each task's "
                  "controls.yaml, as its contract carries it, and the bundle's mode switches "
                  "from deploy/manifests.json, as the controller itself reports them. The pad "
                  "and the keyboard are two paths, each listed on its own.",
        "modes": modes,
        "switches": switches,
    }
    name = MANUAL.format("en")
    (out / name).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return name


def _manual_shape(doc, path: str = "") -> list[str]:
    """Every identifier in a manual, with where it sits: what a translation must keep."""
    if isinstance(doc, dict):
        return [line for k, v in sorted(doc.items()) if k not in MANUAL_TEXT and k != "language"
                for line in _manual_shape(v, f"{path}.{k}")] + [
            f"{path}.{k}:{'text' if isinstance(v, str) else type(v).__name__}"
            for k, v in sorted(doc.items()) if k in MANUAL_TEXT]
    if isinstance(doc, list):
        return [f"{path}[]:{len(doc)}"] + [line for i, v in enumerate(doc)
                                           for line in _manual_shape(v, f"{path}[{i}]")]
    return [f"{path}={json.dumps(doc, ensure_ascii=False)}"]


def translate(bundle: Path, language: str, source: Path) -> int:
    """Add `manual.<language>.json` to a built bundle, and pack it again.

    Checked against the English one before it goes in: the same modes, the
    same controls, the same keys, in the same places -- only the text fields
    (`MANUAL_TEXT`) may differ, and each must still be text. A translation
    that moved a key or dropped a mode would be a manual for another bundle,
    in a language nobody on the English side reads.
    """
    if not (bundle / "bundle.json").is_file():
        raise SystemExit(f"[deploy] {bundle} is not a bundle: no bundle.json")
    manifest = json.loads((bundle / "bundle.json").read_text("utf-8"))
    english = json.loads((bundle / MANUAL.format("en")).read_text("utf-8"))
    try:
        doc = json.loads(source.read_text("utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"[deploy] {source} is not JSON: {e}")
    if doc.get("language") != language:
        raise SystemExit(f"[deploy] {source} says language {doc.get('language')!r}, not {language!r}")
    want, got = _manual_shape(english), _manual_shape(doc)
    if want != got:
        differ = next((f"{a}  !=  {b}" for a, b in zip(want, got) if a != b),
                      f"{len(want)} identifiers against {len(got)}")
        raise SystemExit(f"[deploy] {source} is not a translation of this bundle's manual: "
                         f"it changes more than the text.\n           first difference: {differ}")
    name = MANUAL.format(language)
    (bundle / name).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", "utf-8")
    manifest.setdefault("manuals", {})[language] = name
    manifest["files"][name] = {"bytes": (bundle / name).stat().st_size,
                               "sha256": sha256(bundle / name)}
    manifest["files"] = dict(sorted(manifest["files"].items()))
    (bundle / "bundle.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", "utf-8")
    validate_app(bundle)
    archive = zip_bundle(bundle)
    print(f"[deploy] {bundle.name}: added {name}, packed again -> {archive.name}")
    return 0


#: What a bundle is: every file it may hold, in what format, and `bundle.json`'s
#: shape. Beside the manifests rather than in this script, because it is what
#: a consumer reads too; every build is held to it before it is packed.
APP_SCHEMA = CRATE.parent / "app.schema"


def validate_app(out: Path) -> None:
    """Hold a bundle to `deploy/app.schema`, or refuse it.

    `bundle.json` against the schema, each manual against its `$defs/manual`,
    and the directory against `x-files`: every file is one the schema knows,
    every required one is there, and `bundle.json` lists exactly what is on
    disk. A bundle that passes here is the bundle the schema describes -- which
    is the only way a schema stays true, rather than being a description of
    what the bundler did once.
    """
    try:
        import jsonschema
    except ModuleNotFoundError:
        raise SystemExit("[deploy] jsonschema is not installed, and every bundle is checked "
                         "against deploy/app.schema.\n           pip install -e .")
    schema = json.loads(APP_SCHEMA.read_text("utf-8"))
    validator = jsonschema.Draft202012Validator
    manifest = json.loads((out / "bundle.json").read_text("utf-8"))
    problems = [f"bundle.json: {'.'.join(map(str, e.absolute_path)) or '(top)'}: {e.message}"
                for e in validator(schema).iter_errors(manifest)]
    manual = validator({"$schema": schema["$schema"], "$defs": schema["$defs"],
                        "$ref": "#/$defs/manual"})
    for language, name in manifest.get("manuals", {}).items():
        if not (out / name).is_file():
            continue  # said below, with every other file that is missing
        doc = json.loads((out / name).read_text("utf-8"))
        problems += [f"{name}: {'.'.join(map(str, e.absolute_path)) or '(top)'}: {e.message}"
                     for e in manual.iter_errors(doc)]
        if doc.get("language") != language:
            problems.append(f"{name}: says language {doc.get('language')!r}, listed as {language!r}")

    rules = [(re.compile(f["path"]), f) for f in schema["x-files"]]
    present = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    matched = {id(f): 0 for _, f in rules}
    for name in present:
        rule = next((f for rx, f in rules if rx.search(name)), None)
        if rule is None:
            problems.append(f"{name}: no file of this name belongs in a bundle (deploy/app.schema)")
        else:
            matched[id(rule)] += 1
    problems += [f"missing: a file matching {f['path']} -- {f['is']}"
                 for _, f in rules if f["required"] and not matched[id(f)]]
    listed = set(manifest.get("files", {}))
    problems += [f"{name}: on disk and not in bundle.json's files"
                 for name in present if name != "bundle.json" and name not in listed]
    problems += [f"{name}: in bundle.json's files and not on disk"
                 for name in sorted(listed - set(present))]
    if problems:
        raise SystemExit("[deploy] this bundle is not what deploy/app.schema describes:\n"
                         + "".join(f"           {p}\n" for p in problems))


#: The earliest moment a zip entry can record. Every entry gets it.
#:
#: An archive is the one thing here a person carries somewhere else, and the
#: two builds it comes from are meant to be comparable: `bundle.json` already
#: records a digest for every file, and a zip whose bytes changed on every
#: build -- because a mtime moved -- would be the only artefact that could not
#: be held against the one shipped last week.
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

#: What a bundle's archive is called: `jumper/` travels as `jumper.app`.
#:
#: The bytes are an ordinary zip, and any zip reader opens one. The name says
#: what the file is rather than how it is packed, as `.jar` does for a zip of
#: classes: a bundle is an application -- the controller builds and everything
#: they load.
#:
#: A consumer that picks files by name has to list it: a browser's file picker
#: whose `accept` names only `.zip` does not offer the file at all, and nothing
#: says why.
ARCHIVE_SUFFIX = ".app"


def zip_bundle(out: Path) -> Path:
    """The bundle's directory as a sibling `.app`, a zip with the contents at its root.

    For a consumer that is not a filesystem: a browser's file picker, an upload
    form, a phone, a copy to a robot. `bundle.json` has to be at the top of the
    archive because that is where a consumer looks for it -- a zip of the
    *directory* would put everything one level down and read as a bundle with
    no manifest.

    The directory stays. Everything on this machine reads it directly --
    `--check-reference`, the tests, this script's own verification -- and a zip
    that replaced it would make all of them unpack first.
    """
    archive = out.with_suffix(ARCHIVE_SUFFIX)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bag:
        # Sorted, so two builds of one bundle lay their entries down in one
        # order. `rglob` does not promise one.
        for path in sorted(p for p in out.rglob("*") if p.is_file()):
            entry = zipfile.ZipInfo(path.relative_to(out).as_posix(), date_time=ZIP_EPOCH)
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = (path.stat().st_mode & 0o7777) << 16
            bag.writestr(entry, path.read_bytes())
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="deploy.py",
        description=(__doc__ or "").splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="A plain parser, not the one train/play/export share: this reads no "
               "environment, so --task and --backend would be arguments with nothing "
               "to do.",
        # No prefixes. `--check` was a flag of its own until 2026-09-28, and with
        # abbreviations on it would now be `--check-reference` -- an old command
        # line doing a different thing rather than failing.
        allow_abbrev=False,
    )
    parser.add_argument(
        "--bundle", type=Path, nargs="?", default=None, metavar="DIR",
        help="where to write the app: <DIR>/<name>/ and <DIR>/<name>.app beside it. "
             "(default: out/bundle_<timestamp>/)",
    )
    parser.add_argument(
        "--manifest", nargs="?", const="", metavar="NAME",
        help=f"which bundle of {MANIFESTS.relative_to(REPO)} to build: the exports that "
             f"go in and the key or button that switches into each. "
             f"(default: the only one the file defines)",
    )
    parser.add_argument(
        "--manifests", type=Path, default=MANIFESTS,
        help="read the manifests from here instead.",
    )
    parser.add_argument(
        "--mode", action="append", default=[], metavar="NAME=DIR",
        help="instead of a manifest: a mode and the export directory its policy came "
             "from (actor.onnx + layout.json). Repeat for each mode.",
    )
    parser.add_argument(
        "--fsm", type=Path,
        help="with --mode: the FSM config to ship. Required for more than one mode; a "
             "single mode gets a synthesised one-state config.",
    )
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="skip the cross build and the .rknn conversions, take whatever is already "
             "there, and build the app without what is missing, saying so in its notes. "
             "For a browser, a bench or the tests; an app that looks complete is the "
             "one somebody copies to a robot.",
    )
    parser.add_argument(
        "--require-clean", action="store_true",
        help="refuse to build from an uncommitted controller source (SOURCES: the "
             "crate, every task's deploy hook, the pad dictionary, the IDL). Off by default: an "
             "uncommitted build is recorded as '-dirty' in bundle.json, which is "
             "enough for everything except an app that is going somewhere.",
    )
    parser.add_argument(
        "--translate", type=Path, metavar="BUNDLE",
        help="add a translation of a built bundle's manual.en.json to it, as "
             "manual.<--language>.json, and pack it again. Checked against the "
             "English: only the text may differ.",
    )
    parser.add_argument("--language", help="the translation's language, with --translate: zh")
    parser.add_argument("--manual", type=Path, help="the translated manual, with --translate")
    parser.add_argument(
        "--check-reference", type=Path, metavar="BUNDLE",
        help="replay a bundle's reference.json through the controller it carries "
             "and report where this host differs. Writes nothing.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show cargo output")
    args = parser.parse_args()

    if not CRATE.is_dir():
        raise SystemExit(f"[deploy] no crate at {CRATE}")

    if args.check_reference:
        return check_reference(args.check_reference.resolve())
    if args.translate:
        if not (args.language and args.manual):
            parser.error("--translate needs --language and --manual")
        return translate(args.translate.resolve(), args.language, args.manual.resolve())

    # Which modes, before anything is compiled: every refusal from here to the
    # build is cheaper than cargo.
    fsm_text, provenance, spec, name = None, None, None, "bundle"
    if args.mode:
        if args.manifest is not None:
            parser.error("--manifest and --mode are two answers to one question, which "
                         "policies go in; give one")
        modes = parse_modes(args.mode)
    else:
        if args.fsm:
            parser.error("--fsm goes with --mode; a manifest names its own FSM")
        name, spec = resolve_manifest(args.manifest or None, args.manifests)
        modes, provenance = manifest_modes(spec, args.manifests)
        print(f"[deploy] manifest {name}: "
              + ", ".join(f"{m} <- {d['task']}" for m, d in sorted(provenance.items())))

    # Absolute from here on. `verify_bundle` runs cargo with the crate as its
    # working directory, so a relative `--bundle out/upload` -- the obvious way
    # to type it -- resolved against the crate and the check reported the
    # bundle it had just written as missing, then deleted it. Resolved once,
    # rather than at each use, so there is no second place to forget.
    #
    # The default is timestamped the way `logs/<model>/<task>/<run>` is, so
    # there is one way to read a directory name in this repository.
    root = (args.bundle or REPO / f"out/bundle_{datetime.now():%Y-%m-%d_%H-%M-%S}").resolve()

    cargo = require("cargo", "install Rust: https://rustup.rs")
    commit = source_commit(args.require_clean)
    bindgen_path, cli = check_tooling()
    if not args.allow_incomplete:
        problems = preflight(modes)
        if problems:
            raise SystemExit(
                "[deploy] the board's half of this app cannot be made here:\n"
                + "".join(f"           {p}\n" for p in problems)
                + "           --allow-incomplete builds it without, for a browser or a bench."
            )
        prepare_board(modes)
    build(cargo, args.verbose)
    bindgen(bindgen_path, STAGING)
    meta = build_record(commit, cli)
    # Built even where no host would ship it: generating the reference means
    # *running* this controller, and a different build of the same source
    # would produce a reference that measures nothing.
    extension = compile_extension(cargo, args.verbose)

    if spec is not None:
        # After the modes are known, because the model filenames are what
        # the composed states name.
        fsm_text = compose_fsm(spec, provenance,
                               {m: f"{m}.onnx" for m in modes}, args.manifests)
    out = root / name
    bundle = assemble(out, modes, args.fsm, meta, extension, fsm_text, provenance)
    lacking = [] if args.allow_incomplete else gaps(bundle)
    if lacking:
        # The steps above ran and said nothing, and the app still lacks a part:
        # a reference that failed to generate, a binary that did not land. The
        # notes are the build's own account of why.
        shutil.rmtree(out, ignore_errors=True)
        raise SystemExit(
            f"[deploy] the app is not complete: {'; '.join(lacking)}.\n"
            + "".join(f"           note: {note}\n" for note in bundle["notes"])
            + f"           {out} was removed."
        )
    # After writing, because the check reads the bundle rather than the
    # arguments: what is verified is what a consumer receives. Then packed,
    # so nothing is packed that this script would not hand over unpacked.
    report = verify_bundle(cargo, out, False)
    validate_app(out)
    archive = zip_bundle(out)
    size = sum(e["bytes"] for e in bundle["files"].values())
    print(f"[deploy] bundled {short(commit)} -> {out}")
    print(f"           {size / 1e6:.2f} MB  {len(bundle['files'])} files  -> "
          f"{archive.name} {archive.stat().st_size / 1e6:.2f} MB")
    for host in HOSTS:
        runtime = bundle["runtimes"].get(host)
        print(f"           {host:<6} " + (
            "no runtime" if runtime is None else
            f"{runtime['model']} models, " + (
                runtime.get("wasm") or runtime.get("file")
                or ", ".join(e["file"] for e in runtime["extensions"].values()))))
    for line in report:
        print(f"             {line}")
    for note in bundle["notes"]:
        print(f"             note: {note}")
    if args.allow_incomplete and "board" in bundle["runtimes"]:
        # Taken as found, and filed under this source's commit anyway.
        print(f"             note: runtime/board/controller is {CROSS_BUILT.name} as found, "
              f"not rebuilt; it may predate this source")
    for platform, (_, built, _) in PLAY_CROSS_BUILDS.items():
        if args.allow_incomplete and platform in (
                bundle["runtimes"].get("mjlab") or {}).get("extensions", {}):
            print(f"             note: {extension_file(platform)} is {built.name} as found, "
                  "not rebuilt; it may predate this source")
    # The one step left to an agent: the build cannot write natural Chinese.
    shown = out.relative_to(REPO) if out.is_relative_to(REPO) else out
    print("[deploy] next: the Chinese manual. The bundle-manual skill writes it and adds it with")
    print(f"           python scripts/deploy.py --translate {shown} --language zh --manual <file>")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
