"""The upload bundle: what a consumer receives, and whether it survives the trip.

A bundle carries a compiled controller, an FSM config and one policy per mode to
somebody who has none of this repository. Three failures are worth pinning and
all three produce a directory rather than an error:

- **an FSM nobody can load.** The bundler writes `controller.toml` and cannot itself
  tell whether `FsmConfig::parse` accepts it -- Python cannot run it, and
  re-implementing it here would be the second implementation this crate exists
  to avoid. It asks the crate instead, and a bundle that fails is removed rather
  than left for someone to find and upload.
- **modes and config that disagree.** A `[[fsm.state]]` naming a model with no
  policy beside it loads, runs, and holds forever the first time the cascade
  enters it.
- **a README that has stopped matching the thing it describes.** It is the only
  part a consumer reads before writing code against the rest.

These build a real controller, which needs a Rust toolchain and the wasm target.
Skipped where there is none, because this repository's own suite must run on a
machine that only trains.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXPORT = REPO / "tasks/jumper/tripod/out/rough"
SCRIPT = REPO / "scripts/deploy.py"
#: Where `docker-build.sh` leaves the board's controller; see `deploy.py`.
CROSS_BUILT = REPO / "out/deploy/controller-aarch64"


def _toolchain() -> str | None:
    """`cargo` plus the wasm target, or a reason to skip."""
    if not EXPORT.is_dir():
        return "no exported policy to bundle"
    if not shutil.which("cargo"):
        return "cargo is not installed"
    targets = subprocess.run(
        ["rustup", "target", "list", "--installed"], capture_output=True, text=True, check=False
    )
    if "wasm32-unknown-unknown" not in targets.stdout:
        return "the wasm32-unknown-unknown target is not installed"
    if not shutil.which("wasm-bindgen") and not (Path.home() / ".cargo/bin/wasm-bindgen").is_file():
        return "wasm-bindgen is not installed"
    return None


SKIP = _toolchain()
pytestmark = pytest.mark.skipif(SKIP is not None, reason=SKIP or "")

#: Two modes, one reachable only by a key. Written here rather than committed as
#: a fixture because what it pins is that the *bundler* checks it: a fixture that
#: drifted from the parser would make this pass while the check rotted.
TWO_MODES = """
[fsm]
initial_state = "walk"
warm_start_ref = "walk"
safe_state = "safe"
tilt_limit = 3.15
state_timeout_ms = 200
command_timeout_ms = 3600000
mode_switch_ramp_s = 0.0
pose_reach_tol = 0.10
warm_start_duration_s = 0.0
ramp_kp = 0.15
ramp_kd = 0.01

[[fsm.button]]
name = "slow"
key = "keypad_1"
on = "toggle"

[[fsm.state]]
name = "safe"
hold_current = true
kd = 0.01

[[fsm.state]]
name = "walk"
model = "walk.onnx"

[[fsm.state]]
name = "slow"
model = "slow.onnx"

[[fsm.rule]]
when = "feedback_stale"
enter = "safe"

[[fsm.rule]]
when = "in_state:safe"
enter = "@initial"

[[fsm.rule]]
when = "button:slow"
enter = "slow"

[[fsm.rule]]
when = "always"
enter = "walk"
"""


#: Where a build lands: `<dir>/<name>/`, and `<dir>/<name>.app` beside it.
#: `--mode` builds are named `bundle`, a manifest's by the manifest's name.
def built(out: Path, name: str = "bundle") -> Path:
    return out / name


def run(*args: str, complete: bool = False) -> subprocess.CompletedProcess[str]:
    """A build, with `--allow-incomplete` unless `complete` asks for the default.

    No flag for the tree: an uncommitted crate builds, and says so. The bundler
    used to refuse one unless `--allow-dirty` was passed, which is right for a
    published artifact and wrong for everything else: it made this suite red for
    exactly as long as somebody was editing the controller, so "run the tests
    before committing" and "the tests pass" could not both hold. That is not a
    theory -- two commits went in on a red run because the failure looked like
    cargo contention and was this. `--require-clean` is the opt-in now.

    `--allow-incomplete` because the default cross-builds the board's controller
    and converts every policy, in docker, which is minutes and a container this
    suite must not need -- and what these pin is the assembly. Completeness is
    `test_a_complete_app_is_read_off_what_was_written`'s and the two after it.
    """
    builds = not any(a in ("--check-reference", "--translate") for a in args)
    extra = ["--allow-incomplete"] if builds and not complete else []
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, *extra],
        cwd=REPO, capture_output=True, text=True, check=False,
    )


@pytest.fixture(scope="module")
def two_modes(tmp_path_factory) -> Path:
    """One build, one bundle, two modes on one export. Returns the bundle."""
    out = tmp_path_factory.mktemp("bundles") / "two"
    fsm = out.parent / "two.toml"
    export = with_limits(EXPORT, out.parent / "export")
    fsm.write_text(TWO_MODES, "utf-8")
    result = run("--bundle", str(out), "--fsm", str(fsm),
                 f"--mode=walk={export}", f"--mode=slow={export}")
    assert result.returncode == 0, result.stdout + result.stderr
    return built(out)


def files(bundle: Path) -> set[str]:
    return {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}


def test_one_bundle_carries_every_host(two_modes: Path) -> None:
    """One directory, and each host's part of it named in `bundle.json`.

    It was three directories, one per host, that had to be hashed against each
    other to show they agreed about the FSM, the contracts and the reference.
    One directory has one copy of each, and what differs by host -- the model
    format, the build of the controller -- is under `models/` and `runtime/`,
    with `runtimes` saying which host takes which. A host with somebody else's
    runtime beside its own now carries it; what it must never do is *load* it,
    which is `runtimes`' job.
    """
    platform = sysconfig.get_platform()
    present = files(two_modes)
    shared = {"README.md", "bundle.json", "controller.toml", "reference.json",
              "manual.en.json", "walk.json", "slow.json"}
    assert shared <= present
    assert {"runtime/web/controller.js", "runtime/web/controller.wasm",
            f"runtime/mjlab/{platform}/controller.so"} <= present
    # The board's controller, and play's for Windows and macOS, only when
    # something has cross-built them, which a machine running this suite need
    # not have. The `.rknn` likewise.
    windows = "runtime/mjlab/win-amd64/controller.pyd"
    macos = "runtime/mjlab/macosx-universal2/controller.so"
    board = present - shared - {"runtime/web/controller.js", "runtime/web/controller.wasm",
                                f"runtime/mjlab/{platform}/controller.so"}
    assert board - {"runtime/board/controller", windows, macos} == {"models/slow.onnx"}, board
    # One file, for a consumer that is not a filesystem. Beside the directory,
    # never instead of it: everything on this machine reads the directory.
    assert (two_modes.parent / "bundle.app").is_file()

    manifest = json.loads((two_modes / "bundle.json").read_text("utf-8"))
    assert manifest["schema"] == "kk-policy-bundle/1"
    assert "target" not in manifest, "a bundle is no longer packed for one host"
    assert sorted(manifest["modes"]) == ["slow", "walk"]
    web = manifest["runtimes"]["web"]
    assert (web["model"], web["wasm"]) == ("onnx", "runtime/web/controller.wasm")
    # Said out loud, because the consumer is the only one who can say it to the
    # person who uploaded the bundle.
    assert "held against" in web["unverifiedByTheConsumer"]
    # The native build, keyed by the machine it was built on. Without it `play
    # --app` runs whatever `mjrl_fsm` happens to be installed -- a different
    # build of the same source, of any age.
    mjlab = manifest["runtimes"]["mjlab"]
    assert mjlab["model"] == "onnx"
    assert mjlab["extensions"][platform]["file"] == f"runtime/mjlab/{platform}/controller.so"
    # Windows imports an extension by `.pyd` and no other suffix; macOS is keyed
    # without the version every Mac's platform string carries.
    if windows in present:
        assert mjlab["extensions"]["win-amd64"]["file"] == windows
    if macos in present:
        assert mjlab["extensions"]["macosx-universal2"]["file"] == macos


def test_every_bundle_carries_its_manual_and_is_what_the_schema_says(two_modes: Path) -> None:
    """The manual is written by every build, from the controller's own account
    of its pad -- so the key a mode binds is the key the manual names. And the
    build is held to `deploy/app.schema` before it is packed; this holds the
    result to it again, from outside the script."""
    import importlib.util

    manifest = json.loads((two_modes / "bundle.json").read_text("utf-8"))
    assert manifest["manuals"] == {"en": "manual.en.json"}
    manual = json.loads((two_modes / "manual.en.json").read_text("utf-8"))
    assert manual["modes"][0]["mode"] == "walk", "the default comes first"
    # `slow` is on keypad_1 in TWO_MODES: the keyboard's switch, and the manual
    # lists it as the keyboard's.
    assert any(s["device"] == "keyboard" and s["keys"] == ["Num 1"] and "slow" in s["does"]
               for s in manual["switches"])
    walk = next(m for m in manual["modes"] if m["mode"] == "walk")
    # The sign that has no second chance: W, and the stick pushed forward, are
    # forward -- each on its own list, the pad's and the keyboard's.
    keys = {k["key"]: k["does"] for k in walk["keys"]}
    assert (keys["W"], keys["S"]) == ("walk forward", "walk back"), keys
    ly = next(c for c in walk["controls"] if c["control"] == "Ly")
    assert ly["does"] == "up: walk forward; down: walk back", ly

    spec = importlib.util.spec_from_file_location("deploy_for_bundle", SCRIPT)
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    deploy.validate_app(two_modes)


def test_two_modes_on_one_export_carry_one_model(two_modes: Path) -> None:
    """`claw_left` and `claw_right` are one export carried on two sides; the
    three per-host bundles shipped its model twice each.

    The control group is the contracts, which stay one per mode: a mode's
    contract is what its hook and its observation history are keyed by.
    """
    modes = json.loads((two_modes / "bundle.json").read_text("utf-8"))["modes"]
    assert modes["walk"]["models"] == modes["slow"]["models"] == {"onnx": "models/slow.onnx"}
    assert modes["walk"]["contract"] != modes["slow"]["contract"]


def test_every_file_it_names_is_there_and_hashes(two_modes: Path) -> None:
    import hashlib

    manifest = json.loads((two_modes / "bundle.json").read_text("utf-8"))
    assert set(manifest["files"]) == files(two_modes) - {"bundle.json"}
    for name, entry in manifest["files"].items():
        raw = (two_modes / name).read_bytes()
        assert len(raw) == entry["bytes"], name
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], name


def test_the_glue_and_the_wasm_agree_on_each_others_names(two_modes: Path) -> None:
    """wasm-bindgen writes `<name>_bg.wasm` and points the glue at that name.
    Renaming without rewriting the reference leaves two files that disagree --
    invisible to a consumer that passes bytes, a 404 for one that does not."""
    glue = (two_modes / "runtime/web/controller.js").read_text("utf-8")
    assert "controller_bg.wasm" not in glue
    assert glue.count("controller.wasm") == 1


def test_a_config_the_controller_cannot_load_is_not_written(tmp_path) -> None:
    """The control group.

    Every assertion above passes against a bundler that writes files and checks
    nothing -- which is what it did until this existed. A cascade whose last rule
    is not `always` is refused by `FsmConfig::parse`, and that is the only thing
    that can say so.
    """
    fsm = tmp_path / "broken.toml"
    fsm.write_text(TWO_MODES.replace('when = "always"', 'when = "gripper_active"'), "utf-8")
    out = tmp_path / "broken"
    result = run("--bundle", str(out), "--fsm", str(fsm),
                 f"--mode=walk={EXPORT}", f"--mode=slow={EXPORT}")
    assert result.returncode != 0
    assert "not `always`" in result.stdout + result.stderr
    assert not built(out).exists(), "a bundle that exists is a bundle somebody will try"
    assert not (out / "bundle.app").exists()


def test_modes_and_config_have_to_agree(tmp_path) -> None:
    fsm = tmp_path / "two.toml"
    fsm.write_text(TWO_MODES, "utf-8")
    result = run("--bundle", str(tmp_path / "half"), "--fsm", str(fsm),
                 f"--mode=walk={EXPORT}")
    assert result.returncode != 0
    assert "no --mode slow" in result.stdout + result.stderr

    # ...and the other way: a mode the config never declares.
    result = run("--bundle", str(tmp_path / "extra"), "--fsm", str(fsm),
                 f"--mode=walk={EXPORT}", f"--mode=slow={EXPORT}", f"--mode=fly={EXPORT}")
    assert result.returncode != 0
    assert "--mode fly is not a" in result.stdout + result.stderr


def test_more_than_one_mode_needs_a_config(tmp_path) -> None:
    """One mode is synthesised; a priority order is a decision, and there is no
    default for it that would not be somebody's silent choice."""
    result = run("--bundle", str(tmp_path / "two"),
                 f"--mode=a={EXPORT}", f"--mode=b={EXPORT}")
    assert result.returncode != 0
    assert "need an FSM" in result.stdout + result.stderr

    single = tmp_path / "one"
    export = with_limits(EXPORT, tmp_path / "export")
    result = run("--bundle", str(single), f"--mode=walk={export}")
    assert result.returncode == 0, result.stdout + result.stderr
    written = (built(single) / "controller.toml").read_text("utf-8")
    assert 'model = "walk.onnx"' in written
    assert 'when = "always"' in written


#: The locomotion tasks' controls block as `controls.py` writes it (schema 3,
#: 2026-09-29): the walk on the left stick and on W A S D, the turn on the right
#: stick and on J L. The fixture predates controller blocks altogether.
WALK_CONTROLS = {
    "schema": "operator_controller/2",
    "command": [{"term": "twist", "feeds": "velocity_commands", "frame": "body",
                 "axes": [{"name": "lin_vel_x", "index": 0, "unit": "m/s", "positive": "forward"},
                          {"name": "lin_vel_y", "index": 1, "unit": "m/s", "positive": "left"},
                          {"name": "ang_vel_z", "index": 2, "unit": "rad/s",
                           "positive": "counter-clockwise"}]}],
    "task": {},
    "devices": {
        "gamepad": {"scheme": "absolute", "layout": "xbox",
                    "axes": {"lin_vel_x": {"source": "Ly", "sign": -1},
                             "lin_vel_y": {"source": "Lx", "sign": -1},
                             "ang_vel_z": {"source": "Rx", "sign": -1}},
                    "deadzone": "device_reported_rescaled", "release_button": "B"},
        "keyboard": {"scheme": "keys", "full_after_s": 2.0,
                     "axes": {"lin_vel_x": {"+": ["key_w"], "-": ["key_s"]},
                              "lin_vel_y": {"+": ["key_a"], "-": ["key_d"]},
                              "ang_vel_z": {"+": ["key_j"], "-": ["key_l"]}},
                     "release": ["key_b"]},
    },
}


def with_limits(source: Path, into: Path) -> Path:
    """A copy of the fixture export, with joint limits and the walk's controls
    in its contract.

    The committed fixture predates `joint_limits` and controller blocks, and
    re-exporting it would commit a new 1.5 MB actor.onnx to change two fields.
    Patched here instead -- and the unpatched original is the control group in
    the test below.
    """
    shutil.copytree(source, into)
    layout = into / "layout.json"
    contract = json.loads(layout.read_text("utf-8"))
    contract["joint_limits"] = {
        j: [contract["default_joint_pos"][j] - 1.0, contract["default_joint_pos"][j] + 1.0]
        for j in contract["wire_joint_order"]
    }
    contract["controller"] = WALK_CONTROLS
    layout.write_text(json.dumps(contract) + "\n", "utf-8")
    return into


@pytest.fixture(scope="module")
def deploy():
    """`scripts/deploy.py` as a module, by path; see `test_bundle_archive.py`."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("deploy_for_completeness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert_by_pretending(export: Path) -> None:
    """What `onnx2rknn.py` leaves beside an export, minus a real model."""
    import hashlib

    (export / "actor.rknn").write_bytes(b"RKNN\x00 not a real model, only its name")
    digest = hashlib.sha256((export / "actor.onnx").read_bytes()).hexdigest()
    (export / "actor.rknn.json").write_text(json.dumps({"onnx_sha256": digest}), "utf-8")


def test_a_complete_app_is_read_off_what_was_written(deploy, two_modes: Path) -> None:
    """The default refuses an app that lacks the board's controller, `play`'s
    Windows or macOS one, a `.rknn` or the reference vectors, and it reads that off
    `bundle.json` -- what a consumer receives -- rather than off the steps that
    should have made them.

    `two_modes` was built with `--allow-incomplete` from an export with no
    `.rknn`, so both modes must be named. The control group is the same
    manifest with the parts filled in: a check that named every bundle there is
    would pass the first half.
    """
    manifest = json.loads((two_modes / "bundle.json").read_text("utf-8"))
    lacking = deploy.gaps(manifest)
    assert "slow: no .rknn" in lacking and "walk: no .rknn" in lacking, lacking
    assert "no reference vectors" not in lacking, "two_modes carries one"
    assert ("no board controller" in lacking) == ("board" not in manifest["runtimes"])
    extensions = manifest["runtimes"]["mjlab"]["extensions"]
    assert ("no Windows controller for play" in lacking) == ("win-amd64" not in extensions)
    assert ("no macOS controller for play" in lacking) == ("macosx-universal2" not in extensions)

    manifest["runtimes"]["board"] = {"model": "rknn"}
    extensions["win-amd64"] = {"file": "runtime/mjlab/win-amd64/controller.pyd"}
    extensions["macosx-universal2"] = {"file": "runtime/mjlab/macosx-universal2/controller.so"}
    for entry in manifest["modes"].values():
        entry["models"]["rknn"] = "models/slow.rknn"
    assert deploy.gaps(manifest) == []
    manifest["reference"] = None
    assert deploy.gaps(manifest) == ["no reference vectors"]


def test_an_rknn_counts_only_beside_the_onnx_it_was_made_from(deploy, tmp_path) -> None:
    """A policy exported again into the same directory leaves the old `.rknn`
    beside it, looking finished. The converter records the ONNX's digest, and
    only a matching one counts -- otherwise the default converts again."""
    export = with_limits(EXPORT, tmp_path / "export")
    assert not deploy.converted(export)

    (export / "actor.rknn").write_bytes(b"RKNN\x00 made by something that kept no record")
    assert not deploy.converted(export), "an .rknn with no record of its ONNX"

    convert_by_pretending(export)
    assert deploy.converted(export), "the control: the record matches"

    (export / "actor.onnx").write_bytes((export / "actor.onnx").read_bytes() + b"\0")
    assert not deploy.converted(export), "the ONNX changed under the .rknn"


def test_the_default_says_what_it_cannot_make_before_it_makes_anything(
        deploy, tmp_path) -> None:
    """The board's half takes minutes -- a cross build, a conversion per policy
    -- so what would stop it is asked first, and a refusal writes nothing.

    The case that can be arranged on any machine: an export outside the
    repository, which the converter's container cannot see. The control is the
    same export already converted, which has nothing left to convert.
    """
    export = with_limits(EXPORT, tmp_path / "export")
    assert not export.resolve().is_relative_to(REPO), "the case needs an export outside"
    problems = deploy.preflight({"walk": export})
    assert any(str(export.resolve()) in p and "outside the repository" in p
               for p in problems), problems

    out = tmp_path / "strict"
    result = run("--bundle", str(out), f"--mode=walk={export}", complete=True)
    said = result.stdout + result.stderr
    assert result.returncode != 0, said
    assert "outside the repository" in said and "--allow-incomplete" in said
    assert "cross-building" not in said, "it refused only after starting the board's half"
    assert not out.exists(), "a refused app leaves nothing behind"

    convert_by_pretending(export)
    assert not any(str(export.resolve()) in p for p in deploy.preflight({"walk": export}))


def test_a_bundle_says_what_the_board_is_missing(tmp_path) -> None:
    """With `--allow-incomplete`: a bundle whose models are still ONNX cannot
    run on a board, so it says which modes and gives the command rather than
    shipping something that looks ready; and a bundle with no cross-built
    controller says that too.
    """
    out = tmp_path / "board"
    fsm = tmp_path / "one.toml"
    fsm.write_text(TWO_MODES, "utf-8")
    export = with_limits(EXPORT, tmp_path / "export")
    result = run("--bundle", str(out), "--fsm", str(fsm),
                 f"--mode=walk={export}", f"--mode=slow={export}")
    assert result.returncode == 0, result.stdout + result.stderr

    bundle = built(out)
    manifest = json.loads((bundle / "bundle.json").read_text("utf-8"))
    # No `.rknn` beside those exports, so the modes carry ONNX alone and the
    # bundle refuses to look ready. The command it prints is the one that fixes it.
    assert any("not ready for the board" in n for n in manifest["notes"])
    assert any("onnx2rknn.py" in n for n in manifest["notes"])
    assert manifest["modes"]["walk"]["models"] == {"onnx": "models/slow.onnx"}

    # The invariant, in whichever state this machine is: the board's runtime is
    # either **in** the bundle or **named** in the notes. Never neither -- a
    # bundle that looks complete is one somebody copies to a robot.
    #
    # Asserted as an either/or because both states are normal. It used to assert
    # the note, which was right until somebody ran `docker-build.sh` for the
    # first time and the binary started being there.
    has_binary = (bundle / "runtime/board/controller").is_file()
    says_missing = any("no board runtime" in n for n in manifest["notes"])
    assert has_binary != says_missing, (
        f"controller present={has_binary}, notes={manifest['notes']}"
    )
    assert has_binary == ("board" in manifest["runtimes"])
    if says_missing:
        assert any("docker-build.sh" in n for n in manifest["notes"])
    else:
        board = manifest["runtimes"]["board"]
        assert (board["model"], board["platform"]) == ("rknn", "aarch64-unknown-linux-gnu")
        assert "runtime/board/controller" in (bundle / "README.md").read_text("utf-8")


def test_a_converted_policy_ships_as_rknn_beside_the_onnx(tmp_path) -> None:
    """The other half of the same check: given a `.rknn`, it travels with the
    ONNX -- the board takes one, the browser the other -- and the bundle stops
    warning."""
    export = with_limits(EXPORT, tmp_path / "export")
    (export / "actor.rknn").write_bytes(b"RKNN\x00 not a real model, only its name")

    out = tmp_path / "ready"
    result = run("--bundle", str(out), f"--mode=walk={export}")
    assert result.returncode == 0, result.stdout + result.stderr
    bundle = built(out)
    assert {"models/walk.rknn", "models/walk.onnx"} <= files(bundle)
    manifest = json.loads((bundle / "bundle.json").read_text("utf-8"))
    assert manifest["modes"]["walk"]["models"] == {
        "onnx": "models/walk.onnx", "rknn": "models/walk.rknn"}
    assert not any("not ready" in n for n in manifest["notes"])


def test_a_board_is_not_given_a_policy_with_no_joint_limits(tmp_path) -> None:
    """The reason `joint_limits` exists at all.

    Both simulators read the hard stops off the model they loaded and clamp
    every commanded target to them. The board has no model, so before this it
    clamped to a pair in its own config -- and the pair it held was the
    placeholder out of this crate's unit tests, `[-3.30, 3.30]`, which is wider
    than twenty of this robot's twenty-two joints can travel. The clamp was
    there, ran every tick, and could not fire.

    One bundle carries the board whenever a controller has been cross-built
    for it, so an export predating the field refuses the whole bundle then:
    loading in a browser and not on the board is a broken bundle. `EXPORT` is
    such an export, which is what makes this a real control rather than a
    hand-built one; the same export given limits is the other half.
    """
    if not CROSS_BUILT.is_file():
        pytest.skip("no board runtime is cross-built here, so no board to refuse it")
    assert "joint_limits" not in json.loads((EXPORT / "layout.json").read_text("utf-8"))

    result = run("--bundle", str(tmp_path / "board"), f"--mode=walk={EXPORT}")
    assert result.returncode != 0
    assert "no joint limits" in result.stdout + result.stderr
    assert "Re-export" in result.stdout + result.stderr

    export = with_limits(EXPORT, tmp_path / "export")
    result = run("--bundle", str(tmp_path / "limited"), f"--mode=walk={export}")
    assert result.returncode == 0, result.stdout + result.stderr


MANIFEST = {
    "schema": "kk-deploy-manifests/1",
    "bundles": {
        "two": {
            "description": "two gaits",
            "fsm": "cascade.toml",
            "modes": {
                "walk": {"task": "jumper.tripod", "policy": None, "keys": {"key": ["keypad_1"]}},
                "slow": {"task": "jumper.tripod", "policy": None, "pad": {"button": "A"}},
            },
        }
    },
}

#: The manifest's counterpart: the cascade, and nothing the manifest owns.
CASCADE = """
[fsm]
initial_state = "walk"
warm_start_ref = "walk"
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

[[fsm.rule]]
when = "feedback_stale"
enter = "safe"

[[fsm.rule]]
when = "in_state:safe"
enter = "@initial"

[[fsm.rule]]
when = "button:slow"
enter = "slow"


# `walk` is also where `always` lands, so this rule is what makes its key do
# something -- the crate refuses a binding no rule reacts to, on the grounds
# that a key that does nothing is indistinguishable from one that is broken.
[[fsm.rule]]
when = "button:walk"
enter = "walk"

[[fsm.rule]]
when = "always"
enter = "walk"
"""


def write_manifest(tmp_path: Path, export: Path) -> Path:
    import copy

    data = copy.deepcopy(MANIFEST)
    where = str(export.relative_to(REPO)) if export.is_relative_to(REPO) else str(export)
    for entry in data["bundles"]["two"]["modes"].values():
        entry["policy"] = where
    path = tmp_path / "manifests.json"
    path.write_text(json.dumps(data), "utf-8")
    (tmp_path / "cascade.toml").write_text(CASCADE, "utf-8")
    return path


def test_a_manifest_records_where_every_mode_came_from(tmp_path) -> None:
    """The build's input, so a bundle records its ancestry and not only its
    contents.

    Which checkpoint a mode came from lived on somebody's command line. A
    bundle rebuilt from a different run of the same task differs in nothing the
    bundler looks at -- same mode names, same shapes, same joint order.
    """
    export = with_limits(EXPORT, tmp_path / "export")
    manifest = write_manifest(tmp_path, export)
    out = tmp_path / "b"
    result = run("--bundle", str(out), "--manifest", "two", "--manifests", str(manifest))
    assert result.returncode == 0, result.stdout + result.stderr

    ancestry = json.loads((built(out, "two") / "bundle.json").read_text("utf-8"))["built_from"]
    assert sorted(ancestry) == ["slow", "walk"]
    assert ancestry["walk"]["task"] == "jumper.tripod"
    assert ancestry["walk"]["keys"] == {"key": ["keypad_1"]}
    assert ancestry["slow"]["pad"] == {"button": "A"}

    # The bindings and the model states are composed in, so the file that
    # travels is complete even though neither input file was.
    written = (built(out, "two") / "controller.toml").read_text("utf-8")
    assert 'key = "keypad_1"' in written and 'name = "walk"' in written
    assert 'pad = "A"' in written and 'on = "toggle"' in written
    assert 'model = "walk.onnx"' in written and 'model = "slow.onnx"' in written


def test_neither_file_may_say_the_others_half(tmp_path) -> None:
    """The control group, and the reason the split is worth having.

    Two files naming the same key is how the two come to disagree, and nothing
    reads both. So a cascade declaring a binding -- or a mode -- is refused
    rather than merged over.
    """
    export = with_limits(EXPORT, tmp_path / "export")
    manifest = write_manifest(tmp_path, export)
    args = ["--bundle", str(tmp_path / "b"), "--manifest", "two",
            "--manifests", str(manifest)]

    for extra, expect in (
        ('\n[fsm.key]\nkeypad_9 = "walk"\n', "[fsm.key]"),
        ('\n[[fsm.button]]\nname = "walk"\npad = "B"\non = "rise"\n', "[[fsm.button]]"),
        ('\n[[fsm.state]]\nname = "walk"\nmodel = "walk.onnx"\n', "a [[fsm.state]]"),
    ):
        (tmp_path / "cascade.toml").write_text(CASCADE + extra, "utf-8")
        result = run(*args)
        assert result.returncode != 0, extra
        assert expect in result.stdout + result.stderr, extra
        assert "the manifest owns" in result.stdout + result.stderr

    # Without the overlap it builds. Otherwise the three above would pass
    # against a bundler that refuses every manifest there is.
    (tmp_path / "cascade.toml").write_text(CASCADE, "utf-8")
    assert run(*args).returncode == 0


def test_a_manifest_takes_an_export_and_not_a_checkpoint(tmp_path) -> None:
    """A checkpoint is not something a robot can run.

    Exporting is where the contract is built and where the validations decide
    whether it can run at all, so a manifest naming a checkpoint makes the
    bundler pick an export on the reader's behalf. It did, briefly -- the newest
    matching one -- which is a silent choice about what ships, made at build
    time, from a field that looks like it names a thing.

    The ancestry survives leaving it out: the export records its own checkpoint
    and `built_from` reads it back, so there is no second copy to disagree.
    """
    export = with_limits(EXPORT, tmp_path / "export")
    manifest = write_manifest(tmp_path, export)

    data = json.loads(manifest.read_text("utf-8"))
    for entry in data["bundles"]["two"]["modes"].values():
        entry.pop("policy")
        entry["checkpoint"] = "logs/jumper/jumper.tripod/whenever/model_5700.pt"
    manifest.write_text(json.dumps(data), "utf-8")

    result = run("--bundle", str(tmp_path / "b"), "--manifest", "two",
                 "--manifests", str(manifest))
    assert result.returncode != 0
    out = result.stdout + result.stderr
    assert "names a checkpoint. This takes an export" in out
    assert "scripts/export.py --task jumper.tripod" in out, "it must say how to get one"


def test_the_checkpoint_in_a_bundle_comes_from_the_export(tmp_path) -> None:
    """`built_from` reads the ancestry out of the export, not out of the manifest.

    The manifest names a directory; the directory says which checkpoint it came
    from. Copying that into the manifest as well would be two statements about
    one fact, and the failure would be a bundle whose recorded run is not the
    run that produced it.
    """
    export = with_limits(EXPORT, tmp_path / "export")
    layout = export / "layout.json"
    contract = json.loads(layout.read_text("utf-8"))
    contract["_checkpoint"] = {
        "path": "logs/jumper/jumper.tripod/2026-01-01_00-00-00/model_42.pt",
        "run": "logs/jumper/jumper.tripod/2026-01-01_00-00-00",
        "iteration": "model_42",
    }
    layout.write_text(json.dumps(contract), "utf-8")

    out = tmp_path / "b"
    manifest = write_manifest(tmp_path, export)
    assert run("--bundle", str(out), "--manifest", "two",
               "--manifests", str(manifest)).returncode == 0

    ancestry = json.loads((built(out, "two") / "bundle.json").read_text("utf-8"))["built_from"]
    assert ancestry["walk"]["from"] == "logs/jumper/jumper.tripod/2026-01-01_00-00-00/model_42.pt"

    # An export from before that record says so, rather than the bundle claiming
    # a provenance it does not have.
    del contract["_checkpoint"]
    layout.write_text(json.dumps(contract), "utf-8")
    out2 = tmp_path / "b2"
    assert run("--bundle", str(out2), "--manifest", "two",
               "--manifests", str(manifest)).returncode == 0
    ancestry = json.loads((built(out2, "two") / "bundle.json").read_text("utf-8"))["built_from"]
    assert "records no checkpoint" in ancestry["walk"]["from"]


def test_a_mode_whose_export_is_missing_names_the_command(tmp_path) -> None:
    """The manifest is right and the export has not been run. That is not an
    error about the manifest, so it does not read like one."""
    manifest = write_manifest(tmp_path, tmp_path / "never-exported")
    result = run("--bundle", str(tmp_path / "b"), "--manifest", "two",
                 "--manifests", str(manifest))
    assert result.returncode != 0
    out = result.stdout + result.stderr
    assert "names an export that is not there" in out
    assert "scripts/export.py --task jumper.tripod" in out
    # Exports are timestamped and untracked, so this is the normal way a
    # manifest arrives from another machine -- said, rather than left as a
    # missing path.
    assert "another machine" in out


def test_the_reference_is_replayable_and_can_fail(tmp_path) -> None:
    """The bundle records what its controller does, so three hosts can be held
    against it rather than against each other.

    They compile one source with three toolchains for two architectures and run
    the policy on three inference backends. "Same source" survives all of that
    and so would a divergence, which is why this is a recorded measurement and
    not a comment.

    The control group matters more than usual here: the host that *generates*
    the reference will always match it, so a check that only ran there would be
    green by construction. This perturbs the file and requires the check to go
    red -- and the numbers it reports are the size of the perturbation, not a
    vaguer complaint.
    """
    parent = tmp_path / "ref"
    export = with_limits(EXPORT, tmp_path / "export")
    result = run("--bundle", str(parent), f"--mode=walk={export}")
    assert result.returncode == 0, result.stdout + result.stderr
    out = built(parent)

    manifest = json.loads((out / "bundle.json").read_text("utf-8"))
    assert manifest["reference"] == "reference.json", manifest.get("notes")
    reference = json.loads((out / "reference.json").read_text("utf-8"))
    assert reference["schema"] == "kk-policy-reference/1"
    assert len(reference["frames"]) >= 8
    # Term spans, so a host names the term rather than an index. Without them a
    # difference in `joint_torque` -- the one term a robot sources differently
    # by design -- is indistinguishable from a real bug at index 404.
    assert [t["name"] for t in reference["terms"]], "the terms are not located"
    assert sum(t["dim"] for t in reference["terms"]) == reference["frames"][0]["obs"].__len__()

    clean = run("--check-reference", str(out))
    assert clean.returncode == 0, clean.stdout + clean.stderr
    assert "matches the reference" in clean.stdout

    # One joint target off by a milliradian: what a changed decode produces and
    # an eyeball does not catch.
    reference["frames"][3]["target"][0] += 1e-3
    (out / "reference.json").write_text(json.dumps(reference), "utf-8")
    dirty = run("--check-reference", str(out))
    assert dirty.returncode == 1, dirty.stdout
    assert "does not match" in dirty.stdout
    assert "1.000e-03" in dirty.stdout, dirty.stdout


def test_the_readme_describes_the_bundle_it_is_in(two_modes: Path) -> None:
    """It is the only part a consumer reads before writing code against the rest,
    so a placeholder that failed to substitute is worse than a missing file --
    and a README naming a file that is not here is worse still, because it
    reads as a bundle that is missing something."""
    import re

    readme = (two_modes / "README.md").read_text("utf-8")
    manifest = json.loads((two_modes / "bundle.json").read_text("utf-8"))

    assert not re.search(r"\{[A-Z_]+\}", readme), "an unsubstituted placeholder"

    # The fenced listing at the top, held against the directory. Checking the
    # whole README instead would be both weaker and impossible: `controller` is
    # a filename and also an English word in the first sentence.
    listing = re.search(r"```\n(.*?)```", readme, re.DOTALL)
    assert listing, "the README does not list its own files"
    named = {line.split()[0] for line in listing.group(1).splitlines()
             if line.split() and not line.startswith(" ")}
    on_disk = files(two_modes)
    assert named <= on_disk, f"the README lists {named - on_disk}, which is not here"
    # Everything but the README itself, which does not list itself.
    assert on_disk - named == {"README.md"}, f"unlisted files: {on_disk - named - {'README.md'}}"
    assert manifest["runtimes"]["web"]["commit"] in readme
    # The three that fail silently, and the isolation advice, are the reason
    # this file exists rather than a link. "both edges" replaced "rising edge":
    # a host that reports only presses can drive `toggle` and `rise` and
    # silently nothing else.
    for topic in ("wire order", "body frame", "opaque origin", "both edges"):
        assert topic in readme, topic
