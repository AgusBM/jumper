"""The controller, imported.

`deploy/fsm` is what the robot runs and what a browser runs. This is the same
crate as a Python extension, so `play` can run it too -- and the reason that
matters is not tidiness. An FSM is a file of rules in an order, and until this
existed the only way to find out what a particular file did was to bundle it and
upload it. A rule in the wrong place was a round trip through another repository
away from being visible.

What these pin is the boundary, not the controller: the controller has its own
55 tests in Rust. What can go wrong here is everything about crossing -- a joint
count nobody checked, a dict field read as an attribute, an optional key that
turns out to be required, an error that arrives as a panic instead of an
exception.

The extension is built here, by cargo, from this tree -- the same
`compile_extension` every bundle build runs. It used to be imported from
site-packages, where `deploy.py --python` put it: a build of any age, and on a
machine that had never run that command this whole file skipped and said
nothing a person would read. Skipped now only where there is no Rust toolchain,
which `pip install -e .` deliberately does not ask for: a machine that only
trains has no use for one.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXPORT = REPO / "tasks/jumper/tripod/out/rough"


def _extension():
    """This tree's controller, as the Python extension `play` imports."""
    if not EXPORT.is_dir():
        pytest.skip("no exported policy to drive", allow_module_level=True)
    spec = importlib.util.spec_from_file_location("deploy_for_extension",
                                                  REPO / "scripts/deploy.py")
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    cargo = deploy.which("cargo")
    if cargo is None:
        pytest.skip("cargo is not installed, so the controller extension cannot be built",
                    allow_module_level=True)
    built = deploy.compile_extension(cargo, verbose=False)
    # The name is not free: CPython looks for `PyInit_mjrl_fsm`.
    spec = importlib.util.spec_from_file_location("mjrl_fsm", built)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mjrl_fsm = _extension()

ONE_MODE = """
[fsm]
initial_state = "walk"
warm_start_ref = "walk"
safe_state = "safe"
tilt_limit = 3.15
state_timeout_ms = 200
command_timeout_ms = 500
mode_switch_ramp_s = 0.0
pose_reach_tol = 0.10
warm_start_duration_s = 0.0
ramp_kp = 0.15
ramp_kd = 0.01

# A binding, not the `[fsm.key]` table this used to carry. That table stopped
# being read when a control became a button **plus a gesture**, and this file
# kept working for a while anyway -- against an `mjrl_fsm.so` in site-packages
# built before the change. Rebuilding the extension is what surfaced it, which
# is worth remembering: a compiled artifact is a place a test suite can go on
# passing after the thing it tests has moved.
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


@pytest.fixture(scope="module")
def parts() -> tuple[str, list[str], list[float]]:
    text = (EXPORT / "layout.json").read_text("utf-8")
    contract = json.loads(text)
    wire = contract["wire_joint_order"]
    home = [contract["default_joint_pos"][j] for j in wire]
    return text, wire, home


def make(parts, config: str = ONE_MODE):
    text, wire, _ = parts
    n = len(wire)
    return mjrl_fsm.Fsm(
        config,
        {"walk": text, "slow": text},
        {
            "joint_names": wire,
            "joint_pos_lo": [-3.3] * n,
            "joint_pos_hi": [3.3] * n,
            "output_rate_hz": 50.0,
            "gait_period": 0.32,
            "gait_gate_threshold": 0.05,
        },
    )


def press(fsm, parts, key: str) -> bool:
    """One press of a key: down, a tick, up, a tick.

    The extension reports **edges**, not presses -- `rise` and `fall` are both
    in the vocabulary, so a one-shot `pressed` call could only ever drive half
    of it. This replaced `press_key`, which could not express the difference.

    Returns whether any binding names the key, so "that key does nothing" stays
    distinguishable from "the rule did not fire".
    """
    known = fsm.set_key(key, True)
    drive(fsm, parts, 1)
    fsm.set_key(key, False)
    drive(fsm, parts, 1)
    return known


def drive(fsm, parts, ticks: int, pose: list[float] | None = None) -> int:
    """Run the loop a host would run, and count the inferences."""
    _, wire, home = parts
    n = len(wire)
    q = home if pose is None else pose
    infers = 0
    for t in range(ticks):
        now = t * 20_000
        fsm.set_state(q, [0.0] * n, [0.0] * n, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], now)
        fsm.set_command(0.3, 0.0, 0.0, now_us=now)
        if fsm.tick(now) is not None:
            fsm.resume([0.0] * 20)
            infers += 1
    return infers


def test_it_runs_a_real_contract(parts) -> None:
    fsm = make(parts)
    assert sorted(fsm.modes) == ["slow", "walk"]
    assert fsm.bindings == [("slow", None, "keypad_1", "toggle", None)]

    infers = drive(fsm, parts, 40)
    assert infers == 40, "at the home pose there is no ramp to wait through"
    assert len(fsm.observation()) == json.loads(parts[0])["observation"]["dim"]
    assert len(fsm.positions()) == len(parts[1])
    assert fsm.running_policy


def test_a_pose_the_ramp_cannot_reach_never_starts_the_policy(parts) -> None:
    """The behaviour that is easiest to mistake for a broken build.

    The hand-off waits on the *measured* pose, not on a timer, so gains too soft
    to converge mean it waits indefinitely and the robot soft-holds. That is
    deliberate -- the alternative is starting a policy from a pose it never
    trained around -- and `running_policy` is how a host tells the two apart.
    """
    ramped = ONE_MODE.replace("mode_switch_ramp_s = 0.0", "mode_switch_ramp_s = 1.0")
    fsm = make(parts, ramped)
    away = [0.0] * len(parts[1])
    assert drive(fsm, parts, 200, pose=away) == 0
    assert fsm.mode == "walk", "the FSM chose the mode"
    assert not fsm.running_policy, "...and the policy is not what is running"


def test_a_key_switches_modes_and_an_unbound_one_says_so(parts) -> None:
    fsm = make(parts)
    drive(fsm, parts, 5)
    assert fsm.mode == "walk"

    assert press(fsm, parts, "keypad_1") is True
    assert fsm.mode == "slow"

    # Pressing again leaves: a press is one request and the latch toggles.
    assert press(fsm, parts, "keypad_1") is True
    assert fsm.mode == "walk"

    # An unbound key is not an error and not a silent no-op.
    assert press(fsm, parts, "keypad_9") is False


def test_the_operator_going_away_does_not_leave_a_mode_latched(parts) -> None:
    """From the C++, which learned it the useful way: a stuck mode keeps the FSM
    in a state that may suspend the tilt fallback, with nobody left to release
    it."""
    fsm = make(parts)
    drive(fsm, parts, 5)
    press(fsm, parts, "keypad_1")
    assert fsm.mode == "slow"

    # State keeps arriving; the command does not. `command_timeout_ms` is 500.
    _, wire, home = parts
    n = len(wire)
    now = 10_000_000
    fsm.set_state(home, [0.0] * n, [0.0] * n, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], now)
    fsm.tick(now)
    assert fsm.mode == "walk"


def test_nothing_crosses_the_boundary_unchecked(parts) -> None:
    """Every one of these arrives as an exception with the reason in it. A
    panic across the boundary aborts the interpreter, which in `play` means
    losing the viewer and the episode with it."""
    text, wire, home = parts
    n = len(wire)

    with pytest.raises(ValueError, match="no 'joint_pos_lo'"):
        mjrl_fsm.Fsm(ONE_MODE, {"walk": text}, {"joint_names": wire})

    # A mode the contracts dict does not cover. Looked up by **state name**, not
    # by model filename -- renaming the file would change nothing, which is the
    # first thing this test tried.
    third = ONE_MODE.replace(
        '[[fsm.rule]]\nwhen = "feedback_stale"',
        '[[fsm.state]]\nname = "sprint"\nmodel = "sprint.onnx"\n\n'
        + '[[fsm.rule]]\nwhen = "feedback_stale"',
        1,
    )
    with pytest.raises(ValueError, match="no contract was installed"):
        make(parts, third)

    fsm = make(parts)
    with pytest.raises(ValueError, match=f"{n} joints; q has 3"):
        fsm.set_state([0.0] * 3, [0.0] * n, [0.0] * n, [1, 0, 0, 0], [0, 0, 0], 0)
    with pytest.raises(ValueError, match="quat must be"):
        fsm.set_state(home, [0.0] * n, [0.0] * n, [1, 0, 0], [0, 0, 0], 0)

    # An action nobody asked for, and one of the wrong width.
    with pytest.raises(ValueError, match="no inference outstanding"):
        fsm.resume([0.0] * 20)


def test_play_is_the_host_where_nothing_has_moved(parts) -> None:
    """`source.rs` reports every term a host supplies differently than training
    did. On mjlab that list is empty by construction -- this is where the policy
    trained -- and it is not empty on the robot. A host that reported
    divergences here would be reporting that `play` is not mjlab."""
    fsm = make(parts)
    assert fsm.divergences("walk") == []
