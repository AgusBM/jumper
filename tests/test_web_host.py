"""The browser host, actually loaded and actually driven.

Every other host is exercised somewhere: the board's controller has 101 Rust
tests and a `--check-reference` anybody can run, `play`'s extension has
`tests/test_fsm_extension.py`. The browser had nothing. It is built by CI, it
is packed into every bundle, and until this file existed the only way to find
out whether a bundle would load in a page was to open a page.

That gap produced exactly the failure it invites. `WebFsm::new` took no
argument for a recorded motion, so every bundle carrying `jumper.jump` or
`jumper.dance` -- which is the bundle this repository actually ships -- threw
on construction. The board had solved it by reading the file and `play` by
being handed the text; the browser was simply never tried.

Node rather than a browser, and the wasm handed over as bytes rather than
fetched: what is under test is the controller, not the network. Everything
else is assembled the way `deploy/fsm/BUNDLE_README.md` tells a page to
assemble it, so this also fails when that document goes stale.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LOADER = Path(__file__).parent / "web" / "load_bundle.mjs"


def newest_web_bundle() -> Path | None:
    """The most recently built bundle that carries a browser's runtime, if any."""
    built = sorted(
        (p.parent for p in (REPO / "out").glob("bundle_*/*/bundle.json")
         if "web" in json.loads(p.read_text("utf-8")).get("runtimes", {})),
        key=lambda p: p.stat().st_mtime,
    )
    return built[-1] if built else None


@pytest.fixture(scope="module")
def loaded() -> dict:
    if shutil.which("node") is None:
        pytest.skip("no node on this machine")
    web = newest_web_bundle()
    if web is None:
        pytest.skip(
            "no bundle to load: python scripts/deploy.py --bundle --manifest jumper"
        )
    out = subprocess.run(
        ["node", str(LOADER), str(web)],
        capture_output=True, text=True, check=False, timeout=300,
    )
    assert out.stdout, f"the loader printed nothing.\nstderr:\n{out.stderr[-2000:]}"
    return json.loads(out.stdout)


def test_the_bundle_a_page_would_fetch_actually_builds(loaded: dict) -> None:
    """Construction, with the trajectories a reference-guided mode needs.

    The failure this pins is not subtle once you look -- it is a thrown error on
    the first line a page runs. It survived because nothing ran that line.
    """
    assert loaded["ok"], loaded.get("error")
    assert loaded["modes"], "a bundle with no modes is not a bundle"


def test_a_recorded_motion_travels_with_the_bundle(loaded: dict) -> None:
    """Whichever modes declare a recording got one.

    Reading this from the contracts rather than naming `jump` and `dance`: the
    point is that the set the bundle *declares* and the set the page *supplies*
    are the same set, whatever they contain.
    """
    if not loaded["withTrajectory"]:
        pytest.skip("this bundle carries no reference-guided mode")
    assert set(loaded["withTrajectory"]) <= set(loaded["modes"])


def test_driving_it_produces_inferences_and_a_target(loaded: dict) -> None:
    """Built is not running. Four seconds of ticks, and it has to ask.

    A controller that constructs and then never asks for an inference is the
    other half of the same silence: the page renders, the robot stands, and
    nothing says why.
    """
    assert loaded["infers"] > 0, "four seconds of ticks and it never asked to infer"
    assert loaded["inferredModes"], loaded
    assert loaded["mode"] in loaded["modes"]
    assert len(loaded["positions"]) == loaded["joints"], (
        f"{len(loaded['positions'])} targets for {loaded['joints']} joints -- the "
        f"publish side would be misaligned from wherever they stopped matching"
    )
    assert any(p != 0.0 for p in loaded["positions"]), (
        "every joint target is exactly zero, which is not a pose this robot holds"
    )


def test_this_host_computes_what_the_others_did(loaded: dict) -> None:
    """The reference vectors, replayed here.

    This is the whole reason a bundle carries `reference.json`: the three hosts
    run the same source and must produce the same numbers, and agreement is
    measured rather than assumed. `observation` and `target` are maxima over
    every frame, so anything above rounding is a real difference.
    """
    r = loaded["reference"]
    assert r["frames"] > 0 and r["inferences"] > 0, r
    assert not r["modeMismatches"], r["modeMismatches"]
    assert r["observation"] < 1e-6, f"observation differs by {r['observation']}"
    assert r["target"] < 1e-6, f"target differs by {r['target']}"
    assert not r["divergent"], (
        f"terms this host sources differently than the one that recorded: "
        f"{r['divergent']}"
    )


def test_the_contract_maps_the_stick_not_the_page(loaded: dict) -> None:
    """Raw sticks in, a command out, and the contract is what turned one into
    the other.

    A page used to do this itself -- read the pad, apply the sign, scale by the
    range, hand over a finished command. Two implementations of one contract,
    and they drifted: the browser's table still spelled the axes
    `left_stick_y` long after the dictionary had closed on `Ly`, which is a
    bundle refused for naming a stick the way its own exporter names it.

    So the host now pushes what the Gamepad API reports and nothing else. What
    this pins is the part with no second chance: **the sign**. Pushing a stick
    forward is `Ly = -1` in evdev's convention, and if that arrives as backwards
    the robot drives away from the operator while every screen looks correct.
    """
    pad = loaded.get("pad")
    if not pad:
        pytest.skip("no mode in this bundle observes a command")

    lin_x = pad["axes"].get("lin_vel_x")
    assert lin_x, f"the contract binds no stick to lin_vel_x: {pad['axes']}"
    assert lin_x["source"] == "Ly", (
        f"lin_vel_x is driven by {lin_x['source']!r}; this test reasons about `Ly`"
    )

    centred, forward, back = pad["centred"], pad["forward"], pad["back"]
    assert centred[0] == 0.0, f"a centred stick asked for {centred[0]}"

    # evdev reports a stick pushed **forward** as negative, and the contract
    # carries sign -1, so `Ly = -1` has to come out positive.
    assert forward[0] > 0, (
        f"stick forward (Ly=-1) asked for {forward[0]}, which is backwards. "
        f"Nothing downstream can catch this; a hand on the pad can"
    )
    assert back[0] < 0, f"stick back (Ly=+1) asked for {back[0]}"
    assert forward[0] == pytest.approx(-back[0], rel=1e-6), (
        f"the two ends are not symmetric: {forward[0]} vs {back[0]}"
    )
    # Full deflection is the edge of the range the policy trained on, not an
    # absolute speed -- so it can never be asked for something off-distribution.
    assert 0 < forward[0] <= 5.0, f"full stick asked for {forward[0]} m/s"


def test_a_go_event_actually_starts_the_motion(loaded: dict) -> None:
    """The button that fires a recorded motion, pressed on this host.

    `rise` and `fall` are one frame wide and are derived from the change since
    the last observation, so two observers do not see them twice -- the second
    destroys them. This host had two: `padFrame` latched the buttons and then
    `tick` latched them again, found no edge, and handed the controller an empty
    set. A `toggle` survived, because a toggle is remembered rather than
    re-derived, so entering the mode worked perfectly and the `go` never fired.

    Nothing reported it. The mode was entered, the policy inferred every tick,
    the panel said `jump`, and the robot stood still with `jump_phase` at 0 for
    the whole length of a motion nobody could start. What separates the two is
    the motion's own clock, which is what this reads.
    """
    go = loaded.get("go")
    if not go:
        pytest.skip("this bundle has no mode with a `go_event` bound to a pad button")
    assert go["phasesBeforeGo"] <= 1, (
        f"the motion clock was already moving before the `go`: "
        f"{go['phasesBeforeGo']} distinct phases in {go['mode']}"
    )
    assert go["phasesAfterGo"] > 10, (
        f"{go['button']} on {go['on']} did not start {go['mode']}: its clock took "
        f"{go['phasesAfterGo']} distinct value(s) over three seconds. The mode is "
        f"entered and inferring either way, so nothing else here would notice"
    )


def test_entering_a_motion_that_starts_on_entry_starts_it(loaded: dict) -> None:
    """The recorded motion whose mode is the whole of asking for it, entered on
    this host and nothing else pressed.

    `jumper.jump` waited for a `go` on A's release until 2026-09-26; its contract
    now says `starts_on_entry`, and A only switches into it. So the silent
    failure moved: a clock that never starts looks, from outside, exactly as it
    did when the `go` was lost -- mode entered, policy inferring, the robot
    standing -- and a mode that never hands back is a robot frozen in its
    landing pose. The control group is the half second before the press, when
    the motion's clock must not have moved at all.
    """
    entry = loaded.get("entry")
    if not entry:
        pytest.skip("this bundle has no motion that starts on entry and a pad control to it")
    assert entry["phasesBefore"] == 0, (
        f"{entry['mode']} ran before anything reached it: {entry['phasesBefore']} phases"
    )
    assert entry["phasesDuring"] > 10, (
        f"entering {entry['mode']} on {entry['enter']} did not start it: its clock took "
        f"{entry['phasesDuring']} distinct value(s) over {entry['windowS']:.1f} s"
    )
    assert entry["modeAfter"] != entry["mode"], (
        f"{entry['mode']} ({entry['durationS']:.2f} s) was still running "
        f"{entry['windowS']:.1f} s after it was entered: the recording ended and "
        f"nothing handed the robot back"
    )


def test_a_key_switch_changes_modes_as_a_page_presses_it(loaded: dict) -> None:
    """The keyboard's own switches, pressed as a page presses them.

    The keys a page is handed are browser codes, and a key the dictionary
    names under another code, or one the host does not count as bound, is a
    key that does nothing on a page that looks right -- and a page that does
    not `preventDefault` it scrolls instead. So each keyboard switch that may
    be pressed in the mode the robot starts in is pressed by its code, modifier
    first, and the mode read back. The control group is a chord's key without
    its modifier (1 without Ctrl), which must not reach the chord's mode.
    """
    pressed = loaded.get("keyboard")
    if not pressed:
        pytest.skip("this bundle binds no key to a mode switch")
    for p in pressed:
        assert all(p["codes"]), f"{p['keys']} have no browser code: {p}"
        assert all(p["bound"]), f"{p['codes']} were not reported as bound: {p}"
        assert p["modeAfter"] == p["mode"], (
            f"{' + '.join(p['codes'])} left the robot in {p['modeAfter']!r}, not {p['mode']!r}"
        )
        if p["with"]:
            assert p["alone"] != p["mode"], (
                f"{p['codes'][-1]} without {p['codes'][0]} switched to {p['alone']!r}: "
                f"the modifier is not being read"
            )


def test_a_pad_button_switches_modes_in_a_browser_d_pad_included(loaded: dict) -> None:
    """Every mode switch the bundle binds to the pad, pressed through `setPad`
    the way `BUNDLE_README.md` tells a page to call it -- the d-pad's directions
    by their names. `setPad` used to know only the ten buttons, so `menu` and
    right, the dance, could not be made from a pad in a browser, and the
    `false` it returned read as "nothing binds this". The control group is the
    chord's button without its modifier, which must switch nothing.
    """
    pressed = loaded.get("padSwitch")
    if not pressed:
        pytest.skip("this bundle binds no mode switch to the pad")
    assert any(b.startswith("dpad_") for p in pressed for b in p["buttons"]), (
        "no switch on the d-pad: this test has lost the case it was written for"
    )
    for p in pressed:
        assert all(p["bound"]), f"{p['buttons']} were not reported as bound: {p}"
        assert p["modeAfter"] == p["mode"], (
            f"{' + '.join(p['buttons'])} left the robot in {p['modeAfter']!r}, not {p['mode']!r}"
        )
        if p["alone"] is not None:
            assert p["alone"] != p["mode"], (
                f"{p['buttons'][-1]} alone switched to {p['alone']!r}, the chord's mode"
            )


def test_the_pad_a_page_draws_is_the_pad_the_controller_reads(loaded: dict) -> None:
    """`padGuide()`, the drawing, against `inputs().keys`, what is bound.

    A page draws the pad and the keyboard from the guide -- two lists since
    2026-09-29, a key never drawn on a pad control. A key bound and missing
    from the drawing works with nothing to say so; a key drawn and not bound
    is a label on a key that does nothing -- and the page would not
    `preventDefault` it, so an arrow would scroll the page instead. So the two
    are held to one set, and every key switch pressed above is drawn with the
    mode it enters. The control group is the whole pad being there, used or
    not, which a guide listing only what it binds would fail.
    """
    guide = loaded.get("guide")
    if guide is None or "keyboard" not in guide:
        pytest.skip("this bundle's controller predates the keyboard's own list in padGuide()")
    controls = {c["control"]: c for c in guide["controls"]}
    drawn = sorted({code for k in guide["keyboard"]
                    for code in (*k["codes"]["key"], *k["codes"]["modifier"])})
    assert drawn == loaded["boundCodes"], (
        f"drawn and not bound: {sorted(set(drawn) - set(loaded['boundCodes']))}; "
        f"bound and not drawn: {sorted(set(loaded['boundCodes']) - set(drawn))}"
    )
    assert {"A", "B", "X", "Y", "LB", "RB", "Lx", "Ly", "Rx", "Ry", "LT", "RT",
            "dpad_up", "dpad_down", "dpad_left", "dpad_right", "menu"} <= set(controls)
    assert all("keys" not in c for c in guide["controls"]), "a key is drawn on the pad again"

    strokes = {k["stroke"]: k for k in guide["keyboard"]}
    for p in loaded.get("keyboard") or []:
        stroke = f"{p['with']}+{p['key']}" if p["with"] else p["key"]
        assert stroke in strokes, f"{stroke} switches into {p['mode']} and is not drawn"
        assert any(s["enters"] == p["mode"] for s in strokes[stroke]["switches"]), (
            f"{stroke} switches into {p['mode']} and the drawing does not say so"
        )


def test_a_key_held_is_the_bundles_pad_pushed_and_a_stop_lets_go(loaded: dict) -> None:
    """The keyboard and the page's Stop, through the controller a page loads.

    Three silences this pins. A page hands over `KeyboardEvent.code`s and the
    wasm's own dictionary decides which are keys: a code that is not translated
    is a key that does nothing, on a page that looks right. A key let up that
    keeps driving is a robot walking on after the hand has left the keyboard.
    And a Stop that does not reach a key still held stops nothing, with every
    screen saying it has.

    The control group is the pad: the key held past `full_after_s` must be what
    the full stick asks for, exactly, and held half that must be half of it --
    which is what makes the key's number mean something rather than merely
    non-zero.
    """
    keys = loaded.get("keys")
    if not keys:
        pytest.skip("this bundle's controller does not read raw input")
    inputs = keys["inputs"]
    assert inputs["rawInput"] is True
    assert inputs["controls"], "the operator gave no account of its own bindings"
    assert keys["forwardKey"], (
        f"nothing binds `key_w`, so there is no key to press: {inputs['keys']}"
    )
    assert keys["fullAfterS"], "the bundle's keyboard carries no `full_after_s`"
    assert keys["bound"] is True, "the controller says it binds no such key"
    assert keys["unbound"] is False, "a key nothing binds was reported as bound"

    full = loaded["pad"]["forward"][0]
    # Read at the last inference before the half second, a few milliseconds short.
    assert keys["half"][0] == pytest.approx(full / 2, rel=0.02), (
        f"held {keys['fullAfterS'] / 2} s the key asked for {keys['half'][0]}, the "
        f"full stick for {full}: a key's hold is not its share of the travel"
    )
    assert keys["held"][0] == pytest.approx(full, rel=1e-5), (
        f"held past {keys['fullAfterS']} s, with the browser's auto-repeat, the key "
        f"asked for {keys['held'][0]}, not the full stick's {full}"
    )
    assert keys["letUp"][0] == 0.0, (
        f"let up, the key still asks for {keys['letUp'][0]}: the stick did not centre"
    )
    assert keys["letGo"] is True, "letGo found no operator to let go"
    assert keys["released"][0] == 0.0, (
        f"after letGo, with the key still down, the command is {keys['released'][0]}: "
        f"a Stop button would stop nothing"
    )
