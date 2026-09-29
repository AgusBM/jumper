"""The Windows pad reader, `controller/xinput.py`, without Windows.

`translate` and the struct are plain ctypes and run anywhere, so everything about
what a reading *means* is checked here on Linux against the same `xbox.normalise`
a Linux pad goes through. The DLL call itself is replaced by a fake slot table;
what that cannot cover -- the real `xinput1_4.dll` answering -- is checked by
running `python -m controller` on a Windows machine (or under wine, which loads
the DLL and reports no pad), not pretended about here.
"""

from __future__ import annotations

import sys

import pytest

import controller
from controller import xinput
from controller.xbox import ABS_X, EV_ABS, GamepadState, normalise
from controller.xinput import XInputGamepad, translate


def _state(**fields) -> GamepadState:
    return normalise(translate(XInputGamepad(**fields)), xinput.RANGES)


def _moved(s: GamepadState) -> dict[str, float]:
    """Every field of a snapshot that is not at rest."""
    values = {
        "lx": s.lx, "ly": s.ly, "rx": s.rx, "ry": s.ry, "lt": s.lt, "rt": s.rt,
        "hat_x": s.hat_x, "hat_y": s.hat_y,
    }
    moved = {k: v for k, v in values.items() if v != 0}
    moved.update({b: 1 for b in s.buttons})
    return moved


def test_the_struct_holds_what_the_ex_call_writes() -> None:
    """`XInputGetStateEx` writes 20 bytes where the documented `XINPUT_STATE` is 16.
    A 16-byte buffer does not raise -- the call writes the last four bytes over
    whatever sits next to it."""
    import ctypes

    assert ctypes.sizeof(xinput.XInputState) == 20
    assert xinput.XInputState.Gamepad.offset == 4
    assert xinput.XInputGamepad.sThumbRY.offset == 10


def test_pushing_the_stick_up_reads_as_evdev_up() -> None:
    """XInput's Y is positive **up**; evdev's, and every task's `sign`, is positive
    down. `test_controller.py` pins that `ly = -1` walks forward, so this is the
    half that decides whether a Windows pad walks forward or backward -- silently
    either way."""
    up = _state(sThumbLY=32767, sThumbRY=32767)
    assert up.ly == pytest.approx(-1.0) and up.ry == pytest.approx(-1.0)
    down = _state(sThumbLY=-32768, sThumbRY=-32768)
    assert down.ly == pytest.approx(1.0) and down.ry == pytest.approx(1.0)

    # Control group: the same reading without the flip is the opposite command,
    # so the assertions above are not satisfied by any value at all.
    raw = translate(XInputGamepad(sThumbLY=32767))
    raw[(EV_ABS, 0x01)] = 32767
    assert normalise(raw, xinput.RANGES).ly == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("reading", "expected"),
    [
        ({"sThumbLX": 32767}, {"lx": 1.0}),
        ({"sThumbLX": -32768}, {"lx": -1.0}),
        ({"sThumbLY": 32767}, {"ly": -1.0}),
        ({"sThumbRX": 32767}, {"rx": 1.0}),
        ({"sThumbRY": 32767}, {"ry": -1.0}),
        ({"bLeftTrigger": 255}, {"lt": 1.0}),
        ({"bRightTrigger": 255}, {"rt": 1.0}),
        ({"wButtons": xinput.DPAD_UP}, {"hat_y": -1}),
        ({"wButtons": xinput.DPAD_DOWN}, {"hat_y": 1}),
        ({"wButtons": xinput.DPAD_LEFT}, {"hat_x": -1}),
        ({"wButtons": xinput.DPAD_RIGHT}, {"hat_x": 1}),
    ],
)
def test_each_control_moves_its_own_field_and_nothing_else(reading, expected) -> None:
    """A trigger read as the right stick is the bug the Bluetooth layout had on
    Linux: no error, one control doing another's job. Every other field must stay
    exactly at rest, which is also what catches a range that pulled
    `xbox.layout_of` onto the wrong layout."""
    moved = _moved(_state(**reading))
    assert moved.keys() == expected.keys(), f"{reading} moved {moved}"
    for name, value in expected.items():
        assert moved[name] == pytest.approx(value)


def test_the_buttons_are_the_physical_ones() -> None:
    """XInput names physical buttons, so its X is the dictionary's `X` -- whatever
    evdev's 307/308 turn out to be. Each bit is exactly one name and all eleven
    reach the snapshot."""
    seen = {}
    for bit in xinput.BUTTON_CODES:
        pressed = _state(wButtons=bit).buttons
        assert len(pressed) == 1, f"bit {bit:#06x} pressed {sorted(pressed)}"
        seen[bit] = next(iter(pressed))
    assert seen[0x1000] == "A" and seen[0x2000] == "B"
    assert seen[0x4000] == "X" and seen[0x8000] == "Y"
    assert seen[0x0400] == "home" and seen[0x0020] == "view" and seen[0x0010] == "menu"
    assert sorted(seen.values()) == sorted(
        ["A", "B", "X", "Y", "LB", "RB", "view", "menu", "home", "L3", "R3"]
    )


def test_a_stick_resting_off_centre_is_not_input() -> None:
    """A stick does not return to exactly zero. Read as input, a resting pad
    commands a creep and, through `any_input`, takes the command away from the
    keyboard whenever it is plugged in."""
    resting = {
        "sThumbLX": -3000, "sThumbLY": 2500, "sThumbRX": 4000, "sThumbRY": -3500,
        "bLeftTrigger": 20, "bRightTrigger": 20,
    }
    s = _state(**resting)
    assert _moved(s) == {} and not s.any_input

    # Control group: with the flat `xpad` declares over USB (128), the same
    # reading is input, so it is the dead zone doing the work above.
    narrow = {code: (lo, hi, 128 if hi > 255 else flat)
              for code, (lo, hi, flat) in xinput.RANGES.items()}
    assert normalise(translate(XInputGamepad(**resting)), narrow).any_input

    # Rescaled, not clipped: just past the zone starts near zero.
    edge = xinput.LEFT_THUMB_DEADZONE + 330
    assert 0.0 < _state(sThumbLX=edge).lx < 0.02


class _Slots:
    """Stands in for `xinput.read`: slot -> readings, one per call, then None."""

    def __init__(self, slots: dict[int, list[XInputGamepad]]):
        self.slots = {k: list(v) for k, v in slots.items()}
        self.calls: list[int] = []

    def __call__(self, slot: int) -> XInputGamepad | None:
        self.calls.append(slot)
        readings = self.slots.get(slot)
        return readings.pop(0) if readings else None


def _on_windows(monkeypatch, slots: dict[int, list[XInputGamepad]], guide=True) -> _Slots:
    fake = _Slots(slots)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(xinput, "read", fake)
    monkeypatch.setattr(xinput, "_api", lambda: xinput._Api("xinput1_4", None, guide))
    return fake


def test_windows_opens_the_pad_through_xinput(monkeypatch) -> None:
    """Before this reader existed `open()` returned None on Windows and
    `describe()` blamed the pairing -- a working pad read as no pad."""
    pushed = XInputGamepad(sThumbLX=32767)
    many = [pushed] * 10
    fake = _on_windows(monkeypatch, {1: many})

    assert [p.path for p in controller.find_pads()] == ["xinput:1"]
    assert controller.available()
    assert "xinput:1" in controller.describe()

    pad = controller.open()
    assert isinstance(pad, controller.XInputPad) and pad.connected
    assert "player 2" in pad.name
    fake.calls.clear()
    assert pad.state().lx == pytest.approx(1.0)
    assert fake.calls == [1], "the pad in slot 1 must be the one read"
    assert pad.raw()[(EV_ABS, ABS_X)] == 32767


def test_windows_with_no_pad_says_so(monkeypatch) -> None:
    _on_windows(monkeypatch, {})
    assert controller.open() is None
    assert not controller.available()
    assert "no XInput controller" in controller.describe()


def test_a_dll_without_the_ex_call_says_home_is_unreadable(monkeypatch) -> None:
    _on_windows(monkeypatch, {0: [XInputGamepad()] * 3}, guide=False)
    assert "`home` never reads as pressed" in controller.describe()


def test_a_pad_that_goes_away_keeps_its_last_reading_and_is_not_polled(monkeypatch) -> None:
    """The same contract as the Linux `Gamepad`: `connected` goes False and the
    last snapshot stays. And the empty slot is left alone afterwards, which
    Microsoft's own guidance asks for: polling an empty slot every frame is the
    expensive case of `XInputGetState`."""
    readings = [XInputGamepad(bRightTrigger=255)] * 3  # find_pads, start, one state()
    fake = _on_windows(monkeypatch, {0: readings})
    pad = controller.open()
    assert pad.state().rt == pytest.approx(1.0)

    assert pad.state().rt == pytest.approx(1.0), "the last reading is kept"
    assert not pad.connected
    fake.calls.clear()
    pad.state()
    pad.raw()
    assert fake.calls == [], "a pad that went away must not be polled every step"


def test_macos_names_the_platform_rather_than_the_pairing(monkeypatch) -> None:
    """Not "nothing is paired": on a platform with no reader, pairing is not what
    is wrong."""
    monkeypatch.setattr(sys, "platform", "darwin")
    assert controller.open() is None
    assert controller.find_pads() == []
    assert "darwin" in controller.describe()
