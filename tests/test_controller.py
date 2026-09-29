"""The controller package, and what a stick means for this robot.

No hardware. Everything here is either pure arithmetic on synthetic evdev values
or a check that the absence of a pad is handled -- which is the state this machine
is in, and the one the code has to be right about first: a paired controller is a
nice day, an unplugged one is every other day.

The one thing these cannot cover is a real pad's codes matching `xbox.py`. That is
what `controller.Gamepad.raw()` is for, and it is called out in the docs rather
than pretended about here.
"""

from __future__ import annotations

import struct

import pytest

from controller.xbox import (
    ABS_RX,
    ABS_RZ,
    ABS_X,
    ABS_Y,
    BTN_EAST,
    BTN_SOUTH,
    EV_ABS,
    EV_KEY,
    GamepadState,
    normalise,
)

#: What an Xbox pad reports: sticks signed 16-bit, triggers 0..1023, with the
#: driver's own dead zone in `flat`.
STICK = (-32768, 32767, 128)
TRIGGER = (0, 1023, 0)
RANGES = {ABS_X: STICK, ABS_Y: STICK, ABS_RX: STICK, ABS_RZ: TRIGGER}


def test_the_event_struct_is_the_size_the_kernel_says() -> None:
    """24 bytes on 64-bit Linux. A mismatch would not raise -- it would decode
    the stream misaligned and report button presses that never happened."""
    assert struct.calcsize("llHHi") == 24


def test_a_centred_stick_is_zero() -> None:
    """Not merely small: a resting pad must produce exactly 0.0, or a policy is
    quietly commanded to creep whenever nobody is touching anything."""
    mid = (STICK[0] + STICK[1]) // 2
    s = normalise({(EV_ABS, ABS_X): mid, (EV_ABS, ABS_Y): mid}, RANGES)
    assert s.lx == 0.0 and s.ly == 0.0
    assert not s.any_input


def test_full_deflection_is_one() -> None:
    s = normalise({(EV_ABS, ABS_X): 32767, (EV_ABS, ABS_Y): -32768}, RANGES)
    assert s.lx == pytest.approx(1.0, abs=1e-3)
    assert s.ly == pytest.approx(-1.0, abs=1e-3)


def test_the_dead_zone_is_rescaled_not_just_clipped() -> None:
    """Just past the dead zone the output must start from ~0 and rise smoothly.

    Clipping alone leaves a step: the axis reads 0 up to the edge of the zone and
    then jumps to whatever fraction the zone occupied. On a stick that is a
    control which does nothing and then lurches.
    """
    span = 32767.0
    flat = STICK[2]
    just_outside = int(flat + 0.01 * span)
    s = normalise({(EV_ABS, ABS_X): just_outside}, RANGES)
    assert 0.0 < s.lx < 0.02, f"expected a small value just past the zone, got {s.lx}"

    inside = normalise({(EV_ABS, ABS_X): flat - 1}, RANGES)
    assert inside.lx == 0.0


def test_a_trigger_runs_zero_to_one() -> None:
    assert normalise({(EV_ABS, ABS_RZ): 0}, RANGES).rt == 0.0
    assert normalise({(EV_ABS, ABS_RZ): 1023}, RANGES).rt == pytest.approx(1.0)


def test_buttons_are_named_and_readable_as_attributes() -> None:
    # The dictionary's names, not a spelling of this package's own: aligning a
    # pad to those names is the whole of what this package does.
    s = normalise({(EV_KEY, BTN_SOUTH): 1, (EV_KEY, BTN_EAST): 0}, RANGES)
    assert s.A is True
    assert s.B is False
    with pytest.raises(AttributeError):
        _ = s.a
    assert s.any_input
    with pytest.raises(AttributeError):
        _ = s.nonexistent_button


def test_an_axis_the_pad_does_not_have_reads_zero() -> None:
    """A pad missing an axis must not make it up: no entry in `ranges` means the
    device never reported one, and inventing a midpoint would be a phantom input."""
    s = normalise({(EV_ABS, ABS_X): 32767}, {})
    assert s.lx == 0.0


# ── The two Xbox layouts ────────────────────────────────────────────────
#
# Measured on an Xbox Wireless Controller over Bluetooth: it reports
# ABS_X ABS_Y ABS_Z ABS_RZ ABS_GAS ABS_BRAKE ABS_HAT0X ABS_HAT0Y, sticks 0..65535
# and triggers 0..1023, with **no ABS_RX at all**. The first version of this code
# assumed the USB layout and would have read the right stick as the left trigger.

BT_STICK = (0, 65535, 4095)
BT_TRIGGER = (0, 1023, 63)
BT_RANGES = {
    0x00: BT_STICK, 0x01: BT_STICK, 0x02: BT_STICK, 0x05: BT_STICK,
    0x09: BT_TRIGGER, 0x0A: BT_TRIGGER, 0x10: (-1, 1, 0), 0x11: (-1, 1, 0),
}
USB_RANGES = {
    0x00: STICK, 0x01: STICK, 0x03: STICK, 0x04: STICK,
    0x02: TRIGGER, 0x05: TRIGGER,
}


def test_the_layout_is_taken_from_the_axes_the_pad_reports() -> None:
    from controller.xbox import layout_of

    assert layout_of(BT_RANGES).name == "bluetooth"
    assert layout_of(USB_RANGES).name == "usb"
    # A pad reporting neither discriminator keeps the old behaviour rather than
    # reading its sticks as triggers.
    assert layout_of({0x00: STICK, 0x01: STICK}).name == "usb"


def test_the_bluetooth_right_stick_is_not_read_as_a_trigger() -> None:
    """The exact bug real hardware found.

    On Bluetooth, ABS_Z **is** the right stick. Read with the USB layout it is the
    left trigger, so pushing the right stick would have squeezed a trigger that was
    never touched -- silent, and wrong in a way that only shows with a pad in hand.
    """
    full_right = {(EV_ABS, 0x02): 65535}
    s = normalise(full_right, BT_RANGES)
    assert s.rx == pytest.approx(1.0, abs=1e-3), "ABS_Z must drive the right stick"
    assert s.lt == 0.0, "and must not appear on a trigger"


def test_the_bluetooth_triggers_are_the_pedal_codes() -> None:
    s = normalise({(EV_ABS, 0x09): 1023, (EV_ABS, 0x0A): 1023}, BT_RANGES)
    assert s.rt == pytest.approx(1.0) and s.lt == pytest.approx(1.0)


def test_unsigned_sticks_centre_at_zero() -> None:
    """The Bluetooth sticks run 0..65535, so the rest position is 32767, not 0.
    Assuming a signed axis would leave a resting pad commanding full forward."""
    s = normalise({(EV_ABS, 0x00): 32767, (EV_ABS, 0x01): 32767}, BT_RANGES)
    assert s.lx == 0.0 and s.ly == 0.0
    assert not s.any_input


def test_no_controller_is_not_an_error() -> None:
    """The normal case on a machine with nothing plugged in, and the one that must
    not raise: `play` has to work identically without a pad."""
    import controller

    assert controller.open() is None or controller.available()
    assert isinstance(controller.describe(), str)
    assert controller.describe(), "describe() must say something, even when empty"


# ── What a stick means for this robot ───────────────────────────────────


def _controls():
    """The specification the signs now come from. Any locomotion task's copy;
    `tests/test_controls.py` is what keeps the four identical."""
    from pathlib import Path

    from tasks.jumper.common.mdp.controls import load_controls

    return load_controls(Path(__file__).resolve().parents[1] / "tasks/jumper/tripod/controls.yaml")


#: `(lo, hi)` for vx, vy, wz: symmetric, so a sign is the only thing a test
#: below can get wrong.
_RANGES = ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5))


def _command(state: GamepadState, ranges=_RANGES) -> list[float]:
    """What `play`'s operator commands for one pad frame, through the file."""
    from tasks.jumper.common.mdp.operator import Operator

    class Pad:
        name = "test pad"

        def state(self):
            return state

    controls = _controls()
    op = Operator(controls, Pad(), listen=False)
    names = [a.name for a in controls.command("twist").axes]
    op.attach("twist", {n: (lo, hi, 0.0) for n, (lo, hi) in zip(names, ranges)})
    return op.command("twist")


def test_pushing_the_stick_forward_walks_forward() -> None:
    """evdev's Y is positive **down**, so the sign here is the whole point.

    Getting it backwards produces a robot that reverses when told to advance --
    obvious with a pad in hand and invisible in a config, which is exactly why it
    is worth a test rather than a comment.
    """
    vx, vy, wz = _command(GamepadState(ly=-1.0))
    assert vx == pytest.approx(0.5) and vy == 0.0 and wz == 0.0

    vx, _, _ = _command(GamepadState(ly=1.0))
    assert vx == pytest.approx(-0.5)


def test_stick_left_is_positive_y_and_stick_right_turns_clockwise() -> None:
    _, vy, _ = _command(GamepadState(lx=-1.0))
    assert vy == pytest.approx(0.5), "+y is left in the body frame"

    _, _, wz = _command(GamepadState(rx=1.0))
    assert wz == pytest.approx(-0.5), "yaw is positive counter-clockwise"


def test_full_deflection_stays_inside_the_trained_range() -> None:
    """The stick means "as fast as this policy was ever asked to go", so it cannot
    reach past the range the policy trained on."""
    for state in (
        GamepadState(lx=1.0, ly=1.0, rx=1.0),
        GamepadState(lx=-1.0, ly=-1.0, rx=-1.0),
    ):
        for v in _command(state):
            assert abs(v) <= 0.5 + 1e-9


def test_an_asymmetric_range_scales_each_end_by_its_own_limit() -> None:
    """A range of (-0.2, 1.0) must give 1.0 forward and 0.2 back, not 1.0 both
    ways -- the reverse limit is there because the robot was never trained to run
    backwards as fast as it runs forwards."""
    asym = ((-0.2, 1.0), (-0.5, 0.5), (-0.5, 0.5))
    assert _command(GamepadState(ly=-1.0), asym)[0] == pytest.approx(1.0)
    assert _command(GamepadState(ly=1.0), asym)[0] == pytest.approx(-0.2)
