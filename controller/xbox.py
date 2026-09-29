"""The Xbox pad's evdev codes, and normalising them into a `GamepadState`.

The codes are the Linux `xpad` / `xone` drivers', which is what an Xbox controller
presents over both USB and Bluetooth on a modern kernel. They are kernel
constants, not invented here; the names are `linux/input-event-codes.h`.

## Sign conventions are the kernel's, not the robot's

evdev reports **`ABS_Y` positive when the stick is pushed down**, which is
backwards from every intuition about forward. It is left exactly as the kernel
gives it here, and flipped where the meaning is assigned, in the task. Correcting
it in this file would be a robot opinion smuggled into a device driver: a task
mapping the stick to something else, a camera or a gait frequency, would inherit
a flip that only makes sense for velocity.

## There are two Xbox layouts, and the pad has to be asked which

USB through `xpad` and Bluetooth through `hid-generic` do **not** agree, and the
disagreement is not cosmetic -- it puts the right stick where the other one keeps
a trigger:

    axis          USB (xpad)        Bluetooth (hid-generic)
    ABS_X, ABS_Y  left stick        left stick
    ABS_Z, ABS_RZ triggers          **right stick**
    ABS_RX, ABS_RY right stick      absent
    ABS_GAS, ABS_BRAKE  absent      triggers

Measured on an Xbox Wireless Controller over Bluetooth here: `ABS_X ABS_Y ABS_Z
ABS_RZ ABS_GAS ABS_BRAKE ABS_HAT0X ABS_HAT0Y`, with the sticks 0..65535 and the
triggers 0..1023. There is no `ABS_RX` on that device at all. Assuming the USB
layout would have read pushing the right stick as squeezing the left trigger --
which is not a crash, just a control that does the wrong thing.

`layout_of` picks between the two by **which axes the device actually reports**,
which is evidence from the device rather than a table of models and firmware
revisions that would need extending every time someone brings a different pad.

## Ranges come from the device

Sticks are -32768..32767 on one of those layouts and 0..65535 on the other, and
triggers are 0..1023 or 0..255. `normalise` is given the ranges read from
`EVIOCGABS` and uses them, so nothing here assumes a model -- the unsigned sticks
above work out because the midpoint is computed rather than assumed to be zero.
The dead zone is the driver's own `flat` for the same reason: a pad that reports
its own slop knows better than a constant would.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ABS_AXES", "BUTTONS", "GamepadState", "normalise"]

EV_KEY = 0x01
EV_ABS = 0x03

# ── Axes (EV_ABS) ────────────────────────────────────────────────────────
ABS_X = 0x00        # left stick, right positive       (both layouts)
ABS_Y = 0x01        # left stick, **down positive**    (both layouts)
ABS_Z = 0x02        # USB: left trigger   | Bluetooth: right stick X
ABS_RX = 0x03       # USB: right stick X  | Bluetooth: absent
ABS_RY = 0x04       # USB: right stick Y  | Bluetooth: absent
ABS_RZ = 0x05       # USB: right trigger  | Bluetooth: right stick Y
ABS_GAS = 0x09      # Bluetooth: right trigger
ABS_BRAKE = 0x0A    # Bluetooth: left trigger
ABS_HAT0X = 0x10    # d-pad, -1 / 0 / +1               (both layouts)
ABS_HAT0Y = 0x11

#: Every axis worth asking the device about, across both layouts. `device.py`
#: queries one range per entry and simply skips the ones the pad does not have,
#: which is also how `layout_of` learns which layout it is looking at.
ABS_AXES = (
    ABS_X, ABS_Y, ABS_Z, ABS_RX, ABS_RY, ABS_RZ,
    ABS_GAS, ABS_BRAKE, ABS_HAT0X, ABS_HAT0Y,
)


@dataclass(frozen=True)
class Layout:
    """Which code carries which control on this particular pad."""

    name: str
    rx: int
    ry: int
    lt: int
    rt: int


#: xpad over USB: the right stick is RX/RY and the triggers are the Z pair.
USB_LAYOUT = Layout("usb", rx=ABS_RX, ry=ABS_RY, lt=ABS_Z, rt=ABS_RZ)

#: hid-generic over Bluetooth: the Z pair *is* the right stick, and the triggers
#: moved to the pedal codes.
BLUETOOTH_LAYOUT = Layout("bluetooth", rx=ABS_Z, ry=ABS_RZ, lt=ABS_BRAKE, rt=ABS_GAS)


def layout_of(axes) -> Layout:
    """Pick a layout from the axes the device reports.

    `ABS_RX` is the discriminator because it exists on exactly one of the two.
    Falling back to USB when neither is conclusive keeps an unknown pad working as
    before rather than reading its sticks as triggers.
    """
    if ABS_RX in axes and ABS_RY in axes:
        return USB_LAYOUT
    if ABS_GAS in axes or ABS_BRAKE in axes:
        return BLUETOOTH_LAYOUT
    return USB_LAYOUT

# ── Buttons (EV_KEY) ─────────────────────────────────────────────────────
BTN_SOUTH = 0x130   # A
BTN_EAST = 0x131    # B
BTN_WEST = 0x134    # X   -- note the codes are not in ABXY order
BTN_NORTH = 0x133   # Y
BTN_TL = 0x136      # left bumper
BTN_TR = 0x137      # right bumper
BTN_SELECT = 0x13A  # view
BTN_START = 0x13B   # menu
BTN_MODE = 0x13C    # the Xbox button
BTN_THUMBL = 0x13D
BTN_THUMBR = 0x13E

#: Button code -> the name the **dictionary** gives it.
#:
#: This package's job is to align a pad to those names and stop there; what any
#: of them means belongs to whoever is driving, which is a task. So these are
#: not our own spelling any more -- they are `controller/vocabulary.json`'s, the
#: same names the robot's own pad service publishes and the same ones a
#: `[[fsm.button]]` or a `controls.yaml` may write.
#:
#: `view` is here and not in the dictionary: this pad reports it and the robot's
#: service deliberately does not. Reported rather than dropped, so `--raw` and a
#: person pressing buttons see the truth; `vocabulary.absent()` is where the
#: asymmetry is written down, and nothing downstream may bind it.
#:
#: **X and Y are unresolved.** The dictionary records it: we read 307 as `Y` and
#: 308 as `X`; the service's table reads them the other way round. One of us is
#: wrong about which physical button it is and pressing it settles it. Left
#: disagreeing rather than quietly picked.
BUTTONS = {
    BTN_SOUTH: "A", BTN_EAST: "B", BTN_WEST: "X", BTN_NORTH: "Y",
    BTN_TL: "LB", BTN_TR: "RB", BTN_SELECT: "view", BTN_START: "menu",
    BTN_MODE: "home", BTN_THUMBL: "L3", BTN_THUMBR: "R3",
}


@dataclass(frozen=True)
class GamepadState:
    """One snapshot. Sticks are -1..1, triggers 0..1, buttons bool.

    `ly` and `ry` keep the kernel's sign: **positive is down**.
    """

    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    lt: float = 0.0
    rt: float = 0.0
    hat_x: int = 0
    hat_y: int = 0
    buttons: frozenset[str] = field(default_factory=frozenset)

    def __getattr__(self, item: str) -> bool:
        """`state.A` for the buttons, so a caller does not have to remember
        whether a name is a field or a set member. The names are the
        dictionary's."""
        if item in BUTTONS.values():
            return item in self.buttons
        raise AttributeError(item)

    @property
    def any_input(self) -> bool:
        """Is the operator touching it? What "taking over" is decided on."""
        return (
            bool(self.buttons)
            or self.hat_x != 0
            or self.hat_y != 0
            or max(abs(self.lx), abs(self.ly), abs(self.rx), abs(self.ry)) > 0.0
            or max(self.lt, self.rt) > 0.0
        )


def _axis(
    raw: dict[tuple[int, int], int],
    ranges: dict[int, tuple[int, int, int]],
    code: int,
    bipolar: bool,
) -> float:
    """One axis, scaled to -1..1 (bipolar) or 0..1, with the dead zone removed.

    The dead zone is **rescaled, not just clipped**: an axis that starts moving at
    the edge of the zone should start from zero, not jump to the fraction the zone
    occupied. Clipping alone leaves a visible step at the point the stick starts to
    bite, which reads as a sticky control.
    """
    if code not in ranges:
        return 0.0
    value = raw.get((EV_ABS, code))
    if value is None:
        return 0.0
    lo, hi, flat = ranges[code]

    if bipolar:
        mid = (lo + hi) / 2.0
        span = (hi - lo) / 2.0
        if span <= 0:
            return 0.0
        x = (value - mid) / span
        dead = flat / span if span else 0.0
        if abs(x) <= dead:
            return 0.0
        x = (abs(x) - dead) / (1.0 - dead) * (1.0 if x > 0 else -1.0)
        return max(-1.0, min(1.0, x))

    span = hi - lo
    if span <= 0:
        return 0.0
    x = (value - lo) / span
    dead = flat / span
    if x <= dead:
        return 0.0
    return max(0.0, min(1.0, (x - dead) / (1.0 - dead)))


def normalise(
    raw: dict[tuple[int, int], int], ranges: dict[int, tuple[int, int, int]]
) -> GamepadState:
    """Raw evdev values plus the device's own ranges -> a `GamepadState`.

    The layout is decided from `ranges`, which holds exactly the axes the device
    answered `EVIOCGABS` for. See `layout_of`.
    """
    pressed = frozenset(
        name for code, name in BUTTONS.items() if raw.get((EV_KEY, code), 0)
    )
    layout = layout_of(ranges)
    return GamepadState(
        lx=_axis(raw, ranges, ABS_X, True),
        ly=_axis(raw, ranges, ABS_Y, True),
        rx=_axis(raw, ranges, layout.rx, True),
        ry=_axis(raw, ranges, layout.ry, True),
        lt=_axis(raw, ranges, layout.lt, False),
        rt=_axis(raw, ranges, layout.rt, False),
        hat_x=int(raw.get((EV_ABS, ABS_HAT0X), 0)),
        hat_y=int(raw.get((EV_ABS, ABS_HAT0Y), 0)),
        buttons=pressed,
    )
