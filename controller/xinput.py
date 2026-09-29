"""Reading an Xbox pad on Windows through XInput, with no dependencies.

## Why XInput

XInput is a plain C function in a DLL every Windows since 8 ships
(`xinput1_4.dll`), so `ctypes` reaches it and nothing has to be installed -- the
same trade `device.py` makes with evdev. The price is that it reads **only XInput
devices**: Xbox pads and the ones that present themselves as one. A PlayStation or
Switch pad is not one until Steam Input or DS4Windows sits in between.

## It speaks evdev, not a dialect of its own

Each reading is turned into the `{(type, code): value}` table the Linux `xpad`
driver produces for the same pad over USB, with the ranges `xpad` declares, and
then handed to `xbox.normalise` -- the function that reads a Linux pad. The sign
conventions, the dead-zone rescaling and the button names are therefore one piece
of code on both platforms rather than two that have to be kept agreeing, and
`python -m controller` is the same bench check on either.

## The Y axes are flipped here, and nowhere else

XInput reports a stick pushed **up** as positive; evdev, and everything above this
package, reads **down** as positive (`xbox.py` says why that is kept). The flip is
`~y`, which is what `xpad` itself does to the same 16-bit value: it maps
-32768..32767 onto itself exactly, where negation would leave the range at -32768.
Leaving it out would not raise -- it would walk the robot backwards when the stick
says forwards.

## The dead zone is Microsoft's

evdev hands over the driver's `flat`; XInput reports none. Microsoft documents one
instead, `XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE` and its siblings, because a stick does
not return to exactly centre, and that is what goes into `flat` here -- the
platform's own figure, which is the role the driver's `flat` plays on Linux. It is
wider than what a Linux pad reports (xpad's 128 of 32768 over USB; 4095 of 65535
from `hid-generic` over Bluetooth), so the stick travels further before it bites,
and is rescaled like every dead zone here so full deflection still reads 1.

## Buttons are matched by name

XInput's bits name physical buttons, so each is mapped onto the code `xbox.BUTTONS`
gives **that name**, not onto the code `xpad` would send. The physical X button is
`X` here whatever is eventually decided about evdev's 307 and 308, which
`controller/vocabulary.json` records as unresolved.

## The Xbox button needs the undocumented entry point

`XInputGetState` never reports it. Ordinal 100 of the same DLL, `XInputGetStateEx`,
does -- it is what SDL reads -- and it writes four bytes more than the documented
struct holds, so the struct below carries them. It is used when the DLL exports it
and `home` simply never goes down when it does not (`xinput9_1_0.dll`). Windows
also opens the Game Bar on that button unless the Game Bar is told not to.
"""

from __future__ import annotations

import ctypes
import functools
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

from .device import PadInfo
from .xbox import (
    ABS_HAT0X,
    ABS_HAT0Y,
    ABS_RX,
    ABS_RY,
    ABS_RZ,
    ABS_X,
    ABS_Y,
    ABS_Z,
    BUTTONS,
    EV_ABS,
    EV_KEY,
    GamepadState,
    normalise,
)

__all__ = ["RANGES", "XInputGamepad", "XInputPad", "describe", "find_pads", "read", "translate"]

#: `XUSER_MAX_COUNT`: XInput has four player slots and no more.
SLOTS = 4
ERROR_SUCCESS = 0

# ── The documented constants, from XInput.h ─────────────────────────────
DPAD_UP = 0x0001
DPAD_DOWN = 0x0002
DPAD_LEFT = 0x0004
DPAD_RIGHT = 0x0008
LEFT_THUMB_DEADZONE = 7849
RIGHT_THUMB_DEADZONE = 8689
TRIGGER_THRESHOLD = 30

#: XInput button bit -> the dictionary's name for that physical button. 0x0400 is
#: the Xbox button, which only `XInputGetStateEx` sets.
_BUTTON_NAMES = {
    0x1000: "A", 0x2000: "B", 0x4000: "X", 0x8000: "Y",
    0x0100: "LB", 0x0200: "RB", 0x0020: "view", 0x0010: "menu",
    0x0400: "home", 0x0040: "L3", 0x0080: "R3",
}
_CODE_OF = {name: code for code, name in BUTTONS.items()}
#: XInput button bit -> evdev key code, by way of the name. A name `xbox.BUTTONS`
#: does not know raises here, at import, rather than reading as never pressed.
BUTTON_CODES = {bit: _CODE_OF[name] for bit, name in _BUTTON_NAMES.items()}

#: What `xpad` declares for these axes over USB, with Microsoft's dead zones as the
#: `flat`. Having `ABS_RX`/`ABS_RY` is also what makes `xbox.layout_of` choose the
#: USB layout, so the triggers are read from `ABS_Z`/`ABS_RZ` and not the sticks.
RANGES: dict[int, tuple[int, int, int]] = {
    ABS_X: (-32768, 32767, LEFT_THUMB_DEADZONE),
    ABS_Y: (-32768, 32767, LEFT_THUMB_DEADZONE),
    ABS_RX: (-32768, 32767, RIGHT_THUMB_DEADZONE),
    ABS_RY: (-32768, 32767, RIGHT_THUMB_DEADZONE),
    ABS_Z: (0, 255, TRIGGER_THRESHOLD),
    ABS_RZ: (0, 255, TRIGGER_THRESHOLD),
    ABS_HAT0X: (-1, 1, 0),
    ABS_HAT0Y: (-1, 1, 0),
}


class XInputGamepad(ctypes.Structure):
    """`XINPUT_GAMEPAD`, plus the trailing DWORD `XInputGetStateEx` writes.

    Fixed-width types rather than `ctypes.wintypes`, so the struct -- and
    `translate` with it -- exists on every platform and is tested on Linux.
    """

    _fields_ = [
        ("wButtons", ctypes.c_uint16),
        ("bLeftTrigger", ctypes.c_uint8),
        ("bRightTrigger", ctypes.c_uint8),
        ("sThumbLX", ctypes.c_int16),
        ("sThumbLY", ctypes.c_int16),
        ("sThumbRX", ctypes.c_int16),
        ("sThumbRY", ctypes.c_int16),
        ("dwPaddingReserved", ctypes.c_uint32),
    ]


class XInputState(ctypes.Structure):
    """`XINPUT_STATE` as `XInputGetStateEx` fills it."""

    _fields_ = [("dwPacketNumber", ctypes.c_uint32), ("Gamepad", XInputGamepad)]


# Asserted rather than assumed: a buffer four bytes short of what the Ex call
# writes does not raise, it overwrites whatever ctypes allocated next to it.
assert ctypes.sizeof(XInputGamepad) == 16 and ctypes.sizeof(XInputState) == 20, (
    f"XINPUT_STATE_EX is {ctypes.sizeof(XInputState)} bytes here, not 20"
)


def translate(pad: XInputGamepad) -> dict[tuple[int, int], int]:
    """One XInput reading -> the raw evdev table `xpad` would have produced."""
    buttons = pad.wButtons
    raw = {(EV_KEY, code): int(bool(buttons & bit)) for bit, code in BUTTON_CODES.items()}
    raw[(EV_ABS, ABS_X)] = pad.sThumbLX
    raw[(EV_ABS, ABS_Y)] = ~pad.sThumbLY  # up is positive in XInput, down in evdev
    raw[(EV_ABS, ABS_RX)] = pad.sThumbRX
    raw[(EV_ABS, ABS_RY)] = ~pad.sThumbRY
    raw[(EV_ABS, ABS_Z)] = pad.bLeftTrigger
    raw[(EV_ABS, ABS_RZ)] = pad.bRightTrigger
    raw[(EV_ABS, ABS_HAT0X)] = bool(buttons & DPAD_RIGHT) - bool(buttons & DPAD_LEFT)
    raw[(EV_ABS, ABS_HAT0Y)] = bool(buttons & DPAD_DOWN) - bool(buttons & DPAD_UP)
    return raw


class _Api(NamedTuple):
    dll: str
    get_state: Callable[[int, object], int]
    #: Whether `get_state` is `XInputGetStateEx`, i.e. whether `home` can be read.
    guide: bool


#: Newest first. 1_4 ships with Windows 8 and later; 1_3 comes with the DirectX
#: runtime; 9_1_0 is Windows 7's and has no ordinal 100.
_DLLS = ("xinput1_4", "xinput1_3", "xinput9_1_0")


@functools.lru_cache(maxsize=1)
def _api() -> _Api | None:
    if sys.platform != "win32":
        return None
    for name in _DLLS:
        try:
            dll = ctypes.WinDLL(name)
        except OSError:
            continue
        try:
            fn, guide = dll[100], True
        except AttributeError:
            fn, guide = dll.XInputGetState, False
        fn.argtypes = (ctypes.c_uint32, ctypes.POINTER(XInputState))
        fn.restype = ctypes.c_uint32
        return _Api(name, fn, guide)
    return None


def read(slot: int) -> XInputGamepad | None:
    """The pad in `slot` now, or None if there is none (or no XInput at all)."""
    api = _api()
    if api is None:
        return None
    state = XInputState()
    if api.get_state(slot, ctypes.byref(state)) != ERROR_SUCCESS:
        return None
    return state.Gamepad


def find_pads(readable_only: bool = False) -> list[PadInfo]:
    """Every slot with a pad in it. XInput has no permissions to be refused, so a
    pad that answers is one this process can read, and `readable_only` changes
    nothing."""
    del readable_only
    return [
        PadInfo(f"xinput:{slot}", f"XInput pad, player {slot + 1}", True)
        for slot in range(SLOTS)
        if read(slot) is not None
    ]


def describe() -> str:
    """What was found, or why nothing was."""
    api = _api()
    if api is None:
        return (
            "XInput is not available: none of "
            + ", ".join(f"{name}.dll" for name in _DLLS)
            + " could be loaded."
        )
    pads = find_pads()
    if not pads:
        return (
            "no XInput controller is connected. Windows pads are read through XInput, "
            "which sees Xbox pads and pads that present themselves as one; a "
            "PlayStation or Switch pad needs Steam Input or DS4Windows in between."
        )
    lines = [f"  {p.path}  {p.name}  -- usable, through {api.dll}" for p in pads]
    if not api.guide:
        lines.append(f"  {api.dll} has no XInputGetStateEx, so `home` never reads as pressed")
    return "\n".join(lines)


@dataclass
class XInputPad:
    """An XInput slot, presenting the same surface as `device.Gamepad`.

    Read on the caller's thread, once per `state()`: XInput is a poll API with
    nothing to wait on, so a reader thread would only add latency -- up to the
    15.6 ms system timer tick on Python 3.10, whose sleeps on Windows round up to
    it. When the pad goes away the last reading is kept and `connected` goes False,
    as `Gamepad` does, and the empty slot is not polled again: an empty slot is the
    slow case of `XInputGetState`, which Microsoft's documentation of it says not to
    poll every frame.
    """

    info: PadInfo
    _raw: dict[tuple[int, int], int] = field(default_factory=dict)
    _ranges: dict[int, tuple[int, int, int]] = field(default_factory=lambda: dict(RANGES))
    _alive: bool = False

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def _slot(self) -> int:
        return int(self.info.path.rpartition(":")[2])

    def start(self) -> None:
        self._alive = True
        self._poll()

    def close(self) -> None:
        self._alive = False

    @property
    def connected(self) -> bool:
        return self._alive

    def _poll(self) -> None:
        if not self._alive:
            return
        pad = read(self._slot)
        if pad is None:
            self._alive = False
            return
        self._raw = translate(pad)

    def state(self) -> GamepadState:
        """A normalised snapshot. Never raises."""
        self._poll()
        return normalise(dict(self._raw), self._ranges)

    def raw(self) -> dict[tuple[int, int], int]:
        """The reading as evdev codes, for `python -m controller --raw`."""
        self._poll()
        return dict(self._raw)
