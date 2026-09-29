"""Finding and reading a Linux evdev gamepad, with no dependencies.

## Why evdev and not joydev

`/dev/input/js*` is simpler to parse, and on this machine it does not exist: the
`joydev` module is not loaded, while `event*` is always there. evdev is also the
one that carries the axis ranges, so a trigger that reports 0..1023 on one pad and
0..255 on another normalises correctly without a table of models.

## The wire format

An evdev event is `struct input_event`: a `timeval` then type, code and value.
On 64-bit Linux that is `llHHi`, 24 bytes, which is asserted at import rather
than assumed -- a wrong size here would not raise, it would silently decode
garbage as button presses.

Ranges come from the `EVIOCGABS` ioctl, one per axis, read once at open. The ioctl
numbers are built with the same arithmetic as the kernel's `_IOR` macro rather
than pasted as magic constants, so the derivation is visible.
"""

from __future__ import annotations

import array
import glob
import os
import select
import struct
import sys
import threading
from dataclasses import dataclass, field

try:
    import fcntl
except ImportError:  # Windows. There is no evdev to read there and no ioctl
    # to read it with; pads there go through `xinput.py`, and nothing in this
    # module is called. The import still happens, so it must not take the whole
    # package down with it: the tasks import `controller` for its vocabulary, and
    # that is on the path of every training run.
    fcntl = None

from .xbox import ABS_AXES, GamepadState, normalise

__all__ = ["Gamepad", "GamepadState", "PadInfo", "describe", "find_pads"]

#: `struct input_event` on 64-bit Linux: struct timeval (two longs), then
#: __u16 type, __u16 code, __s32 value.
_EVENT_FORMAT = "llHHi"
_EVENT_SIZE = struct.calcsize(_EVENT_FORMAT)
# Asserted rather than assumed, because a wrong size here would not raise -- it
# would decode the stream misaligned and report button presses that never
# happened. **Only on Linux, which is what the 24 bytes describe**: Windows'
# C `long` is 4 bytes, so the same format measures 16 there and would fail an
# assertion about a layout that platform never uses.
if sys.platform == "linux":
    assert _EVENT_SIZE == 24, (
        f"struct input_event is {_EVENT_SIZE} bytes here, not 24. Decoding would "
        f"silently produce nonsense rather than fail, so this is checked."
    )

EV_KEY = 0x01
EV_ABS = 0x03


def _ioc(direction: int, type_: int, nr: int, size: int) -> int:
    """The kernel's `_IOC` macro. dir 2 is `_IOC_READ`."""
    return (direction << 30) | (size << 16) | (type_ << 8) | nr


def _eviocgname(length: int) -> int:
    return _ioc(2, ord("E"), 0x06, length)


def _eviocgabs(axis: int) -> int:
    #: `struct input_absinfo` is six __s32: value, min, max, fuzz, flat, resolution.
    return _ioc(2, ord("E"), 0x40 + axis, 24)


@dataclass
class PadInfo:
    """A candidate device and what was decided about it."""

    path: str
    name: str
    readable: bool
    reason: str = ""


def _read_name(fd: int) -> str:
    buf = array.array("B", b"\x00" * 256)
    try:
        fcntl.ioctl(fd, _eviocgname(256), buf)
    except OSError:
        return ""
    return bytes(buf).split(b"\x00", 1)[0].decode("utf-8", "replace")


def _read_ranges(fd: int) -> dict[int, tuple[int, int, int]]:
    """Per-axis (min, max, flat) from EVIOCGABS. `flat` is the driver's own idea
    of the dead zone, which is better than a number invented here."""
    ranges: dict[int, tuple[int, int, int]] = {}
    for axis in ABS_AXES:
        buf = array.array("i", [0] * 6)
        try:
            fcntl.ioctl(fd, _eviocgabs(axis), buf)
        except OSError:
            continue  # the pad does not have this axis
        _value, lo, hi, _fuzz, flat, _res = buf
        if hi > lo:
            ranges[axis] = (lo, hi, flat)
    return ranges


#: Buttons whose presence marks a device as something to be played with, from
#: udev's own `input_id` rule.
BTN_JOYSTICK = 0x120
BTN_GAMEPAD = 0x130


def _bitmap(path: str) -> set[int]:
    """Parse a sysfs capability bitmap into the set of bits it has.

    The kernel prints these as space-separated 64-bit hex words, **most
    significant first**, so the last word is bits 0-63. Reading them left to
    right, which is the obvious way, gets every bit index wrong.
    """
    try:
        with open(path) as f:
            words = f.read().split()
    except OSError:
        return set()
    bits: set[int] = set()
    for i, word in enumerate(reversed(words)):
        value = int(word, 16)
        base = i * 64
        while value:
            low = value & -value
            bits.add(base + low.bit_length() - 1)
            value ^= low
    return bits


def find_pads(readable_only: bool = False) -> list[PadInfo]:
    """Every gamepad-looking device, in a stable order.

    Discovery is through `/sys/class/input/event*`, applying udev's own rule --
    the device has absolute axes and at least one gamepad or joystick button.

    **Not through `/dev/input/by-id/*-event-joystick`**, which is what this did
    first and which silently finds nothing over Bluetooth: those symlinks are
    generated from USB and serial identifiers, and a Bluetooth pad has neither.
    Measured here, an Xbox Wireless Controller paired over Bluetooth appears as
    `/dev/input/event17` with no `by-id` entry at all, so a working, connected
    controller read as absent.

    sysfs also answers without opening anything, so a pad the process cannot read
    is still *found*, and can be reported as a permissions problem rather than as
    nothing at all.
    """
    pads: list[PadInfo] = []
    for sysdir in sorted(
        glob.glob("/sys/class/input/event*"),
        key=lambda p: int(os.path.basename(p)[5:] or 0),
    ):
        device = os.path.join(sysdir, "device")
        caps = os.path.join(device, "capabilities")
        ev = _bitmap(os.path.join(caps, "ev"))
        keys = _bitmap(os.path.join(caps, "key"))
        if EV_ABS not in ev:
            continue
        if not (BTN_GAMEPAD in keys or BTN_JOYSTICK in keys):
            continue

        path = os.path.join("/dev/input", os.path.basename(sysdir))
        try:
            with open(os.path.join(device, "name")) as f:
                name = f.read().strip()
        except OSError:
            name = os.path.basename(path)

        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except PermissionError:
            pads.append(
                PadInfo(path, name, False,
                        "permission denied -- add yourself to the 'input' group "
                        "(sudo usermod -aG input \"$USER\", then log out and back "
                        "in), or log in on the seat so logind grants the ACL")
            )
            continue
        except OSError as exc:
            pads.append(PadInfo(path, name, False, str(exc)))
            continue
        try:
            pads.append(PadInfo(path, _read_name(fd) or name, True))
        finally:
            os.close(fd)
    if readable_only:
        return [p for p in pads if p.readable]
    return pads


def describe() -> str:
    """What was found and, for anything rejected, why. `controller.describe()` is
    the entry point; this is its Linux half."""
    pads = find_pads()
    if not pads:
        return (
            "no input device has both absolute axes and a gamepad button. "
            "Nothing is paired, or it connected as something other than a pad."
        )
    lines = []
    for p in pads:
        state = "usable" if p.readable else f"unusable: {p.reason}"
        lines.append(f"  {p.path}  {p.name}  -- {state}")
    return "\n".join(lines)


@dataclass
class Gamepad:
    """A pad being read on a background thread.

    The reader thread only ever writes raw integers into `_raw`; normalisation
    happens in `state()`, on the caller's thread. That keeps the axis ranges and
    the dead zone out of the hot path and, more usefully, means a caller polling
    at 50 Hz sees a consistent snapshot rather than a half-updated one.
    """

    info: PadInfo
    _raw: dict[tuple[int, int], int] = field(default_factory=dict)
    _ranges: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    _fd: int = -1
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)

    @property
    def name(self) -> str:
        return self.info.name

    def start(self) -> None:
        self._fd = os.open(self.info.path, os.O_RDONLY | os.O_NONBLOCK)
        self._ranges = _read_ranges(self._fd)
        self._thread = threading.Thread(
            target=self._run, name="gamepad", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def _run(self) -> None:
        """Poll rather than block on read, so `close()` is not left waiting for a
        stick that nobody is touching."""
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if not ready:
                continue
            try:
                data = os.read(self._fd, _EVENT_SIZE * 64)
            except OSError:
                # The pad went away mid-read: unplugged, or Bluetooth dropped.
                # Leave the last state in place and stop; state() keeps working
                # and `connected` goes False.
                break
            for i in range(0, len(data) - _EVENT_SIZE + 1, _EVENT_SIZE):
                _s, _us, etype, code, value = struct.unpack_from(
                    _EVENT_FORMAT, data, i
                )
                if etype in (EV_KEY, EV_ABS):
                    self._raw[(etype, code)] = value
        self._stop.set()

    @property
    def connected(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def state(self) -> GamepadState:
        """A normalised snapshot. Never blocks and never raises."""
        return normalise(dict(self._raw), self._ranges)

    def raw(self) -> dict[tuple[int, int], int]:
        """Everything seen so far, unnormalised, as `{(type, code): value}`.

        For working out a pad whose mapping does not match `xbox.py` -- press the
        button, see which code moved.
        """
        return dict(self._raw)
