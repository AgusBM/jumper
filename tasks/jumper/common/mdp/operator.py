"""One operator, two devices, every command a task lets a person drive.

The consumer of a task's `controls.yaml` (see `controls.py`) in `play`. The
deployed controller is its transcription, `deploy/fsm/src/operator.rs`, and the
two are held to the same cases -- a disagreement between them is a robot that
answers the pad differently on the bench than in `play`.

## Two paths

**The pad** goes through `deflections`: each stick's travel, its split, its
layer and its sign, exactly as the file's `gamepad.axes` says. **The keyboard**
goes through `Keyboard`: each keystroke bound to one direction of one axis,
pushing it further the longer it is held -- full after the file's
`full_after_s` -- and back at rest the moment it is let go. Neither reads the
other's bindings. Until 2026-09-29 the keyboard was a virtual pad, each key a
stick direction read through the pad's mapping; that made the two agree by
construction, and it could not lay the keys out for hands -- J could not turn
while Shift + J twisted, and Shift, Alt and Ctrl could not hold an arm out.
They agree now because the file names the same axes for both, and the loader
refuses an axis either one leaves out without saying why.

## J and Shift + J

A keystroke is a key, a modifier on its own (`shift`, either Shift key), or a
modifier held with a key (`shift+key_j`). A chord is live while both are down;
a key alone is live while it is down and no modifier is that this file binds in
front of it -- so with Shift held, J is Shift + J and not J as well, and the
two never act together. Each binding remembers when it last became live, and
its ramp starts there: J held and Shift pressed stops the turn and starts the
twist from nothing, and Shift let go again starts the turn over.

## One operator per environment

Both command terms of an environment talk to the same `Operator`, found through
`Operator.for_env`. So there is one take-over, one release and one shift state:
`B` hands both commands back at once, and `R3` cannot bring its layer in for one
term and not the other. The pad is opened once, by whichever term is built first.

## Which device drives

Both are live, and the one touched last drives -- the commands and the task's
own controls alike. Touching the pad (a stick off centre, a button, the d-pad)
also drops the keyboard's holds: a key still down drives nothing until it is
pressed again, so letting go of the pad afterwards leaves the robot at the
neutral command rather than reviving a key held from before. A key going down
that the file binds takes back from an idle pad; a pad being held wins again on
the next step.

## An axis that moves rather than is placed

An axis with `integrate_s` -- jumper.posture's height -- is not where the pad
puts it: the operator keeps a position for it, from -1 to 1 of the way from its
rest to an end, and every step the pad drives moves it by the pad's deflection
times the step's seconds over `integrate_s`. Let go, it stays. The keyboard
places it as it places every other axis (Control-agent 3.1: N and M are the
high and the low stance, back to standing when let go); the pad's position is
kept while the keys drive and is the height again when the pad does. It goes back to
its rest on the release and on a tap of the file's `reset` button -- the button
pressed and let go with no stick it could have been reaching for moving in
between, so that on jumper.posture's R3, which is also the shift, a hold with
the stick pushed is the height and a tap is back to standing. The clock is the
operator's own, a person's time, advanced once a step in `_advance`.

## What the task keeps for itself

A file's `task:` controls drive no command. Each comes out of `task_control`,
by the task's own name for it, for the task's code to answer --
`jumper.five_foot`'s claw, on its trigger or on Space. An `amount` is how far it
is held (a trigger's travel, or a key's ramp), a `press` 1 while it is held and
0 otherwise. Read once per step like everything else, by whichever asks first.

## Inactive until touched

Until the first input the sampled command passes through untouched, so
`play.py --headless` and scripted audits behave as they always did -- a teleop
that defaulted to "commanded zero" would leave the robot standing still with no
indication why. The release hands everything back to the sampler, drops the
keyboard's holds, turns the shift off and puts every moving axis back at its
rest, so the next take-over starts from a known state.
"""

from __future__ import annotations

import math
import threading
import time
import weakref
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjrl.viewer import keys

from controller import vocabulary as _vocabulary

from .controls import _DPAD, _STICK_FIELD, Command, Controls, KeyAxis, TaskControl, stroke_name

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "Keyboard",
    "Operator",
    "OperatorVelocityCommand",
    "OperatorVelocityCommandCfg",
    "connect",
    "deflections",
    "scale",
    "spans",
]

#: How a command axis is shown on a printed line. Only the three velocity axes
#: have a shorter spelling people already read; the rest print as named.
_SHORT = {"lin_vel_x": "vx", "lin_vel_y": "vy", "ang_vel_z": "wz"}

#: GLFW keycode -> the dictionary's name for it, every key the dictionary has,
#: the modifiers' own keys included: what `mjrl.viewer.keys` hands a handler.
_GLFW = {codes["glfw"]: name for name, codes in _vocabulary.key_codes().items()}


# ── The pad's mapping ───────────────────────────────────────────────────────


def _band(s: float, travel: tuple[float, ...]) -> float:
    """A signed deflection through one binding's stretch of travel.

    Zero until `from`, full at `to` and held there beyond it. So `(0.0, 0.5)`
    is full at half travel, and `(0.5, 1.0)` is silent until half travel. With
    a fall, `(0.0, 0.5, 0.5, 0.75)`, it comes back from full at `back` to zero
    at `off` and stays there.
    """
    lo, hi, *fall = travel
    a = abs(s)
    x = min(max((a - lo) / (hi - lo), 0.0), 1.0)
    if fall:
        back, off = fall
        x *= min(max((off - a) / (off - back), 0.0), 1.0)
    return x if s >= 0.0 else -x


def deflections(
    values: Mapping[str, float], shifted: bool, controls: Controls
) -> dict[str, float]:
    """Pad control positions -> each command axis's deflection, -1..1.

    `values` is keyed by the dictionary's names (`Lx` ... `RT`). A source that
    has a shifted binding drives **only** that binding while shifted; every
    other source carries on as it was. The pad's path only: the keyboard has
    bindings of its own (`Keyboard`).
    """
    layered = {b.source for parts in controls.bindings.values() for b in parts if b.shifted}
    out = {}
    for axis, parts in controls.bindings.items():
        x = 0.0
        for b in parts:
            live = shifted if b.shifted else not (shifted and b.source in layered)
            if live:
                x += _band(b.sign * values[b.source], b.travel)
        out[axis] = max(-1.0, min(1.0, x))
    return out


def scale(x: float, span: tuple[float, float, float]) -> float:
    """`full_deflection_is_range_edge`, measured from the axis's rest.

    The two ends are scaled separately. A range like (-0.5, 1.0) gives forward
    twice what it gives back, which is what the policy trained on; taking the
    larger end for both quietly commands something it never saw.
    """
    lo, hi, rest = span
    return rest + x * ((hi - rest) if x >= 0.0 else (rest - lo))


def spans(command: Command, cfg: Any) -> dict[str, tuple[float, float, float]]:
    """`(low, high, rest)` per axis, from a live command config.

    A range is either a `(low, high)` pair, as the velocity command has, or a
    half-range, as the posture command's angles are. The rest is the config
    field the file names, or zero.
    """
    out = {}
    for a in command.axes:
        r = getattr(cfg.ranges, a.name)
        lo, hi = (-float(r), float(r)) if isinstance(r, (int, float)) else map(float, r)
        rest = float(getattr(cfg, a.rest)) if a.rest else 0.0
        if not lo <= rest <= hi:
            raise ValueError(
                f"{command.term}.{a.name} rests at {rest:g}, outside its range "
                f"({lo:g}, {hi:g}). A centred stick would command a value the range "
                "clamps away -- name the rest in the controls file (`rest: <field>`)."
            )
        out[a.name] = (lo, hi, rest)
    return out


# ── The keyboard's path ─────────────────────────────────────────────────────


class Keyboard:
    """The keyboard's half of the operator: which keystrokes are live, since when.

    Fed a key's two edges by the dictionary's name, and asked what an axis or a
    task control reads. A keystroke held for `full_after_s` is full deflection
    and half that is half; the moment it stops being live it reads nothing.
    Several live on one direction are summed, as two keys on one direction of a
    stick would push it together, and the two directions of an axis cancel.
    """

    def __init__(self, controls: Controls):
        self._full = float(controls.full_after_s)
        modifiers = _vocabulary.modifiers()
        #: Keystroke -> (the modifier held with it or `None`, the key or the
        #: modifier it is), every keystroke the file's keyboard block binds.
        self._strokes = {s: _vocabulary.keystroke(s) for s in controls.keystrokes}
        #: Modifier -> its two keys, for the modifiers this block names at all.
        self._modifiers = {
            m: keys for m, keys in modifiers.items()
            if any(m in (mod, key) for mod, key in self._strokes.values())
        }
        #: Key -> the modifiers bound in front of it here. While one of them is
        #: down, the key alone is dead: Shift + J is not J as well.
        self._chorded: dict[str, frozenset[str]] = {}
        for mod, key in self._strokes.values():
            if mod is not None:
                self._chorded[key] = self._chorded.get(key, frozenset()) | {mod}
        #: Every dictionary key this block reads: each keystroke's key, and both
        #: keys of each modifier it names. Any other key is not this file's.
        self.watched = frozenset(
            {key for _mod, key in self._strokes.values() if key not in modifiers}
            | {k for pair in self._modifiers.values() for k in pair}
        )
        #: Keys down, as their edges report them.
        self._down: set[str] = set()
        #: Keys down and pressed since the last `drop`: what drives.
        self._fresh: set[str] = set()
        #: Live keystroke -> when it last became live, on the operator's clock.
        self._live: dict[str, float] = {}

    def press(self, key: str, now: float) -> set[str] | None:
        """`key` went down at `now`. The keystrokes that became live, or `None`
        for a key already down -- a repeat, which is the same hold."""
        if key in self._down:
            return None
        self._down.add(key)
        self._fresh.add(key)
        return self._refresh(now)

    def lift(self, key: str, now: float) -> set[str]:
        """`key` came up at `now`. The keystrokes that became live: a key whose
        modifier was let go of is itself again."""
        self._down.discard(key)
        self._fresh.discard(key)
        return self._refresh(now)

    def drop(self) -> None:
        """Hands off: nothing is live, and a key still down drives nothing --
        as a key or as a modifier -- until it is pressed again."""
        self._fresh.clear()
        self._live.clear()

    def _refresh(self, now: float) -> set[str]:
        held = {m for m, pair in self._modifiers.items() if self._fresh.intersection(pair)}
        live = set()
        for stroke, (mod, key) in self._strokes.items():
            if mod is not None:
                on = key in self._fresh and mod in held
            elif key in self._modifiers:
                on = key in held
            else:
                on = key in self._fresh and not (self._chorded.get(key, frozenset()) & held)
            if on:
                live.add(stroke)
        became = live - self._live.keys()
        self._live = {s: self._live.get(s, now) for s in live}
        return became

    @property
    def live(self) -> frozenset[str]:
        return frozenset(self._live)

    def ramp(self, stroke: str, now: float) -> float:
        """0..1: how long `stroke` has been live, over `full_after_s`."""
        since = self._live.get(stroke)
        if since is None:
            return 0.0
        return min(1.0, max(0.0, (now - since) / self._full))

    def deflection(self, axis: KeyAxis, now: float) -> float:
        """One command axis, -1..1: its `+` keystrokes less its `-` ones."""
        x = sum(self.ramp(s, now) for s in axis.plus) - sum(self.ramp(s, now) for s in axis.minus)
        return max(-1.0, min(1.0, x))

    def amount(self, strokes: Iterable[str], now: float) -> float:
        """A task's `amount`: the furthest of its keystrokes' ramps."""
        return max((self.ramp(s, now) for s in strokes), default=0.0)

    def pressed(self, strokes: Iterable[str]) -> bool:
        """A task's `press`: any of its keystrokes live, at once and in full."""
        return any(s in self._live for s in strokes)


# ── The operator ────────────────────────────────────────────────────────────

_OPERATORS: weakref.WeakKeyDictionary[Any, Operator] = weakref.WeakKeyDictionary()


class Operator:
    """The person, as seen by every command term of one environment.

    Two threads touch it: the viewer's UI thread through `_key`, and the
    stepping thread through `command` and `task_control`. Only Python numbers
    cross between them, under one lock; the device writes happen in the terms,
    on the stepping thread, where the tensors live.
    """

    def __init__(
        self,
        controls: Controls,
        pad: Any = None,
        *,
        listen: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ):
        """
        Args:
            pad: an open `controller.Gamepad`, or anything with `state()` and
                `name`; `None` for the keyboard alone.
            listen: register for the viewer's keys. Off only for a test that
                feeds keys itself and would otherwise leave a handler behind.
            clock: seconds, monotonic -- what a held key's time and a moving
                axis's travel are measured by. A person's time, not the
                simulator's: a key held two seconds is two seconds however fast
                the replay runs. A test passes its own.
        """
        self.controls = controls
        self._pad = pad
        self._clock = clock
        self._keyboard = Keyboard(controls)
        self._lock = threading.Lock()
        self._shifted = False
        self._active = False
        self._source = "keyboard"
        self._held: frozenset[str] = frozenset()
        self._pad_values = dict.fromkeys(_STICK_FIELD, 0.0)
        #: The pad's d-pad as of its last frame, `(hat_x, hat_y)` as
        #: `controller.GamepadState` reports it: evdev's signs, so up is -1.
        self._hat = (0, 0)
        #: Each axis the controls move rather than place -> where it has been
        #: moved to, -1..1 of the way from its rest to an end of its range.
        self._position = dict.fromkeys(controls.integrating, 0.0)
        #: The operator's clock at the last step: what a moving axis moves over.
        self._last: float | None = None
        #: The reset button went down and nothing it could be reaching for has
        #: moved since: let go now, it is a tap.
        self._armed = False
        #: The pad axes whose moving makes a press of the reset button no tap:
        #: every one a shifted binding or a moving axis's binding reads. On
        #: jumper.posture's R3 that is the right stick, both ways.
        self._tap_breakers = frozenset(
            b.source for axis, parts in controls.bindings.items() for b in parts
            if b.shifted or axis in controls.integrating
        )
        self._spans: dict[str, dict[str, tuple[float, float, float]]] = {}
        self._stamp: Any = object()
        self._checked = False
        self._listening = listen
        #: What `for_env` was asked for, so a second term asking for another
        #: pad is refused rather than quietly given the first one's.
        self._wants: tuple[bool, str | None] = (pad is not None, None)
        if listen:
            keys.register(self._key)

    @classmethod
    def of_env(cls, env: Any) -> Operator | None:
        """The environment's operator, if one of its command terms has made one.

        For a term that is not a command and so does not make it -- the event
        manager builds its terms before the command manager does.
        """
        return _OPERATORS.get(env)

    @classmethod
    def for_env(
        cls, env: Any, controls: Controls, *, pad: bool, pad_name: str | None
    ) -> Operator:
        """The environment's operator, made by whichever term asks first."""
        op = _OPERATORS.get(env)
        if op is not None:
            if op.controls.source_sha256 != controls.source_sha256 or op._wants != (pad, pad_name):
                raise ValueError(
                    "two command terms of one environment were given different controls "
                    f"({op.controls.source} and {controls.source}) or different pads. "
                    "One person drives them, so they read one file and one device."
                )
            return op
        device = None
        if pad:
            import controller

            device = controller.open(pad_name)
            if device is None:
                print("[operator] no controller; the keyboard drives")
                print(controller.describe())
        op = cls(controls, device)
        op._wants = (pad, pad_name)
        _OPERATORS[env] = op
        op.print_bindings()
        return op

    def attach(self, term: str, term_spans: dict[str, tuple[float, float, float]]) -> None:
        """A command term announcing itself and the ranges it clamps to."""
        self.controls.command(term)  # raises for a term the file does not describe
        self._spans[term] = term_spans

    def close(self) -> None:
        if self._listening:
            keys.unregister(self._key)
            self._listening = False
        if self._pad is not None and hasattr(self._pad, "close"):
            self._pad.close()

    # ── Input ────────────────────────────────────────────────────────────

    def _key(self, keycode: int, down: bool) -> None:
        """The handler `mjrl.viewer.keys` calls, on the viewer's UI thread, with
        both edges of every key -- Shift, Alt and Ctrl included: a key held is
        down from its press to its release, and its auto-repeat is not reported
        at all. Until 2026-09-29 the viewer gave the press alone and the release
        was inferred from repeats that never arrived, so every key let go 0.75 s
        in -- see that module. A key the dictionary does not have is nobody's."""
        name = _GLFW.get(keycode)
        if name is not None:
            self.key(name, down)

    def key(self, name: str, down: bool) -> bool:
        """A key, by the dictionary's name, going down or coming up now, on the
        operator's clock. Returns whether this file's keyboard reads it.

        A key going down that sets one of the file's keystrokes acting takes
        the operator over for the keyboard -- a modifier with nothing to act on
        does not, as `operator.rs` has it -- and a release keystroke becoming
        live lets go of everything instead.
        """
        if name not in self._keyboard.watched:
            return False
        with self._lock:
            now = self._clock()
            became = self._keyboard.press(name, now) if down else self._keyboard.lift(name, now)
            lines: list[str] = []
            if became and not became.isdisjoint(self.controls.key_release):
                lines = self._release("keyboard")
            elif down and became:
                lines = self._take("keyboard")
                lines.append(self.describe())
        for line in lines:
            print(line)
        return True

    def _poll(self) -> list[str]:
        """Read the pad once. Returns what is worth printing."""
        if self._pad is None:
            return []
        s = self._pad.state()
        held = frozenset(s.buttons)
        pressed, lifted = held - self._held, self._held - held
        self._held = held
        self._pad_values = {name: float(getattr(s, field)) for name, field in _STICK_FIELD.items()}
        self._hat = (int(s.hat_x), int(s.hat_y))
        c = self.controls
        if c.release_button in held:
            return self._release("pad")
        lines = []
        if c.shift_gesture == "hold":
            lines += filter(None, [self._shift_to(c.shift in held)])
        elif c.shift in pressed:
            lines.append(self._shift_to(not self._shifted))
        if c.reset is not None:
            lines += self._tap(c.reset in pressed, c.reset in held, c.reset in lifted)
        if s.any_input:
            if self._source != "pad":
                # Otherwise letting go of the pad would bring back whatever the
                # keys had set before it, which nobody is looking at any more.
                self._keyboard.drop()
            lines += self._take("pad")
        return lines

    def _tap(self, pressed: bool, held: bool, lifted: bool) -> list[str]:
        """The reset button's tap: armed going down, disarmed by a stick it could
        have been reaching for moving while it is held, and on coming up while
        still armed, every moving axis back at its rest."""
        if pressed:
            self._armed = True
        if held and any(self._pad_values[a] != 0.0 for a in self._tap_breakers):
            self._armed = False
        if not (lifted and self._armed):
            return []
        self._armed = False
        for axis in self._position:
            self._position[axis] = 0.0
        return [(f"[operator] {self.controls.reset} tapped -- "
                 f"{', '.join(self._position)} back at rest")]

    def _take(self, device: str) -> list[str]:
        self._source = device
        if self._active:
            return []
        self._active = True
        c = self.controls
        back = "/".join(stroke_name(s) for s in c.key_release)
        return [(f"[operator] taken over by the {device} -- {c.release_button} on the pad or "
                 f"{back} on the keyboard hands both commands back")]

    def _release(self, device: str) -> list[str]:
        """Let go of everything: the commands back to the sampler, the keyboard's
        holds dropped, the shift off and every moving axis at its rest."""
        was = self._active
        self._active = False
        self._shifted = self._armed = False
        self._keyboard.drop()
        for axis in self._position:
            self._position[axis] = 0.0
        if not was:
            return []
        return [f"[operator] released from the {device} -- commands are being sampled again"]

    def _shift_to(self, on: bool) -> str | None:
        """Put the shifted layer in or out; the line to print, or `None` when
        it already was."""
        if on == self._shifted:
            return None
        self._shifted = on
        layered = [a for a, parts in self.controls.bindings.items() if any(b.shifted for b in parts)]
        state = "on: its layer drives " + ", ".join(layered) if on else "off"
        return f"[operator] {self.controls.shift} {state}"

    # ── Output ───────────────────────────────────────────────────────────

    def command(self, term: str, stamp: Any = None) -> list[float] | None:
        """This step's values for one command term, or `None` while inactive.

        `stamp` is the environment's step counter: the pad is read and the clock
        advanced once per step however many terms ask, so a click of the shift
        is one click. `None` does both on every call.
        """
        with self._lock:
            lines = self._advance(stamp)
            values = None
            if self._active:
                x = self._axes(self._clock())
                values = [scale(x[a.name], self._spans[term][a.name])
                          for a in self.controls.command(term).axes]
        for line in lines:
            print(line)
        return values

    def task_control(self, name: str, stamp: Any = None) -> float | None:
        """One of the controls the file keeps for the task, by the task's name for
        it, or `None` while inactive. 0..1, from the device driving: an `amount`
        as far as it is held -- a trigger's travel, or its key's ramp -- and a
        `press` 1 while it is held, at once.

        `stamp` works as it does for `command`: the pad is read once per step,
        by whichever asks first.
        """
        control = self.controls.task.get(name)
        if control is None:
            raise KeyError(
                f"{self.controls.source} keeps {sorted(self.controls.task)} for the task, "
                f"not {name!r}. A control read from outside its `task:` is one neither "
                "device binds to it, and one a command may be bound to."
            )
        with self._lock:
            lines = self._advance(stamp)
            value = self._task_value(control, self._clock()) if self._active else None
        for line in lines:
            print(line)
        return value

    def _task_value(self, control: TaskControl, now: float) -> float:
        if self._source == "pad":
            where = self.controls.pad_task[control.name]
            if where in _DPAD:
                return 1.0 if self._dpad(where) else 0.0
            # An `amount` on a stick reads its magnitude, as far as it is pushed
            # either way; only the triggers carry one today.
            return min(1.0, abs(self._pad_values[where]))
        strokes = self.controls.key_task[control.name]
        if control.kind == "press":
            return 1.0 if self._keyboard.pressed(strokes) else 0.0
        return self._keyboard.amount(strokes, now)

    def _dpad(self, direction: str) -> bool:
        """A d-pad direction held on the pad. `controller.GamepadState` keeps
        evdev's signs, up negative -- the robot's service negates it, and
        `rl/mjrl/app_play.py` does too when it hands the pad on. A rocker cannot
        hold two opposite directions, so they cannot both read held."""
        x, y = self._hat
        return {"dpad_up": y < 0, "dpad_down": y > 0,
                "dpad_left": x < 0, "dpad_right": x > 0}[direction]

    def _advance(self, stamp: Any) -> list[str]:
        """Read the pad and move the moving axes, once per step whoever asks
        first. Under the lock."""
        if stamp is not None and stamp == self._stamp:
            return []
        self._stamp = stamp
        self._check_attached()
        lines = self._poll()
        now = self._clock()
        if self._position and self._active and self._source == "pad" and self._last is not None:
            # At the deflection the device driving has now, over the seconds
            # since the last step: a person's time, so a stick held one second
            # moves the axis `1 / integrate_s` of its way however fast the
            # replay steps.
            dt = max(0.0, now - self._last)
            x = self._deflections(now)
            for axis, seconds in self.controls.integrating.items():
                moved = self._position[axis] + x[axis] * dt / seconds
                self._position[axis] = max(-1.0, min(1.0, moved))
        self._last = now
        return lines

    def _deflections(self, now: float) -> dict[str, float]:
        """Each axis's deflection, -1..1, from the device driving."""
        if self._source == "pad":
            return deflections(self._pad_values, self._shifted, self.controls)
        return {axis: self._keyboard.deflection(k, now)
                for axis, k in self.controls.key_axes.items()}

    def _axes(self, now: float) -> dict[str, float]:
        """What each axis is commanded at, -1..1 before scaling: its deflection,
        or, while the pad drives, where the pad has moved it for an axis that
        moves. The keys place every axis."""
        moved = self._position if self._source == "pad" else {}
        return {**self._deflections(now), **moved}

    def _check_attached(self) -> None:
        """Every command the file describes has a term listening.

        Checked at the first step rather than at construction, because the
        terms are built one after another. A command left out would keep its
        bindings in the file and in the banner and do nothing at all.
        """
        if self._checked:
            return
        missing = [c.term for c in self.controls.commands if c.term not in self._spans]
        if missing:
            raise RuntimeError(
                f"{self.controls.source} describes {missing}, and no command term in "
                "this environment is driven by it. Its bindings would do nothing; "
                "install the operator's term for it in `play`."
            )
        self._checked = True

    def describe(self) -> str:
        """The command every attached term would be given now, on one line."""
        x = self._axes(self._clock())
        parts = []
        for c in self.controls.commands:
            if c.term not in self._spans:
                continue
            for a in c.axes:
                v = scale(x[a.name], self._spans[c.term][a.name])
                name = _SHORT.get(a.name, a.name)
                if a.unit == "rad":
                    parts.append(f"{name} {math.degrees(v):+5.1f}°")
                elif a.unit == "m":
                    parts.append(f"{name} {v:.3f} m")
                else:
                    parts.append(f"{name} {v:+.2f}")
        mode = f"  [{self.controls.shift}]" if self._shifted else ""
        return "[operator] " + "  ".join(parts) + mode

    def print_bindings(self) -> None:
        """What drives what, generated from the file rather than written out:
        the pad and the keyboard as two lists, each in the file's own order."""
        c = self.controls
        pad = f"pad {self._pad.name}" if self._pad is not None else "no pad"
        print(f"[operator] {pad}; the pad and the keyboard are two paths, and the last "
              "one touched drives")

        def binding(b) -> str:
            sign = "+" if b.sign > 0 else "-"
            cut = "" if b.travel == (0.0, 1.0) else f"[{b.travel[0]:g}-{b.travel[1]:g}]"
            if len(b.travel) == 4:
                cut += f"off[{b.travel[2]:g}-{b.travel[3]:g}]"
            layer = f"({c.shift})" if b.shifted else ""
            return f"{sign}{b.source}{cut}{layer}"

        print("[operator] pad   " + "  ".join(
            f"{_SHORT.get(axis, axis)}={''.join(binding(b) for b in parts)}"
            for axis, parts in c.bindings.items()
        ))
        also = []
        if c.shift:
            how = "held" if c.shift_gesture == "hold" else "clicked"
            also.append(f"{c.shift} {how} brings in its layer")
        if c.reset:
            also.append(f"{c.reset} tapped puts {', '.join(c.integrating)} back at rest")
        also.append(f"{c.release_button} lets go of everything")
        print("[operator] pad   " + "; ".join(also))
        if c.pad_task:
            print("[operator] pad   task: " + "  ".join(f"{n} {w}" for n, w in c.pad_task.items()))

        def strokes(listed) -> str:
            return "/".join(stroke_name(s) for s in listed)

        print("[operator] keys  " + "  ".join(
            f"{_SHORT.get(axis, axis)}=+{strokes(k.plus)} -{strokes(k.minus)}"
            for axis, k in c.key_axes.items() if k.unbound is None
        ))
        print(f"[operator] keys  {strokes(c.key_release)} lets go of everything; a key held "
              f"is full after {c.full_after_s:g} s")
        if c.key_task:
            print("[operator] keys  task: " + "  ".join(
                f"{n} {strokes(listed)}" for n, listed in c.key_task.items()))
        for axis, k in c.key_axes.items():
            if k.unbound is not None:
                print(f"[operator] keys  none for {axis}: {k.unbound}")
        for t in c.task.values():
            print(f"[operator] task  {t.name} {t.does}")
        for axis, seconds in c.integrating.items():
            print(f"[operator] {axis} moves rather than is placed: full deflection takes it "
                  f"from its rest to an end of its range in {seconds:g} s, and let go it stays")
        print("[operator] click the viewer window first; keys go to whatever has focus")


def connect(cfg: Any, env: ManagerBasedRlEnv, layout: tuple[str, ...]) -> Operator:
    """Hook a command term up to its environment's operator.

    `layout` is the order the term writes its command in. The file's axis order
    has to be the same, because values are written by position: a file that
    listed `lin_vel_y` first would drive the robot sideways with every number on
    screen correct.
    """
    command = cfg.controls.command(cfg.term)
    names = tuple(a.name for a in command.axes)
    if names != tuple(layout):
        raise ValueError(
            f"{cfg.controls.source}: {cfg.term}'s axes are {names}; the command term "
            f"writes {tuple(layout)}, in that order"
        )
    op = Operator.for_env(env, cfg.controls, pad=cfg.pad, pad_name=cfg.pad_name)
    op.attach(cfg.term, spans(command, cfg))
    return op


# ── The velocity command ────────────────────────────────────────────────────


@dataclass(kw_only=True)
class OperatorVelocityCommandCfg(UniformVelocityCommandCfg):
    """A velocity command the task's operator can take over, from either device.

    Inherits the sampling command whole, so until somebody touches a control
    this *is* the sampling command.
    """

    controls: Controls
    """The task's parsed `controls.yaml`."""

    term: str = "twist"
    """This term's key in `cfg.commands`, which is how the file names it."""

    pad: bool = True
    """Open a pad if one is connected. The keyboard is live either way."""

    pad_name: str | None = None
    """Substring of the device name, for a machine with more than one pad."""

    def build(self, env: ManagerBasedRlEnv) -> OperatorVelocityCommand:
        return OperatorVelocityCommand(self, env)


class OperatorVelocityCommand(UniformVelocityCommand):
    cfg: OperatorVelocityCommandCfg

    #: `vel_command_b`'s columns, which mjlab fixes.
    LAYOUT = ("lin_vel_x", "lin_vel_y", "ang_vel_z")

    def __init__(self, cfg: OperatorVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._operator = connect(cfg, env, self.LAYOUT)

    def compute(
        self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        # Written after `super()`, because writing last is what wins.
        # `CommandTerm.compute` runs `_update_command`, which is where standing
        # environments get zeroed, world-frame ones rotated and heading ones
        # have their yaw rate replaced by a heading controller; overwriting
        # after all of that is what keeps the operator's command from being
        # quietly undone. (mjlab's own joystick does the same, and is not
        # reachable here: only its viser viewer builds it.)
        super().compute(dt, env_ids)
        values = self._operator.command(self.cfg.term, self._env.common_step_counter)
        if values is None:
            return
        # Every environment gets the same command. In replay the other robots
        # are there to show the same policy under the same command, and a field
        # where one robot obeys and fifteen wander is harder to read, not richer.
        for i, v in enumerate(values):
            self.vel_command_b[:, i] = v
        self.vel_command_w[:] = self.vel_command_b
