"""Read a task's `controls.yaml`, the operator-control specification.

**The specification is the source of truth and this is the only reader.** The
signs, the splits and the bindings live in the file; `mdp/operator.py` drives
`play` from what this returns, `scripts/export.py` embeds it in the deployment
contract, and `deploy/fsm/src/operator.rs` drives the robot and the browser from
that. Programs that must agree about which way is forward agree by reading one
file rather than by people reading each other's docstrings.

**A task owns its own file.** Controls are neither a fact about the hardware --
which is what `assets/` holds -- nor a mechanism, so by this repository's own
division they belong beside the task's `env_cfg.py`, one copy each, even where
all four locomotion tasks agree. `velocity_env_cfg` raises rather than
defaulting, for the reason that rule exists: a shared default is a decision four
tasks inherit without making, and it looks exactly like agreement.
`tests/test_controls.py` pins that the copies stay identical, so agreement that
is written down survives one of them changing.

Everything here fails loudly, for the reason `load_servo_curve` gives about the
motor spec: every quiet alternative is worse than a stopped import. A missing
axis, a sign left out, a `sign: 0`, a binding naming a key that does not exist --
each would otherwise become a default, and a default sign is a robot that walks
the wrong way while the screen looks entirely normal.

## What the file says

`command` is a list, one entry per command term an operator drives -- one for a
locomotion task, two for `jumper.posture`. **The pad and the keyboard are two
paths**, each bound straight to what it does, and neither names a control of
the other (2026-09-29):

* **The pad** binds a stick or a trigger to a command axis, with a sign. A stick
  can carry more than one axis: a binding may take part of its travel
  (`travel: [0.0, 0.5]`), hand it on to the next one (`[0.0, 0.5, 0.5, 0.75]`
  climbs over the first half and falls back to zero by three quarters), sit on
  a layer a button brings in (`shifted: true`), or be one of several controls
  summed into one axis (a list). The loader refuses two bindings that would
  climb over the same part of the same stick, which is the one way this could go
  quietly wrong.
* **The keyboard** binds keystrokes to one direction of an axis -- `"+"` is the
  axis's own positive direction, as its `positive:` words it, so the keyboard
  has no signs: there is no stick for a sign to be relative to. A keystroke is a
  key (`key_j`), a modifier on its own (`shift`), or a modifier held with a key
  (`shift+key_j`), which is a different keystroke from the key alone: J and
  Shift + J are two things and never both (`controller/vocabulary.py`). A key
  held pushes its direction further the longer it is held, full after
  `full_after_s`, and let go it is back at rest. An axis a task deliberately
  leaves off the keyboard says why (`unbound:`).

  Until 2026-09-29 the keyboard was a *virtual pad*: a key held moved one of
  the pad's controls, read through the pad's own mapping. That made the two
  devices agree by construction, and it made a keyboard that could not lay
  itself out for hands: a key could only be a stick, so J could not turn while
  Shift + J twisted, and Shift, Alt and Ctrl could not hold an arm out. The
  layout Control-agent 3.1 asks for needs both, so the keyboard became a path
  of its own, and agreement is the file's: both paths name the same axes, and
  `load_controls` refuses a command axis either one leaves out without saying.
* **An axis may move rather than be placed** (`integrate_s`), on the pad. A
  deflection is then a speed -- full deflection carries the axis from its rest
  to either end of its range in that many seconds -- and let go it stays where
  it is. jumper.posture's height, since 2026-09-29. `devices.gamepad.reset`
  names the button whose tap puts every such axis back at its rest. The
  keyboard places it as it places every other axis: a key held pushes it
  towards its end and it is back at rest when let go.
* **A task may keep controls for itself.** The top-level `task:` names controls
  no command reads and what the task does with each -- `jumper.five_foot`'s
  claws and its arm presets -- and each device binds every one of them: the pad
  one control each (an axis for an `amount`, a d-pad direction for a `press`),
  the keyboard one or more keystrokes. The task's own code answers them by
  name, in replay a term of its own (`Operator.task_control`) and on the robot
  its deploy hook (`deploy/fsm/src/hook.rs`).

There used to be a schema 1 as well: one command, a stick per axis, and a
keyboard of `+x`/`-wz` nudges in m/s. It was retired rather than kept beside
this one, and every task was rewritten in this one. This one changed in place
on 2026-09-29 -- the keyboard from a virtual pad to the path of its own above --
rather than becoming a schema 3: it had not been released, so there was nothing
to stay compatible with.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# The control dictionary: what a pad and a keyboard can say. Read, never
# repeated -- see `_AXES` below.
from controller import vocabulary as _vocabulary

__all__ = [
    "CENTRE",
    "SCHEMA",
    "Axis",
    "Binding",
    "Command",
    "Controls",
    "KeyAxis",
    "TaskControl",
    "controller_contract",
    "key_name",
    "load_controls",
    "stroke_name",
]

#: The schema this reader implements, and the contract block it writes is
#: `operator_controller/<this>`.
SCHEMA = 2

#: The scaling rule this repository implements. A file asking for another one is
#: refused rather than run under this one.
_SCALING_RULE = "full_deflection_is_range_edge"

#: The gamepad vocabulary, read from the dictionary rather than repeated.
#:
#: `controller/vocabulary.json` is the list and
#: `python -m controller --vocabulary` prints it. This module only decides which
#: of those words a task's controls file may use and what happens when it does
#: -- the meanings, which is the part that belongs to a task.
#:
#: There is no list here on purpose. There were three for a day: this one,
#: `deploy/fsm/src/vocabulary.rs`, and a prose block in a deploy manifest.
_AXES = _vocabulary.axes()
_BUTTONS = frozenset(_vocabulary.buttons())

#: Axis name -> the attribute a `controller` state object carries. A case
#: change and nothing else: `controller/xbox.py` reports the dictionary's names
#: now, and this is only Python's habit of lowercase attributes.
_STICK_FIELD = {name: name.lower() for name in _AXES}

#: Which of the dictionary's axes are sticks (-1..1) and which are triggers
#: (0..1). Read from the ranges the dictionary gives, not from the names.
_BIPOLAR = frozenset(a["name"] for a in _vocabulary.load()["axes"] if a["range"][0] < 0)

#: The four d-pad directions: where a task's `press` control goes on the pad.
#: A rocker is held or not, which is what a `press` reads, and the ten buttons
#: stay the bundle's for switching modes and the operator's for letting go.
_DPAD = frozenset(b["name"] for b in _vocabulary.load()["dpad"])

#: The notched keyboard's other half, retired on 2026-09-27: one key letting go
#: of every stick and trigger at once, needed while a key left its control where
#: the last press put it. A key let up is back at rest and the release lets go
#: of everything, so it did nothing either did not. Named only to refuse a file
#: that still has it.
CENTRE = "centre"

#: The gestures a `shift` implements: `hold`, the layer in for as long as the
#: button is held, and `toggle`, in from one click to the next. `rise` and
#: `fall` are one tick long, which a layer cannot be.
_SHIFT_GESTURES = ("hold", "toggle")

#: The keyboard scheme: keystrokes bound to what they do. See the module
#: docstring.
_KEYBOARD_SCHEME = "keys"

#: What a task control is: `amount`, 0 to 1 as far as it is held -- a trigger,
#: or a key's ramp -- and `press`, 1 while it is held and 0 otherwise.
_TASK_KINDS = ("amount", "press")


@dataclass(frozen=True)
class Axis:
    name: str
    unit: str
    positive: str
    #: The command config field holding where a centred stick puts this axis.
    #: `None` is zero. A height is the case that needs it -- zero is a
    #: body on the floor, and the range it would be clamped to is not where the
    #: robot stands.
    rest: str | None = None
    #: Seconds full deflection takes to carry the axis from its rest to either
    #: end of its range: the controls move it rather than place it, and let go
    #: it stays where it is. `None` is an axis a control places, as a stick's
    #: position places a walking speed.
    integrate_s: float | None = None


@dataclass(frozen=True)
class Command:
    """One command term an operator drives. The file has a list of these."""

    #: The command term's key in `cfg.commands`.
    term: str
    feeds: str
    frame: str
    axes: tuple[Axis, ...]
    scaling: dict[str, Any]


@dataclass(frozen=True)
class Binding:
    """One pad control's share of one command axis."""

    #: A dictionary axis: `Lx` `Ly` `Rx` `Ry` `LT` `RT`.
    source: str
    sign: float
    #: The part of the control's travel this binding takes, as fractions of
    #: full deflection. `(0.0, 0.5)` reaches full at half travel and holds
    #: there; `(0.5, 1.0)` is silent until half travel and then climbs. Two
    #: more, `(0.0, 0.5, 0.5, 0.75)`, are where it falls back: full until the
    #: third, zero from the fourth on -- a binding that gives the control up to
    #: the one climbing over the same stretch.
    travel: tuple[float, ...] = (0.0, 1.0)
    #: Live only while the `shift` button has the other layer in -- held, or
    #: toggled, as `shift_gesture` says. A source that has a shifted binding
    #: drives **only** that binding while shifted; every other source is
    #: unaffected.
    shifted: bool = False


@dataclass(frozen=True)
class KeyAxis:
    """One command axis on the keyboard: the keystrokes pushing it each way,
    or why none do."""

    #: Keystrokes towards the axis's own positive direction, as its `positive:`
    #: words it, and away from it. Both empty exactly when `unbound` says why.
    plus: tuple[str, ...] = ()
    minus: tuple[str, ...] = ()
    unbound: str | None = None


@dataclass(frozen=True)
class TaskControl:
    """A control the task answers itself, by name: no command reads it."""

    name: str
    #: `amount` (0..1, as far as it is held) or `press` (1 while held).
    kind: str
    #: What the task does with it, in the file's words.
    does: str


@dataclass(frozen=True)
class Controls:
    """The specification, parsed: the commands, one operator for all of them."""

    commands: tuple[Command, ...]
    #: The pad: command axis name -> the controls that drive it, summed.
    bindings: dict[str, tuple[Binding, ...]]
    #: The button that brings the shifted layer in, or `None` for a file with none.
    shift: str | None
    release_button: str
    #: Seconds a key is held for its direction to reach full deflection, or a
    #: task's `amount` to reach 1. It climbs linearly until then and is back at
    #: rest when the key comes up.
    full_after_s: float
    #: The keyboard: command axis name -> its keystrokes, every axis present.
    key_axes: dict[str, KeyAxis]
    #: Keystrokes that let go of everything, as the pad's release button does.
    key_release: tuple[str, ...]
    #: The controls the task keeps for itself, by name, in the file's order.
    #: Empty for most tasks.
    task: dict[str, TaskControl]
    #: Task control -> the one pad control it is on.
    pad_task: dict[str, str]
    #: Task control -> the keystrokes it is on.
    key_task: dict[str, tuple[str, ...]]
    keyboard: dict[str, Any]
    gamepad: dict[str, Any]
    source_sha256: str
    source: Path
    #: How the shift brings its layer in: `hold`, for as long as it is held,
    #: or `toggle`, from one click to the next. `None` with no shift.
    shift_gesture: str | None = None
    #: The button whose tap puts every integrating axis back at its rest, or
    #: `None`. A tap is the button pressed and let go with no stick it could
    #: have been reaching for moved in between.
    reset: str | None = None

    def command(self, term: str) -> Command:
        for c in self.commands:
            if c.term == term:
                return c
        raise KeyError(
            f"{self.source}: describes the commands {[c.term for c in self.commands]}, "
            f"not {term!r}. A term driven from a file that does not describe it "
            "would take its bindings from nowhere."
        )

    @property
    def sources(self) -> frozenset[str]:
        """Every pad axis some command binding reads."""
        return frozenset(b.source for bs in self.bindings.values() for b in bs)

    @property
    def integrating(self) -> dict[str, float]:
        """Axis name -> `integrate_s`, for every axis the controls move rather
        than place."""
        return {a.name: a.integrate_s for c in self.commands for a in c.axes
                if a.integrate_s is not None}

    @property
    def keystrokes(self) -> tuple[str, ...]:
        """Every keystroke the keyboard block binds, once each, in file order."""
        strokes = [s for k in self.key_axes.values() for s in (*k.plus, *k.minus)]
        strokes += self.key_release
        strokes += [s for ss in self.key_task.values() for s in ss]
        return tuple(dict.fromkeys(strokes))


class _StrictLoader(yaml.SafeLoader):
    """`yaml.safe_load`, except that a key written twice is an error.

    PyYAML keeps the last of two equal keys and says nothing, so an axis written
    twice in the keyboard's `axes` is one binding silently gone -- the keyboard
    then cannot do something and nothing points at the line that took it away.
    """


def _mapping_without_duplicates(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict:
    seen: set[Any] = set()
    for key_node, _value in node.value:
        key = loader.construct_object(key_node)
        if key in seen:
            raise ValueError(
                f"line {key_node.start_mark.line + 1}: {key!r} is written twice. YAML "
                "keeps the last one and drops the other without a word."
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping_without_duplicates
)


def _require(doc: Any, path: str, where: Path) -> Any:
    node = doc
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"{where}: missing {path}")
        node = node[part]
    if node is None:
        raise ValueError(f"{where}: {path} is null; it has to be decided, not defaulted")
    return node


def load_controls(path: Path) -> Controls:
    """Parse a task's `controls.yaml`. See the module docstring."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It holds how an operator's input becomes the "
            "command the policy observes, and neither play nor the exporter can "
            "proceed without it."
        )
    raw = path.read_bytes()
    try:
        doc = yaml.load(raw.decode("utf-8"), Loader=_StrictLoader)  # a SafeLoader subclass
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None

    version = _require(doc, "schema_version", path)
    if version != SCHEMA:
        raise ValueError(
            f"{path}: schema_version is {version!r}; this reader implements {SCHEMA}. "
            "Schema 1 was retired -- regenerate the file with "
            ".claude/skills/controls/scripts/write_controls.py."
        )
    return _parse(doc, raw, path)


def _check_button(name: Any, what: str, path: Path) -> None:
    # A button name is a contract, not a convention: `view` is on the pad and
    # the robot's service does not send it, so binding it works on a bench and
    # does nothing on the robot.
    if name not in _BUTTONS:
        extra = (" `view` is on the pad but the service does not publish it."
                 if name == "view" else "")
        raise ValueError(
            f"{path}: {what} is {name!r}, which the robot's gamepad service "
            f"does not publish. It publishes {sorted(_BUTTONS)}.{extra}"
        )


def _check_stroke(stroke: Any, where: str, path: Path) -> tuple[str | None, str]:
    """A keystroke, or a refusal naming where it was written."""
    if not isinstance(stroke, str):
        raise ValueError(f"{path}: {where} has {stroke!r}, which is not a keystroke")
    try:
        return _vocabulary.keystroke(stroke)
    except ValueError as exc:
        raise ValueError(
            f"{path}: {where}: {exc}\n"
            f"           python -m controller --vocabulary\n"
            f"           (A key it lacks is added there, with what it also does in "
            f"MuJoCo's viewer: the viewer calls the user callback in addition to its "
            f"own shortcuts, so most keys do two things at once.)"
        ) from None


def _positive_seconds(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and 0 < value < math.inf)


def _travel_is_ordered(travel: list) -> bool:
    """`0 <= from < to <= 1`, and with a fall `to <= back < off <= 1`.

    A climb or a fall of zero length would be a step, which a stick's travel
    cannot hold still on; the hold between them may be zero, and is, where one
    binding's fall starts as the next one's climb does."""
    a, z, *fall = travel
    if not 0.0 <= a < z <= 1.0:
        return False
    return not fall or z <= fall[0] < fall[1] <= 1.0


def _field(node: Any, key: str, at: str, path: Path) -> Any:
    """`_require` for a node inside a list, so the message says which entry."""
    if not isinstance(node, dict) or key not in node:
        raise ValueError(f"{path}: missing {at}.{key}")
    if node[key] is None:
        raise ValueError(f"{path}: {at}.{key} is null; it has to be decided, not defaulted")
    return node[key]


def _parse(doc: dict, raw: bytes, path: Path) -> Controls:
    """Every check here is a control that would drive nothing, drive two things,
    or be out of one device's reach, without anything raising."""
    commands = _parse_commands(doc, path)
    names = [a.name for c in commands for a in c.axes]
    task = _parse_task(doc, path)

    gamepad = _require(doc, "devices.gamepad", path)
    bindings, shift, gesture, release, reset = _parse_pad(gamepad, commands, path)
    pad_task = _parse_pad_task(gamepad, task, bindings, {release, shift, reset}, path)

    keyboard = _require(doc, "devices.keyboard", path)
    full, key_axes, key_release, key_task = _parse_keyboard(keyboard, names, task, path)

    return Controls(
        commands=commands,
        bindings=bindings,
        shift=shift,
        release_button=release,
        full_after_s=float(full),
        key_axes=key_axes,
        key_release=key_release,
        task=task,
        pad_task=pad_task,
        key_task=key_task,
        keyboard=keyboard,
        gamepad=gamepad,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source=path,
        shift_gesture=gesture if shift is not None else None,
        reset=reset,
    )


def _parse_commands(doc: dict, path: Path) -> tuple[Command, ...]:
    listed = _require(doc, "command", path)
    if not isinstance(listed, list) or not listed:
        raise ValueError(
            f"{path}: `command` is a list of the commands an operator "
            f"drives, one entry per command term; got {type(listed).__name__}"
        )
    commands = []
    for i, entry in enumerate(listed):
        at = f"command[{i}]"
        axes = []
        for j, a in enumerate(_field(entry, "axes", at, path)):
            where = f"{at}.axes[{j}]"
            integrate = a.get("integrate_s") if isinstance(a, dict) else None
            if isinstance(a, dict) and "integrate_s" in a and not _positive_seconds(integrate):
                raise ValueError(
                    f"{path}: {where}.integrate_s is {integrate!r}; it is the seconds full "
                    "deflection takes to carry the axis from its rest to either end of its "
                    "range, a positive number. Leave it out for an axis a control places."
                )
            axes.append(Axis(
                name=_field(a, "name", where, path),
                unit=_field(a, "unit", where, path),
                positive=_field(a, "positive", where, path),
                rest=a.get("rest"),
                integrate_s=float(integrate) if integrate is not None else None,
            ))
        if not axes:
            raise ValueError(f"{path}: {at} names no axes")
        scaling = _field(entry, "scaling", at, path)
        if scaling.get("rule") != _SCALING_RULE:
            raise ValueError(
                f"{path}: {at}.scaling.rule is {scaling.get('rule')!r}; this repository "
                f"implements {_SCALING_RULE!r}. Add the rule rather than running the wrong one."
            )
        commands.append(Command(
            term=_field(entry, "term", at, path),
            feeds=_field(entry, "feeds", at, path),
            frame=_field(entry, "frame", at, path),
            axes=tuple(axes),
            scaling=scaling,
        ))
    terms = [c.term for c in commands]
    if len(set(terms)) != len(terms):
        raise ValueError(f"{path}: a command term is listed twice: {terms}")
    # Bindings are keyed by axis name, so a name two commands share would be one
    # binding driving both.
    names = [a.name for c in commands for a in c.axes]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: an axis name is used twice across the commands: {names}")
    return tuple(commands)


def _parse_task(doc: dict, path: Path) -> dict[str, TaskControl]:
    """The top-level `task:`, the controls the task answers itself."""
    declared = doc.get("task")
    if declared is None:
        if "task" in doc:
            raise ValueError(f"{path}: task is null; leave it out for a task that keeps none")
        return {}
    if not isinstance(declared, dict) or not all(isinstance(n, str) for n in declared):
        raise ValueError(
            f"{path}: task is a {type(declared).__name__}; it maps each control the task "
            "answers itself to its `kind` and what it `does`"
        )
    task = {}
    for name, entry in declared.items():
        kind = _field(entry, "kind", f"task.{name}", path)
        if kind not in _TASK_KINDS:
            raise ValueError(
                f"{path}: task.{name}.kind is {kind!r}; a task control is `amount` (0 to "
                "1, as far as it is held) or `press` (1 while it is held)"
            )
        does = entry.get("does")
        if not isinstance(does, str) or not does.strip():
            raise ValueError(f"{path}: the task keeps {name} and does not say what it does")
        task[name] = TaskControl(name, kind, does)
    return task


def _parse_pad(gamepad: dict, commands: tuple[Command, ...], path: Path):
    """The pad's command bindings, its shift, its release and its reset."""
    names = [a.name for c in commands for a in c.axes]
    for field in ("scheme", "layout", "axes", "deadzone", "release_button"):
        _require(gamepad, field, path)
    if gamepad["scheme"] != "absolute":
        raise ValueError(
            f"{path}: gamepad scheme is {gamepad['scheme']!r}. This file implements "
            "`absolute` only: a stick's position is the command. An axis the controls "
            "should move rather than place says `integrate_s` on the axis, for both "
            "devices at once."
        )
    release = gamepad["release_button"]
    _check_button(release, "release_button", path)

    shift = gesture = None
    if "shift" in gamepad:
        declared = gamepad["shift"]
        shift = _field(declared, "button", "devices.gamepad.shift", path)
        _check_button(shift, "shift.button", path)
        gesture = _field(declared, "gesture", "devices.gamepad.shift", path)
        if gesture not in _vocabulary.gestures():
            raise ValueError(
                f"{path}: shift.gesture is {gesture!r}; the dictionary's gestures are "
                f"{list(_vocabulary.gestures())}"
            )
        if gesture not in _SHIFT_GESTURES:
            raise ValueError(
                f"{path}: shift.gesture is {gesture!r}; a shift is "
                f"{' or '.join(map(repr, _SHIFT_GESTURES))}. "
                "A one-tick gesture brings a layer in for one tick."
            )
        if shift == release:
            raise ValueError(
                f"{path}: {shift} is both the shift and the release button. One "
                "press would hand the command back and change what a stick means."
            )

    bound = gamepad["axes"]
    stray = sorted(set(bound) - set(names))
    if stray:
        raise ValueError(
            f"{path}: gamepad.axes names {stray}, which no command has. A typo "
            f"here is a stick that drives nothing. The axes are {names}"
        )
    bindings: dict[str, tuple[Binding, ...]] = {}
    for name in names:
        entry = bound.get(name)
        if not entry:
            raise ValueError(f"{path}: no control drives {name}; every command axis needs one")
        parts = []
        for part in entry if isinstance(entry, list) else [entry]:
            if not isinstance(part, dict) or "source" not in part:
                raise ValueError(f"{path}: {name} has a binding with no source: {part!r}")
            source = part["source"]
            if source not in _STICK_FIELD:
                raise ValueError(
                    f"{path}: {name} names {source!r}, which the robot's gamepad "
                    f"service does not publish. It publishes {list(_AXES)}"
                )
            sign = part.get("sign")
            if sign not in (1, -1):
                # Not clamped to a default: a sign is the one value here whose
                # wrong answer looks exactly like the right one.
                raise ValueError(f"{path}: {name} has sign {sign!r}; it must be 1 or -1")
            travel = part.get("travel", [0.0, 1.0])
            if (
                not isinstance(travel, list) or len(travel) not in (2, 4)
                or not all(isinstance(t, (int, float)) for t in travel)
                or not _travel_is_ordered(travel)
            ):
                raise ValueError(
                    f"{path}: {name} has travel {travel!r}; it is [from, to] with "
                    "0 <= from < to <= 1, or [from, to, back, off] with "
                    "0 <= from < to <= back < off <= 1, as fractions of full deflection"
                )
            shifted = part.get("shifted", False)
            if type(shifted) is not bool:
                raise ValueError(f"{path}: {name} has shifted {shifted!r}; it is true or false")
            if shifted and shift is None:
                raise ValueError(
                    f"{path}: {name} is on the shifted layer and the file declares no "
                    "shift button, so nothing can ever reach it"
                )
            parts.append(Binding(source, float(sign), tuple(float(t) for t in travel), shifted))
        bindings[name] = tuple(parts)

    # No two bindings may climb over the same stretch of the same control in the
    # same layer: one deflection would then move two axes, and each binding
    # alone looks correct. A binding's fall is not a claim -- it is the one
    # stretch where a deflection may move two axes, because it is how the
    # binding gives the control up to the next one.
    claims: dict[tuple[str, bool], list[tuple[float, float, str]]] = {}
    for name, parts in bindings.items():
        for b in parts:
            claims.setdefault((b.source, b.shifted), []).append((*b.travel[:2], name))
    for (source, layer), spans in claims.items():
        spans.sort()
        for (_a0, b0, n0), (a1, b1, n1) in itertools.pairwise(spans):
            if a1 < b0:
                raise ValueError(
                    f"{path}: {n0} and {n1} both take {source} from {a1:g} to "
                    f"{min(b0, b1):g} of its travel{' on the shifted layer' if layer else ''}. "
                    "One deflection would move both; give them separate travel."
                )
    if shift is not None and not any(b.shifted for parts in bindings.values() for b in parts):
        raise ValueError(f"{path}: {shift} is declared as the shift and no binding is shifted")

    # The reset: a tap puts every integrating axis back at its rest. With no
    # such axis it would be a button that does nothing, bound; on the release
    # button it would be a second meaning for one press. It may be the shift --
    # that is jumper.posture's R3, where a hold is the height and a tap is back
    # to standing, told apart by whether the stick moved.
    reset = gamepad.get("reset")
    if reset is not None or "reset" in gamepad:
        _check_button(reset, "reset", path)
        if reset == release:
            raise ValueError(
                f"{path}: {reset} is both the reset and the release button. The release "
                "already puts every integrating axis back at its rest; a reset on it "
                "is the same press read twice."
            )
        if not any(a.integrate_s is not None for c in commands for a in c.axes):
            raise ValueError(
                f"{path}: gamepad.reset is {reset}, and no axis has `integrate_s`. It "
                "puts an integrating axis back at its rest; with none, it is a button "
                "that does nothing and looks bound."
            )
    return bindings, shift, gesture, release, reset


def _parse_pad_task(
    gamepad: dict, task: dict[str, TaskControl], bindings: dict[str, tuple[Binding, ...]],
    reserved: set, path: Path,
) -> dict[str, str]:
    """`gamepad.task`: every task control on one pad control of its kind."""
    pad_task = _task_block(gamepad, "gamepad", task, path)
    commanded = {b.source for parts in bindings.values() for b in parts}
    owner: dict[str, str] = {}
    for name, control in pad_task.items():
        kind = task[name].kind
        if kind == "press" and control not in _DPAD:
            raise ValueError(
                f"{path}: gamepad.task.{name} is {control!r}; a `press` goes on a d-pad "
                f"direction ({sorted(_DPAD)}) -- the buttons are the bundle's mode "
                "switches and the operator's release"
            )
        if kind == "amount" and control not in _STICK_FIELD:
            raise ValueError(
                f"{path}: gamepad.task.{name} is {control!r}; an `amount` goes on an axis "
                f"the robot's gamepad service publishes, a trigger or a stick ({list(_AXES)})"
            )
        if control in commanded:
            raise ValueError(
                f"{path}: {control} is both the task's ({name}) and a command's; one "
                "deflection would do both"
            )
        if control in reserved:
            raise ValueError(
                f"{path}: {control} is the task's ({name}) and the release, the shift or "
                "the reset button; one press would do both"
            )
        if control in owner:
            raise ValueError(
                f"{path}: {control} is on two task controls, {owner[control]} and {name}; "
                "the task could not tell them apart"
            )
        owner[control] = name
    return pad_task


def _task_block(device: dict, which: str, task: dict[str, TaskControl], path: Path) -> dict:
    """A device's `task:`, naming exactly the controls the top-level `task:` does."""
    block = device.get("task")
    if block is None:
        block = {}
    if not isinstance(block, dict):
        raise ValueError(
            f"{path}: {which}.task is a {type(block).__name__}; it maps each control the "
            "top-level `task:` declares to where it is on this device"
        )
    stray = sorted(set(block) - set(task))
    if stray:
        raise ValueError(
            f"{path}: {which}.task names {stray}, which the top-level `task:` does not "
            f"declare ({sorted(task) or 'nothing'}). A task control is declared once, with "
            "its kind and what it does, and bound on each device by that name."
        )
    missing = [n for n in task if n not in block]
    if missing:
        raise ValueError(
            f"{path}: {which}.task leaves out {missing}. Each device has to be able to "
            "reach every control the task keeps on its own."
        )
    return dict(block)


def _parse_keyboard(keyboard: dict, names: list[str], task: dict[str, TaskControl], path: Path):
    """The keyboard: keystrokes on axis directions, the release and the task's
    controls. Every command axis appears, bound or said to be unbound."""
    if not isinstance(keyboard, dict):
        raise ValueError(f"{path}: devices.keyboard is a {type(keyboard).__name__}")
    if "step" in keyboard:
        raise ValueError(
            f"{path}: the keyboard has a `step`, the notched keyboard retired on "
            "2026-09-26 -- one press, a tenth of the travel, kept until pressed back. A "
            "key now pushes its direction for as long as it is held and is back at rest "
            "when it comes up: write `full_after_s`, the seconds held to full deflection."
        )
    if CENTRE in keyboard:
        raise ValueError(
            f"{path}: the keyboard has `{CENTRE}`, retired on 2026-09-27 with the notched "
            "keyboard it served: a key let up is back at rest, and the release lets go "
            "of everything. Delete it."
        )
    for field in ("scheme", "full_after_s", "axes", "release"):
        _require(keyboard, field, path)
    if keyboard["scheme"] != _KEYBOARD_SCHEME:
        raise ValueError(
            f"{path}: keyboard scheme is {keyboard['scheme']!r}. This file implements "
            f"`{_KEYBOARD_SCHEME}`: each keystroke bound to one direction of an axis, "
            "the release, or a task control."
        )
    full = keyboard["full_after_s"]
    if not _positive_seconds(full):
        raise ValueError(
            f"{path}: keyboard full_after_s is {full!r}; it is the seconds a key is held "
            "for its direction to reach full deflection, a positive number"
        )

    # One keystroke, one thing: an axis direction or the release. The task's
    # controls may share one with each other -- jumper.five_foot's two claws
    # both close on Space, and a mode reads only its own side's -- but not with
    # an axis or the release, or one key would do both.
    owner: dict[str, str] = {}
    parsed: dict[str, tuple[str | None, str]] = {}

    def claim(stroke: Any, what: str, where: str, exclusive: bool = True) -> str:
        parsed[stroke] = _check_stroke(stroke, where, path)
        if stroke in owner and (exclusive or not owner[stroke].startswith("task ")):
            raise ValueError(
                f"{path}: {stroke} is on {owner[stroke]} and on {what}. One keystroke "
                "would do both; give each its own."
            )
        if exclusive:
            owner[stroke] = what
        else:
            owner.setdefault(stroke, what)
        return stroke

    def strokes(listed: Any, where: str) -> list:
        if not isinstance(listed, list) or not listed:
            raise ValueError(
                f"{path}: {where} is {listed!r}; it is a non-empty list of keystrokes"
            )
        return listed

    declared = keyboard["axes"]
    if not isinstance(declared, dict):
        raise ValueError(
            f"{path}: keyboard.axes is a {type(declared).__name__}; it maps every command "
            "axis to its keystrokes"
        )
    stray = sorted(set(declared) - set(names))
    if stray:
        raise ValueError(
            f"{path}: keyboard.axes names {stray}, which no command has. A typo here is "
            f"a key that drives nothing. The axes are {names}"
        )
    key_axes: dict[str, KeyAxis] = {}
    for name in names:
        if name not in declared:
            raise ValueError(
                f"{path}: keyboard.axes leaves out {name}. Every command axis is on the "
                "keyboard or says why not (`unbound: <why>`): an axis quietly missing is "
                "a keyboard that cannot do something the pad can, and nothing says so."
            )
        entry = declared[name]
        if isinstance(entry, dict) and set(entry) == {"unbound"}:
            why = entry["unbound"]
            if not isinstance(why, str) or not why.strip():
                raise ValueError(
                    f"{path}: keyboard.axes.{name} is unbound and does not say why"
                )
            key_axes[name] = KeyAxis(unbound=why)
            continue
        if not isinstance(entry, dict) or set(entry) != {"+", "-"}:
            raise ValueError(
                f"{path}: keyboard.axes.{name} is {entry!r}; it is {{\"+\": [...], \"-\": "
                "[...]}, the keystrokes towards the axis's positive direction and away "
                "from it, or {unbound: <why this axis has no keys>}"
            )
        plus = [claim(s, f"{name} +", f"keyboard.axes.{name}.+")
                for s in strokes(entry["+"], f"keyboard.axes.{name}.+")]
        minus = [claim(s, f"{name} -", f"keyboard.axes.{name}.-")
                 for s in strokes(entry["-"], f"keyboard.axes.{name}.-")]
        key_axes[name] = KeyAxis(plus=tuple(plus), minus=tuple(minus))

    release = tuple(claim(s, "the release", "keyboard.release")
                    for s in strokes(keyboard["release"], "keyboard.release"))

    key_task = {}
    for name, listed in _task_block(keyboard, "keyboard", task, path).items():
        where = f"keyboard.task.{name}"
        seen = [claim(s, f"task {name}", where, exclusive=False) for s in strokes(listed, where)]
        if len(set(seen)) != len(seen):
            raise ValueError(f"{path}: {where} lists a keystroke twice: {seen}")
        key_task[name] = tuple(seen)

    # A modifier held on its own that also modifies a key: holding it for the
    # one would do the other while the hand reaches for the key. A key that is
    # one of the modifier's own keys (`key_left_shift`) held alone is the same.
    modifiers = _vocabulary.modifiers()
    chords = {m for m, _k in parsed.values() if m is not None}
    alone = {k for m, k in parsed.values() if m is None and k in modifiers}
    alone |= {m for m, keys in modifiers.items()
              for _m, k in parsed.values() if _m is None and k in keys}
    both = sorted(chords & alone)
    if both:
        raise ValueError(
            f"{path}: {', '.join(both)} is bound on its own and also modifies a key "
            f"({', '.join(s for s, (m, _k) in parsed.items() if m in both)}). Holding it "
            "for the one would do it while reaching for the other."
        )
    return full, key_axes, release, key_task


def controller_contract(controls: Controls, env_cfg: Any) -> dict[str, Any]:
    """The specification, shaped for an exported policy's contract.

    A transcription of the file rather than a description of the code: what a
    consumer receives is what `play` read. `source_sha256` is carried so it can
    say which revision, and a runner that wants to diff two bundles has a key to
    do it with. `deploy/fsm/src/operator.rs` reads it, as
    `operator_controller/2`; the name is the schema's, and the version is the
    file's, so a reader built for anything else refuses it by name.

    `task` is the file's top-level `task:`, kind and words, and `{}` for a task
    that keeps none: the controller refuses a hook reading a name it does not
    declare. Each device's own `task:` stays in its block, verbatim.

    Args:
        env_cfg: the **environment** config, whose command terms hold the values
            the file names by field -- `rest: neutral_height` becomes the number.
            A consumer on the robot has no config to look a field up in, so the
            block carries the number, and the name beside it for a reader who
            wants to know where it came from. Required: a rest left as a name is
            a height the robot would read as zero, which is a body on the floor.

    Called at export rather than when the config is built, because a task adds
    its other command terms after `velocity_env_cfg` returns -- `velocity_env_cfg`
    attaches this as a callable, the way a reference contract is attached.
    """
    commands = []
    for c in controls.commands:
        term_cfg = (getattr(env_cfg, "commands", None) or {}).get(c.term)
        axes = []
        for i, a in enumerate(c.axes):
            entry = {"name": a.name, "index": i, "unit": a.unit, "positive": a.positive,
                     "range_from": f"command_ranges.{c.term}.{a.name}"}
            if a.rest:
                if term_cfg is None or not hasattr(term_cfg, a.rest):
                    raise ValueError(
                        f"{controls.source}: {c.term}.{a.name} rests at `{a.rest}`, and the "
                        f"environment config has no command term {c.term!r} carrying it. "
                        "The block needs the number: a robot has no config to look it up in."
                    )
                entry["rest"] = float(getattr(term_cfg, a.rest))
                entry["rest_from"] = a.rest
            if a.integrate_s is not None:
                entry["integrate_s"] = float(a.integrate_s)
            axes.append(entry)
        entry = {"term": c.term, "feeds": c.feeds, "frame": c.frame,
                 "axes": axes, "scaling": dict(c.scaling)}
        entry.update(_command_shape(controls, c, term_cfg))
        commands.append(entry)
    return {
        "schema": f"operator_controller/{SCHEMA}",
        "source": {
            "file": controls.source.name,
            "sha256": controls.source_sha256,
            "note": "Parsed from the specification play reads. Comments and provenance "
                    "live in the source file.",
        },
        "command": commands,
        "task": {t.name: {"kind": t.kind, "does": t.does} for t in controls.task.values()},
        "devices": {
            "gamepad": dict(controls.gamepad),
            "keyboard": dict(controls.keyboard),
        },
    }


def _command_shape(controls: Controls, c: Command, term_cfg: Any) -> dict[str, Any]:
    """What happens to a command between the stick and the policy, if anything.

    Two things a command term can carry, and a consumer on the robot has no
    config to find either in, so the block carries the numbers:

    * **`bands`** -- a narrower band some axes are held to while the velocity
      command walks. `command_ranges.<term>` is the standing band, the widest the
      command goes and what a full stick reaches; at or above `standing_below`,
      the norm of the velocity command's `[vx, vy, wz]`, each axis named in
      `moving` is clamped to it. Read off the term's `moving` and
      `stand_threshold`: a half-range or a `(low, high)` pair per axis.
    * **`max_rate`** -- how fast the command may move, in its own units per
      second, because the policy was trained on a command that ramps. Absent is
      a command that steps, which is what every other term does.

    A term that carries neither gets neither, and the block is what it was. A
    `moving` without a threshold, or naming an axis the file does not, raises:
    a band nobody can apply is a clamp the robot silently skips.
    """
    out: dict[str, Any] = {}
    moving = getattr(term_cfg, "moving", None)
    if moving is not None:
        threshold = getattr(term_cfg, "stand_threshold", None)
        velocity = getattr(term_cfg, "velocity_command", None) or getattr(
            term_cfg, "velocity_command_name", None
        )
        if threshold is None or velocity is None:
            raise ValueError(
                f"{controls.source}: {c.term} has a moving band and no "
                "`stand_threshold` / velocity command to decide when it applies"
            )
        names = {a.name for a in c.axes}
        band = {}
        for name, value in vars(moving).items():
            if name.startswith("_"):
                continue
            if name not in names:
                raise ValueError(
                    f"{controls.source}: {c.term}'s moving band names {name!r}, "
                    f"which is not one of the file's axes {sorted(names)}"
                )
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                band[name] = [-float(value), float(value)]
            else:
                low, high = value
                band[name] = [float(low), float(high)]
        out["bands"] = {"standing_below": float(threshold), "velocity_term": str(velocity),
                        "moving": band}
    max_rate = getattr(term_cfg, "max_rate", None)
    if max_rate is not None:
        out["max_rate"] = float(max_rate)
    return out


def key_name(code: int) -> str:
    """A GLFW keycode as something a person can read off a banner.

    A dictionary key is shown by its own name less the `key_` prefix -- `W`,
    `KP8` for `keypad_8` -- and anything else falls back to the number, which
    is what a handler would need to remap it.
    """
    for name, codes in _vocabulary.key_codes().items():
        if codes["glfw"] == code:
            if name.startswith("keypad_"):
                return "KP" + name[len("keypad_"):]
            return name.removeprefix("key_").upper()
    return str(code)


def stroke_name(stroke: str) -> str:
    """A keystroke as a banner shows it: `Shift+J`, `Ctrl`, `SPACE`, `KP1`."""
    modifier, key = _vocabulary.keystroke(stroke)
    shown = (key.capitalize() if key in _vocabulary.modifiers()
             else key_name(_vocabulary.key_codes()[key]["glfw"]))
    return f"{modifier.capitalize()}+{shown}" if modifier else shown
