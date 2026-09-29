#!/usr/bin/env python3
"""Write a task's `controls.yaml`, with every control name taken from the dictionary.

    python3 .claude/skills/controls/scripts/write_controls.py \
        --task jumper.bound --axes lin_vel_x,lin_vel_y,ang_vel_z

    ... --bind lin_vel_x=Ly- --bind ang_vel_z=Rx- --release B

## Why a script and not an instruction

The names a controls file may use come from `controller/vocabulary.json`, and
the whole point of that file is that nobody retypes the list. An instruction to
"pick from the dictionary" is retyping it by hand, once per task, with the model
as the copier -- which is how `left_stick_y` survived in four files.

So this reads the dictionary, refuses a name that is not in it, and writes the
file. What it cannot decide it refuses rather than defaults: a sign is the one
value here whose wrong answer looks exactly like the right one.

## What it writes

One command -- the velocity one every locomotion task has -- with a stick per
axis over the stick's whole travel, and a keyboard of its own beside the pad:
each key bound to one direction of one axis, "+" being the axis's own positive
direction as its `positive:` words it, so the keyboard has no signs to get
wrong. A key held pushes its direction further the longer it is held -- full
after two seconds -- and let go it is back at rest. The keys are the same for
every task, so a hand learns them once. Until 2026-09-29 the keyboard was a
virtual pad, each key a stick direction read through the pad's mapping; a key
is bound to what it does now, and the pad and the keyboard are two paths.

A task that drives more than one command, or splits a stick, puts a binding on
a shift layer, sums two controls, binds a modifier chord, moves an axis rather
than places it (`integrate_s`) or keeps controls for itself (`task:`), edits the
file this writes -- `tasks/jumper/posture/controls.yaml` and
`tasks/jumper/five_foot/controls.yaml` have every part of it between them, and
`load_controls` checks every name in them against the dictionary on load.

It writes no weights, no ranges, no gait parameters. `velocity_env_cfg` raises
for each of those by name, with the reason, and a scaffold that filled them in
would be four tasks inheriting a decision none of them made.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

# After the path insert, necessarily: this runs from a skill directory, not from
# an installed package, and `controller` is at the repository root.
from controller import vocabulary  # noqa: E402

#: Command axis -> (unit, which way is positive).
#:
#: The velocity-tracking vocabulary the observation term uses; the directions
#: are the body frame's, ROS REP-103.
AXIS_MEANING = {
    "lin_vel_x": ("m/s", "forward"),
    "lin_vel_y": ("m/s", "left"),
    "ang_vel_z": ("rad/s", "counter-clockwise"),
}

#: Which stick drives which axis, by default, and with what sign.
#:
#: MEASURED: not against the robot's pad service. This is what every jumper task
#: uses and what `controller/xbox.py` states -- evdev reports `ABS_Y` positive
#: when the stick is pushed **down** and `ABS_X` positive to the **right**, and
#: the pad service passes both through unchanged. So forward, left and
#: counter-clockwise are all the negative end.
#:
#: A default here is defensible where a default gain is not: it is a property of
#: evdev shared by every task, not a tuning choice. It is still written into the
#: file it produces, with that note, so a task that wants another can say so.
DEFAULT_BINDING = {
    "lin_vel_x": ("Ly", -1),
    "lin_vel_y": ("Lx", -1),
    "ang_vel_z": ("Rx", -1),
}

#: The keyboard: command axis -> (the keys towards its positive direction, the
#: keys away from it). W S and the arrows walk forward and back, A D and the
#: arrows step left and right (the arrows asked for on 2026-09-28), J L turn --
#: the layout as the
#: user decided it on 2026-09-29. Two keys on one direction are summed while
#: both are held. Letting go of the keys is taking your hands off; there is no
#: key for it (`centre`, retired 2026-09-27).
#:
#: Every letter is also one of MuJoCo's viewer shortcuts -- W flips the scene to
#: wireframe -- and that was accepted for a keyboard laid out for the hands; the
#: dictionary's `_keys_note` has the table.
KEYS_FOR = {
    "lin_vel_x": (("key_w", "key_up"), ("key_s", "key_down")),
    "lin_vel_y": (("key_a", "key_left"), ("key_d", "key_right")),
    "ang_vel_z": (("key_j",), ("key_l",)),
}

#: The keys that let go of everything, as the pad's release button does: Esc.
#: The keyboard's own, whatever button `--release` gives the pad. It was B until
#: Control-agent 3.1 put the right claw there (2026-09-29).
RELEASE_KEYS = ("key_escape",)

#: Seconds a key is held for its direction to reach full deflection; it is back
#: at rest when the key comes up. Asked for on 2026-09-26, measured against nothing.
FULL_AFTER_S = 2.0


def parse_bind(entry: str) -> tuple[str, str, int]:
    """`lin_vel_x=Ly-` -> ("lin_vel_x", "Ly", -1)."""
    if "=" not in entry:
        raise SystemExit(f"[new-task] --bind wants AXIS=SOURCE[+|-], got {entry!r}")
    axis, source = entry.split("=", 1)
    if not source.endswith(("+", "-")):
        raise SystemExit(
            f"[new-task] --bind {entry!r} has no sign. Write `{axis}={source}-` or "
            f"`{axis}={source}+`.\n"
            f"           Not defaulted: a sign is the one value here whose wrong "
            f"answer looks exactly like the right one."
        )
    return axis, source[:-1], (-1 if source.endswith("-") else 1)


def check(name: str, kind: str, allowed) -> None:
    if name in allowed:
        return
    absent = vocabulary.absent()
    extra = f"\n           {absent[name]}" if name in absent else ""
    raise SystemExit(
        f"[new-task] {name!r} is not {kind} the robot's pad service publishes.\n"
        f"           It publishes: {' '.join(allowed)}{extra}\n"
        f"           python -m controller --vocabulary"
    )


def keyboard_for(axes: list[str]) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Command axis -> its `+` and `-` keystrokes, each checked against the dictionary."""
    out = {}
    for axis in axes:
        for stroke in (*KEYS_FOR[axis][0], *KEYS_FOR[axis][1]):
            try:
                vocabulary.keystroke(stroke)
            except ValueError as exc:
                raise SystemExit(f"[new-task] {exc}\n           python -m controller "
                                 "--vocabulary") from None
        out[axis] = KEYS_FOR[axis]
    return out


def render(task: str, axes: list[str], bind: dict[str, tuple[str, int]],
           release: str, keys: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]) -> str:
    out = [
        f"# {task} operator controls -- how a person's input becomes the command the",
        "# policy observes.",
        "#",
        "# Generated by `.claude/skills/controls/scripts/write_controls.py`, which takes",
        "# every control name from `controller/vocabulary.json`. Run",
        "# `python -m controller --vocabulary` to see the list; it is the same one the",
        "# deployed controller refuses a binding against.",
        "#",
        "# The pad and the keyboard are two paths, each bound straight to the command",
        "# axes -- a stick with a sign, a key with the direction it pushes -- and",
        "# neither names a control of the other. Either device drives alone; the one",
        "# touched last drives.",
        "#",
        "# Two rules this file inherits from the specification it replaced:",
        "#   * Every value carries its source. A number without one cannot be told",
        "#     apart from a guess.",
        "#   * A consumer fails on a missing or malformed value rather than",
        "#     substituting a default. A sign that defaults to +1 is a robot that walks",
        "#     backwards while the screen looks perfectly normal.",
        "",
        "schema_version: 2",
        "",
        "command:",
        "  - term: twist",
        "    # The observation term these numbers are written into.",
        "    feeds: velocity_commands",
        "    # The robot's own frame: +X forward, +Y left, +Z up (ROS REP-103).",
        "    frame: body",
        "    # Order is the order the command term writes them, and therefore the order",
        "    # they appear in the observation.",
        "    axes:",
    ]
    for axis in axes:
        unit, positive = AXIS_MEANING[axis]
        out += [f"      - name: {axis}", f"        unit: {unit}", f"        positive: {positive}"]
    out += [
        "    scaling:",
        "      # Full deflection is the edge of the range the policy was **trained** on,",
        "      # not an absolute speed -- so a stick means \"as fast as this policy was",
        "      # ever asked to go\" and cannot ask for something off-distribution.",
        "      rule: full_deflection_is_range_edge",
        "      formula: axis * (range_max if axis >= 0 else -range_min)",
        "      # A range like (-0.5, 1.0) gives forward twice what it gives back, which is",
        "      # what the policy trained on; taking the larger end for both quietly",
        "      # commands something it never saw.",
        "      ends_scaled_separately: true",
        "",
        "devices:",
        "  gamepad:",
        "    # The stick position *is* the command.",
        "    scheme: absolute",
        "    layout: xbox",
        "    axes:",
    ]
    for axis in axes:
        source, sign = bind[axis]
        out.append(f"      {axis}: {{ source: {source}, sign: {sign} }}")
    out += [
        "    # MEASURED: the signs are not measured against the robot's pad service.",
        "    # `controller/xbox.py` states the same evdev convention -- ABS_Y positive",
        "    # is **down**, ABS_X positive is **right** -- and the service passes both",
        "    # through unchanged, negating `dpad_y` and not `Ly`. Pushing the stick on",
        "    # the robot is what would settle it.",
        "    #",
        "    # The service applies a 0.15 deadband and **rescales**, so full deflection",
        "    # still reaches ±1. A consumer that applies its own on top narrows the",
        "    # range a person can ask for and cannot tell that it has.",
        "    deadzone: device_reported_rescaled",
        "    # Hands the command back to the random sampler in `play --task`; on a robot, which",
        "    # has no sampler, it lets go of everything.",
        f"    release_button: {release}",
        "",
        "  keyboard:",
        "    # Keystrokes bound to what they do: each pushes one direction of one axis,",
        "    # \"+\" being the axis's own positive direction as `positive:` above words",
        "    # it, so the keyboard has no signs of its own -- there is no stick for a",
        "    # sign to be relative to. A virtual pad read through the pad's mapping",
        "    # until 2026-09-29.",
        "    scheme: keys",
        "    # Seconds a key is held for full deflection: the axis climbs linearly",
        "    # until then and is back at rest the moment the key comes up, as a stick",
        "    # let go of. Asked for on 2026-09-26; measured against nothing.",
        f"    full_after_s: {FULL_AFTER_S}",
        "    axes:",
        "      # W S and the arrows walk, A D and the arrows step sideways, J L turn:",
        "      # the layout as the user",
        "      # decided it on 2026-09-29. Every letter is also a MuJoCo viewer",
        "      # shortcut -- W flips the scene to wireframe -- which was accepted for a",
        "      # keyboard laid out for the hands; left and right also step a paused",
        "      # scene.",
    ]
    for axis, (plus, minus) in keys.items():
        out.append(f'      {axis}: {{ "+": [{", ".join(plus)}], "-": [{", ".join(minus)}] }}')
    out += [
        "    # Lets go of everything, as the pad's release button does.",
        f"    release: [{', '.join(RELEASE_KEYS)}]",
        "",
    ]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(prog="write_controls.py", description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, help="task id, e.g. jumper.bound")
    ap.add_argument("--axes", default=",".join(AXIS_MEANING),
                    help="command axes in order (default: all three)")
    ap.add_argument("--bind", action="append", default=[], metavar="AXIS=SOURCE[+|-]",
                    help="which stick drives an axis and with what sign. Repeat. "
                         "Omitted axes take the evdev default.")
    ap.add_argument("--release", default="B",
                    help="the button that hands control back to the sampler")
    ap.add_argument("--out", type=Path, help="where to write (default: the task's directory)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    ap.add_argument("--print", action="store_true", help="write nothing; print it")
    args = ap.parse_args()

    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    for axis in axes:
        if axis not in AXIS_MEANING:
            raise SystemExit(
                f"[new-task] {axis!r} is not a command axis this environment writes.\n"
                f"           It writes: {' '.join(AXIS_MEANING)}"
            )

    known_axes = vocabulary.axes()
    bind = {a: DEFAULT_BINDING[a] for a in axes}
    for entry in args.bind:
        axis, source, sign = parse_bind(entry)
        if axis not in axes:
            raise SystemExit(f"[new-task] --bind names {axis!r}, which is not in --axes")
        check(source, "an axis", known_axes)
        bind[axis] = (source, sign)

    driven = [bind[a][0] for a in axes]
    if len(set(driven)) != len(driven):
        raise SystemExit(
            f"[new-task] two command axes share a stick: {driven}. Splitting a stick's "
            f"travel between two axes is a hand edit -- see jumper.posture's file."
        )
    check(args.release, "a button", vocabulary.buttons())

    text = render(args.task, axes, bind, args.release, keyboard_for(axes))
    if args.print:
        print(text, end="")
        return 0

    out = args.out or REPO / "tasks" / Path(*args.task.split(".")) / "controls.yaml"
    if out.exists() and not args.force:
        raise SystemExit(f"[new-task] {out} exists; pass --force to replace it, or --out")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, "utf-8")
    shown = out.relative_to(REPO) if out.is_relative_to(REPO) else out
    print(f"[new-task] wrote {shown}")
    for axis in axes:
        source, sign = bind[axis]
        print(f"           {axis:<12}{'-' if sign < 0 else '+'}{source}")
    print(f"           release on {args.release}, and on the keyboard "
          f"{' '.join(RELEASE_KEYS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
