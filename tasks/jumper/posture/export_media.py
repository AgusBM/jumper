"""What this task ships beside the deployment bundle: the posture command's
specification, for whoever writes the C++ that builds it.

`actor.onnx` + `layout.json` + `README.md` are what the robot needs, and they are
complete for every *other* task in this repository, because every term those
tasks observe is one `rl-wbc-fsm`'s `ObservationBuilder` already constructs. This
one observes `posture_command`, which it does not -- so the bundle has to carry
the thing a name alone cannot: what the four numbers mean, in what order, in what
units, and what has already been subtracted from one of them.

## Why this is a file and not a comment in a pull request

The failure being designed against is the one `scripts/export.py` records twice
in its own source: a note in a JSON file "is read once, by whoever already knew".
The bundle is what gets copied to the board. A specification that lives anywhere
else is a specification that is not there when someone is on the bench wondering
why the robot leans the wrong way.

The short form is also in `layout.json`'s `deploy_notes`, and therefore in the
generated `README.md`, which is what a reader sees first. This file is the long
form: the reasoning, the two ways to implement it, and what to check.

## Why the export was allowed at all

`export.py::_validate_deployable` refuses any observation term the controller
cannot build, and it is right to -- an unknown term is a robot that will not
start. `posture_command` is in its `_DEPLOY_TERMS` as a **promise** that the
integration below is going to happen, made so that a trained policy could be
exported without retraining it into the shape the controller already has. If the
integration is abandoned, take the name back out: an export that fails is better
than a bundle the robot throws on.

## A note on how this file is built

The document is mostly C++ and Markdown tables, both full of braces, and it is
assembled from **plain strings concatenated with one f-string** rather than from
one big f-string. The first version was the latter, and it failed to import for a
missed `}}` -- after `export.py` had already printed `done`, because the hook runs
last. The bundle was written and the specification was not.
"""

from __future__ import annotations

from pathlib import Path

#: The builder's existing term, quoted verbatim from
#: `rl-wbc-fsm/src/observation.cpp`.
_CPP_BASE_POSE = """```cpp
} else if (term == "base_pose") {        // trunk-attitude command
    frame.push_back(command.base_pitch);
    frame.push_back(command.base_roll);
    frame.push_back(command.base_twist);
}
```"""


def _cpp_posture(stand_z: float) -> str:
    """The suggested implementation, with this robot's standing height in it."""
    return (
        "```cpp\n"
        '} else if (term == "posture_command") {\n'
        "    frame.push_back(command.base_twist);\n"
        "    frame.push_back(command.base_pitch);\n"
        "    frame.push_back(command.base_roll);\n"
        f"    frame.push_back(command.height - {stand_z}f);   // centred, see above\n"
        "}\n"
        "```"
    )


def _spec(task_id: str, checkpoint: Path) -> str:
    from ..common.constants import STAND_Z
    from .env_cfg import (
        HEIGHT_RANGE,
        MOVE_LEAN,
        STAND_PITCH,
        STAND_ROLL,
        STAND_THRESHOLD,
        STAND_TWIST,
    )

    lo, hi = HEIGHT_RANGE
    head = f"""# `posture_command`: what the C++ has to build

Bundle: `{task_id}`, from `{checkpoint.name}`.

The observation contains a term the on-robot `ObservationBuilder` does not
construct yet. This is its specification. **Nothing else in the bundle needs a
change** -- the action transform, the joint orders, the gains and the gait clock
are all the same as any other hexa locomotion policy.

## The four channels

`posture_command` is **4 floats, in this order**:

| i | channel | unit | range | sign |
|---|---|---|---|---|
| 0 | twist  | rad | +/-{STAND_TWIST:.4f} parked, +/-{MOVE_LEAN:.4f} walking | + yaws the body **left** over its planted feet |
| 1 | pitch  | rad | +/-{STAND_PITCH:.4f} parked, +/-{MOVE_LEAN:.4f} walking | + is **nose down** |
| 2 | roll   | rad | +/-{STAND_ROLL:.4f} parked, +/-{MOVE_LEAN:.4f} walking | + is **left side up** |
| 3 | height | m   | {lo - STAND_Z:+.5f} to {hi - STAND_Z:+.5f} | + is taller |

All three angles are right-hand-rule about the **body's own axes** (x forward,
y left, z up), which is the same convention `base_pose` already uses. "Walking"
is a velocity command whose `[vx, vy, wz]` has a norm of at least
{STAND_THRESHOLD}; the controller holds the angles to the walking band then.

## The two things that are silent if you get them wrong

**1. The order is not `base_pose`'s.** The builder already has
"""

    order = f"""
-- the same three quantities as channels 0-2 here, in the order **(pitch, roll,
twist)** against this term's **(twist, pitch, roll)**. Wiring `base_pose`'s three
floats into the first three slots compiles, runs, and puts pitch where the policy
reads twist. Nothing reports it; the robot leans in a direction nobody asked for.

**2. Channel 3 is centred and the operator's command is not.** The observation
carries the height **relative to the standing height**:

    posture_command[3] = height_command_m - {STAND_Z}

So an operator asking for {hi:.2f} m sends {hi - STAND_Z:+.5f}, and "stand
normally" is 0.0, not {STAND_Z}. Feeding the raw height instead is a constant
{STAND_Z} m offset on an input whose whole useful range is {hi - lo:.2f} m wide --
the policy would see a command it never met in training and hold something near
the top of its range permanently.

There is no per-term offset in `layout.json`, and that is why this file exists:
the contract can express a scale and cannot express this. (The scale is 1.0 here,
like every other term in this bundle.)

## Two ways to implement it

**A. A new term, matching this contract.** Add `posture_command` to
`term_width()` (returns 4) and to the builder's chain:
"""

    tail = f"""
The command struct already carries all four fields (`base_twist`, `base_pitch`,
`base_roll`, `height`), so this is the builder and nothing else -- no DDS change,
no new operator channel.

**B. Retrain the policy into the shape the controller already has** -- split into
`base_pose` (pitch, roll, twist) plus `height` as a fourth channel of the
`commands` block, and add `"height"` to `command_terms` in
`config/rl-wbc-fsm.toml`. That needs no C++ at all and leaves one spelling for
one quantity, which is the better end state. It costs a full retrain, which is
why this bundle exists in shape A.

**If B ever happens, delete A rather than keeping both.** Two names for the same
four numbers is how a contract vocabulary rots.

## While you are in there: the observation scales have two sources

Not this task's bug, and worth knowing because it bites the same way. `export.py`
writes each term's training `scale` into `layout.json`, and `contract.cpp` never
reads it -- the robot takes `[observation.scales]` from the hand-written
`config/rl-wbc-fsm.toml`, whose comment says *"Policies are trained WITHOUT
observation normalization -> all scales 1.0"*.

That conclusion is right and the reason given is not: these policies **are**
trained with `EmpiricalNormalization`, and 1.0 is correct only because
`export.py` bakes the normaliser into the ONNX graph (`normalization =
"baked_into_onnx"` in `layout.json`, which is why the README says to feed raw
values). Export a policy some day with the normaliser outside the graph and the
TOML will go on saying 1.0, silently.

The fix is to read the scales from the contract. It is a few lines in
`contract.cpp`, and it removes a second source of truth for a number that
training owns.

## Checking it on the bench

With the robot on a stand, feed a static command and read the body back:

| command | expect |
|---|---|
| all zero | stands at {STAND_Z} m, level, square |
| pitch +0.26 | nose down 15 deg, feet stay put |
| roll +0.26 | left side up 15 deg |
| twist +0.26 | body yaws left 15 deg **over stationary feet** |
| height {hi - STAND_Z:+.3f} | stands {hi:.2f} m tall |
| height {lo - STAND_Z:+.3f} | squats to {lo:.2f} m |

The twist row is the one that catches a mis-ordered wiring: it is the only
channel whose effect cannot be confused with another's, because pitch and roll
both tilt the body and twist does not.

Measured in simulation on a stand, holding {hi:.2f} m costs 0.29 N*m at the worst
joint (17% of the peak rating) and {lo:.2f} m costs 0.35 N*m with 2.5 degrees of
travel left at the tightest joint -- **the squat is the demanding end**, not the
extension.
"""

    return head + _CPP_BASE_POSE + order + _cpp_posture(STAND_Z) + tail


def export_media(out: Path, *, task_id: str, checkpoint: Path, asset, res) -> None:
    """Write the posture command's specification into the bundle.

    Called by `scripts/export.py` through `tasks.load_export_media`; see that
    function for why the seam is shaped this way. `asset` and `res` are part of
    the hook's signature and unused here -- this artifact is a document, not a
    rollout.

    It goes at the top of the bundle rather than in `media/` (where `jumper.dance`
    puts its video) because the importer copies the bundle to the board and this
    is meant to be read there. It is a few kilobytes.
    """
    del asset, res
    (out / "POSTURE_COMMAND.md").write_text(
        _spec(task_id, checkpoint), encoding="utf-8"
    )
    print(f"[export] posture command specification  {out / 'POSTURE_COMMAND.md'}")


__all__ = ["export_media"]
