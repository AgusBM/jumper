"""The brush in the carried claw: a handle along the palm, a soft tip that touches only the floor.

## Where it sits

In `LF_palm_link`'s frame the palm runs along +y to y = 0.144 m, the fixed jaw is
on +z and the finger closes from -z, so the claw's mouth is the strip around
z = 0. The handle lies in that strip along +y and runs past the end of the claw,
which puts the tip `TIP_Y` out along the palm -- 56 mm beyond the claw -- where it
reaches the floor without the palm doing so first. (Extents measured on the V1.6
visual meshes in that frame: palm y [-0.015, 0.144], z [-0.030, 0.025]; finger at
0.0 rad z [-0.031, 0.004].)

The brush is **welded to the palm**, not held by the finger's friction. The finger
is closed onto the handle for the picture (`FINGER_HOLD`), but a grip that could
slip would turn every stroke into a measurement of the grip; that is
`jumper.five_foot`'s claw business, and it is not what this is about.

## What touches what

Only the tip collides, and only with the terrain: contype = the terrain bit,
conaffinity = 0, so no leg, no trunk and no other part of the arm can meet it,
and the handle does not collide at all. The shared collision scheme zeroes every
geom it does not name (`disable_other_geoms`), so the tip has to be added to it
-- `with_brush` -- or it silently passes through the floor.

The tip is **soft** (`TIP_SOLREF`): a time constant several times the feet's
0.008 s, so it sinks a few millimetres under a light press the way bristles
splay. That depth is what the ink pass reads as width; with the feet's stiff
contact it would be a fraction of a millimetre whatever the press.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import mujoco

PALM_BODY = "LF_palm_link"
BRUSH_BODY = "brush"
TIP_GEOM = "brush_tip"
TIP_SITE = "brush_tip"

#: Handle, in the palm's frame: a capsule from HANDLE_Y[0] to HANDLE_Y[1] along +y,
#: centred in the mouth at (x, z) = (0, MOUTH_Z).
MOUTH_Z = -0.004
HANDLE_Y = (0.030, 0.165)
HANDLE_RADIUS = 0.004
#: The bristles: a tapered tuft from the end of the handle to the tip.
TUFT_Y = (0.165, 0.196)
TUFT_RADIUS = 0.0055
#: The colliding tip: a sphere whose far side is the brush's point.
TIP_RADIUS = 0.004
TIP_Y = 0.200 - TIP_RADIUS

#: About a bamboo brush of this size: ~6 g of handle, ~2 g of hair.
HANDLE_MASS = 0.006
TIP_MASS = 0.002

#: Soft and well damped: bristles do not bounce.
TIP_SOLREF = (0.04, 1.0)
TIP_SOLIMP = (0.9, 0.95, 0.004)
#: Wet hair on stone slides; the feet's 1.0 would make the tip stick and drag the arm.
TIP_FRICTION = (0.3, 0.005, 0.0001)

#: The terrain's collision bit (`tasks/jumper/common/constants.py`: the terrain is
#: contype = conaffinity = 1).
TERRAIN_BIT = 1

#: Finger target with the handle in the mouth, rad: closed down onto an 8 mm
#: handle. Between GRIPPER_OPEN (-0.65) and GRIPPER_CLOSED (0.10).
FINGER_HOLD = -0.15


def add_brush(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    palm = spec.body(PALM_BODY)
    if palm is None:
        raise KeyError(f"{PALM_BODY} is not in the robot's spec")
    body = palm.add_body(name=BRUSH_BODY)
    body.add_geom(
        name="brush_handle", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, HANDLE_Y[0], MOUTH_Z, 0, HANDLE_Y[1], MOUTH_Z],
        size=[HANDLE_RADIUS, 0, 0], mass=HANDLE_MASS,
        contype=0, conaffinity=0, group=2, rgba=[0.55, 0.40, 0.22, 1],
    )
    body.add_geom(
        name="brush_tuft", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, TUFT_Y[0], MOUTH_Z, 0, TUFT_Y[1], MOUTH_Z],
        size=[TUFT_RADIUS, 0, 0], mass=0.0,
        contype=0, conaffinity=0, group=2, rgba=[0.10, 0.10, 0.10, 1],
    )
    body.add_geom(
        name=TIP_GEOM, type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0, TIP_Y, MOUTH_Z], size=[TIP_RADIUS, 0, 0], mass=TIP_MASS,
        group=2, rgba=[0.05, 0.05, 0.05, 1],
    )
    body.add_site(name=TIP_SITE, pos=[0, TIP_Y + TIP_RADIUS, MOUTH_Z], size=[0.002, 0, 0],
                  group=4)
    return spec


def _ahead(table: Any, value: Any) -> dict[str, Any]:
    """`table` with the tip's entry first: the scheme takes the first pattern that
    matches, and behind a catch-all ".*" the tip would get a leg's settings."""
    pattern = f"^{TIP_GEOM}$"
    if not isinstance(table, dict):
        return {pattern: value, ".*": table}
    return {pattern: value} | table


def with_brush(base):
    """`base` (an mjlab `CollisionCfg`) with the tip in it: terrain only, soft."""
    def patched(field: str, value: Any) -> Any:
        current = getattr(base, field)
        # An optional patch left as None means "the XML's value" for every geom;
        # a dict for the tip alone then needs only the tip's entry.
        return {f"^{TIP_GEOM}$": value} if current is None else _ahead(current, value)

    return dataclasses.replace(
        base,
        geom_names_expr=tuple(base.geom_names_expr) + (f"^{TIP_GEOM}$",),
        contype=patched("contype", TERRAIN_BIT),
        conaffinity=patched("conaffinity", 0),
        condim=patched("condim", 3),
        priority=patched("priority", 2),
        friction=patched("friction", TIP_FRICTION),
        solref=patched("solref", TIP_SOLREF),
        solimp=patched("solimp", TIP_SOLIMP),
    )


def apply(cfg) -> None:
    """Put the brush on `jumper.five_foot`'s robot in `cfg`, and a sensor on its tip."""
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    robot = cfg.scene.entities["robot"]
    base_spec_fn = robot.spec_fn
    robot.spec_fn = lambda: add_brush(base_spec_fn())
    robot.collisions = tuple(with_brush(c) for c in robot.collisions)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ContactSensorCfg(
            name="brush_ground",
            primary=ContactMatch(mode="geom", pattern=TIP_GEOM, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
        ),
    )
