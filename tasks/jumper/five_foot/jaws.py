"""The left claw's jaws, convex-decomposed -- for this task only.

MuJoCo collides a mesh by its convex hull, and the hull of a hooked jaw is filled
in across the mouth. The shared `assets/jumper/jumper.xml` gives `LF_palm_link` (the
fixed jaw) and `LF_finger_link` (the finger) one hull each, which is right for every
other task: there the left claw is a leg, and a leg wants the cheap single hull.
This task carries things in it, so `decompose_jaws` swaps each hull for the pieces
in `meshes/` while the robot's spec is built, and `JAW_COLLISION` is the collision
scheme that makes those pieces collide and grip.

**Here and not in the asset or `constants.py`**, for the reason
`claw.py::add_payload_body` gives for the payload body: it belongs to this task. It
used to be in both, and every jumper task inherited a decomposed, frictional left
claw that only this one uses. Nothing outside this task sees either now.

Measured end to end on master's model (`tools/grasp_objects.py`, warp, MuJoCo
3.11.0, 2026-09-22):

    jaws               soap          notebook      can           slip while carried
    one hull each      yes           NO            NO            7.2 mm (soap)
    6 x 3 (these)      yes           yes           yes           0.3-1.1 mm

The soap is thin enough to fit into a filled-in mouth; the notebook and the can
are not. `claw_mouth.py` on the same model, at `GRIPPER_OPEN`: 1.6 mm of the mouth
lost with these pieces, 16.2 mm with one hull each (`--pieces assets/jumper/meshes`),
against its 5 mm bar.

**The pieces were cut before master re-exported the claw.** `LF_palm_link`'s visual
mesh is the same shape since (13204 vertices, identical once sorted); `LF_finger_link`
moved 3577 of its 9694 vertices by up to 1.27 mm. The table above is these pieces
on the re-exported model, so the drift is measured rather than assumed away.
`tasks/jumper/five_foot/tools/jaw_decompose.py` re-cuts them from the visual meshes if it ever matters --
the `_col` meshes are already hulls, so there is no hook left in them to cut.

How many pieces, measured as the room in the middle of the mouth lost against the
room between the drawn parts, worst band along the mouth, knuckle excluded
(`tasks/jumper/five_foot/tools/claw_mouth.py`, MuJoCo 3.11.0, 2026-09-15):

    cut (J3 x J4)     finger -1.00    finger -0.50
    one hull each        13.4 mm         18.6 mm
    4 x 3                 1.6             4.6
    6 x 3                 1.1             3.8      <- these
    6 x 5                 2.6             3.8
    8 x 5                 2.6             3.7

With one hull each, at finger -0.50 the middle of the mouth is inside a jaw for the
first 30 mm past the knuckle. Six pieces of the fixed jaw and three of the finger
is where more cuts stop buying anything -- five of the finger does no better than
three. The finger's tip and the two pads on the fixed jaw's tip are nearly convex
(the tip's mesh fills 98% of its hull) and stay one hull each.

Only LF. RF walks on its claw.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import mujoco
from mjlab.utils.spec_config import CollisionCfg

from tasks.jumper.common.constants import HYBRID_COLLISION

#: The pieces, `<link>_col0.STL` upwards. Beside the task rather than in
#: `assets/jumper/meshes/`, so the shared asset directory is exactly the model's.
MESH_DIR = Path(__file__).resolve().parent / "meshes"

#: Jaw link -> how many convex pieces it is cut into.
JAW_PIECES: dict[str, int] = {"LF_palm_link": 6, "LF_finger_link": 3}

#: The collision geoms the pieces add. Piece 0 takes over the link's own
#: `<link>_meshcol`, so a link cut into n adds n - 1: `_meshcol1` upwards.
JAW_PIECE_GEOMS: tuple[str, ...] = tuple(
    f"{link}_meshcol{i}" for link, count in JAW_PIECES.items() for i in range(1, count)
)

#: The carried claw's grip faces: every collision piece of both jaws.
#:
#: These are the faces that pinch, and the shared scheme declares them
#: frictionless -- its rule is "feet grip, everything else is condim 1", which was
#: written when nothing was ever gripped and the only thing a non-foot geom did was
#: stop the robot falling through itself. A gripper's inner face is the one other
#: place on this robot where friction is the point of the surface: the real claw is
#: ridged across the grip, which is what a smooth frictionless hull is the opposite
#: of.
#:
#: `LF_palm_link` is here because it is the **fixed jaw** -- the surface an object is
#: pinched against -- and at condim 1 it held nothing when this rule was written: a
#: prop pushed onto it slid along it. Today's props out-rank it and that is no
#: longer so (`JAW_COLLISION` has the measurement). Of the two pads on its tip,
#: `LF_palm_pad_f_link` is here as well, since the finger's tip closes onto it;
#: `LF_palm_pad_b_link` needs no entry, being a foot (`constants.py::FEET`; RF walks
#: on its twin) and so already frictional. RF's own claw is not here: it is the
#: other leg's, and it walks.
GRIP_GEOMS: tuple[str, ...] = (
    "LF_palm_link_meshcol", "LF_finger_link_meshcol", "LF_finger_tip_link_meshcol",
    "LF_palm_pad_f_link_meshcol",
) + JAW_PIECE_GEOMS

#: Sliding, torsional and rolling friction for the jaw faces: a ridged,
#: rubberised grip face on plastic or wood.
#:
#: **What a prop in the claw actually gets.** Jaw and prop are both priority 1,
#: and MuJoCo gives two equal-priority geoms the element-wise *max* of their
#: friction -- so against a prop (0.35, 0.0002, 0.00012; `tasks/jumper/five_foot/objects.py`) the
#: pinch is sliding 1.0 and torsion 0.005, both the jaw's, while the same prop on
#: the floor keeps its own. The rubber decides the contact it is part of, and the
#: hard floor the one it is part of.
#:
#: **Torsion 0.005 m** is 2/3 x 1.0 x a 7.5 mm patch of rubber, and in a pinch it
#: is the only thing resisting the object's spin about the pinch axis: a single
#: contact point has no lever arm. Under mjlab's pyramidal cone it resists like a
#: damper rather than holding -- `tasks/jumper/five_foot/objects.py::PROP_FRICTION` has the
#: measurement -- which is why the objects scene, the one place anything is
#: pinched, runs its own friction settings (`tasks/jumper/five_foot/objects.py::PROP_CONE`).
#: Rolling stays at MuJoCo's default 0.0001.
#:
#: It was (1.5, 0.02, 0.001), chosen to match props that were themselves 1.5 --
#: which put those props at 1.5 against the *floor* too, well past the 0.42 at
#: which a 50 mm object pushed 60 mm up tips rather than slides
#: (`tasks/jumper/five_foot/objects.py::PROP_FRICTION` has the measurement). The torsional 0.02
#: was a 20 mm contact patch at that friction.
GRIP_FRICTION: tuple[float, float, float] = (1.0, 0.005, 0.0001)


def decompose_jaws(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Swap each jaw's single hull for its convex pieces, in place.

    `<link>_meshcol` is re-pointed at piece 0 and keeps its name, so everything
    that names the jaw's geom -- the shared collision scheme, the non-foot contact
    sensor -- still finds it. Pieces 1 upwards are new geoms made from the same
    default class and colour, which is all the asset sets on the hull besides its
    mesh. `JAW_COLLISION` then gives them their contact parameters.

    **Raises rather than skipping** on a model without the hulls, or one already
    decomposed: either way the model is not the one this was measured on, and
    carrying on would give a claw whose mouth is filled in, or doubled, with
    nothing to say so.
    """
    for link, count in JAW_PIECES.items():
        hull = spec.geom(f"{link}_meshcol")
        if hull is None:
            raise ValueError(
                f"{link}_meshcol is not in this model: decompose_jaws is written "
                f"against assets/jumper/jumper.xml")
        if spec.geom(f"{link}_meshcol1") is not None:
            raise ValueError(f"{link} is already decomposed; decompose_jaws ran twice")
        body = spec.body(link)
        for i in range(count):
            spec.add_mesh(name=f"{link}_col{i}", file=str(MESH_DIR / f"{link}_col{i}.STL"))
        hull.meshname = f"{link}_col0"
        for i in range(1, count):
            piece = body.add_geom(default=hull.classname)
            piece.name = f"{link}_meshcol{i}"
            piece.type = mujoco.mjtGeom.mjGEOM_MESH
            piece.meshname = f"{link}_col{i}"
            piece.rgba = hull.rgba
    return spec


def _with_jaws(base: CollisionCfg) -> CollisionCfg:
    """`base`, plus the jaw pieces, with the grip faces made frictional."""
    grip = "|".join(f"^{name}$" for name in GRIP_GEOMS)

    def ahead_of_catch_all(table: dict[str, Any], value: Any) -> dict[str, Any]:
        # `resolve_expr` takes the first pattern that matches, so the grip entry
        # goes before ".*" -- behind it, every jaw piece falls through to condim 1.
        rest = {k: v for k, v in table.items() if k != ".*"}
        tail = {".*": table[".*"]} if ".*" in table else {}
        return rest | {grip: value} | tail

    return dataclasses.replace(
        base,
        geom_names_expr=tuple(base.geom_names_expr) + JAW_PIECE_GEOMS,
        condim=ahead_of_catch_all(base.condim, 3),
        priority=ahead_of_catch_all(base.priority, 1),
        friction=ahead_of_catch_all(base.friction, GRIP_FRICTION),
    )


#: The shared scheme with the jaw pieces in it and the grip faces frictional.
#:
#: **What it adds is the grip, not the collision.** The shared scheme's names are
#: regular expressions matched from the start, so `LF_palm_link_meshcol` already
#: reaches `LF_palm_link_meshcol1` and the pieces collide under it too -- at condim
#: 1, the way every non-foot geom does. Measured with the pieces and the shared
#: scheme (`tools/grasp_objects.py`, warp, 2026-09-22), that does **not**
#: fail the grasp with today's props: all three pick up, carry and release,
#: slipping 0.2-0.9 mm against 0.3-1.1 with the grip. The props are priority 1, so
#: against a condim-1 jaw their own friction (0.35) decides the contact. The
#: decomposition is what the grasp depends on; this is what makes the pinch the
#: rubber's rather than the prop's.
#:
#: **Naming the pieces is what makes it loud.** Used on a model `decompose_jaws`
#: has not touched, mjlab raises on `LF_palm_link_meshcol1` matching no geom, instead
#: of gripping with a mouth that is filled in.
JAW_COLLISION: CollisionCfg = _with_jaws(HYBRID_COLLISION)


def task_spec(asset: Path | None = None) -> mujoco.MjSpec:
    """`constants.get_spec` with the jaws decomposed, for code that compiles its own.

    The tools and tests that measure the claw build the model themselves rather
    than through the environment. Given `get_spec` they would measure the shared
    single hulls and report on a claw this task does not have. Like `get_spec`, it
    leaves the collision scheme to the caller: apply `JAW_COLLISION`, not
    `HYBRID_COLLISION`, or the pieces are dropped from contact.
    """
    from tasks.jumper.common.constants import get_spec

    return decompose_jaws(get_spec(asset))


__all__ = [
    "GRIP_FRICTION",
    "GRIP_GEOMS",
    "JAW_COLLISION",
    "JAW_PIECES",
    "JAW_PIECE_GEOMS",
    "MESH_DIR",
    "decompose_jaws",
    "task_spec",
]
