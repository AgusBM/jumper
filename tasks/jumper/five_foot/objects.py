"""The graspable objects, in a row on flat ground.

The three things the real robot is handed -- a bar of soap, a notebook and a can
-- side by side half a metre in front of it, and a tray to drop them in. **A row is reachable** -- kk-rl-lab lays
its own pick-place objects out the same way -- where a room puts half its
surfaces at the wrong height for a claw held 60 mm off the floor, and fills the
rest with furniture the robot has to walk around.

The layout is kk-rl-lab's, a row **across** the robot rather than away from it:
every object presents its grip point at the same x, spread along y about the
claw's own y, so the operator walks up once and then strafes between them
(`pick_place.py::object_layout`, `GRIP_X` / `ROW_Y` / `ROW_PITCH`). A row laid
out along +x instead -- which this scene had first -- asks the robot to walk
*through* each object to reach the next one.

The episode also starts the way kk-rl-lab's replay does: the robot at the origin
facing the row, commanding nothing until a key is pressed. That is `START`,
below; without it the task drops the robot anywhere in a 1 m box
facing any direction and walks it off on a sampled command.

## Plain solids, on purpose

Each object is **one shape of one uniform material** -- a box, or a convex hull
that MuJoCo collides exactly as it is drawn -- given the density that makes it
weigh what it weighs, so MuJoCo derives both the mass and the inertia from the
shape itself. How an object slides, tips or rolls then depends on its shape, its
mass and the contact parameters below, and on nothing else. The exceptions are
features that stand proud of the shape where the world can touch them -- the
notebook's belt, the can's tab -- modelled as boxes of their own material
(`Part`).

That replaces a bottle lofted from a profile, a mug of sixteen segments and a
book, each drawn as an STL over a different set of collision primitives. They
were built to look like things and then had to be checked to *be* where they
looked, and what the claw met -- a square grip waist, a handle of three capsules
-- was a property of how they were assembled rather than of what they were made
of. `SOLIDS` is now the whole description of the row, and `PROP_FRICTION` with
its neighbours is the whole description of how any of it touches anything.

## Physical, so replay-only

Everything here has mass and collides, so a policy trained with this scene is not
comparable to one trained without it; `apply_objects` warns about that on
every run. **And it changes the friction model of the whole world**: the props
are held under the elliptic cone at impratio 50 (`PROP_CONE`), and so is
everything else, the robot's feet included, where the task trains on the
pyramidal cone at impratio 1.

Grasping is contact and friction and nothing else. The claw's trigger squeezed
until the jaws stop on an object squeezes it, and it comes up with the claw; let
go, and the claw lets it go. Until the
claw squeezes something the props are moved by contact alone, so walking the jaws
into one pushes it. `tasks/jumper/five_foot/mdp/grasp.py` has what this replaced.

## Cost

Props are not free at a training-sized environment count -- **every environment
gets its own copy of the row**. Measured on an RTX 5090 with the previous row,
whose mug alone was sixteen geoms, one process, five steps after construction:
728 MiB for one environment with no scene, 774 with the row; at 512 environments,
1376 MiB against about 4 GB. Three solids of six geoms cost less. Replay wants
`--num_envs 1`, which `apply_objects` says out loud.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import mujoco

# ══ The props: what the claw is for ══════════════════════════════════════
#
# **These are physical.** Everything a look-only scene adds is drawn and cannot be
# touched; everything below has mass and collides, so a policy trained with this
# row is not comparable to one trained without it. `apply_objects` warns about
# that on every run, which is the intended amount of noise -- do not train with
# `--objects`, replay with it.
#
# **Their sizes are the real objects', and the claw has to fit them.** The jaws
# open to 76 mm at the anvil and shut to 0.1, and the mouth is narrower than that
# further in, so the pinch axis of anything meant to be gripped has to sit inside
# that window or the object is ungrippable however well the robot is driven --
# which is why each stands upright and is gripped across its thinnest side.
# `tasks/jumper/five_foot/claw.py` carries the aperture curve.

#: Collision bit for the props, so they collide with the world and each other
#: without joining the robot's per-leg self-collision channels (`constants.py`
#: reserves bit 0 for the terrain and one bit per leg).
PROP_BIT = 1 << 11

# ── How a prop touches things ─────────────────────────────────────────────
#
# MuJoCo gives a contact no parameters of its own; it derives them from the two
# geoms. **The higher `priority` wins outright; equal priority takes the larger
# `condim` and the element-wise maximum of the friction.** The numbers below are
# chosen through that rule rather than around it:
#
#     prop against the ground plane (priority 0)   the prop's own values
#     prop against another prop                    the prop's own values
#     prop against the claw's jaw (priority 1)     the larger of the two -- the
#                                                  jaw's grip and its torsion
#
# so one set of numbers is a hard-surface contact on the ground and a rubber-jaw
# contact in the claw, without declaring contact pairs. Read off a compiled model,
# a prop on the ground gets (0.35, 0.0002, 0.00012) and the same prop in the jaw
# (1.0, 0.005, 0.00012); `tests/test_five_foot_objects.py` pins both.

#: Beats the ground plane's 0, so the prop's floor friction is the one that
#: applies there; ties the robot's jaws and feet at 1, so theirs apply where they
#: are larger.
PROP_PRIORITY = 1

#: Sliding, torsional and rolling friction.
#:
#: **Sliding 0.35**: painted aluminium, soap or leather on a hard floor. It also
#: decides whether an object the claw walks into slides or tips over, which is a
#: matter of `friction > half-width / push height`: the jaws meet an object about
#: 60 mm up -- 60.3 mm on V1.6, settled -- so the can, standing on a 46 mm ring,
#: tips above 23/60 = 0.38, and
#: the soap and the notebook, 28 and 30 mm across what they stand on, above about
#: 0.24. Measured, pushed at 60 mm and 0.1 m/s for 2 s (millimetres travelled):
#:
#:     floor friction          0.30          0.35          0.40          0.50
#:     can                    slides 174    slides 105    slides 107    tips
#:     soap, along 66         slides 118    slides 134    slides 122    slides 167
#:     notebook, along 105    slides 160    slides 169    slides 176    slides 163
#:     soap, across 36        tips          tips          tips          tips
#:     notebook, across 30    tips          tips          tips          tips
#:
#: So at 0.35 the can and anything pushed along its depth slides, and the two thin
#: objects fall over when the closing jaw pushes them sideways; much grippier and
#: the can falls over too.
#:
#: **Torsional and rolling friction are lengths, not ratios, and both belong to
#: a contact point.** MuJoCo caps a contact's spin torque at `torsional * N` and
#: its rolling torque at `rolling * N`, so each is sized to the patch one contact
#: point stands for -- not to the object's face, whose extent the solver already
#: represents with several points (four under a box, three under a standing
#: cylinder, two under a lying one) and their lever arms.
#:
#: **Torsional 0.0002 m**: 2/3 x 0.35 x a 1 mm patch, a hard corner on a hard
#: floor. The face does the rest. Measured, the soap standing on the floor under
#: this scene's solver holds a 20 mN*m twist whether this is 0.0002 or MuJoCo's
#: default 0.005, and breaks away by 24 with this and by 30 with the default --
#: against 23.8 mN*m from the corners' lever arms alone. Where one point is all
#: there is, which is the claw's pinch, the pair takes the jaw's larger value
#: (`tasks/jumper/five_foot/jaws.py::GRIP_FRICTION`), which is the rubber that is
#: actually there.
#:
#: **Rolling 0.00012 m**: a hard cylinder on a hard floor has a rolling-resistance
#: coefficient of about 0.004, and MuJoCo's rolling friction is that times the
#: radius -- 0.004 x 29 mm, the can's. Measured under this scene's solver, the can
#: rolled on its side decelerates at 0.0272 m/s^2 against the 0.0270 that predicts
#: (`rolling * g / 1.5R`; 0.0258 under the task's pyramidal cone), so the number
#: does what the arithmetic says. The consequence is the physical one: knocked
#: over and rolling at 0.5 m/s it goes some 4.6 m. A softer floor -- a rubber mat,
#: carpet -- is a coefficient near 0.02, which is 0.0006 here; at 0.0005 the can
#: stops in 1.1 m. That is a statement about the floor, not a tuning knob.
#:
#: **What neither does under the task's own solver: hold still.** mjlab's
#: pyramidal friction cone at impratio 1 gives torsion and rolling no static
#: phase -- they resist like a damper, not like a Coulomb limit. Measured, the soap
#: pinched at 10 N a side between two point pads spins at 0.96 rad/s under a
#: 2 mN*m twist, a fiftieth of the 100 mN*m the coefficients allow; the block the
#: row used to have spun at 1.1, faster in proportion to the twist, and ten times
#: the solver iterations changed nothing. Sliding friction does the same to an
#: object in the claw, which is why this scene runs its own friction settings
#: (`PROP_CONE`), under which the same pinch turns at 0.0097 rad/s.
PROP_FRICTION = (0.35, 0.0002, 0.00012)

#: 6: sliding, torsional **and rolling**. MuJoCo evaluates the torsional
#: coefficient only at condim 4 and above and the rolling one only at 6, and a
#: coefficient that is never evaluated still reads correctly in the config.
#: Measured, the can on its side rolled at 0.5 m/s for 10 s under this scene's
#: solver:
#:
#:     condim 3                           4.74 m, still at 0.44 m/s
#:     condim 4                           4.98 m, still at 0.50 m/s
#:     condim 6, rolling 0.00012          3.64 m, still at 0.23 m/s
#:     condim 6, rolling 0.0005           1.11 m, at rest
#:
#: It is not free. Every friction dimension is constraint rows: under this
#: scene's elliptic cone a prop contact is six rows where a foot's condim-3 contact
#: is three, and under the task's pyramidal cone, 2 x (condim - 1), ten and four.
#: `scenes/football.py` met the same evaluation trap with its ball.
PROP_CONDIM = 6

#: MuJoCo's default and the robot's own (`constants.py::_CONTACT_SOLREF`): a
#: 20 ms time constant, four times this task's 5 ms step. The props used 10 ms,
#: which is exactly the 2 x timestep MuJoCo gives as the stiffest stable setting
#: -- the edge rather than a margin.
PROP_SOLREF = (0.02, 1.0)

#: The friction the props are held under: **the elliptic cone at impratio 50**,
#: set by this scene on the whole model (`Scene.cone`, `Scene.impratio`) because
#: the task's own settings do not hold anything.
#:
#: **Why.** mjlab's default is the pyramidal cone at impratio 1, and there
#: friction has no static phase: it resists slip like a damper rather than a
#: Coulomb limit. The row's old 50 mm cylinder standing 15 or 30 mm into the mouth, the claw
#: shut until the finger stalls on it, then the claw raised 15 mm by the shoulder --
#: how far the cylinder came up, as a share of the claw (`jumper.five_foot`, no
#: policy):
#:
#:                                 15 mm in          30 mm in
#:                               warp   native     warp   native
#:     pyramidal, impratio 1      12%    74%         5%    67%
#:     elliptic, impratio 1       82%   102%        69%    90%
#:     pyramidal, impratio 50    118%   104%        95%    94%
#:     elliptic, impratio 50     120%   114%       107%    99%
#:
#: Over 100% is the cylinder sitting further out along the claw than the point
#: measured, so rising more as the shoulder turns. Friction coefficients do not
#: help -- under the task's settings, doubling the jaw's sliding friction to 2.0
#: carried 0 to 42%, and a tenfold torsional friction changed nothing. Impratio,
#: how stiff friction is against the normal force, does most of the work and the
#: cone does the rest, which is plainer at one contact: twisted at a fiftieth of
#: its limit, the old row's 50 mm block pinched as in `tests/test_five_foot_objects.py` turns at
#:
#:                   impratio 1    impratio 10    impratio 50    (rad/s)
#:     pyramidal        1.12          0.19           0.047
#:     elliptic         0.56          0.056          0.011
#:
#: and the soap now in that test at 0.96, 0.040, 0.048 and 0.0097 rad/s -- pyramidal
#: at 1 and 50, elliptic at 10 and 50.
#:
#: **Carried.** The same squeeze and lift, then the policy
#: `2026-09-11_15-11-16/model_6999` walking back 0.6 m, turning 57 degrees and
#: stopping, 12 s in all, on warp. How far the prop moved relative to the claw --
#: the first five rows on the row as it was, the last three on the row as it is.
#: Runs vary -- an earlier model of the can moved 8 mm on one and 30 on the next --
#: so each number is one run:
#:
#:                              elliptic, impratio 10    elliptic, impratio 50
#:     cylinder 15 mm in              13.7 mm                10.6, 8.3 mm
#:     cylinder 30 mm in          fell at the stop                5.3 mm
#:     cylinder 45 mm in                 -                        4.5 mm
#:     block 15 / 30 mm in               -                    5.5 / 6.6 mm
#:     plate 15 / 30 mm in               -                    9.8 / 8.2 mm
#:     soap 30 mm in                     -                        7.9 mm
#:     notebook 30 mm in                 -                       10.5 mm
#:     can 30 mm in                      -                     6.2, 8.4 mm
#:
#: Under the task's settings the old cylinder was not lifted from 15 mm in and fell
#: off as the robot walked from 30. At impratio 10 one carry in two was dropped;
#: at 50, none. The pyramidal cone at 50 was not walked, and its pinch creeps about
#: as fast as the elliptic cone's at 10, the setting that dropped one. MuJoCo's
#: noslip solver would remove the creep entirely, and `mujoco_warp` raises on it.
#:
#: **Not at the tips.** Seated against the anvil itself, where the jaws meet, the
#: old cylinder was squeezed straight out of the mouth under either setting and the
#: jaws shut on nothing. That is the wedge of the jaws, not the solver.
#:
#: **What it costs the gait.** It is every contact, feet included, and the policy
#: was trained on the task's settings. Measured on that policy with an empty claw,
#: walking back 0.6 m and turning 57 degrees: 100% and 99% of the command under
#: the task's settings, 100% and 98% under these, 1 and 2 mm of drift standing.
#: Carrying something costs far more than the friction does -- 67 to 89% of the
#: walk and 45 to 71% of the turn with one of the old row in the claw.
PROP_CONE = "elliptic"

#: See `PROP_CONE`.
PROP_IMPRATIO = 50.0

# ── Where things are, and the one rule that constrains it ────────────────
#
# The props stand in a row across the robot's path, at the one distance the claw
# can be walked up to; `_row` below has the arithmetic.
#
# **Nothing solid may sit within `SPAWN_CLEAR` of the spawn.** A prop closer than
# that starts the episode intersecting the robot, and props collide, so that is
# not a cosmetic problem but an explosion on the first step. Every position below
# is checked against it by `tests/test_five_foot_objects.py`.


def _prop_spec() -> Any:
    """An empty prop spec, **in radians**.

    `mujoco.MjSpec()` starts with `compiler.degree = True`, so an `euler` given in
    radians is read as degrees and a quarter turn becomes 1.57 of one. It does not
    raise, it does not look like an error in the config, and it is only visible in
    the picture: the mug the row used to have came out as twelve wall segments all
    facing the same way. Measured on the compiled model, `euler=[0, 0, 1.5708]`
    gave `geom_quat = [0.9999, 0, 0, 0.0137]` -- a 1.57 degree rotation where 90
    was meant.

    Every asset in this repository is authored in radians (`jumper.xml` sets
    `<compiler angle="radian"/>`), so this makes the props agree with them rather
    than with MjSpec's default, whether or not a prop happens to rotate anything
    today. `tests/test_five_foot_objects.py::test_the_props_are_authored_in_radians`
    pins it.
    """
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    return spec


def _free_body(spec: Any, name: str):
    body = spec.worldbody.add_body(name=name, pos=[0.0, 0.0, 0.0])
    body.add_freejoint()
    return body


def _prop_geom(body, **kw) -> None:
    """One geom of a prop: collidable, on the prop channel, drawn in group 2.

    The contact parameters are `PROP_PRIORITY`, `PROP_CONDIM`, `PROP_FRICTION` and
    `PROP_SOLREF`, which say why. `solimp` is left at MuJoCo's default. The geom
    that collides is the geom that is drawn: there is no visual mesh over any of
    these, so what the operator aims at is what the claw meets.
    """
    kw.setdefault("condim", PROP_CONDIM)
    kw.setdefault("friction", list(PROP_FRICTION))
    kw.setdefault("priority", PROP_PRIORITY)
    kw.setdefault("solref", list(PROP_SOLREF))
    kw.setdefault("group", 2)
    body.add_geom(contype=1 | PROP_BIT, conaffinity=1 | PROP_BIT, **kw)


@dataclass(frozen=True)
class Part:
    """A piece fixed to a solid: a feature that stands proud of its shape.

    A box, with `size` its half-extents and `pos` its centre on the solid's body --
    or, when `hull` holds vertices on the body, that convex hull. `density` is in
    kg/m^3, or None for the solid's own material, and `rgba` None for its colour:
    the can's shoulder is more can, its tab is not.
    """

    name: str
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    density: float | None = None
    rgba: tuple[float, float, float, float] | None = None
    hull: tuple[tuple[float, float, float], ...] = ()

    @property
    def volume(self) -> float:
        if self.hull:
            return _hull_volume(self.hull)
        return 8.0 * self.size[0] * self.size[1] * self.size[2]

    @property
    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """`(lowest xyz, highest xyz)` on the solid's body."""
        if self.hull:
            return (tuple(min(v[i] for v in self.hull) for i in range(3)),
                    tuple(max(v[i] for v in self.hull) for i in range(3)))
        return (tuple(self.pos[i] - self.size[i] for i in range(3)),
                tuple(self.pos[i] + self.size[i] for i in range(3)))


def _hull_volume(vertices: tuple[tuple[float, float, float], ...]) -> float:
    import numpy as np
    from scipy.spatial import ConvexHull

    return float(ConvexHull(np.asarray(vertices)).volume)


@dataclass(frozen=True)
class Solid:
    """One object of the row: a shape of one material, and any `Part`s on it.

    `kind` is ``"box"``, ``"cylinder"`` or ``"hull"``. For the first two `size` is
    MuJoCo's own: half-extents for a box, `(radius, half-height)` for a cylinder
    standing on its end. For a hull it is the half-extents of its bounding box, and
    `hull` holds its vertices on the body: a convex shape that MuJoCo draws,
    collides and weighs as it is.

    `mass` is the whole object's, in kg, as weighed or as its maker states it, and
    the shape is given whatever density makes it and its parts add up to that -- so
    the mass is stated once and cannot disagree with the size.
    """

    name: str
    kind: str
    size: tuple[float, ...]
    mass: float
    rgba: tuple[float, float, float, float]
    parts: tuple[Part, ...] = ()
    hull: tuple[tuple[float, float, float], ...] = ()

    @property
    def half_extents(self) -> tuple[float, float, float]:
        """Half the shape's extent along x (toward the robot), y and z, parts excluded."""
        if self.kind == "cylinder":
            return (self.size[0], self.size[0], self.size[1])
        return (self.size[0], self.size[1], self.size[2])

    @property
    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """`(lowest xyz, highest xyz)` of the whole object, parts included, on its body.

        What the row lays an object out by: a part below the shape -- the can's base
        -- is what it stands on, and one in front of it is what the claw meets first.
        """
        lo = [-e for e in self.half_extents]
        hi = list(self.half_extents)
        for part in self.parts:
            plo, phi = part.bounds
            lo = [min(a, b) for a, b in zip(lo, plo, strict=True)]
            hi = [max(a, b) for a, b in zip(hi, phi, strict=True)]
        return tuple(lo), tuple(hi)

    @property
    def volume(self) -> float:
        """The shape's volume, parts excluded."""
        if self.kind == "box":
            return 8.0 * self.size[0] * self.size[1] * self.size[2]
        if self.kind == "cylinder":
            return math.pi * self.size[0] ** 2 * 2.0 * self.size[1]
        return _hull_volume(self.hull)

    @property
    def density(self) -> float:
        """The density of the object's own material: what is left of `mass` after the
        parts of other materials, over the shape and the parts of its own."""
        other = sum(p.volume * p.density for p in self.parts if p.density is not None)
        own = self.volume + sum(p.volume for p in self.parts if p.density is None)
        return (self.mass - other) / own


def _revolved(profile: tuple[tuple[float, float], ...],
              segments: int = 64) -> tuple[tuple[float, float, float], ...]:
    """Vertices of a solid of revolution about z, from `(radius, z)` pairs."""
    return tuple(
        (r * math.cos(2.0 * math.pi * k / segments), r * math.sin(2.0 * math.pi * k / segments), z)
        for r, z in profile
        for k in range(segments)
    )


def _rounded_bar(half: tuple[float, float, float], bevel: float, corner: float,
                 arc: int = 4) -> tuple[tuple[float, float, float], ...]:
    """Vertices of a bar with bevelled faces and rounded corners, like a bar of soap.

    `half` is `(x, y, z)`. The two large faces are the +-y ones, and every edge
    around them is bevelled by `bevel` at 45 degrees; the four edges that run along
    y are rounded to `corner`.
    """
    hx, hy, hz = half
    out = []
    for y, inset in ((hy, bevel), (hy - bevel, 0.0), (bevel - hy, 0.0), (-hy, bevel)):
        ax, az, radius = hx - inset, hz - inset, max(corner - inset, 1e-4)
        for sx, sz, start in ((1, 1, 0.0), (-1, 1, 90.0), (-1, -1, 180.0), (1, -1, 270.0)):
            cx, cz = sx * (ax - radius), sz * (az - radius)
            for k in range(arc + 1):
                a = math.radians(start + 90.0 * k / arc)
                out.append((cx + radius * math.cos(a), y, cz + radius * math.sin(a)))
    return tuple(out)


#: Oiled leather, for the notebook's belt: vegetable-tanned hide runs 0.8 to
#: 1.0 g/cm^3.
_LEATHER = 900.0

#: The notebook's belt, taken off the maker's photograph against the cover's
#: 105 mm width: a 12 mm strap, centred 81 mm up from the bottom edge, wrapping
#: the fore-edge and running 63 mm back across the front cover, with a keeper
#: loop 10 mm wide and 21 mm tall, 46 mm in from the edge. **Its thickness is not
#: in the photograph**: 3 mm of strap and a keeper standing 5 mm proud are
#: estimates, as is the leather's density.
#:
#: In the notebook's body frame, which is how it stands in the row: spine toward
#: the robot (-x), fore-edge away (+x), front cover to the robot's right (-y).
_BELT_GREEN = (0.36, 0.52, 0.34, 1.0)
_NOTEBOOK_BELT: tuple[Part, ...] = (
    Part("belt", (0.0315, 0.0015, 0.006), (0.021, -0.0165, 0.0085), _LEATHER, _BELT_GREEN),
    Part("belt_edge", (0.0015, 0.018, 0.006), (0.054, 0.0, 0.0085), _LEATHER, _BELT_GREEN),
    Part("keeper", (0.005, 0.0025, 0.0105), (0.0065, -0.0175, 0.0085), _LEATHER, _BELT_GREEN),
)

#: The can, from Ball's customer specification for the 330 ml sleek can, 202 x 204
#: (SPEC001358): a 46.18 mm stand ring, a 58.1 mm body, a 52.4 mm neck plug and a
#: 9.5 mm shoulder on a 145.4 mm open can. **Estimates**: the 8 mm the base takes
#: to reach full width, and a seamed rim 53.6 mm across at 146 mm.
#:
#: **A cylinder where it is gripped and where it rolls**, with its base and shoulder
#: as hulls on it, 0.65 mm inside the body's radius so that lying on its side it
#: touches the floor with the cylinder alone: at 0.15 mm inside, the base still
#: touched, and a can given no rolling friction slowed at 0.055 m/s^2. A hull on
#: its own rolls like the
#: polygon it is: measured under this scene's solver, the whole can as a 64-sided
#: hull decelerated on its side at 0.098 m/s^2 where its rolling friction gives
#: 0.027, still 0.060 at 256 sides, and 0.064 at 128 sides with no rolling
#: friction at all.
_CAN_RADIUS, _CAN_BODY_LO, _CAN_BODY_HI, _CAN_TOP = 0.02905, 0.008, 0.1339, 0.146
_CAN_MID = (_CAN_BODY_LO + _CAN_BODY_HI) / 2


def _can_hull(profile: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    """A hull on the can's body from `(radius, height above the stand)` pairs."""
    return _revolved(tuple((r, z - _CAN_MID) for r, z in profile))


#: The stay-on tab, lying on the lid across the can's middle: about 25 x 16 mm, as
#: tabs measure. Formed from sheet about 0.3 mm thick it weighs something like a
#: quarter of a gram, which is what the density gives this 1.5 mm box -- both
#: estimates. The shoulder's hull fills the lid's recess, so the tab lies on the
#: rim's plane rather than down inside it.
_CAN_PARTS: tuple[Part, ...] = (
    Part("base", hull=_can_hull(((0.02309, 0.0), (0.0284, _CAN_BODY_LO)))),
    Part("shoulder", hull=_can_hull(((0.0284, _CAN_BODY_HI), (0.0262, 0.1434), (0.0268, _CAN_TOP)))),
    Part("tab", (0.0125, 0.008, 0.00075), (-0.002, 0.0, _CAN_TOP - _CAN_MID + 0.00075), 420.0,
         (0.74, 0.75, 0.77, 1.0)),
)
#: The row, in the order it stands along y: the three things the real robot is
#: handed, from what their makers publish until they are weighed and measured here.
#:
#:     soap       Diaopai transparent soap, a 242 g bar (the pack is 242 g x 2).
#:                Its maker publishes a photograph and no size: a bar with
#:                bevelled faces and rounded corners, 1.48 times as long as it is
#:                wide, sized here so 242 g is soap's 1.07 g/cm^3 -- 98 x 66 x
#:                36 mm. The thickness, the 4 mm bevel and the 8 mm corners are
#:                estimates off that photograph; the embossed mark is not modelled.
#:     notebook   Raymay Davinci Grande Roroma Classic, pocket size (larger),
#:                DP3015: 105 x 145 x 30 mm and 188 g, the maker's figures -- the
#:                A6 outline, since the line has no A6
#:     can        a full 330 ml Pepsi in the sleek can, with its tab
#:                (`_CAN_PARTS`): 12 g of aluminium, an estimate, and 330 ml of
#:                cola at 1.04 g/ml make 355 g
#:
#: **Upright, each gripped across its thinnest side** -- 36, 30 and 58 mm --
#: because a claw whose mouth rides 50 to 70 mm off the floor passes over a bar of
#: soap lying flat. The can is the widest thing the claw closes on; the mouth opens
#: to 82 mm at the tips and to 58 mm 55 mm in from the hinge
#: (`tasks/jumper/five_foot/tools/claw_mouth.py`).
#:
#: **Shapes that are drawn, collided and weighed as one.** The soap is a convex hull
#: (`_rounded_bar`), which MuJoCo collides as it is and weighs by its exact volume;
#: the notebook is a box; the can is a cylinder with hulls for its base and
#: shoulder. Each gets the density that makes its object weigh what it weighs:
#: right for the soap, near enough for a notebook of paper and leather, and not
#: right for the can, whose cola does not turn with it -- a limitation stated rather
#: than modelled. What stands proud of a shape is a part: the notebook's belt of
#: leather, the can's aluminium tab.
#:
#: All three sit inside `claw.py::PAYLOAD_RANGE`, which is what the policy walking
#: under them was trained against.
SOLIDS: tuple[Solid, ...] = (
    Solid("soap", "hull", (0.033, 0.018, 0.049), 0.242, (0.98, 0.70, 0.08, 0.88),
          hull=_rounded_bar((0.033, 0.018, 0.049), bevel=0.004, corner=0.008)),
    Solid("notebook", "box", (0.0525, 0.015, 0.0725), 0.188, (0.47, 0.64, 0.44, 1.0),
          parts=_NOTEBOOK_BELT),
    Solid("can", "cylinder", (_CAN_RADIUS, (_CAN_BODY_HI - _CAN_BODY_LO) / 2), 0.355,
          (0.00, 0.33, 0.66, 1.0), parts=_CAN_PARTS),
)


def _solid_spec(solid: Solid) -> Any:
    """A free body holding `solid`'s shape and its parts, each at its own density."""
    spec = _prop_spec()
    body = _free_body(spec, solid.name)
    shape: dict[str, Any] = {}
    if solid.kind == "box":
        shape = {"type": mujoco.mjtGeom.mjGEOM_BOX, "size": list(solid.size)}
    elif solid.kind == "cylinder":
        shape = {"type": mujoco.mjtGeom.mjGEOM_CYLINDER,
                 "size": [solid.size[0], solid.size[1], 0.0]}
    elif solid.kind == "hull":
        mesh = spec.add_mesh(name=f"{solid.name}_hull")
        mesh.uservert = [c for vertex in solid.hull for c in vertex]
        # From the hull's own volume, which is what `Solid.volume` computes too.
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
        shape = {"type": mujoco.mjtGeom.mjGEOM_MESH, "meshname": mesh.name}
    else:
        raise ValueError(f"{solid.name}: no shape called {solid.kind!r}")
    own = solid.density
    _prop_geom(body, name=solid.name, density=own, rgba=list(solid.rgba), **shape)
    for part in solid.parts:
        name = f"{solid.name}_{part.name}"
        look = {"density": own if part.density is None else part.density,
                "rgba": list(solid.rgba if part.rgba is None else part.rgba)}
        if part.hull:
            mesh = spec.add_mesh(name=f"{name}_hull")
            mesh.uservert = [c for vertex in part.hull for c in vertex]
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
            _prop_geom(body, name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, **look)
        else:
            _prop_geom(body, name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(part.size),
                       pos=list(part.pos), **look)
    return spec


def _bin_spec() -> Any:
    """The drop target: a shallow open tray, four walls and a floor.

    **The rim height is the bar the robot has to clear**, and it is 20 mm on
    purpose. kk-rl-lab tried 40 mm and put every object out of reach: the claw
    lifts by pitching the trunk and tops out around 30 mm of lift. Raise it only
    together with the lift.

    Static -- no free joint -- so it stays where it is put and a missed drop
    bounces off it rather than shoving it across the room.
    """
    spec = _prop_spec()
    body = spec.worldbody.add_body(name="dropbin", pos=[0.0, 0.0, 0.0])
    wood = [0.55, 0.42, 0.30, 1.0]
    inner, t, h = BIN_INNER_HALF, BIN_WALL_T, BIN_WALL_H
    _prop_geom(body, name="bin_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[inner + t, inner + t, t / 2], pos=[0.0, 0.0, t / 2],
               rgba=wood, mass=0.4)
    for nm, sx, sy, px, py in (
        ("bin_wall_xp", t / 2, inner + t, inner + t / 2, 0.0),
        ("bin_wall_xm", t / 2, inner + t, -inner - t / 2, 0.0),
        ("bin_wall_yp", inner + t, t / 2, 0.0, inner + t / 2),
        ("bin_wall_ym", inner + t, t / 2, 0.0, -inner - t / 2),
    ):
        _prop_geom(body, name=nm, type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[sx, sy, h / 2], pos=[px, py, t + h / 2],
                   rgba=wood, mass=0.1)
    return spec


BIN_WALL_H, BIN_WALL_T = 0.020, 0.004
#: Sized off the longest object plus a margin, the way kk-rl-lab derives it -- a
#: hard-coded bin stops fitting the moment something longer joins the row.
BIN_INNER_HALF = round(
    max((hi - lo) / 2 for s in SOLIDS for lo, hi in zip(*s.bounds, strict=True)) + 0.100, 3
)


def _prop(spec_fn: Any, pos: tuple[float, float, float]) -> Any:
    """One prop, at `pos`, unrotated.

    Unrotated is a facing for the notebook, and the one its belt is laid out for:
    spine toward the robot, where every object's near face is taken from
    (`GRIP_IN`), and belt on the far edge. The mug the row once had is why that is
    written down -- turned the wrong way, it was one the claw could only knock
    over.
    """
    from mjlab.entity import EntityCfg

    return EntityCfg(spec_fn=spec_fn, init_state=EntityCfg.InitialStateCfg(pos=pos))


# ══ The highlight: which object to fetch next ════════════════════════════
#
# One object is lit at a time and the rest are dimmed, so the room says what the
# job is without a caption. **The order is fixed and the highlight only advances
# when the lit object is in the bin** -- not when it is picked up, not when it is
# put down somewhere else. So it is a queue, not a suggestion, and an object
# delivered out of turn does not skip the queue ahead of it.
#
# Ported from kk-rl-lab's `pick_place.py`, where the rule is looser: its
# `advance_target` moves to the next object *not yet delivered*, so delivering
# out of order reshuffles what is lit. Strict order is what was asked for here and
# it is also the simpler thing to read on screen -- the lit object never changes
# except by being delivered.
#
# Purely visual: `geom_rgba` has no effect on dynamics, so the highlight marks the
# next job without changing the task. It works because these props are coloured by
# `rgba` and carry no material -- `dr.geom_rgba`'s own docstring notes that a
# material would take precedence, which is why the room's furniture could not be
# highlighted this way without changing how it is coloured.

#: The order to fetch things in. **Every name here has to be a prop in the
#: scene**: the term looks each one up at construction and a stale name raises
#: while the environment is being built -- which is how the row's first version
#: was caught still listing objects it no longer had.
HIGHLIGHT_ORDER: tuple[str, ...] = ("soap", "can", "notebook")

#: What an unlit object's colour is multiplied by. 0.35 is dark enough to read as
#: "not this one" against a lit neighbour and light enough that the object is
#: still identifiable -- the point is to say which is next, not to hide the rest.
HIGHLIGHT_DIM = 0.35

#: How high an object's origin may be and still count as delivered, in metres.
#: Sized by the objects rather than by the bin: the tallest one standing on the
#: tray's floor has its origin at the floor plus half its height, and standing on
#: the rim it is a wall higher -- so the line is half a wall above the first,
#: which anything dropped in clears and anything balanced on the rim does not.
HIGHLIGHT_DELIVERED_Z = BIN_WALL_T + max(-s.bounds[0][2] for s in SOLIDS) + BIN_WALL_H / 2


class Highlight:
    """Light the next object in `HIGHLIGHT_ORDER`; dim the ones behind it.

    A `mode="step"` event, and stateful per environment: each one has its own
    place in the queue, because each one has its own robot knocking things about.

    **Writes only on a change.** Rewriting `geom_rgba` for every prop geom on
    every step of every environment is a lot of tensor traffic to say the same
    thing; the queue moves a handful of times in an episode. So the index is
    compared first and the colours are written only for the environments whose
    index actually moved.

    **The highlight lags delivery by one control step**, and that is structural
    rather than a bug to chase: `ManagerBasedRlEnv.step` applies `mode="step"`
    events *before* it calls `sim.forward()`, so the positions this reads are the
    ones from the previous step. 20 ms is not visible to an operator, and paying
    for it would mean a second forward call every step for a colour.

    The queue **catches up rather than advancing one place at a time**: the check
    runs down the order and each pass may move an environment past several
    entries. That is what should happen when things are delivered out of turn --
    put the can in before the soap, and nothing changes until the soap goes in,
    at which point the highlight jumps past both. Strict order is about what is
    *lit*, not about refusing to notice deliveries.
    """

    def __init__(self, cfg: Any, env: Any) -> None:
        import torch

        # Model fields start as a **broadcast view shared by every environment**,
        # so writing one writes all of them. mjlab catches the write and says so
        # rather than letting it through, but the fix has to happen here, once:
        # after this the field is a genuine [N, ...] tensor and each environment's
        # queue can light its own objects. Costs one model copy per environment on
        # the native backend, which is why it is not done by default.
        env.sim.expand_model_fields(("geom_rgba",))

        self._names = list(HIGHLIGHT_ORDER)
        # Geom ids per prop, in the *simulation's* numbering, resolved once.
        self._geoms = [
            torch.as_tensor(env.scene[n].indexing.geom_ids, device=env.device)
            for n in self._names
        ]
        self._base = env.sim.model.geom_rgba.clone()
        self._index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._painted = torch.full_like(self._index, -1)

    def reset(self, env_ids: Any = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        self._index[idx] = 0
        self._painted[idx] = -1        # force a repaint on the next step

    def __call__(self, env: Any, env_ids: Any = None, **kw: Any) -> None:
        import torch

        del env_ids, kw

        # Advance: the lit object is delivered when it is inside the bin's walls
        # and below the rim. Only the lit one is tested -- that is what makes the
        # order strict.
        bin_xy = env.scene["dropbin"].data.root_link_pos_w[:, :2]
        live = self._index < len(self._names)
        for i in range(len(self._names)):
            at = live & (self._index == i)
            if not bool(at.any()):
                continue
            pos = env.scene[self._names[i]].data.root_link_pos_w
            inside = (
                ((pos[:, :2] - bin_xy).abs() < BIN_INNER_HALF).all(dim=1)
                & (pos[:, 2] < HIGHLIGHT_DELIVERED_Z)
            )
            self._index = torch.where(at & inside, self._index + 1, self._index)

        changed = self._index != self._painted
        if not bool(changed.any()):
            return
        envs = torch.nonzero(changed, as_tuple=False).flatten()
        for i, geoms in enumerate(self._geoms):
            # Lit if it is the current target or already behind us in the queue.
            lit = (self._index[envs] >= i).view(-1, 1, 1)
            base = self._base[envs][:, geoms]
            faded = base.clone()
            faded[..., :3] *= HIGHLIGHT_DIM
            env.sim.model.geom_rgba[envs[:, None], geoms] = torch.where(
                lit, base, faded
            )
        self._painted[envs] = self._index[envs]


def _highlight_event() -> Any:
    from mjlab.managers.event_manager import EventTermCfg

    return EventTermCfg(func=Highlight, mode="step", params={})




# ── The row ───────────────────────────────────────────────────────────────
#
# kk-rl-lab's layout, from `pick_place.py::object_layout`: every object presents
# its grip point at the same x, and they are spread along y about the claw's own
# y. The robot walks up to the row once and then strafes between them.

#: How far from the spawn a solid thing has to stay, in metres.
#:
#: Measured rather than guessed: the furthest point of the robot's own collision
#: geometry, on `jumper.five_foot`'s nominal stance (V1.6, the arm at `LF_GRASP`),
#: is 302 mm from the trunk centre horizontally -- and it is the tip piece of
#: `LF_palm_link`, the claw's fixed jaw, i.e. exactly the part that reaches at the
#: objects. Rounded up with 48 mm to spare.
#:
#: **This is the spawn and not a box around it** because `apply_objects` pins the
#: spawn (`START`). Before that it was 0.75 m, which is
#: the task's +-0.5 m reset box plus the robot, and at that clearance nothing can
#: be put where the claw can reach it without walking.
SPAWN_CLEAR = 0.35

#: Where the mouth has to arrive, in metres ahead of the spawn. kk-rl-lab's 0.500,
#: and the same number works here for the same reason: the mouth sits at
#: x = 0.255 ahead of the trunk centre on V1.6's nominal stance (that repository
#: measures 0.260 on its own home pose, so the two robots are within 5 mm of each
#: other), and a policy holding the stance carries it
#: a little closer still, so the row is a quarter of a metre of walking away --
#: close enough to be the obvious thing in front of the robot, far enough that
#: the claw is nowhere near it at reset.
GRIP_X = 0.500

#: The row's centre line, in metres to the robot's left. **The claw's own y**, so
#: the middle object needs no strafing at all.
#:
#: **Derived, not measured, and it is the one number here that still wants a
#: policy run.** It was 0.0, from a trained policy holding the robot up
#: (`model_6999` of 2026-09-11_15-11-16, through
#: `tasks/jumper/five_foot/tools/grasp_objects.py --checkpoint`) which put the claw at y = -0.007
#: -- and that run aimed at the mouth point as it was then computed, which was
#: 48 mm to the right of where the claw actually holds things. Corrected, the
#: same claw's mouth sits at y = +0.112 on the nominal stance against +0.064 for
#: the old point, so the policy figure carries the same +0.048 and lands at
#: +0.041. V1.6's `LF_GRASP` puts the mouth at y = +0.135, 23 mm further left
#: than the pose that was measured from, and the row moves with it to +0.064.
#: Re-run that check against a V1.6 checkpoint and replace this with what it says.
#:
#: kk-rl-lab calls this `ROW_Y` and measures 0.079 on its own robot, which is the
#: same side and the same order of magnitude -- where the old 0.0 was not.
#: Either way the error is well inside one `ROW_PITCH`, which is what strafing is
#: for.
ROW_Y = 0.064

#: Lateral spacing, kk-rl-lab's. The can is 58 mm across y and the notebook 38 with
#: its belt and keeper, so neighbours stand at least 112 mm apart -- room for the
#: claw to come in beside one without touching the next.
ROW_PITCH = 0.16

#: How far **inside** its near face each object wants the mouth, in metres.
#: kk-rl-lab's `grip_in`: aiming at the near face itself leaves only a
#: fingertip's worth of the object between the jaws, and a couple of centimetres
#: in doubles it. One number for the row, because nothing in it has a feature --
#: a handle, a waist -- that wants the mouth somewhere in particular.
GRIP_IN = 0.020


def _row(grip_x: float = GRIP_X) -> dict[str, Any]:
    """The objects side by side, each standing on the floor.

    Each one's centre is placed so that the point the mouth should arrive at --
    `GRIP_IN` metres inside its near face -- lands on `grip_x`, which is what
    makes one walk up serve the whole row whatever each object's depth.

    Resting height is how far the object reaches below its own origin -- half its
    height for a plain shape, more for the can standing on its base. Get it wrong
    and the object
    either starts buried -- and is ejected on the first step, which looks like a
    random object flying across the room -- or falls from a height and bounces
    off to somewhere unintended.

    The bin is last and is not graspable -- it is what a carried object is meant
    to be dropped into, and `Highlight` scores exactly that. Its x is derived the
    way kk-rl-lab derives it: clear of the row's far edge by its own inner half
    plus a wall plus 100 mm, because the bin is sized off the longest object and
    a hard-coded x stops clearing the row the moment something longer joins it.

    `grip_x` is a parameter so a test can lay the row somewhere it must not be;
    nothing else passes it.
    """
    out: dict[str, Any] = {}
    far = -1e9
    for i, solid in enumerate(SOLIDS):
        lo, hi = solid.bounds
        x = grip_x - GRIP_IN - lo[0]
        y = ROW_Y + (i - (len(SOLIDS) - 1) / 2.0) * ROW_PITCH
        out[solid.name] = _prop(partial(_solid_spec, solid), (x, y, -lo[2]))
        far = max(far, x + hi[0])
    bin_x = far + BIN_INNER_HALF + BIN_WALL_T + 0.100
    out["dropbin"] = _prop(_bin_spec, (bin_x, ROW_Y, 0.0))
    return out


# ══ Putting the row into this task ═══════════════════════════════════════
#
# This was a scene, `--scene objects`, until the task moved everything it needs
# inside its own directory. The scene registry would have needed five fields that
# only this row used -- contact capacity, the friction cone, a fixed spawn -- so
# it is the task's own flag now, `--objects` (`__init__.py::cli_args`), and
# `apply_objects` does the whole of what the scene did, in the order the registry
# did it. `--scene` still picks the look on top: a scene applied after this
# replaces the lights and the ground's textures, and leaves the row alone.

#: The props the claw can pick up: the solids, and not the bin.
GRASPABLE: tuple[str, ...] = tuple(solid.name for solid in SOLIDS)

#: Contact slots the row needs, when it needs more than the task reserved.
#:
#: **Only the warp backend preallocates**, so this is invisible on native and a
#: hard failure on the GPU: `mujoco_warp.put_data` raises `nconmax overflow
#: (nconmax must be >= N)` while the environment is being built. Measured on
#: `jumper.five_foot` with the row, which adds seven colliding bodies: 404 contacts
#: against the jumper skeleton's 128. Taken as a floor, never a ceiling: a task that
#: reserved more had a reason.
OBJECTS_NCONMAX = 1024

#: Constraint rows, the same story. Beside `OBJECTS_NCONMAX` because the two move
#: together: every contact contributes rows, so a row that overflows one is
#: usually close to overflowing the other.
OBJECTS_NJMAX = 4096

#: A fixed spawn, and a robot that does nothing until it is driven: (x, y, yaw).
#:
#: This is kk-rl-lab's replay, which is where a claw is actually operated:
#: `deploy/hexa_5d/backends/mujoco/play_mujoco.py` seats the robot at the origin
#: facing +x and starts with `cmd = [0, 0, 0]` -- the command only ever moves when
#: a key is pressed, and its `--random` flag is what asks for the sampler instead.
#: The objects sit half a metre ahead of that, so the episode opens with the robot
#: standing a little in front of the row, looking at it.
#:
#: Both halves are needed and they fail differently. Without the fixed spawn the
#: robot starts anywhere in the task's 1 m reset box at any heading, so the row is
#: behind it about as often as in front. Without the stillness it walks off on a
#: sampled command before the operator's first keypress.
#:
#: The three are **offsets from the robot's own initial state**, which is what
#: mjlab's `reset_root_state_uniform` adds them to; all zero is therefore "exactly
#: where the asset says", the origin facing +x.
START: tuple[float, float, float] = (0.0, 0.0, 0.0)

#: The command fields that, set to 1.0, make every environment command nothing.
_STILL_FIELDS = ("rel_standing_envs", "rel_neutral_envs")

#: The headlight, as (ambient, diffuse, specular). Flat ground, one warm light:
#: what this is for is the objects, and a plain ground is both the cheapest to draw
#: and the easiest to see a small object against.
HEADLIGHT = ((0.20, 0.20, 0.20), (0.45, 0.44, 0.42), (0.10, 0.10, 0.10))


def _lights() -> tuple[Any, ...]:
    from mjlab.utils import spec_config as sc

    return (
        sc.LightCfg(
            name="key", type="directional", pos=(-0.6, -0.8, 2.0),
            dir=(0.25, 0.35, -1.0), castshadow=True,
            diffuse=(0.70, 0.68, 0.64), specular=(0.12, 0.12, 0.12),
            ambient=(0.0, 0.0, 0.0),
        ),
        sc.LightCfg(
            name="fill", type="directional", pos=(2.4, 1.2, 1.2),
            dir=(-0.5, -0.3, -0.8), castshadow=False,
            diffuse=(0.22, 0.22, 0.24), specular=(0.0, 0.0, 0.0),
            ambient=(0.0, 0.0, 0.0),
        ),
    )


def _place_props(env_cfg: Any, props: dict[str, Any]) -> None:
    """Give each prop a reset event, so it lands beside its own robot.

    Adding the entity is not enough and the shortfall is invisible in the config:
    mjlab applies `env_origins` inside `reset_root_state_uniform`, and an entity
    with no reset event stays at (0, 0, 0) in **every** environment. The pose range
    is **empty**, not the prop's position: the entity's own `init_state` already
    supplies that, and passing it again lands the prop at twice the offset. What
    the event contributes is the env_origin and the return to that spot at the
    start of every episode. The same wiring the scene registry gives a scene's
    props (`scenes/registry.py::_place_props`), written here so the task does not
    lean on a private helper of the framework.
    """
    from mjlab.envs.mdp import events as mdp_events
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    for name in props:
        env_cfg.events[f"reset_prop_{name}"] = EventTermCfg(
            func=mdp_events.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg(name),
            },
        )


def _set_headlight(scene_cfg: Any) -> None:
    """Set the headlight through `SceneCfg.spec_fn`, which runs last.

    Nothing earlier survives: the value that reaches the model comes from mjlab's
    own `scene/scene.xml`, the parent spec, so a `<visual>` block in an attached
    entity is overridden without a word. Any callback already installed is kept and
    run first. The scene registry does the same for a scene's headlight
    (`scenes/registry.py::_install_spec_edits`).
    """
    previous = getattr(scene_cfg, "spec_fn", None)
    ambient, diffuse, specular = HEADLIGHT

    def edit(spec: Any) -> None:
        if previous is not None:
            previous(spec)
        spec.visual.headlight.ambient = list(ambient)
        spec.visual.headlight.diffuse = list(diffuse)
        spec.visual.headlight.specular = list(specular)

    scene_cfg.spec_fn = edit


def apply_objects(env_cfg: Any) -> None:
    """Put the row in front of the robot, and the world it needs around it.

    **Physical.** The props have mass and collide, and the friction model changes
    for every contact in the world, the robot's feet included (`PROP_CONE`): a
    policy replayed here walks on friction it was not trained with. It says so on
    every run, because nothing in the observation or the reward would.
    """
    import warnings

    events = _highlight_event_terms()
    clashes = sorted(set(events) & set(env_cfg.events))
    if clashes:
        raise ValueError(f"--objects would overwrite this task's event terms {clashes}")
    env_cfg.events.update(events)

    # Never downward: a task that reserved more had a reason.
    env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax or 0), OBJECTS_NCONMAX)
    env_cfg.sim.njmax = max(int(env_cfg.sim.njmax or 0), OBJECTS_NJMAX)
    env_cfg.sim.mujoco.cone = PROP_CONE
    env_cfg.sim.mujoco.impratio = PROP_IMPRATIO

    props = _row()
    entities = env_cfg.scene.entities
    clashes = sorted(set(props) & set(entities))
    if clashes:
        raise ValueError(f"--objects would overwrite this task's entities {clashes}")
    entities.update(props)
    _place_props(env_cfg, props)
    warnings.warn(
        f"--objects adds {len(props)} physical prop(s): {', '.join(sorted(props))}. "
        f"These have mass and they collide, and the friction model is now "
        f"{PROP_CONE} at impratio {PROP_IMPRATIO:g} for every contact -- a policy "
        f"trained without them is being replayed on physics it did not see. Pass "
        f"--num_envs 1 to look at one robot: every environment gets its own row.",
        RuntimeWarning,
        stacklevel=2,
    )

    _place_the_robot(env_cfg)

    _set_headlight(env_cfg.scene)
    terrain = env_cfg.scene.terrain
    terrain.textures = ()
    terrain.materials = ()
    terrain.lights = _lights()


def _highlight_event_terms() -> dict[str, Any]:
    return {"objects_highlight": _highlight_event()}


def _place_the_robot(env_cfg: Any) -> None:
    """Pin the spawn and command nothing -- see `START`.

    The spawn is pinned by narrowing the task's own `reset_base` rather than by
    replacing the term: `z` keeps whatever drop the task chose (it is a settle,
    not a position) and so does every other parameter.
    """
    x, y, yaw = START
    reset = env_cfg.events.get("reset_base")
    if reset is None:
        raise ValueError("--objects pins the spawn, and this config has no 'reset_base'")
    pose = dict(reset.params.get("pose_range", {}))
    pose["x"] = (x, x)
    pose["y"] = (y, y)
    pose["yaw"] = (yaw, yaw)
    reset.params["pose_range"] = pose

    stilled = []
    for name, command in env_cfg.commands.items():
        for field_name in _STILL_FIELDS:
            if hasattr(command, field_name):
                setattr(command, field_name, 1.0)
                stilled.append(f"{name}.{field_name}")
    if not stilled:
        raise ValueError(
            f"--objects wants the robot to stand still, but none of the command "
            f"terms {sorted(env_cfg.commands)} has any of {list(_STILL_FIELDS)}. "
            f"A command term that cannot be told to command nothing walks the robot "
            f"away from the row on the first step."
        )
    print(
        f"[objects] the robot starts at ({x:+.2f}, {y:+.2f}) yaw {yaw:+.2f}, "
        f"commanding nothing until it is driven ({', '.join(stilled)})"
    )
