"""A coarse anti-slip rubber mat: as much grip as the ground can give.

The other end of `ice.py`, and built the same way -- **friction is the only thing
either scene changes**, through `registry.set_ground_contact`, which carries the
`priority` mechanism and the reason it is needed. Read that and `ice.py`'s module
docstring for why a low-priority ground would silently do nothing.

The pair is the point. A policy that holds up on both has learned to walk rather
than to exploit one coefficient, and the two scenes bracket what the robot will
ever stand on:

    ice     0.03      wet ice, the lowest surface there is
    default 1.2       the feet's own, which is what a bare floor gives
    rubber  1.8       this scene

## Why a mat is worth a scene at all, when more grip sounds free

It is not free, and that is the interesting half. Friction the gait does not need
is friction it can *lean on*: a foot that cannot slip lets the policy push
sideways as hard as it likes, and the resulting gait falls over on anything less.
So this scene is most useful as the **training** end of a pair with `ice` -- or as
the thing that reveals a policy trained only here, which is the failure it exists
to make visible.

## What is deliberately not modelled

A real anti-slip mat is **compliant** -- 5 to 10 mm of EPDM gives underfoot, and
that changes landing forces as much as the grip does. This scene leaves contact
stiffness exactly as it is, because a scene that moved friction *and* softness
would produce a result attributable to neither. `soft.py` is the compliance axis
and can be composed with this one by whoever wants both; the numbers for a mat
would be somewhere between `soft`'s mud and rigid ground.

`foot_friction` domain randomisation is **inert here**, as on `ice` and for the
same reason: the event writes to the foot geoms, which this scene has outranked,
so every sample is discarded before the solver sees it. A run here trains against
one fixed friction, and `constants.JUMPER_FOOT_FRICTION_RANGE` does not describe it.
"""

from __future__ import annotations

from typing import Any

from .registry import GROUND, Headlight, Scene, set_ground_contact

#: Sliding friction, and the one number this scene exists for.
#:
#: Commonly reported coefficients for rubber, which is what both surfaces are
#: here -- the feet and the mat:
#:
#:     rubber on dry concrete   0.7 - 1.0
#:     rubber on rubber         1.0 - 1.5
#:     textured anti-slip mat   1.2 - 1.8     <- this scene
#:
#: 1.8 is the top of that band, which is the point: the feet already carry 1.2
#: (`constants._FOOT_FRICTION`), so anything less than about 1.5 would be a scene
#: whose effect is inside the noise of the friction randomisation it replaces.
#: **A coefficient above 1 is not an error** -- it is the ratio of tangential to
#: normal force at the contact, and a soft rubber pair exceeds 1 by deforming
#: into the tread rather than sliding over it.
#:
#: Not measured on this robot. A literature value, like `ice.py`'s, and worth
#: replacing the first time somebody drags a loaded foot across a real mat with a
#: force gauge -- which is a far easier measurement than the ice one.
RUBBER_SLIDING = 1.8

#: Torsional and rolling friction, kept in the feet's own proportion to sliding.
#:
#: `_FOOT_FRICTION` is (1.2, 0.01, 0.01) -- torsional at 1/120th of sliding -- and
#: the same ratio at 1.8 gives 0.015. The ratio is the assumption; what argues for
#: it here is that a *coarse* mat resists twisting by the same mechanism it resists
#: sliding, the tread interlocking with the foot, so scaling all three together is
#: closer than raising sliding alone and leaving a foot free to pivot on grippy
#: ground.
RUBBER_TORSIONAL = 0.015
RUBBER_ROLLING = 0.015


def _lay_the_mat(spec: Any) -> None:
    """The mat's friction on every ground geom, at the priority that makes it count.

    `registry.set_ground_contact` holds the three traps and defaults `solref` /
    `solimp` to the feet's own, so contact stiffness is untouched and **friction
    is the only thing this scene moves**.
    """
    set_ground_contact(
        spec, "rubber",
        friction=(RUBBER_SLIDING, RUBBER_TORSIONAL, RUBBER_ROLLING),
    )


def scene() -> Scene:
    from mjlab.utils import spec_config as sc

    return Scene(
        # A mat is an indoor surface, so the light is flat and from above rather
        # than the low hard sun the outdoor scenes use: no long shadows, and the
        # tread read through its own contrast instead. Slightly warm, because a
        # bare grey under a neutral light photographs as a void.
        headlight=Headlight(ambient=(0.22, 0.21, 0.20), diffuse=(0.42, 0.41, 0.40)),
        textures=(
            # A tight checker is a tread pattern at this scale. `texrepeat` below
            # puts the cell at roughly a centimetre, which is a third of the foot
            # -- fine enough to read as texture, coarse enough that the eye can
            # see the foot move across it. The two greys are far enough apart to
            # be visible and close enough not to strobe when the camera moves.
            sc.TextureCfg(
                name="rubber_floor", type="2d", builtin="checker",
                rgb1=(0.20, 0.20, 0.21), rgb2=(0.27, 0.27, 0.28),
                width=512, height=512,
            ),
            sc.TextureCfg(
                name="rubber_sky", type="skybox", builtin="gradient",
                rgb1=(0.30, 0.30, 0.32), rgb2=(0.17, 0.17, 0.18),
                width=256, height=256,
            ),
        ),
        materials=(
            # **`reflectance=0.0` is the visual claim.** Rubber is the matte
            # surface; anything above zero reads as wet or as painted concrete,
            # which is the opposite of what this scene is for -- and `ice.py`'s
            # 0.4 is exactly the contrast that makes the pair legible side by
            # side.
            sc.MaterialCfg(
                name="rubber_floor", texture="rubber_floor", texuniform=True,
                texrepeat=(60.0, 60.0), reflectance=0.0, geom_names_expr=GROUND,
            ),
        ),
        lights=(
            sc.LightCfg(name="overhead", type="directional", pos=(0.0, 0.0, 4.0),
                        dir=(-0.15, -0.15, -1.0), diffuse=(0.70, 0.69, 0.67)),
            sc.LightCfg(name="fill", type="directional", pos=(2.0, -2.0, 2.0),
                        dir=(-0.5, 0.5, -1.0), diffuse=(0.22, 0.22, 0.23),
                        castshadow=False),
        ),
        decorate=_lay_the_mat,
    )
