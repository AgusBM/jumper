"""Wet ice: the ground looks the same and the feet cannot hold it.

**This scene changes the physics, not the look**, and the only thing it changes is
friction. Softness, contact stiffness and the ground's shape are all left as they
would be without it, so a run on this scene differs from a run on `default` in
one variable.

## The trap is `priority`, and it is the whole mechanism

MuJoCo mixes two geoms' contact parameters only when their `priority` is equal.
When it differs, **the higher-priority geom's parameters are used outright and the
other's are discarded**. `constants.py` gives the feet `priority=1` so that a
foot's friction is the foot's, whatever it stands on -- which is exactly what
would make a low-friction ground do nothing at all. The terrain is set to
`priority=2` to outrank it. `scenes/soft.py` records the same finding; this is the
scene where getting it wrong would be hardest to spot, because a ground that fails
to be slippery just looks like ordinary ground.

Checked in the compiled model rather than assumed. Of 73 geoms, exactly six carry
`priority=1`, and they are the six feet:

    robot/LF_palm_pad_b_link_meshcol   friction [1.2, 0.01, 0.01]
    robot/RF_palm_pad_b_link_meshcol   friction [1.2, 0.01, 0.01]
    robot/{L,R}{M,R}_foot_tip_link_meshcol   the same

everything else is 0. With this scene the terrain geom reads
`priority=2 friction=[0.03, 0.00025, 0.00025]`, so it outranks them.

**The front feet are `palm_pad_b`, not `finger_tip`.** Reading the friction off the
first foot-shaped name gives `LF_finger_tip`, which is priority 0 and carries the
default friction -- it is not a contact body at all (`build_jumper.py`: the front
pair walks on the rear jaw pad). Walked into while checking this, and it reads as
"the feet are priority 0, so the trap above is imaginary".

And `priority` takes the **whole** contact parameter set with it, not only the
field you meant to change -- so `registry.set_ground_contact` defaults `solref`
and `solimp` to the feet's own numbers. Without them the terrain's contact
stiffness would win too, and the scene would change two things while claiming
one. That helper holds this mechanism for both friction scenes; the numbers here
are all this file contributes.

## What this costs, and it is not small

`foot_friction` domain randomisation is **inert on this scene**. The event writes
to the foot geoms, which this scene has outranked, so every sample it draws is
discarded before the solver sees it. A run here trains against one fixed friction.
That is arguably what an ice scene is for -- the point is a specific, known,
very low number rather than a range -- but it means a policy trained here has seen
no friction variation at all, and `constants.JUMPER_FOOT_FRICTION_RANGE` is not
describing this run. Moving the randomisation onto the terrain geom is the fix if
that matters; copying a constant is not.
"""

from __future__ import annotations

from typing import Any

from .registry import GROUND, Headlight, Scene, set_ground_contact

#: Sliding friction, and the one number this scene exists for.
#:
#: Ice is the lowest friction a robot meets outdoors, and **ice with a film of
#: water on it is the lowest state of ice**: the film lubricates
#: hydrodynamically, which is why a thaw is more treacherous than a hard freeze.
#: Commonly reported coefficients, for rubber and for steel sliders:
#:
#:     dry concrete        0.7 - 1.0
#:     hard ice, cold      0.05 - 0.10
#:     ice near 0 C, wet   0.02 - 0.05     <- this scene
#:
#: 0.03 is the middle of that band. For scale, the feet carry 1.2
#: (`constants._FOOT_FRICTION`), so this is **1/40th** of the ground the robot
#: normally stands on, and a tripod stance can no longer resist a sideways push
#: with anything but geometry.
#:
#: Not measured on this robot, and there is nothing to measure it against -- the
#: hexapod has never been on ice. It is a literature value, which is the best this
#: can be until there is a floor and a force gauge.
ICE_SLIDING = 0.03

#: Torsional and rolling friction, kept in the feet's own proportion to sliding.
#:
#: `_FOOT_FRICTION` is (1.2, 0.01, 0.01), so torsional is 1/120th of sliding
#: there; the same ratio at 0.03 gives 0.00025. **This is an assumption, not a
#: measurement**: the justification is only that one mechanism -- a lubricating
#: film -- reduces every tangential resistance at the contact, so the ratio the
#: feet were given is a better guess than leaving torsional at a value that would
#: then be a third of sliding. A foot on wet ice should be free to spin, and at
#: 0.01 it would not have been.
ICE_TORSIONAL = 0.00025
ICE_ROLLING = 0.00025


def _make_it_ice(spec: Any) -> None:
    """Ice's friction on every ground geom, at the priority that makes it count.

    `registry.set_ground_contact` carries the three traps (the priority, the whole
    parameter set coming with it, and `spec.geoms`) and defaults `solref` /
    `solimp` to the feet's own -- so the contact is as hard as it always was and
    **friction is the only thing this scene moves**.
    """
    set_ground_contact(
        spec, "ice", friction=(ICE_SLIDING, ICE_TORSIONAL, ICE_ROLLING)
    )


def scene() -> Scene:
    from mjlab.utils import spec_config as sc

    return Scene(
        # Low ambient and one hard low sun. Ice reads as ice through its
        # specular highlight and the long shadow across it; a bright overhead
        # ambient washes both out and leaves a flat pale floor that could be
        # anything. The same reasoning as `beach.py`'s note on the headlight.
        headlight=Headlight(ambient=(0.13, 0.14, 0.16), diffuse=(0.30, 0.32, 0.36)),
        textures=(
            # Faint, large-scale mottling rather than a checker: refrozen ice has
            # cloudy patches where air is trapped, and the pattern has to be much
            # larger than a foot or it reads as texture on a floor instead of as
            # depth within a slab. Two blues a hair apart -- any more contrast and
            # it looks like tiling.
            sc.TextureCfg(
                name="ice_floor", type="2d", builtin="checker",
                rgb1=(0.74, 0.82, 0.88), rgb2=(0.70, 0.79, 0.86),
                width=512, height=512,
            ),
            sc.TextureCfg(
                name="ice_sky", type="skybox", builtin="gradient",
                rgb1=(0.62, 0.72, 0.82), rgb2=(0.82, 0.88, 0.92),
                width=256, height=256,
            ),
        ),
        materials=(
            # **`reflectance` is the whole visual trick.** A wet surface is
            # defined by what it mirrors, not by its colour, and 0.4 is high
            # enough to pick up the sky and the robot without turning the floor
            # into a mirror the gait cannot be read against. `texrepeat` is low
            # so the mottling stays large.
            sc.MaterialCfg(
                name="ice_floor", texture="ice_floor", texuniform=True,
                texrepeat=(3.0, 3.0), reflectance=0.4, geom_names_expr=GROUND,
            ),
        ),
        lights=(
            sc.LightCfg(name="sun", type="directional", pos=(-3.0, -2.0, 2.0),
                        dir=(0.6, 0.4, -1.0), diffuse=(0.85, 0.87, 0.92)),
        ),
        decorate=_make_it_ice,
    )
