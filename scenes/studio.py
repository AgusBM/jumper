"""Neutral grey, soft light, no horizon -- for looking at the robot.

Built for screenshots and video, where the question is what the mechanism is
doing. The checkerboard that makes motion legible during training is a
distraction there, and a bright sky pulls the eye away from a small dark robot.

Two lights rather than one: a key from the front-left and a dimmer fill from the
right with shadows off, so the underside of the body and the inner legs are not
solid black. A single directional light -- mjlab's default -- is fine for judging
gait and poor for seeing shape.
"""

from __future__ import annotations

from .registry import GROUND, Headlight, Scene


def scene() -> Scene:
    from mjlab.utils import spec_config as sc

    return Scene(
        headlight=Headlight(ambient=(0.16, 0.16, 0.17), diffuse=(0.32, 0.32, 0.34)),
        textures=(
            sc.TextureCfg(
                name="studio_floor", type="2d", builtin="checker",
                rgb1=(0.42, 0.42, 0.44), rgb2=(0.36, 0.36, 0.38),
                width=512, height=512,
            ),
            sc.TextureCfg(
                name="studio_sky", type="skybox", builtin="gradient",
                rgb1=(0.30, 0.31, 0.34), rgb2=(0.14, 0.14, 0.16),
                width=256, height=256,
            ),
        ),
        materials=(
            sc.MaterialCfg(
                name="studio_floor", texture="studio_floor", texuniform=True,
                texrepeat=(12.0, 12.0), reflectance=0.05, geom_names_expr=GROUND,
            ),
        ),
        lights=(
            sc.LightCfg(name="key", type="directional", pos=(-2.0, -2.0, 3.0),
                        dir=(0.5, 0.5, -1.0), diffuse=(0.75, 0.75, 0.75)),
            sc.LightCfg(name="fill", type="directional", pos=(3.0, 1.0, 2.0),
                        dir=(-0.6, -0.2, -1.0), diffuse=(0.30, 0.30, 0.33),
                        castshadow=False),
        ),
    )
