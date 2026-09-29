"""No textures, no materials -- the ground is flat colour.

Two uses. Domain randomisation over `geom_rgba` needs materials off, because a
material wins over the rgba the randomiser writes and the randomisation then has
no visible effect at all. And it is the cheapest thing to draw, which matters when
the live viewer is the bottleneck.

One flat light with shadows off: shadows are the expensive part of rendering and
the point here is to be cheap.
"""

from __future__ import annotations

from .registry import Headlight, Scene


def scene() -> Scene:
    from mjlab.utils import spec_config as sc

    return Scene(
        headlight=Headlight(ambient=(0.30, 0.30, 0.30), diffuse=(0.60, 0.60, 0.60)),
        textures=(),
        materials=(),
        lights=(
            sc.LightCfg(name="flat", type="directional", pos=(0.0, 0.0, 4.0),
                        dir=(0.0, 0.0, -1.0), castshadow=False,
                        diffuse=(0.8, 0.8, 0.8), ambient=(0.35, 0.35, 0.35)),
        ),
    )
