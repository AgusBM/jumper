"""Blue sky, warm ground, one hard sun -- the outdoor look.

For video meant to read as a robot outdoors. The sun is a single directional
light with shadows on, low enough that legs cast shadows onto the ground, which
is most of what makes contact look real; the ambient term is tinted towards the
sky so shadowed surfaces go blue rather than black, the way daylight actually
behaves.
"""

from __future__ import annotations

from .registry import GROUND, Headlight, Scene


def scene() -> Scene:
    from mjlab.utils import spec_config as sc

    return Scene(
        headlight=Headlight(ambient=(0.14, 0.15, 0.18), diffuse=(0.26, 0.27, 0.30)),
        textures=(
            sc.TextureCfg(
                name="daylight_ground", type="2d", builtin="checker", mark="edge",
                rgb1=(0.62, 0.53, 0.40), rgb2=(0.54, 0.45, 0.33),
                markrgb=(0.72, 0.66, 0.55), width=512, height=512,
            ),
            sc.TextureCfg(
                name="daylight_sky", type="skybox", builtin="gradient",
                rgb1=(0.32, 0.52, 0.82), rgb2=(0.86, 0.92, 0.98),
                width=512, height=512,
            ),
        ),
        materials=(
            sc.MaterialCfg(
                name="daylight_ground", texture="daylight_ground", texuniform=True,
                texrepeat=(8.0, 8.0), reflectance=0.0, geom_names_expr=GROUND,
            ),
        ),
        lights=(
            sc.LightCfg(name="sun", type="directional", pos=(-3.0, -2.0, 4.0),
                        dir=(0.55, 0.35, -1.0), castshadow=True,
                        diffuse=(0.95, 0.92, 0.85), specular=(0.3, 0.3, 0.3),
                        ambient=(0.22, 0.26, 0.34)),
        ),
    )
