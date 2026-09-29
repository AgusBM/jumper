"""mjlab's own look, written out rather than inherited.

Having the baseline as a scene means `--scene default` is something you can name
and diff against, instead of "whatever happens when you pass nothing". The values
are mjlab's defaults from `terrains/terrain_entity.py`; if that file changes,
`tests/test_scenes.py` notices.
"""

from __future__ import annotations

from .registry import GROUND, Headlight, Scene


def scene() -> Scene:
    from mjlab.utils import spec_config as sc

    return Scene(
        headlight=Headlight(ambient=(0.30, 0.30, 0.30), diffuse=(0.60, 0.60, 0.60)),
        textures=(
            sc.TextureCfg(
                name="groundplane", type="2d", builtin="checker", mark="edge",
                rgb1=(0.2, 0.3, 0.4), rgb2=(0.1, 0.2, 0.3),
                markrgb=(0.8, 0.8, 0.8), width=300, height=300,
            ),
        ),
        materials=(
            sc.MaterialCfg(
                name="groundplane", texture="groundplane", texuniform=True,
                texrepeat=(4.0, 4.0), reflectance=0.2, geom_names_expr=GROUND,
            ),
        ),
        lights=(sc.LightCfg(name="sun", pos=(0.0, 0.0, 1.5), type="directional"),),
    )
