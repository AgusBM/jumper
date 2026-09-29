#!/usr/bin/env python3
"""Generate the football pitch's ground textures: mown grass, and the sand around it.

Run from the repository root:

    python assets/scenes/football/tools/make_ground.py

Two scales in one image, because a pitch reads at two distances. Close up it is
individual blades -- fine, high-contrast speckle. From the stands it is the mown
bands, where the roller has laid the grass towards or away from you and the same
grass looks two different greens.

**The pitch markings are not here and cannot be.** The ground is an infinite
plane with a *tiling* texture, so anything drawn once -- a centre circle, a
penalty box -- would repeat every few metres across the whole world. The markings
are geometry instead; see `scenes/football.py`.

The sand is what the infinite plane wears. The grass is a finite slab laid on top
of it at pitch size, so the turf stops at the touchline and everything beyond is
bare ground -- which is what a pitch in a field looks like, and what an infinite
lawn does not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

RES = 512

#: The two greens of a mown pitch. They are the same grass: the difference is
#: which way the blades are lying, so the pair has to be close. Too much contrast
#: and it reads as paint rather than as mowing.
LIGHT = np.array([0.30, 0.52, 0.20])
DARK = np.array([0.22, 0.42, 0.15])

#: Blade grain: fine, and stronger than it looks reasonable on paper. Grass is
#: not a flat colour at any distance, and a smooth green plane reads as felt.
GRAIN = 0.055
SEED = 5


#: The ground beyond the touchline: dry, pale, and finer-grained than the grass
#: so the edge of the pitch reads as a change of surface rather than of colour.
SAND = np.array([0.66, 0.57, 0.43])
SAND_GRAIN = 0.045


def _sand(rng: np.random.Generator) -> np.ndarray:
    img = np.broadcast_to(SAND, (RES, RES, 3)).copy()
    img = img + rng.normal(0.0, SAND_GRAIN, size=(RES, RES, 1))
    return img


def main() -> None:
    rng = np.random.default_rng(SEED)
    # Two stripes across the tile, so one repeat of the texture is two mown bands.
    x = np.arange(RES)[None, :].repeat(RES, 0)
    stripe = (x < RES // 2)[..., None]
    img = np.where(stripe, LIGHT, DARK).astype(float)

    # Grain, plus a slower mottle so the stripes are not perfectly uniform.
    img = img + rng.normal(0.0, GRAIN, size=(RES, RES, 1))
    coarse = rng.normal(0.0, 1.0, size=(RES // 16, RES // 16, 1))
    coarse = np.asarray(
        Image.fromarray(((coarse - coarse.min()) / np.ptp(coarse) * 255).astype(np.uint8)[..., 0])
        .resize((RES, RES), Image.BICUBIC)
    )[..., None] / 255.0
    img = img + (coarse - 0.5) * 0.05

    here = Path(__file__).resolve().parent.parent
    for name, data in (("grass", img), ("sand", _sand(rng))):
        path = here / f"{name}.png"
        Image.fromarray((np.clip(data, 0, 1) * 255).astype(np.uint8)).save(path)
        print(f"  wrote {path.name}")


if __name__ == "__main__":
    main()
