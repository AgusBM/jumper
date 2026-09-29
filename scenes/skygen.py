#!/usr/bin/env python3
"""Write a MuJoCo cube-map skybox from a function of direction.

Shared by the scenes that generate their own sky. It lives in `scenes/` rather
than in `tools/` because scene modules import it and they are installed; `tools/`
is not, and is only importable when the working directory happens to be the
repository root. What is here is the part that
was expensive to work out and must not be copied: which world direction each of
MuJoCo's six faces covers, how it samples the image inside a face, and noise that
is continuous across the seams. A scene supplies only its colours.

## The face table, measured rather than assumed

Both halves of it fail silently -- the cube compiles, the sky looks like a sky,
and it is simply wrong. Measured by rendering a cube with one flat colour per face
and looking along each axis:

    right -> +X      up   -> +Z      front -> -Y
    left  -> -X      down -> -Z      back  -> +Y

Left and right read normally; front and back are mirrored from the obvious
reading. Then, by encoding u and v as colour channels and reading the rendered
corners, the image axes: on all four side faces u increases towards **screen-left**
and v downwards.

Both measurements rest on one fact that is easy to get backwards, and getting it
backwards makes everything self-consistent while rotated half a turn:
**`mjvCamera.azimuth` is the azimuth of the viewing direction**, not of the
camera's position. `azimuth=0` looks along +X.

## Noise on the direction, never on the face

`value_noise` and `fbm` take the direction vector. Neighbouring faces then agree
at their shared edge by construction rather than by getting the axes right, which
is what keeps clouds and stars from tearing at the seams.

Resolution sets what can be drawn: one face is `RES` texels across 90 degrees, so
at 512 a texel is 0.18 degrees. Anything finer than a few texels -- a real sun at
0.27 degrees, the top octave of a noise field -- comes out as speckle or vanishes
entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

#: Texels per face edge, spanning 90 degrees.
RES = 512

#: Forward, and the directions of increasing u and v within the image. See the
#: measurements in the module docstring.
FACES: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    "right": (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, -1.0])),
    "left": (np.array([-1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, -1.0])),
    "up": (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, -1.0, 0])),
    "down": (np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
    "front": (np.array([0, -1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, -1.0])),
    "back": (np.array([0, 1.0, 0]), np.array([-1.0, 0, 0]), np.array([0, 0, -1.0])),
}

#: The order MuJoCo's `cubefiles` expects.
FACE_ORDER = ("right", "left", "up", "down", "front", "back")


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def hash01(cell: np.ndarray, seed: int) -> np.ndarray:
    """A stable pseudo-random value in [0, 1) per integer lattice cell."""
    n = (cell[..., 0] * 374761393 + cell[..., 1] * 668265263
         + cell[..., 2] * 1274126177 + seed * 144665) & 0x7FFFFFFF
    n = (n ^ (n >> 13)) * 1274126177 & 0x7FFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFFFF) / float(0xFFFFFF)


def value_noise(p: np.ndarray, seed: int) -> np.ndarray:
    """Lattice value noise on 3D points, smoothly interpolated."""
    i = np.floor(p).astype(np.int64)
    f = p - i
    w = f * f * (3.0 - 2.0 * f)

    def corner(dx: int, dy: int, dz: int) -> np.ndarray:
        return hash01(i + np.array([dx, dy, dz]), seed)

    def lerp(a, b, t):
        return a + (b - a) * t

    x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), w[..., 0])
    x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), w[..., 0])
    x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), w[..., 0])
    x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), w[..., 0])
    return lerp(lerp(x00, x10, w[..., 1]), lerp(x01, x11, w[..., 1]), w[..., 2])


def fbm(p: np.ndarray, seed: int, octaves: int = 3) -> np.ndarray:
    """Fractal sum: octaves of the same noise, each finer and fainter.

    Three is not a stylistic choice. At `RES` 512 and the frequencies a cloud
    field wants, a fourth octave lands near six texels per cycle and the result
    reads as sand rather than as cloud.
    """
    total = np.zeros(p.shape[:-1])
    amplitude, weight = 1.0, 0.0
    for k in range(octaves):
        total += amplitude * value_noise(p * (2.0**k), seed + k)
        weight += amplitude
        amplitude *= 0.5
    return total / weight


def disc(dirs: np.ndarray, towards: np.ndarray, radius_deg: float) -> np.ndarray:
    """A hard-edged disc of the given angular radius around a direction.

    Used for the sun. Below about a degree there is no room for a soft edge --
    the falloff is spent on the one or two texels the disc occupies and what
    comes out is a grey dot -- so the edge is hard and any softening is the
    caller's business.
    """
    ang = np.degrees(np.arccos(np.clip(np.sum(dirs * towards, axis=-1, keepdims=True), -1.0, 1.0)))
    return (ang < radius_deg).astype(float)


def angle_to(dirs: np.ndarray, towards: np.ndarray) -> np.ndarray:
    """Angle in degrees between each direction and a reference direction."""
    return np.degrees(
        np.arccos(np.clip(np.sum(dirs * towards, axis=-1, keepdims=True), -1.0, 1.0))
    )


def write_cube(out_dir: Path, colour: Callable[[np.ndarray], np.ndarray],
               prefix: str = "sky") -> list[Path]:
    """Render the six faces with `colour(dirs) -> rgb in [0, 1]` and write them."""
    written = []
    t = (np.arange(RES) + 0.5) / RES * 2.0 - 1.0
    u, v = np.meshgrid(t, t)
    for name in FACE_ORDER:
        forward, right, down = FACES[name]
        dirs = forward + right * u[..., None] + down * v[..., None]
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        rgb = np.clip(colour(dirs), 0.0, 1.0)
        path = out_dir / f"{prefix}_{name}.png"
        Image.fromarray((rgb * 255).astype(np.uint8)).save(path)
        written.append(path)
    return written


def cubefiles(asset_dir: Path, prefix: str = "sky") -> tuple[str, ...]:
    """The six paths in MuJoCo's order, for a scene's `TextureCfg(cubefiles=...)`."""
    return tuple(str(asset_dir / f"{prefix}_{name}.png") for name in FACE_ORDER)
