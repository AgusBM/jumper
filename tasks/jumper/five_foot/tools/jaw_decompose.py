#!/usr/bin/env python3
"""Convex-**decompose** a link's collision mesh into slabs, as STL files.

The counterpart of `hull_collision.py`, for the one shape a single hull cannot
represent. That script is right about most of a robot: a thigh, a forearm, a
shell are all nearly convex, and one hull each is cheaper and no less accurate.

**A pincer is the exception, and it is the exception that matters here.** MuJoCo
collides meshes by their convex hull, so the hull of a hooked jaw is filled in
across the mouth, and the mouth is the only part of a claw anyone cares about.
`tasks/jumper/five_foot/tools/grasp_pose.py` has said so in its docstring since the claw
arrived: "the hull of a pincer has no mouth ... the wrong number for anyone who
then tries to close it on a block, who needs the claw links convex-decomposed
first." This is that decomposition.

## How

Slice, then hull each slice. The vertices are projected onto the mesh's longest
principal axis, cut into `--pieces` equal bands, and each band's convex hull is
written as its own STL. For a part that curves in one plane -- which is what a
finger is -- that recovers the curve: each band is short enough to be nearly
convex on its own, and the union of the bands follows the hook where one hull
would bridge it.

Bands overlap by `--overlap` of a band width so the union has no seam a contact
can fall through. The cost of the overlap is a little double-counted volume in
the middle of the part, where nothing touches anything.

**Every piece is still a hull, so nothing grows.** A hull contains its band's
points and no others, so the union is contained in the original hull: contact can
only trigger *later* than it did, never earlier, which is the same safe direction
`hull_collision.py` argues for. What changes is that it now triggers at the jaw
surfaces rather than at a chord across the mouth.

Usage:

    python tasks/jumper/five_foot/tools/jaw_decompose.py assets/jumper/meshes/LF_finger_link.STL --pieces 3

writes `LF_finger_link_col0.STL` ... `_col2.STL` next to the input and prints the
`<mesh>` and `<geom>` lines to paste into the MJCF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def decompose(mesh: trimesh.Trimesh, pieces: int, overlap: float,
               keep: int) -> list[trimesh.Trimesh]:
    """Bands along the longest principal axis, each one hulled."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    centred = v - v.mean(axis=0)
    # The axis the part is longest along, which for a finger is the one it curls
    # around. Principal components rather than the bounding box: a hook's box is
    # dominated by the curl, its first component by the length.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    t = centred @ axis
    edges = np.linspace(t.min(), t.max(), pieces + 1)
    width = (t.max() - t.min()) / pieces
    out: list[trimesh.Trimesh] = []
    for i in range(pieces):
        lo = edges[i] - (overlap * width if i else 0.0)
        hi = edges[i + 1] + (overlap * width if i < pieces - 1 else 0.0)
        band = v[(t >= lo) & (t <= hi)]
        if len(band) < 4:
            raise SystemExit(f"band {i} has {len(band)} vertices; use fewer --pieces")
        out.append(trimesh.convex.convex_hull(_decimate(band, keep)))
    return out


def _decimate(points: np.ndarray, keep: int) -> np.ndarray:
    """The `keep` extreme points of `points`, by support-function sampling.

    `hull_collision.py`'s method and for its reason: on the native backend domain
    randomisation copies one `MjModel` per environment, so vertices are paid for
    4096 times. A support sample only ever picks points that are already on the
    hull, so the result is contained in it -- decimation moves the surface inward,
    never outward, which is the direction that cannot create a contact that should
    not happen.
    """
    if keep <= 0 or len(points) <= keep:
        return points
    # Fibonacci directions: an even spread over the sphere with no clustering at
    # the poles, which a lat/long grid would have.
    i = np.arange(keep) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / keep)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)
    idx = np.unique(np.argmax(points @ dirs.T, axis=0))
    return points[idx]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stl", type=Path, help="the link's full mesh")
    p.add_argument("--pieces", type=int, default=3)
    p.add_argument("--overlap", type=float, default=0.15,
                   help="band overlap, as a fraction of a band width")
    p.add_argument("--keep", type=int, default=48,
                   help="vertices to keep per piece; 0 keeps every hull vertex")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    mesh = trimesh.load(args.stl, process=False)
    parts = decompose(mesh, args.pieces, args.overlap, args.keep)
    stem = args.stl.stem
    print(f"{stem}: {len(mesh.vertices)} vertices -> "
          + " + ".join(f"{len(q.vertices)}" for q in parts))
    for i, part in enumerate(parts):
        out = args.stl.with_name(f"{stem}_col{i}.STL")
        if not args.dry_run:
            part.export(out)
        print(f"  {out.name:<34} {len(part.vertices):>4} vertices  "
              f"volume {part.volume * 1e6:8.2f} cm^3")
    print("\nMJCF:")
    for i in range(len(parts)):
        print(f'    <mesh name="{stem}_col{i}" content_type="model/stl" '
              f'file="{stem}_col{i}.STL"/>')
    for i in range(len(parts)):
        suffix = "" if i == 0 else str(i)
        print(f'                <geom name="{stem}_meshcol{suffix}" class="collision" '
              f'type="mesh" mesh="{stem}_col{i}"/>')


if __name__ == "__main__":
    main()
