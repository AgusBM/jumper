#!/usr/bin/env python3
"""Replace an MJCF's collision meshes with **decimated convex hulls**, written
out as standalone STL files.

This is a **standalone** script depending only on mujoco / numpy / scipy, not on
the mjrl framework.

## Why

MuJoCo already uses the convex hull for mesh collision, so with a raw mesh as the
collision body the great majority of its thousands of vertices are never used --
while still being stored, and with the BVH built over the **full mesh faces**
(measured, `bvh_aabb` is half of a model with visual meshes already stripped).

On the native backend the cost is a hard constraint: domain randomisation copies
one `MjModel` **per environment**. Bringing the hexapod's non-foot links into
collision took the stripped model from 2.6 MiB to 15.3 MiB, i.e. 61 GiB at 4096
environments, which does not fit. With decimated hulls it is 3.0 MiB / 12.2 GiB.

## Why hulls rather than capsules

A fitted capsule **bulges outward**. Measured in the hexapod's standing pose, the
capsules around the gripper segments sit 1.6-9.8 mm **below** the foot meshes --
the robot would stand on its palms rather than its jaws, quietly destroying the
"feet make mesh contact" design.

A convex hull contains **every extreme point** of the original mesh, so its lowest
point is identical and contact timing is unchanged. Decimation only moves it
**inward** (support-function sampling picks points that lie on the original hull),
which is the safe direction: contact triggers slightly later, never earlier. This
script computes that inward shrink and reports it.

## What not to decimate

**Do not touch the meshes that actually touch the ground.** Those are the six in
`constants.FEET`, and even the undecimated hull is wrong for them: it replaces the
rounded pad with flat facets, which moves when the foot lands.

**Compare k against what the robot actually ran before, not against the raw
mesh.** `-k 0` writes the exact hull and is tempting because MuJoCo already
collides a mesh by its hull, so it approximates nothing (0.008 um of support
function on the V1.6 hexapod). It was shipped on that argument and filled a
24 GB card at 4096 environments: the hexapod's previous collision meshes were
56-64 vertex hulls, so the exact hull was a 6.3x regression in collision faces
against the only baseline that mattered. `-k 64` reproduces the old density --
3424 non-foot faces against V1.5's 2954 -- for 7.81 mm of inward shrink.

Measured in `k -> total hull vertices / max inward shrink` on the V1.6 hexapod:
0 -> 28662 / 0.00 mm, 256 -> 6110 / 3.57 mm, 64 -> 1772 / 7.81 mm.

Usage:
    # report only, writing nothing
    python tools/hull_collision.py --model assets/jumper/jumper.xml -k 0 \\
        --exclude LF_palm_pad_b_link,RF_palm_pad_b_link,LM_foot_tip_link,RM_foot_tip_link,LR_foot_tip_link,RR_foot_tip_link

    # write the STL files and repoint the XML's *_meshcol at them
    python tools/hull_collision.py --model assets/jumper/jumper.xml --exclude ... --apply

    # change the decimation density; -k 0 means the full hull, no decimation
    python tools/hull_collision.py --model m.xml -k 128 --apply
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

import numpy as np


# ── Geometry ──────────────────────────────────────────────────────────────


def fibonacci_directions(k: int) -> np.ndarray:
    """k approximately uniform directions on the sphere.

    Used for support-function sampling: take the furthest vertex along each
    direction. Every point so obtained lies **on the surface** of the original
    hull, so the hull can only shrink inward, never grow outward -- and for a link
    that should not be touching the ground, inward means contact detection triggers
    slightly later, which is the safe side.
    """
    i = np.arange(k) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / k)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
        axis=1,
    )


def hull_decimate(verts: np.ndarray, k: int | None = 64) -> np.ndarray:
    """Return the vertices' convex hull, sampling k directions first when k is not
    None.

    Args:
        verts: `[N, 3]` vertices.
        k: number of sampled directions. `None` means the full hull, no decimation.
    """
    from scipy.spatial import ConvexHull

    verts = np.asarray(verts, dtype=float)
    if k is None:
        return verts[ConvexHull(verts).vertices]
    idx = sorted({int(np.argmax(verts @ d)) for d in fibonacci_directions(k)})
    pts = verts[idx]
    if len(pts) <= 4:
        return pts
    return pts[ConvexHull(pts).vertices]


def max_inward_shrink(verts: np.ndarray, hull_pts: np.ndarray) -> float:
    """How far the original mesh surface protrudes beyond the decimated hull, in
    metres. **This is the geometric error of decimation.**

    Decimation only shrinks inward, so the quantity is one-sided: contact detection
    triggers this much later than with the original mesh. A full hull contains
    every original vertex, and the return value is then of the order of floating
    point rounding (measured 1e-16), non-zero only because the face equations
    themselves carry error.

    Computed by taking the hull's face equations `n.x + d <= 0` and finding the
    largest violation over all vertices.
    """
    from scipy.spatial import ConvexHull

    if len(hull_pts) < 4:
        return float("nan")
    eq = ConvexHull(hull_pts).equations  # [F, 4]; rows are (nx, ny, nz, d)
    out = np.asarray(verts, dtype=float) @ eq[:, :3].T + eq[:, 3]
    return float(max(0.0, out.max()))


# ── STL reading and writing ───────────────────────────────────────────────


def read_stl(path: Path) -> np.ndarray:
    """Read a binary STL's vertices, returning `[N, 3]` **in file coordinates**.

    **Read from the file; never take `mesh_vert` from a compiled model.** MuJoCo's
    compiler aligns meshes into its own reference frame (recording the transform in
    `mesh_pos` / `mesh_quat`), and `mesh_vert` holds the coordinates **after** that
    alignment. Writing those out as a new STL makes the compiler align them a
    second time, and the new mesh ends up rotated relative to the original.

    Measured symptom: the hexapod's `base_link` hull came out
    [0.069, 0.181, 0.204] against an appearance mesh of [0.206, 0.182, 0.071] --
    x and z swapped. It is obvious once rendered: the body's hull becomes a slab
    reaching down to the ground and the arm's becomes a blade flying off to one
    side.
    """
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path} does not look like a binary STL "
                         f"(only {len(raw)} bytes)")
    (ntri,) = struct.unpack("<I", raw[80:84])
    if len(raw) < 84 + ntri * 50:
        raise ValueError(f"{path} claims {ntri} triangles but the file is too "
                         f"short (an ASCII STL?)")
    a = np.frombuffer(raw[84 : 84 + ntri * 50], dtype=np.uint8).reshape(ntri, 50)
    return a[:, 12:48].copy().view("<f4").reshape(ntri * 3, 3).astype(np.float64)


def write_stl(path: Path, verts: np.ndarray) -> int:
    """Write the vertices' convex hull as a binary STL, returning the triangle
    count.

    Normals are written as the hull's own outward normals. MuJoCo recomputes the
    hull at compile time, so normals do not affect collision, but getting them
    right means other tools do not open the mesh inside out.
    """
    from scipy.spatial import ConvexHull

    hull = ConvexHull(verts)
    tris, normals = [], []
    for simplex, eq in zip(hull.simplices, hull.equations):
        n = eq[:3]
        tri = verts[simplex]
        if np.dot(np.cross(tri[1] - tri[0], tri[2] - tri[0]), n) < 0:
            tri = tri[[0, 2, 1]]  # make the winding match the outward normal
        tris.append(tri)
        normals.append(n)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"mjrl hull".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        for tri, n in zip(tris, normals):
            f.write(struct.pack("<3f", *n.astype(np.float32)))
            for v in tri:
                f.write(struct.pack("<3f", *v.astype(np.float32)))
            f.write(struct.pack("<H", 0))
    return len(tris)


# ── Finding the targets ───────────────────────────────────────────────────


def collision_mesh_targets(spec, geom_suffix: str) -> list[tuple[str, str]]:
    """Find the `(geom name, mesh name)` pairs to process.

    Selected by geom-name suffix rather than by contype: one MJCF often carries
    several collision schemes at once (the hexapod has mesh, capsule and hybrid),
    and the layer above chooses between them by zeroing contype at the spec stage.
    **The compiled contype therefore depends on which scheme was selected**, so
    using it as the criterion would make the result vary with the caller. A name
    suffix is the asset author's explicit marking and is far more stable.
    """
    out = []
    for body in spec.bodies:
        for g in body.geoms:
            if g.name.endswith(geom_suffix) and g.meshname:
                out.append((g.name, g.meshname))
    return out


def mesh_files(spec) -> dict[str, str]:
    return {me.name: me.file for me in spec.meshes if me.file}


# ── Patching the XML ──────────────────────────────────────────────────────


def patch_xml(path: Path, pairs: list[tuple[str, str, str]], suffix: str,
              file_prefix: str = "") -> int:
    """Repoint `<geom name=A ... mesh=B>` at `B+suffix` and add the mesh assets.

    Deliberately a **targeted text substitution** rather than `spec.to_xml()`: the
    latter reflows the entire file, losing hand-maintained comments, indentation
    and element order, and leaving a reviewer unable to see what changed.

    Idempotent: geoms already pointing at `*suffix` are skipped.

    Args:
        pairs: `[(geom name, old mesh name, new mesh name), ...]`
    Returns:
        The number of geoms actually changed.
    """
    text = path.read_text(encoding="utf-8")
    changed = 0
    for geom_name, old_mesh, new_mesh in pairs:
        # Substitute mesh= only inside that one <geom ...> element
        pat = re.compile(
            r'(<geom\b[^>]*\bname\s*=\s*"' + re.escape(geom_name) + r'"[^>]*?)'
            r'\bmesh\s*=\s*"' + re.escape(old_mesh) + r'"',
            re.S,
        )
        text, n = pat.subn(lambda m: m.group(1) + f'mesh="{new_mesh}"', text)
        if n > 1:
            raise SystemExit(f"geom {geom_name} matched {n} times in the XML; "
                             f"refusing to edit")
        changed += n

    # Add <mesh name=... file=.../> before </asset>, copying the indentation of an
    # existing <mesh> line in the same block so generated assets line up with
    # hand-written ones in a diff.
    want = [new for _, _, new in pairs if f'name="{new}"' not in text]
    if want:
        close = re.search(r"^([ \t]*)</asset>", text, re.M)
        if close is None:
            raise SystemExit("the XML has no </asset>; nowhere to put the mesh assets")
        existing = re.search(r"^([ \t]*)<mesh\b", text, re.M)
        pad = existing.group(1) if existing else close.group(1) + "  "
        block = "".join(
            f'{pad}<mesh name="{n}" file="{file_prefix}{n}.STL"/>\n'
            for n in dict.fromkeys(want)
        )
        text = text[: close.start()] + block + text[close.start() :]
    path.write_text(text, encoding="utf-8")
    return changed


# ── Main ──────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="replace an MJCF's collision meshes with decimated convex hulls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:", 1)[-1],
    )
    ap.add_argument("--model", required=True, type=Path, help="path to the MJCF")
    ap.add_argument(
        "-k", "--dirs", type=int, default=64,
        help="number of sampled directions, setting the decimation density "
             "(default 64; 0 = the full hull, no decimation)",
    )
    ap.add_argument(
        "--exclude", default="",
        help="comma-separated mesh names to leave alone -- **put the meshes that "
             "actually touch the ground here**",
    )
    ap.add_argument("--geom-suffix", default="_meshcol",
                    help="name suffix of the collision geoms")
    ap.add_argument("--out-suffix", default="_col",
                    help="name suffix for the new mesh assets")
    ap.add_argument("--mesh-dir", type=Path, default=None,
                    help="STL directory (inferred from the compiler meshdir by default)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write the hulls, if not beside the source meshes. "
                         "The written <mesh file=...> is this path relative to the "
                         "compiler's meshdir, so the source tree can stay read-only")
    ap.add_argument("--apply", action="store_true",
                    help="actually write the STLs and edit the XML "
                         "(report only by default)")
    args = ap.parse_args(argv)

    import mujoco

    k = None if args.dirs == 0 else args.dirs
    spec = mujoco.MjSpec.from_file(str(args.model))
    meshdir = args.model.parent / (spec.meshdir or "meshes")
    mesh_dir = args.mesh_dir or meshdir
    out_dir = args.out_dir or mesh_dir
    # What the XML must say to find a hull, given where the compiler starts.
    file_prefix = ""
    if out_dir != meshdir:
        file_prefix = out_dir.resolve().relative_to(meshdir.resolve()).as_posix() + "/"
    excluded = {s for s in args.exclude.split(",") if s}

    targets = collision_mesh_targets(spec, args.geom_suffix)
    if not targets:
        print(f"no geom found whose name ends in {args.geom_suffix!r} and has a "
              f"mesh.", file=sys.stderr)
        return 1
    files = mesh_files(spec)

    todo, skipped = [], []
    for geom_name, mesh_name in targets:
        if mesh_name in excluded or mesh_name.endswith(args.out_suffix):
            skipped.append((geom_name, mesh_name))
            continue
        if mesh_name not in files:
            raise SystemExit(f"mesh {mesh_name} has no file (inline vertices?)")
        todo.append((geom_name, mesh_name))

    print(f"model {args.model}   STL directory {mesh_dir}")
    print(f"decimation: {'full hull, none' if k is None else f'k={k} directions'}"
          f"    processing {len(todo)} / skipping {len(skipped)}")
    if skipped:
        print("  skipped: " + ", ".join(sorted({m for _, m in skipped})))
    if not todo:
        print("\nNothing to process -- everything is either excluded or already a "
              "decimated hull.")
        return 0
    print()
    print(f"  {'mesh':32s} {'verts':>8s} {'hull':>6s} {'saved':>6s}  {'max shrink':>12s}")
    print("  " + "-" * 70)

    pairs, total_before, total_after, worst = [], 0, 0, 0.0
    for geom_name, mesh_name in todo:
        verts = read_stl(mesh_dir / files[mesh_name])
        hull = hull_decimate(verts, k)
        err = max_inward_shrink(verts, hull)
        worst = max(worst, 0.0 if np.isnan(err) else err)
        total_before += len(verts)
        total_after += len(hull)
        print(f"  {mesh_name:32s} {len(verts):8d} {len(hull):6d} "
              f"{1 - len(hull) / len(verts):5.1%}  {err * 1e3:8.2f} mm")
        pairs.append((geom_name, mesh_name, f"{mesh_name}{args.out_suffix}", hull))

    print("  " + "-" * 70)
    print(f"  {'total':32s} {total_before:8d} {total_after:6d} "
          f"{1 - total_after / max(total_before, 1):5.1%}  {worst * 1e3:9.2f} mm max")
    print()

    if not args.apply:
        print("(report only; pass --apply to write the STLs and edit the XML)")
        return 0

    for _, mesh_name, new_name, hull in pairs:
        ntri = write_stl(out_dir / f"{new_name}.STL", hull)
        print(f"  wrote {new_name}.STL ({ntri} triangles)")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = patch_xml(args.model, [(g, o, nw) for g, o, nw, _ in pairs],
                  args.out_suffix, file_prefix)
    print(f"  repointed {n} geoms and added the <mesh> assets")

    # Compile once, to confirm the result still works
    m = mujoco.MjModel.from_xml_path(str(args.model))
    print(f"  recompiled successfully: ngeom={m.ngeom} nmesh={m.nmesh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
