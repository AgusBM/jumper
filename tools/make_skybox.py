#!/usr/bin/env python3
"""Render a scene's skybox to its asset directory.

    python tools/make_skybox.py beach
    python tools/make_skybox.py --all

A scene that generates cube textures declares them in a module-level
`CUBEMAPS = {prefix: colour_fn}` beside its `scene()` factory, where each
`colour_fn(dirs) -> rgb` is a function of direction; this writes the six faces of
each into `assets/scenes/<id>/`. The convention is that the skybox is called
`sky`; a football's panels are just another function of direction. Colours are code and live with the scene; the PNGs are
data and live under `assets/`, which is the same split as models.

Re-run it after changing a sky's colours, or the scene will keep loading the
faces that were written last time -- they are committed, not built on demand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import scenes
from scenes import skygen


def generate(scene_id: str) -> int:
    module = __import__(f"scenes.{scene_id}", fromlist=["CUBEMAPS"])
    maps = getattr(module, "CUBEMAPS", None)
    if not maps:
        print(f"  {scene_id}: no CUBEMAPS -- it generates no cube textures", file=sys.stderr)
        return 0
    out = Path(scenes.__file__).resolve().parent / "assets" / scene_id
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for prefix, colour in maps.items():
        written = skygen.write_cube(out, colour, prefix=prefix)
        total += len(written)
        print(f"  {scene_id}: {prefix} -> {len(written)} faces in {out}")
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", help="scene id; see scripts/train.py --list")
    ap.add_argument("--all", action="store_true", help="every scene that has a sky")
    args = ap.parse_args()

    if args.all:
        for spec in scenes.all_specs():
            generate(spec.id)
        return 0
    if not args.scene:
        ap.error("give a scene id, or --all")
    try:
        scenes.get(args.scene)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2
    return 0 if generate(args.scene) else 1


if __name__ == "__main__":
    sys.exit(main())
