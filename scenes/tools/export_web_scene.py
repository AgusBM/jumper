"""Export a scene as a package another simulator can import.

The counterpart of `assets/jumper/tools/export_web_robot.py`, and deliberately its
opposite half: that one ships the robot and no world, this one ships the world
and no robot. A consumer puts them together.

    python scenes/tools/export_web_scene.py --scene football --out out/scenes
    python scenes/tools/export_web_scene.py --all --out out/scenes

## What travels

Everything the scene is, as long as it survives being written down:

  * the **ground** -- its geometry, and the contact parameters that make `ice`
    slippery and `rubber` grippy rather than merely blue and grey;
  * the **look** -- textures, materials, lights, the sky, and the headlight,
    which is the ambient floor and the first thing to reach for when a scene will
    not go dark;
  * **decorative geometry** -- pitch markings, goalposts, hoardings. Visual only,
    and this exporter checks that rather than trusting it;
  * **props** -- bodies with mass that the robot can push. One exists: a
    football. Each is written as its own file, because mjlab places a prop per
    environment and the consumer will have its own idea of where.

## What does not, and why the refusals are loud

**A scene that exists for training is refused**, by its own declaration --
`SceneSpec.use`, set in `scenes/__init__.py` where the judgement is made. This
file keeps no list of scene names. The rule that decides is one line, and the
reasoning lives beside the decision rather than beside the code that reads it.

**A scene whose behaviour is Python is refused.** `soft` applies a force at each
foot every step to fake granular resistance MuJoCo's contact model cannot
express. There is no field in an MJCF file for that, and the failure if it were
ignored is the worst kind: the package would compile, render correctly, and be
wrong only underfoot. `scene.events` being non-empty is the test.

The two gates are independent and catch different things, which is why both
exist. `soft` happens to fail both -- that is a coincidence worth keeping, not a
reason to drop one.

## What is checked before the package is called good

A written file that does not compile is a problem discovered by the consumer, so
this compiles it back and compares:

  * geom, light, texture and material counts against the spec it was written
    from -- a texture that silently failed to resolve is otherwise invisible;
  * **every decorative geom is non-colliding**. `tests/test_scenes.py` pins that
    rule inside this repository; a package that loses it would put invisible
    walls around a visitor's robot, and a goalpost that has become solid looks
    exactly like one that has not;
  * every referenced texture file was copied, and hashes to what was copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA = "kk-scene-package/1"

#: Written into the package so a consumer can say which MuJoCo it was compiled
#: against. Three versions are in play across this project and they do not all
#: parse the same MJCF.
_MUJOCO_MIN = "3.2"


@dataclass(frozen=True)
class Refused:
    """A scene this exporter will not write, and every reason it will not.

    Plural on purpose. `soft` fails both gates, and reporting only the first
    would say it is a training scene and leave someone to discover the Python
    force later -- after changing the declaration and finding it still refused.
    """

    scene_id: str
    reasons: tuple[str, ...]

    @property
    def reason(self) -> str:
        return " ".join(self.reasons)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_spec(scene: Any) -> Any:
    """Compile the scene onto an empty world -- no task, no robot.

    A scene is normally applied to a built environment config (`scenes.apply`),
    which needs the whole simulator. It does not have to be: the terrain entity
    builds its own `MjSpec` from the same four fields, and the headlight and
    `decorate` callback run on that spec exactly as they would on the real one.
    So a package can be produced on a machine that could not train.

    `terrain_type=None` means "keep whatever the task chose", and here there is
    no task. A plane is what every task in this repository chooses, so that is
    what a scene saying nothing gets -- stated here rather than left to the
    default of a config object.
    """
    from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg

    entity = TerrainEntity(
        TerrainEntityCfg(
            terrain_type=scene.terrain_type or "plane",
            terrain_generator=scene.terrain_generator,
            textures=scene.textures,
            materials=scene.materials,
            lights=scene.lights,
        ),
        device="cpu",
    )
    spec = entity.spec
    if scene.headlight is not None:
        spec.visual.headlight.ambient = list(scene.headlight.ambient)
        spec.visual.headlight.diffuse = list(scene.headlight.diffuse)
        spec.visual.headlight.specular = list(scene.headlight.specular)
    if scene.decorate is not None:
        scene.decorate(spec)
    return spec


def _texture_files(spec: Any) -> dict[str, Path]:
    """Every file the spec's textures point at, by the name it should be given.

    Read off the compiled spec rather than off the scene's config, because that
    is what the written XML will refer to. A cube map contributes six.
    """
    files: dict[str, Path] = {}
    for texture in spec.textures:
        if getattr(texture, "file", ""):
            source = Path(texture.file)
            files[source.name] = source
        for face in getattr(texture, "cubefiles", ()) or ():
            if face:
                source = Path(face)
                files[source.name] = source
    return files


def _localise_texture_paths(xml: str, files: dict[str, Path]) -> str:
    """Point the XML at the copied files rather than at this machine.

    `MjSpec` writes absolute paths, which are correct here and useless anywhere
    else. Replacing the directory rather than rewriting each attribute keeps the
    cube-map ordering, which is positional and easy to scramble.
    """
    for name, source in files.items():
        xml = xml.replace(str(source), f"assets/{name}")
    return xml


def _ground_contact(spec: Any) -> dict[str, Any]:
    """What the ground does to a foot, stated rather than left to be read out.

    A consumer that renders the package and ignores this gets a scene that looks
    like ice and grips like tarmac, which is the one failure the physics layer of
    this package exists to prevent. Naming it in the manifest is what lets an
    importer refuse rather than approximate.
    """
    from scenes.registry import GROUND_PRIORITY

    for geom in spec.geoms:
        if geom.name == "terrain":
            return {
                "geom": geom.name,
                "friction": [float(v) for v in geom.friction],
                "priority": int(geom.priority),
                "outranksFeet": int(geom.priority) >= GROUND_PRIORITY,
                "solref": [float(v) for v in geom.solref],
                "condim": int(geom.condim),
            }
    return {}


def _decorative_geoms(spec: Any) -> list[str]:
    """Geoms that carry no collision channel -- the pitch lines, the goalposts."""
    return [
        g.name for g in spec.geoms
        if g.name != "terrain" and int(g.contype) == 0 and int(g.conaffinity) == 0
    ]


def _colliding_geoms(spec: Any) -> list[dict[str, Any]]:
    """Geoms other than the ground that the robot or a prop can hit, with masks.

    Four exist, all in `football`: the perimeter boards, on a private channel
    (`BALL_BIT`) with bit 0 left out, so they are solid to the ball and
    transparent to the robot. That is the whole reason a kicked ball stays on the
    pitch, and it is two integers -- an importer that rounds them to "solid"
    fences the robot in, and one that rounds them to "decorative" loses the ball
    over the boards. Neither shows up as an error.
    """
    return [
        {"geom": g.name, "contype": int(g.contype), "conaffinity": int(g.conaffinity)}
        for g in spec.geoms
        if g.name != "terrain" and (int(g.contype) or int(g.conaffinity))
    ]


def _prop_package(name: str, cfg: Any, out: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    """Write one prop as its own model, and say where it belongs.

    Separate files because a prop is placed **per environment** here -- mjlab
    applies `env_origins` at reset -- so its position is relative to wherever its
    robot is, not to the scene. A consumer with one robot has its own idea of
    where that is, and folding the prop into the world would bake this
    repository's answer in.
    """
    spec = cfg.spec_fn()
    for texture in getattr(cfg, "textures", ()) or ():
        spec.add_texture(**_texture_kwargs(texture))
    for material in getattr(cfg, "materials", ()) or ():
        _bind_material(spec, material)
    spec.compile()
    files = _texture_files(spec)
    xml = _localise_texture_paths(spec.to_xml(), files)
    (out / f"props/{name}.xml").write_text(xml, "utf-8")
    body = next((b for b in spec.bodies if b.name == name), None)
    geoms = [g for g in spec.geoms if g.name == name]
    init = getattr(cfg, "init_state", None)
    return {
        "name": name,
        "file": f"props/{name}.xml",
        "textures": sorted(files),
        # Relative to the robot, not to the scene -- see the docstring.
        "initPositionRelativeToRobot": [float(v) for v in getattr(init, "pos", (0.0, 0.0, 0.0))],
        "mass": float(sum(g.mass for g in geoms)) if geoms else None,
        "freeJoint": bool(body and any(j.type.name.endswith("FREE") for j in body.joints)),
        "contact": [
            {
                "geom": g.name,
                "condim": int(g.condim),
                "friction": [float(v) for v in g.friction],
                "solref": [float(v) for v in g.solref],
            }
            for g in geoms
        ],
    }, files


def _texture_kwargs(texture: Any) -> dict[str, Any]:
    """`TextureCfg` -> `MjSpec.add_texture` keywords, for a prop's own textures.

    The terrain entity does this for the scene's textures; a prop's are attached
    by mjlab at a point this exporter does not reach, so they are applied by hand
    here. Only the fields a prop has ever used are carried, and an unknown one
    raises rather than being dropped -- a texture that silently did not arrive
    renders as flat white, which reads as a lighting problem.
    """
    import mujoco

    known = {"name", "type", "builtin", "file", "cubefiles", "rgb1", "rgb2",
             "markrgb", "mark", "width", "height", "nchannel"}
    fields = {f: getattr(texture, f) for f in known if getattr(texture, f, None) not in (None, "")}
    kinds = {"2d": mujoco.mjtTexture.mjTEXTURE_2D,
             "cube": mujoco.mjtTexture.mjTEXTURE_CUBE,
             "skybox": mujoco.mjtTexture.mjTEXTURE_SKYBOX}
    if "type" in fields:
        fields["type"] = kinds[fields["type"]]
    builtins = {"none": mujoco.mjtBuiltin.mjBUILTIN_NONE,
                "gradient": mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                "checker": mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                "flat": mujoco.mjtBuiltin.mjBUILTIN_FLAT}
    if "builtin" in fields:
        fields["builtin"] = builtins[fields["builtin"]]
    marks = {"none": mujoco.mjtMark.mjMARK_NONE, "edge": mujoco.mjtMark.mjMARK_EDGE,
             "cross": mujoco.mjtMark.mjMARK_CROSS, "random": mujoco.mjtMark.mjMARK_RANDOM}
    if "mark" in fields:
        fields["mark"] = marks[fields["mark"]]
    if "cubefiles" in fields:
        fields["cubefiles"] = list(fields["cubefiles"])
    return fields


def _bind_material(spec: Any, material: Any) -> None:
    """Add a material and put it on the geoms its expression names."""
    import re

    mat = spec.add_material(name=material.name)
    if getattr(material, "texture", None):
        mat.textures[1] = material.texture  # slot 1 is RGB
    for field in ("texuniform", "texrepeat", "reflectance", "shininess", "specular"):
        value = getattr(material, field, None)
        if value is not None:
            setattr(mat, field, list(value) if isinstance(value, tuple) else value)
    for expr in getattr(material, "geom_names_expr", ()) or ():
        for geom in spec.geoms:
            if re.match(expr, geom.name):
                geom.material = material.name


def _rules(scene: Any, spec: Any, decorative: list[str]) -> dict[str, str]:
    """What an importer must not round off, for this scene only.

    Each of these is a fact that survives or does not survive as an integer, and
    whose loss renders identically to its survival. That is the test for belonging
    here -- anything a consumer would notice by looking does not need saying.
    """
    from scenes.registry import GROUND_PRIORITY

    rules: dict[str, str] = {}
    if decorative:
        rules["decorativeGeomsAreNonColliding"] = (
            f"All {len(decorative)} geoms in decorativeGeoms have contype=0 and "
            f"conaffinity=0 and must keep them. They are a look; a goalpost that "
            f"becomes solid is an invisible wall around the robot and looks "
            f"identical to one that has not."
        )
    ground = next((g for g in spec.geoms if g.name == "terrain"), None)
    if ground is not None and int(ground.priority) >= GROUND_PRIORITY:
        rules["groundOutranksTheFeet"] = (
            f"The ground carries priority {int(ground.priority)} so its friction "
            f"reaches the solver instead of the robot's feet -- MuJoCo uses the "
            f"higher geom's whole parameter set and discards the other's. Drop the "
            f"priority and this scene renders exactly as it does now and grips like "
            f"ordinary ground, which is the entire point of it gone."
        )
    private = [g for g in _colliding_geoms(spec) if not (g["contype"] & 1)]
    if private:
        rules["privateCollisionChannels"] = (
            f"{', '.join(g['geom'] for g in private)} leave bit 0 out of their masks: "
            f"they collide with what shares their channel and with nothing else. "
            f"Making them ordinarily solid fences the robot in; making them "
            f"decorative loses whatever they were containing."
        )
    if scene.props:
        rules["propsArePlacedPerRobot"] = (
            "initPositionRelativeToRobot is relative to the robot, not to the scene. "
            "Here each environment has its own copy beside its own robot; a consumer "
            "with one robot decides where that is. Reading it as a world position "
            "puts the prop at the origin, which for this pitch is the centre spot -- "
            "plausible enough to go unquestioned."
        )
    return rules


def refuse(scene_id: str) -> Refused | None:
    """Why this scene may not be exported, or None if it may.

    Both gates live here so a caller cannot take one and miss the other, and so
    `--all` can report every refusal at once rather than stopping at the first.
    """
    import scenes

    reasons: list[str] = []
    spec = scenes.get(scene_id)
    if spec.use == "training":
        reasons.append(
            f"it declares use={spec.use!r}: it exists to make training work, not to "
            f"be walked around in. The reasoning is beside its registration in "
            f"scenes/__init__.py. If that has changed, change it there -- this "
            f"exporter keeps no list."
        )
    scene = scenes.load(scene_id)
    if scene.events:
        reasons.append(
            f"it carries event term(s) {sorted(scene.events)}: behaviour that runs "
            f"in Python every step. There is nothing in an MJCF file that means it, "
            f"so a package would render correctly and be wrong only underfoot."
        )
    return Refused(scene_id, tuple(reasons)) if reasons else None


def export(scene_id: str, out_root: Path) -> Path:
    """Write one scene package. Raises on a refused scene."""
    import scenes

    denial = refuse(scene_id)
    if denial is not None:
        raise ValueError(f"scene {scene_id!r} will not be exported: {denial.reason}")

    meta = scenes.get(scene_id)
    scene = scenes.load(scene_id)
    spec = _build_spec(scene)
    model = spec.compile()

    out = out_root / scene_id
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)
    if scene.props:
        (out / "props").mkdir()

    files = _texture_files(spec)
    missing = sorted(name for name, source in files.items() if not source.exists())
    if missing:
        raise FileNotFoundError(
            f"scene {scene_id!r} refers to texture files that are not on disk: "
            f"{missing}. Cube maps are generated -- see tools/make_skybox.py."
        )
    for name, source in files.items():
        shutil.copy2(source, out / "assets" / name)

    xml = _localise_texture_paths(spec.to_xml(), files)
    (out / "scene.xml").write_text(xml, "utf-8")

    props = []
    for name, cfg in sorted(scene.props.items()):
        prop, prop_files = _prop_package(name, cfg, out)
        for tex_name, source in prop_files.items():
            if not source.exists():
                raise FileNotFoundError(
                    f"prop {name!r} in scene {scene_id!r} refers to {source}, which "
                    f"is not on disk. Cube maps are generated -- see tools/make_skybox.py."
                )
            # A prop's textures live beside the scene's: one assets directory, so
            # the consumer has one place to look.
            shutil.copy2(source, out / "assets" / tex_name)
        props.append(prop)

    counts, decorative = _verify(out / "scene.xml", spec, model)

    manifest = {
        "schema": SCHEMA,
        "id": scene_id,
        "description": meta.description,
        "tags": list(meta.tags),
        "use": meta.use,
        "exportedBy": {
            "repo": "KingKongRobotics/jumper",
            "commit": _git_commit(),
            "tool": "scenes/tools/export_web_scene.py",
            "mujocoAtLeast": _MUJOCO_MIN,
        },
        "world": {
            "file": "scene.xml",
            "terrainType": scene.terrain_type or "plane",
            "ground": _ground_contact(spec),
            "counts": counts,
            # Named individually, not counted: an importer that wants to check
            # the rules below has to know which geoms they apply to.
            "decorativeGeoms": decorative,
            "collidingGeoms": _colliding_geoms(spec),
        },
        "props": props,
        # Only the rules this package actually needs. A rule stated where it does
        # not apply is worse than none: `football`'s ground is the default one at
        # priority 0, and a line claiming it carries 2 would contradict the number
        # three keys above it, in a file whose whole job is to be believed.
        "rules": _rules(scene, spec, decorative),
        "files": {},
    }
    manifest["files"] = {
        str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": _sha256(p)}
        for p in sorted(out.rglob("*")) if p.is_file()
    }
    (out / "scene-package.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    return out


def _verify(path: Path, spec: Any, model: Any) -> tuple[dict[str, int], list[str]]:
    """Compile the written file back and hold it to what it was written from.

    What the counts catch is **geometry that did not survive the round trip** --
    99 pitch markings written, 98 read back. MuJoCo will not catch that: fewer
    lines on a pitch is a valid model.

    What they do not catch is worth being clear about, because the obvious guess
    is wrong: a texture that failed to resolve does not slip through here, it
    raises on compile, because the material that names it no longer resolves
    either. That case is loud already and needs nothing from this function.
    """
    import mujoco

    written = mujoco.MjModel.from_xml_path(str(path))
    counts = {
        "geoms": int(written.ngeom),
        "lights": int(written.nlight),
        "textures": int(written.ntex),
        "materials": int(written.nmat),
    }
    expected = {
        "geoms": int(model.ngeom), "lights": int(model.nlight),
        "textures": int(model.ntex), "materials": int(model.nmat),
    }
    if counts != expected:
        raise AssertionError(
            f"{path} compiles to {counts}, but the scene it was written from is "
            f"{expected}. Something did not survive being written down."
        )

    decorative = _decorative_geoms(spec)
    solid = [
        name for name in decorative
        if (i := mujoco.mj_name2id(written, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0
        and (written.geom_contype[i] or written.geom_conaffinity[i])
    ]
    if solid:
        raise AssertionError(
            f"{path}: decorative geoms {solid} came back able to collide. In this "
            f"repository that rule is pinned by tests/test_scenes.py; a package "
            f"that loses it puts invisible walls around the robot."
        )

    tree = ET.parse(path)
    absolute = [
        value for element in tree.iter() for value in element.attrib.values()
        if value.startswith("/") or (len(value) > 1 and value[1] == ":")
    ]
    if absolute:
        raise AssertionError(
            f"{path} still refers to this machine: {absolute[:3]}. The package has "
            f"to be readable where it lands."
        )
    return counts, decorative


def main(argv: list[str] | None = None) -> int:
    import scenes

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", help="scene id to export")
    parser.add_argument("--all", action="store_true", help="export every scene that may be")
    parser.add_argument("--out", type=Path, default=REPO / "out/scenes")
    parser.add_argument("--list", action="store_true", help="what would be exported, and what not")
    args = parser.parse_args(argv)

    if args.list:
        for scene_id in scenes.list_ids():
            denial = refuse(scene_id)
            mark = "  -" if denial else "  +"
            print(f"{mark} {scene_id:10s} use={scenes.get(scene_id).use:9s} "
                  f"{scenes.get(scene_id).description}")
            for reason in denial.reasons if denial else ():
                print(f"      withheld: {reason}")
        return 0

    if args.all:
        ids = [s for s in scenes.list_ids() if refuse(s) is None]
        withheld = [(s, refuse(s)) for s in scenes.list_ids() if refuse(s) is not None]
    elif args.scene:
        ids, withheld = [args.scene], []
    else:
        parser.error("pass --scene <id>, --all, or --list")
        return 2

    for scene_id in ids:
        out = export(scene_id, args.out)
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        print(f"wrote {out} ({size / 1e6:.2f} MB)")
    # Said out loud rather than left as an absence: a scene missing from the
    # output directory looks exactly like a scene nobody has written yet.
    for scene_id, denial in withheld:
        print(f"withheld {scene_id}: {denial.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
