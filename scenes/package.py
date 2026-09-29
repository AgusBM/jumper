"""Import a `kk-scene-package/1` or `/2` and run it as a scene.

The counterpart of `scenes/tools/export_web_scene.py`: that one writes a package
another simulator can import, this one reads one somebody else wrote. The format
is the same in both directions, which is the point -- a room exported by
scene-generator and a pitch exported from here are the same kind of file.

    python scripts/play.py --task jumper.flat --scene out/bedroom.scene
    python scripts/play.py --task jumper.flat --scene out/bedroom.scene.zip
    python scripts/play.py --task jumper.flat --scene library/maps/bedroom.map

`/2` is what kingkong_design's map library is written in: the `/1` world, plus
the robot the map was authored around (`robot/`) and a thumbnail (`preview/`).
Both are verified like every other file and neither is loaded -- the robot here
is the task's own, the one the policy was trained on -- and `spawn` becomes
required, which is what the next two sections are about.

## Simulation only

An imported package is refused by `scripts/train.py`, and the refusal is the
feature. The package is one room at the world origin; mjlab lays environments out
on a grid by `env_spacing` and replicates nothing that is not an entity, so at
4096 environments exactly one robot would be indoors and 4095 would be walking on
empty ground. Training runs, converges to something, and nothing anywhere reports
a problem -- the same failure class as rough terrain with no height scan, and
loud for the same reason.

## Attached, never spliced

The world arrives as a whole MJCF written against its own compiler settings, so
it is attached through `MjSpec.attach` rather than merged. Splicing was measured
on mujoco 3.11, 3.13 and 3.14 alike to do three silent things to the host robot:

    <compiler angle="degree">          joint limits reinterpreted as degrees,
                                       so +-1.57 rad becomes +-0.027
    <default><geom density="600"/>     the robot's own mass and contacts
                                       rewritten -- 8.0 kg becomes 4.8
    <option timestep="0.002">          the timestep the policy was trained at,
                                       replaced

None of them errors. `attach` isolates all three, keeps the parent's values, and
prefixes every name the package brought.

## The geometry collides

`Scene.decorate` normally may only add geometry the robot cannot touch, because a
look should not be able to change the task. This scene is the exception and says
so: a room the robot walks through is not a room. That is exactly why it is
refused for training -- the rule exists to protect a training run, and the
refusal, not a weaker room, is what keeps that protection.

## Where the robot starts

A package says where a robot belongs -- `spawn`, a point on the ground and a
heading in **degrees** -- while the robot here starts where its task puts it, at
the environment origin facing +X. So the world is attached in a frame that moves
the spawn point to the origin and turns its heading onto +X, and the robot's
initial state is left alone. Ignoring `spawn` is not a small offset: measured on
kingkong_design's 13 maps (c655914), 6 put it away from the origin, and a robot at
the origin instead stands in a shelf (shelf-maze: 0.82 m of shelf within 0.25 m
of it), on a 0.7 m ramp (switchback-slopes) or in bedroom furniture (0.45 m).

## The ground is the package's

A package that names its ground (`world.ground.geom`) replaces the task's. Keeping
the task's plane at z=0 fills in whatever the package has below that height:
narrow-bridge's basin lies at -0.16 m under 94% of the map, and with the plane in
place the bridge is a painted strip on a floor. Switchback-slopes (-0.04 m, 72%)
and shelf-maze (-0.03 m, 60%) are buried the same way.

It is not enough to delete the task's plane. Every jumper task's
`feet_ground_contact` sensor counts contacts with the **body** named `terrain`,
so a foot on anything that lives elsewhere reads as a foot in the air -- and
`jumper.ref_free_jump` decides take-off and landing from that sensor. So every
geom of the package that collides and does not move -- the ground, and equally
a start platform, a bridge deck, a ramp or a table top -- is rebuilt inside that
body (`_Terrain`), the ground under the name `terrain` so that everything that
finds the ground by name keeps finding it. What moves stays where it was.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import Scene

__all__ = ["PackageError", "ScenePackage", "is_package_path", "load_package", "read_package"]

#: The versions this importer reads. An unknown one is refused rather than read
#: as the nearest known one: kingkong_design's contract says a consumer that
#: does not know a version must reject it.
SCHEMAS = ("kk-scene-package/1", "kk-scene-package/2")
MANIFEST = "scene-package.json"
#: The archive extension. A scene is a map; the platforms take what they need.
SUFFIX = ".map"
#: Where a package keeps the forms that are not the shared world. This importer
#: uses the shared one -- `platforms/mjlab/` is the same room split into RL
#: entities for a task author, not for `--scene`. They are not loaded together:
#: each entity ships its own copy of the textures it uses, so the two halves
#: collide on base names, and MuJoCo's asset lookup falls back to the base name.
PLATFORM_DIR = "platforms/"
#: What `/2` adds beside the world, kept out of the shared section for the same
#: reason as `PLATFORM_DIR`: every map in kingkong_design's library carries both
#: `preview/preview.png` and `robot/preview/preview.png`, and the robot's meshes
#: are free to share a base name with the world's. Only for `/2`, where the
#: names mean this -- in a `/1` package `robot/` would be an ordinary directory.
_V2_SECTIONS = ("robot/", "preview/")

#: What this importer can run, in the vocabulary of `requiredCapabilities`. A
#: package asking for anything else is refused, because the contract forbids
#: the alternative -- running it and silently dropping what it could not do.
#: `flex` runs, an order of magnitude slower; `scripts/_cli.py` prints the
#: measurement when a package carries one.
CAPABILITIES = frozenset({"rigid", "hfield", "flex"})

#: The task's ground, as mjlab builds it: a geom `terrain` in a body `terrain`,
#: attached at the identity (`mjlab.terrains.terrain_entity`). The body is what
#: the jumper tasks' foot contact sensor matches; the geom is what
#: `registry.GROUND` and the exporter match.
GROUND_BODY = "terrain"
GROUND_GEOM = "terrain"
#: The name an unnamed geom in a package's world is given, by its index.
_UNNAMED_GEOM = "kk_import_geom_{}"

#: Files a package may carry without declaring them. They are written for people
#: and nothing loads them.
_UNDECLARED_OK = frozenset({MANIFEST, "README.md", "LICENSE"})

#: Sections an imported world may not carry. `<option>` because the host's
#: integration settings are the host's and `attach` would discard them anyway --
#: saying so is what turns a silent discard into a stated rule. `<include>`
#: because it would pull in a file nothing checked. `<plugin>`/`<extension>`
#: because a package is data and assets, not code.
_FORBIDDEN = {
    "option": "the host decides the timestep and the integrator; put the values in `physics`",
    "include": "a package is one world file; <include> brings in content nothing verified",
    "plugin": "a package carries data and assets, not code",
    "extension": "a package carries data and assets, not code",
}

#: How long to let a world settle when measuring what it needs. The bedroom
#: package reaches its steady 143 contacts within 600 steps at its own 2 ms
#: timestep; at reset it reports 4, because nothing has fallen yet.
_SETTLE_STEPS = 600
#: Headroom over the settled world, for the robot that walks into it -- and
#: deliberately lopsided, because the two knobs do not cost the same. Measured in
#: this repository at 4096 environments (see the note above `rough.NCONMAX`):
#: doubling `njmax` from 512 to 1024 cost nothing, while doubling `nconmax` from
#: 128 to 256 cost 4.7 GB of a 23 GB card. So `njmax` is scaled and `nconmax` is
#: not.
#:
#: The multiplier rather than another constant is because the robot's share is
#: dominated by its **contacts with the room**, which scale with how cluttered
#: the room is, not with the robot. Measured with the bedroom package and the
#: jumper: the world settles to 1392 constraints on its own and the run needed
#: 2778, so a fixed +1024 was short by 362 and a fixed +2048 would have been luck.
_CONTACT_MARGIN = 256
_CONSTRAINT_SCALE = 2
_CONSTRAINT_MARGIN = 2048

#: The prefix is interpolated into names this process then looks up, and the file
#: names are opened. Neither may be taken on trust from an uploaded manifest.
_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]*$")
_SAFE_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class PackageError(Exception):
    """A package this importer will not load, and the reason it will not."""


@dataclass
class ScenePackage:
    """A verified package, and where its files are on disk."""

    root: Path
    manifest: dict[str, Any]
    #: Kept alive for as long as this package is: a `.zip` is expanded into a
    #: temporary directory that `root` points into, and the world file is not
    #: read until mjlab builds the environment, well after `load_package`
    #: returned. Dropping this would delete the assets between the two.
    _extracted: Any = None

    @property
    def id(self) -> str:
        return str(self.manifest["id"])

    @property
    def world(self) -> Path:
        return self.root / self.manifest["world"]["file"]

    @property
    def prefix(self) -> str:
        return str(self.manifest["world"].get("attach", {}).get("prefix") or "scn_")

    @property
    def spawn(self) -> dict[str, Any] | None:
        return self.manifest.get("spawn")

    @property
    def ground(self) -> str | None:
        """The name of the package's ground geom, or None if it names none."""
        return (self.manifest["world"].get("ground") or {}).get("geom") or None

    def assets(self) -> dict[str, bytes]:
        """Every file in the package, keyed by its path inside the package.

        Both XMLs in a package refer to assets from the **package root**, not
        from their own directory: `props/ball.xml` names `assets/ball_right.png`.
        `MjSpec.from_file` resolves relative to the file, so it opens
        `props/assets/ball_right.png` and fails. Loading from a string with this
        dictionary is what makes one package work for a world and for a prop.
        """
        return {
            rel: (self.root / rel).read_bytes()
            for rel in self._shared()
        }

    def _shared(self) -> list[str]:
        """The shared section's files: everything outside the other sections.

        One section at a time. The whole package is verified, but only one
        section is ever handed to MuJoCo -- see `PLATFORM_DIR`.
        """
        others = _other_sections(self.manifest["schema"])
        return [
            path.relative_to(self.root).as_posix()
            for path in sorted(self.root.rglob("*"))
            if path.is_file()
            and not path.relative_to(self.root).as_posix().startswith(others)
        ]

    def spec(self, member: str) -> Any:
        """One of the package's MJCF files, as an `MjSpec`."""
        import mujoco

        return mujoco.MjSpec.from_string(
            (self.root / member).read_text("utf-8"), assets=self.assets()
        )

    def contact_budget(self, settle_steps: int = _SETTLE_STEPS) -> tuple[int, int]:
        """`(nconmax, njmax)` this world needs, by settling it and counting.

        mjwarp allocates contact and constraint arrays up front and raises
        `nconmax overflow` when a world exceeds them. mjlab's heuristic is sized
        for a robot on a plane, and a furnished room is not that: the bedroom
        package settles to 143 contacts and 1392 constraints **before a robot is
        in it**, against a heuristic that allowed 131.

        So the numbers come from running the world rather than from a rule about
        geom counts -- what fills these arrays is how the furniture rests, which a
        count cannot express. A room with twenty free bodies stacked on a shelf
        and one with them spread on the floor have the same geoms and very
        different contacts.

        Settling also proves the package simulates at all, which compiling does
        not, and it happens before mjlab builds anything.

        The headroom is for the robot that is about to walk in, and it is not
        symmetric: `njmax` is nearly free and `nconmax` is expensive, so the one
        that can be scaled is. See `_CONTACT_MARGIN` for the measurements.

        If `nconmax` is still short the failure is loud, at construction, and
        names the number it wanted -- which is what makes a tight value safe here
        and a generous one merely expensive.
        """
        import mujoco

        model = self.spec(self.manifest["world"]["file"]).compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        contacts, constraints = data.ncon, data.nefc
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
            contacts = max(contacts, data.ncon)
            constraints = max(constraints, data.nefc)
        return (
            contacts + _CONTACT_MARGIN,
            constraints * _CONSTRAINT_SCALE + _CONSTRAINT_MARGIN,
        )


def is_package_path(value: str) -> bool:
    """Whether `--scene` names a package rather than a registered scene id.

    A registered id is a bare module name, so anything with a separator or a
    `.map`/`.zip` suffix is a path. Checked by shape rather than by trying the registry
    first: a mistyped path would otherwise be reported as an unknown scene, with
    a list of built-in names and no mention of the file that does not exist.
    """
    return value.endswith((SUFFIX, ".zip")) or "/" in value or "\\" in value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _other_sections(schema: str) -> tuple[str, ...]:
    """Directories verified with the package and never handed to MuJoCo."""
    return (PLATFORM_DIR, *(_V2_SECTIONS if schema != SCHEMAS[0] else ()))


def _check_spawn(spawn: Any, *, required: bool) -> None:
    """Refuse a spawn that cannot be placed. Where it is, is `_placement`'s job.

    Optional in `/1`, which this repository's own exporter writes without one,
    and required in `/2`, whose contract makes it so.
    """
    if spawn is None:
        if required:
            raise PackageError("the package has no `spawn`; a kk-scene-package/2 must say "
                               "where the robot starts.")
        return
    if not isinstance(spawn, dict):
        raise PackageError("`spawn` is not an object.")
    position = spawn.get("position")
    if (not isinstance(position, list) or len(position) != 3
            or not all(_finite(v) for v in position)):
        raise PackageError(f"spawn.position {position!r} is not three finite numbers.")
    if not _finite(spawn.get("yaw")):
        raise PackageError(f"spawn.yaw {spawn.get('yaw')!r} is not a finite number of degrees.")
    site = spawn.get("site")
    if site is not None and not _SAFE_NAME.match(str(site)):
        raise PackageError(f"spawn.site {site!r} is not a name.")


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _safe_member(name: str) -> str:
    """A ZIP entry name with separators normalised, or a refusal."""
    cleaned = name.replace("\\", "/").removeprefix("./")
    if not cleaned or cleaned.startswith("/") or ".." in cleaned.split("/"):
        raise PackageError(f"the archive contains an unsafe path: {name!r}")
    return cleaned


def _extract(archive: Path) -> tuple[Path, Any]:
    holder = tempfile.TemporaryDirectory(prefix="kk-scene-")
    root = Path(holder.name)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = _safe_member(info.filename)
            target = root / member
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    # "compress this folder" archives carry one shared top directory; the
    # manifest inside names files without it.
    entries = [p for p in root.iterdir()]
    if len(entries) == 1 and entries[0].is_dir() and not (root / MANIFEST).exists():
        root = entries[0]
    return root, holder


def read_package(source: str | Path) -> ScenePackage:
    """Read and verify a package directory or `.zip`. Loads no model."""
    source = Path(source)
    holder = None
    if source.is_file() and source.suffix in (SUFFIX, ".zip"):
        root, holder = _extract(source)
    elif source.is_dir():
        root = source
    else:
        raise PackageError(
            f"{source} is neither a package directory nor a {SUFFIX}. A scene "
            f"package is a directory containing {MANIFEST}, or that directory "
            f"archived as {SUFFIX}."
        )

    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        raise PackageError(f"{source} has no {MANIFEST}, so it is not a scene package.")
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
    except json.JSONDecodeError as e:
        raise PackageError(f"{manifest_path} is not valid JSON: {e}") from None

    schema = manifest.get("schema")
    if schema not in SCHEMAS:
        raise PackageError(
            f"{source} declares schema {schema!r}; this importer reads "
            f"{', '.join(SCHEMAS)}."
        )
    use = manifest.get("use")
    if use == "training":
        raise PackageError(
            f"scene package {manifest.get('id')!r} declares use=\"training\": it is a "
            f"curriculum, not somewhere to watch a robot. Importing one as a place to "
            f"visit is what its own exporter refuses to do."
        )

    files = manifest.get("files")
    world = manifest.get("world")
    if not isinstance(files, dict) or not isinstance(world, dict):
        raise PackageError(f"{manifest_path} declares no `files` or no `world`.")
    if not _SAFE_PREFIX.match(str(world.get("attach", {}).get("prefix") or "scn_")):
        raise PackageError("the package's attach prefix is not an identifier.")
    if not _SAFE_NAME.match(str(manifest.get("id", ""))):
        raise PackageError(f"the package id {manifest.get('id')!r} is not a name.")
    _check_spawn(manifest.get("spawn"), required=schema != SCHEMAS[0])
    ground = (world.get("ground") or {}).get("geom")
    if ground is not None and not _SAFE_NAME.match(str(ground)):
        raise PackageError(f"the ground geom {ground!r} is not a name.")
    unsupported = sorted(set(manifest.get("requiredCapabilities") or ()) - CAPABILITIES)
    if unsupported:
        raise PackageError(
            f"scene package {manifest.get('id')!r} requires {unsupported}, which this "
            f"importer cannot run (it runs {sorted(CAPABILITIES)}). Running it anyway "
            f"would drop what it needs without a word."
        )

    declared: set[str] = set()
    for name, entry in files.items():
        member = _safe_member(name)
        path = root / member
        if not path.is_file():
            raise PackageError(f"{member} is declared in the manifest but not in the package.")
        if path.stat().st_size != entry.get("bytes"):
            raise PackageError(f"{member} is {path.stat().st_size} bytes, not {entry.get('bytes')}.")
        if _sha256(path) != str(entry.get("sha256", "")).lower():
            raise PackageError(f"{member} failed its SHA-256 check.")
        declared.add(member)
    for path in root.rglob("*"):
        if path.is_file():
            member = path.relative_to(root).as_posix()
            if member not in declared and member not in _UNDECLARED_OK:
                raise PackageError(f"{member} rode along without being declared in the manifest.")

    # MuJoCo's asset lookup falls back to matching on the base name, so two files
    # whose paths differ only by directory are one file as far as it is
    # concerned. Checked within the shared section, because that is the set this
    # importer ever hands to MuJoCo at once: `platforms/` holds the same scene in
    # another form, and each entity there carries its own copy of the textures it
    # uses -- 24 pairs in the bedroom package, none of them a problem. `/2`'s
    # robot and thumbnail are left out for the same reason (`_V2_SECTIONS`).
    others = _other_sections(schema)
    by_base: dict[str, str] = {}
    for member in sorted(m for m in declared if not m.startswith(others)):
        base = member.rsplit("/", 1)[-1].lower()
        if base in by_base:
            raise PackageError(
                f"{by_base[base]} and {member} have the same file name; MuJoCo's asset "
                f"lookup falls back to the base name and would confuse them."
            )
        by_base[base] = member

    world_file = _safe_member(str(world.get("file", "")))
    if world_file not in declared:
        raise PackageError(f"the manifest names {world_file!r} as its world, but does not declare it.")
    _check_world(root / world_file)

    return ScenePackage(root=root, manifest=manifest, _extracted=holder)


def _element_names(xml: str) -> list[str]:
    """The element names an XML document opens, in order.

    Not a parse: only which MJCF sections a file carries is needed. Still done by
    the XML rules rather than by matching the raw text -- comments and CDATA are
    removed first, because `<!-- <option/> -->` is not an option, and after that
    any `<name` really is an element, since XML forbids a bare `<` inside an
    attribute value. A DOCTYPE is refused outright: nothing in MJCF needs one,
    and entity declarations are how a small file becomes a huge one.
    """
    if re.search(r"<!DOCTYPE", xml, re.IGNORECASE):
        raise PackageError("the world file carries a DOCTYPE declaration.")
    stripped = re.sub(r"<!--.*?-->", " ", xml, flags=re.DOTALL)
    stripped = re.sub(r"<!\[CDATA\[.*?\]\]>", " ", stripped, flags=re.DOTALL)
    stripped = re.sub(r"<\?.*?\?>", " ", stripped, flags=re.DOTALL)
    return re.findall(r"<\s*([A-Za-z_][\w.:-]*)", stripped)


def _check_world(path: Path) -> None:
    names = _element_names(path.read_text("utf-8"))
    if not names or names[0] != "mujoco":
        raise PackageError(f"{path.name} does not start with <mujoco>.")
    present = set(names)
    for tag, why in _FORBIDDEN.items():
        if tag in present:
            raise PackageError(f"{path.name} carries <{tag}>: {why}.")


def load_package(source: str | Path) -> Scene:
    """Read a package and return the `Scene` that puts it on screen.

    The textures, materials and lights the terrain entity would otherwise carry
    are set empty on purpose: the package brings its own, and mjlab's default sun
    on top of a lit room is not that room.

    The world is attached in the frame `_placement` computes, so the package's
    spawn is where the robot starts, and its static collision geometry becomes
    the task's terrain (`_Terrain`). A package that names its ground also sets
    `terrain_type="plane"`: its ground replaces the task's whatever that was, and
    a generated one would first have moved every environment origin onto its own
    sub-terrain grid, away from the origin the spawn was moved to. A package that
    names no ground keeps the task's.
    """
    import mujoco

    package = read_package(source)
    prefix = package.prefix
    world_file = str(package.manifest["world"]["file"])

    # Poses are read from the compiled world, so a site inside a rotated body, a
    # floor in a `<body pos=...>` and `<compiler angle="degree">` are all already
    # resolved. Kinematics only: nothing is simulated.
    world = package.spec(world_file)
    _name_unnamed_geoms(world)
    model = world.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    pos, quat = _placement(package, model, data)
    terrain = _Terrain.of(package, model, data, pos, quat)

    def decorate(spec: Any) -> None:
        world = package.spec(world_file)
        _name_unnamed_geoms(world)
        for name in terrain.moved:
            world.delete(world.geom(name))
        heightfields = _heightfield_files(package, world)
        frame = spec.worldbody.add_frame(pos=pos, quat=quat)
        spec.attach(world, prefix=prefix, frame=frame)
        for path, content in heightfields.items():
            if path in spec.assets and spec.assets[path] != content:
                raise PackageError(f"the package's heightfield {path!r} collides with an "
                                   f"asset of the same path already in the environment.")
            spec.assets[path] = content
        terrain.install(spec)

    nconmax, njmax = package.contact_budget()
    return Scene(
        textures=(),
        materials=(),
        lights=(),
        terrain_type="plane" if terrain.replaces_ground else None,
        decorate=decorate,
        props=_entities(package),
        nconmax=nconmax,
        njmax=njmax,
    )


def _name_unnamed_geoms(world: Any) -> None:
    """Name the world's unnamed geoms, so one read in the compiled model can be
    found again in the spec. Deterministic: the same file gives the same names,
    which is what lets `decorate` re-parse the world and find them."""
    taken = {geom.name for geom in world.geoms if geom.name}
    for index, geom in enumerate(world.geoms):
        if not geom.name:
            name = _UNNAMED_GEOM.format(index)
            if name in taken:
                raise PackageError(f"the world already has a geom named {name!r}.")
            geom.name = name


def _heightfield_files(package: ScenePackage, world: Any) -> dict[str, bytes]:
    """The files the world's heightfields load, keyed by the path they name.

    Textures and meshes travel with the attached world; a heightfield's file is
    only opened when the environment compiles, and it is looked up in the
    **environment's** asset table rather than the one the world was parsed with.
    Measured with park-pump-track (mujoco 3.11): without this, the environment
    fails to build with "Error opening file 'assets/terrain.png'". The path it
    opens is the world's `meshdir` joined to the file, which is the file's path
    in the package -- measured the same with `meshdir`, `assetdir` and neither.
    """
    shared = package.assets()
    out: dict[str, bytes] = {}
    for hfield in world.hfields:
        if not hfield.file:
            continue
        path = "/".join(p for p in (world.meshdir.strip("/"), hfield.file) if p)
        if path not in shared:
            raise PackageError(f"heightfield {hfield.name!r} names {path!r}, which is not a "
                               f"file of the package's world.")
        out[path] = shared[path]
    return out


def _placement(package: ScenePackage, model: Any, data: Any) -> tuple[list[float], list[float]]:
    """The frame, `(pos, quat)`, that puts the spawn at the origin heading along +X.

    `spawn.position.z` is the ground height there, so subtracting it puts the
    ground under the spawn at z=0, where the task stands its robot. `yaw` is in
    degrees, zero along +X -- read as radians, 90 would be a quarter turn plus
    fourteen full ones, and a wrong heading is still a heading. With
    `spawn.site` the site's position in the compiled world is used, as the
    contract says; the heading stays `yaw`.
    """
    import mujoco
    import numpy as np

    spawn = package.spawn
    if spawn is None:
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    point = np.array([float(v) for v in spawn["position"]])
    site = spawn.get("site")
    if site:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(site))
        if sid < 0:
            raise PackageError(f"spawn.site {site!r} is not a site in the world.")
        point = data.site_xpos[sid].copy()
    half = math.radians(float(spawn["yaw"])) / 2.0
    quat = np.array([math.cos(half), 0.0, 0.0, -math.sin(half)])  # -yaw about z
    rotated = np.empty(3)
    mujoco.mju_rotVecQuat(rotated, point, quat)
    return (-rotated).tolist(), quat.tolist()


@dataclass(frozen=True)
class _Terrain:
    """The package's static collision geometry, rebuilt inside the task's terrain.

    Every geom that collides and does not move -- the floor, and just as much the
    start platform, the bridge deck, a ramp, a table top -- is taken out of the
    world and rebuilt in the body `terrain`, because that body is what the foot
    contact sensor counts. Only the declared ground was moved at first, and a
    jumper standing on narrow-bridge's start platform then read no foot in
    contact at all, while one on bedroom's floor read six.

    Rebuilt from the compiled model rather than moved: MjSpec cannot re-parent a
    geom, and the compiled values are the ones MuJoCo will actually use --
    defaults applied, angles in radians, the pose of every parent body folded in.
    A mesh's compiled pose also carries the recentring MuJoCo applies to its
    vertices, and compiling the rebuilt geom applies it again, so that offset is
    taken back out here. Anything that moves stays in the world as it was.
    """

    #: The geoms' names in the package, to be deleted from the world before it
    #: is attached.
    moved: tuple[str, ...]
    #: `add_geom` arguments, one per moved geom.
    geoms: tuple[dict[str, Any], ...]
    #: Whether the package named its ground, which then replaces the task's.
    replaces_ground: bool

    @classmethod
    def of(cls, package: ScenePackage, model: Any, data: Any,
           pos: list[float], quat: list[float]) -> _Terrain:
        import mujoco
        import numpy as np

        ground = package.ground
        if ground is not None:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ground)
            if gid < 0:
                raise PackageError(f"the manifest names {ground!r} as its ground, but the "
                                   f"world has no such geom.")
            if not _collides(model, gid):
                raise PackageError(f"the ground {ground!r} collides with nothing; with the "
                                   f"task's ground replaced by it the robot would fall.")
            if model.body_weldid[model.geom_bodyid[gid]] != 0:
                raise PackageError(f"the ground {ground!r} hangs from a body with a joint; "
                                   f"a ground does not move.")

        frame = np.array(quat)
        moved: list[str] = []
        geoms: list[dict[str, Any]] = []
        for gid in range(model.ngeom):
            if not _collides(model, gid) or model.body_weldid[model.geom_bodyid[gid]] != 0:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            kind = mujoco.mjtGeom(int(model.geom_type[gid]))
            if kind == mujoco.mjtGeom.mjGEOM_SDF:
                raise PackageError(f"geom {name!r} is an SDF, which this importer does not "
                                   f"rebuild.")
            at = np.empty(3)
            mujoco.mju_rotVecQuat(at, data.geom_xpos[gid], frame)
            at += np.array(pos)
            own = np.empty(4)
            mujoco.mju_mat2Quat(own, data.geom_xmat[gid])
            turned = np.empty(4)
            mujoco.mju_mulQuat(turned, frame, own)
            attrs: dict[str, Any] = {
                "name": GROUND_GEOM if name == ground else package.prefix + name,
                "type": kind,
                "contype": int(model.geom_contype[gid]),
                "conaffinity": int(model.geom_conaffinity[gid]),
                "condim": int(model.geom_condim[gid]),
                "priority": int(model.geom_priority[gid]),
                "friction": model.geom_friction[gid].tolist(),
                "solmix": float(model.geom_solmix[gid]),
                "solref": model.geom_solref[gid].tolist(),
                "solimp": model.geom_solimp[gid].tolist(),
                "margin": float(model.geom_margin[gid]),
                "gap": float(model.geom_gap[gid]),
                "group": int(model.geom_group[gid]),
                "rgba": model.geom_rgba[gid].tolist(),
            }
            # Assets stay where the attach puts them, under the package's
            # prefix; the rebuilt geom refers to them there.
            if model.geom_matid[gid] >= 0:
                attrs["material"] = package.prefix + mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_MATERIAL, int(model.geom_matid[gid]))
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = int(model.geom_dataid[gid])
                attrs["meshname"] = package.prefix + mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_MESH, mesh)
                # compiled = written * recentring, so written = compiled * recentring^-1
                back = np.empty(4)
                mujoco.mju_negQuat(back, model.mesh_quat[mesh])
                mujoco.mju_mulQuat(turned, turned.copy(), back)
                shift = np.empty(3)
                mujoco.mju_rotVecQuat(shift, model.mesh_pos[mesh], turned)
                at -= shift
            else:
                attrs["size"] = model.geom_size[gid].tolist()
            if kind == mujoco.mjtGeom.mjGEOM_HFIELD:
                attrs["hfieldname"] = package.prefix + mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_HFIELD, int(model.geom_dataid[gid]))
            attrs["pos"] = at.tolist()
            attrs["quat"] = turned.tolist()
            moved.append(name)
            geoms.append(attrs)
        return cls(moved=tuple(moved), geoms=tuple(geoms), replaces_ground=ground is not None)

    def install(self, spec: Any) -> None:
        """Rebuild the geometry in the task's terrain body, replacing the task's
        own ground when the package named one."""
        body = spec.body(GROUND_BODY)
        if body is None:
            raise PackageError(f"the environment has no body {GROUND_BODY!r} to carry the "
                               f"package's terrain; mjlab's terrain entity builds one.")
        # The poses are in world coordinates, so the body has to be at the
        # world's. mjlab attaches it at the identity; anything else is refused
        # rather than composed, because it would mean mjlab changed underneath.
        element = body
        while element is not None:
            if list(element.pos) != [0.0, 0.0, 0.0] or list(element.quat) != [1.0, 0.0, 0.0, 0.0]:
                raise PackageError(f"the {GROUND_BODY!r} body is not at the world origin, so "
                                   f"the package's terrain cannot be placed in it.")
            element = element.frame
        if self.replaces_ground:
            for geom in list(body.geoms):
                spec.delete(geom)
        for attrs in self.geoms:
            body.add_geom(**attrs)


def _collides(model: Any, gid: int) -> bool:
    return bool(model.geom_contype[gid] or model.geom_conaffinity[gid])


def _entities(package: ScenePackage) -> dict[str, Any]:
    """The props that are their own file, as mjlab entities.

    A prop carrying a `file` is the exporter saying it belongs **beside a robot**
    rather than in the room: kk-rl-mjlab writes its football that way because it
    places one per environment at that environment's origin. Rebuilding it as an
    entity is what makes the round trip a round trip -- exported from here,
    imported back, and placed the same way.

    A prop with no `file` is already a body in the world file, sitting where the
    room puts it, and there is nothing to add. That is how scene-generator writes
    a kettle on a bedroom floor. Reading one as the other would stack the room's
    contents at the origin, which is usually a plausible enough place that nobody
    would question it.
    """
    entries = [p for p in package.manifest.get("props", []) if p.get("file")]
    if not entries:
        return {}

    from mjlab.entity import EntityCfg

    out: dict[str, Any] = {}
    for entry in entries:
        name = str(entry["name"])
        if not _SAFE_NAME.match(name):
            raise PackageError(f"prop name {name!r} is not a name.")
        member = _safe_member(str(entry["file"]))
        if not (package.root / member).is_file():
            raise PackageError(f"prop {name!r} names {member}, which is not in the package.")
        position = entry.get("initPositionRelativeToRobot") or (0.0, 0.0, 0.0)
        out[name] = EntityCfg(
            spec_fn=lambda member=member: package.spec(member),
            init_state=EntityCfg.InitialStateCfg(pos=tuple(float(v) for v in position)),
        )
    return out
