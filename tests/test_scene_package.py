"""Importing a `kk-scene-package/1` or `/2`.

The failures worth pinning here all produce a running simulation rather than an
error:

- a package handed to `train.py` puts one room at the world origin, so exactly
  one environment is indoors and every other one trains on empty ground. It
  trains, it converges, nothing reports it;
- a prop's position read with the wrong meaning puts the room's contents at the
  origin, which is usually a plausible enough place to go unquestioned;
- an asset whose base name collides with another file is silently substituted,
  because MuJoCo's asset lookup falls back to the base name;
- the world spliced into the host instead of attached rewrites the robot's joint
  limits, mass and timestep, and compiles either way;
- `spawn` ignored puts the robot at the world origin, which on a real map is
  inside a shelf or on a ramp -- and it still stands there, so it looks fine;
- the task's plane left under the package fills in its pits, and a narrow bridge
  becomes a painted strip on a floor;
- a foot on the package's floor, platform or ramp is invisible to a contact
  sensor that counts the body `terrain`, so the robot reads as airborne.

The package fixtures are built here rather than committed, so the manifest can be
perturbed one field at a time and every hash stays real.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import zipfile
from pathlib import Path

import mujoco
import numpy as np
import pytest

from scenes.package import PackageError, is_package_path, load_package, read_package

#: A world with its own compiler settings and root defaults -- the two things
#: that rewrite a host robot when a package is spliced rather than attached.
WORLD = """<mujoco model="fixture_room">
  <compiler angle="degree" autolimits="true" assetdir="assets" eulerseq="xyz"/>
  <asset>
    <texture name="t" type="2d" builtin="checker" width="8" height="8"
             rgb1="0.2 0.3 0.4" rgb2="0.3 0.4 0.5"/>
    <material name="m" texture="t"/>
  </asset>
  <default>
    <geom condim="4" friction="0.5 0.01 0.0005" density="600"/>
  </default>
  <worldbody>
    <light name="ceiling" type="spot" pos="0 0 2.4" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor_geom" type="box" pos="0 0 -0.02" size="3 3 0.02" material="m"/>
    <body name="wall" pos="0 2.9 1.2">
      <geom name="wall_geom" type="box" size="3 0.06 1.2" material="m"/>
    </body>
    <body name="kettle" pos="0.4 0.2 0.3">
      <freejoint name="kettle_free"/>
      <geom name="kettle_geom" type="box" size="0.06 0.06 0.08" material="m"/>
    </body>
    <site name="spawn" pos="-0.6 -1.2 0" euler="0 0 90" type="sphere" size="0.02" group="4"/>
    <site name="ahead" pos="-0.6 -0.2 0" type="sphere" size="0.02" group="4"/>
  </worldbody>
</mujoco>
"""

#: The task's ground as mjlab builds it: a plane `terrain` in a body `terrain`,
#: attached through an identity frame (`mjlab.terrains.terrain_entity`).
TERRAIN = ('<frame><body name="terrain">'
           '<geom name="terrain" type="plane" size="0 0 0.01"/></body></frame>')


def _host(body: str = "", after: str = "") -> mujoco.MjSpec:
    """An environment the way `decorate` meets one: a ground, and whatever else."""
    return mujoco.MjSpec.from_string(
        f'<mujoco><compiler angle="radian"/><worldbody>{TERRAIN}{body}</worldbody>'
        f"{after}</mujoco>"
    )


def _compiled(package: Path, host: mujoco.MjSpec | None = None):
    spec = host if host is not None else _host()
    load_package(package).decorate(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _id(model, kind, name: str) -> int:
    index = mujoco.mj_name2id(model, kind, name)
    assert index >= 0, f"{name} is not in the compiled model"
    return index


#: A prop written as its own model, the way kk-rl-mjlab writes one.
BALL = """<mujoco model="ball">
  <compiler angle="radian"/>
  <worldbody>
    <body name="ball">
      <joint type="free"/>
      <geom name="ball" type="sphere" size="0.055" mass="0.1" condim="6"/>
    </body>
  </worldbody>
</mujoco>
"""


def _bytes(body: str | bytes) -> bytes:
    return body if isinstance(body, bytes) else body.encode()


def _manifest(files: dict[str, str | bytes], **overrides):
    listed = {
        name: {"bytes": len(_bytes(body)), "sha256": hashlib.sha256(_bytes(body)).hexdigest()}
        for name, body in files.items()
    }
    manifest = {
        "schema": "kk-scene-package/1",
        "id": "fixture_room",
        "description": "a room",
        "use": "watching",
        "world": {
            "file": "scene.xml",
            "assetDir": "assets",
            "terrainType": "none",
            "counts": {"geoms": 3, "bodies": 2, "lights": 1},
            "decorativeGeoms": [],
            "collidingGeoms": [],
            "attach": {"prefix": "scn_"},
        },
        "props": [{"name": "kettle", "body": "kettle", "freeJoint": True,
                   "position": [0.4, 0.2, 0.3], "geoms": ["kettle_geom"]}],
        "spawn": {"position": [-0.6, -1.2, 0.0], "yaw": 90.0, "site": "spawn"},
        "rules": {},
        "files": listed,
    }
    manifest.update(overrides)
    return manifest


def write_package(root: Path, *, files: dict[str, str | bytes] | None = None,
                  **overrides) -> Path:
    files = files if files is not None else {"scene.xml": WORLD}
    root.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_bytes(body))
    manifest = _manifest(files, **overrides)
    (root / "scene-package.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    return root


#: What `/2` adds beside the world, as kingkong_design writes it. The two
#: previews share a base name on purpose: every map in that library does.
V2_EXTRA = {
    "preview/preview.png": "the map, for people",
    "robot/preview/preview.png": "the robot, for people",
    "robot/skin-package.json": "{}",
}


def write_map(root: Path, *, files: dict[str, str | bytes] | None = None, **overrides) -> Path:
    """A `kk-scene-package/2`: the world, the robot beside it, a spawn."""
    files = {**(files if files is not None else {"scene.xml": WORLD}), **V2_EXTRA}
    overrides.setdefault("schema", "kk-scene-package/2")
    overrides.setdefault("requiredCapabilities", ["rigid"])
    return write_package(root, files=files, **overrides)


@pytest.fixture
def package(tmp_path):
    return write_package(tmp_path / "room.scene")


def test_reads_a_package_and_names_how_to_mount_it(package):
    read = read_package(package)
    assert read.id == "fixture_room"
    assert read.prefix == "scn_"
    assert read.spawn["site"] == "spawn"


def test_reads_a_zip_and_keeps_the_files_alive(tmp_path):
    """A `.zip` is expanded somewhere temporary, and the world is not read until
    mjlab builds the environment -- long after the reader returned."""
    source = write_package(tmp_path / "room.scene")
    archive = tmp_path / "room.scene.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(source).as_posix())
    shutil.rmtree(source)

    read = read_package(archive)
    assert read.world.is_file()
    # The temporary directory has to outlive the call that made it.
    assert read.spec("scene.xml").compile().ngeom == 3


def test_reads_a_v2_map_and_hands_mujoco_only_its_world(tmp_path):
    """`/2` carries the robot the map was authored around and a thumbnail. The
    robot here is the task's, and neither is handed to MuJoCo -- where every map
    in kingkong_design's library would collide on `preview.png`.

    The control is the same files declared as `/1`, where `robot/` and
    `preview/` are ordinary directories: refused on that same base name, which
    is what shows the collision in the fixture is real.
    """
    read = read_package(write_map(tmp_path / "room.map"))
    assert read.manifest["schema"] == "kk-scene-package/2"
    assert not [name for name in read.assets() if name.startswith(("robot/", "preview/"))]
    assert "scene.xml" in read.assets()

    as_v1 = write_package(tmp_path / "room.scene", files={"scene.xml": WORLD, **V2_EXTRA})
    with pytest.raises(PackageError, match="same file name"):
        read_package(as_v1)


def test_refuses_a_version_it_does_not_know(tmp_path):
    """Read as the nearest known version, a `/3` would load and be wrong in
    whatever way `/3` differs. The contract says an old consumer rejects it."""
    with pytest.raises(PackageError, match="kk-scene-package/3"):
        read_package(write_map(tmp_path / "room.map", schema="kk-scene-package/3"))


def test_refuses_a_v2_map_without_a_spawn(tmp_path):
    with pytest.raises(PackageError, match="spawn"):
        read_package(write_map(tmp_path / "room.map", spawn=None))


@pytest.mark.parametrize("spawn", [
    {"position": [0.0, 0.0], "yaw": 0.0},
    {"position": [0.0, 0.0, 0.0], "yaw": "90"},
    {"position": [0.0, float("nan"), 0.0], "yaw": 0.0},
])
def test_refuses_a_spawn_that_cannot_be_placed(tmp_path, spawn):
    with pytest.raises(PackageError, match="spawn"):
        read_package(write_map(tmp_path / "room.map", spawn=spawn))


def test_refuses_a_capability_it_cannot_run(tmp_path):
    """Running it anyway is what the contract forbids: whatever needed the
    capability would be dropped, and the rest would look complete."""
    with pytest.raises(PackageError, match="teleport"):
        read_package(write_map(tmp_path / "room.map",
                               requiredCapabilities=["rigid", "teleport"]))
    assert read_package(write_map(tmp_path / "ok.map",
                                  requiredCapabilities=["rigid", "hfield"])).id


def test_refuses_a_file_that_does_not_hash_to_what_was_declared(tmp_path):
    root = write_package(tmp_path / "room.scene")
    (root / "scene.xml").write_text(WORLD.replace("3 3 0.02", "9 9 0.02"), "utf-8")
    with pytest.raises(PackageError, match="SHA-256"):
        read_package(root)


def test_refuses_a_file_that_rode_along_undeclared(package):
    (package / "assets").mkdir(exist_ok=True)
    (package / "assets" / "extra.png").write_bytes(b"not declared")
    with pytest.raises(PackageError, match="without being declared"):
        read_package(package)


def test_refuses_two_files_with_the_same_base_name(tmp_path):
    """MuJoCo's asset lookup falls back to the base name, so these are one file
    as far as it is concerned -- including against a robot's own mesh."""
    root = write_package(
        tmp_path / "room.scene",
        files={"scene.xml": WORLD, "assets/pattern.png": "a", "extra/pattern.png": "b"},
    )
    with pytest.raises(PackageError, match="same file name"):
        read_package(root)


def test_refuses_a_world_that_sets_the_hosts_physics(tmp_path):
    root = write_package(
        tmp_path / "room.scene",
        files={"scene.xml": WORLD.replace("<worldbody>", '<option timestep="0.02"/><worldbody>')},
    )
    with pytest.raises(PackageError, match="<option>"):
        read_package(root)


def test_does_not_mistake_a_comment_for_a_section(tmp_path):
    root = write_package(
        tmp_path / "room.scene",
        files={"scene.xml": WORLD.replace("<worldbody>", "<!-- no <option/> here --><worldbody>")},
    )
    assert read_package(root).id == "fixture_room"


def test_refuses_a_training_scene(tmp_path):
    root = write_package(tmp_path / "room.scene", use="training")
    with pytest.raises(PackageError, match="training"):
        read_package(root)


def test_refuses_a_path_outside_the_archive(tmp_path):
    source = write_package(tmp_path / "room.scene")
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(source / "scene-package.json", "scene-package.json")
        zf.writestr("../escaped.xml", "<mujoco/>")
    with pytest.raises(PackageError, match="unsafe path"):
        read_package(archive)


def test_is_package_path_tells_a_path_from_a_registered_id():
    assert is_package_path("out/bedroom.scene")
    assert is_package_path("bedroom.scene.zip")
    assert not is_package_path("studio")


# ── What the package does once it is a Scene ──────────────────────────────


def test_a_prop_in_the_world_does_not_become_an_entity(package):
    """No `file` means the prop is a body in the world, sitting where the room
    puts it. Turning it into an entity would move it to each environment's
    origin -- a plausible enough place that nobody would question it."""
    scene = load_package(package)
    assert scene.props == {}


def test_a_prop_with_its_own_file_becomes_an_entity(tmp_path):
    """A `file` means the exporter placed it beside a robot, per environment.
    This is the shape kk-rl-mjlab's own exporter writes, so a scene exported
    from here and imported back gets its ball placed the same way."""
    root = write_package(
        tmp_path / "pitch.scene",
        files={"scene.xml": WORLD, "props/ball.xml": BALL},
        props=[{"name": "ball", "file": "props/ball.xml",
                "initPositionRelativeToRobot": [0.4, 0.0, 0.055], "freeJoint": True}],
    )
    scene = load_package(root)
    assert list(scene.props) == ["ball"]
    assert tuple(scene.props["ball"].init_state.pos) == (0.4, 0.0, 0.055)
    # It has to build, not merely be configured.
    assert scene.props["ball"].spec_fn().compile().ngeom == 1


def test_the_contact_budget_is_measured_rather_than_guessed(package):
    """mjwarp allocates these up front and fails with `nconmax overflow` after
    everything has compiled and every banner has printed. A count of geoms
    cannot express it: what fills the arrays is how the furniture comes to rest."""
    scene = load_package(package)
    settled = mujoco.MjData(read_package(package).spec("scene.xml").compile())
    model = read_package(package).spec("scene.xml").compile()
    settled = mujoco.MjData(model)
    for _ in range(600):
        mujoco.mj_step(model, settled)
    assert scene.nconmax > settled.ncon
    assert scene.njmax > settled.nefc


def test_the_world_attaches_without_rewriting_the_host(package):
    """The reason a package is attached and never spliced.

    The control group is the same host compiled alone: without it this test
    passes against a host that was never contaminated to begin with.
    """
    host = f"""<mujoco model="host">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="0.005"/>
      <default><geom condim="3" friction="1 0.005 0.0001" density="1000"/></default>
      <worldbody>{TERRAIN}<body name="robot" pos="0 0 1">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom name="g_robot" type="box" size=".1 .1 .1"/>
      </body></worldbody>
    </mujoco>"""
    alone = mujoco.MjSpec.from_string(host).compile()
    gid_alone = _id(alone, mujoco.mjtObj.mjOBJ_GEOM, "g_robot")
    body_alone = _id(alone, mujoco.mjtObj.mjOBJ_BODY, "robot")

    spec = mujoco.MjSpec.from_string(host)
    load_package(package).decorate(spec)
    model = spec.compile()
    gid = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_robot")
    body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    hip = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip")

    assert model.ngeom > alone.ngeom, "the room has to actually be there"
    assert list(model.jnt_range[hip]) == list(alone.jnt_range[0])
    assert model.geom_condim[gid] == alone.geom_condim[gid_alone]
    assert model.body_mass[body] == alone.body_mass[body_alone]
    assert model.opt.timestep == alone.opt.timestep


def test_the_imported_geometry_collides(package):
    """A room the robot walks through is not a room. Registered scenes may only
    add geometry that cannot be touched; this one is the stated exception."""
    model, _ = _compiled(package)
    wall = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "scn_wall_geom")
    assert model.geom_contype[wall] != 0
    assert model.geom_conaffinity[wall] != 0


# ── Where the robot starts ──────────────────────────────────────────────


def test_the_spawn_lands_at_the_origin_heading_along_x(package):
    """The robot starts at its environment's origin facing +X, so the world is
    moved to put the package's spawn there.

    `ahead` is one metre from the spawn along its heading of 90 **degrees**. The
    assertion on it is what catches a heading read as radians or not applied at
    all; the one on the spawn alone would pass with any heading.
    """
    model, data = _compiled(package)
    spawn = data.site_xpos[_id(model, mujoco.mjtObj.mjOBJ_SITE, "scn_spawn")]
    ahead = data.site_xpos[_id(model, mujoco.mjtObj.mjOBJ_SITE, "scn_ahead")]
    assert list(spawn) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert list(ahead) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_a_package_with_no_spawn_stays_where_it_was(tmp_path):
    """The control for the test above: `/1` may leave `spawn` out -- this
    repository's own exporter does -- and then nothing moves. Without it, the
    placement could be moving every package by a constant and still pass."""
    root = write_package(tmp_path / "room.scene", spawn=None)
    model, data = _compiled(root)
    spawn = data.site_xpos[_id(model, mujoco.mjtObj.mjOBJ_SITE, "scn_spawn")]
    assert list(spawn) == pytest.approx([-0.6, -1.2, 0.0], abs=1e-9)


def test_the_spawn_height_is_the_ground_there(tmp_path):
    """`spawn.position.z` is the height of the ground at the spawn, and the task
    stands its robot on z=0. So a spawn on a platform brings the platform's top
    to z=0; left where it was, the robot would start inside the platform."""
    raised = WORLD.replace(
        "<worldbody>",
        '<worldbody><geom name="platform" type="box" pos="-0.6 -1.2 0.05" size="0.3 0.3 0.05"/>',
    )
    assert raised != WORLD
    root = write_package(tmp_path / "room.scene", files={"scene.xml": raised},
                         spawn={"position": [-0.6, -1.2, 0.1], "yaw": 0.0})
    model, data = _compiled(root)
    platform = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "scn_platform")
    top = data.geom_xpos[platform][2] + model.geom_size[platform][2]
    assert top == pytest.approx(0.0, abs=1e-9)


def test_a_spawn_site_wins_over_the_position(tmp_path):
    """The contract: with `spawn.site`, the site's compiled position is the
    spawn. The position here is deliberately somewhere else."""
    root = write_package(tmp_path / "room.scene",
                         spawn={"position": [9.0, 9.0, 0.0], "yaw": 90.0, "site": "spawn"})
    model, data = _compiled(root)
    spawn = data.site_xpos[_id(model, mujoco.mjtObj.mjOBJ_SITE, "scn_spawn")]
    assert list(spawn) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


# ── The ground is the package's ─────────────────────────────────────────

#: A floor 0.16 m below a platform at the spawn -- narrow-bridge's shape. The
#: platform sits in a body with its own pose, the way bedroom's floor does.
BRIDGE = WORLD.replace(
    '<geom name="floor_geom" type="box" pos="0 0 -0.02" size="3 3 0.02" material="m"/>',
    '<geom name="floor_geom" type="box" pos="0 0 -0.18" size="3 3 0.02" material="m"/>'
    '<body name="stand" pos="-0.6 -1.2 0">'
    '<geom name="platform" type="box" pos="0 0 -0.08" size="0.3 0.3 0.08"/></body>',
)
assert BRIDGE != WORLD


def _world_with_ground(ground: str | None) -> dict:
    world = _manifest({"scene.xml": BRIDGE})["world"]
    if ground is not None:
        world["ground"] = {"geom": ground}
    return world


def _surface(model, data, x: float, y: float) -> float:
    """The height of the first surface below (x, y)."""
    geom = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, np.array([x, y, 3.0]), np.array([0.0, 0.0, -1.0]),
                             None, 1, -1, geom)
    assert distance >= 0, f"nothing below ({x}, {y})"
    return 3.0 - distance


def test_the_packages_ground_replaces_the_tasks(tmp_path):
    """narrow-bridge's basin is 0.16 m below the bridge. With the task's plane
    left at z=0 it is filled in, and the bridge is a strip painted on a floor.

    (0, 1) is one metre to the robot's left, off the platform, over the basin.
    The control is the same package naming no ground: the task's plane stays,
    and the same ray stops at 0 -- which is what shows the ray can see a plane.
    """
    replaced = write_package(tmp_path / "bridge.scene", files={"scene.xml": BRIDGE},
                             world=_world_with_ground("floor_geom"))
    model, data = _compiled(replaced)
    ground = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    assert model.geom_type[ground] == mujoco.mjtGeom.mjGEOM_BOX, "the package's floor"
    assert mujoco.mjtGeom.mjGEOM_PLANE not in set(model.geom_type)
    assert _surface(model, data, 0.0, 1.0) == pytest.approx(-0.16, abs=1e-6)

    kept = write_package(tmp_path / "kept.scene", files={"scene.xml": BRIDGE},
                         world=_world_with_ground(None))
    model, data = _compiled(kept)
    ground = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    assert model.geom_type[ground] == mujoco.mjtGeom.mjGEOM_PLANE
    assert _surface(model, data, 0.0, 1.0) == pytest.approx(0.0, abs=1e-6)


def test_a_foot_on_the_packages_platform_counts_as_ground(tmp_path):
    """Every jumper task's `feet_ground_contact` counts contacts with the body
    `terrain`, and so does this sensor. Measured in `jumper.flat` before the
    package's static geometry was moved there: 0 feet in contact on
    narrow-bridge's start platform, 6 on bedroom's floor -- the declared ground
    alone had been moved.

    The control attaches the same world the same way without that move. Both
    feet are checked to be resting on the platform, so a zero in the control is
    the sensor not seeing it rather than the foot being somewhere else.
    """
    foot = ('<body name="foot" pos="0 0 0.1"><freejoint/>'
            '<geom name="foot" type="sphere" size="0.02"/></body>')
    sensor = '<sensor><contact name="touch" geom1="foot" body2="terrain" data="found"/></sensor>'
    root = write_package(tmp_path / "bridge.scene", files={"scene.xml": BRIDGE},
                         world=_world_with_ground("floor_geom"))

    def settle(spec) -> tuple[float, float]:
        model = spec.compile()
        data = mujoco.MjData(model)
        for _ in range(500):
            mujoco.mj_step(model, data)
        touch = model.sensor_adr[_id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch")]
        return float(data.sensordata[touch]), float(data.xpos[_id(model, mujoco.mjtObj.mjOBJ_BODY, "foot")][2])

    host = _host(foot, sensor)
    load_package(root).decorate(host)
    touching, height = settle(host)
    assert height == pytest.approx(0.02, abs=2e-3), "the foot is on the platform"
    assert touching > 0

    # Control: the world attached where it belongs, the task's plane removed --
    # everything but the move into `terrain`. Turning by -90 degrees takes
    # (x, y) to (y, -x), so the spawn (-0.6, -1.2) needs a shift of (1.2, -0.6).
    control = mujoco.MjSpec.from_string(
        '<mujoco><compiler angle="radian"/><worldbody><frame><body name="terrain"/></frame>'
        f"{foot}</worldbody>{sensor}</mujoco>"
    )
    half = np.radians(90.0) / 2
    control.attach(read_package(root).spec("scene.xml"), prefix="scn_",
                   frame=control.worldbody.add_frame(pos=[1.2, -0.6, 0.0],
                                                     quat=[np.cos(half), 0, 0, -np.sin(half)]))
    touching, height = settle(control)
    assert height == pytest.approx(0.02, abs=2e-3), "the control's foot is on the platform too"
    assert touching == 0


def test_a_mesh_keeps_its_place_when_it_is_rebuilt(tmp_path):
    """MuJoCo recentres a mesh and folds the offset into the geom's compiled
    pose, so a geom rebuilt from that pose gets the offset twice unless it is
    taken back out. Two of kingkong_design's maps have static meshes --
    switchback-slopes' ramps among them.

    Compared relative to the spawn site, which moves with the world, so the
    comparison needs no copy of the placement arithmetic. The first assertion
    is the control: a mesh centred on its own origin passes without the
    correction.
    """
    ramp = WORLD.replace("<asset>", '<asset><mesh name="wedge" '
                         'vertex="0.5 0.5 0  0.9 0.5 0  0.5 0.9 0  0.5 0.5 0.3"/>').replace(
        "<worldbody>",
        '<worldbody><body name="ramp_body" pos="1 1 0" euler="0 0 30">'
        '<geom name="ramp" type="mesh" mesh="wedge" euler="10 0 0"/></body>',
    )
    assert ramp.count("wedge") == 2
    root = write_package(tmp_path / "ramp.scene", files={"scene.xml": ramp})

    alone = read_package(root).spec("scene.xml").compile()
    assert np.linalg.norm(alone.mesh_pos[0]) > 0.1, "the mesh has to be off-centre"
    alone_data = mujoco.MjData(alone)
    mujoco.mj_kinematics(alone, alone_data)

    model, data = _compiled(root)
    assert model.geom_bodyid[_id(model, mujoco.mjtObj.mjOBJ_GEOM, "scn_ramp")] == \
        _id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")

    def relative(m, d, geom: str, site: str):
        g, s = _id(m, mujoco.mjtObj.mjOBJ_GEOM, geom), _id(m, mujoco.mjtObj.mjOBJ_SITE, site)
        frame = d.site_xmat[s].reshape(3, 3)
        return (frame.T @ (d.geom_xpos[g] - d.site_xpos[s]),
                frame.T @ d.geom_xmat[g].reshape(3, 3))

    want_pos, want_rot = relative(alone, alone_data, "ramp", "spawn")
    got_pos, got_rot = relative(model, data, "scn_ramp", "scn_spawn")
    assert got_pos == pytest.approx(want_pos, abs=1e-9)
    assert got_rot.ravel() == pytest.approx(want_rot.ravel(), abs=1e-9)


def test_a_heightfield_file_reaches_the_environment(tmp_path):
    """A heightfield's file is opened when the environment compiles, through the
    environment's asset table: park-pump-track failed to build with "Error
    opening file 'assets/terrain.png'". The control is the same world attached
    plainly, which fails exactly that way -- without it this test would pass on
    a MuJoCo that had stopped needing the help."""
    hill = struct.pack("<ii", 2, 2) + struct.pack("<4f", 0.0, 0.5, 0.5, 1.0)
    world = """<mujoco model="hill">
      <asset><hfield name="hill" file="assets/hill.bin" size="2 2 0.2 0.05"/></asset>
      <worldbody><geom name="ground" type="hfield" hfield="hill"/>
        <site name="spawn" pos="0 0 0"/></worldbody>
    </mujoco>"""
    base = _manifest({"scene.xml": world})["world"]
    root = write_map(tmp_path / "hill.map", files={"scene.xml": world, "assets/hill.bin": hill},
                     world={**base, "ground": {"geom": "ground"}}, props=[],
                     spawn={"position": [0.0, 0.0, 0.0], "yaw": 0.0},
                     requiredCapabilities=["rigid", "hfield"])
    model, _ = _compiled(root)
    ground = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    assert model.geom_type[ground] == mujoco.mjtGeom.mjGEOM_HFIELD

    plain = _host()
    plain.attach(read_package(root).spec("scene.xml"), prefix="scn_",
                 frame=plain.worldbody.add_frame())
    with pytest.raises(ValueError, match="hill.bin"):
        plain.compile()


def test_an_imported_map_sets_a_plane_under_it_only_when_it_brings_a_ground(tmp_path):
    """A package's ground replaces the task's, and a generated ground would first
    have put every environment origin on its own sub-terrain grid, away from the
    spawn. With no ground named, the task's is left exactly as it was."""
    with_ground = write_package(tmp_path / "a.scene", files={"scene.xml": BRIDGE},
                                world=_world_with_ground("floor_geom"))
    without = write_package(tmp_path / "b.scene", files={"scene.xml": BRIDGE},
                            world=_world_with_ground(None))
    assert load_package(with_ground).terrain_type == "plane"
    assert load_package(without).terrain_type is None


# ── A package in a real environment ────────────────────────────────────


def test_the_terrain_body_is_the_one_mjlab_builds(tmp_path):
    """The whole terrain move rests on mjlab building its ground as a body
    `terrain` at the world origin. Checked on the real scene rather than on the
    stand-in above, so an mjlab upgrade that renames or moves it fails here
    instead of leaving the package's floor where no sensor counts it."""
    from mjlab.scene import Scene as MjlabScene

    import scenes
    import tasks

    root = write_map(tmp_path / "bridge.map", files={"scene.xml": BRIDGE},
                     world=_world_with_ground("floor_geom"))
    cfg = tasks.load_env_cfg("jumper.flat")
    scenes.apply_scene(cfg, load_package(root), "bridge")
    cfg.scene.num_envs = 1
    model = MjlabScene(cfg.scene, device="cpu").spec.compile()
    terrain = _id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
    for name in ("terrain", "scn_platform", "scn_wall_geom"):
        assert model.geom_bodyid[_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)] == terrain, name
    assert model.geom_type[_id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")] == \
        mujoco.mjtGeom.mjGEOM_BOX
    kettle = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "scn_kettle_geom")
    assert model.geom_bodyid[kettle] != terrain, "what moves stays out of the terrain"
