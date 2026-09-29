"""Export the robot as a self-contained package a web simulator can import.

The consumer is a separate repository that must not know anything about this
one: not its layout, not its Python, not its config classes. So everything a
runner needs travels in the package, and the package is the only thing that
crosses the boundary.

## What the package carries, and why each part is here

The MJCF in the package is **not** `jumper.xml`. That file is the robot as
exported from CAD, and on its own it is wrong for this simulation in a way that
does not raise: its `<default class="collision">` sets `solref="0.008"`, correct
for its own `timestep="0.001"` and unstable at the 0.005 the policies are
trained at. MuJoCo's constraint solver needs a contact timeconst spanning at
least two timesteps; below that it overshoots, contacts inject energy, and the
robot climbs. This repository replaces the contact parameters in the same place
it raises the timestep (`constants.HYBRID_COLLISION`), and an export that copies
only half of that is a robot that flies.

So the exported model has, baked in:

  * the collision configuration this repository actually trains with -- per-leg
    collision bitmasks, `condim`, foot friction and priority, `solref`/`solimp`;
  * the solver settings, which upstream exist **only on the compiled model**
    (`sim.py` applies them after `spec.compile()`) and therefore cannot be read
    out of any XML;
  * the calibrated standing pose as a keyframe named `init_state`. It has to be
    a keyframe: `mjModel.qpos0` is the joint *reference*, not the initial pose,
    so writing the home pose there would redefine every joint's zero and leave
    the robot folded with its `qpos` reading exactly right. Measured on this
    model, that mistake puts the feet 81 mm off the ground.

What it deliberately does **not** carry is an `<actuator>` block. Actuators are
how you drive the robot, not what the robot is: this repository drives it with
`<motor>` and computes the PD outside MuJoCo, a browser may prefer `<position>`
servos, and hardware does neither. `robot-package.json` states the hardware
instead -- gains, torque limit, armature, and the measured torque-speed curve --
and leaves the representation to the importer.

## Usage

    python assets/jumper/tools/export_web_robot.py --out /tmp/jumper.robot

Needs the editable install (`pip install -e .`), because it reads the collision
and simulation configuration from this repository rather than copying it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from tasks.jumper.common import actuator as servo
from tasks.jumper.common import constants as C

SCHEMA = "kk-robot-package/1"

#: MJCF spells the solver enum with its own capitalisation; the config uses
#: lowercase. An unrecognised keyword is a compile error, which is the good
#: case, but only for whoever compiles it next.
_SOLVER_KEYWORD = {"newton": "Newton", "cg": "CG", "pgs": "PGS"}


def _sim_options() -> dict[str, str]:
    mj = make_velocity_env_cfg().sim.mujoco
    return {
        "timestep": f"{mj.timestep:g}",
        "integrator": mj.integrator,
        "solver": _SOLVER_KEYWORD[mj.solver],
        "iterations": str(mj.iterations),
        "ls_iterations": str(mj.ls_iterations),
        # The hexapod's own override: six feet make far more contacts than the
        # quadruped this default was chosen for.
        "ccd_iterations": "50",
        "gravity": " ".join(f"{v:g}" for v in mj.gravity),
    }


def _resolved_collision() -> dict[str, dict[str, str]]:
    """Per-geom collision attributes, resolved by this repository's own code.

    `HYBRID_COLLISION` keys its masks and contact parameters by regex over geom
    names. Reimplementing that resolution in the exporter would create a second
    copy of it, which is the failure this whole package exists to avoid -- so
    build the spec the trainer builds, let `edit_spec` write the attributes, and
    read them back off the geoms.
    """
    spec = C.get_spec(None)
    C.HYBRID_COLLISION.edit_spec(spec)

    out: dict[str, dict[str, str]] = {}
    for geom in spec.geoms:
        attrs = {
            "contype": str(int(geom.contype)),
            "conaffinity": str(int(geom.conaffinity)),
        }
        if int(geom.contype) | int(geom.conaffinity):
            attrs["condim"] = str(int(geom.condim))
            attrs["priority"] = str(int(geom.priority))
            attrs["friction"] = " ".join(f"{v:g}" for v in geom.friction)
            attrs["solref"] = " ".join(f"{v:g}" for v in geom.solref)
            attrs["solimp"] = " ".join(f"{v:g}" for v in geom.solimp)
        out[geom.name] = attrs
    return out


def _build_model_xml(collision: dict[str, dict[str, str]]) -> tuple[ET.Element, list[ET.Element]]:
    source_xml = Path(C.JUMPER_XML)
    text = source_xml.read_text("utf-8")
    # The CAD export's own comment contains `--`, which is not legal inside an
    # XML comment; ElementTree refuses it. Strip comments from the derived file
    # only -- the original travels alongside, byte for byte.
    supplied = ET.fromstring(_strip_comments(text))

    model = ET.Element("mujoco", model=supplied.get("model", "robot"))
    ET.SubElement(model, "compiler", angle="radian", meshdir="meshes/", autolimits="true")
    ET.SubElement(model, "option", **_sim_options())
    model.append(supplied.find("default"))
    model.append(supplied.find("asset"))

    body = supplied.find("worldbody/body")
    for parent in body.iter():
        for geom in list(parent.findall("geom")):
            name = geom.get("name")
            if geom.get("class") not in {"visual", "collision"}:
                # Reference markers from CAD, not robot geometry.
                parent.remove(geom)
                continue
            if name not in collision:
                raise SystemExit(f"geom {name!r} is not in the training spec")
            # Written explicitly, so the `<default class="collision">` block's
            # own solref cannot reach a geom by inheritance.
            for key, value in collision[name].items():
                geom.set(key, value)
    world = ET.SubElement(model, "worldbody")
    world.append(body)

    joints = [j for j in body.iter("joint") if j.get("type") != "free"]
    keyframe = ET.SubElement(model, "keyframe")
    qpos = [0.0, 0.0, C.STAND_Z, 1.0, 0.0, 0.0, 0.0] + [C.HOME[j.get("name")] for j in joints]
    ET.SubElement(keyframe, "key", name="init_state", qpos=" ".join(f"{v:g}" for v in qpos))

    ET.indent(model)
    return model, joints


def _strip_comments(text: str) -> str:
    out, i = [], 0
    while True:
        start = text.find("<!--", i)
        if start < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:start])
        end = text.find("-->", start)
        if end < 0:
            return "".join(out)
        i = end + 3


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(C.JUMPER_XML).parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return None


def export(out: Path) -> None:
    collision = _resolved_collision()
    model, joints = _build_model_xml(collision)

    out.mkdir(parents=True, exist_ok=True)
    (out / "meshes").mkdir(exist_ok=True)
    for stale in (out / "meshes").glob("*"):
        stale.unlink()

    source_xml = Path(C.JUMPER_XML)
    shutil.copy2(source_xml, out / "source.xml")

    # The package keeps every mesh flat under `meshes/`, which is the `meshdir`
    # its model names. `jumper.xml` does not: V1.6.1 keeps the visual meshes in
    # `urdf/jumper/meshes/visual/` and the collision ones in `meshes/`, each
    # path relative to its own `meshdir`. So each file is found where the source
    # says and renamed to its basename -- and two sources with one basename are
    # refused, because flattening them would put one mesh on both bodies.
    compiler = ET.parse(source_xml).getroot().find("compiler")
    source_mesh_dir = source_xml.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    placed: dict[str, Path] = {}
    for mesh in model.findall("asset/mesh"):
        name = mesh.get("file")
        if not name:
            raise SystemExit(f"mesh {mesh.get('name')!r} names no file")
        src = (source_mesh_dir / name).resolve()
        flat = Path(name).name
        if flat in placed and placed[flat] != src:
            raise SystemExit(f"two meshes flatten to {flat!r}: {placed[flat]} and {src}")
        placed[flat] = src
        shutil.copy2(src, out / "meshes" / flat)
        mesh.set("file", flat)
    # Written after the paths are flattened: the model the importer compiles is
    # the one that names them as the package lays them out.
    ET.ElementTree(model).write(out / "model.xml", encoding="utf-8", xml_declaration=True)

    counts, feet = _verify(out / "model.xml")

    manifest = {
        "schema": SCHEMA,
        "robot": {
            "id": "jumper",
            "sourceModel": model.get("model"),
            # The name inside the package, not upstream: the importer opens this.
            "sourceXml": "source.xml",
            "sourceXmlUpstreamName": source_xml.name,
            "coordinateSystem": "Z_UP_X_FORWARD",
            "units": "metres",
            "rootBody": "base_link",
            "standHeight": C.STAND_Z,
            "keyframe": "init_state",
        },
        "exportedBy": {
            "repository": "KingKongRobotics/jumper",
            "commit": _git_commit(),
            "mujoco": mujoco.__version__,
            "tool": "assets/jumper/tools/export_web_robot.py",
        },
        "model": {
            "file": "model.xml",
            "counts": counts,
            "option": _sim_options(),
            # Quoted so an importer compiling with a different MuJoCo can check
            # that it got the same model rather than assume it.
            "lowestFootAtKeyframe_m": feet,
        },
        "joints": [
            {
                "name": j.get("name"),
                "range": [float(v) for v in j.get("range").split()],
                "axis": [float(v) for v in j.get("axis").split()],
                "home": float(C.HOME[j.get("name")]),
            }
            for j in joints
        ],
        # Hardware, not a control scheme: no <actuator> block ships with the
        # model. An importer builds whatever its runner drives joints with.
        "actuator": {
            "law": "pd_position",
            "stiffness": C.STIFFNESS,
            "damping": C.DAMPING,
            "effortLimit": C.EFFORT_LIMIT,
            "armature": C.ARMATURE,
            "controlRateHz": 1.0 / (make_velocity_env_cfg().sim.mujoco.timestep * 4),
            "torqueSpeedCurve": {
                "model": "exp_decay_with_cutoff",
                "plateauTorque": servo.PLATEAU_TORQUE,
                "cornerSpeed": servo.CORNER_SPEED,
                "decaySpeed": servo.DECAY_SPEED,
                "cutoffSpeed": servo.CUTOFF_SPEED,
                "continuousTorque": servo.CONTINUOUS_TORQUE,
                "thermalTimeConstant": servo.THERMAL_TIME_CONSTANT,
                "note": "MJCF cannot express this. A runner that ignores it drives "
                        "joints stronger at speed than the ones a policy trained against.",
            },
        },
        "files": _hash_tree(out),
    }
    (out / "robot-package.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )

    total = sum(f["bytes"] for f in manifest["files"].values())
    print(f"exported {out}")
    print(f"  {model.get('model')} -- {counts['njnt'] - 1} joints, {counts['nbody'] - 1} bodies, "
          f"{counts['nmesh']} meshes, {total / 1e6:.1f} MB")
    print(f"  lowest foot at the keyframe: {feet * 1e6:.1f} um above the floor")


def _hash_tree(out: Path) -> dict[str, dict]:
    files = {}
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name == "robot-package.json":
            continue
        rel = path.relative_to(out).as_posix()
        data = path.read_bytes()
        files[rel] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return files


def _verify(model_path: Path) -> tuple[dict[str, int], float]:
    """Compile the exported model and check what fails silently.

    Two things, both of which produce a plausible-looking robot when wrong: a
    contact stiffer than the timestep can integrate, and a keyframe whose feet
    are not on the ground.
    """
    model = mujoco.MjModel.from_xml_path(str(model_path))

    need = 2 * model.opt.timestep
    bad = sorted({float(v) for v in model.geom_solref[:, 0] if 0 < v < need})
    if bad:
        raise SystemExit(
            f"contact solref {bad} is below 2*timestep={need:g}: the solver would "
            "inject energy and the robot would gain height"
        )

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    lows = []
    for foot in C.FEET:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot)
        best = np.inf
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid or model.geom_dataid[g] < 0:
                continue
            if not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_meshcol"):
                continue
            mid = model.geom_dataid[g]
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            verts = model.mesh_vert[adr: adr + num]
            rot = data.geom_xmat[g].reshape(3, 3)
            best = min(best, float((rot @ verts.T).T[:, 2].min()) + float(data.geom_xpos[g][2]))
        lows.append(best)

    lowest, spread = min(lows), (max(lows) - min(lows)) * 1000
    if not -1e-3 < lowest < 5e-3 or spread > 1.0:
        raise SystemExit(
            f"the keyframe does not stand: lowest foot {lowest * 1000:.3f} mm, "
            f"spread {spread:.3f} mm"
        )

    counts = {k: int(getattr(model, k)) for k in
              ("nq", "nv", "nu", "njnt", "nbody", "ngeom", "nmesh")}
    return counts, lowest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="package directory to write")
    export(parser.parse_args().out.resolve())
