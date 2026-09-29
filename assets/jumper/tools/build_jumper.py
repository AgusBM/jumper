#!/usr/bin/env python3
"""Generate this framework's jumper.xml from a Liuzu URDF export.

The input is the raw SolidWorks-to-URDF package as it arrives from mechanical
design -- a `urdf/`, a `meshes/` and some ROS launch files. Nothing else is
required: the previous generator wanted a whole upstream repository carrying a
pre-made `robot.mjcf` plus two Isaac-tuned URDFs of collision primitives, and
none of those exist for a fresh export.

The output is structured like mjlab's own robots (go1.xml and friends):

- **no `<actuator>` section** -- actuators are built from configuration by
  `ServoCurveActuatorCfg`
- every geom is named, so `CollisionCfg` regexes can select them

Two geom sets coexist in one XML, distinguished by name:

| Name | Contents | Purpose |
|---|---|---|
| `<body>_visual`  | the link's STL | display only, contype=0 |
| `<body>_meshcol` | the link's STL | the collision geometry |

Both point at the **same, full-resolution** mesh. Decimated convex hulls are a
separate step (`tools/hull_collision.py`), deliberately not run from here, and
`assets/jumper/tools/check_hulls.py` draws the result.

## Why `fusestatic="false"` is not optional

MuJoCo's URDF importer welds a fixed-joint child into its parent body. Every
contact body in this robot hangs off a fixed joint -- the four `*_foot_tip_link`
and the two `*F_palm_pad_b_link` -- so the default import produces a model with
**no feet as bodies at all**: 23 bodies instead of 37, their masses folded into
the parent. `constants.FEET` names those bodies to query contact forces and to
place the foot sites, so what follows would be a model that compiles, simulates,
and silently reports zero foot contact. `fusestatic="false"` keeps all 36 links.

## What the export gets wrong

`sanitize()` below repairs the export's defects before the model is compiled and
prints every one of them, with the evidence for each. **These are repairs to
somebody else's file** -- report them upstream and drop the entry here when a
corrected export lands, rather than letting the list grow into a private fork of
the CAD.

That has already paid off twice. The `V1.6-URDF-2` export fixed two of the three
broken inertia tensors and moved the upper shell 14.85 mm back onto the chassis,
and the entries for those are gone from here. The export after it moved all six
legs the same 14.86 mm back: the mesh of every part is byte-identical across all
three exports, so the shell's leg-mount pockets never moved, and sectioning the
chassis at the mount plane shows the housing landing concentric in its pocket
only at the new origins -- the first two exports had the whole leg cluster 14.86
mm too far forward. Neither revision touched a defect repaired here, which is
why the assertions matter: each repair asserts the exact value it expects to
find, so a revision that changes one **stops the build** instead of silently
skipping a fix that is no longer where it was.

## What this build does NOT repair

Two things in v1.6.1 are geometry rather than values, so there is nothing to
patch here: only a different export, or a different assumption on our side, can
settle them. Both are reported to the CAD team with the measurements, so the next
export can be checked against numbers rather than impressions.

**The right half of this robot is the left half turned 180 degrees about z, not
mirrored.** Three independent measurements say so, and all three are exact under
the turn:

    what was measured          turn about z        a reflection in x-z
    mesh vertices, 9 pairs     0.0000 um front     0.25 to 1.42 mm
                               0.05-0.15 um legs
    joint origins below J0     0.0001 mm           1.4 to 4.2 mm, all in x
    inertia tensors, 18 pairs  0.0000 %            4.5 % to 60.6 %

`J0` itself -- where each leg meets the base -- **is** correctly mirrored, to
0.002 mm. It is each leg's internal chain that is a turned copy, which is what
you get by copying a leg sub-assembly and spinning it into place.

The three compose, so they are one fact and not three. Turn-then-unreflect is
`diag(-1,-1,1) . diag(1,-1,1) = diag(-1,1,1)`, a fore-aft flip, and a reflected
left part flipped fore-aft matches its right partner to 0.0000 mm on all ten
pairs that can be compared.

So this robot is **not sagittally symmetric**, and the residue is not noise. At
the zero pose, each left foot against its right partner in the base frame:

    LF_palm_pad_b_link    0.801 mm      LM/LR_foot_tip_link    1.379 mm

all of it in x, with dy 0.002 mm and dz 0.000. The chain arithmetic closes on
those numbers exactly -- summing the per-joint mirror residual down a leg, which
is legitimate because every rpy is zero and the frames are axis-aligned at the
zero pose:

    LM_J0 +0.0000  LM_J1 +2.2001  LM_J2 -2.3999  LM_foot_tip +1.5784
      = +1.3786 mm against 1.379 measured
    LF_J2 +1.8000  LF_J3 -2.4000  LF_palm_pad_b +1.4010
      = +0.8011 mm against 0.801 measured

so the asymmetry is entirely in `<joint><origin xyz>` and nothing else
contributes. It costs 1.68 mrad of yaw bias and is why
`tests/test_posture.py::test_the_mirror_of_a_posture_is_the_posture_of_the_mirror`
is red by 17 times its 1e-4 tolerance. The exported home pose carries it too:
`HOME`'s `LF_J3` and `RF_J3` sum to +0.0433 rad where every other negating pair
sums below 0.005.

**Whether this is a defect at all is an open question, and the answer decides
which side has to change.** The right mesh's vertices are the left's with x and y
negated to *0.0000 um* -- bit for bit the same numbers, not a separately modelled
mirror part that happens to be close. That is what one part number used on both
sides looks like: turn it round, bolt it on, save a mould. It is a normal and
often deliberate choice. So either

* the hardware really is built that way, the CAD is right, and **our** mirror
  prior is the thing that is wrong -- `common/mdp/symmetry.py`'s augmentation and
  `test_posture.py` are both asserting a symmetry the robot does not have; or
* a mirrored instance was wanted and a turned one was placed by mistake.

The mechanical team has been asked which, with the measurements above. Until they
answer, **the tolerance is not widened**: that test's own comment records what
widening it would buy -- an actor trained on mirrored samples that say "lean left"
beside a body leaning right -- and if the answer is "deliberate" the fix is to drop
the prior, not to loosen the check that found it missing.

**`RM_calf_link.stl` is a different revision of the part from the other three.**
The four middle and rear calves are one part in the assembly. LM, LR and RR are
43074 vertices and 14358 faces each; RM is 40158 and 13386, with the same
47.81 x 130.07 x 34.73 mm bounding box. Sampling each surface at 400k points and
measuring vertices against the other's surface -- which is immune to how either is
triangulated, and reads 0.205 mm RMS between LM and LR, the noise floor of the
sampling -- RM sits **1.077 mm RMS and 14.16 mm peak** from all three. The
asymmetry says which way: 38053 sampled points of RR's surface are more than 2 mm
away from RM and only 14727 the other way, so RM is **missing** surface the others
have, distributed along the part rather than in one place. A suppressed feature or
a body left out of that one export.

## Why the legs are re-declared

`order_legs()` is not tidiness. MuJoCo numbers joints in declaration order,
`Entity.find_joints` returns model order, and the policy's action vector is
built from that -- so the URDF's declaration order, and nothing in
`constants.py`, is what fixes the meaning of each action slot and each entry of
`layout.json`'s `action_joint_order`.

This export declares the right leg of each pair first; every previous model
declared the left. Importing it as-is renumbered all six legs, which would take
a checkpoint trained before it and drive each leg with its mirror's action --
silently, since the shapes still match and the gait still looks like a gait.
`tests/test_joint_order.py` is what keeps that from happening again.

## What is carried over by hand

The `motor` and `collision` default classes hold measurements, not geometry:
armature / damping / frictionloss come from the servo bench (see
`assets/jumper/motor/README.md`) and the contact solver parameters were tuned for
this robot. A CAD export knows none of it, so those numbers live here and
survive a model revision unchanged.

Usage:
    # The whole update, after replacing `assets/jumper/urdf/` with a new export.
    python assets/jumper/tools/build_jumper.py

    python tools/hull_collision.py --model assets/jumper/jumper.xml -k 64 --apply \
        --exclude LF_palm_pad_b_link,RF_palm_pad_b_link,LM_foot_tip_link,\
RM_foot_tip_link,LR_foot_tip_link,RR_foot_tip_link

    python assets/jumper/tools/calibrate_home.py   # paste HOME / STAND_Z back

The `--exclude` list is `constants.FEET`: the six bodies that touch the ground
keep their full mesh, because a hull's flat facets change when a foot lands.

`--src` and `--out` default to the vendored package and the model beside it, so
a normal update passes neither. Both remain for a trial conversion of a package
that has not been accepted: point `--out` into /tmp and the repository is
untouched.

**`build_jumper.py` empties `assets/jumper/meshes/` and refills it from the
export**, so the hull step has to follow it every time. Skipping it leaves
`jumper.xml` referencing `*_col.STL` files that are no longer there, and the
model does not load.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

from tasks.jumper.common.assets import CAMERA_CONFIG
from tasks.jumper.common.constants import FEET

# The six legs, in the canonical order, mapped to the body each one stands on.
# The front pair walks on the rear jaw pad; `*F_palm_pad_f_link`, the pad 15.8 mm
# in front of it on the same link, is not a contact body.
FOOT_SITE_OF = {foot.split("_")[0]: foot for foot in FEET}

# Servo bench measurements, not geometry -- see the module docstring.
MOTOR_ARMATURE = 0.0015
MOTOR_DAMPING = 0.017
MOTOR_FRICTIONLOSS = 0.011

# Contact solver parameters tuned for this robot, likewise not from the CAD.
# Written in full because MjSpec holds them as fixed-width arrays; the trailing
# entries are MuJoCo's own defaults, which is why the previous XML showed only
# `solref="0.008"` and `solimp="0.99 0.999 1e-05"`.
COLLISION_PRIORITY = 1
COLLISION_FRICTION = (1.0, 0.01, 0.01)
COLLISION_SOLREF = (0.008, 1.0)
COLLISION_SOLIMP = (0.99, 0.999, 1e-05, 0.5, 2.0)

# The bound written on every joint. It is **not** the servo's real limit: the
# torque ceiling is the measured torque-speed curve applied by
# `ServoCurveActuatorCfg`, and `constants.EFFORT_LIMIT` (1.7464 N*m) caps that.
# This is the looser MJCF-level backstop the previous model carried, kept
# identical so that swapping the robot revision changes no actuator behaviour.
#
# For the record, the URDF states its own effort limits -- 1.5 N*m on the yaw and
# finger joints, 0.7 N*m elsewhere -- and they are ignored here for the same
# reason the plateau replaced the old 2.0: a number with no recorded provenance
# does not outrank a bench measurement.
ACTUATOR_FRC_RANGE = (-2.0, 2.0)

#: The CAD package, kept **verbatim** in the repository.
#:
#: Nine exports in three weeks each arrived somewhere different on disk and each
#: had to be pointed at by hand. Vendoring the package makes an update one
#: operation -- replace this directory, run the three commands in the module
#: docstring -- and it puts `URDF_CONVENTIONS.md`, which is the naming contract
#: mechanical design now works to, under review with the geometry it describes.
#:
#: Nothing reads it at runtime. `jumper.xml` and `meshes/` are the generated
#: model; this is the source they were generated from, kept so the generation
#: can be repeated and audited.
DEFAULT_SRC = Path(__file__).resolve().parents[1] / "urdf" / "jumper"

#: The model this writes, beside the package it was generated from.
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "jumper.xml"

#: The body the forward camera is mounted on -- the housing the export carries, so
#: the eye moves with the hardware and is placed by the CAD.
CAMERA_BODY = "camera_link"

#: The eye offset used to be duplicated here as `CAMERA_EYE_X`. It is not any
#: more: `camera/camera_config.yaml` holds `sim_eye_offset_m` and
#: `tests/test_camera.py` compares the built model against **that** file, so a
#: copy here is a number quietly meaning two things -- which is exactly how it
#: failed when the bottom-shell revision needed 0.005 to become 0.007.


def sanitize(root: ET.Element, mesh_prefix: str) -> list[str]:
    """Repair the export's defects in place, returning one note per repair.

    Every fix here is a claim that the file is wrong, so every one carries its
    evidence. The robot is mirror-symmetric and each part appears two or four
    times, which makes the siblings an independent source of truth: where one
    link disagrees with its three mirror images on a value that has to be shared,
    the odd one out is the typo.
    """
    notes: list[str] = []
    joints = {j.get("name"): j for j in root.findall("joint")}

    # `RF_palm_pad_f_link`'s inertia tensor used to be repaired here: its `iyy`
    # held a neighbouring field's value, and a later export re-rounded it to
    # seven decimals, which collapsed `ixx` and `ixy` onto the same 1e-07 and
    # left it violating the triangle inequality. **The V1.5-bottom-shell export
    # fixed it**, so the entry is gone rather than kept as dead weight: all four
    # tip links now carry eigenvalues equal to eight significant figures, which
    # is the check that matters for four mirror images of one part.
    #
    # It is worth recording what the assertion bought. That repair asserted the
    # exact six values it expected, so the corrected export **stopped the build**
    # rather than silently applying a fix to numbers that had moved underneath
    # it. That is the whole reason every entry below asserts.

    # `base_link`'s centre of mass used to arrive at y = -3.92 mm on a
    # mirror-symmetric chassis and was zeroed here. The v1.6.1 export puts it at
    # -0.009 mm, which is noise around zero, so the entry is gone.

    # ── 1b. Whitespace in a name ──────────────────────────────────────────
    # V1.6-URDF-2 introduced `name="LF_J1_joint "`, with a trailing
    # space. Nothing rejects it and everything downstream misses by one:
    # `mj_name2id` returns -1 for the name the framework uses, numpy reads index
    # -1 as the *last* joint, so writing `HOME["LF_J1_joint"]` lands
    # on `LR_J2_joint` instead -- silently. The actuator regex `.*_joint`
    # stops matching too, leaving that joint unactuated, also silently.
    #
    # Stripped wholesale rather than pinned to the one known name: a name is
    # never meant to carry surrounding whitespace, so there is no case this could
    # hide, and pinning would let the next one through untouched.
    for element in list(root.findall("link")) + list(root.findall("joint")):
        raw = element.get("name")
        if raw != raw.strip():
            element.set("name", raw.strip())
            notes.append(f"name     {element.tag} {raw!r} -> {raw.strip()!r} (stripped whitespace)")
    for element in root.iter():          # parents and children refer by name too
        for key in ("link",):
            val = element.get(key)
            if val is not None and val != val.strip():
                element.set(key, val.strip())
                notes.append(f"name     {element.tag} {key}={val!r} -> {val.strip()!r}")
    joints = {j.get("name"): j for j in root.findall("joint")}

    # Four repairs lived here and are gone, each because the v1.6.1 export
    # fixed what they asserted: `LM_J2_joint`'s axis (`0 -1 0` where every
    # other ankle turns about x), `RM_J2_joint`'s range (the left ankle's,
    # on a right ankle), the capitalised `*_Wrist_joint`, and four mesh paths
    # spelled `package://` while sixty-eight were relative. Each stopped the
    # build on its assertion rather than applying a fix to numbers that had
    # moved, which is what the assertions are for.

    # ── 2. Three joint limits that are not their mirror's ─────────────────
    # The robot is mirror-symmetric and the export declares each pair, so the
    # sibling is an independent source of truth. Three right-hand or left-hand
    # limits disagree with theirs, and two of them put the home pose **outside
    # the joint's own range**, where MuJoCo pins the joint at the limit and the
    # PD pushes into the constraint forever -- the exact failure a gripper
    # homing at -1.0 produced for months, reported by nothing.
    #
    #     joint         exported        mirror says     home pose
    #     LM_J0_joint   ( 0.75, 1.00)   (-0.75, 1.00)   0.0012  -> 0.749 outside
    #     RF_J4_joint   ( 0.10, 1.60)   (-0.10, 1.60)   0.0000  -> 0.100 outside
    #     RF_J2_joint   (-1.65, 2.10)   (-1.65, 2.20)   0.4414  -> inside
    #
    # The first two are a **sign flipped on the lower bound**, and neither is a
    # plausible design: `LM_J0` would be a walking leg's hip yaw confined to a
    # 0.25 rad window between 43 and 57 degrees. The third took the value of
    # `RF_J3_joint`'s upper bound, which is the "a field holds its neighbour's
    # value" fault this export family has produced eleven times.
    for joint_name, wrong, right, why in (
        ("LM_J0_joint", ("0.75", "1"), ("-0.75", "1"), "mirror of RM_J0 (-1, 0.75)"),
        ("RF_J4_joint", ("0.1", "1.6"), ("-0.1", "1.6"), "mirror of LF_J4 (-1.6, 0.1)"),
        ("RF_J2_joint", ("-1.65", "2.1"), ("-1.65", "2.2"), "mirror of LF_J2 (-2.2, 1.65)"),
    ):
        limit = joints[joint_name].find("limit")
        got = (limit.get("lower"), limit.get("upper"))
        assert got == wrong, (joint_name, got)
        limit.set("lower", right[0])
        limit.set("upper", right[1])
        notes.append(f"range    {joint_name}: [{wrong[0]}, {wrong[1]}] -> "
                     f"[{right[0]}, {right[1]}] ({why})")

    # ── 3. Mesh paths, rewritten to reach the vendored package ───────────
    # The export references its own STLs as `../meshes/visual/<link>.stl`. They
    # are **not copied**: `assets/jumper/meshes/` holds only what is generated
    # for mjlab, and the source geometry stays in the one place it arrived, so
    # there is exactly one copy of every STL in the repository and replacing the
    # export replaces it.
    #
    # The path written here is relative to the model file, which is where the
    # compiler starts once `meshdir` is `.`.
    n_path = 0
    for mesh in root.iter("mesh"):
        stem = mesh.get("filename").rsplit("/", 1)[-1]
        mesh.set("filename", f"{mesh_prefix}{stem}")
        n_path += 1
    notes.append(f"path     {n_path} mesh filenames -> {mesh_prefix}<name>")

    # ── 6. The importer would weld away every contact body ────────────────
    compiler = root.find("mujoco/compiler")
    compiler.set("fusestatic", "false")
    notes.append("compiler fusestatic=false, so the six contact links stay bodies")

    return notes


#: The order the six legs must appear in, left before right at each station.
#:
#: **This is a contract, not a preference.** MuJoCo numbers joints in the order
#: the URDF declares them, `Entity.find_joints` returns model order (its
#: `preserve_order` defaults to False), and the action term is built from that --
#: so this tuple, and nothing in `constants.py`, is what fixes the meaning of
#: each slot in the policy's 20-wide action vector and in `layout.json`'s
#: `action_joint_order`.
#:
#: The V1.6 export declares right before left. Importing it as-is renumbered
#: every leg pair, which would load a checkpoint trained before it and drive the
#: right legs with the left legs' actions -- a robot that runs, and is wrong,
#: with nothing raising. The order below is the one every existing policy and
#: every deployed `layout.json` was built against.
LEG_ORDER = ("LF", "RF", "LM", "RM", "LR", "RR")


def order_legs(root: ET.Element) -> str:
    """Re-declare the six leg subtrees in `LEG_ORDER`, in place.

    Only the leg joints move, and only among the slots they already occupy, so
    the chassis links keep their positions. Sibling order is pure relabelling:
    the six legs are independent children of `base_link`, so which one is
    numbered first changes no kinematics and no dynamics.
    """
    def leg_of(joint: ET.Element) -> str | None:
        prefix = joint.find("child").get("link").split("_")[0]
        return prefix if prefix in LEG_ORDER else None

    children = list(root)
    slots = [i for i, el in enumerate(children)
             if el.tag == "joint" and leg_of(el) is not None]
    legs = [children[i] for i in slots]
    was = [leg_of(j) for j in legs]
    ordered = [j for _, j in sorted(
        enumerate(legs), key=lambda p: (LEG_ORDER.index(leg_of(p[1])), p[0])
    )]
    for slot, joint in zip(slots, ordered):
        children[slot] = joint
    root[:] = children

    now = [leg_of(j) for j in ordered]
    seen = [leg for i, leg in enumerate(now) if i == 0 or now[i - 1] != leg]
    assert seen == list(LEG_ORDER), seen
    # Every leg must keep the same joints, in the same within-leg order.
    for leg in LEG_ORDER:
        before = [j.get("name") for j, lg in zip(legs, was) if lg == leg]
        after = [j.get("name") for j, lg in zip(ordered, now) if lg == leg]
        assert before == after, (leg, before, after)
    first = [leg for i, leg in enumerate(was) if i == 0 or was[i - 1] != leg]
    return f"order    legs {' '.join(first)} -> {' '.join(LEG_ORDER)}"


def mesh_aabb_centre_in_body(model: mujoco.MjModel, geom_name: str) -> np.ndarray:
    """A mesh geom's AABB centre **in its owning body's frame**.

    WARNING: do not take the AABB of `model.mesh_vert` directly. Those vertices
    are in the **mesh's own frame** -- MuJoCo reorients meshes onto their
    principal axes and puts a compensating rotation in the geom's `geom_quat`.
    Using the mesh-frame AABB as if it were the body frame gives a box with
    completely wrong orientation; the previous generator recorded measuring a
    base_link 0.205 m tall where the body is 0.071 m, which sank the robot to its
    belly while nothing raised an error.
    """
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise KeyError(f"the model has no geom {geom_name!r}")
    mid = model.geom_dataid[gid]
    adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    verts = model.mesh_vert[adr : adr + num]

    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
    verts_b = verts @ rot.reshape(3, 3).T + model.geom_pos[gid]
    return (verts_b.min(axis=0) + verts_b.max(axis=0)) / 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="root of the URDF package (the directory holding urdf/ and meshes/)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output path for jumper.xml")
    args = ap.parse_args()

    src, out = args.src.expanduser(), args.out
    urdf_files = sorted((src / "urdf").glob("*.urdf"))
    if len(urdf_files) != 1:
        raise SystemExit(f"expected exactly one .urdf in {src / 'urdf'}, found {len(urdf_files)}")
    urdf_path = urdf_files[0]

    # ── Meshes ────────────────────────────────────────────────────────────
    # Nothing is copied. The model points into the vendored export for its
    # visual geometry and into `meshes/` for the hulls `hull_collision.py`
    # writes, so `meshes/` holds only generated content and the export stays
    # byte-identical to what mechanical design shipped.
    out.parent.mkdir(parents=True, exist_ok=True)
    visual_dir = next((d for d in (src / "meshes" / "visual", src / "meshes")
                       if d.is_dir() and any(d.glob("*.[sS][tT][lL]"))), None)
    if visual_dir is None:
        raise SystemExit(f"no STL files under {src / 'meshes'}")
    mesh_prefix = visual_dir.resolve().relative_to(out.parent.resolve()).as_posix() + "/"
    n_mesh_files = len(list(visual_dir.glob("*.[sS][tT][lL]")))
    print(f"meshes: {n_mesh_files} STL files referenced in place at {mesh_prefix}")

    # ── Sanitise ──────────────────────────────────────────────────────────
    # The source package is read-only as far as this script is concerned: the
    # repaired copy is written beside the output and deleted again.
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    revision = urdf_path.stem
    print(f"source: {urdf_path}  (robot name {root.get('name')!r})")
    for note in sanitize(root, mesh_prefix=mesh_prefix):
        print(f"  fixed  {note}")
    print(f"  fixed  {order_legs(root)}")

    tmp_urdf = out.parent / "_build_tmp.urdf"
    tree.write(tmp_urdf, encoding="utf-8", xml_declaration=True)
    try:
        spec = mujoco.MjSpec.from_file(str(tmp_urdf))
    finally:
        tmp_urdf.unlink(missing_ok=True)

    spec.modelname = revision
    # `.` rather than `meshes`, because the model now references two directories:
    # the vendored export for the visual geometry and `meshes/` for the hulls.
    #
    # The mesh paths are left as the URDF gave them. They used to be reduced to
    # a bare filename here, which was right while every STL was copied into one
    # flat `meshes/`; it is wrong now and fails loudly -- the compiler looks for
    # `LF_upper_arm_link.stl` beside the model and it is not there.
    spec.meshdir = "."

    spec.option.timestep = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    # Camera framing and haze only. No <headlight>: mjlab's scene/scene.xml is
    # the parent spec and sets the same values, so a headlight here is overridden
    # and deleting it changes the compiled model not at all (measured). Ambient
    # light is a property of the scene, not of the robot -- see
    # scenes/registry.py's Headlight.
    spec.visual.global_.azimuth = 120
    spec.visual.global_.elevation = -20
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]

    # ── Default classes ───────────────────────────────────────────────────
    # A class is only half of it. `to_xml` writes out every attribute whose value
    # on the element differs from the class it names, and an element built by the
    # URDF importer carries MuJoCo's *global* defaults -- so naming a class and
    # stopping there emits `contype="1" conaffinity="1" group="0"` onto each
    # visual geom, overriding the class that was supposed to hide it from
    # collision. Both are set below: the class for readability, the element so the
    # writer sees agreement and omits it.
    robot = spec.add_default("robot", spec.default)
    motor = spec.add_default("motor", robot)
    motor.joint.armature = MOTOR_ARMATURE
    motor.joint.damping = [MOTOR_DAMPING, 0.0, 0.0]
    motor.joint.frictionloss = MOTOR_FRICTIONLOSS
    visual = spec.add_default("visual", robot)
    visual.geom.contype = 0
    visual.geom.conaffinity = 0
    visual.geom.group = 2
    collision = spec.add_default("collision", robot)
    collision.geom.group = 1
    collision.geom.priority = COLLISION_PRIORITY
    collision.geom.friction = list(COLLISION_FRICTION)
    collision.geom.solref = list(COLLISION_SOLREF)
    collision.geom.solimp = list(COLLISION_SOLIMP)

    # ── Bodies, joints, geoms ─────────────────────────────────────────────
    base = next(b for b in spec.bodies if b.name == "base_link")
    base.classname = robot
    base.add_freejoint(name="floating_base")

    n_vis = n_col = n_joint = 0
    for body in spec.bodies:
        if body.name in ("world", ""):
            continue
        for joint in body.joints:
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            joint.classname = motor
            joint.armature = MOTOR_ARMATURE
            joint.damping = [MOTOR_DAMPING, 0.0, 0.0]
            joint.frictionloss = MOTOR_FRICTIONLOSS
            joint.actfrcrange = list(ACTUATOR_FRC_RANGE)  # MJCF: actuatorfrcrange
            n_joint += 1
        # The importer gives each link two geoms: the <visual> element becomes
        # group 1 with contype 0, the <collision> element group 0 with contype 1.
        # Discriminate on contype -- group is about to be overwritten by the
        # class, and at spec level classname is not yet resolved to anything.
        for geom in body.geoms:
            if geom.contype == 0:
                geom.name, geom.classname = f"{body.name}_visual", visual
                geom.conaffinity, geom.group = 0, 2
                n_vis += 1
            else:
                geom.name, geom.classname = f"{body.name}_meshcol", collision
                geom.group = 1
                geom.priority = COLLISION_PRIORITY
                geom.friction = list(COLLISION_FRICTION)
                geom.solref = list(COLLISION_SOLREF)
                geom.solimp = list(COLLISION_SOLIMP)
                # **Keep the URDF's material colour on this geom too**, rather
                # than letting it fall back to MuJoCo's grey. The collision set is
                # not always hidden: the native backend strips visual meshes at
                # large environment counts and the viewer then draws precisely
                # this geometry. Grey there would repaint the whole robot for no
                # reason the picture explains -- and the model has already shown
                # how that goes wrong, when three links left out of the collision
                # set stopped the viewer hiding the group at all and the robot
                # rendered pale pink instead of red.
                n_col += 1
    print(f"named: {n_vis} visual / {n_col} collision geoms, {n_joint} motor joints")

    # ── Sites ─────────────────────────────────────────────────────────────
    # mjlab's velocity tasks reference robot/imu_lin_vel, robot/imu_ang_vel and
    # robot/root_angmom directly in their observations, and those must be real
    # <sensor> entries in the MJCF (go1.xml is the same). A URDF has no sensors,
    # so they are added here following go1.xml's shape.
    base.add_site(name="imu", pos=[0.0, 0.0, 0.0], size=[0.006, 0.006, 0.006], group=4)

    # Go1's foot_clearance / foot_slip rewards and the foot_height_scan sensor
    # locate feet by site. Each one goes at the AABB centre of that foot's
    # collision mesh, in the foot body's frame -- pose-independent, and within a
    # couple of millimetres of the contact point for tips this small.
    model = spec.compile()
    n_site = 0
    for leg, foot in FOOT_SITE_OF.items():
        centre = mesh_aabb_centre_in_body(model, f"{foot}_meshcol")
        body = next(b for b in spec.bodies if b.name == foot)
        body.add_site(name=leg, pos=centre.tolist(), size=[0.005, 0.005, 0.005], group=4)
        n_site += 1
        print(f"  site {leg:3s} on {foot:22s} at {np.round(centre, 5).tolist()}")
    print(f"sites: {n_site} foot sites plus imu")

    # ── The forward camera ────────────────────────────────────────────────
    # On `camera_link`, the housing the export carries, so the eye moves with
    # the hardware it belongs to and its placement comes from the CAD rather than
    # from a number here. Optics from `assets/jumper/camera/camera_config.yaml`.
    #
    # A camera costs a training run nothing -- a rigid massless frame offset, no
    # mass, no inertia, no degree of freedom, no geometry -- so nq/nv/nu and the
    # body, geom and site counts are what they were without it. Nothing renders it
    # either, until a task declares a camera sensor.
    #
    # **`CAMERA_EYE_X` is why this is not simply at the housing's origin.** At the
    # origin the eye sits on its own housing's front face, so every ray hits it at
    # zero distance and the picture is the inside of the camera -- measured, 221
    # of 221 rays. Same cause as the pre-V1.6 mount: a closed mesh with no
    # aperture, in a different place.
    #
    # The value is not a clearance, it is **the offset that puts the eye level
    # with the forward-most geometry the robot carries**, so nothing is in front
    # of it at all. That is a property rather than a margin, and it has to be
    # re-measured on a new shell. In `base_link` coordinates, the bottom-shell
    # revision:
    #
    #     base_link_visual      93.816 mm   <- forward-most
    #     upper_shell_link_visual   93.814
    #     base_link_meshcol     93.734      <- the k=64 hull, 0.08 mm inside
    #     tof_sensor_link              93.148
    #     camera_link      91.708      <- the camera housing itself
    #     camera mount origin   87.508
    #
    # so level is 6.308 mm and this rounds to 7.0, putting the eye 0.19 mm ahead
    # of everything. The previous shell needed 5.0 mm for the same rule; its nose
    # sat 1.3 mm closer to the camera mount.
    #
    # **The hull is what made this bite.** `base_link`'s raw mesh has an aperture
    # for the lens and never blocked the view; its convex hull bridges that
    # aperture and swallowed the eye whole -- 221 of 221 rays at 0.07 mm. A hull
    # filling a hole it should not is exactly the cost `check_hulls.py` draws, and
    # `tests/test_camera.py` is what caught it.
    camera = yaml.safe_load(CAMERA_CONFIG.read_text(encoding="utf-8"))
    housing = next(b for b in spec.bodies if b.name == CAMERA_BODY)
    housing.add_camera(
        name="onboard",
        pos=[camera["camera_rig"]["sim_eye_offset_m"], 0.0, 0.0],
        # `camera_link` is unrotated relative to `base_link` (quat 1 0 0 0),
        # so the config's quaternion applies unchanged. MuJoCo cameras look down
        # local -Z with local +Y up; the file states the mapping it intends,
        # because getting it wrong yields a camera facing the robot's own back
        # with the world upside down, and renders perfectly while doing so.
        quat=camera["camera_rig"]["orientation"]["quat_wxyz"],
        # Vertical field of view. `dfov_deg` is the drawing's diagonal and
        # `hfov_deg` the horizontal derived from it, and either would give a much
        # wider camera than the module has with nothing to say so.
        fovy=camera["rgb"]["vfov_deg"],
        # Pins the aspect ratio, and with it the horizontal field. Without it the
        # compiled camera reports 1x1 and an offscreen render gets whatever shape
        # its buffer happens to be.
        resolution=[camera["rgb"]["width"], camera["rgb"]["height"]],
    )
    print(f"camera: onboard on {CAMERA_BODY} at "
          f"x+{camera['camera_rig']['sim_eye_offset_m']}, "
          f"fovy {camera['rgb']['vfov_deg']}")

    # ── The dToF ──────────────────────────────────────────────────────────
    # Mounted by the model rather than by a task because it is part of the robot:
    # `tasks/jumper/common/tof.py` builds the sensor from this camera alone, so a
    # task declares it without placing it, and every task gets the same one. Like
    # `onboard` it costs a run nothing until a sensor reads it.
    #
    # **Two fields of view, not one.** `fovy` alone gives square zones, and 54 x 42
    # at 42 degrees vertical is 52.5 degrees across rather than the datasheet's 55
    # -- a narrower sensor with nothing to say so. `sensorsize` with `focal` gives
    # each axis its own angle. The image plane is placed at unit focal length, so
    # it is 2 tan(fov/2) wide on each axis and the two numbers are the two angles
    # and nothing else: the unit is a convention, not a measured die.
    #
    # The eye offset follows the same rule as `onboard` and was measured the same
    # way; the numbers are beside `tof.sim_eye_offset_m` in the config.
    tof = camera["tof"]
    tof_body = next(b for b in spec.bodies if b.name == tof["parent_link"])
    # The shared quaternion maps camera axes onto *this body's* axes, so it is only
    # right while the body is unrotated relative to `base_link` -- and wrong
    # without anything failing if an export ever tilts it.
    assert list(tof_body.quat) == [1.0, 0.0, 0.0, 0.0], (
        f"{tof['parent_link']} is rotated ({list(tof_body.quat)}); the camera quaternion "
        f"assumes it is not"
    )
    tof_body.add_camera(
        name="tof",
        pos=[tof["sim_eye_offset_m"], 0.0, 0.0],
        # `tof_to_rgb` is the identity, so the ToF looks where `onboard` does.
        quat=camera["camera_rig"]["orientation"]["quat_wxyz"],
        resolution=[tof["width"], tof["height"]],
        sensor_size=[2 * math.tan(math.radians(tof["hfov_deg"]) / 2),
                     2 * math.tan(math.radians(tof["vfov_deg"]) / 2)],
        focal_length=[1.0, 1.0],
    )
    print(f"camera: tof on {tof['parent_link']} at x+{tof['sim_eye_offset_m']}, "
          f"{tof['width']}x{tof['height']} zones, "
          f"{tof['hfov_deg']} x {tof['vfov_deg']} deg")

    spec.add_sensor(name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="imu_lin_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="imu_lin_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="imu_upvector", type=mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
                    objtype=mujoco.mjtObj.mjOBJ_BODY, objname="world",
                    reftype=mujoco.mjtObj.mjOBJ_SITE, refname="imu")
    spec.add_sensor(name="root_angmom", type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
                    objtype=mujoco.mjtObj.mjOBJ_BODY, objname="base_link")
    print("sensors: 5, following go1.xml")

    model = spec.compile()  # final compile, to confirm the output is valid
    print(f"compiled: nbody={model.nbody} nq={model.nq} nv={model.nv} "
          f"ngeom={model.ngeom} nmesh={model.nmesh} "
          f"mass={model.body_mass.sum():.4f} kg")

    out.write_text(spec.to_xml(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
