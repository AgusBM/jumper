#!/usr/bin/env python3
"""Measure -- or re-derive -- the carried claw's hold pose, and the five-foot stance.

    python tasks/jumper/five_foot/tools/grasp_pose.py --report      # the pose in use
    python tasks/jumper/five_foot/tools/grasp_pose.py --aperture    # mouth opening vs finger angle
    python tasks/jumper/five_foot/tools/grasp_pose.py --check-box   # every LF_GRASP_BOX corner is legal
    python tasks/jumper/five_foot/tools/grasp_pose.py --stance      # five-foot support polygon + payload
    python tasks/jumper/five_foot/tools/grasp_pose.py --search      # re-derive the pose for --target
    python tasks/jumper/five_foot/tools/grasp_pose.py --refine --start <4 angles>   # descend from a pose

The counterpart of `check_stance.py`: that one measures where the six feet touch,
this one measures what the left-front arm is doing when it is **not** one of them.

Plain MuJoCo and numpy against `assets/jumper/jumper.xml` -- no mjlab, no torch, no
simulation. Every number it prints is static: forward kinematics on the visual
meshes, which are the real part geometry. That is deliberate and it is also this
script's one caveat, so it is worth stating plainly:

**These are the part's dimensions, not the solver's.** `constants.py` gives each
link a collision shape built from convex pieces, and a convex piece is never
thinner than the part it covers, so the mouth the solver leaves is a few
millimetres narrower than the one measured here at every aperture. Both jaws are
convex-**decomposed** for that reason (`tasks/jumper/five_foot/tools/jaw_decompose.py`); with one hull
each the mouth is filled in completely and the claw cannot pinch anything at all.
`tests/test_five_foot.py::test_the_claw_has_a_mouth_the_solver_can_see` is what
holds the two within a few millimetres of each other.

The pose it reports is the one in `tasks/jumper/five_foot/claw.py::LF_GRASP`, and
`--search` prints a replacement in paste-ready form. The two are kept in step by
hand, the same arrangement `check_stance.py` has with `HOME`.
"""

from __future__ import annotations

import argparse
import itertools

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from tasks.jumper.common.constants import HOME, STAND_Z
from tasks.jumper.five_foot.claw import (
    ARM_JOINTS,
    CARRIED_LEG,
    FINGER_JOINT,
    FIVE_FOOT_LEGS,
    GRASP_BOX,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    KNUCKLE_CLEAR,
    LEG_FOOT,
    LF_GRASP,
    PAYLOAD_BODY,
    add_payload_body,
)

DEFAULT_MODEL = "assets/jumper/jumper.xml"

#: The two jaws. Both `LF_palm_link` and `LF_palm_pad_b_link` are fixed relative to the
#: forearm -- only `LF_J4_joint` moves anything -- which is what makes
#: `--aperture` a function of the finger alone.
#:
#: **The palm is the fixed jaw**, and leaving it out of this tuple is what made
#: the aperture table in `claw.py` say the claw shuts at finger -0.30. It does
#: not: the palm is the hooked upper jaw, with the serrated face an object is
#: actually pinched against, and `LF_palm_pad_b_link` is the small pad on the *outside* of
#: its tip -- the foot the robot walks on in six-foot mode. Measured against
#: `LF_palm_pad_b_link` alone, the number that comes out is the distance from the finger
#: to a pad on the far side of the jaw it is closing on.
MOVING_JAW = ("LF_finger_link", "LF_finger_tip_link")
FIXED_JAW = ("LF_palm_link", "LF_palm_pad_f_link", "LF_palm_pad_b_link")

#: `KNUCKLE_CLEAR` comes from `claw.py`, which is where the numbers this script
#: measures and the task reads are kept in one place. The moving jaw and the palm
#: **touch at the knuckle** -- two halves of one hinge -- so without it the
#: closest pair of the two jaws is 0.0 mm and the anvil lands on the hinge.

#: The arm, shoulder outwards.
ARM_LINKS = (
    "LF_shoulder_link", "LF_upper_arm_link", "LF_forearm_link",
    "LF_palm_link", *MOVING_JAW, *FIXED_JAW,
)

#: The part of the arm whose clearance the *pose* decides, and the part it does
#: not.
#:
#: The two shoulder links are bolted a few millimetres off the trunk by
#: construction, so their distance to it is a property of the machine rather than
#: of the hold. Measured on V1.6 against the full meshes, from the shoulder pitch
#: link outwards: **5.88 mm at the six-foot HOME stance and 5.12 mm at LF_GRASP**,
#: both the same pair, `LF_upper_arm_link` against `base_link`.
#:
#: `REACH_LINKS` is the forearm outwards, which is what the search can actually
#: move and what an obstacle would be hit with: 24.00 mm at LF_GRASP against
#: 26.52 mm at HOME, `LF_forearm_link` against `base_link` in both.
#:
#: kk-rl-lab reports 16 mm for its pose, and a figure like that is a **sampling
#: artefact**: its search subsampled each mesh to 120 points. On V1.6 this script
#: reads 8.60 mm from the shoulder outwards at that setting, 10.30 mm at 800
#: points and 5.12 mm at the full mesh -- a coarse cloud measures the distance
#: between two sets of points rather than between two surfaces, errs
#: optimistically, and not even monotonically. Hence `npts` defaults to more
#: vertices than the largest mesh in the model (39220), i.e. no subsampling at all.
REACH_LINKS = ARM_LINKS[2:]

#: What the claw could run into: the trunk and the two neighbouring legs.
OBSTACLES = (
    "base_link",
    "RF_shoulder_link", "RF_upper_arm_link", "RF_forearm_link",
    "RF_palm_link", "RF_finger_link", "RF_finger_tip_link", "RF_palm_pad_f_link", "RF_palm_pad_b_link",
    "LM_hip_link", "LM_thigh_link", "LM_calf_link", "LM_foot_tip_link",
)

#: Where the mouth should end up, in the base frame. Only `--search` and
#: `--refine` use it, and reaching it is their heaviest term at weight 60.
#:
#: **`--refine` on an existing pose should be given `--target` explicitly** -- the
#: pose's own mouth point, which `--report` prints -- so that term is near zero at
#: the start and the descent can only improve the rest. This default is for
#: `--search`, which has no pose to start from.
#:
#: It is the point the pre-V1.6 `LF_GRASP` held its mouth at, and it is only
#: roughly where the claw wants to be now: `objects.py::ROW_Y` was laid out around
#: it and has since moved twice to follow the pose. Refining the V1.6 pose against
#: this default on V1.6.1 pulled the mouth 18 mm toward it and paid for it in the
#: closing axis, the floor clearance and the row alignment -- see the hold-pose
#: note in `claw.py`, which records both runs.
DEFAULT_TARGET = (0.282, 0.112, -0.045)

#: Isaac's and mjlab's shared soft-limit factor (`soft_joint_pos_limit_factor`).
#: A hold outside the soft limits fights `dof_pos_limits` and, on hardware, drives
#: a joint into its mechanical stop.
SOFT_LIMIT_FACTOR = 0.9

#: The search must leave room for the whole randomisation box, not just the
#: nominal pose. kk-rl-lab's first attempt used a margin *smaller* than the box
#: and produced a pose whose corners reached past the soft limit.
LIMIT_MARGIN = GRASP_BOX + 0.05

#: How fast the shoulder pitch has to raise the anvil, in metres per radian.
#:
#: **A hold the arm cannot lift from is not grasp-ready**, and nothing else in
#: the cost says so. Searched without it on V1.6, the pose came out with the
#: upper arm hanging straight down and the elbow nearly straight: the jaws closed
#: exactly across the mouth, and the shoulder raised the anvil 34 mm per radian
#: -- 10 mm for the 0.30 rad `tasks/jumper/five_foot/tools/grasp_objects.py` lifts by -- while the
#: elbow and wrist only swung it sideways. That check then failed on all three
#: objects. 80 mm/rad is 24 mm over the check's lift, more than the 20 mm rim of
#: `tasks/jumper/five_foot/objects.py`'s bin; the previous pose managed 86.
LIFT_MIN = 0.080


class Claw:
    """Forward kinematics and geometry queries on the left-front claw."""

    def __init__(self, model_path: str = DEFAULT_MODEL, npts: int = 100_000) -> None:
        # Through `add_payload_body`, because `--stance` reads where a carried
        # load sits off that body and the bare XML has no such thing. It holds no
        # mass until `mdp/events.py::claw_payload` writes one, so everything else
        # measured here is the same robot the XML describes.
        spec = mujoco.MjSpec.from_file(model_path)
        add_payload_body(spec)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self._verts = {
            b: self._mesh(b, npts) for b in (*ARM_LINKS, *OBSTACLES)
        }
        self.soft_limits = {j: self._soft_limit(j) for j in ARM_JOINTS}
        self.set_pose([LF_GRASP[j] for j in ARM_JOINTS], GRIPPER_OPEN)
        # The obstacles do not move with the arm, so their cloud is built once --
        # and as a KD-tree, because the pairwise form is what limits how much of
        # the mesh can be measured. Sampled at 120 points per body it reads the
        # shoulder's clearance 8.6 mm; at 800 it is 10.3 mm and at the full mesh
        # 5.1 mm, so a coarse sample **overstates the clearance** (see `REACH_LINKS`).
        self._obstacle_cloud = cKDTree(np.vstack([self._world(b) for b in OBSTACLES]))

    # ── model helpers ────────────────────────────────────────────────────
    def _id(self, objtype, name: str) -> int:
        i = mujoco.mj_name2id(self.m, objtype, name)
        if i < 0:
            raise KeyError(f"{name!r} is not in the model")
        return i

    def _qadr(self, joint: str) -> int:
        return int(self.m.jnt_qposadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, joint)])

    def _mesh(self, name: str, npts: int) -> np.ndarray:
        """A subsample of a body's visual mesh, in body coordinates.

        The visual mesh is named after the body it belongs to (build_jumper.py's
        convention); the collision copies carry a `_col` suffix and are decimated
        hulls, which is not what should be measured here.

        **`mesh_vert` is not in body coordinates and this used to assume it was.**
        The compiler re-expresses every mesh in its own inertial frame and puts
        the compensating transform on each geom that uses it, so a vertex reaches
        the body through `geom_pos` and `geom_quat` first. Skipping that step is
        silent -- the vertices are still a rigid copy of the part, so every
        distance between two points of the *same* mesh survives and only
        distances *between* parts come out wrong. Measured on the model before
        V1.6, at its LF_GRASP, where each vertex landed relative to the part it
        belongs to (link names as V1.6 calls them):

            LF_palm_pad_b_link           0.5 .. 32.0 mm away
            LF_finger_link      0.4 .. 58.1 mm
            LF_palm_link        0.1 .. 83.0 mm

        So the aperture this script printed was the gap between two point clouds
        that were not the jaws, and the pose search aimed a mouth that was not
        the mouth.
        """
        gid = self._id(mujoco.mjtObj.mjOBJ_GEOM, f"{name}_visual")
        mid = int(self.m.geom_dataid[gid])
        v0, nv = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
        verts = self.m.mesh_vert[v0 : v0 + nv]
        take = np.linspace(0, len(verts) - 1, min(npts, len(verts))).astype(int)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.m.geom_quat[gid])
        return verts[take] @ rot.reshape(3, 3).T + self.m.geom_pos[gid]

    def _soft_limit(self, joint: str, factor: float = SOFT_LIMIT_FACTOR):
        jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, joint)
        lo, hi = (float(v) for v in self.m.jnt_range[jid])
        centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        return centre - factor * half, centre + factor * half

    def _world(self, body: str) -> np.ndarray:
        i = self._id(mujoco.mjtObj.mjOBJ_BODY, body)
        return self._verts[body] @ self.d.xmat[i].reshape(3, 3).T + self.d.xpos[i]

    def site_xy(self, leg: str) -> np.ndarray:
        return self.d.site_xpos[self._id(mujoco.mjtObj.mjOBJ_SITE, leg)][:2].copy()

    def com(self) -> tuple[np.ndarray, float]:
        """(centre of mass, total mass) of the robot, world frame."""
        mass = self.m.body_mass[1:]  # body 0 is the world
        return (mass[:, None] * self.d.xipos[1:]).sum(0) / mass.sum(), float(mass.sum())

    # ── posing ───────────────────────────────────────────────────────────
    def set_pose(self, arm, finger: float, carried: bool = True) -> None:
        """Stand the robot at STAND_Z with the other legs at HOME.

        `carried=False` puts the left-front arm back at HOME too, which is the
        six-foot stance the family's other four tasks use.
        """
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self._qadr("floating_base") + 2] = STAND_Z
        for joint, value in HOME.items():
            self.d.qpos[self._qadr(joint)] = value
        if carried:
            for joint, value in zip(ARM_JOINTS, arm, strict=True):
                self.d.qpos[self._qadr(joint)] = value
            self.d.qpos[self._qadr(FINGER_JOINT)] = finger
        mujoco.mj_forward(self.m, self.d)

    # ── measurements ─────────────────────────────────────────────────────
    def _jaws(self):
        """Both jaws' surfaces in world coordinates, knuckle excluded."""
        hinge = self.d.xpos[self._id(mujoco.mjtObj.mjOBJ_BODY, "LF_finger_link")]

        def jaw(bodies):
            pts = np.vstack([self._world(b) for b in bodies])
            return pts[np.linalg.norm(pts - hinge, axis=1) > KNUCKLE_CLEAR]

        return jaw(MOVING_JAW), jaw(FIXED_JAW)

    def anvil(self):
        """The point on the fixed jaw the moving one arrives at, in the palm frame.

        **Where the claw holds things**, and therefore what the aperture is
        measured from -- see `mouth()`. A constant of the claw: both jaws hang
        off the palm and only the finger moves between them, so it does not
        depend on the arm's pose and is solved once, at the shut pose.

        Side-effect free: it poses the model to find the point and puts the pose
        back, because every caller is in the middle of measuring something else.
        """
        if getattr(self, "_anvil", None) is None:
            saved = self.d.qpos.copy()
            self.set_pose([LF_GRASP[j] for j in ARM_JOINTS], GRIPPER_CLOSED)
            moving, fixed = self._jaws()
            gap, nearest = cKDTree(fixed).query(moving)
            point = fixed[nearest[int(gap.argmin())]]
            palm = self._id(mujoco.mjtObj.mjOBJ_BODY, "LF_palm_link")
            self._anvil = self.d.xmat[palm].reshape(3, 3).T @ (point - self.d.xpos[palm])
            self.d.qpos[:] = saved
            mujoco.mj_forward(self.m, self.d)
        return self._anvil

    def mouth(self):
        """(aperture, centre, closing axis), measured from the anvil.

        **The aperture is how far the moving jaw still has to travel to reach the
        fixed jaw's grip face**, which is how thick an object seated against that
        face can be and still be pinched. That is the number the objects in
        `tasks/jumper/five_foot/objects.py` are sized against, and the one `claw.py::APERTURE_MM`
        records.

        It is not the closest pair of the two jaws, which is what this used to
        report. The mouth is a wedge and its narrowest place is the throat by the
        knuckle, where nothing is ever held; measuring there called a claw that
        opens to 79 mm a claw that opens to 47, and made the curve turn over in
        the middle of the travel because which throat point won kept changing.
        """
        palm = self._id(mujoco.mjtObj.mjOBJ_BODY, "LF_palm_link")
        centre = self.d.xpos[palm] + self.d.xmat[palm].reshape(3, 3) @ self.anvil()
        moving, _ = self._jaws()
        gap, nearest = cKDTree(moving).query(centre)
        axis = moving[nearest] - centre
        return (
            float(gap),
            centre,
            axis / max(float(np.linalg.norm(axis)), 1e-9),
        )

    def measure(self, arm, finger: float = GRIPPER_OPEN) -> dict:
        self.set_pose(arm, finger)
        gap, centre, axis = self.mouth()
        shoulder = np.vstack([self._world(b) for b in ARM_LINKS[1:]])
        reach = np.vstack([self._world(b) for b in REACH_LINKS])
        whole = np.vstack([self._world(b) for b in ARM_LINKS])
        ahead = np.array([centre + np.array([t, 0.0, 0.0]) for t in (0.02, 0.04, 0.06)])
        out = {
            "gap": gap,
            "centre": centre,
            "axis": axis,
            # Two clearances, because they answer different questions: see
            # `REACH_LINKS`.
            "d_shoulder": float(self._obstacle_cloud.query(shoulder)[0].min()),
            "d_other": float(self._obstacle_cloud.query(reach)[0].min()),
            "z_min": float(whole[:, 2].min()),
            "corridor": float(
                np.linalg.norm(reach[:, None, :] - ahead[None, :, :], axis=-1).min()
            ),
        }
        out["lift"] = self.lift(arm, finger)
        return out

    def lift(self, arm, finger: float = GRIPPER_OPEN, delta: float = 0.05) -> float:
        """How fast the shoulder pitch raises the anvil, in metres per radian.

        A central difference over +-`delta`, by forward kinematics alone: the
        anvil is a constant of the palm frame, so nothing needs re-measuring. The
        sign is dropped because the check that lifts picks whichever way is up.
        Leaves the model posed at `arm`.
        """
        # Assembled from CARRIED_LEG rather than written out, which is why the V1.6
        # rename did not reach it: `_shoulder_pitch_joint` is `_J1_joint` now, and
        # a name built at runtime is invisible to a search for the old one. This
        # one at least fell over loudly -- `tuple.index` raises. The mirror sign
        # table did not: see `tasks/jumper/common/mdp/symmetry.py`.
        pitch = ARM_JOINTS.index(f"{CARRIED_LEG}_J1_joint")
        palm = self._id(mujoco.mjtObj.mjOBJ_BODY, "LF_palm_link")
        anvil = self.anvil()
        z = []
        for sign in (+1.0, -1.0):
            moved = list(arm)
            moved[pitch] += sign * delta
            self.set_pose(moved, finger)
            z.append(float((self.d.xpos[palm] + self.d.xmat[palm].reshape(3, 3) @ anvil)[2]))
        self.set_pose(arm, finger)
        return abs(z[0] - z[1]) / (2.0 * delta)


def support_margin(points_xy, p) -> float:
    """Signed distance from `p` to the support polygon of `points_xy`, in metres.

    Positive inside, negative outside; for fewer than three points -- a line or a
    single foot, which no static stance survives -- it is the negative distance to
    that segment, so the sign still reads "not supported".
    """
    pts = np.asarray(points_xy, dtype=float)
    if len(pts) < 3:
        a, b = pts[0], pts[-1]
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
        return -float(np.linalg.norm(p - (a + t * ab)))
    centre = pts.mean(0)
    order = np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]))
    hull = pts[order]
    best, inside = np.inf, True
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        edge = b - a
        normal = np.array([edge[1], -edge[0]])
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        d = float(np.dot(p - a, normal))
        best = min(best, abs(d))
        if d > 0.0:
            inside = False
    return best if inside else -best


# ── the search ────────────────────────────────────────────────────────────


def cost(claw: Claw, arm, target: np.ndarray):
    """What "grasp-ready" means, as a number. 1e6 outside the usable range."""
    for joint, value in zip(ARM_JOINTS, arm, strict=True):
        lo, hi = claw.soft_limits[joint]
        if not (lo + LIMIT_MARGIN <= value <= hi - LIMIT_MARGIN):
            return 1e6, None
    r = claw.measure(arm)
    base_centre = r["centre"] - np.array([0.0, 0.0, STAND_Z])
    c = 60.0 * float(np.linalg.norm(base_centre - target)) ** 2   # reach the target
    c += 3.0 * r["axis"][2] ** 2 + 3.0 * r["axis"][0] ** 2        # jaws lateral and level
    c += 800.0 * max(0.0, 0.018 - r["d_other"]) ** 2              # off the trunk / RF leg
    c += 800.0 * max(0.0, 0.020 - r["z_min"]) ** 2                # off the floor
    c += 300.0 * max(0.0, 0.030 - r["gap"]) ** 2                  # mouth stays open
    c += 300.0 * max(0.0, 0.025 - r["corridor"]) ** 2             # an object can get in
    c += 800.0 * max(0.0, LIFT_MIN - r["lift"]) ** 2              # the arm can lift it
    # Prefer mid-range joints: a pose pinned against a stop has no room left for
    # the per-episode randomisation, a payload, or a hardware calibration offset.
    for joint, value in zip(ARM_JOINTS, arm, strict=True):
        lo, hi = claw.soft_limits[joint]
        c += 0.6 * ((value - 0.5 * (lo + hi)) / (0.5 * (hi - lo))) ** 4
    return c, r


def search(claw: Claw, target: np.ndarray, seeds: int = 4, samples: int = 4000):
    """Random restarts plus coordinate descent. The cost is cheap and it is 4-D."""
    lo = np.array([claw.soft_limits[j][0] + LIMIT_MARGIN for j in ARM_JOINTS])
    hi = np.array([claw.soft_limits[j][1] - LIMIT_MARGIN for j in ARM_JOINTS])
    best = (np.inf, None)
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        for _ in range(samples):
            x = lo + rng.random(len(ARM_JOINTS)) * (hi - lo)
            c, _ = cost(claw, x, target)
            if c < best[0]:
                best = (c, x)
        best = refine(claw, best[1], target, best)
    return best


def refine(claw: Claw, start, target: np.ndarray, best=None):
    """Coordinate descent from `start`: `search`'s second half on its own.

    For keeping a pose's basin rather than re-choosing it by cost alone -- see
    `claw.py::LF_GRASP` for why the task's pose is refined, not searched.
    """
    x = np.asarray(start, dtype=float).copy()
    if best is None:
        best = (cost(claw, x, target)[0], x.copy())
    step = 0.20
    while step > 1e-4:
        improved = False
        for k in range(len(ARM_JOINTS)):
            for sign in (+1.0, -1.0):
                y = x.copy()
                y[k] += sign * step
                c, _ = cost(claw, y, target)
                if c < best[0]:
                    best, x, improved = (c, y), y, True
        if not improved:
            step *= 0.6
    return best


# ── reporting ─────────────────────────────────────────────────────────────


def report(claw: Claw, arm, label: str) -> None:
    r = claw.measure(arm)
    base_centre = r["centre"] - np.array([0.0, 0.0, STAND_Z])
    print(label)
    for joint, value in zip(ARM_JOINTS, arm, strict=True):
        lo, hi = claw.soft_limits[joint]
        frac = (value - lo) / (hi - lo)
        print(
            f"    {joint:26s} {value:+.4f}  {np.degrees(value):+7.1f} deg"
            f"   soft [{lo:+.3f}, {hi:+.3f}]  at {frac * 100:4.1f}% of range"
        )
    print(f"    mouth centre  {np.round(base_centre, 4)} m in the base frame")
    print(f"    aperture      {r['gap'] * 1000:.1f} mm at finger {GRIPPER_OPEN}")
    print(f"    closing axis  {np.round(r['axis'], 3)}   (want |x|, |z| ~ 0)")
    print(f"    lift          {r['lift'] * 1000:.0f} mm of anvil per rad of shoulder pitch "
          f"(want >= {LIFT_MIN * 1000:.0f})")
    print(
        f"    clearances    {r['d_other'] * 1000:.1f} mm forearm-onwards to trunk/RF/LM "
        f"({r['d_shoulder'] * 1000:.1f} mm including the shoulder, which the pose "
        f"does not decide)"
    )
    print(
        f"                  {r['z_min'] * 1000:.1f} mm above the floor, "
        f"{r['corridor'] * 1000:.1f} mm of corridor ahead"
    )


def stance(claw: Claw) -> None:
    """The five-foot stance: where the robot stands, and what it may lift."""
    for label, carried in (("six-foot HOME", False), ("five-foot HOME + LF_GRASP", True)):
        claw.set_pose([LF_GRASP[j] for j in ARM_JOINTS], GRIPPER_OPEN, carried=carried)
        com, mass = claw.com()
        legs = LEG_FOOT if carried is False else {k: LEG_FOOT[k] for k in FIVE_FOOT_LEGS}
        pts = np.array([claw.site_xy(leg) for leg in legs])
        print(f"\n{label}:  {mass * 1000:.0f} g, {mass * 9.81:.2f} N")
        print(f"    centre of mass       {np.round(com, 4)} (base standing at {STAND_Z})")
        for leg in legs:
            print(f"      {leg} / {LEG_FOOT[leg]:<18s} foot at {np.round(claw.site_xy(leg), 4)}")
        print(f"    support margin       {support_margin(pts, com[:2]) * 1000:+.1f} mm")

    claw.set_pose([LF_GRASP[j] for j in ARM_JOINTS], GRIPPER_OPEN)
    com, _ = claw.com()
    foot = {leg: claw.site_xy(leg) for leg in FIVE_FOOT_LEGS}
    print("\nfive-foot support margin with each swing group airborne")
    print("  (positive = the centre of mass is still inside what is left)")
    for k in (1, 2, 3):
        rows = []
        for swing in itertools.combinations(FIVE_FOOT_LEGS, k):
            stay = [leg for leg in FIVE_FOOT_LEGS if leg not in swing]
            rows.append(
                (
                    support_margin(np.array([foot[s] for s in stay]), com[:2]),
                    "+".join(swing),
                    "+".join(stay),
                )
            )
        rows.sort(reverse=True)
        print(f"  -- {k} airborne --")
        for margin, swing, stay in rows:
            print(f"     swing {swing:<12s} stance {stay:<20s} {margin * 1000:+7.1f} mm")

    print(f"\npayload at {PAYLOAD_BODY}")
    body = claw._id(mujoco.mjtObj.mjOBJ_BODY, PAYLOAD_BODY)
    shoulder = claw.d.xpos[claw._id(mujoco.mjtObj.mjOBJ_BODY, f"{CARRIED_LEG}_J1_link")]
    at = claw.d.xipos[body]
    arm_m = float(np.linalg.norm((at - shoulder)[:2]))
    base_com, base_mass = claw.com()
    pts = np.array([foot[leg] for leg in FIVE_FOOT_LEGS])
    for extra in (0.1, 0.3, 0.6):
        shifted = (base_com * base_mass + at * extra) / (base_mass + extra)
        print(
            f"    {extra * 1000:4.0f} g   centre of mass {(shifted - base_com)[0] * 1000:+5.1f} mm x"
            f"   support margin {support_margin(pts, shifted[:2]) * 1000:+6.1f} mm"
            f"   shoulder hold {extra * 9.81 * arm_m:.3f} N.m"
        )
    print(f"    (moment arm from the shoulder-pitch axis: {arm_m * 1000:.0f} mm)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help="the model XML")
    ap.add_argument("--report", action="store_true", help="measure the pose in use")
    ap.add_argument("--aperture", action="store_true", help="mouth opening vs finger angle")
    ap.add_argument("--check-box", action="store_true", help="every LF_GRASP_BOX corner")
    ap.add_argument("--stance", action="store_true", help="five-foot support polygon")
    ap.add_argument("--search", action="store_true", help="re-derive the pose")
    ap.add_argument("--refine", action="store_true",
                    help="coordinate descent from --start (default LF_GRASP) toward --target")
    ap.add_argument("--start", type=float, nargs=4, default=None,
                    help="arm angles for --refine to start from, in ARM_JOINTS order")
    ap.add_argument(
        "--target", type=float, nargs=3, default=list(DEFAULT_TARGET),
        help="desired mouth position in the base frame, for --search and --refine",
    )
    args = ap.parse_args()
    if not (args.report or args.aperture or args.check_box or args.stance or args.search
            or args.refine):
        args.report = True

    claw = Claw(args.model)
    nominal = [LF_GRASP[j] for j in ARM_JOINTS]

    if args.report:
        report(claw, nominal, "LF_GRASP, as held by tasks/jumper/five_foot:")

    if args.aperture:
        # Swept over the joint's own travel, not to a hardcoded 0.0. The V1.6
        # table stopped at 0.0 because that was the shut pose then; on V1.6.1 the
        # jaws are still 9.6 mm apart there and only meet at the +0.10 limit, and
        # a sweep that ends before shut cannot show that it moved.
        # The joint's own range, not `_soft_limit`: the shut pose is exactly at the
        # +0.10 stop, and a soft limit would trim off the one end this has to show.
        lo, hi = (float(v) for v in claw.m.jnt_range[
            claw._id(mujoco.mjtObj.mjOBJ_JOINT, FINGER_JOINT)])
        print(f"\nmouth opening vs {FINGER_JOINT} over its travel [{lo:+.2f}, "
              f"{hi:+.2f}] (visual meshes; see the hull caveat)")
        print(f"    {'finger':>6}   {'mouth':>8}   {'closing axis, palm frame':>26}")
        for finger in np.arange(lo, hi + 1e-9, 0.05):
            r = claw.measure(nominal, float(finger))
            print(f"    {finger:+.2f}   {r['gap'] * 1000:6.1f} mm   "
                  f"{np.round(r['axis'], 3)}")

    if args.check_box:
        print(f"\nLF_GRASP_BOX, all {2 ** len(ARM_JOINTS)} corners at +-{GRASP_BOX} rad:")
        worst = {"d_other": np.inf, "d_shoulder": np.inf, "z_min": np.inf, "gap": np.inf,
                 "lift": np.inf}
        illegal = []
        for bits in range(2 ** len(ARM_JOINTS)):
            arm = [
                v + (GRASP_BOX if (bits >> k) & 1 else -GRASP_BOX)
                for k, v in enumerate(nominal)
            ]
            for joint, value in zip(ARM_JOINTS, arm, strict=True):
                lo, hi = claw.soft_limits[joint]
                if not (lo <= value <= hi):
                    illegal.append((joint, value))
            r = claw.measure(arm)
            for key, seen in list(worst.items()):
                worst[key] = min(seen, r[key])
        print(f"    worst clearance, forearm on   : {worst['d_other'] * 1000:.1f} mm")
        print(f"    worst clearance, shoulder on  : {worst['d_shoulder'] * 1000:.1f} mm")
        print(f"    worst height above the floor  : {worst['z_min'] * 1000:.1f} mm")
        print(f"    worst aperture                : {worst['gap'] * 1000:.1f} mm")
        print(f"    worst lift                    : {worst['lift'] * 1000:.0f} mm/rad")
        print(f"    outside a soft joint limit    : {illegal or 'none'}")

    if args.stance:
        stance(claw)

    if args.search:
        target = np.array(args.target)
        c, arm = search(claw, target)
        report(claw, arm, f"\nsearch -> base-frame target {np.round(target, 4)} (cost {c:.4f}):")
        print("\n    paste into LF_GRASP:")
        for joint, value in zip(ARM_JOINTS, arm, strict=True):
            print(f'        "{joint}": {value:.4f},')

    if args.refine:
        target = np.array(args.target)
        start = nominal if args.start is None else args.start
        c, arm = refine(claw, start, target)
        report(claw, arm, f"\nrefine {np.round(start, 4)} -> base-frame target "
                          f"{np.round(target, 4)} (cost {c:.4f}):")
        print("\n    paste into LF_GRASP:")
        for joint, value in zip(ARM_JOINTS, arm, strict=True):
            print(f'        "{joint}": {value:.4f},')


if __name__ == "__main__":
    main()
