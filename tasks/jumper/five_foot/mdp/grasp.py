"""Where a prop sits in the claw, and `play --hold`.

**There is no grasp term.** The claw holds what it squeezes the way a real one
does, by friction: squeeze the trigger until the jaws stop on an object, and it
comes up with the claw; let go, and it drops. What makes that work is not in this
file. It is the
friction settings the objects scene runs (`tasks/jumper/five_foot/objects.py::PROP_CONE`),
without which a squeezed object creeps out of the jaws however hard they squeeze.

What is here is the claw's geometry in its own frame -- the anvil, the direction
the jaws close in, the way into the mouth -- and the one runtime use of it:
starting an episode with a prop already squeezed in the mouth, for looking at what
the trunk does with a load on the front.

## What this used to be

A weld. kk-rl-lab's pincer lost objects under contact alone -- sphere pads let a
block pivot out on the lift, capsules let it squirt past the fixed jaw, parallel
plates at a 33 N pinch let it slide out of the front -- and it settled on welding
a taken object to the fixed jaw (`deploy/hexa_5d/backends/mujoco/pick_place.py`).
This file carried that mechanism through two triggers, and each failed from the
operator's seat:

* whatever came within 5 mm of a point that turned out to be in mid-air beside
  the claw (`_jaw_clouds` has why), with nothing commanded -- walking the claw
  past a prop took it;
* then kk-rl-lab's own rule: the jaws commanded at least half shut, with the prop
  within 20 mm of the anvil. Inside the mouth the wedge is narrower than at the
  tips, so an object there stops the jaws before half shut. Measured on warp, a
  50 mm cylinder standing in the open mouth, the claw shut until the jaws stopped
  on it, then the claw raised 13 to 18 mm by the shoulder:

      how deep in the mouth    jaws stopped at closure    welded    cylinder rose
      at the tips                      0.50                 yes      with the claw
      15 mm in                         0.45                 no       7.8 mm
      30 mm in                         0.45                 no       2.0 mm
      45 mm in                         0.45                 no       2.5 mm

  What the operator saw was a cylinder in a claw that visibly held it, sliding
  down as the claw rose.

That looked like too little friction, and it was a solver whose friction could not
hold -- the thing the weld had been standing in for all along. Under the objects
scene's friction settings the same claw carries all three solids by friction
alone, so the weld went, and with it every threshold that decided when an object
counted as taken.

Convex-decomposed jaws are the other half of it and are what make any of this
usable: with one convex hull per jaw link the mouth is filled in and an object
cannot get between the jaws at all. `tasks/jumper/five_foot/tools/jaw_decompose.py` cuts them and
`five_foot/jaws.py` puts them on this task's robot -- and on no other task's.

## Replay only

`ClawHold` is installed in replay only, like the claw's teleop: the policy never
drives the finger, and there are no props in training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from scipy.spatial import ConvexHull, cKDTree

from ..claw import (
    FINGER_JOINT,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    KNUCKLE_CLEAR,
    LF_GRASP,
)
from .gripper import GripperTeleop, squeeze_limited

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "HOLD_DEPTH",
    "HOLD_STEPS",
    "PLACE_STEPS",
    "SEAT_CLEARANCE",
    "Claw",
    "ClawHold",
    "claw_hold_event",
]

#: How far off the anvil a prop is parked when something puts one in the claw --
#: `play --hold`, and `tasks/jumper/five_foot/tools/grasp_objects.py` -- in metres.
#:
#: The anvil is a vertex of the fixed jaw's *visual* mesh, and the collision
#: pieces covering it sit proud of it, so an object seated exactly on it starts
#: out already pressed into the jaw. Measured on V1.6 as the distance from an
#: 8 mm ball to the nearest claw collision piece, the ball moved off the anvil
#: along `_closing_in_palm` with the claw at `GRIPPER_OPEN`, by the surface
#: sampled 2 mm apart (so each is at most 2 mm high),
#:
#:     clearance off the anvil    2      3      4      5      6 mm
#:     ball to the claw         +1.2   +2.0   +2.8   +3.7   +4.5 mm
#:
#: With the claw opened to -1.00 instead, the closing direction is the one past
#: the turn in `claw.py`'s sweep and the same row reads -0.6, +0.0, +0.6, +1.0
#: and +1.6: so 5 rather than 2, which is clear by more than the sampling error
#: either way. The contact test that decides it is
#: `tests/test_five_foot.py::test_the_left_claw_closes_on_something_in_its_mouth`,
#: which parks its ball this far off.
SEAT_CLEARANCE = 0.005

#: How far past the anvil, into the mouth, a prop is put by the same two, in
#: metres.
#:
#: **Not on the anvil.** That is where the jaws meet, and there they are a wedge:
#: measured on warp on the previous model, a 50 mm cylinder seated against the
#: anvil was squeezed straight out of the mouth as the claw closed, while 15, 30
#: and 45 mm in it was held and carried (`tasks/jumper/five_foot/objects.py::PROP_CONE` has the
#: table).
#:
#: **20 mm on V1.6, down from 30**, and not a free choice: the payload body is
#: seated at this depth too (`claw.py::add_payload_body`), and on V1.6 going deeper
#: runs the object into the wrist housing beside the finger's hinge. Measured
#: against the whole fixed-jaw mesh (knuckle included -- leaving it out, as
#: `_jaw_clouds` does for the anvil, overstates the room by the hinge), at
#: `LF_GRASP` with the claw at `GRIPPER_OPEN`, a cylinder seated this way clears
#: the fixed jaw by
#:
#:     depth                  0     10     15     20     25     30     40 mm
#:     65 mm (PAYLOAD_CYL.)  32.6   36.4   39.0   42.0   38.6   34.2   26.1 mm   radius 32.5
#:     58 mm (the can)       29.4   32.9   35.5   38.5   38.9   34.5   25.9 mm   radius 29.0
#:
#: 20 is where `PAYLOAD_CYLINDER` clears by the most, 9.5 mm, and the can, the
#: widest thing in the row, by 9.5 as well. On the previous hold pose with the
#: claw opened to -1.00 the same cylinder cleared at 15 mm only, and by 0.4.
HOLD_DEPTH = 0.020

#: Control steps at the start of an episode during which `--hold` keeps the prop
#: where it put it, with the claw open: 10 is 0.2 s at 50 Hz, for the robot to
#: settle off its spawn drop without leaving the prop behind.
#:
#: **With the claw open, and never while it squeezes.** A prop whose pose is
#: written back every step cannot yield, so a finger closing on it at full torque
#: drives into it, and the overlap throws the prop out when it is let go. Measured
#: on warp, a 50 mm cylinder held in place through a 0.5 s close: the finger shut on
#: nothing and the cylinder landed 289 mm away, and 1.4 m on a second run. Over
#: three props, three resets each and both backends, closing on a prop held in
#: place carried 16 in 18; closing on one let go of first carried 53 in 54, whether
#: the finger was sent shut at once, over 0.5 s or a key press at a time.
PLACE_STEPS = 10

#: The control step by which `--hold` has shut the claw on the prop and handed it
#: to the operator: 25 steps, 0.5 s, after `PLACE_STEPS`, where the finger's servo
#: takes about 0.2 s to walk the jaw across from wide open.
HOLD_STEPS = 35


@dataclass(frozen=True)
class _Hull:
    """A convex mesh geom as the geometry below takes it, in the geom's own frame.

    `vertices` is `[N, 3]` and `planes` `[M, 4]`, each face as `n . x + d <= 0`
    inside. See `_geom_shape`.
    """

    vertices: np.ndarray
    planes: np.ndarray


def _surface_distance(local: torch.Tensor, gtype: int, size: Any) -> torch.Tensor:
    """Distance from a point to one shape's surface, in the geom's own frame.

    `local` is `[B, 3]`, and the result is negative inside. A hull (`_Hull`) gives
    its largest distance to a face plane: exact inside and over a face, and short
    of the true distance off an edge or a corner, so a prop seated by it is never
    closer than asked. Other mesh geoms fall back to their bounding sphere, which
    parks them further out rather than inside the jaw.
    """
    if gtype == mujoco.mjtGeom.mjGEOM_MESH and isinstance(size, _Hull):
        planes = torch.as_tensor(size.planes, device=local.device, dtype=local.dtype)
        return (local @ planes[:, :3].T + planes[:, 3]).max(dim=1).values
    x, y, z = local[:, 0], local[:, 1], local[:, 2]
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return local.norm(dim=1) - float(size[0])
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        half = float(size[1])
        axial = z.clamp(-half, half)
        return torch.stack([x, y, z - axial], dim=1).norm(dim=1) - float(size[0])
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radial = torch.hypot(x, y) - float(size[0])
        axial = z.abs() - float(size[1])
        outside = torch.hypot(radial.clamp(min=0.0), axial.clamp(min=0.0))
        return torch.where(
            (radial > 0) | (axial > 0), outside, torch.maximum(radial, axial)
        )
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        half = torch.tensor(size[:3], device=local.device, dtype=local.dtype)
        q = local.abs() - half
        return q.clamp(min=0.0).norm(dim=1) + q.max(dim=1).values.clamp(max=0.0)
    return local.norm(dim=1) - float(np.linalg.norm(size[:3]))


def _support(direction: torch.Tensor, gtype: int, size: Any) -> torch.Tensor:
    """How far one shape reaches along `direction`, in the geom's own frame.

    `direction` is `[B, 3]` and unit length; the result is the largest dot product
    of any point of the shape with it -- the half-width the shape presents that
    way, exact for a hull (`_Hull`). Other mesh geoms fall back to their bounding
    sphere, which parks them further out rather than inside the jaw.
    """
    if gtype == mujoco.mjtGeom.mjGEOM_MESH and isinstance(size, _Hull):
        vertices = torch.as_tensor(size.vertices, device=direction.device, dtype=direction.dtype)
        return (direction @ vertices.T).max(dim=1).values
    x, y, z = direction[:, 0].abs(), direction[:, 1].abs(), direction[:, 2].abs()
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return torch.full_like(x, float(size[0]))
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(size[0]) + float(size[1]) * z
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(size[0]) * torch.hypot(x, y) + float(size[1]) * z
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return float(size[0]) * x + float(size[1]) * y + float(size[2]) * z
    return torch.full_like(x, float(np.linalg.norm(size[:3])))


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def _seat_position(
    anvil: torch.Tensor,
    closing: torch.Tensor,
    floor: torch.Tensor,
    shapes: list[tuple[int, np.ndarray, torch.Tensor, torch.Tensor]],
    clearance: float = SEAT_CLEARANCE,
    iterations: int = 30,
) -> torch.Tensor:
    """Where an unrotated prop's origin goes to rest against the anvil, `[B, 3]`.

    `anvil` and `closing` are `[B, 3]` in world coordinates, `floor` is `[B]`, and
    `shapes` is the prop's collision geoms as `(type, size, pos, rotation)` on its
    body. The prop is moved out from the anvil along `closing` until its nearest
    surface is `clearance` away -- found by bisection on `_surface_distance`,
    because the distance from a point to a convex shape translating along a line
    has one crossing and no closed form once boxes, cylinders and a floor are
    involved -- and never lower than its base clearing the floor.
    """
    count = anvil.shape[0]
    down = torch.tensor([0.0, 0.0, -1.0], device=anvil.device, dtype=anvil.dtype).expand(count, 3)
    below = torch.zeros(count, device=anvil.device, dtype=anvil.dtype)
    far = torch.zeros(count, device=anvil.device, dtype=anvil.dtype)
    for gtype, size, pos, rot in shapes:
        below = torch.maximum(below, _support(down @ rot, gtype, size) + down @ pos)
        far = torch.maximum(far, _support(closing @ rot, gtype, size) + closing @ pos)
    lowest = floor + below + clearance

    def place(offset: torch.Tensor) -> torch.Tensor:
        centre = anvil + closing * offset.unsqueeze(1)
        return torch.cat([centre[:, :2], torch.maximum(centre[:, 2], lowest).unsqueeze(1)], dim=1)

    def gap(centre: torch.Tensor) -> torch.Tensor:
        nearest = torch.full((count,), float("inf"), device=anvil.device, dtype=anvil.dtype)
        for gtype, size, pos, rot in shapes:
            local = (anvil - centre - pos) @ rot
            nearest = torch.minimum(nearest, _surface_distance(local, gtype, size))
        return nearest

    # Centred on the anvil the prop contains it; pushed out by its whole extent
    # along `closing` plus the clearance it is at least that far clear of it.
    lo = torch.zeros(count, device=anvil.device, dtype=anvil.dtype)
    hi = far + clearance
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        close = gap(place(mid)) < clearance
        lo = torch.where(close, mid, lo)
        hi = torch.where(close, hi, mid)
    return place(hi)


def _geom_shape(
    model: mujoco.MjModel, g: int, device: Any, dtype: torch.dtype
) -> tuple[int, Any, torch.Tensor, torch.Tensor]:
    """Geom `g` as `_seat_position` takes it: `(type, size, pos, rotation)` on its body.

    A mesh comes with its hull (`_Hull`), from the vertices MuJoCo stores for it --
    in the mesh's own frame, which `geom_pos` and `geom_quat` place on the body, so
    they go through the geom rather than being read against the body.
    """
    gtype = int(model.geom_type[g])
    size: Any = model.geom_size[g]
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[g])
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = np.asarray(model.mesh_vert[start:start + count], dtype=np.float64)
        size = _Hull(vertices, ConvexHull(vertices).equations)
    return (gtype, size,
            torch.as_tensor(model.geom_pos[g], device=device, dtype=dtype),
            torch.as_tensor(_quat_to_mat(model.geom_quat[g]), device=device, dtype=dtype))


def _rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """`quat` applied to `vec`, both `[B, ...]`."""
    w, xyz = quat[:, :1], quat[:, 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=1)
    return vec + w * t + torch.cross(xyz, t, dim=1)


def _named(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    """`name`'s id, as the training model spells it (`robot/...`) or as the bare
    asset does -- and a loud failure when it is neither.

    **Never -1.** The environment's model is built by attaching the robot's spec
    under that prefix; `claw.py::add_payload_body` compiles the bare spec, where
    the same bodies have no prefix at all. These helpers used to look names up
    with the prefix only, and `mj_name2id` answers a miss with -1 rather than an
    error. On the bare asset every joint lookup missed, so the arm was never
    posed, and the palm resolved to index -1, which numpy reads as **the last
    body in the model** (`RR_foot_tip_link`). The result was a plausible-looking
    vector in the frame of a toe: measured in the training model, it put the
    payload 116 mm off, back at the wrist, in every run since the payload body
    arrived.
    """
    for candidate in (f"robot/{name}", name):
        found = mujoco.mj_name2id(model, objtype, candidate)
        if found >= 0:
            return found
    raise ValueError(
        f"no {objtype.name} named {name!r} or 'robot/{name}' in this model; a "
        f"measurement taken without it would be silently wrong, not missing"
    )


def _jaw_clouds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """The two jaws' drawn surfaces, in world coordinates, knuckle excluded.

    **The fixed jaw is `LF_palm_link`.** It is the hooked upper jaw, with the
    serrated face an object is pinched against; `LF_palm_pad_b_link` is the pad on the
    *outside* of its tip, which is what the robot walks on in six-foot mode. Both
    are here because both can be the surface an object arrives on, and the
    knuckle -- where the moving jaw and the palm are two halves of one hinge, and
    therefore touch at every aperture -- is excluded from both, exactly as
    `tasks/jumper/five_foot/tools/grasp_pose.py` does it.

    **Every vertex goes through its geom's frame.** `mesh_vert` is expressed in
    the mesh's own inertial frame, which the compiler re-orients and compensates
    for on each geom; reading it against the *body* frame instead -- which this
    did -- puts the cloud somewhere near the part rather than on it. Measured on
    `jumper.xml` at the grasp pose, the palm's vertices landed up to 83 mm from the
    palm, the finger's up to 58 mm and the pad's up to 32 mm, so the "prong face"
    the weld of the time was measured against was a point in mid-air beside the
    claw. That is what let the claw take an object with a visible gap between
    them.
    """
    hinge = data.xpos[_named(model, mujoco.mjtObj.mjOBJ_BODY, "LF_finger_link")]

    def cloud(body: str) -> np.ndarray:
        gid = _named(model, mujoco.mjtObj.mjOBJ_GEOM, f"{body}_visual")
        mesh = int(model.geom_dataid[gid])
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[start:start + count]
        return verts @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]

    def jaw(bodies: tuple[str, ...]) -> np.ndarray:
        pts = np.vstack([cloud(b) for b in bodies])
        return pts[np.linalg.norm(pts - hinge, axis=1) > KNUCKLE_CLEAR]

    return (jaw(("LF_finger_link", "LF_finger_tip_link")),
            jaw(("LF_palm_link", "LF_palm_pad_f_link", "LF_palm_pad_b_link")))


def _posed(model: mujoco.MjModel, finger: float) -> mujoco.MjData:
    """The arm at `LF_GRASP` with the finger at `finger`, forward-computed."""
    data = mujoco.MjData(model)
    for joint, value in LF_GRASP.items():
        data.qpos[model.jnt_qposadr[_named(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]] = value
    data.qpos[model.jnt_qposadr[_named(model, mujoco.mjtObj.mjOBJ_JOINT, FINGER_JOINT)]] = finger
    mujoco.mj_forward(model, data)
    return data


def _mouth_in_palm(model: mujoco.MjModel) -> np.ndarray:
    """Where the claw holds things, in `LF_palm_link`'s frame.

    **The anvil: the point on the fixed jaw that the moving jaw arrives at when
    the claw shuts.** An object in this claw is pushed against the fixed jaw and
    pinched there, so that point is where it ends up, and the aperture at any
    finger angle is how far the moving jaw still has to travel to reach it
    (`claw.py::APERTURE_MM`, measured exactly this way).

    Both jaws hang off the palm and only the finger joint moves between them, so
    this is a constant of the claw rather than of the arm's pose -- which is what
    lets it survive the hold randomisation that moves the arm by +-0.10 rad every
    episode. kk-rl-lab keeps the same point as a **base**-frame constant measured
    on one pose (`MOUTH_CENTRE_B`); this is that constant one link further down,
    where it stops being an approximation.

    Derived at the shut pose rather than mid-close, and that is the correction
    that matters. Mid-close, the closest pair of the two jaws is not where they
    meet: it is the throat, up by the knuckle, where the wedge is narrowest and
    where nothing can be held -- 11 mm of room across at the open end against
    35 mm out over the grip face. Reading the mouth from there gave a point with
    a third of the room it claimed and put the payload, `--hold` and the grasp
    test all in the same wrong place. Measured on this asset, the answer is
    stable for any knuckle exclusion from 30 mm out to 80 (see `KNUCKLE_CLEAR`);
    below 30 the hinge itself is the closest pair and the answer collapses.
    """
    data = _posed(model, GRIPPER_CLOSED)
    moving, fixed = _jaw_clouds(model, data)
    gap, nearest = cKDTree(fixed).query(moving)
    anvil = fixed[nearest[int(gap.argmin())]]
    palm = _named(model, mujoco.mjtObj.mjOBJ_BODY, "LF_palm_link")
    return data.xmat[palm].reshape(3, 3).T @ (anvil - data.xpos[palm])


def _closing_in_palm(model: mujoco.MjModel) -> np.ndarray:
    """The direction from the anvil toward the moving jaw, in `LF_palm_link`'s frame.

    Which side of the fixed jaw a seated object lies on: it is pinched between
    the anvil and the moving jaw, so it sits this way from the anvil. Taken with
    the claw wide open, where the two jaws are far enough apart for the direction
    to mean something -- shut, the moving jaw is *on* the anvil and there is none.
    """
    data = _posed(model, GRIPPER_OPEN)
    palm = _named(model, mujoco.mjtObj.mjOBJ_BODY, "LF_palm_link")
    rot = data.xmat[palm].reshape(3, 3)
    anvil = data.xpos[palm] + rot @ _mouth_in_palm(model)
    moving, _ = _jaw_clouds(model, data)
    toward = moving[np.argmin(np.linalg.norm(moving - anvil, axis=1))] - anvil
    return rot.T @ (toward / np.linalg.norm(toward))


def _inward_in_palm(model: mujoco.MjModel) -> np.ndarray:
    """The level direction from the anvil into the mouth, in `LF_palm_link`'s frame.

    Toward the finger's hinge with the vertical taken out, at the grasp pose, so a
    prop moved this way from the anvil goes deeper between the jaws without rising
    off the floor or sinking into it. `HOLD_DEPTH` is measured along it.
    """
    data = _posed(model, GRIPPER_OPEN)
    palm = _named(model, mujoco.mjtObj.mjOBJ_BODY, "LF_palm_link")
    rot = data.xmat[palm].reshape(3, 3)
    anvil = data.xpos[palm] + rot @ _mouth_in_palm(model)
    inward = data.xpos[_named(model, mujoco.mjtObj.mjOBJ_BODY, "LF_finger_link")] - anvil
    inward[2] = 0.0
    return rot.T @ (inward / np.linalg.norm(inward))


def _held_in_palm(
    model: mujoco.MjModel, radius: float, height: float, depth: float = HOLD_DEPTH
) -> np.ndarray:
    """Where an upright cylinder squeezed in the claw has its centre, in
    `LF_palm_link`'s frame.

    Seated the way `Claw.seat` seats a prop -- its nearest surface
    `SEAT_CLEARANCE` off the anvil on the moving jaw's side, then `depth` into the
    mouth -- with the arm at `LF_GRASP` on a level base and no floor to lift it:
    this is where the object is in the claw, not where it stood before it was
    picked up. `claw.py::add_payload_body` puts the load here.
    """
    data = _posed(model, GRIPPER_OPEN)
    palm = _named(model, mujoco.mjtObj.mjOBJ_BODY, "LF_palm_link")
    pos, rot = data.xpos[palm], data.xmat[palm].reshape(3, 3)

    def batch(vec: np.ndarray) -> torch.Tensor:
        return torch.tensor(vec, dtype=torch.float64).unsqueeze(0)

    cylinder = (int(mujoco.mjtGeom.mjGEOM_CYLINDER), np.array([radius, 0.5 * height, 0.0]),
                torch.zeros(3, dtype=torch.float64), torch.eye(3, dtype=torch.float64))
    centre = _seat_position(
        batch(pos + rot @ _mouth_in_palm(model)),
        batch(rot @ _closing_in_palm(model)),
        torch.tensor([-np.inf], dtype=torch.float64),
        [cylinder],
    ) + batch(rot @ _inward_in_palm(model)) * depth
    return rot.T @ (centre[0].numpy() - pos)


def _prop_geoms(model: mujoco.MjModel, prop: str) -> list[int]:
    """The collision geoms of the prop entity `prop`, attached under `<prop>/`."""
    geoms = [
        g for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
            or "").startswith(f"{prop}/")
        and model.geom_contype[g]
    ]
    if not geoms:
        raise ValueError(f"{prop!r} has no collision geoms in this model; is it a prop here?")
    return geoms


class Claw:
    """The claw's geometry, solved once for a model, and where a prop goes in it.

    Everything here is a constant of the claw in `LF_palm_link`'s frame, turned
    into world coordinates from the palm's pose when asked. The pose comes from the
    robot entity rather than from the simulator's arrays, so it reads the same on
    both backends.
    """

    def __init__(self, env: ManagerBasedRlEnv) -> None:
        model = env.sim.mj_model
        self._model = model
        self._robot = env.scene["robot"]
        self.palm = self._robot.find_bodies(["LF_palm_link"])[0][0]

        def vec(value: np.ndarray) -> torch.Tensor:
            return torch.tensor(value, device=env.device, dtype=torch.float32)

        self.mouth_palm = vec(_mouth_in_palm(model))
        self.closing_palm = vec(_closing_in_palm(model))
        self.inward_palm = vec(_inward_in_palm(model))

    def _world(self, vec: torch.Tensor) -> torch.Tensor:
        quat = self._robot.data.body_link_quat_w[:, self.palm]
        return _rotate(quat, vec.expand(quat.shape[0], 3))

    def mouth(self) -> torch.Tensor:
        """The anvil in the world, `[num_envs, 3]`."""
        return self._robot.data.body_link_pos_w[:, self.palm] + self._world(self.mouth_palm)

    def seat(self, env: ManagerBasedRlEnv, prop: str, depth: float = HOLD_DEPTH) -> torch.Tensor:
        """Where `prop`'s origin goes to be squeezed in the claw, `[num_envs, 3]`.

        Upright and unrotated, its nearest surface `SEAT_CLEARANCE` from the anvil
        on the moving jaw's side of it, then `depth` further into the mouth.
        `_seat_position` does the geometry from the prop's own collision geoms, so
        it is right for a 30 mm notebook and a 58 mm can alike.

        **It used to offset the prop by its extent along the closing direction,
        and that parks it by a corner.** The claw closes 14 degrees below level
        and 26 off its own y, so an object pushed out by its reach along that line
        touches the plane through the anvil at whichever corner reaches furthest
        -- and a thin, tall plate's is nowhere near its face. Measured on the row
        of the time, the robot holding its reset pose: a 20 mm plate's face sat
        38 mm from the anvil, and a 50 mm block's and cylinder's 16 and 12 where 2
        was meant; the notebook's comes out 33 mm off it the same way.

        **And never through the floor.** The mouth rides 50 to 70 mm up and the
        row's objects stand 100 to 145 mm tall, so an object centred on the mouth
        has its base below the ground: measured with the robot holding its reset
        pose, on the row of the time, a 160 mm plate's by 41 mm and a 100 mm
        block's and cylinder's by 11. The
        solver ejects it on every step it is parked there. Raised until its base
        clears the floor instead, it is gripped below its middle, which is where a
        claw this low grips a tall thing standing on the ground.
        """
        mouth = self.mouth()
        closing = self._world(self.closing_palm)
        inward = self._world(self.inward_palm)
        model = self._model
        # The prop goes in unrotated, so its body frame is the world's and only
        # each geom's own placement on the body is left to account for.
        shapes = [_geom_shape(model, g, mouth.device, mouth.dtype)
                  for g in _prop_geoms(model, prop)]
        floor = env.scene.env_origins[:, 2].to(dtype=mouth.dtype)
        return _seat_position(mouth, closing, floor, shapes) + inward * depth


class ClawHold:
    """Start an episode with a prop squeezed in the claw -- `play --hold`.

    For looking at a carried load: driving up to an object and closing on it is a
    two-hand job with one operator, and what is being watched -- what the trunk
    does with an object hanging off the front -- starts after that.

    For the first `PLACE_STEPS` control steps of an episode it keeps the prop
    where `Claw.seat` puts it, with the claw open; then it lets go of the prop and
    shuts the claw on it, and by `HOLD_STEPS` the squeeze is all that holds it,
    exactly as if the operator had put it there -- never both at once, for the
    reason `PLACE_STEPS` gives. A step event rather than a reset event, because the
    claw's own reset (`hold_carried_arm`, which pins the finger wide open in
    replay) has to have happened first, and the jaws need time to arrive.

    **It also leaves the claw shut.** The teleop writes what the operator's
    trigger asks for every step once the operator has taken over, and a trigger
    at rest asks for open; untold, the step after the hold opens the jaws on the
    prop. So the hold hands the teleop a shut claw (`GripperTeleop.hold_at`),
    which it keeps until the trigger is squeezed all the way -- and letting go
    then opens it. Registered after the teleop, so on the steps it holds its own
    target is the one that stands.

    Inert until `env_cfg.py::_start_holding` (`--hold`) sets `start_holding`.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
        del cfg
        self._robot = env.scene["robot"]
        self._finger = self._robot.find_joints([FINGER_JOINT])[0][0]
        # Solved on first use: every replay builds this term and almost none of
        # them hold anything, and the anvil is a nearest-point search over the
        # jaws' meshes.
        self._claw: Claw | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None = None,
        *,
        start_holding: str | None = None,
        depth: float = HOLD_DEPTH,
        place: int = PLACE_STEPS,
        steps: int = HOLD_STEPS,
        **_: Any,
    ) -> None:
        del env_ids
        if start_holding is None:
            return
        now = env.episode_length_buf
        placing = (now <= place).nonzero(as_tuple=False).flatten()
        closing = ((now > place) & (now <= steps)).nonzero(as_tuple=False).flatten()
        if placing.numel() == 0 and closing.numel() == 0:
            return
        if self._claw is None:
            self._claw = Claw(env)

        if placing.numel():
            prop = env.scene[start_holding]
            quat = torch.zeros(placing.numel(), 4, device=env.device)
            quat[:, 0] = 1.0
            prop.write_root_link_pose_to_sim(
                torch.cat([self._claw.seat(env, start_holding, depth)[placing], quat], dim=1),
                env_ids=placing,
            )
            prop.write_root_com_velocity_to_sim(
                torch.zeros(placing.numel(), 6, device=env.device), env_ids=placing
            )
            opened = torch.full((placing.numel(), 1), GRIPPER_OPEN, device=env.device)
            self._robot.set_joint_position_target(
                opened, joint_ids=[self._finger], env_ids=placing
            )

        if closing.numel():
            closed = torch.full((closing.numel(), 1), GRIPPER_CLOSED, device=env.device)
            angle = self._robot.data.joint_pos[closing][:, [self._finger]]
            # No harder than the operator's own squeeze (`claw.py::SQUEEZE_LEAD`).
            self._robot.set_joint_position_target(
                squeeze_limited(closed, angle), joint_ids=[self._finger], env_ids=closing
            )
            teleop = env.event_manager.get_term_cfg("gripper_teleop").func
            if isinstance(teleop, GripperTeleop):
                teleop.hold_at(GRIPPER_CLOSED)


def claw_hold_event(
    *, depth: float = HOLD_DEPTH, place: int = PLACE_STEPS, steps: int = HOLD_STEPS
) -> EventTermCfg:
    """The event term config, so the wiring reads in one line where it is used."""
    return EventTermCfg(
        func=ClawHold,
        mode="step",
        params={"start_holding": None, "depth": depth, "place": place, "steps": steps},
    )
