"""The pick-place row `jumper.five_foot` puts in front of the robot with `--objects`.

These were scene tests, in `tests/test_scenes.py`, while the row was a scene
(`--scene objects`). The row moved into the task (`tasks/jumper/five_foot/objects.py`)
with everything else the task needs, and its tests moved with it: the props'
geometry and contact parameters are unchanged, and what used to be read off the
scene registry -- the friction cone, the fixed spawn, the list of graspable props --
is read off `apply_objects` and the task's own replay config.
"""

from __future__ import annotations

import warnings

import pytest

pytest.importorskip("mujoco", reason="the props are MuJoCo specs")


class _Event:
    """An `EventTermCfg` stand-in: `apply` only ever reads `params`."""

    def __init__(self, params: dict) -> None:
        self.mode = "reset"
        self.params = params


class _Twist:
    """The velocity command's fields that `apply` can touch.

    Both fractions are here on purpose: `rel_standing_envs` is the one a still
    start sets and `rel_heading_envs` is the one that must be left alone --
    setting it would turn every environment into a heading-controlled one, which
    is the opposite of standing still.
    """

    def __init__(self) -> None:
        self.rel_standing_envs = 0.1
        self.rel_heading_envs = 0.3


class _Pose:
    def __init__(self) -> None:
        self.rel_neutral_envs = 0.25


def _reset_base() -> _Event:
    """mjlab's own `reset_base` range, verbatim (`velocity_env_cfg.py`)."""
    return _Event({
        "pose_range": {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5),
            "z": (0.01, 0.05), "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
    })


def test_every_graspable_prop_is_the_solid_it_declares() -> None:
    """Each object is a primitive of one material plus the parts it lists.

    The failure this pins is the one the row used to be made of: an object whose
    mass is spread by hand over pieces, or whose collision is a set of pieces under
    a mesh, moves and tips in ways that follow from how it was assembled rather
    than from what it is. Nothing raises, and it looks fine.

    So each prop compiles to **its primitive and exactly the parts it declares**,
    every one collidable, weighing what the object weighs. One with no parts has
    to be the uniform solid: the textbook inertia, compared sorted because MuJoCo
    stores a body's principal moments in whatever order its principal-axis frame
    came out, and the centre of mass at the geometric centre. A part has to stand
    proud of the primitive -- one buried inside it adds mass and nothing a contact
    can meet.

    The control is the notebook checked against its own declaration with the belt
    left off: the geom count has to reject it, or the check passes whatever it is
    given.
    """
    import dataclasses

    import mujoco
    import numpy as np

    from tasks.jumper.five_foot.objects import SOLIDS, _solid_spec

    shapes = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER,
              mujoco.mjtGeom.mjGEOM_MESH)

    def problems(solid, model) -> list[str]:
        if model.ngeom != 1 + len(solid.parts):
            return [(f"{solid.name} is {model.ngeom} geoms, not a primitive and "
                     f"{len(solid.parts)} parts")]
        out = []
        if model.geom_type[0] not in shapes:
            out.append(f"{solid.name} is not a box, a cylinder or a hull")
        if not all(model.geom_contype[g] and model.geom_conaffinity[g]
                   for g in range(model.ngeom)):
            out.append(f"a geom of {solid.name} does not collide, so what is drawn is "
                       "not what the claw meets")
        body = int(model.geom_bodyid[0])
        mass = float(model.body_mass[body])
        if mass != pytest.approx(solid.mass, rel=1e-6):
            out.append(f"{solid.name} weighs {mass * 1000:.1f} g, not {solid.mass * 1000:.1f}")
        half = np.array(solid.half_extents)
        for part in solid.parts:
            plo, phi = (np.array(b) for b in part.bounds)
            if not ((phi > half + 1e-9).any() or (plo < -half - 1e-9).any()):
                out.append(f"{solid.name}'s {part.name} is buried in the shape")
        if not solid.parts and solid.kind != "hull":
            a, b, c = solid.half_extents
            if solid.kind == "box":
                want = [mass / 3.0 * (b * b + c * c), mass / 3.0 * (a * a + c * c),
                        mass / 3.0 * (a * a + b * b)]
            else:
                across = mass * (3.0 * a * a + 4.0 * c * c) / 12.0
                want = [across, across, mass * a * a / 2.0]
            if sorted(model.body_inertia[body]) != pytest.approx(sorted(want), rel=1e-6):
                out.append(f"{solid.name}'s inertia is not that of a uniform {solid.kind}")
        if not solid.parts and not np.allclose(model.body_ipos[body], 0.0, atol=1e-9):
            out.append(f"{solid.name}'s centre of mass is off its centre")
        return out

    for solid in SOLIDS:
        found = problems(solid, _solid_spec(solid).compile())
        assert not found, found

    notebook = next(s for s in SOLIDS if s.parts)
    undeclared = dataclasses.replace(notebook, parts=())
    assert problems(undeclared, _solid_spec(notebook).compile()), (
        "the notebook passed with its belt left out of the declaration, so the "
        "check above cannot see an undeclared piece"
    )


def _floor_and(solid, place_jaw: bool = False, priority: int | None = None):
    """A compiled model: MuJoCo's default ground plane, `solid` resting on it.

    The plane carries nothing but MuJoCo's defaults, which is what mjlab's own
    ground plane is (`terrain_entity.py::_import_ground_plane` sets a type and a
    size). `place_jaw` adds a geom with the claw's grip contact touching the
    solid's side, and `priority` overrides the solid's -- for the control.
    """
    import mujoco

    from tasks.jumper.five_foot.jaws import GRIP_FRICTION
    from tasks.jumper.five_foot.objects import _solid_spec

    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep = 0.005
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0.0, 0.0, 0.01])
    child = _solid_spec(solid)
    if priority is not None:
        child.body(solid.name).geoms[0].priority = priority
    _, half_y, _ = solid.half_extents
    rest = -solid.bounds[0][2]
    # A hair into the floor: a hull resting exactly on the plane is not in contact
    # with it as far as MuJoCo's collision is concerned, where a box is -- measured,
    # the soap's hull touched only the jaw.
    spec.attach(child, prefix="prop/",
                frame=spec.worldbody.add_frame(pos=[0, 0, rest - 1e-4]))
    if place_jaw:
        spec.worldbody.add_geom(
            name="jaw", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.01],
            pos=[0.0, half_y + 0.0099, rest], priority=1, condim=3,
            friction=list(GRIP_FRICTION),
        )
    return spec.compile()


def test_a_prop_slides_on_the_floor_and_grips_in_the_jaw() -> None:
    """The friction a contact actually gets, not the numbers on either geom.

    MuJoCo derives a contact's parameters from its two geoms: the higher
    `priority` wins outright, and equal priority takes the larger `condim` and
    the element-wise maximum of the friction. `tasks/jumper/five_foot/objects.py` relies on that
    rule to make one set of prop numbers a plastic-on-floor contact on the ground
    and a rubber-jaw contact in the claw, so what is pinned here is the contact
    as compiled -- `mjContact.friction` -- for both.

    The control is the same prop at priority 0. It then ties the ground plane and
    takes the plane's default sliding friction of 1.0, which is the configuration
    where a prop's own friction is written down and never used.
    """
    import mujoco

    from tasks.jumper.five_foot.jaws import GRIP_FRICTION
    from tasks.jumper.five_foot.objects import PROP_CONDIM, PROP_FRICTION, SOLIDS

    def contacts(model):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        prop = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"prop/{solid.name}")
        out = {}
        for c in data.contact[: data.ncon]:
            # Named by the geom that is not the prop's: the can meets the floor
            # with its base, which is a part rather than the geom named after it.
            other = c.geom2 if model.geom_bodyid[c.geom1] == prop else c.geom1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
            out[name] = (int(c.dim), [float(v) for v in c.friction[:4]])
        return out

    for solid in SOLIDS:
        got = contacts(_floor_and(solid, place_jaw=True))
        assert set(got) == {"floor", "jaw"}, f"{solid.name} touches {sorted(got)}"

        dim, friction = got["floor"]
        assert dim == PROP_CONDIM
        assert friction == pytest.approx(
            [PROP_FRICTION[0], PROP_FRICTION[0], PROP_FRICTION[1], PROP_FRICTION[2]]
        ), f"{solid.name} on the floor is not the prop's own contact"

        dim, friction = got["jaw"]
        assert dim == max(PROP_CONDIM, 3)
        assert friction[0] == pytest.approx(max(PROP_FRICTION[0], GRIP_FRICTION[0]))
        assert friction[2] == pytest.approx(max(PROP_FRICTION[1], GRIP_FRICTION[1]))

    soap = next(s for s in SOLIDS if s.name == "soap")
    solid = soap
    _, friction = contacts(_floor_and(soap, priority=0))["floor"]
    assert friction[0] != pytest.approx(PROP_FRICTION[0]), (
        "at priority 0 the prop still decides its floor friction, so the priority "
        "above is not what makes the assertion true"
    )


def _push(solid, across: bool, seconds: float = 2.0) -> tuple[float, float]:
    """Push `solid` at the claw's height and report (travel, worst tilt in degrees).

    A small sphere with the jaw's contact, on a slide joint driven at 0.1 m/s,
    60 mm off the floor -- about where the jaws meet an object with the trunk
    level. `across` pushes along y instead of x, which is how the closing jaw
    meets the row: a thin object's side.
    """
    import math

    import mujoco
    import numpy as np

    from tasks.jumper.five_foot.jaws import GRIP_FRICTION
    from tasks.jumper.five_foot.objects import _solid_spec

    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep = 0.005
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.01])
    half_x, half_y, _ = solid.half_extents
    spec.attach(_solid_spec(solid), prefix="prop/",
                frame=spec.worldbody.add_frame(pos=[0, 0, -solid.bounds[0][2]]))
    axis = [0.0, 1.0, 0.0] if across else [1.0, 0.0, 0.0]
    start = -(half_y if across else half_x) - 0.010
    pusher = spec.worldbody.add_body(
        name="pusher", pos=[0.0, start, 0.060] if across else [start, 0.0, 0.060]
    )
    pusher.add_joint(name="slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)
    pusher.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.006, 0.0, 0.0], mass=5.0,
                    priority=1, condim=3, friction=list(GRIP_FRICTION))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide")]
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"prop/{solid.name}")
    start_pos = data.xpos[body].copy()
    tilt = 0.0
    for _ in range(round(seconds / model.opt.timestep)):
        data.qvel[dof] = 0.1
        mujoco.mj_step(model, data)
        tilt = max(tilt, math.degrees(math.acos(max(-1.0, min(1.0, data.xmat[body][8])))))
    return float(np.linalg.norm((data.xpos[body] - start_pos)[:2])), tilt


def test_pushed_at_the_claw_s_height_the_stocky_props_slide_and_the_thin_ones_tip() -> None:
    """Whether a pushed object slides or falls over is a property of the model.

    It is `friction > half-width / push height`, give or take the solver, and it
    is exactly what the operator sees when the claw walks into the row. The can,
    and the soap pushed along its 66 mm depth, slide; the soap and the notebook
    pushed on their thin sides, the way the closing jaw meets them, tip -- and a
    friction that looks reasonable in isolation can quietly swap those, which is
    what the props' old 1.5 did.

    The notebook pushed **along** its 105 mm depth is the control: the same object
    and the same code giving the opposite verdict, so the tips below are the
    geometry and not a pusher that knocks everything over.
    """
    from tasks.jumper.five_foot.objects import SOLIDS

    solids = {s.name: s for s in SOLIDS}
    for name in ("can", "soap"):
        travel, tilt = _push(solids[name], across=False)
        assert tilt < 20.0 and travel > 0.10, (
            f"the {name} pushed at 60 mm moved {travel * 1000:.0f} mm and tilted "
            f"{tilt:.0f} deg; it should slide"
        )
    for name in ("soap", "notebook"):
        _, tilt = _push(solids[name], across=True)
        assert tilt > 45.0, f"the {name} pushed on its thin side tilted only {tilt:.0f} deg"
    travel, tilt = _push(solids["notebook"], across=False)
    assert tilt < 20.0 and travel > 0.10, (
        f"the notebook pushed along its depth tilted {tilt:.0f} deg, so the pusher "
        "tips things over whatever their shape"
    )


def _solver(model, scene: bool = True, **override) -> None:
    """Give a plain model the solver the props actually run under.

    `jumper.five_foot`'s own options with the objects scene's friction settings
    laid over them (`PROP_CONE`, `PROP_IMPRATIO`), applied through mjlab's own
    `MujocoCfg.apply` -- so these tests measure the contact a prop meets in replay,
    and not MuJoCo's defaults (100 iterations) or the cone the task trains on.
    `scene=False` is the task's alone. The behaviour of torsion and rolling depends
    on the cone, impratio and iteration count together.
    """
    import dataclasses

    import tasks
    from tasks.jumper.five_foot.objects import PROP_CONE, PROP_IMPRATIO

    cfg = tasks.load_env_cfg("jumper.five_foot", play=True).sim.mujoco
    if scene:
        cfg = dataclasses.replace(cfg, cone=PROP_CONE, impratio=PROP_IMPRATIO)
    dataclasses.replace(cfg, **override).apply(model)


def test_rolling_friction_decelerates_a_rolling_cylinder_by_what_it_says() -> None:
    """The rolling coefficient is a length, and it has to mean one.

    MuJoCo caps a contact's rolling torque at `rolling * N`, which for a solid
    cylinder rolling without slipping is a deceleration of `rolling * g / 1.5R`.
    `tasks/jumper/five_foot/objects.py::PROP_FRICTION` derives its rolling friction from a
    rolling-resistance coefficient on exactly that basis, so what is pinned here
    is the arithmetic holding in the solver the scene uses: measured 1.01 of the
    prediction under its elliptic cone, and 0.96 under the task's pyramidal one.

    The control is the same can at condim 4, where MuJoCo never evaluates
    the rolling coefficient at all -- it reads correctly in the config and the
    can rolls on as if on ice. Without that the deceleration above could be
    sliding friction or damping standing in for rolling.
    """
    import math

    import mujoco
    import numpy as np

    from tasks.jumper.five_foot.objects import PROP_FRICTION, SOLIDS, _solid_spec

    can = next(s for s in SOLIDS if s.name == "can")
    radius = can.size[0]

    def deceleration(condim: int | None = None) -> float:
        spec = mujoco.MjSpec()
        spec.compiler.degree = False
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.01])
        child = _solid_spec(can)
        if condim is not None:
            child.body(can.name).geoms[0].condim = condim
        frame = spec.worldbody.add_frame(pos=[0, 0, radius], euler=[math.pi / 2, 0, 0])
        spec.attach(child, prefix="prop/", frame=frame)
        model = spec.compile()
        _solver(model)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Rolling along +x about world +y, which after the quarter turn about x is
        # the body's -z -- and a free joint's angular velocity is in the body frame.
        data.qvel[0] = 0.5
        data.qvel[5] = -0.5 / radius
        times, speeds = [], []
        for _ in range(round(3.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            if data.time > 0.2:
                times.append(data.time)
                speeds.append(float(np.hypot(data.qvel[0], data.qvel[1])))
        return -float(np.polyfit(times, speeds, 1)[0])

    want = PROP_FRICTION[2] * 9.81 / (1.5 * radius)
    got = deceleration()
    assert 0.8 * want < got < 1.1 * want, (
        f"the can decelerates at {got:.4f} m/s^2 where its rolling friction "
        f"predicts {want:.4f}"
    )
    assert deceleration(condim=4) < 0.1 * want, (
        "at condim 4 the can slowed down too, so it is not the rolling "
        "friction slowing it above"
    )


def test_a_pinch_holds_its_twist_under_the_scene_s_friction_and_not_the_task_s() -> None:
    """Pins why the objects scene runs its own friction settings, at one contact.

    The soap pinched at 10 N a side between two point pads, the pair taking the
    jaw's torsion, is twisted at 2 mN*m: a fiftieth of what the coefficients allow.
    Under the task's pyramidal cone at impratio 1 torsion has no static phase and
    the soap turns anyway -- measured 0.96 rad/s -- where a Coulomb contact would
    not move at all. Under the objects scene's elliptic cone at impratio 50 it
    measured 0.0097, a hundredth. Nothing raises either way; what differs is whether
    an object pivots in a grip that should hold it.

    The bound is a fiftieth, and tight on purpose: the pyramidal cone at impratio
    50 turns at 0.040 and the elliptic cone at 10 at 0.048, and the second of those
    dropped carried objects (`tasks/jumper/five_foot/objects.py::PROP_CONE`). So a scene that gives
    up either half of its settings fails here. Written first with a twentieth, this
    passed with the scene's cone set back to pyramidal.

    The task's creep is the control: without it, "slow under the scene's settings"
    could be a pad that never touches or a twist that never arrives.
    """
    import mujoco
    import numpy as np

    from tasks.jumper.five_foot.jaws import GRIP_FRICTION
    from tasks.jumper.five_foot.objects import SOLIDS, _solid_spec

    soap = next(s for s in SOLIDS if s.name == "soap")
    squeeze, twist = 10.0, 0.002

    def spin_rate(scene: bool) -> float:
        spec = mujoco.MjSpec()
        spec.compiler.degree = False
        spec.attach(_solid_spec(soap), prefix="prop/", frame=spec.worldbody.add_frame())
        for side in (-1, 1):
            pad = spec.worldbody.add_body(
                name=f"pad{side}", pos=[0.0, side * (soap.half_extents[1] + 0.0055), 0.0]
            )
            pad.add_joint(name=f"slide{side}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
            pad.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0.0, 0.0], mass=0.05,
                         priority=1, condim=3, friction=list(GRIP_FRICTION))
        model = spec.compile()
        _solver(model, scene=scene)
        model.opt.gravity[:] = 0.0
        data = mujoco.MjData(model)
        slides = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"slide{k}")]
                  for k in (-1, 1)]
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"prop/{soap.name}")
        rates = []
        velocity = np.zeros(6)
        for k in range(300):
            data.qfrc_applied[slides[0]], data.qfrc_applied[slides[1]] = squeeze, -squeeze
            if k >= 100:
                data.xfrc_applied[body, 4] = twist
            mujoco.mj_step(model, data)
            if k >= 200:
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
                rates.append(abs(velocity[1]))
        return float(np.mean(rates))

    limit = 2.0 * GRIP_FRICTION[1] * squeeze
    assert twist < 0.05 * limit, "the twist is not small against the torsion limit"
    creeping = spin_rate(scene=False)
    assert creeping > 0.2, (
        f"a pinch twisted at {twist * 1000:.0f} mN*m against a {limit * 1000:.0f} mN*m "
        f"limit turns at only {creeping:.3f} rad/s under the task's cone; torsion "
        "holds there now, and tasks/jumper/five_foot/objects.py::PROP_FRICTION says it does not"
    )
    held = spin_rate(scene=True)
    assert held < creeping / 50.0, (
        f"under the objects scene's friction settings the pinch turns at {held:.3f} "
        f"rad/s against {creeping:.3f} under the task's; the scene's settings no "
        "longer hold a grip the way tasks/jumper/five_foot/objects.py::PROP_CONE measured"
    )


def test_a_soap_bar_s_twist_on_the_floor_is_held_by_its_corners_not_its_torsion() -> None:
    """Why the prop's own torsional friction can be a hard corner's 0.0002.

    A bar resting on the floor touches it at its corners, and the solver resists
    a twist with sliding friction at each of them on a lever to the corner -- for
    the soap, 28.7 mm and 23.8 mN*m. The per-point torsional coefficient adds
    `torsional * N` on top, which at MuJoCo's default 0.005 would be 12 mN*m more
    and at 0.0002 is nothing. Measured under the scene's solver, the soap holds
    20 mN*m under either coefficient, and breaks away by 24 with 0.0002 and by 30
    with 0.005. So the coefficient that describes a corner's contact patch changes
    nothing a face does short of the corners' own limit, which is all this pins;
    the twist here is 12.

    The control is a twist well past that limit, which has to break the soap away
    -- otherwise "the same under both coefficients" could be two bars that never
    turn at all.
    """
    import mujoco
    import numpy as np

    from tasks.jumper.five_foot.objects import PROP_FRICTION, SOLIDS, _solid_spec

    soap = next(s for s in SOLIDS if s.name == "soap")

    def spin_rate(torsional: float, twist: float) -> float:
        spec = mujoco.MjSpec()
        spec.compiler.degree = False
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0.0, 0.0, 0.01])
        child = _solid_spec(soap)
        geom = child.body(soap.name).geoms[0]
        geom.friction = [PROP_FRICTION[0], torsional, PROP_FRICTION[2]]
        frame = spec.worldbody.add_frame(pos=[0, 0, soap.half_extents[2]])
        spec.attach(child, prefix="prop/", frame=frame)
        model = spec.compile()
        _solver(model)
        data = mujoco.MjData(model)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"prop/{soap.name}")
        rates = []
        velocity = np.zeros(6)
        for k in range(300):
            if k >= 100:
                data.xfrc_applied[body, 5] = twist
            mujoco.mj_step(model, data)
            if k >= 200:
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
                rates.append(abs(velocity[2]))
        return float(np.mean(rates))

    held = [spin_rate(t, 0.012) for t in (PROP_FRICTION[1], 0.005)]
    assert max(held) < 0.1, f"the soap turns at {held} rad/s under 12 mN*m"
    assert abs(held[0] - held[1]) < 0.02, (
        f"torsion {PROP_FRICTION[1]} and 0.005 give {held[0]:.3f} and {held[1]:.3f} "
        "rad/s, so the per-point coefficient is doing what the corners should"
    )
    assert spin_rate(PROP_FRICTION[1], 0.030) > 5.0, (
        "a 30 mN*m twist, past the corners' 23 mN*m, does not turn the soap"
    )


def test_the_props_are_authored_in_radians() -> None:
    """Every prop spec, including the ones that rotate nothing today -- the next
    one to add a turned geom would otherwise find out from the picture."""
    from functools import partial

    from tasks.jumper.five_foot import objects as _objects

    builders = [partial(_objects._solid_spec, s) for s in _objects.SOLIDS]
    for build in [*builders, _objects._bin_spec]:
        spec = build()
        assert spec.compiler.degree is False, (
            f"{build} builds a spec in degrees; any euler it writes in radians is "
            f"silently reduced by 57x"
        )


def test_no_prop_sits_where_the_robot_spawns() -> None:
    """A prop inside the robot's own footprint starts the episode inside it.

    Props collide, so an overlap is not cosmetic: the solver resolves it on the
    first step and the object leaves at speed, which reads as a random object
    flying across the floor rather than as a placement mistake. The row's first
    version started at 0.45 m against a +-0.5 m spawn box and put two objects
    inside it.

    `SPAWN_CLEAR` is small now (0.35 m, the robot's measured reach) **because the
    spawn is pinned** -- see the test below, which is what keeps this one honest.

    Checked against the declared positions rather than by simulating, so it is
    caught while reading the file.
    """
    import math

    from tasks.jumper.five_foot.objects import GRIP_X, SPAWN_CLEAR, _row

    for name, cfg in _row().items():
        x, y, _ = cfg.init_state.pos
        assert math.hypot(x, y) >= SPAWN_CLEAR, (
            f"prop {name!r} sits {math.hypot(x, y):.2f} m from the spawn, inside "
            f"the {SPAWN_CLEAR} m the robot itself occupies"
        )

    # The control: the bound is not vacuous. The row is laid out from `GRIP_X`,
    # and laid out on top of the robot it fails the very same loop -- so the
    # assertion above is measuring the layout rather than agreeing with it.
    inside = [
        name for name, cfg in _row(grip_x=0.1).items()
        if math.hypot(*cfg.init_state.pos[:2]) < SPAWN_CLEAR
    ]
    assert inside, "a row laid out at the spawn passed the clearance check"
    assert GRIP_X > 0.1


def _near_face(spec_fn) -> float:
    """How far a prop's collision reaches toward the robot, along its own -x.

    Re-derived from the compiled model rather than read off `Solid.half_extents`,
    because the layout is computed from that and a test that asks the same
    number twice agrees with itself.
    """
    import mujoco
    import numpy as np

    model = spec_fn().compile()
    best = -1e9
    for g in range(model.ngeom):
        centre, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        for corner in np.ndindex(2, 2, 2):
            local = centre + half * (np.array(corner) * 2.0 - 1.0)
            best = max(best, float(-(model.geom_pos[g] + rot.reshape(3, 3) @ local)[0]))
    return best


def test_every_object_presents_its_grip_point_at_the_same_x() -> None:
    """The row is one walk-up and then strafing, which is what that buys.

    Each object's centre is `GRIP_X + (its depth) - GRIP_IN`, so the point the
    mouth has to arrive at lands on `GRIP_X` for all of them however deep each
    one is: a 66 mm bar of soap and a 105 mm notebook stand at different x so that their
    near faces do not.

    The control is that the depths really do differ. If every object were as deep
    as every other, a layout that ignored depth altogether would pass the loop.
    """
    from functools import partial

    from tasks.jumper.five_foot.objects import GRIP_IN, GRIP_X, SOLIDS, _row, _solid_spec

    row = _row()
    depth = {}
    for solid in SOLIDS:
        depth[solid.name] = _near_face(partial(_solid_spec, solid))
        grip = row[solid.name].init_state.pos[0] - depth[solid.name] + GRIP_IN
        assert abs(grip - GRIP_X) < 1e-9, (
            f"the mouth would have to reach x={grip:.4f} for the {solid.name}, not "
            f"{GRIP_X}"
        )
    assert max(depth.values()) - min(depth.values()) > 0.02, (
        "every object is the same depth, so the loop above cannot tell a layout "
        "that accounts for depth from one that does not"
    )


def test_the_row_is_only_reachable_because_the_spawn_is_pinned() -> None:
    """Objects half a metre away need a robot that starts where it says it does.

    The task drops the robot anywhere within +-0.5 m of its origin facing any
    direction, which is free when the world is empty and is not when it holds
    objects at arm's length: at that spread the row is behind the robot about as
    often as in front of it, and the near ones are inside the box. So `--objects`
    pins the spawn, and the clearance test above is only valid while it does.

    It also starts the robot **still**. Without that it walks off on a sampled
    command before the operator's first keypress -- which is what the objects are
    there for. Both are read off the task's own replay config, built the way
    `play.py --objects` builds it.
    """
    import tasks

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = tasks.load_env_cfg("jumper.five_foot", play=True, task_args={"objects": True})
    pose = cfg.events["reset_base"].params["pose_range"]
    for axis in ("x", "y", "yaw"):
        lo, hi = pose[axis]
        assert lo == hi == 0.0, f"{axis} is still sampled over {pose[axis]}"
    assert pose["z"] == (0.01, 0.05), "the settle drop is not a position"
    assert cfg.commands["twist"].rel_standing_envs == 1.0
    assert cfg.commands["body_pose"].rel_neutral_envs == 1.0
    # Standing still is not the same as being steered: `rel_heading_envs` names a
    # fraction too, and setting that one to 1.0 would turn every environment over
    # to the heading controller.
    assert cfg.commands["twist"].rel_heading_envs == 0.3

    # The control: the same replay without the row leaves all of it alone.
    other = tasks.load_env_cfg("jumper.five_foot", play=True)
    assert other.events["reset_base"].params["pose_range"]["x"] == (-0.5, 0.5)
    assert other.commands["twist"].rel_standing_envs == 0.1


def test_a_still_start_needs_a_command_that_can_be_stopped() -> None:
    """A command term nobody can stop walks the robot away from the row.

    The two field names are written down in `objects.py::_STILL_FIELDS` rather
    than matched by a pattern, so a renamed field would otherwise be a row the
    robot quietly stops standing still in front of.
    """
    import tasks
    from tasks.jumper.five_foot.objects import apply_objects

    cfg = tasks.load_env_cfg("jumper.five_foot", play=True)
    cfg.commands = {"twist": type("T", (), {})()}
    with pytest.raises(ValueError, match="stand still"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        apply_objects(cfg)

    # And a config with no reset event at all cannot be placed either.
    cfg = tasks.load_env_cfg("jumper.five_foot", play=True)
    del cfg.events["reset_base"]
    with pytest.raises(ValueError, match="reset_base"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        apply_objects(cfg)


def test_every_prop_rests_on_the_floor() -> None:
    """Each object starts at its own half-height, standing rather than buried.

    A free body's `init_state.pos` is its **origin**, so the resting height of a
    thing on flat ground is half its own height. Get it wrong in one direction
    and the object starts inside the floor and is ejected on the first step; in
    the other it falls and bounces away. Both leave a scene that loads, renders
    and does not look like what it says it does.

    The bin is the exception and is the control: it is authored sitting on the
    floor at z = 0, so a test that simply demanded a positive height would be
    testing its own assumption rather than the layout.
    """
    from tasks.jumper.five_foot.objects import _row

    row = _row()
    assert row["dropbin"].init_state.pos[2] == 0.0, "the bin is not on the floor"
    for name, cfg in row.items():
        if name == "dropbin":
            continue
        z = cfg.init_state.pos[2]
        assert 0.0 < z < 0.15, (
            f"prop {name!r} starts at z={z}, which is either inside the floor or "
            "dropped from a height"
        )


def test_every_graspable_prop_is_a_prop_that_can_be_carried() -> None:
    """Pins the two ways a grasp list goes wrong without raising.

    **Listing the furniture.** `tasks/jumper/five_foot/tools/grasp_objects.py` and `play --hold`
    go through this list, so a fixed body on it is a check reporting a claw that
    cannot lift a bookcase, or a `--hold` parking one in the mouth. Everything
    graspable has to be a free body, which for a prop means it declares a
    freejoint.

    **A list that drifts from the props.** A graspable entry with no prop behind it
    fails the check by name, loudly, which is the good case. The quiet one is the
    reverse: a prop added to the row and left off the list is never checked, and
    nobody finds out the claw cannot hold it.
    """
    from tasks.jumper.five_foot.objects import GRASPABLE, _row

    props = _row()
    assert GRASPABLE, "the row has nothing the claw can pick up"
    missing = [n for n in GRASPABLE if n not in props]
    assert not missing, f"GRASPABLE names no such prop: {missing}"

    for name in GRASPABLE:
        spec = props[name].spec_fn()
        body = next(b for b in spec.bodies if b.name == name)
        assert body.joints, (
            f"{name} is graspable but has no joint; a fixed body cannot be carried, "
            "so the grasp check would blame the claw"
        )

    # Control group: the room does contain fixed things, so the assertion above
    # is about this list rather than about props in general.
    fixed = [n for n in props if n not in GRASPABLE]
    assert fixed, "every prop is graspable, so the check above proves nothing"


def test_the_row_brings_the_friction_cone_its_props_are_held_under() -> None:
    """Pins the failure that drops everything the claw picks up and raises nothing.

    The claw holds a prop by friction, and only under the row's elliptic cone at
    impratio 50 (`objects.py::PROP_CONE`): under the task's pyramidal one a squeezed
    object creeps out of the jaws. `--objects` asks for it, `apply_objects` writes
    it into the config's `sim.mujoco`, and `MujocoCfg.apply` puts it on the model.
    A break at any of the three builds, runs and shows a claw that cannot hold.

    Two controls. The task on its own is on the pyramidal cone, so the assertion is
    about the row and not a default that happens to match; and a replay without the
    row keeps the task's cone -- `--objects` is not written into every replay, which
    would put all of them on friction their policy never saw.
    """
    import warnings

    import mujoco

    import tasks
    from tasks.jumper.five_foot.objects import PROP_CONE, PROP_IMPRATIO

    plain = tasks.load_env_cfg("jumper.five_foot", play=True)
    assert plain.sim.mujoco.cone == "pyramidal", "the task changed cone; the control is gone"
    assert (PROP_CONE, PROP_IMPRATIO) != (plain.sim.mujoco.cone, plain.sim.mujoco.impratio)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = tasks.load_env_cfg("jumper.five_foot", play=True, task_args={"objects": True})
    assert (cfg.sim.mujoco.cone, cfg.sim.mujoco.impratio) == (PROP_CONE, PROP_IMPRATIO), (
        "--objects asks for its friction cone and the config does not have it"
    )
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    cfg.sim.mujoco.apply(model)
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.impratio == PROP_IMPRATIO

    again = tasks.load_env_cfg("jumper.five_foot", play=True)
    assert (again.sim.mujoco.cone, again.sim.mujoco.impratio) == ("pyramidal", 1.0), (
        "a replay without --objects is on the row's friction cone"
    )
