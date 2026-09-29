"""Ground the feet sink into -- sand, mud, thick turf.

**This scene changes the physics, not the look.** Like `rough` it is not a
cosmetic choice: a policy trained here is not comparable to one trained on rigid
ground, and the ground is the one thing in the environment the whole gait is
built against.

## Softness is `solimp`, and the trap is `priority`

MuJoCo mixes two geoms' contact parameters only when their `priority` is equal.
When it differs, **the higher-priority geom's parameters are used outright and
the other's are discarded**. `constants.py` gives the feet `priority=1` so that
their silicone `solref`/`solimp` win against a plane, a box or a heightfield
alike -- which means a soft `solimp` written on the ground would be thrown away
and this scene would render and simulate as ordinary rigid ground, with nothing
raising and nothing to see.

So the terrain is set to `priority=2`. That is the entire mechanism, and it is
worth stating plainly because it inverts what `constants.py` set up on purpose:
here the *ground* decides the contact, and the feet's silicone parameters are the
ones being discarded. That is the right way round for this scene -- a foot
pressing into mud is governed by the mud -- but it means the foot compliance
measured in `constants.py` is not in effect while this scene is applied.

## What the numbers mean

`solimp = (d0, dwidth, width, midpoint, power)` is the contact impedance. `d0` is
the impedance at zero penetration: at 1.0 the constraint is enforced immediately
and the surface is rigid, and lowering it lets the constraint be violated -- which
is penetration, which is what sinking *is*. `width` is the depth over which
impedance climbs from `d0` to `dwidth`, so it sets how far the foot travels
before the ground starts pushing back properly.

`solref = (timeconst, dampratio)` is how fast the contact restores. A long
timeconst is a slow, soft push rather than a sharp one; `dampratio` at 1.0 is
critically damped, which is what stops soft ground behaving like a trampoline.

The measured sink is recorded next to the constants below.

## What this is not

Not a deformable surface. The ground does not keep a footprint, does not
displace, and springs back the instant the foot leaves. What it models is the
*compliance* of soft ground, not its plasticity -- a robot walking here does not
leave tracks and does not get progressively deeper in a rut. Modelling that needs
a heightfield edited at runtime, which is a different and much more expensive
thing.
"""

from __future__ import annotations

from typing import Any

from .registry import GROUND, Headlight, Scene


def _drag_event():
    """The drag as an mjlab event term, running every step."""
    from mjlab.managers.event_manager import EventTermCfg

    return EventTermCfg(
        func=GranularDrag,
        mode="step",
        params={"drag_k": DRAG_K, "max_force": DRAG_MAX_FORCE},
    )

#: Contact impedance for the ground. See the module docstring for the fields.
#:
#: `d0 = 0.5` against the rigid default's 0.9, and `width = 0.03` so the impedance
#: climbs over three centimetres.
#:
#: Measured: foot penetration below the surface with the robot standing at rest
#: on all six feet (29.5 N over six, about 4.9 N each), 400 settling steps.
#:
#:     d0      sink mean   sink max   base height
#:     rigid     +0.6 mm    -3.3 mm     98.6 mm     the default, for reference
#:     0.60      -9.0       -14.1       88.5
#:     0.50     -10.6       -15.7       87.3        <- this
#:     0.25     -13.8       -18.0       84.0
#:     0.10     -14.9       -19.2       82.6
#:
#: It saturates: 0.25 to 0.10 buys only 1.2 mm more, because `width` bounds how
#: far the impedance ramp runs. Going softer than about 0.25 changes little.
#:
#: **0.5 is chosen against the robot's foot lift, not for maximum squish.**
#: Measured on a trained policy, `Metrics/peak_height_mean` is 10.9 mm -- so at
#: this setting the foot sinks 10.6 mm and lifts 10.9 mm, and the scene is right
#: at the edge of walkable. That is deliberate: it is soft enough to see and to
#: change the gait, and not so soft that the feet are buried deeper than they can
#: be lifted.
#:
#: **Untested in training.** Nothing has been trained here. If policies cannot
#: walk on it the first move is up (0.6, less soft), not down; the second is to
#: raise `foot_clearance`'s 0.03 m target so the gait lifts clear of its own
#: footprint.
SOFT_SOLIMP = (0.5, 0.95, 0.03, 0.5, 2.0)

#: Contact restitution: a long time constant and critical damping.
#:
#: 0.05 s is 10x the 0.005 s physics timestep, comfortably inside the "at least
#: 2x timestep or the integrator cannot hold the contact" rule that
#: `constants.py::_CONTACT_SOLREF` records measuring. `dampratio = 1.0` is
#: critical: below it soft ground turns springy and the robot bounces, which
#: looks like soft ground behaving as a trampoline rather than as mud.
SOFT_SOLREF = (0.05, 1.0)

#: The terrain's contact priority.
#:
#: **This is the load-bearing line of the whole scene.** The feet are
#: `priority=1` (`constants.py`), and MuJoCo discards the lower-priority geom's
#: contact parameters entirely rather than blending them. At priority 0 or 1 the
#: constants above would never reach the solver.
SOFT_PRIORITY = 2

#: Sliding, torsional and rolling friction for the ground.
#:
#: **`priority` overrides the whole contact parameter set, not just the softness**
#: -- friction and `condim` come with it. Setting only `solimp` and `solref` here
#: left the terrain's own `(1.0, 0.005, 0.0)` in force and silently discarded the
#: feet's `_FOOT_FRICTION`, and with it every `foot_friction` domain-randomisation
#: sample, which is applied to the foot geoms and would never have reached the
#: solver. Measured in the compiled model before this was added: terrain
#: `prio=2 friction=[1.0, 0.005, 0.0]` winning over the foot's `[1.0, 0.01, 0.01]`.
#:
#: So the ground carries the feet's own numbers, and the effective friction is
#: what it would have been without this scene.
#:
#: **`foot_friction` randomisation is still inert here**, and cannot be fixed by
#: copying a constant: the event writes to the foot geoms, which this scene has
#: outranked. A run on soft ground trains against one fixed friction. If that
#: matters, the randomisation has to move to the terrain geom for this scene.
SOFT_FRICTION = (1.2, 0.01, 0.01)


def _soften(spec: Any) -> None:
    """Give every ground geom the soft contact parameters, and the priority to
    make them count.

    Matches `registry.GROUND` -- the same pattern the scenes' materials bind
    with, so a scene that renders on the ground also softens exactly that ground.
    **Raises if it matches nothing**, which is the difference between this and
    the material binding it borrows from: an unbound material is a cosmetic
    surprise, but silently rigid ground here would be a scene that claims to
    change the physics and does not, and every measurement taken on it would be
    wrong without looking wrong.
    """
    import re

    # **`spec.geoms`, not `spec.worldbody.geoms`.** The terrain geom is not on the
    # worldbody at the point `decorate` runs -- written the obvious way first, the
    # worldbody's geom list was empty and the guard below fired on the very first
    # build. `spec.geoms` is every geom in the spec regardless of which body owns
    # it, which is what the name pattern is meant to be matched against anyway.
    pats = [re.compile(p) for p in GROUND]
    hits = 0
    for geom in spec.geoms:
        name = geom.name or ""
        if not any(p.search(name) for p in pats):
            continue
        geom.solref = list(SOFT_SOLREF)
        geom.solimp = list(SOFT_SOLIMP)
        geom.friction = list(SOFT_FRICTION)
        geom.priority = SOFT_PRIORITY
        hits += 1
    if not hits:
        names = [g.name for g in spec.geoms]
        raise RuntimeError(
            f"scene 'soft' matched no ground geom with {GROUND!r}; worldbody has "
            f"{names}. The scene would have left the ground rigid and said nothing."
        )


class GranularDrag:
    """Resist a sunk foot's sideways motion, in proportion to how deep it is.

    **MuJoCo's contact model cannot express this, which is why it is a force term
    rather than a parameter.** `solimp` governs the normal direction only;
    tangential force is `mu * N`, and `N` at rest is the robot's weight however
    deep the foot is. A plane has no material to displace. So the soft ground
    built above is a soft trampoline -- vertically compliant, and sideways exactly
    as slippery as concrete. Walking on sand is not like that: the resistance
    comes from having to push grains out of the way, and it grows with burial.

        F = -k * depth * v_xy        per foot, applied at the foot body

    `depth` is how far the foot is below the surface, from the task's own
    `foot_height_scan` sensor. `v_xy` is the foot's horizontal velocity.

    **This is a phenomenological model and `DRAG_K` has no measured basis.** Real
    granular intrusion is closer to rate-independent and depth-proportional at
    walking speeds, i.e. Coulomb-like rather than viscous. Viscous was chosen
    anyway, and the reason is numerical rather than physical: a rate-independent
    law needs `sign(v)`, which chatters when a planted foot's velocity hovers
    around zero and would fight the contact solver at every stance. Linear in
    velocity is smooth through zero. The cost is that it under-resists slow
    motion and over-resists fast, which is the wrong way round for sand.

    Everything else in this repository has a number from a real run behind it.
    This one does not, and it should be tuned by watching rather than trusted:
    the observable is that a foot planted deep should be hard to drag, and the
    gait should shorten. If it makes the robot unable to move at all, `DRAG_K` is
    too high; if the gait is unchanged from rigid ground, it is too low.
    """

    def __init__(self, cfg, env) -> None:
        del cfg
        from tasks.jumper.common.constants import FEET

        self._ids = [env.scene["robot"].body_names.index(f) for f in FEET]

    def __call__(self, env, env_ids, drag_k: float = 0.0,
                 max_force: float = 0.0) -> None:
        import torch

        del env_ids  # a step-mode term is called for every environment
        robot = env.scene["robot"]
        # **Depth from the foot's own z, not from `foot_height_scan`.** That
        # sensor raycasts down to the terrain and cannot report a foot *below* the
        # surface: written against it first, `heights` read 0 for a foot sitting
        # 10.6 mm under, so `depth` was identically zero and the whole term was
        # inert -- measured, 3 N of shove moved the robot 14.9 mm with the drag on
        # against 14.5 mm with it off.
        #
        # This assumes the ground is a plane at z = 0, which is what this scene
        # sets (`terrain_type="plane"`). On generated terrain it would be wrong,
        # and silently: every foot below z=0 would be charged drag regardless of
        # where the ground actually is.
        depth = (-robot.data.body_link_pos_w[:, self._ids, 2]).clamp(min=0.0)
        v_xy = robot.data.body_link_lin_vel_w[:, self._ids, :2]
        f_xy = (-drag_k * depth.unsqueeze(-1) * v_xy).clamp(-max_force, max_force)

        # **Written to the foot rows only, via `body_ids`.** Writing a whole
        # zeroed buffer would erase every other external wrench on the robot each
        # step -- mjlab's own `push_robot` among them -- and do it silently, since
        # nothing reports a force that was applied and then overwritten before the
        # solver saw it. Caught in the harness rather than in a run: with the
        # full-buffer write, a 3 N shove and the drag were cancelling each other.
        forces = torch.zeros(
            (env.num_envs, len(self._ids), 3), device=f_xy.device, dtype=f_xy.dtype
        )
        forces[:, :, :2] = f_xy
        robot.write_external_wrench_to_sim(
            forces=forces, torques=torch.zeros_like(forces), body_ids=self._ids
        )


#: Drag coefficient, in N per (metre of depth) per (m/s). **Not measured against
#: anything real** -- see `GranularDrag`.
#:
#: Measured effect: a steady 3 N shove on the base for 1 s, and how far the robot
#: slid.
#:
#:     rigid (no scene)     6.0 mm
#:     soft, DRAG_K = 0    14.5 mm
#:     soft, DRAG_K = 300  13.7 mm     <- this
#:     soft, DRAG_K = 1500 10.0 mm
#:
#: **Two things in that table matter more than the coefficient.**
#:
#: First, soft ground with the drag off is *more* slippery than rigid, not less --
#: 14.5 mm against 6.0. A compliant contact yields sideways as well as downwards,
#: so making the ground soft makes it slide easier. That is the opposite of sand
#: and it is what this term exists to correct.
#:
#: Second, 300 barely corrects it (14.5 -> 13.7, 6%) and even 1500 leaves the
#: ground slipperier than concrete. **If the goal is for sand to feel harder to
#: push through than a hard floor, `DRAG_K` needs to be well above 1500**, and
#: nothing above 1500 has been tried -- neither for its effect nor for whether the
#: solver stays stable. 300 is a deliberately timid starting point, not a tuned
#: value.
DRAG_K = 300.0

#: Per-foot saturation, in newtons. A hard stop rather than a soft one: the drag
#: is only a model, and an unbounded model term is a numerical instability
#: waiting for the first time a foot moves fast while buried.
DRAG_MAX_FORCE = 5.0


def scene() -> Scene:
    """The `default` scene's look, with soft ground under it.

    **The visual block is copied from `scenes/default.py` verbatim** -- same
    headlight, same checker texture and material, same single directional light,
    and no skybox override, so the sky is mjlab's own exactly as it is there.

    That is the point rather than laziness. This scene exists to change one
    variable, and if it also changed the lighting then "the robot walks
    differently here" would have two candidate explanations and no way to
    separate them by looking. Sharing the look makes `--scene soft` against no
    scene at all a one-variable comparison, in the same way `jumper.flat` is a
    one-variable control for two of the gait tasks.

    The consequence to keep in mind: **the two are indistinguishable in a
    screenshot.** A frame from this scene and a frame from `default` differ only
    in how deep the feet are, which at 10.6 mm is visible if you look at the feet
    and invisible if you look at anything else. Label recordings.

    If the copy and `default.py` ever drift, this docstring is the only thing
    that says they were meant to match; nothing enforces it, because importing
    `default.scene()` and mutating it would couple the two files in a way that
    makes a change to `default` silently change this scene's physics comparison
    as well.
    """
    from mjlab.utils import spec_config as sc

    return Scene(
        headlight=Headlight(ambient=(0.30, 0.30, 0.30), diffuse=(0.60, 0.60, 0.60)),
        textures=(
            sc.TextureCfg(
                name="groundplane", type="2d", builtin="checker", mark="edge",
                rgb1=(0.2, 0.3, 0.4), rgb2=(0.1, 0.2, 0.3),
                markrgb=(0.8, 0.8, 0.8), width=300, height=300,
            ),
        ),
        materials=(
            sc.MaterialCfg(
                name="groundplane", texture="groundplane", texuniform=True,
                texrepeat=(4.0, 4.0), reflectance=0.2, geom_names_expr=GROUND,
            ),
        ),
        lights=(sc.LightCfg(name="sun", pos=(0.0, 0.0, 1.5), type="directional"),),
        # `terrain_type` is set even though "plane" is usually already the task's
        # choice: it is what makes `scripts/_cli.py` warn that this scene changes
        # the ground. A scene that alters contact physics silently is exactly what
        # that warning exists for, and `decorate` alone does not trigger it.
        terrain_type="plane",
        decorate=_soften,
        events={"soft_ground_drag": _drag_event()},
    )
