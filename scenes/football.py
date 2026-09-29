"""A football pitch at one fifth scale, in bright afternoon light.

A real pitch is 105 by 68 metres and a hexapod is 0.3 m across; at full size the
robot is a speck and the markings are a horizon. One fifth -- **21 by 13.6 m** --
keeps every proportion and puts the whole pitch in frame with the robot still
legible. Every dimension below is a real one divided by five, which is why they
are written as the division rather than as the result: `105 / 5` says where it
came from and `21.0` does not.

The markings and the goals are geometry, not texture, and they are added through
`Scene.decorate`. Texture cannot do it: the ground is an infinite plane with a
*tiling* material, so a centre circle drawn into the image would repeat every few
metres in every direction. The mown stripes **are** texture, because stripes are
the one thing on a pitch that genuinely does repeat.

**Everything added here is visual** -- `contype=0, conaffinity=0`. A goalpost the
robot can walk into would change the task, silently, for every task the scene is
selected on. Making them solid is a decision for a task that wants a pitch, not
for a look.

The perimeter boards are the one exception, and they keep the rule where it
matters: they are solid **to the ball only**, on a collision channel the two
share, so a robot walks through them as if they were not there and no task
changes. The reasoning and the bit are at `BALL_BIT`.

One pitch is drawn, at the world origin. Environments are laid out on a grid by
`env_spacing`, so at 4096 environments most of them are well outside it; this
scene is for watching a handful.
"""

from __future__ import annotations

import math
from typing import Any

from pathlib import Path

import numpy as np

from . import hoardings

# Kept as its own line: `tests/test_scenes.py` checks for this import literally,
# to pin that every sky-generating scene shares one face table rather than
# growing its own copy.
from . import skygen
from .registry import GROUND, Headlight, Scene

#: This scene's images. They live inside the package, beside the code that
#: describes them, so a scene is one module plus one directory.
#: This scene's asset directory, **named from this module** rather than written
#: out. The two used to be independent literals, so renaming the module left the
#: path pointing at a directory that no longer existed -- and a scene whose
#: textures silently fail to bind renders unchanged with no error, which
#: `rough.py`'s docstring already records happening twice.
_ASSET = Path(__file__).resolve().parent / "assets" / Path(__file__).stem

#: A real pitch, divided by five.
LENGTH = 105.0 / 5
WIDTH = 68.0 / 5
LINE = 0.12 / 5          # markings are 12 cm on a real pitch
CIRCLE_R = 9.15 / 5
PENALTY_D = 16.5 / 5     # depth of the penalty area from the goal line
PENALTY_W = 40.32 / 5
GOAL_AREA_D = 5.5 / 5
GOAL_AREA_W = 18.32 / 5
PENALTY_SPOT = 11.0 / 5
GOAL_W = 7.32 / 5
GOAL_H = 2.44 / 5

#: Markings sit just above the plane. Coplanar would z-fight, and the flicker is
#: the kind of thing that looks like a driver problem rather than a modelling one.
#: The turf is a slab between the two, so the order from the ground up is
#: plane, turf, paint.
TURF_Z = 0.002
PAINT_Z = 0.004

_WHITE = (0.93, 0.94, 0.93, 1.0)

# ── The perimeter boards ────────────────────────────────────────────────
#
# These are the **one solid thing** in a scene whose rule is that everything is
# visual, and the exception is deliberate rather than an oversight: a board that
# does not collide does not keep the ball in, which is the whole of its job.
#
# What keeps the rule intact is *who* they are solid to. The boards and the ball
# talk on a collision channel of their own, so the ball bounces off them and the
# robot passes straight through. MuJoCo's pair test is
# `(contype1 & conaffinity2) || (contype2 & conaffinity1)`, so with
#
#     board   contype = BALL_BIT              conaffinity = BALL_BIT
#     ball    contype = 1 | BALL_BIT          conaffinity = 1 | BALL_BIT
#
# board-vs-ball is `BALL_BIT & (1|BALL_BIT)` -> collides, while board-vs-anything
# that does not carry the bit is 0 both ways -> no contact at all, not even a
# solver row. The ball keeps bit 0 so it still rests on the ground and can still
# be kicked by a robot, whose own geoms carry it.
#
# **`BALL_BIT` must not be a bit any robot uses.** Robots take the low ones:
# mjlab's convention is bit 0 for terrain, and `tasks/jumper/common/constants.py`
# allocates bits 1-6 to the six legs for self-collision. Bit 10 is clear of that
# with room to spare, and a robot that ever wanted it would find this comment by
# grepping for the number.
#
# To make the boards solid to everything as well -- a decision for a task, not for
# a look, exactly as the module docstring says of the goalposts -- give them
# `contype=1 | BALL_BIT, conaffinity=1 | BALL_BIT`. Note what that would mean at
# scale: one pitch is drawn at the world origin while environments are spread on a
# grid, so a wall there is a wall through other environments' robots.
BALL_BIT = 1 << 10

#: Clear distance from the touchline and goal line to the inner face of the
#: boards. Unlike every other dimension here this is **not** a real measurement
#: divided by five -- it was specified directly, so it is written as itself.
BOARD_GAP = 0.5

#: Board height and thickness. Pitch-side hoardings are about a metre tall and a
#: quarter of that would be a heavy one, both at a fifth like the rest.
BOARD_H = 1.0 / 5
BOARD_T = 0.25 / 5

#: The bare board, under the hoarding texture. Grey-white, and it has to be close
#: to `hoardings.BOARD_RGB` or the two disagree wherever the texture does not
#: reach -- the top rail and the board's own edges and back faces, which are
#: untextured geometry showing this colour directly.
#:
#: **The material multiplies the texture by this rgba**, it does not replace it,
#: so leaving the old (0.16, 0.19, 0.24) here would have darkened every advert to
#: a fifth of its brightness and looked like a lighting bug rather than a colour
#: that was never changed.
_BOARD_RGBA = (*hoardings.BOARD_RGB, 1.0)


def _line(spec: Any, name: str, centre: tuple[float, float],
          length: float, along_x: bool) -> None:
    half = (length / 2, LINE / 2) if along_x else (LINE / 2, length / 2)
    spec.worldbody.add_geom(
        name=name, type=6,  # mjGEOM_BOX
        size=[half[0], half[1], PAINT_Z / 2],
        pos=[centre[0], centre[1], PAINT_Z / 2],
        rgba=list(_WHITE), contype=0, conaffinity=0, mass=0.0, group=2,
    )


def _arc(spec: Any, name: str, centre: tuple[float, float], radius: float,
         segments: int = 72) -> None:
    """A circle as tangential segments. MuJoCo has no ring primitive, and a
    torus made of a mesh would be a file to keep in step with these numbers."""
    step = 2 * math.pi / segments
    chord = 2 * radius * math.sin(step / 2)
    for i in range(segments):
        theta = i * step
        half = theta / 2 + math.pi / 4  # tangent is theta + 90 degrees
        spec.worldbody.add_geom(
            name=f"{name}_{i}", type=6,
            size=[chord / 2, LINE / 2, PAINT_Z / 2],
            pos=[centre[0] + radius * math.cos(theta),
                 centre[1] + radius * math.sin(theta), PAINT_Z / 2],
            quat=[math.cos(half), 0.0, 0.0, math.sin(half)],
            rgba=list(_WHITE), contype=0, conaffinity=0, mass=0.0, group=2,
        )


def _goal(spec: Any, name: str, x: float) -> None:
    """Two posts and a crossbar. Square section rather than round: at 2.4 cm a
    cylinder is a handful of facets and costs more than it shows."""
    post = LINE
    for side, y in (("l", GOAL_W / 2), ("r", -GOAL_W / 2)):
        spec.worldbody.add_geom(
            name=f"{name}_post_{side}", type=6,
            size=[post / 2, post / 2, GOAL_H / 2],
            pos=[x, y, GOAL_H / 2],
            rgba=list(_WHITE), contype=0, conaffinity=0, mass=0.0, group=2,
        )
    spec.worldbody.add_geom(
        name=f"{name}_bar", type=6,
        size=[post / 2, GOAL_W / 2 + post / 2, post / 2],
        pos=[x, 0.0, GOAL_H + post / 2],
        rgba=list(_WHITE), contype=0, conaffinity=0, mass=0.0, group=2,
    )


def _boards(spec: Any) -> None:
    """A closed rectangle of boards, `BOARD_GAP` clear of the lines.

    Four boxes rather than a single hollow shape, because MuJoCo has no such
    primitive and four boxes is what a hollow shape would compile to anyway.

    The corners are made to meet exactly rather than approximately: the long
    boards span the **outer** length (`inner_x + BOARD_T`) and the short ones span
    only the inner width, so each short board butts against the inside face of the
    long ones. Overlapping them instead would put two solid boxes in the same
    space at every corner, which MuJoCo tolerates but which shows through the
    faces; leaving a gap would let the ball escape at exactly the place it is most
    likely to end up.
    """
    inner_x = LENGTH / 2 + BOARD_GAP
    inner_y = WIDTH / 2 + BOARD_GAP
    half_t = BOARD_T / 2

    for name, centre, half in (
        ("board_n", (0.0, inner_y + half_t), (inner_x + BOARD_T, half_t)),
        ("board_s", (0.0, -inner_y - half_t), (inner_x + BOARD_T, half_t)),
        ("board_e", (inner_x + half_t, 0.0), (half_t, inner_y)),
        ("board_w", (-inner_x - half_t, 0.0), (half_t, inner_y)),
    ):
        spec.worldbody.add_geom(
            name=name, type=6,  # mjGEOM_BOX
            size=[half[0], half[1], BOARD_H / 2],
            pos=[centre[0], centre[1], BOARD_H / 2],
            # **Bound here, not by `geom_names_expr`.** These geoms are created by
            # `decorate`, which runs *after* the scene's materials have been
            # applied, so a name pattern in the MaterialCfg matches nothing --
            # and matches nothing silently: the material is created, binds to no
            # geom, and the boards render in flat `rgba` with no error anywhere.
            # Measured exactly that way first (geom_matid was None on all four).
            # The turf slab above has the same problem and the same answer.
            material="pitch_boards",
            rgba=list(_BOARD_RGBA),
            contype=BALL_BIT, conaffinity=BALL_BIT,
            mass=0.0, group=2,
        )


def _draw_pitch(spec: Any) -> None:
    # mjlab's template hazes the distance towards a dark blue-grey, which is
    # right for dusk and wrong here: on a bright day the far end of the pitch
    # goes *paler*, towards the sky, not darker. Same mechanism as the
    # headlight -- a visual default that belongs to the scene rather than to the
    # framework, and equally invisible until you look for it.
    spec.visual.rgba.haze = [0.78, 0.87, 0.95, 1.0]

    half_l, half_w = LENGTH / 2, WIDTH / 2

    # The turf: a finite slab of grass laid on the sand, so the pitch stops at
    # the touchline. The ground plane itself is infinite and cannot stop, which
    # is why this is geometry and not a second material on it. Visual only --
    # the robot walks on the plane underneath, and its physics is unchanged.
    # Extended by half a line so the touchlines sit on grass rather than
    # straddling the edge.
    spec.worldbody.add_geom(
        name="turf", type=6,
        size=[half_l + LINE / 2, half_w + LINE / 2, TURF_Z / 2],
        pos=[0.0, 0.0, TURF_Z / 2],
        material="pitch_grass",
        contype=0, conaffinity=0, mass=0.0, group=2,
    )

    # Touchlines and goal lines.
    _line(spec, "touch_n", (0.0, half_w), LENGTH, True)
    _line(spec, "touch_s", (0.0, -half_w), LENGTH, True)
    _line(spec, "goalline_w", (-half_l, 0.0), WIDTH, False)
    _line(spec, "goalline_e", (half_l, 0.0), WIDTH, False)

    # Halfway line and the centre.
    _line(spec, "halfway", (0.0, 0.0), WIDTH, False)
    _arc(spec, "centre_circle", (0.0, 0.0), CIRCLE_R)
    _line(spec, "centre_spot", (0.0, 0.0), LINE * 2, True)

    for end, sign in (("w", -1.0), ("e", 1.0)):
        goal_x = sign * half_l
        # Penalty area: the far edge and the two returning sides.
        _line(spec, f"pen_{end}_edge", (goal_x - sign * PENALTY_D, 0.0), PENALTY_W, False)
        for side, y in (("n", PENALTY_W / 2), ("s", -PENALTY_W / 2)):
            _line(spec, f"pen_{end}_{side}", (goal_x - sign * PENALTY_D / 2, y),
                  PENALTY_D, True)
        # Goal area, the same shape inside it.
        _line(spec, f"box_{end}_edge", (goal_x - sign * GOAL_AREA_D, 0.0),
              GOAL_AREA_W, False)
        for side, y in (("n", GOAL_AREA_W / 2), ("s", -GOAL_AREA_W / 2)):
            _line(spec, f"box_{end}_{side}", (goal_x - sign * GOAL_AREA_D / 2, y),
                  GOAL_AREA_D, True)
        _line(spec, f"pen_spot_{end}", (goal_x - sign * PENALTY_SPOT, 0.0), LINE * 2, True)
        _goal(spec, f"goal_{end}", goal_x)

    _boards(spec)


# ── The sky ─────────────────────────────────────────────────────────────
#
# See scenes/beach.py for the same note: colours here, PNGs under
# assets/, re-render with `python tools/make_skybox.py football`.


#: Direction **towards** the sun: high and to one side, so shadows are short but
#: not straight down. Straight down is the one lighting that tells you nothing
#: about shape.
SUN = np.array([-0.4056, 0.4056, 0.8192])   # azimuth 135, elevation 55

ZENITH = np.array([0.17, 0.38, 0.78])
HORIZON = np.array([0.72, 0.85, 0.95])

#: Cumulus: white on top, grey underneath, with a hard-ish edge. Fair weather
#: cloud has a defined outline; softening it turns the sky overcast.
#:
#: Separate puffs rather than a layer is a matter of threshold and squash. A low
#: threshold joins them into sheet, and a high squash flattens them into bands --
#: both read as weather rather than as a nice afternoon. Cutting high on the
#: noise leaves only its peaks, which are islands.
CLOUD_TOP = np.array([1.00, 1.00, 1.00])
CLOUD_BASE = np.array([0.62, 0.66, 0.72])
CLOUD_FREQ = 2.3
CLOUD_THRESHOLD = 0.575
CLOUD_SOFTNESS = 0.085
CLOUD_SQUASH = 1.35
CLOUD_SEED = 17

SUN_RADIUS_DEG = 1.4
SUN_GLOW_DEG = 12.0
SUN_COLOR = np.array([1.0, 0.98, 0.92])


def sky(dirs: np.ndarray) -> np.ndarray:
    sun = SUN / np.linalg.norm(SUN)
    up = skygen.smoothstep(dirs[..., 2:3] / 0.9)
    elev = np.degrees(np.arcsin(np.clip(dirs[..., 2:3], -1.0, 1.0)))
    azim = skygen.angle_to(dirs, sun)

    out = HORIZON + (ZENITH - HORIZON) * up
    # A slight brightening towards the sun, and no more: at this elevation the
    # sky is blue in every direction and an azimuthal gradient would read as a
    # second, smaller sunset.
    out = out + (np.array([0.86, 0.92, 0.98]) - out) * 0.35 * np.exp(-((azim / 55.0) ** 2))

    q = dirs * CLOUD_FREQ
    q = np.stack([q[..., 0], q[..., 1], q[..., 2] * CLOUD_SQUASH], axis=-1)
    density = skygen.fbm(q, CLOUD_SEED)[..., None]
    cover = skygen.smoothstep((density - CLOUD_THRESHOLD) / CLOUD_SOFTNESS)
    cover = cover * skygen.smoothstep(elev / 3.0)
    # Lit from above: the higher in the sky, the more of the top you see.
    lit = CLOUD_BASE + (CLOUD_TOP - CLOUD_BASE) * skygen.smoothstep(up * 1.4)
    out = out + (lit - out) * (0.95 * cover)

    glow = np.exp(-((azim / SUN_GLOW_DEG) ** 2))
    out = out + (SUN_COLOR - out) * (0.55 * glow)
    return out + (SUN_COLOR - out) * skygen.disc(dirs, sun, SUN_RADIUS_DEG)


#: Half a real ball: 11 cm across rather than 22. The pitch is at a fifth and the
#: robot is not scaled at all, so the ball is a compromise between the two -- a
#: full-size football is most of the hexapod's own length and it shoves the robot
#: around rather than the other way about.
#:
#: The mass scales with **area**, not volume: a football is a thin shell, so most
#: of its 430 g is skin. Half the radius is a quarter of the skin, 108 g. Scaling
#: by volume would have given 54 g, which bounces like a beach ball.
BALL_RADIUS = 0.22 / 2 / 2          # half a real ball's diameter
BALL_MASS = 0.43 / 4               # a shell: half the radius is a quarter the skin

#: Rolling resistance. Only has any effect at condim 4 or 6; see `_ball_spec`,
#: which carries the measurements this number comes from.
ROLLING_FRICTION = 0.005

#: Seam half-width in dot-product units; see `ball_texture`.
SEAM_GAP = 0.012


def ball_texture(dirs: np.ndarray) -> np.ndarray:
    """The classic ball, as a function of direction.

    A football is a truncated icosahedron: twelve black pentagons on the vertices
    of an icosahedron and twenty white jumpergons on its faces. That is exactly a
    spherical Voronoi diagram over those 32 points -- whichever of the 32 a
    direction is closest to decides which panel it is in, and the twelve
    vertex-cells come out pentagonal because an icosahedron vertex has five
    neighbours. So there is no mesh and no unwrapping: the pattern is computed
    per texel from the direction, the same way the skies are.

    The stitching is where the two nearest sites are nearly equidistant, which is
    the cell boundary by definition. `SEAM_GAP` is in dot-product units, and the
    conversion matters: adjacent sites are about 40 degrees apart, so a gap `g`
    is a band roughly `g / 0.68` radians wide either side of the boundary. The
    first attempt used 0.045, which is 14 mm of seam on a 220 mm ball -- a real
    one is 2 to 3. 0.012 is about 4 mm, and going much below that puts the seam
    under a texel: one face is 512 texels across 90 degrees.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array(
        [[0, s1 * 1.0, s2 * phi] for s1 in (-1, 1) for s2 in (-1, 1)]
        + [[s1 * 1.0, s2 * phi, 0] for s1 in (-1, 1) for s2 in (-1, 1)]
        + [[s1 * phi, 0, s2 * 1.0] for s1 in (-1, 1) for s2 in (-1, 1)],
        dtype=float,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    # Face centres: every triple of vertices that are mutual nearest neighbours.
    edge = np.min(
        [np.linalg.norm(verts[0] - v) for v in verts[1:]]
    )
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.linalg.norm(verts[i] - verts[j]) > edge * 1.1:
                continue
            for k in range(j + 1, 12):
                if (np.linalg.norm(verts[i] - verts[k]) <= edge * 1.1
                        and np.linalg.norm(verts[j] - verts[k]) <= edge * 1.1):
                    c = verts[i] + verts[j] + verts[k]
                    faces.append(c / np.linalg.norm(c))
    sites = np.concatenate([verts, np.array(faces)], axis=0)
    is_pentagon = np.concatenate([np.ones(12, bool), np.zeros(len(faces), bool)])

    dots = dirs @ sites.T
    order = np.argsort(-dots, axis=-1)
    nearest = order[..., 0]
    gap = np.take_along_axis(dots, order[..., :1], -1)[..., 0] - np.take_along_axis(
        dots, order[..., 1:2], -1
    )[..., 0]

    black = np.array([0.09, 0.09, 0.10])
    white = np.array([0.98, 0.98, 0.97])
    seam = np.array([0.28, 0.28, 0.30])
    out = np.where(is_pentagon[nearest][..., None], black, white)
    return np.where((gap < SEAM_GAP)[..., None], seam, out)


def _ball_spec() -> Any:
    """A football: a sphere on a free joint.

    **`condim=6`, and that is what makes the friction below exist at all.**
    MuJoCo's three friction numbers are sliding, torsional and rolling, but the
    last two are only *evaluated* at `condim` 4 and 6. With `condim=3` -- which
    this had, with a rolling coefficient written next to it -- the ball had
    literally no rolling resistance and coasted until it hit something.

    Measured, kicked along an empty pitch and left to run:

        condim=3, rolling 0.0001     1 m/s -> 8.57 m   2 m/s -> 10.32 m
        condim=6, rolling 0.0001     1 m/s -> 8.50 m   2 m/s -> 10.27 m
        condim=6, rolling 0.005      1 m/s -> 0.51 m   2 m/s ->  1.90 m

    The first two rows are the same to within noise, which is the proof that the
    coefficient was being ignored rather than merely being small. The pitch is
    21 m long, so 8.5 m from a gentle push is most of the way to the far boards.

    **Rolling friction, not mass, is the knob.** At a given launch speed the roll
    distance is *independent* of mass -- measured at 0.107, 0.2, 0.3 and 0.43 kg,
    all four give 8.57 m, because gravity and inertia scale together. Mass only
    changes how fast a given kick launches it. So a ball that "goes too far" is a
    friction problem wearing a mass costume.

    The value is chosen against **the speeds this robot can actually produce**,
    which is the part that is easy to get wrong by tabulating 1, 2 and 3 m/s and
    picking from those. The hexapod walks at about 0.43 m/s, so a nudge sends the
    ball off at well under 1 m/s and the interesting column is the left-hand one:

        rolling   0.3 m/s   0.5 m/s   0.8 m/s   1.0 m/s   2.0 m/s
        0.002      0.38      0.77      1.43      1.89      5.09
        0.005      0.08      0.16      0.34      0.51      1.90
        0.01       0.03      0.07      0.17      0.27      1.06

    0.01 was the first choice and it is too much: a gentle contact moved the ball
    3 cm, which does not read as kicking it, it reads as the ball being stuck.
    0.005 gives a nudge 16 cm and a solid contact half a metre, and still stops a
    2 m/s strike inside two metres. 0.002 is the other way -- a firm kick crosses
    half the pitch again.

    `solref` is where the bounce lives. The default is a nearly inelastic contact,
    which makes a football behave like a beanbag; (0.02, 0.35) gives it something
    back off the turf without turning it into a superball that leaves the pitch on
    the first touch.

    The torsional coefficient starts working at `condim=6` too, which is what stops
    a ball nudged off-centre from spinning on the spot indefinitely.

    The masks carry **bit 0 plus `BALL_BIT`**: bit 0 is what everything else uses,
    so the ball still rests on the ground and can still be kicked, and `BALL_BIT`
    is the private channel it shares with the perimeter boards. Dropping bit 0
    here would leave the ball falling through the world; dropping `BALL_BIT` would
    leave the boards solid to nothing, which looks identical to having no boards
    at all. See the note beside `BALL_BIT`.
    """
    import mujoco

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball", pos=[0.0, 0.0, 0.0])
    body.add_freejoint()
    body.add_geom(
        name="ball",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[BALL_RADIUS, 0.0, 0.0],
        mass=BALL_MASS,
        rgba=[0.97, 0.97, 0.97, 1.0],
        condim=6,
        friction=[0.6, 0.005, ROLLING_FRICTION],
        solref=[0.02, 0.35],
        contype=1 | BALL_BIT,
        conaffinity=1 | BALL_BIT,
        group=2,
    )
    return spec


def _ball() -> Any:
    from mjlab.entity import EntityCfg

    from mjlab.utils import spec_config as sc

    return EntityCfg(
        spec_fn=_ball_spec,
        textures=(sc.TextureCfg(name="ball", type="cube",
                                cubefiles=skygen.cubefiles(_ASSET, prefix="ball")),),
        materials=(sc.MaterialCfg(name="ball", texture="ball",
                                  geom_names_expr=("^ball$",)),),
        init_state=EntityCfg.InitialStateCfg(
            # In front of the robot and to one side, close enough to walk into.
            # Entities are placed per environment, so this is relative to each
            # environment's own origin rather than to the pitch.
            pos=(0.40, 0.0, BALL_RADIUS),
        ),
    )


#: Cube maps this scene generates, by file prefix. Rendered by
#: `python tools/make_skybox.py football`.
CUBEMAPS = {"sky": sky, "ball": ball_texture}


def scene() -> Scene:
    from mjlab.utils import spec_config as sc

    return Scene(
        headlight=Headlight(ambient=(0.34, 0.36, 0.40), diffuse=(0.0, 0.0, 0.0)),
        textures=(
            sc.TextureCfg(name="pitch_grass", type="2d",
                          file=str(_ASSET / "grass.png")),
            sc.TextureCfg(name="pitch_sand", type="2d",
                          file=str(_ASSET / "sand.png")),
            sc.TextureCfg(name="pitch_sky", type="skybox",
                          cubefiles=skygen.cubefiles(_ASSET)),
            sc.TextureCfg(name="pitch_boards", type="2d",
                          file=str(_ASSET / "hoardings.png")),
        ),
        materials=(
            # Sand goes on the infinite plane; grass is bound by name to the
            # turf slab that `decorate` lays on top of it, which is why this one
            # has no `geom_names_expr` -- there is no geom to match yet when the
            # material is created.
            sc.MaterialCfg(
                name="pitch_sand", texture="pitch_sand", texuniform=True,
                texrepeat=(0.35, 0.35), reflectance=0.0, geom_names_expr=GROUND,
            ),
            sc.MaterialCfg(
                name="pitch_grass", texture="pitch_grass", texuniform=True,
                # One repeat of the texture is two mown bands, and a real band is
                # about 5 m, so at a fifth that is a 2 m repeat.
                texrepeat=(0.5, 0.5), reflectance=0.0,
            ),
            # The hoardings, bound to all four boards by name.
            #
            # `texuniform=False` on purpose: uniform mapping scales the texture by
            # world size, which would give the long boards and the short ones
            # different-sized adverts. Off, each face maps the image once per
            # `texrepeat`, so the panels come out the same size on all four.
            #
            # The repeat counts are set from the board lengths so that a panel
            # stays about as wide as it is on a real hoarding rather than being
            # stretched to fit: the long boards are LENGTH + 2*BOARD_GAP across,
            # the short ones WIDTH + 2*BOARD_GAP, and the strip holds six panels.
            sc.MaterialCfg(
                name="pitch_boards", texture="pitch_boards", texuniform=False,
                texrepeat=(2.0, 1.0), reflectance=0.0,
                # No `geom_names_expr`: the boards do not exist yet at this point.
                # `_boards` names this material on each geom instead.
            ),
        ),
        lights=(
            sc.LightCfg(
                name="sun", type="directional", pos=(-6.0, 6.0, 12.0),
                dir=(0.4056, -0.4056, -0.8192), castshadow=True,
                diffuse=(1.05, 1.02, 0.95), specular=(0.25, 0.25, 0.25),
                ambient=(0.0, 0.0, 0.0),
            ),
        ),
        decorate=_draw_pitch,
        props={"ball": _ball()},
    )
