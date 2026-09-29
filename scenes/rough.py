"""Generated rough ground: slopes, stairs and broken surface, at this robot's size.

This is the one scene that changes what the robot walks on rather than how it
looks, and that is not a cosmetic choice -- see `registry.apply`. The tasks in
this repository drop the `terrain_scan` sensor when they set flat ground, so on
rough ground the policy cannot perceive what it is stepping on: training still
runs and still converges, and nothing anywhere reports it. Restore the sensor in
the task's `env_cfg` before reading anything into such a run.

## Two ratios, not one

`ROUGH_TERRAINS_CFG` is sized for the Unitree Go1, and against this hexapod its
defaults are not "difficult", they are geometry the robot cannot interact with at
all: a 0.10 m step is 93% of standing height, a wall rather than a stair. So every
obstacle dimension here is a Go1 number times a ratio.

**There are two ratios, and using one for both directions is what this file got
wrong.** The two machines are differently proportioned:

                          Go1        this robot     ratio
    standing height     0.278 m      0.10704 m      0.385   <- V_SCALE
    fore-aft stance     0.3762 m     0.3310 m       0.880   <- H_SCALE

The hexapod is *long for its height* -- 3.1 stance lengths per standing height
against the Go1's 1.35 -- so a horizontal dimension scaled by the height ratio
comes out 2.3x finer relative to the robot than it was relative to a Go1. That is
not a cosmetic error, because it lands on the one number a statically stable
machine lives by.

## The number a hexapod lives by

A quadruped that meets ground it cannot handle falls over. A hexapod does not: it
keeps its feet and shuffles, so the failure is quiet, and the thing that actually
defeats it is a foothold it cannot **reach**. The reach that runs out first is
downward. Swept over the joint limits with the foot held within 5 cm of where it
stands at HOME, a leg places its foot 0.0785 m *below* that foothold and 0.257 m
above it -- 0.73 of standing height down against 2.4 up.

So the quantity to size terrain by is the height difference between a fore
foothold and an aft one at the same instant, one stance apart. Measured off the
generated mesh at the hardest row, in multiples of the robot's own standing
height:

                            upstream/Go1     this file      this file
                                              before          after
        pyramid_stairs          0.72           1.76            0.90
        hf_pyramid_slope        1.44           1.40            1.40
        random_rough            0.29           0.28            0.33
        wave_terrain            0.45           0.28            0.47

`DIFFICULTY` is 1.25, so 1.25x the upstream column is what "the same fraction of
this robot" means -- except for `random_rough`, which is measured against the foot
lift instead and carries its own `ROUGHNESS_DIFFICULTY` of 1.5. Two columns were
nowhere near where they belonged:

- **The staircase asked 2.4x too much.** 1.76 standing heights is 0.188 m across a
  0.331 m stance, against a downward reach of 0.0785 m. The robot could not place
  the trailing leg at all, and being statically stable it did not fall over to say
  so -- it stood on the pyramid and shuffled. A 0.30 m tread scaled by the *height*
  ratio is 0.113 m, so the stance straddled 2.9 treads where a Go1's straddles
  1.25. Scaled by the stance ratio it is 0.264 m and the count matches.
- **The wave was the easiest column on the board**, easier than `random_rough`,
  which is not what a wave is for. Same cause from the other side: `num_waves` is a
  *count*, so it scales with nothing -- it was left at mjlab's 4 while the tile it
  spans went from 3.02 m to 12 m, and the wavelength quietly went with the tile, to
  2.875 m. At that length the robot walks a gentle hill it cannot tell from a
  slope. It is derived from the tile below rather than written down, which is the
  only form of it that survives the tile changing again.

## What scales by which

- **Vertical obstacle magnitudes** scale by `V_SCALE`: step height, wave amplitude.
- **Horizontal obstacle dimensions** scale by `H_SCALE`: the stair tread, the wave
  wavelength, and the flat platform the robot spawns on -- that one is sized to
  hold the robot, which is a footprint question, not a height question.
- **Layout dimensions do not scale.** Tile size and the border around the grid are
  not about how big the robot is but about how far it travels in an episode. See
  the note on `size`, which is where an earlier version of this got it wrong in the
  other direction.
- **Slope is an angle and does not scale.** A 45-degree hill is 45 degrees for
  anything. What limits a small machine is that its feet must reach across the same
  slope with much shorter legs, so the ceiling is set from capability rather than
  by ratio: 0.4 is about 22 degrees, which a statically stable hexapod should
  manage and a 45-degree face is not.
- **Random roughness is set from the foot lift**, not from either ratio. Continuous
  roughness is unlike a single step: the robot meets it on *every* footfall rather
  than once, so the reference is how high the gait lifts to. Having no Go1 number
  behind it, it is also the one column that can be turned up on its own, and
  `ROUGHNESS_DIFFICULTY` is that dial.

## The ground material, deliberately absent

The generator gives each heightfield tile its own texture, coloured from that
tile's own height data, and that colouring is the cue that makes the relief
readable. A ground material here would have to match `terrain_\\d+` to bind at
all -- the generated geoms are not called `terrain` -- and doing so would paint
over exactly that information. Written the obvious way first, with the same
`GROUND` expression as the flat scenes, it bound to nothing at all: the material
was created, matched no geom, and the ground rendered unchanged with no error
anywhere.

## The light

Low and hard: relief is invisible under a straight-down light, and on rough
ground the shadows are the picture.

**Every ambient term adds, and the sum clips at 1.0.** The first version summed
to 1.57 on a lit surface -- headlight 0.14 + 0.28, sun 0.25 + 0.90 -- and
everything above 1.0 was flattened into white. Measured on a rendered frame:
9.36% of pixels had a channel at 254/255, mean luminance 0.755. The budget now
comes to 0.81 lit and 0.20 in shadow, a 4:1 ratio, and the same frame measures
2.26% and 0.60.

The headlight's *diffuse* is the term to keep small regardless of the total: it
comes from the camera, so it lights exactly what the camera sees head-on and
fills in the shadows that are the reason for the hard sun.

Some saturation remains on the high-relief tiles, and it is not the light. The
generator colours each tile from its own height data and that palette reaches a
full blue channel; a segmentation render attributes 60% of the saturated pixels
to the terrain geoms themselves. It is accepted -- the colouring is what makes
the relief readable, which is the same reason no ground material is set.
"""

from __future__ import annotations

from .registry import Headlight, Scene

#: This robot against the Go1 mjlab's defaults were written for. **Both of this
#: robot's numbers are copies** -- of `constants.STAND_Z`, and of the fore-aft span
#: of the six foot bodies at `constants.HOME` -- so both go stale when the model
#: changes, and both have. Nothing checks them; re-measure them when the model is
#: replaced, at the same time as `constants.STAND_Z` itself.
STAND_Z = 0.10704
STANCE = 0.3310
GO1_STAND_Z = 0.278
GO1_STANCE = 0.3762

#: Vertical obstacle magnitudes -- how tall an obstacle is against how tall the
#: robot is.
V_SCALE = STAND_Z / GO1_STAND_Z  # 0.385

#: Horizontal obstacle dimensions -- how long an obstacle is against how far apart
#: the robot's fore and aft feet are. Not the same ratio as V_SCALE, and the
#: docstring above is what happened when it was treated as the same.
H_SCALE = STANCE / GO1_STANCE  # 0.880

#: How high the gait actually lifts a foot: `foot_target_height`, which every
#: locomotion task sets in its own `env_cfg` (`tasks/jumper/*/env_cfg.py`) because
#: the foot-lift target belongs to the task. All four currently agree on 0.025.
#: It is the reference for continuous roughness, and deliberately **not** the
#: standing height scaled -- see the docstring.
FOOT_LIFT = 0.025

#: Steepest slope, as a gradient rather than a scaled length.
MAX_SLOPE = 0.4

#: A single multiplier on **how big the obstacles are**, on top of the scaling to
#: this robot. 1.0 is "the same fraction of this robot as mjlab's numbers are of a
#: Go1"; above that is deliberately harder than the equivalent.
#:
#: Vertical magnitudes only. Widening a stair tread or a tile would make the
#: terrain *easier*, so multiplying every dimension by one number would push the
#: two directions against each other and change less than it appears to.
#:
#: It does not reach `random_rough`, which has `ROUGHNESS_DIFFICULTY` below.
DIFFICULTY = 1.25

#: `DIFFICULTY` for the broken-surface column alone. That column is the one set
#: from the foot lift rather than from either Go1 ratio -- see the docstring -- so
#: it is the one that can be turned up without reference to how a Go1 was treated,
#: and it has its own dial rather than borrowing `DIFFICULTY` and drifting from it
#: silently. The obstacle columns ask a quarter more than the equivalent; this asks
#: the gait to lift half again as far as it does now.
ROUGHNESS_DIFFICULTY = 1.5

#: One tile. A layout dimension, not an obstacle one -- see the note at `size`.
TILE = 12.0

#: The flat rim mjlab leaves inside each heightfield tile. Named because the wave's
#: wavelength is measured across what is left over.
_HF_BORDER = 0.25

#: The wavelength the wave column is built to. It is an obstacle dimension -- the
#: distance from the crest one foot stands on to the trough the next reaches for --
#: so it scales horizontally. Upstream fits four waves across an 8 m tile less its
#: rim, which is 1.875 m; scaled, 1.650 m.
_WAVE_LENGTH = ((8.0 - 2 * _HF_BORDER) / 4) * H_SCALE

#: The noise floor of the broken-surface column, and also the generator's height
#: quantum (`vertical_scale`), which is why nothing finer than this is expressible.
_NOISE_FLOOR = 0.005


def _terrain():
    from mjlab.terrains.config import (
        box_random_grid,
        hf_pyramid_slope,
        hf_pyramid_slope_inv,
        pyramid_stairs,
        pyramid_stairs_inv,
        random_rough,
        wave_terrain,
    )
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

    return TerrainGeneratorCfg(
        # **Without this, rows are not difficulty levels and the terrain
        # curriculum is a no-op.** The default is False, and `ROUGH_TERRAINS_CFG`
        # leaves it there: every tile then samples its own type *and its own
        # difficulty* independently, so moving a robot between rows moves it to
        # another randomly-hard tile. Measured on the generated mesh, highest
        # point per tile by row:
        #
        #     curriculum=False   row 0: 0.239 m   row 9: 0.043 m
        #     curriculum=True    row 0: 0.000 m   row 9: 0.283 m
        #
        # The first is noise. It also makes `max_init_terrain_level=0` useless --
        # a robot placed on the flattest *row* still gets a full-height staircase,
        # which is exactly what a viewer on one environment showed: spawning on
        # top of a stair pyramid at level 0.
        #
        # In this mode `num_cols` is ignored and the generator uses one column per
        # sub-terrain type, so the grid becomes 10 x 7 rather than 10 x 20.
        curriculum=True,
        # **Boxes go grey; the heightfields keep their own colouring.** The
        # box terrains -- both staircases -- were painted with a ramp of MuJoCo's
        # brand blue by step index, which is where the saturation was: measured on
        # the generated model, box rgba spanned R[0.13, 1.00] and B[0.20, 1.00],
        # and a segmentation render attributed 60% of the frame's clipped pixels
        # to those geoms.
        #
        # `"none"` only touches geoms coloured through `rgba`. The heightfield
        # terrains are textured through a material instead, so all 40 of them keep
        # `color_by_height`'s soft diverging palette -- checked, not assumed: the
        # material count is 40 either way. So this desaturates exactly the terrains
        # that were harsh and leaves the ones that were not.
        #
        # The stairs lose their per-step colour ramp, which was one of two depth
        # cues. The other is the light, and it is the one that survives being
        # looked at from any angle: risers and treads face different ways, so a
        # hard directional sun separates them. That is also why the light here is
        # low and hard rather than overhead.
        color_scheme="none",
        # **Layout dimensions stay at mjlab's**, unlike the obstacle dimensions
        # below. The two answer different questions and only one of them is about
        # the robot's size.
        #
        # A tile has to be big enough that a robot spends its episode on the
        # difficulty it was assigned. Net displacement in a 20 s episode, against
        # the half-tile it has to stay inside:
        #
        #     achieved   displacement   8 m tile   12 m tile
        #     0.42           2.89 m       inside     inside
        #     0.60           4.13 m       OUT        inside
        #     0.80           5.50 m       OUT        inside
        #
        # 0.42 is what the current policy manages and 0.8 is the promotion bar, so
        # a tile has to hold the *upper* end or it stops containing robots exactly
        # as they start succeeding. 8 m held the first row only; 12 m holds all
        # three. Observed in the viewer before it was measured -- robots were
        # walking off their tile.
        #
        # The bias that creates is not symmetric, which is why it matters. Walking
        # into *harder* neighbouring ground is self-correcting: tracking gets
        # worse and the curriculum demotes. Wandering onto *easier* ground is not:
        # the robot tracks well and is promoted on evidence from terrain it was
        # not assigned. The curriculum drifts upward on its own.
        #
        # An earlier version of this scaled the tile so that crossing it took the
        # same *time* as it takes a Go1. That is a real property and the wrong one
        # to preserve -- upstream's 8 m is not "a Go1-sized tile", it is the size
        # at which a Go1 stays put for an episode, and it happens to work here for
        # the same reason.
        size=(TILE, TILE),
        border_width=5.0,
        # **Ten rungs. Five was tried and did not work.** The generator spaces rows
        # as `difficulty = row / (num_rows - 1)`, so the ceiling is difficulty 1.0
        # whatever this number is -- the hardest row is identical either way, and
        # the whole of the difference is how finely the climb to it is divided.
        # That made five look free on paper: half the rungs, half the mesh, and the
        # same terrain at the top. It was run and it was not, so this is back at
        # ten and the attempt is recorded rather than left to be repeated.
        #
        # What five changed is the size of one promotion: a rung of 0.25 of the
        # range rather than 0.111, so a robot that has just earned level n lands
        # that much further past its ability at n+1. That was the predicted cost at
        # the time -- more demotion churn against a shorter climb -- but it is a
        # prediction and not a diagnosis, and what actually went wrong in the run
        # is not written down here because it was not measured.
        #
        # The mesh is the other side of it: ten rows is 10 x 7 = 70 sub-terrains and
        # 120 m in the row axis, twice what five cost. That is the price of the
        # finer ladder and it is worth paying.
        #
        # Read from `NUM_ROWS` rather than written here, because `cli_args` quotes
        # the range in `--terrain-row`'s help and the two were separate literals
        # that had already been 10 and then 5.
        num_rows=NUM_ROWS,
        num_cols=20,
        sub_terrains={
            # Flat ground, but paved rather than cast in one piece. `flat()` is a
            # single 12 m box, and a 12 m box is the worst thing this robot's feet
            # can stand on: MuJoCo collides a foot mesh against a *plane* with a
            # dedicated routine, while a mesh against a *box* goes through CCD,
            # which returns one contact point and loses it at zero penetration.
            # The bigger the box relative to the 1 cm foot, the worse it
            # conditions. Measured as single-step contact blips per foot per
            # second under a walking policy (512 envs, RTX 5090), where a clean
            # gait is 0.00:
            #
            #     one 12 m box   0.96      <- worse than the stair columns
            #     1.0 m paving   0.06      <- as clean as the heightfields
            #     0.5 m paving   0.55
            #
            # 0.5 m is worse than 1.0 m, not better: the foot is 4 cm across and
            # small tiles put it on a seam, touching two boxes at once. So this is
            # a minimum tile size, not "smaller is better".
            #
            # The compliant foot contact in `constants.py::_CONTACT_SOLIMP`
            # attacks the same artefact from the other side -- a tip that
            # compresses sits inside the surface, so there is no zero-penetration
            # degeneracy to lose -- and the obvious question is whether it makes
            # this redundant. Measured as the 2x2, it does not:
            #
            #                    rigid   silicone
            #     one 12 m box    0.75     0.51
            #     1.0 m paving    0.04     0.01
            #
            # Compliance takes a third off and still leaves this column an order
            # of magnitude worse than any other; the geometry is what fixes it.
            #
            # Same surface as before -- `grid_height_range=(0, 0)` keeps every cell
            # at zero, so this column is still the flat control that stops the
            # policy losing flat ground to the curriculum. `merge_similar_heights`
            # must stay off: it would merge the identical cells straight back into
            # the one big box this is replacing.
            "flat": box_random_grid(
                proportion=0.2,
                grid_width=1.0,
                grid_height_range=(0.0, 0.0),
                merge_similar_heights=False,
            ),
            # **The tread is what decides whether this column is terrain or a
            # wall**, and it is horizontal, so it takes `H_SCALE`. At `V_SCALE` it
            # was 0.113 m and the 0.331 m stance straddled 2.9 treads, putting fore
            # and aft feet 0.188 m apart in height against a 0.0785 m downward
            # reach. At 0.264 m the stance straddles 1.25 of them, which is a Go1's
            # number, and the worst case is two risers -- 0.096 m, which the robot
            # reaches by pitching a little. See the table in the docstring.
            #
            # 0.264 m leaves 15 steps a side, so the column still climbs 0.72 m --
            # 6.7 standing heights, and more than an episode ever crosses.
            "pyramid_stairs": pyramid_stairs(
                proportion=0.2,
                step_height_range=(0.0, 0.10 * V_SCALE * DIFFICULTY),
                step_width=0.30 * H_SCALE,
                # The flat centre is where the robot starts, so it is sized to hold
                # the robot -- a footprint, hence `H_SCALE`. At Go1's 3.0 m it was
                # 37% of an 8 m tile, a field of flat ground with a thin fringe of
                # stairs, which is what it looked like in the viewer. Scaled, it is
                # 2.64 m: 22% of a 12 m tile, and five times the robot's own 0.52 m
                # width.
                platform_width=3.0 * H_SCALE,
                border_width=1.0 * H_SCALE,
            ),
            # Descending is the harder direction and this is the column that asks
            # for it: 0.0785 m of downward reach against 0.257 m upward, so a riser
            # the robot climbs comfortably can still be one it cannot step off.
            # Same dimensions as the ascending pyramid, deliberately.
            "pyramid_stairs_inv": pyramid_stairs_inv(
                proportion=0.2,
                step_height_range=(0.0, 0.10 * V_SCALE * DIFFICULTY),
                step_width=0.30 * H_SCALE,
                platform_width=3.0 * H_SCALE,
                border_width=1.0 * H_SCALE,
            ),
            "hf_pyramid_slope": hf_pyramid_slope(
                proportion=0.1,
                slope_range=(0.0, MAX_SLOPE * DIFFICULTY),
                platform_width=2.0 * H_SCALE,
            ),
            "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
                proportion=0.1,
                slope_range=(0.0, MAX_SLOPE * DIFFICULTY),
                platform_width=2.0 * H_SCALE,
            ),
            "random_rough": random_rough(
                proportion=0.1,
                # **`scale_with_difficulty` defaults to False, and that made this
                # the one column the curriculum did nothing to.** The flag's own
                # docstring says it: "the roughness is fixed and `difficulty` is
                # ignored, matching upstream behavior". Neither mjlab's
                # `random_rough()` preset nor this file passed it, so the column
                # sat at its full `noise_range` on every row. Measured by calling
                # each sub-terrain's `function(difficulty, ...)` and taking the
                # heightfield's peak-to-peak relief, in metres:
                #
                #     sub-terrain     d=0.0    d=0.5    d=1.0
                #     hf_slope        0.000    1.305    2.610
                #     wave            0.000    0.080    0.180
                #     random_rough    0.035    0.035    0.035   <- flat curve
                #
                # 0.035 m against a foot lift of 0.025 m, so 10% of environments
                # started on ground with more relief than the gait lifts to, from
                # the first iteration, and never saw it get easier or harder.
                #
                # It also explains a measurement that was previously mis-attributed.
                # Single-step contact blips per foot per second at level 0 read
                # flat 0.02, stairs 0.02, hf_slope 0.05, wave 0.04, random_rough
                # 0.12 -- and the 0.12 was put down to heightfield collision.
                # `hf_slope` and `wave` are heightfields too and sit at 0.05 and
                # 0.04; the difference was simply that this column was not flat at
                # level 0 while they were.
                scale_with_difficulty=True,
                # **What the gait meets is the relief, which is the difference
                # between the two ends and not the upper one.** Both ends scale
                # with difficulty -- that is the mechanism putting hf_slope and wave
                # at 0.000 in the table above -- so a floor of 0.005 subtracts
                # itself from every row. Writing the top as `FOOT_LIFT * DIFFICULTY`
                # therefore asked for a quarter more than it delivered: the built
                # relief was the foot lift exactly, measured 0.030 m against the
                # 0.0375 m the comment claimed.
                #
                # The same quantum costs this column rungs at ten rows, which is
                # worth knowing rather than discovering: measured relief by row is
                # 0.000 at rows 0-1, then 0.005 to 0.035 in 0.005 steps, and rows 8
                # and 9 are both 0.035. So eight of the ten rungs differ here while
                # the other columns give ten. That is the floor of the ladder being
                # finer than the surface can express, not a fault -- flat ground at
                # the bottom of a roughness curriculum is what it should be.
                #
                # Adding the floor back makes the arithmetic say what it means.
                # `vertical_scale` then quantises what survives: 0.005 + 0.0375
                # truncates to a top of 0.040, so the built relief is 0.035 m and
                # the roughest row asks the gait to lift 1.4x what it lifts now
                # rather than the 1.5x written here. Nothing between 1.4 and 1.6 is
                # expressible at this quantum, and the request is left in the form
                # that says what was asked for.
                noise_range=(
                    _NOISE_FLOOR, _NOISE_FLOOR + FOOT_LIFT * ROUGHNESS_DIFFICULTY
                ),
                noise_step=0.005,
            ),
            "wave_terrain": wave_terrain(
                proportion=0.1,
                amplitude_range=(0.0, 0.20 * V_SCALE * DIFFICULTY),
                # **Derived, not written down.** `num_waves` is a count, so it is
                # the one parameter here that silently means something different
                # when the tile changes size -- and the tile did change, from
                # 3.02 m to 12 m, taking mjlab's 4 waves from a 0.63 m wavelength
                # to 2.875 m and leaving this the gentlest column on the board.
                # Computing it from `TILE` is what stops that happening the next
                # time the tile moves.
                #
                # Seven waves across the 11.5 m inside the rim is 1.643 m, within
                # 0.4% of what scaling asks for, and 5.0 stance lengths -- the same
                # figure a Go1 gets from upstream.
                num_waves=max(round((TILE - 2 * _HF_BORDER) / _WAVE_LENGTH), 1),
                # mjlab's own default, named here because `_WAVE_LENGTH` above is
                # measured across the tile *less this rim* and the two have to
                # agree. The rim is also a real edge: the wave starts at full phase
                # against flat ground, so the tile is ringed by a ramp of up to the
                # amplitude in one 0.1 m cell. It is upstream's, it is at the tile
                # boundary rather than where the robot is placed, and a robot
                # meeting it has already walked off its assigned difficulty.
                border_width=_HF_BORDER,
            ),
        },
        add_lights=True,
    )


#: The environment keys behind `--terrain-row` and `--terrain-col`, so that a
#: pinned tile can be set from `.env` like every other default in this
#: repository.
TERRAIN_ROW_ENV = "MJRL_TERRAIN_ROW"
TERRAIN_COL_ENV = "MJRL_TERRAIN_COL"


def cli_args(group):
    """This scene's own command-line arguments; see `scenes.load_cli_args`.

    **A terrain row and column are this scene's vocabulary**, so they are
    declared here rather than in `scripts/_cli.py` -- one parser shared by three
    entry points, and the layering `tests/test_log_layout.py` pins. The mechanism
    is the same one `hexa.swing` uses for `--swing-angle`, and it was built for
    tasks first; `scenes.load_cli_args` is the mirror of it.

    Returns the `dest` names it added, so the entry point knows which parsed
    values belong to this scene without reading argparse's private structures.
    """
    import os

    group.add_argument(
        "--terrain-row", default=os.environ.get(TERRAIN_ROW_ENV, ""), metavar="N",
        help="pin every tile to one difficulty row, 0 (flat) to "
        f"{NUM_ROWS - 1} (hardest). Empty is the full ladder. The terrain "
        "curriculum then has nowhere to promote to, which is the point: a "
        "replay of one difficulty is a replay of one difficulty",
    )
    group.add_argument(
        "--terrain-col", default=os.environ.get(TERRAIN_COL_ENV, ""), metavar="NAME",
        help="pin every tile to one sub-terrain: flat, pyramid_stairs, "
        "pyramid_stairs_inv, hf_pyramid_slope, hf_pyramid_slope_inv, "
        "random_rough, wave_terrain. Empty is the mix",
    )
    return ("terrain_row", "terrain_col")


#: Rows in the generated grid; one difficulty level each. Named so that
#: `cli_args` can quote the range without building the generator, which would
#: import mjlab at parse time -- and read by `_terrain()` below, so the help text
#: and the generator cannot say different numbers.
#:
#: **10, and it has been 10 and 5 and 10 again** -- `rough: two ratios rather
#: than one, and half as many rungs` halved it and `rough: ten rungs again, and a
#: record of the five that did not work` put it back. The ladder is that line of
#: work's to decide; what this constant is for is that the pinning below and the
#: generator read the same number. They were separate literals through both of
#: those changes, so `--terrain-row 9` was rejected as "outside 0..4" on a ten-row
#: ladder for as long as the two disagreed.
NUM_ROWS = 10


def _pin(cfg, row: str, col: str):
    """Restrict a generated terrain to one row, one column, or both.

    **Not a new terrain: the same generator, narrowed.** A tile pinned this way
    is byte-for-byte the tile the full grid would have produced at that row and
    column, which is what makes it a test of the training terrain rather than a
    lookalike.

    A row becomes `num_rows=1` with `difficulty_range` collapsed onto that row's
    difficulty; a column becomes a `sub_terrains` dict of one, whose proportion
    stops meaning anything once it is the only entry.

    Raises on a name that is not there rather than falling back to the mix. A
    typo that silently ran the full terrain would be a test of something other
    than what was asked for, and the run would look exactly like a correct one.
    """
    from dataclasses import replace

    changes: dict = {}
    if col:
        if col not in cfg.sub_terrains:
            raise SystemExit(
                f"[scene] --terrain-col {col!r} is not a sub-terrain of the rough "
                f"scene. Available: {', '.join(sorted(cfg.sub_terrains))}"
            )
        changes["sub_terrains"] = {col: cfg.sub_terrains[col]}
    if row:
        try:
            index = int(row)
        except ValueError:
            raise SystemExit(
                f"[scene] --terrain-row {row!r} is not an integer. Rows are 0 "
                f"(flat) to {cfg.num_rows - 1} (hardest)."
            ) from None
        if not 0 <= index < cfg.num_rows:
            raise SystemExit(
                f"[scene] --terrain-row {index} is outside 0..{cfg.num_rows - 1}."
            )
        # The difficulty this row would have had in the full grid. The generator
        # spreads `difficulty_range` across `num_rows`, so a single row has to be
        # handed the value rather than the index.
        difficulty = index / (cfg.num_rows - 1) if cfg.num_rows > 1 else 0.0
        changes["num_rows"] = 1
        changes["difficulty_range"] = (difficulty, difficulty)
    if not changes:
        return cfg
    pinned = replace(cfg, **changes)
    print(
        "[scene] rough pinned to"
        + (f" row {row} (difficulty {pinned.difficulty_range[0]:.3f})" if row else "")
        + (f" column {col!r}" if col else "")
    )
    return pinned


#: What `mjwarp.put_data` demands of this ground at build time, measured.
#:
#: It checks a single CPU `MjData` at the model's **default pose** -- before
#: `env_origins` are applied, so the robot and every tile are stacked at the
#: origin -- and refuses with `nconmax overflow (nconmax must be >= N)`. That is
#: not what the simulation needs. Over 512 worlds and 200 steps on this ground:
#:
#:     at spawn             nefc 22/world     contacts 0
#:     peak over 200 steps  nefc 146/world    contacts 13/world
#:     put_data demands     nefc 490          contacts 117
#:
#: **And the check moves with the ladder, in the direction nobody would guess.**
#: At five rows it wanted 184 and 758; at ten it wants 117 and 490. Twice the
#: rows is twice the tiles but also twice the grid -- 120 m in the row axis
#: rather than 60 -- so the robot sitting at the origin intersects fewer of them.
#:
#:     rows   put_data wants        task's own budgets
#:       5    ncon 184  nefc 758    128 / 512   <- both overflow, must be raised
#:      10    ncon 117  nefc 490    128 / 512   <- both already cover it
#:
#: Stable across seeds: `seed=None`, but eight builds gave the same pair each
#: time. The default pose does not depend on the random heights.
#:
#: **And it moves with the robot.** Those were the hexa model's numbers. The
#: jumper v1.6.1 export carries more collision geometry -- the jaws' grip inserts
#: are links of their own -- and at the same ten rows it wants
#:
#:     model            put_data wants        task's own budgets
#:     jumper v1.6.1    ncon 131  nefc 546    128 / 512   <- both overflow
#:
#: which is how `--scene rough` stopped building on the GPU, in every task but
#: five_foot, while the suite stayed green: its test compared the budgets with
#: the two numbers below rather than measuring. It measures now (measured again
#: 2026-09-29, on the CPU exactly as `put_data` checks; the dToF and its camera
#: change neither number).
REQUIRED_NCON = 131
REQUIRED_NEFC = 546

#: Contact and constraint budgets to raise the task's to, or None to leave them.
#:
#: **Raised again, for the jumper**: `nconmax` to exactly what the check asks,
#: and `njmax` freely, per the table below. They were None while the hexa model's
#: 117 and 490 fitted inside the task's 128 and 512.
#:
#: They were 192 and 1024, set when the ladder had five rows and genuinely did
#: need them, and carried across the restore to ten where they bought nothing:
#: measured on a real run, 4096 environments, **14439 MB against 11693** -- 2.7
#: GB of a 23 GB card for a raise that was a no-op. `nconmax` is the expensive
#: knob and `njmax` is nearly free, which is the reverse of the arithmetic:
#:
#:     nconmax  njmax    GPU, flat ground, 4096 envs, after ten steps
#:       128      512    8568 MB
#:       128     1024    7665 MB   <- doubling njmax costs nothing
#:       256      512   13277 MB   <- doubling nconmax costs 4.7 GB
#:
#: So if this ever has to be set again, raise `njmax` freely and `nconmax` only
#: to what the check asks for. The failure when it is too low is at construction,
#: in the first seconds, naming the value -- which is what makes a tight number
#: safe here and an over-provisioned one merely expensive.
NCONMAX = 131
NJMAX = 1024


def scene(terrain_row: str = "", terrain_col: str = "") -> Scene:
    """Build the rough scene, optionally pinned to one tile of the grid.

    Args:
        terrain_row: difficulty row to pin every tile to, as a string because it
            arrives from the command line and from `.env`; empty is the full
            ladder. See `cli_args`.
        terrain_col: sub-terrain name to pin to; empty is the mix.
    """
    from mjlab.utils import spec_config as sc

    return Scene(
        headlight=Headlight(ambient=(0.10, 0.11, 0.12), diffuse=(0.06, 0.06, 0.07)),
        textures=(
            sc.TextureCfg(
                name="rough_sky", type="skybox", builtin="gradient",
                rgb1=(0.45, 0.48, 0.55), rgb2=(0.62, 0.66, 0.74),
                width=256, height=256,
            ),
        ),
        materials=(),  # see the note above: the generator colours its own tiles
        lights=(
            sc.LightCfg(name="sun", type="directional", pos=(-4.0, -1.0, 2.5),
                        dir=(0.8, 0.2, -0.6), castshadow=True,
                        diffuse=(0.55, 0.54, 0.51), ambient=(0.10, 0.11, 0.13)),
        ),
        terrain_type="generator",
        terrain_generator=_pin(_terrain(), terrain_row, terrain_col),
        nconmax=NCONMAX,
        njmax=NJMAX,
    )
