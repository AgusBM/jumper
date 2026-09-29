"""A swing frame with a plank seat, sized for the robot to stand on.

**This scene adds a physical prop**, like `football` and unlike the look-only
ones: the frame is solid, the seat swings, and both have mass. Selecting it does
not change how a task is scored, but it does put something in the world the robot
can climb onto and fall off, so a policy trained with it is not comparable to one
trained without. `registry.apply` says so at startup.

## The seat hangs on rope, and rope is a tendon

Four ropes, none of them a link. A rope carries tension and nothing else -- it
cannot push, it cannot hold a bending moment, and it goes slack the moment the
load comes off it. MuJoCo's name for that is a **spatial tendon with an upper
length limit**: the constraint is `length <= L`, one-sided, so the rope pulls the
seat back when it reaches its length and does nothing at all when the seat rises.
A rigid rod expresses none of that, and the difference shows the moment the robot
does anything sharp -- a rod pushes the plank up after the robot, and a rope
simply lets it fall.

Four ropes, two a side. Both of a side's ropes leave the **same** point on the
beam and land `TIE_X` fore and aft of the plank's centre, so each side is a
triangle with its apex on the beam.

**That triangle is the whole anti-tip mechanism.** Pitch the plank and one of the
two ropes has to get longer -- the fore one slackens, the aft one comes up taut
and stops it, and the other way round for the other direction. Measured with a
100 mm off-centre rider the deck reaches 6.1 degrees and comes back; with
`TIE_X = 0`, which collapses each triangle to a single rope, the same load puts it
through 179.8 degrees. A flat seat on two single ropes is in neutral equilibrium
about the line between them, which is why a playground swing hung that way flips.

The consequence worth stating is what it does *not* leave free: the plank rides
**square to the rope**, exactly, at every point of the arc. Measured over a 30
degree release, the deck's pitch tracks the rope's angle with a slope of 1.000 and
a worst-case disagreement of 0.00 degrees. The robot stands square to the deck, so
its own IMU reads the swing angle -- which is what the entire actor observation in
`tasks/jumper/swing/` rests on.

**An earlier version put a knot in each rope**, `BRIDLE_H` above the deck, with a
single rope carrying on to the beam. It worked -- it is the same triangle, moved
down -- but it cost two extra bodies, six extra degrees of freedom and a second
pendulum mode that overshot the rope by 18%, and because the knot's height was
fixed while the rope length varied, at short lengths the rig stopped looking like
a rope holding a plank and started looking like a triangle holding one.

## The length is a per-environment quantity

A rope is a tendon and a tendon's length is `tendon_range[.., 1]`, which both
backends carry **per world** -- so every environment in a batch can hang its seat
at a different length from the same compiled model, and `jumper.swing` does. What
changes is all four ropes together (`ROPES`); each one is the hypotenuse of its
own triangle, so the length written into the model is `rope_for(hang)` and not the
hang itself.

The frame does not move with it. A longer rope hangs the seat lower, which is what
lengthening a real swing does, and it keeps the posts in the ground at every
length. `HANG_L_RANGE` is how far that can be pushed before the rig stops being a
swing at one end and reaches the ground at the other; `HANG_L` is only the
nominal, which is what `--scene swing` builds.

Nothing else in this file scales with the length, and that is deliberate: the tie
spread, the plank and the frame are all sized by the robot or by the frame's own
proportions, so a change of length is a change of *one* number. `TIE_X` staying
fixed costs nothing, and that is the difference from the knot it replaced: a
shorter rope simply makes the triangle wider in angle, where a fixed knot *height*
ate into the hang from the top and left a stub of rope above itself.

## What is a joint here and what is not

The plank has six degrees of freedom, and every one of them is a plain slide or
hinge rather than a freejoint. That is not a modelling preference:

    an mjlab Entity may contain at most one freejoint, and an entity that has
    one is floating-base -- so it is not mocap-wrapped, and its welded parts
    stay at the model origin in every environment

The frame has to be welded and per-environment, so the entity has to stay
fixed-base, so the plank cannot have a freejoint. Three slides and three hinges
are the same six degrees of freedom written in a way that leaves `is_fixed_base`
true.

The cost is Euler angles rather than a quaternion, so one of the three is
singular at 90 degrees -- and **which one is a choice, not a fact**. It used to be
pitch, on the argument that the seat would never get near it; measured on the
finished task, pitch reaches 80 degrees under action noise and a policy that pumps
goes over, which arrives as a NaN in `qpos` and surfaces as "the observation
contains NaN" with nothing naming the swing. Roll is in the middle now, and roll
does not move: 17.6 degrees at the same noise. See `_free_dofs`.

## The plank is sized from a measurement of the robot, not from a real swing

A real swing seat is about 0.45 m wide and 0.20 m deep, which is a person's hips.
This robot's stance is a different shape, so the seat is derived from it instead;
see `STANCE_X` / `STANCE_Y`, which are measured, and `FOOT_MARGIN`, which is not.

## What this scene does not do

**It does not put the robot on the seat.** The prop lands `SWING_X` in front of
each environment's origin and the robot spawns where its task says, which for
every task here is the origin, on the ground. Standing the robot on the plank
means moving the *robot*, and where a robot starts is the task's business -- a
scene that moved it would change the task silently, for every task it is selected
on. `tasks/jumper/swing/` is where that belongs. Until then, `--scene swing` on any
jumper task gives a robot that can walk over and bump into a swing.

**A foot on the plank is invisible to the jumper tasks' contact sensor.**
`velocity_env.py` builds `feet_ground_contact` with `secondary=body "terrain"`, so
it reports contact with the ground and with nothing else. Every gait reward built
on it -- air time, stance load, the gait-phase match -- reads a robot standing on
the plank as a robot with all six feet in the air. Nothing raises; the numbers are
simply about a different robot than the one on screen. A task that wants the
plank scored has to add the seat geom to that sensor.

**No `qpos` is "the swing angle".** Hang the seat from a single rigid hanger and
one hinge answers the question; hang it on rope and the seat has six degrees of
freedom held by four one-sided constraints, and no one of them is the swing.
`seat_pitch` comes closest -- the plank rides square to the rope, so it tracks the
swing exactly -- but it is an Euler angle and keeps counting past 90, so a plank
that has been round once reads 400. Read the seat body's pose instead --
`scene["swing"].data.body_link_pos_w` at the `seat` row -- which is what the
swing physically is, and which goes on meaning the same thing if the rigging
changes.
"""

from __future__ import annotations

from typing import Any

from .registry import GROUND, Headlight, Scene

# ── The plank ───────────────────────────────────────────────────────────
#
# Sized from the robot that has to stand on it.

#: The hexapod's standing footprint, from the six foot bodies at `HOME` with the
#: base at `STAND_Z`. Measured on `assets/jumper/jumper.xml`, this worktree:
#:
#:     LF_palm_pad_b_link ( 0.1899,  0.0455)    LM_foot_tip_link   (-0.0171,  0.2033)
#:     RF_palm_pad_b_link ( 0.1899, -0.0453)    RM_foot_tip_link   (-0.0171, -0.2030)
#:     LR_foot_tip_link   (-0.1538,  0.1633)    RR_foot_tip_link   (-0.1538, -0.1631)
#:
#: so the feet span 0.344 m fore-aft and 0.406 m across. **The middle legs are
#: what set the width**: they splay to +/-0.203 m, more than four times as far out
#: as the front feet, and a plank sized by eye from the body would be far too
#: narrow.
#:
#: **Re-measured on every revision of the hardware.** The front foot has been
#: `LF_F_Link`, then `LF_cehou_tip_link`, and is now `LF_palm_pad_b_link`; across
#: those the span has moved by up to 17 mm and the centre by 18 mm. None of it is
#: visible from here -- the asset is replaced in place, which is this project's
#: convention -- and what caught it was
#: `tests/test_scenes.py::test_the_plank_is_still_the_size_of_the_robot_that_stands_on_it`,
#: on the first upstream change after it was written.
STANCE_X = 0.1995 - (-0.1460)
STANCE_Y = 0.2050 - (-0.2047)

#: Where that footprint is centred, in the robot's own frame. It is **not** under
#: the base: the front legs reach 0.200 m forward and the rear ones only
#: 0.146 m back. A task standing the robot on the plank puts the base
#: here, negated, or the front feet hang 0.027 m closer to the
#: edge than the rear ones do.
STANCE_CENTRE_X = (0.1995 + (-0.1460)) / 2

#: Clear plank outside the outermost foot, on every side. **Not measured** -- there
#: is no run behind it. It is 15% of the stance, chosen so that the robot can shift
#: its weight and take a step without a foot arriving over the edge; the smallest
#: value that still reads as "standing on a plank" rather than "balanced on a
#: pallet". Raising it makes the plank heavier (below) as well as safer.
FOOT_MARGIN = 0.05

BOARD_X = STANCE_X + 2 * FOOT_MARGIN
BOARD_Y = STANCE_Y + 2 * FOOT_MARGIN

#: A 20 mm plank. Thick enough to read as timber rather than as a sheet, and it is
#: the one dimension that costs mass for nothing -- see `BOARD_DENSITY`.
BOARD_T = 0.020

#: The plank's wood, kg/m^3. Paulownia -- a real, light hardwood, and the plank's
#: mass follows from the geometry rather than being chosen, which is the point of
#: writing the density instead of the mass: 0.444 x 0.506 x 0.020 m of it is
#: **1.26 kg**.
#:
#: **This is the number that decides whether the robot can pump at all**, and it
#: was pine at 500 until it was measured. A rider changes the pendulum's centre of
#: mass by moving its own, so its authority is its *share* of the swinging mass.
#: A child on a playground swing is ten times the seat and can pump at will; this
#: robot is 2.77 kg against a seat that was 2.31, i.e. 55%, and most of the
#: pendulum was dead weight.
#:
#: Measured, with a servoed rider leaning fore and aft in time with the swing
#: (open loop at the swing's own frequency, ramped in over 5 s), as the steady
#: amplitude it holds after 60 s:
#:
#:     density  knot   seat kg   robot's share   lean +/-40 mm   +/-60 mm
#:                     (the knot column is the rigging this table was measured
#:                      on; the knots are gone and the seat is 40 g lighter, so
#:                      the chosen row reads 68.8% today rather than 70.2%)
#:     500      50 g    2.309       56.6%           21.9 deg      25.0 deg
#:     350      30 g    1.606       65.2%           23.4          31.7
#:     280      20 g    1.277       70.2%           28.7          37.6      <- this
#:     280      20 g    0.968 *     75.6%           33.5          48.5
#:     200      20 g    0.703 *     81.1%           43.3          64.1
#:                                                  (* at 15 mm thick)
#:
#: **The table was measured against a 3.01 kg robot and the robot is now 2.77.**
#: The chosen row reads 68.1% rather than 70.2% today, and the two amplitude
#: columns will have come down by a few percent with it. It is left as measured
#: rather than adjusted by arithmetic, because what the table is for is the shape
#: -- authority is the rider's *share*, and it climbs steeply through this range --
#: and a row nobody ran is worth less than a row that is slightly out of date and
#: says so.
#:
#: **The two amplitude columns are not a ceiling**, and that took a second
#: measurement to notice: the rider was driven open loop at the swing's
#: small-amplitude period, which detunes as the amplitude grows, so what they
#: record is the cost of pumping at a fixed slightly-wrong frequency. A rider that
#: keeps the phase drives this rig past 85 degrees on +/-20 mm. They are still the
#: right column to choose a density by -- the *share* is what they track, and the
#: share is real -- but nothing here bounds what a policy can reach. See
#: `HANG_L_RANGE`.
#:
#: 68% is where the target amplitude stops needing the robot's whole travel: it
#: reaches 20 degrees on about 20 mm of lean and has 37 available. Going lighter
#: buys more and costs the plank its plausibility -- 15 mm of 200 kg/m^3 is a foam
#: core, not a board somebody would stand on.
BOARD_DENSITY = 280.0

#: The frame's timber. **Inert**: the frame is welded to the world, so its mass
#: never enters the dynamics. It is written honestly rather than left at the
#: board's value so that nobody reads the two as one material.
FRAME_DENSITY = 500.0

# ── The rigging ─────────────────────────────────────────────────────────

#: Beam to the plank's **top face**, measured straight down. Sets the period.
#:
#: Not the rope's own length: a rope runs from the beam to a tie point `TIE_X`
#: fore or aft of centre, so it is `rope_for(HANG_L)` = 1.362 m at the nominal.
#: Everything else in the task is expressed in the hang, because that is the
#: pendulum, and `rope_for` / `hang_for` convert at the one place it matters.
#:
#: **The nominal only.** The length is a per-environment quantity -- see
#: `HANG_L_RANGE` and `upper_rope` -- so this is what the scene builds and what
#: `--scene swing` gives, not what a task that randomises it hangs the seat at.
#:
#: Measured free-swinging at **2.347 s** empty and 2.290 s with a robot on the
#: deck. At a 50 Hz control step that is 115 control steps per swing, and a 30 s
#: episode is **13 swings** -- which is the number to keep in mind when reading
#: anything about pumping, because it bounds how many chances a policy gets within
#: one episode.
#:
#: A point mass hanging 1.35 m would give 2.331 s, so the empty plank's own
#: inertia adds 0.7%. It added 4.8% at a third of this length: a 0.442 x 0.500 m
#: board has a lot of inertia about its own centre, and the longer the rope the
#: less of the pendulum that is. Short swings are compound pendulums and long ones
#: are nearly bobs on strings.
#:
#: The **loaded** period is *shorter* than either, and that is not a rounding
#: error: the robot's centre of mass rides 0.06 m above the deck, so loading the
#: seat moves the pendulum's centre of mass up towards the beam and the effective
#: length down. An earlier reading of 2.360 s -- longer -- came from a measuring
#: rig that added the rider's mass to the seat body by writing `body_mass` and
#: `body_ipos` on the compiled `MjModel`. The mass takes; **`body_ipos` does
#: not**, measured on MuJoCo 3.11, so the rider was silently placed at the deck's
#: centre instead of above it. Build a rider into the spec.
HANG_L = 1.35

#: What lengths this frame can actually be rigged at, in metres.
#:
#: The beam does not move, so a longer rope hangs the seat lower -- which is what
#: adjusting a real swing's ropes does, and it means one frame covers the whole
#: range with its posts still in the ground. Both ends are set by geometry rather
#: than chosen:
#:
#:     0.60   the deck is 1.35 m up and the robot standing on it reaches 1.46 m,
#:            or 1.55 m with the swing at 35 degrees. The beam is at 1.95 m, so
#:            there is 0.40 m of headroom left. Each rope leaves the beam at 17
#:            degrees off vertical, which is a wide rig and still a rig.
#:     1.80   the deck's top face is 0.15 m above the ground. The seat only ever
#:            rises from rest, so this is its lowest point, but there is not much
#:            left to fall off onto.
#:
#: **This was 1.00 m under the previous rigging and the reason has gone away.**
#: With a knot at a fixed 0.25 m above the deck, a short rope left a stub above the
#: knot and the rig read as a triangle rather than as a rope holding a plank --
#: rendering it is what showed that, not arithmetic, and 0.70 m was the version
#: that had to be withdrawn. Four ropes straight to the beam have no fixed-height
#: fitting to run out of, so the short end is now limited by headroom under the
#: beam instead, and that is 0.40 m away rather than 0.
#:
#: Measured across a wider span than the range, because the rows outside it are
#: what the ends are chosen against. A 2.77 kg rider standing on the deck:
#:
#:         L   spread    deck      T   T point   half-life   Q    delta(60mm)
#:     [0.50      20d    1.45  1.365    1.419        16 s   12.0    10.92 deg ]
#:      0.60      17d    1.35  1.497    1.554        18 s   12.0     8.89
#:      0.70      14d    1.25  1.620    1.678        19 s   11.5     7.49
#:      0.90      11d    1.05  1.846    1.903        19 s   10.5     5.70
#:      1.10       9d    0.85  2.049    2.104        20 s   10.0     4.59
#:      1.35       8d    0.60  2.280    2.331        22 s    9.5     3.70
#:      1.50       7d    0.45  2.407    2.457        22 s    9.0     3.31
#:      1.65       6d    0.30  2.529    2.577        21 s    8.5     2.99
#:      1.80       6d    0.15  2.645    2.691        22 s    8.5     2.74
#:
#: `spread` is the angle each rope makes with the vertical, `arctan(TIE_X / L)`.
#: The period across the range spans **1.50 to 2.65 s**, a factor of 1.77, and the
#: half-life in *seconds* barely moves while the half-life in *swings* falls from
#: 12.0 to 8.5.
#:
#: `delta` is the static tilt a rider commands by standing 60 mm forward of centre.
#: **It is monotone now, and it was not before**: with a fixed-height knot it ran
#: roughly as `mu a / L` down to about 1.1 m and then climbed steeply, because the
#: knot took over the geometry as the rope above it shortened. That knee was the
#: same one the render showed, from the dynamics side, and both of them are gone.
#:
#: **Rider authority is not what limits the swing.** A rider that keeps the phase
#: -- leaning with the sign of the rope's angular rate -- drives every length in
#: this table past 85 degrees, on as little as +/-20 mm of travel. The 12 / 18 / 24
#: degree figures recorded elsewhere in this file for +/-20 / 40 / 60 mm were
#: measured with the rider driven **open loop at the small-amplitude period**,
#: which detunes as the amplitude grows; what they measure is the cost of pumping
#: at a fixed slightly-wrong frequency, not what a rider can reach. The limit on
#: this task is timing, not strength.
HANG_L_RANGE = (0.60, 1.80)

#: The four ropes. All four change together when a task varies the length; there
#: is nothing else in the rigging.
ROPES = ("rope_lf", "rope_lb", "rope_rf", "rope_rb")

#: How far fore and aft of centre the ropes tie to the plank. Inset from the
#: 0.221 m ends because a rope hole goes through timber, not through end grain.
#:
#: **This is the anti-tip mechanism, and it is the only one.** Each side's two
#: ropes leave the *same* point on the beam and land `TIE_X` fore and aft of the
#: plank's centre, so the pair is a triangle with the beam at its apex. Pitch the
#: plank and one of the two has to get longer:
#:
#:     |beam - tie|^2 = TIE_X^2 + hang^2 -/+ 2 * hang * TIE_X * sin(pitch)
#:
#: -- the fore rope shortens and goes slack, the aft rope lengthens and stops it.
#: Either direction of pitch is caught by one of the two, and the restoring is
#: stiff: at 1.35 m the derivative is 0.179 m/rad, so one degree of pitch asks for
#: 3.1 mm of stretch out of a rope whose whole working stretch is about 5 mm.
#:
#: The plank therefore rides **square to the rope**, which is what a swing does and
#: what the whole task is built on: the robot stands square to the deck, so its own
#: IMU reads the swing angle.
#:
#: At `TIE_X = 0` the two ropes lie on top of each other, the triangle degenerates
#: to a point, and nothing resists pitch at all -- the plank is in neutral
#: equilibrium about the line between its ropes and flips, which is why a flat
#: playground swing hung on two single ropes flips. That is the control group
#: `_swing_spec` takes an override for, and it is a sharper one than the rigging it
#: replaced: a knot lowered to the deck still had some spread left.
#:
#: **The spread does not scale with the length, and here that costs nothing.**
#: `TIE_X` is a fitting on the plank and the plank is sized by the robot, so a
#: shorter rope makes the triangle wider in angle rather than leaving a stub of
#: rope above a fixed-height knot. That stub is exactly what the previous rigging
#: ran out of at short lengths, and why its usable range stopped at 1.00 m.
TIE_X = 0.18

#: A rope's drawn radius. The ropes themselves are **massless** -- a tendon has no
#: inertia in MuJoCo -- which under-states real rigging by a few hundred grams and
#: is, for once, the error in the useful direction: every gram of rigging is mass
#: the robot cannot move, and this task lives or dies on the robot's share of the
#: swinging mass (see `BOARD_DENSITY`).
ROPE_R = 0.006

#: The plank's top face at rest, above the ground. The robot stands 0.105 m tall,
#: so it rides at 0.705 m and passes 1.24 m clear under the beam.
#:
#: **Nothing walks onto this.** At 0.60 m the deck is nearly six times the robot's
#: standing height, so a task that wants the robot on the plank has to start it
#: there. That is a change from a deck at 0.20 m, where climbing on was at least
#: arguable.
BOARD_TOP_Z = 0.60

#: Beam height. Derived, so that raising the beam lengthens the ropes and leaves
#: the seat where it is, rather than lifting the plank out of reach.
BEAM_Z = BOARD_TOP_Z + HANG_L

#: Viscous damping on the seat's three sliding degrees of freedom, N per m/s.
#:
#: **The damping had to move when the rigging did.** A rope-hung seat has no
#: hinge to put it on, so the loss lives on the plank's own degrees of freedom
#: instead.
#:
#: Measured as the decay of a free swing released with a 1.0 m/s kick through the
#: bottom of its arc -- 0.36 m of amplitude, 15 degrees -- at a 0.005 s step, as
#: how long it takes to lose half of that. Release it by **velocity at the
#: bottom**, not by displacement: a seat set out along `x` is off the arc with
#: every rope over-stretched, and the constraint eats a fifth of the amplitude in
#: the first half-swing. That is a measurement artefact and it read as heavy
#: damping for a while.
#:
#:     SEAT_DAMPING   empty seat     with the robot on it
#:     0.0            never          never
#:     0.1             8 sw / 18 s   24 swings / 56 s
#:     0.15            5 / 12 s      16 / 38 s
#:     0.25            4 /  9 s      10 / 24 s          <- this
#:     0.4             2 /  5 s       6 / 15 s
#:
#: **Re-measured twice without this number being touched**, which is the reason to
#: keep the table rather than the conclusion. Lightening the plank moved the loaded
#: half-life 31 s -> 25 s, because the same coefficient bites harder on less
#: inertia; a revision of the robot then took 8% off its mass and moved it to 24 s.
#: The empty column is no longer a useful reading at all -- the plank is a third of
#: what it was and the robot is 68% of the pendulum, which is the point of the
#: change.
#:
#: **Read the seconds column, not the swings column.** The episode is 20 s
#: whatever the period is, and tripling the swing's height stretched the period by
#: the square root of three without changing how fast energy leaves: the loaded
#: half-life was 28 s at a third of this length and is 31 s here, on the same
#: coefficient. So this number survived the rescaling untouched, and the swings
#: column moved under it.
#:
#: **The top row is the one that certifies the model.** With every damping
#: constant at zero the swing does not decay measurably in 90 s, so six one-sided
#: rope constraints being solved 200 times a second are not a hidden energy sink,
#: and the decay below is entirely these two numbers. That was worth checking
#: rather than assuming: it is exactly the kind of loss that would have been read
#: as physics.
#:
#: The loaded column is the one that matters -- the robot is most of the moment of
#: inertia. 24 s against a 30 s episode means a swing left alone keeps about
#: **42%** of its amplitude across a whole episode:
#: pumping has to earn the rest, and coasting is not free but is not punished hard
#: either. With no loss at all the task would be free, because any amplitude a
#: policy reached it would then keep for nothing.
#:
#: Raise it if a policy turns out to build amplitude and then stop working; that
#: symptom is a swing that is too cheap to hold. Nothing has been trained here yet.
#:
#: Viscous rather than dry friction on purpose, and for the reason `soft.py`
#: records for its drag term: `frictionloss` needs `sign(v)`, which chatters when
#: the swing is near either end of its arc and hovering around zero velocity.
#: Linear in velocity is smooth through zero. Real losses here are rope hysteresis
#: and air drag, neither of which is linear, so this is a stand-in for both.
SEAT_DAMPING = 0.25

#: Damping on the seat's three hinges, N*m per rad/s.
#:
#: **Small on purpose, and it is not a free knob.** The deck is not free to rotate
#: -- it rides square to the rope, so its pitch *is* the swing angle, and damping
#: the pitch damps the swing itself. Measured on the rigging this replaced: at
#: 0.02, with `SEAT_DAMPING` at zero, the swing still lost half its amplitude in 22
#: swings. All of that was bleeding out through the deck's rotation, and it made
#: the decay table above unreadable until it was taken out.
#:
#: **What it was for is gone.** With a knot in each rope the seat was a double
#: pendulum -- the upper rope, then the bridle -- so the deck could swing about the
#: knots as well as with them, and it overshot the rope by 18%. Four ropes straight
#: to the beam have no second mode: the deck tracks the rope with a slope of 1.000
#: and a worst-case disagreement of 0.00 degrees. So this is now damping a degree
#: of freedom that no longer moves independently of the swing, which means it is
#: doing nothing useful and a little harm.
#:
#: Left in at 0.005 rather than zeroed, because the yaw and roll hinges are damped
#: by the same number and those *are* free -- the four ropes leave the plank able
#: to twist about the vertical and to roll a little. Zeroing it would take the
#: settling off those two as well. Worth splitting into a per-axis value if the
#: decay ever needs to be pinned down further than the table above pins it.
SEAT_ANG_DAMPING = 0.005

#: Rope stiffness, as `solreflimit` on the tendon limits: time constant and
#: damping ratio, the same pair as a contact's `solref`.
#:
#: A limit is a soft constraint, so a rope stretches under load, and the default
#: (0.02, 1.0) lets it stretch about 3.7% -- measured on the probe that established
#: this model works at all. That is not far off real rope: polyester rigging runs
#: 2-3% at working load and natural fibre more, so the default is defensible and
#: the reason to tighten it is not realism but the plank's ride height wandering
#: with the robot's weight. 0.01 s halves the stretch and stays comfortably above
#: the 2x-timestep floor the integrator needs at a 0.005 s step -- the same bound
#: `constants.py::_CONTACT_SOLREF` records measuring for contacts.
ROPE_SOLREF = (0.01, 1.0)

# ── The frame ───────────────────────────────────────────────────────────

#: Half the beam's length. The ropes hang at the plank's edges, +/-0.250 m, and
#: the beam carries on past the A-frames outside them.
#:
#: Set by **clearance from the legs**, which is what decides a real swing frame's
#: width: about 0.35 m between the rope and the post, so that a seat swinging out
#: and twisting on the way back does not put a rider into the frame. That is
#: 0.60 m for the A-frames and a short overhang beyond. A 1.95 m frame 0.76 m
#: across, which is what sizing the beam by the plank alone gave, reads as a
#: gantry rather than as a swing.
BEAM_HALF = 0.65

#: Where the two A-frames stand, and how far their feet splay fore and aft.
#:
#: The splay is along the swing direction, which is what an A-frame's triangle is
#: for and which keeps both legs clear of the plank's arc: the plank reaches
#: +/-0.250 m across and the legs stand at +/-0.600 m.
#:
#: The two are set by different things and neither is a copy of the other.
#: `FRAME_SPREAD` is an **angle** -- 25 degrees off vertical -- so it scales with
#: the frame's height, and holding it is what keeps a 1.95 m frame from being a
#: tripod. `FRAME_Y` is a **clearance**, 0.35 m from the rope to the post; see
#: `BEAM_HALF`. The frame ends up 1.80 m fore-aft by 1.30 m across at 1.95 m tall,
#: which is a swing frame's proportions.
FRAME_Y = 0.60
FRAME_SPREAD = 0.90

#: Round timber, and it scales with the square root of the height rather than with
#: the height: a 1.95 m frame in 132 mm poles would be a jetty. 100 mm for the beam
#: and 88 mm for the legs is what a swing frame this size is actually built from.
BEAM_R = 0.050
POST_R = 0.044

#: Where the swing stands, in front of each environment's origin.
#:
#: **Not on top of it.** The robot spawns at the origin and the frame's near legs
#: reach `FRAME_SPREAD` back from the beam, so a swing centred there would stand
#: its posts through the robot in every environment. At 1.30 m the nearest post
#: foot is at 0.40 m, 0.19 m clear of the front feet.
#:
#: The swing's footprint is now 1.8 m of the 2.0 m `env_spacing`, so neighbouring
#: environments' frames nearly touch **in the viewer**. That is a picture, not a
#: collision: worlds are batched, not laid out side by side in one physics scene,
#: and nothing in one environment can reach anything in another. Raise
#: `scene.env_spacing` if the render is too crowded to read.
SWING_X = 1.30

#: Where the ropes meet the plank and the beam: at the plank's two long edges.
ROPE_Y = BOARD_Y / 2

_TIMBER_RGBA = (0.52, 0.36, 0.22, 1.0)
_PLANK_RGBA = (0.78, 0.60, 0.38, 1.0)
_ROPE_RGBA = (0.82, 0.72, 0.52, 1.0)
_KNOT_RGBA = (0.42, 0.34, 0.24, 1.0)

#: The plank's centre at rest, in the frame's coordinates.
_SEAT_Z = BOARD_TOP_Z - BOARD_T / 2


def rope_for(hang_l: float, tie_x: float | None = None):
    """How long each of the four ropes has to be for a hang of `hang_l`.

    The rope runs from a point on the beam to a tie point `tie_x` fore or aft of
    the plank's centre, so it is the hypotenuse and not the drop:

        rope = sqrt(tie_x^2 + hang_l^2)

    Accepts a tensor as well as a float -- `reset_on_the_swing` rigs a whole batch
    at once -- so the square root is taken with `**0.5` rather than `math.sqrt`.

    The one arithmetic a task needs in order to rig this swing at a length other
    than the nominal: write this into `tendon_range[ROPES, 1]` and the seat hangs
    there.
    """
    tie_x = TIE_X if tie_x is None else tie_x
    return (tie_x ** 2 + hang_l ** 2) ** 0.5


def hang_for(rope: float, tie_x: float | None = None):
    """The inverse of `rope_for`: the hang a rope of this length gives.

    `state.rope_length` reads the rigging back out of the model rather than
    remembering what it wrote, and this is the arithmetic that turns a tendon
    limit back into the quantity every other number in the task is expressed in.
    """
    tie_x = TIE_X if tie_x is None else tie_x
    return (rope ** 2 - tie_x ** 2).clamp(min=0.0) ** 0.5 if hasattr(rope, "clamp") \
        else max(rope ** 2 - tie_x ** 2, 0.0) ** 0.5


def seat_drop(hang_l: float) -> float:
    """How far the plank's **centre** rests below the beam at that length.

    `HANG_L` is measured to the top face because that is the surface something
    stands on; the seat body's origin is the plank's centre, half a thickness
    lower, and every pose calculation wants this one instead.
    """
    return hang_l + BOARD_T / 2


def _free_dofs(body: Any, damping: float, ang_damping: float | None = None) -> None:
    """Six degrees of freedom out of ordinary joints, or three out of slides.

    Written as slides and hinges rather than as a freejoint on purpose, and the
    reason is in the module docstring: one freejoint anywhere in the spec makes
    the whole entity floating-base, and a floating-base entity is not
    mocap-wrapped, so the welded frame would sit at the model origin in every
    environment while the seats scattered. Nothing raises -- the swing simply
    comes apart across the batch.
    """
    import mujoco

    for axis, name in (((1, 0, 0), "x"), ((0, 1, 0), "y"), ((0, 0, 1), "z")):
        joint = body.add_joint(
            name=f"{body.name}_slide_{name}",
            type=mujoco.mjtJoint.mjJNT_SLIDE, axis=list(axis),
        )
        joint.damping[0] = damping
    if ang_damping is None:
        return
    # **Yaw, then roll, then pitch, and the order is the whole point.**
    #
    # Three hinges in series are Euler angles, and Euler angles are singular at 90
    # degrees of the **middle** one. Written yaw-pitch-roll, which is the obvious
    # order, that singularity sits on pitch -- the swing's own axis, the one that
    # moves. Measured with random actions on the finished task, `seat_pitch`
    # reaches 65 degrees at an action sigma of 0.35 and **80 at 1.5**, against a
    # singularity at 90; a policy that has learnt to pump goes past it, and what
    # comes out is a NaN in `qpos` that surfaces as "the observation contains NaN"
    # with nothing pointing at the swing. That happened.
    #
    # Roll and yaw do not move: 17.6 and 25.2 degrees at the same sigma of 1.5,
    # because the ropes hold the deck square and resist twist. Putting
    # roll in the middle puts the singularity 72 degrees away from anything the
    # model does, instead of 10.
    #
    # The reset is unaffected: with yaw and roll at zero the composition is still
    # `R_y(theta)` whichever order the last two are in.
    for axis, name in (((0, 0, 1), "yaw"), ((1, 0, 0), "roll"), ((0, 1, 0), "pitch")):
        joint = body.add_joint(
            name=f"{body.name}_{name}",
            type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(axis),
        )
        joint.damping[0] = ang_damping


def _rope(spec: Any, name: str, site_a: str, site_b: str, length: float) -> None:
    """One rope: a spatial tendon that may be shorter than `length` and not longer.

    `range=(0, length)` with `limited` is the whole of what makes this a rope
    rather than a rod. The lower bound is zero, so slack costs nothing; the upper
    bound is one-sided, so the rope pulls and never pushes.

    **Tendons do not collide.** There is no geometry here at all -- MuJoCo draws
    the tendon along its site path and that is the only thing the rope is. Rigid
    hangers would need a private collision channel or an exclude pair here, because
    a hanger hinged *on* the beam starts out inside it and that overlap is a
    permanent seat-frame contact; measured on such a version, a swing released at
    0.35 rad came back through 0.094 and read as too much damping rather than as
    geometry. A rope cannot have the problem.
    """
    import mujoco

    tendon = spec.add_tendon(
        name=name, limited=int(mujoco.mjtLimited.mjLIMITED_TRUE),
        range=[0.0, length], width=ROPE_R, rgba=list(_ROPE_RGBA),
        solref_limit=list(ROPE_SOLREF),
    )
    tendon.wrap_site(site_a)
    tendon.wrap_site(site_b)


def _swing_spec(tie_x: float | None = None) -> Any:
    """The swing as an `MjSpec`: a welded frame, a plank on four ropes.

    ``tie_x`` overrides `TIE_X`, and exists so that the anti-tip claim can have a
    control group. At 0 each side's two ropes lie on top of each other, the
    triangle collapses to a point and the plank flips: measured, 179.8 degrees
    against 6.1 under the same 100 mm off-centre load. A test that cannot build
    that version can only assert that this one does not flip, which any plank
    passes.

    **Fixed base, and that is load-bearing rather than incidental.** The spec has
    no free joint -- see `_free_dofs` -- so mjlab wraps it in a mocap body
    (`utils/spec.py::auto_wrap_fixed_base_mocap`) and `mocap_pos` is per-world,
    which is the only way a welded prop can stand in front of *each* environment
    instead of once at the world origin. `registry._place_props` supplies the
    reset event that writes those poses; without it every swing in the batch would
    be at (0, 0, 0), which is the failure that docstring records measuring on the
    football.

    Only the plank and the frame carry collision, on MuJoCo's defaults
    (`contype=1, conaffinity=1`). The hexapod's legs carry bit 0 (`constants.py`'s
    ground channel) so they collide with both; the frame and the terrain are both
    welded to the world and MuJoCo excludes that pair itself, so the posts standing
    in the ground cost no contacts. The ropes are tendons and have no geometry at
    all, so there is nothing in the rigging for a foot to catch on -- which the
    knots needed `contype=0` to arrange, and which is now true by construction.

    **The plank's contact parameters are not the plank's.** The feet are
    `priority=1` and MuJoCo takes the higher-priority geom's parameters outright,
    so foot-on-plank uses the silicone `friction` / `solref` / `solimp` from
    `constants.py`, exactly as foot-on-ground does. A friction number written here
    would be discarded without a word -- `soft.py` exists because that trap was
    walked into from the other direction.

    Measured, that equivalence holds where it counts. The robot settled on the
    plank and on bare ground gives the **same eight contacts** in the same order
    (mm, negative is penetration):

        LF_palm_pad_b_link  RF_palm_pad_b_link  LM_toe  RM_toe  LR_toe  RR_toe  LF_palm  RF_palm
        plank    -1.04  -1.11   -2.71   -2.79   -2.02   -2.04   -0.28   -0.22
        ground   -1.20  -1.32   -3.83   -4.61   -4.40   -4.84   -0.42   -0.38

    Including the two palms, which touch on both -- that is the standing pose, not
    something the plank does.

    The plank's are consistently shallower, by about 2 mm on the toes, and the
    difference is not the contact model: **the floor yields.** The rig sags 2.12 mm
    under its own weight and 4.76 mm with the robot on it (rope stretch), so a
    couple of millimetres of what
    the foot pushes into rigid ground it instead pushes the plank down by. Standing
    here is standing on a floor that gives, on top of a floor that moves.

    **The posts and the beam are capsules, and a capsule against a mesh is one
    contact point.** mjwarp says so at startup: MULTICCD has no multicontact
    support for that pair, so leaning a leg on a post gives a single point rather
    than a patch. For a thin round pole that is close to right, and nothing here is
    meant to be leaned on; the plank is a box, which is not affected.
    """
    import mujoco

    # Derived here rather than read from the module constant, so that the override
    # reaches the rope lengths too: a tie point that moved without its ropes moving
    # with it is a rig that is pre-stretched at rest, which does not look like
    # anything in particular.
    tie_x = TIE_X if tie_x is None else tie_x
    rope = rope_for(HANG_L, tie_x)

    spec = mujoco.MjSpec()

    frame = spec.worldbody.add_body(name="frame", pos=[0.0, 0.0, 0.0])
    frame.add_geom(
        name="beam", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[BEAM_R, 0.0, 0.0],
        fromto=[0.0, -BEAM_HALF, BEAM_Z, 0.0, BEAM_HALF, BEAM_Z],
        density=FRAME_DENSITY, rgba=list(_TIMBER_RGBA), group=2,
    )
    for sy, side in ((1.0, "l"), (-1.0, "r")):
        for sx, end in ((1.0, "f"), (-1.0, "b")):
            frame.add_geom(
                name=f"post_{side}{end}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[POST_R, 0.0, 0.0],
                fromto=[0.0, sy * FRAME_Y, BEAM_Z,
                        sx * FRAME_SPREAD, sy * FRAME_Y, 0.0],
                density=FRAME_DENSITY, rgba=list(_TIMBER_RGBA), group=2,
            )
        # **Both of this side's ropes leave this one point**, which is what makes
        # the pair a triangle rather than a parallelogram. Two beam points fore and
        # aft of each other would hold the plank *level* at every point of the arc
        # -- a hanging basket, not a swing -- and the robot would then stand
        # upright while the swing moved, so its own IMU would report nothing about
        # the swing at all. The whole actor observation depends on this.
        frame.add_site(name=f"hang_{side}", pos=[0.0, sy * ROPE_Y, BEAM_Z],
                       size=[ROPE_R, 0.0, 0.0], group=4)

    # The seat's body origin is the plank's centre, so its three hinges rotate the
    # plank about itself and every offset below is read straight off the plank.
    seat = frame.add_body(name="seat", pos=[0.0, 0.0, _SEAT_Z])
    _free_dofs(seat, SEAT_DAMPING, SEAT_ANG_DAMPING)
    seat.add_geom(
        name="board", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[BOARD_X / 2, BOARD_Y / 2, BOARD_T / 2],
        density=BOARD_DENSITY, rgba=list(_PLANK_RGBA), group=2,
    )

    # Four ropes, beam to plank, nothing in between. The seat is the only moving
    # body in the rig -- six degrees of freedom, four one-sided constraints.
    for sy, side in ((1.0, "l"), (-1.0, "r")):
        for sx, end in ((1.0, "f"), (-1.0, "b")):
            seat.add_site(
                name=f"tie_{side}{end}",
                pos=[sx * tie_x, sy * ROPE_Y, BOARD_T / 2],
                size=[ROPE_R, 0.0, 0.0], group=4,
            )
            _rope(spec, f"rope_{side}{end}", f"hang_{side}", f"tie_{side}{end}", rope)

    return spec


def swing_entity() -> Any:
    """The prop, as an mjlab entity.

    **Public, unlike the football's ball.** `tasks/jumper/swing/` builds the swing
    itself rather than requiring `--scene swing`, because `--scene` is optional by
    construction and a task that silently trains without its apparatus is exactly
    the kind of failure this repository writes tests about. So this is the shared
    definition the scene and the task both call, and there is one swing described
    in one place.

    `joint_pos` is given explicitly rather than left as None, which would demand a
    keyframe in the spec. Every joint's zero **is** the rest rigging -- the plank
    under the beam with all four ropes just taut -- so this and the batched reset
    agree: `mjwarp.reset_data` zeroes every `qpos` to the joint's `qpos0` before
    any event runs, and that is the same pose. A task that wants the swing already
    moving has to add its own reset event on the seat's joints; setting a value
    here and expecting a head start is the kind of config that reads correct and
    does nothing.
    """
    from mjlab.entity import EntityCfg

    return EntityCfg(
        spec_fn=_swing_spec,
        init_state=EntityCfg.InitialStateCfg(
            # Relative to each environment's own origin -- entities are placed per
            # environment, so this is not a position in the world.
            pos=(SWING_X, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
    )


def scene() -> Scene:
    """Grass, a summer sky and one high sun.

    Built from MuJoCo's builtin patterns only, so the scene is one file with no
    images to keep in step with it. The sun is high rather than low -- the swing is
    a tall object and a low sun lays its posts across the whole ground plane as
    shadows, which reads as clutter next to a robot 0.1 m tall.
    """
    from mjlab.utils import spec_config as sc

    return Scene(
        headlight=Headlight(ambient=(0.16, 0.17, 0.16), diffuse=(0.28, 0.29, 0.28)),
        textures=(
            sc.TextureCfg(
                name="swing_grass", type="2d", builtin="checker", mark="edge",
                rgb1=(0.34, 0.45, 0.26), rgb2=(0.29, 0.40, 0.22),
                markrgb=(0.40, 0.50, 0.31), width=512, height=512,
            ),
            sc.TextureCfg(
                name="swing_sky", type="skybox", builtin="gradient",
                rgb1=(0.33, 0.55, 0.85), rgb2=(0.83, 0.90, 0.97),
                width=512, height=512,
            ),
        ),
        materials=(
            sc.MaterialCfg(
                name="swing_grass", texture="swing_grass", texuniform=True,
                # One check per 0.25 m: fine enough to read as mown grass at the
                # robot's scale rather than as a chessboard under the swing.
                texrepeat=(4.0, 4.0), reflectance=0.0, geom_names_expr=GROUND,
            ),
        ),
        lights=(
            sc.LightCfg(
                name="sun", type="directional", pos=(-2.0, -3.0, 5.0),
                dir=(0.35, 0.5, -1.0), castshadow=True,
                diffuse=(0.98, 0.95, 0.88), specular=(0.25, 0.25, 0.25),
                ambient=(0.20, 0.24, 0.28),
            ),
        ),
        props={"swing": swing_entity()},
    )
