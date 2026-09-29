"""The carried left-front claw: which joints it owns, and the pose it is held in.

A module of its own that **does not import mjlab or torch**, for the same reason
`common/assets.py` does not: the numbers here are read by
`tasks/jumper/five_foot/tools/grasp_pose.py`, which is a plain MuJoCo script and must not
drag the training stack in behind it. Everything that needs a tensor lives in
`mdp/`.

## What "five foot" means mechanically

The left-front leg is a 5-DoF arm, and this task takes it **out of the action**
and holds it stowed against the trunk as a claw, so the policy walks on the other
five feet; on the robot the d-pad swings it out (`deploy/lib.rs`). Three things
follow, and each is a separate mechanism:

- the action drops from 20 joints to :data:`FIVE_FOOT_JOINTS` (16);
- the five LF joints are held by their own PD at :data:`LF_GRASP`, re-sampled per
  episode inside :data:`LF_GRASP_BOX`, by a reset event -- **not** by the home
  pose, see the note on `joint_pos_target` below;
- the left-front leg stops being a foot: :data:`FIVE_FOOT_LEGS` is what the
  contact sensor, the height scan and every foot reward term are built from.

## The trap: an un-driven joint is driven to zero, not held at home

mjlab clears `joint_pos_target` to **0.0** on every reset
(`EntityData.clear_state`), and only the action term writes it afterwards -- for
the joints it drives. A joint taken out of the action is therefore *not* left at
its home angle; it is commanded to 0.0 rad and its PD takes it there. Measured on
`jumper.flat`, whose two grippers are already out of the action: `LF_J4_joint`
starts at HOME's -1.0 and reaches -0.0001 within 25 control steps (0.5 s). That
was the model before V1.6; V1.6's HOME has the fingers at 0.0, which hides the
trap on the finger and not on the four arm joints this task takes out.

So holding the arm needs a reset event that writes the *target*, not only the
state. `mdp/events.py::hold_carried_arm` does both; `tests/test_five_foot.py`
pins it, because the failure mode is a claw that quietly folds to zero over the
first half second of every episode while everything else looks healthy.
"""

from __future__ import annotations

from ..common.constants import FEET, GAIT_JOINTS, LEGS

#: The leg carried as a claw rather than walked on.
CARRIED_LEG = "LF"

#: The five legs that carry the robot, in `LEGS` order with `CARRIED_LEG` removed.
#:
#: This is the tuple `env_cfg.py::_ground_the_five_legs` narrows to, and therefore
#: the column order of every per-foot quantity in this task: contact state, foot
#: height, air time, contact forces, and the gait reward's leg indices
#: (`mdp/rewards.py::GROUP_A` / `GROUP_B` are indices into *this* tuple, not into
#: `LEGS`).
FIVE_FOOT_LEGS: tuple[str, ...] = tuple(leg for leg in LEGS if leg != CARRIED_LEG)

#: Leg prefix -> the leaf body carrying that leg's contact.
#:
#: `FEET` is written in `LEGS` order, and pairing them here is what lets this task
#: name the legs that carry the robot and get the matching geoms back (see
#: `foot_geoms_for`). The foot **site** needs no table: `build_jumper.py` names it by
#: the leg prefix alone.
LEG_FOOT: dict[str, str] = dict(zip(LEGS, FEET, strict=True))


def foot_geoms_for(legs: tuple[str, ...]) -> tuple[str, ...]:
    """The colliding foot geoms of `legs`, in the order given.

    For a task where not every leg is on the ground -- `jumper.five_foot` carries
    the left-front arm as a claw -- the contact sensor, the foot height scan and
    the foot reward terms all have to describe **the same** set of legs, in the
    same order, or the per-foot columns of one silently address the feet of
    another. Deriving all three from one leg tuple is what keeps them aligned;
    `env_cfg.py::_ground_the_five_legs` is where this task applies it.
    """
    unknown = [leg for leg in legs if leg not in LEG_FOOT]
    if unknown:
        raise KeyError(f"not legs of this robot: {unknown}; known: {list(LEG_FOOT)}")
    return tuple(f"{LEG_FOOT[leg]}_meshcol" for leg in legs)


#: Half the trunk's lateral extent, in metres -- the edge of the body's own
#: footprint, seen from above.
#:
#: Measured off `assets/jumper/jumper.xml` rather than written down: the visual mesh's
#: vertices span y in [-0.0907, +0.0907], the collision mesh [-0.0884, +0.0904],
#: and `base_link_collision`'s box half-size agrees at 0.0907. All three within
#: 2 mm, so the number is the body and not one representation of it.
#:
#: What it is for: deciding whether a foothold is *under* the machine or outside
#: it. Measured against `HOME_TOE_XY`, four of the five ground feet are well
#: outboard (|y| of 200, 200, 161 and 161 mm) and **RF is not** -- its foothold
#: sits at y = -36.5 mm, forty percent of the way in from the body's edge. On five
#: feet that is the whole front of the machine standing inside its own shadow,
#: which is the same fact as the -17.1 mm support margin `grasp_pose.py --stance`
#: reports with RF airborne.
TRUNK_HALF_WIDTH = 0.0907

#: The carried arm's joints, shoulder outwards. The finger is listed separately
#: because it is the claw's aperture rather than part of the arm's pose.
ARM_JOINTS: tuple[str, ...] = (
    "LF_J0_joint",
    "LF_J1_joint",
    "LF_J2_joint",
    "LF_J3_joint",
)
FINGER_JOINT = "LF_J4_joint"

#: Every joint of the carried leg: the four arm joints plus the finger.
CARRIED_JOINTS: tuple[str, ...] = ARM_JOINTS + (FINGER_JOINT,)

#: The 16 joints the policy drives: the family's 20 gait joints less the carried
#: arm's four. (The two fingers are already out of `GAIT_JOINTS`.)
FIVE_FOOT_JOINTS: tuple[str, ...] = tuple(
    j for j in GAIT_JOINTS if not j.startswith(f"{CARRIED_LEG}_")
)

#: The 21 joints the observation carries: the 16 driven ones, the carried arm's
#: four, and the claw's finger.
#:
#: **Wider than the action on purpose, and this is where the port of kk-rl-lab's
#: `gripper` command observation went.** There, the claw's closure is a third
#: command term next to velocity and body pose, so the policy is told what the
#: operator is about to do with the claw. That does not survive the trip: the
#: deployment vocabulary is fixed on the far side of the contract
#: (`scripts/export.py::_DEPLOY_TERMS`, and `deploy/fsm/src/obs.rs`, which
#: builds observation terms and command channels by name), so a `gripper` term
#: would fail the export rather than reach the robot.
#:
#: The measurable half of it does survive, and it is the half that matters here:
#: the arm's angles and the claw's aperture are **encoder readings**, so putting
#: them in `joint_pos` / `joint_vel` / `actuator_force` gives the policy the same
#: information through a term the contract already knows and the controller
#: already builds. What is given up is anticipation -- the policy reacts to a
#: claw that has started to close rather than to a closure that has been
#: commanded. For a *locomotion* policy that is the smaller half: the payload it
#: has to walk against is a load, and the load is what the joint angles and
#: actuator forces report.
#:
#: `RF_J4_joint` stays out. It is the other front leg's claw, held shut in
#: FOOT mode, and a constant column is what the note in
#: `common/mdp/observations.py::actuator_force` warns against.
OBSERVED_JOINTS: tuple[str, ...] = FIVE_FOOT_JOINTS + CARRIED_JOINTS

# ── The claw's aperture ───────────────────────────────────────────────────
#
# **The fixed jaw is `LF_palm_link`** -- the hooked upper jaw, with the serrated
# face an object is pinched against. `LF_palm_pad_b_link` is the small pad on the
# *outside* of its tip, which is the foot the robot walks on when this leg is a
# leg. Saying so is half of the correction below: this table used to be measured
# from the moving jaw to `LF_palm_pad_b_link` alone, i.e. to a pad on the far side of the
# jaw the finger is closing on, with the vertices read against the body frame
# rather than the geom frame on top of that (`grasp_pose.py::Claw._mesh` has what
# that cost). It reported the claw shutting at finger -0.30, which it does not do.
#
# The other half is **where along the mouth to measure**. The mouth is a wedge,
# narrow at the knuckle and wide at the tips, so the closest pair of the two jaws
# is the throat -- a place nothing is ever held. The aperture here is the gap at
# the **anvil** instead: the point on the fixed jaw that the moving jaw arrives
# at when the claw shuts, which is where an object pushed into the claw ends up
# and therefore how thick an object can be and still be pinched.
# `grasp_pose.py::Claw.anvil` derives it and `mdp/grasp.py::_mouth_in_palm` is
# the same point in the palm frame.
#
# Re-measured on the V1.6.1 `assets/jumper/jumper.xml` (`grasp_pose.py --aperture`;
# both jaws hang off the palm, so the arm's pose does not enter). The sweep now
# runs the joint's own range rather than stopping at 0.0, which is what showed
# that shut had moved:
#
#     finger  -0.65  -0.55  -0.45  -0.35  -0.25  -0.15  -0.05  +0.05  +0.10
#     mouth    73.3   63.8   54.2   44.5   34.7   24.7   14.6    4.5    0.4  mm
#
# Monotone the whole way and it reaches zero at the joint's own upper limit, so
# **the finger's full travel is the claw's full travel**: there is no angle at
# which the jaws are shut with travel to spare, and none at which they slide past
# one another. A least-squares line over it is -98.2 mm/rad and 0.49 mm out at its
# worst, against 1.5 mm on V1.6 -- straighter, and still kept as a table.
#
# **This claw is a bigger claw, and shut moved with it.** Both jaw parts changed:
# against V1.6, `LF_palm_link` (was `LF_J3_link`) is 3.18 mm RMS and 10.98 mm peak
# away from it and `LF_finger_link` (was `LF_J4_link`) is 4.00 mm and 13.29 mm,
# measured vertex-against-surface on 400k samples where a retriangulation of the
# same shape reads 0.2 mm. `LF_finger_tip_link` did not change (0.14 mm, the noise
# floor). The hinge barely moved -- the child origin is 0.4 mm in y and 1.4 mm in z
# from V1.6's, with the same `1 0 0` axis and the same `(-1.6, 0.1)` range.
#
# Cross-checked **without** the anvil, by the minimum distance between the two
# jaws' own meshes (hinge ball excluded), one piece of code run on both exports, so
# the two columns mean the same thing:
#
#     finger    V1.6 gap   V1.6.1 gap
#     -0.90     48.06 mm     55.44 mm
#     -0.10      7.91         19.67
#      0.00      0.12 <- shut  9.57
#     +0.08      0.08          1.45
#     +0.10      0.07          0.20 <- shut, and the URDF's own upper limit
#
# On V1.6 the jaws met at 0.0 and the remaining 0.10 rad was travel against a jaw
# already in contact -- `upper` was 0.10 rad past the mechanical stop. On V1.6.1 the
# stop and `upper` coincide, which is the export getting *better*; what it breaks is
# the number written here, because 0.0 is no longer shut. The mouth is wider
# everywhere too -- 93.8 mm at -0.90 against 75.7 at the anvil -- so every angle in
# the old table means about 18 mm more opening than it used to.
#
# **The open end is limited by the closing axis, not the aperture.** The mouth
# keeps widening to 121.6 mm at the -1.60 stop, but the nearest part of the moving
# jaw stops being the face across the mouth and becomes its edge, and the closing
# axis jumps with it. On V1.6.1 that happens between -0.76 and -0.75 -- a
# discontinuity, not a swing. Measured in `LF_palm_link`'s frame, where the arm
# does not enter (`grasp_pose.py --aperture` prints the whole column):
#
#     finger   aperture   closing axis, palm        off across
#     -0.60     68.6 mm   [ 0.050, -0.999,  0.002]     2.9 deg
#     -0.65     73.3      [ 0.030, -1.000,  0.003]     1.7       <- GRIPPER_OPEN
#     -0.72     79.8      [ 0.001, -1.000,  0.003]     0.1       <- squarest
#     -0.75     82.6      [ 0.012, -1.000,  0.003]     0.7       <- the widest usable
#     -0.76     83.5      [ 0.221, -0.978,  0.143]    12.9       <- past the turn
#     -0.90     93.8      [ 0.380, -0.915,  0.134]    22.5
#
# The closing axis is the direction the jaws pinch along and it should be the
# claw's y -- side to side, so an object walked into the mouth is caught across
# it. Past the turn the claw is closing 13 to 23 degrees front-to-back on whatever
# is beside it rather than across the mouth.
#
# **`GRIPPER_OPEN` is -0.65 rather than the widest -0.75**, which backs off from
# that turn by 0.10 rad -- the same 0.10 rad `GRASP_BOX` randomises the hold by,
# i.e. the amount by which anything about this pose is allowed to be off. It costs
# 10 mm of mouth out of 83, and the widest thing in `tasks/jumper/five_foot/objects.py` that
# is meant to be gripped is the 58 mm can, which still clears by 15 mm. V1.6
# turned at -0.99 and opened to -0.90; this claw turns 0.24 rad sooner.
#
# NOTE the six-foot tasks are a different regime and their numbers do not
# transfer: there the claw is a **foot**, the finger is out of the action, and
# mjlab's zeroed `joint_pos_target` walks it to 0.0 and holds it there -- which on
# V1.6 was the claw held shut and on V1.6.1 is 9.6 mm of mouth still open. The
# front feet are `LF_palm_pad_b_link` either way and `NOMINAL_FOOT_XY` was measured
# at HOME, where the finger is 0.0, so the footprint already accounts for it.
GRIPPER_OPEN = -0.65
GRIPPER_CLOSED = 0.10

#: How far from the finger's hinge a surface has to be before it counts as the
#: mouth, in metres.
#:
#: The moving jaw and the palm are two halves of one hinge, so they touch at
#: every aperture and the unfiltered closest pair of the two jaws is 0.0 mm
#: always -- which would put the anvil on the hinge. Excluding a ball around it
#: is what leaves the pinching faces. Not a tuning knob: measured, the anvil
#: comes out at the same point for any exclusion from 30 mm out to 80, and only
#: below 30 does the hinge win. Shared with `tasks/jumper/five_foot/tools/grasp_pose.py` and
#: `mdp/grasp.py`, which both measure the same mouth.
KNUCKLE_CLEAR = 0.040

#: The measured aperture curve, `(finger angle, mouth in mm)`, open to shut.
#:
#: A **table rather than a slope**, because the curve is not quite a straight
#: line: on V1.6.1 a least-squares fit over the travel is -98.2 mm/rad and
#: 0.49 mm out at its worst (V1.6: 84 mm/rad, 1.5 mm), which is a fair share of a
#: small object on a readout whose job is to be held up against the object on the
#: floor.
#:
#: Regenerate with `python tasks/jumper/five_foot/tools/grasp_pose.py --aperture`.
APERTURE_MM: tuple[tuple[float, float], ...] = (
    (-0.65, 73.3), (-0.60, 68.6), (-0.55, 63.8), (-0.50, 59.1), (-0.45, 54.2),
    (-0.40, 49.4), (-0.35, 44.5), (-0.30, 39.6), (-0.25, 34.7), (-0.20, 29.7),
    (-0.15, 24.7), (-0.10, 19.7), (-0.05, 14.6), (0.00, 9.6), (0.05, 4.5),
    (0.10, 0.4),
)


def aperture_mm(finger: float) -> float:
    """The mouth opening at a finger angle, in millimetres, from `APERTURE_MM`.

    Linear between the measured points and clamped to the ends, which is what a
    readout wants: outside [GRIPPER_OPEN, GRIPPER_CLOSED] there is no mouth to
    report, only the far side of the turning point.
    """
    from itertools import pairwise

    if finger <= APERTURE_MM[0][0]:
        return APERTURE_MM[0][1]
    for (x0, y0), (x1, y1) in pairwise(APERTURE_MM):
        if finger <= x1:
            return y0 + (y1 - y0) * (finger - x0) / (x1 - x0)
    return APERTURE_MM[-1][1]


# ── How the claw squeezes ─────────────────────────────────────────────────
#
# **A position-controlled claw grips with the error between where it was told to
# go and where the object stopped it**, and that is the whole of it now that
# `GRIPPER_CLOSED` is the joint's own limit rather than an angle short of it.
# Read it off the table: a 42 mm object stops the finger at -0.41 and leaves
# 0.41 rad of overdrive; a 10 mm one stops it at -0.09 and leaves 0.09. The
# thinner the object the less squeeze is left, which is the right way round --
# a thin object needs less.
#
# There used to be a third constant here, `GRIPPER_SQUEEZE = 0.0`, for exactly
# this: travel past a `GRIPPER_CLOSED` that was 0.30 rad short of shut. With the
# aperture measured correctly the two are the same number and the extra name says
# nothing, so a trigger squeezed all the way drives to `GRIPPER_CLOSED` -- and what
# bounds the squeeze is `SQUEEZE_LEAD`, below.

#: How far past its measured angle the finger may be told to go, in radians: the
#: squeeze limit. Stopped by an object, the finger's servo pushes with
#: `STIFFNESS * SQUEEZE_LEAD`, 0.6 N*m, however hard the trigger is held.
#: Opening is never limited, and a finger closing on nothing is only led by it.
#: `mdp/gripper.py::squeeze_limited` applies it in replay, and the task's
#: `deploy/lib.rs` on the robot.
#:
#: **Why a limit at all: the policy sees the finger's torque.** `LF_J4_joint`
#: is an observed joint, and in training it only ever held a sampled aperture with
#: nothing in it -- the normaliser of `2026-09-11_15-11-16/model_6999` has its
#: torque at 0.014 N*m with a standard deviation of 0.056. Pressed shut on the
#: can, the servo saturates at 1.2 N*m, 21 standard deviations out, and the turn
#: comes apart. Measured on warp with the can in the claw, walking back 0.6 m and
#: turning 57 degrees:
#:
#:     finger torque                         turned        walked back   can slipped
#:     none, empty claw                       100%             93%            -
#:     1.2 N*m, no limit                   50, 49, 42%       84-109%       6-15 mm
#:     1.2 N*m, policy fed 0.014           86, 90, 88%        84-90%          5 mm
#:     1.0 N*m, lead 0.10                    76, 59%         97, 120%      8, 13 mm
#:     0.8 N*m, lead 0.08                    83, 80%        108, 109%       7, 7 mm
#:     0.6 N*m, lead 0.06                    95, 95%         96, 101%       6, 7 mm
#:     0.4 N*m, lead 0.04                    84, 93%          85, 97%       5, 5 mm
#:
#: The row fed the training value is the one that says why: the same squeeze with
#: the torque hidden from the policy turns nearly as well as an empty claw, and the
#: rest of the way is the can's weight. 0.06 is the best turn measured, and at it
#: the soap turned 87% and slipped 7.5 mm, the notebook 93% and 4.8 mm.
#:
#: **What it costs the grip.** Through the teleop itself -- `]` pressed twenty
#: times in the keyboard teleop of the time, the trigger squeezed all the way now
#: -- 0.06 carried the can in five runs of six, turning 89 to 97%, and the soap and
#: the notebook in one run each, turning 98 and 96%; unlimited, the same run
#: turned 64%. The one can it lost ended on the floor, as did one of four at 0.08,
#: which fell while walking after closing at -0.66 where the others stopped at
#: -0.73 to -0.78 -- a can taken nearer the tips. Squeezed at the servo's full
#: torque the can was not dropped in nine runs, so the limit trades a little grip
#: for the turn, and the grip it gives up is on objects taken shallow.
#:
#: **A lead rather than a number of presses past contact.** The target follows the
#: finger, so the squeeze holds as an object settles. An operator who stopped on
#: the press the finger stalled on left a lag that closed as the can shifted, and
#: the can fell out of the claw walking.
SQUEEZE_LEAD = 0.06

#: Seconds of the finger's closing speed added to `SQUEEZE_LEAD` while it closes:
#: the lead is `SQUEEZE_LEAD + LEAD_PER_SPEED * max(speed, 0)`. Closing was
#: reported too slow on 2026-09-29: the lead alone pulls a finger closing on
#: nothing with the squeeze's 0.6 N*m against the servo's damping (`kd` 0.5).
#: MEASURED in replay (native, one env, the trigger squeezed all the way at once),
#: the trained 43 degrees from open to within 0.02 rad of shut:
#:
#:     LEAD_PER_SPEED     0      0.05    0.08    0.12    (no lead at all)
#:     seconds          0.80    0.40    0.30    0.28        0.22
#:
#: 0.08 is where more stops buying time. Stopped by an object the finger's speed
#: is zero and the lead is `SQUEEZE_LEAD` again: squeezing the can (`play --hold
#: can`), the finger's torque over the last second of three was 0.600 N*m
#: without the term and 0.601 with it (at most 0.612), so the table above holds.
#: `mdp/gripper.py` applies it in replay and the task's `deploy/lib.rs` on the
#: robot, from the finger's measured speed.
LEAD_PER_SPEED = 0.08

# ── The hold pose ─────────────────────────────────────────────────────────
#
# **Stowed, not grasp-ready, since 2026-09-28.** This is kk-rl-lab's `LF_RAISED`
# at 25d8a77 (`tasks/hexa_5d/locomotion_5foot/velocity_env_cfg.py` there), and it
# is the pose the robot has actually walked under: rl-wbc-fsm's
# `locomotion_5foot` policy is this task's model_6050, trained on the
# `old-pose-five-foot` branch (4a7004e) at exactly these angles. It replaces the
# grasp-ready pose this constant held before (refined on V1.6.1 in a82e0f5, whose
# message and the history of this comment have the derivation), in place and
# under the old name, as 4a7004e did: **the name is historical**, the pose is a
# stow. Reaching for something moves to the robot: `deploy/lib.rs` swings the arm
# out to a d-pad preset -- rl-wbc-fsm's `[arm]` presets -- and the trigger closes
# the claw.
#
# Measured on V1.6.1, robot standing at STAND_Z with the other five legs at HOME
# (`grasp_pose.py --report`), V1.6's numbers from 4a7004e beside it:
#
#                     V1.6.1                      V1.6
#     mouth centre    (0.218, 0.033, -0.025) m    (0.203, 0.040, -0.030)
#     closing axis    (-0.90, -0.43, +0.03)       (-0.96, -0.27, -0.03)
#                     front-to-back: a stow, not a mouth to walk into
#     lift            159 mm/rad of shoulder      150
#     clearance       26.2 mm forearm outwards    26.5
#                     62.3 mm above the floor     62.8
#                     14.6 mm of corridor         5.1
#     joint margin    J0 66%, J1 52%, J2 43%,     18-66%
#                     J3 18% of the soft range
#
# So what the grasp checks measure no longer applies to the hold: run at this
# pose, `tools/grasp_objects.py` grasps none of the soap, the notebook or the can
# (native, 2026-09-28), and the `--objects` row, laid out for the old mouth, is
# not walked into. The props now reach the claw at a d-pad preset, on the robot.
#
# **The finger is not stowed with it.** `old-pose-five-foot` pinned it shut, as
# 25d8a77 does. Here it keeps `GRIPPER_OPEN` as its nominal and keeps the
# per-episode aperture sampling (`env_cfg.py`), because the operator closes it
# with the trigger once the arm is out at a preset, and a policy that has never
# walked with the claw at another aperture is one the trigger would surprise.
#
# **First deployed without retraining.** The export this replaced on the robot
# was trained at the grasp-ready pose. Re-exported against this one, the
# contract's `default_joint_pos` moves with it, and `joint_pos` is observed
# relative to that -- so the policy sees the stowed arm where it saw the old one,
# give or take the same box. The observation is in distribution; the mass
# distribution is not: J3 is 1.54 rad and J0 0.36 rad from where that policy
# trained. A retrain at this pose is what closes it.
LF_GRASP: dict[str, float] = {
    "LF_J0_joint": -0.5236,    # -30.0 deg
    "LF_J1_joint": -1.5708,    # -90.0 deg
    "LF_J2_joint": -0.5236,    # -30.0 deg
    "LF_J3_joint": -1.3090,    # -75.0 deg
    FINGER_JOINT: GRIPPER_OPEN,  # -37.2 deg, the claw's nominal; see above
}

#: Half-width of the per-episode randomisation of the hold, in radians.
#:
#: The arm is out of the action but **not** at one fixed angle: each episode it is
#: locked somewhere in this box, so the policy learns to walk with the arm
#: roughly rather than exactly there.
#:
#: Measured over all 16 corners around the stowed `LF_GRASP` on V1.6.1
#: (`grasp_pose.py --check-box`), swept:
#:
#:     box(rad)  box(deg)   forearm   shoulder    floor     lift      soft limit
#:      0.06       3.4       26.0 mm    5.5 mm   58.4 mm   156 mm/rad    ok
#:      0.10       5.7       17.3       5.4      53.3      152           ok
#:      0.15       8.6        3.3       3.3      46.2      146           ok
#:
#: **At the stow the forearm binds**, not the lift that bound the grasp-ready
#: pose: the arm is folded back towards the trunk, and 0.15 rad leaves 3 mm. 0.10
#: is kept because it is the width every policy of this task has trained under --
#: the one first deployed at this pose included, whose arm observations therefore
#: spread as they did in training -- and it keeps 17 mm, against the 26.2 mm of
#: the nominal pose and 5.9 mm from the shoulder outwards at the robot's own
#: six-foot HOME.
GRASP_BOX = 0.10

#: The sampling box itself, arm joints only. The finger is excluded: its angle is
#: the claw's aperture and is sampled from its own range (see `mdp/events.py`).
LF_GRASP_BOX: dict[str, tuple[float, float]] = {
    j: (LF_GRASP[j] - GRASP_BOX, LF_GRASP[j] + GRASP_BOX) for j in ARM_JOINTS
}

#: What the claw may be holding, in kg, sampled per environment at startup.
#:
#: A point mass added at `PAYLOAD_BODY`'s centre of mass -- which is what
#: `dr.body_mass` models, and the one case its own warning says it is appropriate
#: for. Measured on V1.6 at the grasp pose (`grasp_pose.py --stance`), with the
#: load 187 mm from the shoulder-pitch axis, against a five-foot support margin
#: of 92.3 mm:
#:
#:     payload   CoM shift   support margin   shoulder hold   of plateau   of continuous
#:      100 g     +7.7 mm        84.3 mm       0.183 N*m        10.5%          15.3%
#:      300 g    +21.5 mm        69.8 mm       0.549 N*m        31.4%          45.8%
#:      600 g    +39.2 mm        51.2 mm       1.099 N*m        62.9%          91.6%
#:
#: So the whole range is statically safe on five feet; what it costs is servo
#: headroom and forward centre of mass, which is what the policy has to learn to
#: walk against.
#:
#: **Two ceilings, and the second is the binding one for a carried load.** The
#: servo's measured plateau is 1.7464 N*m but it is a *peak* rating holdable for
#: 300 ms; past that the thermal derate takes the ceiling to 1.2 N*m
#: (`common/actuator.py`). Holding a payload is not a transient, so the column to
#: read is the last one: 600 g asks the shoulder for **92% of what it can hold
#: indefinitely**, before the arm has done anything but exist (74% on the model
#: before V1.6, whose claw held the load 37 mm nearer the shoulder). That is the real
#: argument for treating the top of this range as an upper bound rather than a
#: comfortable operating point, and it is an argument the 2.0 N*m figure this was
#: originally written against could not make.
#:
#: **And the payload is not the joint to watch.** Measured on V1.6 on the settled
#: five-foot stance with an empty claw, 64 environments, 250 steps of zero action,
#: native CPU:
#:
#:     RF_shoulder_pitch  0.967 N*m   80.6% of continuous
#:     LM_knee            0.538       44.9%
#:     RM_knee            0.438       36.5%
#:
#: The carried arm's own shoulder is not in the top eight -- holding it out is
#: cheap. What costs is that **the front of the machine rests on one leg**, and
#: RF_shoulder_pitch is at four fifths of its continuous rating before the claw
#: has picked anything up. The two do not stack on one actuator (the payload loads
#: the LF shoulder, this is RF), but a payload moves the centre of mass forward,
#: which loads this joint further -- so the real budget for `PAYLOAD_RANGE` is set
#: here, not by the table above, and it has not been measured under load.
#:
#: **This is the second rung of a curriculum, and never the first.** kk-rl-lab
#: measured what turning it on for a run that starts from random weights does:
#: 8000 iterations, mean episode length 990 of 1000 and no falls, and 0.001 m/s of
#: travel for a 0.5 m/s command -- the policy found the local optimum of standing
#: perfectly still, which is the same basin this robot's whole reward design is
#: arranged against.
#:
#: **That was reproduced here, twice, and it is not a warning to read past.** Two
#: cold starts at 4096 environments on warp/CUDA, both with the payload on:
#:
#:     run   iterations   travel in 12 s at a commanded 0.30 m/s
#:      1       833        (stopped -- mean reward flat from 319)
#:      2      1014        0.004 m, against 3.60 m asked for
#:
#: Both stood still and lifted their feet in place: peak foot height reached
#: 16 mm and `duty_balance` never left 0.06 of its 1.5. The stage-one checkpoint
#: that this range was then resumed *into* travels 3.31 m over the same 12 s, 92%
#: of the command.
#:
#: It was a second stage -- train empty, set this, resume -- and this constant sat
#: at the second stage's value, so a fresh run on master was the cold start
#: above. It is now `mdp/curriculum.py::PayloadCurriculum`'s top rung: the claw
#: is empty until the policy walks at the top command range, and then every
#: environment is loaded from this range at its next reset, in one run. Changing
#: it changes the load the policy ends up under; it no longer decides whether a
#: run starts loaded (a resume takes that from the checkpoint, or from
#: `MJRL_PAYLOAD_LEVEL`).
#:
#: **What the load does to the hold.** Measured on V1.6, 16 environments, 100
#: steps of zero action, native CPU, seed 0, the training config with pushes off,
#: as the carried arm's drift from where the reset event put it:
#:
#:     payload     worst arm drift        mean      of which LF_shoulder_pitch
#:      0 g          0.0133 rad (0.8 deg) 0.0115    0.013  (i.e. all of it)
#:      0-300 g      0.0370     (2.1)     0.0254    0.037
#:      0-600 g      0.0594     (3.4)     0.0371    0.059
#:
#: So the top of the range droops the claw by 3.4 degrees, about 5 mm of anvil at
#: the 79 mm per radian `grasp_pose.py --report` measures, and it is
#: `LF_J1_joint` doing all of it -- the joint the stance measurements
#: below already single out. **That droop is more than half of `GRASP_BOX`**,
#: which randomises the hold by +-0.10 rad on purpose: a full payload moves the
#: claw most of the way the deliberate randomisation does. It is PD
#: sag rather than a hold that is failing, and it is exactly what the policy is
#: being asked to learn to walk against -- but a grasp planned against `LF_GRASP`
#: is aiming at a mouth that is not there once the jaws are full.
#: The body the payload's mass and inertia go on -- **added by `add_payload_body`
#: below**, and not a link of the robot.
#:
#: It used to be `LF_finger_link`, with `dr.body_mass` adding a point mass at that
#: link's own centre of mass, and three things were wrong with it at once:
#:
#:   - **the mass was in the wrong place.** A held object sits in the mouth; the
#:     finger link's centre of mass is 46 mm behind and 21 mm to the side of it.
#:     What a carried load does to this robot is almost entirely its moment arm
#:     about the shoulder, so the arm is the thing that has to be right.
#:   - **it rode on the moving jaw.** Open the claw and the load swung with the
#:     finger. A held object is pinched against the *fixed* prong -- the moving
#:     one only presses it there -- so the load must not move when the claw does.
#:   - **it had no inertia.** `dr.body_mass` says so in its own warning: it
#:     changes mass and leaves the inertia tensor alone, which is right for a
#:     point mass and wrong for anything 200 mm tall. Nothing the trunk did to the
#:     load, or the load to the trunk, cost anything in rotation.
#:
#: `mdp/events.py::claw_payload` now writes mass **and** a matching inertia onto a
#: body at the centre of a held object, so the load is where the claw actually
#: holds one and weighs what it weighs in every axis.
PAYLOAD_BODY = "LF_payload"
PAYLOAD_RANGE: tuple[float, float] = (0.0, 0.6)

#: What the claw is carrying, as an upright solid cylinder: radius and height, in
#: metres. `PAYLOAD_UNIT_INERTIA` is its inertia per kilogram, and
#: `add_payload_body` puts the load where its centre is when it is held.
PAYLOAD_CYLINDER: tuple[float, float] = (0.0325, 0.200)

#: Inertia per kilogram of what the claw is carrying, about the payload's own
#: centre of mass and in its own frame, in kg*m^2/kg.
#:
#: `PAYLOAD_CYLINDER`, 65 mm across and 200 mm tall standing in the jaws -- sized
#: from the bottle the object row used to carry, 550 g and the top of
#: `PAYLOAD_RANGE`. `(3r^2 + h^2)/12` across the axis and `r^2/2` about it, so at
#: 600 g that is 2.16 g*m^2 across and 0.32 about -- against zero before, which
#: is what a point mass contributes.
#:
#: One shape for every load rather than one per object, because what this models
#: is "the claw is carrying something", and the objects in `tasks/jumper/five_foot/objects.py`
#: (188 to 355 g, 98 to 146 mm tall) all sit inside what this and
#: `PAYLOAD_RANGE` already cover. It is a training quantity: changing it changes
#: what a policy learns to walk under, which is why it did not move with the row.
PAYLOAD_UNIT_INERTIA: tuple[float, float, float] = (
    0.00359740, 0.00359740, 0.00052813
)


def add_payload_body(spec):
    """Add the payload body to the robot's spec, at the centre of a held object.

    **The object's centre, not the jaw it is held against.** An object in this
    claw is squeezed against the fixed jaw by its surface, so its centre is at
    least half its thickness off that jaw; and it is held `mdp/grasp.py::
    HOLD_DEPTH` in from where the jaws meet, because right at the anvil the wedge
    squeezes it out. So the body goes where `play --hold` would seat the object
    this load models, `PAYLOAD_CYLINDER` (`mdp/grasp.py::_held_in_palm`).
    Measured on the V1.6 `assets/jumper/jumper.xml`, that is 51.6 mm from the anvil
    and 42.0 mm clear of the fixed jaw's mesh; the soap, notebook and can seat 36 to 48 mm
    from the anvil the same way.

    The two placements before it were wrong in opposite directions. On the
    anvil, the load was an object whose centre is inside the jaw. At the "mouth
    centre" -- the midpoint of the jaws' closest pair at mid-close, with
    `LF_palm_pad_b_link`'s vertices read against its body frame rather than its geom's --
    it was 38 mm off the anvil, past the tips and 12 mm outside anything the jaws
    enclose even wide open (`mdp/grasp.py::_jaw_clouds` has why that frame is
    wrong).

    **It was 116 mm off until this was measured in the model training builds.**
    The seat is computed on the bare spec -- this runs before mjlab attaches the
    robot and prefixes its names -- and the helper looked names up with the
    prefix only. Every lookup missed and returned -1: the arm was never posed and
    the palm resolved to the last body in the model, a toe. The body landed at the
    wrist, behind even the point mass on the moving jaw it replaced, and a test
    comparing it with the same helper agreed with itself. `mdp/grasp.py::_named`
    now raises instead of returning -1, and the test measures the payload against
    the jaws' own meshes.

    A child of `LF_palm_link`, the fixed jaw, so the load stays put when the claw
    opens. No geom and no joint: an explicit inertial, zero until
    `mdp/events.py::claw_payload` writes a mass into it, so a run with
    `PAYLOAD_RANGE` at (0, 0) carries exactly nothing.
    """
    from .mdp.grasp import _held_in_palm  # local: grasp.py imports this module

    centre = _held_in_palm(spec.compile(), *PAYLOAD_CYLINDER)
    palm = spec.body("LF_palm_link")
    body = palm.add_body(name=PAYLOAD_BODY, pos=list(centre))
    body.explicitinertial = True
    body.mass = 0.0
    body.inertia = [0.0, 0.0, 0.0]
    body.ipos = [0.0, 0.0, 0.0]
    body.iquat = [1.0, 0.0, 0.0, 0.0]
    return spec

#: Where each ground foot sits under the body when the robot stands at HOME, as
#: (x, y) in the yaw-aligned base frame, in `FIVE_FOOT_LEGS` order.
#:
#: Read off the model rather than written down (`grasp_pose.py --stance`), so it
#: follows the asset. `mdp/rewards.py::foot_home_position` is what uses it.
#:
#: Re-read after master's bottom-shell revision, which moved every leg's mount
#: 9.13 mm forward of `base_link`'s origin (`LF_shoulder_link` x 0.04154 -> 0.05068):
#: all five feet +9.1 mm in x, none in y. The old values scored every foot 9 mm
#: behind where HOME actually puts it.
HOME_TOE_XY: tuple[tuple[float, float], ...] = (
    (0.1836, -0.0690),   # RF
    (-0.0068, 0.1974),   # LM
    (-0.0082, -0.1971),  # RM
    (-0.1405, 0.1595),   # LR
    (-0.1416, -0.1584),  # RR
)

__all__ = [
    "APERTURE_MM",
    "ARM_JOINTS",
    "CARRIED_JOINTS",
    "CARRIED_LEG",
    "FINGER_JOINT",
    "FIVE_FOOT_JOINTS",
    "FIVE_FOOT_LEGS",
    "GRASP_BOX",
    "GRIPPER_CLOSED",
    "GRIPPER_OPEN",
    "HOME_TOE_XY",
    "KNUCKLE_CLEAR",
    "LEG_FOOT",
    "LF_GRASP",
    "LF_GRASP_BOX",
    "OBSERVED_JOINTS",
    "PAYLOAD_BODY",
    "PAYLOAD_CYLINDER",
    "PAYLOAD_RANGE",
    "PAYLOAD_UNIT_INERTIA",
    "SQUEEZE_LEAD",
    "TRUNK_HALF_WIDTH",
    "add_payload_body",
    "aperture_mm",
    "foot_geoms_for",
]
