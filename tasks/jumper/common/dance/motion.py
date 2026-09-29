"""The choreography: finding the material, reading it, and converting it for mjlab.

The dance tasks are the ones whose behaviour is set by **data** rather than by
config: the material in each task's own `media/` directory, which the task passes
in. Every dance task commits its clip so it runs from a fresh clone, and this module
is the whole of the contract -- what has to be there, what the file has to contain,
and how it becomes the reference the trainer consumes.

## The material

Files are found **by extension**, exactly one of each kind per directory. Naming is
free, because what a dance is called is the user's business and a fixed
`motion.npz` would only mean everyone renames their file:

    media/<anything>.npz             the choreography -- the only file training reads
    media/<anything>.{mp3,wav,m4a,ogg,flac}   the music, optional
    media/<anything>.{mp4,mov,mkv,webm}       the face-screen animation, optional

Zero matches and several matches are different mistakes, so they are reported
differently, and both name the directory. The music and the video are the
exceptions: absent is an answer for both, because training reads neither. The music
is checked against the dance's length whenever it is there (`_check_audio`); the one
thing that cannot do without it is the performance video `export_media.py` renders,
which refuses to start rather than ship a dance with no music on it.

The music is optional because the demonstration's is not in the repository -- its
source could not be established, so it is not published -- and requiring it would
stop every fresh clone from building an environment over a file training never
opens.

**The reference recording the choreography was made from is deliberately not part
of this.** It was, and it bought nothing: no code ever opened it, and requiring it
meant `media/` held two videos in different roles, which -- searched by extension --
made both ambiguous. The face animation needed a subdirectory to escape a collision
with a file nothing read. Dropping the reference leaves one video slot with one
meaning. Keep your reference recording wherever you keep the rest of the source
material; a copy here now claims to be the face animation.

## The source schema

The `.npz` is the format written by the choreography player (`tools/dance_mujoco.py`
in the motion-planner repository), one 1-D array per channel at the recording rate:

    time, dt, phase              phase: 1 dance, 2 cooldown, 3 hold
    body_{x,y,z,roll,pitch,yaw}  commanded base pose, metres and radians (RPY)
    L{0..5}_j{k}                 commanded joint targets, radians
    meas_*                       the same channels as MuJoCo actually achieved
    bpm, beat_times_sim, beat_times_audio, audio_start_in_sim
    kp, kd                       the gains meas_* was tracked with

### The dtype to write

**Every per-sample channel is float32. The rest is whatever it is.** That is the
whole rule, and it is worth stating to whoever generates these files, because a
rule with an exception in it is a rule someone gets wrong.

Reading is permissive -- float64 clips load unchanged, and the generator currently
writes them -- but the committed `demo.npz` follows the rule, at 30.3 MB against
67.7 for the same clip in float64.

Nothing is lost by it. **The conversion below already emits float32**, because that
is what mjlab's `MotionLoader` holds the reference in, so the extra mantissa never
reached the trainer in the first place. Checked rather than argued, converting from
the downcast source against the float64 original: `joint_pos` bit-identical, body
positions within 3e-8 m, quaternions within 9e-8, finite-differenced velocities
within 1.2e-5 -- which is the position error over `target_dt`, as it should be.

`time` follows the rule too, and it is the one channel where the rule rather than
the arithmetic decided it: it saves 0.12 MB (a linear ramp deflates about 3.3x, so
float64 was never costing its raw size), and the quantisation is 7.6 us against a
1 ms tick. It is also read by nothing here -- the timebase comes from the scalar
`dt` -- which is what makes it safe to say so: downcasting it leaves every key of
the converted clip bit-identical.

### Two silent failures this module exists to prevent

**1. The leg order is not the robot's joint order.** The source numbers legs
`L0..L5` going round the body -- `L0` LF, `L1` LM, `L2` LR, `L3` RR, `L4` RM, `L5`
RF -- while the entity orders them LF, RF, LM, RM, LR, RR (`constants.HOME`, which
`Entity.joint_names` matches exactly). Read positionally, **the left and right rear
legs swap** and so do the middles. Nothing raises: the joint counts agree, every
value is in range, training converges, and the robot dances a different dance. The
map below is the single place that correspondence is written down, and
`tests/test_dance_motion.py` pins it against the clip's own left/right antisymmetry.

**2. `body_pos_w` is indexed by entity body index, not model body index.** mjlab's
`MotionLoader` slices `body_pos_w[:, body_indexes]` where the indexes come from
`Entity.find_bodies`, and `Entity.body_names` is the compiled model's body list
**without `world`**. Emitting the raw model order shifts every body by one, which
again raises nothing -- the arrays are the right shape and the reward simply scores
each link against its neighbour. `_entity_body_names` derives it rather than
assuming it, and asserts the count.

## Which channels are the reference

`REFERENCE = "measured"`, i.e. the `meas_*` channels rather than the commanded
`body_*` / `L*_j*` ones. **This was decided by measurement, against the obvious
choice**, and the numbers are worth keeping because the argument for the commanded
channels is a good one right up until it is checked.

The argument was: the commanded body pose and the commanded joint angles ought to
be one kinematically consistent set, since the support legs were solved by IK to
hold their feet planted while the body moved, and `meas_*` merely adds the
generating PD's tracking error -- its pitch reaches -0.286 rad against a commanded
-0.175, which reads like a lightly-damped overshoot at kd=0.1.

Forward kinematics on both says otherwise. Over 400 frames spanning the sample
clip, the four support feet (`SUPPORT_LEGS`; the two front arms lift as high as
0.29 m and are not support) measured, per frame:

    reference     non-coplanarity of the four support feet     lowest foot
                  mean          max                            (site z)
    command       13.2 mm       13.8 mm                        -6.3 mm
    measured       0.1 mm        4.9 mm                        +2.1 mm

The commanded channels are **PD targets, not a posture**. Nothing ever required
forward kinematics on them to close: the middle legs' targets sit 13 mm below the
rear legs' and 6 mm through the floor, which is what asking a PD for more travel
than the contact allows looks like. `meas_*` is coplanar to a tenth of a
millimetre for the good reason that it is a trajectory MuJoCo actually produced,
with the contacts resolved.

So the commanded pitch is a target the robot never reaches, and the measured pitch
is the body genuinely settling -- imitating the commanded set would score the
policy against a configuration it cannot stand in, capping the tracking reward
permanently and for a reason nothing in the logs would name.

Set `REFERENCE = "command"` to compare; nothing else changes, and
`_check_support_feet` will refuse the conversion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "AUDIO_SUFFIXES",
    "LEG_JOINTS",
    "REFERENCE",
    "SOURCE_JOINT_ORDER",
    "SUPPORT_LEGS",
    "VIDEO_SUFFIXES",
    "DanceMaterial",
    "SourceClip",
    "audio_duration_s",
    "ensure_motion_npz",
    "find_material",
    "load_source",
]

AUDIO_SUFFIXES = (".mp3", ".wav", ".m4a", ".ogg", ".flac")
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm")

#: Which set of channels to imitate: "command" or "measured". See the module
#: docstring for the measurement that decided this, and for why the commanded
#: channels -- the intuitive choice -- do not describe a posture the robot can
#: hold.
REFERENCE = "measured"

#: The four legs that stay on the ground. The two front 5-DoF arms are free: they
#: are what the choreography is mostly made of, and they lift to 0.29 m, so
#: including them in a ground check would fail every frame of the dance.
#:
#: This is the generator's `SUPPORT_IDS` (legs 1-4) under the entity's names.
SUPPORT_LEGS: tuple[str, ...] = ("LM", "RM", "LR", "RR")

#: Source leg index -> the entity's joint names for that leg, in the source's own
#: within-leg order. **The one place the correspondence is written down.**
#:
#: Taken from the generator's `JOINT_NAMES` table, not inferred. The two 5-DoF
#: front arms are `L0` and `L5`; the four 3-DoF legs run round the body in
#: between, which is why `L2` is the *left* rear and `L3` the *right* rear.
LEG_JOINTS: dict[int, tuple[str, ...]] = {
    0: ("LF_J0_joint", "LF_J1_joint", "LF_J2_joint",
        "LF_J3_joint", "LF_J4_joint"),
    1: ("LM_J0_joint", "LM_J1_joint", "LM_J2_joint"),
    2: ("LR_J0_joint", "LR_J1_joint", "LR_J2_joint"),
    3: ("RR_J0_joint", "RR_J1_joint", "RR_J2_joint"),
    4: ("RM_J0_joint", "RM_J1_joint", "RM_J2_joint"),
    5: ("RF_J0_joint", "RF_J1_joint", "RF_J2_joint",
        "RF_J3_joint", "RF_J4_joint"),
}

#: The source's flat channel order, as joint names: leg 0's joints, then leg 1's...
SOURCE_JOINT_ORDER: tuple[str, ...] = tuple(
    name for leg in sorted(LEG_JOINTS) for name in LEG_JOINTS[leg]
)

def _max_reference_speed() -> float:
    """Fastest a reference joint may move between two control steps, in rad/s: the
    servo's corner speed, where its measured curve leaves the torque plateau.

    A discontinuity -- a dropped frame, two takes concatenated -- becomes a step the
    policy is scored against and cannot track, which shows up as a reward ceiling
    rather than as an error. What makes a step untrackable is the servo, not the
    file, so the check is a speed and is made **after resampling to the control
    rate**, on the frames the policy is actually scored against.

    It used to be 0.05 rad per *source tick*, which was 50 rad/s for the 1 kHz clips
    it was written against and a different speed for every other rate: 25 rad/s at
    500 Hz and 2.5 rad/s for a 50 Hz track, where it refused ordinary dancing
    (rl-wbc-fsm's maze moves 0.23 rad per 20 ms frame, 11.5 rad/s). It also refused
    rl-wbc-fsm's Brazilian dance for a single-tick glitch the policy never sees at 50 Hz: its
    legs straighten to exactly 0.000 where the planner's IK ran out of reach, 0.30
    rad in one 2 ms tick, but 22.0 rad/s at the control rate.

    The corner speed rather than the cutoff (64.0 rad/s, where torque reaches zero):
    past the corner the servo moves only by giving up torque, and a reference that
    asks for it is asking the robot to dance with its legs unloaded. Measured at the
    control rate, the fastest any reference joint moves: crab 12.9 rad/s, Brazilian 22.0.
    A 1 rad splice is 50 rad/s and is refused.

    A function rather than a constant so that importing this module does not load
    the servo curve; the source-format checks run without it.
    """
    from ..actuator import CORNER_SPEED

    return float(CORNER_SPEED)


def _check_continuity(joint_pos: np.ndarray, target_dt: float, source: Path) -> None:
    """Refuse a reference that moves a joint faster than the servo can follow.

    `joint_pos` is the clip **at the control rate**, [frames, joints] in the
    entity's order. See `_max_reference_speed` for the bound and why it is a speed.
    """
    if len(joint_pos) < 2:
        return
    speed = np.abs(np.diff(joint_pos, axis=0)) / target_dt
    limit = _max_reference_speed()
    if speed.max() > limit:
        frame, col = np.unravel_index(int(speed.argmax()), speed.shape)
        raise ValueError(
            f"{source.name}: {_entity_joint_names()[col]} moves {speed.max():.1f} rad/s "
            f"between control steps {frame} and {frame + 1} "
            f"(t={frame * target_dt:.2f} s), above the servo's {limit:.1f} rad/s corner "
            f"speed. A choreography is continuous by construction, so this is most "
            f"likely a dropped frame or two takes concatenated -- a step the policy "
            f"would be scored against and could not track."
        )

#: How far outside its limits a reference joint may sit, in radians.
#:
#: **Not zero, because the reference is simulator output.** `meas_*` comes back
#: through MuJoCo's constraint solver, which permits a small violation, and the two
#: finger joints -- both of which home exactly *on* a limit -- come out 1.477 mrad
#: past it. The commanded channels, being IK output, violate nothing at all.
#:
#: Measured on the sample clip, the worst violation in each case:
#:
#:     measured reference        1.477 mrad     (the two fingers, solver slop)
#:     commanded reference       0.000 mrad
#:     middle leg pair swapped  89.1   mrad
#:     rear leg pair swapped   398.3   mrad
#:     front arm pair swapped  888.2   mrad
#:
#: 10 mrad sits 6.8x above the slop and 8.9x below the smallest real mismatch, so
#: the placement is not delicate. A tolerance tight enough to catch the fingers
#: would reject every clip this task exists to train on.
_JOINT_LIMIT_TOL = 0.01

#: How far out of plane the four support feet may sit, in metres, judged on the
#: **median** frame.
#:
#: This is the check that the base pose and the joint angles describe one posture.
#: The two candidate references separate by a factor of 130 on it -- 0.1 mm for
#: `meas_*` against 13.2 mm for the commanded channels -- so anywhere between them
#: works and the placement is not delicate. The median rather than the maximum
#: because a real trajectory has contact transients (the measured clip touches
#: 4.9 mm on one frame in 400 while sitting at 0.1 mm throughout), and a check that
#: fires on the worst frame of a good clip is a check that gets its tolerance
#: raised until it means nothing.
_SUPPORT_COPLANAR_TOL = 0.005

#: How far the support feet's median height may sit from where the same feet sit at
#: `HOME` and `STAND_Z`, in metres.
#:
#: Coplanarity alone cannot see a wholesale vertical offset -- a clip recorded
#: against a different ground reference has its feet perfectly coplanar five
#: centimetres underground. This is the second half of that check, and it is loose
#: because the body genuinely rises 25 mm during the dance.
_SUPPORT_HEIGHT_TOL = 0.015


@dataclass(frozen=True)
class DanceMaterial:
    """The material, resolved.

    Only `motion` is required, and it is the only member training reads. `audio` is
    the music the choreography was made for, and `eyes` is what the robot's face
    screen plays while performing it; both are optional and both exist so the export
    can ship the whole performance rather than only the part physics knows about.
    """

    motion: Path
    audio: Path | None = None
    eyes: Path | None = None


@dataclass(frozen=True)
class SourceClip:
    """A source `.npz`, read and remapped into the entity's joint order.

    `joint_pos` is [T, 22] in `Entity.joint_names` order, `root_pos` [T, 3] and
    `root_rpy` [T, 3]. `dt` is the source tick, not the control step.
    """

    joint_pos: np.ndarray
    root_pos: np.ndarray
    root_rpy: np.ndarray
    phase: np.ndarray
    dt: float
    bpm: float
    audio_start_s: float
    beat_times_s: np.ndarray
    source: Path

    @property
    def duration_s(self) -> float:
        return float(len(self.joint_pos) * self.dt)


def _one_file(directory: Path, suffixes: tuple[str, ...], what: str) -> Path:
    """The single file in `directory` with one of `suffixes`.

    Zero and several are separate mistakes and get separate messages: the first
    means the material was never put there, the second means it is ambiguous which
    take is meant, and picking one (the newest, say) would train on a file the user
    did not choose without saying so.
    """
    if not directory.is_dir():
        raise FileNotFoundError(
            f"{directory} does not exist. A dance task needs material there: see "
            f"tasks/jumper/dance/media/README.md for what goes in it."
        )
    hits = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    )
    listing = ", ".join(suffixes)
    if not hits:
        present = sorted(p.name for p in directory.iterdir() if p.is_file())
        raise FileNotFoundError(
            f"no {what} in {directory} (looked for {listing}). "
            f"Present: {', '.join(present) or '(nothing)'}. "
            f"See tasks/jumper/dance/media/README.md."
        )
    if len(hits) > 1:
        names = ", ".join(p.name for p in hits)
        raise ValueError(
            f"several {what} files in {directory}: {names}. Keep one -- which take "
            f"to train on is not a choice this task should make silently."
        )
    return hits[0]


def _optional_one_file(
    directory: Path, suffixes: tuple[str, ...], what: str
) -> Path | None:
    """Like `_one_file`, but "not there" is an answer rather than an error.

    Ambiguity still raises. Absent and ambiguous are genuinely different: this
    material is optional, so no file means the performance simply goes without it,
    while two files mean someone put both a draft and a final take there and the
    export would otherwise pick one without saying which.
    """
    if not directory.is_dir():
        return None
    hits = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    )
    if not hits:
        return None
    if len(hits) > 1:
        names = ", ".join(p.name for p in hits)
        raise ValueError(
            f"several {what} files in {directory}: {names}. Keep one -- which take "
            f"to ship is not a choice this task should make silently."
        )
    return hits[0]


def find_material(directory: Path) -> DanceMaterial:
    """Resolve the material in a task's `media/`, or say precisely which required
    file is missing. The directory is the task's to name, and there is no default:
    with several dance tasks, a default is one of them chosen for all the others."""
    d = Path(directory)
    return DanceMaterial(
        motion=_one_file(d, (".npz",), "motion clip"),
        # Optional, like the face animation below: training never reads it, and
        # the export that does checks for it itself. See the module docstring.
        audio=_optional_one_file(d, AUDIO_SUFFIXES, "music"),
        # The one video in `media/` is the face animation. Optional, and beside the
        # other two rather than in a subdirectory of its own, because it is the only
        # video the task has a use for -- see the module docstring on why the
        # reference recording is not part of the material.
        eyes=_optional_one_file(d, VIDEO_SUFFIXES, "face animation"),
    )


def load_source(path: Path, reference: str | None = None) -> SourceClip:
    """Read a source `.npz` and remap it into the entity's joint order.

    Every channel is required; a missing one is named rather than allowed to
    surface later as a `KeyError` from inside the conversion.
    """
    ref = REFERENCE if reference is None else reference
    if ref not in ("command", "measured"):
        raise ValueError(f"reference must be 'command' or 'measured', not {ref!r}")
    prefix = "meas_" if ref == "measured" else ""

    with np.load(path) as data:
        keys = set(data.files)
        wanted = (
            [f"{prefix}body_{a}" for a in ("x", "y", "z", "roll", "pitch", "yaw")]
            + [f"{prefix}L{leg}_j{j}"
               for leg in sorted(LEG_JOINTS)
               for j in range(len(LEG_JOINTS[leg]))]
            + ["dt"]
        )
        missing = [k for k in wanted if k not in keys]
        if missing:
            raise KeyError(
                f"{path.name} is missing {len(missing)} channel(s): "
                f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}. "
                f"Expected the schema written by tools/dance_mujoco.py -- see "
                f"{Path(__file__).name} for the full list."
            )
        get = {k: np.asarray(data[k], dtype=np.float64) for k in wanted}
        optional = {
            k: np.asarray(data[k], dtype=np.float64)
            for k in ("phase", "bpm", "audio_start_in_sim", "beat_times_audio")
            if k in keys
        }

    dt = float(get["dt"])
    if not 0.0 < dt < 1.0:
        raise ValueError(f"{path.name}: dt={dt} is not a plausible timestep")

    root_pos = np.stack(
        [get[f"{prefix}body_{a}"] for a in ("x", "y", "z")], axis=1
    )
    root_rpy = np.stack(
        [get[f"{prefix}body_{a}"] for a in ("roll", "pitch", "yaw")], axis=1
    )

    # The remap. Built name -> column from the source's own flat order, then read
    # out in the entity's order by the caller of `_entity_joint_names`.
    columns = {
        name: get[f"{prefix}L{leg}_j{j}"]
        for leg in sorted(LEG_JOINTS)
        for j, name in enumerate(LEG_JOINTS[leg])
    }
    joint_pos = np.stack([columns[n] for n in _entity_joint_names()], axis=1)

    n = len(root_pos)
    if not (len(joint_pos) == len(root_rpy) == n):
        raise ValueError(f"{path.name}: channels disagree on length")
    if n < 2:
        raise ValueError(f"{path.name}: {n} frame(s) is not a trajectory")

    return SourceClip(
        joint_pos=joint_pos,
        root_pos=root_pos,
        root_rpy=root_rpy,
        phase=optional.get("phase", np.ones(n)),
        dt=dt,
        bpm=float(optional.get("bpm", np.array(0.0))),
        audio_start_s=float(optional.get("audio_start_in_sim", np.array(0.0))),
        beat_times_s=optional.get("beat_times_audio", np.zeros(0)),
        source=path,
    )


def _entity_joint_names() -> tuple[str, ...]:
    """The entity's joint order.

    Taken from `constants.HOME`, which `Entity.joint_names` matches exactly
    (verified; `tests/test_dance_motion.py` keeps it that way). Reading it from
    here rather than building an `Entity` keeps this module importable without a
    compiled model, which the source-format tests rely on.
    """
    from ..constants import HOME

    return tuple(HOME)


def _rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """[N, 3] roll/pitch/yaw -> [N, 4] wxyz.

    The source's convention is `R = Rz(yaw) Ry(pitch) Rx(roll)`, stated in the
    generator beside the body-pose assignment. Composing it in the other order
    gives a rotation that is correct only where two of the three angles are zero,
    so it looks right at rest and drifts through the dance.
    """
    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr = np.cos(r * 0.5), np.sin(r * 0.5)
    cp, sp = np.cos(p * 0.5), np.sin(p * 0.5)
    cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
    return np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=1,
    )


def _resample(clip: SourceClip, target_dt: float) -> tuple[np.ndarray, ...]:
    """Bring the clip to the control rate, returning (joint_pos, root_pos, root_rpy).

    An exact integer stride is taken as a stride -- the common case here, 1 kHz to
    50 Hz -- because it is exact and picks real recorded frames. Anything else is
    linearly interpolated, which is safe for the joint angles and for RPY **only
    while the angles do not wrap**; that is checked rather than assumed, since a
    yaw crossing +/-pi would interpolate the long way round and spin the reference
    body through a full turn between two frames.
    """
    ratio = target_dt / clip.dt
    n_out = int(np.floor((len(clip.joint_pos) - 1) * clip.dt / target_dt)) + 1
    stride = round(ratio)
    if abs(ratio - stride) < 1e-9 and stride >= 1:
        idx = np.arange(n_out) * stride
        return clip.joint_pos[idx], clip.root_pos[idx], clip.root_rpy[idx]

    span = np.abs(np.diff(clip.root_rpy, axis=0)).max()
    if span > np.pi:
        raise ValueError(
            f"{clip.source.name}: the base orientation wraps (a {span:.2f} rad "
            f"step), so it cannot be resampled by interpolation. Record the clip "
            f"at a rate that divides the {1.0 / target_dt:.0f} Hz control rate."
        )
    t_src = np.arange(len(clip.joint_pos)) * clip.dt
    t_out = np.arange(n_out) * target_dt

    def lerp(a: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.interp(t_out, t_src, a[:, c]) for c in range(a.shape[1])], axis=1
        )

    return lerp(clip.joint_pos), lerp(clip.root_pos), lerp(clip.root_rpy)


def _entity_body_names(model) -> list[str]:
    """The compiled model's bodies **without `world`**, which is what
    `Entity.body_names` is and therefore what `MotionLoader` indexes into."""
    import mujoco

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]
    if names[0] != "world":
        raise RuntimeError(
            f"body 0 is {names[0]!r}, not 'world'; the entity-body offset this "
            f"conversion relies on no longer holds."
        )
    return names[1:]


def _check_joint_limits(
    model, joint_pos: np.ndarray, jnt_ids: np.ndarray, names: tuple[str, ...]
) -> None:
    """Every reference angle must be reachable by the joint it is assigned to.

    Two things fail here, and the second is why it exists.

    A clip recorded against a different robot, or against an earlier revision of
    this one, asks for angles the hardware cannot reach; the policy is then scored
    against a target it can never hit, and the tracking reward carries a floor
    nothing explains.

    **And it is what catches a swapped front pair.** The FK ground check
    (`_check_support_feet`) sees only the four support legs, so exchanging `L0` and
    `L5` -- the two 5-DoF arms, which the choreography is mostly made of -- passes
    it untouched. But the arms' ranges are mirror images and barely overlap:
    `LF_shoulder_yaw` lives in [-2.75, 0.70] and its clip spans [-1.59, -0.34],
    while `RF_shoulder_yaw` spans [+0.34, +1.59]. Give the left arm the right arm's
    angles and it is asked for +1.59 against a limit of +0.70. So between the two
    checks, **every one of the three possible leg-pair swaps is refused**, which is
    what `tests/test_dance_motion.py` pins.
    """
    lo, hi = model.jnt_range[jnt_ids, 0], model.jnt_range[jnt_ids, 1]
    col_lo, col_hi = joint_pos.min(axis=0), joint_pos.max(axis=0)
    tol = _JOINT_LIMIT_TOL
    bad = np.nonzero((col_lo < lo - tol) | (col_hi > hi + tol))[0]
    if len(bad):
        by = np.maximum(hi[bad] - col_hi[bad], lo[bad] - col_lo[bad]) * -1000.0
        lines = "\n".join(
            f"    {names[i]:26s} clip [{col_lo[i]:+.3f}, {col_hi[i]:+.3f}]  "
            f"limit [{lo[i]:+.3f}, {hi[i]:+.3f}]  out by {b:.1f} mrad"
            for i, b in zip(bad[:8], by[:8])
        )
        raise ValueError(
            f"{len(bad)} joint(s) in the reference are outside the model's limits "
            f"by more than {tol * 1000:.0f} mrad:\n"
            f"{lines}\n"
            f"Either the clip was recorded against a different robot, or the leg "
            f"map has a pair exchanged -- the two front arms' ranges are mirror "
            f"images, so swapping them lands each one outside the other's limits."
        )


def _home_support_height(model, data, q_adr: np.ndarray, site_ids: list[int]) -> float:
    """Where the support feet sit when the robot stands at `HOME` and `STAND_Z`.

    Derived from the model rather than written down, so that a change to the stance
    or to the foot geometry moves the reference with it instead of leaving a stale
    constant that quietly turns this check into noise.
    """
    import mujoco

    from ..constants import HOME, STAND_Z

    data.qpos[:] = 0.0
    data.qpos[2] = STAND_Z
    data.qpos[3] = 1.0
    data.qpos[q_adr] = [HOME[n] for n in _entity_joint_names()]
    mujoco.mj_forward(model, data)
    return float(np.mean([data.site_xpos[s][2] for s in site_ids]))


def _check_support_feet(site_z: np.ndarray, home_z: float, reference: str) -> None:
    """The consistency check: does the reference describe a posture the robot can
    actually stand in?

    `site_z` is [frames, 4], the support feet's site heights over the clip. This is
    what makes "the base pose and the joint angles are one set" a checked fact
    rather than an assumption, and it is the failure that would otherwise be
    invisible -- every joint stays in range, every array is the right shape, and the
    tracking reward simply reports a small plausible error against a configuration
    the robot cannot hold.

    Two ways to fail, reported separately because they point in different
    directions: feet that are not in a plane mean the channels do not belong
    together, while feet that are in a plane at the wrong height mean the clip was
    recorded against a different ground.
    """
    spread = float(np.median(site_z.max(axis=1) - site_z.min(axis=1)))
    if spread > _SUPPORT_COPLANAR_TOL:
        raise ValueError(
            f"the reference does not describe a posture the robot can stand in: "
            f"forward kinematics puts the four support feet {spread * 1000:.1f} mm "
            f"out of plane on the median frame (tolerance "
            f"{_SUPPORT_COPLANAR_TOL * 1000:.0f} mm).\n"
            f"REFERENCE is {reference!r}. The commanded channels are PD *targets* "
            f"and are expected to fail this -- nothing ever required forward "
            f"kinematics on them to close. Use 'measured', or check that the base "
            f"pose and the joint angles come from the same take."
        )
    offset = float(np.median(site_z)) - home_z
    if abs(offset) > _SUPPORT_HEIGHT_TOL:
        raise ValueError(
            f"the support feet are coplanar but sit {offset * 1000:+.1f} mm from "
            f"where they sit at the standing pose (tolerance "
            f"+/-{_SUPPORT_HEIGHT_TOL * 1000:.0f} mm). The clip was most likely "
            f"recorded against a different ground height, or against a model whose "
            f"legs are not this one's."
        )


def audio_duration_s(path: Path) -> float | None:
    """The music's length in seconds, or None if it could not be read.

    Uses the ffmpeg binary `imageio-ffmpeg` ships, which is a declared dependency
    of this project (`pyproject.toml`) and therefore present in any correct
    install -- but a missing or unreadable decoder returns None rather than
    raising, because the music is not something training reads and no policy
    should fail to train over it.
    """
    import re
    import subprocess

    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        # ffmpeg with no output writes the stream summary to stderr and exits
        # non-zero; that is the intended use here, so `check=False`.
        out = subprocess.run(
            [exe, "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=60, check=False,
        ).stderr
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError):
        # ImportError / RuntimeError cover imageio-ffmpeg being absent or unable
        # to find its bundled binary; the rest cover it failing to run. None of
        # them should stop a training run over a file training does not read.
        return None
    m = re.search(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)", out)
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def _check_audio(material: DanceMaterial, clip: SourceClip) -> str:
    """Compare the music's length with the dance's, and return a line to print.

    **The bound is one-sided, and that asymmetry is the whole design.** The clip
    carries its own sync -- `audio_start_in_sim` says when the music comes in and
    `beat_times_audio` where every beat falls -- so the useful comparison is
    against `duration - audio_start`.

    *Too short* is a real failure: the robot goes on dancing after the music stops,
    and nothing else in the pipeline looks at the audio, so the first report would
    be a human watching the video. Half the expected length is well past any
    plausible trim.

    *Too long* is not a failure at all. A track that outlasts the choreography is
    simply not used to the end, which is the normal case for a dance cut from a
    song, and for a soundtrack lifted from a render whose lead-out runs past the
    last frame. This originally refused anything longer than the clip and **that
    was wrong**: the sample dance's own soundtrack, extracted from the video it was
    rendered into, is 236.1 s against a 235.7 s clip and was refused as "the wrong
    file". A check that fires on correct material is worse than no check, because
    the fix is to disable it.

    Nor could a tighter upper bound mean anything: duration cannot tell a wrong
    long track from a right one. What it can tell is silence, and that is what it
    is asked. The measured difference is printed either way, once, as part of the
    conversion summary.
    """
    if material.audio is None:
        return (
            f"  music: none in {material.motion.parent} -- training does not need it, "
            f"the performance video does (export.py --video)"
        )
    audio = audio_duration_s(material.audio)
    if audio is None:
        return f"  music {material.audio.name}: duration not readable"
    expected = clip.duration_s - clip.audio_start_s
    if audio < 0.5 * expected:
        raise ValueError(
            f"{material.audio.name} is {audio:.1f} s but the dance needs about "
            f"{expected:.1f} s of music ({clip.duration_s:.1f} s of dance, starting "
            f"{clip.audio_start_s:.1f} s in). The robot would dance most of this "
            f"clip in silence.\n"
            f"The choreography states its own requirement: {len(clip.beat_times_s)} "
            f"beats at {clip.bpm:.2f} bpm. Check that the music in "
            f"{material.audio.parent} is the track it was made for -- a shorter cut "
            f"of the same song will not do, and looping it only works if the cut is "
            f"a whole number of beats."
        )
    return (
        f"  music {material.audio.name}: {audio:.1f} s against {expected:.1f} s of "
        f"dance from {clip.audio_start_s:.1f} s in ({audio - expected:+.1f} s)"
    )


def _model_digest(model) -> str:
    """The part of the robot the conversion depends on.

    **The cache has to be keyed on the model, and this was learned the hard way.**
    The first version keyed on the clip alone, and merging a new URDF -- more
    bodies, renamed links -- left a cache converted against the previous robot.
    That one happened to raise, because `MotionLoader` indexes bodies and the
    count had changed, but the count changing is luck rather than a check: a
    revision that reorders or renames bodies while keeping their number loads
    cleanly and scores every link against a different link, which is the failure
    the module docstring's point 2 is about, arriving through the back door.

    Hashed is what the conversion reads out of the model and nothing else -- the
    two name lists it maps through, and the kinematic tree forward kinematics
    walks. Not the whole serialised model: that pulls in inertia, collision and
    visual data the conversion never touches, so every mesh retouch would discard
    a cache that is still correct.
    """
    h = hashlib.sha256()
    for name in _entity_body_names(model):
        h.update(f"{name}\0".encode())
    for name in _entity_joint_names():
        h.update(f"{name}\0".encode())
    for arr in (
        model.body_parentid, model.body_pos, model.body_quat,
        model.jnt_type, model.jnt_bodyid, model.jnt_qposadr,
        model.jnt_axis, model.jnt_pos,
    ):
        a = np.ascontiguousarray(arr)
        h.update(f"|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _fingerprint(source: Path, target_dt: float, reference: str, model) -> str:
    """What the cached conversion was built from.

    Content-addressed rather than mtime-based: a clip restored from a backup or
    checked out again has a new mtime and identical content, and rebuilding is
    only wasted minutes -- but the reverse, an edit that preserves mtime, would
    silently train on the previous dance. Hashing costs about a second on a
    30 MB clip and removes that possibility.

    The robot is in here too, via `_model_digest` -- see there for why.
    """
    h = hashlib.sha256()
    h.update(source.read_bytes())
    h.update(f"|{target_dt!r}|{reference}|{_model_digest(model)}|v2".encode())
    return h.hexdigest()


def ensure_motion_npz(
    target_dt: float,
    directory: Path,
    asset: Path | None = None,
    reference: str | None = None,
) -> Path:
    """Convert the material into mjlab's `MotionLoader` format, and return its path.

    The conversion is cached beside the material in `.cache/` and rebuilt whenever
    the source, the control rate, `REFERENCE` **or the robot** changes. It runs at
    environment build time rather than as a step the user is asked to remember,
    because a conversion the user has to re-run by hand is one that eventually does
    not get re-run -- and a stale cache trains the previous dance without a word.

    `MotionLoader` reads six keys and ignores the rest, so the provenance is
    written into the same file rather than into a sidecar that can be separated
    from it.
    """
    import mujoco

    from ..constants import get_spec

    ref = REFERENCE if reference is None else reference
    material = find_material(directory)
    clip = load_source(material.motion, reference=ref)

    # Compiled before the cache is consulted rather than after, because the model
    # is part of what the cache is keyed on. It costs 0.15 s on this machine
    # against an environment build measured in seconds, which is a cheap way to
    # stop a robot revision from silently reusing the previous robot's clip.
    model = get_spec(asset).compile()

    cache = material.motion.parent / ".cache"
    out = cache / f"{material.motion.stem}.motion.npz"
    stamp = cache / f"{material.motion.stem}.motion.json"
    want = _fingerprint(material.motion, target_dt, ref, model)
    if out.is_file() and stamp.is_file():
        try:
            if json.loads(stamp.read_text()).get("fingerprint") == want:
                return out
        except (OSError, ValueError):
            pass  # An unreadable stamp means rebuild, not fail.

    # Before the expensive part: a wrong music file should be reported in the
    # second it takes to read a header, not after the conversion.
    audio_line = _check_audio(material, clip)

    joint_pos, root_pos, root_rpy = _resample(clip, target_dt)
    _check_continuity(joint_pos, target_dt, clip.source)
    root_quat = _rpy_to_quat(root_rpy)
    n = len(joint_pos)

    data = mujoco.MjData(model)
    body_names = _entity_body_names(model)
    n_bodies = len(body_names)

    entity_joints = _entity_joint_names()
    q_adr, jnt_ids = [], []
    for name in entity_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(
                f"the model has no joint {name!r}. The clip is written against "
                f"the default jumper asset; a structurally different --model cannot "
                f"be used with it."
            )
        q_adr.append(model.jnt_qposadr[jid])
        jnt_ids.append(jid)
    q_adr = np.asarray(q_adr, dtype=int)

    # Cheap, and it fails before the per-frame loop rather than after it.
    _check_joint_limits(model, joint_pos, np.asarray(jnt_ids, dtype=int), entity_joints)

    site_ids = []
    for leg in SUPPORT_LEGS:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg)
        if sid < 0:
            raise ValueError(
                f"the model has no site {leg!r}. The six foot sites are generated "
                f"by assets/jumper/tools/build_jumper.py and are what the ground check "
                f"measures against."
            )
        site_ids.append(sid)
    home_z = _home_support_height(model, data, q_adr, site_ids)
    site_z = np.zeros((n, len(site_ids)))

    body_pos = np.zeros((n, n_bodies, 3))
    body_quat = np.zeros((n, n_bodies, 4))
    body_lin = np.zeros((n, n_bodies, 3))
    body_ang = np.zeros((n, n_bodies, 3))
    joint_vel = np.zeros((n, len(entity_joints)))

    def set_qpos(k: int) -> None:
        data.qpos[:] = 0.0
        data.qpos[0:3] = root_pos[k]
        data.qpos[3:7] = root_quat[k]
        data.qpos[q_adr] = joint_pos[k]

    # Velocities by finite difference through `mj_differentiatePos`, which is the
    # only correct way to difference a configuration containing a free joint: the
    # quaternion's four numbers are not a vector space and subtracting them gives
    # an angular velocity that is wrong by a factor depending on the rotation.
    qpos_a = np.zeros(model.nq)
    qpos_b = np.zeros(model.nq)
    qvel = np.zeros(model.nv)
    v_adr = []
    for name in entity_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        v_adr.append(model.jnt_dofadr[jid])
    v_adr = np.asarray(v_adr, dtype=int)

    vel6 = np.zeros(6)
    for k in range(n):
        nxt = min(k + 1, n - 1)
        set_qpos(k)
        qpos_a[:] = data.qpos
        set_qpos(nxt)
        qpos_b[:] = data.qpos
        if nxt != k:
            mujoco.mj_differentiatePos(model, qvel, target_dt, qpos_a, qpos_b)
        else:
            qvel[:] = 0.0  # The last frame holds the previous velocity's target.
        set_qpos(k)
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)

        joint_vel[k] = qvel[v_adr]
        # Entity body i is model body i+1: `world` is dropped, see
        # `_entity_body_names`.
        body_pos[k] = data.xpos[1:]
        body_quat[k] = data.xquat[1:]
        for b in range(n_bodies):
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, b + 1, vel6, 0
            )
            body_ang[k, b] = vel6[0:3]
            body_lin[k, b] = vel6[3:6]

        site_z[k] = [data.site_xpos[s][2] for s in site_ids]

    _check_support_feet(site_z, home_z, ref)

    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin.astype(np.float32),
        body_ang_vel_w=body_ang.astype(np.float32),
        # Ignored by MotionLoader; kept so the file says what it is.
        joint_names=np.array(entity_joints),
        body_names=np.array(body_names),
        fps=np.array(1.0 / target_dt),
    )
    stamp.write_text(json.dumps({
        "fingerprint": want,
        "source": material.motion.name,
        "reference": ref,
        # So the file says which robot it was built for, not only that it was
        # built for some robot the fingerprint agreed with.
        "model_digest": _model_digest(model)[:16],
        "bodies": int(n_bodies),
        "frames": int(n),
        "fps": round(1.0 / target_dt, 6),
        "duration_s": round(n * target_dt, 3),
        "bpm": clip.bpm,
        "audio_start_in_sim_s": clip.audio_start_s,
    }, indent=2) + "\n")
    print(
        f"[mjrl] dance: converted {material.motion.name} "
        f"({clip.duration_s:.1f} s at {1.0 / clip.dt:.0f} Hz) -> {n} frames at "
        f"{1.0 / target_dt:.0f} Hz, reference={ref!r}\n" + audio_line
    )
    return out
