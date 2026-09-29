# Material for `jumper.dance`

**This directory is the dance.** The other tasks describe a robot and a goal; this
one is configured by a recorded performance, so the material is what makes it a
task at all.

The demonstration choreography and its face animation — `demo.npz`, `demo.mp4` —
are committed, 30.5 MB of them, so `--task jumper.dance` trains and plays from a
fresh clone. **The music the dance was made for is not**: its source could not be
established, so it is not published with the repository. Training never reads the
music, so it is optional. What needs it is the performance video `scripts/export.py`
renders, which stops before rendering when there is none — put the track the
choreography was made for here, or export with `--no-video`. Nothing else here is
committed: your own choreography is a performance, not the framework.

## What is here, and what to replace it with

Files are found **by extension**. The names are yours; at most one of each kind:

| Put here | Purpose | |
|---|---|---|
| `<name>.npz` | the choreography — **the only file training reads** | required |
| `<name>.{mp3,wav,m4a,ogg,flac}` | the music the dance was made for; the exported video needs it | optional |
| `<name>.{mp4,mov,mkv,webm}` | the face-screen animation | optional |

Two of the same kind is an error rather than a guess: which take to use is not a
choice this task should make on your behalf. **So swapping in your own dance means
taking the demo out**, not adding beside it.

```
tasks/jumper/dance/media/
├── README.md
├── demo.npz          <- committed, 30.3 MB
├── <music>.mp3       <- not committed, optional: the exported video needs it
├── demo.mp4          <- committed,  0.2 MB, the face animation
└── .cache/           <- generated, ignored, safe to delete
```

`demo` is not a magic name; nothing reads it. Call yours whatever the dance is
called. The `.cache/` entries are named after the `.npz` but keyed on its
**contents**, so renaming the clip and its cache together keeps the cache valid,
and renaming only the clip reconverts rather than going stale.

### The video is the face animation, not the reference recording

**Do not put the reference recording here.** It used to be required, and it bought
nothing — no code ever opened it. What it cost was a collision: `media/` is searched
by extension, so two videos in different roles made both ambiguous, and the face
animation had to hide in an `eyes/` subdirectory to get out of the way of a file
nothing read. Keep your reference recording wherever the rest of the source
material lives; a copy here is now taken for the robot's face.

The export prints the name of the file it shipped, which is where you would notice.

**Training never reads the face animation**; it exists so that `scripts/export.py`
can ship the whole performance rather than only the part physics knows about — see
[Exporting the performance](#exporting-the-performance). Missing is fine and silent.

Nothing checks that its length matches the dance. The demo material is a 57 s loop
against a 235.7 s choreography, which is normal for a face animation and is the
player's business, not this task's.

## What the `.npz` has to contain

The schema written by the choreography player (`tools/dance_mujoco.py` in the
motion-planner repository) — one 1-D array per channel, all the same length:

```
time, dt, phase                        phase: 1 dance, 2 cooldown, 3 hold
body_{x,y,z,roll,pitch,yaw}            commanded base pose (m, rad, RPY)
L{0..5}_j{k}                           commanded joint targets (rad)
meas_*                                 the same channels, as actually achieved
kp, kd                                 the gains meas_* was tracked with
bpm, beat_times_sim, beat_times_audio, audio_start_in_sim
```

### The dtype to write

**Every per-sample channel is float32** — `time`, `phase`, `body_*`, `L*_j*`,
`meas_*`, all of them. Everything else (the gains, the beat grid, the scalars) is
whatever it is. That is the whole rule, and it has no exception in it on purpose:
a rule with one is a rule someone gets wrong.

If you generate these files, that is the line to follow. Reading is permissive —
float64 clips load unchanged, and nothing needs regenerating — but the committed
`demo.npz` follows it, at 30.3 MB against 67.7 for the same clip in float64.

Nothing is lost. The conversion emits float32 regardless, because that is what
mjlab holds the reference in, so the extra mantissa never reached training. Checked
rather than assumed: converting from the downcast clip gives a **bit-identical**
`joint_pos` against the float64 original, body positions within 3e-8 m, and
velocities within 1.2e-5 — the position error over the 20 ms control step, which is
where it should land.

`L0`–`L5` are legs **going round the body** — `L0` LF, `L1` LM, `L2` LR, `L3` RR,
`L4` RM, `L5` RF — with five joints on the two front arms and three on the rest,
22 in all. That is not the robot's own joint order and the conversion remaps it;
you do not need to do anything, but it is why a clip written by hand against
`constants.HOME` would come out with its rear legs swapped.

**`meas_*` is what gets imitated, not the commanded channels.** The commanded ones
are PD targets: run them through forward kinematics and the middle legs sit 13 mm
below the rear ones and 6 mm through the floor. See the module docstring in
[`common/dance/motion.py`](../../common/dance/motion.py) for the measurement and for how to switch.

## What happens on the first run

The clip is converted to the format mjlab's `MotionLoader` reads — resampled to the
50 Hz control rate, run through forward kinematics for the per-body poses,
velocities finite-differenced — and cached in `.cache/`. It takes a few seconds and
happens automatically; the cache is keyed on the contents of your `.npz`, so
replacing the file reconverts and nothing has to be remembered.

Before it writes anything it checks that the clip describes a posture this robot
can hold: the four support feet must be coplanar and at the height they sit at when
it stands. A clip recorded against a different model, a different ground height, or
with the base pose and joint angles taken from different takes fails here, with the
measurement in the message, rather than training for a day into a reward ceiling
nothing explains.

## Checking it took

```bash
python scripts/train.py --task jumper.dance --dry-run
python scripts/play.py --task jumper.dance --agent zero
```

The second stands the robot at the reference and plays the clip with no policy —
the fastest way to see whether the dance you loaded is the dance you meant.

## Exporting the performance

`scripts/export.py` writes the four files every task gets — `actor.onnx`,
`layout.json`, `README.md` and the `model_<N>.pt` it came from — and, for this task
only, the performance beside it:

```
tasks/jumper/dance/out/<date-time>/
├── actor.onnx        the policy; `deploy/` reads this
├── layout.json       its contract, and where to find the clip below
├── demo.motion.trajectory.json    the choreography, as text
├── README.md
├── model_<N>.pt      the checkpoint, copied in
└── media/            NOT part of the bundle
    ├── dance.mp4     the policy dancing the whole clip, with the music on it
    ├── music.mp3     the soundtrack on its own
    └── eyes.mp4      your face animation, copied
```

**Your clip leaves with the policy.** This one observes the reference it is
tracking, so it is not runnable without it, and the board is not asked to have a
copy: `scripts/export.py` writes the tables out beside the ONNX as
`<your name>.motion.trajectory.json` — JSON rather than the npz, because parsing
an npz on the target means a zip reader, raw deflate and a `.npy` parser. The
`reference` block in `layout.json` names the file and says how to read it, and
every number in it is read back off the converted clip rather than typed a second
time.

`media/` is a subdirectory because `deploy/` reads the export directory to assemble
what goes on the robot, and a 40 MB video has no business going to an SD card over
the network.

The render takes a few minutes — one frame per control step for the length of the
clip. `--no-video` skips it, which is what you want when you are re-running the
export just to re-check the contract.

**The music is placed at the clip's own `audio_start_in_sim`**, not at zero. This
choreography opens with a 2.0 s silent lead-in; muxing at zero would put the
performance two and a half beats ahead of the music, which looks close enough to be
believed.

If the policy falls partway through, the video stops there and the export says so.
It is not padded out or restarted — a recording of a robot teleporting back to the
reference and carrying on would misrepresent the policy exactly where it matters.
