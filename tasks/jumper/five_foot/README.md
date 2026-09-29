# jumper.five_foot — walking on five legs with the sixth carried as a claw

The four six-foot locomotion tasks differ from each other by a gait prior. This one differs
by a **leg**: the left-front 5-DoF arm is taken out of the action and held stowed in
front of the trunk -- kk-rl-lab's `LF_RAISED`, the pose rl-wbc-fsm's five-foot policy
walked under -- so the policy walks, turns and stands on the remaining five feet. On the
robot a d-pad direction held holds the arm out at a preset and the trigger closes the claw.

It is a port of kk-rl-lab's `hexa_5d/locomotion_5foot`, which trains the same idea
on Isaac Lab. The two repositories **share the asset exactly** — every LF mesh is
byte-identical and every joint has the same axis and the same range — so the arm's
hold pose crosses over unchanged and was re-measured here rather than re-derived.
What does not cross over is the reward set; see [Rewards](#rewards).

```bash
python scripts/train.py --task jumper.five_foot --headless --max-iterations 10000
python scripts/play.py  --task jumper.five_foot
python scripts/play.py  --task jumper.five_foot --objects --num_envs 1     # the pick-place row
python scripts/play.py  --task jumper.five_foot --objects --hold soap --num_envs 1
python scripts/export.py --task jumper.five_foot --checkpoint <run>/model_9999.pt
python tasks/jumper/five_foot/tools/grasp_pose.py --report --check-box --stance
python tasks/jumper/five_foot/tools/grasp_objects.py                       # the claw, end to end
```

**Everything this task needs is in this directory**, apart from the one line in
`tasks/__init__.py` that declares it and its entry in `tests/test_task_parity.py`.
The shared model, the collision scheme, the skeleton, the teleop, the scene registry
and the entry points are master's, untouched; what this task needs of them it does
to its own config:

| here | what it does | instead of |
|---|---|---|
| `jaws.py` | swaps the left claw's two hulls for convex pieces (`meshes/`), with a grip rule | pieces in `jumper.xml` and `constants.py` |
| `env_cfg.py::_ground_the_five_legs` | narrows the six-legged skeleton to five | a `ground_legs` argument on `velocity_env_cfg` |
| `objects.py`, `--objects` | the pick-place row, its friction model and a fixed spawn | `--scene objects` and five scene-registry fields |
| `--hold` (`__init__.py::cli_args`) | a prop already in the claw | a flag in `scripts/play.py` |
| `mdp/curriculum.py` | the payload, switched on once the policy walks at the top command range | a second stage: edit `PAYLOAD_RANGE`, then resume |
| `mdp/rewards.py::_site_pos_w` | foot sites without the rotation upstream cannot convert on CPU | a fix in the vendored `rl/mjlab/entity/data.py` |
| `tools/` | the claw's measurements and the plain-MuJoCo replay | `tools/`, `tools/checks/`, `assets/jumper/tools/` |
| `deploy/lib.rs` | the claw on the right, on the robot: the mirror, as this task's deploy hook | a mirror flag in the deployed controller |

## What is different, in one table

| | six-foot tasks | `jumper.five_foot` |
|---|---|---|
| action | 20 joints | **16** (the carried arm is not driven) |
| observed joints | 20 | **21** (the arm's four, plus the claw's finger) |
| actor observation | 409 (411 with a gait clock) | **407** |
| legs on the ground | 6 | **5** (`RF LM RM LR RR`) |
| gait prior | tripod / tetrapod / ripple, clocked | **`five_foot_gait`, clockless** |
| symmetry augmentation | on | **off** — the robot is no longer symmetric |
| extra command | — | **`body_pose`** (pitch, roll, twist), observed as `base_pose` |
| extra rewards | — | 14 terms + a non-foot ground-contact penalty; `upright` removed |
| extra termination | — | `too_low` at 0.05 m |
| curricula | command, terrain | command, terrain, then **payload** (0 → 0–0.6 kg) |

Nothing measured on this task transfers to the other four, or back. It is not
their control group.

## The three mechanisms

### 1. The arm is held, not merely undriven

**This is the trap the whole task rests on.** mjlab clears `joint_pos_target` to
0.0 on every reset (`EntityData.clear_state`) and the action term writes it back
only for the joints it drives. A joint taken out of the action is therefore
*commanded to 0.0 rad*, not left at its home angle. Measured on `jumper.flat`
on the model before V1.6, whose two grippers were already out of the action:
`LF_J4_joint` left HOME's −1.0 and reached −0.0001 within 25 control steps
(0.5 s). V1.6's HOME has the fingers at 0.0, which hides the trap on the finger
but not on the four arm joints this task takes out.

So `mdp/events.py::hold_carried_arm` writes both the joint state and the PD
target at every reset. Without the second write the claw folds shut over the
first half second of every episode, and nothing anywhere reports it — the
observation, the reward and the episode length all look healthy.
`tests/test_five_foot.py` pins it, with `RF_J4_joint` as the control group
that shows the drift is real.

The hold is re-sampled per episode: the four arm joints inside ±0.10 rad of
`LF_GRASP` (the stow, under its old name), and the finger across its whole working
aperture. So the policy learns to walk with the arm roughly, not exactly, where it
is held — and with the claw anywhere between open and shut.

### 2. Five legs, named once

`env_cfg.py::_ground_the_five_legs` is the one place the ground legs are named,
and it decides four things that have to agree: the geoms `feet_ground_contact`
matches, the sites the foot height scan attaches to, the sites `foot_clearance` /
`foot_slip` read velocities from, and the feet `foot_friction` randomises. Every
per-foot quantity
downstream is then five columns wide in that order — including the critic's
`foot_height`, `foot_air_time`, `foot_contact` and `foot_contact_forces`.

Getting it wrong is silent: five geoms and six sites leaves both shapes plausible
and pairs foot *i* of one with foot *i* of the other.

### 3. The claw is observed, not commanded

kk-rl-lab gives the claw its own `gripper` **command** term, so the policy is told
what the operator is about to do with it. That does not survive the trip: the
deployment vocabulary is fixed on the far side of the contract
(`scripts/export.py::_DEPLOY_TERMS`, and `deploy/fsm/src/obs.rs`, which builds
observation terms and command channels by name), so a `gripper` term would fail
the export rather than reach the robot.

The measurable half survives, and it is the half a locomotion policy needs: the
arm's angles and the claw's aperture are **encoder readings**, so they go into
`joint_pos` / `joint_vel` / `actuator_force` — terms the contract already knows
and the controller already builds. What is given up is anticipation; the policy
reacts to a claw that has started to close rather than to a closure that has been
commanded.

## Measurements

All from `tasks/jumper/five_foot/tools/grasp_pose.py`, read off `assets/jumper/jumper.xml`.

**The hold pose is a stow, since 2026-09-28**: (−30°, −90°, −30°, −75°), kk-rl-lab's
`LF_RAISED`, in `claw.py::LF_GRASP` under the old name. On V1.6.1 the mouth sits at
(0.218, 0.033, −0.025) m in the base frame with its closing axis (−0.90, −0.43, +0.03)
— front to back, so nothing is walked into it, and `grasp_objects.py` grasps none of the
three props at this pose. Clearance 26.2 mm from the forearm outwards and 62.3 mm above
the floor; every joint at 18–66% of its soft range, and every corner of the ±0.10 rad box
legal, the forearm 17.3 mm from the trunk at the worst. The first policy deployed at it
was trained at the grasp-ready pose and re-exported, not retrained (`claw.py` has what
that leaves out).

The grasp-ready pose it replaced was refined on V1.6 and then V1.6.1, with the lift
requirement in the cost (`grasp_pose.py --refine`), and is in `claw.py`'s history
(a82e0f5). The two notes below are about that pose.

> **The lift is in the cost because a pose without it passed everything else.**
> Searched on V1.6 without it, the hold had the jaws exactly across and the arm
> hanging straight down; raising the shoulder 0.30 rad lifted the claw 10 mm and
> `tasks/jumper/five_foot/tools/grasp_objects.py` failed on all three objects. At this pose the
> same lift raises it 20–24 mm and all three are picked up, carried and released.

> **The numbers moved once before, on 2026-09-12, on the model before V1.6, and
> the claw did not.** Two faults in how it was measured, both silent:
>
> * `grasp_pose.py` read each mesh's vertices against its **body** frame, and the
>   compiler stores them in the mesh's own re-oriented frame. Measured at the
>   grasp pose, the palm's vertices landed up to 83 mm from the palm, the
>   finger's up to 58 and the tip pad's up to 32 — so every distance *between*
>   two parts was a distance between two point clouds that were not the parts.
> * The fixed jaw was taken to be `LF_palm_pad_b_link`, the small pad on the **outside**
>   of the hooked jaw's tip, rather than `LF_palm_link`, which is the hooked jaw
>   itself; and the aperture was taken at the closest pair of the two jaws, which
>   is the throat by the knuckle, where nothing is ever held.
>
> Together they reported a claw that opens to 41 mm and shuts at finger −0.30.
> That claw opened to 89 and shut at 0.0, and its mouth centre was 84 mm from
> where it was thought to be. `tasks/jumper/five_foot/claw.py` carries the
> corrected curve, re-measured on V1.6.1, where the jaws were replaced and shut
> moved from 0.0 to +0.10 (`GRIPPER_CLOSED`).

> Against the robot's own six-foot HOME stance this pose has 5.1 mm from the
> shoulder outwards against 5.9, and 24.0 mm from the forearm outwards against
> 26.5: standing on the leg and holding it out are about equally roomy.

**The mouth the solver sees.** A convex hull of a hooked jaw is filled in across
the mouth, so **both** jaws are convex-*decomposed* — `LF_palm_link` into six
pieces and `LF_finger_link` into three. With each jaw as one hull the middle of the
mouth is *inside* the fixed jaw at the knuckle and 8–16 mm short of it over the
grip face, which is what made the claw shove objects it was not touching and
left the moving jaw nothing to push them into. Measured on V1.6 at finger −0.90,
room in the middle of the mouth (`tasks/jumper/five_foot/tools/claw_mouth.py`):

| from the hinge | drawn | one hull | cut |
|---|---|---|---|
| 40 mm | 12.1 mm | 0.0 mm | 10.5 mm |
| 70 mm | 34.7 mm | 23.1 mm | 33.9 mm |
| 100 mm | 31.2 mm | 27.3 mm | 31.1 mm |

**The stance.** Standing on five feet at HOME, the support margin is 92.3 mm
against the six-foot 152.0 mm, and `STAND_Z` is unchanged because the five
remaining feet are still coplanar. Under zero action the robot settles 8.8 mm
low and 1.9° nose-down, because the corner the carried leg used to hold is
unsupported.

**What may be lifted.** Support margin left by each swing group, measured on V1.6
at the former grasp pose (`mdp/rewards.py::GROUP_A` has the table for the stance
the gait groups start from today):

| airborne | supporting | margin |
|---|---|---|
| RM / LR / RR / RM+LR | … | +92.3 mm |
| LM / LM+RR / LM+RM | … | +36.8 mm |
| LR+RR | RF LM RM | +28.6 mm |
| **RF** | LM RM LR RR | **−28.6 mm** |
| RF LM RR | RM LR | −95.2 mm |

Two facts drive the gait design. **The front-right leg cannot be lifted
statically at all** — with the left-front carried, RF is the only front support —
so every RF swing is a dynamic move, and that is the hardest thing about this
task. So RF steps on its own, with the four gait legs planted: the gait is
2+2+1, LM+RR against RM+LR with each half standing on a triangle, and RF's step a
third phase (`mdp/rewards.py::GROUP_A`). The tripod partition with LF deleted had
one half on a two-point line, 86–95 mm off the centre of mass.

**Payload.** A point mass where `claw.py::add_payload_body` seats a held object,
187 mm from the shoulder-pitch axis — measured on V1.6 at the former grasp pose;
the arm is carried stowed now and the payload has a cylinder's inertia:

| payload | CoM shift | support margin | shoulder hold |
|---|---|---|---|
| 100 g | +7.7 mm | +84.3 mm | 0.183 N·m (15% of continuous) |
| 300 g | +21.5 mm | +69.8 mm | 0.549 N·m (46%) |
| 600 g | +39.2 mm | +51.2 mm | 1.099 N·m (92%) |

Statically safe across the range; what it costs is servo headroom and forward
centre of mass, and at 600 g the shoulder holds 92% of its 1.2 N·m continuous
rating before the robot has taken a step.

## Rewards

kk-rl-lab's five-foot task carries **41 reward terms** at the commit this was
ported from (`fb93b00`) — 11 inherited from Isaac Lab's stock velocity
`RewardsCfg` plus 32 of its own, two of which override an inherited one. **Eight
sit at weight 0**, so 33 are live.

The count is worth reading as a history rather than a number. Across the 24
commits of that branch it goes:

| commit | total | at zero | live |
|---|---|---|---|
| `83ad485` initial commit | 27 | 10 | 17 |
| `298ae2a` diagonal_gait 1.0 → 1.5 | 28 | 10 | 18 |
| `9cb7901` mid_roll_symmetry | 32 | 8 | 24 |
| `2049afd` rear_stride per-side deadzone | 34 | 7 | 27 |
| `6e19073` grasp-ready claw pose | 41 | 8 | 33 |

Seven of the eight zero-weight terms are experiments switched off rather than
removed, each still carrying its parameters and its rationale.

The other reason the list is long is structural: Isaac Lab's stock velocity
rewards have no foot-clearance, foot-slip, swing-height, soft-landing or posture
terms, so the task had to supply them. mjlab's does, and the family has already
calibrated them for this robot.

Six terms survived the crossing, and each because it scores something the
six-foot skeleton has no term for **since a six-foot robot does not have the
problem**. Training has added more since, for the same reason, and taken out
`upright` (attitude is scored against the `body_pose` command instead) and three
terms no measurement here ever read (`body_ang_vel`, `angular_momentum`,
`air_time`). `tasks/jumper/five_foot/mdp/rewards.py` has the mapping table and the
derivation of each, and `env_cfg.py` each weight's history; in short:

| term | weight | what it is for |
|---|---|---|
| `five_foot_gait` | +0.5 | the 2+2+1 alternation, clockless: LM+RR against RM+LR, RF stepping alone |
| `lateral_load` | +1.5 | `group_load_balance`: the two gait groups carrying the same force — by group, not by side |
| `duty_balance` | +1.75 | the four mid/rear feet loaded the same fraction of the time |
| `feet_planted` | +1.0 | feet on the ground while nothing is commanded |
| `rf_step_window` | −1.0 | RF off the ground while a gait group is too |
| `rf_drag` | −2.0 | RF sliding on the ground while walking |
| `foot_home` | −1.0 | feet under the body, outside a 40 mm deadzone |
| `rf_outboard` | −10.0 | one-sided: RF standing inside the body's own footprint |
| `nose_pitch` | −15.0 | pitch away from the command, either way |
| `body_roll` | −20.0 | roll away from the command, either way |
| `low_stance` | −3.0 | one-sided, normalised: the body must not sink |
| `body_height_hold` | −5.0 | height away from `STAND_Z`, either way, past a 15 mm deadzone |
| `track_body_pose` | +2.5 (kernel − 1) | pitch and roll tracking the `body_pose` command |
| `body_twist` | +2.5 (kernel − 1) | the trunk, not the feet, carrying the commanded twist |
| `feet_still` | −2.0 | feet moving while nothing is commanded |
| `ground_contact` | −0.5/substep | anything that is not a foot touching the ground |

Four are positive, and on a statically stable robot that always needs
justifying. The gait, load and duty terms would be **perfect while standing
still**, so they are gated on the command asking for motion and are exactly zero
for a standing environment; `feet_planted` pays *only* at a zero command, which
`env_cfg.py` argues for. The two attitude kernels are offset by −1, so they top
out at zero.

**The weights were sized by arithmetic first and have since been moved by
measured runs**; `env_cfg.py` carries each one's history. Velocity tracking is
4.0 + 5.0, `pose` is 1.0, and `upright` is gone.

## The soft mat, and why the contact model does not port

kk-rl-lab has a stage-3 setting that turns the ground into foam: PhysX's implicit
contact spring on the terrain material, calibrated to 75 N/m per contact point
for a measured 14.2 mm sink. **MuJoCo has no contact stiffness**, so none of that
transfers as a number.

It does not need to: this repository has `--scene soft`, which is the same idea
done properly and independently ([`scenes/soft.py`](../../../scenes/soft.py)).

```bash
python scripts/train.py --task jumper.five_foot --scene soft --resume
```

Two findings from porting it are recorded here anyway, because both are dead ends
that cost time and neither is visible from the scene's side.

**Softening the feet does not model soft ground, and it fails silently.** The
foot geoms carry `priority=1` and so decide the foot-vs-ground pair, which makes
them the obvious place for the compliance. Measured (256 envs, 5 s of zero
action, seed 0, native CPU):

| foot contact | sink |
|---|---|
| `solimp` d0 0.8 → 0.2, width 0.002 | 0.00 → 0.22 mm |
| `solimp` width 0.002 → 0.060, d0 0.4 | 0.18 → 4.51 mm |
| `solref` timeconst 0.02 → 0.20 | 0.00 → 4.67 mm |

It saturates at 4.7 mm however soft the contact is made, and the reason is not
the solver: at the softest setting the feet carry **0.0 N of the robot's 26.6 N**
while five non-foot geoms carry up to 11.2 N. The feet have gone through a plane
that stayed rigid for every other geom, and **the robot is standing on its
shins** — a hole with a hard bottom, not a mat. The compliance has to be on the
terrain at a priority above the feet's, which is what `scenes/soft.py` leads with.

**The sink is bounded by the leg, not by the parameters.** Measured with the
compliance correctly on the terrain, the settled feet sit about 8 mm below the
plane before the shins reach it. So a 14 mm sink of the kind kk-rl-lab calibrates
in PhysX puts this robot's legs on the ground in MuJoCo — which is right for a
real mat, where the shins would sink too, and wrong here, where the ground under
them stays rigid.

**What it costs the five-foot task specifically**: `low_stance` will be non-zero
on soft ground, on top of the 8 mm this robot already rests below `STAND_Z` with
the claw carried. That is the surface, not the policy collapsing.

## Training

One run, from scratch. The claw starts empty and a curriculum loads it:

```bash
python scripts/train.py --task jumper.five_foot --headless --max-iterations 10000
```

The payload is the second rung of [`mdp/curriculum.py`](mdp/curriculum.py). The
claw carries nothing until three things hold, and then every environment takes a
load from `PAYLOAD_RANGE` (0–0.6 kg) at its next reset, never mid-episode:

| condition | value | where |
|---|---|---|
| the command curriculum is on its top rung | level 3 | `Curriculum/payload/commands_maxed` |
| ...and has been for | 500 iterations | `Curriculum/payload/settled` |
| linear tracking error there, episode mean | under 0.12 m/s | `Curriculum/payload/lin_err` against `lin_err_bar` |

`Curriculum/payload/loaded` climbs from 0 to 1 over the next episode (about 42
iterations) as the environments reset. On the cold start the thresholds were
chosen against, the error crossed 0.12 between iterations 3014 and 4521, so expect
the payload to come on at around 3500 to 4000. That is an estimate from that
run; the curriculum itself has not trained a policy yet.

The order is not a preference. kk-rl-lab measured what starting from random
weights with the payload on does: 8000 iterations, mean episode length 990 of
1000 and no falls, and 0.001 m/s of travel for a 0.5 m/s command. It found the
local optimum of standing perfectly still, which is the basin this robot's whole
reward design is arranged against — and two cold starts here did the same. It
used to be two stages, train empty and then set `PAYLOAD_RANGE` and resume; the
curriculum is those two stages in one run, with the switch made by the policy's
tracking rather than by hand.

**Resuming.** Both curricula's levels are in the checkpoint, so `--resume` carries
on where the run stopped, claw loaded or not; the log says where each level came
from. `MJRL_COMMAND_LEVEL` / `MJRL_PAYLOAD_LEVEL` override it, and are the only
source for a checkpoint written before levels were saved (2026-09-28). A policy
from such a checkpoint that already carries the load resumes with both set, or it
is trained empty again until the gate re-opens:

```bash
MJRL_COMMAND_LEVEL=3 MJRL_PAYLOAD_LEVEL=1 python scripts/train.py --task jumper.five_foot --resume --headless
```

A policy trained empty (an older stage-one checkpoint) resumes with neither, and
the payload comes on by itself once the commands have re-climbed and the dwell has
passed.

## On the robot: either claw

The policy is trained with the claw on the **left**. On the robot the operator picks
the side, and the right is the left mirrored: the controller shows the policy the robot
reflected in its own x-z plane -- every joint as its left-right twin, times `+1` for the
front arms' shoulder pitch (a y axis) and `-1` for everything else, the IMU and the
command's lateral, yaw, roll and twist flipped -- and reflects the decoded command back.
The policy sees the claw on the left, exactly as it trained.

That is this task's own deploy library, [`deploy/lib.rs`](deploy/lib.rs), compiled into
the controller on every host by `deploy/fsm/build.rs` and called at the controller's
hook points (`deploy/fsm/src/hook.rs`); the controller itself knows nothing about claws.
A side is a mode, so the operator's choice is a switch and switching sides ramps the arms
over before the policy takes over again -- two modes on one export in
`deploy/manifests.json`, `LB` or `V` for the left and `RB` or `B` for the right, each
pressed from walking or from the other claw:

```json
"claw_left":  { "task": "jumper.five_foot", "policy": "<export>", "hook": { "side": "left" },
                "pad":  { "button": "LB", "from": ["locomotion", "claw_right"] },
                "keys": { "key": ["key_v"], "from": ["locomotion", "claw_right"] } },
"claw_right": { "task": "jumper.five_foot", "policy": "<export>", "hook": { "side": "right" },
                "pad":  { "button": "RB", "from": ["locomotion", "claw_left"] },
                "keys": { "key": ["key_b"], "from": ["locomotion", "claw_left"] } }
```

The bundle latches one mode at a time, so the other claw's switch is that claw at once
and a claw's own switch pressed again hands back to walking.

`side` is required, and `"left"` mirrors nothing: it is the trained side, stated so the
pair reads as a choice. Every mode of this task carries the hook, one side or the other,
because the hook is also what drives the claw -- a mode without it is refused.

**The claw closes on the control at its side** -- `claw_left` with the claw on the left,
`LT` on the pad, and `claw_right` on the right, `RT` -- as far as it is squeezed, and
opens when it is let go; on the keyboard both are Space, closing for as long as it is
held (Control-agent 3.1's layout, asked for on 2026-09-29). The policy never drives it:
the finger is out of its action, and the library sets it after the decode, in the
policy's frame, just before the mirror, never more than `SQUEEZE_LEAD` past the finger's
measured angle, plus `LEAD_PER_SPEED` of its closing speed so a claw closing on nothing
shuts in 0.3 s rather than 0.8 (`claw.py` has why, measured). `controls.yaml` declares both claws among
the controls the task keeps (`task:`) and binds each on both devices by name, which is
what refuses a command binding either and what the controller holds the hook's reads to.
Each mode answers only its own side's -- in the left claw `RT` does nothing, and the
bundle's manual lists `LT` and Space there, not `RT`. Replay answers the same file the
same way (`mdp/gripper.py`): the claw there is always on the left, so it follows
`claw_left`, on `LT` or Space.

**The arm is held out while its control is held.** Up on the d-pad holds it straight
out with the thumb up, down with the thumb down, left with the thumb-web up -- rl-wbc-fsm's
V3.1 `[arm]` presets, at its gains (`kp` 3, `kd` 0.3) and pace -- and let go, it goes back
to the stow. On the keyboard the same three are Shift, Alt and Ctrl, either key of each.
Two held together are the pose between them, the element-wise mean: up and left is the
thumb at 45 degrees, 3.1's "混合时，角度为45度". Thumb up and thumb down together are
neither -- the d-pad's two ends cannot both be down, but Shift and Alt can. Right alone
does nothing. That is Control-agent 3.1's momentary layout, asked for on 2026-09-29;
until then a direction was a press, read when it was let go and pressed again to stow,
and V3.1's upper diagonals swung the arm 45 degrees out to a side -- gone with the press
they were chosen on. `menu` and a direction is a dance: the chord leaves this mode on the
press and never moves the arm. The keyboard's dances, Ctrl and a digit, are not pressed
in a claw mode at all, since Ctrl is the arm's here. The keyboard's twist is H and ;,
as in walking (the doc's revision of 2026-09-29; until then it was Shift + J L there and
unbound here, where Shift holds the arm out).

**The right stick's pitch moves the arm at the stow**, as V3.1's does: the shoulder follows
it, 0.2 rad at a full stick nose down and 1.0 nose up. **A preset held outranks it**
(3.1: "优先级高于Ry对手臂的控制"): while one is held the pitch leaves the arm alone,
and V3.1's elbow follow at up and at down is gone. So the controller holds the arm for as
long as the claw mode runs, at 3 / 0.3 rather than the policy's 10, the stow included. The
policy is shown the arm stowed all that time: it was trained with the arm within 0.10 rad
of the hold, and a preset is a radian away. On the right claw the table and the follow are
mirrored; the thumb's direction is the hand's own, so nothing swaps. The finger opens to
60 degrees with the claw let go and 90 while a preset is held, V3.1's numbers, wider than
the 37 the policy trained on, which is as far as it is shown. `deploy/lib.rs` has the
table, the reasons, and the clearances measured at each, the tight ones included. The
task's own replay (`play --task jumper.five_foot`) does not answer the arm's controls; an
app does.

`play --app` plays all of this in mjlab -- either claw, the triggers, the d-pad and the
keys -- because it runs the controller's own targets and gains into the simulator, past
the hook (`rl/mjrl/app_play.py`); so do the browser and the robot.

**What the mirror assumes, measured.** That the right half is the left's mirror image,
joint for joint. For every pair the axis-derived sign maps each joint's range onto its
twin's exactly (`tests/test_five_foot_deploy.py`, which also holds the library's tables
against `common/mdp/symmetry.py` and the model's axes). The home poses agree to 0.043 rad,
worst at the front arms' `J3` -- the V1.6.1 right half is a rotated copy of the left rather
than a mirrored one, open with the mechanical team -- so the mirrored home is that far off
the robot's own, inside the ramp's 0.10 rad tolerance. `rl-wbc-fsm`'s `carry_mirror` did
the same thing in the controller, with its signs guessed from which home angles share a
sign; the axes are the measurement that guess stood in for.

## Where to look when it does not work

- **The gait term stays near zero.** Expected early; it is gated on both
  `moving_gate` and `cycling_gate`, so a policy that has not started walking
  collects nothing. If it stays low while tracking improves, the policy has
  settled on something other than the diagonal alternation — most likely a ripple
  (one foot airborne at a time), which `five_foot_gait` scores at only 0.13.
  `sigma` is the knob; a ripple *order* is a different task, and the margins above
  say it is the more stable one.
- **The robot lists to one side.** `body_roll` (−20, against the commanded roll)
  is the term for it; `lateral_load` balances the two gait groups now, not the two
  sides.
- **The stance splays.** `foot_home`, and the note in its docstring about how it
  overlaps `pose` — the two are the same constraint in task and joint space, and
  kk-rl-lab replaced its joint-space deadzones with the task-space form rather
  than running both.
- **`ground_contact` stays large.** The robot is standing on something that is not
  a foot. On five legs this is the failure kk-rl-lab had to double its penalty
  for: at a standstill the policy sat on its shins and its claw.
