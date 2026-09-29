# Motor characteristics — the jumper joint servo

What the actuator hardware is: the measured torque-speed curve, friction, thermal derating and
the control interface. Unlike `../camera/camera_config.yaml`, which is read to build the model
and to set up the dToF sensor, **the torque-speed curve here is live** — it bounds the torque in
every simulation step and therefore in training.

| File | What it holds |
|---|---|
| `motor_config.yaml` | The specification, and **the source of truth for the curve** — `tasks/jumper/common/actuator.py` loads it at import. Two entries: `joint_servo`, the 50:1 servo every joint runs, and `joint_servo_100`, the 100:1 variant, defined but not wired (see *The 100:1 servo* below) |
| `torque_speed_curve.csv` | Raw bench samples for `joint_servo`, if any. Header-only — the curve arrived in analytic form |
| `torque_speed_100_24v_reported.csv` | The samples `joint_servo_100`'s curve is fitted to, digitised from the bench report — one row per rpm, 52–189 |

**`null` is not a default.** `load_servo_curve` raises on it, on a missing key, on a string where
a number belongs, and on a `model:` naming a shape the actuator does not implement. Every one of
those would otherwise become a silently chosen motor. Editing this file wrongly stops the next
env build with the file, the field and the reason in the message — verified by breaking it.

### What is duplicated and what guards it

The curve lives here and nowhere else. The remaining overlaps are deliberate and each has a test,
because a copy nobody checks is how a specification starts lying:

| Also appears in | Guarded by |
|---|---|
| `in_mjcf_*` friction ← `jumper.xml` | `test_specification_does_not_restate_what_the_mjcf_owns` compiles the MJCF and compares |
| `control.sim_stiffness/sim_damping` ← `constants.py` | same test |
| `torque_just_below_cutoff_nm`, `cutoff_is_discontinuous`, `symmetric_in_torque_sign` | `test_specification_claims_about_the_curve_are_true_of_the_code` evaluates the actuator instead of trusting the text |

`measured_*` is the one block that is *supposed* to disagree with the simulation — see below.

## The curve

This is `joint_servo`, the 50:1 servo — the one every joint runs.

```
tau_max(w) = plateau                                for |w| <= corner
             plateau * exp(-(|w| - corner) / decay) for corner < |w| < cutoff
             0                                      for |w| >= cutoff
```

| | rpm | rad/s |
|---|---|---|
| plateau torque | **1.7464 N·m** | |
| corner speed | 293.5 | 30.7352 |
| decay constant | 240.5 | 25.1851 |
| cutoff speed | 611.0 | 63.9838 |

| speed | 0 | 200 rpm | 293.5 rpm | 350 rpm | 500 rpm | 610.9 rpm | 611 rpm |
|---|---|---|---|---|---|---|---|
| torque | 1.746 | 1.746 | 1.746 | 1.381 | 0.740 | **0.466** | **0** |

Two properties matter more than the shape:

**The plateau is 1.7464, not the 2.0 the repository assumed.** The old figure had no recorded
provenance — it appeared in `constants.py`, in the MJCF's `actuatorfrcrange` and in the exported
contract, and no file said whether it was a continuous rating, a peak, or a number inherited from
the Isaac configuration. It was 12.7% too generous. Standing does not notice: measured at HOME on
the V1.6 model, the worst leg joint (LM_knee) needs **0.2296 N·m, 13.1% of the plateau**.

**The plateau is a peak rating, holdable for 300 ms.** After that the ceiling becomes
`min(1.2 N·m, curve(speed))` — see the next section; it is enforced.

**The cutoff is discontinuous.** The exponential still stands at 0.466 N·m — 26.7% of peak — when
the cutoff drops it to zero. It is implemented as given rather than smoothed, but two consequences
are untested: a joint hovering at the ceiling can chatter between full torque and none, and above
it nothing slows the joint but `damping` and `frictionloss`, together under 0.1 N·m, whereas a
real servo brakes *hardest* at overspeed. In the rollouts measured so far no joint has reached it.

## The peak budget

The plateau may be held for **300 ms from cold**; after that the ceiling drops to the smaller of
the continuous rating and the curve:

| ceiling | 0 | 20 rad/s | 30.7 (corner) | 40.2 | 50 | 63 |
|---|---|---|---|---|---|---|
| cold | 1.746 | 1.746 | 1.746 | 1.200 | 0.813 | 0.485 |
| hot | 1.200 | 1.200 | 1.200 | 1.200 | 0.813 | 0.485 |
| binding | thermal | thermal | thermal | crossover | curve | curve |

The two constraints cross at **40.19 rad/s (383.7 rpm)**, where the curve has fallen to exactly
1.2 N·m. Above it the servo is already speed-limited below its continuous rating and the budget
is irrelevant; below it the budget is the only thing standing between a policy and torque the
hardware will not deliver.

### The time constant is derived, not measured

`ServoCurveActuator` carries a per-joint normalised winding temperature and integrates
`dθ/dt = ((τ/τ_cont)² − θ) / τ_th` from **the torque actually produced**, not the torque demanded
— it is current that heats a winding, and the clamp has already happened. θ = 1 is the trip
point, which is what "continuous rating" means: the torque whose steady-state temperature sits
exactly there.

That model turns the two supplied numbers into a third for free:

```
1 − exp(−peak_duration / τ_th) = (continuous / plateau)²
τ_th = 0.300 / ln(1 / (1 − (1.2/1.7464)²)) = 0.4695 s
```

**Cooling is therefore not a free parameter** — the same constant governs it, so "300 ms at peak,
then 1.2" fully specifies the behaviour including partial duty cycles, with no third number to
guess. `test_thermal_time_constant_reproduces_the_budget_it_came_from` integrates the model
forward and checks it trips at exactly 300 ms, closing the loop on the algebra; a separate test
drives the real actuator at the real substep rate and checks the derate is wired to anything at
all.

**Training and replay use the same actuator** — one construction site, `get_jumper_robot_cfg`, and
the `play=True` branch never reaches the robot entity;
`test_every_jumper_task_gets_the_servo_in_both_modes` checks all four tasks in both modes. The one
asymmetry is the reset: training episodes are 20 s, so each one hands back a cold servo and its
first 300 ms have full peak available (1.5% of the episode), while replay episodes are
effectively unbounded and only reset on a fall. Replay is therefore the *harsher* of the two, and
a policy can legitimately look slightly weaker there than its training curves suggest.

What this assumes, and what a bench run would have to confirm: that the servo is first-order
thermal, that an episode starts cold (`reset` zeroes the state), and that its controller derates
by clamping rather than on a schedule of its own. One artefact of clamping: just below the trip
point the model alternates — one substep at full curve pushes θ over 1, and it takes roughly
250 ms at 1.2 N·m to come back under. That averages to about the continuous rating with brief
peaks, which is right, but it is bang-bang where a real controller would derate smoothly.

### What could not be measured

*Does the current gait actually exceed 300 ms at peak?* Unanswered — though no longer for want of
something to run. The artefact this was written against, `tasks/jumper/tripod/out/flat/actor.onnx`,
expects a 74-dimensional observation where the current env produces 411: it predates the 5-step
history window, the `actuator_force` term and the removal of `base_lin_vel`, so replaying it
would have meant feeding it noise. `tasks/jumper/tripod/out/rough/` was an export of the shape
current then, and nobody took the rollout. Both have since been deleted with every other committed
export (the last commit carrying them is `89fa4a8`), so answering this starts from a fresh export.

Note that "5.5% of the time near the limit" in `docs/DESIGN.md` cannot answer it either: that is
a *fraction*, and 5.5% arriving as one continuous second is a violation while the same 5.5%
scattered as 5 ms touches is not. Consecutive dwell has never been measured.

## How it reaches the simulation

`tasks/jumper/common/actuator.py::ServoCurveActuator` — a subclass of mjlab's `IdealPdActuator`
that clips the PD output to the curve, with the four numbers read out of `motor_config.yaml` by
`load_servo_curve`. It lives in `tasks/`, so nothing under the vendored `rl/` changed.
`make_actuator_cfg` in `constants.py` builds it, and `EFFORT_LIMIT` is now the plateau.

Three decisions inside it are worth knowing:

- **The spec is loaded, not mirrored.** Keeping the numbers in Python with a test comparing them
  against this file also works, but it makes the file that documents the hardware a mirror of the
  file that runs it, and the useful direction is the other way round: bench numbers arrive in the
  YAML. The cost is that an asset file is now load-bearing for training, which is why the loader
  refuses everything ambiguous rather than defaulting past it.

- **The curve is in `control_law`; `compute` is overridden for the thermal budget.** The curve
  alone would stay fusable (`fused_group.py::_is_fusable` compares the bound function). The peak
  budget is state, so `compute` lowers `force_limit` for a hot joint and integrates heat, which
  leaves the fused path — at no cost here, because one actuator covers all 22 joints and fusion
  batches across actuators.
- **The clip runs every physics substep**, not once per policy step: `manager_based_rl_env.py`
  calls `apply_action()` inside its decimation loop. At 5 ms physics and decimation 4 that is four
  evaluations per 50 Hz policy step, each with the current joint velocity, on both backends.

### Why not mjlab's own DC motors

mjlab ships two, and both clip to a **straight line** — `DcMotorActuatorCfg`'s
`dc_motor_clip` directly, and `BuiltinDcMotorActuatorCfg` through `tau = K(V - K·w)/R`. A line
through (0, plateau) and (cutoff, 0) gives 0.87 N·m at the corner speed where the true limit is
the full 1.75: wrong by a factor of two exactly where the servo spends its time. Neither fits, so
the curve got an actuator of its own.

One thing mjlab's model does that this one does not: `dc_motor_clip` is **asymmetric**. It clips
against the *signed* velocity, so at a given speed it allows more braking than driving — which is
what a real motor does, since back-EMF subtracts from the supply when driving with the motion and
adds to it when braking against it.

| at speed | mjlab drive | mjlab brake | ours (both) |
|---|---|---|---|
| 16 rad/s | 1.310 | 1.746 | 1.746 |
| 32 rad/s | 0.873 | 1.746 | 1.661 |
| 48 rad/s | 0.436 | 1.746 | 0.880 |
| 60 rad/s | 0.109 | 1.746 | 0.546 |

What was measured here is a **magnitude**, so a magnitude is applied to both directions. That
under-models braking, and the decision to leave it that way is measured rather than assumed:

- **Below the corner speed the two are identical.** Our limit is the plateau there, and mjlab's
  braking limit is the plateau too (it is capped by `force_limit`). There is nothing to choose.
- Above the corner, over 28864 samples of deliberately violent actions, **0.30% were at the clamp
  while braking past the corner** — the only place the choice has any effect. The torque withheld
  there was a median of 0.187 N·m (max 0.742). Under gentler, more realistic actions joints
  barely reach the corner at all.

So the symmetric reading costs little, and it errs toward a weaker simulated robot, which is the
safe direction for transfer. Changing it would mean inventing a braking law that was never
measured. **Get bench data for the braking direction before modelling it.**

This is also the mild face of the cutoff problem above: at and past 611 rpm the symmetric reading
gives zero *braking* torque as well as zero drive, which is the opposite of a real servo. No
sample has yet reached that speed, so it stays theoretical — but if a trained gait ever gets a
joint there, this is where it will bite first.

### That it actually binds

The failure to rule out is silent: a curve that is configured, documented and tested while never
reaching the simulation trains a policy against a motor the robot does not have, and the logs just
look like a slightly stronger robot.

Measured over 25 policy steps × 8 environments on the native CPU backend, recording each
`(velocity, torque)` pair **inside the control law** — sampling `actuator_force` after `env.step`
instead compares a torque clipped against the start-of-substep velocity with a limit recomputed
from the end-of-substep velocity, and reports violations for a correct actuator:

| actuator | samples past the corner | worst \|τ\| − limit | violations |
|---|---|---|---|
| `ServoCurveActuator` | 2.1% | **0.000000 N·m** | **0 / 85536** |
| `IdealPdActuator` (control) | 2.1% | +1.096 N·m | 1619 / 85536 |

The control group is mjlab's PD at the same gains and the same flat cap — what the robot used
before. It produces 1.7464 N·m at 55.6 rad/s where the curve allows 0.650, which is what makes the
clean result above mean something. `tests/test_servo_curve.py` runs both.

Note the 2.1%: reaching the decay region at all needed deliberately violent actions. Under normal
commands the joints stay inside the plateau, so **the practical effect of adopting the curve is
mostly the lower plateau, not yet the speed dependence.**

## The 100:1 servo — defined, not wired

`joint_servo_100` is the 100:1 variant of the joint servo. **No joint carries it**: its
`applies_to` is empty, nothing builds an actuator from it, and every joint still runs the 50:1
curve above. `load_linear_decay_curve` reads it and `linear_decay_torque_limit` evaluates it, and
neither is loaded at import, so a broken entry here cannot stop a build that does not use it.
`tests/test_servo_curve_100.py` pins the definition to its samples, and pins that it stays unwired.

```
tau_max(w) = plateau                                  for |w| <= corner
             plateau * (zero - |w|) / (zero - corner) for corner < |w| < zero
             0                                        for |w| >= zero
```

| | rpm | rad/s |
|---|---|---|
| plateau torque | **4.712 N·m**, servo-reported | |
| corner speed | 67.3 | 7.0476 |
| zero-torque speed | 200.8, extrapolated | 21.0277 |

### Where the numbers come from

Bench measurements of the 100:1 servo at 24 V, speed against reported and true torque: every
unit pooled into one, a median line per direction with a P10–P90 band. It was digitised on
2026-09-26 from the 1812×775 original, following each curve by its colour and calibrating against
the axis ticks, to about ±0.02 N·m and ±0.2 rpm. The wiggles in the samples — in the positive
direction 4.85 N·m at 65 rpm, between 4.73 at 60 and 4.55 at 70 — are in the chart, not in the
digitising.

`torque_speed_100_24v_reported.csv` holds one row per rpm from 52 to 189:

| column | |
|---|---|
| `pos_reported_nm`, `neg_reported_nm` | the median line of each direction, as a magnitude |
| `mean_reported_nm` | their mean — **the column the curve is fitted to** |
| `band_p10_nm`, `band_p90_nm` | the P10–P90 band, directions merged: the lower P10, the higher P90 |
| `p10_occluded` | 1 where the P10 edge lies under the chart's true-torque band (164–183 rpm); the value there is an upper bound, about 0.05 N·m high |
| `kt_ratio_pos`, `kt_ratio_neg` | true / reported torque, from the chart's right-hand panel |

### Why a second shape

Least squares on `mean_reported_nm`, every rpm weighted equally:

| shape | rms | worst |
|---|---|---|
| plateau, then a line to zero (`linear_decay`) | **0.077 N·m** | 0.237 at 97 rpm |
| `joint_servo`'s exponential, at its best for these samples | 0.188 N·m | 0.49 at 84 rpm |

The exponential under-reads the middle and over-reads the top — 0.91 N·m at 189 rpm against a
measured 0.51 — and because it never reaches zero it would also need a cutoff dropping about
0.8 N·m at once. The line is continuous, and it is the shape a DC motor takes behind a
current-limited driver: the plateau is the current limit, the line is back-EMF eating into the
supply. That makes it, on the driving side, exactly what mjlab's `dc_motor_clip` computes, with
`force_limit` = 4.712, `velocity_limit` = 21.0277 rad/s and `saturation_effort` = 7.087 N·m (the
line's value at standstill). The 50:1 servo could not use that actuator; this one could, which is
worth weighing against its asymmetric braking when it is wired.

### What it does not say

- **The torque is the servo's report, not the shaft's.** Reported torque was asked for on
  2026-09-26, hence `torque_basis: servo_reported`. True over reported is 0.651 in the positive
  direction and 0.614 in the negative (medians over 80–150 rpm), falling to 0.3–0.4 at 52 rpm and
  near zero at 189. Applied to a joint as it stands, this curve simulates a servo about 1.6 times
  stronger than its shaft. `joint_servo`'s basis was never recorded (`torque_basis: null`), so the
  two entries cannot yet be compared number for number.
- **Nothing below 52 rpm.** The plateau is assumed to hold down to standstill.
- **Nothing above 189 rpm.** The zero-torque speed is the fitted line, extrapolated.
- **The directions differ** by up to 0.30 N·m, 0.11 on average. The definition is their mean,
  applied as a magnitude in both directions, as `joint_servo` is.
- **No thermal budget.** `peak_duration_s` and `thermal_continuous_torque_nm` are `null`. The
  50:1 servo's 300 ms and 1.2 N·m are not borrowed: a different gearbox on the same motor is a
  different thermal load, and a copied number would read as a measured one.
- **24 V only.** The report's 48 V, 100:1 section has no curve; its author notes the servo could
  not hold the theoretical peak torque for more than a few runs.

### What wiring it would take

1. `ServoCurveActuator` runs only `exp_decay_with_cutoff`. It would have to dispatch on the
   model, with a zero-torque speed among its `param_names` — or this servo would go on
   `DcMotorActuatorCfg`, above.
2. A thermal budget, measured or deliberately switched off: `ServoCurveActuator` derates from
   one, and `load_servo_curve` refuses `null`.
3. A decision on the torque basis: scale by the true/reported ratio, or train on reported torque
   knowing the shaft delivers less.
4. `assets/jumper/tools/export_web_robot.py` writes the curve into the web robot package as
   `exp_decay_with_cutoff`, and `scripts/export.py` writes the plateau into every `layout.json`
   as `control.effort_limit`. Both move with it.
5. A retrain. Every policy so far has trained against the 50:1 servo.

## Friction: the simulation and the hardware disagree

| | in the MJCF (live) | measured | |
|---|---|---|---|
| armature | 0.0015 kg·m² | 6.325e-4 | MJCF 2.4× higher |
| frictionloss | 0.011 N·m | **0.08474** | measured **7.7× higher** |
| viscous damping | 0.017 N·m·s/rad | 0.00639 | MJCF 2.7× higher |

**Nothing has been changed to match.** These were supplied to be recorded, and unlike the curve,
applying them is a separate decision with a large effect — 7.7× the dry friction is not a
refinement, it is a different robot, and the directions differ (too much rotor inertia and viscous
damping, far too little dry friction). mjlab's `ActuatorCfg` takes all three, so wiring them is a
three-line change to `make_actuator_cfg` followed by a retrain.

## Traps

**Units.** The curve was measured in rpm; MuJoCo is rad/s. The conversion happens exactly once, in
`actuator.py`, and `tests/test_servo_curve.py` compares the rad/s implementation against the
original rpm function across a sweep that lands exactly on both corners. A curve scaled the wrong
way is still monotone, still bounded by the plateau, and still looks like a plausible servo.

**Reference frame.** Everything here is at the **output shaft**, after the gearbox — the frame
MuJoCo's joint works in. Applying a gear ratio to numbers that already include it is a factor of
N² error in the curve, and it will look plausible.

**This is not a drop-in.** The actuator is weaker at speed and weaker at rest than what existing
checkpoints trained against, and the PD moved from MuJoCo's native position servo to a Python-side
control law. Any policy trained before this needs retraining, and the comparison to make is
against the old policy rather than an assumption that the more faithful model wins.

**The contract has a consumer.** `scripts/export.py` writes `control.effort_limit` into each
`layout.json`, and this repository's controller reads it (`deploy/fsm/src/action.rs`). That value
has changed from 2.0 to 1.7464, and the contract still describes a constant limit rather than a
curve — the hardware enforces the curve physically, so a controller clamping at the plateau is
safe, but the contract is now an incomplete description of the actuator. Worth settling with the deployment side before the next
export.
