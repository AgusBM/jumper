# jumper.jump jump task: version record

Iteration history of the reference-guided jump task (`tasks/jumper/jump/`). The
reward design is ported from kk-rl-lab's `tasks/hexa_5d/jump/` (a scheme brought
up on Isaac Lab on the same 22-DoF hexa_5d), and the reference motion is
`tasks/jumper/jump/ref/high_jump_flat.npz` (crouch-jump-tuck-land, 1.47 s, t_go=0.3 s).

The full checkpoint for each version (`model_2999.pt`, the final model at 3000
iterations) is kept in `logs/jumper/jumper.jump/<run dir>/`; paths are in the table
below -- none of which still load (see the next paragraph). Replay one trained
on the task as it is now:

```bash
python scripts/play.py --task jumper.jump --num_envs 16 \
  --checkpoint logs/jumper/jumper.jump/<run dir>/model_<N>.pt
```

Replay loops whole attempts on its own: the task keeps its 3.5 s episode under
`play=True` and spawns at the start of the motion rather than at the training-time
random phase, so what you watch is a crouch-jump-land from the beginning.

**None of the checkpoints below still load.** `base_lin_vel` has left the actor
group: an IMU gives the angular half, and the linear half comes only from a state
estimator this robot does not have, so `scripts/export.py` refuses a policy that
observes it -- the task trained and replayed perfectly and could not go on the
robot. The actor is now 167 inputs where V1--V4 are 170, and a strict load stops
on the size mismatch rather than replaying something. The rows stay because what
they measured is still what they measured; getting a policy back means retraining.

Quantitative evaluation (apex rise / flight time / push-off speed, 3 episodes with
a deterministic policy):

```bash
python tools/checks/jump_eval.py --ckpt logs/jumper/jumper.jump/<run dir>/model_2999.pt
```

## Version summary

All evaluations were measured on the same machine (RTX 3050, warp:cuda, 200 Hz
control) under the same test protocol.

**V1--V4 were run against the previous reference, `high_jump.npz`**, and the
numbers below are that comparison. They are kept because they were measured;
what they are measured against has changed.

That reference was generated under a constant 2.0 N·m at unlimited speed, so its
0.162 m / 0.335 s was **not strictly attainable** on the real-robot torque-speed
curve -- V4's 98% was already close to the real motor's limit, and the remaining
2% was chasing torque the motors cannot deliver at speed.

`high_jump_flat.npz` is that objection fixed at the source: generated against the
motor's torque-speed envelope (`meta["motor"]`, peak 1.7464 N·m, this robot's
`EFFORT_LIMIT` exactly). It asks for less -- 0.141 m rise against 0.162, and
0.076 rad of tilt against 0.037 -- and what it asks for is reachable.

There is no V5 row because nothing has been **evaluated** against it, not
because nothing has been run. Training has started (the first runs are
`logs/jumper/jumper.jump/2026-09-22_*`) and a policy has been exported --
`model_3100`, the jump the `jumper` bundle ships -- which is what proved the
deployment path end to end. None of it has been through the protocol the
rows below were measured under, and a row filled in from anything less would be
the one thing this table cannot survive: a number that looks like the others
and was not measured the same way.

| | V1 | V2 | V3 | V4 | old npz ref |
|---|---|---|---|---|---|
| **run dir** | `2026-09-03_12-51-23` | `2026-09-07_09-34-36` | `2026-09-07_15-16-33` | `2026-09-07_18-44-59` | — |
| Net rise (m) | 0.153 | 0.154 | 0.157 | **0.159** | 0.162 |
| Flight time (s) | 0.29 | 0.300 | 0.311 | **0.312** | 0.335 |
| Push-off vz (m/s) | 1.78 | 1.75 | 1.78 | 1.78 | 1.87 |
| Crouch-phase ground contact ratio* | 0.88/0.66/0.54 | 0.74/0.71/0.61 | **0.88/0.84/0.68** | 0.54/0.64/0.69 | 1.0 |
| episode consistency | ±0.001 | ±0.001 | ±0.001 | **frame-identical** | — |

\* Actual ground-contact ratio of the six feet across the three crouch phases
(0–0.1 / 0.1–0.2 / 0.2–0.3); the reference requires 1.0 for all three.
Source: `tools/checks/jump_phase_diag.py`.

## Changes in each version

### V1 — initial port (2026-09-03)

reward / termination / RSI reset ported item by item from kk-rl-lab; 200 Hz
control, kp=20, residual actions (baseline = reference actual state q), 2048 envs.

**Open issue**: the middle legs do not touch the ground during the crouch, and
landing is early. Blamed on the reward at the time, which was the wrong direction.

### V2 — reward weight adjustment (2026-09-07 morning)

`ref_contact` 1.0 → 2.5 (the lab's empirical value for the spin task); after two
interruptions from GPU driver crashes it was reduced to 1024 envs and retrained.
Crouch ground contact barely moved (0.74/0.71/0.61), and push-off vz actually
dropped (1.78 → 1.75) — the first proof that **the problem is not the reward
weight**.

### V3 — align the physics to the recording environment (2026-09-07 afternoon)

Traced the root cause of "lifting the legs during the crouch" to the physics layer
and aligned item by item with kk-rl-lab's recording environment (`gen_jump_ref.py`
+ `robot.mjcf`):

- **Geometric reconstruction of the contact schedule**: the npz `contact` channel
  was recorded with a 1 N force threshold, so pre-crouch unloading (foot tip z
  stationary, force < 1 N) was misrecorded as "off the ground", which kept teaching
  the policy to lift the middle legs. Rebuilt from foot-tip geometric height
  instead (grounded baseline +10 mm threshold, accommodating the 5.6 mm swing of
  the forearm origin; 86 mm airborne, a clear separation)
- **Action baseline q → q_cmd**: in the npz, q is the state actually reached and
  q_cmd is the controller command; during push-off the two differ by 0.3 rad on
  average (2.2 rad peak) — using q as the baseline amounts to double lag. A zero
  action now reproduces the recording controller exactly
- **Joint speed limit** 31.42 rad/s (300 rpm), aligned with the recording
  environment's per-physics-step clamping and Isaac's velocity_limit_sim.
  *Since removed*: that wall was the generator's rectangular motor model.
  `high_jump_flat.npz` was generated against the measured curve, whose only
  speed limit is its 611 rpm cutoff, and reaches 338 rpm itself; the servo
  curve now governs speed and torque alone, as in every other task
  (`tests/test_hardware_facts.py`)
- Contact parameter override entry point (contact_params) — a later ablation showed
  the XML's collision class already carries the recorded solimp, so it had no
  actual effect; this port carries no such override

Effect: crouch ground contact improved across the board, flight time 0.29 → 0.31 s,
height +4 mm.

### V4 — measured servo torque-speed curve (2026-09-07 evening)

Wired in the measured common curve for all servos (`servo_tau_max`, constants.py):
1.7464 N·m stall, 293.5 rpm knee, exponential decay at 240.5 rpm beyond it, zero at
611 rpm. Each control step writes `actuator_forcerange` from the current speed (also
written once on reset, to cover high-speed RSI spawns). *Since replaced*: the curve is
now `ServoCurveActuator` (`tasks/jumper/common/actuator.py`), read from
`assets/jumper/motor/motor_config.yaml`, clipping the PD torque every physics substep,
with a 0.3 s peak budget that derates to the 1.2 N·m continuous rating.

Result: 12.5% less peak torque, and **the jump height rose rather than fell**
(0.159 m, 98%) — during push-off the servos mostly work in the flat top of the
curve, and the curve pushes the policy to abandon brute-force twisting in the
high-speed region in favor of deeper crouches and better six-leg coordination.
Early crouch ground contact fell back to 0.54: once torque becomes expensive,
"crouching while planted" costs more, so the policy shifts the balance back toward
"lift a foot to trade for push-off" — this is the reward optimum under the
1.75 N·m real-robot constraint, not a bug.

## Key conclusions (debugging record)

1. **The reference motion's contact channel has threshold artifacts**; the contact
   schedule must be reconstructed geometrically before imitation
2. **q (actual state) ≠ q_cmd (controller command)**: the imitation reward uses q,
   the residual action baseline uses q_cmd
3. This exp-kernel imitation scheme assumes the "reference is physically
   reachable"; when simulation mismatch breaks that assumption, the policy's
   self-invented motion is the optimal response to a bad premise — **fix the
   physics before fixing the reward**
4. The peak power point of this curve is at the knee of 293.5 rpm (53.7 W), and the
   old reference's peak joint speed of 31.42 rad/s (the 300 rpm wall) sat right on top
   of it (`high_jump_flat.npz` reaches 338 rpm, past the knee) — the
   correct form of "saturating the servo" is working at the peak power point, not
   torque saturation
5. A pure imitation reward symmetrically penalizes exceeding the reference and
   **structurally does not incentivize "jumping higher"**; pushing the maximum jump
   height requires changing the task definition (regenerate a maximal reference, or
   take a reference-free route — kk-rl-lab's `tasks/hexa_5d/jumping/` has a
   ready-made scheme). `jumper.ref_free_jump` is that route here: the same
   recording fed forward into the action and as a joint-tracking prior while
   training, both fading to nothing, height paid uncapped, and no trajectory on
   the robot
6. The network structure is pinned to three layers (512,256,128): the jump
   observation is 167-dimensional and does not follow the fourth layer the shared
   baseline added for walking tasks. It was pinned so that existing checkpoints
   would go on loading strictly, and that reason has since expired -- dropping
   `base_lin_vel` made the input three narrower, so nothing trained before it
   loads either way

## What travels with the policy

A residual policy is not runnable without the recording it corrects, so the
recording is part of the contract rather than something the board is assumed to
already have. `env_cfg` attaches a `reference_contract` -- a callable, because
loading the npz is work that `--list` and a dry run must not do -- and
`scripts/export.py` writes the tables beside `actor.onnx` as
`high_jump_flat.trajectory.json`, with a `reference` block in `layout.json`
saying how to read them. JSON rather than the npz: parsing an npz on the target
means a zip reader, raw deflate and a `.npy` parser for one array.

**Both tables ship, and they are not interchangeable.** `q` is the state the
recording reached and is what `ref_future` previews; `q_cmd` is the command the
recording's controller actually sent and is the residual's baseline. They are not
even the same length -- 368 frames at 250 Hz against 295 at 200 Hz. Every number
in the block is read off the loaded reference rather than restated in
`env_cfg.py`: a rate typed twice is a board playing a slightly different
recording than the policy trained against, and nothing would say so.

The block says `starts_on_entry`: switching into the mode is the whole of
asking for a jump. The recording starts when the policy takes over, once the
switch-in ramp has reached the home pose, and when it has played out the mode
hands back. Which control switches into it belongs to the deploy manifest --
`deploy/manifests.json` binds `A`.

That is also what the policy trained on. Every spawn is reference-state
initialised past its go (`rsi_fraction=1.0`, a phase in [0, 0.9)) with a fresh
history, and the command resamples only at reset, so `time_since_go` is never
negative in training: starting at phase 0 on the first tick after the hand-off
is the edge of that range. Until 2026-09-26 the jump waited instead for a go
named `jump_go`, bound to `A` on release, and stood in its go frame waiting for
it -- a state the policy had never seen. A motion that does want a person to pick
the moment inside its mode still can: its block names a `go_event`, the manifest
binds a control to that name, and the crate refuses the two halves disagreeing
in either direction. With neither, the motion runs on the controller's timer
from mode entry, which is enough to watch it on a bench and not something a
person can use.

## Diagnostic tools

| Script | Purpose |
|---|---|
| `tools/checks/jump_eval.py` | Jump quality evaluation (apex/flight/push-off speed) |
| `tools/checks/jump_phase_diag.py` | Ground-contact ratio by phase vs the reference schedule |
| `tools/checks/jump_ref_play.py` | Pure npz reference-motion replay (zero action, no policy) |
| `tools/checks/jump_ref_play_diag.py` | Ground-contact diagnostics for zero-action replay (isolates physics vs policy) |
| `tools/checks/jump_replay_cmp.py` | Compares the two replay methods, q / q_cmd |
