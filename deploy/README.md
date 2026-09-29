# deploy — from a checkpoint to whatever runs it

```
model_*.pt ──scripts/export.py──▶ actor.onnx ──convert/──▶ actor.rknn
                                  layout.json ──┐             │
                                                ▼             ▼
                          scripts/deploy.py [--manifest <name>]
                                                │
                                                ▼
                        controller   observation → NPU → joint targets → dds/
```

An **export directory** is one policy. A **bundle** is what every host loads: the FSM
config, one policy per mode in each model format a host runs, each host's build of the
controller, and a manual of the pad and the keyboard — one directory, and one `.app` of
it, from which the board, a browser and `play` each take their own part.
[`BUNDLE.md`](BUNDLE.md) is its format, file by file, and what a host that loads one has
to do.

A reference-guided policy carries one file more. `jumper.jump`'s action is a residual on a
recorded motion and `jumper.dance` is scored against one, so the recording travels with the
policy as `<name>.trajectory.json` beside the contract, which names it in a `reference`
block; the bundler copies it in and refuses a mode whose contract names one that is not
there. Without it every host refuses the mode as it opens the bundle, naming the file,
rather than loading it and refusing at its first tick. `fsm/src/trajectory.rs` is the
reader.

| | what it is | runs where |
|---|---|---|
| [`convert/`](convert/README.md) | ONNX → RKNN for the RK3576 NPU | a PC (Docker, any host) |
| [`dds/`](dds/README.md) | the robot's wire interface: IDL, QoS, topics | build input |
| [`fsm/`](fsm/README.md) | the controller: observation, action decode, state machine | the board, a browser, and `play` |

`fsm/` is one crate with three hosts. `controller` is the robot's whole program — there is
no separate controller linking it as a library, which is what there used to be.

Each directory has its own README with the detail; this one is the map and the part
that belongs to none of them.

## The principle the whole path rests on

**Every number the robot uses comes from the policy's own `layout.json`, and joints are
paired by name.** Nothing downstream re-declares a gain, a scale, or a joint order.

That principle was stated here long before it was true of everything. Two numbers were
quietly violating it, both in a config file beside a binary where nothing held them
against the robot they described:

- the **joint position clamp**, the last thing between a policy and the motors, was
  `[-3.30, 3.30]` for all 22 joints — the placeholder from the crate's unit tests. Twenty
  of the 22 have both stops inside it, so the clamp ran every tick and could not fire.
- the **stick scales**: 0.80 m/s forward for a policy whose command range is 0.50. Full
  deflection asked for 1.6× anything the curriculum ever commanded, which looks like a
  policy that is bad at going fast rather than one being asked a question it never saw.

Both now come from the contract (`joint_limits`, `command_ranges`). The lesson is the
general one: a number a simulator could have measured does not belong in a host's config,
and "it has always been that value" is not evidence that anyone chose it.

That is not tidiness. On this robot the wire carries 22 joints and a policy takes whatever
subset it was trained on: the locomotion tasks and `jumper.jump` observe and drive 20,
`jumper.dance` all 22. The two the locomotion policies leave alone are at wire indices **4
and 9**, in the middle:

```
obs_joint_order (20) → wire [0,1,2,3, 5,6,7,8, 10,11,…,21]
                                    ↑ 4      ↑ 9   are the fingers
```

Pair those by index instead of by name and the fifth joint onward is wrong, on a real
robot, with nothing raised. Verified against the live machine: the board's
`[robot] joint_names` and the contract's `wire_joint_order` are the same 22 names in the
same order, and every name in the contract resolves.

## What is checked, and where

Each hop is checked against the previous one, because every failure on this path is
silent — a policy that runs and behaves wrongly, not one that crashes.

| hop | check | measured |
|---|---|---|
| torch → ONNX | `scripts/export.py::_validate_onnx` | 3e-7 |
| ONNX → RKNN | `convert/onnx2rknn.py`, simulator vs onnxruntime | 1.1e-3 (fp16 floor) |
| contract → controller | `fsm/`: layout, observation layout, FFI struct layouts, QoS | unit tests |
| controller → robot | **the board**: runtime versions, IDL, joint order | see below |
| host → host | `reference.json` replayed on each: `--check-reference` | 0 on the controller, `joint_torque` included |

That last row used to end "`joint_torque` 2.1 N·m by design", and the 2.1 N·m was not a
design. The servo reports torque — `MotorControl_State` carries it and `dds.rs::take_state`
had been reading it into `state.tau` the whole time — and the claim that it does not was
inherited from the C++ controller, which was wrong about this robot. Declaring the term
reconstructed made the controller re-derive a number it already had, and that gap was the
only thing the replay ever disagreed on. The term is `Measured` now, the replay feeds it
the recorded value, and it agrees exactly like every other term.

`scripts/export.py` additionally refuses to export at all when the actor observes something
the robot cannot measure (`base_lin_vel` — no state estimator) or a term the on-robot
controller cannot build. Better to fail the export than to ship a policy that cannot run.

## Verified against the board

Over ssh, read-only:

| | board | this repository |
|---|---|---|
| OS / glibc | Ubuntu 22.04.5, glibc 2.35 | `fsm/Dockerfile` builds on 22.04 |
| kernel | 6.1.99-rt36 (PREEMPT_RT) | — |
| librknnrt | 2.3.2 | 2.3.2 pinned by `fsm/vendor/rknpu2/fetch.sh`, toolkit 2.3.2 |
| CycloneDDS | 11.0.1, `libddsc.so.11` | container builds the same soname |
| libmbus | 2.4.0 | — |
| IDL headers | 4, from libmbus | identical to what `fsm/build.rs` generated then; since the 2026-09-29 refresh to mbus `d9875988f065` `MotorControl::State` is 800 → 832 bytes, which libmbus 2.4.0 predates -- not rechecked on the board (see [`dds/README.md`](dds/README.md)) |
| `joint_names` | 22 | same order as `wire_joint_order` |

`.claude/skills/deploy/scripts/check_board.py` re-runs all of it.

**A controller was running as a service and active during those checks.** Everything above
is read-only for that reason. Publishing motor commands while one is running means two
controllers driving the same joints — which is also why `controller` has a `--dry-run`.

## Open

- **`rknn_run` on real hardware is not yet verified.** The NPU cannot be exercised off the
  board, so the arithmetic has only been checked through the toolkit's simulator.
- **The control loop has never run against a bus.** `fsm/src/bin/controller.rs` exists and
  `--dry-run` / `--check-reference` go through every step before the bus is opened, but the
  loop's correctness is almost entirely timing — 1 kHz publish against each mode's own
  inference rate — 200 Hz for `jumper.posture`'s walk and the jump, 50 for the claw modes,
  the dances and the gestures — ZOH plus a rate-invariant EMA,
  mode-switch ramps — and none of that is measurable off the board. It also makes no
  attempt at real-time scheduling: no `SCHED_FIFO`, no `mlockall`, no jitter accounting.
- **Nothing holds the two ends of a DDS domain together.** `controller` defaults both to
  0, which is what the board is configured with and what `control-pod-svc` publishes on;
  they agree today and will not stay in agreement by themselves. The warning this bullet
  used to carry was about `robot_control` moving to domain 2 — no longer this controller's
  problem, because it reads the pad directly, which is a reminder that a stale warning is
  worse than none.
- **No gripper state reaches the cascade.** `gripper_active` is in the cascade vocabulary,
  every host passes `false`, and a config using it is refused for that reason: the state
  came from `robot_control`, and this controller reads the pad instead. A claw is driven
  the other way, by its own task -- `jumper.five_foot`'s deploy hook closes it on the
  trigger at its side, or Space, and holds its arm out at a preset while a d-pad
  direction, or Shift, Alt or Ctrl, is held: controls the task's `controls.yaml` keeps
  for it by name (`task:`) -- and no rule sees it.
