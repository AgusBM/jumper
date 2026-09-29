# fsm — the shape it is being given

**Status: built.** The core is there -- the premise, the observation port, the
FSM cascade and the two-phase tick -- and so are all three hosts. Each section
below says which. Read [`README.md`](README.md) for the crate as a whole.

---

## The problem

The same three steps — **observation → inference → execution** — now have three
places they have to run:

| | observation from | inference on | execution to |
|---|---|---|---|
| `kk-rl-mjlab`'s `play` | mjlab's simulation | ONNX, on the host | mjlab actuators |
| BEUNLIMITED's web sim | MuJoCo WASM | ONNX, in the browser | the same actuators |
| the robot | servos and IMU | RKNN, on the NPU | DDS to the motor controller |

That logic existed **three times**: this crate in Rust, `rl-wbc-fsm` in C++,
`policy.ts` in TypeScript. They agreed, and they agreed the expensive way — by
people reading each other's code. A sign corrected in one of them left the other
two driving a robot that was wrong with every other number still correct.

The fix is not to pick one and delete the others: two of the three consumers
cannot run the third's language. The fix is that **the logic is one crate and
the three differences are ports**. The C++ is no longer one of them — this crate
ships its own binary, `src/bin/controller.rs`, and `rl-wbc-fsm` is kept as
reading: its priority cascade, its per-mode contracts and its safe-state rule
are all here, and where a decision below says "the C++ does X", that is where X
came from.

## The premise, verified

Everything below assumes the observation/action/FSM logic contains no syscall,
because a browser cannot provide one. That is now checked rather than assumed:

```bash
cargo check --target wasm32-unknown-unknown --no-default-features   # passes
cargo check                                                          # passes
cargo test                                                           # passes
```

`device` is a feature (default on, so the ordinary build is unchanged) covering
`dds` and `rknn`; `build.rs` compiles in the task hooks and then returns without it,
so `idlc`, `bindgen` and `cc` never run for a target that could not use them. `obs.rs`,
`action.rs`, `layout.rs` and `types.rs` — 1282 lines when this was written, 2666 now —
compiled to `wasm32` **unmodified**. They were already clean; the feature only
stopped the two device modules from dragging them down.

If that check ever fails, the core has stopped being the core, and the web
simulator is back to reimplementing it.

---

## Shape

```
controller  (pure: no I/O, no clock of its own, one build for every target)
  layout    the policy's contract, parsed                        exists
  obs       RawState -> observation vector                       exists
  action    action vector -> joint targets                       exists
  types     the values that cross a port                         exists
  source    what a state source can supply, and how              exists
  config    modes and transition rules                           exists
  vocabulary  the words a pad and a keyboard can say             exists
  trajectory  the recorded motion a reference-guided mode reads  exists
  fsm       the ordered cascade                                  exists
  control   the two-phase tick, the regimes, the per-mode runtimes  exists
  operator  a task's controls: pad and keys -> each mode's command  exists
  guide     what each control does in each mode, as data         exists
  hook      a task's own code at fixed points of a mode's loop   exists
  reference a recorded run replayed, and where hosts differ      exists

hosts  (one per environment, outside the core, each in its own language)
  device    Rust, this crate's `device` feature                 exists
  web       JavaScript, via wasm-bindgen                        exists
  play      Python, via pyo3                                    exists

the bundle format  (std::fs, so the one module a browser does not compile)
  bundle    open a directory, check it, build a host from it    exists
```

The core never calls out. A host calls in, is told what to do, and does it. That
is forced by wasm and it is the right shape anyway: it makes the whole of the
control logic a pure function of its inputs, which is the only way the three
environments can be compared at all.

---

## The observation port

### Ports are not interchangeable, and pretending they are is the failure

The obvious design is one trait with three implementations. It is wrong, and the
evidence is a term already in the shipped contract.

`joint_torque` is 100 numbers of the current jumper observation (20 joints × 5
frames of history). Where it comes from:

| environment | source |
|---|---|
| mjlab | `data.actuator_force` — what the solver applied |
| MuJoCo WASM | the same |
| the robot | `MotorControl_State.motors[i].tau` — what the servo reports |

That last row used to read `tau = kp*(t − q) − kd*qd`, clamped to
`effort_limit`, with the note that the servo does not report torque. **It does**,
and `dds.rs::take_state` had been reading it into `state.tau` the whole time;
the claim came from the C++ controller (`action.cpp:100-103`) and was never
true of this robot. Declaring the term reconstructed made the controller
re-derive a number it already had, and that is the one term the reference
replay ever disagreed on — 2.146 N·m worst case.

The reconstruction is still there, in `applied_torque`, and is still what a
host without servo torque would use. Which one a host takes is decided by what
its `SourceCapability` declares, not by a branch in the loop.

Two consequences the naive design hides:

1. **"observation → inference → execution" is not a chain.** It has a feedback
   edge, and the edge closes in a different place per environment.
2. **A term's name does not tell you what it is.** Two runs can agree on every
   dimension, scale and index and still differ in what the numbers mean.

### So a source declares what it can supply, and how

**Built — `src/source.rs`.**

```rust
enum Provenance {
    Simulated,      // the physics engine computed it: ground truth
    Measured,       // a sensor read it
    Reconstructed,  // this controller computed it from what it commanded
    Estimated,      // a state estimator produced it
}
```

A source declares a `{term -> Provenance}` map. `types.rs` already has half of
this, in the one place it was unavoidable:

```rust
/// Body linear velocity. Almost never available: it needs a state estimator,
/// not an IMU. `has_lin_vel` says whether it means anything.
pub lin_vel: [f32; 3],
pub has_lin_vel: bool,
```

Generalising it is the change. A contract asking for a term no source supplies is
**refused at load**, not discovered at tick 40 000.

### The check produces a list, not a verdict

The valuable output is not "ok". It is the terms whose provenance differs from
the one they had in training:

```
locomotion on `device`:
  joint_torque   trained Simulated, here Measured
  base_lin_vel   trained Simulated, here Estimated
  (6 terms unchanged)
```

**That list is the sim2real gap, enumerated.** Nobody can see it today. It costs
one field per term in the exported contract — every term kk-rl-mjlab exports is
`Simulated`, so producing it is nearly free, and the value is entirely on the
consuming side.

---

## The inference port

### Two phases, because the host owns the model

In the browser the model runs in onnxruntime-web, which is JavaScript. A wasm
module cannot call it and return within one synchronous call. So the tick splits:

**Built — `src/control.rs`.** The cascade decides *which* mode; this decides what
a tick does inside one.

```rust
enum Step {
    /// No inference this tick — warm start, mode-switch ramp, or a hold state.
    /// The targets are final; read them from `command()`.
    Hold,
    /// The host must run `observation()` through this mode's model and call
    /// `resume`.
    Infer { mode: String },
}

fn tick(&mut self, input: &TickInput<'_>, now: Micros) -> Result<Step, Error>;
fn observation(&self) -> &[f32];              // borrowed: no allocation per tick
fn resume(&mut self, action: &[f32]) -> Result<&MotorCommand, Error>;
fn command(&self) -> &MotorCommand;           // valid after either phase
```

`Hold` carries nothing, because the command is readable either way and a payload on
one arm would be a second place to read it from. `TickInput` is the bundle of things
only the host knows — the state, the command, whether each is fresh, the latched
buttons, each mode's operator and its task's controls — so everything derivable from
them is derived here instead of subtly differently in three places.

This is **not** a concession to wasm. `rl-wbc-fsm` already works this way: the
output loop runs at 1 kHz and inference at the active contract's `control_hz`
(50 for locomotion, 200 for jump), with the last decoded target held and an EMA
moving toward it in between. Splitting the tick only writes that fact into the
type instead of leaving it in a timer comparison.

It also makes the decisive property testable: `tick` is a pure function of its
`TickInput` and `now`, so a recorded episode replays bit-exactly. The C++ has
this only for `jump_gate.hpp`, and says why — *"pure, time-injected, unit-tested
in isolation from the DDS control loop"*.

### Rate lives in the core, not the host

`control_hz` comes from the active mode's contract, and the core decides which
ticks infer. A host that decided for itself would be free to get it wrong, and
running a policy at the wrong rate aliases it *and* re-scales
`joint_pos_rate_limit`, which is per-tick.

**Per mode, not per bundle.** `Bundle::output_rate_hz` — which is
`Bundle::control_hz` renamed, because the old name claimed to be the policy rate
and is only the loop's — returns the **fastest** mode's rate and sets the output
filter, nothing else. It used to require every mode to agree, and refused a
bundle holding `jumper.tripod` at 50 Hz beside `jumper.jump` at 200. The reason given
was that one filter serves the whole controller, and the filter is the one thing
this cannot break: `ema_alpha_from_cutoff` is rate-invariant by construction,
which is the entire point of writing it as a cutoff rather than as a coefficient.
What the agreement cost was real. `jumper.jump` runs at 200 Hz because its push-off
is 40 ms, which at 50 Hz is two command steps; requiring agreement meant either
retraining it slower — throwing away the reason it exists — or shipping it in a
bundle of its own, which no operator can switch into. The slower mode simply
infers every fourth tick.

### A non-inferring engine is a state, not an error

`rl-wbc-fsm` ships a `StubEngine` returning zeros, and `is_stub()` so the loop
refuses to call that "running". The interlock that matters:

> **"Zero action == hold still" is not universal.** It holds only while the
> decode baseline is the home pose. For a reference-residual policy a zero action
> means "track the recording exactly" — a full open-loop jump.

So the stub must be visible to the FSM, not hidden behind the port. It is:
`rknn::make_engine` falls back to a stub **with a reason**, `DeviceHost::stub_modes`
reports every one, `controller` prints them at startup and again the first time
such a mode runs, and `Outcome::Inferred` carries `stub` so the loop cannot call
it running.

And a mode whose decode uses a reference **refuses to run at all** while its
engine is a stub. `DeviceHost::note_stub` hands the fact it owns — this engine
is a stub — to `Controller::interlock_stub_engine`, which owns the rule, because
the contract is there: a `reference` block with `residual_action` bars the mode,
anything else does not. A barred mode takes the path a stale-feedback tick
already takes — keep the mode, hold steadily — rather than inventing a second
way to refuse, and `stub_modes` says which mode and why.

Only a residual is barred. For an ordinary policy a stub's zeros **are** the home
pose held, which is what `stub_modes` promises and is true; barring those too
would make a bundle pending conversion useless for no safety gain. The test
carries both directions.

**This paragraph claimed that interlock for some time while nothing did it.** It
was written when the crate had no reference support, read as a decision already
taken, and stayed after the support arrived — so a `jumper.jump` whose `.rknn` had
simply not been converted was a stub returning zeros, and zero residual on
`q_cmd` is the recording played open-loop at 200 Hz with kp=20. A design
document that describes a safety property it does not have is worse than one
that omits it, because the omission is visible.

---

## The execution port

The core produces joint targets in **wire order** and nothing else. What differs:

| | |
|---|---|
| mjlab | write actuator targets; mjlab's `<position>` actuators close the loop |
| web | the same, through the WASM model |
| device | a DDS `MotorCommand` with `pos`/`kp`/`kd` and `tau = 0` — the MCU closes the torque loop |

Two invariants the port may not round off:

- **Wire order is not policy order.** `layout.rs` pairs them by joint *name*;
  the two never share a type, deliberately. A wrong order is a robot that runs
  and is wrong.
- **The clamps are per-tick.** `joint_pos_rate_limit` is rad *per tick*, so a
  host running at a different rate silently changes it. It belongs to the core.

---

## The FSM

### States are data; transitions are an ordered list

`rl-wbc-fsm` has no state enum and no transition table: states are strings in
TOML, and transitions are a priority cascade in `fsm.cpp`, recomputed every tick.
The cascade order is the design — safety before operator intent, recovery before
hand-off — and a `condition -> state` map would lose exactly that.

**Built — `src/config.rs`, `src/fsm.rs`.**

So transitions become an **ordered rule list**, first match wins:

```toml
[[fsm.rule]]  when = "feedback_stale"      enter = "safe"
[[fsm.rule]]  when = "tilted"              enter = "safe"
[[fsm.rule]]  when = "in_state:safe"       enter = "@initial"
[[fsm.rule]]  when = "warm_start_reached"  enter = "@warm_start_ref"
[[fsm.rule]]  when = "in_state:warmstart"  enter = "@stay"
[[fsm.rule]]  when = "button:jump"         enter = "jump"
[[fsm.rule]]  when = "button:claw_left"    enter = "claw_left"
[[fsm.rule]]  when = "always"              enter = "locomotion"
```

Two rules about the rules:

- **`when` is a closed vocabulary, not an expression language.** An unknown word
  fails the load. An expression language here cannot be validated, cannot be
  tested, and would need a parser inside the wasm module.
- **The last rule must be `always`**, checked at load. "No rule matched" then
  cannot happen, and a test can assert the cascade is total.

Starting vocabulary — every entry is something the C++ already decides on:

| `when` | true when |
|---|---|
| `always` | always; required as the last rule |
| `feedback_stale` | no state sample within `state_timeout_ms` |
| `tilted` | tilt exceeds `tilt_limit`, unless the active state sets `skip_tilt_check` |
| `command_stale` | no operator command within `command_timeout_ms` |
| `in_state:<name>` | the active state is `<name>` |
| `warm_start_reached` | the active state is a warm start and the measured pose reached its target |
| `button:<id>` | the operator's latched mode equals `<id>` |
| `gripper_active` | either gripper is engaged — see below |

`@initial` and `@warm_start_ref` are indirections into `[fsm]`. `@stay` matches
and changes nothing — needed because a cascade without it quietly claims every
tick ends somewhere new. A warm start ignores the operator until it has finished,
which the C++ says with a bare `return` mid-function; here it is a rule, visible
beside the order it depends on. The first draft had no `@stay` and walked a robot
away mid-ramp, because the cascade fell through to `always`. The test that caught
it is `the_ordinary_path`.

`gripper_active` is the one word that parses and can never be true. Its state came
from `robot_control`, which this controller stopped reading when it started reading
the pad directly, and the pad carries no gripper — every host passes `false`. So a
config *using* it is refused at load, for the same reason a rule naming a button
nothing latches is: a rule that can never fire looks exactly like one that merely
has not. The word stays in the vocabulary, because the vocabulary is what a file
may say and this will be true again when something supplies it.

### Entry conditions stay on the state

`skip_tilt_check` suspends the attitude fallback **only while the state is
already active** — the tick that enters it is still checked, so the robot cannot
launch from an already-tipped pose. That is a guard on the state, not an edge, and
it stays where the C++ put it. A config whose safe state sets it fails to load,
for the reason the C++ gives: *"exempting it would leave no attitude protection
anywhere."*

### Switching modes ramps; it does not blend

Entering a mode with a model:

1. record the pose at entry;
2. direct-PD ramp toward **that model's own default pose**, at the ramp gains,
   not the policy's;
3. hand off only when the measured pose is within `pose_reach_tol` of it —
   **not when the ramp duration elapses**. Soft gains mean the hand-off waits
   indefinitely and the robot soft-holds. That is by design; raise the gains.

Shipped values: `mode_switch_ramp_s = 1.0`, `pose_reach_tol = 0.10` rad,
`ramp_kp = 15`, `ramp_kd = 0.8`, in the servo's units (`deploy/jumper.controller.toml`
says why not the 0.15 / 0.01 first copied from the C++). Without this a mode change
jumps from the previous stance into the new policy's first command.

### Per-mode, and per-mode only

The split the C++ found, kept verbatim:

> This file holds only what is GLOBAL across policies. Each FSM mode's I/O
> contract (obs/action layout, home pose, action_scale, gains) is loaded at
> runtime from that policy's layout, shipped next to its model.

One mode names one model file. Everything about how that policy is driven comes
from its own contract. The TOML adds only bench knobs, composed rather than
overriding:

```
effective kp = contract.kp * state.kp_scale + state.kp_offset
```

The trained gain stays in the contract; the bench adjustment stays in the config;
neither can be mistaken for the other.

Modes are built **up front**, in `Controller::new`, and so are their engines in
`Bundle::into_host`. This paragraph said "lazily on first entry"; building on
first entry would put every way a contract can be refused — a term this host
cannot supply, a missing recording, a model that will not load — behind an
operator pressing a button, on a robot that is already standing. Start-up is
where a refusal is cheap. A switch costs nothing either way.

### A mode may be driven by a recording

A velocity policy needs nothing but its contract: the command is three numbers
and every observation is measurable now. A reference-guided one is not runnable
without the motion it was trained against, so the motion travels with it — a
`reference` block in the contract, and a companion `<name>.trajectory.json`
beside it that the bundler copies in and `trajectory.rs` reads. Two tasks use it
and they use it differently, which is why it is a block rather than a flag:

| | `jumper.jump` | `jumper.dance` |
|---|---|---|
| the action is | a residual on the recorded `q_cmd` | the joints outright |
| the recording starts | on mode entry (a `go` the operator pressed, until 2026-09-26) | on mode entry |
| it is over | 1.17 s after it starts, from its go frame | after 235 s |

Three things about it are load-time refusals rather than runtime surprises,
because each is silent when wrong. A contract naming a recording that is not
beside it is refused when the bundle opens — the alternative is a mode that loads
and fails at its first tick, which is the right answer at the wrong moment. A
table in a different joint order than the robot's is refused, because a permuted
trajectory is a robot moving the wrong limbs and nothing downstream can see it.
And a residual declared on any baseline other than `q_cmd` is refused rather than
read off `q`: the two differ by about 0.3 rad across the push-off, which is the
difference between a jump and a twitch.

`q` and `q_cmd` are not interchangeable and are not even the same length — 368
frames at 250 Hz against 295 at 200 for `high_jump_flat`. `q` is the state the
recording reached and feeds the preview the policy observes; `q_cmd` is what the
recorded controller commanded and is the residual's baseline.

**A recorded motion ends**, which is the one way it is unlike a gait.
`Controller::take_completed` names the mode whose recording has played out, the
host releases that mode's button on its own latch, and the cascade's final
`always` rule catches the robot. The controller does not leave the mode itself:
that would be a second thing deciding modes, which is the whole complaint against
the C++'s jump latch.

### What a command is differs per mode

From `config.hpp`:

> Policies do not agree on what a command is: locomotion wants
> (lin_vel_x, lin_vel_y, yaw_rate), jump wants (jump_x, jump_y, jump_go).

So a mode may declare its own `command_terms`, falling back to the global list.
The controls half of that has since moved: a task's `controls.yaml` is a list of
commands, one entry per command term an operator drives, and the contract
carries it as `operator_controller/2` -- `jumper.posture` drives the walk and a
posture from one pad, or the keyboard, through it, and `src/operator.rs` is the
whole of how. What
has not moved is `command_terms` itself, the order of the `commands` block,
which is still the FSM config's; taking it from each contract's controls block
instead is the remaining piece.

---

## Deliberately not copied

- `walk_enter_speed`, `walk_exit_speed`, `walk_exit_hold_s` — parsed by the C++,
  read by nothing. Vestigial stand↔walk hysteresis.
- The jump one-shot latch works by rewriting the operator's mode before the FSM
  sees it. It is correct and well-tested, but it means the FSM is not the only
  thing deciding modes. Here `button:jump` is an ordinary cascade rule, and the
  motion starts when the mode is entered, once the switch-in ramp is done. A
  recording that wants a person to pick the moment declares a `go_event` instead:
  a binding the mode reads by name rather than a transition, because a push-off
  40 ms wide is an event inside a mode and not a mode. Leaving is the cascade's too:
  when the recording plays out, `Controller::take_completed` names the mode, the
  host releases that button on its own latch, and the final `always` rule catches
  the robot. One way out, rather than a second one written beside it.
- Key and button semantics were compiled into `control-agent-cpp`, with only axis
  ranges in a file. Here the controller reads the pad itself, and which control
  means what is `[[fsm.button]]` (from the manifest) and each task's
  `controls.yaml`.

---

## What exists today

`README.md` describes the crate as it stands: a state machine whose states are a
file, three hosts that call it, and `src/bin/controller.rs`, which on the robot is
the whole program. This section used to read "no state machine, no `main.rs`" and
to warn that the README's "two modes" were `rl-wbc-fsm`'s; neither is true now.

Built as of this document:

| | |
|---|---|
| the `device` feature split and the wasm check | `Cargo.toml`, `build.rs`, `lib.rs` |
| provenance, capability, divergence list | `src/source.rs` |
| modes and the ordered rule list, with its load-time checks | `src/config.rs` |
| the words a control may be named, read from the dictionary | `src/vocabulary.rs` |
| the cascade, and the guards that belong to a state | `src/fsm.rs` |
| the two-phase tick, four regimes, per-mode runtimes | `src/control.rs` |
| each mode's reading of the pad and keys, command shaping | `src/operator.rs` |
| the pad's manual as data (`pad_guide`) | `src/guide.rs` |
| a task's deploy hook, compiled in from `tasks/**/deploy/lib.rs` by `build.rs` | `src/hook.rs` |
| the recorded motion a reference-guided mode is driven by | `src/trajectory.rs` |
| a recorded run replayed, and where a host differs | `src/reference.rs` |
| the bundle format, one reader for every host with a filesystem | `src/bundle.rs` |
| the robot's host: freshness, the button latch, engine dispatch | `src/device.rs` |
| the browser's host: the JS-facing `WebFsm` | `src/web.rs` |
| `play`'s host: the pyo3 `Fsm` | `src/py.rs` |
| the robot's program: the loop, the two clocks, the shutdown path | `src/bin/controller.rs` |

156 tests. Two of them are control groups, because both of these modules replace
something that was previously written into code and would pass every obvious
assertion while ignoring the file: `the_cascade_is_the_file_s_rather_than_agreeing_with_it`
moves the tilt rule below the operator's button and requires a tilted robot to
jump, and `the_check_reads_what_training_recorded_rather_than_assuming_it` trains
a term as reconstructed and requires simulation to be reported as the divergence.

`control.rs` earned a third test the hard way. Its first `publish` used
`cmd.pos` as both the standing target and the filtered output, so after one
sample the filter converged on itself and stopped -- at 50 Hz on a 1 kHz loop,
nineteen ticks in twenty did nothing. On a robot that reads as a controller
that feels sluggish, with nothing logged.
`the_output_filter_starts_where_the_robot_is` requires convergence to the
decoded target, and measured 0.0295 against 0.25.

`device.rs`'s two latch tests were run against deliberately broken versions
before being trusted. A latch acting on the trigger's *level* rather than its
rising edge fails `a_held_trigger_is_one_request` -- but only because the press
is four frames long: at three, level-acting toggles an odd number of times and
ends latched, passing for exactly the wrong reason. Dropping `latch.forget()`
on a stale command fails `a_latched_button_does_not_outlive_the_operator`.

## Hosts

`device.rs` is the first, and the one whose I/O already existed. It owns what
the core may not: a clock (so `feedback_stale` and `command_stale` can be
decided), the engines, and the latch that turns the operator's trigger pulses
into a `button:<name>` the rules can match. `pump` is compiled against the real
`DdsIo` -- four calls, and that compile is the check that the shapes still fit.

`[[fsm.button]]` names one thing the operator can ask for and which control asks
it: a pad button or a keyboard key, one device per entry, and two entries with
one name share its latch -- the pad's A and the keyboard's Space are both the
jump. It also carries the gesture (`rise`, `fall`, `hold`, `toggle`, or a click
count `single` … `quintuple` counted inside `[fsm] click_window_ms`), an optional
`with` modifier (a pad button, or `ctrl` / `shift` / `alt` for a key), `from`
(the states it may be pressed in) and `leaves` (latches it lets go of). Since
2026-09-29 the keyboard is a path of its own: there is no table making a key a
pad button any more. That
mapping is a decision, and in `control-agent-cpp` it is compiled in, so changing
which button jumps means rebuilding a different program.

Every one of its load-time checks is a binding that loads and then does nothing,
or does two things: a rule naming a button no entry declares; an entry with
neither a pad button nor a key, or with both; two entries on the same control
with the same gesture; a `pad` or a `with` the robot's service does not publish;
a `key` the dictionary does not list; a `with` that is not a modifier of its
entry's device; a `from` naming no state; a `leaves` that is not a moment or
names nothing that latches; a control used as a modifier that is also bound to
`hold`; a binding no rule reacts to; and any key `[fsm]` does not read.
`[fsm] exclusive` makes the latches one at a time, the last switch pressed
being the mode.

**`event = true` is the exception to the last of those.** A control read by a
*mode* rather than by a rule — a recorded motion's `go`, matched against the
contract's `reference.go_event` — has no rule reacting to it, which is exactly
the shape of a dead binding. So it is declared rather than inferred, and
`Controller::new` checks it in both directions: an `event` no contract names, and
a `go_event` no button declares, are each a control that loads and does nothing.

The words themselves come from `controller/vocabulary.json`, schema
`kk-control-vocabulary/2`, `include_str!`-ed so a board carries no file and a
browser fetches nothing. It was `kk-gamepad-vocabulary/1` while the pad was all
it described; `keys` joined it because `key` was the one field nothing checked —
`pad` and `with` were held against the dictionary from the start and `key` took
any string at all, so a typo loaded, bound nothing, and looked exactly like a
rule that had not fired.

In a bundle the halves come from two files on purpose. `deploy/manifests.json` says
which button or key switches into which mode, and from which modes, and its
top-level `buttons` list says what the bindings that are *not* mode switches are —
that is where a recording's `go` would live (the jumper bundle has none: its jump
starts on entry); each task's `controls.yaml` says what the sticks and keys do
inside a mode. Neither is wrong on its own when both pick the same control, and
neither can catch it, so `Bundle::open` is where the sum is checked
(`operator::check_switches`, which every host's operators run again): opening the
bundle is the first place both exist.

`web.rs` is the second, and the one the two-phase tick exists for: the model runs
in onnxruntime-web, which is JavaScript, so a wasm module cannot call it and
return inside one synchronous call. Everything crosses as a flat `Float32Array`
-- no structs per tick -- and the configuration crosses as text once, because a
browser has no filesystem.

```bash
cargo build --release --target wasm32-unknown-unknown --no-default-features --features web
wasm-bindgen --target web --no-typescript --out-dir <dir> \
  target/wasm32-unknown-unknown/release/mjrl_fsm.wasm
```

776 KB of wasm and 19 KB of glue, against MuJoCo's 10.2 MB and onnxruntime's
14 MB in the same page.

**It found a bug on first contact with a real file, which is the point.** Pointed
at an exported `layout.json`, it refused: *observation term 'velocity_commands'
is not implemented*. `rl-wbc-fsm` has mapped that name to the builder's
`commands` since `contract.cpp:27` was written -- locomotion names its command
block `velocity_commands`, the older jumps `jump_command` / `jump_go`, and all
three feed one term. This crate never had the mapping, so it would have refused
every policy in circulation, including `rl-wbc-fsm`'s own
`locomotion.isaac_layout.json`. It went unnoticed because the tests build their
own contracts and named the term `commands` -- what the builder wanted. Two
implementations agreeing with themselves.

`py.rs` is the third, and the one that makes an FSM something you can look at before
you ship it: until it existed, the only way to find out what a `controller.toml` did
was to bundle it and upload it. `Fsm(config, contracts, robot, now_us=0,
trajectories=None, operators=False)` — `trajectories` is `{mode: text}` for the
recordings those contracts name, optional rather than a positional every caller
would pass `{}` for; with `operators`, keys (`set_key`) and whole pad frames
(`set_pad_frame`) are read through each mode's own controls, as on the robot, which
is what `play --app` runs. State crosses as plain sequences of `f32`; errors are
`ValueError` with the reason in them. Buttons are `set_key(key, down)` and
`set_pad(button, down)` — **state, both edges**, because `rise` and `fall` are both
in the vocabulary and a one-shot "was pressed" could only drive half of it — and
`pad_guide()` is what a host draws the pad from; `bindings` lists the mode switches.
The earlier `press_key` and `key_bindings` are gone for exactly that reason: a
one-shot press cannot produce a `fall`.

What `device.rs` does not include is a binary. That is `src/bin/controller.rs`
now: the 1 kHz loop, the two clocks it must not derive one from the other, the
`--dry-run` and `--check-reference` paths, and a shutdown that leaves through the
safe state by telling the FSM the bus has gone stale rather than by writing a
second shutdown behaviour. None of it can be tested off the board.

## Open

- **Provenance in the exported contract.** Needs a field per observation term in
  kk-rl-mjlab's exporter. Cheap to produce, and the whole value of the divergence
  list depends on it.
- **Per-mode command vocabulary**, above.
