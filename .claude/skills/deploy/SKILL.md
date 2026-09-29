---
name: deploy
description: Take a trained policy from a checkpoint to a running robot -- the hardware path specifically, not the browser or `play`, which need no board and no NPU. Export the contract, convert the ONNX to RKNN for the RK3576 NPU, assemble the upload bundle, cross-build the controller and check the board before touching it. Use this for anything on that path: "deploy this policy", "convert to rknn", "why won't the robot run my model", "the robot stands there doing nothing", a DDS topic that stays empty, a policy that behaves differently on hardware than in replay, or a first look at a new board. Reach for it before hand-rolling rknn-toolkit2 calls or scp'ing a model onto the robot, because every failure on this path is silent -- a wrong joint order, an unmatched QoS profile, a stale IDL and a gated-versus-free-running gait clock all produce a robot that runs and is wrong, never one that errors.
---

# Getting a policy onto the robot

The reference documents are [`deploy/README.md`](../../../deploy/README.md) for the map
and one per directory — [`convert/`](../../../deploy/convert/README.md),
[`dds/`](../../../deploy/dds/README.md), [`fsm/`](../../../deploy/fsm/README.md) — each
carrying the measurements and the reasoning for its own hop. **This skill does not restate
them.** It gives the order, says what has to be checked between the hops, and names the
failures that do not announce themselves.

## The one thing to understand first

Nothing on this path fails loudly. The characteristic failure is **a robot that runs and is
wrong**: an observation assembled at the right length from the wrong offsets, a policy fed a
gait clock it was not trained on, a DDS reader that never pairs so the controller sits
holding position and looks merely cautious.

So the rule is: **check each hop against the previous one, and prefer a failed export to a
shipped one.** `scripts/export.py` already refuses to write an export whose actor observes
something the robot cannot measure. Keep that spirit downstream.

## Procedure

Six hops. The middle one is the one people skip: an **export directory** is one
policy, and a **bundle** is what every host actually loads — the FSM config, one
policy per mode, and each host's build of the controller, the robot's included.

### 1. Export the contract

```bash
python scripts/export.py --task jumper.tripod --checkpoint logs/<model>/<task>/<run>/model_N.pt
```

Validations run first. Read the output — `deployable terms`, `every observed term is
measurable`, and the ONNX-vs-torch difference are the lines that matter. Produces
`actor.onnx` + `layout.json` + `README.md` under `tasks/<task>/out/<date-time>/`.
One directory per export, so re-exporting adds rather than replaces; which checkpoint it
came from is recorded in `layout.json`, and the bundler reads it back into `bundle.json` —
a manifest names the export directory itself, under `policy`.

`layout.json` carries `joint_limits`. An export predating that field is fine for a
simulator and is **refused** in any bundle that carries the board's runtime — see the
failures table.

A reference-guided task (`jumper.jump`, `jumper.dance`) writes one file more:
`<name>.trajectory.json`, the recording its contract names. The policy does not run
without it, so from here on it travels with it.

### 2. Convert for the NPU

Step 3 does this for every export in the manifest whose `actor.rknn` is missing or was
made from another `actor.onnx` (it reads the digest the converter writes into
`actor.rknn.json`). By hand, one export:

```bash
bash deploy/convert/docker-convert.sh --bundle tasks/jumper/tripod/out/<name>
```

Docker works on any host and is the **only** option off Linux — rknn-toolkit2 publishes no
macOS or Windows wheel. On Linux, `convert/setup.sh` builds a virtualenv instead.

The converter compares the result against onnxruntime and refuses to write a `.rknn` that
disagrees. fp16 lands around 1.1e-3; int8 was measured at 2.7e-1, which is 3.9° of joint
error, and the default tolerance is set between them so `--quantize` cannot be used by
accident. It writes `actor.rknn` beside the export, where the next step looks for it.

### 3. Assemble the bundle

```bash
python scripts/deploy.py                       # the only bundle in deploy/manifests.json
python scripts/deploy.py --manifest <name>     # one of several
```

Before it bundles, it makes the board's half itself: step 5's cross build, every time, and
step 2's conversion for each export that needs one -- both in docker. An app that still
lacks the board's controller, a `.rknn` or `reference.json` afterwards is refused and
removed. `--allow-incomplete` skips the two docker steps and builds with what is there,
saying in the notes what is missing: for a browser, a bench, or a machine with no docker,
never for the robot.

`deploy/manifests.json` is the build's input, maintained by hand: which **exports** go
in and which control switches into each. An export, never a checkpoint -- exporting is
where the contract is built and where the validations decide whether the policy can run
at all, so naming a checkpoint would make the bundler pick an export for you. Which
checkpoint an export came from is recorded inside it and lands in `bundle.json`.

Exports are timestamped and untracked, so a manifest names a directory that is this
machine\'s. On another one, export first and put the directory it prints under `policy`;
the bundler says exactly that when the path is not there. A control is a pad
button or a key plus a **gesture** -- `rise` `fall` `hold` `toggle`, or a click count `single`
`double` `triple` `quadruple` `quintuple` -- so "press A to crouch, release A to jump" is one
button bound twice. A mode's switches are one per device, the pad's and the keyboard's,
each saying which modes it may be pressed `from`. `python -m controller --vocabulary` prints the
words a manifest may use. That list is `controller/vocabulary.json`, the dictionary of
what a pad **and a keyboard** can say; what a word *means* is a manifest's business, or a
task's. Beside it a `.controller.toml` holds the cascade and the safety limits. **Neither
may say the other's half** — the manifest owns the modes and the bindings, the cascade
file owns everything else — because two files naming the same key is how the two come to
disagree, and nothing reads both. A mode whose export has not been run is not an error
about the manifest, so it prints the `export.py` command.

A reference-guided mode brings its recording with it: the bundler copies the trajectory
its contract names into the bundle beside that contract, and refuses a mode whose
recording is not in the export rather than shipping a policy that cannot run — the
controller looks for it by the name the contract gives. Such a mode also **ends**. When
the recording plays out the host releases that latch and the cascade falls through to its
last `always` rule, so the default mode is not something an operator has to steer back to
— `jumper.dance` is switched into and hands itself back.

`--mode name=<dir> --fsm <file>` still works for a one-off not worth writing down. A
single mode then gets a synthesised config; more than one needs `--fsm`, because a
priority order is a decision and there is no default for it that would not be somebody's
silent choice.

One FSM design, one bundle every host runs: `out/bundle_<timestamp>/<name>/` and
`<name>.app` beside it, the same directory zipped. `controller.toml`, the contracts and
`reference.json` are there once, and each host takes its own part:

| host | model | runtime |
|---|---|---|
| `board` | `models/*.rknn` | `runtime/board/controller`, this crate cross-compiled for aarch64 |
| `web` | `models/*.onnx` | `runtime/web/controller.wasm` + `controller.js` |
| `mjlab` | `models/*.onnx` | `runtime/mjlab/<platform>/controller.so` -- the build machine's, plus `win-amd64`'s `.pyd` and `macosx-universal2`'s, cross-built -- which `play --app` imports |

`bundle.json`'s `runtimes` says which host takes what; [`deploy/BUNDLE.md`](../../../deploy/BUNDLE.md)
is the format, file by file.

The order used to be the trap here: the `.rknn` and the cross-built `controller` had to
exist before bundling, or the bundle shipped without them with a note that read like a
step not yet taken. The bundler now runs both first, so there is no order to get wrong.

The bundler does not check the FSM itself — it asks the crate, the same code the browser
runs, opening the bundle as every host it carries a runtime for. A bundle that fails is
deleted rather than left for someone to find and upload; one that loads is then held to
`deploy/app.schema` before it is packed. Once the board's runtime is in, a single contract
without `joint_limits` refuses the whole bundle.

The bundle carries `reference.json`: recorded frames, so every host can be held against
one set of numbers instead of against each other. Step 6 replays it.

**After every build, give the bundle its Chinese manual.** Each bundle carries
`manual.en.json`, every key and pad button and what it does in each mode, generated from
the controller itself. Run the [`bundle-manual`](../bundle-manual/SKILL.md) skill: it
writes `manual.zh.json` and adds it with `deploy.py --translate`, which checks it against
the English and packs the `.app` again. Commit an `.app` (under
`out/bundle_<timestamp>/<name>.app`) only after that step — the translation changes the
archive.

### 3b. Where the pad's meanings come from -- two files, one sum

The controls a bundle ends up with come from **two** places, and the split is
deliberate:

| | file | says |
|---|---|---|
| between modes | `deploy/manifests.json` -> `[[fsm.button]]` | which control **switches** into a task -- the pad's and the keyboard's, each on its own -- with which gesture, and from which modes |
| inside a mode | `tasks/<task>/controls.yaml` | what the sticks and keys **do** while that task is driving |

Both pick from the same list, `controller/vocabulary.json`. Neither may say the
other's half: a manifest does not bind a stick to an axis, and a task does not
name a mode. **The sum of the two is the whole of how the deployed controller
consumes a gamepad** -- there is no third place, and the FSM never reads a
contract's gamepad block nor a contract the FSM's. A recorded motion's `go` is
not an exception: it is a `[[fsm.button]]` too, in the manifest's `buttons` list
beside `modes` -- where the bindings that are **not** mode switches go -- and the
task names it in its `reference` block, so the sum still holds.

**To write or change either file, use the `controls` skill.** It owns the
dictionary, both consumers, and the script that writes a task's half.

That last fact is why one control can quietly be two things. A manifest
switching modes on `B` and a task using `B` as its `release_button` are each
correct alone, and one press would do both -- and on the keyboard the same with
a key, or with a modifier a mode holds for a chord or for a control of its own.

**You do not have to check this by reading.** `Bundle::open` does it
(`operator::check_switches`, which every host runs again as it loads its
operators), so `scripts/deploy.py` does it on every build:

```
'slow' switches on the pad's B, which mode 'walk''s controls use too: one press
would do both. Pick another control, or say with `from` that the switch is not
pressed in 'walk'
```

Which means: if step 3 printed a bundle, the two halves do not collide. If it
refused with that message, there are two fixes and only two. Move the switch to a
control that mode does not use; or, if the switch is not meant to be pressed in
that mode at all, give it a `from` that lists only the modes where it should be
live, leaving that one out -- the `jumper` bundle's Space is the jump from
`locomotion` and the claw in the claw modes, because the jump's `keys` switch says
`"from": ["locomotion"]`. Do not widen anything, and do not take the control away
from the mode, to make it pass. The one sharing the check allows is a pad chord
that leaves the mode on the press (`menu` + a d-pad direction a claw mode holds its
arm on); the keyboard has no such exception.

To see the words either file may use:

```bash
python -m controller --vocabulary
```

### 4. Check the board *before* building anything for it

```bash
python3 .claude/skills/deploy/scripts/check_board.py --host <user>@<board> \
    --bundle tasks/jumper/tripod/out/<name>
```

`--host` is the board's ssh target and has no default; ask for it rather than guess.

Read-only: it runs no binary on the board and publishes nothing. Verifies glibc, the
librknnrt version, the CycloneDDS soname, the IDL types against the ones this repository
generates, and the board's joint names against the bundle's — by name, which is the check
that matters most.

Exit 0 all passed, 1 something differs, 2 unreachable.

> **It still takes an *export directory*,** not a bundle. The glibc, librknnrt, soname and
> IDL checks are the ones it is for. The joint-order check now needs `--config <file on the
> board>` and has no default: the path it used to assume belonged to a controller that is
> not part of the product, and a board out of the box has no such file.

### 5. Cross-build the controller

```bash
bash deploy/fsm/docker-build.sh          # aarch64 release
bash deploy/fsm/docker-build.sh test     # the host test suite
```

Cross-compiled rather than emulated, on `ubuntu:22.04` because the board's glibc is a floor.
This produces `controller`, the binary that runs the robot — the crate *is* the
controller, there is nothing separate to link it into.

It leaves it at **`out/deploy/controller-aarch64`**, not in `deploy/fsm/target/`: the
container builds into a named volume so nothing it owns can block a host-side `cargo
build`, which also means its target directory is invisible from here. Step 3 copies it
from there into the bundle, as `runtime/board/controller`.

Step 3 runs this every time, not only when there is none: it copies whatever is at that
path into the app under the current source's commit, and a binary says nothing about which
source it came from. With `--allow-incomplete` it is taken as found, and the build says
so — the runtime is either in the bundle or named in the notes, never neither.

### 6. Only then, the robot

Three commands, in this order, and the first two touch no bus:

Copy the bundle to the robot — the `<name>/` directory, or `<name>.app` unzipped there;
it holds everything, including the binary — and run from inside it. The binary opens the
bundle as the board and takes the `.rknn` models; the browser's and `play`'s parts ride
along unused.

```bash
./runtime/board/controller --bundle . --dry-run           # open it, report, publish nothing
./runtime/board/controller --bundle . --check-reference   # ...and replay the recorded frames
./runtime/board/controller --bundle . --machine /etc/mjrl/machine.toml
```

`--dry-run` prints where the numbers it would drive with came from — the stick scales and
their signs, the DDS domains, the QoS file — and which modes are on a stub.
`--check-reference` then replays `reference.json` and reports where this host differs from
the machine that built the bundle: `observation` and `targets` should be 0, and on the
board they are — `0.000e0` on both for the tripod/jump/dance bundle. `joint_torque`
differed by ~2 N·m until the servo turned out to report it and the controller stopped
re-deriving it from the commanded PD.

`--dry-run` and `--check-reference` touch no bus and are safe while another controller
runs. Driving is not.

## Failures worth recognising

| symptom | likely cause |
|---|---|
| topic stays empty, controller holds position | IDL or QoS mismatch — DDS never paired the endpoints, and that is not an error |
| binary will not start, `GLIBC_2.36 not found` | built on a newer base than the board's 22.04 |
| binary will not start, `libddsc.so.11` not found | CycloneDDS soname differs from the board's |
| `.rknn` will not load | board's librknnrt older than the toolkit that built it |
| robot steps in place while commanded to stand | gait clock free-running; the policy was trained with it gated off below `params.command_threshold` |
| a posture gait slightly off after the speed changes, and only then | the contract's `gait_phase` `advance` does not match the order the policy was trained in -- typically a checkpoint from before the field, re-exported with today's code, which writes `after_frame` for it. Export an old checkpoint from the commit it was trained at |
| wrong joints move, from the fifth onward | joint orders paired by index instead of by name |
| a joint drives into its hard stop and sits there | the target clamp is wider than the joint's travel. It comes from the contract's `joint_limits`; a board falling back to its own config had `[-3.30, 3.30]` for all 22, which never fired on 20 of them |
| full stick asks for more than the policy can do | stick scales hardcoded instead of read from `command_ranges`. The measured case was 0.80 m/s commanded against a 0.50 range |
| the bundler refuses the bundle: a contract carries no joint limits | the board's runtime is in the bundle (step 5 has run) and that mode's export predates `joint_limits`. The bundle is opened as every host it carries, so one such contract refuses all of it. Re-export the mode |
| the bundler refuses a reference-guided mode | its `<name>.trajectory.json` is not in the export directory. The contract names the recording, the bundle carries it beside the contract, and the policy does not run without it |
| a key switches nothing | the crate refuses a binding no rule reacts to, and a rule naming a button nothing latches. Both halves have to be there, and they live in two files on purpose. A key name the dictionary does not carry is refused there too -- `key` took any string at all until the keypad was written down |
| one press does two things | a manifest and a task picked the same button, key or modifier, in a mode where the switch may be pressed. Refused at step 3, by name -- each is correct alone, which is why only the sum can catch it. Another control, or the switch's `from` |
| a switch does nothing in one mode | its `from` leaves that mode out, on purpose or not: a switch is pressed only in the modes it lists. A latch it put on still switches off from inside its own mode |

The gait clock and the joint order are the two to internalise. On this robot the
observation covers 20 of 22 joints and the two it skips are at wire indices **4 and 9**,
in the middle — so an index pairing is not off by a constant, it is scrambled. And the
gait gate is invisible to every dimension check: gated or not, `gait_phase` is 2 wide and
the observation is the same length. Only the values differ, and only while standing.

## When something does not match

**Run `--check-reference` first.** It separates two failures that look identical from the
outside: `observation` or `targets` above ~1e-6 is this build of the crate behaving
differently, and a difference only in the action is the inference backend. A quantised
`.rknn` differs there by ~1e-3 and should differ nowhere else.

Do not adapt the controller to the robot's current state by hand. Both sides are generated
from things that are checked in — the policy's `layout.json`, the middleware's IDL — so a
mismatch means one of them is stale, and the fix is to find which. `deploy/dds/README.md`
records the mbus revision its copies came from, and the `git diff` that shows what has
changed since.
