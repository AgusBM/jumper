# Tutorial — three skills, one robot, all the way to the machine

One worked example, start to finish: train `jumper.tripod`, `jumper.jump` and
`jumper.dance`, put all three into **one bundle**, drive it in a browser, and run it on
the robot. Nothing here is invented for the tutorial — it is the bundle this repository
shipped when this page was written, and every command below was run to write it.

> **The shipped `jumper` bundle has changed its controls since** (2026-09-26 and
> 2026-09-29): `jumper.posture` walks in `locomotion`'s place; `A`, or Space, enters the
> jump and entering it is the whole of asking — there is no separate `go` —; four dances
> are Menu + the d-pad (Ctrl + 1 2 3 4) and four gestures the d-pad alone (1 2 3 4);
> `jumper.five_foot`'s claw is on `LB` (left, or `V`) and `RB` (right, or `B`); and a
> mode's switches are one per device, the pad's and the keyboard's.
> [`deploy/manifests.json`](../deploy/manifests.json) is the current entry and
> [`CONTROLS.md`](CONTROLS.md) §5.11 the layout. The path below is the same.

The other documents are references: [`AGENT_SETUP.md`](AGENT_SETUP.md) for bringing a
machine up, [`USAGE.md`](USAGE.md) for tasks and assets, [`../deploy/README.md`](../deploy/README.md)
for the deployment map, [`DESIGN.md`](DESIGN.md) for why any of it is shaped this way.
This one is the **order**, and it links rather than restates.

## What you will have at the end

```
out/bundle_<timestamp>/
├── jumper/       .rknn + runtime/board/controller (aarch64)       ← the robot
│                 .onnx + runtime/web/controller.wasm              ← a browser
│                 .onnx + runtime/mjlab/<platform>/controller.so   ← play --app
└── jumper.app    the same directory, zipped
```

**One** state machine, one bundle, three hosts, each taking its own part. It holds three
policies and the `controller.toml` that switches between them:

| mode | task | observation → action | rate | reached by |
|---|---|---|---|---|
| `locomotion` | `jumper.tripod` | 411 → 20 | 50 Hz | the cascade's `always` rule |
| `jump` | `jumper.jump` | 167 → 20 | 200 Hz | `LB`, then `A` to fire |
| `dance` | `jumper.dance` | 209 → 22 | 50 Hz | `RB` |

`jump` and `dance` are **reference-guided**: each carries a recorded motion that travels
with it. That is the part of this path with the most ways to fail quietly, so it is the
part this tutorial spends the most words on.

---

## 0. The rule that makes the rest make sense

**Nothing on this path fails loudly.** A wrong joint order, an unmatched QoS profile, a
gait clock that free-runs when it was trained gated, a stick scaled by the page instead
of by the contract — every one of them produces a robot that runs and is *wrong*, never
one that errors.

So each step below ends with **what to read in the output**. Skipping that is how a
bundle reaches a robot with a fault nobody can see.

---

## 1. Bring the machine up

Follow [`AGENT_SETUP.md`](AGENT_SETUP.md). The short version, from a fresh clone:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

`pip install -e .` is **mandatory, not a convenience**. `pyproject.toml` declares two
package roots, `rl/` has no `__init__.py`, and nothing under it imports without the
editable install — deliberately, so there is no `sys.path` manipulation anywhere.

Check it took:

```bash
python scripts/train.py --list
python -m pytest tests/ -q
```

`--list` must work on a machine with no simulation dependencies installed, and after it
runs there should not be one heavy import in `sys.modules`. If it prints the task table,
the registry is wired.

On macOS the live viewer needs `.venv/bin/mjpython` rather than `python`; without it a run
goes headless and says so.

---

## 2. Train the three tasks

Three separate runs. They share a robot and nothing else.

```bash
python scripts/train.py --task jumper.tripod --num_envs 4096
python scripts/train.py --task jumper.jump   --num_envs 4096
python scripts/train.py --task jumper.dance  --num_envs 4096
```

Logs land in `logs/<model>/<task>/<timestamp>/`, checkpoints as `model_<N>.pt`.
`--resume` continues from the newest one. `--no-tensorboard` turns the board off for a
single run without editing `.env`.

**Pick `--num_envs` for your backend, not by copying this page.** On CUDA with the warp
backend, more is better until memory says otherwise. On the native CPU backend every
environment holds its own `MjModel` and `MjData`, so memory grows with `num_envs` and a
number sized for a GPU will exhaust a workstation — start small and raise it. (The
thread count no longer follows it: it is capped at 8, see `--cpu_threads`.)

How long each of these takes is a property of your machine and your reward tuning, and
this document does not pretend to know it. What it can tell you is **what to watch**:

- `jumper.tripod` — a velocity-tracking task. The reward to watch is command tracking.
  It is also the *control* for the other gaits: `flat`, `tripod`, `ripple` and `tetrapod`
  differ in the gait and the speed ceiling it implies, and in **nothing else**
  (`tests/test_task_parity.py` fails on any other difference).
- `jumper.jump` — reference-guided. Its action is a **residual** on a recorded motion
  (`tasks/jumper/jump/ref/high_jump_flat.npz`), so the policy is learning a correction,
  not a trajectory. A jump that never leaves the ground usually means the reference is
  not being tracked, not that the reward is wrong.
- `jumper.dance` — scored against one recorded choreography. The choreography and its
  face animation are **committed** (`tasks/jumper/dance/media/demo.{npz,mp4}`), so it runs
  from a fresh clone. The music is not; training does not need it, and only exporting
  the performance video asks for it.
  The first environment build converts the clip and caches it in `media/.cache/`, keyed
  on the source's contents — replace the `.npz` and the next run reconverts, with no step
  to forget and no stale cache to train yesterday's dance from.
  See [`USAGE.md`](USAGE.md#a-task-that-needs-material-jumperdance).

Watch one of them run:

```bash
python scripts/play.py --task jumper.dance
```

With no `--checkpoint` it takes the newest one under `logs/<model>/<task>/`.

---

## 3. Export each policy

An **export directory is one policy**. This is the step where the contract is built and
where the validations decide whether the policy can run at all.

```bash
python scripts/export.py --task jumper.tripod --checkpoint logs/<model>/jumper.tripod/<run>/model_<N>.pt
python scripts/export.py --task jumper.jump   --checkpoint logs/<model>/jumper.jump/<run>/model_<N>.pt
python scripts/export.py --task jumper.dance  --checkpoint logs/<model>/jumper.dance/<run>/model_<N>.pt
```

Each writes `tasks/<task>/out/<date-time>/` containing `actor.onnx`, `layout.json`, a
`README.md` and a copy of the checkpoint. Re-exporting **adds** a directory rather than
replacing one; which checkpoint it came from is recorded inside, and the bundler reads it
back into `bundle.json` — a manifest names the export directory itself, under `policy`.

`jumper.jump` and `jumper.dance` each write one file more — `<name>.trajectory.json`, the
recording their contract names in a `reference` block (`jumper.dance` also renders its
`media/`, unless `--no-video`). From here on it travels with the
policy, and the bundler refuses a mode whose recording is missing rather than shipping a
policy that cannot run.

**Read these lines in the output:**

| line | what it means |
|---|---|
| `deployable terms` | every observed term is one the deployed controller can build |
| `every observed term is measurable` | nothing is observed that the robot cannot measure. An actor trained on `base_lin_vel` fails here — correctly, because no sensor on the robot reports it |
| the ONNX-vs-torch difference | the exported graph agrees with the network it came from |
| *(no line)* `joint_limits` | always written, and export itself refuses a gap or an inverted pair. An export from before 2026-09-20 lacks it, and the bundler refuses that once the board's controller is in it (step 8) |

A failed export is a better outcome than a shipped one. That is the spirit of the whole
path.

---

## 4. Say what the controls mean

A deployed controller's gamepad behaviour comes from **two files, and their sum is the
whole of it**:

| | file | says |
|---|---|---|
| between modes | `deploy/manifests.json` → `[[fsm.button]]` | which control **switches into** a mode, on which device, with which gesture and from which modes |
| inside a mode | `tasks/<task>/controls.yaml` | what the sticks and keys **do** while that mode drives |

Neither may say the other's half: a manifest does not bind a stick to an axis, and a task
does not name a mode. There is no third place.

Both pick from one closed list:

```bash
python -m controller --vocabulary
```

Ten buttons, four dpad directions, six axes, the keys and their three modifiers, nine
gestures. A **control** is a pad button or a key, plus a gesture — `rise` (the tick it goes
down), `fall` (the tick it comes up), `hold` (every tick it is down), `toggle` (down until
pressed again), or a click count from `single` to `quintuple`. So "press A to arm, release
A to go" is one button bound twice, not one binding with two jobs. The pad and the keyboard
are two paths: a key is bound to what it does, never to a pad button.

**Use the `controls` skill to write either file.** It owns the dictionary and refuses a
word that is not in it. Do not retype the list.

For this bundle, a manifest entry in `deploy/manifests.json` reads:

```json
"jumper": {
  "fsm": "jumper.controller.toml",
  "modes": {
    "locomotion": { "task": "jumper.tripod", "policy": "tasks/jumper/tripod/out/<dir>" },
    "jump":       { "task": "jumper.jump",   "policy": "tasks/jumper/jump/out/<dir>",
                    "pad": { "button": "LB", "on": "toggle" } },
    "dance":      { "task": "jumper.dance",  "policy": "tasks/jumper/dance/out/<dir>",
                    "pad": { "button": "RB", "on": "toggle" } }
  },
  "buttons": [ { "name": "jump_go", "pad": { "button": "A", "on": "fall" } } ]
}
```

It is written here as the manifest has it since 2026-09-29: a mode's switches are one per
device, `pad` and `keys`, each with its own gesture and the modes it may be pressed
`from`. When this page was written a mode carried `button`, `key`, `on` and `with`
itself; what it built is the same.

The repository's own `jumper` entry was this one with `jumper.posture` in the
`locomotion` slot: it walks the same tripod gait and takes a posture command on
top -- W A S D and J L drive both tasks, and the posture adds I K, U O, N M and
H ;. Swapping one task for another in a slot is a two-line edit.
It has since moved the jump and the dance, added the claw, three more dances and four
gestures (the note at the top), and its cascade names every one of those modes, so a
three-mode bundle like this one needs a cascade of its own: `button:jump`,
`button:dance`, and the `always`.

Three things worth understanding here:

**`locomotion` has no switch.** With one mode, the cascade's final `always` rule is how
you reach it. A second mode needs a binding or it is unreachable.

**The paths are this machine's.** Exports are timestamped and git-ignored (the few the
shipped manifest names were added to git by hand), so a manifest names a directory that
exists here. On another machine, export first and put the directory
it prints under `policy`. The bundler says exactly that when the path is not there.

**`jump_go` is in `buttons`, not in `modes`.** A recorded motion keyed to a moment needs an
*event*, not a mode switch — `jumper.jump`'s push-off is 40 ms wide. It is an ordinary
`[[fsm.button]]`; the task's `reference` block carries `go_event: "jump_go"` and the
controller starts the motion's clock on that name. Without a `go_event` the motion runs
on a **timer** from mode entry, which is enough to watch it on a bench and not something a
person can use. (`jumper.jump` has since dropped its `go_event` for `starts_on_entry` —
its policy trained starting past its go, never waiting in the go frame — so its export
no longer reads `jump_go`, and a manifest still declaring it is refused. The mechanism is
unchanged for a motion that does want a person to pick the moment.)

Beside the manifest, `deploy/jumper.controller.toml` holds the cascade and the safety
limits — the rule order, the tilt limit, the timeouts. The bundler composes the two.

**You do not have to check the two halves by reading.** `Bundle::open` refuses a
collision, so `scripts/deploy.py` refuses it on every build:

```
'slow' switches on the pad's B, which mode 'walk''s controls use too: one press
would do both. Pick another control, or say with `from` that the switch is not
pressed in 'walk'
```

If the build printed a bundle, the two halves do not collide.

---

## 5. Assemble the bundle

```bash
python scripts/deploy.py
```

With no arguments it builds the one bundle `deploy/manifests.json` defines
(`--manifest jumper` names it when there are several), and before it bundles
anything it makes the board's half: the controller cross-built in docker (step 8
says what that is) and every policy without a current `.rknn` converted in
docker (step 7). The first run builds both images and takes a while; after that
the cross build is incremental and a policy is converted once.

```
[deploy] manifest jumper: dance <- jumper.dance, jump <- jumper.jump, locomotion <- jumper.tripod
[deploy] cross-building the board's controller (deploy/fsm/docker-build.sh)
[deploy] cross-building play's controller for Windows (deploy/fsm/docker-build.sh)
[deploy] cross-building play's controller for macOS (deploy/fsm/docker-build.sh)
[deploy] converting dance for the NPU (tasks/jumper/dance/out/<dir>)
[deploy] converting jump for the NPU (tasks/jumper/jump/out/<dir>)
[deploy] converting locomotion for the NPU (tasks/jumper/tripod/out/<dir>)
[deploy] bundled <commit> -> out/bundle_<timestamp>/jumper
           <size> MB  <n> files  -> jumper.app <size> MB
           board  rknn models, runtime/board/controller
           web    onnx models, runtime/web/controller.wasm
           mjlab  onnx models, runtime/mjlab/linux-x86_64/controller.so, runtime/mjlab/win-amd64/controller.pyd, runtime/mjlab/macosx-universal2/controller.so
             opens as web, mjlab, board
             dance            obs 209 -> act 22
             jump             obs 167 -> act 20
             locomotion       obs 411 -> act 20
             jump_go        A on fall
             dance          RB on toggle
             jump           LB on toggle
             22 joints on the wire, 3 mode(s), 6 rule(s)
             note: dance: its contract has no `controller` block, …
             note: jump: its contract has no `controller` block, …
             note: reference: 24 frames, …
[deploy] next: the Chinese manual. The bundle-manual skill writes it and adds it with
           python scripts/deploy.py --translate out/bundle_<timestamp>/jumper --language zh --manual <file>
```

The bundle is written as a **directory and a `.app` beside it** — a zip under another
name, which any zip reader opens as it is. The directory is what this machine reads; the
archive is what a person carries to a browser's file picker, an upload form or the robot.
The archive's contents sit at its root, because that is where a consumer looks for
`bundle.json`. The whole of today's `jumper` bundle — five modes, converted and
cross-built — is one 10.9 MB `.app`.

An app that still lacks the board's controller, a `.rknn` or the reference vectors
after all that is refused, not written. On a machine with no docker, or for a bundle only
a browser or `play` will load, `--allow-incomplete` skips the two docker steps, takes what
is there, and says in its notes what is missing.

**Read the notes.** They are what the build noticed and did not refuse:

- `its contract has no controller block, so a consumer falls back to its own key
  bindings` — that mode carries no `controls.yaml`. Harmless for a mode nobody steers
  (`dance` and `jump` are not steered); re-export if it should be.
- `reference: N frames, M of them inferences` — the recorded comparison vectors were
  written. Step 10 replays them on the board.

Every host reads the one `controller.toml`, the one set of contracts and the one
`reference.json` in the bundle, so there is nothing for two hosts to disagree about.

The bundle also carries `manual.en.json`: every key and pad button, and what it does in
each mode, written from the controller's own account of its pad and its keys. Give it its
Chinese translation now — the [`bundle-manual`](../.claude/skills/bundle-manual/SKILL.md) skill
writes `manual.zh.json` and adds it with
`python scripts/deploy.py --translate out/bundle_<timestamp>/jumper --language zh --manual <file>`,
which checks it against the English and packs the `.app` again.

For a one-off not worth writing into a manifest, `--mode name=<dir> --fsm <file>` still
works. More than one mode needs `--fsm`, because a priority order is a decision and any
default for it would be somebody's silent choice.

---

## 6. Drive it — the two hosts that need no hardware

### In `play`

```bash
python scripts/play.py --app out/bundle_<timestamp>/jumper.app
```

What runs is then the **app's** controller — the extension it carries for this
machine's platform — not a task's own policy: the same code the robot runs, inside
mjlab's physics, with every mode in it. The app's own buttons and keys switch modes (in
the viewer, each key down and up, as `mjrl.viewer.keys` reports them), each mode
reads the keys and a gamepad through its own controls, and the controller's joint targets
and gains are what the servos track — so a mode switch ramps as it would on the robot and
a task's deploy hook moves the joints it moves there.

### In a browser

Upload `jumper.app` to a simulator that accepts a training-platform bundle. It carries its
own `runtime/web/controller.wasm`, so what runs is the controller this build produced.

Then: the robot stands on `locomotion`. **`RB`** switches into `dance`. **`LB`** switches
into `jump`, and **pressing and releasing `A`** fires the jump — the `go` is bound to
`fall`, the release, not the press. Both recorded motions release themselves when the
recording ends, and the cascade's `always` rule catches the robot, so neither is something
an operator has to steer back out of. (In the shipped bundle today: `A` or Space for the
jump, Menu + the d-pad or Ctrl + 1 2 3 4 for the four dances, the d-pad alone or 1 2 3 4
for the four gestures, `LB` / `RB` or `V` / `B` for the claw, and Menu or Ctrl, let go on
its own, to leave a dance or a gesture early.)

Two things the browser cannot do, worth knowing before you conclude something is broken:

- **A mixed-rate bundle needs a host that can keep up.** `jumper.jump` is a 200 Hz
  contract; the other two are 50 Hz. A host that ticks the controller slower than the
  fastest mode holds each action too long *and* advances the recorded motion's clock too
  slowly — the jump plays in slow motion. It is not an error and nothing reports it, so
  check the host's control rate before blaming the policy.
- **It never runs the `.rknn`.** A browser takes each mode's ONNX by design; what the NPU
  does to the policy is measured on the board, in step 10.

---

## 7. Convert for the NPU

Only for the board, and step 5 has done it: it converts every export in the manifest
whose `actor.rknn` is missing or was made from another `actor.onnx`. By hand, one export
at a time:

```bash
bash deploy/convert/docker-convert.sh --bundle tasks/jumper/tripod/out/<dir>
bash deploy/convert/docker-convert.sh --bundle tasks/jumper/jump/out/<dir>
bash deploy/convert/docker-convert.sh --bundle tasks/jumper/dance/out/<dir>
```

Docker works on any host and is the **only** option off Linux — rknn-toolkit2 publishes no
macOS or Windows wheel. On Linux, `deploy/convert/setup.sh` builds a virtualenv instead.

The converter compares the result against onnxruntime and **refuses to write a `.rknn`
that disagrees**. fp16 lands around 1.1e-3; int8 was measured at 2.7e-1, which is 3.9° of
joint error, and the default tolerance sits between them so `--quantize` cannot be used by
accident. It writes `actor.rknn` beside the export, where the bundler looks for it, and
`actor.rknn.json` with the digest of the ONNX it came from, which is how the bundler knows
the two still belong together.

---

## 8. Cross-build the controller

Step 5 runs this every time before it bundles. By hand:

```bash
bash deploy/fsm/docker-build.sh          # aarch64 release -> out/deploy/controller-aarch64
bash deploy/fsm/docker-build.sh test     # the host test suite
```

Cross-compiled rather than emulated, on `ubuntu:22.04` because the board's glibc is a
floor. The crate **is** the controller — there is no separate program linking it.

Every time, not only when there is none: the bundler copies whatever is at
`out/deploy/controller-aarch64` into the app and files it under the source's commit, and
nothing about a binary says which source it came from. A build left from last week would
ship as this week's. With the source unchanged it is an incremental cargo build in the
container's own target volume.

---

## 9. Check the board before touching it

```bash
python3 .claude/skills/deploy/scripts/check_board.py --host <user>@<board> \
    --bundle tasks/jumper/tripod/out/<dir>
```

`--host` is the board's ssh target; it has no default.
Read-only: runs no binary on the board and publishes nothing. It verifies glibc, the
librknnrt version, the CycloneDDS soname, and the IDL types against the ones this
repository generates. Exit 0 all passed, 1 something differs, 2 unreachable.

It will also compare the board's joint names against the bundle's — **by name**, which is
the comparison that matters most — but only when told where to read them from:
`--config <file on the board>`. There is no default, because a board has no such file
unless somebody put one there.

> Note it takes an *export directory*, not a bundle.

---

## 10. The robot

Copy the bundle to the robot — `jumper/`, or `jumper.app` unzipped there; it holds
everything, including the binary. Then, from inside it, three commands **in this order**.
The first two touch no bus.

```bash
./runtime/board/controller --bundle . --dry-run
```

Prints where the numbers it would drive with came from: the stick scales and their signs,
the DDS domains, the QoS file, and which modes are on a stub. Publishes nothing.

```bash
./runtime/board/controller --bundle . --check-reference
```

Replays `reference.json` — the frames recorded when the bundle was built — and reports
where this host differs from the machine that built it. `observation` and `target` should
be `0`, and on the board they are: `0.000e0` on both for this tripod/jump/dance bundle.

This is the measurement the whole path exists for. It once showed `joint_torque` differing
by about 2 N·m, which turned out to be the controller re-deriving torque from the
commanded PD instead of reading what the servo reports. Nothing else would have found it.

```bash
./runtime/board/controller --bundle . --machine /etc/mjrl/machine.toml
```

This one drives. Everything above was rehearsal.

---

## When something is wrong

The full table is in [`.claude/skills/deploy/SKILL.md`](../.claude/skills/deploy/SKILL.md).
The ones that catch people on this particular bundle:

| symptom | likely cause |
|---|---|
| the mode is entered and the robot just stands there | a recorded motion whose `go` never fired. The mode is entered, the policy infers every tick, and the panel agrees — the motion's own clock is the only thing that distinguishes the two |
| the recorded motion plays in slow motion | the host ticks the controller slower than the mode's `control_hz`. The motion's clock counts control steps |
| the robot steps in place while commanded to stand | gait clock free-running; the policy was trained with it gated off below `params.command_threshold` |
| wrong joints move, from the fifth onward | joint orders paired by index instead of by name |
| a stick does nothing | a name that is not in the dictionary. It is a closed list, and a typo that parses is a binding that drives nothing |
| the robot walks backwards and everything else looks right | a sign. Nothing downstream catches it; a hand on the pad does |
| one press does two things | the same control in both files. The bundler names both — change one, do not widen anything to make it pass |
| the bundler refuses a reference-guided mode | its `<name>.trajectory.json` is not in the export directory |
| the bundler refuses the bundle once the board's controller is in it | a contract has no `joint_limits`. Re-export |
| a topic stays empty and the controller holds position | IDL or QoS mismatch. DDS never paired the endpoints, and that is not an error |
