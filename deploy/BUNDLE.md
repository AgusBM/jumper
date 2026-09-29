# The bundle — what a host loads, and what loading one requires

A **bundle** is one directory every host loads: the state machine, one contract per mode,
each policy in every model format a host runs, and each host's build of the controller.
`scripts/deploy.py` writes it and `deploy/fsm` reads it, on every host, each
taking its own part. This document is the format as those two implement it at this
commit, and what anything that consumes a bundle has to do — this repository's three
hosts, or a fourth written somewhere else.

Where this document and the code disagree, the code is what runs. The readers:

| reads | code |
|---|---|
| a bundle directory, as the host opening it — the board, and the bundler's own check | `fsm/src/bundle.rs`, `Bundle::open(dir, Target)` |
| `controller.toml` | `fsm/src/config.rs`, `FsmConfig::parse` |
| a contract | `fsm/src/layout.rs`, `Contract::from_str` |
| a contract's `controller` block | `fsm/src/operator.rs`, `OperatorSpec::check` |
| a `*.trajectory.json` | `fsm/src/trajectory.rs`, `Trajectory::parse` |
| `reference.json` | `fsm/src/reference.rs`, `Reference::parse` |
| all of the above handed over as text — a browser, `play` | `fsm/src/web.rs`, `fsm/src/py.rs` |
| the whole bundle — `bundle.json`, the manuals, every file's name — against [`app.schema`](app.schema) | `scripts/deploy.py`, `validate_app` |

For a browser page, the bundle's own README — [`fsm/BUNDLE_README.md`](fsm/BUNDLE_README.md)
filled in — is the integration guide for the API. This document is the layer under it.

## Words

- **Export directory** — one policy, as `scripts/export.py` writes it. The bundler reads
  `layout.json`, `actor.onnx`, `actor.rknn` and the `actor.rknn.json` recording which ONNX
  it was converted from (a default build converts the export itself when either is missing
  or stale), and the `*.trajectory.json` a contract names — nothing else. Several tools
  still call an export directory a bundle: `onnx2rknn.py --bundle`, `docker-convert.sh
  --bundle`, `check_board.py --bundle` and the export's own README title. None of them takes what this document
  describes.
- **Build** — one run of `scripts/deploy.py`: a directory holding the bundle and
  its `.app` beside it. The build directory has no `bundle.json` and is **not** a bundle;
  `play --app` refuses it and names the app inside.
- **Bundle** — the directory a build writes, `<name>/`, or its `.app`.
- **Host** — what opens a bundle: `board` (the robot's `controller`), `web` (a browser),
  `mjlab` (`play --app`). A host says which it is and takes its own model format and its
  own build of the controller; the bundle does not say whom it was packed for.
- **Mode** — an FSM state that runs a policy. Its name keys every per-mode file.
- **Wire order** — the robot's joint order, `wire_joint_order` in the contracts. Every
  per-joint array that passes between a host and the controller is in it.

## One build, one bundle

```
out/bundle_<YYYY-mm-dd_HH-MM-SS>/   no --bundle, or a bare one; otherwise the directory given
├── <name>/                         the bundle: every host's part, one copy of the rest
└── <name>.app                      the same directory, zipped
```

`<name>` is the manifest's — `jumper` — or `bundle` for a `--mode` build. Every host reads
the same `controller.toml`, the same contracts and the same `reference.json` because there
is one copy of each. It was three directories, `board/`, `web/` and `mjlab/`, that held
those byte for byte and were hashed against each other to prove it; one directory cannot
disagree with itself. What differs by host is which model format it loads and which build
of the controller it is, and `bundle.json`'s `runtimes` says both:

| host | models | controller |
|---|---|---|
| `board` | `models/*.rknn` | `runtime/board/controller`, this crate cross-compiled for aarch64 |
| `web` | `models/*.onnx` | `runtime/web/controller.wasm` and `controller.js` |
| `mjlab` | `models/*.onnx` | `runtime/mjlab/<platform>/controller.so` (`.pyd` on Windows), which `play --app` imports |

A build is complete or it is not written. By default it cross-builds the board's
controller and converts every missing or stale `.rknn`, both in docker, and refuses — and
removes — an app that still lacks the board runtime, an `.rknn` or `reference.json`.
`--allow-incomplete` skips both steps and writes whatever it has, saying what is missing in
`notes`: for a browser, a bench or the tests.

The input is a manifest — `python scripts/deploy.py [--manifest <name>]` reads
[`manifests.json`](manifests.json) (`--manifest` only when it defines more than one), which names each mode's task, its export directory
and the control that switches into it, plus a `.controller.toml` holding the cascade — or
`--mode NAME=DIR` per mode with `--fsm FILE`, where a single mode may leave out `--fsm`
and gets a synthesised config. The `deploy` skill covers producing one; this document
starts from the directory. An existing bundle directory of the same name is deleted and
rewritten, and one that fails the bundler's check is removed. The bundle is opened as
every host `runtimes` declares, so a bundle that carries the board's runtime and a
contract without `joint_limits` is refused as a whole.

Measured on a Linux x86_64 machine on 2026-09-27, for the `jumper` manifest (five modes
on four exports — `jumper.posture`, `jumper.jump`, `jumper.dance`, and `jumper.five_foot`
as `claw_left` and `claw_right` — every export converted, `controller` cross-built):
`bundle.json` and 23 files, 18.2 MB, packed into a 10.9 MB `jumper.app`. The three
bundles this replaced came to 22.8 MB of `.app` for the same manifest. `jumper.dance`'s
recording is 4.8 MB of it.

## What a bundle holds

The files every host reads sit at the top; the models are under `models/` and each host's
controller under `runtime/<host>/`. [`app.schema`](app.schema)'s `x-files` is the same list
as path patterns, and every build is held to it.

| file | taken by | what it is |
|---|---|---|
| `bundle.json` ● | every host | the manifest: what is here, which host takes what, where it came from, its digests |
| `README.md` ● | a person | `fsm/BUNDLE_README.md` filled in: each host's command line, and the browser's API |
| `controller.toml` ● | every host | the state machine |
| `<mode>.json` ● | every host | the mode's contract: the export's `layout.json`, unchanged |
| `<name>.trajectory.json` ○ | every host | a reference-guided mode's recording, under the name its contract gives |
| `reference.json` ○ | every host | recorded frames for the cross-host check. Absent when the build machine could not generate it — and a note says so. Only an `--allow-incomplete` build lacks it; a default build makes it or refuses the app |
| `manual.en.json` ● | a person, a page | every key and button the bundle reads and what each does, generated at build — below |
| `manual.<language>.json` ○ | a person, a page | a translation of it, added with `deploy.py --translate` |
| `models/<mode>.onnx` ● | `web`, `mjlab` | the policy, one per export. The board's only for a mode not yet converted — and a note says so, in an `--allow-incomplete` build |
| `models/<mode>.rknn` ○ | `board` | the same policy, converted for the RK3576 NPU |
| `runtime/web/controller.wasm` `controller.js` ● | `web` | the controller for a browser, and its wasm-bindgen glue |
| `runtime/mjlab/<platform>/controller.so` ○ | `mjlab` | the controller as a CPython extension, `abi3`, Python 3.10 on, for the build machine's `sysconfig.get_platform()` |
| `runtime/mjlab/win-amd64/controller.pyd` ○ | `mjlab` | the same for Windows 10 on, cross-built with MinGW in the board's container; it imports `python3.dll` and Windows' own DLLs and nothing else. Absent when nothing has cross-built it — and a note says so. Only an `--allow-incomplete` build lacks it; a default build makes it or refuses the app |
| `runtime/mjlab/macosx-universal2/controller.so` ○ | `mjlab` | the same for macOS 11.7.1 on, arm64 and x86_64 in one file, cross-built with zig in the board's container; it links only `/usr/lib`'s `libSystem`, `libiconv` and `libcharset`, and its arm64 slice is signed ad hoc. Absent when nothing has cross-built it — and a note says so. Only an `--allow-incomplete` build lacks it; a default build makes it or refuses the app |
| `runtime/board/controller` ○ | `board` | the robot's whole program, aarch64, mode 0755. Absent when nothing has cross-built it — and a note says so. Only an `--allow-incomplete` build lacks it; a default build makes it or refuses the app |

● always, ○ when it applies. The bundler always writes the extension for the machine it
runs on — it generates the manual and the reference by running it — but the schema allows
a bundle without one.

Four naming rules, one of which catches everybody once:

- **A contract is `<mode>.json`**, whatever the export called it.
- **A model is stored once per export, named after the first mode by name that uses it.**
  `claw_left` and `claw_right` are one export carried on two sides, and both name
  `models/claw_left.onnx` and `models/claw_left.rknn`.
- **The model file is named in `bundle.json`, not in `controller.toml`.** A state names
  its model (`model = "<mode>.onnx"` in a composed config), and that name is no file in
  the bundle. Every host here opens `modes.<mode>.models.<its format>` from
  `bundle.json`; the state's `model` is only read for being non-empty, which is what makes
  the state run a policy.
- **A trajectory keeps its own name** — `high_jump_flat.trajectory.json`, not
  `jump.trajectory.json` — and sits beside the contract, which is where the reader looks.
  Two modes shipping different recordings under one name are refused at build, because
  one would overwrite the other.

## `bundle.json` — `kk-policy-bundle/1`

UTF-8 JSON, two-space indent. The one from the build above, with `notes`, `files`,
`built_from`, the runtimes' `note` and the digests shortened, and `export` as it reads for
an export inside the repository:

```json
{
  "schema": "kk-policy-bundle/1",
  "runtimes": {
    "web": {
      "model": "onnx",
      "glue": "runtime/web/controller.js",
      "wasm": "runtime/web/controller.wasm",
      "tool": "scripts/deploy.py",
      "cargo": "build --release --target wasm32-unknown-unknown --no-default-features --features web",
      "wasmBindgen": "wasm-bindgen 0.2.128",
      "commit": "0d5edc112d5f3110b22f0a73078cadb8976ff1e6",
      "unverifiedByTheConsumer": "A consumer's own controller is held against its own implementation by a parity test. This one is not held against anything: ..."
    },
    "mjlab": {
      "model": "onnx",
      "extensions": {"linux-x86_64": {"file": "runtime/mjlab/linux-x86_64/controller.so",
                                      "abi": "abi3-py310"},
                     "win-amd64": {"file": "runtime/mjlab/win-amd64/controller.pyd",
                                   "abi": "abi3-py310"},
                     "macosx-universal2": {"file": "runtime/mjlab/macosx-universal2/controller.so",
                                           "abi": "abi3-py310"}},
      "commit": "0d5edc112d5f3110b22f0a73078cadb8976ff1e6",
      "note": "..."
    },
    "board": {
      "model": "rknn",
      "file": "runtime/board/controller",
      "platform": "aarch64-unknown-linux-gnu",
      "commit": "0d5edc112d5f3110b22f0a73078cadb8976ff1e6",
      "note": "..."
    }
  },
  "fsm": "controller.toml",
  "modes": {
    "claw_left": {"models": {"onnx": "models/claw_left.onnx", "rknn": "models/claw_left.rknn"},
                  "contract": "claw_left.json", "observation": 407, "action": 16},
    "claw_right": {"models": {"onnx": "models/claw_left.onnx", "rknn": "models/claw_left.rknn"},
                   "contract": "claw_right.json", "observation": 407, "action": 16},
    "dance": {"models": {"onnx": "models/dance.onnx", "rknn": "models/dance.rknn"},
              "contract": "dance.json", "observation": 209, "action": 22,
              "trajectory": "demo.motion.trajectory.json"},
    "jump": {"models": {"onnx": "models/jump.onnx", "rknn": "models/jump.rknn"},
             "contract": "jump.json", "observation": 167, "action": 20,
             "trajectory": "high_jump_flat.trajectory.json"},
    "locomotion": {"models": {"onnx": "models/locomotion.onnx", "rknn": "models/locomotion.rknn"},
                   "contract": "locomotion.json", "observation": 415, "action": 20}
  },
  "notes": ["reference: 24 frames, 18 of them inferences (6 in warm start or ramp)"],
  "files": {"README.md": {"bytes": 19726, "sha256": "68ea0309…"},
            "runtime/board/controller": {"bytes": 1517976, "sha256": "42d1bec5…"}},
  "reference": "reference.json",
  "manuals": {"en": "manual.en.json", "zh": "manual.zh.json"},
  "built_from": {
    "claw_right": {"task": "jumper.five_foot",
                   "from": "tasks/jumper/five_foot/out/2026-09-26_18-02-25/model_25400.pt",
                   "export": "tasks/jumper/five_foot/out/2026-09-27_15-24-01",
                   "hook": {"side": "right"},
                   "pad": {"button": "RB", "on": "toggle", "from": ["locomotion", "claw_left"]},
                   "keys": {"key": ["key_b"], "on": "toggle", "from": ["locomotion", "claw_left"]}}
  }
}
```

| field | type | |
|---|---|---|
| `schema` | string | `"kk-policy-bundle/1"`. A reader that does not know the string refuses the bundle rather than guess which fields it is missing; `Bundle::open` does |
| `runtimes` | {host: object} | each host's build of the controller, keyed `web`, `mjlab`, `board` — below. A host whose build this bundle does not carry has no entry, and a note says how to add it. `web` is always there |
| `fsm` | string | always `"controller.toml"`. No host reads it; each opens `controller.toml` by name |
| `modes` | object | one entry per mode, keyed by state name, sorted: `models`, the policy's file per model format — `onnx` always, `rknn` once the export was converted; `contract`; `observation` and `action` (the widths, copied from the contract for a reader); and `trajectory` when the contract names a recording |
| `notes` | [string] | what the bundler knows is missing or unusual. **A host shows them to a person**: they are the only record of an unconverted model, a missing board runtime or a reference that could not be generated — each of which only an `--allow-incomplete` build can have |
| `files` | {path: {`bytes`, `sha256`}} | every file in the bundle except `bundle.json` itself, by its path inside it, `README.md` and the manuals included. SHA-256 in lowercase hex |
| `reference` | string \| null | `"reference.json"`, or `null` when there is none |
| `manuals` | {language: file} | `en`, `"manual.en.json"`, always; a translation adds its own — `zh`, `"manual.zh.json"` |
| `built_from` | object | manifest builds only. Per mode: `task`; `from`, the checkpoint the export recorded (`_checkpoint.path`) or `"(the export records no checkpoint)"`; `export`, repository-relative when it is inside it; `hook` when the mode carries one; and the mode's `pad` and `keys` switches, as the manifest writes them, where it has them |

`runtimes`, per host:

| host | entry |
|---|---|
| `web` | `model` (`"onnx"`), `glue` and `wasm` (paths in the bundle), `tool`, `cargo` (the build command), `wasmBindgen` (the CLI that wrote the glue), `commit`, and `unverifiedByTheConsumer` — a sentence a consumer is asked to show |
| `mjlab` | `model` (`"onnx"`); `extensions`, `{platform: {file, abi}}`, keyed by `sysconfig.get_platform()` — the building machine's (`linux-x86_64`), and `win-amd64` and `macosx-universal2`, cross-built; macOS without the version a Mac's platform string carries — `abi` being `abi3-py310`; `commit`, `note` |
| `board` | `model` (`"rknn"`), `file` (`runtime/board/controller`), `platform` (`aarch64-unknown-linux-gnu`), `commit`, `note` — **absent when nothing has cross-built the binary**, and a note says so; only in an `--allow-incomplete` build |

`commit` is the last commit touching the controller's source — every file its build
reads: the crate's `Cargo.toml`, `Cargo.lock`, `build.rs`, `src/` and `vendor/`, every
task's `deploy/` hook, `controller/vocabulary.json` and `dds/idl/`
(`scripts/deploy.py::SOURCES`, held to what rustc reads by
`tests/test_deploy_sources.py`) — not the repository's `HEAD`, which moves for reasons
that change no byte of it, with `-dirty` appended when those paths had uncommitted
changes. Until 2026-09-28 it was the crate's three alone, and a hook committed on its
own shipped under the commit before it.
`--require-clean` refuses to build from such a tree instead. Every runtime of one bundle
is built from one commit.

## `controller.toml` — the state machine

One `[fsm]` table and three arrays of tables. `FsmConfig::parse` is the authority;
everything below is what it accepts.

**`[fsm]`** — every key but the last two is required and none has a default: a defaulted
gain is a robot that moves, and a defaulted timeout is a fallback that never fires.

| key | unit | |
|---|---|---|
| `initial_state` | state | entered at start, and by `@initial` |
| `warm_start_ref` | state | where a warm start hands off, and `@warm_start_ref` |
| `safe_state` | state | where the safety rules send the robot. May not set `skip_tilt_check` |
| `tilt_limit` | rad | tilt beyond which `tilted` holds |
| `state_timeout_ms` | ms | robot state older than this makes `feedback_stale` hold |
| `command_timeout_ms` | ms | operator input older than this makes `command_stale` hold, releases every latched button and returns the command to rest |
| `mode_switch_ramp_s` | s | how fast the setpoint slides to a mode's home pose; 0 means no ramp. **Not** how long a switch takes: the policy starts when the *measured* pose arrives, and waits for it indefinitely |
| `pose_reach_tol` | rad | how close every joint has to be to that pose |
| `warm_start_duration_s` | s | a warm start's slide |
| `ramp_kp`, `ramp_kd` | | the gains during the mode-switch ramp |
| `click_window_ms` | ms | how long after one press of a control the next still counts toward the same click gesture. **Required once any `[[fsm.button]]` counts clicks** |
| `exclusive` | bool | one mode latched at a time: a latch coming on releases every other, and the last switch pressed is the mode. `false` when absent — latches are a set, and the cascade's order settles two that are on together. The `jumper` cascade sets it (2026-09-29) |

**`[[fsm.state]]`** — a unique `name`, and:

| key | default | |
|---|---|---|
| `model` | `""` | non-empty makes the state a **mode**: it runs a policy, and needs a contract and a `bundle.json` entry of the same name. The value is not used to find the file |
| `hold_current` | false | damping only, around whatever pose the robot is in |
| `warm_start` | false | slide to `warm_start_ref`'s home pose at this state's `kp` / `kd` |
| `skip_tilt_check` | false | suspend `tilted` while this state is active. Refused on `safe_state` |
| `kp`, `kd` | 0 | the gains of a state that runs no policy |
| `kp_scale`, `kd_scale`, `kp_offset`, `kd_offset` | 1, 1, 0, 0 | a mode's gain is `contract kp × kp_scale + kp_offset`, likewise `kd`: the trained gain stays in the contract and the bench trim here |
| `command_terms` | `[]` | the channels this mode's `commands` term carries, in order. Empty takes the host's list, `lin_vel_x`, `lin_vel_y`, `yaw_rate` |
| `task` | `""` | the task the mode's policy was trained on, as the bundler records it. Informative, and what `hook` is looked up by |
| `hook` | absent | the task's deploy hook and its configuration, an inline table copied from the manifest. Present builds the task's `deploy/lib.rs` for this mode — see below; a hook for a task that compiled none, or on a state with no `model` or no `task`, is refused |

**A task's deploy hook.** Some tasks need something done on the robot that no contract
field describes — `jumper.five_foot` carries the claw on the right by mirroring a policy
trained with it on the left. The controller knows no task by name; a task implements
`ModeHook` (`fsm/src/hook.rs`) in `tasks/<family>/<task>/deploy/lib.rs`, and `build.rs`
compiles every such library into every host. The controller calls a hook at the same
points for every task:

| point | when | frame |
|---|---|---|
| `home_pose` | once, building the mode | the pose the mode ramps to and holds, robot frame |
| `reset` | a policy takes over | |
| `before_observation` | every inference, with the state, the operator's command and the controls the task keeps | robot → policy |
| `after_inference` | the raw action, before decoding | policy |
| `after_decode` | the decoded motor command, before it is published | policy → robot |

The decoder runs in the policy frame, on the contract's home pose, and the robot's joint
stops are applied again after `after_decode`. The observation shows the policy its own
last action, as it came out of the network.

A hook answers the operator only through its task's `controls.yaml`: the controls its
top-level `task:` declares, which the controller hands it by name -- and nothing else --
as the operator reads them from whichever device drives, with a `connected` flag that is
false while nobody is there. `ModeHook::reads` names the ones it answers. A hook reading a
name the file does not declare is refused, and so is a mode whose task declares controls
and that asks for no hook: either is a squeeze that does nothing on the robot.
`jumper.five_foot`'s hook closes the claw on the control at its side -- `claw_left`, `LT`
or Space, with the claw on the left; `claw_right`, `RT` or Space, on the right -- and holds
the arm out at a preset for as long as `arm_thumb_up`, `arm_thumb_down` or `arm_web_up` is
held.

**`[[fsm.rule]]`** — `when` and `enter`, **in order**. Every tick, the first rule whose
condition holds decides the state. The order is the safety argument — feedback before
attitude before the operator — which is why it is written in the file rather than
rebuilt from a table.

| `when` | holds when |
|---|---|
| `feedback_stale` | no robot state within `state_timeout_ms`. Never suspended |
| `tilted` | tilt beyond `tilt_limit`, unless the active state sets `skip_tilt_check` |
| `command_stale` | no operator input within `command_timeout_ms` |
| `warm_start_reached` | the active state is a warm start and the measured pose has arrived |
| `in_state:<state>` | the active state is `<state>` |
| `button:<name>` | the `[[fsm.button]]` called `<name>` is active |
| `gripper_active` | never — no host supplies a gripper, so a config that reads it is refused |
| `always` | always. **The last rule has to be `always`, and no other rule may be**, so the cascade is total |

`enter` is a state, `@initial`, `@warm_start_ref`, or `@stay` — match, and change nothing,
which is how a warm start is made uninterruptible.

**`[[fsm.button]]`** — one control on one device, and what it asks for:

| key | |
|---|---|
| `name` | what `button:<name>` reads, or what a contract's `reference.go_event` names. Several entries may share one — one per device, one per key — and then share its latch |
| `pad` | a pad button — `A` `B` `X` `Y` `LB` `RB` `menu` `home` `L3` `R3` — or `dpad_up` `dpad_down` `dpad_left` `dpad_right` |
| `key` | a key by the dictionary's name (`keypad_1`, `key_w`), not a browser's (`Numpad1`); or a modifier's name used as a key (`ctrl`, either of its two keys) |
| `on` | `rise` the tick it goes down · `fall` the tick it comes up · `hold` every tick it is down · `toggle` down until pressed again · `single` `double` `triple` `quadruple` `quintuple` that many presses, each within `click_window_ms` of the last, switching like `toggle` (an `event` fires once). A control that counts clicks takes no single-press gesture, and a `with` modifier takes no click |
| `with` | what must be held: a pad button for a `pad` binding, a modifier — `ctrl`, `shift` or `alt` — for a `key` binding. A modifier is consumed by what it modifies, and while it is held a binding with no `with` on the same control stays quiet: `menu` + `dpad_up` is not also `dpad_up` |
| `from` | the states the binding may be pressed in; absent is any. It gates **entering**: a latch cannot come on, and a one-shot gesture, an event or a `leaves` cannot fire, in a state not listed. A latch already on can always be switched off by its own control |
| `leaves` | latches this binding releases when it fires. Such a binding is a moment — `rise`, `fall` or a click — that latches nothing and is never active, so no rule may read it; the cascade falls through to whatever it reaches with those latches gone |
| `event` | `true`: read by a mode — a recorded motion's `go` — rather than by a rule |

`pad` or `key`, never both: the pad's `A` and the keyboard's Space are two bindings named
`jump`, each with its own gesture, modifier and `from`. The names are
[`controller/vocabulary.json`](../controller/vocabulary.json)'s (`python -m controller
--vocabulary` prints them), compiled into every build of the controller, so a bundle is
checked against the dictionary of the controller it carries. A host that has no keyboard
-- the board -- never presses a `key` binding.

**A key `[fsm]` does not read** — a misspelt setting, or a table this crate has no use for —
is refused rather than parsed as nothing.

**Refused when the file is parsed** — each a config that would load and then quietly not do
what it says: an unknown condition, or a rule naming an unknown state; a last rule that is
not `always`, or an `always` before the end; two states with one name; `initial_state`,
`warm_start_ref` or `safe_state` naming no state; a `safe_state` that skips the tilt check;
a binding with neither `pad` nor `key`, or with both; a pad button, modifier or key outside
the dictionary; a `with` that is not a modifier of its binding's device; a `from` naming no
state; one control bound twice with the same gesture and modifier (the same control with
*different* gestures is the point: press to arm, release to go); a modifier bound as
`hold`; a `leaves` on a `toggle`, a `hold` or an event, or naming a binding that does not
latch; any rule reading `gripper_active`; a rule naming a button nothing declares, or a
binding with `leaves`; a button, not an `event` or a `leaves`, that no rule reads; and an
`[fsm.keyboard]` table.

**Refused when a controller is built from it**, which is the first place the contracts
are present too: a mode with no contract; a contract whose `reference.go_event` names no
`event = true` button, and an `event = true` button no contract names; a switch that may be
pressed in a mode, on a control that mode's controls use -- a pad button or stick its block
reads, a key or modifier its keyboard block uses -- so one press would do both
(`operator::check_switches`, below); a term the host cannot source; and the contract checks
below.

**Unknown keys are ignored, not refused.** Measured at this commit: `hold_curent = true` in
a state and `tilt_limt = 0.1` under `[fsm]` both load, and the misspelt setting takes its
default. A required key cannot fail this way — misspelling it leaves it missing — and an
optional one can.

### How the bundler writes it

From a manifest (`kk-deploy-manifests/1`), `controller.toml` is the manifest's `fsm` file
with the manifest's half appended: a header comment saying so, the file's text unchanged,
then `[[fsm.button]]` entries — for each entry of the manifest's `buttons`, with
`event = true`; for each entry of its `leave`, with `leaves`; and for each mode's `pad` and
`keys` switches, named after the mode — one for a `pad` switch's button and one per key of
a `keys` switch, each with the switch's `on` (`toggle` when it gives none), `with` and
`from`; and one `[[fsm.state]]` with `model = "<mode>.onnx"` and `task = "<task>"` per mode,
modes sorted, plus the mode's `hook` as an inline table when the manifest gives one. The
`fsm` file may declare no button, no `[fsm.keyboard]` and no state with a model — the
bundler refuses one that does, so the two halves cannot name the same control.

With `--mode` and `--fsm`, the file ships as given, and its model states must be exactly
the modes given. With a single `--mode` and no `--fsm`, the bundle carries a synthesised
config: a `safe` state and the mode; rules `feedback_stale → safe`, `in_state:safe →
@initial`, `always → <mode>`; no tilt rule, no ramp, no warm start;
`state_timeout_ms = 200`, `command_timeout_ms = 3600000`. It suits a simulator. A bundle
aimed at hardware brings its own file.

## `<mode>.json` — the contract

The export's `layout.json`, copied without a byte changed. Its `_schema` says
`isaac_layout v1`, and **no reader checks it**: the contract is read field by field, and
a field the reader does not know is ignored.

What the controller reads:

| field | required | |
|---|---|---|
| `wire_joint_order` | yes | every joint the robot has, in the robot's order. **The same in every mode of a bundle**, or the bundle is refused: one bundle drives one robot |
| `obs_joint_order` | yes | the joints every per-joint observation term covers, in that term's order. Paired with the wire **by name** |
| `action_joint_order` | yes | the joints the action drives, in the action's order; as long as `action.dim`. Paired by name |
| `default_joint_pos` | yes | `{joint: rad}` covering **every** wire joint, driven or not. Joints the policy does not drive are held here |
| `joint_limits` | board | `{joint: [lo, hi]}` in the same radians: every joint or none, `lo < hi`, the home pose inside. Every commanded target is clamped to it. The board refuses a contract without it — it has no model to read stops off — and so does the bundler, for a bundle that carries the board's runtime; a simulator falls back to its own model |
| `action_scale` | yes | a number: `target = baseline + action_scale × action`, the baseline being the home pose or a recording |
| `control.kp` `control.kd` `control.effort_limit` | yes | numbers: the trained PD gains and torque limit, before the state's trim |
| `control.control_hz` | yes | this mode's inference rate, > 0. Each mode infers at its own |
| `control.action_filter_cutoff_hz` | no | cutoff of an output EMA between inferences. Absent or 0 is a zero-order hold, which is every export today |
| `observation.dim` | yes | the observation width. The built terms must sum to it |
| `observation.history_length` | no | the frames per term where a term does not say, 1 |
| `observation.terms[]` | yes | `name`, `dim`, and optionally `history_length`, `history_stride` (≥ 1) and `params` — below |
| `action.dim` | yes | the action width |
| `command_ranges` | no | `{term: {axis: [lo, hi]}}`, the command ranges the policy trained on. Needed for every axis a `controller` block drives |
| `controller` | no | how a pad and a keyboard become the command. `operator_controller/2` only — below |
| `reference` | no | the recording the mode is driven by — see the next section |

`kp`, `kd`, `effort_limit` and `action_scale` have to be numbers. The exporter writes a
`{joint: value}` map instead when joints differ, and the controller refuses that contract
with a JSON type error; no task produces one today.

The bundler reads `_checkpoint.path`, for `built_from`. Nothing in the deployment reads
`_source`, `_schema`, `unactuated_joints`, `use_default_offset`, `raw_action_clip`,
`base_height`, `base_init_pos`, `control.decimation` / `sim_dt` / `control_dt`,
`observation.history_order` (the builder always stacks oldest first, which is what the
exporter declares), `observation.normalization` (`baked_into_onnx`: the model takes raw
observations, so never normalise), a term's `offset`, `source_name`, `base_dim`,
`flatten_history_dim`, `scale` or `clip`, `network` or `deploy_notes`. Two of those would
matter if a task set them — `raw_action_clip` and a term's `scale` / `clip` are training
transforms this controller does not apply — and no task sets them today.

### Observation terms

A term is built **by name**, and a name the controller cannot build fails the load rather
than being fed zeros. `SUPPORTED` in `fsm/src/obs.rs` is the list; `scripts/export.py`
keeps a copy and `tests/test_layout.py` holds the two equal.

| term | width per frame | needs |
|---|---|---|
| `base_ang_vel`, `projected_gravity` | 3 | |
| `base_lin_vel` | 3 | a velocity from the host — an estimator on the board, `setSignal` in a browser. The exporter refuses an actor that observes it |
| `base_pose` | 3 | the command's `[pitch, roll, twist]` |
| `commands` — also read as `velocity_commands`, `jump_command`, `jump_go` | one per channel | the state's `command_terms`, or the host's list |
| `commands_diff` | one per channel | the same |
| `posture_command` | 4 | `params.neutral_height`: `[twist, pitch, roll, height − neutral_height]` |
| `joint_pos`, `joint_vel`, `joint_torque` | `len(obs_joint_order)` | `joint_pos` is relative to the home pose |
| `actions` | `action.dim` | the previous action |
| `gait_phase` | 2 | `params.period` in s for a fixed clock, or `stride`, `freq_min`, `freq_max` and `turn_radius` for one that follows the command. `params.command_threshold` gates it to `(0, 0)` while standing. `params.advance`, on the second kind: `"after_frame"` builds each frame with the phase as it stands and then advances it by that frame's command; absent advances it by each frame's own command before building it, the first frame excepted -- the order of every export from before the field. Any other value is refused |
| `jump_phase` | 1 | a `reference` block and its trajectory |
| `ref_future` | `len(reference.lookahead_s) × len(obs_joint_order)` | the same |
| `clip_phase` | 2 | the same |
| `ref_joint_pos` | `len(obs_joint_order)` | the same |
| `ref_joint_vel` | `len(obs_joint_order)` | the same, and a `qd` table |
| `ref_tilt_error` | 3 | the same, and a `root_quat` table |

A term is `width per frame × history_length` wide. Terms are concatenated in contract
order, each term's frames together and **oldest first**. `history_stride` k means the
frames are k control steps apart rather than consecutive: frame i is `i × k` steps before
the newest.

### The `controller` block

Carried by a mode a person steers, written by the exporter from the task's
`controls.yaml` — and that file by the `controls` skill. Its `schema` has to be
`"operator_controller/2"`; any other, the retired `twist_controller/1` included, is
refused by name.
Shortened from `jumper.posture`'s:

```json
"controller": {
  "schema": "operator_controller/2",
  "command": [
    {"term": "twist", "feeds": "velocity_commands",
     "axes": [{"name": "lin_vel_x"}, {"name": "lin_vel_y"}, {"name": "ang_vel_z"}]},
    {"term": "posture", "feeds": "posture_command",
     "axes": [{"name": "twist"}, {"name": "pitch"}, {"name": "roll"},
              {"name": "height", "rest": 0.10647, "integrate_s": 1.0}],
     "bands": {"standing_below": 0.06, "velocity_term": "twist",
               "moving": {"twist": [-0.26, 0.26], "pitch": [-0.26, 0.26],
                          "roll": [-0.26, 0.26]}}}
  ],
  "task": {},
  "devices": {
    "gamepad": {"scheme": "absolute", "deadzone": "device_reported_rescaled",
                "release_button": "B", "shift": {"button": "R3", "gesture": "hold"},
                "reset": "R3",
                "axes": {"lin_vel_x": {"source": "Ly", "sign": -1},
                         "twist": {"source": "Rx", "sign": -1, "travel": [0.0, 0.5, 0.5, 0.75]},
                         "ang_vel_z": {"source": "Rx", "sign": -1, "travel": [0.5, 1.0]},
                         "height": {"source": "Ry", "sign": -1, "shifted": true}}},
    "keyboard": {"scheme": "keys", "full_after_s": 2.0,
                 "axes": {"lin_vel_x": {"+": ["key_w", "key_up"], "-": ["key_s", "key_down"]},
                          "ang_vel_z": {"+": ["key_j"], "-": ["key_l"]},
                          "twist": {"+": ["key_h"], "-": ["key_semicolon"]},
                          "height": {"+": ["key_n"], "-": ["key_m"]}},
                 "release": ["key_escape"]}
  }
}
```

`OperatorSpec::check` lists every rule it enforces. What a consumer needs from it:

- **The controller turns the pad and the keys into the command itself, on every host**,
  from this block. A host hands over raw input and never maps a stick to a command of its
  own — a second mapping is how a sign corrected in `controls.yaml` stops reaching one
  host.
- **The pad and the keyboard are two paths.** `gamepad.axes` binds sticks and triggers to
  command axes, with a sign, a stretch of travel and a layer; `keyboard.axes` binds
  keystrokes to one direction of each axis — `"+"` the axis's own positive direction, so
  the keyboard carries no sign — or says why an axis has none (`{"unbound": "<why>"}`). A
  keystroke is a key, a modifier alone, or `shift+key_j`, which is not also `key_j`.
  Neither side names a control of the other.
- **An axis may be moved rather than placed.** `integrate_s` on an axis makes a
  deflection a speed that carries it from its rest to either end of its range in that
  many seconds; let go, it stays. The controller keeps the position and puts it back at
  the rest on the release, on a tap of the pad's `reset` button, and when the operator
  goes away.
- **Each mode its own.** A mode's block is how that policy is driven, at that contract's
  `command_ranges`, and modes need not agree: `jumper`'s walk moves the body's height on
  `R3` and the right stick, or `N` and `M`, and walks at up to 0.8 m/s; its claw modes
  close a claw on a trigger, or Space, and walk at up to 0.5. Every mode's operator hears
  every pad frame and every key, whichever mode is running, so a shift or a release
  carries across a switch. A mode without a block reads the host's own command
  (`set_command`). This was one block per bundle, refused otherwise, until 2026-09-26;
  `operator::Operators` has what that cost.
- **One control, one meaning.** Opening a bundle refuses one whose FSM has a switch that
  may be pressed in a mode on a control that mode's block uses — its release, its shift,
  its reset, a stick a command reads, a pad control its task keeps, a key or a modifier its
  keyboard uses — because one press would do both (`operator::check_switches`). A switch
  says with `from` which modes it is pressed in: Space is the jump from walking and the
  claw in a claw mode. The one sharing allowed is a d-pad direction the task keeps and a
  FSM **chord** on it that leaves the mode on the press -- `menu` + up into a dance while
  up alone holds `jumper.five_foot`'s arm out (`FsmConfig::chord_leaves_first`). The
  keyboard has no such exception.
- **A task may keep controls for itself.** The block's `task` names each control the task
  answers itself, with its `kind` — `amount`, 0 to 1 as far as it is held, or `press`, 1
  while it is — and what it `does`; it is `{}` for a task that keeps none. Each device's
  own `task` binds every one: the pad one control each, an axis for an `amount` and a
  d-pad direction for a `press`, and the keyboard one or more keystrokes --
  `jumper.five_foot`: `claw_left` on `LT` and Space, `arm_thumb_up` on `dpad_up` and
  Shift. No command may bind the pad's, and on the robot the mode's hook is the one reader,
  by name (see *A task's deploy hook*).
- **A command may narrow while walking, and may ramp.** Both are how the policy was
  trained, so the controller applies them before every observation, on every host
  (`CommandShaper`, clamp then ramp):
  - `bands` — `command_ranges.<term>` is then the **standing** band, what a full stick
    reaches with the robot parked. While the norm of the `velocity_term` block's
    `[lin_vel_x, lin_vel_y, ang_vel_z]` is at or above `standing_below`, each axis in
    `moving` is clamped to its `[low, high]`. `jumper.posture` and `jumper.five_foot`
    both carry one: twist 30°, pitch 20°, roll 15° standing, 15° on each walking.
  - `max_rate` — how fast the command may move, in its units per second, starting from
    the rest each time a policy takes over. `jumper.five_foot`'s pose ramps at 30°/s
    (0.5236), because its policy never saw a step. Absent is a command that steps.

  Either one malformed — a band outside the standing range, an axis the block does not
  have, a `velocity_term` with no speed in it, a rate of zero — refuses the contract
  rather than being skipped: a skipped band is a walking robot handed the standing lean.

## `<name>.trajectory.json` — a recording

Carried for every mode whose contract has a `reference` block, and required by it: a
reference-guided policy is not runnable without the recording it trained against.
`jumper.jump`'s action is a residual on one; `jumper.dance` drives its joints outright and
observes one.

| key | shape | |
|---|---|---|
| `joint_order` | [string] | the wire order, exactly. The exporter permutes the tables into it |
| `q` | `n_frames × joints` | the state the recording reached, at `rec_hz` |
| `q_cmd` | `n_cmd × joints` | what the recorded controller commanded, at `control_hz`: the residual's baseline. Required when `residual_action` |
| `qd` | `n_frames × joints` | the recorded joint velocities. Required by `ref_joint_vel` |
| `root_quat` | `n_frames × 4` | the anchor body's orientation, `(w, x, y, z)`. Required by `ref_tilt_error` |
| `_source`, `_note` | | for a person |

Values are rounded to five decimals. `q` and `q_cmd` are different tables at different
rates and lengths — 368 rows at 250 Hz against 295 at 200 Hz for `high_jump_flat` — and
are not interchangeable.

The contract's `reference` block says how to read it:

| key | required | |
|---|---|---|
| `file` | yes | the table's file name, beside the contract |
| `residual_action` | yes | the action is a residual on `baseline` rather than on the home pose |
| `rec_hz` | yes | `q`'s rate, > 0 |
| `duration` | yes | the whole recording, s, > 0. The motion has **ended** once `t_go` + time since `go` reaches it |
| `lookahead_s` | yes | the preview offsets `ref_future` samples, s |
| `n_frames` | yes | rows in `q`; has to match the table |
| `n_cmd` | with `q_cmd` | rows in `q_cmd`; has to match the table |
| `control_hz` | residual | `q_cmd`'s rate |
| `baseline` | residual | `"q_cmd"`, the only one implemented |
| `baseline_lead_s` | no | the baseline is sampled this far ahead, s |
| `t_go`, `go_frame` | no | the recorded stand before the motion starts; a motion waiting for `go` reads frame `go_frame` |
| `span` | no | what `jump_phase` runs 0 → 1 across, s |
| `starts_on_entry` | no | the recording starts when the mode is entered — the dance, and the jump since 2026-09-26 |
| `go_event` | no | the name of the `event = true` button that starts it — a motion a person times inside its mode |

`starts_on_entry` and `go_event` may not both be set. With neither, the motion starts on a
timer, `reference_go_delay_s` — 2 s unless the host says otherwise — after the mode is
entered: a bench aid, not something an operator can use. A recording that ends releases
its mode: the controller reports it, the host releases the latch, and the cascade's
`always` rule takes the robot back.

Refused when the table is attached: a table in another joint order; row counts that
disagree with the block; a row of the wrong width; `qd` or `root_quat` not as long as `q`;
a residual with no `q_cmd`, no `control_hz`, or a `baseline` other than `q_cmd`; `rec_hz`,
`duration` or `span` not positive; both `starts_on_entry` and `go_event`; and, when the
mode is built, a term that needs `qd` or `root_quat` from a table without it.

## `reference.json` — `kk-policy-reference/1`

Frames of the controller running, recorded once on the build machine, one copy every host
replays, so each host is held against the same numbers rather than against the others.
Compact JSON.

| key | |
|---|---|
| `schema` | `"kk-policy-reference/1"`. A reader refuses any other |
| `source`, `note` | for a person |
| `commit` | the controller's commit, as the bundle's `runtimes` record it |
| `joints` | the wire order |
| `terms` | `[{name, offset, dim}]`, where each observation term sits, so a difference is reported by term |
| `frames` | below |

Each frame holds what went in — `now_us`; `q`, `qd`, `tau` in wire order; `quat`
`(w, x, y, z)`; `gyro`; `cmd` `[lin_vel_x, lin_vel_y, yaw_rate]`; and `axes`,
`{channel: value}`, when a mode has a `controller` block, applied after `cmd` — and what
came out: `mode`, the state after the tick; `target`, the published joint targets; and on
a frame that inferred, `infer` (the mode), `obs` (the observation) and `act` (the action
onnxruntime returned for it).

How it is made (`write_reference` in `scripts/deploy.py`): the build's own `controller.so`
— the same source as every host's runtime — runs 24 synthetic, seeded frames 20 ms apart,
starting up to 0.25 rad from the home pose and converging on it over the first eight, so
a config with a ramp has the ramp on the record. The home pose, `terms` and `axes` are the
mode the cascade starts in — `warm_start_ref`, else `initial_state` — and not the first
mode by name: in the `jumper` build that is `claw_left`, whose home holds the carried arm
0.5 rad and its finger 0.65 rad from the walking stance (1.6 rad on the arm before the stow
of 2026-09-28), and frames converging on it never reached the pose the warm start waits
for. The command is constant: `cmd` is `[0.3, 0, 0.15]`, and when that mode
has a `controller` block (else the first that does), `axes` puts every axis a quarter of
the way from its rest to the top of its range. Each inference runs the **export's**
`actor.onnx` on CPU — for the board too, whose `.rknn` is exactly what that ONNX has to be
held against. Only an `--allow-incomplete` build can lack `reference.json` (`reference`
is then `null`, and a note says why); a default build refuses the app instead. The `jumper` build has 18
inferences and 6 frames of ramp.

**Replaying it**: for each frame, hand over its state and command, tick, and on a frame
that inferred resume with the **recorded** `act` — never your model's — so a difference in
`obs` or `target` is this host's controller and never the model. Then, separately, run each
recorded `obs` through your own model and compare with `act`:

| comparison | measures | expected |
|---|---|---|
| `obs` | this host's controller | **exact**. Every host uses 1e-6 |
| `target` | the decode, the clamp, the filter | **exact** |
| `mode` | the cascade | the same state on every frame. A mismatch fails at any tolerance |
| `act` | this host's inference backend | ~0 for an ONNX runtime; ~1e-3 for a quantised `.rknn` |

Terms a host declares it sources differently from training — on the board nearly all of
them, measured rather than simulated — are reported apart from `obs` with their worst
difference, and do not fail the check. The replay feeds every host the same inputs, so
they normally agree exactly anyway.

Measured on the `jumper` build: `play`'s extension through `deploy.py --check-reference`,
and the browser's wasm through `tests/web/load_bundle.mjs` under Node 24 — 0 on `obs` and
on `target`, no mode mismatch, on both.

## `manual.<language>.json` — `kk-bundle-manual/1`

For a person holding the pad or sitting at the keyboard: every control the bundle reads,
what it does in each mode, and how each mode is reached — the pad and the keyboard listed
apart, because they are two paths (since 2026-09-29; before, each key was listed under the
pad control it stood in for). `scripts/deploy.py` writes `manual.en.json` on every
build from the controller's own `pad_guide()` — each task's `controls.yaml` as its contract
carries it, the mode switches as `controller.toml` composes them from the manifest — so it
describes exactly the bindings in the bundle, and a bundle whose controller cannot answer is
refused. A mode lists only the task controls its hook answers: the left claw's lists `LT`
and Space, not `RT`. No controller reads it; a page may draw from it. Two-space JSON:

| key | |
|---|---|
| `schema` | `"kk-bundle-manual/1"` |
| `language` | `"en"`, or the translation's code |
| `bundle` | the bundle's name |
| `source` | how it was made, for a person |
| `modes[]` | the cascade's default mode first: `mode`, `task`, `default`, `summary`, `enter` (how it is reached, on either device), `leave`, `keyboard` (how a held key moves what it drives); `controls[]`, the pad in this mode — each `control` by the dictionary's name, `pad` as printed on the pad, and what it `does`; and `keys[]`, the keyboard in this mode — each `key` as printed on the keys (`W`, `Shift + J`, `Space`) and what it `does` |
| `switches[]` | the switches, the pad's and the keyboard's each on its own: `device` (`pad` or `keyboard`), `control` (the pad control, or the keystroke — `ctrl+key_1`), `with` (a held modifier), `keys` (the keyboard's, as printed: `Ctrl + 1`), `pad` (the pad's, as printed: `hold Menu, then D-pad up`), `does`, and `from`, the modes it may be pressed in (`null`: any) |

A translation travels beside it as `manual.<language>.json` and is added after the build:

```bash
python scripts/deploy.py --translate out/bundle_<timestamp>/<name> --language zh --manual <file>
```

It goes in only as the English one with its text changed — `source`, `summary`, `enter`,
`leave`, `keyboard`, `pad` and `does`, in the pad's list, the keyboard's and the switches
alike — and every mode, control, key, modifier, device and `from` where the English has it;
then it is listed in `manuals` and `files`, the bundle is held to
[`app.schema`](app.schema) again, and the `.app` is packed again. The
[`bundle-manual`](../.claude/skills/bundle-manual/SKILL.md) skill is how an agent writes
the Chinese one. The English is always the source: a manual written first in another
language would have no generator holding it to the bindings.

## `<name>.app` — the archive

Written beside the bundle directory once the directory has passed the bundler's check and
[`app.schema`](app.schema), and again by every `--translate`, for a consumer that is not a
filesystem — a file picker, an upload form, a copy to a robot. It is a **zip under another
name**: any zip reader opens it as it is. A consumer that chooses files by name has to list
`.app` — a browser picker whose `accept` names only `.zip` does not offer the file at all,
and nothing says why. The directory's contents sit **at the root** of the archive,
`bundle.json` among them: a zip of the directory itself would put everything one level
down and read as a bundle with no manifest. Entries are sorted, deflated, dated
1980-01-01 00:00:00 and carry the file's Unix permission bits, so
`runtime/board/controller` stays executable and two builds of the same files are the same
bytes. The directory stays: the board and the checks read the directory, and `play --app`
opens either -- the `.app`, unzipped for the run, or the directory beside it.

## `app.schema` — what a bundle may hold

[`app.schema`](app.schema) is the bundle written down for a machine: JSON Schema (draft
2020-12) for `bundle.json`, the same for every manual under `$defs/manual`, and `x-files`,
every file a bundle may hold as a path pattern — whether it is required, its format, and
which `bundle.json` field names it. `validate_app` in `scripts/deploy.py` holds every build
to it before it is packed, and every translation before it is packed again: `bundle.json`
and each manual against the schema, every file on disk matching an `x-files` entry, every
required one present, and `files` listing exactly what is on disk. A bundle that fails is
not packed, and the message names each departure. `tests/test_app_schema.py` pins that
each departure is refused, and holds the newest bundle built on the machine to it.

A consumer can validate against it without reading this document. It says what a bundle's
fields are; what a host has to *do* with them is the rest of this document.

## Consuming a bundle

### What every consumer has to do

1. **Refuse a schema you do not know** — `kk-policy-bundle/1` here, `kk-policy-reference/1`
   for `reference.json`, `operator_controller/2` for a `controller` block. Anything else
   means fields you would be guessing at.
2. **Take your own part, and only it.** Open the bundle as the host you are: your model
   format, `runtimes.<host>.model`, from each mode's `models`, and your build of the
   controller from `runtimes.<host>`. A host with no entry there has no build in this
   bundle — show the note that says how to add one rather than borrowing another host's.
   The board is the one host with no model of the robot, so opening as the board refuses a
   contract without `joint_limits`.
3. **Run the controller the bundle carries**, not another build of the same source — the
   two diverge the moment somebody rebuilds one of them. Where you cannot (another
   platform), say which one you ran.
4. **Show `notes` to a person**, and what the controller reports as it loads: modes on a
   stub, terms sourced differently from training, the bindings it will read.
5. **Load every mode in `modes`**: its contract; its model, from
   `modes.<mode>.models.<your format>` and never from the state's `model`; and when the contract has a `reference` block, the text
   of `reference.file` from beside the contract. A mode without its recording is refused
   by name rather than built from zeros.
6. **Pair joints by name.** Every per-joint array you pass or receive is in wire order. On
   this robot the locomotion policies and `jumper.jump` drive 20 of the 22 joints, and the
   two they leave out sit at wire indices 4 and 9 — so pairing by index is not off by a
   constant, it is scrambled from the fifth joint on.
7. **Take the stops from the contracts.** `joint_limits` is what the robot clamps to. A
   host that also has a model of the robot prefers the contract, and says which it used
   when the contract has none.
8. **Give the controller its clock.** Pass monotonic microseconds to every call. The
   controller has no clock of its own; freshness, inference rates, ramps and timers all
   come from what you pass.
9. **Keep both inputs fresh**: robot state at least every `state_timeout_ms`, operator
   input at least every `command_timeout_ms`. Otherwise the cascade sees stale feedback,
   or forgets every latched mode and puts the command back at rest.
10. **Replay `reference.json` on load**, on a fresh controller — the replay drives it
    through somebody else's frames and leaves it in their state — and treat anything above
    1e-6 on `obs` or `target`, or any mode mismatch, as a controller that is not this one.
11. **If the bundle is an upload, say that nothing has checked its controller**
    (`runtimes.web.unverifiedByTheConsumer`), and isolate it — see the browser, below.

Checking the digests in `files` against the directory is open to any consumer, and no host
does it today; the bundler checks only that `files` lists exactly what is there.

### The step

One crate, three builds, one protocol — the board runs it inside `controller`, the other
two hosts call it:

```
every tick:
    set_state(q, qd, tau, quat, gyro, now_us)        # wire order
    hand over the operator's input                   # below
    mode = tick(now_us)                              # a mode's name, or nothing
    if mode:
        action = model[mode](observation())          # float32 [1, obs dim] -> [1, action dim]
        resume(action)
    publish(positions(), kp(), kd())                 # wire order, every tick
```

- `tick` names a mode only when that mode's inference is due. Each mode infers at its own
  contract's `control_hz`, so a 50 Hz and a 200 Hz mode share one bundle and one loop.
  Call `tick` at least as often as the fastest mode.
- Every other tick — between inferences, in a ramp, a warm start or a hold — is finished
  when `tick` returns, and `positions()` is ready to publish.
- `resume` answers the tick that asked, once, with exactly `action.dim` values; out of turn
  it is an error. A request left unanswered is dropped by the next `tick`.
- The model takes the raw observation. Normalisation is inside the ONNX.
- The mode (`mode()`) and whether its policy is running (`is_running_policy()`) differ
  through a mode-switch ramp, which waits on the measured pose and may wait indefinitely.
  Show both, or a robot holding still looks broken.

### What goes in

| input | what it must be |
|---|---|
| `q`, `qd`, `tau` | wire order; rad, rad/s, N·m. `tau` is the applied joint torque — the servo's on the robot, the solver's in a simulator |
| `quat` | the body's orientation, `(w, x, y, z)` |
| `gyro` | angular velocity **in the body frame**. A simulator usually reports the world frame, and MuJoCo's 6-vector velocity is `(angular, linear)` |
| `now_us` | monotonic microseconds |
| pad sticks and triggers | `Lx` `Ly` `Rx` `Ry` in [-1, 1], `LT` `RT` in [0, 1], **as the robot's pad service reports them: a 0.15 deadband already applied, and rescaled so full deflection still reaches ±1.** The controller applies none. A host that reads a pad itself applies it first — an undeadbanded stick that drifts counts as the pad being touched, and takes the command from the keyboard |
| pad buttons, keys | by the dictionary's names, down **and** up — `fall` and `toggle` need the release, and a key on an axis pushes it only while down: full after the block's `full_after_s` (at once on an axis with `integrate_s`), back at rest on the release. Pad buttons every frame, keys on every event, a modifier's left and right keys each as itself; a browser passes `KeyboardEvent.code` and the controller maps it |
| the command | a mode with a `controller` block has the controller build its command from the pad and the keys, and never reads `set_command`, which still keeps the input fresh. A mode without one reads `set_command(vx, vy, wz)`, in m/s and rad/s |

### What the host supplies

What only the host knows. The board derives it from the bundle and `--machine`; a browser
and `play` pass it as a `robot` object:

| field | required | default | |
|---|---|---|---|
| `joint_names` | yes | | the contracts' `wire_joint_order` |
| `joint_pos_lo`, `joint_pos_hi` | yes | | `joint_limits`, in wire order |
| `output_rate_hz` | in `play` | 50 | how often you call `tick`. It sets the output filter and nothing else |
| `gait_period`, `gait_gate_threshold` | no | 0.32 s, none | only for a `gait_phase` whose contract carries no clock; every current export carries one |
| `reference_go_delay_s` | no | 2 s | the timer for a recording with neither `go_event` nor `starts_on_entry` |
| `joint_pos_rate_limit` | no | 0, off | the largest change of a joint target per policy step, rad |
| `command_terms` | no | `lin_vel_x` `lin_vel_y` `yaw_rate` | the `commands` channels of a state that names none |
| `obs_clip`, `action_clip`, `action_smoothing` | no | 100, 100, 0 | a browser's to set; `play` fixes them at these |

### The board — `runtime/board/controller`

From inside the bundle, copied to the robot or unzipped from the `.app` there:

```bash
./runtime/board/controller --bundle . --dry-run            # open it, report, touch no bus
./runtime/board/controller --bundle . --check-reference    # ...and replay reference.json
./runtime/board/controller --bundle . --machine /etc/mjrl/machine.toml
```

It opens the directory as the board, `Bundle::open(dir, Target::Board)` — every check in
this document not marked as another host's, `joint_limits` included — and builds one
engine per mode from `modes.<mode>.models.rknn`; a model file that is not there fails the
open. A mode never converted has no `rknn` and takes its `onnx` instead, which the board
cannot run: it runs on a **stub** returning zeros — the home pose, held — as does a mode
whose `.rknn` will not load, and each is reported as one. A mode whose action is a residual on a recording is
**barred** instead, because there a stub's zeros would play the recording open loop. The
loop runs at 1 kHz and publishes every tick; each mode infers at its own rate.

`--machine` holds what a bundle cannot know. Every key is optional:

| key | default | |
|---|---|---|
| `motor_domain`, `robot_domain` | 0, 0 | DDS domains. Nothing holds them against the publishers; see `README.md` |
| `qos_xml` | `/etc/mjrl/qos.xml` | the QoS profiles |
| `joint_pos_rate_limit` | 0 | rad per policy step; 0 is off |
| `obs_clip`, `action_clip` | 100, 100 | |
| `action_smoothing` | 0 | |
| `has_velocity_estimator` | false | without one, a contract observing `base_lin_vel` is refused rather than fed zeros |

`--check-reference` implies `--dry-run`. It also runs each recorded `obs` through the
mode's engine when that is not a stub, and exits non-zero only when the controller or the
cascade disagrees; the action's difference is printed for you to judge.

### The browser — `WebFsm`

A page is its own bundle reader: the wasm has no filesystem and never sees `bundle.json`.
So the page does what `Bundle::open` does elsewhere — checks `schema`, loads the controller
from `runtimes.web`'s `glue` and `wasm`, loads every mode in `modes` with its
`models.onnx`, shows `notes` — and then builds the controller:

```js
new WebFsm(
  controllerToml,                  // controller.toml, as text
  JSON.stringify(contracts),       // {mode: <that mode's contract, parsed>}
  JSON.stringify(robot),           // "What the host supplies", above
  nowUs,
  JSON.stringify(trajectories),    // {mode: <the text of its reference.file>}; omit when none
);
```

Note the two maps differ: contracts go in **parsed**, trajectories as **text**.
[`tests/web/load_bundle.mjs`](../tests/web/load_bundle.mjs) is a working loader —
`tests/test_web_host.py` runs it on the newest `out/bundle_*/<name>` that carries a
browser runtime — and the bundle's
README documents the rest of the API. Of the checks the bundler has already run, the
constructor repeats "one control, one meaning" (`Operators::beside`, the same
`operator::check_switches` `Bundle::open` runs); one wire order across the modes is not
checked again.

Hand over every quantity the page can compute with `setSignal`, whether or not today's
bundle reads it — `base_lin_vel`, body frame, m/s, is the one read now, and a bundle that
observes it gets zeros from a page that does not pass it.

A page that runs a bundle somebody uploaded runs code nothing has checked:
`runtime/web/controller.js` has the page's authority. Load it in an `<iframe sandbox="allow-scripts">`
without `allow-same-origin`, talk to it over a `MessagePort`, and tear it down when it
stops answering; the README has the measurements.

### `play` — `play --app`

```bash
python scripts/play.py --app out/bundle_<timestamp>/<name>.app
python scripts/deploy.py --check-reference out/bundle_<timestamp>/<name>
```

`play --app` takes the `.app` or the bundle directory beside it, and imports
`runtimes.mjlab.extensions[<this machine's sysconfig.get_platform()>]` when the bundle
carries one for it — on a Mac, `macosx-universal2`, whatever version its platform string
carries (`mjrl.app_play.extension_for`, which `--check-reference` asks too) — and
otherwise whatever `mjrl_fsm` is installed, printing which
— and, when it had to fall back, the platforms the bundle has. It loads that file by its
path with the loader named, never by what `importlib` makes of the suffix: Windows
recognises `.pyd` alone, and a file it did not recognise used to fall through to the
installed one. On Windows the unzipped `.pyd` stays in the temp directory after `play`
exits, since a loaded DLL cannot be deleted. It takes the stops from the
simulator's model, passes each recorded motion its table, runs each mode's `models.onnx`
on CPU, and hands the controller the viewer's keys and a gamepad's frames as a robot's pad
service would. What drives the simulated servos is the controller's own output —
`positions()`, `kp()` and `kd()` every step — so the ramps, the output filter, the stops
and a task's deploy hook all reach the joints, and a mode's action width is not the
environment's concern. The environment, the app's default mode's task unless `--task`
names another, is the world only. Pointed at the build directory it refuses and names the
`.app` inside. `deploy.py --check-reference` checks this machine's extension, given the
bundle or the build directory around it, and refuses a bundle that carries none for this
platform.

## What a bundle does not carry

- **Anything about the machine.** DDS domains, QoS, the rate limit, the clips and whether
  there is a velocity estimator are the board's `--machine`; a simulator's model, scene
  and step are its own.
- **The control dictionary.** `controller/vocabulary.json` is compiled into every build of
  the controller, so the names a bundle may use are the ones its own runtime knows.
- **The deadband.** The robot's pad service applies it before anything here sees a stick.
- **The weights as trained.** The export keeps the `.pt`; `built_from` says which one.
- **Anything a simulator can measure** — that belongs in the contract. When a number like
  that turns up in a host's config it is moved there: `joint_limits` and the stick scales
  both were.

## Open

Measured against this commit. Each is a place where the implementation disagrees with
itself, or checks less than a reader of the sections above would assume.

- **The bundler builds `play`'s controller, never the board's.** `verify_bundle` runs
  `fsm/examples/check_bundle.rs`, which calls `Bundle::open` — as every host `runtimes`
  declares — and nothing more. Writing the manual does build a controller, through the
  bundle's own extension, and a failure there (a `go_event` and an `event` button that do
  not pair up, say) removes the bundle, even under `--allow-incomplete`. But that
  controller is built without operators and as the `mjlab` host, so what only the
  board's controller refuses — a term the board cannot source — is not checked at build.
- **`reference.json` replays only the mode the cascade reaches with no input.** No frame
  presses anything, so in the `jumper` build all 18 inferences are `locomotion`, and the
  jump, the dances, the gestures and the two claw modes are replayed on no host.
- **One set of stops per controller, from an arbitrary mode.** The clamp uses a single
  `joint_limits`, taken from whichever contract a hash map yields first. Nothing checks
  that the modes agree. Every current export does — 0.0 rad apart across the four
  `jumper` exports, measured 2026-09-26. The home pose is each mode's own for its ramp,
  its observation and its decode; the one a hash map yields first seeds only the targets
  before the first tick and the warm start's target when `warm_start_ref` names no mode —
  in `jumper` it may be `claw_left`'s, 0.5 rad from the stance on the carried arm and
  0.65 on its finger, and neither use reaches the robot there.
- **Unknown keys are ignored** in `controller.toml` and in contracts, so a misspelt
  optional key takes its default without a word.
- **No host checks the digests in `files`** against the directory it loads.
- **Neither cross-built `play` extension is run where it will be loaded.** A bundle
  carries three: the build machine's, `win-amd64` and `macosx-universal2`. On a platform
  it does not carry, `play --app` falls back to the installed extension and says so, and
  `deploy.py --check-reference` refuses. Measured 2026-09-29 on this repository's Linux
  workstation:
  - *Windows*, under wine 10.0 with CPython 3.11.9's embeddable build:
    `--check-reference` passes, and `play --app`'s open, load and close run. wine 8.0
    cannot load it at all — it lacks `bcryptprimitives.dll`, which Rust's standard
    library imports and Windows 10 has.
  - *macOS*, statically only: both slices export `PyInit_mjrl_fsm`, bind CPython's
    symbols at load time, link nothing outside `/usr/lib` and need macOS 11.7.1; the
    arm64 slice's ad-hoc signature holds (716 page hashes, against the file taken back
    out of the `.app`). It has not been loaded on a Mac.
  - *Intel Macs cannot install this repository at all*: `mujoco` 3.11.0, `torch` 2.7.0
    and `warp-lang` 1.14.0 publish macOS wheels for arm64 only (PyPI, 2026-09-29), so
    the x86_64 slice has no environment to load it yet — nor has an arm64 Mac's Python
    under Rosetta.
- **The published targets differ by one float32 ulp from one process to the next.**
  `--check-reference` on one bundle, on one machine, gives `targets worst 0` on some
  runs and `2.235e-08` on others — eight runs of the 2026-09-28 `jumper.app`: five and
  three — on Linux and on Windows alike. The check's tolerance, 1e-6, passes both, though
  its comment expects bit-exact. A hash map's per-process order reaching the targets is
  the likely cause.
- **Under `--allow-incomplete`, `runtimes.board.commit` is the source tree's, not the
  binary's.** The bundler then takes whatever `fsm/docker-build.sh` last left in
  `out/deploy/controller-aarch64` and says it may predate the source. A default build
  cross-builds it from this source first.
- **The board passes the fastest policy's rate as `output_rate_hz` while its loop ticks at
  1 kHz.** That value feeds only the output filter, which no export enables
  (`action_filter_cutoff_hz` is never written), so it changes nothing today.
- **`web.rs` tells a page to pass the Gamepad API's values "unchanged … no deadzone"**,
  which contradicts the operator block's `device_reported_rescaled` and the bundle's own
  README. *What goes in*, above, follows the block.
