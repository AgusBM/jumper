# fsm — the controller, on all three hosts

One crate: the observation, the action decode and the state machine. Three hosts
run it — the robot links it into `controller`, a browser loads it as wasm, `play
--app` imports it as a Python extension — so the logic exists once and the three
differences are ports rather than reimplementations.

This file describes the robot's path, which is the one with hardware in it.
[`DESIGN.md`](DESIGN.md) is the structure the three share.

```
layout.json ─┐
             ├─▶ observation ─▶ RKNN (NPU) ─▶ action ─▶ joint targets ─▶ DDS
robot state ─┘
```

Two kinds of state, and the boundary between them is the design:

| state | what it does | needs a model? |
|---|---|---|
| **holds** | keeps a pose: the one it is in, or the home pose from `layout.json` | no |
| **runs a policy** | builds the observation, runs the model, applies the action | yes |

A holding state touches no model, so it is where the cascade sends a robot whose
feedback has stopped or whose attitude has gone — and what the robot does before
anyone asks it for anything. A mode whose model is missing or unloadable is not
that: it stays a mode and runs on a stub returning zeros, which decodes to the
home pose and is reported as a stub rather than as running — except a mode whose
action is a residual on a recording, where zeros would replay the recording open
loop, so it is barred from running while its engine is a stub.

## Running it

`controller` is the binary the robot runs, and it travels inside the bundle as
`runtime/board/controller`. From the bundle's directory on the robot:

```bash
./runtime/board/controller --bundle . --machine /etc/mjrl/machine.toml
./runtime/board/controller --bundle . --dry-run          # open it, report, touch no bus
./runtime/board/controller --bundle . --check-reference  # ...and replay its frames
./runtime/board/controller --vocabulary                  # the words a manifest may use
```

`--bundle` is the bundle `scripts/deploy.py` writes, the one every host
loads. The controller opens it as the board and takes the board's part: the FSM
config, each mode's contract and `.rknn`, plus the recorded trajectory for any
mode whose contract names one. Everything the policy knows about itself comes
from there. `--machine` is what is left — the DDS domains, the QoS
XML, and the few constants that describe the actuators rather than the policy:

```toml
motor_domain = 0
robot_domain = 0
qos_xml = "/etc/mjrl/qos.xml"
joint_pos_rate_limit = 0.20     # rad per policy step, at each mode's control_hz
has_velocity_estimator = false  # without one, base_lin_vel is absent, not zero
```

That file is short on purpose and gets shorter. Every number in it is a number
nothing holds against the robot it describes, so anything a simulator can
measure belongs in the contract instead — `joint_limits` and the stick scales
both used to live here. Neither has a fallback now: the board refuses a contract
without `joint_limits`, and `--dry-run` prints each mode's controls as its contract
describes them.

## Building

```bash
bash deploy/fsm/docker-build.sh          # cross-build for the board
bash deploy/fsm/docker-build.sh test     # the host test suite
```

Anything after the script name goes to cargo.

A **bundle** is the controller plus the policies it drives, in one directory
and one `.app` of it that somebody with none of this repository can use, on any
of the three hosts:

```bash
python scripts/deploy.py
```

It runs the cross build above first, every time, so the board's binary in the
app is this source's; the browser's wasm and `play`'s extension are different
targets and need no container. It ships [`BUNDLE_README.md`](BUNDLE_README.md) with the substitutions filled
in -- the integration guide, and the place to write down anything a consumer
cannot infer from the files -- and `manual.en.json`, the pad and the keys as
this crate's `pad_guide()` reports them. Before it is packed, the bundle is
loaded back through `examples/check_bundle.rs`, which is this crate reading it
exactly as each host it carries a runtime for will; a bundle that fails is
removed rather than left for someone to upload. Then it is held to
[`../app.schema`](../app.schema).

For the wasm, a cargo build and wasm-bindgen are what it runs. Run it rather
than them: it reads the commit **after** looking at the tree, suffixes it `-dirty`
when the controller's source has uncommitted changes -- the crate, and every file
outside it the build reads, each task's hook among them (`SOURCES` in the script)
-- and holds the wasm-bindgen CLI
against the version the crate compiles with. The hand procedure lasted one day
before it recorded a commit whose source would have built a different controller
-- see the script's docstring.

A dirty tree used to be **fatal** unless `--allow-dirty` was passed. That flag
is gone and the default is now the other way round: an uncommitted build is what
every local run is, and refusing it made the test suite red for exactly as long
as somebody was editing the crate. The discipline survives in the `-dirty`
suffix on the commit the bundle records, which cannot be mistaken for something
reproducible. `--require-clean` is the opt-in, for the build that is going
somewhere.

**Cross-compiled, not emulated.** Running an arm64 container under qemu needs a
host-wide `binfmt` registration carrying the `F` flag; changing that is not a
build's business, and emulation is far slower. The image is x86_64 with an
aarch64 toolchain.

**The base image is `ubuntu:22.04` on purpose.** The board's rootfs is Ubuntu
22.04 (glibc 2.35), and glibc is a floor, not a ceiling: a binary linked against
bookworm's 2.36 dies on the board with `version GLIBC_2.36 not found`, at
startup, far from the build that caused it.

CycloneDDS is compiled twice in the image — x86_64 for `idlc` (a code generator,
so it must run on the build machine) and aarch64 for `libddsc.so` (what the
binary links against). `docker-build.sh` checks the aarch64 soname against the
board's `libddsc.so.11` and warns if they differ, because that mismatch links
cleanly and then fails to load on the robot.

On Linux you can also build natively — `cargo test` needs CycloneDDS installed
and Rockchip's header fetched once with `bash vendor/rknpu2/fetch.sh` (the
header and the runtime are Rockchip's and are not committed; see
[`vendor/rknpu2/README.md`](vendor/rknpu2/README.md)) — but the NPU feature
(`--features rknn`) links an aarch64 `.so` and will not.

## What is tested, and what cannot be

Everything above the two native libraries is tested on a development machine:

- **the observation layout**, against the C++ controller's own test vector
  (`tests/test_observation.cpp`): terms `{base_ang_vel, base_pose, actions}` at
  depth 2 give dim 14, laid out `[ang_vel x2][base_pose x2][actions x2]`, index 0
  the oldest. Pinning both implementations to the same vector is worth more than
  either matching the other's prose.
- **the FFI struct layouts**, against `vendor/rknpu2/include/rknn_api.h` itself.
  `build.rs` compiles a C probe that includes the header and links nothing, so
  `sizeof(rknn_tensor_attr)` and `offsetof(n_elems)` are checked by the C
  compiler on any host. Verified against a deliberately broken version: shortening
  `dims[16]` to `dims[15]` fails the test (372 vs 376). The DDS types need no such
  probe — bindgen bakes layout assertions into the generated file, so drift there
  is a compile error.
- joint-name resolution, the action transform, gain and torque clamping, and the
  rate-invariance of the output filter.

**What cannot be tested off the board**: `rknn_run` itself, and DDS actually
pairing with the robot's publishers. `librknnrt.so` is an aarch64 library talking
to real NPU hardware. A host build compiles `StubEngine` instead, which returns a
zero action — decoding to the home pose, which is safe for every mode but a
residual on a recording, and that one is barred — and reports `is_stub()` so the
loop can refuse to call that "running".

## The gait clock zeroes while standing

If a policy's `gait_phase` term carries `params.command_threshold`, the clock is
multiplied by `‖[lin_vel_x, lin_vel_y, yaw_rate]‖ > threshold`, so standing shows
the policy `(0, 0)`.

**This is not cosmetic, and the C++ controller does not do it.** The policy is
trained that way — zero magnitude is a value no phase can produce, i.e. an
unambiguous "hold still". A free-running unit-magnitude clock was measured during
training to make the robot step in place while commanded to stand: roll-rate
autocorrelation peaked at the clock frequency, and 5.17 of 6 feet were on the
ground.

The failure is invisible to every dimension check: gated or not, `gait_phase` is
2 wide and the observation is the same length. Only the values differ, and only
while standing.

## When the cadence clock moves is the contract's to say

A clock whose tempo follows the command is advanced once a tick, and training has
done that in two orders. `params.advance: "after_frame"` builds the frame with
the phase as it stands and advances it afterwards by that frame's command -- the
order `jumper.posture` trains in since its clock became one object. A contract
without `advance` is an export from before, trained with each frame advanced by
its own command first; `Advance` in `src/obs.rs` keeps that order for it rather
than retiring it. The two differ by `(f_now - f_at_entry) × dt` once the tempo
has moved, up to 6.4° on this robot, and nothing else in a contract tells them
apart.

It is as invisible as the gate. `play --fsm --fsm-diff` -- which compared mjlab's
observation with the controller's, and went with `--fsm` when `play --app`
replaced it on 2026-09-28 -- caught it only once the command's tempo had
changed, and `play`'s own commands mostly sat at the tempo ceiling, where both
orders agree.

## Adapting this to another robot

`dds.rs` is the part that changes; it is the only module that knows the robot's
message types. Everything else is driven by the policy's own `layout.json`.

Two things worth keeping whatever the hardware:

- **Joint orders are resolved by name, never by index.** On this robot the wire
  carries 22 and each policy takes the subset it trained on: the gaits and
  `jumper.jump` observe and drive 20, `jumper.dance` all 22. For the 20-joint ones
  index pairing is wrong from the fifth joint on, and nothing raises.
- **The home pose must cover every joint on the wire**, including the ones the
  policy never drives. They take part in the robot's contact geometry; leaving
  them where they happen to be changes what the feet are standing on.
