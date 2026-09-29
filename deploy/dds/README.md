# dds — the wire interface to the robot

Everything needed to talk to the motor controller and the operator, and nothing else.
Copied from the `mbus` middleware (a CycloneDDS wrapper) rather than reimplemented,
because every file here is one half of a contract whose other half is already running on
the robot.

```
idl/                  the message types
  idl_common.idl        shared enums and headers, included by the rest
  idl_motor_control.idl MotorControl::Control (out) and ::State (in)
  idl_robot_control.idl RobotControl::Control — the operator's commands, interpreted
  idl_robot_control_raw.idl
                        RobotControlRaw::ControlRaw — the gamepad itself
  idl_imu.idl           Imu::ImuOutput — orientation and rates
config/
  mbus_qos.xml          QoS per topic
  mbus_config.xml       CycloneDDS runtime settings (interfaces, discovery)
topics.json           topic name + QoS profile + type, per channel
```

## The channels

| direction | topic | type | QoS profile |
|---|---|---|---|
| subscribe | `RobotControlRaw_ControlRaw` | `RobotControlRaw::ControlRaw` | `RobotControlRaw_ControlRaw` |
| subscribe | `mc/motor_state` | `MotorControl::State` | `MotorControl_State` |
| subscribe | `imu/output` | `Imu::ImuOutput` | `Imu_ImuOutput` |
| **publish** | `mc/motor_control` | `MotorControl::Control` | `MotorControl_Control` |

`robot_control` / `RobotControl::Control` is still here and is on the way out. It is
`control-agent`'s interpretation of the pad — sticks already scaled, buttons already
turned into an action enum — and taking it meant the three signs that turn a stick into
a body-frame command existed a **third** time, inside a program this repository cannot
see, agreed with the other two by nobody. `RobotControlRaw_ControlRaw` is the pad itself,
published by `control-pod-svc` at ~50 Hz, and the mapping comes from the policy's own
contract (`controls.yaml`, exported into `layout.json`) exactly as it does in `play` and
in the browser.

Copied from mbus `master` at `d9875988f065` (2026-09-28), every file from that one
revision; "Provenance" below says what that refresh changed and what it means for the
board.

`topics.json` is mbus's own file, trimmed to the channels in the table above plus
`robot_control`, which leaves when its type does. It is a record, not an input: the
topic name `dds.rs` opens is the one in the table above.

`MotorControl::Control` carries `MotorCmd { pos, kp, dq, kd, tau }` per joint — MIT
impedance control, so the position target and the gains go out together and the joint
torque is closed on the motor controller, not here.

**`imu/output` is not optional.** Until mbus `d9875988f065` the motor controller's
embedded IMU reported no orientation at all. `MotorControl::ImuData` now carries a
quaternion (`q`, with `quat_valid` and `imu_err_code`), but nothing here reads it, so a
controller that skips this topic still has no `projected_gravity` and no tilt angle —
both degrade to "perfectly level", silently. `projected_gravity` is an observation
term for every policy this repository trains.

## Why these files are copies and not descriptions

**The IDL is compiled, not read.** `idlc` turns each `.idl` into a
`dds_topic_descriptor_t`, and that descriptor determines the wire layout and the type hash
DDS matches on. Generating it from the same IDL with the same compiler the robot's side
used makes the two byte-compatible by construction. Retyping the structs in Rust and
hoping they line up is the alternative, and it fails at the first `@extensibility` or
padding difference — as a silently unpaired reader, not an error.

**QoS is matched, not negotiated.** A reader whose reliability or durability disagrees
with the writer's is not an error either: DDS simply never pairs them, and the topic stays
empty forever. `mbus_qos.xml` is therefore verbatim, trimmed to the six profiles here —
`common`, one per channel, and `RobotControl_Control`, which nothing here opens any more
and which has not been dropped yet — and the FSM **loads it at run time** through
CycloneDDS's QoS provider rather than constructing QoS in code — so what this side asks
for is literally the file the robot's side was configured with.

The profiles are not uniform, and one of them matters more than it looks:

| profile | reliability | history |
|---|---|---|
| `MotorControl_Control` / `_State`, `Imu_ImuOutput`, `RobotControlRaw_ControlRaw` | BEST_EFFORT | KEEP_LAST 1 |
| **`RobotControl_Control`** | **RELIABLE** | **KEEP_ALL** |

`RobotControl_Control` is what this table was written for. Those operator commands cross a
wireless link, and `mbus_config.xml` puts them on their own domain with retransmission
tuned for it. A reader built with CycloneDDS's defaults still *pairs* with that writer —
BEST_EFFORT requested against RELIABLE offered is legal — it just silently drops the
reliability the tuning exists to provide. That is why the profiles are loaded rather than
assumed, and why a test asserts what comes back.

The pad this controller actually opens is not on that footing.
`RobotControlRaw_ControlRaw` is BEST_EFFORT / KEEP_LAST 1 like the motor and IMU topics —
a 50 Hz snapshot where the newest sample is the only one worth having — and
`control-pod-svc` publishes it on the same domain 0 the controller defaults to. The
argument for loading the file rather than writing QoS in code is unchanged; which channel
is the expensive one to get wrong has moved.

**Three domains, and the channels are split across them.** `mbus_config.xml` defines
domain 0 (loopback only, no multicast — pure on-board IPC), domain 1 (shared memory
over loopback; its wlan0/eth0 lines have been commented out since mbus `09a57d1`), and
domain 2 (wlan0/eth0, SPDP-only multicast) for cross-machine traffic.
All four channels this FSM uses are on-board; `robot_control`, the one that comes over the
air on domain 2, is the one it stopped reading. The domain ids are configuration, not part
of the topic list — `DdsIo::open` takes one for the motor channels and one for the
operator channel, and `controller` passes 0 for both.

## Provenance

Every file here is from mbus master `d9875988f065` (2026-09-28), refreshed on 2026-09-29.
Before that they had arrived in three takes -- `ea627799ffee` (2026-08-23),
`5a934894bd9a` (2026-09-03) and `d375d737b77f` (2026-09-21) -- and each still matched
its recorded revision byte for byte when the refresh compared them. What the refresh
changed:

| file | change | on the wire |
|---|---|---|
| `idl_motor_control.idl` | `ImuData` gains `imu_temperature`, `q`, `quat_timestamp`, `quat_valid`, `imu_err_code`, `raw_count` | **`State` 800 → 832 bytes**, and it is `@extensibility(FINAL)` |
| `idl_common.idl` | `TopicType` gains `ROBOT_CONTROL_CONTROL_RAW` | none: no channel here carries a `TopicType` |
| `idl_robot_control.idl` | `ActionId` and `GripperPose` renamed after the key that fires them rather than the motion (`ACTION_JUMP` → `ACTION_A`, …), with values added to both | `robot_control` only, which nothing here opens |
| `idl_imu.idl` | adds `Imu::YuanjiOutput` | none: `ImuOutput` is unchanged |
| `mbus_config.xml` | domain 1 binds `lo` only | none: every channel here is on domain 0 |
| `topics.json` | gains `robot_control_raw` | none: `dds.rs` names its own topics |
| `mbus_qos.xml` | not touched: the six profiles kept here compare equal to upstream's | none |
| `idl_robot_control_raw.idl` | unchanged | none |

Quote the revision when refreshing. A copy without one is the thing that drifts; a copy
with one is a claim that can be checked:

```bash
git -C <mbus> diff d9875988f065 -- idl/ config/    # what has changed since
```

**Check this against the robot, not against the repository.** The `libmbus` built
into the kkos staging tree on one development machine dates from 2026-06-24 and
predates the `MotorControl::State` extra-register fields by a month, so its `State`
is a different type from the one here. Which build the board actually runs decides
whether the reader pairs at all -- and a mismatch does not raise, it leaves
`mc/motor_state` empty forever, which reads downstream as "no state, hold position".

The 2026-09-29 refresh is that case again. The `ImuData` fields reached mbus master
only in the 2026-09-28 merge, mbus's CHANGELOG has no release entry for them yet, and
the board last read (`deploy/README.md`) ran libmbus 2.4.0, from 2026-09-02. A
controller built from this tree pairs with an `mc_ctrl_driver` / `mc_forwarder` rebuilt
against `d9875988f065` or later, and with nothing older. `check_board.py` compares the
board's `idl_motor_control.h` with the one generated here and says which it is.

## Regenerating

`idlc` comes with CycloneDDS (`idlc <file>.idl` writes `<file>.c` and `<file>.h`). The
build compiles the generated C in, so nothing here is checked in pre-generated: a stale
descriptor beside a changed IDL is exactly the failure this avoids.

Verified on this repository's side: all five IDL files compile standalone with
`idlc` 0.11, every `topics.json` entry resolves to a QoS profile that is present and a
type that is defined, and a Rust round-trip over `MotorControl::Control` through
`libddsc` returns the sample it wrote, field for field. Rechecked at the 2026-09-29
refresh with `idlc` 11.0.1 on the development machine: `cargo test` in `deploy/fsm`
generates all five and passes, the descriptor-size and QoS tests included, and every
`topics.json` entry still resolves.

## If you are adapting this to your own robot

These are one robot's message definitions. Nothing above is a standard — a different
motor controller means a different `idl_motor_control.idl`, and the FSM's DDS layer is
the only part that should need to change with it. What is worth keeping whatever the
hardware is: the joint order comes from the policy's own `layout.json` and is resolved by
name against the robot's joint list, so the wire order and the policy's action order are
never assumed to agree.
