---
name: controls
description: Define what a gamepad or keyboard control means -- a task's stick, button and key bindings, an FSM mode switch on either device, or the button that starts a recorded motion -- with every word taken from the control dictionary. Use this whenever a control is being given a meaning: "A 按下起跳", "用 LB 切进去", "Ctrl + 1 进舞蹈", "把摇杆绑到转向", "add a button to switch into the dance", "which stick drives yaw", "Shift + J 扭身", writing or editing a `controls.yaml`, adding a mode's `pad` / `keys` switch or a `leave` to `deploy/manifests.json`, or binding a motion's `go`. Reach for it before hand-writing any of those files, because the failures are silent: a name the pad service does not publish parses fine and drives nothing, a sign flipped the wrong way walks the robot backwards while every screen looks correct, and one control bound in two files does two things on one press.
---

# Giving a control a meaning

## The one thing to understand first

**The words are fixed; the meanings are not.** `controller/vocabulary.json` is
the dictionary -- everything a pad or a keyboard can say, and nothing about what
any of it means. Ten buttons, four dpad directions, six axes, the keys -- the
numeric keypad and the letters and digits the tasks and bundles bind -- the three
modifiers (`ctrl`, `shift`, `alt`, each either of its two keys), and nine
gestures. It is read by three programs and written by none of them.

The keys are the **numeric keypad** and **keys asked for knowing what they
cost**: MuJoCo's viewer calls the user callback *in addition to* its own
shortcuts and binds every letter, so a binding on `W` walks the robot and flips
the scene to wireframe, with no way to intercept the second. The keypad is the
set it leaves alone. A key goes into the dictionary when a task or a bundle wants
it, and `_keys_note` records what each one also does. The jumper tasks lay the
keyboard out as the robot's operator guide, Control-agent 3.1, does (as revised
on 2026-09-29): W A S D and the arrows walk, I K pitch, J L turn, H ; twist, U O
roll, N M hold the high and the low stance, Esc lets go; the claw modes close the
claw on Space and hold the arm out on Shift, Alt and Ctrl; and the jumper bundle
switches modes on Space, V, B, the digits 1 to 4 (main row and keypad) and Ctrl. That viewer reports a
key's **press alone** -- no auto-repeat, no release -- and `mjrl.viewer.keys`
hooks the window's GLFW key callback to hand every reader both edges, so a key is
held from its press to its release, as in a browser. Until 2026-09-29 `play`
inferred the release from repeats that never arrived, and every key let go 0.75 s
in. Each key carries the two codes a host might see it under -- GLFW for a MuJoCo
window, `KeyboardEvent.code` for a browser -- because the translation is the
host's job and it used to be stated without being given.

```bash
python -m controller --vocabulary
```

Do not retype that list into a file. Do not paraphrase it from memory. It had a
copy in three places for a day, and adding a word meant remembering two other
files -- forget one and a name reads as supported and is refused at build, or
worse, the other way round.

**The pad and the keyboard are two paths.** Each binds its own inputs straight
to what they do -- a direction of a command axis, a control the task keeps, a
mode switch, the release -- and neither names a control of the other. Until
2026-09-29 the keyboard was a *virtual pad*, each key a pad control read through
the pad's mapping, and a bundle's `keyboard` table made keys *be* pad buttons.
Neither exists any more: a key is never "the pad's A".

A **keystroke** is how a keyboard binding names what is pressed: a key
(`key_j`), a modifier on its own (`ctrl`), or a modifier held with a key
(`shift+key_j`). J and Shift + J are two keystrokes and never both act: while
Shift is held, J answers only Shift + J. No jumper task binds one now; the twist
was Shift + J L until the doc moved it to H ; on 2026-09-29.

A **control** is a pad button or a key plus a **gesture**: `rise` (the tick it
goes down), `fall` (the tick it comes up), `hold` (every tick it is down -- a
dead-man switch, not a mode), `toggle` (down until pressed again). So "press A to
crouch, release A to jump" is one button bound twice, not one binding with two
jobs.

Or a **click count**: `single`, `double`, `triple`, `quadruple`, `quintuple` --
presses of one control (the same pad button, or the same key under the same
modifier), each within `[fsm] click_window_ms` of the last. A click switches a
mode as `toggle` does, and fires once as an `event`. The count fires at once when
no longer click is bound on that control, and after the window otherwise -- so
`single` beside a `double` waits the window out, and that wait is the price of
sharing. A control that counts clicks takes no `rise`/`fall`/`hold`/`toggle`, a
control used as a `with` modifier takes no click, and a d-pad chord a task shares
stays `toggle`; the crate refuses each. The window is required once any click is
bound, and it is the cascade file's (`deploy/jumper.controller.toml`: 300 ms).
The keyboard counts its own: a key and a pad button are two controls, so one
press of Space and one of A are not a double.

## Which file -- there are exactly two, and their sum is the whole

| what you are defining | file | example |
|---|---|---|
| what the sticks and keys **do** while a task is driving | `tasks/<task>/controls.yaml` | left stick forward, or W, is `lin_vel_x` |
| which control **switches into** a mode, leaves one, or fires an event inside one | `deploy/manifests.json` → a mode's `pad` / `keys`, the bundle's `leave` and `buttons` | `A` or Space toggles into `jump` from `locomotion`; `menu` held and `dpad_up`, or Ctrl + 1, into `dance_crab` |

**Neither may say the other's half.** A manifest does not bind a stick to an
axis; a task does not name a mode. There is no third place: the deployed
controller's whole behaviour for a pad and a keyboard is the sum of those two,
and the FSM never reads a contract's controls block nor a contract the FSM's.

That last fact is why one control can quietly be two things -- a manifest
switching on `B` and a task using `B` as its `release_button` are each correct
alone, and one press does both; so is a key switch on Ctrl in a mode whose
keyboard holds the arm out on Ctrl. You do not have to check by reading:
`Bundle::open` refuses it (`operator::check_switches`), so `scripts/deploy.py`
refuses it on every build. If the build printed a bundle, the two halves do not
collide.

## The task side -- run the script, do not write the file

```bash
python3 .claude/skills/controls/scripts/write_controls.py --task jumper.<name>
```

It reads the dictionary, refuses a name that is not in it, and writes the file:
one velocity command, a stick per axis, and a keyboard of its own in the layout
every jumper task shares -- W S and the arrows forward and back, A D and the
arrows left and right, J L turning, B the release. Each key is bound to one
direction of one axis, `"+"` being the axis's own positive direction as its
`positive:` words it, so the keyboard has no signs of its own to get wrong. A key
held pushes its direction further the longer it is held -- full after two seconds
(`full_after_s`) -- and is back at rest the moment it comes up; on an axis with
`integrate_s` it is full at once, a speed. What the script
cannot decide it refuses rather than defaults:

```bash
... --bind lin_vel_x=Ly- --bind ang_vel_z=Rx+ --release LB
```

A `--bind` with no sign is refused. **A sign is the one value in that file whose
wrong answer looks exactly like the right one** -- nothing downstream can catch
it, and the symptom is a robot that walks backwards with every screen normal.

With no `--bind`, each axis takes the evdev default the four existing tasks use:
`lin_vel_x` from `-Ly`, `lin_vel_y` from `-Lx`, `ang_vel_z` from `-Rx`. Fewer
than three axes is `--axes lin_vel_x,ang_vel_z`. `--release` is the pad's; the
keyboard's release is B whatever the pad's is.

When somebody named controls in words ("用 LB 切进去", "A 按下起跳", "Ctrl + 1
跳舞"), write them down verbatim first and pass them through. Do not paraphrase a
control into a different one that seems equivalent.

## Beyond the plain case -- edit what the script wrote

A task a person drives with a second command, or whose pad splits a stick, puts
a binding on a layer a button brings in, sums two controls, binds a modifier
chord, moves an axis rather than places it, or keeps controls for itself, edits
the file the script wrote. `tasks/jumper/posture/controls.yaml` does most of it
-- the walk and the posture from one pad and one keyboard -- and
`tasks/jumper/five_foot/controls.yaml` the task's own controls; copy their shape,
not their bindings. There is one schema, `schema_version: 2`: the plain file is
the case of the full one with one command and nothing split.

`load_controls` is the check, and it refuses every way below of going wrong
quietly. `tests/test_controls.py` runs it on the existing tasks' files only, so
after editing any other file load that file directly:

```bash
python -c "from pathlib import Path; from tasks.jumper.common.mdp.controls import load_controls; load_controls(Path('tasks/jumper/<name>/controls.yaml'))"
```

- `command:` is a **list**, one entry per command term; `term:` is its key in
  `cfg.commands`. An axis whose centred value is not zero names the config field
  that holds it -- `rest: neutral_height` -- because zero height is a body on
  the floor. The exported contract carries the number.
- On the pad, one control can carry several axes. `travel: [0.0, 0.5]` takes part
  of a stick's travel; `[0.0, 0.5, 0.5, 0.75]` also gives it back, falling to
  zero over the last two numbers (the posture twist, unwinding as the turn
  climbs); `shifted: true` puts a binding on the layer that
  `shift: {button: R3, gesture: hold}` brings in while R3 is held (`toggle`
  keeps it in from one click to the next); a list sums controls, clamped. Two
  bindings climbing over the same stretch of one control in one layer are
  refused; a fall may share its stretch with the next binding's climb, which is
  how one hands the stick to the other.
- The keyboard is `scheme: keys`. `axes:` names **every** command axis, each as
  `{"+": [keystrokes], "-": [keystrokes]}` or as `{unbound: "<why>"}` --
  jumper.five_foot's twist, because Shift holds the arm out there. `release:`
  lists the keystrokes that let go of everything. Refused: an axis left out
  without saying why; one keystroke on two things among the axis directions and
  the release; a modifier bound on its own anywhere in the block while it also
  modifies a key there (holding it for the one would do it while reaching for the
  other); a `full_after_s` that is not a positive number of seconds; and the
  notched keyboard's retired `step` and `centre`.
- `integrate_s: <seconds>` on an axis makes it **moved rather than placed**: a
  deflection is a speed that carries it from its rest to either end of its range
  in that many seconds, and let go it stays -- jumper.posture's height, since
  2026-09-29. `reset: <button>` on the pad names the button whose tap -- pressed
  and let go with no stick it could be reaching for moved -- puts every such axis
  back at rest; it may be the shift button (posture's R3 is both), never the
  release, and never in a file with no moved axis.
- `shift` is `hold` or `toggle`. `hold` was refused while a key through
  MuJoCo's viewer reported no release; it reports both edges since 2026-09-29,
  and jumper.posture and jumper.five_foot hold R3 since that day.
- A top-level `task:` keeps controls for the task itself rather than for a
  command, by name: each `{kind: amount | press, does: <words>}` -- an `amount` is
  0 to 1 as far as it is held, a `press` 1 while it is -- and each device binds
  every one of them under its own `task:`. On the pad an `amount` goes on an axis
  (a trigger, as five_foot's claws) and a `press` on a d-pad direction (as its arm
  presets), because the ten buttons are the bundle's switches and the operator's
  release; no command, release, shift or reset may use the same pad control. On
  the keyboard each is a list of keystrokes, and two task controls may share one
  -- both claws close on Space, since a mode reads only its own side's. The
  task's own code answers them by name -- a term of its own in `play --task`
  (`Operator.task_control`), its `deploy/lib.rs` hook on the robot, in a browser
  and in `play --app` -- and the controller refuses a hook that reads a name the
  file does not declare, or a mode whose file declares some and that has no hook.
  The manual lists, per mode, only the ones its hook reads.
- Use `gesture:`, not `on:`, as the key. YAML reads a bare `on` as `true`.

The same file drives every host: `tasks/jumper/common/mdp/operator.py` in
`play --task`, and `deploy/fsm/src/operator.rs` -- as the contract's
`operator_controller/2` -- on the robot, in a browser and in `play --app`. A
page hosting a bundle hands over raw keys, pad values and simulator signals and
decides nothing; see the bundle's own README.

**Each mode reads the operator through its own task's file**, at its own trained
ranges. Different policies take different controls: `jumper.posture`'s R3 and
right stick move the body's height where `jumper.five_foot`'s tip the nose, and
five_foot's triggers close its claw, in one bundle, and nothing requires two
tasks' files to agree (`deploy/fsm/src/operator.rs`, `Operators`). Where two
tasks *should* mean a stick the same way, say so in both files, each naming the
other. Do not add a test for it: tests are for the framework, never for a task
(`CLAUDE.md`, "Tests are for the framework").

## The FSM side -- a name, then a switch per device

`deploy/manifests.json` (`kk-deploy-manifests/1`) owns the modes, the keys and
the buttons. Beside its `task` and `policy` (and a `hook` where its task has one),
a mode's entry carries a switch per device, both optional:

```json
"jump": {
  "task": "jumper.jump", "policy": "tasks/jumper/jump/out/<dir>",
  "pad":  {"button": "A", "on": "toggle", "from": ["locomotion"]},
  "keys": {"key": ["key_space"], "on": "toggle", "from": ["locomotion"]}
}
```

- `pad`: `{button, on, with, from}`. `button` is a pad button or a `dpad_*`
  direction; `with` a pad button that must be held -- `"with": "menu"` plus
  `"button": "dpad_up"` is "Menu and up".
- `keys`: `{key, on, with, from}`. `key` is a **list** -- one binding per key,
  `["key_1", "keypad_1"]` for either row -- and may name a modifier used as a key
  (`"ctrl"`); `with` is a modifier, `ctrl`, `shift` or `alt`, either of its two
  keys.
- `on` defaults to `toggle`, which is what switching into a mode means -- you
  press it and stay there.
- `from` lists the modes the switch may be pressed in; absent is any. It gates
  **entering** only: a switch pressed again from inside the mode it latched still
  leaves it (A pressed in the jump ends the jump, though `from` names only
  `locomotion`).

Both switches become `[[fsm.button]]` entries named after the mode, and entries
with one name share one latch: either device switches the mode on, either
switches it off. Until 2026-09-29 a mode carried `button`, `key`, `on` and `with`
itself, one gesture for both devices, and the bundle a `keyboard` table.

A modifier is **consumed** by what it modifies, so a `menu` bound to something of
its own stays quiet until released; and while a modifier is held, a binding
without it on the same control stays quiet -- Menu + up is the dance and never
also the gesture on up alone, Ctrl + 1 never also the gesture on 1.

**A switch may not be pressed in a mode whose own controls use its control** --
on the pad a button, stick, shift, reset or release the mode's block reads, on
the keyboard a key or modifier its keyboard block uses. One press would do both,
and the crate refuses the pair (`operator::check_switches`). Fix it with another
control, or with `from`: list only the modes where the switch should be live. The
jumper bundle's Space is the jump from `locomotion` and the claw in the claw
modes; its keyboard dances (Ctrl + a digit) leave the claw modes out, because Ctrl
holds the arm out there.

The one sharing the pad allows is a **chord that leaves the mode on the press**:
`menu` + up is a dance while up alone, in a claw mode, holds the arm out. The
bundler checks it (`deploy/fsm/src/config.rs::FsmConfig::chord_leaves_first`):
the switch has to carry `with`, be a `toggle` (a `rise` is true for one tick, so
the cascade leaves, comes back and reads the direction still down as a press),
and have its rule ahead of the one entering the mode, into another. Anything else
is one press doing both. The keyboard has no such exception.

**Leaving.** A bundle-level `leave` list names moments that let go of modes, after
which the cascade falls to its default:

```json
"leave": [{"name": "leave_motion", "leaves": ["dance_crab", "..."],
           "pad":  {"button": "menu", "on": "fall", "from": ["dance_crab", "..."]},
           "keys": {"key": ["ctrl"], "on": "fall", "from": ["dance_crab", "..."]}}]
```

A leave is `rise`, `fall` or a click, never a `toggle`, a `hold` or an event;
no rule may read it, and each name it leaves must latch. The jumper bundle's Menu
(Ctrl) let go on its own leaves any dance or gesture -- `fall`, so the chord that
chose a dance, having spent the modifier, does not leave on the way. Not the jump:
leaving a jump in the air lands the robot out of its landing.

**One mode at a time** is the cascade file's `[fsm] exclusive = true`: a latch
coming on releases every other, so the last switch pressed is the mode. The
jumper cascade sets it; without it, latches are a set and the rule order settles
two that are on together.

## A recorded motion's `go`

A reference-guided policy may be keyed to a moment a person picks *inside* its
mode, and then it needs an event, not a mode switch. `jumper.jump` was, until
2026-09-26 -- `jump_go` on A's release -- and now starts on entry (below).
**This is not a third mechanism.** It is a `[[fsm.button]]` like any other, and
the task refers to it **by name**:

- the manifest binds a control to a name in its `buttons` list, beside `modes`,
  which is where the bindings that are **not** mode switches go, with the same
  `pad` / `keys` switches: `{ "name": "<name>", "pad": {"button": "A", "on":
  "fall"} }`. The crate refuses both directions -- an event no mode reads, and a
  `go_event` no button declares;
- the task's `reference_contract` carries `go_event: "<that name>"`, which
  `scripts/export.py` writes into the contract's `reference` block;
- `control.rs` starts the motion's clock on the rising edge of that name.

Press-to-arm and release-to-go is the same name twice with two gestures, which
the crate already allows. Without a `go_event` the motion runs on a **timer**
from mode entry -- enough to watch it on a bench, not something a person can
use, and the host says so rather than letting it be inferred from a robot that
jumps two seconds after standing up.

A recording meant to start *with* the mode says so instead of leaving it to the
timer: `jumper.dance` is a piece of music, sets `starts_on_entry`, takes no `go`,
and ends -- after which the cascade hands back to the default mode. `jumper.jump`
does the same: switching into it is the whole of asking for a jump, and its
policy was trained starting past its go, never standing in the go frame waiting
for one. A contract
claiming both a `go_event` and `starts_on_entry` is refused, because one of the
two never happens and which one is not discoverable from a robot that did
nothing.

## Checking

```bash
python -m controller --vocabulary                    # the words
python -m pytest tests/test_controls.py -q           # the existing tasks' files parse and bind
python scripts/deploy.py --allow-incomplete          # the two halves do not collide (no docker)
```

## Failures worth recognising

| symptom | likely cause |
|---|---|
| a stick or a key does nothing, in play or on the robot | a name not in the dictionary. It is a closed list and a typo that parses is a binding that drives nothing |
| the robot walks backwards and everything else looks right | a sign on the pad, or a keystroke under the wrong one of `"+"` and `"-"`. Nothing downstream catches it; a hand on the pad does |
| one press does two things | the same control in both files, where the switch may be pressed. The bundler names both; change one, or give the switch a `from` that leaves that mode out -- do not widen anything to make it pass |
| a switch does nothing in one mode | its `from` leaves that mode out -- on purpose, as the jump's does in a claw mode, or not |
| a binding works on a bench and not on the robot | `view`. The pad has it, the service deliberately does not publish it, and `controller/xbox.py` does. Or it is a key: the board has no keyboard |
| a held button fires its rule every tick | `hold` where `rise` or `toggle` was meant |
| a modifier fires on its own when you reach for a direction | it is not declared as `with` on the binding it modifies |
| the motion starts by itself a couple of seconds in | no `go_event` and no `starts_on_entry`, so it is on the timer |

## When something does not match

If a control you want is not in the dictionary, it is not missing from the file
-- it is missing from the wire, or, for a key, from the dictionary, where it goes
in with what it also does in MuJoCo's viewer. `X` and `Y` carry an `unresolved`
note because two sources disagree about which physical button each is, and
pressing one is the only way to settle it. Record what you measure; do not pick
the one that makes the binding look right.
