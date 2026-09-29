// Load a bundle's web half the way a page does, drive it, and replay its
// reference vectors. Prints one JSON object on stdout; `tests/test_web_host.py`
// reads it.
//
// Node rather than a browser, and `initSync` with the bytes rather than the
// `fetch` the glue expects: what is being checked is the controller, not the
// network. Everything else -- the contracts, the trajectories, the robot block
// -- is assembled exactly as `BUNDLE_README.md` tells a page to assemble it, so
// this fails if that document goes stale.
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const dir = resolve(process.argv[2]);
const out = { dir };

try {
  // One bundle carries every host; a browser takes the runtime `bundle.json`
  // names for it.
  const manifest = JSON.parse(readFileSync(join(dir, "bundle.json"), "utf8"));
  const runtime = manifest.runtimes.web;
  const { initSync, WebFsm } = await import(
    pathToFileURL(join(dir, runtime.glue)).href
  );
  initSync({ module: readFileSync(join(dir, runtime.wasm)) });

  const contracts = {};
  const trajectories = {};
  for (const [mode, entry] of Object.entries(manifest.modes)) {
    const contract = JSON.parse(readFileSync(join(dir, entry.contract), "utf8"));
    contracts[mode] = contract;
    // A reference-guided mode is not runnable without the recording its
    // contract names. The board reads the file; a page is handed the text.
    if (contract.reference) {
      trajectories[mode] = readFileSync(join(dir, contract.reference.file), "utf8");
    }
  }
  out.modes = Object.keys(contracts).sort();
  out.joints = Object.values(contracts)[0].wire_joint_order.length;
  out.withTrajectory = Object.keys(trajectories).sort();

  const first = Object.values(contracts)[0];
  const wire = first.wire_joint_order;
  // The mode the cascade runs with nobody touching anything: the warm start's
  // target, or else where it starts. Not the first mode by name -- a bundle's
  // modes are written sorted, and `jumper`'s first is `claw_left`, whose home
  // holds an arm 0.5 rad from the walking stance (1.6 before the stow). Held
  // there, the robot never reached the pose the warm start waits for, and
  // nothing ever inferred.
  const config = readFileSync(join(dir, "controller.toml"), "utf8");
  const field = (key) => config.match(new RegExp(`^${key}\\s*=\\s*"([^"]+)"`, "m"))?.[1];
  const start = [field("warm_start_ref"), field("initial_state")].find((m) => m in contracts)
    ?? Object.keys(contracts)[0];
  out.startMode = start;
  const robot = {
    joint_names: wire,
    joint_pos_lo: wire.map((j) => first.joint_limits[j][0]),
    joint_pos_hi: wire.map((j) => first.joint_limits[j][1]),
    // One loop for every mode, so it has to be quick enough for the quickest.
    output_rate_hz: Math.max(
      ...Object.values(contracts).map((c) => c.control.control_hz),
    ),
  };
  const build = () =>
    new WebFsm(
      config,
      JSON.stringify(contracts),
      JSON.stringify(robot),
      0,
      JSON.stringify(trajectories),
    );

  const fsm = build();
  const n = wire.length;
  const zeros = new Float32Array(n);
  const home = new Float32Array(wire.map((j) => contracts[start].default_joint_pos[j]));
  const inferred = new Set();
  let infers = 0;
  for (let t = 0; t < 4000; t++) {
    const now = t * 1000;
    fsm.set_state(home, zeros, zeros, new Float32Array([1, 0, 0, 0]),
                  new Float32Array([0, 0, 0]), now);
    fsm.set_command(0.3, 0, 0, now);
    const mode = fsm.tick(now);
    if (mode !== undefined) {
      inferred.add(mode);
      infers += 1;
      fsm.resume(new Float32Array(contracts[mode].action.dim));
    }
  }
  out.infers = infers;
  out.inferredModes = [...inferred].sort();
  out.mode = fsm.mode();
  out.runningPolicy = fsm.is_running_policy();
  out.positions = Array.from(fsm.positions());

  // The raw pad: the page pushes sticks by the dictionary's names and the
  // *contract* turns them into a command, the same call the robot makes. What
  // this reads back is the `commands` block of the observation, which is where
  // that command lands -- so a wrong sign or a wrong axis is visible as a
  // number rather than as a robot that drives backwards.
  const isCommand = (n) =>
    n === "commands" || n === "velocity_commands" || n === "jump_command";
  // The contract of the mode that actually runs and observes a command, not
  // whichever one came first: a bundle's modes are written sorted, so "first"
  // here was `dance`, which observes no command at all, and is now `claw_left`,
  // which reads the pad through controls of its own. The same mistake once
  // picked the gait clock off the wrong contract.
  const driven = [contracts[start], ...Object.values(contracts)].find((c) =>
    c.observation.terms.some((t) => isCommand(t.name)),
  );
  const cmdTerm = driven?.observation.terms.find((t) => isCommand(t.name));
  if (cmdTerm) {
    let at = 0;
    for (const t of driven.observation.terms) {
      if (isCommand(t.name)) break;
      at += t.dim;
    }
    const sample = (axis, value) => {
      const f = build();
      for (let t = 0; t < 400; t++) {
        const now = t * 1000;
        f.set_state(home, zeros, zeros, new Float32Array([1, 0, 0, 0]),
                    new Float32Array([0, 0, 0]), now);
        if (axis) f.setAxis(axis, value);
        f.padFrame(now);
        const mode = f.tick(now);
        if (mode !== undefined) f.resume(new Float32Array(contracts[mode].action.dim));
      }
      return Array.from(f.observation()).slice(at, at + cmdTerm.dim);
    };
    out.pad = {
      axes: Object.fromEntries(
        Object.entries(driven.controller?.devices?.gamepad?.axes ?? {}),
      ),
      centred: sample(null, 0),
      forward: sample("Ly", -1),
      back: sample("Ly", 1),
    };
  }

  // The keyboard and the page's own stop, for a bundle whose controller reads
  // raw input. A key held pushes its stick for as long as it is down -- full
  // after the block's `full_after_s` -- so the key that pushes the left stick
  // forward is read held half that, then held past it, then let up; and
  // `letGo`, the page's Stop, is read with the key down again, since that is
  // the key it has to let go of. Read from the same `commands` block as the pad,
  // so the two are compared in one unit.
  if (cmdTerm && JSON.parse(build().inputs()).rawInput) {
    let at = 0;
    for (const t of driven.observation.terms) {
      if (isCommand(t.name)) break;
      at += t.dim;
    }
    const f = build();
    let tick = 0;
    const run = (ticks, act) => {
      for (let i = 0; i < ticks; i++, tick++) {
        const now = tick * 1000;
        f.set_state(home, zeros, zeros, new Float32Array([1, 0, 0, 0]),
                    new Float32Array([0, 0, 0]), now);
        act?.(now);
        f.set_command(0, 0, 0, now);
        const mode = f.tick(now);
        if (mode !== undefined) f.resume(new Float32Array(contracts[mode].action.dim));
      }
      return Array.from(f.observation()).slice(at, at + cmdTerm.dim);
    };
    const inputs = JSON.parse(f.inputs());
    const forwardKey = inputs.keys.find((k) => k.name === "key_w")?.code;
    const fullAfterS = driven.controller?.devices?.keyboard?.full_after_s;
    const halfTicks = Math.round((fullAfterS * 1000) / 2);
    let bound = false;
    const once = (act) => {
      let done = false;
      return (now) => {
        if (!done) act(now);
        done = true;
      };
    };
    run(100);
    const half = forwardKey
      ? run(halfTicks, once((now) => {
          bound = f.setKeyCode(forwardKey, true, false, now);
        }))
      : null;
    // Past `full_after_s`, and with the browser's auto-repeat arriving as it
    // would: the same key still down, which must not start the hold over.
    const held = forwardKey
      ? run(halfTicks + 200, (now) => {
          if (tick % 33 === 0) f.setKeyCode(forwardKey, true, true, now);
        })
      : null;
    const letUp = forwardKey
      ? run(50, once((now) => f.setKeyCode(forwardKey, false, false, now)))
      : null;
    if (forwardKey) run(halfTicks, once((now) => f.setKeyCode(forwardKey, true, false, now)));
    let letGo = null;
    const released = run(50, (now) => {
      if (letGo === null) letGo = f.letGo(now);
    });
    out.keys = {
      inputs, forwardKey, fullAfterS, bound, half, held, letUp,
      letGo, released,
      // An unbound code is handed over too, and says so.
      unbound: f.setKeyCode("F13", true, false, tick * 1000),
    };
  }

  // Where a motion's clock sits in the observation: the widths of the terms
  // before it. This summed every *other* term, which put it at the last index
  // -- the far end of `ref_future`, which happens to move with the clock too,
  // so the check passed while reading the wrong number.
  const isPhase = (n) => n === "jump_phase" || n === "clip_phase";
  const phaseAt = (contract) => {
    let at = 0;
    for (const t of contract.observation.terms) {
      if (isPhase(t.name)) return at;
      at += t.dim;
    }
    return at;
  };

  // A recorded motion, started the way an operator starts one: press the
  // button that reaches the mode, then the one its contract names as the `go`.
  // What this reads back is the motion's own clock, because that is the only
  // thing that distinguishes a `go` that arrived from one that did not -- the
  // mode is entered either way, the policy infers either way, and the robot
  // simply stands there.
  const goMode = Object.entries(contracts).find(
    ([, c]) => c.reference && c.reference.go_event,
  );
  if (goMode) {
    const [name, contract] = goMode;
    const binds = JSON.parse(build().bindings());
    const enter = binds.find((b) => b.name === name);
    const go = binds.find((b) => b.name === contract.reference.go_event);
    const at = phaseAt(contract);
    const phaseTerm = contract.observation.terms.find((t) => isPhase(t.name));
    if (enter?.pad && go?.pad && phaseTerm) {
      const f = build();
      const hz = robot.output_rate_hz;
      const phases = new Set();
      let tick = 0;
      const drive = (seconds, act) => {
        for (let i = 0; i < Math.round(seconds * hz); i++, tick++) {
          const now = (tick * 1e6) / hz;
          f.set_state(home, zeros, zeros, new Float32Array([1, 0, 0, 0]),
                      new Float32Array([0, 0, 0]), now);
          act(f);
          f.padFrame(now);
          const m = f.tick(now);
          if (m !== undefined) {
            f.resume(new Float32Array(contracts[m].action.dim));
            if (m === name) phases.add(Number(f.observation()[at].toFixed(4)));
          }
        }
      };
      drive(0.5, (x) => x.setPad(enter.pad, false));
      drive(0.1, (x) => x.setPad(enter.pad, true));     // enter the mode
      drive(1.5, (x) => x.setPad(enter.pad, false));    // and let its ramp finish
      const before = phases.size;
      drive(0.1, (x) => x.setPad(go.pad, true));        // arm
      drive(3.0, (x) => x.setPad(go.pad, false));       // and release: the `go`
      out.go = {
        mode: name, enter: enter.pad, button: go.pad, on: go.on,
        phasesBeforeGo: before, phasesAfterGo: phases.size,
      };
    }
  }

  // A recorded motion that starts on entering its mode -- `jumper.jump` since
  // 2026-09-26 -- started the way an operator starts it: press the control that
  // reaches the mode, modifier held where the binding names one, and nothing
  // else. Read back as above, through the motion's own clock, and then whether
  // the mode handed itself back once the recording ran out. The shortest such
  // recording, so it can end inside the window.
  const entryMode = Object.entries(contracts)
    .filter(([, c]) => c.reference && c.reference.starts_on_entry)
    .sort(([, a], [, b]) => a.reference.duration - b.reference.duration)[0];
  if (entryMode) {
    const [name, contract] = entryMode;
    const enter = JSON.parse(build().bindings()).find((b) => b.name === name);
    const at = phaseAt(contract);
    const phaseTerm = contract.observation.terms.find((t) => isPhase(t.name));
    if (enter?.pad && phaseTerm) {
      const f = build();
      const hz = robot.output_rate_hz;
      const phases = new Set();
      let tick = 0;
      const press = (down) => (x) => {
        if (enter.with) x.setPad(enter.with, down);
        x.setPad(enter.pad, down);
      };
      const drive = (seconds, act) => {
        for (let i = 0; i < Math.round(seconds * hz); i++, tick++) {
          const now = (tick * 1e6) / hz;
          f.set_state(home, zeros, zeros, new Float32Array([1, 0, 0, 0]),
                      new Float32Array([0, 0, 0]), now);
          act(f);
          f.padFrame(now);
          const m = f.tick(now);
          if (m !== undefined) {
            f.resume(new Float32Array(contracts[m].action.dim));
            if (m === name) phases.add(Number(f.observation()[at].toFixed(4)));
          }
        }
      };
      drive(0.5, press(false));
      const before = phases.size;
      drive(0.1, press(true));        // enter the mode, and that is all
      const window = contract.reference.duration + 1.5;
      drive(window, press(false));
      out.entry = {
        mode: name, enter: enter.pad, with: enter.with ?? null,
        durationS: contract.reference.duration, windowS: window,
        phasesBefore: before, phasesDuring: phases.size, modeAfter: f.mode(),
      };
    }
  }

  // The keyboard's own switches, pressed the way a page hands them over --
  // `KeyboardEvent.code`s, down and up, the modifier first -- for every one
  // that may be pressed in the mode the robot starts in (its `from`). Since
  // 2026-09-29 a key is bound to the switch it presses, not made a pad button
  // by a table. What is read back is the mode, a moment after the keys are let
  // up: a `toggle` latched. The control group is a chord's key with its
  // modifier left out, which must not reach the chord's mode.
  const initial = (config.match(/^\s*initial_state\s*=\s*"([^"]+)"/m) ?? [])[1];
  const modifierKey = { ctrl: "key_left_ctrl", shift: "key_left_shift", alt: "key_left_alt" };
  const codes = Object.fromEntries(
    JSON.parse(build().inputs()).keys.map((k) => [k.name, k.code]),
  );
  const pressable = (b) => b.on === "toggle" && !(b.leaves ?? []).length
    && (b.from == null || b.from.includes(initial));
  out.startMode = out.startMode ?? initial;
  out.keyboard = [];
  for (const b of JSON.parse(build().bindings())) {
    if (!b.key || !pressable(b) || modifierKey[b.key]) continue;
    const key = b.key;
    const mod = b.with ? modifierKey[b.with] : null;
    const press = (held) => {
      const f = build();
      const hz = robot.output_rate_hz;
      let tick = 0;
      const bound = [];
      const drive = (seconds, act) => {
        for (let i = 0; i < Math.round(seconds * hz); i++, tick++) {
          const now = (tick * 1e6) / hz;
          f.set_state(home, zeros, zeros, new Float32Array([1, 0, 0, 0]),
                      new Float32Array([0, 0, 0]), now);
          if (i === 0) act(f, now);
          const m = f.tick(now);
          if (m !== undefined) f.resume(new Float32Array(contracts[m].action.dim));
        }
      };
      drive(0.5, () => {});
      drive(0.1, (x, now) => held.forEach((k) => bound.push(x.setKeyCode(codes[k], true, false, now))));
      drive(0.2, (x, now) => [...held].reverse().forEach((k) => x.setKeyCode(codes[k], false, false, now)));
      return { mode: f.mode(), bound };
    };
    const held = mod ? [mod, key] : [key];
    const pressed = press(held);
    out.keyboard.push({
      mode: b.name, key, with: b.with ?? null, keys: held,
      codes: held.map((k) => codes[k] ?? null),
      bound: pressed.bound, modeAfter: pressed.mode,
      alone: mod ? press([key]).mode : null,
    });
  }

  // The same switches from a gamepad, through `setPad` as a page calls it --
  // modifier first, the d-pad's directions by their names. The dance is
  // `menu` + `dpad_right`, and `setPad` once took no d-pad name at all: the
  // chord could not be made from a pad in a browser, and `setPad` said
  // nothing binds it. The control group is the same button without its
  // modifier.
  out.padSwitch = [];
  for (const b of JSON.parse(build().bindings())) {
    if (!b.pad || !pressable(b)) continue;
    const press = (held) => {
      const f = build();
      const hz = robot.output_rate_hz;
      let tick = 0;
      const bound = [];
      const drive = (seconds, act) => {
        for (let i = 0; i < Math.round(seconds * hz); i++, tick++) {
          const now = (tick * 1e6) / hz;
          f.set_state(home, zeros, zeros, new Float32Array([1, 0, 0, 0]),
                      new Float32Array([0, 0, 0]), now);
          if (i === 0) act(f);
          f.padFrame(now);
          const m = f.tick(now);
          if (m !== undefined) f.resume(new Float32Array(contracts[m].action.dim));
        }
      };
      drive(0.5, () => {});
      drive(0.1, (x) => held.forEach((p) => bound.push(x.setPad(p, true))));
      drive(0.2, (x) => [...held].reverse().forEach((p) => x.setPad(p, false)));
      return { mode: f.mode(), bound };
    };
    const buttons = b.with ? [b.with, b.pad] : [b.pad];
    const pressed = press(buttons);
    out.padSwitch.push({
      mode: b.name, buttons, bound: pressed.bound, modeAfter: pressed.mode,
      alone: b.with ? press([b.pad]).mode : null,
    });
  }

  // The pad, as the controller describes it for a page to draw, beside the
  // keys it says it binds: the two have to be one set.
  {
    const f = build();
    if (typeof f.padGuide === "function") {
      out.guide = JSON.parse(f.padGuide());
      out.boundCodes = JSON.parse(f.inputs()).keys.map((k) => k.code).sort();
    }
  }

  // A fresh one: the replay drives somebody else's frames in and destroys the
  // state of whatever it runs on.
  out.reference = JSON.parse(
    build().checkReference(readFileSync(join(dir, "reference.json"), "utf8")),
  );
  out.ok = true;
} catch (e) {
  out.ok = false;
  out.error = String(e && e.message ? e.message : e);
}

process.stdout.write(JSON.stringify(out));
