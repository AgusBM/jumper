//! What every control of the pad and every keystroke does, as data.
//!
//! A host that draws the controls -- a browser showing a person how to drive
//! the bundle it loaded, the manual the bundler writes -- needs, per control:
//! what it does in each mode, and which modes it switches into. All of that is
//! already here, split across the bundle's `[[fsm.button]]` and each mode's own
//! controls block, and a page that took the contracts and the TOML apart itself
//! would be a second reading of the bindings -- the one that goes stale.
//!
//! The pad and the keyboard are two lists, because they are two paths (since
//! 2026-09-29): a key is shown as what it does, never as the pad button it once
//! stood in for. The pad's list has one entry per control the dictionary lists,
//! used or not, so a host can draw the whole pad and grey out what does
//! nothing; the keyboard's has one entry per keystroke something uses.
//!
//! ```json
//! { "modes": ["claw_left", "locomotion", ...],
//!   "controls": [
//!     { "control": "Ly",
//!       "does": { "locomotion": [{ "kind": "axis", "axis": "lin_vel_x", "sign": -1.0,
//!                                  "travel": [0.0, 1.0], "shifted": false, "moved": false }] },
//!       "switches": [] },
//!     { "control": "A",
//!       "does": {},
//!       "switches": [{ "binding": "jump", "enters": "jump", "leaves": [], "event": false,
//!                      "on": "toggle", "with": null, "from": ["locomotion"] }] } ],
//!   "keyboard": [
//!     { "stroke": "shift+key_j", "modifier": "shift", "key": "key_j",
//!       "codes": { "modifier": ["ShiftLeft", "ShiftRight"], "key": ["KeyJ"] },
//!       "does": { "locomotion": [{ "kind": "axis", "axis": "twist", "dir": "+" }] },
//!       "switches": [] },
//!     { "stroke": "ctrl+key_1", "modifier": "ctrl", "key": "key_1",
//!       "codes": { "modifier": ["ControlLeft", "ControlRight"], "key": ["Digit1"] },
//!       "does": {},
//!       "switches": [{ "binding": "dance_crab", "enters": "dance_crab", ... }] } ] }
//! ```
//!
//! `does` entries on the pad are `axis` (a command axis the control drives,
//! signed, over a stretch of its travel, on the shifted layer or not, `moved`
//! for an axis it moves rather than places), `task` (a control the task keeps,
//! by the task's name for it, with its words), `release`, `shift` and `reset`.
//! A `task` entry is listed in a mode only if the mode answers it: a mode whose
//! hook reads some of its task's controls (`ModeHook::reads`) lists those. The
//! claw carried on the left answers `claw_left`; its task declares
//! `claw_right` as well, for the mode carrying it on the right, and a manual
//! that listed RT in the left one would promise a claw that is not there.
//! On the keyboard they are `axis` (one direction of an axis, `+` its positive as
//! the axis words it), `task` and `release`. A switch lists what its rule
//! `enters`, or the latches it `leaves`, its gesture, its modifier and the modes
//! it may be pressed `from` (`null`: any). A keystroke's `modifier` is `null`
//! for a key on its own; a modifier bound on its own is its `key`, both codes.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::config::{ButtonBinding, Enter, FsmConfig, When};
use crate::operator::Operators;
use crate::vocabulary::{self, dpad_buttons, key_codes, pad_axes, pad_buttons};

/// The browser codes a key, or a modifier's two keys, arrive under.
fn codes(name: &str) -> Vec<&'static str> {
    if vocabulary::is_modifier(name) {
        vocabulary::modifier_keys(name).into_iter().filter_map(|k| key_codes(k).map(|c| c.1)).collect()
    } else {
        key_codes(name).map(|c| c.1).into_iter().collect()
    }
}

/// The pad and the keyboard, control by control, as the bundle and its modes
/// drive them. `reads` is what each mode's hook answers, by mode
/// (`Controller::hook_reads`); a mode it does not name answers every task
/// control its task declares.
pub fn pad_guide(cfg: &FsmConfig, operators: &Operators, reads: &BTreeMap<String, Vec<String>>) -> Value {
    let answers = |mode: &str, control: &str| {
        reads.get(mode).is_none_or(|names| names.iter().any(|n| n == control))
    };
    let modes: Vec<&str> = cfg
        .states
        .iter()
        .filter(|s| !s.model.is_empty())
        .map(|s| s.name.as_str())
        .collect();
    // Which state a binding's rule enters. A binding's name is what rules read
    // (`button:<name>`), not necessarily the state they enter.
    let enters: BTreeMap<&str, &str> = cfg
        .rules
        .iter()
        .filter_map(|r| match (&r.when, &r.enter) {
            (When::Button(b), Enter::State(s)) => Some((b.as_str(), s.as_str())),
            _ => None,
        })
        .collect();

    let mut controls = Vec::new();
    for control in pad_axes().into_iter().chain(pad_buttons()).chain(dpad_buttons()) {
        let mut does = serde_json::Map::new();
        for (mode, op) in operators.iter() {
            let spec = op.spec();
            let pad = &spec.devices.gamepad;
            let moved = |axis: &str| {
                spec.command.iter().flat_map(|c| &c.axes).any(|a| a.name == axis && a.integrate_s.is_some())
            };
            let mut here = Vec::new();
            for (axis, bound) in &pad.axes {
                for b in bound.parts().iter().filter(|b| b.source == control) {
                    here.push(json!({
                        "kind": "axis", "axis": axis, "sign": b.sign,
                        "travel": b.travel, "shifted": b.shifted, "moved": moved(axis),
                    }));
                }
            }
            for (name, on) in &pad.task {
                if on == control && answers(mode, name) {
                    here.push(json!({ "kind": "task", "control": name, "what": spec.task[name].does }));
                }
            }
            if pad.release_button == control {
                here.push(json!({ "kind": "release" }));
            }
            if pad.shift.as_ref().is_some_and(|s| s.button == control) {
                here.push(json!({ "kind": "shift" }));
            }
            if pad.reset.as_deref() == Some(control) {
                here.push(json!({ "kind": "reset" }));
            }
            if !here.is_empty() {
                does.insert(mode.to_string(), Value::Array(here));
            }
        }
        let switches: Vec<Value> = cfg
            .buttons
            .iter()
            .filter(|b| b.pad.as_deref() == Some(control))
            .map(|b| switch(b, &enters))
            .collect();
        controls.push(json!({ "control": control, "does": does, "switches": switches }));
    }

    // The keyboard: every keystroke a mode binds, and every key a switch is on
    // -- written with its modifier, `ctrl+key_1`, as the modes write theirs.
    let mut keyboard: BTreeMap<String, (serde_json::Map<String, Value>, Vec<Value>)> = BTreeMap::new();
    for (mode, op) in operators.iter() {
        let spec = op.spec();
        let kb = &spec.devices.keyboard;
        let mut add = |stroke: &str, what: Value| {
            let (does, _) = keyboard.entry(stroke.to_string()).or_default();
            does.entry(mode.to_string())
                .or_insert_with(|| Value::Array(Vec::new()))
                .as_array_mut()
                .expect("an array")
                .push(what);
        };
        for (axis, keys) in &kb.axes {
            for (dir, strokes) in [("+", &keys.plus), ("-", &keys.minus)] {
                for s in strokes.iter().flatten() {
                    // No `moved`: a key places whatever it drives, a moved
                    // axis included (`operator.rs`).
                    add(s, json!({ "kind": "axis", "axis": axis, "dir": dir }));
                }
            }
        }
        for s in &kb.release {
            add(s, json!({ "kind": "release" }));
        }
        for (name, strokes) in kb.task.iter().filter(|(name, _)| answers(mode, name)) {
            for s in strokes {
                add(s, json!({ "kind": "task", "control": name, "what": spec.task[name].does }));
            }
        }
    }
    for b in cfg.buttons.iter().filter(|b| b.key.is_some()) {
        let key = b.key.as_deref().unwrap_or_default();
        let stroke = b.with.as_deref().map_or(key.to_string(), |m| format!("{m}+{key}"));
        keyboard.entry(stroke).or_default().1.push(switch(b, &enters));
    }
    let keyboard: Vec<Value> = keyboard
        .into_iter()
        .map(|(stroke, (does, switches))| {
            let (modifier, key) = match vocabulary::keystroke(&stroke) {
                Ok((m, k)) => (m.map(str::to_string), k.to_string()),
                Err(_) => (None, stroke.clone()),
            };
            json!({
                "stroke": stroke,
                "modifier": modifier,
                "key": key,
                "codes": { "modifier": modifier.as_deref().map(codes).unwrap_or_default(), "key": codes(&key) },
                "does": does,
                "switches": switches,
            })
        })
        .collect();

    json!({ "modes": modes, "controls": controls, "keyboard": keyboard })
}

fn switch(b: &ButtonBinding, enters: &BTreeMap<&str, &str>) -> Value {
    let enters = if b.event || !b.leaves.is_empty() { None } else { enters.get(b.name.as_str()).copied() };
    json!({
        "binding": b.name,
        "enters": enters,
        "leaves": b.leaves,
        "event": b.event,
        "on": b.on.to_string(),
        "with": b.with,
        "from": b.from,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::Value;

    use crate::config::FsmConfig;
    use crate::operator::tests::{contract_ranged, contract_with, BLOCK, CLAW_BLOCK, CLAW_RANGES};
    use crate::operator::Operators;

    /// Two modes with their own controls, one without; a switch on a pad
    /// button and on a key, one behind a modifier on each device, a leave, and
    /// a key that is a switch in one mode and the task's in another.
    const CONFIG: &str = r#"
[fsm]
initial_state = "walk"
warm_start_ref = "walk"
safe_state = "safe"
tilt_limit = 0.8
state_timeout_ms = 100
command_timeout_ms = 500
mode_switch_ramp_s = 1.0
pose_reach_tol = 0.10
warm_start_duration_s = 2.0
ramp_kp = 0.15
ramp_kd = 0.01

[[fsm.button]]
name = "grab"
pad = "LB"
on = "toggle"

[[fsm.button]]
name = "grab"
key = "key_g"
on = "toggle"

[[fsm.button]]
name = "spin"
pad = "dpad_right"
with = "menu"
on = "toggle"

[[fsm.button]]
name = "spin"
key = "key_4"
with = "ctrl"
on = "toggle"
from = ["walk", "twirl"]

[[fsm.button]]
name = "hop"
key = "key_space"
on = "toggle"
from = ["walk"]

[[fsm.button]]
name = "out"
key = "ctrl"
on = "fall"
leaves = ["spin"]
from = ["twirl"]

[[fsm.state]]
name = "safe"
hold_current = true
kd = 0.01

[[fsm.state]]
name = "walk"
model = "walk.onnx"

[[fsm.state]]
name = "claw"
model = "claw.onnx"

[[fsm.state]]
name = "twirl"
model = "twirl.onnx"

[[fsm.rule]]
when = "feedback_stale"
enter = "safe"

[[fsm.rule]]
when = "in_state:safe"
enter = "@initial"

[[fsm.rule]]
when = "button:spin"
enter = "twirl"

[[fsm.rule]]
when = "button:hop"
enter = "twirl"

[[fsm.rule]]
when = "button:grab"
enter = "claw"

[[fsm.rule]]
when = "always"
enter = "walk"
"#;

    fn guide() -> Value {
        guide_reading(&BTreeMap::new())
    }

    fn guide_reading(reads: &BTreeMap<String, Vec<String>>) -> Value {
        let cfg = FsmConfig::parse(CONFIG).unwrap();
        let (walk, claw) = (contract_with(BLOCK), contract_ranged(CLAW_BLOCK, CLAW_RANGES));
        let ops = Operators::of([("walk", &walk), ("claw", &claw)].into_iter()).unwrap();
        super::pad_guide(&cfg, &ops, reads)
    }

    fn control<'a>(g: &'a Value, name: &str) -> &'a Value {
        g["controls"].as_array().unwrap().iter().find(|c| c["control"] == name).unwrap()
    }

    fn stroke<'a>(g: &'a Value, name: &str) -> &'a Value {
        g["keyboard"].as_array().unwrap().iter().find(|c| c["stroke"] == name)
            .unwrap_or_else(|| panic!("no keystroke {name}"))
    }

    /// The whole pad, once each and in the dictionary's order, whether the
    /// bundle uses a control or not: a host draws the pad from this, and a
    /// control missing from it is a button that is not drawn at all. And no
    /// key anywhere on it: the keyboard is its own list.
    #[test]
    fn every_control_the_dictionary_lists_is_there_once() {
        let g = guide();
        let names: Vec<&str> =
            g["controls"].as_array().unwrap().iter().map(|c| c["control"].as_str().unwrap()).collect();
        let expected: Vec<&str> = crate::vocabulary::pad_axes()
            .into_iter()
            .chain(crate::vocabulary::pad_buttons())
            .chain(crate::vocabulary::dpad_buttons())
            .collect();
        assert_eq!(names, expected);
        assert_eq!(g["modes"], serde_json::json!(["walk", "claw", "twirl"]));
        // The control group for "used": nothing binds X, and it says nothing.
        let x = control(&g, "X");
        assert_eq!(x["switches"].as_array().unwrap().len(), 0);
        assert_eq!(x["does"], serde_json::json!({}));
        assert!(g["controls"].as_array().unwrap().iter().all(|c| c.get("keys").is_none()));
    }

    /// What a control does is each mode's own, and a stick split along its
    /// travel or layered under a shift says so -- the posture mode's right
    /// stick is a twist, then a turn, and a roll with R3 held; the claw mode's
    /// is a turn over the whole travel. R3 is the posture mode's shift and its
    /// reset, and the height it layers is moved rather than placed.
    #[test]
    fn what_a_control_does_is_each_mode_s_own() {
        let g = guide();
        let rx = &control(&g, "Rx")["does"];
        let walk: Vec<(&str, &Value, bool)> = rx["walk"].as_array().unwrap().iter()
            .map(|d| (d["axis"].as_str().unwrap(), &d["travel"], d["shifted"].as_bool().unwrap()))
            .collect();
        assert_eq!(walk.len(), 3, "{walk:?}");
        assert!(walk.contains(&("twist", &serde_json::json!([0.0, 0.5, 0.5, 0.75]), false)), "{walk:?}");
        assert!(walk.contains(&("ang_vel_z", &serde_json::json!([0.5, 1.0]), false)), "{walk:?}");
        assert!(walk.contains(&("roll", &serde_json::json!([0.0, 1.0]), true)), "{walk:?}");
        assert_eq!(rx["claw"].as_array().unwrap().len(), 1);
        assert_eq!(rx["claw"][0]["axis"], "ang_vel_z");

        let ry = control(&g, "Ry")["does"]["walk"].as_array().unwrap().clone();
        assert!(ry.iter().any(|d| d["axis"] == "height" && d["moved"] == true), "{ry:?}");
        assert!(ry.iter().any(|d| d["axis"] == "pitch" && d["moved"] == false), "{ry:?}");

        let lt = &control(&g, "LT")["does"];
        assert!(lt.get("walk").is_none(), "the posture mode keeps no trigger");
        assert_eq!((&lt["claw"][0]["kind"], &lt["claw"][0]["control"], &lt["claw"][0]["what"]),
                   (&"task".into(), &"claw_left".into(), &"closes the left claw".into()));
        assert_eq!(control(&g, "B")["does"]["claw"][0]["kind"], "release");
        let r3: Vec<&Value> = control(&g, "R3")["does"]["walk"].as_array().unwrap().iter()
            .map(|d| &d["kind"]).collect();
        assert_eq!(r3, [&Value::from("shift"), &Value::from("reset")]);
        assert!(control(&g, "R3")["does"].get("claw").is_none(), "the claw block has no shift");
    }

    /// A mode lists the task controls its hook answers, and no others. The
    /// claw block declares both claws, one per side; a mode carrying the claw
    /// on the left reads `claw_left`, and RT -- and Space's second meaning --
    /// would otherwise be written into its manual as a claw that is not there.
    /// Without `reads` both are listed, which is the control group: the filter
    /// is what takes RT out, not the block.
    #[test]
    fn a_mode_lists_only_the_task_controls_its_hook_answers() {
        let every = guide();
        assert_eq!(control(&every, "RT")["does"]["claw"][0]["control"], "claw_right");
        assert_eq!(stroke(&every, "key_space")["does"]["claw"].as_array().unwrap().len(), 2);

        let left: Vec<String> = ["claw_left", "arm_thumb_up", "arm_thumb_down", "arm_web_up"]
            .map(String::from).into();
        let g = guide_reading(&BTreeMap::from([("claw".to_string(), left)]));
        assert!(control(&g, "RT")["does"].get("claw").is_none(), "{}", control(&g, "RT"));
        assert_eq!(control(&g, "LT")["does"]["claw"][0]["control"], "claw_left");
        let space = &stroke(&g, "key_space")["does"]["claw"];
        assert_eq!(space.as_array().unwrap().len(), 1, "{space}");
        assert_eq!(space[0]["control"], "claw_left");
        assert_eq!(stroke(&g, "ctrl")["does"]["claw"][0]["control"], "arm_web_up");
        // A mode `reads` does not name keeps everything its task declares.
        assert_eq!(control(&g, "Ly")["does"]["walk"][0]["axis"], "lin_vel_x");
    }

    /// A keystroke is listed as what it does, in each mode: W is forward in
    /// both, Shift + J the posture mode's twist, Space and Ctrl the claw mode's
    /// claw and arm. "+" is the axis's own positive, so a page never has to
    /// work a sign out -- which is where forward ends up backwards.
    #[test]
    fn a_keystroke_says_what_it_does_in_each_mode() {
        let g = guide();
        let w = stroke(&g, "key_w");
        for mode in ["walk", "claw"] {
            assert_eq!((&w["does"][mode][0]["axis"], &w["does"][mode][0]["dir"]),
                       (&"lin_vel_x".into(), &"+".into()), "{mode}");
        }
        assert_eq!(w["codes"]["key"], serde_json::json!(["KeyW"]));
        let twist = stroke(&g, "shift+key_j");
        assert_eq!((&twist["modifier"], &twist["key"]), (&"shift".into(), &"key_j".into()));
        assert_eq!(twist["codes"]["modifier"], serde_json::json!(["ShiftLeft", "ShiftRight"]));
        assert_eq!(twist["does"]["walk"][0]["axis"], "twist");
        assert!(twist["does"].get("claw").is_none());
        assert!(stroke(&g, "key_n")["does"]["walk"][0].get("moved").is_none(), "a key places the height");
        let ctrl = stroke(&g, "ctrl");
        assert_eq!(ctrl["codes"]["key"], serde_json::json!(["ControlLeft", "ControlRight"]));
        assert_eq!(ctrl["does"]["claw"][0]["control"], "arm_web_up");
    }

    /// The switches, each device's under its own control: the state its rule
    /// enters or the latches it leaves, its modifier, and the modes it may be
    /// pressed from. Space is the jump's switch from `walk` and the claw's in
    /// `claw`, and is listed once with both, so neither reading hides the other.
    #[test]
    fn a_switch_names_what_it_enters_or_leaves_and_where_it_is_pressed() {
        let g = guide();
        let lb = &control(&g, "LB")["switches"][0];
        assert_eq!((&lb["enters"], &lb["with"], &lb["from"]), (&"claw".into(), &Value::Null, &Value::Null));
        assert_eq!(stroke(&g, "key_g")["switches"][0]["enters"], "claw");
        let right = &control(&g, "dpad_right")["switches"][0];
        assert_eq!((&right["enters"], &right["with"]), (&"twirl".into(), &"menu".into()));
        let four = &stroke(&g, "ctrl+key_4")["switches"][0];
        assert_eq!((&four["enters"], &four["from"]), (&"twirl".into(), &serde_json::json!(["walk", "twirl"])));
        let out = &stroke(&g, "ctrl")["switches"][0];
        assert_eq!((&out["enters"], &out["leaves"], &out["on"]),
                   (&Value::Null, &serde_json::json!(["spin"]), &"fall".into()));
        let space = stroke(&g, "key_space");
        assert_eq!(space["switches"][0]["from"], serde_json::json!(["walk"]));
        let claws: Vec<&Value> = space["does"]["claw"].as_array().unwrap().iter().map(|d| &d["control"]).collect();
        assert_eq!(claws, [&Value::from("claw_left"), &Value::from("claw_right")]);
    }
}
