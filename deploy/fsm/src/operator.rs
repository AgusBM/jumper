//! One operator, several commands, two devices: the contract's
//! `operator_controller/2`.
//!
//! The deployed side of a task's `controls.yaml`, and the only thing on any host
//! that turns a pad or a keyboard into a command. `jumper.posture` drives the walk
//! and the posture together; a locomotion task drives the walk alone. One reader
//! for both, because the second is the case of the first with one command and
//! nothing split.
//!
//! What the file says, and this implements, in `play`'s own words
//! (`tasks/jumper/common/mdp/operator.py` is the Python half):
//!
//! * **The pad and the keyboard are two paths.** Each binds its own inputs to
//!   what they do -- a direction of a command axis, a control the task answers
//!   itself, the release -- and neither refers to a control of the other. Until
//!   2026-09-29 the keyboard was a virtual pad: a key held pushed a stick of a
//!   pad nobody held, read through the real pad's mapping. That made the
//!   keyboard a copy of the pad's layout, which is not the layout a hand on a
//!   keyboard wants: Control-agent 3.1 (the robot's own operator guide) turns on
//!   J and L and twists on Shift + J and L, where the pad does both on one stick.
//! * **On the pad, a stick can carry more than one axis.** A binding may take
//!   part of a control's travel (`travel: [0.0, 0.5]` is full at half deflection
//!   and holds there; `[0.5, 1.0]` is silent until half and then climbs; `[0.0,
//!   0.5, 0.5, 0.75]` climbs over the first half and falls back to zero by three
//!   quarters, giving the control up to the binding climbing there), sit on the
//!   layer a `shift` button brings in -- for as long as it is held, or from one
//!   click to the next -- or be one of several controls summed into one axis.
//! * **On the keyboard, a keystroke pushes one direction of one axis**, further
//!   the longer it is held, full after `full_after_s`, and back to rest the
//!   moment it comes up -- every axis, a moved one included: keys place. A
//!   keystroke is a key (`key_j`), a modifier on its own (`shift`), or a
//!   modifier held with a key (`shift+key_j`); J and Shift + J are two
//!   keystrokes and never both act -- while a modifier is held, a key it has a
//!   chord with answers the chord and not itself.
//! * **A centred control is the axis's rest**, which is not always zero: zero
//!   height is a body on the floor. The exporter writes each rest as a number.
//! * **An axis may be moved rather than placed** (`integrate_s`), on the pad: a
//!   deflection is a speed, full deflection carries the axis from its rest to
//!   either end of its range in that many seconds, and let go it stays.
//!   `jumper.posture`'s height, since 2026-09-29. The pad's `reset` button,
//!   tapped -- pressed and let go with no stick it could be reaching for moved
//!   in between -- puts it back. The keyboard places the same axis, as it does
//!   every other (Control-agent 3.1: N and M are the high and the low stance,
//!   back to standing when let go), and where the pad moved it is kept while
//!   the keys drive and is the height again when the pad does.
//! * **The device touched last drives.** Touching the pad drops the keyboard's
//!   holds, so letting go of the pad does not revive a key command from a minute
//!   ago; a key pressed takes over from an idle pad.
//! * **A task may keep controls for itself**, by name -- `jumper.five_foot`'s
//!   `claw_left` is `LT` on the pad and Space on the keys. They come out of
//!   `task_controls` rather than `command`, for the task's hook (`hook.rs`) to
//!   answer on the robot: an `amount` as far as it is held, a `press` while it is.
//!
//! The mapping is transcribed from the Python and the two are held together by
//! the same cases: `tests/test_posture.py` and `tests/test_controls.py` there, the
//! tests at the bottom here. A disagreement between them is a robot that answers
//! the operator differently on the bench than in `play`, with every number on
//! both screens plausible.
//!
//! ## What a release means here
//!
//! In `play` the release hands the command back to the random sampler. A robot
//! has no sampler, so on every host of this crate it means the nearest thing
//! that is still true: let go of everything -- the keyboard's holds dropped, the
//! shift off, a moved axis back at its rest -- which leaves the robot at rest
//! until a control is touched again.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use crate::layout::{CommandRanges, Contract};
use crate::types::{Command, Pad, TaskControls};
use crate::vocabulary::{
    self, dpad_buttons, is_axis, is_button, is_modifier, pad_axes, pad_buttons,
};

/// Whether `name` is one of the four d-pad directions.
fn is_dpad(name: &str) -> bool {
    dpad_buttons().contains(&name)
}

/// A d-pad direction as the wire's `(dpad_x, dpad_y)`, up positive.
fn dpad_vector(name: &str) -> Option<(i8, i8)> {
    match name {
        "dpad_right" => Some((1, 0)),
        "dpad_left" => Some((-1, 0)),
        "dpad_up" => Some((0, 1)),
        "dpad_down" => Some((0, -1)),
        _ => None,
    }
}

/// `absolute`: the stick position *is* the command. The only scheme implemented;
/// an axis the controls should move rather than place says `integrate_s`, on the
/// axis, for both devices at once.
pub const GAMEPAD_SCHEME: &str = "absolute";

/// The deadzone contract this crate relies on: the pad service applies 0.15 and
/// **rescales**, so full deflection still reaches ±1, and a host that applied
/// its own on top would narrow what a person can ask for without being able to
/// see that it has.
pub const GAMEPAD_DEADZONE: &str = "device_reported_rescaled";

/// The notched keyboard's other half, retired on 2026-09-27: one key letting go
/// of every stick and trigger at once. Named only to refuse a contract that
/// still carries it.
pub const CENTRE: &str = "centre";

/// The gestures a shift implements: `hold`, its layer in for as long as the
/// button is held, and `toggle`, in from one click to the next. `rise` and
/// `fall` are one tick long, which a layer cannot be.
pub const SHIFT_GESTURES: [&str; 2] = ["hold", "toggle"];

/// The keyboard scheme implemented: each keystroke bound to what it does.
pub const KEYBOARD_SCHEME: &str = "keys";

/// What a task control is: `amount`, 0 to 1 as far as it is held -- a trigger,
/// or a key's ramp -- and `press`, 1 while it is held.
pub const TASK_KINDS: [&str; 2] = ["amount", "press"];

// ── The block, as the exporter writes it ────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct OperatorSpec {
    pub schema: String,
    /// One entry per command term, in the order the file lists them.
    pub command: Vec<CommandSpec>,
    /// The controls the task answers itself, by name: what each is and does.
    /// Empty for a task that keeps none.
    #[serde(default)]
    pub task: BTreeMap<String, TaskSpec>,
    pub devices: OperatorDevices,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TaskSpec {
    /// `amount` or `press` (`TASK_KINDS`).
    pub kind: String,
    /// What the task does with it, in the file's words; what the manual prints.
    pub does: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CommandSpec {
    /// The command term's key in the environment -- `twist`, `posture`.
    pub term: String,
    /// The observation term the values are written into.
    pub feeds: String,
    pub axes: Vec<AxisSpec>,
    /// A narrower band some of the axes are held to while the velocity command
    /// walks. `command_ranges` is then the standing band: what a full stick
    /// reaches with the robot parked. Absent is a command never narrowed.
    #[serde(default)]
    pub bands: Option<BandsSpec>,
    /// How fast the command may move, in its own units per second, because the
    /// policy was trained on one that ramps there. Absent is a command that
    /// steps, which is how every other term was trained.
    #[serde(default)]
    pub max_rate: Option<f32>,
}

/// When a command's axes narrow, and to what. See `CommandShaper`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct BandsSpec {
    /// The norm of the velocity command's `[vx, vy, wz]`, in those mixed units,
    /// below which the robot counts as standing -- the task's own rule, which is
    /// why it travels in the contract rather than being chosen here.
    pub standing_below: f32,
    /// The command block whose `lin_vel_x`, `lin_vel_y` and `ang_vel_z` that norm
    /// is taken over.
    pub velocity_term: String,
    /// Per axis, the `[low, high]` it is held to while walking; inside the
    /// standing range `command_ranges` gives.
    pub moving: BTreeMap<String, [f32; 2]>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AxisSpec {
    pub name: String,
    #[serde(default)]
    pub unit: String,
    /// Where a centred control puts this axis, in its own units. Absent is zero.
    #[serde(default)]
    pub rest: Option<f32>,
    /// Seconds full deflection takes to carry this axis from its rest to either
    /// end of its range, for an axis the controls move rather than place. Absent
    /// is an axis a control places, as a stick's position places a speed.
    #[serde(default)]
    pub integrate_s: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct OperatorDevices {
    pub gamepad: PadSpec,
    pub keyboard: KeyboardSpec,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PadSpec {
    pub scheme: String,
    /// Command axis name -> the controls that drive it, summed.
    pub axes: BTreeMap<String, OneOrMany>,
    #[serde(default)]
    pub shift: Option<Shift>,
    /// The button whose tap puts every moved axis (`integrate_s`) back at its
    /// rest: pressed and let go with no stick it could be reaching for moved in
    /// between. May be the shift -- `jumper.posture`'s R3, held for the height
    /// and tapped for the standing height.
    #[serde(default)]
    pub reset: Option<String>,
    pub deadzone: String,
    pub release_button: String,
    /// Task control name -> the one pad control it is on: a d-pad direction for
    /// a `press`, a trigger or a stick for an `amount`.
    #[serde(default)]
    pub task: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany {
    One(Binding),
    Many(Vec<Binding>),
}

impl OneOrMany {
    pub fn parts(&self) -> &[Binding] {
        match self {
            OneOrMany::One(b) => std::slice::from_ref(b),
            OneOrMany::Many(v) => v,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Binding {
    /// A dictionary axis: `Lx` `Ly` `Rx` `Ry` `LT` `RT`.
    pub source: String,
    pub sign: f32,
    /// The stretch of the control's travel this binding takes: `[from, to]`,
    /// or `[from, to, back, off]` for one that falls back to zero past `back`.
    /// Two or four, which `check` holds it to.
    #[serde(default = "full_travel")]
    pub travel: Vec<f32>,
    /// Live only while the shift has the other layer in.
    #[serde(default)]
    pub shifted: bool,
}

fn full_travel() -> Vec<f32> {
    vec![0.0, 1.0]
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Shift {
    pub button: String,
    pub gesture: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct KeyboardSpec {
    pub scheme: String,
    /// Seconds a key is held for its direction to reach full deflection, or a
    /// task's `amount` to reach 1. It climbs linearly until then and is back at
    /// rest when the key comes up. Optional only so that its absence is refused
    /// by name in `check`.
    #[serde(default)]
    pub full_after_s: Option<f32>,
    /// The notched keyboard's field, retired on 2026-09-26. Read only to refuse
    /// a contract exported before then by name.
    #[serde(default)]
    pub step: Option<f32>,
    /// The notched keyboard's `centre` key, retired on 2026-09-27; refused.
    #[serde(default)]
    pub centre: Option<serde_json::Value>,
    /// Every command axis -> its keystrokes each way, or why it has none.
    #[serde(default)]
    pub axes: BTreeMap<String, KeyAxisSpec>,
    /// Keystrokes that let go of everything, as the pad's release button does.
    #[serde(default)]
    pub release: Vec<String>,
    /// Task control name -> the keystrokes it is on.
    #[serde(default)]
    pub task: BTreeMap<String, Vec<String>>,
}

/// One command axis on the keyboard: `{"+": [...], "-": [...]}`, towards the
/// axis's own positive direction as its `positive:` words it and away from it,
/// or `{"unbound": "<why>"}`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyAxisSpec {
    #[serde(rename = "+", default)]
    pub plus: Option<Vec<String>>,
    #[serde(rename = "-", default)]
    pub minus: Option<Vec<String>>,
    #[serde(default)]
    pub unbound: Option<String>,
}

/// A keystroke as `check` already accepted it.
fn stroke(spec: &str) -> (Option<&'static str>, &'static str) {
    vocabulary::keystroke(spec).expect("keystrokes are checked before they are resolved")
}

impl OperatorSpec {
    /// Everything the Python loader refuses, refused again here.
    ///
    /// Again because a bundle is a directory of files and a contract can be
    /// edited between the exporter and the robot; this is the last reader, and
    /// each case below is a control that loads and then drives nothing, or
    /// drives two things, with nothing raised.
    pub fn check(&self, ranges: &CommandRanges) -> Result<(), String> {
        if self.command.is_empty() {
            return Err("the controller block describes no commands".into());
        }
        let mut terms = BTreeSet::new();
        let mut names = BTreeSet::new();
        for c in &self.command {
            if !terms.insert(c.term.as_str()) {
                return Err(format!("command term '{}' is listed twice", c.term));
            }
            let term_ranges = ranges.term(&c.term).ok_or_else(|| {
                format!(
                    "the contract carries no command_ranges for '{}', so a stick at full \
                     deflection means nothing this controller can compute",
                    c.term
                )
            })?;
            for a in &c.axes {
                if !names.insert(a.name.as_str()) {
                    return Err(format!("axis '{}' is named by two commands", a.name));
                }
                if Command::default().axis(&a.name).is_none() {
                    return Err(format!(
                        "axis '{}' is not a channel this controller carries; it would be \
                         driven and dropped",
                        a.name
                    ));
                }
                let [lo, hi] = *term_ranges.get(&a.name).ok_or_else(|| {
                    format!("command_ranges.{} has no range for '{}'", c.term, a.name)
                })?;
                let rest = a.rest.unwrap_or(0.0);
                if !(lo <= rest && rest <= hi) {
                    return Err(format!(
                        "{}.{} rests at {rest}, outside its range [{lo}, {hi}]; a centred \
                         stick would command a value the policy never saw",
                        c.term, a.name
                    ));
                }
                if let Some(s) = a.integrate_s {
                    if !(s.is_finite() && s > 0.0) {
                        return Err(format!(
                            "{}.{}: integrate_s {s} is not a time; it is the seconds full \
                             deflection takes to carry the axis to the end of its range",
                            c.term, a.name
                        ));
                    }
                }
            }
        }

        self.check_shape(ranges)?;

        let pad = &self.devices.gamepad;
        if pad.scheme != GAMEPAD_SCHEME {
            return Err(format!(
                "gamepad scheme is {:?}; this controller implements only {GAMEPAD_SCHEME:?}",
                pad.scheme
            ));
        }
        if pad.deadzone != GAMEPAD_DEADZONE {
            return Err(format!(
                "gamepad deadzone is {:?}; the pad service applies {GAMEPAD_DEADZONE:?} and \
                 this controller does not apply a second one",
                pad.deadzone
            ));
        }
        if !is_button(&pad.release_button) {
            return Err(format!(
                "release_button is {:?}, which the pad service does not publish",
                pad.release_button
            ));
        }
        if let Some(shift) = &pad.shift {
            if !is_button(&shift.button) {
                return Err(format!(
                    "shift button {:?} is not one the pad service publishes",
                    shift.button
                ));
            }
            if !SHIFT_GESTURES.contains(&shift.gesture.as_str()) {
                return Err(format!(
                    "shift gesture is {:?}; a shift is {:?} -- a one-tick gesture \
                     would bring a layer in for one tick",
                    shift.gesture, SHIFT_GESTURES
                ));
            }
            if shift.button == pad.release_button {
                return Err(format!(
                    "{} is both the shift and the release button",
                    shift.button
                ));
            }
        }
        for axis in pad.axes.keys() {
            if !names.contains(axis.as_str()) {
                return Err(format!("the gamepad drives '{axis}', which no command has"));
            }
        }
        let mut claims: BTreeMap<(&str, bool), Vec<(f32, f32, &str)>> = BTreeMap::new();
        for name in &names {
            let Some(entry) = pad.axes.get(*name) else {
                return Err(format!("no control drives '{name}'; every axis needs one"));
            };
            for b in entry.parts() {
                if !is_axis(&b.source) {
                    return Err(format!(
                        "'{name}' is driven by {:?}, which the pad service does not publish. \
                         It publishes {:?}",
                        b.source,
                        pad_axes()
                    ));
                }
                if b.sign != 1.0 && b.sign != -1.0 {
                    return Err(format!("'{name}' has sign {}; it must be 1 or -1", b.sign));
                }
                let ordered = match b.travel[..] {
                    [a, z] => 0.0 <= a && a < z && z <= 1.0,
                    [a, z, back, off] => 0.0 <= a && a < z && z <= back && back < off && off <= 1.0,
                    _ => false,
                };
                if !ordered {
                    return Err(format!(
                        "'{name}' has travel {:?}; it must be [from, to] with 0 <= from < to <= 1, \
                         or [from, to, back, off] with 0 <= from < to <= back < off <= 1",
                        b.travel
                    ));
                }
                // The climb is the claim. A fall is how a binding gives the
                // control up to the next one, so it may share a stretch with
                // that one's climb.
                let (a, z) = (b.travel[0], b.travel[1]);
                if b.shifted && pad.shift.is_none() {
                    return Err(format!("'{name}' is on the shifted layer and nothing shifts"));
                }
                claims.entry((b.source.as_str(), b.shifted)).or_default().push((a, z, name));
            }
        }
        for ((source, layer), spans) in claims.iter_mut() {
            spans.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            for w in spans.windows(2) {
                let ((_, b0, n0), (a1, _, n1)) = (w[0], w[1]);
                if a1 < b0 {
                    return Err(format!(
                        "'{n0}' and '{n1}' both take {source} past {a1} of its travel{}",
                        if *layer { " on the shifted layer" } else { "" }
                    ));
                }
            }
        }
        if let Some(shift) = &pad.shift {
            if !pad.axes.values().any(|e| e.parts().iter().any(|b| b.shifted)) {
                return Err(format!("{} is declared as the shift and no binding is shifted", shift.button));
            }
        }

        // The reset: a tap puts every moved axis back at its rest. With no such
        // axis it is a button that does nothing, bound; on the release it is
        // one press read twice.
        if let Some(reset) = &pad.reset {
            if !is_button(reset) {
                return Err(format!("reset is {reset:?}, which the pad service does not publish"));
            }
            if *reset == pad.release_button {
                return Err(format!(
                    "{reset} is both the reset and the release button; the release already \
                     puts every moved axis back at its rest"
                ));
            }
            if !self.command.iter().any(|c| c.axes.iter().any(|a| a.integrate_s.is_some())) {
                return Err(format!(
                    "the gamepad resets on {reset}, and no axis has integrate_s: a button \
                     that does nothing, bound"
                ));
            }
        }

        // What the task keeps for itself, declared once and bound on each
        // device by name.
        for (name, t) in &self.task {
            if !TASK_KINDS.contains(&t.kind.as_str()) {
                return Err(format!(
                    "task control '{name}' is a {:?}; a task control is one of {TASK_KINDS:?}",
                    t.kind
                ));
            }
            if t.does.trim().is_empty() {
                return Err(format!("the task keeps '{name}' and does not say what for"));
            }
        }
        let declared: BTreeSet<&str> = self.task.keys().map(String::as_str).collect();
        let on_pad: BTreeSet<&str> = pad.task.keys().map(String::as_str).collect();
        if let Some(stray) = on_pad.difference(&declared).next() {
            return Err(format!(
                "the gamepad binds task control '{stray}', which the block's `task` does not \
                 declare ({declared:?})"
            ));
        }
        if let Some(missing) = declared.difference(&on_pad).next() {
            return Err(format!(
                "the gamepad leaves out task control '{missing}': each device has to reach \
                 every control the task keeps"
            ));
        }
        // A control a command also reads would do two things with one
        // deflection -- close the claw and move the robot, each correct alone.
        let commanded: BTreeSet<&str> = claims.keys().map(|(source, _)| *source).collect();
        let reserved: BTreeSet<&str> = std::iter::once(pad.release_button.as_str())
            .chain(pad.shift.as_ref().map(|s| s.button.as_str()))
            .chain(pad.reset.as_deref())
            .collect();
        let mut owner: BTreeMap<&str, &str> = BTreeMap::new();
        for (name, control) in &pad.task {
            let press = self.task[name].kind == "press";
            if press && !is_dpad(control) {
                return Err(format!(
                    "task control '{name}' is a press on {control:?}; a press goes on a d-pad \
                     direction ({:?}) -- the buttons are the bundle's switches and the release",
                    dpad_buttons()
                ));
            }
            if !press && !is_axis(control) {
                return Err(format!(
                    "task control '{name}' is an amount on {control:?}, which is not a stick or \
                     a trigger the pad service publishes ({:?})",
                    pad_axes()
                ));
            }
            if commanded.contains(control.as_str()) {
                return Err(format!(
                    "{control} is both the task's ('{name}') and a command's; one deflection \
                     would do both"
                ));
            }
            if reserved.contains(control.as_str()) {
                return Err(format!(
                    "{control} is the task's ('{name}') and the release, the shift or the \
                     reset; one press would do both"
                ));
            }
            if let Some(other) = owner.insert(control, name) {
                return Err(format!(
                    "{control} is on two task controls, '{other}' and '{name}'; the task could \
                     not tell them apart"
                ));
            }
        }

        self.check_keyboard(&names)
    }

    /// The keyboard's half: every axis present, every keystroke one, and one
    /// keystroke doing one thing.
    fn check_keyboard(&self, names: &BTreeSet<&str>) -> Result<(), String> {
        let kb = &self.devices.keyboard;
        if kb.scheme != KEYBOARD_SCHEME {
            return Err(format!(
                "keyboard scheme is {:?}; this controller implements {KEYBOARD_SCHEME:?}",
                kb.scheme
            ));
        }
        if kb.step.is_some() {
            return Err(
                "the keyboard carries `step`, the notched keyboard retired on 2026-09-26: a \
                 key now pushes for as long as it is held (`full_after_s`); re-export the \
                 policy."
                    .into(),
            );
        }
        if kb.centre.is_some() {
            return Err(format!(
                "the keyboard carries `{CENTRE}`, retired on 2026-09-27: a key let up is \
                 back at rest, and the release lets go of everything"
            ));
        }
        match kb.full_after_s {
            Some(s) if s.is_finite() && s > 0.0 => {}
            other => {
                return Err(format!(
                    "keyboard full_after_s is {other:?}: the seconds a key is held for its \
                     direction to reach full deflection, a positive number"
                ))
            }
        }
        for axis in kb.axes.keys() {
            if !names.contains(axis.as_str()) {
                return Err(format!("the keyboard drives '{axis}', which no command has"));
            }
        }

        // One keystroke, one thing: a direction of an axis, or the release. The
        // task's controls may share one with each other -- jumper.five_foot's two
        // claws both close on Space, and a mode reads only its own side's -- but
        // not with a direction or the release.
        let mut owner: BTreeMap<String, String> = BTreeMap::new();
        let mut parsed: Vec<(Option<&'static str>, &'static str)> = Vec::new();
        let mut claim = |s: &String, what: String, shared: bool| -> Result<(), String> {
            let (m, k) = vocabulary::keystroke(s).map_err(|e| format!("the keyboard binds {e}"))?;
            parsed.push((m, k));
            match owner.get(s) {
                Some(other) if !(shared && other.starts_with("task ")) => Err(format!(
                    "{s} is on {other} and on {what}; one keystroke would do both"
                )),
                Some(_) => Ok(()),
                None => {
                    owner.insert(s.clone(), what);
                    Ok(())
                }
            }
        };
        let list = |v: &Option<Vec<String>>, where_: &str| -> Result<Vec<String>, String> {
            match v {
                Some(v) if !v.is_empty() => Ok(v.clone()),
                _ => Err(format!("{where_} is empty; it is a list of keystrokes")),
            }
        };
        for name in names {
            let Some(entry) = kb.axes.get(*name) else {
                return Err(format!(
                    "the keyboard leaves out '{name}': every command axis is on the keyboard \
                     or says why not (`unbound`)"
                ));
            };
            match (&entry.plus, &entry.minus, &entry.unbound) {
                (None, None, Some(why)) => {
                    if why.trim().is_empty() {
                        return Err(format!("'{name}' is unbound on the keyboard and does not say why"));
                    }
                }
                (Some(_), Some(_), None) => {
                    for s in list(&entry.plus, &format!("keyboard axes.{name}.+"))? {
                        claim(&s, format!("{name} +"), false)?;
                    }
                    for s in list(&entry.minus, &format!("keyboard axes.{name}.-"))? {
                        claim(&s, format!("{name} -"), false)?;
                    }
                }
                _ => {
                    return Err(format!(
                        "keyboard axes.{name} is neither {{\"+\": [...], \"-\": [...]}} nor \
                         {{\"unbound\": <why>}}"
                    ))
                }
            }
        }
        if kb.release.is_empty() {
            return Err("the keyboard has no release: nothing on it lets go of everything".into());
        }
        for s in &kb.release {
            claim(s, "the release".into(), false)?;
        }
        let declared: BTreeSet<&str> = self.task.keys().map(String::as_str).collect();
        let on_keys: BTreeSet<&str> = kb.task.keys().map(String::as_str).collect();
        if let Some(stray) = on_keys.difference(&declared).next() {
            return Err(format!(
                "the keyboard binds task control '{stray}', which the block's `task` does not \
                 declare ({declared:?})"
            ));
        }
        if let Some(missing) = declared.difference(&on_keys).next() {
            return Err(format!(
                "the keyboard leaves out task control '{missing}': each device has to reach \
                 every control the task keeps"
            ));
        }
        for (name, strokes) in &kb.task {
            if strokes.is_empty() {
                return Err(format!("task control '{name}' is on no keystroke"));
            }
            let unique: BTreeSet<&String> = strokes.iter().collect();
            if unique.len() != strokes.len() {
                return Err(format!("task control '{name}' lists a keystroke twice: {strokes:?}"));
            }
            for s in strokes {
                claim(s, format!("task {name}"), true)?;
            }
        }

        // A modifier held on its own that also modifies a key: holding it for
        // the one would do the other while the hand reaches for the key. One of
        // its own keys bound alone is the same modifier.
        let alone: BTreeSet<&str> = parsed
            .iter()
            .filter(|(m, _)| m.is_none())
            .flat_map(|(_, k)| {
                vocabulary::modifiers()
                    .into_iter()
                    .filter(move |m| m == k || vocabulary::modifier_keys(m).contains(k))
            })
            .collect();
        let chords: BTreeSet<&str> = parsed.iter().filter_map(|(m, _)| *m).collect();
        if let Some(both) = alone.intersection(&chords).next() {
            return Err(format!(
                "{both} is bound on its own and also modifies a key; holding it for the one \
                 would do it while reaching for the other"
            ));
        }
        Ok(())
    }

    /// The bands and the ramp, each refused rather than skipped: a band this
    /// controller could not apply is a robot walking under a posture its
    /// policy was never trained to hold, with nothing on the bench to say so.
    fn check_shape(&self, ranges: &CommandRanges) -> Result<(), String> {
        for c in &self.command {
            if let Some(rate) = c.max_rate {
                if !(rate.is_finite() && rate > 0.0) {
                    return Err(format!(
                        "{}: max_rate {rate} is not a rate; a command that steps carries none",
                        c.term
                    ));
                }
            }
            let Some(bands) = &c.bands else { continue };
            if !(bands.standing_below.is_finite() && bands.standing_below > 0.0) {
                return Err(format!(
                    "{}: standing_below {} is not a norm; at zero every command walks and \
                     the standing band is never reached",
                    c.term, bands.standing_below
                ));
            }
            let velocity = self
                .command
                .iter()
                .find(|v| v.term == bands.velocity_term)
                .ok_or_else(|| {
                    format!(
                        "{}: the bands follow '{}', which is not a command in this block",
                        c.term, bands.velocity_term
                    )
                })?;
            for needed in ["lin_vel_x", "lin_vel_y", "ang_vel_z"] {
                if !velocity.axes.iter().any(|a| a.name == needed) {
                    return Err(format!(
                        "{}: the bands follow '{}', which has no '{needed}' to take a speed from",
                        c.term, bands.velocity_term
                    ));
                }
            }
            if bands.moving.is_empty() {
                return Err(format!("{}: a band that narrows nothing", c.term));
            }
            let term_ranges = ranges.term(&c.term).expect("checked above");
            for (axis, [lo, hi]) in &bands.moving {
                if !c.axes.iter().any(|a| &a.name == axis) {
                    return Err(format!(
                        "{}: the moving band names '{axis}', which is not one of its axes",
                        c.term
                    ));
                }
                let [slo, shi] = term_ranges[axis];
                // A float's worth of slack: the two ends are the same number written
                // twice, once per band, and may round apart in the last bit.
                let eps = 1e-6;
                if !(lo <= hi && *lo >= slo - eps && *hi <= shi + eps) {
                    return Err(format!(
                        "{}.{axis}: the moving band [{lo}, {hi}] is not inside the standing \
                         range [{slo}, {shi}]; walking would widen what standing allows",
                        c.term
                    ));
                }
            }
        }
        Ok(())
    }

    /// Every button this block reads: the release, the shift and the reset.
    pub fn buttons(&self) -> Vec<&str> {
        let pad = &self.devices.gamepad;
        std::iter::once(pad.release_button.as_str())
            .chain(pad.shift.as_ref().map(|s| s.button.as_str()))
            .chain(pad.reset.as_deref())
            .collect()
    }

    /// Every pad axis some command binding reads.
    pub fn sources(&self) -> BTreeSet<&str> {
        self.devices
            .gamepad
            .axes
            .values()
            .flat_map(|e| e.parts().iter().map(|b| b.source.as_str()))
            .collect()
    }

    /// Every pad control this block uses: what the commands read, the release,
    /// the shift, the reset, and the task's pad controls. What a mode switch
    /// live in this mode may not also be.
    pub fn pad_uses(&self) -> BTreeSet<&str> {
        let mut uses = self.sources();
        uses.extend(self.buttons());
        uses.extend(self.devices.gamepad.task.values().map(String::as_str));
        uses
    }

    /// Every keystroke the keyboard block binds, as written, once each.
    pub fn keystrokes(&self) -> Vec<&str> {
        let kb = &self.devices.keyboard;
        let mut out: Vec<&str> = Vec::new();
        let axes = kb.axes.values().flat_map(|a| a.plus.iter().chain(a.minus.iter()).flatten());
        for s in axes.chain(kb.release.iter()).chain(kb.task.values().flatten()) {
            if !out.contains(&s.as_str()) {
                out.push(s);
            }
        }
        out
    }

    /// Every key this block reacts to, by the dictionary's names: each
    /// keystroke's key, and both keys of each modifier it uses, alone or in a
    /// chord. What a host reports as bound when one goes down.
    pub fn keys(&self) -> BTreeSet<&'static str> {
        let mut out = BTreeSet::new();
        for s in self.keystrokes() {
            let (m, k) = stroke(s);
            for name in m.into_iter().chain([k]) {
                if is_modifier(name) {
                    out.extend(vocabulary::modifier_keys(name));
                } else {
                    out.insert(name);
                }
            }
        }
        out
    }

    /// Whether the keyboard block uses this key or modifier: a key when a
    /// keystroke is on it or it is one of a used modifier's keys, a modifier
    /// when either of its keys is so used. What a key switch live in this mode
    /// may not also be.
    pub fn keyboard_uses(&self, name: &str) -> bool {
        let keys = self.keys();
        if is_modifier(name) {
            vocabulary::modifier_keys(name).iter().any(|k| keys.contains(k))
        } else {
            keys.contains(name)
        }
    }

    /// Every control the task keeps for itself (`task:`), by name.
    pub fn task(&self) -> impl Iterator<Item = &str> {
        self.task.keys().map(String::as_str)
    }
}

// ── The mapping ─────────────────────────────────────────────────────────────

/// A signed deflection through one binding's stretch of travel: zero until
/// `from`, full at `to`, held there beyond it -- or, with a fall, until `back`,
/// and zero again from `off`.
fn band(s: f32, travel: &[f32]) -> f32 {
    let a = s.abs();
    let mut x = ((a - travel[0]) / (travel[1] - travel[0])).clamp(0.0, 1.0);
    if let [_, _, back, off] = *travel {
        x *= ((off - a) / (off - back)).clamp(0.0, 1.0);
    }
    if s >= 0.0 {
        x
    } else {
        -x
    }
}

/// `full_deflection_is_range_edge`, measured from the axis's rest, the two
/// ends scaled separately.
fn scale(x: f32, lo: f32, hi: f32, rest: f32) -> f32 {
    rest + x * if x >= 0.0 { hi - rest } else { rest - lo }
}

#[derive(Debug, Clone)]
struct Resolved {
    name: String,
    lo: f32,
    hi: f32,
    rest: f32,
    /// `(source index into the dictionary's axis order, binding)`.
    bindings: Vec<(usize, Binding)>,
    /// Seconds full deflection takes to the end of the range, for an axis the
    /// controls move rather than place.
    integrate_s: Option<f32>,
}

/// What a keystroke does.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Target {
    /// One direction of an axis, by its index in `Operator::axes`: +1 or -1.
    Axis(usize, f32),
    Release,
    /// A task control, by its index in `Operator::task`.
    Task(usize),
}

/// One keystroke, resolved: the modifier held with it, the key, what it does.
#[derive(Debug, Clone)]
struct KeyBinding {
    modifier: Option<&'static str>,
    key: &'static str,
    target: Target,
}

/// Where a task control is on the pad.
#[derive(Debug, Clone, Copy)]
enum TaskPad {
    /// A trigger or a stick, by the dictionary's axis index.
    Axis(usize),
    /// A d-pad direction, as `(dpad_x, dpad_y)`.
    Dpad(i8, i8),
}

#[derive(Debug, Clone)]
struct TaskResolved {
    name: String,
    press: bool,
    pad: TaskPad,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    Pad,
    Keys,
}

/// The person, as this controller sees them: a pad, a keyboard, and which of
/// the two was touched last.
#[derive(Debug, Clone)]
pub struct Operator {
    spec: OperatorSpec,
    axes: Vec<Resolved>,
    /// Sources with a shifted binding, by index. While shifted, such a source
    /// drives **only** its shifted bindings.
    layered: [bool; 6],
    task: Vec<TaskResolved>,
    release: usize,
    shift: Option<usize>,
    /// The shift is `hold`: its layer is in while its button is down. A
    /// `toggle` flips `shifted` on each click.
    shift_hold: bool,
    reset: Option<usize>,
    /// Pad axes whose moving off centre makes a press of the reset a hold
    /// rather than a tap: the shifted layer's, and the moved axes'.
    tap_watch: [bool; 6],
    /// The reset is down and nothing it could be reaching for has moved.
    tap: bool,
    /// The real pad, as of its last frame.
    pad_axes: [f32; 6],
    pad_dpad: (i8, i8),
    held: [bool; 10],
    pad_connected: bool,
    /// The shifted layer is in.
    shifted: bool,
    /// Microseconds a key is held for full deflection: the block's
    /// `full_after_s`.
    full_after_us: f64,
    strokes: Vec<KeyBinding>,
    /// Key -> the modifiers it has a chord with in this block. While one of
    /// them is held, the key answers the chord and not itself.
    chorded: BTreeMap<&'static str, Vec<&'static str>>,
    /// Every key this block reacts to.
    bound: BTreeSet<&'static str>,
    keys_down: BTreeSet<String>,
    /// Keys down when the keyboard was let go of -- by the release, or by the
    /// pad taking over. They drive nothing until they come up and go down again.
    stale: BTreeSet<String>,
    /// The keystrokes that are acting, by index into `strokes`, and since when:
    /// what a held key's ramp is measured from.
    live_since: BTreeMap<usize, u64>,
    /// A moved axis's position, `-1..1` as a deflection is, by index into
    /// `axes`. Zero -- the rest -- for every other axis.
    moved: Vec<f32>,
    /// The clock the last `at` moved the axes to.
    last_at: Option<u64>,
    /// The host's clock, as of the last `at` or key: what a held key's time is
    /// measured to.
    now: u64,
    /// `forget` since the last input: the operator went away.
    gone: bool,
    source: Source,
}

impl Operator {
    /// Check the block against the ranges it scales to, and resolve it.
    pub fn new(spec: OperatorSpec, ranges: &CommandRanges) -> Result<Self, String> {
        spec.check(ranges)?;
        let axis_index = |name: &str| pad_axes().iter().position(|a| *a == name);
        let button_index = |name: &str| pad_buttons().iter().position(|b| *b == name);
        let pad = &spec.devices.gamepad;
        let mut axes = Vec::new();
        let mut layered = [false; 6];
        let mut tap_watch = [false; 6];
        for c in &spec.command {
            let term = ranges.term(&c.term).expect("checked");
            for a in &c.axes {
                let [lo, hi] = term[&a.name];
                let mut bindings = Vec::new();
                for b in pad.axes[&a.name].parts() {
                    let i = axis_index(&b.source).expect("checked");
                    if b.shifted {
                        layered[i] = true;
                    }
                    if b.shifted || a.integrate_s.is_some() {
                        tap_watch[i] = true;
                    }
                    bindings.push((i, b.clone()));
                }
                axes.push(Resolved {
                    name: a.name.clone(),
                    lo,
                    hi,
                    rest: a.rest.unwrap_or(0.0),
                    bindings,
                    integrate_s: a.integrate_s,
                });
            }
        }
        let task: Vec<TaskResolved> = spec
            .task
            .iter()
            .map(|(name, t)| {
                let control = &pad.task[name];
                let pad = match dpad_vector(control) {
                    Some((x, y)) => TaskPad::Dpad(x, y),
                    None => TaskPad::Axis(axis_index(control).expect("checked")),
                };
                TaskResolved { name: name.clone(), press: t.kind == "press", pad }
            })
            .collect();

        let kb = &spec.devices.keyboard;
        let mut strokes = Vec::new();
        let mut add = |s: &String, target: Target| {
            let (modifier, key) = stroke(s);
            strokes.push(KeyBinding { modifier, key, target });
        };
        for (i, a) in axes.iter().enumerate() {
            let entry = &kb.axes[&a.name];
            for s in entry.plus.iter().flatten() {
                add(s, Target::Axis(i, 1.0));
            }
            for s in entry.minus.iter().flatten() {
                add(s, Target::Axis(i, -1.0));
            }
        }
        for s in &kb.release {
            add(s, Target::Release);
        }
        for (t, tc) in task.iter().enumerate() {
            for s in &kb.task[&tc.name] {
                add(s, Target::Task(t));
            }
        }
        let mut chorded: BTreeMap<&'static str, Vec<&'static str>> = BTreeMap::new();
        for b in &strokes {
            if let Some(m) = b.modifier {
                chorded.entry(b.key).or_default().push(m);
            }
        }
        let bound = spec.keys();

        let release = button_index(&pad.release_button).expect("checked");
        let shift = pad.shift.as_ref().map(|s| button_index(&s.button).expect("checked"));
        let shift_hold = pad.shift.as_ref().is_some_and(|s| s.gesture == "hold");
        let reset = pad.reset.as_deref().map(|r| button_index(r).expect("checked"));
        let full_after_us = kb.full_after_s.expect("checked") as f64 * 1e6;
        let moved = vec![0.0; axes.len()];
        Ok(Self {
            spec,
            axes,
            layered,
            task,
            release,
            shift,
            shift_hold,
            reset,
            tap_watch,
            tap: false,
            pad_axes: [0.0; 6],
            pad_dpad: (0, 0),
            held: [false; 10],
            pad_connected: false,
            shifted: false,
            full_after_us,
            strokes,
            chorded,
            bound,
            keys_down: BTreeSet::new(),
            stale: BTreeSet::new(),
            live_since: BTreeMap::new(),
            moved,
            last_at: None,
            now: 0,
            gone: false,
            source: Source::Keys,
        })
    }

    pub fn spec(&self) -> &OperatorSpec {
        &self.spec
    }

    pub fn shifted(&self) -> bool {
        self.shifted
    }

    /// One frame of the real pad.
    pub fn pad(&mut self, pad: &Pad) {
        let held = if pad.connected { pad.buttons } else { [false; 10] };
        let pressed = |i: usize| held[i] && !self.held[i];
        let let_up = |i: usize| !held[i] && self.held[i];
        let shift_clicked = self.shift.is_some_and(pressed);
        let reset_pressed = self.reset.is_some_and(pressed);
        let reset_let_up = self.reset.is_some_and(let_up);
        self.held = held;
        self.pad_axes = if pad.connected { pad.axes } else { [0.0; 6] };
        self.pad_dpad = if pad.connected { (pad.dpad_x, pad.dpad_y) } else { (0, 0) };
        self.pad_connected = pad.connected;
        self.gone &= !pad.connected;
        if held[self.release] {
            self.let_go();
            return;
        }
        // The reset: armed by its press, disarmed by a stick it could be
        // reaching for, and put back by its release while still armed. Checked
        // before the release is read, so a stick moved in the frame the button
        // comes up still makes it a hold.
        if reset_pressed {
            self.tap = true;
        }
        if self.tap && self.pad_axes.iter().zip(self.tap_watch).any(|(v, w)| w && *v != 0.0) {
            self.tap = false;
        }
        if reset_let_up {
            if self.tap {
                self.moved.fill(0.0);
            }
            self.tap = false;
        }
        if self.shift_hold {
            self.shifted = self.shift.is_some_and(|i| held[i]);
        } else if shift_clicked {
            self.shifted = !self.shifted;
        }
        let touched = pad.connected
            && (pad.axes.iter().any(|v| *v != 0.0)
                || held.iter().any(|b| *b)
                || pad.dpad_x != 0
                || pad.dpad_y != 0);
        if touched {
            if self.source != Source::Pad {
                // Otherwise letting go of the pad would bring back a key still
                // held from before it, which nobody is looking at.
                self.drop_keys();
            }
            self.source = Source::Pad;
        }
    }

    /// The host's clock, in microseconds: what a held key's time is measured
    /// to, and what moves an axis the controls move rather than place. A host
    /// calls it every tick, before the command is read -- a key held and not
    /// repeated has no event of its own to move the clock.
    pub fn at(&mut self, now: u64) {
        let now = self.now.max(now);
        if let Some(last) = self.last_at {
            let dt = now.saturating_sub(last) as f32 / 1e6;
            if dt > 0.0 {
                self.integrate(dt);
            }
        }
        self.last_at = Some(now);
        self.now = now;
    }

    /// A key, by the dictionary's name, going down or up at `now`. Returns
    /// whether this block reacts to it.
    ///
    /// `down` is the level, and only its change means anything: a key held
    /// acts for as long as it stays down, so the auto-repeat of a held key --
    /// which a browser flags on `keydown` -- is the same key still down, not a
    /// second press. A key going down that sets a keystroke acting makes the
    /// keyboard the device that drives; one on the release lets go of
    /// everything instead.
    pub fn key(&mut self, name: &str, down: bool, now: u64) -> bool {
        self.at(now);
        self.gone = false;
        let was = self.keys_down.contains(name);
        if down {
            self.keys_down.insert(name.to_string());
        } else {
            self.keys_down.remove(name);
            self.stale.remove(name);
        }
        if !self.bound.contains(name) {
            return false;
        }
        let before: BTreeSet<usize> = self.live_since.keys().copied().collect();
        self.relive();
        if down && !was {
            let fresh: Vec<usize> =
                self.live_since.keys().copied().filter(|i| !before.contains(i)).collect();
            if fresh.iter().any(|i| self.strokes[*i].target == Target::Release) {
                self.let_go();
            } else if !fresh.is_empty() {
                self.source = Source::Keys;
            }
        }
        true
    }

    /// Whether a key or a modifier is down for the keyboard's purposes: held,
    /// and not held over from before a let-go.
    fn held_key(&self, name: &str) -> bool {
        vocabulary::key_down(name, &|k| self.keys_down.contains(k) && !self.stale.contains(k))
    }

    fn acting(&self, b: &KeyBinding) -> bool {
        self.held_key(b.key)
            && match b.modifier {
                Some(m) => self.held_key(m),
                None => !self.chorded.get(b.key).is_some_and(|ms| ms.iter().any(|m| self.held_key(m))),
            }
    }

    /// Which keystrokes act now; each newly acting one starts its ramp now.
    fn relive(&mut self) {
        let now = self.now;
        let acting: Vec<usize> =
            (0..self.strokes.len()).filter(|i| self.acting(&self.strokes[*i])).collect();
        self.live_since.retain(|i, _| acting.contains(i));
        for i in acting {
            self.live_since.entry(i).or_insert(now);
        }
    }

    /// How far a keystroke acting since `since` has climbed: 0 to 1.
    fn ramp(&self, since: u64) -> f32 {
        (self.now.saturating_sub(since) as f64 / self.full_after_us).min(1.0) as f32
    }

    /// Drop the keyboard's holds: every key down now drives nothing until it
    /// comes up and goes down again.
    fn drop_keys(&mut self) {
        self.stale.extend(self.keys_down.iter().cloned());
        self.live_since.clear();
    }

    /// Hands off: the keyboard's holds dropped, the shift off and every moved
    /// axis back at its rest. What the release does, and what a host does when
    /// the operator goes away.
    pub fn let_go(&mut self) {
        self.drop_keys();
        self.shifted = false;
        self.tap = false;
        self.moved.fill(0.0);
        self.source = Source::Keys;
    }

    /// The operator went away: forget every edge as well, so a control still
    /// down when they come back is read as one fresh press.
    pub fn forget(&mut self) {
        self.let_go();
        self.held = [false; 10];
        self.keys_down.clear();
        self.stale.clear();
        self.pad_axes = [0.0; 6];
        self.pad_dpad = (0, 0);
        self.gone = true;
    }

    /// Each axis's deflection from the device that drives, `-1..1`, in command
    /// order.
    fn raw(&self) -> Vec<f32> {
        match self.source {
            Source::Pad => self
                .axes
                .iter()
                .map(|a| {
                    let mut x = 0.0;
                    for (i, b) in &a.bindings {
                        let live =
                            if b.shifted { self.shifted } else { !(self.shifted && self.layered[*i]) };
                        if live {
                            x += band(b.sign * self.pad_axes[*i], &b.travel);
                        }
                    }
                    x.clamp(-1.0, 1.0)
                })
                .collect(),
            Source::Keys => {
                let mut v = vec![0.0f32; self.axes.len()];
                for (i, since) in &self.live_since {
                    if let Target::Axis(a, sign) = self.strokes[*i].target {
                        v[a] += sign * self.ramp(*since);
                    }
                }
                v.into_iter().map(|x| x.clamp(-1.0, 1.0)).collect()
            }
        }
    }

    /// Move every moved axis on by `dt` seconds of the pad's deflection. The
    /// keyboard places, so it moves nothing: while it drives, the moved axes
    /// keep where the pad left them.
    fn integrate(&mut self, dt: f32) {
        if self.source != Source::Pad || self.axes.iter().all(|a| a.integrate_s.is_none()) {
            return;
        }
        let d = self.raw();
        for (i, a) in self.axes.iter().enumerate() {
            if let Some(s) = a.integrate_s {
                self.moved[i] = (self.moved[i] + d[i] * dt / s).clamp(-1.0, 1.0);
            }
        }
    }

    /// Each axis's deflection, `-1..1`, in command order: where the control is
    /// for an axis it places, where the axis has been moved to for one it moves.
    pub fn deflections(&self) -> Vec<(&str, f32)> {
        let raw = self.raw();
        self.axes
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let moved = a.integrate_s.is_some() && self.source == Source::Pad;
                (a.name.as_str(), if moved { self.moved[i] } else { raw[i] })
            })
            .collect()
    }

    /// The command, in physical units.
    pub fn command(&self) -> Command {
        let mut c = Command::default();
        for (a, (_, x)) in self.axes.iter().zip(self.deflections()) {
            c.set_axis(&a.name, scale(x, a.lo, a.hi, a.rest));
        }
        c
    }

    /// The controls the task keeps for itself, by name, as they read now from
    /// the device that drives -- the rule the command follows. An `amount` is
    /// a trigger's travel on the pad and a key's ramp on the keyboard; a
    /// `press` is its d-pad direction held, or its keystroke acting.
    ///
    /// `connected` is false while nobody is there to let go of anything: after
    /// `forget`, or while the pad in use is unplugged. A hook drops what a press
    /// was holding then rather than reading the silence as a person letting go.
    pub fn task_controls(&self) -> TaskControls {
        let here = !self.gone && !(self.source == Source::Pad && !self.pad_connected);
        let values = self.task.iter().enumerate().map(|(t, tc)| {
            let v = match self.source {
                Source::Pad => match tc.pad {
                    // A stick's magnitude, as far as it is pushed either way,
                    // as `operator.py` reads it; only the triggers carry one.
                    TaskPad::Axis(i) => self.pad_axes[i].abs().min(1.0),
                    TaskPad::Dpad(x, y) => {
                        let (px, py) = self.pad_dpad;
                        if (x != 0 && px == x) || (y != 0 && py == y) {
                            1.0
                        } else {
                            0.0
                        }
                    }
                },
                Source::Keys => {
                    let mut acting = self
                        .live_since
                        .iter()
                        .filter(|(i, _)| self.strokes[**i].target == Target::Task(t))
                        .map(|(_, since)| *since);
                    if tc.press {
                        if acting.next().is_some() {
                            1.0
                        } else {
                            0.0
                        }
                    } else {
                        acting.map(|since| self.ramp(since)).fold(0.0, f32::max)
                    }
                }
            };
            (tc.name.clone(), v)
        });
        TaskControls::new(here, values)
    }

    /// Hands off, as a command: every axis at its rest.
    pub fn rest(&self) -> Command {
        let mut c = Command::default();
        for a in &self.axes {
            c.set_axis(&a.name, a.rest);
        }
        c
    }

    /// One line per binding, for a host to print before the motors are enabled:
    /// each axis's pad binding and its keystrokes, then the release, the shift,
    /// the reset and the task's own.
    pub fn describe(&self) -> String {
        let pad = &self.spec.devices.gamepad;
        let kb = &self.spec.devices.keyboard;
        let mut out = Vec::new();
        for a in &self.axes {
            let parts: Vec<String> = a
                .bindings
                .iter()
                .map(|(_, b)| {
                    let mut cut = if b.travel == [0.0, 1.0] {
                        String::new()
                    } else {
                        format!("[{}-{}]", b.travel[0], b.travel[1])
                    };
                    if let [_, _, back, off] = b.travel[..] {
                        cut += &format!("off[{back}-{off}]");
                    }
                    let layer = match (&pad.shift, b.shifted) {
                        (Some(s), true) => format!("({})", s.button),
                        _ => String::new(),
                    };
                    format!("{}{}{cut}{layer}", if b.sign < 0.0 { "-" } else { "+" }, b.source)
                })
                .collect();
            let keys = match &kb.axes[&a.name] {
                KeyAxisSpec { unbound: Some(_), .. } => "no keys".to_string(),
                k => format!(
                    "+{} -{}",
                    k.plus.iter().flatten().cloned().collect::<Vec<_>>().join("/"),
                    k.minus.iter().flatten().cloned().collect::<Vec<_>>().join("/")
                ),
            };
            let moved = if a.integrate_s.is_some() { " (moved)" } else { "" };
            out.push(format!("{} <- {}{moved} | {keys}", a.name, parts.join("")));
        }
        let shift = pad.shift.as_ref().map(|s| {
            let how = if s.gesture == "hold" { "held brings in" } else { "toggles" };
            format!(", {} {how} the layer", s.button)
        });
        let reset = pad.reset.as_ref().map(|r| format!(", {r} tapped puts moved axes back"));
        let task: Vec<String> = self
            .task
            .iter()
            .map(|t| {
                format!(
                    "{} on {} / {}: {}",
                    t.name,
                    pad.task[&t.name],
                    kb.task[&t.name].join("/"),
                    self.spec.task[&t.name].does
                )
            })
            .collect();
        let task = if task.is_empty() {
            String::new()
        } else {
            format!("; the task's own: {}", task.join(", "))
        };
        format!(
            "{}; release on {} / {}{}{}{task}",
            out.join(", "),
            pad.release_button,
            kb.release.join("/"),
            shift.unwrap_or_default(),
            reset.unwrap_or_default()
        )
    }
}

/// Whether the mode switches in `cfg` and one mode's controls keep to one
/// meaning per control while that mode runs: the check `Bundle::open` and
/// `Operators::beside` both make, in one place so the two cannot disagree.
///
/// A switch is **live in** a mode when it may be pressed there -- its `from`
/// names the mode or it has none -- or when it is the switch that latched the
/// mode, which can always switch it off again. Every host hears every control
/// whichever mode runs, so a switch live in a mode on a control that mode's
/// operator also reads would latch a mode and move the robot on one press, and
/// neither half would look wrong on its own screen:
///
/// * a pad switch's button, or its `with`, may not be a control the mode's pad
///   block uses -- except the chord that leaves the mode on the press
///   (`FsmConfig::chord_leaves_first`): a d-pad direction the mode's task keeps,
///   as the second half of a `with` chord whose rule comes first. The jumper
///   bundle's dances are that, on the claw modes' arm presets.
/// * a key switch's key, or its `with` modifier, may not be a key or a modifier
///   the mode's keyboard block uses. No exception: the keyboard's dances are not
///   live in the claw modes, whose Ctrl is the arm's, and that is said in the
///   manifest with `from` rather than argued here.
pub fn check_switches(
    cfg: &crate::config::FsmConfig,
    mode: &str,
    spec: &OperatorSpec,
) -> Result<(), String> {
    use crate::config::{Enter, When};
    let latches_here = |name: &str| {
        cfg.rules.iter().any(|r| {
            matches!(&r.when, When::Button(n) if n == name) && matches!(&r.enter, Enter::State(s) if s == mode)
        })
    };
    let pad_uses = spec.pad_uses();
    for b in cfg.buttons.iter().filter(|b| b.live_in(mode) || (b.latches() && latches_here(&b.name))) {
        if let Some(pad) = b.pad.as_deref() {
            if let Some(with) = b.with.as_deref().filter(|w| pad_uses.contains(w)) {
                return Err(format!(
                    "'{}' switches on {with} + {pad}, and mode '{mode}''s controls use {with}: \
                     one press would do both",
                    b.name
                ));
            }
            if !pad_uses.contains(pad) {
                continue;
            }
            let kept = is_dpad(pad) && spec.devices.gamepad.task.values().any(|c| c == pad);
            if kept {
                if let Err(why) = cfg.chord_leaves_first(b, mode) {
                    return Err(format!(
                        "mode '{mode}' keeps {pad} for its task, and the FSM latches '{}' on \
                         it{why}. One press would do both.",
                        b.name
                    ));
                }
                continue;
            }
            return Err(format!(
                "'{}' switches on the pad's {pad}, which mode '{mode}''s controls use too: one \
                 press would do both. Pick another control, or say with `from` that the switch \
                 is not pressed in '{mode}'",
                b.name
            ));
        }
        if let Some(key) = b.key.as_deref() {
            for name in [Some(key), b.with.as_deref()].into_iter().flatten() {
                if spec.keyboard_uses(name) {
                    return Err(format!(
                        "'{}' switches on the keyboard's {}, and mode '{mode}''s keyboard uses \
                         {name}: one press would do both. Pick another key, or say with `from` \
                         that the switch is not pressed in '{mode}'",
                        b.name,
                        b.with.as_deref().map_or(key.to_string(), |w| format!("{w} + {key}")),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Each mode's own controls: one operator per mode whose contract describes
/// some, every one of them hearing the same person.
///
/// **Different policies take different controls, and that is the design.** A
/// mode is one policy with one contract, and how a person drives it is part of
/// that contract: `jumper.posture` moves the body's height on the right stick
/// and walks at up to 0.8 m/s, `jumper.five_foot` closes a claw on a trigger
/// and walks at up to 0.5, and one bundle carries both. So each mode reads the
/// person through its own block, scaled to its own `command_ranges`, and nothing
/// here compares one mode's controls with another's.
///
/// **This was one operator per bundle** until 2026-09-26, refusing any bundle
/// whose modes described different controls, on the grounds that two modes
/// disagreeing about which stick goes forward would change the robot's
/// handedness when it changed mode. It arrived inside a larger change rather
/// than as a decision, and what it cost was every bundle mixing policies that
/// are driven differently: the claw could not ship beside the walk. Where two
/// tasks do mean a control the same way, their own files say so and their own
/// tests hold them to it (`tests/test_five_foot_pose_command.py`): a property of
/// those two tasks, not a rule for every bundle.
///
/// **Every operator hears every frame and every key, whichever mode runs.** The
/// shift layer, the keys held down and which device was touched last are the
/// person's rather than a mode's, so a mode switched into mid-gesture reads them
/// as they are now, not as it last left them.
#[derive(Debug, Clone, Default)]
pub struct Operators {
    by_mode: BTreeMap<String, Operator>,
}

impl Operators {
    /// One operator per mode whose contract describes controls, each against
    /// that contract's own ranges. A mode that describes none -- a dance nobody
    /// steers -- has none, and so may every mode: an ordinary bundle driven by
    /// the host's own command, which is empty rather than an error.
    pub fn of<'a>(
        contracts: impl Iterator<Item = (&'a str, &'a Contract)>,
    ) -> Result<Self, String> {
        let mut by_mode = BTreeMap::new();
        for (name, c) in contracts {
            let Some(spec) = c.layout.controller.as_ref() else {
                continue;
            };
            let operator = Operator::new(spec.clone(), &c.layout.command_ranges)
                .map_err(|e| format!("mode '{name}': {e}"))?;
            by_mode.insert(name.to_string(), operator);
        }
        Ok(Self { by_mode })
    }

    /// These operators, refused when a mode switch live in a mode is on a
    /// control that mode's operator reads (`check_switches`).
    pub fn beside(self, cfg: &crate::config::FsmConfig) -> Result<Self, String> {
        for (mode, o) in &self.by_mode {
            check_switches(cfg, mode, o.spec())?;
        }
        Ok(self)
    }

    /// Whether no mode describes controls.
    pub fn is_empty(&self) -> bool {
        self.by_mode.is_empty()
    }

    /// This mode's operator, or `None` for a mode whose contract describes no
    /// controls -- which then reads the command the host built itself.
    pub fn get(&self, mode: &str) -> Option<&Operator> {
        self.by_mode.get(mode)
    }

    /// Each mode's operator, by mode name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Operator)> {
        self.by_mode.iter().map(|(m, o)| (m.as_str(), o))
    }

    /// One frame of the real pad, to every operator.
    pub fn pad(&mut self, pad: &Pad) {
        self.by_mode.values_mut().for_each(|o| o.pad(pad));
    }

    /// A key going down or up at `now`, to every operator. Returns whether any
    /// of them reacts to it.
    pub fn key(&mut self, name: &str, down: bool, now: u64) -> bool {
        // Every operator, not `any`: that stops at the first one binding the
        // key and leaves the rest a press behind.
        self.by_mode.values_mut().fold(false, |bound, o| o.key(name, down, now) | bound)
    }

    /// The host's clock, to every operator: what a held key's time is
    /// measured to and what moves a moved axis. Every tick, before the tick
    /// reads the command.
    pub fn at(&mut self, now: u64) {
        self.by_mode.values_mut().for_each(|o| o.at(now));
    }

    /// Hands off, in every mode. What a page's Stop does.
    pub fn let_go(&mut self) {
        self.by_mode.values_mut().for_each(Operator::let_go);
    }

    /// The operator went away, as every mode sees it.
    pub fn forget(&mut self) {
        self.by_mode.values_mut().for_each(Operator::forget);
    }

    /// Every key some mode's controls react to.
    pub fn keys(&self) -> BTreeSet<&'static str> {
        self.by_mode.values().flat_map(|o| o.spec().keys()).collect()
    }

    /// Every mode's account of its controls on one line, the modes that
    /// describe the same ones named together:
    /// `claw_left, claw_right: ... | locomotion: ...`. For a page, which shows
    /// it as it is; the board prints `iter` a line a mode.
    pub fn describe(&self) -> String {
        let mut groups: Vec<(Vec<&str>, String)> = Vec::new();
        for (mode, o) in self.iter() {
            let text = o.describe();
            match groups.iter_mut().find(|(_, t)| *t == text) {
                Some((modes, _)) => modes.push(mode),
                None => groups.push((vec![mode], text)),
            }
        }
        groups
            .iter()
            .map(|(modes, text)| format!("{}: {text}", modes.join(", ")))
            .collect::<Vec<_>>()
            .join(" | ")
    }

    /// Where a command the host builds itself rests, for every mode at once.
    ///
    /// `play` hands the controller one finished command whatever mode is
    /// running (`py.rs`), so a channel it does not name has to rest where every
    /// mode reading that channel rests it -- the standing height for a posture
    /// policy, zero for the rest. Two modes resting one channel at different
    /// values have no such command between them, and that is refused by name
    /// rather than settled by whichever mode sorts first.
    pub fn shared_rest(&self) -> Result<Command, String> {
        let mut rest = Command::default();
        let mut owner: BTreeMap<&str, (&str, f32)> = BTreeMap::new();
        for (mode, o) in self.iter() {
            for a in &o.axes {
                match owner.get(a.name.as_str()) {
                    Some((other, at)) if (at - a.rest).abs() > 1e-6 => {
                        return Err(format!(
                            "modes '{other}' and '{mode}' rest '{}' at {at} and {}. A command \
                             the host builds itself is handed to both, and cannot be at rest \
                             for both.",
                            a.name, a.rest
                        ))
                    }
                    Some(_) => {}
                    None => {
                        owner.insert(a.name.as_str(), (mode, a.rest));
                        rest.set_axis(&a.name, a.rest);
                    }
                }
            }
        }
        Ok(rest)
    }
}

// ── Between the stick and the policy ────────────────────────────────────────

/// What happens to the operator's command before a policy sees it: the moving
/// bands and the ramp its contract declares, and nothing a contract does not.
///
/// **Both are what the policy was trained on, not a preference of this crate.**
/// A pose command drawn from its standing band while the robot is parked and its
/// moving band while it walks is a command that never asked a walking robot for
/// the standing band's lean -- so a stick at full deflection has to come back in
/// when the robot starts to move. And a command the policy only ever saw ramp at
/// `max_rate` must not be handed a thumb flicking the stick across in a tenth of
/// a second. `play` does both in the task's command term; this is the same two
/// steps for the robot and the browser, in the same order: clamp, then ramp.
///
/// Stateful for the ramp, so it lives with the mode and restarts from the rest
/// -- level, square, standing -- whenever a policy takes over, which is the
/// posture the robot is in and the one training started every episode at.
#[derive(Debug, Clone)]
pub struct CommandShaper {
    blocks: Vec<ShapedBlock>,
    rest: Command,
    dt: f32,
    last: Option<Command>,
}

#[derive(Debug, Clone)]
struct ShapedBlock {
    axes: Vec<String>,
    /// `(standing_below, moving band per axis)`.
    bands: Option<(f32, Vec<(String, [f32; 2])>)>,
    max_rate: Option<f32>,
}

impl CommandShaper {
    /// `None` when no block declares a band or a rate: nothing to do, and a
    /// mode without one passes the command through exactly as before.
    ///
    /// `dt` is the policy's period in seconds, the step the ramp is per.
    pub fn of(spec: &OperatorSpec, dt: f32) -> Option<Self> {
        let blocks: Vec<ShapedBlock> = spec
            .command
            .iter()
            .filter(|c| c.bands.is_some() || c.max_rate.is_some())
            .map(|c| ShapedBlock {
                axes: c.axes.iter().map(|a| a.name.clone()).collect(),
                bands: c.bands.as_ref().map(|b| {
                    (b.standing_below, b.moving.iter().map(|(k, v)| (k.clone(), *v)).collect())
                }),
                max_rate: c.max_rate,
            })
            .collect();
        if blocks.is_empty() {
            return None;
        }
        let mut rest = Command::default();
        for c in &spec.command {
            for a in &c.axes {
                rest.set_axis(&a.name, a.rest.unwrap_or(0.0));
            }
        }
        Some(Self { blocks, rest, dt, last: None })
    }

    /// Start the ramp again from the rest, as a policy taking over does.
    pub fn reset(&mut self) {
        self.last = Some(self.rest);
    }

    /// The command this policy step sees, for the operator's `target`.
    pub fn shape(&mut self, target: &Command) -> Command {
        let mut out = *target;
        // The operator's speed, not a ramped one: the velocity command carries
        // no rate, and the band follows what the legs are about to be asked for.
        let speed = (target.lin_vel_x.powi(2) + target.lin_vel_y.powi(2)
            + target.yaw_rate.powi(2))
        .sqrt();
        for b in &self.blocks {
            if let Some((standing_below, moving)) = &b.bands {
                if speed >= *standing_below {
                    for (axis, [lo, hi]) in moving {
                        if let Some(v) = out.axis(axis) {
                            out.set_axis(axis, v.clamp(*lo, *hi));
                        }
                    }
                }
            }
            if let (Some(rate), Some(prev)) = (b.max_rate, self.last) {
                let step = rate * self.dt;
                for axis in &b.axes {
                    if let (Some(p), Some(v)) = (prev.axis(axis), out.axis(axis)) {
                        out.set_axis(axis, p + (v - p).clamp(-step, step));
                    }
                }
            }
        }
        self.last = Some(out);
        out
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// `jumper.posture`'s block, transcribed from its `controls.yaml` the way
    /// `controls.py::controller_contract` writes it. Kept literal rather than
    /// read from the task, so a test here fails on this crate and not on an
    /// edit somebody made to the task's file.
    pub(crate) const BLOCK: &str = r#"{
      "schema": "operator_controller/2",
      "command": [
        {"term": "twist", "feeds": "velocity_commands", "frame": "body",
         "axes": [{"name": "lin_vel_x", "index": 0, "unit": "m/s"},
                  {"name": "lin_vel_y", "index": 1, "unit": "m/s"},
                  {"name": "ang_vel_z", "index": 2, "unit": "rad/s"}]},
        {"term": "posture", "feeds": "posture_command", "frame": "body",
         "axes": [{"name": "twist", "index": 0, "unit": "rad"},
                  {"name": "pitch", "index": 1, "unit": "rad"},
                  {"name": "roll", "index": 2, "unit": "rad"},
                  {"name": "height", "index": 3, "unit": "m", "rest": 0.107,
                   "rest_from": "neutral_height", "integrate_s": 2.0}]}
      ],
      "task": {},
      "devices": {
        "gamepad": {
          "scheme": "absolute", "layout": "xbox",
          "axes": {
            "lin_vel_x": {"source": "Ly", "sign": -1},
            "lin_vel_y": {"source": "Lx", "sign": -1},
            "twist": {"source": "Rx", "sign": -1, "travel": [0.0, 0.5, 0.5, 0.75]},
            "ang_vel_z": {"source": "Rx", "sign": -1, "travel": [0.5, 1.0]},
            "pitch": {"source": "Ry", "sign": -1},
            "roll": {"source": "Rx", "sign": 1, "shifted": true},
            "height": {"source": "Ry", "sign": -1, "shifted": true}
          },
          "shift": {"button": "R3", "gesture": "hold"},
          "reset": "R3",
          "deadzone": "device_reported_rescaled",
          "release_button": "B"
        },
        "keyboard": {
          "scheme": "keys", "full_after_s": 2.0,
          "axes": {
            "lin_vel_x": {"+": ["key_w", "key_up"], "-": ["key_s", "key_down"]},
            "lin_vel_y": {"+": ["key_a", "key_left"], "-": ["key_d", "key_right"]},
            "ang_vel_z": {"+": ["key_j"], "-": ["key_l"]},
            "twist": {"+": ["shift+key_j"], "-": ["shift+key_l"]},
            "pitch": {"+": ["key_i"], "-": ["key_k"]},
            "roll": {"+": ["key_o"], "-": ["key_u"]},
            "height": {"+": ["key_n"], "-": ["key_m"]}
          },
          "release": ["key_b"]
        }
      }
    }"#;

    fn ranges() -> CommandRanges {
        let pair = |v: &[(&str, [f32; 2])]| v.iter().map(|(k, r)| (k.to_string(), *r)).collect();
        let mut r = CommandRanges {
            twist: pair(&[("lin_vel_x", [-0.8, 0.8]), ("lin_vel_y", [-0.8, 0.8]), ("ang_vel_z", [-4.0, 4.0])]),
            ..Default::default()
        };
        r.other.insert(
            "posture".into(),
            pair(&[("twist", [-0.26, 0.26]), ("pitch", [-0.26, 0.26]), ("roll", [-0.26, 0.26]),
                   ("height", [0.07, 0.15])]),
        );
        r
    }

    fn spec() -> OperatorSpec {
        serde_json::from_str(BLOCK).unwrap()
    }

    fn op() -> Operator {
        Operator::new(spec(), &ranges()).unwrap()
    }

    fn pad(axes: &[(&str, f32)], buttons: &[&str]) -> Pad {
        let mut p = Pad { connected: true, ..Default::default() };
        for (a, v) in axes {
            p.axes[pad_axes().iter().position(|x| x == a).unwrap()] = *v;
        }
        for b in buttons {
            p.buttons[pad_buttons().iter().position(|x| x == b).unwrap()] = true;
        }
        p
    }

    /// Move the operator's clock `seconds` on in 10 ms ticks, as a host at
    /// 100 Hz would: what a moved axis integrates over.
    fn run(o: &mut Operator, seconds: f64) {
        let start = o.now;
        // The first tick a host sends starts the clock rather than moving
        // anything; a host has sent it long before any of this.
        o.at(start);
        let ticks = (seconds * 100.0).round() as u64;
        for i in 1..=ticks {
            o.at(start + i * 10_000);
        }
    }

    /// Put `key` down at the operator's clock and run `seconds` on, leaving the
    /// key down.
    fn hold(o: &mut Operator, key: &str, seconds: f64) {
        let now = o.now;
        assert!(o.key(key, true, now), "{key} is not bound");
        run(o, seconds);
    }

    /// Let `key` up at the operator's clock.
    fn let_up(o: &mut Operator, key: &str) {
        let now = o.now;
        o.key(key, false, now);
    }

    const EPS: f32 = 1e-6;

    /// The right stick's first half twists and its second half turns, with the
    /// twist unwinding to zero by three quarters, where the turn is at half its
    /// top rate. Right is clockwise: negative.
    #[test]
    fn the_right_stick_twists_first_and_turns_second() {
        let mut o = op();
        for (rx, twist, turn) in [(0.25, -0.5, 0.0), (0.5, -1.0, 0.0), (0.625, -0.5, -0.25),
                                  (0.75, 0.0, -0.5), (1.0, 0.0, -1.0), (-0.625, 0.5, 0.25)] {
            o.pad(&pad(&[("Rx", rx)], &[]));
            let c = o.command();
            assert!((c.base_twist - twist * 0.26).abs() < EPS, "rx {rx}: twist {}", c.base_twist);
            assert!((c.yaw_rate - turn * 4.0).abs() < EPS, "rx {rx}: turn {}", c.yaw_rate);
            assert_eq!(c.base_roll, 0.0);
        }
    }

    /// A held shift: its layer is in while R3 is down and out the frame it
    /// comes up -- the turn back at once, the stick still over. The toggle is
    /// the control group: the same block with one word changed, where letting
    /// go of R3 keeps the roll.
    #[test]
    fn a_held_r3_rolls_only_while_it_is_down() {
        let mut o = op();
        o.pad(&pad(&[("Rx", 1.0)], &["R3"]));
        let c = o.command();
        assert!(o.shifted(), "R3 held did not bring the layer in");
        assert_eq!((c.base_twist, c.yaw_rate), (0.0, 0.0), "twist and turn stop while it rolls");
        assert!((c.base_roll - 0.26).abs() < EPS, "right is right side down: +roll");
        o.pad(&pad(&[("Rx", 1.0)], &[]));
        let c = o.command();
        assert!(!o.shifted(), "R3 let go and the layer stayed in");
        assert_eq!(c.base_roll, 0.0);
        assert!(c.yaw_rate < 0.0, "R3 let go and it did not turn again at once");

        let toggled = BLOCK.replace(r#""gesture": "hold""#, r#""gesture": "toggle""#);
        assert_ne!(toggled, BLOCK, "the fixture did not take the toggle");
        let mut o = Operator::new(serde_json::from_str(&toggled).unwrap(), &ranges()).unwrap();
        for _ in 0..3 {
            o.pad(&pad(&[("Rx", 1.0)], &["R3"]));
        }
        o.pad(&pad(&[("Rx", 1.0)], &[]));
        assert!(o.shifted(), "three frames of one press toggled more than once, or not at all");
        assert!(o.command().base_roll > 0.0, "a toggle survives letting go of R3");
    }

    /// A centred pad is the posture the robot stands in. The height is the
    /// standing height and not zero, which is the value a host falling back
    /// to `Command::default()` would have sent.
    #[test]
    fn hands_off_is_the_rest_and_the_rest_is_not_zero() {
        let mut o = op();
        o.pad(&pad(&[("Lx", 0.5), ("Ry", 0.4)], &[]));
        o.pad(&pad(&[], &[]));
        let c = o.command();
        assert_eq!((c.lin_vel_x, c.lin_vel_y, c.yaw_rate, c.base_pitch), (0.0, 0.0, 0.0, 0.0));
        assert!((c.height - 0.107).abs() < EPS, "height {}", c.height);
        assert!((o.rest().height - 0.107).abs() < EPS);
        assert_eq!(Command::default().height, 0.0, "the control group: default is the floor");
    }

    /// The height is moved, not placed: R3 held and the stick pushed up is a
    /// speed, full deflection a range end in `integrate_s`, and let go it stays
    /// where it got to. Pinned because a height placed by the stick -- as it
    /// was until 2026-09-29 -- drops the body the moment a thumb slips. The
    /// pitch, a placed axis on the same stick, is the control group: it is back
    /// at rest the moment the stick is.
    #[test]
    fn the_height_moves_at_its_speed_and_stays_where_it_is_left() {
        let mut o = op();
        o.pad(&pad(&[("Ry", -1.0)], &["R3"]));
        run(&mut o, 1.0);
        let half = 0.107 + 0.5 * (0.15 - 0.107);
        assert!((o.command().height - half).abs() < 1e-4, "a second in: {}", o.command().height);
        run(&mut o, 2.0);
        assert!((o.command().height - 0.15).abs() < EPS, "it stops at the range's end");

        o.pad(&pad(&[], &["R3"]));
        run(&mut o, 1.0);
        o.pad(&pad(&[], &[]));
        run(&mut o, 1.0);
        assert!((o.command().height - 0.15).abs() < EPS, "let go, it came down");

        o.pad(&pad(&[("Ry", 1.0)], &["R3"]));
        run(&mut o, 0.5);
        let lower = 0.15 - 0.25 * (0.15 - 0.107);
        assert!((o.command().height - lower).abs() < 1e-4, "down at the same speed: {}", o.command().height);

        o.pad(&pad(&[("Ry", -1.0)], &[]));
        assert!((o.command().base_pitch - 0.26).abs() < EPS, "unshifted, the stick pitches");
        o.pad(&pad(&[], &[]));
        assert_eq!(o.command().base_pitch, 0.0, "a placed axis is back when the stick is");
    }

    /// R3 pressed and let go without the stick moving is "back to standing";
    /// held with the stick moved it was the height, and letting go keeps it.
    /// The tap and the hold are one button told apart by the stick, so each is
    /// the other's control group.
    #[test]
    fn a_tap_of_r3_puts_the_height_back_and_a_hold_does_not() {
        let mut o = op();
        o.pad(&pad(&[("Ry", -1.0)], &["R3"]));
        run(&mut o, 1.0);
        o.pad(&pad(&[], &["R3"]));
        o.pad(&pad(&[], &[]));
        let up = o.command().height;
        assert!(up > 0.12, "the hold did not raise it: {up}");

        o.pad(&pad(&[], &["R3"]));
        run(&mut o, 0.2);
        o.pad(&pad(&[], &[]));
        assert!((o.command().height - 0.107).abs() < EPS, "the tap did not put it back");

        // A stick moved in the frame R3 comes up still makes it a hold.
        o.pad(&pad(&[("Ry", -1.0)], &["R3"]));
        run(&mut o, 0.5);
        o.pad(&pad(&[("Ry", -1.0)], &[]));
        assert!(o.command().height > 0.107, "a hold let go with the stick over reset it");
    }

    /// Each keystroke pushes its axis its own way, and "+" is the axis's
    /// positive as the block words it: W forward, A left, J counter-clockwise,
    /// I nose down, O right side down -- which is left side up, posture's
    /// +roll. The keyboard has no stick for a sign to be relative to, so the
    /// signs are these words and nothing else; a key that pushed the wrong way
    /// would read as a key that works.
    #[test]
    fn each_keystroke_pushes_its_axis_its_own_way() {
        let cases: [(&[&str], &str, f32); 12] = [
            (&["key_w"], "lin_vel_x", 0.8),
            (&["key_down"], "lin_vel_x", -0.8),
            (&["key_a"], "lin_vel_y", 0.8),
            (&["key_right"], "lin_vel_y", -0.8),
            (&["key_j"], "ang_vel_z", 4.0),
            (&["key_l"], "ang_vel_z", -4.0),
            (&["key_left_shift", "key_j"], "twist", 0.26),
            (&["key_right_shift", "key_l"], "twist", -0.26),
            (&["key_i"], "pitch", 0.26),
            (&["key_k"], "pitch", -0.26),
            (&["key_o"], "roll", 0.26),
            (&["key_u"], "roll", -0.26),
        ];
        for (keys, axis, want) in cases {
            let mut o = op();
            for k in keys {
                let now = o.now;
                o.key(k, true, now);
            }
            run(&mut o, 2.0);
            let c = o.command();
            assert!((c.axis(axis).unwrap() - want).abs() < 1e-5, "{keys:?}: {axis} {:?}", c.axis(axis));
            for other in ["lin_vel_x", "lin_vel_y", "ang_vel_z", "twist", "pitch", "roll"] {
                if other != axis {
                    assert_eq!(c.axis(other), Some(0.0), "{keys:?} moved {other} as well");
                }
            }
        }
        let mut o = op();
        hold(&mut o, "key_n", 3.0);
        assert!(o.command().height > 0.14, "N raises: {}", o.command().height);
        hold(&mut o, "key_m", 0.1);
        let_up(&mut o, "key_n");
        hold(&mut o, "key_m", 5.0);
        assert!((o.command().height - 0.07).abs() < EPS, "M lowers to the bottom: {}", o.command().height);
    }

    /// The keyboard places the axis the pad moves (Control-agent 3.1: N and M
    /// are the high and the low stance, back to standing when let go). N held
    /// one second is half of its ramp, so half-way to the top; let go, the
    /// height is standing again -- a moved axis would have stayed. The pad's
    /// own height is the control group: R3 and the stick up for half a second
    /// move it a quarter of the way, it stays when let go, the keys do not
    /// move it while they drive, and it is the height again when the pad is
    /// touched.
    #[test]
    fn the_keyboard_places_the_axis_the_pad_moves() {
        let top = |h: f32| (h - 0.107) / (0.15 - 0.107);
        let mut o = op();
        hold(&mut o, "key_n", 1.0);
        assert!((top(o.command().height) - 0.5).abs() < 1e-3, "N: {}", o.command().height);
        let_up(&mut o, "key_n");
        run(&mut o, 0.5);
        assert!((o.command().height - 0.107).abs() < EPS, "let go, still {}", o.command().height);

        let mut o = op();
        o.pad(&pad(&[("Ry", -1.0)], &["R3"]));
        run(&mut o, 0.5);
        o.pad(&pad(&[], &["R3"]));
        o.pad(&pad(&[], &[]));
        run(&mut o, 0.5);
        assert!((top(o.command().height) - 0.25).abs() < 1e-2, "the pad moved it: {}", o.command().height);
        hold(&mut o, "key_w", 0.5);
        assert!((o.command().height - 0.107).abs() < EPS, "the keys drive: {}", o.command().height);
        let_up(&mut o, "key_w");
        o.pad(&pad(&[("Lx", 0.5)], &[]));
        assert!((top(o.command().height) - 0.25).abs() < 1e-2, "the pad's height again: {}", o.command().height);
    }

    /// J turns and Shift + J twists, and the two never act together: Shift
    /// pressed while J is held turns J into the chord, which climbs from rest,
    /// and Shift let go turns it back. Pinned because a key read beside its
    /// chord would turn and twist at once, each half correct on its own.
    #[test]
    fn a_modifier_turns_its_key_into_the_chord() {
        let mut o = op();
        hold(&mut o, "key_j", 1.0);
        assert!((o.command().yaw_rate - 2.0).abs() < 1e-4, "J alone turns: {}", o.command().yaw_rate);
        assert_eq!(o.command().base_twist, 0.0, "J alone twisted");

        hold(&mut o, "key_left_shift", 0.5);
        let c = o.command();
        assert_eq!(c.yaw_rate, 0.0, "with Shift down, J still turned");
        assert!((c.base_twist - 0.25 * 0.26).abs() < 1e-4, "the chord climbs from rest: {}", c.base_twist);

        let_up(&mut o, "key_left_shift");
        run(&mut o, 0.5);
        let c = o.command();
        assert_eq!(c.base_twist, 0.0, "Shift let go and it still twisted");
        assert!((c.yaw_rate - 1.0).abs() < 1e-4, "J turns again, from rest: {}", c.yaw_rate);
    }

    /// A key held climbs with time and is back at rest the moment it comes up.
    /// A second `down` for a key already down -- a browser's auto-repeat -- is
    /// the same key still held and does not start it over; the control group
    /// is the key pressed again after letting go, which does.
    #[test]
    fn a_held_key_climbs_with_time_and_is_at_rest_on_release() {
        let mut o = op();
        assert!(o.key("key_w", true, 0));
        o.at(500_000);
        assert!((o.command().lin_vel_x - 0.25 * 0.8).abs() < 1e-5, "{}", o.command().lin_vel_x);
        o.key("key_w", true, 500_000);
        o.at(1_000_000);
        assert!((o.command().lin_vel_x - 0.5 * 0.8).abs() < 1e-5,
                "a repeat started it over: {}", o.command().lin_vel_x);
        o.at(5_000_000);
        assert!((o.command().lin_vel_x - 0.8).abs() < 1e-5, "full is the edge of the range");
        o.key("key_w", false, 5_000_000);
        assert_eq!(o.command().lin_vel_x, 0.0, "let go, it did not stop");
        o.key("key_w", true, 5_000_000);
        o.at(5_500_000);
        assert!((o.command().lin_vel_x - 0.25 * 0.8).abs() < 1e-5,
                "pressed again, it did not start from rest: {}", o.command().lin_vel_x);
    }

    /// Touching the pad lets go of the keyboard, so letting go of the pad
    /// leaves the robot at rest rather than reviving an old key command.
    #[test]
    fn the_device_touched_last_drives() {
        let mut o = op();
        hold(&mut o, "key_w", 1.0);
        o.pad(&pad(&[], &[]));
        assert!((o.command().lin_vel_x - 0.4).abs() < 1e-5, "an idle pad takes nothing");
        o.pad(&pad(&[("Lx", -1.0)], &[]));
        assert_eq!(o.command().lin_vel_x, 0.0, "the pad in hand wins");
        o.pad(&pad(&[], &[]));
        assert_eq!(o.command().lin_vel_y, 0.0, "letting go revived the keys");
        assert_eq!(o.command().lin_vel_x, 0.0, "letting go revived the key still held");
        hold(&mut o, "key_d", 0.25);
        assert_eq!(o.command().lin_vel_x, 0.0, "another key revived the one held over");
        let_up(&mut o, "key_w");
        hold(&mut o, "key_w", 0.25);
        assert!((o.command().lin_vel_x - 0.1).abs() < 1e-5, "the keys start from rest");
    }

    /// The keyboard's release lets go of everything, as B on the pad does: the
    /// held keys drive nothing until pressed again, and the height the pad
    /// moved is standing again when the pad next drives. The same keys pressed
    /// again are the control group.
    #[test]
    fn the_release_key_lets_go_of_everything() {
        let mut o = op();
        o.pad(&pad(&[("Ry", -1.0)], &["R3"]));
        run(&mut o, 1.0);
        o.pad(&pad(&[], &["R3"]));
        o.pad(&pad(&[], &[]));
        assert!(o.command().height > 0.11, "the pad moved it: {}", o.command().height);
        hold(&mut o, "key_w", 1.0);
        assert!(o.command().lin_vel_x > 0.3);
        hold(&mut o, "key_b", 0.5);
        assert_eq!(o.command().lin_vel_x, 0.0, "W held through the release still walks");
        let_up(&mut o, "key_b");
        let_up(&mut o, "key_w");
        o.pad(&pad(&[("Lx", 0.5)], &[]));
        let h = o.command().height;
        assert!((h - 0.107).abs() < EPS, "the release left the pad's height up: {h}");
        hold(&mut o, "key_w", 1.0);
        assert!(o.command().lin_vel_x > 0.3, "pressed again, W walks again");
    }

    /// A block whose task keeps controls: `jumper.five_foot`'s shape -- the
    /// walk, and the claw and the arm, on the pad and on the keys. Literal,
    /// like `BLOCK`.
    pub(crate) const CLAW_BLOCK: &str = r#"{
      "schema": "operator_controller/2",
      "command": [
        {"term": "twist", "feeds": "velocity_commands", "frame": "body",
         "axes": [{"name": "lin_vel_x", "index": 0, "unit": "m/s"},
                  {"name": "lin_vel_y", "index": 1, "unit": "m/s"},
                  {"name": "ang_vel_z", "index": 2, "unit": "rad/s"}]}
      ],
      "task": {
        "claw_left": {"kind": "amount", "does": "closes the left claw"},
        "claw_right": {"kind": "amount", "does": "closes the right claw"},
        "arm_thumb_up": {"kind": "press", "does": "holds the arm out, thumb up"},
        "arm_web_up": {"kind": "press", "does": "holds the arm out, thumb-web up"}
      },
      "devices": {
        "gamepad": {
          "scheme": "absolute", "layout": "xbox",
          "axes": {
            "lin_vel_x": {"source": "Ly", "sign": -1},
            "lin_vel_y": {"source": "Lx", "sign": -1},
            "ang_vel_z": {"source": "Rx", "sign": -1}
          },
          "deadzone": "device_reported_rescaled",
          "release_button": "B",
          "task": {"claw_left": "LT", "claw_right": "RT", "arm_thumb_up": "dpad_up",
                   "arm_web_up": "dpad_left"}
        },
        "keyboard": {
          "scheme": "keys", "full_after_s": 2.0,
          "axes": {
            "lin_vel_x": {"+": ["key_w"], "-": ["key_s"]},
            "lin_vel_y": {"+": ["key_a"], "-": ["key_d"]},
            "ang_vel_z": {"+": ["key_j"], "-": ["key_l"]}
          },
          "release": ["key_b"],
          "task": {"claw_left": ["key_space"], "claw_right": ["key_space"],
                   "arm_thumb_up": ["shift"], "arm_web_up": ["ctrl"]}
        }
      }
    }"#;

    fn claw() -> Operator {
        Operator::new(serde_json::from_str(CLAW_BLOCK).unwrap(), &ranges()).unwrap()
    }

    /// The task's controls come out of `task_controls` by name, from the device
    /// that drives, and nothing else does: a stick the command reads is not
    /// there, which is what holds a hook to what its task's file gives it.
    #[test]
    fn the_task_is_handed_its_controls_by_name_from_the_pad() {
        let mut o = claw();
        o.pad(&Pad { dpad_y: 1, ..pad(&[("LT", 0.7), ("Lx", -1.0)], &[]) });
        let kept = o.task_controls();
        assert_eq!((kept.get("claw_left"), kept.get("claw_right")), (0.7, 0.0));
        assert_eq!((kept.get("arm_thumb_up"), kept.get("arm_web_up")), (1.0, 0.0));
        assert_eq!(kept.iter().count(), 4, "a control nobody declared came through");
        assert!(o.command().lin_vel_y > 0.0, "the control group: the same frame walks");
        assert!(o.describe().contains("claw_left on LT / key_space: closes the left claw"),
                "{}", o.describe());
    }

    /// On the keyboard an `amount` climbs as a held key does and a `press` is
    /// on at once: Space closing the claw over two seconds, Shift holding the
    /// arm out the moment it goes down. Space is both claws' -- a mode reads
    /// its own side's. Touching the pad takes them back, as it takes back the
    /// command.
    #[test]
    fn a_key_closes_the_claw_as_it_is_held_and_holds_the_arm_while_it_is() {
        let mut o = claw();
        hold(&mut o, "key_space", 0.8);
        let kept = o.task_controls();
        assert!((kept.get("claw_left") - 0.4).abs() < 1e-5 && (kept.get("claw_right") - 0.4).abs() < 1e-5);
        let now = o.now;
        o.key("key_right_shift", true, now);
        assert_eq!(o.task_controls().get("arm_thumb_up"), 1.0, "a press is on at once");
        assert_eq!(o.task_controls().get("arm_web_up"), 0.0);
        hold(&mut o, "key_left_ctrl", 0.1);
        assert_eq!(o.task_controls().get("arm_web_up"), 1.0, "two presses at once are both on");
        let_up(&mut o, "key_right_shift");
        let_up(&mut o, "key_space");
        let kept = o.task_controls();
        assert_eq!((kept.get("arm_thumb_up"), kept.get("claw_left")), (0.0, 0.0), "let go, still on");
        o.pad(&pad(&[("LT", 0.25)], &[]));
        let kept = o.task_controls();
        assert_eq!((kept.get("claw_left"), kept.get("arm_web_up")), (0.25, 0.0), "the pad in hand wins");
    }

    /// Nobody there -- the operator forgotten, the pad in use unplugged -- is
    /// `connected: false`, for a hook to drop what a press held rather than read
    /// the silence as letting go. The keyboard in use with the pad unplugged is
    /// still somebody there: the control group.
    #[test]
    fn nobody_there_is_not_connected() {
        let mut o = claw();
        o.pad(&Pad { dpad_y: 1, ..pad(&[], &[]) });
        assert!(o.task_controls().connected);
        o.pad(&Pad { connected: false, ..Default::default() });
        assert!(!o.task_controls().connected, "the pad in use went away and it read as there");
        hold(&mut o, "key_w", 0.1);
        o.pad(&Pad { connected: false, ..Default::default() });
        assert!(o.task_controls().connected, "the keyboard is in use; the pad going away is not the person");
        o.forget();
        assert!(!o.task_controls().connected, "forgotten and still there");
    }

    /// Each case is a block that would load and then drive nothing, or two
    /// things, or be out of one device's reach, with nothing raised. The block
    /// as written loads: the control group.
    #[test]
    fn a_malformed_block_is_refused() {
        let cases: [(&str, &str, &str); 15] = [
            // The climb is a claim and the fall is not: the block as written
            // has twist falling over the turn's climb, and loads.
            (r#""travel": [0.0, 0.5, 0.5, 0.75]"#, r#""travel": [0.0, 0.6, 0.6, 0.75]"#, "both take Rx"),
            (r#""travel": [0.0, 0.5, 0.5, 0.75]"#, r#""travel": [0.0, 0.5, 0.4, 0.75]"#, "must be"),
            (r#""gesture": "hold""#, r#""gesture": "rise""#, "a shift is"),
            (r#""source": "Ry", "sign": -1}"#, r#""source": "right_stick_y", "sign": -1}"#, "does not publish"),
            (r#""rest": 0.107"#, r#""rest": 0.0"#, "outside its range"),
            (r#""name": "roll""#, r#""name": "yaw""#, "not a channel"),
            (r#""integrate_s": 2.0"#, r#""integrate_s": 0.0"#, "is not a time"),
            (r#""reset": "R3""#, r#""reset": "B""#, "both the reset and the release"),
            (r#", "integrate_s": 2.0"#, "", "no axis has integrate_s"),
            (r#""full_after_s": 2.0"#, r#""full_after_s": 0.0"#, "a positive number"),
            (r#""pitch": {"+": ["key_i"], "-": ["key_k"]},"#, "", "leaves out 'pitch'"),
            (r#""pitch": {"+": ["key_i"], "-": ["key_k"]}"#, r#""pitch": {"unbound": " "}"#, "does not say why"),
            (r#""+": ["key_i"]"#, r#""+": ["key_x"]"#, "is not a key"),
            (r#""+": ["key_i"]"#, r#""+": ["key_s"]"#, "one keystroke would do both"),
            (r#""+": ["key_o"]"#, r#""+": ["shift"]"#, "also modifies a key"),
        ];
        for (from, to, expect) in cases {
            assert_eq!(BLOCK.matches(from).count(), 1, "{from} is not in the block once");
            let text = BLOCK.replace(from, to);
            let e = match serde_json::from_str::<OperatorSpec>(&text) {
                Ok(s) => s.check(&ranges()).unwrap_err(),
                Err(e) => e.to_string(),
            };
            assert!(e.contains(expect), "{from} -> {to}: {e}");
        }
        spec().check(&ranges()).unwrap();
    }

    /// The task's half of the same: each a control that would do nothing on
    /// one device, or two things on one press.
    #[test]
    fn a_task_control_that_cannot_work_is_refused() {
        let cases: [(&str, &str, &str); 7] = [
            (r#""arm_thumb_up": "dpad_up""#, r#""arm_thumb_up": "LT""#, "a press goes on a d-pad"),
            (r#""claw_left": "LT""#, r#""claw_left": "dpad_down""#, "not a stick or a trigger"),
            (r#""claw_left": "LT""#, r#""claw_left": "Rx""#, "both the task's"),
            (r#""claw_right": "RT""#, r#""claw_right": "LT""#, "two task controls"),
            (r#""claw_left": ["key_space"], "#, "", "leaves out task control 'claw_left'"),
            (r#""kind": "press", "does": "holds the arm out, thumb up""#,
             r#""kind": "grip", "does": "holds the arm out, thumb up""#, "one of"),
            (r#""arm_web_up": "dpad_left"}"#, r#""arm_web_up": "dpad_left", "spin": "RT"}"#, "does not declare"),
        ];
        for (from, to, expect) in cases {
            assert_eq!(CLAW_BLOCK.matches(from).count(), 1, "{from} is not in the block once");
            let text = CLAW_BLOCK.replace(from, to);
            let e = match serde_json::from_str::<OperatorSpec>(&text) {
                Ok(s) => s.check(&ranges()).unwrap_err(),
                Err(e) => e.to_string(),
            };
            assert!(e.contains(expect), "{from} -> {to}: {e}");
        }
        let _ = claw();
    }

    /// `jumper.posture`'s ranges: the walk at up to 0.8 m/s and the posture.
    const POSTURE_RANGES: &str = r#"{
        "twist": {"lin_vel_x": [-0.8, 0.8], "lin_vel_y": [-0.8, 0.8], "ang_vel_z": [-4.0, 4.0]},
        "posture": {"twist": [-0.26, 0.26], "pitch": [-0.26, 0.26], "roll": [-0.26, 0.26],
                    "height": [0.07, 0.15]}}"#;

    /// `jumper.five_foot`'s: the walk at up to 0.5 m/s, and nothing else `CLAW_BLOCK`
    /// reads.
    pub(crate) const CLAW_RANGES: &str = r#"{
        "twist": {"lin_vel_x": [-0.5, 0.5], "lin_vel_y": [-0.5, 0.5], "ang_vel_z": [-2.0, 2.0]}}"#;

    /// A contract carrying a controls block at the given ranges, for the tests
    /// that need whole contracts.
    pub(crate) fn contract_ranged(controller: &str, ranges: &str) -> Contract {
        let text = format!(
            r#"{{"obs_joint_order": ["a"], "action_joint_order": ["a"], "action_scale": 0.25,
                "default_joint_pos": {{"a": 0.0}},
                "control": {{"kp": 1.0, "kd": 0.1, "effort_limit": 1.0, "control_hz": 200.0}},
                "observation": {{"dim": 1, "terms": [{{"name": "joint_pos", "dim": 1}}]}},
                "action": {{"dim": 1}},
                "command_ranges": {ranges},
                "controller": {controller}}}"#
        );
        Contract::from_str(&text, &["a".to_string()]).unwrap()
    }

    /// A contract carrying `BLOCK` at `jumper.posture`'s ranges.
    pub(crate) fn contract_with(controller: &str) -> Contract {
        contract_ranged(controller, POSTURE_RANGES)
    }

    /// A contract describing no controls: a dance nobody steers.
    fn unsteered() -> Contract {
        let text = r#"{"obs_joint_order": ["a"], "action_joint_order": ["a"], "action_scale": 0.25,
            "default_joint_pos": {"a": 0.0},
            "control": {"kp": 1.0, "kd": 0.1, "effort_limit": 1.0, "control_hz": 50.0},
            "observation": {"dim": 1, "terms": [{"name": "joint_pos", "dim": 1}]},
            "action": {"dim": 1}}"#;
        Contract::from_str(text, &["a".to_string()]).unwrap()
    }

    /// Every channel a command carries, for comparing two whole commands.
    fn channels(c: &Command) -> Vec<Option<f32>> {
        ["lin_vel_x", "lin_vel_y", "ang_vel_z", "twist", "pitch", "roll", "height"]
            .iter()
            .map(|n| c.axis(n))
            .collect()
    }

    /// Each mode reads the pad through its own controls, at its own ranges. One
    /// frame -- full stick forward, the left trigger squeezed -- is 0.8 m/s to
    /// the posture mode, which keeps no trigger, and 0.5 m/s and a closing claw
    /// to the claw mode.
    ///
    /// The failure pinned is a bundle-wide reading: one operator for both, which
    /// hands the claw mode the posture policy's ranges -- a full stick asking it
    /// for 1.6 times the speed it trained on. The control group is each mode's
    /// operator built alone, which reads the same frame the same way.
    #[test]
    fn each_mode_reads_the_pad_through_its_own_controls() {
        let walk = contract_with(BLOCK);
        let claw = contract_ranged(CLAW_BLOCK, CLAW_RANGES);
        let mut ops = Operators::of([("walk", &walk), ("claw", &claw)].into_iter()).unwrap();
        let frame = pad(&[("Ly", -1.0), ("LT", 1.0)], &[]);
        ops.pad(&frame);
        let (w, c) = (ops.get("walk").unwrap(), ops.get("claw").unwrap());
        assert!((w.command().lin_vel_x - 0.8).abs() < EPS, "walk {}", w.command().lin_vel_x);
        assert!((c.command().lin_vel_x - 0.5).abs() < EPS, "claw {}", c.command().lin_vel_x);
        assert_eq!(w.task_controls().iter().count(), 0, "the posture mode keeps nothing");
        assert_eq!(c.task_controls().get("claw_left"), 1.0, "the claw mode's squeeze is its hook's");

        for (name, contract) in [("walk", &walk), ("claw", &claw)] {
            let mut alone = Operators::of([(name, contract)].into_iter()).unwrap();
            alone.pad(&frame);
            assert_eq!(
                channels(&alone.get(name).unwrap().command()),
                channels(&ops.get(name).unwrap().command()),
                "{name} reads the pad differently beside another mode"
            );
        }
    }

    /// The config the switch tests share: `EXAMPLE`, whose `locomotion` runs
    /// `BLOCK` here and whose `carry` runs `CLAW_BLOCK`, with `extra` bindings
    /// and `jump`'s rule where `EXAMPLE` has it, ahead of `carry`'s.
    fn switches(extra: &str) -> crate::config::FsmConfig {
        crate::config::FsmConfig::parse(&format!("{}\n{extra}\n", crate::config::EXAMPLE)).unwrap()
    }

    fn walk_and_carry() -> (Contract, Contract) {
        (contract_with(BLOCK), contract_ranged(CLAW_BLOCK, CLAW_RANGES))
    }

    /// Space is the jump's on the keyboard from locomotion and the claw's in the
    /// claw mode: two meanings for one key, kept apart by the switch's `from`,
    /// which is the jumper bundle's shape. Without `from` the switch is live in
    /// the claw mode too, where one press would close the claw and jump. The
    /// `EXAMPLE` bindings alone are the control group -- a check refusing every
    /// config would pass the refusal.
    #[test]
    fn a_key_switch_may_share_a_key_only_with_a_mode_it_is_not_pressed_in() {
        let (walk, carry) = walk_and_carry();
        let ops = || Operators::of([("locomotion", &walk), ("carry", &carry)].into_iter()).unwrap();
        ops().beside(&switches("")).expect("the example's own switches touch no mode's controls");

        let space = "[[fsm.button]]\nname = \"jump\"\nkey = \"key_space\"\non = \"toggle\"";
        let e = ops().beside(&switches(space)).unwrap_err();
        assert!(e.contains("'carry'") && e.contains("key_space"), "{e}");
        ops().beside(&switches(&format!("{space}\nfrom = [\"locomotion\"]")))
            .expect("Space is not pressed in carry, whose Space it is");
    }

    /// A switch's modifier is as much its control as its key: Ctrl + 1 pressed
    /// in the claw mode would hold the arm out and switch. Shift is the
    /// posture mode's twist chord, so a switch on it is refused there too. The
    /// same switches kept out of those modes by `from` load.
    #[test]
    fn a_key_switch_s_modifier_is_its_control_too() {
        let (walk, carry) = walk_and_carry();
        let ops = || Operators::of([("locomotion", &walk), ("carry", &carry)].into_iter()).unwrap();
        let dance = "[[fsm.button]]\nname = \"jump\"\nkey = \"key_1\"\nwith = \"ctrl\"\non = \"toggle\"";
        let e = ops().beside(&switches(dance)).unwrap_err();
        assert!(e.contains("'carry'") && e.contains("ctrl"), "{e}");
        ops().beside(&switches(&format!("{dance}\nfrom = [\"locomotion\"]"))).unwrap();

        let shifted = dance.replace("ctrl", "shift");
        let e = ops().beside(&switches(&format!("{shifted}\nfrom = [\"locomotion\"]"))).unwrap_err();
        assert!(e.contains("'locomotion'") && e.contains("shift"), "{e}");
    }

    /// A d-pad direction the claw mode keeps for its arm may be the second half
    /// of a chord that leaves the mode on the press -- the jumper bundle's
    /// dances -- and nothing else may press it there. The chord is the control
    /// group for the bare press.
    #[test]
    fn a_kept_d_pad_direction_may_only_be_a_chord_that_leaves_first() {
        let (walk, carry) = walk_and_carry();
        let ops = || Operators::of([("locomotion", &walk), ("carry", &carry)].into_iter()).unwrap();
        let chord = "[[fsm.button]]\nname = \"jump\"\npad = \"dpad_up\"\nwith = \"menu\"\non = \"toggle\"";
        ops().beside(&switches(chord)).expect("the jump's rule comes before carry's");
        let bare = "[[fsm.button]]\nname = \"jump\"\npad = \"dpad_up\"\non = \"toggle\"";
        let e = ops().beside(&switches(bare)).unwrap_err();
        assert!(e.contains("'carry' keeps dpad_up"), "{e}");
        let used = "[[fsm.button]]\nname = \"jump\"\npad = \"Rx\"\non = \"toggle\"";
        assert!(crate::config::FsmConfig::parse(&format!("{}\n{used}\n", crate::config::EXAMPLE)).is_err(),
                "an axis is no button; the parse refuses it before any mode is asked");
    }

    /// Every mode's operator hears every frame and every key, so what belongs to
    /// the person -- the shift layer, the keys held down, the release -- is
    /// the same in whichever mode is switched into. Pinned because feeding only
    /// the running mode's operator is the obvious way to write a host, and it
    /// leaves a mode switched into on a shift or a key from before the switch.
    #[test]
    fn every_mode_hears_the_person_whichever_runs() {
        let (walk, lean) = (contract_with(BLOCK), contract_with(BLOCK));
        let claw = contract_ranged(CLAW_BLOCK, CLAW_RANGES);
        let mut ops =
            Operators::of([("walk", &walk), ("lean", &lean), ("claw", &claw)].into_iter()).unwrap();
        let shifted = |ops: &Operators| ["walk", "lean"].map(|m| ops.get(m).unwrap().shifted());
        assert_eq!(shifted(&ops), [false, false]);
        ops.pad(&pad(&[], &["R3"]));
        assert_eq!(shifted(&ops), [true, true], "R3 held is one layer, in every mode");
        ops.pad(&pad(&[], &[]));

        assert!(ops.key("key_w", true, 0));
        ops.at(500_000);
        for m in ["walk", "lean", "claw"] {
            assert!(ops.get(m).unwrap().command().lin_vel_x > 0.0, "{m} missed the key");
        }
        // `CLAW_BLOCK` binds no `key_i`; the posture modes do, so it is bound.
        assert!(ops.key("key_i", true, 500_000), "a key one mode binds is a bound key");
        ops.key("key_i", false, 500_000);
        assert!(!ops.key("keypad_5", true, 500_000), "and one no mode binds is not");
        assert!(ops.keys().contains("key_left_shift"), "a chord's modifier is a key the modes use");

        // `key_w` still down: the release lets go of it in every mode.
        ops.pad(&pad(&[], &["B"]));
        for m in ["walk", "lean", "claw"] {
            assert_eq!(ops.get(m).unwrap().command().lin_vel_x, 0.0, "{m} kept the key");
        }
    }

    /// A mode describing no controls has no operator and is not an error, and
    /// neither is a bundle where no mode describes any. Both are driven by the
    /// host's own command, which `TickInput` carries beside the operators.
    #[test]
    fn a_mode_nobody_steers_has_no_operator() {
        let (walk, dance) = (contract_with(BLOCK), unsteered());
        let ops = Operators::of([("walk", &walk), ("dance", &dance)].into_iter()).unwrap();
        assert!(ops.get("walk").is_some() && ops.get("dance").is_none());
        assert!(Operators::of([("dance", &dance)].into_iter()).unwrap().is_empty());
    }

    /// One line for a page, the modes that read the person the same way named
    /// together -- a claw on either side is one set of controls.
    #[test]
    fn the_modes_describe_their_controls_together() {
        let (walk, claw) = (contract_with(BLOCK), contract_ranged(CLAW_BLOCK, CLAW_RANGES));
        let ops = Operators::of(
            [("claw_left", &claw), ("claw_right", &claw), ("walk", &walk)].into_iter(),
        )
        .unwrap();
        let line = ops.describe();
        assert!(line.starts_with("claw_left, claw_right: lin_vel_x <- "), "{line}");
        assert!(line.contains(" | walk: lin_vel_x <- "), "{line}");
        assert!(line.contains("twist <- -Rx[0-0.5]off[0.5-0.75] | +shift+key_j -shift+key_l"), "{line}");
        assert!(!line.contains('\n'), "a page shows this as one line: {line}");
    }

    /// A command the host builds itself rests every channel where the modes
    /// reading it do: the standing height for the posture mode, and nothing
    /// else moved by the claw mode, which has no height. Two modes resting one
    /// channel apart have no such command, and are named -- the control group
    /// for the refusal is the same two blocks resting together.
    #[test]
    fn a_host_built_command_rests_where_every_mode_rests() {
        let (walk, claw) = (contract_with(BLOCK), contract_ranged(CLAW_BLOCK, CLAW_RANGES));
        let rest = Operators::of([("walk", &walk), ("claw", &claw)].into_iter())
            .unwrap()
            .shared_rest()
            .unwrap();
        assert!((rest.height - 0.107).abs() < EPS);
        assert_eq!(rest.lin_vel_x, 0.0);

        let taller = contract_with(&BLOCK.replace(r#""rest": 0.107"#, r#""rest": 0.12"#));
        let e = Operators::of([("walk", &walk), ("tall", &taller)].into_iter())
            .unwrap()
            .shared_rest()
            .unwrap_err();
        assert!(e.contains("'tall'") && e.contains("'walk'") && e.contains("'height'"), "{e}");
        let again = contract_with(BLOCK);
        assert!(Operators::of([("walk", &walk), ("again", &again)].into_iter())
            .unwrap()
            .shared_rest()
            .is_ok());
    }

    /// The newest export of a task this repository actually produced whose
    /// controls are this schema, if there is one.
    fn newest_export(task: &str) -> Option<(std::path::PathBuf, Contract)> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tasks/jumper")
            .join(task)
            .join("out");
        let mut runs: Vec<_> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).collect();
        runs.sort();
        runs.iter().rev().find_map(|run| {
            let text = std::fs::read_to_string(run.join("layout.json")).ok()?;
            // Exports from before this schema carry another, which
            // `layout.rs` refuses by name.
            if !text.contains(crate::layout::OPERATOR_CONTROLLER) {
                return None;
            }
            let raw: serde_json::Value = serde_json::from_str(&text).unwrap();
            let wire: Vec<String> = raw["wire_joint_order"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect();
            Some((run.clone(), Contract::from_str(&text, &wire).unwrap()))
        })
    }

    /// A `jumper.posture` export this repository actually produced, when there
    /// is one: strided history read, the operator built, the rest a number, and
    /// the height moved rather than placed.
    ///
    /// The tests above hold this crate against the block as `BLOCK` spells it;
    /// this holds it against what `scripts/export.py` writes, which is where a
    /// field named one way on each side would show. A clone with none skips.
    #[test]
    fn a_real_posture_export_loads_and_drives() {
        let Some((path, c)) = newest_export("posture") else { return };
        assert!(c.term_stride.iter().any(|s| *s > 1), "{path:?}: no strided term");
        let ops = Operators::of([("posture", &c)].into_iter()).unwrap();
        let d = ops.get("posture").unwrap();
        let neutral = c.layout.observation.terms.iter()
            .find(|t| t.name == "posture_command")
            .and_then(|t| t.param_f64("neutral_height"))
            .unwrap() as f32;
        assert!((d.rest().height - neutral).abs() < 1e-6, "the command rests where the obs centres");
        let spec = c.layout.controller.as_ref().unwrap();
        assert!(spec.command.iter().flat_map(|t| &t.axes).any(|a| a.name == "height" && a.integrate_s.is_some()),
                "{path:?}: the height is placed, not moved");
    }

    /// `jumper.posture` and `jumper.five_foot` in one set of operators, as this
    /// repository exports them, each reading a full stick forward as the edge
    /// of its own trained range; and five_foot's claw on Space. A clone without
    /// both exports skips.
    #[test]
    fn real_posture_and_five_foot_exports_read_the_person_apart() {
        let (Some((_, posture)), Some((_, claw))) = (newest_export("posture"), newest_export("five_foot"))
        else {
            return;
        };
        let mut ops =
            Operators::of([("locomotion", &posture), ("claw_left", &claw)].into_iter()).unwrap();
        ops.pad(&pad(&[("Ly", -1.0)], &[]));
        for (mode, c) in [("locomotion", &posture), ("claw_left", &claw)] {
            let edge = c.layout.command_ranges.twist["lin_vel_x"][1];
            let got = ops.get(mode).unwrap().command().lin_vel_x;
            assert!((got - edge).abs() < 1e-6, "{mode}: {got} m/s for a full stick, trained to {edge}");
        }
        ops.key("key_space", true, 0);
        ops.at(2_000_000);
        assert_eq!(ops.get("claw_left").unwrap().task_controls().get("claw_left"), 1.0);
    }

    // ── The shaper ───────────────────────────────────────────────────────────

    /// `BLOCK` with `jumper.posture`'s bands, as `controls.py::_command_shape`
    /// writes them, and ranges that are the standing band -- twist 30, pitch 20,
    /// roll 15 degrees -- with every angle walking at 15. `rate` adds a ramp, as
    /// `jumper.five_foot`'s block carries one.
    fn banded_text(rate: Option<f32>) -> String {
        let rate = rate.map_or(String::new(), |r| format!(r#", "max_rate": {r}"#));
        let block = BLOCK.replace(
            r#""integrate_s": 2.0}]}"#,
            &format!(
                r#""integrate_s": 2.0}}],
                 "bands": {{"standing_below": 0.06, "velocity_term": "twist",
                            "moving": {{"twist": [-0.26, 0.26], "pitch": [-0.26, 0.26],
                                        "roll": [-0.26, 0.26]}}}}{rate}}}"#
            ),
        );
        assert_ne!(block, BLOCK, "the fixture did not take the bands");
        block
    }

    fn banded(rate: Option<f32>) -> (OperatorSpec, CommandRanges) {
        let block = banded_text(rate);
        let mut r = ranges();
        let pair = |v: &[(&str, [f32; 2])]| v.iter().map(|(k, r)| (k.to_string(), *r)).collect();
        r.other.insert(
            "posture".into(),
            pair(&[("twist", [-0.52, 0.52]), ("pitch", [-0.35, 0.35]), ("roll", [-0.26, 0.26]),
                   ("height", [0.07, 0.15])]),
        );
        (serde_json::from_str(&block).unwrap(), r)
    }

    /// A full stick is the standing edge while parked and the walking edge once
    /// the left stick walks. The control group is the parked half: without it the
    /// clamp could be a band that was always there.
    #[test]
    fn a_walking_robot_is_held_to_the_moving_band() {
        let (spec, r) = banded(None);
        let mut o = Operator::new(spec.clone(), &r).unwrap();
        let mut s = CommandShaper::of(&spec, 0.02).expect("a block with bands shapes");

        o.pad(&pad(&[("Ry", -1.0)], &[]));
        let parked = s.shape(&o.command());
        assert!((parked.base_pitch - 0.35).abs() < EPS, "parked, a full stick is the standing edge");

        o.pad(&pad(&[("Ry", -1.0), ("Ly", -1.0)], &[]));
        let walking = s.shape(&o.command());
        assert!(walking.lin_vel_x > 0.06);
        assert!((walking.base_pitch - 0.26).abs() < EPS, "walking, it comes back to 15 degrees");
        assert_eq!(walking.height, o.command().height, "an axis with no band is left alone");
    }

    /// The right stick's second half turns the robot, and a turning robot is a
    /// moving one: the twist its first half set comes back to the walking band.
    #[test]
    fn a_turn_brings_the_twist_back_to_its_walking_band() {
        let (spec, r) = banded(None);
        let mut o = Operator::new(spec.clone(), &r).unwrap();
        let mut s = CommandShaper::of(&spec, 0.02).unwrap();
        o.pad(&pad(&[("Rx", 0.5)], &[]));
        let twisting = s.shape(&o.command());
        assert_eq!(twisting.yaw_rate, 0.0);
        assert!((twisting.base_twist + 0.52).abs() < EPS, "half the stick is the whole standing twist");
        o.pad(&pad(&[("Rx", 0.6)], &[]));
        let turning = s.shape(&o.command());
        assert!(turning.yaw_rate < -0.06);
        assert!((turning.base_twist + 0.26).abs() < EPS, "turning, the twist is the walking band's");
        // Past 0.625 the unwinding comes under the band, and at half the top
        // turn rate it is gone. 0.74 is the control group: not zero before it.
        o.pad(&pad(&[("Rx", 0.74)], &[]));
        assert!(s.shape(&o.command()).base_twist < -0.01, "the twist was gone before 0.75");
        o.pad(&pad(&[("Rx", 0.75)], &[]));
        let half = s.shape(&o.command());
        assert!((half.yaw_rate + 0.5 * 4.0).abs() < 1e-5, "0.75 is half the top turn rate");
        assert!(half.base_twist.abs() < EPS, "at half the turn rate the twist is back to zero");
    }

    /// A ramped command leaves the rest at `max_rate` and gets there; one without
    /// a rate is there at once. The second is the control group -- and the
    /// posture-less block, which shapes nothing, is the reason `of` returns None.
    #[test]
    fn the_command_ramps_from_the_rest_at_its_rate() {
        let (ramped, r) = banded(Some(0.5));
        let mut o = Operator::new(ramped.clone(), &r).unwrap();
        let mut s = CommandShaper::of(&ramped, 0.02).unwrap();
        s.reset();
        o.pad(&pad(&[("Ry", -1.0)], &[]));
        let target = o.command();
        let first = s.shape(&target);
        assert!((first.base_pitch - 0.01).abs() < EPS, "one step is rate x dt, not the jump");
        let mut last = first.base_pitch;
        for _ in 0..40 {
            let next = s.shape(&target).base_pitch;
            assert!(next >= last - EPS, "the ramp went backwards");
            last = next;
        }
        assert!((last - 0.35).abs() < EPS, "35 steps of 0.01 reach the edge, and it stays");
        assert_eq!(s.shape(&target).height, target.height, "the ramp keeps the other axes");

        let (stepping, _) = banded(None);
        let mut s = CommandShaper::of(&stepping, 0.02).unwrap();
        s.reset();
        assert!((s.shape(&target).base_pitch - 0.35).abs() < EPS, "no rate, no ramp");
        assert!(CommandShaper::of(&spec(), 0.02).is_none(), "a block with neither shapes nothing");
    }

    /// Each is a band or a rate this controller would have to skip, and a skipped
    /// band is a walking robot handed the standing lean with nothing to say so.
    #[test]
    fn a_band_the_controller_cannot_apply_is_refused() {
        let (good, r) = banded(Some(0.5));
        assert!(good.check(&r).is_ok(), "the control group: the fixture itself loads");
        let cases: [(&str, &str, &str); 5] = [
            (r#""pitch": [-0.26, 0.26],"#, r#""pitch": [-0.5, 0.5],"#, "not inside the standing"),
            (r#""roll": [-0.26, 0.26]"#, r#""yaw": [-0.26, 0.26]"#, "not one of its axes"),
            (r#""standing_below": 0.06"#, r#""standing_below": 0.0"#, "not a norm"),
            (r#""velocity_term": "twist""#, r#""velocity_term": "walk""#, "not a command"),
            (r#""max_rate": 0.5"#, r#""max_rate": 0"#, "not a rate"),
        ];
        let text = banded_text(Some(0.5));
        for (from, to, why) in cases {
            let bad = text.replace(from, to);
            assert_ne!(bad, text, "{why}: the replacement did not apply");
            let spec: OperatorSpec = serde_json::from_str(&bad).unwrap();
            let e = spec.check(&r).unwrap_err();
            assert!(e.contains(why), "{why}: refused for another reason: {e}");
        }
    }
}
