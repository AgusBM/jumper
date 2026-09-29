//! The gamepad vocabulary, read from the dictionary.
//!
//! `controller/vocabulary.json` is the dictionary and this module is a reader of
//! it -- there is no list here, deliberately. It is `include_str!`-ed, so the
//! bytes are fixed at compile time and a board carries no file and a browser
//! needs no fetch, while still having exactly one source.
//!
//! The words are facts about the robot's gamepad service. Nothing here decides
//! what any of them *means*: that is a bundle's job in `[[fsm.button]]` and a
//! task's in `controls.yaml`, which is the whole point of the split -- a person
//! defining an action picks from a list instead of inventing one.

use std::sync::LazyLock;

use serde::Deserialize;

/// The dictionary, verbatim. Up two from the crate root, because it belongs to
/// `controller/` -- the package whose stated rule is that it knows what a pad
/// can say and nothing about what it means.
const DICTIONARY: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../controller/vocabulary.json"));

const SCHEMA: &str = "kk-control-vocabulary/2";

/// How many buttons `RobotControlRaw::ControlRaw` has.
///
/// A fact about the IDL, not about the dictionary: `dds.rs` reads ten named
/// fields. The two are held together by `the_dictionary_matches_the_wire`, so a
/// dictionary that grew a button without the wire growing one is a build
/// failure rather than an index nobody checks.
pub const PAD_BUTTON_COUNT: usize = 10;

#[derive(Deserialize)]
struct Named {
    name: String,
}

#[derive(Deserialize)]
struct Axis {
    name: String,
    range: [f32; 2],
    #[serde(default)]
    note: Option<String>,
}

/// One key a binding may name, with the codes two hosts report it under.
///
/// The codes are here because the translation is the host's job and three
/// files said so without saying how: `web.rs`, `py.rs` and BUNDLE_README each
/// note that `keypad_1` is not `Numpad1` and none of them gave the mapping.
#[derive(Deserialize)]
struct KeyEntry {
    name: String,
    glfw: u32,
    browser: String,
}

/// A modifier: `ctrl`, down while either of its two keys is.
#[derive(Deserialize)]
struct ModifierEntry {
    name: String,
    keys: Vec<String>,
}

#[derive(Deserialize)]
struct GestureEntry {
    name: String,
    fires: String,
}

#[derive(Deserialize)]
struct Absent {
    name: String,
    why: String,
}

#[derive(Deserialize)]
struct Dictionary {
    schema: String,
    buttons: Vec<Named>,
    dpad: Vec<Named>,
    axes: Vec<Axis>,
    keys: Vec<KeyEntry>,
    modifiers: Vec<ModifierEntry>,
    gestures: Vec<GestureEntry>,
    absent: Vec<Absent>,
    #[serde(rename = "_axes_note")]
    axes_note: String,
    #[serde(rename = "_gestures_note")]
    gestures_note: String,
}

static DICT: LazyLock<Dictionary> = LazyLock::new(|| {
    let d: Dictionary = serde_json::from_str(DICTIONARY)
        .expect("controller/vocabulary.json is not readable by this crate");
    assert_eq!(d.schema, SCHEMA, "controller/vocabulary.json speaks another schema");
    d
});

/// The ten buttons the service publishes, in wire order.
pub fn pad_buttons() -> Vec<&'static str> {
    DICT.buttons.iter().map(|b| b.name.as_str()).collect()
}

/// The four directions, offered as buttons because the wire carries the d-pad
/// as -1 / 0 / +1 and no gesture can read that.
pub fn dpad_buttons() -> Vec<&'static str> {
    DICT.dpad.iter().map(|b| b.name.as_str()).collect()
}

/// Every axis, in the order the wire carries them.
pub fn pad_axes() -> Vec<&'static str> {
    DICT.axes.iter().map(|a| a.name.as_str()).collect()
}

/// Whether a name is a control this hardware can report.
pub fn is_button(name: &str) -> bool {
    DICT.buttons.iter().chain(&DICT.dpad).any(|b| b.name == name)
}

pub fn is_axis(name: &str) -> bool {
    DICT.axes.iter().any(|a| a.name == name)
}

/// Every key a binding may name: the numeric keypad, and the letters a task
/// has asked for.
///
/// MuJoCo's viewer calls the user callback **in addition to** its own
/// shortcuts rather than instead, and binds every letter, so a letter does two
/// things at once there and there is no way to intercept the second. The
/// dictionary's `_keys_note` says which.
pub fn keys() -> Vec<&'static str> {
    DICT.keys.iter().map(|k| k.name.as_str()).collect()
}

/// Whether a name is a key a host can report.
///
/// This check did not exist. `pad` and `with` were checked against the
/// dictionary from the start and `key` was not, so `[[fsm.button]] key = ...`
/// took any string at all -- a typo loaded, bound nothing, and looked exactly
/// like a rule that had not fired.
pub fn is_key(name: &str) -> bool {
    DICT.keys.iter().any(|k| k.name == name)
}

/// The codes this key arrives under: `(glfw, browser)`.
///
/// `glfw` for a host reading a MuJoCo window, `browser` for one reading a web
/// page. A host that gets this wrong has a key that does nothing, which is the
/// same symptom as a binding that was never declared.
pub fn key_codes(name: &str) -> Option<(u32, &'static str)> {
    DICT.keys.iter().find(|k| k.name == name).map(|k| (k.glfw, k.browser.as_str()))
}

/// Whether a name is a modifier: `ctrl`, `shift`, `alt`.
pub fn is_modifier(name: &str) -> bool {
    DICT.modifiers.iter().any(|m| m.name == name)
}

/// The modifiers, by name.
pub fn modifiers() -> Vec<&'static str> {
    DICT.modifiers.iter().map(|m| m.name.as_str()).collect()
}

/// A modifier's keys, both sides; empty for a name that is no modifier.
pub fn modifier_keys(name: &str) -> Vec<&'static str> {
    DICT.modifiers
        .iter()
        .find(|m| m.name == name)
        .map(|m| m.keys.iter().map(String::as_str).collect())
        .unwrap_or_default()
}

/// Whether a key or a modifier is down, given which keys are: a key is itself,
/// and a modifier is either of its keys.
pub fn key_down(name: &str, keys: &dyn Fn(&str) -> bool) -> bool {
    match DICT.modifiers.iter().find(|m| m.name == name) {
        Some(m) => m.keys.iter().any(|k| keys(k)),
        None => keys(name),
    }
}

/// A keyboard binding as written: a key (`key_j`), a modifier on its own
/// (`ctrl`), or a modifier held with a key (`shift+key_j`).
///
/// `(modifier, key)`. Whole names are tried first, since `keypad_+` is a key
/// whose name ends in a plus; a modifier's name has none, so a chord splits at
/// the first. A chord on a modifier's own key -- `shift+key_left_ctrl`, or
/// `shift+ctrl` -- is refused: it is two modifiers, and neither is the key.
pub fn keystroke(spec: &str) -> Result<(Option<&'static str>, &'static str), String> {
    let whole = |name: &str| -> Option<&'static str> {
        DICT.keys
            .iter()
            .map(|k| k.name.as_str())
            .chain(DICT.modifiers.iter().map(|m| m.name.as_str()))
            .find(|n| *n == name)
    };
    if let Some(name) = whole(spec) {
        return Ok((None, name));
    }
    let refused = || {
        format!(
            "{spec:?} is not a key, a modifier ({}), or a modifier and a key (`shift+key_j`)",
            modifiers().join(", ")
        )
    };
    let (m, k) = spec.split_once('+').ok_or_else(refused)?;
    let modifier = DICT.modifiers.iter().find(|x| x.name == m).ok_or_else(refused)?;
    let held = DICT.modifiers.iter().any(|x| x.name == k || x.keys.iter().any(|y| y == k));
    match DICT.keys.iter().find(|x| x.name == k) {
        Some(key) if !held => Ok((Some(modifier.name.as_str()), key.name.as_str())),
        _ => Err(refused()),
    }
}

/// An axis's range, `[lo, hi]`: `[-1, 1]` for a stick, `[0, 1]` for a trigger.
///
/// Read from the dictionary rather than decided by the name, for the reason the
/// dictionary exists: `operator.rs` treats the two kinds differently (a key
/// moves a stick both ways and squeezes a trigger one way), and a second list
/// of which is which is the list that goes out of date.
pub fn axis_range(name: &str) -> Option<[f32; 2]> {
    DICT.axes.iter().find(|a| a.name == name).map(|a| a.range)
}

/// Whether an axis is a stick, i.e. reports both signs.
pub fn is_stick(name: &str) -> bool {
    axis_range(name).is_some_and(|r| r[0] < 0.0)
}

/// The dictionary's name for a key a browser reports as `KeyboardEvent.code`.
///
/// So a page can hand over every key it sees, by the browser's own spelling,
/// and this decides which of them mean anything. A page that translated first
/// would need a new build every time the dictionary grew a key -- which is the
/// arrangement an uploaded bundle carrying its own controller exists to avoid.
pub fn key_for_browser(code: &str) -> Option<&'static str> {
    DICT.keys.iter().find(|k| k.browser == code).map(|k| k.name.as_str())
}

/// The same, for a GLFW keycode from a MuJoCo window.
pub fn key_for_glfw(code: u32) -> Option<&'static str> {
    DICT.keys.iter().find(|k| k.glfw == code).map(|k| k.name.as_str())
}

/// When a control counts as asking for something.
///
/// The variants are code because the evaluator matches on them; the dictionary
/// carries the same names and `the_dictionary_and_the_enum_agree` refuses a
/// build where they have drifted.
///
/// The first four read one press of a control. The five clicks count presses
/// of it inside `[fsm] click_window_ms` -- see `control.rs::Buttons`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Gesture {
    Rise,
    Fall,
    Hold,
    Toggle,
    Single,
    Double,
    Triple,
    Quadruple,
    Quintuple,
}

impl Gesture {
    pub const ALL: [Gesture; 9] = [
        Gesture::Rise,
        Gesture::Fall,
        Gesture::Hold,
        Gesture::Toggle,
        Gesture::Single,
        Gesture::Double,
        Gesture::Triple,
        Gesture::Quadruple,
        Gesture::Quintuple,
    ];

    /// How many presses a click gesture counts; `None` for the four that read
    /// one press.
    pub fn clicks(self) -> Option<u32> {
        match self {
            Gesture::Rise | Gesture::Fall | Gesture::Hold | Gesture::Toggle => None,
            Gesture::Single => Some(1),
            Gesture::Double => Some(2),
            Gesture::Triple => Some(3),
            Gesture::Quadruple => Some(4),
            Gesture::Quintuple => Some(5),
        }
    }
}

impl std::fmt::Display for Gesture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Gesture::Rise => "rise",
            Gesture::Fall => "fall",
            Gesture::Hold => "hold",
            Gesture::Toggle => "toggle",
            Gesture::Single => "single",
            Gesture::Double => "double",
            Gesture::Triple => "triple",
            Gesture::Quadruple => "quadruple",
            Gesture::Quintuple => "quintuple",
        })
    }
}

/// The whole vocabulary, for a person about to write a manifest.
///
/// Rendered from the dictionary, so what you read is what will be accepted.
/// `python -m controller --vocabulary` prints the same thing from the same
/// file; this exists so a board with no Python can answer the question too.
pub fn describe() -> String {
    let d = &*DICT;
    let mut out = String::new();
    out.push_str("buttons  (a `pad = ` in [[fsm.button]])\n  ");
    out.push_str(&pad_buttons().join(" "));
    out.push_str("\n  ");
    out.push_str(&dpad_buttons().join(" "));
    for a in &d.absent {
        out.push_str(&format!("\n  not `{}`: {}", a.name, a.why));
    }
    out.push_str("\n\naxes  (a `source = ` in a task's controls.yaml)\n");
    for a in &d.axes {
        out.push_str(&format!("  {:<8}[{}, {}]", a.name, a.range[0], a.range[1]));
        if let Some(n) = &a.note {
            out.push_str(&format!("  {n}"));
        }
        out.push('\n');
    }
    out.push_str(&format!("  {}\n\n", d.axes_note));
    out.push_str("modifiers  (`with = ` on a key switch; `shift+key_j` in a controls.yaml)\n");
    for m in &d.modifiers {
        out.push_str(&format!("  {}: {}\n", m.name, m.keys.join(" ")));
    }
    out.push('\n');
    out.push_str("gestures  (an `on = ` in [[fsm.button]])\n");
    for g in &d.gestures {
        out.push_str(&format!("  {:<10}{}\n", g.name, g.fires));
    }
    out.push_str(&format!("  {}\n", d.gestures_note));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dictionary and the wire are the same ten buttons.
    ///
    /// `dds.rs` reads ten named fields out of `RobotControlRaw::ControlRaw` and
    /// writes them into a `[bool; 10]` in the dictionary's order. A dictionary
    /// that grew a button without the wire growing one would index past the end
    /// -- or, worse, silently pair the wrong name with the wrong field.
    #[test]
    fn the_dictionary_matches_the_wire() {
        assert_eq!(pad_buttons().len(), PAD_BUTTON_COUNT);
        assert_eq!(pad_axes().len(), 6, "six axes: four stick, two trigger");
        // The order is the wire's, not alphabetical. `dds.rs` relies on it.
        assert_eq!(pad_buttons()[0], "A");
        assert_eq!(pad_axes()[0], "Lx");
    }

    /// The enum is code and the dictionary is data, and they name the same nine.
    ///
    /// The evaluator matches on the variants, so they cannot come from the file;
    /// what can go wrong is the file describing a fifth that nothing implements,
    /// or dropping one that rules already use.
    #[test]
    fn the_dictionary_and_the_enum_agree() {
        let from_file: Vec<String> = DICT.gestures.iter().map(|g| g.name.clone()).collect();
        let from_code: Vec<String> = Gesture::ALL.iter().map(|g| g.to_string()).collect();
        assert_eq!(from_file, from_code);
    }

    /// A binding names a key, a modifier, or the two together, and nothing
    /// else: a name no host reports binds nothing and looks like a key that
    /// did not fire.
    #[test]
    fn a_keystroke_is_a_key_a_modifier_or_both() {
        assert_eq!(keystroke("key_j"), Ok((None, "key_j")));
        assert_eq!(keystroke("ctrl"), Ok((None, "ctrl")));
        assert_eq!(keystroke("shift+key_j"), Ok((Some("shift"), "key_j")));
        // A key whose name ends in a plus is a key, and can be modified.
        assert_eq!(keystroke("keypad_+"), Ok((None, "keypad_+")));
        assert_eq!(keystroke("ctrl+keypad_+"), Ok((Some("ctrl"), "keypad_+")));
        for bad in ["shift+ctrl", "shift+key_left_ctrl", "key_x", "shift+key_x", "hyper+key_j", "+key_j"] {
            assert!(keystroke(bad).is_err(), "{bad} was accepted");
        }
        let down = |k: &str| k == "key_right_ctrl";
        assert!(key_down("ctrl", &down), "either Ctrl is ctrl");
        assert!(!key_down("shift", &down));
        assert!(key_down("key_right_ctrl", &down));
    }

    /// The list and the check are the same list.
    ///
    /// `describe()` is what a person reads before writing a manifest, and
    /// `is_button` is what refuses one. A name in the prose and not in the
    /// check reads as supported and is rejected on build; a name in the check
    /// and not the prose is a control nobody knows they have.
    #[test]
    fn what_it_prints_is_what_it_accepts() {
        let text = describe();
        for name in pad_buttons().iter().chain(dpad_buttons().iter()) {
            assert!(text.contains(name), "{name} is accepted and not printed");
            assert!(is_button(name));
        }
        for g in Gesture::ALL {
            assert!(text.contains(&g.to_string()), "{g} is accepted and not printed");
        }
        for axis in pad_axes() {
            assert!(text.contains(axis), "{axis} is accepted and not printed");
            assert!(is_axis(axis));
        }
        for m in modifiers() {
            assert!(text.contains(m), "{m} is accepted and not printed");
        }
        // The one the service deliberately does not send. Named in the prose so
        // somebody looking for it finds out why, and refused by the check.
        assert!(text.contains("view"));
        assert!(!is_button("view"));
    }
}
