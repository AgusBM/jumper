//! The modes, and the ordered rules that move between them.
//!
//! **States are data and transitions are an ordered list.** `rl-wbc-fsm` has
//! neither a state enum nor a transition table: its states are strings in TOML
//! and its transitions are a priority cascade in C++, recomputed every tick. The
//! cascade *order* is the design -- safety before operator intent, recovery
//! before hand-off -- and a `condition -> state` map loses exactly that, silently
//! and while looking tidier.
//!
//! So the order survives into the file:
//!
//! ```toml
//! [[fsm.rule]]  when = "feedback_stale"      enter = "safe"
//! [[fsm.rule]]  when = "tilted"              enter = "safe"
//! [[fsm.rule]]  when = "in_state:safe"       enter = "@initial"
//! [[fsm.rule]]  when = "warm_start_reached"  enter = "@warm_start_ref"
//! [[fsm.rule]]  when = "button:jump"         enter = "jump"
//! [[fsm.rule]]  when = "gripper_active"      enter = "carry"
//! [[fsm.rule]]  when = "always"              enter = "locomotion"
//! ```
//!
//! Two rules about the rules, both enforced at load:
//!
//! * **`when` is a closed vocabulary, not an expression language.** An unknown
//!   word fails the load. An expression language cannot be validated, cannot be
//!   unit-tested against the states it names, and would put a parser inside the
//!   wasm module for no gain -- the set of things worth branching on here is
//!   small and every member of it is a thing the C++ already decides.
//! * **The last rule must be `always`.** With that checked, "no rule matched"
//!   cannot happen, and the cascade is total by construction rather than by a
//!   fallback buried in the evaluator.
//!
//! Anything absent is an error rather than a default. A defaulted gain is a robot
//! that moves; a defaulted timeout is a safety fallback that never fires.

use serde::Deserialize;
use std::fmt;

/// Conditions a rule may name. Closed, and matched exactly.
///
/// `in_state:` and `button:` take an argument; everything else is bare. Parsed
/// into this rather than compared as strings at tick time, so a typo is a load
/// error and the hot path is a match on an enum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum When {
    /// Always true. Required as the last rule; illegal anywhere else, because a
    /// rule after it could never fire and would look like it might.
    Always,
    /// No state sample within `state_timeout_ms`. Always armed -- a state may
    /// suspend the tilt check, never this one.
    FeedbackStale,
    /// Tilt beyond `tilt_limit`, unless the **active** state sets
    /// `skip_tilt_check`. The tick that enters such a state is still checked, so
    /// nothing can launch from an already-tipped pose.
    Tilted,
    /// No operator command within `command_timeout_ms`.
    CommandStale,
    /// The active state has this name.
    InState(String),
    /// The operator's latched mode is this one.
    Button(String),
    /// Either gripper is engaged.
    GripperActive,
    /// The active state is a warm start and the measured pose reached its target.
    WarmStartReached,
}

impl When {
    fn parse(raw: &str) -> Result<Self, Error> {
        Ok(match raw {
            "always" => When::Always,
            "feedback_stale" => When::FeedbackStale,
            "tilted" => When::Tilted,
            "command_stale" => When::CommandStale,
            "gripper_active" => When::GripperActive,
            "warm_start_reached" => When::WarmStartReached,
            other => match other.split_once(':') {
                Some(("in_state", name)) if !name.is_empty() => When::InState(name.to_string()),
                Some(("button", name)) if !name.is_empty() => When::Button(name.to_string()),
                _ => return Err(Error::UnknownCondition(other.to_string())),
            },
        })
    }
}

impl fmt::Display for When {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            When::Always => f.write_str("always"),
            When::FeedbackStale => f.write_str("feedback_stale"),
            When::Tilted => f.write_str("tilted"),
            When::CommandStale => f.write_str("command_stale"),
            When::InState(s) => write!(f, "in_state:{s}"),
            When::Button(b) => write!(f, "button:{b}"),
            When::GripperActive => f.write_str("gripper_active"),
            When::WarmStartReached => f.write_str("warm_start_reached"),
        }
    }
}

/// Where a rule sends the robot. `@initial` and `@warm_start_ref` are
/// indirections into `[fsm]`, so the recovery target is named once and a rule
/// cannot disagree with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Enter {
    State(String),
    Initial,
    WarmStartRef,
    /// `@stay` -- match, and change nothing.
    ///
    /// Needed because a cascade without it is not only a priority order, it is
    /// also a claim that every tick ends somewhere new. A warm start ignores the
    /// operator entirely until it has finished, and the C++ says so with a bare
    /// `return` in the middle of the function. Here it is a rule, so the
    /// stickiness is visible in the file beside the order it depends on -- and
    /// the first draft of this file, which had no `@stay`, walked a robot away
    /// mid-ramp because the cascade fell through to `always`.
    Stay,
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub when: When,
    pub enter: Enter,
}

/// One mode. The model is named here; everything about how to drive it comes
/// from that model's own contract.
///
/// The gain split is `rl-wbc-fsm`'s and is kept verbatim:
/// `effective = contract.kp * kp_scale + kp_offset`. The trained gain stays in
/// the contract, the bench adjustment stays here, and neither can be mistaken
/// for the other by reading one file.
#[derive(Debug, Clone, Deserialize)]
pub struct StateConfig {
    pub name: String,
    /// Filename under the models directory. Empty means this state runs no
    /// model: it holds, and that is a mode rather than a failure.
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub hold_current: bool,
    #[serde(default)]
    pub warm_start: bool,
    /// Suspend the tilt fallback **while active**. DANGEROUS -- a diverging
    /// policy will not drop to safe on its own. Only for motion that legitimately
    /// exceeds the limit, and never for the safe state itself.
    #[serde(default)]
    pub skip_tilt_check: bool,
    #[serde(default)]
    pub kp: f32,
    #[serde(default)]
    pub kd: f32,
    #[serde(default = "one")]
    pub kp_scale: f32,
    #[serde(default = "one")]
    pub kd_scale: f32,
    #[serde(default)]
    pub kp_offset: f32,
    #[serde(default)]
    pub kd_offset: f32,
    /// Which command channels this state's policy is fed. Empty takes the global
    /// list. Policies do not agree on what a command is -- locomotion wants a
    /// twist, a jump wants a target and a trigger -- and getting it wrong feeds
    /// a policy the wrong numbers with every dimension still correct.
    #[serde(default)]
    pub command_terms: Vec<String>,
    /// The task this state's policy was trained on, as `scripts/deploy.py`
    /// records it from the manifest. Informative, and what `hook` is looked up
    /// by.
    #[serde(default)]
    pub task: String,
    /// The task's deploy hook and its configuration (`hook.rs`): present means
    /// build it, absent means the mode runs without one. The manifest mode's
    /// `hook` object, which the bundler copies here verbatim.
    #[serde(default)]
    pub hook: Option<toml::Table>,
}

fn one() -> f32 {
    1.0
}

impl StateConfig {
    pub fn has_model(&self) -> bool {
        !self.model.is_empty()
    }
}

pub use crate::vocabulary::Gesture;

/// One thing the operator can ask for, on one device.
///
/// A pad button **or** a key, never both. Until 2026-09-29 one binding could
/// carry both, and a bundle's `[fsm.keyboard]` table could make a key *be* a
/// pad button, so a key reached a mode through the pad. The keyboard is a path
/// of its own now, beside the pad rather than behind it: the pad's A and the
/// keyboard's Space are two bindings with one name, and a name is one latch, so
/// either switches the jump on and either switches it off. Each carries its own
/// gesture, its own modifier and its own `from`, because the two devices do not
/// have to agree -- in a claw mode Ctrl is the arm's, so the keyboard's dances
/// cannot be reached from there, while the pad's `menu` chord can.
#[derive(Debug, Clone, Deserialize)]
pub struct ButtonBinding {
    /// What rules call it. `when = "button:<name>"`. Several bindings may share
    /// one name -- one per device, one per key -- and then share its latch.
    pub name: String,
    /// A button the robot's pad service publishes, or one of the four
    /// `dpad_*` pseudo-buttons.
    #[serde(default)]
    pub pad: Option<String>,
    /// A key name in the dictionary (`keypad_1`), not a browser's, or a
    /// modifier's name (`ctrl`), which is either of its keys: Ctrl let go on
    /// its own leaves a dance.
    #[serde(default)]
    pub key: Option<String>,
    pub on: Gesture,
    /// What must be **held** for this binding to count: a pad button for a pad
    /// binding, a modifier (`ctrl`, `shift`, `alt`) for a key.
    ///
    /// `with = "menu"` plus `pad = "dpad_up"` is "Menu and up", which is how
    /// the C++ selected a dance; `with = "ctrl"` plus `key = "key_1"` is the
    /// keyboard's. A modifier is *consumed* by whatever it modifies: once a
    /// modified binding fires, the modifier's own bindings stay quiet until it
    /// is released. Without that, a single `menu` bound to "leave dance mode"
    /// would fire the instant you reached for a direction.
    #[serde(default)]
    pub with: Option<String>,
    /// The states this binding may be pressed in; absent is any.
    ///
    /// It gates **entering** only: a latch cannot come on, and a one-shot
    /// gesture, an event or a `leaves` cannot fire, while the cascade is in a
    /// state not listed. A latch already on can always be switched off by its
    /// own control, from inside the mode it entered -- A pressed again leaves
    /// the jump though the jump is not in its own `from`. Since 2026-09-29, when
    /// the jumper bundle went to one mode at a time and needed to say that A does
    /// nothing in a claw mode: there, Space is the claw's.
    #[serde(default)]
    pub from: Option<Vec<String>>,
    /// Latches this binding releases when it fires. A binding with `leaves`
    /// latches nothing and is never active itself, so no rule reads it.
    ///
    /// `menu` let go on its own leaves whichever dance is on, and the cascade
    /// falls through to its `always` mode -- the default -- rather than to a
    /// mode the dance was entered from: that is what "interrupt and go back to
    /// the default" means with one mode latched at a time.
    #[serde(default)]
    pub leaves: Vec<String>,
    /// This control is read **by a mode**, by name, rather than latched by a
    /// rule.
    ///
    /// A recorded motion's `go` is the case: the jump's push-off is 40 ms wide,
    /// so it is an event inside a mode and not a transition between modes. It
    /// is still an ordinary binding -- same vocabulary, same gestures, same
    /// modifier rules -- and it is still checked, just against the contracts
    /// instead of against the cascade: `Controller::new` refuses an `event`
    /// nothing names and a `go_event` naming nothing.
    ///
    /// Without this flag the check below would reject it as a dead key, which
    /// is the right answer for a button no rule reacts to and the wrong one
    /// here. Declared rather than inferred, so a typo in a rule cannot silently
    /// turn a latch into an event.
    #[serde(default)]
    pub event: bool,
    /// The `[fsm] click_window_ms` this binding counts clicks in, in µs.
    ///
    /// Not a field of the entry: one window for the whole controller, copied
    /// onto each binding at parse so the thing counting clicks cannot be
    /// handed a binding without the window its clicks are counted in. Zero for
    /// a binding that is not a click, and for a config that has none.
    #[serde(skip)]
    pub click_window_us: u64,
}

impl ButtonBinding {
    /// Which control this binding reads, as clicks are counted on it: the same
    /// pad button or key, under the same modifier.
    pub fn control(&self) -> (Option<&str>, Option<&str>, Option<&str>) {
        (self.pad.as_deref(), self.key.as_deref(), self.with.as_deref())
    }

    /// The one control this binding presses: its pad button or its key.
    pub fn control_name(&self) -> &str {
        self.pad.as_deref().or(self.key.as_deref()).unwrap_or_default()
    }

    /// What a person sees the binding as.
    fn shown(&self) -> String {
        match &self.with {
            Some(m) => format!("{m} + {}", self.control_name()),
            None => self.control_name().to_string(),
        }
    }

    /// Whether the binding may be pressed while the cascade is in `state`.
    pub fn live_in(&self, state: &str) -> bool {
        self.from.as_ref().is_none_or(|f| f.iter().any(|s| s == state))
    }

    /// Whether this binding latches: a `toggle`, or a click that is not an
    /// event. A `leaves` never does.
    pub fn latches(&self) -> bool {
        self.leaves.is_empty()
            && (self.on == Gesture::Toggle || (self.on.clicks().is_some() && !self.event))
    }
}

pub use crate::vocabulary::{dpad_buttons, pad_buttons};

/// Whether a pad button, a `dpad_*` direction, a key or a modifier is down:
/// the pad's for a pad name, the keys' otherwise. Names never collide -- the
/// pad's are `A` `menu` `dpad_up`, the keys' `key_space` `keypad_1`, the
/// modifiers' `ctrl` `shift` `alt` -- so one lookup serves a binding's control
/// and its modifier alike.
pub fn is_down(name: &str, pad: &crate::types::Pad, keys: &dyn Fn(&str) -> bool) -> bool {
    match name {
        "dpad_up" => pad.dpad_y > 0,
        "dpad_down" => pad.dpad_y < 0,
        "dpad_right" => pad.dpad_x > 0,
        "dpad_left" => pad.dpad_x < 0,
        other if crate::vocabulary::is_button(other) => pad.button(other),
        other => crate::vocabulary::key_down(other, keys),
    }
}

impl ButtonBinding {
    /// Is this control down, this frame?
    pub fn down(&self, pad: &crate::types::Pad, keys: &dyn Fn(&str) -> bool) -> bool {
        is_down(self.control_name(), pad, keys)
    }
}


#[derive(Debug, Clone)]
pub struct FsmConfig {
    pub states: Vec<StateConfig>,
    pub rules: Vec<Rule>,
    /// What the operator can ask for, and which physical control asks it.
    ///
    /// Here rather than in the host for the same reason the rule order is here:
    /// which control means what is a decision, and compiling it into a program
    /// means changing it requires a different build of that program. A rule
    /// naming a button nothing declares is refused: it can never fire.
    pub buttons: Vec<ButtonBinding>,
    /// One mode latched at a time: a latch coming on releases every other.
    ///
    /// Off, latches are a set and the cascade's order settles two that are on
    /// together -- which is what a file that says nothing gets, as before. On,
    /// the last switch pressed is the mode. The jumper bundle turned it on on
    /// 2026-09-29, asked for with the Control-agent 3.1 layout: LB in the right
    /// claw mode lands in the left one rather than latching behind it, and a
    /// dance entered from a claw mode leaves the claw, so the dance ending --
    /// or Menu interrupting it -- is the default mode again.
    pub exclusive: bool,
    /// Entered on recovery, and at startup.
    pub initial_state: String,
    /// Where a warm start hands off to.
    pub warm_start_ref: String,
    /// Where every safety fallback goes.
    pub safe_state: String,
    pub tilt_limit: f32,
    pub state_timeout_us: u64,
    pub command_timeout_us: u64,
    /// Seconds the mode-switch setpoint takes to slide. **Not** how long the
    /// switch takes: the hand-off waits on measured pose.
    pub mode_switch_ramp_s: f32,
    /// How close, in rad, every joint must be to the incoming model's default
    /// pose before its policy starts.
    pub pose_reach_tol: f32,
    pub warm_start_duration_s: f32,
    pub ramp_kp: f32,
    pub ramp_kd: f32,
    /// How long after one press the next still counts as the same click
    /// gesture, in µs. `None` when no binding counts clicks; required when one
    /// does, since a window nobody chose is a gesture nobody can predict.
    pub click_window_us: Option<u64>,
}

impl FsmConfig {
    pub fn state(&self, name: &str) -> Option<&StateConfig> {
        self.states.iter().find(|s| s.name == name)
    }

    /// Whether some switch reads this key, by the dictionary's name: as its own
    /// key, or as one of the keys of a modifier it is on or held with. What a
    /// host reports back when the key goes down, so "that key does nothing" can
    /// be told from "the rule did not fire".
    pub fn switches_on_key(&self, key: &str) -> bool {
        let reads = |name: &str| name == key || crate::vocabulary::modifier_keys(name).contains(&key);
        self.buttons
            .iter()
            .any(|b| b.key.as_deref().is_some_and(reads) || (b.key.is_some() && b.with.as_deref().is_some_and(reads)))
    }

    /// Whether a FSM binding on a control a mode's task keeps is a **chord that
    /// leaves the mode on the press** -- the one way the two may share it.
    ///
    /// `menu` held and right on the d-pad is the dance; right alone, in a claw mode,
    /// picks an arm preset. Those are two presses, not one, but only because of
    /// when each is read: the chord's rule fires the tick the direction goes down,
    /// the cascade has left the claw mode by the time that mode would tick, and so
    /// the claw's hook never sees the press the dance was entered on. Each condition
    /// below is one way for that to stop being true, and each is refused:
    ///
    /// * no `with`: the bare press is the switch, and the task's reading as well;
    /// * an `event`: read by a mode rather than latched, so nothing leaves;
    /// * anything but `toggle`: `fall` and `hold` switch after the press or during
    ///   it, and `rise` is true for one tick -- the cascade leaves and comes back,
    ///   the mode starts again, and reads the direction still down as a press;
    /// * a rule that keeps the cascade in this mode at or ahead of the chord's:
    ///   the chord's own rule entering it, or `@stay`, or any rule before it that
    ///   can match while this mode runs and enters it or stays. A rule that
    ///   cannot match here -- `in_state:` another mode, `warm_start_reached` in a
    ///   mode that is no warm start -- keeps nothing, wherever it is.
    pub fn chord_leaves_first(&self, b: &ButtonBinding, mode: &str) -> Result<(), String> {
        if b.with.is_none() {
            return Err(String::new());
        }
        if b.event {
            return Err(", as an event a mode reads rather than a switch".into());
        }
        if b.on != Gesture::Toggle {
            return Err(format!(
                " on `{}`: only a `toggle` leaves for as long as the press lasts, so the task \
                 reads the press",
                b.on
            ));
        }
        let warm = self.state(mode).is_some_and(|s| s.warm_start);
        let matches_here = |w: &When| match w {
            When::InState(s) => s == mode,
            When::WarmStartReached => warm,
            _ => true,
        };
        let keeps = |r: &Rule| {
            matches_here(&r.when)
                && match &r.enter {
                    Enter::State(s) => s == mode,
                    Enter::Initial => self.initial_state == mode,
                    Enter::WarmStartRef => self.warm_start_ref == mode,
                    Enter::Stay => true,
                }
        };
        let Some(i) = self.rules.iter().position(|r| matches!(&r.when, When::Button(n) if *n == b.name))
        else {
            return Err(" with no rule entering anything on it, so the cascade stays here".into());
        };
        if keeps(&self.rules[i]) {
            return Err(format!(" and keeps the cascade in '{mode}' itself, so the press stays here"));
        }
        match self.rules[..i].iter().position(keeps) {
            None => Ok(()),
            Some(j) => Err(format!(
                " below rule {} (`{}`), which keeps the cascade in '{mode}' first, so the press \
                 stays here",
                j + 1,
                self.rules[j].when
            )),
        }
    }
}

// ── Parsing ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct RawRule {
    when: String,
    enter: String,
}

/// Unknown keys are refused rather than skipped: a key misspelt, or one this
/// crate does not read, would otherwise load as if it were not there -- a
/// setting that says what it does and does nothing.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFsm {
    #[serde(default, rename = "state")]
    states: Vec<StateConfig>,
    #[serde(default, rename = "rule")]
    rules: Vec<RawRule>,
    #[serde(default, rename = "button")]
    buttons: Vec<ButtonBinding>,
    #[serde(default)]
    exclusive: bool,
    initial_state: String,
    warm_start_ref: String,
    safe_state: String,
    tilt_limit: f32,
    state_timeout_ms: u64,
    command_timeout_ms: u64,
    mode_switch_ramp_s: f32,
    pose_reach_tol: f32,
    warm_start_duration_s: f32,
    ramp_kp: f32,
    ramp_kd: f32,
    #[serde(default)]
    click_window_ms: Option<u64>,
}

#[derive(Deserialize)]
struct RawRoot {
    fsm: RawFsm,
}

#[derive(Debug)]
pub enum Error {
    Toml(String),
    UnknownCondition(String),
    /// A rule names a state that does not exist. It would never be entered, and
    /// nothing at runtime would say why.
    UnknownState { rule: usize, name: String },
    /// No rule, or a rule other than the last one, is `always`.
    NotTotal(String),
    /// A rule names a button no `[[fsm.button]]` declares, so nothing can latch
    /// it. The rule is dead, and a dead rule looks exactly like one that is
    /// merely unused.
    ///
    /// There is no `[fsm.key]` table any more -- a control became a button plus
    /// a gesture, and a key is one of the two sources a single binding may name.
    UnknownButton(String),
    /// Two entries name the same pad button; which one latched would depend on map
    /// order.
    DuplicateButton(String),
    /// A binding with neither a pad button nor a key. Nothing can press it.
    UnreachableButton(String),
    /// A control used as a modifier, bound to `hold`. It is on for the whole
    /// gesture it modifies, so the rule reading it fires throughout.
    ModifierHeld(String),
    /// A rule reads `gripper_active`, which no host can supply.
    NoGripperInput,
    /// A state's `hook` cannot be built: no model to hook, or no task to look
    /// its library up by.
    Hook(String),
    /// A `[fsm.button]` entry names a pad button the wire does not carry.
    UnknownPadButton { button: String, pad: String },
    /// A key latches a button no rule reacts to, so pressing it does nothing.
    DeadKey { key: String, button: String },
    /// A binding names a key no host reports.
    UnknownKey { button: String, key: String },
    /// A binding with both a pad button and a key: the two devices are two
    /// bindings now, each with its own gesture, modifier and `from`.
    BothSources(String),
    /// A `with` that is not a modifier of the binding's own device: a pad
    /// button on a key binding, a key or an unknown word on either.
    UnknownModifier { button: String, with: String },
    /// `from` naming a state no `[[fsm.state]]` declares: a binding that could
    /// never be pressed there, and would look like it could.
    UnknownFrom { button: String, state: String },
    /// A `leaves` that cannot do what it says.
    Leaves(String),
    /// Two states share a name; which one you got would depend on scan order.
    DuplicateState(String),
    /// The safe state exempting itself from the tilt check would leave no
    /// attitude protection anywhere.
    SafeStateSkipsTilt,
    /// A click gesture with no window to count it in, or a window of zero.
    ClickWindow(String),
    /// One control bound with a click gesture and with a gesture that reads a
    /// single press. Every click would be that gesture first.
    ClickBeside { control: String, click: String, edge: String },
    /// A click gesture on a control that modifies another binding. A click
    /// fires before the chord that would spend the modifier is known.
    ClickOnModifier { control: String, button: String },
    Missing(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Toml(e) => write!(f, "{e}"),
            Error::UnknownCondition(c) => write!(
                f,
                "'{c}' is not a condition this FSM knows. The vocabulary is closed on \
                 purpose: an unrecognised word here would be a rule that never fires, \
                 which looks exactly like a rule that never needed to"
            ),
            Error::UnknownState { rule, name } => write!(
                f,
                "rule {rule} enters '{name}', which no [[fsm.state]] declares"
            ),
            Error::NotTotal(what) => write!(
                f,
                "{what}. The last rule must be `when = \"always\"` and no earlier rule \
                 may be: with that, 'no rule matched' cannot happen and the cascade is \
                 total by construction"
            ),
            Error::DuplicateState(n) => write!(f, "two states are named '{n}'"),
            Error::UnknownButton(b) => write!(
                f,
                "a rule names button '{b}', which no [[fsm.button]] declares. \
                 That rule can never fire"
            ),
            Error::DeadKey { key, button } => write!(
                f,
                "'{key}' is bound to '{button}', which no rule reacts to. Pressing \
                 it would do nothing, and a control that does nothing is \
                 indistinguishable from one that is broken"
            ),
            Error::DuplicateButton(id) => write!(
                f,
                "two entries both latch on pad button '{id}'; which one won would \
                 depend on iteration order"
            ),
            Error::UnknownPadButton { button, pad } => write!(
                f,
                "'{button}' latches on '{pad}', which the robot's gamepad service does \
                 not publish. It publishes {:?}",
                pad_buttons()
            ),
            Error::UnknownKey { button, key } => write!(
                f,
                "'{button}' latches on key '{key}', which no host reports. The \
                 dictionary is the whole list -- {:?} -- and says what each one costs \
                 in MuJoCo's viewer. `python -m controller --vocabulary`",
                crate::vocabulary::keys()
            ),
            Error::BothSources(name) => write!(
                f,
                "'{name}' has both a pad button and a key. The pad and the keyboard are \
                 two paths since 2026-09-29: write two [[fsm.button]] entries with the \
                 same name, one for each, and they share its latch"
            ),
            Error::UnknownModifier { button, with } => write!(
                f,
                "'{button}' is modified by '{with}', which is not a modifier of its \
                 device: a pad binding's `with` is a pad button, a key binding's is one \
                 of {:?}",
                crate::vocabulary::modifiers()
            ),
            Error::UnknownFrom { button, state } => write!(
                f,
                "'{button}' may be pressed from '{state}', which no [[fsm.state]] \
                 declares, so that part of its `from` can never apply"
            ),
            Error::Leaves(m) => f.write_str(m),
            Error::UnreachableButton(name) => write!(
                f,
                "'{name}' has neither a pad button nor a key, so nothing can press it"
            ),
            Error::ModifierHeld(name) => write!(
                f,
                "'{name}' is `hold` on a control that also modifies another binding. \
                 A modifier is down for the whole gesture it modifies, so this would \
                 be on throughout it"
            ),
            Error::Hook(m) => f.write_str(m),
            Error::NoGripperInput => f.write_str(
                "a rule reads `gripper_active`, and no host supplies it: every one \
                 passes `false`. The gripper state came from `robot_control`, which \
                 this controller stopped reading when it started reading the pad \
                 itself -- the pad carries no gripper. So the rule can never fire, \
                 and a dead rule looks exactly like one that is merely unused. \
                 Wire an input first, then delete this check",
            ),
            Error::SafeStateSkipsTilt => f.write_str(
                "the safe state sets skip_tilt_check; exempting it would leave no \
                 attitude protection anywhere",
            ),
            Error::ClickWindow(m) => f.write_str(m),
            Error::ClickBeside { control, click, edge } => write!(
                f,
                "'{control}' is bound with a click gesture ({click}) and with one \
                 that reads a single press ({edge}): every click would be that \
                 press first. Bind the single press as `single`, which waits out \
                 the click window when a longer click shares the control"
            ),
            Error::ClickOnModifier { control, button } => write!(
                f,
                "'{button}' counts clicks on '{control}', which also modifies another \
                 binding. A click fires before the chord that would spend the \
                 modifier is known, so reaching for the chord would fire it"
            ),
            Error::Missing(w) => write!(f, "missing {w}"),
        }
    }
}

impl std::error::Error for Error {}

impl FsmConfig {
    pub fn parse(text: &str) -> Result<Self, Error> {
        let raw: RawRoot = toml::from_str(text).map_err(|e| Error::Toml(e.to_string()))?;
        let raw = raw.fsm;

        for (i, s) in raw.states.iter().enumerate() {
            if raw.states[..i].iter().any(|o| o.name == s.name) {
                return Err(Error::DuplicateState(s.name.clone()));
            }
        }
        let named = |n: &str| raw.states.iter().any(|s| s.name == n);
        for (what, n) in [
            ("fsm.initial_state", &raw.initial_state),
            ("fsm.warm_start_ref", &raw.warm_start_ref),
            ("fsm.safe_state", &raw.safe_state),
        ] {
            if !named(n) {
                return Err(Error::Missing(format!("a state named '{n}' for {what}")));
            }
        }
        if raw.states.iter().any(|s| s.name == raw.safe_state && s.skip_tilt_check) {
            return Err(Error::SafeStateSkipsTilt);
        }
        for s in raw.states.iter().filter(|s| s.hook.is_some()) {
            if !s.has_model() {
                return Err(Error::Hook(format!(
                    "state '{}' has a hook and no model: a hook is part of a policy's loop, \
                     and this state runs none",
                    s.name
                )));
            }
            if s.task.is_empty() {
                return Err(Error::Hook(format!(
                    "state '{}' has a hook and no task: a hook is found by the task id the \
                     bundler records beside it",
                    s.name
                )));
            }
        }

        if raw.rules.is_empty() {
            return Err(Error::NotTotal("this FSM declares no rules".into()));
        }
        let mut rules = Vec::with_capacity(raw.rules.len());
        for (i, r) in raw.rules.iter().enumerate() {
            let when = When::parse(&r.when)?;
            let last = i + 1 == raw.rules.len();
            if (when == When::Always) != last {
                return Err(Error::NotTotal(if last {
                    format!("the last rule is `{}`, not `always`", r.when)
                } else {
                    format!("rule {i} is `always`, so no rule after it can ever fire")
                }));
            }
            let enter = match r.enter.as_str() {
                "@initial" => Enter::Initial,
                "@warm_start_ref" => Enter::WarmStartRef,
                "@stay" => Enter::Stay,
                name => {
                    if !named(name) {
                        return Err(Error::UnknownState { rule: i, name: name.to_string() });
                    }
                    Enter::State(name.to_string())
                }
            };
            if let When::InState(name) = &when {
                if !named(name) {
                    return Err(Error::UnknownState { rule: i, name: name.clone() });
                }
            }
            rules.push(Rule { when, enter });
        }

        // Every check here is a binding that loads and then does nothing.
        let known_pad = crate::vocabulary::is_button;
        let known_key = |k: &str| crate::vocabulary::is_key(k) || crate::vocabulary::is_modifier(k);
        let mut by_control = std::collections::BTreeMap::new();
        for b in &raw.buttons {
            match (&b.pad, &b.key) {
                (None, None) => return Err(Error::UnreachableButton(b.name.clone())),
                (Some(_), Some(_)) => return Err(Error::BothSources(b.name.clone())),
                _ => {}
            }
            if let Some(pad) = b.pad.as_deref() {
                if !known_pad(pad) {
                    return Err(Error::UnknownPadButton {
                        button: b.name.clone(),
                        pad: pad.to_string(),
                    });
                }
            }
            // The same check for the key, which did not have one. `pad` and
            // `with` were validated against the dictionary from the start and
            // `key` took any string at all, so a typo loaded, bound nothing,
            // and looked exactly like a rule that had not fired.
            if let Some(key) = b.key.as_deref() {
                if !known_key(key) {
                    return Err(Error::UnknownKey {
                        button: b.name.clone(),
                        key: key.to_string(),
                    });
                }
            }
            // A modifier of the binding's own device: a pad button held for a
            // pad binding, `ctrl` / `shift` / `alt` for a key. A pad button
            // modifying a key would be the keyboard reaching through the pad
            // again, which is what 2026-09-29 retired.
            if let Some(with) = b.with.as_deref() {
                let fits = if b.pad.is_some() {
                    known_pad(with) && !crate::vocabulary::dpad_buttons().contains(&with)
                } else {
                    crate::vocabulary::is_modifier(with)
                };
                if !fits {
                    return Err(if b.pad.is_some() && !known_pad(with) {
                        Error::UnknownPadButton { button: b.name.clone(), pad: with.to_string() }
                    } else {
                        Error::UnknownModifier { button: b.name.clone(), with: with.to_string() }
                    });
                }
            }
            for state in b.from.iter().flatten() {
                if !named(state) {
                    return Err(Error::UnknownFrom { button: b.name.clone(), state: state.clone() });
                }
            }
            // One control, one gesture, one modifier -- twice over and which
            // one won would depend on iteration order. The same control with
            // *different* gestures is the point of the design and is fine:
            // `A` on `rise` and `A` on `fall` is press-to-arm, release-to-go.
            let control = (b.pad.clone(), b.key.clone(), b.on, b.with.clone());
            if let Some(other) = by_control.insert(control, b.name.clone()) {
                return Err(Error::DuplicateButton(format!("{other} and {}", b.name)));
            }
        }

        // `leaves`: a moment that lets go of latches. Each name has to latch,
        // or the leave does nothing; the leave itself is no latch and no event,
        // since it is never active and nothing could read it.
        for b in raw.buttons.iter().filter(|b| !b.leaves.is_empty()) {
            if b.event || matches!(b.on, Gesture::Toggle | Gesture::Hold) {
                return Err(Error::Leaves(format!(
                    "'{}' leaves {:?} on `{}`{}: a leave is a moment -- `rise`, `fall` or a \
                     click -- that releases latches, and is never itself on",
                    b.name,
                    b.leaves,
                    b.on,
                    if b.event { " as an event" } else { "" }
                )));
            }
            for target in &b.leaves {
                let latching = raw.buttons.iter().any(|o| &o.name == target && o.latches());
                if !latching {
                    return Err(Error::Leaves(format!(
                        "'{}' leaves '{target}', which no binding latches, so it would \
                         let go of nothing",
                        b.name
                    )));
                }
            }
        }

        // A modifier is consumed by what it modifies, so a control used as one
        // cannot also be a `hold`: it would be on for the whole gesture it is
        // modifying, and the rule reading it would fire throughout.
        let modifiers: std::collections::BTreeSet<&str> =
            raw.buttons.iter().filter_map(|b| b.with.as_deref()).collect();
        for b in &raw.buttons {
            if b.on == Gesture::Hold && modifiers.contains(b.control_name()) {
                return Err(Error::ModifierHeld(b.name.clone()));
            }
        }

        // Clicks. Counted per control -- one pad button or key, under one
        // modifier -- so everything bound to that control has to be counting
        // too: a `toggle` beside a `double` would flip on the double's first
        // press. And counted on a timestamp, so a config with a click has to
        // say how long the window is; a default would be a gesture nobody chose.
        let clicking: Vec<&ButtonBinding> =
            raw.buttons.iter().filter(|b| b.on.clicks().is_some()).collect();
        for c in &clicking {
            if let Some(e) = raw
                .buttons
                .iter()
                .find(|b| b.on.clicks().is_none() && b.control() == c.control())
            {
                return Err(Error::ClickBeside {
                    control: c.shown(),
                    click: format!("'{}' on `{}`", c.name, c.on),
                    edge: format!("'{}' on `{}`", e.name, e.on),
                });
            }
            if modifiers.contains(c.control_name()) {
                return Err(Error::ClickOnModifier {
                    control: c.control_name().to_string(),
                    button: c.name.clone(),
                });
            }
        }
        let click_window_us = match (raw.click_window_ms, clicking.first()) {
            (Some(0), _) => {
                return Err(Error::ClickWindow(
                    "fsm.click_window_ms is 0: no second press could ever count".into(),
                ))
            }
            (None, Some(c)) => {
                return Err(Error::ClickWindow(format!(
                    "'{}' is `{}`, and counting clicks needs a window: set \
                     fsm.click_window_ms, how long after one press the next still \
                     counts",
                    c.name, c.on
                )))
            }
            (w, _) => w.map(|ms| ms * 1000),
        };

        // The same rule as an unlatchable button, for the same reason.
        if rules.iter().any(|r| r.when == When::GripperActive) {
            return Err(Error::NoGripperInput);
        }

        let declared: std::collections::BTreeSet<&str> =
            raw.buttons.iter().map(|b| b.name.as_str()).collect();
        for rule in &rules {
            if let When::Button(name) = &rule.when {
                if !declared.contains(name.as_str()) {
                    return Err(Error::UnknownButton(name.clone()));
                }
                // A leave is never active, so a rule on it could never fire.
                if raw.buttons.iter().any(|b| &b.name == name && !b.leaves.is_empty()) {
                    return Err(Error::Leaves(format!(
                        "a rule reads '{name}', which leaves latches and is never itself on, \
                         so the rule could never fire"
                    )));
                }
            }
        }

        let reacted: std::collections::BTreeSet<&str> = rules
            .iter()
            .filter_map(|r| match &r.when {
                When::Button(name) => Some(name.as_str()),
                _ => None,
            })
            .collect();
        for b in raw.buttons.iter().filter(|b| b.leaves.is_empty()) {
            if !b.event && !reacted.contains(b.name.as_str()) {
                return Err(Error::DeadKey {
                    key: b.control_name().to_string(),
                    button: b.name.clone(),
                });
            }
        }

        let mut buttons = raw.buttons;
        for b in buttons.iter_mut().filter(|b| b.on.clicks().is_some()) {
            b.click_window_us = click_window_us.unwrap_or(0);
        }

        Ok(FsmConfig {
            states: raw.states,
            rules,
            buttons,
            exclusive: raw.exclusive,
            initial_state: raw.initial_state,
            warm_start_ref: raw.warm_start_ref,
            safe_state: raw.safe_state,
            tilt_limit: raw.tilt_limit,
            state_timeout_us: raw.state_timeout_ms * 1000,
            command_timeout_us: raw.command_timeout_ms * 1000,
            mode_switch_ramp_s: raw.mode_switch_ramp_s,
            pose_reach_tol: raw.pose_reach_tol,
            warm_start_duration_s: raw.warm_start_duration_s,
            ramp_kp: raw.ramp_kp,
            ramp_kd: raw.ramp_kd,
            click_window_us,
        })
    }
}

#[cfg(test)]
pub(crate) const EXAMPLE: &str = r#"
[fsm]
initial_state = "warmstart"
warm_start_ref = "locomotion"
safe_state = "safe"
tilt_limit = 0.8
state_timeout_ms = 100
command_timeout_ms = 500
mode_switch_ramp_s = 1.0
pose_reach_tol = 0.10
warm_start_duration_s = 2.0
ramp_kp = 0.15
ramp_kd = 0.01

# What the operator can ask for, and which control asks it: one device each,
# and the same name for the same thing, so the pad's A and the keyboard's
# keypad 1 are one latch.
[[fsm.button]]
name = "jump"
pad = "A"
on = "toggle"

[[fsm.button]]
name = "jump"
key = "keypad_1"
on = "toggle"

# A dead-man switch: `carry` while LB is down, and not a tick longer.
[[fsm.button]]
name = "carry"
pad = "LB"
on = "hold"

[[fsm.state]]
name = "warmstart"
warm_start = true
kp = 0.1
kd = 0.002

[[fsm.state]]
name = "safe"
hold_current = true
kp = 0.0
kd = 0.01

[[fsm.state]]
name = "locomotion"
model = "locomotion.rknn"
kp_scale = 0.015
kd_scale = 0.015

[[fsm.state]]
name = "carry"
model = "locomotion_5foot.rknn"
kp_scale = 0.015
kd_scale = 0.015

[[fsm.state]]
name = "jump"
model = "jump.rknn"
skip_tilt_check = true
kp_scale = 0.015
kd_scale = 0.015

[[fsm.rule]]
when = "feedback_stale"
enter = "safe"

[[fsm.rule]]
when = "tilted"
enter = "safe"

[[fsm.rule]]
when = "in_state:safe"
enter = "@initial"

[[fsm.rule]]
when = "warm_start_reached"
enter = "@warm_start_ref"

# Until it is. A warm start is not interruptible: the operator's buttons are
# below this line and the robot is still ramping to a known pose.
[[fsm.rule]]
when = "in_state:warmstart"
enter = "@stay"

[[fsm.rule]]
when = "button:jump"
enter = "jump"

[[fsm.rule]]
when = "button:carry"
enter = "carry"

[[fsm.rule]]
when = "always"
enter = "locomotion"
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn a_key_the_dictionary_does_not_know_is_refused() {
        // This check did not exist. `pad` and `with` were validated against the
        // dictionary from the start and `key` took any string at all, so a
        // binding on `KeyW` or `keypad_A` loaded, bound nothing, and looked
        // exactly like a rule that had not fired -- the failure the dictionary
        // exists to make impossible, in the one field it did not cover.
        let with_key = |k: &str| {
            format!("{EXAMPLE}\n[[fsm.button]]\nname = \"jump\"\nkey = \"{k}\"\non = \"toggle\"\n")
        };
        for bad in ["KeyW", "Numpad1", "keypad_A", "keypad_", "1"] {
            let e = FsmConfig::parse(&with_key(bad)).unwrap_err();
            assert!(
                matches!(&e, Error::UnknownKey { key, .. } if key == bad),
                "{bad}: {e}"
            );
        }

        // The control: every name the dictionary does list parses. Without
        // this, a check that refused everything would pass the loop above.
        // `EXAMPLE` already binds `jump` on keypad_1 itself, and the same key
        // with the same gesture twice is refused for its own reason.
        for good in crate::vocabulary::keys().into_iter().filter(|k| *k != "keypad_1") {
            // A second binding with the same name and a different control is
            // allowed and is the point.
            FsmConfig::parse(&with_key(good))
                .unwrap_or_else(|e| panic!("{good} is in the dictionary and was refused: {e}"));
        }
    }

    #[test]
    fn a_key_carries_the_codes_two_hosts_report_it_under() {
        // The mapping three files said was the host's job without saying how.
        assert_eq!(crate::vocabulary::key_codes("keypad_1"), Some((321, "Numpad1")));
        assert_eq!(crate::vocabulary::key_codes("keypad_enter"), Some((335, "NumpadEnter")));
        assert_eq!(crate::vocabulary::key_codes("Numpad1"), None, "that is the other name");
    }

    /// A key `[fsm]` does not read is refused, not skipped: a table of keys
    /// parsed as nothing would switch nothing and say so nowhere, and a
    /// misspelt setting would sit there looking set. The control group is
    /// `EXAMPLE` without it, which loads.
    #[test]
    fn a_key_the_fsm_does_not_read_is_refused() {
        super::FsmConfig::parse(super::EXAMPLE).unwrap();
        for extra in ["[fsm.keyboard]\nkey_space = \"A\"\n", "[fsm.exclusiv]\n"] {
            let e = super::FsmConfig::parse(&format!("{}\n{extra}", super::EXAMPLE)).unwrap_err();
            assert!(e.to_string().contains("unknown field"), "{extra}: {e}");
        }
    }

    /// A binding is one device's. Both on one entry is refused, because the
    /// two do not have to agree -- gesture, modifier and `from` are each
    /// device's own -- and one entry would make them.
    #[test]
    fn a_binding_is_the_pad_s_or_the_keyboard_s() {
        let both = super::EXAMPLE.replacen(
            "name = \"jump\"\npad = \"A\"\n",
            "name = \"jump\"\npad = \"A\"\nkey = \"key_space\"\n",
            1,
        );
        assert_ne!(both, super::EXAMPLE, "the fixture did not take the key");
        let e = super::FsmConfig::parse(&both).unwrap_err();
        assert!(matches!(&e, super::Error::BothSources(n) if n == "jump"), "{e}");
    }

    /// A key switch is modified by a modifier and a pad switch by a pad button:
    /// `ctrl` on a pad binding, or `menu` on a key, would be one device reaching
    /// through the other. A modifier's own name is a key a switch may be on --
    /// Ctrl on its own leaves a dance. The control groups load.
    #[test]
    fn each_device_is_modified_by_its_own() {
        let with = |extra: &str| {
            super::FsmConfig::parse(&format!(
                "{}\n[[fsm.button]]\nname = \"jump\"\n{extra}\n",
                super::EXAMPLE
            ))
        };
        with("key = \"key_1\"\nwith = \"ctrl\"\non = \"toggle\"").unwrap();
        with("pad = \"dpad_up\"\nwith = \"menu\"\non = \"toggle\"").unwrap();
        with("key = \"ctrl\"\non = \"fall\"").unwrap();

        let e = with("key = \"key_1\"\nwith = \"menu\"\non = \"toggle\"").unwrap_err();
        assert!(matches!(&e, super::Error::UnknownModifier { with, .. } if with == "menu"), "{e}");
        let e = with("pad = \"dpad_up\"\nwith = \"ctrl\"\non = \"toggle\"").unwrap_err();
        assert!(matches!(&e, super::Error::UnknownPadButton { pad, .. } if pad == "ctrl"), "{e}");
        let e = with("key = \"key_1\"\nwith = \"key_left_ctrl\"\non = \"toggle\"").unwrap_err();
        assert!(matches!(&e, super::Error::UnknownModifier { .. }), "a key is not a modifier: {e}");
    }

    /// `from` names states and `leaves` names latches, and a leave is a moment
    /// no rule reads. Each refusal is a binding that loads and could never do
    /// what it says; the first parse is the control group.
    #[test]
    fn from_and_leaves_name_what_is_there() {
        let leave = "[[fsm.button]]\nname = \"out\"\npad = \"menu\"\non = \"fall\"\n\
                     leaves = [\"jump\"]\nfrom = [\"jump\"]\n";
        let with = |extra: &str| super::FsmConfig::parse(&format!("{}\n{extra}\n", super::EXAMPLE));
        let cfg = with(leave).unwrap();
        let out = cfg.buttons.iter().find(|b| b.name == "out").unwrap();
        assert!(out.live_in("jump") && !out.live_in("locomotion"));
        assert!(!out.latches(), "a leave latches nothing");

        let e = with(&leave.replace("from = [\"jump\"]", "from = [\"backflip\"]")).unwrap_err();
        assert!(matches!(&e, super::Error::UnknownFrom { state, .. } if state == "backflip"), "{e}");
        let e = with(&leave.replace("leaves = [\"jump\"]", "leaves = [\"carry\"]")).unwrap_err();
        assert!(e.to_string().contains("no binding latches"), "carry is a hold: {e}");
        let e = with(&leave.replace("on = \"fall\"", "on = \"toggle\"")).unwrap_err();
        assert!(e.to_string().contains("a leave is a moment"), "{e}");
        let read = format!(
            "{leave}\n[[fsm.rule]]\nwhen = \"button:out\"\nenter = \"jump\"\n"
        );
        let e = super::FsmConfig::parse(&super::EXAMPLE.replacen(
            "[[fsm.rule]]\nwhen = \"button:jump\"",
            &format!("{read}\n[[fsm.rule]]\nwhen = \"button:jump\""),
            1,
        ))
        .unwrap_err();
        assert!(e.to_string().contains("never itself on"), "{e}");
    }

    use super::*;

    #[test]
    fn the_example_parses_and_keeps_its_order() {
        let cfg = FsmConfig::parse(EXAMPLE).unwrap();
        assert_eq!(cfg.states.len(), 5);
        let order: Vec<String> = cfg.rules.iter().map(|r| r.when.to_string()).collect();
        assert_eq!(
            order,
            [
                "feedback_stale",
                "tilted",
                "in_state:safe",
                "warm_start_reached",
                "in_state:warmstart",
                "button:jump",
                "button:carry",
                "always",
            ],
            "the cascade order is the design; the file is where it is written"
        );
        assert_eq!(cfg.rules[2].enter, Enter::Initial);
        assert_eq!(cfg.state("jump").unwrap().skip_tilt_check, true);
        assert_eq!(cfg.state("safe").unwrap().has_model(), false);
    }

    /// Every one of these has a plausible silent outcome, which is why each is a
    /// load error. A rule that never fires is indistinguishable from a rule that
    /// never needed to.
    #[test]
    fn a_configuration_that_could_fail_quietly_does_not_load() {
        let cases: [(&str, &str, &str); 9] = [
            ("wiggle", r#"when = "button:carry""#, r#"when = "wiggle""#),
            ("no [[fsm.state]] declares", r#"enter = "jump""#, r#"enter = "backflip""#),
            (
                "not `always`",
                "[[fsm.rule]]\nwhen = \"always\"\nenter = \"locomotion\"\n",
                "",
            ),
            (
                "no rule after it can ever fire",
                r#"when = "feedback_stale""#,
                r#"when = "always""#,
            ),
            // Anchored to the state block: `carry` is also a binding name now,
            // and it appears first, so a bare rename would break the rule
            // instead and this case would pass on the wrong error.
            ("two states are named", "[[fsm.state]]\nname = \"carry\"",
             "[[fsm.state]]\nname = \"locomotion\""),
            // A rule naming a button no binding declares.
            ("never fire", "when = \"button:jump\"", "when = \"button:jump_elsewhere\""),
            // A pad button the wire does not carry. `view` is the one worth
            // naming: the pad has it and the service deliberately does not
            // send it, so this binds a button nobody can ever press.
            ("does not publish", "pad = \"A\"", "pad = \"view\""),
            // A gesture outside the four. The vocabulary is closed for the
            // same reason the rule words are: a typo that parses is a binding
            // that quietly does something else.
            ("unknown variant", "on = \"toggle\"", "on = \"longpress\""),
            (
                "no attitude protection",
                "name = \"safe\"\nhold_current = true",
                "name = \"safe\"\nskip_tilt_check = true\nhold_current = true",
            ),
        ];
        for (message, from, to) in cases {
            let broken = EXAMPLE.replacen(from, to, 1);
            let err = FsmConfig::parse(&broken)
                .err()
                .unwrap_or_else(|| panic!("expected {message:?} to be refused"));
            assert!(
                err.to_string().contains(message),
                "for {message:?} got: {err}"
            );
        }
    }

    /// Two bindings, one name: the pad's A and the keyboard's keypad 1 are the
    /// jump's two ways in. Either alone loads; neither is refused, since a name
    /// nothing presses is a rule that can never fire.
    #[test]
    fn two_bindings_share_one_name() {
        let cfg = FsmConfig::parse(EXAMPLE).unwrap();
        let jump: Vec<_> = cfg.buttons.iter().filter(|b| b.name == "jump").collect();
        assert_eq!(jump.len(), 2);
        assert_eq!((jump[0].pad.as_deref(), jump[0].key.as_deref()), (Some("A"), None));
        assert_eq!((jump[1].pad.as_deref(), jump[1].key.as_deref()), (None, Some("keypad_1")));

        let key = "[[fsm.button]]\nname = \"jump\"\nkey = \"keypad_1\"\non = \"toggle\"\n";
        let pad = "[[fsm.button]]\nname = \"jump\"\npad = \"A\"\non = \"toggle\"\n";
        for drop in [key, pad] {
            let one = EXAMPLE.replacen(drop, "", 1);
            assert_ne!(one, EXAMPLE, "{drop:?} is not in the example");
            FsmConfig::parse(&one).unwrap_or_else(|e| panic!("dropping {drop:?}: {e}"));
        }
        let neither = EXAMPLE.replacen("pad = \"A\"\n", "", 1);
        let e = FsmConfig::parse(&neither).unwrap_err().to_string();
        assert!(e.contains("nothing can press it"), "{e}");
    }

    /// Press to arm, release to go: one control, two gestures, two meanings.
    ///
    /// This is the case the old table could not express at all -- it latched on
    /// a rising edge and that was the whole vocabulary -- and it is the one the
    /// C++ used for a jump.
    #[test]
    fn one_control_can_carry_two_gestures() {
        let two = EXAMPLE.replacen(
            "[[fsm.button]]\nname = \"jump\"\npad = \"A\"\non = \"toggle\"\n",
            "[[fsm.button]]\nname = \"jump\"\npad = \"A\"\non = \"rise\"\n\n\
             [[fsm.button]]\nname = \"jump_go\"\npad = \"A\"\non = \"fall\"\n",
            1,
        );
        // `jump_go` needs a rule or it is dead, which is the next check along.
        let with_rule = two.replacen(
            "[[fsm.rule]]\nwhen = \"button:jump\"",
            "[[fsm.rule]]\nwhen = \"button:jump_go\"\nenter = \"jump\"\n\n\
             [[fsm.rule]]\nwhen = \"button:jump\"",
            1,
        );
        let cfg = FsmConfig::parse(&with_rule).unwrap();
        let gestures: Vec<_> = cfg
            .buttons
            .iter()
            .filter(|b| b.pad.as_deref() == Some("A"))
            .map(|b| b.on)
            .collect();
        assert_eq!(gestures, [Gesture::Rise, Gesture::Fall]);

        // The same control, same gesture, twice: which one won would depend on
        // iteration order.
        let same = two.replacen("on = \"fall\"", "on = \"rise\"", 1);
        assert!(FsmConfig::parse(&same).unwrap_err().to_string().contains("iteration order"));
    }

    /// A click needs a window to be counted in, and a control of its own.
    ///
    /// Each refusal here is a config that would load and then click wrongly:
    /// no window is a gesture nobody chose, a window of zero is one no second
    /// press can reach, a `toggle` beside a `double` flips on the double's
    /// first press, and a click on a modifier fires as a hand reaches for the
    /// chord. The control groups are the same configs made right.
    #[test]
    fn a_click_needs_a_window_and_a_control_of_its_own() {
        let jump = "[[fsm.button]]\nname = \"jump\"\npad = \"A\"\non = \"toggle\"\n";
        let double = EXAMPLE.replacen(jump, &jump.replace("toggle", "double"), 1);
        let windowed = |text: &str, ms: u64| {
            text.replacen("ramp_kd = 0.01\n", &format!("ramp_kd = 0.01\nclick_window_ms = {ms}\n"), 1)
        };
        let err = |text: &str| FsmConfig::parse(text).unwrap_err().to_string();

        assert!(err(&double).contains("click_window_ms"), "no window");
        assert!(err(&windowed(&double, 0)).contains("is 0"), "a window of zero");
        let cfg = FsmConfig::parse(&windowed(&double, 300)).unwrap();
        assert_eq!(cfg.click_window_us, Some(300_000));
        let window = |name: &str| cfg.buttons.iter().find(|b| b.name == name).unwrap().click_window_us;
        assert_eq!((window("jump"), window("carry")), (300_000, 0), "on the click, and only there");

        // Beside a gesture that reads one press, on the same control.
        let beside = |on: &str| {
            windowed(&double, 300).replacen(
                "[[fsm.state]]",
                &format!(
                    "[[fsm.button]]\nname = \"once\"\npad = \"A\"\n\
                     on = \"{on}\"\nevent = true\n\n[[fsm.state]]"
                ),
                1,
            )
        };
        assert!(err(&beside("toggle")).contains("reads a single press"));
        assert!(FsmConfig::parse(&beside("single")).is_ok(), "`single` is the way to share it");

        // On a control that modifies another binding.
        let modifier = windowed(&double, 300).replacen(
            "[[fsm.state]]",
            "[[fsm.button]]\nname = \"up\"\npad = \"dpad_up\"\nwith = \"A\"\non = \"rise\"\n\
             event = true\n\n[[fsm.state]]",
            1,
        );
        assert!(err(&modifier).contains("also modifies another binding"));
    }

    /// A word that parses and can never be true.
    ///
    /// `gripper_active` is in the cascade vocabulary and every host passes
    /// `false` for it: the gripper state came from `robot_control`, which this
    /// controller stopped reading when it started reading the pad, and the pad
    /// carries no gripper. So a rule using it loads and never fires -- the same
    /// failure as a rule naming a button nothing latches, which this file has
    /// refused all along.
    ///
    /// The **word** still parses, because the vocabulary is what a file may
    /// say and this one will be true again when something supplies it. What is
    /// refused is a config that depends on it today.
    #[test]
    fn a_rule_nothing_can_make_true_is_refused() {
        assert!(When::parse("gripper_active").is_ok(), "the word is still a word");

        let dead = EXAMPLE.replacen(
            r#"when = "button:carry""#, r#"when = "gripper_active""#, 1);
        let e = FsmConfig::parse(&dead).unwrap_err().to_string();
        assert!(e.contains("no host supplies it"), "{e}");
        assert!(e.contains("delete this check"), "it must say what would make it valid");

        // The control group: the same config with a reachable condition loads,
        // so this is not refusing everything.
        FsmConfig::parse(EXAMPLE).unwrap();
    }

    /// The vocabulary is closed, so this list is the whole of it. A condition
    /// added to `When` without a word here would be unreachable from a file.
    #[test]
    fn every_condition_has_a_word_and_every_word_parses() {
        for word in [
            "always",
            "feedback_stale",
            "tilted",
            "command_stale",
            "gripper_active",
            "warm_start_reached",
            "in_state:safe",
            "button:jump",
        ] {
            let parsed = When::parse(word).unwrap();
            assert_eq!(parsed.to_string(), word, "round trip");
        }
        assert!(When::parse("in_state:").is_err(), "an empty argument is a typo");
        assert!(When::parse("in_state").is_err(), "the argument is not optional");
    }

    /// A hook is part of a policy's loop and is found by its task, so a state
    /// without either cannot have one. The control group is the state with both,
    /// which has to load -- the refusals are otherwise indistinguishable from a
    /// parser that refuses every `hook`.
    #[test]
    fn a_hook_needs_a_model_and_a_task() {
        let with = |state: &str, extra: &str| {
            let at = format!("name = \"{state}\"\n");
            assert_eq!(EXAMPLE.matches(&at).count(), 1, "{state} is declared once");
            EXAMPLE.replace(&at, &format!("{at}{extra}"))
        };
        let both = with("locomotion", "task = \"jumper.five_foot\"\nhook = { side = \"right\" }\n");
        let cfg = FsmConfig::parse(&both).unwrap();
        let loco = cfg.state("locomotion").unwrap();
        assert_eq!(loco.task, "jumper.five_foot");
        assert_eq!(loco.hook.as_ref().unwrap()["side"].as_str(), Some("right"));

        match FsmConfig::parse(&with("locomotion", "hook = { side = \"right\" }\n")) {
            Err(Error::Hook(m)) => assert!(m.contains("no task"), "{m}"),
            other => panic!("a hook with no task: {other:?}"),
        }
        match FsmConfig::parse(&with("warmstart", "task = \"jumper.five_foot\"\nhook = {}\n")) {
            Err(Error::Hook(m)) => assert!(m.contains("no model"), "{m}"),
            other => panic!("a hook on a state that runs no model: {other:?}"),
        }
    }
}
