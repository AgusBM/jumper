//! Which mode is running, decided fresh every tick from an ordered cascade.
//!
//! Not edge-triggered: `update` is a pure function of `(config, current state,
//! inputs)` and is called every tick. Entering the state already active is a
//! no-op, so `entered_at` only moves on a real change. That is `rl-wbc-fsm`'s
//! design and it is worth keeping for a reason beyond parity -- a cascade
//! recomputed from scratch cannot be left in a state by an event that was missed,
//! which is the failure mode of every edge-triggered controller.
//!
//! The order of the rules is the design, and it now lives in the file rather
//! than in this function: see `config.rs`. What lives here is the evaluation and
//! the two guards that belong to a *state* rather than to an edge.
//!
//! ## The guard worth reading twice
//!
//! `skip_tilt_check` suspends the attitude fallback **only while its state is
//! already active**. The tick that enters such a state is still checked, so a
//! robot cannot launch from an already-tipped pose. That asymmetry is the whole
//! safety of the feature and it is one line below.

use crate::config::{Enter, FsmConfig, StateConfig, When};

/// Micros since an arbitrary origin. Injected, never read -- this module has no
/// clock, which is what lets one recorded episode replay identically in a
/// browser, on a bench and in a test.
pub type Micros = u64;

/// What the rules are evaluated against.
///
/// Facts, already reduced: whether feedback is fresh rather than when it
/// arrived, whether the pose is reached rather than what it is. The reduction
/// happens in the host, because "fresh" needs a clock and "reached" needs the
/// incoming model's default pose, and neither belongs in a rule evaluator.
#[derive(Debug, Clone, Default)]
pub struct Inputs {
    pub state_fresh: bool,
    pub command_fresh: bool,
    /// Tilt angle, rad. Compared against `tilt_limit`; ignored when the IMU is
    /// not valid, because an invalid attitude is not a level one.
    pub tilt: Option<f32>,
    /// Everything the operator is asking for, this tick. A set because
    /// bindings are independent: `LB` held while `A` is pressed is two things
    /// at once, and which one wins is the cascade's order.
    pub buttons: std::collections::BTreeSet<String>,
    pub gripper_active: bool,
    /// Every joint within `pose_reach_tol` of the active warm start's target.
    pub pose_reached: bool,
}

/// One thing the FSM did, for a host to print.
///
/// Recorded rather than printed: this crate has no stdout in a browser and no
/// business having one anywhere. A host drains these when it likes.
///
/// `rule` is the index into the configured list, which is the answer to the
/// only question an FSM ever raises -- *why* did it switch. Two rules can lead
/// to the same state for different reasons, and without this the two are
/// indistinguishable from outside.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transition {
    pub at: Micros,
    pub from: String,
    pub to: String,
    pub rule: usize,
    pub why: String,
}

pub struct Fsm {
    cfg: FsmConfig,
    current: String,
    entered_at: Micros,
    log: Vec<Transition>,
}

/// Bounded, because a host that never drains must not grow without limit and a
/// robot that transitions every tick is a bug this should survive rather than
/// compound.
const LOG_CAPACITY: usize = 64;

impl Fsm {
    /// Starts in `initial_state`, as a cold robot does.
    pub fn new(cfg: FsmConfig, now: Micros) -> Self {
        let current = cfg.initial_state.clone();
        Self { cfg, current, entered_at: now, log: Vec::new() }
    }

    pub fn config(&self) -> &FsmConfig {
        &self.cfg
    }

    pub fn current(&self) -> &StateConfig {
        // Every name that can be set here was checked against the state list at
        // load, so this cannot miss.
        self.cfg.state(&self.current).expect("current state was validated at load")
    }

    pub fn entered_at(&self) -> Micros {
        self.entered_at
    }

    /// How long the active state has been active. The warm start's timer
    /// fallback is the only thing that uses it, and it is a fallback: the
    /// hand-off proper waits on measured pose.
    pub fn elapsed_s(&self, now: Micros) -> f32 {
        now.saturating_sub(self.entered_at) as f32 / 1e6
    }

    /// Enter a state. Idempotent, so `entered_at` marks the last real change.
    fn enter(&mut self, name: &str, now: Micros, rule: usize) {
        if self.current == name {
            return;
        }
        if self.log.len() == LOG_CAPACITY {
            self.log.remove(0);
        }
        self.log.push(Transition {
            at: now,
            from: std::mem::replace(&mut self.current, name.to_string()),
            to: name.to_string(),
            rule,
            why: self.cfg.rules[rule].when.to_string(),
        });
        self.entered_at = now;
    }

    /// Every transition since the last call. Draining rather than reading, so a
    /// host that prints them cannot print one twice.
    pub fn take_log(&mut self) -> Vec<Transition> {
        std::mem::take(&mut self.log)
    }

    /// Run the cascade. Returns the state now active.
    pub fn update(&mut self, inputs: &Inputs, now: Micros) -> &StateConfig {
        // Read once, before any rule can change it: `skip_tilt_check` is a
        // property of the state that is *already* running. Reading it after a
        // transition would let a state exempt the very tick that entered it,
        // which is how a robot comes to launch from a tipped pose.
        let armed = !self.current().skip_tilt_check;

        for (index, rule) in self.cfg.rules.iter().enumerate() {
            let hit = match &rule.when {
                When::Always => true,
                When::FeedbackStale => !inputs.state_fresh,
                When::CommandStale => !inputs.command_fresh,
                // An invalid attitude is not a level one, but it is also not a
                // tilt: `tilt: None` means the IMU said nothing, and the
                // feedback-stale rule is what covers a robot with no attitude.
                When::Tilted => armed && inputs.tilt.is_some_and(|t| t > self.cfg.tilt_limit),
                When::InState(name) => &self.current == name,
                When::Button(name) => inputs.buttons.contains(name.as_str()),
                When::GripperActive => inputs.gripper_active,
                When::WarmStartReached => {
                    let s = self.current();
                    s.warm_start
                        && (inputs.pose_reached
                            || self.elapsed_s(now) >= self.cfg.warm_start_duration_s)
                }
            };
            if !hit {
                continue;
            }
            let target = match &rule.enter {
                Enter::State(name) => name.clone(),
                Enter::Initial => self.cfg.initial_state.clone(),
                Enter::WarmStartRef => self.cfg.warm_start_ref.clone(),
                Enter::Stay => return self.current(),
            };
            self.enter(&target, now, index);
            return self.current();
        }
        // Unreachable: `config::parse` refuses a rule list whose last rule is
        // not `always`. Stated rather than left to fall off the end, so that if
        // the invariant is ever weakened this is a panic with a reason and not a
        // robot holding its last state.
        unreachable!("the rule list is total; config::parse enforces it")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EXAMPLE;

    fn fsm() -> Fsm {
        Fsm::new(FsmConfig::parse(EXAMPLE).unwrap(), 0)
    }

    fn healthy() -> Inputs {
        Inputs {
            state_fresh: true,
            command_fresh: true,
            tilt: Some(0.0),
            buttons: Default::default(),
            gripper_active: false,
            pose_reached: false,
        }
    }

    /// A cold robot warm-starts, hands off on measured pose, then walks.
    #[test]
    fn the_ordinary_path() {
        let mut f = fsm();
        assert_eq!(f.current().name, "warmstart");

        f.update(&healthy(), 1_000);
        assert_eq!(f.current().name, "warmstart", "not until the pose is reached");

        let reached = Inputs { pose_reached: true, ..healthy() };
        f.update(&reached, 2_000);
        assert_eq!(f.current().name, "locomotion");

        // `carry` is `LB` on `hold` in the example: while it is down and not a
        // tick longer.
        f.update(&Inputs { buttons: ["carry".to_string()].into(), ..healthy() }, 3_000);
        assert_eq!(f.current().name, "carry");
        f.update(&healthy(), 4_000);
        assert_eq!(f.current().name, "locomotion");
    }

    /// The warm start hands off on time as well as on pose, because a hand-off
    /// that only waits on pose never happens if the gains cannot reach it.
    #[test]
    fn the_warm_start_has_a_timer_as_well_as_a_pose() {
        let mut f = fsm();
        f.update(&healthy(), 1_999_999);
        assert_eq!(f.current().name, "warmstart");
        f.update(&healthy(), 2_000_001);
        assert_eq!(f.current().name, "locomotion", "warm_start_duration_s = 2.0");
    }

    /// From any state, and before anything the operator asked for.
    #[test]
    fn the_safety_rules_fire_from_anywhere_and_first() {
        for start in ["locomotion", "carry", "jump"] {
            let mut f = fsm();
            f.current = start.to_string();

            let asked_for_a_jump = Inputs { buttons: ["jump".to_string()].into(), ..healthy() };
            f.update(&Inputs { state_fresh: false, ..asked_for_a_jump.clone() }, 10);
            assert_eq!(f.current().name, "safe", "stale feedback beats the operator, from {start}");
        }

        // Recovery is automatic and does not wait for the operator: the instant
        // feedback is healthy again, safe hands back to the initial state.
        let mut f = fsm();
        f.current = "safe".into();
        f.update(&healthy(), 100);
        assert_eq!(f.current().name, "warmstart");
    }

    /// `skip_tilt_check` protects the tick that enters, and only suspends the
    /// check afterwards. Without that asymmetry a robot launches from a pose it
    /// is already falling out of.
    #[test]
    fn a_tilt_exempt_state_cannot_be_entered_while_tilted() {
        let tilted = Inputs { tilt: Some(1.2), buttons: ["jump".to_string()].into(), ..healthy() };

        let mut f = fsm();
        f.current = "locomotion".into();
        f.update(&tilted, 10);
        assert_eq!(f.current().name, "safe", "cannot launch while tilted");

        // Already jumping, the same tilt is the jump's own and is allowed.
        let mut f = fsm();
        f.current = "jump".into();
        f.update(&tilted, 10);
        assert_eq!(f.current().name, "jump");

        // But stale feedback is never suspended, even there.
        f.update(&Inputs { state_fresh: false, ..tilted }, 20);
        assert_eq!(f.current().name, "safe");
    }

    /// The control group.
    ///
    /// Every test above passes against an `update` with the cascade written into
    /// it in the same order -- which is the arrangement `config.rs` exists to
    /// end, and is what `rl-wbc-fsm` does today. Move a rule in the file and the
    /// behaviour must move with it.
    #[test]
    fn the_cascade_is_the_file_s_rather_than_agreeing_with_it() {
        // Put the operator's jump above the tilt fallback -- the dangerous
        // ordering, which is exactly why it must be visible in the file.
        let reordered = EXAMPLE.replacen(
            "[[fsm.rule]]\nwhen = \"tilted\"\nenter = \"safe\"\n",
            "",
            1,
        )
        .replacen(
            "[[fsm.rule]]\nwhen = \"button:jump\"",
            "[[fsm.rule]]\nwhen = \"tilted\"\nenter = \"safe\"\n\n[[fsm.rule]]\nwhen = \"button:jump\"",
            1,
        );
        // Sanity: the edit did something, and the two differ only in order.
        assert_ne!(reordered, EXAMPLE);

        let tilted = Inputs { tilt: Some(1.2), buttons: ["jump".to_string()].into(), ..healthy() };

        let mut shipped = fsm();
        shipped.current = "locomotion".into();
        shipped.update(&tilted, 10);
        assert_eq!(shipped.current().name, "safe");

        let mut moved = Fsm::new(FsmConfig::parse(&reordered).unwrap(), 0);
        moved.current = "locomotion".into();
        moved.update(&tilted, 10);
        assert_eq!(
            moved.current().name, "safe",
            "tilt still precedes the button here -- only in_state:safe moved above it"
        );

        // Now the ordering that actually inverts: the button above the tilt.
        let dangerous = EXAMPLE
            .replacen("[[fsm.rule]]\nwhen = \"tilted\"\nenter = \"safe\"\n\n", "", 1)
            .replacen(
                "[[fsm.rule]]\nwhen = \"always\"",
                "[[fsm.rule]]\nwhen = \"tilted\"\nenter = \"safe\"\n\n[[fsm.rule]]\nwhen = \"always\"",
                1,
            );
        let mut f = Fsm::new(FsmConfig::parse(&dangerous).unwrap(), 0);
        f.current = "locomotion".into();
        f.update(&tilted, 10);
        assert_eq!(
            f.current().name, "jump",
            "with the tilt rule demoted below the button, a tilted robot jumps -- \
             which is why the order is in the file and not in a docstring"
        );
    }

    /// An invalid attitude is not a level one. The host passes `None`, and the
    /// robot is caught by the feedback rule rather than by a tilt of 0.
    #[test]
    fn an_absent_attitude_is_not_read_as_upright() {
        let mut f = fsm();
        f.current = "locomotion".into();
        f.update(&Inputs { tilt: None, state_fresh: false, ..healthy() }, 10);
        assert_eq!(f.current().name, "safe");
    }
}
