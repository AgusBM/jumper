//! The robot's host: DDS in, the NPU in the middle, DDS out.
//!
//! The first of the three hosts, and the one that already had its I/O. What it
//! adds to `control.rs` is everything that needs a clock, a bus or a model
//! file -- none of which the core may have, because the same core has to run in
//! a browser.
//!
//! ```text
//!   DdsIo::take_state    ─┐
//!   DdsIo::take_command  ─┼─▶ DeviceHost::step ─▶ Controller::tick
//!   a clock              ─┘                              │
//!                                            Step::Infer │ Step::Hold
//!                                     Engine::infer ◀────┘      │
//!                                            └─▶ resume ────────┤
//!                                                        DdsIo::publish
//! ```
//!
//! ## Freshness is the host's job, and it is the whole of the safety net
//!
//! Two of the FSM's rules -- `feedback_stale` and `command_stale` -- are the
//! only things standing between a cable coming loose and a robot holding its
//! last commanded pose forever. The core cannot decide them: it has no clock.
//! So the host stamps each sample as it arrives and the core is told a boolean.
//!
//! ## The stub is a mode, not an error
//!
//! `make_engine` falls back to a stub with a reason rather than failing,
//! because a controller that will not start is a robot that cannot even be told
//! to stand. The cost is that a stub returns zeros, zeros decode to the home
//! pose, and a robot standing at its home pose looks exactly like a policy that
//! has decided to stand. So `stub_reasons` exists and `step` does not hide it:
//! whoever drives this has to say so.
//!
//! **A zero action means "hold" only while the decode baseline is the home
//! pose.** A reference-residual policy decodes against a recording instead, and
//! there a zero action means "track the recording exactly", i.e. a full
//! open-loop jump off a stub.
//!
//! So `note_stub` does not only record the reason: it hands the fact to
//! `Controller::interlock_stub_engine`, which bars a mode whose contract is a
//! residual from running at all while its engine is a stub. The host owns the
//! fact, the controller owns the rule, because the contract is there.
//!
//! This paragraph used to end "this crate has no reference support yet; anything
//! that adds it must add the interlock with it". The support arrived first and
//! the interlock second, and in between a board bundle whose jump was still
//! ONNX would have played the recording open loop at 200 Hz with kp=20 --
//! push-off included -- while `stub_modes` told the operator it would hold the
//! home pose.

use std::collections::HashMap;

use crate::control::{Buttons, Controller, Step, TickInput};
use crate::fsm::Micros;
use crate::rknn::Engine;
use crate::types::{Command, MotorCommand, Pad, RobotState};

pub struct DeviceHost {
    controller: Controller,
    engines: HashMap<String, Box<dyn Engine>>,
    /// Why a mode is running a stub. Empty when every model loaded.
    stub_reasons: HashMap<String, String>,
    latch: Buttons,
    state: RobotState,
    /// The command this host was handed ready-built (`on_command`), for a mode
    /// whose contract describes no controls. Zero on the robot, which has only
    /// the pad.
    command: Command,
    pad: Pad,
    /// How each mode's contract turns the pad into its command, one operator
    /// per mode that describes controls. Empty when no policy was exported
    /// with a `controller` block -- and then the sticks do nothing, which
    /// `mjrl-fsm` says out loud rather than leaving somebody to wonder why the
    /// robot ignores them.
    operators: crate::operator::Operators,
    last_state_us: Option<Micros>,
    last_command_us: Option<Micros>,
    state_timeout_us: u64,
    command_timeout_us: u64,
}

#[derive(Debug)]
pub enum Error {
    Control(crate::control::Error),
    Infer(crate::rknn::Error),
    Dds(crate::dds::Error),
    /// The FSM asked for a mode this host has no engine for. `Controller::new`
    /// already refuses a state with no contract, so this can only mean the two
    /// were built from different configurations.
    NoEngine(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Control(e) => write!(f, "{e}"),
            Error::Infer(e) => write!(f, "{e}"),
            Error::Dds(e) => write!(f, "{e}"),
            Error::NoEngine(m) => write!(
                f,
                "no engine for mode '{m}': the controller and this host were built from \
                 different configurations"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// What a tick did, for whoever is logging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// A regime that commands without inferring: warm start, ramp, or hold.
    Held,
    /// The policy ran. `stub` is true when the "policy" was a zero action, which
    /// holds the home pose and must not be reported as a policy running.
    Inferred { mode: String, stub: bool },
}

impl DeviceHost {
    pub fn new(controller: Controller, engines: HashMap<String, Box<dyn Engine>>) -> Self {
        Self::with_operators(controller, engines, Default::default())
    }

    pub fn with_operators(
        controller: Controller,
        engines: HashMap<String, Box<dyn Engine>>,
        operators: crate::operator::Operators,
    ) -> Self {
        let cfg = controller.config();
        let (state_timeout_us, command_timeout_us) = (cfg.state_timeout_us, cfg.command_timeout_us);
        let latch = Buttons::new(cfg.exclusive);
        let joints = controller.command().joints();
        Self {
            controller,
            engines,
            stub_reasons: HashMap::new(),
            latch,
            state: RobotState::new(joints),
            // A mode with controls of its own starts at their rest -- for a
            // posture policy the standing height, not zero -- because its
            // operator has heard nothing yet. This is only the others'.
            command: Command::default(),
            pad: Pad::default(),
            operators,
            last_state_us: None,
            last_command_us: None,
            state_timeout_us,
            command_timeout_us,
        }
    }

    /// Record why a mode is on a stub. Reported by `stub_modes`, never hidden.
    ///
    /// Also the interlock: the controller is told, and decides whether a stub
    /// is merely a mode that holds the home pose or a mode that must not run
    /// at all. It is the second for a reference-residual policy, where a zero
    /// action means "track the recording exactly" rather than "hold" -- see
    /// `Controller::interlock_stub_engine`. The reason is amended rather than
    /// replaced, because which stub and why is still what an operator needs.
    pub fn note_stub(&mut self, mode: &str, reason: String) {
        let barred = self.controller.interlock_stub_engine(mode);
        let reason = if barred {
            format!(
                "{reason} -- and this mode's action is a residual on a recording, so a \
                 stub's zeros would play that recording open loop. It will not run at \
                 all until the model loads."
            )
        } else {
            reason
        };
        self.stub_reasons.insert(mode.to_string(), reason);
    }

    pub fn stub_modes(&self) -> impl Iterator<Item = (&String, &String)> {
        self.stub_reasons.iter()
    }

    /// A fresh state sample. The timestamp is the host's, taken when it arrived.
    pub fn on_state(&mut self, state: &RobotState, now: Micros) {
        self.state.q.copy_from_slice(&state.q);
        self.state.qd.copy_from_slice(&state.qd);
        self.state.tau.copy_from_slice(&state.tau);
        self.state.imu = state.imu;
        self.last_state_us = Some(now);
    }

    /// One gamepad frame: latch what it presses, and hand it to every mode's
    /// operator, each of which turns the sticks into that mode's command using
    /// its own contract.
    ///
    /// The contract is where the sticks are interpreted, not here and not on
    /// the bus. `play` and the browser read the same block, so a sign corrected
    /// in `controls.yaml` reaches all three -- which is the whole reason the
    /// robot stopped reading `control-agent`'s already-interpreted command.
    /// Every mode's, not only the running one's: the shift layer and the
    /// release are the person's, and a mode switched into has to have heard
    /// them.
    pub fn on_pad(&mut self, pad: &Pad, now: Micros) {
        self.pad = *pad;
        // No keyboard on the robot; the pad is the only source. A switch's
        // `from` is read against the mode the cascade is in now.
        let state = self.controller.state().name.clone();
        self.latch.observe(&self.controller.config().buttons, pad, &|_| false, now, &state);
        self.operators.pad(pad);
        self.last_command_us = Some(now);
    }

    /// A command a host built itself, for one that has no gamepad -- `play`
    /// with a keyboard, a browser with an on-screen pad. Latches nothing: there is
    /// no pad frame to find an edge in, and those hosts use `toggle`. Read by a
    /// mode whose contract describes no controls; one that does reads its own.
    pub fn on_command(&mut self, command: &Command, now: Micros) {
        self.command = *command;
        self.last_command_us = Some(now);
    }

    /// The last gamepad frame, for a host that wants to show it.
    pub fn pad(&self) -> &Pad {
        &self.pad
    }

    pub fn controller(&self) -> &Controller {
        &self.controller
    }

    /// Replay a bundle's reference vectors through this host.
    ///
    /// Two passes over the same frames, because they answer different
    /// questions. `Reference::replay` drives the controller with the *recorded*
    /// action, so any difference in the targets is the decode or the clamp and
    /// never the model. Then each recorded observation goes through this
    /// machine's own engine, which is where a quantised `.rknn` shows up -- in
    /// the action, by about 1e-3, and nowhere else.
    ///
    /// Consumes nothing and publishes nothing. Safe to run beside a robot that
    /// is standing still, which is the only time anybody will run it.
    pub fn check_reference(
        &mut self,
        reference: &crate::reference::Reference,
    ) -> Result<crate::reference::Report, crate::reference::Error> {
        let mut report = reference.replay(&mut self.controller)?;
        for (i, frame) in reference.frames.iter().enumerate() {
            let (Some(mode), Some(obs), Some(act)) =
                (frame.infer.as_ref(), frame.obs.as_ref(), frame.act.as_ref())
            else {
                continue;
            };
            let Some(engine) = self.engines.get_mut(mode) else {
                continue;
            };
            if engine.is_stub() {
                // A stub returns zeros, which would read as a model that is
                // wrong by exactly the action. `stub_modes` already says so.
                continue;
            }
            match engine.infer(obs) {
                Ok(ours) => report.record_action(i, act, ours),
                Err(e) => {
                    return Err(crate::reference::Error(format!("frame {i}: {e}")))
                }
            }
        }
        Ok(report)
    }

    /// How long feedback may be silent before the FSM treats it as gone.
    /// Read so a caller can drive past it deliberately -- which is what a clean
    /// shutdown does, rather than inventing a second way into the safe state.
    pub fn state_timeout_us(&self) -> u64 {
        self.state_timeout_us
    }

    pub fn command_timeout_us(&self) -> u64 {
        self.command_timeout_us
    }

    /// Every mode change since the last call, and why.
    ///
    /// Drained rather than read: the log is a fixed-size buffer, so a loop that
    /// never takes it would drop the oldest transitions -- and the first few
    /// are the ones that say how the robot got into whatever it is doing now.
    pub fn take_log(&mut self) -> Vec<crate::fsm::Transition> {
        self.controller.take_log()
    }

    pub fn command(&self) -> &MotorCommand {
        self.controller.command()
    }

    /// One tick against whatever has arrived. Read `command()` for what to publish.
    ///
    /// Both phases happen here rather than being handed back: on this host the
    /// engine is in-process and synchronous, so splitting the call would buy
    /// nothing and lose the property that one `step` is one control period. The
    /// browser is where the split pays, and the core keeps it for that.
    pub fn step(&mut self, now: Micros) -> Result<Outcome, Error> {
        let state_fresh = fresh(self.last_state_us, now, self.state_timeout_us);
        let command_fresh = fresh(self.last_command_us, now, self.command_timeout_us);

        // A latched mode must not outlive the operator. Left alone, a held jump
        // keeps the FSM in a state that suspends the tilt fallback with nobody
        // to release it.
        if !command_fresh {
            self.latch.forget();
            // Every mode's operator back to its rest, not to zero: a posture
            // policy commanded a height of zero is being told to lie down, and
            // it would try.
            self.operators.forget();
            self.command = Command::default();
            self.pad = Pad::default();
        }

        // Each mode's command and what its task keeps for itself come from its
        // own operator, once the cascade has picked the mode -- and were let go
        // with it when the operator went away above. The robot has no keyboard,
        // but the clock is what moves an axis the controls move rather than
        // place -- the posture's height, on R3 and the right stick since
        // 2026-09-29 -- so it moves every tick here as on every host.
        self.operators.at(now);
        let input = TickInput {
            state: &self.state,
            command: &self.command,
            operators: Some(&self.operators),
            state_fresh,
            command_fresh,
            buttons: self.latch.active(),
            gripper_active: false,
            controls: &crate::types::TaskControls::default(),
        };
        let step = self.controller.tick(&input, now).map_err(Error::Control)?;
        // A timed policy that has played out hands back to the default. The
        // controller reports it; the latch is this host's, so releasing it is
        // this host's, and the cascade's `always` rule is what catches the
        // robot -- no second path out of a mode.
        if let Some(done) = self.controller.take_completed() {
            self.latch.release(&done);
        }

        match step {
            Step::Hold => Ok(Outcome::Held),
            Step::Infer { mode } => {
                let engine = self
                    .engines
                    .get_mut(&mode)
                    .ok_or_else(|| Error::NoEngine(mode.clone()))?;
                let stub = engine.is_stub();
                let action = engine
                    .infer(self.controller.observation())
                    .map_err(Error::Infer)?
                    .to_vec();
                self.controller.resume(&action).map_err(Error::Control)?;
                Ok(Outcome::Inferred { mode, stub })
            }
        }
    }
}

impl DeviceHost {
    /// One full cycle against the robot's bus: take what arrived, tick, publish.
    ///
    /// Split from `step` so the loop above stays testable without a bus, and
    /// kept here rather than in a binary so it is **compiled against the real
    /// `DdsIo`**. That compile is the check: the four calls below are the whole
    /// of what this host needs from DDS, and if the shapes ever stop fitting,
    /// this file stops building rather than a board stops walking.
    ///
    /// `wall_ns` is Unix-epoch nanoseconds, not a monotonic reading. The motor
    /// firmware reads the wire timestamp as absolute wall-clock time, and a
    /// `steady_clock` value would be misread as a 1970 epoch -- so the caller
    /// passes both clocks and this does not try to derive one from the other.
    pub fn pump(
        &mut self,
        io: &mut crate::dds::DdsIo,
        now: Micros,
        wall_ns: u64,
    ) -> Result<Outcome, Error> {
        if io.take_state(&mut self.state) {
            self.last_state_us = Some(now);
        }
        // The IMU is its own topic. Folded into the same buffer, but its own
        // arrival does not count as motor feedback: a robot whose attitude is
        // fresh and whose joints are silent is exactly the case `feedback_stale`
        // is for.
        io.take_imu(&mut self.state.imu);
        let mut pad = self.pad;
        if io.take_pad(&mut pad) {
            self.on_pad(&pad, now);
        }
        let outcome = self.step(now)?;
        io.publish(self.controller.command(), wall_ns).map_err(Error::Dds)?;
        Ok(outcome)
    }
}

/// Never seen is not fresh. The first tick of a cold controller therefore falls
/// to the safe state, which is where a robot with no feedback belongs.
fn fresh(last: Option<Micros>, now: Micros, timeout_us: u64) -> bool {
    last.is_some_and(|t| now.saturating_sub(t) < timeout_us)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FsmConfig, EXAMPLE};
    use crate::control::Setup;
    use crate::layout::Contract;
    use crate::rknn::StubEngine;

    const JOINTS: [&str; 2] = ["a", "b"];

    /// Returns whatever it is told to, so a test can tell a real inference from
    /// a stub without an NPU.
    struct Fixed {
        out: Vec<f32>,
        calls: std::cell::Cell<usize>,
    }

    impl Engine for Fixed {
        fn input_size(&self) -> usize {
            0
        }
        fn output_size(&self) -> usize {
            self.out.len()
        }
        fn infer(&mut self, _obs: &[f32]) -> Result<&[f32], crate::rknn::Error> {
            self.calls.set(self.calls.get() + 1);
            Ok(&self.out)
        }
        fn name(&self) -> &str {
            "fixed"
        }
    }

    fn contract() -> Contract {
        let text = r#"{"obs_joint_order": ["a","b"], "action_joint_order": ["a","b"],
            "action_scale": 0.25, "default_joint_pos": {"a": 0.0, "b": 0.0},
            "control": {"kp": 10.0, "kd": 0.5, "effort_limit": 2.0, "control_hz": 50.0},
            "observation": {"dim": 6, "history_length": 1,
                "terms": [{"name":"joint_pos","dim":2},{"name":"joint_vel","dim":2},
                          {"name":"actions","dim":2}]},
            "action": {"dim": 2}}"#;
        Contract::from_str(text, &JOINTS.map(String::from)).unwrap()
    }

    fn host(stub_locomotion: bool) -> DeviceHost {
        let cfg = FsmConfig::parse(EXAMPLE).unwrap();
        let mut contracts = HashMap::new();
        let mut engines: HashMap<String, Box<dyn Engine>> = HashMap::new();
        for name in ["locomotion", "carry", "jump"] {
            contracts.insert(name.to_string(), contract());
            let e: Box<dyn Engine> = if stub_locomotion && name == "locomotion" {
                Box::new(StubEngine::new(2))
            } else {
                Box::new(Fixed { out: vec![0.4, -0.4], calls: std::cell::Cell::new(0) })
            };
            engines.insert(name.to_string(), e);
        }
        let setup = Setup {
            reference_go_delay_s: crate::trajectory::DEFAULT_GO_DELAY_S,
            robot: crate::action::RobotLimits {
                default_wire: vec![0.0, 0.0],
                joint_pos_lo: vec![-3.3, -3.3],
                joint_pos_hi: vec![3.3, 3.3],
                joint_pos_rate_limit: 0.0,
            },
            scales: crate::obs::Scales::default(),
            obs_clip: 100.0,
            action_clip: 100.0,
            action_smoothing: 0.0,
            command_terms: vec!["lin_vel_x".into(), "lin_vel_y".into(), "yaw_rate".into()],
            gait_period: 0.32,
            gait_gate_threshold: Some(0.05),
            output_rate_hz: 1000.0,
            source: crate::source::SourceCapability::device(false),
        };
        let controller = Controller::new(cfg, setup, contracts, 0).unwrap();
        DeviceHost::new(controller, engines)
    }

    /// A pad frame with one button down, by the name the wire uses.
    fn pad_pressing(name: &str) -> Pad {
        let mut p = Pad::default();
        p.connected = true;
        p.buttons[crate::vocabulary::pad_buttons().iter().position(|b| *b == name).unwrap()] = true;
        p
    }

    fn upright() -> RobotState {
        let mut s = RobotState::new(2);
        s.imu.quat = [1.0, 0.0, 0.0, 0.0];
        s.imu.valid = true;
        s
    }

    /// Drive until the policy has actually run, feeding fresh samples.
    fn run(h: &mut DeviceHost, ticks: u64, from: Micros) -> (Micros, Vec<Outcome>) {
        let s = upright();
        let c = Command::default();
        let mut out = Vec::new();
        let mut now = from;
        for _ in 0..ticks {
            h.on_state(&s, now);
            h.on_command(&c, now);
            out.push(h.step(now).unwrap());
            now += 1_000;
        }
        (now, out)
    }

    /// A robot that has heard nothing is not a robot standing still. The first
    /// tick of a cold host has no samples at all, and the FSM's `feedback_stale`
    /// rule is the only thing that makes that safe.
    #[test]
    fn a_host_that_has_heard_nothing_is_stale() {
        let mut h = host(false);
        assert_eq!(h.step(0).unwrap(), Outcome::Held);
        assert_eq!(h.controller().state().name, "safe");
        assert_eq!(h.command().kp[0], 0.0, "safe damps, it does not drive");
    }

    /// Feedback that stops arriving drops the robot out of a running policy,
    /// without anything upstream having to notice.
    #[test]
    fn feedback_that_stops_arriving_drops_to_safe() {
        let mut h = host(false);
        let (now, outs) = run(&mut h, 3_000, 0);
        assert!(
            outs.iter().any(|o| matches!(o, Outcome::Inferred { .. })),
            "the policy should be running by now"
        );
        assert_eq!(h.controller().state().name, "locomotion");

        // Nothing more arrives. `state_timeout_ms` is 100.
        h.step(now + 50_000).unwrap();
        assert_eq!(h.controller().state().name, "locomotion", "still inside the timeout");
        assert_eq!(h.step(now + 150_000).unwrap(), Outcome::Held);
        assert_eq!(h.controller().state().name, "safe");
    }

    /// The press is the edge, not the level, and the second press leaves.
    #[test]
    fn a_held_trigger_is_one_request() {
        let mut h = host(false);
        let s = upright();
        // `EXAMPLE` latches `jump` on pad button `A`.
        let mut press = pad_pressing("A");

        // Reach a running policy first, so the jump is a real transition.
        let (mut now, _) = run(&mut h, 3_000, 0);
        assert_eq!(h.controller().state().name, "locomotion");

        // **Four** frames of one press, not three. A latch that acted on the
        // level rather than the edge would toggle once per frame, and an odd
        // number of frames would leave it latched -- passing this test for
        // exactly the wrong reason. Four is even, so level-acting ends up off.
        for _ in 0..4 {
            h.on_state(&s, now);
            h.on_pad(&press, now);
            h.step(now).unwrap();
            now += 1_000;
        }
        assert_eq!(h.controller().state().name, "jump", "one press, one transition");

        // Released, still jumping: the latch holds.
        press = Pad::default();
        h.on_state(&s, now);
        h.on_pad(&press, now);
        h.step(now).unwrap();
        assert_eq!(h.controller().state().name, "jump");

        // Pressed again: it toggles off rather than latching forever.
        press = pad_pressing("A");
        now += 1_000;
        h.on_state(&s, now);
        h.on_pad(&press, now);
        h.step(now).unwrap();
        assert_ne!(h.controller().state().name, "jump");
    }

    /// The one the C++ comment is about. A latched jump suspends the tilt
    /// fallback; if it survived the operator leaving, nothing would release it.
    #[test]
    fn a_latched_button_does_not_outlive_the_operator() {
        let mut h = host(false);
        let s = upright();
        let (mut now, _) = run(&mut h, 3_000, 0);
        let press = pad_pressing("A");
        h.on_state(&s, now);
        h.on_pad(&press, now);
        h.step(now).unwrap();
        assert_eq!(h.controller().state().name, "jump");
        assert!(h.controller().state().skip_tilt_check, "which is why this matters");

        // The operator goes away. State still arrives, so the robot is not
        // stale -- only the command is.
        now += 1_000;
        for _ in 0..1_200 {
            h.on_state(&s, now);
            h.step(now).unwrap();
            now += 1_000;
        }
        assert_ne!(
            h.controller().state().name, "jump",
            "command_timeout_ms is 500; a jump held by nobody is a suspended tilt check"
        );
    }

    /// A stub is reported, not hidden. Zeros decode to the home pose, and a
    /// robot at its home pose looks exactly like a policy that chose to stand.
    #[test]
    fn a_stub_is_visible_in_the_outcome() {
        let mut h = host(true);
        h.note_stub("locomotion", "RKNN not available".into());
        let (_, outs) = run(&mut h, 3_000, 0);
        let ran = outs
            .iter()
            .find_map(|o| match o {
                Outcome::Inferred { mode, stub } => Some((mode.clone(), *stub)),
                _ => None,
            })
            .expect("the policy regime should have been reached");
        assert_eq!(ran, ("locomotion".to_string(), true));
        assert_eq!(h.stub_modes().count(), 1);

        // And the control: a real engine reports false through the same path.
        let mut real = host(false);
        let (_, outs) = run(&mut real, 3_000, 0);
        assert!(outs.iter().any(|o| matches!(o, Outcome::Inferred { stub: false, .. })));
        assert_eq!(real.stub_modes().count(), 0);
    }
}
