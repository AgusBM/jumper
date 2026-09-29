//! The `play` host: the same controller, importable from Python.
//!
//! The third host, and the one that makes an FSM something you can look at
//! before you ship it. Until this existed the only way to find out what a
//! `controller.toml` did was to bundle it and upload it, which is a long way to go to
//! discover a rule in the wrong order.
//!
//! ## Why `play` needs the real thing and not a reading of it
//!
//! `play` already runs a policy, through mjlab's observation manager -- the same
//! code training used. That is what it is for, and it stays: a replay that
//! reproduces training is the only way to see what was trained.
//!
//! This is the other question. The observation a *deployment* builds is this
//! crate's, and the two are not the same code. `play --app` runs a whole app on
//! mjlab through this -- every mode, the person's keys and pad read as the robot
//! reads them, each task's hook, and the controller's own targets and gains on
//! the servos -- which is what the robot will do, short of the robot.
//!
//! ## What crosses
//!
//! Plain sequences of `f32`. `pyo3` will build a `Vec<f32>` from a list, a
//! tuple or a numpy array, at the cost of one copy -- 411 floats at 50 Hz, which
//! is nothing next to a physics step. Taking a numpy array by reference would
//! mean a `numpy` dependency and a lifetime across the boundary, for a saving
//! that does not exist.
//!
//! Errors are `ValueError` with the reason in them. No fallback: a `play` that
//! cannot build the controller should say so, not quietly run something else and
//! leave you comparing the wrong two things.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::action::RobotLimits;
use crate::config::FsmConfig;
use crate::control::{Buttons, Controller, Setup, Step, TickInput};
use crate::layout::Contract;
use crate::obs::Scales;
use crate::source::SourceCapability;
use crate::types::{Command, RobotState};

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// What the host knows and no policy does. The browser's `RobotJson`, as a dict.
///
/// Read field by field rather than derived. `FromPyObject` with `from_item_all`
/// makes every field a required key -- `Option` included, because it maps a
/// missing key to an error rather than to `None` -- and `#[pyo3(default)]` is
/// not a thing it accepts. Twenty lines buys optional fields that are really
/// optional, and an error that names the key instead of the struct.
struct RobotSpec {
    joint_names: Vec<String>,
    joint_pos_lo: Vec<f32>,
    joint_pos_hi: Vec<f32>,
    output_rate_hz: f64,
    gait_period: Option<f64>,
    reference_go_delay_s: Option<f64>,
    gait_gate_threshold: Option<f32>,
    joint_pos_rate_limit: Option<f32>,
    command_terms: Option<Vec<String>>,
}

impl RobotSpec {
    fn read(dict: &Bound<'_, PyAny>) -> PyResult<Self> {
        let need = |key: &str| -> PyResult<Bound<'_, PyAny>> {
            dict.get_item(key).map_err(|_| {
                err(format!("the robot description has no '{key}'"))
            })
        };
        // A key set to `None` is a key not given. `play` passes the gait
        // period it read off the contract, and a clock whose tempo follows the
        // command has none -- which arrived as `None` and failed to extract,
        // before this.
        let maybe = |key: &str| dict.get_item(key).ok().filter(|v| !v.is_none());
        Ok(Self {
            joint_names: need("joint_names")?.extract()?,
            joint_pos_lo: need("joint_pos_lo")?.extract()?,
            joint_pos_hi: need("joint_pos_hi")?.extract()?,
            output_rate_hz: need("output_rate_hz")?.extract()?,
            gait_period: maybe("gait_period").map(|v| v.extract()).transpose()?,
            reference_go_delay_s: maybe("reference_go_delay_s")
                .map(|v| v.extract())
                .transpose()?,
            gait_gate_threshold: maybe("gait_gate_threshold").map(|v| v.extract()).transpose()?,
            joint_pos_rate_limit: maybe("joint_pos_rate_limit").map(|v| v.extract()).transpose()?,
            command_terms: maybe("command_terms").map(|v| v.extract()).transpose()?,
        })
    }
}

#[pyclass(name = "Fsm", module = "mjrl_fsm")]
pub struct PyFsm {
    controller: Controller,
    latch: Buttons,
    keys_down: std::collections::BTreeSet<String>,
    pad: crate::types::Pad,
    state: RobotState,
    command: Command,
    last_state_us: Option<u64>,
    last_command_us: Option<u64>,
    state_timeout_us: u64,
    command_timeout_us: u64,
    joints: usize,
    /// Where the command rests, for every mode at once: what a host that
    /// builds the command itself starts from and hands over through
    /// `set_command_axes` -- the bundler, recording a reference -- and where
    /// whichever mode runs is handed that one command. `play --app` reads the
    /// person through `operators` instead.
    rest: Command,
    /// `guide::pad_guide`, as JSON, for `pad_guide()`.
    guide: String,
    /// Each mode's reading of the person, when this host drives the controller
    /// as the robot does -- `play --app`: keys and pad frames go to them, and
    /// each mode takes its command and its task's controls from its own.
    /// `None` for a host that builds the command itself and hands it over
    /// (`set_command_axes`): the bundler, recording a reference.
    operators: Option<crate::operator::Operators>,
    /// The host's clock as of the latest call that carried one: what a key
    /// reported without a time of its own is stamped with.
    clock_us: u64,
}

impl PyFsm {
    /// Hands off, as a command: zero for a velocity policy, each axis's rest
    /// for a schema-2 one -- the standing height, not a height of zero.
    fn rest(&self) -> Command {
        self.rest
    }
}

#[pymethods]
impl PyFsm {
    /// `config` is `controller.toml` as text, `contracts` a `{mode: layout-json-text}`
    /// mapping, `robot` the host's own facts.
    ///
    /// Contracts arrive as text already read: this crate does no I/O, which is
    /// what lets the identical code run in a browser that has no filesystem.
    /// `trajectories` is `{mode: text}` for the reference tables those
    /// contracts name -- required for a residual policy, meaningless for any
    /// other, and therefore optional rather than a positional every caller
    /// would pass `{}` for.
    ///
    /// `operators`: read the person the way the robot does, through each
    /// mode's own controls -- keys (`set_key`) and pad frames (`set_pad_frame`)
    /// in, each mode's command and its task's controls out. What `play --app`
    /// runs. Off, the host builds one command and hands it over.
    #[new]
    #[pyo3(signature = (config, contracts, robot, now_us = 0, trajectories = None,
                        operators = false))]
    fn new(
        config: &str,
        contracts: HashMap<String, String>,
        robot: &Bound<'_, PyAny>,
        now_us: u64,
        trajectories: Option<HashMap<String, String>>,
        operators: bool,
    ) -> PyResult<Self> {
        let robot = RobotSpec::read(robot)?;
        let cfg = FsmConfig::parse(config).map_err(err)?;
        let joints = robot.joint_names.len();
        if robot.joint_pos_lo.len() != joints || robot.joint_pos_hi.len() != joints {
            return Err(err(format!(
                "this robot has {joints} joints; got {} lower and {} upper limits",
                robot.joint_pos_lo.len(),
                robot.joint_pos_hi.len()
            )));
        }

        let trajectories = trajectories.unwrap_or_default();
        let mut parsed = HashMap::new();
        for (mode, text) in contracts {
            let mut c = Contract::from_str(&text, &robot.joint_names).map_err(err)?;
            // A contract with a `reference` block cannot be built into a
            // runtime without its table, so say which mode and which file
            // rather than letting `ObservationBuilder::new` report a term it
            // cannot construct -- the cause is here and the symptom is there.
            if let Some(spec) = c.layout.reference.clone() {
                let Some(table) = trajectories.get(&mode) else {
                    return Err(err(format!(
                        "mode '{mode}' is a residual on the recording '{}', and no \
                         trajectory was passed for it. Pass `trajectories={{'{mode}': \
                         <text of that file>}}`.",
                        spec.file
                    )));
                };
                c.attach_reference(table, &robot.joint_names).map_err(err)?;
            }
            parsed.insert(mode, c);
        }
        let default_wire = parsed
            .values()
            .next()
            .map(|c| c.default_wire.clone())
            .unwrap_or_else(|| vec![0.0; joints]);

        let setup = Setup {
            robot: RobotLimits {
                default_wire,
                joint_pos_lo: robot.joint_pos_lo,
                joint_pos_hi: robot.joint_pos_hi,
                joint_pos_rate_limit: robot.joint_pos_rate_limit.unwrap_or(0.0),
            },
            scales: Scales::default(),
            obs_clip: 100.0,
            action_clip: 100.0,
            action_smoothing: 0.0,
            command_terms: robot.command_terms.unwrap_or_else(|| {
                vec!["lin_vel_x".into(), "lin_vel_y".into(), "yaw_rate".into()]
            }),
            gait_period: robot.gait_period.unwrap_or(0.32),
            reference_go_delay_s: robot
                .reference_go_delay_s
                .unwrap_or(crate::trajectory::DEFAULT_GO_DELAY_S),
            gait_gate_threshold: robot.gait_gate_threshold,
            output_rate_hz: robot.output_rate_hz,
            // mjlab's solver, which is what every term was in training. The one
            // host where nothing has moved -- `source.rs` says which terms move
            // on the other two.
            source: SourceCapability::mjlab(),
        };
        let (state_timeout_us, command_timeout_us) = (cfg.state_timeout_us, cfg.command_timeout_us);
        let latch = Buttons::new(cfg.exclusive);
        // Each mode's controls, for their rests: this host is handed one
        // finished command, so it has to rest where every mode reading a
        // channel rests it. `Operators::shared_rest` refuses two that disagree.
        let every = crate::operator::Operators::of(parsed.iter().map(|(n, c)| (n.as_str(), c)))
            .map_err(err)?;
        let rest = every.shared_rest().map_err(err)?;
        let controller = Controller::new(cfg, setup, parsed, now_us).map_err(err)?;
        // After the controller, whose hooks say which task controls each mode
        // answers.
        let guide =
            crate::guide::pad_guide(controller.config(), &every, &controller.hook_reads()).to_string();
        // Refused here as the board and a browser refuse it: a key both a mode's
        // control and a switch would do two things on one press.
        let operators =
            if operators { Some(every.beside(controller.config()).map_err(err)?) } else { None };
        Ok(Self {
            guide,
            operators,
            clock_us: now_us,
            command: rest,
            rest,
            controller,
            latch,
            keys_down: Default::default(),
            pad: Default::default(),
            state: RobotState::new(joints),
            last_state_us: None,
            last_command_us: None,
            state_timeout_us,
            command_timeout_us,
            joints,
        })
    }

    /// Every control of the pad, what it does in each mode and which keys press
    /// it, as JSON (`guide.rs` has the shape). What a bundler writes a bundle's
    /// manual from: the same answer a browser draws the pad from.
    fn pad_guide(&self) -> String {
        self.guide.clone()
    }

    /// One frame of robot state, in **wire order** -- `joint_names`, not the
    /// simulator's order and not the policy's. The three are paired by name.
    ///
    /// `gyro` is body-frame angular velocity. mjlab reports the world frame, so
    /// the caller rotates; doing it here would mean this crate knowing which
    /// simulator it is attached to.
    #[pyo3(signature = (q, qd, tau, quat, gyro, now_us))]
    fn set_state(
        &mut self,
        q: Vec<f32>,
        qd: Vec<f32>,
        tau: Vec<f32>,
        quat: Vec<f32>,
        gyro: Vec<f32>,
        now_us: u64,
    ) -> PyResult<()> {
        for (name, got) in [("q", q.len()), ("qd", qd.len()), ("tau", tau.len())] {
            if got != self.joints {
                return Err(err(format!(
                    "this robot has {} joints; {name} has {got}",
                    self.joints
                )));
            }
        }
        if quat.len() != 4 || gyro.len() != 3 {
            return Err(err("quat must be (w, x, y, z) and gyro (x, y, z)"));
        }
        self.state.q.copy_from_slice(&q);
        self.state.qd.copy_from_slice(&qd);
        self.state.tau.copy_from_slice(&tau);
        self.state.imu.quat = [quat[0], quat[1], quat[2], quat[3]];
        self.state.imu.gyro = [gyro[0], gyro[1], gyro[2]];
        self.state.imu.valid = true;
        self.last_state_us = Some(now_us);
        self.clock_us = self.clock_us.max(now_us);
        Ok(())
    }

    /// The operator's command, in physical units -- m/s and rad/s, not stick
    /// deflection, for a host that reads the stick itself. `play --app` does
    /// not: it hands over keys and pad frames, and each mode's `controls.yaml`
    /// scales them (`operators`).
    #[pyo3(signature = (lin_vel_x, lin_vel_y, yaw_rate, now_us = 0))]
    fn set_command(&mut self, lin_vel_x: f32, lin_vel_y: f32, yaw_rate: f32, now_us: u64) {
        self.command = Command { lin_vel_x, lin_vel_y, yaw_rate, ..self.rest() };
        self.last_command_us = Some(now_us);
    }

    /// The operator's command, channel by channel, by the names the contract's
    /// controller block gives them -- `lin_vel_x`, `ang_vel_z`, `twist`,
    /// `height` and so on, in physical units. Channels not named are at rest.
    ///
    /// For a task with more than one command, which `set_command`'s three
    /// numbers cannot carry. A name no channel has is an error rather than a
    /// value dropped: a posture command that silently went nowhere is a policy
    /// shown the neutral posture while the screen says it was asked to lean.
    #[pyo3(signature = (names, values, now_us = 0))]
    fn set_command_axes(&mut self, names: Vec<String>, values: Vec<f32>, now_us: u64) -> PyResult<()> {
        if names.len() != values.len() {
            return Err(err(format!("{} names and {} values", names.len(), values.len())));
        }
        let mut command = self.rest();
        for (name, value) in names.iter().zip(values) {
            if !command.set_axis(name, value) {
                return Err(err(format!("'{name}' is not a command channel this controller carries")));
            }
        }
        self.command = command;
        self.last_command_us = Some(now_us);
        Ok(())
    }

    /// Whether a key is **down**, not that it was pressed.
    ///
    /// `rise` and `fall` are both in the vocabulary, so a host has to report
    /// both edges; a one-shot "pressed" call could only ever drive half of it.
    /// Key names are the contract's (`keypad_1`), not a browser's (`Numpad1`).
    /// Returns whether any switch reads this key -- as its key, or as a key of
    /// its modifier -- so a caller can tell "that key does nothing" from "the
    /// rule did not fire".
    ///
    /// With `operators`, the key goes to every mode's operator as well, at
    /// `now_us` -- or at the latest time this host was told, for a caller
    /// with no clock of its own -- and counts as bound if a mode's controls
    /// bind it.
    #[pyo3(signature = (key, down, now_us = None))]
    fn set_key(&mut self, key: &str, down: bool, now_us: Option<u64>) -> bool {
        if down {
            self.keys_down.insert(key.to_string());
        } else {
            self.keys_down.remove(key);
        }
        let now = now_us.unwrap_or(self.clock_us);
        self.clock_us = self.clock_us.max(now);
        let driven = match self.operators.as_mut() {
            Some(ops) => {
                self.last_command_us = Some(now);
                ops.key(key, down, now)
            }
            None => false,
        };
        driven || self.controller.config().switches_on_key(key)
    }

    /// One whole frame of a real pad, the robot's pad service's way: the
    /// buttons down by the dictionary's names, the six axes by theirs
    /// (unnamed ones at rest), and the d-pad as -1 / 0 / +1 with **up
    /// positive**, as the service reports it. `connected` false is a pad that
    /// is not there, which reads as nothing touched.
    ///
    /// Every frame, pad or no pad, for a host that drives `operators`: it is
    /// also what says the operator is present. The controller forgets a person
    /// it has not heard from in `command_timeout_ms` and lets go of every mode
    /// they latched -- the right answer on a robot whose pad went quiet, and a
    /// claw mode dropping every half second on a keyboard left alone here.
    #[pyo3(signature = (buttons, axes, dpad_x = 0, dpad_y = 0, connected = true, now_us = None))]
    fn set_pad_frame(
        &mut self,
        buttons: Vec<String>,
        axes: HashMap<String, f32>,
        dpad_x: i8,
        dpad_y: i8,
        connected: bool,
        now_us: Option<u64>,
    ) -> PyResult<()> {
        let mut pad = crate::types::Pad { connected, ..Default::default() };
        let names = crate::vocabulary::pad_buttons();
        for b in &buttons {
            let i = names.iter().position(|n| n == b).ok_or_else(|| {
                err(format!("'{b}' is not a button the pad service publishes ({names:?})"))
            })?;
            pad.buttons[i] = true;
        }
        let axis_names = crate::vocabulary::pad_axes();
        for (a, v) in &axes {
            let i = axis_names.iter().position(|n| n == a).ok_or_else(|| {
                err(format!("'{a}' is not an axis the pad service publishes ({axis_names:?})"))
            })?;
            pad.axes[i] = *v;
        }
        pad.dpad_x = dpad_x.signum();
        pad.dpad_y = dpad_y.signum();
        self.pad = pad;
        let now = now_us.unwrap_or(self.clock_us);
        self.clock_us = self.clock_us.max(now);
        if let Some(ops) = self.operators.as_mut() {
            ops.pad(&self.pad);
            self.last_command_us = Some(now);
        }
        Ok(())
    }

    /// Whether a **pad** button is down, for a host reading a gamepad. Same
    /// contract as the key call, with the robot pad service's names.
    fn set_pad(&mut self, button: &str, down: bool) -> bool {
        let Some(i) = crate::vocabulary::pad_buttons().iter().position(|b| *b == button) else {
            return false;
        };
        self.pad.buttons[i] = down;
        self.controller.config().buttons.iter().any(|b| b.pad.as_deref() == Some(button))
    }
    /// One tick. Returns the mode needing an inference, or `None` when the step
    /// is finished and `positions()` is ready.
    fn tick(&mut self, now_us: u64) -> PyResult<Option<String>> {
        self.clock_us = self.clock_us.max(now_us);
        if let Some(ops) = self.operators.as_mut() {
            // A key held and not repeated has no event of its own to move the
            // clock its control climbs by, so every tick moves it.
            ops.at(now_us);
        }
        let state_fresh = fresh(self.last_state_us, now_us, self.state_timeout_us);
        let command_fresh = fresh(self.last_command_us, now_us, self.command_timeout_us);
        if !command_fresh {
            self.latch.forget();
            self.keys_down.clear();
            self.command = self.rest();
            if let Some(ops) = self.operators.as_mut() {
                ops.forget();
            }
        }
        // Every tick, so `rise` and `fall` see an edge. A host that only
        // reported presses could not produce a `fall` at all.
        // The pad's switches read the pad and the keyboard's read the keys: two
        // sets of bindings, one latch per name.
        let keys = std::mem::take(&mut self.keys_down);
        let state = self.controller.state().name.clone();
        self.latch.observe(&self.controller.config().buttons, &self.pad, &|k| keys.contains(k), now_us, &state);
        self.keys_down = keys;
        // With `operators`, each mode reads its command and its task's controls
        // from its own, as on the robot -- `play --app` writes this host's
        // targets into the simulator, so a hook's answer to a control is what
        // moves the joints. Without, the command is the one handed over and no
        // mode's task controls reach anything.
        let controls = crate::types::TaskControls::NONE;
        let input = TickInput {
            state: &self.state,
            command: &self.command,
            operators: self.operators.as_ref(),
            state_fresh,
            command_fresh,
            buttons: self.latch.active(),
            gripper_active: false,
            controls: &controls,
        };
        let step = self.controller.tick(&input, now_us).map_err(err)?;
        // A timed policy that has played out hands back to the default. The
        // controller reports it; the latch is this host's, so releasing it is
        // this host's, and the cascade's `always` rule is what catches the
        // robot -- no second path out of a mode.
        if let Some(done) = self.controller.take_completed() {
            self.latch.release(&done);
        }

        Ok(match step {
            Step::Hold => None,
            Step::Infer { mode } => Some(mode),
        })
    }

    /// Every transition since the last call: `(micros, from, to, rule index,
    /// the condition that fired)`.
    ///
    /// Draining, so a host that prints them cannot print one twice. The rule is
    /// the answer to the only question an FSM raises -- *why* -- and two rules
    /// leading to one state are indistinguishable without it.
    fn take_log(&mut self) -> Vec<(u64, String, String, usize, String)> {
        self.controller
            .take_log()
            .into_iter()
            .map(|t| (t.at, t.from, t.to, t.rule, t.why))
            .collect()
    }

    fn observation(&self) -> Vec<f32> {
        self.controller.observation().to_vec()
    }

    fn resume(&mut self, action: Vec<f32>) -> PyResult<()> {
        self.controller.resume(&action).map_err(err)?;
        Ok(())
    }

    /// Joint position targets, wire order. Valid after any tick, inferring or
    /// not: a hold still publishes and the output filter is still moving.
    fn positions(&self) -> Vec<f32> {
        self.controller.command().pos.clone()
    }

    fn kp(&self) -> Vec<f32> {
        self.controller.command().kp.clone()
    }

    fn kd(&self) -> Vec<f32> {
        self.controller.command().kd.clone()
    }

    #[getter]
    fn mode(&self) -> String {
        self.controller.state().name.clone()
    }

    /// Whether the policy is producing output, as opposed to holding or ramping
    /// toward it. They differ through a mode-switch ramp, which waits on the
    /// measured pose and can wait indefinitely -- a robot holding still looks
    /// broken unless something says which it is.
    #[getter]
    fn running_policy(&self) -> bool {
        self.controller.is_running_policy()
    }

    #[getter]
    fn modes(&self) -> Vec<String> {
        self.controller
            .config()
            .states
            .iter()
            .filter(|s| s.has_model())
            .map(|s| s.name.clone())
            .collect()
    }

    /// Every binding: `(name, pad, key, gesture, modifier)`, for a host
    /// drawing a keypad or wiring a gamepad. A binding is one device's -- a pad
    /// button or a key -- and two with one name share a latch; which modes
    /// each may be pressed from, and what a leave lets go of, are in
    /// `pad_guide()`.
    #[getter]
    fn bindings(&self) -> Vec<(String, Option<String>, Option<String>, String, Option<String>)> {
        self.controller
            .config()
            .buttons
            .iter()
            .map(|b| (b.name.clone(), b.pad.clone(), b.key.clone(), b.on.to_string(),
                      b.with.clone()))
            .collect()
    }

    /// Replay a bundle's `reference.json` and report where this host differs.
    ///
    /// The controller only -- it drives the recorded action back in, so the
    /// numbers are the observation, the decode and the clamp, never the model.
    /// A caller that wants its inference backend checked runs the recorded
    /// observations through it and compares against `act` itself.
    ///
    /// Destroys this controller's state: it is driven through somebody else's
    /// frames. Build a fresh one to check, and a fresh one to run.
    fn check_reference(&mut self, text: &str) -> PyResult<PyReport> {
        let reference = crate::reference::Reference::parse(text).map_err(|e| err(e.to_string()))?;
        let report =
            reference.replay(&mut self.controller).map_err(|e| err(e.to_string()))?;
        Ok(PyReport {
            frames: report.frames,
            inferences: report.inferences,
            observation: report.observation.diff,
            target: report.target.diff,
            divergent: report.divergent.iter().map(|(t, w)| (t.clone(), w.diff)).collect(),
            mode_mismatches: report
                .mode_mismatches
                .iter()
                .map(|(i, want, got)| (*i, want.clone(), got.clone()))
                .collect(),
        })
    }

    /// Every observation term this host supplies differently than training did,
    /// for the active mode. Empty here by construction -- `play` is mjlab, where
    /// the policy trained -- and not empty on the other two hosts, which is the
    /// point of being able to ask.
    fn divergences(&self, mode: &str) -> Vec<(String, String, String)> {
        self.controller
            .divergences(mode)
            .iter()
            .map(|d| (d.term.clone(), d.trained.to_string(), d.here.to_string()))
            .collect()
    }
}

/// A replay's result, flat. Named fields rather than a dict so a caller that
/// misspells one gets an AttributeError instead of a silent `None` that reads
/// as agreement.
#[pyclass(name = "Report", get_all)]
pub struct PyReport {
    pub frames: usize,
    pub inferences: usize,
    /// Worst absolute difference on the terms every host must build alike.
    pub observation: f32,
    /// Worst absolute difference on the published joint targets.
    pub target: f32,
    /// `(term, worst)` for terms this host sources differently than training.
    pub divergent: Vec<(String, f32)>,
    /// `(frame, expected, actual)`. A cascade landing elsewhere is not a
    /// tolerance question.
    pub mode_mismatches: Vec<(usize, String, String)>,
}

fn fresh(last: Option<u64>, now: u64, timeout_us: u64) -> bool {
    last.is_some_and(|t| now.saturating_sub(t) < timeout_us)
}

#[pymodule]
fn mjrl_fsm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReport>()?;
    m.add_class::<PyFsm>()?;
    m.add("__doc__", "The controller kk-rl-mjlab's robot runs, importable.")?;
    Ok(())
}
