//! The browser's host: MuJoCo WASM in, onnxruntime-web in the middle, actuators out.
//!
//! The second host, and the one the two-phase tick exists for. On the robot the
//! engine is in-process and synchronous, so `DeviceHost::pump` does both phases
//! in one call. Here the model runs in JavaScript -- onnxruntime-web -- and a
//! wasm module cannot call it and return inside one synchronous call. So the
//! split is not an abstraction, it is the only shape that works:
//!
//! ```js
//! fsm.set_state(q, qd, tau, quat, gyro, nowMicros);
//! fsm.set_command(vx, vy, wz, nowMicros);
//! const mode = fsm.tick(nowMicros);     // undefined when the tick is finished
//! if (mode !== undefined) fsm.resume(await session.run(fsm.observation()));
//! publish(fsm.positions(), fsm.kp(), fsm.kd());
//! ```
//!
//! The names are the Rust ones: `wasm_bindgen` exports an identifier unchanged
//! unless it is given a `js_name`, and only `setKey`, `setPad`, `bindings` and
//! `checkReference` carry one. This block showed `setState` and `setCommand`,
//! which no build has ever exported, and gave the latter two arguments it does
//! not take.
//!
//! ## What this replaces, and what it must therefore not change
//!
//! BEUNLIMITED already assembles this observation, in TypeScript, from the same
//! `layout.json`. That implementation is correct; the problem is that it is a
//! second one. A sign corrected upstream reaches it only if somebody carries it
//! across, and this whole crate exists because "somebody carries it across" is
//! how the three implementations came to agree in the first place.
//!
//! So the numbers this produces have to be the same numbers. The check that they
//! are is not in this file -- it is a recorded episode replayed through both,
//! which nothing records yet.
//!
//! ## Everything crosses as a flat array
//!
//! No structs across the boundary, and no JSON per tick. `wasm-bindgen` can
//! marshal a struct, but every field becomes a property read and this runs at
//! the simulator's step rate. Flat `Float32Array`s copy once.
//!
//! The configuration crosses as text, once, at construction: a browser has no
//! filesystem, so the host fetches and this parses.

use std::collections::HashMap;

use wasm_bindgen::prelude::*;

use crate::action::RobotLimits;
use crate::config::FsmConfig;
use crate::control::{Buttons, Controller, Setup, Step, TickInput};
use crate::layout::Contract;
use crate::obs::Scales;
use crate::source::SourceCapability;
use crate::types::{Command, RobotState};

/// The robot's fixed facts, as JSON. Sent once, because they belong to the model
/// the browser compiled and not to any policy.
#[derive(serde::Deserialize)]
struct RobotJson {
    joint_names: Vec<String>,
    joint_pos_lo: Vec<f32>,
    joint_pos_hi: Vec<f32>,
    #[serde(default)]
    joint_pos_rate_limit: f32,
    #[serde(default)]
    command_terms: Vec<String>,
    #[serde(default = "default_gait_period")]
    gait_period: f64,
    #[serde(default)]
    gait_gate_threshold: Option<f32>,
    #[serde(default = "default_obs_clip")]
    obs_clip: f32,
    #[serde(default = "default_action_clip")]
    action_clip: f32,
    #[serde(default)]
    action_smoothing: f32,
    #[serde(default = "default_rate")]
    output_rate_hz: f64,
    /// Seconds from entering a reference-guided mode to its `go`. A placeholder
    /// for the button that should start the motion; settable here because the
    /// browser is where somebody watches it and wants it sooner.
    #[serde(default = "default_go_delay")]
    reference_go_delay_s: f64,
}

fn default_go_delay() -> f64 {
    crate::trajectory::DEFAULT_GO_DELAY_S
}

fn default_gait_period() -> f64 {
    0.32
}
fn default_obs_clip() -> f32 {
    100.0
}
fn default_action_clip() -> f32 {
    100.0
}
fn default_rate() -> f64 {
    50.0
}

#[wasm_bindgen]
pub struct WebFsm {
    controller: Controller,
    latch: Buttons,
    keys_down: std::collections::BTreeSet<String>,
    pad: crate::types::Pad,
    state: RobotState,
    command: Command,
    /// Micros of the last sample of each kind. The browser's clock is the
    /// caller's; this only remembers when it last heard.
    last_state_us: Option<u64>,
    last_command_us: Option<u64>,
    state_timeout_us: u64,
    command_timeout_us: u64,
    joints: usize,
    /// How each mode's contract says a person becomes its command, one
    /// operator per mode that describes controls. Empty when no mode does,
    /// which is an ordinary bundle driven by `set_command`.
    operators: crate::operator::Operators,
    /// Named quantities the page supplied beyond `set_state`, kept for the
    /// terms that read them. See `setSignal`.
    signals: HashMap<String, Vec<f32>>,
    /// The latest time the page has given, microseconds: when a key reported
    /// without one (`setKey`) went down.
    clock_us: u64,
}

#[wasm_bindgen]
impl WebFsm {
    /// `config` is the FSM TOML, `contracts` a JSON object of
    /// `{mode: layout}`, `robot` the JSON above.
    ///
    /// `trajectories` is `{mode: text}` for the recordings those contracts name
    /// in their `reference` block -- the contents of the `*.trajectory.json`
    /// files the bundle carries. Omit it for a bundle of ordinary policies; a
    /// mode that needs one and does not get it is refused, because the
    /// alternative is building its reference terms from zeros, which is a
    /// motionless recording that a standing robot tracks perfectly.
    ///
    /// It is last and optional so a page written against the four-argument form
    /// keeps working. Before it existed, a bundle carrying `jumper.jump` or
    /// `jumper.dance` built on the board and in `play` and threw here -- the
    /// other two hosts had each solved it, one by reading the file and one by
    /// being handed the text, and this is the second of those.
    ///
    /// Every failure here is a `JsError` with the reason in it. There is no
    /// fallback: a browser that cannot build this should say so, not run a robot
    /// on defaults nobody chose.
    #[wasm_bindgen(constructor)]
    pub fn new(
        config: &str,
        contracts: &str,
        robot: &str,
        now_us: f64,
        trajectories: Option<String>,
    ) -> Result<WebFsm, JsError> {
        let cfg = FsmConfig::parse(config).map_err(js)?;
        let robot: RobotJson = serde_json::from_str(robot).map_err(js)?;
        let joints = robot.joint_names.len();

        let tables: HashMap<String, String> = match trajectories.as_deref() {
            None | Some("") => HashMap::new(),
            Some(t) => serde_json::from_str(t).map_err(js)?,
        };

        let raw: HashMap<String, serde_json::Value> = serde_json::from_str(contracts).map_err(js)?;
        let mut parsed = HashMap::new();
        for (mode, layout) in raw {
            let text = serde_json::to_string(&layout).map_err(js)?;
            let mut c = Contract::from_str(&text, &robot.joint_names).map_err(js)?;
            // Named here rather than left to the observation builder: the cause
            // is a missing argument to this constructor, and the symptom would
            // otherwise surface as a term it cannot build.
            if let Some(spec) = c.layout.reference.clone() {
                let Some(table) = tables.get(&mode) else {
                    return Err(js(format!(
                        "mode '{mode}' is driven by the recording '{}', and no trajectory \
                         was passed for it. Read that file out of the bundle and pass \
                         `trajectories` as {{\"{mode}\": <its text>}}.",
                        spec.file
                    )));
                };
                c.attach_reference(table, &robot.joint_names).map_err(js)?;
            }
            parsed.insert(mode, c);
        }

        // The home pose is the first model mode's, because a browser has no
        // "the robot's home pose" independent of a policy -- the model it
        // compiled has a keyframe, but that is the scene's, not the contract's.
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
                joint_pos_rate_limit: robot.joint_pos_rate_limit,
            },
            scales: Scales::default(),
            obs_clip: robot.obs_clip,
            action_clip: robot.action_clip,
            action_smoothing: robot.action_smoothing,
            command_terms: if robot.command_terms.is_empty() {
                vec!["lin_vel_x".into(), "lin_vel_y".into(), "yaw_rate".into()]
            } else {
                robot.command_terms
            },
            gait_period: robot.gait_period,
            gait_gate_threshold: robot.gait_gate_threshold,
            reference_go_delay_s: robot.reference_go_delay_s,
            output_rate_hz: robot.output_rate_hz,
            // The same solver the policy trained under, so every term has the
            // provenance it had then -- including `joint_torque`, which here is
            // `actuator_force` and on the robot is a reconstruction. That is the
            // one place the two hosts take different paths through `tick`, and
            // saying so here is what keeps it from being a hidden constant.
            source: SourceCapability::mujoco_wasm(),
        };
        let (state_timeout_us, command_timeout_us) = (cfg.state_timeout_us, cfg.command_timeout_us);
        // Before the controller takes them: the sticks are interpreted by each
        // mode's own contract on every host, and this is where a browser reads
        // that.
        let operators = crate::operator::Operators::of(parsed.iter().map(|(n, c)| (n.as_str(), c)))
            .and_then(|o| o.beside(&cfg))
            .map_err(js)?;
        let latch = Buttons::new(cfg.exclusive);
        let controller = Controller::new(cfg, setup, parsed, now_us as u64).map_err(js)?;
        Ok(WebFsm {
            controller,
            latch,
            keys_down: Default::default(),
            pad: Default::default(),
            state: RobotState::new(joints),
            // The page's own command, for a mode with no controls of its own.
            // One that has them starts at their rest -- the standing height,
            // for a posture policy -- because its operator has heard nothing.
            command: Command::default(),
            last_state_us: None,
            last_command_us: None,
            state_timeout_us,
            command_timeout_us,
            joints,
            operators,
            signals: HashMap::new(),
            clock_us: now_us as u64,
        })
    }

    /// One frame of robot state, in **wire order**.
    ///
    /// `tau` is the applied joint torque. In this simulator it is the solver's
    /// `actuator_force`, which is the same provenance training had -- unlike the
    /// robot, where nothing measures it and the controller reconstructs it from
    /// the PD it commanded. `source.rs` is where that difference is stated;
    /// passing it here is what makes this the easy case.
    pub fn set_state(
        &mut self,
        q: &[f32],
        qd: &[f32],
        tau: &[f32],
        quat: &[f32],
        gyro: &[f32],
        now_us: f64,
    ) -> Result<(), JsError> {
        if q.len() != self.joints || qd.len() != self.joints || tau.len() != self.joints {
            return Err(JsError::new(&format!(
                "this robot has {} joints; got q={} qd={} tau={}",
                self.joints,
                q.len(),
                qd.len(),
                tau.len()
            )));
        }
        if quat.len() != 4 || gyro.len() != 3 {
            return Err(JsError::new("quat must be (w,x,y,z) and gyro (x,y,z)"));
        }
        self.state.q.copy_from_slice(q);
        self.state.qd.copy_from_slice(qd);
        self.state.tau.copy_from_slice(tau);
        self.state.imu.quat = [quat[0], quat[1], quat[2], quat[3]];
        self.state.imu.gyro = [gyro[0], gyro[1], gyro[2]];
        self.state.imu.valid = true;
        self.last_state_us = Some(now_us as u64);
        self.clock_us = self.clock_us.max(now_us as u64);
        Ok(())
    }

    /// The operator's command, already in physical units.
    ///
    /// Scaling from a stick's -1..1 to m/s belongs to whoever reads the stick,
    /// and in this simulator that is the page, which already does it from the
    /// policy's own `command_ranges`. Doing it again here would be a second
    /// place the scaling lives.
    ///
    /// **Never read by a mode whose contract describes controls**
    /// (`takesRawInput`): that mode's contract turns the pad and the keys into
    /// its command itself, and a page's own mapping on top would be a second
    /// one. Only a mode describing none reads this. So a page can send it every
    /// frame whatever it loaded -- which is what lets an uploaded bundle carry a
    /// controls scheme this page was never written for -- and it keeps the
    /// operator fresh either way.
    pub fn set_command(&mut self, lin_vel_x: f32, lin_vel_y: f32, yaw_rate: f32, now_us: f64) {
        self.last_command_us = Some(now_us as u64);
        self.command = Command { lin_vel_x, lin_vel_y, yaw_rate, ..Default::default() };
    }

    /// One tick. Returns the mode that needs an inference, or `null` when the
    /// tick is finished and `positions()` is ready to publish.
    pub fn tick(&mut self, now_us: f64) -> Result<Option<String>, JsError> {
        let now = now_us as u64;
        self.clock_us = self.clock_us.max(now);
        // A key held and not repeated has no event of its own to move the
        // clock its control climbs by, so every tick moves it.
        self.operators.at(now);
        let state_fresh = fresh(self.last_state_us, now, self.state_timeout_us);
        let command_fresh = fresh(self.last_command_us, now, self.command_timeout_us);
        if !command_fresh {
            self.latch.forget();
            self.keys_down.clear();
            // Every mode's operator back to its rest, not to zero: a posture
            // policy commanded a height of zero is being told to lie down.
            self.operators.forget();
            self.command = Command::default();
        }
        // Every tick, and the only place it happens. `rise` and `fall` are
        // one frame wide and are derived from the change since the last
        // observation, so a second observer does not see them twice -- it
        // destroys them. The pad's state is already here in `self.pad`;
        // `padFrame` stores it and this reads it.
        //
        // The pad's switches read the pad and the keyboard's read the keys --
        // two sets of bindings, one latch per name -- each against the mode the
        // cascade is in, which a switch's `from` names.
        let keys = std::mem::take(&mut self.keys_down);
        let state = self.controller.state().name.clone();
        self.latch.observe(&self.controller.config().buttons, &self.pad, &|k| keys.contains(k), now, &state);
        self.keys_down = keys;
        // Each mode's command and its task's controls come from its own
        // operator once the cascade has picked it -- the pad's and the keys'
        // alike, whichever was touched last: the page hands both over raw.
        let input = TickInput {
            state: &self.state,
            command: &self.command,
            operators: Some(&self.operators),
            state_fresh,
            command_fresh,
            buttons: self.latch.active(),
            gripper_active: false,
            controls: &crate::types::TaskControls::NONE,
        };
        let step = self.controller.tick(&input, now).map_err(js)?;
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

    /// Every transition since the last call, as JSON:
    /// `[{at, from, to, rule, why}]`. Draining, so nothing is printed twice.
    pub fn take_log(&mut self) -> String {
        let entries: Vec<_> = self
            .controller
            .take_log()
            .into_iter()
            .map(|t| serde_json::json!({
                "at": t.at, "from": t.from, "to": t.to, "rule": t.rule, "why": t.why,
            }))
            .collect();
        serde_json::to_string(&entries).unwrap_or_else(|_| "[]".into())
    }

    /// The observation for the mode `tick` just named. Copied out, because the
    /// caller is about to hand it to a model that will keep it.
    pub fn observation(&self) -> Vec<f32> {
        self.controller.observation().to_vec()
    }

    pub fn resume(&mut self, action: &[f32]) -> Result<(), JsError> {
        self.controller.resume(action).map_err(js)?;
        Ok(())
    }

    /// Joint position targets, wire order. Read after every tick, whether or not
    /// it inferred -- a held tick still publishes, and the output filter is
    /// still moving.
    pub fn positions(&self) -> Vec<f32> {
        self.controller.command().pos.clone()
    }

    pub fn kp(&self) -> Vec<f32> {
        self.controller.command().kp.clone()
    }

    pub fn kd(&self) -> Vec<f32> {
        self.controller.command().kd.clone()
    }

    /// Whether a key is **down**, not that it was pressed.
    ///
    /// `rise` and `fall` are both in the vocabulary, so a host has to report
    /// both edges; a one-shot "pressed" call could only ever drive half of it.
    /// Key names are the contract's (`keypad_1`), not the browser's
    /// (`Numpad1`); translating between them is the host's job, the same as it
    /// already is for the command keys. Returns whether any binding names this
    /// key, so a caller can tell "that key does nothing" from "the rule did not
    /// fire" -- two outcomes that look identical on screen.
    #[wasm_bindgen(js_name = setKey)]
    pub fn set_key(&mut self, key: &str, down: bool) -> bool {
        // No time of its own: a key reported here went down when the page last
        // said what time it was.
        self.key(key, down, self.clock_us)
    }

    /// A key, by the browser's own name for it: `KeyboardEvent.code`.
    ///
    /// **The call a page should make for every key it sees**, down and up,
    /// with `repeat` as the event reports it. This controller decides which of
    /// them mean anything -- the dictionary it carries says what `KeyW` is --
    /// so a bundle whose keyboard this page has never heard of still works,
    /// and a page does not need a new build when the dictionary grows a key.
    ///
    /// `repeat` is a held key's auto-repeat: the same key, still down, which
    /// is how both readers take it. A key's control climbs for as long as the
    /// key is held -- full after the block's `full_after_s` -- and centres when
    /// it comes up, repeated or not; the mode latch reads levels too.
    ///
    /// Returns whether anything here binds the key, which is what tells a page
    /// whether to `preventDefault` it.
    #[wasm_bindgen(js_name = setKeyCode)]
    pub fn set_key_code(&mut self, code: &str, down: bool, repeat: bool, now_us: f64) -> bool {
        let Some(name) = crate::vocabulary::key_for_browser(code) else {
            return false;
        };
        let now = now_us as u64;
        self.clock_us = self.clock_us.max(now);
        let _ = repeat;
        let bound = self.key(name, down, now);
        if !self.operators.is_empty() {
            self.last_command_us = Some(now_us as u64);
        }
        bound
    }

    fn key(&mut self, key: &str, down: bool, now: u64) -> bool {
        if down {
            self.keys_down.insert(key.to_string());
        } else {
            self.keys_down.remove(key);
        }
        let latched = self.controller.config().switches_on_key(key);
        // To every mode's operator: the keys held are the person's, and a mode
        // switched into has to have heard them.
        let driven = self.operators.key(key, down, now);
        latched || driven
    }

    /// The host's own stop control: a page's Stop button, or Escape.
    ///
    /// Lets go of everything every mode's operator holds, exactly as a release
    /// button does, and says whether there was an operator to let go.
    /// A host control rather than a pad button, because a page has no button of
    /// the bundle's to press on somebody's behalf: which one releases is the
    /// contract's to say, and a page that pressed `B` would be a page that knew.
    #[wasm_bindgen(js_name = letGo)]
    pub fn let_go(&mut self, now_us: f64) -> bool {
        if self.operators.is_empty() {
            return false;
        }
        self.operators.let_go();
        self.last_command_us = Some(now_us as u64);
        true
    }

    /// Whether this bundle turns raw input -- the pad and every key -- into a
    /// command itself, in any of its modes. When it does, those modes never
    /// read `set_command` and a page only has to hand over what it sees:
    /// `setAxis`, `setPad`, `padFrame` and `setKeyCode`. On-screen sticks are a
    /// pad too, by the same names.
    #[wasm_bindgen(js_name = takesRawInput)]
    pub fn takes_raw_input(&self) -> bool {
        !self.operators.is_empty()
    }

    /// A named quantity the simulator can supply beyond `set_state`.
    ///
    /// **A page should pass every one it can compute, every frame**, and this
    /// keeps what its contracts read. That is the whole of the design: the
    /// controller arrives inside an uploaded bundle, the page cannot know
    /// which terms that policy observes, and a page that passed only what
    /// today's policies need is a page an uploaded bundle outgrows without
    /// anything saying so.
    ///
    /// Read today:
    ///
    /// | name | values | frame |
    /// |---|---|---|
    /// | `base_lin_vel` | 3, m/s | body |
    ///
    /// A robot has no such number -- it needs a state estimator -- which is
    /// why no exported policy observes it; a web-only bundle may. Anything
    /// else is stored and ignored, and the return says which: `true` when a
    /// term here reads it.
    #[wasm_bindgen(js_name = setSignal)]
    pub fn set_signal(&mut self, name: &str, values: &[f32]) -> bool {
        self.signals.insert(name.to_string(), values.to_vec());
        match (name, values) {
            ("base_lin_vel", [x, y, z]) => {
                self.state.imu.lin_vel = [*x, *y, *z];
                self.state.imu.has_lin_vel = true;
                true
            }
            _ => false,
        }
    }

    /// What this controller reads, as JSON, for a page that wants to say so:
    /// `{rawInput, keys: [{name, code}], controls, signals: [...]}`. The keys
    /// are every key a mode switch or a controls block reads -- modifiers as
    /// their two keys -- with the browser
    /// code a page receives it under; `controls` is each mode's own account of
    /// its bindings -- what the robot's controller prints, a line a mode --
    /// joined into one line, modes that read the pad alike named together, or
    /// `null`. A page shows these rather than describing the bundle itself.
    #[wasm_bindgen(js_name = inputs)]
    pub fn inputs(&self) -> String {
        // Every key something reads: a switch's key or its modifier's keys, and
        // every key a mode's keyboard block binds or holds as a modifier.
        let keys: std::collections::BTreeSet<&str> = crate::vocabulary::keys()
            .into_iter()
            .filter(|k| self.controller.config().switches_on_key(k))
            .chain(self.operators.keys())
            .collect();
        let keys: Vec<_> = keys
            .into_iter()
            .map(|k| serde_json::json!({
                "name": k,
                "code": crate::vocabulary::key_codes(k).map(|(_, browser)| browser),
            }))
            .collect();
        serde_json::json!({
            "rawInput": self.takes_raw_input(),
            "keys": keys,
            "controls": (!self.operators.is_empty()).then(|| self.operators.describe()),
            "signals": ["base_lin_vel"],
        })
        .to_string()
    }

    /// Every control of the pad, what it does in each mode and which keys
    /// press it, as JSON (`guide.rs` has the shape). For a page that draws the
    /// pad: the account `inputs().controls` gives is one line of prose, and a
    /// page that took the contracts and the TOML apart to do better would be
    /// a second reading of the bindings.
    #[wasm_bindgen(js_name = padGuide)]
    pub fn pad_guide(&self) -> String {
        crate::guide::pad_guide(self.controller.config(), &self.operators, &self.controller.hook_reads())
            .to_string()
    }

    /// Whether a **pad** button is down, for a browser reading the Gamepad
    /// API. Same contract as `setKey`, with the robot pad service's names --
    /// `A` `B` `X` `Y` `LB` `RB` `menu` `home` `L3` `R3` -- so translate your
    /// button index into one of those and the two hosts agree.
    /// One raw stick or trigger, by the name the dictionary gives it.
    ///
    /// `Lx` `Ly` `Rx` `Ry` `LT` `RT`, exactly as the robot's pad service
    /// publishes them and as `controller/vocabulary.json` lists them. Pass the
    /// value the Gamepad API reports, unchanged: no sign, no deadzone, no
    /// scaling. All three of those are the contract's, and a host that applies
    /// its own is a second place they live.
    ///
    /// Returns whether the name is one this hardware reports, so a caller can
    /// tell a typo from a stick nobody is touching.
    #[wasm_bindgen(js_name = setAxis)]
    pub fn set_axis(&mut self, axis: &str, value: f32) -> bool {
        let Some(i) = crate::vocabulary::pad_axes().iter().position(|a| *a == axis) else {
            return false;
        };
        self.pad.axes[i] = value;
        self.pad.connected = true;
        true
    }

    /// One gamepad frame: turn its sticks into a command **using the policy's
    /// own contract**.
    ///
    /// The same call the robot makes (`DeviceHost::on_pad`), for the same
    /// reason: a sign corrected in a task's `controls.yaml` has to reach all
    /// three hosts, and it only does if none of them interprets the pad
    /// itself. A page that mapped sticks to a command before handing it over
    /// was a second implementation of the contract, and the two drifted --
    /// right down to which names the axes have.
    ///
    /// For a host with no pad -- a keyboard, an on-screen stick -- `setCommand`
    /// is still the way in, and latches nothing, because there is no frame to
    /// find an edge in.
    /// Deliberately **not** where the buttons are latched. `tick` observes
    /// them, once, and this used to observe them too -- so a `rise` or a `fall`
    /// fired here and was gone a line later, when `tick` re-observed the same
    /// unchanged pad and found no edge. A `toggle` survived it, because a
    /// toggle is remembered rather than re-derived, which is why entering a
    /// mode worked and the jump's `go` did not: `jump_phase` sat at 0 for the
    /// whole of a motion nobody could start, and no host reported anything.
    #[wasm_bindgen(js_name = padFrame)]
    pub fn pad_frame(&mut self, now_us: f64) {
        if !self.operators.is_empty() {
            // To every mode's operator, whichever mode is running: the shift
            // layer and the release are the person's.
            self.operators.pad(&self.pad);
            self.last_command_us = Some(now_us as u64);
        }
    }

    /// One pad button going down or coming up, by the dictionary's name --
    /// the four d-pad directions included, which the Gamepad API reports as
    /// buttons too. Returns whether anything binds it: a mode switch or a
    /// chord's modifier, or a mode's task that keeps the direction
    /// (`jumper.five_foot`'s arm).
    #[wasm_bindgen(js_name = setPad)]
    pub fn set_pad(&mut self, button: &str, down: bool) -> bool {
        if !self.pad.set_dpad(button, down) {
            let Some(i) = crate::vocabulary::pad_buttons().iter().position(|b| *b == button) else {
                return false;
            };
            self.pad.buttons[i] = down;
        }
        self.pad.connected = true;
        let named = |n: &Option<String>| n.as_deref() == Some(button);
        self.controller
            .config()
            .buttons
            .iter()
            .any(|b| named(&b.pad) || (b.pad.is_some() && named(&b.with)))
            || self.operators.iter().any(|(_, o)| o.spec().pad_uses().contains(button))
    }

    /// Every binding, as JSON: `[{name, pad, key, on, with, from, leaves}]`.
    /// For a host drawing a keypad or wiring a gamepad. A binding is one
    /// device's -- a pad button or a key -- and two with one name share a
    /// latch.
    #[wasm_bindgen(js_name = bindings)]
    pub fn bindings(&self) -> String {
        let v: Vec<_> = self
            .controller
            .config()
            .buttons
            .iter()
            .map(|b| serde_json::json!({
                "name": b.name, "pad": b.pad, "key": b.key,
                "on": b.on.to_string(), "with": b.with, "from": b.from, "leaves": b.leaves,
            }))
            .collect();
        serde_json::to_string(&v).unwrap_or_else(|_| "[]".into())
    }

    /// The modes that run a model, so a host can check it has a session for
    /// every one before the FSM asks for one mid-episode.
    /// Replay a bundle's `reference.json` and report where this host differs.
    ///
    /// The controller only -- it drives the recorded action back in, so the
    /// numbers are the observation, the decode and the clamp, never the model.
    /// A caller that wants its inference backend checked runs the recorded
    /// observations through it and compares against `act` itself.
    ///
    /// Destroys this controller's state: it is driven through somebody else's
    /// frames. Build a fresh one to check, and a fresh one to run.
    /// Returns JSON: `{frames, inferences, observation, target, divergent,
    /// modeMismatches}`. JSON rather than a struct because the numbers go
    /// straight into a test assertion and a panel, and neither wants a
    /// generated wrapper type.
    #[wasm_bindgen(js_name = checkReference)]
    pub fn check_reference(&mut self, text: &str) -> Result<String, JsValue> {
        let reference = crate::reference::Reference::parse(text)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let report = reference
            .replay(&mut self.controller)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(serde_json::json!({
            "frames": report.frames,
            "inferences": report.inferences,
            "observation": report.observation.diff,
            "target": report.target.diff,
            "divergent": report.divergent.iter()
                .map(|(t, w)| serde_json::json!({"term": t, "worst": w.diff}))
                .collect::<Vec<_>>(),
            "modeMismatches": report.mode_mismatches.iter()
                .map(|(i, want, got)| serde_json::json!({"frame": i, "expected": want,
                                                         "actual": got}))
                .collect::<Vec<_>>(),
        })
        .to_string())
    }

    pub fn modes(&self) -> Vec<String> {
        self.controller
            .config()
            .states
            .iter()
            .filter(|s| s.has_model())
            .map(|s| s.name.clone())
            .collect()
    }

    /// The active mode. For a panel that would otherwise read "running" through
    /// a two-second ramp it is not running through.
    pub fn mode(&self) -> String {
        self.controller.state().name.clone()
    }

    /// True only while the policy is producing output, as opposed to holding or
    /// ramping toward it.
    pub fn is_running_policy(&self) -> bool {
        self.controller.is_running_policy()
    }
}

fn fresh(last: Option<u64>, now: u64, timeout_us: u64) -> bool {
    last.is_some_and(|t| now.saturating_sub(t) < timeout_us)
}

fn js<E: std::fmt::Display>(e: E) -> JsError {
    JsError::new(&e.to_string())
}
