//! Open a bundle directory: one reader, for every host that has a filesystem.
//!
//! `scripts/deploy.py` writes a directory — `controller.toml`, one contract and one
//! model per mode, `bundle.json` listing all of it — and until this existed
//! every consumer opened it by hand. The preflight had its own reader, the board
//! would have had a second, and the two would have agreed right up until the
//! format changed. That is the same duplication the crate itself exists to
//! prevent, applied to the crate's own file format, so it is fixed the same way:
//! the format has one reader and every host calls it.
//!
//! Not compiled for `wasm32`. A browser receives a bundle as bytes over a
//! `postMessage`, has no filesystem to point at, and would not link `std::fs`
//! anyway — `web.rs` takes the parsed strings instead, which is why
//! `FsmConfig::parse` and `Contract::from_str` are separate from this.
//!
//! What this refuses is chosen the same way as everything else here: each one is
//! a bundle that *loads and then behaves wrongly*.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use crate::config::FsmConfig;
use crate::layout::Contract;

/// Written by `scripts/deploy.py`. A reader that does not know the string
/// should say so rather than guess which fields it is missing.
pub const SCHEMA: &str = "kk-policy-bundle/1";

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

fn fail<T>(msg: impl Into<String>) -> Result<T, Error> {
    Err(Error(msg.into()))
}

/// Which host is opening the bundle.
///
/// One bundle carries every host: `controller.toml`, the contracts and
/// `reference.json` once, each model in every format a host runs, and each
/// host's build of this crate under `runtime/`. What differs between hosts is
/// which model format they load and which controller they are -- so the host
/// says which it is, rather than the bundle saying who it was packed for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// `.rknn` for the NPU, and `controller` -- this crate cross-compiled for
    /// aarch64, which on the board is the whole program rather than a library
    /// something else links.
    Board,
    /// ONNX plus `controller.wasm` and `controller.js`, for a browser.
    Web,
    /// ONNX plus `controller.so`, for `play --app`.
    Mjlab,
}

impl Target {
    /// The name `bundle.json` keys this host's runtime by.
    pub fn name(&self) -> &'static str {
        match self {
            Target::Board => "board",
            Target::Web => "web",
            Target::Mjlab => "mjlab",
        }
    }

    /// The model format this host runs: the NPU's `.rknn` on the board, ONNX
    /// everywhere else.
    pub fn model_format(&self) -> &'static str {
        match self {
            Target::Board => "rknn",
            Target::Web | Target::Mjlab => "onnx",
        }
    }

    /// Whether this host has to get the joint stops from the bundle.
    ///
    /// The board is the only one with no model of the robot. The other two read
    /// the stops off the model they loaded, so a contract without them is still
    /// perfectly good there.
    pub fn needs_joint_limits(&self) -> bool {
        matches!(self, Target::Board)
    }
}

/// A bundle, read and checked, with everything a host needs to build a
/// controller — except the tuning that belongs to the machine rather than to
/// the policy, which the host still supplies.
#[derive(Debug)]
pub struct Bundle {
    pub dir: PathBuf,
    pub target: Target,
    pub cfg: FsmConfig,
    /// The robot's joint order, from the contracts. One bundle, one robot.
    pub wire: Vec<String>,
    /// One per mode, keyed by **state name**. `Contract` is deliberately not
    /// `Clone` — two modes sharing one would share its observation history.
    pub contracts: HashMap<String, Contract>,
    /// Mode -> the model file on disk, already confirmed to exist.
    pub models: BTreeMap<String, PathBuf>,
    /// What the bundler recorded plus what opening it found. Never fatal, never
    /// silent: a host is expected to put these somewhere a person reads.
    pub notes: Vec<String>,
}

impl Bundle {
    /// Read and check a bundle directory, as `target` will run it.
    pub fn open(dir: impl AsRef<Path>, target: Target) -> Result<Self, Error> {
        let dir = dir.as_ref().to_path_buf();
        let manifest: serde_json::Value = read_json(&dir.join("bundle.json"))?;

        match manifest["schema"].as_str() {
            Some(SCHEMA) => {}
            Some(other) => {
                return fail(format!(
                    "bundle.json says schema '{other}'; this reader knows '{SCHEMA}'"
                ))
            }
            None => return fail("bundle.json has no `schema`"),
        }

        let fsm_path = dir.join("controller.toml");
        let fsm_text = std::fs::read_to_string(&fsm_path)
            .map_err(|e| Error(format!("cannot read {}: {e}", fsm_path.display())))?;
        let cfg = FsmConfig::parse(&fsm_text).map_err(|e| Error(format!("controller.toml: {e}")))?;

        let entries = manifest["modes"].as_object().cloned().unwrap_or_default();
        let declared: Vec<String> =
            cfg.states.iter().filter(|s| s.has_model()).map(|s| s.name.clone()).collect();
        for name in &declared {
            if !entries.contains_key(name) {
                return fail(format!(
                    "bundle.json has no entry for mode '{name}', which names a model"
                ));
            }
        }

        let mut wire: Option<Vec<String>> = None;
        let mut contracts = HashMap::new();
        let mut models = BTreeMap::new();
        let mut notes: Vec<String> = manifest["notes"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let mut unconverted = Vec::new();

        // Sorted, so a bundle with two broken modes reports the same one first
        // every time. A message that moves is a message nobody trusts.
        for name in entries.keys() {
            let entry = &entries[name];
            let path = dir.join(str_field(entry, "contract")?);
            let text = std::fs::read_to_string(&path)
                .map_err(|e| Error(format!("mode '{name}': cannot read {}: {e}", path.display())))?;
            let raw: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| Error(format!("mode '{name}': {}: {e}", path.display())))?;
            let names: Vec<String> = raw["wire_joint_order"]
                .as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            if names.is_empty() {
                return fail(format!("mode '{name}': its contract has no wire_joint_order"));
            }
            // One robot, so one wire order. Two contracts disagreeing is a
            // bundle nobody can load: a controller is built once, with one joint
            // list, and the second mode would be paired by name against names
            // that are not there.
            match &wire {
                None => wire = Some(names.clone()),
                Some(first) if *first != names => {
                    return fail(format!(
                        "mode '{name}' expects a different robot: its wire_joint_order does \
                         not match the first mode's. One bundle drives one robot."
                    ))
                }
                Some(_) => {}
            }
            let mut contract = Contract::from_str(&text, &names)
                .map_err(|e| Error(format!("mode '{name}': {e}")))?;

            // A reference-guided policy is not runnable without its recording,
            // so the companion table is as required as the model. Beside the
            // contract, by the name the contract gives -- a bundler that copied
            // the contract and forgot this produces a mode that loads and
            // refuses at the first tick, which is the right time to find out
            // but the wrong place.
            if let Some(spec) = contract.layout.reference.clone() {
                let rp = path.with_file_name(&spec.file);
                let rt = std::fs::read_to_string(&rp).map_err(|e| {
                    Error(format!(
                        "mode '{name}': its contract names the reference table '{}', \
                         which cannot be read at {}: {e}\n\
                         The policy is a residual on that recording and cannot run \
                         without it.",
                        spec.file,
                        rp.display()
                    ))
                })?;
                contract
                    .attach_reference(&rt, &names)
                    .map_err(|e| Error(format!("mode '{name}': {e}")))?;
            }

            if target.needs_joint_limits() && contract.joint_limits.is_none() {
                // The board is the one host with no model of the robot. Without
                // the stops here it clamps commanded targets to a number in its
                // own config, and the number it held was `[-3.30, 3.30]` for
                // every joint — wide enough that the clamp never fired on twenty
                // of this robot's twenty-two.
                return fail(format!(
                    "mode '{name}': its contract carries no joint limits, so this host would \
                     have nothing to clamp joint targets to.\n\
                     Re-export it: python scripts/export.py --task <id> --checkpoint <ckpt>\n\
                     (a browser and `play` do without it -- a simulator has the model -- \
                     but a bundle is opened as every host it carries, the board included.)"
                ));
            }

            // This host's format, and on the board, ONNX when a mode was never
            // converted: not fatal, because a controller that refuses to boot
            // is a robot that cannot even be told to hold its pose. It runs
            // that mode on a stub and says so.
            let formats = &entry["models"];
            if !formats.is_object() {
                return fail(format!(
                    "mode '{name}' has no `models`: a map from a model format to its file"
                ));
            }
            let file = match formats[target.model_format()].as_str() {
                Some(file) => file,
                None if target == Target::Board => match formats["onnx"].as_str() {
                    Some(file) => {
                        unconverted.push(name.clone());
                        file
                    }
                    None => return fail(format!("mode '{name}' carries no model the board can load")),
                },
                None => {
                    return fail(format!(
                        "mode '{name}' carries no {} model, which the {} runs",
                        target.model_format(),
                        target.name()
                    ))
                }
            };
            let model = dir.join(file);
            if !model.is_file() {
                return fail(format!("mode '{name}': {} is missing", model.display()));
            }
            contracts.insert(name.clone(), contract);
            models.insert(name.clone(), model);
        }

        let Some(wire) = wire else {
            return fail("this bundle declares no modes, so there is nothing to run");
        };

        // Where the two halves meet.
        //
        // A bundle's controls come from two files by design: `controller.toml`
        // says which button switches **between** modes, each mode's contract
        // says what the sticks do **inside** one. Together they are the whole of
        // how this controller consumes a pad -- and nothing had ever looked at
        // them together, so one physical button could be both. Pressing it would
        // hand the command back to the sampler *and* change mode, and neither
        // half is wrong on its own, which is why neither could catch it.
        //
        // One check, `operator::check_switches`, which every host's operators
        // run as well (`Operators::beside`): a switch that may be pressed in a
        // mode, on a control -- a pad button, a key, a modifier -- that mode's
        // own controls use. Since 2026-09-29 a switch says with `from` which
        // modes it is pressed in, so Space can be the jump's from the default
        // mode and the claw's in a claw mode.
        for (mode, contract) in &contracts {
            let Some(spec) = contract.layout.controller.as_ref() else {
                continue;
            };
            if let Err(why) = crate::operator::check_switches(&cfg, mode, spec) {
                return fail(why);
            }
        }
        if !unconverted.is_empty() {
            // Not fatal. `rknn::make_engine` will stub these and the host will
            // report the stub, which is a robot that stands rather than a robot
            // that will not start — and a controller that refuses to boot is a
            // robot that cannot even be told to hold its pose.
            notes.push(format!(
                "still ONNX, so these modes will run on a stub and hold the home pose: {}",
                unconverted.join(", ")
            ));
        }
        Ok(Bundle { dir, target, cfg, wire, contracts, models, notes })
    }

    /// How often a host has to call `tick` for this bundle: the **fastest**
    /// policy's rate.
    ///
    /// Not a rate every mode shares. Each mode infers on its own period, from
    /// its own contract (`ModeRuntime::infer_period_us`), and the loop only has
    /// to be quick enough for the quickest -- a 50 Hz policy in a bundle whose
    /// loop runs at 200 Hz simply infers every fourth tick, which is what a
    /// slower policy beside a faster one already did.
    ///
    /// **This used to require every mode to agree**, and refused a bundle
    /// holding `jumper.tripod` at 50 Hz beside `jumper.jump` at 200. The reason
    /// given was the output filter -- one filter for the controller rather than
    /// one per mode -- and the filter is the one thing this cannot break:
    /// `ema_alpha_from_cutoff` is rate-invariant by construction, which is the
    /// entire point of writing it as a cutoff rather than as a coefficient. The
    /// field it feeds says so too (`Setup::output_rate_hz`: "sets the output
    /// filter and nothing else -- the policy rate comes from each mode's own
    /// contract"), so the refusal contradicted a comment two files away.
    ///
    /// What the agreement did cost was real: `jumper.jump` runs at 200 Hz
    /// deliberately, because its push-off is 40 ms and at 50 Hz that is two
    /// command steps. Requiring agreement meant either retraining it slower --
    /// throwing away the reason it exists -- or shipping it in a bundle of its
    /// own, which no operator can switch into.
    pub fn output_rate_hz(&self) -> Result<f64, Error> {
        let mut fastest: Option<(f64, &str)> = None;
        for (name, c) in &self.contracts {
            let hz = c.layout.control.control_hz;
            if hz <= 0.0 {
                return fail(format!("mode '{name}' declares control_hz {hz}"));
            }
            if fastest.is_none_or(|(best, _)| hz > best) {
                fastest = Some((hz, name));
            }
        }
        fastest.map(|(hz, _)| hz).ok_or_else(|| Error("no modes".into()))
    }

    /// The gait clock's period and gating threshold, from whichever mode
    /// observes `gait_phase`. `None` when no mode does.
    pub fn gait(&self) -> Option<(f64, Option<f32>)> {
        self.contracts.values().find_map(|c| {
            let term = c.layout.observation.terms.iter().find(|t| t.name == "gait_phase")?;
            let period = term.params.get("period")?.as_f64()?;
            let gate = term.params.get("command_threshold").and_then(|v| v.as_f64());
            Some((period, gate.map(|g| g as f32)))
        })
    }

    /// The hard stops in wire order. Present on every mode or on none — checked
    /// when the bundle was opened.
    pub fn joint_limits(&self) -> Option<(&[f32], &[f32])> {
        self.contracts
            .values()
            .find_map(|c| c.joint_limits.as_ref())
            .map(|(lo, hi)| (lo.as_slice(), hi.as_slice()))
    }

    /// The reference vectors, if this bundle carries any.
    ///
    /// `Ok(None)` when there is no `reference.json` -- older bundles, and any
    /// built on a machine without onnxruntime. `Err` when there is one and it
    /// cannot be read, because a check that silently does not run is worse than
    /// no check: it reports agreement it never measured.
    pub fn reference(&self) -> Result<Option<crate::reference::Reference>, Error> {
        let path = self.dir.join("reference.json");
        if !path.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error(format!("cannot read {}: {e}", path.display())))?;
        crate::reference::Reference::parse(&text)
            .map(Some)
            .map_err(|e| Error(e.to_string()))
    }

    /// How each mode's contract turns a person into its command: one operator
    /// per mode that describes controls, empty for a bundle nobody steers.
    ///
    /// Each from its own contract and never compared across modes: different
    /// policies take different controls, at their own ranges. See
    /// `operator::Operators`, which also says why this once required them to
    /// be identical and what that cost.
    pub fn operators(&self) -> Result<crate::operator::Operators, Error> {
        crate::operator::Operators::of(self.contracts.iter().map(|(n, c)| (n.as_str(), c)))
            .and_then(|o| o.beside(&self.cfg))
            .map_err(Error)
    }

    /// The home pose in wire order.
    pub fn default_wire(&self) -> &[f32] {
        self.contracts
            .values()
            .next()
            .map(|c| c.default_wire.as_slice())
            .unwrap_or(&[])
    }
}

/// What a bundle cannot tell a host, because it belongs to the machine rather
/// than to the policy.
///
/// Deliberately small, and it shrinks as the contract grows: `joint_limits`
/// used to live here, in a config file beside a binary, where nothing held it
/// against the robot it described. Anything a simulator can measure belongs in
/// the contract instead of in this struct.
#[cfg(feature = "device")]
#[derive(Debug, Clone)]
pub struct Tuning {
    /// Maximum change in a joint target per control tick, rad. 0 disables.
    /// A property of the actuators, not of the policy.
    pub joint_pos_rate_limit: f32,
    pub scales: crate::obs::Scales,
    pub obs_clip: f32,
    pub action_clip: f32,
    pub action_smoothing: f32,
    /// The command block's vocabulary, in order. Per-controller today; a
    /// per-mode vocabulary would come from each contract's controls block.
    pub command_terms: Vec<String>,
    /// Whether this machine runs a base-velocity estimator. Without one,
    /// `base_lin_vel` is absent rather than zero, and a contract asking for it
    /// is refused instead of fed a number nobody measured.
    pub has_velocity_estimator: bool,
    /// Seconds from entering a reference-guided mode to its `go`.
    ///
    /// **A placeholder.** The motion this keys is 40 ms wide at the push-off
    /// and should start when a person presses something; until that binding
    /// exists, it runs on a timer from mode entry. Long enough that the robot
    /// has settled into the stand first.
    pub reference_go_delay_s: f64,
}

#[cfg(feature = "device")]
impl Default for Tuning {
    fn default() -> Self {
        Self {
            joint_pos_rate_limit: 0.0,
            scales: crate::obs::Scales::default(),
            obs_clip: 100.0,
            action_clip: 100.0,
            action_smoothing: 0.0,
            command_terms: vec!["lin_vel_x".into(), "lin_vel_y".into(), "yaw_rate".into()],
            has_velocity_estimator: false,
            reference_go_delay_s: crate::trajectory::DEFAULT_GO_DELAY_S,
        }
    }
}

#[cfg(feature = "device")]
impl Bundle {
    /// Build the host the robot's loop drives.
    ///
    /// Everything the policy knows about itself comes from the bundle; `tuning`
    /// is the rest. The engines are built here rather than by the caller so
    /// that a mode which cannot load its model becomes a stub **with its reason
    /// recorded** -- `DeviceHost::stub_modes` is how the robot says "I am
    /// standing here holding a pose on purpose" instead of looking broken.
    ///
    /// `now` is the same clock the loop will pass to `pump`.
    pub fn into_host(
        self,
        tuning: Tuning,
        now: crate::fsm::Micros,
    ) -> Result<crate::device::DeviceHost, Error> {
        let output_rate_hz = self.output_rate_hz()?;
        let operators = self.operators()?;
        let (gait_period, gait_gate_threshold) = self.gait().unwrap_or((0.32, None));
        let Some((lo, hi)) = self.joint_limits() else {
            // Only reachable for a bundle opened as a simulator and handed to a
            // board: `open` as the board refuses a contract without them.
            return fail(
                "this bundle carries no joint limits, so nothing here could clamp a joint \
                 target. Open it as the board, which refuses a contract without them.",
            );
        };
        let robot = crate::action::RobotLimits {
            default_wire: self.default_wire().to_vec(),
            joint_pos_lo: lo.to_vec(),
            joint_pos_hi: hi.to_vec(),
            joint_pos_rate_limit: tuning.joint_pos_rate_limit,
        };

        let mut engines: HashMap<String, Box<dyn crate::rknn::Engine>> = HashMap::new();
        let mut stubs: Vec<(String, String)> = Vec::new();
        for (mode, contract) in &self.contracts {
            let (engine, why) = crate::rknn::make_engine(self.models.get(mode).map(|p| p.as_path()),
                                                         contract);
            if let Some(why) = why {
                stubs.push((mode.clone(), why));
            }
            engines.insert(mode.clone(), engine);
        }

        let setup = crate::control::Setup {
            robot,
            scales: tuning.scales,
            obs_clip: tuning.obs_clip,
            action_clip: tuning.action_clip,
            action_smoothing: tuning.action_smoothing,
            command_terms: tuning.command_terms,
            gait_period,
            gait_gate_threshold,
            reference_go_delay_s: tuning.reference_go_delay_s,
            output_rate_hz,
            source: crate::source::SourceCapability::device(tuning.has_velocity_estimator),
        };
        let controller = crate::control::Controller::new(self.cfg, setup, self.contracts, now)
            .map_err(|e| Error(format!("{e}")))?;
        let mut host = crate::device::DeviceHost::with_operators(controller, engines, operators);
        for (mode, why) in stubs {
            host.note_stub(&mode, why);
        }
        Ok(host)
    }
}

fn read_json(path: &Path) -> Result<serde_json::Value, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error(format!("cannot read {}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error(format!("{}: {e}", path.display())))
}

fn str_field(entry: &serde_json::Value, field: &str) -> Result<String, Error> {
    entry[field]
        .as_str()
        .map(String::from)
        .ok_or_else(|| Error(format!("bundle.json: a mode entry has no `{field}`")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bundle on disk, written the way `scripts/deploy.py` writes one.
    ///
    /// Built here rather than committed as a fixture: a fixture that drifted
    /// from the bundler would make every test below pass while the reader
    /// rotted. What keeps the two honest in the other direction is
    /// `tests/test_bundle.py`, which runs the real bundler and then opens the
    /// result with `check_bundle`.
    struct Fixture {
        dir: PathBuf,
        target: Target,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    const JOINTS: [&str; 2] = ["a", "b"];

    fn contract(with_limits: bool) -> String {
        let limits = if with_limits {
            r#""joint_limits": {"a": [-1.0, 1.0], "b": [-1.0, 1.0]},"#
        } else {
            ""
        };
        format!(
            r#"{{
  "obs_joint_order": ["a", "b"],
  "wire_joint_order": ["a", "b"],
  "action_joint_order": ["a", "b"],
  "action_scale": 0.25,
  "default_joint_pos": {{"a": 0.0, "b": 0.1}},
  {limits}
  "control": {{"kp": 10.0, "kd": 0.5, "effort_limit": 1.0, "control_hz": 50.0}},
  "observation": {{"dim": 4, "terms": [{{"name": "joint_pos", "dim": 2}},
                                       {{"name": "joint_vel", "dim": 2}}]}},
  "action": {{"dim": 2}}
}}"#
        )
    }

    const FSM: &str = r#"
[fsm]
initial_state = "walk"
warm_start_ref = "walk"
safe_state = "safe"
tilt_limit = 3.15
state_timeout_ms = 200
command_timeout_ms = 500
mode_switch_ramp_s = 0.0
pose_reach_tol = 0.1
warm_start_duration_s = 0.0
ramp_kp = 0.15
ramp_kd = 0.01

[[fsm.state]]
name = "safe"
hold_current = true
kd = 0.01

[[fsm.state]]
name = "walk"
model = "walk.onnx"

[[fsm.rule]]
when = "feedback_stale"
enter = "safe"

[[fsm.rule]]
when = "always"
enter = "walk"
"#;

    /// A mode's `models`, from its one file: keyed by the file's own suffix.
    fn models(model: &str) -> String {
        let format = model.rsplit('.').next().unwrap();
        format!(r#"{{"{format}": "{model}"}}"#)
    }

    impl Fixture {
        fn new(target: &str, with_limits: bool, model: &str) -> Self {
            static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "mjrl-bundle-{}-{}",
                std::process::id(),
                N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let p = dir.as_path();
            std::fs::write(p.join("controller.toml"), FSM).unwrap();
            std::fs::write(p.join("walk.json"), contract(with_limits)).unwrap();
            std::fs::write(p.join(model), b"not a model, only its name").unwrap();
            std::fs::write(
                p.join("bundle.json"),
                format!(
                    r#"{{"schema": "{SCHEMA}", "notes": [],
                         "modes": {{"walk": {{"models": {}, "contract": "walk.json"}}}}}}"#,
                    models(model)
                ),
            )
            .unwrap();
            let target = match target {
                "board" => Target::Board,
                "web" => Target::Web,
                _ => Target::Mjlab,
            };
            Fixture { dir, target }
        }

        fn path(&self) -> &Path {
            &self.dir
        }

        fn write(&self, name: &str, text: &str) {
            std::fs::write(self.dir.join(name), text).unwrap();
        }

        fn open(&self) -> Result<Bundle, Error> {
            Bundle::open(self.path(), self.target)
        }
    }

    #[test]
    fn two_modes_may_train_at_different_rates() {
        // `jumper.jump` runs at 200 Hz because its push-off is 40 ms; `jumper.tripod`
        // at 50. Requiring them to agree meant either retraining the jump slower,
        // which is the reason it exists, or shipping it in a bundle of its own,
        // which no operator can switch into. The loop takes the faster and the
        // slower mode infers every fourth tick.
        let f = Fixture::new("board", true, "walk.rknn");
        f.write("fast.json", &contract(true).replace(r#""control_hz": 50.0"#, r#""control_hz": 200.0"#));
        f.write("walk.rknn", "not a model, only its name");
        f.write(
            "bundle.json",
            &format!(
                r#"{{"schema": "{SCHEMA}", "notes": [],
                     "modes": {{"walk": {{"models": {{"rknn": "walk.rknn"}}, "contract": "walk.json"}},
                                "fast": {{"models": {{"rknn": "walk.rknn"}}, "contract": "fast.json"}}}}}}"#
            ),
        );
        let b = f.open().unwrap();
        assert_eq!(b.output_rate_hz().unwrap(), 200.0, "the faster of the two");

        // Control: the check is on the *max*, not on the first read, so the
        // same pair in the other order gives the same answer. A `find` over a
        // HashMap would pass this half the time.
        assert_eq!(b.contracts["walk"].layout.control.control_hz, 50.0);
        assert_eq!(b.contracts["fast"].layout.control.control_hz, 200.0);
    }

    #[test]
    fn a_rate_that_could_not_be_a_rate_is_still_refused() {
        let f = Fixture::new("board", true, "walk.rknn");
        f.write("walk.json", &contract(true).replace(r#""control_hz": 50.0"#, r#""control_hz": 0.0"#));
        let e = f.open().unwrap().output_rate_hz().unwrap_err();
        assert!(format!("{e}").contains("control_hz 0"), "{e}");
    }

    #[test]
    fn a_whole_bundle_comes_back() {
        let f = Fixture::new("board", true, "walk.rknn");
        let b = f.open().unwrap();
        assert_eq!(b.target, Target::Board);
        assert_eq!(b.wire, JOINTS);
        assert_eq!(b.output_rate_hz().unwrap(), 50.0);
        assert_eq!(b.default_wire(), &[0.0, 0.1]);
        let (lo, hi) = b.joint_limits().unwrap();
        assert_eq!((lo, hi), (&[-1.0f32, -1.0][..], &[1.0f32, 1.0][..]));
        assert!(b.models["walk"].ends_with("walk.rknn"));
        assert!(b.notes.is_empty(), "{:?}", b.notes);
        assert!(b.gait().is_none(), "no mode observes gait_phase");
    }

    /// The failure this module was written for.
    ///
    /// Both simulators read the stops off the model they loaded. The board has
    /// no model, so without them in the contract it clamps to a number in its
    /// own config -- and the number it held was `[-3.30, 3.30]` for all 22
    /// joints, which is no clamp at all on 20 of them.
    #[test]
    fn a_board_bundle_without_joint_limits_is_refused() {
        let missing = Fixture::new("board", false, "walk.rknn");
        let e = missing.open().unwrap_err().to_string();
        assert!(e.contains("no joint limits"), "{e}");
        assert!(e.contains("export.py"), "it has to say how to fix it: {e}");

        // The control group. Same contract, and a simulator takes it happily:
        // a simulator has the model, so the stops are not the bundle's job.
        let sim = Fixture::new("web", false, "walk.onnx");
        let b = sim.open().unwrap();
        assert!(b.joint_limits().is_none());
    }

    #[test]
    fn an_unconverted_board_bundle_loads_and_says_so() {
        // Not fatal: a controller that refuses to boot is a robot that cannot
        // even be told to hold its pose. It runs on a stub and reports it.
        let f = Fixture::new("board", true, "walk.onnx");
        let b = f.open().unwrap();
        assert_eq!(b.notes.len(), 1, "{:?}", b.notes);
        assert!(b.notes[0].contains("still ONNX"), "{:?}", b.notes);

        // ...and a simulator bundle full of ONNX is simply a simulator bundle.
        assert!(Fixture::new("web", true, "walk.onnx").open().unwrap().notes.is_empty());
    }

    #[test]
    fn a_bundle_that_lies_about_itself_does_not_open() {
        let f = Fixture::new("web", true, "walk.onnx");

        f.write("bundle.json", r#"{"schema": "kk-policy-bundle/9", "modes": {}}"#);
        assert!(f.open().unwrap_err().to_string().contains("this reader knows"));

        // A state naming a model that the manifest never lists. It loads, runs,
        // and holds forever the first time the cascade enters that state.
        f.write("bundle.json", &format!(r#"{{"schema": "{SCHEMA}", "modes": {{}}}}"#));
        assert!(f.open().unwrap_err().to_string().contains("no entry for mode 'walk'"));

        f.write(
            "bundle.json",
            &format!(
                r#"{{"schema": "{SCHEMA}", "modes": {{"walk": {{"models": {{"onnx": "gone.onnx"}},
                     "contract": "walk.json"}}}}}}"#
            ),
        );
        assert!(f.open().unwrap_err().to_string().contains("is missing"));
    }

    /// One bundle drives one robot. Two contracts with different joint lists
    /// produce a controller built once against one of them, with the other
    /// mode's action indices pointing at joints that are not there.
    #[test]
    fn two_modes_cannot_describe_two_robots() {
        let f = Fixture::new("web", true, "walk.onnx");
        f.write("slow.json", &contract(true).replace(r#""b""#, r#""c""#));
        f.write("slow.onnx", "x");
        f.write(
            "bundle.json",
            &format!(
                r#"{{"schema": "{SCHEMA}", "modes": {{
                     "walk": {{"models": {{"onnx": "walk.onnx"}}, "contract": "walk.json"}},
                     "slow": {{"models": {{"onnx": "slow.onnx"}}, "contract": "slow.json"}}}}}}"#
            ),
        );
        f.write("controller.toml", &FSM.replace(
            "[[fsm.rule]]\nwhen = \"feedback_stale\"",
            "[[fsm.state]]\nname = \"slow\"\nmodel = \"slow.onnx\"\n\n\
             [[fsm.rule]]\nwhen = \"feedback_stale\"",
        ));
        let e = f.open().unwrap_err().to_string();
        assert!(e.contains("One bundle drives one robot"), "{e}");
    }
    /// The whole point, end to end: the stops the board clamps to are the ones
    /// the simulator measured, not a number in this process.
    ///
    /// Written with limits that are *not* the old `[-3.30, 3.30]` and not the
    /// crate's defaults, because the bug being pinned is a plumbing bug --
    /// `into_host` reading the contract, finding limits, and then building the
    /// decoder from something else would pass every other test here.
    #[cfg(feature = "device")]
    #[test]
    fn the_host_clamps_to_the_bundles_own_limits() {
        let f = Fixture::new("board", true, "walk.rknn");
        let host = f.open().unwrap().into_host(Tuning::default(), 0).unwrap();
        let (lo, hi) = host.controller().limits();
        assert_eq!((lo, hi), (&[-1.0f32, -1.0][..], &[1.0f32, 1.0][..]));

        // The model is not a model, so the mode is on a stub -- and the reason
        // is recorded rather than left as a robot that stands there looking
        // broken.
        let stubs: Vec<_> = host.stub_modes().map(|(m, why)| (m.clone(), why.clone())).collect();
        assert_eq!(stubs.len(), 1, "{stubs:?}");
        assert_eq!(stubs[0].0, "walk");
        assert!(!stubs[0].1.is_empty());
    }

    /// Full stick must not ask for more than the *tightest* mode was trained
    /// for. One controller has one set of stick scales, and taking the first
    /// mode's -- or a number from a config file, which is what this replaced --
    /// asks some mode for a speed it has never seen. That looks like a policy
    /// that is bad at going fast, not like a miscalibrated stick.
    /// One control cannot be two things.
    ///
    /// A bundle's controls come from two files on purpose: the manifest says
    /// what switches **between** modes, each task's `controls.yaml` says what
    /// the pad and the keys do **inside** one. Neither is wrong on its own when
    /// they pick the same control, and neither could ever catch it -- the FSM
    /// never reads a contract's controls block and a contract never sees the
    /// FSM's. Opening the bundle is the first place both exist, so it is the
    /// only place the sum can be checked: the release button, the shift, the
    /// triggers and every key the controls bind.
    #[test]
    fn a_control_cannot_switch_modes_and_drive_at_once() {
        let f = Fixture::new("web", true, "walk.onnx");
        let op = r#""command_ranges": {"posture": {"pitch": [-0.26, 0.26], "height": [0.07, 0.15]}},
            "controller": {"schema": "operator_controller/2",
              "command": [{"term": "posture", "feeds": "posture_command",
                           "axes": [{"name": "pitch"}, {"name": "height", "rest": 0.107}]}],
              "devices": {
                "gamepad": {"scheme": "absolute", "deadzone": "device_reported_rescaled",
                            "release_button": "B", "shift": {"button": "R3", "gesture": "toggle"},
                            "axes": {"pitch": [{"source": "Ry", "sign": 1},
                                               {"source": "Rx", "sign": 1, "shifted": true}],
                                     "height": [{"source": "RT", "sign": 1},
                                                {"source": "LT", "sign": -1}]}},
                "keyboard": {"scheme": "keys", "full_after_s": 2.0,
                             "axes": {"pitch": {"+": ["key_k"], "-": ["key_i"]},
                                      "height": {"+": ["key_n"], "-": ["shift+key_m"]}},
                             "release": ["key_b"]}}},
            "control":"#;
        f.write("walk.json", &contract(true).replace(r#""control":"#, op));
        let bind = |source: &str| FSM.replace(
            "[[fsm.rule]]\nwhen = \"feedback_stale\"",
            &format!(
                "[[fsm.button]]\nname = \"slow\"\n{source}\non = \"toggle\"\n\n\
                 [[fsm.rule]]\nwhen = \"button:slow\"\nenter = \"walk\"\n\n\
                 [[fsm.rule]]\nwhen = \"feedback_stale\"",
            ),
        );

        for (source, word) in [(r#"pad = "R3""#, "pad's R3"), (r#"pad = "B""#, "pad's B")] {
            f.write("controller.toml", &bind(source));
            let e = f.open().unwrap_err().to_string();
            assert!(e.contains(word) && e.contains("'walk'"), "{source}: {e}");
        }
        for (source, word) in [(r#"key = "key_i""#, "uses key_i"),
                               ("key = \"key_1\"\nwith = \"shift\"", "uses shift")] {
            f.write("controller.toml", &bind(source));
            let e = f.open().unwrap_err().to_string();
            assert!(e.contains(word) && e.contains("'walk'"), "{source}: {e}");
        }

        // The control group: a button and a key the controls leave alone.
        for source in [r#"pad = "LB""#, r#"key = "keypad_1""#, "key = \"key_1\"\nwith = \"ctrl\""] {
            f.write("controller.toml", &bind(source));
            let b = f.open().unwrap_or_else(|e| panic!("{source}: controls the block does not read are free: {e}"));
            assert!(!b.operators().unwrap().is_empty());
        }
    }

    /// A d-pad direction a task keeps may also be the second half of a FSM
    /// chord, and only when the chord leaves the mode on the press.
    ///
    /// `jumper.five_foot` picks an arm preset on right alone while `menu` held
    /// and right is the dance. Each refusal below is a binding under which one
    /// press would reach both -- the switch and the task's hook -- and the first
    /// case is the control group: the same binding as the bundle ships, which
    /// has to open, or every refusal after it could be a check that refuses
    /// everything.
    #[test]
    fn a_task_s_d_pad_may_be_a_chord_s_second_half_and_nothing_more() {
        let f = Fixture::new("web", true, "walk.onnx");
        let op = r#""command_ranges": {"posture": {"pitch": [-0.26, 0.26], "height": [0.07, 0.15]}},
            "controller": {"schema": "operator_controller/2",
              "command": [{"term": "posture", "feeds": "posture_command",
                           "axes": [{"name": "pitch"}, {"name": "height", "rest": 0.107}]}],
              "task": {"arm_out": {"kind": "press", "does": "holds the arm out"}},
              "devices": {
                "gamepad": {"scheme": "absolute", "deadzone": "device_reported_rescaled",
                            "release_button": "B",
                            "axes": {"pitch": [{"source": "Ry", "sign": 1}],
                                     "height": [{"source": "RT", "sign": 1}]},
                            "task": {"arm_out": "dpad_right"}},
                "keyboard": {"scheme": "keys", "full_after_s": 2.0,
                             "axes": {"pitch": {"+": ["key_k"], "-": ["key_i"]},
                                      "height": {"+": ["key_o"], "-": ["key_q"]}},
                             "release": ["key_b"],
                             "task": {"arm_out": ["key_right"]}}}},
            "control":"#;
        f.write("walk.json", &contract(true).replace(r#""control":"#, op));
        // The chord's rule after `ahead`, a list of `(when, enter)` rules.
        let ordered = |ahead: &[(&str, &str)], binding: &str, enter: &str| {
            let ahead: String = ahead
                .iter()
                .map(|(w, e)| format!("[[fsm.rule]]\nwhen = \"{w}\"\nenter = \"{e}\"\n\n"))
                .collect();
            FSM.replace(
                "[[fsm.rule]]\nwhen = \"feedback_stale\"",
                &format!(
                    "[[fsm.button]]\nname = \"dance\"\npad = \"dpad_right\"\n{binding}\n\n\
                     {ahead}[[fsm.rule]]\nwhen = \"button:dance\"\nenter = \"{enter}\"\n\n\
                     [[fsm.rule]]\nwhen = \"feedback_stale\"",
                ),
            )
        };
        let bind = |binding: &str, enter: &str| ordered(&[], binding, enter);
        let chord = "with = \"menu\"\non = \"toggle\"";

        f.write("controller.toml", &bind("with = \"menu\"\non = \"toggle\"", "safe"));
        f.open().expect("a chord that leaves on the press shares the direction");

        f.write("controller.toml", &bind("on = \"toggle\"", "safe"));
        let e = f.open().unwrap_err().to_string();
        assert!(e.contains("'walk' keeps dpad_right for its task"), "{e}");

        f.write("controller.toml", &bind("with = \"menu\"\non = \"fall\"", "safe"));
        let e = f.open().unwrap_err().to_string();
        assert!(e.contains("on `fall`: only a `toggle` leaves"), "{e}");

        f.write("controller.toml", &bind("with = \"menu\"\non = \"toggle\"", "walk"));
        let e = f.open().unwrap_err().to_string();
        assert!(e.contains("keeps the cascade in 'walk' itself"), "{e}");

        // `rise` is true for one tick: the cascade leaves, comes back to walk
        // with the direction still down, and walk's task reads it.
        f.write("controller.toml", &bind("with = \"menu\"\non = \"rise\"", "safe"));
        let e = f.open().unwrap_err().to_string();
        assert!(e.contains("only a `toggle` leaves"), "{e}");

        f.write("controller.toml", &bind("with = \"menu\"\non = \"toggle\"\nevent = true", "safe"));
        let e = f.open().unwrap_err().to_string();
        assert!(e.contains("as an event a mode reads"), "{e}");

        f.write("controller.toml", &bind(chord, "@stay"));
        let e = f.open().unwrap_err().to_string();
        assert!(e.contains("keeps the cascade in 'walk' itself"), "{e}");

        // Order: a rule ahead of the chord that can hold walk -- entering it,
        // or staying while in it -- wins the press.
        for ahead in [("command_stale", "walk"), ("in_state:walk", "@stay"), ("command_stale", "@initial")] {
            f.write("controller.toml", &ordered(&[ahead], chord, "safe"));
            let e = f.open().unwrap_err().to_string();
            assert!(e.contains("keeps the cascade in 'walk' first"), "{ahead:?}: {e}");
        }
        // ... and one that cannot match while walk runs holds nothing: the
        // control group for the three above, and the shipped cascade's shape
        // (`in_state:safe` -> `@initial` ahead of every switch).
        for ahead in [("in_state:safe", "@initial"), ("warm_start_reached", "@warm_start_ref")] {
            f.write("controller.toml", &ordered(&[ahead], chord, "safe"));
            f.open().unwrap_or_else(|e| panic!("{ahead:?} refused: {e}"));
        }
    }

    /// A bundle opened as a simulator and handed to a board. The board's own
    /// `open` refuses a contract with no stops, so this is the one way left to
    /// build a device host without them, and the failure it would otherwise
    /// produce is a robot running with no clamp at all.
    #[cfg(feature = "device")]
    #[test]
    fn a_bundle_opened_for_another_host_cannot_become_a_device_host() {
        let f = Fixture::new("web", false, "walk.onnx");
        let e = match f.open().unwrap().into_host(Tuning::default(), 0) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a board accepted a bundle with no joint limits"),
        };
        assert!(e.contains("Open it as the board"), "{e}");
    }

    /// One bundle, every host: each opens the format it runs, and the board
    /// falls back to ONNX -- a stub, said out loud -- only where a mode was
    /// never converted. The control group is the same bundle opened as a
    /// browser, which takes the ONNX and has nothing to say about it.
    #[test]
    fn each_host_takes_its_own_models_from_one_bundle() {
        let f = Fixture::new("board", true, "walk.onnx");
        f.write("walk.rknn", "not a model, only its name");
        f.write(
            "bundle.json",
            &format!(
                r#"{{"schema": "{SCHEMA}", "modes": {{"walk": {{"contract": "walk.json",
                     "models": {{"onnx": "walk.onnx", "rknn": "walk.rknn"}}}}}}}}"#
            ),
        );
        let board = Bundle::open(f.path(), Target::Board).unwrap();
        assert!(board.models["walk"].ends_with("walk.rknn"));
        assert!(board.notes.is_empty(), "{:?}", board.notes);
        let web = Bundle::open(f.path(), Target::Web).unwrap();
        assert!(web.models["walk"].ends_with("walk.onnx"));

        // No `models` at all is refused by name rather than read as no model.
        f.write(
            "bundle.json",
            &format!(r#"{{"schema": "{SCHEMA}", "modes": {{"walk": {{"contract": "walk.json",
                     "model": "walk.onnx"}}}}}}"#),
        );
        let e = Bundle::open(f.path(), Target::Web).unwrap_err().to_string();
        assert!(e.contains("has no `models`"), "{e}");
    }
}
