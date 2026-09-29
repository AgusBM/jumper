//! The deployment contract: `layout.json`, as written by `scripts/export.py`.
//!
//! Every number the policy needs is read from this file. Nothing here is a
//! default that happens to match training -- a default that silently disagrees
//! with the policy is the failure mode this whole path exists to prevent, so the
//! fields that cannot be guessed have no default at all and fail the load.
//!
//! The one thing this file does **not** describe is the robot's wire order. That
//! comes from the TOML (`[robot] joint_names`), and `resolve` pairs the two **by
//! name**. Pairing them by index is the single most expensive mistake available
//! here: on this robot `obs_joint_order` is 20 long and the wire is 22, so an
//! index pairing is wrong from the fifth joint on, with nothing raised.

use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Control {
    pub kp: f32,
    pub kd: f32,
    pub effort_limit: f32,
    pub control_hz: f64,
    /// Cutoff for the output-side EMA on the published position. 0 (the default)
    /// means a pure zero-order hold between policy ticks.
    #[serde(default)]
    pub action_filter_cutoff_hz: f64,
}

#[derive(Debug, Deserialize)]
pub struct Term {
    pub name: String,
    pub dim: usize,
    /// Frames of this term stacked into the observation. Absent means "take the
    /// group default", which is `Observation::history_length`.
    #[serde(default)]
    pub history_length: Option<usize>,
    /// Control steps between two of those frames. Absent means 1: consecutive.
    ///
    /// **Not a detail of the history, a different history.** `jumper.posture`
    /// runs at 200 Hz and keeps five frames 4 steps apart, so its window reaches
    /// back 80 ms -- the window the five were chosen for at 50 Hz. A builder that
    /// ignored this field would emit exactly the right number of floats from a
    /// window a quarter as long, and nothing would say so. `scripts/export.py`
    /// writes it only when it is not 1, so an unstrided contract is unchanged.
    #[serde(default)]
    pub history_stride: Option<usize>,
    #[serde(default)]
    pub params: HashMap<String, serde_json::Value>,
}

impl Term {
    pub fn param_f64(&self, key: &str) -> Option<f64> {
        self.params.get(key).and_then(|v| v.as_f64())
    }
    pub fn param_str(&self, key: &str) -> Option<&str> {
        self.params.get(key).and_then(|v| v.as_str())
    }
}

#[derive(Debug, Deserialize)]
pub struct Observation {
    pub dim: usize,
    #[serde(default = "one")]
    pub history_length: usize,
    pub terms: Vec<Term>,
}

fn one() -> usize {
    1
}

#[derive(Debug, Deserialize)]
pub struct Action {
    pub dim: usize,
}

/// The one controller schema this crate reads: `operator_controller/2`, the
/// contract's transcription of a task's `controls.yaml`. See `operator.rs`.
///
/// There was a `twist_controller/1` as well -- one command, a stick per axis,
/// and a keyboard left to each host. It was retired and every task re-exported
/// in this one, so a contract naming it is refused by name rather than read as
/// the nearest thing: read as this, it would have no keyboard and no rests.
pub const OPERATOR_CONTROLLER: &str = "operator_controller/2";

/// Parse a contract's `controller` block, refusing any other schema by name.
fn parse_controller(raw: serde_json::Value) -> Result<crate::operator::OperatorSpec, Error> {
    match raw.get("schema").and_then(|v| v.as_str()) {
        Some(OPERATOR_CONTROLLER) => serde_json::from_value(raw)
            .map_err(|e| Error::Inconsistent(format!("controller block: {e}"))),
        other => Err(Error::Inconsistent(format!(
            "the controller block says schema {other:?}; this controller implements only \
             {OPERATOR_CONTROLLER:?}. Re-export the policy: its controls predate the one \
             schema every task now uses."
        ))),
    }
}

/// The command ranges the policy was trained against, per command group.
///
/// Here because the alternative is a number in a config file beside a binary,
/// which is where the stick scales lived: a full stick asking for more than the
/// curriculum ever commanded looks like a policy that is bad at going fast
/// rather than one being asked a question it has never seen.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CommandRanges {
    /// `lin_vel_x` / `lin_vel_y` / `ang_vel_z` / `heading`, each `[lo, hi]`.
    #[serde(default)]
    pub twist: HashMap<String, [f32; 2]>,
    /// Every other command term's, by term name -- `posture` for
    /// `jumper.posture`. Each axis `[lo, hi]`, a half-range already opened into
    /// a pair by the exporter.
    #[serde(flatten)]
    pub other: HashMap<String, HashMap<String, [f32; 2]>>,
}

impl CommandRanges {
    /// One command term's ranges, by the term's name.
    pub fn term(&self, name: &str) -> Option<&HashMap<String, [f32; 2]>> {
        if name == "twist" {
            Some(&self.twist)
        } else {
            self.other.get(name)
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Layout {
    /// Joints the observation carries, in order. Every per-joint term is this
    /// wide per frame.
    pub obs_joint_order: Vec<String>,
    /// Joints the action controls, in the order of the action vector.
    pub action_joint_order: Vec<String>,
    pub action_scale: f32,
    pub default_joint_pos: HashMap<String, f32>,
    /// `[lo, hi]` per joint, in the same radians as the home pose.
    ///
    /// Optional because exports predating it are still perfectly good for a
    /// simulator: a simulator has the model and reads the stops off it. The
    /// robot does not, which is why this is here at all -- see `joint_limits`
    /// on `Contract`.
    #[serde(default)]
    pub joint_limits: HashMap<String, [f32; 2]>,
    pub control: Control,
    pub observation: Observation,
    pub action: Action,
    /// Absent from exports predating it, and from any task with no commands.
    #[serde(default)]
    pub command_ranges: CommandRanges,
    /// How a person's input becomes a command, as it arrived. Parsed into
    /// `controller` by `Contract::from_str`, which is where a schema this
    /// controller does not implement can be refused with a sentence rather
    /// than a serde path.
    #[serde(default, rename = "controller")]
    pub controller_raw: Option<serde_json::Value>,
    /// Absent from any task with no controls file -- a dance has nobody
    /// steering it. See `operator.rs`.
    #[serde(skip)]
    pub controller: Option<crate::operator::OperatorSpec>,
    /// The recording this policy corrects, when it is a reference-guided one.
    ///
    /// Absent for every velocity policy, and that is the common case. Present
    /// means the policy is **not runnable** without the companion table the
    /// block names -- see `Contract::attach_reference`.
    #[serde(default)]
    pub reference: Option<crate::trajectory::ReferenceSpec>,
}

/// The contract resolved against one robot's wire order.
#[derive(Debug)]
pub struct Contract {
    pub layout: Layout,
    /// `obs_wire_idx[slot]` = wire index of observation joint `slot`.
    pub obs_wire_idx: Vec<usize>,
    /// `action_wire_idx[i]` = wire index driven by action element `i`.
    pub action_wire_idx: Vec<usize>,
    /// Home pose in **wire** order, covering every joint the robot has.
    pub default_wire: Vec<f32>,
    /// The robot's joint names in **wire** order: the order every `*_wire`
    /// vector here is in. Kept for a task's deploy hook (`hook.rs`), which
    /// reasons about joints by name -- a mirror pairs `LF_*` with `RF_*`.
    pub joint_names: Vec<String>,
    /// The hard stops in **wire** order, `(lo, hi)`, when the export carried
    /// them. `None` for an older contract, and the distinction matters: a host
    /// with no model of the robot has nothing else to clamp to, and a default
    /// wide enough to be safe on any robot is a default that never fires.
    pub joint_limits: Option<(Vec<f32>, Vec<f32>)>,
    /// Per-term stacked depth, one entry per term, group default applied.
    pub term_history: Vec<usize>,
    /// Per-term control steps between stacked frames, one entry per term.
    pub term_stride: Vec<usize>,
    /// The parsed recording, once a host has read the companion file.
    ///
    /// Separate from the layout because this crate does no I/O: a browser has
    /// no filesystem, so whoever obtained the bundle's bytes attaches the
    /// table. `None` with a `reference` block present is a contract that
    /// cannot be built into a runtime, and `obs.rs` says so rather than
    /// falling back to something that runs.
    pub reference: Option<std::sync::Arc<crate::trajectory::Trajectory>>,
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// A joint the layout names is not on this robot.
    UnknownJoint { field: &'static str, joint: String },
    /// The home pose does not cover a joint the robot has. Those joints still
    /// have to be held somewhere, so this is fatal rather than a warning.
    HomePoseIncomplete(Vec<String>),
    /// The layout contradicts itself. Exported bundles cannot do this; a bundle
    /// assembled by hand can.
    Inconsistent(String),
    /// The companion reference table is missing, unreadable, or describes a
    /// different recording than the contract does.
    Reference(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "cannot read layout: {e}"),
            Error::Json(e) => write!(f, "layout is not valid JSON: {e}"),
            Error::Reference(m) => write!(f, "{m}"),
            Error::UnknownJoint { field, joint } => write!(
                f,
                "{field} names '{joint}', which is not in [robot] joint_names. \
                 The layout and the robot config describe different robots."
            ),
            Error::HomePoseIncomplete(j) => write!(
                f,
                "default_joint_pos does not cover {j:?}; every joint on the wire \
                 needs a home position, including the ones the policy never sees"
            ),
            Error::Inconsistent(m) => write!(f, "layout is inconsistent: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl Contract {
    /// Load `layout.json` and pair it with this robot's wire order, by name.
    pub fn load(path: &std::path::Path, joint_names: &[String]) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path).map_err(Error::Io)?;
        Self::from_str(&text, joint_names)
    }

    pub fn from_str(text: &str, joint_names: &[String]) -> Result<Self, Error> {
        let mut layout: Layout = serde_json::from_str(text).map_err(Error::Json)?;
        let index: HashMap<&str, usize> = joint_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        let resolve = |names: &[String], field: &'static str| -> Result<Vec<usize>, Error> {
            names
                .iter()
                .map(|n| {
                    index.get(n.as_str()).copied().ok_or_else(|| Error::UnknownJoint {
                        field,
                        joint: n.clone(),
                    })
                })
                .collect()
        };
        let obs_wire_idx = resolve(&layout.obs_joint_order, "obs_joint_order")?;
        let action_wire_idx = resolve(&layout.action_joint_order, "action_joint_order")?;

        let missing: Vec<String> = joint_names
            .iter()
            .filter(|n| !layout.default_joint_pos.contains_key(n.as_str()))
            .cloned()
            .collect();
        if !missing.is_empty() {
            return Err(Error::HomePoseIncomplete(missing));
        }
        let default_wire: Vec<f32> =
            joint_names.iter().map(|n| layout.default_joint_pos[n]).collect();

        // All or nothing. A partial table would clamp some joints and silently
        // leave the rest to whatever the host guessed, which is the harder
        // failure of the two to see.
        let joint_limits = if layout.joint_limits.is_empty() {
            None
        } else {
            let missing: Vec<String> = joint_names
                .iter()
                .filter(|n| !layout.joint_limits.contains_key(n.as_str()))
                .cloned()
                .collect();
            if !missing.is_empty() {
                return Err(Error::Inconsistent(format!(
                    "joint_limits covers {} of {} joints; missing {missing:?}",
                    layout.joint_limits.len(),
                    joint_names.len()
                )));
            }
            let mut lo = Vec::with_capacity(joint_names.len());
            let mut hi = Vec::with_capacity(joint_names.len());
            for (i, n) in joint_names.iter().enumerate() {
                let [l, h] = layout.joint_limits[n];
                if !(l < h) {
                    return Err(Error::Inconsistent(format!(
                        "joint_limits[{n}] is [{l}, {h}], not [lo, hi]"
                    )));
                }
                if default_wire[i] < l || default_wire[i] > h {
                    // Every mode switch ramps to the home pose. Outside the
                    // clamp it is a pose the robot is told to reach and then
                    // stopped from holding, and the ramp waits on the measured
                    // pose, so it would wait forever.
                    return Err(Error::Inconsistent(format!(
                        "the home pose for {n} is {} but its limits are [{l}, {h}]",
                        default_wire[i]
                    )));
                }
                lo.push(l);
                hi.push(h);
            }
            Some((lo, hi))
        };

        if let Some(raw) = layout.controller_raw.take() {
            let spec = parse_controller(raw)?;
            spec.check(&layout.command_ranges).map_err(Error::Inconsistent)?;
            layout.controller = Some(spec);
        }

        if layout.action.dim != layout.action_joint_order.len() {
            return Err(Error::Inconsistent(format!(
                "action.dim {} != action_joint_order length {}",
                layout.action.dim,
                layout.action_joint_order.len()
            )));
        }

        // Isaac's term names -> the observation builder's own. The two agree on
        // everything but the command block, which each task names for what it
        // carries: locomotion `velocity_commands`, and the older jump layouts
        // `jump_command` / `jump_go`. All three feed the single `commands` term,
        // whose width comes from the state's `command_terms`.
        //
        // Ported from `rl-wbc-fsm/src/contract.cpp:27`, where it has been all
        // along. Without it this crate refuses every policy in circulation --
        // kk-rl-mjlab's exports and rl-wbc-fsm's own `locomotion.isaac_layout.json`
        // alike -- and it went unnoticed because the tests build their own
        // contracts, which named the term `commands` because that is what the
        // builder wanted. Found the first time the crate was pointed at a real
        // file, by the browser host.
        for term in &mut layout.observation.terms {
            if matches!(term.name.as_str(), "velocity_commands" | "jump_command" | "jump_go") {
                term.name = "commands".to_string();
            }
        }

        let group = layout.observation.history_length.max(1);
        let term_history: Vec<usize> = layout
            .observation
            .terms
            .iter()
            .map(|t| t.history_length.unwrap_or(group).max(1))
            .collect();
        let mut term_stride = Vec::with_capacity(term_history.len());
        for t in &layout.observation.terms {
            match t.history_stride {
                Some(0) => {
                    return Err(Error::Inconsistent(format!(
                        "term {} has history_stride 0; frames cannot be zero steps apart",
                        t.name
                    )))
                }
                s => term_stride.push(s.unwrap_or(1)),
            }
        }

        Ok(Contract {
            layout,
            obs_wire_idx,
            action_wire_idx,
            default_wire,
            joint_names: joint_names.to_vec(),
            joint_limits,
            term_history,
            term_stride,
            reference: None,
        })
    }

    /// Attach the companion table the `reference` block names.
    ///
    /// Split from `from_str` because this crate reads no files: the host that
    /// obtained the bundle's bytes is the one that can obtain these too. A
    /// contract with a block and no table is refused at runtime construction
    /// rather than here, so a tool that only inspects contracts still works.
    pub fn attach_reference(&mut self, text: &str, wire: &[String]) -> Result<(), Error> {
        let Some(spec) = self.layout.reference.clone() else {
            return Err(Error::Reference(
                "this contract has no `reference` block, so there is nothing for a \
                 reference table to belong to."
                    .to_string(),
            ));
        };
        let t = crate::trajectory::Trajectory::parse(spec, text, wire).map_err(Error::Reference)?;
        self.reference = Some(std::sync::Arc::new(t));
        Ok(())
    }

    pub fn obs_joints(&self) -> usize {
        self.obs_wire_idx.len()
    }
    pub fn action_dim(&self) -> usize {
        self.layout.action.dim
    }
    pub fn control_dt(&self) -> f64 {
        if self.layout.control.control_hz > 0.0 {
            1.0 / self.layout.control.control_hz
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mapping ported from `contract.cpp`, and the reason it had to be.
    ///
    /// Run against a `from_str` that passes term names through -- which is what
    /// this crate did until the browser host pointed it at a real file -- this
    /// fails, and so does every policy in circulation: kk-rl-mjlab's exports and
    /// `rl-wbc-fsm`'s own `locomotion.isaac_layout.json` both name the command
    /// block `velocity_commands`.
    ///
    /// It went unnoticed for so long because the tests around it build their own
    /// contracts, and named the term `commands` because that is what the builder
    /// wanted. Two implementations agreeing with themselves.
    #[test]
    fn isaac_command_blocks_all_become_the_builder_s_commands_term() {
        for isaac in ["velocity_commands", "jump_command", "jump_go"] {
            let text = format!(
                r#"{{"obs_joint_order": ["a"], "action_joint_order": ["a"],
                    "action_scale": 0.25, "default_joint_pos": {{"a": 0.0}},
                    "control": {{"kp": 1.0, "kd": 0.1, "effort_limit": 1.0, "control_hz": 50.0}},
                    "observation": {{"dim": 4, "history_length": 1, "terms": [
                        {{"name": "joint_pos", "dim": 1}}, {{"name": "{isaac}", "dim": 3}}]}},
                    "action": {{"dim": 1}}}}"#
            );
            let c = Contract::from_str(&text, &["a".to_string()]).unwrap();
            let names: Vec<&str> =
                c.layout.observation.terms.iter().map(|t| t.name.as_str()).collect();
            assert_eq!(names, ["joint_pos", "commands"], "{isaac} should map");
        }
        // The control: everything else passes through untouched, so the mapping
        // cannot quietly rename a term the builder already knows.
        let text = r#"{"obs_joint_order": ["a"], "action_joint_order": ["a"],
            "action_scale": 0.25, "default_joint_pos": {"a": 0.0},
            "control": {"kp": 1.0, "kd": 0.1, "effort_limit": 1.0, "control_hz": 50.0},
            "observation": {"dim": 2, "history_length": 1, "terms": [
                {"name": "base_ang_vel", "dim": 3}, {"name": "gait_phase", "dim": 2}]},
            "action": {"dim": 1}}"#;
        let c = Contract::from_str(text, &["a".to_string()]).unwrap();
        let names: Vec<&str> = c.layout.observation.terms.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["base_ang_vel", "gait_phase"]);
    }

    /// An export this repository actually produced, parsed end to end.
    ///
    /// The test above would pass against a mapping that covered the three names
    /// and nothing else about a real file. This is the file.
    #[test]
    fn a_real_exported_bundle_loads() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tasks/jumper/tripod/out/rough/layout.json");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            // Exports are not committed as a rule; skip rather than fail on a
            // clone that has none.
            Err(_) => return,
        };
        let raw: serde_json::Value = serde_json::from_str(&text).unwrap();
        let wire: Vec<String> = raw["wire_joint_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let c = Contract::from_str(&text, &wire).unwrap();
        assert!(c.layout.observation.terms.iter().any(|t| t.name == "commands"));
        assert_eq!(c.default_wire.len(), wire.len());
    }

    fn joints() -> Vec<String> {
        ["a", "b", "c"].iter().map(|s| s.to_string()).collect()
    }

    const MINIMAL: &str = r#"{
        "obs_joint_order": ["a", "b"],
        "action_joint_order": ["a", "b"],
        "action_scale": 0.25,
        "default_joint_pos": {"a": 0.1, "b": 0.2, "c": 0.3},
        "control": {"kp": 10.0, "kd": 0.5, "effort_limit": 2.0, "control_hz": 50.0},
        "observation": {"dim": 4, "history_length": 1,
                        "terms": [{"name": "joint_pos", "dim": 2},
                                  {"name": "joint_vel", "dim": 2}]},
        "action": {"dim": 2}
    }"#;

    #[test]
    fn resolves_joints_by_name_not_position() {
        // 'c' is on the wire but not observed, and it sits *last* in joint_names
        // while being absent from both layout orders. An index pairing would put
        // the observation's slot 2 somewhere; a name pairing has no slot 2.
        let c = Contract::from_str(MINIMAL, &joints()).unwrap();
        assert_eq!(c.obs_wire_idx, vec![0, 1]);
        assert_eq!(c.action_wire_idx, vec![0, 1]);
        assert_eq!(c.default_wire, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn wire_order_may_differ_from_the_layout_order() {
        // The robot lists them backwards. Resolution must follow the names.
        let reversed: Vec<String> = ["c", "b", "a"].iter().map(|s| s.to_string()).collect();
        let c = Contract::from_str(MINIMAL, &reversed).unwrap();
        assert_eq!(c.obs_wire_idx, vec![2, 1], "obs slots must follow names");
        assert_eq!(c.default_wire, vec![0.3, 0.2, 0.1], "home pose is wire-ordered");
    }

    #[test]
    fn unknown_joint_is_fatal() {
        let bad = MINIMAL.replace(r#"["a", "b"],
        "action_joint_order""#, r#"["a", "zz"],
        "action_joint_order""#);
        assert!(matches!(
            Contract::from_str(&bad, &joints()),
            Err(Error::UnknownJoint { .. })
        ));
    }

    #[test]
    fn home_pose_must_cover_every_wire_joint() {
        let bad = MINIMAL.replace(r#", "c": 0.3"#, "");
        match Contract::from_str(&bad, &joints()) {
            Err(Error::HomePoseIncomplete(j)) => assert_eq!(j, vec!["c".to_string()]),
            other => panic!("expected HomePoseIncomplete, got {other:?}"),
        }
    }

    /// A stride is read per term, absent is 1, and zero is refused: frames
    /// zero steps apart are one frame, `history_length` times.
    #[test]
    fn a_term_s_stride_is_read_and_zero_is_refused() {
        let strided = MINIMAL.replace(
            r#"{"name": "joint_vel", "dim": 2}"#,
            r#"{"name": "joint_vel", "dim": 2, "history_length": 5, "history_stride": 4}"#,
        );
        let c = Contract::from_str(&strided, &joints()).unwrap();
        assert_eq!(c.term_stride, vec![1, 4], "absent is 1, present is read");
        let zero = strided.replace(r#""history_stride": 4"#, r#""history_stride": 0"#);
        assert!(matches!(Contract::from_str(&zero, &joints()), Err(Error::Inconsistent(_))));
    }

    /// The controller block is `operator_controller/2` and nothing else. The
    /// retired `twist_controller/1` is refused by name, like any schema this
    /// controller does not implement: read as the nearest thing it would have no
    /// keyboard and no rests.
    #[test]
    fn a_controller_schema_this_controller_does_not_implement_is_refused() {
        let with = |block: &str| MINIMAL.replace(
            r#""action": {"dim": 2}"#,
            &format!(r#""action": {{"dim": 2}}, "controller": {block}"#),
        );
        let op = r#"{"schema": "operator_controller/2",
            "command": [{"term": "posture", "feeds": "posture_command",
                         "axes": [{"name": "pitch"}]}],
            "devices": {"gamepad": {"scheme": "absolute", "deadzone": "device_reported_rescaled",
                                    "release_button": "B",
                                    "axes": {"pitch": {"source": "Ry", "sign": 1}}},
                        "keyboard": {"scheme": "keys", "full_after_s": 2.0,
                                     "axes": {"pitch": {"+": ["key_k"], "-": ["key_i"]}},
                                     "release": ["key_b"]}}}"#;
        // Checked against the ranges it scales to: this one has none for its
        // posture term, so a full stick would mean nothing.
        let e = Contract::from_str(&with(op), &joints()).unwrap_err();
        assert!(format!("{e}").contains("no command_ranges for 'posture'"), "{e}");
        let ranged = with(op).replace(
            r#""controller""#,
            r#""command_ranges": {"posture": {"pitch": [-0.26, 0.26]}}, "controller""#,
        );
        let c = Contract::from_str(&ranged, &joints()).unwrap();
        assert!(c.layout.controller.is_some(), "the control group: the real schema loads");

        let old = r#"{"schema": "twist_controller/1", "devices": {"gamepad": {
            "scheme": "absolute", "deadzone": "device_reported_rescaled",
            "release_button": "B", "axes": {"lin_vel_x": {"source": "Ly", "sign": -1}}}}}"#;
        let e = Contract::from_str(&with(old), &joints()).unwrap_err();
        assert!(format!("{e}").contains("implements only"), "{e}");
    }

    #[test]
    fn per_term_history_falls_back_to_the_group_default() {
        let mixed = MINIMAL
            .replace(r#""history_length": 1"#, r#""history_length": 5"#)
            .replace(r#"{"name": "joint_vel", "dim": 2}"#,
                     r#"{"name": "joint_vel", "dim": 2, "history_length": 1}"#);
        let c = Contract::from_str(&mixed, &joints()).unwrap();
        assert_eq!(c.term_history, vec![5, 1], "group default, then the override");
    }
}
