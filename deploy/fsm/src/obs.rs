//! Assembling the observation vector the policy was trained on.
//!
//! Everything here is layout, and layout failures are silent: a vector of the
//! right length whose terms sit at the wrong offsets produces a policy that acts
//! confidently and wrongly, with nothing to log. Two rules carry most of the
//! weight, and both are easy to get backwards:
//!
//! **Term-major, not frame-major.** All of term 0's frames are contiguous, then
//! all of term 1's. This matches Isaac/mjlab's `flatten_history_dim`.
//!
//! **Oldest frame first**, within each term. `hist[0]` is the oldest and
//! `hist[max_hist-1]` is this tick. A term shallower than the deepest one takes
//! only the last `h` slots.
//!
//! A term's own `history_length` overrides the group default, so the two are not
//! interchangeable and `dim` is **not** `frame_dim * history_length` unless every
//! term happens to agree.

use crate::layout::Contract;
use crate::types::{Command, RobotState};

/// Per-quantity observation scales. These are applied here, on the deployment
/// side, and must equal what training applied. All 1.0 is correct for a policy
/// that bakes its normalisation into the ONNX graph, which is what
/// `scripts/export.py` produces.
#[derive(Debug, Clone)]
pub struct Scales {
    pub base_lin_vel: f32,
    pub base_ang_vel: f32,
    pub dof_pos: f32,
    pub dof_vel: f32,
    pub joint_torque: f32,
    pub commands: f32,
}

impl Default for Scales {
    fn default() -> Self {
        Self {
            base_lin_vel: 1.0,
            base_ang_vel: 1.0,
            dof_pos: 1.0,
            dof_vel: 1.0,
            joint_torque: 1.0,
            commands: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ObsConfig {
    pub terms: Vec<String>,
    pub term_history: Vec<usize>,
    /// Which command channels the `commands` term carries, in order.
    pub command_terms: Vec<String>,
    pub scales: Scales,
    /// Symmetric clamp on the finished vector. 0 disables.
    pub clip: f32,
    pub obs_wire_idx: Vec<usize>,
    pub gait_period: f64,
    pub control_dt: f64,
    /// The gait clock is zeroed while the command asks the robot to stand still.
    ///
    /// **This is not cosmetic and the C++ controller does not do it.** The policy
    /// is trained with `clock * (||command[:3]|| > threshold)`, so standing shows
    /// it `(0, 0)` -- a magnitude no phase can produce, i.e. an unambiguous "hold
    /// still". Feeding a free-running unit-magnitude clock instead was measured
    /// during training to make the robot step in place while commanded to stand:
    /// roll-rate autocorrelation peaked at the clock frequency and only 5.17 of 6
    /// feet were on the ground.
    ///
    /// `None` disables the gate, for a policy trained without one.
    pub gait_gate_threshold: Option<f32>,
    /// The gait clock's rate law, when the rate follows the command. `None` is
    /// the fixed `gait_period`, which is every gait but `jumper.posture`'s.
    pub cadence: Option<Cadence>,
    /// Per-term control steps between stacked frames, one entry per term; a
    /// missing entry is 1. See `Term::history_stride`.
    pub term_stride: Vec<usize>,
    /// `posture_command`'s height channel is the commanded height minus this,
    /// read from the term's own params. The term is refused without it: the
    /// uncentred height is 0.107 m out on that channel for ever, and that is
    /// how the task's own replay check found it.
    pub posture_neutral_height: Option<f32>,
    /// Seconds between entering a reference-guided mode and its `go`.
    ///
    /// **A placeholder for an operator's button.** A reference-guided policy is
    /// keyed to a moment -- the jump's push-off is 40 ms wide -- and on the
    /// robot that moment should be a person pressing something. Until the
    /// binding exists, the motion runs on a fixed delay from mode entry, which
    /// is enough to see the whole motion on the bench and is not enough to use.
    pub reference_go_delay_s: f64,
}

/// A gait clock whose rate follows the command.
///
/// `tasks/jumper/posture/mdp/cadence.py`, transcribed:
///
/// ```text
/// v_eff = |(vx, vy)| + |wz| * turn_radius
/// f     = clamp(v_eff / (2 * stride), freq_min, freq_max)
/// ```
///
/// The phase is the integral of `f`, so it is state: zero on the first frame
/// after entering the mode, then advanced once a tick by `f * dt` -- **and when
/// in the tick is the contract's to say**, as `advance`. Training has had two
/// orders, a policy was trained in one of them, and the two differ by
/// `(f_now - f_at_entry) * dt` once the tempo moves: up to
/// `(freq_max - freq_min) * dt`, 6.4 degrees here. See `Advance`.
///
/// In `f32`, as torch does it, so a replay against mjlab differs by rounding
/// rather than by a drift that grows with the episode.
///
/// A fixed period with this policy's contract would not raise; it would show
/// the policy a clock at one tempo while it had been trained to step at the
/// tempo its speed implies, which reads on a robot as a gait slightly off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cadence {
    pub stride: f32,
    pub freq_min: f32,
    pub freq_max: f32,
    pub turn_radius: f32,
    pub advance: Advance,
}

/// When in a tick the cadence clock moves: `gait_phase`'s `params.advance`.
///
/// Both orders are deployed, so neither is a default a host may choose. The
/// field is what tells them apart and the only thing that can:
/// `tasks/jumper/posture/mdp/cadence.py::ADVANCE` has the training side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Advance {
    /// No `advance` in the contract. The frame shows the phase advanced by
    /// **its own** command, the first after entering the mode excepted -- the
    /// order the actor's clock had while the task mounted one clock per
    /// observation group, and so every policy exported before the field
    /// existed was trained in it. Measured through `play --fsm` at the time:
    /// the reward's order, which is `AfterFrame`'s, matched mjlab's actor only
    /// until the first command resample and was 0.016 of a cycle off after it.
    BeforeFrame,
    /// `"after_frame"`. The frame shows the phase as it stands, and the phase
    /// then advances by that frame's command, so frame `k` shows
    /// `phase(k-1) + f(command shown at k-1) * dt`. Training's one clock since
    /// the task stopped mounting one per group: the rewards and both groups
    /// read that value in the same step. Measured through `play --fsm`'s loop,
    /// one environment, a walk redrawn every 0.5 s: 5e-6 degrees off mjlab's
    /// `gait_phase` over 26 tempo changes, where `BeforeFrame` on the same
    /// export was 6.3 degrees off.
    AfterFrame,
}

impl Cadence {
    /// The law from a `gait_phase` term's params, when they carry one.
    ///
    /// `stride` is what marks it: a fixed clock carries `period` instead, and
    /// a term with neither is left to the caller's default. A term that
    /// carries `stride` and not the other three is refused -- half a law is
    /// not a law with defaults.
    ///
    /// `advance` absent is `BeforeFrame`, which is what an export from before
    /// the field means rather than a guess about it. Any value but
    /// `"after_frame"` is refused: an order this controller does not know is
    /// not one it can run a policy in.
    pub fn from_term(term: &crate::layout::Term) -> Option<Result<Self, Error>> {
        term.params.get("stride")?;
        let need = |key: &str| {
            term.param_f64(key).map(|v| v as f32).ok_or_else(|| Error::MissingParam {
                term: term.name.clone(),
                param: key.to_string(),
            })
        };
        Some((|| {
            let advance = match term.params.get("advance") {
                None => Advance::BeforeFrame,
                Some(v) if v.as_str() == Some("after_frame") => Advance::AfterFrame,
                Some(v) => {
                    return Err(Error::UnknownParamValue {
                        term: term.name.clone(),
                        param: "advance".into(),
                        value: v.to_string(),
                    })
                }
            };
            Ok(Cadence {
                stride: need("stride")?,
                freq_min: need("freq_min")?,
                freq_max: need("freq_max")?,
                turn_radius: need("turn_radius")?,
                advance,
            })
        })())
    }

    /// This tick's cadence, in Hz.
    pub fn frequency(&self, c: &Command) -> f32 {
        let v = (c.lin_vel_x * c.lin_vel_x + c.lin_vel_y * c.lin_vel_y).sqrt()
            + c.yaw_rate.abs() * self.turn_radius;
        (v / (2.0 * self.stride)).clamp(self.freq_min, self.freq_max)
    }

    /// `phase` one control step on at `c`'s tempo, wrapped into `[0, 1)`.
    fn advanced(&self, phase: f32, c: &Command, dt: f32) -> f32 {
        (phase + self.frequency(c) * dt).rem_euclid(1.0)
    }
}

#[derive(Debug)]
pub enum Error {
    /// A term this controller cannot build. Refused rather than approximated:
    /// there is no value to emit that is closer to right than failing.
    UnsupportedTerm(String),
    /// A term this controller builds, missing a parameter its contract has to
    /// carry for the numbers to mean what they meant in training.
    MissingParam { term: String, param: String },
    /// A parameter whose value names behaviour this controller does not have.
    UnknownParamValue { term: String, param: String, value: String },
    UnknownCommandChannel(String),
    /// The terms sum to a width the layout does not claim.
    DimMismatch { built: usize, layout: usize },
    /// A term that can only be built from a recording, with no recording
    /// attached to the contract.
    MissingReference(String),
    /// A term whose recording is attached but carries no table for it.
    MissingTable(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::UnsupportedTerm(t) => write!(
                f,
                "observation term '{t}' is not implemented. Supported: {}. \
                 mjlab's own tracking names are not among them and will not be: \
                 `motion_anchor_pos_b` is a base position, which needs a state \
                 estimator this robot does not have. Its orientation half is built, \
                 as `ref_tilt_error`, with the yaw removed -- see \
                 `tasks/jumper/common/dance/observations.py` for why.",
                SUPPORTED.join(", ")
            ),
            Error::MissingParam { term, param } => write!(
                f,
                "observation term '{term}' needs `params.{param}` from its contract, and it \
                 is not there. Re-export the policy: an export that predates the field \
                 would be built from a guess."
            ),
            Error::UnknownParamValue { term, param, value } => write!(
                f,
                "observation term '{term}' has `params.{param}` = {value}, which this \
                 controller does not implement. Refused rather than approximated: the \
                 policy was trained on exactly that, and nothing here is closer to it."
            ),
            Error::UnknownCommandChannel(c) => {
                write!(f, "command channel '{c}' is not one this controller produces")
            }
            Error::MissingReference(t) => write!(
                f,
                "observation term '{t}' is built from the recorded motion, and no \
                 reference table is attached to this contract. Either the bundle is \
                 missing the companion file the `reference` block names, or the host \
                 that opened it did not attach one."
            ),
            Error::MissingTable(t) => write!(
                f,
                "observation term '{t}' reads a table this bundle's reference does not \
                 carry. The export writes it only for a policy that observes it, so the \
                 contract and the trajectory came from different exports."
            ),
            Error::DimMismatch { built, layout } => write!(
                f,
                "observation terms build {built} values but the layout says {layout}; \
                 the model would be fed a vector of the wrong width"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// The observation terms this controller can produce, for anything that needs
/// to enumerate them rather than merely check one -- `source.rs` builds a
/// simulation's capability from this so the two cannot drift apart.
pub fn supported_terms() -> &'static [&'static str] {
    SUPPORTED
}

const SUPPORTED: &[&str] = &[
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "base_pose",
    "commands",
    "commands_diff",
    // `jumper.posture`'s second command: twist, pitch, roll, height.
    "posture_command",
    "joint_pos",
    "joint_vel",
    "joint_torque",
    "actions",
    "gait_phase",
    // Reference-guided policies. Both are built from the contract's `reference`
    // block and its companion table, and both are refused when it is absent:
    // no approximation of a recorded motion is worth emitting in its place.
    // (No quoted phrases in this comment -- tests/test_layout.py reads the
    // quoted strings in this array as the list itself.)
    "jump_phase",
    "ref_future",
    "clip_phase",
    "ref_joint_pos",
    "ref_joint_vel",
    "ref_tilt_error",
];

fn term_width(
    term: &str,
    obs_joints: usize,
    n_cmd: usize,
    action_dim: usize,
    lookahead: usize,
) -> Option<usize> {
    Some(match term {
        "base_lin_vel" | "base_ang_vel" | "projected_gravity" | "base_pose" => 3,
        "commands" | "commands_diff" => n_cmd,
        "posture_command" => 4,
        "joint_pos" | "joint_vel" | "joint_torque" => obs_joints,
        "actions" => action_dim,
        "gait_phase" => 2,
        "jump_phase" => 1,
        "clip_phase" => 2,
        "ref_joint_pos" | "ref_joint_vel" => obs_joints,
        "ref_tilt_error" => 3,
        // One joint vector per preview offset. The offsets are the contract's,
        // never this side's: a preview read at the wrong horizons is a policy
        // shown a motion it is not about to make.
        "ref_future" => lookahead * obs_joints,
        _ => return None,
    })
}

/// Whether a term can only be built from a reference recording.
fn needs_reference(term: &str) -> bool {
    matches!(
        term,
        "jump_phase" | "ref_future" | "clip_phase" | "ref_joint_pos" | "ref_joint_vel"
            | "ref_tilt_error"
    )
}

/// `projected_gravity` from a body quaternion `(w, x, y, z)`: the world's down
/// axis expressed in the body frame.
pub fn projected_gravity(q: [f32; 4]) -> [f32; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    ]
}

pub struct ObservationBuilder {
    cfg: ObsConfig,
    default_wire: Vec<f32>,
    action_dim: usize,
    n_cmd: usize,

    term_off: Vec<usize>,
    term_w: Vec<usize>,
    term_hist: Vec<usize>,
    term_stride: Vec<usize>,
    frame_dim: usize,
    dim: usize,
    /// Frames kept: the widest any term reaches back, `stride * (frames - 1)
    /// + 1`. The newest is the last slot.
    span: usize,


    hist: Vec<Vec<f32>>,
    prev_cmd: Vec<f32>,
    frame: Vec<f32>,
    out: Vec<f32>,
    primed: bool,
    cmd_primed: bool,
    gait_step: i64,
    /// The integrated phase, `[0, 1)`, when the clock follows the command.
    cadence_phase: f32,

    /// The recording, when this mode runs a reference-guided policy.
    reference: Option<std::sync::Arc<crate::trajectory::Trajectory>>,
    /// Ticks since the motion's `go`, or `None` while it waits for one.
    ///
    /// `reset` returns it to the waiting state along with `gait_step`, and that
    /// matters more here: a gait clock resumed mid-cycle is a phase error,
    /// while a motion clock resumed mid-recording starts the second jump in the
    /// middle of the first one's flight.
    ref_step: Option<i64>,
    /// Whether a button reaches this motion. When it does the clock starts on
    /// the event and the timer is not consulted; when it does not, the motion
    /// runs on the timer from mode entry, which is a bench aid.
    go_by_event: bool,
    /// Whether the recording begins with the mode. `jumper.dance` does -- it is a
    /// piece of music, and the thing that starts it is switching into it.
    on_entry: bool,
    /// Scratch for one sampled row, and for the residual baseline the action
    /// decoder reads back. Kept here so a tick allocates nothing.
    ref_row: Vec<f32>,
    baseline: Vec<f32>,
}

impl ObservationBuilder {
    pub fn new(cfg: ObsConfig, contract: &Contract, joint_count: usize) -> Result<Self, Error> {
        let obs_joints = cfg.obs_wire_idx.len();
        let n_cmd = cfg.command_terms.len();
        let action_dim = contract.action_dim();

        // A reference term with no recording attached cannot be built, and
        // there is nothing sensible to build instead -- zeros would be a
        // motionless reference, which is a pose the policy never saw and would
        // track perfectly. So this fails here, before the robot is enabled.
        let reference = contract.reference.clone();
        if reference.is_none() {
            if let Some(t) = cfg.terms.iter().find(|t| needs_reference(t)) {
                return Err(Error::MissingReference(t.clone()));
            }
        }
        let lookahead = reference.as_ref().map_or(0, |r| r.lookahead().len());
        let go_by_event = reference.as_ref().is_some_and(|r| r.go_event().is_some());
        let on_entry = reference.as_ref().is_some_and(|r| r.starts_on_entry());

        // A term needs the table it reads, and a table defaulting to empty
        // would be a motionless reference -- which a standing robot tracks
        // perfectly and no log separates from success.
        if let Some(r) = reference.as_ref() {
            for (term, present) in
                [("ref_joint_vel", r.has_qd()), ("ref_tilt_error", r.has_root_quat())]
            {
                if cfg.terms.iter().any(|t| t == term) && !present {
                    return Err(Error::MissingTable(term.to_string()));
                }
            }
        }

        if cfg.terms.iter().any(|t| t == "posture_command") && cfg.posture_neutral_height.is_none() {
            return Err(Error::MissingParam {
                term: "posture_command".into(),
                param: "neutral_height".into(),
            });
        }

        let (mut term_off, mut term_w, mut term_hist) = (vec![], vec![], vec![]);
        let mut term_stride = vec![];
        let (mut frame_dim, mut dim, mut span) = (0usize, 0usize, 1usize);
        for (k, t) in cfg.terms.iter().enumerate() {
            let w = term_width(t, obs_joints, n_cmd, action_dim, lookahead)
                .ok_or_else(|| Error::UnsupportedTerm(t.clone()))?;
            let h = cfg.term_history.get(k).copied().unwrap_or(1).max(1);
            let stride = cfg.term_stride.get(k).copied().unwrap_or(1).max(1);
            term_off.push(frame_dim);
            term_w.push(w);
            term_hist.push(h);
            term_stride.push(stride);
            frame_dim += w;
            dim += w * h;
            span = span.max(stride * (h - 1) + 1);
        }

        if dim != contract.layout.observation.dim {
            return Err(Error::DimMismatch {
                built: dim,
                layout: contract.layout.observation.dim,
            });
        }

        Ok(Self {
            default_wire: contract.default_wire.clone(),
            action_dim,
            n_cmd,
            term_off,
            term_w,
            term_hist,
            term_stride,
            frame_dim,
            dim,
            span,
            hist: vec![Vec::new(); span],
            prev_cmd: vec![0.0; n_cmd],
            frame: Vec::with_capacity(frame_dim),
            out: Vec::with_capacity(dim),
            primed: false,
            cmd_primed: false,
            gait_step: 0,
            cadence_phase: 0.0,
            ref_row: vec![0.0; joint_count],
            baseline: vec![0.0; joint_count],
            ref_step: if go_by_event { None } else { Some(0) },
            go_by_event,
            on_entry,
            reference,
            cfg,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Seconds since this motion's `go`, negative while it waits.
    ///
    /// Negative is a real state, not a guard: the recording carries a
    /// preparatory stand and `Trajectory::index_of` clamps into it, so a
    /// controller waiting here previews the stand rather than the approach.
    /// Any negative reads the same frame, so the waiting value is a constant
    /// rather than a countdown -- a countdown would walk the preview forward
    /// through the approach while nothing had been asked for.
    fn t_since_go(&self) -> f64 {
        const WAITING: f64 = -1.0;
        match self.ref_step {
            None => WAITING,
            // The timer offset applies only where no button reaches the
            // motion. With one, the clock starts *at* the press and there is
            // nothing to count down.
            Some(n) => {
                let t = n as f64 * self.cfg.control_dt;
                // The timer is the bench aid for a motion that wants a button
                // and has none. A recording that starts with the mode wants
                // neither, and applying it would hold the first bars back.
                if self.go_by_event || self.on_entry {
                    t
                } else {
                    t - self.cfg.reference_go_delay_s
                }
            }
        }
    }

    /// Start the motion, now.
    ///
    /// Re-triggering restarts it: pressing the button again while the robot is
    /// still in the air is a second jump asked for, and running the recording
    /// from the top is what the training env does on its own `go`.
    pub fn trigger_go(&mut self) {
        self.ref_step = Some(0);
    }

    /// Whether the recording this mode plays has run out.
    ///
    /// False for a policy with no recording, which is every gait: a gait does
    /// not end. False as well while a motion is still waiting for its `go` --
    /// it has not started, so it cannot have finished, and a controller that
    /// confused the two would hand back before the operator ever pressed
    /// anything.
    pub fn motion_finished(&self) -> bool {
        let Some(r) = self.reference.as_ref() else { return false };
        self.ref_step.is_some() && r.finished(self.t_since_go())
    }

    /// Whether this mode waits on a button rather than on the timer. Read by
    /// the host so it can say which, rather than leaving it to be inferred from
    /// a robot that jumps two seconds after it stands up.
    pub fn go_is_bound(&self) -> bool {
        self.go_by_event
    }

    /// The residual baseline for the tick just built, in wire order.
    ///
    /// `None` for a policy that is not reference-guided, which is every
    /// velocity policy. Read by the action decoder **after** `build`, so the
    /// observation and the baseline come from one clock reading -- computing it
    /// again at decode time would sample a tick later and put the residual on a
    /// different pose than the one the policy answered.
    pub fn reference_baseline(&self) -> Option<&[f32]> {
        let r = self.reference.as_ref()?;
        r.spec().residual_action.then_some(&self.baseline[..])
    }

    /// Forget the history. Called on every entry into policy mode, so the first
    /// observation is not stitched onto frames from minutes ago.
    pub fn reset(&mut self) {
        self.primed = false;
        self.cmd_primed = false;
        self.gait_step = 0;
        self.cadence_phase = 0.0;
        self.ref_step = if self.go_by_event { None } else { Some(0) };
        self.prev_cmd.iter_mut().for_each(|v| *v = 0.0);
    }

    fn command_channel(&self, name: &str, c: &Command) -> Result<f32, Error> {
        Ok(match name {
            "lin_vel_x" => c.lin_vel_x,
            "lin_vel_y" => c.lin_vel_y,
            "lin_vel_z" => 0.0,
            "yaw_rate" | "ang_vel_yaw" => c.yaw_rate,
            "height" => c.height,
            _ => return Err(Error::UnknownCommandChannel(name.to_string())),
        })
    }

    /// The gate mjrl's `phase_clock` applies: the L2 norm of the first three
    /// command channels against a threshold.
    fn moving(&self, c: &Command) -> f32 {
        match self.cfg.gait_gate_threshold {
            None => 1.0,
            Some(th) => {
                let n = (c.lin_vel_x * c.lin_vel_x
                    + c.lin_vel_y * c.lin_vel_y
                    + c.yaw_rate * c.yaw_rate)
                    .sqrt();
                if n > th {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    fn build_frame(
        &mut self,
        state: &RobotState,
        command: &Command,
        last_action: &[f32],
        last_torque: &[f32],
    ) -> Result<(), Error> {
        let s = self.cfg.scales.clone();
        let grav = projected_gravity(state.imu.quat);
        let tsg = self.t_since_go();
        self.frame.clear();

        for k in 0..self.cfg.terms.len() {
            match self.cfg.terms[k].as_str() {
                "base_lin_vel" => {
                    for i in 0..3 {
                        let v = if state.imu.has_lin_vel { state.imu.lin_vel[i] } else { 0.0 };
                        self.frame.push(v * s.base_lin_vel);
                    }
                }
                "base_ang_vel" => {
                    for i in 0..3 {
                        self.frame.push(state.imu.gyro[i] * s.base_ang_vel);
                    }
                }
                "projected_gravity" => self.frame.extend_from_slice(&grav),
                "base_pose" => {
                    self.frame.push(command.base_pitch);
                    self.frame.push(command.base_roll);
                    self.frame.push(command.base_twist);
                }
                "commands" => {
                    for i in 0..self.n_cmd {
                        let name = self.cfg.command_terms[i].clone();
                        let v = self.command_channel(&name, command)?;
                        self.frame.push(v * s.commands);
                    }
                }
                "posture_command" => {
                    // [twist, pitch, roll, height], the command term's own
                    // order -- not `base_pose`'s [pitch, roll, twist], which
                    // carries three of the same quantities in another layout.
                    // The height is centred on the standing height, as the
                    // observation was in training.
                    let neutral = self.cfg.posture_neutral_height.expect("checked in new()");
                    for v in [
                        command.base_twist,
                        command.base_pitch,
                        command.base_roll,
                        command.height - neutral,
                    ] {
                        self.frame.push(v * s.commands);
                    }
                }
                "commands_diff" => {
                    for i in 0..self.n_cmd {
                        let name = self.cfg.command_terms[i].clone();
                        let cur = self.command_channel(&name, command)? * s.commands;
                        self.frame
                            .push(if self.cmd_primed { cur - self.prev_cmd[i] } else { 0.0 });
                    }
                }
                // **Every observed joint reports its measurement**, whether the
                // policy drives it or not, because training does: mjlab's joint
                // terms know nothing of the action. `jumper.five_foot` observes
                // its carried arm and claw *because* they are measured -- held by
                // a reset event, closed by the operator, and randomised across
                // their travel in training. `rl-wbc-fsm` zeroed an observed joint
                // the action did not drive, which suits a policy trained on a zero
                // there and no policy this repository exports: it showed
                // five_foot a claw frozen at its nominal pose, open, still and
                // weightless -- measured under `play --fsm --fsm-diff` at up to
                // 0.087 rad and 0.2 rad/s off the simulator on the arm alone. A
                // task that wants a joint unseen leaves it out of the observation
                // in training, and then it is not in `obs_joint_order` at all.
                "joint_pos" => {
                    for si in 0..self.cfg.obs_wire_idx.len() {
                        let w = self.cfg.obs_wire_idx[si];
                        self.frame.push((state.q[w] - self.default_wire[w]) * s.dof_pos);
                    }
                }
                "joint_vel" => {
                    for si in 0..self.cfg.obs_wire_idx.len() {
                        let w = self.cfg.obs_wire_idx[si];
                        self.frame.push(state.qd[w] * s.dof_vel);
                    }
                }
                "joint_torque" => {
                    for si in 0..self.cfg.obs_wire_idx.len() {
                        let w = self.cfg.obs_wire_idx[si];
                        let t = last_torque.get(w).copied().unwrap_or(0.0);
                        self.frame.push(t * s.joint_torque);
                    }
                }
                "actions" => {
                    for i in 0..self.action_dim {
                        self.frame.push(last_action.get(i).copied().unwrap_or(0.0));
                    }
                }
                "gait_phase" => {
                    let (sin, cos) = if self.cfg.cadence.is_some() {
                        let a = self.cadence_phase * std::f32::consts::TAU;
                        (a.sin(), a.cos())
                    } else {
                        let mut phi = 0.0f64;
                        if self.cfg.gait_period > 0.0 && self.cfg.control_dt > 0.0 {
                            let t = self.gait_step as f64 * self.cfg.control_dt;
                            phi = (t % self.cfg.gait_period) / self.cfg.gait_period;
                        }
                        let a = phi * std::f64::consts::TAU;
                        (a.sin() as f32, a.cos() as f32)
                    };
                    let gate = self.moving(command);
                    self.frame.push(sin * gate);
                    self.frame.push(cos * gate);
                }
                "jump_phase" => {
                    let r = self.reference.as_ref().expect("checked in new()");
                    self.frame.push(r.phase(tsg));
                }
                "ref_future" => {
                    // Relative to the home pose, matching how the robot's own
                    // `joint_pos` reaches the policy. Mixing the two frames --
                    // reference absolute against measurement relative -- leaves
                    // the policy to learn the constant offset between them,
                    // which it can, and which trains and deploys and is wrong
                    // by exactly that offset if the home pose ever changes.
                    let r = self.reference.as_ref().expect("checked in new()").clone();
                    for k in 0..r.lookahead().len() {
                        r.q_at_into(tsg + r.lookahead()[k], &mut self.ref_row);
                        for si in 0..self.cfg.obs_wire_idx.len() {
                            let w = self.cfg.obs_wire_idx[si];
                            self.frame.push(self.ref_row[w] - self.default_wire[w]);
                        }
                    }
                }
                "clip_phase" => {
                    let r = self.reference.as_ref().expect("checked in new()");
                    self.frame.extend_from_slice(&r.clip_phase(tsg));
                }
                "ref_joint_pos" => {
                    // Home-relative, matching how the robot's own `joint_pos`
                    // reaches the policy. Mixing an absolute reference with a
                    // relative measurement leaves the policy to learn the
                    // offset between them, which it can, and which is wrong by
                    // exactly that offset the day the home pose changes.
                    let r = self.reference.as_ref().expect("checked in new()").clone();
                    r.q_at_into(tsg, &mut self.ref_row);
                    for si in 0..self.cfg.obs_wire_idx.len() {
                        let w = self.cfg.obs_wire_idx[si];
                        self.frame.push(self.ref_row[w] - self.default_wire[w]);
                    }
                }
                "ref_joint_vel" => {
                    // Absolute, unlike the position above: a velocity has no
                    // home to be relative to.
                    let r = self.reference.as_ref().expect("checked in new()").clone();
                    r.qd_at_into(tsg, &mut self.ref_row);
                    for si in 0..self.cfg.obs_wire_idx.len() {
                        self.frame.push(self.ref_row[self.cfg.obs_wire_idx[si]]);
                    }
                }
                "ref_tilt_error" => {
                    // The recording's gravity direction minus the robot's.
                    // Gravity is the only absolute orientation reference this
                    // machine has -- the IMU is six-axis -- and expressing the
                    // error this way is what makes it blind to yaw, which is
                    // the half that would otherwise be drift. See
                    // `tasks/jumper/common/dance/observations.py`.
                    let r = self.reference.as_ref().expect("checked in new()");
                    let q = r.root_quat_at(tsg).expect("checked in new()");
                    let g_ref = projected_gravity(q);
                    for i in 0..3 {
                        self.frame.push(g_ref[i] - grav[i]);
                    }
                }
                other => return Err(Error::UnsupportedTerm(other.to_string())),
            }
        }
        debug_assert_eq!(self.frame.len(), self.frame_dim);
        Ok(())
    }

    /// One tick. Returns the flattened observation, valid until the next call.
    pub fn build(
        &mut self,
        state: &RobotState,
        command: &Command,
        last_action: &[f32],
        last_torque: &[f32],
    ) -> Result<&[f32], Error> {
        let dt = self.cfg.control_dt as f32;
        if let Some(k) = self.cfg.cadence.filter(|k| k.advance == Advance::BeforeFrame) {
            // Before the frame, by this tick's command -- but not on the first
            // frame after entering the mode, which is phase 0. See `Advance`.
            if self.primed {
                self.cadence_phase = k.advanced(self.cadence_phase, command, dt);
            }
        }
        self.build_frame(state, command, last_action, last_torque)?;
        if let Some(k) = self.cfg.cadence.filter(|k| k.advance == Advance::AfterFrame) {
            // After the frame, by the command this frame showed: the next one is
            // a step of this one's tempo on. The first frame is 0 by the reset.
            self.cadence_phase = k.advanced(self.cadence_phase, command, dt);
        }
        // Only a residual action has a baseline, and only a residual contract
        // is guaranteed a `q_cmd` to read one from -- `reference_baseline`
        // already tells the decoder exactly that. A mode that merely *observes*
        // a recording drives absolute targets and ships no `q_cmd` at all, so
        // sampling one here read an empty table.
        if let Some(r) = self.reference.clone() {
            if r.spec().residual_action {
                r.baseline_into(self.t_since_go(), &mut self.baseline);
            }
        }
        self.gait_step += 1;
        if let Some(n) = self.ref_step {
            self.ref_step = Some(n + 1);
        }

        for i in 0..self.n_cmd {
            let name = self.cfg.command_terms[i].clone();
            self.prev_cmd[i] = self.command_channel(&name, command)? * self.cfg.scales.commands;
        }
        self.cmd_primed = true;

        if !self.primed {
            // First tick: pad every slot with this frame rather than with zeros.
            // Zeros would be a step the policy never saw in training.
            for h in self.hist.iter_mut() {
                h.clone_from(&self.frame);
            }
            self.primed = true;
        } else {
            for h in 0..self.span - 1 {
                self.hist.swap(h, h + 1);
            }
            let last = self.span - 1;
            self.hist[last].clone_from(&self.frame);
        }

        // Each term takes its frames `stride` slots apart, oldest first and
        // this tick last -- `CircularBuffer.buffer[:, ::stride]` in training.
        self.out.clear();
        for k in 0..self.term_off.len() {
            let (h, stride) = (self.term_hist[k], self.term_stride[k]);
            for j in 0..h {
                let slot = self.span - 1 - stride * (h - 1 - j);
                let off = self.term_off[k];
                self.out.extend_from_slice(&self.hist[slot][off..off + self.term_w[k]]);
            }
        }
        if self.cfg.clip > 0.0 {
            let c = self.cfg.clip;
            self.out.iter_mut().for_each(|v| *v = v.clamp(-c, c));
        }
        Ok(&self.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Contract;

    fn contract(obs_dim: usize, terms_json: &str, hist: usize) -> Contract {
        let text = format!(
            r#"{{"obs_joint_order": ["a"], "action_joint_order": ["a"], "action_scale": 0.25,
                "default_joint_pos": {{"a": 0.0}},
                "control": {{"kp": 10.0, "kd": 0.5, "effort_limit": 2.0, "control_hz": 50.0}},
                "observation": {{"dim": {obs_dim}, "history_length": {hist}, "terms": {terms_json}}},
                "action": {{"dim": 1}}}}"#
        );
        Contract::from_str(&text, &["a".to_string()]).unwrap()
    }

    fn cfg(terms: &[&str], hist: &[usize]) -> ObsConfig {
        ObsConfig {
            reference_go_delay_s: crate::trajectory::DEFAULT_GO_DELAY_S,
            terms: terms.iter().map(|s| s.to_string()).collect(),
            term_history: hist.to_vec(),
            command_terms: vec!["lin_vel_x".into(), "lin_vel_y".into(), "yaw_rate".into()],
            scales: Scales::default(),
            clip: 100.0,
            obs_wire_idx: vec![0],
            gait_period: 0.32,
            control_dt: 0.02,
            gait_gate_threshold: Some(0.05),
            cadence: None,
            term_stride: vec![],
            posture_neutral_height: None,
        }
    }

    fn state() -> RobotState {
        let mut s = RobotState::new(1);
        s.imu.quat = [1.0, 0.0, 0.0, 0.0];
        s
    }

    /// The C++ controller's own test vector (`tests/test_observation.cpp`):
    /// terms {base_ang_vel, base_pose, actions} at depth 2 give dim 14, laid out
    /// [ang_vel x2][base_pose x2][actions x2], with index 0 the OLDEST ang_vel.x
    /// and index 3 the newest. Reproduced here so the two implementations are
    /// pinned to the same layout rather than to each other's prose.
    #[test]
    fn matches_the_cpp_layout_test_vector() {
        let c = contract(
            14,
            r#"[{"name":"base_ang_vel","dim":6},{"name":"base_pose","dim":6},{"name":"actions","dim":2}]"#,
            2,
        );
        let mut b =
            ObservationBuilder::new(cfg(&["base_ang_vel", "base_pose", "actions"], &[2, 2, 2]), &c, 1)
                .unwrap();
        assert_eq!(b.dim(), 14);

        let (mut s, mut cmd) = (state(), Command::default());
        s.imu.gyro = [1.0, 0.0, 0.0];
        cmd.base_pitch = 10.0;
        b.build(&s, &cmd, &[0.5], &[0.0]).unwrap();

        s.imu.gyro = [2.0, 0.0, 0.0];
        cmd.base_pitch = 20.0;
        let out = b.build(&s, &cmd, &[0.7], &[0.0]).unwrap();

        assert_eq!(out.len(), 14);
        assert_eq!(out[0], 1.0, "index 0 must be the OLDEST base_ang_vel.x");
        assert_eq!(out[3], 2.0, "index 3 must be the newest base_ang_vel.x");
        assert_eq!(out[6], 10.0, "base_pose block starts at 6, oldest first");
        assert_eq!(out[9], 20.0);
        assert_eq!(out[12], 0.5, "actions block starts at 12, oldest first");
        assert_eq!(out[13], 0.7);
    }

    #[test]
    fn first_tick_pads_history_with_itself_not_with_zeros() {
        let c = contract(6, r#"[{"name":"base_ang_vel","dim":6}]"#, 2);
        let mut b = ObservationBuilder::new(cfg(&["base_ang_vel"], &[2]), &c, 1).unwrap();
        let mut s = state();
        s.imu.gyro = [3.0, 4.0, 5.0];
        let out = b.build(&s, &Command::default(), &[], &[]).unwrap();
        assert_eq!(out, &[3.0, 4.0, 5.0, 3.0, 4.0, 5.0], "both frames are this one");
    }

    /// The gate the C++ controller lacks. Zero magnitude is a value no phase can
    /// produce, which is what makes "standing" unambiguous to the policy.
    #[test]
    fn gait_clock_is_zeroed_while_the_command_says_stand() {
        let c = contract(2, r#"[{"name":"gait_phase","dim":2}]"#, 1);
        let mut b = ObservationBuilder::new(cfg(&["gait_phase"], &[1]), &c, 1).unwrap();
        let s = state();

        let still = Command { lin_vel_x: 0.01, ..Default::default() }; // norm 0.01 < 0.05
        assert_eq!(b.build(&s, &still, &[], &[]).unwrap(), &[0.0, 0.0]);

        // Control group: the same tick with a moving command must NOT be zero,
        // or this test would pass against a builder that always emits zeros.
        b.reset();
        let moving = Command { lin_vel_x: 0.5, ..Default::default() };
        let out = b.build(&s, &moving, &[], &[]).unwrap().to_vec();
        assert_eq!(out, vec![0.0, 1.0], "step 0 is phase 0 -> (sin, cos) = (0, 1)");
    }

    #[test]
    fn gait_clock_advances_one_control_step_per_tick() {
        let c = contract(2, r#"[{"name":"gait_phase","dim":2}]"#, 1);
        let mut b = ObservationBuilder::new(cfg(&["gait_phase"], &[1]), &c, 1).unwrap();
        let (s, moving) = (state(), Command { lin_vel_x: 0.5, ..Default::default() });
        // period 0.32 s at dt 0.02 s = 16 steps per cycle; step 4 is a quarter turn.
        for _ in 0..4 {
            b.build(&s, &moving, &[], &[]).unwrap();
        }
        let out = b.build(&s, &moving, &[], &[]).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6, "sin(2pi/4) = 1, got {}", out[0]);
        assert!(out[1].abs() < 1e-6, "cos(2pi/4) = 0, got {}", out[1]);
    }

    /// An observed joint the policy does not drive reports its measurement, as
    /// training observed it: `jumper.five_foot`'s carried arm and claw, which a
    /// reset event holds and the operator closes. They used to read zero here,
    /// `rl-wbc-fsm`'s rule, and the policy was shown a claw that never moved. The
    /// driven joint `a` is the control group: both columns are the same
    /// arithmetic, so a mask would show as the two disagreeing.
    #[test]
    fn a_joint_the_policy_does_not_drive_is_observed_as_measured() {
        let text = r#"{"obs_joint_order": ["a","b"], "action_joint_order": ["a"],
            "action_scale": 0.25, "default_joint_pos": {"a": 0.0, "b": 0.5},
            "control": {"kp": 10.0, "kd": 0.5, "effort_limit": 2.0, "control_hz": 50.0},
            "observation": {"dim": 6, "history_length": 1,
                            "terms": [{"name": "joint_pos", "dim": 2},
                                      {"name": "joint_vel", "dim": 2},
                                      {"name": "joint_torque", "dim": 2}]},
            "action": {"dim": 1}}"#;
        let c = Contract::from_str(text, &["a".to_string(), "b".to_string()]).unwrap();
        let mut cf = cfg(&["joint_pos", "joint_vel", "joint_torque"], &[1, 1, 1]);
        cf.obs_wire_idx = vec![0, 1];
        let mut b = ObservationBuilder::new(cf, &c, 2).unwrap();
        let mut s = RobotState::new(2);
        s.imu.quat = [1.0, 0.0, 0.0, 0.0];
        s.q = vec![0.3, 0.9];
        s.qd = vec![-1.0, 0.25];
        let out = b.build(&s, &Command::default(), &[], &[0.2, -0.4]).unwrap();
        let sc = Scales::default();
        let want = [
            0.3 * sc.dof_pos, (0.9 - 0.5) * sc.dof_pos,
            -1.0 * sc.dof_vel, 0.25 * sc.dof_vel,
            0.2 * sc.joint_torque, -0.4 * sc.joint_torque,
        ];
        for (i, (got, want)) in out.iter().zip(want).enumerate() {
            assert!((got - want).abs() < 1e-6, "column {i}: {got}, not the measured {want}");
        }
    }

    #[test]
    fn a_term_the_controller_cannot_build_is_refused() {
        // mjlab's `motion_anchor_pos_b`: a base position, which this robot
        // has no state estimator to measure. Not a term waiting to be written
        // -- the orientation half of the same pair *is* built, as
        // `ref_tilt_error`, and this one cannot be.
        let c = contract(3, r#"[{"name":"motion_anchor_pos_b","dim":3}]"#, 1);
        assert!(matches!(
            ObservationBuilder::new(cfg(&["motion_anchor_pos_b"], &[1]), &c, 1),
            Err(Error::UnsupportedTerm(_))
        ));
    }

    #[test]
    fn a_reference_term_without_its_recording_is_refused_at_construction() {
        // The failure this pins: zeros are a *motionless* reference, which the
        // policy would track perfectly while the robot stood still. There is no
        // value to emit here that is closer to right than failing, and the time
        // to fail is before the robot is enabled.
        for term in ["jump_phase", "ref_future"] {
            let c = contract(1, &format!(r#"[{{"name":"{term}","dim":1}}]"#), 1);
            let e = ObservationBuilder::new(cfg(&[term], &[1]), &c, 1).err();
            assert!(
                matches!(&e, Some(Error::MissingReference(t)) if t == term),
                "{term}: {e:?}"
            );
        }
    }

    /// The reference path end to end: the contract block, the companion table,
    /// the two terms and the residual baseline.
    fn with_reference(term: &str, dim: usize, lookahead: &str, offsets: usize) -> Contract {
        let text = format!(
            r#"{{"obs_joint_order": ["a"], "action_joint_order": ["a"], "action_scale": 0.25,
                "default_joint_pos": {{"a": 0.5}},
                "control": {{"kp": 10.0, "kd": 0.5, "effort_limit": 2.0, "control_hz": 50.0}},
                "observation": {{"dim": {dim}, "history_length": 1,
                                 "terms": [{{"name": "{term}", "dim": {dim}}}]}},
                "action": {{"dim": 1}},
                "reference": {{"file": "t.json", "residual_action": true, "rec_hz": 100.0,
                               "control_hz": 50.0, "t_go": 0.1, "go_frame": 10,
                               "duration": 0.3, "span": 0.2, "lookahead_s": {lookahead},
                               "baseline": "q_cmd", "baseline_lead_s": 0.02,
                               "n_frames": 30, "n_cmd": 15}}}}"#
        );
        let _ = offsets;
        let mut c = Contract::from_str(&text, &["a".to_string()]).unwrap();
        c.attach_reference(&table_for(&["a".to_string()]), &["a".to_string()]).unwrap();
        c
    }

    /// q[f] = f, q_cmd[f] = 100 + f: a row reads back as its own index, so an
    /// off-by-one or the wrong table is a wrong number rather than a wrong shape.
    fn table_for(wire: &[String]) -> String {
        let q: Vec<Vec<f32>> = (0..30).map(|f| vec![f as f32]).collect();
        let qc: Vec<Vec<f32>> = (0..15).map(|f| vec![100.0 + f as f32]).collect();
        serde_json::json!({"joint_order": wire, "q": q, "q_cmd": qc}).to_string()
    }

    #[test]
    fn the_motion_clock_starts_at_mode_entry_and_waits_out_the_go_delay() {
        // phase stays 0 through the delay, then runs. A clock that started the
        // motion at mode entry would have the robot jumping as it stands up.
        let c = with_reference("jump_phase", 1, "[0.01]", 1);
        let mut cf = cfg(&["jump_phase"], &[1]);
        cf.reference_go_delay_s = 0.10; // five ticks at 50 Hz
        let mut b = ObservationBuilder::new(cf, &c, 1).unwrap();
        let s = RobotState::new(1);
        let mut seen = vec![];
        for _ in 0..12 {
            seen.push(b.build(&s, &Command::default(), &[], &[]).unwrap()[0]);
        }
        assert_eq!(&seen[..5], &[0.0; 5], "waiting for go: {seen:?}");
        assert!(seen[6] > 0.0 && seen[11] > seen[6], "then it runs: {seen:?}");

        // And it restarts, or the second jump begins mid-flight of the first.
        b.reset();
        assert_eq!(b.build(&s, &Command::default(), &[], &[]).unwrap()[0], 0.0);
    }

    #[test]
    fn a_motion_bound_to_a_button_waits_for_it_rather_than_for_a_timer() {
        // The failure this pins: with the timer still applied under an event
        // binding, the robot would jump two seconds after standing up whether
        // or not anybody pressed anything -- and on a bench that reads as the
        // button working.
        let mut c = with_reference("jump_phase", 1, "[0.01]", 1);
        c.layout.reference.as_mut().unwrap().go_event = Some("jump_go".into());
        // Re-attach: the builder reads the event off the *parsed* trajectory,
        // which took its copy of the spec when it was built.
        c.attach_reference(&table_for(&["a".to_string()]), &["a".to_string()]).unwrap();

        let mut cf = cfg(&["jump_phase"], &[1]);
        cf.reference_go_delay_s = 0.10; // would have fired five ticks in
        let mut b = ObservationBuilder::new(cf, &c, 1).unwrap();
        assert!(b.go_is_bound(), "the contract names a button");
        let s = RobotState::new(1);

        for i in 0..40 {
            let p = b.build(&s, &Command::default(), &[], &[]).unwrap()[0];
            assert_eq!(p, 0.0, "tick {i}: nothing pressed, so nothing has happened");
        }

        b.trigger_go();
        let after: Vec<f32> = (0..6)
            .map(|_| b.build(&s, &Command::default(), &[], &[]).unwrap()[0])
            .collect();
        assert_eq!(after[0], 0.0, "the press itself is t = 0");
        assert!(after[5] > after[1] && after[1] > 0.0, "then it runs: {after:?}");

        // And pressing again restarts it, rather than being ignored because the
        // motion is already past its end.
        b.trigger_go();
        assert_eq!(b.build(&s, &Command::default(), &[], &[]).unwrap()[0], 0.0);
    }

    #[test]
    fn the_preview_is_relative_to_the_home_pose_and_reads_q_not_q_cmd() {
        let c = with_reference("ref_future", 2, "[0.0, 0.05]", 2);
        let mut cf = cfg(&["ref_future"], &[1]);
        cf.reference_go_delay_s = 0.0;
        let mut b = ObservationBuilder::new(cf, &c, 1).unwrap();
        let s = RobotState::new(1);
        let out = b.build(&s, &Command::default(), &[], &[]).unwrap();
        // t_since_go 0: frame 10 and frame 15 of q, each minus the home pose.
        assert_eq!(out, &[10.0 - 0.5, 15.0 - 0.5], "q, home-relative");
        assert!(out[0] < 100.0, "q_cmd would be 110 here");
    }

    #[test]
    fn the_residual_baseline_comes_from_the_tick_the_policy_answered() {
        let c = with_reference("jump_phase", 1, "[0.01]", 1);
        let mut cf = cfg(&["jump_phase"], &[1]);
        cf.reference_go_delay_s = 0.0;
        let mut b = ObservationBuilder::new(cf, &c, 1).unwrap();
        let s = RobotState::new(1);

        b.build(&s, &Command::default(), &[], &[]).unwrap();
        // tick 0: t_since_go 0, t_abs = 0 + 0.1 + 0.02 = 0.12 s, q_cmd frame 6.
        assert_eq!(b.reference_baseline(), Some(&[106.0][..]));
        b.build(&s, &Command::default(), &[], &[]).unwrap();
        // tick 1 advances one control step, not one recording frame.
        assert_eq!(b.reference_baseline(), Some(&[107.0][..]));
    }

    /// The dance's shape: a mode that **observes** a recording without riding
    /// one. Copied field-for-field from the exported `dance.json` -- no
    /// `residual_action`, so no `q_cmd`, no `control_hz` and no baseline -- and
    /// the table carries `qd`, which is what it reads.
    fn observing_reference(term: &str, dim: usize) -> Contract {
        let text = format!(
            r#"{{"obs_joint_order": ["a"], "action_joint_order": ["a"], "action_scale": 0.25,
                "default_joint_pos": {{"a": 0.5}},
                "control": {{"kp": 10.0, "kd": 0.5, "effort_limit": 2.0, "control_hz": 50.0}},
                "observation": {{"dim": {dim}, "history_length": 1,
                                 "terms": [{{"name": "{term}", "dim": {dim}}}]}},
                "action": {{"dim": 1}},
                "reference": {{"file": "t.json", "residual_action": false, "rec_hz": 50.0,
                               "duration": 0.6, "lookahead_s": [0.04],
                               "starts_on_entry": true, "n_frames": 30}}}}"#
        );
        // qd[f] = 10 f, so a row reads back as its own index times ten: the
        // wrong table or an off-by-one is a wrong number, not a wrong shape.
        let q: Vec<Vec<f32>> = (0..30).map(|f| vec![f as f32]).collect();
        let qd: Vec<Vec<f32>> = (0..30).map(|f| vec![10.0 * f as f32]).collect();
        let table = serde_json::json!({"joint_order": ["a"], "q": q, "qd": qd}).to_string();
        let mut c = Contract::from_str(&text, &["a".to_string()]).unwrap();
        c.attach_reference(&table, &["a".to_string()]).unwrap();
        c
    }

    #[test]
    fn a_mode_that_observes_a_recording_without_riding_one_still_builds() {
        // `build` sampled a residual baseline off `q_cmd` whenever a reference
        // was attached at all -- and this mode has none, so the sample ran
        // `clamp(0, -1)` on an empty table and aborted. Not a wrong number: the
        // first tick after the button that reaches the mode panics. On the
        // board that is a crash; in a browser it is the Worker dying with
        // `RuntimeError: unreachable` and no mention of a baseline in it.
        //
        // Nothing caught it because every test and every recorded reference
        // vector ran the mode the cascade *starts* in. Pressing the button was
        // the untested act.
        let c = observing_reference("ref_joint_vel", 1);
        let mut cf = cfg(&["ref_joint_vel"], &[1]);
        cf.reference_go_delay_s = 0.0;
        let mut b = ObservationBuilder::new(cf, &c, 1).unwrap();
        let s = RobotState::new(1);
        let out = b.build(&s, &Command::default(), &[], &[]).unwrap();
        assert_eq!(out, &[0.0], "qd frame 0");
        let out = b.build(&s, &Command::default(), &[], &[]).unwrap();
        assert_eq!(out, &[10.0], "qd frame 1 -- the clock advanced, and it read qd");
        // And it offers no baseline at all, rather than a zeroed one the action
        // decoder would add to the residual as though it were a pose.
        assert_eq!(b.reference_baseline(), None);
    }

    #[test]
    fn a_policy_with_no_recording_offers_no_baseline() {
        // The velocity policies, which is every other mode: the decoder must
        // fall back to the home pose rather than to whatever was last in a
        // buffer.
        let c = contract(3, r#"[{"name":"base_ang_vel","dim":3}]"#, 1);
        let mut b = ObservationBuilder::new(cfg(&["base_ang_vel"], &[1]), &c, 1).unwrap();
        b.build(&RobotState::new(1), &Command::default(), &[], &[]).unwrap();
        assert_eq!(b.reference_baseline(), None);
    }

    /// Frames `stride` ticks apart, oldest first, this tick last -- what
    /// `CircularBuffer.buffer[:, ::stride]` gives in training, including the
    /// first push filling every slot.
    ///
    /// Beside it a term with two consecutive frames, so the two spans have to
    /// share one ring: the strided term reaches back four ticks and the other
    /// one, and each has to take its own slots from the same history.
    #[test]
    fn a_strided_term_takes_its_frames_stride_ticks_apart() {
        let c = contract(
            9,
            r#"[{"name":"actions","dim":3,"history_length":3,"history_stride":2},
                {"name":"base_ang_vel","dim":6,"history_length":2}]"#,
            1,
        );
        assert_eq!(c.term_stride, vec![2, 1], "the contract reads the field");
        let mut cfg = cfg(&["actions", "base_ang_vel"], &[3, 2]);
        cfg.term_stride = c.term_stride.clone();
        let mut b = ObservationBuilder::new(cfg, &c, 1).unwrap();
        let mut s = state();
        let mut last = Vec::new();
        for tick in 1..=6 {
            s.imu.gyro = [tick as f32 * 10.0, 0.0, 0.0];
            last = b.build(&s, &Command::default(), &[tick as f32], &[0.0]).unwrap().to_vec();
            if tick == 1 {
                assert_eq!(&last[..3], &[1.0, 1.0, 1.0], "the first push fills every slot");
            }
            if tick == 2 {
                assert_eq!(&last[..3], &[1.0, 1.0, 2.0]);
            }
        }
        assert_eq!(&last[..3], &[2.0, 4.0, 6.0], "frames 2 ticks apart, oldest first");
        assert_eq!((last[3], last[6]), (50.0, 60.0), "the unstrided term takes the last two");

        // The control group: the same term at stride 1 is the last three
        // ticks, which is what a builder ignoring the field would emit for
        // both -- the right width, from a window half as long.
        let mut cfg1 = self::cfg(&["actions", "base_ang_vel"], &[3, 2]);
        cfg1.term_stride = vec![1, 1];
        let mut b1 = ObservationBuilder::new(cfg1, &c, 1).unwrap();
        for tick in 1..=6 {
            last = b1.build(&s, &Command::default(), &[tick as f32], &[0.0]).unwrap().to_vec();
        }
        assert_eq!(&last[..3], &[4.0, 5.0, 6.0]);
    }

    /// `[twist, pitch, roll, height - standing]`: the command term's own order,
    /// centred the way the training observation is.
    #[test]
    fn posture_command_is_the_command_s_order_centred_on_standing() {
        let c = contract(4, r#"[{"name":"posture_command","dim":4}]"#, 1);
        let mut with = cfg(&["posture_command"], &[1]);
        with.posture_neutral_height = Some(0.107);
        let mut b = ObservationBuilder::new(with, &c, 1).unwrap();
        let cmd = Command { base_twist: 0.1, base_pitch: -0.2, base_roll: 0.3, height: 0.12, ..Default::default() };
        let out = b.build(&state(), &cmd, &[0.0], &[0.0]).unwrap();
        let want = [0.1, -0.2, 0.3, 0.12 - 0.107];
        for (got, want) in out.iter().zip(want) {
            assert!((got - want).abs() < 1e-6, "{out:?}");
        }
        // Not `base_pose`, which is three of the same quantities ordered
        // [pitch, roll, twist]: feeding one as the other is silent.
        assert_ne!(out[0], cmd.base_pitch);

        // Without the standing height from the contract it is refused rather
        // than built uncentred -- 0.107 m out on that channel, for ever.
        let e = ObservationBuilder::new(cfg(&["posture_command"], &[1]), &c, 1).err();
        assert!(matches!(e, Some(Error::MissingParam { .. })), "{e:?}");
    }

    /// The tempo follows the command, integrated: first frame at phase 0, each
    /// tick advanced by that tick's `f * dt`, clamped at both ends, and the
    /// clock still running -- shown as (0, 0) -- while the command stands.
    ///
    /// In `BeforeFrame`, the order of a contract with no `advance`: every
    /// posture policy exported before the field existed runs this way, so it
    /// is pinned as it was rather than retired.
    #[test]
    fn a_cadence_clock_integrates_the_commanded_tempo() {
        let c = contract(2, r#"[{"name":"gait_phase","dim":2}]"#, 1);
        let mut with = cfg(&["gait_phase"], &[1]);
        with.control_dt = 0.005;
        let law = Cadence {
            stride: 0.072,
            freq_min: 2.0,
            freq_max: 5.5556,
            turn_radius: 0.2,
            advance: Advance::BeforeFrame,
        };
        with.cadence = Some(law);
        let mut b = ObservationBuilder::new(with, &c, 1).unwrap();

        let walk = Command { lin_vel_x: 0.4, ..Default::default() };
        let fast = Command { lin_vel_x: 1.2, ..Default::default() };
        let turn = Command { yaw_rate: 2.0, ..Default::default() };
        assert!((law.frequency(&walk) - 0.4 / 0.144).abs() < 1e-5);
        assert_eq!(law.frequency(&fast), 5.5556, "clamped at the top");
        assert!((law.frequency(&turn) - 0.4 / 0.144).abs() < 1e-5, "a turn swings a foot too");
        assert_eq!(law.frequency(&Command::default()), 2.0, "and at the bottom");

        // The first frame is phase 0; each later one is advanced by its own
        // command first -- so the speed change on tick 2 shows on tick 2.
        let mut phase = 0.0f32;
        for (i, cmd) in [walk, walk, fast, fast, turn, walk].iter().enumerate() {
            if i > 0 {
                phase = (phase + law.frequency(cmd) * 0.005).rem_euclid(1.0);
            }
            let out = b.build(&state(), cmd, &[0.0], &[0.0]).unwrap().to_vec();
            let a = phase * std::f32::consts::TAU;
            assert!((out[0] - a.sin()).abs() < 1e-6 && (out[1] - a.cos()).abs() < 1e-6,
                    "tick {i}: {out:?} against phase {phase}");
        }
        assert!(phase > 0.0);

        // Standing: the policy is shown (0, 0), and the clock keeps its phase.
        let out = b.build(&state(), &Command::default(), &[0.0], &[0.0]).unwrap().to_vec();
        assert_eq!(out, vec![0.0, 0.0]);

        // The control group: a fixed clock with the same builder settings does
        // not move with the command. Two ticks at different speeds give the
        // same step.
        let mut fixed = cfg(&["gait_phase"], &[1]);
        fixed.control_dt = 0.005;
        let mut f = ObservationBuilder::new(fixed, &c, 1).unwrap();
        f.build(&state(), &walk, &[0.0], &[0.0]).unwrap();
        let slow = f.build(&state(), &walk, &[0.0], &[0.0]).unwrap().to_vec();
        let mut g = ObservationBuilder::new(
            ObsConfig { control_dt: 0.005, ..cfg(&["gait_phase"], &[1]) }, &c, 1).unwrap();
        g.build(&state(), &fast, &[0.0], &[0.0]).unwrap();
        let quick = g.build(&state(), &fast, &[0.0], &[0.0]).unwrap().to_vec();
        assert_eq!(slow, quick, "a fixed clock does not follow the command");
    }

    /// `"after_frame"`: each frame shows the phase one step of the *previous*
    /// frame's tempo on, which is what training has shown since it read one
    /// clock -- so a speed change on tick 2 shows on tick 3, not on tick 2.
    ///
    /// The control is the same commands through `BeforeFrame`: identical until
    /// the tempo changes and different from then on. Without it this would
    /// pass on a builder that ignored the field, for as long as the command
    /// held still.
    #[test]
    fn after_frame_shows_each_frame_the_tempo_of_the_one_before() {
        let c = contract(2, r#"[{"name":"gait_phase","dim":2}]"#, 1);
        let law = |advance| Cadence {
            stride: 0.072,
            freq_min: 2.0,
            freq_max: 5.5556,
            turn_radius: 0.2,
            advance,
        };
        let builder = |advance| {
            let mut with = cfg(&["gait_phase"], &[1]);
            with.control_dt = 0.005;
            with.cadence = Some(law(advance));
            ObservationBuilder::new(with, &c, 1).unwrap()
        };
        let (mut after, mut before) = (builder(Advance::AfterFrame), builder(Advance::BeforeFrame));

        let walk = Command { lin_vel_x: 0.4, ..Default::default() };
        let fast = Command { lin_vel_x: 1.2, ..Default::default() };
        let turn = Command { yaw_rate: 2.0, ..Default::default() };
        let angle = |out: &[f32]| out[0].atan2(out[1]).rem_euclid(std::f32::consts::TAU);

        let k = law(Advance::AfterFrame);
        let (mut phase, mut shown) = (0.0f32, None::<Command>);
        let mut parted = false;
        for (i, cmd) in [walk, walk, fast, fast, turn, walk].iter().enumerate() {
            if let Some(prev) = shown {
                phase = (phase + k.frequency(&prev) * 0.005).rem_euclid(1.0);
            }
            shown = Some(*cmd);
            let got = after.build(&state(), cmd, &[0.0], &[0.0]).unwrap().to_vec();
            let a = phase * std::f32::consts::TAU;
            assert!((got[0] - a.sin()).abs() < 1e-6 && (got[1] - a.cos()).abs() < 1e-6,
                    "tick {i}: {got:?} against phase {phase}");

            let old = before.build(&state(), cmd, &[0.0], &[0.0]).unwrap().to_vec();
            let apart = (angle(&got) - angle(&old)).abs();
            if i < 2 {
                assert!(apart < 1e-5, "tick {i}: the two orders differ before the tempo did");
            }
            parted |= apart > 1e-3;
        }
        assert!(parted, "the two orders never differed, so the field changed nothing");

        // A new entry into the mode starts at 0 in either order.
        after.reset();
        let out = after.build(&state(), &fast, &[0.0], &[0.0]).unwrap().to_vec();
        assert_eq!(out, vec![0.0, 1.0], "the first frame after a reset is phase 0");
    }

    /// The contract says which order, and only two answers are accepted: none,
    /// which is every export from before the field, and `"after_frame"`.
    #[test]
    fn the_contract_says_when_the_cadence_clock_advances() {
        let term = |advance: &str| {
            let text = format!(
                r#"[{{"name":"gait_phase","dim":2,"params":{{"stride":0.072,"freq_min":2.0,
                     "freq_max":5.5556,"turn_radius":0.2,"command_threshold":0.05{advance}}}}}]"#
            );
            contract(2, &text, 1)
        };
        let law = |c: &Contract| Cadence::from_term(&c.layout.observation.terms[0]).unwrap();

        let old = term("");
        assert_eq!(law(&old).unwrap().advance, Advance::BeforeFrame, "absent is the old order");
        let new = term(r#","advance":"after_frame""#);
        assert_eq!(law(&new).unwrap().advance, Advance::AfterFrame);
        for bad in [r#","advance":"before_frame""#, r#","advance":"after""#, r#","advance":1"#] {
            let c = term(bad);
            let e = law(&c).err();
            assert!(matches!(e, Some(Error::UnknownParamValue { .. })), "{bad}: {e:?}");
        }
    }

    #[test]
    fn terms_that_disagree_with_the_layout_width_are_refused() {
        let c = contract(99, r#"[{"name":"base_ang_vel","dim":3}]"#, 1);
        assert!(matches!(
            ObservationBuilder::new(cfg(&["base_ang_vel"], &[1]), &c, 1),
            Err(Error::DimMismatch { built: 3, layout: 99 })
        ));
    }
}
