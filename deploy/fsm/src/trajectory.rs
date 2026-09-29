//! The recorded motion a residual policy corrects, and the clock that reads it.
//!
//! A velocity policy needs none of this: its command is three numbers and every
//! observation is measurable now. A reference-guided one is not runnable without
//! the recording it trained against -- `jumper.jump`'s action *is* a residual on
//! it -- so the recording travels with the policy and this module is how the
//! controller reads it.
//!
//! **Two tables, and they are not interchangeable.** `q` is the state the
//! recording reached and feeds the `ref_future` preview; `q_cmd` is what the
//! recorded controller commanded and is the residual's baseline. They are not
//! even the same length -- 368 frames at 250 Hz against 295 at 200 Hz for
//! `high_jump_flat`. `tasks/jumper/jump/mdp/actions.py` measured the gap at
//! 0.3 rad across the push-off, which is the difference between a jump and a
//! twitch, and nothing downstream can see which one was used.
//!
//! Every formula here is `tasks/jumper/jump/mdp/reference.py`'s, transcribed. The
//! clamps especially: negative `t_since_go` clamping to the go frame is what
//! lets a controller sit in the recorded stand while it waits, and dropping it
//! would read off the front of the table into the approach.

use serde::Deserialize;

/// Seconds from entering a reference-guided mode to its `go`, where no host has
/// been told otherwise.
///
/// **A placeholder for a button.** One named constant rather than a literal per
/// host, because four hosts each picking their own would make the same motion
/// start at four different times and every one of them would look deliberate.
pub const DEFAULT_GO_DELAY_S: f64 = 2.0;

/// A contract's `reference` block: how to read the companion table.
///
/// Every field is a fact about the recording or about how the policy was
/// trained against it, so every field comes from the export rather than from
/// this side. A board that picked its own `rec_hz` would replay the same motion
/// at the wrong speed and look entirely healthy doing it.
#[derive(Debug, Clone, Deserialize)]
pub struct ReferenceSpec {
    /// The companion table, a file name beside the contract.
    pub file: String,
    /// Whether the action is a residual on `baseline` rather than on the home
    /// pose. `jumper.jump`'s is; `jumper.dance`'s is not -- its policy drives the
    /// joints outright and the recording is a target it is scored against.
    pub residual_action: bool,
    /// Whether the recording starts when the mode is entered.
    ///
    /// What begins a recording is not a detail. `jumper.dance` is a piece of
    /// music, and `jumper.jump` is asked for by switching into it, so both
    /// start on entry -- which is when the policy takes over, once the
    /// switch-in ramp has reached the home pose, never while the robot is still
    /// sliding there. A motion that wants a person to pick the moment inside the
    /// mode names a `go_event` instead; the jump did until 2026-09-26, and the
    /// policy had never been trained standing in its go frame waiting for one.
    #[serde(default)]
    pub starts_on_entry: bool,
    /// Rate of the `q` table, Hz -- the recording's own.
    pub rec_hz: f64,
    /// Rate of the `q_cmd` table, Hz. Different from `rec_hz`, deliberately:
    /// they were captured by different things. Absent where there is no
    /// residual baseline.
    #[serde(default)]
    pub control_hz: Option<f64>,
    /// Seconds of recorded preparatory stand before the motion starts. Zero
    /// for a recording that starts at its first frame.
    #[serde(default)]
    pub t_go: f64,
    /// The frame `t_go` lands on. Negative times clamp here, not to zero.
    #[serde(default)]
    pub go_frame: f64,
    /// Whole recording, s.
    pub duration: f64,
    /// `duration - t_go`: what `jump_phase` runs 0 -> 1 across. Absent for a
    /// recording with no such term.
    #[serde(default)]
    pub span: Option<f64>,
    /// Preview offsets, s. The width of `ref_future` is this many times the
    /// observed joint count.
    pub lookahead_s: Vec<f64>,
    /// Which table the residual sits on. Only `q_cmd` is implemented; a
    /// contract naming anything else is refused rather than quietly read off
    /// `q`, which is the 0.3 rad mistake above. Absent where there is no
    /// residual.
    #[serde(default)]
    pub baseline: Option<String>,
    /// The baseline is sampled one control step *ahead*, because the target is
    /// held across the interval. From the training env's `step_dt`.
    #[serde(default)]
    pub baseline_lead_s: f64,
    /// The name of the `[[fsm.button]]` that starts this motion, if the task
    /// declared one.
    ///
    /// A **name**, not a button: which physical control means what is the
    /// robot's decision and belongs to the deploy manifest, exactly as it does
    /// for every other button in this crate. `controller/vocabulary.json` says
    /// what a pad can say; a manifest binds a word to a name; this says which
    /// name is our `go`. No new mechanism -- `Buttons` already tracks the
    /// gesture and already refuses one control meaning two things.
    ///
    /// `None` leaves the motion on the timer, which is a bench aid rather than
    /// something a person can use.
    #[serde(default)]
    pub go_event: Option<String>,
    pub n_frames: usize,
    /// Rows in `q_cmd`. Zero where there is no residual baseline.
    #[serde(default)]
    pub n_cmd: usize,
}

/// The companion file `scripts/export.py` writes beside the contract.
///
/// Which tables are present follows from what the policy observes, and
/// `Trajectory::parse` checks the pairing rather than defaulting a missing one
/// to zeros: a reference of all zeros is a motionless target that a standing
/// robot tracks perfectly.
#[derive(Debug, Deserialize)]
struct Table {
    joint_order: Vec<String>,
    /// The state the recording reached. Always present.
    q: Vec<Vec<f32>>,
    /// What the recorded controller commanded. The residual's baseline.
    #[serde(default)]
    q_cmd: Vec<Vec<f32>>,
    /// Joint velocities, for a policy that observes the reference's.
    #[serde(default)]
    qd: Vec<Vec<f32>>,
    /// The anchor body's orientation, `(w, x, y, z)` per frame, for a policy
    /// that observes how far its base has tilted from the recording's.
    #[serde(default)]
    root_quat: Vec<[f32; 4]>,
}

/// A parsed recording, in wire order, with the spec that says how to read it.
#[derive(Debug)]
pub struct Trajectory {
    spec: ReferenceSpec,
    joints: usize,
    /// Row-major `n_frames x joints`. Flat because a board reads this every
    /// tick, once per lookahead offset.
    q: Vec<f32>,
    q_cmd: Vec<f32>,
    /// Joint velocities, empty where the recording carries none.
    qd: Vec<f32>,
    /// The anchor body's orientation per frame, empty where it carries none.
    root_quat: Vec<[f32; 4]>,
    n_frames: usize,
    n_cmd: usize,
}

impl Trajectory {
    /// Parse the companion table against the spec and the robot's wire order.
    ///
    /// Checks the things that are silent when wrong: a joint order that does
    /// not match the robot's is a robot tracking the wrong limbs, and a row
    /// count that disagrees with the contract means the two files came from
    /// different exports.
    pub fn parse(spec: ReferenceSpec, text: &str, wire: &[String]) -> Result<Self, String> {
        let file = spec.file.clone();
        let bad = |m: String| Err(format!("{file}: {m}"));

        if spec.residual_action && spec.baseline.as_deref() != Some("q_cmd") {
            return bad(format!(
                "the residual sits on baseline {:?}, and this controller only implements \
                 'q_cmd'. Reading it off `q` instead is a 0.3 rad error across the \
                 push-off that looks like a weak jump, not like a bug.",
                spec.baseline
            ));
        }
        if spec.starts_on_entry && spec.go_event.is_some() {
            return bad(
                "the recording both starts on mode entry and names a `go` button. \
                 One of those never happens, and which one is not discoverable from \
                 a robot that did nothing."
                    .into(),
            );
        }
        let t: Table = serde_json::from_str(text)
            .map_err(|e| format!("{file} is not a valid reference table: {e}"))?;

        if t.joint_order != wire {
            return bad(format!(
                "it is in a different joint order than the robot.\n  table: {:?}\n  \
                 robot: {:?}\nThe export permutes it to wire order, so a mismatch means \
                 this table belongs to another robot -- and a permuted trajectory is a \
                 robot moving the wrong joints, which nothing downstream can detect.",
                t.joint_order, wire
            ));
        }
        let joints = wire.len();
        if t.q.len() != spec.n_frames || t.q_cmd.len() != spec.n_cmd {
            return bad(format!(
                "it has {} q rows and {} q_cmd rows; the contract says {} and {}. The \
                 contract and the table came from different exports.",
                t.q.len(),
                t.q_cmd.len(),
                spec.n_frames,
                spec.n_cmd
            ));
        }
        if t.q.is_empty() {
            return bad("it carries no `q` at all".into());
        }
        // Every table that is present has to be the right shape, and every one
        // the spec implies has to be present. A missing table defaulting to
        // empty would read as a motionless reference -- which a standing robot
        // tracks perfectly, and which no log distinguishes from success.
        for (name, rows) in [("q", &t.q), ("q_cmd", &t.q_cmd), ("qd", &t.qd)] {
            if let Some(i) = rows.iter().position(|r| r.len() != joints) {
                return bad(format!("{name} row {i} is {} wide, not {joints}", rows[i].len()));
            }
        }
        // `qd` and `root_quat` are per-frame companions of `q` and are indexed
        // with it. `q_cmd` is **not** and must not be checked against it: it was
        // captured by a different thing at a different rate -- 295 rows against
        // 368 for `high_jump_flat` -- which is the fact the module docstring
        // opens with, and which this check asserted away for one commit.
        if !t.qd.is_empty() && t.qd.len() != t.q.len() {
            return bad(format!("qd has {} rows and q has {}", t.qd.len(), t.q.len()));
        }
        if !t.root_quat.is_empty() && t.root_quat.len() != t.q.len() {
            return bad(format!(
                "root_quat has {} rows and q has {}",
                t.root_quat.len(),
                t.q.len()
            ));
        }
        if spec.residual_action && t.q_cmd.is_empty() {
            return bad("the action is a residual on `q_cmd`, and there is no q_cmd".into());
        }
        if spec.rec_hz <= 0.0 {
            return bad(format!("rec_hz is {}", spec.rec_hz));
        }
        if spec.residual_action && !spec.control_hz.is_some_and(|h| h > 0.0) {
            return bad(format!("control_hz is {:?}, and q_cmd is read at it", spec.control_hz));
        }
        if spec.span.is_some_and(|s| s <= 0.0) {
            return bad(format!("span is {:?}", spec.span));
        }
        if spec.duration <= 0.0 {
            return bad(format!("duration is {}", spec.duration));
        }
        let (n_frames, n_cmd) = (t.q.len(), t.q_cmd.len());
        Ok(Self {
            spec,
            joints,
            q: t.q.into_iter().flatten().collect(),
            q_cmd: t.q_cmd.into_iter().flatten().collect(),
            qd: t.qd.into_iter().flatten().collect(),
            root_quat: t.root_quat,
            n_frames,
            n_cmd,
        })
    }

    pub fn spec(&self) -> &ReferenceSpec {
        &self.spec
    }

    pub fn lookahead(&self) -> &[f64] {
        &self.spec.lookahead_s
    }

    /// The `[[fsm.button]]` name that starts this motion, when there is one.
    pub fn go_event(&self) -> Option<&str> {
        self.spec.go_event.as_deref()
    }

    pub fn joints(&self) -> usize {
        self.joints
    }

    /// Whether the recording has played out.
    ///
    /// A recorded motion is a thing that **ends**, which is the one way it is
    /// unlike a gait. Holding the last frame afterwards is what the tables do
    /// and is a robot frozen in a landing pose; the controller asks this so it
    /// can hand back to whatever it was doing before.
    ///
    /// Measured from the recording's own end, not from `span`: `span` is what
    /// `jump_phase` runs across, and a motion whose phase has reached 1 may
    /// still have frames left.
    pub fn finished(&self, t_since_go: f64) -> bool {
        self.spec.t_go + t_since_go >= self.spec.duration
    }

    /// How far the motion has run, 0 -> 1. `reference.py::phase`.
    ///
    /// Clamped at both ends: before `go` it is 0 and after the motion it is 1,
    /// and the policy saw both. Letting it run past 1 is an input it never saw.
    pub fn phase(&self, t_since_go: f64) -> f32 {
        // `span` is absent for a recording with no phase term; a zero would
        // be a division, so this reports "not started" rather than NaN.
        let Some(span) = self.spec.span else { return 0.0 };
        (t_since_go / span).clamp(0.0, 1.0) as f32
    }

    /// `reference.py::index_of`: negative times clamp to the go frame, whose
    /// pose is the recorded stand, so a controller waiting for `go` previews
    /// the stand rather than the approach.
    fn index_of(&self, t_since_go: f64) -> f64 {
        let idx = (self.spec.t_go + t_since_go) * self.spec.rec_hz;
        idx.clamp(self.spec.go_frame, (self.n_frames - 1) as f64)
    }

    /// The recorded state at `t_since_go`, wire order, into `out`.
    pub fn q_at_into(&self, t_since_go: f64, out: &mut [f32]) {
        lerp_row(&self.q, self.joints, self.n_frames, self.index_of(t_since_go), out);
    }

    /// The residual's baseline at `t_since_go`, wire order, into `out`.
    ///
    /// Sampled `baseline_lead_s` ahead and off `q_cmd`, both from the contract:
    /// `mdp/actions.py` samples the interval's **end** because the target is
    /// held across it. Absolute time here, not time-since-go -- `sample_cmd`
    /// indexes the recording from its start.
    pub fn baseline_into(&self, t_since_go: f64, out: &mut [f32]) {
        let t_abs = t_since_go + self.spec.t_go + self.spec.baseline_lead_s;
        // Checked at parse: a residual contract has both of these.
        let hz = self.spec.control_hz.unwrap_or(self.spec.rec_hz);
        let idx = (t_abs * hz).clamp(0.0, (self.n_cmd.max(1) - 1) as f64);
        lerp_row(&self.q_cmd, self.joints, self.n_cmd, idx, out);
    }
}

impl Trajectory {
    /// Where in the recording a tick sits, as a frame index.
    ///
    /// Clamped at the last frame rather than wrapped. A recording is a thing
    /// that ends -- `finished` is how the controller hands back -- and wrapping
    /// would show the policy the opening bars while it is finishing the closing
    /// ones, in the one signal whose whole purpose is to be predictable.
    pub fn frame_at(&self, t: f64) -> f64 {
        (t * self.spec.rec_hz).clamp(0.0, (self.n_frames - 1) as f64)
    }

    /// Where the recording is on its own circle, as `(sin, cos)`.
    ///
    /// sin/cos rather than the raw fraction for the reason the gait clock uses
    /// them: the quantity is circular, and a value in [0, 1) puts a
    /// discontinuity at the wrap where 0.999 and 0.001 are adjacent in time and
    /// maximally far apart as numbers.
    pub fn clip_phase(&self, t: f64) -> [f32; 2] {
        let a = self.frame_at(t) * std::f64::consts::TAU / self.n_frames as f64;
        [a.sin() as f32, a.cos() as f32]
    }

    /// The recorded joint velocities at `t`, wire order. Empty when the
    /// recording carries none.
    pub fn qd_at_into(&self, t: f64, out: &mut [f32]) -> bool {
        if self.qd.is_empty() {
            return false;
        }
        lerp_row(&self.qd, self.joints, self.n_frames, self.frame_at(t), out);
        true
    }

    /// The anchor body's orientation at `t`, `(w, x, y, z)`.
    ///
    /// Nearest frame rather than interpolated: interpolating quaternions
    /// componentwise is not a rotation, and at the control rate the recording
    /// is sampled at, the nearest frame is the frame.
    pub fn root_quat_at(&self, t: f64) -> Option<[f32; 4]> {
        if self.root_quat.is_empty() {
            return None;
        }
        let i = (self.frame_at(t).round() as usize).min(self.root_quat.len() - 1);
        Some(self.root_quat[i])
    }

    pub fn has_qd(&self) -> bool {
        !self.qd.is_empty()
    }

    pub fn has_root_quat(&self) -> bool {
        !self.root_quat.is_empty()
    }

    /// Whether the recording begins when the mode does.
    pub fn starts_on_entry(&self) -> bool {
        self.spec.starts_on_entry
    }
}

/// Linear interpolation between row `floor(idx)` and the next, as
/// `reference.py::sample` does it. The last row repeats rather than wrapping.
fn lerp_row(flat: &[f32], joints: usize, rows: usize, idx: f64, out: &mut [f32]) {
    // A table the caller believes in and the recording does not have. Both of
    // the panics this replaces name arithmetic rather than the mistake: an
    // empty table reaches `clamp(0, -1)`, which reports `min > max`, and a
    // short one slices past the end. Through wasm both arrive as
    // `RuntimeError: unreachable`.
    assert!(
        rows > 0 && flat.len() >= rows * joints,
        "reference table has {} values, not the {rows} x {joints} the contract says",
        flat.len()
    );
    let lo = (idx.floor() as isize).clamp(0, rows as isize - 1) as usize;
    let hi = (lo + 1).min(rows - 1);
    let w = (idx - lo as f64) as f32;
    let (a, b) = (&flat[lo * joints..][..joints], &flat[hi * joints..][..joints]);
    for j in 0..joints.min(out.len()) {
        out[j] = a[j] + (b[j] - a[j]) * w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ReferenceSpec {
        ReferenceSpec {
            file: "t.json".into(),
            residual_action: true,
            starts_on_entry: false,
            rec_hz: 100.0,
            control_hz: Some(50.0),
            t_go: 0.1,
            go_frame: 10.0,
            duration: 0.3,
            span: Some(0.2),
            lookahead_s: vec![0.01, 0.02],
            baseline: Some("q_cmd".into()),
            baseline_lead_s: 0.02,
            go_event: None,
            n_frames: 30,
            n_cmd: 15,
        }
    }

    /// `q[f][0] = f`, `q_cmd[f][0] = 100 + f`: a row's own index, so a sample
    /// reads back as the frame it came from and an off-by-one is visible.
    fn table() -> String {
        let q: Vec<Vec<f32>> = (0..30).map(|f| vec![f as f32, 0.0]).collect();
        let c: Vec<Vec<f32>> = (0..15).map(|f| vec![100.0 + f as f32, 0.0]).collect();
        serde_json::json!({"joint_order": ["a", "b"], "q": q, "q_cmd": c}).to_string()
    }

    fn wire() -> Vec<String> {
        vec!["a".into(), "b".into()]
    }

    fn traj() -> Trajectory {
        Trajectory::parse(spec(), &table(), &wire()).unwrap()
    }

    #[test]
    fn phase_is_clamped_at_both_ends() {
        let t = traj();
        assert_eq!(t.phase(-1.0), 0.0, "before go");
        assert_eq!(t.phase(0.0), 0.0);
        assert!((t.phase(0.1) - 0.5).abs() < 1e-6, "half of span 0.2");
        assert_eq!(t.phase(0.2), 1.0);
        assert_eq!(t.phase(99.0), 1.0, "past the end is an input the policy saw");
    }

    #[test]
    fn waiting_for_go_previews_the_stand_not_the_approach() {
        // The whole reason index_of clamps to go_frame rather than to 0. A
        // controller sitting at t_since_go = -5 s must read frame 10, not
        // frame -490 saturated to 0, which is the middle of the approach.
        let t = traj();
        let mut out = [0.0f32; 2];
        t.q_at_into(-5.0, &mut out);
        assert_eq!(out[0], 10.0, "the go frame, whose pose is the stand");

        // Control: with the clamp at 0 this would read frame 0 instead.
        t.q_at_into(0.0, &mut out);
        assert_eq!(out[0], 10.0, "t_since_go = 0 is the go frame too");
    }

    #[test]
    fn q_is_sampled_at_the_recording_rate_and_interpolated() {
        let t = traj();
        let mut out = [0.0f32; 2];
        t.q_at_into(0.05, &mut out);
        assert_eq!(out[0], 15.0, "0.1 + 0.05 at 100 Hz is frame 15");
        t.q_at_into(0.055, &mut out);
        assert!((out[0] - 15.5).abs() < 1e-5, "half a frame between 15 and 16, got {}", out[0]);
    }

    #[test]
    fn the_baseline_reads_q_cmd_at_its_own_rate_and_one_step_ahead() {
        // Three things at once, each of which is silent when wrong: the other
        // table, the other rate, and the lead.
        let t = traj();
        let mut out = [0.0f32; 2];
        t.baseline_into(0.0, &mut out);
        // t_abs = 0 + 0.1 + 0.02 = 0.12 s; at 50 Hz that is q_cmd frame 6.
        assert_eq!(out[0], 106.0, "q_cmd frame 6, not q frame 12");
    }

    #[test]
    fn both_tables_run_off_their_own_end_without_wrapping() {
        let t = traj();
        let mut out = [0.0f32; 2];
        t.q_at_into(600.0, &mut out);
        assert_eq!(out[0], 29.0, "last q frame, not frame 0");
        t.baseline_into(600.0, &mut out);
        assert_eq!(out[0], 114.0, "last q_cmd frame");
    }

    #[test]
    fn a_table_in_another_joint_order_is_refused() {
        let e = Trajectory::parse(spec(), &table(), &["b".into(), "a".into()]).unwrap_err();
        assert!(e.contains("different joint order"), "{e}");
    }

    #[test]
    fn a_table_that_disagrees_with_the_contract_is_refused() {
        let mut s = spec();
        s.n_frames = 31;
        let e = Trajectory::parse(s, &table(), &wire()).unwrap_err();
        assert!(e.contains("different exports"), "{e}");
    }

    #[test]
    fn a_baseline_this_controller_does_not_implement_is_refused() {
        // Rather than silently reading `q`, which trains and replays and is
        // 0.3 rad wrong exactly where the jump happens.
        let mut s = spec();
        s.baseline = Some("q".into());
        let e = Trajectory::parse(s, &table(), &wire()).unwrap_err();
        assert!(e.contains("only implements"), "{e}");
    }
}
