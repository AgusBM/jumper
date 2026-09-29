//! Replay a bundle's reference vectors and report where this host disagrees.
//!
//! Three hosts run this crate: `play` imports it, a browser loads it as wasm,
//! the robot links it. That they agree is the design, and until this existed it
//! was also the whole of the evidence. They compile the same source with three
//! toolchains for two architectures and run the policy on three different
//! inference backends; "same source" survives all of that unharmed, and so
//! would a divergence.
//!
//! `scripts/deploy.py` records what the controller does frame by frame. This
//! replays it and returns numbers.
//!
//! Two comparisons, deliberately kept apart, because a wrong observation and a
//! quantised model produce the same symptom -- a robot that walks slightly
//! wrong:
//!
//! | | what it checks | expected |
//! |---|---|---|
//! | `observation` | the crate, on this platform | exact |
//! | `target` | the action decode, clamp and filter | exact |
//! | `action` | this host's inference backend | RKNN differs ~1e-3; ONNX should not differ |
//!
//! The third is not checked here: this module drives the controller and has no
//! model. A host that has one passes its own outputs to [`Report::record_action`]
//! and gets them held against the same frames.

use serde::Deserialize;

use crate::control::{Controller, Step, TickInput};
use crate::fsm::Micros;
use crate::types::{Command, RobotState};

/// Written by `scripts/deploy.py`.
pub const SCHEMA: &str = "kk-policy-reference/1";

#[derive(Debug, Deserialize)]
pub struct Frame {
    pub now_us: u64,
    pub q: Vec<f32>,
    pub qd: Vec<f32>,
    pub tau: Vec<f32>,
    /// `[w, x, y, z]`, unit length.
    pub quat: [f32; 4],
    pub gyro: [f32; 3],
    /// `[lin_vel_x, lin_vel_y, yaw_rate]`.
    pub cmd: [f32; 3],
    /// Every channel, by the controller block's names, when the command is
    /// more than a twist -- `jumper.posture`'s four posture channels ride
    /// here. Applied after `cmd`, so a channel named in both takes this value.
    /// Absent from references written before it, which command a twist only.
    #[serde(default)]
    pub axes: Option<std::collections::BTreeMap<String, f32>>,
    /// The state the FSM was in. Recorded rather than asserted -- a host whose
    /// cascade lands somewhere else has a difference worth naming, not a
    /// tolerance worth widening.
    pub mode: String,
    /// The mode that inferred on this frame, absent when the frame held.
    #[serde(default)]
    pub infer: Option<String>,
    #[serde(default)]
    pub obs: Option<Vec<f32>>,
    #[serde(default)]
    pub act: Option<Vec<f32>>,
    pub target: Vec<f32>,
}

/// Where one term sits in the observation.
#[derive(Debug, Deserialize)]
pub struct TermSpan {
    pub name: String,
    pub offset: usize,
    pub dim: usize,
}

#[derive(Debug, Deserialize)]
pub struct Reference {
    pub schema: String,
    pub joints: Vec<String>,
    /// Absent in the first references written, which reported an index and left
    /// the reader to work out the term.
    #[serde(default)]
    pub terms: Vec<TermSpan>,
    pub frames: Vec<Frame>,
}

/// The worst disagreement found, and where.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Worst {
    pub diff: f32,
    pub frame: usize,
    pub index: usize,
}

impl Worst {
    fn see(&mut self, diff: f32, frame: usize, index: usize) {
        if diff > self.diff {
            *self = Worst { diff, frame, index };
        }
    }
}

/// What a replay found.
#[derive(Debug, Clone, Default)]
pub struct Report {
    pub frames: usize,
    pub inferences: usize,
    /// Terms every host must build identically. Exact, or something is wrong.
    pub observation: Worst,
    /// The terms this host itself declares it sources differently -- measured
    /// in mjlab, reconstructed on the robot -- with how far apart they actually
    /// are. Not a failure: this is the sim2real gap, and the only number
    /// anybody has ever had for it.
    pub divergent: Vec<(String, Worst)>,
    pub target: Worst,
    /// Only populated when a host feeds its own model's output back in.
    pub action: Option<Worst>,
    /// Frames whose FSM state did not match. Named rather than tolerated.
    pub mode_mismatches: Vec<(usize, String, String)>,
}

impl Report {
    fn see_divergent(&mut self, term: &str, diff: f32, frame: usize, index: usize) {
        match self.divergent.iter_mut().find(|(t, _)| t == term) {
            Some((_, w)) => w.see(diff, frame, index),
            None => self.divergent.push((term.to_string(), Worst { diff, frame, index })),
        }
    }

    /// Hold one frame's action from this host's own backend against the
    /// recorded one. Separate from the replay because the replay has no model.
    pub fn record_action(&mut self, frame: usize, theirs: &[f32], ours: &[f32]) {
        let worst = self.action.get_or_insert_with(Worst::default);
        for (i, (a, b)) in theirs.iter().zip(ours).enumerate() {
            worst.see((a - b).abs(), frame, i);
        }
    }

    /// Whether the *controller* matched. The action is judged separately and
    /// with a different tolerance -- it is a different question.
    pub fn controller_agrees(&self, tol: f32) -> bool {
        self.observation.diff <= tol && self.target.diff <= tol && self.mode_mismatches.is_empty()
    }
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "{} frames, {} inferences\n  observation  worst {:.3e} at frame {} index {}\n  \
             targets      worst {:.3e} at frame {} index {}",
            self.frames,
            self.inferences,
            self.observation.diff,
            self.observation.frame,
            self.observation.index,
            self.target.diff,
            self.target.frame,
            self.target.index
        )?;
        if let Some(a) = &self.action {
            writeln!(f, "  action       worst {:.3e} at frame {} index {}",
                     a.diff, a.frame, a.index)?;
        }
        // Only the ones that actually differ. A term whose *source* differs
        // -- `source.rs` marks nearly everything on a robot as measured rather
        // than simulated -- usually still computes to the same number here,
        // because the replay feeds it the recorded input. Saying how many did
        // is the stronger statement: the exemption list is conservative, and
        // this is the measurement of by how much.
        let (differ, agreed): (Vec<_>, Vec<_>) =
            self.divergent.iter().partition(|(_, w)| w.diff > 0.0);
        for (term, w) in &differ {
            writeln!(f, "  {term:<12} worst {:.3e} at frame {} -- this host sources it \
                         differently from training", w.diff, w.frame)?;
        }
        if !agreed.is_empty() {
            writeln!(
                f,
                "  {} other term(s) this host sources differently agreed exactly anyway: {}",
                agreed.len(),
                agreed.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>().join(", ")
            )?;
        }
        for (i, want, got) in &self.mode_mismatches {
            writeln!(f, "  frame {i}: the cascade chose '{got}', the reference has '{want}'")?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Reference {
    /// The term covering observation index `i`, if the reference names them.
    fn term_at(&self, i: usize) -> Option<String> {
        self.terms
            .iter()
            .find(|t| i >= t.offset && i < t.offset + t.dim)
            .map(|t| t.name.clone())
    }

    pub fn parse(text: &str) -> Result<Self, Error> {
        let r: Reference =
            serde_json::from_str(text).map_err(|e| Error(format!("reference.json: {e}")))?;
        if r.schema != SCHEMA {
            return Err(Error(format!(
                "reference.json says schema '{}'; this reader knows '{SCHEMA}'",
                r.schema
            )));
        }
        if r.frames.is_empty() {
            return Err(Error("reference.json has no frames".into()));
        }
        Ok(r)
    }

    /// Drive `controller` through every frame and report the worst difference.
    ///
    /// The **recorded** action is fed back on each inference, not one this host
    /// computed. That is what separates the two questions: with the same action
    /// going in, a difference in the targets is the decode, the clamp or the
    /// filter -- never the model. A host that wants its model checked runs it
    /// as well and calls [`Report::record_action`].
    pub fn replay(&self, controller: &mut Controller) -> Result<Report, Error> {
        let joints = self.joints.len();
        let mut report = Report { frames: self.frames.len(), ..Report::default() };
        let mut state = RobotState::new(joints);
        let mut command = Command::default();

        for (i, frame) in self.frames.iter().enumerate() {
            if frame.q.len() != joints {
                return Err(Error(format!(
                    "frame {i}: {} joints, but the reference names {joints}",
                    frame.q.len()
                )));
            }
            state.q.copy_from_slice(&frame.q);
            state.qd.copy_from_slice(&frame.qd);
            state.tau.copy_from_slice(&frame.tau);
            state.imu.quat = frame.quat;
            state.imu.gyro = frame.gyro;
            command.lin_vel_x = frame.cmd[0];
            command.lin_vel_y = frame.cmd[1];
            command.yaw_rate = frame.cmd[2];
            for (name, value) in frame.axes.iter().flatten() {
                if !command.set_axis(name, *value) {
                    return Err(Error(format!(
                        "frame {i}: '{name}' is not a command channel this controller carries"
                    )));
                }
            }

            let now: Micros = frame.now_us;
            let input = TickInput {
                state: &state,
                command: &command,
                // The recorded command, as `play` built it: every mode is
                // handed it, as it was when the frame was recorded.
                operators: None,
                state_fresh: true,
                command_fresh: true,
                buttons: &Default::default(),
                gripper_active: false,
                // A recording carries no operator, so nothing squeezes.
                controls: &Default::default(),
            };
            let step = controller.tick(&input, now).map_err(|e| Error(format!("frame {i}: {e}")))?;
            let here = controller.state().name.clone();
            if here != frame.mode {
                report.mode_mismatches.push((i, frame.mode.clone(), here));
            }
            if let Step::Infer { mode } = &step {
                report.inferences += 1;
                if let Some(obs) = &frame.obs {
                    // Which terms this host sources differently from training.
                    // Asked of the controller rather than assumed, so a host
                    // that measures everything gets everything checked.
                    let differs: Vec<&str> = controller
                        .divergences(mode)
                        .iter()
                        .map(|d| d.term.as_str())
                        .collect();
                    for (j, (a, b)) in obs.iter().zip(controller.observation()).enumerate() {
                        let diff = (a - b).abs();
                        match self.term_at(j).filter(|t| differs.contains(&t.as_str())) {
                            Some(term) => report.see_divergent(&term, diff, i, j),
                            None => report.observation.see(diff, i, j),
                        }
                    }
                }
                let action = frame.act.as_deref().unwrap_or(&[]);
                controller.resume(action).map_err(|e| Error(format!("frame {i}: {e}")))?;
            }
            for (j, (a, b)) in frame.target.iter().zip(controller.command().pos.iter()).enumerate()
            {
                report.target.see((a - b).abs(), i, j);
            }
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reference_this_reader_does_not_know_is_refused() {
        // Guessing at fields is how a check comes to pass against a format it
        // is not reading.
        let e = Reference::parse(r#"{"schema": "kk-policy-reference/9", "joints": [],
                                     "frames": []}"#)
            .unwrap_err()
            .to_string();
        assert!(e.contains("this reader knows"), "{e}");

        let e = Reference::parse(&format!(
            r#"{{"schema": "{SCHEMA}", "joints": ["a"], "frames": []}}"#
        ))
        .unwrap_err()
        .to_string();
        assert!(e.contains("no frames"), "{e}");
    }

    /// An empty reference would make `controller_agrees` true, which is the one
    /// answer a check must never give for free.
    #[test]
    fn worst_is_the_worst_and_a_clean_report_is_not_a_default() {
        let mut r = Report::default();
        r.observation.see(1e-7, 0, 3);
        r.observation.see(4e-3, 5, 11);
        r.observation.see(2e-3, 6, 1);
        assert_eq!(r.observation, Worst { diff: 4e-3, frame: 5, index: 11 });
        assert!(!r.controller_agrees(1e-5));
        assert!(r.controller_agrees(1e-2));

        // A mode mismatch fails at any tolerance: landing in another state is
        // not a small number, it is a different robot.
        let mut r = Report::default();
        r.mode_mismatches.push((3, "walk".into(), "safe".into()));
        assert!(!r.controller_agrees(f32::INFINITY));
    }

    #[test]
    fn an_action_diff_is_tracked_apart_from_the_controller() {
        let mut r = Report::default();
        assert!(r.action.is_none(), "a host with no model reports no action diff");
        r.record_action(2, &[0.0, 1.0], &[0.0, 1.002]);
        assert_eq!(r.action.as_ref().unwrap().frame, 2);
        assert!((r.action.as_ref().unwrap().diff - 0.002).abs() < 1e-6);
        // ...and it does not leak into the controller's verdict.
        assert!(r.controller_agrees(0.0));
    }
}
