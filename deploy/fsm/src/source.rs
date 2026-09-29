//! Where the numbers in an observation come from, which is not what they are called.
//!
//! The same contract runs in three places -- mjlab, a browser, the robot -- and
//! the obvious way to serve all three is one trait with three implementations.
//! That is wrong, and one term already in the shipped jumper contract shows why.
//!
//! `actions` is one of them. In simulation it is the policy's own output from
//! the previous step; on the robot it is the same number, but the controller
//! has to remember it rather than read it, because nothing on the bus reports
//! what the policy asked for. Two consequences:
//!
//! `joint_torque` used to be the example here, and the example was wrong: this
//! module said "on the robot the servo does not report torque at all, so the
//! controller reconstructs it". The servo reports it, `MotorControl_State`
//! carries it and `dds.rs::take_state` reads it into `state.tau`. The claim was
//! inherited from the C++ controller and cost 2.146 N*m of divergence in every
//! reference replay until it was checked. The reconstruction still exists, in
//! `applied_torque`, for a host whose servos genuinely do not report.
//!
//! * "observation -> inference -> execution" is not a chain. It has a feedback
//!   edge, and the edge closes in a different place per environment.
//! * A term's name does not tell you what it is. Two runs can agree on every
//!   dimension, scale and index and still differ in what the numbers mean.
//!
//! So a source says what it can supply **and how**, and that is checked against
//! the contract when the policy is loaded rather than discovered at tick 40 000.
//!
//! ## The output is a list, not a verdict
//!
//! `check` returns the terms whose provenance differs from the one they had in
//! training. That list is the sim2real gap, enumerated -- the thing nobody can
//! currently see. A source that is missing a term outright is a hard error,
//! because there is no number to hand the policy; a source that supplies it
//! differently is a divergence, because there is one and it is not the same.
//!
//! `types.rs` already had half of this, in the one place it could not be
//! avoided: `ImuSample::has_lin_vel`, with the note that base linear velocity
//! "needs a state estimator, not an IMU". This is that idea, generalised.

use std::collections::BTreeMap;
use std::fmt;

/// How a value came to be.
///
/// Ordered from most to least authoritative, and that order is the point: a term
/// that was `Simulated` in training and is `Estimated` here has moved further
/// than one that has become `Measured`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Provenance {
    /// The physics engine computed it. Ground truth, and what every term in a
    /// policy trained in mjlab was.
    Simulated,
    /// A sensor read it. Carries that sensor's noise and its latency.
    Measured,
    /// This controller computed it from what it commanded, because nothing
    /// reports it. The feedback edge.
    Reconstructed,
    /// A state estimator produced it from other measurements.
    Estimated,
}

impl fmt::Display for Provenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Provenance::Simulated => "simulated",
            Provenance::Measured => "measured",
            Provenance::Reconstructed => "reconstructed",
            Provenance::Estimated => "estimated",
        })
    }
}

/// What one environment can put into a `RobotState`, and how it got it.
#[derive(Debug, Clone)]
pub struct SourceCapability {
    /// For the report. `mjlab`, `mujoco-wasm`, `device`.
    pub name: String,
    pub terms: BTreeMap<String, Provenance>,
}

impl SourceCapability {
    pub fn new(name: &str, terms: &[(&str, Provenance)]) -> Self {
        Self {
            name: name.to_string(),
            terms: terms.iter().map(|(t, p)| ((*t).to_string(), *p)).collect(),
        }
    }

    /// Everything, from the solver. What a policy trained in mjlab saw, and so
    /// also the default assumption about a contract that does not say.
    pub fn mjlab() -> Self {
        Self::new("mjlab", &Self::all(Provenance::Simulated))
    }

    /// MuJoCo compiled to WebAssembly: the same solver, so the same provenance.
    ///
    /// The difference from mjlab is not what can be read but what has to be done
    /// with it -- `cvel` is a spatial velocity about the subtree COM and has to
    /// be moved to the link origin -- and that is arithmetic, not provenance.
    pub fn mujoco_wasm() -> Self {
        Self::new("mujoco-wasm", &Self::all(Provenance::Simulated))
    }

    /// The robot.
    ///
    /// Three of these are not what the same name means in simulation, and each
    /// is a place a policy meets something it was not trained on:
    ///
    /// * `joint_torque` **is** measured. The servo reports it and
    ///   `MotorControl_State` carries it -- `dds.rs::take_state` has been
    ///   filling `state.tau` from it all along. This said "reconstructed from
    ///   the commanded PD, because the servo does not report torque", which was
    ///   inherited from the C++ controller (`action.cpp:100-103`) and was
    ///   wrong about this robot. Declaring it reconstructed made `control.rs`
    ///   ignore the real feedback and re-derive a number it already had, which
    ///   is the one term that showed up in every reference replay: 2.146 N*m
    ///   worst case against the simulator's `actuator_force`.
    ///
    ///   UNMEASURED: that the servo's `tau` agrees with the simulator's
    ///   `actuator_force` in **unit and sign**. Both are nominally N*m at the
    ///   joint, and nothing here has checked it against hardware. A sign flip
    ///   would feed the policy the negative of a term it trained on, at the
    ///   right magnitude, and the robot would simply behave differently -- put
    ///   a known load on one joint and read both.
    /// * `base_lin_vel` needs a state estimator; without one it is absent, and a
    ///   contract asking for it is refused rather than fed zeros.
    /// * `gait_phase` is counted from ticks rather than read from an
    ///   environment, so it is this controller's own construction.
    /// * `clip_phase`, `ref_joint_pos` and `ref_joint_vel` are read off the
    ///   recording on this controller's own clock, like the two below.
    /// * `jump_phase` and `ref_future` are the same kind of thing for a
    ///   reference-guided policy: read off the recording the bundle carries,
    ///   on this controller's own motion clock rather than the environment's.
    ///   The table is the one training used, byte for byte, and the formulas
    ///   are transcribed from `mdp/reference.py` -- so the divergence is the
    ///   clock and nothing else, which is exactly what `gait_phase` already
    ///   says about itself.
    pub fn device(has_velocity_estimator: bool) -> Self {
        let mut terms: BTreeMap<String, Provenance> = [
            ("base_ang_vel", Provenance::Measured),
            ("projected_gravity", Provenance::Measured),
            ("base_pose", Provenance::Measured),
            ("joint_pos", Provenance::Measured),
            ("joint_vel", Provenance::Measured),
            ("joint_torque", Provenance::Measured),
            ("actions", Provenance::Reconstructed),
            ("commands", Provenance::Measured),
            ("commands_diff", Provenance::Measured),
            // The operator's, off the pad, like `commands`.
            ("posture_command", Provenance::Measured),
            ("gait_phase", Provenance::Reconstructed),
            ("jump_phase", Provenance::Reconstructed),
            ("ref_future", Provenance::Reconstructed),
            ("clip_phase", Provenance::Reconstructed),
            ("ref_joint_pos", Provenance::Reconstructed),
            ("ref_joint_vel", Provenance::Reconstructed),
            // Measured, unlike the four above, and the odd one out for a
            // reason worth naming: the half of it that differs between hosts is
            // the robot's **own** orientation, which comes from the IMU here
            // and from the solver in training. The recording's half is the same
            // table either way. It is `projected_gravity` with a recorded
            // constant subtracted, so it is what `projected_gravity` is.
            ("ref_tilt_error", Provenance::Measured),
        ]
        .iter()
        .map(|(t, p)| ((*t).to_string(), *p))
        .collect();
        if has_velocity_estimator {
            terms.insert("base_lin_vel".into(), Provenance::Estimated);
        }
        Self { name: "device".into(), terms }
    }

    fn all(p: Provenance) -> Vec<(&'static str, Provenance)> {
        crate::obs::supported_terms().iter().map(|t| (*t, p)).collect()
    }
}

/// A term this source supplies differently than training did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Divergence {
    pub term: String,
    pub trained: Provenance,
    pub here: Provenance,
}

impl fmt::Display for Divergence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: trained {}, here {}", self.term, self.trained, self.here)
    }
}

#[derive(Debug)]
pub enum Error {
    /// The policy needs a number this environment has none of. There is nothing
    /// sensible to substitute: zeros are a value the policy never saw, and the
    /// robot would walk on them.
    Missing { term: String, source: String, needed_as: Provenance },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Missing { term, source, needed_as } => write!(
                f,
                "this policy observes '{term}' ({needed_as} in training) and '{source}' \
                 cannot supply it. Feeding zeros would be a value the policy never saw, \
                 so the model is refused rather than run on one"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Hold a contract's observation terms against what an environment can supply.
///
/// `trained` is what each term was when the policy was trained; a contract that
/// does not record it is assumed `Simulated`, which is what every export from
/// kk-rl-mjlab is and what the assumption therefore costs nothing to make. It is
/// still worth recording, because the assumption stops being free the first time
/// a policy is trained on replayed hardware data.
pub fn check(
    terms: &[String],
    trained: &BTreeMap<String, Provenance>,
    source: &SourceCapability,
) -> Result<Vec<Divergence>, Error> {
    let mut divergences = Vec::new();
    for term in terms {
        let was = trained.get(term).copied().unwrap_or(Provenance::Simulated);
        match source.terms.get(term) {
            None => {
                return Err(Error::Missing {
                    term: term.clone(),
                    source: source.name.clone(),
                    needed_as: was,
                })
            }
            Some(&here) if here != was => {
                divergences.push(Divergence { term: term.clone(), trained: was, here })
            }
            Some(_) => {}
        }
    }
    Ok(divergences)
}

/// The divergence list as one block of text, for a log line at load time.
///
/// Printed even when empty. "No divergences" is information -- it says the check
/// ran -- and a silent success looks the same as a check nobody wired up.
pub fn report(policy: &str, source: &SourceCapability, divergences: &[Divergence]) -> String {
    if divergences.is_empty() {
        return format!("{policy} on '{}': every term as trained", source.name);
    }
    let mut out = format!(
        "{policy} on '{}': {} term(s) differ from training\n",
        source.name,
        divergences.len()
    );
    for d in divergences {
        out.push_str(&format!("  {d}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    /// The jumper contract as shipped. Every term is available on the robot, so it
    /// is deployable -- but not identically, and that is the whole point.
    #[test]
    fn the_shipped_contract_runs_on_the_robot_but_not_unchanged() {
        let t = terms(&[
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "actions",
            "commands",
            "joint_torque",
            "gait_phase",
        ]);
        let trained = BTreeMap::new(); // mjlab: everything Simulated

        let sim = check(&t, &trained, &SourceCapability::mjlab()).unwrap();
        assert!(sim.is_empty(), "a policy replayed where it was trained diverges nowhere");

        let web = check(&t, &trained, &SourceCapability::mujoco_wasm()).unwrap();
        assert!(web.is_empty(), "the browser runs the same solver");

        let robot = check(&t, &trained, &SourceCapability::device(false)).unwrap();
        let moved: Vec<&str> = robot.iter().map(|d| d.term.as_str()).collect();
        assert_eq!(
            moved,
            ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
             "actions", "commands", "joint_torque", "gait_phase"],
            "on the robot nothing is simulated any more"
        );
        // `joint_torque` is a sensor like the rest: the servo reports it and
        // `dds.rs` reads it. This asserted `Reconstructed` until the claim that
        // the servo reports no torque -- inherited from the C++ -- was found to
        // be wrong about this robot.
        let torque = robot.iter().find(|d| d.term == "joint_torque").unwrap();
        assert_eq!(torque.here, Provenance::Measured);

        // `actions` is the one that is not a sensor: it is the controller's own
        // output, fed back in on the next tick.
        let actions = robot.iter().find(|d| d.term == "actions").unwrap();
        assert_eq!(actions.here, Provenance::Reconstructed);
    }

    /// A term the robot has no source for stops the load. Zeros would be a value
    /// the policy never saw, and it would walk on them.
    #[test]
    fn a_term_the_robot_cannot_supply_is_refused() {
        let t = terms(&["base_lin_vel", "joint_pos"]);
        let trained = BTreeMap::new();

        let err = check(&t, &trained, &SourceCapability::device(false)).unwrap_err();
        let Error::Missing { term, source, .. } = &err;
        assert_eq!(term, "base_lin_vel");
        assert_eq!(source, "device");
        assert!(err.to_string().contains("never saw"));

        // With an estimator it loads, and says so rather than passing silently.
        // Both terms diverge -- on the robot nothing is simulated -- but they are
        // not the same kind of divergence, which is why the list carries how.
        let with = check(&t, &trained, &SourceCapability::device(true)).unwrap();
        let how: Vec<(&str, Provenance)> =
            with.iter().map(|d| (d.term.as_str(), d.here)).collect();
        assert_eq!(
            how,
            [("base_lin_vel", Provenance::Estimated), ("joint_pos", Provenance::Measured)]
        );
    }

    /// The control group. Everything above passes against a `check` that ignores
    /// `trained` entirely and compares against a hardcoded `Simulated` -- which
    /// is true of every contract shipped today and will stop being true the first
    /// time a policy is trained on replayed hardware data.
    #[test]
    fn the_check_reads_what_training_recorded_rather_than_assuming_it() {
        // `actions` rather than `joint_torque`: the robot measures its torque
        // now, and this needs a term it genuinely reconstructs on both sides.
        let t = terms(&["actions"]);
        let mut trained = BTreeMap::new();
        trained.insert("actions".to_string(), Provenance::Reconstructed);

        // Trained against a reconstructed term, running on the robot that also
        // reconstructs it: nothing has moved.
        let robot = check(&t, &trained, &SourceCapability::device(false)).unwrap();
        assert!(robot.is_empty(), "it is only a divergence if it differs from training");

        // ...and the same contract in simulation now *is* a divergence, the other
        // way round. Assuming Simulated would report this backwards.
        let sim = check(&t, &trained, &SourceCapability::mjlab()).unwrap();
        assert_eq!(sim.len(), 1);
        assert_eq!(sim[0].trained, Provenance::Reconstructed);
        assert_eq!(sim[0].here, Provenance::Simulated);
    }

    #[test]
    fn the_report_says_so_even_when_there_is_nothing_to_say() {
        let empty = report("locomotion", &SourceCapability::mjlab(), &[]);
        assert!(empty.contains("every term as trained"));
        let some = report(
            "locomotion",
            &SourceCapability::device(false),
            &[Divergence {
                term: "joint_torque".into(),
                trained: Provenance::Simulated,
                here: Provenance::Reconstructed,
            }],
        );
        assert!(some.contains("joint_torque: trained simulated, here reconstructed"));
    }
}
