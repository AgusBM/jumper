//! Turning the network's output into joint targets, and holding a pose.
//!
//! The transform is the contract's, verbatim:
//!
//! ```text
//! target[joint] = default_joint_pos[joint] + action_scale * action[i]
//! ```
//!
//! A **reference-guided** policy replaces the first term with the recording's
//! command at this instant -- the action is a correction on a motion rather than
//! on a pose. Same scale, same clip, same limiter; only the baseline moves. The
//! baseline is supplied by the observation builder rather than recomputed here,
//! so the residual sits on the pose the policy was actually answering.
//!
//! paired joint by joint through `action_joint_order` -- **never by index**. On
//! this robot the action is 20 wide and the wire is 22, so an index pairing is
//! misaligned from the fifth joint on and nothing raises.
//!
//! Order matters and is not obvious. Clip, then scale onto the baseline, then
//! smooth, then rate-limit against the previous target, then clamp to the joint's
//! hard limits. Rate-limiting before smoothing would let the smoother reintroduce
//! a step the limiter had just removed; clamping first would let the rate limiter
//! track a target outside the limits and stick there.

use crate::types::{MotorCommand, RobotState};

#[derive(Debug, Clone)]
pub struct ActionConfig {
    /// From the layout's `action_scale`.
    pub scale: f32,
    /// Symmetric clamp on the raw network output, before scaling.
    pub clip: f32,
    /// Per-decode EMA on the target. 0 disables. Distinct from the output-stage
    /// filter in the control loop, which runs at the higher publish rate.
    pub smoothing: f32,
    /// From the layout's `control.effort_limit`.
    pub effort_limit: f32,
    /// action element -> wire index.
    pub action_wire_idx: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct RobotLimits {
    /// Home pose, wire order. Every joint, including ones the policy never drives.
    pub default_wire: Vec<f32>,
    pub joint_pos_lo: Vec<f32>,
    pub joint_pos_hi: Vec<f32>,
    /// Maximum change in a joint target per control tick, rad. 0 disables.
    pub joint_pos_rate_limit: f32,
}

pub struct ActionDecoder {
    act: ActionConfig,
    lim: RobotLimits,
    joints: usize,
}

impl ActionDecoder {
    pub fn new(act: ActionConfig, lim: RobotLimits) -> Self {
        let joints = lim.default_wire.len();
        Self { act, lim, joints }
    }

    /// Policy mode. `prev_target` and `applied_torque` are carried across ticks:
    /// the first is what the rate limiter measures against, the second is fed
    /// back into the next observation's `joint_torque` term.
    #[allow(clippy::too_many_arguments)]
    /// The stops every decoded target is clamped to, in wire order.
    ///
    /// Readable so a host can report what it is clamping to. On the robot that
    /// is worth saying out loud: the clamp is the last thing between a policy
    /// and the motors, and one set wide enough to be safe on any robot is one
    /// that never fires -- which is indistinguishable from a working one until
    /// a joint reaches its stop.
    pub fn limits(&self) -> (&[f32], &[f32]) {
        (&self.lim.joint_pos_lo, &self.lim.joint_pos_hi)
    }

    pub fn decode(
        &self,
        raw_action: &[f32],
        state: &RobotState,
        kp: f32,
        kd: f32,
        prev_target: &mut [f32],
        cmd: &mut MotorCommand,
        applied_torque: &mut [f32],
        baseline: Option<&[f32]>,
    ) {
        // Joints the policy does not drive stay at the home pose -- including
        // under a reference, where the recording does have a column for them.
        // The training env locks them at the home pose, so following the
        // recording here would put a pose the policy never saw underneath it.
        // They still take part in the robot's contact geometry, so leaving them
        // wherever they happen to be would change what the feet stand on.
        let mut target = self.lim.default_wire.clone();

        let n = self.act.action_wire_idx.len().min(raw_action.len().max(0));
        for i in 0..self.act.action_wire_idx.len() {
            let w = self.act.action_wire_idx[i];
            let a = if i < n { raw_action[i] } else { 0.0 };
            let a = a.clamp(-self.act.clip, self.act.clip);

            // The recording when there is one, the home pose otherwise. A
            // baseline too short to cover this joint falls back rather than
            // panicking, and the fallback is the pose, not zero.
            let base = baseline
                .and_then(|b| b.get(w).copied())
                .unwrap_or(self.lim.default_wire[w]);
            let mut t = base + self.act.scale * a;

            if self.act.smoothing > 0.0 {
                t = self.act.smoothing * prev_target[w] + (1.0 - self.act.smoothing) * t;
            }
            if self.lim.joint_pos_rate_limit > 0.0 {
                let r = self.lim.joint_pos_rate_limit;
                t = t.clamp(prev_target[w] - r, prev_target[w] + r);
            }
            target[w] = t;
        }

        for w in 0..self.joints {
            let t = target[w].clamp(self.lim.joint_pos_lo[w], self.lim.joint_pos_hi[w]);
            cmd.pos[w] = t;
            cmd.vel[w] = 0.0;
            cmd.kp[w] = kp;
            cmd.kd[w] = kd;
            cmd.tau[w] = 0.0;
            let tau = kp * (t - state.q[w]) - kd * state.qd[w];
            applied_torque[w] = tau.clamp(-self.act.effort_limit, self.act.effort_limit);
            prev_target[w] = t;
        }
    }

    /// Default mode: hold a fixed pose, typically the home pose.
    ///
    /// Not the same as damping around the *measured* pose -- that is a limp robot
    /// that stays wherever it was pushed. This drives to a known configuration,
    /// which is what "stand" has to mean if the next thing that happens is a
    /// switch into a policy trained around that configuration.
    ///
    /// Deliberately does not touch `prev_target`: it is not a policy step, and
    /// letting it seed the rate limiter would make the first policy tick measure
    /// its step against a pose the policy never produced.
    pub fn command_pose(
        &self,
        target_pose: &[f32],
        state: &RobotState,
        kp: f32,
        kd: f32,
        cmd: &mut MotorCommand,
        applied_torque: &mut [f32],
    ) {
        for w in 0..self.joints {
            // A pose that does not cover this joint means "stay where you are",
            // not "go to zero".
            let t = target_pose.get(w).copied().unwrap_or(state.q[w]);
            let t = t.clamp(self.lim.joint_pos_lo[w], self.lim.joint_pos_hi[w]);
            cmd.pos[w] = t;
            cmd.vel[w] = 0.0;
            cmd.kp[w] = kp;
            cmd.kd[w] = kd;
            cmd.tau[w] = 0.0;
            let tau = kp * (t - state.q[w]) - kd * state.qd[w];
            applied_torque[w] = tau.clamp(-self.act.effort_limit, self.act.effort_limit);
        }
    }

    /// The failure path: damp around wherever the robot currently is.
    ///
    /// Used when the state is stale, i.e. when `state.q` may be old. Driving to a
    /// remembered target on stale feedback is how a robot throws itself; damping
    /// only removes energy.
    pub fn hold(
        &self,
        state: &RobotState,
        kd: f32,
        prev_target: &mut [f32],
        cmd: &mut MotorCommand,
        applied_torque: &mut [f32],
    ) {
        for w in 0..self.joints {
            let q = state.q[w];
            cmd.pos[w] = q;
            cmd.vel[w] = 0.0;
            cmd.kp[w] = 0.0;
            cmd.kd[w] = kd;
            cmd.tau[w] = 0.0;
            // kp is zero, so this is -kd * qd: pure damping, by construction.
            let tau = -kd * state.qd[w];
            applied_torque[w] = tau.clamp(-self.act.effort_limit, self.act.effort_limit);
            prev_target[w] = q;
        }
    }
}

/// Gains the layout asks for, adjusted by the deployment's own trim.
///
/// The layout carries the gains the policy was *trained* with; the bench often
/// needs them scaled. Both are clamped non-negative -- a negative gain is a
/// sign flip on the PD loop, which is an actuator driving away from its target.
pub fn commanded_gains(contract_kp: f32, contract_kd: f32, trim: &GainTrim) -> (f32, f32) {
    (
        (contract_kp * trim.kp_scale + trim.kp_offset).max(0.0),
        (contract_kd * trim.kd_scale + trim.kd_offset).max(0.0),
    )
}

#[derive(Debug, Clone)]
pub struct GainTrim {
    pub kp_scale: f32,
    pub kd_scale: f32,
    pub kp_offset: f32,
    pub kd_offset: f32,
}

impl Default for GainTrim {
    fn default() -> Self {
        Self { kp_scale: 1.0, kd_scale: 1.0, kp_offset: 0.0, kd_offset: 0.0 }
    }
}

/// EMA coefficient for a first-order low-pass at `cutoff_hz`, sampled at `dt_s`.
///
/// `alpha = 1 - exp(-dt / tau)`, `tau = 1 / (2*pi*f_c)`. Written this way rather
/// than as a fixed coefficient because it is **rate-invariant**: the same
/// continuous filter whatever the publish rate, which is what lets the loop run
/// at 1 kHz while the policy steps at 50 Hz without changing the smoothing.
pub fn ema_alpha_from_cutoff(cutoff_hz: f64, dt_s: f64) -> f32 {
    if cutoff_hz <= 0.0 || dt_s <= 0.0 {
        return 1.0; // no filter: pure zero-order hold
    }
    let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz);
    (1.0 - (-dt_s / tau).exp()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoder(joints: usize, action_wire: Vec<usize>) -> ActionDecoder {
        ActionDecoder::new(
            ActionConfig {
                scale: 0.25,
                clip: 100.0,
                smoothing: 0.0,
                effort_limit: 2.0,
                action_wire_idx: action_wire,
            },
            RobotLimits {
                default_wire: vec![0.1; joints],
                joint_pos_lo: vec![-3.3; joints],
                joint_pos_hi: vec![3.3; joints],
                joint_pos_rate_limit: 0.0,
            },
        )
    }

    #[test]
    fn action_maps_through_action_wire_idx_not_by_index() {
        // Action element 0 drives wire joint 2. An index pairing would drive 0.
        let d = decoder(3, vec![2]);
        let s = RobotState::new(3);
        let (mut prev, mut cmd, mut tau) = (vec![0.1; 3], MotorCommand::new(3), vec![0.0; 3]);
        d.decode(&[1.0], &s, 10.0, 0.5, &mut prev, &mut cmd, &mut tau, None);
        assert_eq!(cmd.pos[2], 0.1 + 0.25, "wire 2 is the one the action drives");
        assert_eq!(cmd.pos[0], 0.1, "wire 0 stays at the home pose");
        assert_eq!(cmd.pos[1], 0.1);
    }

    #[test]
    fn undriven_joints_hold_the_home_pose_not_their_measurement() {
        let d = decoder(2, vec![0]);
        let mut s = RobotState::new(2);
        s.q = vec![0.0, 1.234]; // joint 1 has been pushed somewhere
        let (mut prev, mut cmd, mut tau) = (vec![0.1; 2], MotorCommand::new(2), vec![0.0; 2]);
        d.decode(&[0.0], &s, 10.0, 0.5, &mut prev, &mut cmd, &mut tau, None);
        assert_eq!(cmd.pos[1], 0.1, "held at home, not at 1.234");
    }

    #[test]
    fn rate_limit_measures_against_the_previous_target() {
        let mut d = decoder(1, vec![0]);
        d.lim.joint_pos_rate_limit = 0.05;
        let s = RobotState::new(1);
        let (mut prev, mut cmd, mut tau) = (vec![0.1], MotorCommand::new(1), vec![0.0]);
        // Asks for +0.25; only +0.05 is allowed this tick.
        d.decode(&[1.0], &s, 10.0, 0.5, &mut prev, &mut cmd, &mut tau, None);
        assert!((cmd.pos[0] - 0.15).abs() < 1e-6, "got {}", cmd.pos[0]);
        // Next tick moves another step, from the new previous target.
        d.decode(&[1.0], &s, 10.0, 0.5, &mut prev, &mut cmd, &mut tau, None);
        assert!((cmd.pos[0] - 0.20).abs() < 1e-6, "got {}", cmd.pos[0]);
    }

    #[test]
    fn hard_joint_limits_win_over_everything() {
        let mut d = decoder(1, vec![0]);
        d.lim.joint_pos_hi = vec![0.2];
        let s = RobotState::new(1);
        let (mut prev, mut cmd, mut tau) = (vec![0.1], MotorCommand::new(1), vec![0.0]);
        d.decode(&[100.0], &s, 10.0, 0.5, &mut prev, &mut cmd, &mut tau, None);
        assert_eq!(cmd.pos[0], 0.2);
    }

    #[test]
    fn applied_torque_is_clamped_to_the_effort_limit() {
        let d = decoder(1, vec![0]);
        let mut s = RobotState::new(1);
        s.q = vec![-3.0]; // enormous error -> enormous PD torque
        let (mut prev, mut cmd, mut tau) = (vec![0.1], MotorCommand::new(1), vec![0.0]);
        d.decode(&[0.0], &s, 10.0, 0.5, &mut prev, &mut cmd, &mut tau, None);
        assert_eq!(tau[0], 2.0, "clamped to effort_limit, not 31");
    }

    #[test]
    fn command_pose_drives_to_the_pose_and_hold_does_not() {
        let d = decoder(1, vec![0]);
        let mut s = RobotState::new(1);
        s.q = vec![0.9];
        let (mut cmd, mut tau) = (MotorCommand::new(1), vec![0.0]);
        d.command_pose(&[0.1], &s, 10.0, 0.5, &mut cmd, &mut tau);
        assert_eq!(cmd.pos[0], 0.1, "default mode drives to the home pose");
        assert!(cmd.kp[0] > 0.0);

        let mut prev = vec![0.0];
        d.hold(&s, 0.5, &mut prev, &mut cmd, &mut tau);
        assert_eq!(cmd.pos[0], 0.9, "hold stays where the robot is");
        assert_eq!(cmd.kp[0], 0.0, "and applies no position gain at all");
    }

    #[test]
    fn negative_gains_are_clamped_away() {
        let t = GainTrim { kp_scale: 1.0, kd_scale: 1.0, kp_offset: -50.0, kd_offset: 0.0 };
        assert_eq!(commanded_gains(10.0, 0.5, &t), (0.0, 0.5));
    }

    #[test]
    fn ema_alpha_is_rate_invariant() {
        // Same continuous filter, two sample rates: the per-tick alphas differ,
        // but applying them over one second must converge to the same place.
        let (f, secs) = (20.0, 1.0);
        let mut a = (0.0f32, 0.0f32);
        let (n1, n2) = (1000, 50);
        let (al1, al2) = (
            ema_alpha_from_cutoff(f, secs / n1 as f64),
            ema_alpha_from_cutoff(f, secs / n2 as f64),
        );
        for _ in 0..n1 {
            a.0 += al1 * (1.0 - a.0);
        }
        for _ in 0..n2 {
            a.1 += al2 * (1.0 - a.1);
        }
        assert!((a.0 - a.1).abs() < 1e-4, "1 kHz {} vs 50 Hz {}", a.0, a.1);
    }

    #[test]
    fn zero_cutoff_means_no_filter() {
        assert_eq!(ema_alpha_from_cutoff(0.0, 0.001), 1.0);
    }
}
