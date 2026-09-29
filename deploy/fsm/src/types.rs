//! The values that cross the DDS boundary, in this controller's own terms.
//!
//! Deliberately not the generated IDL structs: those are `#[repr(C)]` fixed-size
//! arrays sized by `MC_MAX_MOTOR_CNT`, and letting them leak upward would make
//! every module care how many motor slots the wire format happens to reserve.
//! `dds.rs` converts at the edge.
//!
//! **Everything indexed `[w]` here is wire order** -- the order of
//! `[robot] joint_names`, which is the order the motor controller uses. The
//! policy's own orders (`obs_joint_order`, `action_joint_order`) are separate and
//! are mapped by name in `layout.rs`. Confusing the two is the expensive mistake
//! on this path, so the two never share a type.

#[derive(Debug, Clone, Copy, Default)]
pub struct ImuSample {
    /// Body orientation as `(w, x, y, z)`.
    pub quat: [f32; 4],
    /// Body angular velocity, rad/s.
    pub gyro: [f32; 3],
    /// Body linear velocity. Almost never available: it needs a state estimator,
    /// not an IMU. `has_lin_vel` says whether it means anything.
    pub lin_vel: [f32; 3],
    pub has_lin_vel: bool,
    pub valid: bool,
}

#[derive(Debug, Clone)]
pub struct RobotState {
    /// Joint positions, wire order, rad.
    pub q: Vec<f32>,
    /// Joint velocities, wire order, rad/s.
    pub qd: Vec<f32>,
    /// Measured joint torques, wire order, N*m.
    pub tau: Vec<f32>,
    pub imu: ImuSample,
}

impl RobotState {
    pub fn new(joints: usize) -> Self {
        Self {
            q: vec![0.0; joints],
            qd: vec![0.0; joints],
            tau: vec![0.0; joints],
            imu: ImuSample::default(),
        }
    }
}

/// One frame of the gamepad, as `RobotControlRaw::ControlRaw` carries it.
///
/// The robot's own pad service normalises three different pads onto this, so
/// these names mean the same thing on a bench and on the robot. Indexed through
/// `controller/vocabulary.json` so the vocabulary is written
/// down once and everything that names a control goes through it.
///
/// Values arrive already conditioned and must not be conditioned again: the
/// sticks carry a 0.15 deadband that is **rescaled**, so full deflection still
/// reaches ±1, and the triggers are unipolar `[0, 1]`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Pad {
    pub connected: bool,
    /// In the dictionary's button order.
    pub buttons: [bool; 10],
    /// -1 / 0 / +1. `dpad_y` is negated by the service, so +1 is up -- `Ly` is
    /// not, two lines away in the same function.
    pub dpad_x: i8,
    pub dpad_y: i8,
    /// In the dictionary's axis order.
    pub axes: [f32; 6],
}

impl Pad {
    pub fn button(&self, name: &str) -> bool {
        crate::vocabulary::pad_buttons()
            .iter()
            .position(|b| *b == name)
            .is_some_and(|i| self.buttons[i])
    }

    pub fn axis(&self, name: &str) -> f32 {
        crate::vocabulary::pad_axes()
            .iter()
            .position(|a| *a == name)
            .map_or(0.0, |i| self.axes[i])
    }

    /// One d-pad direction going down or coming up, by the dictionary's name,
    /// for a host that hears the pad a button at a time -- a browser's Gamepad
    /// API reports the four as buttons. Up is +1, as the service reports it.
    /// Let go, a direction clears only itself: the other end of the rocker,
    /// pressed since, stays down. `false` for any other name.
    pub fn set_dpad(&mut self, name: &str, down: bool) -> bool {
        let (axis, end) = match name {
            "dpad_right" => (&mut self.dpad_x, 1),
            "dpad_left" => (&mut self.dpad_x, -1),
            "dpad_up" => (&mut self.dpad_y, 1),
            "dpad_down" => (&mut self.dpad_y, -1),
            _ => return false,
        };
        if down {
            *axis = end;
        } else if *axis == end {
            *axis = 0;
        }
        true
    }

    /// Whether a d-pad direction is down, by the dictionary's name. `false` for
    /// any other name, as `axis` is 0.0 for one.
    pub fn dpad(&self, name: &str) -> bool {
        match name {
            "dpad_right" => self.dpad_x > 0,
            "dpad_left" => self.dpad_x < 0,
            // +1 is up: the service negates `dpad_y`.
            "dpad_up" => self.dpad_y > 0,
            "dpad_down" => self.dpad_y < 0,
            _ => false,
        }
    }
}

/// The controls a task answers itself -- `controls.yaml`'s `task:` -- by the
/// names the task gave them, as the operator holds them this tick. Handed to a
/// mode's hook (`hook::ModeHook::before_observation`) and to nothing else.
///
/// **By name, not by pad control**, since 2026-09-29: the pad and the keyboard
/// are two paths to one control (`jumper.five_foot`'s `claw_left` is `LT` on
/// the pad and Space on the keys), so what a hook reads is the control, and
/// which device reached it is the operator's business. Each value is `0..1`:
/// an `amount` as far as it is held, a `press` 1 while it is.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TaskControls {
    /// Somebody is there to let go of a control. False after the operator went
    /// away, and while the device in use is a pad that is unplugged: a hook
    /// drops what a press was holding then, rather than reading the silence as
    /// a person letting go of it.
    pub connected: bool,
    values: std::collections::BTreeMap<String, f32>,
}

impl TaskControls {
    /// Nobody there and nothing held: what a host with no reading of a task's
    /// controls hands over. A constant, so a host can keep it in a `static`.
    pub const NONE: Self = Self { connected: false, values: std::collections::BTreeMap::new() };

    pub fn new(connected: bool, values: impl IntoIterator<Item = (String, f32)>) -> Self {
        Self { connected, values: values.into_iter().collect() }
    }

    /// One control's value, by the task's name for it: 0 for a name nothing
    /// holds or no task declares, as a control nobody touches reads.
    pub fn get(&self, name: &str) -> f32 {
        self.values.get(name).copied().unwrap_or(0.0)
    }

    /// Every control, with its value, by name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, f32)> {
        self.values.iter().map(|(k, v)| (k.as_str(), *v))
    }
}

/// What the operator is asking for, in the physical units the policy trained
/// on. The gamepad is turned into this by the contract's own `controller`
/// block -- see `operator::Operator`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Command {
    pub lin_vel_x: f32,
    pub lin_vel_y: f32,
    pub yaw_rate: f32,
    pub height: f32,
    pub base_pitch: f32,
    pub base_roll: f32,
    pub base_twist: f32,
    /// The operator's action pulse: non-zero for a few frames on a button press.
    /// A **rising edge** is one request, which is why the loop tracks the previous
    /// value rather than acting while it is high -- the pulse repeats for three
    /// frames and would otherwise fire three times.
    pub mode_trigger: f32,
    /// Which action was asked for (`RobotControl::ActionId`). Carried so a mode
    /// can be selective about which button it answers to; the two-mode FSM only
    /// needs the edge.
    pub action_id: u32,
}

impl Command {
    /// Set one channel by the name a contract gives it. `false` for a name no
    /// field carries, so a caller can refuse it rather than drop it.
    ///
    /// The names are the command terms' own axes -- `ang_vel_z`, not the
    /// builder's `yaw_rate`, and `twist` / `pitch` / `roll` / `height` for
    /// `jumper.posture` -- plus `yaw_rate`, which is what `command_terms` has
    /// always called the third velocity channel.
    pub fn set_axis(&mut self, name: &str, value: f32) -> bool {
        match name {
            "lin_vel_x" => self.lin_vel_x = value,
            "lin_vel_y" => self.lin_vel_y = value,
            "ang_vel_z" | "yaw_rate" => self.yaw_rate = value,
            "twist" => self.base_twist = value,
            "pitch" => self.base_pitch = value,
            "roll" => self.base_roll = value,
            "height" => self.height = value,
            _ => return false,
        }
        true
    }

    /// Read one channel by name. `None` for a name no field carries.
    pub fn axis(&self, name: &str) -> Option<f32> {
        Some(match name {
            "lin_vel_x" => self.lin_vel_x,
            "lin_vel_y" => self.lin_vel_y,
            "ang_vel_z" | "yaw_rate" => self.yaw_rate,
            "twist" => self.base_twist,
            "pitch" => self.base_pitch,
            "roll" => self.base_roll,
            "height" => self.height,
            _ => return None,
        })
    }
}

/// One MIT-impedance command per joint, wire order. The motor controller closes
/// the torque loop, so the gains travel with the target rather than being applied
/// here.
#[derive(Debug, Clone)]
pub struct MotorCommand {
    pub pos: Vec<f32>,
    pub vel: Vec<f32>,
    pub kp: Vec<f32>,
    pub kd: Vec<f32>,
    pub tau: Vec<f32>,
}

impl MotorCommand {
    pub fn new(joints: usize) -> Self {
        Self {
            pos: vec![0.0; joints],
            vel: vec![0.0; joints],
            kp: vec![0.0; joints],
            kd: vec![0.0; joints],
            tau: vec![0.0; joints],
        }
    }
    pub fn joints(&self) -> usize {
        self.pos.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each name moves its own axis to its own end and back, and a release
    /// clears only its own end: right let go after left went down leaves left
    /// down. A name that is no direction moves nothing -- the control group.
    #[test]
    fn a_direction_at_a_time_is_the_d_pad() {
        let mut pad = Pad::default();
        assert!(pad.set_dpad("dpad_up", true) && pad.set_dpad("dpad_right", true));
        assert_eq!((pad.dpad_x, pad.dpad_y), (1, 1));
        assert!(pad.dpad("dpad_up") && pad.dpad("dpad_right"));
        pad.set_dpad("dpad_left", true);
        pad.set_dpad("dpad_right", false);
        assert_eq!(pad.dpad_x, -1, "right let go took left with it");
        pad.set_dpad("dpad_left", false);
        pad.set_dpad("dpad_up", false);
        pad.set_dpad("dpad_down", true);
        assert_eq!((pad.dpad_x, pad.dpad_y), (0, -1));
        assert!(!pad.set_dpad("A", true));
        assert_eq!((pad.dpad_x, pad.dpad_y, pad.buttons), (0, -1, [false; 10]));
    }

    /// A name nothing holds reads 0, as an untouched control does, so a hook
    /// can read its controls without first asking whether each was sent. The
    /// named one is the control group: it reads what it was given.
    #[test]
    fn a_task_control_nobody_holds_reads_zero() {
        let c = TaskControls::new(true, [("claw_left".to_string(), 0.4)]);
        assert_eq!(c.get("claw_left"), 0.4);
        assert_eq!((c.get("claw_right"), c.get("")), (0.0, 0.0));
        assert!(!TaskControls::default().connected, "the default is nobody there");
    }
}
