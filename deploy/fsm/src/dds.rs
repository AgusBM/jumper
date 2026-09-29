//! The robot's bus: CycloneDDS, over the types in `../dds/idl/`.
//!
//! Three subscriptions and one publication. The operator's is the **pad
//! itself**, `RobotControlRaw::ControlRaw` from `control-pod-svc`, not
//! `control-agent`'s interpretation of it: the sticks are turned into a command
//! by the policy's own contract, where `play` and the browser get theirs, so
//! the three signs exist once instead of three times.
//! The IDL structs and their topic
//! descriptors are generated at build time by `idlc` and mirrored into Rust by
//! bindgen (see `build.rs`), so nothing here is hand-transcribed and the wire
//! layout cannot drift away from the robot's.
//!
//! **QoS is matched, not negotiated.** A reader whose reliability or durability
//! disagrees with the writer's is not an error in DDS -- the two simply never
//! pair, and the topic stays empty forever. The profiles come from
//! `../dds/config/mbus_qos.xml`, verbatim.
//!
//! Everything crossing this module is converted at the edge: the IDL types stay
//! inside, and the rest of the controller sees `RobotState` / `Command` /
//! `MotorCommand` in wire order.

#![allow(non_upper_case_globals, non_camel_case_types, dead_code)]

use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

use crate::types::{ImuSample, MotorCommand, Pad, RobotState};

mod idl {
    #![allow(non_snake_case, non_camel_case_types, non_upper_case_globals, dead_code)]
    include!(concat!(env!("OUT_DIR"), "/idl_bindings.rs"));
}

use idl::{
    dds_topic_descriptor_t, Imu_ImuOutput, Imu_ImuOutput_desc, MotorControl_Control,
    MotorControl_Control_desc, MotorControl_State, MotorControl_State_desc,
    RobotControlRaw_ControlRaw, RobotControlRaw_ControlRaw_desc,
};

extern "C" {
    fn dds_create_participant(domain: u32, qos: *const c_void, listener: *const c_void) -> c_int;
    fn dds_create_topic(
        participant: c_int,
        descriptor: *const dds_topic_descriptor_t,
        name: *const c_char,
        qos: *const c_void,
        listener: *const c_void,
    ) -> c_int;
    fn dds_create_writer(p: c_int, t: c_int, qos: *const c_void, l: *const c_void) -> c_int;
    fn dds_create_reader(p: c_int, t: c_int, qos: *const c_void, l: *const c_void) -> c_int;
    fn dds_write(writer: c_int, sample: *const c_void) -> c_int;
    fn dds_take(
        reader: c_int,
        bufs: *mut *mut c_void,
        infos: *mut c_void,
        buf_size: usize,
        max_samples: u32,
    ) -> c_int;
    fn dds_delete(entity: c_int) -> c_int;

    // QoS profiles are loaded from the XML rather than built in code, so what
    // this side asks for is literally the file the robot's side was configured
    // with. Available because CycloneDDS is built with DDS_HAS_QOS_PROVIDER.
    fn dds_create_qos_provider(path: *const c_char, provider: *mut *mut c_void) -> c_int;
    fn dds_qos_provider_get_qos(
        provider: *const c_void,
        kind: c_int,
        key: *const c_char,
        qos: *mut *const c_void,
    ) -> c_int;
    fn dds_delete_qos_provider(provider: *mut c_void);
}

/// `dds_qos_kind_t`, in the header's order:
/// PARTICIPANT, PUBLISHER, SUBSCRIBER, **TOPIC**, READER, WRITER.
///
/// Topic for all three entity kinds, deliberately. The READER and WRITER kinds
/// read a profile's `<datareader_qos>` / `<datawriter_qos>` blocks, and
/// `Imu_ImuOutput` has neither -- the provider then hands back an *empty* QoS
/// rather than an error, and an entity created with it silently gets CycloneDDS
/// defaults. `<topic_qos>` is populated for every profile here.
const DDS_TOPIC_QOS: c_int = 3;

/// `dds_sample_info` is 64 bytes on every platform CycloneDDS supports. Only its
/// presence matters here -- `dds_take` requires somewhere to write it, and this
/// module never reads it back.
const SAMPLE_INFO_SIZE: usize = 64;

#[derive(Debug)]
pub enum Error {
    Participant(c_int),
    Topic { name: &'static str, code: c_int },
    Endpoint { topic: &'static str, code: c_int },
    Publish(c_int),
    QosFile { path: String, code: c_int },
    Qos { key: &'static str, code: c_int },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Participant(c) => write!(
                f,
                "dds_create_participant failed ({c}); is CYCLONEDDS_URI pointing at a \
                 readable config, and is the domain id right?"
            ),
            Error::Topic { name, code } => write!(f, "dds_create_topic('{name}') failed ({code})"),
            Error::Endpoint { topic, code } => {
                write!(f, "creating an endpoint on '{topic}' failed ({code})")
            }
            Error::Publish(c) => write!(f, "dds_write failed ({c})"),
            Error::QosFile { path, code } => write!(
                f,
                "cannot load QoS from {path} ({code}); without it every endpoint \
                 would silently take CycloneDDS defaults instead of the robot's profiles"
            ),
            Error::Qos { key, code } => write!(
                f,
                "QoS profile 'mbus::{key}' not found in the XML ({code})"
            ),
        }
    }
}

impl std::error::Error for Error {}

pub struct DdsIo {
    /// Owns every `dds_qos_t` handed out below, so it outlives the entities.
    qos_provider: *mut c_void,
    participant_motor: c_int,
    participant_robot: c_int,
    reader_state: c_int,
    reader_imu: c_int,
    reader_command: c_int,
    writer_control: c_int,
    joints: usize,
    seq: u16,
}

impl DdsIo {
    /// Open the bus. `motor_domain` and `robot_domain` may differ; on this robot
    /// they do, which is why there are two participants.
    pub fn open(
        motor_domain: u32,
        robot_domain: u32,
        qos_xml: &std::path::Path,
        joints: usize,
    ) -> Result<Self, Error> {
        // QoS comes from the XML, not from code. It has to equal what the other
        // side offers or DDS never pairs the endpoints -- silently, leaving the
        // topic empty forever -- and `robot_control` is RELIABLE + KEEP_ALL,
        // which a default-constructed reader would quietly downgrade.
        let path = CString::new(qos_xml.as_os_str().to_string_lossy().as_ref())
            .map_err(|_| Error::Qos { key: "<path>", code: -1 })?;
        let mut provider: *mut c_void = std::ptr::null_mut();
        // SAFETY: path outlives the call; provider is written on success only.
        let ret = unsafe { dds_create_qos_provider(path.as_ptr(), &mut provider) };
        if ret != 0 || provider.is_null() {
            return Err(Error::QosFile { path: qos_xml.display().to_string(), code: ret });
        }
        // Fetch one profile, by the provider's `<library>::<profile>` key.
        let qos = |profile: &'static str| -> Result<*const c_void, Error> {
            let key = CString::new(format!("mbus::{profile}")).unwrap();
            let mut q: *const c_void = std::ptr::null();
            // SAFETY: the returned QoS is owned by the provider, which this
            // struct keeps alive.
            let r = unsafe {
                dds_qos_provider_get_qos(provider, DDS_TOPIC_QOS, key.as_ptr(), &mut q)
            };
            if r != 0 || q.is_null() {
                return Err(Error::Qos { key: profile, code: r });
            }
            Ok(q)
        };
        let q_state = qos("MotorControl_State")?;
        let q_control = qos("MotorControl_Control")?;
        let q_imu = qos("Imu_ImuOutput")?;
        let q_command = qos("RobotControlRaw_ControlRaw")?;

        // SAFETY: plain C calls; the QoS pointers are owned by the provider.
        unsafe {
            let pm = dds_create_participant(motor_domain, std::ptr::null(), std::ptr::null());
            if pm < 0 {
                return Err(Error::Participant(pm));
            }
            let pr = if robot_domain == motor_domain {
                pm
            } else {
                let p = dds_create_participant(robot_domain, std::ptr::null(), std::ptr::null());
                if p < 0 {
                    dds_delete(pm);
                    return Err(Error::Participant(p));
                }
                p
            };

            let mk_topic = |p: c_int, desc: &dds_topic_descriptor_t, name: &'static str,
                            q: *const c_void| {
                let c = CString::new(name).unwrap();
                let t = dds_create_topic(p, desc, c.as_ptr(), q, std::ptr::null());
                if t < 0 {
                    Err(Error::Topic { name, code: t })
                } else {
                    Ok(t)
                }
            };

            // Topic names come from ../dds/topics.json and must match exactly;
            // a typo is a topic that exists and is simply never written to.
            let t_state = mk_topic(pm, &MotorControl_State_desc, "mc/motor_state", q_state)?;
            let t_control = mk_topic(pm, &MotorControl_Control_desc, "mc/motor_control", q_control)?;
            let t_imu = mk_topic(pm, &Imu_ImuOutput_desc, "imu/output", q_imu)?;
            let t_command =
                mk_topic(pr, &RobotControlRaw_ControlRaw_desc, "RobotControlRaw_ControlRaw",
                         q_command)?;

            let rd = |p: c_int, t: c_int, topic: &'static str, q: *const c_void| {
                let r = dds_create_reader(p, t, q, std::ptr::null());
                if r < 0 {
                    Err(Error::Endpoint { topic, code: r })
                } else {
                    Ok(r)
                }
            };

            let reader_state = rd(pm, t_state, "mc/motor_state", q_state)?;
            let reader_imu = rd(pm, t_imu, "imu/output", q_imu)?;
            let reader_command = rd(pr, t_command, "RobotControlRaw_ControlRaw", q_command)?;
            let writer_control =
                dds_create_writer(pm, t_control, q_control, std::ptr::null());
            if writer_control < 0 {
                return Err(Error::Endpoint {
                    topic: "mc/motor_control",
                    code: writer_control,
                });
            }

            Ok(Self {
                qos_provider: provider,
                participant_motor: pm,
                participant_robot: pr,
                reader_state,
                reader_imu,
                reader_command,
                writer_control,
                joints,
                seq: 0,
            })
        }
    }

    /// Take one sample if the reader has one. Returns `None` when it does not --
    /// which is normal, and is why the caller tracks staleness by clock rather
    /// than by "did we get anything".
    fn take_one<T: Default>(reader: c_int) -> Option<T> {
        let mut sample = T::default();
        let mut ptr = &mut sample as *mut T as *mut c_void;
        let mut info = [0u8; SAMPLE_INFO_SIZE];
        // SAFETY: bufs points at one pointer to a T-sized allocation, and
        // max_samples is 1, so CycloneDDS writes at most that one sample.
        let n = unsafe {
            dds_take(reader, &mut ptr, info.as_mut_ptr() as *mut c_void, 1, 1)
        };
        (n > 0).then_some(sample)
    }

    /// Joint feedback, in wire order.
    pub fn take_state(&self, out: &mut RobotState) -> bool {
        let Some(s) = Self::take_one::<MotorControl_State>(self.reader_state) else {
            return false;
        };
        let n = (s.motor_count as usize).min(self.joints).min(s.motors.len());
        for w in 0..n {
            let m = &s.motors[w];
            out.q[w] = m.pos;
            out.qd[w] = m.dq;
            out.tau[w] = m.tau;
        }
        true
    }

    /// Orientation and rates. Only this topic's orientation is read: the motor
    /// controller's `ImuData` has had a quaternion since mbus `d9875988f065`,
    /// unread here, so without this `projected_gravity` degrades to "perfectly
    /// level" and does so silently.
    pub fn take_imu(&self, out: &mut ImuSample) -> bool {
        let Some(s) = Self::take_one::<Imu_ImuOutput>(self.reader_imu) else {
            return false;
        };
        out.quat = [s.q.w, s.q.x, s.q.y, s.q.z];
        out.gyro = [s.gyro.x, s.gyro.y, s.gyro.z];
        // `vel` is the estimator's, when there is one. Left unused: a policy that
        // needs base linear velocity is refused at export time.
        out.lin_vel = [s.vel.x, s.vel.y, s.vel.z];
        out.has_lin_vel = false;
        out.valid = true;
        true
    }

    /// One frame of the gamepad, exactly as the pad service publishes it.
    ///
    /// Nothing is scaled and nothing is deadbanded here. Both were done before
    /// this arrived -- the service applies 0.15 and **rescales**, so a second
    /// deadband would narrow what a person can ask for without being visible --
    /// and turning sticks into a command is the contract's job, not the bus's.
    /// See `operator::Operator`.
    pub fn take_pad(&self, out: &mut Pad) -> bool {
        let Some(c) = Self::take_one::<RobotControlRaw_ControlRaw>(self.reader_command) else {
            return false;
        };
        out.connected = c.connected;
        // In `layout::PAD_BUTTONS` order. Written out rather than looped,
        // because the wire is a struct of named fields and a loop would need a
        // table pairing them by index -- which is the pairing this repository
        // keeps finding scrambled.
        out.buttons = [c.A, c.B, c.X, c.Y, c.LB, c.RB, c.menu, c.home, c.L3, c.R3];
        out.dpad_x = c.dpad_x;
        out.dpad_y = c.dpad_y;
        // In `layout::PAD_AXES` order.
        out.axes = [c.Lx, c.Ly, c.Rx, c.Ry, c.LT, c.RT];
        true
    }

    /// Publish one MIT-impedance command per joint.
    pub fn publish(&mut self, cmd: &MotorCommand, timestamp_ns: u64) -> Result<(), Error> {
        let mut out = MotorControl_Control::default();
        out.timestamp_ns = timestamp_ns;
        self.seq = self.seq.wrapping_add(1);
        out.sequence_id = self.seq;
        let n = cmd.joints().min(out.motors.len());
        out.motor_count = n as u32;
        for w in 0..n {
            let m = &mut out.motors[w];
            m.pos = cmd.pos[w];
            m.kp = cmd.kp[w];
            m.dq = cmd.vel[w];
            m.kd = cmd.kd[w];
            m.tau = cmd.tau[w];
        }
        // SAFETY: `out` is a fully initialised sample of the writer's own type.
        let ret = unsafe { dds_write(self.writer_control, &out as *const _ as *const c_void) };
        if ret != 0 {
            return Err(Error::Publish(ret));
        }
        Ok(())
    }
}

impl Drop for DdsIo {
    fn drop(&mut self) {
        // Deleting the participant takes its readers, writers and topics with it.
        unsafe {
            dds_delete(self.participant_motor);
            if self.participant_robot != self.participant_motor {
                dds_delete(self.participant_robot);
            }
            // After the entities: they hold QoS the provider owns.
            dds_delete_qos_provider(self.qos_provider);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bindgen bakes `size_of` assertions into the generated file, so a layout
    /// drift is a compile error rather than a test failure. This checks the
    /// numbers that matter to *this* module: the motor arrays are wide enough for
    /// the robot, and the four descriptors actually linked.
    #[test]
    fn idl_types_are_wide_enough_for_the_robot() {
        let c = MotorControl_Control::default();
        let s = MotorControl_State::default();
        assert!(c.motors.len() >= 22, "MC_MAX_MOTOR_CNT is {}", c.motors.len());
        assert_eq!(c.motors.len(), s.motors.len());
    }

    #[test]
    fn descriptors_are_linked_and_describe_the_expected_types() {
        // Reading these proves the generated C actually linked; a missing symbol
        // would not have got this far.
        // SAFETY: the descriptors are static data in the generated object.
        let (a, b) = unsafe {
            (MotorControl_Control_desc.m_size, MotorControl_State_desc.m_size)
        };
        assert_eq!(a as usize, std::mem::size_of::<MotorControl_Control>());
        assert_eq!(b as usize, std::mem::size_of::<MotorControl_State>());
    }

    extern "C" {
        fn dds_qget_reliability(qos: *const c_void, kind: *mut c_int, mbt: *mut i64) -> bool;
        fn dds_qget_history(qos: *const c_void, kind: *mut c_int, depth: *mut i32) -> bool;
    }

    /// The QoS the endpoints are actually created with, read back from the
    /// vendored XML.
    ///
    /// Two things would otherwise fail silently. A wrong provider key returns an
    /// error this test would catch; worse, the READER and WRITER kinds return an
    /// *empty* QoS for a profile with no `<datareader_qos>` block, and an entity
    /// built from that gets CycloneDDS defaults with nothing logged. `robot_control`
    /// is the case that matters: it is RELIABLE + KEEP_ALL for weak-network
    /// operation, and a defaulted reader would quietly be BEST_EFFORT.
    #[test]
    fn qos_profiles_load_and_say_what_the_xml_says() {
        let xml = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../dds/config/mbus_qos.xml");
        let path = CString::new(xml.to_string_lossy().as_ref()).unwrap();
        let mut provider: *mut c_void = std::ptr::null_mut();
        // SAFETY: the path outlives the call.
        assert_eq!(unsafe { dds_create_qos_provider(path.as_ptr(), &mut provider) }, 0,
                   "cannot load {}", xml.display());

        let read = |profile: &str| -> (i32, i32, i32) {
            let key = CString::new(format!("mbus::{profile}")).unwrap();
            let mut q: *const c_void = std::ptr::null();
            // SAFETY: provider is live; q is owned by it.
            let r = unsafe {
                dds_qos_provider_get_qos(provider, DDS_TOPIC_QOS, key.as_ptr(), &mut q)
            };
            assert_eq!(r, 0, "profile mbus::{profile} not found");
            let (mut rk, mut mbt, mut hk, mut depth) = (-1, 0i64, -1, -1);
            // SAFETY: q is a valid dds_qos_t from the provider.
            unsafe {
                assert!(dds_qget_reliability(q, &mut rk, &mut mbt), "{profile}: no reliability");
                assert!(dds_qget_history(q, &mut hk, &mut depth), "{profile}: no history");
            }
            (rk, hk, depth)
        };

        // DDS_RELIABILITY_BEST_EFFORT = 0, RELIABLE = 1
        // DDS_HISTORY_KEEP_LAST = 0, KEEP_ALL = 1
        assert_eq!(read("RobotControl_Control"), (1, 1, 0),
                   "robot_control must be RELIABLE + KEEP_ALL");
        // Control group: if the reader above passed because everything reads as
        // RELIABLE, these would fail. The pad is one of them, and it is the one
        // this controller actually subscribes to -- its profile was missing
        // from mbus entirely when the type first appeared, which by the rule
        // above is a reader that never pairs and a topic that stays empty.
        for p in ["MotorControl_Control", "MotorControl_State", "Imu_ImuOutput",
                  "RobotControlRaw_ControlRaw"] {
            assert_eq!(read(p), (0, 0, 1), "{p} must be BEST_EFFORT + KEEP_LAST 1");
        }
        // SAFETY: no QoS from this provider is used past this point.
        unsafe { dds_delete_qos_provider(provider) };
    }
}
