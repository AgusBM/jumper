//! Run a bundle on the robot.
//!
//! ```text
//! controller --bundle /opt/mjrl/current --machine /etc/mjrl/machine.toml
//! controller --bundle /opt/mjrl/current --dry-run           # open and report, touch no bus
//! controller --bundle /opt/mjrl/current --check-reference   # ...and replay its vectors
//! controller --vocabulary                                  # the words a manifest may use
//! ```
//!
//! Everything the policy knows about itself comes out of the bundle; everything
//! about *this machine* comes out of `--machine`. The split is the point: a
//! number that a simulator could have measured belongs in the contract, and
//! `machine.toml` is what is left after that -- the bus, and the handful of
//! constants that describe the actuators rather than the policy.
//!
//! This is the whole device host. There used to be a C++ controller that linked
//! this crate as a library and supplied its own loop; there is not one now, and
//! a loop is not the kind of thing worth having two of.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use mjrl_fsm::bundle::{Bundle, Target, Tuning};
use mjrl_fsm::dds::DdsIo;
use mjrl_fsm::device::Outcome;

/// The control loop's period. The controller decides which ticks infer -- see
/// `DESIGN.md`, "Rate lives in the core" -- so this is only how often the bus is
/// drained and the motors are written.
const TICK_US: u64 = 1_000;

/// How long to hold the safe state on the way out.
///
/// Leaving the loop is not the same as leaving the robot in a good place: the
/// motor controller holds the last target it received. So the shutdown path
/// drives the FSM into its safe state the way a stale bus would, and publishes
/// that for long enough to be received.
const SHUTDOWN_TICKS: u64 = 50;

/// How far this host's controller may differ from the reference.
///
/// Not a tolerance so much as a floor on noise: the observation and the joint
/// targets are the same f32 arithmetic in the same order, so the honest
/// expectation is bit-exact and anything above this is a real difference.
/// Deliberately far below the ~1e-3 a quantised `.rknn` shows in the *action*,
/// which is judged separately and is not this number.
const CONTROLLER_TOL: f32 = 1e-6;

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" {
    fn signal(sig: i32, handler: usize) -> usize;
}

extern "C" fn on_signal(_sig: i32) {
    STOP.store(true, Ordering::SeqCst);
}

fn fail(msg: impl std::fmt::Display) -> ! {
    eprintln!("[controller] {msg}");
    std::process::exit(1)
}

/// What this machine is, as opposed to what the policy is.
struct Machine {
    motor_domain: u32,
    robot_domain: u32,
    qos_xml: PathBuf,
    tuning: Tuning,
}

impl Default for Machine {
    /// Both domains **0**, which is what the board is configured with today --
    /// `deploy/README.md` records reading it off the machine. Not a guess and
    /// not a convention: an invented default here is a participant that pairs
    /// with nothing, and DDS does not report that, it just leaves the topic
    /// empty forever.
    ///
    /// They are printed on every start for the same reason. mbus master has
    /// since moved `robot_control` to domain 2 for the wireless link; both
    /// sides still agree on 0, and they will not stay in agreement by
    /// themselves.
    fn default() -> Self {
        Self {
            motor_domain: 0,
            robot_domain: 0,
            qos_xml: PathBuf::from("/etc/mjrl/qos.xml"),
            tuning: Tuning::default(),
        }
    }
}

fn read_machine(path: &Path) -> Machine {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| fail(format!("cannot read {}: {e}", path.display())));
    let t: toml::Value = text
        .parse()
        .unwrap_or_else(|e| fail(format!("{}: {e}", path.display())));
    let mut m = Machine::default();
    let f = |key: &str| t.get(key).and_then(|v| v.as_float()).map(|v| v as f32);
    let i = |key: &str| t.get(key).and_then(|v| v.as_integer());

    if let Some(v) = i("motor_domain") {
        m.motor_domain = v as u32;
    }
    if let Some(v) = i("robot_domain") {
        m.robot_domain = v as u32;
    }
    if let Some(v) = t.get("qos_xml").and_then(|v| v.as_str()) {
        m.qos_xml = PathBuf::from(v);
    }
    if let Some(v) = f("joint_pos_rate_limit") {
        m.tuning.joint_pos_rate_limit = v;
    }
    if let Some(v) = f("obs_clip") {
        m.tuning.obs_clip = v;
    }
    if let Some(v) = f("action_clip") {
        m.tuning.action_clip = v;
    }
    if let Some(v) = f("action_smoothing") {
        m.tuning.action_smoothing = v;
    }
    if let Some(v) = t.get("has_velocity_estimator").and_then(|v| v.as_bool()) {
        m.tuning.has_velocity_estimator = v;
    }
    m
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut bundle_dir, mut machine_path, mut dry_run) = (None, None, false);
    let mut check_reference = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bundle" => bundle_dir = args.next(),
            "--machine" => machine_path = args.next(),
            // No bundle needed: this is the list itself, printed so that what
            // you read before writing a manifest is what will be accepted.
            "--vocabulary" => {
                print!("{}", mjrl_fsm::vocabulary::describe());
                return;
            }
            "--dry-run" => dry_run = true,
            // Implies --dry-run: it drives the controller through recorded
            // frames, which is not something to do while a bus is attached.
            "--check-reference" => {
                check_reference = true;
                dry_run = true;
            }
            "-h" | "--help" => {
                println!("usage: controller --bundle <dir> [--machine <toml>] \
                          [--dry-run] [--check-reference]\n\
                          \x20      controller --vocabulary");
                return;
            }
            other => fail(format!("unknown argument {other}")),
        }
    }
    let Some(bundle_dir) = bundle_dir else {
        fail("--bundle <dir> is required")
    };
    let machine = machine_path.as_deref().map(Path::new).map(read_machine).unwrap_or_default();

    // As the board: its `.rknn` models, and the joint stops it has no model of
    // the robot to read them off. The bundle carries every host; this binary
    // is the board's.
    let bundle = Bundle::open(&bundle_dir, Target::Board).unwrap_or_else(|e| fail(e));
    let reference = bundle.reference().unwrap_or_else(|e| fail(e));
    println!("[controller] {} -- {} joints, {} mode(s)",
             bundle_dir, bundle.wire.len(), bundle.models.len());
    println!("[controller] DDS domains: motor {} robot {}, QoS from {}",
             machine.motor_domain, machine.robot_domain, machine.qos_xml.display());
    for note in &bundle.notes {
        println!("[controller] note: {note}");
    }
    let joints = bundle.wire.len();
    // A line a mode: each reads the pad through its own contract.
    let operators = bundle.operators().unwrap_or_else(|e| fail(e));
    for (mode, op) in operators.iter() {
        println!("[controller] {mode}: controls from its contract: {}", op.describe());
    }
    if operators.is_empty() {
        // Said out loud rather than left as a robot that ignores the sticks.
        println!("[controller] no policy here carries a `controller` block, so the gamepad \
                  drives nothing. Re-export one to bind the sticks.");
    }

    let mut host = bundle.into_host(machine.tuning, 0).unwrap_or_else(|e| fail(e));
    // Loud, and every time. A stub holds the home pose, which from the outside
    // is a robot that is behaving itself.
    //
    // Two outcomes, and the line has to say which: an ordinary policy on a stub
    // holds home, and a reference-residual one is barred from running at all,
    // because there a stub's zeros would play the recording open loop. Printing
    // one sentence for both used to read "will hold the home pose: ... it will
    // not run at all", which contradicts itself in the one message an operator
    // reads before enabling the motors.
    let barred: Vec<(String, String)> = host
        .stub_modes()
        .map(|(m, w)| (m.clone(), w.clone()))
        .collect();
    for (mode, why) in barred {
        let verdict = if host.controller().is_interlocked(&mode) {
            "will NOT run"
        } else {
            "will hold the home pose"
        };
        println!("[controller] STUB  '{mode}' {verdict}: {why}");
    }
    if check_reference {
        let Some(reference) = reference else {
            fail("this bundle carries no reference.json, so there is nothing to check \
                  against. Rebuild it on a machine with onnxruntime.")
        };
        let report = host.check_reference(&reference).unwrap_or_else(|e| fail(e));
        print!("[controller] {report}");
        // The controller must be exact; the model must not be. Two verdicts,
        // because one number covering both would have to be loose enough for
        // the quantised model and would then never catch the controller.
        if !report.controller_agrees(CONTROLLER_TOL) {
            fail(format!("this host's controller does not match the reference \
                          (tolerance {CONTROLLER_TOL:.0e})"));
        }
        println!("[controller] the controller matches. \
                  The action difference above is this machine's inference backend.");
        return;
    }
    if dry_run {
        println!("[controller] --dry-run: the bus was not opened");
        return;
    }

    let mut io = DdsIo::open(
        machine.motor_domain,
        machine.robot_domain,
        &machine.qos_xml,
        joints,
    )
    .unwrap_or_else(|e| fail(format!("cannot open the bus: {e}")));

    // SAFETY: `on_signal` only stores into an atomic, which is what a handler
    // is allowed to do. SIGINT and SIGTERM.
    unsafe {
        signal(2, on_signal as *const () as usize);
        signal(15, on_signal as *const () as usize);
    }

    println!("[controller] running. SIGINT or SIGTERM stops it through the safe state.");
    let started = std::time::Instant::now();
    let mut tick: u64 = 0;
    let mut last_mode = String::new();
    while !STOP.load(Ordering::SeqCst) {
        let now = tick * TICK_US;
        match host.pump(&mut io, now, wall_ns()) {
            Ok(Outcome::Inferred { mode, stub }) if stub && mode != last_mode => {
                println!("[controller] '{mode}' is running on a stub");
                last_mode = mode;
            }
            Ok(Outcome::Inferred { mode, .. }) => last_mode = mode,
            Ok(Outcome::Held) => {}
            // Not fatal on its own: one bad cycle on a bus is a bus. A cycle
            // that cannot publish leaves the motor controller holding its last
            // target, which the loop corrects on the next one.
            Err(e) => eprintln!("[controller] tick {tick}: {e}"),
        }
        for t in host.take_log() {
            println!("[controller] {:>8.3}s  {} -> {}  (rule {}: {})",
                     t.at as f64 / 1e6, t.from, t.to, t.rule, t.why);
        }
        tick += 1;
        // Against the start rather than against the last tick, so a slow cycle
        // is absorbed instead of shifting every cycle after it.
        let target = std::time::Duration::from_micros((tick + 1) * TICK_US);
        if let Some(rest) = target.checked_sub(started.elapsed()) {
            std::thread::sleep(rest);
        }
    }

    // The bus is still fine; the FSM is told it is not. `feedback_stale` is the
    // first rule in every cascade, so this is the same path a dropped bus takes
    // -- one shutdown behaviour rather than a second one written here.
    println!("[controller] stopping through the safe state");
    let stale_at = (tick + 1) * TICK_US + host.state_timeout_us() + host.command_timeout_us();
    for i in 0..SHUTDOWN_TICKS {
        let now = stale_at + i * TICK_US;
        if let Err(e) = host.step(now) {
            eprintln!("[controller] shutdown tick {i}: {e}");
        }
        if let Err(e) = io.publish(host.command(), wall_ns()) {
            eprintln!("[controller] shutdown tick {i}: {e}");
        }
        std::thread::sleep(std::time::Duration::from_micros(TICK_US));
    }
    println!("[controller] stopped after {tick} ticks");
}

/// Unix-epoch nanoseconds. The motor firmware reads the wire timestamp as
/// absolute wall-clock time, so a monotonic reading would be misread as a 1970
/// epoch -- which is why `pump` takes both clocks and derives neither.
fn wall_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
