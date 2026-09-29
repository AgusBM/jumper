//! A task's own code, at fixed points of a mode's loop.
//!
//! This crate is the robot's whole program and knows no task by name. Some tasks
//! still need something done on the robot that no contract field describes --
//! `jumper.five_foot` is trained with the claw on the left and has to carry it
//! on the right too, by mirroring -- and the answer is not a flag for each of
//! them in here. It is a hook: a task implements [`ModeHook`] in its own
//! directory, and this crate calls it at the points below, the same points for
//! every task.
//!
//! ## The points
//!
//! One inference tick, in order, with the frame each quantity is in. "Robot" is
//! the machine as the bus reports it; "policy" is the machine as the policy was
//! trained to see it. A mode without a hook has one frame and every call below
//! is skipped.
//!
//! ```text
//!   robot state, operator command, task controls        robot frame
//!     -> before_observation(state, command, controls)   -> policy frame
//!   observation -> inference -> action                  policy frame
//!     -> after_inference(action)
//!   decode (default pose, scale, filter, limits)        policy frame
//!     -> after_decode(motor command)                    -> robot frame
//!   publish                                             robot frame
//! ```
//!
//! plus [`ModeHook::home_pose`], once, for the pose the mode ramps to and holds
//! before a policy takes over -- a robot-frame pose, since the ramp drives real
//! joints -- and [`ModeHook::reset`] each time a policy does take over.
//!
//! **The decoder runs in the policy frame**, and keeps its own copy of what it
//! carries between ticks (the previous target, the torque it applied) in that
//! frame too: its default pose is the contract's, which is the pose the policy
//! trained around. What is published is only ever what `after_decode` returns.
//! The last action the observation shows the policy is its own output, before
//! `after_inference`, because that is what it was trained to see.
//!
//! ## The operator's controls
//!
//! A task may answer a control itself rather than through a command -- the
//! claw `jumper.five_foot` closes is no policy's business. Such a control is
//! **declared in the task's `controls.yaml`**, by a name of the task's own,
//! under `task:` -- `claw_left`, an `amount`; `arm_thumb_up`, a `press` -- and
//! each device binds it there beside everything else it does while the task
//! drives: the pad to one of its controls (`LT`, `dpad_up`), the keyboard to
//! keystrokes (Space, Shift). That is what refuses a command binding the same
//! control and keeps the file the whole of what each device means. The hook is
//! handed those controls by name, and only those, as the operator reads them
//! this tick from whichever device was touched last
//! ([`crate::types::TaskControls`], `Operator::task_controls`); a name nothing
//! holds reads zero. Since 2026-09-29, when the keyboard stopped being a copy of
//! the pad: before, a hook read pad controls, and the keyboard reached them only
//! by pressing a virtual pad.
//!
//! [`ModeHook::reads`] says which of them it answers. The controller refuses a
//! hook that reads a name its contract's `task` does not declare, and a mode
//! whose task declares controls and that asks for no hook -- either way a
//! squeeze would do nothing on the robot, and look like it should.
//!
//! ## Where a hook lives
//!
//! `tasks/<family>/<task>/deploy/lib.rs`, one library per task, beside the
//! task's config -- never in this crate. `build.rs` walks `tasks/` for them and
//! compiles each in as a module, keyed by the task id its path spells
//! (`tasks/jumper/five_foot/deploy/` is `jumper.five_foot`), so all three hosts
//! -- the board, the browser and `play`'s extension -- carry every hook with no
//! loading at run time, and the browser could not load one anyway. All three
//! answer with it: `play --app` writes the controller's final targets and gains
//! into the simulator, past `after_decode` (`rl/mjrl/app_play.py`).
//!
//! A task's library sees this module and `crate::types`, and exports one
//! function with the [`HookBuilder`] signature:
//!
//! ```ignore
//! pub fn build(setup: &crate::hook::HookSetup<'_>)
//!     -> Result<Box<dyn crate::hook::ModeHook>, String>
//! ```
//!
//! ## How a mode asks for one
//!
//! A mode in `deploy/manifests.json` carries a `hook` object, its task's
//! configuration. `scripts/deploy.py` writes it into that mode's `[[fsm.state]]`
//! beside the task id, and [`build`] hands it to the task's `build` when the
//! controller is constructed. A mode without `hook` gets none -- its task's
//! library, if there is one, is not consulted -- and a `hook` for a task that
//! compiled none is refused, naming the ones that did.

use crate::layout::Contract;
use crate::types::{Command, MotorCommand, RobotState, TaskControls};

/// What a task may do to one mode's loop. Every method has a default that does
/// nothing, so a task overrides only the points it needs.
pub trait ModeHook: Send {
    /// The task controls this hook answers, by the names its task's
    /// `controls.yaml` declares them under (`task:`); see the module note.
    fn reads(&self) -> Vec<String> {
        Vec::new()
    }

    /// The pose this mode ramps to and holds before its policy takes over, in
    /// wire order, robot frame. Given the contract's own home pose, which is
    /// the policy frame's.
    fn home_pose(&self, default_wire: &[f32]) -> Vec<f32> {
        default_wire.to_vec()
    }

    /// A policy is taking over this mode, from `home_pose`.
    fn reset(&mut self) {}

    /// Before each observation: the robot's state and the operator's command,
    /// rewritten into what the policy should see. The command arrives already
    /// shaped by the contract's bands and ramp. `controls` are the task's own,
    /// by name, read-only: what the hook does with them is its own, and no
    /// policy sees them.
    fn before_observation(
        &mut self,
        _state: &mut RobotState,
        _command: &mut Command,
        _controls: &TaskControls,
    ) {
    }

    /// After each inference, before decoding: the raw action.
    fn after_inference(&mut self, _action: &mut [f32]) {}

    /// After decoding: the motor command, rewritten into what the robot should
    /// receive. This is what is published.
    fn after_decode(&mut self, _motor: &mut MotorCommand) {}
}

/// What a task's `build` is given.
pub struct HookSetup<'a> {
    /// The task id, as the bundler recorded it for this mode.
    pub task: &'a str,
    /// The mode's name, for messages.
    pub mode: &'a str,
    /// The mode's `hook` table from the manifest, verbatim.
    pub config: &'a toml::Table,
    /// The mode's contract: joint names in wire order, the home pose, the
    /// observation and action maps.
    pub contract: &'a Contract,
}

/// The one function a task's `deploy/lib.rs` exports, named `build`.
pub type HookBuilder = fn(&HookSetup<'_>) -> Result<Box<dyn ModeHook>, String>;

/// Every task library `build.rs` found, as modules of this one.
mod compiled {
    include!(concat!(env!("OUT_DIR"), "/task_hooks.rs"));
}

/// The task ids with a hook compiled in, sorted.
pub fn tasks() -> &'static [&'static str] {
    compiled::TASKS
}

/// Build the hook a mode asked for.
///
/// Refuses a task with no library compiled in, and passes on the task's own
/// refusal of its configuration, each with the mode's name: a hook that could
/// not be built is a mode that would run without it, and for a mirror that is
/// a robot carrying its claw on the side the operator did not choose.
pub fn build(setup: &HookSetup<'_>) -> Result<Box<dyn ModeHook>, String> {
    let builder = compiled::builder(setup.task);
    #[cfg(test)]
    let builder = builder.or_else(|| testing::builder(setup.task));
    let builder = builder.ok_or_else(|| {
        format!(
            "mode '{}' asks for a hook from task '{}', which compiled none. Tasks with \
             one: {:?}. A task's hook is `tasks/<family>/<task>/deploy/lib.rs`.",
            setup.mode,
            setup.task,
            tasks()
        )
    })?;
    builder(setup).map_err(|e| format!("mode '{}': {}'s hook: {e}", setup.mode, setup.task))
}

/// A hook whose every effect is visible, for testing the calls rather than any
/// task: `task = "test.visible"`. Kept here so the controller's own tests do not
/// depend on which tasks happen to have a library.
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use std::cell::{Cell, RefCell};

    thread_local! {
        /// Resets seen on this thread. Tests run on their own threads, and the
        /// controller they build runs on the same one.
        pub static RESETS: Cell<usize> = const { Cell::new(0) };
        /// The controls the last observation was handed.
        pub static CONTROLS: RefCell<TaskControls> = RefCell::new(TaskControls::default());
    }

    /// Home `+ offset`; the joints the policy sees `+ 10`; the action doubled;
    /// the published targets `+ shift`. Reads the controls `reads` names, and
    /// records what it is handed.
    struct Visible {
        offset: f32,
        shift: f32,
        reads: Vec<String>,
    }

    impl ModeHook for Visible {
        fn reads(&self) -> Vec<String> {
            self.reads.clone()
        }
        fn home_pose(&self, default_wire: &[f32]) -> Vec<f32> {
            default_wire.iter().map(|q| q + self.offset).collect()
        }
        fn reset(&mut self) {
            RESETS.with(|r| r.set(r.get() + 1));
        }
        fn before_observation(
            &mut self,
            state: &mut RobotState,
            _c: &mut Command,
            controls: &TaskControls,
        ) {
            state.q.iter_mut().for_each(|q| *q += 10.0);
            CONTROLS.with(|c| *c.borrow_mut() = controls.clone());
        }
        fn after_inference(&mut self, action: &mut [f32]) {
            action.iter_mut().for_each(|a| *a *= 2.0);
        }
        fn after_decode(&mut self, motor: &mut MotorCommand) {
            motor.pos.iter_mut().for_each(|p| *p += self.shift);
        }
    }

    fn build(setup: &HookSetup<'_>) -> Result<Box<dyn ModeHook>, String> {
        let get = |k: &str| {
            setup.config.get(k).and_then(|v| v.as_float()).ok_or_else(|| format!("`{k}` is a float"))
        };
        let reads = match setup.config.get("reads") {
            None => Vec::new(),
            Some(v) => v
                .as_array()
                .and_then(|a| a.iter().map(|x| x.as_str().map(String::from)).collect())
                .ok_or("`reads` is a list of task control names")?,
        };
        Ok(Box::new(Visible { offset: get("offset")? as f32, shift: get("shift")? as f32, reads }))
    }

    pub fn builder(task: &str) -> Option<HookBuilder> {
        (task == "test.visible").then_some(build as HookBuilder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `build.rs` walks `tasks/` and compiles what it finds. The one test here
    /// that names a task, because it is the repository's own library that says
    /// whether the walk works: an empty table would pass every other test.
    #[test]
    fn the_build_finds_the_task_libraries() {
        assert!(tasks().contains(&"jumper.five_foot"), "compiled: {:?}", tasks());
        assert!(tasks().windows(2).all(|w| w[0] < w[1]), "sorted and unique: {:?}", tasks());
        for t in tasks() {
            assert!(compiled::builder(t).is_some(), "{t} is listed and has no builder");
        }
        assert!(compiled::builder("test.visible").is_none(), "the test hook is not compiled in");
    }
}
