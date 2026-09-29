//! Run an exported policy on a Rockchip NPU, over the robot's DDS bus.
//!
//! The chain this closes:
//!
//! ```text
//! model_*.pt --export.py--> actor.onnx --onnx2rknn.py--> actor.rknn
//!                           layout.json ------------------------\
//!                                                                v
//!                                          this crate: obs -> NPU -> joint targets
//! ```
//!
//! Two modes, and the boundary between them is the point:
//!
//! - **default** — hold the home pose from `layout.json`. Touches no model, so it
//!   is what a missing, unloadable or misbehaving policy falls back to.
//! - **user-defined** — build the observation, run the policy on the NPU, apply
//!   the action.
//!
//! Everything the policy needs is read from its own `layout.json`. The robot's
//! wire order is the one thing that is not, and the two are paired **by joint
//! name**; see `layout.rs` for why that matters more than it looks.

pub mod action;
// Opening a bundle directory. Off for `wasm32`, which has no filesystem to point
// at and receives its bundle as bytes -- see the module docstring.
#[cfg(not(target_arch = "wasm32"))]
pub mod bundle;
pub mod config;
pub mod control;
pub mod fsm;
// What each control of the pad does and which key presses it, as data, for a
// host that draws the pad. Portable: it reads the config and the operators.
pub mod guide;
// A task's own code at fixed points of a mode's loop, compiled in from
// `tasks/**/deploy/lib.rs` by build.rs. Portable: every host runs the hooks.
pub mod hook;
pub mod layout;
pub mod obs;
// A task's controls: several commands, one operator, the pad and the keyboard
// as two paths. Portable: every host reads a pad, and two of them keys.
pub mod operator;
// Replaying the reference vectors a bundle carries. Portable: the browser is
// the host where a divergence is least visible and most worth measuring.
pub mod reference;
pub mod source;
// The recorded motion a residual policy corrects. Portable: every host that
// runs such a policy needs it, and the tables arrive already parsed because
// this crate does no I/O.
pub mod trajectory;

// The browser host. A `cdylib` for `wasm32`, and off everywhere else.
#[cfg(feature = "web")]
pub mod web;

// The `play` host: an importable Python extension. Off by default --
// `extension-module` links no libpython, so `cargo test` could not link with it.
#[cfg(feature = "py")]
pub mod py;
pub mod types;
// The words the operator's hardware can say. One list, and everything that
// names a control goes through it.
pub mod vocabulary;

// Everything that touches the robot's bus or its NPU. Off in a `wasm32` build,
// which is what keeps the four modules above portable -- see `Cargo.toml`.
#[cfg(feature = "device")]
pub mod dds;
#[cfg(feature = "device")]
pub mod device;
#[cfg(feature = "device")]
pub mod rknn;
