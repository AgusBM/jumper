//! The NPU: `librknnrt.so` through its C API.
//!
//! Six functions and two structs, hand-declared rather than bindgen'd. The
//! surface is small and fixed, and the fields below are exactly the ones the
//! working C++ controller sets -- which matters, because several of the ones it
//! leaves zero are not "don't care": `pass_through = 0` means "convert for me"
//! and `is_prealloc = 0` means "you allocate the output". Getting either wrong
//! produces garbage rather than an error.
//!
//! **This module cannot be exercised off the board.** The runtime is an aarch64
//! `.so` that talks to real NPU hardware, so a host build compiles `StubEngine`
//! only. That is the same split the C++ controller makes, and it is what lets
//! everything above this file be tested on a development machine.

use crate::layout::Contract;

/// What the control loop needs from a model. Implemented by the NPU engine and
/// by the stub, so the loop has no idea which it is talking to.
pub trait Engine {
    /// Observation width the model expects.
    fn input_size(&self) -> usize;
    /// Action width the model produces.
    fn output_size(&self) -> usize;
    /// Run one observation. The slice is valid until the next call.
    fn infer(&mut self, obs: &[f32]) -> Result<&[f32], Error>;
    fn name(&self) -> &str;
    /// True when this is not a real model. The loop refuses to *drive* on a
    /// stub: a zero action means "hold the home pose", which is safe, but it
    /// should never be mistaken for a policy that is running.
    fn is_stub(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// A `librknnrt` call returned non-zero. The API reports errors this way and
    /// never raises, so every call is checked.
    Rknn { call: &'static str, code: i32 },
    /// The model is not the single-input, single-output MLP this controller runs.
    Shape(String),
    /// The observation handed in is not the width the model wants.
    Width { got: usize, want: usize },
    NotAvailable,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "cannot read model: {e}"),
            Error::Rknn { call, code } => write!(f, "{call} failed ({code})"),
            Error::Shape(m) => write!(f, "{m}"),
            Error::Width { got, want } => {
                write!(f, "observation is {got} wide, the model wants {want}")
            }
            Error::NotAvailable => write!(
                f,
                "built without the RKNN runtime (feature 'rknn'); no NPU backend"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Returns a zero action of the right width, which decodes to the home pose.
pub struct StubEngine {
    out: Vec<f32>,
}

impl StubEngine {
    pub fn new(output_size: usize) -> Self {
        Self { out: vec![0.0; output_size] }
    }
}

impl Engine for StubEngine {
    fn input_size(&self) -> usize {
        0 // accepts anything
    }
    fn output_size(&self) -> usize {
        self.out.len()
    }
    fn infer(&mut self, _obs: &[f32]) -> Result<&[f32], Error> {
        Ok(&self.out)
    }
    fn name(&self) -> &str {
        "stub"
    }
    fn is_stub(&self) -> bool {
        true
    }
}

// The type definitions are deliberately outside the `rknn` feature gate while
// the `extern` block below is inside it. That is what lets
// `tests::ffi_structs_match_the_c_header` run on a development machine: the
// header is portable, only `librknnrt.so` is aarch64. The cost is that a host
// build sees these as unused.
#[cfg_attr(not(feature = "rknn"), allow(dead_code))]
mod ffi {
    use std::os::raw::{c_int, c_void};

    pub type RknnContext = u64;

    pub const RKNN_SUCC: c_int = 0;
    pub const RKNN_QUERY_IN_OUT_NUM: c_int = 0;
    pub const RKNN_QUERY_INPUT_ATTR: c_int = 1;
    pub const RKNN_QUERY_OUTPUT_ATTR: c_int = 2;
    pub const RKNN_TENSOR_FLOAT32: c_int = 0;
    /// Not NCHW and not NHWC. A `[1, N]` observation has no spatial layout, and
    /// declaring one would invite the runtime to permute it.
    pub const RKNN_TENSOR_UNDEFINED: c_int = 3;

    #[repr(C)]
    #[derive(Default)]
    pub struct RknnInputOutputNum {
        pub n_input: u32,
        pub n_output: u32,
    }

    /// `rknn_tensor_attr`, field for field from `vendor/rknpu2/include/rknn_api.h`.
    ///
    /// Only `n_elems` is read, but the whole struct is mirrored anyway:
    /// `rknn_query` writes `sizeof` bytes into it, so a layout that is merely
    /// "big enough" would still read `n_elems` from the wrong offset -- a
    /// plausible number, not a crash. `tests::tensor_attr_layout_matches_the_c_header`
    /// checks this against the header itself.
    #[repr(C)]
    pub struct RknnTensorAttr {
        pub index: u32,
        pub n_dims: u32,
        pub dims: [u32; 16],   // RKNN_MAX_DIMS
        pub name: [u8; 256],   // RKNN_MAX_NAME_LEN, `char name[]`
        pub n_elems: u32,
        pub size: u32,
        pub fmt: c_int,      // rknn_tensor_format
        pub type_: c_int,    // rknn_tensor_type
        pub qnt_type: c_int, // rknn_tensor_qnt_type
        pub fl: i8,
        pub zp: i32,
        pub scale: f32,
        pub w_stride: u32,
        pub size_with_stride: u32,
        pub pass_through: u8,
        pub h_stride: u32,
    }

    impl RknnTensorAttr {
        /// The callee fills this; an all-zero start is what the C code does too.
        pub fn zeroed() -> Self {
            // SAFETY: every field is a plain integer or float; no niches.
            unsafe { std::mem::zeroed() }
        }
    }

    #[repr(C)]
    pub struct RknnInput {
        pub index: u32,
        pub buf: *mut c_void,
        pub size: u32,
        pub pass_through: u8,
        pub type_: c_int,
        pub fmt: c_int,
    }

    #[repr(C)]
    pub struct RknnOutput {
        pub want_float: u8,
        pub is_prealloc: u8,
        pub index: u32,
        pub buf: *mut c_void,
        pub size: u32,
    }

    #[cfg(feature = "rknn")]
    extern "C" {
        pub fn rknn_init(
            ctx: *mut RknnContext,
            model: *mut c_void,
            size: u32,
            flag: u32,
            extend: *mut c_void,
        ) -> c_int;
        pub fn rknn_destroy(ctx: RknnContext) -> c_int;
        pub fn rknn_query(ctx: RknnContext, cmd: c_int, info: *mut c_void, size: u32) -> c_int;
        pub fn rknn_inputs_set(ctx: RknnContext, n: u32, inputs: *mut RknnInput) -> c_int;
        pub fn rknn_run(ctx: RknnContext, extend: *mut c_void) -> c_int;
        pub fn rknn_outputs_get(
            ctx: RknnContext,
            n: u32,
            outputs: *mut RknnOutput,
            extend: *mut c_void,
        ) -> c_int;
        pub fn rknn_outputs_release(ctx: RknnContext, n: u32, outputs: *mut RknnOutput) -> c_int;
    }
}

#[cfg(feature = "rknn")]
pub struct RknnEngine {
    ctx: ffi::RknnContext,
    /// The runtime may reference the blob for the context's lifetime, so it is
    /// owned here and not dropped after `rknn_init`.
    _blob: Vec<u8>,
    input_size: usize,
    output_size: usize,
    out: Vec<f32>,
    name: String,
}

#[cfg(feature = "rknn")]
impl RknnEngine {
    /// Load a `.rknn`. `expected_output` is the layout's action width; a model
    /// that disagrees is refused here rather than producing a short action that
    /// leaves the tail of the joint targets at whatever was there before.
    pub fn load(path: &std::path::Path, expected_output: usize) -> Result<Self, Error> {
        use std::os::raw::c_void;

        let mut blob = std::fs::read(path).map_err(Error::Io)?;
        let mut ctx: ffi::RknnContext = 0;
        // SAFETY: blob outlives ctx (owned by the returned struct); the runtime
        // is told the exact byte length.
        let ret = unsafe {
            ffi::rknn_init(
                &mut ctx,
                blob.as_mut_ptr() as *mut c_void,
                blob.len() as u32,
                0,
                std::ptr::null_mut(),
            )
        };
        if ret != ffi::RKNN_SUCC {
            return Err(Error::Rknn { call: "rknn_init", code: ret });
        }

        // From here every failure must destroy the context before returning.
        let finish = |ctx: ffi::RknnContext, e: Error| -> Error {
            unsafe { ffi::rknn_destroy(ctx) };
            e
        };

        let mut io = ffi::RknnInputOutputNum::default();
        let ret = unsafe {
            ffi::rknn_query(
                ctx,
                ffi::RKNN_QUERY_IN_OUT_NUM,
                &mut io as *mut _ as *mut c_void,
                std::mem::size_of::<ffi::RknnInputOutputNum>() as u32,
            )
        };
        if ret != ffi::RKNN_SUCC {
            return Err(finish(ctx, Error::Rknn { call: "rknn_query(IN_OUT_NUM)", code: ret }));
        }
        if io.n_input != 1 || io.n_output != 1 {
            return Err(finish(
                ctx,
                Error::Shape(format!(
                    "expected 1 input and 1 output, model has {} and {}",
                    io.n_input, io.n_output
                )),
            ));
        }

        let query_attr = |cmd: i32, call: &'static str| -> Result<u32, Error> {
            let mut attr = ffi::RknnTensorAttr::zeroed();
            let ret = unsafe {
                ffi::rknn_query(
                    ctx,
                    cmd,
                    &mut attr as *mut _ as *mut c_void,
                    std::mem::size_of::<ffi::RknnTensorAttr>() as u32,
                )
            };
            if ret != ffi::RKNN_SUCC {
                return Err(Error::Rknn { call, code: ret });
            }
            Ok(attr.n_elems)
        };

        let input_size = match query_attr(ffi::RKNN_QUERY_INPUT_ATTR, "rknn_query(INPUT_ATTR)") {
            Ok(v) => v as usize,
            Err(e) => return Err(finish(ctx, e)),
        };
        let output_size = match query_attr(ffi::RKNN_QUERY_OUTPUT_ATTR, "rknn_query(OUTPUT_ATTR)") {
            Ok(v) => v as usize,
            Err(e) => return Err(finish(ctx, e)),
        };

        if expected_output > 0 && output_size != expected_output {
            return Err(finish(
                ctx,
                Error::Shape(format!(
                    "model output is {output_size} wide but the layout's action is \
                     {expected_output}; these are not the same policy"
                )),
            ));
        }

        Ok(Self {
            ctx,
            _blob: blob,
            input_size,
            output_size,
            out: vec![0.0; output_size],
            name: path.display().to_string(),
        })
    }
}

#[cfg(feature = "rknn")]
impl Engine for RknnEngine {
    fn input_size(&self) -> usize {
        self.input_size
    }
    fn output_size(&self) -> usize {
        self.output_size
    }
    fn name(&self) -> &str {
        &self.name
    }

    fn infer(&mut self, obs: &[f32]) -> Result<&[f32], Error> {
        use std::os::raw::c_void;

        if obs.len() != self.input_size {
            return Err(Error::Width { got: obs.len(), want: self.input_size });
        }

        let mut input = ffi::RknnInput {
            index: 0,
            buf: obs.as_ptr() as *mut c_void,
            size: (obs.len() * std::mem::size_of::<f32>()) as u32,
            pass_through: 0,
            type_: ffi::RKNN_TENSOR_FLOAT32,
            fmt: ffi::RKNN_TENSOR_UNDEFINED,
        };
        // SAFETY: obs outlives the call; the runtime copies during inputs_set.
        let ret = unsafe { ffi::rknn_inputs_set(self.ctx, 1, &mut input) };
        if ret != ffi::RKNN_SUCC {
            return Err(Error::Rknn { call: "rknn_inputs_set", code: ret });
        }
        let ret = unsafe { ffi::rknn_run(self.ctx, std::ptr::null_mut()) };
        if ret != ffi::RKNN_SUCC {
            return Err(Error::Rknn { call: "rknn_run", code: ret });
        }

        let mut output = ffi::RknnOutput {
            want_float: 1,
            is_prealloc: 0, // the runtime allocates; released below
            index: 0,
            buf: std::ptr::null_mut(),
            size: 0,
        };
        let ret =
            unsafe { ffi::rknn_outputs_get(self.ctx, 1, &mut output, std::ptr::null_mut()) };
        if ret != ffi::RKNN_SUCC {
            return Err(Error::Rknn { call: "rknn_outputs_get", code: ret });
        }

        let n = self.output_size.min(output.size as usize / std::mem::size_of::<f32>());
        // A short read must not leave stale values in the tail: the previous
        // tick's action would be applied to those joints, silently.
        self.out.iter_mut().for_each(|v| *v = 0.0);
        // SAFETY: the runtime filled buf with `output.size` bytes of f32.
        unsafe {
            std::ptr::copy_nonoverlapping(output.buf as *const f32, self.out.as_mut_ptr(), n);
        }
        unsafe { ffi::rknn_outputs_release(self.ctx, 1, &mut output) };

        if n != self.output_size {
            return Err(Error::Shape(format!(
                "model returned {n} values, expected {}",
                self.output_size
            )));
        }
        Ok(&self.out)
    }
}

#[cfg(feature = "rknn")]
impl Drop for RknnEngine {
    fn drop(&mut self) {
        if self.ctx != 0 {
            unsafe { ffi::rknn_destroy(self.ctx) };
            self.ctx = 0;
        }
    }
}

/// Load the policy for a mode, falling back to the stub with a reason.
///
/// Falling back rather than failing is deliberate: a controller that will not
/// start is a robot that cannot even be told to stand. The caller gets a stub
/// and the reason, and is expected to say so loudly and stay in default mode.
pub fn make_engine(
    model_path: Option<&std::path::Path>,
    contract: &Contract,
) -> (Box<dyn Engine>, Option<String>) {
    let action_dim = contract.action_dim();
    let Some(path) = model_path else {
        return (Box::new(StubEngine::new(action_dim)), Some("no model configured".into()));
    };
    if !path.is_file() {
        return (
            Box::new(StubEngine::new(action_dim)),
            Some(format!("model not found: {}", path.display())),
        );
    }
    #[cfg(feature = "rknn")]
    {
        match RknnEngine::load(path, action_dim) {
            Ok(e) => (Box::new(e), None),
            Err(e) => (Box::new(StubEngine::new(action_dim)), Some(format!("RKNN load failed: {e}"))),
        }
    }
    #[cfg(not(feature = "rknn"))]
    {
        (
            Box::new(StubEngine::new(action_dim)),
            Some(format!("{}", Error::NotAvailable)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" {
        fn mjrl_rknn_attr_size() -> usize;
        fn mjrl_rknn_attr_n_elems_offset() -> usize;
        fn mjrl_rknn_input_size() -> usize;
        fn mjrl_rknn_output_size() -> usize;
    }

    /// The Rust structs must be laid out exactly as the C header lays them out.
    ///
    /// This is the check that cannot be skipped and cannot wait for hardware.
    /// `rknn_query` writes `sizeof(rknn_tensor_attr)` bytes into the struct and
    /// the code then reads `n_elems` from wherever Rust thinks it is; a wrong
    /// offset yields a plausible integer, not a crash, and the first sign of it
    /// would be a model rejected for the wrong width -- or worse, accepted.
    ///
    /// It runs on any host because the header is portable; only `librknnrt.so`
    /// is aarch64.
    #[test]
    fn ffi_structs_match_the_c_header() {
        // SAFETY: the probe returns plain sizes, computed by the C compiler from
        // vendor/rknpu2/include/rknn_api.h.
        let (attr_size, n_elems_off, input_size, output_size) = unsafe {
            (
                mjrl_rknn_attr_size(),
                mjrl_rknn_attr_n_elems_offset(),
                mjrl_rknn_input_size(),
                mjrl_rknn_output_size(),
            )
        };
        assert_eq!(
            std::mem::size_of::<ffi::RknnTensorAttr>(),
            attr_size,
            "rknn_tensor_attr size disagrees with the header"
        );
        assert_eq!(
            std::mem::offset_of!(ffi::RknnTensorAttr, n_elems),
            n_elems_off,
            "n_elems is at the wrong offset; every queried width would be wrong"
        );
        assert_eq!(std::mem::size_of::<ffi::RknnInput>(), input_size);
        assert_eq!(std::mem::size_of::<ffi::RknnOutput>(), output_size);
    }

    #[test]
    fn stub_returns_a_zero_action_of_the_right_width() {
        let mut s = StubEngine::new(20);
        assert_eq!(s.output_size(), 20);
        assert!(s.is_stub());
        let out = s.infer(&[1.0; 411]).unwrap();
        assert_eq!(out.len(), 20);
        assert!(out.iter().all(|&v| v == 0.0), "a zero action decodes to the home pose");
    }
}
