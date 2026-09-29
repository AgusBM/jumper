---
name: setup-env
description: Set up, verify or repair the mjrl-lab development environment on Linux, Windows or macOS, with or without an NVIDIA GPU -- the Python environment and the Rust toolchain the controller (deploy/fsm) builds with. Detects the machine, picks the right torch build, finds which of cargo, the wasm target, wasm-bindgen and CycloneDDS are missing, installs, and runs six verification gates. Use this whenever someone is getting this repository running on a new machine, onboarding, or hitting environment trouble: "ModuleNotFoundError: No module named 'mjlab'", torch not seeing the GPU, warp reporting only ['cpu'], `--list` showing no tasks, mujoco version conflicts, `cargo` not found, `idlc not found`, a wasm-bindgen version mismatch, the controller's tests skipping, or a suspicion that the wrong Python or the wrong checkout is being used. Reach for it before hand-rolling pip or cargo commands, because the failures here are quiet -- an environment that imports everything and trains happily can still be running another checkout's code or ignoring the GPU entirely, and a test suite on a machine without cargo is green because it skipped the controller.
---

# Setting up mjrl-lab

The reference procedure is [`docs/AGENT_SETUP.md`](../../../docs/AGENT_SETUP.md) — about 600 lines,
written to be read by an agent, with the full platform tables and a catalogue of known failures.
**This skill does not restate it.** It sequences it, automates the two parts worth automating,
and knows when to stop and read.

Two scripts carry the weight. Both are plain standard-library Python and run without this skill,
so an agent that cannot load skills can still use them:

| Script | Interpreter | Answers |
|---|---|---|
| `scripts/detect.py` | whatever the machine already has | what this machine is, and the exact commands that follow from it |
| `scripts/gates.py` | the environment's own | whether the environment actually works, and what to fix |

## The shape of the problem

Almost everything here fails **quietly**. Installing a cu126 torch on a Blackwell card succeeds
and dies later inside CUDA. An editable install pointing at a stale checkout imports cleanly and
runs somebody else's code. A `pip` into the wrong interpreter reports success and changes
nothing. A machine without cargo runs the test suite green, because the tests that build the
controller skip. None of these announce themselves, and all of them surface much later as
something that looks unrelated.

That is why this measures before deciding and verifies after acting, rather than running a
sequence of commands and trusting that silence means success.

## Procedure

### 1. Measure, before deciding anything

```bash
python3 .claude/skills/setup-env/scripts/detect.py
```

Run it with the machine's existing interpreter — establishing whether that interpreter is usable
at all is part of what it answers. It reports OS, architecture, Python version, GPU and compute
capability, any environment that already exists, which parts of the controller's toolchain are
already here, and it ends with the exact command sequence for this machine.

Read its output before continuing. In particular:

- **Python out of range** (needs 3.10–3.13): install a compliant interpreter *first*. Installing
  dependencies on an old one produces failures much later that read as unrelated.
- **An environment already exists**: go to step 3 and verify it. The common case after the first
  run is not a bare machine but a broken environment, and building a second one hides the fault
  rather than fixing it.

### 2. Install

Follow the commands `detect.py` printed. They are section 1 of the reference doc with the
branches already resolved for this machine; consult the doc when something does not match — the
Debian `ensurepip` split, the Windows execution policy, and the MSVC prerequisite each have a
paragraph there.

Only one step genuinely branches: **which torch wheel**. `detect.py` chose it from the measured
compute capability, and its reasoning is on the `torch_plan` function if you need to check it.

Before any `pip`, confirm the interpreter is the intended one:

```bash
python -c "import sys; print(sys.prefix)"
```

It must be the environment you just made. Pip reports success regardless of where it installs, so
this is the cheapest place to catch a mistake that otherwise surfaces three steps later.

`pip install -e .` is not optional and is not a convenience: the vendored `mjlab` and `rsl_rl`
live under `rl/`, which is not on `sys.path`. That editable install *is* the mechanism that makes
them importable.

**The controller's toolchain is part of the install, on every machine** — sections 2.1 and 2.2
of the reference doc. It is additive rather than per-checkout, so `detect.py` prints only the
steps this machine is missing, and "nothing missing" is a normal answer on a second checkout.
Two of its parts are easy to get subtly wrong:

- **wasm-bindgen's version comes from `deploy/fsm/Cargo.lock`**, exactly — not from
  `Cargo.toml`'s `"0.2"`, a range every 0.2.x satisfies and `cargo install --version` refuses.
  Take the command `detect.py` printed; `scripts/deploy.py` refuses any other version.
- **CycloneDDS is a source build** (11.0.1, the soname the board carries) and needs
  `sudo ldconfig` after installing into `/usr/local`. It is not needed for `deploy.py`, but once
  cargo exists `tests/test_deploy_sources.py` stops skipping and fails without it.
- **Rockchip's NPU header is per checkout, not per machine.** `bash
  deploy/fsm/vendor/rknpu2/fetch.sh` puts it in the repository, git-ignored because its licence
  is Rockchip's. `detect.py` prints that step whenever it is missing, a second checkout included.

### 3. Verify

```bash
.venv/bin/python .claude/skills/setup-env/scripts/gates.py
```

Six gates, exit code = number of failures. Each prints what it measured and, on failure, the
remedy and where the long form lives:

- **A — warp device enumeration.** On a GPU machine `cuda:0` must appear in the device list.
  `wp.init()` also prints a banner naming the card; that banner appears even when CUDA is
  unusable, so the list is the criterion and the banner is not.
- **B — native MuJoCo's batch interface.** `native:cpu` is built on `mujoco.rollout`, so
  importing `mujoco` is not sufficient evidence.
- **C — the vendored copies are this checkout's.** Stricter than the doc's version, which only
  rules out `site-packages`. An editable install left pointing at an older copy of the repository
  satisfies that and still executes another checkout's code — measured on a training machine on
  2026-09-04, where `tasks` came from the current repository and `mjlab` from a snapshot three
  days stale. Everything imported; the seam being edited simply was not the seam being run.
- **D — the task registry.** Loads no simulation dependencies, so it passes even where the GPU is
  broken. A failure here is a packaging problem, not a simulation one — which is worth knowing
  before looking in the wrong place.
- **E — the Rust toolchain builds the controller's host targets.** `cargo check` of the browser's
  wasm and of `play --app`'s extension, and the wasm-bindgen CLI against `Cargo.lock`. It builds
  rather than reading versions, because a missing target or linker passes `cargo --version`.
- **F — the device build links and loads.** `cargo test --lib -- --list` with the default
  features: CycloneDDS, libclang, Rockchip's NPU header and a `libddsc` the loader finds. `--list` runs the binary
  without running a test, so a failing crate test is not reported as an environment fault.

E and F take about 30 s together, cold, on the i9-14900KF with the crates already downloaded; a
first run also downloads them.

Do not proceed past a failure. Each remedy is printed; the long forms are section 6 of the
reference doc.

### 4. Smoke test

With all six gates green, run training end to end — the commands are section 4 of the reference
doc, and `--headless` keeps the test independent of whether a display exists.

```bash
# with a GPU
python scripts/train.py --task jumper.flat --num_envs 256 --max-iterations 5 --headless
# without
python scripts/train.py --task jumper.flat --backend native --device cpu --num_envs 64 --max-iterations 3 --headless
```

Expect the actor/critic structure, several `Iteration time:` lines, exit code 0. The banner on
the first line reports the backend actually chosen — worth reading, because `auto` resolving to
`native:cpu` on a machine with a GPU means gate A was passed too generously.

## When something fails

Work from the measurement, not from the symptom. `detect.py` and `gates.py` both print what they
observed, and the reference doc's section 6 is indexed by the exact error text — search it for the
message rather than reasoning from the traceback, because several of these have causes that the
traceback does not mention (the `njmax` overflow, the wandb `start_method` rejection, and
`select_gpus` raising `IndexError` are all in that class).

Two rules survive every platform, and both are in section 7 of the doc:

- **Never `pip install mjlab` or `rsl-rl-lib`.** They exist as modifiable copies under `rl/`.
  A PyPI copy in the same environment shadows them ambiguously and is very hard to notice.
- **Never install into an existing Isaac Lab environment.** mjlab needs `mujoco~=3.11`, Isaac Lab
  ships 3.10, and they break each other.

## Reporting back

Say which branch the machine took, which gates passed, and what the smoke test measured. If a
gate was skipped or a step was worked around, say so plainly — an environment reported as ready
when one gate was quietly skipped is worse than one reported as broken, because the next failure
will be attributed to the code instead of the setup.
