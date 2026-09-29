# Contributing

This repository keeps the training framework, robot tasks, model assets and deployment
controller in one tree. Start with the small check that matches the change; run the whole
suite before handing it over.

```bash
git clone https://github.com/KingKongRobotics/jumper.git
cd jumper
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -q
python -m ruff check .
```

The full environment procedure, including platform-specific PyTorch and the Rust toolchain,
is in [`docs/USAGE.md`](docs/USAGE.md#setting-up). Architecture and trade-offs belong in
[`docs/DESIGN.md`](docs/DESIGN.md); day-to-day commands and extension guides belong in
[`docs/USAGE.md`](docs/USAGE.md).

## Directory layout

```
jumper/
├── rl/                     # package root for RL libraries (not a package itself)
│   ├── mjrl/               #   the framework, robot-agnostic
│   │   ├── backend/        #     backend selection, stepping, model bridge
│   │   ├── sensor/         #     native sensor implementations
│   │   └── viewer/         #     live viewer
│   ├── mjlab/              #   vendored fork ┐ verbatim except marked changes
│   └── rsl_rl/             #   vendored fork ┘ see docs/VENDOR.md
├── tasks/                  # a task id is its module path relative to tasks/
│   └── jumper/             #   one family: common/ plus one directory per task
├── scenes/                 # ground, sky, lights and props
├── controller/             # gamepad and keyboard vocabulary and readers
├── assets/                 # model files; each asset carries its own tools/
├── scripts/                # stable user entry points: train, play, export, deploy
├── deploy/                 # controller, board conversion and wire interface
│   ├── convert/            #   ONNX -> RKNN for the RK3576 NPU
│   ├── dds/                #   IDL, QoS and topics
│   └── fsm/                #   observation -> policy -> joint targets
├── tools/                  # development tools and one-off checks
├── tests/                  # pytest; no GPU required
└── docs/                   # manuals, design records and README media
```

## Where a new file goes

| What it is | Where | Test |
|---|---|---|
| Robot-agnostic framework code | `rl/mjrl/` | Useful on a different robot |
| One task's config and hyper-parameters | `tasks/<family>/<task>/` | Only that task needs it |
| An intrinsic property of one robot | `tasks/<family>/common/` | Unchanged across tasks |
| A skeleton several tasks share | `tasks/<family>/common/` | More than one task needs it |
| Generates or calibrates one asset | `assets/<name>/tools/` | Bound to that asset |
| A command run day to day | `scripts/` | User-facing, stable arguments |
| A one-off check or benchmark | `tools/` | Developer-facing, not asset-bound |
| Turns an export into something hardware runs | `deploy/` | Needs the board toolchain, not mjlab |

## Conventions that prevent silent failures

- **`pip install -e .` is mandatory; do not `pip install mjlab` or `rsl-rl-lib`.** The local
  copies live under `rl/`, and a PyPI copy shadows them ambiguously.
- **A task id is its directory.** `tasks/jumper/tripod/` is `jumper.tripod`; it also needs an
  import from `tasks/__init__.py` before the registry can see it.
- **Every tuning number lives in the task's own `env_cfg.py`**, even when tasks agree.
  `tasks/<family>/common/` holds mechanisms and hardware facts.
- **Deploy an export, not a hand-converted checkpoint.** The export carries joint order,
  observation layout, gains and the other contracts each host needs.
- **Mark every local change to vendored code with `[mjrl]`.** See
  [`docs/VENDOR.md`](docs/VENDOR.md) before editing `rl/mjlab/` or `rl/rsl_rl/`.
- **Repository prose is English.** A translated `*.zh.md` sits beside its source and records
  the source digest; update the translation and restamp it in the same change.

## Tests

The suite targets failures that would otherwise look successful: a task missing from the
registry, a wrong joint order, an ignored setting, or two backends disagreeing silently. Tests
cover shared machinery and contracts; a task's own rewards and tuning are checked by training
and replaying it.

Run a focused file while iterating, then the complete checks:

```bash
python -m pytest tests/test_registry.py -q
python -m pytest tests/ -q
python -m ruff check .
```

The controller has its own host, web and board targets. Follow [`deploy/README.md`](deploy/README.md)
and the deployment checks there when a change crosses that boundary.
