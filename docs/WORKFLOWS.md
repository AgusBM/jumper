# Train and Design from one project

Jumper is the training repository and the shared entry point for this project. Design is a
separate public repository, not a directory that this checkout is expected to contain. Ask an AI
coding assistant to read the current Design rules remotely before deciding
whether a Design checkout is needed:

- [Design AGENTS.md](https://github.com/KingKongRobotics/jumper-design/blob/main/AGENTS.md)
- [Design workflow](https://github.com/KingKongRobotics/jumper-design/blob/main/docs/workflow.md)
- [Design content-package protocol](https://github.com/KingKongRobotics/jumper-design/blob/main/docs/content-packages.md)

Clone the repositories independently only when execution requires them. A root training clone
does not require Git LFS:

```bash
git clone https://github.com/KingKongRobotics/jumper.git
cd jumper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The Design checkout is a sibling with its own environment. Do not add it to the Jumper checkout:

```bash
cd ..
git clone https://github.com/KingKongRobotics/jumper-design.git
cd jumper-design
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[sim,dev]"
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Fetch Design Git LFS objects only when
the requested Design execution needs the real model or other LFS-backed assets; pointer files are
not sufficient for validation or rendering:

```bash
git lfs install
git lfs pull
```

## Train

> Train Jumper to walk with a tripod gait; check the environment, choose CPU or GPU settings,
> then train, replay the result, and report the checkpoint and measured performance.

Follow [setup](AGENT_SETUP.md) and [the tutorial](TUTORIAL.md). In the root training checkout:

```bash
python scripts/train.py --list
python scripts/train.py --task jumper.tripod --backend native --device cpu --num_envs 64
python scripts/play.py --task jumper.tripod
```

Training takes time and compute. Report measured results, not an assumed successful behavior. On
macOS, use `.venv/bin/mjpython` for the live viewer as described in the setup guide.

## Design an appearance

> Give Jumper a warm sand ranger appearance with coordinated body and limb colors; show design
> options, then export the chosen design as a validated `.skin` with whole-robot previews.

Read the three current Design links above before using its CLI. New appearances require a concept
selection before geometry generation. Reuse an explicitly selected existing design when
appropriate. Printing and physical fitting are separate from display-only appearance creation.

From the root training checkout, enter the independent sibling checkout and use its environment:

```bash
cd ../jumper-design
source .venv/bin/activate
python scripts/shellflow.py --help
python scripts/shellflow.py verify-package library/skins/jumper-original.skin \
  --profile robots/jumper/profile.json --mujoco
```

Validation requires actual LFS assets when the package references them. Review real front, side,
back, assembled color and gray geometry renders; package integrity and rendering do not prove
physical fit.

## Design a scene

> Create a park pump track for Jumper with rolling terrain, trees and benches; export a `.map`,
> verify the robot spawn and model loading, and show the resulting scene.

Use the standalone Design scene workflow. For example, from `../jumper-design` after installing
its dependencies (and fetching LFS objects if this execution needs them):

```bash
python scripts/shellflow.py export-map examples/maps/obstacle-course.scene.json \
  --output outputs/obstacle-course.map
python scripts/shellflow.py verify-package outputs/obstacle-course.map \
  --profile robots/jumper/profile.json --mujoco
```

To replay a trained policy in this world, return to the root training checkout, activate its
environment, and supply a checkpoint and the separately produced map:

```bash
cd ../jumper
source .venv/bin/activate
python scripts/play.py --task jumper.tripod --checkpoint <checkpoint.pt> \
  --scene ../jumper-design/outputs/obstacle-course.map
```

Imported maps are replay-only: `train.py` refuses them because their terrain and props are not
replicated across parallel environments. Built-in training scenes remain available. Train has no
`.skin` importer, so a Design skin is not a training input.

## Current integration boundaries

- The scene importer accepts `kk-scene-package/1` and `/2` for replay. It loads the world and
  spawn while keeping the training task's robot and controller; it does not substitute the map's
  robot.
- Arbitrary-map training and direct skin loading in Train are not implemented.
- Do not describe assisted simulation forces or digital geometry checks as real-robot validation.
  Rendering, package checks and replay are distinct from physical fit and hardware validation.
- The historical baseline diff is a dated comparison made on 2026-09-29 against Design commit
  `be74e0f2`: `LM_J0_joint` had lower limit `-0.75` in Design versus `+0.75` in the Train
  source URDF.
  That dated comparison is evidence for the comparison only; it does not pin the current Design
  runtime or authorize treating a fresh `main` checkout as that commit.

This workflow keeps training and Design independently versioned and independently installable.
