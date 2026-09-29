---
name: new-task
description: Create a new training task in this repository -- the directory, the registration, the config skeleton and the operator controls, with every control name taken from the gamepad dictionary. Use this whenever someone wants a new task: "创建一个 task", "新建一个任务", "add a task for a bounding gait", "I want a new locomotion task that jumps", "make a task like tripod but slower", or a request that names a gait, a behaviour or a robot and asks for training to be set up for it. Reach for it before hand-copying an existing task directory, because the failures here are quiet -- a task missing its import line does not exist and `--list` simply does not show it, a controls file with an invented stick name loads and drives nothing, and a parameter copied from a sibling is a decision nobody made for this task.
---

# Creating a task

`docs/USAGE.md` is the reference for what a task *is*; `CLAUDE.md`'s "Working
rules" table is where each parameter belongs. **This skill does not restate
them.** It gives the order, and it owns the one step that must not be done by
hand.

## The one thing to understand first

**A task is a directory whose name is its id.** `tasks/jumper/tripod/` is
`"jumper.tripod"`, and `load_env_cfg` turns the id straight into an import. There
is no registry of config values to fill in, and nothing to wire up beyond the
files below plus one import line.

That import line is the trap. Forget it and the task **silently does not
exist** -- no error, `--list` just does not show it. `tests/test_registry.py`
exists for exactly that.

## Procedure

### 1. Settle what differs before writing anything

A new locomotion task must differ from its siblings in **the gait and the speed
ceiling the gait implies, and in nothing else**. `tests/test_task_parity.py`
fails on any other difference and tells you to justify it in `GAIT_EXEMPT` /
`CEILING_EXEMPT`.

So ask, and do not guess:

- what the gait is, and what makes it different from `flat` / `tripod` /
  `ripple` / `tetrapod`
- the speed ceiling it implies
- **which command axes the policy observes** -- normally the three velocity
  axes, and this is the answer step 3 needs
- which stick and which keys drive each axis -- the pad and the keyboard are two
  paths and a controls file binds both, so an axis left off the keyboard needs a
  reason

If the person named buttons ("用 LB 切进去", "A 按下起跳"), write those down
verbatim. They are inputs to steps 3 and 5 and they must not be paraphrased.

### 2. The directory

```
tasks/jumper/<name>/
  __init__.py     register(...) and nothing else -- no config imports
  env_cfg.py      every parameter this task owns
  rl_cfg.py       hyper-parameters
  controls.yaml   step 3 writes this
  mdp/            only the terms this task alone uses
```

`__init__.py` must not import a config module. `--list` has to work on a machine
with no simulation dependencies, and after it runs there must not be one heavy
import in `sys.modules`.

Then the line that is easy to forget, in `tasks/__init__.py`:

```python
from .jumper import <name>  # noqa: F401,E402
```

### 3. The operator controls -- use the `controls` skill

**Invoke the `controls` skill and follow it.** It owns the dictionary and both
files that consume it, and this step does not restate them -- a second copy of
that list is the failure the dictionary exists to prevent.

Hand it what step 1 collected: the command axes, and any controls the person
named verbatim. It runs

```bash
python3 .claude/skills/controls/scripts/write_controls.py --task jumper.<name>
```

which reads `controller/vocabulary.json`, refuses a name that is not in it, and
writes the file. Do not hand-write `controls.yaml` and do not copy a sibling's.

### 4. The parameters

`velocity_env_cfg` **raises rather than defaulting** for everything a task must
own: the reward weights, the command ceilings, the foot-lift target. Fill them
in `env_cfg.py`, one at a time, reading the message each raise gives -- they say
why the parameter belongs to the task rather than to `common/`.

Writing the same value as a sibling is fine and is the point. Agreement that is
written down survives one of them changing; agreement by inheritance is four
tasks running a decision none of them made, which is not hypothetical --
`jumper.tetrapod` ran for months on a command ladder nobody had picked for it.

Anything that is a fact about the **hardware** goes in
`tasks/jumper/common/constants.py` instead. Anything that is **mechanism with no
tuning in it** goes in `tasks/jumper/common/`.

### 5. Buttons that switch modes are not this file

`controls.yaml` says how a person *drives* a policy -- which stick and which key
moves which axis. Which button or key *switches into* this task as an FSM mode is
a property of a deploy bundle, not of the task, and lives in
`deploy/manifests.json` as the mode's `pad` and `keys` switches, each with a
gesture and the modes it may be pressed `from`. Same dictionary, different file --
the `controls` skill covers both halves and why neither may say the other's.

### 6. Check it exists and builds

```bash
python scripts/train.py --list                       # it must appear
python scripts/train.py --task jumper.<name> --dry-run # resolves, builds nothing
python -m pytest tests/test_registry.py tests/test_task_parity.py -q
python -c "from pathlib import Path; from tasks.jumper.common.mdp.controls import load_controls; load_controls(Path('tasks/jumper/<name>/controls.yaml'))"
```

`tests/test_controls.py` loads the existing tasks' files only, so the last line is what
parses the new one.

`--list` not showing it means the import line in step 2 is missing. That is the
failure this whole procedure is ordered around.

## Failures worth recognising

| symptom | likely cause |
|---|---|
| `--list` does not show the task | no import line in `tasks/__init__.py` |
| `--list` is slow, or drags in torch | `__init__.py` imports a config module |
| anything about a control binding | the `controls` skill's own table |
| `test_task_parity` fails on a reward | a locomotion task differing in something other than the gait. Either it is a mistake or it belongs in `GAIT_EXEMPT` with a reason |
| `ValueError: ... is required` at startup | a parameter the task must own. The message says which and why -- that is the design, not an omission |

## When something does not match

Do not copy a value from a sibling task to make an error go away. The raise is
telling you a decision has not been made, and a copied number is that decision
made by accident. Read the sibling to understand what the number *is*, then
choose one for this task -- even if you choose the same one.
