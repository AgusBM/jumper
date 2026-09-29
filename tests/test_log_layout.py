"""The training log directory structure: `logs/<model>/<task>/<date-time>`.

Each of the three segments has its own source, and nowhere is the full path
hardcoded:

    logs/<model>      `tasks.paths.log_root_for`, decided by `--model`
    <task>            experiment_name in each task's `rl_cfg.py` (= the task id)
    <date-time>       generated as a timestamp by mjlab's train entry point

mjlab composes `<log_root>/<experiment_name>/<timestamp>` (see `log_root_path` in
`rl/mjlab/scripts/train.py`), so setting `--log-root` to `logs/<model>` produces
the structure wanted -- **with no change to the vendored code**.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tasks
from tasks.paths import LOGS_DIRNAME, NO_MODEL, log_root_for

REPO = Path(__file__).resolve().parents[1]


# ── logs/<model> ────────────────────────────────────────────────────────


def test_root_is_logs_slash_model_stem() -> None:
    assert log_root_for("/somewhere/jumper.xml") == f"{LOGS_DIRNAME}/jumper"


def test_root_accepts_path_objects() -> None:
    assert log_root_for(Path("assets/smoke/smoke_biped.xml")) == f"{LOGS_DIRNAME}/smoke_biped"


def test_no_asset_keeps_the_depth() -> None:
    """A task with no asset still occupies a segment, or globs like
    `logs/*/<task>/` stop matching."""
    assert log_root_for(None) == f"{LOGS_DIRNAME}/{NO_MODEL}"


def test_different_models_do_not_share_a_subtree() -> None:
    """Training one task against different models must split into separate
    subtrees -- which is why model is the outermost segment."""
    assert log_root_for("a/jumper.xml") != log_root_for("b/jumper_v2.xml")


@pytest.mark.parametrize("spec", tasks.all_specs(), ids=lambda s: s.id)
def test_every_task_yields_a_three_segment_path(spec: tasks.TaskSpec) -> None:
    """Every registered task's default model must compose a full three-segment path."""
    full = f"{log_root_for(spec.resolve_asset(None))}/{spec.id}"
    assert full.count("/") == 2, f"{spec.id} did not compose two levels: {full}"
    assert full.startswith(f"{LOGS_DIRNAME}/")


# ── The second segment comes from each task's rl_cfg ──────────────────────


@pytest.mark.parametrize("spec", tasks.all_specs(), ids=lambda s: s.id)
def test_experiment_name_is_the_task_id(spec: tasks.TaskSpec) -> None:
    """The middle segment must be the task id, or log directories do not line up
    with `--task`.

    Read from source rather than building the config: `agent_cfg()` would pull in
    mjlab, and this convention is a textual one.
    """
    rl_cfg = REPO / "tasks" / spec.id.replace(".", "/") / "rl_cfg.py"
    text = rl_cfg.read_text(encoding="utf-8")
    if "experiment_name" not in text:
        pytest.skip(f"{spec.id}'s rl_cfg is a placeholder")
    assert f'experiment_name="{spec.id}"' in text, (
        f"{spec.id}'s experiment_name does not match its task id"
    )


# ── train.py composes the whole path ──────────────────────────────────────


def test_train_composes_all_three_segments() -> None:
    """`scripts/train.py` builds `logs/<model>/<task>/<timestamp>` itself.

    The log root used to be injected as `--log-root` into an argv handed to
    mjlab's entry point. Now train.py owns the whole path, so this checks the
    composition at its source rather than through an argv fixture.
    """
    train = (REPO / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "log_root_for(asset)" in train, "the first segment must come from log_root_for"
    assert "/ spec.id /" in train, "the second segment must be the task id"
    assert "_timestamp()" in train, "the third segment must be a timestamp"


def test_no_entry_point_lives_outside_scripts() -> None:
    """Every command-line entry point is a script directly under `scripts/`.

    There used to be a second set under `scripts/bridge/` that reached mjlab's own
    entry points, so "how do I replay a policy" had two answers and only one of
    them went through this repository's backend resolution. Now there is one.

    The list is closed, and `deploy.py` was added to it deliberately rather than
    by loosening the rule. What the rule forbids is **two answers to one
    question**; a fourth entry is only a problem if it duplicates one of the other
    three. It does not: `train`, `play` and `export` each build an environment and
    take `--task`, and `deploy` builds the deployment-side controller and reads
    no environment at all. Anything that *would* have gone in `export.py` --
    another thing a checkpoint turns into -- still belongs there.
    """
    assert not (REPO / "scripts" / "bridge").exists(), (
        "scripts/bridge/ is retired; entry points live directly under scripts/"
    )
    entries = {p.name for p in (REPO / "scripts").glob("*.py")}
    assert entries == {"_cli.py", "train.py", "play.py", "export.py", "deploy.py"}, entries


def test_scripts_do_not_know_any_task_s_vocabulary() -> None:
    """Nothing under `scripts/` changes when a task is added.

    That is the rule the whole entry-point layout exists for -- one parser, three
    scripts, and a registry that turns an id into an import -- and it is stated in
    CLAUDE.md. It is also the rule that was quietly broken: `play.py` reached into
    `command_manager.get_command("twist")` to print the commanded speed on its
    replay readout, which is a velocity task's vocabulary sitting in a file that is
    supposed to have none.

    It failed the way a layering violation does. `jumper.swing` has no command, so
    `get_command` returned None rather than raising, the `except (AttributeError,
    KeyError)` written for exactly that case did not catch the `TypeError`, and
    replay died on the first frame for the first task that was not about velocity.
    Adding `TypeError` to that tuple would have silenced it and left the violation
    in place. What fixed it was `tasks.load_play_status`: the script asks the task
    what to print, and the task answers.

    **Code lines only.** The docstrings in `play.py` and `registry.py` quote the
    offending name while explaining why it is not there, and searching the whole
    file takes the explanation for the offence -- the same trap `test_seam.py`'s
    `code()` helper exists for.
    """
    import re

    scripts = sorted((REPO / "scripts").glob("*.py"))

    def code(path) -> str:
        """The lines that run: no docstrings, no comments."""
        source = path.read_text(encoding="utf-8")
        source = re.sub(r'"""[\s\S]*?"""', "", source)
        source = re.sub(r"'''[\s\S]*?'''", "", source)
        return "\n".join(
            line
            for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )

    # **The shape of the violation, not a word list.** A script comes to know a
    # task by reaching into a manager with a **literal**: `get_command("twist")` is
    # the one that shipped, and the same move against the reward, termination or
    # curriculum managers would be the same mistake.
    #
    # Two narrower passes were tried first and both were wrong in an instructive
    # direction. A word list caught `_cli.py` describing "the hexapod's 44 MiB per
    # copy" in an argparse help string -- prose for a user, not a dependency. Then
    # `get_term(` on its own caught `export.py`'s
    # `action_manager.get_term(act_terms[0])`, which looks the name **up** from the
    # manager rather than spelling it, and is exactly the pattern that is allowed.
    # The literal is what separates the two.
    reach = re.compile(r"""(get_command|get_term)\(\s*["']""")
    for script in scripts:
        found = reach.search(code(script))
        assert found is None, (
            f"{script.name} calls {found.group(0)!r} -- a manager indexed by a name "
            f"spelled here, which only one task defines. Look it up from the "
            f"manager, or ask the task, the way `tasks.load_play_status` does"
        )

    # And the task ids themselves, spelled in full. The **last segment alone** is
    # not enough of a name to go on: matching "flat" catches `export.py`'s
    # `flatten`, and matching "jumper" catches `_cli.py` describing "the hexapod's
    # 44 MiB per copy" in help text. Both are words, neither is a dependency.
    for script in scripts:
        running = code(script)
        hits = sorted(t for t in tasks.list_ids() if t in running)
        assert not hits, (
            f"{script.name} names the task(s) {hits}. A script that names one "
            f"cannot be true of the next"
        )
