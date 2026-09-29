"""Parsing `.env` and its precedence chain.

The chain rests on one rule: **`load_dotenv` does not overwrite existing
environment variables**. argparse defaults then read from `os.environ`, so

    command line > shell environment > .env.local > .env > built-in default

follows for free, with no need to ask "did the user give this explicitly" -- which
is exactly the kind of judgement a config layer gets wrong.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from mjrl.dotenv import env_choice, env_default, env_flag, load_dotenv, parse_dotenv

REPO = Path(__file__).resolve().parents[1]


# ── Parsing ───────────────────────────────────────────────────────────────


def test_basic_key_value() -> None:
    assert parse_dotenv("A=1\nB=two\n") == {"A": "1", "B": "two"}


def test_comments_and_blank_lines_ignored() -> None:
    text = "# a comment\n\n  \nA=1\n  # an indented comment\n"
    assert parse_dotenv(text) == {"A": "1"}


def test_export_prefix_accepted() -> None:
    assert parse_dotenv("export A=1\n") == {"A": "1"}


def test_inline_comment_stripped() -> None:
    assert parse_dotenv("A=1  # explanation\n") == {"A": "1"}


def test_quoted_value_keeps_spaces_and_hash() -> None:
    """A `#` inside quotes is not a comment. A path containing one is rare, but
    silent truncation would be very hard to trace."""
    assert parse_dotenv('A="a b # c"\n') == {"A": "a b # c"}
    assert parse_dotenv("A='x y'\n") == {"A": "x y"}


def test_empty_value_is_kept_as_empty_string() -> None:
    assert parse_dotenv("A=\n") == {"A": ""}


def test_line_without_equals_is_ignored() -> None:
    """Tolerate stray prose, so one line of notes does not stop every entry point
    from starting."""
    assert parse_dotenv("this is a sentence\nA=1\n") == {"A": "1"}


def test_value_containing_equals_is_not_split_again() -> None:
    assert parse_dotenv("A=k=v\n") == {"A": "k=v"}


# ── Precedence ────────────────────────────────────────────────────────────


def test_existing_env_var_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value exported explicitly in the shell must beat .env -- the whole
    precedence chain rests on this."""
    monkeypatch.setenv("MJRL_TEST_KEY", "from_shell")
    env = tmp_path / ".env"
    env.write_text("MJRL_TEST_KEY=from_file\n", encoding="utf-8")
    applied = load_dotenv(env)
    assert "MJRL_TEST_KEY" not in applied
    assert os.environ["MJRL_TEST_KEY"] == "from_shell"


def test_later_file_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """.env.local comes after .env and therefore overrides it."""
    monkeypatch.delenv("MJRL_TEST_KEY", raising=False)
    (tmp_path / ".env").write_text("MJRL_TEST_KEY=base\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text("MJRL_TEST_KEY=local\n", encoding="utf-8")
    load_dotenv(tmp_path / ".env", tmp_path / ".env.local")
    assert os.environ["MJRL_TEST_KEY"] == "local"


def test_missing_file_is_not_an_error(tmp_path: Path) -> None:
    """.env.local usually does not exist, and a missing file must not crash an
    entry point."""
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_empty_value_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """`MJRL_MODEL=` in `.env` means "leave it empty, use the built-in default",
    not the empty string as a value."""
    monkeypatch.setenv("MJRL_TEST_KEY", "   ")
    assert env_default("MJRL_TEST_KEY") is None
    assert env_default("MJRL_TEST_KEY", "fallback") == "fallback"


# ── Switches ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", ["on", "ON", "On", "true", "TRUE", "yes", "1", " on "])
def test_every_spelling_of_on(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MJRL_TEST_FLAG", raw)
    assert env_flag("MJRL_TEST_FLAG", fallback=False) is True


@pytest.mark.parametrize("raw", ["off", "OFF", "Off", "false", "FALSE", "no", "0", " off "])
def test_every_spelling_of_off(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """`MJRL_TENSORBOARD=OFF` must turn it off.

    This was the whole bug: the test was `!= "off"`, so every spelling but the
    lowercase one left the feature **on** -- and there is nothing in the output to
    say the setting was ignored, so it reads as a switch that does not work.
    """
    monkeypatch.setenv("MJRL_TEST_FLAG", raw)
    assert env_flag("MJRL_TEST_FLAG", fallback=True) is False


def test_unset_or_empty_takes_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """`MJRL_TENSORBOARD=` in `.env` means "use the built-in default", exactly as
    an empty `MJRL_MODEL` does."""
    monkeypatch.delenv("MJRL_TEST_FLAG", raising=False)
    assert env_flag("MJRL_TEST_FLAG", fallback=True) is True
    monkeypatch.setenv("MJRL_TEST_FLAG", "   ")
    assert env_flag("MJRL_TEST_FLAG", fallback=False) is False


@pytest.mark.parametrize("raw", ["of", "nope", "onn", "2", "-"])
def test_anything_else_raises_rather_than_falling_back(
    raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value that is not a switch must not quietly become the default.

    `MJRL_TENSORBOARD=of` reading as "on" is invisible: the feature behaves
    normally and nothing mentions the setting. Erroring names the value, the way
    a non-integer `MJRL_NUM_ENVS` does.
    """
    monkeypatch.setenv("MJRL_TEST_FLAG", raw)
    with pytest.raises(ValueError, match="MJRL_TEST_FLAG"):
        env_flag("MJRL_TEST_FLAG", fallback=True)


# ── One of a fixed set of values ──────────────────────────────────────────


def test_a_choice_is_matched_case_insensitively(monkeypatch: pytest.MonkeyPatch) -> None:
    """And comes back in the spelling the choices use, so callers can compare it."""
    monkeypatch.setenv("MJRL_TEST_CHOICE", "OFF")
    assert env_choice("MJRL_TEST_CHOICE", ("auto", "on", "off"), "auto") == "off"
    monkeypatch.setenv("MJRL_TEST_CHOICE", " Auto ")
    assert env_choice("MJRL_TEST_CHOICE", ("auto", "on", "off"), "auto") == "auto"


def test_a_choice_outside_the_set_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The value must not travel on to whatever consumes the argument.

    `MJRL_STRIP_VISUAL=gpu` reaching the backend is not a config error any more --
    it is a `KeyError` or a wrong run, somewhere with no mention of `.env`.
    """
    monkeypatch.setenv("MJRL_TEST_CHOICE", "gpu")
    with pytest.raises(ValueError, match="auto, on, off"):
        env_choice("MJRL_TEST_CHOICE", ("auto", "on", "off"), "auto")


def test_an_unset_choice_takes_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MJRL_TEST_CHOICE", raising=False)
    assert env_choice("MJRL_TEST_CHOICE", ("auto", "on"), "auto") == "auto"


# ── Reaching argparse: where these defaults actually land ─────────────────


def _parse(
    argv: list[str],
    env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    replay: bool = False,
):
    """Parse arguments the way an entry point does, and hand back `_cli` with them.

    `replay` builds play's parser rather than train's.

    `scripts/` is not a package and no test may put it on `sys.path`
    (`test_no_sys_path_mutation` in test_layout.py), so `_cli` is loaded from its
    file. Going through the real parser is the point: **argparse checks `choices`
    only for values typed on the command line, never for a default**, so a bad
    value in `.env` is caught here or not at all.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location("_cli_under_test", REPO / "scripts" / "_cli.py")
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    prog = "play.py" if replay else "train.py"
    return cli, cli.build_parser(prog, "", replay=replay).parse_args(argv)


def test_a_choice_from_dotenv_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    """`MJRL_STRIP_VISUAL=OFF` used to reach `resolve_all` verbatim and die there
    as `KeyError: 'OFF'`, with `.env` nowhere in the traceback."""
    cli, args = _parse([], {"MJRL_STRIP_VISUAL": "OFF", "MJRL_BACKEND": "WARP"}, monkeypatch)
    assert args.strip_visual == "off"
    assert args.backend == "warp"
    # The value has to be usable by what consumes it, which is the lookup table.
    assert cli.STRIP_VISUAL[args.strip_visual] is False


@pytest.mark.parametrize(
    "env", [{"MJRL_STRIP_VISUAL": "nope"}, {"MJRL_BACKEND": "gpu"}]
)
def test_a_bad_choice_in_dotenv_stops_the_run(
    env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit):
        _parse([], env, monkeypatch)


def test_the_command_line_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    _, args = _parse(["--strip-visual", "on"], {"MJRL_STRIP_VISUAL": "OFF"}, monkeypatch)
    assert args.strip_visual == "on"


def test_the_choices_and_the_lookup_table_are_one_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What argparse accepts and what `resolve` is handed must not drift apart.

    They were two separate literals, so a fourth `--strip-visual` state added to
    the argument alone would pass the parser and be a `KeyError` in `resolve_all`.
    """
    cli, _ = _parse([], {}, monkeypatch)
    action = next(a for a in cli.build_parser("t", "")._actions if a.dest == "strip_visual")
    assert set(action.choices) == set(cli.STRIP_VISUAL)


# ── Two counts under one flag ─────────────────────────────────────────────


def test_play_does_not_take_trainings_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """`MJRL_NUM_ENVS` is PPO's batch size, and a replay does not read it.

    Sharing one key, the repository's 4096 built 4096 native CPU environments to
    replay an app's controller that reads one, and never got past building them.
    `MJRL_PLAY_NUM_ENVS` is set empty so that this pins the built-in 1 rather than
    whatever `.env` says. Train, parsed from the same environment, is the
    control: it still gets 4096, so the key was not simply dropped.
    """
    env = {"MJRL_NUM_ENVS": "4096", "MJRL_PLAY_NUM_ENVS": ""}
    _, play = _parse([], env, monkeypatch, replay=True)
    _, train = _parse([], env, monkeypatch)
    assert play.num_envs == 1
    assert train.num_envs == 4096


def test_play_reads_its_own_key_and_the_command_line_still_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = {"MJRL_NUM_ENVS": "4096", "MJRL_PLAY_NUM_ENVS": "16"}
    _, play = _parse([], env, monkeypatch, replay=True)
    _, train = _parse([], env, monkeypatch)
    _, typed = _parse(["--num_envs", "8"], env, monkeypatch, replay=True)
    assert (play.num_envs, train.num_envs, typed.num_envs) == (16, 4096, 8)


def test_only_training_is_warned_about_its_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`resolve()` decides the warning; this pins that `resolve_all` tells it which
    entry point is asking. A flag handed to a function and never passed on is the
    failure this checks for, so train with the same count is the control."""
    argv = ["--task", "jumper.flat", "--backend", "native", "--device", "cpu",
            "--num_envs", "8"]
    cli, play = _parse(argv, {}, monkeypatch, replay=True)
    _, train = _parse(argv, {}, monkeypatch)
    _, play_res, _ = cli.resolve_all(play)
    _, train_res, _ = cli.resolve_all(train, for_training=True)
    assert play_res.num_envs == train_res.num_envs == 8
    assert not any("batch size" in n for n in play_res.notes), play_res.notes
    assert any("batch size" in n for n in train_res.notes), train_res.notes


# ── The repository's own .env ─────────────────────────────────────────────


def test_repo_dotenv_is_present_and_parses() -> None:
    """`.env` is committed and is the single source of project-level defaults."""
    env = REPO / ".env"
    assert env.is_file(), "the repository root should have a .env"
    parse_dotenv(env.read_text(encoding="utf-8"))


def test_repo_dotenv_declares_every_cli_backed_key() -> None:
    """Everything the CLI can set should have an entry in `.env`, or that file is
    incomplete."""
    keys = set(parse_dotenv((REPO / ".env").read_text(encoding="utf-8")))
    expected = {
        "MJRL_TASK",
        "MJRL_MODEL",
        "MJRL_BACKEND",
        "MJRL_DEVICE",
        "MJRL_NUM_ENVS",
        "MJRL_PLAY_NUM_ENVS",
        "MJRL_CPU_THREADS",
        "MJRL_TENSORBOARD",
        "MJRL_TB_PORT",
    }
    assert expected <= keys, f"`.env` is missing: {sorted(expected - keys)}"


def test_repo_dotenv_task_is_registered() -> None:
    """The default task in `.env` must really exist, or every entry point errors
    out of the box."""
    import tasks

    task = parse_dotenv((REPO / ".env").read_text(encoding="utf-8")).get("MJRL_TASK", "")
    if task.strip():
        tasks.get(task.strip())  # raises KeyError when not registered
