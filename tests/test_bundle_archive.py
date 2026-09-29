"""The `.app` beside a bundle directory: what somebody actually carries away.

A bundle is a directory on this machine and one file everywhere else --
a browser's file picker, an upload form, a phone. The archive is that file, and
three things about it are worth pinning because none produces an error:

- **where `bundle.json` sits.** A consumer opens the archive and looks at the
  top for it. Zipping the *directory* rather than its contents puts everything
  one level down, and the bundle reads as one with no manifest -- refused, with
  a message about a missing file rather than about an extra directory.
- **what it is called, and what it is.** `jumper.app`, and a zip all the same.
  A browser's file picker offers only the names its `accept` lists, so a
  name the consumer was not told about is a file that is simply not there.
- **whether two builds of one bundle agree.** Every file inside is already
  digested in `bundle.json`; an archive whose bytes moved on every build, for
  no reason but a timestamp, would be the only artefact here that cannot be
  held against the one shipped last week.

No toolchain: this builds a directory and zips it, so it runs on a machine that
only trains.
"""

from __future__ import annotations

import importlib.util
import os
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/deploy.py"


def deploy():
    """`scripts/deploy.py` as a module.

    By path rather than by import: `scripts/` is not a package, and putting it
    on `sys.path` to reach one function would put `_cli` and the rest there too.
    """
    spec = importlib.util.spec_from_file_location("deploy_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def host(tmp_path: Path) -> Path:
    """A bundle directory in miniature, nested file and all."""
    out = tmp_path / "jumper"
    (out / "sub").mkdir(parents=True)
    (out / "bundle.json").write_text('{"schema": "kk-policy-bundle/1"}\n', "utf-8")
    (out / "locomotion.onnx").write_bytes(b"\x00\x01\x02")
    (out / "sub" / "trajectory.json").write_text("[]\n", "utf-8")
    return out


def test_the_manifest_is_at_the_top_of_the_archive(host: Path) -> None:
    archive = deploy().zip_bundle(host)
    with zipfile.ZipFile(archive) as bag:
        names = bag.namelist()
    assert "bundle.json" in names, (
        f"a consumer opens the archive and looks at the top for the manifest; "
        f"this one has {names}"
    )
    assert not any(n.startswith("jumper/") for n in names), (
        f"the bundle directory was zipped instead of its contents: {names}"
    )
    # ...everything else came along, nesting intact, and in a fixed order:
    # `rglob` promises none, and two builds laying the same files down in two
    # orders are two different archives of one bundle.
    assert names == ["bundle.json", "locomotion.onnx", "sub/trajectory.json"]


def test_the_archive_is_called_app_and_is_still_a_zip(host: Path) -> None:
    """The name moved from `.zip` to `.app`; the format did not.

    Neither half errors when it is wrong. A consumer that chooses the file by
    name -- the web simulator's picker, through its `accept` list -- hides one
    it does not recognise, and one that opens it reads a zip: the simulator's
    reader is `fflate`, which knows nothing about `.app`.
    """
    archive = deploy().zip_bundle(host)
    assert archive == host.with_name("jumper.app"), (
        f"a consumer that picks the bundle by name looks for jumper.app; "
        f"this build wrote {archive.name}"
    )
    assert zipfile.is_zipfile(archive), "the name changed and so did the format"


def test_the_directory_survives_being_archived(host: Path) -> None:
    """Both, not either.

    Everything on this machine reads the directory -- `--check-reference`, the
    tests, the bundler's own verification. An archive that replaced it would
    make all of them unpack first.
    """
    deploy().zip_bundle(host)
    assert (host / "bundle.json").is_file()
    assert (host / "sub" / "trajectory.json").is_file()


def test_two_archives_of_one_bundle_are_the_same_bytes(host: Path, tmp_path: Path) -> None:
    mod = deploy()
    first = mod.zip_bundle(host).read_bytes()
    (tmp_path / "jumper.app").unlink()
    # Same bytes, a wholly different mtime. `touch()` is not enough and this
    # test passed with it: a zip timestamp is a DOS one, two seconds wide, and
    # two calls a line apart land in the same bucket. What is being asked is
    # whether two builds a week apart agree, so the clock has to move like it.
    os.utime(host / "locomotion.onnx", (0, 1_000_000_000))
    second = mod.zip_bundle(host).read_bytes()
    assert first == second, (
        "two archives of one unchanged directory differ. Something in them is "
        "not the bundle -- a timestamp, or the order the filesystem happened to "
        "list the files in"
    )


def test_a_changed_bundle_is_a_changed_archive(host: Path, tmp_path: Path) -> None:
    """The control group.

    Every assertion above also passes for a function that writes an empty file,
    or the same file every time. This is the one that fails if it does.
    """
    mod = deploy()
    before = mod.zip_bundle(host).read_bytes()
    (tmp_path / "jumper.app").unlink()
    (host / "locomotion.onnx").write_bytes(b"\x00\x01\x03")
    assert mod.zip_bundle(host).read_bytes() != before
