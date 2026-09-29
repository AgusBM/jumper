"""`tools/hull_collision.py` -- a decimated hull must only shrink, never grow.

This tool changes **collision geometry**, and getting it wrong raises nothing: a
hull that bulges outward has the robot standing on links that should not touch the
ground, and one that shrinks too far delays contact throughout. So two things are
guarded here:

1. The decimated hull is **contained in** the original (it only shrinks inward),
   and the amount of shrink is computed and reported.
2. Editing the XML is a targeted substitution -- only `*_meshcol`'s `mesh=`
   changes, `*_visual` is untouched, and running twice changes nothing more.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
mujoco = pytest.importorskip("mujoco")

REPO = Path(__file__).resolve().parents[1]


def _load():
    """Load `tools/hull_collision.py` by path.

    `tools/` is not a Python package (it holds standalone scripts, deliberately
    outside the `pip install -e .` package list), so it can only be loaded by file
    path -- the same approach as `test_log_layout.py`.
    """
    path = REPO / "tools" / "hull_collision.py"
    spec = importlib.util.spec_from_file_location("_hull_tool_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_T = _load()
hull_decimate = _T.hull_decimate
main = _T.main
max_inward_shrink = _T.max_inward_shrink
patch_xml = _T.patch_xml
read_stl = _T.read_stl
write_stl = _T.write_stl


def _sphere(n: int = 2000, r: float = 0.5) -> np.ndarray:
    """n points on a sphere -- the decimation error has an analytic bound, which
    makes it easy to check against."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return r * np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )


def test_decimation_only_shrinks() -> None:
    """Every decimated vertex must be an **original vertex**, so the hull can only
    shrink."""
    v = _sphere()
    for k in (16, 64, 256):
        hull = hull_decimate(v, k)
        assert len(hull) <= k
        # Every hull vertex is found bit-for-bit among the original vertices
        for p in hull:
            assert np.abs(v - p).sum(axis=1).min() == 0.0, (
                "decimation invented a point that was not there"
            )


def test_shrink_shrinks_as_k_grows() -> None:
    """Denser direction sampling means less shrink; a full hull shrinks by 0."""
    v = _sphere()
    errs = [max_inward_shrink(v, hull_decimate(v, k)) for k in (16, 64, 256)]
    assert errs[0] > errs[1] > errs[2] > 0, f"shrink is not monotone in k: {errs}"
    # A full hull leaves only the face equations' floating-point rounding
    # (measured 1e-16), orders of magnitude below k=256
    full = max_inward_shrink(v, hull_decimate(v, None))
    assert full < errs[2] * 1e-6


def test_shrink_is_measured_not_assumed() -> None:
    """The shrink is **computed**, checked against a hand-built case with a known
    answer.

    A cube's eight corners plus a spike protruding from a face centre. Sampling only
    the six +/-x, +/-y, +/-z directions necessarily misses the spike (along +z it is
    lower than a corner), so the shrink is exactly how far the spike rises above the
    top face.
    """
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    spike = np.array([[0.0, 0.0, 1.3]])
    v = np.vstack([cube, spike])
    hull = np.asarray(cube)  # as if decimation had missed the spike
    assert max_inward_shrink(v, hull) == pytest.approx(0.3, abs=1e-9)


def test_stl_roundtrip(tmp_path: Path) -> None:
    """Write and read back with the vertex set unchanged -- otherwise the whole
    pipeline's coordinates cannot be trusted."""
    v = _sphere(300, r=0.2)
    hull = hull_decimate(v, 64)
    out = tmp_path / "h.STL"
    ntri = write_stl(out, hull)
    assert ntri > 0
    back = read_stl(out)
    # STL stores float32, so compare at 1e-6
    for p in hull:
        assert np.abs(back - p).sum(axis=1).min() < 1e-6


_XML = """<mujoco model="t">
  <asset>
    <mesh name="link" file="link.STL"/>
  </asset>
  <worldbody>
    <body name="b">
      <geom name="link_visual" type="mesh" mesh="link" contype="0" conaffinity="0"/>
      <geom name="link_meshcol" type="mesh" mesh="link"/>
    </body>
  </worldbody>
</mujoco>
"""


def _make_asset(tmp_path: Path) -> Path:
    (tmp_path / "meshes").mkdir()
    write_stl(tmp_path / "meshes" / "link.STL", hull_decimate(_sphere(500, 0.3), None))
    xml = tmp_path / "m.xml"
    xml.write_text(_XML.replace('<asset>', '<compiler meshdir="meshes"/>\n  <asset>'))
    return xml


def test_patch_only_touches_the_collision_geom(tmp_path: Path) -> None:
    """`*_visual` must be left as it was -- it still displays the original mesh."""
    xml = _make_asset(tmp_path)
    n = patch_xml(xml, [("link_meshcol", "link", "link_col")], "_col")
    text = xml.read_text()
    assert n == 1
    assert 'name="link_visual" type="mesh" mesh="link"' in text, (
        "the appearance geom was modified"
    )
    assert 'name="link_meshcol" type="mesh" mesh="link_col"' in text
    assert '<mesh name="link_col" file="link_col.STL"/>' in text


def test_apply_is_idempotent(tmp_path: Path, capsys) -> None:
    """A second run must change nothing -- the new mesh names end in
    `--out-suffix` and are skipped."""
    xml = _make_asset(tmp_path)
    argv = ["--model", str(xml), "--apply", "-k", "32"]
    assert main(argv) == 0
    first = xml.read_text()
    capsys.readouterr()

    assert main(argv) == 0
    assert "Nothing to process" in capsys.readouterr().out
    assert xml.read_text() == first, "the second run edited the XML again"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    """Without --apply, not one byte should reach disk."""
    xml = _make_asset(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert main(["--model", str(xml)]) == 0
    after = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after, "a report-only run touched files"
