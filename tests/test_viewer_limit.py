"""How many environments the live viewer draws -- the cap has to be computed
rather than left to MuJoCo to silently drop.

Without a cap there are two traps, **neither of which raises**:

1. `launch_passive` gives `user_scn` a capacity of `MAX_GEOM` (100000). Once
   full, `mjv_addGeoms` prints **one** WARNING (MuJoCo prints each one only
   once, and the training log scrolls it away) and then silently discards the
   rest. Measured with warp + 4096 environments: 253890 geoms expected, 100000
   arrived -- 1613 environments actually drawn (39%), while the picture just
   shows "a field of robots" with no sign that any are missing.
2. Drawing time grows with the number drawn. Measured on native when drawing
   still hung off `sim.step()` and came straight out of training: 46.6 ms per
   frame at 256 environments (140% of the 30 fps budget), 185.6 ms at 1024
   (557%). It runs on the viewer's own thread now, where the same numbers are a
   slideshow rather than a slower simulation -- and the copy each frame starts
   from is still taken on the simulation thread.
"""

from __future__ import annotations

import types

import pytest

mujoco = pytest.importorskip("mujoco")

XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="b" pos="0 0 1"><freejoint/>
      <geom name="s1" type="sphere" size="0.2"/>
      <geom name="s2" type="sphere" size="0.1" pos="0.3 0 0"/></body>
  </worldbody>
</mujoco>
"""


class _FakeSim:
    """Only the little bit of interface `_resolve_draw_limit` uses."""

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs


def make_viewer(num_envs: int, fps: float = 30.0, origins=None):
    from mjrl.viewer.live import LiveViewer

    v = LiveViewer(_FakeSim(num_envs), env_index=0, fps=fps, env_origins=origins)
    model = mujoco.MjModel.from_xml_string(XML)
    v._scratch = mujoco.MjData(model)
    v._vopt = mujoco.MjvOption()
    v._pert = mujoco.MjvPerturb()
    v._catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC.value
    return v, model


def test_max_geom_is_read_from_upstream() -> None:
    """The capacity must be read from `mujoco.viewer`, not hard-coded.

    Too small and it draws fewer for no reason; too large and it is back to
    silent truncation -- wrong in both directions, and neither raises.
    """
    import mujoco.viewer  # noqa: PLC0415  -- the submodule needs an explicit import

    from mjrl.viewer.live import _max_geom

    upstream = int(mujoco.viewer._Simulate.MAX_GEOM)
    assert _max_geom() == upstream


def test_all_envs_is_the_default() -> None:
    """Draw every environment by default (subject to the cap), no switch needed."""
    import inspect

    from mjrl.viewer.live import LiveViewer

    sig = inspect.signature(LiveViewer.__init__)
    assert sig.parameters["show_all_envs"].default is True


def test_small_scene_draws_every_env() -> None:
    """With few environments, draw them all -- no unexplained reduction."""
    v, model = make_viewer(8)
    v._resolve_draw_limit(model)
    assert v._draw_limit == 7, "all 7 besides the followed one should be drawn"
    assert sorted(v._draw_order) == [1, 2, 3, 4, 5, 6, 7]


def test_env_count_is_capped_at_the_fixed_limit() -> None:
    """Above the cap, draw exactly the cap, and say so.

    The cap is a fixed number rather than adapted to frame time: an adaptive
    number floats with the machine, the backend and contact density, so the same
    command draws a different count on two runs -- which makes "what am I looking
    at" harder to answer, not easier.
    """
    from mjrl.viewer.live import _MAX_DRAW_ENVS

    v, model = make_viewer(4096)
    v._resolve_draw_limit(model)
    assert v._draw_limit + 1 == _MAX_DRAW_ENVS, (
        "the cap counts the camera-followed one too"
    )
    assert len(v._draw_order) == _MAX_DRAW_ENVS - 1


def test_exactly_at_the_limit_draws_everything() -> None:
    """Exactly at the cap, draw them all -- not one fewer."""
    from mjrl.viewer.live import _MAX_DRAW_ENVS

    v, model = make_viewer(_MAX_DRAW_ENVS)
    v._resolve_draw_limit(model)
    assert v._draw_limit + 1 == _MAX_DRAW_ENVS


def test_geom_capacity_still_guards_below_the_limit() -> None:
    """Geom capacity remains the backstop: a sufficiently complex robot could
    fill 100000 with 128 environments.

    If that happens the limit must come from capacity, rather than going back to
    silent dropping.
    """
    from mjrl.viewer.live import _max_geom

    v, model = make_viewer(4096)
    v._resolve_draw_limit(model)
    assert v._draw_limit * v._geoms_per_env <= _max_geom(), "drew past the capacity"


def test_viewer_thread_is_captured_for_joining() -> None:
    """The render thread `launch_passive` starts has to be captured.

    It is a **daemon thread** and is not handed back. On exit, atexit runs LIFO:
    first mujoco's `exit_simulate` (which only signals, it does not wait), then
    glfw's `terminate()` -- nobody waits for the thread to finish, so GLFW is
    torn down while it is still running: **segfault**.

    Measured (before the fix): exit code 0 with no window, 139 as soon as one is
    open, even drawing a single environment. This is a pre-existing problem, but
    with the viewer on by default every training run would end in 139.
    """
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "rl" / "mjrl" / "viewer" / "live.py"
    ).read_text(encoding="utf-8")
    start = src.split("def start(", 1)[1].split("\n    def ", 1)[0]
    assert "threading.enumerate()" in start, "the render thread was not captured"
    assert "atexit.register(self.stop)" in start, "stop() needs the atexit backstop too"
    stop = src.split("def stop(", 1)[1].split("\n    def ", 1)[0]
    assert ".join(" in stop, (
        "stop() must wait for the render thread; close() only signals"
    )


def test_warp_path_takes_a_frame_in_one_transfer(monkeypatch) -> None:
    """On warp, taking a frame must move **everything at once** -- the followed
    environment and every one drawn beside it -- not a `.cpu()` per field and
    environment.

    Every `.cpu()` is a synchronisation as well as a copy, and they are paid on the
    simulation thread. 128 environments one at a time was 254 small transfers,
    dominated by launch overhead; batching per field took a frame from 7.70 to
    5.49 ms (29% off) with the drawn geom positions identical bit for bit.

    Counted rather than read off the source: this used to check for one spelling
    of the batched call, and the batching moved to a different function when
    drawing moved to its own thread.
    """
    import torch

    from mjrl.viewer.live import LiveViewer

    model = mujoco.MjModel.from_xml_string(XML)
    n = 8

    class _WarpLikeSim:  # no `env_mjdata`: this is what makes it take the warp path
        num_envs = n
        mj_model = model
        expanded_fields = frozenset()
        data = types.SimpleNamespace(
            qpos=torch.arange(n * model.nq, dtype=torch.float32).reshape(n, model.nq),
            qvel=torch.arange(n * model.nv, dtype=torch.float32).reshape(n, model.nv),
            ctrl=torch.zeros(n, model.nu),
            mocap_pos=torch.zeros(n, model.nmocap, 3),
            mocap_quat=torch.zeros(n, model.nmocap, 4),
            xfrc_applied=torch.zeros(n, model.nbody, 6),
        )

    sim = _WarpLikeSim()
    v = LiveViewer(sim, env_index=3, show_all_envs=True)
    v._model, v._data = v._pick_render_target()
    v._draw_order = [5, 1, 7]

    calls = []
    real_cpu = torch.Tensor.cpu

    def counting_cpu(self, *args, **kwargs):
        calls.append(tuple(self.shape))
        return real_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)
    frame = v._pull_state()
    monkeypatch.undo()

    assert len(calls) == 1, f"{len(calls)} transfers for one frame: {calls}"
    rows = [3, 5, 1, 7]
    assert (frame.qpos == sim.data.qpos[rows].numpy()).all(), "rows out of order"
    assert (frame.qvel == sim.data.qvel[rows].numpy()).all()


def test_picks_the_envs_nearest_the_camera() -> None:
    """The chosen ones must be **nearest the followed environment**, not lowest
    by index.

    Environments are laid out in a square grid by `env_spacing`, so 4096 is
    64x64 -- taking the first 128 by index is exactly **the two outermost rows**,
    and environment 0, which the camera follows, sits at a corner. A real 4096
    run showed "a field of robots stuck to the edge of the view"; this test comes
    from that.
    """
    import numpy as np

    from mjrl.viewer.live import _MAX_DRAW_ENVS

    side = 64
    n = side * side
    # A 64x64 grid at 2 m spacing, matching mjlab's default layout
    origins = np.array(
        [[(i % side) * 2.0, (i // side) * 2.0, 0.0] for i in range(n)]
    )
    # Follow the middle one, so "nearest" and "low index" are clearly two
    # different sets.
    center = (side // 2) * side + side // 2
    v, model = make_viewer(n, origins=origins)
    v._env_index = center
    v._resolve_draw_limit(model)

    picked = np.array(v._draw_order)
    assert len(picked) == _MAX_DRAW_ENVS - 1
    d = np.linalg.norm(origins[picked] - origins[center], axis=1)
    # Every chosen one must be closer than the nearest of the ones not chosen
    rest = np.setdiff1d(np.arange(n), np.append(picked, center))
    d_rest = np.linalg.norm(origins[rest] - origins[center], axis=1)
    assert d.max() <= d_rest.min() + 1e-9, "these are not the nearest ones"
    # And clearly not "the first 128"
    assert set(picked) != set(range(1, _MAX_DRAW_ENVS)), "back to taking them by index"


def test_falls_back_when_no_origins_given() -> None:
    """With no origins available it must not crash -- fall back to by-index, the
    view simply is not centred."""
    v, model = make_viewer(500, origins=None)
    v._resolve_draw_limit(model)
    assert len(v._draw_order) > 0


def test_camera_tracks_the_followed_env() -> None:
    """The camera has to stay on the followed environment rather than sitting at
    the default free view.

    The default free camera `launch_passive` provides looks at the model's
    `stat.center` (near the origin), while environments are tiled by
    `env_spacing`: 4096 of them span 126 x 126 m, and **environment 0's origin is
    at the (63, -63) corner, 89 m from the origin**. What a real run showed was
    "not around the camera, a field of robots off in the distance" -- this test
    comes from that.

    Using MuJoCo's own `mjCAMERA_TRACKING` (camera centre locked to a body, the
    user still free to orbit and zoom) beats rewriting `lookat` every frame,
    which would fight the user's mouse.
    """
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "rl" / "mjrl" / "viewer" / "live.py"
    ).read_text(encoding="utf-8")
    body = src.split("def _aim_camera(", 1)[1].split("\n    def ", 1)[0]
    assert "mjCAMERA_TRACKING" in body, "not using a tracking camera"
    assert "trackbodyid" in body
    start = src.split("def start(", 1)[1].split("\n    def ", 1)[0]
    assert "_aim_camera(" in start, "start() never aims the camera"


def test_followed_body_is_found_by_free_joint() -> None:
    """Which body to follow is found by its free joint, not a hard-coded index.

    Body 0 is the world and body 1 is often the terrain; where the robot's base
    lands depends on how the scene was assembled (measured: body 2 on jumper). A
    hard-coded index would follow the terrain.
    """
    v, model = make_viewer(4)
    body = v._followed_body(model)
    assert body is not None
    free = int(mujoco.mjtJoint.mjJNT_FREE)
    adr = model.body_jntadr[body]
    assert model.body_jntnum[body] > 0 and model.jnt_type[adr] == free


def test_defaults_to_the_scene_center_env() -> None:
    """Unspecified, follow the environment at **the centre of the scene**, not
    number 0.

    Environments tile into a square and 0 is at a corner: 4096 of them span
    126x126 m with 0 at (63, -63). Following it, the neighbours drawn are only on
    two sides and the other two are empty.
    """
    import numpy as np

    side = 64
    n = side * side
    origins = np.array(
        [[(i % side) * 2.0 - 63, (i // side) * 2.0 - 63, 0.0] for i in range(n)]
    )
    from mjrl.viewer.live import LiveViewer

    # **Do not use make_viewer** -- it passes env_index=0 explicitly, which takes
    # the "explicitly specified" path, so the auto-centre logic never runs at all
    # (that is exactly how the first version of this test fooled itself).
    v = LiveViewer(_FakeSim(n), env_origins=origins)
    picked = origins[v._env_index]
    # Must land near the centroid rather than at a corner
    assert np.linalg.norm(picked[:2]) <= 2.0, f"following {picked}, which is off-centre"
    assert v._env_index != 0


def test_explicit_env_index_wins() -> None:
    """Given explicitly, use what was given -- auto only applies when it is not."""
    import numpy as np

    origins = np.array([[i * 2.0, 0.0, 0.0] for i in range(64)])
    from mjrl.viewer.live import LiveViewer

    v = LiveViewer(_FakeSim(64), env_index=7, env_origins=origins)
    assert v._env_index == 7


def test_no_origins_falls_back_to_zero() -> None:
    """With no origins available, fall back to 0 rather than crashing."""
    from mjrl.viewer.live import LiveViewer

    v = LiveViewer(_FakeSim(32), env_origins=None)
    assert v._env_index == 0


# ── Physics on the stripped model, the window on full appearance ────────

_RENDER_XML = """
<mujoco>
  <asset>
    <mesh name="deco" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    <mesh name="hull" vertex="0 0 0  .5 0 0  0 .5 0  0 0 .5"/>
  </asset>
  <worldbody>
    <!-- The terrain sits on **its own body**, as in mjlab (measured bodyid=1,
         not 0). It has collision geometry only and no appearance stand-in --
         hide it by mistake and the picture has no floor. -->
    <body name="terrain">
      <geom name="floor" type="plane" size="5 5 0.1"/>
    </body>
    <body name="b" pos="0 0 1"><freejoint/>
      <geom name="look" type="mesh" mesh="deco" contype="0" conaffinity="0" mass="0"/>
      <geom name="col" type="mesh" mesh="hull" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _stripped_sim(num_envs: int = 2):
    from mjrl.backend import native_sim as ns
    from mjrl.backend.native_sim import NativeSimulation

    spec = mujoco.MjSpec.from_string(_RENDER_XML)
    old = ns._STRIP_VISUALS
    ns._STRIP_VISUALS = True  # a small model never trips the budget, so force it
    try:
        return NativeSimulation(num_envs, None, None, "cpu", spec=spec)
    finally:
        ns._STRIP_VISUALS = old


def test_render_model_keeps_the_visual_mesh_the_physics_model_dropped() -> None:
    """Stripping saves on the N copies; drawing needs only one -- so do not throw
    the appearance away with them."""
    sim = _stripped_sim()
    try:
        assert sim.visuals_stripped, "this test assumes stripping actually happened"
        assert sim.render_model is not sim.mj_model
        assert sim.render_model.nmesh > sim.mj_model.nmesh, (
            f"the render model should keep the appearance meshes: "
            f"{sim.render_model.nmesh} vs {sim.mj_model.nmesh}"
        )
        # qpos/qvel have to be interchangeable element for element, otherwise the
        # pose drawn in the window is wrong.
        for f in ("nq", "nv", "nbody", "njnt"):
            assert getattr(sim.render_model, f) == getattr(sim.mj_model, f), f
    finally:
        sim.close()


def test_render_model_falls_back_to_the_physics_model() -> None:
    """With nothing stripped the two are the same object, so callers need no
    special case."""
    from mjrl.backend.native_sim import NativeSimulation

    m = mujoco.MjModel.from_xml_string(XML)
    sim = NativeSimulation(2, None, m, "cpu")
    try:
        assert sim.render_model is sim.mj_model
    finally:
        sim.close()


def test_viewer_draws_with_the_render_model() -> None:
    """The window has to take the full-appearance model, not the physics one."""
    from mjrl.viewer.live import LiveViewer

    sim = _stripped_sim()
    try:
        v = LiveViewer(sim, env_index=0, show_all_envs=True)
        model, data = v._pick_render_target()
        assert model is sim.render_model
        assert model is not sim.mj_model, (
            "taking the physics model leaves no appearance to draw"
        )
        assert data is not sim.env_mjdata(0)[1], "the proxy data must not be the live one"
        assert v._live_main is sim.env_mjdata(0)[1], (
            "the live data has to be remembered, or there is nothing to copy frames from"
        )
        assert data.qpos.shape == sim.env_mjdata(0)[1].qpos.shape
    finally:
        sim.close()


def test_the_window_never_draws_live_state_even_when_nothing_was_stripped() -> None:
    """With nothing stripped the window draws the physics model -- but never the
    live `MjData`.

    This used to pin the opposite: no extra MjData and no extra mj_forward, the
    window drawing native's live data directly. That was free while drawing ran
    on the simulation thread. It does not any more, and the frame thread would be
    reading the live data while the next step writes it, which is a torn picture.
    """
    from mjrl.backend.native_sim import NativeSimulation
    from mjrl.viewer.live import LiveViewer

    m = mujoco.MjModel.from_xml_string(XML)
    sim = NativeSimulation(2, None, m, "cpu")
    try:
        v = LiveViewer(sim, env_index=0, show_all_envs=True)
        model, data = v._pick_render_target()
        live_model, live_data = sim.env_mjdata(0)
        assert model is live_model, "nothing was stripped, so there is one model"
        assert data is not live_data, "the window must have an MjData of its own"
        assert v._live_main is live_data, "frames have to be copied from somewhere"
        assert not v._render_proxy
    finally:
        sim.close()


def test_collision_only_groups_are_the_ones_turned_off() -> None:
    """"Appearance only" is decided per group, and **the same way on both
    backends**.

    The default `geomgroup` is [1,1,1,0,0,0] while jumper's collision hulls are in
    group 1 and its appearance in group 2 -- both on, so the hulls are drawn over
    the appearance (measured 31 -> 62 geoms per environment). It is especially
    visible on warp: it does not strip visual meshes, so both sets are there as
    written.
    """
    from mjrl.viewer.live import LiveViewer

    m = mujoco.MjModel.from_xml_string(_GROUPED_XML)
    hide = LiveViewer._collision_only_groups(m)

    assert 1 in hide, (
        "the hull group (all collision, and every owning body has appearance too) "
        "should be off"
    )
    assert 2 not in hide, "the appearance group must stay on"
    # The terrain has group 0 to itself: collision geometry, but its body has no
    # appearance stand-in -- turn it off and there is no floor.
    assert 0 not in hide, "the floor got turned off: the robots hang in the void"


def test_same_group_assets_are_left_alone() -> None:
    """If appearance and collision share a group, groups cannot tell them apart
    -- so turn nothing off.

    Group numbers are the asset author's choice and writing them that way is
    perfectly legitimate. Better to draw one layer too many than to turn the
    appearance off with it.
    """
    from mjrl.viewer.live import LiveViewer

    m = mujoco.MjModel.from_xml_string(_RENDER_XML_SAME_GROUP)
    assert LiveViewer._collision_only_groups(m) == set()


def test_macos_viewer_gate_checks_mjpython_not_sys_executable(monkeypatch) -> None:
    """On macOS, "are we under mjpython" must be answered by
    `mujoco.viewer._MJPYTHON`.

    mjpython is a shell that execs an ordinary interpreter -- `sys.executable` is
    always `.../bin/python`. Judging by basename, macOS **never** opens a window,
    and what it reports is "please start with mjpython", which does not help
    because that is what was done (measured locally 2026-09-04: started from
    `.venv/bin/mjpython` and it still went headless).
    """
    import sys

    import mujoco.viewer

    from mjrl.viewer.live import no_display_reason

    monkeypatch.setattr(sys, "platform", "darwin")
    # Exactly what it looks like under mjpython:
    monkeypatch.setattr(sys, "executable", "/somewhere/bin/python")

    monkeypatch.setattr(mujoco.viewer, "_MJPYTHON", object(), raising=False)
    assert no_display_reason() is None, "claimed no window is possible under mjpython"

    monkeypatch.setattr(mujoco.viewer, "_MJPYTHON", None, raising=False)
    assert "mjpython" in (no_display_reason() or ""), "plain python should be stopped"


def test_stop_survives_an_unjoinable_thread() -> None:
    """Under mjpython what gets captured is a dummy thread and `join` raises --
    that must not take the shutdown down with it.

    On macOS `launch_passive` puts the UI on the main thread, and the difference
    from `threading.enumerate()` is a dummy thread created on the C side, whose
    `Thread.join` goes straight to
    `assert False, "cannot join a dummy thread"`. Not swallowing it ends every
    training run in an AssertionError (measured locally 2026-09-04).
    """

    class _Unjoinable:
        def join(self, timeout=None):
            raise AssertionError("cannot join a dummy thread")

        def is_alive(self):
            return True

    v, _ = make_viewer(4)
    v._viewer_thread = _Unjoinable()
    v.stop()  # must not raise
    assert v._viewer_thread is None


#: Appearance and collision in **separate groups**, as on jumper (hulls group 1,
#: appearance group 2, terrain alone in group 0)
_GROUPED_XML = """
<mujoco>
  <asset>
    <mesh name="deco" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    <mesh name="hull" vertex="0 0 0  .5 0 0  0 .5 0  0 0 .5"/>
  </asset>
  <worldbody>
    <body name="terrain">
      <geom name="floor" type="plane" size="5 5 0.1" group="0"/>
    </body>
    <body name="b" pos="0 0 1"><freejoint/>
      <geom name="col" type="mesh" mesh="hull" group="1" mass="1"/>
      <geom name="look" type="mesh" mesh="deco" group="2"
            contype="0" conaffinity="0" mass="0"/>
    </body>
  </worldbody>
</mujoco>
"""

#: Appearance and collision in **the same group** -- groups cannot separate them
_RENDER_XML_SAME_GROUP = """
<mujoco>
  <asset>
    <mesh name="deco" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    <mesh name="hull" vertex="0 0 0  .5 0 0  0 .5 0  0 0 .5"/>
  </asset>
  <worldbody>
    <body name="terrain"><geom name="floor" type="plane" size="5 5 0.1"/></body>
    <body name="b" pos="0 0 1"><freejoint/>
      <geom name="col" type="mesh" mesh="hull" mass="1"/>
      <geom name="look" type="mesh" mesh="deco" contype="0" conaffinity="0" mass="0"/>
    </body>
  </worldbody>
</mujoco>
"""


# ── Shadows for the environments that are drawn ─────────────────────────


def test_the_shadow_map_covers_the_environments_being_drawn() -> None:
    """MuJoCo fits a directional light's shadow map to the **model**, which here
    is one environment.

    The environments drawn beside it are spread over the grid -- 128 at 2 m
    spacing is more than 20 m across -- and everything outside that little box
    renders with no shadow at all. Nothing reports it: the picture is complete and
    correctly lit, most of the robots simply have no shadow, and the eye reads
    that as them floating rather than as a rendering setting. Measured: identical
    boxes at 0 m and 14 m, and only the near one had a shadow.

    Widening is what the fix is, and it is nearly free -- mjlab asks for an 8192
    shadow map, so even a 126 m box is 15 mm per texel. Appended geometry casts
    shadows like any other, which was worth checking before relying on it.
    """
    import numpy as np

    side = 12
    origins = np.array(
        [[(i % side) * 2.0, (i // side) * 2.0, 0.0] for i in range(side * side)]
    )
    v, model = make_viewer(side * side, origins=origins)
    v._resolve_draw_limit(model)
    before = float(model.vis.map.shadowclip)
    v._widen_shadows(model)
    after = float(model.vis.map.shadowclip)

    pts = origins[list(v._draw_order) + [v._env_index]][:, :2]
    radius = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    assert after > before, "the shadow map was left at the single-environment size"
    assert after * float(model.stat.extent) >= radius, (
        f"a {radius:.0f} m spread of environments needs at least that much "
        f"coverage; got {after * float(model.stat.extent):.0f} m"
    )


def test_widening_never_narrows_what_the_model_asked_for() -> None:
    """A model that already asks for a wide shadow map keeps it: this only ever
    widens, because a scene may have set it for its own reasons."""
    import numpy as np

    origins = np.array([[i * 2.0, 0.0, 0.0] for i in range(16)])
    v, model = make_viewer(16, origins=origins)
    v._resolve_draw_limit(model)
    model.vis.map.shadowclip = 500.0
    v._widen_shadows(model)
    assert float(model.vis.map.shadowclip) == 500.0


def test_widening_is_skipped_without_origins() -> None:
    """With no origins there is nothing to cover, and it must not crash."""
    v, model = make_viewer(8, origins=None)
    v._resolve_draw_limit(model)
    before = float(model.vis.map.shadowclip)
    v._widen_shadows(model)
    assert float(model.vis.map.shadowclip) == before
