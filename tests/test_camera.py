"""Invariants of the native camera sensor, and of the robot's own camera.

Two subjects, both about cameras. The first and larger half is mjrl's offscreen
renderer; the robot's forward camera is at the bottom, under its own banner.


Every one of these comes from a side-by-side run on the training machine on
2026-09-03, and each corresponds to a bug that **raised nothing, produced no NaN
and gave a depth image that looked entirely normal**. Static review finds none of
them; only an assertion pins them down.

The comparison criteria are not here: they need mjwarp running as the reference
(see verify_camera_pixels.py / verify_camera_env.py in the scratchpad). What is
pinned here is the behaviour those comparisons have **already** validated and
which must not drift back.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco", reason="the camera is MuJoCo's offscreen rendering")

import numpy as np  # noqa: E402

from mjrl.sensor.camera import _CATMASK, _no_multisampling, scene_option  # noqa: E402

XML_TEXTURED = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.9"
             rgb2="0.1 0.1 0.1" width="32" height="32"/>
    <material name="gridmat" texture="grid" texrepeat="4 4"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" material="gridmat"/>
    <camera name="cam" pos="0 0 3" quat="1 0 0 0" fovy="45" resolution="8 8"/>
  </worldbody>
</mujoco>
"""

XML = """
<mujoco>
  <statistic extent="2"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="b" pos="0 0 0.4"><freejoint/>
      <geom name="box" type="box" size="0.2 0.2 0.2"/>
    </body>
    <site name="s" pos="0.3 0 0.9" size="0.15"/>
    <camera name="cam" pos="0 0 3" quat="1 0 0 0" fovy="45" resolution="32 24"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def model_data():
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


# ── Upstream behaviour we rely on ───────────────────────────────────────


def test_mj_setconst_clobbers_the_visual_statistics(model_data) -> None:
    """`mj_setConst` recomputes an explicitly set `stat.extent` from the geometry.

    The restore logic in `NativeSimulation._scatter_model` exists for exactly
    that. This test pins **the upstream behaviour**: the day MuJoCo stops
    recomputing stat, that restore becomes dead code and this test fails first,
    prompting its removal rather than leaving it lying around.
    """
    m, d = model_data
    m2 = mujoco.MjModel.from_xml_string(XML)
    assert m2.stat.extent == pytest.approx(2.0), "the extent set explicitly in the XML"
    mujoco.mj_setConst(m2, mujoco.MjData(m2))
    assert m2.stat.extent != pytest.approx(2.0), (
        "mj_setConst no longer recomputes stat.extent -- "
        "the restore logic in NativeSimulation._scatter_model can go"
    )


def test_multisampling_is_what_breaks_depth(model_data) -> None:
    """With MSAA on, depth is systematically biased; with it off, it is not.

    Measured: on NVIDIA + EGL, MSAA's depth resolve introduces a constant offset
    in **inverse depth** -- about 1.5% at 3 m, 4.6% at 10 m -- and it varies with
    the sample count. It is invisible on macOS's GL stack (MuJoCo bypasses that
    path in split mode), so validating on one machine only would miss it.

    This does not compare absolute accuracy (that depends on the GL stack), only
    whether turning it off gets **closer to the analytic truth** -- a relation
    that holds on both stacks.
    """
    m, _ = model_data
    d = mujoco.MjData(m)
    d.qpos[:3] = [0.05, -0.05, 0.4]
    mujoco.mj_forward(m, d)
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    truth = _analytic_depth(m, d, cam_id, 32, 24)

    errs = {}
    for label, ctx in (("MSAA off", _no_multisampling(m)), ("MSAA on", _identity())):
        try:
            with ctx:
                r = mujoco.Renderer(m, height=24, width=32)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"cannot create an offscreen context here: {type(e).__name__}: {e}")
        try:
            _update(r, m, d, cam_id)
            r.enable_depth_rendering()
            z = r.render().copy()
            r.disable_depth_rendering()
        finally:
            try:
                r.close()
            except Exception:  # noqa: BLE001,S110
                pass
        ok = (truth > 0) & (z > 0) & (z < float(m.vis.map.zfar * m.stat.extent) * 0.9)
        assert ok.sum() > 50, f"{label}: too few comparable pixels"
        errs[label] = float(np.median(np.abs(z[ok] - truth[ok]) / truth[ok]))

    assert errs["MSAA off"] <= errs["MSAA on"], (
        f"turning MSAA off made it worse: {errs} -- the premise of "
        f"_no_multisampling has changed"
    )
    assert errs["MSAA off"] < 1e-3, (
        f"still off the analytic truth with MSAA off: {errs['MSAA off']:.3e}"
    )


# ── Our own conventions ─────────────────────────────────────────────────


def test_scene_option_excludes_sites(model_data) -> None:
    """Sites must be excluded.

    mjwarp casts geoms only; `mjv_updateScene` draws sites as well by default. A
    single site in the scene conjures up a patch in native's depth image that
    warp does not have -- and it looks perfectly normal. A site is **not**
    mjCAT_DECOR, so the catmask does not stop it; only zeroing sitegroup does.
    """
    opt = scene_option((0, 1, 2))
    assert list(opt.sitegroup) == [0] * len(opt.sitegroup)
    assert list(opt.geomgroup[:3]) == [1, 1, 1]
    assert list(opt.geomgroup[3:]) == [0] * (len(opt.geomgroup) - 3)

    m, d = model_data
    scn = mujoco.MjvScene(model=m, maxgeom=1000)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    mujoco.mjv_updateScene(m, d, opt, None, cam, _CATMASK, scn)
    types = {int(scn.geoms[i].objtype) for i in range(scn.ngeom)}
    assert int(mujoco.mjtObj.mjOBJ_SITE) not in types, f"a site got in: {types}"

    # Control: with the defaults, sites really are drawn -- otherwise the
    # assertion above proves nothing.
    scn2 = mujoco.MjvScene(model=m, maxgeom=1000)
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, cam,
                           int(mujoco.mjtCatBit.mjCAT_ALL), scn2)
    types2 = {int(scn2.geoms[i].objtype) for i in range(scn2.ngeom)}
    assert int(mujoco.mjtObj.mjOBJ_SITE) in types2, (
        "the control group drew no site either, so this test is vacuous"
    )


def test_textures_are_disabled_via_matid_not_texid(model_data) -> None:
    """`use_textures=False` works by setting `matid = -1`; `texid` does nothing.

    Measured: setting `texid = -1` left the floor's brightness std at 79.10 ->
    79.10 (no change at all); `matid = -1` genuinely removed the checkerboard
    (79.10 -> 34.89, matching the 35.20 mjwarp gives for `use_textures=False`).
    Colour does not go with it -- `mjv_updateScene` has already resolved the
    material's rgba into `geom.rgba`.

    This pins down *which field is the switch*. Getting it wrong raises nothing:
    the texture renders as before while the config says it is off -- another bug
    that looks entirely normal.
    """
    src = (
        pytest.importorskip("mjrl.sensor.camera").__file__
    )
    body = open(src, encoding="utf-8").read()
    render_one = body.split("def _render_one", 1)[1].split("\n    def ", 1)[0]
    assert ".matid = -1" in render_one, "the texture switch has to be matid"
    assert ".texid = -1" not in render_one, "texid does nothing; this was measured"


def test_textured_rgb_divergence_is_announced(model_data) -> None:
    """RGB plus textures makes the two backends differ, so it must warn.

    Measured on jumper: everything untextured differs by 0.1/255, textured terrain
    by 50/255. For a vision policy the texture is most of the picture -- leave it
    unsaid and it becomes a problem discovered three days into training.

    Only RGB is affected; depth and segmentation have nothing to do with
    textures and should not be bothered by this.
    """
    from mjrl.sensor.camera import setup_context

    m = mujoco.MjModel.from_xml_string(XML_TEXTURED)

    def ctx_with(types, use_textures=True):
        cfg = type("C", (), dict(name="c", width=8, height=8, data_types=types,
                                 enabled_geom_groups=(0, 1, 2),
                                 use_textures=use_textures, use_shadows=False))()
        c = type("Ctx", (), {})()
        c.camera_sensors = [type("S", (), dict(cfg=cfg, camera_idx=0))()]
        c._data = type("D", (), {"nworld": 1})()
        return c

    with pytest.warns(RuntimeWarning, match="textured materials"):
        setup_context(ctx_with(("rgb",)), m)

    import warnings as _w

    for types, tex in ((("depth", "segmentation"), True), (("rgb",), False)):
        with _w.catch_warnings():
            _w.simplefilter("error")  # any warning at all counts as a failure
            setup_context(ctx_with(types, tex), m)


def test_no_multisampling_restores_on_exception(model_data) -> None:
    """offsamples must be restored on the exception path too -- it mutates the
    shared model."""
    m, _ = model_data
    before = int(m.vis.quality.offsamples)
    with pytest.raises(RuntimeError):
        with _no_multisampling(m):
            assert int(m.vis.quality.offsamples) == 0
            raise RuntimeError("boom")
    assert int(m.vis.quality.offsamples) == before


def test_setup_context_lays_out_buffers_like_mjwarp() -> None:
    """With several cameras, **each data type gets its own running offset**, and
    a camera that does not enable a type records -1.

    A wrong offset raises nothing: `get_depth()` still returns a
    correctly-shaped tensor, it just holds another camera's content, or reads
    out of bounds and gets zeros. Hence a deliberately different camera
    combination for each of the three data types.
    """
    from mjrl.sensor.camera import setup_context

    def sensor(name, w, h, types, idx):
        cfg = type("C", (), dict(name=name, width=w, height=h, data_types=types,
                                 enabled_geom_groups=(0, 1, 2), use_textures=True,
                                 use_shadows=False))()
        return type("S", (), dict(cfg=cfg, camera_idx=idx))()

    ctx = type("Ctx", (), {})()
    ctx.camera_sensors = [
        sensor("a", 8, 6, ("rgb", "depth", "segmentation"), 0),
        sensor("b", 4, 4, ("depth",), 1),
        sensor("c", 5, 5, ("segmentation", "rgb"), 2),
    ]
    ctx._data = type("D", (), {"nworld": 3})()
    setup_context(ctx, mujoco.MjModel.from_xml_string(XML))

    assert ctx._depth_adr_np == [0, 48, -1], (
        "only cameras that asked for depth take space in the depth buffer"
    )
    assert ctx._seg_adr_np == [0, -1, 48], "seg has its own independent offsets"
    assert ctx._rgb_adr_np == [0, -1, 48], "and rgb another set again"
    assert tuple(ctx._depth_torch.shape) == (3, 48 + 16)
    assert tuple(ctx._seg_torch.shape) == (3, 48 + 25, 2)
    assert tuple(ctx._rgb_torch.shape) == (3, 48 + 25, 3)
    assert ctx._rgb_torch.dtype.__str__() == "torch.uint8", "RGB has to be uint8"
    # get_rgb() uses _rgb_unpacked to decide "is RGB on"; on native it is the
    # very same buffer.
    assert ctx._rgb_unpacked is ctx._rgb_torch
    # Segmentation's background convention is -1, not 0 -- the initial value has
    # to be right, since it can be read before the first frame is rendered.
    assert int(ctx._seg_torch.min()) == -1 and int(ctx._seg_torch.max()) == -1


def test_setup_context_skips_buffers_nobody_asked_for() -> None:
    """With no camera asking for rgb, `_rgb_unpacked` must be None.

    That is exactly what `get_rgb()` uses to decide "is RGB on". Allocating
    unconditionally would hand a misconfigured task an image that is forever
    black instead of an error -- again the kind that looks plausible.
    """
    from mjrl.sensor.camera import setup_context

    cfg = type("C", (), dict(name="d", width=4, height=4, data_types=("depth",),
                             enabled_geom_groups=(0, 1, 2), use_textures=True,
                             use_shadows=False))()
    ctx = type("Ctx", (), {})()
    ctx.camera_sensors = [type("S", (), dict(cfg=cfg, camera_idx=0))()]
    ctx._data = type("D", (), {"nworld": 2})()
    setup_context(ctx, mujoco.MjModel.from_xml_string(XML))

    assert ctx._rgb_torch is None and ctx._rgb_unpacked is None
    assert ctx._seg_torch is None
    assert ctx._rgb_adr_np == [-1] and ctx._seg_adr_np == [-1]


def test_render_flags_match_what_mjwarp_can_do() -> None:
    """Rasteriser-only effects must be off: reflection, fog and haze do not
    exist in mjwarp at all.

    RGB can never match pixel for pixel across the two, but *whether a class of
    effect is drawn* can be aligned. Left on, floor reflection and fog shift the
    colour of everything far away, and that part of the difference **has nothing
    to do with the shading model** -- mixed together, there is no way left to say
    what causes the rest.
    """
    from mjrl.sensor.camera import _align_render_flags

    m = mujoco.MjModel.from_xml_string(XML)
    scn = mujoco.MjvScene(model=m, maxgeom=100)
    f = mujoco.mjtRndFlag
    for flag in (f.mjRND_REFLECTION, f.mjRND_FOG, f.mjRND_HAZE):
        scn.flags[flag] = True  # on first, to confirm they get turned off
    _align_render_flags(scn, use_shadows=False)

    assert not scn.flags[f.mjRND_REFLECTION]
    assert not scn.flags[f.mjRND_FOG]
    assert not scn.flags[f.mjRND_HAZE]
    assert not scn.flags[f.mjRND_SHADOW]
    assert scn.flags[f.mjRND_SKYBOX], "mjlab passes render_skybox=True to mjwarp"

    _align_render_flags(scn, use_shadows=True)
    assert scn.flags[f.mjRND_SHADOW], "use_shadows=True must actually enable shadows"


# ── Helpers ─────────────────────────────────────────────────────────────


import contextlib  # noqa: E402


@contextlib.contextmanager
def _identity():
    yield


def _update(r, m, d, cam_id) -> None:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = cam_id
    mujoco.mjv_updateScene(m, d, scene_option((0, 1, 2)), None, cam, _CATMASK, r.scene)


def _analytic_depth(m, d, cam_id, w, h) -> np.ndarray:
    """Solve the analytic plane depth per pixel with `mujoco.mj_ray`, as a
    reference independent of rasterisation.

    Ray directions are built the way mjwarp's `compute_ray` does, and depth is
    the projection along the optical axis -- matching both backends' convention
    (see the comparison table at the top of mjrl/sensor/camera.py).
    """
    znear = float(m.vis.map.znear * m.stat.extent)
    half_h = znear * np.tan(np.radians(float(m.cam_fovy[cam_id]) / 2))
    half_w = half_h * (w / h)
    pos = d.cam_xpos[cam_id].copy()
    mat = d.cam_xmat[cam_id].reshape(3, 3).copy()
    out = np.zeros((h, w))
    gid = np.zeros(1, np.int32)
    for py in range(h):
        y = half_h - 2 * half_h * ((py + 0.5) / h)
        for px in range(w):
            x = -half_w + 2 * half_w * ((px + 0.5) / w)
            v = np.array([x, y, -znear])
            v /= np.linalg.norm(v)
            dist = mujoco.mj_ray(m, d, pos, mat @ v, None, 1, -1, gid)
            out[py, px] = dist * (-v[2]) if dist >= 0 else 0.0
    return out


# ── The camera module's intrinsics ──────────────────────────────────────
#
# The pose and the mount are not here: the V1.6 model carries its own
# `camera_link` and nothing mounts a `<camera>` on it yet. What is pinned
# is the arithmetic inside `assets/jumper/camera/camera_config.yaml`, which is
# data and holds whichever body the camera ends up on.


def test_the_rgb_intrinsics_agree_with_the_diagonal_they_came_from() -> None:
    """The angles in `camera_config.yaml` have to describe one pinhole.

    They did not, for as long as the module drawing was missing: 视场角(FOV.D) 123°
    is the **diagonal**, it was recorded as `hfov_deg`, and a vertical angle was
    derived from it on a 16:9 frame this 4:3 sensor does not have. Nothing failed.
    The camera simply rendered 4.5 degrees narrower than the hardware sees, which
    is not something anybody notices in a viewer.

    Everything below `dfov_deg` in that block is derived from it, so this recomputes
    the derivation rather than restating its results. `dfov_deg`, `width` and
    `height` are the inputs; if they change, the rest has to be recomputed and this
    is what says so.
    """
    import math

    import yaml

    from tasks.jumper.common.assets import CAMERA_CONFIG

    rgb = yaml.safe_load(CAMERA_CONFIG.read_text(encoding="utf-8"))["rgb"]
    w, h = rgb["width"], rgb["height"]
    f = math.hypot(w, h) / 2 / math.tan(math.radians(rgb["dfov_deg"]) / 2)

    assert rgb["hfov_deg"] == pytest.approx(
        2 * math.degrees(math.atan((w / 2) / f)), abs=0.05
    )
    assert rgb["vfov_deg"] == pytest.approx(
        2 * math.degrees(math.atan((h / 2) / f)), abs=0.05
    )

    k = rgb["intrinsics"]
    assert k["fx_px"] == pytest.approx(f, abs=0.5)
    # Square pixels, and a principal point assumed central -- both are what makes
    # the single `f` above legitimate, and both are what a calibration will change.
    assert k["fx_px"] == pytest.approx(k["fy_px"])
    assert (k["cx_px"], k["cy_px"]) == pytest.approx((w / 2, h / 2))
    assert k["calibrated"] is False, "if it is calibrated, it is no longer derived"

    # The control: the pair that used to be in this file fails the same check, so
    # the assertions above are not satisfiable by any two numbers.
    stale_fx = (w / 2) / math.tan(math.radians(123.0) / 2)
    stale_fy = (h / 2) / math.tan(math.radians(91.2) / 2)
    assert stale_fx != pytest.approx(stale_fy, rel=0.05), (
        "the old hfov/vfov pair is consistent after all -- this control proves nothing"
    )




# ── The camera on the robot ─────────────────────────────────────────────
#
# `jumper.xml` mounts one on `camera_link`, the housing V1.6 carries as a real
# link. The asset is generated but cannot be regenerated on most machines
# (`build_jumper.py` needs the URDF export), so these check the committed artefact.


def _jumper_model(strip_cameras: bool = False):
    """The committed jumper asset, optionally with its cameras -- `onboard` and the
    dToF's `tof` -- removed.

    Through `MjSpec` rather than by editing the XML text, so the mesh directory is
    resolved relative to the asset the way a real load does it.
    """
    from tasks.jumper.common.assets import JUMPER_XML

    spec = mujoco.MjSpec.from_file(str(JUMPER_XML))
    if strip_cameras:
        for cam in list(spec.cameras):
            spec.delete(cam)
    return spec.compile()


def test_the_onboard_camera_can_see_out() -> None:
    """A camera inside a closed mesh renders the mesh, and nothing errors.

    This has now happened twice, in two different places, for the same reason: the
    hardware looks out through an aperture and an STL of the housing is closed. On
    the pre-V1.6 shell the rig pose was 38 mm inside `base_link`. On V1.6 the
    housing's own origin sits on its front face, so a camera at the body origin has
    every ray hit it at zero distance -- measured, 221 of 221.

    So the assertion is not a clearance in millimetres, which is what both mounts
    got wrong. It asks MuJoCo what the camera can see: rays over the full field of
    view at the home pose, and none of them may land on the robot. A revision that
    grows a bumper, moves the ToF forward or re-poses the legs into the frame fails
    here rather than shipping a view of itself.
    """
    import math

    import numpy as np

    model = _jumper_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "onboard")
    assert cam >= 0, "the asset has no onboard camera"

    eye = data.cam_xpos[cam].copy()
    rot = data.cam_xmat[cam].reshape(3, 3)
    half = math.tan(math.radians(model.cam_fovy[cam]) / 2)
    aspect = model.cam_resolution[cam][0] / model.cam_resolution[cam][1]

    blocked = []
    for iy in range(-6, 7):
        for ix in range(-8, 9):
            ray = rot @ np.array([half * aspect * ix / 8, half * iy / 6, -1.0])
            ray /= np.linalg.norm(ray)
            geom = np.zeros(1, dtype=np.int32)
            dist = mujoco.mj_ray(model, data, eye, ray, None, 1, -1, geom)
            if 0.0 <= dist < 0.5:
                blocked.append(
                    (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom[0])),
                     round(float(dist), 4))
                )
    assert not blocked, f"the robot is in its own camera's view: {sorted(set(blocked))}"


def test_the_onboard_camera_changes_no_physics() -> None:
    """The cameras are in every run, including training, and must cost them nothing.

    A MuJoCo camera is a rigid massless frame offset -- no mass, no inertia, no
    degree of freedom, no geometry -- which is why it can live in the asset rather
    than being gated to replay, and why no massless link is interposed to place the
    eye: `nbody` is part of what is asserted here and a link would change it. That
    is an argument until it is measured.
    """
    with_camera, without = _jumper_model(), _jumper_model(strip_cameras=True)
    for field in ("nq", "nv", "nu", "nbody", "ngeom", "njnt", "nsite", "nsensor"):
        assert getattr(with_camera, field) == getattr(without, field), field
    # The control: the one thing that is meant to differ did -- both cameras,
    # the forward camera and the dToF, and nothing else.
    assert with_camera.ncam == without.ncam + 2 == 2


def test_the_onboard_camera_matches_the_module_it_models() -> None:
    """The mount takes its optics from the config, not from numbers of its own.

    `fovy` is vertical and two wider angles sit beside it in the same block, so
    taking `dfov_deg` (123) or `hfov_deg` (111.7) renders a camera far wider than
    the module -- perfectly, and with nothing to report it. `resolution` has to be
    set as well: without it the compiled camera reports 1x1 and the horizontal
    field falls out of whatever buffer renders it.
    """
    import yaml

    from tasks.jumper.common.assets import CAMERA_CONFIG

    cfg = yaml.safe_load(CAMERA_CONFIG.read_text(encoding="utf-8"))
    model = _jumper_model()
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "onboard")

    assert model.cam_fovy[cam] == pytest.approx(cfg["rgb"]["vfov_deg"], abs=0.05)
    assert list(model.cam_resolution[cam]) == [cfg["rgb"]["width"], cfg["rgb"]["height"]]
    # The control: the three angles are distinct, so this can tell them apart.
    assert cfg["rgb"]["vfov_deg"] < cfg["rgb"]["hfov_deg"] < cfg["rgb"]["dfov_deg"]

    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.cam_bodyid[cam])
    assert body == cfg["camera_rig"]["parent_link"], (
        "the camera hangs off a different body than the config claims"
    )
    assert model.cam_pos[cam] == pytest.approx(
        [cfg["camera_rig"]["sim_eye_offset_m"], 0.0, 0.0]
    )


def test_the_onboard_camera_faces_the_way_the_robot_does() -> None:
    """MuJoCo cameras look down local -Z with local +Y up, which nobody guesses.

    `camera_config.yaml` states the mapping its quaternion intends and this checks
    the compiled model against it. Transcribed wrong it gives a view of the robot's
    own back with the world upside down, and renders it faultlessly.
    """
    import numpy as np

    model = _jumper_model()
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "onboard")
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.cam_quat[cam])
    rot = rot.reshape(3, 3)
    assert -rot[:, 2] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6), "not looking forward"
    assert rot[:, 1] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6), "not upright"


def test_the_generator_still_mounts_the_camera() -> None:
    """The generator is the copy that rots, because it is the one nobody runs.

    Nothing fails when `build_jumper.py` loses the mount: the asset already has the
    camera. It fails later, for the next person who *can* run the generator, when
    regenerating silently drops it.
    """
    from tasks.paths import ASSETS_DIR

    source = (ASSETS_DIR / "jumper" / "tools" / "build_jumper.py").read_text(encoding="utf-8")
    assert "housing.add_camera(" in source, "the generator no longer mounts it"
    assert 'CAMERA_BODY = "camera_link"' in source
