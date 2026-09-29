"""Cameras for the native backend: off-screen `mujoco.Renderer` in place of
mjwarp's `render()`.

## Where the seam is

`CameraSensor` itself is **entirely backend-agnostic** -- its `_compute_data()`
calls only three accessors on `SensorContext`:

```
ctx.get_rgb(cam)          -> [N,H,W,3] uint8
ctx.get_depth(cam)        -> [N,H,W,1] float32
ctx.get_segmentation(cam) -> [N,H,W,2] int32
```

and those three are themselves just **slice arithmetic**: take camera i's span out
of one flat buffer at `<type>_adr[i]` and reshape. They read `ctx._depth_torch` /
`_seg_torch` / `_rgb_torch` and the matching `_*_adr_np`.

So native has only two things to do:

1. `setup_context()` -- allocate those buffers on the CPU **in exactly mjwarp's
   layout**, so the three accessors need no change at all.
2. `NativeCameraRenderer.render_into()` -- fill in the pixels each step.

## Pixel conventions on both sides (checked one by one, not assumed)

| | mjwarp `render()` | `mujoco.Renderer` | Handling |
|---|---|---|---|
| Row order | `py=0` is `v~0`, camera-frame +y (**up**) | `np.flipud` at the end, row 0 also up | Same; no flip |
| Depth semantics | `dist*cos(theta)`, i.e. planar depth along the optical axis | OpenGL linearised depth, also along the axis | Same |
| Depth background | **0.0** | **exactly zfar** (derivation below) | Converted to 0.0 here |
| Segmentation background | `(-1,-1)` | `(-1,-1)` | Same |
| Segmentation type channel | `mjwarp ObjType.GEOM` | `mjtObj.mjOBJ_GEOM` | mjwarp's `ObjType` is an alias of `mjtObj`, value for value |
| What is drawn | geoms and flex only | also sites / tendons / joint axes by default | Those visualisation groups are turned off; see difference 3 below |

**Why the background is exactly zfar** (`mujoco/rendering/classic/renderer.py`):
with `readDepthMap = mjDEPTH_ZEROFAR` the raw value at the far plane is 0, and
substituting into its inverse projection `d/(raw+c)` -- where
`c = znear/(zfar-znear)` and `d = zfar*znear/(zfar-znear)` -- gives `d/c = zfar`.
So the test is "is it sitting at zfar", not a magic number.

## Multisampling must be off

`offsamples=0`; see `_no_multisampling` -- **the most important line in this
module**. MSAA depth resolution introduces a systematic error of order 1e-2 on
NVIDIA, and is invisible on macOS (MuJoCo's segmented-rendering workaround applies
there and not here). Leave it on and the depth map looks entirely normal, with
every value short and distant values shorter.

## Three known differences (all intrinsic to rasterisation vs ray casting, not bugs)

1. **Far-plane clipping**: hits beyond `zfar` are clipped by rasterisation (->
   background -> 0), while ray casting has no such limit and returns the true
   distance. With a ground plane extending to infinity, the rows near the horizon
   necessarily disagree.
2. **Near-plane clipping**: geometry closer than `znear` is clipped, likewise.
3. **Geoms only**: mjwarp projects geoms and flex, while `mjv_updateScene` also
   draws sites, tendons and joint axes by default. A single site adds a phantom
   blob to the native depth map that warp does not have -- and it **looks entirely
   normal**. Both have to be turned off: `catmask` takes `STATIC|DYNAMIC` to
   exclude `mjCAT_DECOR` things like contact points and force arrows, and
   `MjvOption`'s `sitegroup` / `jointgroup` / ... are zeroed for the rest.
   (A site is **not** decor -- like a geom it is sorted into STATIC / DYNAMIC by
   its owning body, so catmask alone does not filter it.)

## RGB: the one of the three that cannot be required to match per pixel

Depth and segmentation are purely geometric: both implementations compute the same
mathematical quantity, so per-pixel agreement is a fair requirement (measured:
median relative error 0.000e+00, disagreements entirely on silhouettes). RGB is
not: mjwarp does its own ray shading, MuJoCo rasterises through OpenGL.

**Far more of it matches than expected**:

| | native vs warp, mean absolute difference per channel |
|---|---|
| Everything untextured (geometry, lighting, material base colour, shadows) | **0.1 / 255** |
| Textured surfaces | **50 / 255** |

In a controlled scene, untextured structural correlation is **r = 0.9999**
(negative control 0.20), the chromaticity difference is 1e-4, and channel order
agrees. **The difference is entirely in texture sampling** -- see
`_warn_textured_rgb_divergence`, which also explains why it is worth a dedicated
warning.

Three things align this renderer with warp, each explained where it happens:

1. `_align_render_flags` -- turn off the effects mjwarp does not have (reflections,
   fog, haze).
2. `use_textures=False` is implemented as `matid = -1` (not `texid`, which has no
   effect).
3. MSAA stays off -- mjwarp casts one ray per pixel and has no anti-aliasing.

## Not supported

- **flex**: mjwarp projects flex, whereas here anything outside `mjCAT_DECOR` goes
  through `mjv_updateScene`'s ordinary path, which has not been checked against
  warp. No model in this repository uses flex.
"""

from __future__ import annotations

import contextlib
import warnings

import mujoco
import numpy as np

__all__ = ["NativeCameraRenderer", "scene_option", "setup_context"]

#: Excludes `mjCAT_DECOR` decorations such as contact points, force arrows and
#: joint axes. **This alone is not enough**: a site is not decor, so MjvOption's
#: sitegroup has to be zeroed as well. See difference 3 in the module docstring
#: and `NativeCameraRenderer.__init__`.
_CATMASK = int(mujoco.mjtCatBit.mjCAT_STATIC) | int(mujoco.mjtCatBit.mjCAT_DYNAMIC)


def setup_context(ctx, mj_model: mujoco.MjModel) -> None:
    """Allocate the camera buffers on the CPU in mjwarp's layout.

    Called from the native branch of `SensorContext.__init__` / `.recreate()` in
    place of `_create_context()`, which would build a `mjwarp.RenderContext` that
    native cannot construct.

    The layout matches mjwarp: **one flat buffer per data type**, shaped
    `[nworld, total pixels of that type, C]`, with each camera occupying a span in
    sorted order and its start offset recorded in `_<type>_adr_np` (-1 for a camera
    that does not enable that type). `get_depth()` and friends slice by exactly
    this convention.
    """
    import torch

    cams = ctx.camera_sensors
    if not cams:
        # Ray casting only: none of the three buffers is needed, and the
        # accessors assert before they would use them
        ctx._rgb_adr_np = ctx._depth_adr_np = ctx._seg_adr_np = None
        ctx._rgb_unpacked = ctx._rgb_torch = ctx._depth_torch = ctx._seg_torch = None
        return

    _check_framebuffer(cams, mj_model)
    _warn_textured_rgb_divergence(cams, mj_model)

    nworld = ctx._data.nworld
    ncam = len(cams)

    # Each type keeps its own running offset: a camera that does not enable a
    # type occupies no space in that buffer
    rgb_adr = [-1] * ncam
    depth_adr = [-1] * ncam
    seg_adr = [-1] * ncam
    n_rgb = n_depth = n_seg = 0
    for i, s in enumerate(cams):
        npix = s.cfg.width * s.cfg.height
        if "rgb" in s.cfg.data_types:
            rgb_adr[i] = n_rgb
            n_rgb += npix
        if "depth" in s.cfg.data_types:
            depth_adr[i] = n_depth
            n_depth += npix
        if "segmentation" in s.cfg.data_types:
            seg_adr[i] = n_seg
            n_seg += npix

    ctx._rgb_adr_np = rgb_adr
    ctx._depth_adr_np = depth_adr
    ctx._seg_adr_np = seg_adr
    # native is always CPU; these tensors feed observations directly, so their
    # dtypes must match the warp side
    ctx._rgb_torch = (
        torch.zeros((nworld, n_rgb, 3), dtype=torch.uint8) if n_rgb else None
    )
    # On warp, `_rgb_unpacked` is a `wp.array` and `_rgb_torch` is its torch view.
    # native has no separate warp array and the two are the same buffer. `get_rgb()`
    # only uses it to test whether RGB is enabled
    # (`if self._rgb_unpacked is None: raise`), so pointing both at one object both
    # satisfies that test and is semantically right -- it is one buffer.
    ctx._rgb_unpacked = ctx._rgb_torch
    ctx._depth_torch = (
        torch.zeros((nworld, n_depth), dtype=torch.float32) if n_depth else None
    )
    ctx._seg_torch = (
        torch.full((nworld, n_seg, 2), -1, dtype=torch.int32) if n_seg else None
    )


def _warn_textured_rgb_divergence(cams, mj_model: mujoco.MjModel) -> None:
    """On textured surfaces the two backends' RGB **necessarily differs** -- and
    by a lot.

    ## Measured boundary

    | | native vs warp, mean absolute difference per channel |
    |---|---|
    | **Everything untextured** (robot geometry, plain-coloured objects, lighting, shadows) | **0.1 / 255** |
    | The textured ground (the hexapod's terrain) | **50 / 255** |

    In other words the shading model, the lighting, the material base colours and
    the geometry **do agree** (structural correlation r=0.9999 in a controlled
    scene); the difference is **entirely** in texture sampling.

    ## Two independent causes

    1. **`texuniform` is ignored by mjwarp** (the string does not appear in its
       source). It always samples "repeat per unit length" while MuJoCo obeys the
       model, so a textured material with `texuniform=false` differs even in
       texture **scale** -- measured, a checkerboard alternates 3 times along a
       scanline instead of 13. It is warp that departs from the model here, so this
       implementation does not imitate it.
    2. **Even with `texuniform=true`**, the two texture pipelines filter and sample
       differently (rasterisation has mipmapping and bilinear filtering, ray casting
       point-samples). A textured ground correlates at 0.85 in a controlled scene
       where the untextured version is at 0.9999.

    ## Why this deserves its own warning

    For a **visual policy** the texture usually dominates the frame (the hexapod's
    downward view is 1606 of 1728 pixels ground). "Changing the backend should not
    change the task" does not hold here: a visual policy trained on one backend
    should not be assumed to hold on the other. Left unsaid, this becomes a problem
    discovered three days into a training run.

    Depth and segmentation are unaffected, having nothing to do with textures.
    """
    if not any(s.cfg.use_textures for s in cams if "rgb" in s.cfg.data_types):
        return  # RGB only, and moot once the caller has turned textures off
    role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    textured = [i for i in range(mj_model.nmat) if mj_model.mat_texid[i][role] >= 0]
    if not textured:
        return
    name_of = lambda i: mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, i)  # noqa: E731
    non_uniform = [i for i in textured if not mj_model.mat_texuniform[i]]
    extra = (
        f"\nOf those, {[name_of(i) for i in non_uniform]} also have "
        "texuniform=false -- mjwarp does not read that field, so even the texture "
        "scale differs and the divergence is larger still."
        if non_uniform else ""
    )
    warnings.warn(
        f"A camera has RGB enabled and the scene contains textured materials "
        f"({[name_of(i) for i in textured]}).\n"
        "**native and warp RGB differ noticeably on textured surfaces**: measured "
        "on the hexapod's terrain, about 50/255 per channel, while everything "
        "untextured differs by only 0.1/255 -- shading and geometry agree, and the "
        "difference is entirely in texture sampling.\n"
        f"{extra}\n"
        "A visual policy trained on one backend should not be assumed to hold on "
        "the other.\n"
        "To make them agree, set use_textures=False on the camera (measured to "
        "bring the per-channel difference down to 0.1/255).\n"
        "Depth and segmentation are unaffected.",
        RuntimeWarning,
        stacklevel=3,
    )


def scene_option(enabled_geom_groups) -> mujoco.MjvOption:
    """Build an `MjvOption` whose visibility matches mjwarp's.

    It is a function so that comparison scripts can use **the same** settings when
    checking against warp: a copy inside a script would validate the copy, and the
    original could break while the script still passed.

    Two things:

    1. `geomgroup` follows the sensor's `enabled_geom_groups`, matching mjwarp's
       `enabled_geom_groups`.
    2. **Turn site / joint / tendon / actuator / skin off entirely.** This is what
       actually excludes sites: `mjCAT_DECOR` covers pure decorations such as
       contact points and force arrows, and a site is not among them -- like a
       geom it is sorted into STATIC / DYNAMIC by its owning body, so catmask alone
       does not filter it. Measured: left on, the segmentation map gains pixels
       with objtype=6 that mjwarp, projecting only geoms, does not have -- and it
       looks entirely normal.

    `flexgroup` is left at its default: mjwarp does project flex (the
    `ObjType.FLEX` branch in its `render.py`), no model in this repository uses
    flex, and this has not been checked against warp.
    """
    opt = mujoco.MjvOption()
    groups = set(enabled_geom_groups)
    for g in range(len(opt.geomgroup)):
        opt.geomgroup[g] = 1 if g in groups else 0
    for name in ("sitegroup", "jointgroup", "tendongroup", "actuatorgroup", "skingroup"):
        arr = getattr(opt, name)
        for i in range(len(arr)):
            arr[i] = 0
    return opt


def _check_framebuffer(cams, mj_model: mujoco.MjModel) -> None:
    """The off-screen framebuffer is 640x480 by default; anything larger needs
    `<global offwidth=...>` in the XML.

    `mujoco.Renderer` reports this too, but only when the per-camera Renderer is
    built, and its message does not say which sensor. This reports all of them at
    once, up front.
    """
    bw, bh = mj_model.vis.global_.offwidth, mj_model.vis.global_.offheight
    bad = [
        f"{s.cfg.name} {s.cfg.width}x{s.cfg.height}"
        for s in cams
        if s.cfg.width > bw or s.cfg.height > bh
    ]
    if bad:
        raise ValueError(
            f"camera resolutions exceed the off-screen framebuffer {bw}x{bh}: "
            f"{bad}.\n"
            "Enlarge it in the model XML: "
            "<visual><global offwidth=\"W\" offheight=\"H\"/></visual>"
        )


@contextlib.contextmanager
def _no_multisampling(mj_model: mujoco.MjModel):
    """Temporarily disable off-screen framebuffer multisampling while a
    `Renderer` is constructed.

    ## Why it has to be off (measured, not assumed)

    With MSAA on, reading depth through `mjr_readPixels` first has to blit-resolve
    the multisample depth buffer down to a single sample, and **how several samples
    combine into one is implementation-defined**. On NVIDIA with EGL/GLFW that
    introduces a **constant offset in inverse depth**:

        1/z_read = 1/z_true + eps

    where eps also varies with sample count (3.2e-3 at 1-2 samples, 4.9e-3 at 4,
    4.1e-3 m^-1 at 8). As a depth error that grows as z^2: about 1.5% at 3 m and
    4.6% at 10 m. With `offsamples=0` the median relative error against the
    analytic reference (`mujoco.mj_ray`) drops to **2.6e-05** and eps to 1.8e-08 --
    three orders of magnitude.

    ## Why this trap is particularly nasty

    It is **invisible** on the macOS GL stack: MuJoCo bypasses that path in
    `mjRND_SEGMENT` mode (that is what "Using segmented rendering for depth makes
    the calculated depth more accurate at far distances" in
    `mujoco/rendering/classic/renderer.py` refers to), so the error is 2.8e-06 on
    macOS while on Linux the same workaround does nothing and it stays at 4.9e-03.
    **The same code, the same MuJoCo version and the same scene differ by three
    orders of magnitude between two machines** -- verifying on one of them would
    miss it entirely.

    And it neither raises nor produces NaN: the depth map looks entirely normal,
    with every value slightly short and distant values shorter.

    ## Does turning it off cost anything

    No, and it is more correct anyway:

    - **Depth** is a geometric quantity for which anti-aliasing is meaningless.
    - **Segmentation** is per-pixel integer labels, and MSAA would blend the ids of
      different objects at the edges.
    - **RGB** is the one that wants MSAA -- and mjwarp casts one ray per pixel and
      has no anti-aliasing to match, so it stays off there too.

    ## Why modify in place and restore rather than copying the model

    `offsamples` is read exactly once, when `MjrContext` is constructed. A deep
    copy of a mesh-carrying `MjModel` like the hexapod's costs megabytes, while all
    that is needed is for the value to hold at that instant. `finally` restores it
    on the exception path too.
    """
    quality = mj_model.vis.quality
    saved = int(quality.offsamples)
    quality.offsamples = 0
    try:
        yield
    finally:
        quality.offsamples = saved


class NativeCameraRenderer:
    """Render each environment's cameras into the `SensorContext` buffers each step.

    Created by `NativeSimulation.set_sensor_context()`, called from `sense()` and
    released by `close()`.

    **One `Renderer` per resolution**: a `Renderer` owns a GL context and an
    `MjrContext` with meshes and textures already uploaded, so rebuilding one per
    environment would be very expensive -- while cameras of the same resolution can
    reuse one in sequence, since `update_scene()` refills the scene each time.
    """

    def __init__(self, ctx, mj_model: mujoco.MjModel) -> None:
        self._ctx = ctx
        self._mj_model = mj_model
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._closed = False

        cams = ctx.camera_sensors
        self._rgb_cams = [
            (i, s) for i, s in enumerate(cams) if "rgb" in s.cfg.data_types
        ]
        self._depth_cams = [
            (i, s) for i, s in enumerate(cams) if "depth" in s.cfg.data_types
        ]
        self._seg_cams = [
            (i, s) for i, s in enumerate(cams) if "segmentation" in s.cfg.data_types
        ]

        self._opt = scene_option(cams[0].cfg.enabled_geom_groups if cams else ())
        # The texture switch is applied per geom per frame (see _render_one);
        # this only records whether to do it
        self._strip_textures = bool(cams) and not cams[0].cfg.use_textures
        self._use_shadows = bool(cams) and cams[0].cfg.use_shadows

        # zfar is used to convert "background" from zfar back to mjwarp's 0.0
        extent = mj_model.stat.extent
        self._zfar = np.float32(mj_model.vis.map.zfar * extent)

        for s in cams:
            self._renderer_for(s.cfg.width, s.cfg.height)

    def _renderer_for(self, width: int, height: int) -> mujoco.Renderer:
        key = (width, height)
        r = self._renderers.get(key)
        if r is None:
            try:
                with _no_multisampling(self._mj_model):
                    r = mujoco.Renderer(self._mj_model, height=height, width=width)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"could not create a {width}x{height} off-screen rendering "
                    f"context: {type(e).__name__}: {e}\n"
                    "On a headless machine set MUJOCO_GL=egl (no DISPLAY needed); "
                    "on a machine with X, glfw works too.\n"
                    "Their output is bit-identical in measurement, and egl is "
                    "preferred."
                ) from e
            _align_render_flags(r.scene, use_shadows=self._use_shadows)
            self._renderers[key] = r
        return r

    def render_into(self, models: list, datas: list) -> None:
        """Write this step's depth / segmentation into the `ctx` buffers.

        Args:
            models: one `MjModel` per environment (they differ once domain
                randomisation has expanded them), ordered by environment index.
            datas: one `MjData` per environment, ordered by environment index.

        The loop is grouped by **data type** rather than by environment:
        `enable_depth_rendering()` is a switch on the renderer, and toggling it per
        environment would flip that state twice for nothing.
        """
        if self._closed:
            raise RuntimeError("renderer is closed")
        ctx = self._ctx

        if self._rgb_cams:
            assert ctx._rgb_torch is not None
            buf = ctx._rgb_torch
            # Ordinary mode: neither depth nor segmentation. Each data type gets
            # its own pass -- MuJoCo's Renderer is a mode switch and produces one
            # kind at a time.
            for list_idx, s in self._rgb_cams:
                adr = ctx._rgb_adr_np[list_idx]
                w, h = s.cfg.width, s.cfg.height
                r = self._renderers[(w, h)]
                for env, (m, d) in enumerate(zip(models, datas)):
                    img = self._render_one(r, m, d, s.camera_idx)
                    buf[env, adr : adr + w * h] = _to_tensor(
                        img.reshape(-1, 3), buf
                    )

        if self._depth_cams:
            assert ctx._depth_torch is not None
            buf = ctx._depth_torch
            for r in self._renderers.values():
                r.enable_depth_rendering()
            try:
                for list_idx, s in self._depth_cams:
                    adr = ctx._depth_adr_np[list_idx]
                    w, h = s.cfg.width, s.cfg.height
                    r = self._renderers[(w, h)]
                    for env, (m, d) in enumerate(zip(models, datas)):
                        img = self._render_one(r, m, d, s.camera_idx)
                        # The rasterised background is exactly zfar; mjwarp's
                        # convention is 0.0
                        img = np.where(img >= self._zfar, np.float32(0.0), img)
                        buf[env, adr : adr + w * h] = _to_tensor(img.reshape(-1), buf)
            finally:
                for r in self._renderers.values():
                    r.disable_depth_rendering()

        if self._seg_cams:
            assert ctx._seg_torch is not None
            buf = ctx._seg_torch
            for r in self._renderers.values():
                r.enable_segmentation_rendering()
            try:
                for list_idx, s in self._seg_cams:
                    adr = ctx._seg_adr_np[list_idx]
                    w, h = s.cfg.width, s.cfg.height
                    r = self._renderers[(w, h)]
                    for env, (m, d) in enumerate(zip(models, datas)):
                        img = self._render_one(r, m, d, s.camera_idx)
                        buf[env, adr : adr + w * h] = _to_tensor(
                            img.reshape(-1, 2).astype(np.int32), buf
                        )
            finally:
                for r in self._renderers.values():
                    r.disable_segmentation_rendering()

    def _render_one(self, r, mj_model, data, cam_idx: int) -> np.ndarray:
        """Render one camera of one environment.

        `mjv_updateScene` is called directly rather than
        `Renderer.update_scene()`, because the latter hardcodes `self._model` (the
        one the Renderer was built with) while each environment has its own once
        domain randomisation has expanded the model, and geometry poses have to be
        computed from **that** one.

        > **Known boundary**: the meshes and textures inside `MjrContext` still come
        > from the nominal model. So randomising geom poses and sizes is tracked
        > while randomising the meshes themselves is not -- but mjlab's domain
        > randomisation does not touch meshes.
        """
        _check_same_frustum(mj_model, r)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = cam_idx
        mujoco.mjv_updateScene(
            mj_model, data, self._opt, None, cam, _CATMASK, r.scene
        )
        if self._strip_textures:
            # `use_textures=False` corresponds to mjwarp's "base colour only, no
            # texture sampling". MuJoCo has no global texture switch (there is none
            # in `mjtRndFlag`), so the material reference has to be severed on each
            # scene geom.
            #
            # **It is `matid`, not `texid`**: setting `texid = -1` measurably does
            # nothing (ground brightness std 79.10 -> 79.10), while `matid = -1`
            # really removes the checkerboard (std 79.10 -> 34.89, against mjwarp's
            # 35.20 with use_textures=False; mean brightness 199.05 vs 198.68).
            # Colour is not lost with it -- `mjv_updateScene` has already resolved
            # the material's rgba into `geom.rgba`.
            #
            # It has to be done every frame: `mjv_updateScene` rebuilds the geom
            # list each time.
            for i in range(r.scene.ngeom):
                r.scene.geoms[i].matid = -1
        return r.render()

    def close(self) -> None:
        """Release the GL context. **Call this explicitly; do not leave it to
        `__del__`.**

        The EGL backend raises `EGLError` on its cleanup path at interpreter exit
        (rendering itself is fine; the error is in `close()` -> `free` in
        `mujoco/egl`). Left to `__del__`, that traceback prints as the process exits
        and looks like a crash. It is caught and suppressed here.
        """
        if self._closed:
            return
        self._closed = True
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001,S110
                pass  # an EGL teardown error, unrelated to the output; see above
        self._renderers.clear()


def _align_render_flags(scn: mujoco.MjvScene, *, use_shadows: bool) -> None:
    """Align the rasteriser's render flags with what mjwarp can and cannot do.

    RGB cannot match per pixel between the two (the shading implementations are
    entirely different), but there is **no reason for them to differ in which
    classes of effect are drawn at all** -- that part can be aligned, and once it
    is, the remaining difference is shading alone and can be explained.

    | Flag | mjwarp | Here | Why |
    |---|---|---|---|
    | `SHADOW` | yes, controlled by `use_shadows` | follows cfg | same option |
    | `SKYBOX` | yes, hardcoded True by mjlab | True | matches the `_create_context` call |
    | `REFLECTION` | **no** | False | the rasteriser's ground reflection does not exist on warp |
    | `FOG` / `HAZE` | **no** | False | likewise, and fog shifts distant colours globally |
    | `WIREFRAME` / `ADDITIVE` | no | False | turned off defensively |

    `CULL_FACE` is left alone: back-face culling only affects faces that are not
    visible.
    """
    f = mujoco.mjtRndFlag
    for flag, on in (
        (f.mjRND_SHADOW, use_shadows),
        (f.mjRND_SKYBOX, True),
        (f.mjRND_REFLECTION, False),
        (f.mjRND_FOG, False),
        (f.mjRND_HAZE, False),
        (f.mjRND_WIREFRAME, False),
        (f.mjRND_ADDITIVE, False),
    ):
        scn.flags[flag] = bool(on)


def _check_same_frustum(mj_model: mujoco.MjModel, r) -> None:
    """Ensure the model that built the frustum and the model that linearises depth
    agree on near/far.

    **This guard catches a combination that silently produces wrong depth.**
    `mjv_updateScene` computes the frustum from the model passed to it
    (`near = vis.map.znear * stat.extent`), while `Renderer.render()`'s depth
    inverse projection hardcodes those two values from `r._model`. Let the two
    models' `stat.extent` differ and depth is scaled by their ratio -- and the depth
    map looks entirely normal, every number simply multiplied by a constant.

    This has happened: `mj_setConst` in domain randomisation recomputed a
    per-environment model's `stat.extent` from 2.0 to 0.6322 and depth came out
    3.16x too large. The root cause is fixed in
    `NativeSimulation._scatter_model`; this is the second line of defence, because
    an error of this kind **never surfaces on its own** and can only be asserted
    against.
    """
    a, b = mj_model, r._model
    if a is b:
        return
    for field, x, y in (
        ("stat.extent", a.stat.extent, b.stat.extent),
        ("vis.map.znear", a.vis.map.znear, b.vis.map.znear),
        ("vis.map.zfar", a.vis.map.zfar, b.vis.map.zfar),
    ):
        if abs(float(x) - float(y)) > 1e-9 * max(1.0, abs(float(y))):
            raise RuntimeError(
                f"the model used for rendering and the model the Renderer was "
                f"built with disagree on {field}: "
                f"{float(x):.6g} vs {float(y):.6g}.\n"
                f"Depth would be scaled by "
                f"{float(y) / float(x) if x else float('nan'):.4g} with no other "
                f"symptom at all.\n"
                f"Most likely something recomputed model constants per environment "
                f"(mj_setConst recomputes stat along with them)."
            )


def _to_tensor(arr: np.ndarray, like):
    import torch

    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=like.dtype)
