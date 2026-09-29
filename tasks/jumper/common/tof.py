"""The dToF (VL53L9CX): one ray per zone from the model's camera, and the chip's errors.

**The model carries the sensor; this reads it and makes it behave like the chip.**
`build_jumper.py` mounts the camera `tof` on `tof_sensor_link` with the zone array, both
fields of view and the eye offset from `camera_config.yaml`. `tof_sensor()` builds a
`ToFSensorCfg` from that camera and from `tof_noise_config.yaml`, so every task that
declares it gets the same sensor in the same place with the same errors.

Declaring it is a choice each task makes, like any other sensor:

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

and reading it (`ToFData`):

    data = env.scene["tof"].data
    data.range.view(-1, *TOF_SHAPE)   # [N, 42, 54] as the chip reports it, held between exposures
    data.truth.view(-1, *TOF_SHAPE)   # the same zones, exact and current -- for a critic
    data.fresh                        # [N] whether `range` is a new exposure this step

Both are radial -- the distance along the zone's ray from the Rx optical centre, which is
what the chip's depth map holds (the perpendicular conversion is a host step, §8.2) --
in metres, and -1 where a zone has no valid return. Row 0 is the top of the picture and
column 0 its left, the robot's left: the image as the sensor sees it. `distances`,
inherited from the ray cast, is measured from where the rays start rather than from the
optical centre, and is not the sensor's reading.

Section and table numbers are ST DS14879 Rev 6.

## Seeing it

`play` and `play --app` show `range` for the followed environment in the live
viewer's top-right corner, in every task and scene (`preview`; the viewer finds it on
its own, see `mjrl/viewer/live.py`). Turbo on a log scale across the sensor's range --
dark red at 0.05 m, orange at 0.3, green at 1, cyan at 3, blue at 8.8 -- and black
where a zone has no valid return. It is the chip's reading, noise and all, not
`truth`.

## Geometry

- **Zones sit on an ideal pinhole grid** through each zone's centre, 55 x 42 degrees.
  The lens is not a pinhole (§8.3 corrects it with a radial term); the simulation takes
  the host to have done that correction, so distortion is not simulated.
- **Range is measured from the body's origin, taken to be the Rx optical centre.** The
  CAD's `tof_sensor_link` is one closed housing with nothing modelled inside, but it is
  centred 4.199 mm to the robot's right of its origin and level with it, and the Rx
  aperture is 4.216 mm from the module's centre (Figure 23). The rays themselves start
  at the camera, `tof.sim_eye_offset_m` further forward, because from the origin the
  shell's collision hull bridges the window and swallows all 2268 of them; measuring
  from the origin puts those millimetres back. Inferred, not stated -- confirm with the
  mechanical design.
- The same inference puts Tx on the robot's right, which is the datasheet's drawings
  (Rx left, Tx right, seen from the scene) turned 180 degrees about the optical axis.

## The hardware's zone order is not this one

The Rx lens flips the image both ways (§8.1) and the array streams from its (0, 0)
corner (§5.1), which sees the top right of the scene; which corner of *this* image that
is then depends on the mounting above. Every one of the three is easy to get backwards
and gives a mirrored world with nothing failing, so pin the mapping on the board -- a
hand in one corner of the view -- before a policy reads this sensor there.

## The chip's errors

In the order the hardware makes them; the numbers, and the reasons for them, are in
`tof_noise_config.yaml`:

1. a zone on the boundary between two surfaces reports the other one (not from the
   datasheet);
2. zones outside 0.05-8.8 m report nothing (Tables 22, 23);
3. per device, redrawn every episode: an offset within the accuracy bound and a fixed
   per-zone pattern with the uniformity's standard deviation (Tables 27, 35);
4. per exposure: temporal noise (Table 31), and a 1 % chance of no return (§8.2) --
   **off by default**, `detection.enabled` in the config;
5. a new exposure every 1/50 s, the robot's frame rate, each environment on its own
   phase: every step at a 50 Hz control loop, and at 200 Hz the three steps in between
   read the last one.

Ported from the `feature/dToF` branch (46484b3), which applied the same effects inside
five_foot's height-map term, with four corrections. Its clock discarded the remainder of
each period, so it exposed at 25 Hz rather than 30 (its own measurement: 15 exposures
in 30 control steps). Its dropout doubled toward maximum range, reaching 2 % where the
datasheet says 1 %. Its edge zones only took the neighbour to the left, which no
horizontal edge -- a stair nosing -- can trigger, and the leftmost column wrapped round
to the rightmost; here a zone can take any of its four neighbours, but only one on
another surface (see `_swap_at_edges`). And its bias was one fixed curve for every
robot rather than each device's own error.

## Not modelled

Ambient light (the outdoor columns), reflectance, more than one target per zone,
temperature drift, a cover glass, the frame-to-frame correlation TNR introduces, and
the chip's latency.

## Cost

On the native backend the 2268 rays go to one `mj_multiRay` call per environment per
step (`mjrl/sensor/raycast.py`). One environment, replay, alternating blocks in one
process (2026-09-29, the 5090 D training box), without the dToF -> with it:
jumper.flat 3.24 -> 4.38 ms a step, jumper.five_foot 4.80 -> 7.06 ms. Cast as a loop
of `mj_ray` it was 7.98 and 10.35. The error model is 0.2 ms of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cache

import mujoco
import numpy as np
import torch
import warp as wp
import yaml
from mjlab.sensor import ObjRef, RayCastData, RayCastSensor, RayCastSensorCfg

from .assets import CAMERA_CONFIG, TOF_NOISE_CONFIG

__all__ = [
    "TOF_SHAPE",
    "ToFData",
    "ToFErrorModel",
    "ToFSensor",
    "ToFSensorCfg",
    "ToFZonePatternCfg",
    "tof_sensor",
]

#: The camera `build_jumper.py` mounts. A name the model does not carry fails at
#: scene build (`mj_model.camera` raises), not silently.
TOF_CAMERA = "tof"


@cache
def _camera() -> dict:
    """The camera config's `tof` block, plus the range limits from its `depth` block."""
    cfg = yaml.safe_load(CAMERA_CONFIG.read_text(encoding="utf-8"))
    return {**cfg["tof"], "min_m": cfg["depth"]["min_m"], "max_m": cfg["depth"]["max_m"]}


#: (rows, columns) of the zone image -- (42, 54) -- for reshaping `range` and `truth`.
TOF_SHAPE: tuple[int, int] = (_camera()["height"], _camera()["width"])


@dataclass
class ToFZonePatternCfg:
    """One ray per zone through the zone's centre, taken from a MuJoCo camera.

    Duck-types into `RayCastSensorCfg.pattern`, which only ever calls
    `generate_rays(mj_model, device)`. Resolution, both fields of view and the eye's
    pose are read from the compiled camera, so this holds no numbers of its own.

    mjlab's `PinholeCameraPatternCfg` is close and differs in three ways, each of which
    gives an image that looks entirely normal. It samples **edge to edge**, so its outer
    rays sit on the border of the field of view and the zones are stretched by 54/53;
    `camera_config.yaml` says `angular_sampling: zone_center`, and here zone `i` is at
    `(i + 0.5) / 54` of the field. It emits along its frame's -Z, so it needs a frame
    oriented like a camera -- a site the model would carry beside the camera -- where
    this takes the camera's own pose and attaches to the body the camera hangs off. And
    its first row is the bottom of the image.
    """

    camera: ObjRef
    """The camera the zones look through."""

    body: ObjRef
    """The sensor's `frame`, which must be the body the camera hangs off: the rays
    are expressed in that body's coordinates."""

    def generate_rays(
        self, mj_model: mujoco.MjModel | None, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mj_model is None:
            raise ValueError("ToFZonePatternCfg reads its camera from the compiled model")
        cam = mj_model.camera(self.camera.prefixed_name()).id
        body = mj_model.body(self.body.prefixed_name()).id
        # A frame on another body would put every ray in the wrong place, and
        # plausibly so.
        assert mj_model.cam_bodyid[cam] == body, (
            f"camera {self.camera.prefixed_name()} hangs off body "
            f"{mj_model.body(int(mj_model.cam_bodyid[cam])).name}, "
            f"not the sensor frame {self.body.prefixed_name()}"
        )

        width, height = (int(n) for n in mj_model.cam_resolution[cam])
        # Half the image plane at unit focal length, per axis. With `sensorsize`
        # set the two axes are independent, which is how the model states 55 x 42;
        # without it MuJoCo has only `fovy` and square pixels.
        sensorsize = mj_model.cam_sensorsize[cam]
        if sensorsize[0] > 0 and sensorsize[1] > 0:
            focal = mj_model.cam_intrinsic[cam][:2]
            half_x = sensorsize[0] / (2 * focal[0])
            half_y = sensorsize[1] / (2 * focal[1])
        else:
            half_y = math.tan(math.radians(mj_model.cam_fovy[cam]) / 2)
            half_x = half_y * width / height

        # Zone centres. Rows top to bottom (camera +Y is up), columns left to
        # right (camera +X is right).
        u = ((np.arange(width) + 0.5) / width * 2 - 1) * half_x
        v = (1 - (np.arange(height) + 0.5) / height * 2) * half_y
        uu, vv = np.meshgrid(u, v, indexing="xy")
        dirs_cam = np.stack([uu.ravel(), vv.ravel(), -np.ones(uu.size)], axis=1)
        dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)

        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, mj_model.cam_quat[cam])
        dirs_body = dirs_cam @ rot.reshape(3, 3).T
        offsets = np.broadcast_to(mj_model.cam_pos[cam], dirs_body.shape)

        return (torch.tensor(np.ascontiguousarray(offsets), dtype=torch.float32, device=device),
                torch.tensor(dirs_body, dtype=torch.float32, device=device))


@dataclass(frozen=True)
class ToFErrorModel:
    """What the chip does to a zone's range, from `tof_noise_config.yaml`.

    One field per number in that file, so the numbers a run used are in its config.
    The three bounds that depend on range are methods: accuracy and temporal noise in
    the datasheet's own two regimes, split where its tables split them, and uniformity
    interpolated between the three distances Table 35 gives.
    """

    frame_rate_hz: float
    min_range_m: float
    max_range_m: float
    accuracy_short_below_m: float
    accuracy_short_m: float
    accuracy_long_fraction: float
    uniformity_at_m: tuple[float, ...]
    uniformity_std_m: tuple[float, ...]
    noise_short_below_m: float
    noise_short_m: float
    noise_long_fraction: float
    invalid_probability: float
    dropout: bool
    """Whether `invalid_probability` is applied. Off by default: see `detection`
    in the config."""
    wrong_surface_probability: float

    @classmethod
    def from_datasheet(cls) -> ToFErrorModel:
        cfg = yaml.safe_load(TOF_NOISE_CONFIG.read_text(encoding="utf-8"))
        acc, uni, noise = cfg["accuracy"], cfg["uniformity"], cfg["temporal_noise"]
        return cls(
            frame_rate_hz=cfg["frame"]["rate_hz"],
            min_range_m=_camera()["min_m"],
            max_range_m=_camera()["max_m"],
            accuracy_short_below_m=acc["short_below_m"],
            accuracy_short_m=acc["short_m"],
            accuracy_long_fraction=acc["long_fraction"],
            uniformity_at_m=tuple(uni["at_m"]),
            uniformity_std_m=tuple(uni["std_m"]),
            noise_short_below_m=noise["short_below_m"],
            noise_short_m=noise["short_m"],
            noise_long_fraction=noise["long_fraction"],
            invalid_probability=cfg["detection"]["invalid_probability"],
            dropout=bool(cfg["detection"]["enabled"]),
            wrong_surface_probability=cfg["edge"]["wrong_surface_probability"],
        )

    def accuracy(self, r: torch.Tensor) -> torch.Tensor:
        """The bound on a device's offset at range `r` (Table 27)."""
        return torch.where(r < self.accuracy_short_below_m,
                           r.new_tensor(self.accuracy_short_m), self.accuracy_long_fraction * r)

    def temporal_noise(self, r: torch.Tensor) -> torch.Tensor:
        """The standard deviation of one exposure at range `r` (Table 31)."""
        return torch.where(r < self.noise_short_below_m,
                           r.new_tensor(self.noise_short_m), self.noise_long_fraction * r)

    def uniformity(self, r: torch.Tensor) -> torch.Tensor:
        """The standard deviation of the fixed pattern at range `r` (Table 35):
        linear between the tabulated ranges, held at the end values outside them."""
        at, std = self.uniformity_at_m, self.uniformity_std_m
        x = r.clamp(at[0], at[-1])
        # Segment by segment, each overriding the last from its own start: a few
        # elementwise ops, where `searchsorted` and a gather cost several times more on
        # the CPU the native backend replays on.
        out = torch.full_like(r, std[0])
        for x0, x1, y0, y1 in zip(at, at[1:], std, std[1:]):
            out = torch.where(x >= x0, y0 + (x - x0) * ((y1 - y0) / (x1 - x0)), out)
        return out


@dataclass
class ToFData(RayCastData):
    """`RayCastData` plus the zones as the chip reports them. See the module docstring."""

    truth: torch.Tensor
    """[B, N] Exact radial range from the optical centre, this step. -1 where nothing
    is hit within the ray cast's `max_distance`."""

    range: torch.Tensor
    """[B, N] The last exposure's reported radial range, with the chip's errors.
    -1 where the zone has no valid return."""

    fresh: torch.Tensor
    """[B] True where `range` was exposed this step rather than held."""


@dataclass
class ToFSensorCfg(RayCastSensorCfg):
    """A ray cast whose sensor reports ranges the way the VL53L9CX does."""

    errors: ToFErrorModel = field(default_factory=ToFErrorModel.from_datasheet)
    """The chip's errors, the datasheet's unless replaced."""

    def build(self) -> ToFSensor:
        return ToFSensor(self)


class ToFSensor(RayCastSensor):
    """`RayCastSensor`, then the chip's errors and frame rate on every ray cast.

    **Applied where the rays are cast, once a step.** `postprocess_rays` runs in
    `sim.sense()`, after resets and before observations, so everything that reads the
    sensor in a step -- actor, critic, a viewer -- sees the same exposure and the same
    noise draw. Per-device state (the offset, the fixed pattern, the exposure phase) is
    redrawn in `reset`, so each episode meets a different unit.
    """

    cfg: ToFSensorCfg

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        pattern = self.cfg.pattern
        assert isinstance(pattern, ToFZonePatternCfg), "the errors need the zone grid"
        cam = mj_model.camera(pattern.camera.prefixed_name()).id
        width, height = (int(n) for n in mj_model.cam_resolution[cam])
        assert width * height == self._num_rays, (width, height, self._num_rays)
        self._shape = (height, width)

        n_env = data.nworld
        self._period = 1.0 / self.cfg.errors.frame_rate_hz
        self._elapsed = 0.0
        self._phase = torch.rand(n_env, device=device) * self._period
        self._force = torch.ones(n_env, dtype=torch.bool, device=device)
        self._fresh = torch.zeros(n_env, dtype=torch.bool, device=device)
        self._truth = torch.full((n_env, self._num_rays), -1.0, device=device)
        self._range = self._truth.clone()
        self._device_offset = torch.rand(n_env, 1, device=device) * 2 - 1
        self._fixed_pattern = torch.randn(n_env, self._num_rays, device=device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if not hasattr(self, "_force"):
            return
        ids = slice(None) if env_ids is None else env_ids
        count = self._force[ids].numel()
        device = self._force.device
        # A new episode is a new unit, and a held exposure from the last one would be
        # a picture of somewhere the robot no longer is.
        self._force[ids] = True
        self._phase[ids] = torch.rand(count, device=device) * self._period
        self._device_offset[ids] = torch.rand(count, 1, device=device) * 2 - 1
        self._fixed_pattern[ids] = torch.randn(count, self._num_rays, device=device)

    def update(self, dt: float) -> None:
        super().update(dt)
        self._elapsed += dt

    def postprocess_rays(self) -> None:
        super().postprocess_rays()
        self._expose()

    def preview(self, env_index: int) -> np.ndarray:
        """Environment `env_index`'s last exposure as an RGB image, for the live viewer.

        [42, 54, 3] uint8, row 0 at the top and column 0 the robot's left. Log scale
        because the interesting part of a walking robot's view is its first metre or
        two, which a linear scale to 8.8 m would squeeze into two colours.
        """
        errors = self.cfg.errors
        r = self._range[env_index].detach().cpu().numpy().reshape(self._shape)
        lo, hi = math.log(errors.min_range_m), math.log(errors.max_range_m)
        far = (np.log(np.clip(r, errors.min_range_m, errors.max_range_m)) - lo) / (hi - lo)
        image = _turbo(1.0 - (1.0 - _TURBO_FAR) * far)
        image[r < 0] = 0
        return image

    def _compute_data(self) -> ToFData:
        base = super()._compute_data()
        return ToFData(**vars(base), truth=self._truth, range=self._range, fresh=self._fresh)

    def _expose(self) -> None:
        hit = self._distances >= 0
        from_centre = (self._hit_pos_w - self._pos_w.unsqueeze(1)).norm(dim=-1)
        self._truth = torch.where(hit, from_centre, torch.full_like(from_centre, -1.0))

        # Keep the remainder. Resetting to zero instead is what gave the dToF branch
        # 25 Hz for a nominal 30 against a 50 Hz loop.
        self._phase += self._elapsed
        self._elapsed = 0.0
        due = self._phase >= self._period
        self._phase = torch.where(due, torch.remainder(self._phase, self._period), self._phase)
        take = due | self._force
        self._force = torch.zeros_like(self._force)
        self._fresh = take
        surface = wp.to_torch(self._ray_geomid)
        self._range = torch.where(
            take.unsqueeze(1), self._measure(self._truth, surface), self._range
        )

    def _measure(self, truth: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
        """One exposure: what the chip reports for these exact ranges, `surface` being
        the geom each zone's ray hit."""
        errors = self.cfg.errors
        r = truth
        if errors.wrong_surface_probability > 0:
            r = self._swap_at_edges(truth, surface)
        valid = (r >= errors.min_range_m) & (r <= errors.max_range_m)
        reported = (
            r
            + self._device_offset * errors.accuracy(r)
            + self._fixed_pattern * errors.uniformity(r)
            + torch.randn_like(r) * errors.temporal_noise(r)
        )
        if errors.dropout:
            valid &= torch.rand_like(r) >= errors.invalid_probability
        return torch.where(valid, reported, torch.full_like(r, -1.0))

    def _swap_at_edges(self, r: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
        """A zone takes a neighbour's range when the two lie on different surfaces.

        Each zone picks one of its four neighbours and, with the edge probability, reports
        that neighbour's range -- **but only if the neighbour's ray hit a different
        geom.** A zone spanning two surfaces holds two histogram peaks and the chip
        reports one of them; a zone on one continuous surface holds one peak, however
        steep, and the swap must not touch it. Range differences cannot tell the two
        apart: on the floor near the horizon one row of zones spans metres, so taking
        the zone above there moved readings by metres on flat ground -- reported minus
        exact over a flat jumper.flat frame had a 58 mm standard deviation before this
        test and 5-6 mm with it. Surface identity tells them apart on either backend;
        the native one returns no normals.

        What it gets wrong: two coplanar geoms meeting at a seam count as an edge, and
        one mesh folding in front of itself does not.
        """
        n_env = r.shape[0]
        img = r.view(n_env, *self._shape)
        ids = surface.view(n_env, *self._shape)
        pad = torch.nn.functional.pad(img, (1, 1, 1, 1), value=-1.0)
        pad_ids = torch.nn.functional.pad(ids, (1, 1, 1, 1), value=-1)
        pick = torch.randint(0, 4, img.shape, device=img.device)
        other, other_ids = img.clone(), ids.clone()
        for k, (rows, cols) in enumerate(((slice(0, -2), slice(1, -1)),    # above
                                          (slice(2, None), slice(1, -1)),  # below
                                          (slice(1, -1), slice(0, -2)),    # left
                                          (slice(1, -1), slice(2, None)))):  # right
            chosen = pick == k
            other = torch.where(chosen, pad[:, rows, cols], other)
            other_ids = torch.where(chosen, pad_ids[:, rows, cols], other_ids)
        swap = ((torch.rand_like(img) < self.cfg.errors.wrong_surface_probability)
                & (img >= 0) & (other >= 0) & (other_ids != ids))
        return torch.where(swap, other, img).reshape(n_env, -1)


#: Google's Turbo colour map as a quintic per channel (Mikhailov, 2019): row k holds
#: the x**k coefficients of red, green and blue.
_TURBO = np.array([
    [0.13572138, 0.09140261, 0.10667330],
    [4.61539260, 2.19418839, 12.64194608],
    [-42.66032258, 4.84296658, -60.58204836],
    [132.13108234, -14.18503333, 110.36276771],
    [-152.94239396, 4.27729857, -89.90310912],
    [59.28637943, 2.82956604, 27.34824973],
])


#: Where the maximum range lands on the map. Turbo's own end is nearly black
#: (34, 23, 27), which next to the black of "no return" reads as the same thing;
#: from 0.1 up it is a clear blue.
_TURBO_FAR = 0.1


def _turbo(x: np.ndarray) -> np.ndarray:
    """`x` in [0, 1] -> uint8 RGB, dark blue at 0 through green to dark red at 1."""
    x = np.clip(x, 0.0, 1.0)[..., None]
    rgb = _TURBO[5]
    for c in _TURBO[4::-1]:
        rgb = c + x * rgb
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def tof_sensor(name: str = "tof") -> ToFSensorCfg:
    """The robot's dToF, as the model mounts it. `name` is the scene's key for it."""
    frame = ObjRef(type="body", name=_camera()["parent_link"], entity="robot")
    return ToFSensorCfg(
        name=name,
        frame=frame,
        pattern=ToFZonePatternCfg(
            camera=ObjRef(type="camera", name=TOF_CAMERA, entity="robot"), body=frame
        ),
        ray_alignment="base",
        max_distance=_camera()["max_m"],
        # The housing: the hardware looks out through a window in it, which the
        # closed mesh does not have.
        exclude_parent_body=True,
    )
