"""The native ray cast answers exactly as `mj_ray` does, ray by ray, on both of its paths.

`raycast_into` sends a frame whose rays share an origin to `mj_multiRay`, one call for
the whole frame, and everything else through a loop of `mj_ray`. Nothing downstream
can tell a wrong distance from a right one -- it is a height map or a dToF image that
looks entirely normal -- and the multi-ray path has two ways to be wrong that way: the
call takes **one origin and one excluded body per frame**, so a frame cast from the
neighbouring frame's origin, or excluding the neighbouring frame's body, still returns a
plausible picture. The third failure is quieter still: a pattern that shares an origin
falling back to the loop is correct and ten times slower.

So both paths are compared against `mj_ray` called ray by ray, in a scene built so that
the excluded body, the geom groups, the frame's origin and the environment's state each
change the answer -- asserted first, so the comparison cannot pass vacuously -- and
which function runs is pinned.

Checked against `raycast_into` broken three ways, one at a time: every frame cast from
the first frame's origin, every frame excluding the first frame's body, and the shared
path disabled. The first two fail the comparison and the third fails the pin.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")

from mjrl.sensor.raycast import raycast_into  # noqa: E402

XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <geom name="wall" type="box" size="0.05 2 1" pos="2 0 1"/>
    <!-- In front of the eyes and in group 3, which the mask leaves out. -->
    <geom name="hidden" type="box" size="0.05 0.3 0.3" pos="0.5 0 0.5" group="3"/>
    <!-- Each eye sits inside its own body's sphere; only exclusion lets it see out. -->
    <body name="eye_a" pos="0 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.05"/>
    </body>
    <body name="eye_b" pos="0 0.8 0.3">
      <freejoint/>
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

#: mjwarp's vec6 convention: -1 includes a group, 0 leaves it out.
GROUPS_0_TO_2 = (-1, -1, -1, 0, 0, 0)
MASK_0_TO_2 = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)


def _scene():
    model = mujoco.MjModel.from_xml_string(XML)
    datas = [mujoco.MjData(model) for _ in range(2)]
    # The second environment's eyes are elsewhere, so mixing environments shows.
    datas[1].qpos[0:3] = [0.3, -0.4, 0.7]
    datas[1].qpos[7:10] = [-0.2, 1.1, 0.4]
    for d in datas:
        mujoco.mj_forward(model, d)
    return model, datas


def _fan(n_u: int = 9, n_v: int = 7) -> np.ndarray:
    """Unit directions fanned around +x, from well below the horizon to above it."""
    u = np.linspace(-0.8, 0.8, n_u)
    v = np.linspace(-0.9, 0.5, n_v)
    uu, vv = np.meshgrid(u, v)
    dirs = np.stack([np.ones(uu.size), uu.ravel(), vv.ravel()], axis=1)
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def _sensor(pnt: np.ndarray, vec: np.ndarray, exclude: list[int], offsets: np.ndarray,
            frames: int):
    """A stand-in carrying exactly the attributes `raycast_into` reads."""
    n_env, n_ray, _ = pnt.shape

    def vec3(a):
        arr = wp.zeros((n_env, n_ray), dtype=wp.vec3, device="cpu")
        wp.to_torch(arr).copy_(torch.as_tensor(a, dtype=torch.float32))
        return arr

    s = type("Sensor", (), {})()
    s._ray_pnt, s._ray_vec = vec3(pnt), vec3(vec)
    s._ray_normal = wp.zeros((n_env, n_ray), dtype=wp.vec3, device="cpu")
    s._ray_dist = wp.zeros((n_env, n_ray), dtype=float, device="cpu")
    s._ray_geomid = wp.zeros((n_env, n_ray), dtype=int, device="cpu")
    s._ray_bodyexclude = wp.array(exclude, dtype=int, device="cpu")
    s._geomgroup = GROUPS_0_TO_2
    s._local_offsets = torch.as_tensor(offsets, dtype=torch.float32)
    s._num_frames = frames
    s._num_rays_per_frame = n_ray // frames
    return s


def _reference(model, datas, pnt, vec, exclude, mask=MASK_0_TO_2):
    """`mj_ray`, one ray at a time, from the same float32 inputs the sensor holds."""
    pnt = pnt.astype(np.float32).astype(np.float64)
    vec = vec.astype(np.float32).astype(np.float64)
    dist = np.empty(pnt.shape[:2])
    gid = np.empty(pnt.shape[:2], dtype=np.int32)
    buf = np.zeros(1, dtype=np.int32)
    for b, d in enumerate(datas):
        for r in range(pnt.shape[1]):
            dist[b, r] = mujoco.mj_ray(model, d, pnt[b, r], vec[b, r], mask, 1, exclude[r], buf)
            gid[b, r] = buf[0]
    return dist, gid


def _shared_origin_rays(model, datas):
    """Two frames, one per eye; every ray of a frame leaves that eye's centre."""
    eyes = [model.body("eye_a").id, model.body("eye_b").id]
    fan = _fan()
    per = len(fan)
    pnt = np.stack([np.concatenate([np.repeat(d.xpos[e][None], per, 0) for e in eyes])
                    for d in datas])
    vec = np.broadcast_to(np.concatenate([fan, fan]), pnt.shape).copy()
    exclude = [eyes[0]] * per + [eyes[1]] * per
    return pnt, vec, exclude, per


def test_shared_origin_frames_go_through_multiray_and_match_mj_ray(monkeypatch) -> None:
    model, datas = _scene()
    pnt, vec, exclude, per = _shared_origin_rays(model, datas)
    want_dist, want_gid = _reference(model, datas, pnt, vec, exclude)

    # The controls: every input the multi-ray call takes once per frame changes
    # the answer here, so a frame cast with the wrong one cannot match.
    hits = want_dist >= 0
    assert hits.any() and (~hits).any(), "the scene needs hits and misses both"
    unexcluded, _ = _reference(model, datas, pnt, vec, [-1] * len(exclude))
    assert (np.abs(unexcluded - want_dist) > 1e-9).all(), "exclusion must matter everywhere"
    unmasked, _ = _reference(model, datas, pnt, vec, exclude, mask=None)
    assert (np.abs(unmasked - want_dist) > 1e-9).any(), "the group mask must matter"
    assert not np.allclose(want_dist[:, :per], want_dist[:, per:]), "frames must differ"
    assert not np.allclose(want_dist[0], want_dist[1]), "environments must differ"

    sensor = _sensor(pnt, vec, exclude, np.zeros((per, 3)), frames=2)

    def loop_used(*args, **kwargs):
        raise AssertionError("a frame with one origin fell back to the mj_ray loop")

    monkeypatch.setattr(mujoco, "mj_ray", loop_used)
    raycast_into(sensor, model, datas)

    np.testing.assert_array_equal(wp.to_torch(sensor._ray_geomid).numpy(), want_gid)
    np.testing.assert_allclose(wp.to_torch(sensor._ray_dist).numpy(),
                               want_dist.astype(np.float32), rtol=0, atol=0)


def test_rays_with_their_own_origins_keep_the_loop(monkeypatch) -> None:
    """A grid gives every ray its own origin, which one `mj_multiRay` call cannot take."""
    model, datas = _scene()
    eye = model.body("eye_a").id
    xs, ys = np.meshgrid(np.linspace(0.2, 1.8, 5), np.linspace(-0.6, 0.6, 4))
    offsets = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 1.0)], axis=1)
    pnt = np.stack([d.xpos[eye][None] + offsets for d in datas])
    vec = np.broadcast_to(np.array([0.0, 0.0, -1.0]), pnt.shape).copy()
    exclude = [eye] * len(offsets)
    want_dist, want_gid = _reference(model, datas, pnt, vec, exclude)
    assert (want_dist >= 0).any(), "the grid has to land on something"

    sensor = _sensor(pnt, vec, exclude, offsets, frames=1)

    def multi_used(*args, **kwargs):
        raise AssertionError("rays with different origins went to mj_multiRay")

    monkeypatch.setattr(mujoco, "mj_multiRay", multi_used)
    raycast_into(sensor, model, datas)

    np.testing.assert_array_equal(wp.to_torch(sensor._ray_geomid).numpy(), want_gid)
    np.testing.assert_allclose(wp.to_torch(sensor._ray_dist).numpy(),
                               want_dist.astype(np.float32), rtol=0, atol=0)
