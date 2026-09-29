"""Randomised values must really reach each environment's `MjModel`.

`_scatter_model` now uses a dirty flag to skip most steps (it is 8-10% of
`sim.step`, and domain randomisation does not write model fields during ordinary
stepping at all -- measured, zero accesses over 100 control steps).

**One missed dirty flag is a silent error**: the randomisation lands in the buffer
but never reaches the models, so every environment runs nominal parameters -- no
error, no NaN, just an absent randomisation. Exactly that has happened here before
(`xfrc_applied` did nothing on native for a long time).

So these tests do not check the flag; they check the **end-to-end result**: can the
value written be read back off each environment's `MjModel`. Three ways of writing
are tested, including the one easiest to miss.
"""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="b" pos="0 0 1"><freejoint/>
      <geom name="ball" type="sphere" size="0.2" mass="1"/></body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def sim():
    from mjrl.backend.native_sim import NativeSimulation

    m = mujoco.MjModel.from_xml_string(XML)
    s = NativeSimulation(4, None, m, "cpu")
    s.expand_model_fields(("geom_friction",))
    yield s
    s.close()


def read_back(sim, env: int) -> float:
    """Read from **that environment's own** MjModel, not from the buffer."""
    return float(sim._models[env].geom_friction[1, 0])


def test_direct_write_reaches_the_per_env_model(sim) -> None:
    """This is how mjlab's randomisation writes: `sim.model.<field>[env_ids] = ...`."""
    sim.model.geom_friction[2, 1, 0] = 0.42
    sim.step()
    assert read_back(sim, 2) == pytest.approx(0.42, abs=1e-6)
    assert read_back(sim, 0) != pytest.approx(0.42, abs=1e-6), (
        "other environments must not be affected"
    )


def test_write_through_a_held_reference_still_propagates(sim) -> None:
    """**The case easiest to miss**: get the tensor, then write several steps later.

    `__getattr__` is no longer called at that point and only
    `_ExpandedView.__setitem__` can raise the dirty flag. If this breaks,
    randomisation silently stops working.
    """
    t = sim.model.geom_friction
    sim.step()          # this clears the dirty flag
    sim.step()
    t[3, 1, 0] = 0.77   # bypassing __getattr__
    sim.step()
    assert read_back(sim, 3) == pytest.approx(0.77, abs=1e-6)


def test_repeated_writes_across_steps(sim) -> None:
    """A different value written every step must be followed every step -- the
    dirty flag cannot work only once."""
    for k in range(5):
        v = 0.1 + 0.1 * k
        sim.model.geom_friction[1, 1, 0] = v
        sim.step()
        assert read_back(sim, 1) == pytest.approx(v, abs=1e-6), f"write {k} did not follow"


def test_scatter_is_skipped_when_nothing_changed(sim) -> None:
    """With nothing touched it should skip -- otherwise the optimisation is moot.

    The criterion is the skipping itself rather than how much faster it is: timing
    is unreliable in a test.
    """
    sim.model.geom_friction[0, 1, 0] = 0.5
    sim.step()
    assert not sim._model_dirty, "the flag should be cleared after a step"
    for _ in range(3):
        sim.step()
        assert not sim._model_dirty, "nothing was touched; the flag must not raise itself"
    # And the values must still be there after skipping -- skipping is not reverting
    assert read_back(sim, 0) == pytest.approx(0.5, abs=1e-6)


def test_recompute_constants_does_not_wipe_the_live_state(sim) -> None:
    """After `recompute_constants()`, every environment's pose must be untouched.

    `mj_setConst(m, d)` treats `d` as scratch: it sets `qpos` to `qpos0`, runs
    kinematics to compute constants like `body_invweight0`, and **restores nothing
    afterwards**. This once passed the live per-environment `MjData`, so one call
    flattened every environment's pose.

    What that looks like in a real run: every one of 4096 robots **piled at the
    world origin** the moment training starts, then migrating back to its grid cell
    one at a time as episodes time out and reset -- which reads as "the picture
    gradually fixed itself" but is physics state erased once. No exception, no NaN.
    """
    import numpy as np
    import torch

    # Put each environment at a distinct position, imitating the env_origins grid
    for i in range(sim.num_envs):
        sim.data.qpos[i, :3] = torch.tensor([float(i) + 1.0, -float(i), 0.5])
    sim.forward()
    before = sim.data.qpos[:, :3].clone()

    sim.recompute_constants()
    sim.forward()

    after = sim.data.qpos[:, :3]
    assert np.allclose(before.numpy(), after.numpy(), atol=1e-6), (
        f"recompute_constants wiped the poses:\n{before}\n->\n{after}"
    )
    # The positions really are all different -- do not let an all-zero case pass
    assert len(set(map(tuple, np.round(after.numpy(), 3)))) == sim.num_envs
