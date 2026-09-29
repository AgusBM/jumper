"""`jumper.five_foot`'s foot sites, read without upstream's broken accessor.

`foot_home_position` compares the foot **sites** with `HOME_TOE_XY`, which is itself
a site measurement (`grasp_pose.py --stance`). mjlab's `EntityData.site_pos_w`
cannot be used for that on the CPU: it slices `site_pose_w`, which hands MuJoCo's
flat `[B, S, 9]` `site_xmat` to `quat_from_matrix` and raises `Invalid rotation
matrix shape` (mjwarp hands the matrices over shaped, so the GPU never sees it).
The task used to carry a fix in the vendored `rl/mjlab/entity/data.py`; it now
reads `site_xpos` directly (`mdp/rewards.py::_site_pos_w`), and that is what these
tests pin:

    the site rows `_site_pos_w` reads            test_the_site_positions_are_the_right_sites
    why it exists, and when it can go            test_the_upstream_accessor_is_still_broken_on_cpu
    foot_home switched back to bodies            test_foot_home_position_reads_the_sites
    bodies passed off as the sites               test_bodies_are_not_a_free_substitute...
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="mjlab's math helpers are torch")
mujoco = pytest.importorskip("mujoco", reason="the ground truth is MuJoCo's own")


@pytest.fixture(scope="module")
def five_foot_env():
    """A small five-foot environment on the native CPU backend, sampled twice.

    Yields `(env, at_home, moved)`, where each sample is `(body_xy, site_xy)`
    for the five ground feet: one taken at reset, one after 200 steps of
    settling and random action. Two samples because the interesting fact about
    foot sites and foot bodies is that they agree at one of those and not the
    other.

    Module-scoped for the same reason `test_five_foot.py`'s is: building it is
    the expensive part, and every test below reads the same run.
    """
    import importlib

    from mjlab.envs import ManagerBasedRlEnv
    from mjrl.backend.select import use_backend

    resolve = importlib.import_module("mjrl.backend.resolve")
    use_backend(resolve.resolve(backend="native", device="cpu", num_envs=8))

    from tasks.jumper.five_foot.claw import FIVE_FOOT_LEGS, LEG_FOOT
    from tasks.jumper.five_foot.mdp.rewards import _site_pos_w
    from tasks.registry import load_env_cfg

    cfg = load_env_cfg("jumper.five_foot", play=True)
    cfg.scene.num_envs = 8
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()

    robot = env.scene["robot"]
    bodies, sites = list(robot.body_names), list(robot.site_names)
    body_ids = [bodies.index(LEG_FOOT[leg]) for leg in FIVE_FOOT_LEGS]
    site_ids = [sites.index(leg) for leg in FIVE_FOOT_LEGS]

    def sample():
        return (
            robot.data.body_link_pos_w[:, body_ids, :2].clone(),
            _site_pos_w(robot)[:, site_ids, :2].clone(),
        )

    at_home = sample()
    torch.manual_seed(0)
    width = env.action_manager.total_action_dim
    for step in range(200):
        zero = step < 50  # settle first, then stir the legs about
        env.step(
            torch.zeros(env.num_envs, width)
            if zero
            else torch.randn(env.num_envs, width) * 0.5
        )
    moved = sample()
    try:
        yield env, at_home, moved
    finally:
        env.close()


def test_the_site_positions_are_the_right_sites(five_foot_env) -> None:
    """`_site_pos_w` indexes MuJoCo's global `site_xpos` with the entity's own ids,
    and an off-by-one there is silent: five plausible positions, of the wrong
    sites. Checked against the model's own name lookup, which is a different path
    to the same rows -- global names, prefixed the way mjlab attaches the robot.
    """
    from tasks.jumper.five_foot.claw import FIVE_FOOT_LEGS
    from tasks.jumper.five_foot.mdp.rewards import _site_pos_w

    env, _, _ = five_foot_env
    robot = env.scene["robot"]
    model = env.sim.mj_model
    got = _site_pos_w(robot)
    raw = env.sim.data.site_xpos
    for leg in FIVE_FOOT_LEGS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{leg}")
        assert gid >= 0, f"no site robot/{leg} in the compiled model"
        column = list(robot.site_names).index(leg)
        assert torch.equal(got[:, column], raw[:, gid]), f"{leg}: _site_pos_w reads another site"


def test_the_upstream_accessor_is_still_broken_on_cpu(five_foot_env) -> None:
    """Why `_site_pos_w` exists, kept as a test so it announces its own end.

    `EntityData.site_pos_w` slices `site_pose_w`, which hands the native backend's
    flat `[B, S, 9]` `site_xmat` to `quat_from_matrix` and raises. This task used to
    carry a fix in the vendored `rl/mjlab/entity/data.py`; it reads positions
    directly instead, so the vendored copy stays upstream's. **When this starts
    failing, upstream has fixed the accessor** and `_site_pos_w` can go back to it.
    """
    env, _, _ = five_foot_env
    # TorchScript re-raises it as its own error type, with the message intact.
    with pytest.raises(Exception, match="Invalid rotation matrix shape"):
        _ = env.scene["robot"].data.site_pos_w


def test_foot_home_position_reads_the_sites(five_foot_env) -> None:
    """`foot_home_position` must read sites, because `HOME_TOE_XY` is a site
    measurement: `grasp_pose.py --stance` derives it from `site_xpos`.

    Bodies would compare the foot positions against targets taken from a
    different point on the same link. This pins the reward term's own config
    rather than restating the geometry -- the failure it is aimed at is somebody
    switching the `SceneEntityCfg` back to `body_names` because at HOME it makes
    no visible difference.
    """
    env, _, _ = five_foot_env
    params = env.reward_manager.get_term_cfg("foot_home").params
    asset_cfg = params["asset_cfg"]
    assert asset_cfg.site_names is not None, (
        "foot_home reads bodies again; HOME_TOE_XY is a site measurement"
    )
    assert asset_cfg.body_names is None
    assert asset_cfg.preserve_order, "the pairing with HOME_TOE_XY is positional"
    # Resolved ids, so this also catches a name that no longer exists on the
    # model: `SceneEntityCfg` turns a missing site into an empty list, not a raise.
    assert len(asset_cfg.site_ids) == len(params["home_xy"]), (
        f"foot_home resolved to {len(asset_cfg.site_ids)} sites for "
        f"{len(params['home_xy'])} home positions"
    )


def test_bodies_are_not_a_free_substitute_for_the_sites(five_foot_env) -> None:
    """Why the old workaround looked free and was not, in one assertion.

    The note on `foot_home_position` used to say `build_jumper.py` places each
    foot site at its body's origin. It never did. On the previous model the site
    sat 2.4 mm (RF) to 5.8 mm (the toe tips) away *in the body frame*, pointing
    nearly straight down at HOME, so the two agreed in xy there and only there.
    **On V1.6 they do not agree even at HOME**: the front feet walk on the rear jaw
    pad, `*F_palm_pad_b_link`, and its site is about 30 mm from the pad's origin.

    Both halves are asserted. The first used to be why reading bodies was
    survivable; on this model it is the reason it no longer is. The second is why
    it was the wrong point to read on either.
    """
    _, (home_body, home_site), (moved_body, moved_site) = five_foot_env

    at_home_mm = (home_body - home_site).norm(dim=-1).max().item() * 1000.0
    moved_mm = (moved_body - moved_site).norm(dim=-1).max().item() * 1000.0

    assert at_home_mm > 1.0, (
        f"sites and body origins are {at_home_mm:.4f} mm apart in xy at HOME; on "
        f"V1.6 the front feet's sites sit about 30 mm off their pads' origins, so "
        f"agreeing here means the model or the site placement changed"
    )
    assert moved_mm > 0.1, (
        f"sites and body origins stayed within {moved_mm:.4f} mm in xy through "
        f"200 steps of motion -- if they really are interchangeable now, the "
        f"note on foot_home_position is wrong the other way round"
    )
