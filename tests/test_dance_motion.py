"""What `jumper.dance` gets wrong without saying so.

Every test here pins a failure that produces no error. A dance task is unusually
rich in them, because the reference is data rather than code: a clip read in the
wrong joint order, indexed against the wrong body list, or resampled off by a
frame produces arrays of exactly the right shape holding exactly the wrong
numbers, and training converges on all of them.

Where an assertion could pass vacuously it is paired with a **control group** --
the same assertion against a deliberately broken input, which must fail. Two of
these were written the obvious way first and passed against the bug they were
meant to catch; both say so.

## The material

Most of this file reads a real clip. The committed `demo.npz` means that is
normally there, so these tests run rather than skip -- which is the point of
committing it: a task configured by data needs its data exercised, and material
fetched through a side channel is material nobody checks against. The music it was
made for is not committed (its source could not be established) and is optional,
so the one test that measures a real track skips without one.

The skips are kept anyway, for the case where someone has swapped the demo out for
their own dance and taken it away again. `test_the_material_check_reports_what_is
_missing` covers the absent case itself, on a temporary directory rather than by
waiting for `media/` to be empty.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mjlab", reason="the dance config needs mjlab installed")

from tasks.jumper.common.constants import HOME  # noqa: E402
from tasks.jumper.common.dance import motion as M  # noqa: E402
from tasks.jumper.dance.env_cfg import MEDIA

CONTROL_DT = 0.02  # 50 Hz: the env's decimation (4) times its timestep (0.005)


def _material():
    try:
        return M.find_material(MEDIA)
    except (FileNotFoundError, ValueError) as e:
        pytest.skip(f"no dance material installed: {e}")


def _export_constant(name: str):
    """Read one module-level constant out of `scripts/export.py` **without
    importing it**.

    The value has to come from the real file: a copy of the term list here would
    drift from the one that actually gates the export, which is the entire point
    of the test that uses it.

    But `export.py` imports its sibling `_cli`, which resolves only with `scripts/`
    on the path -- and putting it there is exactly what
    `test_layout.py::test_no_sys_path_mutation` forbids, for the good reason that
    it makes "which directory you ran from" an implicit dependency. Parsing the
    source is the way to have both: the file is read, nothing is imported, and no
    path is rewritten.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "scripts" / "export.py")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            continue
        value = node.value
        # `frozenset({...})` / `set({...})` wrap the literal we want.
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id not in ("frozenset", "set"):
                raise AssertionError(f"{name} is built by {value.func.id}(), unhandled")
            value = value.args[0]
        return ast.literal_eval(value)
    raise AssertionError(f"{name} is no longer defined in {source}")


# ── The joint remap ───────────────────────────────────────────────────────


def test_the_leg_map_covers_every_joint_exactly_once() -> None:
    """`LEG_JOINTS` is a bijection onto the robot's joints.

    A name repeated would silently drop whichever joint it displaced; a name
    missing would leave a column of the reference unread. Neither changes any
    array's shape.
    """
    flat = M.SOURCE_JOINT_ORDER
    assert len(flat) == len(set(flat)), "a joint name appears twice in LEG_JOINTS"
    assert set(flat) == set(HOME), (
        "LEG_JOINTS and constants.HOME describe different joint sets: "
        f"only in LEG_JOINTS {set(flat) - set(HOME)}, "
        f"only in HOME {set(HOME) - set(flat)}"
    )


def test_the_source_order_is_not_the_entity_order() -> None:
    """The premise of the remap.

    If these two ever became the same list the remap would be a no-op, and this
    file's most important test -- the antisymmetry one below -- would pass whether
    or not the remap were applied at all. Then the remap could be deleted and
    nothing would notice until the day the orders diverged again.
    """
    assert M.SOURCE_JOINT_ORDER != tuple(HOME), (
        "the source's leg order now equals the entity's, which makes the remap "
        "untestable -- see this test's docstring before deleting anything"
    )


def test_the_entity_joint_order_is_home_s_order() -> None:
    """`_entity_joint_names` reads `HOME` instead of building an `Entity`.

    That is a shortcut, and this is the check that keeps it true. If mjlab ever
    orders an entity's joints by something other than declaration order, every
    reference column would land on the wrong joint.
    """
    from mjlab.entity import Entity

    from tasks.jumper.common.constants import get_jumper_robot_cfg

    entity = Entity(get_jumper_robot_cfg())
    assert tuple(entity.joint_names) == M._entity_joint_names()


def test_the_remap_puts_mirrored_joints_on_mirrored_columns() -> None:
    """The remapped columns respect the robot's own mirror symmetry.

    A left joint and its right counterpart must sweep ranges related by this
    robot's mirror rule, which is **not** "negate everything": the sign flips for a
    joint whose axis lies in the sagittal plane and is kept for one normal to it.
    `symmetry.py` has already measured which is which -- only the front arms'
    `J1` keeps its sign -- and this reads that table rather than restating it, so
    the two cannot drift apart. It reads the table and not the rule for a reason:
    `test_symmetry.py` restated the rule, agreed with a table the V1.6 rename had
    silently emptied, and passed.

    Written the obvious way first, assuming every pair negates, and it failed
    against a correct map: `LF_J1_joint` spans [-3.168, -0.843] and `RF_J1_joint`
    spans [-3.156, -0.845], both negative, which under a negate-everything rule
    looks like an error of 4.01 rad. That is also exactly what this test reported
    when the table went stale, which is what sent anyone looking at it.

    Measured on the sample clip: LF_J0 [-1.571, -0.349] against RF_J0
    [+0.349, +1.571]; LR_J2 [-2.198, -1.302] against RR_J2 [+1.307, +2.198].

    **The tolerance is 0.2 rad, and it is loose on purpose.** A choreography is not
    obliged to be symmetric, and this one is not quite: `LF_J2` sweeps
    [-0.957, -0.434] against `RF_J2`'s [+0.346, +0.957], so one end of the
    reflection matches to 0.000 rad and the other is out by 0.088. That is the
    dance, not the map. What a tolerance above it can still catch is a **gross**
    mis-assignment -- a `J0` column landing on a `J1`, which moves things by of
    order a radian.

    So this test does not catch a swapped leg pair, and nothing here should be read
    as though it did: an exchanged pair is still mirror-symmetric and satisfies this
    exactly. `test_every_leg_pair_swap_is_refused` is the one that covers that.
    """
    from tasks.jumper.common.mdp.symmetry import _SIGN_KEEP

    clip = M.load_source(_material().motion)
    names = list(M._entity_joint_names())
    col = {n: clip.joint_pos[:, i] for i, n in enumerate(names)}

    compared = 0
    for left in names:
        if not left.startswith("L"):
            continue
        right = "R" + left[1:]
        lo_l, hi_l = float(col[left].min()), float(col[left].max())
        lo_r, hi_r = float(col[right].min()), float(col[right].max())
        if any(k in left for k in _SIGN_KEEP):
            # Same sign: the intervals should simply coincide.
            err = max(abs(lo_l - lo_r), abs(hi_l - hi_r))
        else:
            # Negated: the interval reflects, so the ends swap.
            err = max(abs(lo_l + hi_r), abs(hi_l + lo_r))
        compared += 1
        assert err < 0.2, (
            f"{left} spans [{lo_l:.3f}, {hi_l:.3f}] and {right} spans "
            f"[{lo_r:.3f}, {hi_r:.3f}], which is not its mirror image under this "
            f"robot's rule (off by {err:.3f} rad)."
        )
    assert compared == len(names) // 2, f"compared {compared} pairs, not 11"


@pytest.mark.parametrize("pair", [(2, 3), (1, 4), (0, 5)],
                         ids=["rear", "middle", "arms"])
def test_every_leg_pair_swap_is_refused(pair, tmp_path) -> None:
    """**The control group that matters**, and it took two attempts.

    Reading the source positionally exchanges a leg pair, and the consequence is a
    robot that trains normally and dances a different dance. So each of the three
    possible exchanges has to be refused by *something*.

    The first version of this checked the mirror-range property above, and it does
    not work: swapping a mirror-symmetric pair produces another mirror-symmetric
    pair, so the ranges still satisfy it exactly. The check was blind to the one
    mistake it existed for, and only running it against a deliberately swapped map
    showed that.

    What does work is `_check_joint_limits`, and it covers all three. This robot's
    left and right joint limits are not the same interval but reflections of it --
    `LM_ankle` allows [-2.400, +1.800] and `RM_ankle` [-1.800, +2.400], the fingers
    [-1.600, 0] against [0, +1.600] -- so a leg given its counterpart's angles is
    asked to leave its own range. Measured, an exchanged pair puts four joints out
    of limits.

    `_check_support_feet` independently catches the rear and middle exchanges,
    because those legs carry the robot and their forward kinematics stops closing;
    it cannot see the arms, which are not support. It is not asserted here because
    the limit check runs first and is cheaper -- its own control group is
    `test_the_commanded_channels_are_refused`.
    """
    material = _material()
    original = dict(M.LEG_JOINTS)
    swapped = dict(original)
    a, b = pair
    swapped[a], swapped[b] = original[b], original[a]

    # A private directory so the real cache is neither read nor overwritten. The
    # music is optional, and absent from the committed material.
    for src in (material.motion, material.audio):
        if src is not None:
            (tmp_path / src.name).symlink_to(src)

    try:
        M.LEG_JOINTS = swapped
        with pytest.raises(ValueError, match="outside the model's limits"):
            M.ensure_motion_npz(CONTROL_DT, directory=tmp_path)
    finally:
        M.LEG_JOINTS = original


# ── The body list ─────────────────────────────────────────────────────────


def test_entity_bodies_are_the_model_s_bodies_without_world() -> None:
    """mjlab's `MotionLoader` slices `body_pos_w` by **entity** body index.

    MuJoCo's body 0 is `world` and the entity's list omits it, so emitting the raw
    model order shifts every body by one and each link is scored against its
    neighbour. Shapes stay right; the reward stays plausible.

    Written the obvious way first -- comparing lengths -- and that passed against
    exactly this bug, since dropping `world` and dropping the last body give the
    same count. It compares the lists.
    """
    from mjlab.entity import Entity

    from tasks.jumper.common.constants import get_jumper_robot_cfg

    cfg = get_jumper_robot_cfg()
    entity = Entity(cfg)
    assert list(entity.body_names) == M._entity_body_names(cfg.spec_fn().compile())


# ── Resampling ────────────────────────────────────────────────────────────


def test_resampling_lands_on_recorded_frames_and_keeps_the_endpoints() -> None:
    """1 kHz to 50 Hz is an exact stride of 20, so no value should be invented.

    An off-by-one here shifts the whole dance by up to 20 ms against the music --
    inaudible in a log, visible in a video, and impossible to attribute later.
    """
    clip = M.load_source(_material().motion)
    joints, root_pos, root_rpy = M._resample(clip, CONTROL_DT)

    stride = round(CONTROL_DT / clip.dt)
    assert stride == 20, f"expected a stride of 20, got {stride}"
    expected = int(np.floor((len(clip.joint_pos) - 1) * clip.dt / CONTROL_DT)) + 1
    assert len(joints) == len(root_pos) == len(root_rpy) == expected

    np.testing.assert_array_equal(joints[0], clip.joint_pos[0])
    np.testing.assert_array_equal(joints[1], clip.joint_pos[stride])
    np.testing.assert_array_equal(joints[-1], clip.joint_pos[(expected - 1) * stride])
    # Every output frame is a recorded frame, not an interpolation between two.
    idx = np.arange(expected) * stride
    np.testing.assert_array_equal(joints, clip.joint_pos[idx])


def test_the_clip_is_long_enough_to_train_on() -> None:
    """An episode is a window into the clip, so the clip must exceed one episode.

    A clip shorter than `EPISODE_S` would make every episode a loop over the whole
    dance and the adaptive start sampler meaningless. It would train.
    """
    from tasks.jumper.dance.env_cfg import EPISODE_S

    clip = M.load_source(_material().motion)
    assert clip.duration_s > 2 * EPISODE_S, (
        f"the clip is {clip.duration_s:.1f} s but an episode is {EPISODE_S} s"
    )


# ── The consistency check ─────────────────────────────────────────────────


def test_the_commanded_channels_are_refused() -> None:
    """The control group for `_check_support_feet`, and the record of a decision.

    `REFERENCE = "measured"` was chosen because forward kinematics on the commanded
    channels does not close: the middle legs sit ~13 mm below the rear ones. That
    reasoning is only worth anything if the check can actually tell them apart, so
    this asserts the commanded reference is **rejected** -- if it ever stops being
    rejected, either the clip changed or the check went slack, and in both cases
    the docstring in `common/dance/motion.py` has become fiction.
    """
    material = _material()
    with pytest.raises(ValueError, match="posture the robot can stand in"):
        M.ensure_motion_npz(
            CONTROL_DT, directory=material.motion.parent, reference="command"
        )


def test_the_measured_channels_are_accepted() -> None:
    """The other half: the default reference converts, and the cache is reused."""
    material = _material()
    out = M.ensure_motion_npz(CONTROL_DT, directory=material.motion.parent)
    assert out.is_file()

    with np.load(out) as data:
        for key in (
            "joint_pos", "joint_vel",
            "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
        ):
            assert key in data, f"MotionLoader requires {key!r}"
        n = data["joint_pos"].shape[0]
        assert data["joint_pos"].shape == (n, len(HOME))
        assert data["body_pos_w"].shape[1] == len(data["body_names"])
        assert list(data["joint_names"]) == list(M._entity_joint_names())
        # Unit quaternions, or every orientation reward is scored against nonsense.
        norms = np.linalg.norm(data["body_quat_w"], axis=-1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"quat norms span {norms.min()}..{norms.max()}"


def test_the_cache_is_keyed_on_the_robot_and_not_only_the_clip() -> None:
    """A new robot revision must not reuse the previous robot's conversion.

    This one is written from an incident rather than from imagination. The cache
    was keyed on the clip, the control rate and `REFERENCE`; merging the V1.6 URDF
    left a conversion built against V1.1.6 in place, and the environment failed
    with `IndexError: index 33 is out of bounds` from inside `MotionLoader`.

    **It raised by luck.** The body count changed, and `MotionLoader` indexes
    bodies, so the mismatch hit an array bound. A revision that renames or reorders
    bodies without changing how many there are loads perfectly and scores every
    link against a different link -- the docstring's silent failure 2, reached
    through the cache instead of through the conversion.

    Control group: perturbing one number in the kinematic tree must change the
    digest. Without it this passes against a `_model_digest` that returns a
    constant, which is exactly the bug it is here to prevent.
    """
    from tasks.jumper.common.constants import get_spec

    model = get_spec().compile()
    material = _material()
    base = M._fingerprint(material.motion, CONTROL_DT, M.REFERENCE, model)
    assert base == M._fingerprint(material.motion, CONTROL_DT, M.REFERENCE, model), \
        "the digest is not stable across calls, so the cache would never hit"

    moved = get_spec().compile()
    moved.body_pos[-1, 2] += 1e-6  # one link, one micrometre
    assert M._fingerprint(material.motion, CONTROL_DT, M.REFERENCE, moved) != base, \
        "a change to the kinematic tree leaves the fingerprint alone"

    renamed = get_spec().compile()
    original = M._entity_body_names
    try:
        M._entity_body_names = lambda m: ["x"] + list(original(m))[1:]
        assert M._fingerprint(material.motion, CONTROL_DT, M.REFERENCE, renamed) != base, \
            "a renamed body leaves the fingerprint alone -- the count-preserving case"
    finally:
        M._entity_body_names = original


def test_a_discontinuous_clip_is_refused() -> None:
    """Two takes concatenated, or a dropped frame.

    After decimation a splice becomes a step the policy is scored against and
    cannot track, which shows up as a reward ceiling rather than as an error. The
    check runs at the control rate, on the frames the policy is scored against;
    the clip as it is is the control, so a check that refused everything fails too.
    """
    clip = M.load_source(_material().motion)
    stride = round(CONTROL_DT / clip.dt)
    joints = clip.joint_pos[::stride].copy()
    M._check_continuity(joints, CONTROL_DT, clip.source)  # the control: accepted
    joints[len(joints) // 2:] += 1.0  # a 1 rad splice
    with pytest.raises(ValueError, match="corner speed"):
        M._check_continuity(joints, CONTROL_DT, clip.source)


# ── The material contract ─────────────────────────────────────────────────


def test_the_material_check_reports_what_is_missing(tmp_path) -> None:
    """The first thing every new user meets.

    Three separate mistakes -- nothing there, the wrong kind, two of a kind -- and
    each names the directory and what it looked for, because "jumper.dance failed to
    build" is not something anyone can act on.
    """
    with pytest.raises(FileNotFoundError, match="does not exist"):
        M.find_material(tmp_path / "absent")

    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError, match="no motion clip"):
        M.find_material(tmp_path)

    (tmp_path / "a.npz").touch()
    assert M.find_material(tmp_path).audio is None, "the music is optional"

    (tmp_path / "a.mp3").touch()
    (tmp_path / "b.mp3").touch()
    (tmp_path / "a.mp4").touch()
    with pytest.raises(ValueError, match="several music files"):
        M.find_material(tmp_path)


def test_short_music_is_refused_and_long_music_is_not(monkeypatch) -> None:
    """Music shorter than the dance means the robot finishes to silence.

    Nothing else in the pipeline looks at the audio, so without this the first
    report is a human watching the video. **It caught a real one**: the first
    no-lyrics track supplied for the sample dance was 81.2 s against a
    choreography of 496 beats at 129.88 bpm, which needs about 229 s -- and 81.2 s
    is 175.9 beats, so it does not even tile.

    The bound is **one-sided**, which the same episode settled in the other
    direction. Refusing anything longer than the clip rejected the sample dance's
    own soundtrack -- 236.1 s, extracted from the video it was rendered into,
    against a 235.7 s clip -- because a render's lead-out runs past the last
    frame. A track that outlasts the choreography is simply not used to the end,
    and duration cannot tell a wrong long track from a right one anyway. What it
    can tell is silence.
    """
    material = _material()
    if material.audio is None:
        pytest.skip("no music in the installed material; the demonstration's is not published")
    clip = M.load_source(material.motion)
    expected = clip.duration_s - clip.audio_start_s

    real = M.audio_duration_s(material.audio)
    if real is None:
        pytest.skip("no readable audio duration in the installed material")
    assert M._check_audio(material, clip), "a passing check should still report"

    # Control group: it has to reject something, or the line above proves nothing.
    monkeypatch.setattr(M, "audio_duration_s", lambda _p: 0.3 * expected)
    with pytest.raises(ValueError, match="dance most of this clip in silence"):
        M._check_audio(material, clip)

    # ...and has to accept a track that merely outlasts the dance.
    monkeypatch.setattr(M, "audio_duration_s", lambda _p: clip.duration_s * 1.5)
    assert M._check_audio(material, clip)


# ── The backend seam ──────────────────────────────────────────────────────


def test_the_task_steps_on_the_native_backend() -> None:
    """**Both backends, or the task is only half built.**

    This is the one test here written from a crash rather than from foresight, and
    it caught the same mistake twice. The task was developed and verified on
    `warp:cuda`, and both times a term that works there died on the first step
    under `--backend native`:

        mdp.nan_detection          reads qacc_warmstart   -- removed, see env_cfg
        mdp.electrical_power_cost  reads qfrc_actuator    -- replaced by
                                                             actuator_power_cost

    `native_sim.py` presents mjwarp's interface over plain MuJoCo and carries a
    deliberately short list of derived fields (`_DERIVED_FIELDS`); anything outside
    it raises `AttributeError` naming the field. That is a good error -- but only
    if something asks. Nothing in the config, the type checker or the other tests
    does, because every term involved is a perfectly ordinary mjlab term that
    happens to read a field one backend gathers and the other does not.

    So: build the environment on native and step it. A few steps and a handful of
    environments are enough -- the failure is on the first call to the offending
    term, not a slow divergence.

    ## It puts the seam back afterwards

    `use_backend` writes two module globals -- mjlab's `_SIMULATION_CLS` and
    `native_sim._DEFAULT_NTHREAD` -- and neither is scoped to anything. Left set,
    the thread count leaks into
    `test_resolve.py::test_backend_and_resolve_agree_on_the_default`, which then
    reads this test's `num_envs` (4) where it expects `DEFAULT_MAX_THREADS` (8).

    That leak is **not new** -- `test_servo_curve.py` calls `use_backend` the same
    way -- it was merely invisible, because pytest collects files in alphabetical
    order and `test_servo_curve` runs *after* `test_resolve`. This file begins with
    a 'd' and so runs before it, which is the only reason the leak ever showed. A
    test whose correctness depends on its filename is not one to rely on, so this
    one restores what it changed.
    """
    import torch
    from mjlab.sim import get_simulation_cls, set_simulation_cls
    from mjrl.backend import native_sim
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    _material()

    saved_cls = get_simulation_cls()
    saved_nthread = native_sim._DEFAULT_NTHREAD
    res = resolve(backend="native", device="cpu", num_envs=4)
    # `use_backend` must run before the env is built: `ManagerBasedRlEnv.__init__`
    # reads the registry once and setting it afterwards does nothing, silently.
    use_backend(res)

    from mjlab.envs import ManagerBasedRlEnv

    from tasks.registry import load_env_cfg

    try:
        cfg = load_env_cfg("jumper.dance")
        cfg.scene.num_envs = res.num_envs
        env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
        try:
            env.reset()
            action = torch.zeros(
                env.num_envs, env.action_manager.total_action_dim, device=env.device
            )
            for _ in range(3):
                env.step(action)
            # Every reward and termination term has now been evaluated at least
            # once, which is the point -- a missing field raises on first read.
            assert set(env.reward_manager.active_terms) == set(cfg.rewards)
            assert set(env.termination_manager.active_terms) == set(cfg.terminations)
        finally:
            env.close()
    finally:
        set_simulation_cls(saved_cls)
        native_sim._DEFAULT_NTHREAD = saved_nthread


# ── Exportability ─────────────────────────────────────────────────────────


def test_every_actor_term_is_deployable_now_that_it_carries_its_recording() -> None:
    """This test used to assert the opposite, twice, and both were true then.

    First it said every actor term was deployable -- and it passed, because
    `_DEPLOY_TERMS` claimed all five of the terms that make this a dance while
    `deploy/fsm/src/obs.rs` had a case for none of them. The export would have
    succeeded, the bundle would have looked fine, and the robot would have
    thrown on the bench.

    Then it said the opposite: `jumper.dance` trains and replays and **cannot go
    on a robot**, recorded rather than left to be discovered, and pinned so that
    when the controller learned those terms it would fail and somebody would
    delete it on purpose. It did not fail when that happened, which is the third
    thing worth writing down: the four terms went into `_DEPLOY_TERMS` and
    `_REFERENCE_TERMS` at once, this test subtracted the second from the first,
    and the tripwire cancelled itself out. A test that cannot fail is not a
    weaker test, it is a different one.

    So it asserts the thing that is true now and is worth keeping true: every
    term this actor observes is one the deployment can build, and the ones that
    can only be built from a recording are exactly the ones this task's contract
    carries a `reference` block for.

    `base_lin_vel` is a different thing and still checked below: buildable, and
    **unmeasurable** on this robot, so it belongs to the critic.
    """
    from tasks.registry import load_env_cfg

    deploy = _export_constant("_DEPLOY_TERMS")
    rename = _export_constant("_TERM_RENAME")
    reference_only = _export_constant("_REFERENCE_TERMS")
    unmeasurable = _export_constant("_UNMEASURABLE_TERMS")

    _material()  # the config converts the clip while it builds
    cfg = load_env_cfg("jumper.dance")

    actor = set(cfg.observations["actor"].terms)
    unknown = actor - (set(deploy) | set(rename))
    assert not unknown, (
        f"the controller has no case for {sorted(unknown)}, so this contract would "
        f"not load on the robot"
    )

    # The reference-only ones are admitted by the `reference` block, not by the
    # name -- `jumper.jump` has a `ref_future` too and means something else by it.
    # So this task must produce a block, and the export gate must see one.
    needs_recording = sorted(actor & set(reference_only))
    assert needs_recording == [
        "clip_phase", "ref_future", "ref_joint_pos", "ref_joint_vel", "ref_tilt_error",
    ], "the set of terms that need the recording changed; say whether that was the point"
    assert callable(getattr(cfg, "reference_contract", None)), (
        f"the actor observes {needs_recording}, which the controller builds only from "
        f"a recording, and this config exports no `reference_contract` to ship one"
    )

    # And nothing unmeasurable, which no controller could ever fix.
    for term, why in unmeasurable.items():
        assert term not in actor, f"the actor observes {term!r}, which it cannot: {why}"


def test_no_video_is_required_to_build_an_environment(tmp_path) -> None:
    """A clip is the whole requirement.

    `find_material` runs at env-build time, so anything it insists on is something
    training cannot start without. The only video the task has a use for is the
    face animation, which training never reads -- requiring it would stop every
    clone that has a choreography but no face screen animation, for nothing.

    This also pins the removal of the reference recording from the contract. It
    was required and never opened; reinstating it would break training on material
    that is complete.
    """
    (tmp_path / "a.npz").touch()
    (tmp_path / "a.mp3").touch()

    found = M.find_material(tmp_path)
    assert found.motion.name == "a.npz"
    assert found.audio.name == "a.mp3"
    assert found.eyes is None

    # Nor is the music, which training never reads either.
    (tmp_path / "a.mp3").unlink()
    assert M.find_material(tmp_path).audio is None

    # The control: the clip *is* required and must still be, or this test would
    # pass against a `find_material` that had stopped checking at all.
    (tmp_path / "a.npz").unlink()
    with pytest.raises(FileNotFoundError, match="no motion clip"):
        M.find_material(tmp_path)


def test_the_face_animation_is_the_one_video_beside_the_clip(tmp_path) -> None:
    """Found in `media/` itself, and ambiguity is still refused.

    Absent, present and ambiguous are three different answers and all three are
    load-bearing. Present is the control group: without it the optional half would
    pass against a `find_material` that ignored videos entirely and always returned
    `None`, and the export would quietly ship a performance with no face.

    Two is an error rather than a pick, for the same reason two clips is: the
    export would otherwise put one of them on the robot's face without saying
    which. A subdirectory is *not* searched -- the old layout is gone, and a file
    left in one is not silently still working.
    """
    (tmp_path / "a.npz").touch()
    (tmp_path / "a.mp3").touch()
    assert M.find_material(tmp_path).eyes is None

    (tmp_path / "blink.mp4").touch()
    assert M.find_material(tmp_path).eyes == tmp_path / "blink.mp4"

    stale = tmp_path / "eyes"
    stale.mkdir()
    (stale / "old.mp4").touch()
    assert M.find_material(tmp_path).eyes == tmp_path / "blink.mp4"

    (tmp_path / "take2.mov").touch()
    with pytest.raises(ValueError, match="several face animation files"):
        M.find_material(tmp_path)


def test_only_the_dance_task_contributes_export_artifacts() -> None:
    """`load_export_media` has to find this task's hook and not invent one.

    The silent failure is one-sided and quiet: if the lookup returned `None` for
    `jumper.dance` -- a renamed module, a renamed function -- `scripts/export.py`
    writes the export, prints its three files, exits 0, and simply never
    mentions that the video it was asked for does not exist.

    The control group is the other four tasks: a lookup that returned something
    truthy for everything would satisfy the first assertion while being just as
    broken.
    """
    import tasks

    hook = tasks.load_export_media("jumper.dance")
    assert callable(hook), "jumper.dance's export_media hook did not resolve"

    for other in ("jumper.flat", "jumper.tripod", "jumper.ripple"):
        assert tasks.load_export_media(other) is None, (
            f"{other} has no export_media.py, so the lookup must say so"
        )


def test_export_actually_reads_the_video_switch() -> None:
    """A flag that is declared and never consulted.

    `--no-video` is the documented way to keep the export fast, and an
    `add_argument` with no reader is invisible: `--no-video` is accepted, argparse
    is happy, and the render runs anyway for the minutes it was meant to save.

    Read from the source for the same reason `_export_constant` does -- importing
    `export.py` needs `scripts/` on the path, which `test_layout.py` forbids.
    """
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "scripts" / "export.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    declared = any(
        isinstance(n, ast.Constant) and n.value == "--video"
        for n in ast.walk(tree)
    )
    read = any(
        isinstance(n, ast.Attribute) and n.attr == "video"
        and isinstance(n.value, ast.Name) and n.value.id == "args"
        for n in ast.walk(tree)
    )
    assert declared, "export.py no longer declares --video/--no-video"
    assert read, "export.py declares --video but never reads args.video"



def test_the_joint_tracking_terms_do_not_pay_for_standing_still() -> None:
    """The std is the whole term, and a loose one is silent.

    `exp(-mse/std^2)` is bounded in (0, 1], so a std at the clip's own amplitude
    scores 1/e for a robot that never moves -- most of the term's range,
    collected for doing nothing. Nothing about that shows up as an error: the
    curve rises, the total reward looks healthy, and the robot stands there.

    **Where the test is taken matters, and the obvious place is wrong.** Scored
    across the whole clip the velocity term gives a motionless robot 0.33, which
    looks alarming and is not: about a third of this choreography is nearly
    still, and matching a still reference *is* tracking it. Averaging `exp(-x)`
    over frames is not `exp(-average x)`, and the quiet frames dominate. So the
    discrimination is asserted where the dance is actually moving, and the quiet
    frames are asserted in the other direction -- a term that punished stillness
    would be broken in a way no total-reward plot would separate from this one.

    MEASURED on the installed clip, standing at HOME throughout: 0.014 on the
    position term, and on the velocity term 0.039 over the frames above median
    speed, 0.010 over the most active quarter. At std 1.5 those become 0.204 and
    0.119, which is the looser choice this pins against.

    Read off `env_cfg` rather than restated here. A copy of the stds in a test is
    a test that keeps passing after somebody changes them.
    """
    import numpy as np

    from tasks.jumper.common.constants import HOME
    from tasks.jumper.common.dance import motion as M
    from tasks.registry import load_env_cfg

    _material()  # the config converts the clip while it builds
    cfg = load_env_cfg("jumper.dance")
    pos_std = cfg.rewards["motion_joint_pos"].params["std"]
    vel_std = cfg.rewards["motion_joint_vel"].params["std"]

    clip = M.load_source(_material().motion)
    names = M._entity_joint_names()
    q = np.asarray(clip.joint_pos)
    qd = np.gradient(q, float(clip.dt), axis=0)
    assert q.shape[1] == len(names) == len(HOME), (q.shape, len(names), len(HOME))

    # The do-nothing policy: hold the home pose, at zero velocity, throughout.
    home = np.array([HOME[n] for n in names])
    pos_score = np.exp(-np.square(q - home).mean(axis=1) / pos_std**2)
    vel_score = np.exp(-np.square(qd).mean(axis=1) / vel_std**2)

    assert pos_score.mean() < 0.05, (
        f"standing still scores {pos_score.mean():.3f} of the joint-position term, "
        f"so most of its range is available for doing nothing. std={pos_std}"
    )

    speed = np.sqrt(np.square(qd).mean(axis=1))
    moving = speed > np.median(speed)
    assert vel_score[moving].mean() < 0.10, (
        f"over the moving half of the clip, standing still scores "
        f"{vel_score[moving].mean():.3f} of the joint-velocity term. std={vel_std}"
    )

    # The other direction: during the quiet passages a motionless robot *is*
    # tracking, and must be paid for it. A term that scored near zero here would
    # be asking the robot to keep moving through the still bars of the music.
    assert vel_score[~moving].mean() > 0.5, (
        f"the quiet frames score {vel_score[~moving].mean():.3f}; matching a still "
        f"reference is tracking it, and this term has stopped saying so"
    )

    # Control: a std chosen the obvious way -- the clip's own RMS amplitude --
    # must pay an order of magnitude more for standing still, or the assertion
    # above is passing for some reason other than the std being well chosen.
    #
    # Compared against the configured score rather than a threshold: the
    # absolute number depends on how far the choreography's mean pose sits from
    # HOME, which is a property of this clip and not of the design.
    loose = float(np.sqrt(np.square(q - q.mean(axis=0)).mean()))
    loose_score = float(np.exp(-np.square(q - home).mean(axis=1) / loose**2).mean())
    assert loose_score > 10 * pos_score.mean(), (
        f"a std at the clip's amplitude ({loose:.3f} rad) scores {loose_score:.3f} "
        f"for standing still against the configured {pos_score.mean():.3f} -- less "
        f"than the 10x that would show the choice of std is doing the work"
    )


def test_the_tilt_error_cannot_see_yaw() -> None:
    """The whole reason `ref_tilt_error` is expressible on this robot.

    Gravity is the only absolute orientation reference the hardware has: the
    IMU is six-axis, so its yaw is a free-running integration. Over the 235.7 s
    this clip runs, that drift is plausibly the same order as the 22.3 deg of
    yaw the choreography actually uses -- so a term that could see yaw would be
    mostly reporting drift, and a policy correcting it would slowly turn the
    robot while tracking perfectly.

    Projected gravity cannot see yaw, because a rotation about the world's
    vertical leaves the gravity direction in the body frame unchanged. That is
    a property of the formula rather than a step somebody has to remember, and
    this is the test that says so -- including the control group, because
    "always returns zero" would pass the first assertion on its own.
    """
    import math

    import torch
    from mjlab.utils.lab_api.math import quat_apply_inverse

    from tasks.jumper.common.dance.observations import _GRAVITY_W

    def quat(roll: float, pitch: float, yaw: float) -> torch.Tensor:
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        return torch.tensor([[
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]])

    g = torch.tensor([list(_GRAVITY_W)])

    def tilt(ref: torch.Tensor, robot: torch.Tensor) -> torch.Tensor:
        return quat_apply_inverse(ref, g) - quat_apply_inverse(robot, g)

    # Same roll and pitch, any yaw apart: no error at all.
    for yaw in (math.pi / 2, math.pi, -math.pi / 3, 2.0):
        e = tilt(quat(0.1, 0.2, 0.0), quat(0.1, 0.2, yaw))
        assert torch.allclose(e, torch.zeros(3), atol=1e-6), f"yaw {yaw}: {e}"

    # Control: a pitch difference is reported, and is still reported when the
    # two also differ in yaw. Without this, a term stuck at zero would pass.
    ten = math.radians(10.0)
    for yaw in (0.0, math.pi / 2):
        e = tilt(quat(0.0, 0.2, 0.0), quat(0.0, 0.2 - ten, yaw))
        assert e.norm() > 0.15, f"yaw {yaw}: 10 deg of pitch went unreported, {e}"
