"""The servo torque-speed curve: that it is right, and that it reaches the sim.

Every failure guarded here is silent. A curve that is configured but never
applied trains a policy against a motor the robot does not have, and neither the
reward curves nor the torque logs look wrong -- they look like a slightly
stronger robot. A home pose outside a joint limit saturates one actuator on every
step of every run and reports nothing, which is not hypothetical: both gripper
joints sat a full radian outside the right one's range until it was measured.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]

# The curve exactly as it was handed over, in the rpm it was measured in. The
# implementation under test works in rad/s and is a separate piece of code; this
# stays here as the reference to compare against, so a change to either shows up.
_P, _C, _LAM, _CAP = 1.7464, 293.5, 240.5, 611.0


def reference_tau_max(w_rpm):
    """Maximum output-shaft torque [N*m]; ``w_rpm`` scalar or array."""
    w = np.abs(np.asarray(w_rpm, dtype=float))
    t = np.minimum(_P, _P * np.exp(-np.maximum(w - _C, 0.0) / _LAM))
    return np.where(w >= _CAP, 0.0, t)


def _speeds_rpm() -> np.ndarray:
    """A sweep that lands exactly on both corners, not merely near them."""
    dense = np.linspace(0.0, 900.0, 601)
    edges = np.array([
        _C - 1e-9, _C, _C + 1e-9,          # the plateau/decay join
        _CAP - 1e-6, _CAP, _CAP + 1e-6,    # the discontinuity
        0.0, 1e-12, 1e4,
    ])
    return np.unique(np.concatenate([dense, edges, -edges, -dense]))


def test_torch_curve_matches_the_reference_implementation() -> None:
    """The rad/s implementation must agree with the rpm reference everywhere.

    The unit conversion is the likely error and it is invisible: a curve scaled
    by 2*pi/60 in the wrong direction is still a plausible-looking curve, still
    monotone, still bounded by the plateau. Only comparing against the original
    catches it.
    """
    from tasks.jumper.common.actuator import (
        CORNER_SPEED,
        CUTOFF_SPEED,
        DECAY_SPEED,
        PLATEAU_TORQUE,
        servo_torque_limit,
    )

    rpm = _speeds_rpm()
    rad_s = torch.tensor(rpm * (2.0 * math.pi / 60.0), dtype=torch.float64)
    got = servo_torque_limit(
        rad_s, PLATEAU_TORQUE, CORNER_SPEED, DECAY_SPEED, CUTOFF_SPEED
    ).numpy()
    want = reference_tau_max(rpm)

    worst = float(np.abs(got - want).max())
    assert worst < 1e-9, f"curve differs from the reference by {worst:.3e} N*m"

    # Control: the sweep has to cover the region where the shape matters at all.
    # A straight line from (0, plateau) to (cutoff, 0) -- the model both of
    # mjlab's DC motor actuators implement -- must disagree substantially, or
    # this test would pass against a linear actuator and prove nothing.
    linear = _P * np.clip(1.0 - np.abs(rpm) / _CAP, 0.0, None)
    assert np.abs(linear - want).max() > 0.5, (
        "the sweep does not reach where a linear model differs from this curve"
    )


def test_curve_is_bounded_by_the_plateau_and_zero_past_the_cutoff() -> None:
    """Two properties the rest of the code is allowed to rely on."""
    from tasks.jumper.common.actuator import (
        CORNER_SPEED,
        CUTOFF_SPEED,
        DECAY_SPEED,
        PLATEAU_TORQUE,
        servo_torque_limit,
    )

    w = torch.linspace(-200.0, 200.0, 4001, dtype=torch.float64)
    lim = servo_torque_limit(w, PLATEAU_TORQUE, CORNER_SPEED, DECAY_SPEED, CUTOFF_SPEED)
    assert float(lim.max()) <= PLATEAU_TORQUE + 1e-12
    assert float(lim.min()) >= 0.0
    assert torch.all(lim[w.abs() >= CUTOFF_SPEED] == 0.0)
    on_plateau = lim[w.abs() <= CORNER_SPEED]
    assert on_plateau.numel() > 0
    assert torch.allclose(on_plateau, torch.tensor(PLATEAU_TORQUE, dtype=torch.float64))


def test_the_curve_itself_stays_a_stateless_control_law() -> None:
    """The speed-dependent clip must remain state-free, even though the
    actuator as a whole is not.

    The thermal budget is state, so ``compute`` is overridden and this actuator
    is deliberately *not* fusable -- see `ServoCurveActuator.compute` for why
    that costs nothing here. But the curve is still expressed in ``control_law``
    as a pure function of parameters and command, and ``compute`` reaches it by
    lowering ``force_limit``. That split is what keeps the curve testable on its
    own and keeps the thermal derate from having to know about the curve.

    Also checks that every parameter the control law reads is named in
    ``param_names``, since ``compute`` builds its dict from that list: a
    parameter left out of it becomes a KeyError, but one added to
    ``initialize`` alone is silently never read.
    """
    from mjlab.actuator.pd_actuator import IdealPdActuator

    from tasks.jumper.common.actuator import ServoCurveActuator

    assert ServoCurveActuator.compute is not IdealPdActuator.compute, (
        "the thermal budget is state and cannot live in a stateless control law"
    )
    assert ServoCurveActuator.control_law is not IdealPdActuator.control_law
    for name in ("plateau_torque", "corner_speed", "decay_speed", "cutoff_speed"):
        assert name in ServoCurveActuator.param_names, (
            f"{name} is set by initialize() but missing from param_names"
        )
    # The state lives on the instance, not in the control law's inputs.
    assert "_thermal" not in ServoCurveActuator.param_names


def test_thermal_time_constant_reproduces_the_budget_it_came_from() -> None:
    """The derived constant must give back the 300 ms it was derived from.

    `thermal_time_constant` is the one number here that is neither measured nor
    written down -- it is solved for. Integrating the model forward and checking
    that a cold joint holding the plateau trips at exactly `peak_duration`
    closes that loop; an algebra slip would otherwise produce a plausible
    constant that derates at the wrong time.
    """
    from tasks.jumper.common.actuator import CURVE

    dt = 1e-5
    theta = 0.0
    steady = (CURVE.plateau_torque / CURVE.continuous_torque) ** 2
    t = 0.0
    while theta < 1.0 and t < 10.0:
        theta += (steady - theta) * (dt / CURVE.thermal_time_constant)
        t += dt
    assert t == pytest.approx(CURVE.peak_duration, rel=1e-3), (
        f"holding the plateau from cold trips at {t:.4f} s, but the spec says "
        f"{CURVE.peak_duration} s"
    )

    # Control: holding exactly the continuous rating must never trip, or the
    # trip point is not where "continuous" says it is.
    theta = 0.0
    for _ in range(int(60.0 / dt / 100)):  # 0.6 s is already >> tau_th
        theta += (1.0 - theta) * (dt * 100 / CURVE.thermal_time_constant)
    assert theta < 1.0 + 1e-9 and theta > 0.99


def test_jumper_actuator_config_is_the_curve() -> None:
    """The robot must actually be built with it.

    This is the failure the rest of the file cannot see: the curve can be
    correct, tested and documented while `make_actuator_cfg` returns a
    constant-limit actuator, and every training run quietly ignores it.
    """
    from tasks.jumper.common.actuator import PLATEAU_TORQUE, ServoCurveActuatorCfg
    from tasks.jumper.common.constants import EFFORT_LIMIT, get_jumper_robot_cfg

    actuators = get_jumper_robot_cfg().articulation.actuators
    assert len(actuators) == 1
    cfg = actuators[0]
    assert isinstance(cfg, ServoCurveActuatorCfg), (
        f"the jumper robot is built with {type(cfg).__name__}, not the servo curve"
    )
    assert EFFORT_LIMIT == PLATEAU_TORQUE
    assert cfg.effort_limit == PLATEAU_TORQUE


@pytest.mark.parametrize("task", ["jumper.flat", "jumper.ripple", "jumper.tetrapod", "jumper.tripod"])
def test_every_jumper_task_gets_the_servo_in_both_modes(task: str) -> None:
    """Training and replay must build the same actuator, on every task.

    `scripts/play.py` builds with ``play=True`` and `scripts/train.py` without,
    and the play branch of `velocity_env_cfg` rewrites a good deal of the config
    -- observation noise, disturbances, episode length, curriculum. If it ever
    reached the robot entity, a policy would be watched under a motor it was not
    trained against, and the replay would look completely normal.

    Config-level and therefore cheap, which is the point: all four tasks in both
    modes without building anything.
    """
    import tasks

    from tasks.jumper.common.actuator import (
        CONTINUOUS_TORQUE,
        CORNER_SPEED,
        CUTOFF_SPEED,
        DECAY_SPEED,
        PLATEAU_TORQUE,
        THERMAL_TIME_CONSTANT,
        ServoCurveActuatorCfg,
    )
    from tasks.jumper.common.constants import DAMPING, STIFFNESS

    expected = {
        "stiffness": STIFFNESS,
        "damping": DAMPING,
        "effort_limit": PLATEAU_TORQUE,
        "plateau_torque": PLATEAU_TORQUE,
        "corner_speed": CORNER_SPEED,
        "decay_speed": DECAY_SPEED,
        "cutoff_speed": CUTOFF_SPEED,
        "continuous_torque": CONTINUOUS_TORQUE,
        "thermal_time_constant": THERMAL_TIME_CONSTANT,
    }
    for play in (False, True):
        entities = tasks.load_env_cfg(task, play=play).scene.entities
        actuators = entities["robot"].articulation.actuators
        mode = "play" if play else "train"
        assert len(actuators) == 1, f"{task} {mode}: {len(actuators)} actuators"
        cfg = actuators[0]
        assert isinstance(cfg, ServoCurveActuatorCfg), (
            f"{task} {mode} builds {type(cfg).__name__}, not the servo curve"
        )
        for name, value in expected.items():
            assert getattr(cfg, name) == pytest.approx(value), (
                f"{task} {mode}: {name} is {getattr(cfg, name)}, expected {value}"
            )


def _spec() -> dict:
    return yaml.safe_load((REPO / "assets/jumper/motor/motor_config.yaml").read_text())


def test_the_curve_is_loaded_from_the_specification() -> None:
    """The spec is the source of truth; the module constants come from it.

    Not a comparison of two copies -- there is only one. This pins that the
    loader is what produces the constants, with the rpm-to-rad/s conversion
    applied exactly once and in the right direction.
    """
    from tasks.jumper.common import actuator as A

    ts = _spec()["motors"]["joint_servo"]["torque_speed"]
    rpm = 2.0 * math.pi / 60.0

    assert A.PLATEAU_TORQUE == pytest.approx(ts["plateau_torque_nm"])
    assert A.CORNER_SPEED == pytest.approx(ts["corner_speed_rpm"] * rpm)
    assert A.DECAY_SPEED == pytest.approx(ts["decay_speed_rpm"] * rpm)
    assert A.CUTOFF_SPEED == pytest.approx(ts["cutoff_speed_rpm"] * rpm)
    # The conversion has a direction, and the wrong one still looks like a servo.
    assert A.CORNER_SPEED < ts["corner_speed_rpm"]


@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda ts: ts.__setitem__("plateau_torque_nm", None), "null"),
        (lambda ts: ts.pop("cutoff_speed_rpm"), "missing"),
        (lambda ts: ts.__setitem__("decay_speed_rpm", "240.5"), "not a number"),
        (lambda ts: ts.__setitem__("decay_speed_rpm", 0.0), "must be positive"),
        (lambda ts: ts.__setitem__("corner_speed_rpm", 900.0), "corner_speed_rpm <"),
        (lambda ts: ts.__setitem__("model", "linear"), "implements"),
    ],
)
def test_loader_refuses_every_ambiguous_specification(tmp_path, mutate, expect) -> None:
    """Each of these would otherwise become a silent default.

    A `null` that quietly reads as zero, a missing cutoff that disables the
    ceiling, a model name naming a shape this actuator does not implement -- all
    of them produce a running simulation with a motor nobody chose. The loader
    has to stop the import instead, so the failure arrives before the first
    training step rather than inside the results.
    """
    from tasks.jumper.common.actuator import load_servo_curve

    doc = _spec()
    mutate(doc["motors"]["joint_servo"]["torque_speed"])
    path = tmp_path / "motor_config.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")

    with pytest.raises(ValueError, match=expect):
        load_servo_curve(path)


def test_loader_reports_a_missing_specification() -> None:
    from tasks.jumper.common.actuator import load_servo_curve

    with pytest.raises(FileNotFoundError, match="torque-speed curve"):
        load_servo_curve(REPO / "assets/jumper/motor/does_not_exist.yaml")


def test_specification_does_not_restate_what_the_mjcf_owns() -> None:
    """`in_mjcf_*` must equal what the MJCF actually compiles to.

    Those three exist so the measured friction can be read next to what the
    simulation runs, which is the point -- they are *deliberately* different
    from the `measured_*` values. But they are copies of numbers that live in
    jumper.xml, and jumper.xml is generated: regenerate it with a different default
    and the comparison table starts lying while every test still passes.
    """
    import mujoco

    from tasks.jumper.common.constants import HOME

    friction = _spec()["motors"]["joint_servo"]["friction"]
    model = mujoco.MjModel.from_xml_path(str(REPO / "assets/jumper/jumper.xml"))
    dofs = [
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in HOME
    ]
    for field, key in (
        ("dof_armature", "in_mjcf_armature_kg_m2"),
        ("dof_frictionloss", "in_mjcf_frictionloss_nm"),
        ("dof_damping", "in_mjcf_viscous_damping_nm_s_per_rad"),
    ):
        actual = getattr(model, field)[dofs]
        assert np.allclose(actual, friction[key]), (
            f"the spec says {key} = {friction[key]}, the MJCF compiles to "
            f"{actual.min()}..{actual.max()}"
        )

    # And the gains quoted beside them are constants.py's, not a third opinion.
    from tasks.jumper.common.constants import DAMPING, STIFFNESS

    control = _spec()["motors"]["joint_servo"]["control"]
    assert control["sim_stiffness_nm_per_rad"] == pytest.approx(STIFFNESS)
    assert control["sim_damping_nm_s_per_rad"] == pytest.approx(DAMPING)


def test_specification_claims_about_the_curve_are_true_of_the_code() -> None:
    """The spec asserts three things about behaviour; none of them is free.

    `torque_just_below_cutoff_nm` is a derived number written by hand, and the
    two booleans describe what `control_law` does. Change the clamp to be
    asymmetric, or move a curve parameter, and the file goes on asserting the
    old behaviour to whoever reads it next.
    """
    from tasks.jumper.common.actuator import (
        CORNER_SPEED,
        CUTOFF_SPEED,
        DECAY_SPEED,
        PLATEAU_TORQUE,
        servo_torque_limit,
    )

    ts = _spec()["motors"]["joint_servo"]["torque_speed"]
    args = (PLATEAU_TORQUE, CORNER_SPEED, DECAY_SPEED, CUTOFF_SPEED)

    just_below = torch.tensor([CUTOFF_SPEED - 1e-6], dtype=torch.float64)
    assert float(servo_torque_limit(just_below, *args)) == pytest.approx(
        ts["torque_just_below_cutoff_nm"], abs=5e-5
    )

    assert ts["cutoff_is_discontinuous"] is True
    at_cutoff = torch.tensor([CUTOFF_SPEED], dtype=torch.float64)
    assert float(servo_torque_limit(at_cutoff, *args)) == 0.0

    # `symmetric_in_torque_sign` is a claim about the clamp, not about the curve:
    # that driving and braking get the same bound. mjlab's dc_motor_clip does
    # not, so this is the field that records the difference, and it has to be
    # checked on the control law rather than on the limit function.
    assert ts["symmetric_in_torque_sign"] is True
    from mjlab.actuator.actuator import ActuatorCmd

    from tasks.jumper.common.actuator import ServoCurveActuator

    speed = torch.tensor([[5.0, 35.0, 60.0]])
    params = {
        "stiffness": torch.full_like(speed, 10.0),
        "damping": torch.zeros_like(speed),
        "force_limit": torch.full_like(speed, PLATEAU_TORQUE),
        "plateau_torque": torch.full_like(speed, PLATEAU_TORQUE),
        "corner_speed": torch.full_like(speed, CORNER_SPEED),
        "decay_speed": torch.full_like(speed, DECAY_SPEED),
        "cutoff_speed": torch.full_like(speed, CUTOFF_SPEED),
    }

    def clipped(sign: float) -> torch.Tensor:
        # A position error far past saturation, so the output is the bound itself.
        cmd = ActuatorCmd(
            position_target=torch.full_like(speed, sign * 100.0),
            velocity_target=torch.zeros_like(speed),
            effort_target=torch.zeros_like(speed),
            pos=torch.zeros_like(speed),
            vel=speed,
        )
        return ServoCurveActuator.control_law(params, cmd)

    drive, brake = clipped(+1.0), clipped(-1.0)
    assert torch.allclose(drive, -brake), (
        "driving and braking are bounded differently, but the spec claims the "
        f"clamp is symmetric: {drive.tolist()} against {brake.tolist()}"
    )
    assert float(drive.min()) > 0.0  # control: the bounds are not all zero


def test_home_pose_is_inside_the_joint_limits() -> None:
    """Every HOME target must lie inside its joint's range.

    Both gripper joints homed at -1.0 while the right one's range is [0, 1.6],
    so MuJoCo pinned it at the limit and the PD pushed kp * 1.0 = 10 N*m into the
    constraint on 99.9% of steps, for the entire history of this repository. The
    robot stood, trained and exported normally throughout: the grippers are not
    in GAIT_JOINTS, so no reward, termination or observation could see it.

    The home pose is the one configuration every run starts from and no other
    check looks at it.
    """
    import mujoco

    from tasks.jumper.common.constants import HOME

    model = mujoco.MjModel.from_xml_path(str(REPO / "assets/jumper/jumper.xml"))
    offenders = []
    for name, target in HOME.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid != -1, f"HOME names {name}, which the model does not have"
        lo, hi = model.jnt_range[jid]
        if not lo <= target <= hi:
            offenders.append(f"{name}: {target} outside [{lo:.4f}, {hi:.4f}]")
    assert not offenders, "HOME targets outside their joint range:\n  " + "\n  ".join(offenders)


NENV = 8
STEPS = 20


@pytest.fixture(scope="module")
def sim():
    """One built environment, shared by the parameter checks below.

    Every test that mutates a parameter tensor restores it in a ``finally``;
    they are in-place writes into the fused group's tensors, which is what
    domain randomisation does, so a leak would silently change later tests.
    """
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    res = resolve(backend="native", device="cpu", num_envs=NENV)
    use_backend(res)  # must precede env construction; the seam is read once

    from mjlab.envs import ManagerBasedRlEnv

    from tasks.jumper.flat.env_cfg import env_cfg

    cfg = env_cfg()
    cfg.scene.num_envs = NENV
    env = ManagerBasedRlEnv(cfg, device=res.device)
    robot = env.scene["robot"]
    g = torch.Generator(device="cpu").manual_seed(1234)
    seq = 6.0 * torch.randn(STEPS, NENV, env.action_space.shape[1], generator=g)
    yield env, robot, robot.actuators[0], seq, res.device
    env.close()


def _rollout(env, robot, seq, device, record: bool = False):
    """Replay the fixed action sequence from a seeded reset."""
    from tasks.jumper.common.actuator import ServoCurveActuator

    torch.manual_seed(0)
    np.random.seed(0)
    env.reset()
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = ServoCurveActuator.control_law
    if record:
        def recording(params, cmd):
            out = original(params, cmd)
            pairs.append((cmd.vel.detach().clone(), out.detach().clone()))
            return out

        ServoCurveActuator.control_law = staticmethod(recording)
    try:
        trace = [
            (env.step(seq[i].to(device)), robot.data.joint_pos.detach().clone())[1]
            for i in range(STEPS)
        ]
    finally:
        if record:
            ServoCurveActuator.control_law = staticmethod(original)
    return torch.stack(trace), pairs


def test_compiled_model_carries_the_actuator_parameters(sim) -> None:
    """armature, damping, frictionloss and the force range reached the model.

    Three of these come from different places -- armature from constants.py
    through the actuator config, damping and frictionloss preserved from the
    MJCF because the config passes None -- and a config field that never lands
    in the model changes nothing and reports nothing.
    """
    import mujoco

    from tasks.jumper.common import constants as K
    from tasks.jumper.common.actuator import PLATEAU_TORQUE

    env, _, _, _, _ = sim
    mj = env.sim.mj_model
    dofs = [
        mj.jnt_dofadr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in K.HOME
    ]

    assert mj.nu == 22
    # <motor>, not <position>: the PD now lives in the Python control law, and a
    # built-in position actuator would silently add a second one in the model.
    assert all(mj.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED for a in range(mj.nu))
    assert all(mj.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE for a in range(mj.nu))

    fr = mj.actuator_forcerange[: mj.nu]
    assert np.allclose(fr[:, 0], -PLATEAU_TORQUE) and np.allclose(fr[:, 1], PLATEAU_TORQUE)
    assert np.allclose(mj.dof_armature[dofs], K.ARMATURE)
    assert np.allclose(mj.dof_damping[dofs], 0.017)
    assert np.allclose(mj.dof_frictionloss[dofs], 0.011)


def test_runtime_parameters_reach_the_control_law(sim) -> None:
    """The tensors hold the configured values and stay views into the fused group.

    `fused_group.py` documents the invariant: after fusion an actuator's
    parameter tensors are *views* into the group's, so mutations must be
    in-place. A rebind would detach them, domain randomisation would write to a
    tensor nobody reads, and the run would look normal.
    """
    from tasks.jumper.common import constants as K
    from tasks.jumper.common.actuator import (
        CORNER_SPEED,
        CUTOFF_SPEED,
        DECAY_SPEED,
        PLATEAU_TORQUE,
    )

    env, robot, act, _, _ = sim
    expected = {
        "stiffness": K.STIFFNESS,
        "damping": K.DAMPING,
        "force_limit": PLATEAU_TORQUE,
        "plateau_torque": PLATEAU_TORQUE,
        "corner_speed": CORNER_SPEED,
        "decay_speed": DECAY_SPEED,
        "cutoff_speed": CUTOFF_SPEED,
    }
    for name, value in expected.items():
        t = getattr(act, name)
        assert t is not None, f"{name} was never initialized"
        assert tuple(t.shape) == (NENV, 22), f"{name} has shape {tuple(t.shape)}"
        assert torch.allclose(t, torch.full_like(t, value)), f"{name} is not {value}"

    # Stateful, so it takes the per-actuator path rather than the fused one.
    assert robot._fused_actuator_group._groups == []
    assert list(robot._custom_actuators) == [act]

    # The thermal state is per environment and per joint, and starts cold.
    assert act._thermal is not None
    assert tuple(act._thermal.shape) == (NENV, 22)
    assert act._substep_dt == pytest.approx(env.cfg.sim.mujoco.timestep), (
        "the heat integrator's dt must be the physics substep, since compute "
        "runs once per substep"
    )


def test_every_runtime_parameter_is_load_bearing(sim) -> None:
    """Perturb each parameter; the robot must move differently.

    This is what catches a parameter that is configured, asserted and ignored --
    read from the wrong dict key, or left out of ``param_names`` so the fused
    group never gathers it. Such a parameter passes every check above and moves
    nothing.

    **Replaying the same actions is not bit-reproducible**, so "the trajectory
    changed" needs a scale. The floor is measured here rather than assumed, and
    every perturbation has to clear it by two orders of magnitude.
    """
    env, robot, act, seq, device = sim

    base, _ = _rollout(env, robot, seq, device)
    again, _ = _rollout(env, robot, seq, device)
    floor = float((base - again).abs().max())
    assert floor < 1e-4, (
        f"two identical runs already differ by {floor:.3e} rad, which is too "
        "coarse to attribute any difference below to the perturbation"
    )
    threshold = max(100.0 * floor, 1e-4)

    for name, factor in (
        ("stiffness", 0.5),
        ("damping", 4.0),
        ("force_limit", 0.4),
        ("plateau_torque", 0.4),
        ("corner_speed", 0.2),
        ("decay_speed", 0.15),
        ("cutoff_speed", 0.35),
    ):
        t = getattr(act, name)
        keep = t.clone()
        t[:] = keep * factor
        try:
            perturbed, _ = _rollout(env, robot, seq, device)
        finally:
            t[:] = keep
        diff = float((perturbed - base).abs().max())
        assert diff > threshold, (
            f"scaling {name} by {factor} moved the robot {diff:.3e} rad, within "
            f"the {threshold:.1e} noise floor -- the parameter is being ignored"
        )


def test_peak_budget_derates_after_300ms_in_simulation(sim) -> None:
    """A joint held at peak must lose the peak after 300 ms, and get it back.

    This is the constraint the curve alone could not express: the curve bounds
    torque against speed and has no notion of duration, so before the thermal
    state a policy could sit at 1.7464 N*m for an entire episode and nothing
    would object.

    Driven through the real actuator at the real substep rate, not by
    integrating the model by hand -- the model is checked separately, and what
    is at issue here is whether it is wired to anything.
    """
    from tasks.jumper.common.actuator import CONTINUOUS_TORQUE, PEAK_DURATION, PLATEAU_TORQUE

    _, _, act, _, _ = sim
    dt = act._substep_dt
    keep = act._thermal.clone()
    try:
        act._thermal[:] = 0.0
        # Saturating demand at zero speed: the curve allows the whole plateau,
        # so the only thing that can take it away is the budget.
        ceilings = []
        for _ in range(int(0.6 / dt)):
            hot = act._thermal >= 1.0
            ceiling = float(
                torch.where(hot, torch.minimum(act.force_limit, act.continuous_torque),
                            act.force_limit)[0, 0]
            )
            ceilings.append(ceiling)
            act._integrate_heat(torch.full_like(act._thermal, ceiling))

        # The tolerance is not cosmetic: force_limit is float32, so reading it
        # back gives 1.74639999..., and a bare `< PLATEAU_TORQUE` against the
        # float64 constant reports derating on the very first substep.
        first_derate = next(i for i, c in enumerate(ceilings) if c < PLATEAU_TORQUE - 1e-3)
        assert first_derate * dt == pytest.approx(PEAK_DURATION, abs=2 * dt), (
            f"derating started at {first_derate * dt:.3f} s, not {PEAK_DURATION} s"
        )
        assert ceilings[first_derate] == pytest.approx(CONTINUOUS_TORQUE)
        assert ceilings[-1] == pytest.approx(CONTINUOUS_TORQUE), (
            "the joint recovered the peak while still being held at the limit"
        )

        # And it cools: idle at zero torque and the peak must come back.
        for _ in range(int(3.0 / dt)):
            act._integrate_heat(torch.zeros_like(act._thermal))
        assert bool((act._thermal < 1.0).all()), "the budget never recovers"
        assert float(act._thermal.max()) < 0.01

        # An episode must also start cold. Without this the budget would be
        # spent once and never returned across a reset, and since resets are
        # staggered the fleet would split into hot and cold environments for
        # reasons unrelated to what their policies did.
        act._thermal[:] = 7.0
        act.reset()
        assert float(act._thermal.max()) == 0.0, "reset does not clear the thermal state"
    finally:
        act._thermal[:] = keep


def test_derated_ceiling_is_the_smaller_of_continuous_and_the_curve(sim) -> None:
    """Hot, the ceiling is min(continuous, curve(speed)) -- not just continuous.

    Above the corner the curve already allows less than 1.2 N*m, and it has to
    go on binding there. A thermal derate written as a flat replacement rather
    than a minimum would *raise* the ceiling at high speed, which is the failure
    this pins.
    """
    from tasks.jumper.common.actuator import (
        CONTINUOUS_TORQUE,
        CORNER_SPEED,
        CUTOFF_SPEED,
        DECAY_SPEED,
        PLATEAU_TORQUE,
        servo_torque_limit,
    )

    env, robot, act, seq, device = sim
    # A speed where the curve is below the continuous rating.
    fast = 55.0
    curve_here = float(
        servo_torque_limit(
            torch.tensor([fast]), PLATEAU_TORQUE, CORNER_SPEED, DECAY_SPEED, CUTOFF_SPEED
        )
    )
    assert curve_here < CONTINUOUS_TORQUE, "pick a speed where the curve is the tighter bound"

    keep = act._thermal.clone()
    original = type(act).control_law
    original_reset = type(act).reset

    def recording(params, cmd):
        out = original(params, cmd)
        pairs.append((cmd.vel.detach().clone(), out.detach().clone()))
        return out

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        # `_rollout` resets the env, and the actuator's reset zeroes the thermal
        # state -- correctly, which is why the first version of this test
        # measured a cold servo. Neutralise just that one reset so the rollout
        # runs hot throughout.
        type(act).reset = lambda self, env_ids=None: None
        type(act).control_law = staticmethod(recording)
        act._thermal[:] = 5.0  # thoroughly hot
        _rollout(env, robot, seq, device)
    finally:
        type(act).control_law = staticmethod(original)
        type(act).reset = original_reset
        act._thermal[:] = keep

    vel = torch.cat([p[0].flatten() for p in pairs])
    tau = torch.cat([p[1].flatten() for p in pairs])
    curve = servo_torque_limit(vel, PLATEAU_TORQUE, CORNER_SPEED, DECAY_SPEED, CUTOFF_SPEED)
    ceiling = torch.minimum(curve, torch.full_like(curve, CONTINUOUS_TORQUE))
    assert float((tau.abs() - ceiling).max()) <= 1e-5
    # Control: some sample must actually be bounded by the continuous rating
    # rather than by the curve, or the assertion above is just the curve test.
    assert int((curve > CONTINUOUS_TORQUE).sum()) > 0


def test_cutoff_branch_runs_in_simulation(sim) -> None:
    """The zero-torque branch is reached by the simulation, not only by a unit test.

    No joint in a normal rollout comes near 611 rpm, so the branch that makes the
    curve discontinuous would go entirely unexercised in an integration test.
    Lowering the ceiling into the range the robot does reach -- the same in-place
    write domain randomisation uses -- puts it under load.
    """
    env, robot, act, seq, device = sim

    keep = act.cutoff_speed.clone()
    act.cutoff_speed[:] = 12.0
    try:
        _, pairs = _rollout(env, robot, seq, device, record=True)
    finally:
        act.cutoff_speed[:] = keep

    vel = torch.cat([p[0].flatten() for p in pairs])
    tau = torch.cat([p[1].flatten() for p in pairs])
    above = vel.abs() >= 12.0

    assert int(above.sum()) > 0, "the lowered cutoff was never crossed; nothing was tested"
    assert float(tau[above].abs().max()) == 0.0, (
        f"torque above the cutoff reached {float(tau[above].abs().max()):.3e} N*m, "
        "so the cutoff branch is not running"
    )
    # Control: without this the test would also pass on an actuator producing no
    # torque at all.
    assert float(tau[~above].abs().max()) > 0.1


@pytest.mark.parametrize("actuator", ["servocurve", "idealpd"])
def test_torque_in_a_running_env_respects_the_curve(actuator: str) -> None:
    """End to end: the clip reaches the simulation, and the check can fail.

    ``idealpd`` is the control group -- mjlab's PD actuator at the same gains and
    the same flat cap, which is what the robot used before the curve. It must
    violate the curve, or a clean result for ``servocurve`` would only mean the
    rollout never moved fast enough for the curve to differ from a constant.

    **The pair is recorded inside the control law, not after the step.** Sampling
    ``actuator_force`` and ``joint_vel`` after ``env.step`` compares a torque
    clipped against the start-of-substep velocity with a limit recomputed from
    the end-of-substep velocity, and reports violations for a correct actuator.
    """
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    res = resolve(backend="native", device="cpu", num_envs=8)
    use_backend(res)  # must precede env construction; the seam is read once

    from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg
    from mjlab.envs import ManagerBasedRlEnv

    from tasks.jumper.common import constants as K
    from tasks.jumper.common.actuator import (
        CORNER_SPEED,
        CUTOFF_SPEED,
        DECAY_SPEED,
        PLATEAU_TORQUE,
        ServoCurveActuator,
        servo_torque_limit,
    )
    from tasks.jumper.flat.env_cfg import env_cfg

    cfg = env_cfg()
    cfg.scene.num_envs = 8
    klass = ServoCurveActuator
    if actuator == "idealpd":
        klass = IdealPdActuator
        cfg.scene.entities["robot"].articulation.actuators = (
            IdealPdActuatorCfg(
                target_names_expr=(".*_joint",),
                stiffness=K.STIFFNESS,
                damping=K.DAMPING,
                effort_limit=K.EFFORT_LIMIT,
                armature=K.ARMATURE,
            ),
        )

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = klass.control_law

    def recording(params, cmd):
        out = original(params, cmd)
        pairs.append((cmd.vel.detach().clone(), out.detach().clone()))
        return out

    klass.control_law = staticmethod(recording)
    try:
        env = ManagerBasedRlEnv(cfg, device=res.device)
        env.reset()
        torch.manual_seed(0)
        n_act = env.action_space.shape[1]
        for _ in range(25):
            # Deliberately violent: the curve only differs from a constant cap
            # once joints pass the corner speed, and a settled robot never does.
            env.step(12.0 * torch.randn(env.num_envs, n_act, device=res.device))
        env.close()
    finally:
        klass.control_law = staticmethod(original)

    vel = torch.cat([p[0].flatten() for p in pairs])
    tau = torch.cat([p[1].flatten() for p in pairs])
    limit = servo_torque_limit(vel, PLATEAU_TORQUE, CORNER_SPEED, DECAY_SPEED, CUTOFF_SPEED)
    excess = tau.abs() - limit
    past_corner = int((vel.abs() > CORNER_SPEED).sum())

    assert past_corner > 0, (
        "no joint passed the corner speed, so this rollout cannot tell the curve "
        "apart from a constant limit and the result below means nothing"
    )

    if actuator == "servocurve":
        worst = float(excess.max())
        assert worst <= 1e-5, (
            f"torque exceeded the curve by {worst:.6f} N*m in "
            f"{int((excess > 1e-5).sum())} of {vel.numel()} samples"
        )
    else:
        assert float(excess.max()) > 0.1, (
            "the control group respected the curve without being asked to, so a "
            "clean servocurve result would prove nothing"
        )


# ── The gains the deployment contract carries ───────────────────────────


def _load_export():
    """`scripts/export.py`, imported as a module.

    The three entry points share `scripts/_cli.py` and import it as a bare `_cli`,
    which resolves only because they are run as scripts from that directory. Not by
    putting `scripts/` on `sys.path` -- `tests/test_layout.py` forbids that, and
    rightly: which directory you ran from is exactly the implicit dependency this
    repository spent 12 `sys.path.insert` calls learning to avoid.

    **`_cli` is stubbed rather than executed**, because it calls `load_dotenv` at
    import and would write this machine's `.env` into `os.environ` for the rest of
    the session. Nothing under test here touches the three names export takes from
    it. `sys.modules` is restored either way; the module stays alive through the
    reference returned here.
    """
    import importlib.util
    import sys
    import types

    from tasks.paths import REPO_ROOT

    names = ("_cli", "_export_under_test")
    saved = {n: sys.modules.get(n) for n in names}
    try:
        stub = types.ModuleType("_cli")
        # **Any name, rather than a list of them.** The list was
        # `build_parser, print_task_table, resolve_all`; export.py later added
        # `parse_with_task_args` to the same `from _cli import (...)`, and the
        # stub answered with an ImportError that reads as a broken exporter
        # rather than as a stale stub -- both tests below sat red on it.
        #
        # None rather than a mock: nothing under test calls these, and a name
        # that did get called fails at the call with a TypeError instead of
        # quietly returning a stub-shaped object.
        stub.__getattr__ = lambda _name: None
        sys.modules["_cli"] = stub
        spec = importlib.util.spec_from_file_location(
            "_export_under_test", REPO_ROOT / "scripts" / "export.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["_export_under_test"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_the_contract_carries_the_gains_the_simulation_ran(sim) -> None:
    """kp and kd must be the actuator's, not MuJoCo's defaults for those fields.

    This shipped. `ServoCurveActuatorCfg` compiles to a plain torque actuator and
    runs its PD in Python, so `actuator_gainprm[0]` is 1.0 and `actuator_biasprm[2]`
    is 0.0 -- MuJoCo's defaults, not gains -- and `export.py` read exactly those.
    `model_32200`'s `layout.json` went out with `"kp": 1.0, "kd": -0.0` against the
    10.0 and 0.5 the policy trained under.

    Nothing downstream could catch it. `rl-wbc-fsm`'s `contract.cpp` reads them as
    `ctrl.value("kp", 10.0f)`, so a *missing* field deploys correctly and a present
    wrong one runs a tenth of the stiffness with no damping -- a limp, not a crash,
    and the bench cannot tell you either.
    """
    env, _robot, actuator, _seq, _device = sim
    export = _load_export()
    order = list(actuator.target_names)
    kp, kd, eff = export._actuator_scalars(env, order)

    expected_kp = float(actuator.stiffness[0, 0])
    expected_kd = float(actuator.damping[0, 0])
    assert kp == pytest.approx(expected_kp)
    assert kd == pytest.approx(expected_kd)
    assert eff == pytest.approx(float(actuator.force_limit[0, 0]))

    # The control: the pair the bug produced is not the right one, so these
    # assertions can tell them apart. If the gains are ever tuned to 1.0 / 0.0 this
    # test stops meaning anything and wants rewriting rather than deleting.
    assert (expected_kp, expected_kd) != (1.0, 0.0), (
        "the real gains are now MuJoCo's placeholder pair -- this test is vacuous"
    )


def test_the_exporter_refuses_the_placeholder_pair(sim) -> None:
    """And when it cannot read them, it must not write a contract anyway.

    The test above only holds while somebody runs it; the exporter carries its own
    guard for the same failure, because what it protects is a robot on a bench
    rather than a test run.
    """
    env, _robot, actuator, _seq, _device = sim
    export = _load_export()
    order = list(actuator.target_names)

    stiffness = actuator.stiffness.clone()
    damping = actuator.damping.clone()
    try:
        actuator.stiffness[:] = 1.0
        actuator.damping[:] = 0.0
        with pytest.raises(SystemExit, match="MuJoCo's defaults"):
            export._actuator_scalars(env, order)
    finally:
        actuator.stiffness[:] = stiffness
        actuator.damping[:] = damping
