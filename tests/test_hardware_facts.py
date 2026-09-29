"""Tasks built on a mjlab skeleton directly still train against this robot's hardware.

Every locomotion task builds on `common/velocity_env.py`, which replaces mjlab's
numbers with this robot's from `constants.py`: the IMU noise, the current sensor's
noise, the band the foot friction is randomised over, how hard the base is shoved.
The tasks in `DIRECT` start from one of mjlab's own skeletons instead, so none of
that reaches them unless they apply it themselves -- and neither did:

- `jumper.jump` kept the velocity skeleton's Go1 numbers: a gyro half as noisy as
  the walking policies' (+/-0.2 against +/-0.35 rad/s), a gravity direction at
  under half their tilt error (+/-0.05 against +/-0.12), and feet randomised over
  (0.3, 1.2) where the silicone tips run (0.5, 1.5). It also clamped every joint
  to 300 rpm each control step, on top of the measured torque-speed curve: the
  reference generator's old rectangular motor model, kept after the reference
  itself moved to the curve and started reaching 338 rpm.
- `jumper.dance` wrote the same two IMU numbers out by hand on the tracking
  skeleton, inherited the same friction band, and observed the current sensor
  with no noise at all.

Nothing raised in any of it: every number was plausible, only not this robot's.
`tests/test_task_parity.py` and `tests/test_servo_curve.py` cannot see it -- both
check the four locomotion tasks, and these are deliberately not among them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab", reason="building the configs needs mjlab")

from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

import tasks
from tasks.jumper.common.constants import (
    ACTUATOR_FORCE_NOISE,
    FOOT_GEOMS,
    IMU_ANG_VEL_NOISE,
    IMU_GRAVITY_NOISE,
    JUMPER_FOOT_FRICTION_RANGE,
    PUSH_VELOCITY_RANGE,
)

#: The noise on each sensor's reading, as `constants.py` has it, keyed by the actor
#: term that carries the reading.
SENSOR_NOISE = {
    "base_ang_vel": IMU_ANG_VEL_NOISE,
    "projected_gravity": IMU_GRAVITY_NOISE,
    "actuator_force": ACTUATOR_FORCE_NOISE,
}

#: The tasks that start from one of mjlab's skeletons rather than from
#: `common/velocity_env.py`: the skeleton, and the sensor terms the actor reads.
#: The terms are listed rather than discovered, so that one renamed out from under
#: the check fails it instead of quietly dropping out of it.
DIRECT = {
    "jumper.jump": (make_velocity_env_cfg, ("base_ang_vel", "projected_gravity")),
    "jumper.ref_free_jump": (make_velocity_env_cfg, ("base_ang_vel", "projected_gravity")),
    "jumper.dance": (
        make_tracking_env_cfg,
        ("base_ang_vel", "projected_gravity", "actuator_force"),
    ),
}

#: Facts a task departs from on purpose, and where it says why.
DELIBERATE = {
    ("jumper.dance", "push velocity"): (
        "tasks/jumper/dance/env_cfg.py: gentler and rarer than a walking task's, so "
        "the choreography is not noise"
    ),
}


def expected(sensors) -> dict:
    """What the robot is, as `constants.py` has it."""
    facts = {f"{name} noise": (-SENSOR_NOISE[name], SENSOR_NOISE[name]) for name in sensors}
    facts["foot friction range"] = tuple(JUMPER_FOOT_FRICTION_RANGE)
    facts["foot friction geoms"] = tuple(FOOT_GEOMS)
    facts["push velocity"] = {axis: tuple(r) for axis, r in PUSH_VELOCITY_RANGE.items()}
    return facts


def hardware(cfg, sensors) -> dict:
    """The numbers in `cfg` that describe the robot rather than the task.

    The actor's noise only: it is the group that is corrupted, and the one whose
    counterpart runs on the robot. A term the config does not have reads as
    `"absent"`, and one with no noise as `None`; no fact equals either.
    """
    actor = cfg.observations["actor"].terms
    friction = cfg.events["foot_friction"].params
    push = cfg.events["push_robot"].params["velocity_range"]
    facts = {}
    for name in sensors:
        term = actor.get(name)
        if term is None:
            facts[f"{name} noise"] = "absent"
        elif term.noise is None:
            facts[f"{name} noise"] = None
        else:
            facts[f"{name} noise"] = (term.noise.n_min, term.noise.n_max)
    facts["foot friction range"] = tuple(friction["ranges"])
    facts["foot friction geoms"] = tuple(friction["asset_cfg"].geom_names)
    facts["push velocity"] = {axis: tuple(push[axis]) for axis in PUSH_VELOCITY_RANGE}
    return facts


@pytest.mark.parametrize("task", sorted(DIRECT))
def test_the_task_trains_against_this_robots_hardware(task: str) -> None:
    _, sensors = DIRECT[task]
    got = hardware(tasks.load_env_cfg(task), sensors)
    wrong = {
        k: (got[k], v)
        for k, v in expected(sensors).items()
        if (task, k) not in DELIBERATE and got[k] != v
    }
    assert not wrong, (
        f"{task} is not training against this robot's hardware:\n  "
        + "\n  ".join(f"{k}: {g} where constants.py says {e}" for k, (g, e) in wrong.items())
        + "\n\nIt builds on a mjlab skeleton rather than on common/velocity_env.py, so "
        "each of these has to be applied in its own env_cfg.py -- or, if it departs "
        "on purpose, listed in DELIBERATE with where it says why."
    )


@pytest.mark.parametrize("task", sorted(DIRECT))
def test_the_skeleton_it_builds_on_is_not_this_robot(task: str) -> None:
    """The control group: each fact has to differ from the skeleton's.

    Where one did not, the test above would pass on a config that never read
    `constants.py` at all -- inheriting the default would look exactly like
    applying the fact. This fails when that stops being true of any of them, so
    the check above cannot quietly stop checking.
    """
    skeleton, sensors = DIRECT[task]
    base = hardware(skeleton(), sensors)
    same = [k for k, v in expected(sensors).items() if base[k] == v]
    assert not same, (
        f"{task}'s skeleton already has this robot's {same}, so the test cannot tell "
        f"a task that applied them from one that inherited them"
    )


def test_the_deliberate_departures_still_depart() -> None:
    """An exemption for a fact the task has since come round to is a stale one.

    Left in place it would excuse the next drift on that fact without anyone having
    decided anything, so it has to go as soon as it stops being needed.
    """
    for (task, fact), why in DELIBERATE.items():
        _, sensors = DIRECT[task]
        got = hardware(tasks.load_env_cfg(task), sensors)[fact]
        assert got != expected(sensors)[fact], (
            f"{task} now has this robot's {fact}; drop its DELIBERATE entry ({why})"
        )


@pytest.mark.parametrize("task", sorted(DIRECT))
def test_speed_and_torque_are_the_servo_curve_and_nothing_else(task: str) -> None:
    """How hard a joint pushes and how fast it turns come from one place.

    That place is the measured curve in `assets/jumper/motor/motor_config.yaml`,
    applied by `ServoCurveActuator`: a plateau, a decay past the corner, nothing
    past the cutoff. So the actuator has to be that curve, in training and in
    replay alike, and nothing may rewrite the joint state between control steps
    -- a per-step event is where `jumper.jump`'s 300 rpm wall lived, and a wall
    there is a second motor model the robot does not have.

    The stiffness is not compared: the gain is each task's own (the jump's kp=20
    is argued for in its `env_cfg.py`), and the curve bounds the torque whatever
    the gain asks for.
    """
    from tasks.jumper.common.actuator import (
        CONTINUOUS_TORQUE,
        CORNER_SPEED,
        CUTOFF_SPEED,
        DECAY_SPEED,
        PLATEAU_TORQUE,
        THERMAL_TIME_CONSTANT,
        ServoCurveActuatorCfg,
    )

    curve = {
        "effort_limit": PLATEAU_TORQUE,
        "plateau_torque": PLATEAU_TORQUE,
        "corner_speed": CORNER_SPEED,
        "decay_speed": DECAY_SPEED,
        "cutoff_speed": CUTOFF_SPEED,
        "continuous_torque": CONTINUOUS_TORQUE,
        "thermal_time_constant": THERMAL_TIME_CONSTANT,
    }
    for play in (False, True):
        cfg = tasks.load_env_cfg(task, play=play)
        mode = "play" if play else "train"
        actuators = cfg.scene.entities["robot"].articulation.actuators
        assert len(actuators) == 1, f"{task} {mode}: {len(actuators)} actuators"
        assert isinstance(actuators[0], ServoCurveActuatorCfg), (
            f"{task} {mode} builds {type(actuators[0]).__name__}, not the servo curve"
        )
        for name, value in curve.items():
            got = getattr(actuators[0], name)
            assert got == pytest.approx(value), (
                f"{task} {mode}: {name} is {got}, expected {value}"
            )

        per_step = sorted(name for name, term in cfg.events.items() if term.mode == "step")
        assert not per_step, (
            f"{task} {mode}: {per_step} run every control step. Anything that rewrites "
            "the joint state there limits speed or torque outside the servo curve; if "
            "one is needed for another reason, say so here."
        )
