"""The 100:1 servo's torque-speed definition: that it is its samples' fit, and unwired.

`motors.joint_servo_100` in assets/jumper/motor/motor_config.yaml is a fit to
`torque_speed_100_24v_reported.csv`, samples digitised from the bench report. Each
failure pinned here produces a plausible servo rather than an error:

* the three numbers in the spec drifting from the samples they claim to fit -- a
  typo, or a new CSV dropped in without a refit -- leaves a curve nothing traces
  back to a measurement;
* the wrong shape is invisible from its parameters: the exponential the 50:1
  servo uses fits these samples at 2.4 times the error and still looks like a
  servo;
* a definition that looks wired but is not -- `applies_to` filled in while every
  joint still runs `joint_servo` -- is a spec that lies about the robot.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
MOTOR = REPO / "assets" / "jumper" / "motor"
RPM = 2.0 * math.pi / 60.0

COLUMNS = [
    "speed_rpm",
    "pos_reported_nm",
    "neg_reported_nm",
    "mean_reported_nm",
    "band_p10_nm",
    "band_p90_nm",
    "p10_occluded",
    "kt_ratio_pos",
    "kt_ratio_neg",
]


def _spec() -> dict:
    return yaml.safe_load((MOTOR / "motor_config.yaml").read_text(encoding="utf-8"))


def _torque_speed() -> dict:
    return _spec()["motors"]["joint_servo_100"]["torque_speed"]


def _samples() -> dict[str, np.ndarray]:
    path = MOTOR / _torque_speed()["points_file"]
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {name: np.array([float(r[name]) for r in rows]) for name in COLUMNS}


def _refit(speed_rpm: np.ndarray, torque: np.ndarray) -> tuple[float, float, float]:
    """Least-squares plateau-then-line: the fit the spec's three numbers came from.

    For a fixed corner the model is linear in (plateau, slope), so a fine grid over
    the corner with a linear solve at each point finds the global optimum using
    numpy alone. Returns (plateau N*m, corner rpm, zero-torque speed rpm).
    """
    corners = np.arange(55.0, 90.0 + 1e-9, 0.01)
    best = None
    for corner in corners:
        a = np.column_stack(
            [np.ones_like(speed_rpm), np.where(speed_rpm > corner, speed_rpm - corner, 0.0)]
        )
        coef, *_ = np.linalg.lstsq(a, torque, rcond=None)
        sse = float(((a @ coef - torque) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, float(coef[0]), float(corner), float(corner - coef[0] / coef[1]))
    _, plateau, corner, zero = best
    # An optimum on the edge of the grid is the grid's answer, not the data's.
    assert corners[0] < corner < corners[-1], f"corner {corner} is on the grid edge"
    # The line model ignores the clamp at zero, which is only valid inside the data.
    assert zero > speed_rpm.max()
    return plateau, corner, zero


def test_definition_is_the_least_squares_fit_to_its_samples() -> None:
    """The spec's numbers are what fitting the samples file gives, to their precision.

    The comparison is tight enough to tell a fit of the mean of both directions,
    which the spec documents, from a fit of one direction -- the control below --
    which is the likeliest way to refit this wrongly.
    """
    ts = _torque_speed()
    s = _samples()
    tolerance = {"plateau": 1e-3, "corner": 0.05, "zero": 0.05}  # the digits the spec carries

    def mismatches(column: str) -> list[str]:
        plateau, corner, zero = _refit(s["speed_rpm"], s[column])
        out = []
        for name, got, want in (
            ("plateau", plateau, ts["plateau_torque_nm"]),
            ("corner", corner, ts["corner_speed_rpm"]),
            ("zero", zero, ts["zero_torque_speed_rpm"]),
        ):
            if abs(got - want) > tolerance[name]:
                out.append(f"{name}: refit {got:.4f}, spec {want}")
        return out

    assert mismatches("mean_reported_nm") == []
    # Control: the positive direction alone refits to 4.80 N*m / 65.0 / 202.6 rpm.
    assert mismatches("pos_reported_nm"), "the comparison cannot tell one direction from both"


def test_the_shape_fits_its_samples_and_the_50_to_1_shape_does_not() -> None:
    """The code path, fed the spec, reproduces the samples; the old shape cannot.

    Measured when the fit was made: rms 0.077 N*m, 0.237 at worst (97 rpm). The
    bound sits just above that. The control is the exponential the 50:1 servo
    runs, given its best parameters for these same samples: it lands at rms 0.19,
    so the bound is what separates the two shapes rather than a formality.
    """
    from scipy.optimize import least_squares

    from tasks.jumper.common.actuator import (
        linear_decay_torque_limit,
        load_linear_decay_curve,
        servo_torque_limit,
    )

    s = _samples()
    target = s["mean_reported_nm"]
    speed = torch.tensor(s["speed_rpm"] * RPM, dtype=torch.float64)

    c = load_linear_decay_curve()
    fit = linear_decay_torque_limit(speed, c.plateau_torque, c.corner_speed, c.zero_torque_speed)
    err = fit.numpy() - target
    assert np.sqrt(np.mean(err**2)) <= 0.08
    assert np.abs(err).max() <= 0.25

    def exp_residual(p: np.ndarray) -> np.ndarray:
        plateau, corner_rpm, decay_rpm = p
        limit = servo_torque_limit(speed, plateau, corner_rpm * RPM, decay_rpm * RPM, math.inf)
        return limit.numpy() - target

    best = least_squares(exp_residual, x0=[4.7, 70.0, 80.0])
    assert best.success
    assert np.sqrt(np.mean(best.fun**2)) > 0.15, "the bound does not discriminate the shapes"


def test_torch_limit_matches_the_rpm_reference() -> None:
    """The rad/s implementation agrees with the formula in the rpm it was fitted in.

    A curve converted the wrong way is still flat-then-falling and still bounded
    by the plateau; only a comparison against the rpm numbers as written in the
    spec shows it. The sweep lands exactly on the corner and the zero-torque speed
    in both directions, where an off-by-one in the clamp would hide.
    """
    from tasks.jumper.common.actuator import linear_decay_torque_limit, load_linear_decay_curve

    ts = _torque_speed()
    p, c, z = ts["plateau_torque_nm"], ts["corner_speed_rpm"], ts["zero_torque_speed_rpm"]

    edges = np.array([0.0, 1e-12, c - 1e-9, c, c + 1e-9, z - 1e-9, z, z + 1e-9, 1e4])
    w_rpm = np.unique(np.concatenate([np.linspace(0.0, 300.0, 601), edges]))
    w_rpm = np.concatenate([w_rpm, -w_rpm])
    reference = p * np.clip((z - np.abs(w_rpm)) / (z - c), 0.0, 1.0)

    curve = load_linear_decay_curve()
    got = linear_decay_torque_limit(
        torch.tensor(w_rpm * RPM, dtype=torch.float64),
        curve.plateau_torque,
        curve.corner_speed,
        curve.zero_torque_speed,
    ).numpy()
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-9)

    # The conversion has a direction, and the wrong one still looks like a servo.
    assert curve.corner_speed == pytest.approx(c * RPM)
    assert curve.zero_torque_speed == pytest.approx(z * RPM)
    assert curve.corner_speed < c
    assert curve.torque_basis == ts["torque_basis"] == "servo_reported"


@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda ts: ts.__setitem__("plateau_torque_nm", None), "null"),
        (lambda ts: ts.pop("zero_torque_speed_rpm"), "missing"),
        (lambda ts: ts.__setitem__("corner_speed_rpm", "67.3"), "not a number"),
        (lambda ts: ts.__setitem__("plateau_torque_nm", 0.0), "must be positive"),
        (lambda ts: ts.__setitem__("corner_speed_rpm", 250.0), "zero_torque_speed_rpm"),
        (lambda ts: ts.__setitem__("model", "exp_decay_with_cutoff"), "implements"),
        (lambda ts: ts.pop("torque_basis"), "torque_basis"),
        (lambda ts: ts.__setitem__("torque_basis", "reported"), "torque_basis"),
    ],
)
def test_loader_refuses_every_ambiguous_specification(tmp_path, mutate, expect) -> None:
    """Each of these would otherwise become a curve nobody chose.

    The last two are particular to this servo: a curve that does not say whether
    its torque is the servo's report or a shaft measurement is off by a factor of
    about 1.6 in one of the two readings, and nothing downstream could tell which.
    """
    from tasks.jumper.common.actuator import load_linear_decay_curve

    doc = _spec()
    mutate(doc["motors"]["joint_servo_100"]["torque_speed"])
    path = tmp_path / "motor_config.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError, match=expect):
        load_linear_decay_curve(path)


def test_the_100_to_1_servo_is_defined_not_wired() -> None:
    """No joint claims the 100:1 servo, and the robot still runs the 50:1 curve.

    `applies_to` is the spec's statement of which joints carry a servo, and the
    actuator does not read it -- `make_actuator_cfg` targets every joint with the
    curve `load_servo_curve` returns. So the spec and the robot agree only while
    exactly one entry claims joints and it is the one that loader reads.
    """
    from tasks.jumper.common.actuator import load_linear_decay_curve, load_servo_curve
    from tasks.jumper.common.constants import JUMPER_ACTUATOR_CFG

    motors = _spec()["motors"]
    assert motors["joint_servo_100"]["applies_to"] == []
    assert motors["joint_servo_100"]["count_on_robot"] == 0
    claiming = [name for name, m in motors.items() if m.get("applies_to")]
    assert claiming == ["joint_servo"]

    wired = load_servo_curve()
    defined = load_linear_decay_curve()
    # Unequal, or the next assertion could not tell which curve the robot got.
    assert wired.plateau_torque != pytest.approx(defined.plateau_torque)
    assert JUMPER_ACTUATOR_CFG.plateau_torque == pytest.approx(wired.plateau_torque)


def test_samples_file_is_the_one_the_spec_names() -> None:
    """The points file exists, has the documented columns, and they are consistent.

    The mean is the column the spec was fitted to; a regenerated file with two
    columns swapped would still parse and still refit to something, so the
    columns are checked against each other rather than trusted by name.
    """
    path = MOTOR / _torque_speed()["points_file"]
    with path.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == COLUMNS

    s = _samples()
    np.testing.assert_array_equal(s["speed_rpm"], np.arange(52.0, 190.0))
    np.testing.assert_allclose(
        s["mean_reported_nm"], (s["pos_reported_nm"] + s["neg_reported_nm"]) / 2, atol=1e-3
    )
    assert np.all(s["band_p10_nm"] <= s["mean_reported_nm"])
    assert np.all(s["mean_reported_nm"] <= s["band_p90_nm"])
    assert set(np.unique(s["p10_occluded"])) <= {0.0, 1.0}
    for column in ("kt_ratio_pos", "kt_ratio_neg"):
        assert np.all((s[column] > 0.0) & (s[column] < 1.0)), column
