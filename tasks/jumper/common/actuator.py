"""The jumper servo's measured torque-speed curve, as an mjlab actuator.

The servo is not a DC motor with a straight torque-speed line. Measured at the
output shaft it holds a flat plateau to a corner speed, decays exponentially
above it, and is cut to zero at a hard ceiling:

    tau_max(w) = plateau                                for |w| <= corner
                 plateau * exp(-(|w| - corner) / decay) for corner < |w| < cutoff
                 0                                      for |w| >= cutoff

Neither actuator mjlab ships fits that. `DcMotorActuatorCfg` clips to a straight
line and `BuiltinDcMotorActuatorCfg` derives one from tau = K(V - K*w)/R; fitting
either to this curve is wrong by a factor of two at the corner, where the true
limit is the full plateau and a line through (0, plateau) and (cutoff, 0) gives
roughly half of it. So the curve gets an actuator of its own, here in `tasks/`
rather than as an edit to the vendored `rl/mjlab/`.

**This subclasses `IdealPdActuator` and must not override `compute`.** Keeping the
inherited compute is exactly what marks an actuator as fusable
(`mjlab/actuator/fused_group.py::_is_fusable` compares the bound function), and a
fused group evaluates `control_law` once for all 22 joints instead of per
actuator. Overriding `compute` would still run and still be correct -- it would
just quietly leave the fused path, which is the kind of change nothing reports.

The parameters are per-environment tensors rather than Python floats for the same
reason mjlab's are: the fused group concatenates everything named in
`param_names` along dim 1 and hands the slices back, so domain randomisation can
write into them per environment. `force_limit` (mjlab's name for `effort_limit`)
is applied on top of the curve, so `dr` events that scale motor strength compose
with it instead of fighting it.

The four numbers are **not written here**. They are loaded from
`assets/jumper/motor/motor_config.yaml`, which is the specification the hardware
was measured into, so there is no second copy to drift out of step with it. That
makes an asset file load-bearing for training: `load_servo_curve` therefore
refuses every ambiguous input rather than defaulting past it.

The spec also defines the **100:1 variant** of the servo (`joint_servo_100`), on a
second shape -- flat to a corner, then a straight line to zero torque. It is
**defined, not wired**: `load_linear_decay_curve` reads it and
`linear_decay_torque_limit` evaluates it, and nothing builds an actuator from
either, so every joint still runs the curve above. It is deliberately not loaded
at import: a definition no joint uses must not be able to stop the build.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg, pd_torque

from .assets import MOTOR_CONFIG

#: rpm to rad/s. The curve was measured in rpm and everything MuJoCo touches is in
#: rad/s, so the conversion happens exactly once, in the loader below.
_RPM = 2.0 * math.pi / 60.0

#: The curve shape `ServoCurveActuator` runs. It is checked against the spec's
#: `torque_speed.model` on load: a file describing some other shape must not be
#: silently run through this one, which would keep four plausible numbers and
#: quietly apply the wrong formula to them.
_MODEL = "exp_decay_with_cutoff"

#: The 100:1 servo's shape: flat to a corner speed, then a straight line to zero
#: torque. Only `linear_decay_torque_limit` evaluates it; no actuator runs it yet.
_LINEAR_MODEL = "linear_decay"

#: What a curve's torque axis can measure: the servo's own report of its torque,
#: or a torque sensor on the shaft. On the 100:1 servo the shaft sees 0.61-0.65 of
#: the reported figure, so a curve that does not say which it is cannot be
#: compared with another one, let alone applied to a joint.
_TORQUE_BASES = ("servo_reported", "shaft_measured")


@dataclass(frozen=True)
class ServoCurve:
    """The measured curve and its thermal budget, in SI, at the output shaft."""

    plateau_torque: float  # N*m, available only for peak_duration
    corner_speed: float  # rad/s
    decay_speed: float  # rad/s
    cutoff_speed: float  # rad/s
    continuous_torque: float  # N*m, sustainable indefinitely
    peak_duration: float  # s, how long the plateau may be held from cold

    @property
    def thermal_time_constant(self) -> float:
        """The winding time constant [s] implied by the peak budget.

        **Derived, not measured.** Model the winding as a first-order thermal
        mass, `dtheta/dt = (k*tau^2 - theta) / tau_th`. "Continuous rating" means
        the torque whose steady-state temperature is exactly the trip point, so
        `theta_trip = k * continuous^2`. Holding the plateau from cold,
        `theta(t) = k * plateau^2 * (1 - exp(-t / tau_th))` reaches the trip point
        when

            1 - exp(-t / tau_th) = (continuous / plateau)^2

        and setting that `t` to `peak_duration` fixes `tau_th`. For 1.7464 N*m,
        1.2 N*m and 300 ms it comes to 0.4695 s.

        The point of deriving rather than asking for it is that **cooling is then
        not a free parameter** -- the same constant governs it, so "300 ms at
        peak" and "clamp to continuous afterwards" fully specify the behaviour,
        including partial duty cycles, with no third number to guess.

        What is assumed, and would have to be revisited against a bench
        measurement: that the servo is first-order thermal at all, that the
        episode starts cold, and that its controller derates by clamping rather
        than by some schedule of its own.
        """
        return self.peak_duration / math.log(
            1.0 / (1.0 - (self.continuous_torque / self.plateau_torque) ** 2)
        )


def _torque_speed_spec(
    path: Path, servo: str, model: str, implementer: str
) -> tuple[dict, Callable[[str], float]]:
    """``motors.<servo>.torque_speed`` from the spec, checked to describe ``model``.

    Returns the block and a reader that turns one of its keys into a float or
    raises. Both loaders go through here, so the two shapes refuse exactly the
    same ambiguities.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It holds the servo's measured torque-speed curve "
            "and the simulation cannot be built without it."
        )
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        spec = doc["motors"][servo]["torque_speed"]
    except (TypeError, KeyError) as exc:
        raise ValueError(f"{path}: expected a mapping at motors.{servo}.torque_speed") from exc

    found = spec.get("model")
    if found != model:
        raise ValueError(
            f"{path}: motors.{servo}.torque_speed.model is {found!r}, but {implementer} "
            f"implements {model!r}. Either the spec describes a different curve "
            "shape, in which case it needs code that implements it, or the name is "
            "wrong."
        )

    def number(key: str) -> float:
        where = f"{path}: motors.{servo}.torque_speed.{key}"
        if key not in spec:
            raise ValueError(f"{where} is missing")
        value = spec[key]
        if value is None:
            raise ValueError(
                f"{where} is null. `null` means not measured yet, and there is no "
                "sensible default for it."
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where} is {value!r}, not a number")
        return float(value)

    return spec, number


def load_servo_curve(path: Path = MOTOR_CONFIG, servo: str = "joint_servo") -> ServoCurve:
    """Read the curve the joints run from the servo specification.

    **The specification is the source of truth and this is the only reader of the
    curve the joints run.** Keeping a second copy of the four numbers in Python
    and a test to compare them was the alternative; it works, but it makes the
    file that documents the hardware a mirror of the file that runs it, and the
    useful direction of that relationship is the other way round -- bench numbers
    arrive in the YAML.

    Everything here fails loudly, because every quiet alternative is worse than a
    stopped import. A missing key, a `null` left where a measurement has not
    arrived, a string where a number belongs, a model name this module does not
    implement: each would otherwise become a default, and a default torque-speed
    curve is indistinguishable from a measured one once training starts.
    """
    _, number = _torque_speed_spec(path, servo, _MODEL, "ServoCurveActuator")
    curve = ServoCurve(
        plateau_torque=number("plateau_torque_nm"),
        corner_speed=number("corner_speed_rpm") * _RPM,
        decay_speed=number("decay_speed_rpm") * _RPM,
        cutoff_speed=number("cutoff_speed_rpm") * _RPM,
        continuous_torque=number("thermal_continuous_torque_nm"),
        peak_duration=number("peak_duration_s"),
    )
    if curve.plateau_torque <= 0.0:
        raise ValueError(f"{path}: plateau_torque_nm must be positive")
    if curve.decay_speed <= 0.0:
        raise ValueError(f"{path}: decay_speed_rpm must be positive; it divides an exponent")
    if not 0.0 <= curve.corner_speed < curve.cutoff_speed:
        raise ValueError(
            f"{path}: need 0 <= corner_speed_rpm < cutoff_speed_rpm, got "
            f"{curve.corner_speed / _RPM} and {curve.cutoff_speed / _RPM}"
        )
    if not 0.0 < curve.continuous_torque < curve.plateau_torque:
        # Equal would make the thermal time constant infinite, and greater would
        # make the logarithm undefined -- both are the spec saying the plateau is
        # not a peak after all, which this actuator has no model for.
        raise ValueError(
            f"{path}: need 0 < thermal_continuous_torque_nm < plateau_torque_nm, "
            f"got {curve.continuous_torque} and {curve.plateau_torque}"
        )
    if curve.peak_duration <= 0.0:
        raise ValueError(f"{path}: peak_duration_s must be positive")
    return curve


CURVE = load_servo_curve()

#: Plateau torque [N*m]: the most the servo produces, available from standstill
#: up to the corner speed. It **replaces the 2.0 N*m the repository used to
#: assume** -- see `EFFORT_LIMIT` in constants.py for what that changes.
PLATEAU_TORQUE = CURVE.plateau_torque

#: Corner speed [rad/s]: the plateau ends here and the decay starts.
CORNER_SPEED = CURVE.corner_speed

#: Decay constant [rad/s]: the e-folding speed of the exponential.
DECAY_SPEED = CURVE.decay_speed

#: Cutoff speed [rad/s]: available torque is zero at and above it.
#:
#: **The curve is discontinuous here.** The exponential still stands at 0.4664
#: N*m -- 26.7% of the plateau -- when the cutoff drops it to zero, so a joint
#: crossing the ceiling loses a quarter of peak torque in one step. That is the
#: measurement as given, and it is kept rather than smoothed, but it has two
#: consequences worth knowing: a joint sitting near the cutoff can chatter
#: between full and no torque, and above it the only things left to slow the
#: joint down are `damping` and `frictionloss`, which together produce well under
#: a tenth of a N*m. A real servo brakes hardest at overspeed; this model does
#: not brake at all.
CUTOFF_SPEED = CURVE.cutoff_speed

#: Continuous torque [N*m]: what remains once the peak budget is spent. The
#: ceiling then becomes `min(continuous, curve(speed))` -- above the corner the
#: curve already allows less than this and goes on being the binding constraint.
CONTINUOUS_TORQUE = CURVE.continuous_torque

#: How long the plateau may be held from cold [s].
PEAK_DURATION = CURVE.peak_duration

#: Winding time constant [s], derived from the two above. See `ServoCurve`.
THERMAL_TIME_CONSTANT = CURVE.thermal_time_constant


def servo_torque_limit(
    speed: torch.Tensor,
    plateau: torch.Tensor | float,
    corner: torch.Tensor | float,
    decay: torch.Tensor | float,
    cutoff: torch.Tensor | float,
) -> torch.Tensor:
    """Maximum torque magnitude [N*m] available at ``speed`` [rad/s].

    Symmetric in the sign of the speed and in the sign of the torque: the same
    limit bounds driving and braking. **mjlab's DC motor is asymmetric** --
    `dc_motor_clip` allows more braking than driving at a given speed, which is
    what a real motor does. Whether this servo should be modelled that way is
    open; the measurement supplied is a magnitude, so a magnitude is what this
    applies, and switching would be a change to the clamp on the last line of
    `ServoCurveActuator.control_law` rather than to this function.
    """
    w = speed.abs()
    # The plateau needs no separate min(): the exponent is clamped at zero from
    # above, so the exponential never exceeds 1 and never exceeds `plateau`.
    limit = plateau * torch.exp(-(w - corner).clamp(min=0.0) / decay)
    return torch.where(w >= cutoff, torch.zeros_like(limit), limit)


@dataclass(frozen=True)
class LinearDecayCurve:
    """Flat to a corner speed, then a straight line down to zero torque.

    SI, at the output shaft. ``torque_basis`` says what the torque axis measures:
    ``servo_reported`` is the servo's own estimate of its torque, ``shaft_measured``
    a torque sensor on the bench. There is no thermal budget here -- the one
    servo on this shape has none measured, and `ServoCurve`'s would have to be
    invented to fill the fields.
    """

    plateau_torque: float  # N*m, from standstill to corner_speed
    corner_speed: float  # rad/s
    zero_torque_speed: float  # rad/s, where the line reaches zero
    torque_basis: str


def load_linear_decay_curve(
    path: Path = MOTOR_CONFIG, servo: str = "joint_servo_100"
) -> LinearDecayCurve:
    """Read a ``linear_decay`` curve -- by default the 100:1 servo's -- from the spec.

    **Defined, not wired.** Nothing builds an actuator from this; every joint
    still runs `load_servo_curve`'s curve. What this pins is the definition
    itself, so that wiring it later is transcription rather than a second fit.

    It refuses everything `load_servo_curve` refuses, and one thing more: a curve
    that does not say whether its torque is the servo's report or a shaft
    measurement. On the 100:1 servo those differ by a factor of about 1.6, and a
    reported curve applied as shaft torque would simulate a robot that much
    stronger than the hardware with nothing in the logs to say so.
    """
    spec, number = _torque_speed_spec(path, servo, _LINEAR_MODEL, "linear_decay_torque_limit")
    basis = spec.get("torque_basis")
    if basis not in _TORQUE_BASES:
        raise ValueError(
            f"{path}: motors.{servo}.torque_speed.torque_basis is {basis!r}; it must "
            f"be one of {_TORQUE_BASES}. Reported and shaft torque differ by ~1.6x "
            "on this servo, so the curve means nothing until it says which it is."
        )
    curve = LinearDecayCurve(
        plateau_torque=number("plateau_torque_nm"),
        corner_speed=number("corner_speed_rpm") * _RPM,
        zero_torque_speed=number("zero_torque_speed_rpm") * _RPM,
        torque_basis=basis,
    )
    if curve.plateau_torque <= 0.0:
        raise ValueError(f"{path}: plateau_torque_nm must be positive")
    if not 0.0 <= curve.corner_speed < curve.zero_torque_speed:
        raise ValueError(
            f"{path}: need 0 <= corner_speed_rpm < zero_torque_speed_rpm, got "
            f"{curve.corner_speed / _RPM} and {curve.zero_torque_speed / _RPM}"
        )
    return curve


def linear_decay_torque_limit(
    speed: torch.Tensor,
    plateau: torch.Tensor | float,
    corner: torch.Tensor | float,
    zero_torque_speed: torch.Tensor | float,
) -> torch.Tensor:
    """Maximum torque magnitude [N*m] at ``speed`` [rad/s] on a ``linear_decay`` curve.

    Continuous everywhere, unlike `servo_torque_limit`: the line itself reaches
    zero, so there is no cutoff to fall off. Symmetric in the sign of the speed,
    for the reason given there.
    """
    w = speed.abs()
    # (zero - w) / (zero - corner) is >= 1 on the plateau and <= 0 past the
    # zero-torque speed, so a single clamp produces all three pieces.
    return plateau * ((zero_torque_speed - w) / (zero_torque_speed - corner)).clamp(0.0, 1.0)


@dataclass(kw_only=True)
class ServoCurveActuatorCfg(IdealPdActuatorCfg):
    """A PD actuator whose torque is bounded by the measured servo curve.

    ``effort_limit`` (mjlab calls the runtime tensor ``force_limit``) stays a
    plain cap applied on top of the curve. Leave it at the plateau: setting it
    higher cannot buy torque the curve does not allow, and setting it lower is
    the diagnostic knob `velocity_env_cfg(effort_limit=...)` exposes.
    """

    plateau_torque: float = PLATEAU_TORQUE
    corner_speed: float = CORNER_SPEED
    decay_speed: float = DECAY_SPEED
    cutoff_speed: float = CUTOFF_SPEED
    continuous_torque: float = CONTINUOUS_TORQUE
    thermal_time_constant: float = THERMAL_TIME_CONSTANT

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.plateau_torque <= 0.0:
            raise ValueError("plateau_torque must be positive.")
        if self.decay_speed <= 0.0:
            raise ValueError("decay_speed must be positive; it divides an exponent.")
        if not 0.0 <= self.corner_speed < self.cutoff_speed:
            raise ValueError(
                "need 0 <= corner_speed < cutoff_speed, got "
                f"{self.corner_speed} and {self.cutoff_speed}."
            )
        if not 0.0 < self.continuous_torque <= self.plateau_torque:
            raise ValueError(
                "need 0 < continuous_torque <= plateau_torque, got "
                f"{self.continuous_torque} and {self.plateau_torque}."
            )
        if self.thermal_time_constant <= 0.0:
            raise ValueError("thermal_time_constant must be positive.")

    def build(
        self, entity, target_ids: list[int], target_names: list[str]
    ) -> ServoCurveActuator:
        return ServoCurveActuator(self, entity, target_ids, target_names)


class ServoCurveActuator(IdealPdActuator[ServoCurveActuatorCfg]):
    """PD torque clipped to the servo's measured torque-speed curve."""

    param_names = (
        *IdealPdActuator.param_names,  # stiffness, damping, force_limit
        "plateau_torque",
        "corner_speed",
        "decay_speed",
        "cutoff_speed",
    )

    @staticmethod
    def control_law(params: dict[str, torch.Tensor], cmd) -> torch.Tensor:
        torque = pd_torque(params["stiffness"], params["damping"], cmd)
        limit = servo_torque_limit(
            cmd.vel,
            params["plateau_torque"],
            params["corner_speed"],
            params["decay_speed"],
            params["cutoff_speed"],
        )
        # The flat cap on top of the curve. Equal to the plateau by default, so
        # this is a no-op until something randomises motor strength.
        limit = torch.minimum(limit, params["force_limit"])
        return torch.clamp(torque, -limit, limit)

    def compute(self, cmd) -> torch.Tensor:
        """PD torque under the curve, with the peak budget spent down.

        **This overrides `compute`, which leaves mjlab's fused path**, and that
        is a reversal worth writing down. The curve alone lives in `control_law`
        precisely so it could stay fused; the thermal budget cannot, because it
        is state. `fused_group.py` fuses actuators whose control output is a
        stateless function of parameters and command, and mjlab's own stateful
        actuator (`LearnedMlpActuator`) overrides `compute` for the same reason.

        The cost here is nil: this robot has **one** actuator covering all 22
        joints, and fusion batches *across* actuators, so a group of one saved
        nothing. Give the robot a second actuator group and this is worth
        revisiting.

        The thermal state enters by lowering `force_limit` for the substep, and
        nothing else changes -- `control_law` already returns
        `min(curve, force_limit)`, so a hot joint gets exactly
        `min(continuous, curve(speed))`. Above the corner speed the curve is
        already below the continuous rating and goes on binding, which is the
        intended reading of "the smaller of the two".
        """
        assert self._thermal is not None
        assert self.force_limit is not None
        assert self.continuous_torque is not None

        params = {name: getattr(self, name) for name in self.param_names}
        params["force_limit"] = torch.where(
            self._thermal >= 1.0,
            torch.minimum(self.force_limit, self.continuous_torque),
            self.force_limit,
        )
        torque = type(self).control_law(params, cmd)
        self._integrate_heat(torque)
        return torque

    def _integrate_heat(self, torque: torch.Tensor) -> None:
        """Advance the normalised winding temperature by one physics substep.

        `theta` is scaled so that 1.0 is the trip point: the steady state of the
        continuous rating. Heating follows the torque **actually produced**, not
        the torque demanded, because it is current that heats a winding and the
        clamp has already happened.

        At the trip point the equilibrium is exact -- holding the continuous
        rating gives `dtheta/dt = 0` -- so a joint parked there does not drift
        out of derating. Just below it the model does alternate: one substep at
        full curve pushes `theta` over 1, and it then takes roughly 250 ms at the
        continuous rating to come back under. That averages to about the
        continuous rating with brief peaks, which is the physically right answer,
        but it is bang-bang rather than the smooth derate a real controller would
        apply.
        """
        assert self._thermal is not None
        assert self.continuous_torque is not None
        assert self.thermal_time_constant is not None
        steady = (torque / self.continuous_torque) ** 2
        self._thermal += (steady - self._thermal) * (self._substep_dt / self.thermal_time_constant)
        self._thermal.clamp_(min=0.0)

    def __init__(self, cfg, entity, target_ids, target_names) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self.plateau_torque: torch.Tensor | None = None
        self.corner_speed: torch.Tensor | None = None
        self.decay_speed: torch.Tensor | None = None
        self.cutoff_speed: torch.Tensor | None = None
        self.continuous_torque: torch.Tensor | None = None
        self.thermal_time_constant: torch.Tensor | None = None
        #: Normalised winding temperature; 1.0 is the derating trip point.
        self._thermal: torch.Tensor | None = None
        self._substep_dt: float = 0.0

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        shape = (data.nworld, len(self._target_names))
        for name in (
            "plateau_torque",
            "corner_speed",
            "decay_speed",
            "cutoff_speed",
            "continuous_torque",
            "thermal_time_constant",
        ):
            setattr(
                self,
                name,
                torch.full(shape, getattr(self.cfg, name), dtype=torch.float, device=device),
            )
        self._thermal = torch.zeros(shape, dtype=torch.float, device=device)
        # `compute` runs once per physics substep, so this is the integrator's dt.
        self._substep_dt = float(mj_model.opt.timestep)

    def reset(self, env_ids=None) -> None:
        """Episodes start cold.

        Without this the budget would be spent once and never returned across a
        reset, and since resets are staggered the fleet would drift into a state
        where some environments model a hot servo and others a cold one for
        reasons unrelated to what their policies did.
        """
        super().reset(env_ids)
        if self._thermal is not None:
            self._thermal[slice(None) if env_ids is None else env_ids] = 0.0
