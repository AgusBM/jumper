"""A gait clock whose rate follows the command, so the stride can stay fixed.

`tripod/mdp/phase.py` runs one frequency for every command, and that frequency is
set by the *fastest* one. Every slower command then gets the same cadence and a
shorter stride -- the robot takes more steps per metre the slower it goes, which
is backwards.

Measured on a trained policy, holding each speed for 600 control steps, stride
read back as `v * duty / liftoff rate`:

    commanded   achieved   liftoffs/s/foot   stride    lifts per metre
      0.10        0.083         5.55         8.8 mm        66.9
      0.20        0.163         5.81        16.3 mm        35.6
      0.40        0.357         6.10        31.0 mm        17.1
      0.60        0.511         5.86        46.2 mm        11.5
      0.80        0.620         5.54        59.3 mm         8.9

Two things in that table. **The stride the cadence is derived from is real** -- at
the ceiling the policy uses 59.3 mm against the 0.06 m assumed. And **the policy
does not shuffle at low speed**: it holds 5.5-6.1 Hz all the way down and takes
8.8 mm steps to do it, so the waste is not something it has already worked
around. 7.5x the lifting per metre at 0.1 m/s against the ceiling, and most of
training happens at the slow end -- level 0 of the command ladder is +/-0.4 m/s,
whose mean |vx| is 0.2.

(That table was taken at 200 Hz control. The stride is a property of the legs and
the IK, not of the rate, which is why it carries over; the cadences in it do not
and are not used here.)

## The law

    v_eff = |v_xy| + |wz| * turn_radius
    f     = clamp(v_eff / (2 * stride), freq_min, freq_max)

**`wz` is in it and that is not a detail.** `moving_gate` -- which decides whether
the robot is walking at all -- uses `norm(cmd[:, :3])`, so a pure turn counts as
motion and the gait reward demands stepping. A law reading only `|v_xy|` would
hand that command `freq_min` while the robot has to swing its outermost foot at
`wz * 0.2` m/s to deliver it.

**It reads the command, never the measured velocity.** Two independent reasons,
either sufficient. The robot has no state estimator -- `export.py` lists
`base_lin_vel` under `_UNMEASURABLE_TERMS` for exactly this -- so a measured law
could not be reproduced on hardware at all. And a clock driven by achieved speed
is a loop the policy can ride: slow down, the cadence drops, the gait reward gets
easier.

## Why the phase has to be integrated, and what that costs

`phase.py` opens by calling its clock "a pure function of episode time -- **no new
buffer, no reset hook**", and treats that as a correctness property rather than
brevity: `episode_length_buf` is reset by `_reset_idx`, so the clock restarts with
the episode for free and cannot drift out of sync with anything.

A rate that follows the command ends that. The phase is now the integral of a
signal that changes whenever the command resamples, so it needs a buffer, a reset,
and -- the part that is easy to get wrong -- a guarantee that everything reading
it gets the *same* value in the same step. Six things read it: the actor's and
the critic's `gait_phase` observations, `tripod_gait` and `stance_load` through
`tripod/mdp/phase.py::gait_phase`, `stance_ground_gap` through
`common/mdp/phase.py::gait_phase`, and the swing gate on `foot_clearance`.

That guarantee is not free, because `ManagerBasedRlEnv.step` runs the managers in
the order

    episode_length_buf += 1  ->  rewards  ->  reset  ->  commands  ->  observations

and the command being integrated changes in the middle of it. Three rules make it
hold, and each is there because the first version lacked it:

- **One clock per environment, not per term.** mjlab builds a class-based
  observation term once per *group*, so a term that owns the phase is two phases
  as soon as the actor and the critic both observe it. `CadenceClock` is the
  phase; whichever group mjlab builds first hangs it on the env as `gait_clock`,
  and the other group's term, both `gait_phase` functions and the swing gate all
  find it there.
- **It advances lazily**, keyed on `common_step_counter`: whoever asks first in a
  step brings it up to date and everyone after gets the same number. A reset
  counts as asking -- it catches the clock up before zeroing -- so a new episode
  starts at 0 however early in the step it is reset.
- **It integrates the tempo of the command the policy was shown**, which the
  observation records every time it is computed, and not the command live at the
  moment somebody first asks. The two differ exactly in the step a command is
  redrawn, because the rewards run before the redraw and the observations after.

With all three the value does not depend on who asks first. In practice that is
a reward -- `foot_clearance`'s swing gate, the first of the four the reward
manager computes -- and what every reader gets in step `k` is

    phase_k = phase_(k-1) + f(command shown at step k-1) * step_dt,   0 at a reset

which is the relation the fixed clock has with `episode_length_buf`: one number
per step for the reward and the observation alike, and an action chosen at one
phase scored one step on. A deployment reproduces it by advancing *after* it
builds each frame, by that frame's command -- and is told to by the contract,
because policies trained before this clock existed were trained the other way;
`ADVANCE` has both.

What the first version got wrong, measured on native cpu with the command
redrawn every 20 steps: each group's term was its own clock and `env.gait_clock`
was the critic's, built last. The rewards advanced it before the redraw, the
actor's term advanced its own after it, and the two were identical until a redraw
changed the tempo and 4.94 degrees apart after it -- then one whole control step
apart after every reset, `f * step_dt`, 3.6 degrees at the floor and 10.0 at the
ceiling, because the actor's clock was advanced after the reset had zeroed it.
And `stance_ground_gap` read neither: `common/mdp/phase.py` did not look for an
installed clock, so it ran the fixed one at the ceiling and named the other
stance group on 49.8% of walking steps at the first command rung (64 envs x 2000
steps, i9-14900KF). `tests/test_posture.py` holds all six readers to one value
through a redraw and a staggered reset, in both call orders.

## What stops being required

`phase.py` requires a cycle to be a whole, even number of control steps, and
records a 12.2% left-right split from getting it wrong.

**Integration does not remove that requirement, and the first version of this
paragraph claimed it did.** The claim was that an integrated phase differs by at
most one step per cycle and so cannot accumulate. Half of that is true and the
half that is not is the half that matters: with 15 steps a cycle the phase takes
the values 0, 1/15, ... 14/15, of which **eight are below 0.5 and seven are not**,
every cycle, in the same direction. That is a standing 6.7% bias, and it arrives
however the phase is computed -- it is the parity of the step count, not the
arithmetic. A test caught it.

What integration does remove is the *other* failure, the one `phase.py`'s note is
actually about: truncation residue accumulating coherently across cycles.

So the requirement is answered rather than removed, and it is answered where the
cadence can be *held*: at the two ends of the clamp. **Both are even.** The
ceiling is 18 control steps -- `GAIT_STRIDE` is 72 mm for that reason and not
because 72 was wanted -- and the floor is 50. In between the cadence is whatever
the command asks for, the step count is not an integer, the phase drifts through
and the split averages out: measured over a command sequence drawn uniformly on
+/-0.8 and resampled every 3-8 s, group A is selected 49.99% of the time.

The cost of an odd ceiling, for the record, since 60 mm was the natural stride to
want and gives one: 15 steps splits 8/7, and a command held at full stick sits on
that 53.3% for as long as it is held. `tests/test_posture.py` asserts the ceiling
is even so that a future edit to either the stride or the control rate cannot
reintroduce it quietly.

Swing resolution is the other thing the choice buys: 9 control steps at the
ceiling, against 8 at a 64 mm stride and 7.5 at 60. That is the number to watch if
feet start clipping the ground -- `Metrics/peak_height_mean` -- and it is a reason
to raise the control rate rather than to lower the cadence, because lowering the
cadence at a fixed stride lowers the top speed with it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

#: Where the clock hangs on the environment. `tripod/mdp/phase.py::gait_phase`
#: and `common/mdp/phase.py::gait_phase` both look for this name and defer to it
#: when it is there, which is how `tripod_gait`, `stance_load` and
#: `stance_ground_gap` read the clock the observations show without any of them
#: knowing which kind it is. `rewards.py::_swing_gate` reads it directly.
ENV_ATTR = "gait_clock"

#: When a host advances the phase, as the contract tells it: the `gait_phase`
#: term carries it as `params.advance`, which `scripts/export.py` writes into
#: `layout.json` with the rest of the law.
#:
#: `"after_frame"` is this clock. What a policy is shown at step `k` is
#: `phase_(k-1) + f(command shown at k-1) * step_dt`, so a host builds each frame
#: with the phase it has and advances after it, by that frame's own command; the
#: first frame after entering the mode is 0.
#:
#: **Absent is the order before it**, and a host must keep it that way: advanced
#: *before* each frame but the first, by that frame's command -- the order the
#: actor's own clock had while each observation group mounted one, and so the
#: order every policy exported without this field was trained in. That is why
#: this is a field and not a new default. Both kinds of policy exist, they differ
#: by `(f_now - f_at_entry) * step_dt` -- up to 6.4 degrees here -- once the tempo
#: moves, and the contract is the only place a host can tell them apart.
#:
#: What it cannot catch: an export records the code it was exported with, not the
#: code the checkpoint was trained with, so exporting a checkpoint from before
#: this clock with today's code writes the new order for a policy trained in the
#: old one. Export an old checkpoint from the commit it was trained at.
ADVANCE = "after_frame"


def gait_frequency(
    env: "ManagerBasedRlEnv",
    command_name: str,
    stride: float,
    freq_min: float,
    freq_max: float,
    turn_radius: float,
) -> torch.Tensor:
    """This step's cadence per environment, in Hz. See the module docstring."""
    cmd = env.command_manager.get_command(command_name)
    assert cmd is not None, f"no command term named {command_name!r}"
    v_eff = cmd[:, :2].norm(dim=1) + cmd[:, 2].abs() * turn_radius
    return (v_eff / (2.0 * stride)).clamp(freq_min, freq_max)


class CadenceClock:
    """The integrated phase: one per environment, however many things read it.

    Not an observation term, though one creates it. The first `VariableGaitClock`
    mjlab builds hangs it on the env under `ENV_ATTR`; every other reader -- the
    other group's term, both `gait_phase` functions, the swing gate -- finds it
    there. The module docstring has the three rules it keeps and why each exists.
    """

    def __init__(
        self,
        env: "ManagerBasedRlEnv",
        command_name: str,
        stride: float,
        freq_min: float,
        freq_max: float,
        turn_radius: float,
    ) -> None:
        #: The law's arguments to `gait_frequency`, compared when a second term
        #: finds this clock: one clock cannot run at two laws.
        self.law = (command_name, float(stride), float(freq_min), float(freq_max),
                    float(turn_radius))
        self._env = env

        self._phase = torch.zeros(env.num_envs, device=env.device)
        # The step the phase was last brought up to. Starts at the counter's own
        # value rather than at -1, so the observation computed during the
        # construction-time reset reads phase 0 -- which is what the fixed clock
        # gives at `episode_length_buf = 0`.
        self._step = int(env.common_step_counter)
        # The tempo the next step is integrated at: that of the command the
        # policy was last shown. `show()` keeps it current; this is only the
        # value until the first observation is computed.
        self._rate = self.frequency()

    def frequency(self) -> torch.Tensor:
        """The live command's tempo, in Hz -- not the one being integrated."""
        return gait_frequency(self._env, *self.law)

    def phase(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
        """The gait phase in [0, 1), advanced at most once per environment step.

        Whoever asks first in a step brings it up to date, at the tempo the
        policy was shown, and everyone after gets the same number. Because the
        tempo is the recorded one rather than the live command's, the number
        does not depend on who that first caller is.
        """
        step = int(env.common_step_counter)
        if step != self._step:
            # One advance per step even if several were somehow missed. A skipped
            # step is a bug elsewhere and integrating it here would hide it.
            self._phase = torch.remainder(self._phase + self._rate * env.step_dt, 1.0)
            self._step = step
        return self._phase

    def show(self) -> None:
        """Record the tempo of the command the policy is being shown.

        Called by the observation, which mjlab computes after the step's
        command update and after every reset -- so this is always the command
        the next action will be chosen under, and the next step is integrated
        at it.
        """
        self._rate = self.frequency()

    def reset(self, env_ids=None) -> None:
        """Restart the phase with the episode, as `episode_length_buf` did.

        **A reset is a caller too.** It brings the clock up to date before
        zeroing, so the environments it does not reset have been advanced at the
        right tempo and the ones it does stay at 0 for the rest of the step --
        whether or not anything asked before it. Without that, a step in which
        nothing read the clock before the reset would advance the new episode
        by one step when the observation asked.
        """
        self.phase(self._env)
        if env_ids is None or isinstance(env_ids, slice):
            self._phase.zero_()
        else:
            self._phase[env_ids] = 0.0


class VariableGaitClock:
    """The observation term: `(sin, cos)` of the env's `CadenceClock`.

    Returns `(sin, cos)` of the phase, or `(0, 0)` while the command asks the
    robot to stand -- the same contract `phase.py::phase_clock` has, including the
    measurement behind the off-value: a free-running clock in the observation
    measurably made the robot step in place while parked.

    **The term is a view; the phase is not in it.** mjlab builds a class term once
    per observation group, so the actor's and the critic's are two objects, and
    the version that kept the phase here was two clocks. So whichever is built
    first creates the `CadenceClock` and registers it on the env, and every later
    one uses that. Each resets it, which is idempotent, and each records the
    command it is shown, which is the same command.
    """

    def __init__(self, cfg, env: "ManagerBasedRlEnv") -> None:
        params = cfg.params
        law = (
            params["command_name"],
            float(params["stride"]),
            float(params["freq_min"]),
            float(params["freq_max"]),
            float(params["turn_radius"]),
        )
        self._command = params["command_name"]
        self._threshold = float(params.get("command_threshold", 0.05))

        # Required rather than defaulted: a term without it exports a contract
        # without it, and a host reading none runs the old order -- off this
        # clock by `(f_now - f_at_entry) * step_dt` once the tempo moves, on a
        # robot, with nothing to say so. See `ADVANCE`.
        if params.get("advance") != ADVANCE:
            raise ValueError(
                f"gait_phase says advance={params.get('advance')!r}; this clock "
                f"advances {ADVANCE!r}, and the contract has to say so for a host "
                f"to reproduce it"
            )

        clock = getattr(env, ENV_ATTR, None)
        if clock is None:
            clock = CadenceClock(env, *law)
            setattr(env, ENV_ATTR, clock)
        elif getattr(clock, "law", None) != law:
            raise ValueError(
                f"two gait_phase terms describe different cadence laws, "
                f"{getattr(clock, 'law', None)} and {law}. There is one clock per "
                f"environment, and a second law would be a second phase."
            )
        #: The env's clock, which this term shows. Public so that a test can
        #: check it is the same object in every group, which the numbers alone
        #: cannot: two clocks fed the same commands and resets agree exactly,
        #: right up to the first thing that reaches one and not the other.
        self.clock = clock

    def __call__(self, env: "ManagerBasedRlEnv", **params) -> torch.Tensor:
        del params  # consumed in __init__; the manager passes them back
        angle = self.clock.phase(env) * (2.0 * math.pi)
        self.clock.show()
        sin_cos = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

        from ...common.mdp.rewards import moving_gate

        moving = moving_gate(env, self._command, self._threshold)
        return sin_cos * moving.unsqueeze(-1)

    def reset(self, env_ids=None) -> None:
        self.clock.reset(env_ids)


__all__ = ["ENV_ATTR", "CadenceClock", "VariableGaitClock", "gait_frequency"]
