"""Shared pieces for the gait rewards -- the gating and phase matching the three
gait tasks have in common.

**Why they are needed**: mjlab's velocity task rewards
(`mjlab/tasks/velocity/mdp/rewards.py`) were written for the Unitree Go1
(quadruped) and G1 (humanoid) and contain nothing aimed at a hexapod's
coordination patterns. And coordination is exactly where a hexapod is hard:

A hexapod is statically stable. With all six feet planted the body is heavily
over-constrained, every small disturbance is absorbed by redundant contacts and
the body does not move -- so velocity rewards have **almost no gradient** near that
state, and the policy readily locks into the "stand still" local optimum.

None of the three pieces here depends on body velocity, so they provide gradient
before the body has moved at all and push the policy out of that basin. That is
what fundamentally distinguishes them from mjlab's existing reward terms.

The gait reward functions themselves are **not here** -- each gait is used by
exactly one task and lives in `tasks/jumper/<task>/mdp/rewards.py`:

| task | function | duty_factor | phase grouping | airborne | support |
|---|---|---|---|---|---|
| `jumper.tripod`   | `tripod_gait`   | 0.5 | {LF,RM,LR} / {RF,LM,RR} in antiphase | 3 | 3 |
| `jumper.tetrapod` | `tetrapod_gait` | 2/3 | {LF,RR} / {LM,RM} / {LR,RF} in turn | 2 | 4 |
| `jumper.ripple`   | `ripple_gait`   | 5/6 | six legs offset by 1/6 cycle each | 1 | 5 |

`jumper.tripod` additionally drives its grouping from a **fixed 3.125 Hz phase clock**
(`tasks/jumper/tripod/mdp/phase.py`), so for that task alone the criterion is no
longer purely instantaneous and the "which group lifts first" freedom described
below is gone. Nothing in this module changes as a result; `phase_match` is called
with one group instead of being maximised over both.

The three are points on one spectrum: more airborne legs means faster and a
smaller static stability margin. They are **mutually exclusive** (tripod's ideal
pattern scores 0 under ripple), which is why they are three separate tasks rather
than three switches on one. `jumper.flat` uses none of them and is their control.

Convention: every return value is "larger is better" in [0, 1]; the sign comes
from the weight in the config.

## Why all three need cycle gating (max_air_time + max_contact_time)

These criteria are fundamentally **instantaneous** -- they only ask whether the
contact state right now looks like that gait. An instantaneous criterion can be
satisfied by a static pose, which is its inherent flaw and has to be covered by
gating.

Two traps, two faces of the same hole:

**One: hold them up.** Simply lifting a valid group of legs and standing still
scores full marks forever: measured, holding LF+RR up for 8 seconds and stepping
normally for 0.1 seconds both score 1.000. The answer is `max_air_time` -- if any
airborne leg's `current_air_time` exceeds the limit, the whole term goes to zero.
This follows mjlab's own `feet_air_time` (whose threshold_max is also 0.5 s).

**Two: march in place.** Gating the airborne end alone is not enough. The policy
can lift and drop half the legs briefly (small air_time, easily within the limit)
while the other half stay planted as props and the body never moves. Measured on
the tetrapod gait at weight=2.0 @2000: gait reward 1.59/2.0, mean feet in contact
3.89 (a near-perfect gait shape), and horizontal displacement over 12 seconds of
just **0.065 m** against the control group's 1.69 m -- the gait objective achieved
and the locomotion objective entirely sacrificed. The answer is
`max_contact_time`: any leg whose `current_contact_time` exceeds the limit zeroes
the term as well.

Note that the `_moving` gate stops neither of these: it tests whether the
**command** asks for motion, not whether the robot is **actually** moving. A
non-zero command with the robot marching in place passes it.

`max_contact_time` is set by duty factor, since the support phase is naturally
longer than the swing phase: tripod(1:1)=1.0 s, tetrapod(2:1)=1.0 s,
ripple(5:1)=1.5 s. Set too tight, it strangles a normal gait.

## What this module provides

| Name | Purpose |
|---|---|
| `contact_state` | normalise the contact sensor's `found` into a [num_envs, 6] 0/1 |
| `phase_match`   | how well the current contact state matches a given "which legs should be airborne" phase |
| `cycling_gate`  | are the legs still cycling (gating both ends, closing the two holes above) |
| `moving_gate`   | does the command ask for motion |
| `leg_side`      | which side a leg is on |
| `track_yaw_velocity` | yaw tracking without the torso rates mjlab folds in |
| `body_tilt_rate` | the torso rates, as their own penalty, above a deadband |
| `stance_foot_load` | are the legs the clock designates as support carrying load |

All three gait functions are written as
`phase_match(...) * moving_gate(...) * cycling_gate(...)`.

`track_yaw_velocity` and `body_tilt_rate` are the odd ones out: neither is about
coordination, and they are here rather than in a task directory because all four
jumper tasks use them. They are two halves of one change -- mjlab's
`track_angular_velocity` folded the torso's roll and pitch rate into the *yaw
tracking* exponent, where they dominated it and made the term identically zero on
this robot. Splitting them gives the tracking signal back and gives the torso
rates a term of their own, with a deadband so that it does not become a tax on
walking. Both derivations are in their docstrings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

def track_yaw_velocity(
    env: "ManagerBasedRlEnv",
    std: float,
    command_name: str,
    asset_cfg=None,
) -> torch.Tensor:
    """`exp(-yaw_err^2 / std^2)` -- the commanded axis, and nothing else.

    A drop-in replacement for mjlab's `track_angular_velocity`, which scores

        exp(-(yaw_err^2 + roll_rate^2 + pitch_rate^2) / std^2)

    -- i.e. it folds the body's roll and pitch **rates** into a term named after
    velocity *tracking*, on the assumption that a walking robot's torso is nearly
    still. That holds for the quadruped and the humanoid it was written for. On
    this hexapod it does not, and the consequence is not a bias but a dead term.

    Measured on the trained tripod policy at 3.125 Hz, command range +/-0.15,
    over a full 1000-step episode with the stochastic policy training actually
    collects, mean of each squared component:

        yaw_err^2   0.237
        roll^2      0.159
        pitch^2     0.359   <- the largest single contributor
        sum         0.755      against std^2 = 0.0036  ->  exp(-210) = 0

    **The torso terms are 2.2x the one the reward is named after.** The logged
    value bears it out: `Episode_Reward/track_angular_velocity` sat at 0.0017
    against `track_linear_velocity`'s 1.11 for 37500 iterations -- a term carrying
    weight 2.0, half the intended positive reward budget, paying nothing.

    And it is dead at **every** curriculum level, so it is not something a wider
    range fixes; the same exponent against each level's std gives

        level 0  std 0.060   0.000000
        level 1  std 0.120   0.000000
        level 2  std 0.200   0.000000
        level 3  std 0.300   0.000226

    What that cost: the policy was never paid for turning, so it never learned to.
    Yaw error settled at 0.39 rad/s against a command range of +/-0.15 -- **worse
    than standing still**, which scores 0.71 of the range. `TerrainLevels` gates on
    that error, so terrain stayed pinned at level 0 with `demoted` at 1.0 for the
    entire run.

    ## The incentive gap, which is the number that actually matters

    A reward being small is survivable; a reward being *flat* is not. Scoring two
    counterfactuals on 500 steps x 256 envs of recorded commands and torso rates --
    yaw rate replaced by 0 ("never turns") and by the command ("tracks well"), with
    the measured roll and pitch rates left in place both times, because they are
    there whatever the robot does about yaw:

        level  std    old: never / well / gap      new: never / well / gap
          0    0.06   0.0041  0.0098  0.0057       0.343   1.000   0.657
          1    0.12   0.0157  0.0382  0.0225       0.343   1.000   0.657
          2    0.20   0.0396  0.0978  0.0582       0.343   1.000   0.657
          3    0.30   0.0762  0.1918  0.1156       0.343   1.000   0.657

    At level 0 the old term offered **0.006** for learning to turn perfectly, and
    at weight 2.0 that is 0.011 of reward against tracking terms worth 4.0. Turning
    costs `action_rate_l2` and `foot_slip` considerably more than that, so not
    turning was simply the better deal. The new term offers 0.657, a gap 115x
    larger.

    The second column is the part worth keeping: **the new gap is identical at
    every level.** `STD_ANG_RATIO` scales std with the range precisely so that
    widening the range does not also loosen the ruler -- the invariance
    `mdp/curriculum.py` is built on. With the torso rates in the exponent that
    invariance was destroyed (the gap moved by 20x across the four levels, because
    a range-independent constant was being divided by a range-dependent std).
    Taking them out restores it exactly.

    Roll and pitch rate are deliberately **not** replaced by a penalty here.
    `body_ang_vel` and `angular_momentum` exist for that and are held at weight 0
    on purpose -- see `velocity_env.py`, which records why: they penalise torso
    motion always, so they raise the cost of walking and make standing still more
    attractive, which is the local optimum this task keeps falling into. Body
    attitude is already covered by `upright` (angle, not rate) and, while the
    command says stand, by `standing_sway`.

    `asset_cfg` is accepted and ignored so the term config mjlab ships can be
    reused unchanged; there is one robot.
    """
    del asset_cfg
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"command {command_name!r} not found"
    actual = env.scene["robot"].data.root_link_ang_vel_b
    return torch.exp(-torch.square(command[:, 2] - actual[:, 2]) / std**2)


def leg_side(idx: torch.Tensor) -> torch.Tensor:
    return idx % 2


def contact_state(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    """Normalise the contact sensor's `found` into a [num_envs, 6] 0/1 tensor.

    Note that `found` is a **count of contact points**, not a boolean (measured up
    to 4): one foot can produce several contact points at once.
    """
    found = env.scene.sensors[sensor_name].data.found
    assert found is not None, f"sensor {sensor_name!r} has no found field"
    return (found.reshape(found.shape[0], -1) > 0).float()


def cycling_gate(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    max_air_time: float,
    max_contact_time: float,
) -> torch.Tensor:
    """Are the legs still cycling; returns a [num_envs] 0/1 gate.

    **Both ends have to be gated; neither alone is enough**:

    - **Airborne legs**: every `current_air_time` must be < `max_air_time`. One
      held too long means the policy is farming reward by holding a leg up while
      standing still.
    - **Legs in contact**: every `current_contact_time` must be <
      `max_contact_time`. One planted too long means the policy is farming reward
      by marching in place.

    Gating air_time alone measurably missed the second: half the legs lift and drop
    briefly to satisfy the airborne gate while the other half stay planted as props
    and the body does not move. From the audit of the tetrapod gait at weight=2.0
    @2000 -- gait reward 1.59/2.0, mean feet in contact 3.89 (a perfect gait shape),
    horizontal displacement over 12 seconds just 0.065 m against the control group's
    1.69 m. Gait objective achieved, locomotion objective entirely sacrificed.

    `max_contact_time` has to follow each gait's duty factor, since the support
    phase is naturally longer than the swing phase: tripod(1:1) < tetrapod(2:1) <
    ripple(5:1). Set too tight, it strangles a normal gait.
    """
    sensor = env.scene.sensors[sensor_name]
    contact = contact_state(env, sensor_name)

    t_air = sensor.data.current_air_time
    assert t_air is not None, (
        f"sensor {sensor_name!r} does not have track_air_time enabled, so whether "
        f"the legs are cycling cannot be determined"
    )
    # Airborne legs only: air_time is meaningless for a leg in contact, so it is
    # zeroed and cannot trip the gate
    airborne = t_air.reshape(contact.shape) * (1.0 - contact)

    t_con = sensor.data.current_contact_time
    assert t_con is not None, (
        f"sensor {sensor_name!r} does not have track_air_time enabled, so support "
        f"duration cannot be determined"
    )
    # Likewise, legs in contact only
    stance = t_con.reshape(contact.shape) * contact

    return (
        (airborne.amax(dim=1) < max_air_time)
        & (stance.amax(dim=1) < max_contact_time)
    ).float()


def moving_gate(
    env: "ManagerBasedRlEnv", command_name: str, threshold: float
) -> torch.Tensor:
    """Does the command ask for motion; returns a [num_envs] 0/1 gate."""
    cmd = env.command_manager.get_command(command_name)
    assert cmd is not None
    return (torch.norm(cmd[:, :3], dim=1) > threshold).float()


def standing_sway(
    env: "ManagerBasedRlEnv",
    command_name: str,
    command_threshold: float = 0.05,
    asset_name: str = "robot",
) -> torch.Tensor:
    """How much the body moves **while the command asks it to stand still**.

        sway = (|v|^2 + |w|^2) * [the command is below threshold]

    Returns a [num_envs] value that is larger the worse it is; the weight in the
    config carries the minus sign. Zero for every environment that was asked to
    move, so this term is invisible during walking.

    **Why this is not `body_ang_vel` by another name.** That term (and
    `angular_momentum`) are deliberately held at weight 0, with the reason recorded
    in `velocity_env.py`: they penalise torso motion *always*, so they raise the
    cost of walking and make standing still more attractive -- the wrong direction
    for a robot whose documented failure is refusing to move. Gating on a
    near-zero command inverts that: the term costs nothing to a walking policy, so
    it cannot tilt the walk/stand trade-off at all. It only separates standing
    well from standing badly.

    **What it fixes.** The tracking rewards are `exp(-err^2/std^2)`, so an
    environment commanded to stand collects `exp(0) = 1` on both of them no matter
    what the body is doing -- the note on the angular std in `velocity_env.py`
    calls this out as "a floor that cannot be lowered", since roughly 28% of
    commands are near zero. Within that bucket a robot rocking on its legs and one
    holding perfectly still are **indistinguishable**, on every term the task has.
    This is the term that tells them apart.

    Both the linear and angular parts are included, which means the horizontal
    components overlap with what the tracking rewards already measure. That
    overlap is the point: near a zero command those rewards are flat (their
    gradient vanishes as the error goes to zero), while a squared penalty is
    steepest exactly where they are flattest. The components nothing else
    constrains at all -- roll and pitch rate, and vertical bobbing -- come along
    for free in the same expression.

    The units are mixed, m/s against rad/s, so the weight has no clean physical
    reading; it is calibrated against measured sway instead (see the weight's note
    in `velocity_env.py`).
    """
    asset = env.scene[asset_name]
    standing = 1.0 - moving_gate(env, command_name, command_threshold)
    lin = asset.data.root_link_lin_vel_b
    ang = asset.data.root_link_ang_vel_b
    return (lin.square().sum(dim=1) + ang.square().sum(dim=1)) * standing


def phase_match(
    contact: torch.Tensor, swing: tuple[int, ...], sigma: float
) -> torch.Tensor:
    """How well the current contact state matches the phase "the `swing` group is
    airborne and everything else is in support", in (0, 1].

        n_wrong = number of legs in the wrong state
        score   = exp(-n_wrong^2 / sigma^2)

    **Why a product of means does not work** (which is how all three gaits were
    originally written): a mean dilutes away *whether the grouping is right*,
    leaving only roughly how many legs are in what state. Take the worst case --
    the policy lifts LF+LM (same side, the front-left with no support at all):

        old formula  pair(LF,RR): airborne mean (1+0)/2 = 0.5
                                  other four in contact 3/4 = 0.75  -> **0.375**
        this formula pair(LF,RR): RR should be up and is not, LM should be down and
                                  is not -> n_wrong = 2
                                  -> exp(-4/0.49) = **0.0003**

    An unambiguously wrong grouping scored 37.5% under the old formula. The
    measured consequence: the tetrapod gait trained to a mean of 2.54 feet in
    contact (the target is 4.0) while the gait term still collected 0.43/1.0 -- the
    policy could take nearly half the gait reward for having roughly some legs up,
    with no pressure at all to get the grouping right.

    Raising the weight does not fix this; it amplifies the wrong grouping's score
    too: at weight 2.0 the gait score measured 1.52/2.0 while velocity tracking
    collapsed to 0.40 (1.66 at weight 1.0).

    sigma defaults to 0.7: one leg wrong still scores exp(-1/0.49) ~= 0.13, keeping
    the gradient from "one wrong" to "exactly right", while two or more go
    effectively to zero (0.0003). And **any wrong grouping is at least two wrong**
    (one that should be up is not, one that should not be is). That is how the
    grouping becomes binding.
    """
    desired = torch.ones_like(contact)
    desired[:, list(swing)] = 0.0
    n_wrong = (contact - desired).abs().sum(dim=1)
    return torch.exp(-(n_wrong**2) / sigma**2)


def body_tilt_rate(
    env: "ManagerBasedRlEnv",
    deadband: float = 0.5,
) -> torch.Tensor:
    """Penalise **excess** roll and pitch rate, in (rad/s)^2 above a deadband.

        excess = max(0, wx^2 + wy^2 - deadband)

    Larger is worse; the sign comes from a negative weight in the config. Yaw is
    not here -- it is the commanded axis and `track_yaw_velocity` scores it.

    ## Why this exists as its own term

    mjlab's `track_angular_velocity` folded `wx^2 + wy^2` into the *tracking*
    reward's exponent. On this hexapod those two measured 0.533 against the yaw
    error's 0.237, so they dominated it and the term returned ~0 whatever the
    robot did about yaw -- a flat function, logged at 0.0017 for 37500 iterations.
    Replacing it with `track_yaw_velocity` fixed the tracking signal and left the
    torso rates scored by nothing at all.

    ## Why a deadband rather than a plain penalty

    **A plain penalty on torso motion makes standing still more attractive**, and
    that is this task's documented local optimum -- `velocity_env.py` holds
    `body_ang_vel` and `angular_momentum` at weight 0 for exactly that reason.
    Standing produces no sway, so it pays nothing while walking pays continuously;
    the term would be a tax on locomotion wearing the name of a stability term.

    Gating it on `moving_gate` does not help -- it makes it worse, since that is
    precisely "charge only while walking".

    The deadband is what separates the two. A tripod gait at 3.125 Hz *has* to bob:
    measured over a full episode on the trained policy, `wx^2 + wy^2` came to

        0.182   deterministic policy
        0.533   stochastic, i.e. what training actually collects

    **Those are time means, and this term charges on the instantaneous value.**
    The bob is periodic, so `wx^2 + wy^2` swings above and below its mean -- for a
    sinusoid the peak is about twice it -- and a deadband set at the mean is
    exceeded roughly half the time. Checked on the arithmetic: at the measured RMS
    rates (0.63, 0.60 rad/s) the instantaneous value is 0.757 and the term returns
    0.257, not zero. So a correct gait pays a small but real amount, and at weight
    -0.05 that is about -0.013 per step against tracking's ~1.5.

    That is the honest statement, and an earlier version of this note claimed the
    stronger one ("a correct gait pays approximately nothing"), which the same
    arithmetic falsifies. **If the intent is genuinely zero for a correct gait, the
    deadband belongs at the peak rather than the mean** -- about 1.0 rather than
    0.5. 0.5 is kept because it still charges nothing for standing still, so it
    does not tilt the standing-versus-walking trade the wrong way; what it does not
    do is leave a correct gait entirely alone.

    **The number is tripod's and is due a re-measurement for the other two.**
    Tetrapod and ripple run at 25/6 Hz with four and five legs in support, so both
    bob less and their deadbands are probably lower -- a deadband set too high is
    silent, it simply makes the term inert. Each gait task can pass its own.

    Attitude itself is covered elsewhere and this does not duplicate it: `upright`
    scores the tilt *angle* (measured at 0.9868, i.e. about 3 degrees) and
    `standing_sway` covers the standing case. What was missing was the *rate*
    while walking.
    """
    w = env.scene["robot"].data.root_link_ang_vel_b
    return (w[:, 0].square() + w[:, 1].square() - deadband).clamp(min=0.0)


def stance_foot_load(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    command_name: str,
    groups: tuple[tuple[int, ...], ...],
    freq_hz: float,
    force_threshold: float,
    command_threshold: float = 0.05,
    max_air_time: float = 0.5,
    max_contact_time: float = 1.0,
) -> torch.Tensor:
    """Reward the legs the clock puts in **support** for actually carrying load.

        reward = sum over stance legs of min(force / `force_threshold`, 1)
                 / (number of stance legs)

    in [0, 1], per step. Being a per-step term, the episode return is proportional
    to **how long** the load was held -- integrating it is what turns "above the
    threshold" into "above it for longer pays more", with no duration state to
    keep. Note the consequence: 10 scattered steps of load and 10 consecutive ones
    score the same. If continuity itself turns out to matter (a foot chattering on
    and off the ground at threshold is not the same as one planted), this term has
    to grow within a stance interval, and that does need per-leg state.

    **Why load and not contact.** The gait term already scores which legs are
    touching, and touching is cheap: a foot can rest on the ground carrying almost
    nothing while the body's weight goes through the other legs, or through a leg
    that should be swinging. The two together say "these legs, and really on them".

    **`force_threshold` is per gait and is not optional to think about.** The robot
    weighs about 2 kg (29.5 N), so a fair share is 29.5 / (number of supporting
    legs) and the threshold is about half of it -- comfortably above a foot merely
    grazing the ground, well below what a genuinely loaded leg carries, so it does
    not demand a perfectly even split:

        gait       stance legs   fair share N   threshold N
        tripod          3            9.8            5.0
        tetrapod        4            7.4            3.7
        ripple          5            5.9            3.0

    Carrying tripod's 5.0 over to ripple would ask each of five legs for 85% of a
    fair share, which no even split can provide -- the term would read as failure
    for a correct gait. Each task passes its own; there is deliberately no default.

    **The two gates are not optional either.** Standing on all six feet loads the
    designated stance legs just as well as walking does, so ungated this term would
    pay full marks for standing still -- the local optimum this robot falls into by
    default, and one that reward terms *add* into: the policy could bank this while
    eating a zero from the gait term. `cycling_gate` closes it exactly as it is
    closed there: standing puts `current_contact_time` past `max_contact_time` and
    the term goes to zero.

    Force is the net contact force's magnitude. On flat ground the vertical
    component dominates; if lateral load ever needs excluding, this is the place to
    switch to `force[..., 2]`.
    """
    from .phase import gait_phase, stance_mask

    sensor = env.scene.sensors[sensor_name]
    force = sensor.data.force
    assert force is not None, f"sensor {sensor_name!r} has no force field"
    # **A ramp, not a step.** `loaded` used to be `(force > threshold)`, and that
    # made the term flat across everything short of the threshold: a foot in the
    # air, a foot 1 mm off the ground and a foot pressing 4.9 N all scored zero.
    #
    # Measured on the 25927-iteration tripod run of 2026-09-06 (87 MB of
    # tensorboard under locomotion-training-master), averaged over the last 30%:
    #
    #     stance_load raw 0.673  ->  2.02 of 3 stance feet above 5 N
    #     tripod_gait raw 0.591  ->  exp(-n^2/0.49) gives n_wrong = 0.51 legs in
    #                                the wrong contact state
    #     so of the ~0.98 under-loaded stance legs, ~0.51 are not touching at all
    #     and **~0.47 are touching and carrying too little**
    #
    # That second half is exactly what a step function cannot help: the foot is
    # already down, and going from 2 N to 4 N earns nothing. The two curves are
    # also locked together -- stance_load / tripod_gait held 0.568 to 0.575 with a
    # standard deviation of 0.008 over the whole run, and stance_load did not
    # improve at all between iteration 2000 and 25000. It was not being optimised
    # on its own; it was riding on the gait term.
    #
    # The ramp saturates at the threshold rather than continuing to pay, so a
    # genuinely loaded foot still scores exactly 1 and nothing is gained by
    # stamping harder. What changes is only that the 0 -> threshold stretch is a
    # slope instead of a cliff.
    loaded = (torch.linalg.norm(force, dim=-1) / force_threshold).clamp(0.0, 1.0)

    stance = stance_mask(gait_phase(env, freq_hz), groups, n_legs=loaded.shape[1])
    # Every group leaves the same number of legs in support, so this is a constant
    # -- taken from the first group rather than written out again, so a task that
    # changes its grouping cannot leave a stale divisor behind.
    n_stance = loaded.shape[1] - len(groups[0])

    # Both gates stay. **Standing is not this term's job** -- it is
    # `standing_foot_load`'s, which has its own threshold because a fair share
    # over six legs is not a fair share over three. Folding the two together
    # needed the threshold to scale with the support count and the cycling gate
    # to apply in one regime and not the other; two terms say the same thing and
    # each stays readable.
    return (
        (loaded * stance).sum(dim=1)
        / n_stance
        * moving_gate(env, command_name, command_threshold)
        * cycling_gate(env, sensor_name, max_air_time, max_contact_time)
    )




def standing_foot_load(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    command_name: str,
    force_threshold: float = 3.0,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward all six feet for carrying load **while the command asks the robot
    to stand**.

        reward = mean over the six feet of min(force / `force_threshold`, 1)
                 * [the command is below `command_threshold`]

    `stance_foot_load` is the walking half of this and deliberately switches off
    here: it is gated on `moving_gate`, and on `cycling_gate` besides, because
    standing loads the clock's designated legs perfectly well and an ungated
    positive term would pay for standing still while being told to move.

    **The two are separate terms rather than one with a branch**, and the reason
    is the threshold. `stance_foot_load` uses 5 N, which is half a fair share
    when three legs carry the robot: ~27 N over three is 9 N each. Standing
    spreads the same weight over six, about 4.5 N each, so 5 N is *above* a fair
    share and a robot standing perfectly evenly would score near zero on it.
    A single term would have had to scale its threshold by the support count and
    apply the cycling gate in one regime but not the other; two terms say the
    same thing and each stays legible.

    **3 N is two thirds of that 4.5 N fair share** -- above a foot merely resting
    on the ground, comfortably reachable by all six at once, and it leaves room
    for the uneven split that any real stance has.

    A ramp rather than a step, for the same reason `stance_foot_load` uses one: a
    foot at 2.9 N and a foot in the air are not the same thing, and a step
    function scores them identically. It saturates at the threshold, so pressing
    harder than a fair share earns nothing.
    """
    sensor = env.scene.sensors[sensor_name]
    force = sensor.data.force
    assert force is not None, f"sensor {sensor_name!r} has no force field"

    loaded = (torch.linalg.norm(force, dim=-1) / force_threshold).clamp(0.0, 1.0)
    standing = 1.0 - moving_gate(env, command_name, command_threshold)
    return loaded.mean(dim=1) * standing


#: Where the cached actuator->joint mapping is kept: **on the config object
#: itself**, not in a dict keyed by `id()`.
#:
#: The first version was `dict[int, Tensor]` keyed by `id(asset_cfg)`, and that
#: is wrong in a way that only shows up under load. CPython reuses the address of
#: a collected object -- measured here, five `SceneEntityCfg`s created and
#: dropped in a loop all reported the **same** `id` -- so a fresh config can
#: inherit a dead one's entry and pair every torque with another entity's joints.
#: It produced a test that passed alone and failed in the suite, which is the
#: signature of exactly this and is the only reason it was found.
#:
#: An attribute lives and dies with the object it describes, which is the
#: property the cache needed all along. `state.py` had the same bug for its foot
#: order check, where the consequence was only a skipped check.
_CACHE_ATTR = "_mjrl_actuator_joint_ids"


def _actuator_joint_ids(asset, asset_cfg) -> torch.Tensor:
    """Joint index for each of `asset_cfg`'s actuators, in the actuator's order.

    **Derived from the names rather than assumed equal.** On this robot the two
    lists happen to be identical -- 22 actuators, 22 joints, same order -- so
    `joint_vel[:, actuator_ids]` gives the right answer today and would go on
    giving a wrong one silently the moment a model gained a transmission, an
    unactuated joint before an actuated one, or a second actuator on one joint.
    Pairing a torque with another joint's velocity produces a perfectly plausible
    power, which is the kind of number nothing downstream can question.
    """
    joints_key = tuple(asset.joint_names)
    cached = getattr(asset_cfg, _CACHE_ATTR, None)
    if cached is not None and cached[0] == joints_key:
        return cached[1]
    joints = {name: i for i, name in enumerate(asset.joint_names)}
    names = asset.actuator_names
    ids = asset_cfg.actuator_ids
    selected = names if isinstance(ids, slice) else [names[i] for i in ids]
    missing = [n for n in selected if n not in joints]
    if missing:
        raise KeyError(
            f"these actuators drive no joint of the same name: {missing}. The "
            f"power and headroom terms pair each actuator with its joint by name; "
            f"a model whose actuators are named differently needs that mapping "
            f"supplied rather than guessed."
        )
    out = torch.tensor([joints[n] for n in selected], dtype=torch.long)
    # Stored with the joint list it was derived from, so the same config reused
    # against a different entity recomputes rather than returning a mapping into
    # the wrong robot.
    setattr(asset_cfg, _CACHE_ATTR, (joints_key, out))
    return out


def actuator_power(
    env: "ManagerBasedRlEnv",
    asset_cfg,
) -> torch.Tensor:
    """Mechanical power the actuators are spending, in watts. A cost.

        power = mean over actuators of |tau * qd|

    Larger is worse; the weight carries the minus sign.

    **The absolute value is the decision in this line.** Signed `tau * qd` is
    negative while an actuator brakes, and a term that paid for braking would pay
    a policy for falling into its own joint limits. A real servo does not recover
    that energy either -- it burns it in the driver -- so magnitude is both the
    safer gradient and the better model of the hardware.

    **What it is not**: electrical input power. That is roughly `tau * qd` plus
    `I^2 R`, and the resistive half dominates when a joint holds a load without
    moving -- precisely the case a standing hexapod spends most of its time in,
    and precisely what this term cannot see. `actuator_headroom` is the one that
    charges for holding a large torque at zero speed. The two are complementary
    and neither is a substitute for the other.

    A mean rather than a sum, so the number does not change meaning when a task
    drives a different number of actuators.
    """
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.actuator_ids
    tau = asset.data.actuator_force
    tau = tau if isinstance(ids, slice) else tau[:, ids]
    joint_ids = _actuator_joint_ids(asset, asset_cfg).to(tau.device)
    qd = asset.data.joint_vel[:, joint_ids]
    return (tau * qd).abs().mean(dim=1)


def actuator_headroom(
    env: "ManagerBasedRlEnv",
    deadband: float,
    asset_cfg,
) -> torch.Tensor:
    """How close the actuators are to everything they can currently produce.

        ratio = |tau| / servo_limit(|qd|)          in [0, 1]
        cost  = mean over actuators of
                (clamp(ratio - deadband, 0) / (1 - deadband))^2

    Larger is worse, zero below the deadband, exactly 1 for an actuator pinned at
    its limit. The weight carries the minus sign.

    **The denominator is the speed-dependent curve, not `EFFORT_LIMIT`.** What a
    servo can deliver falls off past its corner speed
    (`common/actuator.py::servo_torque_limit`, measured), so 1.0 N*m is 57% of
    capability while the joint is slow and all of it while the joint is fast.
    Dividing by the flat plateau instead would under-charge exactly the case that
    hurts -- a fast joint being asked for everything it has -- and that case is
    where control authority is actually lost, because there is no torque left to
    reject a disturbance with.

    The ratio cannot exceed 1 in simulation: the actuator model clamps to the same
    curve before the force ever reaches `data.actuator_force`. So this term
    measures how much of the clamp the policy is living against, and a value of 1
    means the clamp is doing the controlling.

    **Quadratic above a deadband, for the reason every other shaping term here
    has one**: a gait has to push, and charging for torque as such would make
    standing still cheaper than walking, which is this robot's documented local
    optimum. Below the deadband torque is free.

    `deadband` is **required**, and what it means is "the fraction of current
    capability beyond which sustained operation is a problem". The hardware fact
    it can be read off is `continuous_torque / plateau_torque` = 1.2 / 1.7464 =
    0.687: the share of peak the servo holds indefinitely before its thermal model
    derates it. That identity is exact only at the plateau -- above the corner
    speed the curve falls below 1.2 N*m and the two stop coinciding -- so it is a
    defensible place to start charging rather than a derivation of the right
    number. Each task passes its own.
    """
    from ..actuator import CURVE, servo_torque_limit

    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.actuator_ids
    tau = asset.data.actuator_force
    tau = tau if isinstance(ids, slice) else tau[:, ids]
    joint_ids = _actuator_joint_ids(asset, asset_cfg).to(tau.device)
    qd = asset.data.joint_vel[:, joint_ids]

    limit = servo_torque_limit(
        qd, CURVE.plateau_torque, CURVE.corner_speed, CURVE.decay_speed,
        CURVE.cutoff_speed,
    )
    # Past `cutoff_speed` the curve is exactly zero, and a ratio with zero under
    # it is not a large number, it is a meaningless one. The floor is a thousandth
    # of the plateau: any torque a joint carries up there is already saturated by
    # every reading of the word, and the clamp keeps the gradient finite instead
    # of handing the optimiser an infinity to chase.
    floor = CURVE.plateau_torque * 1e-3
    ratio = (tau.abs() / limit.clamp(min=floor)).clamp(max=1.0)
    return ((ratio - deadband).clamp(min=0.0) / (1.0 - deadband)).square().mean(dim=1)


def stance_foot_ground_gap(
    env: "ManagerBasedRlEnv",
    height_sensor_name: str,
    command_name: str,
    groups: tuple[tuple[int, ...], ...],
    freq_hz: float,
    max_gap: float = 0.05,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """How far the feet that should be down are from the ground. A cost, so the
    weight is negative.

        cost = mean over the legs that should be down of
               clamp(height above terrain, 0, `max_gap`)

    **This is the dense counterpart to `stance_foot_load`.** That term reads
    contact force, and force is exactly zero until the foot arrives: a foot 5 cm
    up, 1 mm up and pressing 4.9 N all score the same, so it says nothing about
    getting closer. Measured on the 25927-iteration tripod run of 2026-09-06,
    about 0.51 of the ~0.98 under-loaded stance legs were not touching at all --
    that half is what this reaches, and it reaches it the whole way down.

    ## The height is above the **terrain**, not above z = 0

    `height_sensor_name` is a `TerrainHeightSensor`, which reports `frame_z -
    hit_z` per foot: the gap to the ground actually under that foot. On a slope
    or a step the two differ by the whole terrain height, and a z-based version
    would demand that a robot on a rise put its feet through the hill.

    ## Standing counts, and is not a special case

    The legs that "should be down" are the clock's support group while the
    command asks for motion, and **all six while it asks the robot to stand** --
    standing on six feet is the requirement, and there is no gait clock then
    anyway (`phase_clock` returns zeros below the command threshold, so the
    support mask would be meaningless rather than merely unused).

    Because of that this term takes no `moving_gate` and no `cycling_gate`.
    `stance_foot_load` needs both: it is a *positive* term, and standing loads
    the designated legs perfectly well, so ungated it would pay for standing
    still. A cost has the opposite failure mode -- gating it off while standing
    would excuse exactly the case the robot is meant to be judged on.

    The mean is over however many legs should be down, not a fixed divisor, so
    walking (3 of 6 in a tripod) and standing (6 of 6) are on one scale. Without
    that, the same average gap would cost twice as much standing as walking and
    the term would quietly become a standing penalty.

    ## Why the clamp

    `max_gap` bounds what one badly-placed leg can contribute. Past a few
    centimetres the leg is not "about to touch down", it is in the wrong part of
    the gait, and that is `tripod_gait`'s job -- it scores the grouping and
    already falls off sharply. Without the clamp a single leg held high would
    dominate the sum and drag the gradient away from the near-contact region the
    term exists to shape.

    **It will pull against `foot_clearance`**, which costs
    `|height - target| * |v_xy|` for every foot. The overlap is narrow: that term
    is weighted by horizontal foot speed, so a support foot that is nearly
    stationary -- the case here -- contributes almost nothing to it. A support
    foot that is both high and moving fast is a genuine gait error and both terms
    should complain. If a retrain shows the two fighting, the sign to look for is
    swing height collapsing, and the fix is this term's `max_gap` rather than its
    weight.
    """
    from .phase import gait_phase, stance_mask

    sensor = env.scene[height_sensor_name]
    heights = sensor.data.heights  # [num_envs, num_feet], above the terrain
    assert heights is not None, f"sensor {height_sensor_name!r} reports no heights"
    n_legs = heights.shape[1]

    moving = moving_gate(env, command_name, command_threshold).unsqueeze(1)
    walking_stance = stance_mask(gait_phase(env, freq_hz), groups, n_legs=n_legs)
    should_be_down = torch.where(moving > 0.0, walking_stance, torch.ones_like(walking_stance))

    gap = heights.clamp(min=0.0, max=max_gap)
    return (gap * should_be_down).sum(dim=1) / should_be_down.sum(dim=1).clamp(min=1.0)


__all__ = [
    "actuator_headroom",
    "actuator_power",
    "body_tilt_rate",
    "contact_state",
    "cycling_gate",
    "leg_side",
    "moving_gate",
    "phase_match",
    "standing_foot_load",
    "stance_foot_ground_gap",
    "stance_foot_load",
    "standing_sway",
    "track_yaw_velocity",
]
