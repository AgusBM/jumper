"""jumper.five_foot: the places where carrying a leg goes wrong without raising.

Walking on five legs is not a parameter change, it is a set of five things that
have to agree with each other, and every one of them fails quietly:

1. **The carried arm is commanded to zero, not held.** mjlab clears
   `joint_pos_target` on reset and only the action term writes it back -- for the
   joints it drives. Take a joint out of the action and its PD pulls it to
   0.0 rad. Nothing raises; the claw simply folds over the first half second of
   every episode.
2. **Five legs on the ground, six columns somewhere.** The contact sensor, the
   foot height scan and the two foot reward terms are three independent places
   naming the feet. Two of them agreeing and the third not leaves every shape
   plausible and pairs foot *i* of one with foot *i* of another.
3. **A foot in the non-foot penalty list.** The undesired-ground-contact term is
   defined by subtraction; get the subtraction wrong and the task pays a penalty
   for standing on its own feet.
4. **The observation's joint set drifting from the action's.** The deploy
   contract has one `obs_joint_order`, and the on-robot builder sizes `joint_pos`,
   `joint_vel` and `joint_torque` from it. Three terms observing three different
   sets exports cleanly and rebuilds the observation at the wrong offsets on the
   robot.
6. **The claw's keys doing the opposite of what they say.** On this joint *more
   open is the lower number*, so a swapped sign still moves the claw, still stays
   in range and still stops at an end. The same term must also stay out of
   training and write nothing until somebody presses a key, or the per-episode
   aperture randomisation is quietly erased.
5. **The nose-down penalty pointing up.** `asin(projected_gravity_x)` has a sign,
   and the wrong one turns a term meant to stop the machine tipping onto its one
   front foot into a term that encourages it.

Each test below names which of those it pins. Where an assertion could pass
vacuously it carries a control group, per this repository's convention.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="the reward terms are torch")
mujoco = pytest.importorskip("mujoco", reason="the pose checks read the model")

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

import tasks
from tasks.jumper.common.constants import (
    GAIT_JOINTS,
    HOME,
    LEGS,
    STAND_Z,
)
from tasks.jumper.five_foot import env_cfg as ff_env_cfg
from tasks.jumper.five_foot.claw import (
    APERTURE_MM,
    ARM_JOINTS,
    CARRIED_JOINTS,
    CARRIED_LEG,
    FINGER_JOINT,
    FIVE_FOOT_JOINTS,
    FIVE_FOOT_LEGS,
    GRASP_BOX,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_TOE_XY,
    LEG_FOOT,
    LF_GRASP,
    LF_GRASP_BOX,
    OBSERVED_JOINTS,
    PAYLOAD_RANGE,
    TRUNK_HALF_WIDTH,
    foot_geoms_for,
)
from tasks.jumper.five_foot.mdp.rewards import (
    GROUP_A,
    GROUP_B,
    body_roll,
    feet_contact_without_cmd,
    foot_outboard_of_trunk,
    group_load_balance,
    low_stance,
    nose_pitch,
)

MODEL = "assets/jumper/jumper.xml"

#: Isaac Lab's and mjlab's shared `soft_joint_pos_limit_factor`.
SOFT_LIMIT_FACTOR = 0.9


# ── The leg sets, before any simulation is involved ───────────────────────



class _Clock:
    """Seconds, moved by the test: what a key's hold is measured by."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

def test_the_carried_leg_is_out_of_the_action_and_in_the_observation() -> None:
    """Pins failure 4, at the level of the joint lists.

    The arm has to be absent from one and present in the other: absent from the
    action because the policy does not drive it, present in the observation
    because it moves -- the hold is re-sampled every episode and the claw's
    aperture with it -- and because those angles are encoder readings the robot
    can actually supply.
    """
    carried = [j for j in GAIT_JOINTS if j.startswith(f"{CARRIED_LEG}_")]
    assert carried, "the carried leg contributes no gait joints; the prefix is wrong"

    assert not set(carried) & set(FIVE_FOOT_JOINTS), (
        "the carried arm is in the action, so the policy is driving a leg it is "
        "supposed to be holding"
    )
    assert set(FIVE_FOOT_JOINTS) < set(OBSERVED_JOINTS), (
        "every driven joint must also be observed"
    )
    assert set(CARRIED_JOINTS) < set(OBSERVED_JOINTS), (
        "the carried arm is not observed, so the policy cannot tell where the "
        "claw is or what it is holding -- which is the whole of what survived "
        "kk-rl-lab's gripper command"
    )
    # The other front leg's finger is a constant in FOOT mode and must stay out.
    assert "RF_J4_joint" not in OBSERVED_JOINTS
    assert len(FIVE_FOOT_JOINTS) == 16
    assert len(OBSERVED_JOINTS) == 21


def test_five_foot_legs_is_the_family_order_minus_the_carried_one() -> None:
    """Pins failure 2 at its source: one leg tuple, from the family's order.

    Order matters as much as membership -- `HOME_TOE_XY`, `GROUP_A` / `GROUP_B`,
    `RF_FOOT` and the reward config's `MID_REAR_FEET` are all *positions* in this
    tuple.
    """
    assert FIVE_FOOT_LEGS == tuple(leg for leg in LEGS if leg != CARRIED_LEG)
    assert FIVE_FOOT_LEGS == ("RF", "LM", "RM", "LR", "RR")


def test_the_non_foot_geoms_are_exactly_everything_that_is_not_a_foot() -> None:
    """Pins failure 3.

    The two lists come from one subtraction, so the properties to check are that
    they do not overlap (a foot in the penalty list charges the robot for
    standing) and that together they are the whole collision scheme (a geom in
    neither is unpenalised ground contact nobody notices).
    """
    feet = set(ff_env_cfg.GROUND_FOOT_GEOMS)
    others = set(ff_env_cfg.NON_FOOT_GEOMS)
    # The scheme the task installs on its robot, not the shared one: the jaw
    # pieces are in this one only (`jaws.py::JAW_COLLISION`).
    (scheme,) = tasks.load_env_cfg("jumper.five_foot").scene.entities["robot"].collisions
    everything = set(scheme.geom_names_expr)

    assert not feet & others, f"in both lists: {sorted(feet & others)}"
    assert feet | others == everything, (
        f"neither list covers {sorted(everything - (feet | others))}"
    )
    assert len(feet) == len(FIVE_FOOT_LEGS)
    # The carried leg's jaw is a foot in the six-foot tasks and must not be one here.
    assert f"{LEG_FOOT[CARRIED_LEG]}_meshcol" in others


def _robot_jaw_geoms(task: str) -> dict[str, tuple[int, int, tuple[float, ...]]]:
    """The left claw's jaw collision geoms as `task` builds its robot.

    Built the way mjlab builds the entity -- the robot's `spec_fn`, then every
    collision scheme the task gives it -- and read off the compiled model, so it
    is the physics and not the config's opinion of it.
    """
    import mujoco

    from tasks.jumper.five_foot.jaws import JAW_PIECES

    robot = tasks.load_env_cfg(task).scene.entities["robot"]
    spec = robot.spec_fn()
    for scheme in robot.collisions:
        scheme.edit_spec(spec)
    model = spec.compile()
    jaws = tuple(f"{link}_meshcol" for link in JAW_PIECES)
    out = {}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith(jaws):
            out[name] = (int(model.geom_contype[g]), int(model.geom_condim[g]),
                         tuple(float(f) for f in model.geom_friction[g]))
    return out


def test_the_decomposed_jaws_are_this_task_s_and_no_other_s() -> None:
    """Pins both ways the jaws go wrong silently now that the task adds them.

    **Losing the grip.** `jaws.py::decompose_jaws` adds the pieces through the
    robot's `spec_fn` and `JAW_COLLISION` gives the jaw faces their friction. Keep
    the first and lose the second and nothing says so: the shared scheme's names
    reach the pieces anyway, so they still collide, at condim 1. Measured, that
    alone does not fail `grasp_objects.py` with today's props (they out-rank the
    jaw and bring their own 0.35), so this is what pins it -- the grasp check
    would not.

    **Leaking out.** The pieces and the grip friction used to be in the shared
    asset and `constants.py`, so every jumper task walked on a decomposed,
    frictional left claw that only this one uses. `jumper.flat` is the other half
    and the control: one hull per jaw, condim 1, exactly the shared model -- and
    the proof that this reads geoms that can differ.
    """
    from tasks.jumper.five_foot.jaws import GRIP_FRICTION, JAW_PIECES

    five = _robot_jaw_geoms("jumper.five_foot")
    assert len(five) == sum(JAW_PIECES.values()), f"five_foot builds jaws {sorted(five)}"
    for name, (contype, condim, friction) in five.items():
        assert contype, f"{name} is in the model and collides with nothing"
        assert condim == 3, f"{name} is condim {condim}; a grip face at condim 1 holds nothing"
        assert friction == pytest.approx(GRIP_FRICTION), f"{name} friction {friction}"

    flat = _robot_jaw_geoms("jumper.flat")
    assert sorted(flat) == sorted(f"{link}_meshcol" for link in JAW_PIECES), (
        f"jumper.flat builds jaws {sorted(flat)}; the decomposition has leaked out of five_foot"
    )
    assert all(condim == 1 for _, condim, _ in flat.values()), (
        f"jumper.flat's jaws grip: {flat}"
    )


def test_the_gait_groups_are_the_two_diagonals_and_rf_is_in_neither() -> None:
    """The five-foot gait is LM+RR against RM+LR, with RF standing through both.

    It was tripod's partition with the carried leg deleted -- {RF, LM, RR} against
    {RM, LR} -- and the policy trained on it tapped RF 2-5 mm to collect it,
    because lifting RF with LM and RR leaves the trunk on the RM-LR line
    (`mdp/rewards.py::GROUP_A`). Easy to break by renaming a leg or reordering
    `FIVE_FOOT_LEGS`, and broken it is a gait prior that is merely *a* partition:
    so each group is pinned as a true diagonal -- one left and one right, one
    middle and one rear -- and the four legs the gait scores are the four
    `duty_balance` scores, or the two terms shape different gaits.
    """
    from tasks.jumper.five_foot.mdp.rewards import GAIT_FEET

    named = [{FIVE_FOOT_LEGS[i] for i in g} for g in (GROUP_A, GROUP_B)]
    assert named == [{"LM", "RR"}, {"RM", "LR"}]
    assert "RF" not in {FIVE_FOOT_LEGS[i] for i in GAIT_FEET}
    assert sorted(GROUP_A + GROUP_B) == list(GAIT_FEET)
    assert tuple(GAIT_FEET) == tuple(ff_env_cfg.MID_REAR_FEET)
    for group in named:
        assert {leg[0] for leg in group} == {"L", "R"}, f"{group} is one side, not a diagonal"
        assert {leg[1] for leg in group} == {"M", "R"}, f"{group} is one row, not a diagonal"


def _gait_env(contact, air=None, stance=None, command=(0.4, 0.0, 0.0)):
    """A stub env for `five_foot_gait`: the contact sensor and the velocity command."""
    contact = torch.tensor([contact], dtype=torch.float32)
    zeros = torch.zeros_like(contact)
    data = _StubData(
        found=contact,
        current_air_time=torch.tensor([air], dtype=torch.float32) if air else zeros,
        current_contact_time=torch.tensor([stance], dtype=torch.float32) if stance else zeros,
    )
    env = _StubEnv(None, None)
    env.scene.sensors = {"feet_ground_contact": _StubData(data=data)}
    env.command_manager = _StubCommandManager({"twist": torch.tensor([command])})
    return env


def test_rf_neither_earns_nor_costs_the_gait_term() -> None:
    """RF up or down, group A's swing scores the same; LM out of place does not.

    Scored over all five columns, the match wanted RF airborne with group A and
    down with group B -- the demand that made it tap. The control is the same
    change made to a leg that *is* in the gait: LM planted during group A's swing
    is one leg wrong, which the match must still see.
    """
    from tasks.jumper.five_foot.mdp.rewards import five_foot_gait

    def gait(contact):
        return float(five_foot_gait(_gait_env(contact), "feet_ground_contact", "twist")[0])

    #          RF   LM   RM   LR   RR       group A (LM, RR) up
    a_up_rf_down = gait([1.0, 0.0, 1.0, 1.0, 0.0])
    a_up_rf_up = gait([0.0, 0.0, 1.0, 1.0, 0.0])
    assert a_up_rf_down == pytest.approx(1.0) and a_up_rf_up == pytest.approx(1.0), (
        f"RF's state moved the gait term: {a_up_rf_down} planted, {a_up_rf_up} lifted"
    )
    assert gait([1.0, 1.0, 0.0, 0.0, 1.0]) == pytest.approx(1.0), "group B's swing"
    lm_wrong = gait([1.0, 1.0, 1.0, 1.0, 0.0])
    assert lm_wrong < 0.2, f"LM planted in group A's swing still scored {lm_wrong}"


def test_rf_may_stand_as_long_as_it_likes() -> None:
    """The gate ignores RF's stance time, and still sees a gait leg's.

    The family's `cycling_gate` zeroes the gait term when any foot has stood longer
    than `max_contact_time` (1.0 s) -- under it RF, which should stand through
    both phases, zeroed the term by doing so. Control: the same 5 s stance on LM,
    which is marching in place and must still zero it.
    """
    from tasks.jumper.five_foot.mdp.rewards import five_foot_gait

    a_up = [1.0, 0.0, 1.0, 1.0, 0.0]

    def gait(stance):
        env = _gait_env(a_up, stance=stance)
        return float(five_foot_gait(env, "feet_ground_contact", "twist")[0])

    assert gait([5.0, 0.0, 0.2, 0.2, 0.0]) == pytest.approx(1.0), (
        "RF standing 5 s zeroed the gait term"
    )
    assert gait([0.2, 0.0, 5.0, 0.2, 0.0]) == 0.0, "RM standing 5 s did not zero it"

def test_rf_is_charged_for_lifting_outside_its_window() -> None:
    """`rf_step_window`: RF off the ground with any gait leg off it too.

    Its window is the four planted -- with a diagonal pair it stands on a line 86
    mm from the centre of mass (`mdp/rewards.py::GROUP_A`). Each pattern below is
    one a gait actually produces; the control is the window itself, which must
    cost nothing, or the term charges every RF step there is.
    """
    from tasks.jumper.five_foot.mdp.rewards import rf_step_window

    def cost(contact):
        return float(rf_step_window(_gait_env(contact), "feet_ground_contact")[0])

    #             RF   LM   RM   LR   RR
    assert cost([0.0, 1.0, 1.0, 1.0, 1.0]) == 0.0, "RF charged inside its own window"
    assert cost([0.0, 0.0, 1.0, 1.0, 0.0]) == 1.0, "RF lifting with group A"
    assert cost([0.0, 1.0, 0.0, 0.0, 1.0]) == 1.0, "RF lifting with group B"
    assert cost([0.0, 1.0, 1.0, 1.0, 0.0]) == 1.0, "RF lifting with one rear leg"
    assert cost([1.0, 0.0, 1.0, 1.0, 0.0]) == 0.0, "group A's swing charged with RF down"
    assert cost([1.0, 1.0, 1.0, 1.0, 1.0]) == 0.0


def test_the_gait_term_pays_for_rf_s_step_and_not_for_standing() -> None:
    """The third phase, RF alone with the four planted, scores like the other two.

    Without it the four planted together is two legs wrong for either group, so
    the one moment RF may lift would cost the gait term -- and a leg with nowhere
    to step without losing something drags. The control is all five down: the
    same four planted with RF down is standing, and paying for it would pay the
    policy to stop walking.
    """
    from tasks.jumper.five_foot.mdp.rewards import five_foot_gait

    def gait(contact):
        return float(five_foot_gait(_gait_env(contact), "feet_ground_contact", "twist")[0])

    assert gait([0.0, 1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0), "RF's step is not scored"
    standing = gait([1.0, 1.0, 1.0, 1.0, 1.0])
    assert standing < 0.01, f"all five planted scored {standing} while walking"


def _rf_drag(contact, rf_vel, walking=True) -> float:
    """`rf_drag` on the gait stub, with RF's site moving at `rf_vel` (m/s, world)."""
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from tasks.jumper.five_foot.mdp.rewards import rf_drag

    env = _gait_env(contact, command=(0.4, 0.0, 0.0) if walking else (0.0, 0.0, 0.0))
    env.scene._robot = _StubAsset(_StubData(site_lin_vel_w=torch.tensor([[rf_vel]])))
    asset_cfg = SceneEntityCfg("robot")
    asset_cfg.site_ids = [0]
    return float(rf_drag(env, "feet_ground_contact", "twist", 0.05, asset_cfg)[0])


def test_rf_pays_for_sliding_on_the_floor_and_not_for_stepping() -> None:
    """`rf_drag`: RF's speed while touching, walking -- the half the window left out.

    Allowed to lift only in a narrow window and charged nothing for sliding, RF
    slid: 134 mm/s while touching at 0.4 m/s forward on `model_21996`. So a touching
    RF sliding at 0.1 m/s costs 0.1, and three things must cost nothing: RF moving
    through the air, which is the step being asked for; the same slide while
    standing, which `feet_still_when_standing` already charges; and the control, a
    touching RF standing still, or the term is a charge on RF being down at all.
    """
    #          RF   LM   RM   LR   RR
    down = [1.0, 1.0, 1.0, 1.0, 1.0]
    up = [0.0, 1.0, 1.0, 1.0, 1.0]
    assert _rf_drag(down, [0.1, 0.0, 0.0]) == pytest.approx(0.1), "RF sliding along the floor"
    assert _rf_drag(down, [0.06, 0.08, 0.0]) == pytest.approx(0.1), "the slide is |v_xy|"
    assert _rf_drag(up, [0.3, 0.0, 0.1]) == 0.0, "RF charged for stepping through the air"
    assert _rf_drag(down, [0.1, 0.0, 0.0], walking=False) == 0.0, (
        "standing, RF's slide is feet_still's and was charged twice"
    )
    assert _rf_drag(down, [0.0, 0.0, 0.0]) == 0.0, "a still, planted RF costs something"


def test_rf_drag_is_wired_as_its_notes_say(cfg) -> None:
    """RF's site, the gait terms' sensor, and `feet_still`'s standing threshold.

    The threshold is what makes the two a partition -- `feet_still` below it, this
    above -- and a site other than RF's would charge the wrong leg's slide while
    reading RF's contact column.
    """
    term = cfg.rewards["rf_drag"]
    assert term.weight < 0
    assert list(term.params["asset_cfg"].site_names) == ["RF"]
    assert term.params["sensor_name"] == cfg.rewards["five_foot_gait"].params["sensor_name"]
    assert term.params["command_threshold"] == cfg.rewards["feet_still"].params["command_threshold"]


def test_the_rf_window_is_wired_as_its_notes_say(cfg) -> None:
    """A penalty, on RF's own column, reading the sensor the gait term reads.

    `RF_FOOT` is a hard-coded index into `FIVE_FOOT_LEGS`, so a reordering there
    would move the window onto another leg without a word; and a second sensor
    would let the two terms disagree about what airborne means.
    """
    from tasks.jumper.five_foot.mdp.rewards import RF_FOOT

    assert FIVE_FOOT_LEGS[RF_FOOT] == "RF"
    term = cfg.rewards["rf_step_window"]
    assert term.weight < 0
    assert term.params["sensor_name"] == cfg.rewards["five_foot_gait"].params["sensor_name"]


def _model():
    return mujoco.MjModel.from_xml_path(MODEL)


def _soft_limits(m, joint: str) -> tuple[float, float]:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
    lo, hi = (float(v) for v in m.jnt_range[jid])
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    return centre - SOFT_LIMIT_FACTOR * half, centre + SOFT_LIMIT_FACTOR * half


def test_every_corner_of_the_hold_box_is_inside_the_soft_limits() -> None:
    """A hold that reaches a limit is a joint driven into its stop once an episode.

    On hardware that is a mechanical event, and in simulation it is a silent one:
    the joint clamps, the arm sits somewhere other than where the sampler put it,
    and every episode starts from a slightly different lie.
    """
    m = _model()
    for joint in ARM_JOINTS:
        lo, hi = _soft_limits(m, joint)
        for corner in (LF_GRASP[joint] - GRASP_BOX, LF_GRASP[joint] + GRASP_BOX):
            assert lo <= corner <= hi, (
                f"{joint} reaches {corner:+.4f}, outside its soft limits "
                f"[{lo:+.4f}, {hi:+.4f}]"
            )


def test_the_soft_limit_check_can_fail() -> None:
    """Control for the test above, which would pass on any pose inside any range.

    A joint parked at its hard limit must be caught.
    """
    m = _model()
    joint = ARM_JOINTS[0]
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
    at_the_stop = float(m.jnt_range[jid][1])
    lo, hi = _soft_limits(m, joint)
    assert not (lo <= at_the_stop + GRASP_BOX <= hi), (
        "the soft-limit assertion accepts a joint past its hard stop, so the "
        "test above proves nothing"
    )


def test_the_claw_apertures_are_the_working_range_of_the_finger() -> None:
    """`GRIPPER_OPEN` and `GRIPPER_CLOSED` bracket a monotonic stretch of travel.

    On V1.6 the mouth keeps widening past -1.00, but what it closes along does
    not survive: between -0.99 and -1.00 the nearest part of the moving jaw stops
    being its grip face and becomes an edge, and the closing axis swings by 23
    degrees (`claw.py` has the sweep). The curve reaches zero at 0.0, the joint's
    own limit. A hold sampled outside that stretch pinches along a different
    direction than the one the pose was searched for, which is not a fault
    anything else would report.

    The monotonicity is checked against `APERTURE_MM` rather than asserted in
    prose, because that table is the thing that would be regenerated and could
    come back with the turning point inside the range.
    """
    jid = mujoco.mj_name2id(_model(), mujoco.mjtObj.mjOBJ_JOINT, FINGER_JOINT)
    lo, hi = (float(v) for v in _model().jnt_range[jid])
    assert lo < GRIPPER_OPEN < GRIPPER_CLOSED <= hi
    assert APERTURE_MM[0][0] == pytest.approx(GRIPPER_OPEN)
    assert APERTURE_MM[-1][0] == pytest.approx(GRIPPER_CLOSED)
    widths = [mm for _, mm in APERTURE_MM]
    assert widths == sorted(widths, reverse=True), (
        f"the aperture curve is not monotone over [{GRIPPER_OPEN}, "
        f"{GRIPPER_CLOSED}]: {widths}"
    )
    assert widths[-1] < 1.0, "the claw does not shut at GRIPPER_CLOSED"
    # The nominal hold is the open end; `claw.py` records the measured curve.
    assert LF_GRASP[FINGER_JOINT] == GRIPPER_OPEN
    assert FINGER_JOINT not in LF_GRASP_BOX, (
        "the finger belongs to the aperture range, not to the arm's pose box"
    )


def test_the_home_toe_positions_match_the_model() -> None:
    """`HOME_TOE_XY` is a constant standing in for a measurement.

    `foot_home_position` scores every foot against it, so a stale entry is a
    reward term quietly asking for a stance the robot does not have. Re-measured
    here against the asset, with the robot standing at HOME.

    Read from `site_xpos`, because that is what `grasp_pose.py --stance` measured
    it from and what the reward term compares against. The foot *bodies* sit
    2.4-5.8 mm from their sites in the body frame and happen to agree in xy at
    this stance, which is a property of the pose rather than of the model -- so
    checking the bodies would pass today and stop meaning anything the moment a
    leg turned.
    """
    m = _model()
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    free = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
    d.qpos[int(m.jnt_qposadr[free]) + 2] = STAND_Z
    for joint, value in HOME.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
        d.qpos[int(m.jnt_qposadr[jid])] = value
    mujoco.mj_forward(m, d)

    for leg, expected in zip(FIVE_FOOT_LEGS, HOME_TOE_XY, strict=True):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, leg)
        actual = d.site_xpos[sid][:2]
        assert actual == pytest.approx(expected, abs=1e-3), (
            f"{leg}: the model puts its foot at {actual.round(4)}, HOME_TOE_XY "
            f"says {expected}"
        )


# ── The reward terms' signs, on plain tensors ─────────────────────────────


class _StubAsset:
    def __init__(self, data) -> None:
        self.data = data


class _StubData:
    def __init__(self, **kw) -> None:
        self.__dict__.update(kw)


class _StubScene:
    def __init__(self, robot, origins) -> None:
        self._robot = robot
        self.env_origins = origins

    def __getitem__(self, _name):
        return self._robot


class _StubCommandManager:
    def __init__(self, commands: dict) -> None:
        self._commands = commands

    def get_command(self, name: str):
        return self._commands[name]


class _StubEnv:
    def __init__(self, robot, origins) -> None:
        self.scene = _StubScene(robot, origins)


def _pitched(deg: float) -> torch.Tensor:
    """Projected gravity for a body pitched `deg` **nose-down**.

    Derived rather than written down: rotate the body about +y and read where the
    nose (body +x) ends up in the world. A rotation that puts the nose below the
    horizon is nose-down, and the gravity vector is then expressed in that frame.
    """
    half = math.radians(deg) / 2.0
    quat = torch.tensor([[math.cos(half), 0.0, math.sin(half), 0.0]])
    nose_z = float(quat_apply(quat, torch.tensor([[1.0, 0.0, 0.0]]))[0, 2])
    if deg > 0:
        assert nose_z < 0, "a positive rotation about +y should put the nose down"
    return quat_apply_inverse(quat, torch.tensor([[0.0, 0.0, -1.0]]))


def _feet_still(n_envs: int = 2, settle_time: float = 0.5):
    """`feet_still_when_standing` on a stub env: two environments, five feet.

    Returns `(term, call, set_cmd, set_vel)`. The velocities are the sites' world
    linear velocities, which is what the term reads; the command is the velocity
    twist `[vx, vy, wz]`.
    """
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from tasks.jumper.five_foot.mdp.rewards import feet_still_when_standing

    vel = torch.zeros(n_envs, len(FIVE_FOOT_LEGS), 3)
    cmd = {"twist": torch.zeros(n_envs, 3)}
    robot = _StubAsset(_StubData(site_lin_vel_w=vel))
    env = _StubEnv(robot, None)
    env.command_manager = _StubCommandManager(cmd)
    env.num_envs, env.device, env.step_dt = n_envs, "cpu", 0.02
    asset_cfg = SceneEntityCfg("robot")
    asset_cfg.site_ids = list(range(len(FIVE_FOOT_LEGS)))
    params = {"command_name": "twist", "command_threshold": 0.05,
              "settle_time": settle_time, "asset_cfg": asset_cfg}
    term = feet_still_when_standing(RewardTermCfg(func=None, weight=-1.0, params=params), env)

    def call() -> torch.Tensor:
        return term(env, **params)

    def set_cmd(value) -> None:
        cmd["twist"][:] = torch.as_tensor(value, dtype=torch.float32)

    def set_vel(foot: int, value) -> None:
        vel.zero_()
        vel[:, foot] = torch.as_tensor(value, dtype=torch.float32)

    return term, call, set_cmd, set_vel


def test_the_feet_hold_still_only_while_the_robot_is_asked_to_stand() -> None:
    """Pins the gate on `feet_still_when_standing`, which is the whole of it.

    **Inverted**, it charges every step of every walk and says nothing while the
    feet walk round a standing twist -- the failure it exists to catch, with a
    plausible number on the dashboard. **Missing**, it charges walking as well and
    tilts the walk/stand trade-off toward standing, which on this task is the
    documented way for a run to end parked. So one foot moving at 0.1 m/s is
    charged 0.1 standing and nothing walking -- and a turn on the spot counts as
    walking, since yaw rate is inside the norm.

    The control is the resting foot, which costs nothing standing either: the term
    measures motion and not a stance.
    """
    _, call, set_cmd, set_vel = _feet_still(settle_time=0.02)  # one step: no ramp here
    set_vel(0, [0.1, 0.0, 0.0])
    set_cmd([0.0, 0.0, 0.0])
    assert torch.allclose(call(), torch.tensor([0.1, 0.1]), atol=1e-6), "a foot walking while standing"
    set_cmd([0.3, 0.0, 0.0])
    assert torch.allclose(call(), torch.zeros(2)), "charged a foot for walking when told to walk"
    set_cmd([0.0, 0.0, 0.4])
    assert torch.allclose(call(), torch.zeros(2)), "charged a foot for turning when told to turn"

    set_cmd([0.0, 0.0, 0.0])
    set_vel(0, [0.0, 0.0, 0.0])
    call()
    assert torch.allclose(call(), torch.zeros(2)), "a foot at rest costs something"


def test_the_feet_are_charged_for_where_they_stand_not_for_lifting() -> None:
    """Horizontal speed only, and summed over the feet.

    A pad lifting and landing on the same spot moves nothing an operator aimed;
    counted, it would charge the policy for unloading a foot to shift its weight.
    Two feet moving cost twice one, or the term would stop caring about the
    second foot once the first had moved.
    """
    _, call, set_cmd, set_vel = _feet_still(settle_time=0.02)
    set_cmd([0.0, 0.0, 0.0])
    set_vel(2, [0.0, 0.0, 0.3])
    assert torch.allclose(call(), torch.zeros(2)), "a vertical lift was charged as moving the foot"
    set_vel(2, [0.03, 0.04, 0.3])
    assert torch.allclose(call(), torch.full((2,), 0.05), atol=1e-6), "horizontal speed is |v_xy|"


def test_the_charge_waits_for_a_stopping_robot_to_land_its_feet() -> None:
    """The ramp after the command goes idle -- `standing_sway`'s documented problem.

    Told to stop mid-stride, a robot still has swing feet in the air that must
    land; charged from the first idle step, the lesson is an abrupt stop rather
    than a still stance. So the charge ramps in over `settle_time` (25 control
    steps at 0.5 s), restarts whenever the robot is told to move again, and starts
    from zero on a fresh episode. Each half is checked against the full charge it
    must not reach yet, so a term with no ramp fails the first assertion.
    """
    term, call, set_cmd, set_vel = _feet_still(settle_time=0.5)
    set_vel(0, [0.1, 0.0, 0.0])
    set_cmd([0.0, 0.0, 0.0])
    first = call()
    assert torch.allclose(first, torch.full((2,), 0.1 / 25), atol=1e-6), (
        f"the first idle step cost {first.tolist()}; the stopping transient is charged in full"
    )
    for _ in range(30):
        last = call()
    assert torch.allclose(last, torch.full((2,), 0.1), atol=1e-6), "the charge never reaches full"

    set_cmd([0.3, 0.0, 0.0])
    call()
    set_cmd([0.0, 0.0, 0.0])
    assert call()[0] < 0.1 / 2, "a new stop was charged as if the robot had been standing"

    for _ in range(30):
        call()
    term.reset(env_ids=torch.tensor([1]))
    after = call()
    assert after[0] == pytest.approx(0.1) and after[1] < 0.1 / 2, (
        "reset did not restart the ramp for the environment it was given, and only that one"
    )


def test_the_feet_still_term_is_wired_as_its_notes_say(cfg) -> None:
    """The three agreements the term's weight and docstring rest on.

    The five ground feet, in `FIVE_FOOT_LEGS` order. The same standing threshold as
    `feet_slip` and `standing_sway` -- the claim that it is the complement of the
    first and gates like the second is only true while the numbers agree. And a
    settle time equal to the gait's longest swing, which is why a step in progress
    lands before the charge is whole.
    """
    import inspect

    from tasks.jumper.five_foot.mdp.rewards import five_foot_gait

    term = cfg.rewards["feet_still"]
    assert term.weight < 0
    assert list(term.params["asset_cfg"].site_names) == list(FIVE_FOOT_LEGS)
    assert term.params["asset_cfg"].preserve_order
    assert term.params["command_name"] == "twist"
    for other in ("foot_slip", "standing_sway"):
        assert term.params["command_threshold"] == cfg.rewards[other].params["command_threshold"], (
            f"feet_still and {other} disagree about what standing is"
        )
    gait = cfg.rewards["five_foot_gait"].params
    max_air = gait.get(
        "max_air_time", inspect.signature(five_foot_gait).parameters["max_air_time"].default
    )
    assert term.params["settle_time"] == pytest.approx(max_air)


def _twist_rig(settle_time: float = 0.1):
    """`body_twist` on a stub env: one environment standing on `HOME_TOE_XY`.

    Returns `(term, place, call)`. `place(trunk_deg, feet_deg, walking)` puts the
    trunk at a yaw and the five feet at `HOME_TOE_XY` turned about the origin by
    `feet_deg` in the world -- all bearing -- and sets the velocity command to
    walk or stand; `call(twist_cmd_deg)` scores it.
    """
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from tasks.jumper.five_foot.mdp.rewards import body_twist

    n = len(HOME_TOE_XY)
    home = torch.tensor(HOME_TOE_XY, dtype=torch.float32)
    sites = torch.zeros(1, n, 3)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    data = _StubData(
        data=_StubData(site_xpos=sites), indexing=_StubData(site_ids=torch.arange(n)),
        root_link_pos_w=torch.zeros(1, 3), root_link_quat_w=quat,
    )
    env = _StubEnv(_StubAsset(data), None)
    force = torch.zeros(1, n, 3)
    force[..., 2] = 10.0
    env.scene.sensors = {"feet_ground_contact": _StubData(data=_StubData(force=force))}
    cmds = {"body_pose": torch.zeros(1, 3), "twist": torch.zeros(1, 3)}
    env.command_manager = _StubCommandManager(cmds)
    env.num_envs, env.device, env.step_dt = 1, "cpu", 0.02
    asset_cfg = SceneEntityCfg("robot")
    asset_cfg.site_ids = list(range(n))
    params = {"home_xy": HOME_TOE_XY, "sensor_name": "feet_ground_contact",
              "command_name": "body_pose", "std": math.radians(15.0),
              "velocity_command_name": "twist", "command_threshold": 0.05,
              "settle_time": settle_time, "asset_cfg": asset_cfg}
    term = body_twist(RewardTermCfg(func=None, weight=1.0, params=params), env)

    def place(trunk_deg: float, feet_deg: float, walking: bool = False) -> None:
        f = math.radians(feet_deg)
        rot = torch.tensor([[math.cos(f), -math.sin(f)], [math.sin(f), math.cos(f)]])
        sites[0, :, :2] = home @ rot.T
        t = math.radians(trunk_deg) / 2.0
        quat[0] = torch.tensor([math.cos(t), 0.0, 0.0, math.sin(t)])
        cmds["twist"][0] = torch.tensor([0.3 if walking else 0.0, 0.0, 0.0])

    def call(twist_cmd_deg: float) -> float:
        cmds["body_pose"][0, 2] = math.radians(twist_cmd_deg)
        return float(term(env, **params)[0])

    return term, place, call


def test_a_twist_the_feet_walk_in_earns_nothing_while_standing() -> None:
    """Pins the loophole closed, and the intended twist still scored in full.

    The trunk-to-footprint offset is (trunk's turn) - (footprint's turn), so feet
    walking -20 degrees round a still trunk read as a +20 degree twist -- the very
    behaviour `model_16997` learned, paid in full. Once the robot has stood and
    settled, the footprint's own turn is added back, so what is scored is the
    trunk's: zero here, against a command of 20, which the 15 degree kernel
    prices at exp(-(20/15)^2) - 1 = -0.83.

    Two controls. The trunk turning +20 over planted feet -- the command obeyed --
    must still score 0. And walking, where feet have to move, the record follows
    them and the plain offset is scored, so the same feet-walked footprint is
    not charged there.
    """
    _, place, call = _twist_rig()
    for _ in range(10):                  # stand and settle on HOME_TOE_XY
        place(0.0, 0.0)
        call(0.0)
    place(0.0, -20.0)
    walked_in = call(20.0)
    assert walked_in == pytest.approx(math.exp(-((20 / 15) ** 2)) - 1.0, abs=1e-3), (
        f"feet walking round a still trunk scored {walked_in:.3f} as a 20 degree twist"
    )

    _, place, call = _twist_rig()
    for _ in range(10):
        place(0.0, 0.0)
        call(0.0)
    place(20.0, 0.0)
    assert call(20.0) == pytest.approx(0.0, abs=1e-4), "the trunk turning over planted feet"

    _, place, call = _twist_rig()
    for _ in range(10):
        place(0.0, 0.0, walking=True)
        call(0.0)
    place(0.0, -20.0, walking=True)
    assert call(20.0) == pytest.approx(0.0, abs=1e-4), (
        "walking, the footprint was held to a record it is meant to leave"
    )


def test_the_footprint_record_waits_for_the_feet_and_restarts_with_the_episode() -> None:
    """The record is taken once the feet have settled, and again after a reset.

    Taken on the first idle step, it would hold the feet where a stopping robot's
    swing foot happened to be in the air; never retaken, a new episode would be
    scored against the last one's footprint. Each half is checked against the
    charge the other would produce.
    """
    term, place, call = _twist_rig(settle_time=0.1)   # 5 control steps
    place(0.0, 0.0)
    call(0.0)
    place(0.0, -20.0)                                  # still settling: the record follows
    assert call(20.0) == pytest.approx(0.0, abs=1e-4), "the record froze before the feet landed"
    for _ in range(10):
        call(20.0)
    place(0.0, -30.0)                                  # settled on -20: a further -10 is caught
    later = call(30.0)
    assert later == pytest.approx(math.exp(-((10 / 15) ** 2)) - 1.0, abs=1e-3), later

    term.reset(env_ids=torch.tensor([0]))
    for _ in range(10):
        call(0.0)                                      # a new episode settles on -30
    assert call(30.0) == pytest.approx(0.0, abs=1e-4), "a new episode is scored against the old record"


def test_the_idle_hold_lets_go_while_the_robot_turns() -> None:
    """`stand_still_when_idle` must not charge a policy for obeying a turn.

    The term pulls the joints back to the stance as the command approaches zero,
    and kk-rl-lab -- which it is copied from -- takes the norm of the command's
    **linear half** to decide what "zero" means. A pure turn, vx = vy = 0 and
    wz = 0.5, reads as idle under that gate: the robot is asked to spin and then
    charged for every joint it moves to do it. Yaw rate is inside the norm here
    for that reason, and this is what says so.
    """
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    from tasks.jumper.five_foot.mdp.rewards import stand_still_when_idle

    asset = _StubAsset(_StubData(
        joint_pos=torch.tensor([[0.2, -0.1, 0.0]]),
        default_joint_pos=torch.zeros(1, 3),
    ))
    cfg = SceneEntityCfg("robot")
    cfg.joint_ids = slice(None)

    def term(twist, pose=(0.0, 0.0, 0.0)):
        env = _StubEnv(asset, torch.zeros(1, 3))
        env.command_manager = _StubCommandManager({
            "twist": torch.tensor([list(twist)]),
            "body_pose": torch.tensor([list(pose)]),
        })
        return float(stand_still_when_idle(env, "twist", "body_pose", cfg)[0])

    idle = term((0.0, 0.0, 0.0))
    assert idle == pytest.approx(0.3), "the idle penalty is the joints' L1 deviation"
    assert term((0.3, 0.0, 0.0)) == 0.0, "it charges a walking policy"
    assert term((0.0, 0.0, 0.5)) == 0.0, "it charges a turning policy"
    assert term((0.0, 0.0, 0.0), pose=(0.2, 0.0, 0.0)) == 0.0, (
        "it fights the attitude command: the trunk cannot be pitched from the "
        "home joints"
    )

    # Ramped, not switched: half the threshold is half the penalty, so there is
    # no cliff for a policy to sit on the edge of.
    assert term((0.03, 0.0, 0.0)) == pytest.approx(idle * 0.5, abs=1e-6)

    # The control: this is a real departure from the reference, so the reference's
    # own gate has to disagree here. Take the linear norm only, as it does, and a
    # turning robot is charged the full idle penalty.
    twist = torch.tensor([[0.0, 0.0, 0.5]])
    reference_alpha = (1.0 - twist[:, :2].norm(dim=1) / 0.06).clamp(0.0, 1.0)
    assert float(reference_alpha[0]) == 1.0


def test_after_a_reset_the_trigger_still_decides(make_claw) -> None:
    """The trigger is the claw, across a reset as well.

    `hold_carried_arm` pins the finger wide open on reset. Under the keyboard
    teleop this replaced, the operator's stored target then walked the claw back
    shut -- measured on warp/cuda: reset put the joint at -0.700, and it was at
    -0.314 after 0.2 s. With the trigger nothing is stored to walk back to: a
    trigger at rest has the claw open on the next step, and one still squeezed
    closes it again, which is what the same squeeze does on the robot. The
    squeezed half is the control group, that the term writes after a reset at all.
    """
    from tasks.jumper.five_foot.claw import GRIPPER_OPEN

    term, robot = make_claw(resting=GRIPPER_CLOSED)
    term.op.squeeze = 0.6
    term(term.env)
    squeezed = term._target
    assert squeezed > GRIPPER_OPEN, "the squeeze did not close the claw"

    term.env.reset()
    term(term.env)
    assert term._target == squeezed, "a reset overrode the trigger still squeezed"

    term.op.squeeze = 0.0
    term.env.episode_length_buf += 5
    for _ in range(3):
        term(term.env)
    assert float(robot.writes[-1][0].max()) == pytest.approx(GRIPPER_OPEN), (
        "let go after a reset, it stayed shut"
    )


def test_replay_starts_with_an_empty_claw(cfg) -> None:
    """Nothing in the claw until the operator puts something there.

    The payload is a training device: a sampled 0-600 g that the policy has to
    learn to walk under. In replay it is invisible -- there is no object on
    screen and nothing says the claw is loaded -- so a session was driving a
    robot whose posture was wrong for a reason nobody could see, and picking
    something up then stacked the real object on top of it. Measured before this
    gate: a replay session carrying 311 g of nothing.

    What replay gets instead is a real prop: one the claw squeezes, or `play
    --hold`, which starts the episode with one squeezed in the mouth.
    """

    assert "claw_payload" in cfg.events, "training lost its payload"
    assert "claw_payload" not in tasks.load_env_cfg("jumper.five_foot", play=True).events


def test_the_claw_hold_is_replay_only_and_is_what_hold_reaches(cfg) -> None:
    """`play --hold` is one parameter on one term, and both ends of it can go
    quietly wrong.

    **Training reaching it.** A term that parks a prop in the claw and shuts the
    finger has no business in a training run: there are no props there, and a term
    writing the finger's target would override the aperture the policy is meant to
    see sampled. So it is replay-only, like the claw's teleop.

    **The flag not reaching it.** `--hold` is this task's own flag
    (`__init__.py::cli_args`), handed to `env_cfg` by name, and `_start_holding`
    finds the term by name -- a rename on any of the three leaves `--hold` refusing
    a task that can hold things, or succeeding against a term nothing reads. So the
    flag goes the way an entry point sends it: through the task's own argument
    group, parsed, into `load_env_cfg`. The control is the same config with the
    term removed, which has to be refused, or the lookup is not what the first half
    tested.
    """
    import argparse
    import warnings

    assert "claw_hold" not in cfg.events, "a term that parks props in the claw trains"
    play = tasks.load_env_cfg("jumper.five_foot", play=True)
    assert "claw_hold" in play.events, "replay has nothing for --hold to reach"

    parser = argparse.ArgumentParser()
    names = tasks.load_cli_args("jumper.five_foot")(parser.add_argument_group("five_foot"))
    args = parser.parse_args(["--objects", "--hold", "can"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        held = tasks.load_env_cfg(
            "jumper.five_foot", play=True, task_args={n: getattr(args, n) for n in names}
        )
    assert held.events["claw_hold"].params["start_holding"] == "can"

    del held.events["claw_hold"]
    with pytest.raises(SystemExit):
        ff_env_cfg._start_holding(held, "can")
    # And a prop that is not there -- `--hold` without the row -- is refused too.
    with pytest.raises(SystemExit, match="no such prop"):
        tasks.load_env_cfg("jumper.five_foot", play=True, task_args={"hold": "can"})


def _jaw_meshes(model, data) -> tuple:
    """The fixed and the moving jaw's drawn vertices, in world coordinates.

    Through each geom's own frame rather than its body's: `mesh_vert` is in the
    mesh's frame, which the compiler re-orients per geom, and read against the
    body the cloud lands beside the part (`mdp/grasp.py::_jaw_clouds`).
    """
    import mujoco
    import numpy as np

    def cloud(body: str):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{body}_visual")
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[start:start + count]
        return verts @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]

    return (np.vstack([cloud("LF_palm_link"), cloud("LF_palm_pad_f_link"), cloud("LF_palm_pad_b_link")]),
            np.vstack([cloud("LF_finger_link"), cloud("LF_finger_tip_link")]))


def test_the_payload_hangs_where_the_claw_holds_something() -> None:
    """The load sits where a held object's centre is -- measured against the jaws.

    It has been in four wrong places, and a test agreed with more than one:

    - a point mass on `LF_finger_link`, the **moving** jaw, with no inertia;
    - at the wrist, 116 mm off, placed by a helper whose name lookups all missed
      on the bare asset -- and the test compared the body with the same helper's
      output, which agreed with itself exactly;
    - on the anvil, the fixed jaw's own surface: an object whose centre is there
      is inside the jaw;
    - at a "mouth centre" read off `LF_palm_pad_b_link`'s vertices in its body frame, which
      is past the tips and outside the jaws.

    So this does not ask the helper. It poses the arm, reads where the payload
    body ended up, and measures that against the jaws' own meshes: clear of the
    fixed jaw by at least the radius of the object it models, and between the two
    jaws with the claw wide open. The control is that the last three places, each
    as it was in the palm frame, fail one of the two.
    """
    import mujoco
    import numpy as np
    from scipy.spatial import ConvexHull

    from tasks.jumper.common.constants import HOME
    from tasks.jumper.five_foot.claw import (
        FINGER_JOINT,
        GRIPPER_OPEN,
        LF_GRASP,
        PAYLOAD_BODY,
        PAYLOAD_CYLINDER,
        add_payload_body,
    )
    from tasks.jumper.five_foot.jaws import task_spec

    model = add_payload_body(task_spec()).compile()
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PAYLOAD_BODY)
    assert body >= 0, f"no {PAYLOAD_BODY} body in the compiled model"
    parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[body])
    assert parent == "LF_palm_link"
    assert model.body_jntnum[body] == 0, "the payload is on a joint; it would swing"
    # Nothing until the event loads it, so a run with PAYLOAD_RANGE at zero
    # carries nothing rather than a nominal somebody forgot about.
    assert model.body_mass[body] == 0.0, "the payload weighs something before the event"
    assert model.body_inertia[body] == pytest.approx([0.0, 0.0, 0.0])

    data = mujoco.MjData(model)
    pose = list(HOME.items()) + list(LF_GRASP.items()) + [(FINGER_JOINT, GRIPPER_OPEN)]
    for joint, value in pose:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)
    fixed, moving = _jaw_meshes(model, data)
    between = ConvexHull(np.vstack([fixed, moving])).equations

    def clearance(point) -> float:
        return float(np.linalg.norm(fixed - point, axis=1).min())

    def outside(point) -> float:
        return float((between[:, :3] @ point + between[:, 3]).max())

    radius = PAYLOAD_CYLINDER[0]
    at = data.xpos[body]
    assert clearance(at) > radius, (
        f"the payload is {clearance(at) * 1000:.1f} mm from the fixed jaw; an object "
        f"{radius * 2000:.0f} mm across with its centre there is inside the jaw"
    )
    assert outside(at) < 0.0, (
        f"the payload is {outside(at) * 1000:.1f} mm outside the jaws with the claw "
        f"wide open; nothing is held there"
    )

    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LF_palm_link")

    def in_palm(mm: tuple[float, float, float]):
        return data.xpos[palm] + data.xmat[palm].reshape(3, 3) @ (np.array(mm) / 1000.0)

    # On V1.6: the palm's own origin, the anvil, and a point past the jaws' tips.
    assert clearance(in_palm((0.0, 0.0, 0.0))) < radius, "the wrist passes"
    assert clearance(in_palm((-10.0, 130.4, -1.3))) < radius, "the anvil passes"
    assert outside(in_palm((-10.0, 200.0, -1.3))) > 0.0, "a point past the tips passes"


def test_the_claw_measurements_refuse_a_name_they_cannot_find() -> None:
    """A missed lookup is an error, never an index.

    `mj_name2id` returns -1 for a name it does not have, and -1 is a valid numpy
    index -- the last body, the last joint. That is how the payload was placed in
    the frame of a toe with nothing raising.
    """
    import mujoco

    from tasks.jumper.common.constants import get_spec
    from tasks.jumper.five_foot.mdp.grasp import _named

    model = get_spec().compile()
    assert _named(model, mujoco.mjtObj.mjOBJ_BODY, "LF_palm_link") >= 0
    with pytest.raises(ValueError, match="no mjOBJ_BODY"):
        _named(model, mujoco.mjtObj.mjOBJ_BODY, "LF_palm_lnk")


def test_the_payload_inertia_is_the_cylinder_it_models() -> None:
    """The inertia per kilogram is the object it models, not a round number.

    `claw_payload` scales this by the sampled mass, so a typo here is a load that
    resists rotation by the wrong amount at every mass, in every run, with
    nothing to notice it by.

    The cylinder is written out rather than imported: it was sized from the
    bottle the object row used to carry, and the payload is a training quantity
    that did not change when the row did (`claw.py::PAYLOAD_UNIT_INERTIA`).
    """
    from tasks.jumper.five_foot.claw import PAYLOAD_CYLINDER, PAYLOAD_UNIT_INERTIA

    radius, height = 0.0325, 0.200
    # The object the load is placed as (`add_payload_body`) is the one it rotates as.
    assert PAYLOAD_CYLINDER == pytest.approx((radius, height))
    across = (3.0 * radius**2 + height**2) / 12.0
    about = radius**2 / 2.0
    assert PAYLOAD_UNIT_INERTIA == pytest.approx((across, across, about), abs=5e-7)


def test_pitch_is_charged_symmetrically_in_both_directions() -> None:
    """Pins failure 5, in the form it finally took.

    This term was one-sided twice -- nose-down first, then nose-up after
    measurement reversed it -- and each time the policy simply moved into the half
    that was free: -2.98 degrees mean under the nose-down version, +2.93 under the
    nose-up one. A one-sided penalty on this machine does not remove the lean, it
    chooses its direction.

    So the assertions are symmetric, and the control is that a one-sided
    implementation of either sign fails them.
    """

    def cost(deg: float) -> float:
        env = _StubEnv(_StubAsset(_StubData(projected_gravity_b=_pitched(deg))), None)
        return float(nose_pitch(env)[0])

    # **No deadzone any more, so "free" is a size rather than an exact zero.** The
    # term is `err^2`, kk-rl-lab's shape: 1 degree costs 0.0003 against 20
    # degrees' 0.122, a factor of 400, so small pitch is negligible without a band
    # that has to be tuned. The deadzone version this replaces cost 13x more at
    # the errors a policy learning to walk actually makes, and froze three runs.
    assert cost(1.0) == pytest.approx(math.radians(1.0) ** 2, rel=1e-6)
    assert cost(1.0) == pytest.approx(cost(-1.0), rel=1e-9), "1 degree is not symmetric"
    assert cost(1.0) < cost(20.0) / 100.0, "a degree of pitch is not cheap"
    for deg in (3.0, 5.0, 10.0, 20.0):
        assert cost(deg) > 0.0, f"{deg} degrees nose-down is free"
        assert cost(-deg) > 0.0, f"{deg} degrees nose-up is free"
        assert cost(deg) == pytest.approx(cost(-deg), rel=1e-6), (
            f"{deg} degrees costs differently up and down; the term is one-sided"
        )
    assert cost(20.0) > cost(10.0), "the penalty does not grow with the lean"

    # **Superlinear in the excess**, which is what separates a lean from a lurch.
    #
    # An earlier version of this assertion was `cost(20) > 2 * cost(10)` and
    # **passed against the purely linear term it was meant to catch**: subtracting
    # the threshold makes even a linear cost more than double when the angle
    # doubles (0.144 -> 0.319). What actually distinguishes them is the cost *per
    # radian of excess*, flat at 1.0 for a linear term and rising for this one.
    def per_radian(deg: float) -> float:
        return cost(deg) / math.radians(abs(deg))

    assert per_radian(15.0) > 1.5 * per_radian(5.0), (
        "the cost per radian is flat, so the last degrees are as cheap as the "
        "first -- the term cannot tell a lean from a lurch"
    )
    # And it is the square itself, not a square plus something: a linear term
    # riding along is exactly what made the old shape 13x too expensive in the
    # range where a policy is still learning to take a step.
    assert cost(8.0) == pytest.approx(math.radians(8.0) ** 2, rel=1e-6), (
        "the cost at 8 degrees is not a plain square; something linear is riding "
        "along, which is the shape that froze three runs"
    )


def test_body_height_hold_charges_both_directions(cfg) -> None:
    """Pins the failure `low_stance` could not see: a trunk that stands too *tall*.

    The height moves with the attitude command, and it moves **up**: measured on
    `2026-09-11_11-02-10/model_4500`, a 15 degree nose-down command took the trunk
    from 111 mm to 152, and `Episode_Reward/low_stance` read 0.0000 through it
    because that term is one-sided. A replacement that was also one-sided, or that
    had no deadzone and so charged the 111 mm the machine stands at when nothing
    is wrong, would both look like a fix and be one of the two failures.
    """
    from tasks.jumper.five_foot.mdp.rewards import body_height_hold

    nominal, dead = STAND_Z, 0.015

    def cost(height: float) -> float:
        robot = _StubAsset(_StubData(root_link_pos_w=torch.tensor([[0.0, 0.0, height]])))
        env = _StubEnv(robot, torch.zeros(1, 3))
        return float(body_height_hold(env, nominal_height=nominal, deadzone=dead)[0])

    # Both directions, and symmetric -- this is the whole point of the term.
    assert cost(nominal + 0.040) > 0.0, "standing 40 mm tall is free"
    assert cost(nominal - 0.040) > 0.0, "sinking 40 mm is free"
    # rel 1e-5 rather than the default 1e-6: at V1.6's 0.10704 m nominal the two
    # heights round differently in float32 and the costs land 6e-8 apart.
    assert cost(nominal + 0.040) == pytest.approx(cost(nominal - 0.040), rel=1e-5), (
        "the term is one-sided, which is what left the 41 mm rise uncharged"
    )

    # The deadzone covers the stance a level policy actually held: 111 mm, measured
    # on the previous model, four above V1.6's 107 mm nominal.
    assert cost(nominal) == 0.0
    assert cost(0.111) == 0.0, (
        "the posture the machine holds when nothing is wrong is being charged"
    )
    # `== 0.0` fails here on float rounding alone (7.9e-17), which is the
    # deadzone working, not a leak.
    assert cost(nominal + dead) == pytest.approx(0.0, abs=1e-12), (
        "the deadzone does not reach its own edge"
    )
    assert cost(nominal + dead + 0.001) > 0.0, "the deadzone does not end"

    # Normalised by the nominal, so the weight reads as a fraction of the stance:
    # the excess past the deadzone as a fraction of the nominal height, squared.
    assert cost(0.146) == pytest.approx(((0.146 - nominal - dead) / nominal) ** 2, rel=1e-3)

    # And the weight is small enough that obeying the attitude command still pays.
    # Losing that trade is how `body_roll` at -10 cost 30 points of tracking and
    # how the whole attitude block parked three runs.
    weight = abs(cfg.rewards["body_height_hold"].weight)
    at_measured = weight * cost(0.152)
    assert at_measured < abs(cfg.rewards["track_body_pose"].weight), (
        f"holding height costs {at_measured:.2f} per step at the measured 41 mm "
        f"rise, more than the {cfg.rewards['track_body_pose'].weight} the attitude "
        "itself pays; the robot will stop obeying the command to stay level"
    )


def test_low_stance_charges_sinking_and_not_standing_tall() -> None:
    """One-sided and normalised, so the weight reads as a fraction of the stance.

    A two-sided version would penalise a policy for holding itself up, which is
    the opposite of the failure (a machine on five legs settles *low*: measured,
    it rests 8.5 mm under STAND_Z with zero action).
    """
    def cost(height: float) -> float:
        robot = _StubAsset(_StubData(root_link_pos_w=torch.tensor([[0.0, 0.0, height]])))
        env = _StubEnv(robot, torch.zeros(1, 3))
        return float(low_stance(env, target_height=STAND_Z)[0])

    assert cost(STAND_Z) == 0.0
    assert cost(STAND_Z + 0.02) == 0.0, "standing tall is charged"
    assert cost(0.0) == pytest.approx(1.0), "a body on the ground should cost 1.0"
    assert cost(STAND_Z / 2) == pytest.approx(0.25, abs=1e-6)
    assert cost(STAND_Z - 0.02) < cost(STAND_Z - 0.05)


# ── The whole config, built ───────────────────────────────────────────────

pytest.importorskip("mjlab", reason="building the env config needs mjlab")


@pytest.fixture(scope="module")
def cfg():
    from tasks.registry import load_env_cfg

    return load_env_cfg("jumper.five_foot")


def test_every_per_foot_consumer_names_the_same_five_legs(cfg) -> None:
    """Pins failure 2 in the built config.

    Five places name the feet, and they are wired from one tuple
    (`env_cfg.py::_ground_the_five_legs`) precisely because they cannot be checked
    against each other at runtime -- every combination has matching shapes. The
    fifth, `foot_friction`, has no shape at all to give it away: left at six, the
    carried claw's pad draws a friction sample every episode and nothing reads it.
    """
    expected_geoms = list(foot_geoms_for(FIVE_FOOT_LEGS))

    sensor = next(s for s in cfg.scene.sensors if s.name == "feet_ground_contact")
    assert list(sensor.primary.pattern) == expected_geoms

    scan = next(s for s in cfg.scene.sensors if s.name == "foot_height_scan")
    assert [f.name for f in scan.frame] == list(FIVE_FOOT_LEGS)

    for name in ("foot_clearance", "foot_slip"):
        sites = cfg.rewards[name].params["asset_cfg"].site_names
        assert list(sites) == list(FIVE_FOOT_LEGS), f"{name} scores other feet"

    friction = cfg.events["foot_friction"].params["asset_cfg"].geom_names
    assert list(friction) == expected_geoms, "foot_friction randomises other feet"

    home = cfg.rewards["foot_home"].params
    assert len(home["home_xy"]) == len(FIVE_FOOT_LEGS)
    # Sites, the same ones `foot_clearance` and `foot_slip` take -- and the same
    # ones `HOME_TOE_XY` was measured from, which is what makes comparing a live
    # position against it meaningful. This briefly read the foot *bodies* while
    # `EntityData.site_pose_w` was broken upstream; `mdp/rewards.py::_site_pos_w`
    # reads the sites without it, so a body-based asset_cfg here would be a
    # workaround left behind.
    assert list(home["asset_cfg"].site_names) == list(FIVE_FOOT_LEGS)
    assert home["asset_cfg"].preserve_order, (
        "foot_home pairs sites with home_xy positionally, so the order must be kept"
    )


#: The family's per-task contracts this task is held to, as `(test file, check)`.
#: Each is a plain function of the task id in a framework test that parametrizes
#: over the four locomotion tasks only.
_FAMILY_CONTRACTS = (
    ("test_command_contract.py", "test_the_config_range_is_the_curriculum_s_last_level"),
    ("test_command_contract.py", "test_only_a_rung_that_widens_nothing_may_tighten_the_ruler"),
    ("test_ppo_cfg.py", "test_task_states_its_own_hyperparameters"),
    ("test_ppo_cfg.py", "test_experiment_name_is_the_task_id"),
)


def test_the_config_std_is_the_curriculum_s_last_level() -> None:
    """`play`'s ruler is the one training finished on -- the family's check, with
    this task's ratio.

    Same contract as `test_command_contract.py`'s test of that name, and the same
    failure it pins: `play` builds with `curriculum = {}`, so the config's std is
    what a replay is scored against, and a std that is not the last rung's marks
    every replay on a different ruler than the policy trained on. Nothing looks
    wrong -- it is a reward, so it simply reads a number that is not comparable
    with training's.

    **It is here rather than in `_FAMILY_CONTRACTS` because that test reads the
    levels from the task and the ratio from the family.** Its `_ladder` helper
    returns `(levels, ang_levels, lin_scales, ang_scales)` and lets `lin_std` /
    `ang_std` fall back to `STD_LIN_RATIO` / `STD_ANG_RATIO`, which is right for
    the four tasks it parametrizes over -- they all use the family's ratios. This
    task does not: `command_ang_std_ratio` is 0.15, so that test would expect
    0.5333 where the config and the curriculum both say 0.2, and fail a config
    that is correct. Reading the ratio from the task too is a four-line fix to
    `_ladder`, and it belongs to whoever owns that file; until then the contract
    is checked here, against the task's own parameters, which is what it was
    always trying to assert.
    """
    from tasks.jumper.common.mdp.curriculum import ang_std, lin_std

    cfg = tasks.load_env_cfg("jumper.five_foot")
    params = cfg.curriculum["command"].params
    assert cfg.rewards["track_linear_velocity"].params["std"] == pytest.approx(
        lin_std(-1, params["levels"], params.get("lin_std_scales"),
                params["lin_std_ratio"])
    )
    assert cfg.rewards["track_angular_velocity"].params["std"] == pytest.approx(
        ang_std(-1, params["ang_levels"], params.get("ang_std_scales"),
                params["ang_std_ratio"])
    )
    # The control: the family's default ratio is not this task's, so the check
    # above is not the same assertion written twice.
    assert params["ang_std_ratio"] != pytest.approx(0.4)


@pytest.mark.parametrize(("module", "check"), _FAMILY_CONTRACTS)
def test_the_family_s_contracts_hold_here_too(module: str, check: str) -> None:
    """The command curriculum's and the PPO config's contracts, run on this task.

    The framework's tests parametrize these over `jumper.flat`, `tripod`, `tetrapod`
    and `ripple`. This task used to add itself to both lists, which is an edit to
    two framework test files for one task; it runs the same checks from here
    instead, so the framework's files stay as they are and the coverage stays too.
    Loaded by path, because `tests/` is not a package -- the same way this file
    reaches `tools/` scripts.
    """
    import importlib.util
    from pathlib import Path as _Path

    path = _Path(__file__).resolve().parent / module
    spec = importlib.util.spec_from_file_location(f"_family_{path.stem}", path)
    assert spec and spec.loader
    contracts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contracts)
    getattr(contracts, check)("jumper.five_foot")


def test_the_six_foot_tasks_still_name_six(cfg) -> None:
    """Control for the test above: the skeleton names all six, and this task's
    narrowing is its own -- a change that let it reach the shared config would be
    caught here rather than by a training run on the other four tasks."""
    from tasks.registry import load_env_cfg

    flat = load_env_cfg("jumper.flat")
    scan = next(s for s in flat.scene.sensors if s.name == "foot_height_scan")
    assert [f.name for f in scan.frame] == list(LEGS)
    assert len(flat.rewards["foot_clearance"].params["asset_cfg"].site_names) == 6
    sensor = next(s for s in flat.scene.sensors if s.name == "feet_ground_contact")
    assert len(sensor.primary.pattern) == 6
    assert len(flat.events["foot_friction"].params["asset_cfg"].geom_names) == 6


def test_the_three_per_joint_observations_agree(cfg) -> None:
    """Pins failure 4 in the built config.

    The deploy contract has one `obs_joint_order` and `deploy/fsm/src/obs.rs`
    sizes `joint_pos`, `joint_vel` and `joint_torque` from it. Three terms
    observing three different sets exports without complaint and rebuilds the
    observation at the wrong offsets on the robot.
    """
    for group in cfg.observations.values():
        sets = {}
        for name in ("joint_pos", "joint_vel"):
            term = group.terms.get(name)
            if term is not None:
                sets[name] = tuple(term.params["asset_cfg"].joint_names)
        force = group.terms.get("actuator_force")
        if force is not None:
            sets["actuator_force"] = tuple(
                force.params["asset_cfg"].actuator_names
            )
        assert len(set(sets.values())) <= 1, f"disagreeing joint sets: {sets}"
        for names in sets.values():
            assert names == tuple(OBSERVED_JOINTS)


def test_the_action_drives_sixteen_joints(cfg) -> None:
    """`BaseActionCfg` is a plain dataclass: writing `asset_cfg` instead of
    `actuator_names` is accepted in silence and leaves the action 20 wide."""
    assert list(cfg.actions["joint_pos"].actuator_names) == list(FIVE_FOOT_JOINTS)
    assert set(cfg.actions["joint_pos"].scale) == set(FIVE_FOOT_JOINTS)


def test_the_terms_that_would_score_the_carried_arm_are_rescoped(cfg) -> None:
    """`pose` and `dof_pos_limits` reward and penalise joints the policy drives.

    Left at their defaults they cover all 22, so they would score an arm held at
    a per-episode random offset -- a term with no gradient attached to it, and
    one that moves for reasons the policy cannot influence.
    """
    for name in ("pose", "dof_pos_limits"):
        joints = cfg.rewards[name].params["asset_cfg"].joint_names
        assert list(joints) == list(FIVE_FOOT_JOINTS), f"{name} scores the carried arm"


def test_the_hold_event_pairs_its_ranges_with_its_joints(cfg) -> None:
    """The ranges are positional, so `preserve_order` is load-bearing.

    Without it the joint ids come back in model order and the arm is sampled from
    another joint's box -- inside the limits, plausible, and wrong.
    """
    event = cfg.events["hold_carried_arm"]
    asset_cfg = event.params["asset_cfg"]
    assert asset_cfg.preserve_order, "the ranges would land on the wrong joints"
    assert list(asset_cfg.joint_names) == list(CARRIED_JOINTS)
    ranges = event.params["ranges"]
    assert len(ranges) == len(CARRIED_JOINTS)
    for joint, (low, high) in zip(CARRIED_JOINTS, ranges, strict=True):
        assert low < high, f"{joint} has an empty range"
        if joint in LF_GRASP_BOX:
            assert (low, high) == LF_GRASP_BOX[joint]
    # (low, high) -- and on the finger the *open* end is the lower number, which
    # is the pair this convention is easiest to invert on. Inverting it changes
    # nothing about what gets sampled (`low + (high - low) * u` walks the same
    # interval backwards), so only an assertion catches it.
    assert ranges[-1] == (GRIPPER_OPEN, GRIPPER_CLOSED)
    assert GRIPPER_OPEN < GRIPPER_CLOSED


def test_symmetry_is_off_for_this_task() -> None:
    """The robot is no longer left-right symmetric, so the mirror is not a
    symmetry of it -- it maps a carried claw onto a walking leg.

    The failure this pins is not today's (today the permutations are the wrong
    length and raise) but a later edit that makes the shapes line up, at which
    point the mirror loss teaches the policy that its two front legs are
    interchangeable and nothing says so.
    """
    import tasks as tasks_pkg

    assert tasks_pkg.load_agent_cfg("jumper.five_foot").algorithm.symmetry_cfg is None
    # Control: the six-foot tasks do use it, so this is not asserting a default.
    assert tasks_pkg.load_agent_cfg("jumper.flat").algorithm.symmetry_cfg is not None


# ── The hold, in a running environment ────────────────────────────────────


@pytest.fixture(scope="module")
def stepped_env():
    """A small five-foot environment, stepped with zero actions.

    Module-scoped: building it is the expensive part of this file (about three
    seconds on the native CPU backend), and every test below reads the same run.
    """
    import importlib

    from mjlab.envs import ManagerBasedRlEnv
    from mjrl.backend.select import use_backend

    resolve = importlib.import_module("mjrl.backend.resolve")
    use_backend(resolve.resolve(backend="native", device="cpu", num_envs=8))

    from tasks.registry import load_env_cfg

    cfg = load_env_cfg("jumper.five_foot", play=True)
    cfg.scene.num_envs = 8
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    robot = env.scene["robot"]
    order = list(robot.joint_names)
    at_reset = robot.data.joint_pos[:, [order.index(j) for j in CARRIED_JOINTS]].clone()

    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    for _ in range(100):  # 2 s at 50 Hz
        env.step(action)
    try:
        yield env, order, at_reset
    finally:
        env.close()


def test_the_carried_arm_is_still_where_the_reset_put_it(stepped_env) -> None:
    """Pins failure 1, the one that motivates this whole file.

    mjlab clears `joint_pos_target` on every reset and the action term writes back
    only the joints it drives, so a joint taken out of the action is **commanded
    to 0.0 rad**. Holding the arm therefore needs the reset event to write the
    target, not only the state -- and if it does not, the claw folds to zero over
    the first half second of every episode while the observation, the reward and
    the episode length all look entirely healthy.

    Two seconds of zero action is far longer than that drift takes.
    """
    env, order, at_reset = stepped_env
    robot = env.scene["robot"]
    ids = [order.index(j) for j in CARRIED_JOINTS]
    now = robot.data.joint_pos[:, ids]

    drift = (now - at_reset).abs().max().item()
    # The tolerance tracks the payload, and the numbers are measured rather than
    # guessed (16 envs, 100 steps of zero action, native CPU, seed 0; the table is
    # in `claw.py` beside `PAYLOAD_RANGE`): worst arm drift on V1.6 0.0133 rad with
    # an empty claw, 0.0370 at 0-300 g, 0.0594 at 0-600 g, essentially all of it
    # `LF_shoulder_pitch`. That is PD sag under a load this task deliberately
    # carries, not a hold that has stopped holding.
    #
    # The failure this still catches is an order of magnitude larger: a claw that
    # is not held folds to 0.0 rad, which is 0.9 rad of travel on the finger and
    # 1.6 on the shoulder.
    tolerance = 0.03 if PAYLOAD_RANGE[1] == 0.0 else 0.08
    assert drift < tolerance, (
        f"the carried arm moved {drift:.3f} rad from where the reset put it "
        f"(tolerance {tolerance} at payload {PAYLOAD_RANGE}); the hold is not holding"
    )
    target = robot.data.joint_pos_target[:, ids]
    # Against where the reset put the arm rather than against zero: a hold pose
    # can put a joint's box around 0.0 -- V1.6's first searched pose held the
    # elbow at -0.107 +- 0.10 -- and then a correctly written target sits next to
    # mjlab's cleared 0.0 and a "far from zero" check fails a good hold. The
    # comparison is only a test if the reset put the arm somewhere other than zero,
    # which the shoulder yaw's box, nowhere near it, shows.
    yaw = CARRIED_JOINTS.index("LF_J0_joint")
    assert (at_reset[:, yaw].abs() > 0.1).all(), (
        "the reset left the arm at zero, so the target comparison below is vacuous"
    )
    assert torch.allclose(target, at_reset, atol=1e-4), (
        "a carried joint's PD target is not where the reset put the arm -- mjlab "
        "clears it to 0.0 and the hold event is not writing it"
    )


def test_an_undriven_joint_really_does_fall_to_zero(stepped_env) -> None:
    """Control for the test above.

    Without this, "the arm did not move" could just mean nothing moves anything.
    `RF_J4_joint` is the same situation minus the hold -- out of the action,
    out of the observation, and nothing writing its target -- and it must be
    found at 0.0 rather than at its HOME angle, which is what proves the drift
    the test above rules out is real and reachable.
    """
    env, order, _ = stepped_env
    robot = env.scene["robot"]
    idx = order.index("RF_J4_joint")
    assert float(robot.data.joint_pos_target[:, idx].abs().max()) == 0.0
    assert abs(float(robot.data.joint_pos[:, idx].mean())) < 0.02, (
        "an unheld joint stayed at its home angle, so this environment does not "
        "exhibit the drift the hold exists to prevent and the test above is vacuous"
    )


def test_the_hold_differs_between_environments(stepped_env) -> None:
    """The hold is re-sampled per episode; a constant would be a silent
    regression to a single arm angle the policy can overfit to.

    **This fixture builds `play=True`, where the claw is the exception**: the
    aperture is not sampled there, it starts wide open, so the finger is asserted
    to have no spread at all. The claim that training *does* sample it across the
    whole working range is
    `test_the_aperture_is_sampled_in_training_and_pinned_open_in_replay`.
    """
    _env, _order, at_reset = stepped_env
    spread = at_reset.max(dim=0).values - at_reset.min(dim=0).values
    assert (spread[: len(ARM_JOINTS)] > 0.0).all(), "the arm's hold is not random"
    assert (spread[: len(ARM_JOINTS)] <= 2 * GRASP_BOX + 1e-6).all(), (
        "an arm joint left its sampling box"
    )
    finger = at_reset[:, -1]
    assert torch.allclose(finger, torch.full_like(finger, GRIPPER_OPEN), atol=1e-5), (
        f"replay did not start the claw wide open: {finger.tolist()}"
    )


def test_the_five_foot_stance_holds_up_under_zero_action(stepped_env) -> None:
    """A sanity floor on the whole thing: with the arm carried and no policy at
    all, the robot still stands.

    Measured here, 8 environments, native CPU backend, 2 s of zero action: it
    settles about 8 mm under STAND_Z and pitches about 1.6 degrees nose-down,
    because the corner the carried leg used to hold is now unsupported. Both are
    small, and both are what `low_stance` and `nose_down_pitch` are sized against
    -- if this ever fails, those two weights are being asked to fix a stance
    problem instead of a policy one.
    """
    env, _order, _ = stepped_env
    robot = env.scene["robot"]
    height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    assert float(height.min()) > ff_env_cfg.MIN_STAND_HEIGHT, "the robot collapsed"
    assert float(height.mean()) == pytest.approx(STAND_Z, abs=0.02)

    pitch = torch.asin(robot.data.projected_gravity_b[:, 0].clamp(-1.0, 1.0))
    assert float(pitch.abs().max()) < math.radians(10.0), (
        "the unloaded five-foot stance tips more than the reward terms are sized for"
    )


def test_the_duty_ema_is_reset_per_episode(stepped_env) -> None:
    """`duty_balance` keeps state, so it has to be in the manager's reset path.

    The reward manager calls `reset(env_ids=...)` only on terms it classified as
    class-based, and only per environment. A term that kept its exponential
    moving average across an episode boundary would score a fresh episode against
    the previous one's gait for the first couple of seconds -- a bias that is
    invisible in every aggregate, because the value stays in range the whole time.

    Placed last in the file on purpose: it mutates that state deliberately.
    """
    env, _order, _ = stepped_env
    manager = env.reward_manager
    term = manager.get_term_cfg("duty_balance").func

    assert manager.get_term_cfg("duty_balance") in manager._class_term_cfgs, (
        "duty_balance was not recognised as a class-based term, so its reset() "
        "is never called and its EMA carries across episodes"
    )
    assert not torch.allclose(term.duty, torch.full_like(term.duty, 0.5)), (
        "the duty EMA never moved off its seed, so this test cannot tell a reset "
        "from a term that does nothing"
    )

    before = term.duty.clone()
    manager.reset(torch.tensor([0]))
    assert torch.allclose(term.duty[0], torch.full_like(term.duty[0], 0.5))
    assert torch.allclose(term.duty[1:], before[1:]), (
        "resetting one environment reset the others' duty as well"
    )


def test_the_aperture_is_sampled_in_training_and_pinned_open_in_replay(cfg) -> None:
    """Both halves matter, and each one fails silently on its own.

    Pinning the aperture in **training** would leave a constant column in
    `joint_pos` -- what the note in `common/mdp/observations.py::actuator_force`
    warns against -- and, now that `[` and `]` exist, would train a policy that
    has never walked with a closed claw and then hand an operator the key that
    closes it: the teleop would be driving the policy straight off its own
    distribution, and nothing about that reads as an error.

    Sampling it in **replay** is the milder failure and still a real one: a
    demonstration that starts with the jaws half shut for no visible reason reads
    as a fault, and the operator's first press then starts from the middle of the
    travel rather than from an end.

    The last assertion is the control: it is what stops "replay pins the finger"
    from being satisfied by a replay config that pinned everything.
    """
    from tasks.registry import load_env_cfg

    train = cfg.events["hold_carried_arm"].params["ranges"]
    play = load_env_cfg("jumper.five_foot", play=True).events["hold_carried_arm"].params[
        "ranges"
    ]

    assert train[-1] == (GRIPPER_OPEN, GRIPPER_CLOSED), (
        "training stopped sampling the claw across its working range"
    )
    assert play[-1] == (GRIPPER_OPEN, GRIPPER_OPEN), (
        "replay does not start the claw wide open"
    )
    assert train[:-1] == play[:-1], (
        "replay changed an arm joint's sampling box as well; only the claw differs"
    )


# ── The claw, driven from its trigger ─────────────────────────────────────
#
# Pins failure 6. `mdp/gripper.py` is replay-only and moves exactly one joint
# that nothing else in the step loop touches, so every way it can be wrong is a
# claw that quietly does the opposite of what the operator asked, or nothing.


CLAW_JID = 3  # the finger's column in the stubs below; the arm's four are 0..2, 4


class _StubClaw:
    """The one asset `GripperTeleop` touches, recording what it is told to write."""

    def __init__(self, resting: float, num_envs: int = 4, num_joints: int = 5) -> None:
        self.num_joints = num_joints
        # 9.0 everywhere else, so a term that read the wrong column would produce
        # an obviously wrong number rather than a plausible one.
        pos = torch.full((num_envs, num_joints), 9.0)
        pos[:, CLAW_JID] = resting
        self.data = _StubData(
            joint_pos=pos,
            # Still: the squeeze limit's speed term adds nothing (`claw.py::
            # LEAD_PER_SPEED`), so these tests read the lead alone.
            joint_vel=torch.zeros(num_envs, num_joints),
            joint_pos_target=torch.zeros(num_envs, num_joints),
        )
        self.writes: list[tuple[torch.Tensor, object]] = []

    def set_joint_position_target(self, position, joint_ids=None, env_ids=None):
        del env_ids
        self.writes.append((position.clone(), joint_ids))
        self.data.joint_pos_target[:, joint_ids] = position


class _StubOperator:
    """The operator as the claw sees it: how far the trigger is squeezed, or
    `None` while nobody has taken over. Records which control it was asked for."""

    def __init__(self) -> None:
        self.squeeze: float | None = None
        self.asked: list[str] = []

    def task_control(self, name: str, stamp=None) -> float | None:
        del stamp
        self.asked.append(name)
        return self.squeeze


class _StubClawEnv:
    def __init__(self, robot: _StubClaw, num_envs: int) -> None:
        self.scene = _StubScene(robot, None)
        self.num_envs = num_envs
        self.device = "cpu"
        self.common_step_counter = 0
        # Mid-episode by default: the term reads this to notice a reset, and a
        # stub that always looked freshly reset would hide the hold's failures
        # behind "a reset forgets a hold".
        self.episode_length_buf = torch.full((num_envs,), 50)

    def reset(self) -> None:
        self.episode_length_buf = torch.zeros_like(self.episode_length_buf)


@pytest.fixture
def make_claw():
    """Build a `GripperTeleop` on a stub, with a stub operator registered as the
    environment's -- which is how the term finds the real one -- and take it off
    the registry afterwards."""
    from tasks.jumper.common.mdp import operator as operator_module

    made = []

    def _make(resting: float = -0.5, joint_ids=(CLAW_JID,), params=None, **kw):
        from tasks.jumper.five_foot.mdp.gripper import GripperTeleop, gripper_teleop_event

        term_cfg = gripper_teleop_event(**kw)
        # What `ManagerBase._resolve_common_term_cfg` would have done against a
        # real scene, done here by hand.
        term_cfg.params["asset_cfg"].joint_ids = list(joint_ids)
        # `params` edits the built config in place, which is how a caller would
        # actually reach the aperture ends -- `gripper_teleop_event` does not take
        # them, because they are a measured property of this claw.
        term_cfg.params.update(params or {})
        robot = _StubClaw(resting)
        env = _StubClawEnv(robot, 4)
        term = GripperTeleop(cfg=term_cfg, env=env)
        term.env = env  # the tests drive the term directly; this is what they pass
        term.op = _StubOperator()
        operator_module._OPERATORS[env] = term.op
        made.append(env)
        return term, robot

    yield _make

    for env in made:
        operator_module._OPERATORS.pop(env, None)


def _sent(robot: _StubClaw) -> float:
    """The target the last write sent, one number: every environment gets the same."""
    return float(robot.writes[-1][0].max())


def test_the_claw_teleop_is_replay_only(cfg) -> None:
    """A term that can be driven from outside the process makes a training run
    unreproducible, and during training there is no viewer or pad for an input
    to arrive from anyway. Both halves are asserted, so neither can pass
    vacuously."""
    from tasks.registry import load_env_cfg

    assert "gripper_teleop" not in cfg.events, (
        "the operator can reach the claw during training"
    )
    assert "gripper_teleop" in load_env_cfg("jumper.five_foot", play=True).events, (
        "replay has no way to move the claw"
    )


def test_the_trigger_is_the_aperture(make_claw) -> None:
    """Let go is open, all the way is shut, halfway is halfway -- and it is the
    control at the claw's side that is asked, `claw_left`, the one the robot's
    `deploy/lib.rs` answers for this claw too. The finger rests shut, so the
    squeeze limit cannot bind and what is measured is the mapping."""
    term, robot = make_claw(resting=GRIPPER_CLOSED)
    middle = 0.5 * (GRIPPER_OPEN + GRIPPER_CLOSED)
    for squeeze, want in [(0.0, GRIPPER_OPEN), (0.5, middle), (1.0, GRIPPER_CLOSED)]:
        term.op.squeeze = squeeze
        term(term.env)
        assert _sent(robot) == pytest.approx(want), f"squeezed {squeeze}: {_sent(robot)}"
    assert set(term.op.asked) == {"claw_left"}, (
        f"the claw asked the operator for {sorted(set(term.op.asked))}, not the control at its side"
    )


def test_a_claw_held_shut_stays_shut_until_the_trigger_takes_it(make_claw) -> None:
    """`play --hold` shuts the claw on a prop, and a trigger at rest must not open it.

    The teleop writes what the trigger asks for every step once the operator has
    taken over, and a trigger at rest asks for open -- the opposite of what
    `--hold` needs a moment later. Untold, the step after the hold opens the
    jaws, and the prop the episode started with falls out with nothing raised.
    So the hold stands -- untouched, and through a trigger squeezed part way --
    until the trigger is squeezed as far, and the claw follows the trigger from
    there: let go, and it opens. A reset forgets it.

    The control is the precondition: before `hold_at` the same trigger at rest
    writes open, so the shut target after it is the hold's doing.
    """
    term, robot = make_claw(resting=GRIPPER_CLOSED)
    term.op.squeeze = 0.0
    term(term.env)
    assert _sent(robot) == pytest.approx(GRIPPER_OPEN), "the control group: at rest is open"

    term.hold_at(GRIPPER_CLOSED)
    for squeeze in (None, 0.0, 0.5, 0.9):
        term.op.squeeze = squeeze
        term(term.env)
        assert _sent(robot) == pytest.approx(GRIPPER_CLOSED), f"squeezed {squeeze}, the hold opened"

    term.op.squeeze = 1.0
    term(term.env)
    term.op.squeeze = 0.0
    term(term.env)
    assert _sent(robot) == pytest.approx(GRIPPER_OPEN), "the trigger never took the claw back"

    term.hold_at(GRIPPER_CLOSED)
    term.env.reset()
    term(term.env)
    assert _sent(robot) == pytest.approx(GRIPPER_OPEN), "a reset kept the hold"


def test_a_finger_stopped_by_an_object_squeezes_no_harder_than_the_lead(make_claw) -> None:
    """Pins the squeeze limit, whose absence costs the gait rather than the grip.

    The policy observes the finger's torque, and a finger stopped by an object and
    pressed on to `GRIPPER_CLOSED` saturates its servo far outside anything it was
    trained on (`claw.py::SQUEEZE_LEAD`). So the teleop sends a target no more than
    `SQUEEZE_LEAD` past where the finger is -- and nothing raises when it does not:
    the claw simply holds harder and the robot turns worse.

    The stub's finger stays where an object stopped it while the trigger is
    squeezed shut. The control is the opposite direction: a target below the
    finger, an opening, has to go out untouched, or the limit is a claw that
    cannot let go.
    """
    from tasks.jumper.common.actuator import CURVE
    from tasks.jumper.common.constants import STIFFNESS
    from tasks.jumper.five_foot.claw import SQUEEZE_LEAD

    assert STIFFNESS * SQUEEZE_LEAD < CURVE.continuous_torque, (
        "the limit allows more torque than the servo holds anyway, so it limits nothing"
    )
    stall = -0.55
    term, robot = make_claw(resting=stall)
    term.op.squeeze = 1.0
    term(term.env)
    assert term._target == GRIPPER_CLOSED, "the trigger was never squeezed shut; nothing is tested"
    assert _sent(robot) == pytest.approx(stall + SQUEEZE_LEAD), (
        f"a finger stopped at {stall} was sent {_sent(robot):+.3f}, not "
        f"{stall + SQUEEZE_LEAD:+.3f}: the squeeze is not limited"
    )

    term.op.squeeze = 0.0
    term(term.env)
    assert _sent(robot) == pytest.approx(GRIPPER_OPEN), "the limit held back an opening too"


def test_open_is_the_more_negative_direction(make_claw) -> None:
    """The sign trap, and it is the same one `hold_carried_arm` guards its ranges
    against: on this joint **more open is the lower number**. Swap the two and a
    squeeze opens the claw and letting go shuts it, with nothing to report -- the
    claw still moves, still stays in range, still stops at an end."""
    term, _ = make_claw(resting=GRIPPER_CLOSED)
    term.op.squeeze = 0.0
    term(term.env)
    released = term._target
    term.op.squeeze = 0.3
    term(term.env)
    squeezed = term._target

    assert released == GRIPPER_OPEN and released < 0.0, (
        f"a trigger at rest put the finger at {released}, which is not open"
    )
    assert squeezed > released, "squeezing the trigger did not close the claw"


def test_the_aperture_is_clamped_to_the_working_range(make_claw) -> None:
    """Both ends are a place where squeezing further stops meaning what it says.

    Opening past `GRIPPER_OPEN` soon stops closing across the mouth -- the
    nearest part of the moving jaw becomes an edge rather than the grip face and
    the closing axis swings with it (`claw.py`'s sweep) -- so that end is short
    of the turning point rather than at the joint's limit.

    The closing end is `GRIPPER_CLOSED`, which on the corrected measurement is
    the joint's own upper limit and the angle at which the mouth reaches zero.
    The squeeze on an object comes from the error between that target and where
    the object stopped the finger, which is why a full squeeze has to land on it
    exactly rather than a rounding error short: shut is a value something
    downstream compares with one day. A trigger reporting past its travel lands
    on the same ends."""
    term, _ = make_claw(resting=GRIPPER_CLOSED)
    for squeeze, end in [(-0.4, GRIPPER_OPEN), (0.0, GRIPPER_OPEN),
                         (1.0, GRIPPER_CLOSED), (1.6, GRIPPER_CLOSED)]:
        term.op.squeeze = squeeze
        term(term.env)
        assert term._target == end, f"squeezed {squeeze}: {term._target!r}, not {end}"


def test_nothing_is_written_until_the_operator_takes_over(make_claw) -> None:
    """An untouched replay must be the replay it was before this term existed:
    `hold_carried_arm` puts the claw somewhere and it stays there. A term that
    wrote its own idea of the target every step would erase that and look
    completely healthy doing it.

    The take-over at the end is the control group -- without it this test passes
    just as well against a term that never writes at all.
    """
    term, robot = make_claw(resting=-0.5)

    for _ in range(5):
        term(term.env)
    assert robot.writes == [], "the claw was driven before anyone touched a control"

    term.op.squeeze = 0.0  # taken over, the trigger at rest
    term(term.env)
    assert len(robot.writes) == 1, "the operator took over and nothing was written"


def test_the_claw_is_on_the_operator_s_trigger_and_nothing_else(make_claw) -> None:
    """The claw's one input is the control the task's file declares for it,
    read through the operator -- so the keyboard reaches it as the file binds
    it, Space closing the claw for as long as it is held, and through nothing
    of its own.

    It had keys of its own once, `[` `]` `G` beside the operator's: a second
    mapping of one person's input, on keys MuJoCo's viewer binds to its cameras,
    with no counterpart on the robot. Building the term must add no key handler.
    The control group is the operator's own keys, which do reach the claw.
    """
    from mjrl.viewer import keys

    import tasks
    from tasks.jumper.common.mdp.operator import Operator, spans

    before = keys.handler_count()
    make_claw()
    assert keys.handler_count() == before, "the claw registered keys of its own"

    cfg = tasks.load_env_cfg("jumper.five_foot", play=True)
    controls = cfg.commands["twist"].controls
    clock = _Clock()
    op = Operator(controls, listen=False, clock=clock)
    for term in ("twist", "body_pose"):
        op.attach(term, spans(controls.command(term), cfg.commands[term]))
    try:
        assert op.task_control("claw_left") is None, "untouched, the operator has not taken over"
        op.key("key_space", True)
        clock.t = 0.3 * controls.full_after_s
        assert op.task_control("claw_left") == pytest.approx(0.3), "Space held did not close the claw"
        assert op.task_control("arm_thumb_up") == 0.0
        op.key("key_space", False)
        assert op.task_control("claw_left") == 0.0, "Space let up and the claw stayed shut"
        with pytest.raises(KeyError, match="keeps"):
            op.task_control("LT")
    finally:
        op.close()


def test_an_inverted_open_closed_pair_is_refused(make_claw) -> None:
    """`(open, closed)` and not `(low, high)`, because on this joint they are not
    the same ordering. Swapped, the clamp still clamps and the trigger still
    moves the target -- across a two-tenths-of-a-radian interval on the wrong
    side of the travel, which is silent."""
    with pytest.raises(ValueError, match="open_at < closed_at"):
        make_claw(params={"open_at": GRIPPER_CLOSED, "closed_at": GRIPPER_OPEN})


def test_it_refuses_to_drive_more_than_one_joint(make_claw) -> None:
    """`joint_ids` comes from a name lookup, and a pattern that matched both
    fingers would drive the RF claw as well -- which in FOOT mode is a foot."""
    with pytest.raises(ValueError, match="exactly one joint"):
        make_claw(joint_ids=(CLAW_JID, 0))


# ── The claw, in a running environment ────────────────────────────────────


@pytest.fixture(scope="module")
def claw_driven_env():
    """A small five-foot environment with the claw squeezed shut from the keys.

    Its own environment rather than a share of `stepped_env`: this one is driven,
    and the tests above it read a robot that was left alone.
    """
    import importlib

    from mjlab.envs import ManagerBasedRlEnv
    from mjrl.backend.select import use_backend
    from mjrl.viewer import keys

    resolve = importlib.import_module("mjrl.backend.resolve")
    use_backend(resolve.resolve(backend="native", device="cpu", num_envs=2))

    from tasks.jumper.common.mdp.operator import Operator
    from tasks.registry import load_env_cfg

    cfg = load_env_cfg("jumper.five_foot", play=True)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()

    term = env.event_manager.get_term_cfg("gripper_teleop").func
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)

    env.step(action)  # one untouched step, so the claw is where the reset left it
    order = list(env.scene["robot"].joint_names)
    before = env.scene["robot"].data.joint_pos[:, [order.index(j) for j in ARM_JOINTS]]
    before = before.clone()

    # `claw_left`, from the keyboard the way the viewer delivers a held key:
    # every handler sees Space go down, and the operator closes the claw until
    # it comes up -- which here it does not. On this test's clock
    # rather than the machine's, so the hold is full travel and then two seconds
    # more however fast the steps run.
    operator = Operator.of_env(env)
    clock = _Clock()
    operator._clock = clock
    from controller import vocabulary

    squeeze = vocabulary.key_codes()["key_space"]["glfw"]
    keys.dispatch(squeeze, True)
    clock.t = operator.controls.full_after_s
    for _ in range(100):  # 2 s at 50 Hz, the key still held
        clock.t += 0.02
        env.step(action)

    try:
        yield env, order, term, before
    finally:
        operator.close()
        env.close()


def test_the_operator_actually_closes_the_claw(claw_driven_env) -> None:
    """The whole point, and the only test here that can see the two things the
    stubs cannot: that `joint_ids` resolved to the finger's real column, and that
    writing the target from a `mode="step"` event is not undone by the action
    term writing its own sixteen every substep."""
    env, order, term, _ = claw_driven_env
    finger = env.scene["robot"].data.joint_pos[:, order.index(FINGER_JOINT)]

    assert term._target == pytest.approx(GRIPPER_CLOSED)
    assert finger.max().item() == pytest.approx(GRIPPER_CLOSED, abs=0.01), (
        f"the claw was commanded to {GRIPPER_CLOSED} and sits at {finger.tolist()}"
    )


def test_driving_the_claw_leaves_the_carried_arm_alone(claw_driven_env) -> None:
    """Control group for the test above, and the failure it pins is a wrong column:
    the finger and the four arm joints are written by the same call in
    `hold_carried_arm`, and a `joint_ids` off by one would move the wrist instead
    with the claw simply staying where the reset left it -- which looks exactly
    like a term that has not been wired up yet."""
    env, order, _, before = claw_driven_env
    now = env.scene["robot"].data.joint_pos[:, [order.index(j) for j in ARM_JOINTS]]

    drift = (now - before).abs().max().item()
    # Same payload-dependent tolerance as the hold test above, and for the same
    # measured reason: a full claw sags the shoulder by up to 0.056 rad under PD,
    # which has nothing to do with the finger.
    tolerance = 0.03 if PAYLOAD_RANGE[1] == 0.0 else 0.08
    assert drift < tolerance, (
        f"an arm joint moved {drift:.3f} rad while only the claw was commanded"
    )


def test_every_support_shaping_term_is_gated_on_the_command(cfg) -> None:
    """Three terms shape the support pattern -- `five_foot_gait` decides which
    legs are down, `lateral_load` and `duty_balance` shape how the load is shared
    -- and together they are now worth as much as velocity tracking.

    **The gate is what makes that safe, and it is the thing to assert.** All three
    are multiplied by `moving_gate`, so they are exactly zero for a standing
    command and cannot deepen the stand-still basin this robot's whole reward
    design is arranged against. Lose the gate on any one of them and a stationary
    robot starts collecting a support-quality reward it gets for free, which is
    the failure two cold starts in this task's history ran into.
    """
    # **The gait term must exist, at any weight.** Its weight is a tuning knob
    # that has been 1.0, 0.5 and 0.0 across this task's runs; what must not happen
    # is the term being *deleted*, because at weight 0 it still runs and
    # `Episode_Reward/five_foot_gait` is then the only readout of whether the 3+2
    # grouping is holding. Nothing else in the task measures it, and the two runs
    # that lost the grouping (alternating 93.8% -> 71.5% resumed, 27.2% cold) were
    # only diagnosable because the term was still being computed.
    assert "five_foot_gait" in cfg.rewards, (
        "the gait term was removed rather than zeroed; the grouping now has no "
        "readout at all"
    )
    total = sum(
        cfg.rewards[n].weight
        for n in ("lateral_load", "duty_balance", "feet_planted")
        if n in cfg.rewards
    )
    tracking = (
        cfg.rewards["track_linear_velocity"].weight
        + cfg.rewards["track_angular_velocity"].weight
    )
    assert total >= 2.0, (
        f"the support pattern is held by {total} against {tracking} of tracking, "
        f"with the gait term off"
    )
    # No assertion that this stays under tracking: it deliberately does not any
    # more (6.0 of balance against 4.0 of tracking, plus the gait prior's 1.0).
    # The gate above is what keeps that safe; whether it is *wise* is what a run
    # answers, and `env_cfg.py` records the arithmetic.
    for name in ("lateral_load", "duty_balance", "five_foot_gait"):
        if name not in cfg.rewards:
            continue
        params = cfg.rewards[name].params
        assert params.get("command_name") == "twist", (
            f"{name} lost its command gate, so a standing robot now collects it"
        )


def test_the_left_claw_closes_on_something_in_its_mouth() -> None:
    """Pins the failure that looks like nothing at all: a claw that cannot pinch.

    **MuJoCo collides a mesh by its convex hull, and the hull of a hooked jaw is
    filled in across the mouth.** With one hull per jaw link the middle of the
    left claw's mouth was *inside* the fixed jaw over the whole of the grip face,
    so anything driven into the mouth met an invisible wall about 20 mm outside
    the drawn jaw, and the moving jaw had nothing to push it into. Nothing
    raises: the finger reaches its target, the readout counts down to zero
    millimetres, and the object is shoved around by a jaw that is visibly not
    touching it. `tasks/jumper/five_foot/tools/jaw_decompose.py` cuts the jaws and
    `tasks/jumper/five_foot/tools/claw_mouth.py` measures what is left.

    The ball is parked just off the **anvil** -- the point on the fixed jaw the
    moving jaw arrives at, which is where an object in this claw ends up
    (`mdp/grasp.py::_mouth_in_palm`) -- with 2 mm to spare, so that wide open it
    is touching neither jaw and the control below is about the mouth rather than
    about where the ball was put. It is placed in the *palm's* frame, so it stays
    where it is while the finger moves.

    Both halves are needed. Open, the ball must be free, or the test would pass
    against collision geometry that swallowed the whole mouth; closing, it must
    be caught, which is the thing that was broken.
    """
    import mujoco
    import numpy as np

    from tasks.jumper.five_foot.jaws import JAW_COLLISION, task_spec
    from tasks.jumper.five_foot.mdp.grasp import SEAT_CLEARANCE, _jaw_clouds, _mouth_in_palm

    radius = 0.004

    spec = task_spec()
    JAW_COLLISION.edit_spec(spec)
    ball = spec.worldbody.add_body(name="ball")
    ball.add_freejoint()
    # A sphere, so the test says nothing about which way the mouth faces. Its
    # contype/conaffinity have to reach the leg's channels, which `_collision_cfg`
    # gives every claw geom: bit 0 is the ground channel every leg carries.
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius, 0.0, 0.0],
                  mass=0.01, name="ball", contype=1, conaffinity=1)
    model = spec.compile()
    data = mujoco.MjData(model)

    def ident(objtype, name):
        return mujoco.mj_name2id(model, objtype, name)

    claw = {
        g for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(
            ("LF_palm_link_meshcol", "LF_finger_link_meshcol",
             "LF_finger_tip_link_meshcol", "LF_palm_pad_b_link_meshcol"))
    }
    assert len(claw) > 8, (
        f"the left claw has {len(claw)} collision geoms; one hull per jaw link "
        "fills in the mouth and it cannot pinch anything"
    )
    ball_g = ident(mujoco.mjtObj.mjOBJ_GEOM, "ball")
    ball_q = model.jnt_qposadr[model.body_jntadr[ident(mujoco.mjtObj.mjOBJ_BODY, "ball")]]
    palm = ident(mujoco.mjtObj.mjOBJ_BODY, "LF_palm_link")
    anvil = _mouth_in_palm(model)

    def pose(angle: float) -> None:
        data.qpos[:] = model.qpos0
        for name, value in LF_GRASP.items():
            data.qpos[model.jnt_qposadr[ident(mujoco.mjtObj.mjOBJ_JOINT, name)]] = value
        data.qpos[model.jnt_qposadr[ident(mujoco.mjtObj.mjOBJ_JOINT, FINGER_JOINT)]] = angle
        data.qpos[ball_q:ball_q + 3] = (10.0, 10.0, 10.0)   # out of the way
        data.qpos[ball_q + 3:ball_q + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

    # Which way to step off the fixed jaw: toward the moving one, measured wide
    # open where the two are far enough apart for the direction to mean anything.
    pose(GRIPPER_OPEN)
    rot = data.xmat[palm].reshape(3, 3)
    seat = data.xpos[palm] + rot @ anvil
    moving, _ = _jaw_clouds(model, data)
    toward = moving[np.argmin(np.linalg.norm(moving - seat, axis=1))] - seat
    toward /= np.linalg.norm(toward)
    # In the palm's frame, so the ball does not move when the finger does. The
    # clearance off the fixed jaw is `SEAT_CLEARANCE`: the anvil is a vertex of the
    # *visual* mesh and on V1.6 the collision piece covering it sits proud enough
    # that a ball 2 or 3 mm off it starts out already in contact.
    park = anvil + rot.T @ (toward * (radius + SEAT_CLEARANCE))

    def caught(angle: float) -> int:
        pose(angle)
        rot = data.xmat[palm].reshape(3, 3)
        data.qpos[ball_q:ball_q + 3] = data.xpos[palm] + rot @ park
        mujoco.mj_forward(model, data)
        return sum(
            1 for c in data.contact[: data.ncon]
            if (c.geom1 in claw and c.geom2 == ball_g)
            or (c.geom2 in claw and c.geom1 == ball_g)
        )

    # The ball is 8 mm across and sits `SEAT_CLEARANCE` off the fixed jaw, so its far
    # surface is 13 mm out and the moving jaw reaches it at about that mouth:
    # measured on V1.6.1, the jaw is 7.3 mm short at -0.10, 2.4 mm short at -0.05
    # and 2.4 mm into the ball at GRIPPER_CLOSED.
    #
    # **Asserted at `GRIPPER_CLOSED` rather than at a number**, which is the repair
    # this needed: it said -0.05, measured when the jaws met at finger 0.0, and
    # V1.6.1's bigger claw still has 14.6 mm of mouth there. A hardcoded angle on
    # the closing side means "nearly shut" only for one revision of the part.
    assert caught(GRIPPER_CLOSED) > 0, (
        "the claw is closing on an 8 mm ball sitting in its mouth and the "
        "solver reports no contact; the jaws are going through it"
    )
    assert caught(GRIPPER_OPEN) == 0, (
        "the ball is already in contact with the jaws wide open, so the "
        "assertion above is not about closing on it"
    )



def test_a_seated_prop_rests_on_the_anvil_whatever_its_shape() -> None:
    """`play --hold` and the end-to-end check put a prop in the claw; this is where.

    The failure is quiet in the worst way: a prop parked a few centimetres off the
    anvil is not between the jaws when they close, and a grasp check reports it
    as a claw that cannot hold that object. So every solid in the row is seated
    by `_seat_position` against the real anvil and closing direction, and its
    nearest surface has to come out `SEAT_CLEARANCE` from the anvil with its base
    above the floor.

    The control is the rule this replaced -- offset by the prop's extent along
    the closing direction -- which has to leave the notebook measurably off
    `SEAT_CLEARANCE`. Where the claw closes tilted that rule rests a thin, tall
    object on a far corner; without the control a seat that is merely "somewhere
    near" would pass.
    """
    import mujoco
    import torch

    from tasks.jumper.common.constants import HOME, STAND_Z
    from tasks.jumper.five_foot.jaws import task_spec
    from tasks.jumper.five_foot.mdp.grasp import (
        SEAT_CLEARANCE,
        _closing_in_palm,
        _geom_shape,
        _mouth_in_palm,
        _seat_position,
        _support,
        _surface_distance,
    )
    from tasks.jumper.five_foot.objects import SOLIDS, _solid_spec

    model = task_spec().compile()
    data = mujoco.MjData(model)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
    data.qpos[model.jnt_qposadr[base] + 2] = STAND_Z
    for joint, value in {**HOME, **LF_GRASP}.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LF_palm_link")
    rot = data.xmat[palm].reshape(3, 3)
    anvil = torch.tensor(data.xpos[palm] + rot @ _mouth_in_palm(model),
                         dtype=torch.float64).unsqueeze(0)
    closing = torch.tensor(rot @ _closing_in_palm(model), dtype=torch.float64).unsqueeze(0)
    floor = torch.zeros(1, dtype=torch.float64)

    def shapes_of(solid):
        prop = _solid_spec(solid).compile()
        return [_geom_shape(prop, g, "cpu", torch.float64) for g in range(prop.ngeom)]

    def gap(centre, shapes) -> float:
        return min(
            float(_surface_distance((anvil - centre - pos) @ rot_g, gtype, size)[0])
            for gtype, size, pos, rot_g in shapes
        )

    for solid in SOLIDS:
        shapes = shapes_of(solid)
        centre = _seat_position(anvil, closing, floor, shapes)
        assert gap(centre, shapes) == pytest.approx(SEAT_CLEARANCE, abs=2e-4), (
            f"the {solid.name} is seated {gap(centre, shapes) * 1000:.1f} mm from the "
            f"anvil, not {SEAT_CLEARANCE * 1000:.0f}"
        )
        assert float(centre[0, 2]) + solid.bounds[0][2] >= 0.0, (
            f"the seated {solid.name} has its base through the floor"
        )

    notebook = next(s for s in SOLIDS if s.name == "notebook")
    shapes = shapes_of(notebook)
    gtype, size, pos, rot_g = shapes[0]
    extent = float(_support(closing @ rot_g, gtype, size)[0] + (closing @ pos)[0])
    old = anvil + closing * (extent + SEAT_CLEARANCE)
    old[0, 2] = max(float(old[0, 2]), notebook.half_extents[2] + SEAT_CLEARANCE)
    # On the previous model's hold pose the claw closed tilted and the old rule put
    # the notebook 32.7 mm off the anvil. V1.6's `LF_GRASP` closes it almost level
    # and the old rule's error shrinks with the tilt -- soap 9.2 mm, notebook 12.9,
    # can 6.0, against 5.0 -- but the notebook's 7.9 mm is still forty times the
    # 0.2 mm the seat above is held to, which is what makes this a control.
    assert gap(old, shapes) - SEAT_CLEARANCE > 0.002, (
        "seating the notebook by its extent along the closing direction now lands "
        "within 2 mm of the real seat, so this test can no longer tell the two "
        "rules apart"
    )


def test_the_claw_collision_has_the_mouth_the_part_has() -> None:
    """The jaws collide as they are drawn, along the whole mouth.

    The control the test above cannot be: it asks whether the claw closes on one
    8 mm ball at one place, and a mouth that is filled in everywhere *except*
    there would pass it. This walks the length of the mouth and compares the room
    a point in the middle of it has against the collision shapes with the room it
    has against the drawn part. They came apart by 15 mm when `LF_palm_link` was
    a single hull, and by 2.4 mm now.

    `tasks/jumper/five_foot/tools/claw_mouth.py` is the same measurement with a table and a
    `--palm-pieces` switch for comparing one cut of the jaw against another.
    """
    import importlib.util
    from pathlib import Path as _Path

    # The task's `tools/` is not a package -- it holds standalone scripts -- so it
    # loads by file path, the same way `test_hull_tool.py` reaches
    # `tools/hull_collision.py`.
    path = (_Path(__file__).resolve().parents[1]
            / "tasks" / "jumper" / "five_foot" / "tools" / "claw_mouth.py")
    spec = importlib.util.spec_from_file_location("_claw_mouth_under_test", path)
    assert spec and spec.loader
    claw_mouth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(claw_mouth)

    model = claw_mouth.build(MODEL)
    bands = claw_mouth.profile(model, GRIPPER_OPEN)
    assert bands, "no bands along the mouth; KNUCKLE_CLEAR is past the jaws"
    worst = max(drawn - collided for _, _, drawn, collided in bands)
    assert worst <= claw_mouth.ROOM_TOLERANCE, (
        f"the solver's mouth is {worst * 1000:.1f} mm narrower than the part's; "
        "a jaw is hulled where it should be decomposed (five_foot/jaws.py)"
    )
    # The control: the same measurement against the jaw as one hull has to fail,
    # or this passes for everything.
    assert any(collided > 0 for _, _, _, collided in bands), (
        "every point in the middle of the mouth is inside a collision shape, "
        "which is the failure this is meant to detect and not a pass"
    )



def test_no_corner_of_the_hold_box_puts_the_claw_through_the_floor() -> None:
    """What actually bounds `GRASP_BOX`, and it is not the joint limits.

    Swept with `grasp_pose.py --check-box`, every corner is inside the soft limits
    out to 0.25 rad -- so the test next to this one passes at any width anyone is
    likely to try. What fails first is geometry: floor clearance goes 28.3 mm at
    0.06, 19.5 at 0.10, 15.1 at 0.12, 8.5 at 0.15 and **-2.2 mm at 0.20**, where
    the claw is through the ground.

    A claw that intersects the floor at a box corner does not raise anything. It
    gives a contact the task never intended, on a geom that is not a foot, once
    per episode in the environments that drew that corner -- which
    `ground_contact` then charges the policy for.

    Measured here rather than restated: the lowest point of the carried arm, over
    all 16 corners, with the robot standing at STAND_Z.
    """
    m = _model()
    d = mujoco.MjData(m)
    lf_geoms = [
        g
        for g in range(m.ngeom)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("LF_")
    ]
    assert lf_geoms, "no LF geoms found, so this test cannot see the claw"

    def lowest(arm: dict[str, float]) -> float:
        mujoco.mj_resetData(m, d)
        for joint, value in (dict(HOME) | dict(LF_GRASP) | arm).items():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if jid >= 0:
                d.qpos[m.jnt_qposadr[jid]] = value
        d.qpos[2] = STAND_Z
        mujoco.mj_forward(m, d)
        return min(float(d.geom_xpos[g][2]) for g in lf_geoms)

    worst = min(
        lowest({j: LF_GRASP[j] + (GRASP_BOX if (bits >> k) & 1 else -GRASP_BOX)
                for k, j in enumerate(ARM_JOINTS)})
        for bits in range(2 ** len(ARM_JOINTS))
    )
    assert worst > 0.010, (
        f"a hold-box corner puts the claw {worst * 1000:.1f} mm above the floor; "
        f"GRASP_BOX={GRASP_BOX} is too wide"
    )
    # Control: the geom centres this reads really do move with the box, or the
    # assertion above is measuring a constant.
    tight = min(
        lowest({j: LF_GRASP[j] + (0.01 if (bits >> k) & 1 else -0.01)
                for k, j in enumerate(ARM_JOINTS)})
        for bits in range(2 ** len(ARM_JOINTS))
    )
    assert tight > worst + 0.002, (
        "shrinking the box did not raise the claw, so this test is not reading "
        "the box at all"
    )


def test_rf_is_the_only_foothold_inside_the_body() -> None:
    """The measurement the outboard penalty exists for, pinned against the model.

    Four of the five ground feet stand well clear of the trunk; RF does not, and
    on five feet RF is the *whole* front support. If a future asset change moves
    the feet, this is the assertion that says whether the term still has a job --
    and the control at the end is what stops it passing by `TRUNK_HALF_WIDTH`
    simply being wrong.
    """
    m = _model()
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    ys = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != base or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        ys.append(float(abs(m.mesh_vert[a : a + n][:, 1]).max()))
    assert ys, "no trunk meshes found"
    assert TRUNK_HALF_WIDTH == pytest.approx(max(ys), abs=0.003), (
        f"TRUNK_HALF_WIDTH is {TRUNK_HALF_WIDTH} but the trunk meshes reach "
        f"{max(ys):.4f} m"
    )

    inside = [
        leg for leg, (_x, y) in zip(FIVE_FOOT_LEGS, HOME_TOE_XY, strict=True)
        if abs(y) < TRUNK_HALF_WIDTH
    ]
    assert inside == ["RF"], (
        f"the feet inside the body's footprint are {inside}; the outboard penalty "
        f"is scoped to RF alone"
    )


def test_the_outboard_penalty_is_one_sided_and_signed() -> None:
    """Two properties, and each fails silently on its own.

    **One-sided.** Past the body's edge the term must be exactly zero, or it stops
    being "get out from under the body" and becomes "stand at this exact width" --
    a two-sided pull the leg then fights every time it reaches further out.

    **Signed, not `|y|`.** A foot that has swung across the centreline onto the
    *left* side is the worst case there is, and an unsigned distance scores it as
    though it were fine. That is the assertion at the end.
    """
    def cost(y: float) -> float:
        pos = torch.zeros(1, len(HOME_TOE_XY), 3)
        pos[0, 0, 1] = y
        robot = _StubAsset(
            _StubData(
                # What `_site_pos_w` reads: MuJoCo's rows, and the entity's ids.
                data=_StubData(site_xpos=pos),
                indexing=_StubData(site_ids=torch.arange(len(HOME_TOE_XY))),
                root_link_pos_w=torch.zeros(1, 3),
                root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            )
        )
        env = _StubEnv(robot, None)
        asset_cfg = SceneEntityCfg("robot")
        asset_cfg.site_ids = list(range(len(HOME_TOE_XY)))
        return float(
            foot_outboard_of_trunk(
                env,
                foot_index=0,
                min_offset=TRUNK_HALF_WIDTH,
                outboard_sign=-1.0,
                asset_cfg=asset_cfg,
            )[0]
        )

    assert cost(-TRUNK_HALF_WIDTH) == pytest.approx(0.0, abs=1e-9)
    assert cost(-0.15) == 0.0, "the penalty does not stop at the body's edge"
    assert cost(-0.30) == 0.0, "reaching further out is being charged for"
    assert cost(-0.0365) == pytest.approx(TRUNK_HALF_WIDTH - 0.0365, abs=1e-6), (
        "the HOME foothold is not being charged its distance from the edge"
    )
    assert cost(0.0) > cost(-0.0365), "moving inboard is not costing more"
    assert cost(+0.05) > cost(0.0), (
        "a foot that has crossed the centreline scores no worse than one on it, "
        "so the term is reading |y| instead of the signed offset"
    )


class _PlantedEnv:
    def __init__(self, contacts, twist) -> None:
        sensor = _StubAsset(_StubData(found=torch.tensor([contacts], dtype=torch.float)))
        self.scene = _StubScene(sensor, None)
        self.scene.sensors = {"feet": sensor}
        self.command_manager = _StubCommandManager({"twist": torch.tensor([twist])})


def test_feet_planted_pays_per_foot_and_only_at_a_standstill() -> None:
    """A positive term a motionless robot collects in full is what this reward set
    otherwise refuses, so the gate is the whole safety argument and has to be
    asserted, not assumed: under a command it must be exactly zero, or standing
    starts competing with obeying.
    """
    all_down = _PlantedEnv([1, 1, 1, 1, 1], (0.0, 0.0, 0.0))
    one_up = _PlantedEnv([1, 0, 1, 1, 1], (0.0, 0.0, 0.0))
    walking = _PlantedEnv([1, 1, 1, 1, 1], (0.4, 0.0, 0.0))

    kwargs = {"sensor_name": "feet", "command_name": "twist", "command_threshold": 0.1}
    assert feet_contact_without_cmd(all_down, **kwargs).item() == pytest.approx(5.0)
    assert feet_contact_without_cmd(one_up, **kwargs).item() == pytest.approx(4.0), (
        "lifting a foot at a standstill costs nothing, so the term is not per-foot"
    )
    assert feet_contact_without_cmd(walking, **kwargs).item() == pytest.approx(0.0), (
        "a commanded robot is collecting the standstill term"
    )


class _NamedScene:
    def __init__(self, items: dict) -> None:
        self._items = items

    def __getitem__(self, name):
        return self._items[name]


class _GroupLoadEnv:
    """A contact sensor and a command -- what `group_load_balance` reads."""

    def __init__(self, forces, twist=(0.4, 0.0, 0.0)) -> None:
        n = len(FIVE_FOOT_LEGS)
        sensor = _StubAsset(_StubData(force=torch.tensor([forces]).reshape(1, n, 3)))
        self.scene = _NamedScene({"feet": sensor})
        self.command_manager = _StubCommandManager({"twist": torch.tensor([twist])})
        self.num_envs = 1
        self.device = "cpu"


def _group_term(alpha=1.0):
    cfg = RewardTermCfg(
        func=group_load_balance,
        weight=1.0,
        params={"group_a": GROUP_A, "group_b": GROUP_B},
    )
    env = _GroupLoadEnv([[0.0, 0.0, 0.0]] * len(FIVE_FOOT_LEGS))
    term = group_load_balance(cfg=cfg, env=env)
    kwargs = {
        "sensor_name": "feet",
        "group_a": GROUP_A,
        "group_b": GROUP_B,
        "command_name": "twist",
        "alpha": alpha,
    }
    return term, kwargs


#: What `group_load_balance` scores when one group carries everything.
#:
#: **Not zero.** The gap is relative -- `|A - B| / mean(A, B)` -- so with B at
#: zero it is `A / (A/2) = 2` whatever A is, and the kernel bottoms out at
#: `exp(-2)`. The term's range is therefore [0.135, 1], and a completely
#: one-sided stance still collects 13.5% of the weight. The left/right term this
#: replaced had the same floor for the same reason.
_GROUP_LOAD_FLOOR = math.exp(-2.0)


def _forces(per_foot):
    return [[0.0, 0.0, f] for f in per_foot]


def test_group_load_balance_scores_the_gait_partition_not_left_and_right() -> None:
    """It must read `GROUP_A` / `GROUP_B` -- the partition `five_foot_gait` scores
    -- and not some other split of the same five columns.

    A term reading the wrong columns still returns a plausible number in (0, 1]
    and still moves during training; nothing downstream can tell. So the test
    constructs a case that is balanced across the *gait* groups and unbalanced
    left/right, and checks which one the term agrees with.
    """
    term, kwargs = _group_term(alpha=1.0)  # alpha 1.0: no memory, read this frame

    # A = RF, LM, RR ; B = RM, LR.  Equal group means, wildly unequal left/right.
    balanced_by_group = _forces([6.0, 6.0, 6.0, 6.0, 6.0])
    env = _GroupLoadEnv(balanced_by_group)
    assert term(env, **kwargs).item() == pytest.approx(1.0, abs=1e-5)

    # Now load group A and starve group B.
    term, kwargs = _group_term(alpha=1.0)
    env = _GroupLoadEnv(_forces([10.0, 10.0, 1.0, 1.0, 10.0]))
    lopsided = term(env, **kwargs).item()
    assert lopsided < 0.3, (
        f"group A carrying ten times group B still scores {lopsided:.3f}"
    )


def test_group_load_balance_averages_rather_than_scoring_the_instant() -> None:
    """**The whole design rests on this**, and a per-step version is the natural
    thing to write.

    Instantaneously the two groups are *supposed* to be unbalanced: a correct 3+2
    gait puts one group in stance while the other swings, so a per-step term is
    minimised exactly where `five_foot_gait` is maximised. Averaged over a cycle
    the alternation cancels and the question becomes "does each group take its
    turn", which is the one worth asking.

    So: alternate the two groups frame by frame, and the term must end up near 1.0
    -- while the same sequence scored instantaneously stays near 0.
    """
    a_down = _forces([15.0, 15.0, 0.0, 0.0, 15.0])
    b_down = _forces([0.0, 0.0, 15.0, 15.0, 0.0])

    term, kwargs = _group_term(alpha=0.05)
    for i in range(400):
        value = term(_GroupLoadEnv(a_down if i % 2 else b_down), **kwargs)
    assert value.item() > 0.85, (
        f"a gait that alternates evenly scores {value.item():.3f}; the term is "
        f"reading the instant rather than the average"
    )

    # Control on the *data*, so the assertion above cannot pass on an input that
    # was balanced all along: at every single frame one group carries everything
    # and the other nothing. A per-step reading of that is a relative gap of
    # exactly 2 -- the term's floor, `exp(-2)` -- on every frame of the sequence.
    # The score above can therefore only have come from the averaging.
    mean_a = sum(a_down[i][2] for i in GROUP_A) / len(GROUP_A)
    mean_b = sum(a_down[i][2] for i in GROUP_B) / len(GROUP_B)
    assert mean_b == 0.0 and mean_a > 0.0
    instant_gap = abs(mean_a - mean_b) / ((mean_a + mean_b) / 2)
    assert math.exp(-instant_gap) == pytest.approx(_GROUP_LOAD_FLOOR, abs=1e-9)


def test_group_load_balance_declines_to_score_until_both_groups_have_borne() -> None:
    """Each group's average only advances while that group bears, so at the start
    of an episode one of them has no measurement at all.

    Scoring anyway means comparing a real stance load against an uninitialised
    zero and reporting a maximal imbalance the robot has not earned -- for as long
    as it takes the other group to touch down. The term returns exactly 0 instead.

    The control is the frame after the second group finally bears: the score must
    become non-zero, or this test passes against a term that returns 0 always.
    """
    term, kwargs = _group_term(alpha=1.0)
    a_only = _GroupLoadEnv(_forces([15.0, 15.0, 0.0, 0.0, 15.0]))
    for _ in range(50):
        value = term(a_only, **kwargs)
    assert value.item() == 0.0, (
        f"scored {value.item():.3f} with group B never having borne load"
    )

    both = _GroupLoadEnv(_forces([15.0, 15.0, 15.0, 15.0, 15.0]))
    assert term(both, **kwargs).item() > 0.0, (
        "the term stays at zero after both groups have borne, so it is not "
        "measuring anything"
    )

    # And a reset puts it back to having no measurement.
    term.reset(None)
    for _ in range(50):
        value = term(a_only, **kwargs)
    assert value.item() == 0.0, (
        "after a reset the term still reports the previous episode's comparison"
    )


def test_group_load_balance_is_not_diluted_by_duty_cycle() -> None:
    """**The reason the average is conditional on bearing.**

    Group A bears twice as often as group B here, but carries exactly the same
    force while it does. An unconditional average would divide each group's stance
    load by its duty cycle and report a two-to-one imbalance -- which is not a
    load imbalance at all, it is the duty difference `stance_duty_balance` already
    scores. Conditioned on bearing, the two are equal and the term says so.

    The control is the same duty pattern with genuinely unequal stance loads: that
    must still score badly, or the term has stopped seeing force.
    """
    a_down = _forces([12.0, 12.0, 0.0, 0.0, 12.0])
    b_down = _forces([0.0, 0.0, 12.0, 12.0, 0.0])
    idle = _forces([0.0, 0.0, 0.0, 0.0, 0.0])

    term, kwargs = _group_term(alpha=0.05)
    for i in range(600):                      # A bears 2 frames in 3, B bears 1
        term(_GroupLoadEnv(b_down if i % 3 == 2 else a_down), **kwargs)
    equal = term(_GroupLoadEnv(idle), **kwargs).item()
    assert equal > 0.9, (
        f"equal stance loads at unequal duty scored {equal:.3f}; the average is "
        f"being diluted by how often each group is down"
    )

    # Control: halve group B's stance load and the term must notice.
    b_weak = _forces([0.0, 0.0, 4.0, 4.0, 0.0])
    term, kwargs = _group_term(alpha=0.05)
    for i in range(600):
        term(_GroupLoadEnv(b_weak if i % 3 == 2 else a_down), **kwargs)
    unequal = term(_GroupLoadEnv(idle), **kwargs).item()
    assert unequal < 0.6, (
        f"group B carrying a third of group A still scores {unequal:.3f}"
    )


def test_group_load_balance_is_gated_on_the_command(cfg) -> None:
    """Standing still is trivially balanced once the average settles, so ungated
    this becomes a constant added to the do-nothing score -- the basin this whole
    reward set is arranged against.
    """
    assert cfg.rewards["lateral_load"].params["command_name"] == "twist"
    term, kwargs = _group_term(alpha=1.0)
    standing = _GroupLoadEnv(_forces([6.0] * 5), twist=(0.0, 0.0, 0.0))
    assert term(standing, **kwargs).item() == 0.0, "a standing robot collects it"


def _rolled_gravity(deg: float) -> torch.Tensor:
    """Projected gravity for a body rolled `deg` about its own +x axis."""
    half = math.radians(deg) / 2.0
    quat = torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]])
    return quat_apply_inverse(quat, torch.tensor([[0.0, 0.0, -1.0]]))


def test_roll_is_charged_symmetrically_and_priced_like_pitch(cfg) -> None:
    """`upright` was the only term scoring roll angle and it is 42 times cheaper
    per degree than `nose_pitch` is for the other axis. A policy charged that
    unevenly rolls, and two measurements found it doing exactly that -- a +2.39
    degree standing list on `model_2200`, a -3.36 degree list turning left on
    `model_2400`, both near-constant within their phase rather than oscillating.

    So the two axes must be priced alike, and symmetrically: a one-sided roll
    penalty would send the list to the other side, which is what happened twice on
    the pitch axis before it was made symmetric.
    """

    def cost(deg: float) -> float:
        env = _StubEnv(_StubAsset(_StubData(projected_gravity_b=_rolled_gravity(deg))), None)
        return float(body_roll(env)[0])

    # Same plain square as pitch -- see `test_pitch_is_charged_symmetrically...`.
    assert cost(1.0) == pytest.approx(cost(-1.0), rel=1e-9), "1 degree is not symmetric"
    assert cost(1.0) < cost(20.0) / 100.0, "a degree of roll is not cheap"
    for deg in (3.0, 5.0, 10.0, 20.0):
        assert cost(deg) > 0.0 and cost(-deg) > 0.0, f"{deg} degrees of roll is free"
        assert cost(deg) == pytest.approx(cost(-deg), rel=1e-6), (
            f"{deg} degrees costs differently left and right; the term is one-sided"
        )

    # Superlinear, by the same measure used for pitch: cost per radian of excess
    # is flat for a linear term and rises for this one.
    def per_radian(deg: float) -> float:
        return cost(deg) / math.radians(abs(deg))

    assert per_radian(15.0) > 1.5 * per_radian(5.0), (
        "the cost per radian of roll is flat, so a large list is as cheap per "
        "degree as a small one"
    )

    # **The two axes must be priced comparably**, which is the point of the term
    # -- not identically, since the weights are a tuning knob and roll's useful
    # range is the narrower one. What must not come back is the order-of-magnitude
    # gap: `upright` alone made pitch 42 times dearer than roll, and the policy
    # spent the difference.
    # **What has to be bounded is the price of an angle, not a pair of weights.**
    # Written first as `max(w) / min(w) <= 10`, which broke the moment a term was
    # switched off -- it divided by zero -- and it could not see that other terms
    # charge both axes whatever `body_roll`'s weight is. The question the failure
    # asks is "what does ten degrees cost on each axis", so that is what this
    # measures: every registered term that scores an attitude angle, at its
    # configured weight.
    #
    # **When the task registers no roll term the bound does not apply**, and that
    # is a configuration this task ships: `2400.pt`'s reward set has `nose_pitch`
    # and no `body_roll`, and the run after it is that set again. The asymmetry is
    # then real and measured rather than hidden -- 10 degrees of roll costs 0.14
    # against pitch's 5.93, and `2400.pt` held a +0.39 degree standing roll bias,
    # |roll| 1.22 degrees, peaks to 5.08 -- so what is left to pin is that the
    # term is *there to switch on*, and priced sanely when it is.
    def priced(deg: float, axis: str) -> float:
        gravity = _pitched(deg) if axis == "pitch" else _rolled_gravity(deg)
        env = _StubEnv(_StubAsset(_StubData(projected_gravity_b=gravity)), None)
        env.command_manager = _StubCommandManager({"body_pose": torch.zeros(1, 3)})
        total = 0.0
        for name in ("nose_pitch", "body_roll", "track_body_pose"):
            term = cfg.rewards.get(name)
            if term is None or term.weight == 0.0:
                continue
            params = {k: v for k, v in term.params.items() if k != "asset_cfg"}
            total += abs(term.weight * float(term.func(env, **params)[0]))
        return total

    if "body_roll" in cfg.rewards and cfg.rewards["body_roll"].weight != 0.0:
        pitch_cost, roll_cost = priced(10.0, "pitch"), priced(10.0, "roll")
        ratio = pitch_cost / roll_cost
        # **10x, not parity.** The failure this term exists for was an *order of
        # magnitude*: `upright` alone priced 10 degrees of roll at 0.14 against
        # `nose_pitch`'s 5.93, a factor of 42, and the policy spent the difference
        # on a parked 2.39-degree list. Inside that the ratio is a tuning knob and
        # the measurement does not settle it: at -10, roll improved in all seven
        # commanded directions and linear tracking fell from 95.4% to 88.1%.
        assert ratio <= 10.0, (
            f"ten degrees of pitch costs {pitch_cost:.2f} against roll's "
            f"{roll_cost:.2f}, a factor of {ratio:.1f}; the gap this term exists "
            f"to close is back"
        )
    else:
        # The control group for the branch above: the term still exists and still
        # charges the axis, so switching it on is a weight and not a rewrite.
        env = _StubEnv(_StubAsset(_StubData(projected_gravity_b=_rolled_gravity(10.0))), None)
        assert float(body_roll(env)[0]) > 0.0

    # The two penalty shapes are the same function, so at equal angle they differ
    # only by weight -- which is what makes the ratio above a statement about
    # pricing rather than about two different curves.
    pitch_env = _StubEnv(_StubAsset(_StubData(projected_gravity_b=_pitched(10.0))), None)
    assert cost(10.0) == pytest.approx(
        float(nose_pitch(pitch_env)[0]), rel=1e-4
    ), "ten degrees of roll and ten of pitch are not the same function"
