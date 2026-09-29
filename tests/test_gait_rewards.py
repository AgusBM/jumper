"""Do the gait rewards really bind the leg grouping.

This began as a script someone had to remember to run. It needs no GPU, runs in
milliseconds and carries its own criteria -- which makes it a test, so it lives
here and runs every time.

The criterion: **any wrong grouping must score nothing** (< 0.1). That is not
pedantry -- the old product-of-means implementation gave the worst case, lifting
LF+LM (same side, no support at the front-left), a score of 0.375, so the policy
could collect nearly half the gait reward for having roughly some legs up with no
pressure at all to get the grouping right. The measured consequence: the tetrapod
gait trained to a mean of 2.54 feet in contact (the target is 4.0) while the gait
term still collected 0.43/1.0.

The "one leg short" intermediate state **deliberately** keeps about 0.13, as the
gradient from "one wrong" to "exactly right", and is not a leak -- any wrong
grouping is at least two legs off (one that should be up is not, one that should
not be is), so the 0.1 line separates the two exactly.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the gait rewards are implemented in torch")

from tasks.jumper.common.constants import LEGS  # noqa: E402
from tasks.jumper.common.mdp.rewards import phase_match  # noqa: E402
from tasks.jumper.ripple.mdp.rewards import RIPPLE_ORDER  # noqa: E402
from tasks.jumper.tetrapod.mdp.rewards import TETRAPOD_PAIRS  # noqa: E402
from tasks.jumper.tripod.mdp.rewards import TRIPOD_A, TRIPOD_B  # noqa: E402

#: Upper bound on a wrong grouping's score. Above it, the grouping is not bound.
WRONG_MAX = 0.1
#: Lower bound on a legal grouping's score.
RIGHT_MIN = 0.9

LEG_NAMES = list(LEGS)


def contact_with_airborne(airborne: set[str]) -> torch.Tensor:
    """Build a [1, 6] contact tensor: legs in `airborne` are up (0), the rest are
    down (1)."""
    return torch.tensor([[0.0 if n in airborne else 1.0 for n in LEG_NAMES]])


def tetrapod_score(contact: torch.Tensor, sigma: float = 0.7) -> float:
    """Tetrapod: any one of the three pairs airborne is legal; take the best."""
    return float(
        torch.stack([phase_match(contact, p, sigma) for p in TETRAPOD_PAIRS], 1).amax(1)
    )


def tripod_score(contact: torch.Tensor, sigma: float = 0.7) -> float:
    """Tripod: two groups in antiphase; either being airborne is legal."""
    return float(
        torch.maximum(
            phase_match(contact, TRIPOD_A, sigma), phase_match(contact, TRIPOD_B, sigma)
        )
    )


# ── Tetrapod ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pair", TETRAPOD_PAIRS)
def test_tetrapod_legal_pairs_score_full(pair: tuple[int, int]) -> None:
    airborne = {LEG_NAMES[i] for i in pair}
    assert tetrapod_score(contact_with_airborne(airborne)) > RIGHT_MIN


@pytest.mark.parametrize(
    "airborne,why",
    [
        ({"LF", "LM"}, "same side: no support at all at the front-left, the worst case"),
        ({"LF", "RF"}, "both front legs lifted"),
        ({"LM", "LR"}, "same side, rear half"),
        (set(), "all down -- standing still"),
        (set(LEGS), "all airborne"),
    ],
)
def test_tetrapod_wrong_groupings_score_nothing(airborne: set[str], why: str) -> None:
    score = tetrapod_score(contact_with_airborne(airborne))
    assert score < WRONG_MAX, f"a wrong grouping scored {score:.4f} ({why})"


def test_tetrapod_one_leg_off_keeps_gradient() -> None:
    """One leg short must keep some score, or there is no gradient toward the
    correct grouping."""
    score = tetrapod_score(contact_with_airborne({"LF", "RR", "LM"}))
    assert WRONG_MAX < score < RIGHT_MIN, (
        f"the intermediate state scored {score:.4f}, outside the band"
    )


# ── Tripod ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("group", [TRIPOD_A, TRIPOD_B])
def test_tripod_whole_group_airborne_scores_full(group: tuple[int, ...]) -> None:
    airborne = {LEG_NAMES[i] for i in group}
    assert tripod_score(contact_with_airborne(airborne)) > RIGHT_MIN


@pytest.mark.parametrize(
    "airborne,why",
    [
        ({LEG_NAMES[TRIPOD_A[0]], LEG_NAMES[TRIPOD_B[0]]},
         "one leg from each group: three legs off"),
        (set(), "all down: three legs off"),
        (set(LEGS), "all airborne: three legs off"),
    ],
)
def test_tripod_wrong_groupings_score_nothing(airborne: set[str], why: str) -> None:
    score = tripod_score(contact_with_airborne(airborne))
    assert score < WRONG_MAX, f"a wrong grouping scored {score:.4f} ({why})"


def test_tripod_one_leg_off_keeps_gradient() -> None:
    """"Group A with only two lifted" is **the one-leg-short intermediate state**,
    not a wrong grouping, and must keep some score.

    It is one leg off the A phase (LR should be up and is not) and measures 0.1299
    -- exactly the convergence gradient the design deliberately keeps, the same
    0.13 as in the tetrapod case. **Any genuinely wrong grouping is at least two
    legs off** (one that should be up is not, one that should not be is), and the
    0.1 line separates the two exactly.

    The original script marked this as a failure but never asserted on it (its exit
    code counted only the tetrapod cases), and that wrong label was mistaken for a
    criterion.
    """
    airborne = {LEG_NAMES[TRIPOD_A[0]], LEG_NAMES[TRIPOD_A[1]]}
    score = tripod_score(contact_with_airborne(airborne))
    assert WRONG_MAX < score < RIGHT_MIN, (
        f"the intermediate state scored {score:.4f}, outside the band"
    )


# ── The grouping constants themselves ─────────────────────────────────────


def test_gait_groupings_cover_every_leg() -> None:
    """Tripod's two groups and ripple's six must each cover all six legs exactly
    once."""
    assert sorted(TRIPOD_A + TRIPOD_B) == list(range(len(LEGS)))
    assert sorted(RIPPLE_ORDER) == list(range(len(LEGS)))


def test_tetrapod_pairs_are_disjoint_and_complete() -> None:
    """The three pairs lift in turn and together cover exactly the six legs."""
    flat = [i for pair in TETRAPOD_PAIRS for i in pair]
    assert sorted(flat) == list(range(len(LEGS)))


def test_ripple_order_alternates_sides() -> None:
    """Consecutive ripple legs must be on opposite sides -- lifting two same-side
    legs in a row is exactly the defect it avoids.

    Side is decided by the parity of the LEGS index: LF/LM/LR are even (left),
    RF/RM/RR odd (right).
    """
    sides = [i % 2 for i in RIPPLE_ORDER]
    for a, b in zip(sides, sides[1:] + sides[:1]):
        assert a != b, f"RIPPLE_ORDER lifts two same-side legs in a row: {RIPPLE_ORDER}"
