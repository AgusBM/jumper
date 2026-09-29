"""The hexapod's left-right mirror transform, for rsl_rl's symmetry augmentation
and mirror loss.

**Why it is worth doing**: the hexapod has strict sagittal mirror symmetry
(``mirror_pairs = [[0,5],[1,4],[2,3]]`` in the topology, i.e. LF<->RF, LM<->RM,
LR<->RR). Telling the algorithm about that prior buys two things:

- **data augmentation**: every mini-batch doubles for free, twice the effective
  sample count
- **mirror loss**: an explicit penalty on "the policy's output under a mirrored
  observation != the mirror of its original output", which removes left-right
  imbalance in the gait directly

Reference: Mittal et al., *Symmetry Considerations for Learning Task Symmetric
Robot Policies*, ICRA 2024 (rsl_rl's implementation comes from that paper).

## Deriving the mirror transform

The mirror plane is the body's x-z plane (y -> -y), so M = diag(1, -1, 1).

**Vectors** (linear velocity, projected gravity, contact forces): v -> M.v, i.e.
negate the y component.

**Pseudovectors** (angular velocity, rotation about an axis): w -> -M.w, since
det(M) = -1. By joint axis:

| Axis | -M.w | Result |
|---|---|---|
| z (0,0,1) | -(0,0,1) | **negated** |
| y (0,1,0) | (0,1,0) | **kept** |
| x (1,0,0) | -(1,0,0) | **negated** |

Measured axes on this robot: every `J0` is z, the front arms' `J1` is y, and
everything else -- the front arms' `J2` / `J3` / `J4` and the walking legs' `J1` /
`J2` -- is x. Each axis is cardinal to within 1e-9, not a blend. **So only
`LF_J1_joint` and `RF_J1_joint` keep their sign.**

That conclusion is independently confirmed by the HOME pose: HOME is itself mirror
symmetric, so the mirror must map it to itself. Measured across left-right joint
pairs, `J1` on the front arms has the same sign (-1.0658 / -1.0660) and every other
joint is negated. **Two independent sources agree exactly.**

**The table has to name joints in full, and `J1` alone would be wrong.** The V1.6
naming numbers each leg's joints from the base outward, so `J1` is the front arms'
shoulder pitch (y, kept) *and* the walking legs' knee (x, negated) -- one index, two
axes. The names it replaced, `shoulder_pitch` and `knee`, carried that distinction
for free, and the substring rule that read them silently stopped matching anything
when they went away: every joint fell through to -1.0, the front arms' pitch was
negated along with the rest, and the actor would have been handed mirrored samples
whose arm pose is not the mirror of the original. `tests/test_symmetry.py` now
re-derives this table from the model's joint axes so the next rename cannot do it
again quietly.

## Layout

Observation term slices are **derived dynamically from
`env.observation_manager`**, never hardcoded as offsets: any config change (drop
height_scan, add a term) moves the offsets, and hardcoded ones would then be wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..constants import GAIT_JOINTS, LEGS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

# ── The joint mirror table ────────────────────────────────────────────────
# The joints that keep their sign under mirroring -- the ones whose axis is y.
# Full names, not an index substring: see the docstring on why "J1" would catch
# four joints that must negate. `tests/test_symmetry.py` checks this against the
# model's measured axes.
_SIGN_KEEP = ("LF_J1_joint", "RF_J1_joint")

# Left-right leg pairs
_LEG_PAIR = {"LF": "RF", "RF": "LF", "LM": "RM", "RM": "LM", "LR": "RR", "RR": "LR"}


def _build_joint_map() -> tuple[list[int], list[float]]:
    """Return (permutation indices, signs), both of length len(GAIT_JOINTS).

    ``mirrored[i] = sign[i] * original[perm[i]]``
    """
    idx = {j: i for i, j in enumerate(GAIT_JOINTS)}
    perm, sign = [], []
    for j in GAIT_JOINTS:
        prefix, rest = j[:2], j[2:]
        partner = _LEG_PAIR[prefix] + rest
        if partner not in idx:
            raise KeyError(f"no mirror counterpart {partner!r} found for {j!r}")
        perm.append(idx[partner])
        sign.append(1.0 if any(k in j for k in _SIGN_KEEP) else -1.0)
    return perm, sign


def _build_leg_perm() -> list[int]:
    """Left-right permutation indices for per-foot quantities (one scalar or one
    vector per leg)."""
    idx = {n: i for i, n in enumerate(LEGS)}
    return [idx[_LEG_PAIR[n]] for n in LEGS]


JOINT_PERM, JOINT_SIGN = _build_joint_map()
LEG_PERM = _build_leg_perm()

#: How each observation term mirrors. Keys are term names from
#: `observation_manager.active_terms`.
#: "vec3"     -- a vector; negate the y component
#: "pseudo3"  -- a pseudovector; negate the x and z components
#: "joint"    -- a joint quantity; permute by JOINT_PERM and apply JOINT_SIGN
#: "twist"    -- the command (vx, vy, wz); negate vy and wz
#: "posture4" -- a posture (twist, pitch, roll, height); negate twist and roll
#: "leg"      -- one scalar per leg; permute by LEG_PERM
#: "legvec3"  -- one 3-vector per leg; permute, then negate y
#: "joint_tiles" -- several equal-width blocks of GAIT_JOINTS-ordered joint
#:                  values (e.g. the jump task's future-reference preview);
#:                  each block is mirrored independently like "joint"
#: "scalar"   -- a scalar (e.g. the jump phase); left-right symmetric, returned
#:               unchanged
#: "phase_half_shift" -- a (sin, cos) phase clock whose mirror image is the same
#:              clock half a cycle later; negate both components
#: "heightscan" -- a grid of ground heights; permute each ray onto the ray at
#:              mirrored y. The permutation is **derived from the sensor's own ray
#:              coordinates**, not from the grid formula -- see `_scan_perm`.
_TERM_KIND = {
    "base_lin_vel": "vec3",
    "base_ang_vel": "pseudo3",
    "projected_gravity": "vec3",
    "joint_pos": "joint",
    "joint_vel": "joint",
    "actions": "joint",
    "command": "twist",
    "jump_phase": "scalar",
    "ref_future": "joint_tiles",
    "foot_height": "leg",
    "foot_air_time": "leg",
    "foot_contact": "leg",
    "foot_contact_forces": "legvec3",
    # Actuator force is a generalised force about the same axis as its joint, so
    # it mirrors exactly as the joint coordinate does: same permutation, same sign
    # rule. A joint whose angle negates under the mirror has its torque negate too.
    "actuator_force": "joint",
    "height_scan": "heightscan",
    # Only jumper.tripod mounts this term (see its env_cfg.py); the entry is inert
    # for the other three tasks, whose observation groups have no such key.
    "gait_phase": "phase_half_shift",
}


_SCAN_PERM: dict[int, torch.Tensor] = {}


def _scan_perm(env, width: int) -> torch.Tensor:
    """Permutation taking each height-scan ray onto the ray mirrored in y.

    **Read off the sensor's actual ray offsets**, by finding for every ray the one
    whose local position is (x, -y). Not computed from the grid's index
    arithmetic: that would be a second copy of `GridPatternCfg.generate_rays`'s
    ordering, correct only until someone changes the pattern or its `indexing=`,
    and wrong afterwards in complete silence -- the shapes stay identical and the
    mirrored sample is simply a scan of ground that is not there.

    Deriving it also means a non-grid pattern works without a new branch, as long
    as the pattern is itself symmetric about y. It is not, this raises rather than
    quietly pairing rays that do not correspond.
    """
    if width in _SCAN_PERM:
        return _SCAN_PERM[width]

    for e in (env, getattr(env, "unwrapped", None), getattr(env, "env", None)):
        sensors = getattr(getattr(e, "scene", None), "sensors", None)
        if sensors and "terrain_scan" in sensors:
            offsets = sensors["terrain_scan"]._local_offsets
            break
    else:
        raise AttributeError(
            "height_scan is in the observation but the 'terrain_scan' sensor is "
            "not on the scene, so its mirror cannot be derived"
        )

    xy = offsets[:, :2].detach().cpu()
    if xy.shape[0] != width:
        raise ValueError(
            f"height_scan is {width} wide but the sensor has {xy.shape[0]} rays"
        )
    target = xy.clone()
    target[:, 1] *= -1.0
    # Nearest ray to each mirrored position, then check it really is that position.
    d = torch.cdist(target, xy)
    perm = d.argmin(dim=1)
    worst = float((xy[perm] - target).abs().max())
    if worst > 1e-5:
        raise ValueError(
            f"the scan pattern is not symmetric about y: the best match for some "
            f"ray is {worst:.4f} m away, so no permutation mirrors it"
        )
    _SCAN_PERM[width] = perm
    return perm


def _mirror_slice(x: torch.Tensor, kind: str, env=None) -> torch.Tensor:
    """Mirror one observation slice by kind; x has shape [B, D]."""
    if kind == "heightscan":
        if env is None:
            raise ValueError("the height-scan mirror needs the env to read the rays")
        return x[:, _scan_perm(env, x.shape[1]).to(x.device)]
    if kind == "vec3":
        return x * torch.tensor([1.0, -1.0, 1.0], device=x.device)
    if kind == "pseudo3":
        return x * torch.tensor([-1.0, 1.0, -1.0], device=x.device)
    if kind == "twist":  # (vx, vy, wz): vy is a vector component and wz a
                         # pseudovector component; both negate
        return x * torch.tensor([1.0, -1.0, -1.0], device=x.device)
    if kind == "posture4":
        # (twist, pitch, roll, height), the posture command and its measurement.
        # Twist is a rotation about z and roll one about x, so both negate under
        # the mirror (`-M.w` with det M = -1, the same rule the table above
        # derives for the joint axes). Pitch is about y and is kept, and a height
        # is a scalar distance with no handedness at all.
        #
        # **The failure this prevents is a policy that leans the wrong way.** The
        # shapes match whichever signs are used, so a wrong entry here trains the
        # actor on samples saying "roll left" beside a body rolled right.
        return x * torch.tensor([-1.0, 1.0, -1.0, 1.0], device=x.device, dtype=x.dtype)
    if kind == "joint":
        perm = torch.as_tensor(JOINT_PERM, device=x.device)
        sign = torch.as_tensor(JOINT_SIGN, device=x.device, dtype=x.dtype)
        return x[:, perm] * sign
    if kind == "joint_tiles":
        # Several equal-width blocks of GAIT_JOINTS-ordered joint values (each
        # 20 columns); mirror each block independently.
        width = len(JOINT_PERM)
        assert x.shape[1] % width == 0, (
            f"joint_tiles needs a column count that is a multiple of {width}, "
            f"got {x.shape[1]}"
        )
        perm = torch.as_tensor(JOINT_PERM, device=x.device)
        sign = torch.as_tensor(JOINT_SIGN, device=x.device, dtype=x.dtype)
        perm_t = torch.cat([perm + width * k for k in range(x.shape[1] // width)])
        sign_t = sign.repeat(x.shape[1] // width)
        return x[:, perm_t] * sign_t
    if kind == "scalar":
        return x
    if kind == "leg":
        return x[:, torch.as_tensor(LEG_PERM, device=x.device)]
    if kind == "phase_half_shift":
        # A (sin, cos) clock, mirrored by shifting it half a cycle:
        # sin(2pi(p+0.5)) = -sin(2pi p), and likewise for cos -- so negate both.
        #
        # **Why half a cycle is the right shift, and only for tripod.** The mirror
        # maps LF<->RF, LM<->RM, LR<->RR, so tripod's group A = {LF, RM, LR} maps
        # onto {RF, LM, RR} = group B exactly. The two groups are in antiphase, so
        # swapping them *is* a half-cycle shift.
        #
        # This does **not** generalise to the other gaits, which is why the kind is
        # named for the operation rather than for "phase". Tetrapod's pairs are
        # {LF,RR} / {LM,RM} / {LR,RF}: the mirror swaps the first and third and
        # fixes the second, which is a phase *reversal* (p -> -p, i.e. negate sin
        # and keep cos), not a shift. A ripple clock would need its own rule again.
        # Ripple turned out to need this same rule, derived rather than assumed:
        # its order is LF, RM, LR, RF, LM, RR, so the six legs sit at sequence
        # positions 0..5 and the mirror sends LF(0)<->RF(3), RM(1)<->LM(4),
        # LR(2)<->RR(5) -- i.e. position i -> i+3, which with six slots is exactly
        # half a cycle. Tetrapod does not; see `phase_reflect` below.
        return -x
    if kind == "phase_reflect":
        # A (sin, cos) clock mirrored by **reversing** it: p -> -p, so
        # sin(-2pi p) = -sin(2pi p) and cos(-2pi p) = +cos(2pi p). Negate sin,
        # keep cos.
        #
        # This is tetrapod's, and it is a different operation from the half shift
        # above rather than a variant of it. The pairs are {LF,RR} / {LM,RM} /
        # {LR,RF} at phases 0, 1/3, 2/3. The mirror sends {LF,RR} -> {RF,LR},
        # which is the *third* pair, and {LM,RM} -> {RM,LM}, which is itself. So
        # the three slots map 0->0, 1/3->2/3, 2/3->1/3 -- the fixed middle is what
        # makes it a reflection, and it is why no shift can express it: a shift has
        # no fixed point unless it is the identity.
        #
        # Getting this wrong is silent. The policy would be shown a mirrored clock
        # saying "the diagonal {LF,RR} swings" alongside a mirrored contact pattern
        # showing {LR,RF} airborne, and asked to fit both.
        return x * torch.tensor([-1.0, 1.0], device=x.device, dtype=x.dtype)
    if kind == "legvec3":
        v = x.reshape(x.shape[0], len(LEGS), 3)
        v = v[:, torch.as_tensor(LEG_PERM, device=x.device)]
        v = v * torch.tensor([1.0, -1.0, 1.0], device=x.device)
        return v.reshape(x.shape[0], -1)
    raise ValueError(f"unknown mirror kind {kind!r}")


def _obs_manager(env):
    """Get the observation_manager.

    rsl_rl passes an `RslRlVecEnvWrapper` rather than a raw `ManagerBasedRlEnv`, so
    it has to be unwrapped first. Both are accepted, which also makes this callable
    directly from tests.
    """
    for e in (env, getattr(env, "unwrapped", None), getattr(env, "env", None)):
        if e is not None and hasattr(e, "observation_manager"):
            return e.observation_manager
    raise AttributeError(
        f"no observation_manager found on {type(env).__name__} "
        f"(tried .unwrapped and .env)"
    )


def _history_length(env, group: str, name: str) -> int:
    """How many frames of `name` the group carries, 1 if it has no history.

    Read from the observation manager's own term configs rather than from the
    task's, because the group-level `history_length` **overwrites each term's**
    during `_prepare_terms` -- so the task config and the manager disagree by
    design, and only one of them describes the tensor.

    **A term may also stack its own frames**, in which case `history_length` is 0
    and says nothing about the width. `hexa.posture`'s `StridedHistory` does, to
    span a duration rather than a step count at 200 Hz, and it reports the count
    as `history_frames` on the term object. Asking the func first and the cfg
    second means a term that does neither still reads as one frame, so no other
    task changes behaviour here.
    """
    om = _obs_manager(env)
    cfgs = getattr(om, "_group_obs_term_cfgs", {}).get(group)
    names = om.active_terms[group]
    if not cfgs or name not in names:
        return 1
    cfg = cfgs[names.index(name)]
    if not getattr(cfg, "flatten_history_dim", True):
        raise ValueError(
            f"{group}/{name} keeps its history unflattened; the mirror below "
            f"assumes the flattened [frames x dim] layout"
        )
    # Two `getattr`s, because a term cfg need not carry a `func` at all -- the
    # mirror is exercised against stub cfgs that do not, and reaching through
    # the attribute raised rather than falling back.
    own = int(getattr(getattr(cfg, "func", None), "history_frames", 0) or 0)
    if own and getattr(cfg, "history_length", 0):
        raise ValueError(
            f"{group}/{name} stacks {own} frames itself and the manager stacks "
            f"{cfg.history_length} more; the mirror cannot describe the result "
            f"and the observation is not what either number says"
        )
    return max(1, own or int(getattr(cfg, "history_length", 0) or 1))


def _mirror_group(env: ManagerBasedRlEnv, group: str, x: torch.Tensor) -> torch.Tensor:
    """Mirror a whole observation group. Slice boundaries come from the
    observation_manager, never hardcoded.

    **History is mirrored frame by frame.** With `history_length = n` a term's
    slice is `n` consecutive copies of the observation, oldest first
    (`CircularBuffer.buffer` is `[num_envs, n, dim]` reshaped to `[num_envs,
    n*dim]`). Mirroring that slice as if it were one wide observation is wrong in
    the worst way available here: the permutation for a joint term would shuffle
    values *across* frames, so the mirrored sample would be a robot whose left legs
    are five control steps out of step with its right ones. Nothing raises -- the
    shapes are identical -- and the mirror loss simply teaches the policy something
    untrue.

    So the slice is reshaped back to `[batch, frames, dim]`, mirrored on the last
    axis, and flattened again.
    """
    om = _obs_manager(env)
    names = om.active_terms[group]
    dims = [int(d[0]) for d in om.group_obs_term_dim[group]]
    out, off = [], 0
    cfgs = getattr(om, "_group_obs_term_cfgs", {}).get(group) or []
    params = {n: getattr(c, "params", None) or {} for n, c in zip(names, cfgs)}
    for name, total in zip(names, dims):
        # **A term may declare its own kind**, and the phase clock has to: all
        # three gait tasks call the observation `gait_phase` -- the name is in
        # `scripts/export.py`'s deployable whitelist and the on-robot builder
        # constructs terms by it, so renaming per gait is not available -- while
        # the mirror differs between them. Tripod and ripple are half-shifts,
        # tetrapod is a reflection; see the two branches in `_mirror_term`.
        #
        # Declared in the term's `params`, so it travels with the config that
        # chose the grouping instead of living in module state that the four
        # tasks would overwrite for each other when built in one process.
        kind = params.get(name, {}).get("mirror_kind") or _TERM_KIND.get(name)
        if kind is None:
            raise KeyError(
                f"observation term {name!r} has no mirror kind defined. A new "
                f"observation term must be added to _TERM_KIND, or symmetry "
                f"augmentation silently produces wrong samples."
            )
        frames = _history_length(env, group, name)
        if total % frames:
            raise ValueError(
                f"{group}/{name} is {total} wide over {frames} frames, which does "
                f"not divide -- the history layout is not what this assumes"
            )
        chunk = x[:, off : off + total]
        if frames > 1:
            per = total // frames
            chunk = chunk.reshape(x.shape[0] * frames, per)
            chunk = _mirror_slice(chunk, kind, env).reshape(x.shape[0], total)
        else:
            chunk = _mirror_slice(chunk, kind, env)
        out.append(chunk)
        off += total
    assert off == x.shape[1], (
        f"{group} slices total {off}, which does not match the observation "
        f"dimension {x.shape[1]}"
    )
    return torch.cat(out, dim=1)


def mirror_jumper(
    env: ManagerBasedRlEnv,
    obs: dict | None = None,
    actions: torch.Tensor | None = None,
):
    """rsl_rl's symmetry augmentation function.

    Returns ``[original; mirrored]`` concatenated along the batch dimension, i.e.
    num_aug = 2. Either ``obs`` or ``actions`` may be None (rsl_rl calls it for each
    separately).
    """
    obs_out = None
    if obs is not None:
        mirrored = {
            g: torch.cat([obs[g], _mirror_group(env, g, obs[g])], dim=0)
            for g in obs.keys()
        }
        if hasattr(obs, "batch_size"):
            # TensorDict validates batch_size, so the doubled tensors cannot be
            # put back into the original container in place; a new instance with a
            # doubled batch has to be created.
            n = int(obs.batch_size[0]) * 2
            obs_out = obs.__class__(mirrored, batch_size=[n], device=obs.device)
        else:
            obs_out = mirrored

    act_out = None
    if actions is not None:
        act_out = torch.cat([actions, _mirror_slice(actions, "joint")], dim=0)

    return obs_out, act_out


__all__ = ["JOINT_PERM", "JOINT_SIGN", "LEG_PERM", "mirror_jumper"]
