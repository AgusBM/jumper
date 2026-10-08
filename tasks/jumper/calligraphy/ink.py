"""Where the brush left ink, from a `write.py` log: the data the ink pass paints from.

Steps 3 and 4 of the project read the same thing -- `marks(log)` -- so the ink in
the rendered preview and the ink a compositor paints from `ink.json` are one set
of points.

## What counts as ink

A control step leaves ink when the brush's tip is in contact with the floor
**and** the arm is lowering, writing or lifting. Contact anywhere else would be the
brush brushing the floor on the way somewhere; `write.py` reports it as stray
contact (none in the measured runs) and it is not painted here, so a fault shows
in the report rather than being quietly drawn.

Consecutive inked steps of one stretch form a **mark**: one continuous trace of
the brush. A stroke is one mark when nothing interrupted it, and more where the
robot had to walk in the middle of it (a seam) or the tip left the floor for
longer than `GAP_S`. Shorter gaps are bridged: the simulated tip is a rigid ball
that skips for a step or two (contact on 90-96% of writing steps), where wet hair
would stay on the stone -- unbridged, 无 came out as 71 marks instead of 8. A
bridged step is drawn at the minimum width.

## Width

The brush in the simulation is an 8 mm sphere; a calligraphy brush splays to
several times that when pressed. So the width is not measured but derived:
`width = width_full * clip(force / force_full, MIN_FRACTION, MAX_FRACTION)`,
where `force_full` is the press `write.py` held at full press (`--force`) and
`width_full` defaults to 6.5% of the character's size -- Make Me a Hanzi's own
strokes are 60-90 units of its 1024-unit em. Every point carries the force and the
planned press as well, so the ink pass can choose its own rule.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

#: Phases of `write.py` in which a contact is ink (lower, write, lift).
INK_PHASES = ("lower", "write", "lift")
WIDTH_OF_SIZE = 0.065
MIN_FRACTION, MAX_FRACTION = 0.25, 1.3
#: Gaps in contact up to this long, inside one writing stretch, do not break a mark.
GAP_S = 0.1
#: Ink sits this far above the floor in the preview, so it does not z-fight.
LIFT = 0.0004


@dataclass
class Mark:
    stroke: int
    stretch: int
    t: np.ndarray        # (n,) s
    xyz: np.ndarray      # (n, 3) m, the tip
    force: np.ndarray    # (n,) N, normal
    press: np.ndarray    # (n,) planned press, 0..1
    width: np.ndarray    # (n,) m


class Log:
    def __init__(self, path):
        z = np.load(path)
        self.rows = z["log"]
        self.col = {str(n): i for i, n in enumerate(z["columns"])}
        self.phases = [str(p) for p in z["phases"]]
        self.qpos = z["qpos"] if "qpos" in z.files else None
        self.dt = float(z["dt"]) if "dt" in z.files else 0.02

    def __getitem__(self, name: str) -> np.ndarray:
        return self.rows[:, self.col[name]]


def marks(log: Log, plan, force_full: float, width_full: float | None = None) -> list[Mark]:
    if width_full is None:
        width_full = WIDTH_OF_SIZE * plan.size
    ink_ids = [log.phases.index(p) for p in INK_PHASES]
    in_phase = np.isin(log["phase"], ink_ids)
    inked = (log["contact"] > 0) & in_phase
    stretch = log["stretch"]
    # Bridge short gaps between two inked steps of the same stretch.
    gap = max(1, round(GAP_S / log.dt))
    idx = np.flatnonzero(inked)
    for a, b in itertools.pairwise(idx):
        if 1 < b - a <= gap + 1 and stretch[a] == stretch[b] and in_phase[a:b].all():
            inked[a:b] = True
    out: list[Mark] = []
    i, n = 0, len(inked)
    while i < n:
        if not inked[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and inked[j + 1] and stretch[j + 1] == stretch[i]:
            j += 1
        sel = slice(i, j + 1)
        stroke = int(log["stroke"][i])
        samples = np.clip(log["sample"][sel].astype(int), 0, len(plan.strokes[stroke].xy) - 1)
        force = np.abs(log["fz"][sel])
        width = width_full * np.clip(force / force_full, MIN_FRACTION, MAX_FRACTION)
        out.append(Mark(
            stroke=stroke, stretch=int(stretch[i]), t=log["t"][sel].copy(),
            xyz=np.column_stack([log["px"][sel], log["py"][sel], log["pz"][sel]]),
            force=force, press=plan.strokes[stroke].press[samples], width=width,
        ))
        i = j + 1
    return out


def to_json(ms: list[Mark], plan, shots, force_full: float) -> dict:
    out = {
        "character": plan.character,
        "size_m": plan.size,
        "frame": "world metres; character up = +x, right = -y; floor at z = 0",
        "force_full_N": force_full,
        "width_rule": f"width = {WIDTH_OF_SIZE} * size * clip(force / force_full, "
                      f"{MIN_FRACTION}, {MAX_FRACTION})",
        "cameras": [s.to_json() for s in shots],
        "marks": [],
    }
    for m in ms:
        floor = np.column_stack([m.xyz[:, :2], np.zeros(len(m.t))])
        entry = {
            "stroke": m.stroke, "stretch": m.stretch,
            "t": np.round(m.t, 3).tolist(),
            "xy": np.round(m.xyz[:, :2], 5).tolist(),
            "force_N": np.round(m.force, 3).tolist(),
            "press": np.round(m.press, 3).tolist(),
            "width_m": np.round(m.width, 5).tolist(),
            "pixels": {s.name: np.round(s.project(floor), 2).tolist() for s in shots},
        }
        out["marks"].append(entry)
    return out


def to_svg(ms: list[Mark], plan, px: int = 1000) -> str:
    """The ink from above, +x up, as filled circles along each mark: the
    shape a compositor would lay down, at the character's own scale."""
    half = plan.size / 2 * 1.15
    cx, cy = plan.origin
    k = px / (2 * half)
    parts = [(f'<svg xmlns="http://www.w3.org/2000/svg" width="{px}" height="{px}" '
              f'viewBox="0 0 {px} {px}">'),
             '<rect width="100%" height="100%" fill="#d9d4cb"/>']
    for m in ms:
        u = (-(m.xyz[:, 1] - cy) + half) * k
        v = (-(m.xyz[:, 0] - cx) + half) * k
        r = m.width / 2 * k
        parts.append(f'<g fill="#2b2b2e" data-stroke="{m.stroke + 1}">')
        parts += [f'<circle cx="{a:.1f}" cy="{b:.1f}" r="{c:.1f}"/>' for a, b, c in zip(u, v, r)]
        parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)
