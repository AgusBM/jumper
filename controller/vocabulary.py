"""The gamepad vocabulary: what a pad can say, for looking up.

`vocabulary.json` beside this file is the dictionary. This reads it and nothing
else -- there is no list in this module, deliberately, because a list here would
be the second copy and the one that goes out of date.

    python -m controller --vocabulary

Consumers are elsewhere and each attaches its own meanings: a deploy manifest's
`[[fsm.button]]`, a task's `controls.yaml`. That direction is this package's
rule, stated at the top of `__init__.py`: nothing here has an opinion about what
a stick means.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

#: The dictionary. Beside this module so `python -m controller` and an editable
#: install find the same bytes, and shipped as package data -- see `pyproject`.
PATH = Path(__file__).resolve().parent / "vocabulary.json"

SCHEMA = "kk-control-vocabulary/2"


@functools.lru_cache(maxsize=1)
def load() -> dict:
    doc = json.loads(PATH.read_text("utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{PATH}: schema is {doc.get('schema')!r}, not {SCHEMA!r}")
    return doc


def buttons() -> tuple[str, ...]:
    """Every button a binding may name, the ten plus the four directions."""
    doc = load()
    return tuple(b["name"] for b in doc["buttons"] + doc["dpad"])


def axes() -> tuple[str, ...]:
    """Every axis, in the order the wire carries them."""
    return tuple(a["name"] for a in load()["axes"])


def keys() -> tuple[str, ...]:
    """Every key a binding may name.

    The numeric keypad, and the letters a task has asked for. MuJoCo's viewer
    calls the user callback **in addition to** its own shortcuts and binds every
    letter, so a letter does two things at once there. `_keys_note` in the
    dictionary says which, and why a key there is a press and nothing else.
    """
    return tuple(k["name"] for k in load()["keys"])


def key_codes() -> dict[str, dict]:
    """`{name: {"glfw": int, "browser": str}}`, for a host that reads one of
    those and has to arrive at the contract's name."""
    return {k["name"]: {"glfw": k["glfw"], "browser": k["browser"]}
            for k in load()["keys"]}


def modifiers() -> dict[str, tuple[str, ...]]:
    """`{name: keys}`: a modifier is down while either of its keys is.

    `ctrl`, `shift` and `alt`, each the left and the right key. A keyboard
    binding names one on its own or in front of a key -- `shift+key_j` -- and a
    mode switch names one as its `with`. `_modifiers_note` has the rule.
    """
    return {m["name"]: tuple(m["keys"]) for m in load()["modifiers"]}


def keystroke(spec: str) -> tuple[str | None, str]:
    """`"shift+key_j"` -> `("shift", "key_j")`; `"key_j"` -> `(None, "key_j")`;
    `"ctrl"` -> `(None, "ctrl")`.

    What a keyboard binding names: a key, a modifier on its own, or a modifier
    held with a key. Raises for anything else, naming what it is not -- a key
    nobody reports binds nothing, and would look exactly like one that did.
    """
    mods = modifiers()
    known = frozenset(keys()) | frozenset(mods)
    # Whole names first: `keypad_+` is a key whose name ends in a plus, not a
    # chord. A modifier's name has no plus, so a chord splits at the first one.
    if spec in known:
        return None, spec
    modifier, _, key = spec.partition("+")
    held = {k for ks in mods.values() for k in ks}
    if modifier in mods and key in known and key not in mods and key not in held:
        return modifier, key
    raise ValueError(
        f"{spec!r} is not a key, a modifier ({', '.join(mods)}), or a modifier and a key "
        f"(`shift+key_j`). The keys are {' '.join(keys())}"
    )


def gestures() -> tuple[str, ...]:
    return tuple(g["name"] for g in load()["gestures"])


def absent() -> dict[str, str]:
    """Controls the pad has and the service does not publish, and why.

    Kept in the dictionary rather than left out of it: a name that is missing
    reads as an oversight, and `view` is a decision.
    """
    return {a["name"]: a["why"] for a in load()["absent"]}


def describe() -> str:
    """The whole thing, rendered. What `--vocabulary` prints."""
    doc = load()
    out = ["buttons  (a `pad = ` in [[fsm.button]])"]
    out.append("  " + " ".join(b["name"] for b in doc["buttons"]))
    out.append("  " + " ".join(b["name"] for b in doc["dpad"]))
    for name, why in absent().items():
        out.append(f"  not `{name}`: {why}")
    out.append("")
    out.append("keys  (a `key = ` in [[fsm.button]])")
    doc_keys = doc["keys"]
    out.append("  " + " ".join(k["name"] for k in doc_keys))
    # One line per contiguous run of GLFW codes. The keypad is one run and the
    # letters are not -- a single `first-last` span printed "glfw 320-70" once
    # they were added, which reads as a range and is not one.
    runs, start = [], 0
    for i in range(1, len(doc_keys) + 1):
        if i == len(doc_keys) or doc_keys[i]["glfw"] != doc_keys[i - 1]["glfw"] + 1:
            a, b = doc_keys[start], doc_keys[i - 1]
            runs.append(
                f"glfw {a['glfw']}" + (f"-{b['glfw']}" if b is not a else "")
                + f" browser {a['browser']}" + (f"..{b['browser']}" if b is not a else "")
            )
            start = i
    out.append("  " + "; ".join(runs) if len(runs) <= 3 else
               f"  {len(runs)} runs of codes; see `controller/vocabulary.json`")
    out.append("  " + "\n  ".join(doc["_keys_note"][:5]))
    out.append("")
    out.append("modifiers  (`with = ` on a key switch; `shift+key_j` in a controls.yaml)")
    out.append("  " + "  ".join(f"{m}: {' '.join(ks)}" for m, ks in modifiers().items()))
    out.append("")
    out.append("axes  (a `source = ` in a task's controls.yaml)")
    for a in doc["axes"]:
        lo, hi = a["range"]
        out.append(f"  {a['name']:<8}[{lo}, {hi}]" + (f"  {a['note']}" if "note" in a else ""))
    out.append("  " + doc["_axes_note"])
    out.append("")
    out.append("gestures  (an `on = ` in [[fsm.button]])")
    for g in doc["gestures"]:
        out.append(f"  {g['name']:<10}{g['fires']}")
    out.append("  " + doc["_gestures_note"])

    unresolved = [(b["name"], b["unresolved"]) for b in doc["buttons"] if "unresolved" in b]
    if unresolved:
        out.append("")
        out.append("unresolved")
        for name, why in unresolved:
            out.append(f"  {name}: {why}")
    return "\n".join(out) + "\n"
