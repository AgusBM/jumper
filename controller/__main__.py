"""Live readout of the connected pad: `python -m controller`.

The one thing no test on this side can settle is whether pushing a stick forward
reads as forward. Everything else about a controller can be checked from synthetic
values, but the mapping from a physical direction to a number needs a hand on the
pad -- so this exists to make that a five-second check rather than a guess, and to
be the first thing to run when a pad "does nothing".

    python -m controller               # live values
    python -m controller --raw         # every (type, code) the pad emits
    python -m controller --vocabulary  # the words a binding may use

`--raw` is for a pad whose layout `xbox.py` does not know: press the control, see
which code moves, and add it there.
"""

from __future__ import annotations

import sys
import time

import controller
from controller.xbox import layout_of


def main() -> int:
    if "--vocabulary" in sys.argv:
        # No pad needed: this is the dictionary, not a reading of one.
        from . import vocabulary

        print(vocabulary.describe(), end="")
        return 0
    raw_mode = "--raw" in sys.argv

    print(controller.describe())
    pad = controller.open()
    if pad is None:
        print("\nnothing to read. If a pad is listed above as unusable, the reason "
              "is on that line.")
        return 1

    layout = layout_of(pad._ranges)
    print(f"\n{pad.name}: {layout.name} layout, "
          f"axes {sorted(pad._ranges)}")
    print("\nPush each control and watch the numbers. Ctrl-C to stop.")
    print("Expected: left stick up -> ly negative (evdev's Y is positive down),")
    print("          left stick left -> lx negative, right stick right -> rx positive\n")

    try:
        while True:
            if raw_mode:
                items = sorted(pad.raw().items())
                line = " ".join(f"{t:x}:{c:x}={v}" for (t, c), v in items if v)
            else:
                s = pad.state()
                pressed = " ".join(sorted(s.buttons)) or "-"
                line = (
                    f"L({s.lx:+.2f},{s.ly:+.2f})  R({s.rx:+.2f},{s.ry:+.2f})  "
                    f"LT {s.lt:.2f} RT {s.rt:.2f}  "
                    f"hat({s.hat_x:+d},{s.hat_y:+d})  [{pressed}]"
                )
            print(f"\r{line:<100}", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        pad.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
