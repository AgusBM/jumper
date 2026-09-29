#!/usr/bin/env python3
"""Read-only preflight: does this board match what we are about to build for it?

Every mismatch this looks for is one that **does not raise on the robot**. A
CycloneDDS soname that differs makes the binary fail to load; an IDL that differs
makes the DDS reader never pair, leaving the topic empty forever; a joint order
that differs drives the wrong joints. None of them announce themselves, and the
last one moves a real machine.

Standard library plus `ssh` only, so it runs without this skill and without the
repository's virtualenv:

    python3 .claude/skills/deploy/scripts/check_board.py --host <user>@<board>
    python3 .claude/skills/deploy/scripts/check_board.py --host <user>@<board> \\
        --bundle tasks/jumper/tripod/out/<name>

`--host` has no default: a board's address belongs to the network it is on, not
to this repository.

**Read-only. It runs no binary on the board and publishes nothing.** The robot's
own controller usually runs while this does, and two controllers on one motor bus
is not a thing to find out about experimentally.

Exit status: 0 all checks passed, 1 a check failed, 2 could not reach the board.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

#: What this repository builds against. Each is compared, not assumed.
EXPECT_RKNN = "2.3.2"          # deploy/fsm/vendor/rknpu2 and convert/requirements.txt
EXPECT_DDS_SONAME = "libddsc.so.11"


class Result:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool | None, str]] = []

    def add(self, name: str, ok: bool | None, detail: str) -> None:
        self.rows.append((name, ok, detail))

    def report(self) -> int:
        width = max(len(n) for n, _, _ in self.rows)
        for name, ok, detail in self.rows:
            mark = {True: "ok  ", False: "FAIL", None: "--  "}[ok]
            print(f"  [{mark}] {name:<{width}}  {detail}")
        failed = [n for n, ok, _ in self.rows if ok is False]
        skipped = [n for n, ok, _ in self.rows if ok is None]
        print()
        if failed:
            print(f"FAILED: {', '.join(failed)}")
            return 1
        print("all checks passed" + (f" ({len(skipped)} skipped)" if skipped else ""))
        return 0


def ssh(host: str, command: str, timeout: int = 30) -> tuple[int, str]:
    """One read-only command on the board."""
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout.strip()


def check_reachable(host: str) -> str | None:
    code, out = ssh(host, "uname -srm; . /etc/os-release 2>/dev/null && echo $VERSION")
    if code != 0:
        return None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", required=True, help="ssh target, <user>@<board>")
    ap.add_argument(
        "--bundle",
        type=Path,
        help="an exported policy to check the board's joint order against "
             "(the directory holding layout.json)",
    )
    ap.add_argument("--config", default=None,
                    help="a file on the board holding [robot] joint_names, to check the "
                         "bundle's wire order against. There is no default: the path this "
                         "had belonged to a controller that is not part of the product, "
                         "and a default pointing at it made the check look like a step "
                         "rather than a thing you can do if you happen to have such a file")
    args = ap.parse_args()

    if shutil.which("ssh") is None:
        print("ssh not found", file=sys.stderr)
        return 2

    print(f"board: {args.host}\n")
    banner = check_reachable(args.host)
    if banner is None:
        print(f"cannot reach {args.host} over ssh (key auth, BatchMode).", file=sys.stderr)
        print("  Check the address, the route, and that your key is authorised.", file=sys.stderr)
        return 2

    r = Result()
    r.add("reachable", True, banner.replace("\n", "  "))

    # ── glibc: a floor, not a ceiling ──────────────────────────────────
    # A binary built against a newer glibc than the board's dies at startup
    # with "version GLIBC_2.xx not found", nowhere near the build that caused it.
    _, out = ssh(args.host, "ldd --version | head -1")
    m = re.search(r"(\d+\.\d+)\s*$", out)
    r.add("glibc", m is not None, f"{m.group(1) if m else out}  (fsm/Dockerfile builds on 22.04 = 2.35)")

    # ── The NPU runtime must be at least as new as the toolkit that built
    #    the .rknn. Older refuses to load the model.
    _, out = ssh(args.host, "grep -a -o 'librknnrt version: [0-9.]*' /usr/lib/librknnrt.so | head -1")
    got = out.split(":")[-1].strip() if ":" in out else ""
    r.add("librknnrt", bool(got) and got >= EXPECT_RKNN,
          f"{got or '<not found>'}  (need >= {EXPECT_RKNN}, what convert/ builds with)")

    # ── The soname the binary was linked against has to exist on the board.
    _, out = ssh(args.host, "ls /usr/lib/libddsc.so.* 2>/dev/null | head -3")
    r.add("cyclonedds", EXPECT_DDS_SONAME in out, f"{out.replace(chr(10), ' ') or '<none>'}")

    _, out = ssh(args.host, "dpkg -l 2>/dev/null | awk '/libmbus/{print $3}'")
    r.add("libmbus", bool(out), out or "<not installed>")

    # ── The IDL is the wire contract. Compared by substance: idlc stamps the
    #    source path into the header comment and the include guard, so those
    #    differ between any two builds and mean nothing.
    ours = REPO / "deploy" / "dds" / "idl"
    r.add(*check_idl(args.host, ours))

    # ── Joint order. The one that moves the robot if it is wrong.
    if args.bundle and args.config:
        r.add(*check_joints(args.host, args.config, args.bundle))
    elif args.bundle:
        r.add("joint order", None, "pass --config <file on the board> to check it")
    else:
        r.add("joint order", None, "pass --bundle <dir> and --config <file> to check it")

    print()
    return r.report()


def check_idl(host: str, ours: Path) -> tuple[str, bool | None, str]:
    """Compare the board's generated headers with ours, ignoring path noise.

    The repository ships `.idl`, the board ships the `.h` its middleware was
    built from, so ours are generated here with `idlc` and the two compared.
    `idlc` stamps the source path into both the header comment and the include
    guard, and those differ between any two builds -- filtering them is what
    makes this comparison mean "same types" rather than "same file".
    """
    def strip(text: str) -> list[str]:
        skip = re.compile(r"File name:|Source:|DDSC_|Cyclone DDS IDL compiler")
        return [ln for ln in text.splitlines() if ln.strip() and not skip.search(ln)]

    names = ["idl_common", "idl_imu", "idl_motor_control", "idl_robot_control"]
    if not ours.is_dir():
        return "IDL types", None, f"{ours} not found"
    if shutil.which("idlc") is None:
        return ("IDL types", None,
                "idlc not on PATH (it ships with CycloneDDS); cannot generate ours to compare")

    with tempfile.TemporaryDirectory(prefix="mjrl-idl-") as tmp:
        tmpdir = Path(tmp)
        for f in ours.glob("*.idl"):
            shutil.copy(f, tmpdir / f.name)
        differing, absent = [], []
        for n in names:
            gen = subprocess.run(["idlc", f"{n}.idl"], cwd=tmpdir,
                                 capture_output=True, text=True)
            if gen.returncode != 0 or not (tmpdir / f"{n}.h").is_file():
                return "IDL types", None, f"idlc failed on {n}.idl"
            code, board = ssh(host, f"cat /usr/include/{n}.h 2>/dev/null")
            if code != 0 or not board:
                absent.append(n)
                continue
            if strip(board) != strip((tmpdir / f"{n}.h").read_text()):
                differing.append(n)

    if absent:
        return "IDL types", None, f"not on the board: {', '.join(absent)}"
    if differing:
        return ("IDL types", False,
                f"differ: {', '.join(differing)} -- the DDS reader will never pair, "
                f"and the topic stays empty rather than erroring")
    return "IDL types", True, f"{len(names)} headers identical in substance"


def check_joints(host: str, config: str, bundle: Path) -> tuple[str, bool, str]:
    """The board's wire order against the bundle's, by name."""
    layout_path = next(
        (bundle / n for n in ("layout.json", "isaac_layout.json", "contract.json")
         if (bundle / n).is_file()),
        None,
    )
    if layout_path is None:
        return "joint order", None, f"no layout.json in {bundle}"

    code, out = ssh(host, f"sed -n '/^joint_names = \\[/,/^\\]/p' {config}")
    board = re.findall(r'"([^"]+)"', out)
    if code != 0 or not board:
        return "joint order", None, f"no [robot] joint_names in {config}"

    layout = json.loads(layout_path.read_text())
    wire = layout.get("wire_joint_order")
    if wire is None:
        # Bundles from before wire_joint_order existed key the home pose by it.
        wire = list(layout.get("default_joint_pos", {}))
    if board != wire:
        for i, (a, b) in enumerate(zip(board, wire)):
            if a != b:
                return ("joint order", False,
                        f"differ at index {i}: board={a} contract={b}")
        return ("joint order", False,
                f"different lengths: board {len(board)}, contract {len(wire)}")

    index = {n: i for i, n in enumerate(board)}
    missing = [n for n in layout["obs_joint_order"] + layout["action_joint_order"]
               if n not in index]
    if missing:
        return "joint order", False, f"not on the board: {missing}"
    uncovered = [n for n in board if n not in layout.get("default_joint_pos", {})]
    if uncovered:
        return ("joint order", False,
                f"default_joint_pos misses {uncovered}; every wire joint needs a home")
    obs_idx = [index[n] for n in layout["obs_joint_order"]]
    gaps = [i for i in range(len(board)) if i not in obs_idx]
    return ("joint order", True,
            f"{len(board)} joints, same order; observation covers {len(obs_idx)}, "
            f"skipping wire {gaps} -- resolved by name")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.TimeoutExpired:
        print("timed out talking to the board", file=sys.stderr)
        sys.exit(2)
