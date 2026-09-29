"""Start TensorBoard alongside training and point it at this run's logs.

Training already writes TensorBoard event files -- `rsl_rl`'s `Logger` builds a
`SummaryWriter` on `log_dir`, and `WandbLogWriter` **subclasses** it, so the
files are written under either `--logger`. What was missing was that starting
the server, finding the run directory and typing the URL were all manual, and
the directory is four segments deep (`logs/<model>/<task>/<timestamp>`), so the
usual mistake is to point TensorBoard one level too high or too low and read the
wrong run's curves.

Three things this deliberately does not do:

- **It does not bind to 0.0.0.0.** Training metrics would then be readable by
  anything on the network. Over ssh, forward the port instead:
  `ssh -L 6006:localhost:6006 <host>`.
- **It does not fail the run.** A server that will not start is a lost
  convenience, not a lost experiment, so every failure here degrades to a
  printed reason (the same rule the live viewer follows).
- **It does not claim success it has not checked.** The process is polled until
  the port accepts a connection; if it dies first, the reason is printed from
  its own log. A dead TensorBoard whose URL was printed anyway is the failure
  mode worth spending code on -- the browser shows a connection error and it
  looks like a network problem rather than a process that exited.
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

#: Where TensorBoard's own stdout/stderr goes, inside the run directory. Kept
#: out of the training log (its startup chatter would interleave with the first
#: iterations) but on disk, because a launch failure is unreadable without it.
LOG_NAME = "tensorboard.log"

#: How many consecutive ports to try from the requested one. A busy 6006 is the
#: normal case -- a previous run's server, or a colleague on a shared box -- and
#: refusing to start over it would make the feature useless exactly when a
#: second run is what you want to compare against.
_PORT_SEARCH = 20

#: How long to wait for the port to start accepting connections. TensorBoard
#: takes 1-3 s to import and bind; 15 s leaves room on a loaded machine without
#: stalling training noticeably if something is wrong.
_READY_TIMEOUT = 15.0


def resolve_logdir(log_dir: Path, scope: str) -> Path:
    """The directory to serve, given **this run's** log directory.

    `log_dir` is `logs/<model>/<task>/<timestamp>`, so the scopes are just how
    far up to walk:

    - `run`  -> that directory. One run, fastest to load.
    - `task` -> `logs/<model>/<task>`, every run of this task against this
      model. **The default**: TensorBoard names each subdirectory as a run, so
      the timestamps become the series labels and the previous run is right
      there to compare against, which is most of the reason to open it at all.
    - `all`  -> `logs/`, every model and task. Useful occasionally, slow once
      the tree is large.

    Walking up by position rather than by name because the log root is
    **relative to the working directory** (see `tasks.paths.log_root_for`), so
    there is no absolute path to compare against. `test_tensorboard.py` ties
    these three back to a path built by that function, so a change to the layout
    fails there instead of quietly serving the wrong level.
    """
    scopes = {"run": log_dir, "task": log_dir.parent, "all": log_dir.parents[2]}
    if scope not in scopes:
        raise ValueError(f"unknown scope {scope!r}, expected one of {sorted(scopes)}")
    return scopes[scope]


def _free_port(preferred: int) -> int | None:
    """The first port from `preferred` that nothing is listening on.

    `SO_REUSEADDR` is what makes this probe agree with the server it is probing
    for. Without it the answer is not "can TensorBoard bind here" but "is there
    a lingering TIME_WAIT entry", and there always is: the browser tab from the
    last run held a connection open, so terminating that server left its port in
    TIME_WAIT for a couple of minutes. The probe would skip a perfectly usable
    6006, and the URL would climb 6006 -> 6007 -> 6008 run after run -- the one
    thing that makes a bookmark useless. Python's `socketserver` sets the same
    option (`allow_reuse_address`), so this asks the question the way the server
    will answer it.

    It does **not** weaken the check that matters: SO_REUSEADDR never lets two
    sockets listen on one port (that is SO_REUSEPORT), so a port someone is
    actually serving on is still reported as taken.
    """
    for port in range(preferred, preferred + _PORT_SEARCH):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    return None


def _port_is_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.25)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _can_open_browser() -> bool:
    """Whether opening a browser **on this machine** would show it to anyone.

    Over ssh it would not: the page would open on the remote console, where
    nobody is looking. That check comes first because on a remote macOS box
    `sys.platform` alone says yes.
    """
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False
    if sys.platform == "darwin" or sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@contextlib.contextmanager
def sigterm_runs_finally():
    """Turn SIGTERM into a normal unwind for the duration of the block.

    The TensorBoard server is a **child process**, so it outlives anything that
    stops this one without running its `finally` blocks -- and plain
    `kill <pid>` is exactly that, which is how background runs are stopped here.
    The orphan then keeps port 6006 and keeps serving the *previous* run, so the
    bookmark opens on stale curves that look like a training that stopped
    improving. That is worse than no server at all.

    Ctrl-C never had the problem (SIGINT goes to the whole foreground process
    group, so the child gets it too); this makes SIGTERM behave the same way.
    A `kill -9`, or the OOM killer, still leaks -- nothing in-process can help
    there, and `pgrep -f tensorboard.main` finds it.
    """
    import signal

    def _exit(signum, _frame):
        raise SystemExit(128 + signum)

    try:
        previous = signal.signal(signal.SIGTERM, _exit)
    except ValueError:  # not the main thread; leave the disposition alone
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


class TensorBoard:
    """A TensorBoard server tied to the lifetime of one training run."""

    def __init__(self, logdir: Path, port: int, log_file: Path) -> None:
        self.logdir = logdir
        self.port = port
        self.log_file = log_file
        self.url = f"http://localhost:{port}/"
        self._proc: subprocess.Popen | None = None
        self._log_handle = None

    def start(self) -> TensorBoard:
        """Launch the server and wait until it answers on the port."""
        self._log_handle = open(self.log_file, "w", encoding="utf-8")
        # `sys.executable -m` rather than the `tensorboard` console script: the
        # script is only on PATH when the virtualenv is activated, and training
        # is routinely started as `.venv/bin/python scripts/train.py` without
        # activating anything. This way the server is guaranteed to be the one
        # from the same environment as the writer.
        self._proc = subprocess.Popen(
            [
                sys.executable, "-m", "tensorboard.main",
                "--logdir", str(self.logdir),
                "--port", str(self.port),
                "--host", "127.0.0.1",
            ],
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        self._wait_until_ready()
        return self

    def _wait_until_ready(self) -> None:
        """Block until **this** server is listening, or explain why it is not.

        Readiness is the server's own announcement in its log, not merely an
        open port. An open port answers a different question -- "is anything
        here" -- and something else answering on it is precisely the case that
        must not pass: a port that another process grabbed between the probe and
        the launch would be reported ready while this server was still on its
        way to exiting with "Address already in use", and the URL handed out
        would point at a stranger's server.
        """
        deadline = time.monotonic() + _READY_TIMEOUT
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"tensorboard exited with code {self._proc.returncode}: "
                    f"{self._tail()}"
                )
            if self._announced_its_url() and _port_is_open(self.port):
                return
            time.sleep(0.2)
        raise RuntimeError(
            f"tensorboard did not announce port {self.port} within "
            f"{_READY_TIMEOUT:.0f}s: {self._tail()}"
        )

    def _announced_its_url(self) -> bool:
        """Whether the server has printed that it bound our address.

        It writes `TensorBoard <version> at http://127.0.0.1:<port>/` once it is
        serving. Matching the address makes this specific to the process
        launched here -- the point of the check.
        """
        try:
            if self._log_handle is not None:
                self._log_handle.flush()
            text = self.log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return f"http://127.0.0.1:{self.port}" in text

    def _tail(self, lines: int = 3) -> str:
        """The last few lines of its log, for the failure message."""
        try:
            if self._log_handle is not None:
                self._log_handle.flush()
            text = self.log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return f"see {self.log_file}"
        tail = [ln for ln in text.splitlines() if ln.strip()][-lines:]
        return " | ".join(tail) if tail else f"no output, see {self.log_file}"

    def open_browser(self) -> None:
        """Open the page, if there is anyone here to see it."""
        if not _can_open_browser():
            return
        import webbrowser

        try:
            webbrowser.open(self.url)
        except Exception:  # noqa: BLE001,S110 - a browser is never worth an error
            pass

    def stop(self) -> None:
        """Shut the server down. Never raises -- this runs on the way out."""
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:  # noqa: S110
                pass
            self._log_handle = None


def start_tensorboard(log_dir: Path, scope: str = "task", port: int = 6006) -> TensorBoard:
    """Serve `log_dir`'s `scope` on the first free port at or after `port`.

    Raises on failure; the caller decides whether that is fatal (in this
    repository it never is -- see `maybe_tensorboard` in scripts/_cli.py).
    """
    logdir = resolve_logdir(log_dir, scope)
    chosen = _free_port(port)
    if chosen is None:
        raise RuntimeError(
            f"no free port in {port}..{port + _PORT_SEARCH - 1}; pass --tb-port"
        )
    return TensorBoard(logdir, chosen, log_dir / LOG_NAME).start()


__all__ = [
    "LOG_NAME",
    "TensorBoard",
    "resolve_logdir",
    "sigterm_runs_finally",
    "start_tensorboard",
]
