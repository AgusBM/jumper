"""A key registry, so code outside the viewer can react to keys going down and up.

`mujoco.viewer.launch_passive` takes a single `key_callback`, and the viewer is
built in `scripts/_cli.py`, which knows nothing about tasks and must not have to.
This is the seam between the two: `LiveViewer` hands the viewer `from_viewer`, and
anything that wants keys registers a handler -- typically a task's teleoperation
command term, during environment construction, long before the viewer exists.

**The framework stays robot-agnostic**, exactly as it does for the camera and the
collision groups: nothing here knows what a key *means*. A handler receives a raw
GLFW keycode and whether it went down or came up, and decides for itself; which
key does what belongs to the task.

## Both edges, which MuJoCo's viewer does not give

`key_callback` is called for a key's **press and nothing else**: MuJoCo's GLFW
adapter counts only `GLFW_PRESS` as a key event (`GlfwAdapter::IsKeyDownEvent`,
unchanged from 3.1 to 3.11), so a held key's auto-repeat and its release never
reach Python. A held key and a tap arrive as the same single call. Until
2026-09-29 the readers of this registry inferred the release from the silence
after the repeats -- repeats that never came -- so every key was a tap held
0.75 s, and a key that should push a stick all the way reached 37.5% of it.
MEASURED: one call for W held 1.5 s under X11, with the server repeating it
33 times a second.

So `from_viewer`, on the first press a window reports, puts a key callback of its
own in front of MuJoCo's on that GLFW window -- the viewer's thread is inside
GLFW's event processing then, so the window is the current context -- and from
then on reports each key's press and release itself. MuJoCo's callback still runs
first, for its shortcuts. GLFW is the library `mujoco.viewer` handed its
`simulate` (`set_glfw_dlhandle(glfw._glfw._handle)`), so this is the same window,
not a second library's idea of one. Should no window be current, it says so once
and reports each press as a tap: a handler then sees a button click and a stick
that does not move, rather than a key stuck down.

## The viewer's own shortcuts still fire

MuJoCo's built-in bindings are handled in the C++ `Simulate` class and the user
callback is invoked **in addition to** them, not instead. So a handler that claims
a key the viewer already uses gets both behaviours at once -- press `W` for
"walk forward" and the scene also flips to wireframe. There is no way to intercept
that from here, which is why the key *choice* matters and why it is left to the
task. `mjrl.viewer.keys.KEYPAD` below is the safe set.

## Handlers must not raise

A handler runs on the viewer's UI thread. An exception there would surface far
from its cause, and in the worst case take the render loop with it, so `dispatch`
logs and swallows. A handler that needs to report a problem should do it through
its own state, not by raising.

## When a key does nothing

Set `MJRL_KEY_DEBUG=1` and every keycode that arrives is printed, down or up, with
the number of handlers registered. That separates the two failures that look identical from
the outside:

- **nothing printed** -- the key never reached the process. The window did not have
  focus, or the viewer went headless, or the press went to the terminal instead of
  the render window.
- **printed, but nothing happened** -- the key arrived under a different code than
  the keymap expects (a different layout, or the keypad reporting as something
  else), and the fix is to remap to the code that actually shows up.
"""

from __future__ import annotations

import ctypes
import os
import threading
from typing import Callable

__all__ = [
    "KEYPAD", "claim", "dispatch", "from_viewer", "handler_count", "register", "unregister",
]

#: GLFW keycodes for the numeric keypad, which MuJoCo's viewer does not bind.
#: Provided as a convenience for handlers that want keys guaranteed not to also
#: trigger a viewer shortcut; using it is entirely up to the handler.
KEYPAD = {
    "0": 320, "1": 321, "2": 322, "3": 323, "4": 324,
    "5": 325, "6": 326, "7": 327, "8": 328, "9": 329,
    ".": 330, "/": 331, "*": 332, "-": 333, "+": 334, "enter": 335,
}

#: A handler: a GLFW keycode, and True as it goes down, False as it comes up.
Handler = Callable[[int, bool], None]

_lock = threading.Lock()
_handlers: list[Handler] = []


def register(handler: Handler) -> None:
    """Add a key handler. Registering the same handler twice is a no-op, so a
    command term rebuilt on a second `env` does not get its keys doubled."""
    with _lock:
        if handler not in _handlers:
            _handlers.append(handler)


def unregister(handler: Handler) -> None:
    """Remove a handler; unknown handlers are ignored."""
    with _lock:
        if handler in _handlers:
            _handlers.remove(handler)


def claim(handler: Handler) -> int:
    """Make `handler` the only one, and return how many were set aside.

    For a run in which one reader of the person is the whole of it: `play
    --app` drives the robot through the deployment controller, which reads
    every key, and a task's own teleop term -- registered when the environment
    was built -- would read the same keys into a command nothing uses, and say
    so on the terminal. Set aside rather than handed back: nothing restores
    them, because nothing in that run wants them.
    """
    with _lock:
        others = [h for h in _handlers if h is not handler]
        _handlers[:] = [handler]
    return len(others)


def handler_count() -> int:
    """How many handlers are registered. For tests and for the viewer's banner."""
    with _lock:
        return len(_handlers)


def dispatch(keycode: int, down: bool) -> None:
    """Deliver a key going down or coming up to every handler. Never raises."""
    with _lock:
        handlers = list(_handlers)
    if os.environ.get("MJRL_KEY_DEBUG"):
        # Printed before the handlers run, so a keycode that arrives is visible
        # even if every handler ignores it -- which is the case being diagnosed.
        edge = "down" if down else "up"
        print(f"[mjrl] key {keycode} {edge} ({len(handlers)} handler(s))", flush=True)
    for handler in handlers:
        try:
            handler(keycode, down)
        except Exception as exc:  # noqa: BLE001 - see the module docstring
            print(f"[mjrl] key handler {handler!r} raised on keycode {keycode}: {exc}")


# ── The viewer's side ────────────────────────────────────────────────────────

#: GLFW's key actions. A repeat is neither edge and is not passed on.
_RELEASE, _PRESS = 0, 1

#: `GLFWkeyfun`: window, key, scancode, action, mods.
_KEYFUN = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                           ctypes.c_int)

#: Per window, the key callback that was there before ours -- MuJoCo's -- which
#: ours calls first. Touched on the viewer's thread only.
_previous: dict[int, object] = {}
_warned = False


class _Glfw:
    """The two GLFW calls the hook needs, bound to the library MuJoCo's viewer
    runs on. Their own prototypes rather than pyGLFW's wrappers, whose argument
    types expect windows pyGLFW made. A test replaces `_glfw`."""

    def __init__(self) -> None:
        import glfw

        lib = glfw._glfw  # the handle `mujoco.viewer` passed to `set_glfw_dlhandle`
        self.current = ctypes.CFUNCTYPE(ctypes.c_void_p)(("glfwGetCurrentContext", lib))
        self.set_key_callback = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("glfwSetKeyCallback", lib))


_glfw: _Glfw | None = None


def _on_glfw_key(window, key, scancode, action, mods) -> None:
    previous = _previous.get(window)
    if previous is not None:
        previous(window, key, scancode, action, mods)
    if action == _PRESS:
        dispatch(key, True)
    elif action == _RELEASE:
        dispatch(key, False)


#: Kept for as long as the process: GLFW holds a raw pointer to it.
_ON_GLFW_KEY = _KEYFUN(_on_glfw_key)
_ON_GLFW_KEY_ADDRESS = ctypes.cast(_ON_GLFW_KEY, ctypes.c_void_p).value


def from_viewer(keycode: int) -> None:
    """The `key_callback` to hand `mujoco.viewer.launch_passive`.

    Called on the viewer's thread for a press only; see the module docstring.
    The first press on a window puts `_on_glfw_key` in front of the viewer's own
    callback and reports this press, whose release the hook will see. Every
    press after that comes through the hook, which reports it itself, so here it
    is dropped. Setting the callback is also how a window is told apart from one
    already hooked: GLFW hands back the one it replaces, which is ours on a
    hooked window and MuJoCo's on a new one -- even one opened at the address a
    closed viewer's window had.
    """
    global _glfw, _warned
    try:
        if _glfw is None:
            _glfw = _Glfw()
        window = _glfw.current()
    except Exception as exc:  # noqa: BLE001 - see "Handlers must not raise"
        window, reason = None, f"GLFW could not be reached ({exc})"
    else:
        reason = "no GLFW window is current on the viewer's thread"
    if not window:
        if not _warned:
            _warned = True
            print(f"[mjrl] keys: {reason}, so a key's release cannot be seen and each "
                  "press is reported as a tap -- a button clicks, a stick does not move")
        dispatch(keycode, True)
        dispatch(keycode, False)
        return
    replaced = _glfw.set_key_callback(window, _ON_GLFW_KEY_ADDRESS)
    if replaced == _ON_GLFW_KEY_ADDRESS:
        return  # through the hook, which reports this press itself
    _previous[window] = _KEYFUN(replaced) if replaced else None
    dispatch(keycode, True)
