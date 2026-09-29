"""Simulation backends.

Only `resolve` (backend/device resolution) is exported here. `native_sim` and
`model_slim` are parts used by the native path and are imported from their own
modules on demand -- re-exporting them here would make `import mjrl.backend` pull
in mujoco and torch.

There is no separate backend protocol: the seam is mjlab's `Simulation` class
being replaceable. See section 3 of docs/DESIGN.md.
"""

from .resolve import BackendUnavailable, Resolution, resolve

__all__ = ["Resolution", "resolve", "BackendUnavailable"]
