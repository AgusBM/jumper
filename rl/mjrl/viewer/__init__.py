"""Visualisation.

`live.LiveViewer` -- watch while training. **Both backends are supported**:
taking a frame is a host copy on native and one device-to-host transfer on warp,
and drawing it happens on a thread of the viewer's own. The difference is cost,
not capability; see the head of that module.

`keys` -- the seam that lets code outside the viewer receive key presses, without
this package having to know what any key means. A task registers a handler; the
viewer dispatches raw keycodes to it.
"""
