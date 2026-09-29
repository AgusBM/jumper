"""Tools that act on scenes, rather than being one.

A package rather than a loose script under the repository's `tools/`, for the
reason `skygen.py` gives about living in `scenes/`: `scenes` is installed and
`tools` is not, so this is importable from a test and that one is not. The tests
here run the exporter as a library, which is the only way to check what it
refuses as well as what it writes.

`tests/test_scenes.py` globs `scenes/*.py` to find scene modules, so a directory
is also how a tool avoids being mistaken for a scene.
"""
