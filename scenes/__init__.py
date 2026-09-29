"""Scenes: the ground, the sky and the light.

A scene is the other axis from a task. A task decides what the robot is asked to
do and what it observes; a scene decides what the world looks like and, in one
case, what the ground is. Keeping them apart means the same gait task can be run
under a neutral studio light for a screenshot and a sunset for a video without
touching the task, and the same look can be reused across every task.

Pick one with `--scene`, or set `MJRL_SCENE` in `.env`. Passing neither leaves the
task's own configuration untouched, which is the previous behaviour exactly.

Registration is metadata only -- **do not import the scene modules here**. This
file runs during `import scenes`, and `--list` has to stay usable on a machine
without the simulation dependencies. `registry.load` imports the module.

`--scene` also takes a **path** to an imported `kk-scene-package/1` or `/2` --
a directory, a `.zip` or a `.map` that scene-generator, kingkong_design's map
library or this repository's own `scenes/tools/export_web_scene.py` wrote.
`scripts/play.py` accepts one and `scripts/train.py` refuses it;
`scenes/package.py` says why.

To add a scene: write `scenes/<id>.py` with a `scene()` factory and add one
`register` call below. `tests/test_scenes.py` checks that every module has an
entry and that every entry loads.

Every entry also has to say what it is **for** -- `use=` -- because that decides
whether the scene is exported for someone outside this repository to walk around
in. Some of these are places; some are instruments that exist to make training
work or to hold a variable still, and handing one of those to a visitor is
offering a measuring device as a playground. The comments beside the three
`use="training"` entries below are the reasoning, kept where the decision is made
rather than in the exporter that reads it.
"""

from __future__ import annotations

from .registry import (
    Headlight,
    Scene,
    SceneSpec,
    all_specs,
    apply,
    apply_scene,
    get,
    list_ids,
    load,
    load_cli_args,
    register,
)

register(
    id="default",
    description="mjlab's own look: blue-grey checker, one directional sun",
    tags=("baseline",),
    use="both",
)
register(
    id="studio",
    description="neutral grey, key plus fill, dark sky -- for screenshots and video",
    tags=("presentation",),
    use="watching",
)
register(
    id="daylight",
    description="blue sky, warm ground, one hard sun with shadows",
    tags=("presentation", "outdoor"),
    use="watching",
)
register(
    id="beach",
    description="low warm sun over damp sand, dusk sky, long shadows",
    tags=("presentation", "outdoor"),
    use="watching",
)
register(
    id="football",
    description="a pitch at one fifth scale: mown turf, markings, goals, afternoon sun",
    tags=("presentation", "outdoor"),
    use="watching",
)
register(
    id="swing",
    description="**adds a swing**: an A-frame and a plank seat the robot can stand on",
    tags=("prop", "outdoor"),
    # `watching`, on the evidence in `jumper.swing`'s own docstring: that task
    # builds its swing itself, and `--scene swing` on top of it **raises** --
    # two swings, one name. So nothing trains on this one, and a prop a robot
    # can stand on is exactly what a web simulator wants.
    #
    # `both` if a task ever does train against it. The consequence of the choice
    # is narrow: `use="training"` is the only value `export_web_scene.py`
    # refuses.
    use="watching",
)
register(
    id="soft",
    description="**changes the ground**: default's look, but the feet sink into it",
    tags=("physics", "ground"),
    # Withheld twice over, and either reason alone would do. Its ground
    # behaviour is a per-step Python force (`GranularDrag`), which does not
    # survive being written to a file; and its look is `default`'s, copied
    # verbatim so the two are one-variable comparable. An export would be a
    # scene that looks exactly like `default` and, unlike `default`, claims
    # to be soft.
    use="training",
)
register(
    id="ice",
    description="**changes the ground**: wet ice, sliding friction 0.03 against "
                "the feet's 1.2",
    tags=("physics", "ground"),
    use="both",
)
register(
    id="rubber",
    description="**changes the ground**: a coarse anti-slip mat, sliding friction "
                "1.8 -- the other end of `ice`",
    tags=("physics", "ground"),
    use="both",
)
register(
    id="plain",
    description="no textures or materials -- for geom_rgba randomisation, and cheapest to draw",
    tags=("minimal",),
    # No textures and no materials at all -- that is what it is for. There is
    # nothing here to look at, by construction.
    use="training",
)
register(
    id="rough",
    description="**changes the ground**: generated slopes, stairs and rough surface",
    tags=("terrain",),
    # A 10 x 7 grid of difficulty tiles with a flat control column, sized from
    # this robot's foot-lift height. An instrument for the curriculum to move
    # a robot through, not a place. Stairs worth walking on for fun would be a
    # new scene, not this one with its levels shipped.
    use="training",
)

__all__ = [
    "Headlight",
    "Scene",
    "SceneSpec",
    "all_specs",
    "apply",
    "apply_scene",
    "get",
    "list_ids",
    "load",
    "load_cli_args",
    "register",
]
