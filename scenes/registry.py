"""The scene registry.

A **scene** is how the world looks and what the ground is: the floor's texture and
material, the lights, the skybox, and optionally the terrain geometry. It is
deliberately separate from the task, because those are two different axes -- the
same gait task is worth running on flat ground for a baseline and on rough ground
for robustness, and the same terrain is worth looking at in a neutral studio grey
for a screenshot and under a daylight sky for a video.

## One scene = one module = one fixed name

    scenes/studio.py     scene id is "studio"
    `-- def scene() -> Scene

Scenes are single modules rather than directories, unlike tasks: a task carries an
environment config, hyper-parameters and often its own mdp/, while a scene is one
factory. The registration table below is the exception to "metadata lives with the
thing" that this buys -- it is the price of not having a two-file directory for
every colour scheme.

## Why the table holds no config values

The same reason as `tasks.registry`: registration happens at import time, when the
backend is not yet resolved and mjlab must not be pulled in. `register()` stores
strings; `load()` is what imports a scene module. So `--list` still works on a
machine with no simulation dependencies installed.

## Image files

Builtin patterns need no files. A scene using image textures puts them in
`assets/scenes/<id>/` and refers to them by path -- data belongs under `assets/`,
the code describing it belongs here (the same split as models; see docs/USAGE.md).
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass, field
from typing import Any

#: What a scene is for, and therefore whether it leaves this repository.
#:
#: The split is between a **place** and an **instrument**. An instrument exists to
#: produce a training effect or to hold one variable still for a comparison; it is
#: not somewhere anyone wants to watch a robot, and shipping one is offering a
#: measuring device as a playground.
#:
#: "watching" does not mean a scene cannot be trained in -- `studio` is only
#: lights. It means training in it is beside the point, and in `football`'s case
#: actively misleading: one pitch is drawn at the world origin while environments
#: sit on a spacing grid, so with 4096 of them exactly one robot has a ball.
USES: dict[str, str] = {
    "training": "an instrument: exists for a training effect or a controlled comparison",
    "watching": "a place, built to be looked at",
    "both": "a place that is also where policies are trained",
}

__all__ = [
    "GROUND",
    "GROUND_PRIORITY",
    "USES",
    "Headlight",
    "Scene",
    "SceneSpec",
    "all_specs",
    "apply",
    "apply_scene",
    "get",
    "list_ids",
    "load",
    "register",
    "set_ground_contact",
]


@dataclass(frozen=True)
class Scene:
    """What a scene sets on the terrain entity.

    Every field is optional and an empty one **means empty**, not "leave it
    alone": `textures=()` is how a scene says "no textures", which is what domain
    randomisation over `geom_rgba` needs. The one field that can opt out is
    `terrain_type`, because changing the ground geometry is not a cosmetic choice
    -- see `apply`.
    """

    textures: tuple[Any, ...] = ()
    """`mjlab.utils.spec_config.TextureCfg`, including `type="skybox"` for the sky."""

    materials: tuple[Any, ...] = ()
    """`MaterialCfg`. Its `geom_names_expr` has to match the ground geom, which is
    named `terrain`; a material that matches nothing is built and then bound to
    nothing, and the picture simply does not change."""

    lights: tuple[Any, ...] = ()
    """`LightCfg`. Empty means no lights at all, which renders black."""

    terrain_type: str | None = None
    """``"plane"`` / ``"generator"``, or None to keep whatever the task chose."""

    terrain_generator: Any | None = None
    """`TerrainGeneratorCfg`, required when `terrain_type` is ``"generator"``."""

    nconmax: int | None = None
    njmax: int | None = None
    """Contact and constraint budgets this ground needs, or None to leave the
    task's alone. **`apply` only ever raises them**, so a task that already asks
    for more keeps what it asked for.

    A task sizes these for the ground it was designed on, and a scene that
    replaces the ground invalidates that sizing without touching the task. The
    failure is at construction and it is loud -- `mjwarp.put_data` raises
    ``nconmax overflow (nconmax must be >= 184)`` -- but it names neither the
    scene nor the knob, and the value it asks for is the one *this* spawn needed
    rather than a safe one. `scenes/rough.py` has the measurement behind its
    pair."""

    events: dict[str, Any] = field(default_factory=dict)
    """Event terms the scene adds, as `{name: EventTermCfg}`. **These are
    physical**, like `props` and unlike `decorate`.

    For a scene whose ground behaves in a way MuJoCo's contact model does not
    express -- granular drag is the case this exists for -- and which therefore
    needs a force applied every step rather than a parameter set once. A term
    here with `mode="step"` runs inside `ManagerBasedRlEnv.step`.

    Merged into the task's own events. A name clash raises rather than
    overwriting: a scene must not replace a task's domain randomisation.
    """

    props: dict[str, Any] = field(default_factory=dict)
    """Entities the scene adds, as `{name: EntityCfg}`. **These are physical.**

    Unlike `decorate`, which may only add geometry the robot cannot touch, a prop
    is a body in the simulation: it has mass, it collides, and the robot's
    contacts, contact count and solver load all change because it is there. A
    scene that carries one is no longer only a look, and `apply` says so out
    loud -- the same treatment as generated terrain, and for the same reason. The
    observation and the reward are untouched, so nothing else would report it.

    Entities are placed per environment by mjlab, using `env_origins`, which is
    why a prop goes here rather than into `decorate`: geometry added to the spec's
    worldbody sits at one place in every environment, so with 4096 of them only
    the robot at the origin could ever reach it."""

    decorate: Any = None
    """Optional `(spec) -> None` run on the compiled-out `MjSpec`, for a scene
    that needs geometry rather than only a material -- pitch markings, goalposts.

    It runs through `SceneCfg.spec_fn` after everything is attached, which is the
    only point where the whole world is visible. **Anything added here should be
    visual**: `contype=0, conaffinity=0`. A scene is a look, and geometry the
    robot can collide with changes the task it is being trained on, silently and
    everywhere, which is not something `--scene` should be able to do."""

    headlight: Headlight | None = None
    """The camera-attached ambient. None keeps whatever the model brings, which
    is mjlab's scene template rather than MuJoCo's default -- see `Headlight`."""


@dataclass(frozen=True)
class Headlight:
    """MuJoCo's camera-attached light -- the scene's ambient floor.

    It is easy to miss and it dominates everything else. It follows the camera,
    so it lights every surface the viewer can see regardless of where the sun is,
    which means it sets the floor on how dark a shadow can get. mjlab's scene
    template ships 0.3 ambient / 0.6 diffuse, three times MuJoCo's own default,
    and at that strength halving a fill light changes almost nothing -- measured
    while trying to do exactly that.

    So it belongs to the scene rather than to the asset or the framework: "how
    dark is the dark" is a lighting choice, and it is the first thing to reach for
    when a scene will not go dark.
    """

    ambient: tuple[float, float, float]
    diffuse: tuple[float, float, float]
    specular: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SceneSpec:
    """A scene's metadata. **No config values** -- those come from `scenes/<id>.py`."""

    id: str
    """The fixed name used on `--scene`. Must match the module under `scenes/`."""

    description: str = ""
    """A one-line description, shown by `--list`."""

    tags: tuple[str, ...] = field(default_factory=tuple)
    """Free-form tags, e.g. ``("terrain",)`` for scenes that change the ground."""

    use: str = ""
    """Who the scene is for -- one of `USES`. **Required**, and the reason it is
    a field rather than something read off `tags`.

    A scene leaves this repository when this is not ``"training"``: that is what
    `scenes/tools/export_web_scene.py` reads, and the only thing it reads. The
    alternative was a list of scene ids in the exporter, which is a second place
    the judgement lives -- add a scene, forget the list, and it is silently either
    shipped or not shipped with nothing to notice. Inferring it from `tags` is the
    same mistake wearing a better hat: `("presentation",)` says how a scene looks,
    not that it is fit to hand to someone else.

    `register` raises on a missing or unknown value rather than defaulting,
    because both defaults are wrong in a way that looks right. Defaulting to
    exported ships a curriculum grid as a place to play; defaulting to withheld
    means a new scene quietly never arrives.
    """


#: The ground geom mjlab builds is named `terrain`. A material whose
#: `geom_names_expr` misses it is created, bound to nothing, and the picture does
#: not change -- no error, no warning. Anchored here so the scenes cannot drift
#: apart on it, and so one rename fixes all of them.
GROUND: tuple[str, ...] = ("terrain$",)


#: The contact priority a scene has to use to change the ground's friction.
#:
#: The feet are `priority=1` (`constants.py`), and MuJoCo does not blend contact
#: parameters across a priority difference -- **the higher geom's set is used
#: outright and the other's is discarded**. So a ground at 0 or 1 has its friction
#: thrown away before the solver sees it, and the scene silently does nothing.
#: Verified on the compiled model: of 73 geoms exactly six carry priority 1, and
#: they are the six foot collision meshes at friction (1.2, 0.01, 0.01).
GROUND_PRIORITY = 2


def set_ground_contact(spec: Any, scene_id: str, friction, solref=None, solimp=None) -> int:
    """Give every ground geom a contact parameter set, and the priority to count.

    The shared half of `ice.py` and `rubber.py`, which differ only in the numbers
    they pass. `soft.py` predates this and does the same thing inline; it is worth
    migrating the next time that file is opened, not worth a separate change now.

    Three traps, all of which produce a scene that reports nothing and changes
    nothing:

    - **`GROUND_PRIORITY`**, without which the friction never reaches the solver.
    - **priority replaces the whole parameter set**, not the field you meant. So
      `solref` / `solimp` default to the *feet's* own values rather than being
      left alone: left out, the terrain's own contact stiffness would win too and
      the scene would change two things while claiming one.
    - **`spec.geoms`, not `spec.worldbody.geoms`.** The terrain geom does not hang
      off the worldbody at the point `decorate` runs, so the obvious spelling
      matches nothing.

    Raises when it matches no ground, which is the difference from the material
    binding whose expression it borrows: an unbound material is a cosmetic
    surprise, while ground that quietly kept its old friction would make every
    measurement taken on the scene wrong without looking wrong.

    Returns how many geoms it touched.
    """
    import re

    if solref is None or solimp is None:
        from tasks.jumper.common.constants import _CONTACT_SOLIMP, _CONTACT_SOLREF

        solref = _CONTACT_SOLREF if solref is None else solref
        solimp = _CONTACT_SOLIMP if solimp is None else solimp

    pats = [re.compile(p) for p in GROUND]
    hits = 0
    for geom in spec.geoms:
        if not any(p.search(geom.name or "") for p in pats):
            continue
        geom.friction = list(friction)
        geom.solref = list(solref)
        geom.solimp = list(solimp)
        geom.priority = GROUND_PRIORITY
        hits += 1
    if not hits:
        raise RuntimeError(
            f"scene {scene_id!r} matched no ground geom with {GROUND!r}; the spec "
            f"has {[g.name for g in spec.geoms]}. The ground would have kept its "
            f"own friction and the scene would have said nothing."
        )
    return hits


_REGISTRY: dict[str, SceneSpec] = {}


def register(
    id: str,  # noqa: A002
    description: str = "",
    tags: tuple[str, ...] = (),
    *,
    use: str,
) -> SceneSpec:
    """Register a scene. Registering the same id twice raises, so nothing is
    silently overwritten, and `use` has to be decided -- see `SceneSpec.use`."""
    if id in _REGISTRY:
        raise ValueError(
            f"scene id registered twice: {id!r} (an earlier import already defined it)"
        )
    if use not in USES:
        raise ValueError(
            f"scene {id!r} declares use={use!r}; it must be one of {sorted(USES)}. "
            f"This decides whether the scene is exported for someone else to walk "
            f"around in, so it has to be chosen rather than defaulted:\n"
            + "\n".join(f"  {k:10s} {v}" for k, v in USES.items())
        )
    spec = SceneSpec(id=id, description=description, tags=tags, use=use)
    _REGISTRY[id] = spec
    return spec


def get(scene_id: str) -> SceneSpec:
    """Look up a SceneSpec, listing every available id on failure rather than
    raising a bare KeyError."""
    try:
        return _REGISTRY[scene_id]
    except KeyError:
        available = "\n  ".join(sorted(_REGISTRY)) or "(the registry is empty)"
        raise KeyError(
            f"unknown scene {scene_id!r}. Available scenes:\n  {available}"
        ) from None


def list_ids(tag: str | None = None) -> list[str]:
    ids = sorted(_REGISTRY)
    return ids if tag is None else [i for i in ids if tag in _REGISTRY[i].tags]


def all_specs() -> list[SceneSpec]:
    return [_REGISTRY[i] for i in sorted(_REGISTRY)]


def load_cli_args(scene_id: str) -> Any | None:
    """This scene's own command-line arguments, or None if it takes none.

    The mirror of `tasks.load_cli_args`, and it exists for the same reason: a
    knob that belongs to one scene must not appear in `scripts/_cli.py`, which is
    one parser shared by three entry points and is not allowed any scene's or
    task's vocabulary (`tests/test_log_layout.py` pins that).

    **Absent by default and that is not an error.** Every scene but `rough` takes
    no options; `getattr` returning None is the normal case.

    Declared in the scene module rather than in a table here, for the reason the
    module docstring gives for holding no config values: registration runs at
    import time and must not pull in mjlab. Declaring an argument needs argparse
    and nothing else, but the module is only imported when the scene is actually
    selected, so the cost is paid by the run that asked for it.
    """
    get(scene_id)
    module = importlib.import_module(f"{__package__}.{scene_id}")
    return getattr(module, "cli_args", None)


def load(scene_id: str, **scene_args: Any) -> Scene:
    """Build a scene by importing `scenes/<id>.py` and calling its `scene()`.

    `scene_args` are the values the scene's own `cli_args` collected, keyed by
    `dest` -- empty for every scene that declares none, which is why `scene()`
    stays a no-argument factory for all but one of them.
    """
    get(scene_id)  # confirm the id is registered first, for a better error
    dotted = f"{__package__}.{scene_id}"
    try:
        module = importlib.import_module(dotted)
    except ModuleNotFoundError as e:
        if e.name != dotted:
            raise  # a dependency of the scene module, not the module itself
        raise ModuleNotFoundError(
            f"{dotted} not found. A scene id must match a module under scenes/ -- "
            f"see scenes/registry.py."
        ) from None
    return module.scene(**scene_args) if scene_args else module.scene()


def apply(env_cfg: Any, scene_id: str, **scene_args: Any) -> Scene:
    """Apply a scene to an already-built environment config, and return it.

    Called after `tasks.load_env_cfg`, so the task has had its say first and the
    scene overrides the look. That order matters: a task legitimately configures
    the ground it was designed for, and a scene is the user overruling it for this
    run.

    **Changing the terrain type is not cosmetic**, which is why `terrain_type=None`
    means "leave it". Rough ground without the height-scan sensor in the
    observation gives a policy that cannot see what it is walking on -- it trains,
    it converges to something, and nothing anywhere reports a problem. The tasks in
    this repository drop that sensor when they set flat ground, so pairing a
    terrain scene with one of them has to be a deliberate choice: **nothing here
    checks it**, and `terrain_scan` has to be restored in the task's `env_cfg`
    before the result means anything.
    """
    return apply_scene(env_cfg, load(scene_id, **scene_args), scene_id)


def apply_scene(env_cfg: Any, scene: Scene, label: str) -> Scene:
    """Apply an already-built `Scene`. `apply` is this with a registry lookup.

    Split out because a scene need not come from the registry: an imported
    `kk-scene-package/1` or `/2` is read off disk and becomes a `Scene` like any other
    (see `scenes.package`). Everything a scene is allowed to change -- events,
    props and their per-environment placement, the spec edits, the ground, the
    contact budget -- belongs here rather than in each caller, so an imported
    scene cannot quietly skip a check a registered one gets.

    `label` is what the errors and the warning call this scene: an id for a
    registered one, the package id for an imported one.
    """
    terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    if terrain is None:
        raise ValueError(
            f"this environment config has no scene.terrain, so scene {label!r} "
            f"has nothing to apply to"
        )
    if scene.events:
        clashes = sorted(set(scene.events) & set(env_cfg.events))
        if clashes:
            raise ValueError(
                f"scene {label!r} would overwrite the task's event terms "
                f"{clashes}. A scene may add physics, not replace the task's."
            )
        env_cfg.events.update(scene.events)
    if scene.props:
        entities = getattr(env_cfg.scene, "entities", None)
        if entities is None:
            raise ValueError(
                f"this environment config has no scene.entities, so scene "
                f"{label!r} cannot add its props"
            )
        clashes = sorted(set(scene.props) & set(entities))
        if clashes:
            raise ValueError(
                f"scene {label!r} would overwrite the task's entities "
                f"{clashes}. Rename the prop; a scene must not replace the robot."
            )
        entities.update(scene.props)
        _place_props(env_cfg, scene.props)
        warnings.warn(
            f"scene {label!r} adds {len(scene.props)} physical prop(s): "
            f"{', '.join(sorted(scene.props))}. These have mass and they collide, "
            f"so contacts, contact count and solver load all change -- a policy "
            f"trained with them is not comparable to one trained without. The "
            f"observation and the reward are untouched, so nothing else reports "
            f"this.",
            RuntimeWarning,
            stacklevel=2,
        )

    if scene.headlight is not None or scene.decorate is not None:
        _install_spec_edits(env_cfg.scene, scene.headlight, scene.decorate)

    terrain.textures = scene.textures
    terrain.materials = scene.materials
    terrain.lights = scene.lights

    if scene.terrain_type is not None:
        terrain.terrain_type = scene.terrain_type
        terrain.terrain_generator = scene.terrain_generator

    # Raised, never lowered: the task's number is a floor it chose for its own
    # ground and this is what the new ground additionally needs.
    sim = getattr(env_cfg, "sim", None)
    for name in ("nconmax", "njmax"):
        wanted = getattr(scene, name)
        if wanted is None or sim is None:
            continue
        current = getattr(sim, name, None)
        if current is None or current < wanted:
            setattr(sim, name, wanted)
    return scene


def _place_props(env_cfg: Any, props: dict[str, Any]) -> None:
    """Give each prop a reset event, so it lands beside its own robot.

    Adding the entity is not enough and the shortfall is invisible in the config:
    mjlab applies `env_origins` inside `reset_root_state_uniform`, and an entity
    with no reset event stays at (0, 0, 0) in **every** environment. Measured
    before this existed -- one ball, at the world origin, half-buried, reachable
    only by whichever robot happened to be near it, while the config said the
    ball was at (0.45, 0, 0.11).

    So the placement is wired here rather than left to whoever writes a prop.

    The pose range is **empty**, not the prop's position: the entity's own
    `init_state` already supplies that, and passing it again lands the prop at
    twice the offset -- measured, a ball configured at 0.45 m appearing at 0.90 m.
    What the event contributes is the env_origin and the return to that spot at
    the start of every episode, which is what makes episodes comparable.
    """
    from mjlab.envs.mdp import events as mdp_events
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    for name in props:
        env_cfg.events[f"reset_prop_{name}"] = EventTermCfg(
            func=mdp_events.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg(name),
            },
        )


def _install_spec_edits(scene_cfg: Any, light: "Headlight | None", decorate: Any) -> None:
    """Set the headlight through `SceneCfg.spec_fn`, which runs last.

    Nothing earlier survives. The value that reaches the model comes from
    mjlab's own `scene/scene.xml`, which is the **parent** spec, so a `<visual>`
    block in an attached entity -- the robot asset, the terrain -- is overridden
    without a word. Measured: deleting the headlight from `assets/jumper/jumper.xml`
    changed the compiled model not at all, because the template sets the same
    values. `spec_fn` runs after everything is attached, which is why it is the
    one place this can be done without editing vendored code.

    Any callback the task already installed is kept and run first: a task setting
    something here has its own reason, and a scene is only claiming the lighting.
    """
    previous = getattr(scene_cfg, "spec_fn", None)

    def edit(spec: Any) -> None:
        if previous is not None:
            previous(spec)
        if light is not None:
            spec.visual.headlight.ambient = list(light.ambient)
            spec.visual.headlight.diffuse = list(light.diffuse)
            spec.visual.headlight.specular = list(light.specular)
        if decorate is not None:
            decorate(spec)

    scene_cfg.spec_fn = edit
