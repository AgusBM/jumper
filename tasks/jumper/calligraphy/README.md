# Calligraphy — Jumper writes a Chinese character on the floor

Jumper holds a brush in its carried claw and writes a character on the floor, stroke
by stroke, the way 地书 (water calligraphy on paving stones) is written: walking to
where the next stroke can be reached, putting the brush down, writing, lifting it.

```bash
python tasks/jumper/calligraphy/tools/strokes.py                # step 1: 无 -> plan.json + plan.svg
python tasks/jumper/calligraphy/tools/write.py --plan-only      # the cut into stretches, drawn
python tasks/jumper/calligraphy/tools/write.py                  # step 2: write it, ~5 min on CPU
MUJOCO_GL=osmesa python tasks/jumper/calligraphy/tools/render.py logs/calligraphy/u65e0/<run>
                                                                # step 4: two shots, gif, ink export
```

Outputs go to `logs/calligraphy/u65e0/<run>/` (the character's code point):

| file | step | what it is |
|---|---|---|
| `plan.json`, `stretches.json` | 1, 2 | the strokes on the floor, and how they were cut and written |
| `log.npz` | 3 | one row per 20 ms control step -- phase, stroke, target and measured tip, contact and force, trunk pose, command -- and the full `qpos` |
| `model.mjb` | 3 | the compiled model that `qpos` belongs to |
| `topview.png` | 2 | the plan beside where the brush touched the floor |
| `top.mp4`, `low.mp4` | 4 | overhead and low shots, real time, 960x720 at 30 fps, ink drawn as it is laid (`--no-ink`: clean plates) |
| `wu.gif` | 4 | the overhead shot at 6x, 480x360, for the README |
| `ink.json`, `ink.svg` | 3, 4 | the ink, for the compositor: see below |

**This is not a task.** Nothing is trained: `jumper.five_foot`'s shipped policy
(`tasks/jumper/five_foot/out/example/model_86600.pt`) walks and stands on five legs,
and everything here sits on top of it -- the arm that policy leaves out of its action
is driven by inverse kinematics, and its velocity command by a steering loop. There
is no `env_cfg.py` or `rl_cfg.py`, so the registry does not see the directory.

## What is where

| file | what it does |
|---|---|
| `hanzi.py` | a character's medians (Make Me a Hanzi) -> stroke trajectories on the floor, with a press profile |
| `data/` | one verbatim `graphics.txt` entry per character, and the Arphic Public License they are under |
| `brush.py` | the brush: welded in the claw's mouth, a soft tip that collides only with the floor, a contact sensor |
| `arm.py` | IK for the brush tip against the trunk's measured pose; the band of floor it can write on; floor-safe fold paths |
| `stations.py` | cuts each stroke into stretches the arm can write from one place, and where to stand for each |
| `tools/strokes.py` | step 1 |
| `tools/write.py` | step 2: the controller, the log and a top view |
| `ink.py` | step 3: which logged steps are ink, grouped into marks, with a width |
| `cameras.py` | the two shots, one definition for the renderer and the ink's pixels |
| `tools/render.py` | step 4: replays the log in plain MuJoCo and renders it |

## Frame

The character's up is world +x and its right is world -y: the robot faces +x and
writes the way a person writes on a sheet in front of them. Seen from above with +x
at the top it reads the right way round. Make Me a Hanzi's medians are in a 1024-unit
em square with **y up** -- the SVG on its site flips them, and copying that flip
writes every character upside down. See `hanzi.py`.

## What was measured, and what it decided

All on a 4-core Linux container, `native:cpu`, MuJoCo 3.11.0, 2026-10-08.

**The policy stands with the arm out only in front.** 16 arm poses held 8 s each,
no pushes, the tip 15 mm above the floor: with the palm tip >= 0.15 m ahead of the
trunk's centre the trunk drifted 1.4-2.8 cm and turned 0.6-3.8 deg (stowed: 0.4 cm,
2 deg); nearer the trunk, up to 36 cm and 67 deg. No pose fell. So the arm writes
only with the palm tip >= 0.13 m forward (`arm.PALM_X_MIN`).

**The writable floor is a band, not a square.** With that limit and the brush, the
tip can be put on the floor and 15 mm above it over 536 cm², shaped as an arc about
the shoulder; the largest square inside it is 6.5 cm. So `stations.py` fits each
stretch of stroke into the band, at one of five headings, rather than into a box. 无
at 0.30 m comes out as 5 stretches -- strokes 1, 2 and 3 whole.

**It does not walk with the arm out.** Commanded 0.05/0.08/0.12 m/s forward with the
arm held at a writing pose it moved 0.001-0.002 m/s; stowed, 0.027/0.059/0.109. So
the arm folds between stretches, along a path searched to keep it off the floor
(interpolating the four joints at once swept the brush through the floor on a
third of the steps).

**Pressing by depth pushes the robot around.** A fixed 3 mm press gave 0-6 N, and on
the long middle stroke turned the trunk 48 deg while the brush bounced. The press
is held at `press * 1.0 N` by an integrator on the tip's height instead.

**Result, 无 at 0.30 m** (`tools/write.py`, defaults). The robot starts each run
from a randomised pose, so runs differ; two of them:

| | simulated | stretches | tip to stroke, median / p95 / max | brush down while writing | force, median |
|---|---|---|---|---|---|
| run 6 | 86 s | 7 | 1.1 / 2.9 / 5.8 mm | 96% | 1.13 N |
| run 7 (the README's gif) | 103 s | 8 | 1.2 / 6.4 / 17.4 mm | 90% | 1.20 N |

In both the brush touched the floor nowhere outside the strokes. Each extra
stretch is a replan: the trunk stopped too far from its station (run 7: three
times) or drifted out of reach while writing (once).

## The ink, for post-production

`ink.json` holds every **mark** -- one continuous trace of the brush -- with, per
point: the time it was laid, its floor xy, the normal force, the planned press and
a suggested width; and its **pixel position in each shot**, with each shot's
intrinsics and pose. So a compositor can lay the ink down frame by frame on the
clean plates (`render.py --no-ink`) without knowing anything about the simulation.

- A contact counts as ink only while the arm is lowering, writing or lifting.
  Gaps of up to 0.1 s inside a stretch are bridged -- the simulated tip is a rigid
  ball that skips where wet hair would not -- which takes 无 from 71 marks to 10.
- The width is derived, not measured: `0.065 * size * clip(force / 1 N, 0.25, 1.3)`.
  The raw force and press are there for any other rule.
- The pixels are checked against the renderer (`render.py --check`: markers at
  known floor points): <= 0.5 px overhead, 1.4-2.1 px in the low shot.

The renderer is plain MuJoCo on the run's `model.mjb` and `qpos` -- no mjlab, no
torch -- because OSMesa, the CPU renderer this needs without a GPU, crashes the
process when torch is loaded beside it. It repaints the training floor as paving
stones, draws the ink as flat wet patches and lights the scene with a low sun
(the scene's spot light leaves its shadow map's edge on the far floor). It is
slow on a CPU: run 7 (103 s) took 78 min on this 4-core container for both shots
and the gif -- 6614 frames, ~0.7 s each, OSMesa on all four cores.

`docs/media/calligraphy-wu.gif` in the repository's README is `wu.gif` of the run
the README quotes.

## Not done yet

- The policy was never trained with the arm out or with a load on the floor.
  Training `five_foot` with the arm sampled over the writing band would widen the
  band and let it walk while writing (option B).
