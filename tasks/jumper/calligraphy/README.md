# Calligraphy — Jumper writes a Chinese character on the floor

Jumper holds a brush in its carried claw and writes a character on the floor, stroke
by stroke, the way 地书 (water calligraphy on paving stones) is written: walking to
where the next stroke can be reached, putting the brush down, writing, lifting it.

```bash
python tasks/jumper/calligraphy/tools/strokes.py                # step 1: 无 -> plan.json + plan.svg
python tasks/jumper/calligraphy/tools/write.py --plan-only      # the cut into stretches, drawn
python tasks/jumper/calligraphy/tools/write.py                  # step 2: write it, ~5 min on CPU
```

Outputs go to `logs/calligraphy/u65e0/` (the character's code point).

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

**Result, 无 at 0.30 m** (`tools/write.py`, defaults): 86 s simulated, 7 stretches
(one replanned after the trunk stopped 1.2 cm off its station). While writing, the
tip is 1.1 mm from the stroke (median; p95 2.9 mm, max 5.8 mm), on the floor 96% of
the time at 1.13 N (median), and it touches the floor nowhere else.

## Not done yet

- Step 3's log exists (`log.npz`, one row per 20 ms control step) but is not yet a
  documented export format.
- Step 4: the renders from above and from a low angle, and the stroke export for the
  ink pass.
- The policy was never trained with the arm out or with a load on the floor.
  Training `five_foot` with the arm sampled over the writing band would widen the
  band and let it walk while writing (option B).
