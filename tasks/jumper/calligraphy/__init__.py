"""Jumper writing a Chinese character on the floor with a brush, 地书-style.

Not a task: there is no `env_cfg.py` or `rl_cfg.py` here, so nothing registers
and nothing trains. It drives `jumper.five_foot`'s trained policy -- which walks
on five legs and carries the left-front arm -- and moves that arm itself, so this
directory holds only what the writing adds on top: the stroke data and planner
(`hanzi.py`), the brush (`brush.py`) and the tools that run them (`tools/`).
See `README.md`.
"""
