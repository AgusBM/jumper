"""What all jumper tasks share. **This is not a task.**

| Module | Contents |
|---|---|
| `assets.py`       | Model asset declarations. Lightweight, and **does not import mjlab** |
| `constants.py`    | Joint names, HOME pose, collision scheme, mjlab `EntityCfg` |
| `tof.py`          | The dToF: the model's camera as a ray cast, with the chip's errors |
| `velocity_env.py` | The flat-ground velocity-tracking environment skeleton, shared by all four |
| `ppo.py`          | The PPO baseline shared by all four |
| `dance/`          | The dance tasks' clip loader, terms and environment (`dance_env_cfg`) |
| `mdp/`            | MDP terms specific to this robot: gait rewards, sagittal mirroring |

The test: something belongs here if and only if **more than one jumper task needs
it**. Anything used by a single task belongs in that task's own directory.
"""
