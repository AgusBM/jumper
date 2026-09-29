from mjlab.sim.sim import MujocoCfg as MujocoCfg
from mjlab.sim.sim import Simulation as Simulation
from mjlab.sim.sim import SimulationCfg as SimulationCfg
from mjlab.sim.sim_data import TorchArray as TorchArray
from mjlab.sim.sim_data import WarpBridge as WarpBridge

# ── [mjrl] new (against mjlab 1.6.0) ──────────────────────────────────
# Reason: this framework cuts a seam between the manager layer and the physics
# engine so that one set of manager / reward / rsl_rl code runs on both mjwarp
# (GPU) and native MuJoCo (many-core CPU).
#
# The least invasive approach was chosen deliberately: no new abstract protocol,
# just letting that one construction in `ManagerBasedRlEnv` pick a class.
# `mjrl.backend.native_sim.NativeSimulation` has members of the same name and
# meaning as `Simulation` -- a member-for-member substitute (see docs/DESIGN.md
# section 3).
#
# When upgrading upstream: move this block together with the [mjrl] marker in
# manager_based_rl_env.py.

_SIMULATION_CLS: type = Simulation


def set_simulation_cls(cls: type | None) -> None:
  """Register the Simulation implementation to build environments with.

  `None` restores mjlab's default. Called by `mjrl.backend.select.use_backend()`
  **before** the environment is built.
  """
  global _SIMULATION_CLS
  _SIMULATION_CLS = Simulation if cls is None else cls


def get_simulation_cls() -> type:
  """Return the registered Simulation implementation, or mjlab's own if none."""
  return _SIMULATION_CLS
