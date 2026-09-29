"""Verify mesh stripping, which must happen after the collision scheme is chosen
(see the precondition on strip_visual_meshes).

It lives here rather than in the top-level `tools/checks/` because it compiles the
model straight from `jumper.xml` (`MjSpec.from_file`) and is bound to that specific
asset rather than to a task or to framework logic.

Usage: python assets/jumper/tools/check_model_slim.py
"""
import copy, gc, numpy as np, mujoco
from mjrl.backend.model_slim import strip_visual_meshes, model_nbytes
from tasks.jumper.common.constants import JUMPER_XML, HYBRID_COLLISION

XML = str(JUMPER_XML)   # single source of truth for the asset path

def build(slim):
    spec = mujoco.MjSpec.from_file(XML)
    HYBRID_COLLISION.edit_spec(spec)   # choose the collision scheme, then strip
    if slim:
        strip_visual_meshes(spec)
    return spec.compile()

for scheme in ("hybrid",):
    full, sl = build(False), build(True)
    fb, sb = model_nbytes(full), model_nbytes(sl)
    ncol = sum(bool(full.geom_contype[i] or full.geom_conaffinity[i])
               for i in range(full.ngeom))
    print(f"\n######## {scheme} ({ncol} colliding geoms) ########")
    print(f"  model  {fb/1e6:>8.2f} MB -> {sb/1e6:>8.2f} MB   saved {(1-sb/fb)*100:>5.1f}%")
    print(f"  nmesh  {full.nmesh:>8} -> {sl.nmesh:>8}      ngeom {full.ngeom} -> {sl.ngeom}")
    for f in ("nq","nv","nbody","njnt","nsensor","nsite"):
        a,b = getattr(full,f), getattr(sl,f)
        if a != b: print(f"  FAIL {f} changed {a} -> {b}")
    dyn = max(np.abs(np.asarray(getattr(full,f))-np.asarray(getattr(sl,f))).max()
              for f in ("body_mass","body_ipos","body_iquat","body_inertia"))
    print(f"  max dynamics difference {dyn:.3e}  {'ok' if dyn<1e-9 else 'FAIL'}")

    def run(m, seed=0):
        d = mujoco.MjData(m); mujoco.mj_resetData(m, d)
        rng = np.random.default_rng(seed)
        c = rng.uniform(-0.2, 0.2, size=(200, max(m.nu,1)))
        for t in range(200):
            if m.nu: d.ctrl[:] = c[t][:m.nu]
            mujoco.mj_step(m, d)
        return np.array(d.qpos), np.array(d.qvel)
    q1,v1 = run(full); q2,v2 = run(sl)
    e = max(np.abs(q1-q2).max(), np.abs(v1-v2).max())
    print(f"  max trajectory difference over 200 steps {e:.3e}  {'identical' if e<1e-9 else 'FAIL: diverged'}")
    print(f"  extrapolated to N=512: {fb*512/1e9:.2f} GB -> {sb*512/1e9:.2f} GB")

# Measure copy memory by RSS delta
import resource
sl = build(True)
gc.collect(); m0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
cp = [copy.deepcopy(sl) for _ in range(128)]
m1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
print(f"\n128 stripped hybrid copies: {m1-m0:.1f} MB total -> {(m1-m0)/128:.3f} MB each")
del cp; gc.collect()
