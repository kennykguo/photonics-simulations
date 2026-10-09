"""Builds every output of this folder.

    cd ~/photonics-simulations/visuals/03_coupled_modes && ../../.venv/bin/python run.py

Stages (each can also be run on its own):
  solve.py        mode solves and numbers   -> out/cache/, out/results.json   (~2 min, ~8 min first time)
  pages.py        the step pages            -> out/pages/*.png, out/coupled_modes_walkthrough.pdf
  anim.py         the walk along the coupler -> out/coupler_walk.mp4 / .gif
  interactive.py  rotatable 3-D             -> out/explore_3d.html

The 3-D FDTD check is separate because it takes ~25 min and needs the Meep
interpreter; run it once and run.py picks up its result:
  ../../.meep/bin/python fdtd3d.py

--resolve forces the mode solves to run again (otherwise the cache is reused).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

if __name__ == "__main__":
    t0 = time.time()
    if "--resolve" in sys.argv or not (OUT / "cache" / "solved.npz").exists():
        print("solving modes ...")
        import solve
        solve.main()
    import anim
    import interactive
    import pages

    D = dict(np.load(OUT / "cache" / "solved.npz"))
    R = json.loads((OUT / "results.json").read_text())
    fd = pages.load_fdtd()
    if fd is None:
        print("  (no FDTD result yet: step 10 shows a placeholder; run fdtd3d.py with the Meep interpreter)")
    print("building pages ...")
    paths = pages.build_all(D, R, fd)
    (OUT / "results.json").write_text(json.dumps(R, indent=2, default=float))
    print(f"  {len(paths)} pages in out/pages/ and out/coupled_modes_walkthrough.pdf")
    print("animating ...")
    anim.main()
    interactive.main()
    V = R["vector"]
    print(f"\nkappa_c = {V['kappa_exact']:.4f} rad/um (exact), {V['kappa_overlap']:.4f} (overlap formula); "
          f"L_c = {V['L_c_exact']:.1f} um.  Done in {time.time()-t0:.0f} s.")
