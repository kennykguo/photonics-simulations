# visuals/

One folder per visual request. Each folder has its own `run.py`, `README.md` (how to run
it, what every output shows, checks and limits) and `out/`. Improved versions of an
earlier visual go into a `v2/` next to the original; nothing earlier is deleted.

| folder | the request | start with |
|---|---|---|
| [01_neff_and_ng](01_neff_and_ng/) | How to visualise n_eff and n_g through a waveguide | `out/phase_vs_group.mp4` |
| [02_te_vs_tm](02_te_vs_tm/) | Explain quasi-TE versus quasi-TM, as accurately as possible | `out/te_vs_tm_poster.png` |
| [03_coupled_modes](03_coupled_modes/) | The coupled-mode derivation of κ_c, step by step, one question per page | `out/coupled_modes_walkthrough.pdf` |

The 13-experiment lab in `experiments/` (the first request) predates this folder and
stays where it is.

## Environment

All folders share the repo-root interpreters:

- `../.venv` (uv, Python 3.12): numpy, scipy, matplotlib, femwell, scikit-fem, gmsh,
  pyvista, plotly, imageio-ffmpeg. `bash setup.sh` here creates a minimal one;
  `bash ../env/setup.sh` installs the full lab.
- `../.meep` (micromamba, conda-forge): Meep for the 3-D FDTD in `03_coupled_modes`.

`indices.py` here holds the Sellmeier fits and slab solvers shared by every folder.
