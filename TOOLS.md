# Tools used in photonics-simulations

One section per distinct tool, library or program used anywhere in the 13 experiments, in the
order: package managers and interpreters, the general scientific stack, the notebook stack, the
animation engine, the circuit simulator, the electromagnetic solvers, the circuit/layout stack,
the media tools, and finally the packages that are installed but that no experiment uses. Each
section has the same six parts: what it is, version and location on this machine, where it is
used in this repo, the results it produced (with links relative to the repo root), how to
observe it, and the limitations or gotchas that were actually hit. A concept-to-tool map and a
section on reproducing the environment close the file.

Versions were read on 2026-09-17 with `env/check_env.py`; every experiment's `out/tools.json`
carries the versions it saw at its own run time. Paths are relative to
`/Users/kennyg/photonics-simulations` unless absolute.

Contents: [Homebrew](#homebrew) · [uv](#uv) · [Python](#python) · [micromamba](#micromamba) ·
[numpy](#numpy) · [scipy](#scipy) · [matplotlib](#matplotlib) · [sympy](#sympy) ·
[python-control](#python-control) · [Jupyter](#jupyter) · [ipywidgets](#ipywidgets) ·
[nbformat](#nbformat) · [Manim](#manim) · [ngspice](#ngspice) · [Meep](#meep) ·
[MPB](#mpb-eigenmode-solver) · [Harminv](#harminv) · [Tidy3D](#tidy3d-mode-solver) ·
[femwell](#femwell) · [scikit-fem](#scikit-fem) · [gmsh](#gmsh) · [meshio](#meshio) ·
[shapely](#shapely) · [SAX](#sax) · [JAX](#jax) · [gdsfactory](#gdsfactory) ·
[ffmpeg / ffprobe](#ffmpeg-and-ffprobe) · [imageio](#imageio) · [Pillow](#pillow) ·
[scikit-rf](#scikit-rf) · [fdtd](#fdtd-python-package) · [other installed](#other-installed-packages) ·
[Tool map](#tool-map) · [Reproducing the environment](#reproducing-the-environment)

---

## Homebrew

**What it is.** The package manager for macOS. It installs command-line programs and libraries
into `/opt/homebrew` (Apple silicon). In research and industry it is the usual way to get
compiled tools such as ffmpeg or a SPICE simulator onto a Mac without building them.

**Version and location.** Homebrew 7.0.4, `/opt/homebrew/bin/brew`. The programs it installed
for this repo: `/opt/homebrew/bin/ffmpeg`, `/opt/homebrew/bin/ffprobe`,
`/opt/homebrew/bin/ngspice`, `/opt/homebrew/bin/micromamba`, `/opt/homebrew/bin/uv`.

**Where it is used.** Only in [env/setup.sh](env/setup.sh) (`brew install ffmpeg ngspice
micromamba uv`). No experiment calls `brew`.

**Results.** The four programs above, which every experiment depends on (ffmpeg for every
video, ngspice for 03, micromamba for the Meep environment of 06, 07, 09, 10, 11, 12).

**How to observe it.** `brew --version`; `brew list ffmpeg ngspice micromamba uv`;
`env/check_env.py` prints the version and path of each program.

**Limitations and gotchas.** Most experiments hard-code the absolute paths
(`/opt/homebrew/bin/...`) instead of relying on `PATH`, because `run.py` may be started from a
shell without Homebrew on the path (for example from `run_all.sh` or a cron job): ngspice in 03,
and ffmpeg for the matplotlib video writer in 01, 02, 05, 06, 07, 08, 10 (in `run.py`) and 12
(in `ring_sax.py` and `ring_meep.py`). Experiments 03, 04, 09, 11 and 13 leave `FFMpegWriter` on
its default `ffmpeg` and so need `/opt/homebrew/bin` on `PATH` for their videos (03, 04 and 13
use the absolute path only to read the version). If Homebrew lives elsewhere (Intel Mac:
`/usr/local`), those constants at the top of each script are the only thing to change.

## uv

**What it is.** A fast Python package and environment manager (a drop-in replacement for
`pip` and `venv`, written in Rust). It creates virtual environments pinned to a Python version
and resolves and installs wheels in seconds; it is increasingly the default tool for
reproducible Python environments.

**Version and location.** uv 0.11.12, `/opt/homebrew/bin/uv`.

**Where it is used.** [env/setup.sh](env/setup.sh): `uv venv --python 3.12 .venv` and
`uv pip install --python .venv/bin/python <packages>`. Nothing at run time.

**Results.** The `.venv` environment (Python 3.12.13) with the whole non-Meep stack listed in
[README.md](README.md#setup).

**How to observe it.** `uv --version`; `uv pip list --python .venv/bin/python` shows every
installed package with its version. To add a package: `uv pip install --python
.venv/bin/python <name>`.

**Limitations and gotchas.** pymeep has no wheel on PyPI, which is why Meep lives in a
separate micromamba environment rather than in `.venv`. `uv` does not register a Jupyter
kernel; the `photonics-sims` kernel was registered with `ipykernel install` (see
[Jupyter](#jupyter)).

## Python

**What it is.** The language every experiment is written in. Two interpreters are used because
Meep is only available from conda-forge.

**Version and location.**
- `.venv/bin/python`: CPython 3.12.13 (uv). Site-packages: `.venv/lib/python3.12/site-packages/`.
- `.meep/bin/python`: CPython 3.12.14 (micromamba, conda-forge). Site-packages:
  `.meep/lib/python3.12/site-packages/`.

**Where it is used.** Every `run.py`, helper module and notebook. Experiments 01 to 05, 08 and
13 run under `.venv`; 06, 07, 09, 10, 11 and 12 carry a `.uses_meep` marker and run under
`.meep` (12's `run.py` accepts either and dispatches its two halves to the right interpreter
via `subprocess`; 11's `run.py` runs its SAX and gdsfactory parts with `.venv` the same way).
Each `run.py` starts with `sys.path.insert(0, <repo root>)` so that `from common import REF,
use_style, SERIES, PALETTE` works from inside the experiment folder.

**Results.** Everything under `experiments/*/out/`.

**How to observe it.** `.venv/bin/python --version`, `.meep/bin/python --version`;
`env/check_env.py` asserts that it is being run by `.venv/bin/python` and calls `.meep/bin/python
-c "import meep"` in a subprocess. Always `cd` into the experiment folder before running its
`run.py` (the scripts resolve `out/` relative to `__file__` but some helper imports and the
notebook execution assume the working directory).

**Limitations and gotchas.** The two environments share nothing: `sax`, `tidy3d`, `femwell`,
`nbformat` are not importable from `.meep`, and `meep` is not importable from `.venv`. Pure
numpy/scipy helper modules (`slab_analytic.py`, `slab_coupler.py`, `analytic.py`) were written to
import under either. numpy is 2.4.6 in `.venv` and 2.5.3 in `.meep`, which is why the versions
in `out/tools.json` differ between experiments.

## micromamba

**What it is.** A small, static, single-binary implementation of the conda package manager. It
creates conda environments from conda-forge without a full Anaconda or Miniconda install. It is
how compiled scientific packages that are not on PyPI (Meep, MPB, Harminv, HDF5 with MPI, ...)
are normally installed.

**Version and location.** micromamba 2.9.0, `/opt/homebrew/bin/micromamba`. Root prefix
`.mamba/` (set with `MAMBA_ROOT_PREFIX=$PWD/.mamba`); the environment it created is `.meep/`.

**Where it is used.** [env/setup.sh](env/setup.sh): `micromamba create -y -p .meep -c conda-forge
python=3.12 pymeep numpy scipy matplotlib h5py imageio imageio-ffmpeg`. Not used at run time.

**Results.** `.meep/`: pymeep 1.34.0 with `libmeep`, `libmpb`, `libharminv`, `libctlgeom` in
`.meep/lib/`, a `harminv` command-line binary in `.meep/bin/`, and the Python packages the Meep
experiments need (numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, h5py 3.16.0, imageio 2.37.4).

**How to observe it.** `micromamba --version`; `MAMBA_ROOT_PREFIX=$PWD/.mamba micromamba list
-p .meep` lists the environment. To add a package to the Meep environment:
`MAMBA_ROOT_PREFIX=$PWD/.mamba micromamba install -y -p $PWD/.meep -c conda-forge <pkg>`.

**Limitations and gotchas.** The root prefix must be passed (or exported) on every call,
otherwise micromamba looks in `~/micromamba`. Do not activate the environment in a shell for
the experiments; call `.meep/bin/python` by path so that `run_all.sh` and the subprocess calls
in 11 and 12 stay deterministic.

## numpy

**What it is.** The array library underneath all of scientific Python: n-dimensional arrays,
vectorised arithmetic, complex numbers, FFTs, linear algebra, polynomial fits. Every other
numerical tool here (scipy, matplotlib, the solvers' outputs) reads and writes numpy arrays.

**Version and location.** 2.4.6 in `.venv` (`.venv/lib/python3.12/site-packages/numpy`), 2.5.3
in `.meep`.

**Where it is used.** Every experiment. The specific things it computed:
- 01: E0 cos(ωt - βz) on (z, t) grids, the complex phasor and its real part, crest tracking
  with parabolic refinement, `np.polyfit` for v_p, the round-trip phase βL.
- 02: Re{e^{j(ωt - k_x x - k_z z)}} with real and imaginary k_x on 420x420 grids, zero-crossing
  wavelength measurements, the E and H phasors and ⟨S⟩ = ½Re{E x H*}.
- 03: the analytic χ(Ω), n = √(1+χ), least-squares sinusoid fits to the ngspice transients, the
  Sellmeier-to-Lorentz conversion, the Drude estimate.
- 04, 05: evaluation of the lambdified Sellmeier expressions; in 05 `numpy.fft` is the
  propagator (FFT, multiply by e^{-jβ(ω)L}, inverse FFT), the NRZ filtering and the eye folding.
- 06: α, n'', a, γ from the closed forms; `np.polyfit` slopes of ln|E|² and of the unwrapped
  phase; `np.gradient` for ∇·S; sub-grid node finding.
- 07, 10, 11: the slab eigenvalue functions, F(x), overlap integrals, transfer matrices.
- 08: Sellmeier materials, finite-difference group indices, the tail fit, the ring comb.
- 09: Fresnel and frustrated-TIR closed forms, the Gaussian-beam angular spectrum, log-linear
  fits.
- 12: the closed-form ring numbers (FSR, FWHM, Q, T_min, build-up) and their inversion.
- 13: the exact discretisation of the thermal plant, the discrete PI, the survival map (77
  simulations), `np.linalg.solve` for the crosstalk equilibrium.

**Results.** A few representative ones: phasor identity to 1.9e-15 and crest velocity
1.19917e8 m/s = c/2.5 to 1e-10 relative (01,
[crest_tracking.png](experiments/01_travelling_wave/out/crest_tracking.png)); evanescent
⟨S_x⟩/⟨S_z⟩ = 7e-17 and reactive part γ/β = 0.815 (02,
[transverse_phasor_and_power.png](experiments/02_wavevector_evanescent/out/transverse_phasor_and_power.png));
Gaussian FWHM after 2 km at 1550 nm 20.39 ps from the FFT versus 20.39 ps analytic, 99.9995 %
(05, [gaussian_broadening.png](experiments/05_pulse_dispersion/out/gaussian_broadening.png));
a = 0.9446 from 125 dB/cm versus REF 0.945 (06, 12); ramp error 0.608 pm = 40 000 pm/s / K_v
to 100.00 % (13, [scenario_b_ramp.png](experiments/13_capstone_ring_lock/out/scenario_b_ramp.png)).

**How to observe it.** Every number is printed by `run.py` and written to
`experiments/NN/out/results.json`. The physics helpers are importable on their own, for
example `cd experiments/01_travelling_wave && ../../.venv/bin/python -c "import twave;
print(twave.wave_numbers(1310, 2.7))"` or `cd experiments/10_slab_mode_fdtd && ../../.venv/bin/python
slab_analytic.py`.

**Limitations and gotchas.** The two interpreters carry different numpy versions (2.4.6 and
2.5.3); nothing depends on the difference. Integrating a Poynting profile on the Meep grid with a
pixel mask underestimates the confinement of a mode that peaks near the core edge (10 found 6 %
for TE1): interpolate to a fine grid first, as `core_fraction` in 10 does.

## scipy

**What it is.** Scientific algorithms on top of numpy: root finding and optimisation
(`scipy.optimize`), signal processing (`scipy.signal`), integration, interpolation, special
functions, sparse linear algebra. It is the standard numerical toolbox in scientific Python.

**Version and location.** 1.18.1 in both environments (`.venv/lib/python3.12/site-packages/scipy`).

**Where it is used.**
- `optimize.brentq` (bracketed root finder): the slab TE eigenvalue equations in 07, 10, 11; the
  zero-dispersion wavelength in 04 and 05; the exact half-eye reach in 05.
- `optimize.curve_fit`: the evanescent envelope fit in 02, the TE0/TE1 beat-length cosine fit
  in 10, the ring energy build-up fit in 12.
- `signal.find_peaks`: crests of a snapshot in 01 and 10. `signal.hilbert`: packet envelopes in
  10 and 11. `signal.max_len_seq`, `signal.bessel`, `signal.freqs`: the PRBS and the receiver
  filter in 05.
- `scipy.sparse.linalg` (ARPACK shift-invert) is what femwell calls in 08.
- 03, 09, 13: least-squares fits, interpolated peak and half-power extraction, plant integration.

**Results.** Fitted 1/γ = 102.38 nm versus analytic 102.38 nm (02); ZDW of silica 1272.8 nm
(04, 05, [dispersion_D.png](experiments/04_sellmeier_dispersion/out/dispersion_D.png)); crest
spacing 524.00 nm (01); slab TE0 n_eff = 2.9879 (07,
[07_graphical_construction.png](experiments/07_slab_modes_graphical/out/07_graphical_construction.png));
half-eye reach 2.69 km at 1550 nm to 1 m (05); beat length 1.806 ± 0.003 µm versus 1.815 µm
(10, [10_offcentre_beat.png](experiments/10_slab_mode_fdtd/out/10_offcentre_beat.png)).

**How to observe it.** `out/results.json` of each experiment (keys named after the quantity);
change a bracket or a fit window in the `run.py` section named in that experiment's README and
re-run. `brentq` needs a sign change on the bracket: 07 brackets each mode on
(mπ/2, (m+1)π/2) so no mode can be missed or duplicated.

**Limitations and gotchas.** A coarse sweep only brackets a crossing; 05 refines the reach with
`brentq` on the eye-opening function because the 250 m grid alone would justify one decimal.
Finite-difference second derivatives are fragile (04 shows the D check degrading from 4e-4 to
0.16 ps/(nm km) when the step goes from 1e-4 to 1e-2 µm), which is why the derivatives are taken
symbolically with sympy.

## matplotlib

**What it is.** The standard Python plotting library: static figures, image maps, quiver
plots, 3-D axes, and, through `matplotlib.animation.FuncAnimation` plus `FFMpegWriter`, videos
encoded by ffmpeg. Its mathtext engine renders a TeX subset without a LaTeX installation.

**Version and location.** 3.11.2 in both environments
(`.venv/lib/python3.12/site-packages/matplotlib`). Every script sets `matplotlib.use("Agg")`
before importing pyplot (headless, no windows); 01, 02, 05, 06, 07, 08, 10 (in `run.py`) and
12 (in `ring_sax.py` and `ring_meep.py`) also set
`rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"`, while 03, 04, 09, 11 and 13
rely on `ffmpeg` being on `PATH` (see [ffmpeg / ffprobe](#ffmpeg-and-ffprobe)).

**Where it is used.** Every experiment, for every PNG and every mp4. The shared style comes from
[common/style.py](common/style.py) (`use_style()`: SERIES colour order, `RdBu_r` centred on zero
for signed fields, `Blues` for magnitudes, ffmpeg as the animation writer). Videos: 01 (contact
sheets of the Manim renders), 02 (`plane_wave_tilted.mp4`, `evanescent_sweep.mp4`), 03
(`frequency_sweep.mp4`), 04 (`tangent_sliding.mp4`), 05 (`gaussian_envelope.mp4`,
`nrz_eye_closing.mp4`), 06 (`absorption_vs_evanescence.mp4`), 07
(`07_plane_wave_decomposition.mp4`), 08 (`08_mode_vs_width.mp4`), 09 (`tir_beams.mp4`,
`ftir.mp4`), 10 (`10_te0_travelling_mode.mp4`, `10_offcentre_pulse.mp4`), 11
(`pulse_hopping.mp4`, `cw_phase_zoom.mp4`), 12 (`A5_notch_sliding.mp4`, `B5_buildup.mp4`), 13
(`notch_lock.mp4`, `four_ring_lock.mp4`). Mathtext instead of LaTeX: 04 renders the derived
equations to [equations.png](experiments/04_sellmeier_dispersion/out/equations.png).

**Results.** 118 PNG figures and 19 mp4 videos (each ≤ 20 s at 30 fps, with a
`*_frames.png` contact sheet of 4 to 6 stills) under `experiments/*/out/`. Examples:
[three_ways.png](experiments/01_travelling_wave/out/three_ways.png),
[fig2_poynting.png](experiments/06_absorption_vs_evanescence/out/fig2_poynting.png),
[poynting_three_angles.png](experiments/09_tir_evanescent_fdtd/out/poynting_three_angles.png),
[bode_loop.png](experiments/13_capstone_ring_lock/out/bode_loop.png).

**How to observe it.** Open the PNGs; play the mp4s in QuickTime or VLC. Frame counts, fps and
sweep ranges are constants near each `FuncAnimation` call (`N_FR`, `FPS`, `n_frames`,
`lam_anim`, ...) named in each README's "How to observe it".

**Limitations and gotchas.** (i) `FFMpegWriter` with `yuv420p` needs even frame dimensions;
figure sizes and dpi in 02 were chosen to give even pixel counts. (ii) `tight_layout` is not
compatible with an aspect-equal panel in an animated figure; 03 uses explicit gridspec margins.
(iii) Asking Meep for a frame interval that is not a whole number of time steps rounds up by a
step; 10 computes `FRAME_STEPS` as an integer. (iv) The `findfont: Failed to find font weight
medium` line in logs is harmless (the style asks for a medium title weight that DejaVu Sans does
not ship). (v) `mathtext` covers the subset used here (Greek, sub/superscripts, fractions,
roots); anything beyond that would need LaTeX, which is absent.

## sympy

**What it is.** A pure-Python computer-algebra system: symbolic differentiation, simplification,
series expansion, pretty-printing, and `lambdify` to turn a symbolic expression into a fast numpy
function. It is normally used to derive and check formulas before they are coded numerically.

**Version and location.** 1.14.0, `.venv/lib/python3.12/site-packages/sympy`.

**Where it is used.**
- 04 ([sellmeier.py](experiments/04_sellmeier_dispersion/sellmeier.py)): derives
  n_g = c dk/dω = n - λ dn/dλ and D = -(λ/c) d²n/dλ² from k(ω) = n(λ(ω)) ω/c, the chain rule for
  n = √(1+S), the UV/IR series of one Sellmeier term, and lambdifies n, dn/dλ, d²n/dλ² for
  Malitson silica and Salzberg-Villa silicon so every number is the evaluated derivation.
- 05 ([silica.py](experiments/05_pulse_dispersion/silica.py)): exact β(ω) = n(ω)ω/c and its
  derivatives β1, β2, β3 for the FFT propagator, so no finite-difference error enters it.

**Results.** Both identities assert-simplify to 0
([derivation.txt](experiments/04_sellmeier_dispersion/out/derivation.txt),
[equations.png](experiments/04_sellmeier_dispersion/out/equations.png)); n(1310) = 1.44680
(Malitson 1.4468, 100.00 %), n_g(1310) = 1.4616, D(1310) = +3.582 ps/(nm km) equal to an
independent ω-route finite difference to 100.00 %, D(1550) = 21.91, ZDW 1272.8 nm; β2(1550) =
-27.95 ps²/km equal to -λ²D/(2πc) to 12 digits (05); the finite-difference cross-check of the
lambdified derivatives agrees to 1.1e-9 in n_g.

**How to observe it.** `cd experiments/04_sellmeier_dispersion && ../../.venv/bin/python run.py`
then read `out/derivation.txt`; or `cd experiments/05_pulse_dispersion && ../../.venv/bin/python -c
"import silica; print(silica.symbolic_summary())"`. To add a material, wrap a Sellmeier
expression in `Material(name, n_expr, valid_um)` in `sellmeier.py`; all derivatives follow.

**Limitations and gotchas.** Symbolic simplification of the D identity needs the substitution
S = n² - 1 to be done explicitly. Silicon's Salzberg-Villa fit has its pole at 1.107 µm, so
its expression is only evaluated above 1.2 µm (below the pole the fit is meaningless and the
material absorbs).

## python-control

**What it is.** The Python control-systems library (a MATLAB Control Toolbox work-alike):
transfer functions and state space, series/feedback interconnection, Bode and Nyquist plots,
gain and phase margins, step and forced responses, Padé delay approximations. Used to design and
analyse feedback loops before they are coded.

**Version and location.** 0.10.2, `.venv/lib/python3.12/site-packages/control` (import name
`control`).

**Where it is used.** 13 only ([ringlock.py](experiments/13_capstone_ring_lock/ringlock.py):
`loop_tf`, `design_pi`, `exact_margins`): the linearised ring-lock loop
L(s) = C(s) e^{-sT_d} G_th(s) (50 pm/K)(dI/dδ) with the two-pole thermal plant and a Padé(3)
delay, PI gains by pole cancellation and |L(jω_c)| = 1 at ω_c = 1/(2T_d), `control.margin`,
sensitivity S = 1/(1+L), the ambient-to-error transfer, and linear step and forced responses
overlaid on the nonlinear simulation.

**Results.** Crossover 7958 Hz (target 1/(4πT_d), 100.00 %), phase margin 60.14 deg (hand
formula 60.14 deg; exact-delay check 60.14 deg), gain margin 9.92 dB at 25 kHz, M_s = 1.60,
closed-loop bandwidth 17.9 kHz, K_v = 65 827 1/s; the re-designs at ω_c T_d = 1 (PM 32.1 deg,
M_s 3.16) and T_d = 50 µs (crossover 1591.5 Hz, K_v 12 725 1/s).
[bode_loop.png](experiments/13_capstone_ring_lock/out/bode_loop.png),
[sensitivity.png](experiments/13_capstone_ring_lock/out/sensitivity.png),
[linear_step.png](experiments/13_capstone_ring_lock/out/linear_step.png).

**How to observe it.** `cd experiments/13_capstone_ring_lock && ../../.venv/bin/python run.py`;
the `design` block of `out/results.json`. Change `T_D`, the crossover argument of
`rl.design_pi(..., wc=...)` or `tau_i` in `run.py`; the notebook has sliders for the same.

**Limitations and gotchas.** A Padé approximation of the delay is only accurate below a few
times the crossover; 13 cross-checks every margin against the exact e^{-jωT_d} frequency
response (`exact_margins`) and they agree to 0.01 deg and 0.001 dB. The continuous model of a
sampled loop is conservative: the simulated 5 K step peaks at 128 pm against 139 pm predicted
because the sample-and-hold is effectively faster than e^{-s T_s}.

## Jupyter

**What it is.** JupyterLab is the browser-based notebook environment; `nbconvert` executes and
converts notebooks headlessly; `nbclient` is the executor `nbconvert` uses (usable in-process);
`ipykernel` is the Python kernel. Notebooks mix code, output and prose and are the standard
vehicle for interactive exploration in research.

**Version and location.** JupyterLab 4.6.3, nbconvert 7.17.1, nbclient 0.11.0, ipykernel 7.3.0,
jupyter_client 8.10.0, all in `.venv`; launcher `.venv/bin/jupyter`. Kernel `photonics-sims`
(`~/Library/Jupyter/kernels/photonics-sims/kernel.json`, pointing at `.venv/bin/python`).

**Where it is used.** Experiments 01, 03, 04, 07, 12, 13 each ship an `explore.ipynb` that
`run.py` builds with nbformat and executes with `jupyter nbconvert --to notebook --execute
--inplace --ExecutePreprocessor.kernel_name=photonics-sims` (01 falls back to in-process
`nbclient` if nbconvert exits non-zero; 04 retries and, on a third attempt, skips the cells
tagged `widgets`). The notebooks: 01 sliders for n, sign, t, z0 and the round-trip phasor; 03
sliders for Γ, Ω, B, a cell that runs ngspice for three Q values, the silica mapping and a
carrier-density slider; 04 the sliding tangent, the movable IR pole, the ring FSR and channel
fit, pm/GHz conversion; 07 thickness, wavelength, n1, n2 sliders on the graphical construction
and a cutoff map; 12 sliders for a, t, n_g and temperature plus the FDTD overlay; 13 the sensor,
the loop design with live Bode and margins, the ambient step, the 4-ring crosstalk scenario, and
the DAC/noise study.

**Results.** Six executed notebooks with saved outputs, e.g.
[experiments/13_capstone_ring_lock/explore.ipynb](experiments/13_capstone_ring_lock/explore.ipynb)
(a static figure per section plus the widget state) and
[experiments/04_sellmeier_dispersion/explore.ipynb](experiments/04_sellmeier_dispersion/explore.ipynb)
(asserts its own evaluation equals `out/results.json` to 1e-12). Execution times are recorded in
each `results.json` (`notebook_execution_s`, `notebook_seconds`, `notebook_status`).

**How to observe it.** `cd experiments/NN && ../../.venv/bin/jupyter lab explore.ipynb`, Run All,
drag the sliders. Headless: `../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace
explore.ipynb`. `.venv/bin/jupyter kernelspec list` must show `photonics-sims`.

**Limitations and gotchas.** (i) Intermittent stall: with ipykernel 7.3.0 / ipywidgets 8.1.9 /
jupyter_client 8.10 the headless executor sometimes sits idle for exactly one cell timeout right
after an `interact` cell (the kernel replied within a second; the client picked the reply up at
the timeout). 01 and 04 cap it with `--ExecutePreprocessor.timeout=60`, record a stall flag, and
04 retries without the widget cells; the saved notebooks are complete either way. (ii) The step
is load-sensitive: 04's notebook took 11 minutes once while three Meep jobs saturated the CPU
(`SKIP_NOTEBOOK=1` skips it). (iii) A subprocess launched from inside an ipywidgets callback
deadlocks the headless kernel, so 03's ngspice cell is a plain function call, not a slider.
(iv) `nbconvert --inplace` overwrites the notebook; the source of truth is the `make_notebook.py`
/ `build_notebook.py` script, not the `.ipynb`.

## ipywidgets

**What it is.** Interactive HTML widgets for Jupyter: sliders, dropdowns and buttons that call
a Python function whenever a value changes (`interact`, `FloatSlider`, ...). It turns a plotting
function into a small instrument without writing a GUI.

**Version and location.** 8.1.9, `.venv/lib/python3.12/site-packages/ipywidgets`.

**Where it is used.** The `interact(...)` cells of the six notebooks listed under
[Jupyter](#jupyter); 13 additionally imports it in `ringlock.py`-based helper cells.

**Results.** The slider cells themselves; when executed headlessly they save the figure for the
default slider values (13 calls each plot function once before building the widget so the
static figure is visible on GitHub or in any viewer without a kernel).

**How to observe it.** Open any `explore.ipynb` in JupyterLab and move a slider; the function
named in the `interact` call redraws. Slider ranges are the `min`/`max` arguments in
`make_notebook.py`.

**Limitations and gotchas.** Widget state does not survive headless execution beyond the
default value, and the intermittent executor stall described under Jupyter always follows an
`interact` cell. Widgets in 04 are tagged `widgets` so the retry logic can skip them.

## nbformat

**What it is.** The reference implementation of the Jupyter notebook file format: create
notebooks programmatically (`v4.new_notebook`, `new_code_cell`, `new_markdown_cell`), read,
write and validate them. It is how notebooks are generated from scripts so that the notebook is
reproducible and never hand-edited.

**Version and location.** 5.11.1, `.venv/lib/python3.12/site-packages/nbformat`.

**Where it is used.** `make_notebook.py` in 01, 04, 12, 13 and `build_notebook.py` in 03, 07:
each writes `explore.ipynb` from Python strings, sets the kernel metadata to `photonics-sims`,
and tags cells where needed.

**Results.** The six `explore.ipynb` files.

**How to observe it.** `cd experiments/NN && ../../.venv/bin/python make_notebook.py` (or
`build_notebook.py`) rewrites the notebook; then execute it as under Jupyter. `env/check_env.py`
creates and validates a one-cell notebook in memory.

**Limitations and gotchas.** Edit the builder script, not the notebook: `run.py` deletes and
rebuilds `explore.ipynb` on every run (12 deletes it explicitly).

## Manim

**What it is.** Manim Community Edition, the programmatic animation engine popularised by
3Blue1Brown. Scenes are Python classes; mobjects (text, axes, arrows, curves) are vector
graphics rendered frame by frame with Cairo and encoded by ffmpeg; `ValueTracker` and
`always_redraw` drive continuous animations. It is normally used for explanatory mathematics
videos.

**Version and location.** Manim Community v0.21.0, `.venv/lib/python3.12/site-packages/manim`,
CLI `.venv/bin/manim`.

**Where it is used.** 01 only ([scene.py](experiments/01_travelling_wave/scene.py):
`TravellingWaveScene`, `SignConventionScene`), invoked by `run.py` as `manim -qm
--disable_caching -o <name>.mp4 scene.py <Scene>`; the result under `media/videos/...` is copied
into `out/`.

**Results.** [travelling_wave.mp4](experiments/01_travelling_wave/out/travelling_wave.mp4)
(13.3 s, 1280x720, 30 fps): the crest dot advances exactly one guided wavelength (524 nm) per
period (4.37 fs), the phasor at z0 rotates once per period with its projection drawing the time
trace, and the λ_g bracket is anchored to the tracked crest;
[sign_convention.mp4](experiments/01_travelling_wave/out/sign_convention.mp4) (9.0 s): cos(ωt - βz)
and cos(ωt + βz) side by side with crests moving at ±1.199e8 m/s. Stills:
[travelling_wave_frames.png](experiments/01_travelling_wave/out/travelling_wave_frames.png),
[sign_convention_frames.png](experiments/01_travelling_wave/out/sign_convention_frames.png).
Render times 40 s and 17 s (`videos.*.render_s` in `results.json`).

**How to observe it.** `cd experiments/01_travelling_wave && ../../.venv/bin/python run.py`
renders both; one scene alone: `../../.venv/bin/manim -qm --disable_caching scene.py
TravellingWaveScene` (output under `media/videos/scene/720p30/`). Change `REF.neff` or
`REF.lambda_nm` in `common/params.py`, or `z0_nm` / `Z_MAX_NM` in `scene.py`.

**Limitations and gotchas.** (i) No LaTeX on this machine: `MathTex` and `Tex` fail. Every
label is `Text()` with Unicode (ω, β, λ, subscripts), and `Axes(..., label_constructor=Text)`
is required or the axis labels try to use LaTeX. (ii) `Text()` kerning swallows a single space
after "+z"; the scene uses a double space. (iii) The z window must be an integer number of λ_g
(3 λ_g = 1572 nm) or a tracked crest falls off its crest after wrapping. (iv) `--disable_caching`
avoids stale partial-movie files; `experiments/*/media/` and `partial_movie_files/` are
git-ignored. (v) Render time depends strongly on machine load (15 to 80 s per scene).

## ngspice

**What it is.** The open-source SPICE circuit simulator (Berkeley SPICE3 lineage, with the KLU
sparse solver): DC operating point, AC small-signal, transient and noise analyses of a text
netlist. It is the standard free tool for analogue circuit simulation and is run headlessly here
in batch mode.

**Version and location.** ngspice-47, `/opt/homebrew/bin/ngspice` (Homebrew).

**Where it is used.** 03 only: the series RLC (L = 100 mH, R = 62.83 Ω, C = 253.3 nF, f0 =
1 kHz, Q = 10) that is term-for-term the Lorentz bound-electron equation. `run.py` writes
netlists with a `.control ... run ... wrdata ... quit .endc` block, runs `ngspice -b file.cir`,
and parses the whitespace tables with numpy: an AC sweep 10 Hz to 100 kHz (4001 points) of the
capacitor voltage (charge = C v) and the loop current, two more AC sweeps at Q = 2 and Q = 50,
and three transients at Ω = 0.1, 1, 3 with the instantaneous powers R i² and V i. The notebook
also writes and runs a netlist for any (f0, Q).

**Results.** |χ̂| within 0.00014 % and phase within 0.0001 deg of A/(ω0² - ω² + jγω) over four
decades; lag at f0 90.000 deg; displacement peak 997.49 Hz versus f0√(1 - Γ²/2) = 997.50 Hz;
power FWHM 100.02 Hz versus γ/2π = 100.00 Hz; peak |χ̂| 2.0656 / 10.0124 / 49.9999 for Q = 2 /
10 / 50; transient lags 0.579 / 90.018 / 177.848 deg; mean resistor power at resonance 7.9505 mW
versus ½V0²/R = 7.9577 mW. [ac_sweep.png](experiments/03_driven_electron_rlc/out/ac_sweep.png),
[transients.png](experiments/03_driven_electron_rlc/out/transients.png),
[phasor_power.png](experiments/03_driven_electron_rlc/out/phasor_power.png); netlists
[rlc_ac.cir](experiments/03_driven_electron_rlc/out/rlc_ac.cir), `rlc_tran_{low,res,high}.cir`
with their `.txt` tables and `.log` files.

**How to observe it.** `cd experiments/03_driven_electron_rlc && ../../.venv/bin/python run.py`;
one netlist by hand: `/opt/homebrew/bin/ngspice -b out/rlc_ac.cir`. Change `Q_FACTOR`, `F0` or
the `DRIVES` dictionary at the top of `run.py`.

**Limitations and gotchas.** (i) ngspice reports `i(v1)` flowing into the source's + terminal,
so the loop current is `-i(v1)`. (ii) Transients need a settling time: 8/γ left the resonant
amplitude 1 % low; 16/γ (e^{-8}) is used. (iii) ngspice writes its own internal time points, so
the last periods are resampled uniformly before fitting. (iv) `wrdata` tables have a header-free
whitespace layout that changes with the number of vectors; parse by column index. (v) Do not call
it from an ipywidgets callback (see Jupyter).

## Meep

**What it is.** MIT's open-source finite-difference time-domain (FDTD) Maxwell solver
(pymeep is its Python interface). It steps E and H on a Yee grid in time with PML absorbing
boundaries, Bloch-periodic boundaries, dispersive and conductive materials, point, line,
Gaussian-beam and eigenmode sources, and can Fourier-transform the running fields into
single-frequency phasors (`add_dft_fields`), integrate power through surfaces (`add_flux`) and
decompose fields into modes. It is the standard free tool for photonic component simulation and
embeds MPB (eigenmodes) and Harminv (resonance extraction).

**Version and location.** meep 1.34.0 (conda-forge pymeep) in `.meep`
(`.meep/lib/python3.12/site-packages/meep`, `libmeep.38.dylib` in `.meep/lib`). Interpreter
`.meep/bin/python`; the six experiments that use it carry a `.uses_meep` marker.

**Where it is used.**
- 06: a 1-D cell with n = n' - jn'' entered as `D_conductivity` (`sims.meep_lossy_medium`:
  ε∞ = n'² - n''², σ_D = ω 2n'n''/ε∞), DFT phasors of E_x, H_y; a 2-D Si/SiO2 interface made
  Bloch-periodic with `k_point` = β so that only the x-dependence is physics, DFT phasors of
  E_y, H_x, H_z and ⟨S⟩ from them.
- 09: 61 runs (55 phasor, 6 continuous-wave) of a `GaussianBeamSource` (waist 2 µm) hitting a
  flat interface at 12 angles, with `add_dft_fields` and `add_flux` for T and R, a second silicon
  block for frustrated TIR at 5 gaps, and a resolution check at 30/40/50/60/80 px/µm; six at a
  time in a process pool.
- 10: `EigenModeSource` of TE0 (and TE1 in a 400 nm slab) into a 220 nm slab, DFT phasors for
  n_eff, λ_g, Γ, ⟨S⟩; an off-centre Gaussian pulse decomposed by overlap and by Meep's own
  `get_eigenmode_coefficients` at four flux planes; real-time frames for the videos and the
  arrival-time group index.
- 11: two slabs at gaps 150/200/300 nm, `EigenModeSource` in guide 1, DFT flux in each
  half-plane versus z fitted to sin²(κ_c z); a pulse run for the hopping video.
- 12 (Part B, `ring_meep.py`): a 2-D ring (R = 6.3 µm, 500 nm wide, 100 nm gap) with the doped
  loss calibrated through `D_conductivity` on a straight guide, a lossless gap scan measuring κ²
  from the first-pass flux, a pulsed through-spectrum with Harminv, and a CW build-up movie.

**Results.**
- 06: α_FDTD/α_analytic = 1.0035 at four n'' (numerical dispersion, same at all four); one lap at
  125 dB/cm gives |E(L)|/|E(0)| = 0.9444 versus a = 0.9446; interface: 1/γ = 102.6 nm versus 102.4
  nm, ⟨S_x⟩/⟨S_z⟩ = 1e-8, H_z/E_y phase -90.000 deg
  ([fig2_poynting.png](experiments/06_absorption_vs_evanescence/out/fig2_poynting.png)).
- 09: T at 15 deg 0.7516 versus 0.7484 beam average; tail 123.5 nm versus 121.2 nm (plane wave)
  and 123.3 nm (finite beam) at 40 deg; tunnelling slope 18.6 /µm versus 2γ = 19.5 /µm; a +10 nm
  gap multiplies the tunnelled power by 0.83 (analytic 0.82)
  ([evanescent_decay.png](experiments/09_tir_evanescent_fdtd/out/evanescent_decay.png),
  [ftir_vs_gap.png](experiments/09_tir_evanescent_fdtd/out/ftir_vs_gap.png)).
- 10: TE0 n_eff 3.0024 versus 2.9879 (+0.48 %, converging +0.88 / +0.48 / +0.11 % at 30/50/80
  px/µm), Γ 0.8584 versus 0.8623, 1/γ 79.9 versus 79.8 nm, ⟨S_x⟩ zero to 8.8e-6; off-centre
  launch into TE0 0.204 + TE1 0.231 + radiation, flat along z
  ([10_offcentre_decomposition.png](experiments/10_slab_mode_fdtd/out/10_offcentre_decomposition.png)).
- 11: κ_c = 0.2362 / 0.1266 / 0.0364 rad/µm at 150/200/300 nm (102 to 104 % of analytic, within
  1.5 % of the same-grid eigenmode value), P1 + P2 constant to 0.008 %
  ([power_vs_z.png](experiments/11_directional_coupler/out/power_vs_z.png)).
- 12: modal loss calibrated to 125.0 dB/cm; κ² = 0.0221 / 0.0054 / 0.0014 at 100/150/200 nm
  (slope -27.6 /µm versus -2γ = -28.9); FSR 11.73 nm (2-D n_g); central notch at 1313.09 nm
  with T_min 0.656, FWHM 312.5 pm (interpolated; 327.7 pm on the raw 17.2 pm grid), Q 4201
  (flux; 4007 on the raw grid) / 4391 (Harminv) / 4509 (CW build-up), inverted to t = 0.992 and
  a = 0.929 (calibrated 0.945: the ring loses 0.087 dB/lap more than the straight guide);
  build-up τ = 6.29 ps versus Qλ/(πc) = 6.12 ps
  ([B3_gap_scan.png](experiments/12_round_trip_to_ring/out/B3_gap_scan.png),
  [B4_spectrum.png](experiments/12_round_trip_to_ring/out/B4_spectrum.png)).

**How to observe it.** `cd experiments/NN && ../../.meep/bin/python run.py` (06 about 100 s, 09
about 5 min, 10 about 2 min, 11 about 2 to 3 min, 12 Part B 408 s). Resolution constants
(`RES`, `RES_1D`/`RES_2D`, `RES_SHOW`, `RES_MAIN`) and geometry (`THICK`, `GAPS_FDTD`, `GAP`,
`SHOW_ANGLES`) sit at the top of each `run.py`; every README lists which ones to try.
`mp.verbosity(0)` is set everywhere; the bare "Elapsed run time" lines in logs are Meep's own.

**Limitations and gotchas.**
- Units: lengths in µm, frequency = 1/λ[µm] (1310 nm is 0.7634), time in µm/c (one optical
  period is 1.31 time units), ε (not n) in `Medium`. `D_conductivity` is defined by
  ε(ω) = ε∞(1 + iσ_D/ω) in Meep's e^{-iωt} convention; 06 checked the mapping empirically at four
  n'' values before trusting it, and 12 found that the bulk guess σ_D = α/n overshoots the modal
  loss by 4 % (part of the mode is in the lossless cladding), so it calibrates σ_D on a straight
  guide.
- Sign convention: Meep's DFT phasors are e^{-iωt}; every experiment conjugates them once so
  that a +z wave is e^{-jβz} as in the notes. The independent test is the -90 deg phase of
  H_z/E_y in an evanescent tail (06), which would read +90 deg without the conjugation.
- Meep's flux monitor returns Re∫E x H* without the ½ (09 and 10 verified the factor 0.5000
  against the field-derived ½Re∫E x H*); ratios cancel it.
- 2-D TE-like slab modes use E_z out of plane; the notes' (x, y, z) map to Meep's (y, z, x) and
  every README states the mapping. Eigenmode parity `EVEN_Y + ODD_Z` selects TE0.
- Numerical dispersion: n_eff comes out 0.1 to 0.9 % off on 40 to 100 px/µm grids and converges
  as resolution^-2 (06, 10, 11); differences of two β on the same grid (κ_c) are much less
  affected. Resolutions were chosen to keep each run inside the 10-minute budget (30 to 100
  px/µm, PML 1 µm, 2-D cells of 5 to 40 µm); 09 documents a resolution^-1.4 convergence of T at
  the critical angle and 12 notes that 30 px/µm puts only 3 pixels across its 100 nm gap.
- A Bloch cell with `k_point` = β forces the z-dependence, so "|E| constant along z" and
  ∇·S = 0 are properties of the boundary condition, not physics (06 lists them as consistency
  rows); a finite beam (09) is the non-imposed test.
- Grid coordinates of DFT arrays: use `sim.get_array_metadata(dft_cell=...)`, not `np.linspace`
  over the nominal size (Meep's DFT arrays include one extra Yee point; a linspace biases every
  fitted slope by about 0.5 %).
- `get_array` on a simulation that has not run yet segfaults; call `sim.init_sim()` first (10).
- A source placed inside the PML silently launches nothing useful (06's first attempt); keep an
  explicit margin layer between PML and source.
- The truncated ends of a finite source line launch weak wide-angle components that put a
  numerical floor of 3e-5 to 4e-4 under transmissions that should be zero (09).
- Meep's 2-D 500 nm guide of index 3.5 is multimode and confines much more strongly than the
  3-D strip (n_eff 3.34, n_g 3.59, 1/γ 69 nm), so 12's FDTD ring is not the reference ring; the
  README traces every difference.
- Resonances move by more than one linewidth between grids: keep the CW movie at the same
  resolution as the spectrum run (`RES_CW = RES_MAIN`, 12).
- pymeep on conda-forge here is the serial build; 09 parallelises over runs with
  `multiprocessing` instead of MPI.

## MPB eigenmode solver

**What it is.** MPB (MIT Photonic Bands) is a plane-wave frequency-domain eigensolver for the
modes of periodic and waveguide structures. Inside Meep it is exposed as
`Simulation.get_eigenmode` (modes on a line or plane of the FDTD grid at a given frequency,
returning β, group velocity and field profiles), `EigenModeSource` (launch a clean mode) and
`get_eigenmode_coefficients` (project FDTD fields onto modes). It is what every "mode source"
and "mode monitor" in an FDTD workflow relies on.

**Version and location.** Bundled with meep 1.34.0 (`libmpb.6.dylib` in `.meep/lib`); no separate
Python package is used.

**Where it is used.**
- 07: `get_eigenmode` on a 1-D line across the 220 nm slab (`EVEN_Y` for TE0, `ODD_Y` for TE1)
  at 100 px/µm, at six thicknesses (150 to 650 nm, up to TE3) and four resolutions (25 to 200
  px/µm), plus Meep's group velocity for n_g.
- 10: `EigenModeSource` (eig_band 1 and 2) and `get_eigenmode_coefficients` at four planes.
- 11: β of the single slab and β± of the even/odd supermodes at six gaps on the FDTD grid
  (`fdtd_coupler.eigen_supermodes`), plus the `EigenModeSource` launch.
- 12: n_eff, n_g (from `group_velocity` and by finite difference), the profile and the second
  mode of the 2-D 500 nm guide (section B1), and the `EigenModeSource` in the bus.

**Results.** 07: TE0 n_eff 2.98675 versus analytic 2.98795 (99.96 %), TE1 1.48512 versus
1.48532, n_g 3.6306 versus 3.6323, 14 mode/thickness pairs within 0.073 %, error falling from
-0.31 % at 25 px/µm to -0.010 % at 200 px/µm
([07_meep_validation.png](experiments/07_slab_modes_graphical/out/07_meep_validation.png)). 10:
Meep's coefficients agree with the analytic overlap decomposition within 0.006 of the launched
power at all planes. 11: single-slab n_eff 2.9607 versus 2.9879 (-0.9 % on a 40 px/µm grid) but
κ_c = (β+ - β-)/2 = 0.1248 rad/µm at 200 nm, 101.3 % of analytic
([supermode_beat.png](experiments/11_directional_coupler/out/supermode_beat.png),
[kappa_vs_gap.png](experiments/11_directional_coupler/out/kappa_vs_gap.png)). 12: n_eff 3.339,
n_g 3.594 (finite difference 3.594), 1/γ 69 nm, second mode 2.825
([B1_mode2d.png](experiments/12_round_trip_to_ring/out/B1_mode2d.png)).

**How to observe it.** The `meep` block of 07's `results.json` (`modes`,
`te0_resolution_convergence`, `thickness_sweep_points`); change `res=100` in the `meep_modes(...)`
calls. In 11, edit `GAPS_EIG`. In 12, change `W` or `N_SI` in `ring_meep.py`.

**Limitations and gotchas.** The eigensolver on a coarse grid lowers every n_eff (a 220 nm core
is 8.8 pixels at 40 px/µm: -0.026 in n_eff), but the splitting between two modes on the same grid
is far less sensitive. `kpoint` is only the solver's starting guess; change it for a very
different index contrast. Mode normalisation differs from the analytic one; profiles are
compared after one complex normalisation. The mode-plane (PEC) boundary matters for a mode near
cutoff (08's TE1, 07's TE1 with its 647 nm tail).

## Harminv

**What it is.** A harmonic-inversion (filter-diagonalisation) routine that extracts the complex
frequencies (resonance frequency, decay rate, hence Q) and amplitudes of the modes present in a
short time signal, far more accurately than a Fourier transform of the same record. It ships with
Meep as `mp.Harminv` and as a command-line program.

**Version and location.** Bundled with meep 1.34.0 (`libharminv.3.dylib`, `.meep/bin/harminv`).

**Where it is used.** 12 only (`ring_meep.py`): E_z at a point on the ring centre-line after a
Gaussian pulse, in the lossless gap scan (coupling Q) and in the lossy main run (loaded Q and the
resonance wavelength that seeds the CW movie); a filter keeps the fundamental family (strong
amplitude, Q > 300, duplicates within 0.5 nm merged).

**Results.** Fundamental family at 1290.06 / 1301.48 / 1313.11 / 1324.92 / 1336.97 nm with
Q = 4248 / 4341 / 4391 / 4214 / 3362; the Harminv FSR (11.73 nm) equals the flux-dip FSR to
0.01 nm and the resonance wavelengths match the dips to 0.02 nm; Harminv Q 4391 versus the
flux-notch Q 4201 (4007 on the raw grid; the notch is broadened by the finite 7000 µm/c record,
3.81 amplitude lifetimes, and by the 17.2 pm flux grid) and the CW build-up Q 4509. Lossless scan Q_c = 16 100 / 48 400 / 110 600 at 100/150/200 nm. Dashed lines with Q
labels in [B4_spectrum.png](experiments/12_round_trip_to_ring/out/B4_spectrum.png); full list
under `harminv_all` in `out/B_results.json`.

**How to observe it.** `cd experiments/12_round_trip_to_ring && ../../.meep/bin/python
ring_meep.py`; move `RING_PT` or the bandwidth in `mp.Harminv(mp.Ez, RING_PT, F0, 0.05)`.

**Limitations and gotchas.** Harminv sees every mode the probe point excites: the 2-D guide is
multimode, so a filter on amplitude and Q is needed, and converting the lossless-ring Q_c to κ²
overestimates the first-pass value 2 to 4x because that Q also contains radiation and
coupler-junction loss. Harminv is the more trustworthy Q when the time record is short (3.8
amplitude lifetimes here).

## Tidy3D mode solver

**What it is.** Tidy3D is Flexcompute's FDTD package (the FDTD itself runs in their cloud). Its
`tidy3d.plugins.mode.ModeSolver` is a finite-difference frequency-domain eigenmode solver on a
Yee grid that runs locally without an account, giving n_eff, a built-in finite-difference group
index, TE fraction and full-vector field arrays of a waveguide cross-section. It is normally used
to define mode sources and monitors for Tidy3D FDTD simulations.

**Version and location.** tidy3d 2.12.0, `.venv/lib/python3.12/site-packages/tidy3d`.

**Where it is used.** 08 only ([solvers.py](experiments/08_soi_strip_mode_solvers/solvers.py)
`solve_tidy3d`): the 500 x 220 nm strip on a uniform 10 nm grid over a 2.61 x 2.01 µm plane, 4
modes; the width sweep 300 to 700 nm, the wavelength sweep 1260 to 1360 nm with Sellmeier
materials, and the thermo-optic re-solve with n_Si + 1.86e-4 x 10 K and n_SiO2 + 1e-5 x 10 K.

**Results.** TE0 n_eff = 2.7134 (femwell 2.7119, spread 0.06 %; textbook 2.5 is 7.8 % low), TE
fraction 0.993, Γ_P = 0.853 (REF 0.85), n_g,wg = 3.964, n_g total = 4.158 (REF 4.2, -1.0 %),
dn_eff/dT = 2.017e-4 /K, dλ_r/dT = 63.6 pm/K (REF 50, +27 %); TE1 cutoff width 323 nm; TM0
n_eff 2.188 (femwell 2.166, 1 % apart).
[08_mode_fields.png](experiments/08_soi_strip_mode_solvers/out/08_mode_fields.png) (top row),
[08_mode_gallery.png](experiments/08_soi_strip_mode_solvers/out/08_mode_gallery.png),
[08_width_sweep.png](experiments/08_soi_strip_mode_solvers/out/08_width_sweep.png),
[08_thermo_optic.png](experiments/08_soi_strip_mode_solvers/out/08_thermo_optic.png).

**How to observe it.** `cd experiments/08_soi_strip_mode_solvers && ../../.venv/bin/python run.py`
(about 5 minutes); keys `reference_modes.tidy3d`, `group_index`, `thermo_optic.per_solver.tidy3d`
in `out/results.json`. Change `TD_DL` (grid pitch) or `TD_BOX` at the top of `run.py`, or call
`solve_tidy3d(width, height, lam, n_core, n_clad, ...)` from `solvers.py`. The known-good snippet
is in [docs/BRIEF.md](docs/BRIEF.md).

**Limitations and gotchas.** (i) The local solver warns about subpixel averaging on every call;
that is expected (08 sets `td.config.logging.level = "ERROR"` and wraps calls in
`warnings.catch_warnings()`). (ii) Grid alignment matters because there is no subpixel averaging:
with a 2.6 x 2.0 µm plane at 10 nm pitch the core edges fell on grid lines and n_eff jittered by
±0.005 between geometries with a +0.37 % bias; sizing the plane 2.61 x 2.01 µm (an odd multiple
of the pitch) puts every edge mid-cell and the bias drops to +0.05 %. Keep the box an odd multiple
of the pitch. (iii) TM modes, whose dominant component is normal to the large top/bottom faces,
feel the staircasing more (1 % off femwell); femwell is preferred for TM. (iv) The mode-plane
boundary is PEC, which does not matter for TE0 but does for near-cutoff TE1. (v) Sellmeier
materials for the wavelength sweep are entered per wavelength as constant-permittivity media.
(vi) Import prints a banner and may try to check for updates; nothing here needs a network
connection or an account.

## femwell

**What it is.** An open-source finite-element photonics package built on scikit-fem. It meshes a
2-D cross-section with gmsh (triangles that conform exactly to material boundaries) and solves
the vector Maxwell eigenproblem with Nédélec elements for the transverse field and Lagrange
elements for the longitudinal one; it also provides overlap, power and perturbation ("confinement
factor") integrals. Used in research for waveguide modes, coupling coefficients, thermal and
electro-optic overlaps.

**Version and location.** femwell 0.1.12, `.venv/lib/python3.12/site-packages/femwell`
(with scikit-fem 12.0.2, gmsh 4.15.2, shapely 2.1.2, meshio 5.3.5).

**Where it is used.** 08 only (`solvers.py`: `femwell_mesh`, `solve_femwell`): the same strip
on a boundary-conforming mesh (5674 triangles, 10 nm elements in the core, 2nd-order elements),
`compute_modes` for 3 to 4 modes; `calculate_power` restricted to the core elements for Γ_P;
`calculate_confinement_factor` for the exact sensitivity ∂n_eff/∂n_Si; n_g by central finite
difference ±5 nm; the width sweep (which also supplies the frames of the video); n_eff(T) at
seven temperatures for linearity; the E_x probe 0.1 nm either side of the sidewall.

**Results.** TE0 n_eff = 2.7119 (converged to 1e-5 between 20 and 5 nm core meshes; 1st to
2nd-order elements move it by 1e-3), TE fraction 0.993, Γ_P = 0.852, Γ_E = 0.951, S_Si =
∂n_eff/∂n_Si = 1.077 (direct re-solve 1.077; equal to Γ_E n_g,wg/n_Si to < 0.1 %), S_SiO2 = 0.134,
n_g,wg = 3.963, n_g total = 4.157, dλ_r/dT = 63.6 pm/K, TE1 cutoff width 328 nm, E_x jump at the
wall 5.77 (ε ratio 5.83), n_eff(T) linear to 1.3e-6 over 115 K.
[08_profile_cuts.png](experiments/08_soi_strip_mode_solvers/out/08_profile_cuts.png),
[08_mode_vs_width.mp4](experiments/08_soi_strip_mode_solvers/out/08_mode_vs_width.mp4) (21
femwell solutions between 300 and 700 nm width).

**How to observe it.** Same `run.py`; bottom rows of `08_mode_fields.png` and
`08_mode_gallery.png`. Change `FEM_RES_REF` / `FEM_RES_SWEEP` in `run.py`, or the `box` /
`resolution` arguments of `femwell_mesh` in `solvers.py`; gmsh rebuilds the mesh each run. The
one-liner from 08's README: `solve_femwell(femwell_mesh(0.5, 0.22, 0.01), 1.31, 3.50 + 1e-3,
1.45)[0].n_eff` minus the unperturbed value, divided by 1e-3, gives 1.077 (not 0.85).

**Limitations and gotchas.** (i) `mesh_from_OrderedDict(polys, resolutions, ...)` takes the
resolution as a dict keyed by polygon name with `{"resolution": <element size>, "distance":
<falloff>}` values, and the polygons as an `OrderedDict` (order sets material precedence);
`default_resolution_max` bounds the far-field elements. Passing a plain float fails. (ii) The
mesh is built through `skfem.io.meshio.from_meshio`, and the permittivity is assigned on an
`ElementTriP0` basis with `basis0.get_dofs(elements=<name>)`. (iii) 1st-order elements are 1e-3
off in n_eff; use `order=2`. (iv) `calculate_confinement_factor` is the perturbation
sensitivity, not a power fraction (08 confirmed by direct re-solve). (v) A femwell solve takes a
few seconds; the width sweep dominates 08's runtime.

## scikit-fem

**What it is.** A lightweight pure-Python finite-element assembly library: meshes, element
families (P0, P1, P2, Nédélec, ...), bases, and assembly of bilinear forms into scipy sparse
matrices. femwell builds its Maxwell eigenproblem on it.

**Version and location.** scikit-fem 12.0.2, `.venv/lib/python3.12/site-packages/skfem` (import
name `skfem`).

**Where it is used.** 08, through femwell and directly for `Basis(mesh, ElementTriP0())`, the
permittivity vector `eps[basis0.get_dofs(elements="core")] = n²`, and `from_meshio`.

**Results.** The femwell results above; the mesh statistics (5674 triangles) in 08's log.

**How to observe it.** `env/check_env.py` builds a P0 basis on a refined `MeshTri`; 08's
`solvers.py` shows the pattern.

**Limitations and gotchas.** Element degrees of freedom are per element for P0, so the
permittivity is piecewise constant per triangle; a boundary-conforming mesh is what makes that
exact.

## gmsh

**What it is.** A 3-D finite-element mesh generator with a built-in CAD kernel (OCC) and a
Python API. It produces the triangle meshes femwell solves on.

**Version and location.** gmsh 4.15.2 (Python wheel), `.venv/lib/python3.12/site-packages/gmsh.py`
with the bundled library.

**Where it is used.** 08, called by `femwell.mesh.mesh_from_OrderedDict`.

**Results.** The 08 meshes (10 nm elements in the core, up to 250 nm at the box edge).

**How to observe it.** `env/check_env.py` initialises gmsh, meshes a unit square and finalises.
Mesh size is the `resolution` dict passed by `femwell_mesh`.

**Limitations and gotchas.** `gmsh.initialize()` / `gmsh.finalize()` must be paired; femwell
handles this internally. gmsh prints to the terminal unless `General.Terminal` is 0. The mesh is
rebuilt on every call (no caching), which is fine at a few seconds per mesh.

## meshio

**What it is.** A reader/writer for mesh file formats (Gmsh, VTK, XDMF, ...) and an in-memory
`Mesh` object. It is the bridge between gmsh's output and scikit-fem's `from_meshio`.

**Version and location.** meshio 5.3.5, `.venv/lib/python3.12/site-packages/meshio`.

**Where it is used.** 08, inside femwell's mesh pipeline (`skfem.io.meshio.from_meshio`).

**Results.** No file of its own; the meshes stay in memory.

**How to observe it.** `env/check_env.py` builds a one-triangle `meshio.Mesh`.

**Limitations and gotchas.** None hit.

## shapely

**What it is.** Planar geometry (points, polygons, boxes, unions, buffers) on the GEOS library.
It is how femwell's cross-section polygons are defined, and gdsfactory uses it too.

**Version and location.** shapely 2.1.2, `.venv/lib/python3.12/site-packages/shapely`.

**Where it is used.** 08: `shapely.box(-w/2, -h/2, w/2, h/2)` for the core and a larger box for
the cladding, in an `OrderedDict` for `mesh_from_OrderedDict`. (11 reads gdsfactory polygons
back as numpy arrays rather than shapely objects.)

**Results.** The 08 geometry.

**How to observe it.** `env/check_env.py` checks the area of the 500 x 220 nm core box.

**Limitations and gotchas.** Polygon order in the `OrderedDict` decides which material wins
where polygons overlap (core first).

## SAX

**What it is.** The JAX-based S-parameter circuit simulator of the gdsfactory ecosystem.
Components are Python functions returning S-dictionaries keyed by port pairs
(`sax.reciprocal({...})`), a netlist connects the ports, and `sax.circuit` composes the
frequency-domain response of the whole circuit, vectorised over wavelength or any model
parameter and differentiable. It is used to simulate photonic integrated circuits (rings,
Mach-Zehnders, filters) from per-component models.

**Version and location.** sax 0.18.2 (with jax 0.9.2), `.venv/lib/python3.12/site-packages/sax`.

**Where it is used.**
- 11 ([sax_part.py](experiments/11_directional_coupler/sax_part.py), run with `.venv`): the 2 x 2
  coupler [[t, -jK], [-jK, t]] with t = cos(κ_c L), K = sin(κ_c L); a numerical unitarity and
  reciprocity check; |t|², |K|² versus length; two coherent inputs through the S-matrix and
  through coupler, phase shifter, coupler (an MZI); the all-pass ring (coupler + 39.6 µm doped
  waveguide) at κ² for gap errors of -10 / 0 / +10 nm.
- 12 ([ring_sax.py](experiments/12_round_trip_to_ring/ring_sax.py), Part A): the same ring
  circuit swept 1290 to 1332 nm in 1 pm steps for the doped (125 dB/cm) and passive (3 dB/cm)
  rings and three couplers, the coupling regimes, and the thermal slide at a fixed laser.

**Results.** 11: max |S†S - I| = 2e-16, cross-port phase -90.0 deg, |s1 + s2|² swings 0 to 2 W
while |s1|² + |s2|² stays 1 W, MZI equals sin²(φ/2) to 7e-16, ring FSR 10.33 nm, FWHM 368 pm
(1 pm grid; analytic 372, REF 374), T_min 0.0099 / 0.0098 for ±10 nm of gap
([sax_interference.png](experiments/11_directional_coupler/out/sax_interference.png),
[critical_coupling_lottery.png](experiments/11_directional_coupler/out/critical_coupling_lottery.png)).
12: a = 0.9446, FSR 10.318 nm = 1.803 THz (REF 10.3 / 1.8), FWHM 374.0 pm (REF 374), Q 3503
(REF 3500), T_min 1.4e-5 at t = a, SAX versus the closed form to 3e-6, one FSR = 206.4 K,
0.1 K = 5.0 pm, slope -0.174 per K at δ_opt, heater 0.440 nm/mW
([A2_spectrum.png](experiments/12_round_trip_to_ring/out/A2_spectrum.png),
[A4_thermal.png](experiments/12_round_trip_to_ring/out/A4_thermal.png),
[A5_notch_sliding.mp4](experiments/12_round_trip_to_ring/out/A5_notch_sliding.mp4)).

**How to observe it.** `cd experiments/11_directional_coupler && ../../.venv/bin/python
sax_part.py` (numbers in `out/sax_results.json`) or `cd experiments/12_round_trip_to_ring &&
../../.venv/bin/python ring_sax.py` (about 20 s: 18.6 s this run, `out/A_results.json`). In 12 every knob is a
keyword of the models: `ring_T(wl, coupling=..., loss_db_cm=..., ng=...)`; in 11 the single
constant `A_RT` feeds both the closed form and the waveguide loss. The snippet in
[docs/BRIEF.md](docs/BRIEF.md) is the minimal ring.

**Limitations and gotchas.** (i) JAX defaults to float32: the residual between SAX and the closed
form (3e-6 in the script, 6.5e-5 in the notebook) is float32 acting on a round-trip phase of
2π x 76 ≈ 477 rad, not physics. (ii) The waveguide model must carry the group index explicitly,
n(λ) = n_eff - (λ - λ0)(n_g - n_eff)/λ0, or the FSR comes out from n_eff (wrong by 1.7x). (iii)
The resonance positions depend on n_eff, which is only known to 1 %; 12 nudges n_eff to 2.514 to
put a resonance at 1310.000 nm and only quotes differences (FSR, FWHM, depth). (iv) A 0.0004
change in a moves T_min visibly when t - a is the quantity of interest (11 found 0.0092 / 0.0105
with a = 0.9446 versus 0.0099 / 0.0098 with a = 0.945), so a and the loss must come from one
constant.

## JAX

**What it is.** Google's array library with automatic differentiation and XLA compilation; `jax.numpy`
mirrors numpy. SAX is built on it so that circuit responses are vectorised and differentiable.

**Version and location.** jax 0.9.2, jaxlib 0.9.2, `.venv/lib/python3.12/site-packages/jax`
(CPU backend).

**Where it is used.** 11 and 12, inside the SAX component models (`jnp.exp(-2j π n L/λ)`,
`jnp.abs`, array sweeps over wavelength). Nothing calls `jax.grad` yet.

**Results.** The SAX results above.

**How to observe it.** `env/check_env.py` evaluates |e^{-jβL}| = 1 with `jax.numpy` and reports
the backend.

**Limitations and gotchas.** float32 by default (see SAX); `jax.config.update("jax_enable_x64",
True)` would remove the 3e-6 residual at the cost of speed. The first call compiles (a second or
two). On this Mac the backend is CPU; a `JAX_PLATFORMS=cpu` environment variable silences the
"no GPU" note.

## gdsfactory

**What it is.** A Python layout generator for photonic (and other) chips built on KLayout
(`kfactory`, `klayout`): parametric components such as rings, couplers, bends and MMIs with ports
and cross-sections, a PDK concept, GDSII export, and polygon read-back. It is the open-source
standard for photonic layout and the companion of SAX.

**Version and location.** gdsfactory 9.51.0 (kfactory 3.0.4, klayout 0.30.12),
`.venv/lib/python3.12/site-packages/gdsfactory`.

**Where it is used.** 11 only ([gds_part.py](experiments/11_directional_coupler/gds_part.py),
run with `.venv`): `gf.gpdk.PDK.activate()`, then `gf.components.ring_single(radius=6.3,
gap=0.2, length_x=0.01, length_y=0.01, bend="bend_circular", ...)` (the capstone's ring-bus point
coupler with 500 nm strips) and `gf.components.coupler(gap=0.2, length=12.76, dy=3, dx=6,
cross_section="strip")` (a straight directional coupler with its parallel section equal to
L_c(200 nm)), both written to GDS; the polygons are read back to measure the gap and radius, to
check the parabola z²/(2R) against the ring centreline circle and to integrate κ(z) numerically.

**Results.** [ring_bus_coupler.gds](experiments/11_directional_coupler/out/ring_bus_coupler.gds),
[directional_coupler.gds](experiments/11_directional_coupler/out/directional_coupler.gds),
[gds_layout.png](experiments/11_directional_coupler/out/gds_layout.png),
[point_coupler_geometry.png](experiments/11_directional_coupler/out/point_coupler_geometry.png);
measured gap 200.0 nm, centreline radius 6.306 µm, width 0.503 µm (7 polygons); the parabola
deviates from the centreline circle by 0.5 nm inside |z| < 1.14 µm and 14.4 nm at |z| = 2.30 µm
(expected z⁴/(8R³) = 13.9 nm); L_eff = √(2πR/γ) = 2.01 µm (textbook γ) against a numerical
circle integral of 2.00 µm.

**How to observe it.** `cd experiments/11_directional_coupler && ../../.venv/bin/python
gds_part.py`; open the GDS files in KLayout (or the PNGs); numbers in `out/gds_results.json`.
Change `R`, `GAP`, or the `dy`/`dx` of the S-bends in `gds_part.py`.

**Limitations and gotchas.** (i) A PDK must be activated before building components
(`gf.gpdk.PDK.activate()`); without it component functions raise or use an unset layer stack.
(ii) `c.plot()` draws with matplotlib; save it with `plt.savefig` (the `Agg` backend applies).
(iii) The point-coupler check must use the ring *centreline* circle (lower vertices scaled to
R), not the outer edge: an earlier "6.7 nm against the outer edge" agreement was a coincidence of
the radii R and R + w/2. (iv) Import takes a few seconds and logs warnings about missing optional
plugins; harmless. (v) gdsfactory versions move fast and change component signatures; the calls
above are for 9.51.0.

## ffmpeg and ffprobe

**What it is.** The universal command-line video encoder/decoder (ffmpeg) and its stream
inspector (ffprobe). matplotlib's `FFMpegWriter` pipes rendered frames into ffmpeg to encode
H.264 mp4 files; Manim uses it the same way; ffprobe reports duration, frame size and codec.

**Version and location.** ffmpeg 9.0.1 and ffprobe 9.0.1, `/opt/homebrew/bin/ffmpeg`,
`/opt/homebrew/bin/ffprobe` (Homebrew). imageio-ffmpeg also bundles its own ffmpeg binary
(see imageio); the experiments use the Homebrew one by absolute path.

**Where it is used.** All 13 experiments produce at least one mp4 through
`FFMpegWriter(fps=30)` (some with `bitrate=2500`) with libx264 and `yuv420p`. Experiments 01,
02, 05, 06, 07, 08 and 10 set
`matplotlib.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"` in `run.py`, and
12 sets it in `ring_sax.py` and `ring_meep.py`; 03, 04, 09, 11 and 13 do not set it, so their
videos depend on ffmpeg being on `PATH`. 01 to 08, 10 and 13 read the version live with
`ffmpeg -version` into `tools.json` (09 records it as a constant); 01 and 06 read durations back
with ffprobe.

**Results.** The 19 mp4 files listed under [matplotlib](#matplotlib) plus the two Manim renders
of 01 (21 in total), all H.264, 30 fps, 2.4 to 13.3 s each.

**How to observe it.** `/opt/homebrew/bin/ffprobe experiments/02_wavevector_evanescent/out/evanescent_sweep.mp4`
prints duration, resolution and codec; to extract one frame:
`/opt/homebrew/bin/ffmpeg -ss 7.5 -i <file>.mp4 -frames:v 1 frame.png`.

**Limitations and gotchas.** `yuv420p` needs even frame dimensions (see matplotlib). QuickTime
needs `yuv420p`; the default pixel format would not play there. Encoding is a large share of
several runtimes (the 360-frame video in 03 is 40 of 50 s).

## imageio

**What it is.** A Python library for reading and writing images and video frames with a
uniform API; `imageio-ffmpeg` provides the ffmpeg plugin (and a bundled ffmpeg binary) so that
video frames can be read back into numpy arrays.

**Version and location.** imageio 2.37.4 and imageio-ffmpeg 0.6.0 in `.venv`
(`.venv/lib/python3.12/site-packages/imageio`); imageio 2.37.4 in `.meep` as well.

**Where it is used.** 01 only: `imageio.get_reader(mp4, "ffmpeg")` reads six evenly spaced
frames from each Manim mp4 to build the contact sheets. The other experiments build their
contact sheets by re-drawing selected frames with matplotlib (or Pillow in 13) instead of
reading the video back.

**Results.** [travelling_wave_frames.png](experiments/01_travelling_wave/out/travelling_wave_frames.png),
[sign_convention_frames.png](experiments/01_travelling_wave/out/sign_convention_frames.png).

**How to observe it.** The contact-sheet loop at the end of the Manim section of 01's `run.py`
(change the number of `picks`).

**Limitations and gotchas.** The `"ffmpeg"` plugin needs imageio-ffmpeg installed; reading frame
N seeks through the file, so pick a handful of frames rather than iterating over all of them.

## Pillow

**What it is.** The Python Imaging Library fork: open, create, paste and save raster images.

**Version and location.** Pillow 12.3.0, `.venv/lib/python3.12/site-packages/PIL`.

**Where it is used.** 13: `Image.new` and `paste` assemble the two video contact sheets from
PNG frames rendered into memory (`io.BytesIO`). matplotlib uses Pillow internally to write PNGs
everywhere.

**Results.** [notch_lock_frames.png](experiments/13_capstone_ring_lock/out/notch_lock_frames.png),
[four_ring_lock_frames.png](experiments/13_capstone_ring_lock/out/four_ring_lock_frames.png).

**How to observe it.** The `make_frame_fig` / contact-sheet block in 13's `run.py`.

**Limitations and gotchas.** None hit.

## scikit-rf

**What it is.** A Python library for RF and microwave engineering: Touchstone files, network
(S-parameter) objects, cascading, de-embedding, calibration and Smith-chart plotting. It is the
RF engineer's counterpart of SAX.

**Version and location.** scikit-rf 1.13.0, `.venv/lib/python3.12/site-packages/skrf` (import
name `skrf`). Installed by `env/setup.sh`.

**Where it is used.** Not used by any experiment. The S-matrix work (2 x 2 coupler, unitarity,
ring circuit) was done in SAX, which is wavelength-native and matches the gdsfactory ecosystem;
scikit-rf's frequency-domain `Network` would have worked equally for the coupler check.

**Results.** None.

**How to observe it.** `env/check_env.py` imports it and builds a `Frequency` object.

**Limitations and gotchas.** None hit. Kept in the environment because the brief listed it.

## fdtd (Python package)

**What it is.** A small pure-Python/PyTorch-optional 3-D FDTD library (`fdtd` on PyPI, 0.3.5):
a `Grid` with objects, sources, detectors and PML, written for teaching and quick experiments.

**Version and location.** fdtd 0.3.5, `.venv/lib/python3.12/site-packages/fdtd`. Installed by
`env/setup.sh`.

**Where it is used.** Not used by any experiment: all FDTD work uses Meep (which has eigenmode
sources, DFT monitors, Harminv and conductive media). Note that experiment 09 has a local module
named `fdtd.py` (`import fdtd` inside that folder resolves to it, not to the package).

**Results.** None.

**How to observe it.** `env/check_env.py` creates a 10 x 10 `fdtd.Grid`.

**Limitations and gotchas.** The name clash with 09's `fdtd.py` means the package cannot be
imported from inside `experiments/09_tir_evanescent_fdtd/`; nothing needs it there.

## Other installed packages

Installed by `env/setup.sh` and importable, but not used by any experiment: **plotly 7.1.0**
(interactive HTML plots; every figure here is a static matplotlib PNG so that it can be embedded
in a README), **pandas 3.0.5** (tabular data; the two CSV tables in 04 are written with numpy),
**h5py 3.16.0** in both environments (HDF5; Meep can write `.h5` field dumps, which are
git-ignored under `experiments/*/out/*.h5`, but every experiment takes fields with `get_array` /
`get_dft_array` instead), **tqdm 4.70.1** (progress bars). `env/check_env.py` smoke-tests the
first three.

---

## Tool map

| concept (notes section) | experiment | tool | why that tool |
|---|---|---|---|
| E0 cos(ωt - βz) three ways, phasor rotation, sign convention (2, 3) | 01 | Manim | a narrated, continuous animation of a crest, a rotating arrow and a growing trace is what the concept needs; matplotlib videos cannot animate text and geometry as cleanly |
| crest tracking, λ_g measurement, phasor identity (2, 3, 22) | 01 | numpy, scipy.signal | measured from the computed field rather than read off the formula |
| k components, k_x turning imaginary, ⟨S⟩ of a tail (4, 17, 25) | 02 | numpy + matplotlib animation | one formula evaluated with a complex k_x; no solver needed, the point is that nothing else changes |
| the bound electron as a transfer function (7, 8) | 03 | ngspice | the Lorentz ODE *is* a series RLC; a circuit simulator measures χ(ω) instead of assuming it, in Kenny's native language |
| Sellmeier, n_g = n - λ dn/dλ, D from curvature (9 to 11) | 04, 05 | sympy | the derivation is symbolic; lambdified derivatives avoid finite-difference error in D |
| pulse propagation, group delay, eye diagrams (12) | 05 | numpy FFT, scipy.signal | the notes' frequency integral done exactly with the full β(ω) |
| absorption versus evanescence, Poynting's theorem (6, 25, 28) | 06 | Meep (1-D conductive medium, 2-D Bloch interface) | the fields must come from Maxwell's equations, not from the closed form being tested |
| slab eigenvalue equations and the graphical construction (14 to 19) | 07 | numpy/scipy brentq | a transcendental equation with a guaranteed bracket per branch |
| validation of the slab modes | 07 | Meep MPB `get_eigenmode` | an independent numerical eigensolver on the same geometry |
| the real 500 x 220 nm strip, n_eff, n_g, Γ, dλ/dT (22, 23, capstone) | 08 | Tidy3D ModeSolver and femwell | no closed form exists; two solvers with different discretisations (Yee grid versus conforming FEM) agree to 0.06 % |
| TIR of a real beam, tunnelling through a gap (24, 25, 27) | 09 | Meep (GaussianBeamSource) | a finite beam is the non-imposed test of "⟨S_x⟩ averages to zero" and shows what a plane wave cannot (the critical-angle rounding) |
| a mode propagating, a non-modal launch decomposing (20, 21) | 10 | Meep (EigenModeSource, get_eigenmode_coefficients) | time stepping assumes nothing about modes; watching the shape stay fixed is the demonstration |
| two coupled guides, supermodes, κ_c versus gap (26, 27) | 11 | Meep FDTD + MPB, transfer matrix | power sloshing seen in the field; supermode splitting checked three ways |
| 2 x 2 S-matrix, unitarity, amplitudes adding, ring circuit (26, 27) | 11, 12 | SAX (+ JAX) | a circuit-level S-matrix solver is the right abstraction once the coupler is a black box; vectorised over wavelength |
| the ring-bus point coupler geometry, L_eff | 11 | gdsfactory | the actual layout object, exported as GDS and read back as polygons |
| the all-pass notch, FSR, FWHM, Q, thermal slide (28, capstone) | 12 | SAX | closes the coupler and a lossy waveguide into a loop; matches the closed form to 3e-6 |
| resonances and Q of a real 2-D ring | 12 | Meep + Harminv | resonance frequencies and decay rates from a short time record, more precise than an FFT |
| the thermal plant, PI design, margins (capstone) | 13 | python-control | transfer functions, margins and Padé delays are exactly its job; the nonlinear time simulation is plain numpy |
| interactive exploration of any of the above | 01, 03, 04, 07, 12, 13 | Jupyter + ipywidgets + nbformat | sliders on the same functions `run.py` uses, generated from a script so they cannot go stale |
| every figure and video | all | matplotlib + ffmpeg | one style (`common/style.py`), one encoder |

## Reproducing the environment

1. Install Homebrew (https://brew.sh), then run [env/setup.sh](env/setup.sh) from the repo root.
   It installs ffmpeg, ngspice, micromamba and uv with Homebrew; creates `.venv` (Python 3.12)
   with uv and installs the full package list; creates `.meep` from conda-forge with micromamba
   (`MAMBA_ROOT_PREFIX=$PWD/.mamba`).
2. Register the notebook kernel if `.venv/bin/jupyter kernelspec list` does not show
   `photonics-sims`:
   `.venv/bin/python -m ipykernel install --user --name photonics-sims --display-name "Python (photonics-sims)"`.
3. Run `.venv/bin/python env/check_env.py`. It prints every version and path above, smoke-tests
   each library (import plus one trivial call), runs a one-resistor ngspice netlist in batch mode,
   checks ffmpeg/ffprobe/micromamba/uv/brew, checks the Meep environment through a subprocess
   `import meep` and the presence of `libmeep`, `libmpb`, `libharminv`, and exits 1 on any FAIL.
4. Pinned versions that produced the current outputs (2026-09-17): Python 3.12.13 / 3.12.14,
   numpy 2.4.6 / 2.5.3, scipy 1.18.1, matplotlib 3.11.2, sympy 1.14.0, python-control 0.10.2,
   jupyterlab 4.6.3, nbconvert 7.17.1, nbclient 0.11.0, nbformat 5.11.1, ipywidgets 8.1.9,
   ipykernel 7.3.0, manim 0.21.0, femwell 0.1.12, scikit-fem 12.0.2, gmsh 4.15.2, shapely 2.1.2,
   meshio 5.3.5, tidy3d 2.12.0, sax 0.18.2, jax/jaxlib 0.9.2, gdsfactory 9.51.0, kfactory 3.0.4,
   klayout 0.30.12, imageio 2.37.4, imageio-ffmpeg 0.6.0, Pillow 12.3.0, meep 1.34.0, ngspice 47,
   ffmpeg 9.0.1, micromamba 2.9.0, uv 0.11.12, Homebrew 7.0.4. `env/setup.sh` does not pin
   them; to reproduce exactly, add `==<version>` to the `uv pip install` line and
   `pymeep=1.34.0` to the micromamba line.
5. Things that are deliberately *not* required: LaTeX (Manim uses `Text()`, matplotlib uses
   mathtext), a Tidy3D account (only the local mode solver is used), MPI (Meep runs serially;
   09 parallelises over independent runs with `multiprocessing`), a GPU (JAX runs on CPU).
6. Machine: macOS (Darwin 25.5, Apple silicon). Total wall time of `run_all.sh` is about 30
   minutes; the slowest experiments are 12 (about 440 s: Part A 19 s, Part B 408 s, notebook about 10 s), 08 (about 5 min, mode
   solver sweeps) and 09 (about 5 min, 61 Meep runs).
