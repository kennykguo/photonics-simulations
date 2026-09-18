# photonics-simulations

A simulation lab that gives a second, visual and numerical, perspective on Kenny's notes
([docs/NOTES.md](docs/NOTES.md), "from Maxwell to the directional coupler") and ties every
concept to the Lightmatter capstone: keeping silicon microring modulators locked to their
laser wavelengths against thermal drift (10 to 125 C ambient, 50 pm/K, 4 to 8 rings on a bus,
a heater as the only actuator, a tapped photocurrent as the only sensor).

The notes derive the equations. Each of the 13 experiments here simulates one of those
concepts with a real tool (an FDTD solver, two mode solvers, a SPICE simulator, a circuit
S-matrix solver, a control library, a computer-algebra system, an animation engine), produces
figures, videos and a `results.json`, and writes a README that quotes every headline number
next to the analytic expectation from the notes with a percent agreement. Several experiments
also ship an executed Jupyter notebook with sliders. This is not a website: the deliverables
are scripts, solver outputs, rendered media and documentation.

The brief that every experiment was written against is [docs/BRIEF.md](docs/BRIEF.md). The
exhaustive documentation of every tool (what it is, version, where it is used, what it
produced, how to observe it, gotchas) is [TOOLS.md](TOOLS.md).

## Layout

```
README.md            this file
TOOLS.md             one section per tool, plus a concept -> tool map
env/setup.sh         reproduces the two Python environments and the Homebrew tools
env/check_env.py     prints every tool's version and location and smoke-tests each one
run_all.sh           runs every experiment headlessly with the right interpreter
common/params.py     REF: the reference numbers of the capstone (one place, imported everywhere)
common/style.py      matplotlib style, SERIES colours, RdBu_r / Blues conventions
common/units.py      dB, alpha, small-change-rule helpers
docs/NOTES.md        Kenny's notes, sections 1 to 28
docs/BRIEF.md        the authoring brief and README contract
experiments/NN_name/ run.py (headless entry point), README.md, optional explore.ipynb,
                     helper modules, out/ (png, mp4, json, txt, csv, cir, gds, logs),
                     .uses_meep marker when run.py must use the Meep interpreter
.venv/               Python 3.12.13 (uv): numpy, scipy, matplotlib, sympy, control, femwell,
                     tidy3d, sax/jax, gdsfactory, manim, jupyterlab, ...   (not in git)
.meep/               Python 3.12.14 (micromamba, conda-forge): pymeep 1.34.0 with MPB and
                     Harminv, numpy, scipy, matplotlib, h5py, imageio          (not in git)
.mamba/              micromamba root prefix for .meep                          (not in git)
```

## Setup

Everything runs on macOS (Apple silicon) with two interpreters and three Homebrew programs.
[env/setup.sh](env/setup.sh) reproduces it from scratch:

```bash
cd /Users/kennyg/photonics-simulations
bash env/setup.sh
```

which does, in order:

1. `brew install ffmpeg ngspice micromamba uv` (video encoder, circuit simulator, conda-forge
   installer, Python environment manager).
2. `uv venv --python 3.12 .venv` and `uv pip install ...` of the scientific and photonics stack
   into `.venv` (numpy, scipy, matplotlib, jupyterlab, ipywidgets, sympy, scikit-rf, control,
   plotly, pandas, pillow, imageio, imageio-ffmpeg, manim, femwell, tidy3d, sax, gdsfactory,
   fdtd, jax, meshio, gmsh, shapely, scikit-fem, h5py, tqdm, nbformat, nbclient).
3. `micromamba create -p .meep -c conda-forge python=3.12 pymeep numpy scipy matplotlib h5py
   imageio imageio-ffmpeg` with `MAMBA_ROOT_PREFIX=$PWD/.mamba`, because pymeep has no pip
   wheel.

The Jupyter kernel `photonics-sims` (which points at `.venv/bin/python`) is registered at
`~/Library/Jupyter/kernels/photonics-sims/kernel.json`; if it is missing after a fresh setup,
register it with `.venv/bin/python -m ipykernel install --user --name photonics-sims
--display-name "Python (photonics-sims)"`.

Then verify the whole environment (about 10 s; prints one OK/FAIL line per tool and exits 1 on
any failure):

```bash
.venv/bin/python env/check_env.py
```

There is no LaTeX on this machine. Manim scenes use `Text()` with Unicode and matplotlib
figures use mathtext, and nothing here needs LaTeX.

Interpreters, always by absolute or repo-relative path (never `python` from the shell):

| what | interpreter |
|---|---|
| every experiment without a `.uses_meep` marker | `.venv/bin/python` |
| experiments 06, 07, 09, 10, 11, 12 (`.uses_meep` present) | `.meep/bin/python` |
| notebooks | `.venv/bin/jupyter lab` (kernel `photonics-sims`) |
| Manim by hand | `.venv/bin/manim` |
| ngspice, ffmpeg, ffprobe | `/opt/homebrew/bin/ngspice`, `/opt/homebrew/bin/ffmpeg`, `/opt/homebrew/bin/ffprobe` |

## Running

Everything at once (about 30 minutes on this laptop; each experiment reports its own runtime in
`out/results.json`):

```bash
./run_all.sh
```

It loops over `experiments/[0-9][0-9]_*/`, picks `.meep/bin/python` when the folder contains
`.uses_meep` and `.venv/bin/python` otherwise, runs `run.py` from inside the experiment folder,
and prints `!!! <dir> failed` for any non-zero exit instead of stopping.

One experiment (always from inside its folder; `run.py` regenerates all of `out/` and, where
there is one, rebuilds and re-executes `explore.ipynb`):

```bash
cd experiments/04_sellmeier_dispersion && ../../.venv/bin/python run.py    # .venv experiment
cd experiments/09_tir_evanescent_fdtd && ../../.meep/bin/python run.py     # .uses_meep experiment
```

Notebooks, interactively (Run All, then drag the sliders):

```bash
cd experiments/13_capstone_ring_lock && ../../.venv/bin/jupyter lab explore.ipynb
```

or headlessly, which is what `run.py` does at the end of experiments 01, 03, 04, 07, 12, 13:

```bash
../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace explore.ipynb
```

Runtimes of the runs that produced the current `out/` folders (from each `results.json`):
01 66 s, 02 50 s, 03 49 s, 04 82 s, 05 51 s, 06 81 s, 07 19 s, 08 279 s, 09 270 s, 10 111 s,
11 105 s, 12 about 440 s (Part A 19 s, Part B 408 s, notebook about 10 s), 13 128 s.
The notebook step of 01 and 04 is load-sensitive (an intermittent ipywidgets/nbclient stall,
bounded by a 60 s per-cell timeout and documented in those READMEs); `SKIP_NOTEBOOK=1` skips it
in 04.

## Index

Section numbers refer to [docs/NOTES.md](docs/NOTES.md). Output links are relative to the repo
root; every experiment folder has more figures than listed here, and its own README embeds all
of them.

| # | Title | Notes sections | Tools | Main outputs | Capstone connection |
|---|---|---|---|---|---|
| 01 | [The travelling wave, three ways, and why phasors work](experiments/01_travelling_wave/README.md) | 2, 3 (also 4, 15, 28) | Manim, numpy, scipy, matplotlib, ffmpeg/imageio, Jupyter + ipywidgets | [three_ways.png](experiments/01_travelling_wave/out/three_ways.png), [travelling_wave.mp4](experiments/01_travelling_wave/out/travelling_wave.mp4), [phasor.png](experiments/01_travelling_wave/out/phasor.png), [capstone_round_trip.png](experiments/01_travelling_wave/out/capstone_round_trip.png), [explore.ipynb](experiments/01_travelling_wave/explore.ipynb) | One lap of the ring is e^{-jβL} in this phasor convention; the round-trip phasor turns 1.74 deg/K; the resonance spacing is set by n_g (10.3 nm), not n_eff |
| 02 | [Wavevector components and the evanescent field](experiments/02_wavevector_evanescent/README.md) | 4, 17 (also 3, 18, 24, 25) | numpy, scipy, matplotlib, ffmpeg | [evanescent_sweep.mp4](experiments/02_wavevector_evanescent/out/evanescent_sweep.mp4), [transverse_phasor_and_power.png](experiments/02_wavevector_evanescent/out/transverse_phasor_and_power.png), [capstone_gap_sensitivity.png](experiments/02_wavevector_evanescent/out/capstone_gap_sensitivity.png) | The 102 nm tail (n_eff 2.5 in silica) is what couples ring to bus: κ ∝ e^{-γ gap}, a 10 nm gap error is 10 % of κ and 20 % of κ² |
| 03 | [The driven electron is a series RLC](experiments/03_driven_electron_rlc/README.md) | 7, 8 (also 6, 9, 10) | ngspice, scipy/numpy, matplotlib, Jupyter + ipywidgets | [ac_sweep.png](experiments/03_driven_electron_rlc/out/ac_sweep.png), [transients.png](experiments/03_driven_electron_rlc/out/transients.png), [frequency_sweep.mp4](experiments/03_driven_electron_rlc/out/frequency_sweep.mp4), [bound_vs_free.png](experiments/03_driven_electron_rlc/out/bound_vs_free.png) | The modulator's plasma-dispersion effect is the same χ with the spring removed; the thermo-optic effect is a shift of the resonance; the full 65 pm data swing equals 1.3 K of drift |
| 04 | [Sellmeier dispersion: group index, zero dispersion and the ring FSR](experiments/04_sellmeier_dispersion/README.md) | 9, 10, 11 (also 8, 12) | sympy, numpy/scipy, matplotlib, Jupyter + ipywidgets | [n_and_ng.png](experiments/04_sellmeier_dispersion/out/n_and_ng.png), [dispersion_D.png](experiments/04_sellmeier_dispersion/out/dispersion_D.png), [tangent_sliding.mp4](experiments/04_sellmeier_dispersion/out/tangent_sliding.mp4), [capstone_fsr.png](experiments/04_sellmeier_dispersion/out/capstone_fsr.png) | Why the O-band (silica ZDW 1273 nm); FSR = c/(n_g L) = 1.80 THz with n_g = 4.2; 1 nm = 174.7 GHz; dλ_r/dT carries n_g in its denominator |
| 05 | [Pulse dispersion: how a pulse is delayed and distorted](experiments/05_pulse_dispersion/README.md) | 12 (built on 9 to 11) | numpy FFT, sympy, scipy.signal, matplotlib, ffmpeg | [gaussian_broadening.png](experiments/05_pulse_dispersion/out/gaussian_broadening.png), [eye_diagrams.png](experiments/05_pulse_dispersion/out/eye_diagrams.png), [nrz_eye_closing.mp4](experiments/05_pulse_dispersion/out/nrz_eye_closing.mp4), [eye_vs_length.png](experiments/05_pulse_dispersion/out/eye_vs_length.png) | 2 km of fibre at 53 Gbaud: 3.3 ps of spread at 1310 nm, 28 ps at 1550 nm; the link is a wavelength-locking problem, not a dispersion problem |
| 06 | [Absorption versus evanescence](experiments/06_absorption_vs_evanescence/README.md) | 6, 24, 25, 28 | Meep (1-D lossy medium, 2-D Bloch interface), numpy, matplotlib, ffmpeg | [fig1_absorption_vs_evanescence.png](experiments/06_absorption_vs_evanescence/out/fig1_absorption_vs_evanescence.png), [fig2_poynting.png](experiments/06_absorption_vs_evanescence/out/fig2_poynting.png), [absorption_vs_evanescence.mp4](experiments/06_absorption_vs_evanescence/out/absorption_vs_evanescence.mp4) | 125 dB/cm gives a = 0.945 per lap (FDTD 0.9444); absorbed light heats the ring (self-heating, up to 1.85 mW), confined light couples and is lossless |
| 07 | [Slab TE modes: the graphical solution, checked against Meep](experiments/07_slab_modes_graphical/README.md) | 14 to 19, 21 to 23 | numpy/scipy (brentq), Meep MPB `get_eigenmode`, matplotlib, Jupyter + ipywidgets | [07_graphical_construction.png](experiments/07_slab_modes_graphical/out/07_graphical_construction.png), [07_mode_profiles.png](experiments/07_slab_modes_graphical/out/07_mode_profiles.png), [07_meep_validation.png](experiments/07_slab_modes_graphical/out/07_meep_validation.png), [07_plane_wave_decomposition.mp4](experiments/07_slab_modes_graphical/out/07_plane_wave_decomposition.mp4) | 1/γ = 80 nm; the thermal drift depends only on the silicon energy fraction Γ_E (n_g cancels); the slab gives 68 pm/K, so the strip needs a 2-D solver |
| 08 | [The real SOI strip: two full-vector mode solvers and the thermo-optic number](experiments/08_soi_strip_mode_solvers/README.md) | 22, 23 (also 16, 18, 25) | Tidy3D ModeSolver, femwell (+ scikit-fem, gmsh, shapely, meshio), numpy/scipy, matplotlib, ffmpeg | [08_mode_fields.png](experiments/08_soi_strip_mode_solvers/out/08_mode_fields.png), [08_width_sweep.png](experiments/08_soi_strip_mode_solvers/out/08_width_sweep.png), [08_thermo_optic.png](experiments/08_soi_strip_mode_solvers/out/08_thermo_optic.png), [08_mode_vs_width.mp4](experiments/08_soi_strip_mode_solvers/out/08_mode_vs_width.mp4) | n_eff = 2.71 (textbook 2.5), n_g = 4.16 (REF 4.2), Γ_P = 0.85; dλ_r/dT = 63.6 pm/K from the solvers versus the 50 pm/K reference; 1 nm of width = 346 pm |
| 09 | [Total internal reflection and the evanescent field, watched in FDTD](experiments/09_tir_evanescent_fdtd/README.md) | 4, 17, 24, 25, 27 | Meep (GaussianBeamSource, DFT and flux monitors, process pool), numpy/scipy, matplotlib, ffmpeg | [poynting_three_angles.png](experiments/09_tir_evanescent_fdtd/out/poynting_three_angles.png), [evanescent_decay.png](experiments/09_tir_evanescent_fdtd/out/evanescent_decay.png), [ftir_vs_gap.png](experiments/09_tir_evanescent_fdtd/out/ftir_vs_gap.png), [tir_beams.mp4](experiments/09_tir_evanescent_fdtd/out/tir_beams.mp4), [ftir.mp4](experiments/09_tir_evanescent_fdtd/out/ftir.mp4) | The tunnelled power through a gap falls as e^{-2γ gap}: -17 % per +10 nm at the strip-mode angle; the heater must sit several decay lengths away |
| 10 | [Slab mode in FDTD, and what happens to a field that is not a mode](experiments/10_slab_mode_fdtd/README.md) | 20, 21 (also 14 to 19, 22, 23, 25) | Meep (EigenModeSource, add_dft_fields, get_eigenmode_coefficients), numpy/scipy, matplotlib, ffmpeg | [10_te0_travelling_mode.mp4](experiments/10_slab_mode_fdtd/out/10_te0_travelling_mode.mp4), [10_te0_profile.png](experiments/10_slab_mode_fdtd/out/10_te0_profile.png), [10_offcentre_decomposition.png](experiments/10_slab_mode_fdtd/out/10_offcentre_decomposition.png), [10_offcentre_pulse.mp4](experiments/10_slab_mode_fdtd/out/10_offcentre_pulse.mp4) | The mode shape F(x) is what Γ (the thermo-optic weight) and the coupling depend on; the coupler injects TE0 plus radiation that is gone within microns |
| 11 | [The evanescent directional coupler: two guides trading power](experiments/11_directional_coupler/README.md) | 26, 27 (also 16 to 19, 21, 25, 28) | Meep FDTD + MPB supermodes, transfer-matrix solver, SAX + JAX, gdsfactory, matplotlib, ffmpeg | [power_vs_z.png](experiments/11_directional_coupler/out/power_vs_z.png), [kappa_vs_gap.png](experiments/11_directional_coupler/out/kappa_vs_gap.png), [pulse_hopping.mp4](experiments/11_directional_coupler/out/pulse_hopping.mp4), [critical_coupling_lottery.png](experiments/11_directional_coupler/out/critical_coupling_lottery.png), [ring_bus_coupler.gds](experiments/11_directional_coupler/out/ring_bus_coupler.gds) | κ² = 0.107 for the reference ring; ±20 % per 10 nm of gap; the point coupler acts like a 2 µm straight coupler; critical coupling t = a is a fabrication lottery (T_min = 0.016 is a 13 nm gap error) |
| 12 | [From one round trip to the ring](experiments/12_round_trip_to_ring/README.md) | 28 (also 3, 6, 16, 22, 24, 26, 27) | SAX + JAX, Meep FDTD + Harminv + MPB, numpy/scipy, matplotlib, Jupyter + ipywidgets | [A2_spectrum.png](experiments/12_round_trip_to_ring/out/A2_spectrum.png), [A4_thermal.png](experiments/12_round_trip_to_ring/out/A4_thermal.png), [A5_notch_sliding.mp4](experiments/12_round_trip_to_ring/out/A5_notch_sliding.mp4), [B4_spectrum.png](experiments/12_round_trip_to_ring/out/B4_spectrum.png), [B5_buildup.mp4](experiments/12_round_trip_to_ring/out/B5_buildup.mp4) | This is the plant: the SAX ring gives FSR 10.318 nm / 1.803 THz, FWHM 374.0 pm, Q 3503 with a = 0.9446 (REF 10.3 nm / 1.8 THz / 374 pm / 3500 / 0.945, t = a); one FSR is 206.4 K, 0.1 K is 5.0 pm; the laser sits at δ = +108 pm where the slope is -0.174 per K; the 2-D FDTD ring has FSR 11.73 nm and a notch with T_min 0.656, FWHM 312.5 pm, Q 4201 (flux) / 4391 (Harminv) / 4509 (CW build-up) |
| 13 | [Capstone ring lock: heater, photocurrent and a PI loop](experiments/13_capstone_ring_lock/README.md) | capstone objective (also 22, 27, 28) | python-control, numpy/scipy, matplotlib, Pillow, Jupyter + ipywidgets | [block_diagram.png](experiments/13_capstone_ring_lock/out/block_diagram.png), [bode_loop.png](experiments/13_capstone_ring_lock/out/bode_loop.png), [scenario_a_step.png](experiments/13_capstone_ring_lock/out/scenario_a_step.png), [notch_lock.mp4](experiments/13_capstone_ring_lock/out/notch_lock.mp4), [four_ring_lock.mp4](experiments/13_capstone_ring_lock/out/four_ring_lock.mp4) | The deliverable's inner loop: PI with 8 kHz crossover and 60 deg phase margin, 0.6 pm error on an 800 K/s ramp, 128 pm transient on a 5 K step, survives 6 K instantaneous steps, 2.5 pm of neighbour error while a ring turns on; a 12-bit heater DAC is needed for 0.1 K |

## Reading order

The experiments are numbered in the order of the notes, and that is the order to read them
in. If time is short, three shorter paths:

1. **The capstone in five experiments (control-first).** 13 (the loop), then 12 (the plant it
   controls), 08 (where 50 pm/K and n_g come from and why the solvers say 64 pm/K), 11 (why
   κ² and the notch depth are a fabrication lottery), 06 (why absorbed light heats the ring).
2. **Guided light from scratch (physics-first).** 01 (travelling wave and phasors), 02 (k
   components and the evanescent tail), 07 (slab modes by hand), 10 (the same mode in FDTD),
   09 (TIR with a real beam), 11 (two guides coupling), 12 (closing the loop into a ring).
3. **Materials and why 1310 nm.** 03 (the electron as an RLC, ngspice), 04 (Sellmeier, group
   index, zero dispersion), 05 (pulses and eyes over 2 km).

Whichever path, keep two facts in mind while reading: (a) the textbook's n_eff = 2.5 is a
placeholder and the solvers give about 2.7 for the strip, and every experiment states which of
its numbers depend on n_eff (mode order, phase, λ_g, decay length) and which on n_g (FSR, every
shift-per-index formula); (b) every number in a README table was computed by `run.py` and is in
that experiment's `out/results.json` with the expectation and the agreement.

## Reference numbers

All experiments import `REF` from [common/params.py](common/params.py), so a change there
propagates everywhere. Source tags as written in that file: [LM] Lightmatter OFC 2025
microring Tx/Rx paper; [P] Padmaraju and Bergman 2013; [T] Flexcompute thermally tuned ring
example; [N] meeting notes; [C] Choi and Stojanovic. Fields without a tag in the file are
rounded textbook values or are derived from tagged ones as noted.

| field | value | source / derivation |
|---|---|---|
| `lambda_nm` | 1310 | [LM] O-band carrier |
| `n_si`, `n_sio2` | 3.50, 1.45 | rounded material indices at 1310 nm (Sellmeier gives 3.5005 and 1.4468, experiment 04) |
| `wg_width_um`, `wg_height_um` | 0.50, 0.22 | [T] |
| `neff` | 2.5 | textbook reference; solvers give 2.71 (experiment 08), slab 2.99 (07) |
| `ng` | 4.2 | group index that sets the FSR; solvers give 4.16 (08) |
| `confinement` | 0.85 | power fraction in silicon; solvers give 0.852 to 0.853 (08) |
| `radius_um`, `round_trip_um` | 6.3, 39.6 | ring radius; L = 2πR |
| `fsr_thz` | 1.8 | [LM] |
| `fsr_nm` | 10.3 | from `fsr_thz` through the small-change rule (174.7 GHz/nm) |
| `q_loaded` | 3500 | [LM] |
| `fwhm_pm` | 374 | λ/Q |
| `a_round_trip` | 0.945 | e^{-αL/2} with 125 dB/cm and L = 39.6 µm (experiments 06, 12: 0.9446) |
| `t_coupler` | 0.945 | critical coupling t = a |
| `kappa2` | 0.107 | 1 - t² |
| `loss_db_cm_doped`, `loss_db_cm_passive` | 125, 3 | equivalent propagation loss of the doped ring; undoped strip |
| `dlambda_dT_pm_per_K` | 50 | consistent with [LM] `heater_nm_per_mw` and `r_th_K_per_mw` (0.44 nm/mW / 8.8 K/mW); solvers give 63.6 pm/K (08) |
| `dn_si_dT`, `dn_sio2_dT` | 1.86e-4, 1.0e-5 /K | thermo-optic coefficients |
| `heater_nm_per_mw` | 0.44 | [LM] |
| `r_th_K_per_mw` | 8.8 | `heater_nm_per_mw` / `dlambda_dT_pm_per_K` |
| `tau_th_us`, `tau_slow_us`, `slow_share` | 10, 300, 0.25 | thermal time constants; the slow tail is an assumption (experiment 13 says so) |
| `crosstalk_nearest`, `crosstalk_next` | 0.10, 0.03 | thermal crosstalk between rings; assumed |
| `ring_pitch_um` | 15 | [N] |
| `ambient_min_c`, `ambient_max_c` | 10, 125 | [N] |
| `channel_spacing_ghz`, `n_channels` | 200, 8 | link grid |
| `mod_eff_pm_per_v`, `swing_vpp` | 50, 1.3 | [LM] |
| `p_in_dbm`, `responsivity_a_per_w` | 4, 0.9 | [LM] |
| `baud` | 53.125e9 | link symbol rate |
| `t_min` | 0.016 | on-resonance through transmission (18 dB extinction) |
| `delta_opt_pm` | 108 | maximum-OMA bias, FWHM/(2√3) |

Constants in the same file: c, q_e, m_e, ε0, μ0, ħ. Helpers: `k0_per_um(lambda_nm)`,
`omega_rad_s`, `freq_thz`; `common.units` has dB and small-change-rule conversions.

## Where the solvers disagreed with the textbook

Every experiment reports its agreement honestly; these are the places where the disagreement is
real (not a grid artefact) and what it means for the capstone.

- **n_eff of the strip: 2.5 (textbook) versus 2.71 (Tidy3D 2.7134, femwell 2.7119, experiment
  08), and 2.99 for the 220 nm slab (07, 10).** n_eff only enters the mode order
  m = n_eff L/λ (76 versus 82), the phase per lap, λ_g (524 versus 483 nm) and the evanescent
  decay length (102 versus 92 nm). The FSR, dλ_r/dT, the heater efficiency and the FWHM are
  group-index quantities and do not move. Experiments 01, 02, 04, 07 and 08 each show this
  split in a figure.
- **dλ_r/dT: 50 pm/K (reference, consistent with the measured 0.44 nm/mW and 8.8 K/mW) versus
  63.6 pm/K from both mode solvers (65.8 pm/K with thermal expansion), experiment 08; the slab
  gives 68 pm/K (07).** The textbook shortcut weights dn_Si/dT by the power fraction Γ_P = 0.85,
  but perturbation theory weights it by S_Si = Γ_E n_g,wg/n_Si = 1.08. Experiment 07 shows that
  n_g cancels and only the energy fraction Γ_E remains; 10 shows the Γ-weighted heuristic with
  the FDTD Γ (50.2 pm/K) alongside the perturbation value (58 pm/K for the slab). Consequence:
  the loop gain of experiment 13 should be checked against a ±30 % plant-gain uncertainty, and
  either R_th or the heater efficiency is off if 64 pm/K is right.
- **The 1550 nm eye after 2 km is degraded, not closed (experiment 05).** The brief's rule of
  thumb (27 ps spread against an 18.8 ps bit) is reproduced (28.0 ps), but the chirp-free NRZ eye
  is still 71 % open because most of the modulation energy sits well inside the ±40 GHz band; a
  chirped transmitter (α = 3) does close it. The conclusion (1310 nm for 2 km) is unchanged.
- **Bulk silica D(1550) = 21.9 ps/(nm km) versus the fibre's 17 (experiment 04, 05).** Expected:
  the fibre figure includes waveguide dispersion, which the bulk Sellmeier model does not.
- **The 2-D FDTD ring of experiment 12 is not the reference ring:** n_g = 3.594 instead of 4.2
  (FSR 11.73 nm instead of 10.3), κ² = 0.022 at a 100 nm gap instead of 0.107 at 200 nm (the 2-D
  tail is 69 nm, not 102 nm), and the ring loses 0.087 dB/lap more than the straight-guide
  calibration: the central notch at 1313.09 nm (T_min 0.656, FWHM 312.5 pm, Q 4201 from the
  flux spectrum, 4391 from Harminv, 4509 from the CW build-up with τ = 6.29 ps) inverts to
  t = 0.992 and a = 0.929 versus the calibrated 0.945. Each is traced to the 2-D geometry; the
  Part A SAX ring, which is what the capstone uses, matches the reference to 0.2 %.
- **Silica's UV resonance: the brief's ω0 = 1.9e16 rad/s versus the strength-weighted 2.11e16
  rad/s of Malitson's two UV terms (experiment 03).** Either is "far below resonance"; n(1310)
  is only reproduced when all three Sellmeier terms are kept.
- **Small, explained grid effects, not disagreements:** FDTD n_eff 0.5 % high at 50 px/µm and
  converging (10); the Meep eigenmode solver 0.9 % low on a 40 px/µm grid while the supermode
  splitting is within 1.3 % (11); T at the critical angle +2.4 % converging as
  resolution^-1.4 (09); the SAX ring FWHM 368 pm on a 1 pm grid versus 374 pm (11).

## Notes on the current outputs

- Experiment 12's `out/` was regenerated end to end by its `run.py` (437.6 s: Part A 18.6 s,
  Part B 408.3 s, notebook the rest), so its merged `out/results.json` and `out/tools.json` are
  current and every number quoted for 12 here, in [TOOLS.md](TOOLS.md) and in its README comes
  from that run.
- `experiments/*/out/*.h5`, `experiments/*/media/` and `__pycache__/` are ignored by git; the
  Manim renders are copied from `media/` into `out/` by `run.py`.
