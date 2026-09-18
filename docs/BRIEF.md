# Authoring brief: photonics-simulations

You are writing ONE experiment in a suite whose purpose is to let Kenny (EE undergraduate,
no optics background, strong on circuits/signals/control) understand the physics of guided
light by *simulating it with real tools*, and to connect each concept to his capstone with
Lightmatter: keeping silicon microring modulators locked to their laser wavelengths against
thermal drift (10 to 125 C ambient, 50 pm/K, 4 to 8 rings on a bus, heater-only actuator,
photocurrent sensor). His notes (docs/NOTES.md) already derive the equations; the suite must
give a *second, visual and numerical* perspective and let him experiment.

This is NOT a website. Deliverables are Python scripts, notebooks, rendered videos/figures,
solver outputs, and documentation that describes every tool used.

## Repository layout (fixed)

```
common/            shared params (REF), plotting style, unit helpers  -> import with sys.path insert of repo root
experiments/NN_name/
    run.py         headless entry point; regenerates everything in out/; must run from its own directory
    README.md      required sections below
    explore.ipynb  (only for experiments that list it) interactive notebook, executed and saved with outputs
    out/           figures (png), videos (mp4), data (txt/csv/json), gds, netlists, logs
    .uses_meep     empty marker file if run.py must be executed with the Meep interpreter
docs/NOTES.md      Kenny's notes (the concepts to cover)
```

Interpreters (absolute paths; do not create new environments):
- `/Users/kennyg/photonics-simulations/.venv/bin/python` : numpy, scipy, matplotlib, sympy, control, scikit-rf, plotly,
  ipywidgets, jupyterlab, nbformat/nbclient, manim, femwell (+skfem, gmsh, shapely, meshio), tidy3d, sax (+jax), gdsfactory, fdtd, h5py, imageio(-ffmpeg)
- `/Users/kennyg/photonics-simulations/.meep/bin/python` : meep (pymeep, includes MPB eigenmode solver), numpy, scipy, matplotlib, h5py, imageio(-ffmpeg). Nothing else. (If you truly need another package there: `MAMBA_ROOT_PREFIX=/Users/kennyg/photonics-simulations/.mamba micromamba install -y -p /Users/kennyg/photonics-simulations/.meep -c conda-forge <pkg>`)
- `/opt/homebrew/bin/ngspice` (v47, batch mode `ngspice -b file.cir`), `/opt/homebrew/bin/ffmpeg`
- There is NO LaTeX on this machine. Manim `MathTex`/`Tex` will fail. Use `Text`/`MarkupText` with Unicode (ω, β, λ, ₀, ², −) or render equations with matplotlib mathtext to PNG and use `ImageMobject`.
- `jupyter` is available as `.venv/bin/jupyter`; execute notebooks with `.venv/bin/jupyter nbconvert --to notebook --execute --inplace explore.ipynb` (kernel name `photonics-sims` is installed) and keep the outputs saved in the file.

Run everything from the experiment directory: `cd experiments/NN_name && ../../.venv/bin/python run.py` (or `../../.meep/bin/python run.py`).
At the top of run.py:
```python
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import REF, use_style, SERIES, PALETTE
OUT = pathlib.Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
use_style()
```
`REF` fields: lambda_nm 1310, n_si 3.50, n_sio2 1.45, wg_width_um 0.50, wg_height_um 0.22, neff 2.5 (textbook value; solvers give ~2.7, see 08), ng 4.2, radius_um 6.3, round_trip_um 39.6, fsr_thz 1.8, fsr_nm 10.3, q_loaded 3500, fwhm_pm 374, a_round_trip 0.945, t_coupler 0.945, kappa2 0.107, loss_db_cm_doped 125, loss_db_cm_passive 3, dlambda_dT_pm_per_K 50, dn_si_dT 1.86e-4, dn_sio2_dT 1e-5, heater_nm_per_mw 0.44, r_th_K_per_mw 8.8, tau_th_us 10, tau_slow_us 300, slow_share 0.25, crosstalk_nearest 0.10, crosstalk_next 0.03, ring_pitch_um 15, ambient 10..125 C, channel_spacing_ghz 200, n_channels 8, mod_eff_pm_per_v 50, swing_vpp 1.3, p_in_dbm 4, responsivity 0.9, baud 53.125e9, t_min 0.016, delta_opt_pm 108. Helpers: `common.params.k0_per_um(lambda_nm)`, `omega_rad_s`, `freq_thz`; `common.units` has dB and small-change-rule conversions.

## Conventions (must match the notes)
- Engineering phasor convention e^{jωt}; a +z wave is e^{-jβz}; complex index n = n' − j n'' with n'' > 0 for loss.
- δ = λ_L − λ_r (laser minus resonance), positive = laser on the red side.
- Units: state them on every axis and in every printed number. Wavelength in nm or µm, distances on chip in µm, time in fs/ps/µs as natural. SI internally.
- Numbers: every headline number in the README must be produced by the code (printed to out/results.json or out/results.txt) and compared with the analytic expectation from the notes where one exists. State the agreement (percent).

## Figure and video rules
- Call `use_style()`; use `SERIES` colours in order; legend whenever ≥ 2 series; axis labels with units; titles that state what to see (a sentence, not a noun phrase).
- Signed fields: `cmap="RdBu_r"` centred on zero (`vmin=-m, vmax=+m`). Magnitudes: `cmap="Blues"`.
- Videos: `matplotlib.animation.FuncAnimation` + `FFMpegWriter(fps=30)` to `out/*.mp4`, ≤ 20 s each, plus a PNG contact sheet of 4 to 6 frames (`out/*_frames.png`) so the README can show stills. Manim: `manim -qm --disable_caching -o name.mp4 scene.py SceneName` writes under `media/videos/...`; copy the final mp4 into out/.
- Keep total run time of run.py under ~10 minutes on a laptop; choose Meep resolutions accordingly (2-D, resolution 30–50 px/µm, PML 1 µm).

## README.md contract (every experiment; keep headings exactly)
```
# NN. Title
**Concept** (which sections of docs/NOTES.md; one paragraph in plain words on what the simulation lets you see that the equations alone do not)
**Tools used** (one subsection per tool, each with four labelled lines:
   *What it is:* one or two sentences on the tool in general and what it is normally used for.
   *What I used it for here:* specifically.
   *Result:* the numbers/figures it produced, with the comparison to the analytic expectation.
   *How to observe it:* exact command(s) and output file paths; what to look at in the figure/video; how to change a parameter and re-run.)
**What the simulation does** (steps, geometry, parameters, with the equations from the notes it implements)
**Results** (embedded images `![](out/x.png)`, links to mp4s, a table of numbers with expectation and agreement)
**Experiments to try** (3 to 5 concrete parameter changes and what should happen, so Kenny can use it as a lab)
**Capstone connection** (how this concept enters the Lightmatter ring-locking problem, with the reference numbers)
**Checks** (what was validated and how; known limitations of the model or solver)
**Files** (list of everything in the folder)
```
Also write `out/tools.json`: a list of `{"tool": name, "version": "...", "what_it_is": "...", "used_for": "...", "result": "...", "how_to_observe": "..."}` (same content as the README's tool subsections) and `out/results.json` with the headline numbers. These feed the top-level TOOLS.md.

## Known-good snippets (tested on this machine 2026-09-17)
Tidy3D local mode solver (runs locally, no account needed; warns about subpixel averaging, that is fine):
```python
import tidy3d as td; from tidy3d.plugins.mode import ModeSolver
lam=1.31; si=td.Medium(permittivity=3.50**2); sio2=td.Medium(permittivity=1.45**2)
wg=td.Structure(geometry=td.Box(center=(0,0,0), size=(td.inf,0.5,0.22)), medium=si)
sim=td.Simulation(size=(2,3,3), grid_spec=td.GridSpec.auto(min_steps_per_wvl=20, wavelength=lam), structures=[wg], medium=sio2, run_time=1e-12, boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()))
ms=ModeSolver(simulation=sim, plane=td.Box(center=(0,0,0), size=(0,3,3)), mode_spec=td.ModeSpec(num_modes=2, group_index_step=True), freqs=[td.C_0/lam])
data=ms.solve()   # data.n_eff, data.n_group, data.Ex/Ey/Ez field arrays; ~10 s
```
femwell FEM mode solver:
```python
from collections import OrderedDict; import shapely
from skfem import Basis, ElementTriP0; from skfem.io.meshio import from_meshio
from femwell.mesh import mesh_from_OrderedDict; from femwell.maxwell.waveguide import compute_modes
polys=OrderedDict(core=shapely.box(-0.25,-0.11,0.25,0.11), clad=shapely.box(-1.5,-1.5,1.5,1.5))
mesh=from_meshio(mesh_from_OrderedDict(polys, {"core":{"resolution":0.02,"distance":0.4}}, default_resolution_max=0.25))
basis0=Basis(mesh, ElementTriP0()); eps=basis0.zeros()
for sub,n in {"core":3.50,"clad":1.45}.items(): eps[basis0.get_dofs(elements=sub)]=n**2
modes=compute_modes(basis0, eps, wavelength=1.31, num_modes=2, order=2)   # m.n_eff, m.te_fraction, m.plot(m.E.real) ; ~2 s
```
Meep 2-D slab with eigenmode source (Meep 2-D: TE-like slab mode uses Ez out of plane):
```python
import meep as mp
cell=mp.Vector3(12,6,0); si=mp.Medium(index=3.5); sio2=mp.Medium(index=1.45)
geom=[mp.Block(size=mp.Vector3(mp.inf,0.22,mp.inf), material=si)]
src=[mp.EigenModeSource(mp.ContinuousSource(wavelength=1.31), center=mp.Vector3(-4,0), size=mp.Vector3(0,4), eig_band=1, eig_parity=mp.EVEN_Y+mp.ODD_Z)]
sim=mp.Simulation(cell_size=cell, geometry=geom, sources=src, default_material=sio2, resolution=40, boundary_layers=[mp.PML(1.0)])
sim.run(until=40); ez=sim.get_array(center=mp.Vector3(), size=mp.Vector3(10,4), component=mp.Ez)   # <1 s
```
Meep units: lengths in µm, frequency = 1/λ[µm] (1310 nm -> 0.7634), time in units of µm/c. Use `mp.Harminv` for resonances, `sim.add_flux` for power, `mp.GaussianSource` for spectra, `sim.get_eigenmode` / `get_eigenmode_coefficients` for modal decomposition.
SAX circuit (coupler + waveguide):
```python
import sax, jax.numpy as jnp
def coupler(coupling=0.107):
    k=coupling**0.5; t=(1-coupling)**0.5
    return sax.reciprocal({("in0","out0"):t, ("in0","out1"):-1j*k, ("in1","out0"):-1j*k, ("in1","out1"):t})
def waveguide(wl=1.31, length=39.6, neff=2.5, ng=4.2, wl0=1.31, loss_db_cm=125.0):
    n=neff-(wl-wl0)*(ng-neff)/wl0; amp=10**(-loss_db_cm*length*1e-4/20)
    return sax.reciprocal({("in0","out0"): amp*jnp.exp(-2j*jnp.pi*n*length/wl)})
ring,_=sax.circuit(netlist={"instances":{"c":"coupler","r":"waveguide"},"connections":{"c,out1":"r,in0","r,out0":"c,in1"},"ports":{"in":"c,in0","out":"c,out0"}}, models={"coupler":coupler,"waveguide":waveguide})
S=ring(wl=wl_array)  # S["in","out"] complex transmission
```
gdsfactory: `import gdsfactory as gf; gf.gpdk.PDK.activate(); c=gf.components.ring_single(radius=6.3, gap=0.2, length_x=0.01, length_y=0.01); c.write_gds("x.gds"); c.plot()` (matplotlib) — save with `plt.savefig`.
ngspice batch: write a `.cir` with `.control ... run ... wrdata out.txt v(node) ... quit .endc`, run `ngspice -b file.cir`, parse the whitespace table with numpy.
Manim: `Text("E(z,t) = E₀ cos(ωt − βz)")`, `Axes`, `ax.plot(lambda x: ...)`, `always_redraw`, `ValueTracker`; render `-qm` (720p30). Tested OK.
python-control: `control.tf`, `control.step_response`, `control.bode_plot`, `control.feedback`.

## The experiments (each agent gets one)
01_travelling_wave  [notes 2,3]  Manim video + notebook. Show E(z,t)=E0 cos(ωt−βz) three ways (fixed z: oscillation in time; fixed t: snapshot; both: crest tracking gives v_p=ω/β), the phasor Ẽ(z)e^{jωt} rotating with its real-part projection, and the sign convention (ωt+βz travels backward). Use 1310 nm numbers: T = 4.37 fs, λ0 = 1310 nm, and inside silicon λ = 374 nm, n_eff = 2.5 -> λ_g = 524 nm. Notebook: sliders for n, sign, time. Capstone: the ring transfer function is written in exactly this phasor convention; a round trip is e^{-jβL}.
02_wavevector_evanescent  [notes 4,17]  matplotlib animation. 2-D plane wave cos(ωt − k·r) with k tilted by θ; show k_x and k_z, the phase fronts, wavelengths along each axis (2π/k_z > λ). Then sweep k_z past n2 k0 so k_x becomes imaginary: the same formula turns into e^{-γx}e^{-jβz}; show decay length 1/γ = 102 nm for n_eff 2.5 in silica. Capstone: the tail sets ring-bus coupling κ ∝ e^{-γ·gap}: a 10 nm gap error is 10 % in κ.
03_driven_electron_rlc  [notes 7,8]  ngspice + scipy + notebook. Map the Lorentz oscillator (m, γ, ω0, drive −eE) onto a series RLC (L, R, 1/C, V); run ngspice AC sweep (magnitude+phase of capacitor charge = ∫i) and a transient at three drive frequencies (ω ≪ ω0, ≈ ω0, ≫ ω0) showing phase lag; overlay with the analytic χ(ω)=A/(ω0²−ω²+jγω) computed in numpy, then n'=Re√(1+χ), n''. Show the "in phase = no absorption, 90° = max absorption" power argument with the SPICE resistor power. Scale: choose L,R,C giving f0 = 1 kHz-ish for SPICE, and show the dimensionless mapping to the UV resonance (ω0 ≈ 1.9e16 rad/s) of silica. Capstone: the modulator's plasma-dispersion effect is a change of N (free carriers) in the same χ; the thermo-optic effect is a shift of the resonance/density with T (state qualitatively).
04_sellmeier_dispersion  [notes 9,10,11]  sympy + numpy + notebook. Derive symbolically (print with sympy) n_g = n − λ dn/dλ and D = −(λ/c) d²n/dλ² from n(λ), then evaluate Malitson fused silica (B: 0.6961663, 0.4079426, 0.8974794; C: 0.0684043², 0.1162414², 9.896161² µm²) and crystalline silicon (Salzberg–Villa: n² = 11.6858 + 0.939816/λ² + 0.00810461·1.1071²/(λ² − 1.1071²), λ in µm). Plot n, n_g, D (ps/nm/km) vs λ 0.6–2.0 µm; show the UV and IR Sellmeier terms' separate curvature contributions; find the zero-dispersion wavelength of silica numerically (expect ≈1.27 µm) and D(1550) ≈ 20–22 ps/nm/km for bulk silica (fiber's 17 includes waveguide dispersion: say so). Verify n(1310)=1.4468 for silica; n_g(1310). Check the small-change rule 1 nm ↔ 174.7 GHz. Capstone: why O-band; the group index (not n_eff) sets the ring FSR = c/(n_g L).
05_pulse_dispersion  [notes 12]  numpy FFT propagation. Propagate (a) a Gaussian pulse and (b) a 53.125 Gbaud NRZ PRBS through L = 2 km of a medium with β(ω) from the silica Sellmeier fit (material dispersion only, state it) at 1310 and 1550 nm; show delay t_g = β1 L, broadening from β2 = −λ²D/(2πc); animate the envelope along z; plot eye diagrams at the output for both wavelengths. Expect: at 1310 the eye stays open; at 1550 ~27 ps spread vs 18.8 ps bit period closes it. Capstone: the O-band choice for 2 km links and why Lightmatter works at 1310 nm.
06_absorption_vs_evanescence  [notes 6,25 table]  matplotlib + Meep (.uses_meep). Two panels: (a) plane wave in a lossy medium with n = n' − jn'' (Meep: Medium with D_conductivity, or analytic) decaying along z, intensity α = 4πn''/λ; (b) evanescent field decaying along x at a Si/SiO2 boundary, no decay along z. Compute time-averaged Poynting vectors for both from fields and show S_x = 0 in the evanescent region while S_z > 0; show that the lossy case has ∇·S < 0 (power absorbed). Numbers: doped ring 125 dB/cm -> n'' , α, per-lap retention a = e^{-αL/2} = 0.945. Capstone: the difference between the light that is confined (evanescent, no loss) and the light that heats the ring (absorbed, self-heating, Chapter 9 of the textbook).
07_slab_modes_graphical  [notes 14–19, 21–23]  numpy/scipy + Meep MPB (.uses_meep) + notebook. Solve the symmetric slab (n1=3.50, n2=1.45, 2d = 220 nm, λ=1310 nm) TE eigenvalue equations h tan(hd)=γ and −h cot(hd)=γ with the circle (hd)²+(γd)²=V²: plot the graphical construction, mark roots, list modes (β, n_eff, 1/γ), plot F(x) and also H_x, H_z (∝ F') across x to show E_y and F' continuity at the boundary; sweep thickness and wavelength for cutoff of the second mode. Validate β against Meep's eigenmode solver (`sim.get_eigenmode` on the 2-D slab, or MPB via `mp.MPB`? use meep's `get_eigenmode` with a 1-D cross-section) — report agreement. Show the standing-in-x/travelling-in-z decomposition into two plane waves at θ = asin(n_eff/n1). Notebook with thickness/λ sliders. Capstone: the 2-D strip needs a numerical solver (08); n_eff vs n_g.
08_soi_strip_mode_solvers  [notes 22,23 + capstone]  Tidy3D + femwell (.venv). The actual 500×220 nm SOI strip at 1310 nm: n_eff, n_g, TE fraction, confinement Γ (power in Si), mode field plots (Ex, |E|²) from BOTH solvers; sweep width 300–700 nm (n_eff, n_g, Γ; single-mode cutoff); sweep wavelength 1260–1360. Then the capstone number: perturb n_si by dn_Si/dT·ΔT (ΔT = 10 K, plus silica dn/dT) and recompute n_eff → dn_eff/dT → dλ_r/dT = (λ/n_g)·dn_eff/dT; expect ~50 pm/K (compare with the textbook's Γ-weighted estimate 0.85·1.86e-4 → 49 pm/K); also compute the modulator-relevant "1 K = how many pm = what fraction of the 374 pm FWHM". Note honestly that the textbook's reference n_eff = 2.5 is lower than the solver value (~2.7); explain which quantities depend on n_eff (mode order m, phase) and which on n_g (FSR, all shift formulas), so the capstone numbers stand.
09_tir_evanescent_fdtd  [notes 24,25]  Meep (.uses_meep). 2-D FDTD of a Gaussian beam hitting a flat Si/SiO2 interface at incidence angles below and above the critical angle 24.5° (e.g. 15°, 24.5°, 40°): animate the field; compute the time-averaged Poynting vector field from E and H arrays (Ez, Hx, Hy in 2-D) and draw arrows: below critical, power crosses; above critical, the arrows in silica run parallel to the interface and the normal component averages to zero; measure the evanescent decay length and compare with 1/γ = λ/(2π√(n1² sin²θ − n2²)). Also a Frustrated-TIR variant: bring a second silicon slab within 100–300 nm and show power tunnelling across (the seed of the directional coupler). Capstone: the evanescent tail is what couples ring to bus and what heaters must not touch.
10_slab_mode_fdtd  [notes 20,21]  Meep (.uses_meep). Launch the fundamental TE mode into the 220 nm slab with an eigenmode source; animate E(x,z,t) (standing in x, travelling in z); measure λ_g from peak spacing along z and compare to λ0/n_eff from 07; plot |E|² vs x from the simulation over the analytic profile; compute S_z(x) and the confinement fraction; show a second-order mode (eig_band=2, thicker slab) with its node. Then excite the guide off-centre with a Gaussian source to show a non-modal input decomposing into modes + radiation (the "arbitrary launched field" sentence). Capstone: the mode shape is what Γ (thermo-optic weighting) and coupling depend on.
11_directional_coupler  [notes 26,27]  Meep (.uses_meep) + SAX + gdsfactory. Two identical 220 nm slabs with gap 150/200/300 nm: FDTD field animation of power sloshing; power in each guide vs z (flux monitors or |E|² integrals) fitted to cos²(κ_c z)/sin²; κ_c vs gap on a log axis (exponential with the decay length); supermode picture: compute even and odd supermode β± with the eigenmode solver and check κ_c = (β+ − β−)/2; coupling length. SAX: the 2×2 coupler S-matrix, unitarity check, and the interference of two coherent inputs |s1+s2|² vs |s1|²+|s2|² as a function of relative phase. gdsfactory: draw the ring-bus point coupler geometry (ring R = 6.3 µm, gap 200 nm, 500 nm strips) and a straight directional coupler, export GDS and a PNG; explain the point coupler's effective length √(2πR/γ) ≈ 2 µm. Capstone: κ² ≈ 10.7 % for the reference ring; a 10 nm gap error is ±20 % in κ²; critical coupling t = a is a fabrication lottery.
12_round_trip_to_ring  [notes 28 → capstone]  SAX + Meep (.uses_meep for the Meep part; put the SAX part in a separate script run with .venv, and run.py orchestrates both via subprocess with the right interpreters) + notebook. Part A (SAX): the retention factor a = e^{-αL/2} for the doped ring (125 dB/cm, L = 39.6 µm -> a = 0.945) and passive ring (3 dB/cm -> 0.9986); the all-pass ring spectrum T(λ) = |(t − a e^{-jφ})/(1 − t a e^{-jφ})|²: FSR (expect 10.3 nm / 1.8 THz with n_g 4.2), FWHM (374 pm), Q (3500), critical coupling t = a gives T_min = 0, under/over coupling; then thermal shift: T(λ_L) vs temperature at a fixed laser using 50 pm/K, showing one FSR = 206 K and the 0.1 K = 5 pm resolution question. Part B (Meep 2-D): a ring of radius 6.3 µm (2-D, 500 nm wide, n 3.5 in 1.45) side-coupled to a bus, Gaussian pulse source, Harminv resonances and Q, and a transmission spectrum via flux; compare FSR with c/(n_g L) using the 2-D n_g; animate the field at a resonance (build-up in the ring). Notebook: sliders for a, t, n_g, temperature. Capstone: this is the plant; the laser sits at δ = +108 pm on the slope for maximum OMA; 1 K moves it 50 pm.
13_capstone_ring_lock  [capstone objective]  python-control + numpy + notebook. Build the single-ring thermal plant (heater power → ΔT: R_th 8.8 K/mW, τ 10 µs with a 300 µs 25 % slow tail) → λ_r shift (50 pm/K) → detuning δ = λ_L − λ_r → through-port Lorentzian with T_min 0.016, FWHM 374 pm → photocurrent (0.9 A/W, 2.5 mW, 5 % tap). Sensor gain at δ_opt = 108 pm. Design a PI lock (pole cancellation, crossover ~1/(2·T_d) with T_d = 10 µs sample+delay) with python-control: Bode, margins, step. Time-domain simulation with anti-windup and heater ≥ 0: scenarios (a) 5 K ambient step, (b) 800 K/s ramp for 10 ms, (c) 4 rings with the crosstalk matrix K (10 %/3 %) and a neighbour turning on. Animate the notch sliding under the fixed laser with the lock catching it (video). Report steady-state error in pm and in fraction of FWHM, and the OMA penalty. Capstone: this is the deliverable's core loop; state clearly which numbers are assumed.

Each experiment README must be written for Kenny to read standalone. Prefer clarity over brevity, define every symbol once, and state every assumption.
