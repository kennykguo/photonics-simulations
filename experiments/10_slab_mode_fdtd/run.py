"""10_slab_mode_fdtd: launch the slab's TE modes with FDTD and watch them (docs/NOTES.md sections 20, 21).

Run from this directory with the Meep interpreter:
    cd experiments/10_slab_mode_fdtd && ../../.meep/bin/python run.py

Coordinate mapping between the notes and Meep (Meep 2-D lives in its x-y plane):
    notes z (propagation)  = Meep x        notes x (transverse, across the slab) = Meep y
    notes E_y (the TE field) = Meep Ez     notes H_x (prop. to F)  = Meep Hy      notes H_z (prop. to F') = Meep Hx
so the longitudinal Poynting flux of the notes, <S_z> = -1/2 Re(E_y H_x*), is  -1/2 Re(Ez Hy*)  in Meep components,
and the transverse one, <S_x> = 1/2 Re(E_y H_z*), is  1/2 Re(Ez Hx*).
In the figures the axes are labelled with the notes' names (z along the guide, x across it).
Meep units: lengths in um, frequency 1/lambda[um], time in um/c (one optical period at 1310 nm = 1.31 time units).
Phasor convention: Meep's DFT fields use exp(-i omega t), so a +z wave comes out as exp(+i beta z). The notes use
exp(+j omega t) with a +z wave exp(-j beta z); phasors() conjugates the DFT arrays once so that everything downstream
(phase slope, Poynting, overlaps) is in the notes' convention. Real parts of E x H* are unchanged by this.
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import REF, use_style, SERIES, PALETTE
OUT = pathlib.Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
use_style()

import json, time, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks, hilbert
from scipy.optimize import curve_fit
import meep as mp
from slab_analytic import solve_te_modes, profile, confinement, mode_norm_integral

mp.verbosity(0)
T_START = time.time()
HERE = pathlib.Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- parameters
LAM = REF.lambda_nm * 1e-3          # 1.31 um
N1, N2 = REF.n_si, REF.n_sio2       # 3.50 / 1.45
THICK = REF.wg_height_um            # 0.22 um  (the reference slab of notes section 23)
THICK2 = 0.40                       # um, a thicker slab that guides TE0 and TE1 comfortably
FCEN = 1 / LAM                      # Meep frequency 0.7634
PERIOD = LAM                        # one optical period in Meep time units (um/c)
RES = 50                            # px/um  (20 nm pixels; the 80 nm decay length is 4 px, lambda_g is 22 px)
DT = 0.5 / RES                      # Meep time step (default Courant factor 0.5): 0.01 um/c at 50 px/um
CELL = mp.Vector3(14, 5, 0)         # um, propagation along Meep x
PML = 1.0
WIN = (12.0, 3.0)                   # analysis window (everything outside the PML)
X_SRC = -5.0                        # source plane
STEADY_T = 80.0                     # time units before the CW DFT window opens (61 periods)
DFT_T = 40.0                        # CW DFT accumulation window (30 periods)
FFMPEG = "/opt/homebrew/bin/ffmpeg"
plt.rcParams["animation.ffmpeg_path"] = FFMPEG

for f in OUT.iterdir():             # regenerate everything from scratch
    if f.is_file():
        f.unlink()

RESULTS = {"inputs": {"lambda_nm": REF.lambda_nm, "n1_si": N1, "n2_sio2": N2, "thickness_nm": THICK * 1e3,
                      "thickness2_nm": THICK2 * 1e3, "resolution_px_per_um": RES, "time_step_um_per_c": DT,
                      "cell_um": [CELL.x, CELL.y], "pml_um": PML, "steady_state_time_units": STEADY_T,
                      "dft_window_time_units": DFT_T,
                      "phasor_convention": "notes' exp(+j omega t): Meep DFT arrays (exp(-i omega t)) are conjugated in phasors()"}}

def pct(sim_val, ref_val):
    return 100.0 * (sim_val - ref_val) / ref_val

def fmt(v, spec=".3f", none="n/a"):
    """Format a number, or print 'n/a' for a mode that does not exist (single-mode slab)."""
    return none if v is None else format(v, spec)

def steps_time(n):
    """Argument for sim.run(until=...) that advances exactly n time steps.

    Meep stops when round_time() >= t0 + until, so asking for exactly n*DT can round to n+1 steps; asking for
    (n - 1/2) DT always gives n steps."""
    return (n - 0.5) * DT

# ----------------------------------------------------------------------------- analytic expectations (07 / notes 19,22,23)
modes220 = solve_te_modes(N1, N2, THICK, LAM)
modes400 = solve_te_modes(N1, N2, THICK2, LAM)
TE0 = modes220[0]
if len(modes400) < 2:
    raise SystemExit(f"THICK2 = {THICK2} um guides only {len(modes400)} TE mode at {LAM} um; Part B (TE1) and the beat/arrival "
                     f"measurements need two. Raise THICK2 (TE1 cutoff thickness is {LAM/(2*np.sqrt(N1**2-N2**2)):.3f} um).")
TE1_400 = modes400[1]
N_MODES_220 = len(modes220)          # 2 at 220 nm / 1310 nm; 1 if THICK or LAM is changed past the TE1 cutoff (README item 2, 5)
for label, ms in (("220 nm", modes220), ("400 nm", modes400)):
    print(f"analytic slab {label}:", ", ".join(f"TE{m['order']} n_eff={m['neff']:.4f} lambda_g={1e3*m['lambda_g_um']:.1f} nm "
                                                f"1/gamma={m['decay_len_nm']:.1f} nm Gamma={confinement(m):.4f}" for m in ms))
if N_MODES_220 == 1:
    print(f"    the {1e3*THICK:.0f} nm slab is single-mode at {1e3*LAM:.0f} nm: TE1 entries will be null in results.json")
def group_index(thick, order, dl=1e-4):
    """n_g = n_eff - lambda dn_eff/dlambda by central finite difference (notes section 22)."""
    lo = solve_te_modes(N1, N2, thick, LAM - dl)[order]["neff"]; hi = solve_te_modes(N1, N2, thick, LAM + dl)[order]["neff"]
    return solve_te_modes(N1, N2, thick, LAM)[order]["neff"] - LAM * (hi - lo) / (2 * dl)

NG400 = [group_index(THICK2, i) for i in range(2)]
NG220 = [group_index(THICK, i) for i in range(N_MODES_220)]
print(f"analytic group indices: 220 nm " + " ".join(f"TE{i} {ng:.4f}" for i, ng in enumerate(NG220)) +
      f"; 400 nm TE0 {NG400[0]:.4f} TE1 {NG400[1]:.4f}")
RESULTS["analytic"] = {
    "ng_220nm": NG220, "ng_400nm": NG400,
    "slab_220nm": [{"m": m["order"], "parity": m["parity"], "neff": m["neff"], "beta_rad_per_um": m["beta_per_um"],
                    "lambda_g_nm": 1e3 * m["lambda_g_um"], "decay_len_nm": m["decay_len_nm"], "confinement": confinement(m),
                    "F_edge_over_F_peak": m["edge_over_peak"]} for m in modes220],
    "slab_400nm": [{"m": m["order"], "parity": m["parity"], "neff": m["neff"], "beta_rad_per_um": m["beta_per_um"],
                    "lambda_g_nm": 1e3 * m["lambda_g_um"], "decay_len_nm": m["decay_len_nm"], "confinement": confinement(m)}
                   for m in modes400],
    "notes_sec22_neff": 2.97, "notes_sec22_lambda_g_nm": 441.0, "notes_sec23_decay_len_nm": 80.0,
    "beat_length_400nm_um": 2 * np.pi / (modes400[0]["beta_per_um"] - modes400[1]["beta_per_um"]),
}
# Analytic expectations for the README's "Experiments to try" (same solver, other thickness / wavelength), so that every
# number quoted there is computed here rather than remembered from another experiment's brief.
def _cutoff_thickness_um(lam):
    return lam / (2 * np.sqrt(N1 ** 2 - N2 ** 2))            # TE1 cutoff: V = pi/2
m1550 = solve_te_modes(N1, N2, THICK, 1.55)
m150 = solve_te_modes(N1, N2, 0.15, LAM); m300 = solve_te_modes(N1, N2, 0.30, LAM)
RESULTS["analytic"]["experiments_to_try"] = {
    "220nm_at_1550nm": {"n_modes": len(m1550), "te0_neff": m1550[0]["neff"], "te0_lambda_g_nm": 1e3 * m1550[0]["lambda_g_um"],
                        "te0_decay_len_nm": m1550[0]["decay_len_nm"], "te0_confinement": confinement(m1550[0])},
    "150nm_at_1310nm": {"n_modes": len(m150), "te0_confinement": confinement(m150[0]), "te0_decay_len_nm": m150[0]["decay_len_nm"]},
    "300nm_at_1310nm": {"n_modes": len(m300), "te0_confinement": confinement(m300[0]), "te0_decay_len_nm": m300[0]["decay_len_nm"],
                        "te1_neff": m300[1]["neff"] if len(m300) > 1 else None,
                        "beat_length_um": (2 * np.pi / (m300[0]["beta_per_um"] - m300[1]["beta_per_um"])) if len(m300) > 1 else None},
    "te1_cutoff_thickness_nm_at_1310": 1e3 * _cutoff_thickness_um(LAM), "te1_cutoff_thickness_nm_at_1550": 1e3 * _cutoff_thickness_um(1.55),
    "te1_cutoff_wavelength_nm_for_220nm": 1e3 * 2 * THICK * np.sqrt(N1 ** 2 - N2 ** 2),
}
_e = RESULTS["analytic"]["experiments_to_try"]
print(f"analytic for the README's experiments: 220 nm at 1550 nm n_eff {_e['220nm_at_1550nm']['te0_neff']:.4f}, lambda_g "
      f"{_e['220nm_at_1550nm']['te0_lambda_g_nm']:.1f} nm, 1/gamma {_e['220nm_at_1550nm']['te0_decay_len_nm']:.1f} nm; Gamma at 150/300 nm "
      f"{_e['150nm_at_1310nm']['te0_confinement']:.3f}/{_e['300nm_at_1310nm']['te0_confinement']:.3f}; TE1 cutoff {_e['te1_cutoff_thickness_nm_at_1310']:.1f} nm "
      f"(1310) / {_e['te1_cutoff_thickness_nm_at_1550']:.1f} nm (1550); cutoff wavelength for 220 nm {_e['te1_cutoff_wavelength_nm_for_220nm']:.1f} nm; "
      f"300 nm beat length {fmt(_e['300nm_at_1310nm']['beat_length_um'])} um")

# ----------------------------------------------------------------------------- Meep helpers
def make_sim(thick, sources, res=RES):
    geom = [mp.Block(size=mp.Vector3(mp.inf, thick, mp.inf), material=mp.Medium(index=N1))]
    return mp.Simulation(cell_size=CELL, geometry=geom, sources=sources, default_material=mp.Medium(index=N2),
                         resolution=res, boundary_layers=[mp.PML(PML)])


def window_axes(sim):
    x, y, _, _ = sim.get_array_metadata(center=mp.Vector3(), size=mp.Vector3(*WIN))
    return np.asarray(x), np.asarray(y)


def phasors(sim, dft):
    """Complex field phasors on the analysis window, as (Ez, Hx, Hy) arrays indexed [ix, iy].

    Meep's DFT uses exp(-i omega t), so its phasor of a +z wave is exp(+i beta z). The arrays are conjugated here so
    that the phasors follow the notes' exp(+j omega t) convention (a +z wave is exp(-j beta z), section 3 / 15)."""
    return tuple(np.conj(sim.get_dft_array(dft, c, 0)) for c in (mp.Ez, mp.Hx, mp.Hy))


def poynting(Ez, Hx, Hy):
    """Time-averaged Poynting components in the notes' names: S_long (along the guide), S_trans (across)."""
    return -0.5 * np.real(Ez * np.conj(Hy)), 0.5 * np.real(Ez * np.conj(Hx))


def core_fraction(y, prof, d):
    """Integral of prof over |x| < d divided by the integral over the window, on a fine interpolated grid."""
    yf = np.linspace(y[0], y[-1], 20001)
    pf = np.interp(yf, y, prof)
    return np.trapezoid(pf[np.abs(yf) <= d], yf[np.abs(yf) <= d]) / np.trapezoid(pf, yf)


def neff_from_phase(x, Ez_row, xr=(-3, 3)):
    """beta from the slope of the unwrapped phase of the phasor along the guide.

    In the notes' convention the +z mode is F(x) exp(-j beta z), so the phase falls linearly with z and the fitted
    slope is -beta (no abs(): a positive slope would mean the wave runs the other way)."""
    sel = (x > xr[0]) & (x < xr[1])
    ph = np.unwrap(np.angle(Ez_row))
    slope, _ = np.polyfit(x[sel], ph[sel], 1)
    beta = -slope
    return beta * LAM / (2 * np.pi), beta


def lambda_g_from_peaks(x, Ez_row, xr=(-4.5, 4.5)):
    """Mean crest spacing of a real snapshot along the guide (parabolic sub-pixel refinement)."""
    sel = (x > xr[0]) & (x < xr[1])
    xs, es = x[sel], Ez_row[sel]
    pk, _ = find_peaks(es, height=0.3 * es.max())
    xp = []
    for i in pk:
        if 0 < i < len(es) - 1:
            a, b, c = es[i - 1], es[i], es[i + 1]
            off = 0.5 * (a - c) / (a - 2 * b + c)
            xp.append(xs[i] + off * (xs[1] - xs[0]))
    xp = np.array(xp)
    sp = np.diff(xp)
    return xp, sp.mean(), sp.std()


FRAME_STEPS = int(round(PERIOD / 16 / DT))    # 8 steps of 0.01 = 0.08 um/c = 0.061 period at 50 px/um (exact multiple of dt)
FRAME_DT = FRAME_STEPS * DT

def run_cw_mode(thick, band, parity, res=RES, n_frames=0, frame_steps=FRAME_STEPS):
    """Launch one eigenmode with a CW eigenmode source, reach steady state, then record the DFT phasors
    (and optionally a set of real-time snapshots for the video, exactly frame_steps time steps apart)."""
    src = [mp.EigenModeSource(mp.ContinuousSource(FCEN, width=10), center=mp.Vector3(X_SRC, 0),
                              size=mp.Vector3(0, 3), eig_band=band, eig_parity=parity)]
    sim = make_sim(thick, src, res)
    sim.run(until=STEADY_T)
    dft = sim.add_dft_fields([mp.Ez, mp.Hx, mp.Hy], [FCEN], center=mp.Vector3(), size=mp.Vector3(*WIN), yee_grid=False)
    sim.run(until=DFT_T)
    x, y = window_axes(sim)
    Ez, Hx, Hy = phasors(sim, dft)
    frames, times = [], []
    for _ in range(n_frames):
        frames.append(sim.get_array(center=mp.Vector3(), size=mp.Vector3(*WIN), component=mp.Ez))
        times.append(sim.meep_time())
        sim.run(until=steps_time(frame_steps))
    snap = sim.get_array(center=mp.Vector3(), size=mp.Vector3(*WIN), component=mp.Ez)
    return dict(x=x, y=y, Ez=Ez, Hx=Hx, Hy=Hy, frames=frames, times=np.array(times), snapshot=snap, t_end=sim.meep_time())


# ============================================================================= Part A: the TE0 mode of the 220 nm slab
print("\n[A] TE0 of the 220 nm slab, CW eigenmode source ...")
tA = time.time()
A = run_cw_mode(THICK, band=1, parity=mp.EVEN_Y + mp.ODD_Z, n_frames=120)
x, y = A["x"], A["y"]
iy0 = np.argmin(np.abs(y))
d = THICK / 2
print(f"    Meep time at end {A['t_end']:.1f}; window {A['Ez'].shape} px; {time.time()-tA:.1f} s")

# (1) lambda_g two ways: crest spacing in a real snapshot, and the phase slope of the phasor
crests, lg_peaks, lg_peaks_std = lambda_g_from_peaks(x, A["snapshot"][:, iy0])
neff_phase, beta_phase = neff_from_phase(x, A["Ez"][:, iy0])
lg_phase = 2 * np.pi / beta_phase
print(f"    lambda_g from crest spacing: {1e3*lg_peaks:.2f} +- {1e3*lg_peaks_std:.2f} nm ({len(crests)} crests); "
      f"from phase slope: {1e3*lg_phase:.1f} nm; analytic lambda0/n_eff = {1e3*TE0['lambda_g_um']:.1f} nm")

# (2) transverse profile |E|^2 and S_z(x) vs analytic; decay length; confinement
xsel = (x > -3) & (x < 3)
I_sim = np.mean(np.abs(A["Ez"][xsel]) ** 2, axis=0)
I_sim /= I_sim.max()
F_an = profile(TE0, y)
I_an = F_an ** 2 / (F_an ** 2).max()
S_long, S_trans = poynting(A["Ez"], A["Hx"], A["Hy"])
Sz_prof = S_long[xsel].mean(axis=0)
Sx_prof = S_trans[xsel].mean(axis=0)
gamma_sim = core_fraction(y, Sz_prof, d)
gamma_an = confinement(TE0)
gamma_an_sampled = core_fraction(y, I_an, d)      # analytic profile sampled on the Meep grid, same integrator
tail = (y > d + 0.04) & (y < d + 0.32)
slope_tail, _ = np.polyfit(y[tail], np.log(np.sqrt(I_sim[tail])), 1)
decay_len_sim_nm = -1e3 / slope_tail
rms_dev = np.sqrt(np.mean((I_sim - I_an) ** 2))
edge_ratio_sim = np.sqrt(np.interp(d, y, I_sim) / 1.0)
S_trans_ratio = np.max(np.abs(Sx_prof)) / np.max(Sz_prof)
print(f"    n_eff (phase slope) {neff_phase:.4f} vs analytic {TE0['neff']:.4f} ({pct(neff_phase, TE0['neff']):+.2f} %)")
print(f"    confinement Gamma {gamma_sim:.4f} vs analytic {gamma_an:.4f} ({pct(gamma_sim, gamma_an):+.2f} %); "
      f"decay length {decay_len_sim_nm:.1f} nm vs {TE0['decay_len_nm']:.1f} nm; rms |E|^2 deviation {rms_dev:.4f}; "
      f"max|S_x|/max S_z = {S_trans_ratio:.2e}")

RESULTS["te0_220nm"] = {
    "neff_fdtd_phase_slope": neff_phase, "neff_analytic": TE0["neff"], "neff_diff_pct": pct(neff_phase, TE0["neff"]),
    "beta_fdtd_rad_per_um": beta_phase, "beta_analytic_rad_per_um": TE0["beta_per_um"],
    "lambda_g_crest_spacing_nm": 1e3 * lg_peaks, "lambda_g_crest_spacing_std_nm": 1e3 * lg_peaks_std, "n_crests": int(len(crests)),
    "lambda_g_phase_slope_nm": 1e3 * lg_phase, "lambda_g_analytic_nm": 1e3 * TE0["lambda_g_um"],
    "lambda_g_crest_vs_analytic_pct": pct(lg_peaks, TE0["lambda_g_um"]), "lambda_g_phase_vs_analytic_pct": pct(lg_phase, TE0["lambda_g_um"]),
    "lambda_g_crest_vs_notes_441nm_pct": pct(1e3 * lg_peaks, 441.0),
    "confinement_fdtd": gamma_sim, "confinement_analytic": gamma_an, "confinement_analytic_sampled_on_grid": gamma_an_sampled,
    "confinement_diff_pct": pct(gamma_sim, gamma_an), "confinement_ref_strip_textbook": REF.confinement,
    "decay_length_fit_nm": decay_len_sim_nm, "decay_length_analytic_nm": TE0["decay_len_nm"],
    "decay_length_diff_pct": pct(decay_len_sim_nm, TE0["decay_len_nm"]),
    "profile_rms_deviation_of_normalised_intensity": rms_dev,
    "F_edge_over_F_peak_fdtd": edge_ratio_sim, "F_edge_over_F_peak_analytic": TE0["edge_over_peak"],
    "max_transverse_over_max_longitudinal_poynting": S_trans_ratio,
    "phase_slope_rad_per_um": -beta_phase, "phase_slope_note": "d(phase)/dz of the conjugated (notes-convention) phasor = -beta",
    "video_frame_spacing_time_units": float(A["times"][1] - A["times"][0]),
    "video_frame_spacing_periods": float((A["times"][1] - A["times"][0]) / PERIOD),
}

# ---- figure: field snapshot with the crest spacing annotated
def draw_core(ax, thick, x0=-6, x1=6, color=PALETTE["ink2"]):
    for s in (-1, 1):
        ax.plot([x0, x1], [s * thick / 2, s * thick / 2], "--", color=color, lw=0.9)

def field_map(ax, x, y, F, thick, vmax=None, cmap="RdBu_r", title=None):
    m = vmax or np.abs(F).max()
    im = ax.imshow(F.T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap=cmap,
                   vmin=-m if cmap == "RdBu_r" else 0, vmax=m, aspect="equal", interpolation="nearest")
    draw_core(ax, thick)
    ax.set_xlabel("z along the guide (µm)"); ax.set_ylabel("x across the slab (µm)")
    if title:
        ax.set_title(title)
    return im

fig, ax = plt.subplots(figsize=(11, 2.9))
snap = A["snapshot"]
im = field_map(ax, x, y, snap, THICK, title=f"TE0 of the 220 nm slab: the x-shape is fixed, crests repeat every λ_g = {1e3*lg_peaks:.0f} nm along z")
ax.set_ylim(-1.0, 1.0)
c0, c1 = crests[len(crests) // 2], crests[len(crests) // 2 + 2]
ax.annotate("", xy=(c1, 0.6), xytext=(c0, 0.6), arrowprops=dict(arrowstyle="<->", color=PALETTE["ink"]))
ax.text((c0 + c1) / 2, 0.68, f"2 λ_g = {1e3*(c1-c0):.0f} nm", ha="center", fontsize=9)
ax.plot(crests, np.zeros_like(crests), "v", color=PALETTE["ink"], ms=4, label="crests found along x = 0")
ax.legend(loc="lower right")
fig.colorbar(im, ax=ax, label="E_y (a.u.)", fraction=0.03, pad=0.02)
fig.tight_layout(); fig.savefig(OUT / "10_te0_field_snapshot.png"); plt.close(fig)

# ---- figure: lambda_g measurement two ways
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
axs[0].plot(x, snap[:, iy0] / np.abs(snap[:, iy0]).max(), color=SERIES[0], label="E_y(z) on the axis, one snapshot")
axs[0].plot(crests, np.interp(crests, x, snap[:, iy0]) / np.abs(snap[:, iy0]).max(), "v", color=SERIES[1], label="crests")
axs[0].set_xlim(-2, 2); axs[0].set_ylim(-1.15, 1.8); axs[0].set_xlabel("z (µm)"); axs[0].set_ylabel("E_y (normalised)")
axs[0].set_title(f"Crests are {1e3*lg_peaks:.1f} ± {1e3*lg_peaks_std:.2f} nm apart: λ_g = λ₀/n_eff\n(analytic {1e3*TE0['lambda_g_um']:.1f} nm)", fontsize=10)
axs[0].legend(loc="upper center", ncol=2, fontsize=8, framealpha=0.95)     # in the empty band above the trace, not on it
ph = np.unwrap(np.angle(A["Ez"][:, iy0]))
sel = (x > -3) & (x < 3)
p = np.polyfit(x[sel], ph[sel], 1)
axs[1].plot(x, ph - p[1], color=SERIES[0], label="unwrapped phase of the phasor Ẽ_y(z) (e^{jωt} convention)")
axs[1].plot(x[sel], p[0] * x[sel], "--", color=SERIES[1], label=f"fit: slope = −β = {p[0]:.3f} rad/µm")
axs[1].set_xlabel("z (µm)"); axs[1].set_ylabel("phase (rad)")
axs[1].set_title(f"Phase falls linearly along z: Ẽ_y ∝ e^(−jβz)\nn_eff = β/k₀ = {neff_phase:.4f} (analytic {TE0['neff']:.4f})", fontsize=10)
axs[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "10_lambda_g_measurement.png"); plt.close(fig)

# ---- figure: profile, tail, Poynting
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
for ax_ in axs:
    ax_.axvspan(-d, d, color=PALETTE["line"], alpha=0.6, lw=0)
axs[0].plot(y, I_sim, color=SERIES[0], label="FDTD |Ẽ_y|² (DFT at 1310 nm)")
axs[0].plot(y, I_an, "--", color=SERIES[1], label="analytic F(x)² (notes §19, §23)")
axs[0].set_xlim(-0.6, 0.6); axs[0].set_ylim(0, 1.5); axs[0].set_xlabel("x across the slab (µm)"); axs[0].set_ylabel("|E|² / peak")
axs[0].set_title(f"The FDTD profile is the analytic mode\n(rms deviation {rms_dev:.4f} of the peak)", fontsize=10); axs[0].legend(loc="upper right", fontsize=8)
axs[1].semilogy(y, np.sqrt(I_sim), color=SERIES[0], label="FDTD |Ẽ_y|")
axs[1].semilogy(y, np.abs(F_an), "--", color=SERIES[1], label="analytic |F|")
yy = np.linspace(d, d + 0.45, 50)
axs[1].semilogy(yy, np.sqrt(np.interp(d + 0.04, y, I_sim)) * np.exp(-(yy - d - 0.04) / (decay_len_sim_nm * 1e-3)), ":", color=SERIES[3],
                label=f"fit e^(−x/δ): δ = {decay_len_sim_nm:.0f} nm (analytic {TE0['decay_len_nm']:.0f} nm)")
axs[1].set_xlim(-0.6, 0.6); axs[1].set_ylim(1e-3, 1.5); axs[1].set_xlabel("x (µm)"); axs[1].set_ylabel("|E| / peak")
axs[1].set_title("The cladding tail is a straight line\non a log axis: e^(−γ|x|)", fontsize=10); axs[1].legend(loc="lower center", fontsize=8)
axs[2].plot(y, Sz_prof / Sz_prof.max(), color=SERIES[0], label="⟨S_z⟩ = −½Re(Ẽ_y H̃_x*), along the guide")
axs[2].plot(y, Sx_prof / Sz_prof.max(), color=SERIES[4], label="⟨S_x⟩ = ½Re(Ẽ_y H̃_z*), across (≈ 0)")
axs[2].set_xlim(-0.6, 0.6); axs[2].set_ylim(-0.1, 1.5); axs[2].set_xlabel("x (µm)"); axs[2].set_ylabel("⟨S⟩ / peak ⟨S_z⟩")
axs[2].set_title(f"Power flows only along z\nΓ = {gamma_sim:.3f} inside the core (analytic {gamma_an:.3f})", fontsize=10)
axs[2].legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "10_te0_profile.png"); plt.close(fig)

# ---- video: standing across x, travelling along z (notes section 21)
print("    rendering 10_te0_travelling_mode.mp4 ...")
frames, times = A["frames"], A["times"]
m = max(np.abs(f).max() for f in frames)
fig = plt.figure(figsize=(11, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
ax_map = fig.add_subplot(gs[0, :]); ax_z = fig.add_subplot(gs[1, 0]); ax_x = fig.add_subplot(gs[1, 1])
im = field_map(ax_map, x, y, frames[0], THICK, vmax=m)
ax_map.set_ylim(-0.9, 0.9)
ttl = ax_map.set_title("")
ln_z, = ax_z.plot(x, frames[0][:, iy0] / m, color=SERIES[0])
ax_z.axvline(0, color=PALETTE["muted"], lw=0.8)
ax_z.set_xlim(-2, 2); ax_z.set_ylim(-1.1, 1.1); ax_z.set_xlabel("z (µm)"); ax_z.set_ylabel("E_y(z, x=0) / max")
ax_z.set_title("Along z the pattern travels (crest moves at v_p = ω/β)", fontsize=10)
ix0 = np.argmin(np.abs(x))
ax_x.axvspan(-d, d, color=PALETTE["line"], alpha=0.6, lw=0)
ln_x, = ax_x.plot(y, frames[0][ix0, :] / m, color=SERIES[1])
Fn = F_an * np.sqrt(np.interp(0, y, I_sim)) / np.abs(F_an).max()
ax_x.plot(y, Fn, "--", color=PALETTE["muted"], lw=1, label="±F(x) analytic envelope")
ax_x.plot(y, -Fn, "--", color=PALETTE["muted"], lw=1)
ax_x.set_xlim(-0.6, 0.6); ax_x.set_ylim(-1.1, 1.1); ax_x.set_xlabel("x (µm)"); ax_x.set_ylabel("E_y(z=0, x) / max")
ax_x.set_title("Across x the shape stands: F(x) scales and flips sign", fontsize=10); ax_x.legend(loc="upper right", fontsize=8)

def upd(i):
    im.set_data(frames[i].T)
    ln_z.set_ydata(frames[i][:, iy0] / m)
    ln_x.set_ydata(frames[i][ix0, :] / m)
    ttl.set_text(f"TE0 mode, 220 nm slab, 1310 nm: E_y(x, z, t) = F(x) cos(ωt − βz)   t = {(times[i]-times[0])/PERIOD:.2f} periods")
    return im, ln_z, ln_x, ttl

upd(0); fig.tight_layout()          # lay out with the title text in place so it is not clipped at the top edge
anim = FuncAnimation(fig, upd, frames=len(frames), blit=False)
anim.save(OUT / "10_te0_travelling_mode.mp4", writer=FFMpegWriter(fps=30, bitrate=2500))
plt.close(fig)

def contact_sheet(frames, times, idx, fname, thick, title, cmap="RdBu_r", ylim=(-0.9, 0.9), tlabel="periods", tscale=PERIOD, t0=None):
    fig, axs = plt.subplots(len(idx), 1, figsize=(10, 1.9 * len(idx) + 0.6), sharex=True)
    m = max(np.abs(frames[i]).max() for i in idx)
    for ax_, i in zip(axs, idx):
        field_map(ax_, x, y, frames[i], thick, vmax=m, cmap=cmap)
        ax_.set_ylim(*ylim); ax_.set_xlabel("")
        ax_.text(0.01, 0.85, f"t = {(times[i]-(times[0] if t0 is None else t0))/tscale:.2f} {tlabel}", transform=ax_.transAxes, fontsize=9)
    axs[-1].set_xlabel("z along the guide (µm)")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(OUT / fname); plt.close(fig)

contact_sheet(frames, times, [0, 4, 8, 12, 16, 24], "10_te0_travelling_mode_frames.png", THICK,
              "Stills from 10_te0_travelling_mode.mp4: the crests move right, the transverse shape never changes")

# ============================================================================= Part A2: resolution check of the FDTD n_eff
print("\n[A2] resolution check of n_eff from the FDTD phase slope ...")
conv = []
for res in (30, 50, 80):
    if res == RES:
        conv.append({"resolution": res, "neff": neff_phase, "diff_pct": pct(neff_phase, TE0["neff"])}); continue
    t0 = time.time()
    B = run_cw_mode(THICK, band=1, parity=mp.EVEN_Y + mp.ODD_Z, res=res)
    ne, _ = neff_from_phase(B["x"], B["Ez"][:, np.argmin(np.abs(B["y"]))])
    conv.append({"resolution": res, "neff": ne, "diff_pct": pct(ne, TE0["neff"])})
    print(f"    res {res}: n_eff {ne:.4f} ({pct(ne, TE0['neff']):+.2f} %) in {time.time()-t0:.1f} s")
RESULTS["te0_220nm"]["resolution_convergence"] = conv

# ============================================================================= Part B: the second-order mode in a 400 nm slab
print("\n[B] TE1 of a 400 nm slab, eig_band = 2 ...")
tB = time.time()
Bm = run_cw_mode(THICK2, band=2, parity=mp.ODD_Z)
d2 = THICK2 / 2
iy_off = np.argmin(np.abs(y - 0.12))
neff1, beta1 = neff_from_phase(x, Bm["Ez"][:, iy_off])
crests1, lg1, lg1_std = lambda_g_from_peaks(x, Bm["snapshot"][:, iy_off])
S1_long, _ = poynting(Bm["Ez"], Bm["Hx"], Bm["Hy"])
Sz1 = S1_long[xsel].mean(axis=0)
gamma1_sim = core_fraction(y, Sz1, d2)
gamma1_an = confinement(TE1_400)
# real part of the phasor at one z (rotated so the profile is real): the node is where it crosses zero
col = Bm["Ez"][ix0]
col_r = np.real(col * np.exp(-1j * np.angle(col[iy_off])))
col_r /= np.abs(col_r).max()
F1 = profile(TE1_400, y); F1 /= np.abs(F1).max()
inner = np.abs(y) < 0.15
node_sim = float(np.interp(0.0, col_r[inner], y[inner]))      # zero crossing of the monotonic inner part
# The zero crossing lands on x = 0 to machine precision because the geometry, the grid and the launched band-2 mode are
# all symmetric about x = 0, so it is not a test of the solver. The meaningful checks are how *antisymmetric* the FDTD
# profile is (max |F(x) + F(-x)| / max |F| over the window, which any leakage into the even TE0 or into radiation would
# spoil) and the residual field on the axis itself (x = 0 is a grid point, no interpolation).
antisym_err = float(np.max(np.abs(col_r + col_r[::-1])) / np.max(np.abs(col_r)))   # the y grid is symmetric about 0
if np.sign(F1[iy_off]) != np.sign(col_r[iy_off]):          # same sign convention as the FDTD column for the rms comparison
    F1 = -F1
te1_rms_dev = float(np.sqrt(np.mean((col_r - F1) ** 2)))
print(f"    TE1 n_eff {neff1:.4f} vs analytic {TE1_400['neff']:.4f} ({pct(neff1, TE1_400['neff']):+.2f} %); "
      f"lambda_g {1e3*lg1:.1f} nm vs {1e3*TE1_400['lambda_g_um']:.1f}; Gamma {gamma1_sim:.3f} vs {gamma1_an:.3f}; "
      f"antisymmetry error {antisym_err:.1e}, rms deviation from analytic F1 {te1_rms_dev:.4f}; {time.time()-tB:.1f} s")
RESULTS["te1_400nm"] = {
    "eig_band": 2, "neff_fdtd_phase_slope": neff1, "neff_analytic": TE1_400["neff"], "neff_diff_pct": pct(neff1, TE1_400["neff"]),
    "lambda_g_crest_spacing_nm": 1e3 * lg1, "lambda_g_analytic_nm": 1e3 * TE1_400["lambda_g_um"], "lambda_g_diff_pct": pct(lg1, TE1_400["lambda_g_um"]),
    "confinement_fdtd": gamma1_sim, "confinement_analytic": gamma1_an, "confinement_diff_pct": pct(gamma1_sim, gamma1_an),
    "node_position_nm": 1e3 * node_sim,
    "node_note": "x = 0 to machine precision by construction (symmetric geometry, grid and source); see antisymmetry_error instead",
    "antisymmetry_error_max_abs_F(x)+F(-x)_over_peak": antisym_err,
    "profile_rms_deviation_from_analytic_F1": te1_rms_dev,
    "te0_400nm_neff_analytic": modes400[0]["neff"],
}

fig = plt.figure(figsize=(11, 7.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35])
ax = fig.add_subplot(gs[0, :])
field_map(ax, x, y, Bm["snapshot"], THICK2, title=f"TE1 (eig_band = 2) in a 400 nm slab flips sign across the axis; its crests repeat every λ_g = {1e3*lg1:.0f} nm (analytic {1e3*TE1_400['lambda_g_um']:.0f} nm)")
ax.set_xlim(-5, 5); ax.set_ylim(-0.9, 0.9)
ax = fig.add_subplot(gs[1, 0])
ax.axvspan(-d2, d2, color=PALETTE["line"], alpha=0.6, lw=0)
ax.plot(col_r, y, color=SERIES[0], label="FDTD Re Ẽ_y(x)")
ax.plot(F1, y, "--", color=SERIES[1], label="analytic sin(hx) / e^(−γ|x|)")
ax.axhline(node_sim, color=PALETTE["muted"], lw=0.8); ax.text(0.55, node_sim + 0.03, f"node on the axis; antisymmetry error {antisym_err:.0e}", fontsize=8)
ax.set_ylim(-0.9, 0.9); ax.set_xlabel("F(x) / peak"); ax.set_ylabel("x (µm)"); ax.set_title(f"The profile is odd: it changes sign across x = 0\nn_eff = {neff1:.4f} (analytic {TE1_400['neff']:.4f})", fontsize=10); ax.legend(fontsize=8, loc="lower right")
ax = fig.add_subplot(gs[1, 1])
ax.axvspan(-d2, d2, color=PALETTE["line"], alpha=0.6, lw=0)
ax.plot(Sz1 / Sz1.max(), y, color=SERIES[0], label="FDTD ⟨S_z⟩")
ax.plot(F1 ** 2, y, "--", color=SERIES[1], label="analytic F²")
ax.set_ylim(-0.9, 0.9); ax.set_xlabel("⟨S_z⟩ / peak"); ax.set_title(f"Power flows in two lobes, none on the axis\nΓ = {gamma1_sim:.3f} in the core (analytic {gamma1_an:.3f})", fontsize=10); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "10_te1_mode.png"); plt.close(fig)

# ============================================================================= Part C: an off-centre Gaussian launch decomposes into modes + radiation
print("\n[C] off-centre Gaussian launch (pulse, run until the fields decay) ...")
X_PLANES = (-4.0, -1.0, 2.0, 5.0)
PULSE_FRAMES, PULSE_STEPS, PULSE_T0 = 190, int(round(0.4 / DT)), 20.0     # 40 steps = 0.40 um/c between video frames; record to t = 96
# 190 frames end at t = 20 + 190 * 0.4 = 96 um/c, well after the TE1 packet (arrives ~74 at z = 5 um) has passed the
# arrival plane, so the Hilbert envelopes used for the group-delay measurement fall back to ~0 before the record ends.
PULSE_DT = PULSE_STEPS * DT

def run_offcentre(thick, n_modes, waist=0.25, offset=0.05):
    d_ = thick / 2
    y0 = d_ + offset
    src = [mp.Source(mp.GaussianSource(FCEN, fwidth=0.2 * FCEN), component=mp.Ez, center=mp.Vector3(X_SRC, y0),
                     size=mp.Vector3(0, 1.5), amp_func=lambda p: np.exp(-(p.y / waist) ** 2))]
    sim = make_sim(thick, src)
    dft = sim.add_dft_fields([mp.Ez, mp.Hx, mp.Hy], [FCEN], center=mp.Vector3(), size=mp.Vector3(*WIN), yee_grid=False)
    fluxes = [sim.add_flux(FCEN, 0, 1, mp.FluxRegion(center=mp.Vector3(xp, 0), size=mp.Vector3(0, WIN[1]))) for xp in X_PLANES]
    sim.init_sim()                      # get_array before the first run() segfaults without this
    sim.run(until=PULSE_T0)             # the Gaussian pulse peaks at ~33 time units; skip the blank start
    frames, times = [], []
    for _ in range(PULSE_FRAMES):
        frames.append(sim.get_array(center=mp.Vector3(), size=mp.Vector3(*WIN), component=mp.Ez))
        times.append(sim.meep_time())
        sim.run(until=steps_time(PULSE_STEPS))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(20, mp.Ez, mp.Vector3(5, 0), 1e-5))
    Ez, Hx, Hy = phasors(sim, dft)
    coeffs = []
    for fl in fluxes:
        r = sim.get_eigenmode_coefficients(fl, list(range(1, n_modes + 1)), eig_parity=mp.ODD_Z)
        a2 = np.abs(r.alpha[:, 0, 0]) ** 2       # forward power in each guided band (only the bands that are guided)
        coeffs.append({"flux": float(mp.get_fluxes(fl)[0]), "P_m": [float(v) for v in a2]})
    return dict(Ez=Ez, Hx=Hx, Hy=Hy, frames=frames, times=np.array(times), coeffs=coeffs, y0=y0, waist=waist, t_end=sim.meep_time())


def decompose(C, modes_, x, y):
    """Modal powers along z by overlap with the analytic modes (notes section 20, 26) and the total flux in the window.

    The projection and the norm are both taken over the analysis window (|x| < 1.5 um): a_m = int_win E F_m / int_win F_m^2,
    and the modal power P_m = (beta_m / 2 omega mu) |a_m|^2 int_win F_m^2 is the part of mode m's power that flows inside
    the window, consistent with P_tot which is the total flux through the same window. For the loosely bound TE1 of the
    220 nm slab (647 nm tail) about 1 % of the mode's power lies outside the window; the fraction inside is returned."""
    S_long, _ = poynting(C["Ez"], C["Hx"], C["Hy"])
    P_tot = np.trapezoid(S_long, y, axis=1)
    P_m, win_frac = [], []
    for m_ in modes_:
        F = profile(m_, y)
        N_win = np.trapezoid(F ** 2, y); N_full = mode_norm_integral(m_)
        a = np.trapezoid(C["Ez"] * F[None, :], y, axis=1) / N_win        # modal amplitude a_m(z) (orthogonality of F_m)
        P_m.append(m_["beta_per_um"] / (2 * 2 * np.pi * FCEN) * np.abs(a) ** 2 * N_win)   # (beta/2 omega mu)|a|^2 int F^2  (mu = 1)
        win_frac.append(float(N_win / N_full))
    return P_tot, P_m, win_frac

KEY1 = f"offcentre_{int(round(THICK*1e3))}nm"; KEY2 = f"offcentre_{int(round(THICK2*1e3))}nm"
CC = {}
for thick, ms in ((THICK, modes220), (THICK2, modes400)):
    t0 = time.time()
    C = run_offcentre(thick, len(ms))
    P_tot, P_m, win_frac = decompose(C, ms, x, y)
    ix_ref = np.argmin(np.abs(x - X_PLANES[0]))
    P_ref = P_tot[ix_ref]
    flux_ref_meep = C["coeffs"][0]["flux"]      # Meep's own flux monitor at the reference plane z = X_PLANES[0]
    P_guided = sum(P_m)
    rows = []
    for xp, cf in zip(X_PLANES, C["coeffs"]):
        ix = np.argmin(np.abs(x - xp))
        row = {"z_um": xp, "P_total_window": P_tot[ix] / P_ref, "P_total_meep_flux": cf["flux"] / flux_ref_meep}
        for i in range(2):
            has = i < len(ms)
            row[f"P_te{i}_overlap"] = P_m[i][ix] / P_ref if has else None
            # fully Meep: |alpha_m|^2 from get_eigenmode_coefficients over Meep's flux monitor at the reference plane
            row[f"P_te{i}_meep_coeff"] = cf["P_m"][i] / flux_ref_meep if has else None
        row["radiation_remainder"] = (P_tot[ix] - P_guided[ix]) / P_ref
        rows.append(row)
    far = (x > 0) & (x < 5)
    key = KEY1 if thick == THICK else KEY2
    CC[key] = dict(C=C, P_tot=P_tot, P_m=P_m, P_ref=P_ref, thick=thick, modes=ms)
    RESULTS[key] = {"source": {"type": "Gaussian-profile Ez line source, pulse fcen=1/1.31, fwidth=0.2 fcen",
                               "centre_x_um": C["y0"], "waist_um": C["waist"], "z_um": X_SRC},
                    "normalisation": f"overlap powers divided by the DFT-integrated flux through the window at z = {X_PLANES[0]} um; "
                                     f"Meep coefficients |alpha_m|^2 divided by Meep's own flux monitor at the same plane (independent of the overlap code); "
                                     f"modal powers are the part flowing inside the |x| < {WIN[1]/2} um window",
                    "meep_flux_monitor_over_dft_window_poynting_integral_at_reference_plane": flux_ref_meep / P_ref,
                    "meep_flux_monitor_note": "Meep's add_flux integrates Re(E x H*) without the 1/2 of the time average used in poynting(), "
                                              "so this ratio is ~2; every power here is a ratio at fixed convention, so it cancels",
                    "n_guided_modes": len(ms),
                    "mode_power_fraction_inside_window": win_frac + [None] * (2 - len(ms)),
                    "planes": rows,
                    "meep_time_at_end": C["t_end"], "runtime_s": time.time() - t0}
    for i in range(2):
        has = i < len(ms)
        RESULTS[key][f"P_te{i}_mean_z_0_to_5"] = float(np.mean(P_m[i][far]) / P_ref) if has else None
        RESULTS[key][f"P_te{i}_std_z_0_to_5"] = float(np.std(P_m[i][far]) / P_ref) if has else None
    print(f"    {key}: at z = {X_PLANES[-1]} um  P_TE0 = {fmt(rows[-1]['P_te0_overlap'])} (Meep coeff {fmt(rows[-1]['P_te0_meep_coeff'])}), "
          f"P_TE1 = {fmt(rows[-1]['P_te1_overlap'])} (Meep {fmt(rows[-1]['P_te1_meep_coeff'])}), remainder {rows[-1]['radiation_remainder']:.3f}; "
          f"mode power inside window {', '.join(f'{w:.4f}' for w in win_frac)}; {time.time()-t0:.1f} s")

# largest disagreement between the analytic-overlap and Meep's own modal powers, over all planes and both guided modes
overlap_vs_meep_max = max(abs(p[f"P_te{i}_overlap"] - p[f"P_te{i}_meep_coeff"])
                          for k in (KEY1, KEY2) for p in RESULTS[k]["planes"]
                          for i in range(2) if p[f"P_te{i}_overlap"] is not None)
RESULTS["overlap_vs_meep_coeff_max_abs_difference"] = overlap_vs_meep_max
print(f"    overlap decomposition vs get_eigenmode_coefficients: max |difference| {overlap_vs_meep_max:.4f} over all planes and modes")

# beat length in the 400 nm slab from the sloshing of the intensity centroid across the core.
# With only TE0 and TE1 present, <x>(z) = c + A cos((beta0 - beta1) z + phi) exactly (the cross term of |a0 F0 + a1 F1 e^{-j dbeta z}|^2
# is the only odd-in-x part, and int F0 F1 dx = 0 keeps the denominator constant), so a cosine fit over the region where the
# field is modal gives the beat length. Near the source the radiation field still pollutes the centroid, so the fit starts at
# z = 1 um. The older find_peaks window from z = -3.5 um is kept as a diagnostic of how much that contamination shifts the answer.
C4 = CC[KEY2]
I4 = np.abs(C4["C"]["Ez"]) ** 2
core4 = np.abs(y) <= d2
cent = np.trapezoid(I4[:, core4] * y[core4], y[core4], axis=1) / np.trapezoid(I4[:, core4], y[core4], axis=1)
beat_an = RESULTS["analytic"]["beat_length_400nm_um"]
BEAT_FIT_Z = (1.0, 5.5)
sel_f = (x > BEAT_FIT_Z[0]) & (x < BEAT_FIT_Z[1])
zf, cf_ = x[sel_f], cent[sel_f]

def _cosine(z, c, A, K, phi):
    return c + A * np.cos(K * z + phi)

# coarse scan of K with a linear fit of (c, a, b) for c + a cos(Kz) + b sin(Kz), then a nonlinear refinement from the best K
best = None
for K in np.linspace(0.5 * 2 * np.pi / beat_an, 1.5 * 2 * np.pi / beat_an, 2001):
    M = np.column_stack([np.ones_like(zf), np.cos(K * zf), np.sin(K * zf)])
    coef, res, *_ = np.linalg.lstsq(M, cf_, rcond=None)
    r = float(np.sum((M @ coef - cf_) ** 2))
    if best is None or r < best[0]:
        best = (r, K, coef)
_, K0, (c0, a0, b0) = best
popt, pcov = curve_fit(_cosine, zf, cf_, p0=[c0, np.hypot(a0, b0), K0, np.arctan2(-b0, a0)])
K_fit, K_err = popt[2], float(np.sqrt(pcov[2, 2]))
beat_meas = float(2 * np.pi / K_fit)
beat_err = float(beat_meas * K_err / K_fit)                # 1-sigma from the fit covariance
beat_fit_rms = float(np.sqrt(np.mean((_cosine(zf, *popt) - cf_) ** 2)))
cent_fit = _cosine(x, *popt)
# diagnostic: the peak-spacing estimate over the old window that starts next to the source
sel_b = (x > -3.5) & (x < 5.5)
pk_b, _ = find_peaks(cent[sel_b], prominence=0.02)
sp_b = np.diff(x[sel_b][pk_b])
beat_peaks_old = float(sp_b.mean()) if len(sp_b) > 1 else float("nan")
beat_peaks_old_std = float(sp_b.std()) if len(sp_b) > 1 else float("nan")
RESULTS[KEY2]["beat_length_measured_um"] = beat_meas
RESULTS[KEY2]["beat_length_measured_1sigma_um"] = beat_err
RESULTS[KEY2]["beat_length_method"] = f"c + A cos(K z + phi) fitted to the core intensity centroid <x>(z) over z in {BEAT_FIT_Z} um (field modal there); L = 2 pi / K"
RESULTS[KEY2]["beat_length_fit_rms_residual_um"] = beat_fit_rms
RESULTS[KEY2]["beat_length_centroid_amplitude_um"] = float(abs(popt[1]))
RESULTS[KEY2]["beat_length_analytic_um"] = beat_an
RESULTS[KEY2]["beat_length_diff_pct"] = pct(beat_meas, beat_an)
RESULTS[KEY2]["beat_length_peaks_from_z_minus3p5_um"] = beat_peaks_old
RESULTS[KEY2]["beat_length_peaks_from_z_minus3p5_spacing_std_um"] = beat_peaks_old_std
RESULTS[KEY2]["beat_length_peaks_from_z_minus3p5_diff_pct"] = pct(beat_peaks_old, beat_an)
RESULTS[KEY2]["beat_length_peaks_from_z_minus3p5_note"] = ("diagnostic only: find_peaks on the centroid from z = -3.5 um, where radiation "
                                                          "still distorts it; the spread of the spacings is the size of that contamination")
print(f"    400 nm slab: TE0/TE1 beat length {beat_meas:.3f} ± {beat_err:.3f} um (cosine fit, z in {BEAT_FIT_Z}) vs 2π/(β0−β1) = {beat_an:.3f} um "
      f"({pct(beat_meas, beat_an):+.1f} %); peak spacing from z = -3.5 um: {beat_peaks_old:.3f} ± {beat_peaks_old_std:.3f} um ({pct(beat_peaks_old, beat_an):+.1f} %, "
      f"{len(sp_b)} spacings, contaminated by radiation near the source)")

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(x, 1e3 * cent, color=SERIES[0], label="intensity centroid ⟨x⟩(z) inside the core (FDTD)")
ax.plot(x[sel_f], 1e3 * cent_fit[sel_f], "--", color=SERIES[1], lw=1.6, label=f"fit c + A cos(2πz/L + φ), z ∈ [{BEAT_FIT_Z[0]:.0f}, {BEAT_FIT_Z[1]:.1f}] µm: L = {beat_meas:.3f} ± {beat_err:.3f} µm")
ax.axvspan(-6, BEAT_FIT_Z[0], color=PALETTE["line"], alpha=0.5, lw=0, label="radiation still contaminates the centroid here")
ax.set_xlim(-5, 6); ax.set_ylim(-140, 215); ax.set_xlabel("z (µm)"); ax.set_ylabel("⟨x⟩ of |Ẽ_y|² in the core (nm)")
ax.set_title(f"The light sloshes across the 400 nm core every L = {beat_meas:.2f} µm (analytic 2π/(β₀ − β₁) = {beat_an:.3f} µm, {pct(beat_meas, beat_an):+.1f} %)", fontsize=10)
ax.legend(fontsize=8, loc="upper left", ncol=1, framealpha=0.95)      # in the empty band above the trace
fig.tight_layout(); fig.savefig(OUT / "10_offcentre_beat.png"); plt.close(fig)

# group velocity: modal amplitude envelopes vs time at several planes from the real-time frames (notes section 22, n_g).
# Each mode's packet is projected out of every frame at z = 0, 1, ..., 5 um (past the radiation zone); the arrival time of
# its envelope peak is linear in z and the slope dt/dz is n_g of that mode (time in um/c, so v_g = c/n_g). This measures
# n_g0 and n_g1 separately, which is a better test than the single delay at z = 5 um: the delay is the difference of two
# group indices, so it inherits a relative error several times larger than either one.
X_ARR = 5.0
ARR_PLANES = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
fr400 = C4["C"]["frames"]; t400 = C4["C"]["times"]

def arrival_from_columns(cols, t, modes_, y_):
    """cols[k] = list of E_y(x) columns (one per frame) at plane k. Returns per-mode peak times [mode][plane],
    the normalised envelopes at the last plane, their end-of-record levels and the energy-centroid times at the last plane."""
    t_pk = [[] for _ in modes_]; env_last, env_end_, t_cen_last = [], [], []
    for mi, m_ in enumerate(modes_):
        F = profile(m_, y_); N = np.trapezoid(F ** 2, y_)              # window norm, as in decompose()
        for k, col_list in enumerate(cols):
            a_t = np.array([np.trapezoid(c * F, y_) / N for c in col_list])
            e = np.abs(hilbert(a_t))
            i = int(np.argmax(e)); a_, b_, c_ = e[i - 1], e[i], e[i + 1]
            t_pk[mi].append(float(t[i] + 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_) * (t[1] - t[0])))
            if k == len(cols) - 1:
                env_last.append(e / e.max()); env_end_.append(float(e[-1] / e.max()))
                t_cen_last.append(float(np.trapezoid(t * e ** 2, t) / np.trapezoid(e ** 2, t)))
    return t_pk, env_last, env_end_, t_cen_last

cols50 = [[f[np.argmin(np.abs(x - zp)), :] for f in fr400] for zp in ARR_PLANES]
t_pk, env, env_end, t_cent = arrival_from_columns(cols50, t400, modes400, y)
ng_fit = [np.polyfit(ARR_PLANES, t_pk[i], 1) for i in range(2)]          # slope = n_g, intercept = time at z = 0
ng_meas = [float(p[0]) for p in ng_fit]
ng_resid = [float(np.sqrt(np.mean((np.polyval(p, ARR_PLANES) - np.array(tp)) ** 2))) for p, tp in zip(ng_fit, t_pk)]
t_launch = [float(p[1] + p[0] * X_SRC) for p in ng_fit]                  # extrapolated back to the source plane
t_peak = [t_pk[i][-1] for i in range(2)]
delay_meas = t_peak[1] - t_peak[0]
delay_cent = t_cent[1] - t_cent[0]
delay_an = (NG400[1] - NG400[0]) * (X_ARR - X_SRC)
dng_meas = ng_meas[1] - ng_meas[0]; dng_an = NG400[1] - NG400[0]
# does the 20 % pulse bandwidth bias the comparison? weight n_g(lambda) of each mode by its packet's power spectrum at z = 5 um
spec_ng, spec_lam = [], []
for mi, m_ in enumerate(modes400):
    F = profile(m_, y); N = np.trapezoid(F ** 2, y)
    a_t = np.array([np.trapezoid(c * F, y) / N for c in cols50[-1]])
    fq = np.fft.rfftfreq(len(a_t), t400[1] - t400[0]); w = np.abs(np.fft.rfft(a_t)) ** 2
    sel_q = (fq > 0.6 * FCEN) & (fq < 1.4 * FCEN)
    spec_lam.append(float(1e3 / (np.sum(fq[sel_q] * w[sel_q]) / np.sum(w[sel_q]))))
    spec_ng.append(float(np.sum(w[sel_q] * np.array([group_index(THICK2, mi) if abs(1 / f_ - LAM) < 1e-9 else
                                                   (lambda lam_: solve_te_modes(N1, N2, THICK2, lam_)[mi]["neff"] - lam_ *
                                                    (solve_te_modes(N1, N2, THICK2, lam_ + 1e-4)[mi]["neff"] - solve_te_modes(N1, N2, THICK2, lam_ - 1e-4)[mi]["neff"]) / 2e-4)(1 / f_)
                                                   for f_ in fq[sel_q]])) / np.sum(w[sel_q])))
RESULTS[KEY2]["group_index_from_arrival_slope"] = {
    "packet_spectrum_centre_wavelength_nm": spec_lam,
    "packet_spectrum_weighted_ng_analytic": spec_ng,
    "packet_spectrum_note": "n_g(lambda) of each mode weighted by |a_m(f)|^2 of its packet at z = 5 um; equal to n_g(1310 nm) means the pulse bandwidth does not bias the comparison",
    "method": f"peak time of each mode's Hilbert envelope at z = {ARR_PLANES.tolist()} um, straight-line fit: slope dt/dz = n_g (t in um/c)",
    "ng_te0_fdtd": ng_meas[0], "ng_te0_analytic": NG400[0], "ng_te0_diff_pct": pct(ng_meas[0], NG400[0]),
    "ng_te1_fdtd": ng_meas[1], "ng_te1_analytic": NG400[1], "ng_te1_diff_pct": pct(ng_meas[1], NG400[1]),
    "ng_difference_fdtd": dng_meas, "ng_difference_analytic": dng_an, "ng_difference_diff_pct": pct(dng_meas, dng_an),
    "fit_rms_residual_time_units": ng_resid, "arrival_time_vs_z": {"z_um": ARR_PLANES.tolist(), "te0": t_pk[0], "te1": t_pk[1]},
    "extrapolated_launch_time_at_source_plane": t_launch,
}
RESULTS[KEY2]["arrival_time_te0_at_z5um"] = t_peak[0]
RESULTS[KEY2]["arrival_time_te1_at_z5um"] = t_peak[1]
RESULTS[KEY2]["arrival_delay_te1_minus_te0_measured"] = delay_meas
RESULTS[KEY2]["arrival_delay_te1_minus_te0_energy_centroid"] = delay_cent
RESULTS[KEY2]["arrival_delay_analytic_(ng1-ng0)L"] = delay_an
RESULTS[KEY2]["arrival_delay_diff_pct"] = pct(delay_meas, delay_an)
RESULTS[KEY2]["arrival_delay_energy_centroid_diff_pct"] = pct(delay_cent, delay_an)
RESULTS[KEY2]["arrival_delay_uncertainty_time_units"] = float(max(abs(delay_cent - delay_meas), max(ng_resid)))
RESULTS[KEY2]["arrival_delay_note"] = ("the delay is (n_g1 - n_g0) L with both n_g slightly high from grid dispersion (see "
                                      "group_index_from_arrival_slope and its resolution table): the difference inherits a larger relative error")
RESULTS[KEY2]["arrival_envelope_at_end_of_record_over_peak"] = env_end
RESULTS[KEY2]["arrival_record_end_time_units"] = float(t400[-1])
RESULTS[KEY2]["frame_spacing_time_units"] = float(t400[1] - t400[0])
print(f"    400 nm slab: arrival-time slopes give n_g0 = {ng_meas[0]:.4f} ({pct(ng_meas[0], NG400[0]):+.2f} %), n_g1 = {ng_meas[1]:.4f} "
      f"({pct(ng_meas[1], NG400[1]):+.2f} %); launch times extrapolated to the source plane {t_launch[0]:.1f} / {t_launch[1]:.1f}; "
      f"packet centre wavelengths {spec_lam[0]:.1f} / {spec_lam[1]:.1f} nm, spectrum-weighted analytic n_g {spec_ng[0]:.4f} / {spec_ng[1]:.4f}")
print(f"    400 nm slab: TE1 arrives {delay_meas:.2f} time units after TE0 at z = 5 um (envelope peaks; energy centroids give {delay_cent:.2f}); "
      f"(n_g1 - n_g0) L = {delay_an:.2f} ({pct(delay_meas, delay_an):+.1f} %); envelopes at the end of the record (t = {t400[-1]:.1f}): "
      f"TE0 {env_end[0]:.3f}, TE1 {env_end[1]:.3f} of peak")

# resolution check of the group indices: the same launch, recording only the six columns (no video, no DFT), at 30 and 80 px/um
def run_offcentre_columns(thick, res, planes, waist=0.25, offset=0.05):
    d_ = thick / 2
    src = [mp.Source(mp.GaussianSource(FCEN, fwidth=0.2 * FCEN), component=mp.Ez, center=mp.Vector3(X_SRC, d_ + offset),
                     size=mp.Vector3(0, 1.5), amp_func=lambda p: np.exp(-(p.y / waist) ** 2))]
    sim = make_sim(thick, src, res)
    dt_ = 0.5 / res; steps = int(round(0.4 / dt_))
    sim.init_sim()
    sim.run(until=PULSE_T0)
    y_ = np.asarray(sim.get_array_metadata(center=mp.Vector3(), size=mp.Vector3(0, WIN[1]))[1])
    cols = [[] for _ in planes]; times = []
    for _ in range(PULSE_FRAMES):
        for k, zp in enumerate(planes):
            cols[k].append(sim.get_array(center=mp.Vector3(zp, 0), size=mp.Vector3(0, WIN[1]), component=mp.Ez))
        times.append(sim.meep_time())
        sim.run(until=(steps - 0.5) * dt_)
    return cols, np.array(times), y_

ng_conv = []
for res in (30, 50, 80):
    if res == RES:
        ng_conv.append({"resolution": res, "ng_te0": ng_meas[0], "ng_te0_diff_pct": pct(ng_meas[0], NG400[0]), "ng_te1": ng_meas[1],
                        "ng_te1_diff_pct": pct(ng_meas[1], NG400[1]), "ng_difference_diff_pct": pct(dng_meas, dng_an)}); continue
    t0 = time.time()
    cols_r, t_r, y_r = run_offcentre_columns(THICK2, res, ARR_PLANES)
    tp_r, *_ = arrival_from_columns(cols_r, t_r, modes400, y_r)
    ng_r = [float(np.polyfit(ARR_PLANES, tp_r[i], 1)[0]) for i in range(2)]
    ng_conv.append({"resolution": res, "ng_te0": ng_r[0], "ng_te0_diff_pct": pct(ng_r[0], NG400[0]), "ng_te1": ng_r[1],
                    "ng_te1_diff_pct": pct(ng_r[1], NG400[1]), "ng_difference_diff_pct": pct(ng_r[1] - ng_r[0], dng_an)})
    print(f"    res {res}: n_g0 {ng_r[0]:.4f} ({pct(ng_r[0], NG400[0]):+.2f} %), n_g1 {ng_r[1]:.4f} ({pct(ng_r[1], NG400[1]):+.2f} %), "
          f"difference {pct(ng_r[1] - ng_r[0], dng_an):+.1f} % in {time.time()-t0:.1f} s")
RESULTS[KEY2]["group_index_from_arrival_slope"]["resolution_convergence"] = ng_conv

fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axs[0]
for i, (e, c) in enumerate(zip(env, SERIES)):
    ax.plot(t400, e, color=c, label=f"TE{i} envelope |a_{i}(t)| at z = {X_ARR:.0f} µm")
    ax.axvline(t_peak[i], color=c, ls=":", lw=1)
ax.annotate("", xy=(t_peak[1], 1.04), xytext=(t_peak[0], 1.04), arrowprops=dict(arrowstyle="<->", color=PALETTE["ink"]))
ax.text((t_peak[0] + t_peak[1]) / 2, 1.07, f"Δt = {delay_meas:.2f} ± {RESULTS[KEY2]['arrival_delay_uncertainty_time_units']:.2f} µm/c  (analytic (n_g1 − n_g0)·L = {delay_an:.2f})", ha="center", fontsize=9)
ax.set_xlabel("t (µm/c)"); ax.set_ylabel("modal amplitude envelope / peak"); ax.set_ylim(0, 1.18); ax.set_xlim(t400[0], t400[-1])
ax.set_title("The two packets arrive at different times, and both have fully passed\nbefore the record ends", fontsize=10)
ax.legend(loc="upper left", fontsize=8)
ax = axs[1]
zz = np.linspace(X_SRC, X_ARR + 0.5, 10)
for i, c in enumerate(SERIES[:2]):
    ax.plot(ARR_PLANES, t_pk[i], "o", color=c, ms=5, label=f"TE{i} envelope peak (FDTD)")
    ax.plot(zz, np.polyval(ng_fit[i], zz), "-", color=c, lw=1.2, label=f"fit: slope n_g = {ng_meas[i]:.3f} (analytic {NG400[i]:.3f})")
ax.axvline(X_SRC, color=PALETTE["muted"], lw=0.8, ls="--"); ax.text(X_SRC + 0.1, np.polyval(ng_fit[0], X_SRC) + 1, "source plane", fontsize=8)
ax.set_xlabel("z (µm)"); ax.set_ylabel("arrival time of the packet peak (µm/c)")
ax.set_title("Each packet's arrival time is linear in z: the slope is n_g\n(both lines extrapolate to the source pulse peak)", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "10_offcentre_arrival.png"); plt.close(fig)

# ---- figure: steady-state intensity maps and the power bookkeeping along z
fig, axs = plt.subplots(2, 2, figsize=(13, 7.2), gridspec_kw={"width_ratios": [1.6, 1]}, layout="constrained")
for row, key in enumerate((KEY1, KEY2)):
    D = CC[key]; thick = D["thick"]; ms = D["modes"]
    I = np.abs(D["C"]["Ez"]) ** 2; I /= I[(x > -4)].max()
    ax = axs[row, 0]
    im_b = ax.imshow(np.sqrt(I).T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="Blues", vmin=0, vmax=1, aspect="equal")
    draw_core(ax, thick, color=PALETTE["orange"])
    ax.plot([X_SRC], [D["C"]["y0"]], "o", color=PALETTE["red"], ms=5, label=f"Gaussian source, centre x = {1e3*D['C']['y0']:.0f} nm")
    for xp in X_PLANES:
        ax.axvline(xp, color=PALETTE["muted"], lw=0.7, ls=":")
    ax.set_xlim(-6, 6); ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("z (µm)"); ax.set_ylabel("x (µm)")
    if key == KEY2:
        ax.set_title(f"In the 400 nm slab TE0 and TE1 beat with period {beat_meas:.2f} µm while the radiation leaves", fontsize=10)
    elif len(ms) > 1:
        ax.set_title(f"The {1e3*thick:.0f} nm slab carries TE0 plus a loosely bound TE1 (n_eff {ms[1]['neff']:.3f}, tail {ms[1]['decay_len_nm']:.0f} nm) plus radiation", fontsize=10)
    else:
        ax.set_title(f"The {1e3*thick:.0f} nm slab is single-mode at {1e3*LAM:.0f} nm, so only TE0 and radiation remain", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax = axs[row, 1]
    Pn = D["P_tot"] / D["P_ref"]
    ax.plot(x, Pn, color=PALETTE["ink"], label="total flux through the 3 µm window")
    for i in range(len(ms)):
        ax.plot(x, D["P_m"][i] / D["P_ref"], color=SERIES[i], label=f"TE{i} (overlap with analytic F{'₀₁'[i]})")
    ax.plot(x, Pn - sum(D["P_m"]) / D["P_ref"], color=SERIES[3], label="remainder = radiation still in window")
    rows = RESULTS[key]["planes"]
    for i in range(len(ms)):
        ax.plot([r["z_um"] for r in rows], [r[f"P_te{i}_meep_coeff"] for r in rows], "x", color=SERIES[i], ms=8, mew=2,
                label="Meep get_eigenmode_coefficients" if i == 0 else None)
    ax.set_xlim(-4.5, 6); ax.set_ylim(-0.05, 1.3)
    ax.set_xlabel("z (µm)"); ax.set_ylabel(f"power / total at z = {X_PLANES[0]:.0f} µm")
    ax.set_title("Modal powers stay constant; only the radiation part fades", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
fig.colorbar(im_b, ax=[axs[0, 0], axs[1, 0]], fraction=0.025, pad=0.02, label="|Ẽ_y| / max (steady state at 1310 nm)")
fig.suptitle("A non-modal launch = Σ a_m e_m e^(−jβ_m z) + radiation (notes §20): the guided parts keep their power, the rest leaves")
fig.savefig(OUT / "10_offcentre_decomposition.png"); plt.close(fig)

# ---- video: the pulse launched off-centre, both slabs
print("    rendering 10_offcentre_pulse.mp4 ...")
fr220 = CC[KEY1]["C"]["frames"]
tt = t400
m220 = max(np.abs(f).max() for f in fr220[20:]); m400 = max(np.abs(f).max() for f in fr400[20:])
fig, axs = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
im1 = field_map(axs[0], x, y, fr220[0], THICK, vmax=m220); axs[0].set_xlabel("")
im2 = field_map(axs[1], x, y, fr400[0], THICK2, vmax=m400)
for ax_, D in zip(axs, (CC[KEY1], CC[KEY2])):
    ax_.plot([X_SRC], [D["C"]["y0"]], "o", color=PALETTE["red"], ms=5)
    ax_.set_ylim(-1.5, 1.5)
t1 = axs[0].set_title(""); t2 = axs[1].set_title("")

def upd2(i):
    im1.set_data(fr220[i].T); im2.set_data(fr400[i].T)
    t1.set_text(f"{1e3*THICK:.0f} nm slab, Gaussian pulse launched at x = +{1e3*CC[KEY1]['C']['y0']:.0f} nm:   t = {tt[i]:.1f} µm/c ({tt[i]/PERIOD:.0f} periods)")
    t2.set_text(f"{1e3*THICK2:.0f} nm slab: TE0 and TE1 are both excited; they beat, and travel at different group velocities")
    return im1, im2, t1, t2

upd2(0); fig.tight_layout(rect=[0, 0, 1, 0.97])   # lay out with the titles in place so the top one is not clipped
anim = FuncAnimation(fig, upd2, frames=len(fr220), blit=False)
anim.save(OUT / "10_offcentre_pulse.mp4", writer=FFMpegWriter(fps=30, bitrate=2500))
plt.close(fig)
contact_sheet(fr400, tt, [20, 40, 60, 80, 110, 135], "10_offcentre_pulse_frames.png", THICK2,      # last still: TE1 packet reaching z = 5 um
              "Stills from 10_offcentre_pulse.mp4 (400 nm slab): radiation fans out, the guided part keeps going", ylim=(-1.5, 1.5),
              tlabel="µm/c (source peaks at ≈33)", tscale=1.0, t0=0.0)

# ============================================================================= capstone numbers
dl_dT_slab = (REF.lambda_nm * 1e3 / REF.ng) * (gamma_sim * REF.dn_si_dT + (1 - gamma_sim) * REF.dn_sio2_dT)
dl_dT_text = (REF.lambda_nm * 1e3 / REF.ng) * (REF.confinement * REF.dn_si_dT + (1 - REF.confinement) * REF.dn_sio2_dT)
# First-order perturbation theory for a TE slab mode: d(n_eff) = sum_i (n_i / n_eff) dn_i f_i with f_i the fraction of
# int |E|^2 in region i (equal to the S_z fraction Gamma for TE, since S_z = (beta/2 omega mu) |F|^2). The n_i/n_eff
# weights are what the Gamma-weighted textbook heuristic above leaves out; experiment 08 does this properly for the strip.
dneff_dT_pert = (N1 / neff_phase) * gamma_sim * REF.dn_si_dT + (N2 / neff_phase) * (1 - gamma_sim) * REF.dn_sio2_dT
dl_dT_pert = (REF.lambda_nm * 1e3 / REF.ng) * dneff_dT_pert
gap_err_nm = 10.0
kappa_change = np.exp(-gap_err_nm / decay_len_sim_nm) - 1
RESULTS["capstone"] = {
    "dlambda_dT_pm_per_K_from_fdtd_confinement": dl_dT_slab, "dlambda_dT_pm_per_K_textbook_Gamma_0p85": dl_dT_text,
    "dlambda_dT_pm_per_K_reference": REF.dlambda_dT_pm_per_K,
    "dlambda_dT_vs_reference_pct": pct(dl_dT_slab, REF.dlambda_dT_pm_per_K),
    "formula": "Gamma-weighted textbook heuristic: dlambda_r/dT = (lambda/n_g) [Gamma dn_Si/dT + (1-Gamma) dn_SiO2/dT], lambda 1310 nm, n_g 4.2, dn_Si/dT 1.86e-4, dn_SiO2/dT 1e-5",
    "dneff_dT_perturbation_theory_per_K": dneff_dT_pert,
    "dlambda_dT_pm_per_K_perturbation_theory_slab": dl_dT_pert,
    "perturbation_formula": "dn_eff/dT = (n_Si/n_eff) Gamma dn_Si/dT + (n_SiO2/n_eff) (1-Gamma) dn_SiO2/dT (first-order, TE slab, FDTD n_eff and Gamma); this is the proper weighting, done for the strip in 08",
    "coupling_kappa_change_for_10nm_gap_error_pct": 100 * kappa_change,
    "coupling_kappa2_change_for_10nm_gap_error_pct": 100 * ((1 + kappa_change) ** 2 - 1),
    "coupling_note": "kappa is proportional to the evanescent tail e^(-gamma gap): a 10 nm gap error changes kappa by e^(-10/delta) - 1 with delta the fitted decay length",
    "one_kelvin_in_fwhm_fractions": REF.dlambda_dT_pm_per_K / REF.fwhm_pm,
}
RESULTS["runtime_s"] = time.time() - T_START

with open(OUT / "results.json", "w") as fh:
    json.dump(RESULTS, fh, indent=2, default=float)
with open(OUT / "results.txt", "w") as fh:
    r = RESULTS["te0_220nm"]
    fh.write("10_slab_mode_fdtd headline numbers (Meep 2-D FDTD, resolution %d px/um)\n" % RES)
    fh.write(f"TE0 220 nm slab: n_eff {r['neff_fdtd_phase_slope']:.4f} vs analytic {r['neff_analytic']:.4f} ({r['neff_diff_pct']:+.2f} %)\n")
    fh.write(f"  lambda_g crest spacing {r['lambda_g_crest_spacing_nm']:.1f} nm, phase slope {r['lambda_g_phase_slope_nm']:.1f} nm, analytic {r['lambda_g_analytic_nm']:.1f} nm, notes 441 nm\n")
    fh.write(f"  confinement {r['confinement_fdtd']:.4f} vs analytic {r['confinement_analytic']:.4f} ({r['confinement_diff_pct']:+.2f} %)\n")
    fh.write(f"  decay length {r['decay_length_fit_nm']:.1f} nm vs analytic {r['decay_length_analytic_nm']:.1f} nm ({r['decay_length_diff_pct']:+.2f} %)\n")
    fh.write(f"  max|S_x|/max S_z = {r['max_transverse_over_max_longitudinal_poynting']:.2e}\n")
    r = RESULTS["te1_400nm"]
    fh.write(f"TE1 400 nm slab: n_eff {r['neff_fdtd_phase_slope']:.4f} vs {r['neff_analytic']:.4f} ({r['neff_diff_pct']:+.2f} %), antisymmetry error {r['antisymmetry_error_max_abs_F(x)+F(-x)_over_peak']:.1e}, Gamma {r['confinement_fdtd']:.3f} vs {r['confinement_analytic']:.3f}\n")
    for key in (KEY1, KEY2):
        p = RESULTS[key]["planes"][-1]
        fh.write(f"{key}: at z = {p['z_um']} um P_TE0 {fmt(p['P_te0_overlap'])} (Meep {fmt(p['P_te0_meep_coeff'])}), P_TE1 {fmt(p['P_te1_overlap'])} (Meep {fmt(p['P_te1_meep_coeff'])}), radiation remainder {p['radiation_remainder']:.3f}\n")
    fh.write(f"overlap vs get_eigenmode_coefficients: max |difference| {overlap_vs_meep_max:.4f} (all planes, both modes)\n")
    g = RESULTS[KEY2]["group_index_from_arrival_slope"]
    fh.write(f"400 nm group indices from arrival-time slopes: n_g0 {g['ng_te0_fdtd']:.4f} vs {g['ng_te0_analytic']:.4f} ({g['ng_te0_diff_pct']:+.2f} %), n_g1 {g['ng_te1_fdtd']:.4f} vs {g['ng_te1_analytic']:.4f} ({g['ng_te1_diff_pct']:+.2f} %)\n")
    fh.write(f"400 nm TE1-TE0 arrival delay at z=5 um {RESULTS[KEY2]['arrival_delay_te1_minus_te0_measured']:.2f} +- {RESULTS[KEY2]['arrival_delay_uncertainty_time_units']:.2f} vs (ng1-ng0)L {RESULTS[KEY2]['arrival_delay_analytic_(ng1-ng0)L']:.2f} time units ({RESULTS[KEY2]['arrival_delay_diff_pct']:+.1f} %, difference of two grid-dispersed n_g)\n")
    fh.write(f"400 nm beat length {RESULTS[KEY2]['beat_length_measured_um']:.3f} +- {RESULTS[KEY2]['beat_length_measured_1sigma_um']:.3f} um (cosine fit, z > 1 um) vs {RESULTS[KEY2]['beat_length_analytic_um']:.3f} um ({RESULTS[KEY2]['beat_length_diff_pct']:+.1f} %)\n")
    fh.write(f"capstone: dlambda/dT from FDTD Gamma (textbook Gamma-weighted heuristic) = {RESULTS['capstone']['dlambda_dT_pm_per_K_from_fdtd_confinement']:.1f} pm/K (reference 50); "
             f"first-order perturbation weighting = {RESULTS['capstone']['dlambda_dT_pm_per_K_perturbation_theory_slab']:.1f} pm/K\n")
    fh.write(f"runtime {RESULTS['runtime_s']:.0f} s\n")

# ----------------------------------------------------------------------------- tools.json
def ver(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()[0]
    except Exception as e:  # pragma: no cover
        return f"unknown ({e})"

import scipy, matplotlib as _mpl
r0 = RESULTS["te0_220nm"]; r1 = RESULTS["te1_400nm"]; c4 = RESULTS[KEY2]["planes"][-1]; c2 = RESULTS[KEY1]["planes"][-1]
g4 = RESULTS[KEY2]["group_index_from_arrival_slope"]
TOOLS = [
    {"tool": "Meep (pymeep) 2-D FDTD with EigenModeSource, add_dft_fields, add_flux, get_eigenmode_coefficients", "version": f"meep {mp.__version__}",
     "what_it_is": "Meep is the open-source finite-difference time-domain (FDTD) Maxwell solver from MIT: it steps E and H on a Yee grid in time and needs no assumption about modes, so whatever field pattern the equations allow is what appears. It embeds the MPB eigenmode solver, which is used here to build mode sources and to project fields onto modes.",
     "used_for": f"A 2-D slab (n 3.50 core in n 1.45 cladding, 220 nm and 400 nm thick, 14 x 5 um cell, 1 um PML, 50 px/um). (A) a CW EigenModeSource launches TE0 (eig_band=1, EVEN_Y+ODD_Z); after 80 time units the complex phasors of Ez, Hx, Hy at 1310 nm are accumulated with add_dft_fields for 30 periods (Meep's DFT uses exp(-i omega t); the arrays are conjugated once to the notes' exp(+j omega t) convention) and 120 real-time snapshots, {FRAME_STEPS} time steps = {FRAME_DT:.2f} um/c = {FRAME_DT/PERIOD:.3f} period apart, are taken for the video. (B) the same with eig_band=2 in the 400 nm slab for TE1. (C) a Gaussian-profile pulsed Ez line source 50 nm outside the core edge, run until the fields decay; flux monitors at four z planes plus get_eigenmode_coefficients give the TE0/TE1 power at each plane; {PULSE_FRAMES} real-time frames {PULSE_DT:.1f} um/c apart (to t = {PULSE_T0 + PULSE_FRAMES*PULSE_DT:.0f}) give each mode's packet arrival time at z = 0 to 5 um, and the same launch at 30 and 80 px/um (recording only those six columns) checks how the group indices converge.",
     "result": f"TE0: n_eff from the phasor phase slope {r0['neff_fdtd_phase_slope']:.4f} vs analytic {r0['neff_analytic']:.4f} ({r0['neff_diff_pct']:+.2f} %, resolution table in results.json: {', '.join(f'{c['resolution']} px/um -> {c['diff_pct']:+.2f} %' for c in conv)}); crest spacing lambda_g {r0['lambda_g_crest_spacing_nm']:.1f} nm vs lambda0/n_eff {r0['lambda_g_analytic_nm']:.1f} nm ({r0['lambda_g_crest_vs_analytic_pct']:+.2f} %); confinement {r0['confinement_fdtd']:.4f} vs {r0['confinement_analytic']:.4f} ({r0['confinement_diff_pct']:+.2f} %); tail decay length {r0['decay_length_fit_nm']:.1f} nm vs {r0['decay_length_analytic_nm']:.1f} nm ({r0['decay_length_diff_pct']:+.1f} %); max transverse Poynting {r0['max_transverse_over_max_longitudinal_poynting']:.1e} of the longitudinal peak. TE1 (400 nm): n_eff {r1['neff_fdtd_phase_slope']:.4f} vs {r1['neff_analytic']:.4f} ({r1['neff_diff_pct']:+.2f} %), antisymmetry error max|F(x)+F(-x)|/max|F| = {r1['antisymmetry_error_max_abs_F(x)+F(-x)_over_peak']:.1e}. Off-centre launch, 400 nm slab, at z = 5 um: TE0 {c4['P_te0_overlap']:.3f} / TE1 {c4['P_te1_overlap']:.3f} of the launched forward power by overlap vs {c4['P_te0_meep_coeff']:.3f} / {c4['P_te1_meep_coeff']:.3f} from get_eigenmode_coefficients normalised by Meep's flux monitor at z = -4 um; 220 nm slab: TE0 {fmt(c2['P_te0_overlap'])} / TE1 {fmt(c2['P_te1_overlap'])} vs {fmt(c2['P_te0_meep_coeff'])} / {fmt(c2['P_te1_meep_coeff'])}; beat length {RESULTS[KEY2]['beat_length_measured_um']:.3f} +- {RESULTS[KEY2]['beat_length_measured_1sigma_um']:.3f} um (cosine fit to the core intensity centroid over z = 1 to 5.5 um) vs 2pi/(beta0-beta1) = {RESULTS[KEY2]['beat_length_analytic_um']:.3f} um ({RESULTS[KEY2]['beat_length_diff_pct']:+.1f} %); group indices from the slope of each packet's arrival time vs z: n_g0 {g4['ng_te0_fdtd']:.3f} vs {g4['ng_te0_analytic']:.3f} ({g4['ng_te0_diff_pct']:+.1f} %), n_g1 {g4['ng_te1_fdtd']:.3f} vs {g4['ng_te1_analytic']:.3f} ({g4['ng_te1_diff_pct']:+.1f} %) (grid dispersion, converging: {', '.join(f'{c['resolution']} px/um -> {c['ng_te0_diff_pct']:+.1f} / {c['ng_te1_diff_pct']:+.1f} %' for c in ng_conv)}); the TE1 packet arrives {RESULTS[KEY2]['arrival_delay_te1_minus_te0_measured']:.2f} +- {RESULTS[KEY2]['arrival_delay_uncertainty_time_units']:.2f} time units after TE0 at z = 5 um vs (n_g1-n_g0)L = {RESULTS[KEY2]['arrival_delay_analytic_(ng1-ng0)L']:.2f} ({RESULTS[KEY2]['arrival_delay_diff_pct']:+.1f} %, the difference of two n_g each ~1 % high).",
     "how_to_observe": f"cd experiments/10_slab_mode_fdtd && ../../.meep/bin/python run.py  (about 1.5 to 2.5 min depending on what else the CPU is doing; this run took {RESULTS['runtime_s']:.0f} s, see results.json runtime_s). Watch out/10_te0_travelling_mode.mp4 (top: E_y(x,z,t); bottom-left: the crest moving along z; bottom-right: the fixed shape across x scaling and flipping sign), out/10_offcentre_pulse.mp4 (the pulse breaking into a guided part and radiation), and the stills out/10_te0_field_snapshot.png, out/10_te1_mode.png, out/10_offcentre_decomposition.png. Change RES (30-80), THICK (0.15-0.30; below the 205.6 nm TE1 cutoff the slab is single-mode and the TE1 entries become null), THICK2, LAM, the source offset/waist in run_offcentre(), or STEADY_T/DFT_T at the top of run.py and re-run."},
    {"tool": "numpy + scipy (analytic slab eigenvalue solver, overlap integrals, peak finding, least-squares fits)", "version": f"numpy {np.__version__}, scipy {scipy.__version__}",
     "what_it_is": "numpy is the array library every scientific Python tool is built on; scipy adds numerical routines (root finding, signal processing, optimisation).",
     "used_for": "slab_analytic.py solves h tan(hd) = gamma and -h cot(hd) = gamma with brentq (the roots from 07), builds F(x), and integrates F^2 in closed form for the confinement; run.py finds crests with scipy.signal.find_peaks (parabolic sub-pixel refinement), fits the unwrapped phase for beta, fits log|E| in the cladding for the decay length, integrates S_z(x) on an interpolated fine grid for Gamma, and projects the FDTD phasors onto the analytic F_m (orthogonality of the slab modes) to get a_m(z) and the modal powers along z; scipy.optimize.curve_fit fits c + A cos(K z + phi) to the core intensity centroid over z = 1 to 5.5 um for the beat length (1-sigma from the covariance), and numpy.polyfit of each packet's Hilbert-envelope peak time against z gives n_g as the slope.",
     "result": f"Analytic TE0 at 220 nm: n_eff {TE0['neff']:.4f}, lambda_g {1e3*TE0['lambda_g_um']:.1f} nm, 1/gamma {TE0['decay_len_nm']:.1f} nm, Gamma {gamma_an:.4f} (the notes' rounded 2.97 / 441 nm / 80 nm, section 22-23). Overlap decomposition (projection and norm both over the 3 um window) agrees with Meep's own get_eigenmode_coefficients to within {overlap_vs_meep_max:.3f} (absolute, in units of the launched power; max over both modes and all eight monitored planes).",
     "how_to_observe": "python slab_analytic.py prints the mode tables; the fits are in run.py functions neff_from_phase, lambda_g_from_peaks, core_fraction and decompose; their outputs are in out/results.json under te0_220nm, te1_400nm, offcentre_*."},
    {"tool": "matplotlib (figures, FuncAnimation + FFMpegWriter videos)", "version": f"matplotlib {_mpl.__version__}",
     "what_it_is": "The standard Python plotting library; its animation module writes frame sequences to video through ffmpeg.",
     "used_for": f"All PNG figures with the shared style (RdBu_r for signed fields centred on zero, Blues for |E|), the two MP4 videos (30 fps, 120 and {PULSE_FRAMES} frames) and their contact sheets.",
     "result": "out/10_te0_field_snapshot.png, 10_lambda_g_measurement.png, 10_te0_profile.png, 10_te1_mode.png, 10_offcentre_decomposition.png, 10_offcentre_beat.png, 10_offcentre_arrival.png, 10_te0_travelling_mode.mp4 (+_frames.png), 10_offcentre_pulse.mp4 (+_frames.png).",
     "how_to_observe": f"Open the files in out/. Frame spacing is FRAME_STEPS = {FRAME_STEPS} time steps = {FRAME_DT:.2f} um/c = {FRAME_DT/PERIOD:.3f} period for the mode video (the nearest whole number of 0.01 um/c steps to PERIOD/16) and PULSE_STEPS = {PULSE_STEPS} steps = {PULSE_DT:.2f} um/c for the pulse video; change n_frames / PULSE_FRAMES in run.py for longer clips."},
    {"tool": "ffmpeg", "version": ver([FFMPEG, "-version"]),
     "what_it_is": "The command-line video encoder/decoder that matplotlib's FFMpegWriter pipes frames into.",
     "used_for": "Encoding the two MP4 (H.264) videos.",
     "result": f"out/10_te0_travelling_mode.mp4 (4 s) and out/10_offcentre_pulse.mp4 ({PULSE_FRAMES/30:.1f} s).",
     "how_to_observe": "/opt/homebrew/bin/ffprobe out/10_te0_travelling_mode.mp4 shows the stream; FFMpegWriter(fps=30, bitrate=2500) in run.py sets the encoding."},
]
with open(OUT / "tools.json", "w") as fh:
    json.dump(TOOLS, fh, indent=2)

print(f"\nDone in {RESULTS['runtime_s']:.0f} s. Outputs in {OUT}")
