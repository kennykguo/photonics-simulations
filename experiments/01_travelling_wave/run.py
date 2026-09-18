"""01_travelling_wave: what E(z,t) = E0 cos(wt - bz) means, three ways, and why phasors work.

Notes sections 2 (travelling wave: fixed z, fixed t, crest tracking -> v_p = w/b) and
3 (phasor E~(z) e^{jwt}, real-part projection, the e^{jwt}/e^{-jbz} sign convention).
Everything in this experiment is ONE closed-form field evaluated with numpy; nothing is
"solved". The point is to see the same formula from three viewpoints, to check the
phasor identity numerically, to measure v_p and lambda_g from the pictures instead of
reading them off the formula, and to connect e^{-jbz} to the ring's round trip e^{-jbL}.

Run from this directory:
    cd experiments/01_travelling_wave && ../../.venv/bin/python run.py
Everything in out/ is regenerated (figures, two Manim videos + contact sheets,
results.json, tools.json, run_log.txt) and explore.ipynb is rebuilt and executed.
"""
import sys, pathlib, json, time, math, shutil, subprocess, platform
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import REF, use_style, SERIES, PALETTE, C0
from common.params import k0_per_um, omega_rad_s, freq_thz
OUT = pathlib.Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
use_style()

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"
import numpy as np
import scipy
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import imageio.v2 as imageio

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from twave import (wave_numbers, E_real, E_phasor, E_from_phasor, track_crest, fit_velocity,
                   slab_profile_illustrative, round_trip_phase, neff_of_lambda)

VENV = ROOT / ".venv" / "bin"
FFMPEG, FFPROBE = "/opt/homebrew/bin/ffmpeg", "/opt/homebrew/bin/ffprobe"
T_START = time.time()
RESULTS, LOG = {}, []


def log(msg=""):
    print(msg); LOG.append(str(msg))


def pct(a, b):
    """Percent agreement of a with the expectation b: 100 * (1 - |a-b|/|b|)."""
    return 100.0 * (1.0 - abs(a - b) / abs(b))


# --------------------------------------------------------------------------- clean out/
for f in OUT.iterdir():
    if f.is_file():
        f.unlink()
for d in (HERE / "media", HERE / "__pycache__"):
    shutil.rmtree(d, ignore_errors=True)

# --------------------------------------------------------------------------- 1. numbers
lam0_nm = REF.lambda_nm
W0 = wave_numbers(lam0_nm, n=1.0)          # vacuum
WSI = wave_numbers(lam0_nm, n=REF.n_si)    # bulk silicon plane wave
W = wave_numbers(lam0_nm, n=REF.neff)      # the guided mode, textbook n_eff = 2.5
OMEGA, BETA = W["omega_rad_s"], W["beta_rad_m"]
T_FS = W["T_s"] * 1e15
LAM_G_NM = W["lambda_med_m"] * 1e9
VP = W["v_p_m_s"]

RESULTS.update({
    "lambda0_nm": lam0_nm,
    "frequency_THz": W["f_Hz"] / 1e12,
    "frequency_THz_expected": freq_thz(lam0_nm),
    "omega_rad_per_s": OMEGA,
    "omega_rad_per_s_expected": omega_rad_s(lam0_nm),
    "period_fs": T_FS,
    "period_fs_expected_brief": 4.37,
    "k0_rad_per_um": W["k0_rad_m"] * 1e-6,
    "k0_rad_per_um_expected": k0_per_um(lam0_nm),
    "n_si": REF.n_si,
    "lambda_in_bulk_si_nm": WSI["lambda_med_m"] * 1e9,
    "lambda_in_bulk_si_nm_expected_brief": 374.0,
    "neff": REF.neff,
    "beta_rad_per_um": BETA * 1e-6,
    "lambda_g_nm": LAM_G_NM,
    "lambda_g_nm_expected_brief": 524.0,
    "v_p_m_per_s": VP,
    "v_p_m_per_s_expected": C0 / REF.neff,
})
log(f"lambda0 = {lam0_nm:.0f} nm  f = {W['f_Hz']/1e12:.2f} THz  omega = {OMEGA:.4e} rad/s  T = {T_FS:.3f} fs")
log(f"k0 = {W['k0_rad_m']*1e-6:.4f} rad/um; bulk Si (n={REF.n_si}): lambda = {WSI['lambda_med_m']*1e9:.1f} nm")
log(f"mode n_eff = {REF.neff}: beta = {BETA*1e-6:.3f} rad/um, lambda_g = {LAM_G_NM:.1f} nm, v_p = {VP:.4e} m/s")

# --------------------------------------------------------------------------- 2. the phasor identity
z_chk = np.linspace(0, 3 * LAM_G_NM, 601) * 1e-9
t_chk = np.linspace(0, 3 * T_FS, 301) * 1e-15
ZZ, TT = np.meshgrid(z_chk, t_chk)
err = np.max(np.abs(E_from_phasor(ZZ, TT, OMEGA, BETA) - E_real(ZZ, TT, OMEGA, BETA)))
RESULTS["phasor_identity_max_abs_error"] = float(err)
log(f"max |Re{{E~(z) e^(jwt)}} - E0 cos(wt - bz)| over a {ZZ.shape} grid = {err:.2e}  (NOTES 3)")

# --------------------------------------------------------------------------- 3. figure: three ways
z_nm = np.linspace(0, 3 * LAM_G_NM, 1200)
t_fs = np.linspace(0, 3 * T_FS, 600)
Zg, Tg = np.meshgrid(z_nm * 1e-9, t_fs * 1e-15)
E_zt = E_real(Zg, Tg, OMEGA, BETA)

fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
ax = axs[0]
for i, zf in enumerate([0.0, LAM_G_NM / 4, LAM_G_NM / 2]):
    ax.plot(t_fs, E_real(zf * 1e-9, t_fs * 1e-15, OMEGA, BETA), color=SERIES[i],
            label=f"z = {zf:.0f} nm (βz = {BETA*zf*1e-9/np.pi:.2f}π)")
ax.set_xlabel("t (fs)"); ax.set_ylabel("E / E₀")
ax.set_title(f"Fixed z: a sinusoid in time, period T = {T_FS:.2f} fs;\nmoving the probe along z only delays it by βz/ω", fontsize=10)
ax.set_ylim(-1.2, 1.75); ax.legend(loc="upper center", ncol=3, fontsize=7, columnspacing=0.8)
ax = axs[1]
for i, tf in enumerate([0.0, T_FS / 4, T_FS / 2]):
    ax.plot(z_nm, E_real(z_nm * 1e-9, tf * 1e-15, OMEGA, BETA), color=SERIES[i],
            label=f"t = {tf:.2f} fs")
ax.set_xlabel("z (nm)"); ax.set_ylabel("E / E₀")
ax.set_title(f"Fixed t: a snapshot in z, crest spacing λ_g = {LAM_G_NM:.0f} nm;\nlater snapshots = same shape shifted by v_p·Δt", fontsize=10)
ax.set_ylim(-1.2, 1.75); ax.legend(loc="upper center", ncol=3, fontsize=8)
ax = axs[2]
m = 1.0
im = ax.imshow(E_zt, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-m, vmax=m,
               extent=[z_nm[0], z_nm[-1], t_fs[0], t_fs[-1]])
for k in range(-2, 4):
    zc = k * LAM_G_NM + VP * t_fs * 1e-15 * 1e9
    ok = (zc >= 0) & (zc <= z_nm[-1])
    ax.plot(zc[ok], t_fs[ok], color=PALETTE["ink"], lw=1.0, ls="--")
ax.plot([], [], color=PALETTE["ink"], lw=1.0, ls="--", label="crests ωt − βz = 2πm, slope v_p = ω/β")
ax.set_xlabel("z (nm)"); ax.set_ylabel("t (fs)")
ax.set_title("Both varying: E(z,t) as a map. Crests are straight\nlines of slope dz/dt = ω/β: the wave moves to +z", fontsize=10)
ax.legend(loc="upper left", fontsize=8); ax.grid(False)
cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("E / E₀")
fig.tight_layout(); fig.savefig(OUT / "three_ways.png"); plt.close(fig)

# --------------------------------------------------------------------------- 4. crest tracking -> v_p
z_trk = np.linspace(0, 3 * LAM_G_NM, 4000) * 1e-9
t_trk = np.linspace(0, 1.8 * T_FS, 400) * 1e-15
zc_fwd = track_crest(z_trk, t_trk, OMEGA, BETA, sign=-1, z_start=LAM_G_NM * 1e-9)
zc_bwd = track_crest(z_trk, t_trk, OMEGA, BETA, sign=+1, z_start=2 * LAM_G_NM * 1e-9)
v_fwd, b_fwd = fit_velocity(t_trk, zc_fwd)
v_bwd, b_bwd = fit_velocity(t_trk, zc_bwd)
RESULTS["v_p_tracked_forward_m_per_s"] = float(v_fwd)
RESULTS["v_p_tracked_forward_agreement_pct"] = pct(v_fwd, VP)
RESULTS["v_p_tracked_backward_m_per_s"] = float(v_bwd)
RESULTS["v_p_tracked_backward_agreement_pct"] = pct(v_bwd, -VP)
log(f"crest tracking: forward  v = {v_fwd:.5e} m/s vs w/b = {VP:.5e} ({pct(v_fwd, VP):.4f} %)")
log(f"crest tracking: backward v = {v_bwd:.5e} m/s vs -w/b = {-VP:.5e} ({pct(v_bwd, -VP):.4f} %)")

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
ax = axs[0]
ax.plot(t_trk * 1e15, zc_fwd * 1e9, color=SERIES[0], lw=3, alpha=0.6, label="tracked crest of cos(ωt − βz)")
ax.plot(t_trk * 1e15, (v_fwd * t_trk + b_fwd) * 1e9, color=SERIES[0], lw=1, ls="--",
        label=f"fit: slope {v_fwd:.4e} m/s")
ax.plot(t_trk * 1e15, zc_bwd * 1e9, color=SERIES[4], lw=3, alpha=0.6, label="tracked crest of cos(ωt + βz)")
ax.plot(t_trk * 1e15, (v_bwd * t_trk + b_bwd) * 1e9, color=SERIES[4], lw=1, ls="--",
        label=f"fit: slope {v_bwd:.4e} m/s")
ax.set_xlabel("t (fs)"); ax.set_ylabel("crest position z_c (nm)")
ax.set_title(f"Following one crest numerically gives a straight line\nwhose slope is ω/β = c/n_eff = {VP:.4e} m/s")
ax.legend(fontsize=8)
ax = axs[1]
ax.plot(t_trk * 1e15, (zc_fwd - (v_fwd * t_trk + b_fwd)) * 1e15, color=SERIES[0], label="forward")
ax.plot(t_trk * 1e15, (zc_bwd - (v_bwd * t_trk + b_bwd)) * 1e15, color=SERIES[4], label="backward")
ax.set_xlabel("t (fs)"); ax.set_ylabel("residual (fm)")
ax.set_title("Fit residual is femtometres (parabolic sub-grid peak\nestimate on a 0.39 nm grid): the speed is constant", fontsize=10)
ax.set_ylim(-0.2, 0.24)
ax.legend(loc="upper center", ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "crest_tracking.png"); plt.close(fig)

# --------------------------------------------------------------------------- 5. lambda_g measured from a snapshot
snap = E_real(z_trk, 0.37 * T_FS * 1e-15, OMEGA, BETA)
pk, _ = find_peaks(snap)
lam_meas = np.mean(np.diff(z_trk[pk])) * 1e9
RESULTS["lambda_g_measured_from_snapshot_nm"] = float(lam_meas)
RESULTS["lambda_g_measured_agreement_pct"] = pct(lam_meas, LAM_G_NM)
log(f"crest spacing in a snapshot = {lam_meas:.2f} nm vs lambda0/n_eff = {LAM_G_NM:.2f} nm ({pct(lam_meas, LAM_G_NM):.3f} %)")

# --------------------------------------------------------------------------- 6. figure: three wavelengths
z3 = np.linspace(0, 1500, 3000)
fig, ax = plt.subplots(figsize=(11, 3.8))
for i, (lab, Wx) in enumerate([("vacuum, n = 1", W0), (f"bulk silicon, n = {REF.n_si}", WSI),
                                (f"guided mode, n_eff = {REF.neff}", W)]):
    y = E_real(z3 * 1e-9, 0.0, Wx["omega_rad_s"], Wx["beta_rad_m"])
    ax.plot(z3, y + 2.6 * (2 - i), color=SERIES[i],
            label=f"{lab}: λ = {Wx['lambda_med_m']*1e9:.0f} nm, β = {Wx['beta_rad_m']*1e-6:.2f} rad/µm")
ax.set_xlabel("z (nm)"); ax.set_yticks([]); ax.set_ylabel("E / E₀  (traces offset by 2.6)")
ax.set_title("Same frequency (228.8 THz, T = 4.37 fs) everywhere; the wavelength along z shrinks by the index:\n"
             "1310 nm in vacuum, 374 nm in bulk silicon, 524 nm for the mode (n_eff between cladding and core)")
ax.set_ylim(-1.3, 8.0)
ax.legend(loc="upper center", fontsize=8, ncol=3)
fig.tight_layout(); fig.savefig(OUT / "three_wavelengths.png"); plt.close(fig)

# --------------------------------------------------------------------------- 7. figure: phasor helix + clock
fig = plt.figure(figsize=(13, 4.6))
ax3 = fig.add_subplot(1, 3, 1, projection="3d")
zh = np.linspace(0, 2 * LAM_G_NM, 500)
ph = E_phasor(zh * 1e-9, BETA)
ax3.plot(zh, ph.real, ph.imag, color=SERIES[0], lw=2, label="Ẽ(z) = E₀ e^{−jβz}")
ax3.plot(zh, ph.real, -1.35 * np.ones_like(zh), color=SERIES[3], lw=1.2, label="Re Ẽ(z) = E(z, t=0)")
ax3.set_xlabel("z (nm)"); ax3.set_ylabel("Re Ẽ / E₀"); ax3.set_zlabel("Im Ẽ / E₀")
ax3.set_title("The spatial phasor is a helix along z\n(one turn per λ_g = 524 nm, clockwise for +z)", fontsize=10)
ax3.legend(fontsize=7, loc="upper left"); ax3.view_init(elev=22, azim=-60)

ax = fig.add_subplot(1, 3, 2)
z0 = 250e-9
tks = np.linspace(0, T_FS, 7)[:-1]
cmap = plt.get_cmap("viridis")
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(th), np.sin(th), color=PALETTE["line"], lw=1)
ax.axhline(0, color=PALETTE["muted"], lw=0.8); ax.axvline(0, color=PALETTE["muted"], lw=0.8)
for i, tk in enumerate(tks):
    p = E_phasor(z0, BETA) * np.exp(1j * OMEGA * tk * 1e-15)
    ax.add_patch(FancyArrowPatch((0, 0), (p.real, p.imag), arrowstyle="-|>", mutation_scale=14,
                                 color=cmap(i / len(tks)), lw=1.8))
    ax.plot([p.real, p.real], [p.imag, 0], color=cmap(i / len(tks)), lw=0.8, ls=":")
    ax.plot(p.real, 0, "o", color=cmap(i / len(tks)), ms=6)
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
ax.set_xlabel("Re Ẽ / E₀  (dimensionless)"); ax.set_ylabel("Im Ẽ / E₀  (dimensionless)")
ax.set_title(f"At z₀ = {z0*1e9:.0f} nm the phasor Ẽ(z₀)e^{{jωt}} turns\ncounter-clockwise once per T = {T_FS:.2f} fs (dark → light)", fontsize=10)

ax = fig.add_subplot(1, 3, 3)
tt = np.linspace(0, 1.5 * T_FS, 400)
ax.plot(tt, E_real(z0, tt * 1e-15, OMEGA, BETA), color=SERIES[3], label="E₀ cos(ωt − βz₀)")
ax.plot(tt, E_from_phasor(z0, tt * 1e-15, OMEGA, BETA), color=SERIES[1], ls="--", lw=1.2,
        label="Re{Ẽ(z₀) e^{jωt}}")
for i, tk in enumerate(tks):
    ax.plot(tk, E_real(z0, tk * 1e-15, OMEGA, BETA), "o", color=cmap(i / len(tks)), ms=6)
ax.set_xlabel("t (fs)"); ax.set_ylabel("E / E₀")
ax.set_title(f"Its real-axis projection IS the field at z₀;\nmax |difference| over a {ZZ.shape[0]}×{ZZ.shape[1]} (t,z) grid = {err:.1e}", fontsize=10)
ax.set_ylim(-1.75, 1.15)
ax.legend(fontsize=8, loc="lower center", ncol=2)
fig.tight_layout(); fig.savefig(OUT / "phasor.png"); plt.close(fig)

# --------------------------------------------------------------------------- 8. figure: sign convention
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, sgn, name, col, vfit in [(axs[0], -1, "cos(ωt − βz): +z", SERIES[0], v_fwd),
                                 (axs[1], +1, "cos(ωt + βz): −z", SERIES[4], v_bwd)]:
    Ezt = E_real(Zg, Tg, OMEGA, BETA, sign=sgn)
    im = ax.imshow(Ezt, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                   extent=[z_nm[0], z_nm[-1], t_fs[0], t_fs[-1]])
    for k in range(-3, 5):
        zc = k * LAM_G_NM - sgn * VP * t_fs * 1e-15 * 1e9
        ok = (zc >= 0) & (zc <= z_nm[-1])
        ax.plot(zc[ok], t_fs[ok], color=PALETTE["ink"], lw=0.9, ls="--")
    ax.set_xlabel("z (nm)"); ax.grid(False)
    ax.set_title(f"{name} travelling; tracked crest velocity\n{vfit:+.4e} m/s = {vfit/C0*REF.neff:+.4f} × c/n_eff", color=col)
axs[0].set_ylabel("t (fs)")
cb = fig.colorbar(im, ax=axs, pad=0.02, fraction=0.04); cb.set_label("E / E₀")
fig.suptitle("With e^{jωt} time dependence, e^{−jβz} moves toward +z and e^{+jβz} toward −z (NOTES 2, 3)", fontsize=11, y=1.04)
fig.savefig(OUT / "sign_convention.png", bbox_inches="tight"); plt.close(fig)

# --------------------------------------------------------------------------- 9. figure: a mode keeps its shape
x_nm = np.linspace(-500, 500, 401)
F, h, g = slab_profile_illustrative(x_nm * 1e-9)
RESULTS["illustrative_profile_h_rad_per_um"] = float(h * 1e-6)
RESULTS["illustrative_profile_gamma_rad_per_um"] = float(g * 1e-6)
RESULTS["illustrative_profile_decay_length_nm"] = float(1e9 / g)
zm = np.linspace(0, 2 * LAM_G_NM, 400)
fig, axs = plt.subplots(1, 4, figsize=(15.5, 3.9), gridspec_kw={"width_ratios": [0.7, 1, 1, 1]},
                        constrained_layout=True)
axs[0].plot(F, x_nm, color=SERIES[0])
axs[0].axhspan(-REF.wg_height_um * 500, REF.wg_height_um * 500, color=PALETTE["line"], alpha=0.6, label="silicon core")
axs[0].set_xlabel("F(x) (arb.)"); axs[0].set_ylabel("x (nm)"); axs[0].legend(fontsize=8, loc="upper left")
axs[0].set_title("Transverse profile F(x)\n(illustrative cos/exp, 1/γ = %.0f nm)" % (1e9 / g), fontsize=10)
for i, tf in enumerate([0.0, T_FS / 4, T_FS / 2]):
    ax = axs[i + 1]
    Exz = F[:, None] * np.cos(OMEGA * tf * 1e-15 - BETA * zm[None, :] * 1e-9)
    im = ax.imshow(Exz, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                   extent=[zm[0], zm[-1], x_nm[0], x_nm[-1]])
    ax.axhline(REF.wg_height_um * 500, color=PALETTE["ink"], lw=0.8, ls=":")
    ax.axhline(-REF.wg_height_um * 500, color=PALETTE["ink"], lw=0.8, ls=":")
    ax.set_xlabel("z (nm)"); ax.set_title(f"E_y(x,z) = F(x) cos(ωt − βz) at t = {tf:.2f} fs", fontsize=10); ax.grid(False)
    if i == 0:
        ax.set_ylabel("x (nm)")
cb = fig.colorbar(im, ax=list(axs[1:]), pad=0.01, fraction=0.03); cb.set_label("E_y / E₀")
fig.suptitle("A guided mode keeps its cross-sectional shape F(x); only the common factor cos(ωt − βz) slides along z (NOTES 2, 14, 21)", fontsize=11)
fig.savefig(OUT / "mode_keeps_shape.png"); plt.close(fig)

# --------------------------------------------------------------------------- 10. capstone: the round trip e^{-j beta L}
L_um = REF.round_trip_um
phi_rt = round_trip_phase(lam0_nm, L_um)
m_rt = phi_rt / (2 * np.pi)
dneff_dT_si_only = REF.confinement * REF.dn_si_dT
dneff_dT = REF.confinement * REF.dn_si_dT + (1 - REF.confinement) * REF.dn_sio2_dT
dphi_dT = W["k0_rad_m"] * (L_um * 1e-6) * dneff_dT                    # rad/K
dlam_dT_pm = lam0_nm * 1e3 * dneff_dT / REF.ng                         # pm/K, (lambda/n_g) dn_eff/dT
dlam_dT_pm_si_only = lam0_nm * 1e3 * dneff_dT_si_only / REF.ng
fsr_phase_index_nm = lam0_nm**2 / (REF.neff * L_um * 1e3)               # spacing if n_eff did not disperse
fsr_group_index_nm = lam0_nm**2 / (REF.ng * L_um * 1e3)                 # the real FSR
K_per_fsr_phase = 2 * np.pi / dphi_dT
K_per_fsr_lambda = REF.fsr_nm * 1e3 / REF.dlambda_dT_pm_per_K
ambient_span_K = REF.ambient_max_c - REF.ambient_min_c
ambient_shift_nm = ambient_span_K * dlam_dT_pm * 1e-3            # from the code's own Gamma-weighted dlambda/dT
ambient_shift_ref_nm = ambient_span_K * REF.dlambda_dT_pm_per_K * 1e-3   # REF x REF, for comparison only

# resonances found numerically from the phase condition beta(lambda) L = 2 pi m
lam_scan = np.linspace(1280, 1340, 600001)
def crossings(phase):
    frac = np.mod(phase / (2 * np.pi), 1.0)
    idx = np.where(np.abs(np.diff(frac)) > 0.5)[0]   # a wrap of the fractional part => beta L crossed 2 pi m
    return lam_scan[idx]
res_disp = crossings(round_trip_phase(lam_scan, L_um, dispersive=True))
res_nodisp = crossings(round_trip_phase(lam_scan, L_um, dispersive=False))
fsr_disp_num = float(np.mean(np.diff(res_disp)))
fsr_nodisp_num = float(np.mean(np.diff(res_nodisp)))

# the resonances that bracket the 1310 nm laser (first-order dispersion model, closed form):
# n_eff(lambda) L = m lambda with n_eff(lambda) = n_g - lambda (n_g - n_eff)/lambda0
#   -> lambda_m = n_g L / (m + L (n_g - n_eff)/lambda0)
def lambda_res_nm(m):
    return REF.ng * (L_um * 1e3) / (m + (L_um * 1e3) * (REF.ng - REF.neff) / lam0_nm)
m_blue = math.ceil(m_rt)                     # next integer order: shorter wavelength (blue side)
m_red = math.floor(m_rt)                     # previous integer order: longer wavelength (red side)
lam_blue, lam_red = lambda_res_nm(m_blue), lambda_res_nm(m_red)
# which one is nearest to the laser
m_near, lam_near = (m_blue, lam_blue) if abs(lam0_nm - lam_blue) <= abs(lam_red - lam0_nm) else (m_red, lam_red)
lam_near_numeric = float(res_disp[np.argmin(np.abs(res_disp - lam_near))])   # same crossing from the scan
turn_to_blue = m_blue - m_rt                 # fraction of a 2pi turn still missing at 1310 nm
turn_to_red = m_rt - m_red
offset_near_nm = lam0_nm - lam_near          # > 0 means the resonance is on the blue side of the laser

RESULTS.update({
    "round_trip_um": L_um,
    "round_trip_phase_rad_at_1310": float(phi_rt),
    "round_trip_phase_in_2pi_units_m": float(m_rt),
    "resonance_spacing_phase_index_only_nm": float(fsr_phase_index_nm),
    "resonance_spacing_phase_index_only_numeric_nm": fsr_nodisp_num,
    "fsr_group_index_nm": float(fsr_group_index_nm),
    "fsr_group_index_numeric_nm": fsr_disp_num,
    "fsr_nm_expected_ref": REF.fsr_nm,
    "fsr_agreement_pct": pct(fsr_disp_num, REF.fsr_nm),
    "nearest_resonance_order_m": int(m_near),
    "nearest_resonance_wavelength_nm": float(lam_near),
    "nearest_resonance_wavelength_numeric_scan_nm": lam_near_numeric,
    "nearest_resonance_wavelength_agreement_pct": pct(lam_near_numeric, lam_near),
    "nearest_resonance_offset_nm": float(offset_near_nm),
    "nearest_resonance_offset_in_fsr": float(offset_near_nm / fsr_disp_num),
    "fractional_turn_to_blue_resonance": float(turn_to_blue),
    "fractional_turn_to_red_resonance": float(turn_to_red),
    "blue_resonance_order_m": int(m_blue),
    "blue_resonance_wavelength_nm": float(lam_blue),
    "red_resonance_order_m": int(m_red),
    "red_resonance_wavelength_nm": float(lam_red),
    "dneff_dT_per_K_gamma_weighted_si_plus_sio2": float(dneff_dT),
    "dneff_dT_per_K_gamma_weighted_si_only": float(dneff_dT_si_only),
    "round_trip_phase_drift_rad_per_K": float(dphi_dT),
    "round_trip_phase_drift_deg_per_K": float(np.degrees(dphi_dT)),
    "dlambda_dT_pm_per_K": float(dlam_dT_pm),
    "dlambda_dT_pm_per_K_si_only": float(dlam_dT_pm_si_only),
    "dlambda_dT_pm_per_K_expected_ref": REF.dlambda_dT_pm_per_K,
    "dlambda_dT_agreement_pct": pct(dlam_dT_pm, REF.dlambda_dT_pm_per_K),
    "kelvin_per_fsr_from_phase_drift": float(K_per_fsr_phase),
    "kelvin_per_fsr_from_ref_numbers": float(K_per_fsr_lambda),
    "ambient_range_shift_nm": float(ambient_shift_nm),
    "ambient_range_shift_in_fsr": float(ambient_shift_nm / REF.fsr_nm),
    "ambient_range_shift_in_fwhm": float(ambient_shift_nm * 1e3 / REF.fwhm_pm),
    "ambient_range_shift_nm_from_ref_numbers": float(ambient_shift_ref_nm),
    "ambient_range_shift_agreement_pct": pct(ambient_shift_nm, ambient_shift_ref_nm),
})
log(f"round trip L = {L_um} um: beta L = {phi_rt:.2f} rad = {m_rt:.3f} x 2pi  (m is the mode order)")
log(f"resonance spacing with n_eff frozen: {fsr_phase_index_nm:.2f} nm (numeric {fsr_nodisp_num:.2f}); with n_g: {fsr_group_index_nm:.2f} nm (numeric {fsr_disp_num:.2f}) vs REF FSR {REF.fsr_nm} nm ({pct(fsr_disp_num, REF.fsr_nm):.1f} %)")
log(f"resonances bracketing 1310 nm (n_eff(lambda) model): m = {m_blue} at {lam_blue:.2f} nm ({turn_to_blue:.4f} turn to the blue), "
    f"m = {m_red} at {lam_red:.2f} nm ({turn_to_red:.4f} turn to the red); nearest is m = {m_near} at {lam_near:.2f} nm "
    f"(scan: {lam_near_numeric:.2f} nm, {pct(lam_near_numeric, lam_near):.3f} %), {offset_near_nm:+.2f} nm = {offset_near_nm/fsr_disp_num:.3f} FSR from the laser")
log(f"dn_eff/dT = {dneff_dT:.3e}/K -> round-trip phase drift {dphi_dT:.4f} rad/K = {np.degrees(dphi_dT):.2f} deg/K")
log(f"dlambda_r/dT = (lambda/n_g) dn_eff/dT = {dlam_dT_pm:.1f} pm/K (Si-only {dlam_dT_pm_si_only:.1f}) vs REF {REF.dlambda_dT_pm_per_K} ({pct(dlam_dT_pm, REF.dlambda_dT_pm_per_K):.1f} %)")
log(f"one FSR of phase (2pi) = {K_per_fsr_phase:.0f} K; FSR/(50 pm/K) = {K_per_fsr_lambda:.0f} K; ambient 10..125 C moves the resonance {ambient_shift_nm:.2f} nm = {ambient_shift_nm/REF.fsr_nm:.2f} FSR (REF x REF: {ambient_shift_ref_nm:.2f} nm, {pct(ambient_shift_nm, ambient_shift_ref_nm):.1f} %)")

fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
ax = axs[0]
ax.plot(np.cos(th), np.sin(th), color=PALETTE["line"], lw=1)
ax.axhline(0, color=PALETTE["muted"], lw=0.8); ax.axvline(0, color=PALETTE["muted"], lw=0.8)
dTs = [0, 5, 10, 20, 40]
for i, dT in enumerate(dTs):
    p = np.exp(-1j * (phi_rt + dphi_dT * dT))
    ax.add_patch(FancyArrowPatch((0, 0), (p.real, p.imag), arrowstyle="-|>", mutation_scale=14,
                                 color=cmap(i / len(dTs)), lw=1.8))
    # anchor each label just outside the circle and let the text extend radially OUTWARD
    # (away from the neighbouring arrow shafts), so close arrows never get overlapping labels
    ha = "right" if p.real < -0.05 else ("left" if p.real > 0.05 else "center")
    va = "bottom" if p.imag > 0.05 else ("top" if p.imag < -0.05 else "center")
    ax.text(1.07 * p.real, 1.07 * p.imag, f"+{dT} K", fontsize=8, ha=ha, va=va, color=cmap(i / len(dTs)))
ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_aspect("equal")
ax.set_xlabel("Re e^{−jβL}  (dimensionless)"); ax.set_ylabel("Im e^{−jβL}  (dimensionless)")
ax.set_title(f"Round-trip factor e^{{−jβL}} of the ring (L = {L_um} µm) at 1310 nm\nrotates clockwise by {np.degrees(dphi_dT):.2f}° per kelvin of ring temperature", fontsize=10)
ax = axs[1]
lam_plot = np.linspace(1295, 1325, 3001)
for dsp, col, lab in [(False, SERIES[1], f"n_eff frozen at {REF.neff}: spacing {fsr_nodisp_num:.1f} nm"),
                      (True, SERIES[0], f"n_eff(λ) with n_g = {REF.ng}: spacing {fsr_disp_num:.1f} nm = FSR")]:
    frac = np.mod(round_trip_phase(lam_plot, L_um, dispersive=dsp) / (2 * np.pi), 1.0)
    frac[np.abs(np.diff(frac, prepend=frac[0])) > 0.5] = np.nan
    ax.plot(lam_plot, frac, color=col, label=lab)
# the laser and the nearest resonance of the dispersive model (m = m_near), computed above
ax.axvline(lam0_nm, color=PALETTE["ink"], lw=0.8, ls=":", label=f"laser 1310 nm: βL = {m_rt:.3f} × 2π")
ax.plot(lam_near, 0.0, "o", color=SERIES[0], ms=6, zorder=5,
        label=f"nearest resonance m = {m_near} at {lam_near:.1f} nm ({offset_near_nm/fsr_disp_num:.2f} FSR away)")
ax.set_xlabel("λ (nm)"); ax.set_ylabel("βL / 2π  (fractional part)")
ax.set_title("Resonance = βL is a whole number of 2π. The phase uses n_eff,\nbut the SPACING of resonances uses n_g (slope of β vs ω)", fontsize=10)
ax.set_ylim(0, 1.03)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False)   # below the axes, never on the data
ax = axs[2]
Tc = np.linspace(REF.ambient_min_c, REF.ambient_max_c, 200)
ax.plot(Tc, (Tc - 25) * dlam_dT_pm * 1e-3, color=SERIES[0], label=f"Δλ_r = (λ/n_g)(dn_eff/dT) ΔT = {dlam_dT_pm:.1f} pm/K")
ax.axhline(REF.fsr_nm, color=SERIES[1], ls="--", label=f"one FSR = {REF.fsr_nm} nm")
ax.axhline(REF.fwhm_pm * 1e-3, color=SERIES[4], ls=":", label=f"one FWHM = {REF.fwhm_pm:.0f} pm")
ax.set_xlabel("ring temperature (°C)"); ax.set_ylabel("resonance shift from 25 °C (nm)")
ax.set_title(f"Across the {REF.ambient_min_c:.0f}–{REF.ambient_max_c:.0f} °C ambient range the resonance walks\n{ambient_shift_nm:.2f} nm = {ambient_shift_nm/REF.fsr_nm:.2f} FSR = {ambient_shift_nm*1e3/REF.fwhm_pm:.0f} linewidths", fontsize=10)
ax.set_ylim(-1.5, 15.0)                       # headroom above the FSR line for the legend
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "capstone_round_trip.png"); plt.close(fig)

# --------------------------------------------------------------------------- 11. Manim videos + contact sheets
MANIM = VENV / "manim"
media = HERE / "media"
scenes = [("TravellingWaveScene", "travelling_wave"), ("SignConventionScene", "sign_convention")]
video_info = {}
for scene_name, base in scenes:
    t0 = time.time()
    cmd = [str(MANIM), "-qm", "--disable_caching", "--media_dir", str(media), "-o", f"{base}.mp4",
           "scene.py", scene_name]
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    (OUT / f"manim_{base}.log").write_text(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        log(r.stdout[-3000:]); log(r.stderr[-3000:])
        raise SystemExit(f"manim failed for {scene_name}")
    src = next(media.glob(f"videos/scene/*/{base}.mp4"))
    dst = OUT / f"{base}.mp4"
    shutil.copy(src, dst)
    dur = float(subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of",
                                "csv=p=0", str(dst)], capture_output=True, text=True).stdout.strip())
    video_info[base] = {"duration_s": dur, "render_s": time.time() - t0}
    log(f"rendered {dst.name}: {dur:.1f} s of video in {time.time()-t0:.0f} s")
    # contact sheet: 6 evenly spaced frames
    rd = imageio.get_reader(str(dst), "ffmpeg")
    nfr = rd.count_frames()
    picks = np.linspace(0.04, 0.96, 6) * (nfr - 1)
    fig, axs = plt.subplots(2, 3, figsize=(15, 5.9))
    for axx, fi in zip(axs.ravel(), picks):
        fr = rd.get_data(int(fi))
        axx.imshow(fr); axx.set_axis_off()
        axx.set_title(f"t_video = {fi/nfr*dur:.1f} s", fontsize=9)
    rd.close()
    fig.suptitle(f"{dst.name}: six stills", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / f"{base}_frames.png"); plt.close(fig)
shutil.rmtree(media, ignore_errors=True)
RESULTS["videos"] = video_info

# --------------------------------------------------------------------------- 12. notebook
from make_notebook import build_notebook
nb_path = HERE / "explore.ipynb"
build_notebook(nb_path)
t0 = time.time()
# Every cell of the notebook runs in well under 1 s (the whole file in ~3 s), so the per-cell
# timeout is set low. Reason: on this machine the headless executor INTERMITTENTLY sits
# idle for exactly one cell timeout right after the ipywidgets `interact` cell (the kernel
# has already replied; the client only notices at the timeout, then carries on and exits 0),
# so the timeout is the price of a stall, not a limit any cell needs. Seen twice so far.
NB_TIMEOUT_S = 60
cmd = [str(VENV / "jupyter"), "nbconvert", "--to", "notebook", "--execute", "--inplace",
       "--ExecutePreprocessor.kernel_name=photonics-sims", f"--ExecutePreprocessor.timeout={NB_TIMEOUT_S}", str(nb_path)]
log("$ " + " ".join(cmd))
r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
(OUT / "nbconvert.log").write_text(r.stdout + "\n" + r.stderr)
if r.returncode != 0:
    # If a cell genuinely times out (or nbconvert fails for any other reason), fall back to
    # executing in-process with nbclient (the same executor nbconvert uses) so run.py completes.
    log("nbconvert failed (see out/nbconvert.log); retrying in-process with nbclient")
    import nbformat
    from nbclient import NotebookClient
    build_notebook(nb_path)
    nb = nbformat.read(nb_path, as_version=4)
    NotebookClient(nb, timeout=2 * NB_TIMEOUT_S, kernel_name="photonics-sims",
                   resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, nb_path)
nb_exec_s = time.time() - t0
RESULTS["notebook_execution_s"] = nb_exec_s
RESULTS["notebook_cell_timeout_s"] = NB_TIMEOUT_S
RESULTS["notebook_execution_stalled"] = bool(nb_exec_s > 0.8 * NB_TIMEOUT_S)
log(f"executed explore.ipynb in {nb_exec_s:.0f} s" + ("  (stalled for one cell timeout; see README Checks)" if RESULTS["notebook_execution_stalled"] else ""))

# --------------------------------------------------------------------------- 13. tools.json + results.json
import manim, nbformat, nbclient, ipywidgets, jupyter_core
ffv = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True).stdout.split("\n")[0].split()[2]
TOOLS = [
    {"tool": "Manim Community", "version": manim.__version__,
     "what_it_is": "A programmatic animation engine (the 3Blue1Brown tool, community edition): scenes are Python classes, objects are vector graphics rendered frame by frame with Cairo and encoded with ffmpeg. Normally used for explanatory maths videos.",
     "used_for": "Two videos. travelling_wave.mp4: a live snapshot E(z) with one crest tracked at v_p, the phasor at z0 rotating in the complex plane with its real-part projection, and the time trace E(z0,t) being drawn (NOTES 2, 3). sign_convention.mp4: cos(wt - bz) and cos(wt + bz) side by side with tracked crests moving in opposite directions. All labels are Text() with Unicode because there is no LaTeX on this machine; Axes use label_constructor=Text for the same reason.",
     "result": f"travelling_wave.mp4 ({video_info['travelling_wave']['duration_s']:.1f} s, 1280x720 30 fps) and sign_convention.mp4 ({video_info['sign_convention']['duration_s']:.1f} s). The crest dot advances one guided wavelength (524 nm) per period (4.37 fs), i.e. v_p = 1.199e8 m/s = c/2.5; the aqua lambda_g bracket is anchored to the tracked crest so it spans crest to crest at every instant; the phasor turns once per 4.37 fs and its projection traces the violet time trace.",
     "how_to_observe": "cd experiments/01_travelling_wave && ../../.venv/bin/python run.py, then open out/travelling_wave.mp4 and out/sign_convention.mp4 (stills: out/*_frames.png). Watch the orange crest dot, the violet phasor arrow and the violet trace grow together. To re-render one scene alone: ../../.venv/bin/manim -qm --disable_caching scene.py TravellingWaveScene (output under media/). Change n_eff or lambda in common/params.py, or Z_MAX_NM / z0_nm in scene.py, and re-run."},
    {"tool": "numpy", "version": np.__version__,
     "what_it_is": "The array library for numerical Python: vectorised arithmetic, complex numbers, linear least squares. Used everywhere numbers are crunched.",
     "used_for": "Evaluating the one formula E = E0 cos(wt - bz) on (z,t) grids, the complex phasor E0 exp(-jbz) exp(jwt) and its real part, the crest-tracking loop (nearest local maximum per time step with parabolic sub-grid refinement), np.polyfit for the crest velocity, and the round-trip phase bL of the ring (twave.py).",
     "result": f"Phasor identity max error {err:.1e}; tracked crest velocity {v_fwd:.5e} m/s vs w/b = {VP:.5e} m/s ({pct(v_fwd, VP):.4f} %); backward wave {v_bwd:.5e} m/s; round trip bL = {m_rt:.2f} x 2pi, nearest resonance m = {m_near} at {lam_near:.1f} nm ({turn_to_blue:.2f} turn = {offset_near_nm/fsr_disp_num:.2f} FSR to the blue); resonance spacing {fsr_disp_num:.2f} nm vs REF FSR {REF.fsr_nm} nm ({pct(fsr_disp_num, REF.fsr_nm):.1f} %); dlambda/dT = {dlam_dT_pm:.1f} pm/K vs REF 50.",
     "how_to_observe": "All numbers are printed by run.py and written to out/results.json and out/run_log.txt. The functions live in twave.py (wave_numbers, E_real, E_phasor, track_crest, fit_velocity, round_trip_phase); import them in the notebook or a REPL and change n, sign, lambda_nm."},
    {"tool": "scipy", "version": scipy.__version__,
     "what_it_is": "Scientific algorithms on top of numpy (signal processing, optimisation, integration).",
     "used_for": "scipy.signal.find_peaks to locate the crests of a snapshot E(z) at fixed t and measure the guided wavelength from their mean spacing, independently of the formula lambda0/n_eff.",
     "result": f"crest spacing {lam_meas:.2f} nm vs lambda0/n_eff = {LAM_G_NM:.2f} nm ({pct(lam_meas, LAM_G_NM):.3f} %).",
     "how_to_observe": "out/results.json keys lambda_g_measured_from_snapshot_nm; the snapshot itself is the middle panel of out/three_ways.png. Change REF.neff and the spacing changes as lambda0/n_eff."},
    {"tool": "matplotlib", "version": matplotlib.__version__,
     "what_it_is": "The standard Python plotting library (static figures, 3-D axes, image maps, animations via ffmpeg).",
     "used_for": "All PNG figures: the three-ways panel (fixed z, fixed t, z-t map with crest lines), crest-tracking fit, three wavelengths, the phasor helix / clock / projection panel, the sign-convention z-t maps, the mode-keeps-its-shape maps, the capstone round-trip panel, and the contact sheets built from video frames.",
     "result": "out/three_ways.png, crest_tracking.png, three_wavelengths.png, phasor.png, sign_convention.png, mode_keeps_shape.png, capstone_round_trip.png, travelling_wave_frames.png, sign_convention_frames.png.",
     "how_to_observe": "Open the PNGs in out/. Signed fields use RdBu_r centred on zero. Edit the parameter lists near each figure block in run.py (e.g. the probe positions z = 0, lambda_g/4, lambda_g/2) and re-run."},
    {"tool": "ffmpeg / ffprobe (via imageio-ffmpeg)", "version": ffv,
     "what_it_is": "The standard command-line video encoder/decoder; Manim uses it to encode frames to H.264, imageio wraps it to read frames back into numpy.",
     "used_for": "Encoding the Manim videos, measuring their duration (ffprobe), and reading six evenly spaced frames from each mp4 for the *_frames.png contact sheets.",
     "result": f"Durations: travelling_wave {video_info['travelling_wave']['duration_s']:.1f} s, sign_convention {video_info['sign_convention']['duration_s']:.1f} s; two contact sheets.",
     "how_to_observe": "/opt/homebrew/bin/ffprobe out/travelling_wave.mp4; the contact-sheet code is the loop at the end of the Manim section in run.py (change the number of picks)."},
    {"tool": "Jupyter (nbformat / nbclient / ipywidgets)", "version": f"nbformat {nbformat.__version__}, nbclient {nbclient.__version__}, ipywidgets {ipywidgets.__version__}",
     "what_it_is": "The notebook format and executor; ipywidgets adds sliders that re-run a Python function on every change.",
     "used_for": f"explore.ipynb is generated by make_notebook.py with nbformat and executed headlessly by nbconvert (kernel photonics-sims, per-cell timeout {NB_TIMEOUT_S} s because the executor intermittently stalls for one timeout after the widget cell; nbclient in-process fallback if nbconvert fails). It has sliders for the index n, the propagation sign and the time t, plus static cells that recompute v_p by crest tracking and the ring round-trip phasor for a chosen temperature.",
     "result": f"explore.ipynb with saved outputs (figures for three parameter sets, tracked v_p, round-trip phase); executed in {nb_exec_s:.0f} s this run" + (" (one cell-timeout stall included)." if RESULTS["notebook_execution_stalled"] else " (no stall this run)."),
     "how_to_observe": "../../.venv/bin/jupyter lab explore.ipynb, run all, then drag the n / sign / t sliders. Rebuild+re-execute headlessly with run.py or ../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace explore.ipynb."},
]
(OUT / "tools.json").write_text(json.dumps(TOOLS, indent=2))
RESULTS["runtime_s"] = time.time() - T_START
RESULTS["interpreter"] = sys.executable
RESULTS["platform"] = platform.platform()
(OUT / "results.json").write_text(json.dumps(RESULTS, indent=2))
log(f"done in {RESULTS['runtime_s']:.0f} s")
(OUT / "run_log.txt").write_text("\n".join(LOG) + "\n")
shutil.rmtree(HERE / "__pycache__", ignore_errors=True)
