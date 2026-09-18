"""Part A of experiment 12: from the round-trip retention factor a to the all-pass ring.

Run with the .venv interpreter (SAX + JAX):  ../../.venv/bin/python ring_sax.py
Everything analytic here follows docs/NOTES.md section 28 (retention factor) and
the ring transfer function
    T(lambda) = | (t - a e^{-j phi}) / (1 - t a e^{-j phi}) |^2 ,  phi = beta L = 2 pi n(lambda) L / lambda
with the e^{j omega t} phasor convention (a +z wave is e^{-j beta z}).

Outputs (all in out/):
  A1_retention.png          field/power retained lap after lap, doped vs passive ring
  A2_spectrum.png           SAX all-pass spectrum: FSR, FWHM, Q, phase; passive vs doped
  A3_coupling_regimes.png   under / critical / over coupling, and the phasor-circle picture
  A4_thermal.png            T(lambda_L) at a fixed laser vs ring temperature (50 pm/K)
  A5_notch_sliding.mp4      animation: the notch slides under the fixed laser as T ramps 10 -> 125 C
  A5_notch_sliding_frames.png  contact sheet
  A_results.json            every number quoted in the README
"""
import sys, pathlib, json, time
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import REF, use_style, SERIES, PALETTE, C0
from common.units import db_per_cm_to_alpha_per_um
OUT = pathlib.Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
use_style()

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
plt.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"
import numpy as np
import sax, jax.numpy as jnp

T0 = time.time()
R = {}                                   # results dictionary -> A_results.json
LAM0_NM = REF.lambda_nm                  # 1310 nm
L_UM = REF.round_trip_um                 # 39.6 um
NG = REF.ng                              # 4.2

# ---------------------------------------------------------------------------
# 1. Retention factor a = exp(-alpha L / 2)   (notes 28; alpha from dB/cm, notes 6)
# ---------------------------------------------------------------------------
def retention(loss_db_cm, L_um):
    alpha = db_per_cm_to_alpha_per_um(loss_db_cm)       # power attenuation, 1/um
    a = np.exp(-alpha * L_um / 2)                        # field retention per lap
    return alpha, a

alpha_d, a_d = retention(REF.loss_db_cm_doped, L_UM)
alpha_p, a_p = retention(REF.loss_db_cm_passive, L_UM)
R.update({
    "alpha_doped_per_um": alpha_d, "a_doped": a_d, "a2_doped": a_d**2,
    "loss_db_per_lap_doped": REF.loss_db_cm_doped * L_UM * 1e-4,
    "alpha_passive_per_um": alpha_p, "a_passive": a_p, "a2_passive": a_p**2,
    "loss_db_per_lap_passive": REF.loss_db_cm_passive * L_UM * 1e-4,
    "a_doped_expected": REF.a_round_trip,
    "a_doped_agreement_pct": 100 * (1 - abs(a_d - REF.a_round_trip) / REF.a_round_trip),
    "n_pp_doped": alpha_d * (LAM0_NM * 1e-3) / (4 * np.pi),   # imaginary index n'' = alpha lambda / 4 pi
})
print(f"[A1] doped   125 dB/cm: alpha = {alpha_d:.4e} /um, a = {a_d:.4f}, a^2 = {a_d**2:.4f}, {R['loss_db_per_lap_doped']:.3f} dB/lap")
print(f"[A1] passive   3 dB/cm: alpha = {alpha_p:.4e} /um, a = {a_p:.5f}, a^2 = {a_p**2:.5f}, {R['loss_db_per_lap_passive']:.4f} dB/lap")

# Figure A1: field amplitude along the unrolled ring for five laps
z = np.linspace(0, 5 * L_UM, 2000)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for (alpha, a, lab, c) in [(alpha_d, a_d, f"doped ring, {REF.loss_db_cm_doped:.0f} dB/cm", SERIES[0]),
                           (alpha_p, a_p, f"passive ring, {REF.loss_db_cm_passive:.0f} dB/cm", SERIES[1])]:
    ax[0].plot(z, np.exp(-alpha * z / 2), color=c, label=f"{lab}: a = {a:.4f} per lap")
    ax[1].plot(z, np.exp(-alpha * z), color=c, label=f"{lab}: a² = {a**2:.4f} per lap")
for k in range(1, 6):
    for A in ax:
        A.axvline(k * L_UM, color=PALETTE["line"], lw=1)
    ax[0].plot(k * L_UM, a_d**k, "o", color=SERIES[0], ms=5)
    ax[1].plot(k * L_UM, a_d**(2 * k), "o", color=SERIES[0], ms=5)
ax[0].set(xlabel="distance travelled inside the ring z (µm)", ylabel="field amplitude |A(z)| / |A(0)|",
          title="Field: each lap multiplies the amplitude by a = e^{−αL/2}")
ax[1].set(xlabel="distance travelled inside the ring z (µm)", ylabel="power P(z) / P(0)",
          title="Power: each lap multiplies the power by a² = e^{−αL}")
for A in ax:
    A.legend(loc="lower left"); A.set_ylim(0.5, 1.04)
    for k in range(1, 6):
        A.text((k - 0.5) * L_UM, 1.02, f"lap {k}", ha="center", va="bottom", color=PALETTE["ink2"], fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "A1_retention.png"); plt.close(fig)

# ---------------------------------------------------------------------------
# 2. SAX all-pass ring (coupler + lossy dispersive waveguide closed on itself)
# ---------------------------------------------------------------------------
# Put a resonance exactly at 1310.00 nm: the phase index is only known to ~1 %, so we
# choose the mode order m = round(neff L / lambda) = 76 and the neff that makes it exact.
M_ORDER = int(round(REF.neff * L_UM / (LAM0_NM * 1e-3)))
NEFF = M_ORDER * (LAM0_NM * 1e-3) / L_UM
R.update({"mode_order_m": M_ORDER, "neff_used": NEFF, "neff_ref": REF.neff})
print(f"[A2] mode order m = {M_ORDER}, n_eff nudged {REF.neff} -> {NEFF:.4f} so that a resonance sits at {LAM0_NM:.0f} nm")

def coupler(coupling=REF.kappa2):
    k = coupling ** 0.5; t = (1 - coupling) ** 0.5
    return sax.reciprocal({("in0", "out0"): t, ("in0", "out1"): -1j * k,
                           ("in1", "out0"): -1j * k, ("in1", "out1"): t})

def waveguide(wl=1.31, length=L_UM, neff=NEFF, ng=NG, wl0=LAM0_NM * 1e-3, loss_db_cm=REF.loss_db_cm_doped):
    n = neff - (wl - wl0) * (ng - neff) / wl0                 # first-order dispersion: gives group index ng
    amp = 10 ** (-loss_db_cm * length * 1e-4 / 20)            # field retention a over this length
    return sax.reciprocal({("in0", "out0"): amp * jnp.exp(-2j * jnp.pi * n * length / wl)})

ring, _ = sax.circuit(
    netlist={"instances": {"c": "coupler", "r": "waveguide"},
             "connections": {"c,out1": "r,in0", "r,out0": "c,in1"},
             "ports": {"in": "c,in0", "out": "c,out0"}},
    models={"coupler": coupler, "waveguide": waveguide})

def ring_T(wl_um, coupling=REF.kappa2, loss_db_cm=REF.loss_db_cm_doped, ng=NG, neff=NEFF):
    S = ring(wl=jnp.asarray(wl_um), c={"coupling": coupling},
             r={"loss_db_cm": loss_db_cm, "ng": ng, "neff": neff})
    s = np.asarray(S["in", "out"])
    return np.abs(s) ** 2, np.unwrap(np.angle(s))

def analytic_T(wl_um, t, a, ng=NG, neff=NEFF, L=L_UM, wl0=LAM0_NM * 1e-3):
    n = neff - (wl_um - wl0) * (ng - neff) / wl0
    phi = 2 * np.pi * n * L / wl_um
    s = (t - a * np.exp(-1j * phi)) / (1 - t * a * np.exp(-1j * phi))
    return np.abs(s) ** 2, np.unwrap(np.angle(s))

def analytic_numbers(t, a, ng=NG, L=L_UM, lam_nm=LAM0_NM):
    """Textbook formulas (Bogaerts et al. 2012): FSR, FWHM, Q, T_min, finesse, build-up factor."""
    fsr_nm = lam_nm ** 2 / (ng * L * 1e3)                                  # lambda^2 / (ng L)
    fsr_thz = C0 / (ng * L * 1e-6) / 1e12                                  # c / (ng L)
    fwhm_pm = (1 - t * a) * lam_nm ** 2 / (np.pi * ng * L * 1e3 * np.sqrt(t * a)) * 1e3
    q = lam_nm * 1e3 / fwhm_pm
    t_min = ((t - a) / (1 - t * a)) ** 2
    build_up = (1 - t ** 2) / (1 - t * a) ** 2                             # |E_ring|^2 / |E_in|^2 on resonance
    return dict(fsr_nm=fsr_nm, fsr_thz=fsr_thz, fwhm_pm=fwhm_pm, q=q, t_min=t_min,
                finesse=fsr_nm * 1e3 / fwhm_pm, build_up=build_up)

t_c = REF.t_coupler                                       # 0.945 = a  -> critical coupling
kappa2 = 1 - t_c ** 2
an = analytic_numbers(t_c, a_d)
R.update({"kappa2_from_t": kappa2, "t_coupler": t_c, **{f"{k}_analytic": v for k, v in an.items()}})

# fine sweep over three FSRs (1 pm step)
wl_nm = np.arange(1290.0, 1332.0, 0.001)
T_sax, ph_sax = ring_T(wl_nm * 1e-3, coupling=kappa2)
T_an, ph_an = analytic_T(wl_nm * 1e-3, t_c, a_d)
R["sax_vs_analytic_max_abs_diff"] = float(np.max(np.abs(T_sax - T_an)))
print(f"[A2] SAX vs closed-form |T| max abs difference: {R['sax_vs_analytic_max_abs_diff']:.2e}")

def dips(wl, T):
    """Local minima of T(lambda) (resonances) by comparing with neighbours."""
    i = np.where((T[1:-1] < T[:-2]) & (T[1:-1] <= T[2:]))[0] + 1
    return wl[i], T[i]

def fwhm_of_dip(wl, T, lam_res, T_min):
    half = (1 + T_min) / 2          # half depth of the notch
    i0 = np.argmin(np.abs(wl - lam_res))
    lo = i0
    while lo > 0 and T[lo] < half: lo -= 1
    hi = i0
    while hi < len(T) - 1 and T[hi] < half: hi += 1
    return wl[hi] - wl[lo]

lam_res, T_res = dips(wl_nm, T_sax)
i_c = np.argmin(np.abs(lam_res - LAM0_NM))
fsr_meas = np.diff(lam_res)
# the FSR grows with lambda^2, so quote the LOCAL FSR at 1310 nm: mean of the two spacings next to that dip
fsr_local = float(np.mean(fsr_meas[max(i_c - 1, 0):i_c + 1]))
R.update({
    "resonances_nm_sax": lam_res.tolist(), "t_min_sax": float(T_res[i_c]),
    "fsr_nm_sax": fsr_local, "fsr_spacings_nm_sax": fsr_meas.tolist(),
    "fsr_thz_sax": float(C0 * fsr_local * 1e-9 / (LAM0_NM * 1e-9) ** 2 / 1e12),
    "fsr_nm_ref": REF.fsr_nm, "fsr_thz_ref": REF.fsr_thz,
    "fwhm_pm_sax": float(fwhm_of_dip(wl_nm, T_sax, lam_res[i_c], T_res[i_c]) * 1e3),
    "fwhm_pm_ref": REF.fwhm_pm, "q_ref": REF.q_loaded,
})
R["q_sax"] = lam_res[i_c] * 1e3 / R["fwhm_pm_sax"]
f_opt_hz = C0 / (LAM0_NM * 1e-9)
R.update({"tau_amp_ps": R["q_sax"] * LAM0_NM * 1e-9 / (np.pi * C0) * 1e12,          # amplitude lifetime tau = Q lambda/(pi c) = 2Q/omega
          "tau_energy_ps": R["q_sax"] * LAM0_NM * 1e-9 / (2 * np.pi * C0) * 1e12,   # energy lifetime tau_E = tau/2 = Q/omega
          "linewidth_ghz_f_over_Q": f_opt_hz / R["q_sax"] / 1e9,                      # FWHM in frequency = f/Q = 1/(2 pi tau_E)
          "bandwidth_ghz_f_over_2Q": f_opt_hz / (2 * R["q_sax"]) / 1e9})              # 1/(2 pi tau) = f/(2Q): half the linewidth
print(f"[A2] time-domain Q: tau_amp = Q lambda/(pi c) = {R['tau_amp_ps']:.2f} ps, tau_E = tau/2 = {R['tau_energy_ps']:.2f} ps; linewidth f/Q = 1/(2 pi tau_E) = {R['linewidth_ghz_f_over_Q']:.1f} GHz, f/(2Q) = 1/(2 pi tau) = {R['bandwidth_ghz_f_over_2Q']:.1f} GHz")
R["finesse_sax"] = R["fsr_nm_sax"] * 1e3 / R["fwhm_pm_sax"]
print(f"[A2] resonances (nm): {np.round(lam_res, 3)}")
print(f"[A2] FSR spacings (nm): {np.round(fsr_meas, 3)} (grows with λ²)")
print(f"[A2] FSR   SAX (local at 1310) {R['fsr_nm_sax']:.3f} nm = {R['fsr_thz_sax']:.3f} THz | analytic {an['fsr_nm']:.3f} nm / {an['fsr_thz']:.3f} THz | ref {REF.fsr_nm} nm / {REF.fsr_thz} THz")
print(f"[A2] FWHM  SAX {R['fwhm_pm_sax']:.1f} pm | analytic {an['fwhm_pm']:.1f} pm | ref {REF.fwhm_pm} pm")
print(f"[A2] Q     SAX {R['q_sax']:.0f} | analytic {an['q']:.0f} | ref {REF.q_loaded:.0f};  T_min SAX {R['t_min_sax']:.2e} (critical coupling: 0)")
print(f"[A2] finesse {R['finesse_sax']:.1f}; intracavity build-up |E_ring/E_in|^2 = {an['build_up']:.2f} on resonance")

# the passive ring with the same coupler: heavily over-coupled, shallow notch
an_p_same = analytic_numbers(t_c, a_p)
t_p_crit = a_p
an_p_crit = analytic_numbers(t_p_crit, a_p)
R.update({"passive_same_coupler": an_p_same, "passive_critical": {**an_p_crit, "t": t_p_crit, "kappa2": 1 - t_p_crit ** 2}})
T_pass_same, _ = ring_T(wl_nm * 1e-3, coupling=kappa2, loss_db_cm=REF.loss_db_cm_passive)
T_pass_crit, _ = ring_T(wl_nm * 1e-3, coupling=1 - t_p_crit ** 2, loss_db_cm=REF.loss_db_cm_passive)
print(f"[A2] passive ring, same coupler (t=0.945, a={a_p:.4f}): T_min = {an_p_same['t_min']:.3f} (over-coupled, shallow notch), FWHM {an_p_same['fwhm_pm']:.0f} pm")
print(f"[A2] passive ring at ITS critical coupling t = a = {a_p:.4f}: kappa^2 = {1-t_p_crit**2:.4f}, FWHM {an_p_crit['fwhm_pm']:.1f} pm, Q = {an_p_crit['q']:.0f}")

# Figure A2
fig = plt.figure(figsize=(12, 7.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
ax0 = fig.add_subplot(gs[0, :]); ax1 = fig.add_subplot(gs[1, 0]); ax2 = fig.add_subplot(gs[1, 1])
ax0.plot(wl_nm, T_sax, color=SERIES[0], label=f"doped ring (a = {a_d:.3f}), t = {t_c} = a: critical coupling")
ax0.plot(wl_nm, T_pass_same, color=SERIES[1], label=f"passive ring (a = {a_p:.4f}), same coupler t = {t_c}: over-coupled, T_min = {an_p_same['t_min']:.2f}")
ax0.plot(wl_nm, T_pass_crit, color=SERIES[2], lw=1.2, label=f"passive ring at its own critical coupling t = {t_p_crit:.4f} (Q = {an_p_crit['q']:.0f})")
for k in range(len(lam_res) - 1):
    ax0.annotate("", xy=(lam_res[k + 1], 1.06), xytext=(lam_res[k], 1.06),
                 arrowprops=dict(arrowstyle="<->", color=PALETTE["ink2"]))
    ax0.text((lam_res[k] + lam_res[k + 1]) / 2, 1.08, f"FSR = {fsr_meas[k]:.2f} nm", ha="center", fontsize=9)
ax0.set(xlabel="laser wavelength λ (nm)", ylabel="through-port power transmission T", ylim=(-0.03, 1.2),
        title=f"All-pass ring, L = {L_UM} µm, n_g = {NG}: a notch every FSR = λ²/(n_g L) = {an['fsr_nm']:.2f} nm\n(the passive ring with the same coupler barely notches; it needs its own, much weaker, coupler)")
ax0.legend(loc="lower left", fontsize=8.5)
sel = np.abs(wl_nm - LAM0_NM) < 1.5
ax1.plot(wl_nm[sel] - LAM0_NM, T_sax[sel], color=SERIES[0], label="SAX circuit")
ax1.plot(wl_nm[sel] - LAM0_NM, T_an[sel], "--", color=SERIES[4], lw=1.2, label="closed form |(t − a e^{−jφ})/(1 − t a e^{−jφ})|²")
half = (1 + R["t_min_sax"]) / 2
ax1.axhline(half, color=PALETTE["muted"], lw=1, ls=":")
ax1.annotate("", xy=(R["fwhm_pm_sax"] / 2e3, half), xytext=(-R["fwhm_pm_sax"] / 2e3, half), arrowprops=dict(arrowstyle="<->", color=SERIES[3]))
ax1.text(-0.25, half + 0.06, f"FWHM = {R['fwhm_pm_sax']:.0f} pm\nQ = λ/FWHM = {R['q_sax']:.0f}", ha="center", color=SERIES[3], fontsize=9)
ax1.axvline(REF.delta_opt_pm / 1e3, color=SERIES[1], lw=1.2, ls="--")
ax1.text(REF.delta_opt_pm / 1e3 + 0.05, 0.15, f"δ_opt = +{REF.delta_opt_pm:.0f} pm\n(laser bias, max OMA)", color=SERIES[1], fontsize=8.5)
ax1.set(xlabel="detuning δ = λ_L − λ_r (pm → shown in nm)", ylabel="T", xlim=(-1.5, 1.5),
        title="Zoom on the 1310 nm notch: FWHM and the laser bias point")
ax1.legend(loc="upper right", fontsize=8)
ax2.plot(wl_nm[sel] - LAM0_NM, np.degrees(ph_sax[sel] - ph_sax[sel][0]), color=SERIES[0], label="doped, critical (t = a)")
_, ph_p = ring_T(wl_nm * 1e-3, coupling=kappa2, loss_db_cm=REF.loss_db_cm_passive)
ax2.plot(wl_nm[sel] - LAM0_NM, np.degrees(ph_p[sel] - ph_p[sel][0]), color=SERIES[1], label="passive, over-coupled (t < a)")
ax2.set(xlabel="detuning δ (nm)", ylabel="through-port phase (degrees)", xlim=(-1.5, 1.5),
        title="Through-port phase: over-coupled = all-pass phase shifter (360°)")
ax2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "A2_spectrum.png"); plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Coupling regimes and the phasor-circle picture
# ---------------------------------------------------------------------------
delta_pm = np.linspace(-1500, 1500, 3001)
wl_reg = (LAM0_NM + delta_pm * 1e-3) * 1e-3
regimes = [("under-coupled  t = 0.980 > a", 0.980, SERIES[2]),
           ("critical        t = 0.945 = a", 0.945, SERIES[0]),
           ("over-coupled   t = 0.900 < a", 0.900, SERIES[1])]
R["regimes"] = {}
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for lab, t, c in regimes:
    T, ph = analytic_T(wl_reg, t, a_d)
    nums = analytic_numbers(t, a_d)
    R["regimes"][lab.split()[0]] = {"t": t, **nums}
    ax[0].plot(delta_pm, T, color=c, label=f"{lab}: T_min = {nums['t_min']:.3f}, FWHM {nums['fwhm_pm']:.0f} pm")
    ax[1].plot(delta_pm, np.degrees(ph - ph[0]), color=c, label=lab.split()[0])
    phi = np.linspace(0, 2 * np.pi, 400)
    num = t - a_d * np.exp(-1j * phi)
    ax[2].plot(num.real, num.imag, color=c, label=f"{lab.split()[0]}: circle centre t = {t}, radius a = {a_d:.3f}")
    ax[2].plot(t, 0, "o", color=c, ms=4)
ax[2].plot(0, 0, "k+", ms=12, mew=2); ax[2].annotate("origin: T = 0", xy=(0, 0), xytext=(0.12, -0.25), fontsize=8, arrowprops=dict(arrowstyle="->", color=PALETTE["ink2"], lw=0.8))
ax[2].set_aspect("equal")
ax[0].set(xlabel="detuning δ (pm)", ylabel="T", title="Same loss a, three couplers:\nonly t = a gives T_min = 0")
ax[0].legend(fontsize=7.5, loc="lower left")
ax[1].set(xlabel="detuning δ (pm)", ylabel="through phase (deg)", title="Phase: over-coupled wraps 360°,\nunder-coupled returns to 0")
ax[1].legend(fontsize=8)
ax[2].set(xlabel="Re(t − a e^{−jφ})", ylabel="Im(t − a e^{−jφ})", title="Numerator t − a e^{−jφ}: a circle of radius a\ncentred at t, through the origin only if t = a")
ax[2].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "A3_coupling_regimes.png"); plt.close(fig)

# Which coupler gives the modulator's 18 dB extinction (T_min = 0.016) with a = 0.945?
r = np.sqrt(REF.t_min)
t_under = (a_d + r) / (1 + r * a_d); t_over = (a_d - r) / (1 - r * a_d)
R.update({"t_for_tmin_0p016_under": t_under, "t_for_tmin_0p016_over": t_over,
          "kappa2_for_tmin_0p016_under": 1 - t_under ** 2, "kappa2_for_tmin_0p016_over": 1 - t_over ** 2})
print(f"[A3] T_min = {REF.t_min} (18 dB) needs t = {t_under:.4f} (under, κ² = {1-t_under**2:.3f}) or t = {t_over:.4f} (over, κ² = {1-t_over**2:.3f}) with a = {a_d:.3f}")

# ---------------------------------------------------------------------------
# 4. Thermal shift at a fixed laser: T(lambda_L) vs ring temperature (50 pm/K)
# ---------------------------------------------------------------------------
dldT = REF.dlambda_dT_pm_per_K                    # pm/K
T_REF_C = 25.0                                    # temperature at which lambda_r = 1310.000 nm
lam_L = LAM0_NM + REF.delta_opt_pm * 1e-3         # laser parked at delta_opt on the red side at 25 C
K_per_fsr = R["fsr_nm_sax"] * 1e3 / dldT
R.update({"K_per_fsr": K_per_fsr, "K_per_fsr_ref": REF.fsr_nm * 1e3 / dldT,
          "pm_per_0p1K": 0.1 * dldT, "fraction_fwhm_per_0p1K": 0.1 * dldT / R["fwhm_pm_sax"],
          "ambient_span_K": REF.ambient_max_c - REF.ambient_min_c,
          "ambient_span_nm": (REF.ambient_max_c - REF.ambient_min_c) * dldT * 1e-3,
          "ambient_span_fsr": (REF.ambient_max_c - REF.ambient_min_c) * dldT * 1e-3 / R["fsr_nm_sax"],
          "heater_mw_per_fsr": R["fsr_nm_sax"] / REF.heater_nm_per_mw,
          "heater_nm_per_mw_check": dldT * REF.r_th_K_per_mw * 1e-3, "heater_nm_per_mw_ref": REF.heater_nm_per_mw,
          "laser_lambda_nm": lam_L, "T_ref_C": T_REF_C})

def T_at_laser(temp_c, t=t_c, a=a_d):
    """Through transmission seen by the fixed laser when the ring is at temperature temp_c."""
    lam_r = LAM0_NM + (temp_c - T_REF_C) * dldT * 1e-3                 # resonance moves red with T
    delta = lam_L - lam_r                                              # detuning in nm
    # evaluate the closed form at the laser wavelength with the resonance shifted:
    # shifting lambda_r by d is the same as evaluating T at lambda_L - d
    T, _ = analytic_T(np.atleast_1d(lam_L - (lam_r - LAM0_NM)) * 1e-3, t, a)
    return T, delta

temps = np.linspace(-20, 260, 5601)
T_vs_T, delta_vs_T = T_at_laser(temps)
h = 1e-3
slope = (T_at_laser(T_REF_C + h)[0] - T_at_laser(T_REF_C - h)[0]) / (2 * h)   # dT/dK at the bias point
R.update({"T_at_bias": float(T_at_laser(T_REF_C)[0][0]), "dT_dK_at_bias": float(slope[0]),
          "dT_per_0p1K_at_bias": float(0.1 * slope[0]),
          "dT_ddelta_per_pm_at_bias": float(slope[0] / dldT)})
# the temperature at which the laser is exactly on resonance, and the next resonance (one FSR later)
print(f"[A4] one FSR of thermal shift = {K_per_fsr:.1f} K; ambient 10..125 C = {R['ambient_span_nm']:.2f} nm = {R['ambient_span_fsr']:.2f} FSR")
print(f"[A4] 0.1 K = {R['pm_per_0p1K']:.1f} pm = {100*R['fraction_fwhm_per_0p1K']:.2f} % of the FWHM; at the bias point dT/dK = {slope[0]:+.4f} per K -> {100*0.1*slope[0]:+.2f} % transmission per 0.1 K")
print(f"[A4] heater: 50 pm/K x 8.8 K/mW = {R['heater_nm_per_mw_check']:.3f} nm/mW (ref {REF.heater_nm_per_mw}); one FSR costs {R['heater_mw_per_fsr']:.1f} mW")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(temps, T_vs_T, color=SERIES[0])
ax[0].axvspan(REF.ambient_min_c, REF.ambient_max_c, color=SERIES[3], alpha=0.10)
ax[0].text((REF.ambient_min_c + REF.ambient_max_c) / 2, 1.05, "ambient range 10..125 °C", ha="center", color=SERIES[3], fontsize=9)
ax[0].plot(T_REF_C, R["T_at_bias"], "o", color=SERIES[1], ms=6)
ax[0].annotate(f"operating point: laser at δ = +{REF.delta_opt_pm:.0f} pm, T = {R['T_at_bias']:.2f}",
               xy=(T_REF_C, R["T_at_bias"]), xytext=(60, 0.55), fontsize=8.5, color=SERIES[1],
               arrowprops=dict(arrowstyle="->", color=SERIES[1]))
# The laser is parked on the RED side (delta = lambda_L - lambda_r = +108 pm) and lambda_r red-shifts with T,
# so the notch reaches the laser when the ring is delta_opt/(50 pm/K) = +2.16 K HOTTER than the bias point.
dT_on_res = REF.delta_opt_pm / dldT                  # +2.16 K
Tr1 = T_REF_C + dT_on_res                            # 27.16 C: laser exactly on resonance
R.update({"dT_laser_on_resonance_K": dT_on_res, "T_laser_on_resonance_C": Tr1,
          "T_at_laser_on_resonance": float(T_at_laser(Tr1)[0][0]), "T_at_10C": float(T_at_laser(REF.ambient_min_c)[0][0])})
print(f"[A4] laser on resonance when the ring is {dT_on_res:+.2f} K from the bias point, i.e. at {Tr1:.2f} C (T = {R['T_at_laser_on_resonance']:.1e}); at 10 C T = {R['T_at_10C']:.2f}")
ax[0].annotate("", xy=(Tr1 + K_per_fsr, 0.02), xytext=(Tr1, 0.02), arrowprops=dict(arrowstyle="<->", color=PALETTE["ink2"]))
ax[0].text(Tr1 + K_per_fsr / 2, 0.06, f"one FSR = {K_per_fsr:.0f} K", ha="center", fontsize=9)
ax[0].set(xlabel="ring temperature (°C)", ylabel="transmission at the fixed laser T(λ_L)", ylim=(-0.03, 1.15),
          title=f"Fixed laser at {lam_L:.3f} nm: the notch sweeps past it\nevery FSR = {K_per_fsr:.0f} K of ring temperature ({dldT:.0f} pm/K)")
zoom = np.abs(temps - T_REF_C) < 6
ax[1].plot(temps[zoom] - T_REF_C, T_vs_T[zoom], color=SERIES[0])
ax[1].plot(0, R["T_at_bias"], "o", color=SERIES[1], ms=6)
dt_line = np.array([-1, 1])
ax[1].plot(dt_line, R["T_at_bias"] + slope[0] * dt_line, "--", color=SERIES[1], lw=1.2, label=f"slope {slope[0]:+.3f} per K")
ax[1].axvspan(-0.1, 0.1, color=SERIES[4], alpha=0.25, label=f"±0.1 K = ±{0.1*dldT:.0f} pm = ±{100*R['fraction_fwhm_per_0p1K']:.1f} % FWHM")
ax[1].axvline(dT_on_res, color=PALETTE["muted"], ls=":", lw=1)
ax[1].text(dT_on_res + 0.15, 0.62, f"laser on resonance\nΔT = {dT_on_res:+.2f} K\n(ring hotter: λ_r red-shifts\nonto λ_L)", ha="left", fontsize=8, color=PALETTE["ink2"])
ax[1].set(xlabel="ring temperature change from the bias point ΔT (K)", ylabel="T(λ_L)",
          title=f"Zoom around the bias point: 0.1 K = {0.1*dldT:.0f} pm\n= {100*abs(0.1*slope[0]):.1f} % change of the transmitted power")
ax[1].legend(fontsize=8.5, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "A4_thermal.png"); plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Animation: notch sliding under the fixed laser while the temperature ramps 10 -> 125 C
# ---------------------------------------------------------------------------
frames = 300                                             # 10 s at 30 fps
ramp = np.linspace(REF.ambient_min_c, REF.ambient_max_c, frames)
wl_win = np.linspace(LAM0_NM - 3, LAM0_NM + 9, 3000)
T_hist = np.array([T_at_laser(tc)[0][0] for tc in ramp])
fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.2))
line_spec, = axl.plot([], [], color=SERIES[0])
laser_line = axl.axvline(lam_L, color=SERIES[1], lw=2, label=f"laser λ_L = {lam_L:.3f} nm (fixed)")
dot, = axl.plot([], [], "o", color=SERIES[1], ms=7)
txt = axl.text(0.02, 0.06, "", transform=axl.transAxes, fontsize=9)
axl.set(xlim=(wl_win[0], wl_win[-1]), ylim=(-0.03, 1.1), xlabel="wavelength (nm)", ylabel="through transmission T",
        title="The ring's notch red-shifts 50 pm/K while the laser stays put")
axl.legend(loc="upper right", fontsize=8.5)
trace, = axr.plot([], [], color=SERIES[0])
dot2, = axr.plot([], [], "o", color=SERIES[1], ms=7)
axr.set(xlim=(REF.ambient_min_c, REF.ambient_max_c), ylim=(-0.03, 1.1), xlabel="ring temperature (°C)", ylabel="T at the laser",
        title="What the photodiode sees as the ambient ramps 10 → 125 °C")
fig.tight_layout()

def draw(i):
    tc = ramp[i]
    shift = (tc - T_REF_C) * dldT * 1e-3
    T, _ = analytic_T((wl_win - shift) * 1e-3, t_c, a_d)
    line_spec.set_data(wl_win, T)
    dot.set_data([lam_L], [T_hist[i]])
    txt.set_text(f"ring temperature {tc:6.1f} °C   resonance shift {shift*1e3:+6.0f} pm   δ = {lam_L - LAM0_NM - shift:+.3f} nm")
    trace.set_data(ramp[:i + 1], T_hist[:i + 1]); dot2.set_data([tc], [T_hist[i]])
    return line_spec, dot, txt, trace, dot2

anim = FuncAnimation(fig, draw, frames=frames, blit=True)
anim.save(OUT / "A5_notch_sliding.mp4", writer=FFMpegWriter(fps=30, bitrate=1800))
plt.close(fig)
# contact sheet
sheet_idx = np.linspace(0, frames - 1, 6).astype(int)
fig, axs = plt.subplots(2, 3, figsize=(12, 6))
for A, i in zip(axs.ravel(), sheet_idx):
    tc = ramp[i]; shift = (tc - T_REF_C) * dldT * 1e-3
    T, _ = analytic_T((wl_win - shift) * 1e-3, t_c, a_d)
    A.plot(wl_win, T, color=SERIES[0]); A.axvline(lam_L, color=SERIES[1], lw=2); A.plot(lam_L, T_hist[i], "o", color=SERIES[1])
    A.set(title=f"{tc:.0f} °C: T(λ_L) = {T_hist[i]:.2f}", xlabel="wavelength (nm)", ylabel="T", ylim=(-0.03, 1.1))
fig.suptitle("A5 contact sheet: the notch (blue) slides past the fixed laser (orange) as the ring heats up")
fig.tight_layout(); fig.savefig(OUT / "A5_notch_sliding_frames.png"); plt.close(fig)

R["runtime_s"] = time.time() - T0
with open(OUT / "A_results.json", "w") as f:
    json.dump(R, f, indent=2, default=float)
print(f"[A] done in {R['runtime_s']:.1f} s -> out/A_results.json")
