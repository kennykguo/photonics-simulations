"""13_capstone_ring_lock: the single-ring thermal lock, from plant model to PI design to 4-ring simulation.

Capstone objective (docs/BRIEF.md): keep a silicon microring's notch parked at δ = +108 pm from a fixed laser
against 10..125 C ambient drift with a heater-only actuator and a tapped photocurrent sensor.
Concepts used: notes §22 (n_g sets the thermal shift), §27-28 (the ring notch is a resonance, hence Lorentzian),
plus the capstone numbers in common/params.py.

Run:  cd experiments/13_capstone_ring_lock && ../../.venv/bin/python run.py     (~3-4 min, mostly video encoding)
Regenerates everything in out/ and rebuilds + executes explore.ipynb.
"""
import sys, pathlib, json, time, platform, subprocess, shutil
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import REF, use_style, SERIES, PALETTE, C0, Q_E
from common.units import delta_f_ghz_from_delta_lambda_nm
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
use_style()
for _p in OUT.iterdir():           # regenerate from scratch: nothing stale survives
    if _p.is_file(): _p.unlink()
    elif _p.is_dir(): shutil.rmtree(_p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyBboxPatch
import numpy as np
import scipy, control as ct
sys.path.insert(0, str(HERE))
import ringlock as rl

T0 = time.time()
LOG = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

res = {}
pl, se = rl.Plant(), rl.Sensor()
D_OPT = se.delta_opt_pm
I_SET = float(se.I(D_OPT))
FWHM = se.fwhm_pm
T_S = 10e-6                 # controller sample period (s): mean of I over the last T_S, then heater update
T_D = 10e-6                 # total delay used in the continuous design (T_S/2 averaging + T_S/2 hold)
P_MAX = 20.0                # assumed heater ceiling (mW)
T_AMB0 = 60.0               # reference ambient for the scenarios (C)
P0 = rl.nominal_heater_mw(pl, T_AMB0)     # heater bias at 60 C such that 0.5 mW is left at 125 C
c_blue, c_orange, c_aqua, c_violet, c_red, c_yellow = SERIES

say("=" * 78); say("13_capstone_ring_lock"); say("=" * 78)

# =============================================================================================
# 1. Static numbers: sensor, bias, OMA, heater, conversions
# =============================================================================================
say("\n[1] Sensor and bias point")
dgrid = np.linspace(0.0, 600.0, 600001)
slope = se.dT(dgrid)
d_opt_num = float(dgrid[np.argmax(slope)])
h = 1e-3
kI_fd = float((se.I(D_OPT + h) - se.I(D_OPT - h)) / (2 * h))
res.update({
    "p_in_mw": se.p_in_mw, "i_fs_ua": se.i_fs_ua, "fwhm_pm": FWHM, "t_min": se.t_min,
    "delta_opt_analytic_pm": D_OPT, "delta_opt_numeric_pm": d_opt_num,
    "T_at_delta_opt": float(se.T(D_OPT)), "T_at_delta_opt_expected_0.25+0.75Tmin": 0.25 + 0.75 * se.t_min,
    "i_set_ua": I_SET,
    "sensor_gain_ua_per_pm": float(se.dI(D_OPT)), "sensor_gain_ua_per_pm_finite_difference": kI_fd,
    "dT_ddelta_max_analytic_per_pm": se.dT_max_analytic(), "dT_ddelta_at_delta_opt_per_pm": float(se.dT(D_OPT)),
    "sensor_gain_ua_per_K": float(se.dI(D_OPT)) * pl.dlam_dT,
    "sensor_gain_ua_per_0p1K": float(se.dI(D_OPT)) * pl.dlam_dT * 0.1,
    "i_mean_with_modulation_ua": float(se.I_modulated_mean(D_OPT)), "i_cw_ua": I_SET,
    "modulation_swing_pm": se.mod_swing_pm,
})
say(f"  P_in = {se.p_in_mw:.3f} mW (4 dBm), tap 5 %, R = 0.9 A/W -> I_fs = {se.i_fs_ua:.2f} µA")
say(f"  δ_opt: numeric argmax of dT/dδ = {d_opt_num:.2f} pm, analytic FWHM/(2√3) = {D_OPT:.2f} pm, REF {REF.delta_opt_pm}")
say(f"  T(δ_opt) = {se.T(D_OPT):.4f} (expected 0.25 + 0.75 T_min = {0.25 + 0.75 * se.t_min:.4f}); I_set = {I_SET:.2f} µA")
say(f"  sensor gain dI/dδ = {se.dI(D_OPT):.4f} µA/pm (finite difference {kI_fd:.4f}); = {se.dI(D_OPT) * pl.dlam_dT:.2f} µA/K; 0.1 K -> {se.dI(D_OPT) * pl.dlam_dT * 0.1:.2f} µA")
say(f"  mean I with 65 pm NRZ modulation on = {se.I_modulated_mean(D_OPT):.3f} µA vs CW {I_SET:.3f} µA (inflection point: curvature ~0)")

# OMA vs bias
bias = np.linspace(0, 400, 4001)
oma = se.oma_mw(bias)
b_oma_opt = float(bias[np.argmax(oma)])
oma_opt = float(se.oma_mw(D_OPT))
def oma_penalty_db(err_pm):
    return float(10 * np.log10(se.oma_mw(D_OPT + err_pm) / oma_opt))
res.update({"oma_at_delta_opt_mw": oma_opt, "oma_finite_swing_optimum_bias_pm": b_oma_opt, "oma_max_mw": float(oma.max()),
            "er_db_at_delta_opt": float(se.er_db(D_OPT)),
            "oma_penalty_db_5pm": oma_penalty_db(5.0), "oma_penalty_db_m5pm": oma_penalty_db(-5.0),
            "oma_penalty_db_50pm": oma_penalty_db(50.0), "oma_penalty_db_m50pm": oma_penalty_db(-50.0),
            "oma_penalty_db_108pm": oma_penalty_db(108.0), "oma_penalty_db_m108pm": oma_penalty_db(-108.0)})
say(f"  OMA(δ_opt) = {oma_opt:.3f} mW with 65 pm swing (ER {se.er_db(D_OPT):.2f} dB); finite-swing optimum bias {b_oma_opt:.1f} pm (max OMA {oma.max():.3f} mW)")
say(f"  OMA penalty: ±5 pm (0.1 K) -> {oma_penalty_db(5):.3f}/{oma_penalty_db(-5):.3f} dB; ±50 pm (1 K) -> {oma_penalty_db(50):.2f}/{oma_penalty_db(-50):.2f} dB; ±108 pm -> {oma_penalty_db(108):.2f}/{oma_penalty_db(-108):.2f} dB")

# heater / conversions
one_fsr_K = REF.fsr_nm * 1e3 / pl.dlam_dT
amb_range = REF.ambient_max_c - REF.ambient_min_c
res.update({"heater_pm_per_mw": pl.heater_pm_per_mw, "heater_pm_per_mw_ref": REF.heater_nm_per_mw * 1e3,
            "one_fsr_K": one_fsr_K, "ambient_range_K": amb_range, "ambient_range_pm": amb_range * pl.dlam_dT,
            "heater_for_ambient_range_mw": amb_range / pl.r_th, "heater_bias_at_60C_mw": P0,
            "heater_at_10C_mw": rl.nominal_heater_mw(pl, 10.0), "heater_at_125C_mw": rl.nominal_heater_mw(pl, 125.0),
            "pm_per_0p1K": 0.1 * pl.dlam_dT, "fraction_fwhm_0p1K": 0.1 * pl.dlam_dT / FWHM, "fraction_fwhm_1K": pl.dlam_dT / FWHM,
            "ghz_per_K": delta_f_ghz_from_delta_lambda_nm(pl.dlam_dT * 1e-3, REF.lambda_nm),
            "fwhm_ghz": delta_f_ghz_from_delta_lambda_nm(FWHM * 1e-3, REF.lambda_nm)})
B_noise = 1 / (2 * T_S)
i_shot = np.sqrt(2 * Q_E * I_SET * 1e-6 * B_noise)                 # A rms
res.update({"noise_bandwidth_hz": B_noise, "shot_noise_na_rms": i_shot * 1e9, "shot_noise_limited_delta_pm_rms": i_shot * 1e6 / float(se.dI(D_OPT)),
            "delta_error_pm_per_percent_pin": 0.01 * I_SET / float(se.dI(D_OPT))})
say(f"  heater efficiency R_th·dλ/dT = {pl.heater_pm_per_mw:.0f} pm/mW (REF 0.44 nm/mW); one FSR = {one_fsr_K:.0f} K; 10..125 C = {amb_range * pl.dlam_dT / 1e3:.2f} nm = {amb_range / pl.r_th:.1f} mW of heater")
say(f"  heater bias: {rl.nominal_heater_mw(pl, 10):.2f} mW at 10 C, {P0:.2f} mW at 60 C, {rl.nominal_heater_mw(pl, 125):.2f} mW at 125 C (0.5 mW margin rule)")
say(f"  0.1 K = {0.1 * pl.dlam_dT:.0f} pm = {100 * 0.1 * pl.dlam_dT / FWHM:.2f} % of FWHM; 1 K = {100 * pl.dlam_dT / FWHM:.1f} % of FWHM; 50 pm/K = {res['ghz_per_K']:.2f} GHz/K")
say(f"  shot noise at I_set in B = 1/(2T_s) = {B_noise / 1e3:.0f} kHz: {i_shot * 1e9:.2f} nA rms -> {res['shot_noise_limited_delta_pm_rms']:.4f} pm rms; a 1 % P_in error reads as {res['delta_error_pm_per_percent_pin']:.2f} pm")

# --- figure: sensor curve
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
dd = np.linspace(-600, 600, 1201)
ax = axs[0]; ax.plot(dd, se.T(dd), color=c_blue, label="T(δ) through port")
ax.axvline(D_OPT, color=c_orange, ls="--", label=f"δ_opt = {D_OPT:.0f} pm")
ax.axvspan(-600, 0, color=PALETTE["line"], alpha=0.5, label="wrong side (slope < 0 with heating)")
ax.plot([D_OPT], [se.T(D_OPT)], "o", color=c_orange)
ax.plot([D_OPT - se.mod_swing_pm / 2, D_OPT + se.mod_swing_pm / 2], se.T(np.array([D_OPT - se.mod_swing_pm / 2, D_OPT + se.mod_swing_pm / 2])), "s", color=c_aqua, label="data levels (±32.5 pm)")
ax.set_xlabel("δ = λ_L − λ_r (pm)"); ax.set_ylabel("transmission"); ax.set_title("The notch: T_min = 0.016, FWHM = 374 pm"); ax.legend(fontsize=7, loc="lower left")   # lower left: clear of the notch curve and the data-level markers
ax = axs[1]; ax.plot(dd, se.I(dd), color=c_blue, label="I(δ) = I_fs·T(δ)")
ax.axhline(I_SET, color=c_orange, ls=":", label=f"I_set = {I_SET:.1f} µA"); ax.axvline(D_OPT, color=c_orange, ls="--")
ax.set_xlabel("δ (pm)"); ax.set_ylabel("photocurrent (µA)"); ax.set_title(f"Tapped photocurrent, I_fs = {se.i_fs_ua:.0f} µA"); ax.legend(fontsize=8)
ax = axs[2]; ax.plot(dd, se.dI(dd), color=c_blue, label="dI/dδ")
ax.plot([D_OPT], [se.dI(D_OPT)], "o", color=c_orange, label=f"max {se.dI(D_OPT):.3f} µA/pm at {D_OPT:.0f} pm")
ax.axhline(0, color=PALETTE["muted"], lw=0.8)
ax.set_xlabel("δ (pm)"); ax.set_ylabel("sensor gain (µA/pm)"); ax.set_title("Sensor gain: peak at FWHM/(2√3), sign flips at δ = 0", fontsize=9); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "sensor_curve.png"); plt.close(fig)

# --- figure: OMA vs bias and penalty vs lock error
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axs[0]; ax.plot(bias, oma, color=c_blue, label="OMA (65 pm swing)")
ax.axvline(D_OPT, color=c_orange, ls="--", label=f"δ_opt = {D_OPT:.0f} pm (small-signal)")
ax.axvline(b_oma_opt, color=c_aqua, ls=":", label=f"finite-swing optimum {b_oma_opt:.0f} pm")
ax.set_xlabel("bias δ (pm)"); ax.set_ylabel("OMA at bus output (mW)"); ax.set_title("OMA vs bias: the lock must hold δ near 108 pm"); ax.legend(fontsize=8)
err = np.linspace(-150, 150, 601)
ax = axs[1]; ax.plot(err, [oma_penalty_db(e) for e in err], color=c_blue)
for e_, lab in [(5, "0.1 K"), (50, "1 K")]:
    ax.axvline(e_, color=c_orange, ls="--", lw=1); ax.axvline(-e_, color=c_orange, ls="--", lw=1)
    ax.text(e_, -0.2, lab, fontsize=8, color=c_orange, ha="left")
ax.set_xlabel("lock error δ − δ_opt (pm)"); ax.set_ylabel("OMA penalty (dB)"); ax.set_title("OMA penalty vs lock error (0 dB at δ_opt)")
fig.tight_layout(); fig.savefig(OUT / "oma_vs_bias.png"); plt.close(fig)

# --- figure: thermal step response
tt = np.linspace(0, 1.5e-3, 3001)
tot, fast, slow = pl.thermal_step(tt, 1.0)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, xmax in zip(axs, (100e-6, 1.5e-3)):
    ax.plot(tt * 1e6, tot, color=c_blue, label="ΔT total (1 mW step)")
    ax.plot(tt * 1e6, fast, color=c_orange, ls="--", label="fast 75 %, τ = 10 µs")
    ax.plot(tt * 1e6, slow, color=c_aqua, ls="--", label="slow 25 %, τ = 300 µs")
    ax.axhline(pl.r_th, color=PALETTE["muted"], ls=":", label=f"R_th = {pl.r_th} K/mW")
    ax.set_xlim(0, xmax * 1e6); ax.set_xlabel("time (µs)"); ax.set_ylabel("ΔT (K)  [×50 = pm]")
axs[0].set_title("First 100 µs: 75 % of the heat arrives in ~30 µs"); axs[1].set_title("The 25 % tail takes ~1 ms")
for ax in axs: ax.legend(fontsize=8, loc="center right")     # centre right is clear of all four curves in both panels
fig.tight_layout(); fig.savefig(OUT / "thermal_step.png"); plt.close(fig)

# --- figure: block diagram
fig, ax = plt.subplots(figsize=(16, 4.2)); ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 4.2)
def box(x, y, w, h, text, color=c_blue, fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc="white", ec=color, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
def arrow(x0, y0, x1, y1, text="", dy=-0.55):      # labels sit just below the box row so they never overprint a box edge
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=PALETTE["ink2"], lw=1.3))
    if text: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", fontsize=7.2, color=PALETTE["ink2"])
y, hgt = 2.1, 0.8
boxes = [  # (x, w, text, colour)
    (1.15, 2.05, "PI controller\nC(s) = K_p·(1 + 1/(τ_i s))\nanti-windup, 0 ≤ P_h ≤ 20 mW", c_blue),
    (3.55, 1.55, "sample 10 µs\n+ hold\n≈ e^(−s·10 µs)", c_violet),
    (5.5, 2.5, "thermal G_th(s) = R_th·[0.75/(1+10µs·s)\n+ 0.25/(1+300µs·s)]\nR_th = 8.8 K/mW", c_orange),
    (9.05, 1.5, "×50 pm/K\nλ_r = λ_r0 + 50·ΔT", c_orange),
    (10.95, 1.5, "δ = λ_L − λ_r\n(laser fixed)", c_aqua),
    (12.85, 2.2, "Lorentzian notch T(δ)\nT_min 0.016, FWHM 374 pm\nI = P_in·tap·R·T(δ)", c_aqua),
]
for x, wd, txt, col in boxes: box(x, y, wd, hgt, txt, col)
ym = y + hgt / 2
ax.add_patch(plt.Circle((0.55, ym), 0.17, fc="white", ec=PALETTE["ink"])); ax.text(0.55, ym, "Σ", ha="center", va="center")
ax.text(0.55, ym + 0.85, "I_set = 29.6 µA", ha="center", fontsize=7.5); arrow(0.55, ym + 0.75, 0.55, ym + 0.2); ax.text(0.68, ym + 0.32, "−", fontsize=9)
arrow(0.73, ym, 1.15, ym, "e (µA)")
arrow(3.2, ym, 3.55, ym); arrow(5.1, ym, 5.5, ym, "P_h (mW)")
ax.add_patch(plt.Circle((8.4, ym), 0.17, fc="white", ec=PALETTE["ink"])); ax.text(8.4, ym, "Σ", ha="center", va="center")
arrow(8.0, ym, 8.22, ym); ax.text(8.4, ym + 0.95, "ambient ΔT_amb (K)\n+ crosstalk K·P", ha="center", fontsize=7.5); arrow(8.4, ym + 0.75, 8.4, ym + 0.2)
arrow(8.58, ym, 9.05, ym, "ΔT (K)"); arrow(10.55, ym, 10.95, ym); arrow(12.45, ym, 12.85, ym, "δ (pm)")
ax.plot([15.05, 15.35, 15.35, 0.55, 0.55], [ym, ym, ym - 1.15, ym - 1.15, ym - 0.2], color=PALETTE["ink2"], lw=1.3)
ax.annotate("", xy=(0.55, ym - 0.18), xytext=(0.55, ym - 0.5), arrowprops=dict(arrowstyle="->", color=PALETTE["ink2"], lw=1.3))
ax.text(8.0, ym - 1.05, "photocurrent I = I_fs·T(δ)  (µA; the controller uses its mean over the last 10 µs, which also averages out the 53 Gbaud data)", ha="center", fontsize=7.6, color=PALETTE["ink2"])
ax.text(8.0, 0.35, f"linearised loop gain at δ_opt:  L(s) = C(s)·e^(−sT_d)·G_th(s)·(50 pm/K)·(dI/dδ = {se.dI(D_OPT):.3f} µA/pm);   static heater→photocurrent gain {rl.loop_gain_static(pl, se):.0f} µA/mW", ha="center", fontsize=8)
ax.set_title("The ring-lock loop as implemented in ringlock.py   (sign: heating red-shifts λ_r, lowers δ, lowers I;  e = I − I_set > 0 means 'heat more')", fontsize=10)
fig.savefig(OUT / "block_diagram.png", bbox_inches="tight"); plt.close(fig)

# =============================================================================================
# 2. Linear design with python-control
# =============================================================================================
say("\n[2] PI design (python-control)")
des = rl.design_pi(pl, se, t_d=T_D)                       # tuning A: zero cancels the 10 µs pole
desB = rl.design_pi(pl, se, t_d=T_D, tau_i=pl.tau_slow)   # tuning B: zero at the 300 µs pole (comparison)
L, L0, C, PK = rl.loop_tf(pl, se, des["kp"], des["ki"], T_D)
LB, L0B, CB, _ = rl.loop_tf(pl, se, desB["kp"], desB["ki"], T_D)
gm, pm, wcg, wcp = ct.margin(L)
gmB, pmB, wcgB, wcpB = ct.margin(LB)
w = np.logspace(2, 7, 6000)
ex = rl.exact_margins(L0, T_D, w); exB = rl.exact_margins(L0B, T_D, w)
# hand estimate of the phase margin for tuning A: −90 (integrator, fast pole cancelled) + zero of the plant − slow pole − delay
tau_z = (1 - pl.slow_share) * pl.tau_slow + pl.slow_share * pl.tau_fast     # plant zero: 227.5 µs
pm_hand = 180 - 90 + np.degrees(np.arctan(des["wc"] * tau_z)) - np.degrees(np.arctan(des["wc"] * pl.tau_slow)) - np.degrees(des["wc"] * T_D)
S = ct.feedback(1, L); Tcl = ct.feedback(L, 1)
Sr = ct.frequency_response(S, w); Tr = ct.frequency_response(Tcl, w)
Ms = float(Sr.magnitude.max()); w_Ms = float(w[np.argmax(Sr.magnitude)])
Gd = ct.minreal(-pl.dlam_dT * pl.thermal_tf(normalised=True) * S, verbose=False)      # ambient ΔT (K) -> δ error (pm), closed loop
Gd_open = -pl.dlam_dT * pl.thermal_tf(normalised=True)
Gdr = ct.frequency_response(Gd, w); Gdo = ct.frequency_response(Gd_open, w)
i3 = np.where(Tr.magnitude < 1 / np.sqrt(2))[0]; f_bw = float(w[i3[0]] / 2 / np.pi) if len(i3) else np.nan
res.update({
    "design": {"kp_mw_per_ua": des["kp"], "ki_mw_per_ua_s": des["ki"], "tau_i_s": des["tau_i"], "wc_target_rad_s": des["wc"], "t_d_s": T_D,
               "K_I_ua_per_pm": des["K_I"], "K_lambda_pm_per_K": des["K_lambda"], "Kv_per_s": des["Kv"], "plant_zero_s": tau_z,
               "kp_pm_per_pm": des["kp"] * des["K_I"] * pl.heater_pm_per_mw},
    "designB_zero_at_slow_pole": {"kp_mw_per_ua": desB["kp"], "ki_mw_per_ua_s": desB["ki"], "tau_i_s": desB["tau_i"], "Kv_per_s": desB["Kv"],
                                  "pm_deg_pade": float(pmB), "gm_db_pade": float(20 * np.log10(gmB)), "crossover_hz_pade": float(wcpB / 2 / np.pi)},
    "crossover_hz_target": des["wc"] / 2 / np.pi, "crossover_hz_pade": float(wcp / 2 / np.pi), "crossover_hz_exact_delay": ex["wc"] / 2 / np.pi,
    "phase_margin_deg_pade": float(pm), "phase_margin_deg_exact_delay": ex["pm_deg"], "phase_margin_deg_hand": float(pm_hand),
    "gain_margin_db_pade": float(20 * np.log10(gm)), "gain_margin_db_exact_delay": ex["gm_db"], "phase_crossover_hz_pade": float(wcg / 2 / np.pi),
    "sensitivity_peak_Ms": Ms, "sensitivity_peak_hz": w_Ms / 2 / np.pi, "closed_loop_bandwidth_hz": f_bw,
    "delay_phase_at_crossover_deg": float(np.degrees(des["wc"] * T_D)),
})
say(f"  tuning A (zero cancels τ_f = 10 µs): K_p = {des['kp']:.5f} mW/µA, K_i = {des['ki']:.1f} mW/(µA·s), K_v = {des['Kv']:.0f} 1/s")
say(f"     K_p in pm-per-pm terms (heater pm per pm of error) = {res['design']['kp_pm_per_pm']:.3f}")
say(f"     crossover: target {des['wc'] / 2 / np.pi:.0f} Hz, python-control (Pade 3) {wcp / 2 / np.pi:.0f} Hz, exact delay {ex['wc'] / 2 / np.pi:.0f} Hz")
say(f"     phase margin: Pade {pm:.2f}°, exact delay {ex['pm_deg']:.2f}°, hand estimate {pm_hand:.2f}° (delay costs {np.degrees(des['wc'] * T_D):.1f}°)")
say(f"     gain margin: Pade {20 * np.log10(gm):.2f} dB at {wcg / 2 / np.pi / 1e3:.0f} kHz, exact {ex['gm_db']:.2f} dB; M_s = {Ms:.2f} at {w_Ms / 2 / np.pi / 1e3:.1f} kHz; closed-loop −3 dB {f_bw / 1e3:.1f} kHz")
say(f"  tuning B (zero at τ_s = 300 µs, same crossover): K_p = {desB['kp']:.5f}, K_i = {desB['ki']:.2f}, K_v = {desB['Kv']:.0f} 1/s, PM {pmB:.1f}°, GM {20 * np.log10(gmB):.1f} dB")

# --- figure: Bode of L with margins
Pr = ct.frequency_response(PK, w); Cr = ct.frequency_response(C, w)
fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
f = w / 2 / np.pi
ax = axs[0]
ax.semilogx(f, 20 * np.log10(Pr.magnitude), color=c_aqua, ls="--", label="plant P·K_λ·K_I (heater→photocurrent)")
ax.semilogx(f, 20 * np.log10(Cr.magnitude), color=c_violet, ls="--", label="PI  C(s), tuning A")
ax.semilogx(f, 20 * np.log10(ex["mag"]), color=c_blue, label="loop L = C·P·K·e^(−sT_d), tuning A")
ax.semilogx(f, 20 * np.log10(exB["mag"]), color=c_orange, label="loop, tuning B (zero at slow pole)")
ax.axhline(0, color=PALETTE["muted"], lw=0.8); ax.axvline(des["wc"] / 2 / np.pi, color=c_red, ls=":", label=f"f_c = {des['wc'] / 2 / np.pi / 1e3:.2f} kHz = 1/(4πT_d)")
for tau, lab, side in [(pl.tau_slow, "slow pole 1/(2π·300 µs)", "right"), (tau_z, "plant zero 1/(2π·227.5 µs)", "left"), (pl.tau_fast, "fast pole 1/(2π·10 µs)", "right")]:
    # the 530 Hz and 700 Hz lines are only 0.12 decades apart: put their labels on opposite sides of their lines
    ax.axvline(1 / (2 * np.pi * tau), color=PALETTE["line"], lw=1); ax.text(1 / (2 * np.pi * tau), 62, lab, fontsize=7, rotation=90, va="top", ha=side, color=PALETTE["muted"])
ax.set_ylabel("magnitude (dB)"); ax.set_ylim(-40, 70); ax.legend(fontsize=8, loc="upper right")
ax.set_title(f"Loop gain: crossover {ex['wc'] / 2 / np.pi / 1e3:.2f} kHz, PM {ex['pm_deg']:.1f}°, GM {ex['gm_db']:.1f} dB (tuning A);  B has 13× less low-frequency gain")
ax = axs[1]
ax.semilogx(f, ex["phase_deg"], color=c_blue, label="tuning A (exact delay)")
ax.semilogx(f, exB["phase_deg"], color=c_orange, label="tuning B")
Lr = ct.frequency_response(L, w); ax.semilogx(f, np.degrees(np.unwrap(Lr.phase)), color=c_blue, ls=":", lw=1, label="tuning A, Pade(3) delay")
ax.axhline(-180, color=PALETTE["muted"], lw=0.8); ax.axvline(des["wc"] / 2 / np.pi, color=c_red, ls=":")
ax.annotate(f"PM = {ex['pm_deg']:.1f}°", xy=(des["wc"] / 2 / np.pi, -180 + ex["pm_deg"]), xytext=(des["wc"] / 2 / np.pi * 3, -100), arrowprops=dict(arrowstyle="->", color=c_red), color=c_red)
ax.set_ylim(-360, 0); ax.set_ylabel("phase (deg)"); ax.set_xlabel("frequency (Hz)"); ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "bode_loop.png"); plt.close(fig)

# --- figure: sensitivity and disturbance rejection
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
ax = axs[0]
ax.semilogx(f, 20 * np.log10(Sr.magnitude), color=c_blue, label="|S| = 1/|1+L|  (error / disturbance)")
ax.semilogx(f, 20 * np.log10(Tr.magnitude), color=c_orange, label="|T| = |L/(1+L)|  (setpoint tracking)")
ax.axhline(20 * np.log10(Ms), color=c_red, ls=":", label=f"M_s = {Ms:.2f} ({20 * np.log10(Ms):.1f} dB)")
ax.axvline(des["wc"] / 2 / np.pi, color=PALETTE["muted"], ls=":")
ax.set_ylim(-60, 10); ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("|S|, |T| (dB)"); ax.legend(fontsize=8, loc="lower right"); ax.set_title("Sensitivity: disturbances below ~3 kHz are attenuated")
ax = axs[1]
ax.loglog(f, Gdo.magnitude, color=PALETTE["muted"], ls="--", label="lock off: |δ/ΔT_amb| (50 pm/K × thermal filter)")
ax.loglog(f, Gdr.magnitude, color=c_blue, label="lock on: |δ/ΔT_amb| = 50·|P_n·S|")
GdrB = ct.frequency_response(ct.minreal(-pl.dlam_dT * pl.thermal_tf(normalised=True) * ct.feedback(1, LB), verbose=False), w)
ax.loglog(f, GdrB.magnitude, color=c_orange, label="lock on, tuning B")
ax.axhline(0.1 * pl.dlam_dT, color=c_red, ls=":", label="5 pm = 0.1 K equivalent per 1 K disturbance")
ax.set_xlabel("frequency of the ambient disturbance (Hz)"); ax.set_ylabel("residual detuning per K of ambient (pm/K)")
ax.set_title("Residual error per K of ambient wobble, vs its frequency"); ax.legend(fontsize=7.5, loc="lower right"); ax.set_ylim(1e-3, 100)
fig.tight_layout(); fig.savefig(OUT / "sensitivity.png"); plt.close(fig)

# --- linear closed-loop step responses
tl = np.linspace(0, 1.2e-3, 2401)
sr = ct.step_response(Tcl, tl)
dr = ct.forced_response(Gd, tl, np.ones_like(tl))          # 1 K ambient step (through the thermal filter)
srB = ct.step_response(ct.feedback(LB, 1), tl)
drB = ct.forced_response(ct.minreal(-pl.dlam_dT * pl.thermal_tf(normalised=True) * ct.feedback(1, LB), verbose=False), tl, np.ones_like(tl))
def settle(t, y, tol, final=0.0):
    bad = np.where(np.abs(y - final) > tol)[0]
    return float(t[bad[-1]]) if len(bad) else 0.0
res.update({"linear_setpoint_step_overshoot_percent": float(100 * (sr.outputs.max() - 1)), "linear_setpoint_step_settle_2pct_us": settle(tl, sr.outputs, 0.02, 1.0) * 1e6,
            "linear_1K_step_peak_error_pm": float(dr.outputs.min()), "linear_1K_step_settle_to_0p5pm_us": settle(tl, dr.outputs, 0.5) * 1e6,
            "linear_1K_step_peak_error_pm_tuningB": float(drB.outputs.min())})
say(f"  linear closed loop (A): setpoint step overshoot {res['linear_setpoint_step_overshoot_percent']:.1f} %, 2 % settle {res['linear_setpoint_step_settle_2pct_us']:.0f} µs (the 25 % slow tail);")
say(f"     1 K ambient step -> peak error {dr.outputs.min():.1f} pm, back within 0.5 pm after {res['linear_1K_step_settle_to_0p5pm_us']:.0f} µs;  tuning B peak {drB.outputs.min():.1f} pm")
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
ax = axs[0]; ax.plot(tl * 1e6, sr.outputs, color=c_blue, label="tuning A"); ax.plot(tl * 1e6, srB.outputs, color=c_orange, label="tuning B")
ax.axhline(1, color=PALETTE["muted"], lw=0.8); ax.set_xlabel("time (µs)"); ax.set_ylabel("δ / δ_step"); ax.set_title("Setpoint step (linear): fast rise, then the 25 % slow-tail creep"); ax.legend(fontsize=8)
ax = axs[1]; ax.plot(tl * 1e6, dr.outputs, color=c_blue, label="tuning A"); ax.plot(tl * 1e6, drB.outputs, color=c_orange, label="tuning B")
ax.plot(tl * 1e6, -pl.dlam_dT * pl.thermal_step(tl, 1.0)[0] / pl.r_th, color=PALETTE["muted"], ls="--", label="lock off (−50 pm/K × filter)")
ax.set_xlabel("time (µs)"); ax.set_ylabel("δ − δ_opt (pm)"); ax.set_title("1 K ambient step (linear): error per K of disturbance"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "linear_step.png"); plt.close(fig)

# =============================================================================================
# 3. Nonlinear time-domain scenarios
# =============================================================================================
def make_ctrl(**kw):
    base = dict(kp=des["kp"], ki=des["ki"], i_set=I_SET, ts=T_S, p_max=P_MAX, p_init=P0)
    base.update(kw); return rl.PIController(**base)
def err_stats(r, j=0, t_from=None):
    e = r.delta[:, j] - D_OPT
    m = np.ones_like(e, bool) if t_from is None else (r.t >= t_from)
    ee = e[m]; k = np.argmax(np.abs(ee))
    return dict(peak_pm=float(ee[k]), peak_frac_fwhm=float(ee[k] / FWHM), min_delta_pm=float(r.delta[m, j].min()),
                final_pm=float(e[-200:].mean()), rms_last_pm=float(e[-200:].std()))

# (a) 5 K ambient step -------------------------------------------------------------------------
say("\n[3a] 5 K ambient step at t = 100 µs (through the ring's own thermal filter)")
STEP_K, T_STEP = 5.0, 100e-6
amb_a = lambda t: np.array([STEP_K if t >= T_STEP else 0.0])
ra = rl.simulate(pl, se, [make_ctrl()], 1.5e-3, ambient_fn=amb_a, p_nominal_mw=P0)
ra_off = rl.simulate(pl, se, [make_ctrl()], 1.5e-3, ambient_fn=amb_a, p_nominal_mw=P0, lock_enabled=[False])
sa = err_stats(ra, t_from=T_STEP)
e_a = ra.delta[:, 0] - D_OPT
ra_mid = rl.simulate(pl, se, [make_ctrl()], 1.5e-3, ambient_fn=lambda t: np.array([STEP_K if t >= T_STEP + T_S / 2 else 0.0]), p_nominal_mw=P0)   # step halfway between ticks
ra_small = rl.simulate(pl, se, [make_ctrl()], 1.5e-3, ambient_fn=lambda t: np.array([0.1 if t >= T_STEP else 0.0]), p_nominal_mw=P0)            # 0.1 K: linear regime of the sensor
peak_mid = float((ra_mid.delta[:, 0] - D_OPT).min()); peak_small_x50 = float((ra_small.delta[:, 0] - D_OPT).min() * 50)
t_settle = settle(ra.t - T_STEP, e_a, 5.0)               # last time |e| > 5 pm (0.1 K)
tlin = np.linspace(0, 1.5e-3, 3001)
lin = ct.forced_response(Gd, tlin, np.where(tlin >= T_STEP, STEP_K, 0.0))
res["scenario_a_step"] = {"step_K": STEP_K, "step_pm": STEP_K * pl.dlam_dT, **sa, "settle_to_5pm_us": t_settle * 1e6,
                          "linear_model_peak_error_pm": float(lin.outputs.min()), "peak_error_pm_step_between_ticks": peak_mid, "peak_error_pm_0p1K_step_times_50": peak_small_x50,
                          "heater_change_mw": float(ra.p[-1, 0] - P0), "heater_change_expected_mw": -STEP_K / pl.r_th,
                          "oma_penalty_db_at_peak": oma_penalty_db(sa["peak_pm"]), "oma_penalty_db_final": oma_penalty_db(sa["final_pm"]),
                          "lock_off_final_error_pm": float(ra_off.delta[-1, 0] - D_OPT), "lock_off_oma_penalty_db": oma_penalty_db(float(ra_off.delta[-1, 0] - D_OPT))}
say(f"  peak error {sa['peak_pm']:.1f} pm ({100 * sa['peak_frac_fwhm']:.0f} % of FWHM), notch bottom passes to δ_min = {sa['min_delta_pm']:.1f} pm; linear model predicts {float(lin.outputs.min()):.1f} pm")
say(f"  same step landing halfway between controller ticks: {peak_mid:.1f} pm (worst phase); 0.1 K step × 50 = {peak_small_x50:.1f} pm (so the sensor nonlinearity costs only {sa['peak_pm'] - peak_small_x50:+.1f} pm; the rest is sampled-vs-continuous delay)")
say(f"  back inside ±5 pm (0.1 K) after {t_settle * 1e6:.0f} µs; steady-state error {sa['final_pm']:.4f} pm; heater {ra.p[-1, 0] - P0:+.3f} mW (expected −5/8.8 = {-STEP_K / pl.r_th:+.3f})")
say(f"  OMA penalty at the peak {oma_penalty_db(sa['peak_pm']):.2f} dB, at steady state {oma_penalty_db(sa['final_pm']):.4f} dB; lock off: error {ra_off.delta[-1, 0] - D_OPT:.0f} pm -> {res['scenario_a_step']['lock_off_oma_penalty_db']:.1f} dB")

fig, axs = plt.subplots(2, 2, figsize=(12, 7))
tu = (ra.t - T_STEP) * 1e6
ax = axs[0, 0]; ax.plot(tu, ra.delta[:, 0], color=c_blue, label="lock on (nonlinear sim)")
ax.plot(tu, ra_off.delta[:, 0], color=PALETTE["muted"], ls="--", label="lock off")
ax.plot((tlin - T_STEP) * 1e6, D_OPT + lin.outputs, color=c_orange, ls=":", label="linear model (python-control)")
ax.axhline(D_OPT, color=c_aqua, lw=0.8); ax.axhline(0, color=c_red, lw=0.8, ls=":"); ax.text(1200, 8, "notch bottom (δ = 0)", color=c_red, fontsize=8)
ax.set_xlim(-50, 1400); ax.set_ylabel("δ = λ_L − λ_r (pm)"); ax.set_title(f"5 K step: the notch overshoots by {sa['peak_pm']:.0f} pm, then is pulled back"); ax.legend(fontsize=8)
ax = axs[0, 1]; ax.plot(tu, ra.i[:, 0], color=c_blue, label="photocurrent I(t)"); ax.step(ra.t_tick * 1e6 - T_STEP * 1e6, ra.i_meas[:, 0], where="post", color=c_violet, lw=1, label="sampled mean (10 µs)")
ax.axhline(I_SET, color=c_aqua, lw=0.8, label="I_set"); ax.set_xlim(-50, 1400); ax.set_ylabel("photocurrent (µA)"); ax.set_title("What the controller sees: the sampled mean photocurrent"); ax.legend(fontsize=8)
ax = axs[1, 0]; ax.plot(tu, ra.p[:, 0], color=c_orange, label="heater P_h (lock on)"); ax.axhline(P0 - STEP_K / pl.r_th, color=PALETTE["muted"], ls=":", label="P0 − 5 K / R_th")
ax.set_xlim(-50, 1400); ax.set_xlabel("time after the step (µs)"); ax.set_ylabel("heater (mW)"); ax.set_title("Heater backs off by 5 K / 8.8 K/mW = 0.57 mW"); ax.legend(fontsize=8)
ax = axs[1, 1]; ax.plot(tu, ra.dT[:, 0] - ra.dT[0, 0], color=c_blue, label="ring ΔT, lock on"); ax.plot(tu, ra_off.dT[:, 0] - ra_off.dT[0, 0], color=PALETTE["muted"], ls="--", label="ring ΔT, lock off")
ax.set_xlim(-50, 1400); ax.set_xlabel("time after the step (µs)"); ax.set_ylabel("ring ΔT (K)"); ax.set_title("Ring temperature: held constant by backing the heater off"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "scenario_a_step.png"); plt.close(fig)

# (a2) survival map: step size × extra rise time -------------------------------------------------
say("\n[3a2] Survival map: how big and how fast an ambient step can the lock take?")
sizes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15], float)
rises = np.array([0.0, 20e-6, 50e-6, 100e-6, 200e-6, 500e-6, 1e-3])
peak_map = np.full((len(sizes), len(rises)), np.nan); final_map = np.full_like(peak_map, np.nan); surv = np.zeros_like(peak_map, bool)
for i_, sz in enumerate(sizes):
    for j_, tr in enumerate(rises):
        if tr == 0: fn = lambda t, sz=sz: np.array([sz if t >= T_STEP else 0.0])
        else: fn = lambda t, sz=sz, tr=tr: np.array([sz * (1 - np.exp(-(t - T_STEP) / tr)) if t >= T_STEP else 0.0])
        rr = rl.simulate(pl, se, [make_ctrl()], 3e-3 + 3 * tr, ambient_fn=fn, p_nominal_mw=P0)
        st = err_stats(rr, t_from=T_STEP)
        peak_map[i_, j_] = st["peak_pm"]; final_map[i_, j_] = st["final_pm"]; surv[i_, j_] = abs(st["final_pm"]) < 2.0
largest_ok = {f"{tr * 1e6:.0f}us": float(sizes[surv[:, j_]].max()) if surv[:, j_].any() else 0.0 for j_, tr in enumerate(rises)}
peak_5K_200us = float(peak_map[int(np.where(sizes == 5.0)[0][0]), int(np.where(np.isclose(rises, 200e-6))[0][0])])   # quoted for the notebook's rise-time slider
# the smallest instantaneous step that loses the lock (7 K): where does the notch end up? With the heater pinned at
# P_max the ring is (P_max − P0)·η + ΔT_amb·50 pm/K to the red of where it should be ("Experiments to try" 5)
i_lost = int(np.where(~surv[:, 0])[0][0]); sz_lost = float(sizes[i_lost])
lost_step = {"step_K": sz_lost, "rise_s": 0.0, "final_error_pm": float(final_map[i_lost, 0]), "final_delta_pm": D_OPT + float(final_map[i_lost, 0]),
             "notch_red_of_laser_nm": -(D_OPT + float(final_map[i_lost, 0])) / 1e3,
             "expected_error_pm_heater_at_rail": -((P_MAX - P0) * pl.heater_pm_per_mw + sz_lost * pl.dlam_dT)}
res["survival_map"] = {"step_K": sizes.tolist(), "extra_rise_time_s": rises.tolist(), "peak_error_pm": peak_map.tolist(), "final_error_pm": final_map.tolist(), "survived": surv.tolist(),
                       "largest_surviving_step_K_by_rise_time": largest_ok, "peak_error_pm_5K_200us_rise": peak_5K_200us, "smallest_lost_instantaneous_step": lost_step}
say("  largest step that stays locked, by extra rise time: " + ", ".join(f"{k}: {v:.0f} K" for k, v in largest_ok.items()))
say(f"  a 5 K step with a 200 µs rise peaks at {peak_5K_200us:.1f} pm (vs {sa['peak_pm']:.1f} pm through the ring filter alone)")
say(f"  {sz_lost:.0f} K instantaneous step: lock lost, heater pinned at {P_MAX:.0f} mW, final δ = {lost_step['final_delta_pm']:.0f} pm, i.e. the notch ends {lost_step['notch_red_of_laser_nm']:.2f} nm to the red of the laser "
    f"(error {lost_step['final_error_pm']:.1f} pm vs −[(P_max − P0)·η + ΔT·50] = {lost_step['expected_error_pm_heater_at_rail']:.1f} pm)")
fig, ax = plt.subplots(figsize=(8, 4.5))
for j_, tr in enumerate(rises):
    lab = "ring filter only" if tr == 0 else f"+ {tr * 1e6:.0f} µs rise"
    ok = surv[:, j_]
    ax.plot(sizes[ok], -peak_map[ok, j_], "o-" if j_ < 6 else "s--", color=SERIES[j_ % 6], ms=4, label=lab)
    if (~ok).any(): ax.plot(sizes[~ok], np.full((~ok).sum(), -sizes[~ok] * 0 + 420), "x", color=SERIES[j_ % 6], ms=8)
ax.axhline(D_OPT, color=c_red, ls="--", label="δ_opt = 108 pm: notch bottom reached")
ax.axhline(FWHM / 2, color=PALETTE["muted"], ls=":", label="FWHM/2")
ax.text(0.5, 425, "× = lock lost (ends on the wrong side)", fontsize=8)
ax.set_xlabel("ambient step (K)"); ax.set_ylabel("peak error |δ − δ_opt| (pm)"); ax.set_ylim(0, 450)
ax.set_title(f"Peak error vs step size and speed: instantaneous steps survive to {largest_ok['0us']:.0f} K, 100 µs-rise steps to ≥ {largest_ok['100us']:.0f} K", fontsize=9.5); ax.legend(fontsize=7.5, ncol=2)
fig.tight_layout(); fig.savefig(OUT / "step_survival.png"); plt.close(fig)

# (b) 800 K/s ramp for 10 ms --------------------------------------------------------------------
say("\n[3b] Ambient ramp 800 K/s for 10 ms (8 K total)")
RAMP, T_R0, T_R1 = 800.0, 200e-6, 10.2e-3
amb_b = lambda t: np.array([RAMP * min(max(t - T_R0, 0.0), T_R1 - T_R0)])
rb = rl.simulate(pl, se, [make_ctrl()], 12e-3, ambient_fn=amb_b, p_nominal_mw=P0)
rbB = rl.simulate(pl, se, [make_ctrl(kp=desB["kp"], ki=desB["ki"])], 12e-3, ambient_fn=amb_b, p_nominal_mw=P0)
mid = (rb.t > 4e-3) & (rb.t < 10e-3)
eb, ebB = rb.delta[:, 0] - D_OPT, rbB.delta[:, 0] - D_OPT
pred = RAMP * pl.dlam_dT / des["Kv"]; predB = RAMP * pl.dlam_dT / desB["Kv"]
res["scenario_b_ramp"] = {"ramp_K_per_s": RAMP, "ramp_pm_per_s": RAMP * pl.dlam_dT, "duration_ms": (T_R1 - T_R0) * 1e3, "total_K": RAMP * (T_R1 - T_R0),
                          "tracking_error_pm_sim": float(-eb[mid].mean()), "tracking_error_pm_predicted_Kv": pred, "tracking_error_frac_fwhm": float(-eb[mid].mean() / FWHM),
                          "tracking_error_pm_sim_tuningB": float(-ebB[mid].mean()), "tracking_error_pm_predicted_tuningB": predB,
                          "peak_error_pm": float(eb[np.argmax(np.abs(eb))]), "heater_change_mw": float(rb.p[-1, 0] - P0), "heater_change_expected_mw": -RAMP * (T_R1 - T_R0) / pl.r_th,
                          "oma_penalty_db_tracking": oma_penalty_db(float(eb[mid].mean()))}
say(f"  ramp = {RAMP * pl.dlam_dT / 1e3:.0f} pm/ms; tracking error A: sim {-eb[mid].mean():.3f} pm vs K_v prediction {pred:.3f} pm ({100 * -eb[mid].mean() / FWHM:.2f} % FWHM, OMA penalty {oma_penalty_db(float(eb[mid].mean())):.5f} dB)")
say(f"  tuning B: sim {-ebB[mid].mean():.2f} pm vs prediction {predB:.2f} pm; heater ends {rb.p[-1, 0] - P0:+.3f} mW (expected {-RAMP * (T_R1 - T_R0) / pl.r_th:+.3f})")
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
ax = axs[0]; ax.plot(rb.t * 1e3, rb.t_amb[:, 0], color=c_orange); ax.set_xlabel("time (ms)"); ax.set_ylabel("ambient rise (K)"); ax.set_title("Disturbance: 800 K/s for 10 ms = 8 K = 400 pm")
ax = axs[1]; ax.plot(rb.t * 1e3, eb, color=c_blue, label="tuning A (zero at 10 µs pole)"); ax.plot(rbB.t * 1e3, ebB, color=c_orange, label="tuning B (zero at 300 µs pole)")
ax.axhline(-pred, color=c_blue, ls=":", lw=1, label=f"K_v prediction A: {pred:.2f} pm"); ax.axhline(-predB, color=c_orange, ls=":", lw=1, label=f"K_v prediction B: {predB:.1f} pm")
ax.set_xlabel("time (ms)"); ax.set_ylabel("δ − δ_opt (pm)"); ax.set_title("Tracking error = ramp rate / K_v"); ax.legend(fontsize=7.5)
ax = axs[2]; ax.plot(rb.t * 1e3, rb.p[:, 0], color=c_blue, label="A"); ax.plot(rbB.t * 1e3, rbB.p[:, 0], color=c_orange, label="B"); ax.set_xlabel("time (ms)"); ax.set_ylabel("heater (mW)"); ax.set_title("Heater ramps down 8 K / 8.8 K/mW = 0.91 mW"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "scenario_b_ramp.png"); plt.close(fig)

# (f) the two "Experiments to try" re-designs, computed here so that the README quotes numbers the code produced ----
say("\n[3f] 'Experiments to try' 1 and 2: crossover pushed to ω_c·T_d = 1, and a slow controller with T_d = T_s = 50 µs")
def zero_crossings(y, floor=1e-3):
    """How many times a response changes sign (ignoring |y| < floor): 1 = one overshoot, more = ringing."""
    s = np.sign(y[np.abs(y) > floor]); return int(np.sum(s[1:] != s[:-1]))
def linear_summary(des_, t_d_):
    L_, L0_, _, _ = rl.loop_tf(pl, se, des_["kp"], des_["ki"], t_d_)
    ex_ = rl.exact_margins(L0_, t_d_, w)
    S_ = ct.feedback(1, L_); Sr_ = ct.frequency_response(S_, w)
    sr_ = ct.step_response(ct.feedback(L_, 1), tl)
    dr_ = ct.forced_response(ct.minreal(-pl.dlam_dT * pl.thermal_tf(normalised=True) * S_, verbose=False), tl, np.ones_like(tl))
    pmh = 90 + np.degrees(np.arctan(des_["wc"] * tau_z)) - np.degrees(np.arctan(des_["wc"] * pl.tau_slow)) - np.degrees(des_["wc"] * t_d_)
    return {"t_d_s": t_d_, "wc_times_td": des_["wc"] * t_d_, "crossover_hz_target": des_["wc"] / 2 / np.pi, "crossover_hz_exact_delay": ex_["wc"] / 2 / np.pi,
            "phase_margin_deg_exact_delay": ex_["pm_deg"], "phase_margin_deg_hand": float(pmh), "gain_margin_db_exact_delay": ex_["gm_db"],
            "sensitivity_peak_Ms": float(Sr_.magnitude.max()), "sensitivity_peak_hz": float(w[np.argmax(Sr_.magnitude)] / 2 / np.pi),
            "kp_mw_per_ua": des_["kp"], "ki_mw_per_ua_s": des_["ki"], "Kv_per_s": des_["Kv"], "ramp_error_pm_predicted_Kv": RAMP * pl.dlam_dT / des_["Kv"],
            "setpoint_step_overshoot_percent": float(100 * (sr_.outputs.max() - 1)), "linear_1K_step_peak_error_pm": float(dr_.outputs.min()),
            "linear_1K_step_zero_crossings": zero_crossings(dr_.outputs)}
def ramp_error_sim(des_, ts_):
    r_ = rl.simulate(pl, se, [make_ctrl(kp=des_["kp"], ki=des_["ki"], ts=ts_)], 12e-3, ambient_fn=amb_b, p_nominal_mw=P0)
    m_ = (r_.t > 4e-3) & (r_.t < 10e-3); return float(-(r_.delta[m_, 0] - D_OPT).mean())
base_lin = linear_summary(des, T_D)
# 1. push the crossover to ω_c·T_d = 1 (twice the design value; the notebook's ω_c·T_d slider at 1.0)
desF = rl.design_pi(pl, se, t_d=T_D, wc=1.0 / T_D)
fast = linear_summary(desF, T_D); fast["ramp_error_pm_sim"] = ramp_error_sim(desF, T_S)
# 2. slow the controller: T_s = T_d = 50 µs with the same design rule ω_c = 1/(2 T_d), and how big a step it survives
T_D_SLOW = 50e-6
desS = rl.design_pi(pl, se, t_d=T_D_SLOW)
slow = linear_summary(desS, T_D_SLOW); slow["ramp_error_pm_sim"] = ramp_error_sim(desS, T_D_SLOW)
sizesS = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]); peakS, survS = [], []
for sz in sizesS:
    rr = rl.simulate(pl, se, [make_ctrl(kp=desS["kp"], ki=desS["ki"], ts=T_D_SLOW)], 6e-3, ambient_fn=lambda t, sz=sz: np.array([sz if t >= T_STEP else 0.0]), p_nominal_mw=P0)
    st = err_stats(rr, t_from=T_STEP); peakS.append(st["peak_pm"]); survS.append(bool(abs(st["final_pm"]) < 2.0))
slow.update({"instantaneous_step_K": sizesS.tolist(), "peak_error_pm": peakS, "survived": survS,
             "largest_surviving_instantaneous_step_K": float(sizesS[np.array(survS)].max()) if any(survS) else 0.0})
res["experiments_to_try"] = {"baseline_tuning_A": base_lin, "1_push_crossover_wc_td_1": fast, "2_slow_controller_td_50us": slow}
say(f"  baseline (ω_c·T_d = 0.5): PM {base_lin['phase_margin_deg_exact_delay']:.1f}°, M_s {base_lin['sensitivity_peak_Ms']:.2f}, setpoint overshoot {base_lin['setpoint_step_overshoot_percent']:.0f} %, 1 K step crosses zero {base_lin['linear_1K_step_zero_crossings']} time(s)")
say(f"  1. ω_c·T_d = 1.0: crossover {fast['crossover_hz_exact_delay'] / 1e3:.2f} kHz, PM {fast['phase_margin_deg_exact_delay']:.1f}° (hand {fast['phase_margin_deg_hand']:.1f}°), GM {fast['gain_margin_db_exact_delay']:.1f} dB, M_s {fast['sensitivity_peak_Ms']:.2f} at {fast['sensitivity_peak_hz'] / 1e3:.0f} kHz, "
    f"K_v {fast['Kv_per_s']:.0f} 1/s ({fast['Kv_per_s'] / des['Kv']:.2f}× A), ramp error sim {fast['ramp_error_pm_sim']:.3f} pm (K_v: {fast['ramp_error_pm_predicted_Kv']:.3f}), setpoint overshoot {fast['setpoint_step_overshoot_percent']:.0f} %, 1 K step crosses zero {fast['linear_1K_step_zero_crossings']} times")
say(f"  2. T_d = 50 µs: crossover {slow['crossover_hz_exact_delay'] / 1e3:.2f} kHz (1/(4πT_d) = {1 / (4 * np.pi * T_D_SLOW) / 1e3:.2f}), PM {slow['phase_margin_deg_exact_delay']:.1f}°, K_v {slow['Kv_per_s']:.0f} 1/s ({des['Kv'] / slow['Kv_per_s']:.1f}× less), "
    f"ramp error sim {slow['ramp_error_pm_sim']:.2f} pm (K_v: {slow['ramp_error_pm_predicted_Kv']:.2f}); instantaneous steps: " + ", ".join(f"{s_:g} K {'ok' if ok_ else 'LOST'} ({p_:.0f} pm)" for s_, ok_, p_ in zip(sizesS, survS, peakS)))

# (c) four rings with crosstalk; ring 3 acquires from cold ----------------------------------------
say("\n[3c] Four rings on one bus, thermal crosstalk K (10 % / 3 %); ring 3 turns on and acquires lock")
N = 4; K = rl.crosstalk_matrix(N)
ACQ = 2                                          # ring index that starts cold (ring 3 in 1-based counting)
locked = [i != ACQ for i in range(N)]
# equilibrium heater powers for the locked rings with the cold ring at 0 mW: K_SS P_S = P0·1 − K_SU·0
S_idx = [i for i in range(N) if locked[i]]
P_S = np.linalg.solve(K[np.ix_(S_idx, S_idx)], np.full(len(S_idx), P0))
p_init = np.zeros(N); p_init[S_idx] = P_S
P_final_expected = np.linalg.solve(K, np.full(N, P0))
D_CAPTURE = 300.0
ctrls = [make_ctrl(p_init=float(p_init[i]), acquire=(i == ACQ), i_capture=float(se.I(D_CAPTURE)), sweep_rate=4000.0) for i in range(N)]
rc = rl.simulate(pl, se, ctrls, 5e-3, K=K, p_nominal_mw=P0)
stats_c = {f"ring{i + 1}": err_stats(rc, i, t_from=0.0 if i != ACQ else None) for i in range(N)}
k_cap = int(np.argmax(rc.mode[:, ACQ] > 0.5)); t_cap = float(rc.t_tick[k_cap])
e3 = rc.delta[:, ACQ] - D_OPT; t_lock3 = settle(rc.t, e3, 5.0)
nb_peak = {f"ring{i + 1}": float((rc.delta[:, i] - D_OPT)[np.argmax(np.abs(rc.delta[:, i] - D_OPT))]) for i in range(N) if i != ACQ}
nb_peak_time_ms = {f"ring{i + 1}": float(rc.t[np.argmax(np.abs(rc.delta[:, i] - D_OPT))] * 1e3) for i in range(N) if i != ACQ}
m_sw = (rc.t > 0.5e-3) & (rc.t < t_cap - 0.1e-3)
nb_sweep_mean = {f"ring{i + 1}": float((rc.delta[m_sw, i] - D_OPT).mean()) for i in range(N) if i != ACQ}
xt_K = {f"ring{i + 1}": float(K[i, ACQ] * P_final_expected[ACQ] * pl.r_th) for i in range(N) if i != ACQ}
sweep_ramp_pm_s = 4000.0 * pl.r_th * pl.dlam_dT
# quasi-static prediction of the neighbours' error during the sweep: each locked ring's own heater must ramp at
# dP_S/dt = −K_SS⁻¹·K_S3·(dP_3/dt) (full compensation of the cross-heating); its net disturbance ramp equals that rate,
# so the type-1 tracking error is e_i = −(dP_i/dt)·η / K_v   (η = 440 pm/mW)
rate_S = np.linalg.solve(K[np.ix_(S_idx, S_idx)], K[S_idx, ACQ]) * 4000.0          # mW/s, per locked ring
pred_sweep = {f"ring{i + 1}": float(-r_ * pl.heater_pm_per_mw / des["Kv"]) for i, r_ in zip(S_idx, rate_S)}
res["scenario_c_four_rings"] = {"crosstalk_matrix": K.tolist(), "acquiring_ring": ACQ + 1, "sweep_rate_mw_per_s": 4000.0, "capture_delta_pm": D_CAPTURE,
                                "initial_heaters_mw": p_init.tolist(), "final_heaters_mw_sim": rc.p[-1].tolist(), "final_heaters_mw_expected_Kinv": P_final_expected.tolist(),
                                "capture_time_ms": t_cap * 1e3, "ring3_locked_within_5pm_after_ms": t_lock3 * 1e3,
                                "neighbour_peak_error_pm": nb_peak, "neighbour_peak_time_ms": nb_peak_time_ms, "neighbour_mean_error_during_sweep_pm": nb_sweep_mean, "crosstalk_heating_K_from_ring3": xt_K,
                                "crosstalk_ramp_error_predicted_pm_nearest_first_order": -0.10 * sweep_ramp_pm_s / des["Kv"], "crosstalk_ramp_error_predicted_pm_next_first_order": -0.03 * sweep_ramp_pm_s / des["Kv"],
                                "neighbour_error_during_sweep_predicted_pm_Kinv": pred_sweep,
                                "errors": stats_c, "isolated_heater_mw": P0}
say(f"  initial heaters (mW): {np.round(p_init, 3).tolist()};  final sim {np.round(rc.p[-1], 3).tolist()} vs K⁻¹ solution {np.round(P_final_expected, 3).tolist()}")
say(f"  ring 3 captured at {t_cap * 1e3:.2f} ms (sweep 4 mW/ms = {sweep_ramp_pm_s / 1e6:.2f} nm/ms), locked within 5 pm after {t_lock3 * 1e3:.2f} ms")
say(f"  crosstalk heating from ring 3's {P_final_expected[ACQ]:.2f} mW: " + ", ".join(f"{k} {v:.2f} K = {v * pl.dlam_dT:.0f} pm" for k, v in xt_K.items()))
say(f"  neighbour mean error during the sweep (pm): {', '.join(f'{k} {v:+.2f}' for k, v in nb_sweep_mean.items())};  first-order K_v prediction: nearest −{0.10 * sweep_ramp_pm_s / des['Kv']:.2f} pm, next −{0.03 * sweep_ramp_pm_s / des['Kv']:.2f} pm")
say(f"  with the neighbours' own back-off included (K_SS⁻¹·K_S3·rate): {', '.join(f'{k} {v:+.2f}' for k, v in pred_sweep.items())}  (ring 1 is cooled by ring 2 backing off)")
say(f"  neighbour peak errors (pm): {', '.join(f'{k} {v:+.2f} at {nb_peak_time_ms[k]:.2f} ms' for k, v in nb_peak.items())}  (ring 3's post-capture heater transient, not the sweep)")
fig, axs = plt.subplots(2, 2, figsize=(12, 7))
tm = rc.t * 1e3
ax = axs[0, 0]
for i in range(N): ax.plot(tm, rc.delta[:, i], color=SERIES[i], label=f"ring {i + 1}")
ax.axhline(D_OPT, color=PALETTE["muted"], ls=":"); ax.set_yscale("symlog", linthresh=200); ax.set_ylabel("δ (pm, symlog)"); ax.set_title(f"Ring 3 sweeps in from {rc.delta[0, ACQ] / 1e3:.1f} nm away (heater 0 → {P_final_expected[ACQ]:.1f} mW)"); ax.legend(fontsize=8)
ax = axs[0, 1]
for i in range(N):
    if i != ACQ: ax.plot(tm, rc.delta[:, i] - D_OPT, color=SERIES[i], label=f"ring {i + 1}")
ax.axvline(t_cap * 1e3, color=c_aqua, ls="--", lw=1, label="ring 3 capture"); ax.set_ylabel("δ − δ_opt (pm)"); ax.set_title("Neighbours: a few pm of error while ring 3 heats them"); ax.legend(fontsize=8)
ax = axs[1, 0]
for i in range(N): ax.plot(tm, rc.p[:, i], color=SERIES[i], label=f"ring {i + 1}")
for i in range(N): ax.axhline(P_final_expected[i], color=SERIES[i], ls=":", lw=0.8)
ax.set_xlabel("time (ms)"); ax.set_ylabel("heater (mW)"); ax.set_title("Heaters: neighbours back off by K_ij·P_3 (dotted: K⁻¹ solution)"); ax.legend(fontsize=8)
ax = axs[1, 1]
for i in range(N): ax.plot(tm, rc.dT[:, i] - rc.dT[0, i], color=SERIES[i], label=f"ring {i + 1}")
ax.set_xlabel("time (ms)"); ax.set_ylabel("ΔT since t = 0 (K)"); ax.set_title(f"Temperatures: ring 3 rises {rc.dT[-1, ACQ] - rc.dT[0, ACQ]:.0f} K; the others are held"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "scenario_c_four_rings.png"); plt.close(fig)

# (d) heater saturation and anti-windup ----------------------------------------------------------
say("\n[3d] Heater ceiling: ambient 10 C needs 13.6 mW; with a 12 mW heater the lock saturates, then ambient rises to 30 C")
P_MAX_D = 12.0
amb_d = lambda t: np.array([(10.0 - T_AMB0) + (20.0 if t >= 2e-3 else 0.0)])
rd_aw = rl.simulate(pl, se, [make_ctrl(p_max=P_MAX_D, anti_windup=True, p_init=P_MAX_D)], 8e-3, ambient_fn=amb_d, p_nominal_mw=P0)
rd_no = rl.simulate(pl, se, [make_ctrl(p_max=P_MAX_D, anti_windup=False, p_init=P_MAX_D)], 8e-3, ambient_fn=amb_d, p_nominal_mw=P0)
e_aw, e_no = rd_aw.delta[:, 0] - D_OPT, rd_no.delta[:, 0] - D_OPT
t_rec_aw = settle(rd_aw.t - 2e-3, np.where(rd_aw.t >= 2e-3, e_aw, 0), 5.0); t_rec_no = settle(rd_no.t - 2e-3, np.where(rd_no.t >= 2e-3, e_no, 0), 5.0)
lost_no = abs(e_no[-200:].mean()) > 5.0
res["scenario_d_antiwindup"] = {"p_max_mw": P_MAX_D, "heater_needed_at_10C_mw": rl.nominal_heater_mw(pl, 10.0), "heater_needed_at_30C_mw": rl.nominal_heater_mw(pl, 30.0),
                                "saturated_error_pm": float(e_aw[(rd_aw.t > 1.5e-3) & (rd_aw.t < 2e-3)].mean()),
                                "saturated_error_expected_pm": (rl.nominal_heater_mw(pl, 10.0) - P_MAX_D) * pl.heater_pm_per_mw,
                                "recovery_to_5pm_us_antiwindup": t_rec_aw * 1e6, "recovery_to_5pm_us_no_antiwindup": (None if lost_no else t_rec_no * 1e6), "lock_lost_without_antiwindup": bool(lost_no),
                                "final_delta_pm_no_antiwindup": float(rd_no.delta[-1, 0]), "final_heater_mw_no_antiwindup": float(rd_no.p[-1, 0]), "integrator_estimate_mw_no_antiwindup_at_2ms": float(P_MAX_D + des["ki"] * 2e-3 * (se.I(D_OPT + (rl.nominal_heater_mw(pl, 10.0) - P_MAX_D) * pl.heater_pm_per_mw) - I_SET)),
                                "final_heater_mw": float(rd_aw.p[-1, 0])}
say(f"  saturated error {res['scenario_d_antiwindup']['saturated_error_pm']:.1f} pm (expected (13.57 − 12) mW × 440 = {res['scenario_d_antiwindup']['saturated_error_expected_pm']:.1f} pm)")
say(f"  recovery to ±5 pm after the ambient rise: anti-windup {t_rec_aw * 1e6:.0f} µs; plain integrator: " + ("LOCK LOST (heater pinned at %.1f mW, δ = %.0f pm: wrong side of the notch)" % (rd_no.p[-1, 0], rd_no.delta[-1, 0]) if lost_no else f"{t_rec_no * 1e6:.0f} µs") + f"; final heater {rd_aw.p[-1, 0]:.2f} mW (needs {rl.nominal_heater_mw(pl, 30.0):.2f})")
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
ax = axs[0]; ax.plot(rd_aw.t * 1e3, e_aw, color=c_blue, label="with anti-windup (integrator clamped at the rail)"); ax.plot(rd_no.t * 1e3, e_no, color=c_orange, label="without: integrator winds to ~70 mW, notch flies past the laser")
ax.axhline(-D_OPT, color=c_red, lw=0.8, ls=":"); ax.text(4, -D_OPT - 40, "δ = 0: notch bottom on the laser; below = wrong side", fontsize=7.5, color=c_red)
ax.axvline(2, color=PALETTE["muted"], ls=":"); ax.set_xlabel("time (ms)"); ax.set_ylabel("δ − δ_opt (pm)"); ax.set_title("Heater pinned at 12 mW until ambient rises 20 K at t = 2 ms"); ax.legend(fontsize=8)
ax = axs[1]; ax.plot(rd_aw.t * 1e3, rd_aw.p[:, 0], color=c_blue, label="heater, anti-windup"); ax.plot(rd_no.t * 1e3, rd_no.p[:, 0], color=c_orange, label="heater, no anti-windup")
ax.axhline(rl.nominal_heater_mw(pl, 30.0), color=PALETTE["muted"], ls=":", label="needed at 30 C"); ax.set_xlabel("time (ms)"); ax.set_ylabel("heater (mW)"); ax.set_title("Same heater output while saturated; different recovery"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "scenario_d_antiwindup.png"); plt.close(fig)

# (e) heater DAC quantisation and photocurrent noise ----------------------------------------------
say("\n[3e] Steady-state jitter from heater-DAC resolution and photocurrent noise")
quant = {}
for bits in [8, 10, 12, 14, None]:
    rq = rl.simulate(pl, se, [make_ctrl(dac_bits=bits)], 3e-3, p_nominal_mw=P0)
    e = rq.delta[int(1e-3 / 1e-6):, 0] - D_OPT
    lsb_pm = (P_MAX / (2 ** bits - 1)) * pl.heater_pm_per_mw if bits else 0.0
    quant[str(bits)] = {"lsb_pm": lsb_pm, "rms_error_pm": float(e.std()), "mean_error_pm": float(e.mean()), "pk_pk_pm": float(e.max() - e.min())}
    say(f"  DAC {bits} bit: LSB = {lsb_pm:.2f} pm, error mean {e.mean():+.3f} pm, rms {e.std():.3f} pm, pk-pk {e.max() - e.min():.2f} pm")
rn = rl.simulate(pl, se, [make_ctrl()], 3e-3, p_nominal_mw=P0, i_noise_ua=0.05, seed=1)
en = rn.delta[int(1e-3 / 1e-6):, 0] - D_OPT
res["quantisation"] = quant
res["noise_50nA"] = {"i_noise_ua_rms_per_sample": 0.05, "rms_error_pm": float(en.std()), "equivalent_input_pm_rms": 0.05 / float(se.dI(D_OPT))}
say(f"  50 nA rms TIA noise per 10 µs sample (70× shot noise): δ rms {en.std():.3f} pm (input-referred {0.05 / se.dI(D_OPT):.3f} pm)")
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
ax = axs[0]
for bits, col in zip([8, 10, 12], [c_red, c_orange, c_blue]):
    rq = rl.simulate(pl, se, [make_ctrl(dac_bits=bits)], 2e-3, p_nominal_mw=P0)
    ax.plot(rq.t * 1e3, rq.delta[:, 0] - D_OPT, color=col, lw=1.2, label=f"{bits}-bit DAC, LSB = {quant[str(bits)]['lsb_pm']:.1f} pm")
ax.set_xlabel("time (ms)"); ax.set_ylabel("δ − δ_opt (pm)"); ax.set_title("Heater DAC resolution: 8 bits over 20 mW = 34 pm per step"); ax.legend(fontsize=8)
ax = axs[1]; ax.plot(rn.t * 1e3, rn.delta[:, 0] - D_OPT, color=c_blue, lw=1); ax.set_xlabel("time (ms)"); ax.set_ylabel("δ − δ_opt (pm)")
ax.set_title(f"50 nA rms photocurrent noise → {en.std():.2f} pm rms (shot noise: {res['shot_noise_limited_delta_pm_rms']:.3f} pm)", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "quantisation_noise.png"); plt.close(fig)

# =============================================================================================
# 4. Videos
# =============================================================================================
say("\n[4] Videos")
def contact_sheet(make_frame_fig, frame_ids, path, ncols=3):
    n = len(frame_ids); nrows = int(np.ceil(n / ncols))
    import io
    from PIL import Image
    imgs = []
    for k in frame_ids:
        figk = make_frame_fig(k); buf = io.BytesIO(); figk.savefig(buf, format="png", dpi=80); plt.close(figk); buf.seek(0); imgs.append(Image.open(buf).convert("RGB"))
    W, H = imgs[0].size
    sheet = Image.new("RGB", (ncols * W, nrows * H), "white")
    for i, im in enumerate(imgs): sheet.paste(im, ((i % ncols) * W, (i // ncols) * H))
    sheet.save(path)

# video 1: the notch sliding under the laser, lock on vs off (scenario a)
V1_T0, V1_T1, V1_N = -50e-6, 550e-6, 300        # 10 s at 30 fps, 60 µs of sim per second
v1_times = np.linspace(V1_T0, V1_T1, V1_N)
lam_axis = np.linspace(-450, 450, 901)          # λ − λ_L (pm)
def v1_state(tsim):
    i = min(max(int(np.searchsorted(ra.t - T_STEP, tsim)), 0), len(ra.t) - 1)
    return ra.delta[i, 0], ra_off.delta[i, 0], ra.p[i, 0], ra.i[i, 0], ra_off.i[i, 0]
def v1_fig(k, fig=None):
    tsim = v1_times[k]; d_on, d_off, p_on, i_on, i_off = v1_state(tsim)
    fig = fig or plt.figure(figsize=(11, 6.2))
    fig.clf()
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(lam_axis, se.T(lam_axis + d_off), color=PALETTE["muted"], ls="--", lw=1.6, label="lock OFF: notch drifts with temperature")
    ax.plot(lam_axis, se.T(lam_axis + d_on), color=c_blue, lw=2.2, label="lock ON: heater pulls the notch back")
    ax.axvline(0, color=c_red, lw=1.5, label="laser λ_L (fixed)")
    ax.plot([0], [se.T(d_on)], "o", color=c_blue, ms=9); ax.plot([0], [se.T(d_off)], "o", color=PALETTE["muted"], ms=7)
    ax.plot([-se.mod_swing_pm / 2, se.mod_swing_pm / 2], [se.T(d_on + se.mod_swing_pm / 2), se.T(d_on - se.mod_swing_pm / 2)], "s", color=c_aqua, ms=6, label="data levels (±32.5 pm)")
    ax.axvline(-D_OPT, color=c_orange, ls=":", lw=1, label="target notch centre: λ_L − 108 pm")
    ax.set_xlim(-450, 450); ax.set_ylim(0, 1.05); ax.set_xlabel("λ − λ_L (pm)"); ax.set_ylabel("through transmission")
    ax.set_title(f"t = {tsim * 1e6:+6.0f} µs after a 5 K ambient step   |   δ_on = {d_on:6.1f} pm   δ_off = {d_off:6.1f} pm   heater = {p_on:5.2f} mW   I = {i_on:5.1f} µA")
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    ax2 = fig.add_subplot(gs[1, 0]); m = (ra.t - T_STEP) <= tsim
    ax2.plot((ra.t[m] - T_STEP) * 1e6, ra.delta[m, 0], color=c_blue); ax2.plot((ra_off.t[m] - T_STEP) * 1e6, ra_off.delta[m, 0], color=PALETTE["muted"], ls="--")
    ax2.axhline(D_OPT, color=c_orange, ls=":", lw=1); ax2.axhline(0, color=c_red, lw=0.8)
    ax2.set_xlim(V1_T0 * 1e6, V1_T1 * 1e6); ax2.set_ylim(-160, 130); ax2.set_xlabel("time (µs)"); ax2.set_ylabel("δ (pm)"); ax2.set_title("Detuning: the lock pulls δ back to +108 pm", fontsize=9)
    ax3 = fig.add_subplot(gs[1, 1]); ax3.plot((ra.t[m] - T_STEP) * 1e6, ra.p[m, 0], color=c_orange)
    ax3.set_xlim(V1_T0 * 1e6, V1_T1 * 1e6); ax3.set_ylim(7.2, 8.0); ax3.set_xlabel("time (µs)"); ax3.set_ylabel("heater (mW)"); ax3.set_title("Heater backs off by 5 K / R_th = 0.57 mW", fontsize=9)
    fig.tight_layout()
    return fig
figv = plt.figure(figsize=(11, 6.2))
anim = FuncAnimation(figv, lambda k: v1_fig(k, figv), frames=V1_N, blit=False)
anim.save(OUT / "notch_lock.mp4", writer=FFMpegWriter(fps=30, bitrate=2500)); plt.close(figv)
contact_sheet(lambda k: v1_fig(k), [int(x) for x in np.linspace(25, V1_N - 1, 6)], OUT / "notch_lock_frames.png")
say(f"  out/notch_lock.mp4: {V1_N} frames, {V1_N / 30:.0f} s, sim −50..550 µs around the 5 K step")

# video 2: four rings, ring 3 acquiring
V2_T1, V2_N = 4.0e-3, 360                       # 12 s
v2_times = np.linspace(0, V2_T1, V2_N)
def v2_fig(k, fig=None):
    tsim = v2_times[k]; i = min(int(np.searchsorted(rc.t, tsim)), len(rc.t) - 1)
    fig = fig or plt.figure(figsize=(12, 6.2)); fig.clf()
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1])
    for j in range(N):
        ax = fig.add_subplot(gs[0, j]); d = rc.delta[i, j]
        ax.plot(lam_axis, se.T(lam_axis + d), color=SERIES[j], lw=2)
        ax.axvline(0, color=c_red, lw=1.2); ax.axvline(-D_OPT, color=PALETTE["muted"], ls=":", lw=1)
        ax.plot([0], [se.T(d)], "o", color=SERIES[j], ms=8)
        if d > 450: ax.annotate("", xy=(-430, 0.5), xytext=(-250, 0.5), arrowprops=dict(arrowstyle="<-", color=SERIES[j], lw=2)); ax.text(-240, 0.47, f"notch {d / 1e3:.2f} nm\nto the blue", fontsize=8, color=SERIES[j])
        ax.set_xlim(-450, 450); ax.set_ylim(0, 1.05); ax.set_title(f"ring {j + 1}: δ = {d:7.1f} pm, P = {rc.p[i, j]:.2f} mW", fontsize=9)
        ax.set_xlabel("λ − λ_L,i (pm)")
        if j == 0: ax.set_ylabel("through transmission")
    m = rc.t <= tsim
    ax = fig.add_subplot(gs[1, :2])
    for j in range(N):
        if j != ACQ: ax.plot(rc.t[m] * 1e3, rc.delta[m, j] - D_OPT, color=SERIES[j], label=f"ring {j + 1}")
    ax.set_xlim(0, V2_T1 * 1e3); ax.set_ylim(-12, 12); ax.set_xlabel("time (ms)"); ax.set_ylabel("δ − δ_opt (pm)"); ax.set_title("locked neighbours: error from ring 3's heat", fontsize=9); ax.legend(fontsize=7, loc="upper left")
    ax = fig.add_subplot(gs[1, 2:])
    for j in range(N): ax.plot(rc.t[m] * 1e3, rc.p[m, j], color=SERIES[j], label=f"ring {j + 1}")
    ax.set_xlim(0, V2_T1 * 1e3); ax.set_ylim(0, 9); ax.set_xlabel("time (ms)"); ax.set_ylabel("heater (mW)"); ax.set_title(f"t = {tsim * 1e3:.2f} ms: ring 3 sweeps up (4 mW/ms), neighbours back off", fontsize=9); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); return fig
figv = plt.figure(figsize=(12, 6.2))
anim = FuncAnimation(figv, lambda k: v2_fig(k, figv), frames=V2_N, blit=False)
anim.save(OUT / "four_ring_lock.mp4", writer=FFMpegWriter(fps=30, bitrate=2500)); plt.close(figv)
contact_sheet(lambda k: v2_fig(k), [int(x) for x in np.linspace(0, V2_N - 1, 6)], OUT / "four_ring_lock_frames.png", ncols=2)
say(f"  out/four_ring_lock.mp4: {V2_N} frames, {V2_N / 30:.0f} s, sim 0..4 ms")

# =============================================================================================
# 5. Headline table, JSON, tools, notebook
# =============================================================================================
def agree(v, e, scale=None):
    """Percent agreement 100·(1 − |v − e|/|e|); when the expectation is zero (or a scale is given) the
    difference is measured against `scale` instead, so that a zero expectation never yields NaN."""
    v, e = float(v), float(e)
    if scale is None and e == 0:
        raise ValueError("expected == 0 needs an explicit scale")
    ref = abs(e) if scale is None else float(scale)
    return float(round(100 * (1 - abs(v - e) / ref), 2))
headline = {}
def H(key, value, expected, source, unit="", scale=None):
    headline[key] = {"value": float(value), "expected": float(expected), "agreement_percent": agree(value, expected, scale), "source": source, "unit": unit}
    if scale is not None:
        headline[key]["agreement_scale"] = float(scale)
H("i_fs_ua", se.i_fs_ua, 0.9 * 2.5 * 0.05 * 1e3, "brief: 0.9 A/W × 2.5 mW × 5 % = 112.5 µA (code uses 4 dBm = 2.512 mW)", "µA")
H("delta_opt_numeric_pm", d_opt_num, REF.delta_opt_pm, "REF.delta_opt_pm = FWHM/(2√3)", "pm")
H("T_at_delta_opt", se.T(D_OPT), 0.25 + 0.75 * se.t_min, "Lorentzian at u = 1/√3: 1 − 0.75(1−T_min)", "")
H("sensor_gain_ua_per_pm", se.dI(D_OPT), kI_fd, "central finite difference of I(δ)", "µA/pm")
H("sensor_gain_ua_per_K", se.dI(D_OPT) * pl.dlam_dT, se.i_fs_ua * se.dT_max_analytic() * pl.dlam_dT, "(1−T_min)·3√3/(4·FWHM) · I_fs · 50 pm/K", "µA/K")
H("i_mean_with_modulation_ua", se.I_modulated_mean(D_OPT), I_SET, "CW value (inflection point ⇒ mean unchanged)", "µA")
H("oma_finite_swing_optimum_bias_pm", b_oma_opt, D_OPT, "small-signal δ_opt", "pm")
H("heater_pm_per_mw", pl.heater_pm_per_mw, REF.heater_nm_per_mw * 1e3, "REF.heater_nm_per_mw = 0.44 nm/mW", "pm/mW")
H("one_fsr_K", one_fsr_K, 206.0, "brief: one FSR = 206 K", "K")
H("crossover_hz_pade", wcp / 2 / np.pi, des["wc"] / 2 / np.pi, "design target ω_c = 1/(2T_d)", "Hz")
H("phase_margin_deg_pade", pm, pm_hand, "hand estimate: 90 + atan(ω_c·227.5µs) − atan(ω_c·300µs) − ω_c·T_d", "deg")
H("phase_margin_deg_exact_delay", ex["pm_deg"], pm, "python-control margin() with Pade(3)", "deg")
H("gain_margin_db_exact_delay", ex["gm_db"], 20 * np.log10(gm), "python-control margin() with Pade(3)", "dB")
H("ramp_tracking_error_pm", -eb[mid].mean(), pred, "ramp rate (pm/s) / K_v", "pm")
H("ramp_tracking_error_pm_tuningB", -ebB[mid].mean(), predB, "ramp rate / K_v (tuning B)", "pm")
H("step5K_peak_error_pm", sa["peak_pm"], lin.outputs.min(), "linear model (python-control forced_response); the sampled loop is slightly faster than its continuous e^(−sT_d) model (see Checks), the sensor nonlinearity is worth only ~2 pm", "pm")
H("step5K_heater_change_mw", ra.p[-1, 0] - P0, -STEP_K / pl.r_th, "−ΔT_amb / R_th", "mW")
H("step5K_steady_state_error_pm", sa["final_pm"], 0.0, "type-1 loop: zero steady-state error to a step; agreement measured against the 5 pm (0.1 K) resolution scale", "pm", scale=0.1 * pl.dlam_dT)
H("four_rings_final_heater_ring2_mw", rc.p[-1, 1], P_final_expected[1], "K⁻¹ · P0·1 (crosstalk equilibrium)", "mW")
H("four_rings_final_heater_ring3_mw", rc.p[-1, ACQ], P_final_expected[ACQ], "K⁻¹ · P0·1", "mW")
H("four_rings_ring2_mean_error_during_sweep_pm", nb_sweep_mean["ring2"], pred_sweep["ring2"], "quasi-static: −(K_SS⁻¹·K_S3·sweep rate)·η / K_v (first order: 10 % of the sweep ramp / K_v = −2.67 pm)", "pm")
H("four_rings_ring1_mean_error_during_sweep_pm", nb_sweep_mean["ring1"], pred_sweep["ring1"], "quasi-static: −(K_SS⁻¹·K_S3·sweep rate)·η / K_v (first order 3 % would give −0.80 pm; ring 2's back-off cools ring 1)", "pm")
H("saturated_error_pm", res["scenario_d_antiwindup"]["saturated_error_pm"], res["scenario_d_antiwindup"]["saturated_error_expected_pm"], "(P_needed − P_max) × 440 pm/mW", "pm")
H("shot_noise_delta_pm_rms", res["shot_noise_limited_delta_pm_rms"], np.sqrt(2 * Q_E * I_SET * 1e-6 * B_noise) * 1e6 / (se.i_fs_ua * se.dT_max_analytic()), "√(2qIB)/(dI/dδ)", "pm")
H("lock_lost_7K_step_final_error_pm", lost_step["final_error_pm"], lost_step["expected_error_pm_heater_at_rail"], f"heater pinned at P_max = {P_MAX:.0f} mW: −[(P_max − P0)·η + {sz_lost:.0f} K × 50 pm/K]", "pm")
H("push_crossover_phase_margin_deg", fast["phase_margin_deg_exact_delay"], fast["phase_margin_deg_hand"], "ω_c·T_d = 1: hand estimate 90 + atan(ω_c·227.5µs) − atan(ω_c·300µs) − ω_c·T_d", "deg")
H("push_crossover_ramp_error_pm", fast["ramp_error_pm_sim"], fast["ramp_error_pm_predicted_Kv"], "ω_c·T_d = 1: ramp rate / K_v", "pm")
H("slow_controller_crossover_hz", slow["crossover_hz_exact_delay"], 1 / (4 * np.pi * T_D_SLOW), "T_d = 50 µs: 1/(4πT_d)", "Hz")
H("slow_controller_ramp_error_pm", slow["ramp_error_pm_sim"], slow["ramp_error_pm_predicted_Kv"], "T_d = 50 µs: ramp rate / K_v", "pm")
res["headline"] = headline
res["assumptions"] = {
    "controller": f"discrete PI at T_s = {T_S * 1e6:.0f} µs; measurement = mean photocurrent over the previous T_s (averages the 53 Gbaud data); heater updated at the tick and held; the continuous design uses e^(−s·T_d) with T_d = T_s (T_s/2 averaging + T_s/2 hold)",
    "heater": f"0 ≤ P_h ≤ {P_MAX} mW (ceiling assumed); heater bias set so that 0.5 mW remains at 125 C; scenarios at 60 C ambient (P0 = {P0:.2f} mW)",
    "ambient": "ambient temperature deviations enter the ring through the same two-pole thermal filter as the heater (a ring cannot jump in temperature); an instantaneous ambient step is therefore a pessimistic test",
    "crosstalk": "K_ij scales heater power before the shared thermal dynamics (cross-heating assumed as fast as self-heating; in reality it is slower and more diffusive)",
    "optics": "Lorentzian through-port notch with T_min = 0.016, FWHM = 374 pm (Q = 3500); laser wavelength and power fixed; self-heating by absorbed light ignored (it is an offset the integrator absorbs); no RIN; data modulation 50 pm/V × 1.3 Vpp = 65 pm around the bias",
    "sensor": "0.9 A/W, 5 % tap of the 4 dBm (2.512 mW) input; I_set calibrated to I(δ_opt) = 0.262·I_fs; a single photocurrent cannot tell the two sides of the notch apart (capture only from the blue side, i.e. δ > 0)",
    "thermal": "R_th = 8.8 K/mW, 75 % with τ = 10 µs and 25 % with τ = 300 µs, dλ_r/dT = 50 pm/K (η = 440 pm/mW = REF 0.44 nm/mW); temperature independent (no thermal runaway, no R_th(T))",
}
res["runtime_seconds_before_notebook"] = time.time() - T0
try:
    ffmpeg_ver = subprocess.run(["/opt/homebrew/bin/ffmpeg", "-version"], capture_output=True, text=True, timeout=20).stdout.split("\n")[0].split(" ")[2]
except Exception:
    ffmpeg_ver = "unknown"
import ipywidgets, nbformat
res["versions"] = {"python": platform.python_version(), "python-control": ct.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                   "matplotlib": matplotlib.__version__, "ffmpeg": ffmpeg_ver, "ipywidgets": ipywidgets.__version__, "nbformat": nbformat.__version__}

tools = [
    {"tool": "python-control", "version": ct.__version__,
     "what_it_is": "The Python control-systems library (a MATLAB Control Toolbox work-alike): transfer functions and state space, series/feedback interconnection, Bode and Nyquist plots, gain/phase margins, step and forced responses, Pade delay approximations. Normally used to design and analyse feedback loops before they are coded.",
     "used_for": "Building the linearised ring-lock loop L(s) = C(s)·e^(−sT_d)·G_th(s)·(50 pm/K)·(dI/dδ) with the two-pole thermal plant and a Pade(3) delay, computing the PI gains by pole cancellation and |L(jω_c)| = 1 at ω_c = 1/(2T_d), the margins with margin(), the sensitivity S = 1/(1+L), the ambient-to-error transfer −50·P_n(s)·S(s), and linear step/forced responses that are overlaid on the nonlinear simulation.",
     "result": f"Crossover {wcp / 2 / np.pi:.0f} Hz (target {des['wc'] / 2 / np.pi:.0f}, {headline['crossover_hz_pade']['agreement_percent']:.2f} %), phase margin {pm:.1f}° (hand estimate {pm_hand:.1f}°, exact-delay check {ex['pm_deg']:.1f}°), gain margin {20 * np.log10(gm):.1f} dB, M_s = {Ms:.2f}, K_v = {des['Kv']:.0f} 1/s which predicts the 800 K/s ramp error {pred:.3f} pm (simulated {-eb[mid].mean():.3f} pm, {headline['ramp_tracking_error_pm']['agreement_percent']:.2f} %). Linear 5 K step peak {lin.outputs.min():.0f} pm vs nonlinear {sa['peak_pm']:.0f} pm.",
     "how_to_observe": "cd experiments/13_capstone_ring_lock && ../../.venv/bin/python run.py ; look at out/bode_loop.png (loop gain, margins, tuning A vs B), out/sensitivity.png, out/linear_step.png, and the 'design' block of out/results.json. Change T_D, the crossover (design_pi(..., wc=...)) or tau_i in run.py / ringlock.design_pi and re-run; the notebook has sliders for the same."},
    {"tool": "numpy + scipy", "version": f"numpy {np.__version__}, scipy {scipy.__version__}",
     "what_it_is": "numpy is the array library underneath all scientific Python; scipy adds numerical algorithms. Used here for the plant integration and all the static numbers.",
     "used_for": "ringlock.simulate: an exact (matrix-exponential per state) discretisation of the two thermal states per ring at 1 µs, the Lorentzian sensor, the discrete PI with clamping anti-windup, acquisition sweep and optional DAC quantisation, and the 4-ring crosstalk matrix K; np.linalg.solve for the K⁻¹ heater equilibrium; the survival map (77 simulations); the OMA/penalty numbers; shot-noise and resolution arithmetic.",
     "result": f"δ_opt (numeric argmax of dT/dδ) = {d_opt_num:.1f} pm vs 108; sensor gain {se.dI(D_OPT):.3f} µA/pm = {se.dI(D_OPT) * pl.dlam_dT:.1f} µA/K; 5 K step: peak error {sa['peak_pm']:.0f} pm ({100 * abs(sa['peak_frac_fwhm']):.0f} % FWHM), back within 5 pm in {t_settle * 1e6:.0f} µs, steady-state {sa['final_pm']:.1e} pm; ramp error {-eb[mid].mean():.3f} pm; 4 rings: ring 2's mean error {nb_sweep_mean['ring2']:+.2f} pm while ring 3 sweeps in (quasi-static prediction {pred_sweep['ring2']:+.2f} pm, {headline['four_rings_ring2_mean_error_during_sweep_pm']['agreement_percent']:.1f} %; peak {nb_peak['ring2']:+.1f} pm at capture), final heaters match K⁻¹·P0·1 to {headline['four_rings_final_heater_ring2_mw']['agreement_percent']:.2f} %; the lock survives instantaneous steps up to {largest_ok['0us']:.0f} K and steps up to {largest_ok['100us']:.0f} K (the largest tested) with a ≥ 100 µs rise; heater saturated {rl.nominal_heater_mw(pl, 10.0) - P_MAX_D:.2f} mW short gives {res['scenario_d_antiwindup']['saturated_error_pm']:.0f} pm of error (expected {res['scenario_d_antiwindup']['saturated_error_expected_pm']:.1f}).",
     "how_to_observe": "out/scenario_a_step.png, out/step_survival.png, out/scenario_b_ramp.png, out/scenario_c_four_rings.png, out/scenario_d_antiwindup.png, out/quantisation_noise.png and the scenario_* blocks of out/results.json. Edit STEP_K, RAMP, the crosstalk numbers (rl.crosstalk_matrix), sweep_rate, dac_bits or P_MAX in run.py and re-run."},
    {"tool": "matplotlib (+ ffmpeg)", "version": f"matplotlib {matplotlib.__version__}, ffmpeg {ffmpeg_ver}",
     "what_it_is": "The standard Python plotting library; FuncAnimation + FFMpegWriter feed frames to the ffmpeg encoder to produce mp4 videos.",
     "used_for": "13 figures plus 2 video contact sheets (15 PNGs: block diagram, sensor curve, OMA vs bias, thermal step, Bode, sensitivity, linear steps, five scenario figures, quantisation/noise) and two videos: notch_lock.mp4 (the notch sliding under the fixed laser after a 5 K step, lock on vs off) and four_ring_lock.mp4 (ring 3 sweeping in while its neighbours hold), each with a contact sheet.",
     "result": f"out/notch_lock.mp4 ({V1_N} frames, {V1_N / 30:.0f} s; 1 s of video = 60 µs) and out/four_ring_lock.mp4 ({V2_N} frames, {V2_N / 30:.0f} s; 1 s = 0.33 ms) plus out/notch_lock_frames.png and out/four_ring_lock_frames.png.",
     "how_to_observe": "open out/notch_lock.mp4: the grey notch drifts 250 pm to the red and stays; the blue one dips past the laser (δ crosses 0 for ~30 µs) and is pulled back to −108 pm by the heater trace below. Change V1_T1 / V1_N or the frame lambda in run.py to re-time the video."},
    {"tool": "Jupyter notebook + ipywidgets", "version": f"nbformat {nbformat.__version__}, ipywidgets {ipywidgets.__version__}, kernel photonics-sims",
     "what_it_is": "Jupyter notebooks mix code, output and prose; ipywidgets adds sliders that re-run a function when moved. Used for interactive exploration where a script would need a re-run per parameter.",
     "used_for": "explore.ipynb (built by make_notebook.py with nbformat, executed headless with nbconvert): sliders for the sensor (FWHM, T_min, P_in, tap), the loop design (T_d, crossover fraction ω_c·T_d, PI zero placement, slow share) with live Bode + margins + step, the ambient step scenario (size, rise time, lock on/off, T_d), the 4-ring crosstalk scenario (K_nearest, K_next, sweep rate) and the DAC/noise study; it re-imports ringlock.py so the numbers are the same as run.py's and asserts K_p against out/results.json.",
     "result": f"Each section first calls its plot function once (a static figure for the default slider values, saved as an ordinary cell output so it is visible without a live kernel) and then builds the live widget; the printed numbers are identical to run.py's (the K_p assertion guarantees it). Raising ω_c·T_d from 0.5 to 1.0 drops the phase margin from {ex['pm_deg']:.0f}° to {fast['phase_margin_deg_exact_delay']:.0f}° and raises M_s from {Ms:.2f} to {fast['sensitivity_peak_Ms']:.2f} (the step response rings); moving the PI zero to the slow pole cuts K_v {des['Kv'] / desB['Kv']:.0f}× (ramp error {-ebB[mid].mean():.1f} pm instead of {-eb[mid].mean():.1f} pm); a 5 K step with a 200 µs rise peaks at about {abs(peak_5K_200us):.0f} pm instead of {abs(sa['peak_pm']):.0f} pm; an 8-bit heater DAC dithers ±{quant['8']['pk_pk_pm'] / 2:.0f} pm.",
     "how_to_observe": "cd experiments/13_capstone_ring_lock && ../../.venv/bin/jupyter lab explore.ipynb and drag the sliders; to rebuild headless: ../../.venv/bin/python make_notebook.py && ../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace explore.ipynb"},
]
(OUT / "tools.json").write_text(json.dumps(tools, indent=2, ensure_ascii=False))

say("\nHEADLINE NUMBERS")
for k_, hv in headline.items():
    say(f"  {k_:40s} = {hv['value']:12.4f} {hv['unit']:6s} expected {hv['expected']:12.4f}   agreement {hv['agreement_percent']:7.2f} %")

# notebook: build with nbformat, execute with nbconvert (kernel photonics-sims)
say("\n[5] Notebook")
(OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))     # written first: the notebook reads it
try:
    PY = str(ROOT / ".venv" / "bin" / "python"); JUP = str(ROOT / ".venv" / "bin" / "jupyter")
    subprocess.run([PY, "make_notebook.py"], cwd=HERE, check=True, timeout=120)
    t_nb = time.time()
    subprocess.run([JUP, "nbconvert", "--to", "notebook", "--execute", "--inplace", "--ExecutePreprocessor.timeout=600", "explore.ipynb"], cwd=HERE, check=True, timeout=900, capture_output=True, text=True)
    res["notebook"] = {"executed": True, "seconds": time.time() - t_nb}
    say(f"  explore.ipynb built and executed in {time.time() - t_nb:.0f} s")
except Exception as exc:
    res["notebook"] = {"executed": False, "error": str(exc)[:500]}
    say(f"  notebook step failed: {exc}")
res["runtime_seconds"] = time.time() - T0
(OUT / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
say(f"\nrun.py finished in {res['runtime_seconds']:.1f} s")
(OUT / "results.txt").write_text("\n".join(LOG) + "\n")
